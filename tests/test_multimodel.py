"""One hotkey, several inference models: scoring slots in the validator, the state migration and the router
(docs/DESIGN-MULTIROLE.md sections 4 and 6).

Weight tests run on synthetic state (compute_weights); round tests use real aiohttp miners on 127.0.0.1 and a faked
chain, as in test_tiers.py. A "multi-model miner" is one aiohttp endpoint with one hotkey whose requests are routed
to one S1Miner per model.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import time
from dataclasses import replace
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import elements as E
from reax_subnet import s1validator, tiers
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.config import load_validator
from reax_subnet.elements import SLOT_STRIDE, slot_key, slot_uid
from reax_subnet.manifest import commitment_text, load_manifest
from reax_subnet.miner import S1Miner
from reax_subnet.router import Router, RouterConfig
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1protocol import parse_s1_request
from reax_subnet.s1scoring import PoolParams
from reax_subnet.s1validator import S1Validator
from reax_subnet.state import MinerState, ValidatorState, load, save
from reax_subnet.tiers import CapacityParams, TierBudgets, compute_tiered_weights
from test_tiers import Machine, OnMachine

DEV = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1"}
MID = {m: load_manifest(m).manifest_id for m in E.MODEL_ORDER}
SPEC = "reax-spec/1:demand_forecasting:alice/df-model@" + "a" * 40


@pytest.fixture(autouse=True)
def _release_floors_off(monkeypatch):
    from reax_subnet import competition_split as _cs
    for _k in list(_cs.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(_cs.TRACK_ENTRY_FLOOR, _k, None)


@pytest.fixture
def started(monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 0)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 0)


def multi_text(*models: str) -> str:
    return E.inference_text({m: MID[m] for m in models})


def role(model: str) -> str:
    return f"{model}:{MID[model]}"


# ---------------------------------------------------------------------------------------------- synthetic weights
def _validator(tmp_path, wallet, fake_chain_cls, **env) -> S1Validator:
    cfg = load_validator({**DEV, "REAX_FEATURE_TIERED_POOLS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json"),
                          "REAX_BURN_FRACTION": "0.0", "REAX_WARMUP_ROUNDS": "0", **env})
    return S1Validator(cfg, wallet, fake_chain_cls(neurons=[]), {m: S1HashBackend(m) for m in s1validator.POOL_PARAMS})


def _neuron(uid, commitment="", fields=()):
    return Neuron(uid, f"hk{uid}", f"ck{uid}", False, 0.0, "127.0.0.1", 9000 + uid, commitment=commitment,
                  commitment_fields=tuple(fields))


def _snap(uids, owner=0):
    return Snapshot(100, tuple(_neuron(u) for u in sorted({owner, 1, *uids})), "hk0")


def _put(val, uid, model, ema, capacity):
    val.state.miners[slot_key(uid, model)] = MinerState(hotkey=f"hk{uid}", ema=ema, role=role(model),
                                                        capacity=capacity)


def _fleet_weights(val, entries):
    """entries: [(uid, model, ema, capacity)] -> compute_weights over a fresh state (one snapshot, owner uid 0)."""
    snap = _snap({e[0] for e in entries})
    val.state.miners.clear()
    val.state.sync(snap)
    for uid, model, ema, capacity in entries:
        _put(val, uid, model, ema, capacity)
    return val.compute_weights(snap)


def _background(n_fast=2, n_pro=1, n_lite=1, n_cpu=1):
    rng = random.Random(7)
    out, uid = [], 40
    for model, n in (("s1-fast", n_fast), ("s1-pro", n_pro), ("s1-lite", n_lite), ("s1-cpu", n_cpu)):
        for _ in range(n):
            target = tiers.CAPACITY[model].target_dps
            out.append((uid, model, rng.uniform(0.6, 0.95), target * rng.uniform(0.5, 3.0)))
            uid += 1
    return out


@pytest.mark.parametrize("models", [("s1-fast", "s1-pro"), ("s1-fast", "s1-pro", "s1-lite"),
                                    ("s1-pro", "s1-lite", "s1-cpu"), ("s1-fast", "s1-pro", "s1-lite", "s1-cpu")])
@pytest.mark.parametrize("background", [dict(), dict(n_fast=6, n_pro=0, n_lite=0, n_cpu=0)])
def test_a_multi_model_hotkey_earns_what_separate_hotkeys_with_the_same_numbers_earn(
        models, background, make_wallet, fake_chain_cls, tmp_path):
    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    per_model = {m: (0.5 + 0.1 * i, tiers.CAPACITY[m].target_dps * (1.0 + i)) for i, m in enumerate(models)}
    together = [(2, m, ema, cap) for m, (ema, cap) in per_model.items()]
    apart = [(30 + i, m, ema, cap) for i, (m, (ema, cap)) in enumerate(per_model.items())]
    bg = _background(**background)
    w_together = _fleet_weights(val, together + bg)
    w_apart = _fleet_weights(val, apart + bg)
    assert w_together[2] == pytest.approx(math.fsum(w_apart[30 + i] for i in range(len(models))), rel=1e-12)
    for uid in (e[0] for e in bg):
        assert w_together.get(uid, 0.0) == pytest.approx(w_apart.get(uid, 0.0), rel=1e-12)
    assert math.fsum(w_together.values()) == pytest.approx(1.0)


def test_the_tier_order_cap_counts_slots_not_hotkeys(make_wallet, fake_chain_cls, tmp_path):
    """6 plain s1-fast miners and one hotkey with s1-fast + s1-pro + s1-lite: the GPU tier has 8 miners (slots), so
    the lone lite slot is held to the average of 8, exactly as for separate hotkeys (counting hotkeys, 7, would pay
    it more)."""
    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    fast, pro, lite = tiers.CAPACITY["s1-fast"], tiers.CAPACITY["s1-pro"], tiers.CAPACITY["s1-lite"]
    bg = [(40 + i, "s1-fast", 0.9, fast.target_dps) for i in range(6)]
    together = [(2, "s1-fast", 0.9, fast.target_dps), (2, "s1-pro", 0.9, pro.target_dps),
                (2, "s1-lite", 0.9, lite.target_dps)]
    apart = [(30, "s1-fast", 0.9, fast.target_dps), (31, "s1-pro", 0.9, pro.target_dps),
             (32, "s1-lite", 0.9, lite.target_dps)]
    w_together, w_apart = _fleet_weights(val, together + bg), _fleet_weights(val, apart + bg)
    assert w_together[2] == pytest.approx(w_apart[30] + w_apart[31] + w_apart[32], rel=1e-12)
    # The tier-order rule binds: the lone lite slot gets less than its 0.15 tier budget (the rest moved up).
    assert w_apart[32] < 0.15 - 1e-3


def test_the_cap_of_each_model_applies_to_its_own_slot(make_wallet, fake_chain_cls, tmp_path):
    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    cap = {m: tiers.CAPACITY[m].cap_dps for m in E.MODEL_ORDER}
    bg = _background()

    def hotkey(fast_dps, pro_dps):
        return _fleet_weights(val, [(2, "s1-fast", 0.9, fast_dps), (2, "s1-pro", 0.9, pro_dps)] + bg)[2]

    at_cap = hotkey(cap["s1-fast"], cap["s1-pro"])
    assert hotkey(10 * cap["s1-fast"], cap["s1-pro"]) == pytest.approx(at_cap, rel=1e-12)  # fast capped on its own
    assert hotkey(cap["s1-fast"], 10 * cap["s1-pro"]) == pytest.approx(at_cap, rel=1e-12)  # pro capped on its own
    # The pro slot is paid at the pro cap even though the same number of decisions/s is far below the fast cap, and
    # the fast slot is not capped at the pro cap: both factors apply separately.
    assert cap["s1-pro"] < cap["s1-fast"]
    assert hotkey(cap["s1-pro"], cap["s1-pro"]) < at_cap
    assert hotkey(cap["s1-fast"], cap["s1-pro"] / 2) < at_cap


@pytest.mark.parametrize("tiered", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test_legacy_single_model_hotkeys_get_the_weights_of_the_base_logic(seed, tiered, make_wallet, fake_chain_cls,
                                                                      tmp_path):
    """Golden comparison: single-model hotkeys (state entries at their slots, as migrated) give exactly what the
    pre-slot weight functions give on plain UID keys."""
    from reax_subnet.s1scoring import compute_pool_weights

    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls,
                     REAX_FEATURE_TIERED_POOLS="1" if tiered else "0")
    rng = random.Random(seed)
    entries, uid = [], 10
    for model in val.pools:
        for _ in range(rng.randint(0, 4)):
            entries.append((uid, model, rng.choice([0.0, 0.2, rng.uniform(0.3, 1.0)]),
                            tiers.CAPACITY[model].target_dps * rng.uniform(0.0, 5.0)))
            uid += 1
    got = _fleet_weights(val, entries)
    ema = {m: {u: e for u, mm, e, _c in entries if mm == m} for m in val.pools}
    if tiered:
        cap = {m: {u: c for u, mm, _e, c in entries if mm == m} for m in val.pools}
        want = compute_tiered_weights(ema, cap, val.budgets, 0, val.params.min_ema, val.params.burn_fraction)
    else:
        want = compute_pool_weights(ema, val.shares, 0, val.params.min_ema, val.params.burn_fraction)
    assert got == want


# ------------------------------------------------------------------------------------------------------- state
def _write_state(path, miners):
    path.write_text(json.dumps({"version": 1, "netuid": 9, "hotkey": "vk", "rounds": 5, "saved_at": 1,
                                "miners": miners}))


def _legacy(hotkey, role="", ema=0.0, capacity=None):
    return {"hotkey": hotkey, "ema": ema, "last_score": ema, "last_fidelity": 1.0, "rounds": 7, "role": role,
            "strikes": 2, **({"capacity": capacity} if capacity is not None else {})}


def test_a_legacy_entry_of_a_higher_model_moves_to_its_slot_without_a_reset(tmp_path):
    path = tmp_path / "state.json"
    _write_state(path, {"2": _legacy("a", role(f"s1-fast"), 0.9), "3": _legacy("b", role("s1-pro"), 0.8, 55.0),
                        "4": _legacy("c", role("s1-lite"), 0.7, 5.0), "5": _legacy("d", role("s1-cpu"), 0.6, 2.0),
                        "6": _legacy("e", "", 0.0)})
    st = load(str(path), 9, "vk")
    assert sorted(st.miners) == [2, 3, 4, 5, 6, slot_key(3, "s1-pro"), slot_key(4, "s1-lite"), slot_key(5, "s1-cpu")]
    moved = st.miners[slot_key(3, "s1-pro")]
    assert (moved.hotkey, moved.ema, moved.role, moved.capacity, moved.strikes, moved.rounds) == (
        "b", 0.8, role("s1-pro"), 55.0, 2, 7)
    assert st.miners[3].hotkey == "b" and st.miners[3].role == "" and st.miners[3].ema == 0.0  # plain entry kept
    assert st.miners[2].role == role("s1-fast") and st.miners[2].ema == 0.9  # s1-fast stays at its UID
    # idempotent: save, load again, same entries
    save(st, str(path))
    again = load(str(path), 9, "vk")
    assert again.miners == st.miners


def test_a_slot_already_present_wins_over_a_legacy_duplicate(tmp_path):
    path = tmp_path / "state.json"
    _write_state(path, {"3": _legacy("b", role("s1-pro"), 0.8), str(slot_key(3, "s1-pro")): _legacy("b", role("s1-pro"), 0.5)})
    st = load(str(path), 9, "vk")
    assert st.miners[slot_key(3, "s1-pro")].ema == 0.5 and st.miners[3].role == ""


@pytest.mark.parametrize("key", [str(4 * SLOT_STRIDE), str(-1), str(10 * SLOT_STRIDE + 3), "x"])
def test_a_state_file_with_an_impossible_slot_key_is_set_aside(tmp_path, key):
    path = tmp_path / "state.json"
    _write_state(path, {"2": _legacy("a"), key: _legacy("b")})
    st = load(str(path), 9, "vk")
    assert st.miners == {} and not path.exists() and list(tmp_path.glob("state.json.corrupt-*"))


def _state(*entries):
    st = ValidatorState(9, "vk")
    for slot, hk, ema in entries:
        st.miners[slot] = MinerState(hotkey=hk, ema=ema, role="r")
    return st


def _sync_snap(**uid_to_hotkey):
    return Snapshot(1, tuple(Neuron(int(k[1:]), v, "c", False, 0.0, None, None) for k, v in uid_to_hotkey.items()), None)


def test_sync_resets_all_slots_of_a_recycled_uid_and_keeps_the_others():
    pro, lite = slot_key(2, "s1-pro"), slot_key(2, "s1-lite")
    st = _state((2, "a", 0.9), (pro, "a", 0.8), (lite, "a", 0.7), (3, "b", 0.5))
    reset = st.sync(_sync_snap(u2="new", u3="b"))
    assert reset == [2] and sorted(st.miners) == [2, 3]
    assert st.miners[2].hotkey == "new" and st.miners[2].ema == 0.0 and st.miners[3].ema == 0.5


def test_sync_moves_all_slots_of_a_renumbered_hotkey():
    pro = slot_key(2, "s1-pro")
    st = _state((2, "a", 0.9), (pro, "a", 0.8))
    reset = st.sync(_sync_snap(u7="a"))
    assert reset == [] and sorted(st.miners) == [7, slot_key(7, "s1-pro")]
    assert st.miners[7].ema == 0.9 and st.miners[slot_key(7, "s1-pro")].ema == 0.8


def test_sync_drops_slots_of_departed_uids_and_recreates_the_plain_entry():
    pro = slot_key(2, "s1-pro")
    st = _state((pro, "a", 0.8), (3, "b", 0.5))  # only a model slot of "a" (its plain entry was never stored)
    st.sync(_sync_snap(u2="a", u3="b"))
    assert st.miners[pro].ema == 0.8 and st.miners[2].ema == 0.0 and st.miners[2].hotkey == "a"
    st.sync(_sync_snap(u3="b"))
    assert sorted(st.miners) == [3]


# --------------------------------------------------------------------------------------------- the round arena
class MultiModelMiner(S1Miner):
    """One endpoint, one hotkey, one S1Miner per model; every request is logged as (model, request id, decisions, t)."""

    def __init__(self, wallet, backends, log, **kw):
        super().__init__(wallet, next(iter(backends.values())), **kw)
        self.subs = {m: S1Miner(wallet, b, **kw) for m, b in backends.items()}
        self.log = log

    async def _decide(self, caller, body):
        req = parse_s1_request(body)
        self.log.append((req.model, req.request_id, len(req.decisions), time.monotonic()))
        sub = self.subs.get(req.model)
        if sub is None:
            return self._reject(400, f"this endpoint does not serve {req.model}")
        return await sub._decide(caller, body)


async def _serve(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


PROBE_ITEMS = 6  # a probe request carries 6 decisions, a normal round's request 4


@contextlib.asynccontextmanager
async def arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch, models=("s1-fast", "s1-pro", "s1-lite"),
                bad=(), engine=None, competitions=None, commitment_fields=None):
    """uid 2 = one hotkey serving ``models`` (backends of the ``bad`` models use another model's weights),
    uid 3 = a plain s1-fast miner. Yields a namespace with the validator, chain, request log and neurons."""
    monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 1)
    probe = dict(inflight=2, window_s=1.5, items_per_request=PROBE_ITEMS, alpha=1.0)
    table = {"s1-fast": CapacityParams(target_dps=40.0, **probe), "s1-pro": CapacityParams(target_dps=10.0, **probe),
             "s1-lite": CapacityParams(target_dps=8.0, **probe)}
    monkeypatch.setattr(tiers, "CAPACITY", table)
    monkeypatch.setattr(s1validator, "CAPACITY", table)
    owner, vwallet = make_wallet("//Owner"), make_wallet("//Validator")
    vk = vwallet.hotkey.ss58_address
    kw = dict(max_concurrent=8, per_caller_concurrency=8, per_caller_per_minute=600, max_pending_per_ip=64)
    log, runners = [], []
    try:
        mw = make_wallet("//Miner/multi")
        # each model computes on its own machine at 400 decisions/s (a hash backend that answers instantly would
        # saturate the validator's probe loop)
        backends = {m: OnMachine(m, Machine(400.0), salt="tiny-model" if m in bad else "reference") for m in models}
        multi = MultiModelMiner(mw, backends, log, **kw)
        multi.set_validators(frozenset({vk}))
        runner, port = await _serve(multi.app())
        runners.append(runner)
        sw = make_wallet("//Miner/single")
        single = S1Miner(sw, OnMachine("s1-fast", Machine(400.0)), **kw)
        single.set_validators(frozenset({vk}))
        runner, sport = await _serve(single.app())
        runners.append(runner)
        text = multi_text(*models) if len(models) > 1 else commitment_text(models[0], MID[models[0]])
        fields = tuple(commitment_fields or (text,))
        neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
                   Neuron(1, vk, "c1", True, 5000.0, None, None),
                   Neuron(2, mw.hotkey.ss58_address, "c2", False, 0.0, "127.0.0.1", port, commitment="".join(fields),
                          commitment_block=50, commitment_fields=fields),
                   Neuron(3, sw.hotkey.ss58_address, "c3", False, 0.0, "127.0.0.1", sport,
                          commitment=commitment_text("s1-fast", MID["s1-fast"]), commitment_block=50)]
        chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
        cfg = load_validator({**DEV, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json"),
                              "REAX_EMA_ALPHA": "0.5", "REAX_BURN_FRACTION": "0.0", "REAX_WARMUP_ROUNDS": "0",
                              "REAX_FEATURE_TIERED_POOLS": "1", "REAX_FEATURE_S1_PRO_POOL": "1"})
        pools = {m: PoolParams(m, p.temperature, tau_mean=0.2, tau_strike=0.5, tau_item=2.0, items_per_round=4,
                               verified_per_round=4, latency_fast_s=0.0, latency_slow_s=4.0)
                 for m, p in s1validator.POOL_PARAMS.items() if m in table}
        refs = {m: S1HashBackend(m, salt="reference") for m in pools}
        val = S1Validator(cfg, vwallet, chain, refs, pools=pools, engine=engine, competitions=competitions)
        session = aiohttp.ClientSession(trust_env=False)
        try:
            yield SimpleNamespace(val=val, chain=chain, session=session, log=log, neurons=neurons, tmp=tmp_path,
                                  rounds=lambda n: _rounds(val, session, n))
        finally:
            await session.close()
    finally:
        for r in runners:
            await r.cleanup()


async def _rounds(val, session, n):
    out = None
    for _ in range(n):
        out = await val.run_round(session)
    return out


def _set_commitment(a, uid, *fields):
    i = next(k for k, n in enumerate(a.chain.neurons) if n.uid == uid)
    a.chain.neurons[i] = replace(a.chain.neurons[i], commitment="".join(fields), commitment_fields=tuple(fields))


async def test_every_model_of_a_hotkey_is_queried_with_its_own_request_and_probed_in_one_window(
        tmp_path, make_wallet, fake_chain_cls, monkeypatch, started):
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch) as a:
        s = await a.rounds(1)  # round 1 only confirms the roles
        assert s["targets"] == 0 and a.log == []
        s = await a.rounds(1)
        assert s["targets"] == 4  # 3 slots of uid 2 + the plain s1-fast miner
        normal = [(m, rid) for m, rid, n, _t in a.log if n == 4]
        assert sorted(m for m, _ in normal) == ["s1-fast", "s1-lite", "s1-pro"]
        assert len({rid for _, rid in normal}) == 3  # own request per model
        slots = {slot_key(2, "s1-fast"), slot_key(2, "s1-pro"), slot_key(2, "s1-lite")}
        assert slots <= set(s["scores"]) and 3 in s["scores"]
        for slot in slots:
            r = s["scores"][slot]
            assert r["model"] == slot_model_of(slot) and r["strike"] is False and r["score"] > 0.8 and r["error"] is None
            assert a.val.state.miners[slot].role == role(r["model"]) and a.val.state.miners[slot].ema > 0
        # one common probe window covers every slot of every pool: first probe requests of all three models of the
        # hotkey leave within the send-skew bound, and all slots got a capacity.
        assert set(s["capacity"]) >= slots | {3}
        first = {}
        for m, _rid, n, t in a.log:
            if n == PROBE_ITEMS:
                first.setdefault(m, t)
        assert set(first) == {"s1-fast", "s1-pro", "s1-lite"}
        assert max(first.values()) - min(first.values()) < s1validator.PROBE_SEND_SKEW_MAX_S + 0.5
        assert all(a.val.state.miners[slot].capacity is not None for slot in slots)
        # weights: the hotkey is paid once per model, folded onto uid 2; no slot key reaches the chain
        uids, weights, _ = a.chain.set_calls[-1]
        assert all(u < SLOT_STRIDE for u in uids) and 2 in uids and 3 in uids
        assert math.fsum(weights) == pytest.approx(1.0)


def slot_model_of(slot):
    return E.slot_model(slot)


async def test_a_strike_on_one_model_slot_leaves_the_other_slots_alone(tmp_path, make_wallet, fake_chain_cls,
                                                                       monkeypatch, started):
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch, bad=("s1-pro",)) as a:
        s = await a.rounds(4)
        pro, fast, lite = (slot_key(2, m) for m in ("s1-pro", "s1-fast", "s1-lite"))
        assert s["scores"][pro]["strike"] is True
        st = a.val.state.miners
        assert st[pro].ema == 0.0 and st[pro].strikes >= 1 and st[pro].capacity == 0.0
        for slot in (fast, lite):
            assert st[slot].strikes == 0 and st[slot].ema > 0.5 and (st[slot].capacity or 0) > 0
        uids, weights, _ = a.chain.set_calls[-1]
        assert 2 in uids  # still paid for fast and lite


async def test_removing_a_model_from_the_commitment_drops_only_its_slot(tmp_path, make_wallet, fake_chain_cls,
                                                                        monkeypatch, started):
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch) as a:
        await a.rounds(3)
        fast, pro, lite = (slot_key(2, m) for m in ("s1-fast", "s1-pro", "s1-lite"))
        assert {fast, pro, lite} <= set(a.val.state.miners)
        before = {k: a.val.state.miners[k].ema for k in (fast, pro)}
        _set_commitment(a, 2, multi_text("s1-fast", "s1-pro"))
        await a.rounds(1)  # seen once: nothing changes yet
        assert lite in a.val.state.miners and a.val.state.miners[lite].role == role("s1-lite")
        await a.rounds(1)  # confirmed
        assert lite not in a.val.state.miners
        assert a.val.state.miners[pro].role == role("s1-pro") and a.val.state.miners[fast].role == role("s1-fast")
        assert a.val.state.miners[pro].ema >= before[pro] * 0.9 and a.val.state.miners[fast].ema >= before[fast] * 0.9
        saved = json.loads((tmp_path / "state.json").read_text())["miners"]
        assert str(lite) not in saved and str(pro) in saved
        # down to a single model: the pro slot goes too, the s1-fast entry keeps its UID key and its score
        _set_commitment(a, 2, commitment_text("s1-fast", MID["s1-fast"]))
        await a.rounds(2)
        assert pro not in a.val.state.miners and a.val.state.miners[2].role == role("s1-fast")
        assert a.val.state.miners[2].ema >= before[fast] * 0.9


async def test_a_model_added_to_the_commitment_starts_a_new_slot_and_keeps_the_others(
        tmp_path, make_wallet, fake_chain_cls, monkeypatch, started):
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch, models=("s1-fast", "s1-pro", "s1-lite")) as a:
        _set_commitment(a, 2, commitment_text("s1-fast", MID["s1-fast"]))
        await a.rounds(3)
        fast = a.val.state.miners[2]
        assert fast.role == role("s1-fast") and fast.ema > 0.5 and slot_key(2, "s1-pro") not in a.val.state.miners
        ema, rounds = fast.ema, fast.rounds
        _set_commitment(a, 2, multi_text("s1-fast", "s1-pro"))
        await a.rounds(1)
        assert a.val.state.miners[2].role == role("s1-fast")  # unchanged, still scored: the s1-fast role is the same
        await a.rounds(2)
        pro = a.val.state.miners[slot_key(2, "s1-pro")]
        assert pro.role == role("s1-pro") and pro.ema > 0
        assert a.val.state.miners[2].ema >= ema * 0.9 and a.val.state.miners[2].rounds > rounds  # no reset


# ----------------------------------------------------------------------------------------------- start constants
async def test_before_the_start_block_a_multi_model_commitment_has_no_role(tmp_path, make_wallet, fake_chain_cls,
                                                                           monkeypatch, caplog):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", None)
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch) as a:
        with caplog.at_level(logging.WARNING, logger="reax.s1validator"):
            s = await a.rounds(3)
        assert s["targets"] == 1 and a.log == []  # only the plain s1-fast miner is queried
        assert not any(slot_uid(k) == 2 and a.val.state.miners[k].role for k in a.val.state.miners)
        msgs = [r.getMessage() for r in caplog.records if "uid 2 is not queried" in r.getMessage()]
        assert len(msgs) == 1 and "held before the start block" in msgs[0]


async def test_a_multi_model_commitment_counts_from_the_start_block(tmp_path, make_wallet, fake_chain_cls, monkeypatch):
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch) as a:
        first = a.chain.block + 10
        monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", first + 25)  # FakeChain adds 10 blocks per snapshot
        s1 = await a.rounds(1)
        assert s1["targets"] == 0
        await a.rounds(1)
        assert a.log == []  # not yet started: no model of uid 2 is queried
        monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 0)
        await a.rounds(2)
        assert {m for m, *_ in a.log} == {"s1-fast", "s1-pro", "s1-lite"}


def test_a_stale_manifest_or_disabled_pool_voids_only_that_model(tmp_path, make_wallet, fake_chain_cls, started,
                                                                 caplog):
    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls, REAX_FEATURE_S1_PRO_POOL="1")
    del val.pools["s1-cpu"]
    stale = "0123456789abcdef"
    text = f"reax-s1m/1:s1-fast:{MID['s1-fast']},s1-pro:{stale},s1-cpu:{MID['s1-cpu']}"
    n = _neuron(2, text, (text,))
    assert val.slot_roles(n) == {"s1-fast": role("s1-fast")}
    snap = Snapshot(100, (n,), None)
    with caplog.at_level(logging.WARNING, logger="reax.s1validator"):
        val._log_rejected_commitments(snap)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert "different releases" in msg and stale in msg and "s1-cpu pool is not enabled" in msg
    assert "s1-fast manifest" not in msg


# --------------------------------------------------------------------------------------------------- dual role
class FakeDriver:
    background = True

    def __init__(self):
        self.refreshed, self.kicked = [], []

    def refresh(self, snap):
        self.refreshed.append(snap)

    def kick(self, snap, exclude, between=None):
        self.kicked.append((snap, exclude))

    def due(self):
        return False


class FakeEngine:
    def __init__(self):
        self.weight_snaps = []

    def is_engine_commitment(self, commitment):
        return commitment.startswith("reax-eng/1:")

    def weights(self, base, snap, exclude, burn):
        self.weight_snaps.append(snap)
        return base


async def test_a_dual_role_commitment_is_served_for_inference_and_competes_with_its_competition_element(
        tmp_path, make_wallet, fake_chain_cls, monkeypatch, started):
    driver, engine = FakeDriver(), FakeEngine()
    inference = multi_text("s1-fast", "s1-pro")
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch, models=("s1-fast", "s1-pro"),
                     engine=engine, competitions=driver, commitment_fields=(inference, SPEC)) as a:
        s = await a.rounds(3)
        # inference: both models of uid 2 are queried and scored
        assert {m for m, *_ in a.log} == {"s1-fast", "s1-pro"}
        assert slot_key(2, "s1-pro") in s["scores"] and 2 in s["scores"]
        assert a.val.state.miners[2].role == role("s1-fast")
        # the inference view is what the serving round and the weights use ...
        assert {n.uid: n.commitment for n in a.val._latest_snap.neurons}[2] == inference
        # ... the competition code gets the competition element, with the plain neurons otherwise unchanged
        assert driver.refreshed and driver.kicked
        for snap in (driver.refreshed[-1], driver.kicked[-1][0], engine.weight_snaps[-1]):
            by_uid = {n.uid: n for n in snap.neurons}
            assert by_uid[2].commitment == SPEC and by_uid[3].commitment == ""
            assert by_uid[2].ip == "127.0.0.1" and by_uid[2].hotkey == a.neurons[2].hotkey
        assert driver.kicked[-1][1] == {1, 0}  # the validator and the owner never compete


async def test_a_dual_role_commitment_before_its_start_block_has_no_role(tmp_path, make_wallet, fake_chain_cls,
                                                                         monkeypatch, caplog):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 0)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", None)
    driver, engine = FakeDriver(), FakeEngine()
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch, models=("s1-fast", "s1-pro"),
                     engine=engine, competitions=driver,
                     commitment_fields=(multi_text("s1-fast", "s1-pro"), SPEC)) as a:
        with caplog.at_level(logging.WARNING, logger="reax.s1validator"):
            await a.rounds(3)
        assert a.log == []
        assert {n.uid: n.commitment for n in driver.refreshed[-1].neurons}[2] == ""
        assert any("multi-field commitments count from block None" in r.getMessage() for r in caplog.records)


async def test_a_competition_only_commitment_and_a_plain_one_behave_as_before(tmp_path, make_wallet, fake_chain_cls,
                                                                             monkeypatch):
    """Single-field commitments: the competition code sees exactly the raw commitment of a competition entrant and
    nothing of an inference miner, the inference round the other way round."""
    driver, engine = FakeDriver(), FakeEngine()
    async with arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch, models=("s1-fast",),
                     engine=engine, competitions=driver) as a:
        _set_commitment(a, 3, SPEC)  # uid 3 becomes a competition-only entrant
        await a.rounds(3)
        comp = {n.uid: n.commitment for n in driver.kicked[-1][0].neurons}
        inf = {n.uid: n.commitment for n in a.val._latest_snap.neurons}
        assert comp[3] == SPEC and comp[2] == "" and inf[3] == "" and inf[2].startswith("reax-s1/1:")
        assert {m for m, *_ in a.log} == {"s1-fast"}


# ------------------------------------------------------------------------------------------------------- router
def _router(tmp_path, make_wallet, fake_chain_cls, neurons, scores, models=("s1-fast", "s1-pro")):
    wallet = make_wallet("//RouterValidator")
    vcfg = SimpleNamespace(common=SimpleNamespace(netuid=9), state_path=str(tmp_path / "state.json"),
                           allow_private_ips=True)
    cfg = RouterConfig("x" * 32, "plumb-pin", "tokenizer-pin", models=models, timeout_s=0.25, shadow_rate=0,
                       strikes_file=str(tmp_path / "strikes.jsonl"))
    router = Router(cfg, vcfg, wallet, fake_chain_cls(list(neurons)), lambda model, decisions: 1)
    state = ValidatorState(9, wallet.hotkey.ss58_address)
    state.miners.update(scores)
    save(state, vcfg.state_path)
    router.fresh_at = 0.0
    return router


async def test_the_router_picks_a_multi_model_hotkey_for_each_of_its_models(tmp_path, make_wallet, fake_chain_cls,
                                                                            started):
    text = multi_text("s1-fast", "s1-pro")
    n = replace(_neuron(2, text, (text,)), hotkey="hk2")
    scores = {2: MinerState("hk2", ema=0.9, role=role("s1-fast")),
              slot_key(2, "s1-pro"): MinerState("hk2", ema=0.8, role=role("s1-pro"))}
    router = _router(tmp_path, make_wallet, fake_chain_cls, [n], scores)
    await router.refresh()
    assert [(x.uid, ema) for x, ema in router.miners["s1-fast"]] == [(2, 0.9)]
    assert [(x.uid, ema) for x, ema in router.miners["s1-pro"]] == [(2, 0.8)]
    assert [x.uid for x, _ in router.candidates("s1-pro")] == [2]
    # a model whose slot score is below the minimum is not offered, the other model still is
    scores[slot_key(2, "s1-pro")] = MinerState("hk2", ema=0.1, role=role("s1-pro"))
    router = _router(tmp_path, make_wallet, fake_chain_cls, [n], scores)
    await router.refresh()
    assert router.miners["s1-fast"] and not router.miners["s1-pro"]


async def test_the_router_reads_legacy_state_files_and_ignores_held_multi_model_commitments(
        tmp_path, make_wallet, fake_chain_cls, monkeypatch):
    single_fast = replace(_neuron(2, commitment_text("s1-fast", MID["s1-fast"])), hotkey="hk2")
    single_pro = replace(_neuron(3, commitment_text("s1-pro", MID["s1-pro"])), hotkey="hk3")
    legacy = {2: MinerState("hk2", ema=0.9, role=role("s1-fast")), 3: MinerState("hk3", ema=0.8, role=role("s1-pro"))}
    router = _router(tmp_path, make_wallet, fake_chain_cls, [single_fast, single_pro], legacy)
    await router.refresh()  # state written before the slot migration: s1-pro still at its plain UID
    assert [x.uid for x, _ in router.miners["s1-fast"]] == [2] and [x.uid for x, _ in router.miners["s1-pro"]] == [3]
    migrated = {2: legacy[2], 3: MinerState("hk3"), slot_key(3, "s1-pro"): legacy[3]}
    router = _router(tmp_path, make_wallet, fake_chain_cls, [single_fast, single_pro], migrated)
    await router.refresh()
    assert [x.uid for x, _ in router.miners["s1-pro"]] == [3]
    text = multi_text("s1-fast", "s1-pro")
    held = replace(_neuron(4, text, (text,)), hotkey="hk4")  # MULTIMODEL_START_BLOCK is None
    router = _router(tmp_path, make_wallet, fake_chain_cls, [held],
                     {4: MinerState("hk4", ema=0.9, role=role("s1-fast"))})
    await router.refresh()
    assert not router.miners["s1-fast"] and not router.miners["s1-pro"]


# --------------------------------------------------------------------------------------------- probe shard groups
def test_probe_shards_keep_all_slots_of_a_hotkey_together_even_without_a_coldkey():
    from reax_subnet import probeshard

    def n(slot, hotkey, coldkey):
        return SimpleNamespace(uid=slot, hotkey=hotkey, coldkey=coldkey)

    targets = [n(2, "a", "ck"), n(5, "b", ""), n(slot_key(2, "s1-pro"), "a", "ck"), n(slot_key(5, "s1-pro"), "b", ""),
               n(7, "c", "")]
    groups = [[t.uid for t in g] for g in probeshard.sibling_groups(targets)]
    assert groups == [[2, slot_key(2, "s1-pro")], [5, slot_key(5, "s1-pro")], [7]]


def test_rc10_upgrade_keeps_paying_lite_pro_cpu_from_their_slots(make_wallet, fake_chain_cls, tmp_path):
    """Regression (21 Oct 08:51Z miner report): after the rc10 upgrade a miner that serves only s1-lite / s1-pro / s1-cpu
    keeps a plain UID row with role "" and ema 0 (the rounds of that row score 0), while its EMA, role and capacity move to
    its slot. The weights must come from the slot: every such UID is paid exactly what it was paid before the upgrade."""
    path = tmp_path / "legacy-post8.json"   # not the validator's own REAX_STATE_PATH
    legacy = {"2": _legacy("hk2", role("s1-fast"), 0.9, tiers.CAPACITY["s1-fast"].target_dps),
              "3": _legacy("hk3", role("s1-pro"), 0.8, tiers.CAPACITY["s1-pro"].target_dps * 2),
              "4": _legacy("hk4", role("s1-lite"), 0.7, tiers.CAPACITY["s1-lite"].target_dps),
              "5": _legacy("hk5", role("s1-cpu"), 0.6, tiers.CAPACITY["s1-cpu"].target_dps)}
    _write_state(path, legacy)
    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    st = load(str(path), 9, "vk")
    val.state.miners.clear()
    val.state.miners.update(st.miners)
    snap = Snapshot(100, tuple(Neuron(u, f"hk{u}", f"ck{u}", False, 0.0, "127.0.0.1", 9000 + u) for u in (0, 1, 2, 3, 4, 5)), "hk0")
    w = val.compute_weights(snap)
    assert all(w.get(u, 0.0) > 0 for u in (2, 3, 4, 5)), w
    for u, model in ((3, "s1-pro"), (4, "s1-lite"), (5, "s1-cpu")):
        assert val.state.miners[u].role == "" and val.state.miners[slot_key(u, model)].role == role(model)
        # as live after the upgrade: the plain row keeps scoring 0 (role "", ema 0) - it must not change any weight
        val.state.miners[u] = replace(val.state.miners[u], ema=0.0, last_score=0.0, rounds=14)
    assert val.compute_weights(snap) == w
    # the same numbers as plain single-model UIDs (the pre-upgrade layout) give the same weights
    val.state.miners.clear()
    for k, v in legacy.items():
        val.state.miners[slot_key(int(k), v["role"].split(":")[0])] = MinerState(hotkey=v["hotkey"], ema=v["ema"], role=v["role"],
                                                                                 capacity=v.get("capacity"))
    assert val.compute_weights(snap) == pytest.approx(w, rel=1e-12)
