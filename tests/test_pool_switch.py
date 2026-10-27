"""Timed manifest switch of the s1-pro pool (1.2.0rc11.post8.1): Rune -> mammoth-1 at a fixed UTC time.

Before the switch time everything is as in post7.2. From it on the validator accepts the new manifest and asks the new
reference runtime; a miner still committed to the previous manifest is held (EMA frozen, no weight, no strike, no reset)
for 24 h and then handled like any stale manifest. Miners are real aiohttp servers, the chain is faked, the clock is ours.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import aiohttp
import pytest

from reax_subnet import manifest as mf
from reax_subnet.chain import Neuron
from reax_subnet.config import load_validator
from reax_subnet.elements import slot_key
from reax_subnet.manifest import PoolSwitch, commitment_text, load_manifest, manifest_name
from reax_subnet.miner import S1Miner
from reax_subnet.s1backends import S1HashBackend, UnavailableBackend
from reax_subnet.s1scoring import PoolParams
from reax_subnet.s1validator import POOL_PARAMS, POOL_SWITCH_PARAMS, SHARES_BOTH, S1Validator
from reax_subnet.state import load
from tests.test_s1 import TIMEOUT, _serve

T = datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc)
OLD = load_manifest("s1-pro", name="s1-pro-v1.json")
NEW = load_manifest("s1-pro", name="s1-pro-v2-mammoth.json")
FAST = load_manifest("s1-fast")
SWITCH = {"s1-pro": PoolSwitch("s1-pro", T, "s1-pro-v2-mammoth.json")}


# ------------------------------------------------------------------ the constant and the manifest file
@pytest.mark.real_switch
def test_release_constant():
    sw = mf.POOL_MANIFEST_SWITCH
    assert set(sw) == {"s1-pro"}
    assert sw["s1-pro"].at == datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc)
    assert sw["s1-pro"].grace == timedelta(hours=24) and sw["s1-pro"].manifest_name == "s1-pro-v2-mammoth.json"
    # the clock decides for every caller of load_manifest (miner, selfcheck, multinode)
    assert load_manifest("s1-pro", T - timedelta(seconds=1)).manifest_id == OLD.manifest_id
    assert load_manifest("s1-pro", T).manifest_id == NEW.manifest_id


def test_manifest_ids_are_pinned():
    # the old id is the one the shipped self-check data and every live commitment carry
    assert OLD.manifest_id == "5b11440bf57b5434"
    assert NEW.manifest_id == "3f3558e583b0a1d6"
    assert OLD.manifest_id != NEW.manifest_id and NEW.data["model"] == "s1-pro"


def test_new_manifest_pins_the_calibrated_reference():
    d = NEW.data
    assert d["runtime"]["image"].endswith("sha256:3080a593060542bccbdaf7cf0c66593fc182d9753b3facd50206df86d93b8588")
    assert d["weights"]["hf_repo"] == "baselabs/mammoth-1"
    assert d["weights"]["revision"] == "61cec44c281c693992e3d6ebeb44dd3402dfa186"
    assert NEW.temperature == 1.0 == POOL_SWITCH_PARAMS["s1-pro"].temperature
    files = NEW.weight_files
    assert len(files) == 14 and all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in files.values())
    assert files["model-00001-of-00002.safetensors"].startswith("29701f9a")
    assert files["model-00002-of-00002.safetensors"].startswith("4da0ff54")
    assert not {"README.md", ".gitattributes", "assets/banner.png"} & set(files)


def test_switch_params_are_valid_and_only_the_temperature_differs():
    new, old = POOL_SWITCH_PARAMS["s1-pro"], POOL_PARAMS["s1-pro"]
    assert set(POOL_SWITCH_PARAMS) == {"s1-pro"}
    assert replace(new, temperature=old.temperature) == old


def test_manifest_name_before_after_and_other_models():
    assert manifest_name("s1-pro", T - timedelta(seconds=1), SWITCH) == "s1-pro-v1.json"
    assert manifest_name("s1-pro", T, SWITCH) == "s1-pro-v2-mammoth.json"
    assert manifest_name("s1-pro", T + timedelta(days=90), SWITCH) == "s1-pro-v2-mammoth.json"
    far = T + timedelta(days=365)
    for model in ("s1-fast", "s1-lite", "s1-cpu"):
        assert manifest_name(model, far, SWITCH) == mf.RELEASE_MANIFESTS[model]
    assert manifest_name("s1-pro", far, {}) == "s1-pro-v1.json"  # nothing configured = today's behaviour


def test_no_switch_commitments_stay_valid():
    assert mf.parse_commitment(commitment_text("s1-pro", NEW.manifest_id)) == ("s1-pro", NEW.manifest_id)


# ------------------------------------------------------------------ the validator
class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def _backend(model: str, salt: str, manifest) -> S1HashBackend:
    b = S1HashBackend(model, salt=salt)
    b.manifest = manifest
    return b


@pytest.fixture
async def arena(tmp_path, make_wallet, fake_chain_cls):
    owner = make_wallet("//Owner")
    vwallet = make_wallet("//Validator")
    vk = vwallet.hotkey.ss58_address
    runners = []
    neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
               Neuron(1, vk, "c1", True, 5000.0, None, None)]
    slots: dict[str, int] = {}

    async def add(name, uid, model, backend, manifest):
        w = make_wallet(f"//Miner/{name}")
        m = S1Miner(w, backend, max_concurrent=4, per_caller_concurrency=2)
        m.set_validators(frozenset({vk}))
        runner, port = await _serve(m.app())
        runners.append(runner)
        neurons.append(Neuron(uid, w.hotkey.ss58_address, f"c{uid}", False, 0.0, "127.0.0.1", port,
                              commitment=commitment_text(model, manifest.manifest_id)))
        slots[name] = slot_key(uid, model)

    await add("fast", 2, "s1-fast", _backend("s1-fast", "reference", FAST), FAST)
    await add("rune", 3, "s1-pro", _backend("s1-pro", "rune", OLD), OLD)
    await add("rune2", 5, "s1-pro", _backend("s1-pro", "rune", OLD), OLD)
    await add("mammoth", 4, "s1-pro", _backend("s1-pro", "mammoth", NEW), NEW)
    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
    env = {
        "REAX_NETUID": "9", "REAX_ALLOW_DEV_MODELS": "1", "REAX_NETWORK": "ws://127.0.0.1:9951",
        "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json"), "REAX_EMA_ALPHA": "0.5",
        "REAX_BURN_FRACTION": "0.0", "REAX_WARMUP_ROUNDS": "0", "REAX_FEATURE_S1_PRO_POOL": "1",
        "REAX_FEATURE_TIERED_POOLS": "0",
    }
    cfg = load_validator(env)
    pools = {
        "s1-fast": PoolParams("s1-fast", 2.07, tau_mean=0.2, tau_strike=0.5, tau_item=2.0, items_per_round=8,
                              verified_per_round=8, latency_fast_s=0.0, latency_slow_s=TIMEOUT),
        "s1-pro": PoolParams("s1-pro", 2.0, tau_mean=0.2, tau_strike=0.5, tau_item=2.0, items_per_round=8,
                             verified_per_round=3, latency_fast_s=0.0, latency_slow_s=TIMEOUT),
    }
    next_pools = {"s1-pro": replace(pools["s1-pro"], temperature=1.0)}
    refs = {"s1-fast": S1HashBackend("s1-fast", salt="reference"), "s1-pro": _backend("s1-pro", "rune", OLD)}
    nxt = {"s1-pro": _backend("s1-pro", "mammoth", NEW)}
    clock = Clock(T - timedelta(hours=1))

    def make(**kw):
        args = dict(pools=pools, shares=SHARES_BOTH, switches=SWITCH, next_backends=nxt, next_pools=next_pools,
                    clock=clock)
        args.update(kw)
        return S1Validator(cfg, vwallet, chain, refs, **args)

    session = aiohttp.ClientSession(trust_env=False)
    yield {"make": make, "val": make(), "chain": chain, "session": session, "slots": slots, "refs": refs, "next": nxt,
           "clock": clock, "cfg": cfg, "neurons": neurons, "pools": pools, "vwallet": vwallet}
    await session.close()
    for r in runners:
        await r.cleanup()


async def _rounds(a, n, val=None):
    out = None
    for _ in range(n):
        out = await (val or a["val"]).run_round(a["session"])
    return out


def _weights(a, val=None):
    val = val or a["val"]
    snap = a["chain"].snapshot()
    return val.compute_weights(snap)


async def test_before_the_switch_nothing_changes(arena):
    a = arena
    val, slots = a["val"], a["slots"]
    assert val.manifests["s1-pro"].manifest_id == OLD.manifest_id and val.backends["s1-pro"] is a["refs"]["s1-pro"]
    assert val.pools["s1-pro"].temperature == 2.0
    summary = await _rounds(a, 3)
    assert summary["scores"][slots["rune"]]["score"] > 0.8 and val.state.miners[slots["rune"]].ema > 0.5
    assert slots["mammoth"] not in summary["scores"]  # the new manifest is not accepted yet
    assert val.state.miners.get(slots["mammoth"]) is None or val.state.miners[slots["mammoth"]].ema == 0.0
    assert a["next"]["s1-pro"].calls == 0  # the new reference is not used before T
    assert val.state.miners[slots["fast"]].ema > 0.5


async def test_switch_moves_manifest_reference_and_params(arena):
    a = arena
    val, slots, clock = a["val"], a["slots"], a["clock"]
    await _rounds(a, 3)
    old_calls = a["refs"]["s1-pro"].calls
    clock.t = T
    summary = await _rounds(a, 3)
    assert val.manifests["s1-pro"].manifest_id == NEW.manifest_id
    assert val.backends["s1-pro"] is a["next"]["s1-pro"]
    assert val.pools["s1-pro"].temperature == 1.0
    assert summary["scores"][slots["mammoth"]]["passed"] == summary["scores"][slots["mammoth"]]["verified"] > 0
    assert val.state.miners[slots["mammoth"]].role == f"s1-pro:{NEW.manifest_id}"
    assert a["refs"]["s1-pro"].calls == old_calls  # the old runtime is not asked after T
    assert a["next"]["s1-pro"].calls > 0
    # requests carry the new manifest id
    assert val._new_request("s1-pro").manifest_id == NEW.manifest_id
    assert val._new_request("s1-fast").manifest_id == FAST.manifest_id


async def test_grace_old_manifest_miner_is_frozen_not_struck_not_reset(arena):
    a = arena
    val, slots, clock = a["val"], a["slots"], a["clock"]
    await _rounds(a, 4)
    rune = slots["rune"]
    before = val.state.miners[rune]
    assert before.ema > 0.5 and before.role == f"s1-pro:{OLD.manifest_id}"
    snapshot = (before.ema, before.strikes, before.rounds, before.role, before.last_score)
    for hours in (0, 1, 12, 23):
        clock.t = T + timedelta(hours=hours, minutes=59 if hours == 23 else 0)
        summary = await _rounds(a, 3)
        m = val.state.miners[rune]
        assert (m.ema, m.strikes, m.rounds, m.role, m.last_score) == snapshot, hours
        assert rune not in summary["scores"]  # not queried, so neither decayed nor struck
    # no weight while held (only current-manifest roles count); the s1-fast miner and the mammoth miners keep theirs
    w = _weights(a)
    assert 3 not in w and 5 not in w
    assert w[4] > 0 and w[2] > 0
    assert math.fsum(w.values()) == pytest.approx(1.0)
    assert w[2] == pytest.approx(0.35, rel=1e-6)  # the s1-fast share is untouched by the switch


async def test_after_grace_normal_stale_manifest_handling(arena):
    a = arena
    val, slots, clock = a["val"], a["slots"], a["clock"]
    await _rounds(a, 4)
    clock.t = T + timedelta(hours=23, minutes=59, seconds=59)
    await _rounds(a, 2)
    assert val.state.miners[slots["rune"]].ema > 0.5
    clock.t = T + timedelta(hours=24)
    await _rounds(a, 2)
    m = val.state.miners.get(slots["rune"])
    assert m is None or (m.role == "" and m.ema == 0.0)  # as any stale manifest: dropped, score gone
    assert 3 not in _weights(a)


async def test_miner_switching_inside_grace_starts_on_the_new_manifest(arena):
    a = arena
    val, slots, clock, chain = a["val"], a["slots"], a["clock"], a["chain"]
    await _rounds(a, 3)
    clock.t = T + timedelta(hours=2)
    await _rounds(a, 2)
    # rune2 now serves mammoth and commits the new manifest
    idx = next(i for i, n in enumerate(chain.neurons) if n.uid == 5)
    other = next(n for n in chain.neurons if n.uid == 4)
    chain.neurons[idx] = replace(chain.neurons[idx], port=other.port,
                                 commitment=commitment_text("s1-pro", NEW.manifest_id))
    # (same endpoint as the mammoth miner would fail the axon check, so only the role handling is asserted)
    await _rounds(a, 3)
    m = val.state.miners[slots["rune2"]]
    assert m.role == f"s1-pro:{NEW.manifest_id}"


async def test_other_pools_unchanged_across_the_switch(arena):
    a = arena
    val, slots, clock = a["val"], a["slots"], a["clock"]
    emas, strikes = [], []
    for t in (T - timedelta(hours=2), T - timedelta(seconds=1), T, T + timedelta(hours=30)):
        clock.t = t
        s = await _rounds(a, 2)
        assert s["scores"][slots["fast"]]["score"] > 0.8 and not s["scores"][slots["fast"]]["strike"]
        emas.append(val.state.miners[slots["fast"]].ema)
        strikes.append(val.state.miners[slots["fast"]].strikes)
    assert strikes == [0, 0, 0, 0] and emas == sorted(emas) and emas[-1] > 0.9  # keeps rising, never reset or decayed
    assert val.manifests["s1-fast"].manifest_id == FAST.manifest_id and val.pools["s1-fast"] == a["pools"]["s1-fast"]
    assert set(val.manifests) == {"s1-fast", "s1-pro"}


async def test_no_next_runtime_leaves_the_pool_unscored_not_struck(arena):
    a = arena
    val = a["make"](next_backends={})
    slots, clock = a["slots"], a["clock"]
    await _rounds(a, 4, val)
    mammoth_before = val.state.miners.get(slots["mammoth"])
    old_calls = a["refs"]["s1-pro"].calls
    clock.t = T + timedelta(minutes=5)
    summary = await _rounds(a, 4, val)
    assert isinstance(val.backends["s1-pro"], UnavailableBackend)
    assert a["refs"]["s1-pro"].calls == old_calls  # never falls back to the old runtime for the new manifest
    m = val.state.miners[slots["mammoth"]]
    assert m.ema == 0.0 and m.strikes == 0 and (mammoth_before is None or mammoth_before.strikes == 0)
    sc = summary["scores"].get(slots["mammoth"])
    assert sc is None or sc["error"] == "reference unavailable"
    # the s1-fast pool keeps scoring while s1-pro has no reference
    assert summary["scores"][slots["fast"]]["score"] > 0.8


async def test_restart_after_the_switch_starts_on_the_new_manifest(arena):
    a = arena
    a["clock"].t = T + timedelta(hours=3)
    val = a["make"]()
    assert val.manifests["s1-pro"].manifest_id == NEW.manifest_id and val.backends["s1-pro"] is a["next"]["s1-pro"]
    summary = await _rounds(a, 3, val)
    assert summary["scores"][a["slots"]["mammoth"]]["score"] > 0.8


async def test_no_switch_configured_is_exactly_the_old_behaviour(arena):
    a = arena
    a["clock"].t = T + timedelta(days=30)
    val = a["make"](switches={})
    assert val.manifests["s1-pro"].manifest_id == OLD.manifest_id and val.pools["s1-pro"].temperature == 2.0
    summary = await _rounds(a, 3, val)
    assert summary["scores"][a["slots"]["rune"]]["score"] > 0.8
    assert a["slots"]["mammoth"] not in summary["scores"]


async def test_state_compatibility_both_ways(arena):
    a = arena
    slots, clock = a["slots"], a["clock"]
    # state written by a validator without a switch (post7.2 behaviour), read by the switching one
    old_val = a["make"](switches={})
    await _rounds(a, 4, old_val)
    ema = old_val.state.miners[slots["rune"]].ema
    assert ema > 0.5
    path = a["cfg"].state_path
    raw = json.loads(open(path).read())
    assert "switch" not in json.dumps(raw)  # the state has no switch fields at all
    clock.t = T + timedelta(hours=1)
    new_val = a["make"]()
    assert new_val.state.miners[slots["rune"]].ema == pytest.approx(ema)
    await _rounds(a, 3, new_val)
    frozen = new_val.state.miners[slots["rune"]]
    assert frozen.ema == pytest.approx(ema) and frozen.role == f"s1-pro:{OLD.manifest_id}"
    # state written after the switch (frozen slot, new-manifest slot), read by a validator that does not switch
    rolled_back = a["make"](switches={})
    assert rolled_back.state.miners[slots["rune"]].ema == pytest.approx(ema)
    assert rolled_back.state.miners[slots["mammoth"]].role == f"s1-pro:{NEW.manifest_id}"
    await _rounds(a, 3, rolled_back)  # a rollback validator carries on: rune scored again, mammoth is stale for it
    assert rolled_back.state.miners[slots["rune"]].ema > 0.5
    assert load(path, 9, rolled_back.hotkey).miners  # the file stays loadable


# ------------------------------------------------------------------ configuration
def _env(tmp_path, **extra):
    tok = tmp_path / "token"
    tok.write_text("t" * 32)
    env = {"REAX_NETUID": "9", "REAX_ALLOW_DEV_MODELS": "1", "REAX_NETWORK": "ws://127.0.0.1:9951",
           "REAX_STATE_PATH": str(tmp_path / "s.json"), "REAX_S1_FAST_RUNTIME_URL": "http://127.0.0.1:8101",
           "REAX_S1_PRO_RUNTIME_URL": "http://127.0.0.1:8102", "REAX_RUNTIME_TOKEN_FILE": str(tok),
           "REAX_MECHANISM": "s1"}
    env.update(extra)
    return env


def test_next_runtime_url_is_read_and_must_be_loopback(tmp_path):
    from reax_subnet.config import ConfigError

    cfg = load_validator(_env(tmp_path, REAX_S1_PRO_NEXT_RUNTIME_URL="http://127.0.0.1:8112"))
    assert cfg.common.next_runtime_url("s1-pro") == "http://127.0.0.1:8112"
    assert cfg.common.next_runtime_url("s1-fast") == ""
    assert cfg.common.runtime_url("s1-pro") == "http://127.0.0.1:8102"  # the old runtime stays as it was
    assert load_validator(_env(tmp_path)).common.next_runtime_url("s1-pro") == ""
    with pytest.raises(ConfigError):
        load_validator(_env(tmp_path, REAX_S1_PRO_NEXT_RUNTIME_URL="http://203.0.113.9:8112"))


@pytest.mark.real_switch
def test_next_backend_carries_the_new_manifest(tmp_path):
    from reax_subnet.cli import _s1_next_backend

    cfg = load_validator(_env(tmp_path, REAX_S1_PRO_NEXT_RUNTIME_URL="http://127.0.0.1:8112"))
    b = _s1_next_backend(cfg.common, "s1-pro")
    assert b.manifest.manifest_id == NEW.manifest_id and b.url == "http://127.0.0.1:8112"
    assert _s1_next_backend(cfg.common, "s1-fast") is None  # no switch for the other pools


@pytest.mark.real_switch
def test_engine_models_borrowing_s1pro_keep_the_release_manifest_after_the_switch(monkeypatch):
    """critic-post81 F1: quyet-1-large / jeffan-31b borrow the s1-pro manifest; the timed pool switch must not change it."""
    from datetime import datetime, timezone
    from reax_subnet import s1backends as B, manifest as M
    before = B._manifest_for("quyet-1-large").manifest_id
    after_t = datetime(2026, 10, 31, tzinfo=timezone.utc)
    real = M.load_manifest
    monkeypatch.setattr(B, "load_manifest", lambda m, now=None, **kw: real(m, after_t if now is None else now, **kw))
    assert B._manifest_for("quyet-1-large").manifest_id == before
    assert B._manifest_for("jeffan-31b").manifest_id == before
    assert B._manifest_for("s1-pro").manifest_id != before      # the pool itself does switch
