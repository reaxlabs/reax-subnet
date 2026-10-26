"""Probe verification off the round cadence (1.2.0rc11.post6, REAX_PROBE_BACKGROUND).

On mainnet a probe's s1-cpu references (~2 per slot for ~162 slots on the single-lock llama.cpp runtime) held every probe
round at ~340 s instead of ~120 s. The probe's reference verification and capacity fold now run in the background while
the next rounds go on; the rounds' own references go first. These tests run today's slot counts (162 s1-cpu + 148
s1-lite, the release REFERENCE_CAPs) against fake single-lock reference runtimes; the miners' answers and the probe
window are stubbed (no sockets), everything from the window's result on is the validator's real code.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
import time
from dataclasses import replace

import pytest

from reax_subnet import probeshard, s1validator
from reax_subnet.chain import Neuron
from reax_subnet.config import load_validator
from reax_subnet.elements import slot_key
from reax_subnet.manifest import commitment_text, load_manifest
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1contract import normalize_answer
from reax_subnet.s1validator import REFERENCE_CAP, S1Validator
from reax_subnet.validator import QueryResult

DEV = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1"}
N_CPU, N_LITE = 162, 148
DELAY = {"s1-cpu": 0.02, "s1-lite": 0.002, "s1-fast": 0.0}  # seconds per decision on the fake runtime


CALLER = contextvars.ContextVar("caller", default="probe")  # "round" inside S1Validator._decide_foreground


class SingleLockRef(S1HashBackend):
    """A reference runtime that evaluates one decision at a time (gguf_server: asyncio LOCK). ``log`` records which
    caller each decision came from, in the order the runtime evaluated them."""

    def __init__(self, model: str, delay: float):
        super().__init__(model, salt="reference")
        self.delay = delay
        self.lock = asyncio.Lock()
        self.log: list[str] = []

    async def decide_each(self, decisions, retries=1):
        tag = CALLER.get()

        async def one(d):
            async with self.lock:
                await asyncio.sleep(self.delay)
                self.log.append(tag)
                return (await self.decide([d]))[0]

        return list(await asyncio.gather(*(one(d) for d in decisions)))


def honest(ref: S1HashBackend, decisions) -> dict[str, tuple[float, ...]]:
    """What an honest miner answers (S1HashBackend.decide without the event loop)."""
    out = {}
    for d in decisions:
        z = ref.logits(d)
        m = max(z)
        e = [math.exp((x - m) / ref.manifest.temperature) for x in z]
        s = math.fsum(e)
        probs = {k: v / s for k, v in zip(d.question.keys, e)}
        raw = {"type": "noul", "noul": probs["true"]} if d.question.kind == "noul" else \
            {"type": d.question.kind, "probabilities": probs}
        a = normalize_answer(d.question, raw)
        out[d.ident] = tuple(float(a["probabilities"][k]) for k in d.question.keys)
    return out


def build(make_wallet, fake_chain_cls, tmp_path, monkeypatch, background: bool, n_cpu=N_CPU, n_lite=N_LITE):
    monkeypatch.setenv("REAX_PROBE_BACKGROUND", "1" if background else "0")
    monkeypatch.delenv("REAX_REFERENCE_CAP_S1_CPU", raising=False)
    monkeypatch.delenv("REAX_REFERENCE_CAP_S1_LITE", raising=False)
    owner, vwallet = make_wallet("//Owner"), make_wallet("//Validator")
    neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
               Neuron(1, vwallet.hotkey.ss58_address, "c1", True, 5000.0, None, None)]
    uid = 2
    for model, n in (("s1-cpu", n_cpu), ("s1-lite", n_lite)):
        man = load_manifest(model)
        for _ in range(n):
            neurons.append(Neuron(uid, f"hk-{uid}", f"ck-{uid}", False, 0.0, "127.0.0.1", 9,
                                  commitment=commitment_text(model, man.manifest_id)))
            uid += 1
    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
    cfg = load_validator({**DEV, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_STATE_PATH": str(tmp_path / f"state-{background}.json"),
                          "REAX_EMA_ALPHA": "0.5", "REAX_BURN_FRACTION": "0.0", "REAX_WARMUP_ROUNDS": "0",
                          "REAX_FEATURE_TIERED_POOLS": "1", "REAX_FEATURE_S1_PRO_POOL": "0"})
    cfg = replace(cfg, round_interval_s=7)
    refs = {m: SingleLockRef(m, DELAY[m]) for m in ("s1-fast", "s1-lite", "s1-cpu")}
    val = S1Validator(cfg, vwallet, chain, refs)
    truth = {m: S1HashBackend(m, salt="reference") for m in refs}

    async def query(session, sem, n, req, timeout):  # every miner answers honestly at once
        return QueryResult(n.uid, honest(truth[req.model], req.decisions), 0.0, None)

    def check_s1(n, req, raw, headers, max_age):  # the probe's answers: honest, signature not under test here
        return honest(truth[req.model], req.decisions)

    async def window(targets, roles, prepared, ready=None, go=None, lanes=None):
        start = time.monotonic()
        results = []
        for n in targets:
            reqs = prepared[n.uid]
            k = 1 + n.uid % min(4, len(reqs))  # some slots answer more of their budget than others
            done = [(i, b"", {}) for i in range(k)]
            secs = s1validator.CAPACITY[roles[n.uid]].window_s
            results.append((n.uid, done, secs, sum(reqs[i][0] for i in range(k)), start, k))
        return probeshard.ProbeWindow(results=results, started=start)

    fg = val._decide_foreground

    async def foreground(model, decisions):
        token = CALLER.set("round")
        try:
            return await fg(model, decisions)
        finally:
            CALLER.reset(token)

    monkeypatch.setattr(val, "_decide_foreground", foreground)
    monkeypatch.setattr(val, "_query_s1", query)
    monkeypatch.setattr(val, "_check_s1", check_s1)
    monkeypatch.setattr(val, "_probe_window", window)
    monkeypatch.setattr(probeshard, "shard_count", lambda lanes_total, uids, env=None: 1)
    # Deterministic sample (the real one is secret and random): both modes verify the same items.
    monkeypatch.setattr(s1validator, "choose_verified", lambda ids, k: sorted(ids)[:k])
    slots = {slot_key(n.uid, model): model for n in neurons[2:] for model in ("s1-cpu", "s1-lite")
             if f":{model}:" in n.commitment}
    return val, chain, refs, slots


async def timed_round(val):
    t0 = time.monotonic()
    summary = await val.run_round(None)
    return summary, time.monotonic() - t0


def test_mainnet_slot_counts_exceed_the_reference_caps():
    """The case this release fixes: more small-pool slots than half the pool's reference cap (2 per slot floor)."""
    assert 2 * N_CPU > REFERENCE_CAP["s1-cpu"] and 2 * N_LITE > REFERENCE_CAP["s1-lite"]


async def test_probe_round_returns_within_the_interval_and_the_later_fold_equals_the_synchronous_one(
        make_wallet, fake_chain_cls, tmp_path, monkeypatch):
    # Synchronous probe (rc11.post5 behaviour, REAX_PROBE_BACKGROUND=0) as the reference result.
    sval, _c, srefs, slots = build(make_wallet, fake_chain_cls, tmp_path, monkeypatch, background=False)
    await sval.run_round(None)  # confirms the roles
    srefs["s1-cpu"].log.clear()
    ssum, sync_elapsed = await timed_round(sval)  # first answered round: every slot is unmeasured -> probe
    assert "capacity" in ssum and len(ssum["capacity"]) == N_CPU + N_LITE
    sync_cap = {s: sval.state.miners[s].capacity for s in slots}
    # The round verifies 160 s1-cpu decisions (its cap), the probe 2 per s1-cpu slot (the floor): 320 + 160.
    assert len(srefs["s1-cpu"].log) == REFERENCE_CAP["s1-cpu"] + 2 * N_CPU
    assert sync_elapsed > sval.cfg.round_interval_s  # the probe's references held the round

    val, _c, refs, _s = build(make_wallet, fake_chain_cls, tmp_path, monkeypatch, background=True)
    await val.run_round(None)
    for r in refs.values():
        r.log.clear()
    summary, elapsed = await timed_round(val)
    assert elapsed < val.cfg.round_interval_s + 0.5
    assert "capacity" not in summary and summary["capacity_pending"] == N_CPU + N_LITE
    assert val._probe_task is not None and not val._probe_task.done()  # references still pending
    assert all(val.state.miners[s].capacity is None for s in slots)  # nothing folded yet

    # The next round runs while the probe's references are pending: its own references go first (it waits at most one
    # chunk per runtime), it stays within the interval, and no new probe starts while one is being verified.
    monkeypatch.setattr(val, "_probe_due", lambda: True)
    for r in refs.values():
        r.log.clear()
    summary2, elapsed2 = await timed_round(val)
    assert elapsed2 < val.cfg.round_interval_s
    assert "capacity_pending" not in summary2 and val._probe_owed
    cpu_log = refs["s1-cpu"].log
    first, last = cpu_log.index("round"), len(cpu_log) - 1 - cpu_log[::-1].index("round")
    assert cpu_log.count("round") == REFERENCE_CAP["s1-cpu"]
    # Between the round's first and last reference only probe decisions already in flight (one chunk) ran.
    assert cpu_log[first:last + 1].count("probe") <= s1validator.PROBE_REFERENCE_CHUNK
    assert not val._probe_task.done()  # still verifying in the gaps

    await asyncio.wait_for(val._probe_task, 30)
    probe_cpu = cpu_log.count("probe")
    cap = {s: val.state.miners[s].capacity for s in slots}
    assert cap == sync_cap  # identical fold, only later
    assert all(c is not None and c > 0 for c in cap.values())
    assert val._probe_scale == sval._probe_scale and val._probe_holds == sval._probe_holds
    assert val._probe_last == sval._probe_last
    assert probe_cpu <= 2 * N_CPU  # (the rest ran during the probe round itself, before the log was cleared)

    # The report reaches the next round's summary (EU ledger, dashboard); the owed probe runs now.
    summary3 = await val.run_round(None)
    assert summary3["capacity"] == ssum["capacity"]
    assert summary3["capacity_pending"] == N_CPU + N_LITE and not val._probe_owed
    val._probe_task.cancel()


async def test_background_probe_sends_the_same_references_as_the_synchronous_one(make_wallet, fake_chain_cls,
                                                                                  tmp_path, monkeypatch):
    val, _c, refs, _s = build(make_wallet, fake_chain_cls, tmp_path, monkeypatch, background=True)
    await val.run_round(None)
    for r in refs.values():
        r.log.clear()
    await val.run_round(None)
    await asyncio.wait_for(val._probe_task, 30)
    assert refs["s1-cpu"].log.count("probe") == 2 * N_CPU
    assert refs["s1-lite"].log.count("probe") == 2 * N_LITE
    assert refs["s1-cpu"].log.count("round") == REFERENCE_CAP["s1-cpu"]


async def test_a_slot_whose_hotkey_changed_before_the_fold_is_not_folded(make_wallet, fake_chain_cls, tmp_path,
                                                                         monkeypatch):
    val, chain, refs, slots = build(make_wallet, fake_chain_cls, tmp_path, monkeypatch, background=True,
                                    n_cpu=6, n_lite=4)
    refs["s1-cpu"].delay = 0.3  # the probe's 12 s1-cpu references take ~3.6 s
    await val.run_round(None)
    summary = await val.run_round(None)
    assert summary["capacity_pending"] == 10 and not val._probe_task.done()
    cpu = sorted(s for s, m in slots.items() if m == "s1-cpu")
    victim = cpu[0]
    uid = victim % s1validator.SLOT_STRIDE
    chain.neurons = [n if n.uid != uid else replace(n, hotkey="hk-new-owner") for n in chain.neurons]
    monkeypatch.setattr(val, "_probe_due", lambda: False)
    await val.run_round(None)  # the new owner's slot is reset; the probe is still verifying
    assert not val._probe_task.done()
    await asyncio.wait_for(val._probe_task, 30)
    assert val.state.miners[victim].hotkey == "hk-new-owner" and val.state.miners[victim].capacity is None
    assert all(val.state.miners[s].capacity is not None for s in slots if s != victim)
    assert victim not in val._probe_report
    # Every folded entry names the hotkey that was checked at the window (the EU ledger attributes by it).
    assert val._probe_report and all(e["hotkey"] == val.state.miners[s].hotkey for s, e in val._probe_report.items())


async def test_synchronous_mode_keeps_the_old_behaviour(make_wallet, fake_chain_cls, tmp_path, monkeypatch):
    val, _c, _r, slots = build(make_wallet, fake_chain_cls, tmp_path, monkeypatch, background=False, n_cpu=3, n_lite=3)
    assert not val.probe_background
    await val.run_round(None)
    summary = await val.run_round(None)
    assert len(summary["capacity"]) == 6 and "capacity_pending" not in summary and val._probe_task is None
    assert all(val.state.miners[s].capacity is not None for s in slots)


async def test_background_references_count_only_their_own_time(make_wallet, fake_chain_cls, tmp_path, monkeypatch):
    """Waiting for a round's references is not charged to the probe's reference budget, but the runtime's own time is:
    a probe whose chunks alone exceed the budget stops like a synchronous probe that timed out."""
    val, _c, refs, _s = build(make_wallet, fake_chain_cls, tmp_path, monkeypatch, background=True, n_cpu=1, n_lite=0)
    model = "s1-cpu"
    ds = [s1validator.parse_decision(it.decision(), model) for it in s1validator.generate_s1_items(40)]
    refs[model].delay = 0.01

    async def hold_round():  # a round's references occupy the runtime for 0.5 s
        val._ref_busy[model] = 1
        val._idle(model).clear()
        await asyncio.sleep(0.5)
        val._ref_busy[model] = 0
        val._idle(model).set()

    task = asyncio.create_task(hold_round())
    await asyncio.sleep(0)
    t0 = time.monotonic()
    out = await val._decide_background(model, ds, budget=0.45)  # own time ~0.4 s, wall ~0.9 s
    assert time.monotonic() - t0 >= 0.85 and len(out) == 40
    await task
    with pytest.raises(asyncio.TimeoutError):
        await val._decide_background(model, ds, budget=0.1)
