"""Probe diagnostics observe existing zero/fold/drop decisions without changing them."""
from __future__ import annotations

import collections
import logging
import math
import time

import pytest

from reax_subnet import auth, probeshard, s1validator as sv
from reax_subnet.chain import Neuron
from reax_subnet.state import MinerState, ValidatorState
from test_capv2_fixes import _generator


def records(caplog):
    return [r.getMessage() for r in caplog.records if "probe diagnostic " in r.getMessage()]


@pytest.mark.parametrize("kind,reason,folded,zero", [
    ("quality", "quality_gate", True, True),
    ("none", "no_answers", True, True),
    ("empty", "empty_reference", True, True),
    ("good", "kept", True, False),
    ("strike", "strike", True, True),
    ("unscored", "reference_unscored", False, False),
    ("changed", "hotkey_or_role_changed", False, False),
])
async def test_fold_diagnostics_preserve_scoring(kind, reason, folded, zero, caplog):
    val = object.__new__(sv.S1Validator)
    val.pools = {"s1-fast": sv.POOL_PARAMS["s1-fast"]}
    val.state = ValidatorState(9, "validator", miners={7: MinerState("hk", role="role", capacity=100.0)})
    val._probe_last, val._probe_holds = {}, {}
    val._probe_drop_reasons = collections.deque()
    val._probe_drops, val.last_probe_stats = 0, {}
    params = val.pools["s1-fast"]
    distance = (params.tau_mean + params.tau_strike) / 2 if kind == "quality" else (
        params.tau_item + 1 if kind == "strike" else 0)
    p = 1 / (1 + math.exp(-2 * distance / params.temperature))
    got = ({"x": (p, 1 - p)}, {})
    refs = {} if kind == "empty" else {7: {"x": (0.5, 0.5)}}

    async def references(*a, **kw):
        return refs, {7} if kind == "unscored" else set(), set()

    val._references_isolated = references
    pending = sv._PendingProbe(
        probed=[(7, [] if kind == "none" else [(0, b"", {})], 2.0, 0 if kind == "none" else 40)],
        checked={} if kind == "none" else {7: got}, verify={}, roles={7: "s1-fast"}, budgets={7: 160.0},
        scale=1.0, refused={7: 1}, left=10, ended=time.monotonic(), owners={7: ("hk", "role")},
        diagnostics={7: {"received_requests": 0 if kind == "none" else 1, "expected_requests": 10,
                         "answers_received": 0 if kind == "none" else 1, "expected_answers": 40,
                         "invalid_answers": 0, "invalid_signatures": 0, "stale_signatures": 0,
                         "start_missing": False, "lane_errors": {"TimeoutError": 2}}})
    if kind == "changed":
        val.state.miners[7].hotkey = "new-hotkey"
    caplog.set_level(logging.INFO, logger="reax.s1validator")
    report = await val._probe_verify(pending)
    lines = records(caplog)
    assert len(lines) == 1
    line = lines[0]
    for expected in (f"reason={reason}", f"folded={folded}", "tau_mean=", "tau_strike=", "tau_item=",
                     "mean_distance=", "max_distance=", "reference_count=", "start_missing=False",
                     "lane_errors={'TimeoutError': 2}", "invalid_signatures=0", "stale_signatures=0",
                     "answers_received=", "received_requests="):
        assert expected in line
    if folded:
        expected_dps = 0.0 if zero else 20.0
        assert report[7]["dps"] == expected_dps
        expected_capacity, _ = sv.fold_capacity_bounded(0 if kind == "strike" else 100, expected_dps,
                                                       160, 1, sv.CAPACITY["s1-fast"].alpha, 0)
        assert val.state.miners[7].capacity == expected_capacity
        assert val.state.miners[7].strikes == (kind == "strike")
    else:
        assert report == {} and val.state.miners[7].capacity == 100
    if kind == "empty":
        assert "empty_reference=True" in line and "mean_distance=inf" in line
    if kind == "quality":
        assert "mean_distance=None" not in line and "max_distance=None" not in line


@pytest.mark.parametrize("mode", ["dropped", "missing_result", "missing_start", "invalid", "stale"])
async def test_measure_diagnostics(caplog, monkeypatch, make_wallet, fake_chain_cls, tmp_path, mode):
    val = _generator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    n = Neuron(7, "hk", "ck", False, 0, "127.0.0.1", 9)
    val.state.miners[7] = MinerState("hk", role="role", capacity=100)
    monkeypatch.setattr(probeshard, "shard_count", lambda *a: 1)

    async def window(*a, **kw):
        return probeshard.ProbeWindow(
            results=[] if mode == "missing_result" else [(7, [(0, b"", {})], 2., 4, None if mode == "missing_start" else time.monotonic(), 1)],
            started=time.monotonic(), failed=["shard failed"] if mode in ("dropped", "missing_result") else [],
            lane_errors={7: {"http_503": 2}}, refused={7: 2})

    def check(n, req, raw, headers, max_age):
        if mode in ("invalid", "stale"):
            raise (auth.http_auth.StaleRequest("old") if mode == "stale" else auth.http_auth.BadSignature("bad"))
        return {d.ident: tuple(1 / len(d.question.keys) for _ in d.question.keys) for d in req.decisions}

    async def references(verify, *a, **kw):
        return {u: {d.ident: tuple(1 / len(d.question.keys) for _ in d.question.keys) for d in ds}
                for u, (_model, ds) in verify.items()}, set(), set()

    monkeypatch.setattr(val, "_probe_window", window)
    monkeypatch.setattr(val, "_check_s1", check)
    monkeypatch.setattr(val, "_references_isolated", references)
    caplog.set_level(logging.INFO)
    report = await val._probe_capacity(None, [n], {7: "s1-fast"})
    lines = records(caplog)
    assert len(lines) == 1
    line = lines[0]
    if mode in ("dropped", "missing_result"):
        assert report == {} and val.state.miners[7].capacity == 100
        assert "folded=False reason=validator_drop: shard failed" in line
        assert "mean_distance=None" in line and "reference_count=None" in line
        assert "invalid_signatures=None" in line  # discarded probe: no extra signature checks
        assert ("received_requests=None/" if mode == "missing_result" else "received_requests=1/") in line
    elif mode == "missing_start":
        # Observe the defensive gap without adding a new gate: the pre-existing behaviour still folds it.
        assert "start_missing=True" in line and "folded=True reason=kept" in line
        assert report[7]["dps"] == 2
    else:
        assert "folded=True reason=invalid_answer" in line and report[7]["dps"] == 0
        assert "invalid_signatures=1" in line
        assert f"stale_signatures={int(mode == 'stale')}" in line
    assert "tau_mean=" in line and "answers_received=" in line


def test_shard_lane_errors_survive_merge():
    merged = sv._merge_windows([probeshard.ProbeWindow([], lane_errors={7: {"TimeoutError": 3}}),
                                probeshard.ProbeWindow([], lane_errors={8: {"http_503": 2}})], 2)
    assert merged.lane_errors == {7: {"TimeoutError": 3}, 8: {"http_503": 2}}
