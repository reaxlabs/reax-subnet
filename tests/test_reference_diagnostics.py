"""Offline, deterministic timeout provenance and unchanged reference failure isolation."""
import asyncio
import collections
import copy
import logging
from types import SimpleNamespace

import pytest

from reax_subnet import s1validator as sv
from reax_subnet.state import MinerState, ValidatorState


def setup(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(sv, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(sv, "PROBE_REFERENCE_CHUNK_GPU", 1)
    val = object.__new__(sv.S1Validator)
    val.pools = {m: sv.POOL_PARAMS[m] for m in ("s1-fast", "s1-pro")}
    val._ref_idle, val._ref_busy = {}, {}
    val.state = ValidatorState(9, "validator", miners={7: MinerState("hk", role="role", capacity=100.0)})
    val._probe_last, val._probe_holds = {}, {}
    val._probe_drop_reasons = collections.deque()
    val._probe_drops, val.last_probe_stats = 0, {}
    decisions = [SimpleNamespace(ident=str(i), question=SimpleNamespace(keys=("a", "b"))) for i in range(3)]

    async def healthy(ds):
        return [{"probabilities": {"a": 0.5, "b": 0.5}} for _ in ds]

    val.backends = {m: SimpleNamespace(decide_each=healthy) for m in val.pools}
    return val, clock, decisions


def diagnostics(caplog, model="s1-pro"):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(f"{model} reference diagnostic ")]


@pytest.mark.parametrize("mode,background,reason", [
    ("own", True, "own_time_budget"),
    ("outer_wait", True, "outer_wall_deadline"),
    ("outer_active", True, "outer_wall_deadline"),
    ("outer_partial", True, "outer_wall_deadline"),
    ("backend", True, "backend_timeout"),
    ("backend", False, "backend_timeout"),
    ("backend_error", True, "computation_error"),
    ("foreground", False, "outer_wall_deadline"),
])
async def test_timeout_provenance_and_unscored_probe(mode, background, reason, monkeypatch, caplog):
    val, clock, ds = setup(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="reax.s1validator")
    entered = asyncio.Event()
    cancelled = []
    calls = []

    async def backend(items):
        calls.append(len(items))
        if mode == "outer_partial" and len(calls) == 1:
            clock.now += 0.25
            return [{"probabilities": {"a": 0.5, "b": 0.5}} for _ in items]
        if mode == "backend_error":
            raise ValueError("PRIVATE payload must never enter diagnostics")
        if mode == "backend":
            clock.now += 2
            raise TimeoutError("PRIVATE payload must never enter diagnostics")
        if mode == "own":
            clock.now += 2
            return [{"probabilities": {"a": 0.5, "b": 0.5}} for _ in items]
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    val.backends["s1-pro"].decide_each = backend
    if mode == "outer_wait":
        val._idle("s1-pro").clear()

    if mode in ("outer_wait", "outer_active", "outer_partial", "foreground"):
        async def deadline(coro, timeout):
            assert timeout == (sv.PROBE_BACKGROUND_WALL_FACTOR if background else 1)
            task = asyncio.create_task(coro)
            await asyncio.sleep(0)  # healthy pool completes; pro reaches its suspension
            if task.done():
                return task.result()
            if mode != "outer_wait":
                assert entered.is_set()
            clock.now += 3
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            raise TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", deadline)

    verify = {7: ("s1-pro", ds), 8: ("s1-fast", ds[:1])}
    refs, unscored, failed = await val._references_isolated(verify, 1, background=background)
    assert set(refs) == {8} and unscored == {7} and failed == ["s1-pro"]
    line, = diagnostics(caplog)
    for part in (f"path={'background' if background else 'foreground'}", f"reason={reason}",
                 "slots=1 items=3", "wall_s=", "wall_budget_s=", "own_budget_s=",
                 "active_s=", "waiting_s=", "completed_chunks=", "completed_items="):
        assert part in line
    assert "PRIVATE" not in caplog.text
    if mode == "own":
        assert "active_s=4.0" in line and "completed_chunks=2 completed_items=2" in line
        assert f"own_budget_s={sv.PROBE_BACKGROUND_OWN_FACTOR * 1}" in line
    if mode == "outer_wait":
        assert "waiting_s=3.0" in line and "active_s=0.0" in line
    if mode == "outer_active":
        assert "active_s=3.0" in line and "completed_chunks=0 completed_items=0" in line
    if mode == "outer_partial":
        assert "active_s=3.25" in line and "completed_chunks=1 completed_items=1" in line
    assert cancelled == ([True] if mode in ("outer_active", "outer_partial", "foreground") else [])
    assert not val._ref_busy.get("s1-pro", 0)
    if not background:
        assert val._idle("s1-pro").is_set()

    # The real isolation result must still bypass strikes and capacity folding in the probe consumer.
    before = copy.deepcopy(val.state.miners[7])

    async def isolated(*args, **kwargs):
        return refs, unscored, failed

    val._references_isolated = isolated
    pending = sv._PendingProbe(probed=[(7, [], 2., 40)], checked={}, verify=verify,
        roles={7: "s1-pro"}, budgets={7: 160.}, scale=1., refused={}, left=1., ended=0.,
        owners={7: ("hk", "role")}, diagnostics={})
    assert await val._probe_verify(pending, background=background) == {}
    assert val.state.miners[7] == before
    assert val._probe_last == {} and val._probe_holds == {}
    assert sv._reference_progress.get() is None


@pytest.mark.parametrize("background", [False, True])
async def test_success_and_method_patch_compatibility(background, monkeypatch, caplog):
    val, clock, ds = setup(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="reax.s1validator")
    result = await val._references_isolated({7: ("s1-pro", ds)}, 1, background=background)
    assert result == ({7: {str(i): (0.5, 0.5) for i in range(3)}}, set(), [])
    line, = diagnostics(caplog)
    assert "reason=completed" in line
    assert ("completed_chunks=3 completed_items=3" if background else "completed_chunks=None") in line
    caplog.clear()

    async def patched(sub, budget):  # existing two-argument patches must keep working
        raise TimeoutError("PRIVATE")

    val._references = patched
    assert await val._references_isolated({7: ("s1-pro", ds)}, 1, background=background) == ({}, {7}, ["s1-pro"])
    assert "reason=backend_timeout" in diagnostics(caplog)[0]
    assert "PRIVATE" not in caplog.text


async def test_background_own_budget_covers_last_chunk_past_one_budget(monkeypatch, caplog):
    """Mainnet post8.11: 5 chunks cost a bit more than one reference budget and the last partial chunk was dropped."""
    val, clock, ds = setup(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="reax.s1validator")
    assert sv.PROBE_BACKGROUND_OWN_FACTOR == 2.0 and sv.PROBE_BACKGROUND_OWN_FACTOR < sv.PROBE_BACKGROUND_WALL_FACTOR

    async def backend(items):
        clock.now += 0.6
        return [{"probabilities": {"a": 0.5, "b": 0.5}} for _ in items]

    val.backends["s1-pro"].decide_each = backend
    refs, unscored, failed = await val._references_isolated({7: ("s1-pro", ds)}, 1, background=True)
    assert set(refs[7]) == {"0", "1", "2"} and unscored == set() and failed == []
    line, = diagnostics(caplog)
    assert "reason=completed" in line and "own_budget_s=2.0" in line and "wall_budget_s=4.000" in line
    assert "completed_chunks=3 completed_items=3" in line
    caplog.clear()
    refs, unscored, failed = await val._references_isolated({7: ("s1-pro", ds)}, 1, background=False)
    assert "own_budget_s=None" in diagnostics(caplog)[0]


@pytest.mark.parametrize("waiting", [False, True])
async def test_external_cancellation_propagates_with_progress(waiting, monkeypatch, caplog):
    val, clock, ds = setup(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="reax.s1validator")
    entered = asyncio.Event()
    cancelled = []

    async def backend(items):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    val.backends["s1-pro"].decide_each = backend
    if waiting:
        val._idle("s1-pro").clear()
    task = asyncio.create_task(val._references_isolated({7: ("s1-pro", ds)}, 100, background=True))
    if waiting:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    else:
        await entered.wait()
    clock.now += 4
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    line, = diagnostics(caplog)
    assert "reason=cancelled" in line
    assert ("waiting_s=4.0" if waiting else "active_s=4.0") in line
    assert cancelled == ([] if waiting else [True])
    assert sv._reference_progress.get() is None


@pytest.mark.parametrize("background", [False, True])
async def test_real_wait_for_deadline_cancels_backend(background, monkeypatch, caplog):
    val, clock, ds = setup(monkeypatch)
    cancelled = []

    async def backend(items):
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    val.backends["s1-pro"].decide_each = backend
    assert await val._references_isolated({7: ("s1-pro", ds)}, 0.001, background) == ({}, {7}, ["s1-pro"])
    assert cancelled == [True]
    assert "reason=outer_wall_deadline" in diagnostics(caplog)[0]
    # Operators' watchers key on the legacy timeout line; it must survive with the new provenance appended.
    legacy = [r.getMessage() for r in caplog.records if "s1-pro reference runtime did not finish 1 miners' items" in r.getMessage()]
    assert len(legacy) == 1 and legacy[0].endswith(f"path={'background' if background else 'foreground'} "
                                                   "reason=outer_wall_deadline")


async def test_yield_time_excluded_and_final_chunk_budget_semantics_unchanged(monkeypatch, caplog):
    val, clock, ds = setup(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="reax.s1validator")
    calls = []

    async def backend(items):
        calls.append(len(items))
        clock.now += 1.5
        if len(calls) == 1:
            ev = val._idle("s1-pro")
            ev.clear()

            def release_foreground():
                clock.now += 100
                ev.set()

            asyncio.get_running_loop().call_soon(release_foreground)
        return [{"probabilities": {"a": 0.5, "b": 0.5}} for _ in items]

    val.backends["s1-pro"].decide_each = backend
    # The final chunk takes own time above the 2 s own budget (2.0 x 1 s): the guard checks only before the NEXT chunk.
    refs, bad, failed = await val._references_isolated({7: ("s1-pro", ds[:2])}, 1, background=True)
    assert set(refs[7]) == {"0", "1"} and not bad and not failed and calls == [1, 1]
    line, = diagnostics(caplog)
    assert "active_s=3.0 waiting_s=100.0 completed_chunks=2 completed_items=2" in line and "own_budget_s=2.0" in line
