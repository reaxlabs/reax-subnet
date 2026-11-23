"""Owner allocations preserve protected payouts, count successful submissions only, and never stop weight setting."""
import json
import math
import random
from datetime import timedelta
from types import SimpleNamespace

import pytest

from reax_subnet import competition_split as CS, owner_alloc as OA
from test_emission_need import AFTER, BEFORE, GEN, OWNER, SERVING

HK = "5FLhkmTrKBUCJWnmCEY7xzcuVe6Mpk7FUgLdwfj7SbH33wo2"
N = [SimpleNamespace(uid=114, hotkey=HK)]


def row(**kw):
    return dict(hotkey=HK, share=.06, sets=3, start_block=100, reason="approved credit", **kw)


def write(tmp_path, rows=None):
    p = tmp_path / OA.FILE_NAME
    p.write_text(json.dumps([row()] if rows is None else rows))
    return str(p)


def mix(serving=SERVING, **kw):
    args = dict(owner_uid=OWNER, now=AFTER, inference=.4)
    args.update(kw)
    return CS.combine_weights(serving, 30, 1., GEN, **args)


@pytest.fixture(autouse=True)
def need_on(monkeypatch):
    monkeypatch.setattr(CS, "NEED_RULE", True)
    monkeypatch.setattr(OA, "_LATCHED", {})


FULL, PLAIN = {0: .9, 109: .04, 114: .06}, {0: .9, 109: .1}


def send(plan, success=True):
    """The validator's path: attach both vectors, begin (journal), settle."""
    plan.attach(FULL, PLAIN)
    sent, live = plan.begin(dict(FULL))
    if live is not None:
        live.finish(success)
    return sent


def journal(tmp_path):
    return json.loads((tmp_path / OA.STATE_NAME).read_text())


def test_absent_empty_and_null_byte_identical_over_fixtures(tmp_path):
    rng = random.Random(89)
    for _ in range(100):
        serving = {OWNER: 0., **{u: rng.random() for u in (10, 11, 12, 13)}}
        for now in (BEFORE, AFTER):
            expected = mix(serving, now=now)
            for path in (str(tmp_path / "missing"), write(tmp_path, [])):
                plan = OA.Plan(path, 100, N)
                assert mix(serving, now=now, owner_allocation=plan.shares) == expected
                assert repr(mix(serving, now=now, owner_allocation=None)) == repr(expected)
    assert not (tmp_path / OA.STATE_NAME).exists()


@pytest.mark.parametrize("burn", [0., .1])
def test_only_inference_reduced_competition_ka_burn_unchanged(burn):
    args = dict(serving={**SERVING, OWNER: burn}, burn_fraction=burn,
                legacy_keepalive=[41], provisional={"generalist": [42]}, engine_waiting=[43])
    before = mix(**args)
    after = mix(**args, owner_allocation={114: .06})
    assert after[114] == pytest.approx((1-burn)*.06)
    assert sum(before.get(u, 0)-after.get(u, 0) for u in (10,11,12,13)) == pytest.approx((1-burn)*.06)
    for u in (OWNER, 20, 21, 22, 30, 41, 42, 43):
        assert after.get(u, 0) == pytest.approx(before.get(u, 0))
    assert math.fsum(after.values()) == pytest.approx(1.)


def test_scaled_to_available_inference_and_zero_serving(caplog):
    before = mix(inference=.04)
    after = mix(inference=.04, owner_allocation={114:.06, 83:.06})
    assert after[114] == pytest.approx(.02)
    assert after[83] == pytest.approx(.02)
    assert sum(after.get(u,0) for u in (10,11,12,13)) == pytest.approx(0)
    assert after[OWNER] == pytest.approx(before[OWNER])
    assert "scaled" in caplog.text
    assert CS._need_weights({OWNER:.1}, {}, OWNER, .1, .4, 0., {114:.06}).get(114, 0) == 0


def test_countdown_restart_and_compute_does_not_consume(tmp_path):
    p = write(tmp_path)
    assert not OA.Plan(p, 99, N).shares
    for i in range(3):
        plan = OA.Plan(p, 100+i*360, N)
        assert plan.shares == {114:.06}
        assert OA.Plan(p, 100+i*360, N).shares == plan.shares  # recomputation/dry run
        assert send(plan) == FULL
        assert journal(tmp_path)["counts"][OA.row_id(row())] == i+1
        assert journal(tmp_path)["pending"] is None
    assert not OA.Plan(p, 1180, N).shares


def test_missing_uid_does_not_count(tmp_path, caplog):
    p = write(tmp_path)
    assert not OA.Plan(p, 100, []).shares
    assert not (tmp_path / OA.STATE_NAME).exists()
    assert "no uid" in caplog.text


@pytest.mark.parametrize("outcome", [False, None])
def test_failed_or_unknown_send_not_counted_flags_and_skips_until_cleared(tmp_path, caplog, outcome):
    p = write(tmp_path)
    assert send(OA.Plan(p, 100, N), outcome) == FULL  # the attempted vector carried the allocation
    st = journal(tmp_path)
    assert st["counts"] == {} and st["pending"] is None
    assert list(st["needs_reconcile"]) == [OA.row_id(row())] and st["needs_reconcile"][OA.row_id(row())]["block"] == 100
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "NEEDS-RECONCILE" in errors[0].message and "block 100" in errors[0].message
    for b in (460, 820):  # skipped (weights still computed and sent plain) until the operator clears the flag
        plan = OA.Plan(p, b, N)
        assert not plan.shares and send(plan) == FULL and plan.begin(dict(PLAIN)) == (PLAIN, None)
    assert journal(tmp_path)["counts"] == {}
    st["needs_reconcile"] = {}  # operator: chain receipt says no allocation landed; clear flag
    (tmp_path / OA.STATE_NAME).write_text(json.dumps(st))
    assert send(OA.Plan(p, 1180, N)) == FULL
    assert journal(tmp_path)["counts"] == {OA.row_id(row()): 1}


def test_restart_with_pending_flags_and_sends_plain(tmp_path, caplog):
    p = write(tmp_path)
    plan = OA.Plan(p, 100, N)
    plan.attach(FULL, PLAIN)
    assert plan.begin(dict(FULL))[0] == FULL  # crash: never finished
    assert journal(tmp_path)["pending"]["block"] == 100
    caplog.clear()
    after = OA.Plan(p, 460, N)  # restart
    assert not after.shares and after.begin(dict(PLAIN)) == (PLAIN, None)
    st = journal(tmp_path)
    assert st["pending"] is None and st["counts"] == {} and OA.row_id(row()) in st["needs_reconcile"]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "block 100" in errors[0].message and "restart" in errors[0].message
    assert not OA.Plan(p, 820, N).shares


def test_corrupt_journal_never_overwritten_allocation_off(tmp_path, caplog):
    p = write(tmp_path)
    (tmp_path / OA.STATE_NAME).write_text("garbage")
    for b in (460, 820):
        assert not OA.Plan(p, b, N).shares
    assert (tmp_path / OA.STATE_NAME).read_text() == "garbage"
    assert len([r for r in caplog.records if r.levelname == "ERROR"]) == 1


def test_read_only_approvals_dir_sends_plain_and_disables(tmp_path, caplog, monkeypatch):
    p = write(tmp_path)
    plan = OA.Plan(p, 100, N)
    assert plan.shares == {114: .06}  # computation is read-only
    def ro(*a, **k):
        raise OSError(30, "Read-only file system")
    monkeypatch.setattr(OA.tempfile, "mkstemp", ro)
    assert send(plan) == PLAIN
    assert not (tmp_path / OA.STATE_NAME).exists()
    assert OA.STATE_NAME in "".join(OA._LATCHED) and not OA.Plan(p, 460, N).shares
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "block 100" in errors[0].message


def test_plain_vector_is_exactly_the_no_allocation_vector(tmp_path):
    rng = random.Random(890)
    for _ in range(30):
        serving = {OWNER: 0., **{u: rng.random() for u in (10, 11, 12, 13)}}
        plan = OA.Plan(write(tmp_path), 100, N)
        full, plain = mix(serving, owner_allocation=plan.shares), mix(serving, owner_allocation=None)
        plan.attach(full, plain)
        assert repr(plain) == repr(mix(serving))
        assert full != plain and full[114] == pytest.approx(.06)


def test_no_effect_allocation_is_not_counted(tmp_path):
    p = write(tmp_path)
    plan = OA.Plan(p, 100, N)
    plan.attach(PLAIN, PLAIN)
    assert plan.begin(dict(PLAIN)) == (PLAIN, None)
    assert not (tmp_path / OA.STATE_NAME).exists()


@pytest.mark.parametrize("change", [dict(share=True), dict(share=float('nan')), dict(share=.101),dict(share=0),
    dict(sets=True),dict(sets=0),dict(sets=25),dict(start_block=True),dict(start_block=-1),dict(reason=" "),
    dict(reason="x"*301),dict(hotkey="no"),dict(extra=1)])
def test_malformed_whole_file_ignored(tmp_path, change, caplog):
    r = row(); r.update(change)
    assert not OA.Plan(write(tmp_path, [row(),r]), 100, N).shares
    assert "ignored" in caplog.text


def test_active_total_cap_and_future_rows(tmp_path):
    rows = [{**row(),"share":.09,"reason":str(i)} for i in range(3)]
    assert not OA.Plan(write(tmp_path, rows), 100, N).shares
    rows[-1]["start_block"] = 500
    assert OA.Plan(write(tmp_path, rows), 100, N).shares == {114:.18}
    assert not OA.Plan(write(tmp_path, rows), 500, N).shares


def test_duplicate_json_keys_ignored(tmp_path):
    p = tmp_path/OA.FILE_NAME
    p.write_text('[{"share":0.01,"share":0.10}]')
    assert not OA.load(str(p))


from test_weight_timing import arena  # reusable real Validator submission harness


class Engine:
    """Stand-in for EngineCompetition: a fresh Plan per weight computation."""
    def __init__(self, path):
        self.path, self.plan = path, None

    def weights(self, snap):
        self.plan = OA.Plan(self.path, snap.block, N)
        if not self.plan.shares:
            return dict(PLAIN)
        self.plan.attach(FULL, PLAIN)
        return dict(FULL)

    def allocation_plan(self):
        return self.plan


def sent(chain, i=-1):
    uids, ws, _ = chain.set_calls[i]
    return dict(zip(uids, ws))


async def test_submission_success_counts_exactly_once(arena, tmp_path):
    from test_weight_timing import snapshot
    chain, val = arena
    val.engine = eng = Engine(write(tmp_path))
    val.compute_weights = eng.weights
    chain.since_update = 10000
    assert await val.maybe_set_weights(snapshot())
    assert sent(chain) == FULL
    state = journal(tmp_path)
    assert state["counts"] == {OA.row_id(row()): 1} and state["pending"] is None and not state["needs_reconcile"]


async def test_send_exception_then_weights_still_sent_without_allocation(arena, tmp_path, caplog):
    from test_weight_timing import snapshot
    chain, val = arena
    val.engine = eng = Engine(write(tmp_path))
    val.compute_weights = eng.weights
    chain.since_update = 10000
    chain.fail_set_weights = True
    assert not await val.maybe_set_weights(snapshot())  # same as post8.8: retried next round
    state = journal(tmp_path)
    assert state["counts"] == {} and OA.row_id(row()) in state["needs_reconcile"]
    assert len([r for r in caplog.records if r.levelname == "ERROR"]) == 1
    chain.fail_set_weights = False
    assert await val.maybe_set_weights(snapshot())
    assert sent(chain) == PLAIN and journal(tmp_path)["counts"] == {}


async def test_restart_with_pending_weights_sent_without_allocation(arena, tmp_path):
    from test_weight_timing import snapshot
    chain, val = arena
    p = write(tmp_path)
    crashed = OA.Plan(p, 100, N)
    crashed.attach(FULL, PLAIN)
    crashed.begin(dict(FULL))  # process died during the send
    val.engine = eng = Engine(p)
    val.compute_weights = eng.weights
    chain.since_update = 10000
    assert await val.maybe_set_weights(snapshot())
    assert sent(chain) == PLAIN
    state = journal(tmp_path)
    assert state["counts"] == {} and state["pending"] is None and OA.row_id(row()) in state["needs_reconcile"]


async def test_read_only_approvals_dir_weights_sent_allocation_disabled(arena, tmp_path, caplog):
    from test_weight_timing import snapshot
    import os, stat
    chain, val = arena
    d = tmp_path / "approvals"
    d.mkdir()
    val.engine = eng = Engine(write(d))
    val.compute_weights = eng.weights
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(d, os.W_OK):
            pytest.skip("running as root: chmod cannot make the directory read-only")
        chain.since_update = 10000
        assert await val.maybe_set_weights(snapshot())
        assert sent(chain) == PLAIN
        assert not (d / OA.STATE_NAME).exists() and OA._LATCHED
        assert len([r for r in caplog.records if r.levelname == "ERROR"]) == 1
        assert await val.maybe_set_weights(snapshot())  # next set: allocation stays off, weights still go out
        assert sent(chain) == PLAIN and len(chain.set_calls) == 2
    finally:
        os.chmod(d, stat.S_IRWXU)


async def test_rate_limit_and_dry_run_never_journal(arena, tmp_path):
    from test_weight_timing import snapshot
    chain, val = arena
    val.engine = eng = Engine(write(tmp_path))
    val.compute_weights = eng.weights
    chain.since_update = 0
    assert not await val.maybe_set_weights(snapshot())
    chain.since_update = 10000
    object.__setattr__(val.cfg, "set_weights", False)
    assert not await val.maybe_set_weights(snapshot())
    assert not (tmp_path / OA.STATE_NAME).exists() and not chain.set_calls


def test_owner_recipient_file_refused_and_double_begin_never_double_counts(tmp_path):
    p = write(tmp_path)
    assert not OA.Plan(p, 100, N, owner_uid=114).shares
    plan = OA.Plan(p, 100, N)
    plan.attach(FULL, PLAIN)
    first, live = plan.begin(dict(FULL))
    assert first == FULL
    assert plan.begin(dict(FULL)) == (PLAIN, None)
    live.finish(True)
    live.finish(True)
    assert journal(tmp_path)["counts"] == {OA.row_id(row()): 1}


def test_removed_readded_and_reordered_rows_do_not_repeat(tmp_path):
    r = {**row(), "sets": 1}
    p = write(tmp_path, [r])
    send(OA.Plan(p, 100, N))
    write(tmp_path, [])
    assert not OA.Plan(p, 460, N).shares
    write(tmp_path, [r])
    assert not OA.Plan(p, 820, N).shares


def test_too_many_rows_bad_top_level_and_oversize(tmp_path):
    for raw in ({"rows": []}, [row()]*7, [row()]*2, None):
        assert not OA.Plan(write(tmp_path, raw) if raw is not None else write(tmp_path, {}), 100, N).shares
    p = tmp_path/OA.FILE_NAME
    p.write_text(' '* (OA.MAX_BYTES+1))
    assert not OA.load(str(p))


def test_plan_never_raises_into_weight_setting(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(OA, "load", boom)
    plan = OA.Plan(write(tmp_path), 100, N)
    assert not plan.shares and plan.begin(dict(PLAIN)) == (PLAIN, None)


def test_remote_fallback_vector_carries_no_allocation_plan(tmp_path, monkeypatch):
    from reax_subnet.remote import RemoteEngineCompetition
    from reax_subnet.engvalidator import EngineCompetition
    eng = RemoteEngineCompetition(str(tmp_path/"remote.json"))
    monkeypatch.setattr(eng, "engine_active", lambda: False)
    def fails(self, *args, **kw):
        self._owner_alloc_plan = OA.Plan(write(tmp_path), 100, N)
        raise RuntimeError("payment code")
    monkeypatch.setattr(EngineCompetition, "weights", fails)
    out = eng.weights({OWNER: 1.}, SimpleNamespace(owner_uid=OWNER), set())
    assert out and eng.allocation_plan() is None


def test_engine_plain_vector_equals_vector_without_file_and_full_carries_allocation(tmp_path, monkeypatch):
    """Real EngineCompetition: with a file, the returned vector carries the allocation and the plan's plain
    vector is exactly what the validator computes with no file (post8.8)."""
    import test_ka_override as K
    for k in list(CS.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, k, None)
    now = [max(K.AFTER, CS.NEED_START_UTC + timedelta(hours=1))]
    assert CS.need_active(now[0])
    eng, adapter = K._system_with_file(tmp_path, now, None)
    (tmp_path / "b").mkdir()
    base_eng, base_adapter = K._system_with_file(tmp_path / "b", now, None)
    s = K._snap_with_new()
    hk = next(n.hotkey for n in s.neurons if n.uid == K.NEW)
    write(tmp_path, [{**row(), "hotkey": hk, "start_block": 0}])
    full = K.pay(eng, adapter, s)
    plan = eng.allocation_plan()
    expected = K.pay(base_eng, base_adapter, K._snap_with_new())
    assert base_eng.allocation_plan() is None or not base_eng.allocation_plan().shares
    assert plan is not None and plan.shares == {K.NEW: .06}
    assert plan.plain == expected and plan.full == full != expected
    assert full[K.NEW] > expected.get(K.NEW, 0.0)


def test_engine_allocation_math_error_returns_exactly_the_plain_vector(tmp_path, monkeypatch, caplog):
    """Any exception in the allocation-bearing computation: the post8.8 vector, no plan, nothing journalled."""
    import test_ka_override as K
    from reax_subnet import engvalidator as EV
    for k in list(CS.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, k, None)
    now = [max(K.AFTER, CS.NEED_START_UTC + timedelta(hours=1))]
    eng, adapter = K._system_with_file(tmp_path, now, None)
    (tmp_path / "b").mkdir()
    base_eng, base_adapter = K._system_with_file(tmp_path / "b", now, None)
    s = K._snap_with_new()
    hk = next(n.hotkey for n in s.neurons if n.uid == K.NEW)
    write(tmp_path, [{**row(), "hotkey": hk, "start_block": 0}])
    real = EV.combine_weights

    def broken(*a, owner_allocation=None, **kw):
        if owner_allocation:
            raise ZeroDivisionError("allocation math")
        return real(*a, owner_allocation=owner_allocation, **kw)
    monkeypatch.setattr(EV, "combine_weights", broken)
    got = K.pay(eng, adapter, s)
    assert got == K.pay(base_eng, base_adapter, K._snap_with_new())
    assert eng.allocation_plan() is None
    assert not (tmp_path / OA.STATE_NAME).exists()
    assert "weights without allocation" in caplog.text
