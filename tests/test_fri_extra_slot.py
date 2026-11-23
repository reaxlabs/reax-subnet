"""One-time extra generalist slot Fri 30 Oct 2026 16:30Z and its 2 h confirmation (Florian #31857, public in #92).

The same file runs on the worker (1.2.0rc11.post4.5.2) and the validator (1.2.0rc11.post8.4) branches: the pinned constants
below must hold on both, so the two sides agree on slots, due times and keep-alive windows."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from reax_subnet import competition_split as CS
from reax_subnet.spec import config as C, evaluator as E, machine as M, state as ST
from conftest import RELEASE_CROWN_SLOTS_UTC
from spec_helpers import commit, neuron


def _t(day, h, m=0):
    return datetime(2026, 10, day, h, m, tzinfo=timezone.utc).timestamp()


MON26, THU, FRI = _t(26, 14, 30), _t(29, 14, 30), _t(30, 16, 30)
MON = datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc).timestamp()
CUT = FRI - 6 * 3600   # Fri 10:30Z


@pytest.fixture(autouse=True)
def release_slots(monkeypatch):
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", RELEASE_CROWN_SLOTS_UTC)   # conftest empties it for the other tests


def sub(uid=2, committed=_t(29, 10), track="generalist", **kw):
    c = commit(track, f"miner{uid}/entry")
    n = neuron(uid, commitment=c)
    return ST.Submission(key=f"{n.hotkey}|{c}", uid=uid, hotkey=n.hotkey, coldkey=n.coldkey, track_id=track,
                         repo=f"miner{uid}/entry", revision="a" * 40, first_seen_block=10000 + uid,
                         first_seen_ts=committed, chain_commit_ts=committed, status=ST.PENDING_CONFIRM,
                         delta_checked=True, score=80, nonproc=80, draw1="first", **kw)


def evaluator(*subs):
    ev = object.__new__(E.SpecEvaluator)
    ts = ST.TrackState(track_id="generalist", submissions={s.key: s for s in subs}, draws={"first": None})
    ev.cfg = SimpleNamespace(confirm_delay_s=M.CONFIRM_DELAY_S, tracks=())
    ev.state = SimpleNamespace(tracks={"generalist": ts})
    return ev, ts


def first_pass(ev, ts, s, at):
    ev._set_confirm_due(ts, s, at, at)
    return s.confirm_after_ts


GEN = C.TrackDef("generalist", "generalist", _t(21, 13), end_ts=None)


# ------------------------------------------------------------------ constants (identical on worker and validator)
def test_pinned_constants():
    assert CS.EXTRA_CROWN_SLOTS_UTC == (datetime(2026, 10, 30, 16, 30, tzinfo=timezone.utc),)
    assert M.FAST_CONFIRM_SLOT_UTC == datetime(2026, 10, 30, 16, 30, tzinfo=timezone.utc)
    assert M.FAST_CONFIRM_FROM_UTC == datetime(2026, 10, 29, 14, 30, tzinfo=timezone.utc)
    assert M.FAST_CONFIRM_DELAY_S == 2 * 3600 and M.CONFIRM_DELAY_S == 12 * 3600
    assert CS.GENERALIST_COMMIT_CUTOFF_S == 6 * 3600 and CS.SLOT_K1_DUE_LEAD_S == 3600.0 and CS.SLOT_CLOCK_S == 12 * 3600.0
    assert M.FAST_CONFIRM_SLOT_UTC in CS.CROWN_SLOTS_UTC


def test_fri_slot_added_and_mon_thu_generator_unchanged():
    slots = CS.CROWN_SLOTS_UTC
    assert list(slots) == sorted(set(slots))
    gen = CS._crown_slots(datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc),
                          datetime(2027, 3, 31, 23, 59, tzinfo=timezone.utc))
    assert set(slots) - set(gen) == {datetime(2026, 10, 30, 16, 30, tzinfo=timezone.utc)} and set(gen) <= set(slots)
    assert [t.timestamp() for t in slots[:4]] == [MON26, THU, FRI, MON]


def test_thu_slot_behaves_as_before_and_fri_slot_is_a_full_slot():
    SE = E.SpecEvaluator
    assert SE._crown_slot(THU - 3600) == THU and SE._crown_slot(THU) == THU
    assert SE._crown_slot(THU + 1) == FRI and SE._crown_slot(FRI) == FRI and SE._crown_slot(FRI + 1) == MON
    assert SE._current_slot(THU + 1) == THU and SE._current_slot(THU + CS.SLOT_CLOCK_S - 1) == THU
    assert SE._current_slot(THU + CS.SLOT_CLOCK_S) is None and SE._current_slot(FRI - 1) is None   # no overlap
    assert SE._current_slot(FRI + 1) == FRI and SE._current_slot(FRI + CS.SLOT_CLOCK_S) is None
    # an entry committed before Thu 08:30Z still plays Thursday; one committed later plays Friday (not Monday)
    assert E._cutoff_slot(sub(committed=THU - 6 * 3600), THU) == THU
    assert E._cutoff_slot(sub(committed=THU - 6 * 3600 + 1), THU) == FRI


# ------------------------------------------------------------------ scoped confirmation delay
def test_first_pass_thu_1500_is_due_1700_and_joins_fri_by_k1a():
    s = sub()
    ev, ts = evaluator(s)
    assert first_pass(ev, ts, s, _t(29, 15)) == _t(29, 17)
    assert s.first_pass_ts == _t(29, 15) and E._first_pass_ts(s) == _t(29, 15) and E._keepalive_since(s) == _t(29, 15)
    assert ev._k1a_outstanding(ts, FRI + 1) == [s] and ev._k1a_slot(ts, s, FRI + 1) == FRI


def test_first_pass_fri_1400_is_due_1600_after_the_k1_lead_and_plays_monday():
    s = sub()
    ev, ts = evaluator(s)
    assert first_pass(ev, ts, s, _t(30, 14)) == _t(30, 16) > FRI - CS.SLOT_K1_DUE_LEAD_S
    assert ev._k1a_outstanding(ts, FRI + 1) == []          # does not hold the Fri slot
    assert ev._k1a_slot(ts, s, FRI + 60) is None            # a confirmation after the slot ...
    assert E.SpecEvaluator._crown_slot(FRI + 60) == MON     # ... is stored for Mon 2 Nov
    s.history.append({"kind": "crown_pending", "ts": FRI + 60, "slot": MON})
    assert E.SpecEvaluator._slot_of(s) == MON


@pytest.mark.parametrize("committed,fast", [(CUT, True), (CUT + 60, False)])
def test_commit_cutoff_fri_1030(committed, fast):
    s = sub(committed=committed)
    ev, ts = evaluator(s)
    at = committed + 1800
    assert first_pass(ev, ts, s, at) == at + (M.FAST_CONFIRM_DELAY_S if fast else M.CONFIRM_DELAY_S)
    assert E._cutoff_slot(s, FRI) == (FRI if fast else MON)
    assert s.first_pass_ts == (at if fast else None)       # 12 h: the state is written exactly as before


@pytest.mark.parametrize("at,track", [(_t(28, 20), "generalist"),     # Wed, before the Thu slot
                                      (THU - 1, "generalist"),
                                      (FRI, "generalist"),            # at the slot: the next slot is Monday
                                      (_t(31, 9), "generalist"),
                                      (_t(29, 15), "demand_forecasting")])
def test_outside_the_window_keeps_12h_and_the_old_state_shape(at, track):
    s = sub(committed=min(at, _t(29, 10)) - 3600, track=track)
    ev, ts = evaluator(s)
    assert first_pass(ev, ts, s, at) == at + M.CONFIRM_DELAY_S
    assert s.first_pass_ts is None and E._first_pass_ts(s) == at


def test_first_pass_wed_still_plays_thursday():
    s = sub(committed=_t(28, 10))
    ev, ts = evaluator(s)
    assert first_pass(ev, ts, s, _t(28, 20)) == _t(29, 8)
    assert ev._k1a_outstanding(ts, THU + 1) == [s] and ev._k1a_slot(ts, s, THU + 1) == THU


def test_confirm_delay_s_unit():
    f = M.confirm_delay_s
    assert f(_t(29, 15), FRI) == 7200 and f(THU, FRI) == 7200 and f(FRI - 1, FRI) == 7200
    assert f(THU - 1, FRI) == f(FRI, FRI) == f(_t(29, 15), MON) == f(_t(29, 15), None) == M.CONFIRM_DELAY_S
    assert f(_t(29, 15), FRI, default=0.0) == 0.0           # a shorter configured delay is never lengthened


def test_switch_off(monkeypatch):
    monkeypatch.setattr(M, "FAST_CONFIRM_SLOT_UTC", None)
    s = sub()
    ev, ts = evaluator(s)
    assert first_pass(ev, ts, s, _t(29, 15)) == _t(30, 3) and s.first_pass_ts is None
    assert M.confirm_delay_s(_t(29, 15), FRI) == M.CONFIRM_DELAY_S


# ------------------------------------------------------------------ entries that first passed before the release ran
def test_pull_in_once_for_window_entries_only():
    late = sub(2)                       # first pass Thu 16:00Z under the old release: due Fri 04:00Z, no first_pass_ts
    late.confirm_after_ts = _t(30, 4)
    wed = sub(3, committed=_t(28, 9))   # first pass Wed 20:00Z: untouched
    wed.confirm_after_ts = _t(29, 8)
    ran = sub(4)                        # its confirmation already ran: untouched
    ran.confirm_after_ts, ran.confirm_draw = _t(30, 4), "conf"
    held = sub(5, committed=CUT + 60)   # committed after the Fri cutoff: plays Monday, untouched
    held.confirm_after_ts = _t(30, 11) + M.CONFIRM_DELAY_S
    ev, ts = evaluator(late, wed, ran, held)
    ev._fast_confirm_pull_in([GEN], _t(29, 17))
    assert late.confirm_after_ts == _t(29, 18) and late.first_pass_ts == _t(29, 16)
    assert any("confirmation brought in for the Fri 30 Oct 16:30Z slot" in str(e) for e in ts.events)
    assert (wed.confirm_after_ts, wed.first_pass_ts) == (_t(29, 8), None)
    assert (ran.confirm_after_ts, ran.first_pass_ts) == (_t(30, 4), None)
    assert (held.confirm_after_ts, held.first_pass_ts) == (_t(30, 11) + M.CONFIRM_DELAY_S, None)
    n = len(ts.events)
    ev._fast_confirm_pull_in([GEN], _t(29, 17, 5))
    assert late.confirm_after_ts == _t(29, 18) and len(ts.events) == n     # idempotent


@pytest.mark.parametrize("now,due", [(_t(29, 19), _t(29, 19)), (THU - 60, None), (FRI, None)])
def test_pull_in_never_before_now_and_only_before_the_slot(now, due):
    s = sub()
    s.confirm_after_ts = _t(30, 4)      # first pass Thu 16:00Z
    ev, ts = evaluator(s)
    ev._fast_confirm_pull_in([GEN], now)
    assert s.confirm_after_ts == (due if due is not None else _t(30, 4))


# ------------------------------------------------------------------ validator keep-alive windows (H1)
@pytest.mark.skipif(not hasattr(E, "_decision_end"), reason="validator keep-alive (H1) lives on the validator branch")
def test_validator_pending_ka_runs_to_the_fri_slot():
    s = sub()
    ev, ts = evaluator(s)
    first_pass(ev, ts, s, _t(29, 15))
    assert E._decision_end(s) == FRI + CS.SLOT_CLOCK_S          # decided at Fri 16:30Z (not Thursday, not Monday)
    late = sub(3)
    first_pass(ev, ts, late, _t(30, 14))                         # due 16:00Z, after the K1 lead: Monday decides it
    assert E._decision_end(late) == MON + CS.SLOT_CLOCK_S
    wed = sub(4, committed=_t(28, 9))
    first_pass(ev, ts, wed, _t(28, 20))
    assert E._decision_end(wed) == THU + CS.SLOT_CLOCK_S         # unchanged
    s.history.append({"kind": "crown_pending", "ts": _t(30, 9), "slot": FRI})
    assert E._decision_end(s) == FRI + CS.SLOT_CLOCK_S
