"""Validator-only waiting keep-alive from existing worker restart history."""

import math

import pytest
from test_keepalive import _sub
from test_provisional_ka import _snapr
from test_waiting_ka_hotfix import H, THU, release_slots  # noqa: F401

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import review as RV
from reax_subnet.spec import state as ST

TRACK = "generalist"
pytestmark = pytest.mark.usefixtures("rc5_entry_floors", "release_slots")


def _restart(st, kind="slot_restart", uid=1, coldkey=None):
    s = _sub(st, uid, ST.PENDING_CONFIRM, 70.0, track=TRACK, coldkey=coldkey)
    s.hash_named = f"hash{uid}"
    s.first_seen_ts = THU - 1000 * H  # old commitment window is exhausted
    s.draw1, s.first_pass_ts = "old-draw", THU - 500 * H
    if kind == "slot_restart":
        E.SpecEvaluator._slot_restart(st.track(TRACK), s, THU, "new champion")
    else:
        E.SpecEvaluator._v5_restart(st.track(TRACK), s, THU, "v5", "v5")
    return s


def _paths(st, subs, now=THU):
    snap = _snapr(subs)
    return (E.compute_provisional(st, [TRACK], snap, now=now)[TRACK],
            E.compute_keepalive(st, [TRACK], snap, {TRACK: 0.0}, now=now)[TRACK])


@pytest.mark.parametrize("kind", ["slot_restart", "v5_restart"])
def test_restart_transfers_between_waiting_pending_and_decided(kind):
    st = ST.SpecState()
    s = _restart(st, kind)
    assert s.score == 70.0 and E._our_wait(s)
    assert _paths(st, [s]) == ([1], [])
    s.status, s.draw1, s.first_pass_ts = ST.PENDING_CONFIRM, "new-draw", THU + H
    s.confirm_after_ts = THU + 13 * H
    s.history.append({"kind": "first", "draw": s.draw1, "ts": THU + H})
    prov, pending = _paths(st, [s], THU + H)
    assert prov == [] and [u for u, _ in pending] == [1]
    s.status = ST.NOT_BETTER
    assert _paths(st, [s], THU + 2 * H) == ([], [])


@pytest.mark.parametrize("kind", ["slot_restart", "v5_restart"])
def test_restart_window_is_96_hours_from_restart(kind):
    st = ST.SpecState()
    s = _restart(st, kind)
    assert _paths(st, [s], THU + 96 * H)[0] == [1]
    assert _paths(st, [s], THU + 96 * H + 1) == ([], [])


@pytest.mark.parametrize("mutate", [
    lambda s: setattr(s, "draw1", "new-draw"),
    lambda s: s.history.append({"kind": "first", "draw": "new-draw", "ts": THU}),
    lambda s: s.history[-1].update(kind="requeue"),
    lambda s: setattr(s, "status", ST.NOT_BETTER),
    lambda s: setattr(s, "status", ST.REJECTED),
    lambda s: setattr(s, "status", ST.PENDING_VERIFY),
    lambda s: s.history.insert(0, {"kind": RV.FLAG_KIND, "flags": ["reparametrised"]}),
])
def test_restart_requires_latest_history_no_draw_and_checks(mutate):
    st = ST.SpecState()
    s = _restart(st)
    mutate(s)
    assert _paths(st, [s]) == ([], [])


def test_operator_hold_excludes_restart(monkeypatch):
    st = ST.SpecState()
    s = _restart(st)
    monkeypatch.setenv(RV.HOLD_ENV, f"{s.hotkey}@{s.revision}")
    assert _paths(st, [s]) == ([], [])


@pytest.mark.parametrize("at", [None, True, "bad", float("nan"), float("inf"), THU + E.KEEPALIVE_FUTURE_S + 1])
def test_invalid_restart_time_fails_closed(at):
    st = ST.SpecState()
    s = _restart(st)
    s.history[-1]["ts"] = at
    assert _paths(st, [s]) == ([], [])


@pytest.mark.parametrize("void", [False, True])
def test_restart_does_not_bypass_reveal_deadline(void):
    st = ST.SpecState()
    s = _restart(st)
    s.nonpublic, s.reveal_void, s.reveal_due_ts = True, void, THU - 1
    assert not ST.reveal_allows_keepalive(s, THU)
    assert _paths(st, [s]) == ([], [])


def test_unrevealed_restart_keeps_existing_pre_deadline_rule():
    st = ST.SpecState()
    s = _restart(st)
    s.nonpublic, s.reveal_due_ts = True, THU + H
    assert _paths(st, [s])[0] == [1]
    assert _paths(st, [s], THU + H + 1) == ([], [])


def test_nonrestart_close_call_keeps_original_commitment_window():
    st = ST.SpecState()
    s = _restart(st)
    s.history[-1]["kind"] = "close_draws"
    s.close_draws = ["old-draw"]
    assert E._our_wait(s)
    assert _paths(st, [s]) == ([], [])  # no restart window for an ordinary close call
    s.first_seen_ts = THU - H
    assert _paths(st, [s])[0] == [1]


def test_many_restarts_keep_order_coldkey_limits_and_share_cap(monkeypatch):
    st = ST.SpecState()
    subs = [_restart(st, uid=u, coldkey=f"pair{(u - 1) // 2}") for u in range(1, 19)]
    # Checked restarts use our-wait classification even with an unverified cap of zero.
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 0)
    prov, pending = _paths(st, subs)
    assert prov == list(range(1, 19, 2)) + list(range(2, 19, 2)) and pending == []
    weights = CS.combine_weights({100: 1.0}, None, 0.0, {TRACK: []}, provisional={TRACK: prov})
    assert all(weights[u] > 0 for u in prov)
    assert math.fsum(weights[u] for u in prov) <= CS.PROVISIONAL_KA_MAX_SHARE * CS.profile_for().tracks[0] + 1e-12
    monkeypatch.setattr(CS, "KA_PROV_TWO_PER_COLDKEY", False)
    assert _paths(st, subs)[0] == list(range(1, 19, 2))
    monkeypatch.setattr(CS, "KA_PROV_TWO_PER_COLDKEY", True)
    extra = _restart(st, uid=19, coldkey="pair0")
    assert _paths(st, subs + [extra])[0] == list(range(3, 19, 2)) + list(range(4, 19, 2))


@pytest.mark.parametrize("kind", ["slot_restart", "v5_restart"])
def test_restarted_first_draw_failure_ends_waiting_immediately(kind):
    st = ST.SpecState()
    s = _restart(st, kind)
    assert _paths(st, [s])[0] == [1]
    s.status, s.draw1 = ST.NOT_BETTER, "failed-first-draw"
    s.history.append({"kind": "first", "draw": s.draw1, "ts": THU + H})
    assert _paths(st, [s], THU + H) == ([], [])


def test_reader_is_pure_and_specialist_restart_is_unchanged():
    import copy

    st = ST.SpecState()
    s = _restart(st)
    before = copy.deepcopy(st)
    assert _paths(st, [s])[0] == [1]
    assert st == before
    s.track_id = "demand_forecasting"
    assert E._restart_wait_since(s) is None and not E._our_wait(s)
