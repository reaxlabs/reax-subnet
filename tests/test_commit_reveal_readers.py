"""Commit/reveal payment READERS on the validator line (ported from the cr-fix worker branch, no worker code).

The worker owns the reveal state; the validator only reads it: an unrevealed private entry whose reveal deadline is set is
not paid, a re-gated or voided one stops at once, a waiting keep-alive before the first slot (no deadline yet) is still
paid, and a state without the new fields is byte-identical to the base."""

import json
import subprocess
import sys
import importlib.util
import os

import pytest
from spec_helpers import REV, snapshot
from test_keepalive import DF, NOW, _snap, _sub

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
DUE = NOW + 86400.0


def _private(st, uid, status, score, **kw):
    s = _sub(st, uid, status, score)
    s.nonpublic = True
    s.reveal_hash = "original"
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _prov(st, uid, **kw):
    from test_provisional_ka import _p
    s = _p(st, uid)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _pay(st, subs, now=NOW):
    snap = _snap(subs)
    return (E.compute_leaders(st, [DF], snap, now=now)[DF], E.compute_keepalive(st, [DF], snap, {DF: 0.0}, now=now)[DF])


def test_public_entry_and_legacy_fields_are_paid():
    st = ST.SpecState()
    s = _sub(st, 1, ST.CHAMPION, 40.0)
    lead, ka = _pay(st, [s])
    assert [u for u, _ in lead] == [1] and [u for u, _ in ka] == [1]
    assert ST.reveal_allows_payment(s, NOW) and ST.reveal_allows_payment(object(), NOW)


def test_unrevealed_private_champion_gets_no_share_but_keepalive_until_the_deadline():
    st = ST.SpecState()
    s = _private(st, 1, ST.CHAMPION, 40.0, reveal_due_ts=DUE)
    lead, ka = _pay(st, [s])
    assert lead == [] and [u for u, _ in ka] == [1]  # #29731: prune protection stays until reveal_due_ts
    assert _pay(st, [s], now=DUE + 1) == ([], [])
    s.revealed_ts = NOW - 10.0                      # the receipt is what counts, not the clock
    lead, ka = _pay(st, [s], now=DUE + 5)
    assert [u for u, _ in lead] == [1] and [u for u, _ in ka] == [1]
    s.revealed_ts = DUE + 1                         # revealed late
    assert _pay(st, [s], now=DUE + 5) == ([], [])


def test_private_entry_before_its_first_slot_keeps_the_waiting_keepalive():
    st = ST.SpecState()
    s = _private(st, 1, ST.PENDING_CONFIRM, 30.0)   # reveal_due_ts is None: no slot played yet
    assert ST.reveal_allows_payment(s, NOW)
    assert [u for u, _ in _pay(st, [s])[1]] == [1]


def test_regated_entry_stops_paying_at_once():
    st = ST.SpecState()
    s = _private(st, 1, ST.CHAMPION, 40.0, reveal_due_ts=DUE, revealed_ts=NOW - 10.0)
    assert [u for u, _ in _pay(st, [s])[0]] == [1]
    s.public_lost_ts = NOW
    assert _pay(st, [s]) == ([], [])
    s.public_lost_ts = None                         # public again within the worker's window: payment resumes
    assert [u for u, _ in _pay(st, [s])[0]] == [1]


def test_void_champion_is_not_paid_and_not_ranked_as_former():
    st = ST.SpecState()
    champ = _private(st, 1, ST.CHAMPION, 40.0, reveal_due_ts=DUE, revealed_ts=NOW - 10.0, reveal_void=True)
    former = _private(st, 2, ST.FORMER, 35.0, reveal_void=True)
    assert not ST.reveal_allows_payment(champ, NOW)
    assert _pay(st, [champ, former]) == ([], [])


def test_unrevealed_champion_share_goes_to_the_next_ranked_entry_not_the_champion():
    """cr-critic2 finding C: the unpaid seat is not kept: a former champion moves up (documented behaviour)."""
    st = ST.SpecState()
    champ = _private(st, 1, ST.CHAMPION, 40.0, reveal_due_ts=DUE)
    former = _sub(st, 2, ST.FORMER, 35.0)
    lead, _ = _pay(st, [champ, former])
    assert [u for u, _ in lead] == [2]


def test_stale_state_cannot_extend_a_reveal_deadline():
    st = ST.SpecState()
    s = _private(st, 1, ST.CHAMPION, 40.0, reveal_due_ts=DUE)
    assert _pay(st, [s], now=DUE + 1) == ([], [])    # status still champion, no void flag written yet
    s.revealed_ts = float("nan")
    s.reveal_due_ts = float("nan")
    assert not ST.reveal_allows_payment(s, NOW)


def test_provisional_and_mark_provisional_need_the_reveal_receipt(monkeypatch):
    from test_provisional_ka import _snapr
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)    # the stored-window path (compute_provisional); the hotfix path is below
    st = ST.SpecState()
    s = _prov(st, 1, nonpublic=True, reveal_due_ts=DUE)
    snap = _snapr([s])
    assert E.compute_provisional(st, [DF], snap, now=NOW)[DF] == [1]    # before the deadline: prune protection (#29731)
    s.reveal_due_ts = NOW - 1.0
    assert E.compute_provisional(st, [DF], snap, now=NOW)[DF] == []
    s.provisional_since_ts = None
    assert E.mark_provisional(st, [DF], snap, NOW) is False and s.provisional_since_ts is None
    s.reveal_due_ts = DUE
    assert E.mark_provisional(st, [DF], snap, NOW) is True and s.provisional_since_ts is not None   # deadline ahead
    s.revealed_ts, s.provisional_since_ts = NOW - 5.0, NOW - 3600.0
    assert [u for u in E.compute_provisional(st, [DF], snap, now=NOW)[DF]] == [1]
    s.reveal_void = True
    assert E.compute_provisional(st, [DF], snap, now=NOW)[DF] == []


def test_committed_waiting_keepalive_before_the_slot_is_paid_after_a_deadline_it_is_not():
    from test_provisional_ka import _snapr
    st = ST.SpecState()
    s = _prov(st, 1, nonpublic=True)
    snap = _snapr([s], cblocks={1: 19990})
    s.provisional_since_ts = None
    assert E._committed_waiting(st, [DF], snap, NOW)[DF] == [1]       # waiting keep-alive before the slot: paid
    s.reveal_due_ts = DUE
    assert E._committed_waiting(st, [DF], snap, NOW)[DF] == [1]       # slot played, deadline ahead: still protected
    assert E._committed_waiting(st, [DF], snap, DUE + 1)[DF] == []    # deadline passed, no receipt: not paid
    s.revealed_ts = NOW - 1.0
    assert E._committed_waiting(st, [DF], snap, NOW)[DF] == [1]
    s.public_lost_ts = NOW
    assert E._committed_waiting(st, [DF], snap, NOW)[DF] == []


def test_default_fields_are_omitted_and_state_bytes_equal_the_base(tmp_path):
    st = ST.SpecState()
    s = _sub(st, 1, ST.CHAMPION, 40.0)
    encoded = ST._to_json(st)["tracks"][DF]["submissions"][s.key]
    for key in ("nonpublic", "reveal_hash", "reveal_due_ts", "revealed_ts", "reveal_void", "access_ready_ts", "gated_seen_ts",
                "credential_wait_ts", "public_seen_ts", "reveal_infra_fail_ts", "reveal_grace", "reveal_void_ts",
                "public_checked_ts", "public_lost_ts", "admission_after_ts"):
        assert key not in encoded
    base = tmp_path / "base_state.py"
    base.write_bytes(subprocess.check_output(["git", "show", "233dbd2:src/reax_subnet/spec/state.py"]))
    name = "reax_subnet.spec._commit_reveal_base_state"
    spec = importlib.util.spec_from_file_location(name, base)
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    try:
        spec.loader.exec_module(old)
        legacy = old.SpecState()
        args = ("k", 1, "h", "c", DF, "r/m", REV, 1, 1.0)
        a_state = ST.SpecState()
        a_state.track(DF).submissions["k"] = ST.Submission(*args)
        legacy.track(DF).submissions["k"] = old.Submission(*args)
        a, b = ST._to_json(a_state), old._to_json(legacy)
        a["saved_at"] = b["saved_at"] = 0
        assert json.dumps(a, sort_keys=True).encode() == json.dumps(b, sort_keys=True).encode()
    finally:
        del sys.modules[name]


def test_new_fields_round_trip_through_the_state_file(tmp_path):
    st = ST.SpecState()
    s = _private(st, 1, ST.CHAMPION, 40.0, reveal_due_ts=DUE, revealed_ts=NOW, public_lost_ts=NOW + 1, access_ready_ts=NOW - 9)
    path = str(tmp_path / "state.json")
    ST.save(st, path)
    back = ST.load(path)
    t = back.track(DF).submissions[s.key]
    assert (t.nonpublic, t.reveal_hash, t.reveal_due_ts, t.revealed_ts, t.public_lost_ts, t.access_ready_ts) == (
        True, "original", DUE, NOW, NOW + 1, NOW - 9)


def test_compute_provisional_on_the_waiting_hotfix_path_is_gated_too():
    from test_provisional_ka import _snapr
    assert CS.WAITING_KA_HOTFIX is True
    st = ST.SpecState()
    s = _prov(st, 1, nonpublic=True, reveal_due_ts=NOW - 1.0)
    snap = _snapr([s], cblocks={1: 19990})
    assert E.compute_provisional(st, [DF], snap, now=NOW)[DF] == []
    s.reveal_due_ts = DUE
    assert E.compute_provisional(st, [DF], snap, now=NOW)[DF] == [1]
    s.reveal_due_ts, s.revealed_ts = NOW - 1.0, NOW - 5.0
    assert E.compute_provisional(st, [DF], snap, now=NOW)[DF] == [1]


def test_private_entry_pending_confirmation_after_its_slot_is_protected_until_the_deadline():
    """#29724/#29731: uid-162 shape with commit-reveal: first draw played, waiting for the confirmation draw."""
    st = ST.SpecState()
    s = _private(st, 1, ST.PENDING_CONFIRM, 30.0, reveal_due_ts=DUE)
    assert not ST.reveal_allows_payment(s, NOW) and ST.reveal_allows_keepalive(s, NOW)
    lead, ka = _pay(st, [s])
    assert lead == [] and [u for u, _ in ka] == [1]
    assert _pay(st, [s], now=DUE + 1) == ([], [])
    s.revealed_ts = DUE + 1                         # a late receipt does not buy protection
    assert not ST.reveal_allows_keepalive(s, DUE + 2)
    s.revealed_ts, s.public_lost_ts = NOW - 1.0, NOW  # revealed, then gated again: stops at once
    assert not ST.reveal_allows_keepalive(s, NOW)
    s.public_lost_ts, s.reveal_void = None, True
    assert not ST.reveal_allows_keepalive(s, NOW)
    assert not ST.reveal_allows_keepalive(_private(st, 2, ST.PENDING_CONFIRM, 30.0, reveal_due_ts=float("nan")), NOW)
