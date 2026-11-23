"""Operator restarts void old decisions; non-clean entries get bounded unverified time."""

from copy import deepcopy

import pytest
from test_ka_restart import TRACK, _paths, _restart
from test_provisional_ka import _snapr
from test_waiting_ka_hotfix import H, THU, release_slots  # noqa: F401

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import review as RV
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors", "release_slots")


def _entry(st, kind="slot_restart", uid=1, coldkey=None, age=2, condition="derived"):
    s = _restart(st, kind, uid, coldkey)
    s.history[-1]["ts"] = THU - age * H
    if condition == "unchecked":
        s.delta_checked = False
    elif condition == "unhashed":
        s.hash_named = ""
    else:
        s.derived_from = "parent-key"
    return s


def _assert_paid(st, subs, expected, now=THU):
    prov, pending = _paths(st, subs, now)
    assert prov == expected and pending == []
    weights = CS.combine_weights({100: 1.0}, None, 0.0, {TRACK: []}, provisional={TRACK: prov})
    for s in subs:
        assert (weights.get(s.uid, 0.0) > 0) == (s.uid in expected)


@pytest.mark.parametrize("kind", ["slot_restart", "v5_restart"])
@pytest.mark.parametrize("condition", ["unchecked", "unhashed", "derived"])
@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.parametrize("age", [2, 24, 25])
def test_nonclean_restart_gets_unverified_window_from_restart(kind, condition, scored, age):
    st = ST.SpecState()
    s = _entry(st, kind, condition=condition, age=age)
    if not scored:
        s.score = None
    assert s.first_seen_ts < THU - CS.PROVISIONAL_KA_OUR_WAIT_HOURS * H
    before = deepcopy(st)
    _assert_paid(st, [s], [1] if age <= CS.PROVISIONAL_KA_UNVERIFIED_HOURS else [])
    assert st == before


@pytest.mark.parametrize("kind", ["slot_restart", "v5_restart"])
def test_clean_restart_keeps_our_wait_window(kind, monkeypatch):
    st = ST.SpecState()
    s = _restart(st, kind)
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 0)
    _assert_paid(st, [s], [1], THU + CS.PROVISIONAL_KA_OUR_WAIT_HOURS * H)
    _assert_paid(st, [s], [], THU + CS.PROVISIONAL_KA_OUR_WAIT_HOURS * H + 1)


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("dirty", [False, True])
def test_review_or_hold_restart_stays_excluded(monkeypatch, held, dirty):
    st = ST.SpecState()
    s = _entry(st) if dirty else _restart(st)
    if held:
        monkeypatch.setenv(RV.HOLD_ENV, f"{s.hotkey}@{s.revision}")
    else:
        s.history.insert(0, {"kind": RV.FLAG_KIND, "flags": ["reparametrised"]})
    _assert_paid(st, [s], [])


def test_unverified_cap_counts_scored_derived_restarts_oldest_commitment_first(monkeypatch):
    st = ST.SpecState()
    subs = [_entry(st, uid=uid) for uid in (1, 2, 3)]
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 2)
    snap = _snapr(subs, cblocks={1: 19003, 2: 19001, 3: 19002})
    assert E.compute_provisional(st, [TRACK], snap, now=THU)[TRACK] == [2, 3]
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 0)
    _assert_paid(st, subs, [])


@pytest.mark.parametrize("two", [False, True])
def test_per_coldkey_and_open_entry_limits_still_bind(monkeypatch, two):
    st = ST.SpecState()
    subs = [_entry(st, uid=uid, coldkey="same") for uid in (1, 2)]
    monkeypatch.setattr(CS, "KA_PROV_TWO_PER_COLDKEY", two)
    _assert_paid(st, subs, [1, 2] if two else [1])
    subs.append(_entry(st, uid=3, coldkey="same"))
    _assert_paid(st, subs, [])


@pytest.mark.parametrize("own_fake", [False, True])
def test_abuse_clause_excludes_unverified_restart_with_fake(monkeypatch, own_fake):
    st = ST.SpecState()
    s = _entry(st, coldkey="same")
    fake = s if own_fake else _entry(st, uid=2, coldkey="same")
    fake.first_seen_ts = THU - H
    fake.reason = E.WAITING_FOR_REPO + ": 404"
    assert CS.KA_ABUSE_CLAUSE and E._is_fake(fake)
    _assert_paid(st, [s], [])
    monkeypatch.setattr(CS, "KA_ABUSE_CLAUSE", False)
    _assert_paid(st, [s], [1])


def test_restart_bypasses_hotkey_rewindow_without_consuming_another_window():
    st = ST.SpecState()
    s = _entry(st)
    earlier = deepcopy(s)
    earlier.key += "-older"
    earlier.status = ST.WITHDRAWN
    earlier.first_seen_ts = THU - 30 * H
    st.track(TRACK).submissions[earlier.key] = earlier
    _assert_paid(st, [s], [1])  # ordinary rewindow would be expired


@pytest.mark.parametrize("other_age, restart_age", [(1, 80), (30, 2)])
def test_restart_does_not_change_other_entry_rewindow(other_age, restart_age):
    st = ST.SpecState()
    restarted = _entry(st, age=restart_age)
    # Another track's current entry shares the restarted generalist's hotkey.
    # The operator restart must neither anchor nor reopen its ordinary window.
    other = deepcopy(restarted)
    other.track_id = "demand_forecasting"
    other.key = other.key.replace(":generalist:", f":{other.track_id}:")
    other.history = []
    other.score = None
    other.first_seen_ts = THU - other_age * H
    other.delta_checked = False
    st.track(other.track_id).submissions[other.key] = other
    ordinary = deepcopy(st)
    ordinary.track(TRACK).submissions[restarted.key].history = []
    snap = _snapr([other])
    actual = E.compute_provisional(st, [other.track_id], snap, now=THU)[other.track_id]
    assert actual == E.compute_provisional(ordinary, [other.track_id], snap, now=THU)[other.track_id]
    assert actual == ([1] if other_age == 1 else [])
