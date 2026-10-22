"""Opening generalist cohort: playoff places 2 and 3 take the 17 % / 8 % former-champion slots
(competition_split.OPENING_PLAYOFF_SLOTS_START_UTC; docs/GENERALIST-RELEASE.md, "Opening playoff slots")."""

import copy
import math
from datetime import datetime, timedelta, timezone

import pytest
from spec_helpers import commit, neuron, snapshot

from reax_subnet import competition_split as CS
from reax_subnet.competition_split import combine_weights
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import compute_leaders, opening_playoff_slots

TRACK = "generalist"
OPEN = CS.GENERALIST_START_UTC.timestamp()
GATE = CS.GENERALIST_START_UTC + timedelta(hours=6)
DAY = 86400.0
BASE_ID = "81a3eb978ff3c5b90c273a100f36499a86dfec698652c2d181f43e11e89fe08d"      # floor 58.01
NO_FLOOR_BASE = "f" * 64
NEED = CS.KEEPALIVE_BASE_FLOOR[BASE_ID][TRACK] + CS.OPENING_PLAYOFF_BASE_MARGIN     # 59.51
SERVING = {11: 0.6, 12: 0.4}


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setattr(CS, "OPENING_PLAYOFF_SLOTS_START_UTC", GATE)


def sub(uid, score, nonproc=None, *, status=ST.NOT_BETTER, crowned=None, base=BASE_ID, **kw):
    key = f"{neuron(uid).hotkey}|{commit(TRACK, f'a/m{uid}')}"
    s = ST.Submission(key=key, uid=uid, hotkey=neuron(uid).hotkey, coldkey=f"c{uid}", track_id=TRACK, repo=f"a/m{uid}",
                      revision="a" * 40, first_seen_block=100 + uid, first_seen_ts=OPEN - 3600.0, status=status, score=score,
                      champion_since_ts=crowned, base_id=base, delta_checked=True, **kw)
    s.history.append({"draw": "d1", "kind": "playoff", "score": score, "field": 5,
                      "nonproc": score - 1.0 if nonproc is None else nonproc, "by_tier": {}})
    return s


def make_state(*, play_ts=OPEN + 3600.0, crowned=None, subs=None):
    """Winner uid 1 (70), place 2 uid 2 (66), place 3 uid 3 (62), place 4 uid 4 (61); one playoff draw d1."""
    crowned = play_ts + 60.0 if crowned is None else crowned
    subs = subs or [sub(1, 70.0, status=ST.CHAMPION, crowned=crowned), sub(2, 66.0), sub(3, 62.0), sub(4, 61.0)]
    ts = ST.TrackState(TRACK, submissions={s.key: s for s in subs}, champion=subs[0].key)
    ts.draws["d1"] = ST.Draw("d1", play_ts, 1000, 1, [])
    return ST.SpecState(tracks={TRACK: ts})


def view(uids=(1, 2, 3, 4)):
    return snapshot(2000, [neuron(u, commit(TRACK, f"a/m{u}")) for u in uids])


def uids(state, now, **kw):
    return [u for u, _ in compute_leaders(state, [TRACK], view(**kw), now=now)[TRACK]]


T1 = OPEN + 3600.0 + 60.0 + 6 * DAY      # inside the window after a post-gate crowning
POST = GATE.timestamp() + 3600.0


def test_gate_none_is_bit_identical(monkeypatch):
    monkeypatch.setattr(CS, "OPENING_PLAYOFF_SLOTS_START_UTC", None)
    st = make_state(play_ts=POST)
    before = copy.deepcopy(st)
    res = compute_leaders(st, [TRACK], view(), now=POST + DAY)[TRACK]
    assert [u for u, _ in res] == [1] and st == before
    assert compute_leaders(st, [TRACK], view(), now=POST + DAY) == compute_leaders(st, [TRACK], view())
    assert opening_playoff_slots(st.tracks[TRACK], POST + DAY) == []
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=POST + DAY)
    assert w[1] == pytest.approx(0.30 * 0.75) and set(w) == {1, 11, 12}


def test_places_two_and_three_take_the_former_champion_ranks():
    st = make_state(play_ts=POST)
    res = compute_leaders(st, [TRACK], view(), now=POST + DAY)[TRACK]
    assert res == [(1, 70.0), (2, 66.0), (3, 62.0)]                 # place 4 (uid 4) is not a slot
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=POST + DAY)
    assert (w[1], w[2], w[3]) == pytest.approx((0.30 * 0.75, 0.30 * 0.17, 0.30 * 0.08))
    assert 4 not in w and math.fsum(w.values()) == pytest.approx(1.0)


def test_no_extra_emission_same_competition_share_as_real_formers():
    st = make_state(play_ts=POST)
    slots = compute_leaders(st, [TRACK], view(), now=POST + DAY)
    real = CS.RankedLeaders([(1, 70.0), (2, 66.0), (3, 62.0)])
    now = POST + DAY
    w_slots = combine_weights(SERVING, None, 0.0, slots, now=now)
    w_real = combine_weights(SERVING, None, 0.0, {TRACK: real}, now=now)
    assert w_slots == w_real
    assert sum(w_slots[u] for u in (1, 2, 3)) == pytest.approx(CS.profile_for("50/50", datetime.fromtimestamp(now, timezone.utc)).tracks[0])


def test_before_the_gate_nothing_changes_playoff_before_vs_after():
    pre = make_state(play_ts=OPEN + 3600.0)                      # playoff and crowning before the gate
    g = GATE.timestamp()
    assert uids(pre, g - 1) == [1]                                # off before the date
    assert uids(pre, g) == [1, 2, 3]                              # boundary instant: on, slots START at the gate
    assert opening_playoff_slots(pre.tracks[TRACK], g + 7 * DAY - 1) and not opening_playoff_slots(pre.tracks[TRACK], g + 7 * DAY)
    post = make_state(play_ts=POST)                              # playoff after the gate: slots start at the crowning
    crown = POST + 60.0
    assert uids(post, crown) == [1, 2, 3]
    assert uids(post, crown + 7 * DAY - 1) == [1, 2, 3] and uids(post, crown + 7 * DAY) == [1]


def test_slots_count_from_the_gate_not_from_an_earlier_crowning():
    pre = make_state(play_ts=OPEN + 3600.0)
    g = GATE.timestamp()
    assert (OPEN + 3660.0) + 7 * DAY < g + 7 * DAY               # the crowning's own 7 days would end earlier
    assert uids(pre, OPEN + 3660.0 + 7 * DAY + 10) == [1, 2, 3]


@pytest.mark.parametrize("what", ["base_margin", "no_base_floor", "no_base", "copy", "withdrawn", "rejected",
                                  "unchecked", "not_committed"])
def test_each_qualification_condition(what):
    st = make_state(play_ts=POST)
    s2 = st.tracks[TRACK].submissions[sub(2, 0).key]
    h = s2.history[0]
    if what == "base_margin":
        h["nonproc"] = NEED - 0.01
    elif what == "no_base_floor":
        s2.base_id = NO_FLOOR_BASE
    elif what == "no_base":
        s2.base_id = ""
    elif what == "copy":
        s2.derived_from = "x|y"
    elif what == "withdrawn":
        s2.status = ST.WITHDRAWN
    elif what == "rejected":
        s2.status = ST.REJECTED
    elif what == "unchecked":
        s2.delta_checked = False
    ids = (1, 3, 4) if what == "not_committed" else (1, 2, 3, 4)
    got = uids(st, POST + DAY, uids=ids)
    # a failing place is not filled by a lower one (place 4 stays out); place 3 moves up one rank like any unranked entry
    assert got == [1, 3]


def test_base_margin_boundary_is_strict():
    st = make_state(play_ts=POST)
    s2 = st.tracks[TRACK].submissions[sub(2, 0).key]
    s2.history[0]["nonproc"] = NEED
    assert uids(st, POST + DAY) == [1, 3]                       # exactly floor + 1.5 does not qualify
    s2.history[0]["nonproc"] = NEED + 1e-9
    assert uids(st, POST + DAY) == [1, 2, 3]


def test_slots_survive_the_pruned_playoff_draw():
    from reax_subnet.spec.evaluator import SpecEvaluator
    st = make_state(play_ts=POST)
    ts = st.tracks[TRACK]
    for i in range(41):
        ts.draws[f"x{i}"] = ST.Draw(f"x{i}", POST + 100 + i, 1000, 1, [])
    SpecEvaluator._prune_draws(ts, POST + DAY)
    assert "d1" not in ts.draws
    assert uids(st, POST + DAY) == [1, 2, 3]


def test_lone_crowning_has_no_slots():
    st = make_state(play_ts=POST)
    for s in st.tracks[TRACK].submissions.values():
        s.history.clear()
    assert uids(st, POST + DAY) == [1]


def test_skipped_entry_above_the_winner_holds_no_place():
    st = make_state(play_ts=POST, subs=[sub(1, 70.0, status=ST.CHAMPION, crowned=POST + 60.0),
                                        sub(2, 72.0, derived_from="w"), sub(3, 66.0), sub(4, 62.0)])
    assert uids(st, POST + DAY) == [1, 3, 4]


def test_pushed_out_is_for_good_when_the_former_stops_committing():
    st = make_state(play_ts=POST)
    ts = st.tracks[TRACK]
    f = sub(5, 65.0, status=ST.FORMER, crowned=POST + 2 * DAY)
    f.history.clear()
    ts.submissions[f.key] = f
    now = POST + 3 * DAY
    assert [u for u, _ in compute_leaders(st, [TRACK], view((1, 2, 3, 4, 5)), now=now)[TRACK]] == [1, 5, 2]
    assert [u for u, _ in compute_leaders(st, [TRACK], view((1, 2, 3, 4)), now=now)[TRACK]] == [1, 2]


def test_entry_floor_is_exclusive_and_the_place_is_not_refilled():
    F = CS.GENERALIST_ENTRY_FLOOR
    low = [sub(1, 70.0, status=ST.CHAMPION, crowned=POST + 60.0), sub(2, F, nonproc=NEED + 0.01), sub(3, F - 3.0, nonproc=NEED + 0.01),
           sub(4, F - 4.0, nonproc=NEED + 0.01)]
    assert uids(make_state(play_ts=POST, subs=low), POST + DAY) == [1]          # at the floor is not above it
    ok = [low[0], sub(2, F + 0.01, nonproc=NEED + 0.01), low[2], low[3]]
    assert uids(make_state(play_ts=POST, subs=ok), POST + DAY) == [1, 2]       # place 3 is below the floor, place 4 is no slot


def test_unapproved_slot_keeps_its_rank_and_the_share_goes_to_inference():
    st = make_state(play_ts=POST)
    res = compute_leaders(st, [TRACK], view(), now=POST + DAY)
    w = combine_weights(SERVING, None, 0.0, res, now=POST + DAY, award_ok=lambda t, u: u != 2)
    assert 2 not in w and w[3] == pytest.approx(0.30 * 0.08)                  # rank 3 is not promoted
    assert w[1] == pytest.approx(0.30 * 0.75)


def test_pushed_out_by_new_former_champions():
    st = make_state(play_ts=POST)
    ts = st.tracks[TRACK]
    # uid 1 is dethroned by uid 5 (a later champion): uid 1 becomes the newest former champion
    s5 = sub(5, 80.0, status=ST.CHAMPION, crowned=POST + 2 * DAY)
    s5.history.clear()
    ts.submissions[s5.key] = s5
    ts.champion = s5.key
    s1 = ts.submissions[sub(1, 0).key]
    s1.status = ST.FORMER
    v = view((1, 2, 3, 4, 5))
    now = POST + 3 * DAY
    assert [u for u, _ in compute_leaders(st, [TRACK], v, now=now)[TRACK]] == [5, 1, 2]       # place 3 pushed out
    # a second dethroning pushes place 2 out as well
    s6 = sub(6, 85.0, status=ST.CHAMPION, crowned=POST + 2.5 * DAY)
    s6.history.clear()
    ts.submissions[s6.key] = s6
    ts.champion = s6.key
    ts.submissions[s5.key].status = ST.FORMER
    v = view((1, 2, 3, 4, 5, 6))
    assert [u for u, _ in compute_leaders(st, [TRACK], v, now=now)[TRACK]] == [6, 5, 1]
    # after the 7 days the remaining slot is gone too
    assert opening_playoff_slots(ts, POST + 60.0 + 7 * DAY) == []


def test_a_slot_is_never_rank_one():
    st = make_state(play_ts=POST)
    assert uids(st, POST + DAY, uids=(2, 3, 4)) == []                 # the champion no longer commits: no slot takes rank 1


def test_only_the_opening_cohort_counts():
    # a playoff before the generalist opening, a later playoff, and a field with a post-opening entrant: no slots
    assert uids(make_state(play_ts=OPEN - 3600.0, crowned=OPEN - 3000.0), POST + DAY) == [1]
    late = make_state(play_ts=POST)
    for s in late.tracks[TRACK].submissions.values():
        s.first_seen_ts = OPEN + 10.0
    assert uids(late, POST + DAY) == [1]
    # a second, later playoff (another cohort) never adds slots on top of the opening one
    st = make_state(play_ts=POST)
    ts = st.tracks[TRACK]
    ts.draws["d2"] = ST.Draw("d2", POST + 5 * DAY, 1100, 2, [])
    extra = sub(7, 90.0)
    extra.history[0]["draw"] = "d2"
    ts.submissions[extra.key] = extra
    assert uids(st, POST + DAY, uids=(1, 2, 3, 4, 7)) == [1, 2, 3]


def test_no_opening_winner_no_slots():
    st = make_state(play_ts=POST)
    for s in st.tracks[TRACK].submissions.values():
        s.champion_since_ts = None
        s.status = ST.NOT_BETTER
    st.tracks[TRACK].champion = ""
    assert uids(st, POST + DAY) == []
    only_non_generalist = make_state(play_ts=POST)
    only_non_generalist.tracks[TRACK].track_id = "demand_forecasting"
    assert opening_playoff_slots(only_non_generalist.tracks[TRACK], POST + DAY) == []


def test_extension_group_ranks_first_and_ties_go_to_the_earlier_commitment():
    st = make_state(play_ts=POST)
    subs = st.tracks[TRACK].submissions
    s1, s2, s3, s4 = (subs[sub(u, 0).key] for u in (1, 2, 3, 4))
    for s in (s1, s3, s4):                                          # extension group: the winner, uid 3 and 4 (draw2 present)
        s.history[0]["draw2"] = "d2"
    s4.history[0]["score"], s4.score = 64.0, 64.0
    s3.history[0]["score"], s3.score = 64.0, 64.0                  # tie: uid 3 committed earlier
    s4.history[0]["nonproc"] = s3.history[0]["nonproc"] = 63.0
    assert [u for u, _ in compute_leaders(st, [TRACK], view(), now=POST + DAY)[TRACK]] == [1, 3, 4]


def test_validators_agree_on_the_same_state():
    a, b = make_state(play_ts=POST), make_state(play_ts=POST)
    assert compute_leaders(a, [TRACK], view(), now=POST + DAY) == compute_leaders(b, [TRACK], view(), now=POST + DAY)
