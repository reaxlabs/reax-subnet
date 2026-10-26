"""Generalist opening co-champions (competition_split.GENERALIST_CO_CHAMPIONS, 1.2.0rc11.post1): payout, places 2/3, keep-alive."""

import copy
import math
from datetime import datetime, timezone

import pytest
from spec_helpers import commit, neuron, snapshot
from test_opening_playoff_slots import BASE_ID, GATE, OPEN, POST, SERVING, TRACK, DAY, make_state, sub, view

from reax_subnet import competition_split as CS
from reax_subnet.competition_split import combine_weights, rank_shares
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import _crown_pending, compute_keepalive, compute_leaders, cochamp_members, opening_playoff_slots

CO_UIDS = (5, 1, 6)           # uid 1 is the champion (70), uid 5 (69.5) and uid 6 (69.0) are pending co-champions


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setattr(CS, "OPENING_PLAYOFF_SLOTS_START_UTC", GATE)


def key_of(uid):
    return f"{neuron(uid).hotkey}|{commit(TRACK, f'a/m{uid}')}"


def case(monkeypatch, *, co=CO_UIDS, champion=1, extra=()):
    """Playoff field: 1 (70, champion), 5 (69.5), 6 (69), 2 (66), 3 (62), 4 (61)."""
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", tuple(key_of(u) for u in co))
    play_ts = POST
    subs = [sub(1, 70.0, status=ST.CHAMPION, crowned=play_ts + 60.0),
            sub(5, 69.5, status=ST.PENDING_CONFIRM), sub(6, 69.0, status=ST.PENDING_CONFIRM),
            sub(2, 66.0), sub(3, 62.0), sub(4, 61.0), *extra]
    return make_state(play_ts=play_ts, subs=subs)


NOW = POST + DAY
UIDS = (1, 2, 3, 4, 5, 6)


def leaders(st, uids=UIDS):
    return compute_leaders(st, [TRACK], view(uids), now=NOW)[TRACK]


def test_default_release_constant_lists_the_three_state_keys():
    assert len(CS.GENERALIST_CO_CHAMPIONS) == 3 and len(set(CS.GENERALIST_CO_CHAMPIONS)) == 3
    assert {k.split("|")[1].split(":")[2] for k in CS.GENERALIST_CO_CHAMPIONS} == {
        "Andrew93110/sn92-gen-v1@4d2a519d78377e63c46cbf35c26b09a71c28adf8",
        "putty77/sn92-generalist-a@dbeec5635aa717e4654b8a719499a0ce80c370ea",
        "reax-knockout/sn92-gen-a1@d57f18b92547b9c669bf93fa74ee64443ddb45d8"}
    from conftest import RELEASE_CROWN_SLOTS_UTC
    # rc11.post3 (#24089): every Monday and Thursday 14:30Z from Mon 26 Oct 2026, generated through 31 Mar 2027
    assert RELEASE_CROWN_SLOTS_UTC[:4] == tuple(datetime(2026, m, d, 14, 30, tzinfo=timezone.utc)
                                                for m, d in ((10, 26), (10, 29), (11, 2), (11, 5)))
    assert all(t.weekday() in (0, 3) and (t.hour, t.minute) == (14, 30) for t in RELEASE_CROWN_SLOTS_UTC)
    assert all((b - a).days in (3, 4) for a, b in zip(RELEASE_CROWN_SLOTS_UTC, RELEASE_CROWN_SLOTS_UTC[1:]))
    assert RELEASE_CROWN_SLOTS_UTC[-1] == datetime(2027, 3, 29, 14, 30, tzinfo=timezone.utc)


def test_gate_off_is_bit_identical(monkeypatch):
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", ())
    st = make_state(play_ts=POST)
    before = copy.deepcopy(st)
    res = leaders(st, (1, 2, 3, 4))
    assert [u for u, _ in res] == [1, 2, 3] and res.cochamps == {} and st == before
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW)
    assert (w[1], w[2], w[3]) == pytest.approx((0.30 * 0.75, 0.30 * 0.17, 0.30 * 0.08))
    assert rank_shares(res) == {1: 0.75, 2: 0.17, 3: 0.08}
    # the same state with the set listed but another champion: also unchanged
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", (key_of(7), key_of(8)))
    assert leaders(st, (1, 2, 3, 4)) == res and leaders(st, (1, 2, 3, 4)).cochamps == {}


def test_three_co_champions_share_rank_one_and_places_two_three_skip_them(monkeypatch):
    st = case(monkeypatch)
    res = leaders(st)
    assert [u for u, _ in res] == [1, 2, 3]                      # places 2/3 = the next two in the playoff order (66, 62)
    assert res.cochamps == {1: pytest.approx(1 / 3), 5: pytest.approx(1 / 3), 6: pytest.approx(1 / 3)}
    assert {s.uid for s, _ in opening_playoff_slots(st.tracks[TRACK], NOW)} == {2, 3}
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: True)
    third = 0.30 * 0.75 / 3
    assert (w[1], w[5], w[6]) == pytest.approx((third, third, third))
    assert (w[2], w[3]) == pytest.approx((0.30 * 0.17, 0.30 * 0.08)) and 4 not in w
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_without_the_set_the_pending_members_would_hold_the_places(monkeypatch):
    st = case(monkeypatch)
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", ())
    # not listed -> they are plain pending entries; they sort above nobody, but the exclusion is only for listed keys
    assert [u for u, _ in leaders(st)] == [1, 5, 6]


def test_unapproved_member_third_goes_to_inference_not_to_the_others(monkeypatch):
    st = case(monkeypatch)
    res = leaders(st)
    third = 0.30 * 0.75 / 3
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: u != 5)
    assert (w[1], w[6]) == pytest.approx((third, third)) and 5 not in w
    assert math.fsum(w.values()) == pytest.approx(1.0)
    assert w[11] + w[12] == pytest.approx(1 - 2 * third - 0.30 * 0.25)      # inference part keeps the rest
    # the champion itself unapproved: its third is not paid either
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: u != 1)
    assert 1 not in w and w[5] == pytest.approx(third) and w[6] == pytest.approx(third)


def test_withdrawn_or_uncommitted_member_is_dropped(monkeypatch):
    st = case(monkeypatch)
    st.tracks[TRACK].submissions[key_of(6)].status = ST.WITHDRAWN
    res = leaders(st)
    assert res.cochamps == {1: pytest.approx(1 / 3), 5: pytest.approx(1 / 3)}
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: True)
    assert 6 not in w and w[1] == pytest.approx(0.30 * 0.75 / 3)
    # uid 5 no longer commits to the listed text
    st = case(monkeypatch)
    res = compute_leaders(st, [TRACK], snapshot(2000, [neuron(u, commit(TRACK, f"a/m{u}")) if u != 5 else neuron(5, commit(TRACK, "a/other"))
                                                       for u in UIDS]), now=NOW)[TRACK]
    assert set(res.cochamps) == {1, 6}
    # no uid at all for uid 6
    res = leaders(st, (1, 2, 3, 4, 5))
    assert set(res.cochamps) == {1, 5}


def test_champion_gone_the_lead_is_the_first_committed_member(monkeypatch):
    st = case(monkeypatch)
    res = leaders(st, (2, 3, 4, 5, 6))                           # uid 1 deregistered
    assert [u for u, _ in res][0] == 5 and set(res.cochamps) == {5, 6}
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: True)
    assert w[5] == w[6] == pytest.approx(0.30 * 0.75 / 3) and 1 not in w


def test_set_beaten_every_entry_is_ordinary(monkeypatch):
    st = case(monkeypatch)
    ts = st.tracks[TRACK]
    new = sub(7, 80.0, status=ST.CHAMPION, crowned=POST + 7200.0)
    ts.submissions[new.key] = new
    ts.submissions[key_of(1)].status = ST.FORMER
    ts.champion = new.key
    assert cochamp_members(ts) == []
    res = leaders(st, (1, 2, 3, 4, 5, 6, 7))
    assert res.cochamps == {} and [u for u, _ in res][0] == 7 and set(res.formset) == {1, 5, 6}   # rc11.post2: the set is one former
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: True)
    assert w[7] == pytest.approx(0.30 * 0.75)                      # one champion again, not thirds


def test_cochamps_survive_the_competition_copy(monkeypatch):
    from reax_subnet.competitions import _copy
    res = leaders(case(monkeypatch))
    assert _copy(res).cochamps == res.cochamps and _copy(res) == res


def test_keepalive_treats_members_like_the_champion(monkeypatch):
    from test_keepalive import NOW as KNOW, _snap, _sub
    GEN = "generalist"
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0, nonproc=70.0, block=1, track=GEN),
            _sub(st, 2, ST.PENDING_CONFIRM, 40.0, nonproc=71.0, block=2, track=GEN),
            _sub(st, 3, ST.PENDING_CONFIRM, 40.0, nonproc=72.0, block=3, track=GEN),
            _sub(st, 4, ST.PENDING_CONFIRM, 40.0, nonproc=99.0, block=4, track=GEN)]
    ts = st.track(GEN)
    # the pending members' own window (first pass long ago) would be over; the crowning keeps it open
    for s in subs[1:3]:
        s.confirm_after_ts = KNOW - 100 * 3600.0
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", tuple(s.key for s in subs[:3]))
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS_CROWNED", datetime.fromtimestamp(KNOW - 3600.0, timezone.utc))
    nofloor = {GEN: 0.0}
    kw = dict(now=KNOW, nonproc_floors=nofloor, base_floors={GEN: {}})
    out = compute_keepalive(st, [GEN], _snap(subs), nofloor, **kw)[GEN]
    assert [u for u, _ in out] == [1, 2, 3, 4]            # all three listed (champion-like), plus the ordinary pending entry
    assert cochamp_members(ts) == [subs[0], subs[1], subs[2]]
    # the set ended: the stale pending members get nothing again
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", ())
    out = compute_keepalive(st, [GEN], _snap(subs), nofloor, **kw)[GEN]
    assert {u for u, _ in out} == {1, 4}


def test_keepalive_approved_member_has_no_slot_an_unapproved_one_keeps_it():
    tracks = {TRACK: _ranked({5: 1 / 3, 1: 1 / 3, 6: 1 / 3})}
    base = {11: 0.5, 12: 0.5}
    ka = {TRACK: [(5, 70.0), (1, 70.0), (6, 70.0), (9, 60.0)]}
    approved = combine_weights(base, None, 0.0, tracks, keepalive=ka, award_ok=lambda t, u: True, now=NOW)
    unapproved5 = combine_weights(base, None, 0.0, tracks, keepalive=ka, award_ok=lambda t, u: u != 5, now=NOW)
    third = 0.30 * 0.75 * 0.75 / 3          # the keep-alive total (25 % of the track share) is taken before the ranks
    # approved members: paid by their third only; entry 9 holds the only slot (the whole 25 % of 0.30)
    assert approved[5] == approved[1] == approved[6] == pytest.approx(third)
    assert approved[9] == pytest.approx(0.30 * 0.25)
    # uid 5 unapproved: no third, but the champion-like slot, shared with entry 9
    assert unapproved5[5] == unapproved5[9] == pytest.approx(0.30 * 0.25 / 2)
    assert unapproved5[1] == pytest.approx(third) and unapproved5[6] == pytest.approx(third)


def _ranked(cochamps):
    from reax_subnet.competition_split import RankedLeaders
    r = RankedLeaders([(next(iter(cochamps)), 70.0)])
    r.cochamps = dict(cochamps)
    return r


# ---------------------------------------------------------------- worker: the challenger bar (C) and the crown slots (D)

async def _setup(tmp_path, monkeypatch):
    """Champion uid 1 plus pending uids 2 and 3 (they beat it on their first draw), declared co-champions on the small test track."""
    from test_spec_evaluator import TRACK as T, Env, crown
    env = Env(tmp_path, {})
    await crown(env, 1, "a/c1", 0.65)
    env.add(2, "a/c2", 0.93)
    env.add(3, "a/c3", 0.95)
    await env.cycle({1: "a/c1", 2: "a/c2", 3: "a/c3"})
    assert env.sub(2).status == env.sub(3).status == ST.PENDING_CONFIRM
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", tuple(env.sub(u).key for u in (1, 2, 3)))
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS_TRACK", T)
    return env, T


async def test_challenger_bar_is_the_best_co_champion_on_the_challengers_own_draw(tmp_path, monkeypatch):
    env, T = await _setup(tmp_path, monkeypatch)
    ts = env.ev.state.track(T)
    allc = {1: "a/c1", 2: "a/c2", 3: "a/c3"}
    env.add(4, "a/ch4", 0.80)                    # above the champion (0.65) but below the best co-champion
    env.server.served.clear()
    await env.cycle({**allc, 4: "a/ch4"})
    assert env.sub(4).status == ST.NOT_BETTER
    assert sorted(env.server.served) == sorted(["a/c1", "a/c2", "a/c3", "a/ch4"])      # all three co-champions on the draw
    assert any("vs best co-champion uid" in e for e in ts.events) and any("co-champions: uid 1" in e for e in ts.events)
    assert env.sub(2).status == env.sub(3).status == ST.PENDING_CONFIRM and ts.champion == env.sub(1).key   # members untouched
    h = [x for x in env.sub(4).history if x["kind"] == "first"][-1]
    assert set(h["bar_scores"]) == {"1", "2", "3"} and h["incumbent"] == pytest.approx(max(h["bar_scores"].values()))
    assert h["bar_scores"][str(h["bar_uid"])] == h["incumbent"] and h["bar_uid"] in (2, 3) and h["bar_se"] > 0
    # the extra serves count against the hourly budget: 4 models + 1 for the earlier cycles are in eval_times
    assert len(env.ev.state.eval_times) >= 4


@pytest.mark.parametrize("v3", [False, True])
async def test_stronger_challenger_ends_the_set(tmp_path, monkeypatch, v3):
    env, T = await _setup(tmp_path, monkeypatch)
    env.cfg.v3_decisions = v3
    allc = {1: "a/c1", 2: "a/c2", 3: "a/c3", 6: "a/ch6"}
    env.add(6, "a/ch6", 0.995)
    await env.cycle(allc)
    assert env.sub(6).status == ST.PENDING_CONFIRM, env.sub(6).reason
    env.clock.advance(13 * 3600)
    out = await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert env.sub(6).status == ST.CHAMPION and ts.champion == env.sub(6).key and env.sub(1).status == ST.FORMER
    assert cochamp_members(ts) == []                              # the constant no longer matches the champion
    assert any("beat the best co-champion uid" in e for e in ts.events)
    assert out[T][0][0] == 6 and getattr(out[T], "cochamps", {}) == {}


async def test_gate_off_serves_only_the_champion(tmp_path, monkeypatch):
    env, T = await _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", ())
    env.add(4, "a/ch4", 0.80)
    env.server.served.clear()
    await env.cycle({1: "a/c1", 2: "a/c2", 3: "a/c3", 4: "a/ch4"})
    assert sorted(env.server.served) == ["a/c1", "a/ch4"]            # exactly the pre-post1 behaviour: champion + challenger
    ts = env.ev.state.track(T)
    assert not any("co-champion" in e for e in ts.events)


async def test_crown_slots_store_the_decision_until_the_slot(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from test_spec_evaluator import TRACK as T, Env, crown
    env = Env(tmp_path, {})
    await crown(env, 1, "a/c1", 0.65)
    env.add(2, "a/better", 0.95)
    allc = {1: "a/c1", 2: "a/better"}
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    slot = datetime.fromtimestamp(env.clock() + 5 * 3600, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot - timedelta(days=1), slot, slot + timedelta(days=1)))
    out = await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert env.sub(2).status == ST.PENDING_CONFIRM and _crown_pending(env.sub(2)) is not None and ts.champion == env.sub(1).key
    assert [u for u, _ in out[T]] == [1] and env.sub(2).score and "takes effect" in env.sub(2).reason     # score published
    env.clock.advance(3600)
    await env.cycle(allc)
    assert ts.champion == env.sub(1).key                            # before the slot: nothing changes
    env.clock.advance(5 * 3600)
    out = await env.cycle(allc)
    assert ts.champion == env.sub(2).key and env.sub(1).status == ST.FORMER and _crown_pending(env.sub(2)) is None
    assert [u for u, _ in out[T]] == [2, 1]


async def test_crown_slots_empty_crown_at_once(tmp_path, monkeypatch):
    from test_spec_evaluator import TRACK as T, Env, crown
    env = Env(tmp_path, {})
    await crown(env, 1, "a/c1", 0.65)
    env.add(2, "a/better", 0.95)
    await env.cycle({1: "a/c1", 2: "a/better"})
    env.clock.advance(13 * 3600)
    await env.cycle({1: "a/c1", 2: "a/better"})
    assert env.sub(2).status == ST.CHAMPION and _crown_pending(env.sub(2)) is None


def test_state_file_has_no_new_field(tmp_path):
    import dataclasses
    names = {f.name for f in dataclasses.fields(ST.Submission)}
    assert "crown_pending_ts" not in names
    from reax_subnet.spec.evaluator import _crown_pending
    s = next(iter(make_state(play_ts=POST).tracks[TRACK].submissions.values()))
    s.status = ST.PENDING_CONFIRM
    assert _crown_pending(s) is None
    s.history.append({"kind": "crown_pending", "ts": 5.0})
    assert _crown_pending(s) == 5.0
    s.history.append({"kind": "crown_done", "ts": 6.0})
    assert _crown_pending(s) is None


# ---------------------------------------------------------------- rc11.post2: crown slot without freeze + slot playoff

async def _slot_env(tmp_path, monkeypatch, skills, *, slot_in=40 * 3600):
    """Champion uid 1 (0.65); uids in ``skills`` (uid -> skill) pass their first draw; the slot is ``slot_in`` seconds ahead."""
    from datetime import datetime, timedelta, timezone
    from test_spec_evaluator import TRACK as T, Env, crown
    env = Env(tmp_path, {})
    await crown(env, 1, "a/c1", 0.65)
    allc = {1: "a/c1"}
    for u, sk in skills.items():
        env.add(u, f"a/e{u}", sk)
        allc[u] = f"a/e{u}"
    slot = datetime.fromtimestamp(env.clock() + slot_in, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot,))
    return env, T, allc, slot


async def test_no_freeze_two_challengers_earning_on_different_cycles_are_both_stored(tmp_path, monkeypatch):
    env, T, allc, slot = await _slot_env(tmp_path, monkeypatch, {2: 0.95})
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert _crown_pending(env.sub(2)) is not None and ts.champion == env.sub(1).key
    env.add(3, "a/e3", 0.97)                     # a later challenger is still evaluated while uid 2 waits
    allc[3] = "a/e3"
    env.server.served.clear()
    await env.cycle(allc)
    assert env.sub(3).status == ST.PENDING_CONFIRM and "a/e3" in env.server.served
    assert "a/e2" not in env.server.served       # the stored entry is not re-challenged
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    assert _crown_pending(env.sub(2)) is not None and _crown_pending(env.sub(3)) is not None
    assert ts.champion == env.sub(1).key
    st2 = ST.SpecState.from_json(env.ev.state.to_json()) if hasattr(ST.SpecState, "from_json") else None
    if st2 is not None:    # round trip keeps the crown_pending rows
        t2 = st2.track(T)
        assert sum(1 for s in t2.submissions.values() if _crown_pending(s) is not None) == 2


async def test_slot_with_one_earned_entry_crowns_it(tmp_path, monkeypatch):
    env, T, allc, slot = await _slot_env(tmp_path, monkeypatch, {2: 0.95}, slot_in=20 * 3600)
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert _crown_pending(env.sub(2)) is not None
    env.clock.advance(8 * 3600)
    await env.cycle(allc)
    assert ts.champion == env.sub(2).key and env.sub(1).status == ST.FORMER and _crown_pending(env.sub(2)) is None
    assert not any("slot playoff" in e for e in ts.events)


async def test_slot_with_two_earned_entries_runs_one_shared_playoff_draw(tmp_path, monkeypatch):
    env, T, allc, slot = await _slot_env(tmp_path, monkeypatch, {2: 0.90, 3: 0.99}, slot_in=20 * 3600)
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert _crown_pending(env.sub(2)) is not None and _crown_pending(env.sub(3)) is not None
    env.clock.advance(8 * 3600)
    env.server.served.clear()
    await env.cycle(allc)
    assert sorted(env.server.served) == ["a/e2", "a/e3"]             # the champion is not in the slot playoff
    assert ts.champion == env.sub(3).key and env.sub(1).status == ST.FORMER
    # rc11.post4: an entry that is not crowned is queued for a new first draw against the new champion (slot rules section 2/6);
    # both repos are under the same Hugging Face account here, so the holder rule (4.4) keeps the better one
    assert env.sub(2).status == ST.QUEUED and "in the slot, not crowned (4.4" in env.sub(2).reason and not env.sub(2).draw1
    assert any("slot round 1" in e for e in ts.events)
    assert _crown_pending(env.sub(2)) is None and _crown_pending(env.sub(3)) is None


# rc11.post3: the rc11.post2 extension / highest-mean tests are replaced by tests/test_slot_rules.py (up to three rounds while
# not separated by the margin, then the 0.5-point gap; no highest-mean fallback).


async def test_withdrawn_earned_entry_drops_out_of_the_slot(tmp_path, monkeypatch):
    env, T, allc, slot = await _slot_env(tmp_path, monkeypatch, {2: 0.90, 3: 0.99}, slot_in=20 * 3600)
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    env.clock.advance(8 * 3600)
    withdraw = {u: r for u, r in allc.items() if u != 3}              # uid 3 no longer commits: withdrawn
    ts = env.ev.state.track(T)
    await env.cycle(withdraw)
    assert env.sub(3).status == ST.WITHDRAWN
    assert ts.champion == env.sub(2).key                              # the only remaining earned entry is crowned


async def test_slot_gate_off_is_rc11_post1(tmp_path, monkeypatch):
    env, T, allc, slot = await _slot_env(tmp_path, monkeypatch, {2: 0.90, 3: 0.99})
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", ())
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert ts.champion in (env.sub(2).key, env.sub(3).key)
    assert not any(_crown_pending(s) for s in ts.submissions.values()) and not any("slot" in e for e in ts.events)


async def test_crowning_makes_all_three_co_champions_former(tmp_path, monkeypatch):
    env, T = await _setup(tmp_path, monkeypatch)
    allc = {1: "a/c1", 2: "a/c2", 3: "a/c3", 6: "a/ch6"}
    env.add(6, "a/ch6", 0.995)
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert ts.champion == env.sub(6).key
    assert {env.sub(u).status for u in (1, 2, 3)} == {ST.FORMER}


# ---------------------------------------------------------------- rc11.post2: the beaten set is one former champion

def _beaten(monkeypatch, *, all_former=True):
    st = case(monkeypatch)
    ts = st.tracks[TRACK]
    new = sub(7, 80.0, status=ST.CHAMPION, crowned=POST + 7200.0)
    ts.submissions[new.key] = new
    ts.submissions[key_of(1)].status = ST.FORMER
    if all_former:
        ts.submissions[key_of(5)].status = ts.submissions[key_of(6)].status = ST.FORMER
    ts.champion = new.key
    return st


@pytest.mark.parametrize("all_former", [True, False])
def test_beaten_set_is_one_former_champion_paid_in_thirds(monkeypatch, all_former):
    st = _beaten(monkeypatch, all_former=all_former)          # False = a state written by the post1 worker
    res = leaders(st, (1, 2, 3, 4, 5, 6, 7))
    assert [u for u, _ in res][0] == 7 and res.formset == {u: pytest.approx(1 / 3) for u in (1, 5, 6)}
    assert len({u for u, _ in res} & {1, 5, 6}) == 1             # one place for the set
    assert [u for u, _ in res][1] in (1, 5, 6) and [u for u, _ in res][2] == 2   # place 3 is the opening place-holder
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: True)
    third = 0.30 * 0.17 / 3
    assert w[7] == pytest.approx(0.30 * 0.75) and (w[1], w[5], w[6]) == pytest.approx((third, third, third))
    assert w[2] == pytest.approx(0.30 * 0.08)
    assert math.fsum(w.values()) == pytest.approx(1.0)
    assert rank_shares(res)[7] == 0.75 and sum(rank_shares(res).values()) == pytest.approx(1.0)


def test_beaten_set_unapproved_or_dropped_member_third_is_not_redistributed(monkeypatch):
    st = _beaten(monkeypatch)
    res = leaders(st, (1, 2, 3, 4, 5, 7))                       # uid 6 no longer on the chain
    assert set(res.formset) == {1, 5}
    w = combine_weights(SERVING, None, 0.0, {TRACK: res}, now=NOW, award_ok=lambda t, u: u != 5)
    third = 0.30 * 0.17 / 3
    assert w[1] == pytest.approx(third) and 6 not in w
    assert 5 not in w or w[5] < third                            # unapproved: no share of the set's place


def test_beaten_set_gate_off_and_formset_survive_the_copy(monkeypatch):
    from reax_subnet.competitions import _copy
    st = _beaten(monkeypatch)
    res = leaders(st, (1, 2, 3, 4, 5, 6, 7))
    assert _copy(res).formset == res.formset
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", ())
    off = leaders(st, (1, 2, 3, 4, 5, 6, 7))
    assert off.formset == {} and rank_shares(off) == {u: s for u, s in zip([u for u, _ in off][:3], (0.75, 0.17, 0.08))}
