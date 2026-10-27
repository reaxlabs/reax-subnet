"""v1 -> v2 switch (board #22 #26409, CoS promise check of 25 Oct 2026): the calendar, the rights that carry over (payout bridge
ending in the 17 % former-champion place, the shifted chain, the 14-day decay), one rule set on Fri 30 Oct for dm-all, and the
commitment format. The numbers follow the published calendar: Tue 27 Oct 14:30Z is the last v1 crowning, dm-all is crowned from
Fri 30 Oct 14:30Z, the bridge ends at the latest at Tue 3 Nov 14:30Z."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import competition_split as cs
from reax_subnet import engcomp as EC
from reax_subnet import engcomposite as ecmp
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import Champion, EvalResult, Submission, TrackState, observe, parse_engine_commitment, update_champion
from reax_subnet.engcomposite import default_track, parse_declared

from test_engcomposite import D1, D2, D3, TRACK, _LabelRunner, _snapshot

UTC = timezone.utc
TUE = datetime(2026, 10, 27, 14, 30, tzinfo=UTC)
FRI = datetime(2026, 10, 30, 14, 30, tzinfo=UTC)
END = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)
DA, DB = (hashlib.sha256(x.encode()).hexdigest() for x in ("former-1", "former-2"))
DAY = 7200


def blk(when: datetime) -> int:
    b = EC.block_at_utc(when)
    assert b is not None
    return b


@pytest.fixture(autouse=True)
def _frozen(monkeypatch):
    from conftest import RELEASE_ENGINE_CROWN_SLOTS_UTC

    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", None)
    monkeypatch.setattr(cs, "ENGINE_TRACK_START_BLOCK", 9326382)  # the finney anchor (blocks <-> time for the bridge end)
    monkeypatch.setattr(cs, "ENGINE_TRACK_START_UTC", datetime(2026, 10, 20, 18, 0, tzinfo=UTC))
    monkeypatch.setattr(cs, "ENGINE_CROWN_SLOTS_UTC", RELEASE_ENGINE_CROWN_SLOTS_UTC)


class Approved:
    def approved(self, track, hotkey, sid):
        return True


# ------------------------------------------------------------------------------------------------ 1. calendar
def test_the_release_constants_are_the_published_calendar():
    assert cs.ENGINE_V2_START_UTC == TUE == cs.ENGINE_QUANT_RULE_START_UTC == cs.ENGINE_CROWN_SLOTS_UTC[0]
    assert cs.ENGINE_V1_BRIDGE_END_UTC == END and END.weekday() == 1  # Tuesday slot
    assert FRI in cs.ENGINE_CROWN_SLOTS_UTC and cs.ENGINE_QUANT_CONVERSION_END_UTC == FRI
    assert cs.release_v2_start() == TUE


def _comp(tmp_path, now, bridge=None, start=TUE, gate=None, sub="s", runner=None):
    return EV.EngineCompetition(runner or _LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / sub / "e.json"),
                                composites={"dm-all": TRACK}, start_utc=None, utcnow=lambda: now, award_gate=gate,
                                v1_bridge=bridge, composite_start_utc=start)


async def test_before_the_v2_start_no_dm_all_cycle_runs_and_dm_all_pays_nothing(tmp_path):
    class Boom(_LabelRunner):
        async def preflight(self):
            raise AssertionError("a dm-all cycle started before the v2 start")

    snap = _snapshot([("dm-all", D1)], 500)
    comp = _comp(tmp_path, TUE - timedelta(seconds=1), runner=Boom({}), gate=Approved())
    comp.states["dm-all"].champion = Champion(D1, "hk0", 0, 1.3, 450)  # even a champion in the file earns nothing yet
    assert not comp.composites_active() and not comp.due()
    assert await comp.run_cycle(snap, set()) == {}
    assert comp.weights({5: 1.0}, snap, set()) == {5: 1.0}
    assert comp.submissions(snap, set()) == {"dm-all": []}
    later = _comp(tmp_path, TUE, runner=Boom({}), gate=Approved(), sub="t")  # from 14:30:00Z on: active
    later.states["dm-all"].champion = Champion(D1, "hk0", 0, 1.3, 450)
    assert later.composites_active() and later.due()
    assert later.weights({5: 1.0}, snap, set())[0] > 0


def _challenger_state(track, now):
    p = track.params()
    sub = Submission(7, "hk", p.name, "repo/image", D1, 50)
    st = TrackState(derived={D1: []})
    observe(st, [sub], 100)
    for i in range(p.confirmations):
        st.results[D1] = EvalResult(D1, p.name, True, score=1.3, block=100 + i, session=f"s{i}")
        update_champion(st, [sub], p, 100 + i, now=now)
    return st, sub, p


def test_the_switch_slot_never_crowns_dm_all_and_the_first_v2_crowning_is_friday():
    # confirmed before the switch (cannot happen with the cycle gate, but the slot guard must hold on its own)
    st, sub, p = _challenger_state(TRACK, TUE - timedelta(hours=1))
    assert st.challenger["slot_utc"] == TUE.isoformat()
    assert EC.finalize_engine_slot(st, [sub], p, TUE, 1000) is None and st.champion is None
    # exactly at the switch instant: still the v1 slot, not a v2 crowning
    st, sub, p = _challenger_state(TRACK, TUE)
    assert EC.finalize_engine_slot(st, [sub], p, TUE, 1000) is None
    # measured after the switch: the next slot is Friday, and Friday crowns (with the weights screen)
    st, sub, p = _challenger_state(TRACK, TUE + timedelta(hours=1))
    assert st.challenger["slot_utc"] == FRI.isoformat()
    assert EC.finalize_engine_slot(st, [sub], p, FRI - timedelta(seconds=1), 1000) is None
    st.weights_ok.add(D1)
    assert EC.finalize_engine_slot(st, [sub], p, FRI, 1000).digest == D1
    # the v1 track is untouched: its Tuesday slot still crowns, without a screen (inside the conversion window)
    st1, sub1, p1 = _challenger_state(_V1Like(), TUE - timedelta(hours=1))
    assert EC.finalize_engine_slot(st1, [sub1], p1, TUE, 1000).digest == D1


class _V1Like:
    def params(self):
        return EV.TRACKS["s1-fast"]


# ------------------------------------------------------------------------------------------------ 2. carried rights
def _v1_file(path, h_block):
    """Frozen v1 engine state: H (uid 0, DA..) is the champion since h_block; F1 and F2 are the former champions."""
    st = TrackState()
    p = EV.TRACKS["s1-fast"]
    for d, hk in ((DB, "hk3"), (DA, "hk2"), (D3, "hk0")):
        st.results[d] = EvalResult(d, "s1-fast", True, score=2.0, block=10, session="v1")
        st.first_seen[d] = (5, hk)
    blocks = {DB: blk(datetime(2026, 10, 21, 18, 0, tzinfo=UTC)), DA: blk(datetime(2026, 10, 22, 14, 30, tzinfo=UTC)), D3: h_block}
    for d, hk in ((DB, "hk3"), (DA, "hk2"), (D3, "hk0")):
        st.history.append({"block": blocks[d], "event": "crowned", "digest": d, "hotkey": hk, "score": 2.0})
    st.champion = Champion(D3, "hk0", 0, 2.0, h_block)
    assert p.name == "s1-fast"
    st.slot_results.append({"slot_utc": "2026-10-27T14:30:00+00:00", "event": "crowned"})  # the switch slot is finalised in the frozen file
    path.write_text(json.dumps({"version": 1, "tracks": {"s1-fast": st.to_json()}}))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _chain(uids=(("s1-fast", D3), ("dm-all", D1), ("s1-fast", DA), ("s1-fast", DB))):
    return list(uids)  # uid0 = v1 holder H (hk0), uid1 = a dm-all entry (hk1), uid2 = F1 (hk2), uid3 = F2 (hk3)


H_BLOCK = blk(datetime(2026, 10, 24, 10, 0, tzinfo=UTC))


def _carry_comp(tmp_path, now, sub, crowned_at=None):
    path, sha = _v1_file(tmp_path / f"{sub}-v1.json", H_BLOCK)
    comp = _comp(tmp_path, now, (path, END, sha), gate=Approved(), sub=sub)
    if crowned_at is not None:  # the first dm-all champion C (uid 1)
        b = blk(crowned_at)
        st = comp.states["dm-all"]
        st.champion = Champion(D1, "hk1", 1, 1.3, b)
        st.history.append({"block": b, "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3})
    return comp


def test_the_bridge_pays_the_v1_holder_and_the_v1_chain_until_the_first_v2_crowning(tmp_path):
    now = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)  # F2's 7 days (from F1's crowning on 22 Oct) run to 29 Oct 14:30Z
    comp = _carry_comp(tmp_path, now, "pre")
    w = comp.weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    assert w[0] > w[2] > w[3] > 0 and 1 not in w  # H champion share, F1 17 %, F2 8 % (as v1 pays them)


def test_the_v1_holder_moves_to_the_17_percent_place_as_if_beaten_and_the_chain_shifts(tmp_path):
    crowned = FRI
    now = crowned + timedelta(hours=3)
    comp = _carry_comp(tmp_path, now, "fri", crowned_at=crowned)
    w = comp.weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    # C champion (rank 1), H rank 2 (17 %), F1 rank 3 (8 %), F2 out: as after a normal crowning
    assert set(w) == {5, 0, 1, 2}
    assert w[0] / w[1] == pytest.approx(0.17 / 0.75, rel=1e-6)
    assert w[2] / w[1] == pytest.approx(0.08 / 0.75, rel=1e-6)
    assert comp._bridge_ended  # the first crowning ended the bridge for good
    # H's 7 days run from the crowning of C (the bridge end): paid on day 6, gone after day 7
    inside = crowned + timedelta(days=6, hours=23)
    comp2 = _carry_comp(tmp_path, inside, "in", crowned_at=crowned)
    w2 = comp2.weights({5: 1.0}, _snapshot(_chain(), blk(inside)), set())
    assert 0 in w2 and 1 in w2
    outside = crowned + timedelta(days=7, hours=1)
    comp3 = _carry_comp(tmp_path, outside, "out", crowned_at=crowned)
    w3 = comp3.weights({5: 1.0}, _snapshot(_chain(), blk(outside)), set())
    assert 0 not in w3 and 2 not in w3 and 1 in w3  # H's and F1's windows are over; C keeps its (decayed) champion share


def test_the_14_day_decay_of_the_v1_title_is_not_restarted_by_the_switch(tmp_path):
    # the carried place is a share of the NEW title holder's decayed part: an older C pays H less than a fresh one
    fresh = FRI + timedelta(hours=3)
    late = FRI + timedelta(days=3)
    a = _carry_comp(tmp_path, fresh, "a", crowned_at=FRI).weights({5: 1.0}, _snapshot(_chain(), blk(fresh)), set())
    b = _carry_comp(tmp_path, late, "b", crowned_at=FRI).weights({5: 1.0}, _snapshot(_chain(), blk(late)), set())
    assert a[0] / a[1] == pytest.approx(0.17 / 0.75, rel=1e-6)  # F1 (until 31 Oct 10:00Z) still holds rank 3
    assert b[0] / b[1] == pytest.approx(0.17 / 0.83, rel=1e-6) and 2 not in b  # F1's window is over: rank 1 keeps its 8 %
    assert b[0] < a[0] and b[1] < a[1]  # both parts shrink with C's decay


def test_the_bridge_end_without_a_v2_champion_gives_the_v1_holder_the_17_percent_place(tmp_path):
    for hours, expect in ((1, True), (24 * 7 - 1, True), (24 * 7 + 1, False)):
        now = END + timedelta(hours=hours)
        comp = _carry_comp(tmp_path, now, f"end{hours}")
        w = comp.weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
        assert (0 in w) is expect, hours
        assert 1 not in w and 2 not in w and 3 not in w  # nobody else: F1 and F2 had their 7 days, there is no v2 champion
    now = END + timedelta(hours=1)
    assert _carry_comp(tmp_path, END - timedelta(hours=1), "pre").weights(
        {5: 1.0}, _snapshot(_chain(), blk(END - timedelta(hours=1))), set())[0] > 0  # before the end: still the champion share
    w = _carry_comp(tmp_path, now, "pay").weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    assert w[0] > 0 and w[5] < 1.0


def test_a_v2_crowning_after_the_bridge_end_does_not_restart_the_v1_holders_window(tmp_path):
    crowned = END + timedelta(days=2)  # Thu 5 Nov
    now = END + timedelta(days=7, hours=1)  # H's 7 days (from the bridge end) are over, C's are not
    w = _carry_comp(tmp_path, now, "late", crowned_at=crowned).weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    assert 0 not in w and 1 in w
    mid = END + timedelta(days=3)
    w = _carry_comp(tmp_path, mid, "mid", crowned_at=crowned).weights({5: 1.0}, _snapshot(_chain(), blk(mid)), set())
    assert w[0] / w[1] == pytest.approx(0.17 / 0.83, rel=1e-6)  # H still inside its window (from the bridge end), F1 is not


def test_the_carried_place_needs_the_v1_approval_and_the_frozen_file(tmp_path):
    class OnlyV1Track(Approved):
        asked = []

        def approved(self, track, hotkey, sid):
            self.asked.append((track, hotkey))
            return track == "s1-fast" and hotkey == "hk0" or track == "dm-all" and hotkey == "hk1"

    now = FRI + timedelta(hours=3)
    comp = _carry_comp(tmp_path, now, "gate", crowned_at=FRI)
    comp.award_gate = OnlyV1Track()
    w = comp.weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    assert 0 in w and 2 not in w  # H is approved on its own track (s1-fast), F1 is not
    assert ("s1-fast", "hk0") in OnlyV1Track.asked
    # a file that does not hash to the pin carries nothing (the champion share of C is unaffected)
    path, sha = _v1_file(tmp_path / "x-v1.json", H_BLOCK)
    bad = _comp(tmp_path, now, (path, END, "0" * 64), gate=Approved(), sub="bad")
    st = bad.states["dm-all"]
    st.champion = Champion(D1, "hk1", 1, 1.3, blk(FRI))
    st.history.append({"block": blk(FRI), "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3})
    w = bad.weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    assert 1 in w and 0 not in w and 2 not in w


def test_finney_refuses_v2_with_the_bridge_end_set_and_no_pinned_state(monkeypatch):
    from reax_subnet import competition_split as _cs
    monkeypatch.setattr(_cs, "ENGINE_V1_BRIDGE_STATE_SHA256", None)  # before the pin of the frozen v1 file (27 Oct slot)
    from reax_subnet.config import ConfigError
    from reax_subnet.worker import composite_track_from_env

    real = ecmp.default_track
    monkeypatch.setattr(ecmp, "default_track", lambda cal=None: real(cal, variant="v21"))  # a release-ready track
    assert cs.ENGINE_V1_BRIDGE_END_UTC is not None and cs.ENGINE_V1_BRIDGE_STATE_SHA256 is None
    with pytest.raises(ConfigError, match="not pinned"):
        composite_track_from_env({}, release_mode=True)
    composite_track_from_env({}, release_mode=False)  # development networks are not asked
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_STATE_SHA256", "a" * 64)
    assert composite_track_from_env({}, release_mode=True).name == "dm-all"
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_END_UTC", None)
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_STATE_SHA256", None)
    assert cs.release_bridge_problem() is None


# ------------------------------------------------------------------------------------------------ 3. one rule set on Fri 30 Oct
def test_a_non_compliant_dm_all_entry_cannot_take_the_friday_title():
    st, sub, p = _challenger_state(TRACK, TUE + timedelta(hours=1))
    assert EC.finalize_engine_slot(st, [sub], p, FRI, 1000) is None  # no weights screen (Option A)
    assert st.history[-1]["event"] == "blocked_noncompliant" and st.champion is None
    st.weights_ok.add(D1)
    assert EC.finalize_engine_slot(st, [sub], p, FRI, 1000).digest == D1
    # a sitting dm-all champion is never revoked by the screen
    assert EC.title_eligible(st, D2, FRI - timedelta(seconds=1))  # inside the window nobody needs it


def test_a_dm_all_crown_can_only_happen_at_a_slot():
    st, sub, p = _challenger_state(TRACK, TUE + timedelta(hours=1))
    st.weights_ok.add(D1)
    for when in (FRI - timedelta(days=1), FRI - timedelta(minutes=1)):
        assert update_champion(st, [sub], p, 2000, now=when) is None and st.champion is None  # confirmed, waiting for the slot
        assert EC.finalize_engine_slot(st, [sub], p, when, 2000) is None
    assert EC.finalize_engine_slot(st, [sub], p, FRI, 2000) is not None
    assert not any(e.get("slot_utc") not in (None, FRI.isoformat()) for e in st.history if e.get("event") == "crowned")


def test_the_parent_rule_applies_to_dm_all_at_the_slot(monkeypatch):
    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", datetime(2026, 10, 23, 13, 0, tzinfo=UTC))
    p = TRACK.params()
    child = Submission(7, "hk", p.name, "repo/image", D2, 50)
    parent = Submission(8, "hkp", p.name, "repo/parent", D1, 40)

    def state(child_score):
        st = TrackState(derived={D2: [D1]})
        observe(st, [parent, child], 100)
        st.results[D1] = EvalResult(D1, p.name, True, score=1.5, block=90, session="p")
        st.hold(st.results[D1], p)
        for i in range(p.confirmations):
            st.results[D2] = EvalResult(D2, p.name, True, score=child_score, block=100 + i, session=f"s{i}")
            update_champion(st, [parent, child], p, 100 + i, now=TUE + timedelta(hours=1))
        st.weights_ok.add(D2)
        return st

    weak = state(1.52)  # beats the parent by 1.3 %: below the full margin
    assert weak.challenger is None or EC.finalize_engine_slot(weak, [parent, child], p, FRI, 1000) is None
    assert weak.champion is None
    strong = state(1.9)
    assert EC.finalize_engine_slot(strong, [parent, child], p, FRI, 1000).digest == D2


# ------------------------------------------------------------------------------------------------ 4. commitment format
def test_v2_commitments_are_dm_all_and_a_v1_s1_fast_commitment_does_not_carry(tmp_path):
    ok = f"reax-eng/1:dm-all:ghcr.io/x/e@sha256:{D1}"
    assert parse_engine_commitment(ok) == ("dm-all", "ghcr.io/x/e", D1)
    track = default_track()
    assert track.name == "dm-all" and track.carried == () and track.carried_declaration("s1-fast") is None
    assert parse_declared(None, track) == frozenset(s.name for s in track.slots) and len(track.slots) == 5  # absent label = all five (v20)
    assert "plumb" not in {s.name for s in track.slots}
    comp = EV.EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "e.json"), composites={"dm-all": track},
                                start_utc=None, utcnow=lambda: FRI)
    subs = comp.submissions(_snapshot([("s1-fast", D1), ("dm-all", D2)], 500), set())
    assert [s.digest for s in subs["dm-all"]] == [D2]  # the v1 image has no v2 entry; the miner commits to dm-all anew


# ------------------------------------------------------------------------------------------------ 5. critic22 M1: a context reset
def test_a_dm_all_context_reset_does_not_wipe_the_carried_v1_places(tmp_path):
    v1 = TrackState()
    for d, hk, b in ((DB, "hk3", 100), (DA, "hk2", 200), (D3, "hk0", 300)):
        v1.history.append({"block": b, "event": "crowned", "digest": d, "hotkey": hk, "score": 2.0})
    v1.champion = Champion(D3, "hk0", 0, 2.0, 300)
    plain = TrackState()
    plain.history = [{"block": 400, "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3}]
    plain.champion = Champion(D1, "hk1", 1, 1.3, 400)
    expect = [g["pairs"] for g in EC.former_ranks(EC.carried_view(plain, v1, 1000), "s1-fast", 500, frozenset())]
    assert expect and expect[0] == ((D3, "hk0"),)
    for hist in (
        [{"block": 390, "event": "context_changed", "reason": "x"}, {"block": 400, "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3}],
        [{"block": 400, "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3}, {"block": 450, "event": "context_changed", "reason": "x"}],
    ):
        v2 = TrackState()
        v2.history = hist
        v2.champion = Champion(D1, "hk1", 1, 1.3, 400) if hist[0]["event"] == "context_changed" else None  # a reset drops the champion
        view = EC.carried_view(v2, v1, 1000)
        got = [g["pairs"] for g in EC.former_ranks(view, "s1-fast", 500, frozenset())]
        assert got == expect or (v2.champion is None and got[:3] == expect[:3]), (hist, got)
        assert EC.dethroned_blocks(view)[(D3, "hk0")] == 400  # H was beaten by the first composite crowning, reset or not
    # the composite's OWN title and former champions stay a context reset's business: the reset wipes them, as before
    v2 = TrackState()
    v2.history = [{"block": 400, "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3},
                  {"block": 410, "event": "crowned", "digest": D2, "hotkey": "hk4", "score": 1.4},
                  {"block": 450, "event": "context_changed", "reason": "x"}]
    pairs = [p for g in EC.former_ranks(EC.carried_view(v2, v1, 1000), "s1-fast", 500, frozenset()) for p in g["pairs"]]
    assert (D1, "hk1") not in pairs and (D3, "hk0") in pairs and (DA, "hk2") in pairs


# ------------------------------------------------------------------------------------------------ 6. critic22 M3 / m2: the carried place belongs to the hotkey
def _with(snap, uid, commitment):
    import dataclasses

    ns = tuple(dataclasses.replace(n, commitment=commitment) if n.uid == uid else n for n in snap.neurons)
    return dataclasses.replace(snap, neurons=ns)


def test_a_carried_place_stays_with_the_hotkey_while_it_keeps_s1_fast_or_commits_dm_all(tmp_path):
    other = hashlib.sha256(b"other").hexdigest()
    keep = f"reax-eng/1:s1-fast:ghcr.io/x/e@sha256:{D3}"
    to_dm = f"reax-eng/1:dm-all:ghcr.io/x/e@sha256:{other}"
    to_other_s1 = f"reax-eng/1:s1-fast:ghcr.io/x/e@sha256:{other}"
    now = FRI + timedelta(hours=3)
    for label, commit, paid in (("keep", keep, True), ("dm", to_dm, True), ("othersf", to_other_s1, False),
                                ("none", None, False), ("garbage", "hello", False)):
        comp = _carry_comp(tmp_path, now, f"m3{label}", crowned_at=FRI)
        w = comp.weights({5: 1.0}, _with(_snapshot(_chain(), blk(now)), 0, commit), set())
        assert (0 in w) is paid, label
        assert 1 in w and (2 in w) is True, label  # the others are unaffected
    # the same hotkey rule for the champion share during the bridge (before any dm-all crowning)
    pre = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    for label, commit, paid in (("keep", keep, True), ("dm", to_dm, True), ("othersf", to_other_s1, False), ("none", None, False)):
        comp = _carry_comp(tmp_path, pre, f"m3b{label}")
        w = comp.weights({5: 1.0}, _with(_snapshot(_chain(), blk(pre)), 0, commit), set())
        assert (0 in w) is paid, label
    # a former champion (F1, uid 2) moving to dm-all keeps its 8 % place as well; deregistered (the uid now belongs to another hotkey) it is gone
    comp = _carry_comp(tmp_path, now, "m3f1", crowned_at=FRI)
    assert 2 in comp.weights({5: 1.0}, _with(_snapshot(_chain(), blk(now)), 2, to_dm), set())
    import dataclasses
    snap = _snapshot(_chain(), blk(now))
    snap = dataclasses.replace(snap, neurons=tuple(dataclasses.replace(n, hotkey="newcomer") if n.uid == 2 else n for n in snap.neurons))
    assert 2 not in _carry_comp(tmp_path, now, "m3dereg", crowned_at=FRI).weights({5: 1.0}, snap, set())


def test_a_carried_pair_with_a_v1_copy_strike_earns_no_place(tmp_path, monkeypatch):
    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", datetime(2026, 10, 23, 13, 0, tzinfo=UTC))
    now = FRI + timedelta(hours=3)
    comp = _carry_comp(tmp_path, now, "strike0", crowned_at=FRI)
    assert 2 in comp.weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    path, _sha = _v1_file(tmp_path / "strike-v1.json", H_BLOCK)
    raw = json.loads(open(path).read())
    raw["tracks"]["s1-fast"]["copy_strikes"] = {"hk2": [DA]}
    open(path, "w").write(json.dumps(raw))
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    comp = _comp(tmp_path, now, (path, END, sha), gate=Approved(), sub="strike1")
    st = comp.states["dm-all"]
    st.champion = Champion(D1, "hk1", 1, 1.3, blk(FRI))
    st.history.append({"block": blk(FRI), "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3})
    w = comp.weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    assert 0 in w and 2 not in w


# ------------------------------------------------------------------------------------------------ 7. critic22 M2 / m3: local mode = remote mode
def _local_and_remote(tmp_path, monkeypatch, now, sub="par"):
    import time

    from reax_subnet import engrunner
    from reax_subnet import worker as W
    from reax_subnet.remote import RemoteEngineCompetition

    path, sha = _v1_file(tmp_path / f"{sub}-v1.json", H_BLOCK)
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_STATE_SHA256", sha)
    monkeypatch.setattr(engrunner, "DockerRunner", lambda *a, **k: object())
    monkeypatch.setattr(W, "composite_from_env", lambda env, e, runner: (_LabelRunner({}), TRACK))
    monkeypatch.setattr(W, "composite_references", lambda env, release: {})
    seen = {}

    class Capture:
        def __init__(self, *a, **k):
            seen["a"], seen["k"] = a, k

    monkeypatch.setattr(W, "EngineCompetition", Capture)
    (tmp_path / "w").mkdir(exist_ok=True)
    env = {"REAX_NETUID": "92", "REAX_NETWORK": "finney", "REAX_COMPETITIONS_STATE_DIR": str(tmp_path / sub / "state"),
           "REAX_ENGINE_WEIGHTS_DIR": str(tmp_path / "w"), "REAX_ENGINE_WORKDIR": str(tmp_path / "e"),
           "REAX_ENGINE_COMPOSITE": "1", "REAX_ENGINE_V1_STATE_PATH": path}
    wc = W.load_worker(env, spec=False)
    W.build_local_stack(wc.engine, "", env, award_gate=Approved())
    kw = seen["k"]
    st = TrackState()
    st.champion = Champion(D1, "hk1", 1, 1.3, blk(FRI))
    st.history.append({"block": blk(FRI), "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3})
    state_path = str(tmp_path / sub / "composite.json")
    (tmp_path / sub).mkdir(exist_ok=True)
    open(state_path, "w").write(json.dumps({"version": 1, "saved_at": time.time(), "tracks": {"dm-all": st.to_json()}}))
    kw = {**kw, "utcnow": lambda: now, "start_utc": None, "engine_start_utc": None, "cycle_gate_file": ""}
    kw.pop("cycle_gate_max_age_s", None)
    kw.pop("chain_id", None)
    local = EV.EngineCompetition(seen["a"][0], seen["a"][1], state_path, **kw)
    remote = RemoteEngineCompetition(state_path, composites={"dm-all": TRACK}, start_utc=None, utcnow=lambda: now,
                                     award_gate=Approved(), v1_bridge=W.v1_bridge_from_env(env, state_path, {"dm-all": TRACK}, True),
                                     composite_start_utc=cs.ENGINE_V2_START_UTC)
    remote.refresh_state()
    local.states["dm-all"] = TrackState.from_json(st.to_json())  # the local mode ran the cycles itself: same state
    return local, remote, kw


def test_local_competitions_mode_builds_the_same_bridge_and_pays_the_same_weights_as_the_remote_validator(tmp_path, monkeypatch):
    for label, now in (("fri", FRI + timedelta(hours=3)), ("late", FRI + timedelta(days=3))):
        local, remote, kw = _local_and_remote(tmp_path, monkeypatch, now, sub=label)
        assert kw["v1_bridge"] is not None and kw["v1_bridge"][1] == END and kw["v1_bridge"][2] == cs.ENGINE_V1_BRIDGE_STATE_SHA256
        snap = _snapshot(_chain(), blk(now))
        wl, wr = local.weights({5: 1.0}, snap, set()), remote.weights({5: 1.0}, snap, set())
        assert wl == wr and 0 in wl and (2 in wl) is (label == "fri"), (label, wl, wr)  # carried places paid in both modes, identically


def test_finney_refuses_to_start_when_the_frozen_v1_file_is_missing_or_mismatching(tmp_path, monkeypatch):
    from reax_subnet.config import ConfigError
    from reax_subnet.worker import v1_bridge_from_env

    path, sha = _v1_file(tmp_path / "v1.json", H_BLOCK)
    env = {"REAX_ENGINE_V1_STATE_PATH": path}
    comps = {"dm-all": TRACK}
    sp = str(tmp_path / "e.json")
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_STATE_SHA256", None)  # the release pin is replaced by the fixture's own
    assert v1_bridge_from_env(env, sp, comps, True) is None  # no pin yet: nothing configured (release_bridge_problem refuses elsewhere)
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_STATE_SHA256", sha)
    assert v1_bridge_from_env(env, sp, comps, True) == (path, END, sha)
    assert v1_bridge_from_env(env, sp, None, True) is None
    with pytest.raises(ConfigError, match="unusable"):
        v1_bridge_from_env({"REAX_ENGINE_V1_STATE_PATH": str(tmp_path / "missing.json")}, sp, comps, True)
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_STATE_SHA256", "0" * 64)
    with pytest.raises(ConfigError, match="not the pinned"):
        v1_bridge_from_env(env, sp, comps, True)
    assert v1_bridge_from_env(env, sp, comps, False) is not None  # development networks start; the bridge then fails closed at runtime
    # default location: next to the engine state
    assert v1_bridge_from_env({}, sp, comps, False)[0] == str(tmp_path / "engine-competition-v1.json")


def test_a_frozen_v1_file_without_the_switch_slot_is_unusable(tmp_path, caplog):
    path, _ = _v1_file(tmp_path / "v1.json", H_BLOCK)
    raw = json.loads(open(path).read())

    def pin(r):
        open(path, "w").write(json.dumps(r))
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    ok = pin(raw)
    assert EV.read_frozen_v1(path, ok)[0].champion is not None
    # frozen before the Tue slot: no slot result, no crowning at/after it, no later slot schedule
    early = json.loads(json.dumps(raw))
    early["tracks"]["s1-fast"]["slot_results"] = [{"slot_utc": "2026-10-23T14:30:00+00:00"}]
    sha = pin(early)
    with pytest.raises(ValueError, match="frozen before"):
        EV.read_frozen_v1(path, sha)
    comp = _comp(tmp_path, FRI + timedelta(hours=3), (path, END, sha), gate=Approved(), sub="early")
    comp.states["dm-all"].champion = Champion(D1, "hk1", 1, 1.3, blk(FRI))
    comp.states["dm-all"].history.append({"block": blk(FRI), "event": "crowned", "digest": D1, "hotkey": "hk1", "score": 1.3})
    with caplog.at_level("WARNING"):
        w = comp.weights({5: 1.0}, _snapshot(_chain(), blk(FRI + timedelta(hours=3))), set())
    assert 0 not in w and 2 not in w and 1 in w and "frozen before" in caplog.text
    # a crowning at the Tuesday slot counts, so does a slot schedule that moved past it
    crowned = json.loads(json.dumps(early))
    crowned["tracks"]["s1-fast"]["history"].append({"block": 1, "event": "crowned", "digest": D3, "hotkey": "hk0", "slot_utc": TUE.isoformat()})
    EV.read_frozen_v1(path, pin(crowned))
    later = json.loads(json.dumps(early))
    later["tracks"]["s1-fast"]["next_slot_utc"] = FRI.isoformat()
    EV.read_frozen_v1(path, pin(later))
    closed = json.loads(json.dumps(early))  # the slot passed without a crowning (engslot.close_switch_slot)
    closed["tracks"]["s1-fast"]["history"].append({"block": 1, "event": "slot_closed", "slot_utc": TUE.isoformat(), "crowned": False})
    EV.read_frozen_v1(path, pin(closed))
    stale = json.loads(json.dumps(early))  # ... but only for the switch slot or later
    stale["tracks"]["s1-fast"]["history"].append({"block": 1, "event": "slot_closed", "slot_utc": "2026-10-23T14:30:00+00:00"})
    with pytest.raises(ValueError, match="frozen before"):
        EV.read_frozen_v1(path, pin(stale))


def test_the_v2_switch_flag_is_strict():
    from reax_subnet.competition_split import engine_composite_enabled

    for off in ({}, {"REAX_ENGINE_COMPOSITE": ""}, {"REAX_ENGINE_COMPOSITE": "0"}, {"REAX_ENGINE_COMPOSITE": "false"},
                {"REAX_ENGINE_COMPOSITE": " False "}, {"REAX_ENGINE_COMPOSITE": "no"}, {"REAX_ENGINE_COMPOSITE": "off"}):
        assert engine_composite_enabled(off) is False, off
    for on in ("1", "true", "TRUE", "yes", "on"):
        assert engine_composite_enabled({"REAX_ENGINE_COMPOSITE": on}) is True, on
    for bad in ("2", "enabled", "v2"):
        with pytest.raises(ValueError, match="REAX_ENGINE_COMPOSITE"):
            engine_composite_enabled({"REAX_ENGINE_COMPOSITE": bad})


def test_a_v2_crowning_at_the_bridge_end_instant_pays_like_the_bridge_end(tmp_path):
    # Tue 3 Nov 14:30Z is both a crown slot and the bridge end: C crowned at that instant, H gets the 17 % place once
    now = END + timedelta(hours=1)
    same = _carry_comp(tmp_path, now, "same", crowned_at=END).weights({5: 1.0}, _snapshot(_chain(), blk(now)), set())
    assert 0 in same and 1 in same and 2 not in same and 3 not in same
    assert same[0] / same[1] == pytest.approx(0.17 / 0.83, rel=1e-6)
    late = END + timedelta(days=7, hours=1)  # H's 7 days run from that instant, C keeps the title
    w = _carry_comp(tmp_path, late, "same-late", crowned_at=END).weights({5: 1.0}, _snapshot(_chain(), blk(late)), set())
    assert 0 not in w and 1 in w
    before = END - timedelta(minutes=1)  # one minute earlier the bridge still pays H the champion share
    w = _carry_comp(tmp_path, before, "same-before").weights({5: 1.0}, _snapshot(_chain(), blk(before)), set())
    assert w[0] > 0 and 1 not in w


# ------------------------------------------------------------------------------------------------ 8. critic22 m4: no payout gap at the bridge end
def test_the_bridge_end_block_is_the_first_observed_block_not_the_estimate(tmp_path):
    drift = 40  # the real chain is 40 blocks (8 min) behind the 12 s/block estimate
    est = blk(END)
    now = END + timedelta(minutes=1)
    comp = _carry_comp(tmp_path, now, "gap")
    snap = _snapshot(_chain(), est - drift + 5)  # the first block seen after the end time
    w = comp.weights({5: 1.0}, snap, set())
    assert 0 in w and w[0] > 0  # H is paid from the very first block after the end (the estimate would be in the future)
    first = est - drift + 5
    marker = tmp_path / "gap" / EV.BRIDGE_END_BLOCK_FILE
    assert json.loads(marker.read_text())["block"] == first
    w2 = comp.weights({5: 1.0}, _snapshot(_chain(), first + 100), set())  # later rounds use the persisted block
    assert json.loads(marker.read_text())["block"] == first and w2[0] > 0
    # a restarted validator reads the same block
    again = _carry_comp(tmp_path, now + timedelta(hours=1), "gap")
    assert again._end_block(first + 500) == first
    # without a persisted block (unwritable directory) the smaller of estimate and current block is used
    other = _carry_comp(tmp_path, now, "nogap")
    assert other._end_block(est + 30) == est  # a late look is capped at the estimate
    assert _carry_comp(tmp_path, now, "nogap2")._end_block(est - 30) == est - 30


# ------------------------------------------------------------------------------------------------ 9. critic22 m8: slots without a dict key
def test_a_slot_without_a_reference_digest_key_is_refused(monkeypatch):
    from reax_subnet import config
    from reax_subnet import worker as W

    track = default_track()
    real = dict(config.ENGINE_COMPOSITE_REFERENCE_IMAGES)
    for k in ("mammoth", "jeffan"):  # v20 digests are filled by the lead later: pin placeholders for this test
        monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, k, f"ghcr.io/x/{k}@sha256:{'0' * 64}")
    monkeypatch.delitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "blink")  # key gone: not "None", gone
    assert any("slot blink has no published reference image digest" in r for r in track.release_ready())
    with pytest.raises(Exception, match="blink"):
        W.composite_references({}, True)
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "blink", real["blink"])
    assert W.composite_references({}, True)["blink"] == real["blink"]
