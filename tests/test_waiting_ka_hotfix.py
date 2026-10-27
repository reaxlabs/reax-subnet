"""Waiting-state keep-alive HOTFIX (Florian 26 Oct 2026: no deregistration just for waiting; job keepalive-hotfix-20261026).

H1 pending / confirmed entries keep the keep-alive until the slot that decides them has run; H3 the provisional window
stays open while the wait is ours; H4 a new entry after a pause opens a new window for its hotkey (uid 67); H5 a champion
awaiting approval keeps it 72 h; H6 engine entries waiting for a measurement / for their Tue-Fri slot. The switch
``WAITING_KA_HOTFIX = False`` restores the rc11.post4 windows exactly."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from spec_helpers import HOTKEYS, REV, snapshot
from test_keepalive import DF, _snap, _sub
from test_provisional_ka import BLOCK, _p, _snapr

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet.chain import Neuron
from reax_subnet.engcomp import Submission, TrackState
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST
from reax_subnet.spec.commitment import SpecCommitment
from reax_subnet.spec.evaluator import compute_keepalive, compute_provisional

pytestmark = pytest.mark.usefixtures("rc5_entry_floors", "release_slots")
H = 3600.0


@pytest.fixture
def release_slots(monkeypatch):
    """The release's Mon/Thu generalist and Tue/Fri engine slots (conftest clears them for the older tests)."""
    import conftest
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", conftest.RELEASE_CROWN_SLOTS_UTC)
    if hasattr(conftest, "RELEASE_ENGINE_CROWN_SLOTS_UTC"):     # rc11.post5+; the H6 slot rows come from the state anyway
        monkeypatch.setattr(CS, "ENGINE_CROWN_SLOTS_UTC", conftest.RELEASE_ENGINE_CROWN_SLOTS_UTC)


def _ts(*a):
    return datetime(*a, tzinfo=timezone.utc).timestamp()


# Thu 29 Oct 2026 15:00Z: just after a slot; the next slot is Mon 2 Nov 14:30Z
THU = _ts(2026, 10, 29, 15, 0)
MON_SLOT = _ts(2026, 11, 2, 14, 30)


def _ka(st, subs, now):
    return [u for u, _ in compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=now)[DF]]


def _pending(st, uid, first_pass):
    s = _sub(st, uid, ST.PENDING_CONFIRM, 30.0)
    s.first_pass_ts = first_pass
    s.confirm_after_ts = first_pass + M.CONFIRM_DELAY_S
    return s


def test_release_constants():
    assert CS.WAITING_KA_HOTFIX is True
    assert (CS.PENDING_KA_MAX_HOURS, CS.CHAMPION_APPROVAL_KA_HOURS, CS.PROVISIONAL_KA_OUR_WAIT_HOURS) == (216.0, 72.0, 96.0)
    assert (CS.PROVISIONAL_KA_UNVERIFIED_HOURS, CS.PROVISIONAL_KA_UNVERIFIED_MAX) == (24.0, 16)
    assert (CS.KA_REWINDOW_HOURS, CS.ENGINE_KA_WAIT_HOURS, CS.ENGINE_SLOT_KA_MAX_HOURS) == (72.0, 24.0, 120.0)
    # the existing caps are untouched
    assert (CS.KEEPALIVE_MAX_ENTRIES, CS.KEEPALIVE_MAX_SHARE, CS.PROVISIONAL_KA_MAX_SHARE, CS.ENGINE_KA_MAX_SHARE) == \
        (8, 0.25, 0.05, 0.05)
    assert (CS.PROVISIONAL_KA_MAX_HOURS, CS.ENGINE_KA_MAX_HOURS, E.KEEPALIVE_HOURS) == (12.0, 12.0, 24.0)


# ------------------------------------------------------------------------------------------------ H1 generalist pending
def test_h1_confirmed_entry_keeps_the_keepalive_until_its_slot_has_decided_it():
    st = ST.SpecState()
    s = _pending(st, 2, THU)
    s.history.append({"kind": "crown_pending", "ts": THU + 13 * H})          # confirmed Fri 04:00Z, waits for Mon
    assert _ka(st, [s], THU + 23 * H) == [2]
    assert _ka(st, [s], MON_SLOT - 2 * H) == [2]                               # 3.4 days after the first pass: was 0
    assert _ka(st, [s], MON_SLOT + CS.SLOT_CLOCK_S - 60) == [2]                # the slot clock still runs
    assert _ka(st, [s], MON_SLOT + CS.SLOT_CLOCK_S + 60) == [2]                # slot did not decide it (host fault): next slot
    s.status = ST.NOT_BETTER                                                     # decided: ends at once
    assert _ka(st, [s], MON_SLOT + CS.SLOT_CLOCK_S + 60) == []


def test_h1_confirmation_delayed_by_us_keeps_it_until_the_k1a_slot_and_beyond_while_undecided():
    st = ST.SpecState()
    tue = _ts(2026, 10, 27, 10, 0)                                              # due Tue 22:00Z -> K1-a slot Thu 29 14:30Z
    s = _pending(st, 2, tue)
    thu_slot = _ts(2026, 10, 29, 14, 30)
    assert _ka(st, [s], tue + 30 * H) == [2]                                    # the confirmation has not run yet
    assert _ka(st, [s], thu_slot + CS.SLOT_CLOCK_S - 60) == [2]
    assert _ka(st, [s], thu_slot + CS.SLOT_CLOCK_S + 60) == [2]                # its confirmation did not run: next slot
    s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 401", ST.WAIT_MINER         # ... unless the miner's repo is gone
    assert _ka(st, [s], thu_slot + CS.SLOT_CLOCK_S + 60) == []


def test_h1_due_after_the_k1a_cutoff_waits_for_the_following_slot():
    st = ST.SpecState()
    s = _pending(st, 2, _ts(2026, 10, 29, 2, 0))                               # due Thu 14:00Z > 13:30Z cutoff -> Mon
    assert _ka(st, [s], MON_SLOT) == [2]


def test_h1_void_slot_reanchors_at_the_next_slot_within_the_cap():
    st = ST.SpecState()
    first = _ts(2026, 10, 27, 10, 0)
    s = _pending(st, 2, first)
    s.history += [{"kind": "crown_pending", "ts": first + 13 * H}, {"kind": "crown_done", "ts": THU}]  # slot void, back on the path
    assert _ka(st, [s], MON_SLOT) == [2]
    assert _ka(st, [s], first + CS.PENDING_KA_MAX_HOURS * H + 60) == []          # the hard cap after the first pass


def test_h1_hard_cap_after_the_first_pass():
    st = ST.SpecState()
    first = _ts(2026, 10, 27, 10, 0)
    s = _pending(st, 2, first)
    s.history.append({"kind": "crown_pending", "ts": first, "k1a": True, "slot": first + 30 * 24 * H})   # absurd slot
    assert _ka(st, [s], first + CS.PENDING_KA_MAX_HOURS * H - 60) == [2]
    assert _ka(st, [s], first + CS.PENDING_KA_MAX_HOURS * H + 60) == []


def test_h1_decided_entry_loses_it_at_once():
    st = ST.SpecState()
    s = _pending(st, 2, THU)
    s.status = ST.NOT_BETTER                                                     # the confirmation failed
    assert _ka(st, [s], THU + 2 * H) == []


# ------------------------------------------------------------------------------------------------ H5 champion approval
def test_h5_champion_awaiting_approval_keeps_it_72_hours():
    st = ST.SpecState()
    c = _sub(st, 1, ST.CHAMPION, 40.0)
    c.champion_since_ts = THU
    assert _ka(st, [c], THU + 25 * H) == [1]
    assert _ka(st, [c], THU + 71.9 * H) == [1]
    assert _ka(st, [c], THU + 72.1 * H) == []


def test_h5_co_champion_awaiting_approval_keeps_it_72_hours():
    st = ST.SpecState()
    m = _sub(st, 2, ST.PENDING_CONFIRM, 39.0)
    assert E._keepalive_end(m, THU, True, THU + 30 * H) == THU + CS.CHAMPION_APPROVAL_KA_HOURS * H


def test_switch_off_restores_the_rc11_post4_windows(monkeypatch):
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    st = ST.SpecState()
    c = _sub(st, 1, ST.CHAMPION, 40.0)
    c.champion_since_ts = THU
    s = _pending(st, 2, THU)
    s.history.append({"kind": "crown_pending", "ts": THU + 13 * H})
    assert _ka(st, [c, s], THU + 23 * H) == [1, 2]
    assert _ka(st, [c, s], THU + 25 * H) == []


# ------------------------------------------------------------------------------------------------ G1 committed = kept alive
CB_NOW = 1_000_000                                                              # the snapshot block of the G1 tests


def _prov(st, subs, now, track=DF, **kw):
    kw.setdefault("block", CB_NOW)
    return compute_provisional(st, [track], _snapr(subs, **kw), now=now)[track]


def _commit_block(hours_ago):
    return CB_NOW - int(hours_ago * H / E.BLOCK_SECONDS)


def _neuron(uid, track=DF, repo=None, cb=0, coldkey=None, reg=1):
    repo = repo or f"m/m{uid}"
    return Neuron(uid, HOTKEYS[uid], coldkey or f"c{uid}", False, 0.0, None, None,
                  commitment=SpecCommitment(track, repo, REV).text(), commitment_block=cb, registered_block=reg)


def test_g1_a_commitment_the_evaluator_has_not_seen_yet_is_kept_alive_from_its_block():
    st = ST.SpecState()
    snap = snapshot(CB_NOW, [_neuron(7, cb=_commit_block(0.5))])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [7]                 # the first weight set after the commit
    late = snapshot(CB_NOW, [_neuron(7, cb=_commit_block(12.1))])
    assert compute_provisional(st, [DF], late, now=THU)[DF] == [7]                # unverified: 24 h
    later = snapshot(CB_NOW, [_neuron(7, cb=_commit_block(24.1))])
    assert compute_provisional(st, [DF], later, now=THU)[DF] == []
    unknown = snapshot(CB_NOW, [_neuron(7, cb=0)])
    assert compute_provisional(st, [DF], unknown, now=THU)[DF] == []              # no block, not seen: fail closed


def _queued(st, uid, *, fetched=True, private=False, first_seen=None, **kw):
    s = _p(st, uid, **kw)
    s.provisional_since_ts = None                                                # the hotfix does not need the stored start
    s.first_seen_ts = first_seen if first_seen is not None else THU - 1000 * H
    if not fetched:
        s.hash_named, s.delta_checked = "", False
    if private:
        s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 401 Client Error", ST.WAIT_MINER
    return s


def test_g1_a_401_that_turns_out_public_has_no_gap():
    st = ST.SpecState()
    s = _queued(st, 1, fetched=False, private=True)
    cb = {1: _commit_block(3)}
    assert _prov(st, [s], THU, cblocks=cb, regs={1: 1}) == [1]                    # private at our fetch: 24 h
    s.reason, s.wait_kind, s.hash_named, s.delta_checked = "", "", "h1", True       # public, fetched and checked
    for hours in (10, 30, 90):
        assert _prov(st, [s], THU, cblocks={1: _commit_block(hours)}, regs={1: 1}) == [1], hours
    assert _prov(st, [s], THU, cblocks={1: _commit_block(96.5)}, regs={1: 1}) == []   # safety bound


def test_g1_our_queue_our_boxes_our_base_download_keep_it_until_the_first_draw():
    st = ST.SpecState()
    a = _queued(st, 1, fetched=False)                                            # in our fetch queue, no failed fetch
    b = _queued(st, 2)                                                           # fetched + checked, waiting for a box
    c = _queued(st, 3, status=ST.PENDING_VERIFY, checked=False)                  # our base download
    cb = {u: _commit_block(40) for u in (1, 2, 3)}
    assert _prov(st, [a, b, c], THU, cblocks=cb, regs={1: 1, 2: 1, 3: 1}) == [1, 2, 3]
    b.score, b.status = 31.0, ST.PENDING_CONFIRM                                 # its first draw decided: keep-alive takes over
    assert _prov(st, [a, b, c], THU, cblocks=cb, regs={1: 1, 2: 1, 3: 1}) == [1, 3]


@pytest.mark.parametrize("status", [ST.REJECTED, ST.NOT_BETTER, ST.WITHDRAWN, ST.PENDING_CONFIRM, ST.CHAMPION, ST.FORMER])
def test_g1_a_decided_entry_gets_no_provisional(status):
    st = ST.SpecState()
    s = _queued(st, 1)
    s.status = status
    assert _prov(st, [s], THU, cblocks={1: _commit_block(1)}) == []


def test_g1_miner_side_waits_get_the_24_hour_unverified_window():
    st = ST.SpecState()
    a = _queued(st, 1, status=ST.PENDING_VERIFY, checked=False)
    a.wait_kind, a.hash_named = ST.WAIT_MINER, ""                                # declared base not on the hub
    b = _queued(st, 2, fetched=False, private=True)
    cb = {1: _commit_block(24.2), 2: _commit_block(24.2)}
    assert _prov(st, [a, b], THU, cblocks=cb) == []
    assert _prov(st, [a, b], THU, cblocks={1: _commit_block(23), 2: _commit_block(23)}) == [1, 2]


def test_g1_unverified_recommits_cannot_chain_windows_but_a_verified_entry_always_gets_its_own():
    st = ST.SpecState()
    old = _queued(st, 1, repo="m/old", first_seen=THU - 30 * H)   # its unverified window: 30 h ago, over
    old.status = ST.WITHDRAWN
    new = _queued(st, 1, repo="m/new", fetched=False, private=True, first_seen=THU - 1 * H)
    cb = {1: _commit_block(1)}
    assert _prov(st, [new], THU, cblocks=cb) == []                               # its hotkey's unverified window began 30 h ago
    new.reason, new.wait_kind, new.hash_named, new.delta_checked = "", "", "h1", True
    assert _prov(st, [new], THU, cblocks=cb) == [1]                              # verified: from its own commitment
    old.first_seen_ts = THU - 73 * H
    new.reason, new.wait_kind, new.hash_named, new.delta_checked = E.WAITING_FOR_REPO, ST.WAIT_MINER, "", False
    assert _prov(st, [new], THU, cblocks=cb) == [1]                              # > 72 h later: a new unverified window


def test_g1_two_per_coldkey_and_track_and_the_open_limit():
    st = ST.SpecState()
    a, b = _queued(st, 1, coldkey="ck"), _queued(st, 2, coldkey="ck")
    cb = {1: _commit_block(2), 2: _commit_block(1)}
    assert _prov(st, [a, b], THU, cblocks=cb, cold={1: "ck", 2: "ck"}) == [1, 2]  # both allowed open entries


def _uid67(st, fetched):
    """The live state of 26 Oct 2026 14:51Z: a former risk-assessment champion (hotkey window from 21 Oct 08:38Z) commits a
    generalist entry on 26 Oct 10:54Z; our fetch saw 401 at 14:43Z, the repo was public at 14:47Z."""
    hk = HOTKEYS[5]
    ra = ST.Submission(key=f"{hk}|{SpecCommitment('risk_assessment', 'g/ra', REV).text()}", uid=5, hotkey=hk, coldkey="c5",
                       track_id="risk_assessment", repo="g/ra", revision=REV, first_seen_block=1,
                       first_seen_ts=_ts(2026, 10, 19, 4, 31), status=ST.FORMER, score=60.0, delta_checked=True)
    ra.provisional_since_ts = None
    st.track("risk_assessment").submissions[ra.key] = ra
    g = _queued(st, 5, fetched=fetched, private=not fetched, first_seen=_ts(2026, 10, 26, 10, 54))
    g.provisional_since_ts = _ts(2026, 10, 21, 8, 38)                           # inherited per hotkey (the defect)
    return g


def _snap67(at):
    cb = CB_NOW - int((at - _ts(2026, 10, 26, 10, 54)) / E.BLOCK_SECONDS)
    return {"cblocks": {5: cb}, "regs": {5: 1}}


@pytest.mark.parametrize("hhmm", [(11, 30), (14, 43), (14, 53), (15, 17), (22, 50)])
def test_g1_uid67_would_have_been_kept_alive_through_the_401(hhmm):
    st = ST.SpecState()
    g = _uid67(st, fetched=False)
    at = _ts(2026, 10, 26, *hhmm)
    assert _prov(st, [g], at, **_snap67(at)) == [5]


def test_g1_uid67_after_the_fetch_until_the_first_draw_and_rc11_post4_never_paid_it(monkeypatch):
    st = ST.SpecState()
    g = _uid67(st, fetched=True)
    at = _ts(2026, 10, 28, 12, 0)                                                 # 49 h after the commitment, still waiting on us
    assert _prov(st, [g], at, **_snap67(at)) == [5]
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    at = _ts(2026, 10, 26, 15, 0)
    assert _prov(st, [g], at, **_snap67(at)) == []                                # the defect: window from 21 Oct


def test_g1_topup_pays_a_waiting_hotkey_with_a_small_serving_weight(monkeypatch):
    serving = {11: 0.4, 12: 0.3, 13: 0.2, 14: 0.1, 20: 0.00001}                 # 20 serves a little and waits
    pure = CS.combine_weights({u: w for u, w in serving.items() if u != 20}, None, 0.0, {}, owner_uid=0,
                              provisional={"generalist": [20]})
    w = CS.combine_weights(serving, None, 0.0, {}, owner_uid=0, provisional={"generalist": [20]})
    assert pure[20] > 0 and w[20] == pytest.approx(pure[20], rel=0.05)          # topped up to the same level
    assert sum(w.values()) == pytest.approx(1.0)
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    old = CS.combine_weights(serving, None, 0.0, {}, owner_uid=0, provisional={"generalist": [20]})
    assert old[20] < 0.001 * pure[20] * 1000 and old[20] < pure[20] / 100       # rc11.post4: left at its tiny serving weight


# ------------------------------------------------------------------------------------------------ H6 engine
DA, DB, DC = (hashlib.sha256(x.encode()).hexdigest() for x in "abc")
BPH = int(3600 / EC.ENGINE_BLOCK_SECONDS)                                       # 300 blocks per hour
B0 = 100_000


def _esub(uid, hotkey, digest, block):
    return Submission(uid, hotkey, "s1-fast", "ghcr.io/a/e", digest, block)


@pytest.fixture
def no_opening(monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_TRACK_START_BLOCK", None)


def test_h6_first_measurement_wait_is_24_hours(no_opening, monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_KA_UNTIL_MEASURED", False)       # the 24 h window of H6; H6b tests: until measured
    w = _esub(4, "w1", DA, B0)
    st = TrackState(first_seen={DA: (B0, "w1")}, owners={DA: "w1"})
    assert EC.waiting_uids(st, [w], B0 + 20 * BPH, {}) == [4]                   # was gone after 12 h
    assert EC.waiting_uids(st, [w], B0 + 25 * BPH, {}) == []


def test_h6_pending_challenger_is_kept_while_pending_up_to_the_cap(no_opening):
    w = _esub(4, "w1", DA, B0)
    first = B0 + 2 * BPH
    st = TrackState(first_seen={DA: (B0, "w1")}, owners={DA: "w1"})
    st.challenger = {"digest": DA, "hotkey": "w1", "measurements": [[first, 2.0, "s1"], [first + 20 * BPH, 2.0, "s2"]],
                     "confirmed_at": "2026-10-27T10:00:00+00:00", "slot_utc": "2026-10-30T14:30:00+00:00"}
    for hours in (0, 30, 90, 119):
        assert EC.waiting_uids(st, [w], first + int(hours * BPH), {}) == [4], hours
    assert EC.waiting_uids(st, [w], first + 121 * BPH, {}) == []
    st.challenger = None                                                         # decided: crowned or dropped
    st.results[DA] = EC.EvalResult(DA, "s1-fast", True, score=2.0, block=first, session="s1")
    assert EC.waiting_uids(st, [w], first + 30 * BPH, {}) == []


def test_g1_engine_every_digest_waits_in_its_own_window(no_opening, monkeypatch):
    """uid 57, 26 Oct: a former champion's new image (its hotkey's first digest 67 h earlier) waited for our next batch at 0."""
    monkeypatch.setattr(CS, "ENGINE_KA_UNTIL_MEASURED", False)       # the 24 h window of H6; H6b tests: until measured
    new = _esub(4, "w1", DB, B0 + 67 * BPH)
    st = TrackState(first_seen={DA: (B0, "w1"), DB: (B0 + 67 * BPH, "w1")}, owners={DA: "w1"})
    st.results[DA] = EC.EvalResult(DA, "s1-fast", True, score=1.0, block=B0 + BPH, session="s")
    assert EC.waiting_uids(st, [new], B0 + 70 * BPH, {}) == [4]
    assert EC.waiting_uids(st, [new], B0 + 67 * BPH + 25 * BPH, {}) == []        # 24 h per digest


def test_g1_engine_no_count_cut_to_zero(no_opening):
    many = [_esub(10 + i, f"h{i}", hashlib.sha256(bytes([i])).hexdigest(), B0) for i in range(9)]
    st = TrackState(first_seen={s.digest: (B0, s.hotkey) for s in many}, owners={s.digest: s.hotkey for s in many})
    assert len(EC.waiting_uids(st, many, B0 + 10, {})) == 9
    serving = {u: 1.0 for u in range(100, 120)}
    w = CS.combine_weights(serving, 1, 1.0, {}, engine_waiting=[s.uid for s in many])
    assert min(w[s.uid] for s in many) > 0
    assert sum(w[s.uid] for s in many) <= CS.ENGINE_KA_MAX_SHARE * CS.profile_for().engine + 1e-12


def test_engine_switch_off_is_the_old_rule(no_opening, monkeypatch):
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    w = _esub(4, "w1", DA, B0)
    st = TrackState(first_seen={DA: (B0, "w1")}, owners={DA: "w1"})
    assert EC.waiting_uids(st, [w], B0 + 11 * BPH, {}) == [4]
    assert EC.waiting_uids(st, [w], B0 + 13 * BPH, {}) == []
def test_f5_our_hub_failure_is_our_wait():
    st = ST.SpecState()
    s = _queued(st, 1, fetched=False)
    s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 503 from the hub", ST.WAIT_INFRA
    assert E._our_wait(s)
    assert _prov(st, [s], THU, cblocks={1: _commit_block(40)}) == [1]           # 96 h, not the 24 h of a private repo


def test_f3_unverified_entries_are_limited_per_track_oldest_first_verified_unlimited(monkeypatch):
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 8)                # the rule, with a cap the 20 test hotkeys can exceed
    monkeypatch.setattr(CS, "KA_GAUNTLET_R2", False)    # critic F3 as shipped; gauntlet F3 (unseen entries not counted): test_ka_gauntlet_r2.py
    st = ST.SpecState()
    dummies = [_neuron(u, cb=_commit_block(20 - u)) for u in range(1, 11)]       # 10 unseen commitments, uid 1 oldest
    assert compute_provisional(st, [DF], snapshot(CB_NOW, dummies), now=THU)[DF] == list(range(1, 9))
    real = [_queued(st, u, first_seen=THU - 2 * H) for u in range(11, 20)]      # 9 verified entries waiting on us
    snap2 = _snapr(real, cblocks={u: _commit_block(2) for u in range(11, 20)}, block=CB_NOW)
    paid = compute_provisional(st, [DF], snapshot(CB_NOW, list(snap2.neurons) + dummies), now=THU)[DF]
    assert set(range(11, 20)) <= set(paid) and len(set(paid) & set(range(1, 11))) == 8   # verified: no count limit


@pytest.mark.parametrize("n_ka", [8, 12, 30, 100])
def test_n1_a_keepalive_crowd_cannot_squeeze_the_provisional_entries_to_zero(n_ka):
    serving = {u: 1.0 for u in range(1000, 1300)}
    alive = {"generalist": [(u, 50.0) for u in range(1, n_ka + 1)]}
    prov = {"generalist": [500, 501, 502, 503, 504]}
    w = CS.combine_weights(serving, None, 0.0, {}, owner_uid=0, keepalive=alive, provisional=prov)
    assert min(w[u] for u in prov["generalist"]) > 0 and min(w[u] for u in range(1, n_ka + 1)) > 0
    share = CS.profile_for().tracks[0]
    total = sum(w[u] for u in list(range(1, n_ka + 1)) + prov["generalist"])
    assert total <= CS.KEEPALIVE_MAX_SHARE * share + 1e-12 and sum(w.values()) == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------ the guarantee, end to end
def _weight_at(st, s, at, cb):
    """The entry's weight in a weight set at ``at``: the validator's pure path (compute_keepalive, compute_provisional,
    combine_weights) over a serving base, the entry's uid a pure entrant out of immunity."""
    snap = _snapr([s], cblocks={s.uid: cb}, regs={s.uid: 1}, block=CB_NOW)
    alive = compute_keepalive(st, [DF], snap, {DF: 0.0}, now=at)
    prov = compute_provisional(st, [DF], snap, now=at)
    serving = {11: 0.4, 12: 0.3, 13: 0.2, 14: 0.1}
    w = CS.combine_weights(serving, None, 0.0, {}, owner_uid=0, keepalive={t: v for t, v in alive.items() if v},
                           provisional={t: v for t, v in prov.items() if v}, award_ok=lambda t, u: False,
                           now=at)
    return w.get(s.uid, 0.0)


@pytest.mark.parametrize("fetch_h,draw_h", [(0.5, 2), (3, 40), (20, 90)])
def test_guarantee_no_zero_weight_set_from_commitment_to_slot(fetch_h, draw_h):
    """Commit Thu 29 Oct 15:00Z (right after a slot), 401 at our first fetch until ``fetch_h``, our queue until ``draw_h``,
    first draw passes, confirmation 12 h later, stored crowning, Mon 2 Nov 14:30Z slot, crowned, award approval pending:
    every hourly weight set until 72 h after the crowning pays the uid."""
    st = ST.SpecState()
    commit = THU
    s = _queued(st, 1, fetched=False, private=True, first_seen=commit + 0.2 * H)
    first_pass = commit + draw_h * H
    crowned = MON_SLOT + 0.5 * H
    t, zero = commit + 0.25 * H, []                                              # the first weight set after the commitment
    while t < crowned + 71 * H:
        if t >= commit + fetch_h * H and s.status == ST.QUEUED and not s.hash_named:
            s.reason, s.wait_kind, s.hash_named, s.delta_checked = "", "", "h1", True
        if t >= first_pass and s.status == ST.QUEUED:
            s.status, s.score, s.nonproc, s.first_pass_ts = ST.PENDING_CONFIRM, 31.0, 99.0, first_pass
            s.confirm_after_ts = first_pass + M.CONFIRM_DELAY_S
        if s.status == ST.PENDING_CONFIRM and t >= s.confirm_after_ts + H and not s.history:
            s.history.append({"kind": "crown_pending", "ts": s.confirm_after_ts + H})
        if t >= crowned and s.status == ST.PENDING_CONFIRM:
            s.status, s.champion_since_ts = ST.CHAMPION, crowned
            s.history.append({"kind": "crown_done", "ts": crowned})
            st.track(DF).champion = s.key
        cb = CB_NOW - int((t - commit) / E.BLOCK_SECONDS)
        if _weight_at(st, s, t, cb) <= 0.0:
            zero.append(round((t - commit) / H, 2))
        t += H
    assert zero == [], f"zero-weight sets at hours after the commitment: {zero}"


def test_guarantee_rc11_post4_had_the_gaps(monkeypatch):
    """The same timeline under the rc11.post4 windows (switch off) has zero-weight sets: the test above can fail."""
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    with pytest.raises(AssertionError, match="zero-weight sets"):
        test_guarantee_no_zero_weight_set_from_commitment_to_slot(3, 40)


# ------------------------------------------------------------------------------------------------ G1 engine
def test_g1_engine_digest_not_recorded_yet_counts_from_its_commitment_block(no_opening, monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_KA_UNTIL_MEASURED", False)       # the 24 h window of H6; H6b tests: until measured
    w = _esub(4, "w1", DA, B0)
    st = TrackState()                                                            # the runner has not seen it yet
    assert EC.waiting_uids(st, [w], B0 + 2 * BPH, {}) == [4]
    assert EC.waiting_uids(st, [w], B0 + 25 * BPH, {}) == []


def test_g1_engine_switch_off_needs_the_runner(no_opening, monkeypatch):
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    assert EC.waiting_uids(TrackState(), [_esub(4, "w1", DA, B0)], B0 + 2 * BPH, {}) == []


# ------------------------------------------------------------------------------------------------ H6b engine (uid 165, 26 Oct)
def test_h6b_immune_engine_entry_window_opens_a_commit_reveal_lead_before_the_immunity_ends(no_opening):
    """uid 165: registered 9365609, immunity to 9372809, image waiting for our measurement: the paying set must land
    >= 2 epochs (720 blocks) before the immunity ends; post7.1 opened the window only 600 blocks before."""
    w = _esub(4, "w1", DA, B0)
    reg = B0 - 1000
    end = reg + CS.CHAIN_IMMUNITY_BLOCKS
    st = TrackState(first_seen={DA: (B0, "w1")}, owners={DA: "w1"})
    assert EC.waiting_uids(st, [w], end - CS.ENGINE_KA_LEAD_BLOCKS - 10, {}, {4: reg}) == []   # immune, too early
    assert EC.waiting_uids(st, [w], end - CS.ENGINE_KA_LEAD_BLOCKS + 10, {}, {4: reg}) == [4]
    assert CS.ENGINE_KA_LEAD_BLOCKS >= 2 * 360 + 360


def test_h6b_image_without_measurement_waits_until_measured_up_to_the_cap(no_opening):
    w = _esub(4, "w1", DA, B0)
    st = TrackState(first_seen={DA: (B0, "w1")}, owners={DA: "w1"})
    for hours in (2, 25, 60, 119):
        assert EC.waiting_uids(st, [w], B0 + hours * BPH, {}) == [4], hours
    assert EC.waiting_uids(st, [w], B0 + 121 * BPH, {}) == []
    st.results[DA] = EC.EvalResult(DA, "s1-fast", True, score=2.0, block=B0, session="s")      # measured: no longer waiting
    assert EC.waiting_uids(st, [w], B0 + 25 * BPH, {}) == []


def test_h6b_not_recorded_digest_and_two_per_coldkey_and_not_owner(no_opening):
    a, b = _esub(4, "w1", DA, B0), _esub(5, "w2", DB, B0 + 10)
    assert EC.waiting_uids(TrackState(), [a, b], B0 + 40 * BPH, {4: "ck", 5: "ck"}) == [4, 5]  # two per coldkey, oldest first
    st = TrackState(first_seen={DA: (B0, "w1")}, owners={DA: "other"})
    assert EC.waiting_uids(st, [a], B0 + 40 * BPH, {}) == []                                   # image bound to another hotkey


def test_h6b_switch_off_is_bit_identical_to_post71(no_opening, monkeypatch):
    w = _esub(4, "w1", DA, B0)
    reg = B0 - 1000
    end = reg + CS.CHAIN_IMMUNITY_BLOCKS
    st = TrackState(first_seen={DA: (B0, "w1")}, owners={DA: "w1"})
    monkeypatch.setattr(CS, "ENGINE_KA_UNTIL_MEASURED", False)
    assert EC.waiting_uids(st, [w], end - CS.PROVISIONAL_KA_LEAD_BLOCKS - 10, {}, {4: reg}) == []
    assert EC.waiting_uids(st, [w], end - CS.PROVISIONAL_KA_LEAD_BLOCKS + 10, {}, {4: reg}) == [4]
    assert EC.waiting_uids(st, [w], end - 1000, {}, {4: reg}) == []                            # 1000 > 600: still closed
    assert EC.waiting_uids(st, [w], B0 + 20 * BPH, {}) == [4]
    assert EC.waiting_uids(st, [w], B0 + 25 * BPH, {}) == []                                   # 24 h again


# ------------------------------------------------------------------------------------------------ beyond the 8 slots
@pytest.mark.parametrize("serving_w", [1.0, 1e-4])                              # share-cap bound / median bound
def test_g1_no_waiting_entry_is_cut_to_zero_by_the_eight_slots(monkeypatch, serving_w):
    serving = {u: serving_w for u in range(100, 120)}
    alive = {"generalist": [(u, 50.0 - u / 100) for u in range(1, 13)]}           # 12 pending entries, 8 slots
    w = CS.combine_weights(serving, None, 0.0, {}, owner_uid=0, keepalive=alive)
    paid = [w.get(u, 0.0) for u in range(1, 13)]
    assert min(paid) > 0 and max(paid) == pytest.approx(min(paid))              # all 12, evenly
    share = CS.profile_for().tracks[0]
    assert sum(paid) <= CS.KEEPALIVE_MAX_SHARE * share + 1e-12                   # the share cap holds
    assert sum(w.values()) == pytest.approx(1.0)
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    w0 = CS.combine_weights(serving, None, 0.0, {}, owner_uid=0, keepalive=alive)
    assert sum(1 for u in range(1, 13) if w0.get(u, 0.0) == 0.0) == 4           # rc11.post4: four cut to zero


# ------------------------------------------------------------------------------------------------ abuse clause (pending Florian)
def _fake(st, uid, reason, *, coldkey, first_seen, status=ST.REJECTED, repo=None):
    s = _queued(st, uid, coldkey=coldkey, first_seen=first_seen, repo=repo or f"m/fake{uid}")
    s.status, s.reason = status, reason
    return s


def test_abuse_a_404_commitment_gets_no_unverified_window_a_401_keeps_it():
    st = ST.SpecState()
    s = _queued(st, 1, fetched=False, private=True)
    assert _prov(st, [s], THU, cblocks={1: _commit_block(2)}) == [1]           # 401: private, 24 h as before
    s.reason = E.WAITING_FOR_REPO + ": repo or revision not found (still private?): 404 Client Error. Revision Not Found"
    assert _prov(st, [s], THU, cblocks={1: _commit_block(2)}) == []            # 404: the revision does not exist


def test_abuse_one_fake_costs_the_coldkey_the_unverified_window_for_30_days_verified_still_paid():
    st = ST.SpecState()
    _fake(st, 2, "no safetensors weights listed", coldkey="ck", first_seen=THU - 10 * 24 * H)
    s = _queued(st, 1, fetched=False, private=True, coldkey="ck")
    assert _prov(st, [s], THU, cblocks={1: _commit_block(2)}, cold={1: "ck"}) == []
    s.reason, s.wait_kind, s.hash_named, s.delta_checked = "", "", "h1", True       # a real, checked model
    assert _prov(st, [s], THU, cblocks={1: _commit_block(2)}, cold={1: "ck"}) == [1]
    s.reason, s.wait_kind, s.hash_named, s.delta_checked = E.WAITING_FOR_REPO + ": 401", ST.WAIT_MINER, "", False
    assert _prov(st, [s], THU + 21 * 24 * H, cblocks={1: _commit_block(2)}, cold={1: "ck"}) == [1]   # after 30 days


def test_abuse_two_fakes_exclude_the_coldkey_from_the_provisional_keepalive():
    st = ST.SpecState()
    _fake(st, 2, "weights are identical to the declared base model", coldkey="ck", first_seen=THU - 5 * 24 * H)
    _fake(st, 3, "hub revision does not match the committed revision", coldkey="ck", first_seen=THU - 3 * 24 * H)
    s = _queued(st, 1, coldkey="ck")                                             # even a verified one
    assert _prov(st, [s], THU, cblocks={1: _commit_block(2)}, cold={1: "ck"}) == []


def test_abuse_clause_off_is_the_g1_rule(monkeypatch):
    monkeypatch.setattr(CS, "KA_ABUSE_CLAUSE", False)
    st = ST.SpecState()
    _fake(st, 2, "no safetensors weights listed", coldkey="ck", first_seen=THU - 24 * H)
    s = _queued(st, 1, fetched=False, private=True, coldkey="ck")
    assert _prov(st, [s], THU, cblocks={1: _commit_block(2)}, cold={1: "ck"}) == [1]
