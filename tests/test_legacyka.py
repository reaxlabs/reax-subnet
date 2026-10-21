"""Legacy keep-alive (competition_split.LEGACY_KA_*, rc10; Florian 21 Oct 2026 07:05Z, board #22667): after the DF/RA tracks
end at GENERALIST_START_UTC, their champion and approved former champions get a bounded keep-alive for 7 days."""

import json
import math
import random
from datetime import timedelta

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import remote as RM
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.config import ENGINE_STATE_FILE, REMOTE_MAX_AGE_S, SPEC_STATE_FILE
from reax_subnet.engcomp import Champion, EvalResult
from reax_subnet.engvalidator import EngineCompetition
from reax_subnet.spec import config as SC
from reax_subnet.spec import state as ST
from reax_subnet.spec.commitment import SpecCommitment
from reax_subnet.spec.evaluator import SpecEvaluator, legacy_leaders

START = CS.GENERALIST_START_UTC
AFTER = START + timedelta(seconds=30)
BEFORE = START - timedelta(seconds=1)
END = START + timedelta(days=CS.LEGACY_KA_DAYS)
DF, RA, GEN = "demand_forecasting", "risk_assessment", "generalist"
REV = "c" * 40
D = "ab" * 32
# 20 serving miners with weights 1..20 (median 10.5 of 210): per_entry = 0.05, 0.5 x = 0.025
SERVING = {0: 0.0, **{100 + i: float(i + 1) for i in range(20)}}
# uid -> (track, repo, status, crowned); DF: 20 champion, 21 + 22 former; RA: 30 champion, 31 former
ENTRIES = {20: (DF, "a/df1", ST.CHAMPION, 300.0), 21: (DF, "a/df2", ST.FORMER, 200.0), 22: (DF, "a/df3", ST.FORMER, 100.0),
           30: (RA, "a/ra1", ST.CHAMPION, 300.0), 31: (RA, "a/ra2", ST.FORMER, 200.0)}


def hk(uid):
    return f"hk{uid}"


def commitment(uid, track=None, repo=None):
    t, r, _, _ = ENTRIES[uid]
    return SpecCommitment(track or t, repo or r, REV).text()


def snap(commits=None, extra=(), block=900):
    """All ENTRIES committed as in the state unless ``commits`` overrides {uid: commitment text or None}."""
    commits = {**{u: commitment(u) for u in ENTRIES}, **(commits or {})}
    neurons = [Neuron(0, "owner", "c0", False, 0.0, None, None)]
    neurons += [Neuron(u, hk(u), f"c{u}", False, 0.0, None, None) for u in SERVING if u]
    neurons += [Neuron(u, hk(u), f"c{u}", False, 0.0, None, None, commitment=c) for u, c in commits.items() if c is not None]
    neurons += list(extra)
    return Snapshot(block=block, neurons=tuple(neurons), owner_hotkey="owner")


def spec_state(path=None, *, gen=None):
    st = ST.SpecState()
    for uid, (track, repo, status, crowned) in ENTRIES.items():
        ts = st.track(track)
        key = f"{hk(uid)}|{commitment(uid)}"
        ts.submissions[key] = ST.Submission(key=key, uid=uid, hotkey=hk(uid), coldkey=f"c{uid}", track_id=track, repo=repo,
                                            revision=REV, first_seen_block=1, first_seen_ts=1.0, status=status,
                                            score=40.0, champion_since_ts=crowned, delta_checked=True)
        if status == ST.CHAMPION:
            ts.champion = key
    if gen is not None:   # a generalist champion (uid, repo)
        uid, repo = gen
        ts = st.track(GEN)
        c = SpecCommitment(GEN, repo, REV).text()
        key = f"{hk(uid)}|{c}"
        ts.submissions[key] = ST.Submission(key=key, uid=uid, hotkey=hk(uid), coldkey=f"c{uid}", track_id=GEN, repo=repo,
                                            revision=REV, first_seen_block=1, first_seen_ts=1.0, status=ST.CHAMPION,
                                            score=50.0, champion_since_ts=400.0, delta_checked=True)
        ts.champion = key
    if path is not None:
        ST.save(st, str(path))
        raw = json.load(open(path))
        raw["saved_at"] = int(AFTER.timestamp())
        json.dump(raw, open(path, "w"))
    return st


class Gate:
    """award.AwardGate stand-in: approved (track, hotkey) pairs."""

    def __init__(self, *ok):
        self.ok = set(ok)

    def approved(self, track, hotkey, sub_id):
        return (track, hotkey) in self.ok


class _R:
    binds_hotkey = True


def system(tmp_path, now, gate=None, wall=None, gen=None):
    sp = tmp_path / SPEC_STATE_FILE
    spec_state(sp, gen=gen)
    ep = tmp_path / ENGINE_STATE_FILE
    comp = EngineCompetition(_R(), {"s1-fast": "ghcr.io/r/ref@sha256:" + D}, str(ep), utcnow=lambda: now[0])
    comp.save()
    raw = json.load(open(ep))
    raw["saved_at"] = int(AFTER.timestamp())
    json.dump(raw, open(ep, "w"))
    clock = wall or (lambda: now[0].timestamp())
    adapter = RM.RemoteSpecAdapter(RM.RemoteSpecEvaluator(str(sp), clock=clock), free_mib=None)
    eng = RM.RemoteEngineCompetition(str(ep), specialist=adapter, utcnow=lambda: now[0], wall=clock, award_gate=gate)
    return eng, adapter, sp


def pay(eng, adapter, s):
    adapter.refresh(s)
    return eng.weights(SERVING, s, {0})


def legacy_paid(w):
    return {u: v for u, v in w.items() if u in ENTRIES}


@pytest.fixture(autouse=True)
def _floors_off(monkeypatch):
    for k in list(CS.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, k, None)


# ------------------------------------------------------------------------------------------------ the window
def test_window_and_tracks():
    assert CS.legacy_ka_tracks(BEFORE) == ()
    assert CS.legacy_ka_tracks(START) == (DF, RA)
    assert CS.legacy_ka_tracks(END - timedelta(microseconds=1)) == (DF, RA)
    assert CS.legacy_ka_tracks(END) == ()
    assert CS.active_tracks(BEFORE) == [DF, RA] and CS.active_tracks(AFTER) == [GEN]
    assert CS.LEGACY_KA_MAX_ENTRIES >= len(CS.LEGACY_KA_TRACKS) * SC.MAX_LEADERS


def test_before_the_switch_nothing_new_is_paid(tmp_path):
    now = [BEFORE]
    eng, adapter, _ = system(tmp_path, now, Gate((DF, hk(21))))
    w = pay(eng, adapter, snap())
    # DF/RA are still active: only the normal ranks with the award gate (21 approved, rank 2 of DF), nothing on top
    assert set(legacy_paid(w)) == {21} and w[21] == pytest.approx(0.15 * 0.17)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CS, "LEGACY_KA_TRACKS", ())
        mp.setattr(RM, "LEGACY_KA_TRACKS", ())
        assert repr(pay(eng, adapter, snap())) == repr(w)


# ------------------------------------------------------------------------------------------------ who is paid
def test_after_the_switch_champions_and_approved_formers_are_paid(tmp_path):
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate((DF, hk(21))))
    w = pay(eng, adapter, snap())
    got = legacy_paid(w)
    assert set(got) == {20, 21, 30}                         # 22 and 31 are not approved
    each = min(CS.LEGACY_KA_FRACTION * 0.05, CS.LEGACY_KA_MAX_SHARE / 3)
    assert each == pytest.approx(0.01)                     # the total cap binds: 3 x 0.025 > 0.03
    for u in got:
        assert got[u] == pytest.approx(each)
    # from the inference part only: every serving uid keeps (1 - 0.03) of its weight
    for u in range(100, 120):
        assert w[u] == pytest.approx(SERVING[u] / 210.0 * (1 - 3 * each))
    assert sum(w.values()) == pytest.approx(1.0)


def test_unapproved_champion_is_still_paid_and_without_gate_every_former(tmp_path):
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate())
    assert set(legacy_paid(pay(eng, adapter, snap()))) == {20, 30}
    eng, adapter, _ = system(tmp_path, now, None)          # development network: no award gate
    assert set(legacy_paid(pay(eng, adapter, snap()))) == {20, 21, 22, 30, 31}


def test_amount_without_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "LEGACY_KA_MAX_SHARE", 1.0)
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate())
    got = legacy_paid(pay(eng, adapter, snap()))
    assert got[20] == pytest.approx(0.025) and got[30] == pytest.approx(0.025)   # 0.5 x median serving


def test_champion_withdrew_the_first_former_is_not_rank_one_for_the_rule(tmp_path):
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate())
    # 20 no longer commits: 21 is rank 1 of DF but not the champion, so it needs approval
    s = snap({20: None})
    assert set(legacy_paid(pay(eng, adapter, s))) == {30}
    st = spec_state()
    assert legacy_leaders(st, CS.LEGACY_KA_TRACKS, s)[DF] == [(21, False), (22, False)]


def test_window_end_exact_edge(tmp_path):
    now = [END - timedelta(seconds=1)]
    eng, adapter, _ = system(tmp_path, now, Gate(), wall=lambda: AFTER.timestamp())
    assert set(legacy_paid(pay(eng, adapter, snap()))) == {20, 30}
    now[0] = END
    w = pay(eng, adapter, snap())
    assert legacy_paid(w) == {} and w == SERVING            # nothing to pay: exactly the serving weights


def test_hotkey_that_commits_to_another_track_or_repo_drops_out(tmp_path):
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate((DF, hk(21)), (RA, hk(31))))
    s = snap({20: commitment(20, track=GEN), 21: commitment(21, repo="a/new"), 30: commitment(30, track=RA, repo="x/y"),
              31: commitment(31, track=DF)})
    assert legacy_paid(pay(eng, adapter, s)) == {}
    # deregistered (hotkey gone) as well
    assert legacy_paid(pay(eng, adapter, snap({20: None, 21: None, 22: None, 30: None, 31: None}))) == {}


def test_uid_with_serving_weight_or_other_pay_gets_none(tmp_path):
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate((GEN, hk(30))), gen=(30, "g/x"))
    s = snap({30: SpecCommitment(GEN, "g/x", REV).text()})   # 30 now the generalist champion (another track)
    w = pay(eng, adapter, s)
    assert set(legacy_paid(w)) == {20, 30}
    assert w[30] == pytest.approx(0.30 * 0.75)               # paid as generalist rank 1 only, no legacy KA on top
    # combine_weights level: a serving uid, a generalist rank and the engine champion are left out
    base = {**{u: v for u, v in SERVING.items()}, 20: 5.0}
    out = CS.combine_weights(base, 30, 1.0, {GEN: [(31, 50.0)]}, now=AFTER.timestamp(), owner_uid=0,
                             legacy_keepalive=[20, 30, 31, 0, 22])
    ref = CS.combine_weights(base, 30, 1.0, {GEN: [(31, 50.0)]}, now=AFTER.timestamp(), owner_uid=0,
                             legacy_keepalive=[22])
    assert out == ref and 0 not in out


def test_one_payment_per_uid_order_and_entry_cap(monkeypatch):
    base = dict(SERVING)
    out = CS.combine_weights(base, None, 0.0, {}, now=AFTER.timestamp(), owner_uid=0, legacy_keepalive=[7, 7, 8])
    assert out[7] == pytest.approx(out[8]) and out[7] == pytest.approx(0.015 / (1.0))
    many = list(range(500, 520))
    out = CS.combine_weights(base, None, 0.0, {}, now=AFTER.timestamp(), owner_uid=0, legacy_keepalive=many)
    assert [u for u in out if u >= 500] == many[:CS.LEGACY_KA_MAX_ENTRIES]
    assert math.fsum(out[u] for u in many[:CS.LEGACY_KA_MAX_ENTRIES]) == pytest.approx(CS.LEGACY_KA_MAX_SHARE)
    assert CS.legacy_ka_uids({DF: [(1, True), (2, False)], RA: [(1, True), (3, False)]}, (DF, RA),
                             lambda t, u: u == 3) == [1, 3]


def test_burn_and_engine_share_are_untouched():
    base = {0: 0.5, **{100 + i: float(i + 1) for i in range(20)}}
    plain = CS.combine_weights(base, 7, 1.0, {}, now=AFTER.timestamp(), owner_uid=0, burn_fraction=0.4)
    out = CS.combine_weights(base, 7, 1.0, {}, now=AFTER.timestamp(), owner_uid=0, burn_fraction=0.4,
                             legacy_keepalive=[8])
    assert out[7] == pytest.approx(plain[7]) == pytest.approx(0.20 * 0.6)    # engine share of the miner part
    assert out[8] == pytest.approx(CS.LEGACY_KA_MAX_SHARE * 0.6)             # cap of the miner part (burn 0.4)


def test_every_approved_former_counts_not_only_the_three_ranked(tmp_path, monkeypatch):
    """Live case of 21 Oct: RA champion + 4 formers; the 3rd former by crowning time (approved) is outside MAX_LEADERS."""
    monkeypatch.setitem(ENTRIES, 32, (RA, "a/ra3", ST.FORMER, 150.0))
    monkeypatch.setitem(ENTRIES, 33, (RA, "a/ra4", ST.FORMER, 120.0))
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate((RA, hk(33))))
    s = snap()
    assert adapter.track_results([RA], BEFORE) == {}                 # RA is not active: no ranks
    assert legacy_leaders(spec_state(), (RA,), s)[RA] == [(30, True), (31, False), (32, False), (33, False)]
    assert set(legacy_paid(pay(eng, adapter, s))) == {20, 30, 33}


# ------------------------------------------------------------------------------------------------ fail closed
def test_stale_missing_or_invalid_state_pays_no_legacy(tmp_path):
    now = [AFTER]
    clock = [AFTER.timestamp()]
    eng, adapter, sp = system(tmp_path, now, Gate(), wall=lambda: clock[0])
    assert set(legacy_paid(pay(eng, adapter, snap()))) == {20, 30}
    clock[0] += REMOTE_MAX_AGE_S + 60
    assert legacy_paid(pay(eng, adapter, snap())) == {}
    clock[0] = AFTER.timestamp()
    sp.write_text("garbage")
    assert legacy_paid(pay(eng, adapter, snap())) == {}
    sp.unlink()
    assert legacy_paid(pay(eng, adapter, snap())) == {}


def test_a_broken_legacy_reader_changes_nothing_else(tmp_path, monkeypatch):
    now = [AFTER]
    eng, adapter, _ = system(tmp_path, now, Gate((GEN, hk(40))), gen=(40, "g/x"))
    s = snap(extra=[Neuron(40, hk(40), "c40", False, 0.0, None, None, commitment=SpecCommitment(GEN, "g/x", REV).text())])
    with monkeypatch.context() as mp:
        mp.setattr(CS, "LEGACY_KA_TRACKS", ())
        mp.setattr(RM, "LEGACY_KA_TRACKS", ())
        ref = pay(eng, adapter, s)
    monkeypatch.setattr(RM, "legacy_leaders", lambda *a: 1 / 0)
    assert pay(eng, adapter, s) == ref and ref[40] > 0.2
    monkeypatch.setattr(adapter, "legacy", lambda *a: 1 / 0)
    assert eng.weights(SERVING, s, {0}) == ref


def test_local_evaluator_path(tmp_path):
    sp = tmp_path / SPEC_STATE_FILE
    spec_state(sp)
    cfg = SC.SpecConfig(sealed_dir=str(tmp_path), work_dir=str(tmp_path / "w"), state_path=str(sp))
    local = SpecEvaluator(cfg)
    s = snap()
    assert local.legacy(s) == {DF: [(20, True), (21, False), (22, False)], RA: [(30, True), (31, False)]}
    remote = RM.RemoteSpecEvaluator(str(sp), clock=lambda: AFTER.timestamp())
    remote.leaders(cfg.active_tracks(AFTER.timestamp()), s)
    assert remote.legacy(s) == local.legacy(s)
    # a state without the DF/RA tracks: nothing, and the state is not changed
    empty = ST.SpecState()
    assert legacy_leaders(empty, CS.LEGACY_KA_TRACKS, s) == {} and empty.tracks == {}


def test_nothing_new_in_the_state_file(tmp_path):
    """Rollback safety: the legacy keep-alive reads only fields rc9.post8 already writes (no new state key)."""
    sp = tmp_path / SPEC_STATE_FILE
    st = spec_state(sp)
    raw = json.load(open(sp))
    assert set(raw) == {"version", "saved_at", "eval_times", "tracks"}
    assert ST.from_json(raw).tracks.keys() == st.tracks.keys()


# ------------------------------------------------------------------------------------------------ byte-identical
def test_none_or_empty_is_byte_identical():
    rng = random.Random(7)
    for _ in range(3000):
        base = {u: rng.random() for u in rng.sample(range(0, 60), rng.randint(1, 15))}
        tr = {GEN: [(rng.randrange(60), 40 + rng.random() * 20) for _ in range(rng.randint(0, 3))]}
        kw = dict(owner_uid=rng.choice([None, 0]), burn_fraction=rng.random() * 0.5, now=AFTER.timestamp(),
                  keepalive={GEN: [(rng.randrange(60), 50.0)]} if rng.random() < 0.5 else None,
                  engine_waiting=rng.sample(range(60), 2) if rng.random() < 0.3 else None)
        ch = rng.choice([None, rng.randrange(60)])
        a = CS.combine_weights(base, ch, 0.8, tr, **kw)
        assert repr(CS.combine_weights(base, ch, 0.8, tr, legacy_keepalive=None, **kw)) == repr(a)
        assert repr(CS.combine_weights(base, ch, 0.8, tr, legacy_keepalive=[], **kw)) == repr(a)


def test_only_entries_crowned_before_the_switch():
    st = spec_state()
    st.track(RA).submissions[f"{hk(30)}|{commitment(30)}"].champion_since_ts = START.timestamp()   # crowned at/after the switch
    st.track(DF).submissions[f"{hk(21)}|{commitment(21)}"].champion_since_ts = None                # no crowning time
    lg = legacy_leaders(st, CS.LEGACY_KA_TRACKS, snap())
    assert lg == {DF: [(20, True), (22, False)], RA: [(31, False)]}
