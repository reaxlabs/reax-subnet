"""Six-hour chain commitment cutoff, effective 29 October, and bounded held-entry payments."""
from datetime import datetime, timezone
from dataclasses import replace

import pytest

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E, state as ST
from spec_helpers import commit, neuron, snapshot

SLOT = datetime(2026, 10, 29, 14, 30, tzinfo=timezone.utc).timestamp()
NEXT = datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc).timestamp()
CUT = SLOT - 6 * 3600


@pytest.fixture(autouse=True)
def slots(monkeypatch):
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", tuple(datetime.fromtimestamp(t, timezone.utc)
                        for t in (SLOT-3*86400, SLOT, NEXT)))


def sub(uid=2, committed=CUT, track="generalist"):
    c = commit(track, f"miner{uid}/entry")
    n = neuron(uid, commitment=c)
    return ST.Submission(key=f"{n.hotkey}|{c}", uid=uid, hotkey=n.hotkey, coldkey=n.coldkey,
                         track_id=track, repo=f"miner{uid}/entry", revision="a" * 40,
                         first_seen_block=10000, first_seen_ts=SLOT, chain_commit_ts=committed,
                         status=ST.PENDING_CONFIRM, delta_checked=True, score=80, nonproc=80,
                         first_pass_ts=SLOT-48*3600,
                         history=[{"kind": "crown_pending", "ts": SLOT-3600, "slot": SLOT}])


@pytest.mark.parametrize("offset,expected", [(-12, SLOT), (0, SLOT), (12, NEXT)])
def test_chain_block_cutoff_inclusive(offset, expected):
    assert E.SpecEvaluator._slot_of(sub(committed=CUT+offset)) == expected


def test_activation_and_other_tracks():
    before = CS.CROWN_SLOTS_UTC[0].timestamp()
    assert E._cutoff_slot(sub(committed=before), before) == before
    assert E._cutoff_slot(sub(committed=SLOT, track="risk_assessment"), SLOT) == SLOT


def test_chain_evidence_not_observation_time_and_capture_once():
    s = sub()
    s.chain_commit_ts = None
    n = replace(neuron(2, commitment=commit("generalist", "miner2/entry")), commitment_block=8200)
    snap = replace(snapshot(10000, [n]), block=10000)
    E.SpecEvaluator._capture_chain_commit(s, n, snap, SLOT)
    assert s.chain_commit_block == 8200 and s.chain_commit_ts == CUT
    E.SpecEvaluator._capture_chain_commit(s, n, replace(snap, block=10010), SLOT+123)
    assert s.chain_commit_ts == CUT
    assert E.SpecEvaluator._slot_of(s) == SLOT


@pytest.mark.parametrize("bad", [None, 0, 10001, True])
def test_unknown_chain_block_never_invents_early_commit(bad):
    s = sub(); s.chain_commit_ts = None
    n = replace(neuron(2), commitment_block=bad)
    E.SpecEvaluator._capture_chain_commit(s, n, replace(snapshot(10000, [n]), block=10000), SLOT)
    assert s.chain_commit_ts is None
    assert E.SpecEvaluator._slot_of(s) == NEXT


def test_due_field_excludes_late_entry_and_keeps_entitlement_across_crown():
    on, late = sub(2), sub(3, CUT+12)
    ts = ST.TrackState(track_id="generalist", submissions={s.key:s for s in (on, late)})
    ev = object.__new__(E.SpecEvaluator)
    assert ev._slot_due(ts, SLOT) == [on]
    assert late.cutoff_hold_until_ts == NEXT and "held for 2026-11-02 14:30Z" in late.reason
    ts.champion = on.key; on.status = ST.CHAMPION
    ev._release_pending_crowns(ts, SLOT)
    assert late.status == ST.QUEUED and not late.draw1 and E._crown_pending(late) is None
    assert E._cutoff_held(late, SLOT+1)
    assert not E._cutoff_held(late, NEXT)
    assert not ev._slot_due(ts, SLOT+1)


def test_keepalive_until_held_slot_requires_live_commit_and_copy_check():
    s = sub(committed=CUT+12); s.cutoff_hold_until_ts=NEXT
    st = ST.SpecState(); st.track("generalist").submissions[s.key] = s
    n = neuron(s.uid, commitment=commit("generalist", s.repo))
    snap = snapshot(10000, [n])
    def paid(now):
        return E.compute_keepalive(st, ["generalist"], snap, floors={"generalist":0},
                                   nonproc_floors={"generalist":0}, now=now)["generalist"]
    assert paid(SLOT+1) == [(s.uid, 80)]
    E.SpecEvaluator._slot_restart(st.track("generalist"), s, SLOT, "new bar")
    assert paid(SLOT+1) == [(s.uid, 80)]
    assert paid(NEXT) == []
    s.delta_checked=False
    assert paid(SLOT+1) == []
    s.delta_checked=True; s.status=ST.REJECTED
    assert paid(SLOT+1) == []
    s.status=ST.QUEUED; snap=snapshot(10000, [replace(n, commitment="")])
    assert paid(SLOT+1) == []


def test_hold_state_roundtrip(tmp_path):
    s=sub(); s.cutoff_hold_until_ts=NEXT
    st=ST.SpecState(); st.track("generalist").submissions[s.key]=s
    p=tmp_path/"state.json"; ST.save(st,p)
    loaded=ST.load(p).track("generalist").submissions[s.key]
    assert loaded.chain_commit_ts == CUT and loaded.cutoff_hold_until_ts == NEXT


@pytest.mark.parametrize("gauntlet", [False, True])
def test_held_entries_keep_existing_entry_and_coldkey_caps(monkeypatch, gauntlet):
    monkeypatch.setattr(CS, "KA_GAUNTLET_R2", gauntlet)
    st=ST.SpecState(); subs=[sub(u, CUT+12) for u in range(2,14)]
    for s in subs:
        s.cutoff_hold_until_ts=NEXT
        st.track("generalist").submissions[s.key]=s
    ns=[neuron(s.uid, commitment=commit("generalist", s.repo)) for s in subs]
    def alive(ns):
        return E.compute_keepalive(st, ["generalist"], snapshot(10000, ns),
                                   floors={"generalist":0}, nonproc_floors={"generalist":0}, now=SLOT+1)["generalist"]
    # post8.2's gauntlet lists every qualifying entry and divides the share cap;
    # the pre-gauntlet count cap remains eight.
    assert len(alive(ns)) == (len(subs) if gauntlet and CS.WAITING_KA_HOTFIX else CS.KEEPALIVE_MAX_ENTRIES)
    assert CS.KEEPALIVE_MAX_ENTRIES == 8
    assert len(alive([replace(n, coldkey="same-owner") for n in ns])) == 2


@pytest.mark.parametrize("configured", [False, True])
def test_legacy_k1a_slot_before_activation_is_unchanged(monkeypatch, configured):
    before = SLOT - 3 * 86400
    if not configured:
        monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", ())
    s = sub(committed=None)
    s.history = [{"kind": "crown_pending", "ts": before + 60, "slot": before, "k1a": True}]
    assert E.SpecEvaluator._slot_of(s) == before
