"""Keep-alive gauntlet round 2 (switch ``CS.KA_GAUNTLET_R2``): the unit cases of the fixes F1, F2, F3, F5 and the switch.

The property tests (test_ka_gauntlet_props.py) run the same fixes over generated timelines; these are the small, direct cases on
``compute_keepalive`` / ``compute_provisional``. Each fix is shown with the switch on and with the switch off (= 233dbd2)."""

import pytest
from spec_helpers import HOTKEYS, snapshot
from test_keepalive import DF, _sub
from test_provisional_ka import _snapr
from test_waiting_ka_hotfix import CB_NOW, H, THU, _commit_block, _ka, _neuron, _pending, _prov, _queued, release_slots  # noqa: F401

from reax_subnet import competition_split as CS
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import compute_keepalive, compute_provisional


def _many_pending(st, n):
    subs = []
    for uid in range(1, n + 1):
        s = _pending(st, uid, THU - (2 + uid) * H)
        s.score = s.nonproc = 40.0 + uid
        subs.append(s)
    return subs


def test_switch_default_is_on():
    assert CS.KA_GAUNTLET_R2 is True and CS.PROVISIONAL_KA_UNSEEN_GRACE_HOURS == 6.0


def test_f1_no_count_cut_in_compute_keepalive_under_the_hotfix():
    st = ST.SpecState()
    subs = _many_pending(st, 14)
    got = _ka(st, subs, THU)
    assert sorted(got) == list(range(1, 15)) and got[0] == 14          # nobody cut; the deterministic rank order stays (best first)


def test_f1_switch_off_cuts_to_eight(monkeypatch):
    monkeypatch.setattr(CS, "KA_GAUNTLET_R2", False)
    st = ST.SpecState()
    subs = _many_pending(st, 14)
    assert len(_ka(st, subs, THU)) == 8


def test_f1_hotfix_off_still_cuts_to_eight(monkeypatch):
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    st = ST.SpecState()
    subs = _many_pending(st, 14)
    assert len(_ka(st, subs, THU)) == 8


def test_f1_combine_weights_pays_all_of_them_inside_the_cap():
    st = ST.SpecState()
    subs = _many_pending(st, 14)
    ka = compute_keepalive(st, [DF], __import__("test_keepalive")._snap(subs), {DF: 0.0}, now=THU)
    serving = {u: 1.0 for u in range(100, 140)}
    w = CS.combine_weights(serving, 1, 1.0, {DF: []}, keepalive=ka)
    assert all(w.get(u, 0.0) > 0 for u in range(1, 15))


def _recommit_snapshot(hours_ago, uid=1):
    return snapshot(CB_NOW, [_neuron(uid, repo="m/new-revision", cb=_commit_block(hours_ago))])


def test_f2_unrecorded_recommit_gets_a_grace_from_its_own_block(monkeypatch):
    st = ST.SpecState()
    _queued(st, 1, first_seen=THU - 30 * H)                                    # the hotkey's first sighting, 30 h ago
    assert compute_provisional(st, [DF], _recommit_snapshot(1.0), now=THU)[DF] == [1]
    assert compute_provisional(st, [DF], _recommit_snapshot(5.9), now=THU)[DF] == [1]
    assert compute_provisional(st, [DF], _recommit_snapshot(6.1), now=THU)[DF] == []          # short: 6 h, then the normal rules
    monkeypatch.setattr(CS, "KA_GAUNTLET_R2", False)
    assert compute_provisional(st, [DF], _recommit_snapshot(1.0), now=THU)[DF] == []


def test_f2_two_per_coldkey_and_the_abuse_clause_still_hold():
    st = ST.SpecState()
    _queued(st, 1, first_seen=THU - 30 * H)
    _queued(st, 2, first_seen=THU - 30 * H)
    snap = snapshot(CB_NOW, [_neuron(1, repo="m/new1", cb=_commit_block(1), coldkey="shared"),
                             _neuron(2, repo="m/new2", cb=_commit_block(2), coldkey="shared")])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [2, 1]         # two per coldkey (earlier commitment first)
    if not hasattr(CS, "KA_ABUSE_CLAUSE"):          # the Tue (p8) line has no abuse clause
        return
    fake = _queued(st, 3, first_seen=THU - 30 * H, private=True)
    fake.reason = "waiting for the repo: 404 Client Error"
    snap2 = snapshot(CB_NOW, [_neuron(4, repo="m/x", cb=_commit_block(1), coldkey=fake.coldkey)])
    assert compute_provisional(st, [DF], snap2, now=THU)[DF] == []             # a fake's coldkey gets no unverified window (F4 policy)


def test_f3_unseen_commitments_do_not_count_against_the_unverified_cap(monkeypatch):
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 8)
    st = ST.SpecState()
    dummies = [_neuron(u, cb=_commit_block(20 - u)) for u in range(1, 11)]
    assert compute_provisional(st, [DF], snapshot(CB_NOW, dummies), now=THU)[DF] == list(range(1, 11))
    # entries the evaluator saw and found miner-side still count, oldest commitment first
    seen = [_queued(st, u, fetched=False, private=True, first_seen=THU - 2 * H) for u in range(11, 20)]
    snap2 = _snapr(seen, cblocks={u: _commit_block(1.5) for u in range(11, 20)}, block=CB_NOW)
    paid = compute_provisional(st, [DF], snapshot(CB_NOW, list(snap2.neurons) + dummies), now=THU)[DF]
    assert len(set(paid) & set(range(11, 20))) == 8 and set(range(1, 11)) <= set(paid)
    monkeypatch.setattr(CS, "KA_GAUNTLET_R2", False)
    assert compute_provisional(st, [DF], snapshot(CB_NOW, dummies), now=THU)[DF] == list(range(1, 9))


def _same_tensors(st, old_fp_h, own_fp_h):
    old = _pending(st, 1, THU - old_fp_h * H)
    old.hash_named = own = "hsame"
    old.status = ST.WITHDRAWN
    new = _pending(st, 2, THU - own_fp_h * H)
    new.hash_named = own
    new.score = new.nonproc = 50.0
    return [old, new]


def test_f5_same_tensors_minimum_is_for_the_plain_window_only(release_slots):  # noqa: F811
    st = ST.SpecState()
    subs = _same_tensors(st, 250, 5)               # first pass 10.4 days ago, withdrawn; the same tensors again 5 h ago
    assert 2 in _ka(st, subs, THU)                 # its own slot (Mon 2 Nov) decides it: covered until then
    st2 = ST.SpecState()
    subs2 = _same_tensors(st2, 250, 5 + 24 * 40)   # control: no window left from either first pass
    assert 2 not in _ka(st2, subs2, THU)


def test_f5_switch_off_is_the_old_cap(release_slots, monkeypatch):               # noqa: F811
    monkeypatch.setattr(CS, "KA_GAUNTLET_R2", False)
    st = ST.SpecState()
    subs = _same_tensors(st, 250, 5)
    assert 2 not in _ka(st, subs, THU)
