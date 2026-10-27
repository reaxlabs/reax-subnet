"""Final gauntlet critic fixes (keepalive-gauntlet-20261026): F-A (a coldkey's second entry never takes an unverified slot from
another coldkey's only entry). F-B (unseen grace vs the 72 h rewindow) is accepted, not changed: denying it also zeroes an honest
recommit after a 401 until the evaluator records it; the dilution it allows stays inside the 5 % provisional cap and 2 per coldkey."""
import pytest

from reax_subnet import competition_split as CS
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.spec import evaluator as E, state as ST
from reax_subnet.spec.commitment import SpecCommitment

NOW = 1793000000.0
BLOCK = 9_400_000
H = 3600.0


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for k in ("WAITING_KA_HOTFIX", "KA_BIG_UNDECIDED", "KA_GAUNTLET_R2", "KA_EXPOSED_FIRST", "KA_PROV_TWO_PER_COLDKEY"):
        monkeypatch.setattr(CS, k, True)


def _cap_world():
    st = ST.SpecState()
    ns = []
    for uid in range(17):       # coldkey X: uids 0 and 1 (oldest commitments); c2..c16 one entry each: 17 unverified, 16 coldkeys
        ck = "X" if uid < 2 else f"c{uid}"
        hk = f"5Hk{uid:03d}" + "x" * 40
        rev = f"{uid:040x}"
        com = SpecCommitment("generalist", f"m/m{uid}", rev).text()
        s = ST.Submission(key=f"{hk}|{com}", uid=uid, hotkey=hk, coldkey=ck, track_id="generalist", repo=f"m/m{uid}", revision=rev,
                          first_seen_block=BLOCK - 600 + uid, first_seen_ts=NOW - 2 * H + uid, status=ST.QUEUED)
        s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 401", ST.WAIT_MINER
        st.track("generalist").submissions[s.key] = s
        ns.append(Neuron(uid, hk, ck, False, 0.0, None, None, commitment=com, commitment_block=BLOCK - 600 + uid,
                         registered_block=BLOCK - 20000))
    return st, Snapshot(block=BLOCK, neurons=tuple(ns), owner_hotkey=None)


def test_fa_second_entry_does_not_take_another_coldkeys_only_unverified_slot():
    st, snap = _cap_world()
    out = E.compute_provisional(st, ["generalist"], snap, now=NOW)["generalist"]
    assert sorted(set(range(17)) - set(out)) == [1]       # X's second entry waits, every single-entry coldkey is paid


def test_fa_second_slot_still_paid_when_the_cap_has_room():
    st, snap = _cap_world()
    neurons = tuple(n for n in snap.neurons if n.uid != 16)
    out = E.compute_provisional(st, ["generalist"], Snapshot(block=BLOCK, neurons=neurons, owner_hotkey=None), now=NOW)["generalist"]
    assert sorted(out) == list(range(16))


def test_switches_off_restore_old_order(monkeypatch):
    for k in ("KA_GAUNTLET_R2", "KA_EXPOSED_FIRST", "KA_PROV_TWO_PER_COLDKEY"):
        monkeypatch.setattr(CS, k, False)
    st, snap = _cap_world()
    out = E.compute_provisional(st, ["generalist"], snap, now=NOW)["generalist"]
    assert sorted(set(range(17)) - set(out)) == [1]       # one per coldkey: X keeps only uid 0


def _uid25_world():
    """Owner #28382 fixture: coldkey A has a checked immune entry (uid 1) and an exposed private-repo entry (uid 2);
    16 other coldkeys have one private-repo entry each (uids 10..25), all inside their windows."""
    st, ns = ST.SpecState(), []

    def add(uid, ck, cb, reg, checked):
        hk = f"5Hk{uid:03d}" + "x" * 40
        rev = f"{uid:040x}"
        com = SpecCommitment("generalist", f"m/m{uid}", rev).text()
        s = ST.Submission(key=f"{hk}|{com}", uid=uid, hotkey=hk, coldkey=ck, track_id="generalist", repo=f"m/m{uid}", revision=rev,
                          first_seen_block=cb, first_seen_ts=NOW - (BLOCK - cb) * 12, status=ST.QUEUED)
        if checked:
            s.hash_named, s.delta_checked = "h" * 64, True
        else:
            s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 401", ST.WAIT_MINER
        st.track("generalist").submissions[s.key] = s
        ns.append(Neuron(uid, hk, ck, False, 0.0, None, None, commitment=com, commitment_block=cb, registered_block=reg))

    add(1, "A", BLOCK - 100, BLOCK - 100, True)
    add(2, "A", BLOCK - 99, BLOCK - 20000, False)
    for i in range(16):
        add(10 + i, f"B{i}", BLOCK - 90 + i, BLOCK - 20000, False)
    return st, Snapshot(block=BLOCK, neurons=tuple(ns), owner_hotkey=None)


def test_uid25_exposed_unverified_does_not_take_the_checked_entrys_place_at_the_cap():
    st, snap = _uid25_world()
    out = E._committed_waiting(st, ["generalist"], snap, NOW)["generalist"]
    assert set(range(10, 26)) <= set(out)      # every single-entry coldkey keeps its slot (uid 25 included)
    assert 1 in out and 2 not in out           # A: checked entry paid; its exposed unverified entry waits for a free cap slot


def test_uid25_exposed_unverified_paid_when_the_cap_has_room():
    st, snap = _uid25_world()
    snap = Snapshot(block=BLOCK, neurons=tuple(n for n in snap.neurons if n.uid != 25), owner_hotkey=None)
    out = E._committed_waiting(st, ["generalist"], snap, NOW)["generalist"]
    assert sorted(out) == [1, 2] + list(range(10, 25))


def test_unseen_spare_commitment_does_not_let_the_counted_entry_jump_the_cap():
    """Critic on 1ae200b: X has an older unseen commitment (not counted against the cap) and a newer exposed private-repo
    entry; 16 single-entry coldkeys. X's counted entry must not take X's early first-pass position at the cap."""
    st, snap = _uid25_world()
    subs = st.track("generalist").submissions
    hk1 = next(n.hotkey for n in snap.neurons if n.uid == 1)
    for k in [k for k in subs if k.startswith(hk1 + "|")]:
        del subs[k]                                         # uid 1: committed on chain, not recorded by the evaluator yet
    snap = Snapshot(block=BLOCK, neurons=tuple(n._replace(coldkey="A") if hasattr(n, "_replace") else n for n in snap.neurons),
                    owner_hotkey=None)
    out = E._committed_waiting(st, ["generalist"], snap, NOW)["generalist"]
    assert set(range(10, 26)) <= set(out)       # every single-entry coldkey keeps its slot
    assert 1 in out and 2 not in out


def test_fg_second_slots_follow_their_own_commitment_age_at_the_cap():
    """Critic F-G on 46df10c: A and B each have a checked entry plus a private one; A's private entry is the newest, B's is
    older than every single-entry coldkey's. 15 single-entry coldkeys fill the cap to 15: the last counted slot goes to B's
    older second entry, not to A's newer one riding on A's checked position."""
    st, ns = ST.SpecState(), []

    def add(uid, ck, age, checked):
        cb, hk, rev = BLOCK - age, f"5Hk{uid:03d}" + "x" * 40, f"{uid:040x}"
        com = SpecCommitment("generalist", f"m/m{uid}", rev).text()
        s = ST.Submission(key=f"{hk}|{com}", uid=uid, hotkey=hk, coldkey=ck, track_id="generalist", repo=f"m/m{uid}", revision=rev,
                          first_seen_block=cb, first_seen_ts=NOW - age * 12, status=ST.QUEUED)
        if checked:
            s.hash_named, s.delta_checked = "h" * 64, True
        else:
            s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 401", ST.WAIT_MINER
        st.track("generalist").submissions[s.key] = s
        ns.append(Neuron(uid, hk, ck, False, 0.0, None, None, commitment=com, commitment_block=cb, registered_block=BLOCK - 20000))

    add(1, "A", 1000, True), add(2, "A", 100, False)
    add(3, "B", 900, True), add(4, "B", 800, False)
    for i in range(15):
        add(10 + i, f"S{i}", 500 - i, False)
    out = E._committed_waiting(st, ["generalist"], Snapshot(block=BLOCK, neurons=tuple(ns), owner_hotkey=None), NOW)["generalist"]
    assert 4 in out and 2 not in out
    assert {1, 3} | set(range(10, 25)) <= set(out)
