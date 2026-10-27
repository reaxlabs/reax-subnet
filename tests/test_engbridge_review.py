"""Regression tests for the review of the v1 payout bridge (findings M1: lifecycle, M2: pinned snapshot)."""
import json
import time

from test_engcomposite import (_bridge_comp, _v1_state_file, _pin, _snapshot, BRIDGE_END, D1, D3, TRACK)
from reax_subnet.engcomp import Champion, update_champion
from reax_subnet.remote import RemoteEngineCompetition

NOW = BRIDGE_END.replace(day=10)


class Approved:
    def approved(self, track, hotkey, sid):
        return track == "s1-fast" and hotkey == "hk0" and sid == "sha256:" + D3


def test_bridge_does_not_restart_after_first_composite_title_is_vacated(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    comp = _bridge_comp(tmp_path, NOW, _pin(v1), Approved())
    snap = _snapshot([("s1-fast", D3), ("dm-all", D1)], 500)
    comp.states["dm-all"].champion = Champion(D1, "hk1", 1, 1.3, 450)
    w = comp.weights({5: 1.0}, snap, set())
    assert 0 in w  # C (hk1) has no dm-all approval here: the previously approved v1 holder is restored (carried right)
    update_champion(comp.states["dm-all"], [], comp.tracks["dm-all"], 501)  # the winner withdraws
    assert comp.states["dm-all"].champion is None
    assert comp.states["dm-all"].history[-1]["event"] == "vacated"
    assert comp.weights({5: 1.0}, _snapshot([("s1-fast", D3)], 501), set()) == {5: 1.0}


def test_bridge_stays_ended_after_a_restart_even_with_empty_composite_state(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    snap = _snapshot([("s1-fast", D3), ("dm-all", D1)], 500)
    first = _bridge_comp(tmp_path, NOW, _pin(v1), Approved(), sub="state")
    assert 0 in first.weights({5: 1.0}, snap, set())  # bridge pays before any crowning
    first.states["dm-all"].champion = Champion(D1, "hk1", 1, 1.3, 450)
    w = first.weights({5: 1.0}, snap, set())
    assert 0 in w  # the bridge is over; the v1 holder is paid as a carried former champion (restored: C has no approval here)
    restarted = _bridge_comp(tmp_path, NOW, _pin(v1), Approved(), sub="state")  # new object, same directory, no history
    assert restarted.states["dm-all"].champion is None and not restarted.states["dm-all"].history
    assert restarted.weights({5: 1.0}, snap, set()) == {5: 1.0}


def test_bridge_stays_ended_when_remote_composite_snapshot_goes_missing(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    current = tmp_path / "composite.json"
    comp = RemoteEngineCompetition(str(current), start_utc=None, utcnow=lambda: NOW, composites={"dm-all": TRACK},
                                   v1_bridge=_pin(v1), award_gate=Approved())
    st = comp.states["dm-all"]
    st.champion = Champion(D1, "hk1", 1, 1.3, 450)
    current.write_text(json.dumps({"version": 1, "saved_at": time.time(), "tracks": {"dm-all": st.to_json()}}))
    snap = _snapshot([("s1-fast", D3), ("dm-all", D1)], 500)
    w = comp.weights({5: 1.0}, snap, set())
    assert 0 in w  # carried former champion
    current.unlink()
    assert comp.weights({5: 1.0}, snap, set()) == {5: 1.0}


def test_bridge_fails_closed_when_the_remote_composite_state_was_never_available(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    comp = RemoteEngineCompetition(str(tmp_path / "composite.json"), start_utc=None, utcnow=lambda: NOW,
                                   composites={"dm-all": TRACK}, v1_bridge=_pin(v1), award_gate=Approved())
    snap = _snapshot([("s1-fast", D3)], 500)
    assert comp.weights({5: 1.0}, snap, set()) == {5: 1.0}  # no valid state: no bridge payment
    (tmp_path / "composite.json").write_text(json.dumps({"version": 1, "saved_at": time.time(), "tracks": {}}))
    assert 0 in comp.weights({5: 1.0}, snap, set())  # valid state, nobody crowned: the bridge pays


def test_bridge_rejects_a_snapshot_that_does_not_match_the_pin(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    pinned = _pin(v1)
    snap = _snapshot([("s1-fast", D3)], 500)
    assert 0 in _bridge_comp(tmp_path, NOW, pinned, Approved()).weights({5: 1.0}, snap, set())  # parity: correct pin pays
    raw = json.loads(open(v1).read())
    raw["saved_at"] = 1  # any change to the bytes
    open(v1, "w").write(json.dumps(raw))
    assert _bridge_comp(tmp_path, NOW, pinned, Approved()).weights({5: 1.0}, snap, set()) == {5: 1.0}
    assert _bridge_comp(tmp_path, NOW, (v1, BRIDGE_END, ""), Approved()).weights({5: 1.0}, snap, set()) == {5: 1.0}


def test_bridge_keeps_the_v1_ownership_mode_of_the_snapshot(tmp_path):
    # v1 bound images to hotkeys: an image recorded as owned by another hotkey earns nothing, whatever the dm-all runner says
    from reax_subnet.engcomp import TrackState

    st = TrackState()
    st.champion = Champion(D3, "hk0", 0, 2.0, 400)
    st.owners[D3] = "hkX"
    p = tmp_path / "v1.json"
    st.slot_results.append({"slot_utc": "2026-10-27T14:30:00+00:00", "event": "crowned"})  # the switch slot is finalised in the frozen file
    p.write_text(json.dumps({"version": 1, "binds_hotkey": True, "tracks": {"s1-fast": st.to_json()}}))
    snap = _snapshot([("s1-fast", D3)], 500)
    assert _bridge_comp(tmp_path, NOW, _pin(p), Approved()).weights({5: 1.0}, snap, set()) == {5: 1.0}


def test_bridge_pays_nothing_when_the_marker_directory_is_unwritable_even_after_history_loss(tmp_path):
    import os
    import stat
    v1 = _v1_state_file(tmp_path / "v1.json")
    snap = _snapshot([("s1-fast", D3), ("dm-all", D1)], 500)
    comp = _bridge_comp(tmp_path, NOW, _pin(v1), Approved(), sub="st")
    d = tmp_path / "st"
    d.mkdir(exist_ok=True)
    try:
        os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)
        comp.states["dm-all"].champion = Champion(D1, "hk1", 1, 1.3, 450)
        w = comp.weights({5: 1.0}, snap, set())  # crowning seen, marker not writable: no champion share for the v1 holder,
        assert 0 in w  # only the derived (history-based) former-champion place
        assert not (d / "engine-v1-bridge-ended.json").exists()
        os.chmod(d, stat.S_IRWXU)
        w = comp.weights({5: 1.0}, snap, set())  # bridge disabled for good in memory
        assert 0 in w
        os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)
        restarted = _bridge_comp(tmp_path, NOW, _pin(v1), Approved(), sub="st")  # history wiped, no champion
        assert restarted.weights({5: 1.0}, _snapshot([("s1-fast", D3)], 501), set()) == {5: 1.0}
    finally:
        os.chmod(d, stat.S_IRWXU)
    healthy = _bridge_comp(tmp_path, NOW, _pin(v1), Approved(), sub="ok")
    assert 0 in healthy.weights({5: 1.0}, _snapshot([("s1-fast", D3)], 501), set())  # writable dir: bridge pays
    assert not list((tmp_path / "ok").glob(".bridge-probe-*"))  # probe cleaned up


def test_bridge_with_a_timezone_naive_end_time_does_not_raise(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    path, end, sha = _pin(v1)
    comp = _bridge_comp(tmp_path, NOW, (path, end.replace(tzinfo=None), sha), Approved())
    assert comp.v1_bridge[1].tzinfo is not None
    assert 0 in comp.weights({5: 1.0}, _snapshot([("s1-fast", D3)], 500), set())  # before the end: pays
    late = _bridge_comp(tmp_path, end.replace(tzinfo=None) .replace(day=21), (path, end.replace(tzinfo=None), sha), Approved())
    late._utcnow = lambda: end.replace(day=21)
    assert late.weights({5: 1.0}, _snapshot([("s1-fast", D3)], 500), set()) == {5: 1.0}
