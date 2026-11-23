"""The v1 payout bridge markers live in a writable directory of their own (REAX_ENGINE_V1_MARKER_DIR; remote validator: the
validator's own state directory), because the competitions state may be mounted read-only."""
import json
import os
import stat
import time
from datetime import datetime, timedelta, timezone

import pytest

import reax_subnet.competition_split as cs
from test_engcomposite import (_bridge_comp, _v1_state_file, _pin, _snapshot, BRIDGE_END, D1, D3, TRACK)
from reax_subnet.config import ConfigError
from reax_subnet.engcomp import Champion
from reax_subnet.engcomp import block_at_utc
from reax_subnet.remote import RemoteEngineCompetition
from reax_subnet.worker import v1_bridge_from_env, v1_marker_dir

NOW = BRIDGE_END.replace(day=10)


class Approved:
    def approved(self, track, hotkey, sid):
        return track == "s1-fast" and hotkey == "hk0" and sid == "sha256:" + D3


def _ro(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IXUSR)


def _rw(path):
    os.chmod(path, stat.S_IRWXU)


def test_remote_with_a_read_only_competitions_dir_pays_the_v1_champion_and_persists_markers_in_its_own_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "ENGINE_TRACK_START_BLOCK", 9326382)  # the finney anchor (blocks <-> time for the bridge end)
    monkeypatch.setattr(cs, "ENGINE_TRACK_START_UTC", datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc))
    v1 = _v1_state_file(tmp_path / "v1.json")
    comp_dir, own = tmp_path / "competitions-state", tmp_path / "state"
    comp_dir.mkdir()
    own.mkdir()
    state = comp_dir / "engine-competition.json"
    state.write_text(json.dumps({"version": 1, "saved_at": time.time(), "tracks": {}}))
    comp = RemoteEngineCompetition(str(state), start_utc=None, utcnow=lambda: NOW, composites={"dm-all": TRACK},
                                   v1_bridge=_pin(v1), award_gate=Approved(), v1_marker_dir=str(own))
    snap = _snapshot([("s1-fast", D3), ("dm-all", D1)], 500)
    _ro(comp_dir)
    try:
        assert 0 in comp.weights({5: 1.0}, snap, set())  # the bridge pays although the competitions dir is read-only
        comp.states["dm-all"].champion = Champion(D1, "hk1", 1, 1.3, 450)
        crowned = comp.states["dm-all"].to_json()
        _rw(comp_dir)  # the worker (not this process) writes the state file
        state.write_text(json.dumps({"version": 1, "saved_at": time.time(), "tracks": {"dm-all": crowned}}))
        _ro(comp_dir)
        assert 0 in comp.weights({5: 1.0}, snap, set())  # crowning seen: the carried place is paid ...
        assert (own / "engine-v1-bridge-ended.json").exists()  # ... and the one-way marker is in the own dir
        assert not (comp_dir / "engine-v1-bridge-ended.json").exists()
        late = RemoteEngineCompetition(str(state), start_utc=None, utcnow=lambda: BRIDGE_END + timedelta(hours=1),
                                       composites={"dm-all": TRACK}, v1_bridge=_pin(v1), award_gate=Approved(),
                                       v1_marker_dir=str(own))
        est = block_at_utc(BRIDGE_END)
        assert late._end_block(est - 5) == est - 5  # first block seen after the end time ...
        assert json.loads((own / "engine-v1-bridge-end-block.json").read_text())["block"] == est - 5  # ... persisted in the own dir
        assert not (comp_dir / "engine-v1-bridge-end-block.json").exists()
    finally:
        _rw(comp_dir)
    again = RemoteEngineCompetition(str(state), start_utc=None, utcnow=lambda: NOW, composites={"dm-all": TRACK},
                                    v1_bridge=_pin(v1), award_gate=Approved(), v1_marker_dir=str(own))
    state.write_text(json.dumps({"version": 1, "saved_at": time.time(), "tracks": {}}))
    assert again.weights({5: 1.0}, snap, set()) == {5: 1.0}  # restart with lost history: the marker keeps the bridge ended


def test_without_a_marker_dir_the_read_only_state_dir_pays_nothing(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    d = tmp_path / "competitions-state"
    d.mkdir()
    state = d / "engine-competition.json"
    state.write_text(json.dumps({"version": 1, "saved_at": time.time(), "tracks": {}}))
    comp = RemoteEngineCompetition(str(state), start_utc=None, utcnow=lambda: NOW, composites={"dm-all": TRACK},
                                   v1_bridge=_pin(v1), award_gate=Approved())
    _ro(d)
    try:
        assert comp.weights({5: 1.0}, _snapshot([("s1-fast", D3)], 500), set()) == {5: 1.0}
    finally:
        _rw(d)


def test_local_mode_keeps_the_markers_next_to_the_engine_state(tmp_path):
    v1 = _v1_state_file(tmp_path / "v1.json")
    comp = _bridge_comp(tmp_path, NOW, _pin(v1), Approved(), sub="st")
    assert comp.v1_marker_dir == ""
    assert comp._bridge_marker() == str(tmp_path / "st" / "engine-v1-bridge-ended.json")


def test_marker_dir_resolution():
    assert v1_marker_dir({}) == ""
    assert v1_marker_dir({}, "/state") == "/state"
    assert v1_marker_dir({"REAX_ENGINE_V1_MARKER_DIR": "/m"}, "/state") == "/m"


def test_finney_refuses_to_start_when_the_marker_dir_is_not_writable(tmp_path, monkeypatch):
    v1 = _v1_state_file(tmp_path / "v1.json")
    path, _end, sha = _pin(v1)
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_STATE_SHA256", sha)
    monkeypatch.setattr(cs, "ENGINE_V1_BRIDGE_END_UTC", BRIDGE_END)
    ro = tmp_path / "ro"
    ro.mkdir()
    ok = tmp_path / "ok"
    env = {"REAX_ENGINE_V1_STATE_PATH": path}
    sp = str(ro / "engine-competition.json")
    comps = {"dm-all": TRACK}
    _ro(ro)
    try:
        with pytest.raises(ConfigError, match="marker directory"):
            v1_bridge_from_env(env, sp, comps, True)  # default (next to the engine state) is read-only
        with pytest.raises(ConfigError, match="marker directory"):
            v1_bridge_from_env(env, sp, comps, True, str(ro / "sub"))  # cannot even be created
        assert v1_bridge_from_env(env, sp, comps, True, str(ok)) is not None  # a writable own dir fixes it
        assert not list(ok.glob(".bridge-probe-*"))
        assert v1_bridge_from_env(env, sp, comps, False) is not None  # development networks still start
    finally:
        _rw(ro)
