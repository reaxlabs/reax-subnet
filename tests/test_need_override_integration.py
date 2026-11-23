"""The merge keeps both operator-file inputs in the remote weight path."""
import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from reax_subnet import competition_split as cs, ka_override, need, remote
from reax_subnet.chain import Neuron
from reax_subnet.config import ENGINE_STATE_FILE
from test_ka_override import HK
from test_remote_competitions import Clock, SERVING, snap, stamp, write_engine_state


def test_need_file_and_keepalive_override_apply_together(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "NEED_RULE", True)
    now = cs.NEED_START_UTC + timedelta(hours=1)
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    stamp(path, now.timestamp())
    gate = SimpleNamespace(path=str(tmp_path / "approvals.json"), approved=lambda *a, **kw: True)
    (tmp_path / need.FILE_NAME).write_text(json.dumps({
        "inference": [{"from_utc": cs.NEED_START_UTC.isoformat(), "share": .35, "reason": "test demand"}],
        "pauses": []}))
    override = tmp_path / ka_override.FILE_NAME
    override.write_text(json.dumps([{"hotkey": HK, "kind": "legacy", "until_utc": (now + timedelta(hours=1)).isoformat(),
                                     "reason": "test operator override"}]))
    comp = remote.RemoteEngineCompetition(str(path), utcnow=lambda: now, wall=Clock(now.timestamp()), award_gate=gate)
    snapshot = snap()
    snapshot = replace(snapshot, neurons=(*snapshot.neurons, Neuron(67, HK, "c67", False, 0, None, None)))
    w = comp.weights(SERVING, snapshot, {0}, 0.)
    assert 0 < w[67] <= cs.LEGACY_KA_MAX_SHARE
    assert sum(w[u] for u in (11, 12, 67)) == pytest.approx(.35)
    assert w[3] == pytest.approx(.20)
    assert sum(w.values()) == pytest.approx(1.)
    override.unlink()
    without = comp.weights(SERVING, snapshot, {0}, 0.)
    assert without.get(67, 0) == 0
    assert without[11] + without[12] == pytest.approx(.35)
