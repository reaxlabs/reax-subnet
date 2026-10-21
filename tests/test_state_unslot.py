"""scripts/state_unslot.py: the multirole state file back in the pre-multirole layout (rollback)."""

from __future__ import annotations

import importlib.util
import json
import os

from reax_subnet import state as S
from reax_subnet.elements import slot_key

_spec = importlib.util.spec_from_file_location(
    "state_unslot", os.path.join(os.path.dirname(__file__), "..", "scripts", "state_unslot.py"))
unslot_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unslot_mod)


def _legacy(tmp_path):
    st = S.ValidatorState(netuid=92, hotkey="v")
    st.miners[1] = S.MinerState(hotkey="a", ema=0.9, role="s1-fast:aaaa")
    st.miners[2] = S.MinerState(hotkey="b", ema=0.8, role="s1-pro:bbbb", capacity=150.0)
    st.miners[3] = S.MinerState(hotkey="c", ema=0.7, role="s1-lite:cccc")
    st.miners[4] = S.MinerState(hotkey="d")
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(st.to_json()))
    return p


def test_round_trip_legacy_to_slots_and_back(tmp_path):
    p = _legacy(tmp_path)
    before = json.loads(p.read_text())["miners"]
    new = S.load(str(p), 92, "v")  # the multirole release migrates on load
    assert new.miners[slot_key(2, "s1-pro")].ema == 0.8 and new.miners[2].role == ""
    S.save(new, str(tmp_path / "new.json"))
    out, notes = unslot_mod.unslot(json.loads((tmp_path / "new.json").read_text()))
    for uid in ("1", "2", "3", "4"):
        for field in ("hotkey", "ema", "role", "capacity"):
            assert out["miners"][uid].get(field) == before[uid].get(field), (uid, field)
    assert all(int(k) < 65536 for k in out["miners"])


def test_multi_model_uid_keeps_the_best_slot_and_never_overrides_a_plain_role():
    raw = {"version": 1, "netuid": 92, "hotkey": "v", "rounds": 3, "miners": {
        "5": {"hotkey": "e", "ema": 0.5, "role": "s1-fast:aaaa", "last_score": 0, "last_fidelity": 0, "rounds": 1},
        str(slot_key(5, "s1-pro")): {"hotkey": "e", "ema": 0.9, "role": "s1-pro:bbbb", "last_score": 0,
                                     "last_fidelity": 0, "rounds": 1},
        "6": {"hotkey": "f", "ema": 0.0, "role": "", "last_score": 0, "last_fidelity": 0, "rounds": 1},
        str(slot_key(6, "s1-pro")): {"hotkey": "f", "ema": 0.4, "role": "s1-pro:bbbb", "last_score": 0,
                                     "last_fidelity": 0, "rounds": 1},
        str(slot_key(6, "s1-lite")): {"hotkey": "f", "ema": 0.6, "role": "s1-lite:cccc", "last_score": 0,
                                      "last_fidelity": 0, "rounds": 1}}}
    out, notes = unslot_mod.unslot(raw)
    assert out["miners"]["5"]["role"] == "s1-fast:aaaa"  # the plain role is never replaced
    assert out["miners"]["6"]["role"] == "s1-lite:cccc"  # highest moving average wins
    assert set(out["miners"]) == {"5", "6"} and out["rounds"] == 3


def test_cli_refuses_to_overwrite(tmp_path):
    import subprocess
    import sys

    p = _legacy(tmp_path)
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "state_unslot.py")
    r = subprocess.run([sys.executable, script, str(p), str(p)], capture_output=True, text=True)
    assert r.returncode != 0 and "refusing" in r.stderr
