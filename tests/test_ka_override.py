"""Operator keep-alive override (ka_override.py; waiting-state hotfix 26 Oct 2026, Florian's decision for Ichigeki)."""

import json
from datetime import timedelta

import pytest
from test_legacyka import AFTER, END, GEN, REV, SERVING, Gate, pay, snap, system

from reax_subnet import competition_split as CS
from reax_subnet import ka_override as KO
from reax_subnet.chain import Neuron
from reax_subnet.spec.commitment import SpecCommitment

HK = "5DvW2JJXq9v6Y2rS7Y3bUj4bEqQ1nYwq8rC8TzW4o7h2kJcV"[:48]
NEW = 67


@pytest.fixture(autouse=True)
def _floors_off(monkeypatch):
    for k in list(CS.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, k, None)


def _row(kind="legacy", until=None, **kw):
    r = {"hotkey": HK, "kind": kind, "until_utc": (until or END).isoformat(), "reason": "Florian 26 Oct: Ichigeki new hotkey"}
    r.update(kw)
    return r


def test_parse_accepts_a_strict_file_and_refuses_everything_else():
    rows, problem = KO.parse([_row(), _row("waiting")], AFTER)
    assert problem == "" and [r["kind"] for r in rows] == ["legacy", "waiting"]
    bad = [
        {"rows": []}, [_row()] * 6, [{**_row(), "extra": 1}], [_row(hotkey="not-an-address")], [_row(kind="champion")],
        [_row(reason="")], [{**_row(), "until_utc": "2026-10-28T13:00:00"}], [_row(until=AFTER + timedelta(days=9))],
    ]
    for raw in bad:
        assert KO.parse(raw, AFTER) == ([], KO.parse(raw, AFTER)[1]) and KO.parse(raw, AFTER)[1], raw


def test_apply_only_rows_in_force_with_a_live_hotkey():
    rows, _ = KO.parse([_row(), _row("waiting", until=AFTER + timedelta(hours=1))], AFTER)
    n = [Neuron(NEW, HK, "c67", False, 0.0, None, None)]
    assert KO.apply(rows, n, AFTER) == ([NEW], [NEW])
    assert KO.apply(rows, n, AFTER + timedelta(hours=2)) == ([NEW], [])
    assert KO.apply(rows, [], AFTER) == ([], [])


def _system_with_file(tmp_path, now, content):
    eng, adapter, sp = system(tmp_path, now)
    appr = tmp_path / "approvals.json"
    appr.write_text("[]")
    gate = Gate()
    gate.path = str(appr)
    eng.award_gate = gate
    if content is not None:
        (tmp_path / KO.FILE_NAME).write_text(content if isinstance(content, str) else json.dumps(content))
    return eng, adapter


def _snap_with_new():
    new = Neuron(NEW, HK, "c67", False, 0.0, None, None, commitment=SpecCommitment(GEN, "gurujustin/sn92-gen-f", REV).text())
    return snap(extra=[new])


def test_override_pays_the_new_hotkey_within_the_legacy_caps(tmp_path):
    now = [AFTER]
    eng, adapter = _system_with_file(tmp_path, now, [_row()])
    w = pay(eng, adapter, _snap_with_new())
    assert 0 < w[NEW] <= CS.LEGACY_KA_MAX_SHARE
    assert sum(w.values()) == pytest.approx(1.0)
    now[0] = END + timedelta(seconds=1)                                         # past until_utc: nothing from the override
    (tmp_path / "x").mkdir()
    eng2, adapter2 = _system_with_file(tmp_path / "x", now, [_row(until=END)])
    assert pay(eng2, adapter2, _snap_with_new()).get(NEW, 0.0) == 0.0


@pytest.mark.parametrize("content", [None, "{not json", [_row(kind="champion")]])
def test_absent_or_refused_file_changes_nothing(tmp_path, content):
    now = [AFTER]
    eng, adapter = _system_with_file(tmp_path, now, content)
    (tmp_path / "b").mkdir()
    base_eng, base_adapter = _system_with_file(tmp_path / "b", now, None)
    assert pay(eng, adapter, _snap_with_new()) == pay(base_eng, base_adapter, _snap_with_new())


def test_waiting_hotfix_off_ignores_operator_override(tmp_path, monkeypatch):
    """The release off-switch must also disable the filesystem override, even with a valid file present."""
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    now = [AFTER]
    eng, adapter = _system_with_file(tmp_path, now, [_row()])
    (tmp_path / "b").mkdir()
    base_eng, base_adapter = _system_with_file(tmp_path / "b", now, None)
    assert pay(eng, adapter, _snap_with_new()) == pay(base_eng, base_adapter, _snap_with_new())
    assert pay(eng, adapter, _snap_with_new()).get(NEW, 0.0) == 0.0
