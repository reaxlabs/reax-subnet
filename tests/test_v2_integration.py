"""rc11.post8 integration checks of the v2.0 package: the v1 payout is untouched while the composite is off, the bridge pays like v1
until the first dm-all crowning, and a composite validator logs the values operators compare."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import competition_split as cs
from reax_subnet import engcomp as EC
from reax_subnet import engcomposite as ecmp
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import TrackState
from reax_subnet.remote import RemoteEngineCompetition

import test_engine_switch as sw
from test_engine_switch import (D3, DA, DB, END, FRI, TUE, Approved, _chain, _comp, _snapshot, _v1_file, blk, H_BLOCK)
from test_engcomposite import D1

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _frozen(monkeypatch):
    from conftest import RELEASE_ENGINE_CROWN_SLOTS_UTC

    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", None)
    monkeypatch.setattr(cs, "ENGINE_TRACK_START_BLOCK", 9326382)
    monkeypatch.setattr(cs, "ENGINE_TRACK_START_UTC", datetime(2026, 10, 20, 18, 0, tzinfo=UTC))
    monkeypatch.setattr(cs, "ENGINE_CROWN_SLOTS_UTC", RELEASE_ENGINE_CROWN_SLOTS_UTC)


def _v1(tmp_path, now, raw, name):
    comp = EV.EngineCompetition(sw._LabelRunner({}), {"s1-fast": "ref"}, str(tmp_path / f"v1-{name}.json"), start_utc=None,
                                utcnow=lambda: now, award_gate=Approved())
    comp.states["s1-fast"] = TrackState.from_json(raw["tracks"]["s1-fast"])
    return comp


CHAINS = {
    "all": _chain(),
    "holder only": [("s1-fast", D3)],
    "former champion withdrawn": [("s1-fast", D3), ("dm-all", D1), ("s1-fast", "0" * 64), ("s1-fast", DB)],
}


def test_the_bridge_pays_exactly_like_v1_between_the_v2_start_and_the_first_dm_all_crowning(tmp_path):
    """Same frozen v1 state, same chain: the v1 competition (s1-fast, no composite) and the v2 bridge return identical weights at every
    time from just after the Tue 27 Oct slot to just before the Fri 30 Oct slot (champion decay, former-champion windows, approvals)."""
    path, sha = _v1_file(tmp_path / "frozen.json", H_BLOCK)
    raw = json.loads((tmp_path / "frozen.json").read_text())
    times = [TUE + timedelta(minutes=m) for m in (1, 30)] + [TUE + timedelta(hours=h) for h in (2, 6, 12, 24, 30, 36, 48, 60, 71)]
    times.append(FRI - timedelta(seconds=1))
    for name, entries in CHAINS.items():
        for now in times:
            snap = _snapshot(entries, blk(now))
            a = _v1(tmp_path, now, raw, name).weights({5: 1.0}, snap, set())
            b = _comp(tmp_path, now, (path, END, sha), gate=Approved(), sub=f"v2-{name}").weights({5: 1.0}, snap, set())
            assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a), (name, now.isoformat(), a, b)
            assert a[0] > 0  # the holder is really paid in these cases


def test_a_holder_that_recommits_to_dm_all_keeps_its_share_only_on_the_bridge(tmp_path):
    """The one deliberate difference (decision M3): v1 drops a hotkey that commits another track, the bridge keeps its right. A validator
    on the old release would pay differently here, which is why every validator must run post8 before the title can change hands."""
    path, sha = _v1_file(tmp_path / "frozen.json", H_BLOCK)
    raw = json.loads((tmp_path / "frozen.json").read_text())
    now = TUE + timedelta(hours=12)
    snap = _snapshot([("dm-all", D3), ("dm-all", D1), ("s1-fast", DA), ("s1-fast", DB)], blk(now))
    a = _v1(tmp_path, now, raw, "rc").weights({5: 1.0}, snap, set())
    b = _comp(tmp_path, now, (path, END, sha), gate=Approved(), sub="rc").weights({5: 1.0}, snap, set())
    assert 0 not in a and b[0] > 0


def test_a_remote_composite_validator_logs_the_rules_and_context_hashes(tmp_path, caplog):
    track = ecmp.default_track()
    with caplog.at_level(logging.INFO, logger="reax.engcomp"):
        comp = RemoteEngineCompetition(str(tmp_path / "engine-competition.json"), composites={track.name: track},
                                       composite_start_utc=cs.ENGINE_V2_START_UTC)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert f"dm-all={ecmp.composite_rules_sha256(track)}" in text and comp.context_sha256 in text
    assert f"thresholds sha256 {comp.thresholds_sha256}" in text


def test_the_release_composite_track_is_ready_and_pinned():
    track = ecmp.default_track()
    assert track.release_ready() == []
    assert track.budget_s() == 9660.0 and [s.name for s in track.slots] == ["blink", "vision", "reax1", "jeffan", "mammoth"]
