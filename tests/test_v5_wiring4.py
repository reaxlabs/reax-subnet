"""Last v5 activation blockers: base readout failure, v5-only guards, keep-alive ranking, and pool dry-run."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import state as ST
from reax_subnet.spec import joint
from spec_helpers import make_weights, snapshot, write_model
from test_keepalive import NOW, _snap, _sub
from test_v5_wiring import Gen, v51_rows, write_v51


BAD_BASE_MODEL = "Qwen/Qwen3_5-8B@" + "c" * 40


@pytest.fixture(scope="module")
def rows():
    return v51_rows()


@pytest.fixture
def gen(tmp_path, monkeypatch, rows):
    return Gen(tmp_path, monkeypatch, rows)


async def test_f3_qwen35_base_without_readout_waits_only_its_challengers(gen):
    g = gen
    await g.crown(1, "a/champion", .50)
    g.seat(1)
    g.gate_on()

    base_dir = os.path.join(g.env.cfg.base_dir, BAD_BASE_MODEL.replace("/", "__", 1))
    write_model(base_dir, make_weights(987), skill_key="base/no-readout", category="generalist")
    with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"architectures": ["Qwen3_5Model"]}, fh)
    with pytest.raises(ValueError, match="requires a readout head"):
        joint.detect(base_dir, category="generalist", now_ts=g.env.clock.t)

    g.add(3, "a/good", .95)
    g.env.add(4, "a/bad-base", .99, category="generalist", base_model=BAD_BASE_MODEL)
    g.commits[4] = "a/bad-base"
    # Exercise the real joint detector while keeping successful non-joint models on the fake serving backend.
    g.env.ev.backend_factory = g.env.server
    g.env.ev._docker_backend = g.env.server

    await g.cycle()

    good, bad = g.sub(3), g.sub(4)
    assert good.status == ST.PENDING_CONFIRM
    assert bad.status in (ST.QUEUED, ST.PENDING_CONFIRM) and bad.score is None
    assert bad.reason.startswith("waiting") and "could not be served" in bad.reason
    assert not any(h.get("kind") == "first" for h in bad.history)
    assert "generalist" not in g.env.ev._host_failed
    assert any("declared base" in event and "could not be served" in event for event in g.ts.events)


async def test_v5_track_does_not_need_v4_calibration_but_gate_off_still_does(gen, monkeypatch):
    g = gen
    g.gate_on()
    monkeypatch.setattr(CS, "GENERALIST_G_MED", None)
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", None)
    g.env.cfg.g_med = {}
    g.env.cfg.entry_floor = float("nan")
    g.env.cfg.x_tier_mode = "shadow"
    g.add(3, "a/v5", .95)
    await g.cycle()
    assert g.sub(3).status == ST.PENDING_CONFIRM
    assert any(h.get("scoring_regime") == "v5" for h in g.sub(3).history)

    # With the gate off the same missing release calibration still blocks the generalist track.
    g2 = gen
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", None)
    g2.add(4, "a/v4-blocked", .95)
    await g2.cycle()
    blocked = g2.sub(4)
    assert blocked.status == ST.QUEUED and not blocked.history


def test_provisional_guard_uses_v4_calibration_only_when_gate_is_off(monkeypatch):
    from test_provisional_ka import BLOCK, _p, _snapr

    monkeypatch.setattr(CS, "GENERALIST_G_MED", None)
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", float("nan"))
    st = ST.SpecState()
    sub = _p(st, 1, since=None, track="generalist")
    sub.hash_named = ""
    snap = _snapr([sub], regs={1: BLOCK - 500}, cblocks={1: BLOCK - 1800})
    live = {n.hotkey: n for n in snap.neurons}
    assert E._provisional_ok(sub, live, snap.block, from_commitment=False) is None
    assert E._provisional_ok(sub, live, snap.block, from_commitment=True) is snap.neurons[0]


def test_v5_nonpositive_scores_keep_leader_and_keepalive_slots(monkeypatch):
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(NOW - 60, timezone.utc))
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS_TRACK", "generalist")
    st = ST.SpecState()
    sub = _sub(st, 1, ST.CHAMPION, -2.0, track="generalist")
    sub.by_tier = {"T": -2.0, "N": -2.0}
    sub.history.append({"kind": "confirm", "scoring_regime": "v5", "s_v51": -2.0, "incumbent": 0.0})
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", (sub.key,))
    snap = _snap([sub])

    assert E.compute_leaders(st, ["generalist"], snap, now=NOW)["generalist"] == [(1, -2.0)]
    assert E.compute_keepalive(st, ["generalist"], snap, floors={"generalist": 10.0}, now=NOW)["generalist"] == [(1, -2.0)]

    monkeypatch.setattr(S, "EVAL_V5_START_UTC", None)
    assert E.compute_leaders(st, ["generalist"], snap, now=NOW)["generalist"] == []
    assert E.compute_keepalive(st, ["generalist"], snap, floors={"generalist": 10.0}, now=NOW)["generalist"] == []


def test_v5_keepalive_top_eight_uses_deciding_reference_margin(monkeypatch):
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(NOW - 60, timezone.utc))
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", ())
    st = ST.SpecState()
    subs = []
    for uid in range(1, 10):
        sub = _sub(st, uid, ST.PENDING_CONFIRM, -1.0, track="generalist")
        sub.first_seen_ts = CS.GENERALIST_START_UTC.timestamp() + 1
        sub.by_tier = {"T": -1.0, "N": -1.0}
        margin = float(uid)
        sub.history.append({"kind": "first", "scoring_regime": "v5", "s_v51": -1.0,
                            "incumbent": -1.0 - margin})
        subs.append(sub)
    snap = _snap(subs)

    alive = E.compute_keepalive(st, ["generalist"], snap, floors={"generalist": 0.0}, now=NOW)["generalist"]
    assert [uid for uid, _ in alive] == list(range(9, 1, -1))
    assert all(score == -1.0 for _, score in alive)  # the output score remains S_v5.1; only the cut ranks by margin


async def test_slot_playoff_releases_field_when_every_miner_is_unavailable(gen, monkeypatch):
    from test_provisional_ka import _p

    g = gen
    now = g.env.clock.t
    a = _p(g.env.ev.state, 3, status=ST.PENDING_CONFIRM, track="generalist")
    b = _p(g.env.ev.state, 4, status=ST.PENDING_CONFIRM, track="generalist")
    for sub in (a, b):
        sub.history.append({"kind": "crown_pending", "ts": now - 60, "slot": now})
        sub.confirm_draw, sub.confirm_score = "draw", sub.score

    async def miner_unavailable(tdef, sub, snap, at):
        sub.wait_kind, sub.reason = ST.WAIT_MINER, "waiting for miner repo"
        return None

    monkeypatch.setattr(g.env.ev, "_prepare_cycle", miner_unavailable)
    await g.env.ev._run_playoff(g.env.cfg.tracks[0], g.ts, [a, b], snapshot(1000, []), now, 2, slot=True)

    # rc11.post3 (_slot_unserved, post2 critic F2): nobody crowned, each entry back on the challenger path with its first draw
    # kept (post2 put them back in the queue); the slot is recorded with no crowning
    for sub in (a, b):
        assert sub.status == ST.PENDING_CONFIRM and sub.confirm_draw == "" and sub.confirm_score is None
        assert E._crown_pending(sub) is None
        assert sub.history[-1]["kind"] == "crown_done"
    rec = g.ts.slot_results[-1]
    assert rec["result"] == "none" and rec["crowned"] == [] and {e["uid"] for e in rec["entries"]} == {3, 4}


def test_v5_pool_dryrun_reports_synthetic_pool_counts_only(tmp_path, rows):
    t_rows, n_rows = copy.deepcopy(rows)
    wide = t_rows[0]
    keys = [f"wide-{i}" for i in range(30)]
    wide["questions"][0]["criteria"] = {key: key for key in keys}
    wide["questions"][0]["expected"] = keys[0]
    pool_dir = write_v51(tmp_path / "pool", t_rows, n_rows)
    root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run([sys.executable, "tools/v5_pool_dryrun.py", "--pool-dir", pool_dir, "--seed", "7"],
                            cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["pool"]["by_part"] == {"N": len(n_rows), "T": len(t_rows)}
    assert report["cap_statistics"]["over_26_including_idk"] == 1
    assert report["simulated_draw"]["by_part"] == {"N": 360, "T": 540}
    assert report["simulated_draw"]["items"] == 900 and report["simulated_draw"]["public_gap_check_items"] == 0
    assert wide["id"] not in result.stdout and "wide-0" not in result.stdout


def test_v5_pool_dryrun_exits_one_on_load_or_coverage_error(tmp_path, rows):
    t_rows, n_rows = copy.deepcopy(rows)
    pool_dir = write_v51(tmp_path / "pool", t_rows, n_rows, mode=0o644)
    root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run([sys.executable, "tools/v5_pool_dryrun.py", "--pool-dir", pool_dir],
                            cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "dry run failed" in result.stderr
    assert not any(row["id"] in result.stderr for row in t_rows[:3])
