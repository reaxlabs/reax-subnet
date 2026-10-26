"""Eval v5.1 armed for Mon 26 Oct 2026 14:30Z with pool v0.1 (Florian GO 23 Oct, DECISIONS v5-monday-go-despite-gate2):
the release constant, the v0.1 panel stats, fail closed without stats for the active pool version, the per-cycle config cache,
gate armed but not reached = gate off byte for byte, and the rule-A transition at the slot. Fake models only."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone

import pytest
from conftest import RELEASE_CROWN_SLOTS_UTC, RELEASE_EVAL_V5_START_UTC
from test_v5_wiring import BASE_REPO, FAST, Gen, v51_rows

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import state as ST
from reax_subnet.spec import v5, v5wire as W

START = datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc)
T_START = START.timestamp()
ALL_AREAS = dict(v5.HF_AREAS)


@pytest.fixture(scope="module")
def rows():
    return v51_rows()


@pytest.fixture
def gen(tmp_path, monkeypatch, rows):
    return Gen(tmp_path, monkeypatch, rows)


def write_config(d, version="v0.1", areas=None, mode=0o600):
    p = os.path.join(d, W.CONFIG_FILE)
    with open(p, "w") as fh:
        json.dump({"pool_version": version, "hf_areas": areas or ALL_AREAS}, fh)
    os.chmod(p, mode)
    return p


# ---------------------------------------------------------------------------------------------- release constants
def test_release_arms_eval_v51_at_the_monday_slot():
    assert RELEASE_EVAL_V5_START_UTC == START and RELEASE_EVAL_V5_START_UTC.tzinfo is not None
    assert START in RELEASE_CROWN_SLOTS_UTC                                  # the gate opens at the crown slot (rule A)
    assert S.EVAL_V5_START_UTC is None                                        # conftest: tests run gate-off unless they set it
    assert v5.EVALUATOR_WIRED


def test_panel_stats_v01_are_the_stock_panel_run():
    p = v5.PANEL_STATS["v0.1"]
    assert p == v5.panel_from_json({"pool_version": "v0.1", "T": {"mean": 58.443470442129176, "sd": 19.08197856407689},
                                    "N": {"mean": 28.490328496088534, "sd": 8.20230568361892},
                                    "Pub": {"mean": 46.68485197351358, "sd": 19.22683663229349}})
    assert E.C.from_env({"REAX_SPEC_SEALED_DIR": "/pool"}).v5_w_n == 0.4      # the panel was measured at the default w_N


# ------------------------------------------------------------------------------------------ (d) fail closed: panel
def test_active_panel_resolution_and_fail_closed(caplog):
    v01 = v5.PANEL_STATS["v0.1"]
    cfg = ("v0.1", ALL_AREAS)
    assert W.active_panel(cfg, None) is v01                                  # the release stats of the config's pool version
    own = v5.PanelStats("v0.1", t=(50.0, 10.0), n=(30.0, 5.0))
    assert W.active_panel(cfg, own) is own                                   # REAX_SPEC_V5_PANEL for the same version wins
    caplog.set_level(logging.ERROR, logger="reax.spec")
    for config, panel, why in ((None, None, "pool version unknown"), (None, own, "pool version unknown"),
                               (("v0.2", ALL_AREAS), None, "no panel stats for pool version 'v0.2'"),
                               (cfg, v5.PanelStats("v0.2", t=(50.0, 10.0)), "is for pool version 'v0.2'")):
        caplog.clear()
        with pytest.raises(S.SampleError, match="panel stats"):
            W.active_panel(config, panel)
        assert any("FAIL CLOSED" in r.getMessage() and why in r.getMessage() for r in caplog.records), caplog.text


def test_params_take_a_preloaded_config_without_reading_the_directory(tmp_path):
    p = W.params(pool_dir=str(tmp_path / "missing"), config=("v0.1", {**ALL_AREAS, "arts": 0.0}))
    assert p.hf_areas["arts"] == 0.0
    assert W.params(config=None).hf_areas == W.params().hf_areas == v5.HF_AREAS     # None = no file: the defaults


def _armed_gen(tmp_path, monkeypatch, rows, *, version="v0.1", config=True):
    g = Gen(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)
    if config:
        write_config(g.env.cfg.v5_pool_dir, version)
    return g


@pytest.mark.parametrize("version,config", [("v0.2-no-stats", True), ("v0.1", False)])
async def test_missing_panel_stats_decide_nothing_after_the_start(tmp_path, monkeypatch, rows, caplog, version, config):
    """(d) At/after the start without stats for the active pool version (or without a pool version): no v5.1 draw, nothing served,
    nothing decided or crowned, a loud log; with the stats the same entry is decided."""
    g = _armed_gen(tmp_path, monkeypatch, rows, version=version, config=config)
    await g.crown(1, "a/champ", .60)                                         # crowned under v4 (Wed 21 Oct, before the start)
    g.seat(1)
    g.env.clock.t = T_START + 60
    g.add(3, "a/new", .95)
    caplog.set_level(logging.ERROR, logger="reax.spec")
    g.env.server.served.clear()
    await g.cycle()
    new = g.sub(3)
    assert new.status == ST.QUEUED and not new.history and not new.draw1, (new.status, new.reason)
    assert g.ts.champion == g.sub(1).key and g.sub(1).status == ST.CHAMPION
    assert not g.env.server.served                                           # nothing served: the draw is refused first
    assert any("no draw" in e and "panel stats" in e for e in g.ts.events), g.ts.events[-5:]
    assert any("FAIL CLOSED" in r.getMessage() for r in caplog.records)
    write_config(g.env.cfg.v5_pool_dir, "v0.1")                              # the stats arrive: the next cycle decides
    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", lambda self_: W.params(resamples=40, panel=v5.PANEL_STATS["v0.1"],
                                                                                pool_dir=self_.cfg.v5_pool_dir))
    await g.cycle(E.C.RETRY_UNAVAILABLE_S + 1)
    assert new.status in (ST.PENDING_CONFIRM, ST.NOT_BETTER) and new.history[-1]["scoring_regime"] == "v5"


# ------------------------------------------------------------------------------- (b) at/after the start: the v5.1 path
async def test_at_the_start_the_fixture_pool_is_drawn_and_scored_with_the_v01_stats(tmp_path, monkeypatch, rows):
    g = _armed_gen(tmp_path, monkeypatch, rows)
    await g.crown(1, "a/champ", .55)
    g.seat(1)
    seen = []
    real = E.SpecEvaluator._v5_params

    def params(self_):                       # the real resolution (config + PANEL_STATS), fewer resamples for speed
        p = real(self_)
        seen.append(p)
        return W.v5.V5Params(**{**{f: getattr(p, f) for f in p.__dataclass_fields__}, "resamples": 40})

    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", params)
    g.env.clock.t = T_START                                                  # exactly the start instant counts as v5
    g.add(3, "a/top", .99)
    g.env.server.served.clear()
    await g.cycle()
    top = g.sub(3)
    first = next(h for h in top.history if h["kind"] == "first")
    assert first["scoring_regime"] == "v5" and W.scored_under_v5(top.by_tier) and BASE_REPO in g.env.server.served
    assert seen and all(p.panel is v5.PANEL_STATS["v0.1"] and p.w_n == 0.4 for p in seen)
    d = g.ts.draws[top.draw1]
    pool = {r["id"]: r for r in [*rows[0], *rows[1]]}
    assert sum(1 for i in d.item_ids if pool.get(i, {}).get("part") == "T") == 540
    assert sum(1 for i in d.item_ids if pool.get(i, {}).get("part") == "N") == 360


async def test_config_is_read_once_per_cycle(tmp_path, monkeypatch, rows):
    """Item 2 (round2 #24356): ``v51.config.json`` is read once a cycle, next to the pools; the 0600 check stays (next cycle)."""
    g = _armed_gen(tmp_path, monkeypatch, rows)
    await g.crown(1, "a/champ", .55)
    g.seat(1)
    reads = []
    real = W.load_config
    monkeypatch.setattr(W, "load_config", lambda d: reads.append(d) or real(d))
    g.env.clock.t = T_START + 10
    g.add(3, "a/top", .99)
    await g.cycle()
    assert g.sub(3).history and len(reads) == 1
    os.chmod(os.path.join(g.env.cfg.v5_pool_dir, W.CONFIG_FILE), 0o644)
    g.add(4, "a/next", .995)
    await g.cycle(E.C.RETRY_UNAVAILABLE_S + 1)
    assert len(reads) == 2 and not g.sub(4).history                          # read again next cycle and refused: no draw
    assert any("no draw" in e and "0600" in e for e in g.ts.events), g.ts.events[-4:]


# ------------------------------------------------------------------- (a) armed but not reached = gate off, byte for byte
async def _scenario(g):
    out = []
    await g.crown(1, "a/champ", .60)
    g.add(2, "a/new", .95)
    g.add(3, "a/weak", .40)
    out.append(await g.cycle())
    out.append(await g.cycle(13 * 3600))
    out.append(await g.cycle(3600))
    return out


@pytest.mark.parametrize("gate", ["off", "armed"])
async def test_gate_armed_before_the_start_never_touches_v51(gen, monkeypatch, gate):
    """The gate-off test of test_v5_wiring with the armed release constant: before 26 Oct 14:30Z no v5.1 code runs."""
    if gate == "armed":
        monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)

    def boom(*a, **k):
        raise AssertionError("v5.1 code reached before the start")

    for name in ("draw", "load_pool", "load_config", "active_panel", "flatten", "build_context", "responses"):
        monkeypatch.setattr(W, name, boom)
    monkeypatch.setattr(E.SpecEvaluator, "_v5_bases", boom)
    monkeypatch.setattr(v5, "draw_v5", boom)
    g = gen
    await _scenario(g)
    assert not S.v5_active(g.env.clock.t) and g.env.clock.t < T_START
    assert BASE_REPO not in g.env.server.served
    assert all("scoring_regime" not in h and h["kind"] != "v5_restart" for s in g.ts.submissions.values() for h in s.history)
    assert all(len(d.item_ids) == 300 for d in g.ts.draws.values())          # v4 draws (30 per category, the DRAWS_V4 gate is later)
    assert g.env.ev._v5_ctx is None and not g.env.ev._v5_served and g.env.ev._v5_config is None


async def test_gate_armed_before_the_start_is_byte_identical_to_gate_off(tmp_path, monkeypatch, rows):
    """Decisions, draws, weights and the saved state file of the same scenario, gate None vs the armed release gate."""
    results = {}
    for gate in (None, RELEASE_EVAL_V5_START_UTC):
        with monkeypatch.context() as mp:
            mp.setattr(S, "EVAL_V5_START_UTC", gate)
            d = tmp_path / ("armed" if gate else "off")
            d.mkdir()
            g = Gen(d, mp, rows)
            weights = await _scenario(g)
            with open(g.env.cfg.state_path, "rb") as fh:     # saved_at is the wall clock of the write, not a decision
                state = re.sub(rb'"saved_at": [0-9.]+', b'"saved_at": 0', fh.read().replace(str(d).encode(), b"<tmp>"))
            results[gate is not None] = (weights, state, g.env.clock.t, sorted(g.env.server.served))
    off, armed = results[False], results[True]
    assert off[2] < T_START
    assert armed[0] == off[0] and armed[3] == off[3]                          # the same weights each cycle, the same models served
    assert armed[1] == off[1]                                                 # the state file byte for byte (decisions, draws, uses)
    st = json.loads(off[1])
    assert st and any(s.get("status") == ST.CHAMPION for t in st["tracks"].values() for s in t["submissions"].values())


@pytest.mark.parametrize("now", [START - timedelta(seconds=1), START, START + timedelta(seconds=1)])
def test_weights_floor_switches_at_the_start(now, monkeypatch):
    """Validator side (combine_weights, same constant): the v4-scale generalist entry floor holds until 14:29:59 and is off from
    the start on (S_v5.1 is on the DI scale); a keep-alive entry below the v4 floor is paid only then."""
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)
    floor = CS.TRACK_ENTRY_FLOOR["generalist"]
    assert floor is not None and floor > 1.0
    w = CS.combine_weights({1: 1.0}, None, 0.0, {}, keepalive={"generalist": [(7, floor / 2)]}, now=now.timestamp())
    assert (w.get(7, 0.0) > 0) == (now >= START)


# --------------------------------------------------------------------------- (c) rule A around Mon 26 Oct 14:30:00Z
@pytest.mark.parametrize("now,active,v4_draw_at", [
    (START - timedelta(seconds=1), False, START - timedelta(seconds=1)),
    (START, True, START - timedelta(seconds=1)),
    (START + timedelta(seconds=1), True, START - timedelta(seconds=1)),
])
def test_rule_a_instants(now, active, v4_draw_at, monkeypatch):
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)
    t = now.timestamp()
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    assert S.v5_active(t) is active and S.v5_active(now) is active and ev._v5_on("generalist", t) is active
    assert not ev._v5_on("demand_forecasting", t)
    assert E.SpecEvaluator._v4_draw_now(t) == v4_draw_at.timestamp()          # the slot playoff draws under v4 from 14:29:59
    assert v5.scoring_regime(t) == ("v5" if active else "v4")
    assert S.generalist_per_category(E.SpecEvaluator._v4_draw_now(t)) == S.GENERALIST_PER_CATEGORY_V4   # a v4 draw: 90 per category


async def test_rule_a_end_to_end_at_the_release_slot(gen, monkeypatch):
    """Rule A with the release constants: two entries confirm under v4 before Monday; at 14:29:59 nothing is crowned yet; at the
    14:30:00 slot the v4 slot playoff crowns one and the unconfirmed v4 stage restarts as a v5.1 first stage (once); at 14:30:01
    the gate stays open and nothing is restarted twice; an hour later the restarted entry is decided under v5.1."""
    g = gen
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", RELEASE_CROWN_SLOTS_UTC)
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)
    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", lambda self_: FAST)
    g.env.clock.t = T_START - 29 * 3600                                     # crowned at T - 16 h (as the gate-relative test)
    await g.crown(1, "a/old", .60)
    g.seat(1)
    g.add(3, "a/e1", .90)
    g.add(5, "a/e2", .95)
    await g.cycle()
    g.add(4, "a/late", .99)
    await g.cycle(5 * 3600)
    await g.cycle(8 * 3600)                                                  # both confirm under v4, both crownings stored
    e1, e2, late = g.sub(3), g.sub(5), g.sub(4)
    assert all(E._crown_pending(s) is not None for s in (e1, e2)) and g.ts.champion == g.sub(1).key
    assert late.status == ST.PENDING_CONFIRM and E._crown_pending(late) is None   # its v4 first stage is not confirmed at the slot
    await g.cycle(T_START - 1 - g.env.clock.t)                               # Mon 14:29:59
    assert not S.v5_active(g.env.clock.t) and g.ts.champion == g.sub(1).key
    assert all(E._crown_pending(s) is not None for s in (e1, e2))
    assert not any(h["kind"] == "v5_restart" for h in late.history)
    await g.cycle(1)                                                         # Mon 14:30:00: the slot and the gate
    assert g.env.clock.t == T_START and S.v5_active(g.env.clock.t)
    crowned = [s for s in (e1, e2) if s.status == ST.CHAMPION]
    assert len(crowned) == 1 and g.ts.champion == crowned[0].key and g.sub(1).status == ST.FORMER, (
        [(s.uid, s.status, s.reason, [h["kind"] for h in s.history]) for s in g.ts.submissions.values()], g.ts.champion[:12])
    assert not any(h.get("scoring_regime") == "v5" for s in (e1, e2) for h in s.history if h["kind"] in ("playoff", "crown_done"))
    assert sum(h["kind"] == "v5_restart" for h in late.history) == 1
    await g.cycle(1)                                                         # Mon 14:30:01
    assert sum(h["kind"] == "v5_restart" for h in late.history) == 1 and g.ts.champion == crowned[0].key
    await g.cycle(3600)
    first = [h for h in late.history if h["kind"] == "first"][-1]
    assert first["scoring_regime"] == "v5" and set(first["ref_totals"]) == {crowned[0].key, W.base_key(late.base_model)}
