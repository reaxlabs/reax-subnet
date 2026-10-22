"""Generalist release gates, exact transition, sampler and diagnostics. No real models or sealed pools."""
import dataclasses
import importlib.util
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import remote
from reax_subnet.spec import config as C, evaluator as E, rules as R, sampling as S, scoring as SC, state as ST


@pytest.mark.parametrize("day", [23, 26, 30])
def test_exact_transition_in_worker_validator_and_weights(monkeypatch, tmp_path, day):
    switch = datetime(2026, 10, day, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(CS, "GENERALIST_START_UTC", switch)
    cfg = C.SpecConfig("", "", "")
    validator = remote.RemoteSpecEvaluator(str(tmp_path / "state"))
    for seconds in (-1, 0, 1):
        now = switch + timedelta(seconds=seconds)
        want = ["demand_forecasting", "risk_assessment"] if seconds < 0 else ["generalist"]
        assert CS.active_tracks(now) == want
        assert [t.track_id for t in cfg.active_tracks(now.timestamp())] == want
        assert [t.track_id for t in validator.cfg.active_tracks(now.timestamp())] == want
        prof = CS.profile_for(now=now)
        assert prof.tracks == ((0.15, 0.15) if seconds < 0 else (0.30,))
        assert prof.engine == .20 and prof.inference == .50
        tracks = {t: CS.RankedLeaders([(10 + i * 3, 80), (11 + i * 3, 70), (12 + i * 3, 60)])
                  for i, t in enumerate(want)}
        weights = CS.combine_weights({1: 1.0}, 2, 1.0, tracks, now=now.timestamp(), floors={})
        assert sum(weights[u] for u in weights if u not in (1, 2)) == pytest.approx(.30)
        assert weights[1] == pytest.approx(.50)
        assert weights[2] == pytest.approx(.20)
    # Exact fractional-day end survives the evaluator conversion (an opening moved to 18:00).
    monkeypatch.setattr(CS, "GENERALIST_START_UTC", switch + timedelta(hours=6))
    cfg = C.SpecConfig("", "", "")
    assert cfg.tracks[0].end_ts == min(CS._PAIR_1_END, switch + timedelta(hours=6)).timestamp()
    assert cfg.tracks[0].active(cfg.tracks[0].end_ts - 1)
    assert not cfg.tracks[0].active(cfg.tracks[0].end_ts)


def test_unclaimed_generalist_share_returns_to_inference():
    at = CS.GENERALIST_START_UTC.timestamp()
    assert CS.combine_weights({1: .6, 2: .4}, 3, 1, {}, now=at) == pytest.approx({1: .48, 2: .32, 3: .2})
    assert CS.combine_weights({1: .6, 2: .4}, None, 0, {}, now=at) == {1: .6, 2: .4}


def test_pre_switch_fixed_vector_matches_live_rc9p5_byte_for_byte(tmp_path):
    path = tmp_path / "live_split.py"
    path.write_bytes(subprocess.check_output(["git", "show", "dac09bb:src/reax_subnet/competition_split.py"]))
    import sys
    spec = importlib.util.spec_from_file_location("live_split", path)
    live = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = live
    spec.loader.exec_module(live)
    try:
        base = {0: .1, 1: .3, 2: .25, 3: .2, 4: .15}
        kw = dict(owner_uid=0, burn_fraction=.1, now=CS.GENERALIST_START_UTC.timestamp() - 1,
                  keepalive={"demand_forecasting": [(6, 60)]}, provisional={"risk_assessment": [7]})
        tracks = {"demand_forecasting": [(10, 50), (11, 40)], "risk_assessment": [(12, 60)]}
        old = live.combine_weights(base, 8, .75, tracks, **kw)
        new = CS.combine_weights(base, 8, .75, tracks, **kw)
        assert json.dumps(new, sort_keys=True).encode() == json.dumps(old, sort_keys=True).encode()
    finally:
        del sys.modules[spec.name]


def pool():
    rows = []
    for cat in S.GENERALIST_CATEGORIES:
        for kind, count in (("choice", 90), ("noul", 60), ("score", 30)):
            for i in range(count):
                rows.append({"id": f"{cat}-{kind}-{i}", "category": cat, "question": {"type": kind},
                             "tier": ("proc_sealed", "same_family", "new_domain", "new_family", S.X_TIER)[i % 5]})
    return rows


def test_stratified_deterministic_type_proportions_and_use_limit():
    rows = pool()
    kw = dict(n=300, seed=42, generalist=True, x_mode="binding")
    got, fresh = S.draw(rows, **kw)
    assert fresh and len(got) == len({r["id"] for r in got}) == 300
    assert Counter(r["category"] for r in got) == {c: 30 for c in S.GENERALIST_CATEGORIES}
    for c in S.GENERALIST_CATEGORIES:
        assert Counter(r["question"]["type"] for r in got if r["category"] == c) == {"choice": 15, "noul": 10, "score": 5}
    assert S.draw(list(reversed(rows)), **kw) == (got, fresh)
    counts = {r["id"]: 24 for r in got}
    next_, _ = S.draw(rows, use_counts=counts, **kw)
    assert not {r["id"] for r in got} & {r["id"] for r in next_}
    counts = {r["id"]: 23 for r in rows}
    assert not S.draw(rows, use_counts=counts, **kw)[1]
    with pytest.raises(S.SampleError):
        S.draw(rows, use_counts={r["id"]: 24 for r in rows}, **kw)


def test_thin_category_and_exclusions_fail_closed():
    rows = pool()
    thin = [r for r in rows if r["category"] != S.GENERALIST_CATEGORIES[0]][:]
    thin += [r for r in rows if r["category"] == S.GENERALIST_CATEGORIES[0]][:29]
    with pytest.raises(S.SampleError, match="need 30"):
        S.draw(thin, n=300, seed=1, generalist=True, x_mode="binding")
    with pytest.raises(S.SampleError):
        S.draw(rows, n=300, seed=1, generalist=True,
               exclude={r["id"] for r in rows if r["category"] == S.GENERALIST_CATEGORIES[0]})


@pytest.mark.parametrize("category", [None, "wrong"])
def test_loader_rejects_missing_or_invalid_category(tmp_path, category):
    rows = pool()[:2]
    rows[0]["category"] = category
    dest = tmp_path / "generalist" / "sealed"
    dest.mkdir(parents=True)
    (dest / "sealed.clean.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(S.SampleError, match="category"):
        S.load_pool(str(tmp_path), "generalist")


def test_generalist_caps_and_specialists_stay_small():
    from test_open_class import facts
    f = facts(params=36_000_000_000, weight_gib=68)
    at = CS.GENERALIST_START_UTC.timestamp()
    for category in ("demand_forecasting", "risk_assessment", "generalist"):
        f.manifest["category"] = category
        for seconds in (-1, 0):
            bad = R.check_facts(f, category=category, hotkey=f.manifest["hotkey"], rules=R.rules_at(at + seconds, category=category))
            assert bool(bad) is (category != "generalist" or seconds < 0)


def test_category_scores_pooled_exactly_and_persisted(tmp_path):
    from test_keepalive import _sub
    st = ST.SpecState()
    sub = _sub(st, 1, ST.QUEUED, None, track="generalist")
    cat = S.GENERALIST_CATEGORIES[0]
    all_items = []
    for j, outcomes in enumerate(([True, False, True], [False, False])):
        items = [SC.ItemScore(f"{j}-{i}", "choice", "same_family", True, v, .25, 0, .2 if v else 1, .75) for i, v in enumerate(outcomes)]
        all_items += items
        draw = ST.Draw(str(j), 0, j, j, [it.item_id for it in items], item_categories={it.item_id: cat for it in items})
        E.SpecEvaluator._category_diagnostics(sub, draw, items)
        E.SpecEvaluator._category_diagnostics(sub, draw, items)  # idempotent
    assert sub.category_pooled[cat]["n"] == 5
    assert sub.category_pooled[cat]["score"] == pytest.approx(SC.score_items(all_items).total)
    assert sub.score is None and len(sub.category_draws) == 2
    ST.save(st, str(tmp_path / "s"))
    loaded = ST.load(str(tmp_path / "s")).track("generalist").submissions[sub.key]
    assert loaded.category_pooled == sub.category_pooled


@pytest.mark.parametrize("gate", ["floor", "g_med", "mode"])
async def test_uncalibrated_generalist_does_not_run(tmp_path, monkeypatch, gate):
    from unittest.mock import Mock
    from spec_helpers import commit, neuron, snapshot, write_pool
    from test_spec_evaluator import Env
    env = Env(tmp_path, {}, sample=300)
    env.cfg.tracks = C.DEFAULT_TRACKS[-1:]
    env.cfg.x_tier_mode = "off" if gate == "mode" else "binding"
    env.cfg.g_med = {"generalist": 0.0}
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", math.inf if gate == "floor" else 0.0)
    monkeypatch.setattr(CS, "GENERALIST_G_MED", None if gate == "g_med" else 0.0)
    write_pool(str(tmp_path / "sealed"), "generalist", pool())
    env.add(1, "a/general", .95, category="generalist")
    now = CS.GENERALIST_START_UTC.timestamp()
    snap = snapshot(100, [neuron(1, commit("generalist", "a/general"))])
    env.ev._sync(env.cfg.tracks[0], snap, now)
    assert len(env.ev.state.track("generalist").submissions) == 1
    draw = Mock(side_effect=AssertionError("uncalibrated track drew items"))
    monkeypatch.setattr(S, "draw", draw)
    await env.ev._run_track(env.cfg.tracks[0], snap, now)
    assert env.hub.fetches == []
    draw.assert_not_called()
    assert not env.ev.state.eval_times


async def test_commit_notice_gate_and_all_preopening_commitments_share_cohort(tmp_path, monkeypatch):
    from spec_helpers import commit, neuron, snapshot
    switch = CS.GENERALIST_START_UTC.timestamp()
    notice = switch - 3 * 86400
    monkeypatch.setattr(CS, "GENERALIST_COMMIT_FROM_UTC", datetime.fromtimestamp(notice, timezone.utc))
    cfg = C.SpecConfig("", str(tmp_path / "w"), str(tmp_path / "s"))
    now = [notice - 1]
    ev = E.SpecEvaluator(cfg, clock=lambda: now[0])
    a = neuron(1, commit("generalist", "a/model1"))
    b = neuron(2, commit("generalist", "a/model2"))
    await ev.evaluate_cycle(snapshot(100, [a]))
    assert not ev.state.track("generalist").submissions
    now[0] = notice
    await ev.evaluate_cycle(snapshot(101, [a]))
    now[0] = switch - 1
    await ev.evaluate_cycle(snapshot(200, [a, b]))
    ts = ev.state.track("generalist")
    subs = list(ts.submissions.values())
    assert len(subs) == 2 and {s.first_seen_block for s in subs} == {101, 200}
    assert {ev._cohort_block(s) for s in subs} == {0}
    for s in subs:
        s.status, s.confirm_draw, s.confirm_score = ST.PENDING_CONFIRM, f"d{s.uid}", 80.0
    ready, pending = ev._cohort(ts, 0)
    assert len(ready) == 2 and not pending
    # A post-opening arrival is a later cohort.
    now[0] = switch + 1
    c = neuron(3, commit("generalist", "a/model3"))
    ev._sync(cfg.tracks[-1], snapshot(201, [a, b, c]), now[0])
    late = next(s for s in ts.submissions.values() if s.uid == 3)
    assert ev._cohort_block(late) == 201


def test_combined_keepalive_bounded_at_quarter_of_generalist_share():
    at = CS.GENERALIST_START_UTC.timestamp()
    weights = CS.combine_weights({1: 1}, None, 0, {}, now=at, floors={},
                                 keepalive={"generalist": [(2, 80)]}, provisional={"generalist": [3]})
    assert weights.get(2, 0) + weights.get(3, 0) <= .30 * .25 + 1e-12


def test_tier_totals_are_preserved():
    rows, _ = S.draw(pool(), n=300, seed=42, generalist=True, x_mode="binding")
    counts = Counter(r["tier"] for r in rows)
    assert counts["proc_sealed"] == 45
    assert counts[S.X_TIER] == 60


def test_base_budget_switch_and_operator_override():
    cfg = C.SpecConfig("", "", "")
    at = CS.GENERALIST_START_UTC.timestamp()
    assert cfg.base_budget_for(at - 1) == 40 * R.GIB
    assert cfg.base_budget_for(at) == 80 * R.GIB
    cfg = C.from_env({"REAX_SPEC_SEALED_DIR": "/pool", "REAX_SPEC_BASE_BUDGET_GB": "120"})
    assert cfg.base_budget_for(at - 1) == cfg.base_budget_for(at) == 120 * R.GIB


def test_category_binding_cannot_change_after_use(tmp_path):
    dest = tmp_path / "generalist" / "sealed"
    dest.mkdir(parents=True)
    rows = pool()
    (dest / "sealed.clean.jsonl").write_text("\n".join(map(json.dumps, rows)))
    cfg = C.SpecConfig(str(tmp_path), str(tmp_path / "w"), str(tmp_path / "s"))
    ev = E.SpecEvaluator(cfg)
    ev.state.track("generalist").item_categories[rows[0]["id"]] = S.GENERALIST_CATEGORIES[1]
    with pytest.raises(S.SampleError, match="changed"):
        ev._pool("generalist")


async def test_generalist_first_and_confirmation_draws_store_diagnostics(tmp_path, monkeypatch):
    from spec_helpers import commit, neuron, snapshot, synthetic_pool, write_pool
    from test_spec_evaluator import Env
    env = Env(tmp_path, {}, sample=300)
    env.cfg.tracks = C._from_schedule()[-1:]
    env.cfg.x_tier_mode = "binding"
    env.cfg.g_med = {"generalist": 0.0}
    monkeypatch.setattr(CS, "GENERALIST_G_MED", 0.0)
    monkeypatch.setattr(CS, "GENERALIST_COMMIT_FROM_UTC", CS.GENERALIST_START_UTC)
    env.cfg.base_fetch = False
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", 0.0)
    monkeypatch.setitem(CS.KEEPALIVE_NONPROC_FLOOR, "generalist", 0.0)
    env.clock.t = CS.GENERALIST_START_UTC.timestamp() + 3600
    rows = []
    for category in S.GENERALIST_CATEGORIES:
        for i, row in enumerate(synthetic_pool(600)):
            row["id"] = f"{category}-{row['id']}"
            row["category"] = category
            if i % 5 == 0:
                row["tier"] = S.X_TIER
            rows.append(row)
    write_pool(str(tmp_path / "sealed"), "generalist", rows)
    env.add(1, "a/general", .95, category="generalist")
    snap = snapshot(1000, [neuron(1, commit("generalist", "a/general"))])
    await env.ev.evaluate_cycle(snap)
    ts = env.ev.state.track("generalist")
    sub = next(iter(ts.submissions.values()))
    assert sub.status == ST.PENDING_CONFIRM
    assert len(sub.category_draws) == 1
    assert {c: d["n"] for c, d in sub.category_pooled.items()} == {c: 30 for c in S.GENERALIST_CATEGORIES}
    env.clock.advance(13 * 3600)
    await env.ev.evaluate_cycle(snapshot(5000, snap.neurons))
    assert sub.status == ST.CHAMPION
    assert {c: d["n"] for c, d in sub.category_pooled.items()} == {c: 60 for c in S.GENERALIST_CATEGORIES}


def test_remote_validator_generalist_keepalive_uses_base_floor(tmp_path, monkeypatch):
    from test_open_class import _remote_state, BIG, SMALL
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", 0)
    monkeypatch.setitem(CS.KEEPALIVE_NONPROC_FLOOR, "generalist", 0)
    now = CS.GENERALIST_START_UTC.timestamp() + 3600
    ev, snap, _ = _remote_state(tmp_path, now, [(1, BIG), (2, SMALL)])
    active = ev.cfg.active_tracks(now)
    assert [t.track_id for t in active] == ["generalist"]
    ev.leaders(active, snap)
    assert [u for u, _ in ev.keepalive(active, snap)["generalist"]] == [2]


def test_uncalibrated_generalist_cannot_pay_provisional_entries(monkeypatch):
    from generalist_defaults import defaults
    from test_provisional_ka import _p, _snapr
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", defaults()["GENERALIST_ENTRY_FLOOR"])  # the release sets both
    monkeypatch.setattr(CS, "GENERALIST_G_MED", defaults()["GENERALIST_G_MED"])
    st = ST.SpecState()
    now = CS.GENERALIST_START_UTC.timestamp() + 3600
    sub = _p(st, 1, track="generalist", since=now - 60)
    snap = _snapr([sub])
    assert E.compute_provisional(st, ["generalist"], snap, now=now) == {"generalist": []}
    sub.provisional_since_ts = None
    assert not E.mark_provisional(st, ["generalist"], snap, now)


def test_category_tier_quotas_borrow_across_types_and_missing_x():
    for shape in ("x_choice", "proc_tiny", "no_x"):
        rows = pool()
        for row in rows:
            kind = row["question"]["type"]
            if row["tier"] == S.X_TIER and (shape == "no_x" or kind != "choice"):
                row["tier"] = "same_family"
        if shape == "proc_tiny":
            rows = [r for r in rows if r["question"]["type"] == "choice" or int(r["id"].rsplit("-", 1)[1]) == 0]
            for row in rows:
                if row["question"]["type"] != "choice":
                    row["tier"] = "proc_sealed"
        got, _ = S.draw(rows, n=300, seed=1, generalist=True, x_mode="binding")
        assert Counter(r["category"] for r in got) == {c: 30 for c in S.GENERALIST_CATEGORIES}
        counts = Counter(r["tier"] for r in got)
        assert counts["proc_sealed"] == 45
        assert counts[S.X_TIER] == (0 if shape == "no_x" else 60)
        shortfalls = S.generalist_shortfalls(got, "binding", S.X_TIER_SHARE)
        assert all(d[S.X_TIER] == (6 if shape == "no_x" else 0) for d in shortfalls.values())


def test_degenerate_category_still_fills_thirty():
    rows = pool()
    for row in rows:
        row["tier"] = "proc_sealed"
    got, _ = S.draw(rows, n=300, seed=1, generalist=True, x_mode="binding")
    assert Counter(r["category"] for r in got) == {c: 30 for c in S.GENERALIST_CATEGORIES}


def test_consecutive_draws_only_fail_on_thin_category():
    rows, use = pool(), Counter()
    draws = 0
    while True:
        try:
            got, _ = S.draw(rows, n=300, seed=draws, generalist=True, x_mode="binding", use_counts=use)
        except S.SampleError as exc:
            assert "eligible items, need 30" in str(exc)
            break
        use.update(r["id"] for r in got)
        draws += 1
        assert draws <= 144  # 180 * 24 uses/category, 30 per draw
    assert draws > 108  # critic's even-tier probe stalled at 108 on a type bucket
    assert max(use.values()) <= 24


def test_x_shortfall_is_persisted_on_draw_not_pool_categories(tmp_path):
    from spec_helpers import snapshot, write_pool
    rows = pool()
    for row in rows:
        if row["tier"] == S.X_TIER:
            row["tier"] = "same_family"
    write_pool(str(tmp_path), "generalist", rows)
    cfg = C.SpecConfig(str(tmp_path), str(tmp_path / "w"), str(tmp_path / "s"), x_tier_mode="binding",
                       g_med={"generalist": 0.0})
    ev = E.SpecEvaluator(cfg)
    ts = ev.state.track("generalist")
    ev._pool("generalist")
    assert not ts.item_categories
    # A never-drawn row may be corrected on refresh.
    rows[0]["category"] = S.GENERALIST_CATEGORIES[1]
    write_pool(str(tmp_path), "generalist", rows)
    ev._pools.clear()
    ev._pool("generalist")
    assert not ts.item_categories
    got, fresh = S.draw(rows, n=300, seed=1, generalist=True, x_mode="binding")
    draw_id = ev._record_draw(ts, rows, got, fresh, 1, snapshot(100, []), 1.0)
    assert len(ts.item_categories) == 300
    assert ts.draws[draw_id].x_shortfall == {c: 6 for c in S.GENERALIST_CATEGORIES}
    restored = ST.load(cfg.state_path).track("generalist")
    assert restored.draws[draw_id].x_shortfall == ts.draws[draw_id].x_shortfall
    assert len(restored.item_categories) == 300


def test_pruning_category_draws_keeps_pooled_and_pending(tmp_path):
    ts = ST.SpecState().track("generalist")
    sub = ST.Submission("k", 1, "hk", "ck", "generalist", "a/b", "a" * 40, 1, 1,
                        status=ST.PENDING_CONFIRM, draw1="0")
    ts.submissions[sub.key] = sub
    sub.category_pooled = {"math_quant": {"n": 100}}
    for i in range(E.MAX_UNREFERENCED_DRAWS + 10):
        d = str(i)
        ts.draws[d] = ST.Draw(d, i, i, i, [])
        sub.category_draws[d] = {"math_quant": {"n": 30}}
    E.SpecEvaluator._prune_draws(ts, now=1000)
    assert "0" in ts.draws
    assert len(ts.draws) == E.MAX_UNREFERENCED_DRAWS + 1
    assert set(sub.category_draws) == set(ts.draws)
    assert sub.category_pooled == {"math_quant": {"n": 100}}


def test_source_gmed_is_required_for_provisional(monkeypatch):
    from test_provisional_ka import _p, _snapr
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", 0.0)
    st = ST.SpecState()
    now = CS.GENERALIST_START_UTC.timestamp() + 3600
    sub = _p(st, 1, track="generalist", since=now - 60)
    snap = _snapr([sub])
    monkeypatch.setattr(CS, "GENERALIST_G_MED", None)
    assert E.compute_provisional(st, ["generalist"], snap, now=now) == {"generalist": []}
    monkeypatch.setattr(CS, "GENERALIST_G_MED", 0.0)
    assert E.compute_provisional(st, ["generalist"], snap, now=now) == {"generalist": [1]}


async def test_unset_commit_notice_never_syncs(tmp_path, monkeypatch):
    from spec_helpers import commit, neuron, snapshot
    monkeypatch.setattr(CS, "GENERALIST_COMMIT_FROM_UTC", None)
    now = CS.GENERALIST_START_UTC.timestamp()
    cfg = C.SpecConfig("", str(tmp_path), str(tmp_path / "s"))
    ev = E.SpecEvaluator(cfg, clock=lambda: now)
    await ev.evaluate_cycle(snapshot(100, [neuron(1, commit("generalist", "a/b"))]))
    assert not ev.state.track("generalist").submissions


def test_cli_default_tracks_transition(tmp_path):
    # cli.validator_main passes this exact import-time sequence to the remote evaluator.
    from reax_subnet import cli
    from reax_subnet.config import load_validator
    from test_remote_competitions import FINNEY
    cfg = load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "remote",
                          "REAX_COMPETITIONS_STATE_DIR": str(tmp_path)})
    _, driver = cli._engine_competition(cfg)
    validator = driver.spec.evaluator
    assert tuple(validator.cfg.tracks) == C.DEFAULT_TRACKS
    start = CS.GENERALIST_START_UTC.timestamp()
    assert [t.track_id for t in validator.cfg.active_tracks(start - 1)] == ["demand_forecasting", "risk_assessment"]
    assert [t.track_id for t in validator.cfg.active_tracks(start)] == ["generalist"]


def test_precommit_budget_is_deferred_and_separate(tmp_path):
    from spec_helpers import commit, neuron, snapshot
    start = CS.GENERALIST_START_UTC.timestamp()
    cfg = C.SpecConfig("", str(tmp_path), str(tmp_path / "s"))
    ev = E.SpecEvaluator(cfg)
    df, ra, general = cfg.tracks
    # Four DF/RA submissions in the previous week cannot reject a generalist entry.
    for i in range(4):
        ts = ev.state.track(df.track_id if i % 2 else ra.track_id)
        sub = ST.Submission(str(i), i, f"hk{i}", "cold", ts.track_id, "a/b", "a" * 40, i, start - 86400,
                            status=ST.NOT_BETTER)
        ts.submissions[sub.key] = sub
    ns = [neuron(i, commit("generalist", f"a/m{i}"), coldkey="cold") for i in range(5, 8)]
    ev._sync(general, snapshot(100, ns), start - 3600)
    ts = ev.state.track("generalist")
    assert all(s.status == ST.QUEUED and not s.generalist_budget_checked for s in ts.submissions.values())
    ev._sync(general, snapshot(101, ns), start)
    subs = sorted(ts.submissions.values(), key=lambda s: s.uid)
    assert [s.status for s in subs] == [ST.QUEUED, ST.QUEUED, ST.REJECTED]
    assert all(s.generalist_budget_checked for s in subs)
    # A pre-commit cannot turn three historical DF/RA entries into a weekly-budget rejection.
    ev.state.track(df.track_id).submissions.pop("1")
    candidate = ST.Submission("new", 9, "new", "cold", df.track_id, "a/b", "a" * 40, 100, start - 1)
    assert ev._budget_violation(candidate, start - 1) == ""


def test_generalist_weekly_budget_applies_at_opening(tmp_path):
    from spec_helpers import commit, neuron, snapshot
    start = CS.GENERALIST_START_UTC.timestamp()
    cfg = C.SpecConfig("", str(tmp_path), str(tmp_path / "s"))
    ev = E.SpecEvaluator(cfg)
    tdef = cfg.tracks[-1]
    for i in range(1, 6):
        ns = [neuron(j, commit("generalist", f"a/m{j}"), coldkey="cold") for j in range(1, i + 1)]
        ev._sync(tdef, snapshot(100 + i, ns), start + i)
        sub = next(s for s in ev.state.track("generalist").submissions.values() if s.uid == i)
        if i <= 4:
            assert sub.status == ST.QUEUED
            sub.status = ST.NOT_BETTER  # no longer open, still counts in weekly budget
        else:
            assert sub.status == ST.REJECTED and "7 days" in sub.reason


@pytest.mark.parametrize("offset,expired", [(0, False), (86400, False), (86401, True)])
def test_preopening_publish_deadline_uses_time_since_opening(tmp_path, offset, expired):
    from spec_helpers import snapshot
    from reax_subnet.spec import fetch as F
    start = CS.GENERALIST_START_UTC.timestamp()
    cfg = C.SpecConfig("", str(tmp_path), str(tmp_path / "s"))
    ev = E.SpecEvaluator(cfg)
    ts = ev.state.track("generalist")
    sub = ST.Submission("k", 1, "hk", "ck", "generalist", "a/b", "a" * 40, 1, start - 3 * 86400)
    ts.submissions[sub.key] = sub
    ev._fetch_failure(cfg.tracks[-1], sub, snapshot(100000, []), start + offset, F.FetchError("unavailable", "private"))
    assert (sub.status == ST.REJECTED) == expired
    if expired:
        assert "24 hours after generalist opening" in sub.reason


def test_postopening_publish_deadline_keeps_block_semantics(tmp_path):
    from spec_helpers import snapshot
    from reax_subnet.spec import fetch as F
    start = CS.GENERALIST_START_UTC.timestamp()
    cfg = C.SpecConfig("", str(tmp_path), str(tmp_path / "s"))
    ev = E.SpecEvaluator(cfg)
    sub = ST.Submission("k", 1, "hk", "ck", "generalist", "a/b", "a" * 40, 100, start + 1)
    ev.state.track("generalist").submissions[sub.key] = sub
    ev._fetch_failure(cfg.tracks[-1], sub, snapshot(100 + C.PUBLISH_DEADLINE_BLOCKS + 1, []),
                      start + 60, F.FetchError("unavailable", "private"))
    assert sub.status == ST.REJECTED


def test_dryrun_cli_outputs_aggregates_only(tmp_path):
    from pathlib import Path
    import sys
    rows = pool()
    for row in rows:
        row["state"] = {"secret": "PRIVATE_ITEM_TEXT"}
    path = tmp_path / "pool.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    tool = Path(__file__).resolve().parents[1] / "tools" / "generalist_draw_dryrun.py"
    result = subprocess.run([sys.executable, str(tool), str(path), "--pairs", "3"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["draws_possible"] == 6 and report["pairs_possible"] == 3
    assert "PRIVATE_ITEM_TEXT" not in result.stdout + result.stderr
    assert not any(r["id"] in result.stdout for r in rows)


def test_release_defaults_and_invalid_notice_fail_closed():
    import re
    from pathlib import Path
    from generalist_defaults import defaults
    d = defaults()  # what scripts/set_generalist_start.py default restores: the committed fail-closed state
    assert d["GENERALIST_COMMIT_FROM_UTC"] is None
    assert math.isinf(d["GENERALIST_ENTRY_FLOOR"]) and math.isinf(d["GENERALIST_KEEPALIVE_NONPROC_FLOOR"])
    assert d["GENERALIST_G_MED"] is None
    assert CS.KEEPALIVE_NONPROC_FLOOR["generalist"] == CS.GENERALIST_KEEPALIVE_NONPROC_FLOOR  # the table follows the constant
    assert CS.TRACK_ENTRY_FLOOR["generalist"] == CS.GENERALIST_ENTRY_FLOOR
    source = Path(CS.__file__).read_text().split("@dataclass", 1)[0]
    invalid, n = re.subn(r"^GENERALIST_COMMIT_FROM_UTC: .*$",
                         "GENERALIST_COMMIT_FROM_UTC = GENERALIST_START_UTC + timedelta(seconds=1)", source, flags=re.M)
    assert n == 1
    with pytest.raises(AssertionError):
        exec(compile(invalid, CS.__file__, "exec"), {})


def test_opening_cohort_goes_before_post_opening_entries_on_vacant_track():
    # rc10.post3 (#23162): immunity order put post-opening hotkeys past immunity before still-immune cohort members
    switch = CS.GENERALIST_START_UTC.timestamp()
    ts = ST.TrackState(track_id="generalist")

    def sub(uid, seen_ts, block):
        s = ST.Submission(key=f"k{uid}", uid=uid, hotkey=f"hk{uid}", coldkey=f"c{uid}", track_id="generalist", repo="m/m",
                          revision="a" * 40, first_seen_block=block, first_seen_ts=seen_ts, status=ST.QUEUED)
        ts.submissions[s.key] = s
        return s

    post_a, cohort_a, post_b, cohort_b = (sub(17, switch + 60, 300), sub(250, switch - 3600, 100),
                                          sub(91, switch + 120, 301), sub(65, switch - 60, 150))
    order = [post_a, cohort_a, post_b, cohort_b]                       # as the immunity order returned it
    assert [s.uid for s in E.SpecEvaluator._opening_cohort_first(ts, order)] == [250, 65, 17, 91]
    ts.champion = cohort_a.key                                         # crowned: the plain queue order again
    assert [s.uid for s in E.SpecEvaluator._opening_cohort_first(ts, order)] == [17, 250, 91, 65]
    df = ST.TrackState(track_id="demand_forecasting")
    assert E.SpecEvaluator._opening_cohort_first(df, order) == order


def _reentry_ts():
    # rc10.post4 (#23203/#23204/#23211): pruned opening-cohort entry 167 + live entries, as on 21 Oct
    switch = CS.GENERALIST_START_UTC.timestamp()
    ts = ST.TrackState(track_id="generalist")

    def sub(key, uid, hotkey, coldkey, seen_ts, block, rev="a" * 40, repo="d/pplx-s3v", **kw):
        s = ST.Submission(key=key, uid=uid, hotkey=hotkey, coldkey=coldkey, track_id="generalist", repo=repo,
                          revision=rev, first_seen_block=block, first_seen_ts=seen_ts, **kw)
        ts.submissions[s.key] = s
        return s

    old = sub("hk167|c", 167, "hk167", "ck1", switch - 60, 100, status=ST.WITHDRAWN, withdrawn_status=ST.PENDING_CONFIRM,
              withdrawn_kind="pruned", withdrawn_ts=switch + 3 * 3600, confirm_after_ts=switch + 3600, draw1="d1",
              first_gain=1.0, score=76.4, hash_named="h1")
    return switch, ts, sub, old


@pytest.mark.parametrize("hotkey,coldkey,rev,repo,late,match", [
    ("hk167", "other", "a" * 40, "d/pplx-s3v", 0, True),     # same hotkey re-registered
    ("hkX", "ck1", "a" * 40, "d/pplx-s3v", 0, True),         # another hotkey of the same coldkey
    ("hkX", "other", "a" * 40, "d/pplx-s3v", 0, False),      # a reused uid / foreign keys prove nothing
    ("hk167", "ck1", "b" * 40, "d/pplx-s3v", 0, False),      # another revision (CharlieBT uid 91 case)
    ("hk167", "ck1", "a" * 40, "d/other", 0, False),         # another repo
    ("hk167", "ck1", "a" * 8, "d/pplx-s3v", 0, False),       # not a full revision
    ("hk167", "ck1", "a" * 40, "d/pplx-s3v", 1, False),      # first seen after 22 Oct 07:30:00Z
])
def test_reentry_match_terms(hotkey, coldkey, rev, repo, late, match):
    _, ts, sub, old = _reentry_ts()
    new = sub("new", 91, hotkey, coldkey, E.REENTRY_DEADLINE_TS + late, 900, rev=rev, repo=repo, status=ST.QUEUED)
    assert E.reentry_match(old, new) is match
    assert E.SpecEvaluator._reentrant(ts, new) is match


def test_reentry_holds_opening_playoff_until_deadline_and_goes_first():
    switch, ts, sub, old = _reentry_ts()
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    ready = sub("r113", 113, "hk113", "ck113", switch - 900, 90, status=ST.PENDING_CONFIRM, confirm_draw="c1",
                confirm_score=75.0, draw1="d0")
    post = sub("p17", 17, "hk17", "ck17", switch + 60, 300, repo="x/y", status=ST.QUEUED)
    ev.clock = lambda: E.REENTRY_DEADLINE_TS - 1
    r, open_ = ev._cohort(ts, 0)
    assert r == [ready] and open_ == [old]                       # the playoff waits for the pruned entry
    ev.clock = lambda: E.REENTRY_DEADLINE_TS
    assert ev._cohort(ts, 0)[1] == []                            # deadline passed: no further wait
    back = sub("hk167|c2", 205, "hk167", "ck1", E.REENTRY_DEADLINE_TS - 10, 950, status=ST.QUEUED)
    assert ev._cohort(ts, 0)[1] == [back]                         # a re-entry seen by then still holds while it is open
    assert [s.uid for s in E.SpecEvaluator._opening_cohort_first(ts, [post, back])] == [205, 17]
    back.reason = "waiting for the repo: 404"
    assert ev._cohort(ts, 0)[1] == []                             # an unavailable repo never blocks the cohort


def test_reentry_carryover_keeps_opening_cohort_place():
    switch, ts, sub, old = _reentry_ts()
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    new = sub("hk167|c2", 205, "hk167", "ck1", switch + 6 * 3600, 950, status=ST.QUEUED, hash_named="h1")
    snap = type("Snap", (), {"neurons": [type("N", (), {"hotkey": "hk167", "registered_block": 940})()]})()
    ev._carryover(ts, new, snap, switch + 7 * 3600)
    assert new.carried_from == old.key and new.status == ST.PENDING_CONFIRM and new.score == old.score
    assert E.SpecEvaluator._cohort_block(new) == 0 and new.first_seen_block == 950
    assert not E.SpecEvaluator._reentry_waiting(old)
    # same tensors under another repo (rule 17 Oct carry-over): the draw travels, the opening place does not
    _, ts2, sub2, old2 = _reentry_ts()
    other2 = sub2("hk9|c", 9, "hk9", "ck1", switch + 6 * 3600, 951, repo="d/renamed", status=ST.QUEUED,
                  hash_named="h1")
    ev._carryover(ts2, other2, snap, switch + 7 * 3600)
    assert other2.carried_from == old2.key and E.SpecEvaluator._cohort_block(other2) == 951


def test_reentry_window_and_playoff_cap_follow_the_deadline():
    # critic r1 F1/F2: the 72 h carry-over window and the 12 h cohort cap never cut a re-entry before 07:30Z
    from types import SimpleNamespace
    switch, ts, sub, old = _reentry_ts()
    old.withdrawn_ts = None                                       # legacy record: 72 h window already over
    new = sub("hk167|c2", 205, "hk167", "ck1", E.REENTRY_DEADLINE_TS - 60, 950, status=ST.QUEUED, hash_named="h1")
    assert E.carryover_eligible(old, new, set(), new.first_seen_ts)
    foreign = sub("hkF|c", 206, "hkF", "ckF", E.REENTRY_DEADLINE_TS - 60, 951, status=ST.QUEUED, hash_named="h1")
    assert not E.carryover_eligible(old, foreign, set(), foreign.first_seen_ts)
    del ts.submissions[new.key], ts.submissions[foreign.key]
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    ev._crown = lambda *a, **k: None
    ts.draws["c1"] = SimpleNamespace(ts=switch + 3600)
    for uid in (113, 199):
        sub(f"r{uid}", uid, f"hk{uid}", f"ck{uid}", switch - 900, 90, status=ST.PENDING_CONFIRM, confirm_draw="c1",
            confirm_score=75.0 - uid / 1000, draw1="d0")
    ev.clock = lambda: E.REENTRY_DEADLINE_TS - 1                  # far past the 12 h cap, before the deadline
    assert ev._cohort_step(ts, E.REENTRY_DEADLINE_TS - 1) == []
    ev.clock = lambda: E.REENTRY_DEADLINE_TS
    assert [s.uid for s in ev._cohort_step(ts, E.REENTRY_DEADLINE_TS)] == [113, 199]
    old.withdrawn_kind = "changed"                                # critic r1 F4: only pruned records wait
    assert not E.SpecEvaluator._reentry_waiting(old)


def test_reentry_is_not_a_new_weekly_submission_and_takes_its_own_original():
    switch, ts, sub, old = _reentry_ts()
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    ev.state = ST.SpecState()
    ev.state.tracks["generalist"] = ts
    now = switch + 6 * 3600
    for i in range(3):
        sub(f"w{i}", 300 + i, f"hkw{i}", "ck1", switch + i, 400 + i, repo=f"d/w{i}", status=ST.NOT_BETTER,
            generalist_budget_checked=True)
    old.generalist_budget_checked = True
    new = sub("hk167|c2", 205, "hk167", "ck1", now, 950, status=ST.QUEUED, hash_named="h1")
    assert ev._budget_violation(new, now) == ""                  # 3 + the re-entry; the original is not counted twice
    other = sub("hkO|c", 207, "hkO", "ck1", now, 951, repo="d/new", status=ST.QUEUED)
    assert "submissions in 7 days" in ev._budget_violation(other, now)
    # an earlier eligible record of the same tensors (another coldkey, same repo) does not take the re-entry's place
    early = sub("hkE|c", 50, "hkE", "ckE", switch - 7200, 10, status=ST.WITHDRAWN, withdrawn_status=ST.PENDING_CONFIRM,
                withdrawn_kind="pruned", withdrawn_ts=switch + 3600, confirm_after_ts=switch + 3600, draw1="dE",
                first_gain=1.0, score=70.0, hash_named="h1")
    snap = type("Snap", (), {"neurons": []})()
    ev._carryover(ts, new, snap, now)
    assert new.carried_from == old.key and not early.carried_to and E.SpecEvaluator._cohort_block(new) == 0
