"""Rules v4, change 3a: generalist draws of 900 items (90 per category) and margin floor 1.0 from DRAWS_V4_START_UTC."""
import json
import math
import subprocess
import types
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from reax_subnet.spec import evaluator as E, machine as M, sampling as S

GATE = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
TS = GATE.timestamp()
CATS = S.GENERALIST_CATEGORIES


def pool(per_cat=200):
    rows = []
    for cat in CATS:
        for i in range(per_cat):
            kind = ("choice", "choice", "noul", "score")[i % 4] if i % 2 else ("choice", "noul", "score")[i % 3]
            rows.append({"id": f"{cat}-{i}", "category": cat, "question": {"type": kind},
                         "tier": ("proc_sealed", "same_family", "new_domain", "new_family", S.X_TIER)[i % 5]})
    return rows


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setattr(S, "DRAWS_V4_START_UTC", GATE)


def test_release_constant_is_t1():
    assert S.DRAWS_V4_START_UTC == datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)  # rc11
    assert S.GENERALIST_PER_CATEGORY_V4 == 90 and M.MARGIN_FLOOR_V4 == 1.0
    assert (S.SAMPLE_SIZE, S.GENERALIST_PER_CATEGORY, M.MARGIN_FLOOR) == (300, 30, 1.5)


@pytest.mark.parametrize("now", [None, 0.0, TS, TS + 10 ** 9, GATE, datetime(2030, 1, 1, tzinfo=timezone.utc)])
def test_gate_none_is_the_old_rules_whatever_the_time(monkeypatch, now):
    monkeypatch.setattr(S, "DRAWS_V4_START_UTC", None)
    assert not S.v4_active(now)
    assert S.generalist_per_category(now) == 30 and S.draw_size(300, True, now) == 300
    assert M.margin_floor(now) == 1.5 and M.required_margin(0.2, now) == 1.5 and M.close_min_gain(now) == 0.75


def test_gate_before_boundary_after(gated):
    before, boundary, after = TS - 1, TS, TS + 1
    assert [S.v4_active(t) for t in (before, boundary, after)] == [False, True, True]
    assert [S.v4_active(t) for t in (GATE - timedelta(microseconds=1), GATE, GATE + timedelta(seconds=1))] == [False, True, True]
    assert [S.generalist_per_category(t) for t in (before, boundary, after)] == [30, 90, 90]
    assert [S.draw_size(300, True, t) for t in (before, boundary, after)] == [300, 900, 900]
    assert [M.margin_floor(t) for t in (before, boundary, after)] == [1.5, 1.0, 1.0]
    assert [M.close_min_gain(t) for t in (before, boundary, after)] == [0.75, 0.5, 0.5]
    # no time given: the old rules even when the gate is set
    assert S.generalist_per_category() == 30 and M.margin_floor() == 1.5 and M.required_margin(0.1) == 1.5
    # a naive datetime is read as UTC
    assert S.v4_active(GATE.replace(tzinfo=None)) and not S.v4_active(GATE.replace(tzinfo=None) - timedelta(seconds=1))


def test_required_margin_uses_floor_or_two_se(gated):
    assert M.required_margin(0.3, TS - 1) == 1.5 and M.required_margin(0.3, TS) == 1.0
    assert M.required_margin(0.9, TS) == pytest.approx(1.8) and M.required_margin(0.9, TS - 1) == pytest.approx(1.8)
    assert M.first_pass(10.0, 8.8, 0.3, now=TS - 1).passed is False
    fp = M.first_pass(10.0, 8.8, 0.3, now=TS)
    assert fp.passed and fp.delta == 1.0
    assert M.first_pass(10.0, 8.8, 0.3).delta == 1.5   # no time: old floor
    assert M.confirm(10.0, 8.8, 1.5, new_incumbent=True, se=0.3, now=TS) is True
    assert M.confirm(10.0, 8.8, 1.5, new_incumbent=True, se=0.3, now=TS - 1) is False


def test_draw_size_other_than_the_live_one_is_refused(gated):
    rows = pool()
    with pytest.raises(S.SampleError, match="exactly 900"):
        S.draw(rows, n=300, seed=1, generalist=True, now=TS)
    with pytest.raises(S.SampleError, match="exactly 300"):
        S.draw(rows, n=900, seed=1, generalist=True, now=TS - 1)
    with pytest.raises(S.SampleError, match="exactly 300"):
        S.draw(rows, n=900, seed=1, generalist=True)
    assert S.draw_size(120, True, TS) == 120   # a non-default configured size is left to ``draw`` to refuse
    assert S.draw_size(300, False, TS) == 300   # specialist tracks keep their size


def test_old_draw_is_identical_with_and_without_time(monkeypatch):
    monkeypatch.setattr(S, "DRAWS_V4_START_UTC", None)
    rows = pool()
    kw = dict(n=300, seed=7, generalist=True, x_mode="binding")
    assert S.draw(rows, **kw) == S.draw(rows, now=TS, **kw) == S.draw(rows, now=1.0, **kw)


def test_draw_of_900_with_quotas(gated):
    rows = pool()
    got, fresh = S.draw(rows, n=900, seed=3, generalist=True, x_mode="binding", now=TS)
    assert fresh and len(got) == len({r["id"] for r in got}) == 900
    assert Counter(r["category"] for r in got) == {c: 90 for c in CATS}
    tiers = Counter(S.X_TIER if r["tier"] in S.X_TIERS else r["tier"] for r in got)
    assert sum(tiers.values()) == 900 and tiers["proc_sealed"] == 135 and tiers[S.X_TIER] == 180
    quotas = S.generalist_tier_quotas("binding", S.X_TIER_SHARE, 90)
    assert all(sum(q.values()) == 90 for q in quotas.values())
    assert sum(q["proc_sealed"] for q in quotas.values()) == 135 and sum(q[S.X_TIER] for q in quotas.values()) == 180
    for c in CATS:   # rounding is carried across categories, so a category is within one item of the exact share
        assert abs(quotas[c]["proc_sealed"] - 13.5) <= 1 and abs(quotas[c][S.X_TIER] - 18) <= 1
    assert not any(S.generalist_shortfalls(got, "binding", S.X_TIER_SHARE, 90)[c][t] for c in CATS for t in quotas[c])
    assert S.generalist_shortfalls(got, "binding", S.X_TIER_SHARE)[CATS[0]][S.X_TIER] == 0   # 30/category targets are met
    # deterministic, and a disjoint confirmation draw fits a 200 per category pool
    assert S.draw(list(reversed(rows)), n=900, seed=3, generalist=True, x_mode="binding", now=TS) == (got, fresh)
    second, _ = S.draw(rows, n=900, seed=4, generalist=True, x_mode="binding", now=TS, exclude={r["id"] for r in got})
    assert not {r["id"] for r in got} & {r["id"] for r in second} and len(second) == 900


def test_thin_category_for_900_items_fails_closed(gated):
    rows = [r for r in pool(per_cat=100) if r["tier"] not in S.X_TIERS]   # 80 usable per category
    with pytest.raises(S.SampleError, match="only 80 eligible items, need 90"):
        S.draw(rows, n=900, seed=1, generalist=True, now=TS)
    assert len(S.draw(rows, n=300, seed=1, generalist=True, now=TS - 1)[0]) == 300


def test_default_quotas_unchanged():
    q = S.generalist_tier_quotas("binding", S.X_TIER_SHARE)
    assert q == S.generalist_tier_quotas("binding", S.X_TIER_SHARE, 30)
    assert sum(sum(v.values()) for v in q.values()) == 300


def test_first_stage_v3_uses_the_floor_in_force(gated):
    # z = 2.4 (>= Z_PASS), gain 1.2: between the two floors
    args = (1.2, 0.5, 1.2, 0.5)
    assert M.first_stage_v3(*args, close_used=False) == "fail"
    assert M.first_stage_v3(*args, close_used=False, now=TS - 1) == "fail"
    assert M.first_stage_v3(*args, close_used=False, now=TS) == "pass"
    assert M.first_stage_v3(*args, close_used=False, now=TS + 86400) == "pass"
    # close call: z 1.4, gain 0.6 earns a close draw only at the half floor of 0.5
    assert M.first_stage_v3(0.6, 0.43, 0.6, 0.43, close_used=False, now=TS - 1) == "fail"   # 0.6 < 0.75
    assert M.first_stage_v3(0.6, 0.43, 0.6, 0.43, close_used=False, now=TS) == "close"
    assert M.first_stage_v3(0.6, 0.43, 0.6, 0.43, close_used=True, now=TS) == "fail"


def test_confirm_v3_uses_the_floor_in_force(gated):
    draws = [(1.2, 0.4), (1.2, 0.4)]   # pooled z 4.2, mean 1.2: above 1.0 only
    confirm = draws[1:]
    assert M.confirm_v3(draws, confirm, extra_used=False) == "fail"
    assert M.confirm_v3(draws, confirm, extra_used=False, now=TS - 1) == "fail"
    assert M.confirm_v3(draws, confirm, extra_used=False, now=TS) == "crown"
    assert M.confirm_v3(draws, confirm, extra_used=True, now=TS) == "crown"
    assert M.confirm_v3([(0.9, 0.2), (0.9, 0.2)], [(0.9, 0.2)], extra_used=False, now=TS) != "crown"   # below 1.0 never


def test_mixed_size_pooling_is_the_se_of_the_equal_weighted_mean():
    # a 300-item first stage (se 1.2) and a 900-item confirmation (se 0.4): equal-weighted mean, rms(se)/sqrt(2)
    mean, se = M.pooled([2.0, 1.0], [1.2, 0.4])
    assert mean == pytest.approx(1.5)
    assert se == pytest.approx(math.sqrt((1.2 ** 2 + 0.4 ** 2) / 2) / math.sqrt(2))
    assert se == pytest.approx(0.5 * math.sqrt(1.2 ** 2 + 0.4 ** 2))
    # it is the SE of the plain mean of two independent draws with those SEs, so pooling stays valid across sizes
    assert M.pooled([3.0], [0.4]) == (3.0, 0.4)


def test_evaluator_draw_size_follows_the_draw_time(gated):
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    ev.cfg = types.SimpleNamespace(sample_size=S.SAMPLE_SIZE)
    assert [ev._draw_n("generalist", t) for t in (TS - 1, TS, TS + 1)] == [300, 900, 900]
    assert [ev._draw_n("demand_forecasting", t) for t in (TS - 1, TS, TS + 1)] == [300, 300, 300]


def test_margin_floor_v4_is_generalist_only(gated):
    # the specialist tracks keep their published 1.5 floor after the gate (lead review: specialist DF/RA are not part of 3a)
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    assert ev._margin_now("generalist", TS + 1) == TS + 1
    assert ev._margin_now("demand_forecasting", TS + 1) is None
    assert M.margin_floor(ev._margin_now("generalist", TS + 1)) == M.MARGIN_FLOOR_V4
    assert M.margin_floor(ev._margin_now("risk_assessment", TS + 1)) == M.MARGIN_FLOOR
    assert M.required_margin(0.3, ev._margin_now("risk_assessment", TS + 1)) == M.MARGIN_FLOOR


def test_dryrun_cli_per_category(tmp_path):
    rows = pool(per_cat=200)
    for row in rows:
        row["state"] = {"secret": "PRIVATE_ITEM_TEXT"}
    path = tmp_path / "pool.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    tool = Path(__file__).resolve().parents[1] / "tools" / "generalist_draw_dryrun.py"

    def run(*extra):
        res = subprocess.run([sys.executable, str(tool), str(path), "--pairs", "1", *extra], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "PRIVATE_ITEM_TEXT" not in res.stdout + res.stderr
        return json.loads(res.stdout)

    base, v4 = run(), run("--per-category", "90")
    assert base == run("--per-category", "30")   # default = the live constant
    assert base["pairs_possible"] == v4["pairs_possible"] == 1 and v4["draws_possible"] == 2
    thin = tmp_path / "thin.jsonl"
    thin.write_text("\n".join(map(json.dumps, pool(per_cat=100))))
    res = subprocess.run([sys.executable, str(tool), str(thin), "--pairs", "1", "--per-category", "90"], capture_output=True, text=True)
    assert res.returncode == 0 and json.loads(res.stdout)["pairs_possible"] == 0   # a pool too thin for 90: reported, not a crash
    bad = subprocess.run([sys.executable, str(tool), str(path), "--per-category", "0"], capture_output=True, text=True)
    assert bad.returncode == 2
