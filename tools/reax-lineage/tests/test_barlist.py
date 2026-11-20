"""RULES 3.6 (final, checker 1.4.0): at most the 3 weight overlaps with the largest weight in the joint non-negative fit of the entry's own change join the
bar list (ties: the larger cos, then the earlier commitment); the joint fit is computed before the ranking; if it fails for an entry the order falls back to
beta (noted). Every detection stays listed on the UID page as information. Unit tests for the order, end-to-end tests through the whole check."""
import copy
import json

import numpy as np
import pytest

import reax_lineage.check as check_mod
from reax_lineage.check import Check
from reax_lineage.measure import Thresholds
from synth import HOTKEY, REV, SEED_B, SPEC, World, combine, delta, finetune, make_base, write_repo
from test_pipeline import codes, rec_for, run

START = "org/start@" + REV(2)


# ------------------------------------------------------------------ the order (unit) ----

class FakeDB:
    def __init__(self, models, anchor=None):
        self.models, self.index = models, ({"anchor": anchor} if anchor else {})

    def get(self, k):
        return self.models.get(k)


def ov(model, w, cos, beta=0.0, share=1.0):
    return {"model": model, "joint_weight": w, "cos": cos, "beta": beta, "share": share}


def order(overlaps, models, anchor=None, by="joint_weight"):
    return [d["model"] for d in Check.rank_overlaps(overlaps, FakeDB(models, anchor), by)]


def test_the_largest_joint_weight_comes_first_whatever_beta_cos_and_share():
    ms = {"a": {"block": 1}, "b": {"block": 2}}
    # a: the largest beta, cos and share but the smaller weight in the joint fit (a composite whose parts are in the fit too)
    assert order([ov("a", 0.10, 0.99, beta=0.9, share=1.0), ov("b", 0.80, 0.10, beta=0.2, share=0.5)], ms) == ["b", "a"]


def test_equal_weight_is_decided_by_the_larger_cos():
    ms = {"a": {"block": 1}, "b": {"block": 2}, "c": {"block": 3}}
    assert order([ov("a", 0.5, 0.20), ov("b", 0.5, 0.31), ov("c", 0.5, 0.25)], ms) == ["b", "c", "a"]


def test_equal_weight_and_cos_is_decided_by_the_earlier_commitment_block_then_extrinsic():
    ms = {"late": {"block": 510, "extrinsic": 0}, "early": {"block": 500, "extrinsic": 3}, "mid": {"block": 500, "extrinsic": 7}, "none": {"block": 500}}
    got = order([ov(k, 0.5, 0.3) for k in ("late", "mid", "early", "none")], ms)
    assert got == ["none", "early", "mid", "late"], got                    # extrinsic missing counts as 0


def test_public_models_count_from_their_hugging_face_time():
    ms = {"p_new": {"public_at": "2026-03-01T00:00:00Z"}, "p_old": {"public_at": "2025-12-01T12:00:00Z"}}
    assert order([ov("p_new", 0.5, 0.3), ov("p_old", 0.5, 0.3)], ms) == ["p_old", "p_new"]


def test_entry_against_public_model_without_a_chain_anchor_the_public_model_is_earlier():
    ms = {"entry": {"block": 1}, "pub": {"public_at": "2026-11-01T00:00:00Z"}}
    assert order([ov("entry", 0.5, 0.3), ov("pub", 0.5, 0.3)], ms) == ["pub", "entry"]


def test_entry_against_public_model_with_a_chain_anchor_compares_by_time():
    anchor = {"block": 1000, "time": "2026-06-01T00:00:00Z"}                     # 12 s per block
    ms = {"entry": {"block": 1100}, "pub_before": {"public_at": "2026-06-01T00:10:00Z"}, "pub_after": {"public_at": "2026-06-01T00:30:00Z"}}
    # block 1100 = anchor + 1200 s = 00:20:00
    got = order([ov(k, 0.5, 0.3) for k in ("pub_after", "entry", "pub_before")], ms, anchor)
    assert got == ["pub_before", "entry", "pub_after"], got


def test_the_order_is_total_even_for_identical_values():
    ms = {}
    a = order([ov("m/z@1", 0.5, 0.3), ov("m/a@1", 0.5, 0.3), ov("m/k@1", 0.5, 0.3)], ms)
    b = order([ov("m/k@1", 0.5, 0.3), ov("m/z@1", 0.5, 0.3), ov("m/a@1", 0.5, 0.3)], ms)
    assert a == b == ["m/a@1", "m/k@1", "m/z@1"]


def test_an_overlap_that_is_not_in_the_fit_counts_as_zero():
    ms = {"fitted0": {"block": 1}, "unfitted": {"block": 2}, "fitted": {"block": 3}}
    # weight None (not in the fit) is 0.0: it ties with a fitted weight of 0.0 (then the cos decides) and is below every positive weight
    got = order([ov("unfitted", None, 0.60), ov("fitted0", 0.0, 0.40), ov("fitted", 0.05, 0.01)], ms)
    assert got == ["fitted", "unfitted", "fitted0"], got


def test_fallback_orders_by_beta_and_ignores_the_joint_weight():
    ms = {"a": {"block": 1}, "b": {"block": 2}, "c": {"block": 3}}
    got = order([ov("a", None, 0.9, beta=0.30), ov("b", None, 0.1, beta=0.80), ov("c", None, 0.5, beta=0.30)], ms, by="beta")
    assert got == ["b", "a", "c"], got                                     # beta, then the larger cos (a before c)
    got = order([ov("a", 0.9, 0.9, beta=0.10), ov("b", 0.1, 0.1, beta=0.80)], ms, by="beta")
    assert got == ["b", "a"], got                                          # a weight is not used in the fallback


def test_cap_is_a_published_threshold():
    t = Thresholds.load()
    assert t.bar_list_max_overlaps == 3
    assert json.load(open(__import__("reax_lineage.measure", fromlist=["x"]).THRESHOLDS_FILE))["bar_list_max_overlaps"] == 3


# ------------------------------------------------------------------ whole check ----

def part_independent(t, base, seed, layer):
    """`t` with the tensors of one layer replaced by an independent change of `base` (the model shares the other layers with the entry)."""
    other = finetune(base, seed, 0.15)
    out = dict(t)
    for k in t:
        if k.startswith("model.layers.%d." % layer) and t[k].ndim == 2:
            out[k] = other[k]
    return out


@pytest.fixture(scope="module")
def bw(tmp_path_factory):
    """Five earlier models of the base family (two entries, three public) with the same start. P1's change matches the entry in 7 of the 8 layers
    only (layer share 0.875) but with the largest coefficient; E2, E3, P4, P5 match in all layers with falling coefficients."""
    w = World(tmp_path_factory.mktemp("bw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    d = {i: delta(finetune(start, 40 + i, 0.15), start) for i in range(1, 6)}
    P1 = part_independent(combine(start, (1.0, d[1])), start, 90, 7)
    w.add("p1h/model", P1, start=START, rev=REV(11), public_at="2026-01-01T00:00:00Z")
    w.add("e2h/model", combine(start, (1.0, d[2])), kind="entry", start=START, block=500, uid=22, result="PASS", rev=REV(12))
    w.add("e3h/model", combine(start, (1.0, d[3])), kind="entry", start=START, block=510, uid=23, result="PASS", rev=REV(13))
    w.add("p4h/model", combine(start, (1.0, d[4])), start=START, rev=REV(14), public_at="2026-01-02T00:00:00Z")
    w.add("p5h/model", combine(start, (1.0, d[5])), start=START, rev=REV(15), public_at="2026-01-03T00:00:00Z")
    w.tensors = dict(root=root, start=start, **{"d%d" % i: d[i] for i in d})
    w.list_path = w.finish()
    return w


def entry_mix(w, seed, coefs):
    return combine(finetune(w.tensors["start"], seed, 0.1), *[(c, w.tensors["d%d" % i]) for i, c in coefs.items()])


COEFS = {1: 1.0, 2: 0.9, 3: 0.6, 4: 0.45, 5: 0.3}


def short(res, key="weight_overlaps"):
    return [d["model"].split("@")[0] for d in res[key]]


def test_five_overlaps_the_bar_list_takes_the_three_with_the_largest_joint_weight(bw):
    res = run(bw, "eve/five", entry_mix(bw, 70, COEFS), rec_for(bw, "eve/five"))
    assert res["result"] == "PASS", res["reasons"]
    assert [codes(res, k) for k in "NRH"] == [[], [], []]
    ov_ = res["weight_overlaps"]
    # all five are detected and listed, in the order of their weight in the joint fit; the weights read the coefficients of the mix.
    # P1 matches only 7 of 8 layers (layer share 0.875, which put it LAST under the share order of v1.3.0) but has the largest weight (median over layers) -> first
    assert short(res) == ["p1h/model", "e2h/model", "e3h/model", "p4h/model", "p5h/model"], [(d["model"], d["joint_weight"], d["cos"], d["beta"]) for d in ov_]
    assert [d["joint_weight"] for d in ov_] == pytest.approx([1.0, 0.9, 0.6, 0.45, 0.3], abs=0.07)
    assert ov_[0]["share"] == pytest.approx(0.875, abs=0.001) and all(d["share"] == 1.0 for d in ov_[1:])
    assert [d["bar_list_rank"] for d in ov_] == [1, 2, 3, 4, 5]
    assert [d["on_bar_list"] for d in ov_] == [True, True, True, False, False] and all(d["in_joint_fit"] for d in ov_)
    # the bar list: start chain + exactly the three
    assert [(b["model"].split("@")[0], b["role"]) for b in res["bar_list"]] == [
        ("org/start", "start"), ("org/root", "ancestor"), ("p1h/model", "weight_overlap"), ("e2h/model", "weight_overlap"), ("e3h/model", "weight_overlap")]
    assert [b["rank"] for b in res["bar_list"] if b["role"] == "weight_overlap"] == [1, 2, 3]
    assert all(b["joint_weight"] is not None for b in res["bar_list"] if b["role"] == "weight_overlap")
    # every detection stays on the UID page as information, also those without a place on the bar list
    assert {d["model"].split("@")[0] for d in res["sources_detected"]} >= {"p1h/model", "e2h/model", "e3h/model", "p4h/model", "p5h/model"}
    flags = {d["model"].split("@")[0]: (d.get("weight_overlap"), d.get("on_bar_list")) for d in res["sources_detected"]}
    assert flags["p5h/model"] == (True, False) and flags["p4h/model"] == (True, False) and flags["p1h/model"] == (True, True)
    json.dumps(res, default=str)


def test_the_order_can_be_redone_from_the_uid_page(bw):
    """The values that decide are the published ones: weight (joint_share.by_model = the overlap's joint_weight), then cos, on the page itself."""
    res = run(bw, "eve/page", entry_mix(bw, 76, COEFS), rec_for(bw, "eve/page"))
    js = res["joint_share"]
    assert js["order_sample"] in ("A", "B") and set(js["by_model"]) >= {d["model"] for d in res["weight_overlaps"]}
    assert all(d["joint_weight"] == js["by_model"][d["model"]] for d in res["weight_overlaps"])
    redone = sorted(res["weight_overlaps"], key=lambda d: (-d["joint_weight"], -d["cos"]))
    assert [d["model"] for d in redone] == [d["model"] for d in res["weight_overlaps"]]
    assert res["bar_list_order"] == {"key": "joint_weight", "sample": js["order_sample"], "fallback": False, "why": None, "ties": "larger cos, then earlier commitment"}
    assert "joint_fit_failed" not in codes(res, "I") and not any("joint fit" in n for n in res["notes"])


def test_overlaps_beyond_the_cap_get_an_information_reason_and_no_consequence(bw):
    res = run(bw, "eve/five2", entry_mix(bw, 71, COEFS), rec_for(bw, "eve/five2"))
    listed = [r for r in res["reasons"] if r["code"] == "weight_overlap"]
    unlisted = [r for r in res["reasons"] if r["code"] == "weight_overlap_not_listed"]
    assert len(listed) == 3 and len(unlisted) == 2 and all(r["class"] == "I" for r in listed + unlisted)
    assert "It joins your bar list" in listed[0]["message"] and "number 1" in listed[0]["message"] and "p1h/model" in listed[0]["message"]
    assert "largest weight in the joint fit" in listed[0]["message"] and ", weight in the joint fit " in listed[0]["message"]
    assert all("Shown for information only" in r["message"] and "No correction, no penalty" in r["message"] and "joins" not in r["message"] for r in unlisted)
    assert "number 4 of 5" in unlisted[0]["message"] and "number 5 of 5" in unlisted[1]["message"]
    assert res["correction_deadline"] is None and res["result"] == "PASS"
    assert res["joint_share"] is not None and len(res["joint_share"]["sources"]) == 5          # the joint share (information) still covers all five


def test_three_or_fewer_overlaps_are_all_on_the_bar_list(bw):
    res = run(bw, "eve/three", entry_mix(bw, 72, {2: 0.9, 3: 0.6, 4: 0.45}), rec_for(bw, "eve/three"))
    assert res["result"] == "PASS" and short(res) == ["e2h/model", "e3h/model", "p4h/model"]
    assert all(d["on_bar_list"] for d in res["weight_overlaps"]) and "weight_overlap_not_listed" not in codes(res, "I")
    assert sum(b["role"] == "weight_overlap" for b in res["bar_list"]) == 3


def test_the_cap_comes_from_the_threshold(bw):
    from reax_lineage.sources import Hub
    thr = Thresholds.load()
    thr.bar_list_max_overlaps = 1
    repo = "eve/capone"
    rev = REV(5000)
    rec = copy.deepcopy(rec_for(bw, repo))
    rec["entry"] = {"repo": repo, "revision": rev}
    write_repo(bw.hub, repo, rev, entry_mix(bw, 73, COEFS), record=rec)
    res = Check(repo, rev, hotkey=HOTKEY, block=1000, commit_time="2026-06-01T00:00:00Z", list_path=bw.list_path, db_path=bw.dbdir, hub=Hub(hub_dir=bw.hub),
                spec=SPEC, cache_dir=None, dry_run=False, seed_b=SEED_B, jobs=1, tier_b=False, thr=thr).run()
    assert [b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "weight_overlap"] == ["p1h/model"] and len(res["weight_overlaps"]) == 5


def test_a_teacher_that_is_also_an_overlap_uses_one_of_the_three_places(bw):
    """Reading of 3.6 (documented in CALIBRATION.md 16.1): the three places go to the three largest overlaps, whether or not one of them is on the list
    anyway (here as a teacher). The teacher keeps its role and gets the measurement and rank."""
    rec = rec_for(bw, "eve/teach")
    rec["teachers"] = [{"name": "E3", "repo_or_api": "e3h/model", "use": "distillation"}]
    res = run(bw, "eve/teach", entry_mix(bw, 74, COEFS), rec)
    assert res["result"] == "PASS", res["reasons"]
    roles = {b["model"].split("@")[0]: b["role"] for b in res["bar_list"]}
    assert roles == {"org/start": "start", "org/root": "ancestor", "e3h/model": "teacher", "p1h/model": "weight_overlap", "e2h/model": "weight_overlap"}, roles
    t = [b for b in res["bar_list"] if b["model"].startswith("e3h/model@")][0]
    assert t["weight_overlap"]["rank"] == 3 and t["weight_overlap"]["joint_weight"] == pytest.approx(0.6, abs=0.07) and "p4h/model" not in roles


def test_own_holder_entries_do_not_use_up_a_place(tmp_path_factory):
    """An own-holder model measured in the weights is not an overlap at all (3.6), so the three places go to the three largest others."""
    w = World(tmp_path_factory.mktemp("bwo"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    d = {i: delta(finetune(start, 40 + i, 0.15), start) for i in range(1, 6)}
    w.add("sam/earlier", combine(start, (1.0, d[1])), kind="entry", start=START, block=500, uid=31, result="PASS", rev=REV(11), holder="hf:sam")
    for i in (2, 3, 4, 5):
        w.add("o%dh/model" % i, combine(start, (1.0, d[i])), start=START, rev=REV(11 + i), public_at="2026-01-0%dT00:00:00Z" % i)
    w.tensors = dict(start=start, **{"d%d" % i: d[i] for i in d})
    w.list_path = w.finish()
    res = run(w, "sam/later", entry_mix(w, 75, {1: 1.0, 2: 0.9, 3: 0.6, 4: 0.45, 5: 0.3}), rec_for(w, "sam/later"), holder_ids=["hf:sam"])      # holder ids come from the caller only (v1.5.0)
    assert res["result"] == "PASS", res["reasons"]
    assert short(res) == ["o2h/model", "o3h/model", "o4h/model", "o5h/model"] and [d["on_bar_list"] for d in res["weight_overlaps"]] == [True, True, True, False]
    assert any(d["model"].startswith("sam/earlier@") and d["own_holder"] for d in res["sources_detected"])


# ------------------------------------------------------------------ a composite: the case the share, beta and cos orders get wrong ----

@pytest.fixture(scope="module")
def cw(tmp_path_factory):
    """E1 = start + d1, E2 = start + d2 (earlier entries), P4 = start + d4, P5 = start + d5, and X = start + d1 + d2 + d3 (a merge of E1, E2 and an unrelated
    d3 by a third holder). An entry that took 1.0 d1 + 0.9 d2 + 0.45 d4 + 0.3 d5 has its largest cos and (after d1, d2) beta with the composite X, while the joint
    fit puts the weight on the parts and none on X."""
    w = World(tmp_path_factory.mktemp("cw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    d = {i: delta(finetune(start, 40 + i, 0.15), start) for i in range(1, 6)}
    w.add("e1h/model", combine(start, (1.0, d[1])), kind="entry", start=START, block=500, uid=21, result="PASS", rev=REV(11))
    w.add("e2h/model", combine(start, (1.0, d[2])), kind="entry", start=START, block=510, uid=22, result="PASS", rev=REV(12))
    w.add("p4h/model", combine(start, (1.0, d[4])), start=START, rev=REV(14), public_at="2026-01-02T00:00:00Z")
    w.add("p5h/model", combine(start, (1.0, d[5])), start=START, rev=REV(15), public_at="2026-01-03T00:00:00Z")
    w.add("xh/composite", combine(start, (1.0, d[1]), (1.0, d[2]), (1.0, d[3])), start=START, rev=REV(16), public_at="2026-01-04T00:00:00Z")
    w.tensors = dict(root=root, start=start, **{"d%d" % i: d[i] for i in d})
    w.list_path = w.finish()
    return w


CW_COEFS = {1: 1.0, 2: 0.9, 4: 0.45, 5: 0.3}


def test_a_composite_with_the_largest_cos_and_beta_does_not_take_a_place_from_its_parts(cw):
    res = run(cw, "eve/comp", entry_mix(cw, 80, CW_COEFS), rec_for(cw, "eve/comp"))
    assert res["result"] == "PASS", res["reasons"]
    by = {d["model"].split("@")[0]: d for d in res["weight_overlaps"]}
    assert set(by) == {"e1h/model", "e2h/model", "p4h/model", "p5h/model", "xh/composite"}, list(by)
    x, p4 = by["xh/composite"], by["p4h/model"]
    # the case this test is about: the composite beats P4 on cos AND beta (so a cos or a beta order would give it P4's place) ...
    assert x["cos"] > p4["cos"] and x["beta"] > p4["beta"] and x["cos"] == max(d["cos"] for d in by.values())
    # ... but the joint fit gives it (almost) no weight, the parts keep theirs
    assert x["joint_weight"] < 0.1 and by["e1h/model"]["joint_weight"] == pytest.approx(1.0, abs=0.1) and by["e2h/model"]["joint_weight"] == pytest.approx(0.9, abs=0.1)
    assert short(res) == ["e1h/model", "e2h/model", "p4h/model", "p5h/model", "xh/composite"]
    assert [b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "weight_overlap"] == ["e1h/model", "e2h/model", "p4h/model"]
    assert by["xh/composite"]["on_bar_list"] is False and by["xh/composite"]["bar_list_rank"] == 5


# ------------------------------------------------------------------ the fallback to beta ----

def beta_order(res):
    return [d["model"].split("@")[0] for d in sorted(res["weight_overlaps"], key=lambda d: (-d["beta"], -d["cos"]))]


def check_fallback(res):
    assert res["result"] == "PASS", res["reasons"]
    assert res["joint_share"] is None
    assert res["bar_list_order"]["key"] == "beta" and res["bar_list_order"]["fallback"] is True and res["bar_list_order"]["sample"] is None and res["bar_list_order"]["why"]
    assert any(n.startswith("joint fit of the entry's own change failed") and "ordered by beta" in n for n in res["notes"]), res["notes"]
    jf = [r for r in res["reasons"] if r["code"] == "joint_fit_failed"]
    assert len(jf) == 1 and jf[0]["class"] == "I" and "ordered by beta" in jf[0]["message"] and "No correction, no penalty" in jf[0]["message"]
    assert [d["model"].split("@")[0] for d in res["weight_overlaps"]] == beta_order(res)
    assert all(d["joint_weight"] is None and d["in_joint_fit"] is False for d in res["weight_overlaps"])
    assert [d["bar_list_rank"] for d in res["weight_overlaps"]] == list(range(1, len(res["weight_overlaps"]) + 1))
    assert sum(d["on_bar_list"] for d in res["weight_overlaps"]) == 3 and sum(b["role"] == "weight_overlap" for b in res["bar_list"]) == 3
    assert any("largest beta" in r["message"] and "joint fit of your change failed" in r["message"] for r in res["reasons"] if r["code"] == "weight_overlap")
    assert res["correction_deadline"] is None
    json.dumps(res, default=str)


def test_if_the_joint_fit_raises_the_order_falls_back_to_beta_and_says_so(cw, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure of the fit")
    monkeypatch.setattr(check_mod, "joint_fit", boom)
    res = run(cw, "eve/fb1", entry_mix(cw, 81, CW_COEFS), rec_for(cw, "eve/fb1"))
    check_fallback(res)
    assert "RuntimeError: synthetic failure of the fit" in res["bar_list_order"]["why"]
    # the beta order is not the weight order: the composite (beta 0.6) is third and takes the place of P4
    assert short(res)[:3] == ["e1h/model", "e2h/model", "xh/composite"]
    assert [b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "weight_overlap"] == ["e1h/model", "e2h/model", "xh/composite"]


def test_a_non_finite_weight_counts_as_a_failure_of_the_fit(cw, monkeypatch):
    real = check_mod.joint_summary

    def nan_summary(fit, subset):
        s = real(fit, subset)
        k = next(iter(s["by_model"]))
        s["by_model"][k] = float("nan")
        return s
    monkeypatch.setattr(check_mod, "joint_summary", nan_summary)
    res = run(cw, "eve/fb2", entry_mix(cw, 82, CW_COEFS), rec_for(cw, "eve/fb2"))
    check_fallback(res)
    assert "non-finite" in res["bar_list_order"]["why"]


def test_a_fit_without_layers_counts_as_a_failure_of_the_fit(cw, monkeypatch):
    monkeypatch.setattr(check_mod, "joint_summary", lambda fit, subset: {"share": 0.0, "se": 0.0, "lower_bound": 0.0, "by_model": {}})
    res = run(cw, "eve/fb3", entry_mix(cw, 83, CW_COEFS), rec_for(cw, "eve/fb3"))
    check_fallback(res)


def test_a_failure_on_one_sample_does_not_cost_the_joint_order(cw, monkeypatch):
    """Sample A's fit raises, sample B's works: the order uses B (published as `order_sample`), the failure of A is noted, no fallback."""
    real, calls = check_mod.joint_fit, []

    def first_fails(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("sample A fails")
        return real(*a, **k)
    monkeypatch.setattr(check_mod, "joint_fit", first_fails)
    res = run(cw, "eve/fb4", entry_mix(cw, 84, CW_COEFS), rec_for(cw, "eve/fb4"))
    assert res["result"] == "PASS" and res["bar_list_order"]["key"] == "joint_weight" and res["bar_list_order"]["sample"] == "B" and res["joint_share"]["order_sample"] == "B"
    assert any(n.startswith("joint fit sample A: RuntimeError: sample A fails") for n in res["notes"]), res["notes"]
    assert "joint_fit_failed" not in codes(res, "I") and set(res["joint_share"]["samples"]) == {"B"}
    assert [b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "weight_overlap"] == ["e1h/model", "e2h/model", "p4h/model"]


def test_an_overlap_outside_the_fit_counts_as_zero_and_stays_listed(cw, monkeypatch):
    """Only the 4 strongest overlaps by beta are fitted (the limit is 16 in production, `Check.JOINT_MAX`); P5 is the fifth: no weight (null), it counts as 0.0
    in the order (a tie with the composite's fitted 0.0, which the cos decides), stays on the UID page and says so in its reason."""
    monkeypatch.setattr(Check, "JOINT_MAX", 4)
    res = run(cw, "eve/fb5", entry_mix(cw, 85, CW_COEFS), rec_for(cw, "eve/fb5"))
    assert res["result"] == "PASS" and res["bar_list_order"]["key"] == "joint_weight"
    by = {d["model"].split("@")[0]: d for d in res["weight_overlaps"]}
    assert [d["model"] for d in res["weight_overlaps"] if not d["in_joint_fit"]] == [by["p5h/model"]["model"]] and by["p5h/model"]["joint_weight"] is None
    assert short(res) == ["e1h/model", "e2h/model", "p4h/model", "xh/composite", "p5h/model"], [(d["model"], d["joint_weight"], d["cos"]) for d in res["weight_overlaps"]]
    assert 0.0 <= by["xh/composite"]["joint_weight"] < 0.1 and by["xh/composite"]["cos"] > by["p5h/model"]["cos"]
    assert [d["on_bar_list"] for d in res["weight_overlaps"]] == [True, True, True, False, False] and len(res["joint_share"]["sources"]) == 4
    assert any("not in the joint fit" in r["message"] and "p5h/model" in r["message"] for r in res["reasons"] if r["code"] == "weight_overlap_not_listed")
    assert "joint_fit_failed" not in codes(res, "I")


@pytest.mark.parametrize("high", ["A", "B"])
def test_the_sample_with_the_highest_lower_bound_orders_the_list_and_is_published(cw, monkeypatch, high):
    """Both samples are fitted; the one with the highest lower bound is the one published (`joint_share`) and used for the order (`bar_list_order.sample`)."""
    real, seen = check_mod.joint_summary, {}

    def rigged(fit, subset):
        s = real(fit, subset)
        sid = "AB"[len(seen)]                                           # the samples are fitted in the order A, B
        seen[sid] = dict(s["by_model"])
        s["lower_bound"] = 5.0 if sid == high else 0.1
        return s
    monkeypatch.setattr(check_mod, "joint_summary", rigged)
    res = run(cw, "eve/smp" + high, entry_mix(cw, 86, CW_COEFS), rec_for(cw, "eve/smp" + high))
    assert set(seen) == {"A", "B"} and seen["A"] != seen["B"]            # the two samples read slightly different weights, so the test can tell them apart
    assert res["bar_list_order"]["sample"] == high and res["joint_share"]["order_sample"] == high and res["joint_share"]["lower_bound"] == 5.0
    assert {d["model"]: d["joint_weight"] for d in res["weight_overlaps"]} == seen[high] == res["joint_share"]["by_model"]
