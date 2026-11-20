"""The joint share (information on the UID page since 1.2.0, no verdict) is a joint fit, not a sum of betas over overlapping candidates."""
import numpy as np
import pytest

from synth import REV, World, combine, delta, finetune, make_base
from test_pipeline import codes, rec_for, run


def unit_delta(start, seed, norm_like, scale=0.15):
    return delta(finetune(start, seed, scale), start)


@pytest.fixture(scope="module")
def jworld(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("jw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    dA, dB, dC = (delta(finetune(start, s, 0.15), start) for s in (41, 42, 43))
    A, B, C = (combine(start, (1.0, d)) for d in (dA, dB, dC))
    for i, (n, t) in enumerate((("alice/ftA", A), ("bob/ftB", B), ("cora/ftC", C))):
        w.add(n, t, start="org/start@" + REV(2), rev=REV(3 + i), public_at="2026-01-0%dT00:00:00Z" % (i + 1))
    # composite published by a third holder: exactly 0.5 A + 0.5 B (a merge of the two)
    X = combine(start, (0.5, dA), (0.5, dB))
    w.add("dina/mergeAB", X, start="org/start@" + REV(2), rev=REV(6), public_at="2026-01-05T00:00:00Z")
    # same-data siblings: every one = start + shared direction u (a) + its own noise (cos between two of them ~ 0.18)
    u = delta(finetune(start, 77, 0.15), start)
    n0 = [delta(finetune(start, 80 + i, 0.15), start) for i in range(8)]
    sib = [combine(start, (0.47, u), (1.0, n)) for n in n0]
    for i, t in enumerate(sib):
        w.add("sib%d/data" % i, t, start="org/start@" + REV(2), rev=REV(10 + i), public_at="2026-02-0%dT00:00:00Z" % (i + 1))
    w.tensors = dict(root=root, start=start, dA=dA, dB=dB, dC=dC, u=u, A=A, B=B, X=X)
    w.u = u
    w.list_path = w.finish()
    return w


def test_composite_and_its_parent_do_not_double_count(jworld):
    st = jworld.tensors
    c = combine(finetune(st["start"], 91, 0.1), (0.3, st["dA"]))          # 30 % of A; the merge A+B is in the comparison set too
    res = run(jworld, "eve/part", c, rec_for(jworld, "eve/part"))
    det = {d["model"].split("@")[0]: d for d in res["sources_detected"]}
    assert "alice/ftA" in det and "dina/mergeAB" in det, det.keys()          # both are detected, both are weight overlaps ...
    assert {"alice/ftA", "dina/mergeAB"} <= {d["model"].split("@")[0] for d in res["weight_overlaps"]}
    js = res["joint_share"]
    assert js["sum_beta_overlaps"] >= 0.5, js                                # ... the sum of betas would have crossed 0.5
    assert js["share"] == pytest.approx(0.3, abs=0.06), js                   # the joint fit reads the true 0.3
    assert res["result"] == "PASS" and [codes(res, k) for k in "NRH"] == [[], [], []]


def test_independent_sources_add_up_in_the_information_but_not_in_the_verdict(jworld):
    st = jworld.tensors
    c = combine(finetune(st["start"], 92, 0.1), (0.35, st["dA"]), (0.35, st["dC"]))
    res = run(jworld, "eve/two", c, rec_for(jworld, "eve/two"))
    assert res["result"] == "PASS", (res["reasons"], res.get("joint_share"))      # MISDECLARATION_CANDIDATE before 1.2.0
    js = res["joint_share"]
    assert js["share"] == pytest.approx(0.7, abs=0.06) and js["lower_bound"] >= 0.5
    assert {"alice/ftA", "cora/ftC"} <= {m.split("@")[0] for m in js["by_model"]}


def test_same_data_siblings_are_weight_overlaps(jworld):
    st = jworld.tensors
    n = delta(finetune(st["start"], 99, 0.15), st["start"])
    c = combine(st["start"], (0.47, st["u"]), (1.0, n))                      # an honest entry on the same data as the siblings
    res = run(jworld, "fay/sib", c, rec_for(jworld, "fay/sib"))
    sib = [d for d in res["weight_overlaps"] if d["model"].startswith("sib")]
    assert len(sib) >= 6 and all(d["beta"] < 0.25 for d in sib), res["weight_overlaps"]
    assert res["result"] == "PASS", (res["reasons"], res.get("joint_share"))
    assert sum(b["role"] == "weight_overlap" for b in res["bar_list"]) == 3                  # all >= 6 stay on the UID page, the bar list takes the 3 largest (v1.3.0)
    assert [d["on_bar_list"] for d in res["weight_overlaps"]] == [True] * 3 + [False] * (len(res["weight_overlaps"]) - 3)


def test_many_overlaps_that_add_up_are_shown_and_never_a_finding(jworld):
    st = jworld.tensors
    n = delta(finetune(st["start"], 98, 0.15), st["start"])
    c = combine(st["start"], (0.47, st["u"]), (1.0, n))
    res = run(jworld, "fay/sib2", c, rec_for(jworld, "fay/sib2"))
    assert res["result"] == "PASS" and [codes(res, k) for k in "NRH"] == [[], [], []]
    assert "weak_overlaps_add_up" not in codes(res) and "weak_overlap_joint_share" not in res
    assert 0.5 <= res["joint_share"]["lower_bound"] <= 1.1                   # saturates towards 1 with many same-data siblings (CALIBRATION 6, limit 4)


def test_nnls_recovers_exact_mixture():
    from reax_lineage.measure import joint_fit, joint_summary
    rng = np.random.default_rng(3)
    keys = ["model.layers.%d.mlp.up_proj.weight" % l for l in range(6)]
    P = {n: {k: rng.standard_normal((4, 64)).astype(np.float32) for k in keys} for n in ("a", "b", "c")}
    P["ab"] = {k: 0.5 * P["a"][k] + 0.5 * P["b"][k] for k in keys}                    # composite
    oc = {k: 0.3 * P["a"][k] + 0.2 * P["c"][k] for k in keys}
    fit = joint_fit(oc, P, keys)
    s = joint_summary(fit, ["a", "ab", "b", "c"])
    assert s["share"] == pytest.approx(0.5, abs=1e-3)
    assert all(v >= 0 for v in s["by_model"].values())
