"""RULES-v2 3.3 / 3.6 / 3.8 (final text of 13 Nov, checker 1.2.0): a detected source of the same base family (an earlier entry of ANY start, or a public model)
that is not a copy (3.5), not of the own holder and not in the start chain is a weight overlap: bar list, PASS, no correction, no refusal, no
misdeclaration (C4: honest same-start, same-data siblings are inseparable from merges). The joint share stays as information. Copies, own-holder
models and the start chain keep their handling."""
import json

import numpy as np
import pytest

from reax_lineage.check import Check
from synth import REV, World, combine, delta, finetune, make_base
from test_pipeline import codes, rec_for, run

START, START2 = "org/start@" + REV(2), "org/start2@" + REV(6)


@pytest.fixture(scope="module")
def ow(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("ow"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    start2 = finetune(root, 32, 1.0)
    w.add("org/start2", start2, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(6), public_at="2025-07-02T00:00:00Z")
    d1, d2, dA, dX, dS = (delta(finetune(start, s, 0.15), start) for s in (41, 42, 43, 44, 45))
    dX2 = delta(finetune(start2, 46, 0.15), start2)                       # own change of an entry that started from start2
    e1, e2 = combine(start, (1.0, d1)), combine(start, (1.0, d2))
    A = combine(start, (1.0, dA))
    x1 = combine(start2, (1.0, dX2))
    w.add("e1h/model", e1, kind="entry", start=START, block=500, uid=11, result="PASS", rev=REV(20))
    w.add("e2h/model", e2, kind="entry", start=START, block=510, uid=12, result="PASS", rev=REV(21))
    w.add("x1h/model", x1, kind="entry", start=START2, block=520, uid=13, result="PASS", rev=REV(22))
    w.add("alice/ftA", A, start=START, rev=REV(3), public_at="2026-01-01T00:00:00Z")
    dF = delta(finetune(start, 48, 0.15), start)
    w.add("flag/pubF", combine(start, (1.0, dF)), start=START, rev=REV(7), public_at="2026-01-02T00:00:00Z", flagged=True)      # a flagged public model
    own = combine(start, (1.0, dS))
    w.add("sam/earlier", own, kind="entry", start=START, block=530, uid=14, result="PASS", rev=REV(23), holder="hf:sam")
    dSO = delta(finetune(start2, 47, 0.15), start2)
    own2 = combine(start2, (1.0, dSO))                                  # an own entry of the same holder that started from ANOTHER list start
    w.add("sam/other", own2, kind="entry", start=START2, block=540, uid=15, result="PASS", rev=REV(24), holder="hf:sam")
    w.tensors = dict(root=root, start=start, start2=start2, d1=d1, d2=d2, dA=dA, dX2=dX2, dS=dS, dSO=dSO, dF=dF, e1=e1, e2=e2, A=A, x1=x1, own=own, own2=own2)
    w.list_path = w.finish()
    return w


def mix(w, seed, *terms):
    return combine(finetune(w.tensors["start"], seed, 0.1), *terms)


def keyof(res, repo):
    return next(d["model"] for d in res["sources_detected"] if d["model"].startswith(repo + "@"))


def test_same_start_entry_overlap_goes_on_the_bar_list_without_a_correction(ow):
    c = mix(ow, 91, (0.6, ow.tensors["d1"]))
    res = run(ow, "eve/sib", c, rec_for(ow, "eve/sib"))
    assert res["result"] == "PASS", (res["reasons"], res.get("hidden_share"))
    assert [codes(res, k) for k in "NRH"] == [[], [], []]
    (ov,) = res["weight_overlaps"]
    assert ov["model"].startswith("e1h/model@") and ov["same_start_entry"] and ov["uid"] == 11
    assert ov["beta"] == pytest.approx(0.6, abs=0.04) and ov["cos"] > 0.3
    item = [b for b in res["bar_list"] if b["model"] == ov["model"]]
    assert len(item) == 1 and item[0]["role"] == "weight_overlap" and item[0]["reason"] == "weight_overlap"
    assert item[0]["beta"] == ov["beta"] and item[0]["cos"] == ov["cos"] and item[0]["uid"] == 11
    assert [b["model"].split("@")[0] for b in res["bar_list"]][:2] == ["org/start", "org/root"]
    assert "weight_overlap" in codes(res, "I") and "hidden_share" not in res and "weak_overlaps" not in res
    assert ov["relation"] == "same_start_entry" and item[0]["relation"] == "same_start_entry" and item[0]["kind"] == "entry"
    msg = [r["message"] for r in res["reasons"] if r["code"] == "weight_overlap"][0]
    assert "(uid 11)" in msg and "an earlier entry of your base family with your start. It joins your bar list" in msg and "no penalty" in msg
    js = res["joint_share"]                                         # information only
    assert js["share"] == pytest.approx(0.6, abs=0.07) and "no verdict" in js["rule"] and js["old_limit_for_reference"] == 0.5
    json.dumps(res, default=str)


def test_overlap_below_the_old_action_level_is_on_the_bar_list_too(ow):
    """A weak overlap (beta < 0.25) used to be only recorded; it is a bar-list overlap now."""
    c = mix(ow, 92, (0.12, ow.tensors["d1"]))
    res = run(ow, "eve/weak", c, rec_for(ow, "eve/weak"))
    assert res["result"] == "PASS" and [b["role"] for b in res["bar_list"]].count("weight_overlap") == 1
    assert res["weight_overlaps"][0]["beta"] == pytest.approx(0.12, abs=0.03)


def test_two_same_start_overlaps_never_add_up_to_a_misdeclaration(ow):
    st = ow.tensors
    c = mix(ow, 93, (0.5, st["d1"]), (0.5, st["d2"]))                          # 1.0 together: a MISDECLARATION_CANDIDATE before
    res = run(ow, "eve/two", c, rec_for(ow, "eve/two"))
    assert res["result"] == "PASS", (res["reasons"], res.get("hidden_share"))
    assert sorted(b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "weight_overlap") == ["e1h/model", "e2h/model"]
    assert "hidden_share" not in res and codes(res, "H") == []
    assert res["joint_share"]["share"] == pytest.approx(1.0, abs=0.1)         # shown to the reader, no verdict


def test_public_model_next_to_an_entry_overlap_is_an_overlap_too(ow):
    """Final rule: a public model of the base family in the weights (the one of another holder, 30 %) joins the bar list next to the entry."""
    st = ow.tensors
    c = mix(ow, 94, (0.7, st["d1"]), (0.3, st["dA"]))
    res = run(ow, "eve/mix", c, rec_for(ow, "eve/mix"))
    assert res["result"] == "PASS", res["reasons"]
    assert [codes(res, k) for k in "NRH"] == [[], [], []]
    rel = {d["model"].split("@")[0]: d["relation"] for d in res["weight_overlaps"]}
    assert rel == {"e1h/model": "same_start_entry", "alice/ftA": "public"}, rel
    assert sorted(b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "weight_overlap") == ["alice/ftA", "e1h/model"]
    assert res["joint_share"]["share"] == pytest.approx(1.0, abs=0.1) and res["joint_share"]["lower_bound"] >= 0.5     # information only


def test_a_statistical_share_alone_is_never_a_misdeclaration(ow):
    """1.2 in all, 0.6 of it from a public model of another holder, plus an entry: before 1.2.0 a MISDECLARATION_CANDIDATE (joint lower bound >= 0.5)."""
    st = ow.tensors
    c = mix(ow, 95, (0.6, st["d1"]), (0.6, st["dA"]))
    res = run(ow, "eve/hid", c, rec_for(ow, "eve/hid"))
    assert res["result"] == "PASS", (res["reasons"], res.get("joint_share"))
    assert [codes(res, k) for k in "NRH"] == [[], [], []] and "hidden_share" not in res
    js = res["joint_share"]
    assert js["share"] == pytest.approx(1.2, abs=0.12) and js["lower_bound"] >= 0.5          # the old limit is crossed; shown, no verdict
    assert any("alice/ftA" in m for m in js["by_model"]) and any("e1h" in m for m in js["by_model"])
    assert {d["model"].split("@")[0] for d in res["weight_overlaps"]} == {"e1h/model", "alice/ftA"}


def test_a_different_start_entry_is_a_weight_overlap_too(ow):
    st = ow.tensors
    for share in (0.35, 0.7):
        c = mix(ow, 96, (share, st["dX2"]))
        res = run(ow, "eve/diff%d" % int(share * 100), c, rec_for(ow, "eve/diff"))
        assert res["result"] == "PASS", (share, res["reasons"], res.get("joint_share"))
        assert [codes(res, k) for k in "NRH"] == [[], [], []]
        x = [d for d in res["sources_detected"] if d["model"].startswith("x1h/model@")][0]
        assert x["same_start_entry"] is False and x["relation"] == "other_start_entry" and x["weight_overlap"]
        (item,) = [b for b in res["bar_list"] if b["role"] == "weight_overlap"]
        assert item["model"] == x["model"] and item["uid"] == 13 and item["relation"] == "other_start_entry"


def test_a_copy_of_a_same_start_entry_is_still_a_copy(ow):
    st = ow.tensors
    res = run(ow, "kim/copyE", dict(st["e1"]), rec_for(ow, "kim/copyE"))
    assert res["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(res, "H"), res["reasons"]
    assert res["weight_overlaps"] == []
    rec = rec_for(ow, "kim/copyE2")
    rec["mentioned_not_used"] = [{"id": "e1h/model", "reason": "comparison only"}]
    res2 = run(ow, "kim/copyE2", dict(st["e1"]), rec, readme="Compared with e1h/model.\n")
    assert res2["result"] == "REFUSED" and "copy" in codes(res2, "R")
    assert res2["weight_overlaps"] == []


def test_own_holder_overlap_is_own_work_and_not_on_the_bar_list(ow):
    st = ow.tensors
    c = mix(ow, 97, (0.6, st["dS"]))
    res = run(ow, "sam/later", c, rec_for(ow, "sam/later"), holder_ids=["hf:sam"])       # holder ids come from the caller only (v1.5.0, RULES 2.9)
    assert res["result"] == "PASS"
    d = [d for d in res["sources_detected"] if d["model"].startswith("sam/earlier@")][0]
    assert d["own_holder"] and d["same_start_entry"] and res["weight_overlaps"] == []
    assert all(b["role"] != "weight_overlap" for b in res["bar_list"]) and "weight_overlap" not in codes(res, "I")
    # the same own-holder entry copied is a duplicate
    res2 = run(ow, "sam/dup", dict(st["own"]), rec_for(ow, "sam/dup"), holder_ids=["hf:sam"])
    assert res2["result"] == "REFUSED" and "duplicate" in codes(res2, "R")


def test_flagged_model_in_the_weights_is_an_overlap_with_a_note(ow):
    """Even a flagged model of the base family measured in the weights has no consequence from the weights alone (RULES 3.8 final): bar list + a note."""
    c = mix(ow, 101, (0.5, ow.tensors["dF"]))
    res = run(ow, "zoe/flagmix", c, rec_for(ow, "zoe/flagmix"))
    assert res["result"] == "PASS" and [codes(res, k) for k in "NRH"] == [[], [], []], res["reasons"]
    (d,) = [d for d in res["weight_overlaps"] if d["model"].startswith("flag/pubF@")]
    assert d["flagged"] and d["beta"] == pytest.approx(0.5, abs=0.05)
    assert any("flagged" in n and "flag/pubF" in n for n in res["notes"])
    assert any(b["role"] == "weight_overlap" and b["model"].startswith("flag/pubF@") for b in res["bar_list"])


def test_own_holder_entry_in_the_weights_is_left_out_but_its_ancestors_stay(ow):
    """3.6: entries of the own holder are left out of the bar list, their ancestors stay (here: the other list start of the own earlier entry)."""
    st = ow.tensors
    c = mix(ow, 100, (0.5, st["dSO"]))
    res = run(ow, "sam/later3", c, rec_for(ow, "sam/later3"), holder_ids=["hf:sam"])
    d = [d for d in res["sources_detected"] if d["model"].startswith("sam/other@")][0]
    assert d["own_holder"] and d["relation"] == "other_start_entry" and res["weight_overlaps"] == []
    assert res["result"] == "PASS" and "weight_overlap" not in codes(res, "I")
    roles = {b["model"].split("@")[0]: b["role"] for b in res["bar_list"]}
    assert roles == {"org/start": "start", "org/root": "ancestor", "org/start2": "ancestor of own entry"}, roles


def test_splice_only_hit_is_a_weight_overlap_too(ow):
    """Two layers copied from an earlier entry: not the whole-model pattern, but a detected source of the base family that is not a copy: an overlap
    (before 1.2.0 NEEDS_CORRECTION). The statistical pattern is the same as for near-copy siblings, so it carries no consequence either."""
    st = ow.tensors
    c = finetune(st["start"], 98, 0.1)
    for k in list(c):
        if k.startswith(("model.layers.2.", "model.layers.5.")) and c[k].ndim == 2:
            c[k] = st["e1"][k].copy()
    res = run(ow, "ned/splice", c, rec_for(ow, "ned/splice"))
    d = [d for d in res["sources_detected"] if d["model"].startswith("e1h/model@")][0]
    assert d["splice_layers"] >= 2 and d["main_test"] is False and d["same_start_entry"] and d["weight_overlap"]
    assert res["result"] == "PASS" and [codes(res, k) for k in "NRH"] == [[], [], []]
    assert [b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "weight_overlap"] == ["e1h/model"]


def test_model_named_in_the_repo_and_not_allowed_is_an_overlap_not_a_refusal(ow):
    """Final rule: even a model the record names and that is not on the List (a public model; or flagged) in the weights is only measured."""
    st = ow.tensors
    c = mix(ow, 99, (0.8, st["dA"]))
    rec = rec_for(ow, "eve/named")
    rec["mentioned_not_used"] = [{"id": "alice/ftA", "reason": "comparison"}]
    res = run(ow, "eve/named", c, rec, readme="Compared with alice/ftA.\n")
    assert res["result"] == "PASS" and [codes(res, k) for k in "NRH"] == [[], [], []]
    (d,) = [d for d in res["weight_overlaps"] if d["model"].startswith("alice/ftA@")]
    assert d["named"] and not d["allowed"] and not d["flagged"] and not any("flagged" in n for n in res["notes"])


def test_same_start_rule_is_exact():
    ss = Check.same_start_entry
    assert ss({"kind": "entry", "start": "org/s@abc"}, "org/s@abc")
    assert ss({"kind": "entry", "start": "Org/S@ABC"}, "org/s@abc")
    assert not ss({"kind": "entry", "start": "org/s@abc"}, "org/s@abd")             # another revision of the start
    assert not ss({"kind": "public", "start": "org/s@abc"}, "org/s@abc")            # public models keep the old handling
    assert not ss({"kind": "entry", "start": None}, None) and not ss({"kind": "entry", "start": None}, "org/s@abc")
    assert not ss({"kind": "entry", "start": "org/s@abc"}, None)                    # an entry trained from scratch has no common start


def test_beta_action_is_still_the_documented_value():
    from reax_lineage.measure import Thresholds
    t = Thresholds.load()
    assert t.beta_action == 0.25 and t.deliberate_share == 0.5
