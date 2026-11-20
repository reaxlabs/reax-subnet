"""Checker 1.6.0 (RULES-v2 d8 / d8b): declared merges with 2-4 parents. Record schema reax-lineage/2 (parents, from_scratch), the confirmation of every declared parent by the
joint fit, the parents rule (REFUSED), one engine template, the bar list without a cap for declared parents, parents of other holders only when public at the commitment,
copy of one parent, and the comparison set without private entries in the dry run. Every test here fails on checker 1.5.2 (which has no /2 schema and refuses or ignores merges)."""
import copy
import json

import numpy as np
import pytest

from reax_lineage.check import Check
from reax_lineage.sources import Hub
from synth import CONFIG, HOTKEY, REV, SEED_B, SPEC, World, combine, delta, finetune, make_base, record, tokenizer, write_repo
from test_pipeline import COMMIT, codes, run

SA, SB, SC, SD = ("org/startA@" + REV(2), "org/startB@" + REV(3), "org/startC@" + REV(4), "org/startD@" + REV(5))


def prec(repo, parents, **kw):
    """A reax-lineage/2 record. `parents` = [(key 'repo@rev', kind, share, uid?)]; the entry block is filled in by `run`."""
    r = record(repo, REV(0), "org/startA", REV(2))
    r["schema"] = "reax-lineage/2"
    del r["start"]
    r["parents"] = []
    for p in parents:
        key, kind, share = p[0], p[1], p[2]
        rp, _, rv = key.partition("@")
        d = {"kind": kind, "repo": rp, "revision": rv, "share": share}
        if kind == "entry":
            d["uid"] = p[3] if len(p) > 3 else 7
        if kind == "own":
            d["parents"] = p[3]
        r["parents"].append(d)
    r.update(kw)
    return r


@pytest.fixture(scope="module")
def mw(tmp_path_factory):
    """root; four list starts (all fine-tunes of the root with a big change, like instruct models); entries / public models of other holders and of the same holder."""
    w = World(tmp_path_factory.mktemp("mw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    S = {}
    for n, (name, seed) in enumerate((("startA", 31), ("startB", 32), ("startC", 33), ("startD", 34))):
        S[name] = finetune(root, seed, 1.0)
        w.add("org/" + name, S[name], kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2 + n), public_at="2025-07-01T00:00:00Z")
    # an entry of another holder (public since 2026-01-02): a fine-tune of startA with its own change
    E1 = finetune(S["startA"], 41, 0.2)
    w.add("alice/entry1", E1, kind="entry", start=SA, block=500, uid=21, result="PASS", rev=REV(11), public_at="2026-01-02T00:00:00Z")
    # an entry of another holder that is still private
    E2 = finetune(S["startB"], 42, 0.2)
    w.add("bob/entry2", E2, kind="entry", start=SB, block=510, uid=22, result="PASS", rev=REV(12), private=True)
    # a public model of another holder (not on the list): never an allowed parent
    P1 = finetune(S["startA"], 43, 0.2)
    w.add("carl/pub1", P1, kind="public", start=SA, rev=REV(13), public_at="2026-01-03T00:00:00Z")
    w.tensors = dict(root=root, E1=E1, E2=E2, P1=P1, **S)
    w.list_path = w.finish()
    return w


def mix(tensors, root, parts, own_seed=None, own_scale=0.05):
    """root + sum(coef * (model - root)) (+ a small own change): a linear merge of the parents with convex coefficients `parts` = [(coef, name)]."""
    c = combine(root, *[(coef, delta(tensors[name], root)) for coef, name in parts])
    return finetune(c, own_seed, own_scale) if own_seed is not None else c


def merge_world_run(w, repo, parts, parents, **kw):
    t = w.tensors
    c = mix(t, t["root"], parts, own_seed=kw.pop("own_seed", 77))
    return run(w, repo, c, prec(repo, parents), **kw)


# --------------------------------------------------------- a declared merge is checked, not refused ----

def test_two_list_parents_are_confirmed_and_both_are_on_the_bar_list(mw):
    res = merge_world_run(mw, "dina/merge2", [(0.5, "startA"), (0.5, "startB")], [(SA, "list", 0.5), (SB, "list", 0.5)])
    assert res["result"] == "PASS", (res["reasons"], res.get("parents"))
    assert res["record_schema"] == "reax-lineage/2" and res["from_scratch"] is False
    assert [p["confirmed"] for p in res["parents"]] == [True, True]
    for p in res["parents"]:
        assert p["measured_weight"] == pytest.approx(0.5, abs=0.06) and p["measured_lower_bound"] > 0
    roles = {b["model"].split("@")[0]: b["role"] for b in res["bar_list"]}
    assert roles == {"org/startA": "parent", "org/startB": "parent", "org/root": "ancestor"}
    assert res["start_confirmed"]["confirmed"] and res["start_confirmed"]["multi_parent"]
    assert res["parent_keys"] == [SA, SB]
    assert [b["declared_share"] for b in res["bar_list"] if b["role"] == "parent"] == [0.5, 0.5]


def test_three_parents_one_an_entry_of_another_holder_and_its_ancestors_join_the_bar_list(mw):
    """alice/entry1 (public since 2 Jan, fine-tune of startA) + startB + startC: confirmed; the entry's start (startA) and the root are ancestors on the list."""
    res = merge_world_run(mw, "dina/merge3", [(0.4, "E1"), (0.3, "startB"), (0.3, "startC")],
                          [("alice/entry1@" + REV(11), "entry", 0.4, 21), (SB, "list", 0.3), (SC, "list", 0.3)])
    assert res["result"] == "PASS", (res["reasons"], [(p["key"], p["measured_weight"]) for p in res["parents"]])
    assert [round(p["measured_weight"], 1) for p in res["parents"]] == [0.4, 0.3, 0.3]
    roles = [(b["model"].split("@")[0], b["role"]) for b in res["bar_list"]]
    assert roles[:3] == [("alice/entry1", "parent"), ("org/startB", "parent"), ("org/startC", "parent")]
    assert ("org/startA", "ancestor") in roles and ("org/root", "ancestor") in roles


def test_a_parent_that_is_the_family_root_is_confirmed_by_what_the_others_leave(mw):
    """A parent that is the root has no change from the root: its weight is 1 minus the others'."""
    root_key = "org/root@" + REV(1)
    mw.listing["models"].append({"repo": "org/root", "revision": REV(1), "family": "tiny", "licence": "Apache-2.0", "use": "start", "added": "2025-01-01T00:00:00Z", "flagged": False, "official": True})
    json.dump(mw.listing, open(mw.list_path, "w"))
    res = merge_world_run(mw, "dina/rootmix", [(0.6, "startA")], [(root_key, "list", 0.4), (SA, "list", 0.6)])
    assert res["result"] == "PASS", (res["reasons"], [(p["key"], p["measured_weight"], p["measured_lower_bound"]) for p in res["parents"]])
    assert res["parents"][0]["is_root"] and res["parents"][0]["measured_weight"] == pytest.approx(0.4, abs=0.06) and res["parents"][1]["measured_weight"] == pytest.approx(0.6, abs=0.06)
    # the same entry without the root share is not "found": 100 % of startA's change, the root's implicit weight is 0
    res2 = merge_world_run(mw, "dina/rootmix2", [(1.0, "startA")], [(root_key, "list", 0.4), (SA, "list", 0.6)])
    assert res2["result"] == "NEEDS_CORRECTION" and codes(res2, "N") == ["parents_not_confirmed"], res2["reasons"]
    assert res2["reasons"][0]["field"] == "parents/0"


def test_the_weights_of_a_parent_that_is_not_in_the_entry_are_not_confirmed(mw):
    """Declared A, B and C; the weights hold A and B only: NEEDS_CORRECTION for C, naming C's measured weight; A and B stay confirmed; no penalty."""
    res = merge_world_run(mw, "dina/missing", [(0.5, "startA"), (0.5, "startB")], [(SA, "list", 0.4), (SB, "list", 0.4), (SC, "list", 0.2)])
    assert res["result"] == "NEEDS_CORRECTION" and res["correction_deadline"], res["reasons"]
    assert [(r["class"], r["code"], r["field"]) for r in res["reasons"]] == [("N", "parents_not_confirmed", "parents/2")]
    assert [p["confirmed"] for p in res["parents"]] == [True, True, False]
    msg = res["reasons"][0]["message"]
    assert "org/startC" in msg and "declared share 0.20" in msg and "no penalty" not in codes(res, "R") and not codes(res, "R") and not codes(res, "H")
    assert abs(res["parents"][2]["measured_weight"]) < 0.05


def test_a_parent_declared_with_a_wrong_share_is_still_confirmed_when_its_weight_is_found(mw):
    """The shares are approximate; the check confirms that each parent is FOUND (lower bound above 0), it does not police the share beyond 0.10 / sum 1."""
    res = merge_world_run(mw, "dina/approx", [(0.7, "startA"), (0.3, "startB")], [(SA, "list", 0.5), (SB, "list", 0.5)])
    assert res["result"] == "PASS" and [p["confirmed"] for p in res["parents"]] == [True, True], res["reasons"]
    assert [round(p["measured_weight"], 1) for p in res["parents"]] == [0.7, 0.3]


# ------------------------------------------------------------------- the parents rule (RULES 2.2 / 3.3, d8b M6) ----

def test_more_than_four_parents_is_refused_with_no_penalty_and_nothing_is_measured(mw):
    parents = [(SA, "list", 0.2), (SB, "list", 0.2), (SC, "list", 0.2), (SD, "list", 0.2), ("alice/entry1@" + REV(11), "entry", 0.2, 21)]
    res = merge_world_run(mw, "dina/five", [(0.2, "startA"), (0.2, "startB"), (0.2, "startC"), (0.2, "startD"), (0.2, "E1")], parents)
    assert res["result"] == "REFUSED" and [(r["class"], r["code"]) for r in res["reasons"]] == [("R", "parents_rule")], res["reasons"]
    assert "at most 4" in res["reasons"][0]["message"] and res["reasons"][0]["field"] == "parents"
    assert res["weights"] == {} and res["correction_deadline"] is None and not res["bar_list"]


@pytest.mark.parametrize("share", [0.05, 0.0999, 0.0, -0.2])
def test_a_share_below_0_10_is_refused_not_corrected(mw, share):
    res = merge_world_run(mw, "dina/small", [(0.9, "startA"), (0.1, "startB")], [(SA, "list", 1.0 - share if share > 0 else 1.0), (SB, "list", share)])
    assert res["result"] == "REFUSED" and [r["code"] for r in res["reasons"]] == ["parents_rule"], res["reasons"]
    assert res["reasons"][0]["field"] == "parents/1/share" and "below 0.10" in res["reasons"][0]["message"] and res["weights"] == {}


def test_four_parents_at_0_25_and_a_share_of_exactly_0_10_are_allowed(mw):
    res = merge_world_run(mw, "dina/four", [(0.25, "startA"), (0.25, "startB"), (0.25, "startC"), (0.25, "startD")],
                          [(SA, "list", 0.25), (SB, "list", 0.25), (SC, "list", 0.25), (SD, "list", 0.25)])
    assert res["result"] == "PASS", res["reasons"]
    assert sorted(b["model"].split("@")[0] for b in res["bar_list"] if b["role"] == "parent") == ["org/startA", "org/startB", "org/startC", "org/startD"]
    res = merge_world_run(mw, "dina/ten", [(0.9, "startA"), (0.1, "startB")], [(SA, "list", 0.9), (SB, "list", 0.10)])
    assert res["result"] == "PASS", res["reasons"]


def test_the_refusal_beats_a_record_correction(mw):
    """Five parents AND an id in the README that the record does not name: REFUSED (a correction cannot help), not NEEDS_CORRECTION."""
    parents = [(SA, "list", 0.2), (SB, "list", 0.2), (SC, "list", 0.2), (SD, "list", 0.2), ("alice/entry1@" + REV(11), "entry", 0.2, 21)]
    t = mw.tensors
    c = mix(t, t["root"], [(0.2, "startA")], own_seed=1)
    res = run(mw, "dina/five2", c, prec("dina/five2", parents), readme="Thanks to bob/entry2.\n")
    assert res["result"] == "REFUSED" and codes(res) == ["parents_rule"], res["reasons"]


def test_shares_that_do_not_add_up_to_one_are_a_record_correction(mw):
    res = merge_world_run(mw, "dina/sum", [(0.5, "startA"), (0.5, "startB")], [(SA, "list", 0.5), (SB, "list", 0.6)])
    assert res["result"] == "NEEDS_CORRECTION" and [(r["class"], r["code"], r["field"]) for r in res["reasons"]] == [("N", "parent_shares", "parents")], res["reasons"]
    assert "1.100" in res["reasons"][0]["message"] and res["weights"] == {}
    res = merge_world_run(mw, "dina/sum2", [(0.5, "startA"), (0.5, "startB")], [(SA, "list", 0.51), (SB, "list", 0.5)])       # 1.01: inside 1 +/- 0.02
    assert res["result"] == "PASS", res["reasons"]
    res = merge_world_run(mw, "dina/sum3", [(0.5, "startA"), (0.5, "startB")], [(SA, "list", 0.53), (SB, "list", 0.5)])       # 1.03: outside
    assert res["result"] == "NEEDS_CORRECTION" and codes(res) == ["parent_shares"]


def test_the_same_parent_twice_or_the_entry_as_its_own_parent_is_a_record_correction(mw):
    res = merge_world_run(mw, "dina/dup", [(1.0, "startA")], [(SA, "list", 0.5), (SA, "list", 0.5)])
    assert res["result"] == "NEEDS_CORRECTION" and codes(res) == ["parents_duplicate"], res["reasons"]
    t = mw.tensors
    me = run(mw, "dina/self", mix(t, t["root"], [(1.0, "startA")], own_seed=2), prec("dina/self", [(SA, "list", 0.5), (SB, "list", 0.5)]))
    rec = prec("dina/self2", [(SA, "list", 0.5), ("dina/self2@" + REV(1234), "list", 0.5)])
    rev = REV(1000 + abs(hash("dina/self2")) % 10**6)
    rec["parents"][1]["revision"] = rev
    res = run(mw, "dina/self2", mix(t, t["root"], [(1.0, "startA")], own_seed=2), rec)
    assert res["result"] == "NEEDS_CORRECTION" and codes(res) == ["parent_self"], res["reasons"]


# ------------------------------------------------------------------- one engine template, config, tokenizer ----

def test_parents_of_different_engine_templates_are_refused(mw, tmp_path):
    w = World(tmp_path)
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    a, b = finetune(root, 31, 0.3), finetune(root, 32, 0.3)
    w.add("org/startA", a, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    w.add("org/startB", b, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(3), public_at="2025-07-01T00:00:00Z",
          config=dict(CONFIG, architectures=["OtherForCausalLM"], model_type="other", num_hidden_layers=9))
    w.list_path = w.finish()
    c = combine(root, (0.5, delta(a, root)), (0.5, delta(b, root)))
    res = run(w, "dina/tmpl", c, prec("dina/tmpl", [(SA, "list", 0.5), (SB, "list", 0.5)]))
    assert res["result"] == "REFUSED" and "parents_template_mismatch" in codes(res, "R"), res["reasons"]
    msg = next(r["message"] for r in res["reasons"] if r["code"] == "parents_template_mismatch")
    assert "org/startA" in msg and "org/startB" in msg and "num_hidden_layers" in msg and "architectures" in msg
    assert res["weights"] == {}                                                        # nothing is measured for parents no engine template serves


def test_the_text_only_export_of_a_multimodal_parent_is_the_same_template(mw):
    from reax_lineage.evidence import template_differences
    mm = {"architectures": ["XForConditionalGeneration"], "model_type": "x", "text_config": {"model_type": "x_text", "hidden_size": 8, "num_hidden_layers": 2}, "vision_config": {}}
    txt = {"architectures": ["XForCausalLM"], "model_type": "x_text", "hidden_size": 8, "num_hidden_layers": 2}
    assert template_differences({"a": mm, "b": txt}) == []
    assert template_differences({"a": mm, "b": dict(txt, hidden_size=16)})[0]["fields"] == {"hidden_size": [8, 16]}
    # settings that are no part of the template: rope, window, vocabulary
    assert template_differences({"a": dict(txt, rope_theta=1e4, vocab_size=100, sliding_window=64), "b": dict(txt, rope_theta=1e6, vocab_size=120, sliding_window=128)}) == []


def test_the_entry_config_must_equal_every_parents_apart_from_declared_changes(mw):
    t = mw.tensors
    c = mix(t, t["root"], [(0.5, "startA"), (0.5, "startB")], own_seed=3)
    rec = prec("dina/cfg", [(SA, "list", 0.5), (SB, "list", 0.5)])
    res = run(mw, "dina/cfg", c, rec, config=dict(CONFIG, rms_norm_eps=1e-5))
    assert res["result"] == "PASS", res["reasons"]
    rs = [r for r in res["reasons"] if r["code"] == "config" and r["class"] == "I"]
    # the same finding against both parents is ONE problem that names both parents
    assert len(rs) == 1 and rs[0]["field"] == "rms_norm_eps", rs
    assert "[against parent org/startA@00000000, org/startB@00000000]" in rs[0]["message"], rs[0]["message"]
    rec2 = prec("dina/cfg2", [(SA, "list", 0.5), (SB, "list", 0.5)])
    rec2["architecture_changes"]["config_fields"] = {"rms_norm_eps": 1e-5}
    res2 = run(mw, "dina/cfg2", c, rec2, config=dict(CONFIG, rms_norm_eps=1e-5))
    assert res2["result"] == "PASS", res2["reasons"]


def test_added_tokens_are_reported_against_every_parent(mw):
    t = mw.tensors
    c = mix(t, t["root"], [(0.5, "startA"), (0.5, "startB")], own_seed=4)
    rec = prec("dina/tok", [(SA, "list", 0.5), (SB, "list", 0.5)])
    res = run(mw, "dina/tok", c, rec, tok=tokenizer(extra=["<new>"]))
    assert res["result"] == "PASS" and codes(res, "I") == ["tokenizer"], res["reasons"]
    assert "1 added tokens are not in architecture_changes.added_tokens" in res["reasons"][0]["message"] and "[against parent" in res["reasons"][0]["message"]
    rec2 = prec("dina/tok2", [(SA, "list", 0.5), (SB, "list", 0.5)])
    rec2["architecture_changes"]["added_tokens"] = ["<new>"]
    cfg = dict(CONFIG, vocab_size=1201)
    res2 = run(mw, "dina/tok2", c, rec2, tok=tokenizer(extra=["<new>"]), config=cfg)
    assert "tokenizer" not in codes(res2)


# ------------------------------------------------- another holder's entry: public at the commitment (RULES 2.2, d8b M2) ----

def test_a_private_entry_of_another_holder_is_not_an_allowed_parent(mw):
    res = merge_world_run(mw, "dina/priv", [(0.5, "startA"), (0.5, "E2")], [(SA, "list", 0.5), ("bob/entry2@" + REV(12), "entry", 0.5, 22)])
    assert res["result"] == "REFUSED" and [(r["class"], r["code"]) for r in res["reasons"] if r["class"] == "R"] == [("R", "parent_not_public")], res["reasons"]
    assert res["reasons"][0]["field"] == "parents/1" and "still private" in res["reasons"][0]["message"]


def test_an_entry_that_became_public_after_the_commitment_was_not_public_at_the_commitment(mw, tmp_path):
    w = World(tmp_path)
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    a, b = finetune(root, 31, 0.3), finetune(root, 32, 0.3)
    w.add("org/startA", a, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    w.add("bob/late", b, kind="entry", start="org/root@" + REV(1), block=500, uid=22, result="PASS", rev=REV(12), public_at="2026-06-02T00:00:00Z")     # COMMIT = 2026-06-01
    w.add("bob/early", b, kind="entry", start="org/root@" + REV(1), block=501, uid=23, result="PASS", rev=REV(13), public_at="2026-05-31T23:59:59Z")
    w.list_path = w.finish()
    c = combine(root, (0.5, delta(a, root)), (0.5, delta(b, root)))
    late = run(w, "dina/late", c, prec("dina/late", [(SA, "list", 0.5), ("bob/late@" + REV(12), "entry", 0.5, 22)]))
    assert late["result"] == "REFUSED" and "parent_not_public" in codes(late, "R")
    early = run(w, "dina/early", c, prec("dina/early", [(SA, "list", 0.5), ("bob/early@" + REV(13), "entry", 0.5, 23)]))
    assert early["result"] == "PASS", early["reasons"]


def test_one_start_that_is_a_private_entry_of_another_holder_is_refused_for_a_v1_record_too(mw):
    from test_pipeline import rec_for
    rec = record("dina/v1priv", REV(0), "bob/entry2", REV(12), kind="entry", uid=22)
    t = mw.tensors
    res = run(mw, "dina/v1priv", finetune(t["E2"], 5, 0.1), rec)
    assert res["result"] == "REFUSED" and codes(res, "R") == ["start_not_public"] and res["reasons"][0]["field"] == "start", res["reasons"]


def test_void_flagged_and_not_pass_entries_cannot_be_parents(mw, tmp_path):
    w = World(tmp_path)
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    a, b = finetune(root, 31, 0.3), finetune(root, 32, 0.3)
    w.add("org/startA", a, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    for n, (nm, kw) in enumerate((("void", {"void": True}), ("flag", {"flagged": True}), ("nopass", {})), start=12):
        w.add("bob/" + nm, b, kind="entry", start="org/root@" + REV(1), block=500 + n, uid=n, result=("NEEDS_CORRECTION" if nm == "nopass" else "PASS"), rev=REV(n), **kw)
    w.list_path = w.finish()
    c = combine(root, (0.5, delta(a, root)), (0.5, delta(b, root)))
    for nm, n, code in (("void", 12, "parent_void"), ("flag", 13, "parent_flagged"), ("nopass", 14, "parent_not_pass")):
        res = run(w, "dina/" + nm, c, prec("dina/" + nm, [(SA, "list", 0.5), ("bob/%s@%s" % (nm, REV(n)), "entry", 0.5, n)]))
        assert res["result"] == "REFUSED" and code in codes(res, "R"), (nm, res["reasons"])


def test_a_public_model_that_is_not_on_the_list_is_not_an_allowed_parent(mw):
    """carl/pub1 is a public model of another holder, not an entry and not on the list; declared as an entry it is not a known entry; as a list model it is not listed."""
    t = mw.tensors
    for kind in ("entry", "list"):
        extra = [21] if kind == "entry" else []
        res = run(mw, "dina/pub-" + kind, mix(t, t["root"], [(0.5, "startA"), (0.5, "P1")], own_seed=6),
                  prec("dina/pub-" + kind, [(SA, "list", 0.5), (("carl/pub1@" + REV(13)), kind, 0.5, *extra)]))
        assert res["result"] == "REFUSED" and (["parent_not_listed"] == [c for c in codes(res, "R")]), (kind, res["reasons"])


def test_legacy_teacher_notes_do_not_gate_intake_or_create_a_bar(mw):
    t = mw.tensors
    c = mix(t, t["root"], [(0.5, "startA"), (0.5, "startB")], own_seed=7)
    rec = prec("dina/teach", [(SA, "list", 0.5), (SB, "list", 0.5)])
    rec["teachers"] = [{"name": "E2", "repo_or_api": "bob/entry2", "use": "distillation"}]
    res = run(mw, "dina/teach", c, rec)
    assert res["result"] == "PASS" and codes(res, "R") == []
    rec["teachers"] = [{"name": "E1", "repo_or_api": "alice/entry1", "use": "distillation"}]
    res = run(mw, "dina/teach2", c, rec)
    assert res["result"] == "PASS" and all(b["role"] != "teacher" for b in res["bar_list"]), res["reasons"]


# ------------------------------------------------------------------- own models as parents (RULES 2.2, 3.1) ----

@pytest.fixture(scope="module")
def ow(tmp_path_factory):
    """The merge world plus: dina's own public model (on the hub only, not in the comparison DB), dina's own earlier ENTRY (in the DB, her holder id) and bob's model that
    is not hers."""
    w = World(tmp_path_factory.mktemp("ow"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    S = {}
    for n, (name, seed) in enumerate((("startA", 31), ("startB", 32), ("startC", 33))):
        S[name] = finetune(root, seed, 1.0)
        w.add("org/" + name, S[name], kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2 + n), public_at="2025-07-01T00:00:00Z")
    own1 = finetune(S["startA"], 61, 0.2)
    write_repo(w.hub, "dina/own1", REV(50), own1, tok=tokenizer())                    # own public model, hub only
    own_entry = finetune(S["startB"], 62, 0.2)
    w.add("dina/ownentry", own_entry, kind="entry", start=SB, block=400, uid=31, result="PASS", rev=REV(51), holder="hf:dina", private=True)
    bobs = finetune(S["startA"], 63, 0.2)
    write_repo(w.hub, "bob/notmine", REV(52), bobs, tok=tokenizer())
    w.tensors = dict(root=root, own1=own1, ownentry=own_entry, bobs=bobs, **S)
    w.list_path = w.finish()
    return w


def test_an_own_public_model_is_a_parent_with_its_own_parents_in_the_record(ow):
    t = ow.tensors
    c = mix(t, t["root"], [(0.6, "own1"), (0.4, "startB")], own_seed=8)
    rec = prec("dina/withown", [("dina/own1@" + REV(50), "own", 0.6, [{"kind": "list", "repo": "org/startA", "revision": REV(2)}]), (SB, "list", 0.4)])
    res = run(ow, "dina/withown", c, rec)
    assert res["result"] == "PASS", (res["reasons"], [(p["key"], p["measured_weight"]) for p in res["parents"]])
    assert [round(p["measured_weight"], 1) for p in res["parents"]] == [0.6, 0.4] and res["parents"][0]["kind"] == "own"
    roles = [(b["model"].split("@")[0], b["role"], b.get("declared_by", "").split("@")[0]) for b in res["bar_list"]]
    assert roles[:2] == [("dina/own1", "parent", ""), ("org/startB", "parent", "")]
    assert ("org/startA", "ancestor", "dina/own1") in roles and ("org/root", "ancestor", "") in roles       # its declared parent and the root


def test_an_own_parents_own_parents_must_be_allowed_parents_too(ow):
    t = ow.tensors
    c = mix(t, t["root"], [(0.6, "own1"), (0.4, "startB")], own_seed=9)
    rec = prec("dina/ownbad", [("dina/own1@" + REV(50), "own", 0.6, [{"kind": "list", "repo": "carl/unlisted", "revision": REV(99)}]), (SB, "list", 0.4)])
    res = run(ow, "dina/ownbad", c, rec)
    assert res["result"] == "REFUSED" and [(r["code"], r["field"]) for r in res["reasons"] if r["class"] == "R"] == [("parent_not_listed", "parents/0/parents/0")], res["reasons"]
    rec = prec("dina/ownbad2", [("dina/own1@" + REV(50), "own", 0.6, [{"kind": "own", "repo": "bob/notmine", "revision": REV(52), "parents": [{"kind": "list", "repo": "org/startA", "revision": REV(2)}]}]), (SB, "list", 0.4)])
    res = run(ow, "dina/ownbad2", c, rec)
    assert "parent_not_own" in codes(res, "N") and res["reasons"][-1]["field"].startswith("parents/0/parents/0"), res["reasons"]      # a nested own model of somebody else
    assert res["result"] == "NEEDS_CORRECTION"


def test_an_own_entry_in_the_comparison_set_is_a_parent_even_while_it_is_private(ow):
    """Own entries need no public_at: they are yours. The check reads their ancestors from the comparison DB."""
    t = ow.tensors
    c = mix(t, t["root"], [(0.5, "ownentry"), (0.5, "startA")], own_seed=10)
    rec = prec("dina/ownent", [("dina/ownentry@" + REV(51), "own", 0.5, [{"kind": "list", "repo": "org/startB", "revision": REV(3)}]), (SA, "list", 0.5)])
    res = run(ow, "dina/ownent", c, rec)
    assert res["result"] == "PASS", res["reasons"]
    assert any(b["model"].startswith("dina/ownentry@") and b.get("private") for b in res["bar_list"]) and ("org/startB", "ancestor") in [(b["model"].split("@")[0], b["role"]) for b in res["bar_list"]]


def test_a_model_declared_as_own_that_is_not_provably_yours_is_a_correction(ow):
    t = ow.tensors
    c = mix(t, t["root"], [(0.6, "bobs"), (0.4, "startB")], own_seed=11)
    rec = prec("dina/notmine", [("bob/notmine@" + REV(52), "own", 0.6, [{"kind": "list", "repo": "org/startA", "revision": REV(2)}]), (SB, "list", 0.4)])
    res = run(ow, "dina/notmine", c, rec)
    assert res["result"] == "NEEDS_CORRECTION" and [(r["class"], r["code"], r["field"]) for r in res["reasons"]] == [("N", "parent_not_own", "parents/0")], res["reasons"]
    res = run(ow, "dina/notmine2", c, rec, holder_ids=["hf:bob"])                          # the caller knows the link (RULES 2.9)
    assert res["result"] == "PASS", res["reasons"]


# --------------------------------------------------------------- a merge that is a copy of one parent; undeclared sources ----

def test_a_merge_that_is_a_copy_of_one_of_its_parents_is_a_copy(mw):
    t = mw.tensors
    res = run(mw, "dina/copyA", dict(t["startA"]), prec("dina/copyA", [(SA, "list", 0.5), (SB, "list", 0.5)]))
    assert res["result"] == "REFUSED" and "copy" in codes(res, "R") and "no penalty" in next(r["message"] for r in res["reasons"] if r["code"] == "copy"), res["reasons"]
    assert "copy_undeclared" not in codes(res, "H") and res["copy_share"]["A"]["vs_parents"][0] >= 0.999 and res["copy_share"]["A"]["vs_parents"][1] < 0.5
    # the same weights as a copy of another holder's model that is not a parent: a candidate, as for one start
    res = run(mw, "dina/copyP", dict(t["P1"]), prec("dina/copyP", [(SA, "list", 0.5), (SB, "list", 0.5)]))
    assert res["result"] in ("MISDECLARATION_CANDIDATE", "NEEDS_CORRECTION")           # carl/pub1 is not named anywhere: undeclared copy ...
    assert "copy_undeclared" in codes(res, "H") or "parents_not_confirmed" in codes(res, "N")


@pytest.fixture(scope="module")
def cw(tmp_path_factory):
    """Four list parents and four public models (other holders) whose own changes (vs startA) are partly in the entry: more undeclared sources than the bar list takes."""
    w = World(tmp_path_factory.mktemp("cw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    S = {}
    for n, (name, seed) in enumerate((("startA", 31), ("startB", 32), ("startC", 33), ("startD", 34))):
        S[name] = finetune(root, seed, 1.0)
        w.add("org/" + name, S[name], kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2 + n), public_at="2025-07-01T00:00:00Z")
    Q = {}
    for i in range(1, 5):
        Q[i] = delta(finetune(S["startA"], 70 + i, 0.2), S["startA"])
        w.add("q%d/model" % i, combine(S["startA"], (1.0, Q[i])), start=SA, rev=REV(20 + i), public_at="2026-01-0%dT00:00:00Z" % i)
    Z = finetune(S["startA"], 99, 0.2)                          # a public model built on startA whose start the comparison DB does not know: its change from the root contains startA's
    w.add("zed/nostart", Z, kind="public", start=None, rev=REV(40), public_at="2026-01-09T00:00:00Z")
    w.tensors = dict(root=root, Q=Q, Z=Z, **S)
    w.list_path = w.finish()
    return w


def test_the_cap_of_three_is_for_undeclared_sources_only_declared_parents_are_never_capped(cw):
    t = cw.tensors
    base = combine(t["root"], *[(0.25, delta(t[n], t["root"])) for n in ("startA", "startB", "startC", "startD")])
    c = combine(base, (0.5, t["Q"][1]), (0.4, t["Q"][2]), (0.3, t["Q"][3]), (0.2, t["Q"][4]))
    res = run(cw, "dina/cap", c, prec("dina/cap", [(SA, "list", 0.25), (SB, "list", 0.25), (SC, "list", 0.25), (SD, "list", 0.25)]))
    assert res["result"] == "PASS", res["reasons"]
    roles = [b["role"] for b in res["bar_list"]]
    assert roles.count("parent") == 4 and roles.count("weight_overlap") == 3 and roles.count("ancestor") == 1, roles          # four parents, three of four overlaps, the root
    assert len(res["weight_overlaps"]) == 4 and [d["on_bar_list"] for d in res["weight_overlaps"]] == [True, True, True, False]
    assert res["weight_overlaps"][0]["model"].startswith("q1/model") and "weight_overlap_not_listed" in codes(res, "I")
    # a declared parent is never its own overlap
    assert all(d["model"].split("@")[0] not in ("org/startA", "org/startB", "org/startC", "org/startD") for d in res["weight_overlaps"])


def test_head_source_from_a_declared_parent_is_not_a_correction(mw):
    """C5: a head equal to a declared parent's head is nothing new (head_source compares with every declared parent)."""
    from reax_lineage.measure import Thresholds, head_source
    from synth import fp_of
    t = mw.tensors
    thr = Thresholds.load()
    rng = np.random.default_rng(5)
    # the entry: startB's embeddings / head with a changed body
    Cd = {k: (v.copy() if ("embed_tokens" in k or "lm_head" in k or v.ndim != 2) else v + (rng.standard_normal(v.shape) * 0.01).astype(np.float32)) for k, v in t["startB"].items()}
    A, B, X, Cf = fp_of(t["startA"], "x/a", "r1"), fp_of(t["startB"], "x/b", "r2"), fp_of(t["startB"], "x/x", "r4"), fp_of(Cd, "x/c", "r3")      # X: a candidate with startB's head
    assert head_source(Cf, A, X, thr) is True                                                     # the one reference (startA) does not have X's head: taken from X
    assert head_source(Cf, [A], X, thr) is True
    assert head_source(Cf, [A, B], X, thr) is False                                               # startB IS a declared parent: not "from another model"
    assert head_source(Cf, B, X, thr) is False                                                    # (one start, as in v1.5.2)


# --------------------------------------------------------------- private entries: the dry run and the confidential rows ----

@pytest.fixture(scope="module")
def pw(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("pw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    S = {}
    for n, (name, seed) in enumerate((("startA", 31), ("startB", 32))):
        S[name] = finetune(root, seed, 1.0)
        w.add("org/" + name, S[name], kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2 + n), public_at="2025-07-01T00:00:00Z")
    Qp = delta(finetune(S["startA"], 81, 0.2), S["startA"])
    w.add("secretbob/privsrc", combine(S["startA"], (1.0, Qp)), kind="entry", start=SA, block=500, uid=77, result="PASS", rev=REV(30), private=True)
    Qr = delta(finetune(S["startA"], 82, 0.2), S["startA"])
    w.add("pubcarl/revealed", combine(S["startA"], (1.0, Qr)), kind="entry", start=SA, block=501, uid=78, result="PASS", rev=REV(31), public_at="2026-02-01T00:00:00Z")
    w.tensors = dict(root=root, Qp=Qp, Qr=Qr, **S)
    w.list_path = w.finish()
    return w


def private_source_entry(pw):
    t = pw.tensors
    base = combine(t["root"], (0.5, delta(t["startA"], t["root"])), (0.5, delta(t["startB"], t["root"])))
    return combine(base, (0.5, t["Qp"]), (0.4, t["Qr"]))


def test_the_check_compares_with_private_entries_and_names_them_only_as_a_private_entry(pw):
    res = run(pw, "dina/privsrc", private_source_entry(pw), prec("dina/privsrc", [(SA, "list", 0.5), (SB, "list", 0.5)]))
    assert res["result"] == "PASS", res["reasons"]
    det = {d["model"].split("@")[0]: d for d in res["weight_overlaps"]}
    assert set(det) == {"secretbob/privsrc", "pubcarl/revealed"} and det["secretbob/privsrc"]["private"] and not det["pubcarl/revealed"].get("private")
    assert det["secretbob/privsrc"]["public_name"] == "a private entry of this round (uid 77)"
    bl = {b["model"].split("@")[0]: b for b in res["bar_list"]}
    assert bl["secretbob/privsrc"]["private"] and bl["secretbob/privsrc"]["public_name"] == "a private entry of this round (uid 77)"
    text = json.dumps([r["message"] for r in res["reasons"]] + res["notes"])
    assert "secretbob" not in text and "a private entry of this round" in text and "pubcarl/revealed" in text, text


def test_the_dry_run_compares_only_with_public_models_and_revealed_entries(pw):
    res = run(pw, "dina/dryprivsrc", private_source_entry(pw), prec("dina/dryprivsrc", [(SA, "list", 0.5), (SB, "list", 0.5)]), dry=True)
    assert res["result"] == "PASS" and res["dry_run"], res["reasons"]
    assert [d["model"].split("@")[0] for d in res["weight_overlaps"]] == ["pubcarl/revealed"]
    assert "secretbob" not in json.dumps(res["weights"]["samples"]) and "secretbob" not in json.dumps(res["top5_nearest"])


def test_a_private_entry_is_no_parent_in_the_dry_run_either(pw):
    t = pw.tensors
    res = run(pw, "dina/dryparent", combine(t["startA"], (0.5, t["Qp"])), prec("dina/dryparent", [(SA, "list", 0.5), ("secretbob/privsrc@" + REV(30), "entry", 0.5, 77)]), dry=True)
    assert res["result"] == "REFUSED" and codes(res, "R") == ["parent_not_public"], res["reasons"]


def test_the_comparison_db_never_exports_a_private_row_or_its_fingerprints(pw, tmp_path):
    from reax_lineage.db import ComparisonDB
    db = ComparisonDB(pw.dbdir)
    out = db.export_public(str(tmp_path / "pub"))
    assert out["skipped_private"] == 1 and out["rows"] == len(db.models) - 1
    idx = json.load(open(tmp_path / "pub" / "index.json"))
    assert not any("secretbob" in k for k in idx["models"]) and "secretbob" not in json.dumps(idx)
    import os
    assert not any("secretbob" in f for f in os.listdir(tmp_path / "pub" / "fp")) and len(os.listdir(tmp_path / "pub" / "fp")) == 2 * out["rows"]
    with pytest.raises(ValueError):
        db.export_public(str(tmp_path / "pub"))                                              # a non-empty target is refused
    again = ComparisonDB(str(tmp_path / "pub"))
    assert again.candidates("x@y", "tiny", None, 0, "2027-01-01T00:00:00Z", set(), "A", include_private=True) == again.candidates("x@y", "tiny", None, 0, "2027-01-01T00:00:00Z", set(), "A")


# ------------------------------------------------------- the record: schema /2, /1 read as one parent, from scratch, licence ----

def valid_errors(rec, hotkey=HOTKEY):
    from reax_lineage.evidence import validate_record
    return [(p.code, p.field) for p in validate_record(rec, hotkey, rec["entry"]["repo"], rec["entry"]["revision"])]


def test_a_v2_record_validates_and_the_limits_are_not_a_schema_matter():
    rec = prec("dina/x", [(SA, "list", 0.5), (SB, "list", 0.5)])
    rec["entry"] = {"repo": "dina/x", "revision": REV(0)}
    assert valid_errors(rec) == []
    five = prec("dina/x", [(SA, "list", 0.2)] * 5)
    five["entry"] = {"repo": "dina/x", "revision": REV(0)}
    from reax_lineage.evidence import validate_record
    assert [p for p in validate_record(five, HOTKEY, "dina/x", REV(0)) if p.code == "schema"] == []       # five parents and a 0.05 share pass the schema (REFUSED later: parents_rule)


@pytest.mark.parametrize("mutate,field", [
    (lambda r: r.update(parents=[]), "parents"),                                                           # no parent and no from_scratch
    (lambda r: r.update(from_scratch=True), "parents"),                                                    # from_scratch with parents
    (lambda r: r["parents"][0].update(kind="start"), "parents/0/kind"),
    (lambda r: r["parents"][0].pop("share"), "parents/0"),
    (lambda r: r["parents"][0].update(share="half"), "parents/0/share"),
    (lambda r: r["parents"][0].update(kind="entry"), "parents/0"),                                         # entry needs a uid
    (lambda r: r["parents"][0].update(kind="own"), "parents/0"),                                           # own needs its parents
    (lambda r: r["parents"][0].update(parents=[{"kind": "list", "repo": "a/b", "revision": REV(1)}]), "parents/0"),   # a list parent has none
    (lambda r: r.update(schema="reax-lineage/3"), "schema"),
    (lambda r: r.update(start={"kind": "none"}), ""),                                                      # the v1 field is not part of /2
])
def test_v2_schema_problems_are_schema_corrections(mutate, field):
    rec = prec("dina/x", [(SA, "list", 0.5), (SB, "list", 0.5)])
    rec["entry"] = {"repo": "dina/x", "revision": REV(0)}
    mutate(rec)
    errs = valid_errors(rec)
    assert errs and all(c == "schema" for c, _ in errs), errs
    assert any(f == field or f.startswith(field) for _, f in errs) or field == "", errs


def test_a_record_from_scratch_is_parents_empty_and_from_scratch_true(mw):
    t = mw.tensors
    rec = prec("dina/scratch", [])
    rec["from_scratch"] = True
    res = run(mw, "dina/scratch", make_base(99), rec)
    assert res["result"] == "PASS" and res["from_scratch"] is True and res["parents"] == [] and res["parent_keys"] == [] and res["bar_list"] == [], res["reasons"]
    rec2 = prec("dina/scratch2", [])
    rec2["from_scratch"] = False
    res2 = run(mw, "dina/scratch2", make_base(99), rec2)
    assert res2["result"] == "NEEDS_CORRECTION" and codes(res2) == ["schema"]


def test_a_v1_record_is_read_as_one_parent_with_share_1(mw):
    t = mw.tensors
    rec = record("dina/v1", REV(0), "org/startA", REV(2))
    res = run(mw, "dina/v1", finetune(t["startA"], 12, 0.1), rec)
    assert res["result"] == "PASS" and res["record_schema"] == "reax-lineage/1", res["reasons"]
    assert [(p["key"], p["declared_share"], p["kind"]) for p in res["parents"]] == [(SA, 1.0, "list")] and res["parent_keys"] == [SA]
    assert [(b["model"], b["role"]) for b in res["bar_list"]] == [(SA, "start"), ("org/root@" + REV(1), "ancestor")]
    # a /2 record with ONE parent behaves the same (role start, the single-start path)
    res2 = run(mw, "dina/v2one", finetune(t["startA"], 12, 0.1), prec("dina/v2one", [(SA, "list", 1.0)]))
    assert res2["result"] == "PASS" and [(b["model"], b["role"]) for b in res2["bar_list"]] == [(SA, "start"), ("org/root@" + REV(1), "ancestor")], res2["reasons"]
    assert res2["start_confirmed"]["confirmed"] and "multi_parent" not in res2["start_confirmed"]


def test_minimal_schema_and_legacy_dispatch():
    from reax_lineage.evidence import load_schema, validate_record

    for rec in ({"hotkey": HOTKEY, "from_scratch": True},
                {"hotkey": HOTKEY, "parents": [{"kind": "list", "repo": "org/startA", "revision": REV(2), "share": 1}]}):
        assert validate_record(rec, HOTKEY, "dina/x", REV(10)) == []
    assert validate_record({"hotkey": HOTKEY}, HOTKEY, "dina/x", REV(10))
    assert validate_record({"from_scratch": True}, HOTKEY, "dina/x", REV(10))
    for version in ("reax-lineage/1", "reax-lineage/2"):
        schema = load_schema(version)
        assert schema["properties"]["schema"]["const"] == version
        assert "entry" not in schema["required"]


def test_a_declared_parent_named_in_the_readme_is_no_undeclared_id(mw):
    t = mw.tensors
    c = mix(t, t["root"], [(0.5, "startA"), (0.5, "startB")], own_seed=2)
    res = run(mw, "dina/readme", c, prec("dina/readme", [(SA, "list", 0.5), (SB, "list", 0.5)]), readme="A merge of org/startA and org/startB.\n")
    assert res["result"] == "PASS", res["reasons"]
    res = run(mw, "dina/readme2", c, prec("dina/readme2", [(SA, "list", 0.5), (SB, "list", 0.5)]), readme="A merge of org/startA, org/startB and alice/entry1.\n")
    assert res["result"] == "PASS" and "undeclared_id" in codes(res, "I")


# ------------------------------------------------------------------- Tier B, a rotated merge, the confirmation without Tier B ----

@pytest.fixture(scope="module")
def tw(tmp_path_factory):
    """Small deltas (like real fine-tunes): a merge of two parents stays in the frame of its mixture, and Tier B can align it."""
    w = World(tmp_path_factory.mktemp("tw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    S = {}
    for n, (name, seed) in enumerate((("startA", 31), ("startB", 32))):
        S[name] = finetune(root, seed, 0.2)
        w.add("org/" + name, S[name], kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2 + n), public_at="2025-07-01T00:00:00Z")
    w.tensors = dict(root=root, **S)
    w.list_path = w.finish()
    return w


def test_a_merge_whose_weights_are_in_another_frame_is_not_confirmed_without_tier_b(tw):
    from test_disguise import disguise
    t = tw.tensors
    c = disguise(mix(t, t["root"], [(0.5, "startA"), (0.5, "startB")], own_seed=3, own_scale=0.03))
    res = run(tw, "dina/rotnob", c, prec("dina/rotnob", [(SA, "list", 0.5), (SB, "list", 0.5)]), tier_b=False)
    assert res["result"] == "NEEDS_CORRECTION" and [(r["code"], r["field"]) for r in res["reasons"]][:1] == [("parents_not_confirmed", "parents")], res["reasons"]
    assert "Tier B is not available" in res["reasons"][0]["message"] and not any(p["confirmed"] for p in res["parents"])


def test_a_rotated_merge_is_a_transform_found_correction_with_tier_b(tw):
    from test_disguise import disguise
    t = tw.tensors
    c = disguise(mix(t, t["root"], [(0.5, "startA"), (0.5, "startB")], own_seed=3, own_scale=0.03))
    res = run(tw, "dina/rot", c, prec("dina/rot", [(SA, "list", 0.5), (SB, "list", 0.5)]), tier_b=True)
    assert res["tier_b_used"] and res["result"] == "NEEDS_CORRECTION", (res["reasons"], res["start_confirmed"])
    assert [(r["class"], r["code"], r["field"]) for r in res["reasons"]] == [("N", "transform_found", "parents")] and res["transform_found"] is True
    assert "rotated, permuted or rescaled form of your parents" in res["reasons"][0]["message"] and "no penalty" in res["reasons"][0]["message"]
    assert not codes(res, "H") and not codes(res, "R")


def test_an_undisguised_merge_with_small_deltas_passes_without_tier_b(tw):
    t = tw.tensors
    res = run(tw, "dina/plain", mix(t, t["root"], [(0.5, "startA"), (0.5, "startB")], own_seed=3, own_scale=0.03), prec("dina/plain", [(SA, "list", 0.5), (SB, "list", 0.5)]), tier_b=True)
    assert res["result"] == "PASS" and not res["tier_b_used"] and res["start_confirmed"]["frame_vs_mixture"]["A"] >= 0.97, res["reasons"]


def test_the_weights_of_an_own_parent_must_contain_the_parents_the_record_names_for_it(ow):
    t = ow.tensors
    # own1 = fine-tune of startA; the record says it came from startC (not in its weights): not found
    c = mix(t, t["root"], [(0.6, "own1"), (0.4, "startB")], own_seed=12)
    rec = prec("dina/ownlie", [("dina/own1@" + REV(50), "own", 0.6, [{"kind": "list", "repo": "org/startC", "revision": REV(4)}]), (SB, "list", 0.4)])
    res = run(ow, "dina/ownlie", c, rec)
    assert res["result"] == "NEEDS_CORRECTION" and [(r["code"], r["field"]) for r in res["reasons"] if r["class"] == "N"] == [("own_parent_not_confirmed", "parents/0/parents/0")], res["reasons"]
    assert "org/startC" in res["reasons"][0]["message"] and not codes(res, "R")
    # the right parent, and two parents of which one is a decoy that is also in the weights (a merge of startA and startB inside the own model)
    merged_own = mix(t, t["root"], [(0.7, "startA"), (0.3, "startB")], own_seed=13)
    write_repo(ow.hub, "dina/own2", REV(53), merged_own, tok=tokenizer())
    c2 = mix(t, t["root"], [(0.5, "startC")], own_seed=14)
    c2 = combine(c2, (0.5, delta(merged_own, t["root"])))
    rec2 = prec("dina/ownok", [("dina/own2@" + REV(53), "own", 0.5, [{"kind": "list", "repo": "org/startA", "revision": REV(2)}, {"kind": "list", "repo": "org/startB", "revision": REV(3)}]), (SC, "list", 0.5)])
    res2 = run(ow, "dina/ownok", c2, rec2)
    assert res2["result"] == "PASS", res2["reasons"]


def test_the_frame_reference_is_the_mixture_of_the_parents_normalised_to_one(mw):
    """The fit of an entry with a large change of its own can add up to less than 1 (0.56 for round-1 v5h); the mixture the frame check uses is built from the PROPORTIONS."""
    from reax_lineage.fp import SampleSpec
    from synth import fp_of
    t = mw.tensors
    R, A, B = fp_of(t["root"], "o/r", "r0"), fp_of(t["startA"], "o/a", "r1"), fp_of(t["startB"], "o/b", "r2")
    keys = sorted(k for k in R.body)
    small = {"keys": keys, "per": [{"weight": 0.2, "is_root": False}, {"weight": 0.2, "is_root": False}]}
    half = {"keys": keys, "per": [{"weight": 0.5, "is_root": False}, {"weight": 0.5, "is_root": False}]}
    uneven = {"keys": keys, "per": [{"weight": 0.15, "is_root": False}, {"weight": 0.45, "is_root": False}]}
    m1, m2, m3 = Check.mixture_fp(R, [A, B], small), Check.mixture_fp(R, [A, B], half), Check.mixture_fp(R, [A, B], uneven)
    k = keys[0]
    assert np.allclose(m1.body[k], m2.body[k], atol=1e-6) and np.allclose(m2.body[k], R.body[k] + 0.5 * (A.body[k] - R.body[k]) + 0.5 * (B.body[k] - R.body[k]), atol=1e-5)
    assert np.allclose(m3.body[k], R.body[k] + 0.25 * (A.body[k] - R.body[k]) + 0.75 * (B.body[k] - R.body[k]), atol=1e-5)
    # a parent that is the root contributes nothing of its own change; a negative weight counts as 0
    rooted = {"keys": keys, "per": [{"weight": 0.4, "is_root": True}, {"weight": 0.6, "is_root": False}]}
    assert np.allclose(Check.mixture_fp(R, [R, A], rooted).body[k], R.body[k] + 0.6 / 1.0 * (A.body[k] - R.body[k]), atol=1e-5)


def test_an_own_parent_that_is_only_on_the_hub_and_named_in_the_readme_is_not_its_own_weight_overlap(ow):
    """dina/own1 is not in the comparison DB; the README names it. The evidence scan would fingerprint it as a model "named in the repo": it is a declared parent, never a candidate."""
    t = ow.tensors
    c = mix(t, t["root"], [(0.6, "own1"), (0.4, "startB")], own_seed=15)
    rec = prec("dina/ownreadme", [("dina/own1@" + REV(50), "own", 0.6, [{"kind": "list", "repo": "org/startA", "revision": REV(2)}]), (SB, "list", 0.4)])
    res = run(ow, "dina/ownreadme", c, rec, readme="Built on dina/own1 and org/startB.\n")
    assert res["result"] == "PASS", res["reasons"]
    assert res["weight_overlaps"] == [] and res["sources_detected"] == [], [d["model"] for d in res["sources_detected"]]
    assert not any(r["model"].startswith("dina/own1@") for r in res["top5_nearest"])                      # it is not even measured as a candidate


def test_a_row_flagged_private_is_private_even_when_it_carries_a_public_at(tmp_path):
    """`private` decides, not `public_at` alone: a row that is flagged private (still unrevealed) with a stale / early public_at is not public."""
    w = World(tmp_path)
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    a, b = finetune(root, 31, 0.3), finetune(root, 32, 0.3)
    w.add("org/startA", a, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    key = w.add("bob/flagged", b, kind="entry", start="org/root@" + REV(1), block=500, uid=22, result="PASS", rev=REV(12), private=True)
    w.db.models[key]["public_at"] = "2026-01-02T00:00:00Z"
    w.list_path = w.finish()
    c = combine(root, (0.5, delta(a, root)), (0.5, delta(b, root)))
    res = run(w, "dina/flag", c, prec("dina/flag", [(SA, "list", 0.5), ("bob/flagged@" + REV(12), "entry", 0.5, 22)]))
    assert res["result"] == "REFUSED" and codes(res, "R") == ["parent_not_public"] and "still private" in res["reasons"][0]["message"], res["reasons"]


def test_a_model_that_resembles_a_declared_parent_is_no_overlap_of_the_merge(cw):
    """The candidates are measured against the entry's change AFTER its declared parents (fit_parents), not against its whole change from the root: zed/nostart contains startA's
    change (its start is unknown), startA is a declared parent with 0.25, and that must not make zed/nostart a weight overlap of the entry."""
    t = cw.tensors
    c = combine(t["root"], *[(0.25, delta(t[n], t["root"])) for n in ("startA", "startB", "startC", "startD")])
    res = run(cw, "dina/resemble", c, prec("dina/resemble", [(SA, "list", 0.25), (SB, "list", 0.25), (SC, "list", 0.25), (SD, "list", 0.25)]))
    assert res["result"] == "PASS" and res["sources_detected"] == [] and res["weight_overlaps"] == [], ([d["model"] for d in res["sources_detected"]], res["reasons"])
    assert [b["role"] for b in res["bar_list"]].count("weight_overlap") == 0


def test_a_single_own_parent_is_the_start_of_the_entry_with_its_parents_on_the_bar_list(ow):
    """One parent of kind own (a /2 record): the single-start path, role start, the parents the record names for it as declared ancestors."""
    t = ow.tensors
    c = finetune(t["own1"], 16, 0.05)
    rec = prec("dina/singleown", [("dina/own1@" + REV(50), "own", 1.0, [{"kind": "list", "repo": "org/startA", "revision": REV(2)}])])
    res = run(ow, "dina/singleown", c, rec)
    assert res["result"] == "PASS", res["reasons"]
    assert [(b["model"].split("@")[0], b["role"]) for b in res["bar_list"]] == [("dina/own1", "start"), ("org/startA", "ancestor"), ("org/root", "ancestor")], res["bar_list"]
    assert res["start_confirmed"]["confirmed"] and "multi_parent" not in res["start_confirmed"]
    bad = prec("dina/singleown2", [("dina/own1@" + REV(50), "own", 1.0, [{"kind": "list", "repo": "org/startC", "revision": REV(4)}])])
    res = run(ow, "dina/singleown2", c, bad)
    assert res["result"] == "NEEDS_CORRECTION" and "own_parent_not_confirmed" in codes(res, "N"), res["reasons"]


def test_the_cli_export_db_command_writes_the_public_rows_only(pw, tmp_path, capsys):
    from reax_lineage import cli
    out = str(tmp_path / "exp")
    assert cli.main(["export-db", "--comparison-db", pw.dbdir, "--out", out]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["skipped_private"] == 1 and printed["rows"] >= 3
    assert "secretbob" not in open(out + "/index.json").read()
    assert cli.build_parser().parse_args(["dry-run", "--repo", "a/b"]).cmd == "dry-run"


def test_a_failing_parent_fit_never_crashes_the_check_and_is_said_to_the_miner(mw, monkeypatch):
    """If the joint fit of the parents cannot be computed (an exception on a sample), the check ends with a result: NEEDS_CORRECTION `parents_not_confirmed` per parent that names the
    failure, no candidate is measured (said in a note), nothing is refused, no penalty."""
    def boom(self, C, R_fp, P_fp):
        raise ValueError("no tensor of the entry has the same shape and sampled rows as the family root and every parent")
    monkeypatch.setattr(Check, "fit_parents", boom)
    res = merge_world_run(mw, "dina/fitfail", [(0.5, "startA"), (0.5, "startB")], [(SA, "list", 0.5), (SB, "list", 0.5)])
    assert res["result"] == "NEEDS_CORRECTION" and [(r["class"], r["code"], r["field"]) for r in res["reasons"]] == [("N", "parents_not_confirmed", "parents/0"), ("N", "parents_not_confirmed", "parents/1")]
    assert "the joint fit of your weights on your parents failed" in res["reasons"][0]["message"] and "ValueError" in res["reasons"][0]["message"]
    assert any("no candidate was measured" in n for n in res["notes"]) and res["weight_overlaps"] == [] and not codes(res, "R") and not codes(res, "H")
    assert [p["confirmed"] for p in res["parents"]] == [False, False] and res["parents_fit"]["A"]["error"].startswith("ValueError")


@pytest.mark.parametrize("dry", [False, True])
@pytest.mark.parametrize("schema", ["reax-lineage/1", "reax-lineage/2"])
@pytest.mark.parametrize("legacy_entry_revision", [{}, {"revision": "b" * 40}, {"revision": "placeholder"}])
def test_external_revision_pins_record_and_result(mw, dry, schema, legacy_entry_revision):
    repo, rev = "dina/external-pin", REV(101)
    if schema == "reax-lineage/1":
        rec = record(repo, rev, "org/startA", REV(2))
    else:
        rec = prec(repo, [(SA, "list", 1)])
    rec["entry"] = {"repo": repo, **legacy_entry_revision}
    # A different file at a legacy SHA must never be selected.
    write_repo(mw.hub, repo, "b" * 40, mw.tensors["startA"])
    write_repo(mw.hub, repo, rev, finetune(mw.tensors["startA"], 89, 0.15), record=rec)
    res = Check(repo, rev, hotkey=HOTKEY, block=1000, commit_time=COMMIT,
                list_path=mw.list_path, db_path=mw.dbdir, hub=Hub(hub_dir=mw.hub),
                spec=SPEC, dry_run=dry, seed_b=None if dry else SEED_B, jobs=1).run()
    assert res["result"] == "PASS", res["reasons"]
    assert res["entry"]["revision"] == rev
    assert set(res["weights"]["samples"]) == ({"A"} if dry else {"A", "B"})


def test_self_parent_uses_external_revision():
    from reax_lineage.evidence import validate_record

    repo, rev, prior = "dina/entry", REV(101), REV(100)
    rec = prec(repo, [(repo + "@" + rev, "entry", 1)])
    rec["entry"] = {"repo": repo, "revision": prior}
    assert [p.code for p in validate_record(rec, HOTKEY, repo, rev)] == ["parent_self"]

    rec["parents"][0]["revision"] = prior
    assert validate_record(rec, HOTKEY, repo, rev) == []
