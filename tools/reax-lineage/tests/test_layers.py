"""RULES-v2 2.4 / 2.5: an entry that physically duplicates (or drops) layers of its own start and declares `architecture_changes.layer_map`
reads as the start's weights, not as a copy of, or a merge with, another model. Synthetic tests on fingerprints, then through the whole check."""
import numpy as np
import pytest

from reax_lineage.fp import effective_map, layer_view
from reax_lineage.measure import Thresholds, aligned_keys, copy_stats, frame_check
from synth import CONFIG, REV, SPEC, World, combine, delta, finetune, fp_of, make_base, record
from test_pipeline import codes, rec_for, run

THR = Thresholds.load()
EARLY = [0, 1, 2, 1, 2, 3, 4, 5, 6, 7]            # layers 1 and 2 copied and inserted after layer 2 (entry has 10 layers)
LATE = [0, 1, 2, 3, 4, 3, 4, 5, 6, 7]
ODD = [0, 1, 2, 3, 3, 4, 5, 6, 7]                 # one layer copied, 9 layers: positions of copies coincide with other layers' hashes
PRUNED = [0, 2, 4, 6, 7]


def dup(t, lm):
    """tensors of an entry whose layer i is (a copy of) the start's layer lm[i]; everything else is copied"""
    out = {k: v.copy() for k, v in t.items() if "layers." not in k}
    for i, hl in enumerate(lm):
        p = "model.layers.%d." % hl
        for k, v in t.items():
            if k.startswith(p):
                out["model.layers.%d.%s" % (i, k[len(p):])] = v.copy()
    return out


def cfg_for(lm):
    return dict(CONFIG, num_hidden_layers=len(lm))


def as_dict(lm):
    return {i: v for i, v in enumerate(lm)}


# ------------------------------------------------------------- fingerprints ----

@pytest.fixture(scope="module")
def models():
    root = make_base(1)
    start = finetune(root, 31, 1.0)
    return root, start


@pytest.mark.parametrize("lm", [EARLY, LATE, ODD, PRUNED], ids=["early", "late", "odd", "pruned"])
def test_entry_with_duplicated_or_dropped_layers_is_in_the_frame_of_its_start(models, lm):
    root, start = models
    entry = dup(finetune(start, 61, 0.15), lm)
    S = fp_of(start, "org/start", "s")
    C = fp_of(entry, "x/dup", "e", layer_map=as_dict(lm))
    # without the view: shifted layers (the former behaviour). Early copies leave < half of the keys aligned or, by coincidence of the row
    # positions, compare unrelated layers (min cosine near 0 / negative)
    bare = frame_check(C, S, THR)
    rho = effective_map(as_dict(lm), S)
    V = layer_view(S, rho, C.n_layers, direct=as_dict(lm))
    assert aligned_keys(C, V) == sorted(C.body)                           # every entry key has its start layer
    f = frame_check(C, V, THR)
    assert f["ok"] and f["coverage"] == 1.0 and f["frame_share"] == 1.0 and f["min_cos"] > 0.9, f
    assert bare["coverage"] < 1.0                                          # the unmapped comparison lost layers in every case ...
    if lm != LATE:
        assert not bare["ok"]                                              # ... and failed the frame check unless the copies sit late (coverage exactly 0.5)


def test_early_copies_were_not_confirmed_before(models):
    root, start = models
    C = fp_of(dup(finetune(start, 62, 0.15), EARLY), "x/dup", "e", layer_map=as_dict(EARLY))
    S = fp_of(start, "org/start", "s")
    bare = frame_check(C, S, THR)
    assert bare["coverage"] < 0.5 and not bare["ok"]                      # what the unmapped check saw: only layers 0-2 aligned


def test_duplicated_layers_are_not_a_copy_of_the_start_or_of_another_model(models):
    root, start = models
    other = finetune(start, 63, 0.15)                                      # a public fine-tune of the same start
    entry = dup(finetune(start, 64, 0.15), EARLY)
    lm = as_dict(EARLY)
    C = fp_of(entry, "x/dup", "e", layer_map=lm)
    S, P = fp_of(start, "org/start", "s"), fp_of(other, "pub/ft", "p")
    rho = effective_map(lm, S)
    VS, VP = layer_view(S, rho, C.n_layers, direct=lm), layer_view(P, rho, C.n_layers)
    assert copy_stats(C, VS, THR)["body"] < 0.2 and not copy_stats(C, VS, THR)["is_copy"]
    assert copy_stats(C, VP, THR)["body"] < 0.2 and not copy_stats(C, VP, THR)["is_copy"]
    # a pure depth-up-scaling without any training IS the start's weights: a copy of the start (refused, no penalty), as for any untrained export
    pure = fp_of(dup(start, EARLY), "x/pure", "e", layer_map=lm)
    assert copy_stats(pure, VS, THR)["is_copy"] or copy_stats(pure, VS, THR)["body"] >= 0.999


def test_effective_map():
    S = fp_of(make_base(1), "org/start", "s")
    assert effective_map(None, S) is None
    assert effective_map({i: i for i in range(8)}, S) is None
    assert effective_map(as_dict(EARLY), S) == as_dict(EARLY)
    S2 = fp_of(dup(make_base(2), EARLY), "e/ent", "e", layer_map=as_dict(EARLY))         # a start that is itself depth-changed
    assert effective_map(None, S2) == as_dict(EARLY)                                       # no change declared: it inherits the start's map
    comp = effective_map({0: 0, 1: 1, 2: 2, 3: 9}, S2)                                      # entry layer 3 <- start layer 9 <- root layer 7
    assert comp[3] == 7 and comp[2] == 2


def test_view_of_a_model_with_the_same_map_is_itself(models):
    root, start = models
    lm = as_dict(EARLY)
    P = fp_of(dup(finetune(start, 65, 0.1), EARLY), "p/dup", "p", layer_map=lm)
    assert layer_view(P, effective_map(lm, None), P.n_layers) is P


def test_first_copy_is_used_for_models_other_than_the_start(models):
    root, start = models
    lm = as_dict(EARLY)
    R = fp_of(root, "org/root", "r")
    V = layer_view(R, effective_map(lm, None), 10)
    assert V.n_layers == 10
    for i, hl in enumerate(EARLY):
        for post in ("mlp.up_proj.weight", "self_attn.q_proj.weight"):
            assert np.array_equal(V.body["layers.%d.%s" % (i, post)], R.body["layers.%d.%s" % (hl, post)])
            assert V.row_start("layers.%d.%s" % (i, post)) == R.row_start("layers.%d.%s" % (hl, post))


# --------------------------------------------------------------- whole check ----

@pytest.fixture(scope="module")
def lw(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("lw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    dA, dB = (delta(finetune(start, s, 0.15), start) for s in (41, 42))
    w.add("alice/ftA", combine(start, (1.0, dA)), start="org/start@" + REV(2), rev=REV(3), public_at="2026-01-01T00:00:00Z")
    w.add("bob/ftB", combine(start, (1.0, dB)), start="org/start@" + REV(2), rev=REV(4), public_at="2026-01-02T00:00:00Z")
    # an earlier ENTRY with the same start and the same depth change (to meet an overlap through identical maps) and one without
    dE = delta(finetune(start, 44, 0.15), start)
    w.add("dupe/entry", dup(combine(start, (1.0, dE)), EARLY), kind="entry", start="org/start@" + REV(2), block=500, uid=21, result="PASS",
          rev=REV(30), layer_map=as_dict(EARLY), config=cfg_for(EARLY))
    w.add("plain/entry", combine(start, (1.0, dE)), kind="entry", start="org/start@" + REV(2), block=510, uid=22, result="PASS", rev=REV(31))
    w.tensors = dict(root=root, start=start, dA=dA, dB=dB, dE=dE)
    w.list_path = w.finish()
    return w


def lrec(w, repo, lm, **kw):
    rec = rec_for(w, repo, **kw)
    rec["architecture_changes"]["layer_map"] = lm
    return rec


@pytest.mark.parametrize("lm", [EARLY, LATE, ODD, PRUNED], ids=["early", "late", "odd", "pruned"])
def test_honest_entry_with_declared_layer_change_passes_as_the_start(lw, lm):
    c = dup(finetune(lw.tensors["start"], 70 + len(lm), 0.15), lm)
    res = run(lw, "zed/dup%d" % len(lm), c, lrec(lw, "zed/dup%d" % len(lm), lm), config=cfg_for(lm))
    assert res["result"] == "PASS", (res["reasons"], res["start_confirmed"])
    assert res["start_confirmed"]["confirmed"] and res["sources_detected"] == [] and res["weight_overlaps"] == []
    assert not [r for r in res["reasons"] if r["class"] in "NRH"]
    assert [b["model"].split("@")[0] for b in res["bar_list"]] == ["org/start", "org/root"]
    for sid, s in res["weights"]["samples"].items():
        assert s["frame"]["coverage"] == 1.0 and not s["copy_vs_start"]["is_copy"]
    assert res["weights"]["layer_map"]["entry_layers"] == len(lm)
    assert any("layer_map in effect" in n for n in res["notes"])


def test_dict_form_of_the_map_reads_the_same(lw):
    c = dup(finetune(lw.tensors["start"], 80, 0.15), LATE)
    res = run(lw, "zed/dict", c, lrec(lw, "zed/dict", {str(i): v for i, v in enumerate(LATE) if i != v}), config=cfg_for(LATE))
    assert res["result"] == "PASS", res["reasons"]


def test_undeclared_layer_change_is_not_read_as_the_start(lw):
    """The same weights without a declared layer_map: the config check asks for the declaration and the frame check does not confirm the start."""
    c = dup(finetune(lw.tensors["start"], 81, 0.15), EARLY)
    res = run(lw, "zed/undeclared", c, rec_for(lw, "zed/undeclared"), config=cfg_for(EARLY))
    assert res["result"] == "NEEDS_CORRECTION" and "config" in codes(res, "N")


def test_a_merge_cannot_hide_behind_duplicated_layers(lw):
    """Depth change + 50 % of a public model's change on every layer: the source is still found at the right strength."""
    st = lw.tensors
    c = dup(combine(finetune(st["start"], 82, 0.1), (0.5, st["dA"])), EARLY)
    res = run(lw, "zed/mergedup", c, lrec(lw, "zed/mergedup", EARLY), config=cfg_for(EARLY))
    det = {d["model"].split("@")[0]: d for d in res["sources_detected"]}
    assert "alice/ftA" in det and det["alice/ftA"]["beta"] == pytest.approx(0.5, abs=0.05), res["sources_detected"]
    assert res["result"] == "PASS" and "bob/ftB" not in det               # RULES 3.6 final: a weight overlap (bar list), never a finding
    assert any(b["role"] == "weight_overlap" and b["model"].startswith("alice/ftA@") for b in res["bar_list"])
    # a copy of a public model with duplicated layers is not hidden either: the copy test reads it in the entry's numbering
    pub = dup(combine(st["start"], (1.0, st["dB"])), EARLY)
    res2 = run(lw, "zed/copydup", pub, lrec(lw, "zed/copydup", EARLY), config=cfg_for(EARLY))
    assert res2["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(res2, "H"), (res2["result"], res2["reasons"])


def test_overlap_with_an_earlier_entry_is_found_through_the_view(lw):
    """The earlier same-start entries are in the comparison set once with the same depth change (identical maps) and once without."""
    st = lw.tensors
    c = dup(combine(finetune(st["start"], 83, 0.1), (0.6, st["dE"])), EARLY)
    res = run(lw, "zed/ovl", c, lrec(lw, "zed/ovl", EARLY), config=cfg_for(EARLY))
    ov = {d["model"].split("@")[0]: d for d in res["weight_overlaps"]}
    assert set(ov) == {"dupe/entry", "plain/entry"}, (res["result"], res["reasons"])
    assert all(d["beta"] == pytest.approx(0.6, abs=0.06) for d in ov.values())
    # rule 3.6 (lead, 13 Nov): only a positive joint weight takes a bar-list place. The two sources are one model in two layouts, so the joint fit
    # gives one of them the weight; the other stays on the uid page (weight_overlaps) but not on the bar list.
    on = [b for b in res["bar_list"] if b["role"] == "weight_overlap"]
    assert res["result"] == "PASS" and len(on) >= 1
    assert all((d.get("joint_weight") or 0.0) > 0.0 for d in ov.values() if d["on_bar_list"])
    assert len(ov) == 2


def test_start_that_is_a_depth_changed_entry_is_read_in_its_own_layout(lw):
    """E2 starts from the earlier entry dupe/entry (10 layers) and declares nothing: its rows must sit where the start's rows are."""
    st = lw.tensors
    rev, tensors = lw.models["dupe/entry"]
    c = finetune(tensors, 84, 0.1)
    rec = record("zed/child", REV(0), "dupe/entry", rev, kind="entry", uid=21)
    res = run(lw, "zed/child", c, rec, config=cfg_for(EARLY))
    assert res["result"] == "PASS", (res["reasons"], res["start_confirmed"])
    assert res["start_confirmed"]["confirmed"]
    assert [b["model"].split("@")[0] for b in res["bar_list"]][:3] == ["dupe/entry", "org/start", "org/root"]
    assert res["weights"]["layer_map"]["effective"] == {str(i): v for i, v in enumerate(EARLY)}
