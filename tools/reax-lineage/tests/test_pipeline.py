"""End-to-end stage-0 check on an offline hub: record, listing, repo ids, config, tokenizer, weights, classification."""
import copy
import json
import os

import numpy as np
import pytest

from reax_lineage.check import Check
from reax_lineage.fp import SampleSpec
from reax_lineage.sources import Hub
from synth import (CONFIG, HOTKEY, REV, SEED_B, SPEC, World, combine, delta, finetune, make_base, record, tokenizer,
                   write_repo, fp_of)

COMMIT = "2026-06-01T00:00:00Z"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("w"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)                                   # the listed start (instruct-like, big change)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    A = finetune(start, 41, 0.15)                                     # public fine-tunes of the start by other holders
    B = finetune(start, 42, 0.15)
    w.add("alice/ftA", A, start="org/start@" + REV(2), rev=REV(3), public_at="2026-01-01T00:00:00Z")
    w.add("bob/ftB", B, start="org/start@" + REV(2), rev=REV(4), public_at="2026-01-02T00:00:00Z")
    # later model that must never count (published after the commitment)
    w.add("late/ftL", finetune(start, 43, 0.15), start="org/start@" + REV(2), rev=REV(5), public_at="2026-09-01T00:00:00Z")
    # a second official start (for the wrong-start test), derived from the root
    start2 = finetune(root, 32, 1.0)
    w.add("org/start2", start2, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(6), public_at="2025-07-02T00:00:00Z")
    w.add("flagged/fl", finetune(root, 33, 1.0), kind="list", start="org/root@" + REV(1), listed="both", flagged=True, rev=REV(7))
    w.listing["models"][-1]["flagged"] = True
    w.tensors = {"root": root, "start": start, "start2": start2, "A": A, "B": B}
    w.list_path = w.finish()
    return w


def run(world, repo, tensors, rec, dry=False, seed_b=SEED_B, readme="# model\n", config=None, tok=None, extra_files=None, hotkey=HOTKEY, **kw):
    """`hotkey` = the committing hotkey the check is given (None: unknown, as in a manual run without --hotkey)."""
    rev = REV(1000 + abs(hash(repo)) % 10**6)
    rec = copy.deepcopy(rec)
    if rec is not None:
        rec["entry"] = {"repo": repo, "revision": rev}
    write_repo(world.hub, repo, rev, tensors, config=config, tok=tok, readme=readme, record=rec, extra_files=extra_files)
    ck = Check(repo, rev, hotkey=hotkey, block=1000, commit_time=COMMIT, list_path=world.list_path, db_path=world.dbdir,
               hub=Hub(hub_dir=world.hub), spec=SPEC, cache_dir=None, dry_run=dry, seed_b=None if dry else seed_b, jobs=1,
               tier_b=kw.pop("tier_b", False), **kw)
    return ck.run()


def codes(res, cls=None):
    return [r["code"] for r in res["reasons"] if cls is None or r["class"] == cls]


def rec_for(world, repo, start="org/start", **kw):
    rev, _ = world.models[start]
    return record(repo, REV(0), start, rev, **kw)


def test_honest_finetune_passes_with_bar_list(world):
    c = finetune(world.tensors["start"], 51, 0.15)
    res = run(world, "carol/honest", c, rec_for(world, "carol/honest"))
    assert res["result"] == "PASS", res["reasons"]
    assert res["start_confirmed"]["confirmed"]
    assert [b["model"].split("@")[0] for b in res["bar_list"]] == ["org/start", "org/root"]
    assert res["sources_detected"] == []
    assert res["check_script_sha256"] and res["sample_B_seed_sha256"] and "sealed" not in json.dumps(res)   # only the hash
    assert set(res["weights"]["samples"]) == {"A", "B"}
    assert all(r["model"].split("@")[0] != "late/ftL" for r in res["top5_nearest"])           # later models never candidates


def test_dry_run_is_sample_a_only(world):
    c = finetune(world.tensors["start"], 52, 0.15)
    res = run(world, "carol/dry", c, rec_for(world, "carol/dry"), dry=True)
    assert res["result"] == "PASS" and set(res["weights"]["samples"]) == {"A"} and res["dry_run"]
    assert res["sample_B_seed_sha256"] is None


def test_start_only_candidate_is_not_a_source(world):
    c = finetune(world.tensors["start"], 53, 0.15)
    res = run(world, "carol/own", c, rec_for(world, "carol/own", kind="list"))
    assert res["result"] == "PASS" and all(d["model"].split("@")[0] not in ("org/start", "org/root") for d in res["sources_detected"])


@pytest.mark.parametrize("share", [0.5, 0.3])
def test_other_holders_public_models_in_the_weights_are_weight_overlaps(world, share):
    """RULES 3.6 / 3.8 final: a statistical share of another holder's public model, however large, is a weight overlap: bar list, PASS."""
    st = world.tensors
    c = combine(finetune(st["start"], 54, 0.1), (share, delta(st["A"], st["start"])), (share if share == 0.5 else 0.0, delta(st["B"], st["start"])))
    res = run(world, "dave/merge%d" % int(share * 10), c, rec_for(world, "dave/merge"))
    assert res["result"] == "PASS", (res["reasons"], res["sources_detected"])
    assert [codes(res, k) for k in "NRH"] == [[], [], []]
    det = {d["model"].split("@")[0]: d for d in res["sources_detected"]}
    assert "alice/ftA" in det and det["alice/ftA"]["beta"] == pytest.approx(share, abs=0.03)
    ov = {d["model"].split("@")[0]: d for d in res["weight_overlaps"]}
    assert set(ov) == set(det) and ov["alice/ftA"]["relation"] == "public" and not ov["alice/ftA"]["same_start_entry"]
    bl = {b["model"].split("@")[0]: b for b in res["bar_list"]}
    assert bl["alice/ftA"]["role"] == "weight_overlap" and bl["alice/ftA"]["kind"] == "public" and ("bob/ftB" in bl) == (share == 0.5)
    js = res["joint_share"]                                     # information on the UID page, no verdict
    assert js["share"] == pytest.approx(2 * share if share == 0.5 else share, abs=0.07) and "hidden_share" not in res
    assert (js["lower_bound"] >= 0.5) == (share == 0.5)


def test_source_named_in_readme_but_not_in_record_needs_correction_first(world):
    st = world.tensors
    c = combine(finetune(st["start"], 55, 0.1), (0.5, delta(st["A"], st["start"])), (0.5, delta(st["B"], st["start"])))
    res = run(world, "erin/named", c, rec_for(world, "erin/named"), readme="Merged from alice/ftA and bob/ftB.\n")
    assert res["result"] == "NEEDS_CORRECTION" and codes(res, "N").count("undeclared_id") == 2
    # once the record names them as mentioned the weights are measured: two public models of another holder in the weights are weight overlaps now
    # (before 1.2.0: REFUSED as "two starts"); the entry passes and both models are on its bar list
    rec = rec_for(world, "erin/named2")
    rec["mentioned_not_used"] = [{"id": "alice/ftA", "reason": "comparison"}, {"id": "bob/ftB", "reason": "comparison"}]
    res2 = run(world, "erin/named2", c, rec, readme="Merged from alice/ftA and bob/ftB.\n")
    assert res2["result"] == "PASS", res2["reasons"]
    assert sorted(b["model"].split("@")[0] for b in res2["bar_list"] if b["role"] == "weight_overlap") == ["alice/ftA", "bob/ftB"]
    assert all(d["named"] for d in res2["weight_overlaps"])


def test_missing_model_id_in_readme(world):
    c = finetune(world.tensors["start"], 56, 0.15)
    res = run(world, "frank/readme", c, rec_for(world, "frank/readme"), readme="Based on bob/ftB.\n")
    assert res["result"] == "NEEDS_CORRECTION"
    assert any(r["code"] == "undeclared_id" and "bob/ftb" in r["message"] for r in res["reasons"])
    rec = rec_for(world, "frank/readme2")
    rec["mentioned_not_used"] = [{"id": "bob/ftB", "reason": "comparison only"}]
    assert run(world, "frank/readme2", c, rec, readme="Based on bob/ftB.\n")["result"] == "PASS"


def test_undeclared_tokenizer_change(world):
    c = finetune(world.tensors["start"], 57, 0.15)
    res = run(world, "gina/tok", c, rec_for(world, "gina/tok"), tok=tokenizer(extra=["<new_tok>"]))
    assert res["result"] == "NEEDS_CORRECTION" and "tokenizer" in codes(res, "N")
    rec = rec_for(world, "gina/tok2")
    rec["architecture_changes"]["added_tokens"] = ["<new_tok>"]
    assert run(world, "gina/tok2", c, rec, tok=tokenizer(extra=["<new_tok>"]))["result"] == "PASS"


def test_undeclared_config_change(world):
    c = finetune(world.tensors["start"], 58, 0.15)
    cfg = dict(CONFIG, rope_theta=500000.0)
    res = run(world, "hank/cfg", c, rec_for(world, "hank/cfg"), config=cfg)
    assert res["result"] == "NEEDS_CORRECTION" and "config" in codes(res, "N")
    rec = rec_for(world, "hank/cfg2")
    rec["architecture_changes"]["config_fields"] = {"rope_theta": 500000.0}
    assert run(world, "hank/cfg2", c, rec, config=cfg)["result"] == "PASS"


def test_wrong_start_is_found_as_likely_start(world):
    """Weights are start2 + own change, the record says start."""
    c = finetune(world.tensors["start2"], 59, 0.15)
    res = run(world, "ivy/wrong", c, rec_for(world, "ivy/wrong", start="org/start"))
    assert res["result"] == "NEEDS_CORRECTION" and "start_not_confirmed" in codes(res, "N")
    assert res["start_confirmed"]["likely_start"]["model"].startswith("org/start2@")


def test_copy_of_the_start_is_refused(world):
    res = run(world, "jon/copy", dict(world.tensors["start"]), rec_for(world, "jon/copy"))
    assert res["result"] == "REFUSED" and "copy" in codes(res, "R")


def test_undeclared_copy_of_another_holders_model_is_misdeclaration(world):
    res = run(world, "kim/copy", dict(world.tensors["A"]), rec_for(world, "kim/copy"))
    assert res["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(res, "H")


def test_quantisation_round_trip_refused(world):
    from test_measure import q_int
    c0 = finetune(world.tensors["start"], 60, 0.15)
    c = {k: (q_int(v, 4) if v.ndim == 2 and "layers." in k else v) for k, v in c0.items()}
    res = run(world, "lou/int4", c, rec_for(world, "lou/int4"))
    assert res["result"] == "REFUSED" and "quantisation_round_trip" in codes(res, "R")


def _round_trip(world, seed):
    """An int4 round trip of a fine-tune of the start (the entry of `test_quantisation_round_trip_refused`)."""
    from test_measure import q_int
    c0 = finetune(world.tensors["start"], seed, 0.15)
    return {k: (q_int(v, 4) if v.ndim == 2 and "layers." in k else v) for k, v in c0.items()}


def test_declared_quantisation_aware_training_is_not_a_round_trip_refusal(world):
    """v1.5.1 (RULES 2.4, CRITIQUE-4 M4): `architecture_changes.quantisation_aware_training: true` is the exception to the round-trip refusal. The grid share stays in the
    result (`grid_share`, `weights.samples.*.grid`: the uid page) and an information reason says so; nothing else changes."""
    c = _round_trip(world, 70)
    rec = rec_for(world, "lou/qat")
    rec["architecture_changes"]["quantisation_aware_training"] = True
    res = run(world, "lou/qat", c, rec)
    assert res["result"] == "PASS", res["reasons"]
    assert "quantisation_round_trip" not in codes(res) and codes(res, "R") == [] and codes(res, "N") == [] and codes(res, "H") == []
    assert "quantisation_aware_declared" in codes(res, "I")
    msg = next(r["message"] for r in res["reasons"] if r["code"] == "quantisation_aware_declared")
    assert "quantisation-aware training" in msg and "not a refusal" in msg and "no penalty" in msg
    thr = Check("a/b", "main").thr
    assert set(res["grid_share"]) == {"A", "B"} and all(v >= thr.grid_share for v in res["grid_share"].values())       # the grid share is still in the output
    for sid in ("A", "B"):
        g = res["weights"]["samples"][sid]["grid"]
        assert g["entry"]["max"] >= thr.grid_share and g["start_max"] < thr.grid_share
    dry = run(world, "lou/qat2", c, rec_for(world, "lou/qat2"), dry=True)
    assert dry["result"] == "REFUSED" and "quantisation_round_trip" in codes(dry, "R")                                  # the same entry, undeclared: refused
    rec2 = rec_for(world, "lou/qat3")
    rec2["architecture_changes"]["quantisation_aware_training"] = True
    dry2 = run(world, "lou/qat3", c, rec2, dry=True)
    assert dry2["result"] == "PASS" and dry2["grid_share"]["A"] >= thr.grid_share and "quantisation_aware_declared" in codes(dry2, "I")


def test_an_undeclared_or_false_quantisation_aware_training_is_still_refused(world):
    c = _round_trip(world, 71)
    res = run(world, "lou/noqat", c, rec_for(world, "lou/noqat"))                       # the field is absent
    assert res["result"] == "REFUSED" and "quantisation_round_trip" in codes(res, "R") and "quantisation_aware_declared" not in codes(res)
    rec = rec_for(world, "lou/falseqat")
    rec["architecture_changes"]["quantisation_aware_training"] = False
    res2 = run(world, "lou/falseqat", c, rec)
    assert res2["result"] == "REFUSED" and "quantisation_round_trip" in codes(res2, "R") and "quantisation_aware_declared" not in codes(res2)
    assert all(v >= 0.9 for v in res2["grid_share"].values())


def test_quantisation_aware_training_must_be_the_boolean_true(world):
    """The schema field is a boolean; a string is a schema problem (NEEDS_CORRECTION), never a declaration."""
    c = _round_trip(world, 72)
    rec = rec_for(world, "lou/strqat")
    rec["architecture_changes"]["quantisation_aware_training"] = "true"
    res = run(world, "lou/strqat", c, rec)
    assert res["result"] == "NEEDS_CORRECTION" and "schema" in codes(res, "N") and "quantisation_aware_declared" not in codes(res)
    assert Check.declares_qat({"architecture_changes": {"quantisation_aware_training": "true"}}) is False
    assert Check.declares_qat({"architecture_changes": {"quantisation_aware_training": 1}}) is False
    assert Check.declares_qat({"architecture_changes": None}) is False and Check.declares_qat({}) is False and Check.declares_qat(None) is False
    assert Check.declares_qat({"architecture_changes": {"quantisation_aware_training": True}}) is True


def test_declared_quantisation_aware_training_without_a_round_trip_changes_nothing(world):
    c = finetune(world.tensors["start"], 73, 0.15)
    rec = rec_for(world, "lou/qat-honest")
    rec["architecture_changes"]["quantisation_aware_training"] = True
    res = run(world, "lou/qat-honest", c, rec)
    assert res["result"] == "PASS" and "quantisation_aware_declared" not in codes(res) and codes(res, "R") == []
    assert all(v < 0.9 for v in res["grid_share"].values())


def test_declared_quantisation_aware_training_does_not_allow_non_float_storage(world):
    """Only the round-trip reason is exempt: weights stored in a non-float dtype (rule 2.4: bf16 / fp16 / fp32 only) stay REFUSED."""
    import struct
    from synth import f32_to_bf16_bits
    t = finetune(world.tensors["start"], 74, 0.15)
    rec = rec_for(world, "lou/qat-i8")
    rec["architecture_changes"]["quantisation_aware_training"] = True
    victim = "model.layers.2.mlp.down_proj.weight"
    rev = REV(1000 + abs(hash("lou/qat-i8")) % 10**6)
    rec["entry"] = {"repo": "lou/qat-i8", "revision": rev}
    p = write_repo(world.hub, "lou/qat-i8", rev, t, record=rec)
    hdr, blobs, off = {}, [], 0
    for k, v in sorted(t.items()):
        if k == victim:
            b, dt_ = np.clip(np.round(v / np.abs(v).max() * 127), -127, 127).astype(np.int8).tobytes(), "I8"
        else:
            b, dt_ = f32_to_bf16_bits(v).tobytes(), "BF16"
        hdr[k] = {"dtype": dt_, "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    with open(os.path.join(p, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    ck = Check("lou/qat-i8", rev, hotkey=HOTKEY, block=1000, commit_time=COMMIT, list_path=world.list_path, db_path=world.dbdir,
               hub=Hub(hub_dir=world.hub), spec=SPEC, cache_dir=None, dry_run=True, jobs=1, tier_b=False)
    res = ck.run()
    assert "quantised_dtype" in codes(res, "R") and res["result"] in ("REFUSED", "NEEDS_CORRECTION"), res["reasons"]
    assert any(r["code"] == "quantised_dtype" and r["class"] == "R" for r in res["reasons"])


def _named_copy(world, repo, **kw):
    """A byte-for-byte copy of alice/ftA (a public model of another holder, not the start) with the given record and README."""
    return run(world, repo, dict(world.tensors["A"]), kw.pop("rec"), **kw)


def test_a_declared_copy_is_refused_without_penalty_and_an_unnamed_copy_is_a_candidate(world):
    """RULES 3.5 (S3): a copy of a model that the record or the repo files name (the start included) is REFUSED, no penalty; a copy of a model they do not name is a
    misdeclaration candidate (3.8(a)). The checker's `named` = the copied repo id is in the record (any field the record names ids in) or in the repo files, or is the start."""
    # (1) neither the record nor the files name it: candidate
    r1 = _named_copy(world, "sc/unnamed", rec=rec_for(world, "sc/unnamed"))
    assert r1["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(r1, "H") and "copy" not in codes(r1, "R")
    # (2) the record names it (and the files do not): refused, no candidate; the record fields `mentioned_not_used` and `own_models`, any letter case
    for repo, field, entry in (("sc/rec-mnu", "mentioned_not_used", {"id": "alice/ftA", "reason": "compared against it"}),
                               ("sc/rec-case", "mentioned_not_used", {"id": "Alice/FTA", "reason": "compared against it"}),
                               ("sc/rec-own", "own_models", {"repo": "alice/ftA", "how": "my own earlier model"})):
        rec = rec_for(world, repo)
        rec[field].append(entry)
        r2 = _named_copy(world, repo, rec=rec)
        assert r2["result"] == "REFUSED" and "copy" in codes(r2, "R") and "copy_undeclared" not in codes(r2, "H"), (repo, r2["reasons"])
        assert "no penalty" in next(r["message"] for r in r2["reasons"] if r["code"] == "copy")
    # (3) the repo files name it, the record does not: the missing record entry is a NEEDS_CORRECTION first (`undeclared_id`, 24 h to correct), the copy is listed as a
    #     refused declared copy, and it is never a candidate; after the correction (the record names it) it is REFUSED
    r3 = _named_copy(world, "sc/file", rec=rec_for(world, "sc/file"), readme="Compared with alice/ftA.\n")
    assert r3["result"] == "NEEDS_CORRECTION" and "undeclared_id" in codes(r3, "N") and r3["correction_deadline"]
    assert "copy" in codes(r3, "R") and "copy_undeclared" not in codes(r3, "H")
    rec = rec_for(world, "sc/file2")
    rec["mentioned_not_used"].append({"id": "alice/ftA", "reason": "compared against it"})
    r4 = _named_copy(world, "sc/file2", rec=rec, readme="Compared with alice/ftA.\n")
    assert r4["result"] == "REFUSED" and "copy" in codes(r4, "R") and "copy_undeclared" not in codes(r4, "H")
    # (4) a copy of ANOTHER model than the named one is still a candidate (naming bob/ftB does not cover a copy of alice/ftA)
    rec = rec_for(world, "sc/other")
    rec["mentioned_not_used"].append({"id": "bob/ftB", "reason": "compared against it"})
    r5 = _named_copy(world, "sc/other", rec=rec)
    assert r5["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(r5, "H")
    # (5) the start is named by definition (exists as test_copy_of_the_start_is_refused): REFUSED copy
    r6 = run(world, "sc/start", dict(world.tensors["start"]), rec_for(world, "sc/start"))
    assert r6["result"] == "REFUSED" and "copy" in codes(r6, "R") and codes(r6, "H") == []


def test_start_not_on_list_or_flagged_is_refused(world):
    c = finetune(world.tensors["start"], 61, 0.15)
    res = run(world, "max/unlisted", c, record("max/unlisted", REV(0), "alice/ftA", world.models["alice/ftA"][0]))
    assert res["result"] == "REFUSED" and "start_not_listed" in codes(res, "R")
    cf = finetune(world.models["org/root"][1], 62, 1.0)
    res2 = run(world, "max/flag", cf, record("max/flag", REV(0), "flagged/fl", world.models["flagged/fl"][0]))
    assert res2["result"] == "REFUSED" and "start_flagged" in codes(res2, "R")


def test_teacher_and_dataset_and_licence_rules(world):
    c = finetune(world.tensors["start"], 63, 0.15)
    rec = rec_for(world, "nat/teach")
    rec["teachers"] = [{"name": "Big", "repo_or_api": "closed/model", "use": "distillation"}]
    r1 = run(world, "nat/teach", c, rec)
    assert r1["result"] == "REFUSED" and "teacher_not_allowed" in codes(r1, "R")
    rec = rec_for(world, "nat/data")
    rec["datasets"][0]["licence"] = "CC-BY-NC-4.0"
    assert "dataset_not_allowed" in codes(run(world, "nat/data", c, rec), "R")
    rec = rec_for(world, "nat/lic")
    rec["licence"] = "GPL-3.0"
    assert run(world, "nat/lic", c, rec)["result"] == "NEEDS_CORRECTION"        # not in the schema's licence list


def test_schema_and_hotkey_problems(world):
    c = finetune(world.tensors["start"], 64, 0.15)
    bad = rec_for(world, "olga/bad")
    del bad["teachers"]
    r = run(world, "olga/bad", c, bad)
    assert r["result"] == "NEEDS_CORRECTION" and r["correction_deadline"] and "schema" in codes(r)
    other = rec_for(world, "olga/hk", hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    assert run(world, "olga/hk", c, other)["result"] == "NEEDS_CORRECTION"
    r3 = run(world, "olga/norecord", c, None)
    assert r3["result"] == "NEEDS_CORRECTION" and "record_missing" in codes(r3)


def test_sealed_sample_catches_edits_of_the_published_rows(world):
    """A copy of A whose Sample-A rows are reset to the start: A sees nothing, the sealed Sample B sees the copy."""
    st = world.tensors
    c = {k: v.copy() for k, v in st["A"].items()}
    rng = np.random.default_rng(77)
    ref = fp_of(c, "x/x", seed="A0")                       # row positions of the public sample
    own = lambda ref_rows: ref_rows + (rng.standard_normal(ref_rows.shape) * ref_rows.std() * 0.15).astype(np.float32)
    for key, (r0, n) in ref.meta["row_starts"].items():    # replace the published rows by start + own noise
        l, suf = key.split(".", 2)[1], key.split(".", 2)[2]
        name = "model.layers.%s.%s" % (l, suf)
        c[name][r0:r0 + n] = own(st["start"][name][r0:r0 + n])
    for grp, (name, starts) in ref.meta["head_blocks"].items():         # and the published head rows
        for r0 in starts:
            c[name][r0:r0 + SPEC.head_block] = own(st["start"][name][r0:r0 + SPEC.head_block])
    res = run(world, "pia/tamper", c, rec_for(world, "pia/tamper"))
    # the sealed Sample B sees the copy (whole-model test: beta 1.0) although Sample A sees nothing; as a statistical finding that is a weight overlap
    # (RULES 3.8 final: no misdeclaration from a share alone; a real-size copy with edited sample rows is >= 99.9 % equal on B and a copy)
    assert res["result"] == "PASS", (res["result"], res["reasons"])
    d = next(d for d in res["sources_detected"] if d["model"].startswith("alice/ftA@"))
    assert d["detected_on"] == ["B"] and d["sample_disagreement"] and d["weight_overlap"] and d["beta"] > 0.9
    assert any(b["role"] == "weight_overlap" and b["model"].startswith("alice/ftA@") for b in res["bar_list"])
    dry = run(world, "pia/tamper2", c, rec_for(world, "pia/tamper2"), dry=True)
    assert dry["result"] == "PASS" and dry["weight_overlaps"] == []     # the dry run (Sample A only) cannot see it: that is why B is sealed


def test_head_source(world):
    st = world.tensors
    c = finetune(st["start"], 65, 0.15)
    c["lm_head.weight"] = st["A"]["lm_head.weight"]
    res = run(world, "quin/head", c, rec_for(world, "quin/head"))
    assert res["result"] == "NEEDS_CORRECTION" and "head_source" in codes(res, "N")


def test_start_none_but_weights_are_the_start(world):
    c = finetune(world.tensors["start"], 66, 0.15)
    res = run(world, "ray/none", c, record("ray/none", REV(0), kind="none"))
    assert res["result"] == "NEEDS_CORRECTION" and "start_not_declared" in codes(res, "N")


def test_own_holder_source_is_own_work(world):
    """A candidate of the same holder (same HF account) that is detected is own work, not an undeclared source."""
    st = world.tensors
    own = finetune(st["start"], 67, 0.15)
    world.add("sam/earlier", own, start="org/start@" + REV(2), rev=REV(8), public_at="2026-02-01T00:00:00Z")
    world.finish()
    c = combine(finetune(st["start"], 68, 0.1), (0.6, delta(own, st["start"])))
    res = run(world, "sam/later", c, rec_for(world, "sam/later"), holder_ids=["hf:sam"])        # v1.5.0: holder ids come from the caller only (RULES 2.9), as the intake service passes them
    assert res["result"] == "PASS" and any(d["own_holder"] for d in res["sources_detected"])
