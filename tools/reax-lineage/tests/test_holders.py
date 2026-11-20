"""F3 (intake replay, 13 Nov; RULES-v2 2.9): holder ids come ONLY from the caller.

2.9: entries belong to one holder if they have the same coldkey, or share the Hugging Face account in the repo id of an entry that passed intake and whose lineage
record names the committing hotkey (also through a chain of such links); "a commitment of someone else's repo creates no link". v1.4.1 added hf:<owner of the repo id>
to the entry's own holder ids and to every earlier model's by itself (check.py, 4 places), so a commitment of ALICE's repo compared as Alice's own entry: a copy of
Alice's earlier model was a `duplicate` (REFUSED, no penalty) instead of `copy_undeclared` (3.8(a) candidate), and Alice's models were left out of its bar list.
v1.5.0: nothing is derived from a repo id. The intake service passes the coldkey + the HF accounts linked through PASSED entries (`--holder`); the comparison DB's
`holder_ids` carry the other side. `--auto-hf-holder` (default off) restores the old derivation for manual use.

v1.5.1 (RULES 2.9, last sentence, d7): "For the intake check of an entry, the Hugging Face account in its own repo id also counts as yours when its record names the
committing hotkey, so your own earlier public models count as your own work from your first entry on." So, for the entry's OWN check only, `hf:<account of the entry's repo id>`
is added to the own holder ids exactly when the record names the committing hotkey; never for another model's ids; never when the record does not name it (or no committing
hotkey is known). The tests of v1.5.0 below therefore run with `hotkey=None` (the record cannot be said to name the committing hotkey: someone else's repo, nothing linked);
with a hotkey they would be the new-rule cases at the end of this file."""
import pytest

from reax_lineage.check import Check
from reax_lineage.cli import build_parser
from synth import REV, World, combine, delta, finetune, make_base
from test_pipeline import codes, rec_for, run

ATTACKER = "5attackercoldkey"
OTHER_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def run_unnamed(world, repo, tensors, rec, **kw):
    """The check is not given a committing hotkey, so the record does not "name the committing hotkey" (a manual run without --hotkey)."""
    return run(world, repo, tensors, rec, hotkey=None, **kw)


@pytest.fixture(scope="module")
def hw(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("holders"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    earlier = finetune(start, 41, 0.15)
    # Alice's earlier ENTRY (passed intake): the service wrote the holder ids of that entry into the DB (its coldkey and the account hf:alice)
    w.add("alice/earlier", earlier, kind="entry", start="org/start@" + REV(2), block=500, uid=21, result="PASS", rev=REV(10), holder="hf:alice")
    # a public model of Alice's (the DB builder writes hf:<account> for public models)
    pub = finetune(start, 42, 0.15)
    w.add("alice/public", pub, start="org/start@" + REV(2), rev=REV(11), public_at="2026-01-02T00:00:00Z")
    # a public model of Dora's that the DB builder left WITHOUT holder ids ("a model without them belongs to nobody")
    dora = finetune(start, 43, 0.15)
    k = w.add("dora/public", dora, start="org/start@" + REV(2), rev=REV(12), public_at="2026-01-03T00:00:00Z")
    w.db.index["models"][k]["holder_ids"] = []
    w.tensors = dict(root=root, start=start, earlier=earlier, pub=pub, dora=dora)
    w.list_path = w.finish()
    return w


def test_a_commitment_of_someone_elses_repo_that_copies_that_owners_earlier_model_is_copy_undeclared_not_duplicate(hw):
    """The attacker's holder ids are only its coldkey (the service made no link: the repo is Alice's, nothing of the attacker's passed there)."""
    res = run_unnamed(hw, "alice/later", dict(hw.tensors["earlier"]), rec_for(hw, "alice/later"), holder_ids=[ATTACKER])
    assert res["result"] == "MISDECLARATION_CANDIDATE", res["reasons"]
    assert "copy_undeclared" in codes(res, "H") and "duplicate" not in codes(res, "R")
    d = next(d for d in res["sources_detected"] if d["model"].startswith("alice/earlier@"))
    assert d["copy"] and d["own_holder"] is False


def test_the_same_commitment_without_any_holder_ids_is_also_copy_undeclared(hw):
    """The caller passed nothing (a dry run of an unknown person): still nothing is derived from the repo id."""
    res = run_unnamed(hw, "alice/later2", dict(hw.tensors["earlier"]), rec_for(hw, "alice/later2"))
    assert res["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(res, "H") and "duplicate" not in codes(res, "R")


def test_with_the_link_passed_by_the_caller_it_is_a_duplicate(hw):
    """Alice (or whoever the service linked to hf:alice through a PASSED entry) copying her own earlier entry: a duplicate, refused, no penalty."""
    res = run_unnamed(hw, "alice/later3", dict(hw.tensors["earlier"]), rec_for(hw, "alice/later3"), holder_ids=[ATTACKER, "hf:alice"])
    assert res["result"] == "REFUSED" and "duplicate" in codes(res, "R") and "copy_undeclared" not in codes(res, "H")


def test_a_copy_of_the_owners_public_model_by_a_stranger_is_copy_undeclared_too(hw):
    res = run_unnamed(hw, "alice/later4", dict(hw.tensors["pub"]), rec_for(hw, "alice/later4"), holder_ids=[ATTACKER])
    assert res["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(res, "H")
    res2 = run_unnamed(hw, "alice/later5", dict(hw.tensors["pub"]), rec_for(hw, "alice/later5"), holder_ids=["hf:alice"])
    assert res2["result"] == "REFUSED" and "duplicate" in codes(res2, "R")


def test_the_owners_earlier_model_in_the_weights_is_a_weight_overlap_for_a_stranger_and_own_work_for_the_linked_holder(hw):
    """Not a copy: 60 % of Alice's earlier entry in the weights. A stranger committing Alice's repo name gets it as a weight overlap on the bar list
    (3.6: only entries of the entry's own holder are left out); the linked holder gets own work (left out)."""
    c = combine(finetune(hw.tensors["start"], 68, 0.1), (0.6, delta(hw.tensors["earlier"], hw.tensors["start"])))
    res = run_unnamed(hw, "alice/mix1", c, rec_for(hw, "alice/mix1"), holder_ids=[ATTACKER])
    assert res["result"] == "PASS"
    d = next(d for d in res["sources_detected"] if d["model"].startswith("alice/earlier@"))
    assert d["own_holder"] is False and not d["copy"]
    assert [o["model"].split("@")[0] for o in res["weight_overlaps"]] == ["alice/earlier"]
    assert any(b["role"] == "weight_overlap" and b["model"].startswith("alice/earlier@") for b in res["bar_list"])
    res2 = run_unnamed(hw, "alice/mix2", c, rec_for(hw, "alice/mix2"), holder_ids=["hf:alice"])
    d2 = next(d for d in res2["sources_detected"] if d["model"].startswith("alice/earlier@"))
    assert res2["result"] == "PASS" and d2["own_holder"] is True and res2["weight_overlaps"] == []


def test_auto_hf_holder_restores_the_v1_4_1_derivation_for_manual_use(hw):
    res = run_unnamed(hw, "alice/later6", dict(hw.tensors["earlier"]), rec_for(hw, "alice/later6"), holder_ids=[ATTACKER], auto_hf_holder=True)
    assert res["result"] == "REFUSED" and "duplicate" in codes(res, "R") and "copy_undeclared" not in codes(res, "H")
    res2 = run_unnamed(hw, "alice/later7", dict(hw.tensors["earlier"]), rec_for(hw, "alice/later7"), holder_ids=[ATTACKER], auto_hf_holder=False)
    assert "copy_undeclared" in codes(res2, "H")


def test_the_cli_flag_exists_and_is_off_by_default():
    p = build_parser()
    base = ["check", "--repo", "a/b"]
    assert p.parse_args(base).auto_hf_holder is False and p.parse_args(base + ["--holder", "hf:x"]).holder == ["hf:x"]
    assert p.parse_args(base + ["--auto-hf-holder"]).auto_hf_holder is True
    assert p.parse_args(["dry-run", "--repo", "a/b", "--auto-hf-holder"]).auto_hf_holder is True
    assert "auto_hf_holder" in Check.__init__.__code__.co_varnames and Check("a/b", "main").auto_hf_holder is False


def test_holder_id_helpers():
    c = Check("Alice/x", "main", holder_ids=["5CK", "HF:Bob"])
    assert c.own_holder_ids() == {"5ck", "hf:bob"}                                      # lower-cased, nothing derived from "Alice/x" (no record, no hotkey)
    assert Check("Alice/x", "main").own_holder_ids() == set()
    meta = {"repo": "Carol/y", "holder_ids": ["5Other"]}
    assert c.model_holder_ids(meta) == {"5other"} and c.model_holder_ids({"repo": "Carol/y"}) == set()
    on = Check("Alice/x", "main", holder_ids=["5CK"], auto_hf_holder=True)
    assert on.own_holder_ids() == {"5ck", "hf:alice"} and on.model_holder_ids(meta) == {"5other", "hf:carol"}


# ---- v1.5.1: RULES 2.9, last sentence ----------------------------------------------------------------------------------------------------------------------------------

def mix_of(hw, name, seed=69):
    """60 % of the named public model's change on top of the start, plus an own change (not a copy)."""
    return combine(finetune(hw.tensors["start"], seed, 0.1), (0.6, delta(hw.tensors[name], hw.tensors["start"])))


def test_a_first_entry_whose_record_names_the_committing_hotkey_counts_its_accounts_earlier_public_models_as_own_work(hw):
    """RULES 2.9 (d7): nothing is passed by the caller (a first entry: only its coldkey, no entry has passed yet), but the record names the committing hotkey, so
    `hf:alice` is the entry's own holder id and Alice's earlier public model is own work: not a weight overlap, not on the bar list."""
    c = mix_of(hw, "pub")
    res = run(hw, "alice/first", c, rec_for(hw, "alice/first"))                         # HOTKEY committed, record names HOTKEY, holder_ids not passed
    d = next(d for d in res["sources_detected"] if d["model"].startswith("alice/public@"))
    assert res["result"] == "PASS" and d["own_holder"] is True and not d["copy"], res["reasons"]
    assert res["weight_overlaps"] == [] and all(b["role"] != "weight_overlap" for b in res["bar_list"]) and "weight_overlap" not in codes(res, "I")
    # the same weights, the same repo, but no committing hotkey known: the record cannot be said to name it, the model is a weight overlap on the bar list (v1.5.0 behaviour)
    res0 = run_unnamed(hw, "alice/first-b", c, rec_for(hw, "alice/first-b"))
    d0 = next(d for d in res0["sources_detected"] if d["model"].startswith("alice/public@"))
    assert res0["result"] == "PASS" and d0["own_holder"] is False
    assert [o["model"].split("@")[0] for o in res0["weight_overlaps"]] == ["alice/public"]
    assert any(b["role"] == "weight_overlap" and b["model"].startswith("alice/public@") for b in res0["bar_list"])
    dry = run(hw, "alice/first-c", c, rec_for(hw, "alice/first-c"), dry=True)
    assert dry["weight_overlaps"] == [] and next(d for d in dry["sources_detected"] if d["model"].startswith("alice/public@"))["own_holder"] is True


def test_the_same_rule_makes_a_copy_of_the_accounts_earlier_entry_a_duplicate(hw):
    """The other face of the rule: the entry of Alice's repo whose record names the committing hotkey compares Alice's earlier entry as its own: a copy is a `duplicate`
    (REFUSED, no penalty). OPEN ISSUE for the lead (RESULT.md): the same holds for a stranger who commits Alice's repo and then sends a signed correction that names the
    stranger's hotkey, because the corrected record "names the committing hotkey" too."""
    res = run(hw, "alice/dupe", dict(hw.tensors["earlier"]), rec_for(hw, "alice/dupe"))
    assert res["result"] == "REFUSED" and "duplicate" in codes(res, "R") and "copy_undeclared" not in codes(res, "H"), res["reasons"]
    d = next(d for d in res["sources_detected"] if d["model"].startswith("alice/earlier@"))
    assert d["copy"] and d["own_holder"] is True


def test_the_rule_never_applies_to_other_accounts_or_to_models_without_holder_ids(hw):
    """(1) Another account's public model is not the entry's own work although the record names the committing hotkey. (2) The rule adds the id to the ENTRY's side only: a
    model of the entry's own account that the DB builder left without `holder_ids` belongs to nobody, so it stays a weight overlap."""
    res = run(hw, "bob/first", mix_of(hw, "pub"), rec_for(hw, "bob/first"))
    d = next(d for d in res["sources_detected"] if d["model"].startswith("alice/public@"))
    assert res["result"] == "PASS" and d["own_holder"] is False and [o["model"].split("@")[0] for o in res["weight_overlaps"]] == ["alice/public"]
    res2 = run(hw, "dora/first", mix_of(hw, "dora", 70), rec_for(hw, "dora/first"))
    d2 = next(d for d in res2["sources_detected"] if d["model"].startswith("dora/public@"))
    assert res2["result"] == "PASS" and d2["own_holder"] is False and [o["model"].split("@")[0] for o in res2["weight_overlaps"]] == ["dora/public"]
    # and a copy of it from the same account is an undeclared copy, not a duplicate
    res3 = run(hw, "dora/copy", dict(hw.tensors["dora"]), rec_for(hw, "dora/copy"))
    assert res3["result"] == "MISDECLARATION_CANDIDATE" and "copy_undeclared" in codes(res3, "H") and "duplicate" not in codes(res3, "R")


def test_a_record_that_names_another_hotkey_is_a_hotkey_correction_before_any_weight_is_measured(hw):
    """The service flow of "someone else's repo, the record does not name the committer": the check is given the committing hotkey, the record names another one: NEEDS_CORRECTION
    `hotkey` at the record step, no weight measured, no link, no `duplicate`."""
    rec = rec_for(hw, "alice/other-key", hotkey=OTHER_HOTKEY)
    res = run(hw, "alice/other-key", dict(hw.tensors["earlier"]), rec, holder_ids=[ATTACKER])
    assert res["result"] == "NEEDS_CORRECTION" and "hotkey" in codes(res, "N") and codes(res, "R") == [] and codes(res, "H") == []
    assert not res["weights"].get("samples") and res["sources_detected"] == []


def test_own_holder_ids_rule_as_a_unit():
    from synth import HOTKEY
    named, other = {"hotkey": HOTKEY}, {"hotkey": OTHER_HOTKEY}
    c = Check("Alice/x", "main", hotkey=HOTKEY, holder_ids=["5CK"])
    assert c.own_holder_ids(named) == {"5ck", "hf:alice"}                                # the caller's ids plus the account of the entry's repo id, lower-cased
    assert c.own_holder_ids(other) == {"5ck"} and c.own_holder_ids({}) == {"5ck"} and c.own_holder_ids(None) == {"5ck"} and c.own_holder_ids() == {"5ck"}
    assert Check("Alice/x", "main", holder_ids=["5CK"]).own_holder_ids(named) == {"5ck"}              # no committing hotkey known: cannot be established
    assert Check("Alice/x", "main", hotkey="", holder_ids=["5CK"]).own_holder_ids({"hotkey": ""}) == {"5ck"}
    assert Check("localname", "main", hotkey=HOTKEY).own_holder_ids(named) == set()                    # a local checkpoint named without an account: nothing to add
    assert c.record_names_hotkey(named) and not c.record_names_hotkey(other) and not c.record_names_hotkey("x") and not c.record_names_hotkey(None)
    meta = {"repo": "Alice/y", "holder_ids": ["5Other"]}
    assert c.model_holder_ids(meta) == {"5other"} and c.model_holder_ids({"repo": "Alice/y"}) == set()      # never for other models' ids, whatever the record says
    assert Check("Alice/x", "main", hotkey=HOTKEY, auto_hf_holder=True).own_holder_ids(None) == {"hf:alice"}    # --auto-hf-holder: unconditional, as before


def _run_with_correction(world, repo, tensors, pinned_rec, corrected_rec, tmp_path, **kw):
    """The repo holds `pinned_rec` at the pinned revision; the check reads `corrected_rec` from --record (a signed correction)."""
    import copy, json
    from test_pipeline import write_repo, COMMIT, SPEC, SEED_B, HOTKEY, Hub
    rev = REV(2000 + abs(hash(repo)) % 10**6)
    pinned_rec = copy.deepcopy(pinned_rec); corrected_rec = copy.deepcopy(corrected_rec)
    pinned_rec["entry"] = {"repo": repo, "revision": rev}; corrected_rec["entry"] = {"repo": repo, "revision": rev}
    write_repo(world.hub, repo, rev, tensors, record=pinned_rec)
    p = tmp_path / "correction.json"; p.write_text(json.dumps(corrected_rec))
    ck = Check(repo, rev, hotkey=HOTKEY, block=1000, commit_time=COMMIT, list_path=world.list_path, db_path=world.dbdir,
               hub=Hub(hub_dir=world.hub), spec=SPEC, cache_dir=None, seed_b=SEED_B, jobs=1, tier_b=False, record=str(p), **kw)
    return ck.run()


def test_a_correction_alone_cannot_create_the_own_repo_link(hw, tmp_path):
    """v1.5.2 (lead, closing checker-v7 open point 3): a stranger commits Alice's repo (pinned file names Alice's hotkey) and corrects the record to
    name the stranger's hotkey: the own-repo link is NOT made, so a copy of Alice's earlier entry stays an undeclared copy, never an own duplicate."""
    pinned = rec_for(hw, "alice/stolen", hotkey=OTHER_HOTKEY)
    corrected = rec_for(hw, "alice/stolen")
    res = _run_with_correction(hw, "alice/stolen", dict(hw.tensors["earlier"]), pinned, corrected, tmp_path, holder_ids=[ATTACKER])
    assert "duplicate" not in codes(res, "R") and "copy_undeclared" in codes(res, "H"), res["reasons"]


def test_a_correction_keeps_the_link_when_the_pinned_file_already_named_the_hotkey(hw, tmp_path):
    """The honest case: the pinned file named the committing hotkey and the correction fixes another field: the link stays (copy = duplicate)."""
    pinned = rec_for(hw, "alice/fixed")
    corrected = rec_for(hw, "alice/fixed")
    res = _run_with_correction(hw, "alice/fixed", dict(hw.tensors["earlier"]), pinned, corrected, tmp_path)
    assert "duplicate" in codes(res, "R") and "copy_undeclared" not in codes(res, "H"), res["reasons"]
