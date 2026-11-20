"""Tier B through the whole check (checker 1.3.0): the entry's weights are the declared start's only after Tier B undoes a rotation / norm folding /
permutation. That is NEEDS_CORRECTION (as in v1.1.0; v1.2.0 made it a MISDECLARATION_CANDIDATE; v1.5.1 words the message as a finding the miner can dispute instead of "manual review"),
never a candidate and never a penalty.
Tier B only aligns an entry to the DECLARED START: RULES 3.8(b) for other holders' models is not implemented (CALIBRATION.md 16.3), and these tests pin
that limit: a disguised copy of another holder's model ends as NEEDS_CORRECTION, never PASS and never a candidate."""
import numpy as np
import pytest

from synth import REV, World, combine, delta, finetune, make_base
from test_pipeline import codes, rec_for, run
from test_tierb import attack, rand_orth
from synth import D, FF, HEADS


@pytest.fixture(scope="module")
def dw(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("dw"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    A = finetune(start, 41, 0.15)
    w.add("alice/ftA", A, start="org/start@" + REV(2), rev=REV(3), public_at="2026-01-01T00:00:00Z")
    start2 = finetune(root, 32, 1.0)                                  # a second list start and a public model of another holder built on it
    w.add("org/start2", start2, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(6), public_at="2025-07-02T00:00:00Z")
    B = finetune(start2, 42, 0.15)
    w.add("bob/ftB", B, start="org/start2@" + REV(6), rev=REV(4), public_at="2026-01-02T00:00:00Z")
    w.tensors = dict(root=root, start=start, start2=start2, A=A, B=B)
    w.list_path = w.finish()
    return w


def disguise(t, seed=12):
    Q = rand_orth(D, seed)
    g = lambda l: (1.0 + 0.3 * np.random.default_rng(600 + l).standard_normal(D)).astype(np.float32)
    perms = {l: np.random.default_rng(400 + l).permutation(FF) for l in range(8)}
    hp = {l: np.random.default_rng(500 + l).permutation(HEADS) for l in range(8)}
    return attack(t, Q=Q, g=g, mlp_perm=lambda l: perms[l], head_perm=lambda l: hp[l])


def test_rotated_start_plus_own_training_needs_manual_review(dw):
    own = finetune(dw.tensors["start"], 51, 0.15)
    res = run(dw, "mal/rot", disguise(own), rec_for(dw, "mal/rot"), tier_b=True)
    assert res["tier_b_used"] and res["transform_found"] and res["start_confirmed"]["confirmed"], (res["reasons"], res["start_confirmed"])
    assert res["result"] == "NEEDS_CORRECTION", res["reasons"]
    assert codes(res, "N") == ["transform_found"] and codes(res, "H") == [] and codes(res, "R") == []
    assert res["correction_deadline"] is not None                         # the 24 h correction window of NEEDS_CORRECTION (as v1.1.0)
    msg = [r["message"] for r in res["reasons"] if r["code"] == "transform_found"][0]
    # v1.5.1 (RULES 3.3, S4): the miner is told what the check found and that it can be disputed; no "manual review"; the machine fields are unchanged
    assert msg == ("our check finds your weights are a rotated, permuted or rescaled form of your start; "
                   "if you think the check is wrong, dispute it (the dispute pauses the 24 hours). This result carries no penalty.")
    assert "manual review" not in msg and "3.8" not in msg
    assert [(r["class"], r["code"]) for r in res["reasons"]] == [("N", "transform_found")] and res["transform_found"] is True


def test_rotated_start_with_a_public_overlap_is_still_manual_review_not_a_candidate(dw):
    st = dw.tensors
    own = combine(finetune(st["start"], 52, 0.1), (0.5, delta(st["A"], st["start"])))
    res = run(dw, "mal/rot2", disguise(own, 13), rec_for(dw, "mal/rot2"), tier_b=True)
    assert res["result"] == "NEEDS_CORRECTION" and "transform_found" in codes(res, "N") and codes(res, "H") == [], res["reasons"]


def test_a_disguised_copy_of_another_holders_model_is_never_a_candidate_and_never_pass(dw):
    """3.8(b) for other holders' models is NOT implemented: Tier B only aligns to the declared start. A rotated copy of a public model built on the
    declared start ends as `transform_found` (the other model is not named: the weights are read as the start plus an own change); a rotated copy of
    a model built on ANOTHER start ends as `start_not_confirmed`. Both are NEEDS_CORRECTION, manual review; neither is PASS, neither a candidate."""
    st = dw.tensors
    res = run(dw, "mal/rotA", disguise(dict(st["A"]), 15), rec_for(dw, "mal/rotA"), tier_b=True)
    assert res["result"] == "NEEDS_CORRECTION" and "transform_found" in codes(res, "N") and codes(res, "H") == [], res["reasons"]
    assert set(codes(res, "N")) <= {"transform_found", "head_source"}      # the synthetic disguise leaves the head as A's, which `head_source` (2.3) also sees
    assert res["weight_overlaps"] == [] and not any(d["copy"] for d in res["sources_detected"])      # no other-holder model is named or found by Tier B
    res2 = run(dw, "mal/rotB", disguise(dict(st["B"]), 16), rec_for(dw, "mal/rotB"), tier_b=True)
    assert res2["result"] == "NEEDS_CORRECTION" and "start_not_confirmed" in codes(res2, "N") and codes(res2, "H") == [], res2["reasons"]
    assert not res2["transform_found"]


def test_without_tier_b_a_disguise_is_never_a_candidate(dw):
    """Tier B off (`--no-tier-b`): the failed frame check ends as NEEDS_CORRECTION, manual review - never PASS and never a candidate."""
    res = run(dw, "mal/rot3", disguise(finetune(dw.tensors["start"], 53, 0.15), 14), rec_for(dw, "mal/rot3"), tier_b=False)
    assert res["result"] == "NEEDS_CORRECTION" and "start_not_confirmed" in codes(res, "N") and codes(res, "H") == []


def test_honest_entry_and_unrelated_weights_are_not_disguises(dw):
    res = run(dw, "ok/honest", finetune(dw.tensors["start"], 54, 0.15), rec_for(dw, "ok/honest"), tier_b=True)
    assert res["result"] == "PASS" and not res["tier_b_used"] and not res["transform_found"]
    other = make_base(seed=77)
    res2 = run(dw, "ok/other", other, rec_for(dw, "ok/other"), tier_b=True)
    assert res2["result"] == "NEEDS_CORRECTION" and "start_not_confirmed" in codes(res2, "N") and not res2["transform_found"] and codes(res2, "H") == []


def test_misdeclaration_candidates_come_only_from_undeclared_copies(dw):
    """The whole list of H reason codes is the rule 3.8(a) case; a disguise of the declared start (`transform_found`) is class N, everything
    statistical is class I."""
    from reax_lineage.check import Check
    assert Check.MISDECL_H == ("copy_undeclared",) and "transform_found" in Check.START_N
    import inspect
    src = inspect.getsource(Check)
    # one H reason in the whole class; `transform_found` is raised in two places, both class N: for a start (one parent) and for the parents of a merge (v1.6.0)
    assert src.count('self.reason("H"') == 1 and src.count('self.reason("N", "transform_found"') == 2


def test_the_full_check_runs_tier_b_with_consume_and_the_result_is_the_same(dw, monkeypatch):
    """v1.5.0 (memory, CALIBRATION.md 18.4): run_tier_b asks canonicalise to release the entry's layers; the check's outcome is the one of the tests above."""
    import reax_lineage.tierb as tierb
    seen = []
    real = tierb.canonicalise

    def spy(entry, start, thr, head_dim=None, consume=False):
        seen.append(consume)
        return real(entry, start, thr, head_dim=head_dim, consume=consume)
    monkeypatch.setattr(tierb, "canonicalise", spy)
    own = finetune(dw.tensors["start"], 53, 0.15)
    res = run(dw, "mal/rot3", disguise(own, 14), rec_for(dw, "mal/rot3"), tier_b=True)
    assert seen == [True]
    assert res["tier_b_used"] and res["transform_found"] and res["result"] == "NEEDS_CORRECTION" and codes(res, "N") == ["transform_found"]


def test_a_transient_hugging_face_failure_in_tier_b_is_raised_not_turned_into_needs_correction(dw, monkeypatch):
    """v1.5.1 (checker-v6 open issue 2): an HFError of kind `transient` (429 / 5xx / timeout) inside Tier B used to become the note "tier B failed" and the reason
    `tier_b_unavailable` (NEEDS_CORRECTION: the miner's problem). Now it propagates (the CLI prints error_kind `transient`, exit 2, and the service retries). An HFError
    that is not transient (a 404 on a layer) and any other exception keep the old behaviour."""
    import reax_lineage.tierb as tierb
    from reax_lineage.hf import HFError
    own = finetune(dw.tensors["start"], 54, 0.15)

    def boom(status, kind=None):
        def f(*a, **k):
            raise HFError("HTTP %s for https://hf.example/x" % status, status, kind)
        return f
    for status, kind in ((503, None), (429, None), (None, "transient")):
        monkeypatch.setattr(tierb, "fetch_tierb", boom(status, kind))
        with pytest.raises(HFError) as ei:
            run(dw, "mal/rot-t%s" % (status or "x"), disguise(own, 15), rec_for(dw, "mal/rot-t%s" % (status or "x")), tier_b=True)
        assert ei.value.kind == "transient"
    monkeypatch.setattr(tierb, "fetch_tierb", boom(404))
    res = run(dw, "mal/rot-404", disguise(own, 15), rec_for(dw, "mal/rot-404"), tier_b=True)       # not transient: the old behaviour
    assert res["result"] == "NEEDS_CORRECTION" and "tier_b_unavailable" in codes(res, "N") and any("tier B failed" in n for n in res["notes"])

    def other(*a, **k):
        raise ValueError("a bug, not a hiccup")
    monkeypatch.setattr(tierb, "fetch_tierb", other)
    res2 = run(dw, "mal/rot-bug", disguise(own, 15), rec_for(dw, "mal/rot-bug"), tier_b=True)
    assert res2["result"] == "NEEDS_CORRECTION" and "tier_b_unavailable" in codes(res2, "N")


def test_the_cli_reports_a_transient_failure_in_tier_b_as_error_kind_transient(dw, monkeypatch, capsys):
    """The end of the chain: `check` ends with the clean JSON line and exit 2 (what the intake service reads as `transient`), not with a result."""
    import reax_lineage.tierb as tierb
    from reax_lineage import cli
    from reax_lineage.hf import HFError

    def boom(*a, **k):
        raise HFError("HTTP 503 for https://hf.example/x", 503)
    monkeypatch.setattr(tierb, "fetch_tierb", boom)
    own = finetune(dw.tensors["start"], 55, 0.15)
    from synth import write_repo, HOTKEY
    import copy
    rec = copy.deepcopy(rec_for(dw, "mal/rot-cli"))
    rev = "%040x" % 4242
    rec["entry"] = {"repo": "mal/rot-cli", "revision": rev}
    write_repo(dw.hub, "mal/rot-cli", rev, disguise(own, 16), record=rec)
    code = cli.main(["check", "--repo", "mal/rot-cli", "--rev", rev, "--hotkey", HOTKEY, "--block", "1000", "--commit-time", "2026-06-01T00:00:00Z", "--list", dw.list_path,
                     "--comparison-db", dw.dbdir, "--hub-dir", dw.hub, "--jobs", "1", "--seed", "A0"])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 2 and "Traceback" not in "".join(out)
    import json
    j = json.loads(out[-1])
    assert j["error_kind"] == "transient" and j["http_status"] == 503 and "result" not in j
