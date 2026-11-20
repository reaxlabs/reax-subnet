"""The command line: fingerprint --local (what the sibling run uses on checkpoints), dry-run, check, and the stale-cache guard."""
import json
import os

import numpy as np
import pytest

import reax_lineage.check as check_mod
from reax_lineage.cli import main
from reax_lineage.fp import Fingerprint, SampleSpec, build_fingerprint
from reax_lineage.hf import LocalSource
from synth import HOTKEY, REV, SPEC, World, finetune, make_base, record, write_repo, write_safetensors
from test_pipeline import COMMIT, rec_for


def run_cli(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    return code, (json.loads(out[-1]) if out else None)


def test_fingerprint_local_matches_the_in_memory_build_and_never_returns_a_stale_cache(tmp_path, capsys):
    d1 = tmp_path / "ckpt"
    d1.mkdir()
    t = make_base(5)
    write_safetensors(str(d1 / "model.safetensors"), t)
    cache = str(tmp_path / "cache")
    spec = SampleSpec(rows=16, min_elems=20_000)
    code, o1 = run_cli(capsys, "fingerprint", "--local", str(d1), "--name", "run/final", "--rows", "16", "--min-elems", "20000", "--cache", cache,
                       "--out", str(tmp_path / "a.npz"))
    assert code == 0 and o1["repo"] == "run/final" and o1["n_tensors"] > 0 and o1["seed_sha256"]
    ref = build_fingerprint(LocalSource(str(d1)), "A0", "A", spec)
    fa = Fingerprint.load(str(tmp_path / "a.npz"))
    assert sorted(fa.body) == sorted(ref.body) and all(np.array_equal(fa.body[k], ref.body[k]) for k in ref.body)
    # a later checkpoint of the same run: same names, same header, other values, same path -> must be fingerprinted again, not served from the cache
    t2 = {k: (v + 0.01 if v.ndim == 2 else v) for k, v in t.items()}
    write_safetensors(str(d1 / "model.safetensors"), t2)
    os.utime(str(d1 / "model.safetensors"), (1_900_000_000, 1_900_000_000))
    run_cli(capsys, "fingerprint", "--local", str(d1), "--name", "run/final", "--rows", "16", "--min-elems", "20000", "--cache", cache,
            "--out", str(tmp_path / "b.npz"))
    fb = Fingerprint.load(str(tmp_path / "b.npz"))
    k = next(iter(fb.body))
    assert not np.array_equal(fa.body[k], fb.body[k]) and np.allclose(fb.body[k] - fa.body[k], 0.01, atol=0.01)


def test_dry_run_and_check_through_the_command_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(check_mod, "SampleSpec", lambda *a, **k: SPEC)
    monkeypatch.setenv("REAX_LINEAGE_SEED_B", "sealed-test-seed")
    w = World(tmp_path / "w")
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    lp = w.finish()
    entry = finetune(start, 51, 0.15)
    rec = record("carol/cli", REV(900), "org/start", REV(2))
    write_repo(w.hub, "carol/cli", REV(900), entry, record=rec)
    common = ["--repo", "carol/cli", "--rev", REV(900), "--hotkey", HOTKEY, "--block", "1000", "--commit-time", COMMIT, "--list", lp,
              "--comparison-db", w.dbdir, "--hub-dir", w.hub, "--jobs", "1", "--no-tier-b"]
    code, o = run_cli(capsys, "dry-run", *common, "--out", str(tmp_path / "dry.json"))
    dry = json.load(open(tmp_path / "dry.json"))
    assert code == 0 and o["result"] == "PASS" and dry["dry_run"] is True and dry["sample_B_seed_sha256"] is None and set(dry["weights"]["samples"]) == {"A"}
    code, o = run_cli(capsys, "check", *common, "--out", str(tmp_path / "full.json"))
    full = json.load(open(tmp_path / "full.json"))
    assert o["result"] == "PASS" and set(full["weights"]["samples"]) == {"A", "B"} and full["sample_B_seed_sha256"]
    assert "sealed-test-seed" not in json.dumps(full)
    assert full["check_script_sha256"] == check_mod.script_sha256() and len(full["check_script_sha256"]) == 64
    assert full["blas_gflops"] > 0


def test_local_checkpoint_as_the_entry(tmp_path, capsys, monkeypatch):
    """check --local DIR: the entry is a checkpoint on disk (offline); the record comes from --record."""
    monkeypatch.setattr(check_mod, "SampleSpec", lambda *a, **k: SPEC)
    w = World(tmp_path / "w")
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    lp = w.finish()
    ck = tmp_path / "ckpt"
    write_repo(str(tmp_path / "hubx"), "me/local", REV(901), finetune(start, 52, 0.15), record=None)
    rec = record("me/local", REV(901), "org/start", REV(2))
    rp = tmp_path / "rec.json"
    rp.write_text(json.dumps(rec))
    code, o = run_cli(capsys, "dry-run", "--local", str(tmp_path / "hubx" / "me" / "local" / REV(901)), "--repo", "me/local", "--rev", REV(901),
                      "--hotkey", HOTKEY, "--block", "1000", "--commit-time", COMMIT, "--record", str(rp), "--list", lp, "--comparison-db", w.dbdir,
                      "--hub-dir", w.hub, "--jobs", "1", "--no-tier-b", "--out", str(tmp_path / "o.json"))
    assert code == 0 and o["result"] == "PASS", o


def test_tier_b_unavailable_is_never_a_silent_pass(tmp_path, capsys, monkeypatch):
    """--no-tier-b with a frame mismatch ends as NEEDS_CORRECTION (manual), never PASS."""
    monkeypatch.setattr(check_mod, "SampleSpec", lambda *a, **k: SPEC)
    w = World(tmp_path / "w")
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z")
    lp = w.finish()
    other = finetune(root, 77, 1.0)                                    # not derived from the declared start
    rec = record("zed/wrong", REV(902), "org/start", REV(2))
    write_repo(w.hub, "zed/wrong", REV(902), other, record=rec)
    code, o = run_cli(capsys, "dry-run", "--repo", "zed/wrong", "--rev", REV(902), "--hotkey", HOTKEY, "--block", "1000", "--commit-time", COMMIT,
                      "--list", lp, "--comparison-db", w.dbdir, "--hub-dir", w.hub, "--jobs", "1", "--no-tier-b", "--out", str(tmp_path / "o.json"))
    assert o["result"] == "NEEDS_CORRECTION" and "start_not_confirmed" in o["reasons"]


def test_intake_state_schema_accepts_the_example_and_rejects_bad_rows():
    import copy
    import jsonschema
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema = json.load(open(os.path.join(here, "schema", "intake-state.schema.json")))
    ex = json.load(open(os.path.join(here, "schema", "intake-state.example.json")))
    v = jsonschema.Draft7Validator(schema)
    jsonschema.Draft7Validator.check_schema(schema)
    assert not list(v.iter_errors(ex))
    bad = copy.deepcopy(ex)
    bad["entries"][0]["state"] = "banana"
    assert list(v.iter_errors(bad))
    bad = copy.deepcopy(ex)
    del bad["entries"][1]["correction_deadline"]                    # needs_correction must carry its deadline
    assert list(v.iter_errors(bad))
    bad = copy.deepcopy(ex)
    bad["entries"][0]["preview"] = {"status": "done", "beats_start": True}   # no preview before PASS
    assert list(v.iter_errors(bad))
    bad = copy.deepcopy(ex)
    bad["entries"][2]["preview"] = None                              # pass_unscored always has a preview object
    assert list(v.iter_errors(bad))
    bad = copy.deepcopy(ex)
    bad["entries"][0]["extra"] = 1
    assert list(v.iter_errors(bad))
