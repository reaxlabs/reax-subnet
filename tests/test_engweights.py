"""Option A inventory screen and title transition; no image code is executed."""
import copy
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from reax_subnet import competition_split as cs
from reax_subnet.engcomp import (Champion, EvalResult, Submission, TrackParams, TrackState,
                                title_eligible, update_champion)
from reax_subnet.engreview import (MAX_CHECKED_FILE_BYTES, REFERENCE_TOKENIZER_FILES, WEIGHT_FILE_SUFFIXES,
                                  inventory_files, load_inventory, weight_file_screen)

D, E = "a" * 64, "b" * 64
END = cs.ENGINE_QUANT_CONVERSION_END_UTC


def row(path, size=10, **kwargs):
    return {"path": path, "size": size, "sha256": D, "is_text": False, **kwargs}


@pytest.mark.parametrize("suffix", sorted(WEIGHT_FILE_SUFFIXES))
def test_extensions_anywhere_even_small_text(suffix):
    assert "extension" in weight_file_screen([row("/unusual/path/model" + suffix, 0, is_text=True)])[0]["reason"]
    assert weight_file_screen([row("/MODEL" + suffix.upper())])


def test_exact_tokenizer_manifest_names():
    manifest = json.loads((Path(__file__).parents[1] / "src/reax_subnet/manifests/s1-fast-v1.json").read_text())
    assert REFERENCE_TOKENIZER_FILES == {n for n in manifest["weights"]["files"] if n.startswith("tokenizer")}
    for name in REFERENCE_TOKENIZER_FILES:
        from reax_subnet.engreview import REFERENCE_TOKENIZER_SIZE, REFERENCE_TOKENIZER_SHA256
        assert not weight_file_screen([row("/model/" + name, REFERENCE_TOKENIZER_SIZE, sha256=REFERENCE_TOKENIZER_SHA256)])
    for fake in ("tokenizer.bin", "tokenizer.pt", "vocab.npy"):
        assert weight_file_screen([row("/" + fake)])


def test_size_boundary_and_text_is_not_code():
    assert not weight_file_screen([row("/opaque", MAX_CHECKED_FILE_BYTES)])
    findings = weight_file_screen([row("/opaque", MAX_CHECKED_FILE_BYTES + 1, is_text=True)])
    assert "64 MiB" in findings[0]["reason"]
    assert "64 MiB" in weight_file_screen([row("/weights.bin", MAX_CHECKED_FILE_BYTES + 1)])[0]["reason"]


@pytest.mark.parametrize("suffix", (".py", ".so", ".so.1.2", ".cu", ".cubin", ".ptx", ".c", ".cc", ".cpp",
                                   ".h", ".txt", ".md", ".toml", ".yaml", ".sh"))
def test_code_exempt_from_size_rule(suffix):
    assert not weight_file_screen([row("/app/code" + suffix, MAX_CHECKED_FILE_BYTES + 1)])


def test_layers_preserve_overwritten_weights_and_do_not_mutate():
    data = {"7": {"/app/x.pt": {"size": 30, "sha256": D}}, "8": {"/app/x.pt": {"size": 0, "sha256": E}}}
    before = copy.deepcopy(data)
    assert len(weight_file_screen(data)) == 2
    assert data == before
    assert len(inventory_files([row("/x")])) == 1


@pytest.mark.parametrize("data", (None, {"x": 4}, [row("x", -1)], [row("x", True)], [{"path": "x"}], [3]))
def test_bad_inventory_rejected(data):
    with pytest.raises(ValueError):
        weight_file_screen(data)


def test_cli_json_tsv_and_failure(tmp_path):
    p = tmp_path / "files.json"
    cli_env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")
               + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p.write_text(json.dumps({"7": {"/a.pt": {"size": 5, "sha256": D}}}))
    assert load_inventory(str(p))[0]["layer"] == "7"
    result = subprocess.run([sys.executable, "-m", "reax_subnet.engreview", "weights", "--inventory", str(p)],
                            capture_output=True, text=True, env=cli_env)
    assert result.returncode == 1 and not json.loads(result.stdout)["compliant"]
    p.write_text(f"{D}\t5\t/a.py\nSYMLINK\t0\t/link -> /a.py\n")
    assert len(load_inventory(str(p))) == 1
    result = subprocess.run([sys.executable, "-m", "reax_subnet.engreview", "weights", "--inventory", str(p)],
                            capture_output=True, text=True, env=cli_env)
    assert result.returncode == 0 and json.loads(result.stdout)["compliant"]
    p.write_text("bad inventory")
    result = subprocess.run([sys.executable, "-m", "reax_subnet.engreview", "weights", "--inventory", str(p)],
                            capture_output=True, text=True, env=cli_env)
    assert result.returncode == 2


def test_title_transition_and_json_roundtrip():
    state = TrackState()
    assert title_eligible(state, D, END - timedelta(microseconds=1))
    assert not title_eligible(state, D, END)
    assert not title_eligible(state, D, END + timedelta(days=1))
    state.weights_ok.add(D)
    state = TrackState.from_json(json.loads(json.dumps(state.to_json())))
    assert state.weights_ok == {D} and title_eligible(state, D, END)
    assert not title_eligible(state, E, END)
    assert TrackState.from_json({}).weights_ok == set()
    for value in (D, {D: True}, True, ["bad", None, 3]):
        assert not title_eligible(TrackState.from_json({"weights_ok": value}), D, END)


def test_champion_selection_and_pending_across_deadline():
    params = TrackParams("s1-fast", pool="s1-fast", temperature=2.07, confirmations=2)
    sub = Submission(5, "hk", "s1-fast", "ghcr.io/a/e", D)
    state = TrackState(first_seen={D: (1, "hk")})
    state.results[D] = EvalResult(digest=D, track="s1-fast", ok=True, score=2.0, block=1, session="one")
    update_champion(state, [sub], params, 1, now=END - timedelta(seconds=1))
    assert state.challenger is not None and state.champion is None
    state.results[D].session = "two"
    update_champion(state, [sub], params, 2, now=END)
    assert state.challenger is None and state.champion is None
    state.weights_ok.add(D)
    update_champion(state, [sub], params, 3, now=END)
    state.results[D].session = "three"
    update_champion(state, [sub], params, 4, now=END)
    assert state.champion.digest == D
    # A title that already exists is retained even without a compliance marker.
    state.weights_ok.clear()
    update_champion(state, [sub], params, 5, now=END)
    assert state.champion.digest == D


def test_header_inventory_from_early_reviews(tmp_path):
    p = tmp_path / "filelist.tsv"
    p.write_text("layer\tpath\tsize\tsha256\ttype\tsame_path_in_ref\n"
                 f"layer06\t/app/a.pt\t16\t{D}\tdata\tno\n"
                 "layer06\t/link\tLINK\t/a.py\t\tno\n"
                 "layer07\t/.wh.old\tWHITEOUT\t\t\tno\n")
    findings = weight_file_screen(load_inventory(str(p)))
    assert findings[0]["layer"] == "layer06" and findings[0]["size"] == 16
