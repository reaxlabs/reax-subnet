#!/usr/bin/env python3
"""CPU tests (stdlib only): item hashes, loaders, scoring pins, and an end-to-end run against the mock server.

  python3 -m unittest discover -s tests -v        (from tools/reax-yardstick/)
"""
import ast
import hashlib
import json
import math
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reax_yardstick import items as itemlib, scoring  # noqa: E402
from reax_yardstick.common import content_sha256, sha256_file, sha256_text, canonical_json  # noqa: E402
from reax_yardstick.readout import render_prompt, render_prompt_aa  # noqa: E402

# sha256 of each scoring function's source, identical to the REAX preview scorer (see README "Same scoring").
SCORING_PINS = {
    "question_correct": "40a0ed6296d401ec1e993a20df303e2d3a28634b446d6abb454d30f858c6df84",
    "item_score": "9e5b00c9c7f79057321b5c1f06044e4b9f8331f45a3fae49b939ae05be558597",
    "score_journal": "7920526bb3a95c5666582eae6ecc5c009412bcc8697f83ef15d04097a42b7f64",
    "mean": "ae0b9cb71bee6c566a33f7c87fe3f9c54c57337d0dabf4c415257ad36fcae6d1",
    "sd": "ced13feba380ac02d1bdd9e53573a78b69ddf87662857ca4b4a42d78eaa8ceed",
    "family_summary": "efa0d1ce2b89f874b04f53a2b8522023c33475834b7b6e2dc1e983b8029647cc",
    "weighted_aggregate": "2c010e043375d766c08474b1804ead9ba7ce18be311c0fb7ed66776d32b62b01",
}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestScoring(unittest.TestCase):
    def test_functions_pinned(self):
        src = (ROOT / "reax_yardstick" / "scoring.py").read_text()
        tree = ast.parse(src)
        got = {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode()).hexdigest() for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, pin in SCORING_PINS.items():
            self.assertEqual(got.get(name), pin, name)

    def test_question_rules(self):
        self.assertTrue(scoring.question_correct("choice", {"choice": "B"}, "B"))
        self.assertFalse(scoring.question_correct("choice", {"choice": "A"}, "B"))
        self.assertTrue(scoring.question_correct("score", {"score": "3"}, 3))
        self.assertTrue(scoring.question_correct("noul", {"noul": 0.5}, True))
        self.assertFalse(scoring.question_correct("noul", {"noul": 0.49}, True))
        self.assertIsNone(scoring.question_correct("noul", {"noul": 0.9}, None))

    def test_aggregate(self):
        fam = {"PV-HF": {"mean": 50.0, "se": 2.0, "n": 10}, "PV-W2C": {"mean": 80.0, "se": 1.0, "n": 10}}
        agg = scoring.weighted_aggregate(fam, itemlib.WEIGHTS)
        self.assertAlmostEqual(agg["mean"], (0.4 * 50 + 0.2 * 80) / 0.6)
        self.assertAlmostEqual(agg["se"], math.sqrt((0.4 / 0.6 * 2) ** 2 + (0.2 / 0.6 * 1) ** 2))


class TestPrompts(unittest.TestCase):
    def test_render(self):
        q = {"type": "choice", "instructions": "Pick", "criteria": {"A": "yes", "B": "no"}}
        self.assertEqual(render_prompt_aa("S", q), "State:\nS\n\nQuestion:\nPick\n\nOptions:\nA: yes\nB: no\n\nReply with exactly one option key.")
        self.assertEqual(render_prompt({}, {"type": "noul", "instructions": "Ok?"}),
                         "State:\n(empty)\n\nQuestion:\nOk?\n\nOptions:\nfalse: False\ntrue: True\n\nReply with exactly one option key.")


class TestItems(unittest.TestCase):
    def test_manifest_files(self):
        m = itemlib.manifest()
        body = {k: v for k, v in m.items() if k != "manifest_sha256"}
        self.assertEqual(sha256_text(canonical_json(body)), m["manifest_sha256"])
        for fam, spec in m["families"].items():
            for name, h in spec["files"].items():
                self.assertEqual(sha256_file(itemlib.ITEMS_DIR / spec["dir"] / name), h, f"{fam}/{name}")
            self.assertEqual(len(itemlib.load_index(fam)), spec["n"])

    def test_text_families_verify(self):
        for fam in ("PV-W2C", "PV-ONET"):
            pairs, rep = itemlib.load_family(fam)
            self.assertEqual(rep["loaded"], rep["indexed"], fam)
            self.assertEqual(rep["hash_mismatch"], [], fam)

    def _items_dir(self, tmp, fam, index_rows):
        d = Path(tmp) / "items"
        (d / itemlib.FAMILIES[fam]["dir"]).mkdir(parents=True)
        (d / itemlib.FAMILIES[fam]["dir"] / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index_rows))
        return d

    def test_di_loader(self):
        rows = [{"id": "c1", "state": {"text": "a"}, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"A": "1", "B": "2"}}},
                 "expected": {"q": "B"}, "_evaluation": {"run_id": "1:T:T:test:c1"}},
                {"id": "c2", "state": "b", "questions": {"q": {"type": "noul", "instructions": "y"}}, "expected": {"q": True},
                 "_evaluation": {"run_id": "2:T:T:test:c2"}}]
        idx = [{"id": "1:T:T:test:c1", "content_sha256": content_sha256(rows[0]["state"], rows[0]["questions"], rows[0]["expected"])},
               {"id": "v0.3:2:T:T:test:c2", "source_run_id": "2:T:T:test:c2", "content_sha256": content_sha256(rows[1]["state"], rows[1]["questions"], rows[1]["expected"])},
               {"id": "9:T:T:test:absent", "content_sha256": "0" * 64}]
        with tempfile.TemporaryDirectory() as tmp:
            d = self._items_dir(tmp, "PV-HF", idx)
            suite = Path(tmp) / "suite.jsonl"
            suite.write_text("".join(json.dumps(r) + "\n" for r in rows))
            pairs, rep = itemlib.load_family("PV-HF", di_suite=[str(suite)], items_dir=d)
            self.assertEqual(rep["loaded"], 2)
            self.assertEqual(rep["missing"], ["9:T:T:test:absent"])
            self.assertEqual({p[0]["_evaluation"]["run_id"] for p in pairs}, {"1:T:T:test:c1", "v0.3:2:T:T:test:c2"})
            rows[0]["expected"] = {"q": "A"}  # tampered gold -> refused
            suite.write_text("".join(json.dumps(r) + "\n" for r in rows))
            pairs, rep = itemlib.load_family("PV-HF", di_suite=[str(suite)], items_dir=d)
            self.assertEqual(rep["hash_mismatch"], ["1:T:T:test:c1"])
            rows[0]["expected"] = {"q": "B"}
            rows[0]["questions"]["q"]["criteria"] = {"B": "2", "A": "1"}  # same options, other order -> other prompt -> refused
            suite.write_text("".join(json.dumps(r) + "\n" for r in rows))
            _, rep = itemlib.load_family("PV-HF", di_suite=[str(suite)], items_dir=d)
            self.assertEqual(rep["hash_mismatch"], ["1:T:T:test:c1"])
            _, rep = itemlib.load_family("PV-HF", items_dir=d)
            self.assertIn("no --di-suite", rep["note"])

    def test_hle_loader(self):
        hle = {"id": "abc123", "question": "Which is right?\n\nAnswer Choices:\nA. one\nB. two\nlines\nC. three", "answer": "B", "answer_type": "multipleChoice"}
        state = "Which is right?" + itemlib.HLE_SUFFIX
        crit = {"A": "one", "B": "two\nlines", "C": "three", "D": "I don't know"}
        q = {"answer": {"type": "choice", "instructions": itemlib.HLE_INSTRUCTIONS, "criteria": crit}}
        idx = [{"id": "hle/abc123", "source_id": "abc123", "decline_key": "D", "decline_text": "I don't know", "n_options": 4,
                "content_sha256": content_sha256(state, q, {"answer": "B"})}]
        with tempfile.TemporaryDirectory() as tmp:
            d = self._items_dir(tmp, "PV-HLE", idx)
            src = Path(tmp) / "hle.jsonl"
            src.write_text(json.dumps(hle) + "\n")
            pairs, rep = itemlib.load_family("PV-HLE", hle_source=str(src), items_dir=d)
            self.assertEqual(rep["loaded"], 1, rep)
            self.assertEqual(pairs[0][0]["questions"], q)
            hle["answer"] = "A"
            src.write_text(json.dumps(hle) + "\n")
            _, rep = itemlib.load_family("PV-HLE", hle_source=str(src), items_dir=d)
            self.assertEqual(rep["hash_mismatch"], ["hle/abc123"])


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="yardstick-test-"))
        cls.port = free_port()
        items = itemlib.ITEMS_DIR
        args = [sys.executable, "-I", "-B", str(ROOT / "tests" / "mock_server.py"), "--port", str(cls.port), "--skill", "0.6", "--seed", "t"]
        for fam in ("when2call", "onet"):
            args += ["--rows", str(items / fam / "requests.jsonl.gz"), "--gold", str(items / fam / "gold.jsonl.gz")]
        cls.srv = subprocess.Popen(args, start_new_session=True)
        for _ in range(200):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/health", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.srv.terminate()
        cls.srv.wait()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_cli(self, *extra):
        cmd = [sys.executable, "-m", "reax_yardstick", "run", "--model", "mock/model@0123456789abcdef0123456789abcdef01234567",
               "--engine-url", f"http://127.0.0.1:{self.port}", "--families", "PV-W2C,PV-ONET", "--limit", "80",
               "--concurrency", "8", "--out", str(self.tmp / "run"), *extra]
        return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    def test_run_and_resume(self):
        p = self.run_cli()
        self.assertIn(p.returncode, (0, 3), p.stderr[-2000:])  # 3 = incomplete (two families on purpose)
        out = json.loads((self.tmp / "run" / "yardstick.json").read_text())
        for fam in ("PV-W2C", "PV-ONET"):
            self.assertEqual(out["families"][fam]["n"], 80)
            self.assertEqual(out["quality"][fam]["n_error"], 0)
            self.assertTrue(30 < out["families"][fam]["mean"] < 95, out["families"][fam])
        w = out["weights_used"]
        self.assertAlmostEqual(w["PV-W2C"], 0.5)
        self.assertAlmostEqual(out["composite"], 0.5 * out["families"]["PV-W2C"]["mean"] + 0.5 * out["families"]["PV-ONET"]["mean"])
        self.assertFalse(out["complete"])
        n1 = len((self.tmp / "run" / "PV-W2C" / "results.jsonl").read_text().splitlines())
        p2 = self.run_cli()
        n2 = len((self.tmp / "run" / "PV-W2C" / "results.jsonl").read_text().splitlines())
        self.assertEqual(n1, n2, "resume must not ask finished items again")
        out2 = json.loads((self.tmp / "run" / "yardstick.json").read_text())
        self.assertEqual(out["composite"], out2["composite"])
        self.assertIn("Composite", (self.tmp / "run" / "summary.md").read_text())
        p3 = self.run_cli("--model", "mock/different@0123456789abcdef0123456789abcdef01234567")
        self.assertEqual(p3.returncode, 1)
        self.assertIn("use a new --out directory", p3.stderr)
        self.assertEqual(out2, json.loads((self.tmp / "run" / "yardstick.json").read_text()))


if __name__ == "__main__":
    unittest.main()
