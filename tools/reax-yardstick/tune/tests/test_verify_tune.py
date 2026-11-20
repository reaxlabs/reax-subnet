import csv, gzip, hashlib, io, json, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import verify_tune as V


def make_fixture(d, n=6):
    """Synthetic upstream-shaped rows (no real benchmark content)."""
    lines = []
    for i in range(n):
        row = {"id": f"x{i}", "state": {"task": f"synthetic {i}"}, "questions": {"q": {}}, "expected": {"q": "A"},
               "_evaluation": {"run_id": f"9:SYN:{i}", "catalog_id": "9", "dataset": "SYN", "group_id": f"g{i}",
                               "track": "SYN", "payload_sha256": hashlib.sha256(f"p{i}".encode()).hexdigest()}}
        lines.append(json.dumps(row, separators=(",", ":")).encode())
    raw = b"\n".join(lines) + b"\n"
    with gzip.open(os.path.join(d, "selected-rows.jsonl.gz"), "wb") as f:
        f.write(raw)
    roster = []
    for l in lines:
        e = json.loads(l)["_evaluation"]
        roster.append({"catalog_id": "9", "dataset": "SYN", "track": "SYN", "run_id": e["run_id"], "group_id": e["group_id"],
                       "payload_sha256": e["payload_sha256"], "upstream_file": "selected-rows.jsonl",
                       "upstream_row_sha256": hashlib.sha256(l).hexdigest()})
    return roster, {"selected-rows.jsonl": hashlib.sha256(raw).hexdigest()}


class T(unittest.TestCase):
    def test_fixture_ok_and_extract(self):
        with tempfile.TemporaryDirectory() as d:
            roster, pins = make_fixture(d)
            out = os.path.join(d, "o.jsonl")
            rep = V.verify(d, roster, pins=pins, extract=out)
            self.assertTrue(rep["ok"], rep)
            self.assertEqual((rep["mapped"], rep["unmapped"]), (6, 0))
            self.assertEqual(len(open(out).read().splitlines()), 6)

    def test_detects_unmapped_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            roster, pins = make_fixture(d)
            roster.append(dict(roster[0], run_id="9:SYN:missing"))
            roster[1] = dict(roster[1], payload_sha256="0" * 64)
            rep = V.verify(d, roster, pins=pins)
            self.assertFalse(rep["ok"])
            self.assertEqual(rep["unmapped"], 1)
            self.assertEqual(rep["payload_mismatch"], 1)

    def test_pin_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            roster, pins = make_fixture(d)
            out = os.path.join(d, "invalid.jsonl")
            rep = V.verify(d, roster, pins={"selected-rows.jsonl": "1" * 64}, extract=out)
            self.assertFalse(rep["ok"])
            self.assertFalse(os.path.exists(out))

    def test_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as d:
            roster, _ = make_fixture(d)
            self.assertTrue(V.roster_problems(roster + [roster[0]]))

    def test_shipped_roster_consistent(self):
        rows = V.load_roster(os.path.join(ROOT, "tune-roster.csv"))
        self.assertEqual(V.roster_problems(rows, os.path.join(ROOT, "benchmarks.csv")), [])
        self.assertEqual(len(rows), 10484)
        self.assertEqual(len({r["run_id"] for r in rows}), 10484)
        from collections import Counter
        c = Counter(r["payload_sha256"] for r in rows)
        self.assertEqual((len([k for k, v in c.items() if v > 1]), sum(v for v in c.values() if v > 1)), (20, 43))

    def test_no_forbidden_columns(self):
        with open(os.path.join(ROOT, "tune-roster.csv")) as f:
            hdr = next(csv.reader(f))
        for bad in ("question", "text", "gold", "expected", "split", "seed", "source_path", "response"):
            self.assertNotIn(bad, hdr)

    def test_manifest(self):
        m = os.path.join(ROOT, "MANIFEST.sha256")
        if not os.path.exists(m):
            self.skipTest("manifest not generated yet")
        for line in open(m):
            h, name = line.rstrip("\n").split("  ", 1)
            self.assertEqual(hashlib.sha256(open(os.path.join(ROOT, name), "rb").read()).hexdigest(), h, name)


if __name__ == "__main__":
    unittest.main()
