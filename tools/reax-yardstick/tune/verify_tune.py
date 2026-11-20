#!/usr/bin/env python3
"""Verify (and optionally extract) the Round 2 optional public TUNE development panel
against YOUR OWN copy of the upstream Decision Index 0.3 suite rows.

Stdlib only.  This tool ships no request text; it only checks IDs and hashes in
tune-roster.csv against the rows you rebuilt/downloaded yourself.

  python3 -I verify_tune.py --suite /path/to/suite-0.3 [--extract OUT.jsonl]
  python3 -I verify_tune.py --roster-only          # offline self-consistency of the shipped files

Exit code 0 only if every roster ID is found exactly once with matching payload_sha256
and matching row hash (and, unless --skip-pins, the upstream files match the pinned hashes).
"""
import argparse, csv, gzip, hashlib, json, os, sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
# Upstream decision-index edition 0.3 ("release-v3"): SHA-256 of the UNCOMPRESSED row files
# (values from the upstream package's editions.py; gzip bytes may differ between rebuilds).
PINS_V03 = {
    "selected-rows.jsonl": "b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5",
    "added-rows.jsonl": "7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
    "gsm8k-rows.jsonl": "759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a",
}
EXPECTED_REQUESTS = 10484


def load_roster(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def roster_problems(rows, benchmarks_path=None):
    """Offline checks: counts, duplicates, per-benchmark totals."""
    probs = []
    ids = [r["run_id"] for r in rows]
    if len(ids) != len(set(ids)):
        probs.append("duplicate run_id in roster")
    # NB: run_id is the key. A few distinct run_ids legitimately share a payload_sha256
    # (20 hashes / 43 rows in the shipped roster: BPoMP, CLadder, SGD) - informational only.
    if len({(r["run_id"], r["group_id"]) for r in rows}) != len(rows):
        probs.append("duplicate (run_id, group_id) in roster")
    if len(rows) != EXPECTED_REQUESTS:
        probs.append(f"roster has {len(rows)} rows, expected {EXPECTED_REQUESTS}")
    if benchmarks_path and os.path.exists(benchmarks_path):
        tot = {}
        for r in rows:
            tot[r["catalog_id"]] = tot.get(r["catalog_id"], 0) + 1
        with open(benchmarks_path, newline="", encoding="utf-8") as f:
            for b in csv.DictReader(f):
                if tot.get(b["catalog_id"], 0) != int(b["requests"]):
                    probs.append(f"benchmark {b['catalog_id']}: roster {tot.get(b['catalog_id'], 0)} != benchmarks.csv {b['requests']}")
    return probs


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _file_sha(path):
    h = hashlib.sha256()
    with _open(path) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(suite_dir, roster, pins=PINS_V03, skip_pins=False, extract=None):
    rep = {"pins_checked": not skip_pins, "pin_results": {}, "mapped": 0, "unmapped": 0,
           "duplicates_upstream": 0, "payload_mismatch": 0, "row_hash_mismatch": 0, "source_file_mismatch": 0,
           "ok": False}
    if extract and skip_pins:
        raise ValueError("extraction requires matching upstream file pins")
    target = Path(extract) if extract else None
    work = tempfile.TemporaryDirectory(dir=target.parent) if target else None
    staged = Path(work.name) / "verified.jsonl" if work else None
    want = {r["run_id"]: r for r in roster}
    seen = {}
    out = staged.open("wb") if staged else None
    try:
        for name, pin in pins.items():
            path = os.path.join(suite_dir, name + ".gz")
            if not os.path.exists(path):
                path = os.path.join(suite_dir, name)
            if not os.path.exists(path):
                rep["pin_results"][name] = "missing"
                continue
            if not skip_pins:
                rep["pin_results"][name] = "match" if _file_sha(path) == pin else "MISMATCH"
            with _open(path) as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    line = raw.rstrip(b"\r\n")
                    ev = json.loads(line)["_evaluation"]
                    rid = ev["run_id"]
                    if rid not in want:
                        continue
                    if rid in seen:
                        rep["duplicates_upstream"] += 1
                        continue
                    r = want[rid]
                    seen[rid] = True
                    if name != r["upstream_file"]:
                        rep["source_file_mismatch"] += 1
                    if ev.get("payload_sha256") != r["payload_sha256"]:
                        rep["payload_mismatch"] += 1
                    if hashlib.sha256(line).hexdigest() != r["upstream_row_sha256"]:
                        rep["row_hash_mismatch"] += 1
                    if out:
                        out.write(line + b"\n")
        rep["mapped"] = len(seen)
        rep["unmapped"] = len(want) - len(seen)
        pins_ok = skip_pins or (pins and all(v == "match" for v in rep["pin_results"].values()))
        rep["ok"] = bool(pins_ok and rep["unmapped"] == 0 and rep["duplicates_upstream"] == 0
                         and rep["payload_mismatch"] == 0 and rep["row_hash_mismatch"] == 0
                         and rep["source_file_mismatch"] == 0)
        if out:
            out.close()
            if rep["ok"]:
                os.link(staged, target)
    finally:
        if out and not out.closed:
            out.close()
        if work:
            work.cleanup()
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", help="directory holding your upstream suite-0.3 row files")
    ap.add_argument("--roster", default=os.path.join(HERE, "tune-roster.csv"))
    ap.add_argument("--roster-only", action="store_true")
    ap.add_argument("--skip-pins", action="store_true", help="do not check upstream file hashes (result is then UNVERIFIED)")
    ap.add_argument("--extract", help="write the matching upstream rows (verbatim) to this file")
    a = ap.parse_args(argv)
    roster = load_roster(a.roster)
    probs = roster_problems(roster, os.path.join(os.path.dirname(os.path.abspath(a.roster)), "benchmarks.csv"))
    for p in probs:
        print("ROSTER PROBLEM:", p)
    if a.extract and (not a.suite or a.roster_only or a.skip_pins or probs):
        ap.error("--extract requires a validated roster and a pinned upstream suite")
    if a.extract and Path(a.extract).exists():
        ap.error("--extract destination already exists")
    if a.roster_only or not a.suite:
        print(f"roster rows={len(roster)} problems={len(probs)}")
        return 1 if probs else 0
    rep = verify(a.suite, roster, skip_pins=a.skip_pins, extract=a.extract)
    print(json.dumps(rep, indent=2))
    if a.extract and rep["ok"]:
        print("NOTE: extracted rows are upstream mixed-licence content for EVALUATION ONLY. Do not train on them.")
    return 0 if rep["ok"] and not probs else 1


if __name__ == "__main__":
    sys.exit(main())
