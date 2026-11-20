#!/usr/bin/env python3
"""Fingerprint every model of a manifest (one subprocess per model, so peak RSS is per model), resumable.
Usage: fetch_set.py <manifest.json> --cache DIR --budget SECONDS [--seed-id A|B] [--runtime-log FILE] [--only label,label]
Exit 0 = all done, 3 = budget used up (call again), 1 = some failed."""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from reax_lineage.fp import cache_path, seed_sha256, SEED_A_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("--cache", required=True); ap.add_argument("--budget", type=float, default=540)
ap.add_argument("--seed-id", default="A"); ap.add_argument("--runtime-log"); ap.add_argument("--only")
a = ap.parse_args()
seed = os.environ.get("REAX_LINEAGE_SEED_B") if a.seed_id == "B" else (os.environ.get("REAX_LINEAGE_SEED_A") or SEED_A_DEFAULT)
t0 = time.time()
man = [m for m in json.load(open(a.manifest)) if m.get("ok")]
if a.only:
    keep = set(a.only.split(","))
    man = [m for m in man if m["label"] in keep]
todo, failed = 0, 0
for m in man:
    p = cache_path(a.cache, m["repo"], m["rev"], a.seed_id, seed_sha256(seed))
    if os.path.exists(p):
        continue
    if time.time() - t0 > a.budget:
        todo += 1
        continue
    cmd = [sys.executable, "-m", "reax_lineage", "fingerprint", "--repo", m["repo"], "--rev", m["rev"], "--seed-id", a.seed_id,
           "--cache", a.cache]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(HERE, ".."), env={**os.environ, "PYTHONPATH": os.path.join(HERE, "..")})
    wall = round(time.time() - t, 1)
    if r.returncode != 0:
        failed += 1
        print("FAIL", m["repo"], r.stderr.strip()[-200:], flush=True)
        continue
    info = json.loads(r.stdout.strip().splitlines()[-1])
    info.update(label=m["label"], wall_s=wall)
    print("OK  %-52s %6.1fs rss=%sMB tensors=%s" % (m["repo"], wall, info["peak_rss_mb"], info["n_tensors"]), flush=True)
    if a.runtime_log:
        open(a.runtime_log, "a").write(json.dumps(info) + "\n")
print("remaining:", todo, "failed:", failed, "elapsed", round(time.time() - t0))
sys.exit(3 if todo else (1 if failed else 0))
