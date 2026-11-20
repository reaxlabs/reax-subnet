#!/usr/bin/env python3
"""C2 public null: independent public fine-tunes of one base, all ordered pairs.

Usage: c2.py <cache dir> <out dir> --family NAME --root REPO --start REPO --manifest M.json [--jobs N] [--readmes DIR]
  root  = official pretrained base R; start = the model all of them are declared fine-tunes of (R itself for Qwen, gemma-it for Gemma)
Pairs whose cards declare a relation (one README names the other's repo id) are dropped. Writes c2_<family>.json."""
import argparse, glob, json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from reax_lineage.db import ComparisonDB, mkey
from reax_lineage.measure import Thresholds, frame_check, copy_stats, grid_shares
from reax_lineage.check import scan_candidates

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("out")
ap.add_argument("--family", required=True); ap.add_argument("--root", required=True); ap.add_argument("--start", required=True)
ap.add_argument("--manifest", required=True, nargs="+"); ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--readmes")
a = ap.parse_args()
thr = Thresholds.load()
c1 = {m["repo"]: m for m in json.load(open(os.path.join(HERE, "c1_manifest.json"))) if m.get("ok")}
null = []
for mf in a.manifest:
    null += [m for m in json.load(open(mf)) if m.get("ok")]
seen, models = set(), []
for m in null:
    if m["repo"] not in seen:
        seen.add(m["repo"]); models.append(m)
db = ComparisonDB(os.path.join(a.out, "db_" + a.family))
def fpath(repo, rev):
    h = glob.glob(os.path.join(a.cache, "%s@%s.A-*.npz" % (repo.replace("/", "__"), rev)))
    return os.path.abspath(h[0]) if h else None
root_rev, start_rev = c1[a.root]["rev"], c1[a.start]["rev"]
db.index["families"][a.family] = {"root": mkey(a.root, root_rev)}
db.add({"repo": a.root, "revision": root_rev, "family": a.family, "kind": "official", "start": None, "official": True}, {"A": fpath(a.root, root_rev)})
if a.start != a.root:
    db.add({"repo": a.start, "revision": start_rev, "family": a.family, "kind": "official", "start": mkey(a.root, root_rev), "official": True}, {"A": fpath(a.start, start_rev)})
used = []
for m in models:
    p = fpath(m["repo"], m["rev"])
    if not p or m["repo"] in (a.root, a.start):
        continue
    db.add({"repo": m["repo"], "revision": m["rev"], "family": a.family, "kind": "public", "start": mkey(a.start, start_rev),
            "public_at": m.get("last_modified"), "holder_ids": ["hf:" + m["repo"].split("/")[0].lower()]}, {"A": p})
    used.append(m)
db.save()
print(a.family, "models with fingerprints:", len(used))
# README relations
readme = {}
if a.readmes:
    for m in used:
        f = os.path.join(a.readmes, m["repo"].replace("/", "__") + ".md")
        readme[m["repo"]] = open(f, errors="replace").read().lower() if os.path.exists(f) else ""
def related(x, y):
    return (y.lower() in readme.get(x, "")) or (x.lower() in readme.get(y, ""))
R = db.fingerprint(mkey(a.root, root_rev), "A")
S = db.fingerprint(mkey(a.start, start_rev), "A") if a.start != a.root else None
pairs, ents = [], {}
t0 = time.time()
keys = [mkey(m["repo"], m["rev"]) for m in used]
for m in used:
    key = mkey(m["repo"], m["rev"])
    C = db.fingerprint(key, "A", keep=False)
    fr = frame_check(C, S if S is not None else R, thr)
    ents[m["repo"]] = {"frame_ok": fr["ok"], "frame_share": fr["frame_share"], "scale_share": fr["scale_share"], "min_cos": fr["min_cos"],
                       "copy_vs_start": copy_stats(C, S if S is not None else R, thr)["body"], "grid_max": grid_shares(C)["max"]}
    cands = [k for k in keys if k != key]
    scan = scan_candidates(C, S, R, cands, db, "A", thr, {}, {}, a.jobs)
    for k, s in scan.items():
        r = k.split("@")[0]
        pairs.append({"entry": m["repo"], "cand": r, "beta": s.get("beta"), "cos": s.get("cos"), "share": s.get("share"), "null": s.get("null"),
                      "se": s.get("se"), "q10": s.get("beta_q10"), "layers_ge_05": s.get("layers_ge_05"), "splice": s.get("splice_layers"), "detected": bool(s.get("detected")), "copy": (s.get("copy") or {}).get("body"),
                      "same_author": m["repo"].split("/")[0].lower() == r.split("/")[0].lower(), "declared_relation": related(m["repo"], r),
                      "cov": s.get("coverage")})
    print("%-60s %5.0fs" % (m["repo"], time.time() - t0), flush=True)
json.dump({"family": a.family, "root": a.root, "start": a.start, "n_models": len(used), "entries": ents, "pairs": pairs,
           "thresholds": thr.to_dict()}, open(os.path.join(a.out, "c2_%s.json" % a.family), "w"), indent=0)
print("pairs:", len(pairs))
