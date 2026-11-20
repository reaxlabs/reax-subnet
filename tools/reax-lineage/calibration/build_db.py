#!/usr/bin/env python3
"""Build a comparison DB (index.json) from fingerprints in the cache.
Usage: build_db.py <db dir> <cache dir> <spec.json>
spec.json: {"families": {fam: {"root": "repo@rev"}}, "models": [{"repo", "rev"(opt; resolved from manifests), "family", "kind", "start": "repo",
            "public_at"(opt)}], "manifests": [manifest.json, ...]}
Fingerprints are found by repo@rev in the cache (seed A); nothing is copied (index paths are absolute)."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reax_lineage.db import ComparisonDB, mkey

dbdir, cache, specf = sys.argv[1:4]
spec = json.load(open(specf))
rev_of, pub_of = {}, {}
for mf in spec.get("manifests", []):
    for m in json.load(open(mf)):
        if m.get("ok"):
            rev_of[m["repo"]] = m["rev"]
            pub_of[m["repo"]] = m.get("last_modified")
db = ComparisonDB(dbdir)
db.index["families"] = spec["families"]
for m in spec["models"]:
    repo = m["repo"]
    rev = m.get("rev") or rev_of.get(repo)
    if not rev:
        continue
    pat = os.path.join(cache, "%s@%s.A-*.npz" % (repo.replace("/", "__"), rev))
    hits = glob.glob(pat)
    if not hits:
        continue
    st = m.get("start")
    meta = {"repo": repo, "revision": rev, "family": m["family"], "kind": m.get("kind", "public"),
            "start": (mkey(st, rev_of[st]) if st else None), "public_at": m.get("public_at") or pub_of.get(repo),
            "block": m.get("block"), "uid": m.get("uid"), "holder_ids": m.get("holder_ids") or ["hf:" + repo.split("/")[0].lower()],
            "flagged": bool(m.get("flagged")), "result": m.get("result"), "official": m.get("kind") == "official"}
    db.add(meta, {"A": os.path.abspath(hits[0])})
db.save()
print("db models:", len(db.models), "->", os.path.join(dbdir, "index.json"))
