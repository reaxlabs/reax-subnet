#!/usr/bin/env python3
"""Collect the small repo files (README, NOTICE, merge/soup reports) of the C1 entries: the declared recipes (evidence only)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reax_lineage.hf import HFClient, HFSource
c = HFClient()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_repo_files")
os.makedirs(out, exist_ok=True)
man = [m for m in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_manifest.json"))) if m.get("ok")]
for m in man:
    d = os.path.join(out, m["repo"].replace("/", "__"))
    if os.path.exists(d):
        continue
    try:
        s = HFSource(m["repo"], m["rev"], c)
        files = s.list_files()
        os.makedirs(d, exist_ok=True)
        json.dump({p: fi.size for p, fi in files.items()}, open(os.path.join(d, "_files.json"), "w"), indent=0)
        for p, fi in files.items():
            b = os.path.basename(p)
            if fi.size <= 300_000 and (b.lower().startswith(("readme", "notice", "license")) or b in ("merge_report.json", "soup_info.json", "reax-specialist.json", "mergekit_config.yml", "recipe.json", "config.json") or "merge" in b.lower() or "recipe" in b.lower()):
                open(os.path.join(d, b), "wb").write(s.read_file(p))
        print("ok", m["repo"], len(files))
    except Exception as e:
        print("FAIL", m["repo"], e)
