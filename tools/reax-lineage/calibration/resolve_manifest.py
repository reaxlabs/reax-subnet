#!/usr/bin/env python3
"""Resolve a list of 'repo@rev-or-main' strings to full commit shas via the HF API and record availability.
Usage: resolve_manifest.py <in.txt> <out.json>     (one id per line: org/name[@rev] [label])"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reax_lineage.hf import HFClient, HFError

c = HFClient()
out = []
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if not line:
        continue
    parts = line.split()
    rid, label = parts[0], (parts[1] if len(parts) > 1 else "")
    repo, _, rev = rid.partition("@")
    rec = {"repo": repo, "asked_rev": rev or "main", "label": label}
    try:
        try:
            info = c.resolve_revision(repo, rev or "main")
            rec["pinned"] = bool(rev)
        except HFError:
            if not rev:
                raise
            info = c.resolve_revision(repo, "main")      # pin no longer resolves -> HEAD, flagged
            rec["pinned"] = False
            rec["note"] = "pin %s not found, using HEAD" % rev
        rec.update(rev=info["sha"], last_modified=info["last_modified"], gated=info["gated"],
                   base_model=[t for t in info["tags"] if t.startswith("base_model:")], ok=True)
    except Exception as e:
        rec.update(ok=False, error=str(e)[:200])
    out.append(rec)
    print(("OK  " if rec["ok"] else "FAIL"), rec["repo"], rec.get("rev", "")[:8], rec.get("last_modified", ""), rec.get("note", rec.get("error", "")))
json.dump(out, open(sys.argv[2], "w"), indent=1)
