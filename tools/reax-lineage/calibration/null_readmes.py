#!/usr/bin/env python3
"""Download README.md of every model in the given manifests (evidence for 'cards declare a relation'). Usage: null_readmes.py out_dir manifest..."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ThreadPoolExecutor
from reax_lineage.hf import HFClient, HFError
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
c = HFClient()
ms = []
for f in sys.argv[2:]:
    ms += [m for m in json.load(open(f)) if m.get("ok")]
def one(m):
    p = os.path.join(out, m["repo"].replace("/", "__") + ".md")
    if os.path.exists(p): return 0
    try:
        open(p, "wb").write(c.read(m["repo"], m["rev"], "README.md")[:400000]); return 1
    except HFError:
        open(p, "wb").write(b""); return 0
with ThreadPoolExecutor(8) as ex:
    print("readmes fetched:", sum(ex.map(one, ms)), "of", len(ms))
