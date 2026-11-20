#!/usr/bin/env python3
"""List public fine-tunes of a base model from the HF model tree (tags base_model:finetune:<base>), newest metadata via API.
Usage: list_finetunes.py <base repo> <out.json> [--limit N]"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reax_lineage.hf import HFClient

base, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 200
c = HFClient()
rows = c.api("models", {"filter": "base_model:finetune:%s" % base, "sort": "downloads", "direction": "-1", "limit": str(limit),
                         "full": "true", "config": "false"})
res = []
for m in rows:
    st = m.get("safetensors") or {}
    res.append({"repo": m["id"], "downloads": m.get("downloads"), "likes": m.get("likes"), "last_modified": m.get("lastModified"),
                "gated": m.get("gated"), "private": m.get("private"), "sha": m.get("sha"),
                "params": st.get("total"), "dtypes": list((st.get("parameters") or {}).keys()),
                "tags": [t for t in m.get("tags", []) if t.startswith("base_model:") or t in ("merge", "mergekit", "adapter", "gguf", "quantized")],
                "pipeline": m.get("pipeline_tag"), "library": m.get("library_name")})
json.dump(res, open(out, "w"), indent=1)
print(base, len(res), "models;", sum(1 for r in res if r["params"] and abs(r["params"] - (res[0]["params"] or 0)) < 1e9), "with a params value")
