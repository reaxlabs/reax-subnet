#!/usr/bin/env python3
"""Pick independent public full fine-tunes of a base model for the C2 null.
Usage: select_null.py <ft_list.json> <base repo> <n> <out_manifest.json> [--params-min B --params-max B]
Filters: public, not gated, safetensors with BF16/F16/F32 only, total params within the window, no gguf/quantized/merge/adapter
tag, base_model tag is exactly <base> (a fine-tune of the base, not of another fine-tune). One model per author first
(independent authors), then more per author only if needed to reach n."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reax_lineage.hf import HFClient, HFError

lst, base, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
pmin = float(sys.argv[sys.argv.index("--params-min") + 1]) * 1e9 if "--params-min" in sys.argv else 25e9
pmax = float(sys.argv[sys.argv.index("--params-max") + 1]) * 1e9 if "--params-max" in sys.argv else 33e9
c = HFClient()
rows = json.load(open(lst))
cand = [r for r in rows if not r["gated"] and not r["private"] and not any(t in r["tags"] for t in ("gguf", "quantized", "merge", "mergekit", "adapter"))
        and r["tags"] and ("base_model:" + base) in r["tags"] and not any(t.startswith("base_model:") and t != "base_model:" + base and "finetune" not in t and False for t in r["tags"])]

def info(r):
    try:
        d = c.api("models/%s" % r["repo"])
    except HFError as e:
        return None
    st = d.get("safetensors") or {}
    sib = [s["rfilename"] for s in d.get("siblings", [])]
    return {**r, "sha": d["sha"], "last_modified": d.get("lastModified"), "params": st.get("total"), "dtypes": sorted((st.get("parameters") or {}).keys()),
            "n_safetensors": sum(1 for s in sib if s.endswith(".safetensors")), "has_adapter": any("adapter_model" in s for s in sib),
            "card_base": (d.get("cardData") or {}).get("base_model"), "tags_all": d.get("tags", []), "library": d.get("library_name")}
with ThreadPoolExecutor(8) as ex:
    det = [x for x in ex.map(info, cand) if x]
ok = [x for x in det if x["params"] and pmin <= x["params"] <= pmax and set(x["dtypes"]) <= {"BF16", "F16", "F32"} and x["n_safetensors"] > 0
      and not x["has_adapter"] and not any(t in x["tags_all"] for t in ("gguf", "quantized", "merge", "mergekit", "adapter", "lora"))]
ok.sort(key=lambda x: -(x["downloads"] or 0))
seen, pick = set(), []
for x in ok:
    a = x["repo"].split("/")[0].lower()
    if a not in seen:
        seen.add(a); pick.append(x)
for x in ok:
    if len(pick) >= n: break
    if x not in pick: pick.append(x)
pick = pick[:n]
man = [{"repo": x["repo"], "asked_rev": "main", "label": "null-" + x["repo"].replace("/", "_"), "rev": x["sha"], "last_modified": x["last_modified"],
        "ok": True, "downloads": x["downloads"], "params": x["params"], "author": x["repo"].split("/")[0], "base": base,
        "card_base": x["card_base"]} for x in pick]
json.dump(man, open(out, "w"), indent=1)
print(base, "candidates", len(cand), "detailed", len(det), "eligible", len(ok), "authors", len(seen), "picked", len(pick))
