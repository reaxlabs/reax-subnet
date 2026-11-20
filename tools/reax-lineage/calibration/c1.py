#!/usr/bin/env python3
"""C1 known pairs (LINEAGE-PROCEDURE.md section 10): every still-public round-1 generalist entry against all other models
of its family. Ground truth = the entries' own NOTICE / merge_report.json (merge weights of the parents).

Usage: c1.py <cache dir> <out dir> [--jobs N]
Writes c1_results.json (every ordered pair), c1_report.md (expectation table), db_c1/ (comparison DB, absolute fp paths)."""
import json, os, sys, time, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from reax_lineage.db import ComparisonDB, mkey
from reax_lineage.fp import Fingerprint
from reax_lineage.measure import Thresholds, aligned_keys, copy_stats, frame_check, grid_shares
from reax_lineage.check import scan_candidates, start_beta

cache, outdir = sys.argv[1], sys.argv[2]
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 16
os.makedirs(outdir, exist_ok=True)
thr = Thresholds.load()

man = {m["repo"]: m for m in json.load(open(os.path.join(HERE, "c1_manifest.json"))) if m.get("ok")}
Q35, MAM = "Qwen/Qwen3.5-27B", "baselabs/mammoth-1"
G_PT, G_IT = "google/gemma-4-31B", "google/gemma-4-31B-it"
FAMILY = {}
START = {}
KIND = {}
for r in man:
    if r in (Q35, "Qwen/Qwen3.8-27B", G_PT):
        KIND[r] = "official"
    elif r in (MAM, G_IT):
        KIND[r] = "list"
    else:
        KIND[r] = "round1" if r not in ("tattrongvu/Jeffan-31b-1710", "PelaAI/KnowLine-31B-Gen1", "sthanika-ai/Sieve-31B-v3", "turion-ai/turion-jev-31b") else "public"
GEMMA = {G_PT, G_IT, "CharlieBT/gx31-decider", "PelaAI/KnowLine-31B-Gen1", "tattrongvu/Jeffan-31b-1710", "sthanika-ai/Sieve-31B-v3",
         "turion-ai/turion-jev-31b", "Poindexter-Labs/Copernicus-31B", "OmniJev/OneJev1.6g-31B"}
Q38 = {"Qwen/Qwen3.8-27B", "tattrongvu/Jeffan-27B-1211", "vllm-sr/d3-ultra"}
for r in man:
    FAMILY[r] = "gemma-4-31b" if r in GEMMA else ("qwen3.8-27b" if r in Q38 else "qwen3.5-27b")
    if r in (Q35, "Qwen/Qwen3.8-27B", G_PT):
        START[r] = None
    elif r == G_IT:
        START[r] = G_PT
    elif r == MAM:
        START[r] = Q35
    elif FAMILY[r] == "gemma-4-31b":
        START[r] = {"CharlieBT/gx31-decider": "tattrongvu/Jeffan-31b-1710", "Poindexter-Labs/Copernicus-31B": "PelaAI/KnowLine-31B-Gen1"}.get(r, G_IT)
    elif FAMILY[r] == "qwen3.8-27b":
        START[r] = "Qwen/Qwen3.8-27B"
    else:
        START[r] = {"gurujustin/sn92-gen-p5": "ronanlee777/sn92-gen-ma"}.get(r, MAM)

# --- ground truth: parents and merge weights (NOTICE / merge_report.json of each entry; see c1_repo_files/) ---
M, D, H, M1B, V3A = "isomsom/sn92-gen-m87b", "xflud/sn92-decider-1", "edinlong83/sn92-generalist-h5", "gurujustin/sn92-gen-m1b", "danyloooah/sn92-generalist-m1-v3a"
TRUTH = {   # entry -> ({parent: weight}, tolerance, kind of recipe)
    "ronanlee777/sn92-gen-ma": ({M: .5, D: .5}, .10, "linear"),
    "ronanlee777/sn92-gen-mb": ({M: .7, D: .7}, .15, "task arithmetic"),
    "ronanlee777/sn92-gen-me": ({M: .3, D: 1.0}, .15, "task arithmetic"),
    "ronanlee777/sn92-gen-mf": ({M: .5, D: 1.0, H: .5}, .15, "task arithmetic"),
    "solanaexpert950105/sn92-gen-md": ({M: .5, D: .5, H: 1.0}, .15, "task arithmetic"),
    "jacksonchen1103/sn92-gen-mc": ({M: .77, D: .75}, .20, "layerwise task arithmetic (mean coefficient)"),
    "danyloooah/sn92-generalist-m1-mdv": ({M: .5, D: .5, V3A: .35}, .15, "NOTICE weights"),
    "danyloooah/sn92-generalist-m1-mx5": ({M: .5, D: .5, M1B: .5, H: .5, V3A: .35}, .15, "NOTICE weights"),
    "gurujustin/sn92-gen-mx2": ({M: .25, D: .25, M1B: .25}, .15, "NOTICE weights"),
    "danyloooah/sn92-generalist-m1-core2": ({M: .7, D: .7, V3A: .25}, .15, "NOTICE weights"),
    "Lucas989/sn92-generalist-m1-core7a": ({M: .7, D: .7}, .15, "NOTICE weights"),
    "Lucas989/sn92-generalist-m1-wide7": ({M: .3, D: .27, M1B: .25, H: .2, V3A: .14}, .15, "NOTICE weights"),
    "mufasa1228/sn92-gen-r3": ({M: .5, D: .5}, .15, "linear + LoRA own training"),
}
PURE = [M, D, H, M1B, V3A]        # plain fine-tunes of mammoth-1 on the public round-1 training data (siblings by data)
GEMMA_PURE = ["tattrongvu/Jeffan-31b-1710", "PelaAI/KnowLine-31B-Gen1", "sthanika-ai/Sieve-31B-v3", "turion-ai/turion-jev-31b", "OmniJev/OneJev1.6g-31B"]

# --- DB ---
db = ComparisonDB(os.path.join(outdir, "db_c1"))
db.index["families"] = {"qwen3.5-27b": {"root": mkey(Q35, man[Q35]["rev"])}, "qwen3.8-27b": {"root": mkey("Qwen/Qwen3.8-27B", man["Qwen/Qwen3.8-27B"]["rev"])},
                        "gemma-4-31b": {"root": mkey(G_PT, man[G_PT]["rev"])}}
missing = []
for r, m in man.items():
    hits = glob.glob(os.path.join(cache, "%s@%s.A-*.npz" % (r.replace("/", "__"), m["rev"])))
    if not hits:
        missing.append(r)
        continue
    st = START[r]
    db.add({"repo": r, "revision": m["rev"], "family": FAMILY[r], "kind": KIND[r], "start": mkey(st, man[st]["rev"]) if st else None,
            "public_at": m.get("last_modified"), "block": None, "uid": None, "holder_ids": ["hf:" + r.split("/")[0].lower()],
            "flagged": False, "result": None, "official": KIND[r] == "official"}, {"A": os.path.abspath(hits[0])})
db.save()
print("db:", len(db.models), "models; missing fingerprints:", missing)

res = {"thresholds": thr.to_dict(), "pairs": {}, "entries": {}, "missing": missing}
t0 = time.time()
for r, m in sorted(man.items()):
    key = mkey(r, m["rev"])
    if key not in db.models or KIND[r] == "official" or r == G_IT or r == MAM:
        continue
    fam = FAMILY[r]
    root = db.root_of(fam)
    S_key = db.models[key]["start"]
    C = db.fingerprint(key, "A", keep=False)
    S = db.fingerprint(S_key, "A") if S_key else None
    R = db.fingerprint(root, "A")
    # candidates: all other models of the family except the start, its ancestors and the root (time order recorded, not applied)
    excl = {key, root, S_key, *db.ancestors(key)} if S_key else {key, root}
    cands = [k for k, mm in db.models.items() if mm["family"] == fam and k not in excl]
    scan = scan_candidates(C, S, R, cands, db, "A", thr, {}, {}, jobs)
    ent = {"family": fam, "start": S_key, "n_candidates": len(scan)}
    if S is not None:
        fr = frame_check(C, S, thr)
        sb = start_beta(C, S, R, db, "A", thr) if S.name != R.name else None
        ent["frame"] = fr
        ent["start_beta"] = None if sb is None else round(sb["beta"], 4)
        cp = copy_stats(C, S, thr)
        ent["copy_vs_start"] = {"body": cp["body"], "head": cp["head"], "is_copy": cp["is_copy"]}
    g = grid_shares(C)
    ent["grid"] = {k: round(v, 5) for k, v in g.items()}
    res["entries"][key] = ent
    for k, s in scan.items():
        pa = db.models[k].get("public_at") or ""
        res["pairs"]["%s <- %s" % (key, k)] = {"entry": key, "cand": k, "beta": round(s.get("beta", 0), 5), "cos": round(s.get("cos", 0), 5),
            "share": round(s.get("share", 0), 4), "null": round(s.get("null", 0), 5), "se": round(s.get("se", 0), 5), "splice": s.get("splice_layers", 0),
            "detected": bool(s.get("detected")), "copy_body": (s.get("copy") or {}).get("body"), "head_source": s.get("head_source"),
            "cand_earlier": pa[:19] < (db.models[key].get("public_at") or "")[:19]}
    print("%-48s %5.1fs  frame=%s start_beta=%s  detected=%s" % (r, time.time() - t0, ent.get("frame", {}).get("ok"), ent.get("start_beta"),
          [(k.split("@")[0].split("/")[1], round(v["beta"], 2)) for k, v in scan.items() if v.get("detected")][:8]), flush=True)
json.dump(res, open(os.path.join(outdir, "c1_results.json"), "w"), indent=1)
print("done", round(time.time() - t0), "s")
