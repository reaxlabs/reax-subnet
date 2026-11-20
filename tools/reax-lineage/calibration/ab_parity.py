#!/usr/bin/env python3
"""Sample A vs Sample B (different sealed seed): the same measurements on both samples. Usage: ab_parity.py <cache> <out>"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from reax_lineage.fp import Fingerprint
from reax_lineage.measure import Thresholds, aligned_keys, copy_stats, frame_check, measure_candidate, own_delta
cache, out = sys.argv[1], sys.argv[2]
thr = Thresholds.load()
man = {m["repo"]: m for m in json.load(open(os.path.join(HERE, "c1_manifest.json"))) if m.get("ok")}
def L(repo, sid):
    h = glob.glob(os.path.join(cache, "%s@%s.%s-*.npz" % (repo.replace("/", "__"), man[repo]["rev"], sid)))
    return Fingerprint.load(h[0])
Q, MAM = "Qwen/Qwen3.5-27B", "baselabs/mammoth-1"
M, D, H = "isomsom/sn92-gen-m87b", "xflud/sn92-decider-1", "edinlong83/sn92-generalist-h5"
rows = []
tests = [("ronanlee777/sn92-gen-ma", MAM, Q, [M, D, H]), ("ronanlee777/sn92-gen-mb", MAM, Q, [M, D, H]), ("ronanlee777/sn92-gen-me", MAM, Q, [M, D, H]),
         ("ronanlee777/sn92-gen-mf", MAM, Q, [M, D, H]), ("solanaexpert950105/sn92-gen-md", MAM, Q, [M, D, H]),
         ("gurujustin/sn92-gen-p5", "ronanlee777/sn92-gen-ma", Q, [M, D, "ronanlee777/sn92-gen-ma"]),
         (D, MAM, Q, [M, H]), (M, MAM, Q, [D, H]),
         ("tattrongvu/Jeffan-31b-1710", "google/gemma-4-31B-it", "google/gemma-4-31B", ["PelaAI/KnowLine-31B-Gen1"]),
         ("PelaAI/KnowLine-31B-Gen1", "google/gemma-4-31B-it", "google/gemma-4-31B", ["tattrongvu/Jeffan-31b-1710"])]
for e, s, r, cands in tests:
    for p in cands:
        row = {"entry": e, "cand": p}
        for sid in ("A", "B"):
            C, S, R, P = L(e, sid), L(s, sid), L(r, sid), L(p, sid)
            PS = L(MAM, sid) if (p not in (Q, MAM) and "gemma" not in p.lower() and "Jeffan" not in p and "KnowLine" not in p and p != "ronanlee777/sn92-gen-ma") else (S if "gemma" in s else None)
            if p == "ronanlee777/sn92-gen-ma": PS = L(MAM, sid)
            if "gemma" in s.lower(): PS = S
            keys = aligned_keys(C, S, R, P, *( [PS] if PS is not None else []))
            oc = own_delta(C, S if S.name != R.name else None, R, keys)
            op = own_delta(P, PS, R, keys)
            m = measure_candidate(oc, op, keys, C.n_layers, thr)
            row[sid] = {"beta": m["beta"], "cos": m["cos"], "z": m["z"], "share": m["share"], "detected": m["detected"], "tensors": len(keys)}
            fr = frame_check(C, S, thr)
            row[sid]["frame"] = (fr["frame_share"], fr["scale_share"])
            row[sid]["copy_vs_start"] = copy_stats(C, S, thr)["body"]
        rows.append(row)
d = [abs(r["A"]["beta"] - r["B"]["beta"]) for r in rows]
json.dump({"rows": rows, "max_abs_beta_diff": max(d), "mean_abs_beta_diff": float(np.mean(d))}, open(os.path.join(out, "ab_parity.json"), "w"), indent=1)
print("%-34s %-34s | beta A / B | cos A / B | z A / B | det A/B | copy-vs-start A / B" % ("entry", "candidate"))
for r in rows:
    print("%-34s %-34s | %7.4f %7.4f | %6.3f %6.3f | %7.1f %7.1f | %s/%s | %.4f %.4f" % (r["entry"][:34], r["cand"][:34], r["A"]["beta"], r["B"]["beta"], r["A"]["cos"], r["B"]["cos"], r["A"]["z"], r["B"]["z"], r["A"]["detected"], r["B"]["detected"], r["A"]["copy_vs_start"], r["B"]["copy_vs_start"]))
print("max |beta_A - beta_B| = %.4f, mean %.4f; detection agreement %d/%d" % (max(d), np.mean(d), sum(r["A"]["detected"] == r["B"]["detected"] for r in rows), len(rows)))
