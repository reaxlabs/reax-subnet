#!/usr/bin/env python3
"""Re-evaluate the stored raw metrics of C1 / C2 / C3 under rule variants. Usage: sweep.py <calib_out> -> sweep.json + printed summary.
Rule V0 = LINEAGE-PROCEDURE.md 6.2 as proposed (beta>=.05, cos>=.02, share>=.8, beta>=5 null; or splice>=2).
Rule V1 = calibrated (beta>=.05, cos>=.01, z>=15, share>=.8, beta>=5 null; or splice>=2 with >=4 tensors per layer). Level 2 (action) = V1 and beta>=.25 (or splice)."""
import json, os, re, sys
import numpy as np
out = sys.argv[1]
def v0(m): return (m["beta"] >= .05 and m["cos"] >= .02 and m["share"] >= .8 and m["beta"] >= 5 * (m.get("null") or 0)) or m["splice"] >= 2
def v1(m):
    z = m["beta"] / m["se"] if m.get("se") else (1e9 if m["beta"] > 0 else 0)
    return (m["beta"] >= .05 and m["cos"] >= .01 and z >= 15 and m["share"] >= .8 and m["beta"] >= 5 * (m.get("null") or 0)) or m["splice"] >= 2
def v1a(m): return v1(m) and (m["beta"] >= .25 or m["splice"] >= 2)
EDIT = re.compile(r"uncensor|abliterat|heretic|derestrict|decensor|unfilter", re.I)
res = {}
# ---- C2
c2 = {}
for fam in ("gemma-4-31b", "qwen3.5-27b", "qwen3.8-27b"):
    d = json.load(open(os.path.join(out, "c2_%s.json" % fam)))
    bad = {r for r, e in d["entries"].items() if not e["frame_ok"]}
    allp = [{**p, "splice": p.get("splice") or 0, "null": p.get("null") or 0} for p in d["pairs"] if p["beta"] is not None]
    kept0 = [p for p in allp if not p["declared_relation"] and p["entry"] not in bad and p["cand"] not in bad]
    edit = [p for p in kept0 if EDIT.search(p["entry"]) and EDIT.search(p["cand"])]
    kept = [p for p in kept0 if not (EDIT.search(p["entry"]) and EDIT.search(p["cand"]))]
    def cnt(ps, f): return sum(1 for p in ps if f(p))
    z = np.array([(p["beta"] / p["se"]) if p.get("se") else 0 for p in kept])
    c = np.array([p["cos"] for p in kept]); b = np.array([p["beta"] for p in kept]); sh = np.array([p["share"] for p in kept])
    c2[fam] = {"models": d["n_models"], "pairs_all": len(allp), "dropped_declared": sum(p["declared_relation"] for p in allp), "frame_failures": sorted(bad),
               "pairs_independent": len(kept), "same_recipe_edit_pairs": len(edit),
               "indep": {"V0": cnt(kept, v0), "V1": cnt(kept, v1), "V1_action": cnt(kept, v1a)},
               "edit": {"V0": cnt(edit, v0), "V1": cnt(edit, v1), "V1_action": cnt(edit, v1a)},
               "cos_max": float(c.max()), "cos_p999": float(np.quantile(c, .999)), "beta_p999": float(np.quantile(b, .999)), "z_max": float(z.max()), "z_p999": float(np.quantile(z, .999)),
               "share_max": float(sh.max()), "beta_max_noncopy": float(b[c < .5].max())}
    # gray zone = pairs with cos < 0.2 (no copy-like evidence)
    gz = [p for p in kept if p["cos"] < .2]
    zg = np.array([(p["beta"] / p["se"]) if p.get("se") else 0 for p in gz])
    c2[fam]["gray"] = {"pairs": len(gz), "V1": cnt(gz, v1), "V1_action": cnt(gz, v1a), "beta_max_of_detected": max([p["beta"] for p in gz if v1(p)] or [0]),
                       "cos_max": float(max(p["cos"] for p in gz)), "z_max": float(zg.max()),
                       "detected_pairs": [(p["entry"], p["cand"], round(p["beta"], 3), round(p["cos"], 3), round(p["share"], 2)) for p in gz if v1(p)]}
res["c2"] = c2
# ---- C3
c3 = json.load(open(os.path.join(out, "c3_results.json")))["cases"]
rows = []
for c in c3:
    A = c["A"]; m = {"beta": A["beta"], "cos": A["cos"], "share": A["share"], "null": A["null"], "se": A["se"], "splice": A["splice_layers"]}
    rows.append({"family": c["family"], "case": c["case"], **{k: c.get(k) for k in ("share", "kappa", "rel_noise", "noise_x_dA", "n")}, "beta": A["beta"], "cos": A["cos"], "z": A["z"],
                 "V0": bool(v0(m)), "V1": bool(v1(m)), "V1_action": bool(v1a(m)), "copyA": A["copy"], "grid": c["grid"]["max"], "scale_share": c["frame"]["scale_share"]})
res["c3"] = rows
# ---- C1 (recipes)
c1 = json.load(open(os.path.join(out, "c1_results.json")))
res["c1_pairs_detected_V0"] = sum(1 for p in c1["pairs"].values() if v0({**p, "splice": p["splice"]}))
res["c1_pairs_detected_V1"] = sum(1 for p in c1["pairs"].values() if v1({**p, "splice": p["splice"]}))
json.dump(res, open(os.path.join(out, "sweep.json"), "w"), indent=1)
for fam, r in c2.items():
    print("C2 %-12s indep pairs %5d | V0 FP %2d  V1 FP %2d  V1+action FP %2d | edit pairs %3d V0 %2d V1 %2d V1a %2d | cos max %.4f p99.9 %.4f z max %.1f p99.9 %.1f share max %.2f" % (
        fam, r["pairs_independent"], r["indep"]["V0"], r["indep"]["V1"], r["indep"]["V1_action"], r["same_recipe_edit_pairs"], r["edit"]["V0"], r["edit"]["V1"], r["edit"]["V1_action"], r["cos_max"], r["cos_p999"], r["z_max"], r["z_p999"], r["share_max"]))
    print("   gray zone (cos<0.2): pairs %d, V1 detected %d (action %d), max beta of detected %.3f; %s" % (r["gray"]["pairs"], r["gray"]["V1"], r["gray"]["V1_action"], r["gray"]["beta_max_of_detected"], r["gray"]["detected_pairs"]))
print("\nC3 detection by construction (V0 / V1), rows = share:")
for fam in ("qwen", "gemma"):
    for kind in ("merge/linear", "merge/task_arith", "merge/ties", "merge/dare"):
        print(" %-6s %-18s" % (fam, kind), "  ".join("%.2f:%s/%s" % (r["share"], "Y" if r["V0"] else "n", "Y" if r["V1"] else "n") for r in rows if r["family"] == fam and r["case"] == kind))
    for kappa in (0.0, 1.0, 3.0, 10.0):
        print(" %-6s own-noise k=%-5s" % (fam, kappa), "  ".join("%.2f:%s/%s" % (r["share"], "Y" if r["V0"] else "n", "Y" if r["V1"] else "n") for r in rows if r["family"] == fam and r["case"] == "own-noise" and r["kappa"] == kappa))
print("C1 pairs detected: V0 %d, V1 %d of %d" % (res["c1_pairs_detected_V0"], res["c1_pairs_detected_V1"], len(c1["pairs"])))
