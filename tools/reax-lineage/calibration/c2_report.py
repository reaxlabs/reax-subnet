#!/usr/bin/env python3
"""C2 report: null percentiles of beta and cos over independent public fine-tunes. Usage: c2_report.py <out dir> <family> [<family> ...]"""
import json, os, sys
import numpy as np
out = sys.argv[1]
rep = {}
lines = []
P = lines.append
for fam in sys.argv[2:]:
    d = json.load(open(os.path.join(out, "c2_%s.json" % fam)))
    pairs = [p for p in d["pairs"] if p["beta"] is not None]
    thr = d["thresholds"]
    frame_bad = [r for r, e in d["entries"].items() if not e["frame_ok"]]
    import re
    EDIT = re.compile(r"uncensor|abliterat|heretic|derestrict|decensor|unfilter", re.I)
    bad = set(frame_bad)
    drop = [p for p in pairs if p["declared_relation"]]
    keep0 = [p for p in pairs if not p["declared_relation"] and p["entry"] not in bad and p["cand"] not in bad]
    edit_pairs = [p for p in keep0 if EDIT.search(p["entry"]) and EDIT.search(p["cand"])]
    keep = [p for p in keep0 if not (EDIT.search(p["entry"]) and EDIT.search(p["cand"]))]
    indep = [p for p in keep if not p["same_author"]]
    sib = [p for p in keep if p["same_author"]]
    def stats(ps, label):
        if not ps: return None
        b = np.array([p["beta"] for p in ps]); c = np.array([p["cos"] for p in ps]); sh = np.array([p["share"] for p in ps])
        r = {"pairs": len(ps), "beta_max": float(b.max()), "beta_p999": float(np.quantile(b, .999)), "beta_p99": float(np.quantile(b, .99)),
             "abs_beta_max": float(np.abs(b).max()), "abs_beta_p999": float(np.quantile(np.abs(b), .999)),
             "cos_max": float(c.max()), "cos_p999": float(np.quantile(c, .999)), "cos_p99": float(np.quantile(c, .99)), "abs_cos_max": float(np.abs(c).max()),
             "share_max": float(sh.max()), "detected": int(sum(p["detected"] for p in ps)),
             "joint_beta_ge_.05_and_cos_ge_.02": int(((b >= thr["beta_min"]) & (c >= thr["cos_min"])).sum())}
        r["tau_beta_3x"] = 3 * r["beta_p999"]; r["tau_cos_3x"] = 3 * r["cos_p999"]
        return r
    rep[fam] = {"models": d["n_models"], "entries_failing_frame_check": frame_bad, "dropped_declared_relation": len(drop),
                "independent_authors": stats(indep, "indep"), "same_author": stats(sib, "sib"), "all_kept": stats(keep, "all"), "deterministic_edit_pairs": stats(edit_pairs, "edit"),
                "dropped_frame_failures": len(pairs) - len(keep0) - len([p for p in pairs if p["declared_relation"] and (p["entry"] not in bad and p["cand"] not in bad)]),
                "copy_vs_start_max": max(e["copy_vs_start"] for e in d["entries"].values()), "grid_max_max": max(e["grid_max"] for e in d["entries"].values()),
                "scale_share_max": max(e["scale_share"] for e in d["entries"].values()),
                "top_pairs": [(p["entry"], p["cand"], round(p["beta"], 4), round(p["cos"], 4), round(p["share"], 3), p["same_author"]) for p in sorted(keep, key=lambda p: -abs(p["beta"]))[:8]],
                "top_dropped": [(p["entry"], p["cand"], round(p["beta"], 4), round(p["cos"], 4), p["detected"]) for p in sorted(drop, key=lambda p: -abs(p["beta"]))[:6]]}
json.dump(rep, open(os.path.join(out, "c2_summary.json"), "w"), indent=1)
for fam, r in rep.items():
    print("==", fam, "models", r["models"], "frame failures", r["entries_failing_frame_check"], "dropped (declared relation)", r["dropped_dropped"] if False else r["dropped_declared_relation"])
    for k in ("independent_authors", "same_author", "all_kept", "deterministic_edit_pairs"):
        s = r[k]
        if s: print("  %-20s pairs %5d | beta max %.4f p99.9 %.4f | |beta| max %.4f | cos max %.4f p99.9 %.4f | share max %.2f | 3x p99.9: beta %.4f cos %.4f | detected %d | beta>=.05&cos>=.02: %d" % (k, s["pairs"], s["beta_max"], s["beta_p999"], s["abs_beta_max"], s["cos_max"], s["cos_p999"], s["share_max"], s["tau_beta_3x"], s["tau_cos_3x"], s["detected"], s["joint_beta_ge_.05_and_cos_ge_.02"]))
    print("  copy vs start max %.4f | grid max %.4f | scale share max %.4f" % (r["copy_vs_start_max"], r["grid_max_max"], r["scale_share_max"]))
    print("  top kept pairs:", r["top_pairs"][:5]); print("  top dropped pairs (declared relation):", r["top_dropped"][:4])
