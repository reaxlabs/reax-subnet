#!/usr/bin/env python3
"""C1 report: recipes recovered vs the entries' own merge weights, false detections, null stats of plain fine-tunes, start confirmation."""
import json, os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__))
pass
src = open(os.path.join(HERE, "c1.py")).read()
# reuse TRUTH / PURE definitions without running c1.py
ns = {}
exec(src[src.index("M, D, H, M1B, V3A"):src.index("# --- DB ---")], ns)
TRUTH, PURE, GEMMA_PURE = ns["TRUTH"], ns["PURE"], ns["GEMMA_PURE"]
outdir = sys.argv[1]
res = json.load(open(os.path.join(outdir, "c1_results.json")))
thr = res["thresholds"]
pairs, ents = res["pairs"], res["entries"]
short = lambda k: k.split("@")[0]
by_entry = {}
for p in pairs.values():
    by_entry.setdefault(short(p["entry"]), {})[short(p["cand"])] = p
lines = []
P = lines.append
P("## C1a. Recipes recovered (truth = the entry's own NOTICE / merge_report.json)\n")
P("| entry | recipe kind | parent | stated weight | measured beta | cos | detected | within tol | ")
P("|---|---|---|---:|---:|---:|:--:|:--:|")
ok = tot = 0
errs = []
for e, (par, tol, kind) in TRUTH.items():
    for p, w in par.items():
        r = by_entry.get(e, {}).get(p)
        if r is None:
            P("| %s | %s | %s | %.2f | n/a | | | |" % (e, kind, p, w)); continue
        good = r["detected"] and abs(r["beta"] - w) <= tol
        ok += good; tot += 1
        errs.append(r["beta"] - w)
        P("| %s | %s | %s | %.2f | %.3f | %.3f | %s | %s (+-%.2f) |" % (e.split("/")[1], kind, p.split("/")[1], w, r["beta"], r["cos"], "yes" if r["detected"] else "NO", "yes" if good else "NO", tol))
import statistics as st
P("\nRecovered %d of %d stated parent weights within tolerance; error beta - stated weight: mean %+.3f, sd %.3f, min %+.3f, max %+.3f.\n" % (ok, tot, st.mean(errs), st.pstdev(errs), min(errs), max(errs)))
# p5 / 189 / gx chain checks
P("## C1b. Start chains (own change measured from the true start)\n")
for e, s in (("gurujustin/sn92-gen-p5", "start = ronanlee777/sn92-gen-ma"), ("CharlieBT/gx31-decider", "start = Jeffan-31b-1710"), ("Poindexter-Labs/Copernicus-31B", "start = KnowLine-31B-Gen1 (HEAD)")):
    ent = [v for k, v in ents.items() if short(k) == e][0]
    det = [(short(c).split("/")[1], round(p["beta"], 3)) for c, p in by_entry[e].items() if p["detected"]]
    P("- %s (%s): frame ok=%s, start beta=%s, copy-vs-start %.3f; detected beyond the start: %s" % (e, s, ent["frame"]["ok"], ent["start_beta"], ent["copy_vs_start"]["body"], det or "none"))
# false detections
P("\n## C1c. Detections that the recipes do not explain\n")
related = {}
for e, (par, _, _) in TRUTH.items():
    related[e] = set(par)
composites = set(TRUTH) | {"gurujustin/sn92-gen-p5", "CharlieBT/gx32-decider", "CharlieBT/gx33-decider", "cxmplex/sn92-gen-g5c", "cxmplex/sn92-gen-g5f3", "cxmplex/sn92-gen-g5f4", "cxmplex/sn92-gen-g5f5", "Taro1229/sn92-gen-m1-s25"}
fp_list = []
for e, row in by_entry.items():
    for c, p in row.items():
        if not p["detected"]:
            continue
        par_e = related.get(e, set())
        # explained: candidate is a stated parent, or a composite that shares a stated parent / is itself a composite of the entry's parents
        if c in par_e:
            continue
        if e in related and c in composites and (related.get(c, set()) & par_e or c in ("CharlieBT/gx32-decider", "CharlieBT/gx33-decider", "cxmplex/sn92-gen-g5c", "cxmplex/sn92-gen-g5f3", "cxmplex/sn92-gen-g5f4", "cxmplex/sn92-gen-g5f5", "Taro1229/sn92-gen-m1-s25")):
            continue
        fp_list.append((e, c, p))
P("Candidates detected although they are neither a stated parent nor a composite that shares a stated parent with the entry (%d pairs):\n" % len(fp_list))
P("| entry | candidate | beta | cos | share | comment |")
P("|---|---|---:|---:|---:|---|")
for e, c, p in sorted(fp_list, key=lambda x: -x[2]["beta"])[:60]:
    P("| %s | %s | %.3f | %.3f | %.2f | %s |" % (e.split("/")[1], c.split("/")[1], p["beta"], p["cos"], p["share"], "entry has no stated recipe" if e not in TRUTH else "entry has a stated recipe: candidate unexplained"))
# null among plain fine-tunes
P("\n## C1d. Plain fine-tunes of the same start trained on the same public data (sibling-by-data null, real models)\n")
def nullset(names, label):
    vals = [by_entry[a][b] for a in names for b in names if a != b and a in by_entry and b in by_entry[a]]
    if not vals: return
    P("- %s: %d ordered pairs; max beta %.4f, max |beta| %.4f, max cos %.4f, detected: %d" % (label, len(vals), max(v["beta"] for v in vals), max(abs(v["beta"]) for v in vals), max(v["cos"] for v in vals), sum(v["detected"] for v in vals)))
    worst = sorted(vals, key=lambda v: -abs(v["beta"]))[:4]
    for v in worst:
        P("    - %s <- %s: beta %.4f cos %.4f share %.2f null %.4f" % (short(v["entry"]).split("/")[1], short(v["cand"]).split("/")[1], v["beta"], v["cos"], v["share"], v["null"]))
nullset(PURE, "Qwen3.5/mammoth line: m87b, decider-1, h5, m1b, v3a")
nullset(GEMMA_PURE, "Gemma-4-31B-it line: Jeffan, KnowLine, Sieve, turion, OneJev")
P("\n## C1e. Start confirmation (frame check + start beta) and copy / grid shares of all entries\n")
P("| entry | start | frame share | scale share | min cos | start beta | equal to start (bf16) | grid max |")
P("|---|---|---:|---:|---:|---:|---:|---:|")
bad = []
for k, e in sorted(ents.items(), key=lambda kv: kv[0]):
    if "frame" not in e: continue
    f = e["frame"]
    P("| %s | %s | %.4f | %.4f | %.3f | %s | %.4f | %.4f |" % (short(k).split("/")[1], short(e["start"]).split("/")[1], f["frame_share"], f["scale_share"], f["min_cos"] or 0, e["start_beta"], e["copy_vs_start"]["body"], e["grid"]["max"]))
    if not f["ok"]: bad.append(short(k))
P("\nFrame check failures against the declared start: %s" % (bad or "none"))
vals = [e["copy_vs_start"]["body"] for e in ents.values() if "copy_vs_start" in e]
P("Equal-to-start share (copy statistic) of all honest-or-merged entries: min %.4f, max %.4f (threshold %.3f)." % (min(vals), max(vals), thr["copy_share"]))
open(os.path.join(outdir, "c1_report.md"), "w").write("\n".join(lines))
print("\n".join(lines))
