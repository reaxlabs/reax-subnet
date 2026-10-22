"""Simulate per-coldkey diminishing returns (tiers.COLDKEY_DR_*) on the public dashboard data.

    python tools/sim_coldkey_dr.py latest.json --out-dir DIR     # writes DIR/sim.json and DIR/SIM.md

Rebuilds ema_by_pool / capacity_by_pool from the dashboard's per-miner rows (uid, pool, ema, capacity.dps), runs today's
``compute_tiered_weights`` and the variants, and prints before/after distributions. Read-only on the input.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reax_subnet import tiers  # noqa: E402

POOLS = ("s1-fast", "s1-pro", "s1-lite", "s1-cpu")
MIN_EMA = 0.3  # ScoringParams.min_ema
T0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
DEFAULT_POOLS = ("s1-lite", "s1-cpu")  # tiers.COLDKEY_DR_POOLS default (headline variant); POOLS = the "all pools" alternative
BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), (">10", 11, 10**9))


def load(path):
    d = json.load(open(path))
    ema = {p: {} for p in POOLS}
    cap = {p: {} for p in POOLS}
    coldkey, hotkey, published = {}, {}, {}
    for m in d["miners"]:
        coldkey[m["uid"]], hotkey[m["uid"]] = m["coldkey"], m["hotkey"]
        published[m["uid"]] = m.get("weight") or 0.0
        pools = ([m["pool"]] if m["pool"] else []) + list(m.get("also_pools") or [])
        for p in pools:
            if p in ema and m.get("ema") is not None:
                ema[p][m["uid"]] = m["ema"]
                if m.get("capacity"):
                    cap[p][m["uid"]] = m["capacity"]["dps"]
    return d, ema, cap, coldkey, hotkey, published


def run(ema, cap, coldkey, *, dr, free=2, ratio=0.7, dr_pools=DEFAULT_POOLS, brake=0.5, coldkey_override=None):
    """Weights per uid. dr=False is today's rule; dr_pools is tiers.COLDKEY_DR_POOLS (the other pools pay as today)."""
    saved = (tiers.COLDKEY_DR_START_UTC, tiers.COLDKEY_FREE_SLOTS, tiers.COLDKEY_DR_RATIO, tiers.COLDKEY_DR_POOLS)
    tiers.COLDKEY_DR_START_UTC, tiers.COLDKEY_FREE_SLOTS, tiers.COLDKEY_DR_RATIO = T0, free, ratio
    tiers.COLDKEY_DR_POOLS = frozenset(dr_pools)
    try:
        cmap = coldkey_override or coldkey
        return tiers.compute_tiered_weights(ema, cap, tiers.TierBudgets(order_ratio=brake), None, MIN_EMA,
                                            coldkey_of=cmap if dr else None, now=T0 if dr else None)
    finally:
        tiers.COLDKEY_DR_START_UTC, tiers.COLDKEY_FREE_SLOTS, tiers.COLDKEY_DR_RATIO, tiers.COLDKEY_DR_POOLS = saved


def gini(xs):
    xs = sorted(x for x in xs)
    n, s = len(xs), sum(xs)
    return 0.0 if n == 0 or s == 0 else sum((2 * i - n + 1) * x for i, x in enumerate(xs)) / (n * s)


def corr(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) if va and vb else float("nan")


def bucket(n):
    return next(b for b, lo, hi in BUCKETS if lo <= n <= hi)


def summarize(w, coldkey, uids_by_ck):
    per_ck = defaultdict(float)
    for u, x in w.items():
        per_ck[coldkey[u]] += x
    top5 = sorted(per_ck.items(), key=lambda kv: -kv[1])[:5]
    return {"gini_hotkeys": gini(list(w.values())), "gini_coldkeys": gini(list(per_ck.values())),
            "top5_coldkeys": [(c[:8], round(x, 5)) for c, x in top5], "top5_share": round(sum(x for _, x in top5), 5)}


def per_tier(w, ema):
    out = {}
    for t in tiers.TIER_ORDER:
        us = {u for p in POOLS if tiers.POOL_TIER[p] == t for u in ema[p]}
        out[t] = round(sum(w.get(u, 0.0) for u in us), 5)
    return out


def per_pool_ck(w, ema, coldkey):
    out = {}
    for p in POOLS:
        d = defaultdict(float)
        for u in ema[p]:
            d[coldkey[u]] += w.get(u, 0.0)
        out[p] = {"total": round(sum(d.values()), 5), "gini_coldkeys": round(gini(list(d.values())), 4),
                  "top5": [(c[:8], round(x, 5)) for c, x in sorted(d.items(), key=lambda kv: -kv[1])[:5]]}
    return out


def compare_variant(base, new, coldkey, ema):
    """Per-coldkey change new/base-1 for paid coldkeys; buckets by hotkeys with weight in the base run."""
    n_hk = defaultdict(int)
    tot_b, tot_n = defaultdict(float), defaultdict(float)
    for u, x in base.items():
        n_hk[coldkey[u]] += 1
        tot_b[coldkey[u]] += x
    for u, x in new.items():
        tot_n[coldkey[u]] += x
    rows = {}
    for b, lo, hi in BUCKETS:
        cks = [c for c, n in n_hk.items() if lo <= n <= hi]
        if not cks:
            continue
        ch = [tot_n[c] / tot_b[c] - 1 for c in cks if tot_b[c] > 0]
        per_hk_b = [base[u] for u in base if coldkey[u] in cks]
        per_hk_n = [new.get(u, 0.0) for u in base if coldkey[u] in cks]
        rows[b] = {"coldkeys": len(cks), "hotkeys": len(per_hk_b), "median_hotkey_before": statistics.median(per_hk_b),
                   "median_hotkey_after": statistics.median(per_hk_n),
                   "coldkey_change_min": min(ch), "coldkey_change_median": statistics.median(ch), "coldkey_change_max": max(ch),
                   "within_5pct": sum(abs(x) <= 0.05 for x in ch), "within_10pct": sum(abs(x) <= 0.10 for x in ch), "n": len(ch)}
    small = [c for c, n in n_hk.items() if n <= 2 and tot_b[c] > 0]
    ch = [tot_n[c] / tot_b[c] - 1 for c in small]
    return rows, {"small_coldkeys": len(small), "min": min(ch), "max": max(ch), "within_5pct": sum(abs(x) <= 0.05 for x in ch),
                  "within_10pct": sum(abs(x) <= 0.10 for x in ch)}


def uplift(base, new, coldkey, ema):
    """Per pool: factor w'/w of the miners DR does not touch (single hotkey of a coldkey in the pool; the pool's freed
    weight and the tier-cap change reach them all alike). The honest reference for 'unchanged'."""
    out = {}
    for p in POOLS:
        cnt = defaultdict(int)
        for u in ema[p]:
            cnt[coldkey[u]] += 1
        xs = [new.get(u, 0.0) / base[u] for u in ema[p] if u in base and cnt[coldkey[u]] == 1]
        out[p] = statistics.median(xs) if xs else float("nan")
    return out


def relative_small(base, new, coldkey, ema):
    """Coldkeys with <= 2 paid hotkeys: pay change relative to the untouched miners of their pool (0 = DR does nothing)."""
    up = uplift(base, new, coldkey, ema)
    pool_of = {u: p for p in POOLS for u in ema[p]}
    n_hk = defaultdict(int)
    for u in base:
        n_hk[coldkey[u]] += 1
    rel = {}
    for c, n in n_hk.items():
        if n > 2:
            continue
        us = [u for u in base if coldkey[u] == c]
        exp = sum(base[u] * up[pool_of[u]] for u in us)
        rel[c] = sum(new.get(u, 0.0) for u in us) / exp - 1
    return rel


def relative_buckets(base, new, coldkey, ema):
    """Per hotkey-count bucket: coldkey pay change relative to the uplift of the untouched miners of its pools."""
    up = uplift(base, new, coldkey, ema)
    pool_of = {u: p for p in POOLS for u in ema[p]}
    n_hk, exp, tot = defaultdict(int), defaultdict(float), defaultdict(float)
    for u, x in base.items():
        n_hk[coldkey[u]] += 1
        exp[coldkey[u]] += x * up[pool_of[u]]
        tot[coldkey[u]] += new.get(u, 0.0)
    out = {}
    for b, lo, hi in BUCKETS:
        ch = [tot[c] / exp[c] - 1 for c, n in n_hk.items() if lo <= n <= hi and exp[c] > 0]
        if ch:
            out[b] = {"rel_min": min(ch), "rel_median": statistics.median(ch), "rel_max": max(ch)}
    return out


def notice_table(base, new, coldkey, ema, lite_farm):
    """Per pool: raw pay change of a miner that is its coldkey's only hotkey in the pool (median), and the raw change of
    the s1-lite farm coldkey (whole coldkey total)."""
    out = {}
    for p in POOLS:
        cnt = defaultdict(int)
        for u in ema[p]:
            cnt[coldkey[u]] += 1
        xs = [new.get(u, 0.0) / base[u] - 1 for u in ema[p] if u in base and cnt[coldkey[u]] == 1]
        out[p] = {"single_hotkey_median": statistics.median(xs) if xs else None, "single_hotkey_min": min(xs) if xs else None,
                  "single_hotkey_max": max(xs) if xs else None, "n": len(xs)}
    us = [u for u in base if coldkey[u] == lite_farm]
    out["lite_farm"] = {"hotkeys": len(us), "change": sum(new.get(u, 0.0) for u in us) / sum(base[u] for u in us) - 1}
    return out


def farm_stats(w, coldkey, ck):
    us = [u for u in w if coldkey[u] == ck]
    xs = sorted((w[u] for u in us), reverse=True)
    return {"hotkeys": len(xs), "total": sum(xs), "per_hotkey_mean": sum(xs) / len(xs) if xs else 0.0,
            "first": xs[0] if xs else 0.0, "last": xs[-1] if xs else 0.0}


def evasion(ema, cap, coldkey, farm, n_list, **kw):
    """The farm spreads its hotkeys round-robin over n coldkeys it controls."""
    us = sorted(u for u in coldkey if coldkey[u] == farm)
    out = {}
    for n in n_list:
        ov = dict(coldkey)
        for i, u in enumerate(us):
            ov[u] = f"{farm}#{i % n}"
        w = run(ema, cap, coldkey, dr=True, coldkey_override=ov, **kw)
        out[n] = sum(w.get(u, 0.0) for u in us)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    d, ema, cap, coldkey, hotkey, published = load(a.data)
    out = {"source": a.data, "generated_at": d["generated_at"]}

    base = run(ema, cap, coldkey, dr=False)
    out["baseline"] = {"miners_paid": len(base), "sum": sum(base.values()), "per_tier": per_tier(base, ema),
                       "per_pool": per_pool_ck(base, ema, coldkey), **summarize(base, coldkey, None)}

    # 1. against the published weights (inference part only)
    pooled = sorted(base)
    pub_in = {u: published[u] for u in pooled}
    s_pub = sum(pub_in.values())
    pub_norm = [pub_in[u] / s_pub for u in pooled]
    sim_v = [base[u] for u in pooled]
    diffs = [abs(p - s) for p, s in zip(pub_norm, sim_v)]
    worst = sorted(zip(diffs, pooled), reverse=True)[:5]
    out["vs_published"] = {"miners_compared": len(pooled), "published_sum_over_pooled": s_pub,
                           "published_sum_all": sum(published.values()),
                           "correlation_normalised": corr(pub_norm, sim_v),
                           "max_abs_diff_normalised": max(diffs), "mean_abs_diff": statistics.fmean(diffs),
                           "max_abs_diff_raw": max(abs(published[u] - base[u]) for u in pooled),
                           "worst": [(u, hotkey[u][:6], round(published[u] / s_pub, 5), round(base[u], 5)) for _, u in worst]}

    # 2. before/after: headline = default pools (s1-lite, s1-cpu), alternative = all four pools; free 2, ratio 0.7
    n_by = defaultdict(int)
    for u in base:
        n_by[coldkey[u]] += 1
    farms = [c for c, n in n_by.items() if n > 10]
    lite_farm = max((c for c in farms if any(c == coldkey[u] for u in ema["s1-lite"])), key=lambda c: n_by[c])
    out["farm_pools"] = {c[:8]: sorted({p for p in POOLS for u in ema[p] if coldkey[u] == c}) for c in farms}
    for key, pools in (("default", DEFAULT_POOLS), ("all_pools", POOLS)):
        after = run(ema, cap, coldkey, dr=True, dr_pools=pools)
        blk = {"free": 2, "ratio": 0.7, "pools": list(pools), "per_tier": per_tier(after, ema),
               "per_pool": per_pool_ck(after, ema, coldkey), **summarize(after, coldkey, None)}
        rows, small = compare_variant(base, after, coldkey, ema)
        rel = relative_buckets(base, after, coldkey, ema)
        for k in rows:
            rows[k].update(rel.get(k, {}))
        blk["buckets"], blk["small"] = rows, small
        blk["uplift"] = {k: round(x, 3) for k, x in uplift(base, after, coldkey, ema).items()}
        blk["notice"] = notice_table(base, after, coldkey, ema, lite_farm)
        blk["farms"] = {c[:8]: {"before": farm_stats(base, coldkey, c), "after": farm_stats(after, coldkey, c)} for c in farms}
        blk["evasion"] = {c[:8]: {"hotkeys": n_by[c], "base": sum(base[u] for u in base if coldkey[u] == c),
                                  "by_coldkeys": evasion(ema, cap, coldkey, c, sorted({1, 2, 3, 4, 6, 8, 13, n_by[c]}), dr_pools=pools)}
                          for c in farms}
        out[key] = blk
    out["lite_farm"] = lite_farm[:8]

    # 3. variants
    var = []
    for free in (1, 2, 3):
        for ratio in (0.5, 0.6, 0.7, 0.8):
            for scope, dp in (("lite+cpu only (default)", DEFAULT_POOLS), ("all pools", POOLS)):
                w = run(ema, cap, coldkey, dr=True, free=free, ratio=ratio, dr_pools=dp)
                _, sm = compare_variant(base, w, coldkey, ema)
                big = {c[:8]: sum(w.get(u, 0.0) for u in w if coldkey[u] == c) / max(1e-12, sum(base[u] for u in base if coldkey[u] == c)) - 1
                       for c in farms}
                rel = list(relative_small(base, w, coldkey, ema).values())
                var.append({"rel_min": min(rel), "rel_max": max(rel), "rel_within_5": sum(abs(x) <= 0.05 for x in rel),
                            "uplift": {k: round(x, 3) for k, x in uplift(base, w, coldkey, ema).items()},
                            "eff_limit": free + ratio / (1 - ratio), "free": free, "ratio": ratio, "scope": scope, "small_min": sm["min"], "small_max": sm["max"],
                            "small_within_5": sm["within_5pct"], "small_n": sm["small_coldkeys"],
                            "small_within_10": sm["within_10pct"], "farm_change": big, "gini_hotkeys": gini(list(w.values()))})
    out["variants"] = var
    # brake-only reference: no DR but order_ratio 1.0 (isolates what the removed brake alone does)
    nobrake = run(ema, cap, coldkey, dr=False, brake=1.0)
    _, sm = compare_variant(base, nobrake, coldkey, ema)
    out["no_brake_no_dr"] = {"per_tier": per_tier(nobrake, ema), "small": sm,
                             "farm_change": {c[:8]: sum(nobrake.get(u, 0) for u in nobrake if coldkey[u] == c)
                                             / sum(base[u] for u in base if coldkey[u] == c) - 1 for c in farms}}

    out["_variant_note"] = "farm_change = farm total share after / before - 1"

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "sim.json").write_text(json.dumps(out, indent=1, default=float))
    (od / "SIM.md").write_text(render(out))
    print(render(out))


def pct(x):
    return f"{x * 100:+.1f}%"


def render(o):
    L = [f"# Per-coldkey diminishing returns: simulation on {o['generated_at']} dashboard data", "",
         "Generated by `tools/sim_coldkey_dr.py` (inference part only: owner/burn 0, competition layer left out, min_ema 0.3).", ""]
    b, v = o["baseline"], o["vs_published"]
    L += ["## 1. Today's rule against the published weights", "",
          f"Baseline `compute_tiered_weights` (TierBudgets defaults, order_ratio 0.5) pays {b['miners_paid']} miners. Per tier: {b['per_tier']}.",
          f"Published `weight` summed over those miners: {v['published_sum_over_pooled']:.4f} (all rows: {v['published_sum_all']:.4f}; the rest goes to the",
          "competition layer / owner rows). Compared after normalising the published weights of the pooled miners to sum 1:", "",
          f"* correlation {v['correlation_normalised']:.4f}, max abs diff {v['max_abs_diff_normalised']:.5f}, mean abs diff {v['mean_abs_diff']:.5f}",
          f"* worst: {v['worst']} (uid, hotkey, published normalised, simulated)", "",
          "The gap is expected: the published weight is the validator's final vector (inference part scaled down by the competition",
          "layer, plus keep-alive shares), emitted at 09:04Z from the EMA/capacity state of that moment, while the dashboard rows are 10:12Z;",
          "dashboard `ema` and `capacity.dps` are rounded, and uids that moved between pools in between differ. Not forced to match.", ""]
    L += ["## 2. Before / after", "",
          "Headline: **default** = DR in s1-lite and s1-cpu only (`COLDKEY_DR_POOLS`), 2 free slots, ratio 0.7, brake 0.5 -> 1.0. "
          "Alternative: **all pools** (the same constants, all four pools). *raw* = change of the coldkey's total share vs today; "
          "*rel.* = change relative to the uplift of the untouched miners of the same pools (0 = the rule does nothing to the coldkey).", "",
          f"Baseline Gini over hotkeys {b['gini_hotkeys']:.4f}, over coldkeys {b['gini_coldkeys']:.4f}; top-5 coldkeys hold {b['top5_share']:.4f}: {b['top5_coldkeys']}",
          f"Baseline tier totals {b['per_tier']}", ""]
    for key, title in (("default", "2a. Default (s1-lite, s1-cpu)"), ("all_pools", "2b. Alternative: all pools")):
        v = o[key]
        L += [f"### {title}", "",
              f"After: Gini hotkeys {v['gini_hotkeys']:.4f}, coldkeys {v['gini_coldkeys']:.4f}; top-5 hold {v['top5_share']:.4f}: {v['top5_coldkeys']}",
              f"Tier totals before {b['per_tier']} after {v['per_tier']}; uplift of untouched single-hotkey miners fast/pro/lite/cpu: "
              f"{'/'.join(f'{x:.2f}' for x in v['uplift'].values())}", "", "Per pool (total, Gini over coldkeys, top-5 coldkeys), before -> after:", ""]
        for p in b["per_pool"]:
            x, y = b["per_pool"][p], v["per_pool"][p]
            L.append(f"* {p}: {x['total']} / {x['gini_coldkeys']} / {x['top5']}  ->  {y['total']} / {y['gini_coldkeys']} / {y['top5']}")
        L += ["", "By hotkeys per coldkey (paid hotkeys; change of the coldkey's total share, raw and relative):", "",
              "| hotkeys | coldkeys | hotkeys | median/hotkey before | after | raw min / median / max | rel. min / median / max | within 5 % | within 10 % |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, r in v["buckets"].items():
            L.append(f"| {k} | {r['coldkeys']} | {r['hotkeys']} | {r['median_hotkey_before']:.5f} | {r['median_hotkey_after']:.5f} | "
                     f"{pct(r['coldkey_change_min'])} / {pct(r['coldkey_change_median'])} / {pct(r['coldkey_change_max'])} | "
                     f"{pct(r['rel_min'])} / {pct(r['rel_median'])} / {pct(r['rel_max'])} | {r['within_5pct']}/{r['n']} | {r['within_10pct']}/{r['n']} |")
        L += ["", "The four big coldkeys (more than 10 paid hotkeys):", "",
              "| coldkey | pools | hotkeys | total before | after | change | mean/hotkey before | after | last/first hotkey before | after |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c, f in v["farms"].items():
            bb, aa = f["before"], f["after"]
            L.append(f"| {c}.. | {','.join(o['farm_pools'][c])} | {bb['hotkeys']} | {bb['total']:.5f} | {aa['total']:.5f} | {pct(aa['total'] / bb['total'] - 1)} | "
                     f"{bb['per_hotkey_mean']:.5f} | {aa['per_hotkey_mean']:.5f} | {bb['last'] / bb['first']:.2f} | {aa['last'] / aa['first']:.2f} |")
        L += ["", "Evasion: the coldkey spreads its hotkeys over N coldkeys it controls (total share of its hotkeys; `base` is today's):", ""]
        for c, e in v["evasion"].items():
            L.append(f"* {c}.. ({e['hotkeys']} hotkeys, today {e['base']:.5f}): " + ", ".join(f"N={n}: {x:.5f} ({pct(x / e['base'] - 1)})" for n, x in e["by_coldkeys"].items()))
        L.append("")
    L += ["### 2c. Table for the public notice (default variant, whole percent)", "",
          f"`single-hotkey miner` = median raw pay change of a miner that is its coldkey's only hotkey in the pool (includes the weight the rule frees "
          f"and the lifted brake). `{o['lite_farm']}..` = the {o['default']['notice']['lite_farm']['hotkeys']}-hotkey s1-lite coldkey, raw change of its total.", "",
          "| pool | single-hotkey miner pay change (raw) | 26-hotkey s1-lite coldkey change (raw) |", "|---|---|---|"]
    for p in POOLS:
        med = o["default"]["notice"][p]["single_hotkey_median"]
        farm = f"{o['default']['notice']['lite_farm']['change'] * 100:+.0f}%" if p == "s1-lite" else "n/a"
        L.append(f"| {p} | {'n/a' if med is None else f'{med * 100:+.0f}%'} | {farm} |")
    L.append("")
    nb = o["no_brake_no_dr"]
    L += ["", f"Reference: removing only the brake (order_ratio 1.0, no DR) moves the tiers to {nb['per_tier']}; coldkeys with <= 2 hotkeys change "
          f"{pct(nb['small']['min'])} .. {pct(nb['small']['max'])} ({nb['small']['within_5pct']}/{nb['small']['small_coldkeys']} within 5 %); farms {({k: pct(x) for k, x in nb['farm_change'].items()})}.", ""]
    L += ["## 3. Variants", "", "`small` = coldkeys with <= 2 paid hotkeys. *raw* = change of the coldkey's total share vs today (includes the redistribution of the farms' freed weight, which lifts every untouched miner of a pool by the same *uplift* factor). *rel.* = change relative to that uplift (0 = the rule does nothing to the coldkey; this is the 'honest operator unchanged' test). farm = total share change.", "",
          "| free | ratio | scope | max effective hotkeys per coldkey | uplift of untouched miners fast/pro/lite/cpu | small rel. min | small rel. max | rel. within 5 % | small raw min | small raw max | raw within 5 % | " + " | ".join(f"farm {k}" for k in o["farm_pools"]) + " | Gini hotkeys |", "|" + "---|" * (11 + len(o["farm_pools"]) + 1)]
    for r in o["variants"]:
        L.append(f"| {r['free']} | {r['ratio']} | {r['scope']} | {r['eff_limit']:.2f} | {'/'.join(f'{x:.2f}' for x in r['uplift'].values())} | "
                 f"{pct(r['rel_min'])} | {pct(r['rel_max'])} | {r['rel_within_5']}/{r['small_n']} | {pct(r['small_min'])} | {pct(r['small_max'])} | {r['small_within_5']}/{r['small_n']} | "
                 + " | ".join(pct(x) for x in r["farm_change"].values()) + f" | {r['gini_hotkeys']:.4f} |")
    L += ["", "Honest limit: DR can only see a coldkey link on chain. A farm that registers hotkeys under N different coldkeys keeps",
          "two free slots per coldkey, so the rule raises its cost (one more coldkey per two hotkeys) but does not remove the incentive.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
