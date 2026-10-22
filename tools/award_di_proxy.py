#!/usr/bin/env python3
"""Award-time Decision-Index-proxy check: paired candidate-vs-base numbers on the frozen house DI-proxy eval.

Inputs are per-item JSONL results (id, unit|area, weight_group, score in [0,1], protocol) for the candidate and the
official base on the SAME frozen eval. The tool refuses anything that is not a clean paired comparison and prints the
"di_proxy" block (points on a 0-100 scale, 4 decimals) as JSON on stdout and a one-line summary on stderr.
It only produces numbers; the pass/fail decision (lower90 >= -0.5) and the manual card on failure belong to the signer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

Z90 = 1.2816  # one-sided 90 % normal quantile
THRESHOLD = -0.5  # non-inferiority margin in points: lower90 = diff - Z90 * se must be >= this
BOOTSTRAP_RESAMPLES = 2000


class ProxyError(ValueError):
    """The inputs are not a clean paired comparison; no numbers are produced."""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_items(path, stratum_field: str = "unit") -> dict:
    """id -> {"stratum", "score", "protocol"}. Refuses duplicate ids, NaN/out-of-range scores, missing fields."""
    out: dict = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ProxyError(f"{path}:{n}: invalid JSON ({e})") from None
        if not isinstance(row, dict):
            raise ProxyError(f"{path}:{n}: not an object")
        iid = row.get("id")
        if not isinstance(iid, str) or not iid:
            raise ProxyError(f"{path}:{n}: missing id")
        if iid in out:
            raise ProxyError(f"{path}:{n}: duplicate id {iid!r}")
        stratum = row.get(stratum_field)
        if stratum is None and stratum_field == "unit":
            stratum = row.get("area")  # `area` is accepted as the unit's alias
        if not isinstance(stratum, str) or not stratum:
            raise ProxyError(f"{path}:{n}: missing {stratum_field} for {iid!r}")
        score = row.get("score")
        if isinstance(score, bool):
            score = float(score)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ProxyError(f"{path}:{n}: score for {iid!r} is missing, NaN or not a number")
        if not 0.0 <= score <= 1.0:
            raise ProxyError(f"{path}:{n}: score for {iid!r} outside [0, 1]")
        protocol = row.get("protocol")
        if not isinstance(protocol, str) or not protocol:
            raise ProxyError(f"{path}:{n}: missing protocol for {iid!r}")
        out[iid] = {"stratum": stratum, "score": float(score), "protocol": protocol}
    if not out:
        raise ProxyError(f"{path}: no items")
    return out


def pair_items(cand: dict, base: dict, eval_ids=None):
    """Return (ids, strata, protocols, d) with d = candidate - base per item; refuses any set/unit/protocol mismatch."""
    if set(cand) != set(base):
        missing_c = sorted(set(base) - set(cand))[:3]
        missing_b = sorted(set(cand) - set(base))[:3]
        raise ProxyError(f"item sets differ (missing in candidate: {missing_c}; missing in base: {missing_b})")
    if eval_ids is not None and set(cand) != set(eval_ids):
        raise ProxyError("item set differs from the frozen eval file")
    ids = sorted(cand)
    for i in ids:
        if cand[i]["stratum"] != base[i]["stratum"]:
            raise ProxyError(f"unit differs between candidate and base for {i!r}")
        if cand[i]["protocol"] != base[i]["protocol"]:
            raise ProxyError(f"protocol differs between candidate and base for {i!r}")
    strata = [cand[i]["stratum"] for i in ids]
    protocols = [cand[i]["protocol"] for i in ids]
    d = np.array([cand[i]["score"] - base[i]["score"] for i in ids], dtype=float)
    return ids, strata, protocols, d


def normalised_weights(strata_present, weights=None) -> dict:
    """unit -> weight summing to 1. Default equal over the units present; a given table must cover exactly those units."""
    units = sorted(set(strata_present))
    if weights is None:
        return {u: 1.0 / len(units) for u in units}
    if set(weights) != set(units):
        raise ProxyError(f"weight units {sorted(weights)} do not match item units {units}")
    for u, w in weights.items():
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w) or w <= 0:
            raise ProxyError(f"weight for {u!r} must be a positive finite number")
    total = float(sum(weights.values()))
    return {u: float(weights[u]) / total for u in units}


def _groups(strata, values, weights):
    by: dict = {}
    for s, v in zip(strata, values):
        by.setdefault(s, []).append(v)
    for u, vals in by.items():
        if len(vals) < 2:
            raise ProxyError(f"unit {u!r} has {len(vals)} item(s); at least 2 are needed for a standard error")
    return {u: np.array(vals, dtype=float) for u, vals in by.items()}


def weighted_mean(strata, values, weights) -> float:
    """Composite = sum_u w_u * mean of the unit's item values."""
    return float(sum(weights[u] * g.mean() for u, g in _groups(strata, values, weights).items()))


def analytic_se(strata, d, weights) -> float:
    """Paired stratified SE: sqrt(sum_u w_u^2 * var_u(d) / n_u), var with ddof=1. Exact McNemar form on binary data."""
    return float(math.sqrt(sum(weights[u] ** 2 * g.var(ddof=1) / len(g) for u, g in _groups(strata, d, weights).items())))


def bootstrap_se(strata, d, weights, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> float:
    """Stratified paired item-level bootstrap SE of the weighted diff (resample item diffs within each unit)."""
    rng = np.random.default_rng(seed)
    groups = _groups(strata, d, weights)
    draws = np.zeros(resamples)
    for u in sorted(groups):  # fixed order keeps the seeded stream reproducible
        g = groups[u]
        idx = rng.integers(0, len(g), size=(resamples, len(g)))
        draws += weights[u] * g[idx].mean(axis=1)
    return float(draws.std(ddof=1))


def compute(cand: dict, base: dict, *, weights=None, seed: int = 0, resamples: int = BOOTSTRAP_RESAMPLES, eval_ids=None) -> dict:
    """Numbers for the di_proxy block (points, 4 decimals): score, base_score, diff, se (larger of bootstrap and analytic)."""
    ids, strata, protocols, d = pair_items(cand, base, eval_ids)
    w = normalised_weights(strata, weights)
    cs = np.array([cand[i]["score"] for i in ids])
    bs = np.array([base[i]["score"] for i in ids])
    score = 100.0 * weighted_mean(strata, cs, w)
    base_score = 100.0 * weighted_mean(strata, bs, w)
    diff = 100.0 * weighted_mean(strata, d, w)
    se = 100.0 * max(analytic_se(strata, d, w), bootstrap_se(strata, d, w, resamples, seed))
    protos = sorted(set(protocols))
    return {"items": len(ids), "protocol": protos[0] if len(protos) == 1 else "mixed:" + "/".join(protos),
            "score": round(score, 4), "base_score": round(base_score, 4), "diff": round(diff, 4), "se": round(se, 4)}


def build_block(cand, base, *, eval_id, eval_sha, base_id, weights=None, seed=0, eval_ids=None) -> dict:
    r = compute(cand, base, weights=weights, seed=seed, eval_ids=eval_ids)
    return {"done": True, "eval_id": eval_id, "eval_sha": eval_sha, "protocol": r["protocol"], "base_id": base_id,
            "items": r["items"], "score": r["score"], "base_score": r["base_score"], "diff": r["diff"], "se": r["se"]}


def _eval_item_ids(path):
    """Ids of the frozen eval file when it is JSONL with an `id` per line; None when it has another shape."""
    ids = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                ids.append(row["id"])
    except (ValueError, KeyError, TypeError):
        return None
    return ids or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, help="candidate per-item JSONL")
    ap.add_argument("--base", required=True, help="base per-item JSONL")
    ap.add_argument("--eval-file", required=True, help="frozen eval item file (sha256 is computed and checked)")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--eval-sha", required=True, help="expected sha256 of --eval-file")
    ap.add_argument("--base-id", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--weights", required=True,
                    help="JSON file: unit -> weight (normalised); required so the block always uses the frozen composite")
    ap.add_argument("--weight-field", default="unit", choices=["unit", "area", "weight_group"],
                    help="item field the weights apply to (default unit; area is its alias)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-items", type=int, default=1000, help="refuse below this many paired items (signer rule)")
    a = ap.parse_args(argv)
    try:
        sha = file_sha256(a.eval_file)
        if sha != a.eval_sha.lower():
            raise ProxyError(f"eval sha mismatch: file {sha} != --eval-sha {a.eval_sha}")
        weights = json.loads(Path(a.weights).read_text(encoding="utf-8"))
        eval_ids = _eval_item_ids(a.eval_file)
        if eval_ids is None:   # critic F6b: never skip the binding of the item set to the pinned eval
            raise ProxyError("eval file is not JSONL with an id per line: the item set cannot be bound to it")
        field = "unit" if a.weight_field == "area" else a.weight_field
        cand = load_items(a.candidate, field)
        base = load_items(a.base, field)
        block = build_block(cand, base, eval_id=a.eval_id, eval_sha=sha, base_id=a.base_id, weights=weights,
                            seed=a.seed, eval_ids=eval_ids)
        if block["items"] < a.min_items:
            raise ProxyError(f"only {block['items']} paired items (needs >= {a.min_items})")
    except (ProxyError, OSError, json.JSONDecodeError) as e:
        print(f"award_di_proxy: REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(block, sort_keys=True))
    lower90 = block["diff"] - Z90 * block["se"]
    verdict = "meets" if lower90 >= THRESHOLD else "BELOW"
    print(f"di_proxy {a.candidate_id} vs {a.base_id}: diff {block['diff']:+.4f} se {block['se']:.4f} "
          f"lower90 {lower90:+.4f} ({verdict} threshold {THRESHOLD}) items {block['items']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
