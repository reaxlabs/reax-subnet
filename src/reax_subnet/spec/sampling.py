"""Fresh draws from the sealed pool (host directory ``REAX_SPEC_SEALED_DIR``, mounted read-only by the operator).

Layout: ``<dir>/<category>/sealed/sealed.clean.jsonl`` (``sealed.jsonl`` as fallback). Inside a tier, items that were
drawn before are used last (a fresh item of the tier is always preferred), an item used ``ITEM_MAX_USES`` times is
retired, and a confirmation draw is disjoint from the draws it confirms.
Sealed rows never leave this process: only item ids are persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence, Union

SAMPLE_SIZE = 300   # 1.2.0rc6 (was 600): with 85 % non-proc items, a first draw and its disjoint confirmation fit the pools
MIN_SAMPLE = 200
# Share of the draw per tier (docs/SPECIALIST-TRACKS.md/5). Since 1.2.0rc6 (owner decision 16 Oct 2026) 15 % programmatic
# gold from the published procedural families and 85 % LLM-written items, split 50/25/25 same-family, new-domain,
# new-family; 1.2.0rc5 drew 50 % programmatic gold, which let a model that only learned the public procedures clear the
# entry floor. A tier with too few unused items fills its quota with its own least-used items first (1.2.0rc6:
# before, the shortfall went to the other tiers' fresh items, so the public-family proc tier grew as the small tiers were
# used up); only a tier with too few eligible items at all hands the rest to the other non-proc tiers, then to proc.
TIER_MIX = {"proc_sealed": 0.15, "same_family": 0.425, "new_domain": 0.2125, "new_family": 0.2125}
# An item is reused by rule (the miner's model runs in our sandbox, so the miner never sees the item) but not without
# bound: an item drawn this many times is retired, never drawn again, and becomes eligible for later publication.
# 24: sized at the 17 Oct 2026 rate of about one 300-item draw per hour and track; at the rc8 maximum of two per hour a
# ~3,500-item pool lasts about 5 days, so pools are refilled on the "refill needed" warning (8 would retire the binding same_family tier of demand_forecasting after ~54 more draws, ~2 days).
ITEM_MAX_USES = 24
# A track whose LLM-written (non-proc) items have fewer uses left than this many draws' worth logs "refill needed".
# rc8 (review-rc8 F1): counted in remaining non-proc uses, not items: use counts stay even across items, so the item count
# drops off a cliff at the end, and the proc tier (at most 15 % of a draw) kept it above the old 3-draw bar until draws
# failed. 48 draws = about one day at the rc8 maximum of two draws per hour and track.
REFILL_DRAWS = 48
PRIMITIVES = ("choice", "noul", "score")
# Cross-author tier (eval mechanism v3, stage 2 plumbing, OFF by default): items written by authors outside the generator
# families, to measure how much a model's score depends on the generator families' style. A pool without such items
# behaves exactly as before in every mode. "shadow": ``round(n x share)`` of these items are drawn IN ADDITION to the
# normal ``n`` and scored separately (never part of the total or of a decision). "scored": they replace that share of the
# non-proc tiers inside the ``n`` and count in the total. They never take proc's share.
# "binding" (1.2.0rc9.post3): drawn exactly like "scored" (inside the ``n``, counted in the total) and, in the evaluator, the
# capped generalisation penalty of ``scoring.gap_penalty`` is applied to the score of every decision.
# ``X_TIERS``: every tier name a pool row may carry to be a cross-author item. The real-data items arrive as
# "award_realdata"; the installed pool file normally has them rewritten to ``X_TIER``, both are the same tier here.
X_TIER = "cross_author"
X_TIERS = frozenset({X_TIER, "award_realdata"})
X_TIER_MODES = ("off", "shadow", "scored", "binding")
X_TIER_MODE = "off"
X_TIER_SHARE = 0.20


GENERALIST_CATEGORIES = (
    "math_quant", "knowledge_reasoning", "rules_policy_law", "finance_commerce_insurance",
    "support_ops_logistics", "coding_data_it", "tools_agent_routing", "safety_security_harm",
    "language_taste", "forecasting_risk_science",
)
GENERALIST_PER_CATEGORY = 30
# Rules v4, change 3a (owner decision 21 Oct 2026): from this instant a generalist draw has GENERALIST_PER_CATEGORY_V4
# items in each of the ten categories (900 instead of 300) and the margin floor is ``machine.MARGIN_FLOOR_V4``. None = off,
# every function below behaves bit for bit as before. No env override: every validator must use the same instant. The date
# is set in the release notes. Specialist tracks (DF/RA) keep their sizes.
DRAWS_V4_START_UTC: Optional[datetime] = datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)
GENERALIST_PER_CATEGORY_V4 = 90


def v4_active(now: Union[float, datetime, None]) -> bool:
    """True from ``DRAWS_V4_START_UTC`` on (the boundary instant counts as active). ``now``: unix seconds or a datetime;
    None (no time given) or the gate off: False, the old rules."""
    if now is None or DRAWS_V4_START_UTC is None:
        return False
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now >= DRAWS_V4_START_UTC
    return float(now) >= DRAWS_V4_START_UTC.timestamp()


# Eval v5 (docs/EVAL-V5.md): from this instant a generalist draw is 540 mirror items (M) + 360 held-out items (P), scored by
# ``v5.score_v5``. None = off: every function behaves bit for bit as under rules v4. No env override.
# Armed for Mon 26 Oct 2026 14:30Z with pool v0.1 MAIN (Florian GO 23 Oct, DECISIONS v5-monday-go-despite-gate2); the crown slot
# at that instant still crowns on the current eval (rule A, ``SpecEvaluator._v4_draw_now``). The evaluator refuses to start with
# the gate set unless it is wired (``v5.EVALUATOR_WIRED``).
EVAL_V5_START_UTC: Optional[datetime] = datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc)
MIRROR_TIER = "mirror"
MIRROR_FILE = "sealed.mirror.jsonl"


def v5_active(now: Union[float, datetime, None]) -> bool:
    """True from ``EVAL_V5_START_UTC`` on. ``now`` None or the gate off: False."""
    if now is None or EVAL_V5_START_UTC is None:
        return False
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now >= EVAL_V5_START_UTC
    return float(now) >= EVAL_V5_START_UTC.timestamp()


def generalist_per_category(now: Union[float, datetime, None] = None) -> int:
    """Items per category of a generalist draw made at ``now``: 30, from ``DRAWS_V4_START_UTC`` on 90."""
    return GENERALIST_PER_CATEGORY_V4 if v4_active(now) else GENERALIST_PER_CATEGORY


def draw_size(sample_size: int, generalist: bool, now: Union[float, datetime, None] = None) -> int:
    """Items of a draw made at ``now``: the configured ``sample_size``, except that a generalist draw at the default size
    is 10 x ``generalist_per_category(now)`` (any other configured size is left alone and refused by ``draw``)."""
    if generalist and sample_size == SAMPLE_SIZE:
        return len(GENERALIST_CATEGORIES) * generalist_per_category(now)
    return sample_size


class SampleError(RuntimeError):
    pass


def load_pool(sealed_dir: str, category: str) -> list[dict[str, Any]]:
    base = os.path.join(sealed_dir, category, "sealed")
    for name in ("sealed.clean.jsonl", "sealed.jsonl"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            rows = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(json.loads(line))
            if category == "generalist" and EVAL_V5_START_UTC is not None:   # eval v5: the private mirror file, if mounted
                mirror = os.path.join(base, MIRROR_FILE)
                if os.path.exists(mirror):
                    with open(mirror, encoding="utf-8") as fh:
                        rows += [json.loads(line) for line in fh if line.strip()]
            if category == "generalist":
                _validate_generalist(rows)
            return rows
    raise SampleError(f"no sealed pool for {category} under {sealed_dir}")


def _spread(rows: list[dict], n: int, rng: random.Random, use: Mapping[str, int]) -> list[dict]:
    """n rows, about 1/3 per primitive, least-used first (random inside equal use counts)."""
    by: dict[str, list[dict]] = {p: [] for p in PRIMITIVES}
    for r in rows:
        by.setdefault(r["question"]["type"], []).append(r)
    for lst in by.values():
        rng.shuffle(lst)
        lst.sort(key=lambda r: use.get(r["id"], 0))
    kinds = [k for k in by if by[k]]
    picked: list[dict] = []
    while len(picked) < n and kinds:
        for k in list(kinds):
            if len(picked) >= n:
                break
            if by[k]:
                picked.append(by[k].pop(0))
            else:
                kinds.remove(k)
    return picked


def _take_x(rows: list[dict], k: int, rng: random.Random, use: Mapping[str, int]) -> list[dict]:
    """``k`` cross-author items: fresh ones first, then the least used (about 1/3 per primitive, like a tier)."""
    take = _spread([r for r in rows if use.get(r["id"], 0) == 0], k, rng, use)
    if len(take) < k:
        have = {r["id"] for r in take}
        take += _spread([r for r in rows if r["id"] not in have], k - len(take), rng, use)
    return take


def x_pool_gen(pool: Sequence[dict[str, Any]]) -> str:
    """Fingerprint of the pool's cross-author items: a gap measured on one set of them is comparable only with the others
    measured on the same set (the "pool generation" of ``scoring.gap_penalty``). "" = the pool has none."""
    ids = sorted(str(r["id"]) for r in pool if r.get("tier") in X_TIERS)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:8] if ids else ""


def eligible_count(pool: Sequence[dict[str, Any]], use_counts: Mapping[str, int]) -> int:
    """Items that can still be drawn (used fewer than ``ITEM_MAX_USES`` times)."""
    return sum(1 for r in pool if use_counts.get(r["id"], 0) < ITEM_MAX_USES)


def x_draws_left(pool: Sequence[dict[str, Any]], use_counts: Mapping[str, int], sample_size: int = SAMPLE_SIZE,
                 share: float = X_TIER_SHARE) -> float:
    """Draws of ``sample_size`` the cross-author items can still fill: their remaining uses over the X items of one draw
    (``round(sample_size x share)``). Counted apart from the LLM-written tiers (``nonproc_draws_left``)."""
    left = sum(max(0, ITEM_MAX_USES - use_counts.get(r["id"], 0)) for r in pool if r.get("tier") in X_TIERS)
    per_draw = int(round(sample_size * share))
    return left / per_draw if per_draw > 0 else float("inf")


def nonproc_draws_left(pool: Sequence[dict[str, Any]], use_counts: Mapping[str, int], sample_size: int = SAMPLE_SIZE) -> float:
    """Draws of ``sample_size`` the LLM-written tiers can still fill: their remaining uses (``ITEM_MAX_USES`` minus the
    uses so far, per item) over the non-proc part of one draw."""
    left = sum(max(0, ITEM_MAX_USES - use_counts.get(r["id"], 0)) for r in pool if r.get("tier") not in ("proc_sealed", MIRROR_TIER) and r.get("tier") not in X_TIERS)
    per_draw = (1.0 - TIER_MIX.get("proc_sealed", 0.0)) * sample_size
    return left / per_draw if per_draw > 0 else float("inf")


def draw(pool: Sequence[dict[str, Any]], *, n: int, seed: int, use_counts: Mapping[str, int] | None = None,
         exclude: set[str] | frozenset[str] = frozenset(), min_n: int = MIN_SAMPLE, x_mode: str = "off",
         x_share: float = X_TIER_SHARE, generalist: bool = False,
         now: Union[float, datetime, None] = None, per_category: Optional[int] = None) -> tuple[list[dict], bool]:
    """Pick ``n`` items. Returns (rows, fresh) where fresh is False if any already-used item had to be reused.

    Each tier takes its ``TIER_MIX`` quota from its own items, fresh ones first, then its least-used ones. Only what a
    tier cannot fill from its own eligible items goes to the other non-proc tiers (fresh, then least used); the proc
    tier never takes more than its ``TIER_MIX`` share of the draw that results, so a draw without enough non-proc items
    is smaller. Items in ``exclude`` are never picked, nor are retired items (used ``ITEM_MAX_USES`` times or more). If fewer than ``min_n`` eligible items exist, or the draw came out
    smaller than both ``n`` and ``min_n``, SampleError.

    Cross-author items (``X_TIERS``) are never part of the normal draw. ``x_mode`` "shadow" adds ``round(n x x_share)`` of
    them to the result (the caller scores them apart); "scored" and "binding" take them out of the non-proc tiers' share of
    the ``n`` (fewer usable X items than that: the rest of the share is filled by the normal tiers, so the draw still has
    ``n`` items); without eligible cross-author items in the pool all of them are the same as "off".

    ``generalist``: exactly ``generalist_per_category(now)`` items in each category (``now``: the draw's time, unix seconds
    or datetime; None = the old 30 per category), so ``n`` must be ``draw_size(SAMPLE_SIZE, True, now)``. ``per_category``
    overrides that size (the release dry-run tool only; the evaluator never sets it)."""
    if generalist and v5_active(now):
        from . import v5
        return v5.draw_v5(pool, n=n, seed=seed, use_counts=use_counts or {}, exclude=exclude)
    if generalist:
        per_cat = per_category or generalist_per_category(now)
        if n != per_cat * len(GENERALIST_CATEGORIES):
            raise SampleError(f"generalist draws require exactly {per_cat * len(GENERALIST_CATEGORIES)} items")
        return _draw_generalist(pool, seed, use_counts or {}, exclude, x_mode, x_share, per_cat)
    use = use_counts or {}
    rng = random.Random(seed)
    usable = [r for r in pool if r["id"] not in exclude and use.get(r["id"], 0) < ITEM_MAX_USES]
    eligible = [r for r in usable if r.get("tier") not in X_TIERS]
    x_rows = [r for r in usable if r.get("tier") in X_TIERS] if x_mode in ("shadow", "scored", "binding") else []
    if len(eligible) < min_n:
        raise SampleError(f"only {len(eligible)} eligible sealed items, need {min_n}")
    n = min(n, len(eligible))
    n_x = min(int(round(n * x_share)), len(x_rows)) if x_rows else 0
    chosen: list[dict] = []
    left = n
    fresh = True
    quota = {t: int(round(n * w)) for t, w in TIER_MIX.items()}
    if n_x and x_mode in ("scored", "binding"):
        # the cross-author items replace their share of the non-proc tiers (proc keeps its own share, never takes it)
        x_taken = _take_x(x_rows, n_x, random.Random(seed ^ 0x5EED), use)
        nonproc_w = 1.0 - TIER_MIX.get("proc_sealed", 0.0)
        f = max(0.0, (nonproc_w * n - len(x_taken)) / (nonproc_w * n)) if nonproc_w > 0 else 0.0
        quota = {t: (q if t == "proc_sealed" else int(round(q * f))) for t, q in quota.items()}
        chosen += x_taken
        left -= len(x_taken)
    for t in TIER_MIX:
        rows = [r for r in eligible if r.get("tier") == t]
        take = _spread([r for r in rows if use.get(r["id"], 0) == 0], min(quota[t], left), rng, use)
        short = min(quota[t], left) - len(take)
        if short > 0:   # the tier's own least-used items before any other tier's fresh ones
            have = {r["id"] for r in take}
            again = _spread([r for r in rows if r["id"] not in have], short, rng, use)
            take += again
            fresh = fresh and not again
        chosen += take
        left -= len(take)
    if left > 0:   # a tier with too few eligible items: the other non-proc tiers, fresh before reused
        have = {r["id"] for r in chosen}
        rest = [r for r in eligible if r["id"] not in have and r.get("tier") != "proc_sealed"]
        for reused in (False, True):
            part = [r for r in rest if (use.get(r["id"], 0) > 0) == reused]
            take = _spread(part, left, rng, use)
            chosen += take
            left -= len(take)
            fresh = fresh and not (reused and take)
    # The proc tier never takes another tier's share (1.2.0rc6): without enough non-proc items the draw is smaller, and
    # proc keeps at most its TIER_MIX share of the draw that results.
    w = TIER_MIX.get("proc_sealed", 0.0)
    proc = [r for r in chosen if r.get("tier") == "proc_sealed"]
    others = len(chosen) - len(proc)
    cap = len(proc) if w >= 1.0 else int(w / (1.0 - w) * others)
    if len(proc) > cap:
        drop = {r["id"] for r in proc[cap:]}
        chosen = [r for r in chosen if r["id"] not in drop]
    if len(chosen) < min(n, min_n):   # a smaller draw asked for (REAX_SPEC_SAMPLE_SIZE) is fine when it is complete
        raise SampleError(f"only {len(chosen)} items fit the tier shares, need {min_n}")
    if n_x and x_mode == "shadow":   # in addition to the normal draw, from its own random stream
        extra = _take_x(x_rows, n_x, random.Random(seed ^ 0x5EED), use)
        chosen += extra
    chosen.sort(key=lambda r: r["id"])   # a stable order, the same for every model of the draw
    return chosen, fresh


def _validate_generalist(rows: Sequence[dict]) -> None:
    ids: set[str] = set()
    for row in rows:
        if row.get("tier") == MIRROR_TIER:   # eval v5 mirror rows have no category and their own schema
            from . import v5
            v5.validate_mirror_row(row)
            if row["id"] in ids:
                raise SampleError(f"duplicate generalist item id {row['id']!r}")
            ids.add(row["id"])
            continue
        if row.get("category") not in GENERALIST_CATEGORIES:
            raise SampleError(f"invalid or missing generalist category on {row.get('id')!r}")
        if row.get("id") in ids:
            raise SampleError(f"duplicate generalist item id {row['id']!r}")
        ids.add(row["id"])
        if row.get("question", {}).get("type") not in PRIMITIVES:
            raise SampleError(f"invalid generalist question type on {row['id']!r}")
        if row.get("tier") not in {*TIER_MIX, *X_TIERS}:
            raise SampleError(f"invalid generalist tier on {row['id']!r}")


def _quotas(n: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Largest remainder allocation, stable key order breaks ties."""
    total = sum(weights.values())
    exact = {k: n * w / total for k, w in weights.items()} if total else {}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: -(exact[k] - out[k]))[:n - sum(out.values())]:
        out[k] += 1
    return out


def generalist_tier_quotas(x_mode: str, x_share: float,
                           per_category: int = GENERALIST_PER_CATEGORY) -> dict[str, dict[str, int]]:
    """Tier targets apply across types within each category; carry rounding to keep full-draw totals."""
    total = per_category * len(GENERALIST_CATEGORIES)
    weights = dict(TIER_MIX)
    if x_mode in ("scored", "binding"):
        nonproc = 1 - weights["proc_sealed"]
        weights = {t: w if t == "proc_sealed" else w * (nonproc - x_share) / nonproc
                   for t, w in weights.items()}
        weights[X_TIER] = x_share
    targets = _quotas(total, weights)
    assigned = {t: 0 for t in weights}
    order = []
    for i in range(total):
        tier = max((t for t in weights if assigned[t] < targets[t]),
                   key=lambda t: (i + 1) * weights[t] - assigned[t])
        order.append(tier)
        assigned[tier] += 1
    return {c: {t: order[i * per_category:(i + 1) * per_category].count(t)
                for t in weights} for i, c in enumerate(GENERALIST_CATEGORIES)}


def generalist_shortfalls(rows: Sequence[dict], x_mode: str, x_share: float,
                          per_category: int = GENERALIST_PER_CATEGORY) -> dict[str, dict[str, int]]:
    """Aggregate diagnostics only; no ids, text or predictions."""
    quotas = generalist_tier_quotas(x_mode, x_share, per_category)
    return {c: {t: max(0, n - sum(r["category"] == c and
                                (X_TIER if r["tier"] in X_TIERS else r["tier"]) == t for r in rows))
                for t, n in target.items()} for c, target in quotas.items()}


def _draw_generalist(pool: Sequence[dict], seed: int, use: Mapping[str, int], exclude: set[str] | frozenset[str],
                     x_mode: str, x_share: float, per_cat: int = GENERALIST_PER_CATEGORY) -> tuple[list[dict], bool]:
    """Exactly ``per_cat`` (30; 90 from ``DRAWS_V4_START_UTC``) per category. Category tier quotas may borrow across types, never categories.

    Type proportions are targets within the tier constraints. Missing tiers use non-proc/non-X
    first. If those are exhausted, remaining eligible rows preserve category liveness; tier
    shortfalls are diagnostic. Only a category below ``per_cat`` eligible items prevents a valid draw.
    """
    _validate_generalist(pool)
    rng = random.Random(seed)
    usable = [r for r in sorted(pool, key=lambda r: r["id"])
              if r.get("tier") != MIRROR_TIER and r["id"] not in exclude and use.get(r["id"], 0) < ITEM_MAX_USES
              and (r["tier"] not in X_TIERS or x_mode in ("scored", "binding"))]
    chosen: list[dict] = []
    targets = generalist_tier_quotas(x_mode, x_share, per_cat)
    for category in GENERALIST_CATEGORIES:
        rows = [r for r in usable if r["category"] == category]
        if len(rows) < per_cat:
            raise SampleError(f"generalist category {category}: only {len(rows)} eligible items, need {per_cat}")
        quotas = _quotas(per_cat,
                         {p: sum(r["question"]["type"] == p for r in rows) for p in PRIMITIVES})
        type_counts = {p: 0 for p in PRIMITIVES}
        picked: list[dict] = []
        have: set[str] = set()

        def take(candidates: list[dict], count: int) -> None:
            by_type = {p: [r for r in candidates if r["question"]["type"] == p and r["id"] not in have]
                       for p in PRIMITIVES}
            for part in by_type.values():
                rng.shuffle(part)
                part.sort(key=lambda r: use.get(r["id"], 0), reverse=True)
            for _ in range(count):
                kinds = [p for p in PRIMITIVES if by_type[p]]
                if not kinds:
                    break
                kind = max(kinds, key=lambda p: (quotas[p] - type_counts[p],
                                                -use.get(by_type[p][-1]["id"], 0)))
                row = by_type[kind].pop()
                picked.append(row)
                have.add(row["id"])
                type_counts[kind] += 1

        for tier, count in targets[category].items():
            take([r for r in rows if (X_TIER if r["tier"] in X_TIERS else r["tier"]) == tier], count)
        take([r for r in rows if r["tier"] != "proc_sealed" and r["tier"] not in X_TIERS],
             per_cat - len(picked))
        # Degenerate all-proc/all-X categories still meet their category quota. The release
        # dry-run exposes shortages; publication/calibration must review such pool shapes.
        take(rows, per_cat - len(picked))
        chosen += picked
    chosen.sort(key=lambda r: r["id"])
    return chosen, all(use.get(r["id"], 0) == 0 for r in chosen)
