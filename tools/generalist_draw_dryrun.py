#!/usr/bin/env python3
"""Exercise consecutive first/confirmation draws without exporting item content or ids."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reax_subnet.spec import sampling as S


def dryrun(pool: list[dict], pairs: int, seed: int = 0, per_category: int = S.GENERALIST_PER_CATEGORY) -> dict:
    use: Counter = Counter()
    shortfalls = {c: Counter() for c in S.GENERALIST_CATEGORIES}
    draws = 0
    complete_pairs = 0
    exhausted = False
    for pair in range(pairs):
        exclude: set[str] = set()
        for phase in range(2):
            try:
                rows, _ = S.draw(pool, n=per_category * len(S.GENERALIST_CATEGORIES), seed=seed + 2 * pair + phase,
                                 use_counts=use, exclude=exclude, generalist=True, x_mode="binding",
                                 per_category=per_category)
            except S.SampleError:
                exhausted = True
                break
            draws += 1
            for category, counts in S.generalist_shortfalls(rows, "binding", S.X_TIER_SHARE, per_category).items():
                shortfalls[category].update(counts)
            use.update(r["id"] for r in rows)
            exclude = {r["id"] for r in rows}
        if exhausted:
            break
        complete_pairs += 1
    return {"pairs_requested": pairs, "pairs_possible": complete_pairs, "draws_possible": draws,
            "shortfalls": {c: dict(counts) for c, counts in shortfalls.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path, help="pool JSONL")
    parser.add_argument("--pairs", "-n", type=int, default=100, help="number of first + confirmation pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-category", type=int, default=S.GENERALIST_PER_CATEGORY,
                        help="items per category of a draw (default the live constant; 90 = rules v4 draws of 900)")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.per_category < 1:
        parser.error("--per-category must be positive")
    try:
        with args.pool.open(encoding="utf-8") as fh:
            pool = [json.loads(line) for line in fh if line.strip()]
        S._validate_generalist(pool)
        report = dryrun(pool, args.pairs, args.seed, args.per_category)
    except (OSError, ValueError, KeyError, TypeError, S.SampleError):
        # Never echo malformed input, item ids, or JSON decoder context from a sealed pool.
        parser.exit(2, "invalid or unreadable generalist pool\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
