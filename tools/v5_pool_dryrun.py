#!/usr/bin/env python3
"""Validate and summarize an eval v5.1 pool without printing item text or ids.

    PYTHONPATH=src python tools/v5_pool_dryrun.py --pool-dir /path/to/v51 [--panel panel.json] [--seed 0]

The simulated draw covers the private T/N pool only. Public gap-check rows come from the separate sealed generalist pool and
are not supplied by this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reax_subnet.spec import v5, v5wire as W  # noqa: E402


def _counts(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _cap_counts(rows: list[dict]) -> dict[str, object]:
    questions = over = 0
    by_part = Counter()
    for row in rows:
        idk = bool(row.get("idk"))
        for question in row["questions"]:
            questions += 1
            if len(question["criteria"]) + int(idk) > W.OPTION_CAP:
                over += 1
                by_part[row["part"]] += 1
    return {"questions": questions, "over_26_including_idk": over,
            "over_26_by_part": dict(sorted(by_part.items()))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True, type=Path, metavar="D", help="directory containing v51.T.jsonl and v51.N.jsonl")
    parser.add_argument("--panel", type=Path, metavar="P", help="optional stock-panel JSON file")
    parser.add_argument("--seed", type=int, default=0, metavar="N", help="simulated draw seed (default: 0)")
    args = parser.parse_args(argv)

    try:
        panel = v5.panel_from_json(json.loads(args.panel.read_text(encoding="utf-8"))) if args.panel else None
        config = W.load_config(str(args.pool_dir))
        params = W.params(panel=panel, pool_dir=str(args.pool_dir))
        pool = W.load_pool(str(args.pool_dir), params)
        # No public pool path is part of this CLI. Simulate the private T/N allocation; production adds 90 public gap-check
        # rows from the ordinary sealed pool.
        draw_params = replace(params, pub_items=0)
        drawn, fresh = W.draw(pool, [], seed=args.seed, use_counts={}, p=draw_params)
    except Exception:  # noqa: BLE001 - errors can contain private row text or identifiers; never echo them
        print("v5 pool dry run failed: load, panel, or draw validation error", file=sys.stderr)
        return 1

    report = {
        "pool": {
            "items": len(pool),
            "by_part": _counts(pool, "part"),
            "by_area": _counts(pool, "area"),
            "by_task_type": _counts(pool, "task_type"),
        },
        "pool_config": {"file": W.CONFIG_FILE, "found": config is not None,
                        "pool_version": config[0] if config else None, "hf_areas": dict(params.hf_areas)},
        "cap_statistics": _cap_counts(pool),
        "simulated_draw": {
            "seed": args.seed,
            "items": len(drawn),
            "fresh": fresh,
            "by_part": _counts(drawn, "part"),
            "by_area": _counts(drawn, "area"),
            "by_task_type": _counts(drawn, "task_type"),
            "public_gap_check_items": 0,
        },
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
