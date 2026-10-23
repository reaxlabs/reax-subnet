#!/usr/bin/env python3
"""Offline eval-v5.1 scorer: ``S_v5.1, T, N, T_eq, N_eq, Pub, G, SE_G, gap_flag`` of one model against its declared base.
No chain, no validator state.

    PYTHONPATH=/path/to/eval-v51/wt/src python tools/score_v5.py --items typed.jsonl [--items public.jsonl ...] \\
        --responses model.jsonl --base-responses base.jsonl (--panel-stats panel.json | --unscaled) [--wN 0.4] \\
        [--exclude-types types.json] [--seed 0] [--resamples 400] [--min-type-items 12]

``--items``: JSONL rows (typed rows, tier "mirror" with part T / N / pub, area, task_type, gold; and/or v4-format rows of the public
tiers same_family / proc_sealed and the old own tiers); mount the private file read-only. ``--responses`` / ``--base-responses``: JSONL
``{"id": <item id>, "probs": {key: p, ...}}`` (a list of such dicts for an item with several questions).
``--panel-stats``: JSON file (or inline JSON) ``{"pool_version": "v0", "T": {"mean": m, "sd": s}, "N": {...}, "Pub": {...}}`` from the
stock-panel run; ``--unscaled`` instead scores with X_eq = X (flagged in the detail). ``--wN``: weight of N (default 0.4; 0 = T only).
``--exclude-types``: JSON list of task type ids that overlap with the entry's training data (type score min(entry, base)).
Prints one JSON object on stdout. Coverage below 95% is rejected unless ``--allow-partial`` is set; every T / N task type needs at
least ``--min-type-items`` items (default 12 per draw; pass 12 x draws for pooled draws; 0 disables).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace


MAX_ROWS = 200_000


class InputFormatError(ValueError):
    """A malformed input row with a file and physical line number."""


def read_jsonl(paths, max_rows=MAX_ROWS, *, require_probs=False, unique_ids=False):
    rows = []
    for path in paths:
        seen = set()
        count = 0
        try:
            with open(path, "rb") as fh:
                for line_no, raw_line in enumerate(fh, 1):
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise InputFormatError(f"{path}:line {line_no}: invalid UTF-8") from exc
                    if not line.strip():
                        continue
                    if count >= max_rows:
                        raise InputFormatError(f"{path}:line {line_no}: exceeds --max-rows={max_rows}")
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise InputFormatError(f"{path}:line {line_no}: {exc.msg}") from exc
                    if not isinstance(row, dict):
                        raise InputFormatError(f"{path}:line {line_no}: expected a JSON object")
                    if "id" not in row:
                        raise InputFormatError(f"{path}:line {line_no}: missing id")
                    try:
                        hash(row["id"])
                    except TypeError as exc:
                        raise InputFormatError(f"{path}:line {line_no}: id is not hashable") from exc
                    if require_probs and "probs" not in row:
                        raise InputFormatError(f"{path}:line {line_no}: missing probs")
                    if unique_ids:
                        try:
                            duplicate = row["id"] in seen
                        except TypeError as exc:
                            raise InputFormatError(f"{path}:line {line_no}: id is not hashable") from exc
                        if duplicate:
                            raise InputFormatError(f"{path}:line {line_no}: duplicate id {row['id']!r}")
                        seen.add(row["id"])
                    rows.append(row)
                    count += 1
        except OSError as exc:
            raise InputFormatError(f"{path}: {exc.strerror or exc}") from exc
    return rows


def read_responses(path, max_rows=MAX_ROWS):
    out = {}
    for row in read_jsonl([path], max_rows, require_probs=True, unique_ids=True):
        out[row["id"]] = row["probs"]
    return out


def read_json_arg(value, what):
    """A JSON file path, or inline JSON text."""
    text = value
    if not value.lstrip().startswith(("{", "[")):
        try:
            with open(value, "rb") as fh:
                text = fh.read().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise InputFormatError(f"{what}: {value}: {getattr(exc, 'strerror', None) or exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"{what}: invalid JSON: {exc.msg}") from exc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", action="append", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--base-responses", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int)
    ap.add_argument("--ignore-unscored-tiers", action="store_true", help="drop rows of unknown tiers instead of failing")
    ap.add_argument("--wN", dest="w_n", type=float, help="weight of N in S (default 0.4; 0 = twins only)")
    ap.add_argument("--panel-stats", help="stock-panel mean / sd per part (JSON file or inline JSON)")
    ap.add_argument("--unscaled", action="store_true", help="no stock-panel scaling: X_eq = X")
    ap.add_argument("--exclude-types", help="JSON list of overlap-excluded task type ids (file or inline)")
    ap.add_argument("--min-type-items", type=int, default=12, help="minimum items per T / N task type (default 12; 0 = off)")
    ap.add_argument("--a-floor", type=float)
    ap.add_argument("--a-se-mult", type=float)
    ap.add_argument("--gold-weight", type=float, help="weight of gold task types inside their area (default 1.2)")
    ap.add_argument("--no-const-baseline", action="store_true", help="type zero point = random chance only, not the best constant answer")
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS, help=f"maximum nonblank rows per input file (default {MAX_ROWS})")
    ap.add_argument("--allow-partial", action="store_true", help="allow coverage below 95%; unanswered items count as wrong")
    ap.add_argument("--omniscience-mode", choices=("aa_index", "half_half"),
                    help="Omniscience: AA index (100 + correct% - incorrect%) / 2 (default) or the old half/half mix")
    ap.add_argument("--omni-acc-weight", type=float, help="half_half only: Omniscience accuracy share (default 0.5)")
    ap.add_argument("--wildguard-mode", choices=("const_corrected", "raw"),
                    help="WildGuard: F1 chance-corrected against the best constant answer (default) or raw F1")
    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if a.max_rows < 1:
        print("score_v5: --max-rows must be >= 1", file=sys.stderr)
        return 2
    if a.min_type_items < 0:
        print("score_v5: --min-type-items must be >= 0", file=sys.stderr)
        return 2
    if bool(a.panel_stats) == bool(a.unscaled):
        print("score_v5: give exactly one of --panel-stats and --unscaled", file=sys.stderr)
        return 2

    try:
        from reax_subnet.spec import v5
        kw = {k: v for k, v in (("w_n", a.w_n), ("a_floor", a.a_floor), ("a_se_mult", a.a_se_mult), ("resamples", a.resamples),
                                ("gold_weight", a.gold_weight), ("omni_acc_weight", a.omni_acc_weight),
                                ("omniscience_mode", a.omniscience_mode), ("wildguard_mode", a.wildguard_mode)) if v is not None}
        if a.no_const_baseline:
            kw["const_baseline"] = False
        if a.panel_stats:
            kw["panel"] = v5.panel_from_json(read_json_arg(a.panel_stats, "--panel-stats"))
        params = replace(v5.DEFAULT, **kw)
        excl = []
        if a.exclude_types:
            excl = read_json_arg(a.exclude_types, "--exclude-types")
            if not isinstance(excl, list) or not all(isinstance(t, str) for t in excl):
                raise InputFormatError("--exclude-types: expected a JSON list of task type ids")

        items = read_jsonl(a.items, a.max_rows)
        responses = read_responses(a.responses, a.max_rows)
        base_responses = read_responses(a.base_responses, a.max_rows)
        strict = not a.ignore_unscored_tiers
        split = v5.split_items(items, strict=strict)
        scored_items = split.scored
        coverages = {
            "responses": v5.response_coverage(responses, scored_items),
            "base responses": v5.response_coverage(base_responses, scored_items),
        }
        if not a.allow_partial:
            failures = [(name, coverage) for name, coverage in coverages.items() if coverage["fraction"] < 0.95]
            if failures:
                for name, coverage in failures:
                    extras = json.dumps(coverage["extra_ids"], ensure_ascii=False)
                    print(f"score_v5: {name} coverage {coverage['answered']}/{coverage['expected']} "
                          f"({coverage['fraction']:.1%}) is below 95%; extra ids: {extras}; "
                          "use --allow-partial to proceed", file=sys.stderr)
                return 2
        thin = v5.type_coverage(split, a.min_type_items)
        if thin:
            print(f"score_v5: task types below {a.min_type_items} items: " + ", ".join(f"{t} ({c})" for t, c in thin.items())
                  + "; use --min-type-items 0 to proceed", file=sys.stderr)
            return 2

        res = v5.score_v5_detail(responses, items, base_responses, params, a.seed, strict=strict, allow_partial=True,
                                 exclude_types=excl)
        print(json.dumps(res.as_dict(), indent=2, sort_keys=True))
        return 0
    except InputFormatError as exc:
        print(f"score_v5: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"score_v5: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
