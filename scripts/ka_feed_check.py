#!/usr/bin/env python3
"""Read-only fresh-feed gate: exit 0 only if the keep-alive feed parses with the validator's reader rules
and generated_at is younger than --max-age-hours (default 6). Prints age and entry count only; never
the opaque preview, scores or any entry content.

Usage: ka_feed_check.py <feed.json> [--max-age-hours 6]
Exit: 0 fresh and valid, 1 invalid/missing/stale (reason printed), 2 usage.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reax_subnet import ka_feed  # noqa: E402


def main(argv=None, now=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--max-age-hours", type=float, default=6.0)
    args = ap.parse_args(argv)
    now = time.time() if now is None else now
    result = ka_feed.load_feed(args.path, now)
    if not result.ok:
        print(f"FAIL: {result.why}")
        return 1
    age = (now - result.generated_at) / 3600
    print(f"age_hours={age:.2f} entries={len(result.rows)}")
    if age >= args.max_age_hours:
        print(f"FAIL: older than {args.max_age_hours:g} hours")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
