#!/usr/bin/env python3
"""Set the generalist release constants in the build (src/reax_subnet/competition_split.py).

    scripts/set_generalist_start.py START=2026-10-30T12:00:00Z [COMMIT_FROM=<utc>|none] [ENTRY_FLOOR=<float>|inf]
                                    [G_MED=<float>|none] [NONPROC_FLOOR=<float>|inf]
    scripts/set_generalist_start.py default      # back to the committed fail-closed defaults

Rewrites GENERALIST_START_UTC, GENERALIST_COMMIT_FROM_UTC, GENERALIST_ENTRY_FLOOR, GENERALIST_G_MED and
GENERALIST_KEEPALIVE_NONPROC_FLOOR (only the keys given) and asserts the result by re-reading the file.
Choosing the values is the release owner's decision (docs/GENERALIST-RELEASE.md).
"""
from __future__ import annotations

import math
import pathlib
import re
import sys
from datetime import datetime, timezone

TARGET = pathlib.Path(__file__).resolve().parent.parent / "src" / "reax_subnet" / "competition_split.py"
DEFAULTS = {"START": "2026-10-30T12:00:00Z", "COMMIT_FROM": "none", "ENTRY_FLOOR": "inf", "G_MED": "none",
            "NONPROC_FLOOR": "inf"}


def _utc(text: str) -> datetime:
    dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    return dt.replace(tzinfo=timezone.utc)


def _dt_src(dt: datetime) -> str:
    parts = [dt.year, dt.month, dt.day, dt.hour, dt.minute] + ([dt.second] if dt.second else [])
    return f"datetime({', '.join(map(str, parts))}, tzinfo=timezone.utc)"


def _current_start() -> datetime:
    m = re.search(r"^GENERALIST_START_UTC = datetime\(([\d, ]+), tzinfo=timezone\.utc\)$", TARGET.read_text(), flags=re.M)
    if m is None:
        raise ValueError("cannot read the current GENERALIST_START_UTC")
    nums = [int(x) for x in m.group(1).split(",")]
    return datetime(*nums, tzinfo=timezone.utc)


def _float(text: str, allow: str) -> float | None:
    if text.lower() == allow:
        return math.inf if allow == "inf" else None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


def build(args: dict[str, str]) -> dict[str, tuple[str, str]]:
    """name -> (annotation regex tail, new source line); also validates every value."""
    out: dict[str, tuple[str, str]] = {}
    start = commit = None
    if "START" not in args and "COMMIT_FROM" in args:
        start = _current_start()  # a lone COMMIT_FROM is checked against the START already in the file
    if "START" in args:
        start = _utc(args["START"])
        out["GENERALIST_START_UTC"] = (r"", f"GENERALIST_START_UTC = {_dt_src(start)}")
    if "COMMIT_FROM" in args:
        if args["COMMIT_FROM"].lower() == "none":
            out["GENERALIST_COMMIT_FROM_UTC"] = (r": Optional\[datetime\]", "GENERALIST_COMMIT_FROM_UTC: Optional[datetime] = None")
        else:
            commit = _utc(args["COMMIT_FROM"])
            out["GENERALIST_COMMIT_FROM_UTC"] = (r": Optional\[datetime\]",
                                                 f"GENERALIST_COMMIT_FROM_UTC: Optional[datetime] = {_dt_src(commit)}")
    if start is not None and commit is not None and commit > start:
        raise ValueError("COMMIT_FROM must not be after START")
    if "ENTRY_FLOOR" in args:
        v = _float(args["ENTRY_FLOOR"], "inf")
        out["GENERALIST_ENTRY_FLOOR"] = (r"", f"GENERALIST_ENTRY_FLOOR = {'math.inf' if v == math.inf else repr(v)}")
    if "NONPROC_FLOOR" in args:
        v = _float(args["NONPROC_FLOOR"], "inf")
        out["GENERALIST_KEEPALIVE_NONPROC_FLOOR"] = (
            r"", f"GENERALIST_KEEPALIVE_NONPROC_FLOOR = {'math.inf' if v == math.inf else repr(v)}")
    if "G_MED" in args:
        v = _float(args["G_MED"], "none")
        out["GENERALIST_G_MED"] = (r": Optional\[float\]", f"GENERALIST_G_MED: Optional[float] = {'None' if v is None else repr(v)}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        if argv[1] == "default":
            args = dict(DEFAULTS)
        else:
            args = {}
            for item in argv[1:]:
                key, sep, val = item.partition("=")
                if not sep or key.upper() not in DEFAULTS:
                    raise ValueError(f"bad argument {item!r}")
                args[key.upper()] = val.strip()
        edits = build(args)
    except ValueError as exc:
        print(f"{exc}\n{__doc__}", file=sys.stderr)
        return 2
    text = TARGET.read_text()
    for name, (ann, line) in edits.items():
        text, n = re.subn(rf"^{name}{ann} ?= .*$", line, text, flags=re.M)
        if n != 1:
            print(f"{name}: expected exactly one definition, found {n}", file=sys.stderr)
            return 1
    TARGET.write_text(text)
    after = TARGET.read_text()
    for name, (_, line) in edits.items():
        assert after.count(line + "\n") == 1, name
    print(f"{TARGET}: " + "; ".join(line for _, line in edits.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
