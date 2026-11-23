"""Emission follows need: the operator file for our decisions between releases (docs/EMISSION-NEED.md).

The inference share and the pauses of the competition tracks are our decisions (Florian 26 Oct 2026: principle plus our
discretion, no automatic thresholds). The release constants (competition_split.INFERENCE_SHARE_STEPS, TRACK_PAUSES) hold
the defaults; a small JSON file next to the award approvals (``<dir of REAX_AWARD_APPROVALS>/need.json``) adds dated
decisions without a new image:

    {"inference": [{"from_utc": "2026-11-03T14:30:00+00:00", "share": 0.35, "reason": "..."}],
     "pauses":    [{"track": "engine", "from_utc": "...", "until_utc": null, "reason": "..."}]}

``inference``: a step from ``from_utc`` on (the latest step at or before now wins, file or release). ``pauses``: ``track``
is "engine" or a specialist track id; paused while ``from_utc <= now < until_utc`` (null = until a later file says
otherwise). Strict: exactly these keys, ISO times with a timezone, no time before NEED_START_UTC, shares inside
INFERENCE_SHARE_SANE, at most MAX_ROWS rows each; anything else and the WHOLE file is ignored (fail closed to the release
constants, logged). An absent file is a no-op. Pure apart from reading the file.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, Sequence

from . import competition_split as _cs

log = logging.getLogger(__name__)

FILE_NAME = "need.json"
MAX_ROWS = 20
MAX_BYTES = 16384
ENGINE = "engine"   # the pause key of the engine track


def path_for(approvals_path: Optional[str]) -> str:
    """The file next to the award approvals file; "" without one."""
    return os.path.join(os.path.dirname(approvals_path), FILE_NAME) if approvals_path else ""


def _time(v: Any) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(v) if isinstance(v, str) else None
    except ValueError:
        return None
    return t if t is not None and t.tzinfo is not None else None


def _reason(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip()) and len(v) <= 300


def parse(raw: Any) -> tuple[list[tuple[datetime, float]], list[tuple[str, datetime, Optional[datetime]]], str]:
    """(inference steps, pauses, problem): the validated rows, or empty lists and why the content is refused."""
    bad: tuple[list, list] = ([], [])
    if not isinstance(raw, dict) or not set(raw) <= {"inference", "pauses"}:
        return *bad, "must be an object with the keys inference and/or pauses"
    steps: list[tuple[datetime, float]] = []
    pauses: list[tuple[str, datetime, Optional[datetime]]] = []
    inf, pz = raw.get("inference", []), raw.get("pauses", [])
    if not isinstance(inf, list) or not isinstance(pz, list) or len(inf) > MAX_ROWS or len(pz) > MAX_ROWS:
        return *bad, f"inference and pauses must be lists of at most {MAX_ROWS} rows"
    lo, hi = _cs.INFERENCE_SHARE_SANE
    for i, r in enumerate(inf):
        if not isinstance(r, dict) or set(r) != {"from_utc", "share", "reason"}:
            return *bad, f"inference row {i}: keys must be exactly from_utc, share, reason"
        t, v = _time(r["from_utc"]), r["share"]
        if t is None or t < _cs.NEED_START_UTC:
            return *bad, f"inference row {i}: from_utc must be an ISO time with a timezone, not before the start"
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            return *bad, f"inference row {i}: share must be a number in [{lo}, {hi}]"
        if not _reason(r["reason"]):
            return *bad, f"inference row {i}: reason must be a non-empty string of at most 300 characters"
        steps.append((t, float(v)))
    for i, r in enumerate(pz):
        if not isinstance(r, dict) or set(r) != {"track", "from_utc", "until_utc", "reason"}:
            return *bad, f"pause row {i}: keys must be exactly track, from_utc, until_utc, reason"
        key, a, b = r["track"], _time(r["from_utc"]), r["until_utc"]
        if not isinstance(key, str) or not key or len(key) > 64:
            return *bad, f"pause row {i}: track must be 'engine' or a track id"
        if a is None or a < _cs.NEED_START_UTC:
            return *bad, f"pause row {i}: from_utc must be an ISO time with a timezone, not before the start"
        until = None if b is None else _time(b)
        if b is not None and (until is None or until <= a):
            return *bad, f"pause row {i}: until_utc must be null or an ISO time with a timezone after from_utc"
        if not _reason(r["reason"]):
            return *bad, f"pause row {i}: reason must be a non-empty string of at most 300 characters"
        pauses.append((key, a, until))
    return steps, pauses, ""


_last_good: dict[str, tuple[list[tuple[datetime, float]], list[tuple[str, datetime, Optional[datetime]]]]] = {}


def load(path: str) -> tuple[list[tuple[datetime, float]], list[tuple[str, datetime, Optional[datetime]]]]:
    """The validated rows of ``path`` (absent: none). A refused or unreadable file (for example half written) keeps the
    last good content of this process for ``path`` (critic M3: a typo must not un-pause a paused track), else none; both
    with a warning. Write the file atomically (write a temp file, then rename)."""
    if not path or not os.path.exists(path):
        _last_good.pop(path, None)
        return [], []
    try:
        if os.path.getsize(path) > MAX_BYTES:
            raise ValueError(f"larger than {MAX_BYTES} bytes")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        problem = ""
    except (OSError, ValueError) as exc:
        raw, problem = None, str(exc)
    if not problem:
        steps, pauses, problem = parse(raw)
        if not problem:
            _last_good[path] = (steps, pauses)
            return steps, pauses
    kept = _last_good.get(path)
    log.warning("emission follows need: %s refused (%s); %s", path, problem,
                "the last good content stays in force" if kept else "the release constants apply")
    return kept if kept else ([], [])


def decide(now: datetime, path: str, track_ids: Sequence[str]) -> tuple[Optional[float], list[str]]:
    """(inference share or None when the rule is not active, paused keys among ``ENGINE`` and ``track_ids``) at ``now``."""
    share, starts = decide_with_starts(now, path, track_ids)
    return share, list(starts)


def decide_with_starts(now: datetime, path: str, track_ids: Sequence[str]) -> tuple[Optional[float], dict[str, float]]:
    """Like decide, retaining each continuous pause's from_utc for champion run-out."""
    if not _cs.need_active(now):
        return None, {}
    steps, pauses = load(path)
    share = _cs.inference_share(now, steps)
    known = (ENGINE, *track_ids)
    for key in sorted({k for k, _, _ in pauses} - set(known)):
        log.warning("emission follows need: %s pauses %r, which is not 'engine' or an active track; ignored", path, key)
    paused = {k: at.timestamp() for k in known if (at := _cs.pause_start(k, now, pauses)) is not None}
    if paused:
        log.info("emission follows need: paused %s; inference share %.3f", paused, share if share is not None else -1)
    return share, paused
