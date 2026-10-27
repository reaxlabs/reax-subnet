"""Operator keep-alive override (waiting-state hotfix, 26 Oct 2026; Florian's decision for Ichigeki, DECISIONS
ichigeki-keepalive-restore-new-hotkey).

A small JSON file next to the award approvals (``<dir of REAX_AWARD_APPROVALS>/ka-override.json``) lists hotkeys that get
a keep-alive by an explicit operator decision, within every existing cap:

    [{"hotkey": "<ss58>", "kind": "legacy" | "waiting", "until_utc": "2026-10-28T13:00:00+00:00", "reason": "..."}]

``legacy``: paid like the ended DF/RA tracks' legacy keep-alive (``combine_weights`` ``legacy_keepalive``: pure entrants
only, LEGACY_KA_MAX_ENTRIES, LEGACY_KA_MAX_SHARE). ``waiting``: added to the generalist provisional keep-alive candidates
(PROVISIONAL_KA_MAX_SHARE, one per coldkey and track). A row applies while ``now < until_utc`` and its hotkey holds a uid.
Strict: at most MAX_ROWS rows, exactly these keys, ``until_utc`` timezone-aware and at most MAX_DAYS ahead of ``now``;
anything else and the WHOLE file is ignored (fail closed, logged). An absent file is a no-op. Pure apart from reading the
file."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

log = logging.getLogger(__name__)

FILE_NAME = "ka-override.json"
MAX_ROWS = 5
MAX_DAYS = 8
MAX_BYTES = 8192
KINDS = ("legacy", "waiting")
_SS58 = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,48}$")


def path_for(approvals_path: Optional[str]) -> str:
    """The override file next to the award approvals file; "" without one."""
    return os.path.join(os.path.dirname(approvals_path), FILE_NAME) if approvals_path else ""


def parse(raw: Any, now: datetime) -> tuple[list[dict[str, Any]], str]:
    """(rows, problem): the validated rows, or [] and why the content is refused."""
    if not isinstance(raw, list) or len(raw) > MAX_ROWS:
        return [], f"must be a list of at most {MAX_ROWS} rows"
    rows = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict) or set(r) != {"hotkey", "kind", "until_utc", "reason"}:
            return [], f"row {i}: keys must be exactly hotkey, kind, until_utc, reason"
        hk, kind, until, why = r["hotkey"], r["kind"], r["until_utc"], r["reason"]
        if not isinstance(hk, str) or not _SS58.match(hk):
            return [], f"row {i}: hotkey is not an ss58 address"
        if kind not in KINDS:
            return [], f"row {i}: kind must be one of {KINDS}"
        if not isinstance(why, str) or not why.strip() or len(why) > 300:
            return [], f"row {i}: reason must be a non-empty string of at most 300 characters"
        try:
            t = datetime.fromisoformat(until) if isinstance(until, str) else None
        except ValueError:
            t = None
        if t is None or t.tzinfo is None:
            return [], f"row {i}: until_utc must be an ISO time with a timezone"
        if t > now + timedelta(days=MAX_DAYS):
            return [], f"row {i}: until_utc is more than {MAX_DAYS} days ahead"
        rows.append({"hotkey": hk, "kind": kind, "until": t, "reason": why.strip()})
    return rows, ""


def load(path: str, now: datetime) -> list[dict[str, Any]]:
    """The validated rows of ``path`` (absent: []); a refused file gives [] and a warning."""
    if not path or not os.path.exists(path):
        return []
    try:
        if os.path.getsize(path) > MAX_BYTES:
            raise ValueError(f"larger than {MAX_BYTES} bytes")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        log.warning("keep-alive override %s ignored: %s", path, exc)
        return []
    rows, problem = parse(raw, now)
    if problem:
        log.warning("keep-alive override %s ignored: %s", path, problem)
    return rows


def apply(rows: Sequence[Mapping[str, Any]], neurons: Sequence[Any], now: datetime) -> tuple[list[int], list[int]]:
    """(legacy uids, waiting uids) of the rows in force at ``now`` whose hotkey holds a uid; each applied row is logged."""
    uid_of = {n.hotkey: n.uid for n in neurons}
    legacy: list[int] = []
    waiting: list[int] = []
    for r in rows:
        if now >= r["until"] or r["hotkey"] not in uid_of:
            continue
        uid = uid_of[r["hotkey"]]
        (legacy if r["kind"] == "legacy" else waiting).append(uid)
        log.info("keep-alive override: uid %d (%s...) %s until %s: %s", uid, r["hotkey"][:8], r["kind"],
                 r["until"].isoformat(), r["reason"])
    return legacy, waiting
