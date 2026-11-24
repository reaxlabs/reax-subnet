"""Set-bound Round 2 podium record: one JSON line per submitted weight set (consumer: the money monitor's I-GR2-POD).

``<validator state dir>/gr2-podium-sets.jsonl`` (the directory of ``REAX_STATE_PATH``, i.e. ``/state`` in the container and
``/w/validator-state`` on rxnode-03; NOT ``REAX_COMPETITIONS_STATE_DIR``, which the validator mounts read-only). Exactly::

    {"v":1,"set_block":int,"window":str|null,"fragment_sha256":hex|null,"source_block":int|null,"pool":float,
     "keepalive":float,"eligible":{slot:uid|null} (every slot of the window table),"podium":{"<uid>":float},"burn":float,
     "status":"paid"|"burned:<reason>"|"no-window","vector_sha256":hex}

plus ``"submitted":true|false`` on every line (false: the chain call of that set failed).

Amounts are weight units of the vector before the miner-slot factor and before chain u16 encoding: fractions of the whole
vector, i.e. the miner part (1 - owner burn fraction) is already applied. ``pool`` is the Generalist slot, ``keepalive`` the
actual Generalist keep-alive and provisional pay taken from it, ``podium`` the paid amount per UID
(``(pool - keepalive) x window share``) and ``burn`` the part of ``pool - keepalive`` whose slots are not paid
(``(pool - keepalive) x sum of the unpaid shares``). ``vector_sha256`` is the sha256 of the canonical JSON
(``json.dumps({str(uid): weight}, sort_keys=True, separators=(",", ":"))``, Python float repr, keys sorted as strings) of the
final float vector handed to ``set_weights``, i.e. AFTER the miner-slot factor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Mapping, Optional

log = logging.getLogger("reax.gr2_podium")

FILE_NAME = "gr2-podium-sets.jsonl"
VERSION = 1


def path_for(state_path: str) -> str:
    """The record file next to the validator's own state file (writable; the competitions state mount is read-only)."""
    return os.path.join(os.path.dirname(os.path.abspath(state_path)), FILE_NAME)


def vector_sha256(weights: Mapping[int, float]) -> str:
    doc = {str(int(uid)): float(w) for uid, w in weights.items()}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _all_slots(window: Optional[str], eligible: Mapping[str, Any]) -> dict[str, Any]:
    """Every slot of the window's share table, null when vacant (the monitor treats a missing former slot that is paid as a finding)."""
    from . import gr2_podium
    table = gr2_podium.OPENING_SHARES if window == gr2_podium.OPENING_WINDOW else (
        gr2_podium.STANDARD_SHARES if window in gr2_podium.LATER_WINDOWS else {})
    out: dict[str, Any] = {slot: None for slot in table}
    out.update({str(k): v for k, v in eligible.items()})
    return out


def partial(kind: str, reason: Optional[str], window: Optional[str], sha256: Optional[str], source_block: Optional[int],
            used: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the record that comes from the weight computation (before the set block and the submitted vector).

    ``used`` is what ``combine_weights`` filled (empty when the generalist track had no slot)."""
    podium = {str(uid): float(w) for uid, w in sorted((used.get("podium") or {}).items())}
    out: dict[str, Any] = {
        "window": window, "fragment_sha256": sha256, "source_block": source_block,
        "pool": float(used.get("pool", 0.0)), "keepalive": float(used.get("keepalive", 0.0)),
        "eligible": _all_slots(window, used.get("eligible") or {}), "podium": podium, "burn": float(used.get("burn", 0.0)),
    }
    if kind == "no-window":
        out.update(window=None, fragment_sha256=None, status="no-window")
    elif kind == "burn":
        out.update(fragment_sha256=None, status=f"burned:{reason}")
    elif not used.get("window"):
        out.update(status="burned:no_generalist_slot")   # the generalist track held no slot in this computation
    else:
        out.update(status="paid" if podium else "burned:no_eligible_slot")
    return out


def build(part: Mapping[str, Any], set_block: int, weights: Mapping[int, float], submitted: bool = True) -> dict[str, Any]:
    """The full line for the set at ``set_block`` that submitted ``weights``."""
    rec: dict[str, Any] = {"v": VERSION, "set_block": int(set_block)}
    for key in ("window", "fragment_sha256", "source_block", "pool", "keepalive", "eligible", "podium", "burn", "status"):
        rec[key] = part[key]
    rec["vector_sha256"] = vector_sha256(weights)
    rec["submitted"] = bool(submitted)   # explicit on every line: the monitor binds only submitted:true lines
    return rec


def append(path: str, record: Mapping[str, Any]) -> bool:
    """Append one line. Never raises: a failed write is logged (the monitor then sees no record for the set)."""
    try:
        line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except Exception as exc:  # noqa: BLE001 - observational; never stops weight setting
        log.error("generalist podium record for set %s not written: %r", record.get("set_block"), exc)
        return False
