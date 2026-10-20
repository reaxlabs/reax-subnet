"""Router-side reading of 1.2.0rc10 commitments (multi-model and dual-role).

A port of the minimal logic of the validator's ``elements`` module (p2p-wip: ``parse_inference``, ``kind_of``,
``split_elements``, ``slot_key``); keep it in sync, do not import p2p-wip. Differences, all on the safe side: a competition
element (``reax-spec/1:`` / ``reax-eng/1:``) is recognised by its prefix only, because the router never uses it; a hotkey
with a malformed competition element is still gated by the validator's score (no score role, no routing).
"""
from __future__ import annotations

from typing import Optional, Sequence

PROTOCOL_S1 = "reax-s1/1"
PROTOCOL_S1M = "reax-s1m/1"
COMPETITION_PREFIXES = ("reax-spec/1:", "reax-eng/1:")
MAX_FIELDS = 3
MAX_FIELD_BYTES = 128
MODEL_ORDER = ("s1-fast", "s1-pro", "s1-lite", "s1-cpu")
MODEL_INDEX = {m: i for i, m in enumerate(MODEL_ORDER)}
SLOT_STRIDE = 1 << 16


def slot_key(uid: int, model: str) -> int:
    return uid + SLOT_STRIDE * MODEL_INDEX[model]


def _manifest_id_ok(mid: str) -> bool:
    return len(mid) == 16 and all(c in "0123456789abcdef" for c in mid)


def parse_inference(text) -> Optional[dict]:
    """An inference element -> {model: manifest_id} (1 entry for ``reax-s1/1``, 2-4 for ``reax-s1m/1``); else None."""
    if not isinstance(text, str) or not text.isascii() or len(text.encode()) > MAX_FIELD_BYTES:
        return None
    head, sep, body = text.strip().partition(":")
    if not sep:
        return None
    if head == PROTOCOL_S1:
        model, sep, mid = body.partition(":")
        if not sep or model not in MODEL_INDEX or not _manifest_id_ok(mid):
            return None
        return {model: mid}
    if head != PROTOCOL_S1M:
        return None
    out: dict = {}
    for part in body.split(","):
        model, sep, mid = part.partition(":")
        if not sep or model not in MODEL_INDEX or model in out or not _manifest_id_ok(mid):
            return None
        out[model] = mid
    return out if 2 <= len(out) <= len(MODEL_ORDER) else None


def _started(start: Optional[int], block: int) -> bool:
    return start is not None and block >= start


def inference_models(fields: Sequence[str], block: int, multimodel_start: Optional[int] = None,
                     dualrole_start: Optional[int] = None) -> dict:
    """{model: manifest_id} the commitment serves at ``block``; {} = no inference role (invalid, held or competition only).

    Same rules as the validator's ``split_elements``: one field = today's format (a multi-model element only from
    ``multimodel_start``); two fields only from ``dualrole_start`` and then exactly one inference and one competition
    element, of which only the inference element is read. Anything else (empty, 3+ fields, unknown or duplicate kinds,
    a field that is no element) is invalid."""
    fields = tuple(fields)
    if not 1 <= len(fields) <= MAX_FIELDS:
        return {}
    inference: dict = {}
    competition = False
    n_inf = 0
    for f in fields:
        if not isinstance(f, str):
            return {}
        f = f.strip()
        if len(f.encode("utf-8", "replace")) > MAX_FIELD_BYTES:
            return {}
        if f.startswith((PROTOCOL_S1 + ":", PROTOCOL_S1M + ":")):
            models = parse_inference(f)
            if models is None or n_inf:
                return {}
            n_inf, inference = 1, models
            multi = f.startswith(PROTOCOL_S1M + ":")
        elif f.startswith(COMPETITION_PREFIXES):
            if competition:
                return {}
            competition = True
        else:
            return {}
    if not n_inf:
        return {}
    if multi and not _started(multimodel_start, block):
        return {}  # held: a multi-model element has no role (validator: multi_held), alone or next to a competition field
    if len(fields) == 1:
        return inference
    if not _started(dualrole_start, block) or len(fields) != 2 or not competition:
        return {}
    return inference
