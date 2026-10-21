"""Commitments with several elements: all inference models and one competition entry on one hotkey.

docs/DESIGN-MULTIROLE.md. A commitment is 1-3 ``Raw`` fields of the Commitments pallet (finney: MaxFields 3, 128 bytes
per field); each field is exactly one element:

* inference, one model (unchanged):   ``reax-s1/1:<model>:<manifest_id>``
* inference, several models (new):    ``reax-s1m/1:<model>:<manifest_id>,<model>:<manifest_id>[,...]``
* one competition entry (unchanged):  ``reax-spec/1:...`` or ``reax-eng/1:...``

Parsed strictly: at most one inference and at most one competition element, no unknown, empty or duplicate field;
anything else makes the whole commitment invalid (no role), like a malformed commitment today. A one-field commitment is
exactly today's format. Multi-model elements count from ``MULTIMODEL_START_BLOCK`` and a second (competition) field from
``DUALROLE_START_BLOCK`` on (release constants, None = held): before that such a commitment has no role.

Each subsystem sees one element through ``role_views``: the serving rounds the inference element, the competition code
(unchanged) the competition element. Both views carry the chain's commitment block (the same for every validator); the
competition order that must survive an inference rewrite is the competition worker's own persisted first sighting
(spec ``first_seen_block``, engine ``first_seen`` = minimum per digest), which a rewrite never moves.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .chain import Neuron, Snapshot


# Release constants (consensus): the first block at which a validator accepts a multi-model inference element / a
# commitment with an inference AND a competition field. None = held (such commitments have no role).
MULTIMODEL_START_BLOCK: Optional[int] = 9331987
DUALROLE_START_BLOCK: Optional[int] = 9331987

PROTOCOL_S1 = "reax-s1/1"
PROTOCOL_S1M = "reax-s1m/1"
MAX_FIELDS = 3  # Commitments pallet MaxFields on finney (read 20 Oct 2026, block 9,324,949)
MAX_FIELD_BYTES = 128  # largest Raw variant

# Pool order: canonical order of a multi-model element and the slot index of each model (slot_key).
MODEL_ORDER = ("s1-fast", "s1-pro", "s1-lite", "s1-cpu")
MODEL_INDEX = {m: i for i, m in enumerate(MODEL_ORDER)}
# Scoring slot of (uid, model) = uid + SLOT_STRIDE x MODEL_INDEX[model]: the s1-fast slot of a UID is the UID itself.
SLOT_STRIDE = 1 << 16

INFERENCE = "inference"
COMPETITION = "competition"


def _spec_ok(text: str) -> bool:
    from .spec.commitment import parse_commitment

    return parse_commitment(text) is not None


def _eng_ok(text: str) -> bool:
    from .engcomp import parse_engine_commitment

    return parse_engine_commitment(text) is not None


# Competition element prefixes this release knows, with their strict parser. The generalist track adds its prefix here.
COMPETITION_PARSERS: dict[str, Callable[[str], bool]] = {"reax-spec/1:": _spec_ok, "reax-eng/1:": _eng_ok}


# ------------------------------------------------------------------------------------------------- inference elements
def _manifest_id_ok(mid: str) -> bool:
    return len(mid) == 16 and all(c in "0123456789abcdef" for c in mid)


def parse_inference(text: Any) -> Optional[dict[str, str]]:
    """An inference element -> {model: manifest_id} (one entry for ``reax-s1/1``, 2-4 for ``reax-s1m/1``); else None.

    Only the shape is checked here (known model names, 16-hex manifest ids, no duplicates); whether a manifest id is this
    release's is decided per model by the validator, so one stale model never voids the others."""
    if not isinstance(text, str) or not text.isascii() or len(text.encode()) > MAX_FIELD_BYTES:
        return None
    text = text.strip()
    head, sep, body = text.partition(":")
    if not sep:
        return None
    if head == PROTOCOL_S1:
        model, sep, mid = body.partition(":")
        if not sep or model not in MODEL_INDEX or not _manifest_id_ok(mid):
            return None
        return {model: mid}
    if head != PROTOCOL_S1M:
        return None
    out: dict[str, str] = {}
    for part in body.split(","):
        model, sep, mid = part.partition(":")
        if not sep or model not in MODEL_INDEX or model in out or not _manifest_id_ok(mid):
            return None
        out[model] = mid
    if not 2 <= len(out) <= len(MODEL_ORDER):
        return None  # a one-model element must use reax-s1/1
    return out


def inference_text(models: Mapping[str, str]) -> str:
    """The canonical inference element for {model: manifest_id}: ``reax-s1/1`` for one model, else ``reax-s1m/1`` in
    pool order."""
    if not models:
        raise ValueError("no model")
    unknown = [m for m in models if m not in MODEL_INDEX]
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(sorted(unknown))}")
    ordered = sorted(models, key=MODEL_INDEX.__getitem__)
    if len(ordered) == 1:
        text = f"{PROTOCOL_S1}:{ordered[0]}:{models[ordered[0]]}"
    else:
        text = PROTOCOL_S1M + ":" + ",".join(f"{m}:{models[m]}" for m in ordered)
    if parse_inference(text) is None:
        raise ValueError(f"not a valid inference element: {text!r}")
    return text


def is_multimodel(text: str) -> bool:
    return isinstance(text, str) and text.strip().startswith(PROTOCOL_S1M + ":")


# ------------------------------------------------------------------------------------------------------- commitments
def kind_of(text: str) -> Optional[str]:
    """INFERENCE / COMPETITION for a strictly valid element, else None."""
    if not isinstance(text, str) or not text or len(text.encode("utf-8", "replace")) > MAX_FIELD_BYTES:
        return None
    if text.startswith((PROTOCOL_S1 + ":", PROTOCOL_S1M + ":")):
        return INFERENCE if parse_inference(text) is not None else None
    for prefix, ok in COMPETITION_PARSERS.items():
        if text.startswith(prefix):
            return COMPETITION if ok(text) else None
    return None


@dataclass(frozen=True)
class Elements:
    inference: str = ""  # the inference element text ("" = none)
    competition: str = ""  # the competition element text ("" = none)


def split_elements(fields: Sequence[str], block: int) -> Optional[Elements]:
    """The elements of a commitment's fields as committed at ``block``; None = invalid (the hotkey has no role).

    A single field is today's format: any one element (a single-field ``reax-s1m/1`` counts from
    MULTIMODEL_START_BLOCK). Two or three fields: from DUALROLE_START_BLOCK, exactly one inference and one competition
    element (a third field is never valid today; MaxFields leaves room for a later element kind)."""
    fields = tuple(fields)
    if not 1 <= len(fields) <= MAX_FIELDS:
        return None
    found: dict[str, str] = {}
    for f in fields:
        if not isinstance(f, str):
            return None
        f = f.strip()  # as the one-element parsers always did
        kind = kind_of(f)
        if kind is None or kind in found:
            return None
        found[kind] = f
    inf = found.get(INFERENCE, "")
    comp = found.get(COMPETITION, "")
    # a well-formed multi-model element before its start has no role; it never voids a competition element next to it
    # (holding or rolling back multimodel must not withdraw entries)
    multi_held = bool(inf) and is_multimodel(inf) and not _started(MULTIMODEL_START_BLOCK, block)
    if len(fields) == 1:
        return None if multi_held else Elements(inference=inf, competition=comp)
    if not _started(DUALROLE_START_BLOCK, block) or len(found) != 2 or len(fields) != 2:
        return None
    return Elements(inference="" if multi_held else inf, competition=comp)


def _started(start: Optional[int], block: int) -> bool:
    return start is not None and block >= start


def fields_of(n: Neuron) -> tuple[str, ...]:
    """The commitment's Raw fields; a Neuron built without them (tests, old readers) has its text as one field."""
    fields = getattr(n, "commitment_fields", None)
    if fields:
        return tuple(fields)
    return (n.commitment,) if n.commitment else ()


def loose_kind(text: str) -> Optional[str]:
    """The kind of a field for merging: a valid element of this release, or an element of a LATER release (a prefix this
    release does not know: ``reax-s1...`` counts as inference, any other ``reax-`` as competition). A field with a prefix
    this release knows that does not parse is garbage (None): no release will ever read it (critic round 2 NEW-1).
    None = not an element at all."""
    kind = kind_of(text)
    if kind is not None:
        return kind
    known = (PROTOCOL_S1 + ":", PROTOCOL_S1M + ":", *COMPETITION_PARSERS)
    if not isinstance(text, str) or text.startswith(known):
        return None
    if not text.isprintable() or len(text.encode("utf-8", "replace")) > MAX_FIELD_BYTES:
        return None
    if text.startswith("reax-s1"):
        return INFERENCE
    if text.startswith("reax-"):
        return COMPETITION
    return None


def merge_fields(current: Sequence[str], element: str, *, drop: Optional[str] = None) -> list[str]:
    """The fields to write so that ``element`` replaces the element of its kind and the other kind's current element is
    kept verbatim (every write replaces all fields). ``drop`` = a kind to remove instead of keeping. Fields are compared
    stripped, as validators read them. A field of a later release (unknown ``reax-`` prefix) is kept like a known one: a
    writer never deletes an element it cannot read. Only fields that are no element at all (unreadable, not ``reax-``) are
    left out. Raises ValueError on an invalid ``element`` or when the current commitment holds two elements of the kind
    to keep (ambiguous: the caller must not guess which one the owner wants)."""
    element = element.strip() if isinstance(element, str) else element
    kind = kind_of(element)
    if kind is None:
        raise ValueError(f"not a valid element: {str(element)[:80]!r}")
    keep: list[str] = []
    for f in current:
        if not isinstance(f, str):
            continue
        f = f.strip()
        k = loose_kind(f)
        if k is None or k == kind or k == drop:
            continue
        keep.append(f)
    if len(keep) > 1:
        raise ValueError(f"the current commitment holds {len(keep)} elements of the other kind; refusing to guess")
    return ([element] + keep) if kind == INFERENCE else (keep + [element])


# ------------------------------------------------------------------------------------------------------ scoring slots
def slot_key(uid: int, model: str) -> int:
    return uid + SLOT_STRIDE * MODEL_INDEX[model]


def slot_uid(slot: int) -> int:
    return slot % SLOT_STRIDE


def slot_model(slot: int) -> str:
    return MODEL_ORDER[slot // SLOT_STRIDE]


# ------------------------------------------------------------------------------------------------------------ views
@dataclass(frozen=True)
class RoleViews:
    inference: Snapshot  # commitment = the inference element ("" = none)
    competition: Snapshot  # commitment = the competition element ("" = none)
    elements: dict[int, Optional[Elements]]  # uid -> parsed elements (None = invalid or no commitment)


def role_views(snap: Snapshot) -> RoleViews:
    """Split every neuron's commitment into its inference and competition element (see module doc).

    One-field commitments give exactly today's snapshot in the view of their kind (and "" in the other). Both views keep
    the chain's commitment block: deterministic, the same for every validator (critic round 1, F3)."""
    inf: list[Neuron] = []
    comp: list[Neuron] = []
    parsed: dict[int, Optional[Elements]] = {}
    for n in snap.neurons:
        fields = fields_of(n)
        el = split_elements(fields, snap.block) if fields else None
        parsed[n.uid] = el
        i_text = el.inference if el else ""
        c_text = el.competition if el else ""
        inf.append(dataclasses.replace(n, commitment=i_text, commitment_block=n.commitment_block if i_text else 0))
        comp.append(dataclasses.replace(n, commitment=c_text, commitment_block=n.commitment_block if c_text else 0))
    return RoleViews(Snapshot(snap.block, tuple(inf), snap.owner_hotkey),
                     Snapshot(snap.block, tuple(comp), snap.owner_hotkey), parsed)
