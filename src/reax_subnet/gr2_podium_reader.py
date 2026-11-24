"""Round 2 Generalist podium: the public fragment reader (feature flag ``REAX_GR2_PODIUM``, default OFF).

Transport (confirmed 14 Nov 2026): the generalist row of ``https://reax.co/data/miners/competitions.json`` carries
``podium_window`` (the Thursday window id, which keeps its name through a move, reveal or rerun) and ``podium_sha256``
(sha256 of the file at ``https://reax.co/data/miners/podium/<window>.json``). A rerun replaces the file at the same path and
the row gets a new sha: the sha, not the window, identifies the fragment.

Rules of this reader (nothing is inferred, nothing is stale):

* the sha256 of the RAW bytes is checked before anything is parsed;
* a missing file (404), an unreachable host, a sha mismatch, a missing or mistyped field, a wrong share table, an unknown
  window, a duplicate UID or holder, inconsistent title tenures or any ``problems`` entry other than the documented
  coldkey-holder note makes the result a typed ``burn`` with a reason: the podium share burns and the weights are still set;
* a verified fragment is cached by its sha only; the row is re-read for every weight set, so a new sha is always fetched;
* the window can never move back (``window_regressed``), the file must name the row's window (``window_mismatch``);
* no row, no podium: a reachable row without ``podium_window`` is ``no-window`` (the ordinary rank path is unchanged).

A slot is payable at the instant ``t`` iff its record is not held, ``effective_from`` is set and <= t and
``effective_until`` is null or later; the 70 % title share follows ``podium.title_tenures`` (the row with
``effective_from`` <= t < ``effective_until``, or open), never ``champion.effective_until``. The caller still resolves the
current chain binding (``resolve``) and applies the award gate (``competition_split.combine_weights``).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from . import gr2_podium as _pod
from .ka_feed import timestamp as _timestamp

log = logging.getLogger("reax.gr2_podium")

FEATURE_ENV = "REAX_GR2_PODIUM"
COMPETITIONS_URL = "https://reax.co/data/miners/competitions.json"
FRAGMENT_URL = "https://reax.co/data/miners/podium/{window}.json"
TRACK_ID = "generalist"
FRAGMENT_SCHEMA = "reax-round2-award-fragment/1"
PODIUM_SCHEMA = "reax-round2-podium/1"
MAX_ROW_BYTES = 8 * 1024 * 1024
MAX_FRAGMENT_BYTES = 512 * 1024
FETCH_TIMEOUT_S = 5.0
MAX_AGE_S = 300.0          # a row snapshot older than this is re-read before a weight set uses it
CACHE_ENTRIES = 16
SLOTS = ("champion", "former[0]", "former[1]", "place2", "place3")
_WINDOW = re.compile(r"\d{4}-\d{2}-\d{2}")
_SHA = re.compile(r"[0-9a-f]{64}")
# The one `problems` line the contract allows (AWARD-FRAGMENT.md section 6): intake gave no holder id, the coldkey was used.
_COLDKEY_NOTE = re.compile(r"holder of podium\.[a-z0-9_\[\]]+(?:, podium\.[a-z0-9_\[\]]+)* is the coldkey: "
                           r"the intake service exposes no holder id")


def enabled(env: Mapping[str, str]) -> bool:
    """The feature flag. OFF unless explicitly switched on; OFF means no fetch, no record, the old vector."""
    return str(env.get(FEATURE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


class PodiumError(Exception):
    """The podium cannot be used; ``reason`` is a stable token (letters, digits, ``_`` and ``.``) for the record."""

    def __init__(self, reason: str, window: Optional[str] = None):
        super().__init__(reason)
        self.reason = reason
        self.window = window


class FetchError(Exception):
    """kind: ``not_found`` (HTTP 404), ``too_large`` or ``unavailable`` (anything else)."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # a redirect is an error, never a different file
        return None


def http_fetch(url: str, max_bytes: int, timeout: float = FETCH_TIMEOUT_S, allowed_prefix: str = "https://reax.co/") -> bytes:
    """GET ``url`` (only below ``allowed_prefix``, no redirects, at most ``max_bytes``). Raises ``FetchError``."""
    if not url.startswith(allowed_prefix):
        raise FetchError("unavailable")
    request = urllib.request.Request(url, headers={"User-Agent": "reax-validator", "Accept": "application/json",
                                                   "Cache-Control": "no-cache"})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            if response.status != 200:
                raise FetchError("unavailable")
            data = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        exc.close()
        raise FetchError("not_found" if exc.code == 404 else "unavailable") from None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        raise FetchError("unavailable") from None
    if len(data) > max_bytes:
        raise FetchError("too_large")
    return data


def _loads(raw: bytes, reason: str) -> Any:
    """Strict JSON: UTF-8, no duplicate keys, no NaN/Infinity."""
    def pairs(items):
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate key")
            out[key] = value
        return out

    def constant(name):
        raise ValueError("non-finite number")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise PodiumError(reason) from None


def _int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _need(doc: Any, key: str, path: str) -> Any:
    if not isinstance(doc, dict) or key not in doc:
        raise PodiumError(f"missing_field.{path}.{key}")
    return doc[key]


def _typed(ok: bool, path: str) -> None:
    if not ok:
        raise PodiumError(f"bad_type.{path}")


def _time(value: Any, path: str) -> float:
    _typed(isinstance(value, str), path)
    try:
        return _timestamp(value)
    except ValueError:
        raise PodiumError(f"bad_time.{path}") from None


def _optional_time(value: Any, path: str) -> Optional[float]:
    return None if value is None else _time(value, path)


# ---------------------------------------------------------------------------------------------------------- the row
def parse_row(raw: bytes) -> tuple[Optional[str], Optional[str]]:
    """``(podium_window, podium_sha256)`` of the generalist row of competitions.json; ``(None, None)`` = the row names no
    window. Raises ``PodiumError`` for a file without exactly one generalist row, a half-set pair or a malformed value."""
    doc = _loads(raw, "row_invalid")
    tracks = doc.get("tracks") if isinstance(doc, dict) else None
    if not isinstance(tracks, list):
        raise PodiumError("row_invalid")
    rows = [t for t in tracks if isinstance(t, dict) and t.get("id") == TRACK_ID]
    if len(rows) != 1:
        raise PodiumError("row_missing" if not rows else "row_ambiguous")
    window, sha = rows[0].get("podium_window"), rows[0].get("podium_sha256")
    if window is None and sha is None:
        return None, None
    if window is None or sha is None:
        raise PodiumError("row_incomplete")
    if not isinstance(window, str) or not _WINDOW.fullmatch(window):
        raise PodiumError("window_invalid")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise PodiumError("sha_invalid", window)
    return window, sha


# ----------------------------------------------------------------------------------------------------- the fragment
@dataclass(frozen=True)
class Record:
    slot: str
    uid: int
    hotkey: str
    repo: str
    revision: str
    held: bool
    start: Optional[float]
    end: Optional[float]
    holder_ids: frozenset


@dataclass(frozen=True)
class Tenure:
    uid: int
    hotkey: str
    start: float
    end: Optional[float]
    state: str


@dataclass(frozen=True)
class Pick:
    """A payable slot at one instant, before the chain binding and the award gate."""
    slot: str
    uid: int
    hotkey: str
    repo: str
    revision: str


@dataclass(frozen=True)
class Fragment:
    """A verified, validated award fragment (only what payment needs; scores and private fields are never kept)."""
    window: str
    sha256: str
    shares: Mapping[str, float]
    records: Mapping[str, Record]
    tenures: tuple[Tenure, ...]

    def eligible_at(self, ts: float) -> dict[str, Pick]:
        """The payable slots (positive share in this window) at unix time ``ts``."""
        out: dict[str, Pick] = {}

        def pick(slot: str, rec: Record) -> Pick:
            return Pick(slot, rec.uid, rec.hotkey, rec.repo, rec.revision)

        if self.shares.get("champion"):
            active = [t for t in self.tenures if t.start <= ts and (t.end is None or ts < t.end)]
            if len(active) == 1:
                holder = next((r for r in self.records.values() if r.uid == active[0].uid), None)
                if holder is not None and not holder.held:
                    out["champion"] = pick("champion", holder)
        for slot in ("former[0]", "former[1]", "place2", "place3"):
            rec = self.records.get(slot)
            if not self.shares.get(slot) or rec is None or rec.held or rec.start is None or rec.start > ts:
                continue
            if rec.end is None or ts < rec.end:
                out[slot] = pick(slot, rec)
        return out


def _record(raw: Any, slot: str, path: str) -> Record:
    _typed(isinstance(raw, dict), path)
    uid = _need(raw, "uid", path)
    _typed(_int(uid), f"{path}.uid")
    hotkey = _need(raw, "hotkey", path)
    _typed(isinstance(hotkey, str) and bool(hotkey), f"{path}.hotkey")
    repo, revision = _need(raw, "repo", path), _need(raw, "revision", path)
    _typed(isinstance(repo, str) and bool(repo), f"{path}.repo")
    _typed(isinstance(revision, str) and bool(revision), f"{path}.revision")
    held = _need(raw, "held", path)
    _typed(type(held) is bool, f"{path}.held")
    start = _optional_time(_need(raw, "effective_from", path), f"{path}.effective_from")
    end = _optional_time(_need(raw, "effective_until", path), f"{path}.effective_until")
    ids = raw.get("holder_ids")
    if ids is not None:
        _typed(isinstance(ids, list) and all(isinstance(i, str) and i for i in ids), f"{path}.holder_ids")
    holder = raw.get("holder")
    if holder is not None:
        _typed(isinstance(holder, str), f"{path}.holder")
    keys = frozenset(ids or ([holder] if holder else [f"uid:{uid}"]))
    return Record(slot, uid, hotkey, repo, revision, held, start, end, keys)


def parse_fragment(raw: bytes, window: str, sha256: str) -> Fragment:
    """Verify ``raw`` against ``sha256`` (before parsing), then validate it as the fragment of ``window``.

    Ports the checks of the kit's ``award.validate_podium`` that matter for payment (field types, consecutive title
    tenures, former places start where their title tenure ends, a held record is not paid, a pending winner leaves the
    beaten champion's tenure open) plus one place per holder among the records present. Raises ``PodiumError``."""
    if len(raw) > MAX_FRAGMENT_BYTES:
        raise PodiumError("fragment_too_large", window)
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise PodiumError("sha_mismatch", window)
    try:
        return _validated(_loads(raw, "fragment_not_json"), window, sha256)
    except PodiumError as exc:
        exc.window = window
        raise


def _validated(doc: Any, window: str, sha256: str) -> Fragment:
    _typed(isinstance(doc, dict), "fragment")
    if _need(doc, "schema", "fragment") != FRAGMENT_SCHEMA or _need(doc, "track_id", "fragment") != TRACK_ID:
        raise PodiumError("fragment_schema")
    if _need(doc, "window", "fragment") != window:
        raise PodiumError("window_mismatch")
    problems = _need(doc, "problems", "fragment")
    _typed(isinstance(problems, list) and all(isinstance(p, str) for p in problems), "fragment.problems")
    if any(not _COLDKEY_NOTE.match(p) for p in problems):
        raise PodiumError("fragment_problems")
    pd = _need(doc, "podium", "fragment")
    _typed(isinstance(pd, dict), "podium")
    if _need(pd, "schema", "podium") != PODIUM_SCHEMA:
        raise PodiumError("fragment_schema")
    if _need(pd, "window", "podium") != window:
        raise PodiumError("window_mismatch")
    shares = _need(pd, "shares", "podium")
    _typed(isinstance(shares, dict), "podium.shares")
    try:
        _pod.payout_fractions(window, shares, {})
    except ValueError as exc:
        raise PodiumError("unknown_window" if "unknown" in str(exc) else "wrong_shares") from None

    records: dict[str, Record] = {}
    for slot, key in (("champion", "champion"), ("place2", "place2"), ("place3", "place3")):
        value = _need(pd, key, "podium")
        if value is not None:
            records[slot] = _record(value, slot, f"podium.{key}")
    former = _need(pd, "former", "podium")
    _typed(isinstance(former, list) and len(former) == 2, "podium.former")
    for i, value in enumerate(former):
        if value is not None:
            records[f"former[{i}]"] = _record(value, f"former[{i}]", f"podium.former.{i}")
    pending_raw = pd.get("pending_champion")
    pending = _record(pending_raw, "pending_champion", "podium.pending_champion") if pending_raw is not None else None
    present = list(records.values()) + ([pending] if pending is not None else [])

    uids = [r.uid for r in present]
    if len(set(uids)) != len(uids):
        raise PodiumError("duplicate_uid")
    seen: set = set()
    for rec in present:
        if seen & rec.holder_ids:
            raise PodiumError("duplicate_holder")
        seen |= rec.holder_ids

    tenures: list[Tenure] = []
    rows = _need(pd, "title_tenures", "podium")
    _typed(isinstance(rows, list), "podium.title_tenures")
    for i, row in enumerate(rows):
        path = f"podium.title_tenures.{i}"
        _typed(isinstance(row, dict), path)
        uid, hotkey, state = _need(row, "uid", path), _need(row, "hotkey", path), _need(row, "state", path)
        _typed(_int(uid) and isinstance(hotkey, str) and bool(hotkey) and state in ("ended", "current"), path)
        start = _time(_need(row, "effective_from", path), f"{path}.effective_from")
        end = _optional_time(_need(row, "effective_until", path), f"{path}.effective_until")
        if (state == "current") != (end is None) or (end is not None and end < start):
            raise PodiumError("tenure_state")
        tenures.append(Tenure(uid, hotkey, start, end, state))
    for a, b in zip(tenures, tenures[1:]):
        if a.end != b.start:
            raise PodiumError("tenure_gap")
    for t in tenures:
        rec = next((r for r in present if r.uid == t.uid), None)
        if rec is not None and rec.hotkey != t.hotkey:
            raise PodiumError("tenure_hotkey")
        if t.state == "ended" and rec is not None and rec.slot.startswith("former") and rec.start != t.end:
            raise PodiumError("tenure_former_start")
    champion = records.get("champion")
    if tenures and champion is not None and tenures[-1].uid != champion.uid:
        raise PodiumError("tenure_champion")
    if pending is not None:
        if champion is not None and champion.end is not None:
            raise PodiumError("pending_tenure_closed")   # a private winner is pending: the beaten champion's tenure stays open
        if pending.start is not None and not (isinstance(pending_raw, dict) and pending_raw.get("revealed_at")):
            raise PodiumError("pending_paid")

    paid_shares = _need(pd, "paid_shares", "podium")
    _typed(isinstance(paid_shares, dict) and set(paid_shares) <= set(SLOTS), "podium.paid_shares")
    if any(r.held and r.slot in paid_shares for r in records.values()):
        raise PodiumError("held_but_paid")
    paid_total, burned_total = _need(pd, "paid_total", "podium"), _need(pd, "burned_total", "podium")
    _typed(all(type(v) in (int, float) and math.isfinite(v) for v in (paid_total, burned_total)), "podium.totals")
    if abs(paid_total + burned_total - 1.0) > 1e-9:
        raise PodiumError("totals_mismatch")
    return Fragment(window, sha256, dict(shares), records, tuple(tenures))


# ------------------------------------------------------------------------------------------------------- the reader
@dataclass(frozen=True)
class PodiumRead:
    """``kind``: ``ok`` (a verified fragment), ``burn`` (``reason``: the podium share burns) or ``no-window``."""
    kind: str
    reason: Optional[str] = None
    window: Optional[str] = None
    fragment: Optional[Fragment] = None

    @property
    def sha256(self) -> Optional[str]:
        return self.fragment.sha256 if self.fragment is not None else None


@dataclass
class _Snapshot:
    at: float
    read: PodiumRead


class PodiumReader:
    """Reads the row and the fragment. ``prefetch`` does the network I/O (call it off the event loop); ``read`` hands out
    the latest snapshot and falls back to a ``prefetch`` when there is none or it is older than ``max_age_s``. Never raises."""

    def __init__(self, fetch: Callable[[str, int], bytes] = http_fetch, *, competitions_url: str = COMPETITIONS_URL,
                 fragment_url: str = FRAGMENT_URL, clock: Callable[[], float] = time.monotonic,
                 max_age_s: float = MAX_AGE_S):
        self._fetch, self._row_url, self._fragment_url = fetch, competitions_url, fragment_url
        self._clock, self._max_age = clock, max_age_s
        self._lock = threading.Lock()
        self._snap: Optional[_Snapshot] = None
        self._cache: dict[str, Fragment] = {}
        self._high: Optional[str] = None     # the newest window this process has verified a fragment for

    def read(self) -> PodiumRead:
        snap = self._snap
        if snap is None or self._clock() - snap.at > self._max_age:
            return self.prefetch()
        return snap.read

    def prefetch(self) -> PodiumRead:
        with self._lock:
            try:
                result = self._load()
            except PodiumError as exc:
                result = PodiumRead("burn", exc.reason, exc.window)
            except Exception as exc:  # noqa: BLE001 - whatever the transport or a file does: burn, never raise
                log.error("generalist podium: unexpected %s", type(exc).__name__)
                result = PodiumRead("burn", "internal_error")
            self._snap = _Snapshot(self._clock(), result)
        if result.kind == "burn":
            log.warning("generalist podium BURNED: %s (window %s)", result.reason, result.window)
        return result

    def _get(self, url: str, limit: int, what: str, window: Optional[str] = None) -> bytes:
        try:
            return self._fetch(url, limit)
        except FetchError as exc:
            raise PodiumError(f"{what}_{exc.kind}", window) from None

    def _load(self) -> PodiumRead:
        window, sha = parse_row(self._get(self._row_url, MAX_ROW_BYTES, "row"))
        if window is None:
            return PodiumRead("no-window")
        if self._high is not None and window < self._high:
            raise PodiumError("window_regressed", window)
        fragment = self._cache.get(sha)
        if fragment is None:
            raw = self._get(self._fragment_url.format(window=window), MAX_FRAGMENT_BYTES, "fragment", window)
            fragment = parse_fragment(raw, window, sha)
            while len(self._cache) >= CACHE_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[sha] = fragment
        elif fragment.window != window:
            raise PodiumError("window_mismatch", window)
        self._high = window if self._high is None else max(self._high, window)
        return PodiumRead("ok", None, window, fragment)


# -------------------------------------------------------------------------------------------- the chain binding
@dataclass(frozen=True)
class Resolved:
    """What one weight set does with the podium. ``plan`` None = the ordinary Generalist rank path stays as it is."""
    kind: str
    reason: Optional[str]
    window: Optional[str]
    sha256: Optional[str]
    plan: Optional[_pod.PodiumPlan]
    dropped: tuple = field(default_factory=tuple)   # ((slot, why), ...) slots that were payable in the fragment but not bound


def resolve(read: PodiumRead, ts: float, neurons: Sequence[Any]) -> Resolved:
    """Turn a read into the plan of one weight set at unix time ``ts``.

    A slot is payable only while the UID still carries the fragment's hotkey and (when the record names it) the committed
    repo and revision of the fragment. Everything else, including the owner-approval gate (applied again in
    ``combine_weights``), burns the slot's own share."""
    from .spec.commitment import parse_commitment as parse_spec_commitment

    if read.kind == "no-window":
        return Resolved("no-window", None, None, None, None)
    if read.kind != "ok" or read.fragment is None:
        return Resolved("burn", read.reason or "unavailable", read.window, None, _pod.PodiumPlan())
    fragment = read.fragment
    by_uid = {n.uid: n for n in neurons}
    eligible: dict[str, int] = {}
    dropped = []
    for slot, pick in fragment.eligible_at(ts).items():
        neuron = by_uid.get(pick.uid)
        if neuron is None or neuron.hotkey != pick.hotkey:
            dropped.append((slot, "not_bound"))
            continue
        commitment = parse_spec_commitment(neuron.commitment)
        if (commitment is None or commitment.track_id != TRACK_ID
                or (commitment.repo, commitment.revision) != (pick.repo, pick.revision)):
            dropped.append((slot, "revision_changed"))
            continue
        eligible[slot] = pick.uid
    try:
        _pod.payout_fractions(fragment.window, fragment.shares, eligible)
    except ValueError:      # e.g. one UID in two slots at this instant
        return Resolved("burn", "invalid_eligibility", fragment.window, fragment.sha256, _pod.PodiumPlan(), tuple(dropped))
    return Resolved("ok", None, fragment.window, fragment.sha256,
                    _pod.PodiumPlan(fragment.window, dict(fragment.shares), eligible), tuple(dropped))
