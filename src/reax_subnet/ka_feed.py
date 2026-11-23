"""Round-2 intake keep-alive sidecar. No scores or intake-state data enter this reader.

The JSON decoder discards the opaque diagnostic field at the decoding boundary:
its contents are syntax checked without constructing any Python value. Only the
validated payment projection can escape the reader or enter its memory cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
import re

log = logging.getLogger(__name__)
GR2_KA_FEED_START_UTC = datetime(2026, 11, 21, 15, tzinfo=timezone.utc)
# Round-1 approval survives recommitment, but never a hotkey replacement.
GR2_ROUND1_FORMER = {
    83: "5F3vZ89Kx7Yd8FipTqnJUrQ7MoysUeHdJZhy8enrmYCxQAnH",
    114: "5D5goWB72691wTobFgrT961FKJ2JVGDnNDgm9iaGrn3uifpp",
    116: "5GNAwERfaf3Rd5bFrg2JJvyo6L53AovkAv2HEALLkGNDxGhF",
    199: "5CwV6dNFeyNqY6FQmZm9bQdta3qWe2xGuFWtXrS198N5K6r1",
    250: "5DUJxXJpZrFKrTLNAWw3H6FT1Du1z2zLNzEg3vzG32t6WTos",
}
GR2_ROUND1_FORMER_UNTIL = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
FUTURE_TOLERANCE_S = 300
MAX_AGE_S = 26 * 3600
MAX_ENTRIES = 64
MAX_BYTES = 8 * 1024 * 1024
ROUND = 2
_OPAQUE = "preview"
_SS58 = r"5[1-9A-HJ-NP-Za-km-z]{46,47}"
_REPO = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
_SHA = r"[0-9a-f]{40}"
_STRING = re.compile(r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"')
_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})")


def active(now, env=None):
    env = os.environ if env is None else env
    return bool(env.get("REAX_KA_FEED_PATH")) and stamp(now) >= GR2_KA_FEED_START_UTC.timestamp()


def stamp(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timezone required")
        return value.timestamp()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite time required")
    return float(value)


def timestamp(value):
    if not isinstance(value, str) or not _TIME.fullmatch(value):
        raise ValueError("invalid timestamp")
    return datetime.fromisoformat(value.upper().replace("Z", "+00:00")).timestamp()


class _Decoder:
    """Small strict JSON decoder; opaque objects are skipped, never materialised."""
    def __init__(self, text):
        self.text, self.pos = text, 0

    def ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def take(self, token):
        self.ws()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def value(self, build=True, depth=0):
        if depth > 64:
            raise ValueError("JSON nesting limit")
        self.ws()
        if self.take("{"):
            out = {} if build else None
            seen = set() if build else None
            if self.take("}"):
                return out
            while True:
                self.ws()
                match = _STRING.match(self.text, self.pos)
                if match is None:
                    raise ValueError("invalid JSON key")
                self.pos = match.end()
                key = json.loads(match.group()) if build else None
                if build:
                    if key in seen:
                        raise ValueError("duplicate JSON key")
                    seen.add(key)
                if not self.take(":"):
                    raise ValueError("invalid JSON object")
                if build and key == _OPAQUE:
                    self.ws()
                    if not (self.text.startswith("{", self.pos) or self.text.startswith("null", self.pos)):
                        raise ValueError("invalid opaque field type")
                    self.value(False, depth + 1)
                    # Retain presence only, so the required-key contract is checked.
                    out[key] = None
                else:
                    val = self.value(build, depth + 1)
                    if build:
                        out[key] = val
                if self.take("}"):
                    return out
                if not self.take(","):
                    raise ValueError("invalid JSON object")
        if self.take("["):
            out = [] if build else None
            if self.take("]"):
                return out
            while True:
                val = self.value(build, depth + 1)
                if build:
                    out.append(val)
                if self.take("]"):
                    return out
                if not self.take(","):
                    raise ValueError("invalid JSON array")
        self.ws()
        match = _STRING.match(self.text, self.pos) or _NUMBER.match(self.text, self.pos)
        if match:
            self.pos = match.end()
            return json.loads(match.group()) if build else None
        for literal, val in (("true", True), ("false", False), ("null", None)):
            if self.take(literal):
                return val if build else None
        raise ValueError("invalid JSON value")

    def decode(self):
        out = self.value()
        self.ws()
        if self.pos != len(self.text):
            raise ValueError("trailing JSON data")
        return out


def _keys(obj, required, optional=()):
    if not isinstance(obj, dict) or not set(required) <= obj.keys() or obj.keys() - set(required) - set(optional):
        raise ValueError("invalid contract keys")


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer")


def _pattern(value, pattern):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError("invalid string pattern")


_SIDE_REQUIRED = ("repo", "revision", "repo_at_rev", "hotkey", "state", "since", "commitment_block",
                  "extrinsic_index", "commit_time", "disqualified", _OPAQUE)
_SIDE_OPTIONAL = ("private", "public_at", "reveal_due", "revealed_at", "void", "void_reason",
                  "objection_deadline", "waiting", "first_window", "keepalive_ends_at")
_STATES = ("intake", "needs_correction", "refused", "held", "pass_unscored", "overflow", "scored")


def _side(side):
    _keys(side, _SIDE_REQUIRED, _SIDE_OPTIONAL)
    for key, pattern in (("repo", _REPO), ("revision", _SHA), ("repo_at_rev", _REPO + "@" + _SHA), ("hotkey", _SS58)):
        _pattern(side[key], pattern)
    if side["state"] not in _STATES or type(side["disqualified"]) is not bool:
        raise ValueError("invalid state or boolean")
    for key in ("since", "commit_time", "public_at", "reveal_due", "revealed_at", "objection_deadline"):
        if key in side:
            timestamp(side[key])
    for key in ("commitment_block", "extrinsic_index"):
        _integer(side[key])
    if "private" in side and side["private"] is not None and type(side["private"]) is not bool:
        raise ValueError("invalid private flag")
    for key in ("void", "waiting"):
        if key in side and side[key] is not True:
            raise ValueError("invalid constant")
    if "void_reason" in side and side["void_reason"] != "not_revealed":
        raise ValueError("invalid reason")
    if "first_window" in side and side["first_window"] is not None and not isinstance(side["first_window"], str):
        raise ValueError("invalid window")
    if "keepalive_ends_at" in side and side["keepalive_ends_at"] is not None:
        timestamp(side["keepalive_ends_at"])
    # Payment projection only. The opaque key itself is also removed.
    return {k: side[k] for k in ("hotkey", "state", "commitment_block", "extrinsic_index", "commit_time",
                               "disqualified", "void", "keepalive_ends_at") if k in side}


@dataclass(frozen=True)
class FeedResult:
    ok: bool
    why: str = ""
    rows: tuple = ()
    generated_at: float = 0.0


def load_feed(path, now, expected_round=ROUND):
    """Fail closed without exposing file contents, field names or exception text."""
    try:
        now = stamp(now)
        with open(path, encoding="utf-8") as handle:
            text = handle.read(MAX_BYTES + 1)
        if len(text.encode("utf-8")) > MAX_BYTES:
            return FeedResult(False, "file too large")
        raw = _Decoder(text).decode()
        _keys(raw, ("schema", "generated_at", "round", "entries"))
        if raw["schema"] != "reax-keepalive-feed/1":
            return FeedResult(False, "wrong schema id")
        _integer(raw["round"], 1)
        if raw["round"] != expected_round:
            return FeedResult(False, "round mismatch")
        generated = timestamp(raw["generated_at"])
        if now - generated > MAX_AGE_S:
            return FeedResult(False, "feed older than 26 hours")
        if generated > now + FUTURE_TOLERANCE_S:
            return FeedResult(False, "feed timestamp in future")
        if not isinstance(raw["entries"], list):
            raise ValueError("invalid entries")
        rows, seen = [], set()
        for row in raw["entries"]:
            _keys(row, ("uid", "coldkey", "qualifies", "basis", "keepalive_entry", "newest_entry"), ("holder_id", "holder_ids"))
            _integer(row["uid"])
            if row["uid"] in seen:
                raise ValueError("duplicate uid")
            seen.add(row["uid"])
            _pattern(row["coldkey"], _SS58)
            if type(row["qualifies"]) is not bool or row["basis"] not in ("newest_entry", "entry_for_next_window", "none"):
                raise ValueError("invalid row enum or boolean")
            if "holder_id" in row and (not isinstance(row["holder_id"], str) or not row["holder_id"]):
                raise ValueError("invalid holder id")
            if "holder_ids" in row and (not isinstance(row["holder_ids"], list) or not row["holder_ids"] or
                                       any(not isinstance(v, str) or not v for v in row["holder_ids"])):
                raise ValueError("invalid holder ids")
            _side(row["newest_entry"])
            side = _side(row["keepalive_entry"]) if row["keepalive_entry"] is not None else None
            rows.append({"uid": row["uid"], "coldkey": row["coldkey"], "qualifies": row["qualifies"], "keepalive_entry": side})
        return FeedResult(True, rows=tuple(rows), generated_at=generated)
    except FileNotFoundError:
        return FeedResult(False, "file missing")
    except OSError:
        return FeedResult(False, "file unreadable")
    except (ValueError, TypeError, OverflowError, RecursionError):
        return FeedResult(False, "invalid feed contract")


def entry_set(feed_rows, snapshot, now, former_champions=()):
    """Champions first, then commitment order; holder cap excludes champions."""
    now = stamp(now)
    out = list(dict.fromkeys(former_champions))
    former = set(out)
    live = {n.uid: n for n in snapshot.neurons}
    candidates = []
    for row in feed_rows:
        side = row["keepalive_entry"]
        if not row["qualifies"] or side is None or side["disqualified"] or "void" in side or side["state"] not in (
                "intake", "needs_correction", "held", "pass_unscored", "overflow"):
            continue
        # Older feeds without the additive end field cannot establish the bounded entitlement.
        if "keepalive_ends_at" not in side:
            continue
        ends = side["keepalive_ends_at"]
        if ends is not None and now >= timestamp(ends):
            continue
        neuron = live.get(row["uid"])
        if neuron is None or neuron.hotkey != side["hotkey"] or neuron.coldkey != row["coldkey"]:
            continue
        if row["uid"] not in former:
            candidates.append((side["commitment_block"], side["extrinsic_index"], row["uid"], row["coldkey"]))
    counts = {}
    for _, _, uid, coldkey in sorted(candidates):
        if uid in out or counts.get(coldkey, 0) >= 2:
            continue
        counts[coldkey] = counts.get(coldkey, 0) + 1
        out.append(uid)
    return out[:MAX_ENTRIES]


class FeedReader:
    """Per-validator memory only; re-filter last-good rows at each payment boundary."""
    def __init__(self, persist_path=None):
        self.last_good = None
        self.path = None
        self._failure = None
        # Feed-mode on-disk last-good (ka_feed_lastgood); None disables persistence.
        self.persist_path = persist_path
        self._saved = None
        self._disk_tried = False

    def read(self, path, now):
        now = stamp(now)
        if self.path != path:
            self.last_good = None
            self.path = path
        if self.last_good is not None and not -FUTURE_TOLERANCE_S <= now - self.last_good.generated_at <= MAX_AGE_S:
            self.last_good = None
        result = load_feed(path, now)
        if result.ok:
            self.last_good = result
            self._failure = None
            self._disk_tried = False
            if self.persist_path and self._saved != (path, result.generated_at):
                from . import ka_feed_lastgood as _persist
                if _persist.save(self.persist_path, path, result):
                    self._saved = (path, result.generated_at)
        else:
            if self.last_good is None and self.persist_path and not self._disk_tried:
                self._disk_tried = True
                # Restart (or first failed read): fall back to the on-disk copy, bounded by generated_at.
                from . import ka_feed_lastgood as _persist
                self.last_good = _persist.load(self.persist_path, path, now)
            failure = (result.why, self.last_good is not None)
            if self._failure != failure:
                self._failure = failure
                # Fixed, sanitised lines; never log raw exceptions or file contents.
                if self.last_good is not None:
                    log.warning("generalist keep-alive feed: %s; using last-good, age %.1f hours",
                                result.why, (now - self.last_good.generated_at) / 3600)
                else:
                    log.warning("generalist keep-alive feed: %s; former champions only", result.why)
            if self.last_good is not None:
                return self.last_good
        return result


def former_champions(leaders):
    """Use the relay's chain-checked title metadata, never a hard-coded uid."""
    uids = set(getattr(leaders, "crowned", {}))
    uids.update(getattr(leaders, "champions", None) or ())
    for group in (getattr(leaders, "cochamps", {}), getattr(leaders, "formset", {}),
                  *(getattr(leaders, "formsets", ()) or ())):
        uids.update(group)
    return sorted(uids)


def approved_former(snapshot, now, relay_former=(), award_ok=None):
    """Pinned approvals first, followed by award-gated relay titles, without duplicates."""
    live = {n.uid: n.hotkey for n in snapshot.neurons}
    pinned = [uid for uid, hotkey in GR2_ROUND1_FORMER.items()
              if stamp(now) < GR2_ROUND1_FORMER_UNTIL.timestamp() and live.get(uid) == hotkey]
    return pinned + [uid for uid in dict.fromkeys(relay_former)
                     if uid not in pinned and (award_ok is None or award_ok("generalist", uid))]
