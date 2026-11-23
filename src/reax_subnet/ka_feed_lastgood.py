"""On-disk last-good copy of the feed-mode keep-alive payment projection.

Separate file and schema from the legacy ``ka_lastgood`` list: neither mode can read the other's file.
Only the payment projection (what ``ka_feed.load_feed`` already lets escape) is stored: never the
opaque preview, scores or intake-state data. Age is bounded by the feed's own ``generated_at``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile

from . import ka_feed as _f

log = logging.getLogger("reax.ka_feed_lastgood")
FILE_NAME = "generalist-ka-feed-lastgood.json"
SCHEMA = "reax-keepalive-feed-lastgood/1"
MODE = "feed"
MAX_BYTES = 8 * 1024 * 1024
_SIDE_KEYS = ("hotkey", "state", "commitment_block", "extrinsic_index", "commit_time", "disqualified",
              "void", "keepalive_ends_at")


def _check_side(side):
    if not isinstance(side, dict) or not {"hotkey", "state", "commitment_block", "extrinsic_index",
                                          "commit_time", "disqualified"} <= side.keys() or side.keys() - set(_SIDE_KEYS):
        raise ValueError("side keys")
    _f._pattern(side["hotkey"], _f._SS58)
    if side["state"] not in _f._STATES or type(side["disqualified"]) is not bool:
        raise ValueError("side enum")
    _f._integer(side["commitment_block"])
    _f._integer(side["extrinsic_index"])
    _f.timestamp(side["commit_time"])
    if "void" in side and side["void"] is not True:
        raise ValueError("void")
    if "keepalive_ends_at" in side and side["keepalive_ends_at"] is not None:
        _f.timestamp(side["keepalive_ends_at"])


def save(path, feed_path, result):
    """Atomically persist a GOOD FeedResult; failure never affects payment. Returns True when written."""
    tmp = None
    try:
        if not result.ok or not math.isfinite(result.generated_at):
            return False
        doc = {"schema": SCHEMA, "mode": MODE, "feed_path": os.fspath(feed_path),
               "generated_at": result.generated_at, "round": _f.ROUND,
               "rows": [{"uid": r["uid"], "coldkey": r["coldkey"], "qualifies": r["qualifies"],
                         "keepalive_entry": r["keepalive_entry"]} for r in result.rows]}
        with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(path) or ".",
                                         prefix=".generalist-ka-feed-", delete=False) as fh:
            tmp = fh.name
            json.dump(doc, fh, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
        fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except Exception:  # a cache failure must not change the payment vector
        log.warning("generalist keep-alive feed: last-good copy could not be persisted")
        return False
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load(path, feed_path, now):
    """Return a good FeedResult or None. Age comes from the feed's own generated_at (<= 26 h)."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read(MAX_BYTES + 1)
        if len(raw.encode("utf-8")) > MAX_BYTES:
            raise ValueError("size")
        doc = json.loads(raw)
        if (not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("mode") != MODE
                or doc.get("round") != _f.ROUND or doc.get("feed_path") != os.fspath(feed_path)):
            raise ValueError("tag")
        generated = doc["generated_at"]
        if isinstance(generated, bool) or not isinstance(generated, (int, float)) or not math.isfinite(generated):
            raise ValueError("generated_at")
        now = _f.stamp(now)
        age = now - generated
        if age > _f.MAX_AGE_S or age < -_f.FUTURE_TOLERANCE_S:
            log.warning("generalist keep-alive feed: persisted last-good is %.1f hours old; former champions only",
                        age / 3600)
            return None
        rows, seen = [], set()
        if not isinstance(doc["rows"], list) or len(doc["rows"]) > _f.MAX_ENTRIES * 1000:
            raise ValueError("rows")
        for row in doc["rows"]:
            if not isinstance(row, dict) or set(row) != {"uid", "coldkey", "qualifies", "keepalive_entry"}:
                raise ValueError("row keys")
            _f._integer(row["uid"])
            if row["uid"] in seen or type(row["qualifies"]) is not bool:
                raise ValueError("row")
            seen.add(row["uid"])
            _f._pattern(row["coldkey"], _f._SS58)
            if row["keepalive_entry"] is not None:
                _check_side(row["keepalive_entry"])
            rows.append({"uid": row["uid"], "coldkey": row["coldkey"], "qualifies": row["qualifies"],
                         "keepalive_entry": row["keepalive_entry"]})
        return _f.FeedResult(True, rows=tuple(rows), generated_at=float(generated))
    except FileNotFoundError:
        return None
    except Exception:
        log.warning("generalist keep-alive feed: persisted last-good unavailable or invalid; former champions only")
        return None
