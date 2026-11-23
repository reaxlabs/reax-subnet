"""Validator-owned generalist KA slots. Never stores relay state or private preview data."""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile

log = logging.getLogger("reax.ka_lastgood")
MAX_AGE_S = 26 * 3600
FILE_NAME = "generalist-ka-lastgood.json"
MAX_BYTES = 1024 * 1024


def save(path, now, slots, snapshot, award_ok=None):
    """Atomically replace even an empty good list; failed writes leave the old timestamp intact."""
    tmp = None
    try:
        neurons = {n.uid: n for n in snapshot.neurons}
        seen = set()
        rows = []
        for kind in ("keepalive", "provisional"):
            for uid in slots.get(kind, ()):
                if uid in neurons and uid != snapshot.owner_uid and uid not in seen:
                    rows.append({"uid": uid, "hotkey": neurons[uid].hotkey, "kind": kind,
                                 "commitment": neurons[uid].commitment,
                                 "gated": award_ok is not None and bool(award_ok("generalist", uid))})
                    seen.add(uid)
        doc = {"version": 1, "saved_at": now, "entries": rows}
        with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(path) or ".",
                                         prefix=".generalist-ka-", delete=False) as fh:
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
    except Exception:  # a cache failure must not change the good-state vector
        log.warning("generalist keep-alive: last-good list could not be persisted")
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load(path, now, snapshot, award_ok):
    """Recheck age, uid/hotkey/commitment, and approvals for rows gated at save."""
    empty = {"keepalive": [], "provisional": []}
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read(MAX_BYTES + 1)
        if len(raw.encode("utf-8")) > MAX_BYTES:
            raise ValueError("size")
        doc = json.loads(raw)
        saved = doc["saved_at"]
        if (doc["version"] != 1 or isinstance(saved, bool) or not isinstance(saved, (int, float))
                or not math.isfinite(saved) or not 0 <= now - saved <= MAX_AGE_S):
            log.warning("generalist keep-alive: last-good list expired or invalid; none paid")
            return empty
        neurons = {n.uid: n for n in snapshot.neurons}
        out = {"keepalive": [], "provisional": []}
        seen = set()
        for row in doc["entries"]:
            uid, hk, kind = row["uid"], row["hotkey"], row["kind"]
            commitment, gated = row["commitment"], row["gated"]
            if (type(uid) is not int or not isinstance(hk, str) or kind not in out
                    or (commitment is not None and not isinstance(commitment, str)) or type(gated) is not bool):
                raise ValueError("row")
            if (uid not in seen and uid != snapshot.owner_uid and uid in neurons
                    and neurons[uid].hotkey == hk and neurons[uid].commitment == commitment
                    and (not gated or award_ok is None or award_ok("generalist", uid))):
                seen.add(uid)
                out[kind].append(uid)
        log.info("generalist keep-alive: using last-good list, age %.1f h", (now - saved) / 3600)
        return out
    except Exception:
        log.warning("generalist keep-alive: last-good list unavailable or invalid; none paid")
        return empty
