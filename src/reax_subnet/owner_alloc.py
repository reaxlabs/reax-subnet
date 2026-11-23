"""One-off owner allocations from inference or residual burn; operator file beside approvals.

Computation is read-only. The submission journal is fsynced before an
allocation-bearing send and the set is counted only on proven success. This module
never blocks, delays or raises into weight setting: on ANY uncertainty (send
exception, unknown outcome, corrupt/unwritable journal, read-only mount, restart
with a pending entry) the weights go out WITHOUT the allocation (exactly the
post8.8 vector), the set is not counted, the rows are marked needs-reconcile with
one ERROR line, and allocation stays off until the operator clears the flag.
Never infer success from elapsed blocks or silently reset a corrupt journal.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

from .ka_override import _SS58

log = logging.getLogger(__name__)
FILE_NAME = "owner-allocation.json"
STATE_NAME = "allocation-progress.json"
MAX_BYTES = 16384
MAX_SETS = 24


# State paths whose needs-reconcile flag could not be written (corrupt/read-only journal):
# allocation stays off for this process; cleared only by an operator restart after repair.
_LATCHED: dict = {}


def path_for(approvals_path):
    return os.path.join(os.path.dirname(approvals_path), FILE_NAME) if approvals_path else ""


def parse(raw):
    if not isinstance(raw, list) or len(raw) > 6:
        raise ValueError("expected at most six rows")
    seen = set()
    for row in raw:
        if not isinstance(row, dict) or set(row) not in ({"hotkey", "share", "sets", "start_block", "reason"},
                                                         {"hotkey", "share", "sets", "start_block", "reason", "source"}):
            raise ValueError("row keys must be hotkey, share, sets, start_block, reason, optionally source")
        if row.get("source", "inference") not in ("burn", "inference"):
            raise ValueError("source must be burn or inference")
        if not isinstance(row["hotkey"], str) or not _SS58.fullmatch(row["hotkey"]):
            raise ValueError("invalid hotkey")
        if type(row["share"]) not in (int, float) or not math.isfinite(row["share"]) or not 0 < row["share"] <= .10:
            raise ValueError("share must be in (0, 0.10]")
        if type(row["sets"]) is not int or not 1 <= row["sets"] <= MAX_SETS:
            raise ValueError("sets must be an integer in [1, 24]")
        if type(row["start_block"]) is not int or row["start_block"] < 0:
            raise ValueError("start_block must be a nonnegative integer")
        if not isinstance(row["reason"], str) or not row["reason"].strip() or len(row["reason"]) > 300:
            raise ValueError("reason must have 1..300 characters")
        key = row_id(row)
        if key in seen:
            raise ValueError("duplicate row")
        seen.add(key)
    return raw


def row_id(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path):
    if os.path.getsize(path) > MAX_BYTES:
        raise ValueError("file too large")
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def load(path):
    if not path or not os.path.exists(path):
        return []
    try:
        return parse(_read(path))
    except (OSError, ValueError, TypeError) as exc:
        log.warning("owner allocation %s ignored: %s", path, exc)
        return []


def _state(path):
    if not os.path.exists(path):
        return {"version": 1, "counts": {}, "pending": None, "needs_reconcile": {}}
    s = _read(path)
    if isinstance(s, dict) and set(s) == {"version", "counts", "pending"}:
        s["needs_reconcile"] = {}
    if (not isinstance(s, dict) or set(s) != {"version", "counts", "pending", "needs_reconcile"} or s["version"] != 1
            or not isinstance(s["counts"], dict)
            or any(not isinstance(k, str) or len(k) != 64 or type(v) is not int or not 0 <= v <= MAX_SETS
                   for k, v in s["counts"].items())
            or not isinstance(s["needs_reconcile"], dict)
            or any(not isinstance(k, str) or len(k) != 64 for k in s["needs_reconcile"])
            or (s["pending"] is not None and (not isinstance(s["pending"], dict)
                                              or not isinstance(s["pending"].get("rows"), list)))):
        raise ValueError("invalid allocation progress")
    return s


def _write(path, data):
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".allocation-", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def flag(path, block, ids, why):
    """Mark rows needs-reconcile: ONE ERROR line naming the set; never raises.

    Persisted into the journal (pending cleared, its rows moved to needs_reconcile) when
    the journal is readable and writable; otherwise latched in memory for this process."""
    ids = list(ids)
    log.error("owner allocation NEEDS-RECONCILE: weight set at block %s rows %s: %s; set NOT counted, "
              "weights go out without allocation until the operator clears needs_reconcile in %s",
              block, ",".join(i[:12] for i in ids) or "-", why, path)
    try:
        s = _state(path)
        for i in ids + list((s["pending"] or {}).get("rows", [])):
            s["needs_reconcile"].setdefault(str(i), {"block": block, "why": str(why)[:200]})
        if not s["needs_reconcile"]:
            s["needs_reconcile"]["0" * 64] = {"block": block, "why": str(why)[:200]}
        s["pending"] = None
        _write(path, s)
    except Exception as exc:  # noqa: BLE001 - corrupt/read-only journal: in-memory latch
        _LATCHED[path] = f"{why}; flag not persisted ({type(exc).__name__}: {exc})"


class Shares(dict):
    """Combined uid totals for compatibility; separate budgets for the weight computation."""
    def __init__(self):
        super().__init__()
        self.inference, self.burn = {}, {}
        self.funded = None
        self.burn_weights = {}
        self.miner_part = 1.0


class Plan:
    """A weight computation's immutable rows and progress snapshot; one sender only.

    Never raises. ``shares`` feed the computation; ``attach`` records the full and the
    plain (post8.8) vector; ``begin`` decides at send time which one goes out."""
    def __init__(self, path, block, neurons, owner_uid=None):
        self.path = os.path.join(os.path.dirname(path), STATE_NAME) if path else ""
        self.block = block
        self.owner_uid = owner_uid
        self.rows = []
        self.shares = Shares()
        self.state = None
        self.full = self.plain = None
        try:
            self._init(path, neurons, owner_uid)
        except Exception as exc:  # noqa: BLE001 - never stop weight setting
            self.rows, self.shares = [], Shares()
            log.error("owner allocation disabled for the set at block %s: %s: %s", block, type(exc).__name__, exc)

    def _init(self, path, neurons, owner_uid):
        rows = load(path)
        if not rows:
            return
        if self.path in _LATCHED:
            log.warning("owner allocation skipped at block %s: needs-reconcile (%s); restart after repair",
                        self.block, _LATCHED[self.path])
            return
        try:
            self.state = _state(self.path)
        except (OSError, ValueError, TypeError) as exc:
            flag(self.path, self.block, [], f"allocation journal unreadable/corrupt ({exc})")
            return
        if self.state["pending"] is not None:
            p = self.state["pending"]
            flag(self.path, p.get("block"), p.get("rows", []),
                 f"pending allocation send found at restart (seen at block {self.block}); outcome unknown")
            return
        if self.state["needs_reconcile"]:
            log.warning("owner allocation skipped at block %s: needs_reconcile %s in %s", self.block,
                        ",".join(k[:12] for k in self.state["needs_reconcile"]), self.path)
            return
        active = [r for r in rows if self.block >= r["start_block"]
                  and self.state["counts"].get(row_id(r), 0) < r["sets"]]
        if math.fsum(r["share"] for r in active) > .25:
            log.warning("owner allocation file ignored: active share exceeds 0.25")
            return
        uid_of = {n.hotkey: n.uid for n in neurons}
        if owner_uid is not None and any(uid_of.get(r["hotkey"]) == owner_uid for r in active):
            log.warning("owner allocation file ignored: owner burn UID cannot receive an allocation")
            return
        for r in active:
            uid = uid_of.get(r["hotkey"])
            if uid is None:
                log.warning("owner allocation %s has no uid at block %s; set not counted", r["hotkey"], self.block)
                continue
            self.rows.append(r)
            self.shares[uid] = self.shares.get(uid, 0.0) + r["share"]
            budget = getattr(self.shares, r.get("source", "inference"))
            budget[uid] = budget.get(uid, 0.0) + r["share"]

    def attach(self, full, plain, burn_plain=None):
        """The vector with the allocation and the identical computation without it."""
        self.full, self.plain = dict(full), dict(plain)
        if self.shares.funded is not None:
            # Zero-room rows consume no set; partial proportional payments count as before.
            self.rows = [r for r in self.rows if r.get("source", "inference") in self.shares.funded]
        if burn_plain is not None:
            # An approval/state read can change between the two computations. Never
            # send a burn allocation that also moves a protected component.
            amount = math.fsum(self.shares.burn_weights.values())
            owner_delta = {u: self.full.get(u, 0.0) - burn_plain.get(u, 0.0)
                           - self.shares.burn_weights.get(u, 0.0)
                           for u in set(self.full) | set(burn_plain) | set(self.shares.burn_weights) | {self.owner_uid}}
            if any(not math.isfinite(v) or abs(v + (amount if u == self.owner_uid else 0.0)) > 1e-12
                   for u, v in owner_delta.items()):
                raise ValueError("burn allocation moved a protected component")
        if burn_plain is not None and self.shares.burn_weights:
            log.info("owner allocation burn vector block %s %s", self.block,
                     json.dumps(dict(plain=burn_plain, full=full, burn=self.shares.burn_weights, miner_part=self.shares.miner_part),
                                sort_keys=True, separators=(",", ":")))
        if self.rows and self.full == self.plain:
            log.warning("owner allocation had no effect at block %s (no inference to take from); set not counted",
                        self.block)
            self.rows = []

    def begin(self, weights):
        """Return (vector to send, plan to finish or None). Never raises, never withholds weights.

        The allocation-bearing vector goes out only when the pending journal is durably
        written; otherwise the plain vector goes out and nothing is counted."""
        try:
            if not self.rows:
                return weights, None
            if self.state is None or self.state["pending"] is not None:  # begun before: never a second count
                log.error("owner allocation: plan of block %s already sent; sending without allocation", self.block)
                return (self.plain if self.plain is not None and weights == self.full else weights), None
            if self.full is None or self.plain is None or weights != self.full:
                log.error("owner allocation: vector to send is not this plan's (block %s); allocation not counted",
                          self.block)
                return weights, None
            try:
                if _state(self.path) != self.state:
                    raise ValueError("allocation journal changed after weight computation")
                pending = dict(self.state, pending={"block": self.block, "rows": [row_id(r) for r in self.rows]})
                _write(self.path, pending)
            except Exception as exc:  # noqa: BLE001 - read-only/unwritable/changed journal
                flag(self.path, self.block, [row_id(r) for r in self.rows],
                     f"could not journal the allocation before sending ({type(exc).__name__}: {exc})")
                return self.plain, None
            self.state = pending
            return self.full, self
        except Exception as exc:  # noqa: BLE001
            log.error("owner allocation: unexpected %s at block %s; sending without allocation", exc, self.block)
            return (self.plain if self.plain is not None else weights), None

    def finish(self, success):
        """Settle the send journalled by ``begin``. Counts exactly once on success; never raises."""
        ids = [row_id(r) for r in self.rows]
        try:
            if not self.rows or self.state is None or self.state.get("pending") is None:
                return
            if success is not True:
                flag(self.path, self.block, ids, "allocation send raised or outcome unknown")
                return
            if _state(self.path) != self.state:
                flag(self.path, self.block, ids, "allocation journal changed or already settled during the send")
                return
            done = {**self.state, "counts": dict(self.state["counts"]), "pending": None}
            for r in self.rows:
                k = row_id(r)
                done["counts"][k] = done["counts"].get(k, 0) + 1
            _write(self.path, done)
            for r in self.rows:
                log.info("owner allocation committed block %s hotkey %s set %s/%s source %s reason %s row_id %s row_share %.12f",
                         self.block, r["hotkey"], done["counts"][row_id(r)], r["sets"], r.get("source", "inference"), r["reason"], row_id(r), r["share"])
        except Exception as exc:  # noqa: BLE001
            flag(self.path, self.block, ids, f"could not record the settled send ({type(exc).__name__}: {exc})")
        finally:
            self.rows = []
            self.state = None


def take(shares, available, source="inference"):
    """Scale miner-part fractions only to their own source's available budget."""
    request = getattr(shares, source, shares if source == "inference" else {})
    if isinstance(shares, Shares) and shares.funded is None:
        shares.funded = set()
    if not request:
        return {}, 0.0
    total = math.fsum(request.values())
    factor = min(1.0, max(0.0, available) / total)
    if factor < 1:
        log.warning("owner allocation scaled by %.12f: requested %.12f %s %.12f", factor, total, source, available)
    if factor * total <= 1e-9:
        return {}, 0.0
    paid = {u: s * factor for u, s in request.items() if s * factor > 0}
    if paid and isinstance(shares, Shares):
        shares.funded.add(source)
    return paid, math.fsum(paid.values())
