"""Validator state that survives restarts.

The file holds only per-UID moving averages and bookkeeping, no items, no
miner answers and no keys. It is written atomically (temp file, fsync,
rename), so a crash leaves either the old or the new state. A corrupt or
foreign file is moved aside and the validator starts from zero, which is safe:
scores rebuild within a few rounds.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from .chain import Snapshot
from .elements import MODEL_INDEX, MODEL_ORDER, SLOT_STRIDE, slot_key

log = logging.getLogger("reax.state")
STATE_VERSION = 1


@dataclass
class MinerState:
    hotkey: str
    ema: float = 0.0
    last_score: float = 0.0
    last_fidelity: float = 0.0
    rounds: int = 0
    # reax-s1: "<model>:<manifest_id>" the hotkey committed to; a change resets the score.
    role: str = ""
    strikes: int = 0
    # Tiered pools: moving average of the verified throughput (decisions/s) from capacity probes; None = never probed.
    capacity: float | None = None


@dataclass
class ValidatorState:
    netuid: int
    hotkey: str
    # Scoring slots (docs/DESIGN-MULTIROLE.md section 4): key = elements.slot_key(uid, model). The s1-fast slot of a UID
    # is the UID itself, so a UID's plain entry exists for every registered neuron, as before; the slots of its other
    # models (one hotkey serving several models) sit at uid + SLOT_STRIDE x model index.
    miners: dict[int, MinerState] = field(default_factory=dict)
    rounds: int = 0

    def sync(self, snapshot: Snapshot) -> list[int]:
        """Drop UIDs that left and reset UIDs whose hotkey changed.

        A new hotkey on a recycled UID must never inherit the previous owner's score: all of that UID's slots are
        reset. A hotkey that moved to a new UID (owner trim) takes all its slots along. Returns the UIDs that were
        reset.
        """
        current = {n.uid: n.hotkey for n in snapshot.neurons}
        by_hotkey: dict[str, dict[int, MinerState]] = {}
        for slot, m in self.miners.items():
            by_hotkey.setdefault(m.hotkey, {})[slot // SLOT_STRIDE] = m
        had = {slot % SLOT_STRIDE for slot in self.miners}
        new: dict[int, MinerState] = {}
        reset = []
        for uid, hotkey in current.items():
            moved = by_hotkey.pop(hotkey, None)
            if moved is not None:
                # Same hotkey, possibly under a new UID (owner trim renumbers UIDs).
                for index, m in moved.items():
                    new[uid + SLOT_STRIDE * index] = m
            if uid not in new:
                new[uid] = MinerState(hotkey=hotkey)
                if moved is None and uid in had:
                    reset.append(uid)
        self.miners = new
        return reset

    def update(self, uid: int, score: float, fidelity: float, alpha: float) -> None:
        m = self.miners[uid]
        m.ema = (1.0 - alpha) * m.ema + alpha * score
        m.last_score = score
        m.last_fidelity = fidelity
        m.rounds += 1

    def set_role(self, uid: int, role: str) -> bool:
        """Record the miner's committed role; a new role starts from zero. Returns True on reset."""
        m = self.miners[uid]
        if m.role == role:
            return False
        reset = bool(m.role) or m.ema > 0.0
        self.miners[uid] = MinerState(hotkey=m.hotkey, role=role)
        return reset

    def strike(self, uid: int, fidelity: float) -> None:
        """A failed verification: the moving average drops to zero."""
        m = self.miners[uid]
        m.ema = 0.0
        m.last_score = 0.0
        m.last_fidelity = fidelity
        m.rounds += 1
        m.strikes += 1
        if m.capacity is not None:
            m.capacity = 0.0

    def update_capacity(self, uid: int, dps: float, alpha: float) -> None:
        """Tiered pools: fold one verified-throughput measurement (decisions/s) into the moving average."""
        m = self.miners[uid]
        m.capacity = dps if m.capacity is None else (1.0 - alpha) * m.capacity + alpha * dps

    def ema(self) -> dict[int, float]:
        return {uid: m.ema for uid, m in self.miners.items()}

    def drop_slot(self, slot: int) -> None:
        """Forget a model slot (index > 0) whose model left the commitment; a plain UID entry is never dropped."""
        if slot >= SLOT_STRIDE:
            self.miners.pop(slot, None)

    def to_json(self) -> dict:
        return {
            "version": STATE_VERSION,
            "netuid": self.netuid,
            "hotkey": self.hotkey,
            "rounds": self.rounds,
            "saved_at": int(time.time()),
            "miners": {
                str(uid): {
                    "hotkey": m.hotkey,
                    "ema": m.ema,
                    "last_score": m.last_score,
                    "last_fidelity": m.last_fidelity,
                    "rounds": m.rounds,
                    "role": m.role,
                    "strikes": m.strikes,
                    **({"capacity": m.capacity} if m.capacity is not None else {}),
                }
                for uid, m in sorted(self.miners.items())
            },
        }


def save(state: ValidatorState, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state.to_json(), fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:  # make the rename itself durable; not supported on every platform/filesystem
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _capacity(value: object) -> float | None:
    """An unusable capacity is dropped (the UID is simply probed again), never the whole state file."""
    if not _valid_number(value) or not 0.0 <= value <= 1e6:  # type: ignore[operator]
        return None
    return float(value)  # type: ignore[arg-type]


def _migrate_legacy_slots(state: ValidatorState) -> None:
    """Before multi-model hotkeys every model's entry sat at its UID. An entry at a plain UID whose role is
    ``<model>:<manifest>`` of a model with slot index > 0 moves to that model's slot (no score reset) and the UID's
    plain entry is recreated empty for the hotkey. Idempotent: the recreated entry has no role, and an entry that is
    already at its slot is kept (the legacy duplicate is dropped)."""
    for uid in sorted(u for u in state.miners if u < SLOT_STRIDE):
        m = state.miners[uid]
        model = m.role.partition(":")[0]
        if MODEL_INDEX.get(model, 0) > 0:
            slot = slot_key(uid, model)
            state.miners.setdefault(slot, m)
            state.miners[uid] = MinerState(hotkey=m.hotkey)


def load(path: str, netuid: int, hotkey: str) -> ValidatorState:
    fresh = ValidatorState(netuid=netuid, hotkey=hotkey)
    if not os.path.exists(path):
        return fresh
    try:
        with open(path) as fh:
            raw = json.load(fh)
        if raw.get("version") != STATE_VERSION or raw.get("netuid") != netuid or raw.get("hotkey") != hotkey:
            raise ValueError("state belongs to a different version, subnet or hotkey")
        state = ValidatorState(netuid=netuid, hotkey=hotkey, rounds=int(raw.get("rounds", 0)))
        for uid_s, m in raw["miners"].items():
            if not 0 <= int(uid_s) < SLOT_STRIDE * len(MODEL_ORDER):
                raise ValueError(f"bad slot key {uid_s}")
            values = (m["ema"], m["last_score"], m["last_fidelity"])
            if not all(_valid_number(v) and 0.0 <= v <= 1.0 for v in values):
                raise ValueError(f"bad numbers for uid {uid_s}")
            state.miners[int(uid_s)] = MinerState(
                hotkey=str(m["hotkey"]),
                ema=float(m["ema"]),
                last_score=float(m["last_score"]),
                last_fidelity=float(m["last_fidelity"]),
                rounds=int(m["rounds"]),
                role=str(m.get("role", ""))[:64],
                strikes=int(m.get("strikes", 0)),
                capacity=_capacity(m.get("capacity")),
            )
        _migrate_legacy_slots(state)
        return state
    except Exception as exc:
        aside = f"{path}.corrupt-{int(time.time())}"
        log.warning("ignoring unusable state file (%s); moved to %s", exc, aside)
        try:
            os.replace(path, aside)
        except OSError:
            pass
        return fresh

