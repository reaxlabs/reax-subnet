"""Keep the miners' absolute alpha when the subnet's MINER emission slot shrinks (post8.10).

The subnet owner moves part of the miner emission slot to the owner slot (``AdminUtils.sudo_set_subnet_emission_weights``,
lock instead of burn; Florian #35779). Our validator (UID 3) is the only weight setter, so every miner's alpha is
``miner slot x weight``. To keep every miner's ABSOLUTE alpha, every weight except the owner/burn UID's is multiplied by
``factor = old miner share / new miner share``; the owner UID absorbs the difference (it is the burn).

Operator file next to the award approvals (``<dir of REAX_AWARD_APPROVALS>/miner-slot.json``, like need.json):

    {"factor": 1.2424242424242424, "from_utc": "2026-10-30T23:00:00+00:00", "reason": "..."}

Exactly these keys; ``factor`` a finite number in [FACTOR_MIN, FACTOR_MAX]; ``from_utc`` an ISO time with a timezone;
``reason`` 1..300 characters. Before ``from_utc``, with an absent file, or with ANY problem (unreadable, refused, no owner
UID) the factor is 1.0 and the vector is returned unchanged (the same object). Write the file only after reading the new
split on chain: ``python -m reax_subnet.miner_slot`` prints the exact factor (read-only).

``apply`` is the LAST step before sending (after need/burn, pauses/run-out, keep-alive and owner allocation) and never
raises: weight setting is never blocked or delayed by this module.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

log = logging.getLogger(__name__)

FILE_NAME = "miner-slot.json"
MAX_BYTES = 4096
FACTOR_MIN, FACTOR_MAX = 1.0, 1.5

# SubtensorModule.SubnetEmissionWeights[92] = [burn, recycle, owner, validator, miner], parts of u64::MAX, read at block
# 9399298 (30 Oct 2026, before the lock-not-burn change): owner 18 % / validator 41 % / miner 41 %. The miners' absolute
# alpha of this split is what the factor keeps.
NETUID = 92
MINER_SLOT = 4
BASE_EMISSION_WEIGHTS = (0, 0, 3320329489486196244, 7563207292111677685, 7563207292111677686)
BASE_BLOCK = 9399298


def path_for(approvals_path: Optional[str]) -> str:
    """The file next to the award approvals file; "" without one."""
    return os.path.join(os.path.dirname(approvals_path), FILE_NAME) if approvals_path else ""


def _time(v: Any) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(v) if isinstance(v, str) else None
    except ValueError:
        return None
    return t if t is not None and t.tzinfo is not None else None


def parse(raw: Any) -> tuple[float, datetime, str]:
    """(factor, from_utc, reason) of a decoded file; ValueError says why it is refused."""
    if not isinstance(raw, dict) or set(raw) != {"factor", "from_utc", "reason"}:
        raise ValueError("keys must be exactly factor, from_utc, reason")
    f = raw["factor"]
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or not FACTOR_MIN <= f <= FACTOR_MAX:
        raise ValueError(f"factor must be a number in [{FACTOR_MIN}, {FACTOR_MAX}]")
    at = _time(raw["from_utc"])
    if at is None:
        raise ValueError("from_utc must be an ISO time with a timezone")
    reason = raw["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
        raise ValueError("reason must be a non-empty string of at most 300 characters")
    return float(f), at, reason


def _read(path: str) -> Any:
    if os.path.getsize(path) > MAX_BYTES:
        raise ValueError(f"larger than {MAX_BYTES} bytes")

    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def factor_at(path: str, now: datetime) -> float:
    """The factor in force at ``now``: 1.0 without a file, before from_utc or for a refused file (logged). Never raises."""
    try:
        if not path or not os.path.exists(path):
            return 1.0
        f, at, reason = parse(_read(path))
    except Exception as exc:  # noqa: BLE001 - any problem: no scaling
        log.warning("miner slot: %s refused (%s); factor 1.0", path, exc)
        return 1.0
    if now < at:
        log.info("miner slot: factor %.6f from %s not in force yet; factor 1.0", f, at.isoformat())
        return 1.0
    return f


def scale(weights: Mapping[int, float], owner_uid: Optional[int], factor: float) -> tuple[dict[int, float], float, bool]:
    """(new vector, effective multiplier of the non-owner weights, clamped). Every non-owner weight x ``factor``; the
    owner UID gets the rest of the vector's total (1 for every normalized vector). If the scaled non-owner weights exceed
    the total, they are scaled to exactly the total and the owner gets 0 (clamped)."""
    if owner_uid is None:
        raise ValueError("no owner UID to absorb the difference")
    if not math.isfinite(factor) or not FACTOR_MIN <= factor <= FACTOR_MAX:
        raise ValueError(f"factor {factor!r} outside [{FACTOR_MIN}, {FACTOR_MAX}]")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("vector has a negative or non-finite weight")
    total = math.fsum(weights.values())
    rest = math.fsum(w for u, w in weights.items() if u != owner_uid)
    if total <= 0 or rest <= 0:
        raise ValueError("nothing to scale")
    eff, clamped = factor, rest * factor > total
    if clamped:
        eff = total / rest
    out = {u: (w * eff if u != owner_uid else 0.0) for u, w in weights.items()}
    out[owner_uid] = max(0.0, total - math.fsum(w for u, w in out.items() if u != owner_uid))
    return out, eff, clamped


def apply(weights: dict[int, float], owner_uid: Optional[int], path: str, now: datetime, block: int) -> dict[int, float]:
    """The vector to send: ``weights`` itself (unchanged object) at factor 1.0 or on any error, else the scaled vector.
    Never raises."""
    try:
        f = factor_at(path, now)
        if f == 1.0:
            return weights
        out, eff, clamped = scale(weights, owner_uid, f)
    except Exception as exc:  # noqa: BLE001 - never block or delay a weight set
        log.error("miner slot: %s: %s at block %s; factor 1.0", type(exc).__name__, exc, block)
        return weights
    if clamped:
        log.warning("miner slot: factor %.6f would give the non-owner weights more than the total; clamped to "
                    "effective %.9f, owner UID %s gets 0", f, eff, owner_uid)
    log.info("miner slot factor %.9f effective %.9f at block %d: owner uid %s %.6f -> %.6f%s", f, eff, block, owner_uid,
             weights.get(owner_uid, 0.0), out[owner_uid], " CLAMPED" if clamped else "")
    return out


# ----------------------------------------------------------------------------------------- read-only operator CLI
def miner_share(emission_weights) -> float:
    """The miner slot's fraction of SubnetEmissionWeights ([burn, recycle, owner, validator, miner])."""
    w = [int(x) for x in emission_weights]
    if len(w) != 5 or any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"unexpected SubnetEmissionWeights value {emission_weights!r}")
    return w[MINER_SLOT] / sum(w)


def factor_for(current) -> float:
    """old miner share (BASE_EMISSION_WEIGHTS) / current miner share."""
    cur = miner_share(current)
    if cur <= 0:
        raise ValueError("the miner slot is 0: no factor can keep the miners' alpha")
    return miner_share(BASE_EMISSION_WEIGHTS) / cur


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m reax_subnet.miner_slot",
                                 description="Read SN92's emission split from chain (read-only) and print the "
                                             "miner-slot.json factor. Never writes.")
    ap.add_argument("--network", default=os.environ.get("REAX_NETWORK", "finney") or "finney")
    ap.add_argument("--netuid", type=int, default=NETUID)
    ap.add_argument("--check", metavar="FILE", help="also check an existing miner-slot.json against the chain")
    ap.add_argument("--weights", metavar="W0,W1,W2,W3,W4", help="offline: use these u64 values instead of the chain")
    a = ap.parse_args(argv)
    if a.weights:
        cur, block = [int(x) for x in a.weights.split(",")], None
    else:
        import bittensor as bt

        sub = bt.Subtensor(network=a.network)
        cur = [int(x) for x in sub.query(("SubtensorModule", "SubnetEmissionWeights"), [a.netuid])]
        try:
            block = sub.block_info().number
        except Exception:  # noqa: BLE001
            block = None
    pct = lambda w: ", ".join(f"{x / sum(w) * 100:.4f}" for x in w)  # noqa: E731
    print(f"SubnetEmissionWeights[{a.netuid}] at block {block} [burn, recycle, owner, validator, miner] = {cur}")
    print(f"  percent: {pct(cur)}")
    print(f"base (block {BASE_BLOCK}) percent: {pct(BASE_EMISSION_WEIGHTS)}")
    f = factor_for(cur)
    print(f"factor = old miner share / new miner share = {f!r}")
    rc = 0
    if abs(f - 1.0) < 1e-12:
        print("split unchanged: factor 1.0 -> no miner-slot.json needed (remove it if present)")
    elif not FACTOR_MIN <= f <= FACTOR_MAX:
        print(f"REFUSED: factor outside [{FACTOR_MIN}, {FACTOR_MAX}]; do not write miner-slot.json, ask the lead")
        rc = 2
    else:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        print("miner-slot.json content (write atomically next to approvals.json):")
        print(json.dumps({"factor": f, "from_utc": now, "reason": f"SN92 miner slot {miner_share(cur) * 100:.4f} % "
                          f"(was {miner_share(BASE_EMISSION_WEIGHTS) * 100:.4f} %) at block {block}; keep miners' alpha"}))
    if a.check:
        try:
            ff, at, _ = parse(_read(a.check))
        except Exception as exc:  # noqa: BLE001
            print(f"CHECK {a.check}: REFUSED by the validator rules ({exc}) -> the validator uses factor 1.0")
            return 3
        ok = abs(ff - f) <= 1e-9
        print(f"CHECK {a.check}: factor {ff!r} from {at.isoformat()} -> {'OK matches chain' if ok else 'MISMATCH'}")
        rc = rc or (0 if ok else 4)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
