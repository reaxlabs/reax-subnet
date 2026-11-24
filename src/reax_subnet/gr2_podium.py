"""Round 2 payout tables for verified Generalist podium results."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Optional

OPENING_WINDOW = "2026-11-26"
OPENING_SHARES = {
    "champion": 0.70, "former[0]": 0.0, "former[1]": 0.0, "place2": 0.20, "place3": 0.10,
}
STANDARD_SHARES = {
    "champion": 0.70, "former[0]": 0.10, "former[1]": 0.05, "place2": 0.10, "place3": 0.05,
}
LATER_WINDOWS = {"2026-12-03", "2026-12-10"}


def payout_fractions(
    window: str, shares: Mapping[str, float], eligible: Mapping[str, int]
) -> dict[int, float]:
    """Resolve a verified window's payable slots without redistributing empty shares.

    The caller verifies the publication digest, holder identity, reveal and
    hold state, and approval before supplying the eligible slot-to-UID map.
    """
    if window == OPENING_WINDOW:
        expected = OPENING_SHARES
    elif window in LATER_WINDOWS:
        expected = STANDARD_SHARES
    else:
        raise ValueError("unknown Round 2 window")
    if dict(shares) != expected or any(type(value) not in {float, int} for value in shares.values()):
        raise ValueError("wrong podium shares for the window")
    if not eligible.keys() <= expected.keys() or any(type(uid) is not int or uid < 0 for uid in eligible.values()):
        raise ValueError("invalid podium slot or UID")
    if len(set(eligible.values())) != len(eligible):
        raise ValueError("one UID cannot hold multiple podium places")
    return {eligible[slot]: share for slot, share in expected.items() if share and slot in eligible}


def payout_after_keepalive(pool: float, keepalive: float, fractions: Mapping[int, float]) -> dict[int, float]:
    """Compute the podium component of the weight vector after actual keep-alive."""
    if not isfinite(pool) or not isfinite(keepalive) or not 0 <= keepalive <= pool:
        raise ValueError("invalid Generalist pool or keep-alive")
    return {uid: (pool - keepalive) * share for uid, share in fractions.items()}


@dataclass(frozen=True)
class PodiumPlan:
    """What ``combine_weights`` needs to pay the Generalist podium of one weight set.

    ``window`` and ``shares`` are the verified fragment's; ``eligible`` maps slot -> UID for the slots that are payable
    right now (the caller resolved the fragment's tenure/reveal/hold state and the current chain binding; the award gate
    is applied again inside ``combine_weights``). ``window`` None = the podium is burned whole (no readable fragment):
    nobody is paid and no ordinary Generalist rank share is paid either.
    """

    window: Optional[str] = None
    shares: Mapping[str, float] = field(default_factory=dict)
    eligible: Mapping[str, int] = field(default_factory=dict)
