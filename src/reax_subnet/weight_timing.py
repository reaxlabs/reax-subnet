"""Earlier weight submission for newly committed, exposed competition entries.

Only timing changes: the normal weight calculation decides eligibility and amount.
The chain cooldown, SDK encryption, version key and ordinary cadence are unchanged.
"""
from __future__ import annotations

from .chain import Snapshot
from .competition_split import CHAIN_IMMUNITY_BLOCKS, ENGINE_KA_LEAD_BLOCKS
from .elements import fields_of, split_elements

# Release switch. False restores the configured weight-set cadence exactly.
KA_NEWCOMMIT_SET = True


def new_exposed_payees(snap: Snapshot, weights: dict[int, float], validator: str,
                       submitted_block: int, submitted_payees: set[tuple[int, str]]) -> list[int]:
    if not KA_NEWCOMMIT_SET:
        return []
    mine = next((n for n in snap.neurons if n.hotkey == validator), None)
    # A pinned LastUpdate avoids estimating it from a later RPC's "blocks since".
    # After a restart this also prevents re-submitting a commitment already covered by a set.
    if mine is None or mine.last_update <= 0:
        return []
    last = max(mine.last_update, submitted_block)
    top = max(weights.values(), default=0.0)
    if top <= 0:
        return []
    out = []
    for n in snap.neurons:
        # Start before immunity expires so commit-reveal can count the first paying set in time.
        remaining_immunity = n.registered_block + CHAIN_IMMUNITY_BLOCKS - snap.block
        if (n.hotkey == validator or n.uid == snap.owner_uid or n.incentive != 0.0
                or n.registered_block <= 0 or remaining_immunity >= ENGINE_KA_LEAD_BLOCKS
                or not last < n.commitment_block <= snap.block
                or (n.uid, n.hotkey) in submitted_payees
                or round(weights.get(n.uid, 0.0) / top * 65535) <= 0):
            continue
        # The competition view keeps the original Raw fields. Parse them too so invalid
        # dual-role fields cannot trigger a set; a test/tool may supply an unsplit snapshot.
        el = split_elements(fields_of(n), snap.block)
        if el is not None and el.competition:
            out.append(n.uid)
    return sorted(out)
