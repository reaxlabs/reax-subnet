#!/usr/bin/env python3
"""Rollback helper for the multirole release (docs/DESIGN-MULTIROLE.md section 6). OPERATOR TOOL, run with the validator
stopped, BEFORE starting a release older than multirole on this state file.

Since multirole, a UID's s1-pro / s1-lite / s1-cpu score sits at its scoring slot ``uid + 65536 x index`` and the plain
UID entry is empty. An older release reads those slot keys as unknown UIDs and loses the score (a reset). This writes a
copy in the old layout: every slot entry with a role moves back to its plain UID when that entry has no role. A UID with
several model slots (only possible from a multi-model commitment, which an older release does not score anyway) keeps
the slot with the highest moving average. Nothing else changes. The input file is never modified.

Usage: state_unslot.py <state.json> <out.json>
"""

from __future__ import annotations

import json
import os
import sys

SLOT_STRIDE = 1 << 16  # elements.SLOT_STRIDE (kept literal: this tool must run on any release's interpreter)


def unslot(raw: dict) -> tuple[dict, list[str]]:
    miners = {int(k): v for k, v in raw["miners"].items()}
    notes: list[str] = []
    best: dict[int, int] = {}
    for key, m in sorted(miners.items()):
        if key < SLOT_STRIDE or not m.get("role"):
            continue
        uid = key % SLOT_STRIDE
        if miners.get(uid, {}).get("role"):
            notes.append(f"uid {uid}: plain entry has role {miners[uid]['role']!r}; slot {key} dropped")
            continue
        if uid not in best or m.get("ema", 0.0) > miners[best[uid]].get("ema", 0.0):
            best[uid] = key
    out = {k: v for k, v in miners.items() if k < SLOT_STRIDE}
    for uid, key in sorted(best.items()):
        out[uid] = miners[key]
        notes.append(f"uid {uid}: role {miners[key]['role']!r} moved back from slot {key}")
    dropped = sorted(k for k in miners if k >= SLOT_STRIDE and k not in best.values())
    if dropped:
        notes.append(f"slots without a place in the old layout dropped: {dropped}")
    return {**raw, "miners": {str(k): v for k, v in sorted(out.items())}}, notes


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("refusing to overwrite the input; write a copy and swap it in yourself")
    with open(src) as fh:
        raw = json.load(fh)
    out, notes = unslot(raw)
    with open(dst, "x") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\n".join(notes) or "nothing to move")


if __name__ == "__main__":
    main()
