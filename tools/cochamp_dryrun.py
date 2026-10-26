#!/usr/bin/env python3
"""Dry run of the generalist payout (co-champions, places 2/3, keep-alive) on a worker state file and a chain snapshot.

    PYTHONPATH=src python tools/cochamp_dryrun.py spec-state.json snapshot.json [--now ISO|unix] [--unapproved 5,6]
                                                  [--serving serving.json] [--track generalist]

``spec-state.json``: the competition worker's state file (``/w/comp/state/spec-state.json``).
``snapshot.json``: ``{"block": 123, "owner_hotkey": "5..." | null, "neurons": [{"uid": 1, "hotkey": "5...", "coldkey": "5...",
"commitment": "reax-spec/1:generalist:owner/repo@<40 hex>", "commitment_block": 0, "registered_block": 0}, ...]}``;
only ``uid``, ``hotkey`` and ``commitment`` are needed (missing keys get neutral defaults).
``--serving``: optional ``{"uid": weight}`` of the inference part (default: one equal weight per neuron in the snapshot).
``--unapproved``: comma-separated uids whose owner approval is treated as missing (default: every ranked uid approved).

Prints the leaders (rank order, co-champion fractions), the keep-alive list and the payout of the generalist track as
``uid -> share of the total weight`` (the weights with the track minus the weights without it). Read-only; no network.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from reax_subnet import competition_split as CS
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.spec import state as spec_state
from reax_subnet.spec.evaluator import compute_keepalive, compute_leaders, keepalive_protected


def load_snapshot(path: str) -> Snapshot:
    raw = json.load(open(path))
    neurons = tuple(Neuron(uid=int(n["uid"]), hotkey=n["hotkey"], coldkey=n.get("coldkey", f"cold-{n['uid']}"),
                           validator_permit=bool(n.get("validator_permit", False)), stake=float(n.get("stake", 0.0)),
                           ip=n.get("ip"), port=n.get("port"), commitment=n.get("commitment", "") or "",
                           commitment_block=int(n.get("commitment_block", 0) or 0),
                           registered_block=int(n.get("registered_block", 0) or 0)) for n in raw["neurons"])
    return Snapshot(block=int(raw.get("block", 0)), neurons=neurons, owner_hotkey=raw.get("owner_hotkey"))


def parse_now(text: str | None) -> float:
    if not text:
        return datetime.now(timezone.utc).timestamp()
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("snapshot")
    ap.add_argument("--now", help="ISO time or unix seconds (default: the wall clock)")
    ap.add_argument("--unapproved", default="", help="uids without owner approval, comma-separated")
    ap.add_argument("--serving", help="JSON {uid: weight} of the inference part")
    ap.add_argument("--track", default="generalist")
    args = ap.parse_args(argv)

    state = spec_state.from_json(json.load(open(args.state)))
    snap = load_snapshot(args.snapshot)
    now = parse_now(args.now)
    track = args.track
    unapproved = {int(x) for x in args.unapproved.split(",") if x.strip()}
    serving = ({int(k): float(v) for k, v in json.load(open(args.serving)).items()} if args.serving
               else {n.uid: 1.0 for n in snap.neurons})
    award_ok = lambda t, uid: uid not in unapproved   # noqa: E731

    leaders = compute_leaders(state, [track], snap, now=now)[track]
    ka = compute_keepalive(state, [track], snap, now=now)
    prot = keepalive_protected(state, [track], snap)
    ts = state.track(track)
    champ = ts.submissions.get(ts.champion)
    print(f"now {datetime.fromtimestamp(now, timezone.utc):%Y-%m-%d %H:%M:%S}Z  block {snap.block}  champion "
          f"{'uid %d' % champ.uid if champ else '-'}  co-champion keys listed: {len(CS.GENERALIST_CO_CHAMPIONS)}")
    print("leaders (rank order):", [(u, round(s, 2)) for u, s in leaders])
    print("co-champion fractions of rank 1:", {u: round(f, 4) for u, f in sorted(getattr(leaders, "cochamps", {}).items())} or "none")
    print("keep-alive list:", [(u, round(s, 2)) for u, s in ka.get(track, [])], " protected:", prot.get(track, []))
    print("rank shares (before approvals):", {u: round(s, 4) for u, s in CS.rank_shares(leaders, CS.TRACK_ENTRY_FLOOR.get(track)).items()})
    print("unapproved uids:", sorted(unapproved) or "none")
    kw = dict(owner_uid=snap.owner_uid, award_ok=award_ok, now=now)
    with_track = CS.combine_weights(serving, None, 0.0, {track: leaders}, keepalive={track: ka.get(track, [])} if ka.get(track) else None,
                                    keepalive_protected=prot or None, **kw)
    without = CS.combine_weights(serving, None, 0.0, {}, **kw)
    pay = {u: with_track.get(u, 0.0) - without.get(u, 0.0) for u in set(with_track) | set(without)}
    total = sum(v for v in pay.values() if v > 0)
    print(f"\ngeneralist payout (weights with the track minus without; sum of gains {total:.5f} of the total weight)")
    print("uid   share      kind")
    ranks = CS.rank_shares(leaders, CS.TRACK_ENTRY_FLOOR.get(track))
    kaset = {u for u, _ in ka.get(track, [])}
    for u, v in sorted(pay.items(), key=lambda kv: -kv[1]):
        if v > 1e-12:
            kind = "rank" if u in ranks else ("keep-alive" if u in kaset else "other")
            print(f"{u:<5d} {v:.6f}  {kind}")
    print(f"unpaid share of the track (to the inference part): see the difference to the track share "
          f"{CS.profile_for(CS.SPLIT_PROFILE, datetime.fromtimestamp(now, timezone.utc)).tracks[0]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
