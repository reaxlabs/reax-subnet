"""Offline engine slot finalisation. Input is accepted state plus a fresh chain snapshot.

The engreview chain_snapshot.py list format is accepted with --block HEAD. An envelope
{block: HEAD, rows: [...]} carries that same list and its observed head together.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from . import competition_split as cs
from .engcomp import Champion, Submission, TrackState, finalize_engine_slot, parse_engine_commitment


def track_params():
    # Pure transition checks must import without chain/award/SDK dependencies.
    try:
        from .engvalidator import TRACKS
    except ImportError as exc:
        raise RuntimeError("engine slot dependencies unavailable; use the release CHAIN_PY virtualenv") from exc
    return TRACKS


def snapshot_submissions(snapshot, track, state, block=None, binds_hotkey=False):
    rows = snapshot if isinstance(snapshot, list) else snapshot["rows"]
    head = block if block is not None else snapshot.get("block") if isinstance(snapshot, dict) else None
    if not isinstance(head, int) or isinstance(head, bool) or head <= 0:
        raise ValueError("snapshot needs a positive head block (use --block with a chain_snapshot.py list)")
    out = []
    for row in rows:
        if not isinstance(row.get("uid"), int) or isinstance(row["uid"], bool):
            continue  # no longer registered
        text = row.get("text", "")
        pos = text.find("reax-eng/1:") if isinstance(text, str) else -1
        p = parse_engine_commitment(text[pos:]) if pos >= 0 else None
        if p is None or p[0] != track:
            continue
        hk, dg = row["hotkey"], p[2]
        if f"{dg}:{hk}" in state.not_owner:
            continue
        if binds_hotkey:
            if state.owners.get(dg) != hk:
                continue
        elif state.first_seen.get(dg, (0, None))[1] != hk:
            continue
        out.append(Submission(row["uid"], hk, *p, block=row["block"]))
    return out, head


def finalize_document(raw, snapshot, now, block=None):
    """Preserve every unrelated byte of JSON data; change only tracks which crown or record a compliance block."""
    out = deepcopy(raw)
    changed = False
    TRACKS = track_params()
    for name, before in raw["tracks"].items():
        if name not in TRACKS:
            continue
        st = TrackState.from_json(before)
        eligible, head = snapshot_submissions(snapshot, name, st, block, raw.get("binds_hotkey", False))
        crown = finalize_engine_slot(st, eligible, TRACKS[name], now, head)
        if crown is None and st.history == before.get("history", []):
            continue
        after = deepcopy(before)
        serialized = st.to_json()
        for key in ("champion", "challenger", "champion_invalid", "champion_last_session", "next_slot_utc", "pending", "slot_results", "history"):
            if key in serialized:
                after[key] = serialized[key]
        out["tracks"][name] = after
        changed = True
    if changed:
        out["saved_at"] = int(now.timestamp())
    return out


def check_transition(old, new, now, mode="final", snapshot=None, created_at=None):
    """Additional check_state gate: slot mode is an exact offline replay, batches crown only near slots.

    Raising ValueError rejects the file. The shell's existing context, history, review and chain checks still apply.
    """
    if mode == "weights":
        expected = deepcopy(old)
        if set(new.get("tracks", {})) != set(old.get("tracks", {})):
            raise ValueError("weights transition changed tracks")
        for name, before in old["tracks"].items():
            after = new["tracks"][name]
            events = after.get("history", [])[len(before.get("history", [])):]
            if len(events) != 1 or events[0].get("event") != "weights_ok":
                raise ValueError("weights transition needs one compliance history event per track")
            event = events[0]
            if after["history"] != before.get("history", []) + [event]:
                raise ValueError("weights transition rewrote existing history")
            allowed = set(before.get("weights_ok", []))
            digest, action = event.get("digest"), event.get("action")
            if action == "add": allowed.add(digest)
            elif action == "remove": allowed.discard(digest)
            else: raise ValueError("invalid compliance action")
            if set(after.get("weights_ok", [])) != allowed:
                raise ValueError("weights transition differs from compliance action")
            expected["tracks"][name]["weights_ok"] = after.get("weights_ok", [])
            expected["tracks"][name]["history"] = after["history"]
        if expected != new:
            raise ValueError("weights transition changed unrelated state")
        return
    if set(new.get("tracks", {})) != set(old.get("tracks", {})):
        raise ValueError("batch changed track set")
    for name, st in new["tracks"].items():
        prev = old["tracks"][name]
        if set(st.get("weights_ok", [])) != set(prev.get("weights_ok", [])):
            raise ValueError("box changed Sandy weights_ok approvals")
    if mode == "slot":
        if snapshot is None:
            raise ValueError("slot mode needs a fresh Sandy chain snapshot")
        if not any(0 <= (now - s).total_seconds() <= 4 * 3600 for s in cs.ENGINE_CROWN_SLOTS_UTC):
            raise ValueError("slot publication outside slot clock allowance")
        expected = finalize_document(old, snapshot, now)
        if expected != new:
            raise ValueError("slot state differs from deterministic pending-to-champion transition")
        return
    for name, st in new["tracks"].items():
        prev = old.get("tracks", {}).get(name, {})
        previous_slots, slots = prev.get("slot_results", []), st.get("slot_results", [])
        if slots[:len(previous_slots)] != previous_slots or len(slots) > len(previous_slots) + 1:
            raise ValueError("box rewrote or invented slot_results")
        added_slots = slots[len(previous_slots):]
        ch, och = st.get("champion"), prev.get("champion")
        crown_changed = ch is not None and (not och or (ch.get("digest"), ch.get("hotkey")) != (och.get("digest"), och.get("hotkey")))
        if added_slots and (not crown_changed or added_slots[0] not in st.get("history", []) or
                            added_slots[0].get("event") != "crowned" or
                            added_slots[0].get("digest") != ch.get("digest") or
                            added_slots[0].get("hotkey") != ch.get("hotkey") or
                            added_slots[0].get("block") != ch.get("crowned_block")):
            raise ValueError("slot_results addition has no matching transition crown")
        _check_challenger(prev, st, now, created_at)
        if not cs.ENGINE_CROWN_SLOTS_UTC:
            continue
        if ch is None or (och and (ch.get("digest"), ch.get("hotkey")) == (och.get("digest"), och.get("hotkey"))):
            continue  # vacating and champion rechecks keep their existing rules
        events = [e for e in st.get("history", []) if e.get("event") == "crowned"
                  and e.get("digest") == ch.get("digest") and e.get("hotkey") == ch.get("hotkey")
                  and e.get("block") == ch.get("crowned_block")]
        if not events:
            raise ValueError("champion change has no matching crown event")
        event = events[-1]
        try:
            slot = datetime.fromisoformat(event["slot_utc"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError):
            raise ValueError("champion change has no valid slot") from None
        # Every batch crown must reproduce the seed's confirmed pending challenger. A late worker
        # recovery is a scheduled-slot crown, never a new instant measurement crown.
        pending = prev.get("challenger") or {}
        recovery = (pending.get("digest"), pending.get("hotkey"), pending.get("slot_utc")) == (
            ch.get("digest"), ch.get("hotkey"), slot.isoformat()) and bool(pending.get("confirmed_at"))
        if slot not in cs.ENGINE_CROWN_SLOTS_UTC or slot > now or not recovery:
            raise ValueError("champion change outside an engine slot or without seed confirmations")
        seed = TrackState.from_json(prev)
        sub = Submission(ch["uid"], ch["hotkey"], name, "unused", ch["digest"])
        got = finalize_engine_slot(seed, [sub], track_params()[name], now, ch["crowned_block"])
        actual = asdict(Champion.from_json(ch))
        if got is None:
            raise ValueError("slot crown does not match seed confirmations")
        expected = asdict(got)
        # The cycle may have remeasured the newly crowned champion afterwards: its held score can only fall.
        if not (0 < actual["score"] <= expected["score"]):
            raise ValueError("batch raised the scheduled champion score")
        actual["score"] = expected["score"]
        if actual != expected or seed.history[-1] != event or event not in st.get("slot_results", []):
            raise ValueError("slot crown does not match seed confirmations and slot receipt")



def _check_challenger(prev, st, now, created_at):
    c = st.get("challenger")
    if c is None:
        return  # withdrawal, failed bar or crown
    seed = prev.get("challenger") or {}
    same = (c.get("digest"), c.get("hotkey")) == (seed.get("digest"), seed.get("hotkey"))
    old_ms = seed.get("measurements", []) if same else []
    ms = c.get("measurements", [])
    if not ms or ms[:len(old_ms)] != old_ms:
        raise ValueError("challenger rewrote seed measurements")
    additions = ms[len(old_ms):]
    result = st.get("results", {}).get(c.get("digest"), {})
    available = ([[result.get("block"), result.get("score"), result.get("session", "")]]
                 if result != prev.get("results", {}).get(c.get("digest")) else [])
    # Earlier results within this batch are retained as audit receipts by the worker.
    available += [[e.get("block"), e.get("score"), e.get("session", "")]
                  for e in st.get("cycle_events", [])
                  if e.get("event") == "measurement" and e.get("digest") == c.get("digest")
                  and e not in prev.get("cycle_events", []) and e.get("ok") is True]
    if any(m not in available for m in additions) or (additions and not result.get("ok")):
        raise ValueError("challenger measurements have no batch results")
    owner = st.get("owners", {}).get(c.get("digest")) or (st.get("first_seen", {}).get(c.get("digest")) or [None, None])[1]
    if owner != c.get("hotkey"):
        raise ValueError("challenger identity differs from measured owner")
    if seed.get("confirmed_at") and same:
        if any(c.get(k) != seed.get(k) for k in ("confirmed_at", "slot_utc")):
            raise ValueError("challenger changed seed confirmation")
    if not additions:
        if not same or any(c.get(k) != seed.get(k) for k in c if k != "uid"):
            raise ValueError("challenger changed without a batch result")
        return
    # Replay the rule for fresh, confirming AND already-confirmed challengers. Retain the
    # sitting champion in eligible so replay cannot vacate it and accidentally lower its bar.
    params = track_params()[result["track"]]
    replay = TrackState.from_json(st)
    replay.challenger = deepcopy(seed) if same else None
    replay.weights_ok = set(prev.get("weights_ok", []))
    sub = Submission(c["uid"], c["hotkey"], params.name, "unused", c["digest"])
    eligible = [sub]
    if replay.champion is not None:
        ch = replay.champion
        eligible.append(Submission(ch.uid, ch.hotkey, params.name, "unused", ch.digest))
    from .engcomp import copyline_active, update_champion
    confirmed = (datetime.fromisoformat(c["confirmed_at"].replace("Z", "+00:00"))
                 if c.get("confirmed_at") else now)
    if confirmed.tzinfo is None or confirmed > now:
        raise ValueError("challenger confirmation is in the future")
    if c.get("confirmed_at") and not (seed.get("confirmed_at") and same) and created_at is not None:
        if created_at.tzinfo is None or (created_at - confirmed).total_seconds() > 600:
            raise ValueError("challenger confirmation predates batch creation")
    # An established confirmation is preserved by update_champion; new confirmations
    # use their claimed time, bounded by Sandy creation/collection, to reproduce their calendar slot.
    replay_now = now if seed.get("confirmed_at") and same else confirmed
    for measurement in additions:
        r = type_result(result)
        r.block, r.score, r.session = measurement
        replay.results[c["digest"]] = r
        update_champion(replay, eligible, params, r.block, now=replay_now, copyline=copyline_active(replay_now))
    if replay.challenger != c:
        raise ValueError("challenger differs from batch measurement replay")


def type_result(raw):
    from .engcomp import EvalResult
    return EvalResult.from_json(raw)


def weights_document(raw, action, digest, review_dir, inventory):
    import getpass
    import hashlib
    import re
    from .engreview import load_inventory, weight_file_screen
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("digest must be 64 lowercase hex characters")
    inventory_hash = hashlib.sha256(Path(inventory).read_bytes()).hexdigest()
    if action == "add":
        if not (Path(review_dir) / "REVIEWED" / digest).is_file():
            raise ValueError("missing REVIEWED digest marker")
        findings = weight_file_screen(load_inventory(inventory))
        if findings:
            raise ValueError("engreview weights screen found weight-like files")
    who, when = getpass.getuser(), datetime.now(timezone.utc).isoformat()
    out = deepcopy(raw)
    event = dict(event="weights_ok", action=action, digest=digest, who=who, utc=when,
                 inventory_sha256=inventory_hash, inventory=str(Path(inventory).resolve()))
    for st in out["tracks"].values():
        allowed = set(st.get("weights_ok", []))
        if action == "add": allowed.add(digest)
        else: allowed.discard(digest)
        st["weights_ok"] = sorted(allowed)
        st.setdefault("history", []).append(dict(event))
    # The candidate is the compliance log: it is copied to durable history by the wrapper.
    # No append to the review directory or mutation of the input state is performed.
    return out, event


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("finalize")
    for name in ("state", "commitments", "now", "out"):
        f.add_argument("--" + name, required=True)
    f.add_argument("--block", type=int)
    w = sub.add_parser("weights-ok")
    w.add_argument("action", choices=("add", "remove", "list"))
    w.add_argument("--state", required=True)
    for name in ("digest", "review-dir", "inventory", "out"):
        w.add_argument("--" + name)
    args = ap.parse_args(argv)
    if args.command == "weights-ok":
        raw = json.loads(Path(args.state).read_text())
        if args.action == "list":
            print(json.dumps({name: st.get("weights_ok", []) for name, st in raw["tracks"].items()}, sort_keys=True))
            return
        if not all((args.digest, args.review_dir, args.inventory, args.out)):
            ap.error("add/remove require --digest --review-dir --inventory --out")
        if Path(args.out).resolve() == Path(args.state).resolve():
            ap.error("--out must differ from accepted state")
        try:
            out, event = weights_document(raw, args.action, args.digest, args.review_dir, args.inventory)
        except (ValueError, OSError) as exc:
            ap.error(str(exc))
        check_transition(raw, out, datetime.now(timezone.utc), "weights")
        target = Path(args.out)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n")
        tmp.replace(target)
        print(json.dumps(event, sort_keys=True))
        return
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
    if now.tzinfo is None:
        ap.error("--now must have a timezone")
    raw = json.loads(Path(args.state).read_text())
    snapshot = json.loads(Path(args.commitments).read_text())
    try:
        out = finalize_document(raw, snapshot, now, args.block)
    except RuntimeError as exc:
        ap.error(str(exc))
    target = Path(args.out)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(target)


if __name__ == "__main__":
    main()
