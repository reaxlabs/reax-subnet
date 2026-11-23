"""Operator-only pause scheduler. Run on rxnode-03; dry-run unless --write is given."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
from pathlib import Path
import tempfile
from datetime import datetime, timedelta, timezone

from . import need


def utc(value: str) -> datetime:
    at = need._time(value)
    if at is None or at.utcoffset() != timedelta(0):
        raise ValueError('time must be ISO UTC with Z or +00:00')
    return at.astimezone(timezone.utc)


def holder_times(track: str, paths: list[str], gate=None) -> list[float]:
    """Read title holders' effective run-out starts, including co-members and a frozen v1 bridge.

    Unknown crown times refuse scheduling; unapproved new-rule holders are skipped
    with a stderr warning. State paths are explicit so the operator
    selects the same engine/spec/bridge snapshots configured on the validator.
    """
    times = []
    found = False

    def add_time(value, member, identity):
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'title holder {member} missing crowning time; recover timestamp before scheduling')
        from .competition_split import runout_start
        reader = getattr(gate, "approved_at", None)
        start = runout_start(value, reader(*identity) if reader is not None else None)
        if start is None:
            print(f'WARNING: skipping title holder {member}: no verified approval time; run-out clock has not started', file=sys.stderr)
            return
        times.append(start)
    for path in paths:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw.get('tracks'), dict):
            raise ValueError('state must contain tracks')
        if track == 'engine':
            from .engcomp import TrackState, co_champion_set
            from .engvalidator import EngineCompetition
            for key, row in raw['tracks'].items():
                if key not in ('s1-fast', 'dm-all'):
                    continue
                found = True
                state = TrackState.from_json(row)
                ch = state.champion
                if ch is None:
                    continue
                co = co_champion_set(ch)
                pairs = co[1] if co else ((ch.digest, ch.hotkey),)
                for d, h in pairs:
                    add_time(EngineCompetition._crowned_time(state, d, h), f'{key} hotkey={h} digest={d}',
                             (key, h, 'sha256:' + d))
        elif track in raw['tracks']:
            from .spec.state import from_json
            from .spec.evaluator import cochamp_members, cochamp_since
            state = from_json(raw).tracks[track]
            found = True
            members = cochamp_members(state)
            group_since = cochamp_since(state) if members else None
            if not members and state.champion:
                members = [state.submissions[state.champion]]
            for m in members:
                at = m.champion_since_ts
                if at is None:
                    at = group_since
                add_time(at, f'{track} uid={m.uid} key={m.key}',
                         (track, m.hotkey, f'{m.repo}@{m.revision}'))
    if not found:
        raise ValueError('requested track missing from supplied state; supply the validator state (and v1 bridge if active)')
    return times


def schedule(path: Path, track: str, start: datetime, until: datetime | None,
             reason: str, crowns: list[float], announced: datetime | None, write: bool = False) -> dict:
    if any(start.timestamp() - t >= 7 * 86400 for t in crowns):
        if announced is None or start < announced + timedelta(hours=24):
            raise ValueError('run-out clock at least 7 days old: --announced-at and at least 24 hours notice required')
    if announced is not None and announced > datetime.now(timezone.utc):
        raise ValueError('announcement must already have happened')
    original = path.read_bytes() if path.exists() else None
    if original is not None and len(original) > need.MAX_BYTES:
        raise ValueError('existing need.json exceeds byte limit')
    raw = json.loads(original) if original is not None else {}
    problem = need.parse(raw)[2]
    if problem:
        raise ValueError(problem)
    step = dict(track=track, from_utc=start.isoformat(), until_utc=until.isoformat() if until else None, reason=reason)
    raw.setdefault('pauses', []).append(step)
    problem = need.parse(raw)[2]
    payload = (json.dumps(raw, indent=2) + '\n').encode()
    if problem or len(payload) > need.MAX_BYTES:
        raise ValueError(problem or 'result exceeds need.json byte limit')
    if write:
        # A backup is exclusive and durable before replacement; the CLI holds an operator lock.
        if original is not None:
            backup = path.with_name(path.name + '.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.bak')
            with backup.open('xb') as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
        fd, tmp = tempfile.mkstemp(prefix='.need-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                if original is not None:
                    os.fchmod(stream.fileno(), path.stat().st_mode & 0o777)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return step


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=Path('/w/approvals/need.json'))
    parser.add_argument('--track', required=True)
    parser.add_argument('--state', action='append', required=True, help='validator state; repeat for active v1 bridge')
    parser.add_argument('--from-utc', required=True)
    parser.add_argument('--until-utc')
    parser.add_argument('--announced-at')
    parser.add_argument('--reason', required=True)
    parser.add_argument('--write', action='store_true', help='commit atomically with backup; default only prints')
    args = parser.parse_args(argv)
    from .award import AwardGate
    gate = AwardGate.from_env(os.environ)
    try:
        start = utc(args.from_utc)
        until = utc(args.until_utc) if args.until_utc else None
        announced = utc(args.announced_at) if args.announced_at else None
        if args.write:
            with args.path.with_name(args.path.name + '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                step = schedule(args.path, args.track, start, until, args.reason,
                                holder_times(args.track, args.state, gate), announced, True)
        else:
            step = schedule(args.path, args.track, start, until, args.reason,
                            holder_times(args.track, args.state, gate), announced)
        print(json.dumps(step, sort_keys=True))
        print('Written with backup.' if args.write else 'Dry run; pass --write to apply.')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'refused: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
