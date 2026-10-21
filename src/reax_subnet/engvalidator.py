"""The validator side of the engine competition (unreleased; docs/ENGINE-COMPETITION.md).

Hooked into the s1 validator loop: after a serving round, when an evaluation cycle is due, the validator measures
up to ``max_evals`` new engine submissions (oldest commitment first) plus the current champion, each against the
reference engine in the same session, updates the champion and mixes the competition share into the next weight
vector. Serving rounds go on during a cycle (background mode): the GPU is shared with the serving runtimes, which is
why every session measures the reference alone first and treats later reference runs far from that baseline as
inconclusive (engrunner.reference_baseline).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import asyncio
import secrets
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional

from .award import engine_submission_id, spec_submission_id
from .chain import Snapshot
from . import competition_split as _csplit
from .competition_split import (
    COMPETITIONS_START_UTC, SPLIT_PROFILE, NoSpecialistEvaluator, SpecialistEvaluator, active_tracks, combine_weights,
    ENGINE_RESTORE_PREVIOUS, engine_keepalive_enabled, keep_share_enabled,
)
from .engcomp import (
    Champion, EvalResult, RewardParams, Submission, TrackParams, TrackState, _valid, champion_fraction, competition_weights,
    co_champion_set, confirmation_due, former_champions, needs_remeasurement, observe, parse_engine_commitment, pending, protected_uids,
    track_thresholds, update_champion, waiting_uids,
)
from .spec.commitment import parse_commitment as parse_spec_commitment
from .engthresholds import FROZEN_SHA256, thresholds_sha256
from .engrunner import AwaitingReview, EngineError, InfraError, ReferenceError_, ReferenceFailedWithCandidate, ReferenceSelfCheckFailed, Runner, clean, evaluate, reference_alone_ok

log = logging.getLogger("reax.engcomp")

# Release candidates; calibrate on the track's GPU class before activation (docs/ENGINE-COMPETITION.md section 8).
TRACKS: dict[str, TrackParams] = {
    "s1-fast": TrackParams("s1-fast", pool="s1-fast", temperature=2.07),
}
STATE_VERSION = 1


def context_sha256(context: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rules_sha256(tracks: Mapping[str, TrackParams]) -> str:
    """sha256 over the threshold fields of the tracks in use; equals FROZEN_SHA256 for the release tracks."""
    if set(tracks) == {"s1-fast"}:
        return thresholds_sha256(track_thresholds(tracks["s1-fast"]))
    return thresholds_sha256({t: track_thresholds(p) for t, p in sorted(tracks.items())})


class EngineCompetition:
    def __init__(self, runner: Runner, reference_refs: Mapping[str, str], state_path: str,
                 tracks: Optional[Mapping[str, TrackParams]] = None, reward: Optional[RewardParams] = None,
                 interval_s: float = 6 * 3600, max_evals: int = 4, clock=time.monotonic,
                 start_utc: Optional[datetime] = COMPETITIONS_START_UTC, profile: str = SPLIT_PROFILE,
                 specialist: Optional[SpecialistEvaluator] = None,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 engine_start_utc: Optional[datetime] = None, award_gate: Optional[Any] = None,
                 chain_id: str = "", cycle_gate_file: str = "", cycle_gate_max_age_s: float = 3600.0):
        """``cycle_gate_file`` (REAX_ENGINE_CYCLE_GATE_FILE; unset = no gate): a cycle runs, and goes on starting
        sessions, only while this file exists and was modified less than ``cycle_gate_max_age_s`` ago; a cycle that ran
        ends with a ``cycle_end`` history event, so an orchestrator can bracket the candidate runs around the reference
        rotation. ``start_utc``: before it the weights are exactly the serving (1.1) weights and no cycle runs; None = always
        active (development networks and tests only; finney uses the release constant). ``profile``: split profile
        (release constant, competition_split.py). ``specialist``: plug-in for the use-case tracks.
        ``engine_start_utc``: the engine track's own start (finney: ``competition_split.release_engine_start()``, never
        while the track is held); None = with ``start_utc``. Before it no engine cycle runs and no champion is paid.
        ``award_gate``: ``award.AwardGate`` (finney); every award needs the owner's signed approval. None = no gate
        (development networks without REAX_AWARD_APPROVALS, tests). ``chain_id``: network and netuid; part of the
        measurement context (see ``measurement_context``)."""
        self.start_utc = start_utc
        self.engine_start_utc = engine_start_utc
        self.award_gate = award_gate
        self.cycle_gate_file = cycle_gate_file
        self.cycle_gate_max_age_s = cycle_gate_max_age_s
        self._gate_logged: Optional[bool] = None
        self.keep_share = keep_share_enabled(os.environ)   # REAX_SPEC_NC_KEEP_SHARE, default off
        self.engine_keepalive = engine_keepalive_enabled(os.environ)   # REAX_ENGINE_KEEPALIVE, default off
        self._held_logged = False
        self.profile = profile
        self.specialist = specialist or NoSpecialistEvaluator()
        self._utcnow = utcnow
        self.runner = runner
        self.tracks = dict(tracks or TRACKS)
        self.reference_refs = {t: r for t, r in reference_refs.items() if t in self.tracks}
        missing = [t for t in self.tracks if t not in self.reference_refs]
        if missing:
            raise ValueError(f"no reference engine image for track(s): {', '.join(missing)}")
        self.reward = reward or RewardParams()
        self.interval_s = interval_s
        self.max_evals = max_evals
        self.state_path = state_path
        self._clock = clock
        self._last_cycle: Optional[float] = None
        self.states: dict[str, TrackState] = {t: TrackState() for t in self.tracks}
        self.thresholds_sha256 = rules_sha256(self.tracks)
        self.state_thresholds_sha256: Optional[str] = None  # what the loaded state file was written under
        self.chain_id = chain_id
        self.context = self.measurement_context()
        self.context_sha256 = context_sha256(self.context)
        self.state_context_sha256: Optional[str] = None  # the context of the loaded state file (None: legacy file)
        self.context_reset = False
        self._loaded_context: Any = None
        # An unreadable/missing weights directory is not a context change: keep the saved state, run and pay nothing.
        self.context_ok = not str(self.context.get("weights", "")).startswith("unreadable:")
        self._load()
        log.info("engine competition thresholds sha256 %s (release %s)%s", self.thresholds_sha256, FROZEN_SHA256,
                 "" if self.thresholds_sha256 == FROZEN_SHA256 else " - NOT the frozen release thresholds")
        if self.state_thresholds_sha256 not in (None, self.thresholds_sha256):
            log.warning("engine competition state was written under thresholds %s, now %s: scores of different "
                        "protocols are not comparable", self.state_thresholds_sha256, self.thresholds_sha256)

    # ------------------------------------------------------------------ persistence
    def measurement_context(self) -> dict[str, Any]:
        """What a stored score is only valid under: the reference engine image (digest), the model weights identity, the
        frozen track parameters (thresholds and the track set) and the chain. A result measured under another context
        is not evidence about this one and must be re-measured and re-confirmed before it pays."""
        weights = getattr(self.runner, "weights_identity", None)
        return {"references": {t: self.reference_refs[t] for t in sorted(self.reference_refs)},
                "weights": weights() if callable(weights) else "",
                "thresholds": self.thresholds_sha256, "tracks": sorted(self.tracks), "chain": self.chain_id}

    def _load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            if raw.get("version") != STATE_VERSION:
                raise ValueError("version")
            self.state_thresholds_sha256 = raw.get("thresholds_sha256")
            self.state_context_sha256 = raw.get("context_sha256")
            self._loaded_context = raw.get("context")
            for t, st in raw.get("tracks", {}).items():
                if t in self.states:
                    self.states[t] = TrackState.from_json(st)
            if not self.context_ok:
                log.error("engine competition: weights directory unreadable (%s); saved state kept, no engine cycle "
                          "and no engine payment until it is readable", self.context.get("weights"))
            elif self.state_context_sha256 != self.context_sha256:
                # F3: a legacy file (no context) cannot show that it is compatible either
                why = ("state file without a measurement context" if self.state_context_sha256 is None
                       else "reference, weights, track parameters or chain changed")
                for st in self.states.values():
                    st.reset_measurements(0, why)
                self.context_reset = True
                log.warning("engine competition state %s: %s; saved results and champions are dropped and must be "
                            "re-measured and re-confirmed before they pay", self.state_path, why)
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001 - a bad file restarts the competition state, never the validator
            log.warning("engine competition state %s unreadable (%s); starting empty", self.state_path, exc)
            try:
                os.replace(self.state_path, self.state_path + ".bad")
            except OSError:
                pass

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.state_path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".engstate-", dir=directory)
        with os.fdopen(fd, "w") as fh:
            # saved_at: the validator in remote mode treats a file older than 26 h as missing (remote.py);
            # binds_hotkey: whether the runner that produced the results proved that an image belongs to its committer
            json.dump({"version": STATE_VERSION, "saved_at": int(time.time()),
                       "binds_hotkey": bool(getattr(self.runner, "binds_hotkey", False)),
                       "thresholds_sha256": self.thresholds_sha256,
                       # unknown context (weights unreadable): keep the one the loaded file was written under
                       "context": self.context if self.context_ok else self._loaded_context,
                       "context_sha256": self.context_sha256 if self.context_ok else self.state_context_sha256,
                       "tracks": {t: s.to_json() for t, s in self.states.items()}},
                      fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------ chain view
    def submissions(self, snap: Snapshot, exclude: set[int]) -> dict[str, list[Submission]]:
        out: dict[str, list[Submission]] = {t: [] for t in self.tracks}
        for n in snap.neurons:
            if n.uid in exclude:
                continue
            parsed = parse_engine_commitment(n.commitment)
            if parsed is None or parsed[0] not in self.tracks:
                continue
            track, image, digest = parsed
            out[track].append(Submission(n.uid, n.hotkey, track, image, digest, n.commitment_block))
        return out

    def is_engine_commitment(self, commitment: str) -> bool:
        return parse_engine_commitment(commitment) is not None

    # ------------------------------------------------------------------ evaluation cycle
    def active(self) -> bool:
        return self.start_utc is None or self._utcnow() >= self.start_utc

    def engine_active(self) -> bool:
        return self.active() and (self.engine_start_utc is None or self._utcnow() >= self.engine_start_utc)

    def gate_open(self) -> bool:
        """True without a gate file configured. With one: the file exists as a regular file and is fresh."""
        if not self.cycle_gate_file:
            return True
        try:
            st = os.stat(self.cycle_gate_file)
            age = time.time() - st.st_mtime
            ok = os.path.isfile(self.cycle_gate_file) and -60.0 <= age < self.cycle_gate_max_age_s
        except OSError:
            ok = False
        if ok != self._gate_logged:  # log the changes only
            self._gate_logged = ok
            log.info("engine competition: cycle gate %s is %s", self.cycle_gate_file, "OPEN" if ok else "CLOSED")
        return ok

    def due(self) -> bool:
        if not self.engine_active() or not self.context_ok or not self.gate_open():
            return False
        return self._last_cycle is None or self._clock() - self._last_cycle >= self.interval_s

    async def run_cycle(self, snap: Snapshot, exclude: set[int],
                        between: Optional[Callable[[], Awaitable[object]]] = None) -> dict[str, list[dict]]:
        """One evaluation cycle. ``between`` runs after every evaluation (the validator keeps its weights fresh).
        With a cycle gate: a closed gate means no cycle, and a cycle that ran ends with a ``cycle_end`` event."""
        if not self.gate_open():
            return {}
        report: dict[str, list[dict]] = {}
        try:
            report = await self._run_cycle(snap, exclude, between)
            return report
        finally:
            if self.cycle_gate_file:
                n = sum(len(v) for v in report.values())
                for st in self.states.values():
                    st.add_cycle_event({"block": snap.block, "event": "cycle_end", "evaluated": n,
                                        "gate_open": self.gate_open()})
                log.info("engine competition: cycle_end (%d evaluation(s), gate %s)", n,
                         "open" if self.gate_open() else "closed")
                self.save()

    async def _run_cycle(self, snap: Snapshot, exclude: set[int],
                         between: Optional[Callable[[], Awaitable[object]]] = None) -> dict[str, list[dict]]:
        self._last_cycle = self._clock()
        report: dict[str, list[dict]] = {}
        preflight = getattr(self.runner, "preflight", None)
        if preflight is not None:
            try:
                await preflight()
            except ReferenceSelfCheckFailed as exc:
                for st in self.states.values():
                    st.add_cycle_event({"block": snap.block, "event": "session_void", "reason": clean(str(exc))})
                log.warning("engine competition: VOID: %s", exc)
                self.save()
                return report
            except InfraError as exc:
                log.warning("engine competition: cycle skipped, retried next interval: %s", exc)
                return report
        bind = bool(getattr(self.runner, "binds_hotkey", False))
        for track, subs in self.submissions(snap, exclude).items():
            st = self.states[track]
            params = self.tracks[track]
            eligible = observe(st, subs, snap.block, first_committer_only=not bind)
            eligible = await self._owned(st, eligible, bind)
            awaiting = getattr(self.runner, "awaiting_review", None)
            todo = pending(st, [s for s in eligible if awaiting is None or not awaiting(s.ref)], self.max_evals)
            again = needs_remeasurement(st, eligible)
            ch = st.champion
            if ch is not None and todo and not any(s.digest == ch.digest for s in again):
                again += [s for s in eligible if s.digest == ch.digest and s.hotkey == ch.hotkey]
            todo += [s for s in again if s not in todo]  # challenger confirmation; champion re-measured when challenged
            report[track] = []
            before = st.champion.digest if st.champion else None
            stopped = await self._run_sessions(st, track, params, todo, snap, report[track], between)
            eligible = await self._owned(st, observe(st, subs, snap.block, first_committer_only=not bind), bind)
            ch = update_champion(st, eligible, params, snap.block)
            # Confirmation = a second, fresh session (new containers, new secret items) right after the first, in
            # this cycle; the champion is measured again with the challenger, so the comparison stays paired.
            for _ in range(params.confirmations - 1):
                if stopped or not confirmation_due(st, eligible, params):
                    break
                stopped = await self._run_sessions(st, track, params, needs_remeasurement(st, eligible), snap,
                                                   report[track], between)
                ch = update_champion(st, eligible, params, snap.block)
            if (ch.digest if ch else None) != before:
                log.info("engine competition [%s]: champion is now %s", track,
                         f"uid {ch.uid} {ch.digest[:12]} score {ch.score:.3f}" if ch else "vacant")
        self.save()
        return report

    async def _owned(self, st: TrackState, eligible: list[Submission], bind: bool) -> list[Submission]:
        """Keep only submissions whose image is bound to the committing hotkey (F1). An image carries one hotkey label, so
        a digest whose owner is known (verified when it was first evaluated) is checked without any lookup; a second
        hotkey committing a cached digest is marked ``not_owner`` and can neither take the title nor vacate the owner's.
        A digest with a result but no recorded owner (state of an older build) is looked up once; a failed lookup
        leaves the submission out of this cycle, undecided."""
        if not bind:
            return eligible
        out = []
        for s in eligible:
            owner = st.owners.get(s.digest)
            if owner is None and s.digest in st.results:
                ch = st.champion
                if ch is not None and ch.digest == s.digest and ch.hotkey == s.hotkey:
                    owner = s.hotkey  # crowned after an ownership check; nothing to ask
                else:
                    try:
                        owner = await asyncio.wait_for(self.runner.owner_of(s.ref), self.tracks[s.track].eval_budget_s)
                    except Exception as exc:  # noqa: BLE001 - undecided, never a verdict
                        log.warning("engine competition: owner of cached %s not verified (%r); left out of this cycle",
                                    s.digest[:12], exc)
                        continue
                    discard = getattr(self.runner, "discard", None)
                    if discard is not None:
                        await discard(s.ref)
                if owner == s.hotkey:
                    st.owners[s.digest] = owner
            if owner is not None and owner != s.hotkey:
                st.not_owner.add(f"{s.digest}:{s.hotkey}")
                log.info("engine competition: uid %d committed %s, which is bound to another hotkey; ignored",
                         s.uid, s.digest[:12])
                continue
            out.append(s)
        return out

    async def _run_sessions(self, st: TrackState, track: str, params: TrackParams, todo: list[Submission],
                            snap: Snapshot, report: list[dict], between) -> bool:
        """One fresh session per submission in ``todo``; True when the cycle must stop (our reference is broken)."""
        for sub in todo:
            if not self.gate_open():
                log.info("engine competition: the cycle gate closed; no further sessions in this cycle")
                return True
            res = await self._evaluate_one(st, track, params, sub, snap.block)
            if res == "stop":
                self.save()
                return True
            if isinstance(res, EvalResult):
                res.session = secrets.token_hex(8)  # identifies this measurement (confirmations never count one twice)
                st.results[sub.digest] = res
                st.attempts.pop(sub.digest, None)
                report.append({"uid": sub.uid, **res.to_json()})
                log.info("engine competition [%s] uid %d %s: %s score %.3f (%.2f vs %.2f dec/s, mean_d %s)%s",
                         track, sub.uid, sub.digest[:12], "VALID" if res.ok else "REJECTED", res.score,
                         res.throughput, res.ref_throughput,
                         "-" if res.mean_distance == float("inf") else f"{res.mean_distance:.3f}",
                         f" reason={res.reason}" if res.reason else "")
            self.save()
            if between is not None:
                try:
                    await between()
                except Exception as exc:  # noqa: BLE001
                    log.warning("engine competition: callback between evaluations failed: %r", exc)
        return False

    async def _evaluate_one(self, st: TrackState, track: str, params: TrackParams, sub: Submission, block: int):
        """An EvalResult to record, None (inconclusive, retried later) or "stop" (our reference is broken)."""
        reference = self.reference_refs[track]

        def fail(reason: str) -> EvalResult:
            return EvalResult(digest=sub.digest, track=track, ok=False, reason=reason, block=block)

        def inconclusive(reason: str):
            n = st.attempts[sub.digest] = st.attempts.get(sub.digest, 0) + 1
            log.warning("engine competition: uid %d %s inconclusive (%d/%d): %s", sub.uid, sub.digest[:12], n,
                        params.max_infra_attempts, reason)
            return fail(f"inconclusive {n} times: {reason}") if n >= params.max_infra_attempts else None

        progress: dict[str, Any] = {}
        try:
            if getattr(self.runner, "binds_hotkey", False):
                try:
                    owner = await asyncio.wait_for(self.runner.owner_of(sub.ref), params.eval_budget_s)
                except asyncio.TimeoutError:  # registry/pull trouble before any engine ran: not the candidate's fault
                    return inconclusive(f"owner lookup exceeded its {params.eval_budget_s:.0f} s budget")
                if owner != sub.hotkey:
                    discard = getattr(self.runner, "discard", None)
                    if discard is not None:
                        await discard(sub.ref)  # somebody else's image: give the disk back
                    st.not_owner.add(f"{sub.digest}:{sub.hotkey}")
                    log.info("engine competition: uid %d committed %s, which is bound to another hotkey (label %r); "
                             "ignored", sub.uid, sub.digest[:12], (owner or "")[:12])
                    return None
                st.owners[sub.digest] = owner
            review = getattr(self.runner, "review", None)
            if review is not None:
                try:
                    await asyncio.wait_for(review(sub.ref), params.eval_budget_s)
                except asyncio.TimeoutError:
                    return inconclusive(f"submission check exceeded its {params.eval_budget_s:.0f} s budget")
                except AwaitingReview as exc:  # no strike, no attempt counted: runs once the operator has reviewed it
                    log.warning("engine competition: uid %d %s waits for review (%s)", sub.uid, sub.digest[:12], exc)
                    discard = getattr(self.runner, "discard", None)
                    if discard is not None:
                        await discard(sub.ref)  # the record keeps the verdict; re-pulled once REVIEWED exists
                    return None
                except OSError as exc:  # our review directory/disk, not the submission: VOID, no attempt counted
                    log.error("engine competition: submission check of %s failed on this host (%r); not counted",
                              sub.digest[:12], exc)
                    discard = getattr(self.runner, "discard", None)
                    if discard is not None:
                        await discard(sub.ref)  # retried next cycle: do not keep a pulled image meanwhile
                    return None
            return await asyncio.wait_for(
                evaluate(self.runner, reference, sub.ref, track, params, block=block, digest=sub.digest,
                         progress=progress),
                params.eval_budget_s)
        except asyncio.TimeoutError:
            if not progress.get("measuring"):
                # the budget ran out in pull/start/warm-up, before any timed request: no measurement failed, so this is
                # an infrastructure attempt (bounded by max_infra_attempts), not a verdict on the submission
                return inconclusive(f"evaluation exceeded its {params.eval_budget_s:.0f} s budget before the first "
                                    "timed request")
            return fail(f"evaluation exceeded its {params.eval_budget_s:.0f} s budget")
        except EngineError as exc:  # e.g. an image above the size limit, a registry that is not allowed
            return fail(clean(str(exc)))
        except InfraError as exc:
            return inconclusive(str(exc))
        except ReferenceSelfCheckFailed as exc:
            st.add_cycle_event({"block": block, "event": "session_void", "digest": sub.digest,
                                "reason": clean(str(exc))})
            log.warning("engine competition: VOID reference self-check (%s); no score or strike", exc)
            return "stop"
        except ReferenceFailedWithCandidate as exc:
            if await reference_alone_ok(self.runner, reference, params):
                return inconclusive(f"reference failed while this candidate was loaded ({exc})")
            log.error("engine competition: reference engine is broken (%s); cycle stopped, nothing recorded", exc)
            return "stop"
        except ReferenceError_ as exc:
            st.add_cycle_event({"block": block, "event": "session_void", "digest": sub.digest,
                                "reason": clean(str(exc))})
            log.error("engine competition: reference engine failed (%s); cycle stopped, nothing recorded", exc)
            return "stop"
        except Exception as exc:  # noqa: BLE001 - a submission must never stop the validator
            log.warning("engine competition: evaluation of uid %d crashed: %r", sub.uid, exc)
            return inconclusive(f"internal: {type(exc).__name__}")

    # ------------------------------------------------------------------ weights
    def weights(self, base: Mapping[int, float], snap: Snapshot, exclude: set[int],
                burn_fraction: float = 0.0) -> dict[int, float]:
        """Mix the competitions into ``base`` (competition_split.combine_weights). Before the start time, or with
        nothing to pay, ``base`` is returned unchanged (= the 1.1 weights)."""
        if not self.active():
            return dict(base)
        champion: Optional[int] = None
        frac = 0.0
        cochamps: dict[int, float] = {}
        waiting: list[int] = []
        if not self.engine_active() or not self.context_ok:
            if not self._held_logged:
                log.info("engine competition: the engine track is held (not open); no engine champion is paid and the "
                         "engine share goes to the inference part")
                self._held_logged = True
            tracks: list[str] = []
        else:
            self._held_logged = False
            tracks = list(self.tracks)
        subs = self.submissions(snap, exclude) if tracks else {}
        for track in tracks:
            st = self.states[track]
            bind = bool(getattr(self.runner, "binds_hotkey", False))
            eligible = [s for s in subs[track] if f"{s.digest}:{s.hotkey}" not in st.not_owner
                        and (bind or st.first_seen.get(s.digest, (0, s.hotkey))[1] == s.hotkey)
                        and (not bind or st.owners.get(s.digest, s.hotkey) == s.hotkey)]
            if self.engine_keepalive and track == "s1-fast":
                waiting = waiting_uids(st, eligible, snap.block, {n.uid: n.coldkey for n in snap.neurons})
            ch = st.champion
            live = next((s for s in eligible if ch is not None and s.digest == ch.digest and s.hotkey == ch.hotkey),
                        None)
            if ch is None or live is None:
                continue  # champion withdrew since the last cycle: no competition weight until re-evaluated
            ch.uid = live.uid  # UIDs can be renumbered; the title belongs to the hotkey
            if self.engine_keepalive and track == "s1-fast" and self.award_gate is not None:
                # protected entries first (ENGINE_KA_PROTECTED_MAX): champion awaiting approval, newest former champion(s)
                approved = self.award_gate.approved(track, ch.hotkey, engine_submission_id(ch.digest))
                prot = protected_uids(st, eligible, self.tracks[track], snap.block, approved)
                waiting = prot + [u for u in waiting if u not in prot]
            comp = competition_weights(st, eligible, self.tracks[track], self.reward, snap.block)
            if track == "s1-fast" and ch.uid in comp:
                co = co_champion_set(ch)
                if co is not None and co[0] == track:
                    # opening co-champions (ENGINE_CO_CHAMPIONS): each listed entry gets 1/n of the champion's part,
                    # only with its own approval; an unpaid part goes to the inference part (no former-champion fallback)
                    since = min(ch.crowned_block, _csplit.ENGINE_CO_CHAMPION_CROWNED.get(track, ch.crowned_block))
                    part = champion_fraction(self.reward, snap.block - since) / len(co[1])
                    for digest, hotkey in co[1]:
                        sub = next((s for s in eligible if s.digest == digest and s.hotkey == hotkey), None)
                        if sub is None or (digest != ch.digest and not _valid(st.results.get(digest), self.tracks[track])):
                            continue
                        if self.award_gate is not None and not self.award_gate.approved(track, hotkey,
                                                                                        engine_submission_id(digest)):
                            continue
                        cochamps[sub.uid] = cochamps.get(sub.uid, 0.0) + part
                    continue
                if self.award_gate is not None and not self.award_gate.approved(track, ch.hotkey,
                                                                                engine_submission_id(ch.digest)):
                    # awaiting owner approval or rejected: the previous approved champion keeps the champion share
                    # (ENGINE_RESTORE_PREVIOUS); without one the engine share goes to the inference part
                    if ENGINE_RESTORE_PREVIOUS:
                        prev = self._previous_approved(track, st, eligible, snap.block)
                        if prev is not None:
                            champion, frac = prev
                    continue
                champion, frac = ch.uid, comp[ch.uid]
        ids = active_tracks(self._utcnow(), self.profile)
        results = self.specialist.track_results(ids, self._utcnow()) if ids else {}
        scores = {t: results[t] for t in ids if results.get(t)}
        ka = getattr(self.specialist, "keepalive", None)
        alive = ka(ids, self._utcnow()) if ka is not None and ids else {}
        alive = {t: alive[t] for t in ids if alive.get(t)}
        pv = getattr(self.specialist, "provisional", None)
        prov = pv(ids, self._utcnow()) if pv is not None and ids else {}
        prov = {t: prov[t] for t in ids if prov.get(t)}
        if champion is None and not cochamps and not scores and not alive and not prov and not waiting:
            return dict(base)  # nothing to pay: exactly the 1.1 weights
        return combine_weights(base, champion, frac, scores, self.profile, owner_uid=snap.owner_uid,
                               burn_fraction=burn_fraction, award_ok=self._spec_award_ok(snap), keepalive=alive,
                               provisional=prov, engine_waiting=waiting, keep_share=self.keep_share, now=self._utcnow().timestamp(),
                               engine_cochampions=cochamps or None)

    def _previous_approved(self, track: str, st: TrackState, eligible: list[Submission],
                           block: int) -> Optional[tuple[int, float]]:
        """(uid, champion fraction) of the most recently crowned former champion of ``track`` that is approved by the
        owner, still committed (and bound to its hotkey) and whose last result is still valid; None if there is none.
        Its decay runs on from its own crowning block."""
        if self.award_gate is None:
            return None
        for f in former_champions(st):
            sub = next((s for s in eligible if s.digest == f["digest"] and s.hotkey == f["hotkey"]), None)
            r = st.results.get(f["digest"])
            if sub is None or not _valid(r, self.tracks[track]) or not isinstance(f["block"], int):
                continue
            if co_champion_set(Champion(f["digest"], f["hotkey"], sub.uid, 0.0, f["block"])) is not None:
                continue  # opening co-champions are never paid as a former champion (their share goes to inference)
            if self.award_gate.approved(track, sub.hotkey, engine_submission_id(sub.digest)):
                return sub.uid, champion_fraction(self.reward, block - f["block"])
        return None

    def _spec_award_ok(self, snap: Snapshot) -> Optional[Callable[[str, int], bool]]:
        """The gate for a specialist rank: the uid's hotkey and the model it commits to now must be approved."""
        gate = self.award_gate
        if gate is None:
            return None
        by_uid = {n.uid: n for n in snap.neurons}

        def ok(track_id: str, uid: int) -> bool:
            n = by_uid.get(uid)
            c = parse_spec_commitment(n.commitment) if n is not None else None
            if c is None or c.track_id != track_id:
                return False
            return gate.approved(track_id, n.hotkey, spec_submission_id(c.repo, c.revision))
        return ok
