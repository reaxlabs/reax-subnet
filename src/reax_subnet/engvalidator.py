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
import math
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
from . import ka_override as _kao
from .competition_split import (
    COMPETITIONS_START_UTC, SPLIT_PROFILE, NoSpecialistEvaluator, SpecialistEvaluator, active_tracks, combine_weights,
    ENGINE_RESTORE_PREVIOUS, engine_keepalive_enabled, keep_share_enabled, legacy_ka_tracks, legacy_ka_uids,
)
from . import engcopyline, engreview
from .engcomp import (
    EvalResult, RewardParams, Submission, TrackParams, TrackState, _valid, champion_fraction, competition_weights,
    block_at_utc, carried_view, co_champion_set, confirmation_due, copy_digests, former_ranks, needs_remeasurement, observe, parse_engine_commitment, pending,
    finalize_engine_slot, next_engine_slot, protected_uids, track_thresholds, update_champion, waiting_uids, copyline_active, parent_reference_due,
    parent_session_active, ENGINE_PARENT_RUNS_PER_CYCLE, ENGINE_PARENT_RUN_BUDGET_S, prune_parent_sessions,
)
from .spec.commitment import parse_commitment as parse_spec_commitment
from .engthresholds import FROZEN_SHA256, thresholds_sha256
from .engcomposite import CompositeTrack, SlotTapes, composite_rules_sha256, evaluate_composite, parse_declared
from .engrunner import AwaitingReview, EngineError, InfraError, ItemTape, ReferenceError_, ReferenceFailedWithCandidate, ReferenceSelfCheckFailed, Runner, clean, evaluate, reference_alone_ok

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


BRIDGE_ENDED_FILE = "engine-v1-bridge-ended.json"  # next to the engine state; one-way
BRIDGE_END_BLOCK_FILE = "engine-v1-bridge-end-block.json"  # first block observed at/after the bridge end time; one-way
BRIDGE_TRACK = "s1-fast"  # the v1 engine track the payout bridge keeps paying (EngineCompetition.v1_bridge)



def _switch_slot_final(st: TrackState) -> bool:
    """True if ``st`` (the v1 state) shows the Tue 27 Oct 14:30Z slot (competition_split.ENGINE_V2_START_UTC) as finalised: a
    ``slot_results`` entry for it, a crowning at or after it, a ``slot_closed`` event of the slot finaliser (the slot passed
    without a crowning, engslot.close_switch_slot) or a slot schedule that already moved past it."""
    start = _csplit.ENGINE_V2_START_UTC
    if start is None:
        return True
    def at_or_after(raw: Any, strict: bool = False) -> bool:
        try:
            when = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return False
        return when.tzinfo is not None and (when > start if strict else when >= start)
    return (any(at_or_after(e.get("slot_utc")) for e in st.slot_results)
            or any(e.get("event") in ("crowned", "slot_closed") and at_or_after(e.get("slot_utc")) for e in st.history)
            or (st.next_slot_utc is not None and at_or_after(st.next_slot_utc, strict=True)))


def read_frozen_v1(path: str, pinned: Optional[str]) -> tuple[TrackState, bool]:
    """(frozen v1 ``s1-fast`` state, whether v1 bound images to hotkeys) from ``path``. Raises ValueError (or OSError) unless the
    file exists, hashes to ``pinned``, parses, and holds the finalisation of the switch slot (a file frozen before the Tue 27 Oct
    14:30Z slot lacks that slot's winner). Used by the payout bridge on every read and by the finney start check."""
    with open(path, "rb") as fh:
        data = fh.read()
    digest = hashlib.sha256(data).hexdigest()
    if not pinned or digest != pinned.lower():
        raise ValueError(f"sha256 {digest} is not the pinned {pinned}")
    raw = json.loads(data.decode("utf-8"))
    if raw.get("version") != STATE_VERSION:
        raise ValueError(f"version {raw.get('version')!r}")
    st = TrackState.from_json(raw["tracks"][BRIDGE_TRACK])
    if not _switch_slot_final(st):
        raise ValueError("the file was frozen before the %s slot was finalised (no slot result, crowning, slot_closed event or "
                         "later slot schedule at or after it); run the slot finaliser after the slot, then freeze" % (_csplit.ENGINE_V2_START_UTC,))
    return st, bool(raw.get("binds_hotkey", True))  # v1 ran the docker runner: images bound to their hotkey


class EngineCompetition:
    def __init__(self, runner: Runner, reference_refs: Mapping[str, str], state_path: str,
                 tracks: Optional[Mapping[str, TrackParams]] = None, reward: Optional[RewardParams] = None,
                 interval_s: float = 6 * 3600, max_evals: int = 4, clock=time.monotonic,
                 start_utc: Optional[datetime] = COMPETITIONS_START_UTC, profile: str = SPLIT_PROFILE,
                 specialist: Optional[SpecialistEvaluator] = None,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 engine_start_utc: Optional[datetime] = None, award_gate: Optional[Any] = None,
                 chain_id: str = "", cycle_gate_file: str = "", cycle_gate_max_age_s: float = 3600.0,
                 composites: Optional[Mapping[str, CompositeTrack]] = None,
                 slot_references: Optional[Mapping[str, Mapping[str, str]]] = None,
                 v1_bridge: Optional[tuple[str, datetime, str]] = None,
                 composite_start_utc: Optional[datetime] = None):
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
        measurement context (see ``measurement_context``). ``composites``: composite tracks (engcomposite.py, engine
        competition v2), name -> CompositeTrack; each is measured on every declared model and its composite score runs
        through the same champion rule. With composites and no ``tracks``, only the composites are run.
        ``slot_references``: composite track -> {slot: reference image} (one reference per model); a slot without an
        entry uses the track's reference in ``reference_refs``. Part of the measurement context.
        ``v1_bridge``: (path of the frozen v1 state file, end time, sha256 of that file) or None (both release constants
        competition_split.ENGINE_V1_BRIDGE_END_UTC and ENGINE_V1_BRIDGE_STATE_SHA256 must be set): until the end time,
        and until a composite track is crowned for the first time, the v1 ``s1-fast`` champion of that file is paid
        exactly as in v1 (``weights``). The first crowning ends the bridge for good (one-way marker file next to the
        state, see ``_bridge_state``); a composite state that was not loaded this round pays no bridge either. The
        file must hash to the pinned digest on every read. The bridge touches neither the cycle, the saved state nor
        the measurement context (its identity is logged separately, ``bridge_identity``). When the bridge is over (first
        composite crowning or its end time) the v1 title holder is a former champion of the composite, as if beaten at
        that moment: it gets the 17 % place for 7 days and the v1 former champions shift down (``carried_view``), paid
        from the frozen v1 state (its results, ownership and the v1 award approvals). ``composite_start_utc`` (finney:
        competition_split.ENGINE_V2_START_UTC): before it no composite cycle runs and no composite entry is paid; a
        composite is crowned only at a slot strictly after it (engcomp.finalize_engine_slot)."""
        self.composite_start_utc = composite_start_utc
        if v1_bridge is not None and v1_bridge[1].tzinfo is None:  # a naive end time is UTC (never compare naive with aware)
            v1_bridge = (v1_bridge[0], v1_bridge[1].replace(tzinfo=timezone.utc), v1_bridge[2])
        self.v1_bridge = v1_bridge
        self._bridge_unwritable_logged = False
        self._bridge_down = False
        self._bridge_ended = False
        self.composite_state_ok = True  # remote mode: the composite state was read successfully in this round
        self.start_utc = start_utc
        self.engine_start_utc = engine_start_utc
        self.award_gate = award_gate
        self._verify_ownership: Optional[Callable[[str, str, str], bool]] = None  # tests; else sr25519_verifier(), lazily
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
        self.composites: dict[str, CompositeTrack] = dict(composites or {})
        self.slot_references = {t: dict(m) for t, m in (slot_references or {}).items() if t in self.composites}
        self._carried: dict[tuple[str, str], str] = {}  # (digest, hotkey) -> the older track it was committed to
        self._declared: dict[str, frozenset[str]] = {}  # composite: models a digest declared in its latest session
        self.tracks = dict(tracks) if tracks is not None else ({} if self.composites else dict(TRACKS))
        self.tracks.update({name: c.params() for name, c in self.composites.items()})
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
        self._last_tape: Optional[tuple[str, ItemTape]] = None  # (digest, items) of the latest session, for a parent replay
        self._parent_runs = 0  # parent reference runs started in this cycle (ENGINE_PARENT_RUNS_PER_CYCLE)
        self._parent_gpu_s = 0.0  # seconds the composite parent runs of this cycle used (at most one session budget of the child's models)
        self._parent_failed: set[tuple[str, str]] = set()  # (track, parent digest) whose run failed in this cycle
        self.states: dict[str, TrackState] = {t: TrackState() for t in self.tracks}
        self.thresholds_sha256 = rules_sha256(self.tracks)
        if self.composites:  # the per-model rules of a composite are part of what a stored score means
            self.thresholds_sha256 = context_sha256({"tracks": self.thresholds_sha256, "composites": {
                n: composite_rules_sha256(c) for n, c in sorted(self.composites.items())}})
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
        if self.v1_bridge is not None:  # not part of the measurement context (that must not change): logged for audit
            log.info("engine v1 payout bridge: state %s sha256 %s, ends %s", self.v1_bridge[0], self.v1_bridge[2],
                     self.v1_bridge[1].isoformat())
        log.info("engine competition thresholds sha256 %s (release %s)%s", self.thresholds_sha256, FROZEN_SHA256,
                 "" if self.thresholds_sha256 == FROZEN_SHA256 else " - NOT the frozen release thresholds")
        if self.composites:  # every validator must report the same two values before the first dm-all crowning (operations check)
            log.info("engine v2 composite rules sha256 %s, measurement context sha256 %s",
                     ", ".join(f"{n}={composite_rules_sha256(c)}" for n, c in sorted(self.composites.items())), self.context_sha256)
        if self.state_thresholds_sha256 not in (None, self.thresholds_sha256):
            log.warning("engine competition state was written under thresholds %s, now %s: scores of different "
                        "protocols are not comparable", self.state_thresholds_sha256, self.thresholds_sha256)

    # ------------------------------------------------------------------ persistence
    def measurement_context(self) -> dict[str, Any]:
        """What a stored score is only valid under: the reference engine image (digest), the model weights identity, the
        frozen track parameters (thresholds and the track set) and the chain. A result measured under another context
        is not evidence about this one and must be re-measured and re-confirmed before it pays."""
        weights = getattr(self.runner, "weights_identity", None)
        refs: dict[str, Any] = {t: self.reference_refs[t] for t in sorted(self.reference_refs)}
        for t, m in sorted(getattr(self, "slot_references", {}).items()):
            refs[t] = {"default": refs.get(t), **{s: m[s] for s in sorted(m)}}
        return {"references": refs,
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
        if _csplit.ENGINE_CROWN_SLOTS_UTC:
            for st in self.states.values():
                st.next_slot_utc = next_engine_slot(self._utcnow(), st)
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
        self._carried = {}  # rebuilt from the current chain view: a re-commit to the composite drops the old entry
        for n in snap.neurons:
            if n.uid in exclude:
                continue
            parsed = parse_engine_commitment(n.commitment)
            if parsed is None:
                continue
            track, image, digest = parsed
            if track not in self.tracks:  # an older track a composite carries (v1 s1-fast -> dm-all, Plumb only)
                track = next((c for c, comp in self.composites.items()
                              if comp.carried_declaration(track) is not None), "")
                if not track:
                    continue
                self._carried[(digest, n.hotkey)] = parsed[0]
            out[track].append(Submission(n.uid, n.hotkey, track, image, digest, n.commitment_block))
        return {t: (v if self._track_open(t) else []) for t, v in out.items()}

    def is_engine_commitment(self, commitment: str) -> bool:
        return parse_engine_commitment(commitment) is not None

    # ------------------------------------------------------------------ evaluation cycle
    def active(self) -> bool:
        return self.start_utc is None or self._utcnow() >= self.start_utc

    def engine_active(self) -> bool:
        return self.active() and (self.engine_start_utc is None or self._utcnow() >= self.engine_start_utc)

    def composites_active(self) -> bool:
        """Engine v2 is in force (``composite_start_utc``; None = always)."""
        return self.composite_start_utc is None or self._utcnow() >= self.composite_start_utc

    def _track_open(self, track: str) -> bool:
        return track not in self.composites or self.composites_active()

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
        if not any(self._track_open(t) for t in self.tracks):
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
        self._parent_runs, self._parent_failed, self._parent_gpu_s = 0, set(), 0.0
        report: dict[str, list[dict]] = {}
        if not any(self._track_open(t) for t in self.tracks):
            return report  # before the v2 start nothing is measured, not even a preflight
        # Slot finalisation is independent of GPU preflight and runs even if this batch cannot measure.
        if _csplit.ENGINE_CROWN_SLOTS_UTC:
            slot_now = self._utcnow()
            bind = bool(getattr(self.runner, "binds_hotkey", False))
            for track, subs in self.submissions(snap, exclude).items():
                if not self._track_open(track):
                    continue
                st, params = self.states[track], self.tracks[track]
                elig = observe(st, subs, snap.block, first_committer_only=not bind)
                if bind:
                    elig = [s for s in elig if st.owners.get(s.digest) == s.hotkey]
                if copyline_active(slot_now):
                    self._apply_copy_markers(st, elig, snap.block)
                    self._refresh_derived(st, elig, snap, params)
                    elig = self._unpaused(st, elig, snap.block)
                before = st.champion.digest if st.champion else None
                if finalize_engine_slot(st, elig, params, slot_now, snap.block):
                    if st.champion.digest != before:
                        log.info("engine competition [%s]: champion is now uid %d %s (%.3f)",
                                 track, st.champion.uid, st.champion.digest[:12], st.champion.score)
                    self.save()
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
            if not self._track_open(track):
                continue
            st = self.states[track]
            params = self.tracks[track]
            eligible = observe(st, subs, snap.block, first_committer_only=not bind)
            eligible = await self._owned(st, eligible, bind)
            awaiting = getattr(self.runner, "awaiting_review", None)
            cl = copyline_active(self._utcnow())  # ENGINE_COPYLINE_START_UTC; False while the constant is None
            if cl:
                await self._recheck_measured(st, eligible)
                self._apply_copy_markers(st, eligible, snap.block)
                self._refresh_derived(st, eligible, snap, params)
            elig = self._unpaused(st, eligible, snap.block) if cl else eligible
            limit = min(self.max_evals, self.composites[track].max_evals) if track in self.composites else self.max_evals
            todo = pending(st, [s for s in elig if awaiting is None or not awaiting(s.ref)], limit)
            again = needs_remeasurement(st, elig)
            ch = st.champion
            if ch is not None and todo and not any(s.digest == ch.digest for s in again):
                again += [s for s in eligible if s.digest == ch.digest and s.hotkey == ch.hotkey]
            todo += [s for s in again if s not in todo]  # challenger confirmation; champion re-measured when challenged
            report[track] = []
            before = st.champion.digest if st.champion else None
            stopped = await self._run_sessions(st, track, params, todo, snap, report[track], between)
            eligible = await self._owned(st, observe(st, subs, snap.block, first_committer_only=not bind), bind)
            if cl:  # the static check writes a new image's review record inside the session: read it before any crowning
                self._apply_copy_markers(st, eligible, snap.block)
                self._refresh_derived(st, eligible, snap, params)
                eligible = self._unpaused(st, eligible, snap.block)
            ch = update_champion(st, eligible, params, snap.block, copyline=cl, now=self._utcnow())
            # Confirmation = a second, fresh session (new containers, new secret items) right after the first, in
            # this cycle; the champion is measured again with the challenger, so the comparison stays paired.
            for _ in range(params.confirmations - 1):
                if stopped or not confirmation_due(st, eligible, params):
                    break
                stopped = await self._run_sessions(st, track, params, needs_remeasurement(st, eligible), snap,
                                                   report[track], between)
                if cl:
                    self._refresh_derived(st, eligible, snap, params)
                ch = update_champion(st, eligible, params, snap.block, copyline=cl, now=self._utcnow())
            if cl:
                prune_parent_sessions(st)
            if (ch.digest if ch else None) != before:
                log.info("engine competition [%s]: champion is now %s", track,
                         f"uid {ch.uid} {ch.digest[:12]} score {ch.score:.3f}" if ch else "vacant")
        self.save()
        return report

    async def _recheck_measured(self, st: TrackState, eligible: list[Submission]) -> None:
        """Copy line: an eligible digest measured before the start has a review record without the ``copyline`` mark and is
        never reviewed again (review runs for unmeasured digests only). Its static check is re-run once (labels only, the
        runner's ``recheck_copyline``); until then it is not crownable (``_refresh_derived`` leaves it out of ``derived``)
        but stays in the pool. A failure leaves it for the next cycle."""
        recheck = getattr(self.runner, "recheck_copyline", None)
        if recheck is None:
            return
        for s in eligible:
            if s.digest in st.results:
                try:
                    await recheck(s.ref)
                except Exception as exc:  # noqa: BLE001 - retried next cycle; the digest stays not crownable
                    log.warning("engine competition: copy-line re-check of %s failed (%r)", s.digest[:12], exc)

    def _ownership_verifier(self, rec: Mapping[str, Any]) -> Optional[Callable[[str, str, str], bool]]:
        """The sr25519 verifier, loaded only when an image actually carries an ownership proof (None: no proof counts)."""
        if self._verify_ownership is None and rec.get("ownership_label"):
            self._verify_ownership = engcopyline.sr25519_verifier()
        return self._verify_ownership

    def _unpaused(self, st: TrackState, eligible: list[Submission], block: int) -> list[Submission]:
        """Copy line: a paused hotkey's submissions are out of the challenger pool, the confirmation and the crowning until
        the pause ends. Its sitting champion entry stays (it is not a new entry; the payouts it already earns continue)."""
        ch = st.champion
        return [s for s in eligible if not st.copy_paused(s.hotkey, block)
                or (ch is not None and (s.digest, s.hotkey) == (ch.digest, ch.hotkey))]

    def _apply_copy_markers(self, st: TrackState, eligible: list[Submission], block: int) -> None:
        """Copy line: ``COPY/<digest>`` on an eligible image (measured or not, reviewed or not) = a failed result and a strike
        in this cycle (once per digest; the second and every later one pauses the hotkey). A copy that holds the title
        is vacated and a pending challenge of it dropped; the payout helpers leave copies out (``copy_digests``)."""
        review_dir = getattr(self.runner, "review_dir", "")
        if not review_dir:
            return
        for s in eligible:
            if not engreview.valid_digest(s.digest) or not engreview.copy_marker(review_dir, s.digest):
                continue
            n = st.record_copy(s.hotkey, s.digest, block)
            r = st.results.get(s.digest)
            if r is None or r.ok:
                log.warning("engine competition: uid %d %s is a copy (undeclared reuse of another entry's code), strike %d",
                            s.uid, s.digest[:12], n)
                st.results[s.digest] = EvalResult(digest=s.digest, track=s.track, ok=False, block=block,
                                                  reason="copy: undeclared reuse of another entry's code (declare parents "
                                                         f"in the image label {engcopyline.PARENTS_LABEL})")
            if st.challenger is not None and st.challenger.get("digest") == s.digest:
                st.challenger = None
            ch = st.champion
            if ch is not None and (ch.digest, ch.hotkey) == (s.digest, s.hotkey):
                st.history.append({"block": block, "event": "vacated", "digest": ch.digest, "hotkey": ch.hotkey})
                st.champion, st.champion_invalid, st.champion_last_session = None, 0, ""

    def _refresh_derived(self, st: TrackState, eligible: list[Submission], snap: Snapshot,
                         params: Optional[TrackParams] = None) -> None:
        """Copy line: from each eligible image's review record set ``st.derived[digest]`` = the declared parents it must beat
        by the full margin, without the ones the same owner holds (same coldkey in the snapshot, or a valid signed ownership
        proof); ``[]`` = read, no parents. An entry is crownable only with a key here: a missing or unreadable record, a
        label with any problem, or an error while reading it leaves the key out (fail closed, that digest only; the cycle
        goes on). Also keeps the best valid score of every digest (``held_scores``). Recomputed from the record each time."""
        for r in st.results.values():
            st.hold(r, params)
        review_dir = getattr(self.runner, "review_dir", "")
        if not review_dir:
            return
        coldkeys = {n.hotkey: n.coldkey for n in snap.neurons}
        uid_hotkey = {n.uid: n.hotkey for n in snap.neurons}
        for s in eligible:
            try:
                self._derive_one(st, s, review_dir, coldkeys, uid_hotkey)
            except Exception as exc:  # noqa: BLE001 - one bad record must not stop the cycle
                st.derived.pop(s.digest, None)
                log.warning("engine competition: copy-line record of %s unreadable (%r); not crownable", s.digest[:12], exc)

    def _derive_one(self, st: TrackState, s: Submission, review_dir: str, coldkeys: Mapping[str, str],
                    uid_hotkey: Mapping[int, str]) -> None:
        rec = engreview.load_record(review_dir, s.digest) if engreview.valid_digest(s.digest) else None
        if not rec or not rec.get("copyline"):  # no record, or written before the start (no label fields): not read
            st.derived.pop(s.digest, None)
            return
        parents, problems = engcopyline.parse_parents(rec.get("parents_label"))
        if problems:
            st.derived.pop(s.digest, None)
            log.warning("engine competition: parents label of %s has problems (%s); not crownable", s.digest[:12],
                        "; ".join(problems))
            return
        own = []
        hk_of: dict[str, list[str]] = {}
        for p in parents:
            phk = st.owners.get(p.digest) or st.first_seen.get(p.digest, (0, ""))[1]  # the parent digest's committer
            fixed = st.derived_hk.get(s.digest, {}).get(p.digest)
            if fixed is not None:  # read once for this child: a later uid re-registration changes nothing
                hk_of[p.digest] = fixed
            elif phk:  # the committer decides; a declared uid that is someone else's is a label problem
                if uid_hotkey.get(p.uid) not in (None, phk):
                    st.derived.pop(s.digest, None)
                    log.warning("engine competition: parents label of %s names uid %d for parent %s, which was committed by "
                                "another hotkey; not crownable", s.digest[:12], p.uid, p.digest[:12])
                    return
                hk_of[p.digest] = [phk]
            elif uid_hotkey.get(p.uid):  # committer unknown: the declared uid's hotkey, fixed at this first reading
                hk_of[p.digest] = [uid_hotkey[p.uid]]
            if phk and (engcopyline.same_coldkey(s.hotkey, phk, coldkeys) or engcopyline.valid_ownership_proof(
                    rec.get("ownership_label"), p.digest, phk, s.hotkey, self._ownership_verifier(rec))):
                own.append(p.digest)
        if hk_of:
            st.derived_hk[s.digest] = hk_of
        st.derived[s.digest] = engcopyline.constraint_parents((p.digest for p in parents), (), own)

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
                if _csplit.ENGINE_CROWN_SLOTS_UTC:
                    st.add_cycle_event({"event": "measurement", "digest": sub.digest, "block": res.block,
                                        "score": res.score, "session": res.session, "ok": res.ok})
                if copyline_active(self._utcnow()):
                    st.hold(res, params)  # copy line: the best valid score survives a later re-measurement
                st.attempts.pop(sub.digest, None)
                tape = self._last_tape[1] if self._last_tape and self._last_tape[0] == sub.digest else None
                self._last_tape = None
                if parent_session_active(self._utcnow()):
                    await self._parent_reference(st, track, params, sub, res, snap, tape)
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

        tape = SlotTapes() if track in self.composites else ItemTape()  # per-model tapes on a composite track
        self._last_tape = (sub.digest, tape)
        progress: dict[str, Any] = {"items": tape}
        review_dir = getattr(self.runner, "review_dir", "")
        if review_dir and copyline_active(self._utcnow()) and engreview.copy_marker(review_dir, sub.digest):
            # review found undeclared reuse (COPY/<digest>): not evaluated; a strike (the second pauses the hotkey)
            n = st.record_copy(sub.hotkey, sub.digest, block)
            log.warning("engine competition: uid %d %s is a copy (undeclared reuse of another entry's code), strike %d",
                        sub.uid, sub.digest[:12], n)
            return fail("copy: undeclared reuse of another entry's code (declare parents in the image label "
                        f"{engcopyline.PARENTS_LABEL})")
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
            comp = self.composites.get(track)
            if comp is not None:
                labels = getattr(self.runner, "models_label", None)
                old = self._carried.get((sub.digest, sub.hotkey))
                try:
                    declared = (comp.carried_declaration(old) if old else None) or parse_declared(
                        await labels(sub.ref) if labels is not None else None, comp)
                except ValueError as exc:
                    return fail(clean(str(exc)))
                self._declared[sub.digest] = declared  # a parent's reference run in this session measures the same models
                work = evaluate_composite(self.runner, reference, sub.ref, comp, declared, block=block,
                                          digest=sub.digest, progress=progress,
                                          references=self.slot_references.get(track))
                try:
                    return await asyncio.wait_for(work, comp.budget_s(declared))
                except asyncio.TimeoutError:
                    # every model has its own budget inside (a slow measurement fails there); the overall cap is ours
                    return inconclusive(f"composite evaluation exceeded its {comp.budget_s(declared):.0f} s cap")
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

    async def _parent_reference(self, st: TrackState, track: str, params: TrackParams, sub: Submission, res: EvalResult,
                                snap: Snapshot, tape: Optional[ItemTape]) -> None:
        """Parent measured in the same session (engcomp.ENGINE_PARENT_SESSION_START_UTC): when ``sub``'s session just made it
        a near-miss of its stored parent line (engcomp.parent_reference_due), measure the binding parent image right after
        it on this box, against the same reference, with the same items (``tape`` replayed; fresh items when the session
        recorded none) and store the run under the child's session id (``st.parent_sessions``). A reference run only: it
        never writes ``results``, ``held_scores``, ``attempts`` or strikes. Every failure (image not committed by anyone any
        more, not pullable, not reviewed, failed, over budget, gate closed) stores a failed run: the stored line applies."""
        if not res.ok or not res.session:
            return
        try:  # the child's record may have been written in this very session
            self._refresh_one(st, sub, snap)
        except Exception as exc:  # noqa: BLE001
            log.warning("engine competition: copy-line record of %s unreadable (%r)", sub.digest[:12], exc)
        parent = parent_reference_due(st, sub.digest, params)
        if parent is None:
            return
        entry: dict[str, Any] = {"child": sub.digest, "parent": parent, "ok": False, "score": 0.0, "block": snap.block,
                                 "session": "", "items": "same" if tape is not None and tape.recorded else "fresh"}
        # the image name comes from the parent's own committer (verified owner, else first sighting): anybody can commit the
        # same digest under another name, and that name must not decide whether the reference run can pull the image
        owner_hk = st.owners.get(parent) or st.first_seen.get(parent, (0, ""))[1]
        ref = next((f"{img}@sha256:{d}" for n in snap.neurons if owner_hk and n.hotkey == owner_hk
                    for t, img, d in [parse_engine_commitment(n.commitment) or ("", "", "")]
                    if t == track and d == parent), None)
        reason = ""
        if ref is None:
            reason = "parent image not committed by its owner"
        elif not self.gate_open():
            reason = "cycle gate closed"
        elif (track, parent) in self._parent_failed:
            reason = "parent run failed earlier in this cycle"
        elif self._parent_runs >= ENGINE_PARENT_RUNS_PER_CYCLE:
            log.info("engine competition [%s] uid %d %s: no parent run (cap of %d per cycle reached); the stored line applies",
                     track, sub.uid, sub.digest[:12], ENGINE_PARENT_RUNS_PER_CYCLE)
            return
        elif (track in self.composites and self.composites[track].budget_s(self._declared.get(sub.digest) or frozenset())
              - self._parent_gpu_s <= 0):
            log.info("engine competition [%s] uid %d %s: no parent run (the composite parent runs of this cycle used one "
                     "session budget); the stored line applies", track, sub.uid, sub.digest[:12])
            return
        else:
            self._parent_runs += 1
            comp = self.composites.get(track)
            progress: dict[str, Any] = {"items": tape.replay() if tape is not None and tape.recorded
                                        else (SlotTapes() if comp is not None else ItemTape())}
            try:
                if comp is not None:
                    # composite track: the parent is measured on the models the child declared, with the child's items per
                    # model, inside the composite's own cap for those models (a failed or missing model = a failed run)
                    declared = self._declared.get(sub.digest)
                    if not declared:
                        raise EngineError("the child's declared models are unknown")
                    left = comp.budget_s(declared) - self._parent_gpu_s  # all parent runs of a cycle: one session budget
                    began = self._clock()
                    try:
                        pr = await asyncio.wait_for(
                            evaluate_composite(self.runner, self.reference_refs[track], ref, comp, declared,
                                               block=snap.block, digest=parent, progress=progress,
                                               references=self.slot_references.get(track)),
                            left)
                    finally:
                        self._parent_gpu_s += max(0.0, self._clock() - began)
                else:
                    pr = await asyncio.wait_for(evaluate(self.runner, self.reference_refs[track], ref, track, params,
                                                         block=snap.block, digest=parent, progress=progress),
                                                min(params.eval_budget_s, ENGINE_PARENT_RUN_BUDGET_S))
                good = bool(pr.ok) and math.isfinite(pr.score)  # a non-finite score is no run (it would read as 0.0)
                entry.update(ok=good, score=float(pr.score) if good else 0.0, session=secrets.token_hex(8))
                reason = "" if good else clean(pr.reason or "non-finite score")
            except Exception as exc:  # noqa: BLE001 - a reference run never stops the cycle; the stored line applies
                reason = clean(f"{type(exc).__name__}: {exc}")
        if reason:
            entry["reason"] = reason
            if ref is not None:
                self._parent_failed.add((track, parent))
        st.parent_sessions[res.session] = entry
        st.add_cycle_event({"event": "parent_reference", **{k: v for k, v in entry.items() if k != "items"},
                            "child_session": res.session, "child_score": round(res.score, 4)})
        log.info("engine competition [%s] uid %d %s: parent %s measured in the same session: %s%s", track, sub.uid,
                 sub.digest[:12], parent[:12], f"score {entry['score']:.3f}" if entry["ok"] else "no valid run",
                 f" ({reason})" if reason else "")
        self.save()

    def _refresh_one(self, st: TrackState, sub: Submission, snap: Snapshot) -> None:
        review_dir = getattr(self.runner, "review_dir", "")
        if not review_dir:
            return
        self._derive_one(st, sub, review_dir, {n.hotkey: n.coldkey for n in snap.neurons},
                         {n.uid: n.hotkey for n in snap.neurons})

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
            tracks = [t for t in self.tracks if self._track_open(t)]
        subs = self.submissions(snap, exclude) if tracks else {}
        states = self.states
        # the bridge also pays before the v2 start (the composite is not open yet, the v1 title holder is still paid)
        bridge, bridge_bind = (self._bridge_state(tracks)
                               if tracks or (self.composites and self.engine_active() and self.context_ok) else (None, False))
        if bridge is not None:  # v1 payee for the time being: the s1-fast track as v1 pays it, from the frozen state
            states = {**self.states, BRIDGE_TRACK: bridge}
            subs[BRIDGE_TRACK] = self._v1_submissions(snap, exclude, bridge)
            tracks = tracks + [BRIDGE_TRACK]
        for track in tracks:
            st = states[track]
            bind = bridge_bind if track == BRIDGE_TRACK and track not in self.tracks \
                else bool(getattr(self.runner, "binds_hotkey", False))  # the bridge keeps the v1 file's ownership mode
            eligible = [s for s in subs[track] if f"{s.digest}:{s.hotkey}" not in st.not_owner
                        and (bind or st.first_seen.get(s.digest, (0, s.hotkey))[1] == s.hotkey)
                        and (not bind or st.owners.get(s.digest, s.hotkey) == s.hotkey)]
            paid = track == "s1-fast" or track in self.composites  # the engine tracks (award.ENGINE_TRACK_IDS)
            keepalive = self.engine_keepalive and paid and track in self.tracks  # the bridge pays shares, no keep-alive
            ch = st.champion
            cl = copyline_active(self._utcnow())
            bad = copy_digests(st) if cl else frozenset()  # copies earn no former-champion rank and no keep-alive
            carry = self._carry(st, snap.block) if track in self.composites else None
            if paid and carry is not None:  # v1 -> v2: the v1 holder and its former champions are former champions of dm-all
                ranks = self._carried_ranks(st, carry, snap, exclude, bad | (copy_digests(carry[0]) if cl else frozenset()))  # + v1 copy strikes
            else:
                ranks = former_ranks(st, track, snap.block, bad) if paid else []
            if keepalive:
                waiting = waiting_uids(st, self._unpaused(st, eligible, snap.block) if cl else eligible, snap.block, {n.uid: n.coldkey for n in snap.neurons},
                                       {n.uid: n.registered_block for n in snap.neurons})
            live = next((s for s in eligible if ch is not None and s.digest == ch.digest and s.hotkey == ch.hotkey),
                        None)
            if ch is None and carry is not None:
                # no composite champion (yet): the carried former champions keep their places, on the decayed part
                # counted from the crowning of the newest of them (the title holder of the former rules)
                pot = champion_fraction(self.reward, snap.block - ranks[0]["block"]) if ranks else 0.0
                for g in ranks:
                    if g["share"] > 0:
                        self._add(cochamps, self._parts(track, st, eligible, g["pairs"], pot * g["share"], gated=True,
                                                        src=g.get("src")))
                continue
            if ch is None or live is None:
                continue  # champion withdrew since the last cycle: no competition weight until re-evaluated
            ch.uid = live.uid  # UIDs can be renumbered; the title belongs to the hotkey
            if keepalive and self.award_gate is not None:
                # protected entries first (ENGINE_KA_PROTECTED_MAX): champion awaiting approval, former champions in a slot
                approved = self.award_gate.approved(track, ch.hotkey, engine_submission_id(ch.digest))
                prot = protected_uids(st, eligible, self._params(track), snap.block, approved, track, bad)
                waiting = prot + [u for u in waiting if u not in prot]
            comp = competition_weights(st, eligible, self._params(track), self.reward, snap.block)
            if paid and ch.uid in comp:
                # former-champion ranks (rc10.post1): rank 1 keeps what the held former slots do not take
                top = 1.0 - math.fsum(g["share"] for g in ranks)
                slots = [g for g in ranks if g["share"] > 0]
                pot = comp[ch.uid]  # the title holder's (decayed) part of the engine share
                co = co_champion_set(ch)
                if co is not None and co[0] == track:
                    # opening co-champions (ENGINE_CO_CHAMPIONS): each listed entry gets 1/n of the champion's part,
                    # only with its own approval; an unpaid part goes to the inference part
                    since = min(ch.crowned_block, _csplit.ENGINE_CO_CHAMPION_CROWNED.get(track, ch.crowned_block))
                    pot = champion_fraction(self.reward, snap.block - since)
                    self._add(cochamps, self._parts(track, st, eligible, co[1], pot * top, champion_digest=ch.digest))
                elif self.award_gate is not None and not self.award_gate.approved(track, ch.hotkey,
                                                                                  engine_submission_id(ch.digest)):
                    # awaiting owner approval or rejected: the previous approved champion keeps the champion share
                    # (ENGINE_RESTORE_PREVIOUS); without one the engine share goes to the inference part
                    if ENGINE_RESTORE_PREVIOUS:
                        prev = self._previous_approved(track, st, eligible, snap.block, ranks)
                        if prev is not None:
                            g, parts = prev
                            # it takes rank 1's share; its own slot is not paid on top and not held against rank 1
                            # (decision 1: the restored payee gets the full champion share of its pot, less other slots)
                            slots = [s for s in slots if s is not g]
                            top = 1.0 - math.fsum(s["share"] for s in slots)
                            parts = {uid: part * top for uid, part in parts.items()}
                            if g["co"] or slots:
                                self._add(cochamps, parts)
                            else:  # exactly the rule before the former-champion ranks
                                champion, frac = next(iter(parts.items()))
                else:
                    champion, frac = ch.uid, pot * top
                for g in slots:  # ranks 2 and 3: their share of the title holder's (decayed) part
                    self._add(cochamps, self._parts(track, st, eligible, g["pairs"], pot * g["share"],
                                                    gated=True, src=g.get("src")))
        now = self._utcnow()  # one instant for the track list, the profile and the payout (GENERALIST_START_UTC switch)
        ids = active_tracks(now, self.profile)
        results = self.specialist.track_results(ids, now) if ids else {}
        scores = {t: results[t] for t in ids if results.get(t)}
        ka = getattr(self.specialist, "keepalive", None)
        alive = ka(ids, now) if ka is not None and ids else {}
        alive = {t: alive[t] for t in ids if alive.get(t)}
        pv = getattr(self.specialist, "provisional", None)
        prov = pv(ids, now) if pv is not None and ids else {}
        prov = {t: prov[t] for t in ids if prov.get(t)}
        pr = getattr(self.specialist, "protected", None)
        protected = pr(ids, now) if pr is not None and ids else {}
        legacy: list[int] = []
        lg = getattr(self.specialist, "legacy", None)
        lids = legacy_ka_tracks(now, self.profile) if lg is not None else ()
        if lids:
            # legacy keep-alive (LEGACY_KA_*): ended DF/RA champion always, former champions only with approval
            try:
                legacy = legacy_ka_uids(lg(lids, now), lids, self._spec_award_ok(snap))
            except Exception as exc:  # noqa: BLE001 - fail closed: no legacy keep-alive, everything else unchanged
                log.warning("legacy keep-alive: could not compute the entries (%r); none paid", exc)
                legacy = []
        # operator keep-alive override (ka_override.py; waiting-state hotfix 26 Oct 2026): within the existing caps
        ov_path = _kao.path_for(getattr(self.award_gate, "path", "") or "") if self.award_gate is not None else ""
        if _csplit.WAITING_KA_HOTFIX and ov_path:
            try:
                ov_legacy, ov_wait = _kao.apply(_kao.load(ov_path, now), snap.neurons, now)
            except Exception as exc:  # noqa: BLE001 - fail closed: no override, everything else unchanged
                log.warning("keep-alive override: not applied (%r)", exc)
                ov_legacy, ov_wait = [], []
            legacy = legacy + [u for u in ov_legacy if u not in legacy]
            if ov_wait and "generalist" in ids:
                prov = {**prov, "generalist": prov.get("generalist", []) + [u for u in ov_wait if u not in prov.get("generalist", [])]}
        if champion is None and not cochamps and not scores and not alive and not prov and not waiting and not legacy:
            return dict(base)  # nothing to pay: exactly the 1.1 weights
        return combine_weights(base, champion, frac, scores, self.profile, owner_uid=snap.owner_uid,
                               burn_fraction=burn_fraction, award_ok=self._spec_award_ok(snap), keepalive=alive,
                               provisional=prov, engine_waiting=waiting, keep_share=self.keep_share, now=now.timestamp(),
                               engine_cochampions=cochamps or None, legacy_keepalive=legacy or None,
                               keepalive_protected=protected or None)

    def _params(self, track: str) -> TrackParams:
        return self.tracks[track] if track in self.tracks else TRACKS[track]  # the bridge track is not one of ours

    def _v1_submissions(self, snap: Snapshot, exclude: set[int], v1: Optional[TrackState] = None) -> list[Submission]:
        """The carried v1 entries that are paid now. A carried place (and, during the bridge, the champion share) belongs to
        the HOTKEY: it stays paid while the hotkey's current commitment is its original ``s1-fast`` image (the pair as in
        v1) or any commitment to a composite track (``dm-all``: the miner may compete in v2 without losing the earned
        right; every v1 image the frozen state ties to that hotkey is then kept). A different commitment, none, or a
        deregistered hotkey (the uid belongs to another hotkey now) loses it, as in v1."""
        out = []
        for n in snap.neurons:
            parsed = parse_engine_commitment(n.commitment) if n.uid not in exclude else None
            if parsed is None:
                continue
            if parsed[0] == BRIDGE_TRACK:
                out.append(Submission(n.uid, n.hotkey, BRIDGE_TRACK, parsed[1], parsed[2], n.commitment_block))
            elif parsed[0] in self.composites and v1 is not None:
                for d, (_b, first_hotkey) in v1.first_seen.items():
                    if v1.owners.get(d, first_hotkey) == n.hotkey:
                        out.append(Submission(n.uid, n.hotkey, BRIDGE_TRACK, "", d, n.commitment_block))
        return out

    def _bridge_marker(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(self.state_path)), BRIDGE_ENDED_FILE)

    def _marker_writable(self) -> bool:
        """A real create+write+remove in the marker directory (os.access would miss read-only mounts, quotas). The bridge
        pays only while the one-way ended marker could be persisted; else a restart with lost history would pay again."""
        d = os.path.dirname(self._bridge_marker())
        try:
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".bridge-probe-", dir=d)
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write("probe")
                    fh.flush()
                    os.fsync(fh.fileno())
            finally:
                os.unlink(tmp)
            return True
        except Exception as exc:  # noqa: BLE001
            if not self._bridge_unwritable_logged:
                log.warning("engine v1 payout bridge DISABLED: marker directory %s is not writable (%s: %s); "
                            "nothing is paid through the bridge", d, type(exc).__name__, exc)
            self._bridge_unwritable_logged = True
            return False

    def _end_bridge(self, why: str) -> None:
        """One-way: remember (atomically, durably) that the bridge is over. A write failure still ends it in memory."""
        self._bridge_ended = True
        path = self._bridge_marker()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".bridge-ended-", dir=os.path.dirname(path))
            with os.fdopen(fd, "w") as fh:
                json.dump({"ended_at": self._utcnow().isoformat(), "why": why}, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            log.warning("engine v1 payout bridge: cannot write %s (%s: %s); bridge disabled for good (in "
                        "memory; restarts are covered by the writability probe)", path, type(exc).__name__, exc)
        log.info("engine v1 payout bridge ended for good: %s", why)

    @staticmethod
    def _ever_crowned(st: TrackState) -> bool:
        return st.champion is not None or any(e.get("event") in ("crowned", "vacated") for e in st.history)

    def _bridge_state(self, tracks: list[str]) -> tuple[Optional[TrackState], bool]:
        """(frozen v1 ``s1-fast`` state, whether v1 bound images to hotkeys) while the payout bridge is on, else
        (None, False). On means: a bridge is configured, it has not ended, the composite state was loaded in this round
        (fail closed), no composite track has ever been crowned (a champion now, or a crowning/vacating in its history;
        the first sighting writes the durable marker ``engine-v1-bridge-ended.json``, which survives vacating, restarts
        and missing remote files), and the v1 file hashes to the pinned digest and has a champion. Never raises (an
        unusable file only means no bridge payment: that share goes to the inference part)."""
        if self.v1_bridge is None or not self.composites:
            return None, False
        path, end, pinned = self.v1_bridge
        if self._bridge_ended or os.path.exists(self._bridge_marker()):
            self._bridge_ended = True
            return None, False
        if not self._marker_writable():  # fail closed: the ended marker could not be persisted
            return None, False
        if self._utcnow() >= end or not self.composite_state_ok:
            return None, False
        if any(self._ever_crowned(self.states[t]) for t in self.composites if t in self.states):
            self._end_bridge("a composite track was crowned")
            return None, False
        loaded = self._load_frozen()
        if loaded is None:
            return None, False
        st, bind = loaded
        return (st, bind) if st.champion is not None else (None, False)

    def _load_frozen(self) -> Optional[tuple[TrackState, bool]]:
        """(frozen v1 ``s1-fast`` state, whether v1 bound images to hotkeys) if the file hashes to the pinned digest, parses
        and contains the finalisation of the Tue 27 Oct slot, else None (logged once). Never raises."""
        assert self.v1_bridge is not None
        path, _end, pinned = self.v1_bridge
        try:
            res = read_frozen_v1(path, pinned)
            self._bridge_down = False
            return res
        except Exception as exc:  # noqa: BLE001 - whatever a missing or corrupt file contains
            if not self._bridge_down:
                log.warning("engine v1 payout bridge: %s unusable (%s: %s); no v1 champion or carried place is paid", path,
                            type(exc).__name__, exc)
            self._bridge_down = True
            return None

    def _end_block(self, block: int) -> Optional[int]:
        """The chain block that stands for the bridge end. The first block this validator observed at or after the end time
        is persisted (``engine-v1-bridge-end-block.json`` next to the marker; capped at the 12 s/block estimate for a validator that
        looks late) and used from then on, so the real chain decides: no payout gap after the end time. Before it exists (or if
        it cannot be written) the smaller of the estimate and the current block is used."""
        assert self.v1_bridge is not None
        est = block_at_utc(self.v1_bridge[1])
        if est is None:  # no opening anchor (development network): no end-by-time place, as before
            return None
        path = os.path.join(os.path.dirname(self._bridge_marker()), BRIDGE_END_BLOCK_FILE)
        try:
            with open(path, encoding="utf-8") as fh:
                return int(json.load(fh)["block"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if block <= 0:
            return est
        value = min(est, block)  # a validator that looks late must not push the end past the estimate
        if self._utcnow() >= self.v1_bridge[1]:
            try:
                fd, tmp = tempfile.mkstemp(prefix=".bridge-end-block-", dir=os.path.dirname(path))
                with os.fdopen(fd, "w") as fh:
                    json.dump({"block": value, "observed_at": self._utcnow().isoformat()}, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except OSError as exc:
                log.warning("engine v1 payout bridge: cannot persist the end block (%s); using the estimate", exc)
        return value

    def _carry(self, st: TrackState, block: int = 0) -> Optional[tuple[TrackState, bool, TrackState]]:
        """(frozen v1 state, its ownership mode, ``carried_view``) once the payout bridge is over (a composite was crowned,
        the end time passed or the ended marker exists) and the frozen file is usable; else None. The v1 title holder is
        then a former champion of the composite, beaten when the bridge ended (the first composite crowning, or the
        bridge end block if that comes first). Nothing is carried while the composite state was not loaded this round."""
        if self.v1_bridge is None or not self.composites or not self.composite_state_ok:
            return None
        if not (self._bridge_ended or os.path.exists(self._bridge_marker()) or self._utcnow() >= self.v1_bridge[1]
                or any(self._ever_crowned(self.states[t]) for t in self.composites if t in self.states)):
            return None
        loaded = self._load_frozen()
        if loaded is None:
            return None
        v1, bind = loaded
        return v1, bind, carried_view(st, v1, self._end_block(block))

    def _carried_ranks(self, st: TrackState, carry: tuple[TrackState, bool, TrackState], snap: Snapshot,
                       exclude: set[int], bad: frozenset[str]) -> list[dict[str, Any]]:
        """``former_ranks`` of the composite track over the carried view; ranks whose entries were crowned on v1 carry
        ``src`` = (track, frozen state, eligible v1 submissions) for ``_parts``."""
        v1, bind, view = carry
        elig = [s for s in self._v1_submissions(snap, exclude, v1) if f"{s.digest}:{s.hotkey}" not in v1.not_owner
                and (bind or v1.first_seen.get(s.digest, (0, s.hotkey))[1] == s.hotkey)
                and (not bind or v1.owners.get(s.digest, s.hotkey) == s.hotkey)]
        mine = {(e.get("digest"), e.get("hotkey")) for e in st.history if e.get("event") == "crowned"}
        # the opening co-champions are listed under "s1-fast": they stay one carried former champion
        ranks = former_ranks(view, BRIDGE_TRACK, snap.block, bad)
        for g in ranks:
            if not any(p in mine for p in g["pairs"]):
                g["src"] = (BRIDGE_TRACK, v1, elig)
        return ranks

    def _previous_approved(self, track: str, st: TrackState, eligible: list[Submission], block: int,
                           ranks: list[dict[str, Any]]) -> Optional[tuple[dict[str, Any], dict[int, float]]]:
        """(rank entry, {uid: champion fraction}) of the most recently crowned former champion of ``track`` (``ranks``,
        no time limit) that is approved by the owner, still committed (and bound to its hotkey) and whose last result is
        still valid; None if there is none. Its decay runs on from its own crowning block.

        The opening co-champion set counts as ONE former champion (rc10.post1, Florian 21 Oct 2026 05:25Z): each listed
        entry that qualifies on its own gets 1/n (decay from the earlier of its crowning and ENGINE_CO_CHAMPION_CROWNED,
        so it never restarts); a single member is never paid the full share. When no member qualifies, the walk goes on to
        older former champions, as for a single former champion."""
        if self.award_gate is None:
            return None
        for g in ranks:
            src = g.get("src")
            since = (min(g["block"], _csplit.ENGINE_CO_CHAMPION_CROWNED.get(src[0] if src else track, g["block"]))
                     if g["co"] else g["block"])
            parts = self._parts(track, st, eligible, g["pairs"], champion_fraction(self.reward, block - since),
                                gated=True, src=src)
            if parts:
                return g, parts
        return None

    def _parts(self, track: str, st: TrackState, eligible: list[Submission], pairs: tuple[tuple[str, str], ...],
               frac: float, champion_digest: Optional[str] = None, gated: bool = False,
               src: Optional[tuple[str, TrackState, list[Submission]]] = None) -> dict[int, float]:
        """{uid: frac / n} for each of the n listed ``pairs`` that is still committed by its hotkey, holds a valid
        result (the current champion ``champion_digest`` needs none) and has its own approval (``gated``: no award gate =
        nobody is approved); the rest is not paid (it goes to the inference part). ``src``: (track, state, eligible) of the
        track the pairs were crowned on, for entries carried over from the v1 ``s1-fast`` track (results, ownership and
        approvals are those of that track)."""
        if src is not None:
            track, st, eligible = src
        part = frac / len(pairs)
        out: dict[int, float] = {}
        for digest, hotkey in pairs:
            sub = next((s for s in eligible if s.digest == digest and s.hotkey == hotkey), None)
            if sub is None or (digest != champion_digest and not _valid(st.results.get(digest), self._params(track))):
                continue
            if self.award_gate is None:
                if gated:
                    continue
            elif not self.award_gate.approved(track, hotkey, engine_submission_id(digest)):
                continue
            out[sub.uid] = out.get(sub.uid, 0.0) + part
        return out

    @staticmethod
    def _add(acc: dict[int, float], parts: Mapping[int, float]) -> None:
        for uid, part in parts.items():
            acc[uid] = acc.get(uid, 0.0) + part

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
