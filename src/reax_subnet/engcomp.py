"""Engine competition ``reax-eng/1`` (unreleased; docs/ENGINE-COMPETITION.md).

Competitors do not serve traffic. A competitor hotkey commits a pinned engine image on chain::

    reax-eng/1:<track>:<image>@sha256:<64 hex>

The validator pulls that exact digest, runs it next to the track's reference engine on its own GPU in the same
session (no network, read-only, the same pinned weights mounted read-only) and drives both with the same fresh,
secretly perturbed decisions. This module holds the pure parts: commitment parsing, the correctness gate
(logit distance against the reference engine's own answers, s1scoring.logit_distance), the speed score, the
champion rule and the weight split. engrunner.py starts engines and measures them.

Rules (all from the validator's own measurements, nothing the engine reports is trusted):

* correctness: every timed answer is compared with the reference engine's answer for the same item:
  mean distance <= mean_tau, every item <= item_tau, top choice agrees on >= min_top_agreement of the items;
* anti-cache probes: items that repeat an earlier state with a different question (and the reverse) must also
  match the reference, so an engine that caches by a partial key fails;
* determinism: a re-run of some items at concurrency 1 must reproduce the timed answers within determinism_tau;
* speed: throughput ratio candidate / reference at the track's concurrency, median over interleaved runs;
  concurrency-1 latency may not get worse than max_latency_ratio x the reference;
* champion: the best valid submission that beats the reference by min_gain; a challenger must beat the
  champion's score by margin (copying the champion therefore never wins); an identical digest counts only for
  the hotkey that committed it first.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from . import competition_split as _cs
from .engconfigs import PINNED_REFERENCE_NOISE_STD, required_engine_margin
from .engthresholds import THRESHOLDS as _T
from .s1scoring import logit_distance

ENGINE_BLOCK_SECONDS = 12.0

# Copy line (Florian 21 Oct 2026, board #22920, variant B with closed binaries; docs/ENGINE-COMPETITION.md, "Reusing other
# entries"). Release constant, no environment override: validators must agree. None = off, the rules are exactly the
# previous ones. From this instant on: declared reuse of another entry's code (source or binaries) makes the entry
# "derived" (it must beat every parent by the full margin), undeclared reuse found by review is a copy (strike; the
# second one and every later one pauses the hotkey for ENGINE_COPYLINE_PAUSE_BLOCKS), the same owner may reuse its own code.
ENGINE_COPYLINE_START_UTC: Optional[datetime] = datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)
assert ENGINE_COPYLINE_START_UTC is None or ENGINE_COPYLINE_START_UTC.tzinfo is not None  # naive would raise every cycle
ENGINE_COPYLINE_PAUSE_BLOCKS = 7 * 24 * 3600 // 12  # 7 days of 12 s blocks
ENGINE_COPYLINE_PAUSE_AT_STRIKE = 2  # from the second detected undeclared reuse on (the first has no consequence beyond the copy)


def copyline_active(now: datetime) -> bool:
    """True from ``ENGINE_COPYLINE_START_UTC`` on (the start instant itself counts); never while the constant is None."""
    return ENGINE_COPYLINE_START_UTC is not None and now >= ENGINE_COPYLINE_START_UTC


# Parent measured in the same session (Florian 24 Oct 2026 13:02Z, card 18502, option A; docs/ENGINE-COMPETITION.md, "Reusing
# other entries"). Release constant, no environment override. None = off: the parent line is the stored one (best valid score
# ever), exactly the previous rule. From this instant on (and only while the copy line is active): when a derived entry's
# session clears the champion bar but is below, or within the margin of, its stored parent line, the binding parent image is
# measured right after it on the same box with the same items, and the line for that session is
# max(parent in this session, the parent image's lowest valid score) x (1 + margin). Not retroactive.
ENGINE_PARENT_SESSION_START_UTC: Optional[datetime] = datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc)
assert ENGINE_PARENT_SESSION_START_UTC is None or ENGINE_PARENT_SESSION_START_UTC.tzinfo is not None
PARENT_SESSIONS_MAX = 50
# Cost bounds of the reference runs: at most this many per engine cycle (all tracks), each with at most this many seconds
# (or the track's eval_budget_s when that is lower); a parent digest whose run failed or timed out earlier in the same
# cycle is not run again in that cycle (the stored line applies).
ENGINE_PARENT_RUNS_PER_CYCLE = 3
ENGINE_PARENT_RUN_BUDGET_S = 1200.0


def parent_session_active(now: datetime) -> bool:
    """True from ``ENGINE_PARENT_SESSION_START_UTC`` on while the copy line is active; never while the constant is None."""
    return copyline_active(now) and ENGINE_PARENT_SESSION_START_UTC is not None and now >= ENGINE_PARENT_SESSION_START_UTC
PROTOCOL_ENG = "reax-eng/1"
# Image references are kept short enough that the whole commitment fits the chain's 128-byte raw field.
_COMMIT_RE = re.compile(
    r"^reax-eng/1:(?P<track>[a-z0-9][a-z0-9-]{0,15}):"
    r"(?P<image>[a-z0-9][a-z0-9._/-]{0,79}(?::[0-9]{1,5}/[a-z0-9._/-]{1,60})?)"
    r"@sha256:(?P<digest>[0-9a-f]{64})$"
)
MAX_COMMITMENT_BYTES = 128


@dataclass(frozen=True)
class TrackParams:
    """Release constants of one competition track (to calibrate per GPU class before activation)."""

    name: str
    pool: str  # the reax-s1 pool whose model and temperature the engine serves
    temperature: float
    timed_items: int = _T["timed_items"]  # unique items per timed run
    concurrency: int = 16  # in-flight requests during a timed run
    runs: int = _T["runs"]  # interleaved reference/candidate pairs; the score is the median ratio
    latency_items: int = _T["latency_items"]  # items sent one at a time for the latency gate
    probe_pairs: int = _T["probe_pairs"]  # anti-cache probe pairs (same state + other question, same question + other state)
    determinism_items: int = _T["determinism_items"]  # timed items re-run at concurrency 1
    mean_tau: float = _T["mean_tau"]  # stricter than serving (0.20): the tolerance must not become a speed budget
    item_tau: float = _T["item_tau"]  # stricter than serving (2.0)
    min_top_agreement: float = _T["min_top_agreement"]
    determinism_tau: float = _T["determinism_tau"]  # the reference is bit-identical at concurrency 1; batched engines move ~0.01
    max_latency_ratio: float = _T["max_latency_ratio"]
    min_gain: float = _T["min_gain"]  # must beat the reference by 3 % to count at all
    margin: float = _T["margin"]  # must beat the champion's score by 3 % to take the title
    reference_noise_std: float = PINNED_REFERENCE_NOISE_STD
    confirmations: int = _T["confirmations"]  # separate sessions (new containers, fresh items; may follow in the same cycle) that must all clear the bar
    confirm_gap_blocks: int = _T["confirm_gap_blocks"]  # ... at least this many blocks apart (0: a fresh session right after the first is enough)
    eval_budget_s: float = 2700.0  # one evaluation, start to finish; over budget = the candidate fails
    max_infra_attempts: int = 3  # inconclusive tries (pull/start trouble) before a submission is recorded as failed
    pause_idle: bool = True  # pause the idle engine during every measurement (False only for calibration)
    reference_drift: float = 0.12  # a reference run outside +-12 % of its own solo baseline is inconclusive (docs: ENGINE-COMPETITION.md)
    start_timeout_s: float = 900.0
    request_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        required_engine_margin(self.reference_noise_std)
        ok = (
            self.temperature > 0 and self.timed_items >= 8 and 1 <= self.concurrency <= 256 and self.runs >= 1
            and 0 < self.mean_tau <= self.item_tau and 0 < self.min_top_agreement <= 1
            and self.determinism_tau > 0 and self.max_latency_ratio >= 1 and self.min_gain >= 0 and self.margin >= 0
            and self.determinism_items <= self.timed_items and self.confirmations >= 1
            and self.confirm_gap_blocks >= 0 and self.eval_budget_s > 0 and self.max_infra_attempts >= 1
            and 0 < self.reference_drift < 1
            and self.determinism_items >= 1 and self.latency_items >= 1 and self.probe_pairs >= 0
            and 0 < self.start_timeout_s < math.inf and 0 < self.request_timeout_s < math.inf
            and all(math.isfinite(x) for x in (
                self.temperature, self.mean_tau, self.item_tau, self.min_top_agreement, self.determinism_tau,
                self.max_latency_ratio, self.min_gain, self.margin, self.eval_budget_s, self.reference_drift))
        )
        if not ok:
            raise ValueError(f"invalid track parameters for {self.name}")


@dataclass(frozen=True)
class RewardParams:
    """How the competition's share of the miner weights is split (docs/ENGINE-COMPETITION.md section 6)."""

    share: float = 0.20  # of the miner part of every weight vector, only while a champion exists (80/20 split)
    champion_start: float = 1.0  # champion's part of the share when crowned
    champion_floor: float = 0.5  # ... decaying linearly to this; the decayed part goes back to the serving pools
    decay_blocks: int = 100_800  # 14 days at 12 s
    # Runner-ups (halving by rank) would receive the decayed part instead. Off: a rebuilt copy of the champion or of
    # the reference with a lucky measurement could hold a runner-up slot (docs/ENGINE-COMPETITION.md section 6).
    runner_ups: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.share <= 0.5 and 0 <= self.champion_floor <= self.champion_start <= 1
                and self.decay_blocks >= 1 and self.runner_ups >= 0):
            raise ValueError("invalid engine competition reward parameters")


@dataclass(frozen=True)
class Submission:
    uid: int
    hotkey: str
    track: str
    image: str
    digest: str
    block: int = 0  # on-chain commitment block (0 = unknown)

    @property
    def ref(self) -> str:
        return f"{self.image}@sha256:{self.digest}"


def commitment_text(track: str, image: str, digest: str) -> str:
    text = f"{PROTOCOL_ENG}:{track}:{image}@sha256:{digest}"
    if parse_engine_commitment(text) is None:
        raise ValueError("not a valid reax-eng/1 commitment (image name too long or invalid characters?)")
    return text


def parse_engine_commitment(text: Any) -> Optional[tuple[str, str, str]]:
    """``reax-eng/1:<track>:<image>@sha256:<digest>`` -> (track, image, digest); anything else -> None."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        return None
    m = _COMMIT_RE.match(text.strip())
    if m is None or ".." in m["image"] or "//" in m["image"]:
        return None
    return m["track"], m["image"], m["digest"]


# ------------------------------------------------------------------------------------------------ measurements
@dataclass
class EvalResult:
    """One validator's measurement of one engine digest against the reference in the same session."""

    digest: str
    track: str
    ok: bool
    reason: str = ""
    score: float = 0.0  # median throughput ratio candidate / reference
    throughput: float = 0.0  # decisions/s, median over the runs
    ref_throughput: float = 0.0
    latency_p50_s: float = math.inf
    ref_latency_p50_s: float = math.inf
    mean_distance: float = math.inf
    max_distance: float = math.inf
    top_agreement: float = 0.0
    probe_max_distance: float = math.inf
    determinism_max_distance: float = math.inf
    items: int = 0
    block: int = 0
    session: str = ""  # unique id of the evaluation session (new containers, fresh items); set by the validator

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, float) and not math.isfinite(v):
                out[k] = None
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "EvalResult":
        kw = {}
        for f in cls.__dataclass_fields__.values():
            if f.name not in raw:
                continue
            v = raw[f.name]
            if v is None and f.type == "float":
                v = math.inf
            kw[f.name] = v
        return cls(**kw)


Probs = Sequence[float]


def _argmax(p: Probs) -> int:
    return max(range(len(p)), key=lambda i: (p[i], -i))


@dataclass(frozen=True)
class Correctness:
    ok: bool
    reason: str
    mean_distance: float
    max_distance: float
    top_agreement: float


def check_correctness(answers: Mapping[str, Optional[Probs]], references: Mapping[str, Probs],
                      params: TrackParams, top_agreement: bool = True) -> Correctness:
    """Every reference item must be answered and match within the track tolerances.

    ``top_agreement=False`` skips the top-choice rate (small sets, where one honest near-tie would be > 1 %).
    """
    if not references:
        return Correctness(False, "no reference answers", math.inf, math.inf, 0.0)
    ds, agree = [], 0
    for ident in sorted(references):
        ref = references[ident]
        got = answers.get(ident)
        if got is None or len(got) != len(ref):
            return Correctness(False, f"missing or malformed answer for item {ident[:12]}", math.inf, math.inf, 0.0)
        ds.append(logit_distance(got, ref, params.temperature))
        agree += _argmax(got) == _argmax(ref)
    mean = math.fsum(ds) / len(ds)
    worst = max(ds)
    top = agree / len(ds)
    if worst > params.item_tau:
        return Correctness(False, f"item distance {worst:.3f} > {params.item_tau}", mean, worst, top)
    if mean > params.mean_tau:
        return Correctness(False, f"mean distance {mean:.3f} > {params.mean_tau}", mean, worst, top)
    if top_agreement and top < params.min_top_agreement:
        return Correctness(False, f"top choice agrees on {top:.1%} < {params.min_top_agreement:.0%}", mean, worst, top)
    return Correctness(True, "", mean, worst, top)


def max_distance(a: Mapping[str, Optional[Probs]], b: Mapping[str, Probs], temperature: float) -> float:
    worst = 0.0
    for ident, ref in b.items():
        got = a.get(ident)
        if got is None or len(got) != len(ref):
            return math.inf
        worst = max(worst, logit_distance(got, ref, temperature))
    return worst


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    if not s:
        return math.nan
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def speed_score(cand_throughputs: Sequence[float], ref_throughputs: Sequence[float]) -> float:
    """Median of the per-pair ratios (pairs measured back to back, so slow drift of the machine cancels)."""
    ratios = [c / r for c, r in zip(cand_throughputs, ref_throughputs) if r > 0 and math.isfinite(c)]
    return median(ratios) if ratios else 0.0


# ------------------------------------------------------------------------------------------------ champion state
@dataclass
class Champion:
    digest: str
    hotkey: str
    uid: int
    score: float
    crowned_block: int

    @classmethod
    def from_json(cls, raw: Any) -> "Champion":
        """Type-checked: a state file written by something else must not reach the payment code as garbage."""
        if not isinstance(raw, dict) or set(raw) != {"digest", "hotkey", "uid", "score", "crowned_block"}:
            raise ValueError("champion must be an object with digest, hotkey, uid, score and crowned_block")
        def is_int(v: Any) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)
        if not (isinstance(raw["digest"], str) and isinstance(raw["hotkey"], str) and is_int(raw["uid"])
                and is_int(raw["crowned_block"]) and isinstance(raw["score"], (int, float))
                and not isinstance(raw["score"], bool) and math.isfinite(raw["score"])):
            raise ValueError("champion field of the wrong type")
        return cls(raw["digest"], raw["hotkey"], raw["uid"], float(raw["score"]), raw["crowned_block"])


CYCLE_EVENTS_MAX = 100


@dataclass
class TrackState:
    results: dict[str, EvalResult] = field(default_factory=dict)  # digest -> latest result
    first_seen: dict[str, tuple[int, str]] = field(default_factory=dict)  # digest -> (block, hotkey) first committed
    champion: Optional[Champion] = None
    # A challenger that cleared the bar once: {"digest", "hotkey", "measurements": [[block, score], ...]}.
    challenger: Optional[dict[str, Any]] = None
    attempts: dict[str, int] = field(default_factory=dict)  # digest -> inconclusive tries so far
    not_owner: set[str] = field(default_factory=set)  # "digest:hotkey" pairs whose image is bound to another hotkey
    owners: dict[str, str] = field(default_factory=dict)  # digest -> hotkey its image label named when it was verified
    champion_invalid: int = 0
    champion_last_session: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)  # title changes, newest last (max 50)
    # cycle_end / session_void per cycle: audit only, kept apart so they cannot push title changes out of ``history``
    cycle_events: list[dict[str, Any]] = field(default_factory=list)  # newest last (max CYCLE_EVENTS_MAX)
    # Copy line (add-only; absent from the file while empty, so a rollback to a build without it reads the file unchanged):
    # hotkey -> digests found to reuse another entry's code undeclared; hotkey -> block until which it is paused;
    # child digest -> parent digests it must beat by the full margin (declared or detected, same-owner ones left out).
    copy_strikes: dict[str, list[str]] = field(default_factory=dict)
    copy_paused_until: dict[str, int] = field(default_factory=dict)
    derived: dict[str, list[str]] = field(default_factory=dict)
    # child digest -> parent digest -> hotkeys whose digests all set the bar for THAT child (the committer of the declared
    # digest, or the hotkey of the declared uid when the committer is unknown; fixed at the first reading, never shared
    # between children); digest -> best valid score ever measured (survives re-measurement).
    derived_hk: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    held_scores: dict[str, float] = field(default_factory=dict)
    # Parent measured in the same session (add-only like the copy-line fields): digest -> lowest valid score seen (the floor
    # of a same-session parent line); child session id -> the binding parent's reference run right after that session
    # ({"child", "parent", "ok", "score", "block", "session", "items"}). A reference run never touches ``results``,
    # ``held_scores`` or ``low_scores``: it changes nothing about the parent's own standing.
    low_scores: dict[str, float] = field(default_factory=dict)
    parent_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)

    next_slot_utc: Optional[str] = None
    slot_results: list[dict[str, Any]] = field(default_factory=list)
    weights_ok: set[str] = field(default_factory=set)  # compliance-reviewed digests, supplied by Sandy

    def hold(self, result: "EvalResult", params: Optional["TrackParams"] = None) -> None:
        """Copy line: remember the best valid score of a digest, so a parent's bar does not drop when it is measured again
        (or fails later), and the lowest VALID one (the floor of a same-session parent line: ok and at least the track's
        gain threshold, ``valid_min``; without ``params`` no floor is recorded). Not called while the copy line is off."""
        if result.ok:
            self.held_scores[result.digest] = max(self.held_scores.get(result.digest, 0.0), result.score)
            if params is not None and math.isfinite(result.score) and result.score >= valid_min(params):
                self.low_scores[result.digest] = min(self.low_scores.get(result.digest, math.inf), result.score)

    def record_copy(self, hotkey: str, digest: str, block: int) -> int:
        """Count an undeclared reuse (once per digest) against ``hotkey``; the second one and every later one pauses the
        hotkey until ``block`` + ENGINE_COPYLINE_PAUSE_BLOCKS (never shortening a pause that runs longer). Returns the
        hotkey's strike count."""
        seen = self.copy_strikes.setdefault(hotkey, [])
        if digest not in seen:
            seen.append(digest)
            if len(seen) >= ENGINE_COPYLINE_PAUSE_AT_STRIKE:
                self.copy_paused_until[hotkey] = max(self.copy_paused_until.get(hotkey, 0), block + ENGINE_COPYLINE_PAUSE_BLOCKS)
            self.add_cycle_event({"block": block, "event": "copy_strike", "hotkey": hotkey, "digest": digest,
                                  "strikes": len(seen)})  # audit only: must not push title changes out of ``history``
        return len(seen)

    def copy_paused(self, hotkey: str, block: int) -> bool:
        """A paused hotkey's new commitments are not evaluated until the pause block."""
        return block < self.copy_paused_until.get(hotkey, 0)

    def add_cycle_event(self, event: dict[str, Any]) -> None:
        self.cycle_events.append(event)
        del self.cycle_events[:-CYCLE_EVENTS_MAX]

    def reset_measurements(self, block: int, why: str) -> None:
        """Drop everything that was measured (results, champion, challenger, strikes) but keep what was observed on chain
        or verified about images (first sightings, image owners). Used when the measurement context changed: scores of
        another reference, weights, track configuration or chain are not evidence about the current one."""
        self.history.append({"block": block, "event": "context_changed", "reason": why,
                             "dropped_champion": self.champion.digest if self.champion else None,
                             "dropped_results": len(self.results)})
        self.results, self.attempts = {}, {}
        self.held_scores = {}
        self.low_scores, self.parent_sessions = {}, {}
        self.champion, self.challenger = None, None
        self.champion_invalid, self.champion_last_session = 0, ""

    def to_json(self) -> dict[str, Any]:
        return {
            **({"next_slot_utc": self.next_slot_utc, "pending": pending_crown(self),
                "crown_slots_utc": [s.isoformat() for s in _cs.ENGINE_CROWN_SLOTS_UTC]}
               if _cs.ENGINE_CROWN_SLOTS_UTC else {}),
            **({"slot_results": self.slot_results[-50:]} if self.slot_results else {}),
            "results": {d: r.to_json() for d, r in self.results.items()},
            "first_seen": {d: [b, h] for d, (b, h) in self.first_seen.items()},
            "champion": asdict(self.champion) if self.champion else None,
            **({"weights_ok": sorted(self.weights_ok)} if self.weights_ok else {}),
            "challenger": self.challenger,
            "attempts": self.attempts,
            "not_owner": sorted(self.not_owner),
            "owners": self.owners,
            "history": self.history[-50:],
            "cycle_events": self.cycle_events[-CYCLE_EVENTS_MAX:],
            "champion_invalid": self.champion_invalid,
            "champion_last_session": self.champion_last_session,
            **({"copy_strikes": self.copy_strikes} if self.copy_strikes else {}),
            **({"copy_paused_until": self.copy_paused_until} if self.copy_paused_until else {}),
            **({"derived": self.derived} if self.derived else {}),
            **({"derived_hk": self.derived_hk} if self.derived_hk else {}),
            **({"held_scores": self.held_scores} if self.held_scores else {}),
            **({"low_scores": self.low_scores} if self.low_scores else {}),
            **({"parent_sessions": self.parent_sessions} if self.parent_sessions else {}),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "TrackState":
        st = cls()
        approved = raw.get("weights_ok", [])
        # Only a list of exact digest strings grants compliance; malformed input fails closed.
        if isinstance(approved, list):
            st.weights_ok = {d for d in approved if isinstance(d, str) and re.fullmatch(r"[0-9a-f]{64}", d)}
        st.results = {d: EvalResult.from_json(r) for d, r in raw.get("results", {}).items()}
        st.first_seen = {d: (int(v[0]), str(v[1])) for d, v in raw.get("first_seen", {}).items()}
        ch = raw.get("champion")
        st.champion = Champion.from_json(ch) if ch else None
        st.next_slot_utc = raw.get("next_slot_utc")
        st.slot_results = list(raw.get("slot_results", []))[-50:]
        st.challenger = raw.get("challenger")
        if st.challenger is not None and not isinstance(st.challenger, dict):
            raise ValueError("challenger must be an object")
        st.attempts = {d: int(n) for d, n in raw.get("attempts", {}).items()}
        st.not_owner = set(raw.get("not_owner", []))
        st.owners = {str(d): str(h) for d, h in raw.get("owners", {}).items()}
        st.champion_invalid = int(raw.get("champion_invalid", 0))
        st.champion_last_session = str(raw.get("champion_last_session", ""))
        st.history = list(raw.get("history", []))[-50:]
        st.cycle_events = list(raw.get("cycle_events", []))[-CYCLE_EVENTS_MAX:]
        st.copy_strikes = {str(h): [str(d) for d in ds] for h, ds in raw.get("copy_strikes", {}).items()}
        st.copy_paused_until = {str(h): int(b) for h, b in raw.get("copy_paused_until", {}).items()}
        st.derived = {str(d): [str(x) for x in ps] for d, ps in raw.get("derived", {}).items()}
        st.derived_hk = {str(d): {str(p): [str(x) for x in hs] for p, hs in per.items()}
                         for d, per in raw.get("derived_hk", {}).items() if isinstance(per, dict)}
        st.held_scores = {str(d): float(v) for d, v in raw.get("held_scores", {}).items()}
        st.low_scores = {str(d): float(v) for d, v in raw.get("low_scores", {}).items()}
        st.parent_sessions = {str(k): v for k, v in raw.get("parent_sessions", {}).items() if _parent_entry_ok(v)}
        return st


def _parent_entry_ok(v: Any) -> bool:
    """A stored same-session parent run is used only when well formed; anything else is dropped (the stored line applies)."""
    return (isinstance(v, dict) and isinstance(v.get("child"), str) and isinstance(v.get("parent"), str)
            and isinstance(v.get("ok"), bool) and isinstance(v.get("score"), (int, float)) and not isinstance(v.get("score"), bool)
            and math.isfinite(v["score"]))


def effective_block(block: int) -> int:
    """The block a commitment counts at: its own, but a commitment made before the engine track opened counts as made at
    the opening block (competition_split.ENGINE_PRE_OPEN_AS_START / ENGINE_TRACK_START_BLOCK). 0 (unknown) stays 0."""
    start = _cs.ENGINE_TRACK_START_BLOCK
    if not _cs.ENGINE_PRE_OPEN_AS_START or start is None or block <= 0:
        return block
    return max(block, start)


_start_hash: Optional[str] = None


def set_start_block_hash(value: Optional[str]) -> None:
    """The hash of ENGINE_TRACK_START_BLOCK as read from the chain (hex, 0x optional); None = not known yet."""
    global _start_hash
    v = str(value).lower().removeprefix("0x") if value else ""
    _start_hash = v or None


def start_block_hash() -> Optional[str]:
    return _start_hash


# The opening block's hash is read only this many blocks after it, so a reorg of the chain head cannot give two
# validators different hashes (the first successful read is kept for the whole process).
START_HASH_CONFIRM_BLOCKS = 12


async def sync_start_block_hash(chain: Any, head_block: int, timeout: float) -> None:
    """Fill the opening block's hash from ``chain.block_hash`` once the block is START_HASH_CONFIRM_BLOCKS deep; until
    then the tie order falls back to uid. A chain without ``block_hash`` or a failed read leaves it unset. Used by the
    validator and by the competition worker (remote mode), which computes first_seen and the champion."""
    import asyncio
    import logging

    start = _cs.ENGINE_TRACK_START_BLOCK
    reader = getattr(chain, "block_hash", None)
    if (not _cs.ENGINE_PRE_OPEN_AS_START or start is None or reader is None or _start_hash
            or head_block < start + START_HASH_CONFIRM_BLOCKS):
        return
    log = logging.getLogger(__name__)
    try:
        value = await asyncio.wait_for(asyncio.to_thread(reader, start), timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("engine opening block hash not readable: %r", exc)
        return
    if isinstance(value, str) and value:
        set_start_block_hash(value)
        log.info("engine opening block %d hash %s", start, value)


def tie_key(block: int, hotkey: str, digest: str) -> str:
    """Order among commitments that count at the opening block: the hash above; "" for every other commitment, and for
    all of them while the opening block hash is not known (then the uid decides, as before)."""
    start, bh = _cs.ENGINE_TRACK_START_BLOCK, _start_hash
    if not _cs.ENGINE_PRE_OPEN_AS_START or start is None or not bh or block <= 0 or block > start:
        return ""
    return hashlib.sha256(f"reax-eng-tie/1|{bh}|{hotkey}|{digest}".encode()).hexdigest()


def commit_order(s: "Submission", unknown_block: int = 0) -> tuple[int, str, int]:
    """Sort key of a submission: (effective commitment block, opening tie hash, uid). ``unknown_block`` is used when the
    chain gave no commitment block."""
    return (effective_block(s.block or unknown_block), tie_key(s.block, s.hotkey, s.digest), s.uid)


def seen_order(state: TrackState, s: "Submission") -> tuple[int, str, int]:
    """Sort key of an observed submission from its persisted first sighting (``state.first_seen``): (effective block, opening
    tie hash, uid). Unlike ``commit_order`` it does not move when the hotkey rewrites its commitment later (e.g. the
    inference element of a dual-role hotkey; docs/DESIGN-MULTIROLE.md section 2)."""
    first = state.first_seen[s.digest][0]
    return (effective_block(first), tie_key(first, s.hotkey, s.digest), s.uid)


def observe(state: TrackState, subs: Sequence[Submission], block: int,
            first_committer_only: bool = True) -> list[Submission]:
    """Record first sightings and return the *eligible* submissions.

    Order = the on-chain commitment block when known, else the block this validator first saw it.
    ``first_committer_only`` (runners without image-to-hotkey binding): a digest committed by a second hotkey is
    ignored for that hotkey. With binding (the image names its hotkey) every committer is eligible here and the
    runner rejects the ones the image is not bound to (``not_owner``), so nobody can squat a published digest.
    """
    for s in sorted(subs, key=lambda s: commit_order(s, block)):
        seen = state.first_seen.get(s.digest)
        when = effective_block(s.block or block)
        if seen is None or when < seen[0]:
            state.first_seen[s.digest] = (when, s.hotkey)
    eligible = []
    for s in subs:
        if f"{s.digest}:{s.hotkey}" in state.not_owner:
            continue
        if first_committer_only and state.first_seen[s.digest][1] != s.hotkey:
            continue
        eligible.append(s)
    return eligible


def pending(state: TrackState, eligible: Sequence[Submission], limit: int) -> list[Submission]:
    """Not yet evaluated digests, oldest commitment first, at most ``limit``. ``state.attempts`` counts inconclusive
    (infrastructure) tries only, so a retry keeps its commitment-order place; the cap is ``max_infra_attempts``, after which
    the submission is recorded as failed and leaves this list."""
    todo = [s for s in eligible if s.digest not in state.results]
    # a digest that already failed twice for an infrastructure reason (a slow or oversized image is indistinguishable from
    # our own trouble) goes behind the untried ones, so a few such entries cannot hold the head of the queue for
    # max_infra_attempts cycles; the first retry keeps its commitment-order place
    todo.sort(key=lambda s: (state.attempts.get(s.digest, 0) >= 2, *seen_order(state, s)))
    return todo[:limit]


def track_thresholds(params: TrackParams) -> dict[str, Any]:
    """The frozen threshold fields (engthresholds.THRESHOLDS) as one track uses them."""
    return {k: getattr(params, k) for k in _T}


def valid_min(params: TrackParams) -> float:
    """The lowest score that counts as valid on a track (the gain threshold over the reference)."""
    return 1.0 + max(params.min_gain, required_engine_margin(params.reference_noise_std))


def _valid(r: Optional[EvalResult], params: TrackParams) -> bool:
    return r is not None and r.ok and r.score >= valid_min(params)


def _mkey(m: Sequence[Any]) -> str:
    """Identity of one measurement: its session id (EvalResult.session), else the block it was taken at."""
    return str(m[2]) if len(m) > 2 and m[2] else f"block:{m[0]}"


def _msession(m: Sequence[Any]) -> str:
    """The session id of a challenger measurement ([block, score, session]); "" for an older two-field one."""
    return str(m[2]) if len(m) > 2 and m[2] else ""


def _pooled(measurements: Sequence[Sequence[Any]]) -> float:
    """Pooled score of a challenger: the geometric mean of its measurements (ratios, so log scale)."""
    return math.exp(math.fsum(math.log(m[1]) for m in measurements) / len(measurements))


def _new_measurement(c: Mapping[str, Any], r: EvalResult, params: TrackParams) -> bool:
    key = r.session or f"block:{r.block}"
    if any(_mkey(m) == key for m in c["measurements"]):
        return False  # the same measurement seen again (e.g. after a restart) counts once
    last = max((m[0] for m in c["measurements"]), default=None)
    return last is None or r.block - last >= params.confirm_gap_blocks


def co_champion_set(ch: Optional["Champion"]) -> Optional[tuple[str, tuple[tuple[str, str], ...]]]:
    """(track, listed (digest, hotkey) pairs) when ``ch`` is one of the ``competition_split.ENGINE_CO_CHAMPIONS``;
    None otherwise (also with ``ch`` None)."""
    if ch is None:
        return None
    for track, pairs in _cs.ENGINE_CO_CHAMPIONS.items():
        if (ch.digest, ch.hotkey) in pairs:
            return track, pairs
    return None


def _parent_lines(state: TrackState, digest: str) -> list[tuple[float, Optional[str]]]:
    """Per declared parent of ``digest``: (best valid score ever of the parent's family, the digest that holds it); (inf,
    None) for a parent without any valid score. A declared parent counts as its hotkey: the family is every digest that
    hotkey committed (the committer; ``state.derived_hk[digest][parent]`` holds the declared uid's hotkey only where the
    committer was unknown, per child)."""
    out = []
    for p in state.derived.get(digest, ()):
        hks = set(state.derived_hk.get(digest, {}).get(p, ()))
        first = state.first_seen.get(p)
        if first is not None:
            hks.add(first[1])
        if p in state.owners:
            hks.add(state.owners[p])
        fam = {p} | {d for d, (_, h) in state.first_seen.items() if h in hks} | {d for d, h in state.owners.items() if h in hks}
        best: dict[str, float] = {}
        for d in fam:
            xs = ([state.held_scores[d]] if d in state.held_scores else []) + (
                [state.results[d].score] if d in state.results and state.results[d].ok else [])
            if xs:
                best[d] = max(xs)
        if not best:
            out.append((math.inf, None))
        else:
            top = max(best, key=lambda d: (best[d], d))
            out.append((best[top], top))
    return out


def parent_floor(state: TrackState, digest: str, floor_min: float = 0.0) -> float:
    """The lowest valid score known for ``digest`` (``low_scores``, recorded for valid results only; a state written before
    that field existed: the lower of the held best and the latest VALID result, i.e. ok and >= ``floor_min`` (the track's
    ``valid_min``). That is never below the true lowest valid score, so the floor is never too low)."""
    xs = [state.low_scores[digest]] if digest in state.low_scores else []
    xs += [state.held_scores[digest]] if digest in state.held_scores else []
    r = state.results.get(digest)
    xs += [r.score] if r is not None and r.ok and r.score >= floor_min else []
    return min(xs) if xs else 0.0


def binding_parent(state: TrackState, digest: str) -> Optional[str]:
    """The parent image that sets the stored parent line of ``digest`` (the best-scoring digest of the highest parent
    family); None without parents or while a parent has no valid score (the entry is held anyway)."""
    lines = _parent_lines(state, digest)
    if not lines or any(d is None for _, d in lines):
        return None
    return max(lines, key=lambda x: (x[0], x[1]))[1]


def parent_bar(state: TrackState, digest: str, margin: float, session: str = "", floor_min: float = 0.0) -> float:
    """Copy line: the score a derived entry must reach, ``(1 + margin) x`` the best valid score among the digests of its
    declared parents; 0 for an entry without parents. A declared parent counts as its hotkey: the best valid score ever held
    over ALL digests that hotkey committed (``_parent_lines``), so naming a weak or failed digest of the parent does not lower
    the bar. A declared parent with no valid score at all returns ``inf``: the entry is held until the parent is measured,
    never crowned on a missing bar.

    ``session``: the child's session id. When the binding parent image was measured right after that session
    (``state.parent_sessions[session]``, a valid run of the same child), that parent's line for this session is
    max(its score in this session, the image's lowest valid score ``parent_floor``) instead of its best ever. Without such a
    run (none scheduled, parent not pullable, failed, or the binding image changed since) the stored line applies.
    ``floor_min``: the track's ``valid_min``, for the floor of a state without ``low_scores``."""
    lines = _parent_lines(state, digest)
    if not lines:
        return 0.0
    run = state.parent_sessions.get(session) if session else None
    if run is not None and not (run.get("child") == digest and run.get("ok") is True and _parent_entry_ok(run)):
        run = None
    bars = []
    for best, top in lines:
        if top is None:
            return math.inf
        if run is not None and run["parent"] == top:
            best = max(float(run["score"]), parent_floor(state, top, floor_min))
        bars.append(best)
    return max(bars) * (1.0 + margin)


def champion_bar(state: TrackState, params: TrackParams) -> float:
    """The score a challenger must reach against the sitting champion (its held score, or the opening co-champion bar,
    x (1 + margin)); 0 without a champion. As in ``update_champion``, before the champion's latest session is applied."""
    ch = state.champion
    if ch is None:
        return 0.0
    held = ch.score
    co = co_champion_set(ch)
    if co is not None:
        held = max(held, _cs.ENGINE_CO_CHAMPION_BAR.get(co[0], 0.0))
    return held * (1.0 + max(params.margin, required_engine_margin(params.reference_noise_std)))


def parent_reference_due(state: TrackState, digest: str, params: TrackParams) -> Optional[str]:
    """The parent digest to measure right after ``digest``'s latest session, or None. Due when the entry is derived, its
    latest result is valid and clears the champion bar less one margin of slack (score >= champion_bar / (1 + margin): the
    champion's same-cycle re-measure may still lower the bar; crowning itself uses the real bars), and it is below, or
    within the margin of, its stored parent line (score < stored line x (1 + margin)). Not for the sitting champion, nor
    while a parent has no valid score."""
    r = state.results.get(digest)
    if not _valid(r, params) or not state.derived.get(digest):
        return None
    if state.champion is not None and state.champion.digest == digest:
        return None
    margin = max(params.margin, required_engine_margin(params.reference_noise_std))
    stored = parent_bar(state, digest, margin)
    if not (0.0 < stored < math.inf) or r.score < champion_bar(state, params) / (1.0 + margin) or r.score >= stored * (1.0 + margin):
        return None
    top = binding_parent(state, digest)
    return top if top is not None and top != digest else None


def prune_parent_sessions(state: TrackState) -> None:
    """Keep the same-session parent runs that can still matter: those of the latest result of each digest and of the
    pending challenger's measurements (newest PARENT_SESSIONS_MAX)."""
    keep = {r.session for r in state.results.values() if r.session}
    c = state.challenger
    if isinstance(c, dict):
        keep |= {str(m[2]) for m in c.get("measurements", []) if isinstance(m, (list, tuple)) and len(m) > 2 and m[2]}
    items = [(k, v) for k, v in state.parent_sessions.items() if k in keep]
    state.parent_sessions = dict(items[-PARENT_SESSIONS_MAX:])


def blocked_noncompliant(state, digest, block):
    reason = "digest absent from Sandy weights_ok"
    logging.getLogger(__name__).info("engine contender %s blocked by weights_ok: %s", digest[:12], reason)
    state.history.append({"event": "blocked_noncompliant", "digest12": digest[:12], "reason": reason, "block": block})


def title_eligible(state: TrackState, digest: str, now: datetime) -> bool:
    """May TAKE the title; never revokes a sitting title or a historical award.

    Sandy supplies weights_ok from explicit compliance reviews, independently of
    copy-line derived markers. Missing compliance fails closed after conversion.
    """
    return now < _cs.ENGINE_QUANT_CONVERSION_END_UTC or digest in state.weights_ok


def update_champion(state: TrackState, eligible: Sequence[Submission], params: TrackParams, block: int,
                    copyline: bool = False, now: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)) -> Optional[Champion]:
    """Apply the champion rule over the currently committed, eligible submissions.

    * A challenger needs ``confirmations`` separate sessions, each beating the champion's held score by ``margin``;
      it is crowned at the lowest of them (winner's curse). The champion's score is held at the minimum of its
      measurements, so one lucky re-measurement cannot raise the bar.
    * The pending challenger is protected: a different submission displaces it only by beating its pooled score by
      the same ``margin`` (and the champion bar). A near-copy that merely scores a little higher, or one that misses
      the champion bar, leaves its confirmations untouched.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if _cs.ENGINE_CROWN_SLOTS_UTC:
        state.next_slot_utc = next_engine_slot(now, state)
    live = {(s.digest, s.hotkey): s for s in eligible}  # an identical digest committed by a second hotkey is another entry
    ch = state.champion
    if ch is not None:
        sub = live.get((ch.digest, ch.hotkey))
        r = state.results.get(ch.digest)
        withdrawn = sub is None
        if r is not None:
            key = r.session or f"block:{r.block}"
            if key != state.champion_last_session:
                state.champion_last_session = key
                state.champion_invalid = 0 if _valid(r, params) else state.champion_invalid + 1
                if state.champion_invalid == 1:
                    state.history.append({"block": block, "event": "champion_recheck", "digest": ch.digest})
        if withdrawn or state.champion_invalid >= 2:
            state.history.append({"block": block, "event": "vacated", "digest": ch.digest, "hotkey": ch.hotkey})
            ch = None
            state.champion_invalid = 0
            state.champion_last_session = ""
        elif _valid(r, params):
            ch.score = min(ch.score, r.score)
            ch.uid = sub.uid
    held = ch.score if ch is not None else 0.0
    co = co_champion_set(ch)
    if co is not None:  # opening co-champions: the bar is the higher co-champion score (ENGINE_CO_CHAMPION_BAR)
        held = max(held, _cs.ENGINE_CO_CHAMPION_BAR.get(co[0], 0.0))
    bar = held * (1.0 + max(params.margin, required_engine_margin(params.reference_noise_std))) if ch is not None else 0.0
    for sub in eligible:
        if (ch is None or sub.digest != ch.digest) and not title_eligible(state, sub.digest, now) and (
                _valid(state.results.get(sub.digest), params) or
                (state.challenger or {}).get("digest") == sub.digest):
            blocked_noncompliant(state, sub.digest, block)
    contenders = sorted(
        (s for s in eligible if _valid(state.results.get(s.digest), params) and title_eligible(state, s.digest, now)),
        key=lambda s: (-state.results[s.digest].score, *seen_order(state, s)),
    )
    margin = max(params.margin, required_engine_margin(params.reference_noise_std))
    def bar_of(digest: str, session: Optional[str] = None) -> float:
        # copy line: an entry whose review record was not read into ``derived`` is not crownable (fail closed). The parent
        # line is the one of the given session (default: the digest's latest result), see parent_bar.
        if not copyline:
            return bar
        if session is None:
            session = state.results[digest].session if digest in state.results else ""
        return math.inf if digest not in state.derived else max(bar, parent_bar(state, digest, margin, session, valid_min(params)))

    pool = [s for s in contenders if (ch is None or s.digest != ch.digest) and (ch is None or state.results[s.digest].score >= bar)
            and (not copyline or state.results[s.digest].score >= bar_of(s.digest))]
    best = None
    if pool:
        # Within-margin rule: candidates not separated from the top score by the displacement margin are a tie on merit
        # (measurement noise); the earliest commitment (persisted first sighting) wins, so a later copy does not take the
        # title from the original by a lucky run.
        top = state.results[pool[0].digest].score
        best = min((s for s in pool if state.results[s.digest].score * (1.0 + margin) > top), key=lambda s: seen_order(state, s))
    c = state.challenger
    cand: Optional[Submission] = None
    if c is not None:
        cand = next((s for s in contenders if s.digest == c["digest"] and s.hotkey == c["hotkey"]
                     and (ch is None or s.digest != ch.digest)), None)
        if cand is not None:
            r = state.results[cand.digest]
            if _new_measurement(c, r, params):
                if r.score >= bar_of(cand.digest):
                    c["measurements"].append([r.block, r.score, r.session])
                else:
                    cand = None  # failed its confirmation against the current bar
            if cand is not None and any(m[1] < bar_of(cand.digest, _msession(m)) for m in c["measurements"]):
                cand = None  # the bar rose (new champion score) above what it was measured at (parent line: per session)
    if best is not None and (cand is None or best.digest != cand.digest):
        rb = state.results[best.digest]
        pooled = _pooled(c["measurements"]) if cand is not None else 0.0
        beats_pending = (cand is None or rb.score >= pooled * (1.0 + margin)
                         or (rb.score * (1.0 + margin) > pooled and seen_order(state, best) < seen_order(state, cand)))
        if rb.score >= bar_of(best.digest) and beats_pending:
            cand = best
            c = {"digest": best.digest, "hotkey": best.hotkey, "measurements": [[rb.block, rb.score, rb.session]]}
    if cand is None:
        state.challenger = None
    else:
        if _cs.ENGINE_CROWN_SLOTS_UTC:
            c["uid"] = cand.uid
        state.challenger = c
        if len(c["measurements"]) >= params.confirmations and _cs.ENGINE_CROWN_SLOTS_UTC:
            if not c.get("confirmed_at"):
                c["confirmed_at"] = now.astimezone(timezone.utc).isoformat()
                c["slot_utc"] = next_engine_slot(now, state)
        elif len(c["measurements"]) >= params.confirmations:
            # Winner's curse: the title is held at the lowest of the confirming measurements.
            score = min(m[1] for m in c["measurements"])
            state.champion_invalid = 0
            state.champion_last_session = state.results[cand.digest].session or f"block:{state.results[cand.digest].block}"
            state.history.append({"block": block, "event": "crowned", "digest": cand.digest, "hotkey": cand.hotkey,
                                  "score": round(score, 4), "pooled": round(_pooled(c["measurements"]), 4),
                                  "sessions": len(c["measurements"])})
            ch = Champion(cand.digest, cand.hotkey, cand.uid, score, block)
            state.challenger = None
    state.champion = ch
    return ch


def confirmation_due(state: TrackState, eligible: Sequence[Submission], params: TrackParams) -> bool:
    """A pending challenger that still needs another session (it may run right after the first, same cycle)."""
    c = state.challenger
    return bool(state.champion is not None and state.champion_invalid) or (c is not None and len(c["measurements"]) < params.confirmations
            and any(s.digest == c["digest"] and s.hotkey == c["hotkey"] for s in eligible))


def needs_remeasurement(state: TrackState, eligible: Sequence[Submission]) -> list[Submission]:
    """The pending challenger (for its confirmation) and, when it is challenged, the champion."""
    out = []
    c = state.challenger
    if c is not None:
        out += [s for s in eligible if s.digest == c["digest"] and s.hotkey == c["hotkey"]]
    ch = state.champion
    if ch is not None and (out or state.champion_invalid):
        out += [s for s in eligible if s.digest == ch.digest and s.hotkey == ch.hotkey]
    return out


def champion_fraction(reward: RewardParams, age_blocks: int) -> float:
    """The champion's part of the engine share ``age_blocks`` after its crowning (linear decay to the floor)."""
    age = max(0, age_blocks)
    return reward.champion_start - (reward.champion_start - reward.champion_floor) * min(1.0, age / reward.decay_blocks)


def copy_digests(state: TrackState) -> frozenset[str]:
    """Digests struck as copies (``copy_strikes``); the validator passes them as ``exclude`` to the payout helpers only
    while the copy line is active."""
    return frozenset(d for ds in state.copy_strikes.values() for d in ds)


def former_champions(state: TrackState, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Champions that were crowned and later replaced, newest crowning first, from the ``history`` events
    (``crowned`` / ``vacated`` / ``context_changed``): ``{"digest", "hotkey", "block", "score"}``. The current champion, a
    vacated one and everything before a context change are not in the list. History keeps the last 50 events. ``exclude``:
    digests left out (copies, copy line)."""
    out: list[dict[str, Any]] = []
    for e in state.history:
        kind = e.get("event")
        if kind == "context_changed":
            out = []
        elif kind == "crowned":
            out = [f for f in out if (f["digest"], f["hotkey"]) != (e.get("digest"), e.get("hotkey"))]
            out.append({"digest": e.get("digest"), "hotkey": e.get("hotkey"), "block": e.get("block"),
                        "score": e.get("score")})
        elif kind == "vacated":
            out = [f for f in out if (f["digest"], f["hotkey"]) != (e.get("digest"), e.get("hotkey"))]
    ch = state.champion
    return [f for f in reversed(out) if f["digest"] not in exclude
            and (ch is None or (f["digest"], f["hotkey"]) != (ch.digest, ch.hotkey))]


def dethroned_blocks(state: TrackState) -> dict[tuple[str, str], Any]:
    """(digest, hotkey) -> the block at which that champion lost the title to the next ``crowned`` event (its latest
    reign), from ``history``. A champion that is crowned again loses the entry until it is dethroned again; a vacated
    title and everything before a context change dethrone nobody. Used for the former-champion slot time
    (competition_split.ENGINE_FORMER_SLOT_BLOCKS)."""
    out: dict[tuple[str, str], Any] = {}
    holder: Optional[tuple[str, str]] = None
    for e in state.history:
        kind = e.get("event")
        if kind == "context_changed":
            out, holder = {}, None
        elif kind == "crowned":
            key = (e.get("digest"), e.get("hotkey"))
            if holder is not None and holder != key:
                out[holder] = e.get("block")
            out.pop(key, None)
            holder = key
        elif kind == "vacated":
            holder = None
    return out


def waiting_uids(state: TrackState, eligible: Sequence[Submission], block: int,
                 coldkeys: Mapping[int, str], registered: Optional[Mapping[int, int]] = None) -> list[int]:
    """Engine keep-alive candidates (competition_split.ENGINE_KA_*): eligible submissions that wait for a decision,
    at most one per coldkey (a missing coldkey counts as its own), oldest first, at most ENGINE_KA_MAX_ENTRIES. Pure.

    Waiting = no result yet and the image is bound to the committing hotkey (``owners``) or, not pulled yet, the hotkey
    is the digest's first committer and no other hotkey is bound to it; or the pending challenger (its first session
    beat the bar, the confirming one is due). The ENGINE_KA_MAX_HOURS window counts from the first sighting (effective
    commitment block of the hotkey's first digest, so a new digest does not restart it; the challenger: its first session)
    or, if later and ``registered`` gives the uid's registration
    block, from the end of its immunity minus PROVISIONAL_KA_LEAD_BLOCKS (an immune hotkey is not pruned)."""
    max_blocks = _cs.ENGINE_KA_MAX_HOURS * 3600.0 / ENGINE_BLOCK_SECONDS
    c = state.challenger
    cands = []
    for s in eligible:
        first = state.first_seen.get(s.digest)
        if first is None:
            continue
        pending = (c is not None and c.get("digest") == s.digest and c.get("hotkey") == s.hotkey and c.get("measurements")
                   and (state.champion is None or state.champion.digest != s.digest))
        if pending:
            try:
                since = max(effective_block(first[0]), int(c["measurements"][0][0]))
            except (TypeError, ValueError, IndexError, KeyError):
                continue    # malformed challenger record: fail closed
        elif s.digest in state.results:
            continue
        else:
            owner = state.owners.get(s.digest)
            if not (owner == s.hotkey or (owner is None and first[1] == s.hotkey)):
                continue
            # one window per hotkey: its earliest first sighting on any digest (``first_seen`` is never pruned), so
            # committing a fresh digest does not restart it
            since = min([effective_block(b) for b, hk in state.first_seen.values() if hk == s.hotkey] or [effective_block(first[0])])
        reg = (registered or {}).get(s.uid, 0)
        if isinstance(reg, int) and reg > 0:
            since = max(since, reg + _cs.CHAIN_IMMUNITY_BLOCKS - _cs.PROVISIONAL_KA_LEAD_BLOCKS)
        if since <= 0 or since > block + 5 or block - since > max_blocks:
            continue
        cands.append((seen_order(state, s), s))
    out: list[int] = []
    seen: set[str] = set()
    for _, s in sorted(cands, key=lambda c: c[0]):
        ck = coldkeys.get(s.uid) or f"uid:{s.uid}"
        if ck in seen or s.uid in out:
            continue
        seen.add(ck)
        out.append(s.uid)
    return out[: _cs.ENGINE_KA_MAX_ENTRIES]


def former_ranks(state: TrackState, track: str, block: int,
                 exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """The former champions of ``track`` as ranks below the title holder, most recently crowned first (1.2.0rc10.post1,
    Florian 21 Oct 2026 05:30Z "B"): ``{"pairs", "block", "co", "share"}``. Pure.

    The opening co-champion set (ENGINE_CO_CHAMPIONS) is ONE former champion, ranked by its newest member's crowning; it
    is left out while it holds the title. ``share`` is ENGINE_RANK_SPLIT[1] / [2] for the first two entries while their
    slot runs (ENGINE_FORMER_SLOT_BLOCKS from the block at which they were dethroned, ``dethroned_blocks``), else 0.
    Entries with a malformed crowning block are left out, and the ``exclude`` digests (copies, copy line)."""
    co_seen = co_champion_set(state.champion) is not None
    deth = dethroned_blocks(state)
    out: list[dict[str, Any]] = []
    for f in former_champions(state, exclude):
        if not isinstance(f["block"], int):
            continue
        co = co_champion_set(Champion(f["digest"], f["hotkey"], -1, 0.0, f["block"]))
        if co is not None and (co_seen or co[0] != track):
            continue
        co_seen = co_seen or co is not None
        d = deth.get((f["digest"], f["hotkey"]))
        rank = len(out) + 1
        held = (rank < len(_cs.ENGINE_RANK_SPLIT) and isinstance(d, int)
                and 0 <= block - d < _cs.ENGINE_FORMER_SLOT_BLOCKS)
        out.append({"pairs": co[1] if co is not None else ((f["digest"], f["hotkey"]),), "block": f["block"],
                    "co": co is not None, "share": _cs.ENGINE_RANK_SPLIT[rank] if held else 0.0})
    return out


def protected_uids(state: TrackState, eligible: Sequence[Submission], params: TrackParams, block: int,
                   champion_approved: bool, track: str = "s1-fast", exclude: frozenset[str] = frozenset()) -> list[int]:
    """Engine keep-alive ahead of the waiting entries (competition_split.ENGINE_KA_PROTECTED_MAX). Pure.

    (a) The live current champion while ``champion_approved`` is False (it awaits the owner's award decision).
    (b) Every former champion that holds a slot (``former_ranks``: rank 2 / 3, within ENGINE_FORMER_SLOT_BLOCKS of its
    own dethroning), in rank order: each listed pair (all members of the opening co-champion set) that is still committed
    by its hotkey and has a valid last result. At most ENGINE_KA_PROTECTED_MAX uids. A uid that is paid anything else
    (its slot share, the champion share) gets no keep-alive: combine_weights leaves it out."""
    ch = state.champion
    if ch is None:
        return []
    by_pair = {(s.digest, s.hotkey): s for s in eligible}
    out: list[int] = []
    live = by_pair.get((ch.digest, ch.hotkey))
    if live is not None and not champion_approved:
        out.append(live.uid)
    for g in former_ranks(state, track, block, exclude):
        if g["share"] <= 0:
            continue
        for digest, hotkey in g["pairs"]:
            sub = by_pair.get((digest, hotkey))
            if sub is not None and sub.uid not in out and _valid(state.results.get(digest), params):
                out.append(sub.uid)
    return out[: _cs.ENGINE_KA_PROTECTED_MAX]


def competition_weights(state: TrackState, eligible: Sequence[Submission], params: TrackParams,
                        reward: RewardParams, block: int) -> dict[int, float]:
    """Split of the competition share inside the track (sums to 1, or empty when there is no champion)."""
    ch = state.champion
    if ch is None:
        return {}
    frac = champion_fraction(reward, block - ch.crowned_block)
    others = sorted(
        (s for s in eligible if s.digest != ch.digest and _valid(state.results.get(s.digest), params)),
        key=lambda s: (-state.results[s.digest].score, *seen_order(state, s)),
    )[: reward.runner_ups]
    weights = {ch.uid: frac}
    if not others:
        return weights  # the decayed part (1 - frac) stays with the serving pools
    ranks = [0.5 ** i for i in range(len(others))]
    total = math.fsum(ranks)
    for s, r in zip(others, ranks):
        weights[s.uid] = weights.get(s.uid, 0.0) + (1.0 - frac) * r / total
    return weights


def merge_weights(base: Mapping[int, float], comp: Mapping[int, float], share: float,
                  owner_uid: Optional[int] = None, burn_fraction: float = 0.0) -> dict[int, float]:
    """Give the competition ``share`` x (its total) of the *miner* part of ``base``.

    The owner's burn fraction is kept as it is; everything else in ``base`` (serving miners, and the owner's
    fallback for pools without an eligible miner) is scaled down by the same factor. With no champion, or an empty
    base (nothing is set at all), ``base`` is returned unchanged.
    """
    total = math.fsum(comp.values())
    if not comp or share <= 0 or total <= 0 or not base:
        return dict(base)
    burn = burn_fraction if owner_uid is not None and owner_uid in base else 0.0
    miner_part = 1.0 - burn
    k = 1.0 - share * total
    out = {}
    for uid, w in base.items():
        out[uid] = burn + (w - burn) * k if uid == owner_uid else w * k
    for uid, w in comp.items():
        out[uid] = out.get(uid, 0.0) + share * miner_part * w
    norm = math.fsum(out.values())
    return {uid: w / norm for uid, w in sorted(out.items())}


def next_engine_slot(now: datetime, state: Optional[TrackState] = None) -> Optional[str]:
    """First configured UTC slot at or after now; None after the published calendar ends."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    used = {e.get("slot_utc") for e in state.slot_results} if state else set()
    return next((s.isoformat() for s in _cs.ENGINE_CROWN_SLOTS_UTC if s >= now and s.isoformat() not in used), None)


def pending_crown(state: TrackState) -> Optional[dict[str, Any]]:
    c = state.challenger
    if not c or not c.get("confirmed_at") or not c.get("slot_utc"):
        return None
    return {"uid": c.get("uid"), "digest12": c["digest"][:12], "digest": c["digest"],
            "pooled_score": _pooled(c["measurements"]), "slot_utc": c["slot_utc"],
            "confirmed_at": c["confirmed_at"]}


def finalize_engine_slot(state: TrackState, eligible: Sequence[Submission], params: TrackParams,
                         now: datetime, block: int) -> Optional[Champion]:
    """Crown a due, still-valid confirmed challenger, without measurements or network calls.

    Mutates state on a crown or records a blocked_noncompliant history event; returns the new champion or None.
    A missed slot keeps its scheduled time in the receipt; decay starts at the supplied finalisation block.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    c = state.challenger
    if not _cs.ENGINE_CROWN_SLOTS_UTC or not c or not c.get("confirmed_at") or not c.get("slot_utc"):
        return None
    try:
        slot = datetime.fromisoformat(c["slot_utc"].replace("Z", "+00:00"))
        confirmed = datetime.fromisoformat(c["confirmed_at"].replace("Z", "+00:00"))
        if slot.tzinfo is None or confirmed.tzinfo is None or slot not in _cs.ENGINE_CROWN_SLOTS_UTC:
            return None
        if now < slot or confirmed > slot or next_engine_slot(confirmed, state) != slot.isoformat():
            return None
        if any(e.get("slot_utc") == slot.isoformat() for e in state.slot_results):
            return None
        ms = c["measurements"]
        if len(ms) < params.confirmations or len({_mkey(m) for m in ms}) != len(ms):
            return None
        if any(not math.isfinite(m[1]) or m[1] < valid_min(params) for m in ms):
            return None
        if any(b[0] - a[0] < params.confirm_gap_blocks for a, b in zip(ms, ms[1:])):
            return None
        sub = next((s for s in eligible if (s.digest, s.hotkey) == (c["digest"], c["hotkey"])), None)
        if sub is None or not _valid(state.results.get(sub.digest), params):
            return None
        if not title_eligible(state, sub.digest, now):
            blocked_noncompliant(state, sub.digest, block)
            return None
        if f"{sub.digest}:{sub.hotkey}" in state.not_owner or state.owners.get(sub.digest, sub.hotkey) != sub.hotkey:
            return None
        if state.champion and state.champion.digest == sub.digest:
            return None
        margin = max(params.margin, required_engine_margin(params.reference_noise_std))
        bar = champion_bar(state, params)
        cl = copyline_active(now)
        if cl and (sub.digest not in state.derived or sub.digest in copy_digests(state)
                   or state.copy_paused_until.get(sub.hotkey, 0) > block):
            return None
        def bar_for(session):
            return max(bar, parent_bar(state, sub.digest, margin, session, valid_min(params))) if cl else bar
        r = state.results[sub.digest]
        if r.score < bar_for(r.session) or any(m[1] < bar_for(_msession(m)) for m in ms):
            return None
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return None
    score = min(m[1] for m in ms)
    crown_block = block
    ch = Champion(sub.digest, sub.hotkey, sub.uid, score, crown_block)
    event = {"block": crown_block, "event": "crowned", "digest": sub.digest, "hotkey": sub.hotkey,
             "score": round(score, 4), "pooled": round(_pooled(ms), 4), "sessions": len(ms),
             "slot_utc": slot.isoformat()}
    state.history.append(event)
    state.slot_results.append(dict(event))
    state.champion = ch
    state.challenger = None
    state.champion_invalid = 0
    state.champion_last_session = r.session or f"block:{r.block}"
    state.next_slot_utc = next_engine_slot(now, state)
    return ch
