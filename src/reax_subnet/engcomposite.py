"""Composite engine track ``dm-all`` (engine competition v2, unreleased; docs/ENGINE-COMPETITION-V2.md).

One engine image is measured on every GPU model we serve on decisionmodels.io, each against our pinned reference engine
for that model, in its own fresh session (the v1 measurement of engrunner.evaluate, unchanged, once per model). The
per-model speed ratios are combined into one score:

* every model the image declares (label ``ai.reax.engine.models``; no label = all models) must pass all v1 gates for
  that model (distance, top choice, anti-cache probes, determinism, latency, reference drift) and must not be slower
  than the reference beyond that model's noise (the no-regression floor); a GPU memory budget per model keeps the
  win a real production win (a faster engine that needs twice the memory serves fewer slots per card);
* a model the image does not declare counts as exactly 1.0 (we would keep the reference for it) and is not measured;
* composite = weighted geometric mean of the per-model ratios (weights are release constants, sum 1).

The composite is turned into one EvalResult per digest, so the v1 champion rule (engcomp.update_champion: min gain,
margin, second fresh session, held at the lower score, re-measurement when challenged) applies to the composite score
without any change. The composite's noise unit is the weighted combination of the per-model noise units, so the
margin over a champion is ``max(3 %, 3 x composite noise)`` as in v1.

This module holds the pure parts and the session loop; engrunner/engvalidator are unchanged apart from three small
hooks (per-model environment of the runner, the GPU memory reading, the dispatch in the validator).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from .engcomp import EvalResult, TrackParams
from .s1contract import IMAGE_POOLS
from . import s1contract as _contract
from .s1vision import GENERATOR_RULE as IMAGE_ITEMS_RULE

COMPOSITE_PROTOCOL_TRACK = "dm-all"  # the <track> field of a reax-eng/1 commitment (the commitment format is unchanged)
MODELS_LABEL = "ai.reax.engine.models"  # comma-separated slot names, or "all"; absent = all
MODEL_ENV = "REAX_MODEL"  # the engine container learns which model it serves from this variable (and /model)

# Release constants of engine competition v2.0 (the lineup decisionmodels.io serves after its switch, 25 Oct 2026). ONE
# weights table per variant, each sums to 1 (asserted at import); the active one is part of composite_rules_sha256 through
# the slot weights and the carry-over.
#   v20            mammoth 0.37, blink 0.33, reax1 0.10, jeffan 0.10, vision 0.10 (DEFAULT; Florian's pick 26 Oct, card
#                  18618 "Floor"): a 10 % floor for every served model, the remaining 0.70 split between the two main
#                  engines by GPU-cost share since 23 Oct (50.7 : 44.9). mammoth (baselabs/mammoth-1, served as s1-pro)
#                  inherits the share of the s1-pro engine it replaces. ``jeffan`` is Jeffan-31b-1710, ``vision`` is
#                  Blink with one image. (25 Oct placeholder was mammoth .50 / blink .20 / reax1 .20 / jeffan .05 / vision .05.)
#   v20-d3         FALLBACK as data: v20 with slot ``d3`` in mammoth's place (same weight 0.37). Used only if, at the
#                  binding, decisionmodels.io still serves d3-turbo as s1-pro. The lead and the release owner decide
#                  and announce it before the binding; the code never switches by itself.
#   v22            d3 0.45, blink 0.20, vision 0.05, reax1 0.20, pplx 0.05, quyet 0.05 (DEFAULT; served traffic and GPU time
#                  over 7 days to 25 Oct, plus a 5 % floor for every served model so each one matters). ``vision`` is the
#                  Blink engine with one image per item (s1-vision), ``quyet`` is Quyet-1.0-Large (quyet-1-large).
#   v21            d3 0.50, blink 0.25, reax1 0.20, pplx 0.05 (22 Oct; four slots, no vision, no quyet); Plumb is not a
#                  slot, v1 ``s1-fast`` commitments are not carried.
#   reax-high      reax1 0.40, d3 0.35, blink 0.20, pplx 0.05 (REAX 1 stays on top); no Plumb, no carry-over.
#   approved-1021  the 21 Oct approved set: reax1 0.40, d3 0.30, blink 0.15, pplx 0.10, plumb 0.05; a v1 ``s1-fast``
#                  commitment is a Plumb-only entry (no re-commit).
COMPOSITE_WEIGHTS_VARIANT = "v20"
COMPOSITE_WEIGHTS: dict[str, dict[str, float]] = {
    "v20": {"mammoth": 0.37, "blink": 0.33, "reax1": 0.10, "jeffan": 0.10, "vision": 0.10},
    "v20-d3": {"d3": 0.37, "blink": 0.33, "reax1": 0.10, "jeffan": 0.10, "vision": 0.10},
    "v22": {"d3": 0.45, "blink": 0.20, "vision": 0.05, "reax1": 0.20, "pplx": 0.05, "quyet": 0.05},
    "v21": {"d3": 0.50, "blink": 0.25, "reax1": 0.20, "pplx": 0.05},
    "reax-high": {"reax1": 0.40, "d3": 0.35, "blink": 0.20, "pplx": 0.05},
    "approved-1021": {"reax1": 0.40, "d3": 0.30, "blink": 0.15, "pplx": 0.10, "plumb": 0.05},
}
COMPOSITE_CARRIED: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "v20": (),
    "v20-d3": (),
    "v22": (),
    "v21": (),
    "reax-high": (),
    "approved-1021": (("s1-fast", ("plumb",)),),
}
assert set(COMPOSITE_WEIGHTS) == set(COMPOSITE_CARRIED)
for _variant, _weights in COMPOSITE_WEIGHTS.items():
    assert abs(math.fsum(_weights.values()) - 1.0) < 1e-9, f"weights of variant {_variant} must sum to 1"
# Evaluation order = fail-fast order, cheapest first; a slot a variant does not weight is not part of that track.
# vision right after blink (same engine and weights, a broken image path dies early), then the 31B slots (quyet, jeffan:
# one engine, FP8) before the 27B slots; mammoth and d3 last (longest budgets). Re-order after the first measured batch if
# the times say otherwise.
COMPOSITE_EVAL_ORDER = ("plumb", "blink", "vision", "quyet", "reax1", "jeffan", "pplx", "mammoth", "d3")
# Slots that run on the weights directory of another slot (the engine and the model are the same, only the items differ).
SLOT_WEIGHTS_FROM = {"vision": "blink"}
# Model id served by slot ``d3``: the production s1-pro engine switched to d3-turbo (vllm-sr/d3-turbo@cfa0d41d...) on
# 22 Oct. Roll-back of the production canary = set this to "d3" (weights 5ce995d); nothing else changes.
D3_SLOT_MODEL = "d3-turbo"

FLOOR_MIN = 0.02  # a declared model may be at most this much slower than its reference (or 3 x its noise, if larger)
FLOOR_NOISE_MULTIPLIER = 3.0


@dataclass(frozen=True)
class ModelSlot:
    """One production model inside the composite track."""

    name: str  # slot name used in the label, the results and the docs ("d3")
    model: str  # model id the engine serves (REAX_MODEL and the "model" field of every decision)
    weight: float  # share of the composite (release constant; all slots sum to 1)
    params: TrackParams  # the v1 gates, item counts, concurrency and temperature of this model
    memory_budget_mib: int  # largest GPU memory the candidate may add next to the loaded reference (0 = not checked)
    calibrated: bool = False  # True once noise, temperature and gates were measured on the published GPU class
    calibrated_on: str = ""  # the reference image digest the calibration was measured on (release_ready compares it)

    @property
    def floor(self) -> float:
        return 1.0 - max(FLOOR_MIN, FLOOR_NOISE_MULTIPLIER * self.params.reference_noise_std)


@dataclass(frozen=True)
class CompositeTrack:
    name: str
    slots: tuple[ModelSlot, ...]  # also the evaluation order: cheapest first, so a broken image fails fast
    min_gain: float = 0.03
    margin: float = 0.03
    confirmations: int = 2
    confirm_gap_blocks: int = 0
    slack_s: float = 900.0  # pulls and start-up of the first model on top of the per-model budgets
    # Commitments to older tracks read as entries of this one with a fixed declaration (migration from v1: an
    # ``s1-fast`` image is a Plumb-only entry; it keeps its commitment block and needs no new commitment).
    carried: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required: tuple[str, ...] = ()  # slots every image must declare (empty: any non-empty subset)
    max_evals: int = 2  # new submissions per cycle (a full composite evaluation is up to ~2 h of GPU)

    def __post_init__(self) -> None:
        names = [s.name for s in self.slots]
        if not self.slots or len(set(names)) != len(names):
            raise ValueError(f"composite track {self.name}: slot names must be unique and non-empty")
        if any(not (s.weight > 0) for s in self.slots) or abs(math.fsum(s.weight for s in self.slots) - 1.0) > 1e-9:
            raise ValueError(f"composite track {self.name}: slot weights must be positive and sum to 1")
        if any(s.memory_budget_mib < 0 for s in self.slots) or "all" in names:
            raise ValueError(f"composite track {self.name}: invalid slot")
        for old, models in self.carried:
            if old == self.name or not models or not set(models) <= set(names):
                raise ValueError(f"composite track {self.name}: invalid carried track {old!r}")
        if not set(self.required) <= set(names) or self.max_evals < 1:
            raise ValueError(f"composite track {self.name}: invalid required slots or max_evals")

    def carried_declaration(self, track: str) -> Optional[frozenset[str]]:
        """The fixed declaration of a commitment to an older track that this track carries, else None."""
        for old, models in self.carried:
            if old == track:
                return frozenset(models)
        return None

    def slot(self, name: str) -> ModelSlot:
        for s in self.slots:
            if s.name == name:
                return s
        raise KeyError(name)

    def noise(self) -> float:
        """Noise unit of the composite: the per-model noise units combined with the weights (log scale, independent
        sessions), all models declared. A subset has less noise, so this is the conservative release constant."""
        return math.sqrt(math.fsum((s.weight * s.params.reference_noise_std) ** 2 for s in self.slots))

    def budget_s(self, declared: Optional[frozenset[str]] = None) -> float:
        return self.slack_s + math.fsum(s.params.eval_budget_s for s in self.slots
                                        if declared is None or s.name in declared)

    def params(self) -> TrackParams:
        """The TrackParams the champion rule (engcomp.update_champion) sees for the composite score."""
        return TrackParams(self.name, pool=self.name, temperature=1.0, min_gain=self.min_gain, margin=self.margin,
                           reference_noise_std=self.noise(), confirmations=self.confirmations,
                           confirm_gap_blocks=self.confirm_gap_blocks, eval_budget_s=self.budget_s(),
                           max_infra_attempts=max(s.params.max_infra_attempts for s in self.slots))

    def release_ready(self) -> list[str]:
        """Reasons the track may not be opened yet (empty = ready): a slot that is not calibrated (its own marker, see
        ``default_track``), a slot of the release table whose reference image digest is not published yet
        (config.ENGINE_COMPOSITE_REFERENCE_IMAGES) and a calibrated slot whose published digest is not the one it was
        calibrated on (a digest bump clears the calibration)."""
        from .config import ENGINE_COMPOSITE_REFERENCE_IMAGES as published

        out = [f"slot {s.name} is not calibrated" for s in self.slots if not s.calibrated]
        out += [f"slot {s.name} has no published reference image digest" for s in self.slots
                if not published.get(s.name)]  # a slot without a dict key counts as unpublished
        out += [f"slot {s.name} calibrated on a different digest" for s in self.slots
                if s.calibrated and published.get(s.name) and s.calibrated_on != published[s.name]]
        return out


# ------------------------------------------------------------------------------------------------ declaration
def parse_declared(label: Any, track: CompositeTrack) -> frozenset[str]:
    """The models an image declares (its ``ai.reax.engine.models`` label). Absent or "all" = every slot. An unknown
    name, an empty list or a malformed label is a ValueError (the submission fails; a typo must not silently drop a
    model)."""
    every = frozenset(s.name for s in track.slots)
    if label is None:
        return every
    if not isinstance(label, str) or len(label) > 256:
        raise ValueError("models label must be a short string")
    text = label.strip().lower()
    if text == "all":
        return every
    names = [n.strip() for n in text.split(",")]
    if not names or any(not n for n in names):
        raise ValueError("models label lists an empty name")
    unknown = sorted(set(names) - every)
    if unknown:
        raise ValueError(f"models label names unknown models: {', '.join(unknown)[:80]}")
    return frozenset(names)


# ------------------------------------------------------------------------------------------------ scoring
@dataclass(frozen=True)
class CompositeScore:
    ok: bool
    reason: str
    score: float  # weighted geometric mean of the per-model ratios (undeclared = 1.0); 0.0 when not ok
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)


def _part(slot: ModelSlot, declared: bool, r: Optional[EvalResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"model": slot.model, "weight": slot.weight, "declared": declared, "floor": round(slot.floor, 6)}
    if not declared:
        out["score"] = 1.0
    elif r is not None:
        out.update({"score": r.score, "ok": r.ok, "reason": r.reason, "throughput": r.throughput,
                    "ref_throughput": r.ref_throughput, "mean_distance": _finite(r.mean_distance),
                    "top_agreement": r.top_agreement, "memory_mib": r.cand_memory_mib})
    return out


def _finite(x: float) -> Optional[float]:
    return x if math.isfinite(x) else None


def slot_failure(slot: ModelSlot, r: Optional[EvalResult]) -> str:
    """Why a declared model's measurement disqualifies the image ("" = it does not)."""
    if r is None:
        return f"{slot.name}: not measured"
    if not r.ok:
        return f"{slot.name}: {r.reason or 'failed'}"
    if slot.memory_budget_mib and (r.cand_memory_mib or 0) > slot.memory_budget_mib:
        return f"{slot.name}: GPU memory {r.cand_memory_mib} MiB > budget {slot.memory_budget_mib} MiB"
    if not (math.isfinite(r.score) and r.score > 0):
        return f"{slot.name}: no speed score"
    if r.score < slot.floor:
        return f"{slot.name}: {r.score:.3f}x is below the no-regression floor {slot.floor:.3f}x"
    return ""


def composite_score(track: CompositeTrack, declared: frozenset[str],
                    results: Mapping[str, Optional[EvalResult]]) -> CompositeScore:
    """Combine one image's per-model measurements (slot name -> EvalResult). Pure."""
    if not declared:
        return CompositeScore(False, "the image declares no model", 0.0)
    missing = [n for n in track.required if n not in declared]
    if missing:
        return CompositeScore(False, f"the image must declare {', '.join(missing)}", 0.0)
    parts, logs, reason = {}, [], ""
    for slot in track.slots:
        inside = slot.name in declared
        r = results.get(slot.name) if inside else None
        parts[slot.name] = _part(slot, inside, r)
        if not inside:
            continue
        why = slot_failure(slot, r)
        if why and not reason:
            reason = why
        if not why:
            logs.append(slot.weight * math.log(r.score))  # type: ignore[union-attr]
    if reason:
        return CompositeScore(False, reason, 0.0, parts)
    return CompositeScore(True, "", math.exp(math.fsum(logs)), parts)


def to_eval_result(track: CompositeTrack, digest: str, block: int, cs: CompositeScore,
                   results: Mapping[str, Optional[EvalResult]]) -> EvalResult:
    """The composite as one EvalResult of track ``track.name`` (what TrackState.results stores)."""
    measured = [r for r in results.values() if r is not None]
    return EvalResult(digest=digest, track=track.name, ok=cs.ok, reason=cs.reason, score=cs.score,
                      items=sum(r.items for r in measured), block=block, parts=cs.parts)


# ------------------------------------------------------------------------------------------------ frozen rules
def composite_rules(track: CompositeTrack) -> dict[str, Any]:
    """Everything that decides a composite result, as plain data (hashed into the state file like engthresholds)."""
    from .engcomp import track_thresholds

    return {
        "name": track.name, "min_gain": track.min_gain, "margin": track.margin, "confirmations": track.confirmations,
        "confirm_gap_blocks": track.confirm_gap_blocks, "floor_min": FLOOR_MIN,
        "floor_noise_multiplier": FLOOR_NOISE_MULTIPLIER, "carried": [[o, sorted(m)] for o, m in track.carried],
        "slots": [{"name": s.name, "model": s.model, "pool": s.params.pool, "weight": s.weight,
                   "memory_budget_mib": s.memory_budget_mib, "concurrency": s.params.concurrency,
                   "temperature": s.params.temperature, "eval_budget_s": s.params.eval_budget_s,
                   "max_options": _contract.MAX_OPTIONS.get(s.params.pool),
                   **({"image_items": IMAGE_ITEMS_RULE, "max_image_bytes": _contract.MAX_IMAGE_BYTES,
                       "max_image_pixels": _contract.MAX_IMAGE_PIXELS} if s.params.pool in IMAGE_POOLS else {}),
                   **track_thresholds(s.params)} for s in track.slots],
    }


def composite_rules_sha256(track: CompositeTrack) -> str:
    return hashlib.sha256(json.dumps(composite_rules(track), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ item tapes
class SlotTapes:
    """The item tapes of one composite session, one engrunner.ItemTape per model slot. A composite session measures every
    declared model with its own items; the declared parent's reference run (engcomp.parent_reference_due) replays the
    same items per model: ``replay()`` gives a reader over the recorded tapes. Same two members as ItemTape
    (``recorded``, ``replay``), so the validator treats both alike."""

    def __init__(self) -> None:
        self._tapes: dict[str, Any] = {}

    def for_slot(self, name: str) -> Any:
        from .engrunner import ItemTape

        if name not in self._tapes:
            self._tapes[name] = ItemTape()
        return self._tapes[name]

    @property
    def recorded(self) -> int:
        return sum(t.recorded for t in self._tapes.values())

    def replay(self) -> "SlotTapes":
        out = SlotTapes()
        out._tapes = {n: t.replay() for n, t in self._tapes.items()}
        return out


# ------------------------------------------------------------------------------------------------ the session loop
async def evaluate_composite(runner: Any, reference_ref: str, candidate_ref: str, track: CompositeTrack,
                             declared: frozenset[str], block: int = 0, digest: str = "",
                             progress: Optional[dict[str, Any]] = None,
                             references: Optional[Mapping[str, str]] = None) -> EvalResult:
    """Measure one image on every declared model (track order, a fresh v1 session per model, both engines restarted)
    and return the composite EvalResult. Stops at the first model that disqualifies the image. Reference trouble
    (engrunner.ReferenceError_ and subclasses) and infrastructure trouble (InfraError) propagate unchanged, so the
    caller treats the whole composite exactly like one v1 evaluation (void / inconclusive / retried).

    ``progress["items"]``: optional SlotTapes (a per-model item tape: recorded, or replayed for a parent's reference run).
    ``references``: optional per-model reference image (slot name -> ref); default ``reference_ref`` for all, i.e. one
    multi-model reference image. The candidate image is kept between models and given back at the end."""
    import asyncio

    from .engrunner import InfraError, ReferenceError_, ReferenceFailedWithCandidate, evaluate, reference_alone_ok

    results: dict[str, Optional[EvalResult]] = {}
    if any(n not in declared for n in track.required):
        cs = composite_score(track, declared, results)
        return to_eval_result(track, digest or candidate_ref.rsplit(":", 1)[-1], block, cs, results)
    try:
        for slot in track.slots:
            if slot.name not in declared:
                continue
            ref = (references or {}).get(slot.name, reference_ref)
            mine: dict[str, Any] = {}  # this model's own progress: "measuring" means inside ITS timed phase
            tapes = (progress or {}).get("items")
            if isinstance(tapes, SlotTapes):  # record (or replay) this model's items
                mine["items"] = tapes.for_slot(slot.name)
            try:
                r = await asyncio.wait_for(
                    evaluate(runner, ref, candidate_ref, track.name, slot.params, block=block, digest=digest,
                             progress=mine),
                    slot.params.eval_budget_s)
            except asyncio.TimeoutError:
                if not mine.get("measuring"):  # pull/start/load of this model: ours as much as theirs -> retried
                    raise InfraError(f"{slot.name}: budget of {slot.params.eval_budget_s:.0f} s ran out before the "
                                     "first timed request") from None
                r = EvalResult(digest=digest, track=track.name, ok=False, block=block,
                               reason=f"evaluation exceeded its {slot.params.eval_budget_s:.0f} s budget")
            except ReferenceFailedWithCandidate as exc:
                # decided here, with this model's parameters (the caller only knows the composite's)
                if await reference_alone_ok(runner, ref, slot.params):
                    raise InfraError(f"{slot.name}: reference failed while this candidate was loaded ({exc})") from None
                raise ReferenceError_(f"{slot.name}: reference engine is broken ({exc})") from None
            if progress is not None and mine.get("measuring"):
                progress["measuring"] = True
            r.track = f"{track.name}/{slot.name}"
            if slot.memory_budget_mib and r.ok and (r.cand_memory_mib is None or r.cand_memory_mib < 0):
                # our nvidia-smi failed (None) or this runner cannot read memory (-1): never a verdict, and never a
                # silently skipped budget
                raise InfraError(f"{slot.name}: GPU memory could not be read")
            results[slot.name] = r
            if slot_failure(slot, r):
                break
    finally:
        done = getattr(runner, "finish_candidate", None)
        if done is not None:
            await done(candidate_ref)
    cs = composite_score(track, declared, results)
    return to_eval_result(track, digest or candidate_ref.rsplit(":", 1)[-1], block, cs, results)


# ------------------------------------------------------------------------------------------------ the track
def _slot_params(name: str, pool: str, temperature: float, concurrency: int, noise: float, budget_s: float) -> TrackParams:
    """``pool``: the decision contract the items are built for (s1contract.MAX_OPTIONS: option count and description
    rules); the engine itself learns its model from REAX_MODEL, the request's model field is not used by it."""
    return TrackParams(name, pool=pool, temperature=temperature, concurrency=concurrency, reference_noise_std=noise,
                       eval_budget_s=budget_s)


# Measured 20 Oct 2026 (REAX 1: 21 Oct) on one RTX PRO 6000 Blackwell Server Edition with reax-engine main 69b6dc5 run natively (not yet the
# release reference images in their containers): calib/results of job engine-composite-track-20261020, summary in
# docs/ENGINE-COMPETITION-V2.md section 7. Gates = honest reference-vs-reference worst case (fresh processes, the slot's
# concurrency) with ~1.5-2x headroom; noise = fresh-process relative std of the reference throughput at that concurrency.
# The track is release-ready only once these numbers are re-measured on the release reference images (R3) and this flag
# is set by the release owner.
CALIBRATED_ON_RELEASE_REFERENCES = True  # 23 Oct: calibrated on the reference digests in config.ENGINE_COMPOSITE_REFERENCE_IMAGES

_DEFAULT_SLOTS = (
    # plumb: bit-identical across processes and concurrency; the v1 gates. 102.6 dec/s c16, 15.3 GiB, start ~40 s.
    dict(name="plumb", model="plumb-4b", pool="s1-fast", temperature=2.07, concurrency=16,
         noise=0.0070, memory_budget_mib=20_480, budget_s=1200.0, gates={}),
    # v2.1 release calibration, 23 Oct 2026 (engine-composite-open-20261023 gpucheck: RunPod native + Verda docker/runsc,
    # reference images reax-engine PR #9; qwen35 slots with REAX_MAX_WAIT_MS=4, which removes the greedy scheduler's
    # bimodal throughput). Noise = rel std of fresh-process runs at the slot concurrency; gates = worst honest
    # ref-vs-ref pair with headroom; memory budget ~1.3x steady; budget_s from the longest runsc session x 1.8.
    # blink (vLLM 0.30, preset ref-prod = production flags, gpu_memory_utilization 0.45): bit-identical across c1/c8/c16 and
    # repeats inside one process (across starts see the 26 Oct note below); 160.6 dec/s c16, 42.1 GiB (2x 84.2), start 142 s native / 202-255 s runsc; throughput is
    # bimodal inside vLLM (7.5 %), no engine-side knob.
    # Gates widened 26 Oct (hotfix, #28021/#28022): with CUDA graphs + torch.compile the reference autotunes a compiled kernel
    # per start, so two starts of the same image can differ. Measured: two co-resident starts (gpucheck3, 256 items, ab02e141
    # vs 4a5b94b3 whose text path differs only by a no-op top_p=1.0) moved 42/256 items, mean 0.042, max item 1.125, top
    # agreement 98.83 %; an identical-image session (gpucheck2) mean 0.004, max 0.643, top 99.41 %; a miner measured ~7 % of
    # items moving with max 1.03. GPU confirm (gpucheck-blinkgates-b, Shadeform runsc, byte copy of ab02e141): sessions mean
    # 0.003 / 0.030, item 0.464 / 0.625, top 100 % / 99.22 %; a raw co-resident pair of fresh starts 213/256 exact, mean 0.033,
    # item 0.469, top 98.44 % (old 99 % gate FAILS). The old gates (0.05 / 1.0 / 99 %) could fail a byte copy. New: mean 0.08
    # (~2x the worst 0.042), item 1.6 (as reax1; worst 1.125), top 97 % (as mammoth/d3; worst 98.44 %, so 98 % would leave one
    # flip per 256 items of headroom). Mean stays stricter than every qwen35 slot and carries the lossy-engine defence
    # (a lossy engine moves every item, honest drift moves 5-17 % of them). determinism_tau stays 0.05 (same process: bit-identical).
    dict(name="blink", model="blink-v0.3-26b-a4b", pool="s1-fast", temperature=1.3, concurrency=16,
         noise=0.0751, memory_budget_mib=52_224, budget_s=1200.0,
         gates={"mean_tau": 0.08, "item_tau": 1.6, "min_top_agreement": 0.97, "determinism_tau": 0.05}),
    # vision = s1-vision: the Blink v0.3 engine of slot blink (same weights, same reference image family, same REAX_MODEL), but
    # every timed item carries exactly one procedurally drawn picture (s1vision.py), as in production (field ``images``).
    # Release calibration 26 Oct (gpucheck-lineup-b, Verda runsc, ref-blink 97d06a91 = eager): two co-resident eager engines
    # are bit-identical on 69 image items at c1/c16 (with CUDA graphs only 16/69 were); vs production s1-vision every top
    # choice equal, max |dp| 0.049 (lead bound 0.05). 2/2 identical-image sessions pass (slot 1.005 / 1.016, all distances
    # 0); noise 0.30 % (3 fresh processes x 6 runs, fresh images each run, c16, 16.9 dec/s; repeated images hit the prefix
    # cache and run ~6x faster, which is why the probe tape uses fresh pictures); pair peak 89.6 GB (44.9 GB per engine);
    # sessions ~362 s.
    dict(name="vision", model="blink-v0.3-26b-a4b", params_name="s1-vision", pool="s1-vision", temperature=1.3,
         concurrency=16, noise=0.0030, memory_budget_mib=52_224, budget_s=1500.0,
         gates={"timed_items": 256, "mean_tau": 0.05, "item_tau": 1.0, "min_top_agreement": 0.99, "determinism_tau": 0.05},
         release_calibrated=True),
    # quyet = quyet-1-large: chinhnc/Quyet-1.0-Large@d997d589 (Gemma-4-31B fine-tune, Apache-2.0), production vLLM 0.30, FP8 weights +
    # FP8 KV, template quyet, at most 10 options, 8000 prompt tokens, temperatures choice 1.3007 / score 1.3159 / noul 1.4957
    # (node-letters configs/quyet.json). Started as provisional values (gates like pplx, concurrency 8, temperature of the
    # choice questions (the most frequent type; the engine applies the per-type ones itself), noise 5 % (a vLLM engine: between
    # pplx 0.5 % and blink 7.5 %), memory ~1.16x the vLLM pool (gpu_memory_utilization 0.40 of 96 GB), budget 1500 s.
    # quyet: release calibration 25 Oct (gpucheck4, ref-quyet da746016, runsc): 2/2 identical-image sessions pass these
    # gates (slot 0.947 / 0.955, worst mean 0.0026, top 99.95 %); noise 5.36 % all-run (3 fresh processes x 6, c8,
    # bimodal 46/52 dec/s); 41.5 GiB steady per engine, pair peak 85.3 GB; sessions 535-548 s. Golden parity vs
    # production |dp| <= 1.1e-16 (gpucheck3). TrackParams.temperature only scales the distance (engine applies
    # choice 1.3007 / score 1.3159 / noul 1.4957 itself).
    dict(name="quyet", model="quyet-1-large", pool="quyet-1-large", temperature=1.3007, concurrency=8,
         noise=0.0536, memory_budget_mib=51_200, budget_s=1500.0,
         gates={"timed_items": 1024, "mean_tau": 0.13, "item_tau": 1.1, "min_top_agreement": 0.98,
                "determinism_tau": 0.8},
         release_calibrated=True),
    # jeffan = Jeffan-31b-1710: tattrongvu/Jeffan-31b-1710@cebfa86c (Gemma-4-31B, Apache-2.0), served as decisionmodels.io
    # jeffan-31b. vLLM readout reference (ref-jeffan, FP8 weights + FP8 KV, util 0.44, max-len 16448, batch invariant), one
    # question per request, up to 26 options (public gateway limit; the engine takes 255), 8192 input tokens.
    # temperature 1.0 = the engine's own readout temperature (JEFFAN_DEFAULTS in reax-engine vllm_readout.py). TrackParams.temperature
    # only scales the logit distance of the gates (the engine applies its temperature itself), so it must be the engine's T.
    # Release calibration 26 Oct (gpucheck-lineup/-b, Verda runsc, ref-jeffan fdebc6e5): golden-500 tops 500/500 (max |dp|
    # 0.011 = the restart drift val-jeffan documented; same-process repeats exact); pair fits (peak 83.3 GB, 42.8 GB per
    # engine); 2/2 identical-image sessions pass (slot 0.960 / 0.956, distances 0); noise 2.17 % all-run (3 fresh x 6, c8,
    # 19.4 dec/s). Honest different engine (preset fp8-kv-colo, not batch-invariant) vs the reference passes these gates:
    # mean 0.021, item max 0.625, top 99.8 %, slot 1.042; the re-read (top-1 < 0.9) fires on ~79 % of the competition items
    # in both, so an engine that skips it fails the item gate. Sessions ~720 s -> budget 2160 s kept.
    # noise 3.5 % (not the 2.17 % throughput noise): the session scores of honest engines spread ~9 points (0.956 / 0.960
    # for reference-vs-reference, the co-resident candidate runs ~1 % slower; 1.042 for the other preset), so the floor
    # is 0.895 (critic-lineup2 N1); a 10 % regression on this 5 % slot moves the composite by 0.5 %.
    dict(name="jeffan", model="jeffan-31b-1710", pool="jeffan-31b", temperature=1.0, concurrency=8,
         noise=0.0350, memory_budget_mib=51_200, budget_s=2160.0,
         gates={"timed_items": 1024, "mean_tau": 0.13, "item_tau": 1.1, "min_top_agreement": 0.98,
                "determinism_tau": 0.8},
         release_calibrated=True),
    # reax1 (REAX 1, decisionmodels/reax-1@ca2a0cc5, joint v1.2 path through the REAX contract): not batch-invariant;
    # c1 bit-identical across processes; 66.3 dec/s c8 (window 4 ms), 29.8 GiB, start 183 s native / 196-443 s runsc.
    # item_tau 1.6: an identical-image session hit item distance 1.325 (FP8 batch-noise tail, 1024-item study max 0.94).
    # noise 3.25 %: validation round (runsc, final digest) measured 2.49 % and 3.24 % all-run rel std (window 4 ms).
    dict(name="reax1", model="reax-1", pool="s1-pro", temperature=1.9754, concurrency=8,
         noise=0.0325, memory_budget_mib=39_936, budget_s=1500.0,
         gates={"timed_items": 1024, "mean_tau": 0.19, "item_tau": 1.6, "min_top_agreement": 0.98,
                "determinism_tau": 1.15}),
    # pplx FP8 native: 67.7 dec/s c8 (window 4 ms), 29.3 GiB, start 123 s native / 135-267 s runsc; c1 bit-identical to
    # the 20 Oct reference answers.
    dict(name="pplx", model="pplx-decider-v1.1-27b", pool="s1-pro", temperature=1.0087417621345625,
         concurrency=8, noise=0.0051, memory_budget_mib=39_936, budget_s=1200.0,
         gates={"timed_items": 1024, "mean_tau": 0.13, "item_tau": 1.1, "min_top_agreement": 0.98,
                "determinism_tau": 0.8}),
    # d3 = d3-turbo (D3_SLOT_MODEL; vllm-sr/d3-turbo@cfa0d41d, FP8, prefix_bidir + state-once): c1 equals the production
    # d3-turbo reference F2 on 500 items (max |dp| 5e-17); 62.2 dec/s c8 (window 4 ms), 33.2 GiB, start 447 s native /
    # 664 s to ready under runsc; a runsc session took 1244 s -> budget 2400 s.
    dict(name="d3", model=None, pool="s1-pro", temperature=1.0, concurrency=8,
         noise=0.0057, memory_budget_mib=45_056, budget_s=2400.0,
         # mean_tau 0.17: an identical-image session failed the anti-cache probe mean at 0.134 (> 0.13) once; repro 0.095
         gates={"timed_items": 1024, "mean_tau": 0.17, "item_tau": 1.1, "min_top_agreement": 0.97,
                "determinism_tau": 1.1}),
    # mammoth = baselabs/mammoth-1@61cec44c281c693992e3d6ebeb44dd3402dfa186 (MIT, Qwen3.5), served as decisionmodels.io s1-pro
    # (the s1-pro engine d3-turbo is replaced): the s1-pro item pool and limits, like slot d3. Engine = REAX Qwen3.5
    # reference (ref-mammoth = ref-qwen35 with REAX_MODEL mammoth-1), state-once production recipe.
    # Release calibration 26 Oct (gpucheck-lineup-b, Verda runsc, ref-mammoth 3080a593): c1 bit-identical to the production
    # engine 5117ec3 on 500 items (and c1 repeat exact); at c16 both engines move with batch composition (tops 500/500, max
    # |dp| 0.018). Pair fits (67.9 GB, 33.9 GB per engine, cold start ~467 s). 2/2 identical-image sessions pass with the d3
    # gates (worst mean 0.087, item 0.333, top 99.4 %, probe 0.32, determinism 0.39); slot 0.980 / 0.996, ~1086 s.
    # Throughput noise 0.33-0.36 % (3 fresh x 6, c8, 51.9 dec/s), but the session score of an identical engine spread
    # 1.6 points, so noise is set to 1 % (floor 0.97 instead of 0.98: an honest equal engine must not fail the floor).
    dict(name="mammoth", model="mammoth-1", pool="s1-pro", temperature=1.0, concurrency=8,
         noise=0.0100, memory_budget_mib=45_056, budget_s=2400.0,
         gates={"timed_items": 1024, "mean_tau": 0.17, "item_tau": 1.1, "min_top_agreement": 0.97,
                "determinism_tau": 1.1},
         release_calibrated=True),
)


def default_track(calibration: Optional[Mapping[str, Mapping[str, Any]]] = None,
                  variant: Optional[str] = None) -> CompositeTrack:
    """The ``dm-all`` track (docs/ENGINE-COMPETITION-V2.md, section 3) with the measured per-model values above.
    ``calibration`` (development networks): per-slot overrides (``temperature``, ``concurrency``, ``noise``,
    ``memory_budget_mib``, ``budget_s``, ``gates``: TrackParams fields); a slot with an entry counts as calibrated.
    Without it a slot is calibrated when its own ``release_calibrated`` row marker says so; a row without the marker (the
    four slots of v2.1) follows the global CALIBRATED_ON_RELEASE_REFERENCES. ``mammoth``, ``jeffan``, ``quyet`` and ``vision`` carry ``False``
    until the GPU check has measured them (``release_ready`` then names them).
    ``variant``: weights table (default COMPOSITE_WEIGHTS_VARIANT); decides the slots, weights and the v1 carry-over."""
    from .config import ENGINE_COMPOSITE_CALIBRATED_ON, ENGINE_COMPOSITE_REFERENCE_IMAGES

    variant = variant or COMPOSITE_WEIGHTS_VARIANT
    if variant not in COMPOSITE_WEIGHTS:
        raise ValueError(f"unknown composite weights variant {variant!r}")
    weights = COMPOSITE_WEIGHTS[variant]
    cal = calibration or {}
    slots = []
    rows = {row["name"]: row for row in _DEFAULT_SLOTS}
    for row in (rows[n] for n in COMPOSITE_EVAL_ORDER if n in weights):
        c = cal.get(row["name"], {})
        model = D3_SLOT_MODEL if row["name"] == "d3" else row["model"]  # looked up here: one constant to roll back
        params = _slot_params(row.get("params_name", model), row["pool"], float(c.get("temperature", row["temperature"])),
                              int(c.get("concurrency", row["concurrency"])), float(c.get("noise", row["noise"])),
                              float(c.get("budget_s", row["budget_s"])))
        params = replace(params, **{**row["gates"], **dict(c.get("gates", {}))})
        calibrated = bool(c) or (CALIBRATED_ON_RELEASE_REFERENCES if row.get("release_calibrated") is None
                                 else bool(row["release_calibrated"]))
        # development calibration entries are measured by the developer on whatever is published: bound to it as of now
        calibrated_on = (ENGINE_COMPOSITE_REFERENCE_IMAGES.get(row["name"]) if c
                         else ENGINE_COMPOSITE_CALIBRATED_ON.get(row["name"])) or ""
        slots.append(ModelSlot(row["name"], model, weights[row["name"]], params,
                               int(c.get("memory_budget_mib", row["memory_budget_mib"])),
                               calibrated=calibrated, calibrated_on=calibrated_on))
    return CompositeTrack(COMPOSITE_PROTOCOL_TRACK, tuple(slots), carried=COMPOSITE_CARRIED[variant])


# ------------------------------------------------------------------------------------------------ the runner
class CompositeRunner:
    """One runner per model (each a DockerRunner with that model's weights directory, FLA config root and
    ``model_env={MODEL_ENV: model id}``), all on the same GPU, behind the Runner interface the v1 code uses. The engine
    for a measurement is chosen by ``params.name`` (= the slot's model id). The candidate image is pulled once and kept
    until ``finish_candidate``; image checks (owner label, submission review) go through the first runner."""

    binds_hotkey = True

    def __init__(self, runners: Mapping[str, Any]):
        if not runners:
            raise ValueError("CompositeRunner needs at least one runner")
        self.runners = dict(runners)
        self.first = next(iter(self.runners.values()))
        # image checks run on the first runner only; every runner starts candidate code, so all must share the gates
        for attr in ("release_mode", "docker_runtime", "review_dir", "user", "pull", "registries"):
            values = {repr(getattr(r, attr, None)) for r in self.runners.values()}
            if len(values) > 1:
                raise ValueError(f"CompositeRunner: all runners need the same {attr}")

    def _for(self, params: TrackParams) -> Any:
        try:
            return self.runners[params.name]
        except KeyError:
            raise ValueError(f"no runner for model {params.name!r}") from None

    async def preflight(self) -> None:
        for r in self.runners.values():
            pre = getattr(r, "preflight", None)
            if pre is not None:
                await pre()

    async def start(self, ref: str, model: str, params: TrackParams, keep_image: bool = False):
        return await self._for(params).start(ref, model, params, keep_image=True)

    async def owner_of(self, ref: str) -> Optional[str]:
        return await self.first.owner_of(ref)

    @property
    def review_dir(self) -> str:
        """The copy line (engvalidator ``_refresh_derived`` / ``_derive_one``) reads the submission reviews from here; the
        runners share it (checked in ``__init__``). Without it a composite entry would never be crownable."""
        return getattr(self.first, "review_dir", "") or ""

    async def recheck_copyline(self, ref: str) -> None:
        rec = getattr(self.first, "recheck_copyline", None)
        if rec is not None:
            await rec(ref)

    async def models_label(self, ref: str) -> Optional[str]:
        get = getattr(self.first, "models_label", None)
        return await get(ref) if get is not None else None

    async def review(self, ref: str) -> None:
        rev = getattr(self.first, "review", None)
        if rev is not None:
            await rev(ref)

    def awaiting_review(self, ref: str) -> bool:
        aw = getattr(self.first, "awaiting_review", None)
        return bool(aw(ref)) if aw is not None else False

    async def gpu_used_mib(self) -> Optional[int]:
        used = getattr(self.first, "gpu_used_mib", None)
        return await used() if used is not None else None

    async def discard(self, ref: str) -> None:
        for r in self.runners.values():  # only the runner that pulled an image removes it
            await r.discard(ref)

    async def finish_candidate(self, ref: str) -> None:
        await self.discard(ref)

    def weights_identity(self) -> str:
        parts = []
        for name, r in sorted(self.runners.items()):
            ident = getattr(r, "weights_identity", None)
            parts.append(f"{name}={ident() if callable(ident) else ''}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest() if parts else ""
