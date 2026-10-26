"""The specialist-track evaluator the validator runs: ``await SpecEvaluator(cfg).evaluate_cycle(snapshot)``.

Per cycle and active track: read the commitments, check new submissions (rules, tensor hashes, near-duplicates) without
GPU time, then run groups within the shared hourly budget, each on one fresh shared draw: the incumbent plus the
challengers in commitment order (first pass), or the incumbent plus the challengers whose 12-hour wait is over
(confirmation, on a disjoint draw). On a vacant track the first champion is decided per cohort (1.2.0rc7): entries that
passed their confirmation wait until every entry first seen in the same block or earlier is decided, then meet on one
shared draw (the playoff). State is saved after every group. The result maps each active track to its ranked
leaders ``[(uid, score), ...]`` in rank order (champion first, see ``compute_leaders``); turning that into weights is the job of the split layer.
"""

from __future__ import annotations

import asyncio
import copy
import errno
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncContextManager, Awaitable, Callable, Collection, Mapping, Sequence

from .. import competition_split as split
from ..chain import Snapshot
from ..competition_split import (CHAIN_IMMUNITY_BLOCKS, KEEPALIVE_BASE_FLOOR, KEEPALIVE_BIG_MARGIN, KEEPALIVE_BIG_PARAMS,
                                 KEEPALIVE_LEGACY_FLOOR, KEEPALIVE_MAX_ENTRIES, KEEPALIVE_NONPROC_FLOOR, PROVISIONAL_KA,
                                 PROVISIONAL_KA_LEAD_BLOCKS, PROVISIONAL_KA_MAX_HOURS,
                                 TRACK_ENTRY_FLOOR, RankedLeaders, spec_open_class_active)
from . import bases as B
from . import config as C
from . import fetch as F
from . import machine as M
from . import review as RV
from . import sampling as S
from . import scoring as SC
from . import state as ST
from . import tensors as T
from . import v5wire as V5W
from .commitment import SpecCommitment, parse_commitment
from .vllm import (Backend, GPUPreflightUnavailable, HostUnavailable, ModelBudgetOOM, ModelUnavailable, VllmDockerBackend,
                   model_specific_failure)

log = logging.getLogger("reax.spec")
SERVE_TIMEOUT_S = 3600
MARKER = ".reax-verified"
RENEWAL_WAIT = "renewal waits for disk: earlier sketches must be renewed before the near-duplicate check"
MAX_UNREFERENCED_DRAWS = 40    # draws kept in the state besides those a pending entry still needs (``_prune_draws``)
# A PENDING_VERIFY entry whose wait is this host's (``ST.WAIT_INFRA``: base download deferred by the per-cycle budget or
# the disk, in progress, not storable; sketch renewal waiting for disk) holds back its cohort's first crowning. A wait on
# the miner's side (private or missing repo, base not on the hub, weights not comparable) does not. No open entry holds
# a cohort back longer than COHORT_MAX_WAIT_S after its first passed confirmation (a base larger than the cycle's download
# budget is deferred on every cycle; a model that never starts on this host fails every playoff).
COHORT_MAX_WAIT_S = 12 * 3600
# A first-champion playoff whose best and second-best scores differ by less than this many paired standard errors of
# their difference runs one more shared draw for the tied top group (decided on the mean of both); still tied: the earlier
# commitment wins. (The first DF crown was 77.09 vs 76.27 vs 76.15 on one draw, about half a standard error.)
PLAYOFF_MIN_GAP_SE = 1.0

BackendFactory = Callable[[str, int], AsyncContextManager[Backend]]

# Behavioural near-duplicate check (docs/SPECIALIST-TRACKS.md, "Near-duplicates"). On the shared draw, a candidate whose
# answers are those of an earlier submission served on the same draw is derived from it, whatever its weights look
# like (permuted hidden units, rescaled layers, noise on the sampled positions). Calibration, in Jensen-Shannon bits:
# serving the same weights twice moves the logits by numerical noise only (BF16 logits hold ~2^-5 near |logit| 8; eager
# vs CUDA graphs moved the competence by ~0.2 points on 513 items), i.e. a logit noise sd <= 0.05 and a mean JSD
# <= ~5e-4 bits (tests/test_copycheck2.py measures it on the fixture). Two independent fine-tunes differ in their own
# logit shifts; an independent part of sd >= 0.3 already gives >= ~0.012 bits. The threshold sits a factor 10 above the
# noise ceiling. The rank rule catches a copy whose probabilities were rescaled (a temperature on lm_head): the order of
# the labels stays the same on (almost) every item, which two independent models do not reach. It counts only items with
# at least 3 labels: on a yes/no item the order is just the argmax, which two independent fine-tunes of one task share
# on nearly every item (review rc6 F5). It also needs the JSD below TWIN_ORDER_JSD_BITS, far above what a temperature
# near 1 moves. The rule: twin = items >= TWIN_MIN_ITEMS and (jsd < TWIN_JSD_BITS or (order items >= TWIN_MIN_ORDER_ITEMS
# and same order >= TWIN_SAME_ORDER and jsd < TWIN_ORDER_JSD_BITS)).
TWIN_JSD_BITS = 0.005          # mean JSD below this: the same behaviour
TWIN_SAME_ORDER = 0.99         # or: the labels in the same order (equal within TWIN_TIE) on at least this share of the items
TWIN_ORDER_JSD_BITS = 0.05     # ... with at least 3 labels, and a mean JSD below this
TWIN_TIE = 1e-6
TWIN_MIN_ITEMS = 50            # items valid for both models needed for any verdict
TWIN_MIN_ORDER_ITEMS = 20      # items with at least 3 labels needed for the order rule
# Near-copy rules (REAX_SPEC_NC_R1/R2/R3, all off by default; with all off the behaviour is bit for bit the one above).
# R1: a twin or weight parent (the incumbent included) must also be beaten on competence by this many points (and 2 paired SE).
TWIN_COMP_FLOOR = 1.0
# R2: the probabilities of one model raised to a power s (= its letter logits scaled by s) and renormalised; the best s on
# this grid is tried against the other model. A pure temperature copy then has a JSD of numerical noise at its s.
TWIN_TEMP_GRID = tuple(round(0.40 + 0.05 * i, 2) for i in range(43))     # 0.40 .. 2.50
NC_ALPHA_TOL = 0.05            # R3: derived when cos > 0.98 and |alpha - 1| < this, whatever the relative L2 distance
TWIN_REL_TIE = 1e-6            # R2: two probabilities are tied when they differ by at most this share of the larger
# Near-champion rule (REAX_SPEC_NC_NEAR_CHAMP, off by default; decided 19 Oct 2026, announced in #92 before it is switched on).
# A challenger whose weights are close to the incumbent's on the validator's delta sketch (cos > NEAR_CHAMP_COS or relative
# L2 < NEAR_CHAMP_REL, the incumbent's delta as the reference) must, on every draw that decides it against the incumbent
# (first draw, close call, confirmation), also answer more items correctly than the incumbent and gain at least
# max(the draw's margin, NEAR_CHAMP_FLOOR) points. Rescaling the champion's answer letters or adding weight noise moves the
# calibration part of the score, not the answers; this asks a near copy for better answers and a clear lead.
# "Items right" here is the most probable label being the gold one (``ItemScore.top_hit``): unlike the scored correctness
# (noul decision thresholds, rounded expected level), a temperature (one scale for all answer letters) cannot change it. Also close: cos >
# NEAR_CHAMP_ALPHA_COS and the challenger's delta has the champion's length along it (|alpha - 1| < NEAR_CHAMP_ALPHA_TOL),
# i.e. the champion plus noise orthogonal to its change, however much (critic-nc F2; no legitimate entry comes near it).
# A challenger on another declared base is sketched against the champion's base for this comparison (critic-nc F1).
NEAR_CHAMP_COS = 0.995
NEAR_CHAMP_REL = 0.15
NEAR_CHAMP_ALPHA_COS = 0.98
NEAR_CHAMP_ALPHA_TOL = 0.05
NEAR_CHAMP_FLOOR = 3.0


@dataclass(frozen=True)
class BehaviourStats:
    items: int            # items with a valid answer from both models
    jsd_bits: float       # mean Jensen-Shannon divergence of the option distributions (0 = identical, 1 = disjoint)
    mean_abs: float       # mean absolute probability difference per option
    same_order: float     # share of the items with at least 3 labels that have the labels in the same order
    order_items: int = 0  # items with at least 3 labels (the base of ``same_order``)
    jsd_temp_bits: float | None = None   # R2: lowest mean JSD over the temperature grid (None = R2 off)
    temp_scale: float | None = None      # R2: the grid power that reached it

    def twin(self) -> bool:
        if self.items < TWIN_MIN_ITEMS:
            return False
        if self.jsd_temp_bits is not None and self.jsd_temp_bits < TWIN_JSD_BITS:
            return True
        return self.jsd_bits < TWIN_JSD_BITS or (self.order_items >= TWIN_MIN_ORDER_ITEMS
                                                 and self.same_order >= TWIN_SAME_ORDER and self.jsd_bits < TWIN_ORDER_JSD_BITS)

    def as_dict(self) -> dict[str, float]:
        out = {"items": self.items, "jsd": round(self.jsd_bits, 6), "abs": round(self.mean_abs, 6), "order": round(self.same_order, 4),
               "order_items": self.order_items}
        if self.jsd_temp_bits is not None:
            out.update(jsd_temp=round(self.jsd_temp_bits, 6), temp_scale=self.temp_scale)
        return out


def _order(p: Sequence[float], rel_tie: bool = False) -> tuple[int, ...]:
    if rel_tie:   # R2: a tie is a relative notion (tail probabilities of 1e-9 differ by far less than 1e-6 absolute)
        return tuple(0 if abs(a - b) <= TWIN_REL_TIE * max(a, b) else (1 if a > b else -1) for i, a in enumerate(p) for b in p[i + 1:])
    return tuple(0 if abs(a - b) < TWIN_TIE else (1 if a > b else -1) for i, a in enumerate(p) for b in p[i + 1:])


def _jsd_over_grid(va: Sequence[float], vb: Sequence[float]) -> list[float]:
    """JSD (bits) between normalise(va ** s) and vb for every s of TWIN_TEMP_GRID (an unusable grid point gives 1.0)."""
    out: list[float] = []
    for s in TWIN_TEMP_GRID:
        pw = [x ** s if x > 0 else 0.0 for x in va]
        z = math.fsum(pw)
        if z <= 0:
            out.append(1.0)
            continue
        j = 0.0
        for x, y in zip(pw, vb):
            x /= z
            m = (x + y) / 2
            j += (x * math.log2(x / m) if x > 0 else 0.0) + (y * math.log2(y / m) if y > 0 else 0.0)
        out.append(max(0.0, j / 2))
    return out


def behaviour_stats(rows: Sequence[Mapping[str, Any]], a: Sequence[Any], b: Sequence[Any], *, temp_invariant: bool = False,
                    rel_tie: bool = False) -> BehaviourStats:
    """Per-item option distributions of two models on the same rows, compared. ``temp_invariant`` / ``rel_tie``: near-copy
    rule R2 (a temperature-invariant JSD and a relative order tie); both off = the original statistics."""
    jsd = mad = 0.0
    grid = [0.0] * len(TWIN_TEMP_GRID)
    same = n = n_order = 0
    for r, pa, pb in zip(rows, a, b):
        labels = [str(x) for x in r["labels"]]
        ca, cb = SC.clean_probs(pa, labels), SC.clean_probs(pb, labels)
        if ca is None or cb is None:
            continue
        va, vb = [ca[lab] for lab in labels], [cb[lab] for lab in labels]
        j = 0.0
        for x, y in zip(va, vb):
            m = (x + y) / 2
            j += (x * math.log2(x / m) if x > 0 else 0.0) + (y * math.log2(y / m) if y > 0 else 0.0)
        jsd += max(0.0, j / 2)
        if temp_invariant:
            grid = [g + x for g, x in zip(grid, _jsd_over_grid(va, vb))]
        mad += sum(abs(x - y) for x, y in zip(va, vb)) / len(labels)
        if len(labels) >= 3:
            same += _order(va, rel_tie) == _order(vb, rel_tie)
            n_order += 1
        n += 1
    if n == 0:
        return BehaviourStats(0, 1.0, 1.0, 0.0)
    if temp_invariant:
        k = min(range(len(grid)), key=lambda i: (grid[i], i))
        return BehaviourStats(n, jsd / n, mad / n, same / n_order if n_order else 0.0, n_order, grid[k] / n, TWIN_TEMP_GRID[k])
    return BehaviourStats(n, jsd / n, mad / n, same / n_order if n_order else 0.0, n_order)


def nonproc_scores(items: Sequence[SC.ItemScore]) -> tuple[float | None, dict[str, float]]:
    """The score on the draw's non-proc items (tier != "proc_sealed"; None if it has none) and the score per tier."""
    rest = [it for it in items if it.tier != "proc_sealed"]
    nonproc = SC.score_items(rest).total if rest else None
    return nonproc, {t: round(sc.total, 4) for t, sc in SC.by_tier(items).items()}


async def score_model(backend: Backend, rows: Sequence[Mapping[str, Any]]) -> tuple[list[SC.ItemScore], SC.Score]:
    """Shared by the validator and the miner CLI: serve all rows, score the answers."""
    probs = await backend.decide(rows)
    return SC.score_run(rows, probs)


def _crown_pending(s: ST.Submission) -> float | None:
    """CROWN_SLOTS_UTC: when the crowning decision of this entry was taken, while it waits for its slot (a ``crown_pending``
    history row not followed by ``crown_done``, status still pending confirmation); None otherwise. Kept in the history so
    the state file has no new field."""
    if s.status != ST.PENDING_CONFIRM:
        return None
    rows = [h for h in s.history if h.get("kind") in ("crown_pending", "crown_done")]
    at = rows[-1].get("ts") if rows and rows[-1]["kind"] == "crown_pending" else None
    return float(at) if isinstance(at, (int, float)) and not isinstance(at, bool) and math.isfinite(at) else None


def cochamp_keys(ts: ST.TrackState) -> tuple[str, ...]:
    """The keys of the co-champion set that may hold the title on this track: the shared group crowned at a slot
    (``TrackState.cochamps``, rc11.post3) if the state has one, else the hard-coded generalist opening set
    (``competition_split.GENERALIST_CO_CHAMPIONS``) on its track. Whether it is in force: ``cochamp_members``. Pure."""
    if getattr(ts, "cochamps", None):    # getattr: a TrackState parsed by an older loader has no such field
        return tuple(ts.cochamps)
    return tuple(split.GENERALIST_CO_CHAMPIONS) if ts.track_id == split.GENERALIST_CO_CHAMPIONS_TRACK else ()


def cochamp_since(ts: ST.TrackState) -> float | None:
    """When the co-champion set of ``cochamp_keys`` was crowned (None = unknown)."""
    if getattr(ts, "cochamps", None):
        return ts.cochamps_since
    at = split.GENERALIST_CO_CHAMPIONS_CROWNED
    return at.timestamp() if at is not None else None


def cochamp_members(ts: ST.TrackState, keys: Sequence[str] | None = None) -> list[ST.Submission]:
    """The submissions of the co-champion set in force (``cochamp_keys``: a shared group crowned at a slot, else the
    generalist opening set ``competition_split.GENERALIST_CO_CHAMPIONS``): ``[]`` while there is no set, or when the
    track's champion is not one of the listed keys (a challenger has taken the title: the set has ended and every entry is
    an ordinary one). Otherwise the listed submissions that exist and are the champion, or pending confirmation and
    copy-checked, in the order of the list (the champion is moved first). A withdrawn, rejected, not-better or former
    record is no member. Whether a member still commits on chain, and its owner approval, are the share layer's business.
    A shared group crowned at a slot (``TrackState.cochamps``) stays in force while the track has no champion seat at all
    (rc11.post4, CRITERIA 6: its leader withdrew or was rejected): the remaining members hold the title, in list order, and
    the set is vacant only when the last member has left (the worker re-anchors it, ``SpecEvaluator._reanchor_cochamps``).
    Pure."""
    dynamic = keys is None and bool(getattr(ts, "cochamps", None))
    if keys is None:
        keys = cochamp_keys(ts)
    elif ts.track_id != split.GENERALIST_CO_CHAMPIONS_TRACK:
        return []
    if not keys or (ts.champion not in keys and not (dynamic and not ts.champion)):
        return []
    out = [ts.submissions[k] for k in keys if k in ts.submissions and (
        k == ts.champion or (ts.submissions[k].status == ST.PENDING_CONFIRM and ts.submissions[k].delta_checked))]
    out.sort(key=lambda m: m.key != ts.champion)
    return out


def v5_bar_members(ts: ST.TrackState, now: float) -> list[ST.Submission]:
    """The references of a v5.1 challenger draw besides the champion: the CURRENT title holders only (Florian, 22 Oct, rule A).
    While the opening co-champion set holds the title (``cochamp_members``): every seated co-champion, champion first. After a
    rule-A crowning at the crown slot (the set has ended for the payments, rc11.post1) the new champion alone is the bar, with
    the challenger's own base: the former champions and the pending members of the set are not in it and are drawable and
    servable as normal entries. Pure. Gate off: the rc11 set, nothing more (the evaluator only reads it under the gate)."""
    return cochamp_members(ts)


def _decided_under_v5(s: ST.Submission) -> bool:
    """True when the entry's confirmation (the decision that stored its crowning) was measured under v5.1."""
    return any(h.get("kind") == "confirm" and h.get("scoring_regime") == V5W.REGIME for h in s.history)


def _v5_staged(s: ST.Submission) -> bool:
    """True when the entry's open first stage was measured under v5.1 (its first-draw row carries the regime)."""
    return any(h.get("kind") == "first" and h.get("scoring_regime") == V5W.REGIME
               for h in s.history if h.get("draw") in {s.draw1, *s.close_draws})


def opening_playoff_slots(ts: ST.TrackState, now: float) -> list[tuple[ST.Submission, float]]:
    """Generalist opening cohort (``competition_split.OPENING_PLAYOFF_SLOTS_START_UTC``): the entries that hold a
    former-champion slot at ``now`` as playoff places 2 and 3, ``[(submission, playoff score), ...]`` in place order.
    ``[]`` while the constant is None, before it, outside the entry's 7 days or without an opening playoff.

    The opening playoff is the one the track's first champion (earliest ``champion_since_ts``) was crowned from: its
    ``kind == "playoff"`` history row names the draw id, the field is every entry with a playoff row on that id (no row = a
    lone crowning = no slots; independent of ``ts.draws``, which is pruned). It counts if the crowning was at or after
    ``GENERALIST_START_UTC`` and the whole field was committed before it (cohort block 0). The playoff's own order is rebuilt
    from the rows: entrants that took the extension draw (``draw2``) first, then the stored score, best first, ties by earlier
    commitment (``_decide_playoff``'s ``order``); entries that sort above the winner were skipped by the decision and hold no
    place; the champion is place 1, the next ``OPENING_PLAYOFF_PLACES`` entries are places 2 and 3. Each former champion
    crowned since the opening crowning (any commitment) removes the lowest place for good. A place whose entry does not
    qualify is not filled by a lower one. Qualification (state only): playoff score above the track entry floor; non-proc
    score (total for a legacy row) > the declared base's ``KEEPALIVE_BASE_FLOOR`` for the track +
    ``OPENING_PLAYOFF_BASE_MARGIN`` (strict, as ``_big_ok``; no base hash or floor: no); not rejected, withdrawn, a copy/twin
    (``derived_from``) or unchecked (``delta_checked``); still a plain non-champion, non-former entry (those hold their
    normal rank). Owner approval is not read here (``combine_weights`` asks ``award_ok`` as for any rank). Both slots run from
    max(the crowning, the constant) for ``OPENING_PLAYOFF_SLOT_DAYS``. Pure."""
    start = split.OPENING_PLAYOFF_SLOTS_START_UTC
    if start is None or ts.track_id != "generalist" or now < start.timestamp():
        return []
    open_ts = split.GENERALIST_START_UTC.timestamp()
    # anchor on the track's first champion, not on ts.draws (the playoff draw is pruned once 40 newer ones exist)
    crowned = [(s.champion_since_ts, s.key) for s in ts.submissions.values()
               if isinstance(s.champion_since_ts, (int, float)) and math.isfinite(s.champion_since_ts)]
    if not crowned:
        return []
    crowned_at, winner = min(crowned)
    first = ts.submissions[winner]
    draw_id = next((h["draw"] for h in first.history if h.get("kind") == "playoff" and isinstance(h.get("draw"), str)), None)
    if draw_id is None or crowned_at < open_ts:      # no playoff row: a lone crowning; before the opening
        return []
    field = [(s, h) for s in ts.submissions.values() for h in s.history
             if h.get("kind") == "playoff" and h.get("draw") == draw_id]
    if any(s.first_seen_ts >= open_ts for s, _ in field):
        return []
    slot_start = max(crowned_at, start.timestamp())
    if not slot_start <= now < slot_start + split.OPENING_PLAYOFF_SLOT_DAYS * 86400.0:
        return []
    ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)  # noqa: E731
    order = lambda x: ("draw2" not in x[1], -x[1]["score"], x[0].first_seen_block, x[0].uid, x[0].key)  # noqa: E731
    won = next((x for x in field if x[0].key == winner and ok(x[1].get("score"))), None)
    if won is None:
        return []
    # entries that sort above the winner were skipped by the decision (twin, no margin, below the floor after the
    # extension): they hold no place
    # the opening co-champions (GENERALIST_CO_CHAMPIONS) are paid as champions and hold no place; the next entries in the
    # playoff's own order take places 2 and 3
    co = set(split.GENERALIST_CO_CHAMPIONS)
    rest = sorted((x for x in field if x[0].key != winner and x[0].key not in co and ok(x[1].get("score"))
                   and order(x) > order(won)), key=order)
    # a former champion crowned since the opening crowning pushed one slot out for good, whatever it commits to now
    pushed = sum(1 for s in ts.submissions.values() if s.status == ST.FORMER and s.delta_checked
                 and isinstance(s.champion_since_ts, (int, float)) and s.champion_since_ts >= crowned_at)
    floor = TRACK_ENTRY_FLOOR.get(ts.track_id)
    out: list[tuple[ST.Submission, float]] = []
    for s, h in rest[:max(0, split.OPENING_PLAYOFF_PLACES - pushed)]:
        base = (KEEPALIVE_BASE_FLOOR.get(s.base_id) or {}).get(ts.track_id) if s.base_id else None
        qual = h.get("nonproc") if h.get("nonproc") is not None else h["score"]
        if s.status in (ST.REJECTED, ST.WITHDRAWN, ST.CHAMPION, ST.FORMER) or s.derived_from or not s.delta_checked:
            continue
        if floor is not None and h["score"] <= floor:
            continue
        if not ok(base) or not ok(qual) or qual <= base + split.OPENING_PLAYOFF_BASE_MARGIN:
            continue
        out.append((s, float(h["score"])))
    return out


def _v51_scored(s: ST.Submission, now: float | None) -> bool:
    """The entry's latest score is S_v5.1 (a DI-scale number) and the v5.1 gate is open at ``now`` (critic F2: a gate that is
    unset again, or a state shared with a release without the gate, makes the scores v4 again). The v4-scale release floors
    (entry floor, non-proc floor, base floors) mean nothing for it: it earned its place by beating the best of the
    co-champions and its declared base by the margin on the same items (``_decide``), so those floors are not applied to it.
    The marker is written only for challengers, the champion and the seated co-champions of a draw, never for a parent."""
    return S.v5_active(now) and V5W.scored_under_v5(s.by_tier)


def _v51_keepalive_margin(s: ST.Submission) -> float:
    """Average S_v5.1 minus the deciding reference across this entry's v5 first/confirmation draws.

    Keep-alive's top-eight cut can include v4 and v5 scores. A v5 absolute score is on a different scale, so rank it by the
    margin the entry had to clear on its deciding draws. Old/imported v5 markers with no deciding rows remain eligible and
    sort at a neutral margin.
    """
    margins = []
    for row in s.history:
        if row.get("kind") not in ("first", "confirm") or row.get("scoring_regime") != V5W.REGIME:
            continue
        score, reference = row.get("s_v51"), row.get("incumbent")
        if (isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
                and isinstance(reference, (int, float)) and not isinstance(reference, bool) and math.isfinite(reference)):
            margins.append(float(score) - float(reference))
    return math.fsum(margins) / len(margins) if margins else 0.0


def compute_leaders(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot,
                    limit: int | None = C.MAX_LEADERS, now: float | None = None) -> dict[str, list[tuple[int, float]]]:
    """Champion first, then former champions (most recently crowned first), only submissions their hotkey still
    commits to. This order IS the rank (docs/SPECIALIST-TRACKS.md, "Ranks"): the result is a ``RankedLeaders``, which the
    share layer pays in this order. The scores come from different draws and are reported, never compared. ``limit``:
    at most this many per track (None = all; only the legacy keep-alive uses None).

    Pure function of the saved state and the chain snapshot (no docker, no GPU): the validator in remote mode calls it on
    the state file the competition worker wrote, so the hotkey/uid and commitment are re-verified on every call.

    ``now`` (unix seconds, default the wall clock) matters only while ``OPENING_PLAYOFF_SLOTS_START_UTC`` is set: the
    generalist's opening-playoff entries (``opening_playoff_slots``) follow the real former champions, so they fill ranks 2
    and 3 only (never rank 1) and are cut first at ``limit``; their score is the playoff score."""
    live = {n.hotkey: (n.uid, n.commitment) for n in snapshot.neurons}
    out: dict[str, list[tuple[int, float]]] = {}
    for track_id in track_ids:
        ts = state.track(track_id)
        ranked: list[ST.Submission] = []
        champ = ts.submissions.get(ts.champion)
        members = cochamp_members(ts)
        co_n = len(cochamp_keys(ts))
        co_fracs: dict[int, float] = {}
        if members:
            # opening co-champions (GENERALIST_CO_CHAMPIONS): the committed members share rank 1, 1/n of it each. Rank 1 of
            # the list is the first member that still commits; the others are not listed (``cochamps`` carries them)
            floor_t = TRACK_ENTRY_FLOOR.get(track_id)
            for m in members:
                uid_c = live.get(m.hotkey)
                c = parse_commitment(uid_c[1]) if uid_c is not None else None
                v51 = _v51_scored(m, now)
                if c is None or f"{m.hotkey}|{c.text()}" != m.key or m.score is None or not math.isfinite(m.score) \
                        or (m.score <= 0 and not v51) or (floor_t is not None and m.score <= floor_t and not v51):
                    continue
                if uid_c[0] not in co_fracs:
                    co_fracs[uid_c[0]] = 1.0 / co_n
                    if not ranked:
                        ranked.append(m)
        elif champ is not None:
            ranked.append(champ)
        # a former champion crowned by a release whose copy check could pass without the base is not ranked
        former = [s for s in ts.submissions.values() if s.status == ST.FORMER and s.delta_checked]
        form_fracs: dict[int, float] = {}
        set_keys = split.GENERALIST_CO_CHAMPIONS if track_id == split.GENERALIST_CO_CHAMPIONS_TRACK else ()
        # the set is beaten once any listed entry is a former champion; the listed entries still pending count with it, so a
        # state written by the rc11.post1 worker (only the champion member former) pays the same as a post2 one
        form_set = [s for s in ts.submissions.values() if s.key in set_keys and s.delta_checked
                    and s.status in (ST.FORMER, ST.PENDING_CONFIRM)]
        if not any(s.status == ST.FORMER for s in form_set):
            form_set = []
        # the shared groups crowned at a slot and beaten since (TrackState.former_sets, rc11.post3): ONE former champion each
        dyn_sets = [(fs["keys"], fs.get("since")) for fs in getattr(ts, "former_sets", None) or [] if fs.get("keys")]
        in_sets = set(set_keys) | {k for keys, _ in dyn_sets for k in keys}
        timed = [(s.champion_since_ts or 0.0, s) for s in former if s.key not in in_sets]
        dyn_fracs: list[dict[int, float]] = []
        for keys, since in dyn_sets:
            fracs: dict[int, float] = {}
            standing = None
            for k in keys:
                m = ts.submissions.get(k)
                if m is None or m.status != ST.FORMER or not m.delta_checked or m.score is None:
                    continue
                uid_c = live.get(m.hotkey)
                c = parse_commitment(uid_c[1]) if uid_c is not None else None
                if c is None or f"{m.hotkey}|{c.text()}" != m.key:
                    continue
                if uid_c[0] not in fracs:
                    fracs[uid_c[0]] = 1.0 / len(keys)
                    standing = standing or m
            if standing is not None:
                when = since if isinstance(since, (int, float)) else max(ts.submissions[k].champion_since_ts or 0.0
                                                                          for k in keys if k in ts.submissions)
                timed.append((when, standing))
                dyn_fracs.append(fracs)
        dyn_now = bool(getattr(ts, "cochamps", None))
        if form_set and not (members and not dyn_now):    # not while the opening set itself holds the title
            # the opening co-champion set after a challenger beat it (rc11.post2): ONE former champion. Its first member
            # that still commits stands for it in the list; its place is split 1/n over the committed members (formset)
            crowned_set = split.GENERALIST_CO_CHAMPIONS_CROWNED
            when = crowned_set.timestamp() if crowned_set is not None else max(s.champion_since_ts or 0.0 for s in form_set)
            standing = None
            for m in sorted(form_set, key=lambda m: set_keys.index(m.key)):
                uid_c = live.get(m.hotkey)
                c = parse_commitment(uid_c[1]) if uid_c is not None else None
                if c is None or f"{m.hotkey}|{c.text()}" != m.key or m.score is None:
                    continue
                if uid_c[0] not in form_fracs:
                    form_fracs[uid_c[0]] = 1.0 / len(set_keys)
                    standing = standing or m
            if standing is not None:
                timed.append((when, standing))
        elif form_set:
            timed += [(s.champion_since_ts or 0.0, s) for s in form_set if s.status == ST.FORMER]
        ranked += [s for _, s in sorted(timed, key=lambda t: -t[0])]
        slots: dict[str, float] = {}
        if split.OPENING_PLAYOFF_SLOTS_START_UTC is not None:
            opening = opening_playoff_slots(ts, time.time() if now is None else now)
            slots = {s.key: sc for s, sc in opening}
            ranked += [s for s, _ in opening]
        res = RankedLeaders()
        v5_floor = TRACK_ENTRY_FLOOR.get(track_id) if track_id == "generalist" and S.v5_active(now) else None
        for s in ranked:
            uid_c = live.get(s.hotkey)
            if uid_c is None or (s.score is None and s.key not in slots):
                continue
            if s.key in slots and (not res or any(u == uid_c[0] for u, _ in res)):
                continue    # a slot is a former-champion rank (never rank 1) and a uid counts once
            c = parse_commitment(uid_c[1])
            if c is None or f"{s.hotkey}|{c.text()}" != s.key:
                continue
            if any(u == uid_c[0] for u, _ in res):
                continue
            score_c = slots[s.key] if s.key in slots else float(s.score)
            if v5_floor is not None and score_c <= v5_floor and not _v51_scored(s, now):
                continue    # critic F6: once the share layer drops the v4 floor, an entry that is not scored under v5.1 keeps it
            res.append((uid_c[0], score_c))
            if isinstance(s.champion_since_ts, (int, float)) and math.isfinite(s.champion_since_ts):
                res.crowned.setdefault(uid_c[0], float(s.champion_since_ts))

            if limit is not None and len(res) >= limit:
                break
        if co_fracs and res and res[0][0] in co_fracs:
            res.cochamps = dict(co_fracs)
            if dyn_now:     # a shared group crowned at a slot: its crowning time stands for the whole group
                res.cochamps_dynamic = True
                if ts.cochamps_since is not None:
                    res.crowned[res[0][0]] = float(ts.cochamps_since)
        if form_fracs and any(u in form_fracs for u, _ in res):
            res.formset = dict(form_fracs)
        res.formsets = [dict(f) for f in dyn_fracs if any(u in f for u, _ in res)]
        out[track_id] = res
    return out


def legacy_leaders(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot) -> dict[str, list[tuple[int, bool]]]:
    """Legacy keep-alive input (competition_split.LEGACY_KA_*): ``compute_leaders`` of the ended tracks, each entry as
    ``(uid, is_champion)``; True only for the track's current champion (rank 1 can also be a former champion when the
    champion no longer commits). Same checks as ``compute_leaders`` (hotkey registered, still commits to the SAME
    submission key), so a hotkey that commits to another track drops out. Not cut at ``MAX_LEADERS``: every approved
    former champion counts (the share layer caps the uids at ``LEGACY_KA_MAX_ENTRIES``). A track missing from the state
    gives no entry (the state is not changed). Pure."""
    live = {n.hotkey: (n.uid, n.commitment) for n in snapshot.neurons}
    ids = [t for t in track_ids if t in state.tracks]
    leaders = compute_leaders(state, ids, snapshot, limit=None)
    out: dict[str, list[tuple[int, bool]]] = {}
    for track_id in ids:
        ts = state.tracks[track_id]
        champ = ts.submissions.get(ts.champion) if ts.champion else None
        champ_uid = None
        if champ is not None and champ.status == ST.CHAMPION and champ.score is not None and champ.hotkey in live:
            c = parse_commitment(live[champ.hotkey][1])
            if c is not None and f"{champ.hotkey}|{c.text()}" == champ.key:
                champ_uid = live[champ.hotkey][0]
        res = leaders.get(track_id, ())
        crowned = getattr(res, "crowned", {})
        # only entries crowned before the switch: a DF/RA worker still evaluating after it never adds a payee
        out[track_id] = [(uid, i == 0 and uid == champ_uid) for i, (uid, _) in enumerate(res)
                         if crowned.get(uid, math.inf) < split.GENERALIST_START_UTC.timestamp()]
    return out


# Keep-alive (docs/SPECIALIST-TRACKS.md, "Keep-alive"): the statuses that can hold a slot, in slot priority order (a
# coldkey with more qualifying entries than slots keeps the champion first). A keep-alive bridges a decision, it is not
# a standing payment: an entry pending confirmation holds it for at most KEEPALIVE_HOURS after its first pass, a new
# champion (waiting for the owner's approval of its award) for at most KEEPALIVE_HOURS after it was crowned. Decided
# entries (former champions, not better, rejected) and QUEUED, PENDING_VERIFY or WITHDRAWN entries never qualify. An entry
# must also score above ``KEEPALIVE_NONPROC_FLOOR`` on the non-proc items of its latest draw (``Submission.nonproc``), so
# learning the public procedural families alone does not earn it. An entry last scored before 1.2.0rc6 has no such score
# (``nonproc`` None and no ``by_tier``); it qualifies only with a total score above ``KEEPALIVE_LEGACY_FLOOR`` (1.2.0rc7),
# which a rules-only model cannot reach under the rc5 mix. In the open class (competition_split.SPEC_OPEN_CLASS_START_UTC,
# 1.2.0rc9.post6) an entry above ``KEEPALIVE_BIG_PARAMS`` parameters needs the base floor of its declared base on top
# (``_big_ok``; a larger entry without a recorded base hash gets none).
KEEPALIVE_STATUSES = (ST.CHAMPION, ST.PENDING_CONFIRM)
KEEPALIVE_HOURS = 24.0
KEEPALIVE_FUTURE_S = 3600.0    # a window start further in the future than this gives no slot


def _big_ok(s: ST.Submission, track_id: str, now: float, base_floors: Mapping[str, Mapping[str, float]] | None = None,
            qual: float | None = None) -> bool:
    """Open-class base floor (``KEEPALIVE_BASE_FLOOR``). Before the open class starts: always True. From it on: an entry with
    at most ``KEEPALIVE_BIG_PARAMS`` parameters is not affected; a larger entry needs its declared base's tensor hash
    (``base_id``) in the table with a floor for the track and a qualifying score ``qual`` above that floor +
    ``KEEPALIVE_BIG_MARGIN`` (no score, e.g. a provisional entry: never). A missing base hash or floor, or an empty table,
    gives False for a larger entry (fail closed). A parameter count that is not recorded (``None``) counts as small: every fetch
    of this release records it, so ``None`` means fetched by an earlier release, under the 9 B cap, or not fetched yet (only the
    time-bounded provisional keep-alive applies to those, and it stops once the fetch records a count above 10 B). A recorded
    value that is not a non-negative int gives False."""
    if track_id != "generalist" or not spec_open_class_active(now):
        return True
    p = s.params
    if p is None:
        return True
    if isinstance(p, bool) or not isinstance(p, int) or p < 0:
        return False
    if p <= KEEPALIVE_BIG_PARAMS:
        return True
    if _v51_scored(s, now):
        return True     # critic F2: only the v4-scale base-floor comparison is skipped (the entry beat its base on S_v5.1)
    table = KEEPALIVE_BASE_FLOOR if base_floors is None else base_floors
    floor = (table.get(s.base_id) or {}).get(track_id) if s.base_id else None
    ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)  # noqa: E731
    return bool(ok(floor) and ok(qual) and qual > floor + KEEPALIVE_BIG_MARGIN)


def _first_pass_ts(s: ST.Submission) -> float | None:
    """When the entry's first pass was taken: ``first_pass_ts``, or for a state written before that field
    ``confirm_after_ts - CONFIRM_DELAY_S`` (exact then: the confirmation was always due one delay after the first pass).
    A ``confirm_after_ts`` pulled in before a track close (``M.confirm_due_ts``) no longer says when the first pass was."""
    if isinstance(s.first_pass_ts, (int, float)) and not isinstance(s.first_pass_ts, bool):
        return float(s.first_pass_ts)
    c = s.confirm_after_ts
    return c - M.CONFIRM_DELAY_S if isinstance(c, (int, float)) else None


def _keepalive_since(s: ST.Submission) -> float | None:
    """When the entry's keep-alive window started: crowning for a champion, the first pass for a pending one. The first
    pass is read from ``first_pass_ts`` (set once, at the first pass; an older state: ``confirm_after_ts`` - ``CONFIRM_DELAY_S``), not from
    ``last_eval_ts``, which a re-fetch at every confirmation attempt moves (an entry whose confirmation keeps failing must
    not keep its keep-alive)."""
    if s.status == ST.CHAMPION:
        t = s.champion_since_ts
    else:
        t = _first_pass_ts(s)
    return float(t) if isinstance(t, (int, float)) and math.isfinite(t) else None


def compute_keepalive(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot,
                      floors: Mapping[str, float | None] | None = None,
                      now: float | None = None,
                      nonproc_floors: Mapping[str, float | None] | None = None,
                      legacy_floors: Mapping[str, float | None] | None = None,
                      base_floors: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, list[tuple[int, float]]]:
    """Per track, the keep-alive entries ``[(uid, score), ...]`` the share layer pays the median serving weight each (at most ``KEEPALIVE_MAX_SHARE`` of the track share in total).

    An entry is a submission that passed the copy checks (exact hashes and the near-duplicate check against its base:
    ``delta_checked``, not pending verification, not rejected), whose hotkey still holds its uid and still commits to
    it, and whose latest score on a shared draw is above the track's entry floor (``floors``, default the release
    ``TRACK_ENTRY_FLOOR``), in one of the ``KEEPALIVE_STATUSES``: pending confirmation (first pass taken) for at most
    ``KEEPALIVE_HOURS`` after that first pass, or champion for at most ``KEEPALIVE_HOURS`` after its crowning. A
    missing, non-finite or far-future window start gives nothing (fail closed). Former champions, not-better, queued,
    pending-verification, withdrawn, rejected, never-scored and below-floor entries get nothing. The same draw's score on
    its non-proc items must be above ``nonproc_floors`` (default the release ``KEEPALIVE_NONPROC_FLOOR``); without a
    non-proc score and without per-tier scores (last scored before 1.2.0rc6) its total score must instead be above
    ``legacy_floors`` (default the release ``KEEPALIVE_LEGACY_FLOOR``; a track without one: no slot). In the open class (at
    ``now``) an entry above ``KEEPALIVE_BIG_PARAMS`` parameters must also beat its base's floor in ``base_floors`` (default the
    release ``KEEPALIVE_BASE_FLOOR``) by ``KEEPALIVE_BIG_MARGIN`` (``_big_ok``; fail closed). Whether a champion's
    award is approved is not visible here: ``combine_weights`` drops an approved champion's slot. At most
    ``MAX_OPEN_PER_COLDKEY_TRACK`` entries per coldkey and track (by status priority, then commitment order); of what is
    left, at most ``KEEPALIVE_MAX_ENTRIES`` non-champion entries per track with any champion listed on top (``combine_weights``
    pays at most ``KEEPALIVE_MAX_ENTRIES``), the highest qualifying score first (the non-proc score, the total
    for a legacy entry; ties by earlier commitment block, then uid): the result is in that rank order. Pure, like
    ``compute_leaders``."""
    now = time.time() if now is None else now
    live = {n.hotkey: n for n in snapshot.neurons}
    out: dict[str, list[tuple[int, float]]] = {}
    for track_id in track_ids:
        floor = floors.get(track_id) if floors is not None else TRACK_ENTRY_FLOOR.get(track_id)
        np_floor = (nonproc_floors if nonproc_floors is not None else KEEPALIVE_NONPROC_FLOOR).get(track_id)
        legacy_floor = (legacy_floors if legacy_floors is not None else KEEPALIVE_LEGACY_FLOOR).get(track_id)
        ts = state.track(track_id)
        # one window per model: a pending entry whose tensors were already committed (withdrawn and committed again, under
        # the same or a new hotkey of the coldkey) counts from the model's earliest first pass in this track
        first_pass: dict[str, float] = {}
        for s in ts.submissions.values():
            c = _first_pass_ts(s)
            if s.hash_named and c is not None and math.isfinite(c):
                first_pass[s.hash_named] = min(first_pass.get(s.hash_named, math.inf), c)
        cands = []
        # opening co-champions (GENERALIST_CO_CHAMPIONS): treated like the champion, i.e. status priority 0 and the window
        # counted from the crowning of the opening playoff; combine_weights drops an approved member (paid by its third)
        co_keys = {m.key for m in cochamp_members(ts)}
        co_since = cochamp_since(ts)
        for s in ts.submissions.values():
            if s.status not in KEEPALIVE_STATUSES or not s.delta_checked or RV.under_review(s):
                continue    # rc11.post5: an entry under review (reparametrisation flag, operator hold) gets no keep-alive
            is_co = s.key in co_keys and s.status != ST.CHAMPION and co_since is not None
            since = co_since if is_co else _keepalive_since(s)
            if since is not None and s.status != ST.CHAMPION and not is_co and s.hash_named in first_pass:
                since = min(since, first_pass[s.hash_named])
            # fail closed on a timestamp more than an hour in the future (a clock that jumped): it would never expire
            if since is None or now - since > KEEPALIVE_HOURS * 3600.0 or since > now + KEEPALIVE_FUTURE_S:
                continue
            score = s.score
            v51 = _v51_scored(s, now)
            if not isinstance(score, (int, float)) or not math.isfinite(score) or (score <= 0 and not v51) \
                    or (floor is not None and score <= floor and not v51):
                continue
            nonproc = s.nonproc
            qual = _v51_keepalive_margin(s) if v51 else float(score)   # compare v5 entries by their deciding margin, not raw DI score
            if v51:
                pass      # S_v5.1: the v4-scale floors do not apply
            elif nonproc is None and not s.by_tier:
                # scored before 1.2.0rc6: no non-proc score; the total must be out of a rules-only model's reach
                if legacy_floor is None or score <= legacy_floor:
                    continue
            elif not isinstance(nonproc, (int, float)) or not math.isfinite(nonproc) \
                    or (np_floor is not None and nonproc <= np_floor):
                continue
            else:
                qual = float(nonproc)
            if not _big_ok(s, track_id, now, base_floors, qual):
                continue
            n = live.get(s.hotkey)
            c = parse_commitment(n.commitment) if n is not None else None
            if c is None or f"{s.hotkey}|{c.text()}" != s.key:
                continue
            # the chain's commitment block, never the local first_seen_block (it depends on when this evaluator noticed
            # the commitment); unknown (0/None) sorts after the known ones, by uid only
            cb = n.commitment_block if isinstance(n.commitment_block, int) and n.commitment_block > 0 else sys.maxsize
            cands.append((0 if is_co else KEEPALIVE_STATUSES.index(s.status), cb, n.uid, s.key, n, float(score), qual))
        per_coldkey: dict[str, int] = {}
        res: list[tuple[int, float]] = []
        rank: dict[int, tuple[float, ...]] = {}
        # rc10.post5 (CoS #23473, 22 Oct 2026): on the vacant generalist track the opening-cohort entries (first seen before
        # GENERALIST_START_UTC) rank before post-opening entries for the KEEPALIVE_MAX_ENTRIES cut, so no cohort entry is
        # pruned while it waits for its draws and the playoff. Other tracks and a crowned track: unchanged (0 for all).
        vacant_gen = track_id == "generalist" and not ts.champion
        start_ts = split.GENERALIST_START_UTC.timestamp()
        cohort_key = {s.key: (0 if not vacant_gen or s.first_seen_ts < start_ts else 1) for s in ts.submissions.values()}
        for *head, n, score, qual in sorted(cands, key=lambda c: c[:4]):
            if per_coldkey.get(n.coldkey, 0) >= C.MAX_OPEN_PER_COLDKEY_TRACK or any(u == n.uid for u, _ in res):
                continue
            per_coldkey[n.coldkey] = per_coldkey.get(n.coldkey, 0) + 1
            res.append((n.uid, score))
            # a crowned champion waiting for the owner's approval always keeps its slot (status first), then the score
            rank[n.uid] = (head[0], cohort_key.get(head[3], 1), -qual, head[1], n.uid)
        champ_n = sum(1 for u, _ in res if rank[u][0] == 0)
        if len(res) > KEEPALIVE_MAX_ENTRIES:
            # the 2-per-coldkey rule above came first; now only the strongest entries keep waiting alive: a champion
            # first, then (vacant generalist track only) the opening cohort, then the highest qualifying score, ties by earlier on-chain commitment block (unknown: uid only), then uid. The rest get nothing.
            # rc8 (review-rc8 F2): the champion is listed on top of the 8 pending entries, not inside them; combine_weights
            # drops an approved champion (paid by its rank) and then pays at most KEEPALIVE_MAX_ENTRIES in this order, so an
            # approved champion leaves 8 pending entries paid and a champion awaiting approval keeps its slot first.
            res.sort(key=lambda r: rank[r[0]])
            # rc10.post5 (Florian via CoS #23485): every qualifying opening-cohort entry of the vacant generalist track is
            # listed, even beyond KEEPALIVE_MAX_ENTRIES (they rank first, see cohort_key); post-opening entries fill the rest
            cohort_n = sum(1 for u, _ in res if rank[u][0] != 0 and rank[u][1] == 0) if vacant_gen else 0
            keep = champ_n + max(KEEPALIVE_MAX_ENTRIES, cohort_n)
            if len(res) > keep:
                log.info("track %s: %d entries qualify for keep-alive, only the top %d pending entries (plus a champion) are "
                         "listed; no keep-alive for uids %s", track_id, len(res), KEEPALIVE_MAX_ENTRIES, [u for u, _ in res[keep:]])
            res = res[:keep]
        out[track_id] = res
    return out


def keepalive_protected(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot) -> dict[str, list[int]]:
    """rc10.post5 (Florian via CoS #23485, 22 Oct 2026): per track, the uids whose keep-alive slot ``combine_weights``
    never cuts: the open opening-cohort entries (first seen before ``GENERALIST_START_UTC``, status in
    ``KEEPALIVE_STATUSES``) of the generalist track while it has no champion, so none is pruned before the playoff. Only
    entries that qualify in ``compute_keepalive`` are paid at all; this list only exempts them from the slot cut. Other
    tracks and a crowned generalist track: none. A record whose hotkey no longer holds its uid and commits to it in
    ``snapshot`` protects nothing (critic F1). Pure."""
    out: dict[str, list[int]] = {}
    live = {n.hotkey: n for n in snapshot.neurons}
    start_ts = split.GENERALIST_START_UTC.timestamp()
    for track_id in track_ids:
        if track_id != "generalist":
            continue
        ts = state.track(track_id)
        if ts.champion:
            continue
        uids: set[int] = set()
        for s in ts.submissions.values():
            if s.status not in KEEPALIVE_STATUSES or s.status == ST.CHAMPION or not s.first_seen_ts < start_ts or RV.under_review(s):
                continue
            n = live.get(s.hotkey)
            c = parse_commitment(n.commitment) if n is not None else None
            if c is not None and n.uid == s.uid and f"{s.hotkey}|{c.text()}" == s.key:
                uids.add(n.uid)
        uids = sorted(uids)
        if uids:
            out[track_id] = uids
    return out


# First-pass queue order (mechanism v2): the chain prunes the lowest-emission non-immune uid once a hotkey's immunity has
# run out, and a queued entry earns no keep-alive, so the entry whose immunity ends soonest is evaluated first. The code
# does not read the chain's ImmunityPeriod hyperparameter; this is the subnet's value as a release constant (7200 blocks,
# about 24 h, since block ~9,305,000 on 17 Oct 2026; it was 5000 before, and the 5000 left the provisional keep-alive
# window mostly inside the real immunity). Entries whose registration block is unknown (0) keep the plain commitment
# order, after the others.
IMMUNITY_BLOCKS_ASSUMED = CHAIN_IMMUNITY_BLOCKS   # 7200


def queue_order(cands: Sequence[ST.Submission], snap: Snapshot) -> list[ST.Submission]:
    """``cands`` (already in commitment order) reordered: entries already past their immunity first, then those with
    immunity left, the soonest end first; then those with an unknown registration block. Inside each group the given
    order is kept (the sort is stable), so entries past immunity keep today's relative order."""
    reg = {n.hotkey: n.registered_block for n in snap.neurons}

    def key(s: ST.Submission) -> tuple[int, int]:
        r = reg.get(s.hotkey, 0)
        if not isinstance(r, int) or r <= 0:
            return (2, 0)
        end = r + IMMUNITY_BLOCKS_ASSUMED
        return (0, 0) if end <= snap.block else (1, end)

    return sorted(cands, key=key)


WAITING_FOR_REPO = "waiting for the repo"    # ``Submission.reason`` after a fetch attempt that did not find the repo
MAX_SKIPS_PER_CYCLE = 4    # unusable queue entries (unfetchable, rejected, held) passed over per track and cycle (rc8.post1)


# Rule 17 Oct 2026 (carry-over of a pruned entry's first draw): entries pruned before the rule existed (no withdrawn_ts)
# have 72 h from the decision, 17 Oct 2026 18:25 UTC -> 20 Oct 2026 18:25 UTC.
CARRYOVER_RETRO_END_TS = 1792520700
CARRYOVER_WINDOW_S = 72 * 3600
CARRYOVER_DRAW_GRACE_S = 72 * 3600   # a recommit seen inside the window may be fetched later (repo not public yet, queue)


def _carryover_waiting(s: ST.Submission) -> bool:
    """A withdrawn entry that passed draw 1 but never completed confirmation, including old state."""
    return (s.status == ST.WITHDRAWN and not s.carried_to
            and s.withdrawn_status in ("", ST.PENDING_CONFIRM)
            and s.confirm_after_ts is not None and bool(s.draw1) and s.first_gain is not None
            and not s.confirm_draw and s.confirm_score is None)


def _pruned_now(old: ST.Submission, registered: Mapping[str, int] | set[str]) -> bool:
    """The withdrawn entry's hotkey left the subnet: it is not registered now, or it registered again after the entry was
    first seen (pruned and re-registered on a new uid). A hotkey registered all along withdrew its commitment."""
    if old.hotkey not in registered:
        return True
    reg = registered.get(old.hotkey) if isinstance(registered, Mapping) else None
    return isinstance(reg, int) and reg > old.first_seen_block


def carryover_eligible(old: ST.Submission, new: ST.Submission, registered: Mapping[str, int] | set[str],
                       at: float) -> bool:
    """Pure eligibility check, after the new entry has passed the fetch, rule and copy checks. ``registered``: the
    snapshot's hotkeys (hotkey -> registration block); ``at``: when the recommit was first seen (critic B4: the 72 h
    count to the commitment, not to the later fetch)."""
    if not _carryover_waiting(old) or old.track_id != new.track_id:
        return False
    if not new.hash_named or old.hash_named != new.hash_named:
        return False
    if old.coldkey != new.coldkey and old.repo != new.repo:
        return False
    if old.withdrawn_kind != "pruned" and (old.withdrawn_kind or not _pruned_now(old, registered)):
        return False
    if reentry_match(old, new):    # a re-entry (#23211) seen by its deadline is never cut by the 72 h window
        return at <= max(_carryover_end(old), REENTRY_DEADLINE_TS)
    return at <= _carryover_end(old)


# Re-entry (Florian 21 Oct 2026, board #23203/#23204/#23211): a pruned opening-cohort entry of the vacant generalist track
# that passed its first draw keeps its opening-cohort place when the SAME repo@revision (full 40 hex) is committed again
# by the same hotkey or by a hotkey of the same coldkey, first seen by 22 Oct 2026 07:30:00 UTC. The opening playoff waits
# for such entries until then, and afterwards for a re-entry seen by then that is still open. A UID number proves nothing.
REENTRY_DEADLINE_TS = 1792654200.0   # 2026-10-22T07:30:00Z


def reentry_match(old: ST.Submission, new: ST.Submission) -> bool:
    """``new`` re-enters the withdrawn ``old`` under the published terms (same repo@revision, same hotkey or coldkey)."""
    return (old.track_id == new.track_id == "generalist" and old.repo == new.repo and len(new.revision) == 40
            and old.revision == new.revision and new.key != old.key
            and (old.hotkey == new.hotkey or bool(old.coldkey) and old.coldkey == new.coldkey)
            and new.first_seen_ts <= REENTRY_DEADLINE_TS)


def _carryover_end(old: ST.Submission) -> float:
    return old.withdrawn_ts + CARRYOVER_WINDOW_S if old.withdrawn_ts is not None else CARRYOVER_RETRO_END_TS


def _first_provisional_ts(state: ST.SpecState) -> dict[str, float]:
    """Per hotkey, the earliest ``provisional_since_ts`` over all its submissions in all tracks. The 12 hours count from
    it, so a new commitment (a recommit, another track) never restarts the window of a hotkey."""
    out: dict[str, float] = {}
    for ts in state.tracks.values():
        for s in ts.submissions.values():
            v = s.provisional_since_ts
            if isinstance(v, (int, float)) and math.isfinite(v) and v < out.get(s.hotkey, math.inf):
                out[s.hotkey] = v
    return out


BLOCK_SECONDS = 12.0   # chain block time, for the age of an on-chain commitment (D4)


def _commitment_ts(n: Any, block: int, now: float) -> float:
    """When the neuron's commitment was made on chain, from its commitment block (``block`` is the snapshot's block, one
    block = ``BLOCK_SECONDS``); ``now`` when the block is unknown (0/None) or lies in the future. Never later than ``now``."""
    cb = getattr(n, "commitment_block", 0)
    if not isinstance(cb, int) or isinstance(cb, bool) or cb <= 0 or cb > block:
        return now
    return now - (block - cb) * BLOCK_SECONDS


def _provisional_ok(s: ST.Submission, live: Mapping[str, Any], block: int, open_for_coldkey: int = 0,
                    from_commitment: bool = False) -> Any:
    """The live neuron if ``s`` meets every provisional keep-alive condition except the 12-hour window and the
    per-coldkey limit, else None. The entry waits for its first evaluation and was never scored. Two shapes:
    - not fetched yet (QUEUED, ``hash_named`` empty: it waits for the evaluation budget; the queue fetches the entries
      whose immunity ran out first). Nothing of it was checked yet, so the rule is: it is still QUEUED (a rejected or
      not-better key is never eligible, a failed check turns it REJECTED at once), its hotkey still holds exactly
      this commitment, which parses as a valid commitment, and its coldkey is within the open-entry limit that is
      enforced when the entry is queued;
    - fetched (``hash_named``: the fetch, the rule checks and the exact tensor-hash check passed, else the entry would
      have been rejected): QUEUED and copy-checked (``delta_checked``), or held for this host's infrastructure.
    Either way the hotkey's immunity ends within ``PROVISIONAL_KA_LEAD_BLOCKS`` or is over, registration block known.
    ``from_commitment`` (D4, eval v5 gate): the slot starts at the on-chain commitment, so the immunity lead is not asked for."""
    floor = TRACK_ENTRY_FLOOR.get("generalist", math.inf)
    if s.track_id == "generalist" and not from_commitment and (
            not isinstance(floor, (int, float)) or isinstance(floor, bool) or not math.isfinite(floor)
            or split.GENERALIST_G_MED is None):
        return None
    if s.score is not None:
        # a close call that waits for its extra draw (QUEUED with ``close_draws``) is still undecided: it keeps the slot
        # until the verdict (critic reg-sniping 20 Oct: uid 85 earned nothing between its first draw and its verdict)
        if not (s.status == ST.QUEUED and s.close_draws and s.hash_named and s.delta_checked):
            return None
    elif not s.hash_named:
        if s.status != ST.QUEUED or open_for_coldkey > C.MAX_OPEN_PER_COLDKEY_TRACK:
            return None
        if s.reason.startswith(WAITING_FOR_REPO):
            return None    # the last fetch attempt found no repo (missing or private): nothing to keep alive yet
    elif s.status == ST.QUEUED:
        if not s.delta_checked:
            return None
    elif s.status != ST.PENDING_VERIFY or s.wait_kind != ST.WAIT_INFRA:
        return None
    n = live.get(s.hotkey)
    c = parse_commitment(n.commitment) if n is not None else None
    if c is None or f"{s.hotkey}|{c.text()}" != s.key:
        return None
    if from_commitment:
        return n
    r = n.registered_block
    if not isinstance(r, int) or r <= 0 or block < r + IMMUNITY_BLOCKS_ASSUMED - PROVISIONAL_KA_LEAD_BLOCKS:
        return None
    return n


def _open_counts(state: ST.SpecState, track_id: str) -> dict[str, int]:
    """Open entries (queued or waiting) per coldkey in one track."""
    out: dict[str, int] = {}
    for s in state.track(track_id).submissions.values():
        if s.status in (ST.QUEUED, ST.PENDING_CONFIRM, ST.PENDING_VERIFY):
            out[s.coldkey] = out.get(s.coldkey, 0) + 1
    return out


def mark_provisional(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot, now: float,
                     probed: Collection[str] = ()) -> bool:
    """Persist the moment each entry first qualified for the provisional keep-alive (``provisional_since_ts``, set once;
    the 12 hours of ``compute_provisional`` count from it). Run by whoever owns the state file (the evaluator). True if
    anything was set, i.e. the state must be saved. ``probed`` (D4, critic F5): keys of the not yet fetched entries whose repo
    a probe found (exists, public, revision as committed, safetensors listed); under the eval v5 gate such an entry gets its
    window only then, so a missing or dummy repo earns no slot."""
    if not PROVISIONAL_KA:
        return False
    live = {n.hotkey: n for n in snapshot.neurons}
    first = _first_provisional_ts(state)
    changed = False
    d4 = S.v5_active(now)    # D4: from the eval v5 gate on the half keep-alive runs from the on-chain commitment
    for track_id in track_ids:
        opened = _open_counts(state, track_id)
        for s in state.track(track_id).submissions.values():
            if (s.provisional_since_ts is None and _provisional_ok(s, live, snapshot.block, opened.get(s.coldkey, 0), d4) is not None
                    and (not d4 or s.hash_named or s.key in probed) and _big_ok(s, track_id, now)):
                # the window belongs to the hotkey: a later entry of it inherits the hotkey's earliest start
                s.provisional_since_ts = first.setdefault(s.hotkey, _commitment_ts(live[s.hotkey], snapshot.block, now) if d4 else now)
                state.track(track_id).log(now, f"{s.key[:12]}: provisional keep-alive window starts")
                changed = True
    return changed


def compute_provisional(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot,
                        now: float | None = None) -> dict[str, list[int]]:
    """Per track, the uids with a provisional keep-alive slot (``combine_weights`` pays them): see
    ``_provisional_ok`` for the conditions, at most ``PROVISIONAL_KA_MAX_HOURS`` after ``provisional_since_ts`` (not set
    or more than an hour in the future: nothing, fail closed), at most one per coldkey and track (earlier on-chain
    commitment block, unknown last, then uid). In the open class an entry with a recorded parameter count above
    ``KEEPALIVE_BIG_PARAMS`` gets none (``_big_ok``; not fetched yet = no count = eligible as before). Pure, like ``compute_keepalive``; never counts against the top-8."""
    out: dict[str, list[int]] = {t: [] for t in track_ids}
    if not PROVISIONAL_KA:
        return out
    now = time.time() if now is None else now
    live = {n.hotkey: n for n in snapshot.neurons}
    first = _first_provisional_ts(state)
    d4 = S.v5_active(now)
    for track_id in track_ids:
        cands = []
        opened = _open_counts(state, track_id)
        for s in state.track(track_id).submissions.values():
            since = s.provisional_since_ts
            if not isinstance(since, (int, float)) or not math.isfinite(since):
                continue
            since = min(since, first.get(s.hotkey, since))      # the hotkey's window, not the entry's
            if now - since > PROVISIONAL_KA_MAX_HOURS * 3600.0 or since > now + KEEPALIVE_FUTURE_S:
                continue
            n = _provisional_ok(s, live, snapshot.block, opened.get(s.coldkey, 0), d4)
            if n is None or not _big_ok(s, track_id, now) or RV.under_review(s):
                continue
            cb = n.commitment_block if isinstance(n.commitment_block, int) and n.commitment_block > 0 else sys.maxsize
            cands.append((cb, n.uid, s.key, n))
        seen: set[str] = set()
        for _, uid, _, n in sorted(cands, key=lambda c: c[:3]):
            if n.coldkey in seen or uid in out[track_id]:
                continue
            seen.add(n.coldkey)
            out[track_id].append(uid)
    return out


def _no_base(sub: ST.Submission, detail: str = "") -> str:
    why = f"declared base model {sub.base_model or '(none)'} is not available on this validator; the near-duplicate check waits for it"
    return f"{why} ({detail})" if detail else why


class _Prepared:
    def __init__(self, sub: ST.Submission, path: str, weight_bytes: int):
        self.sub, self.path, self.weight_bytes = sub, path, weight_bytes
        self.joint = False
        self.preflight_failure = False
        self.model_specific_failure = False
        self.infra_failure = False  # the last _serve failed for a host reason (docker, socket), not the model's
        self.probs: list[Any] = []  # the answers of the last _serve (behavioural near-duplicate check)


class _BaseSub:
    """Stand-in 'submission' of a declared base served on a v5.1 draw (the attributes ``_serve`` reads and writes); it is
    never stored in the state."""

    def __init__(self, base_model: str, track_id: str):
        self.key = V5W.base_key(base_model)
        self.track_id = track_id
        self.last_eval_ts: float | None = None
        self.reason = ""
        self.uid = -1


class SpecEvaluator:
    def __init__(self, cfg: C.SpecConfig, *, fetcher: F.Fetcher | None = None, backend_factory: BackendFactory | None = None,
                 base_provider: T.BaseProvider | None = None, clock: Callable[[], float] = time.time,
                 seed_source: Callable[[str, int], int] | None = None,
                 free_bytes: Callable[[str], int] | None = None, base_fetcher: B.BaseFetcher | None = None):
        """``base_fetcher``: downloads missing declared bases (spec/bases.py). By default only the production setup (no
        injected ``fetcher``, ``base_dir`` set, ``base_fetch`` on) gets one; tests pass their own with a fake hub."""
        if S.EVAL_V5_START_UTC is not None:   # eval v5: the draw and the offline scorer exist, the serving/decision wiring does not
            from . import v5
            if not v5.EVALUATOR_WIRED:
                raise RuntimeError("EVAL_V5_START_UTC is set but the evaluator is not wired for eval v5 (v5.EVALUATOR_WIRED)")
        self.cfg = cfg
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self.fetcher = fetcher or F.HfFetcher(min_free_bytes=cfg.min_free_bytes, free_bytes=self._free_bytes)
        self.backend_factory = backend_factory or self._docker_backend
        self.base_provider = base_provider or C.dir_base_provider(cfg.base_dir)
        if base_fetcher is None and fetcher is None and cfg.base_dir and cfg.base_fetch:
            base_fetcher = B.BaseFetcher.from_config(cfg)
        self.base_fetcher = base_fetcher
        if base_fetcher is not None:
            base_fetcher.in_use = self._bases_in_use
        self.clock = clock
        self.seed_source = seed_source or (lambda track, block: int.from_bytes(
            hashlib.sha256(f"{track}|{block}|{os.urandom(16).hex()}".encode()).digest()[:8], "big"))
        self.state = ST.load(cfg.state_path)
        self._lock = asyncio.Lock()
        self._pools: dict[str, list[dict]] = {}
        self._playoff_waits: set[str] = set()   # tracks whose cohort playoff waited for budget in the last cycle
        self._cycle_prepared: dict[str, _Prepared | None] = {}
        self._cycle_skips: dict[str, int] = {}
        self._cycle_fetch_rejections: dict[str, int] = {}
        self._probes_halted = False    # a hub infrastructure error (429, 5xx, timeout) ends this cycle's probing
        self._retry_tracks: set[str] = set()
        self._close_noted: set[str] = set()     # tracks whose close with pending confirmations was logged (this process)
        self._host_failed: set[str] = set()
        self._disk_wait = False                  # a download did not fit the free-space floor: no further download this cycle
        self._oversize_used = False
        self._gmed_warned: set[str] = set()      # tracks whose missing G_med was logged in this cycle (binding mode)
        self._track_cap: int | None = None       # this track's share of the cycle's budget (None = all that is left)
        self._base_ids: dict[tuple, str] = {}    # (base dir, its files and sizes) -> tensor hash of the base model
        self._seed = T.load_sketch_seed(cfg.work_dir)   # secret sample positions of the delta sketch; never logged
        self._seed_id = T.seed_id(self._seed)
        self._near_cache: dict[tuple[str, str, str], T.DeltaStats | None] = {}   # near-champion rule: (incumbent, challenger, seed id)
        self._review_seen: set[tuple[str, str]] = set()   # rc11.post5: (key, why) of entries under review already logged
        self._d4_probed: set[str] = set()            # D4: keys of unfetched entries whose repo a probe found (in memory only)
        self._d4_probe_failed: dict[str, float] = {}  # key -> when its probe failed (retried after ``RETRY_UNAVAILABLE_S``)
        self._v5_ctx: V5W.DrawContext | None = None       # eval v5.1: the numbers of the draw being decided (``_decide`` reads them)
        self._slot_ctx: dict[str, V5W.DrawContext] | None = None   # a slot on eval v5.1 (rc11.post4): each round's numbers, by draw id
        self._v5_served: dict[tuple[str, str], tuple[list, list, bool]] = {}   # (model key, sha(draw id + sorted row ids)) -> (item scores, answers, joint flag); a base shared by entries is served once
        self._v5_pools: tuple[list[dict], list[dict]] | None = None     # (T/N rows, public rows), per cycle
        self._v5_config: tuple | None = None    # (``V5W.load_config`` result,), per cycle: v51.config.json is read once a cycle
        os.makedirs(os.path.join(cfg.work_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(cfg.work_dir, "sketches"), exist_ok=True)

    # ------------------------------------------------------------------ public

    async def evaluate_cycle(self, snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        if self.base_fetcher is not None:
            self.base_fetcher.start_preseed(self.cfg.base_preseed)    # once, in the background
        async with self._lock:
            now = self.clock()
            if self.base_fetcher is not None:
                self.base_fetcher.new_cycle()
            warning = self.cfg.budget_warning(now)
            if warning:
                log.warning(warning)
            self._disk_wait = False
            active = self.cfg.active_tracks(now)
            self._pools.clear()
            self._v5_pools = None
            self._v5_config = None
            self._slot_ctx = None
            self._v5_served.clear()    # critic F4: the served results of a draw are never carried into another cycle
            self._cycle_prepared.clear()
            self._cycle_skips.clear()
            self._cycle_fetch_rejections.clear()
            self._probes_halted = False
            self._retry_tracks.clear()
            self._oversize_used = False
            self._gmed_warned.clear()
            for tdef in self.cfg.tracks:
                if tdef.track_id == "generalist" and (split.GENERALIST_COMMIT_FROM_UTC is None
                        or now < split.GENERALIST_COMMIT_FROM_UTC.timestamp()):
                    continue
                if tdef in active or (tdef.track_id == "generalist" and now < tdef.start_ts):
                    self._sync(tdef, snapshot, now)
            for tdef in active:
                try:
                    await self._retry_waiting(tdef, snapshot, now)
                except Exception:  # noqa: BLE001 - a failed metadata probe must not stop other tracks
                    log.exception("track %s: repo availability probes failed", tdef.track_id)
            self._confirm_before_close(active, now)
            # start the windows of the entries that qualify now, before a long evaluation cycle can eat their lead
            await self._d4_probe([t.track_id for t in active], snapshot, self.clock())
            mark_provisional(self.state, [t.track_id for t in active], snapshot, self.clock(), self._d4_probed)
            ST.save(self.state, self.cfg.state_path)
            # the hourly budget is shared: the track evaluated least recently goes first (1.2.0rc6; before, the first track
            # in the schedule always went first and could use the whole budget while the other waited)
            # a cohort playoff that waits for a full hour's budget goes first and holds the budget for itself (review G1):
            # the other tracks do not take the slots it waits for, so the wait ends when the rolling hour has emptied
            order = self._track_order(active)
            self._playoff_waits = set()
            self._host_failed = set()
            failed_tracks: set[str] = set()
            limited_tracks: set[str] = set()
            for i, tdef in enumerate(order):
                # fair share (1.2.0rc8.post1): while a later track has work, a track takes at most its share of what is
                # left of the hour's budget (at least 2: the champion and one entry). Before, both tracks served with
                # the same cycle timestamp tied, the first in the schedule took the whole budget and the other track
                # got a turn only every second cycle (RA first pass stalled behind its confirmations, #20923)
                waiting = sum(1 for t in order[i + 1:] if self._has_work(t, now))
                self._track_cap = None if not waiting else max(2, math.ceil(self._budget(now) / (waiting + 1)))
                if self._track_cap is not None:
                    limited_tracks.add(tdef.track_id)
                try:
                    await self._run_track(tdef, snapshot, now)
                except Exception:  # noqa: BLE001 - one failing track must not stop the others
                    failed_tracks.add(tdef.track_id)
                    log.exception("track %s: evaluation failed", tdef.track_id)
                await self._d4_probe([t.track_id for t in active], snapshot, self.clock())
                mark_provisional(self.state, [t.track_id for t in active], snapshot, self.clock(), self._d4_probed)
                ST.save(self.state, self.cfg.state_path)
                if tdef.track_id in self._playoff_waits:
                    log.info("track %s: the other tracks wait this cycle (the cohort playoff holds the budget)",
                             tdef.track_id)
                    break
            # Redistribute unused shares to tracks limited on the first pass. Tracks with no cap already had
            # all available slots; do not advance a second confirmation group on those tracks in this cycle.
            # The same preparation cache and skip allowance apply to both passes.
            self._track_cap = None
            if not self._playoff_waits:
                for tdef in order:
                    if self._budget(now) < 1:
                        break
                    if (tdef.track_id not in limited_tracks or tdef.track_id in failed_tracks
                            or tdef.track_id in self._host_failed
                            or not self._has_work(tdef, now)):
                        continue
                    try:
                        await self._run_track(tdef, snapshot, now)
                    except Exception:  # noqa: BLE001 - keep failures local to a track
                        log.exception("track %s: evaluation failed", tdef.track_id)
                    ST.save(self.state, self.cfg.state_path)
                    if self._playoff_waits:
                        break
            await self._d4_probe([t.track_id for t in active], snapshot, self.clock())
            mark_provisional(self.state, [t.track_id for t in active], snapshot, self.clock(), self._d4_probed)
            ST.save(self.state, self.cfg.state_path)
            self._cleanup(active)
            return self.leaders(active, snapshot)

    async def _d4_probe(self, track_ids: Sequence[str], snapshot: Snapshot, now: float) -> None:
        """D4 (critic F5): before an unfetched entry gets the half keep-alive from its on-chain commitment, its repo is probed:
        it exists, is public, the committed revision is there and a ``.safetensors`` file is listed (what ``WAITING_FOR_REPO``
        waits for). Metadata only, at most ``WAIT_REPO_PROBES_PER_CYCLE`` per track and cycle, a failed one again after
        ``RETRY_UNAVAILABLE_S``. Gate off: nothing."""
        if not (PROVISIONAL_KA and S.v5_active(now)):
            return
        live = {n.hotkey: n for n in snapshot.neurons}
        for track_id in track_ids:
            opened = _open_counts(self.state, track_id)
            todo = sorted((s for s in self.state.track(track_id).submissions.values()
                           if s.provisional_since_ts is None and not s.hash_named and s.key not in self._d4_probed
                           and now - self._d4_probe_failed.get(s.key, -math.inf) >= C.RETRY_UNAVAILABLE_S
                           and _provisional_ok(s, live, snapshot.block, opened.get(s.coldkey, 0), True) is not None
                           and _big_ok(s, track_id, now)),
                          key=lambda s: (s.first_seen_block, s.uid, s.key))
            for s in todo[:C.WAIT_REPO_PROBES_PER_CYCLE]:
                commitment = SpecCommitment(s.track_id, s.repo, s.revision)
                try:
                    listing = getattr(self.fetcher, "probe_listing", None)
                    if listing is not None:
                        files = (await listing(commitment)).files
                        if not any(n.endswith(".safetensors") and sz > 0 for n, sz in files.items()):
                            raise F.FetchError("rejected", "no safetensors weights listed")
                    else:
                        await self.fetcher.probe(commitment)
                except F.FetchError as exc:
                    if exc.kind == "infra":
                        return      # the hub is struggling: try again next cycle, nothing is concluded
                    self._d4_probe_failed[s.key] = now
                    continue
                except Exception:   # noqa: BLE001 - a probe bug must not stop the cycle; no slot then
                    log.exception("D4 repo probe of %s failed", s.key[:12])
                    self._d4_probe_failed[s.key] = now
                    continue
                self._d4_probed.add(s.key)

    def leaders(self, active: Sequence[C.TrackDef], snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        return compute_leaders(self.state, [t.track_id for t in active], snapshot, now=self.clock())

    def keepalive(self, active: Sequence[C.TrackDef], snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        return compute_keepalive(self.state, [t.track_id for t in active], snapshot,
                                 {t.track_id: self._entry_floor(t) for t in active}, now=self.clock())

    def protected(self, active: Sequence[C.TrackDef], snapshot: Snapshot) -> dict[str, list[int]]:
        return keepalive_protected(self.state, [t.track_id for t in active], snapshot)

    def legacy(self, snapshot: Snapshot) -> dict[str, list[tuple[int, bool]]]:
        """Leaders of the ended DF/RA tracks from this evaluator's state (``legacy_leaders``); the window is checked by
        the share layer."""
        return legacy_leaders(self.state, split.LEGACY_KA_TRACKS, snapshot)

    def provisional(self, active: Sequence[C.TrackDef], snapshot: Snapshot) -> dict[str, list[int]]:
        ids = [t.track_id for t in active]
        now = self.clock()
        if mark_provisional(self.state, ids, snapshot, now, self._d4_probed):
            ST.save(self.state, self.cfg.state_path)
        return compute_provisional(self.state, ids, snapshot, now=now)

    # --------------------------------------------------------------- chain sync

    def _sync(self, tdef: C.TrackDef, snap: Snapshot, now: float) -> None:
        ts = self.state.track(tdef.track_id)
        live: dict[str, tuple[Any, SpecCommitment]] = {}
        for n in snap.neurons:
            c = parse_commitment(n.commitment)
            if c is not None and c.track_id == tdef.track_id:
                live[f"{n.hotkey}|{c.text()}"] = (n, c)
        for key, (n, c) in live.items():
            sub = ts.submissions.get(key)
            if sub is not None and sub.status != ST.WITHDRAWN:
                sub.uid, sub.coldkey = n.uid, n.coldkey
                continue
            if sub is not None:
                # the same hotkey holds this commitment again (re-registered after a deregistration, or committed back):
                # a new entry in a later cohort; a pruned first pass may be restored after the copy checks.
                # The withdrawn one stays on record under its own key and keeps its hashes (the copy check decides).
                old_key = f"{key}#withdrawn@{sub.first_seen_block}"
                sub.key = old_key
                ts.submissions[old_key] = ts.submissions.pop(key)
                for entry in ts.submissions.values():
                    if entry.carried_from == key:
                        entry.carried_from = old_key
                    if entry.carried_to == key:
                        entry.carried_to = old_key
                for h, owner_key in list(ts.hashes.items()):
                    if owner_key == key:
                        ts.hashes[h] = old_key
                ts.log(now, f"{key[:12]}: committed again after its withdrawal (uid {n.uid}), a new entry")
            sub = ST.Submission(key=key, uid=n.uid, hotkey=n.hotkey, coldkey=n.coldkey, track_id=tdef.track_id, repo=c.repo,
                                revision=c.revision, first_seen_block=snap.block, first_seen_ts=now)
            ts.submissions[key] = sub
            ts.log(now, f"new submission uid {n.uid} {c.repo}@{c.revision[:8]}")
            why = self._budget_violation(sub, now) if tdef.track_id != "generalist" else ""
            if why:
                sub.status, sub.reason = ST.REJECTED, why
        for key, sub in ts.submissions.items():
            if key in live or sub.status in (ST.WITHDRAWN, ST.REJECTED, ST.NOT_BETTER):
                continue
            if sub.status == ST.FORMER:
                continue    # stays listed; leaders() checks that the hotkey still holds it
            if ts.champion == key:
                ts.champion = ""    # a shared group's leader: the group stays the champion (``_reanchor_cochamps`` below)
            sub.withdrawn_ts, sub.withdrawn_status = now, sub.status
            sub.withdrawn_kind = ("pruned" if _pruned_now(sub, {n.hotkey: n.registered_block for n in snap.neurons})
                                  else "changed")
            sub.status, sub.reason = ST.WITHDRAWN, "commitment changed or hotkey left"
            ts.log(now, f"withdrawn {key[:12]}")
        if tdef.track_id == "generalist" and now >= tdef.start_ts:
            # Admit pre-commitments only at opening, in observed order. Unadmitted peers cannot
            # consume each other's budget; pre-opening commitments never consume DF/RA's budget.
            for sub in sorted(ts.submissions.values(), key=lambda s: (s.first_seen_block, s.first_seen_ts, s.key)):
                if sub.status == ST.QUEUED and not sub.generalist_budget_checked:
                    why = self._budget_violation(sub, now)
                    sub.generalist_budget_checked = True
                    if why:
                        sub.status, sub.reason = ST.REJECTED, why
        if ts.champion and ts.submissions[ts.champion].status != ST.CHAMPION:
            ts.champion = ""
        self._reanchor_cochamps(ts, now)

    def _budget_violation(self, sub: ST.Submission, now: float) -> str:
        """docs/SPECIALIST-TRACKS.md: at most 2 open submissions per coldkey and track, 4 new per coldkey per 7 days."""
        generalist = sub.track_id == "generalist"
        if generalist and now < split.GENERALIST_START_UTC.timestamp():
            return ""
        ts = self.state.track(sub.track_id)
        open_ = [s for s in ts.submissions.values()
                 if s is not sub and s.coldkey == sub.coldkey
                 and (not generalist or s.generalist_budget_checked)
                 and s.status in (ST.QUEUED, ST.PENDING_CONFIRM, ST.PENDING_VERIFY)]
        if len(open_) >= C.MAX_OPEN_PER_COLDKEY_TRACK:
            return f"coldkey already has {C.MAX_OPEN_PER_COLDKEY_TRACK} open submissions in this track"
        week = [s for t in self.state.tracks.values() for s in t.submissions.values()
                if s is not sub and s.coldkey == sub.coldkey and s.status != ST.REJECTED
                and not (s.status == ST.WITHDRAWN and reentry_match(s, sub))   # re-entry (#23211): not a new submission
                and (s.track_id == "generalist") == generalist
                and (not generalist or s.generalist_budget_checked)
                and (max(s.first_seen_ts, split.GENERALIST_START_UTC.timestamp()) if generalist
                     else s.first_seen_ts) > now - 7 * 86400]
        if len(week) >= C.MAX_NEW_PER_COLDKEY_WEEK:
            return f"coldkey already made {C.MAX_NEW_PER_COLDKEY_WEEK} submissions in 7 days"
        return ""

    # ------------------------------------------------------------------- helpers

    def _model_dir(self, sub: ST.Submission) -> str:
        return os.path.join(self.cfg.work_dir, "models", hashlib.sha256(sub.key.encode()).hexdigest()[:20])

    def _sketch_path(self, sub: ST.Submission) -> str:
        return os.path.join(self.cfg.work_dir, "sketches", hashlib.sha256(sub.key.encode()).hexdigest()[:20] + ".npz")

    def _docker_backend(self, model_dir: str, weight_bytes: int) -> AsyncContextManager[Backend]:
        return VllmDockerBackend(model_dir, weight_bytes, gpu_device=self.cfg.gpu_device, gpu_total_gb=self.cfg.gpu_total_gb,
                                 budget_gb=self.cfg.gpu_budget_gb, image=self.cfg.image, work_dir=self.cfg.work_dir,
                                 user=self.cfg.user, runtime=self.cfg.docker_runtime, cpus=self.cfg.cpus,
                                 memory=self.cfg.container_mem, startup_timeout_s=self.cfg.startup_timeout_s)

    def _joint_backend(self, model_dir: str, weight_bytes: int) -> AsyncContextManager[Backend]:
        from .joint_docker import JointDockerBackend
        return JointDockerBackend(model_dir, weight_bytes, gpu_device=self.cfg.gpu_device,
                                  gpu_total_gb=self.cfg.gpu_total_gb, budget_gb=self.cfg.gpu_budget_gb,
                                  image=self.cfg.image, work_dir=self.cfg.work_dir, user=self.cfg.user,
                                  runtime=self.cfg.docker_runtime, cpus=self.cfg.cpus,
                                  memory=self.cfg.container_mem, startup_timeout_s=self.cfg.startup_timeout_s)

    def _pool(self, category: str) -> list[dict]:
        if category not in self._pools:
            rows = S.load_pool(self.cfg.sealed_dir, category)
            if category == "generalist":
                known = self.state.track(category).item_categories
                if any(r["id"] in known and known[r["id"]] != r["category"] for r in rows):
                    raise S.SampleError("generalist item category changed after use")
            self._pools[category] = rows
        return self._pools[category]

    @staticmethod
    def _category_diagnostics(sub: ST.Submission, draw: ST.Draw, items: Sequence[SC.ItemScore]) -> None:
        if draw.draw_id in sub.category_draws:
            return
        per_draw: dict[str, Any] = {}
        for category in S.GENERALIST_CATEGORIES:
            part = [it for it in items if draw.item_categories.get(it.item_id) == category]
            if not part:
                continue
            score = SC.score_items(part)
            per_draw[category] = score.as_dict()
            pooled = sub.category_pooled.setdefault(category, {"n": 0, "primitives": {}})
            pooled["n"] += len(part)
            for it in part:
                stats = pooled["primitives"].setdefault(it.kind, {"n": 0, "valid": 0, "correct": 0,
                    "chance": 0.0, "err": 0.0, "brier": 0.0, "brier_uniform": 0.0})
                for name in stats:
                    stats[name] += 1 if name == "n" else getattr(it, name)
            # The existing scorer consumes per-primitive weighted means; reconstruct its arrays from the sums.
            import numpy as np
            counts = np.array([s["n"] for s in pooled["primitives"].values()], dtype=float)
            arrays = {kind: {"idx": np.array([i]), **{name: np.array([s[source] / s["n"]])
                      for name, source in (("correct", "correct"), ("chance", "chance"), ("err", "err"),
                                           ("brier", "brier"), ("bu", "brier_uniform"), ("valid", "valid"))}}
                      for i, (kind, s) in enumerate(pooled["primitives"].items())}
            competence, calibration, _ = SC._score_weighted(arrays, counts)
            pooled["score"] = SC.combine(competence, calibration)
        sub.category_draws[draw.draw_id] = per_draw

    def _track_close(self, track_id: str) -> float | None:
        """When the track closes (unix seconds, as ``TrackDef.active`` reads it); None = it does not close (generalist)."""
        for t in self.cfg.tracks:
            if t.track_id == track_id:
                return M.finite_track_end(t.end_ts if t.end_ts is not None else t.start_ts + t.days * 86400)
        return None

    def _set_confirm_due(self, ts: ST.TrackState, ch: ST.Submission, first_ts: float, now: float) -> None:
        """``confirm_after_ts`` of a first pass taken at ``first_ts`` (``M.confirm_due_ts``). ``first_pass_ts`` is stored only
        when the due time is not ``first_ts + delay`` (pulled in before the close): None keeps the state as it was before
        the gate and the keep-alive derives the first pass from ``confirm_after_ts`` as always."""
        delay = self.cfg.confirm_delay_s
        ch.confirm_after_ts = M.confirm_due_ts(first_ts, delay, self._track_close(ts.track_id), now)
        ch.first_pass_ts = None if ch.confirm_after_ts == first_ts + delay else first_ts

    def _cap_to_close(self, track_id: str, due: float, now: float) -> float:
        """``due`` (a confirmation time that is already ``>= now``) held back to the close lead of the track once the
        confirm-before-close gate is open; an entry inside the lead is due now. Unchanged without the gate or a close."""
        if not M.confirm_before_close_active(now):
            return due
        close = self._track_close(track_id)
        if close is None:
            return due
        return min(due, max(close - M.CONFIRM_CLOSE_LEAD_S, now))

    def _confirm_before_close(self, active: Sequence[C.TrackDef], now: float) -> None:
        """Every pending confirmation gets its draw before the track closes (gate ``M.CONFIRM_BEFORE_CLOSE_START_UTC``).

        Idempotent pass at the start of the cycle. A pending entry (first pass taken, no confirmation yet) whose
        ``confirm_after_ts`` lies after ``close - CONFIRM_CLOSE_LEAD_S`` and in the future is moved to
        ``max(close - lead, now, first pass)``: entries that became pending before the gate opened (12 h from their first
        pass, possibly past the close) are brought in once; one logged line each, the old value in it. The first pass
        time is kept in ``first_pass_ts`` (a state without it: derived from the old ``confirm_after_ts`` first), so the
        keep-alive window, which counts from the first pass, neither starts earlier nor ends earlier than before.
        A track that has closed with entries still pending (no confirmation draw happened, e.g. the downloads failed) is
        logged once per process: no grace period, nobody is crowned after the close."""
        if not M.confirm_before_close_active(now):
            return
        for tdef in self.cfg.tracks:
            close = self._track_close(tdef.track_id)
            ts = self.state.tracks.get(tdef.track_id)
            if close is None or ts is None:
                continue
            if tdef not in active:
                if now >= close and tdef.track_id not in self._close_noted:
                    self._close_noted.add(tdef.track_id)
                    for sub in ts.ordered(ST.PENDING_CONFIRM):
                        if not sub.confirm_draw:
                            text = f"{sub.key[:12]}: track closed with the confirmation still pending"
                            if any(text in str(e) for e in ts.events[-200:]):
                                continue   # critic F4: once per entry, also across restarts
                            ts.log(now, f"{sub.key[:12]}: track closed with the confirmation still pending "
                                        f"(due {int(sub.confirm_after_ts or 0)}); no grace period, no crowning after the close")
                            log.warning("track %s closed with %s still pending confirmation", tdef.track_id, sub.key[:12])
                continue
            limit = close - M.CONFIRM_CLOSE_LEAD_S
            for sub in ts.ordered(ST.PENDING_CONFIRM):
                c = sub.confirm_after_ts
                if sub.confirm_draw or c is None or c <= limit or c <= now:
                    continue
                first = _first_pass_ts(sub)
                sub.first_pass_ts = first
                sub.confirm_after_ts = max(limit, now, first if first is not None else 0.0)
                ts.log(now, f"{sub.key[:12]}: confirmation pulled forward before the track closes "
                            f"(was due {int(c)}, now {int(sub.confirm_after_ts)})")

    def _entry_floor(self, tdef: C.TrackDef) -> float:
        """The release constant ``competition_split.TRACK_ENTRY_FLOOR`` (None = none) or the configured floor, if higher."""
        return max(self.cfg.entry_floor, TRACK_ENTRY_FLOOR.get(tdef.track_id) or 0.0)

    def _track_order(self, active: Sequence[C.TrackDef]) -> list[C.TrackDef]:
        """A track whose cohort playoff waited for budget first, then the track evaluated least recently."""
        return sorted(active, key=lambda t: (t.track_id not in self._playoff_waits, self._last_served(t)))

    def _last_served(self, tdef: C.TrackDef) -> float:
        """When a model of this track was last evaluated (0 = never); ties keep the schedule order (sorted is stable)."""
        ts = self.state.track(tdef.track_id)
        return max((s.last_eval_ts or 0.0 for s in ts.submissions.values() if s.status != ST.QUEUED), default=0.0)

    def _has_work(self, tdef: C.TrackDef, now: float) -> bool:
        """Something ``_run_track`` could evaluate this cycle: a queued entry not waiting for its repo, or a due confirmation."""
        ts = self.state.track(tdef.track_id)
        co = {m.key for m in v5_bar_members(ts, now)}
        if self._slot_due(ts, now):
            return True     # CROWN_SLOTS_UTC: the slot has come, the stored crowning(s) are decided this cycle
        for s in ts.submissions.values():
            if s.status == ST.CHAMPION and s.serve_retry_pending:
                return True
            # waiting repos get metadata probes each cycle; an unavailable entry alone does not claim a share
            if s.status == ST.QUEUED and not s.reason.startswith("waiting"):
                return True
            if s.status == ST.PENDING_CONFIRM and (_crown_pending(s) is not None or s.key in co):
                continue    # a stored crowning waits for its slot (no re-challenge); a co-champion is not a challenger
            if (s.status == ST.PENDING_CONFIRM and not s.confirm_draw and s.confirm_after_ts is not None
                    and now >= s.confirm_after_ts and s.draw1 in ts.draws
                    and not s.reason.startswith(WAITING_FOR_REPO) and not RV.under_review(s)):    # as ``_run_track``'s ``due`` (review F2, critic FX2)
                return True
        return False

    def _budget(self, now: float) -> int:
        self.state.eval_times = [t for t in self.state.eval_times if t > now - 3600]
        return max(0, self.cfg.max_evals_per_hour - len(self.state.eval_times))

    def _cleanup(self, active: Sequence[C.TrackDef]) -> None:
        keep = set()
        for ts in self.state.tracks.values():
            for s in ts.submissions.values():
                if s.status == ST.CHAMPION:     # served every round; a pending entry is fetched again (pinned revision)
                    keep.add(self._model_dir(s))
            keep.update(self._model_dir(m) for m in v5_bar_members(ts, self.clock()))   # the opening co-champions are served every round too
        root = os.path.join(self.cfg.work_dir, "models")
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if p not in keep:
                shutil.rmtree(p, ignore_errors=True)

    # ----------------------------------------------------- submission preparation

    def _fetch_failure(self, tdef: C.TrackDef, sub: ST.Submission, snap: Snapshot, now: float,
                       exc: F.FetchError) -> None:
        """Apply the existing fetch failure rules to a download or a due metadata retry."""
        ts = self.state.track(tdef.track_id)
        if sub.status == ST.WITHDRAWN:
            # only ever prepared as a parent (critic C1): an unfetchable withdrawn parent drops out of the group,
            # its record (and a later carry-over) is not touched
            sub.reason = f"withdrawn; parent not fetchable: {exc}"[:300]
            return None
        preopening = sub.track_id == "generalist" and sub.first_seen_ts < split.GENERALIST_START_UTC.timestamp()
        publish_expired = (now > split.GENERALIST_START_UTC.timestamp() + 24 * 3600 if preopening
                           else snap.block > sub.first_seen_block + C.PUBLISH_DEADLINE_BLOCKS)
        rejects = (exc.kind == "rejected" or (exc.kind == "unavailable" and
                   ((sub.hash_named and sub.unavailable_fetches + 1 >= C.MAX_ATTEMPTS) or
                    (not sub.hash_named and publish_expired))))
        if rejects and self._cycle_fetch_rejections.get(tdef.track_id, 0) >= MAX_SKIPS_PER_CYCLE:
            sub.reason, sub.wait_kind = f"waiting for the repo: {exc}"[:300], ST.WAIT_MINER
            return None
        if exc.kind == "rejected":
            sub.status, sub.reason = ST.REJECTED, str(exc)[:300]
            ts.log(now, f"rejected {sub.key[:12]}: {sub.reason[:120]}")
        elif exc.kind == "infra":
            sub.reason, sub.wait_kind = f"waiting for the repo: {exc}"[:300], ST.WAIT_INFRA
            ts.log(now, f"fetch deferred {sub.key[:12]}: infrastructure failure, not counted: {str(exc)[:100]}")
        elif sub.hash_named:
            sub.unavailable_fetches += 1
            if sub.unavailable_fetches >= C.MAX_ATTEMPTS:
                sub.status, sub.reason = ST.REJECTED, f"no longer available ({sub.unavailable_fetches} fetches failed)"
                sub.serve_retry_pending = False
                if ts.champion == sub.key:
                    ts.champion = ""
                ts.log(now, f"rejected {sub.key[:12]}: {sub.reason}")
            else:
                sub.reason, sub.wait_kind = f"waiting for the repo: {exc}"[:300], ST.WAIT_MINER
        elif not sub.hash_named and publish_expired:
            deadline = "24 hours after generalist opening" if preopening else f"{C.PUBLISH_DEADLINE_BLOCKS} blocks"
            sub.status, sub.reason = ST.REJECTED, f"not available within {deadline}: {exc}"[:300]
            ts.log(now, f"expired {sub.key[:12]}")
        else:
            # Missing/private revisions are skipped and re-checked, never serve-failure strikes.
            sub.reason, sub.wait_kind = f"waiting for the repo: {exc}"[:300], ST.WAIT_MINER
        if sub.status == ST.REJECTED:
            self._cycle_fetch_rejections[tdef.track_id] = self._cycle_fetch_rejections.get(tdef.track_id, 0) + 1
        return None

    async def _prepare(self, tdef: C.TrackDef, sub: ST.Submission, snap: Snapshot, now: float) -> _Prepared | None:
        """Fetch and check one submission. None = not usable now (the status says why)."""
        ts = self.state.track(tdef.track_id)
        dest = self._model_dir(sub)
        commitment = SpecCommitment(sub.track_id, sub.repo, sub.revision)
        kept = self._kept(sub, dest)
        if kept is not None:
            self._backfill_params(sub, dest)
            if sub.delta_checked:
                return kept
            # kept by a release whose base check could pass without the base: check it before it is served again
            if not await self._refresh_sketches(tdef, ts, sub):
                return self._renewal_wait(ts, kept, now)
            return await self._checked(ts, kept, now, self._near_check)
        if (sub.reason.startswith(WAITING_FOR_REPO) and sub.last_fetch_ts is not None
                and now - sub.last_fetch_ts < C.RETRY_UNAVAILABLE_S):
            return None
        if self._disk_wait:
            return None
        sub.last_fetch_ts = now
        try:
            ins = await self.fetcher.fetch(commitment, dest, category=tdef.category, hotkey=sub.hotkey,
                                          rules=self.cfg.rules_for(now, tdef.category))
        except OSError as exc:
            # A full/read-only filesystem affects the host: abandon this track rather than blaming its entries.
            if exc.errno in (errno.ENOSPC, errno.EDQUOT, errno.EROFS):
                ts.log(now, f"fetch infrastructure failure {sub.key[:12]}: {exc}; nothing counted")
                raise
            sub.reason, sub.wait_kind = f"waiting for the repo: local I/O failure: {exc}"[:300], ST.WAIT_INFRA
            ts.log(now, f"fetch deferred {sub.key[:12]}: {sub.reason[:120]}")
            log.warning("fetch %s deferred (host I/O): %s", sub.key[:12], exc)
            return None
        except F.FetchError as exc:
            if isinstance(exc, F.DiskLow):
                self._disk_wait = True
                log.warning("track %s: %s waits for this host, no further download this cycle: %s", tdef.track_id, sub.key[:12], exc)
            return self._fetch_failure(tdef, sub, snap, now, exc)
        sub.unavailable_fetches = 0
        sub.wait_kind = ""
        assert ins.hashes is not None
        if sub.status == ST.WITHDRAWN and sub.hash_named and ins.hashes.named != sub.hash_named:
            shutil.rmtree(dest, ignore_errors=True)
            return None
        if sub.hash_named and ins.hashes.named != sub.hash_named:
            # fetched again (a pending entry's files are not kept, 1.2.0rc6): only the weights that took the first draw
            shutil.rmtree(dest, ignore_errors=True)
            sub.status, sub.reason = ST.REJECTED, "fetched again: other weights than at its first check"
            ts.log(now, f"rejected {sub.key[:12]}: {sub.reason}")
            return None
        if sub.params is None:
            sub.params = int(ins.hashes.params)
        with open(os.path.join(dest, MARKER), "w") as fh:
            json.dump({"hash": ins.hashes.named, "weight_bytes": ins.hashes.weight_bytes, "files": ins.facts.files}, fh)
        if sub.status == ST.WITHDRAWN:
            # a withdrawn parent (critic C1): same weights as recorded, no rule/copy check again (they could reject it
            # or the author's carried recommit)
            if not sub.hash_named:
                shutil.rmtree(dest, ignore_errors=True)
                return None
            return _Prepared(sub, dest, ins.hashes.weight_bytes)
        if sub.hash_named and sub.delta_checked and sub.status == ST.PENDING_CONFIRM:
            # a pending entry fetched again with the weights of its first check: as if its files had been kept (``_kept``)
            return _Prepared(sub, dest, ins.hashes.weight_bytes)
        sub.hash_named, sub.hash_unnamed = ins.hashes.named, ins.hashes.unnamed
        sub.hash_backbone = ins.hashes.backbone if sub.track_id == "generalist" else ""
        sub.base_model = str((ins.facts.manifest or {}).get("base_model", ""))
        await self._fetch_base(sub)
        if not await self._refresh_sketches(tdef, ts, sub):
            return self._renewal_wait(ts, _Prepared(sub, dest, ins.hashes.weight_bytes), now)
        p = await self._checked(ts, _Prepared(sub, dest, ins.hashes.weight_bytes), now, self._copy_check)
        if p is not None:
            self._carryover(ts, sub, snap, now)
        return p

    def _carryover(self, ts: ST.TrackState, sub: ST.Submission, snap: Snapshot, now: float) -> None:
        if sub.status != ST.QUEUED or sub.carried_from:
            return
        registered = {n.hotkey: n.registered_block for n in snap.neurons}
        # a re-entry's own original first (critic r1 F5), then the earliest eligible record
        for old in sorted(ts.submissions.values(), key=lambda s: (not reentry_match(s, sub), s.first_seen_block, s.uid, s.key)):
            if not carryover_eligible(old, sub, registered, min(now, sub.first_seen_ts)):
                continue
            for name in ("score", "draw1", "delta", "first_gain", "first_incumbent", "nonproc", "by_tier",
                         "provisional_since_ts", "decision_rule", "close_draws", "confirm_draws"):
                setattr(sub, name, copy.deepcopy(getattr(old, name)))
            # v3: the pooled evidence of the current stage travels with the entry (first-stage rows, and the confirmation rows
            # of an entry that waits for its extra confirmation, gap fields included)
            stage = {"first": {old.draw1, *old.close_draws}, "confirm": set(old.confirm_draws)}
            sub.history = copy.deepcopy([h for h in old.history if h.get("draw") in stage.get(h.get("kind"), ())
                                         or h.get("kind") == RV.FLAG_KIND]
                                        + [h for h in sub.history if h.get("kind") == RV.FLAG_KIND])   # rc11.post5: a review flag travels
            sub.status, sub.reason = ST.PENDING_CONFIRM, ""
            uncapped = max(old.confirm_after_ts, now)
            sub.confirm_after_ts = self._cap_to_close(ts.track_id, uncapped, now)
            # first_pass_ts only where confirm_after_ts no longer says it (None otherwise: the state stays as before the gate)
            sub.first_pass_ts = old.first_pass_ts
            if sub.first_pass_ts is None and sub.confirm_after_ts != uncapped:
                sub.first_pass_ts = uncapped - M.CONFIRM_DELAY_S
            sub.carried_from, old.carried_to = old.key, sub.key
            # critic B2: the recommit keeps the original's derivation; a near-copy someone committed after the original
            # (while it was pruned) does not become its parent
            sub.derived_from, sub.derived_how = old.derived_from, old.derived_how
            ts.log(now, f"{sub.key}: same tensors as pruned {old.key} (uid {old.uid}), "
                        "first draw carried over (rule 17 Oct)")
            if self._cohort_block(old) == 0 and reentry_match(old, sub):
                # re-entry (#23211): the opening-cohort place travels with the entry (first seen as the original)
                sub.first_seen_ts = old.first_seen_ts
                ts.log(now, f"{sub.key}: re-entry of opening-cohort entry {old.key} (uid {old.uid}), keeps its opening-cohort place")
            break

    async def _checked(self, ts: ST.TrackState, p: _Prepared, now: float,
                       check: Callable[[ST.TrackState, ST.Submission, str], tuple[str, str]]) -> _Prepared | None:
        """Run a copy check; a refused or unverifiable submission is not served (its status and reason say why)."""
        sub = p.sub
        # reads about half of the candidate and of its base from disk: never on the event loop
        verdict, why = await asyncio.get_running_loop().run_in_executor(None, check, ts, sub, p.path)
        if not verdict:
            return p
        if verdict == ST.PENDING_VERIFY:
            self._hold(ts, sub, why, now, self._base_wait_kind(sub))
        else:
            sub.status, sub.reason = verdict, why
            ts.log(now, f"rejected {sub.key[:12]}: {why[:120]}")
        shutil.rmtree(p.path, ignore_errors=True)
        return None

    def _review_wait(self, ts: ST.TrackState, s: ST.Submission, now: float) -> str:
        """rc11.post5: why the confirmation of ``s`` waits for a reviewer ("" = it does not; ``review.under_review``). Logged
        once per entry and reason; an entry without a reason of its own shows it."""
        why = RV.under_review(s)
        if why and (s.key, why) not in self._review_seen:
            self._review_seen.add((s.key, why))
            ts.log(now, f"{s.key[:12]} under review, its confirmation waits: {why}"[:300])
            log.warning("track %s: %s under review, not confirmed: %s", ts.track_id, s.key[:12], why)
        if why and not s.reason:
            s.reason = f"under review: {why}"
        return why

    def _renewal_wait(self, ts: ST.TrackState, p: _Prepared, now: float) -> None:
        """Earlier sketches must be renewed before ``p`` can be checked, and the disk is too full for the re-fetch: held
        (``_run_track`` queues it again next cycle) instead of checked against an incomplete set of sketches."""
        self._hold(ts, p.sub, RENEWAL_WAIT, now, ST.WAIT_INFRA)
        shutil.rmtree(p.path, ignore_errors=True)
        return None

    def _bases_in_use(self) -> set[str]:
        """Declared bases the base cache must keep: those of every submission that is not rejected or withdrawn, and the preseed."""
        return {s.base_model for ts in self.state.tracks.values() for s in ts.submissions.values()
                if s.base_model and s.status not in (ST.REJECTED, ST.WITHDRAWN)} | set(self.cfg.base_preseed)

    async def _fetch_base(self, sub: ST.Submission) -> None:
        """Download the declared base of ``sub`` if it is missing here and the base fetcher may fetch it now."""
        if self.base_fetcher is not None and sub.base_model and self.base_provider(sub.base_model) is None:
            await self.base_fetcher.ensure(sub.base_model)

    def _base_detail(self, sub: ST.Submission) -> str:
        return self.base_fetcher.reason(sub.base_model) if self.base_fetcher is not None and sub.base_model else ""

    def _base_wait_kind(self, sub: ST.Submission) -> str:
        """WAIT_INFRA if the declared base is missing for this host's reason (``BaseFetcher.waits_here``), else WAIT_MINER
        (the base is here and the weights are not comparable, or the base itself cannot be had)."""
        missing = bool(sub.base_model) and self.base_provider(sub.base_model) is None
        here = missing and self.base_fetcher is not None and self.base_fetcher.waits_here(sub.base_model)
        return ST.WAIT_INFRA if here else ST.WAIT_MINER

    @staticmethod
    def _hold(ts: ST.TrackState, sub: ST.Submission, why: str, now: float, kind: str = ST.WAIT_MINER) -> None:
        """PENDING_VERIFY: kept, not served, not ranked, not crowned; checked again every cycle. ``kind``: whose wait it
        is (``ST.WAIT_INFRA`` holds back the cohort's first crowning)."""
        sub.status, sub.reason, sub.wait_kind = ST.PENDING_VERIFY, why, kind
        sub.close_draws, sub.confirm_draws = [], []     # v3: a requeued entry starts a fresh stage (no close priority, no pooling)
        if ts.champion == sub.key:
            ts.champion = ""            # no award while the copy check is pending
        log.warning("track %s: %s (uid %d) is pending verification: %s", sub.track_id, sub.key[:12], sub.uid, why)
        ts.log(now, f"pending verification {sub.key[:12]}: {why[:120]}")

    @staticmethod
    def _kept(sub: ST.Submission, dest: str) -> "_Prepared | None":
        """A model directory kept from an earlier group (champion, pending confirmation): reuse it if untouched."""
        try:
            with open(os.path.join(dest, MARKER)) as fh:
                m = json.load(fh)
            files = {k: v for k, v in F.list_dir(dest).items() if k != MARKER}
            if m["hash"] == sub.hash_named and m["files"] == files:
                return _Prepared(sub, dest, int(m["weight_bytes"]))
        except (OSError, ValueError, KeyError):
            pass
        shutil.rmtree(dest, ignore_errors=True)
        return None

    @staticmethod
    def _backfill_params(sub: ST.Submission, dest: str) -> None:
        """An entry kept from before 1.2.0rc9.post6 has no recorded parameter count: read it from the safetensors headers."""
        if sub.params is None:
            try:
                sub.params = int(T.param_count(T.read_model(dest)))
            except (T.SafetensorsError, OSError, ValueError) as exc:
                # stays None on purpose (unknown counts as small, _big_ok); the next successful read fills it in
                log.warning("could not read the parameter count of %s from %s: %s", sub.key[:12], dest, exc)

    @staticmethod
    def _earlier(a: ST.Submission, b: ST.Submission) -> bool:
        return (a.first_seen_block, a.uid, a.key) < (b.first_seen_block, b.uid, b.key)

    @staticmethod
    def _foreign_original(other: ST.Submission, sub: ST.Submission) -> bool:
        """``other`` is a withdrawn (pruned or replaced) commitment of somebody else: another coldkey and another repo.
        Its weights stay in the copy registry (exact hashes and the near-duplicate sketch), so a copy of a pruned
        entrant's work is linked to it (``derived_from``) instead of passing as new work (reg-sniping review 20 Oct)."""
        return other.status == ST.WITHDRAWN and other.coldkey != sub.coldkey and other.repo != sub.repo

    def _copy_check(self, ts: ST.TrackState, sub: ST.Submission, model_dir: str) -> tuple[str, str]:
        """Exact tensor hashes (re-sharding and renaming proof), then the near-duplicate check.

        Returns ``(verdict, reason)``: verdict "" = passed, ``ST.REJECTED`` or ``ST.PENDING_VERIFY``."""
        if sub.track_id == "generalist":
            # Backfill retained pre-upgrade champions before comparing a new readout head.
            for other in ts.submissions.values():
                if other.hash_backbone or not other.hash_named:
                    continue
                dest = self._model_dir(other)
                if self._kept(other, dest) is not None:
                    hashes = T.canonical_hashes(dest)
                    if hashes.named == other.hash_named:
                        other.hash_backbone = hashes.backbone
                        ts.hashes.setdefault(hashes.backbone, other.key)
        why = self._exact_check(ts, sub)
        if why:
            return ST.REJECTED, why
        return self._near_check(ts, sub, model_dir)

    def _exact_check(self, ts: ST.TrackState, sub: ST.Submission) -> str:
        """Exact tensor hashes against every earlier commitment of the track. Returns a rejection reason. A withdrawn
        commitment of the same coldkey or HF repo does not count (control proven by the bound manifest)."""
        for h in filter(None, (sub.hash_named, sub.hash_unnamed, sub.hash_backbone)):
            owner_key = ts.hashes.get(h)
            owner = ts.submissions.get(owner_key) if owner_key else None
            if owner is None or owner is sub:
                continue
            if owner.status == ST.WITHDRAWN and (owner.coldkey == sub.coldkey or owner.repo == sub.repo):
                # re-registration (1.2.0rc7): the same coldkey commits its withdrawn model again, under a new uid or a
                # new hotkey. The same repo also proves control via its checked hotkey binding. The hash passes on.
                continue
            # a carried recommit (rule 17 Oct) is as old as its pruned original: an exact copy committed during the prune
            # window but fetched after the recommit must not take its hash (critic delta 2, D1)
            first = (ts.submissions.get(owner.carried_from) if owner.carried_from else None) or owner
            if self._earlier(first, sub) or (owner.status in (ST.CHAMPION, ST.FORMER)
                                             and (owner.coldkey == sub.coldkey or sub.first_seen_block >= first.first_seen_block)):
                return f"identical tensors to an earlier commitment (uid {owner.uid})"
            # ``sub`` is the earlier commitment of another coldkey (first seen in a strictly earlier sync, never by the uid
            # tie-break), checked later (e.g. queued behind the copy because the copier's immunity ended first): the copy
            # is rejected even if it was crowned (reg-sniping review 20 Oct)
            owner.status, owner.reason = ST.REJECTED, f"identical tensors to an earlier commitment (uid {sub.uid})"
            if ts.champion == owner.key:
                ts.champion = ""
        ts.hashes[sub.hash_named] = sub.key
        ts.hashes[sub.hash_unnamed] = sub.key
        if sub.hash_backbone:
            ts.hashes[sub.hash_backbone] = sub.key
        return ""

    def _sketch_current(self, sub: ST.Submission) -> bool:
        path = self._sketch_path(sub)
        try:
            return os.path.exists(path) and T.sketch_seed_id(path) == self._seed_id
        except Exception:   # unreadable or damaged file (zipfile.BadZipFile from np.load, ...): renew it
            return False

    async def _refresh_sketches(self, tdef: C.TrackDef, ts: ST.TrackState, sub: ST.Submission) -> bool:
        """Before ``sub`` is checked: recompute the sketch of every earlier checked submission on the same base whose
        stored sketch was taken with another seed or rule (rc5 sketches, a changed seed, a lost work directory). Sketches
        are only comparable at the same positions. The weights are re-fetched (the commitment pins the revision) unless
        they are still on disk, and removed again afterwards unless the submission is kept anyway.

        False = a re-fetch would leave less than ``min_free_bytes`` free: ``sub`` must wait (``RENEWAL_WAIT``). Any
        other failure on one earlier submission is logged and skipped (its sketch stays old and is not compared)."""
        base_dir = self.base_provider(sub.base_model) if sub.base_model else None
        if base_dir is None:
            return True
        base_id = self._base_identity(base_dir)
        loop = asyncio.get_running_loop()
        for other in sorted(ts.submissions.values(), key=lambda s: (s.first_seen_block, s.uid, s.key)):
            if (other is sub or not self._earlier(other, sub) or not other.delta_checked or other.status == ST.REJECTED
                    or (other.status == ST.WITHDRAWN and not self._foreign_original(other, sub))
                    or self._resolved_base(other) != base_id or self._sketch_current(other)):
                continue
            dest = self._model_dir(other)
            try:
                kept = self._kept(other, dest)
                if kept is None:
                    free = self._free_bytes(self.cfg.work_dir)
                    if free < self.cfg.min_free_bytes:
                        log.warning("track %s: sketch of %s not renewed, %.0f GB free, %.0f GB needed before a download",
                                    tdef.track_id, other.key[:12], free / 1e9, self.cfg.min_free_bytes / 1e9)
                        return False
                    try:
                        ins = await self.fetcher.fetch(SpecCommitment(other.track_id, other.repo, other.revision), dest,
                                                       category=tdef.category, hotkey=other.hotkey,
                                                       rules=self.cfg.rules_for(self.clock(), tdef.category))
                    except F.FetchError as exc:
                        log.warning("track %s: sketch of %s cannot be renewed: %s", tdef.track_id, other.key[:12], exc)
                        continue
                    if ins.hashes is None or ins.hashes.named != other.hash_named:
                        log.warning("track %s: sketch of %s cannot be renewed: other weights than at its check",
                                    tdef.track_id, other.key[:12])
                        shutil.rmtree(dest, ignore_errors=True)
                        continue
                    with open(os.path.join(dest, MARKER), "w") as fh:
                        json.dump({"hash": ins.hashes.named, "weight_bytes": ins.hashes.weight_bytes, "files": ins.facts.files}, fh)
                sk = await loop.run_in_executor(None, lambda: T.delta_sketch_ex(dest, base_dir, seed=self._seed))
                T.save_sketch(sk.deltas, self._sketch_path(other), sk.seed_id)
                ts.log(self.clock(), f"{other.key[:12]}: sketch renewed for the current sampling rule")
            except Exception as exc:    # ENOSPC, a damaged download, ...: not this cycle's whole track
                log.warning("track %s: sketch of %s cannot be renewed: %r", tdef.track_id, other.key[:12], exc)
            finally:
                # a kept submission's directory is checked against its MARKER before it is served (``_kept``)
                if other.status not in (ST.CHAMPION, ST.PENDING_CONFIRM):
                    shutil.rmtree(dest, ignore_errors=True)
        return True

    def _base_identity(self, base_dir: str) -> str:
        """Tensor hash of a base model directory: the same base under another name (an alias, a mirror, another
        directory) has the same identity. Cached per directory content, computed once per process."""
        key = (os.path.realpath(base_dir), tuple(sorted(F.list_dir(base_dir).items())))
        if key not in self._base_ids:
            self._base_ids[key] = T.canonical_hashes(base_dir).named
        return self._base_ids[key]

    def _resolved_base(self, sub: ST.Submission) -> str:
        """The base identity of an earlier submission (resolved now if a state written before ``base_id`` lacks it)."""
        if sub.base_id or not sub.delta_checked:
            return sub.base_id
        base_dir = self.base_provider(sub.base_model) if sub.base_model else None
        return self._base_identity(base_dir) if base_dir else ""

    @staticmethod
    def _is_near(st: "T.DeltaStats") -> bool:
        """Close to the champion's weights (near-champion rule): cos > 0.995, or relative L2 < 0.15, or the champion's change
        plus orthogonal noise (cos > 0.98 and alpha within 0.05 of 1)."""
        if not (st.compared > 0 and st.norm_a > 0):
            return False
        return (st.cosine > NEAR_CHAMP_COS or st.relative_l2 < NEAR_CHAMP_REL
                or (st.cosine > NEAR_CHAMP_ALPHA_COS and abs(st.alpha - 1.0) < NEAR_CHAMP_ALPHA_TOL))

    def _near_champ(self, ch: ST.Submission, champ: ST.Submission, ch_path: str = "") -> "T.DeltaStats | None":
        """The delta-sketch comparison of ``ch`` with the incumbent (near-champion rule), or None when it cannot be made
        (logged and recorded as ``comparable: false``). Both on the same resolved base: the stored sketches. A challenger
        that declares another base (any public repo can be declared, the champion's own included) is sketched again from
        its served weights against the CHAMPION's base, so a changed manifest does not hide a copy; tensors that do not
        match the champion's base by name and shape give no comparison (another architecture: not close)."""
        if champ.key == ch.key or not self._sketch_current(champ):
            return None
        base = self._resolved_base(champ)
        same = bool(base) and base == self._resolved_base(ch)
        ck = (champ.key, ch.key, self._seed_id)
        if ck in self._near_cache:
            return self._near_cache[ck]   # None here: the weights do not map onto the champion's base (cached, it cannot change)
        try:
            ref = T.load_sketch(self._sketch_path(champ))
            if same and self._sketch_current(ch):
                other = T.load_sketch(self._sketch_path(ch))
            else:
                base_dir = self.base_provider(champ.base_model) if champ.base_model else None
                if base_dir is None or not ch_path or not os.path.isdir(ch_path):
                    log.warning("near-champion check %s vs %s: not comparable (champion's base or the served weights missing)",
                                ch.key[:12], champ.key[:12])
                    return None
                other = T.delta_sketch_ex(ch_path, base_dir, seed=self._seed).deltas
                cov = T.sketch_coverage(other, ch_path, base_dir)
                if not cov.enough():   # critic-nc2 #1: another layout (or another architecture): recorded, never silently "far"
                    log.info("near-champion check %s vs %s: only %.1f%% of the champion's base maps onto the weights: not comparable",
                             ch.key[:12], champ.key[:12], 100 * cov.fraction)
                    self._near_cache[ck] = None
                    return None
        except Exception as e:   # unreadable sketch or weights: not comparable this time (recorded, retried next draw)
            log.warning("near-champion check %s vs %s: %s", ch.key[:12], champ.key[:12], e)
            return None
        self._near_cache[ck] = T.compare_sketches(ref, other)
        return self._near_cache[ck]

    @staticmethod
    def _lineage(ts: ST.TrackState, key: str) -> list[str]:
        """``[key, parent, grandparent, ...]`` along ``derived_from`` (each key once, so a cycle ends the walk)."""
        out: list[str] = []
        k = key
        while k and k not in out:
            out.append(k)
            sub = ts.submissions.get(k)
            k = sub.derived_from if sub is not None else ""
        return out

    def _near_check(self, ts: ST.TrackState, sub: ST.Submission, model_dir: str) -> tuple[str, str]:
        """Weight change against the resolved declared base, compared with earlier submissions on the same base.

        Without the base on this host the submission is not served and not crowned: it waits in PENDING_VERIFY and is
        checked again once the base is there. A submission that was checked before stays checked."""
        base_dir = self.base_provider(sub.base_model) if sub.base_model else None
        if base_dir is None:
            if sub.delta_checked:
                return "", ""
            return ST.PENDING_VERIFY, _no_base(sub, self._base_detail(sub))
        sub.base_id = self._base_identity(base_dir)
        sub.sketch_rule = T.SKETCH_RULE
        sk = T.delta_sketch_ex(model_dir, base_dir, seed=self._seed)
        sketch = sk.deltas
        cov = T.sketch_coverage(sketch, model_dir, base_dir)
        fp8 = T.coverage_note(cov)
        if not sketch:
            return ST.PENDING_VERIFY, (f"no weight of the model matches the declared base model {sub.base_model} "
                                       f"(name, shape): the near-duplicate check cannot be made{fp8}")
        if not cov.enough():
            return ST.PENDING_VERIFY, (f"the weights comparable with the declared base model {sub.base_model} hold only "
                                       f"{cov.fraction:.1%} of its float parameters ({T.MIN_SKETCH_COVERAGE:.0%} needed): "
                                       f"the near-duplicate check cannot be made{fp8}")
        sub.delta_checked = True
        if sk.review and RV.record(sub, sk.review, self.clock()):
            # rc11.post5: a function-preserving rewrite of the weights (or a delta no fine-tune makes): drawn, never
            # confirmed, crowned or kept alive until a reviewer clears it (review.py)
            ts.log(self.clock(), f"{sub.key[:12]} under review: weights reparametrised against the declared base "
                                 f"({', '.join(sk.review)[:200]})")
            log.warning("spec %s: %s flagged for review: %s", sub.track_id, sub.key[:12], ", ".join(sk.review))
        if not sk.changed:
            return ST.REJECTED, "weights are identical to the declared base model"
        for other in sorted(ts.submissions.values(), key=lambda s: (s.first_seen_block, s.uid, s.key)):
            if other is sub or other.status == ST.REJECTED or self._resolved_base(other) != sub.base_id:
                continue
            if other.status == ST.WITHDRAWN and not self._foreign_original(other, sub):
                continue    # the author's own withdrawn model (re-registration, carry-over): not a copy
            if not self._earlier(other, sub) or not self._sketch_current(other):
                continue    # another seed or rule samples other positions: not comparable (_refresh_sketches renews it)
            stats = T.compare_sketches(T.load_sketch(self._sketch_path(other)), sketch)
            if stats.derived(alpha_tol=NC_ALPHA_TOL if self.cfg.nc_r3 else None):
                sub.derived_from, sub.derived_how = other.key, "weights"
                ts.log(self.clock(), f"{sub.key[:12]} is derived from {other.key[:12]} (cos {stats.cosine:.3f}, rel {stats.relative_l2:.3f}"
                                       + (f", alpha {stats.alpha:.3f})" if self.cfg.nc_r3 else ")"))
                if other.status == ST.WITHDRAWN:
                    # a copy of a withdrawn (pruned or replaced) entrant's work: its author cannot defend it as a parent
                    # (the repo may be gone, the hotkey is), so the copy is refused, not crowned (reg-sniping review 20 Oct)
                    T.save_sketch(sketch, self._sketch_path(sub), sk.seed_id)
                    return ST.REJECTED, (f"near-copy of the weights of an earlier, withdrawn commitment (uid {other.uid}, "
                                         f"cos {stats.cosine:.3f}, rel {stats.relative_l2:.3f})")
                break
        T.save_sketch(sketch, self._sketch_path(sub), sk.seed_id)
        return "", ""

    # ------------------------------------------------------------------ one group

    async def _serve(self, p: _Prepared, rows: list[dict], now: float) -> list[SC.ItemScore] | None:
        p.joint = False
        p.infra_failure = p.preflight_failure = p.model_specific_failure = False
        p.sub.last_eval_ts = now
        self.state.eval_times.append(now)
        try:
            factory = self.backend_factory
            # Explicit category/time gating; custom test backends retain their original two-argument contract.
            if factory == self._docker_backend:
                from .joint import detect
                try:
                    info = detect(p.path, category=p.sub.track_id, now_ts=now)
                except (ValueError, OSError, TypeError) as exc:
                    raise ModelUnavailable(f"invalid config: joint readout: {exc}") from exc
                if info is not None:
                    factory = self._joint_backend
                    p.joint = True
            async with factory(p.path, p.weight_bytes) as backend:
                probs = await asyncio.wait_for(backend.decide(rows), SERVE_TIMEOUT_S)
        except (ModelUnavailable, asyncio.TimeoutError, OSError) as exc:
            from .joint_docker import JointModelUnavailable
            p.infra_failure = isinstance(exc, (HostUnavailable, OSError, asyncio.TimeoutError))
            p.preflight_failure = isinstance(exc, GPUPreflightUnavailable)
            p.model_specific_failure = (not p.infra_failure
                                        and (isinstance(exc, (ModelBudgetOOM, JointModelUnavailable)) or model_specific_failure(str(exc))))
            log.warning("serving %s failed%s: %s", p.sub.key[:12], " (host problem, not counted)" if p.infra_failure else "", exc)
            p.sub.reason = f"could not be served: {exc}"[:300]
            return None
        items, _ = SC.score_run(rows, probs)
        p.probs = list(probs)
        return items

    @staticmethod
    def _serve_round(ts: ST.TrackState, sub: ST.Submission, outcome: str, now: float, *, joint: bool = False) -> None:
        """One verdict per scheduled round; retries share the verdict. Old consecutive counters are not migrated."""
        if sub.status == ST.WITHDRAWN:
            return
        if joint:
            sub.joint_infra_outcomes = sub.joint_infra_outcomes + 1 if outcome == "infra" else 0
            if outcome == "infra" and sub.joint_infra_outcomes > 3:
                outcome = "model"
                ts.log(now, f"joint {sub.key[:12]} exceeded three consecutive infrastructure outcomes; model strike")
        sub.serve_log.append([now, outcome])
        del sub.serve_log[:-C.START_FAIL_LOG_SIZE]
        sub.attempts = sum(entry[1] == "model" for entry in sub.serve_log[-C.START_FAIL_WINDOW:])
        if outcome == "ok":
            sub.last_serve_ok_ts = now
            sub.champion_rounds_without_ok = 0
            if sub.pay_suspended_since_ts is not None:
                ts.log(now, f"champion {sub.key[:12]} serving again; award suspension cleared")
                sub.pay_suspended_since_ts = None
            sub.serve_retry_pending = False
            sub.reason = (f"start-up failures: {sub.attempts} of the last {C.START_FAIL_WINDOW} rounds"
                          if sub.attempts else "")
        elif outcome == "infra":
            ts.log(now, f"serve infrastructure failure {sub.key[:12]}: not counted ({sub.reason[:120]})")
        else:
            detail = sub.reason
            sub.reason = f"start-up failures: {sub.attempts} of the last {C.START_FAIL_WINDOW} rounds"
            ts.log(now, f"{sub.key[:12]} {sub.reason}: {detail[:120]}")
            if sub.attempts >= C.START_FAIL_LIMIT:
                sub.status = ST.REJECTED
                sub.reason += "; could not be served after repeated attempts"
                if ts.champion == sub.key:
                    ts.champion = ""
                sub.serve_retry_pending = False
                ts.log(now, f"rejected {sub.key[:12]}: {sub.reason}")

        if sub.status == ST.CHAMPION and outcome != "ok":
            sub.champion_rounds_without_ok += 1
            SpecEvaluator._champion_liveness(ts, sub, now)

    @staticmethod
    def _champion_liveness(ts: ST.TrackState, sub: ST.Submission, now: float) -> None:
        since = sub.last_serve_ok_ts if sub.last_serve_ok_ts is not None else sub.champion_since_ts
        if (sub.status == ST.CHAMPION and since is not None and now - since >= C.CHAMPION_LIVENESS_S
                and sub.champion_rounds_without_ok >= 3 and sub.pay_suspended_since_ts is None):
            sub.pay_suspended_since_ts = now
            ts.log(now, f"champion {sub.key[:12]} award suspension requested: no successful round for 72 h; no strike")

    @staticmethod
    def _retry_at(sub: ST.Submission) -> float:
        # Old state files used last_eval_ts for fetches as well as serves.
        return sub.last_fetch_ts if sub.last_fetch_ts is not None else (sub.last_eval_ts or 0.0)

    async def _prepare_cycle(self, tdef: C.TrackDef, sub: ST.Submission, snap: Snapshot, now: float) -> _Prepared | None:
        if sub.key not in self._cycle_prepared:
            if self._cycle_skips.get(tdef.track_id, 0) >= MAX_SKIPS_PER_CYCLE:
                return None
            p = await self._prepare(tdef, sub, snap, now)
            self._cycle_prepared[sub.key] = p
            if p is None:
                self._cycle_skips[tdef.track_id] = self._cycle_skips.get(tdef.track_id, 0) + 1
        return self._cycle_prepared[sub.key]

    async def _retry_waiting(self, tdef: C.TrackDef, snap: Snapshot, now: float) -> None:
        # Probe waiting entries up to the per-track bound, without downloading. Ordinary 401s spend no skips.
        if tdef.track_id in self._retry_tracks:
            return
        self._retry_tracks.add(tdef.track_id)
        waiting = sorted((s for s in self.state.track(tdef.track_id).submissions.values()
                          if s.status in (ST.QUEUED, ST.PENDING_CONFIRM) and s.reason.startswith(WAITING_FOR_REPO)),
                         key=lambda s: (s.first_seen_block, s.uid, s.key))
        for sub in waiting[:C.WAIT_REPO_PROBES_PER_CYCLE]:
            if self._probes_halted:
                return
            retry_due = now - self._retry_at(sub) >= C.RETRY_UNAVAILABLE_S
            try:
                await self.fetcher.probe(SpecCommitment(sub.track_id, sub.repo, sub.revision))
            except F.FetchError as exc:
                if exc.kind == "infra":
                    # 429/5xx/timeout: the hub is struggling, not the miner. Stop hitting it until the next cycle
                    # (sequential 60 s timeouts would also stall the cycle); no attempt is counted.
                    self._probes_halted = True
                    return
                # Keep the existing deadline/attempt rules and rejection bound; ordinary 401s cost no skip.
                if retry_due:
                    sub.last_fetch_ts = now
                    self._fetch_failure(tdef, sub, snap, now, exc)
                continue
            # Metadata cannot repair a local staging/disk error. Preserve its download retry delay.
            if sub.wait_kind == ST.WAIT_INFRA and not retry_due:
                continue
            sub.reason, sub.wait_kind = "", ""

    async def _run_track(self, tdef: C.TrackDef, snap: Snapshot, now: float) -> None:
        if tdef.track_id == "generalist" and not self._v5_on(tdef.track_id, now) and (
                split.GENERALIST_G_MED is None or not math.isfinite(self._entry_floor(tdef)) or
                not math.isfinite(self.cfg.g_med.get("generalist", math.inf)) or self.cfg.x_tier_mode != "binding"):
            log.warning("generalist waits for release calibration: finite entry floor, frozen G_med and binding mode required")
            return
        ts = self.state.track(tdef.track_id)
        self._reanchor_cochamps(ts, now)
        champ = ts.submissions.get(ts.champion)
        if champ is not None:
            self._champion_liveness(ts, champ, now)
        for s in ts.ordered(ST.PENDING_VERIFY):
            if s.reason == RENEWAL_WAIT:
                s.status, s.reason = ST.QUEUED, ""     # the disk check runs again before every renewal fetch
            elif s.base_id and s.base_model and s.sketch_rule < T.SKETCH_RULE and self.base_provider(s.base_model) is not None:
                # held by an older copy-check rule although the base is here (e.g. rc5: another dtype than the base)
                s.status, s.reason = ST.QUEUED, ""
                ts.log(now, f"{s.key[:12]}: copy-check rule changed, queued for the copy check again")
            elif not s.base_id and s.base_model:
                await self._fetch_base(s)
                if self.base_provider(s.base_model) is not None:
                    s.status, s.reason = ST.QUEUED, ""
                    ts.log(now, f"{s.key[:12]}: declared base model now available, queued for the copy check")
                elif self._base_detail(s):
                    s.reason, s.wait_kind = _no_base(s, self._base_detail(s)), self._base_wait_kind(s)
        for s in list(ts.submissions.values()):
            # admitted by a release whose copy check passed without the base: held at once if the base is missing here
            # (with the base, the check runs before it is served again)
            if s.status in (ST.CHAMPION, ST.PENDING_CONFIRM) and not s.delta_checked:
                await self._fetch_base(s)
                if not s.base_model or self.base_provider(s.base_model) is None:
                    self._hold(ts, s, _no_base(s, self._base_detail(s)), now, self._base_wait_kind(s))
        await self._retry_waiting(tdef, snap, now)
        slot_field = self._finalise_crown(ts, now)
        self._v5_restart_v4_stages(ts, now)    # after the crown: a crowning due at the slot is not re-judged (the field of a slot playoff keeps its stage)
        playoff = slot_field or self._cohort_step(ts, now)
        budget = self._budget(now)
        if budget < 1:
            if slot_field:
                self._playoff_waits.add(tdef.track_id)
            return
        if playoff:
            await self._run_playoff(tdef, ts, playoff, snap, now, budget, slot=bool(slot_field))
            return
        global_budget = budget
        if self._track_cap is not None:
            budget = min(budget, self._track_cap)
        queued = queue_order([s for s in ts.ordered(ST.QUEUED)
                              if not s.reason.startswith(WAITING_FOR_REPO)
                              and (not s.reason.startswith("waiting") or now - self._retry_at(s) >= C.RETRY_UNAVAILABLE_S)], snap)
        queued = self._opening_cohort_first(ts, queued)
        # the opening co-champions (GENERALIST_CO_CHAMPIONS) are served as the bar, never as challengers (a confirmation
        # against the champion could turn a member into not-better); an entry whose crowning waits for its slot
        # (CROWN_SLOTS_UTC) is not challenged again, the other challengers go on against the current bar
        co = {m.key for m in v5_bar_members(ts, now)}
        queued = [s for s in queued if s.key not in co]
        if self.cfg.v3_decisions:
            # v3: an entry in a close call takes its one extra first-stage draw before fresh entries; like the confirmation
            # group, one earlier draw per group (joining the exclusions of different draws could exhaust the pool)
            close = [s for s in queued if s.close_draws]
            if close:
                queued = [s for s in close if s.close_draws[0] == close[0].close_draws[0]] + [s for s in queued if not s.close_draws]
        # A lost first draw cannot be confirmed and must not hold the confirmation queue.
        due = [s for s in ts.ordered(ST.PENDING_CONFIRM)
               if not s.confirm_draw and s.confirm_after_ts is not None and now >= s.confirm_after_ts and s.draw1 in ts.draws
               and not s.reason.startswith(WAITING_FOR_REPO) and _crown_pending(s) is None and s.key not in co
               and not self._review_wait(ts, s, now)]
        if due:
            k1a = {s.key for s in self._k1a_outstanding(ts, now)}
            if k1a:     # K1-a: confirmations due before a running slot first (the slot waits for them)
                due.sort(key=lambda s: s.key not in k1a)
        if due:
            # One first draw per group: joining exclusions from different draws can exhaust the pool.
            due = [s for s in due if s.draw1 == due[0].draw1]
        confirm_mode = bool(due)
        # K1-a at the flip slot (slot rules section 2/3): a confirmation due before the slot whose first stage was measured under v4
        # runs on the slot's evaluation (``slot_eval_ts``, v4 for Monday 26 Oct), never as a v4/v5.1 hybrid
        k1a_v4 = confirm_mode and self._v5_on(tdef.track_id, now) and all(s.key in k1a and not _v5_staged(s) for s in due)
        retry_champ = ts.submissions.get(ts.champion)
        if not due and not queued and not (retry_champ and retry_champ.serve_retry_pending):
            return
        free = self._free_bytes(self.cfg.work_dir)
        if free < self.cfg.min_free_bytes:
            log.warning("track %s: cycle skipped, %.0f GB free on the work directory's filesystem, %.0f GB needed before a "
                        "download", tdef.track_id, free / 1e9, self.cfg.min_free_bytes / 1e9)
            ts.log(now, "infrastructure failure: insufficient free disk before download; nothing counted")
            return

        cands = due if confirm_mode else queued
        champ = ts.submissions.get(ts.champion) if ts.champion else None
        want = budget - (len(v5_bar_members(ts, now)) or (1 if champ is not None else 0))
        if want < 1 and not (champ and champ.serve_retry_pending):
            return
        prepared: list[_Prepared] = []
        ref: list[_Prepared] = []
        parents: dict[str, _Prepared] = {}
        models: dict[str, _Prepared] = {}
        for sub in cands:
            if len(models) >= budget:
                break
            p = await self._prepare_cycle(tdef, sub, snap, now)
            if p is None:
                continue
            if not confirm_mode and sub.carried_from:
                # prepared as queued, restored as pending by the carry-over rule (17 Oct): its confirmation is selected on
                # the next cycle, it never gets a second first draw
                continue
            if champ is not None and champ.status == ST.REJECTED and not ref:
                # this entry's exact check found the champion to be a copy of it (reg-sniping review 20 Oct): the track
                # is vacant from now on, the copy is not served as the incumbent
                ts.log(now, f"champion {champ.key[:12]} rejected: {champ.reason[:100]}")
                ts.champion, champ = "", None
            if champ is not None and not ref:
                incumbent_tried = champ.key in self._cycle_prepared
                cp = await self._prepare_cycle(tdef, champ, snap, now)
                if cp is None:
                    if incumbent_tried or champ.key not in self._cycle_prepared:  # no repeated failure charge
                        return
                    if champ.status == ST.REJECTED:
                        ts.log(now, f"champion {champ.key[:12]} unusable: {champ.reason[:100]}")
                        champ.status, ts.champion = ST.REJECTED, ""
                    return
                ref.append(cp)
                models[champ.key] = cp
                for m in v5_bar_members(ts, now):    # the opening co-champions are the bar: served on the same draw
                    if m is champ:
                        continue
                    mp = await self._prepare_cycle(tdef, m, snap, now)
                    if mp is None:
                        ts.log(now, f"co-champion {m.key[:12]} could not be prepared; group abandoned, re-measure next cycle")
                        return
                    ref.append(mp)
                    models[m.key] = mp
            dependencies = {p.sub.key: p}
            pk = p.sub.derived_from
            par = ts.submissions.get(pk) if pk else None
            if par is not None and par is not champ:
                pp = models.get(pk) or await self._prepare_cycle(tdef, par, snap, now)
                if pp is not None:
                    dependencies[pk] = pp
            if self.cfg.nc_r3:   # R3: the root parents of a derived parent are served too (they must be beaten as well)
                for ak in self._lineage(ts, p.sub.key)[1:]:
                    anc = ts.submissions.get(ak)
                    if anc is None or anc is champ or ak in dependencies:
                        continue
                    ap = models.get(ak) or await self._prepare_cycle(tdef, anc, snap, now)
                    if ap is not None:
                        dependencies[ak] = ap
            group_size = len(models.keys() | dependencies.keys())
            if group_size > budget:
                # Only the first complete comparison can exceed a share, once per cycle. No global overspend.
                # Least-recently-served track order gives the displaced track the next turn.
                if prepared or self._oversize_used or group_size > global_budget:
                    break
                self._oversize_used = True
                budget = group_size
            models.update(dependencies)
            if par is not None and par is not champ and pk in dependencies:
                parents[pk] = dependencies[pk]
            if self.cfg.nc_r3:
                for ak in self._lineage(ts, p.sub.key)[1:]:
                    if ak in dependencies and ak != pk and ts.submissions.get(ak) is not champ:
                        parents[ak] = dependencies[ak]
            prepared.append(p)
        # A queued parent may sort behind its child by immunity. Its serve also fills its challenger role,
        # even if the dependency group filled the budget before the loop reached that queue position.
        prepared = [models[s.key] for s in cands if s.key in models and s is not champ
                    and (confirm_mode or not s.carried_from)]   # critic B1: a carried entry is only a parent here
        if not prepared:
            if not (champ and champ.serve_retry_pending):
                return
            cp = await self._prepare_cycle(tdef, champ, snap, now)
            if cp is None:
                if champ.status == ST.REJECTED:
                    ts.champion = ""
                self._host_failed.add(tdef.track_id)
                return
            ref = [cp]
        # one fresh draw shared by every model of the group
        exclude: set[str] = set()
        if confirm_mode:
            lost = [p for p in prepared if p.sub.draw1 not in ts.draws]
            if lost:   # its first draw was pruned from the state: a disjoint confirmation is impossible, so it waits
                for p in lost:
                    log.error("track %s: %s cannot be confirmed: its first draw %s is no longer in the state",
                              tdef.track_id, p.sub.key[:12], p.sub.draw1)
                    ts.log(now, f"confirmation of {p.sub.key[:12]} waits: first draw {p.sub.draw1} not in the state")
                prepared = [p for p in prepared if p not in lost]
                if not prepared:
                    return
            for p in prepared:
                exclude |= set(ts.draws[p.sub.draw1].item_ids)
        # every earlier draw of an entry in a close call / with a confirmation draw taken (v3). Also with the flag off: the
        # fields stay in the state, and turning v3 off mid-flight must not reuse those items (critic-v3int #2)
        for p in prepared:
            for d in (*p.sub.close_draws, *p.sub.confirm_draws):
                if d in ts.draws:
                    exclude |= set(ts.draws[d].item_ids)
        seed = self.seed_source(tdef.track_id, snap.block)
        v5on = self._v5_on(tdef.track_id, now) and not k1a_v4
        draw_t = self.slot_eval_ts(self._current_slot(now) or now) if k1a_v4 else now
        bases: dict[str, _Prepared] = {}
        if v5on:   # eval v5.1: every challenger's declared base is served on the same items (the co-champions are in ``ref``)
            # the bases are serves too: fewer challengers per group if the hour's budget cannot hold them all
            def serves(ps: list[_Prepared]) -> int:
                return len({p.sub.key for p in [*ref, *parents.values(), *ps]}) + len({p.sub.base_model for p in ps})
            while len(prepared) > 1 and serves(prepared) > max(budget, 1):
                prepared = prepared[:-1]
            if serves(prepared) > global_budget:
                ts.log(now, f"v5.1 group needs {serves(prepared)} serves (incl. declared bases), {global_budget} left this hour: waits")
                return
            bases, no_base = await self._v5_bases(ts, prepared, now)
            if no_base:   # critic F3: an unavailable base holds back only its own challenger
                prepared = [p for p in prepared if p not in no_base]
                if not prepared:
                    self._host_failed.add(tdef.track_id)
                    return
        try:
            if v5on:
                v5pool, v5public = self._v5_pool()
                rows, fresh = V5W.draw(v5pool, v5public, seed=seed, use_counts=ts.used, exclude=exclude, p=self._v5_params())
            else:
                rows, fresh = S.draw(self._pool(tdef.category), n=self._draw_n(tdef.track_id, draw_t), seed=seed, use_counts=ts.used,
                                     exclude=exclude, x_mode=self._x_mode_for(tdef.track_id), x_share=self.cfg.x_tier_share,
                                     generalist=tdef.track_id == "generalist", now=draw_t)
        except S.SampleError as exc:
            log.warning("track %s: no draw this cycle: %s", tdef.track_id, exc)
            ts.log(now, f"no draw: {exc}; refill the sealed pool")
            return
        # A failed incumbent abandons the comparison, including any challenger host probe. Do not charge
        # item uses or prune prior draws until it serves successfully. Attempts still spend the serving budget.
        draw_id = hashlib.sha256(f"{ts.track_id}|{seed}".encode()).hexdigest()[:16]
        flat = V5W.flatten(rows) if v5on else rows    # v5.1: one row per question (the rows a backend serves and the v4 scorer reads)

        def record() -> None:
            if v5on:
                self._record_draw_v5(ts, rows, fresh, seed, snap, now)
            else:
                self._record_draw(ts, self._pool(tdef.category), rows, fresh, seed, snap, now)

        async def serve(p: _Prepared) -> list[SC.ItemScore] | None:
            return await (self._serve_v5(p, flat, draw_id, self.clock()) if v5on else self._serve(p, rows, self.clock()))

        if champ is None:
            record()
        # serve one model at a time
        results: dict[str, list[SC.ItemScore]] = {}
        group = list({p.sub.key: p for p in ref + list(parents.values()) + prepared}.values())
        failed: list[_Prepared] = []
        attempted: list[_Prepared] = []
        champion_failed = False
        for p in group:
            attempted.append(p)
            items = await serve(p)
            if items is None and p.sub is champ:
                first_infra = p.infra_failure
                first_specific = p.model_specific_failure
                first_reason = p.sub.reason
                ts.log(now, f"champion {p.sub.key[:12]} start-up retry on draw {draw_id[:8]}")
                items = await serve(p)
                ts.log(now, f"champion {p.sub.key[:12]} start-up retry {'succeeded' if items is not None else 'failed'}")
                if items is None:
                    if p.preflight_failure:
                        # No second model start took place: retain the first attempt's attribution.
                        p.infra_failure, p.model_specific_failure = first_infra, first_specific
                        p.sub.reason = first_reason
                    else:
                        p.infra_failure = first_infra or p.infra_failure
                        p.model_specific_failure = first_specific and p.model_specific_failure
                    if len(group) == 1 and not p.model_specific_failure:
                        p.infra_failure = True
                    p.sub.serve_retry_pending = True
                    champion_failed = True
            if items is None:
                failed.append(p)
            else:
                results[p.sub.key] = items
            if champion_failed and p.sub is not champ:
                break  # at most one other model as a host probe
        host_problem = len(attempted) > 1 and len(failed) == len(attempted)
        for p in attempted:
            outcome = "ok" if p.sub.key in results else ("infra" if host_problem or p.infra_failure else "model")
            self._serve_round(ts, p.sub, outcome, now, joint=p.joint)
        if host_problem:
            ts.log(now, "no model of the group could be served: host problem, nothing counted")
            self._host_failed.add(tdef.track_id)
            return
        if champ is not None and any(p.sub.key not in results for p in ref):
            ts.log(now, "incumbent retry failed; group abandoned, re-measure next cycle")
            self._host_failed.add(tdef.track_id)
            return
        if v5on and any(p.sub.key not in results for p in ref):
            ts.log(now, "a seated co-champion could not be served: group abandoned, re-measure next cycle")
            self._host_failed.add(tdef.track_id)
            return
        base_served: dict[str, _Prepared] = {}
        if v5on:   # the declared bases, once per (base, draw)
            bad_bases: set[str] = set()
            for bm, bp in bases.items():
                if await serve(bp) is None:
                    ts.log(now, f"declared base {bm[:40]} could not be served: its challengers wait, re-measure next cycle")
                    bad_bases.add(bm)
                    continue
                base_served[bm] = bp
            if bad_bases:   # critic F3: only the challengers of that base wait; the group goes on with the others
                gone = [p for p in prepared if p.sub.base_model in bad_bases]
                for p in gone:
                    self._v5_wait(ts, p.sub, f"declared base {p.sub.base_model[:60]} could not be served on the v5.1 draw", now)
                prepared = [p for p in prepared if p not in gone]
                if not any(p.sub.key in results for p in prepared):
                    self._host_failed.add(tdef.track_id)
                    return
        if champ is not None:
            record()
        try:
            twins = self._behaviour_twins(ts, [p for p in group if p.sub.key in results], prepared, flat, draw_id, now)
        except Exception:   # a measurement bug must not discard the group's scores
            log.exception("track %s: behavioural near-duplicate check failed, no twins this group", tdef.track_id)
            twins = {}
        decided = [p.sub for p in prepared if p.sub.key in results]
        if v5on:
            resp = {p.sub.key: V5W.responses(rows, flat, p.probs) for p in [*group, *base_served.values()]
                    if p.sub.key in results or p.sub.key.startswith("base:")}
            self._v5_ctx = V5W.build_context(draw_id, rows, resp, {c.key: V5W.base_key(c.base_model) for c in decided},
                                             [p.sub.key for p in ref], seed, self._v5_params())
            bad = self._v5_ctx.bad    # critic F3: a model without a finite S_v5.1 is not servable
            if any(p.sub.key in bad for p in ref):
                ts.log(now, "a seated model has no finite S_v5.1 on this draw: group abandoned, re-measure next cycle")
                self._v5_ctx = None
                self._host_failed.add(tdef.track_id)
                return
            if bad:
                def dependencies(c: ST.Submission) -> set[str]:
                    deps = {V5W.base_key(c.base_model), c.key}
                    if c.derived_from:
                        deps.add(c.derived_from)
                    if self.cfg.nc_r3:
                        deps |= set(self._lineage(ts, c.key)[1:])
                    return deps
                for c in [c for c in decided if dependencies(c) & bad]:
                    self._v5_wait(ts, c, "no finite S_v5.1 for it, its base or its parent on the v5.1 draw", now)
                decided = [c for c in decided if not dependencies(c) & bad]
                results = {k: v for k, v in results.items() if k not in bad}
                if not decided:
                    self._v5_ctx = None
                    self._host_failed.add(tdef.track_id)
                    return
        try:
            self._decide(ts, tdef, decided, results, champ, confirm_mode, draw_id, seed, now,
                         twins, paths={p.sub.key: p.path for p in prepared})
        finally:
            self._v5_ctx = None
        if confirm_mode:
            self._cohort_step(ts, now)    # a lone confirmed entry of a complete cohort is crowned in the same cycle

    # ------------------------------------------------------------- eval v5.1 wiring (spec/v5wire.py, docs/EVAL-V5.md)

    def _v5_on(self, track_id: str, now: float) -> bool:
        """The generalist track decides under eval v5.1 from ``EVAL_V5_START_UTC`` on (gate ``None``: never)."""
        return track_id == "generalist" and S.v5_active(now)

    def _v5_params(self) -> "V5W.v5.V5Params":
        """The v5.1 scorer parameters of this cycle: area weights from the cycle's ``v51.config.json`` and the panel stats of its pool
        version (``V5W.active_panel``). Raises ``SampleError`` (fail closed: no draw, nothing decided) when either is not usable."""
        if self._v5_config is None:
            self._v5_config = (V5W.load_config(self.cfg.v5_pool_dir),)     # a refused file raises and is read again next call
        config = self._v5_config[0]
        return V5W.params(w_n=self.cfg.v5_w_n, panel=V5W.active_panel(config, self.cfg.v5_panel), config=config)

    def _v5_pool(self) -> tuple[list[dict], list[dict]]:
        """(T/N rows, public rows) for this cycle. Raises ``SampleError`` when the private pool is not usable (no draw then)."""
        if self._v5_pools is None:
            rows = V5W.load_pool(self.cfg.v5_pool_dir, self._v5_params())
            self._v5_pools = (rows, V5W.public_rows(self._pool("generalist")))
        return self._v5_pools

    async def _v5_bases(self, ts: ST.TrackState, prepared: list[_Prepared], now: float
                        ) -> tuple[dict[str, _Prepared], list[_Prepared]]:
        """(the declared base of each challenger as a servable model, one per distinct base; the challengers whose base is not
        available now). A challenger is never decided without its base on the same items, but one unavailable base only holds
        back its own challenger (critic F3), which waits like a missing base in the copy check. The seated co-champions are
        already in the group (``cochamp_members``, rc11.post1)."""
        bases: dict[str, _Prepared] = {}
        missing: list[_Prepared] = []
        for p in prepared:
            bm = p.sub.base_model
            path = self.base_provider(bm) if bm else None
            if path is None:
                self._v5_wait(ts, p.sub, _no_base(p.sub, self._base_detail(p.sub)), now, self._base_wait_kind(p.sub))
                missing.append(p)
                continue
            if bm not in bases:
                size = sum(n for name, n in F.list_dir(path).items() if name.endswith(".safetensors") and n > 0)
                bases[bm] = _Prepared(_BaseSub(bm, p.sub.track_id), path, size)
        return bases, missing

    @staticmethod
    def _v5_wait(ts: ST.TrackState, sub: ST.Submission, why: str, now: float, kind: str = ST.WAIT_MINER) -> None:
        """One challenger cannot be measured on a v5.1 draw (no base, base not servable, no finite score): it keeps its place and
        stage, the reason says why, and a queued entry is tried again after ``RETRY_UNAVAILABLE_S`` (the ``waiting`` reason)."""
        sub.reason, sub.wait_kind, sub.last_fetch_ts = f"waiting: {why}"[:300], kind, now
        ts.log(now, f"{sub.key[:12]} waits on the v5.1 draw: {why[:140]}")
        log.warning("track %s: %s (uid %d) waits on the v5.1 draw: %s", sub.track_id, sub.key[:12], sub.uid, why)

    async def _serve_v5(self, p: _Prepared, flat: list[dict], draw_id: str, now: float) -> list[SC.ItemScore] | None:
        """``_serve`` with the per-(model, draw) cache: a model needed twice on one draw (a base shared by several
        entries, a co-champion) is served once."""
        # critic F4: the key is the item set, not only the draw id (the same seed can give another set after an abandoned group)
        ckey = (p.sub.key, hashlib.sha256((draw_id + "|" + "|".join(sorted(r["id"] for r in flat))).encode()).hexdigest()[:24])
        hit = self._v5_served.get(ckey)
        if hit is not None and len(hit[1]) == len(flat) and len(hit[0]) == len(flat):
            p.probs, p.joint = list(hit[1]), hit[2]
            return hit[0]
        items = await self._serve(p, flat, now)
        if items is not None:
            self._v5_served[ckey] = (items, list(p.probs), p.joint)
            while len(self._v5_served) > 64:
                self._v5_served.pop(next(iter(self._v5_served)))
        return items

    @staticmethod
    def _v5_row(c: V5W.DrawContext, key: str, ts: ST.TrackState, now: float) -> dict[str, Any]:
        """The v5.1 numbers of one entry on this draw for its history row (published numbers: S_v5.1, T, N, the gap check)."""
        fin = lambda x: x if isinstance(x, float) and math.isfinite(x) else None   # noqa: E731 - the state file holds no Infinity
        p, ref, gap = c.parts[key], c.ref.get(key), c.gap.get(key)
        row: dict[str, Any] = {"scoring_regime": V5W.REGIME, "s_v51": fin(p.s), "t": fin(p.t), "n": fin(p.n), "pub": fin(p.pub)}
        if ref is not None:
            row.update(ref_best=ref.best, ref_totals={k: fin(v) for k, v in ref.totals.items()})
        if gap is not None:
            row.update(gap=fin(gap.g), se_gap=fin(gap.se_g), gap_allowance=fin(gap.allowance), gap_flag=gap.flag)
        return row

    def _record_draw_v5(self, ts: ST.TrackState, rows: list[dict], fresh: bool, seed: int, snap: Snapshot, now: float) -> str:
        draw_id = hashlib.sha256(f"{ts.track_id}|{seed}".encode()).hexdigest()[:16]
        ts.draws[draw_id] = ST.Draw(draw_id, now, snap.block, seed, [r["id"] for r in rows], fresh)
        for r in rows:
            ts.used[r["id"]] = ts.used.get(r["id"], 0) + 1
        self._prune_draws(ts, now)
        pool, public = self._v5_pool()
        p = self._v5_params()
        need = p.t_items + p.n_items
        left = sum(1 for r in pool if ts.used.get(r["id"], 0) < p.mirror_max_uses)
        if left < need:
            log.warning("track %s: v5.1 refill needed: %d T/N items below %d uses, a draw needs %d", ts.track_id, left,
                        p.mirror_max_uses, need)
            ts.log(now, f"v5.1 refill needed: {left} T/N items below {p.mirror_max_uses} uses")
        ST.save(self.state, self.cfg.state_path)
        return draw_id

    def _v5_restart_v4_stages(self, ts: ST.TrackState, now: float) -> None:
        """Transition rule A (Florian, 22 Oct): from ``EVAL_V5_START_UTC`` on, an entry whose first stage was measured under v4 and
        that has not confirmed (and has no stored crowning) restarts as a v5.1 first stage: same treatment for all, no v4
        confirmation after the flip. It runs after ``_finalise_crown`` (a crowning due at the slot goes first and is not
        re-judged). Also a vacant-track confirmation taken under v4 that waits for its cohort playoff (rc11.post5): no v4 crowning
        after the flip. Idempotent; the opening co-champion bar and the champion are no challengers. Gate off: no-op."""
        if not self._v5_on(ts.track_id, now):
            return
        bar = {m.key for m in v5_bar_members(ts, now)}
        k1a = {s.key for s in self._k1a_outstanding(ts, now)}    # K1-a: their v4 confirmation runs first (slot rules section 2)
        for s in ts.submissions.values():
            if s.key in bar or s.key in k1a or s.status not in (ST.PENDING_CONFIRM, ST.QUEUED) or _crown_pending(s) is not None:
                continue
            if s.status == ST.PENDING_CONFIRM:
                if _v5_staged(s) or (s.confirm_draw and any(
                        h.get("kind") == "confirm" and h.get("draw") == s.confirm_draw and h.get("scoring_regime") == V5W.REGIME
                        for h in s.history)):
                    continue    # a v5.1 stage / a vacant-track confirmation taken on v5.1 stays
                # rc11.post5 (#25186): a vacant-track confirmation taken under v4 that still waits for its cohort playoff is no
                # crowning after the flip (rule A, the cohort playoff is not wired for v5.1): it restarts as a v5.1 first stage
            elif not s.close_draws or _v5_staged(s):    # queued: only a v4 close call carries stage state
                continue
            self._v5_restart(ts, s, now, "first stage taken under v4 did not confirm before the v5.1 start",
                             "v4 stage unconfirmed at the flip")

    @staticmethod
    def _v5_restart(ts: ST.TrackState, s: ST.Submission, now: float, why: str, short: str) -> None:
        """Rule A: the entry's v4 stage is dropped and it is queued for a v5.1 first stage (no recommit needed)."""
        was = s.status
        s.status, s.reason = ST.QUEUED, ""
        s.delta = s.first_gain = s.first_pass_ts = s.confirm_after_ts = s.confirm_score = None
        s.draw1, s.confirm_draw, s.first_incumbent, s.decision_rule = "", "", None, ""
        s.close_draws, s.confirm_draws = [], []
        s.history.append({"kind": "v5_restart", "ts": now, "was": was})
        ts.log(now, f"{s.key[:12]}: {why}; restarts as a v5.1 first stage")
        log.info("track %s: %s restarts as a v5.1 first stage (%s)", ts.track_id, s.key[:12], short)

    def _margin_now(self, track_id: str, now: float) -> float | None:
        """The time the margin rules read (rules v4 floor, ``M.margin_floor``): the decision time on the generalist track,
        ``None`` (today's floor) on the specialist tracks, which keep their published 1.5 margin floor."""
        return now if track_id == "generalist" else None

    @staticmethod
    def _v4_draw_now(now: float) -> float:
        """The time a slot playoff's v4 draw is made at (rule A, rc11.post2): the playoff of the flip slot decides under the
        evaluation that earned the titles, i.e. v4, although the gate opens at that very slot, and ``S.draw`` is a v5 draw
        from the gate on. One second before the gate while it is open; ``now`` otherwise (gate off or not reached)."""
        return min(now, S.EVAL_V5_START_UTC.timestamp() - 1.0) if S.v5_active(now) else now

    def _draw_n(self, track_id: str, now: float) -> int:
        """Items of a draw of this track made at ``now`` (generalist: 10 x ``S.generalist_per_category(now)``)."""
        return S.draw_size(self.cfg.sample_size, track_id == "generalist", now)

    def _record_draw(self, ts: ST.TrackState, pool: list[dict], rows: list[dict], fresh: bool, seed: int, snap: Snapshot, now: float) -> str:
        draw_id = hashlib.sha256(f"{ts.track_id}|{seed}".encode()).hexdigest()[:16]
        ts.draws[draw_id] = ST.Draw(draw_id, now, snap.block, seed, [r["id"] for r in rows], fresh)
        if ts.track_id == "generalist":
            categories = {r["id"]: r["category"] for r in rows}
            ts.draws[draw_id].item_categories = categories
            ts.item_categories.update(categories)
            shortfalls = S.generalist_shortfalls(rows, self._x_mode_for(ts.track_id), self.cfg.x_tier_share,
                                                 S.generalist_per_category(now))
            ts.draws[draw_id].x_shortfall = {c: counts[S.X_TIER] for c, counts in shortfalls.items()
                                              if counts.get(S.X_TIER, 0)}
        for r in rows:
            ts.used[r["id"]] = ts.used.get(r["id"], 0) + 1
        self._prune_draws(ts, now)
        # reuse is normal (ITEM_MAX_USES); the only refill trigger is too few non-proc uses left (REFILL_DRAWS draws' worth)
        size = self._draw_n(ts.track_id, now)   # the draw's own size: 900 for a generalist draw from DRAWS_V4_START_UTC on
        left = S.nonproc_draws_left(pool, ts.used, size)
        if left < S.REFILL_DRAWS:
            items = S.eligible_count(pool, ts.used)
            log.warning("track %s: refill needed: about %.0f draws of %d left in the LLM-written items (fewer than %d); "
                        "%d sealed items below %d uses", ts.track_id, left, size, S.REFILL_DRAWS, items,
                        S.ITEM_MAX_USES)
            ts.log(now, f"refill needed: about {left:.0f} draws left, {items} sealed items below {S.ITEM_MAX_USES} uses")
        self._x_runway(ts, pool, rows, now)
        ST.save(self.state, self.cfg.state_path)
        return draw_id

    def _x_runway(self, ts: ST.TrackState, pool: list[dict], rows: list[dict], now: float) -> None:
        """Cross-author items are counted apart from the LLM-written tiers (their own runway alarm). In binding mode a
        draw that got fewer of them than its share is logged: the normal tiers filled the rest."""
        mode = self._x_mode_for(ts.track_id)
        if mode == "off":
            return
        x_rows = sum(1 for r in pool if r.get("tier") in S.X_TIERS)
        if not x_rows and mode != "binding":
            return
        size = self._draw_n(ts.track_id, now)
        want = int(round(size * self.cfg.x_tier_share))
        got = sum(1 for r in rows if r.get("tier") in S.X_TIERS)
        if mode == "binding" and got < want:
            log.warning("track %s: cross-author shortage: %d of %d items drawn (%d in the pool); the normal tiers filled the rest",
                        ts.track_id, got, want, x_rows)
            ts.log(now, f"cross-author shortage: {got} of {want} items drawn ({x_rows} in the pool), the normal tiers filled the rest")
        left = S.x_draws_left(pool, ts.used, size, self.cfg.x_tier_share)
        if left < S.REFILL_DRAWS:
            log.warning("track %s: cross-author refill needed: about %.0f draws of %d X items left (fewer than %d); %d X items "
                        "in the pool", ts.track_id, left, want, S.REFILL_DRAWS, x_rows)
            ts.log(now, f"cross-author refill needed: about {left:.0f} draws left, {x_rows} X items in the pool")

    @staticmethod
    def _prune_draws(ts: ST.TrackState, now: float | None = None) -> None:
        """Keep every draw a pending entry still needs (its first draw, which its confirmation must exclude, and the draw
        it passed its confirmation on) and the newest ``MAX_UNREFERENCED_DRAWS`` others (1.2.0rc7; before, the 40 newest
        draws, so a pending entry could lose its first draw and never be confirmed)."""
        now = time.time() if now is None else now
        # A pruned first pass must survive the recommit window so confirmation can exclude its items.
        recoverable = [s for s in ts.submissions.values() if _carryover_waiting(s)
                       and s.withdrawn_kind in ("", "pruned")
                       and now <= _carryover_end(s) + CARRYOVER_DRAW_GRACE_S]
        needed = {d for s in ts.submissions.values()
                  if s.status == ST.PENDING_CONFIRM or s in recoverable or (s.status == ST.QUEUED and s.close_draws)
                  for d in (s.draw1, s.confirm_draw, *s.close_draws, *s.confirm_draws) if d}
        rest = [d for d in sorted(ts.draws.values(), key=lambda d: d.ts) if d.draw_id not in needed]
        for old in rest[:-MAX_UNREFERENCED_DRAWS]:
            ts.draws.pop(old.draw_id, None)
        for sub in ts.submissions.values():
            sub.category_draws = {d: scores for d, scores in sub.category_draws.items() if d in ts.draws}

    def _behaviour_twins(self, ts: ST.TrackState, served: list[_Prepared], challengers: list[_Prepared], rows: list[dict],
                         draw_id: str, now: float) -> dict[str, list[str]]:
        """For each challenger: the earlier submissions of the group (incumbent, parents, other challengers) whose answers
        on this draw it shares (``BehaviourStats.twin``). Every measured pair is kept in the challenger's history, so the
        state holds the real noise for a later calibration. A twin must be beaten by the full margin; the first one
        becomes ``derived_from`` if the weights did not name a parent."""
        out: dict[str, list[str]] = {}
        keep = None   # shadow cross-author items (stage 2) take no part in a decision: not in the twin check either
        if self._x_mode_for(ts.track_id) == "shadow" and any(r.get("tier") in S.X_TIERS for r in rows):
            keep = [i for i, r in enumerate(rows) if r.get("tier") not in S.X_TIERS]
            rows = [rows[i] for i in keep]

        def probs_of(p: _Prepared) -> Sequence[Any]:
            return p.probs if keep is None else [p.probs[i] for i in keep]

        for c in challengers:
            if c.sub.key not in {p.sub.key for p in served}:
                continue
            measured: dict[str, dict[str, float]] = {}
            for o in served:
                if o.sub is c.sub or not self._earlier(o.sub, c.sub):
                    continue
                st = behaviour_stats(rows, probs_of(o), probs_of(c), temp_invariant=self.cfg.nc_r2, rel_tie=self.cfg.nc_r2)
                measured[o.sub.key] = st.as_dict()
                if st.twin():
                    out.setdefault(c.sub.key, []).append(o.sub.key)
                    ts.log(now, f"{c.sub.key[:12]} answers like {o.sub.key[:12]} (jsd {st.jsd_bits:.4f} bits, same order "
                                f"{st.same_order:.1%} of {st.items})")
            if measured:
                c.sub.history.append({"draw": draw_id, "kind": "behaviour", "vs": measured, "twins": out.get(c.sub.key, [])})
            if out.get(c.sub.key) and not c.sub.derived_from:
                c.sub.derived_from, c.sub.derived_how = out[c.sub.key][0], "behaviour"
        return out

    # ----------------------------------------------------- decisions (pure state updates)

    @staticmethod
    def _first_incumbent(ch: ST.Submission) -> str | None:
        """Key of the incumbent the first pass was measured against ("" = vacant track). For a state written before the
        field existed, the first-pass history still tells whether the track was vacant; otherwise None (unknown)."""
        if ch.first_incumbent is not None:
            return ch.first_incumbent
        first = next((h for h in ch.history if h.get("kind") == "first" and h.get("draw") == ch.draw1), None)
        return "" if first is not None and first.get("incumbent") is None else None

    @staticmethod
    def _crown_slot(now: float) -> float | None:
        """CROWN_SLOTS_UTC: the first slot instant at or after ``now`` (unix seconds); None = no slots configured or none
        left (a decision crowns at once, as before)."""
        later = [t.timestamp() for t in split.CROWN_SLOTS_UTC if t.timestamp() >= now]
        return min(later) if later else None

    @staticmethod
    def _slot_of(s: ST.Submission) -> float:
        """The slot instant of a stored crowning: the first slot instant at or after the decision, or (K1-a) the slot the
        row names; the decision time itself after the last slot. Call only while ``_crown_pending(s)`` is set."""
        rows = [h for h in s.history if h.get("kind") == "crown_pending"]
        at = float(rows[-1]["ts"])
        if rows[-1].get("k1a") and isinstance(rows[-1].get("slot"), (int, float)):
            return float(rows[-1]["slot"])
        later = [t.timestamp() for t in split.CROWN_SLOTS_UTC if t.timestamp() >= at]
        return min(later) if later else at

    @staticmethod
    def _current_slot(now: float) -> float | None:
        """The slot whose clock is running at ``now`` (slot <= now < slot + SLOT_CLOCK_S, the latest such); None = none."""
        open_ = [t.timestamp() for t in split.CROWN_SLOTS_UTC if t.timestamp() <= now < t.timestamp() + split.SLOT_CLOCK_S]
        return max(open_) if open_ else None

    def _k1a_outstanding(self, ts: ST.TrackState, now: float) -> list[ST.Submission]:
        """K1-a (slot rules section 2): entries whose confirmation was due (``confirm_after_ts`` = first pass + the
        confirmation delay) at or before slot - SLOT_K1_DUE_LEAD_S and has not run yet. While any is left (and the slot's
        clock runs and the slot is not decided), the slot waits and these confirmations run first (``_run_track``); one
        that passes joins the slot (``_k1a_slot``). An entry waiting for its repo on the miner's side does not hold the
        slot; one waiting because of our fetch (``WAIT_INFRA``) does, up to the slot clock. ``[]`` with K1-b."""
        slot = self._current_slot(now)
        if slot is None or not split.SLOT_K1_A or any(r.get("slot") == slot for r in ts.slot_results):
            return []
        co = {m.key for m in cochamp_members(ts)}
        return [s for s in ts.ordered(ST.PENDING_CONFIRM)
                if s.key not in co and _crown_pending(s) is None and not s.confirm_draw and s.confirm_after_ts is not None
                and s.confirm_after_ts <= slot - split.SLOT_K1_DUE_LEAD_S and s.draw1 in ts.draws
                and not (s.reason.startswith(WAITING_FOR_REPO) and s.wait_kind != ST.WAIT_INFRA)
                and not RV.under_review(s)]    # rc11.post5: an entry under review does not hold the slot

    def _k1a_slot(self, ts: ST.TrackState, ch: ST.Submission, now: float) -> float | None:
        """The slot a confirmation passing at ``now`` joins under K1-a (its slot has started, is not decided, and the
        confirmation was due by slot - SLOT_K1_DUE_LEAD_S); None = the ordinary rule (the next slot)."""
        slot = self._current_slot(now)
        if slot is None or not split.SLOT_K1_A or now <= slot or any(r.get("slot") == slot for r in ts.slot_results):
            return None
        due = ch.confirm_after_ts
        return slot if due is not None and due <= slot - split.SLOT_K1_DUE_LEAD_S else None

    def _slot_unserved(self, ts: ST.TrackState, group: Sequence[ST.Submission], slot_ts: float, now: float) -> None:
        """No entry of the slot could be loaded, each for a reason on its own side (rule 4.1): nobody is crowned, the entries
        go back to the challenger path, the slot is recorded (rc11.post3; critic F2 of rc11.post2: it no longer blocks the
        track)."""
        left = {s.key: {"round": 1, "rule": "4.1", "why": f"not loaded: {s.reason[:80]}"} for s in group}
        for s in group:
            s.history.append({"kind": "crown_done", "ts": now})
            s.confirm_draw, s.confirm_score = "", None
            ts.log(now, f"{s.key[:12]} back on the challenger path: in the slot, not loaded (its own side)")
        ts.log(now, "slot: nobody could be served; nobody crowned, the current places continue")
        self._slot_record(ts, group, [], [], {}, {}, [], "none", left, {}, slot_ts, self.slot_eval_ts(slot_ts), 0, "", now)

    def _slot_due(self, ts: ST.TrackState, now: float) -> list[ST.Submission]:
        """CROWN_SLOTS_UTC: the entries whose stored crowning has reached its slot (``_slot_of``), restricted to the
        earliest such slot; best stored score first. They are one group: all of them earned the title against the bar,
        the slot decides among them."""
        due: list[tuple[float, ST.Submission]] = []
        for s in ts.submissions.values():
            at = _crown_pending(s)
            if at is None or RV.under_review(s):
                continue    # rc11.post5: a stored crowning of an entry under review waits (it is never crowned while held)
            slot = self._slot_of(s)
            if now >= slot:
                due.append((slot, s))
        if not due:
            return []
        first = min(sl for sl, _ in due)
        return sorted((s for sl, s in due if sl == first), key=lambda s: (-(s.score or 0.0), s.first_seen_block, s.uid, s.key))

    @staticmethod
    def _retire_champions(ts: ST.TrackState, current: ST.Submission | None) -> None:
        """A challenger takes the title: the champion becomes a former champion and, while the generalist opening
        co-champion set is in force (GENERALIST_CO_CHAMPIONS), every listed co-champion with it (rc11.post2: the set is
        one former champion, the validator pays it as one). Call before ``ts.champion`` changes."""
        members = cochamp_members(ts)
        if current is not None:
            current.status = ST.FORMER
        for m in members:
            m.status = ST.FORMER
        if ts.cochamps:     # a shared group crowned at a slot (rc11.post3): ONE former champion from now on
            ts.former_sets.append({"keys": list(ts.cochamps), "since": ts.cochamps_since})
            ts.cochamps, ts.cochamps_since = [], None

    @staticmethod
    def _reanchor_cochamps(ts: ST.TrackState, now: float) -> None:
        """CRITERIA 6 (rc11.post4): a shared group crowned at a slot (``TrackState.cochamps``) stays the champion when its
        leader withdraws or is rejected while other members remain. The track lost its champion seat (``ts.champion`` is
        empty), so the first remaining member in the group's order takes the seat: the champion machinery (the bar of every
        challenger draw, liveness, the slot's retirement) follows it. The group keeps its keys, so each member's share stays
        1/n of the crowned group and the part of a member who left goes to inference. The last member gone: the title is
        vacant and the set ends. Idempotent; no-op without a dynamic set or while the seat is held."""
        if not getattr(ts, "cochamps", None) or ts.champion:
            return
        members = cochamp_members(ts)
        if not members:
            ts.log(now, "shared group: the last member has left; the title is vacant")
            log.info("track %s: shared group ended, no member left; the title is vacant", ts.track_id)
            ts.cochamps, ts.cochamps_since = [], None
            return
        new = members[0]
        new.status, new.reason = ST.CHAMPION, ""
        new.champion_since_ts = ts.cochamps_since if ts.cochamps_since is not None else now
        new.history.append({"kind": "cochamp_reanchor", "ts": now})
        ts.champion = new.key
        ts.log(now, f"shared group: its leader left; {new.key[:12]} (uid {new.uid}) holds the group's champion seat, "
                    f"{len(members)} of {len(ts.cochamps)} members remain")
        log.info("track %s: shared group re-anchored on %s (uid %d)", ts.track_id, new.key[:12], new.uid)

    def _finalise_crown(self, ts: ST.TrackState, now: float) -> list[ST.Submission]:
        """CROWN_SLOTS_UTC: decide the entries whose stored crowning has reached its slot. Exactly one: crowned as earned.
        Several: returned as the slot playoff's field (``_run_playoff(slot=True)``), nothing changes yet. Entries that were
        withdrawn or rejected meanwhile are dropped (``_crown_pending``). Idempotent; ``[]`` without slots."""
        due = self._slot_due(ts, now)
        if not due:
            return []
        if self._k1a_outstanding(ts, now):
            log.info("track %s: slot waits for the K1-a confirmations due before it", ts.track_id)
            return []
        if len(due) > 1:
            return due
        win = due[0]
        slot_ts = self._slot_of(win)
        current = ts.submissions.get(ts.champion)
        self._retire_champions(ts, current if current is not None and current.status == ST.CHAMPION else None)
        win.status, win.reason, win.champion_since_ts = ST.CHAMPION, "", now
        win.history.append({"kind": "crown_done", "ts": now})
        ts.champion = win.key
        ts.log(now, f"{win.key[:12]} is the new champion ({win.score or 0.0:.2f}) (crown slot)")
        self._release_pending_crowns(ts, now)
        self._slot_record(ts, [win], [], [], {}, {}, [win], "single_entry", {}, {}, slot_ts, self.slot_eval_ts(slot_ts), 0, "", now)
        return []

    @classmethod
    def _release_pending_crowns(cls, ts: ST.TrackState, now: float) -> None:
        """A crowning ended the wait of every other stored decision (a later slot's, or one the playoff could not serve):
        slot rules section 2 (rc11.post4), the entry has to beat the new champion, starting with a new first draw. The same
        for an entry whose stage was taken against the former title holders and has not confirmed yet (first pass taken,
        or a close call waiting for its extra draw; FAQ: "an entry that is not in the slot starts again with a new first
        draw against it"). The new title holders are not touched."""
        holders = {m.key for m in cochamp_members(ts)} | {ts.champion}
        for s in ts.ordered(ST.PENDING_CONFIRM):
            if s.key in holders:
                continue
            if _crown_pending(s) is not None:
                s.history.append({"kind": "crown_done", "ts": now})
                cls._slot_restart(ts, s, now, "a new champion was crowned at the slot: a new first draw against it")
            elif s.draw1:
                cls._slot_restart(ts, s, now, "a new champion was crowned at the slot before this entry confirmed: "
                                              "a new first draw against it")
        for s in ts.ordered(ST.QUEUED):
            if s.close_draws and s.key not in holders:
                cls._slot_restart(ts, s, now, "a new champion was crowned at the slot during this close call: "
                                              "a new first draw against it")

    @staticmethod
    def _slot_restart(ts: ST.TrackState, s: ST.Submission, now: float, why: str) -> None:
        """Slot rules section 2 (rc11.post4): after a slot crowned a new champion, an entry of that slot that was not
        crowned and every entry stored for a later slot is queued for a new first draw against the new title holders (its
        stage is dropped as for a requeue; history, commitment order and checks stay; no recommit). Queued without a stage, it is
        never restarted a second time by rule A (``_v5_restart_v4_stages``); its new first draw is on the evaluation in force."""
        was = s.status
        s.status, s.reason = ST.QUEUED, why
        s.delta = s.first_gain = s.first_pass_ts = s.confirm_after_ts = s.confirm_score = None
        s.draw1, s.confirm_draw, s.first_incumbent, s.decision_rule = "", "", None, ""
        s.close_draws, s.confirm_draws = [], []
        s.history.append({"kind": "slot_restart", "ts": now, "was": was})
        ts.log(now, f"{s.key[:12]} back in the queue: {why}")
        log.info("track %s: %s queued for a new first draw (new champion at the slot)", ts.track_id, s.key[:12])

    @staticmethod
    def _release_slot_field(ts: ST.TrackState, field: Sequence[ST.Submission], now: float) -> None:
        """A slot field that cannot be fetched at all goes back to the queued challenger path, ready for a later draw."""
        for s in field:
            if s.status != ST.PENDING_CONFIRM or _crown_pending(s) is None:
                continue
            s.history.append({"kind": "crown_done", "ts": now})
            s.status, s.reason = ST.QUEUED, "slot playoff field unavailable: returned to the challenger queue"
            s.delta = s.first_gain = s.first_pass_ts = s.confirm_after_ts = s.confirm_score = None
            s.draw1, s.confirm_draw, s.first_incumbent, s.decision_rule = "", "", None, ""
            s.close_draws, s.confirm_draws = [], []
            ts.log(now, f"{s.key[:12]} slot playoff unavailable: returned to the challenger queue")

    def _decide(self, ts: ST.TrackState, tdef: C.TrackDef, challengers: list[ST.Submission], results: dict[str, list[SC.ItemScore]],
                champ: ST.Submission | None, confirm_mode: bool, draw_id: str, seed: int, now: float,
                twins: Mapping[str, Sequence[str]] | None = None, paths: Mapping[str, str] | None = None) -> None:
        """``twins``: per challenger, the earlier submissions it answers like on this draw (``_behaviour_twins``).
        ``paths``: the served model directories (near-champion rule: a challenger on another base is sketched again)."""
        v5c = self._v5_ctx      # eval v5.1: S_v5.1 replaces the v4 score; the gap penalty is retired (never applied)
        if tdef.track_id == "generalist" and v5c is None:
            for key, items in results.items():
                self._category_diagnostics(ts.submissions[key], ts.draws[draw_id], items)
        twins = twins or {}
        v3 = self.cfg.v3_decisions
        # cross-author items (stage 2 plumbing, off unless the pool has the tier and X_TIER_MODE says so): scored apart; in
        # "shadow" mode they are left out of every score, margin and decision below. In "binding" mode (1.2.0rc9.post3)
        # they are part of the draw and of the score, and every score below is S_eff = S - P (the capped gap penalty,
        # ``_binding_penalty``): the floor, the margins, pooled evidence, the confirmation and the displayed score.
        mode = "off" if v5c is not None else self._x_mode_for(tdef.track_id)
        binding = mode == "binding"
        xs: dict[str, dict[str, Any]] = {}
        if mode != "off":
            xs = self._x_stats(results, seed, se=binding)
            if xs and mode == "shadow":
                results = {k: (xs[k]["gen"] if k in xs else v) for k, v in results.items()}
        total = ({k: v5c.parts[k].s for k in results} if v5c is not None
                 else {k: SC.score_items(v).total for k, v in results.items()})
        entry_floor = -math.inf if v5c is not None else self._entry_floor(tdef)   # v5.1: the reference (co-champions, base) is the bar
        pen: dict[str, float] = {}
        pdet: dict[str, dict[str, Any]] = {}
        if binding:
            pen, pdet = self._binding_penalties(ts, tdef.track_id, results, [(draw_id, xs)])
            raw = total
            total = {k: raw[k] - pen[k] for k in raw}
            for k in raw:
                pdet[k].update(raw=raw[k], s_eff=total[k])
        tiers = {k: nonproc_scores(v) for k, v in results.items()}

        def scored(s: ST.Submission, parent: bool = False) -> None:
            s.score = total[s.key]
            if v5c is not None:   # S_v5.1 is the only score; T and N are the per-part monitoring numbers
                if parent:
                    return   # critic F2: a parent served as a reference only is not marked v5 (no floor exemption from it)
                s.nonproc, s.by_tier = total[s.key], {"T": v5c.parts[s.key].t, "N": v5c.parts[s.key].n}   # by_tier {T, N} marks v5
                return
            np_ = tiers[s.key][0]
            s.nonproc, s.by_tier = (np_ if np_ is None or not pen else np_ - pen[s.key]), dict(tiers[s.key][1])

        x_gen = S.x_pool_gen(self._pool(tdef.category)) if xs else ""

        def xrec(key: str) -> dict[str, Any]:
            out: dict[str, Any] = {"s_gen": xs[key]["s_gen"], "s_x": xs[key]["s_x"], "x_gen": x_gen} if key in xs else {}
            if binding:
                if key in xs:
                    out.update(gap=xs[key]["s_gen"] - xs[key]["s_x"], se_gap=xs[key]["se"])
                out["gap_binding"] = dict(pdet[key])
            return out

        if v5c is not None:
            line = f"draw {draw_id[:8]} S_v5.1 (T/N): " + ", ".join(
                f"{k[:12]} {p.s:.2f} ({p.t:.2f}/{p.n:.2f})" for k, p in v5c.parts.items())
            ts.log(now, line)
            log.info("track %s: %s", tdef.track_id, line)
            for k, g in v5c.gap.items():
                if g.flag:
                    ts.log(now, f"{k[:12]} gap check flagged: G {g.g:+.2f} > A {g.allowance:.2f}; overlap check before award or keep-alive promotion")
        line = f"draw {draw_id[:8]} non-proc: " + ", ".join(
            f"{k[:12]} {'-' if np_ is None else f'{np_:.2f}'}" for k, (np_, _) in tiers.items())
        ts.log(now, line)
        log.info("track %s: %s", tdef.track_id, line)
        if binding:
            line = f"draw {draw_id[:8]} S_eff (S - gap penalty): " + ", ".join(
                f"{k[:12]} {pdet[k]['raw']:.2f}-{pen[k]:.2f}={total[k]:.2f}" for k in total)
            ts.log(now, line)
            log.info("track %s: %s", tdef.track_id, line)
        if champ is not None:
            scored(champ)
            champ.last_eval_ts = now
            if champ.key in xs:   # the incumbent's own gap on this draw (the reference for the others' G_med)
                champ.history.append({"draw": draw_id, "kind": "incumbent", **xrec(champ.key)})
        # opening co-champions (GENERALIST_CO_CHAMPIONS): all of them were served on this draw, the bar of a challenger is the
        # best of their scores on it (as long as the title holder is still the champion at the start of the decision)
        co_bar = [m for m in v5_bar_members(ts, now) if m.key in results] if champ is not None and champ.key in results else []
        if len(co_bar) < 2:
            co_bar = []
        for m in co_bar:
            if m is not champ:
                scored(m)
                m.last_eval_ts = now
                if m.key in xs:
                    m.history.append({"draw": draw_id, "kind": "incumbent", **xrec(m.key)})
        if co_bar:
            line = f"draw {draw_id[:8]} co-champions: " + ", ".join(
                f"uid {m.uid} {total[m.key]:.2f}" for m in co_bar) + f"; bar = best co-champion uid {max(co_bar, key=lambda m: total[m.key]).uid}"
            ts.log(now, line)
            log.info("track %s: %s", tdef.track_id, line)
        parent_keys = {c.derived_from for c in challengers if c.derived_from}
        if self.cfg.nc_r3:
            parent_keys |= {ak for c in challengers for ak in self._lineage(ts, c.key)[1:]}
        for pk in parent_keys:
            if (pk in results and pk in ts.submissions and (champ is None or pk != champ.key)
                    and ts.submissions[pk].status != ST.WITHDRAWN):   # critic C2: keep the score a carry-over restores
                scored(ts.submissions[pk], parent=True)
        current = champ
        promoted = False
        for ch in challengers:
            scored(ch)
            ch.last_eval_ts = now
            inc_items = results.get(current.key) if current else None
            inc_total = total[current.key] if current else None
            inc_key = current.key if current else ""   # also the pooling label of the stage rows when a co-champion is the bar
            bar: ST.Submission | None = None
            ref_extended = False
            if co_bar and current is champ:
                # the best co-champion on THIS draw is the incumbent of the challenger's gain, SE and margin; ties: the champion
                bar = max(co_bar, key=lambda m: (total[m.key], m is champ))
                inc_items, inc_total = results[bar.key], total[bar.key]
            if v5c is not None:   # D3: the best of the seated co-champions and the declared base, on the same items
                ref5 = v5c.ref[ch.key]
                se, inc_total, inc_key = ref5.se, ref5.total, ref5.key
                bar = next((m for m in co_bar if m.key == ref5.best), None)   # None: the declared base is the bar on this draw
                if promoted and current is not None and current.key in total and current.key != ref5.best:
                    # critic F1: an entry crowned earlier in this loop is part of the bar of every later challenger of the group
                    # (it beat the reference on this very draw); the stage rows are labelled with the extended reference so that
                    # they are never pooled with draws against another one
                    if total[current.key] >= inc_total:
                        se, inc_total, bar = v5c.pair_se(ch.key, current.key), total[current.key], current
                    inc_key = f"{ref5.key}|{current.key}"
                    ref_extended = True
            else:
                se = SC.paired_se(results[ch.key], inc_items, seed) if inc_items is not None else 0.0
            have_inc = inc_total is not None
            gain = ch.score - inc_total if inc_total is not None else None
            z = M.z_of(gain, se) if gain is not None else None
            if bar is not None:
                line = (f"{ch.key[:12]} vs best co-champion uid {bar.uid} ({bar.key[:12]}): {ch.score:.2f} vs bar {inc_total:.2f}, "
                        f"gain {gain:+.2f}, SE {se:.2f}, z {z:.2f}")
                ts.log(now, line)
                log.info("track %s: %s", tdef.track_id, line)
            parent_ok, parent_gain = True, None
            # the weight parent (unless it is the incumbent, which the margin already covers) and every behavioural twin,
            # the incumbent included: the full margin over each, also on the confirmation draw
            must_beat = [k for k in twins.get(ch.key, ()) if k in results]
            par = ts.submissions.get(ch.derived_from) if ch.derived_from else None
            if par is not None and par.key in results and par.key not in must_beat and (
                    self.cfg.nc_r1 or par.key != (current.key if current else "")):   # R1: the incumbent parent counts too
                must_beat.append(par.key)
            if self.cfg.nc_r3:   # R3: an entry derived from a derived entry inherits the root parents
                must_beat += [k for k in self._lineage(ts, ch.key)[1:] if k in results and k not in must_beat]
            comp_txt = ""
            for pk in must_beat:
                if self.cfg.nc_r1:
                    pse, cse = SC.paired_se_parts(results[ch.key], results[pk], seed)
                    if v5c is not None:
                        pse = v5c.pair_se(ch.key, pk)
                    cgain = SC.score_items(results[ch.key]).competence - SC.score_items(results[pk]).competence
                    if cgain < max(TWIN_COMP_FLOOR, M.SE_MULTIPLE * cse):
                        parent_ok = False
                        comp_txt = f"; does not beat {pk[:12]} on competence ({cgain:+.2f})"
                    ch.history.append({"draw": draw_id, "kind": "comp_gate", "vs": pk, "comp_gain": round(cgain, 4), "comp_se": round(cse, 4)})
                else:
                    pse = v5c.pair_se(ch.key, pk) if v5c is not None else SC.paired_se(results[ch.key], results[pk], seed)
                gain_p = total[ch.key] - total[pk]
                parent_gain = gain_p if parent_gain is None else min(parent_gain, gain_p)
                parent_ok = parent_ok and gain_p >= M.required_margin(pse, self._margin_now(ts.track_id, now))
            parent_txt = "" if parent_ok else f"; does not beat its parent ({parent_gain:+.2f})" + comp_txt
            if self.cfg.nc_near_champ and current is not None and inc_items is not None and gain is not None:
                near = self._near_champ(ch, current, (paths or {}).get(ch.key, ""))
                if near is None:
                    ch.history.append({"draw": draw_id, "kind": "near_champ", "vs": current.key, "comparable": False})
                elif self._is_near(near):
                    n_ch = sum(it.valid and it.top_hit for it in results[ch.key])
                    n_inc = sum(it.valid and it.top_hit for it in results[current.key])   # the champion's own items (its weights were compared)
                    need = max(M.required_margin(se, self._margin_now(ts.track_id, now)), NEAR_CHAMP_FLOOR)
                    near_ok = n_ch > n_inc and gain >= need
                    ch.history.append({"draw": draw_id, "kind": "near_champ", "vs": current.key, "cos": round(near.cosine, 4),
                                       "rel": round(near.relative_l2, 4), "alpha": round(near.alpha, 4), "correct": n_ch, "inc_correct": n_inc,
                                       "gain": gain, "need": need, "ok": near_ok})
                    if not near_ok:
                        parent_ok = False
                        parent_txt += (f"; close to the champion's weights (cos {near.cosine:.3f}, rel {near.relative_l2:.3f}): "
                                       f"needs more items right ({n_ch} vs {n_inc}) and a gain of {need:.2f} ({gain:+.2f})")
            if not confirm_mode:
                fp = M.first_pass(ch.score, inc_total, se, entry_floor=entry_floor, now=self._margin_now(ts.track_id, now))
                row = {"draw": draw_id, "kind": "first", "score": ch.score, "incumbent": inc_total, "gain": fp.gain, "delta": fp.delta,
                       "nonproc": ch.nonproc, "by_tier": dict(ch.by_tier), **xrec(ch.key)}
                if v5c is not None:
                    row.update(self._v5_row(v5c, ch.key, ts, now))
                if v3:
                    row.update(se=se, inc_key=inc_key, z=z)
                if bar is not None:
                    row.update(bar_uid=bar.uid, bar_key=bar.key, bar_se=se, bar_scores={str(m.uid): total[m.key] for m in co_bar})
                ch.history.append(row)
                if v3 and have_inc:
                    self._first_stage_v3(ts, ch, row, inc_key, se, parent_ok, parent_txt, draw_id, now)
                    self._gap_shadow(ts, ch, seed, xs)
                    continue
                if fp.passed and parent_ok:
                    ch.status, ch.reason = ST.PENDING_CONFIRM, ""
                    ch.delta, ch.first_gain, ch.draw1 = fp.delta, fp.gain, draw_id
                    ch.confirm_draw, ch.confirm_score = "", None
                    ch.first_incumbent = inc_key if v5c is not None else (current.key if current else "")
                    self._set_confirm_due(ts, ch, now, now)
                    ch.decision_rule = M.V3_RULE if v3 else ""
                    ch.confirm_draws = []
                    ts.log(now, f"{ch.key[:12]} passed the first draw ({fp.gain:+.2f} vs margin {fp.delta:.2f})")
                else:
                    ch.status = ST.NOT_BETTER
                    ch.reason = f"first draw: gain {fp.gain:+.2f}, margin {fp.delta:.2f}" + parent_txt
                    ts.log(now, f"{ch.key[:12]} {ch.reason}")
                self._gap_shadow(ts, ch, seed, xs)
                continue
            first_inc = self._first_incumbent(ch)
            new_inc = ((v5c is not None or current is not None) and first_inc is not None and first_inc != inc_key) or ref_extended
            row = {"draw": draw_id, "kind": "confirm", "score": ch.score, "incumbent": inc_total,
                   "nonproc": ch.nonproc, "by_tier": dict(ch.by_tier), **xrec(ch.key)}
            if v5c is not None:
                row.update(self._v5_row(v5c, ch.key, ts, now))
            if v3:
                row.update(se=se, inc_key=inc_key, gain=gain, z=z)
            if bar is not None:
                row.update(bar_uid=bar.uid, bar_key=bar.key, bar_se=se, bar_scores={str(m.uid): total[m.key] for m in co_bar})
            verdict, why = "", ""
            if v3 and have_inc and ch.decision_rule == M.V3_RULE:
                verdict, why = self._confirm_v3(ch, row, inc_key, gain, se, parent_ok, parent_txt, self._margin_now(ts.track_id, now))
                ok = verdict == "crown"
            else:   # the old rule: no flag, an entry whose first pass was decided before v3 (grandfathered), or a vacant track
                ok = M.confirm(ch.score, inc_total, ch.delta or M.margin_floor(self._margin_now(ts.track_id, now)), entry_floor=entry_floor,
                               promoted_this_round=promoted, new_incumbent=new_inc, se=se, now=self._margin_now(ts.track_id, now)) and parent_ok
            ch.history.append(row)
            self._gap_shadow(ts, ch, seed, xs)
            held = RV.under_review(ch)
            if held and (ok or verdict == "extra"):
                # rc11.post5: never confirmed while under review (a hold set after the draw was scheduled): stays pending
                ch.reason = f"under review: {held}"
                ts.log(now, f"{ch.key[:12]} passed the confirmation draw but is under review: not confirmed")
                continue
            if verdict == "extra":
                # stays pending: one more confirmation draw next cycle, disjoint from all its earlier draws
                ch.confirm_draws.append(draw_id)
                ch.reason = why
                ts.log(now, f"{ch.key[:12]} {why}")
                continue
            if ok and current is None and not (v5c is not None and self._crown_slot(now) is not None):
                # vacant track (1.2.0rc7): not crowned yet; the cohort playoff decides (``_cohort_step``). On eval v5.1 with
                # crowning slots (rc11.post5, #25186) the decision is stored for the next slot below instead: the slot rounds
                # decide among the entries that confirmed on v5.1, as on an occupied v5.1 track (no v5.1 cohort playoff)
                ch.confirm_draw, ch.confirm_score = draw_id, ch.score
                _, open_ = self._cohort(ts, self._cohort_block(ch))
                ts.log(now, f"{ch.key[:12]} passed the confirmation draw ({ch.score:.2f}); waits for its cohort "
                            f"(first seen by block {ch.first_seen_block}, {len(open_)} entries still open)")
                continue
            k1a = self._k1a_slot(ts, ch, now) if ok else None
            if ok and (slot := (k1a if k1a is not None else self._crown_slot(now))) is not None:
                # CROWN_SLOTS_UTC: the decision is stored, the scores are published, the champion stays until the slot
                # (K1-a: a confirmation due before the slot that ran after it joins that slot)
                ch.history.append({"kind": "crown_pending", "ts": now, "slot": slot, **({"k1a": True} if k1a is not None else {})})
                ch.reason = f"crowning decided; takes effect at {datetime.fromtimestamp(slot, timezone.utc):%Y-%m-%d %H:%M}Z"
                ts.log(now, f"{ch.key[:12]} would be the new champion ({ch.score:.2f}"
                            + (f", bar uid {bar.uid} {inc_total:.2f}" if bar is not None else "") + f"): {ch.reason}")
            elif ok:
                self._retire_champions(ts, current)
                ch.status, ch.reason, ch.champion_since_ts = ST.CHAMPION, "", now
                ts.champion, current, promoted = ch.key, ch, True
                ts.log(now, f"{ch.key[:12]} is the new champion ({ch.score:.2f})" + (f" ({why})" if why else "")
                       + (f"; beat the best co-champion uid {bar.uid} ({inc_total:.2f}), gain {gain:+.2f}, SE {se:.2f}" if bar is not None else ""))
            else:
                ch.status = ST.NOT_BETTER
                ch.reason = why or ("confirmation draw: gain too small" if inc_total is not None else "confirmation draw: below the entry floor")
                if not parent_ok and "parent" not in ch.reason:
                    ch.reason += parent_txt
                ts.log(now, f"{ch.key[:12]} {ch.reason}")

    # ------------------------------------------------- decision rule v3 (machine.py; docs/SPECIALIST-TRACKS.md)

    @staticmethod
    def _stage_rows(ch: ST.Submission, inc_key: str, kind: str, draws: Collection[str]) -> list[dict[str, Any]]:
        """History rows of this entry of ``kind`` ("first" | "confirm") with a paired SE, measured against ``inc_key``, on
        these draws only (the draw ids of the CURRENT stage: rows of an abandoned earlier stage are never pooled)."""
        return [h for h in ch.history if h.get("kind") == kind and h.get("draw") in draws and h.get("inc_key") == inc_key
                and h.get("se") is not None and h.get("gain") is not None]

    def _first_stage_v3(self, ts: ST.TrackState, ch: ST.Submission, row: dict[str, Any], inc_key: str, se: float,
                        parent_ok: bool, parent_txt: str, draw_id: str, now: float) -> None:
        """First-stage verdict under rule v3 against an incumbent: pass -> pending confirmation, close call -> stay queued
        for ONE more draw (disjoint from the first), else not better. Pools only draws against this same incumbent."""
        prior = [h for h in self._stage_rows(ch, inc_key, "first", ch.close_draws) if h is not row and h["draw"] != draw_id]
        pg, pse = M.pooled([h["gain"] for h in prior] + [row["gain"]], [h["se"] for h in prior] + [se])
        pz = M.z_of(pg, pse)
        row["pooled_z"] = pz
        close_used = len(ch.close_draws) >= M.MAX_CLOSE_DRAWS
        verdict = M.first_stage_v3(row["gain"], se, pg, pse, close_used=close_used, now=self._margin_now(ts.track_id, now))
        if not parent_ok:
            verdict = "fail"
        if verdict == "pass":
            stage = [h["draw"] for h in prior] + [draw_id]
            draw1 = stage[0]
            ch.status, ch.reason = ST.PENDING_CONFIRM, ""
            ch.delta, ch.first_gain, ch.draw1 = max(M.margin_floor(self._margin_now(ts.track_id, now)), M.Z_PASS * pse), pg, draw1
            ch.confirm_draw, ch.confirm_score, ch.confirm_draws = "", None, []
            ch.first_incumbent = inc_key
            if ch.close_draws and draw_id not in ch.close_draws:
                ch.close_draws.append(draw_id)
            first_ts = ts.draws[draw1].ts if draw1 in ts.draws else now
            self._set_confirm_due(ts, ch, first_ts, now)    # counted from the FIRST draw: a close call adds no delay
            ch.decision_rule = M.V3_RULE
            ts.log(now, f"{ch.key[:12]} passed the first stage (z {row['z']:.2f}, pooled z {pz:.2f} over {len(stage)} draw(s), "
                        f"gain {pg:+.2f})")
        elif verdict == "close":
            ch.status = ST.QUEUED
            ch.reason = f"close call (z {pz:.1f}): one more draw"
            ch.close_draws = [*ch.close_draws, draw_id]
            ts.log(now, f"{ch.key[:12]} {ch.reason} (gain {row['gain']:+.2f}, SE {se:.2f})")
        else:
            ch.status = ST.NOT_BETTER
            if prior or close_used:
                ch.reason = (f"close call, second draw: pooled evidence {pz:.1f} SE (gain {pg:+.2f}), "
                             f"{M.Z_PASS:.1f} SE and {M.margin_floor(self._margin_now(ts.track_id, now)):.1f} points needed" + parent_txt)
            else:
                ch.reason = (f"first draw: gain {row['gain']:+.2f}, margin {row['delta']:.2f} (z {row['z']:.1f})" + parent_txt)
            ts.log(now, f"{ch.key[:12]} {ch.reason}")

    def _confirm_v3(self, ch: ST.Submission, row: dict[str, Any], inc_key: str, gain: float, se: float, parent_ok: bool,
                    parent_txt: str, now: float | None = None) -> tuple[str, str]:
        """Confirmation verdict under rule v3: ("crown" | "extra" | "fail", text). Pools every draw against the incumbent of
        this confirmation (first stage included when it was measured against the same one)."""
        first = [h for h in self._stage_rows(ch, inc_key, "first", {ch.draw1, *ch.close_draws}) if h is not row]
        confirms = [h for h in self._stage_rows(ch, inc_key, "confirm", {ch.confirm_draw, *ch.confirm_draws})
                    if h is not row and h["draw"] != row["draw"]]
        confirm_pairs = [(h["gain"], h["se"]) for h in confirms] + [(gain, se)]
        pairs = [(h["gain"], h["se"]) for h in first] + confirm_pairs
        pg, pse = M.pooled([g for g, _ in pairs], [e for _, e in pairs])
        pz = M.z_of(pg, pse)
        row["pooled_z"] = pz
        extra_used = len(ch.confirm_draws) >= M.MAX_EXTRA_CONFIRMS
        verdict = M.confirm_v3(pairs, confirm_pairs, extra_used=extra_used, now=now)
        if not parent_ok:
            return "fail", f"confirmation draw: gain too small{parent_txt}"
        if verdict == "crown":
            return verdict, f"pooled evidence {pz:.1f} SE over {len(pairs)} draw(s)"
        if verdict == "extra":
            return verdict, (f"confirmation: pooled evidence {pz:.1f} SE < {M.Z_CROWN:.2f}, one more draw")
        return verdict, (f"confirmation draw: gain too small (pooled evidence {pz:.1f} SE, {M.Z_CROWN:.2f} SE and "
                         f"{M.margin_floor(now):.1f} points needed)")

    # ------------------------------------------------- cross-author gap (stage 2 plumbing: reported, never applied)

    def _gap_shadow(self, ts: ST.TrackState, ch: ST.Submission, seed: int, xs: Mapping[str, Mapping[str, Any]]) -> None:
        """Record the entry's cross-author gap and the penalty it WOULD get (``scoring.gap_penalty``) in its newest history
        row as ``gap_shadow``. Never applied: off/shadow/scored report only; "binding" mode records ``gap_binding`` instead
        (``_decide``) and has no live median."""
        if ch.key not in xs or not ch.history or self._x_mode_for(ts.track_id) == "binding":
            return
        row = ch.history[-1]
        x = xs[ch.key]
        gap_now = x["s_gen"] - x["s_x"]
        row["gap"] = gap_now
        row["gap_se"] = SC.gap_se(x["gen"], x["x"], seed)
        gen_id = row.get("x_gen", "")
        gaps, ses = [], []
        for h in ch.history:
            if h.get("x_gen", "") == gen_id and "s_gen" in h and "s_x" in h:
                gaps.append(h["s_gen"] - h["s_x"])
                ses.append(h.get("gap_se"))
        by_coldkey: dict[str, list[float]] = {}
        for s in ts.submissions.values():
            g = [h["s_gen"] - h["s_x"] for h in s.history if h.get("x_gen", "") == gen_id and "s_gen" in h and "s_x" in h]
            if g:
                by_coldkey.setdefault(s.coldkey, []).append(sum(g) / len(g))
        g_med = SC.median_gap({c: sum(v) / len(v) for c, v in by_coldkey.items()}, self._gap_reference(ts, gen_id))
        penalty, details = SC.gap_penalty(gaps, ses, g_med, cap=self.cfg.gap_penalty_cap)
        details["coldkeys"] = len(by_coldkey)
        if g_med is None:
            details["reason"] = "fewer than %d coldkeys: reported only" % SC.GAP_MIN_COLDKEYS
        row["gap_shadow"] = details
        ts.log(self.clock(), f"{ch.key[:12]} cross-author gap {gap_now:+.2f} (pooled {details['gap']:+.2f}, G_med "
                             f"{'-' if g_med is None else f'{g_med:+.2f}'}, would-be penalty {penalty:.2f}, not applied)")

    # ------------------------------------------------- binding mode: capped generalisation penalty (1.2.0rc9.post3)

    def _x_mode_for(self, track_id: str) -> str:
        """The cross-author mode in effect for a track. "binding" needs a frozen G_med for the track (``REAX_SPEC_G_MED``);
        without one the track runs as "shadow" (X items on top, reported, no penalty), logged once per cycle."""
        mode = self.cfg.x_tier_mode
        if mode == "binding" and track_id not in self.cfg.g_med:
            if track_id not in self._gmed_warned:
                self._gmed_warned.add(track_id)
                log.warning("track %s: binding mode without a configured G_med (REAX_SPEC_G_MED): running as shadow, no penalty",
                            track_id)
                self.state.track(track_id).log(self.clock(), "binding mode: no G_med configured for this track, running as shadow (no penalty)")
            return "shadow"
        return mode

    @staticmethod
    def _x_stats(results: Mapping[str, Sequence[SC.ItemScore]], seed: int, se: bool = False) -> dict[str, dict[str, Any]]:
        """Per model with cross-author AND other items in ``results``: s_gen, s_x (and, with ``se``, ``se`` = the bootstrap
        SE of s_gen - s_x on this draw, None when it is not finite), plus the two item groups."""
        out: dict[str, dict[str, Any]] = {}
        for k, v in results.items():
            gen, xi = SC.split_x(v, S.X_TIERS)
            if xi and gen:
                out[k] = {"s_gen": SC.score_items(gen).total, "s_x": SC.score_items(xi).total, "gen": gen, "x": xi}
                if se:
                    e = SC.gap_se(gen, xi, seed)
                    out[k]["se"] = e if math.isfinite(e) else None
        return out

    @staticmethod
    def _x_draws(sub: ST.Submission | None) -> dict[str, tuple[float, float]]:
        """The entry's draws with cross-author items from its history: draw id -> (s_gen - s_x, SE of that gap). Rows
        without a finite SE are skipped (fail closed: no penalty is built on a draw whose noise is unknown). A playoff row
        may carry a second draw (``draw2`` with its numbers in ``x2``)."""
        out: dict[str, tuple[float, float]] = {}
        for h in (sub.history if sub is not None else ()):
            for d, rec in ((h.get("draw"), h), (h.get("draw2"), h.get("x2"))):
                if not d or not isinstance(rec, dict) or "s_gen" not in rec or "s_x" not in rec:
                    continue
                se = rec.get("se_gap", rec.get("gap_se"))
                if isinstance(se, (int, float)) and math.isfinite(se) and math.isfinite(rec["s_gen"]) and math.isfinite(rec["s_x"]):
                    out[d] = (rec["s_gen"] - rec["s_x"], float(se))
        return out

    def _binding_penalty(self, sub: ST.Submission | None, track_id: str,
                         cur: Sequence[tuple[str, float, float | None]] = ()) -> tuple[float, dict[str, Any]]:
        """(P, details) of an entry as of now: ``scoring.gap_penalty`` over the entry's earlier draws with cross-author items
        and ``cur`` (the draws being decided, (draw id, gap, SE or None)), with the track's frozen G_med and the cap.
        An entry without such a draw has P = 0."""
        draws = self._x_draws(sub)
        for d, gap, se in cur:
            if se is not None and math.isfinite(se) and math.isfinite(gap):
                draws[d] = (gap, se)
        pen, det = SC.gap_penalty([g for g, _ in draws.values()], [e for _, e in draws.values()],
                                  self.cfg.g_med.get(track_id), cap=self.cfg.gap_penalty_cap)
        return pen, det

    def _binding_penalties(self, ts: ST.TrackState, track_id: str, results: Mapping[str, Any],
                           parts: Sequence[tuple[str, Mapping[str, Mapping[str, Any]]]]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """P and its details for every model of ``results``, from its history and the draws of ``parts``
        ([(draw id, ``_x_stats`` of that draw)])."""
        pen, det = {}, {}
        for k in results:
            cur = [(d, xs[k]["s_gen"] - xs[k]["s_x"], xs[k].get("se")) for d, xs in parts if k in xs]
            pen[k], det[k] = self._binding_penalty(ts.submissions.get(k), track_id, cur)
        return pen, det

    def _gap_reference(self, ts: ST.TrackState, gen_id: str) -> list[float]:
        """Gaps of reference models on the same pool generation (none are configured yet)."""
        return []

    # ------------------------------------------------------- first champion: cohort playoff

    @classmethod
    def _opening_cohort_first(cls, ts: ST.TrackState, queued: list[ST.Submission]) -> list[ST.Submission]:
        """rc10.post3 (Florian 21 Oct 2026, board #23162): on the vacant generalist track every opening-cohort entry takes
        its first draw before any post-opening entry, whatever the immunity order says (the cohort playoff waits for all of
        them). Inside each part the given order is kept. Other tracks and a crowned track: unchanged."""
        if ts.track_id != "generalist" or ts.champion:
            return list(queued)
        first = [s for s in queued if cls._cohort_block(s) == 0 or cls._reentrant(ts, s)]
        return first + [s for s in queued if s not in first]

    @classmethod
    def _reentry_waiting(cls, s: ST.Submission) -> bool:
        """A pruned opening-cohort entry of the generalist track that passed its first draw and has not re-entered yet."""
        return (s.track_id == "generalist" and _carryover_waiting(s) and s.withdrawn_kind == "pruned"
                and cls._cohort_block(s) == 0)

    @classmethod
    def _reentrant(cls, ts: ST.TrackState, s: ST.Submission) -> bool:
        """``s`` (live, not carried over yet) re-enters a pruned opening-cohort entry (#23211)."""
        if s.status == ST.WITHDRAWN or s.carried_from or cls._cohort_block(s) == 0:
            return False
        return any(cls._reentry_waiting(o) and reentry_match(o, s) for o in ts.submissions.values())

    @staticmethod
    def _cohort_block(s: ST.Submission) -> int:
        if s.track_id == "generalist" and s.first_seen_ts < split.GENERALIST_START_UTC.timestamp():
            return 0   # every pre-opening commitment belongs to the same first-champion playoff
        return s.first_seen_block

    def _cohort(self, ts: ST.TrackState, block: int) -> tuple[list[ST.Submission], list[ST.Submission]]:
        """The cohort of ``block`` on a vacant track: (its entries that passed their confirmation and wait for the playoff,
        best confirmation score first; the entries first seen in that block or earlier that are still open)."""
        ready = [s for s in ts.submissions.values()
                 if s.status == ST.PENDING_CONFIRM and s.confirm_draw and self._cohort_block(s) == block and not RV.under_review(s)]
        ready.sort(key=lambda s: (-(s.confirm_score if s.confirm_score is not None else -math.inf),
                                  s.first_seen_block, s.uid, s.key))
        open_ = [s for s in ts.submissions.values() if self._cohort_block(s) <= block and self._cohort_open(ts, s)]
        if block == 0 and ts.track_id == "generalist":
            # re-entry (#23211): a pruned opening entry holds the opening playoff until the deadline; a re-entry seen by
            # then holds it while it is open (like every entry: one whose repo is unavailable does not hold it)
            now = self.clock()
            open_ += [s for s in ts.submissions.values()
                      if (self._reentry_waiting(s) and now < REENTRY_DEADLINE_TS)
                      or (self._reentrant(ts, s) and self._cohort_open(ts, s))]
        return ready, sorted(open_, key=lambda s: (s.first_seen_block, s.uid, s.key))

    @staticmethod
    def _cohort_open(ts: ST.TrackState, s: ST.Submission) -> bool:
        """Still undecided in a way that holds back its cohort's crowning: queued (unless the miner's repo is not
        available), waiting for this host (``wait_kind`` WAIT_INFRA, set where the wait starts; never the reason text,
        which can embed the miner's base id), or pending a confirmation that can still run. An entry under review
        (rc11.post5) holds nothing back."""
        if RV.under_review(s):
            return False
        if s.status == ST.QUEUED:
            return not s.reason.startswith("waiting for the repo")
        if s.status == ST.PENDING_VERIFY:
            return s.wait_kind == ST.WAIT_INFRA
        if s.status == ST.PENDING_CONFIRM:
            return not s.confirm_draw and s.draw1 in ts.draws    # a lost first draw can never be confirmed: it waits alone
        return False

    def _cohort_step(self, ts: ST.TrackState, now: float) -> list[ST.Submission]:
        """The earliest cohort with confirmed entries, on a vacant track: nothing while one of its entries is still open; a
        lone confirmed entry is crowned at once (it cleared the floor on both its draws, as before 1.2.0rc7); several
        are returned for the playoff. With a champion, waiting entries go back to the normal challenger path. Open entries
        hold the cohort back for at most COHORT_MAX_WAIT_S after the earliest passed confirmation of its ready entries."""
        if ts.champion:
            self._release_waiting(ts, now, "")
            return []
        blocks = [self._cohort_block(s) for s in ts.submissions.values() if s.status == ST.PENDING_CONFIRM and s.confirm_draw
                  and not RV.under_review(s)]
        if not blocks:
            return []
        block = min(blocks)
        ready, open_ = self._cohort(ts, block)
        if open_:
            since = min((ts.draws[s.confirm_draw].ts for s in ready if s.confirm_draw in ts.draws), default=now)
            if now - since < COHORT_MAX_WAIT_S:
                return []
            if now < REENTRY_DEADLINE_TS and any(self._reentry_waiting(s) for s in open_):
                return []   # re-entry (#23211): the opening playoff waits for pruned entries until the deadline
            log.info("track %s: cohort of block %d complete by its deadline (%.0f h); %d open entries no longer wait: %s",
                     ts.track_id, block, COHORT_MAX_WAIT_S / 3600, len(open_), ", ".join(s.key[:12] for s in open_))
        if len(ready) == 1:
            self._crown(ts, ready[0], now, f"the only confirmed entry of its cohort (block {block})")
            return []
        return ready

    def _crown(self, ts: ST.TrackState, sub: ST.Submission, now: float, why: str) -> None:
        sub.status, sub.reason, sub.champion_since_ts = ST.CHAMPION, "", now
        sub.confirm_draw, sub.confirm_score = "", None
        sub.serve_retry_pending = False
        sub.champion_rounds_without_ok = 0
        sub.pay_suspended_since_ts = None
        ts.champion, ts.playoff_aborts = sub.key, 0
        ts.log(now, f"{sub.key[:12]} is the new champion ({sub.score:.2f}): {why}")
        log.info("track %s: %s (uid %d) is the new champion: %s", ts.track_id, sub.key[:12], sub.uid, why)
        self._release_waiting(ts, now, sub.key)

    @staticmethod
    def _release_waiting(ts: ST.TrackState, now: float, champion: str) -> None:
        """Entries still waiting for a playoff once the track has a champion: they stay pending and are confirmed against
        the champion with the full margin (``M.confirm``, ``new_incumbent``) on the next confirmation draw."""
        for s in ts.ordered(ST.PENDING_CONFIRM):
            if s.confirm_draw and s.key != champion:
                s.confirm_draw, s.confirm_score = "", None
                s.reason = "a champion was crowned: confirms against it with the full margin"
                ts.log(now, f"{s.key[:12]} back on the challenger path ({s.reason})")

    async def _run_playoff(self, tdef: C.TrackDef, ts: ST.TrackState, ready: list[ST.Submission], snap: Snapshot,
                           now: float, budget: int, slot: bool = False) -> None:
        """The confirmed entries of a complete cohort, best confirmation score first and at most ``budget`` of them, on one
        shared fresh draw. Every one of them must be ready to serve; otherwise the playoff waits for the next cycle.

        It runs only with a full field: every entry, or a whole hour's evaluations (``max_evals_per_hour``; a configured
        limit of 1 an hour: the best confirmation alone). The confirmation scores come from different draws, so the field
        is not cut by whatever is left of the hour (review F5). Tracks are served least recently served first, so a
        declining playoff track goes first once the hour's budget is back. Starvation note: while another track keeps
        taking freed slots one at a time, the playoff can wait up to about an hour (until the rolling window empties).

        ``slot`` (rc11.post2, CROWN_SLOTS_UTC): the field is the entries that earned the title against the current bar and
        wait for the slot; the same machinery serves round 1, ``_slot_extensions_and_decide`` runs rounds 2-3 and decides
        (rc11.post3: every confirmed entry is in, no budget cut; a model failure is retried once on the same items; a
        round with every member failing or a host fault is void and re-run on new items; all rounds use the slot's
        pinned evaluation time ``slot_eval_ts``)."""
        label = "slot playoff" if slot else "cohort playoff"
        v5slot = slot and self._slot_v5(tdef.track_id, self._slot_of(ready[0]))
        if self._v5_on(tdef.track_id, now) and ((not slot) or (not v5slot and any(_decided_under_v5(s) for s in ready))):
            # eval v5.1 has no vacant-track playoff yet (the opening was decided under v4; a seated co-champion set is the bar), and
            # a slot on v4 (the flip slot, rule A) decides the field earned under v4, never entries that earned the title under
            # v5.1. Fail closed: no draw of the old pool is scored with the new rules
            ts.log(now, f"{label} not run: not wired for eval v5.1" + (" (entries decided under v5.1)" if slot else " (vacant track)"))
            log.error("track %s: %s under eval v5.1 is not wired for this field; nothing served", tdef.track_id, label)
            return
        if v5slot:
            # rc11.post4: a slot on eval v5.1 decides entries that confirmed on v5.1 only (slot rules section 3: all rounds use the
            # evaluation the entries confirmed on). A stored crowning earned under v4 cannot be in it: rule A, a v5.1 first stage
            for s in [s for s in ready if not _decided_under_v5(s)]:
                s.history.append({"kind": "crown_done", "ts": now})
                self._v5_restart(ts, s, now, "crowning earned under v4 is not in a slot on eval v5.1", "v4 crowning at a v5.1 slot")
            ready = [s for s in ready if _decided_under_v5(s)]
            if not ready:
                return
        need = min(len(ready), self.cfg.max_evals_per_hour)
        if budget < need:
            log.info("track %s: %s of %d entries waits for budget (%d left, %d needed)", tdef.track_id, label,
                     len(ready), budget, need)
            self._playoff_waits.add(tdef.track_id)
            return
        field = list(ready) if slot else ready[:budget]
        slot_ts = self._slot_of(ready[0]) if slot else None
        eval_ts = self.slot_eval_ts(slot_ts) if slot_ts is not None else now
        free = self._free_bytes(self.cfg.work_dir)
        if free < self.cfg.min_free_bytes:
            log.warning("track %s: %s skipped, %.0f GB free on the work directory's filesystem, %.0f GB needed",
                        tdef.track_id, free / 1e9, self.cfg.min_free_bytes / 1e9)
            ts.log(now, f"{label}: infrastructure failure, insufficient free disk; nothing counted")
            return
        prepared: list[_Prepared] = []
        preparation_infra: list[ST.Submission] = []
        for s in sorted(ready, key=lambda s: s.reason.startswith(WAITING_FOR_REPO)):
            if len(prepared) >= len(field):
                break
            p = await self._prepare_cycle(tdef, s, snap, now)
            if p is None:
                if s.status == ST.REJECTED or s.wait_kind == ST.WAIT_MINER:
                    ts.log(now, f"{label} skips {s.key[:12]}: {s.reason[:80]}")
                    continue  # unavailable miner repos never block the rest of the queue
                if s.wait_kind != ST.WAIT_INFRA:
                    return  # preparation allowance exhausted; no infrastructure failure was observed
                preparation_infra.append(s)
                continue
            prepared.append(p)
        if preparation_infra and slot:
            ts.log(now, f"{label} waits: infrastructure preparation failure (our side); nothing counted, re-run when it works")
            self._host_failed.add(tdef.track_id)
            return
        if not prepared and slot:
            # every entry failed to load on its own side (repo gone/private, model refused): nobody is in, nobody crowned
            self._slot_unserved(ts, ready, slot_ts if slot_ts is not None else now, now)
            return
        if preparation_infra:
            ts.playoff_aborts += 1
            if ts.playoff_aborts < C.PLAYOFF_MAX_ABORTS:
                ts.log(now, f"{label} waits: infrastructure preparation failure; nothing counted")
                self._host_failed.add(tdef.track_id)
                return
        if not prepared:
            if preparation_infra:
                self._host_failed.add(tdef.track_id)
            elif slot:
                active = [s for s in ready if s.status == ST.PENDING_CONFIRM and _crown_pending(s) is not None]
                if active and all(s.wait_kind == ST.WAIT_MINER for s in active):
                    self._release_slot_field(ts, active, now)
            return
        # the pools are small: the playoff draw may reuse items of the entries' earlier draws (least-used items first)
        seed = self.seed_source(tdef.track_id, snap.block)
        if v5slot:
            drawn = self._slot_draw_v5(ts, seed, set(), snap, now, first=True)
            if drawn is None:
                return
            rows, draw_id = drawn
        else:
            try:
                rows, fresh = S.draw(self._pool(tdef.category), n=self._draw_n(tdef.track_id, eval_ts), seed=seed, use_counts=ts.used,
                                     now=eval_ts, **self._playoff_x(tdef))
            except S.SampleError as exc:
                log.warning("track %s: no playoff draw this cycle: %s", tdef.track_id, exc)
                ts.log(now, f"no draw: {exc}; refill the sealed pool")
                return
            draw_id = self._record_draw(ts, self._pool(tdef.category), rows, fresh, seed, snap, now)
        ts.log(now, f"{label} on draw {draw_id[:8]}: " + ", ".join(f"uid {p.sub.uid}" for p in prepared))
        if slot:
            self._slot_ctx = {} if v5slot else None
            results, failed, void = await self._slot_serve_round(ts, prepared, rows, now, draw_id=draw_id, seed=seed)
            if void:
                if slot_ts is not None and self.clock() >= slot_ts + split.SLOT_CLOCK_S:
                    # slot rules section 7: a round in which every entry fails is void and is re-run with the same entries;
                    # nobody is released for it (critic S4). Past the slot clock this needs a human: alert loudly.
                    log.error("track %s: slot %s round 1 still void after the %d h slot clock (every member failed); "
                              "entries kept, investigate the host", tdef.track_id, int(slot_ts), split.SLOT_CLOCK_S // 3600)
                ts.log(now, f"slot round 1 on draw {draw_id[:8]} void (our side: every member failed or a host fault); "
                            "nothing counted, re-run on new items next cycle")
                self._host_failed.add(tdef.track_id)
                return
            served = [p for p in prepared if p.sub.key in results]
            try:
                twins = self._behaviour_twins(ts, served, served, V5W.flatten(rows) if v5slot else rows, draw_id, now)
            except Exception:   # noqa: BLE001 - a measurement bug must not discard the slot's scores
                log.exception("track %s: behavioural near-duplicate check failed, no twins this slot", tdef.track_id)
                twins = {}
            ts.playoff_aborts = 0
            await self._slot_extensions_and_decide(tdef, ts, ready, served, results, rows, draw_id, seed, twins, snap, now,
                                                   slot_ts=slot_ts, failed1=[p.sub for p in failed])
            return
        results: dict[str, list[SC.ItemScore]] = {}
        failed: list[_Prepared] = []
        for p in prepared:
            items = await self._serve(p, rows, self.clock())
            if items is None:
                failed.append(p)
                continue
            results[p.sub.key] = items
        host_problem = len(prepared) > 1 and len(failed) == len(prepared)
        for p in prepared:
            outcome = "ok" if p.sub.key in results else ("infra" if host_problem or p.infra_failure else "model")
            self._serve_round(ts, p.sub, outcome, now, joint=p.joint)
        if preparation_infra or host_problem or any(p.infra_failure for p in failed):
            if not preparation_infra:
                ts.playoff_aborts += 1
            if ts.playoff_aborts < C.PLAYOFF_MAX_ABORTS or not results:
                ts.log(now, f"{label}: infrastructure failure; nothing decided, next cycle again")
                self._host_failed.add(tdef.track_id)
                return
            ts.log(now, f"{label}: {ts.playoff_aborts} infrastructure aborts; decided among the entries served")
            deferred = preparation_infra + [p.sub for p in failed if p.infra_failure or host_problem]
            for s in deferred:
                if s.status == ST.PENDING_CONFIRM:
                    s.confirm_draw, s.confirm_score = "", None
                    s.reason = "playoff infrastructure bound: first draw kept; back on the challenger path"
        served = [p for p in prepared if p.sub.key in results]
        try:
            twins = self._behaviour_twins(ts, served, served, rows, draw_id, now)
        except Exception:   # noqa: BLE001 - a measurement bug must not discard the playoff's scores
            log.exception("track %s: behavioural near-duplicate check failed, no twins this playoff", tdef.track_id)
            twins = {}
        ts.playoff_aborts = 0
        extra, draw2 = await self._playoff_extension(tdef, ts, served, results, rows, draw_id, seed, snap, now)
        self._decide_playoff(ts, tdef, [p.sub for p in served if p.sub.status != ST.REJECTED],
                             results, draw_id, seed, now, twins, extra, draw2)

    async def _playoff_extension(self, tdef: C.TrackDef, ts: ST.TrackState, served: list[_Prepared],
                                 results: dict[str, list[SC.ItemScore]], rows: list[dict], draw_id: str, seed: int,
                                 snap: Snapshot, now: float) -> tuple[dict[str, list[SC.ItemScore]], str]:
        """PLAYOFF_MIN_GAP_SE: when the best and the second-best playoff score differ by less than that many paired
        standard errors, ONE more shared draw is run for the tied top group alone (entries within that distance of the
        best, among all served entries). Returns (the group's scores on the second draw, its draw id), or ({}, "") when no
        extension happens. It runs right after the first draw whatever is left of the hour's budget; its evaluations are
        charged to the rolling budget like any other, so the other tracks wait in the next cycle. It is skipped (today's
        single-draw rule) only when one of its models cannot be served: a half-extended group would compare entries on
        different data."""
        total = {p.sub.key: SC.score_items(results[p.sub.key]).total for p in served}
        if self._x_mode_for(tdef.track_id) == "binding":   # S_eff of the first playoff draw decides who is tied
            pen, _ = self._binding_penalties(ts, tdef.track_id, results, [(draw_id, self._x_stats(results, seed, se=True))])
            total = {k: v - pen[k] for k, v in total.items()}
        group = self._tied_group(served, results, total, seed)
        if len(group) < 2:
            return {}, ""
        seed2 = self._extension_seed(seed)
        first = {r["id"] for r in rows}
        pool = self._pool(tdef.category)
        xkw = self._playoff_x(tdef)
        try:
            rows2, fresh2 = S.draw(pool, n=self._draw_n(tdef.track_id, now), seed=seed2, use_counts=ts.used, exclude=first, now=now, **xkw)
        except S.SampleError:   # the pool does not allow a disjoint draw: least-used items, as the first playoff draw
            try:
                rows2, fresh2 = S.draw(pool, n=self._draw_n(tdef.track_id, now), seed=seed2, use_counts=ts.used, now=now, **xkw)
            except S.SampleError as exc:
                log.warning("track %s: no playoff extension draw: %s", tdef.track_id, exc)
                ts.log(now, f"no extension draw: {exc}; refill the sealed pool")
                return {}, ""
        draw2 = self._record_draw(ts, pool, rows2, fresh2, seed2, snap, now)
        ts.log(now, f"playoff extension on draw {draw2[:8]}: the top {len(group)} are within {PLAYOFF_MIN_GAP_SE:g} SE: "
                    + ", ".join(f"uid {p.sub.uid}" for p in group))
        extra: dict[str, list[SC.ItemScore]] = {}
        failed: list[_Prepared] = []
        for p in group:
            items = await self._serve(p, rows2, self.clock())
            if items is None:
                failed.append(p)
            else:
                extra[p.sub.key] = items
        host_problem = len(failed) == len(group)
        for p in group:
            outcome = "ok" if p.sub.key in extra else ("infra" if host_problem or p.infra_failure else "model")
            self._serve_round(ts, p.sub, outcome, now, joint=p.joint)
        if failed:
            ts.log(now, "playoff extension abandoned: a model could not be served; decided on the first draw")
            return {}, ""
        return extra, draw2

    @staticmethod
    def _extension_seed(seed: int) -> int:
        return int(hashlib.sha256(f"{seed}|playoff-extension".encode()).hexdigest()[:8], 16)

    def _playoff_x(self, tdef: C.TrackDef) -> dict[str, Any]:
        """Draw arguments of a playoff draw: cross-author items are part of it in binding mode only (as before, no other
        mode puts them into a playoff draw)."""
        if self._x_mode_for(tdef.track_id) != "binding":
            return {"generalist": tdef.track_id == "generalist"}
        return {"x_mode": "binding", "x_share": self.cfg.x_tier_share, "generalist": tdef.track_id == "generalist"}

    @staticmethod
    def _tied_group(served: list[_Prepared], results: Mapping[str, Sequence[SC.ItemScore]], total: Mapping[str, float],
                    seed: int) -> list[_Prepared]:
        """The best entry and every other served entry whose score is less than PLAYOFF_MIN_GAP_SE paired standard errors
        of the difference below it. The entry floor is not applied here: it applies to the two-draw means (a runner-up just
        below the floor may still clear it). A lone entry gives a group of fewer than two."""
        ranked = sorted(served, key=lambda p: -total[p.sub.key])
        if len(ranked) < 2:
            return []
        best = ranked[0]
        return [best] + [p for p in ranked[1:]
                         if total[best.sub.key] - total[p.sub.key]
                         < PLAYOFF_MIN_GAP_SE * SC.paired_se(results[best.sub.key], results[p.sub.key], seed)]

    # ------------------------------------------------- slot playoff (rc11.post2, CROWN_SLOTS_UTC)

    def _slot_totals(self, ts: ST.TrackState, tdef: C.TrackDef, draws: Sequence[tuple[str, int, dict[str, list[SC.ItemScore]]]],
                     ) -> tuple[dict[str, float], dict[str, list[float]], dict[str, float], dict[str, dict[str, Any]]]:
        """(mean score per entry over the draws it took, its per-draw scores, its binding penalty, the penalty details). In
        binding mode every draw score is S_eff = S - P, with P from the entry's history and all the draws it took.
        On eval v5.1 (rc11.post4, ``self._slot_ctx``): the per-round S_v5.1 and their mean; the gap penalty is retired."""
        if self._slot_ctx is not None:
            per5 = {k: [self._slot_ctx[d].parts[k].s for d, _, res in draws if k in res] for k in draws[0][2]}
            return ({k: sum(v) / len(v) for k, v in per5.items()}, per5, {k: 0.0 for k in per5}, {})
        per: dict[str, list[float]] = {k: [] for k in draws[0][2]}
        for _, _, res in draws:
            for k, items in res.items():
                per[k].append(SC.score_items(items).total)
        pen: dict[str, float] = {k: 0.0 for k in per}
        pdet: dict[str, dict[str, Any]] = {}
        if self._x_mode_for(tdef.track_id) == "binding":
            parts = [(d, self._x_stats(res, sd, se=True)) for d, sd, res in draws]
            pen, pdet = self._binding_penalties(ts, tdef.track_id, draws[0][2], parts)
        mean = {k: sum(v) / len(v) - pen[k] for k, v in per.items()}
        return mean, per, pen, pdet

    # ------------------------------------------------- slot rounds (rc11.post3; public slot rules sections 4-7, CRITERIA v2)

    @staticmethod
    def _slot_g(margin_now: float | None) -> float:
        """The tie gap G: ``SLOT_G_FLOOR_FRACTION`` x the margin floor of the pinned evaluation (0.5 points on v4)."""
        return split.SLOT_G_FLOOR_FRACTION * M.margin_floor(margin_now)

    @staticmethod
    def _holder_keys(s: ST.Submission) -> set[str]:
        """Rule 4.4: what makes two entries the same holder: the coldkey, and (K4-a) the Hugging Face account or
        organisation of the repo id (case-insensitive)."""
        out = {f"ck:{s.coldkey}"} if s.coldkey else set()
        ns = s.repo.split("/", 1)[0].strip().lower() if "/" in s.repo else ""
        if split.SLOT_OWNER_COLDKEY_OR_HF and ns:
            out.add(f"hf:{ns}")
        return out

    @classmethod
    def _holder_groups(cls, subs: Sequence[ST.Submission]) -> list[list[ST.Submission]]:
        """Entries joined by a shared holder key, transitively (A and B share a coldkey, B and C an HF account: one holder)."""
        groups: list[tuple[set[str], list[ST.Submission]]] = []
        for s in subs:
            keys = cls._holder_keys(s)
            hit = [g for g in groups if g[0] & keys]
            merged = (set(keys), [s])
            for g in hit:
                merged[0].update(g[0])
                merged[1].extend(g[1])
                groups.remove(g)
            groups.append(merged)
        return [g[1] for g in groups]

    async def _slot_serve_round(self, ts: ST.TrackState, live: list[_Prepared], rows: list[dict], now: float,
                                draw_id: str = "", seed: int = 0) -> tuple[dict[str, list[SC.ItemScore]], list[_Prepared], bool]:
        """One slot round on ``rows`` for the live members (rule 4.1): a member whose model fails is retried
        ``SLOT_SERVE_RETRIES`` times on the same items. Returns (scores, members that failed for good, void). The round is
        void (counts for nobody) when every member failed or a failure is flagged as a host fault (ours).
        On eval v5.1 (``self._slot_ctx`` is a dict, rc11.post4) ``rows`` are the typed items of the draw: every member is
        served their questions (``V5W.flatten``), the round's S_v5.1 numbers go to ``self._slot_ctx[draw_id]``
        (``V5W.slot_context``), and a member without a finite S_v5.1 counts as failed on its own side."""
        v5 = self._slot_ctx is not None
        flat = V5W.flatten(rows) if v5 else rows

        async def serve(p: _Prepared) -> list[SC.ItemScore] | None:
            return await (self._serve_v5(p, flat, draw_id, self.clock()) if v5 else self._serve(p, flat, self.clock()))

        res: dict[str, list[SC.ItemScore]] = {}
        failed: list[_Prepared] = []
        for p in live:
            items = await serve(p)
            if items is None:
                failed.append(p)
            else:
                res[p.sub.key] = items
        void = (not res) or any(p.infra_failure for p in failed)
        if failed and not void:
            for _ in range(split.SLOT_SERVE_RETRIES):
                again = []
                for p in failed:
                    items = await serve(p)
                    if items is None:
                        again.append(p)
                    else:
                        res[p.sub.key] = items
                failed = again
                if not failed or any(p.infra_failure for p in failed):
                    break
            void = any(p.infra_failure for p in failed)
        if v5 and res and not void:
            ctx = V5W.slot_context(draw_id, rows, {p.sub.key: V5W.responses(rows, flat, p.probs) for p in live if p.sub.key in res},
                                   seed, self._v5_params())
            for p in live:
                if p.sub.key in ctx.bad:
                    ts.log(now, f"{p.sub.key[:12]} has no finite S_v5.1 on slot draw {draw_id[:8]}: counts as failed (its side)")
                    del res[p.sub.key]
                    failed.append(p)
            void = not res
            if not void:
                self._slot_ctx[draw_id] = ctx
        for p in live:
            outcome = "ok" if p.sub.key in res else ("infra" if void or p.infra_failure else "model")
            self._serve_round(ts, p.sub, outcome, now, joint=p.joint)
        return res, failed, void

    def _slot_draw_v5(self, ts: ST.TrackState, seed: int, used: set[str], snap: Snapshot, now: float,
                      first: bool = False) -> tuple[list[dict], str] | None:
        """A slot round draw on eval v5.1 (rc11.post4): 540 T + 360 N + the public items, as ``V5W.draw`` gives every v5.1
        evaluation, disjoint from the slot's earlier rounds (``used``; if the pool does not allow it, no draw: the rounds end early).
        Fail closed (v5arm): no usable v5.1 pool, config or panel stats -> no draw, nothing served or decided; the slot waits
        (our side: nobody is crowned or removed for it) and runs once the pool is back. None = no draw."""
        try:
            pool, public = self._v5_pool()
            p5 = self._v5_params()
        except S.SampleError as exc:
            log.error("track %s: eval v5.1 FAIL CLOSED at the slot: %s; no slot round, nothing decided or crowned",
                      ts.track_id, exc)
            ts.log(now, f"no slot round draw: {exc}; the slot waits (our side)")
            return None
        try:
            rows, fresh = V5W.draw(pool, public, seed=seed, use_counts=ts.used, exclude=used, p=p5)
        except S.SampleError as exc:
            if first:
                log.warning("track %s: no v5.1 slot draw this cycle: %s", ts.track_id, exc)
                ts.log(now, f"no draw: {exc}; refill the sealed pool")
                return None
            # SLOT-RULES s.4 "rounds share no items" (post4 critic S2): no least-used reuse on v5.1; the rounds end early and
            # the slot is decided on the completed rounds ("up to 3 rounds")
            log.warning("track %s: no disjoint v5.1 slot round draw: %s; the rounds end here", ts.track_id, exc)
            ts.log(now, f"no disjoint slot round draw: {exc}; rounds end early, refill the sealed pool")
            return None
        return rows, self._record_draw_v5(ts, rows, fresh, seed, snap, now)

    def _slot_draw(self, tdef: C.TrackDef, ts: ST.TrackState, seed: int, used: set[str], eval_ts: float, snap: Snapshot,
                   now: float) -> tuple[list[dict], str] | None:
        """A fresh round draw disjoint from the slot's earlier rounds (least-used items if the pool does not allow it, as
        coded for the playoff extension); None = no draw at all. On eval v5.1: ``_slot_draw_v5``."""
        if self._slot_ctx is not None:
            return self._slot_draw_v5(ts, seed, used, snap, now)
        pool = self._pool(tdef.category)
        xkw = self._playoff_x(tdef)
        n = self._draw_n(tdef.track_id, eval_ts)
        try:
            rows, fresh = S.draw(pool, n=n, seed=seed, use_counts=ts.used, exclude=used, now=eval_ts, **xkw)
        except S.SampleError:
            try:
                rows, fresh = S.draw(pool, n=n, seed=seed, use_counts=ts.used, now=eval_ts, **xkw)
                ts.log(now, "slot round draw: not enough unused items, least-used items taken")
            except S.SampleError as exc:
                log.warning("track %s: no slot round draw: %s", tdef.track_id, exc)
                ts.log(now, f"no slot round draw: {exc}; refill the sealed pool")
                return None
        return rows, self._record_draw(ts, pool, rows, fresh, seed, snap, now)

    async def _slot_extensions_and_decide(self, tdef: C.TrackDef, ts: ST.TrackState, group: list[ST.Submission],
                                          served: list[_Prepared], results: dict[str, list[SC.ItemScore]], rows: list[dict],
                                          draw_id: str, seed: int, twins: Mapping[str, Sequence[str]], snap: Snapshot,
                                          now: float, slot_ts: float | None = None,
                                          failed1: Sequence[ST.Submission] = ()) -> None:
        """The slot rounds after round 1 (``rows``/``results``, already served) and the decision (public slot rules
        sections 4-7, rc11.post3). After every round, in this order, on the members still in: 4.1 serving (done while
        serving: ``failed1`` failed round 1 twice), 4.2 entry floor (pooled mean at or below it), 4.3 parents and twins and
        4.4 one entry per holder (round 1 only), 4.5 separation: a member leaves when the leader (highest pooled mean) is
        ahead of it by at least ``M.required_margin(pooled paired SE)``. Members only leave. One member left: the clear
        winner. Two or more after ``SLOT_MAX_ROUNDS`` rounds (or when the slot clock, the void-round retries or the pool
        end the rounds early): the G test (section 5). Every round uses the slot's pinned evaluation time (``eval_ts``).
        Everything runs in this one call; the result and its evidence are stored at the end (``_slot_apply``)."""
        slot_ts = slot_ts if slot_ts is not None else now
        eval_ts = self.slot_eval_ts(slot_ts)
        mnow = self._margin_now(ts.track_id, eval_ts)
        v5c = self._slot_ctx       # eval v5.1 (rc11.post4): the rounds' S_v5.1 numbers, by draw id; None on v4
        # v5.1: S_v5.1 is on the DI scale, the v4-scale entry floor does not apply (as in ``_decide``: the bar was the reference)
        floor = -math.inf if v5c is not None else self._entry_floor(tdef)
        draws: list[tuple[str, int, dict[str, list[SC.ItemScore]]]] = [(draw_id, seed, results)]
        used = {r["id"] for r in rows}
        live = list(served)
        left: dict[str, dict[str, Any]] = {s.key: {"round": 1, "rule": "4.1", "why": "model failed twice on the round's items"}
                                           for s in failed1}
        cmp: dict[str, dict[str, Any]] = {}      # per member, the last comparison with the leader
        voids = 0
        stop = ""
        rnd = 1
        while True:
            mean, _, _, _ = self._slot_totals(ts, tdef, draws)
            pooled = {p.sub.key: [it for _, _, r in draws if p.sub.key in r for it in r[p.sub.key]] for p in live}
            for p in list(live):                                       # 4.2 entry floor
                if mean[p.sub.key] <= floor:
                    live.remove(p)
                    left[p.sub.key] = {"round": rnd, "rule": "4.2", "why": f"pooled mean {mean[p.sub.key]:.3f} at or below the entry floor {floor:g}"}
            if rnd == 1:
                for p in list(live):                                   # 4.3 parents and twins, round 1 only
                    short = self._playoff_short(ts, p.sub, results, mean, twins, seed, eval_ts,
                                                pse=(v5c[draw_id].pair_se if v5c is not None else None))
                    if short:
                        live.remove(p)
                        left[p.sub.key] = {"round": 1, "rule": "4.3", "why": f"does not beat {short[0][:12]} by the full copy margin"}
                for grp in self._holder_groups([p.sub for p in live]):  # 4.4 one entry per holder, round 1 only
                    if len(grp) < 2:
                        continue
                    keep = min(grp, key=lambda s: (-mean[s.key], s.first_seen_block, s.uid, s.key))
                    for s in grp:
                        if s is not keep:
                            live = [p for p in live if p.sub is not s]
                            left[s.key] = {"round": 1, "rule": "4.4", "why": f"same holder as uid {keep.uid}, which scored higher in round 1"}
            if live:                                                   # 4.5 separation, every member against the leader only
                lead = min(live, key=lambda p: (-mean[p.sub.key], p.sub.first_seen_block, p.sub.uid, p.sub.key))
                cmp[lead.sub.key] = {"round": rnd, "leader": True}
                for p in list(live):
                    if p is lead:
                        continue
                    se = (self._slot_pair_se_v5(lead.sub.key, p.sub.key, draws) if v5c is not None
                          else self._slot_pair_se(pooled[lead.sub.key], pooled[p.sub.key], seed))
                    margin = M.required_margin(se, mnow)
                    gap = mean[lead.sub.key] - mean[p.sub.key]
                    cmp[p.sub.key] = {"round": rnd, "leader_uid": lead.sub.uid, "gap": round(gap, 4), "se": round(se, 4),
                                      "margin": round(margin, 4)}
                    if gap >= margin:
                        live.remove(p)
                        left[p.sub.key] = {"round": rnd, "rule": "4.5", "why": f"separated: {gap:.3f} behind uid {lead.sub.uid}, margin {margin:.3f}"}
            line = (f"slot round {rnd} (draw {draws[-1][0][:8]}): " + ", ".join(
                f"uid {ts.submissions[k].uid} {mean[k]:.3f}" + (f" (gap {cmp[k]['gap']:.3f}, SE {cmp[k]['se']:.3f}, margin {cmp[k]['margin']:.3f})" if "gap" in cmp.get(k, {}) and cmp[k]["round"] == rnd else "")
                + (f" out {left[k]['rule']}" if k in left and left[k]["round"] == rnd else "")
                for k in sorted(draws[-1][2], key=lambda k: -mean[k])))
            ts.log(now, line)
            log.info("track %s: %s", tdef.track_id, line)
            if len(live) <= 1 and (split.SLOT_STOP_WHEN_SEPARATED or not live):
                break
            if rnd >= split.SLOT_MAX_ROUNDS:
                break
            if self.clock() >= slot_ts + split.SLOT_CLOCK_S:
                stop = f"the slot clock ran out after round {rnd}"
                break
            nxt = None
            while nxt is None:
                seed_n = self._extension_seed(draws[-1][1] + voids)
                drawn = self._slot_draw(tdef, ts, seed_n, used, eval_ts, snap, now)
                if drawn is None:
                    stop = f"no item draw for round {rnd + 1}"
                    break
                rows_n, did = drawn
                used |= {r["id"] for r in rows_n}
                res_n, failed_n, void = await self._slot_serve_round(ts, live, rows_n, now, draw_id=did, seed=seed_n)
                if void:
                    voids += 1
                    ts.log(now, f"slot round {rnd + 1} on draw {did[:8]} void (our side: every member failed or a host fault); "
                                f"{voids} void round(s)")
                    if voids > split.SLOT_VOID_RETRIES or self.clock() >= slot_ts + split.SLOT_CLOCK_S:
                        stop = f"round {rnd + 1} void {voids} time(s)"
                        break
                    continue
                for p in failed_n:
                    live.remove(p)
                    left[p.sub.key] = {"round": rnd + 1, "rule": "4.1", "why": "model failed twice on the round's items"}
                nxt = (did, seed_n, res_n)
            if nxt is None:
                break
            draws.append(nxt)
            rnd += 1
        if stop:
            ts.log(now, f"slot rounds ended early: {stop}; decided on {len(draws)} completed round(s)")
        mean, per, pen, pdet = self._slot_totals(ts, tdef, draws)
        g = self._slot_g(mnow)
        crowned: list[ST.Submission] = []
        result = "none"
        if len(live) == 1:
            crowned, result = [live[0].sub], "clear_winner"
        elif len(live) > 1:
            ordered = sorted(live, key=lambda p: (-mean[p.sub.key], p.sub.first_seen_block, p.sub.uid, p.sub.key))
            lead = ordered[0]
            shared = [lead.sub]
            for p in ordered[1:]:                                      # section 5: within G of the leader, no chaining
                gap = mean[lead.sub.key] - mean[p.sub.key]
                cmp.setdefault(p.sub.key, {})["g"] = g
                if gap <= g:
                    shared.append(p.sub)
                else:
                    left[p.sub.key] = {"round": len(draws), "rule": "5", "why": f"not separated but {gap:.3f} behind, more than G {g:g}"}
            if split.SLOT_GROUP_CAP is not None and len(shared) > split.SLOT_GROUP_CAP:
                for s in shared[split.SLOT_GROUP_CAP:]:
                    left[s.key] = {"round": len(draws), "rule": "K6", "why": f"group cap {split.SLOT_GROUP_CAP}"}
                shared = shared[:split.SLOT_GROUP_CAP]
            if len(shared) > 1:
                crowned, result = shared, "shared_group"
            elif split.SLOT_K8B_LEADER_ALONE:
                crowned, result = shared, "leader_alone_beyond_g"
                ts.log(now, f"slot: the others were not separated, but were more than G = {g:g} behind, so not practically "
                            f"tied: uid {lead.sub.uid} is crowned alone")
            else:
                result = "none_unresolved"
        self._slot_apply(ts, tdef, group, [p.sub for p in served if p.sub.status != ST.REJECTED], draws, mean, per, pen,
                         pdet, crowned, result, left, cmp, slot_ts, eval_ts, voids, stop, now)

    def _slot_pair_se(self, a: Sequence[SC.ItemScore], b: Sequence[SC.ItemScore], seed: int) -> float:
        """Rule 4.5: the paired SE of a pair over all the rounds both took (the items concatenated, bootstrap ``paired_se``).
        Together with ``_slot_totals`` (the pooled means) this is what the v5 wiring replaces to decide a v5 slot on S_v5.1."""
        return SC.paired_se(a, b, seed)

    def _slot_pair_se_v5(self, a: str, b: str, draws: Sequence[tuple[str, int, dict[str, list[SC.ItemScore]]]]) -> float:
        """Rule 4.5 on eval v5.1 (rc11.post4): the paired SE of the pooled S_v5.1 means of ``a`` and ``b`` over the rounds both
        took (``V5W.slot_pair_se``: the v5.1 paired bootstrap per round, pooled like the mean)."""
        ctxs = [self._slot_ctx[d] for d, _, r in draws if a in r and b in r]
        return V5W.slot_pair_se(ctxs, a, b) if ctxs else math.inf

    def slot_eval_ts(self, slot_ts: float) -> float:
        """The instant whose evaluation a slot uses (rule A, slot rules section 3): just before the slot instant, so draw
        size, margin floor and every time-switched evaluation rule are those the entries confirmed on, even if a new
        evaluation binds at the slot instant itself. The v5 wiring reads its v5 switch at this instant for slot rounds
        and K1-a confirmations (the hook of item F).
        rc11.post4: always ``slot - 1 s``. The flip slot (Mon 26 Oct 14:30Z, slot - 1 s before ``EVAL_V5_START_UTC``) stays on
        v4; every later slot (from Thu 29 Oct) is decided on eval v5.1 (``_slot_v5``)."""
        return slot_ts - 1.0

    def _slot_v5(self, track_id: str, slot_ts: float) -> bool:
        """True when the slot at ``slot_ts`` is decided on eval v5.1: the evaluation in force at ``slot_eval_ts``."""
        return self._v5_on(track_id, self.slot_eval_ts(slot_ts))

    def _slot_apply(self, ts: ST.TrackState, tdef: C.TrackDef, group: list[ST.Submission], field: list[ST.Submission],
                    draws: Sequence[tuple[str, int, dict[str, list[SC.ItemScore]]]], mean: Mapping[str, float],
                    per: Mapping[str, list[float]], pen: Mapping[str, float], pdet: Mapping[str, dict[str, Any]],
                    crowned: list[ST.Submission], result: str, left: Mapping[str, dict[str, Any]],
                    cmp: Mapping[str, dict[str, Any]], slot_ts: float, eval_ts: float, voids: int, stop: str,
                    now: float) -> None:
        """Store the slot's decision: scores and history rows of the members served, the crowning (one champion, or a
        shared group stored as the state's co-champion set), every other entry of the slot back on the challenger path,
        and the public evidence record (``TrackState.slot_results``)."""
        draw_id = draws[0][0]
        v5c = self._slot_ctx
        for s in field:
            if v5c is not None:     # eval v5.1 (rc11.post4): S_v5.1 is the score; by_tier {T, N} marks it (``_v51_scored``)
                took = [d for d, _, res in draws if s.key in res]
                s.score, s.nonproc, s.last_eval_ts = mean[s.key], mean[s.key], now
                s.by_tier = {t: sum(getattr(v5c[d].parts[s.key], t.lower()) for d in took) / len(took) for t in ("T", "N")}
                entry = {"draw": draw_id, "kind": "playoff", "score": s.score, "field": len(field), "slot": True,
                         "scoring_regime": V5W.REGIME, "s_v51": s.score, "t": s.by_tier["T"], "n": s.by_tier["N"],
                         "nonproc": s.nonproc, "by_tier": dict(s.by_tier), "draws": took,
                         "scores": [round(v, 4) for v in per[s.key]]}
                if s.key in left:
                    entry["left"] = dict(left[s.key])
                s.history.append(entry)
                continue
            if tdef.track_id == "generalist":
                for did, _, res in draws:
                    if s.key in res:
                        self._category_diagnostics(s, ts.draws[did], res[s.key])
            nps = [nonproc_scores(res[s.key]) for _, _, res in draws if s.key in res]
            vals = [n for n, _ in nps if n is not None]
            nonproc = sum(vals) / len(vals) if vals else None
            if nonproc is not None and pen.get(s.key):
                nonproc -= pen[s.key]
            tiers = {t: round(sum(bt[t] for _, bt in nps if t in bt) / sum(1 for _, bt in nps if t in bt), 4)
                     for t in {t for _, bt in nps for t in bt}}
            s.score, s.nonproc, s.by_tier, s.last_eval_ts = mean[s.key], nonproc, tiers, now
            entry = {"draw": draw_id, "kind": "playoff", "score": s.score, "field": len(field), "slot": True,
                     "nonproc": s.nonproc, "by_tier": dict(tiers), "draws": [d for d, _, r in draws if s.key in r],
                     "scores": [round(v, 4) for v in per[s.key]]}
            if s.key in left:
                entry["left"] = dict(left[s.key])
            if s.key in pdet:
                entry["gap_binding"] = dict(pdet[s.key])
            s.history.append(entry)
        keys = [s.key for s in crowned]
        if crowned:
            lead = crowned[0]
            current = ts.submissions.get(ts.champion)
            self._retire_champions(ts, current if current is not None and current.status == ST.CHAMPION else None)
            lead.history.append({"kind": "crown_done", "ts": now})
            why = {"clear_winner": "clear winner", "shared_group": f"shared group of {len(crowned)} (practically tied)",
                   "leader_alone_beyond_g": "leader alone (the others not separated but more than G behind)"}[result]
            self._crown(ts, lead, now, f"slot of {len(field)} on {len(draws)} round(s), first draw {draw_id[:8]}: {why}")
            if len(crowned) > 1:
                for s in crowned[1:]:
                    s.history.append({"kind": "crown_done", "ts": now})
                    s.reason = f"co-champion: shares the title with {', '.join(f'uid {m.uid}' for m in crowned if m is not s)}"
                    s.confirm_draw, s.confirm_score = "", None
                ts.cochamps, ts.cochamps_since = keys, now
                ts.log(now, "shared group crowned (not separated, within G of the leader): "
                            + ", ".join(f"uid {m.uid} {mean[m.key]:.3f}" for m in crowned) + f"; {75 / len(crowned):.2f} % of the champion share each")
        for s in group:
            if s.key in keys:
                continue
            s.history.append({"kind": "crown_done", "ts": now})
            s.confirm_draw, s.confirm_score = "", None
            why = f"in the slot, not crowned ({left[s.key]['rule']}: {left[s.key]['why']})" if s.key in left else "in the slot, not served"
            if crowned and s.status == ST.PENDING_CONFIRM:      # slot rules section 2: a new first draw against the new champion
                self._slot_restart(ts, s, now, f"{why}; a new first draw against the new champion")
                continue
            s.reason = f"{why}; confirms against the champion with the full margin"
            ts.log(now, f"{s.key[:12]} back on the challenger path: {s.reason}")
        if crowned:
            self._release_pending_crowns(ts, now)
        else:
            ts.log(now, f"slot: nobody crowned ({result}); the current places continue")
        self._slot_record(ts, group, field, draws, mean, per, crowned, result, left, cmp, slot_ts, eval_ts, voids, stop, now)

    def _slot_record(self, ts: ST.TrackState, group: Sequence[ST.Submission], field: Sequence[ST.Submission],
                     draws: Sequence[tuple[str, int, dict[str, list[SC.ItemScore]]]], mean: Mapping[str, float],
                     per: Mapping[str, list[float]], crowned: Sequence[ST.Submission], result: str,
                     left: Mapping[str, dict[str, Any]], cmp: Mapping[str, dict[str, Any]], slot_ts: float, eval_ts: float,
                     voids: int, stop: str, now: float) -> None:
        """The public evidence of one slot (slot rules section 9) as a small record in ``TrackState.slot_results``:
        aggregates only (no item ids, prompts or answers). Format: RESULT.md of rc11p3d / CHANGELOG 1.2.0rc11.post3."""
        from ..version import __version__
        served = {s.key for s in field}
        entries = []
        for s in sorted(group, key=lambda s: (-(mean.get(s.key, -math.inf)), s.uid, s.key)):
            pending = [h for h in s.history if h.get("kind") == "crown_pending"]
            e: dict[str, Any] = {"uid": s.uid, "key": s.key, "repo": s.repo, "revision": s.revision,
                                 "in_by": "k1a" if pending and pending[-1].get("k1a") else "confirmed",
                                 "served": s.key in served}
            if s.key in mean:
                e.update(scores=[round(v, 4) for v in per[s.key]], mean=round(mean[s.key], 6))
            if s.key in cmp:
                e.update({k: v for k, v in cmp[s.key].items() if k != "round"}, compared_round=cmp[s.key].get("round"))
            e["outcome"] = "crowned" if s in crowned else "left"
            if s.key in left:
                e["left"] = dict(left[s.key])
            entries.append(e)
        v5slot = self._v5_on(ts.track_id, eval_ts)
        rec = {"slot": slot_ts, "decided": now, "track": ts.track_id, "release": __version__,
               "evaluation": "v5.1" if v5slot else ("v4" if S.v4_active(eval_ts) else "v3"), "eval_ts": eval_ts,
               "rounds": len(draws), "draws": [d for d, _, _ in draws], "void_rounds": voids, "ended_early": stop,
               "g": self._slot_g(self._margin_now(ts.track_id, eval_ts)),
               "margin_floor": M.margin_floor(self._margin_now(ts.track_id, eval_ts)),
               "result": result, "crowned": [s.uid for s in crowned],
               "share_each": (round(1.0 / len(crowned), 6) if crowned else 0.0), "entries": entries}
        if v5slot:      # rc11.post4: the v5.1 pool version (v51.config.json) and the panel stats the scores are scaled with
            try:
                p5 = self._v5_params()
                rec["pool_version"] = self._v5_config[0][0] if self._v5_config and self._v5_config[0] else None
                rec["panel"] = p5.panel.version if p5.panel is not None else None
            except S.SampleError:     # evidence only (a slot decided without a v5.1 draw: single entry, nobody served)
                rec["pool_version"] = rec["panel"] = None
        pool = self._pool_sha(ts, v5slot)
        if pool:
            rec["pool_sha256"] = pool
        ts.slot_results.append(rec)
        del ts.slot_results[:-split.SLOT_RESULTS_KEEP]

    def _pool_sha(self, ts: ST.TrackState, v5: bool = False) -> str:
        """sha256 of the track's sealed item ids (sorted), "" when unknown. Identifies the pool, reveals no item. ``v5``:
        the v5.1 T/N pool (rc11.post4)."""
        try:
            if v5:
                ids = sorted(str(r.get("id")) for r in self._v5_pool()[0])
                return hashlib.sha256("\n".join(ids).encode()).hexdigest() if ids else ""
            tdef = next(t for t in self.cfg.tracks if t.track_id == ts.track_id)
            ids = sorted(str(r.get("id")) for r in self._pool(tdef.category))
        except Exception:   # noqa: BLE001 - evidence only
            return ""
        return hashlib.sha256("\n".join(ids).encode()).hexdigest() if ids else ""

    def _playoff_short(self, ts: ST.TrackState, s: ST.Submission, results: Mapping[str, Sequence[SC.ItemScore]],
                       total: Mapping[str, float], twins: Mapping[str, Sequence[str]], seed: int, now: float,
                       pse: Callable[[str, str], float] | None = None) -> list[str]:
        """The entries of the playoff that ``s`` answers like (twins), derives from or descends from and does not beat by
        the full margin on the playoff draw (empty = may be crowned). ``pse`` (eval v5.1 slots): the paired SE of the
        score difference (``DrawContext.pair_se``) instead of the v4 bootstrap, as ``_decide`` does under v5.1."""
        must_beat = [k for k in twins.get(s.key, ()) if k in results]
        if s.derived_from in results and s.derived_from not in must_beat:
            must_beat.append(s.derived_from)
        if self.cfg.nc_r3:
            must_beat += [k for k in self._lineage(ts, s.key)[1:] if k in results and k not in must_beat]
        if self.cfg.nc_r1:
            short = []
            for k in must_beat:
                se_k, cse = SC.paired_se_parts(results[s.key], results[k], seed)
                if pse is not None:
                    se_k = pse(s.key, k)
                cgain = SC.score_items(results[s.key]).competence - SC.score_items(results[k]).competence
                if total[s.key] - total[k] < M.required_margin(se_k, self._margin_now(ts.track_id, now)) or cgain < max(TWIN_COMP_FLOOR, M.SE_MULTIPLE * cse):
                    short.append(k)
        else:
            short = [k for k in must_beat
                     if total[s.key] - total[k] < M.required_margin(pse(s.key, k) if pse is not None else SC.paired_se(results[s.key], results[k], seed),
                                                                   self._margin_now(ts.track_id, now))]
        return short

    def _decide_playoff(self, ts: ST.TrackState, tdef: C.TrackDef, field: list[ST.Submission],
                        results: dict[str, list[SC.ItemScore]], draw_id: str, seed: int, now: float,
                        twins: Mapping[str, Sequence[str]], extra: Mapping[str, Sequence[SC.ItemScore]] | None = None,
                        draw2: str = "") -> None:
        """The best score on the playoff draw that clears the entry floor is crowned, unless it answers like (or derives
        from) another entry of the playoff and does not beat it by the full margin there. Ties go to the earlier
        commitment. If nobody clears the floor, nobody is crowned and the entries served are not better.

        ``extra`` (PLAYOFF_MIN_GAP_SE) holds the tied top group's scores on a second shared draw: only that group can win,
        on the mean of its two draw scores; if the best of it still leads the next by less than PLAYOFF_MIN_GAP_SE paired
        standard errors (both draws pooled), the earliest commitment among the entries within that distance wins."""
        total = {k: SC.score_items(v).total for k, v in results.items()}
        extra = extra or {}
        binding = self._x_mode_for(tdef.track_id) == "binding"
        pen: dict[str, float] = {}
        pdet: dict[str, dict[str, Any]] = {}
        xs1: dict[str, dict[str, Any]] = {}
        xs2: dict[str, dict[str, Any]] = {}
        if binding:   # S_eff: each entry's penalty from its own cross-author draws, both playoff draws included
            xs1 = self._x_stats(results, seed, se=True)
            xs2 = self._x_stats(extra, self._extension_seed(seed), se=True) if extra else {}
            pen, pdet = self._binding_penalties(ts, tdef.track_id, results, [(draw_id, xs1), (draw2, xs2)])
            for k in results:   # raw = the mean over the entry's playoff draws, as the score itself
                raw = (total[k] + SC.score_items(extra[k]).total) / 2 if k in extra else total[k]
                pdet[k].update(raw=raw, s_eff=raw - pen[k])
            total = {k: v - pen[k] for k, v in total.items()}
        total1 = dict(total)
        for k, items in extra.items():
            total[k] = (total1[k] + (SC.score_items(items).total - pen.get(k, 0.0))) / 2
        if tdef.track_id == "generalist":
            for s in field:
                self._category_diagnostics(s, ts.draws[draw_id], results[s.key])
                if s.key in extra:
                    self._category_diagnostics(s, ts.draws[draw2], extra[s.key])
        floor = self._entry_floor(tdef)
        for s in field:
            np_, by_tier = nonproc_scores(results[s.key])
            np2, tiers2 = nonproc_scores(extra[s.key]) if s.key in extra else (None, {})
            if s.key in extra:    # the same two-draw mean as the score: keep-alive must not read draw one alone
                vals = [v for v in (np_, np2) if v is not None]
                nonproc = sum(vals) / len(vals) if vals else None
                tiers = {t: round((by_tier.get(t, tiers2.get(t)) + tiers2.get(t, by_tier.get(t))) / 2, 4)
                         for t in set(by_tier) | set(tiers2)}
            else:
                nonproc, tiers = np_, by_tier
            if nonproc is not None and pen:
                nonproc -= pen[s.key]
            s.score, s.nonproc, s.by_tier, s.last_eval_ts = total[s.key], nonproc, dict(tiers), now
            entry = {"draw": draw_id, "kind": "playoff", "score": s.score, "field": len(field),
                     "nonproc": s.nonproc, "by_tier": dict(s.by_tier)}
            if s.key in extra:
                entry.update({"draw2": draw2, "score1": total1[s.key], "score2": 2 * total[s.key] - total1[s.key],
                              "nonproc1": np_, "nonproc2": np2, "by_tier1": dict(by_tier), "by_tier2": dict(tiers2)})
            if binding:
                if s.key in xs1:
                    entry.update(s_gen=xs1[s.key]["s_gen"], s_x=xs1[s.key]["s_x"], se_gap=xs1[s.key]["se"],
                                 gap=xs1[s.key]["s_gen"] - xs1[s.key]["s_x"])
                if s.key in xs2:
                    entry["x2"] = {"s_gen": xs2[s.key]["s_gen"], "s_x": xs2[s.key]["s_x"], "se_gap": xs2[s.key]["se"]}
                entry["gap_binding"] = dict(pdet[s.key])
            s.history.append(entry)
        line = f"playoff draw {draw_id[:8]}: " + ", ".join(f"uid {s.uid} {total1[s.key]:.2f}" for s in field)
        if extra:
            line += f"; extension draw {draw2[:8]}, mean of both: " + ", ".join(f"uid {s.uid} {total[s.key]:.2f}" for s in field if s.key in extra)
        ts.log(now, line)
        log.info("track %s: %s", tdef.track_id, line)
        order = sorted(field, key=lambda s: (-total[s.key], s.first_seen_block, s.uid, s.key))
        if extra:   # only the extended group competes; a still-tied top goes to the earlier commitment
            group = [s for s in order if s.key in extra]
            best = group[0]
            tied = [best] + [s for s in group[1:] if total[best.key] - total[s.key] < PLAYOFF_MIN_GAP_SE * SC.paired_se(
                results[best.key] + list(extra[best.key]), results[s.key] + list(extra[s.key]), seed)]
            tied.sort(key=lambda s: (s.first_seen_block, s.uid, s.key))
            order = tied + [s for s in group if s not in tied] + [s for s in order if s.key not in extra]
        for s in order:
            if total[s.key] < floor:
                continue    # not sorted by score any more once a tied group was extended
            short = self._playoff_short(ts, s, results, total, twins, seed, now)
            if short:
                ts.log(now, f"{s.key[:12]} not crowned: does not beat {short[0][:12]} by the full margin on the playoff draw")
                continue
            self._crown(ts, s, now, f"best of the cohort playoff of {len(field)} on draw {draw_id[:8]}")
            return
        for s in field:
            s.status, s.reason = ST.NOT_BETTER, "cohort playoff draw: below the entry floor"
            s.confirm_draw, s.confirm_score = "", None
            ts.log(now, f"{s.key[:12]} {s.reason}")
