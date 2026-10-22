"""Margin and confirmation rules (docs/SPECIALIST-TRACKS.md), as pure functions.

* A challenger must beat the incumbent by ``delta = max(1.5 points, 2 x paired SE)`` on a fresh shared draw.
* It is then confirmed on a second, disjoint fresh draw taken at least 12 hours later: the gain over the incumbent
  there must again be positive and at least ``delta / 2``. Only then does it become the incumbent.
* Challengers are handled in commitment order. One that passed the confirmation earlier in the same round becomes the
  incumbent for the challengers after it, and they must beat it by the full margin on the same (second) draw.
* The same holds for any incumbent the first pass was not measured against: one crowned in an earlier round, or any
  incumbent at all when the track was vacant at the first pass. Half of a margin taken against another model (or the
  entry floor itself, on a vacant track) says nothing about this pair.
* With no incumbent the first model must reach the entry floor on both draws.

Rules v4 (``sampling.DRAWS_V4_START_UTC``, off by default): ``MARGIN_FLOOR`` is replaced by ``MARGIN_FLOOR_V4`` (1.0) for
every decision taken at or after that instant, and ``CLOSE_MIN_GAIN`` is half the floor in force. Draws of different sizes
pool unchanged (``pooled``).

Decision rule v3 (variance-aware, ``V3_DECISIONS``; docs/SPECIALIST-TRACKS.md "Decision rule v3"): the same ideas, but
the evidence is counted in standard errors and a close call gets one more draw instead of a verdict on one noisy draw.
Per challenger, only draws measured against the SAME incumbent are pooled: ``z = mean(gain) / (rms(se) / sqrt(k))``.

* First stage: ``z >= Z_PASS`` and ``gain >= MARGIN_FLOOR`` (exactly the margin rule above) -> pending confirmation.
  ``Z_CLOSE <= z < Z_PASS`` -> ONE more first-stage draw, disjoint from the first; the pooled z of the two must reach
  ``Z_PASS`` (and the pooled mean ``MARGIN_FLOOR``). ``z < Z_CLOSE`` -> not better.
* Confirmation: crown when the pooled z over all draws against that incumbent is ``>= Z_CROWN``, the confirmation gain
  is positive and the pooled mean ``>= MARGIN_FLOOR``. Between ``Z_EXTRA`` and ``Z_CROWN`` ONE extra confirmation draw;
  after it the pooled z must reach ``Z_CROWN`` and the two confirmation gains must add up above zero.
* Against an incumbent the first stage was not measured against (a new champion, or one crowned earlier in the same
  round) only the draws against that incumbent count: the same thresholds. A tie inside the noise therefore goes to
  the earlier commitment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, Union

from . import sampling

MARGIN_FLOOR = 1.5
# Rules v4 (3a): from ``sampling.DRAWS_V4_START_UTC`` on the floor is 1.0 point (draws of 900 items have a paired SE of
# about 58 % (1/sqrt(3)) of the 300-item draws, so the z-gates keep the false-crown rate while a real +2 gain is crowned far more
# often). The z-gates are unchanged. With the gate off (None) or no time given, everything is as before.
MARGIN_FLOOR_V4 = 1.0
SE_MULTIPLE = 2.0
CONFIRM_DELAY_S = 12 * 3600
# Every pending confirmation gets its draw before its track closes (Florian 21 Oct 2026). OFF (None) until the dated release
# notice: a first pass at/after this instant (or an entry still pending when the first cycle at/after it runs) is due at
# ``min(first pass + CONFIRM_DELAY_S, track end - CONFIRM_CLOSE_LEAD_S)``, never earlier than its first pass. No env override:
# validators must agree.
CONFIRM_BEFORE_CLOSE_START_UTC: Optional[datetime] = datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)
CONFIRM_CLOSE_LEAD_S = 6 * 3600     # the confirmation, a possible v3 extra confirmation draw and a cohort step fit under MAX_EVALS_PER_HOUR
_NO_END_TS = 253370764800.0         # 1 Jan 9999: a track end at/after this (ENGINE_HELD = datetime.max) is "no close"
CONFIRM_FRACTION = 0.5

# Decision rule v3. Z_PASS = 2 is the old "gain >= 2 x SE", Z_CROWN = 2.75 is the pooled bar for the crowning (about the
# strength of the old two-draw rule), Z_CLOSE / Z_EXTRA decide which borderline results get one more draw.
Z_PASS = 2.0
Z_CLOSE = 1.0
Z_EXTRA = 1.5
Z_CROWN = 2.75
V3_DECISIONS = False    # OFF until the dated #92 notice (critic H1); default of ``SpecConfig.v3_decisions`` (env REAX_SPEC_V3_DECISIONS=0 restores the old rule, no state migration)
V3_RULE = "v3"          # ``Submission.decision_rule`` of an entry whose first stage was decided under v3
MAX_CLOSE_DRAWS = 1     # extra first-stage draws per submission
MAX_EXTRA_CONFIRMS = 1  # extra confirmation draws per submission
Z_CONFIRM_MIN = 0.5     # the confirmation draw itself must show at least this many SE (it must confirm, not just not contradict)
Z_RECORD_MAX = 99.0     # recorded z values are clamped (no Infinity in the state file)
CLOSE_MIN_GAIN = 0.5 * MARGIN_FLOOR  # smallest first-stage gain that earns a close-call draw


def margin_floor(now: Union[float, datetime, None] = None) -> float:
    """The margin floor in force at ``now`` (unix seconds or datetime): ``MARGIN_FLOOR_V4`` from
    ``sampling.DRAWS_V4_START_UTC`` on, else ``MARGIN_FLOOR``. Decided by the time of the decision, whatever the sizes of
    the draws pooled in it."""
    return MARGIN_FLOOR_V4 if sampling.v4_active(now) else MARGIN_FLOOR


def close_min_gain(now: Union[float, datetime, None] = None) -> float:
    """Half the margin floor in force: the smallest first-stage gain that earns a close-call draw."""
    return 0.5 * margin_floor(now)


def pooled(gains: Sequence[float], ses: Sequence[float]) -> tuple[float, float]:
    """(mean gain, its standard error) over draws against the same incumbent: ``rms(se) / sqrt(k)``. One draw is
    returned unchanged (no rounding), so a single-draw z is exactly gain / se."""
    k = len(gains)
    if k == 0:
        return 0.0, math.inf
    if k == 1:
        return float(gains[0]), float(ses[0])
    mean = sum(gains) / k
    rms = math.sqrt(sum(s * s for s in ses) / k)
    return mean, rms / math.sqrt(k)


def z_of(mean: float, se: float) -> float:
    """mean / se with the degenerate cases settled: no usable SE gives 0, a zero SE gives +-inf by the sign of the gain."""
    if math.isnan(mean) or math.isnan(se) or math.isinf(se):
        return 0.0
    if se <= 0:
        return Z_RECORD_MAX if mean > 0 else (-Z_RECORD_MAX if mean < 0 else 0.0)
    return max(-Z_RECORD_MAX, min(Z_RECORD_MAX, mean / se))   # finite: the value is written to the state JSON


def reaches(mean: float, se: float, z: float) -> bool:
    """``mean >= z x se``, written as a product so one draw equals the old ``gain >= 2 x SE`` bit for bit."""
    if math.isnan(mean) or math.isnan(se):
        return False
    return mean >= z * se


def required_margin(se: float, now: Union[float, datetime, None] = None) -> float:
    return max(margin_floor(now), SE_MULTIPLE * se)


@dataclass(frozen=True)
class FirstPass:
    passed: bool
    gain: float
    delta: float


def first_pass(challenger: float, incumbent: float | None, se: float, *, entry_floor: float = 0.0,
               now: Union[float, datetime, None] = None) -> FirstPass:
    """Draw 1. ``incumbent`` None = vacant track: the entry floor is the bar."""
    if incumbent is None:
        return FirstPass(challenger >= entry_floor, challenger - entry_floor, entry_floor)
    delta = required_margin(se, now)
    gain = challenger - incumbent
    return FirstPass(gain >= delta, gain, delta)


def confirm_before_close_active(now: float) -> bool:
    """True from ``CONFIRM_BEFORE_CLOSE_START_UTC`` on (False while it is None). ``now``: unix seconds."""
    start = CONFIRM_BEFORE_CLOSE_START_UTC
    return start is not None and now >= start.timestamp()


def finite_track_end(track_end_ts: float | None) -> float | None:
    """The track's close as unix seconds, or None for a track that does not close (None, inf, nan, ENGINE_HELD)."""
    if track_end_ts is None or not math.isfinite(track_end_ts) or track_end_ts >= _NO_END_TS:
        return None
    return float(track_end_ts)


def confirm_due_ts(first_pass_ts: float, delay_s: float, track_end_ts: float | None, now_for_gate: float) -> float:
    """When the confirmation draw of a first pass at ``first_pass_ts`` is due: ``first_pass_ts + delay_s``, but once
    ``CONFIRM_BEFORE_CLOSE_START_UTC`` has passed at ``now_for_gate`` not later than ``track_end_ts - CONFIRM_CLOSE_LEAD_S``
    and never earlier than ``first_pass_ts`` (a first pass inside the last lead is due at once). A track without a close
    (None, inf, ENGINE_HELD) and a closed gate give ``first_pass_ts + delay_s`` unchanged."""
    due = first_pass_ts + delay_s
    end = finite_track_end(track_end_ts)
    if end is None or not confirm_before_close_active(now_for_gate):
        return due
    return max(first_pass_ts, min(due, end - CONFIRM_CLOSE_LEAD_S))


def confirmation_due(first_pass_ts: float, now: float, delay_s: float = CONFIRM_DELAY_S) -> bool:
    return now >= first_pass_ts + delay_s


def confirm(challenger: float, incumbent: float | None, first_delta: float, *, entry_floor: float = 0.0,
            promoted_this_round: bool = False, new_incumbent: bool = False, se: float = 0.0,
            now: Union[float, datetime, None] = None) -> bool:
    """Draw 2. Against the incumbent of the first pass: gain >= first_delta / 2 (and > 0). Against a challenger
    promoted earlier in the same round, or any other incumbent the first pass was not measured against
    (``new_incumbent``): gain >= the full margin for this pair. Vacant: the entry floor."""
    if incumbent is None:
        return challenger >= entry_floor
    gain = challenger - incumbent
    if promoted_this_round or new_incumbent:
        return gain >= required_margin(se, now)
    return gain > 0 and gain >= CONFIRM_FRACTION * first_delta


def first_stage_v3(gain: float, se: float, pooled_gain: float, pooled_se: float, *, close_used: bool,
                   now: Union[float, datetime, None] = None) -> str:
    """Verdict of a first-stage draw against an incumbent: "pass", "close" (one more draw) or "fail".

    ``gain``/``se``: this draw. ``pooled_gain``/``pooled_se``: this draw pooled with the earlier first-stage draws
    against the same incumbent (equal to the draw's own when there are none). ``close_used``: the one extra draw was
    already taken (whatever incumbent it was measured against). ``now``: the decision's time (selects the margin floor)."""
    if reaches(pooled_gain, pooled_se, Z_PASS) and pooled_gain >= margin_floor(now):
        return "pass"
    if close_used:
        return "fail"
    # a close call must at least be half the margin floor: a tiny gain with a tiny SE can never pass, its draw is waste
    if (not reaches(pooled_gain, pooled_se, Z_PASS) and reaches(pooled_gain, pooled_se, Z_CLOSE)
            and pooled_gain >= close_min_gain(now)):
        return "close"
    return "fail"


def confirm_v3(draws: Sequence[tuple[float, float]], confirm: Sequence[tuple[float, float]], *, extra_used: bool,
               now: Union[float, datetime, None] = None) -> str:
    """Verdict of a confirmation draw: "crown", "extra" (one more confirmation draw) or "fail".

    ``draws``: (gain, se) of every draw of the current stage against THIS incumbent, the one just taken last (first stage
    included when it was measured against the same incumbent). ``confirm``: (gain, se) of the confirmation draws among
    them, the new one last. Pairs, so a gain can never be paired with another draw's SE.
    ``extra_used``: this is already the extra confirmation draw. ``now``: the decision's time (margin floor)."""
    mean, se = pooled([g for g, _ in draws], [e for _, e in draws])
    strong = reaches(mean, se, Z_CROWN) and mean >= margin_floor(now)
    cmean, cse = pooled([g for g, _ in confirm], [e for _, e in confirm])
    confirmed = cmean > 0 and reaches(cmean, cse, Z_CONFIRM_MIN)
    if not extra_used:
        if strong and confirmed:
            return "crown"
        if reaches(mean, se, Z_EXTRA) and (not reaches(mean, se, Z_CROWN) or not confirmed) and cmean > 0:
            return "extra"
        return "fail"
    return "crown" if strong and confirmed else "fail"
