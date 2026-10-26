"""Competition share layer (docs/COMPETITIONS.md): how the miner weights are split between serving and competitions.

Pure and deterministic. Release constants only; nothing here is operator-configurable on finney.

    inference (serving)  +  engine champion  +  use-case (specialist) tracks

Profiles (``SPLIT_PROFILE`` is the active one):

* ``50/50`` (ACTIVE since the owner approved it on 15 Oct 2026): inference 0.50, engine 0.20, TWO active tracks
  0.15 + 0.15.
* ``80/20`` (inactive, kept ready): inference 0.80, engine 0.10, ONE active use-case track 0.10.

Rule that never changes: a competition share that is unclaimed or inactive (no active track, no champion, a struck
champion, the decayed part of the champion, a missing track rank) goes back to the inference part. It is never
burned and never paid to the owner UID.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

log = logging.getLogger(__name__)

SPLIT_PROFILE = "50/50"
# Before this instant the weights are exactly the 1.1 weights; from it on the competitions are active.
COMPETITIONS_START_UTC = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
# The use-case (specialist) tracks start with the competitions.
SPECIALIST_TRACKS_START_UTC = COMPETITIONS_START_UTC
# The engine track has its own start. None = HELD: no engine cycle runs, no engine submission is accepted and the engine
# share goes to the inference part, whatever a state file says. The owner opens it in a later release.
# 1.2.0rc9.post5: opens Tue 20 Oct 2026 18:00 UTC (DECISIONS 19 Oct 11:05Z; gates: runsc GO + calibration, FLA reference
# 9bd41632, RUNBOOK-P re-review, candidate-box spend card 17879).
ENGINE_TRACK_START_UTC: Optional[datetime] = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
# Quantisation Option A: announced switch and three-day conversion window (25 Oct decision).
ENGINE_QUANT_RULE_START_UTC = datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc)
ENGINE_QUANT_CONVERSION_END_UTC = datetime(2026, 10, 30, 14, 30, tzinfo=timezone.utc)
# Generalist replaces all specialist slots at this exact instant. Only generalist gets the post6 open class.
# Default target only: the release owner sets the agreed opening and notice times before shipping both images.
GENERALIST_START_UTC = datetime(2026, 10, 21, 13, 0, tzinfo=timezone.utc)
# Rules notice time; the release owner sets this before accepting commitments.
GENERALIST_COMMIT_FROM_UTC: Optional[datetime] = datetime(2026, 10, 20, 19, 9, 28, tzinfo=timezone.utc)
assert GENERALIST_COMMIT_FROM_UTC is None or GENERALIST_COMMIT_FROM_UTC <= GENERALIST_START_UTC
# Compatibility name for serving/base resource gates; only generalist gets the open-class rules.
SPEC_OPEN_CLASS_START_UTC = GENERALIST_START_UTC
# Uncalibrated release fails closed. Replace with s1-fast + 1.5 and the frozen release median gap.
GENERALIST_ENTRY_FLOOR = 36.8
GENERALIST_KEEPALIVE_NONPROC_FLOOR = 37.4
GENERALIST_G_MED: Optional[float] = 14.84
# Stands for "never" where a datetime is needed (a held engine track).
ENGINE_HELD = datetime.max.replace(tzinfo=timezone.utc)
# Opening fairness (board #22, miner report 21905). A commitment made at or before ``ENGINE_TRACK_START_BLOCK`` (finney
# block 9326382, predicted for about 2026-10-20T18:00:00Z from block 9321860 at 02:55:36Z, 12 s per block) counts as made at
# that block, so committing early (and risking deregistration while waiting) buys no ordering advantage. Entries that share
# that block are ordered by sha256("reax-eng-tie/1|<start block hash>|<hotkey>|<image digest>"): the block hash does not
# exist when they commit, so it cannot be ground for. Validators read it from the chain once the block exists
# (Chain.block_hash, engcomp.set_start_block_hash); until then they are ordered by uid, as before.
# ENGINE_PRE_OPEN_AS_START False = off; ENGINE_TRACK_START_BLOCK None = inactive.
ENGINE_PRE_OPEN_AS_START = True
ENGINE_TRACK_START_BLOCK: Optional[int] = 9326382

# Engine champion fallback (miner report 21901). While the current champion has no owner approval (award pending or
# rejected), the engine share is paid to the most recently crowned FORMER champion that is approved, still committed and
# still valid (its decay keeps running from its own crowning); with former-champion slots held it takes rank 1's share
# (ENGINE_RANK_SPLIT). False = rank 1's share goes to the inference part.
ENGINE_RESTORE_PREVIOUS = True

# Engine co-champions (1.2.0rc9.post7; Florian 20 Oct 2026 21:33Z, board #22468): one-time exception for the opening
# round, whose two best entries measured within the margin of each other. While the current champion of a track is one
# of the listed (digest, hotkey) pairs, the champion's part of the engine share (decaying from the champion's crowning
# block) is split evenly over the listed pairs. Each half is paid only to an entry that is still committed by that hotkey,
# holds a valid result and has its OWN owner approval; an unpaid half goes to the inference part (never to the other
# co-champion, never to a former champion). The challenger bar is max(champion score, ENGINE_CO_CHAMPION_BAR) x
# (1 + margin) while they hold the title (engcomp.update_champion). The decay runs from ENGINE_CO_CHAMPION_CROWNED (also
# when a co-champion is crowned again later, e.g. after the other withdrew). Once a challenger takes the title, the normal
# rule applies again and the set is ONE former champion (1.2.0rc10.post1, Florian 21 Oct 2026 05:25Z; see
# ENGINE_RANK_SPLIT): as the restored champion while the new one waits for its approval (ENGINE_RESTORE_PREVIOUS) and in
# its former-champion slot, with the same per-entry split, approvals and decay; a single co-champion is never paid the
# whole share. Empty = off.
ENGINE_CO_CHAMPIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "s1-fast": (
        ("576fea84b347fc85160f788d737db616be4b5ca4cde19268e0ce1d1f5cdfe12f",  # uid 198
         "5FLhkmTrKBUCJWnmCEY7xzcuVe6Mpk7FUgLdwfj7SbH33wo2"),
        ("ec2ee6ed3df69903bf9df5420cde8f869280647ce7d23a3837a2397b2ce34272",  # uid 4
         "5H6iSGe3t74QSUq2RgKtA4sET9EauSYsNBkEjGynW4jJ9htc"),
    ),
}
ENGINE_CO_CHAMPION_BAR: dict[str, float] = {"s1-fast": 2.82332066708034}  # uid 4's opening score
ENGINE_CO_CHAMPION_CROWNED: dict[str, int] = {"s1-fast": 9326968}  # the decay runs from the opening crowning block

# Engine keep-alive (miner report 21903), release constants; also needs REAX_ENGINE_KEEPALIVE=1 (default OFF). An engine
# entry that waits for its FIRST measurement earns 0 and is pruned. While it is committed, its image was pulled and bound to
# its hotkey, it has no result yet, and at most ENGINE_KA_MAX_HOURS passed since it was first seen (opening block counts as
# first seen for pre-opening commitments), it is paid ENGINE_KA_FRACTION x the median positive serving weight; at most one
# per coldkey, at most ENGINE_KA_MAX_ENTRIES entries, together at most ENGINE_KA_MAX_SHARE x the engine share, taken from
# the champion's part of it (nothing is taken when there is no champion). Unclaimed: inference.
ENGINE_KA_FRACTION = 0.5
ENGINE_KA_MAX_HOURS = 12.0
ENGINE_KA_MAX_ENTRIES = 4
ENGINE_KA_MAX_SHARE = 0.05
# Protected engine entries (Florian 21 Oct 05:25Z/05:30Z, rc10): the engine keep-alive above also pays, ahead of the waiting
# entries, (a) the current champion while it awaits the owner's award approval and (b) every former champion that holds a
# slot (ENGINE_RANK_SPLIT, ENGINE_FORMER_SLOT_BLOCKS from its own dethroning; all members of the opening co-champion set),
# in rank order, if still committed with a valid result (engcomp.protected_uids). A keep-alive is not an award; a uid paid
# anything else (incl. its slot share) gets none. At most ENGINE_KA_PROTECTED_MAX such uids on top of ENGINE_KA_MAX_ENTRIES
# (4 = the champion, a two-member co set and the other slot holder).
ENGINE_KA_PROTECTED_MAX = 4
# Reg-sniping review (20 Oct 2026, Florian 18:55Z "protect waiting entries on all tracks"): the engine keep-alive also covers
# (a) an entry whose image was not pulled yet if its hotkey is the digest's first committer and no other hotkey is bound to
# the image (an entry waiting for the review or the queue earned 0 and was pruned), and (b) the pending challenger between
# its first and confirming session; and the ENGINE_KA_MAX_HOURS window starts when the hotkey's immunity is about to end
# (registration block + CHAIN_IMMUNITY_BLOCKS - PROVISIONAL_KA_LEAD_BLOCKS) instead of at the commitment, so an entrant
# committed right after registering is not paid during its immunity and then dropped when it needs the slot.
# Legacy keep-alive (Florian 21 Oct 2026 07:05Z, board #22667, rc10): when the DF/RA tracks end at GENERALIST_START_UTC,
# their champion (always; a keep-alive is not an award) and their former champions that have the owner's approval are
# paid a bounded keep-alive for LEGACY_KA_DAYS from that instant (window [GENERALIST_START_UTC, + LEGACY_KA_DAYS)), so they
# are not pruned before the dual role (DUALROLE_START_BLOCK) lets them serve inference and compete on one hotkey. Read
# from the frozen DF/RA part of the validator's spec state (spec.evaluator.legacy_leaders: only entries their hotkey still
# commits to, so it ends early when the hotkey commits to another track). Each uid at most once (DF first, then RA, rank
# order), pure entrants only (no serving weight, nothing else paid), at most LEGACY_KA_MAX_ENTRIES (every approved former
# champion counts, not only the MAX_LEADERS = 3 ranked ones; on 21 Oct 07:00Z DF + RA held 9 champions/formers); each gets
# LEGACY_KA_FRACTION x the keep-alive median, all together at most LEGACY_KA_MAX_SHARE of the miner part (0.03: at
# today's median serving weight ~0.0045 each gets ~0.0022, so the cap binds only above 13 entries or at a much higher
# median), taken from the inference part (never from the engine share or a track's share).
LEGACY_KA_TRACKS = ("demand_forecasting", "risk_assessment")
LEGACY_KA_DAYS = 7
LEGACY_KA_MAX_ENTRIES = 12
LEGACY_KA_FRACTION = 0.5
LEGACY_KA_MAX_SHARE = 0.03
CHAIN_IMMUNITY_BLOCKS = 7200   # the subnet's ImmunityPeriod since ~block 9,305,000 (spec: evaluator.IMMUNITY_BLOCKS_ASSUMED)
ENGINE_DECAY_DAYS = 14  # the champion's share falls linearly from full to ENGINE_DECAY_FLOOR over this time
ENGINE_DECAY_FLOOR = 0.5
TRACK_RANK_SPLIT = (0.75, 0.17, 0.08)  # winner-take-most inside a use-case track, ranks 1-3
# Engine former-champion ranks (1.2.0rc10.post1; Florian 21 Oct 2026 05:30Z, decision "B"). Of the title holder's part of
# the engine share (decaying from ITS crowning, engcomp.champion_fraction; the decayed part goes to the inference part as
# before), the previous champion gets ENGINE_RANK_SPLIT[1] and the one before ENGINE_RANK_SPLIT[2] while their slot runs;
# rank 1 gets the rest (0.75 with both slots held, 0.83 with one, everything with none = the rule before). A slot ends
# when a newer crowning pushes the entry below rank 3, or ENGINE_FORMER_SLOT_BLOCKS after the block at which the entry
# was dethroned (the next champion's crowning block, engcomp.dethroned_blocks), whichever comes first. A former champion
# is paid only with its own approval (award gate), still committed by its hotkey and with a valid last result; the opening
# co-champion set is ONE former champion, each member 1/n of the slot with its own approval. An unpaid slot (no approval,
# withdrawn, invalid) goes to the inference part, never to another rank. While rank 1 waits for its approval
# (ENGINE_RESTORE_PREVIOUS), the most recently crowned approved former champion takes rank 1's share (its own decay) and
# its own slot is not paid on top: it gets 1 - (the other held slots), i.e. the whole share when no other slot is held
# (decision 1). The engine keep-alive is deducted from all paid engine shares in proportion; a former champion that holds
# a slot but is not paid gets the protected engine keep-alive (ENGINE_KA_PROTECTED_MAX, REAX_ENGINE_KEEPALIVE).
ENGINE_RANK_SPLIT = TRACK_RANK_SPLIT
ENGINE_FORMER_SLOT_BLOCKS = 7 * 7200  # 7 days at 12 s per block
ROTATION_DAYS = 14


@dataclass(frozen=True)
class SplitProfile:
    name: str
    engine: float  # fraction of the miner part reserved for the engine champion
    tracks: tuple[float, ...]  # one entry per active use-case track slot

    @property
    def competition(self) -> float:
        return self.engine + math.fsum(self.tracks)

    @property
    def inference(self) -> float:
        return 1.0 - self.competition


PROFILES: dict[str, SplitProfile] = {
    "80/20": SplitProfile("80/20", engine=0.10, tracks=(0.10,)),
    "50/50": SplitProfile("50/50", engine=0.20, tracks=(0.15, 0.15)),
}


@dataclass(frozen=True)
class TrackSchedule:
    track_id: str
    start: datetime
    end: datetime  # exclusive

    def active(self, now: datetime) -> bool:
        return self.start <= now < self.end


_PAIR_1_END = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)

# Release constant and the ONE source of truth for which use-case tracks run when (the specialist evaluator,
# reax_subnet.spec.config, derives its schedule from this list). First entry = the only track of the 80/20 profile.
def release_schedule() -> tuple[TrackSchedule, ...]:
    end = min(_PAIR_1_END, GENERALIST_START_UTC)
    return (
        TrackSchedule("demand_forecasting", COMPETITIONS_START_UTC, end),
        TrackSchedule("risk_assessment", COMPETITIONS_START_UTC, end),
        TrackSchedule("generalist", GENERALIST_START_UTC, ENGINE_HELD),
    )


TRACK_SCHEDULE = release_schedule()

# Release constant, per track: submissions scoring at or below the floor earn nothing from the track; None = no floor.
# Entry floor = the s1-fast reference model's (Plumb-4B) score + 1.5 points. 1.2.0rc7.post2 (sealed pools refilled with
# 1,116 / 1,095 new LLM-written rows, 17 Oct 2026): measured with the validator's evaluator on the competition worker box
# (H100; on the same 1800 items it scored 0.47 / 0.37 above the RTX PRO 6000 on 16 Oct) on six consecutive 300-item
# draws of the release sampler from the refilled pools and the live use counts of 17 Oct 03:09Z (so mostly new rows):
# demand_forecasting 27.32, risk_assessment 28.54 (1800 items each). On the same draws a model that is perfect on the
# procedural items and uniform elsewhere reaches at most 20.7 / 21.6 per draw. 1.2.0rc6 (draws of 300 items, 15 % proc,
# owner decision 16 Oct 2026): measured with the validator's evaluator (reax-spec-eval, reference validator RTX PRO 6000)
# on six consecutive 300-item draws of the release sampler from the deployed pools (writer-note rows removed) and the
# live use counts of 16 Oct 14:05Z: demand_forecasting 38.17, risk_assessment 31.66 (1800 items each) -> 39.7 / 33.2.
# 1.2.0rc5 (600 items, 50 % proc, full pool): 23.28 / 32.19 -> 24.8 / 33.7.
# A specialist must beat the general model that is already served; at or below the floor a submission is not ranked.
TRACK_ENTRY_FLOOR: dict[str, Optional[float]] = {"demand_forecasting": 28.8, "risk_assessment": 30.0,
                                                  "generalist": GENERALIST_ENTRY_FLOOR}

# Participation keep-alive (docs/SPECIALIST-TRACKS.md, "Keep-alive"), release constant. Each keep-alive entry of an
# active track (copy-checked submissions, still committed, that scored above the entry floor on a real draw; at most 2
# per coldkey) is paid like a typical inference miner: the MEDIAN of the positive serving weights (see ``combine_weights``),
# BEFORE the ranks are paid from the rest of the track's share. The track's keep-alive total is bounded by this fraction
# of the track's share; when ``n x median`` would exceed it, each entry gets ``KEEPALIVE_MAX_SHARE x share / n``. A
# competition hotkey cannot serve inference, so without it an entrant earns 0 while it is evaluated, waits for its
# confirmation or for the owner's approval, and is deregistered once its immunity ends. Not an award: the owner-approval
# gate does not apply. No entry: nothing is reserved and the weights are exactly the rc5 weights. 0 = off.
# (Before 1.2.0rc7.post1 this was ``KEEPALIVE_SHARE = 0.05``: a fixed 5 % of the track share split evenly.)
KEEPALIVE_MAX_SHARE = 0.25

# At most this many keep-alive entries per track are paid (mechanism v2), release constant. The share cap above is
# already reached at 8 entries on demand_forecasting, so every further entry (e.g. cheap fine-tunes of a public
# above-floor model on fresh coldkeys) would only dilute the honest waiting entries toward the prune line. When more
# entries qualify, the 8 with the highest qualifying score (the non-proc score; the total for an entry scored before
# 1.2.0rc6) are paid, ties by earlier commitment block, then uid; each gets min(median, cap / n_paid).
KEEPALIVE_MAX_ENTRIES = 8

# Provisional keep-alive (mechanism v2, anti-churn), release constants. An entry that waits for its FIRST evaluation has
# no score, so it earns no keep-alive, and once its immunity (7200 blocks) is over it is the lowest-emission UID and the
# chain prunes it before it was ever scored. While all of these hold, such an entry gets a provisional slot
# (``spec.evaluator.compute_provisional``): queued (or waiting for this host's infrastructure) and not yet scored,
# prepared and checked (fetched, rule checks, exact tensor-hash check, copy check done), still committed, its hotkey's
# immunity ends within PROVISIONAL_KA_LEAD_BLOCKS (600 blocks, ~2 h: weights lag the chain by 1-2 h) or is over, at most
# PROVISIONAL_KA_MAX_HOURS since it first became eligible, at most one per coldkey and track. Pay: PROVISIONAL_KA_FRACTION
# x the median positive serving weight (the median of the keep-alive), all provisional entries of a track together at
# most PROVISIONAL_KA_MAX_SHARE of the track's share (then split evenly), taken from the track's share like the keep-alive
# (before the ranks), never from another track, and not counted against KEEPALIVE_MAX_ENTRIES. Unclaimed: serving.
# Why it cannot be farmed: a junk entry gets at most 12 h x 0.5 median (about 0.14 TAO at today's prices), and only near
# or after the end of its own immunity, against a registration of at least 0.5 TAO; the immunity-aware queue evaluates
# expired entries first, and a first draw below the keep-alive floors ends the provisional slot. PROVISIONAL_KA = False
# turns it off.
PROVISIONAL_KA = True
PROVISIONAL_KA_LEAD_BLOCKS = 600
PROVISIONAL_KA_MAX_HOURS = 12.0
PROVISIONAL_KA_FRACTION = 0.5
PROVISIONAL_KA_MAX_SHARE = 0.05

# Keep-alive non-proc floor (1.2.0rc6), release constant: a keep-alive entry must also score above this on the NON-proc
# items of its latest draw (same_family, new_domain, new_family; tier != "proc_sealed"), so learning the public
# procedural families alone never earns it. Value: the reference model's (Plumb-4B = s1-fast) score on the non-proc items
# of the same six release draws as TRACK_ENTRY_FLOOR + 1.5 points: 1.2.0rc7.post2 (refilled pools) demand_forecasting
# 31.10, risk_assessment 32.95 (1530 items each); 1.2.0rc6 48.34 / 36.27 -> 49.8 / 37.8.
KEEPALIVE_NONPROC_FLOOR: dict[str, Optional[float]] = {"demand_forecasting": 32.6, "risk_assessment": 34.5,
                                                     "generalist": GENERALIST_KEEPALIVE_NONPROC_FLOOR}

# Keep-alive legacy floor (1.2.0rc7), release constant: an entry whose latest score was taken before 1.2.0rc6 has no
# non-proc score (``Submission.nonproc`` None and no per-tier scores); it qualifies when that total score is above this
# instead. Value: the score of a model that is perfect on the procedural items and uniform elsewhere under the rc5 mix
# (50 % proc, 600 items, the deployed pools: demand_forecasting 49.8, risk_assessment 31.8, workers/leak-audit), i.e. the
# best a rules-only model reaches without any skill on the non-proc tiers, plus a margin (10.2 / 18.2 points).
KEEPALIVE_LEGACY_FLOOR: dict[str, Optional[float]] = {"demand_forecasting": 60.0, "risk_assessment": 50.0}

# Keep-alive base floor (1.2.0rc9.post6), release constant, open class only (from ``SPEC_OPEN_CLASS_START_UTC``): an entry
# with more than ``KEEPALIVE_BIG_PARAMS`` parameters qualifies for the keep-alive (and the provisional keep-alive) only if
# this table has its declared base's tensor hash (``Submission.base_id``) for the track and the score that qualified it (the
# non-proc score, the total for a legacy entry) is above that floor + ``KEEPALIVE_BIG_MARGIN``. A larger model must beat
# its own base, not only a rules-only model. Maps base tensor hash -> {track id: floor}. Ships EMPTY (filled after
# calibration): until then no entry above 10 B parameters gets a keep-alive. A larger entry without a recorded base hash gets
# none; an entry without a recorded parameter count (fetched by an earlier release, under the 9 B cap, or not fetched yet) counts as small. An entry with at most ``KEEPALIVE_BIG_PARAMS`` parameters is not affected.
KEEPALIVE_BIG_PARAMS = 10_000_000_000
KEEPALIVE_BIG_MARGIN = 1.5
# REAX 1 rows, on the r8 footing (shipped pplx floor 58.01 + the paired non-proc offset vs pplx on the same 6 draws of the final
# pool, 21 Oct): decisionmodels/reax-1@ca2a0cc5 (4031ce5e..., the official base) 60.43 (+2.42); the earlier public revision
# @ea9d5d8b (572884f6..., readout not rescaled) 55.34 (-2.67).
KEEPALIVE_BASE_FLOOR: dict[str, dict[str, float]] = {'0e8d23478b5b6ccefff3136e5e53ecfb7d053451ec405d2b0fe31ce3a49a713e': {'generalist': 39.23}, '1647929265009944f8157465e27417ec84014893953f3b3ed0874e373bb276fc': {'generalist': 62.23}, '19d89e343b9109fcb478d685cb08f15914c4aded296d02c68c2c3fa9f0fdb4df': {'generalist': 39.93}, '4031ce5ec8b6f10e388f09df862810315e4132f429f8cdf308f13566bd159eef': {'generalist': 60.43}, '564471ef18768dd040eb3ce7effaf0561b65b3b2a50cb10980974bae3ae713dd': {'generalist': 55.01}, '572884f6e3788b5debd2084e2789a009622e50272fe6c226e1d93dcb24ec6d41': {'generalist': 55.34}, '81a3eb978ff3c5b90c273a100f36499a86dfec698652c2d181f43e11e89fe08d': {'generalist': 58.01}, 'dac9897fe790514fec2d32f666b5627e098205b051b48fe6630a3699a5138e10': {'generalist': 49.52}}

# Opening generalist cohort: playoff places 2 and 3 (Florian 21 Oct 2026 ~14:45Z, board #23036, "ErPing's proposal"). Of the
# generalist track's opening cohort playoff (the first cohort playoff on the vacant track after GENERALIST_START_UTC), the
# entries in places 2 and 3 of the playoff's own ranking take the track's two former-champion ranks (17 % / 8 % of the track's
# share, TRACK_RANK_SPLIT; no extra emission) while no real former champion holds them, if they (a) score above the track
# entry floor (TRACK_ENTRY_FLOOR["generalist"]), (b) reach KEEPALIVE_BASE_FLOOR[declared base]["generalist"] +
# OPENING_PLAYOFF_BASE_MARGIN (strictly above) on the non-proc score of that playoff (the total for a legacy entry), an entry whose base has no
# floor does not qualify, (c) are no copy/twin (``derived_from``), withdrawn, pruned or rejected entry, and (d) pass the normal
# award review (award_ok, unchanged). Each slot starts at max(the crowning, this instant) and ends OPENING_PLAYOFF_SLOT_DAYS
# later or when newer former champions push it out of the MAX_LEADERS ranks. None = off: the weights are exactly the ones
# without this rule. No env override: validators must agree. The release owner sets the date in the release notes.
OPENING_PLAYOFF_SLOTS_START_UTC: Optional[datetime] = datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)
OPENING_PLAYOFF_PLACES = 2          # places 2 and 3 of the playoff (place 1 is the champion)
OPENING_PLAYOFF_SLOT_DAYS = 7
OPENING_PLAYOFF_BASE_MARGIN = 1.5   # points above the declared base's floor



# Generalist opening co-champions (1.2.0rc11.post1; Florian 22 Oct 2026 ~12:2xZ, DECISIONS): uid 116, 199 and 250 are
# co-champions of the generalist opening, equal thirds of the champion's rank-1 share (TRACK_RANK_SPLIT[0]) until a new
# entry beats the best of them (the worker's challenger bar is the best co-champion on the challenger's own draw,
# spec.evaluator._decide). The listed keys are "hotkey|commitment text" of the submissions. While the state's generalist
# champion is one of them each listed entry that is still committed (its hotkey holds a uid and commits exactly that text),
# not withdrawn or rejected, and has its OWN owner approval (award_ok, per hotkey and submission) is paid one third of
# rank 1's share; an unapproved or dropped member's third goes to the inference part, never to the other members. A
# member takes no keep-alive slot while its third is paid (an unapproved one keeps the champion's protected slot). Once
# a challenger holds the title the set ends and every entry is an ordinary one. The spec track has no champion decay,
# so GENERALIST_CO_CHAMPIONS_CROWNED (the opening playoff's crowning) only starts the members' keep-alive window.
# Empty = off: exactly the weights without this rule.
GENERALIST_CO_CHAMPIONS: tuple[str, ...] = (
    "5GNAwERfaf3Rd5bFrg2JJvyo6L53AovkAv2HEALLkGNDxGhF|reax-spec/1:generalist:Andrew93110/sn92-gen-v1@4d2a519d78377e63c46cbf35c26b09a71c28adf8",  # uid 116
    "5CwV6dNFeyNqY6FQmZm9bQdta3qWe2xGuFWtXrS198N5K6r1|reax-spec/1:generalist:putty77/sn92-generalist-a@dbeec5635aa717e4654b8a719499a0ce80c370ea",  # uid 199
    "5DUJxXJpZrFKrTLNAWw3H6FT1Du1z2zLNzEg3vzG32t6WTos|reax-spec/1:generalist:reax-knockout/sn92-gen-a1@d57f18b92547b9c669bf93fa74ee64443ddb45d8",  # uid 250
)
GENERALIST_CO_CHAMPIONS_TRACK = "generalist"   # the track the set belongs to (a constant so the tests can run it on a small track)
GENERALIST_CO_CHAMPIONS_CROWNED: Optional[datetime] = datetime(2026, 10, 22, 9, 7, 25, tzinfo=timezone.utc)

# Crown slots (1.2.0rc11.post1/post2). When non-empty, a challenger decision that would crown on a specialist track is stored as
# "pending crown" (scores published, the champion unchanged; the other challengers keep being evaluated against the current
# bar) and finalised at the first instant in this list that is >= the decision: one stored entry is crowned, several run a
# slot playoff (spec.evaluator._run_playoff slot=True, up to three shared draws). A decision after the last instant crowns at
# once. Empty = a decision crowns at once, as before.
def _crown_slots(first: datetime, last: datetime, weekdays: tuple[int, ...] = (0, 3)) -> tuple[datetime, ...]:
    """Every ``weekdays`` (0 = Monday, 3 = Thursday) at the time of day of ``first``, from ``first`` through ``last``."""
    out, day = [], first
    while day <= last:
        if day.weekday() in weekdays:
            out.append(day)
        day += timedelta(days=1)
    return tuple(out)


# rc11.post3 (Florian #24089, slots-mon-thu-1430z): every Monday AND Thursday 14:30Z from Mon 26 Oct 2026 (26 Oct, 29 Oct,
# 2 Nov, 5 Nov, ...), generated through 31 Mar 2027; the same rules at every slot. Fixed UTC (15:30 Berlin in winter time).
CROWN_SLOTS_UTC: tuple[datetime, ...] = _crown_slots(datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc),
                                                     datetime(2027, 3, 31, 23, 59, tzinfo=timezone.utc))


# Engine titles change on Tuesday and Friday, independently of measurement batches. Empty = legacy instant crowns.
ENGINE_CROWN_SLOTS_UTC: tuple[datetime, ...] = _crown_slots(
    datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc),
    datetime(2027, 3, 31, 23, 59, tzinfo=timezone.utc), weekdays=(1, 4))


# Shared leading group at a crown slot (1.2.0rc11.post3; CRITERIA v2 and the public slot rules, reax.co/competitions/faq/#slot-rules;
# Florian's decisions 22 Oct, card 18227 and DECISIONS tie-numbers-k). Every knob is ONE constant with the approved default, so a
# change is a one-liner (it must be announced 48 h before the first slot it applies to, slot rules section 1).
SLOT_MAX_ROUNDS = 3                 # shared rounds (fresh draws, the track's normal draw size: 900 items on v4) per slot
SLOT_G_FLOOR_FRACTION = 0.5         # K8: G = this x the margin floor of the pinned evaluation (0.5 x 1.0 = 0.5 points on v4)
SLOT_K8B_LEADER_ALONE = True        # K8b-a: not separated but every other member more than G behind -> the leader is crowned alone.
                                    # False (K8b-b): nobody is crowned and the entries go back to the challenger path
SLOT_OWNER_COLDKEY_OR_HF = True     # K4-a: one entry per holder = same coldkey OR same Hugging Face account/org; False = coldkey only (K4-b)
SLOT_GROUP_CAP: Optional[int] = None  # K6: no cap on the shared group (an int n keeps the n best pooled means)
SLOT_K1_A = True                    # K1-a: a confirmation due at or before slot - SLOT_K1_DUE_LEAD_S that has not run by the slot is
                                    # run first, on the slot's evaluation, and joins the slot if it passes. False = K1-b
SLOT_K1_DUE_LEAD_S = 3600.0         # "due" = first pass + CONFIRM_DELAY_S at or before slot - this (13:30:00Z for a 14:30Z slot)
SLOT_CLOCK_S = 12 * 3600.0          # K7: everything of a slot (K1-a confirmations, rounds, retries) finishes by slot + this
SLOT_VOID_RETRIES = 2               # K7: a round voided by our side (every member failed / host fault) is re-run on new items at most this often
SLOT_SERVE_RETRIES = 1              # rule 4.1: a member whose model fails on a round is retried this often on the same items
SLOT_STOP_WHEN_SEPARATED = True     # K2: stop as soon as one member is left (False: always run SLOT_MAX_ROUNDS rounds)
SLOT_RESULTS_KEEP = 40              # slot result records kept in the state (TrackState.slot_results, newest last)


class RankedLeaders(list):
    """``[(uid, score), ...]`` already in rank order, rank 1 first (docs/SPECIALIST-TRACKS.md, "Ranks"): the specialist
    evaluator's champion, then its former champions, most recently crowned first. ``track_payout`` pays it in this
    order and never re-sorts it by score; the scores come from different draws and are not comparable.

    ``crowned``: uid -> when that entry was crowned champion (unix seconds), for the entries that have such a time (the
    champion; a former champion keeps the time it was crowned). Read only by the keep-share rule (``APPROVAL_GRACE_S``).

    ``cochamps``: uid -> the fraction of rank 1's share that uid is paid, for the members of the opening co-champion set
    (``GENERALIST_CO_CHAMPIONS``) that are still committed (1/n each, n = the size of the set; a dropped member's fraction is
    not listed and not redistributed). Empty = rank 1 is one entry. The first entry of the list is then a member.

    ``formset`` (1.2.0rc11.post2): the same for the opening co-champion set AFTER a challenger beat it. The set is one former
    champion: it takes ONE place in the list (its first committed member stands for it, placed by
    ``GENERALIST_CO_CHAMPIONS_CROWNED``) and that place's share is split 1/n over ``formset`` (n = the size of the set; a
    dropped member's fraction is not redistributed). Empty = no such set.

    ``formsets`` (1.2.0rc11.post3): the same for every shared group crowned at a slot and later beaten (each its own
    ``{uid: 1/n}``, n = the size of the group); each is ONE former champion with one place in the list. Empty = none.

    ``cochamps_dynamic`` (1.2.0rc11.post3): True when ``cochamps`` is a shared group crowned at a slot (the state's
    ``TrackState.cochamps``), False for the hard-coded opening set (no keep-share rule for it, as in rc11.post2)."""

    crowned: dict[int, float]
    cochamps: dict[int, float]
    formset: dict[int, float]
    formsets: list[dict[int, float]]
    cochamps_dynamic: bool

    def __init__(self, *args):
        super().__init__(*args)
        self.crowned = {}
        self.cochamps = {}
        self.formset = {}
        self.formsets = []
        self.cochamps_dynamic = False


# Per track: [(uid, score)]. A ``RankedLeaders`` (what the specialist evaluator returns) is paid in its order; any
# other list or mapping is ranked by score, best first.
TrackResults = dict[str, list[tuple[int, float]]]


class SpecialistEvaluator(Protocol):
    """Plug-in point for the use-case tracks. ``track_results`` must be cheap (read the evaluator's own state)."""

    def track_results(self, track_ids: Sequence[str], now: datetime) -> TrackResults: ...


class NoSpecialistEvaluator:
    """Default: no track has results, so every track share goes to the inference part."""

    def track_results(self, track_ids: Sequence[str], now: datetime) -> TrackResults:
        return {}


def spec_open_class_active(now_ts: Optional[float] = None) -> bool:
    """True from ``GENERALIST_START_UTC`` on (``now_ts``: unix seconds, default the wall clock)."""
    return (time.time() if now_ts is None else now_ts) >= GENERALIST_START_UTC.timestamp()


def competitions_active(now: datetime) -> bool:
    return now >= COMPETITIONS_START_UTC


def release_engine_start() -> datetime:
    """The engine track's start on finney: ``ENGINE_HELD`` while ``ENGINE_TRACK_START_UTC`` is None."""
    return ENGINE_HELD if ENGINE_TRACK_START_UTC is None else ENGINE_TRACK_START_UTC


def engine_track_open(now: datetime) -> bool:
    return now >= release_engine_start()


def profile_for(name: str = SPLIT_PROFILE, now: Optional[datetime] = None) -> SplitProfile:
    if name == "50/50" and now is not None and now >= GENERALIST_START_UTC:
        return SplitProfile(name, engine=0.20, tracks=(0.30,))
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown split profile {name!r}; choose from {', '.join(PROFILES)}") from None


def active_tracks(now: datetime, profile: str = SPLIT_PROFILE,
                  schedule: Optional[Sequence[TrackSchedule]] = None) -> list[str]:
    """Track ids that are active at ``now``, in schedule order, at most one per slot of the profile."""
    slots = len(profile_for(profile, now).tracks)
    return [s.track_id for s in (release_schedule() if schedule is None else schedule) if s.active(now)][:slots]


def legacy_ka_tracks(now: datetime, profile: str = SPLIT_PROFILE) -> tuple[str, ...]:
    """The ended tracks whose leaders get the legacy keep-alive at ``now``: ``LEGACY_KA_TRACKS`` that are not active,
    inside [GENERALIST_START_UTC, GENERALIST_START_UTC + LEGACY_KA_DAYS); otherwise ()."""
    if not GENERALIST_START_UTC <= now < GENERALIST_START_UTC + timedelta(days=LEGACY_KA_DAYS):
        return ()
    active = set(active_tracks(now, profile))
    return tuple(t for t in LEGACY_KA_TRACKS if t not in active)


def legacy_ka_uids(leaders: Mapping[str, Sequence[tuple[int, bool]]], track_ids: Sequence[str],
                   award_ok: Optional[Callable[[str, int], bool]] = None) -> list[int]:
    """Legacy keep-alive candidates from ``leaders`` (track id -> ``[(uid, is_champion), ...]`` in rank order,
    ``spec.evaluator.legacy_leaders``): per track in ``track_ids`` order, the champion always, a former champion only if
    ``award_ok(track_id, uid)`` (None = no gate); each uid once. Not cut: ``combine_weights`` drops uids paid elsewhere
    first, then keeps ``LEGACY_KA_MAX_ENTRIES``. Pure."""
    out: list[int] = []
    for track_id in track_ids:
        for uid, champ in leaders.get(track_id, ()):
            if uid not in out and (champ is True or award_ok is None or award_ok(track_id, uid)):
                out.append(uid)
    return out


def engine_decay_fraction(age_s: float) -> float:
    """The champion's part of the engine share: 1.0 when crowned, linear to 0.5 after 14 days, then flat."""
    frac = min(1.0, max(0.0, age_s) / (ENGINE_DECAY_DAYS * 86400.0))
    return 1.0 - (1.0 - ENGINE_DECAY_FLOOR) * frac


# Keep-share rule (REAX_SPEC_NC_KEEP_SHARE, off by default; docs/SPECIALIST-TRACKS.md "Owner approval"). While a track's
# rank 1 was crowned less than ``APPROVAL_GRACE_S`` ago and has no approval yet, its share is paid to the most recently
# crowned former champion that is approved (the next ranks, in rank order), who then gives up its own rank share (that
# share goes to the inference part as today). Afterwards, or once rank 1 is approved, the ranks are paid as before.
APPROVAL_GRACE_S = 48 * 3600.0


def keep_share_enabled(env: Mapping[str, str]) -> bool:
    """REAX_SPEC_NC_KEEP_SHARE: strict on/off switch, unset = off; an unknown value is refused (as the other flags)."""
    raw = (env.get("REAX_SPEC_NC_KEEP_SHARE") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"REAX_SPEC_NC_KEEP_SHARE must be 1/0 (or true/false, yes/no, on/off), got {raw!r}")


def engine_keepalive_enabled(env: Mapping[str, str]) -> bool:
    """REAX_ENGINE_KEEPALIVE: strict on/off switch for the engine keep-alive, unset = off; an unknown value is refused."""
    raw = (env.get("REAX_ENGINE_KEEPALIVE") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"REAX_ENGINE_KEEPALIVE must be 1/0 (or true/false, yes/no, on/off), got {raw!r}")


# a crowning time further ahead of the validator clock than this is treated as unknown (no keep-share rule)
CROWN_FUTURE_SLACK_S = 600.0


def keep_share_ranks(ranks: Mapping[int, float], crowned: Optional[Mapping[int, float]], now: float,
                     approved: Callable[[int], bool]) -> dict[int, float]:
    """``ranks`` (uid -> share, rank 1 first, from ``track_payout``) with the keep-share rule applied. A pure function of
    the ranks, the crowning times, ``now`` and the approvals: validators that see the same state compute the same
    vector. Unchanged unless rank 1 is unapproved, was crowned (known time) less than the grace ago, and a later rank
    is approved."""
    items = list(ranks.items())
    if len(items) < 2 or crowned is None or approved(items[0][0]):
        return dict(ranks)
    since = crowned.get(items[0][0])
    if not isinstance(since, (int, float)) or not math.isfinite(since) or now - since >= APPROVAL_GRACE_S \
            or since > now + CROWN_FUTURE_SLACK_S:  # a crowning time in the future (clock skew, corrupt state) is unknown
        return dict(ranks)
    for i in range(1, len(items)):
        if approved(items[i][0]):
            # the approved former champion takes rank 1's share and gives up its own rank share (that stays with the
            # inference part); the unapproved crown keeps rank 1 (it is not paid anyway: the award gate skips it)
            out = {uid: share for j, (uid, share) in enumerate(items) if j != i}
            out[items[i][0]] = items[0][1]
            return out
    return dict(ranks)


def keep_share_set_ranks(ranks: Mapping[int, float], scores: Any, floor: Optional[float], now: float,
                         approved: Callable[[int], bool]) -> dict[int, float]:
    """The keep-share rule when rank 1 and/or a former-champion place is a shared group (rc11.post3; slot rules section 8,
    CRITERIA 5.3). ``ranks`` = ``rank_shares(scores, floor)``; ``scores`` = the ``RankedLeaders`` (order, crowning times,
    ``cochamps`` / ``formset`` / ``formsets``). A unit is one place: rank 1's group (``cochamps``) or its one entry, a beaten
    group (ONE former champion) or one former champion. For up to ``APPROVAL_GRACE_S`` from rank 1's crowning, the part of
    rank 1's share that belongs to its members NOT yet approved goes to the first later unit with an approved member,
    split like that unit's own place (1/n per member of a group; the award gate still skips an unapproved member, whose
    part stays with the inference part), INSTEAD of the unit's own place share (which goes to the inference part).
    Unchanged when every member of rank 1 is approved, the crowning time is unknown / older than the grace / in the
    future, or no later unit has an approved member. For a single champion beating a single champion this is exactly
    ``keep_share_ranks``. Pure."""
    items = list(track_payout(scores, floor).items())
    co = getattr(scores, "cochamps", None) or {}
    sets = [fs for fs in [getattr(scores, "formset", None), *(getattr(scores, "formsets", None) or [])] if fs]
    order = [u for u, _ in scores] if not isinstance(scores, Mapping) else list(scores)
    units: list[dict[int, float]] = []
    for i, (rep, _) in enumerate(items):
        if i == 0 and co and rep in co:
            units.append({u: min(1.0, f) for u, f in co.items()})
            continue
        fs = next((f for f in sets if rep in f and next((u for u in order if u in f), None) == rep), None)
        units.append({u: min(1.0, f) for u, f in fs.items()} if fs is not None else {rep: 1.0})
    crowned = getattr(scores, "crowned", None)
    if len(units) < 2 or crowned is None:
        return dict(ranks)
    pending = math.fsum(ranks.get(u, 0.0) for u in units[0] if not approved(u))
    since = crowned.get(items[0][0])
    if pending <= 0 or not isinstance(since, (int, float)) or not math.isfinite(since) or now - since >= APPROVAL_GRACE_S \
            or since > now + CROWN_FUTURE_SLACK_S:
        return dict(ranks)
    for unit in units[1:]:
        if any(approved(u) for u in unit):
            out = {u: share for u, share in ranks.items() if u not in unit}
            for u, frac in sorted(unit.items()):
                out[u] = pending * frac
            return out
    return dict(ranks)


def track_payout(scores: Mapping[int, float] | Sequence[tuple[int, float]],
                 floor: Optional[float] = None) -> dict[int, float]:
    """75/17/8 for ranks 1-3. A ``RankedLeaders`` keeps its order (a uid counts at its first place); anything else is
    ranked best score first, ties by lower uid. Missing ranks are NOT redistributed: the returned fractions sum to at
    most 1 and the caller sends the rest to the inference part. A score at or below ``floor`` (``TRACK_ENTRY_FLOOR``;
    None = no floor), or not a positive finite number, is not ranked; the entries after it move up one rank."""
    in_order = isinstance(scores, RankedLeaders)
    items = scores.items() if isinstance(scores, Mapping) else scores
    best: dict[int, float] = {}
    for uid, s in items:
        if not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 or (floor is not None and s <= floor):
            continue
        if uid in best and (in_order or s <= best[uid]):
            continue
        best[uid] = float(s)
    ranked = list(best) if in_order else sorted(best, key=lambda u: (-best[u], u))
    return {uid: share for uid, share in zip(ranked[: len(TRACK_RANK_SPLIT)], TRACK_RANK_SPLIT)}


def rank_shares(scores: Mapping[int, float] | Sequence[tuple[int, float]], floor: Optional[float] = None) -> dict[int, float]:
    """``track_payout`` with the co-champion sets applied: rank 1's share is split over ``scores.cochamps`` (a
    ``RankedLeaders``; empty = no set, exactly ``track_payout``), only when rank 1 is itself a listed member; the place of a
    beaten set (``formset`` / ``formsets``, each ONE former champion) is split 1/n over its members."""
    ranks = track_payout(scores, floor)
    sets = [fs for fs in [getattr(scores, "formset", None), *(getattr(scores, "formsets", None) or [])] if fs]
    reps: dict[int, Mapping[int, float]] = {}   # the beaten sets (rc11.post2/post3): one former-champion place, split 1/n
    for fs in sets:
        rep = next((u for u, _ in scores if u in fs), None)
        if rep in ranks:
            reps[rep] = fs
    co = getattr(scores, "cochamps", None)
    items = list(ranks.items())
    co_on = bool(co) and bool(items) and items[0][0] in co
    if not ranks or not (reps or co_on):
        return ranks
    out: dict[int, float] = {}
    for i, (uid, share) in enumerate(items):
        if i == 0 and co_on:
            for m, frac in sorted(co.items()):
                out[m] = out.get(m, 0.0) + share * min(1.0, frac)
        elif uid in reps:
            for m, frac in sorted(reps[uid].items()):
                out[m] = out.get(m, 0.0) + share * min(1.0, frac)
        else:
            out[uid] = out.get(uid, 0.0) + share
    return out


def keepalive_uids(entries: Sequence[tuple[int, float]], floor: Optional[float] = None,
                   owner_uid: Optional[int] = None) -> list[int]:
    """The uids of a track's keep-alive entries ``[(uid, score), ...]`` (the specialist evaluator's ``keepalive``) that
    get a slot: a positive finite score above ``floor``, never the owner UID, each uid once, in the given order."""
    out: list[int] = []
    for uid, s in entries:
        if not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 or (floor is not None and s <= floor):
            continue
        if uid == owner_uid or uid in out:
            continue
        out.append(uid)
    return out


def combine_weights(serving: Mapping[int, float], engine_champion: Optional[int], engine_share_frac: float,
                    tracks: Mapping[str, Mapping[int, float] | Sequence[tuple[int, float]]],
                    profile: str = SPLIT_PROFILE, *, owner_uid: Optional[int] = None,
                    burn_fraction: float = 0.0,
                    floors: Mapping[str, Optional[float]] = TRACK_ENTRY_FLOOR,
                    award_ok: Optional[Callable[[str, int], bool]] = None,
                    keepalive: Optional[Mapping[str, Sequence[tuple[int, float]]]] = None,
                    keepalive_share: Optional[float] = None,
                    provisional: Optional[Mapping[str, Sequence[int]]] = None,
                    engine_waiting: Optional[Sequence[int]] = None,
                    keep_share: bool = False, now: Optional[float] = None,
                    engine_cochampions: Optional[Mapping[int, float]] = None,
                    legacy_keepalive: Optional[Sequence[int]] = None,
                    keepalive_protected: Optional[Mapping[str, Sequence[int]]] = None) -> dict[int, float]:
    """Mix the competitions into the serving weights.

    ``keep_share`` (with ``now``, unix seconds; needs ``award_ok``): the keep-share rule above (``keep_share_ranks``).
    False = exactly the weights without it.

    ``serving``: the 1.1 weight vector (may contain the owner's burn entry). ``engine_champion``: uid or None;
    ``engine_share_frac``: the champion's part of the engine share (1.0 .. 0.5, see ``engine_decay_fraction``).
    ``tracks``: scores per ACTIVE track (only the first ``len(profile.tracks)`` track ids in sorted order are paid).

    ``award_ok(track_id, uid)``: the owner-approval gate (award.py), asked AFTER the ranking: an entry without approval
    keeps its rank, and its share stays with the inference part (the next rank is not promoted). None = no gate.

    ``keepalive``: per active track, the keep-alive entries ``[(uid, score), ...]`` (``keepalive_uids``). Each entry of
    a track is paid ``per_entry`` = the median of the positive serving weights of ``serving`` (normalised, as a fraction
    of the miner part; computed before any competition share is applied), excluding the owner UID and every uid that is
    itself competition-paid (the engine champion, all keep-alive entries, all uids in the active tracks' ranks). The
    median of an even count is the mean of the two middle values; no such serving uid: ``per_entry`` is unbounded and
    the cap decides. The track's keep-alive total is at most ``keepalive_share`` x the track's share (default
    ``KEEPALIVE_MAX_SHARE``): if ``n x per_entry`` exceeds it, each entry gets ``keepalive_share x share / n``. No approval
    gate applies to it; the ranks are paid from the rest of the track's share (share minus the keep-alive paid). A
    track without entries pays exactly as before. An entry the track's ranks pay and ``award_ok`` approves is paid by
    its rank only: it gets no keep-alive slot. ``keepalive_share`` is that cap fraction (it replaces the fixed
    5 % of rc6/rc7); 0 turns the keep-alive off.

    Dual role (docs/DESIGN-MULTIROLE.md section 5): an entry that also serves has ``own`` = its serving weight in the unit
    of ``per_entry`` (``serving / miner part``, 0 for a pure entrant). The keep-alive is a top-up: the entry is paid
    ``max(0, each - own)`` and the track's rank share is reduced by what is actually paid. Only entries with
    ``own < per_entry`` compete for the ``KEEPALIVE_MAX_ENTRIES`` slots, pure entrants first, then the others, each group
    in the given order. The median leaves out every keep-alive and provisional candidate and every rank of any track.
    Only a track with a payable entry (or ranks) holds one of the profile's slots.
    The provisional pay goes to pure entrants only (it keeps an unserved entry from being pruned; a serving hotkey is not
    at that risk). Without a serving entry the result is exactly the pre-dual-role one.

    ``provisional``: per active track, the uids of the entries that wait for their first evaluation
    (``compute_provisional``; ignored with ``PROVISIONAL_KA`` False). Each is paid ``PROVISIONAL_KA_FRACTION`` x the same
    median as the keep-alive entries; a track's provisional total is at most ``PROVISIONAL_KA_MAX_SHARE`` x the track's
    share (then split evenly), taken from the track's share before the ranks, in addition to the keep-alive. It does not
    use a keep-alive slot (``KEEPALIVE_MAX_ENTRIES``); the owner UID, the engine champion and a uid that holds a
    keep-alive slot or a rank of any track are left out. Unclaimed: stays with ``serving``.

    ``engine_waiting``: uids of engine entries that wait for their first measurement (``engcomp.waiting_uids``; None or
    empty = none). Each is paid ``ENGINE_KA_FRACTION`` x the keep-alive median, at most ``ENGINE_KA_MAX_ENTRIES`` entries
    and ``ENGINE_KA_MAX_SHARE`` x the engine share together, deducted from the engine champion's payment (nothing is
    deducted without a champion). The owner UID and every uid paid elsewhere are left out. Like the provisional
    keep-alive it keeps an unmeasured entry from being pruned, so only pure entrants (no serving weight) get it
    (docs/DESIGN-MULTIROLE.md section 5); every engine-waiting uid is left out of the median.

    ``engine_cochampions``: ``{uid: part}`` of the engine share for approved co-champions (``ENGINE_CO_CHAMPIONS``) and
    approved former champions in their slots (``ENGINE_RANK_SPLIT``), paid like ``engine_champion``; the engine
    keep-alive is deducted from all engine payees in proportion to their payment. None or empty = exactly the weights
    without it.

    ``legacy_keepalive``: uids of the ended DF/RA tracks' leaders (``legacy_ka_uids``, ``LEGACY_KA_*``). Paid like the
    engine keep-alive (pure entrants only: no serving weight, not the owner, nothing else paid), at most
    ``LEGACY_KA_MAX_ENTRIES`` in the given order, each ``min(LEGACY_KA_FRACTION x per_entry, LEGACY_KA_MAX_SHARE / n)``
    of the miner part, taken from the inference part only (``keep``). None or empty = exactly the weights without it.

    The competition fractions apply to the miner part (the owner's burn stays as it is, as in 1.1). Everything not
    claimed stays with ``serving``. The result is normalised to sum 1; an empty ``serving`` returns ``{}``.
    """
    prof = profile_for(profile, datetime.fromtimestamp(now, timezone.utc) if now is not None else None)
    if floors is TRACK_ENTRY_FLOOR and now is not None and TRACK_ENTRY_FLOOR.get("generalist") is not None:
        from .spec import sampling as _sampling    # eval v5: the generalist scores are S_v5.1 (DI scale); the v4-scale release floor is off
        if _sampling.v5_active(now):
            floors = {**TRACK_ENTRY_FLOOR, "generalist": None}
    base_total = math.fsum(w for w in serving.values() if w > 0)
    if base_total <= 0:
        return {}
    base = {uid: w / base_total for uid, w in serving.items() if w > 0}
    burn = burn_fraction if owner_uid is not None and owner_uid in base else 0.0
    miner_part = 1.0 - burn

    paid: dict[int, float] = {}  # fractions of the miner part
    if engine_champion is not None and engine_share_frac > 0:
        paid[engine_champion] = prof.engine * min(1.0, engine_share_frac)
    for uid, part in sorted((engine_cochampions or {}).items()):
        if part > 0 and uid != owner_uid:
            paid[uid] = paid.get(uid, 0.0) + prof.engine * min(1.0, part)
    engine_payees = [u for u in sorted(paid)]  # the champion and/or the co-champions
    ka_cap = KEEPALIVE_MAX_SHARE if keepalive_share is None else keepalive_share
    # a uid's own serving weight in the unit of per_entry (docs/DESIGN-MULTIROLE.md section 5): 0 for a pure entrant
    own = {uid: w / miner_part for uid, w in base.items()}
    cand: dict[str, list[int]] = {}  # keep-alive candidates per track, before the eligibility check and the cut
    if ka_cap > 0:
        for track_id, entries in (keepalive or {}).items():
            uids = keepalive_uids(entries, floors.get(track_id), owner_uid)
            if award_ok is not None and uids:
                # an approved entry is paid by its rank: it does not also hold a keep-alive slot
                ranked = rank_shares(tracks.get(track_id, ()), floors.get(track_id))
                uids = [uid for uid in uids if not (uid in ranked and award_ok(track_id, uid))]
            if uids:
                cand[track_id] = uids
    prov_cand: dict[str, list[int]] = {}  # provisional candidates per track, before the taken-check
    if PROVISIONAL_KA and ka_cap > 0:
        # keepalive_share=0 turns the provisional slots off too
        for track_id, uids in (provisional or {}).items():
            ids = []
            for uid in uids:
                if uid != owner_uid and uid not in ids:
                    ids.append(uid)
            if ids:
                prov_cand[track_id] = ids
    ranks_of = {t: rank_shares(tracks[t], floors.get(t)) for t in tracks}

    def cut(track_id: str, uids: list[int]) -> list[int]:
        # rc10.post5 (Florian via CoS #23485): protected uids (open opening-cohort entries of the vacant generalist track,
        # spec.evaluator.keepalive_protected) keep their slot first, beyond KEEPALIVE_MAX_ENTRIES if needed; the others
        # fill what is left of the KEEPALIVE_MAX_ENTRIES slots in the given order
        prot = set((keepalive_protected or {}).get(track_id, ()))
        if prot:
            keep = [u for u in uids if u in prot]
            rest = [u for u in uids if u not in prot]
            room = max(0, KEEPALIVE_MAX_ENTRIES - len(keep))
            if len(rest) > room:
                log.info("track %s: keep-alive for %d protected cohort entries and %d more, %d qualifying entries get none: "
                         "uids %s", track_id, len(keep), room, len(rest) - room, rest[room:])
            return keep + rest[:room]
        if len(uids) > KEEPALIVE_MAX_ENTRIES:
            # ``keepalive`` arrives in rank order (compute_keepalive): entries outside the top ones get nothing
            log.info("track %s: keep-alive for %d entries only, %d qualifying entries get none: uids %s",
                     track_id, KEEPALIVE_MAX_ENTRIES, len(uids) - KEEPALIVE_MAX_ENTRIES, uids[KEEPALIVE_MAX_ENTRIES:])
            return uids[:KEEPALIVE_MAX_ENTRIES]
        return uids

    # The median leaves out every keep-alive and provisional candidate and every rank of ANY track (eligible or not, cut or
    # not, with a slot or not), the engine champion and the owner: a dual-role uid's serving weight must not move the
    # pay of the entrants, and per_entry does not depend on which tracks hold a slot, so the eligibility can be decided
    # before the slots (critic round 2 NEW-2). For pure entrants (no serving weight) this is the median as before.
    ewait_cand = [u for u in dict.fromkeys(engine_waiting or ()) if u != owner_uid] if ka_cap > 0 else []
    legacy_cand = [u for u in dict.fromkeys(legacy_keepalive or ()) if u != owner_uid] if ka_cap > 0 else []
    per_entry = math.inf
    # a payable legacy candidate has no serving weight, so it never moves the median (it is not added to ``competing``)
    if cand or prov_cand or ewait_cand or legacy_cand:
        competing = ({u for ids in cand.values() for u in ids} | {u for ids in prov_cand.values() for u in ids}
                     | set(paid) | set(ewait_cand))
        for ranked_t in ranks_of.values():
            competing.update(ranked_t)
        competing.add(owner_uid)
        typical = sorted(w for uid, w in base.items() if uid not in competing)
        if typical:
            mid = len(typical) // 2
            median = typical[mid] if len(typical) % 2 else math.fsum((typical[mid - 1], typical[mid])) / 2
            per_entry = median / miner_part
    # keep-alive entries: only those that still gain (own < per_entry) take a slot, pure entrants first
    alive: dict[str, list[int]] = {}
    for track_id in cand:
        elig = [u for u in cand[track_id] if own.get(u, 0.0) < per_entry]
        elig = [u for u in elig if u not in own] + [u for u in elig if u in own]
        elig = cut(track_id, elig)
        if elig:
            alive[track_id] = elig
    taken = {u for ids in alive.values() for u in ids} | set(paid)
    for ranked_t in ranks_of.values():
        taken.update(ranked_t)
    # provisional pay keeps an UNSERVED waiting entry from being pruned: a hotkey that serves is not at that risk, so only
    # pure entrants get it (critic round 1 F9) and only they make a track hold a slot (round 2 NEW-2)
    prov_all = {t: [u for u in ids if u not in taken and u not in own] for t, ids in prov_cand.items()}
    # which tracks hold a slot: only tracks with a payable entry (or ranks), so an entry that would be paid nothing never
    # pushes another track's entrants out of the profile's slots
    slots = list(zip(sorted(set(tracks) | {t for t, v in alive.items() if v} | {t for t, v in prov_all.items() if v}),
                     prof.tracks))  # (track id, slot share), as paid
    slot_ids = {t for t, _ in slots}
    prov = {t: ids for t, ids in prov_all.items() if t in slot_ids and ids}
    ewait: list[int] = []
    if ewait_cand:
        # engine keep-alive (miner report 21903): pure entrants only, never a uid paid elsewhere
        taken = {u for t in slot_ids for u in alive.get(t, ())} | {u for ids in prov.values() for u in ids} | set(paid)
        for ranked_t in ranks_of.values():
            taken.update(ranked_t)
        for uid in ewait_cand:
            if uid not in taken and uid not in own and len(ewait) < ENGINE_KA_MAX_ENTRIES + ENGINE_KA_PROTECTED_MAX:
                ewait.append(uid)
    if ewait:
        each_ew = min(ENGINE_KA_FRACTION * per_entry, ENGINE_KA_MAX_SHARE * prof.engine / len(ewait))
        for uid in ewait:
            paid[uid] = each_ew
        # taken from the engine share, i.e. from the champion's part of it (split by payment over co-champions)
        if not engine_cochampions:
            if engine_champion in paid:  # exactly the rule before the co-champions
                paid[engine_champion] = max(0.0, paid[engine_champion] - each_ew * len(ewait))
        else:
            held = math.fsum(paid[u] for u in engine_payees)
            if held > 0:
                take = each_ew * len(ewait)
                for u in engine_payees:
                    paid[u] = max(0.0, paid[u] - take * paid[u] / held)
        log.info("engine keep-alive for uids %s, each %.5f of the miner part", ewait, each_ew)
    if legacy_cand:
        # legacy keep-alive (LEGACY_KA_*): pure entrants only, never a uid paid elsewhere; from the inference part
        taken = {u for t in slot_ids for u in alive.get(t, ())} | {u for ids in prov.values() for u in ids} | set(paid)
        for ranked_t in ranks_of.values():
            taken.update(ranked_t)
        legacy = [u for u in legacy_cand if u not in taken and u not in own][:LEGACY_KA_MAX_ENTRIES]
        if legacy:
            each_lk = min(LEGACY_KA_FRACTION * per_entry, LEGACY_KA_MAX_SHARE / len(legacy))
            for uid in legacy:
                paid[uid] = each_lk
            log.info("legacy keep-alive (ended DF/RA tracks) for uids %s, each %.5f of the miner part", legacy, each_lk)
    for track_id, slot_share in slots:  # a track without a slot pays nothing
        rank_share = slot_share
        if track_id in alive:
            n = len(alive[track_id])
            # generalist: the keep-alive cap is never above KEEPALIVE_MAX_SHARE even if keepalive_share is raised
            cap = min(ka_cap, KEEPALIVE_MAX_SHARE) if track_id == "generalist" else ka_cap
            each = min(per_entry, cap * slot_share / n)
            for uid in alive[track_id]:
                paid[uid] = paid.get(uid, 0.0) + max(0.0, each - own.get(uid, 0.0))
            # the top-ups actually paid; a serving entrant gives up only what it does not already earn
            rank_share = slot_share - (each * n - math.fsum(min(each, own.get(u, 0.0)) for u in alive[track_id]))
        ids = prov.get(track_id, [])
        if ids:
            n_prov = len(ids)
            prov_cap = PROVISIONAL_KA_MAX_SHARE * slot_share
            if track_id == "generalist":
                prov_cap = min(prov_cap, max(0.0, min(ka_cap, KEEPALIVE_MAX_SHARE) * slot_share - (slot_share - rank_share)))
            each_prov = min(PROVISIONAL_KA_FRACTION * per_entry, prov_cap / n_prov)
            for uid in ids:
                paid[uid] = paid.get(uid, 0.0) + each_prov
            rank_share -= each_prov * n_prov
            log.info("track %s: provisional keep-alive for uids %s, each %.5f of the miner part",
                     track_id, ids, each_prov)
        ranked_pay = ranks_of.get(track_id) or rank_shares(tracks.get(track_id, ()), floors.get(track_id))
        if keep_share and award_ok is not None and now is not None:
            tr = tracks.get(track_id)
            if getattr(tr, "cochamps", None) and not getattr(tr, "cochamps_dynamic", False):
                pass    # the opening co-champion set holds rank 1: no keep-share rule (as in rc11.post2)
            elif getattr(tr, "cochamps", None) or getattr(tr, "formset", None) or getattr(tr, "formsets", None):
                # a shared group at rank 1 or a beaten group as a former place (rc11.post3, slot rules section 8: pro rata)
                ranked_pay = keep_share_set_ranks(ranked_pay, tr, floors.get(track_id), now, lambda u, t=track_id: award_ok(t, u))
            else:
                ranked_pay = keep_share_ranks(ranked_pay, getattr(tr, "crowned", None), now, lambda u, t=track_id: award_ok(t, u))
        for uid, share in ranked_pay.items():
            if award_ok is not None and not award_ok(track_id, uid):
                continue
            paid[uid] = paid.get(uid, 0.0) + rank_share * share
    paid.pop(owner_uid, None)  # never to the owner UID: its share stays with the inference part
    claimed = math.fsum(paid.values())
    if claimed <= 0:
        return dict(sorted(base.items()))

    keep = 1.0 - claimed
    out: dict[int, float] = {}
    for uid, w in base.items():
        out[uid] = burn + (w - burn) * keep if uid == owner_uid else w * keep
    for uid, f in paid.items():
        out[uid] = out.get(uid, 0.0) + f * miner_part
    norm = math.fsum(out.values())
    return {uid: w / norm for uid, w in sorted(out.items())}
