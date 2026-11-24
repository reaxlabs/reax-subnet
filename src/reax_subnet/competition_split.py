"""Competition share layer (docs/COMPETITIONS.md): how the miner weights are split between serving and competitions.

Pure and deterministic. Release constants only; nothing here is operator-configurable on finney.

    inference (serving)  +  engine champion  +  use-case (specialist) tracks

Profiles (``SPLIT_PROFILE`` is the active one):

* ``50/50`` (ACTIVE since the owner approved it on 15 Oct 2026): inference 0.50, engine 0.20, TWO active tracks
  0.15 + 0.15.
* ``80/20`` (inactive, kept ready): inference 0.80, engine 0.10, ONE active use-case track 0.10.

Until NEED_START_UTC (Fri 30 Oct 2026 14:30Z): a competition share that is unclaimed or inactive (no active track, no
champion, a struck champion, the decayed part of the champion, a missing track rank) goes back to the inference part; it
is never burned and never paid to the owner UID. From NEED_START_UTC on ("emission follows need", docs/EMISSION-NEED.md):
inference gets exactly the current inference share, and everything not paid goes to the owner UID (burned).
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
# Engine competition v2.1 payout bridge (docs/ENGINE-COMPETITION-V2.md): None = off. When set, a validator running the
# composite track ``dm-all`` keeps paying the v1 ``s1-fast`` champion's engine share, read from the frozen v1 state file
# (engvalidator.EngineCompetition ``v1_bridge``), until dm-all is crowned for the first time (one-way, survives vacating and restarts) or this time, whichever is first.
# The bridge is on only if BOTH constants are set; the state file must hash to ENGINE_V1_BRIDGE_STATE_SHA256 on every read.
# Calendar (CoS promise check, board #22 #26409): the Tue 27 Oct 14:30Z crown slot is the LAST v1 (``s1-fast``) crowning. Engine v2
# (``dm-all``) is in force right after that slot: before ENGINE_V2_START_UTC no composite cycle runs and no composite entry is paid;
# from it on measurements may run, but a composite is only ever crowned at a slot strictly AFTER it (the first one is Fri 30 Oct
# 14:30Z), so the v1 slot and the v2 start at the same instant cannot both crown.
ENGINE_V2_START_UTC: Optional[datetime] = datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc)
# The v1 title holder at the switch keeps the champion share until the first v2 crowning, at most until this slot. Then it moves to
# the 17 % former-champion place as if beaten (engvalidator: carried places), the v1 former champions shift down with it.
ENGINE_V1_BRIDGE_END_UTC: Optional[datetime] = datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc)
# sha256 (hex) of the frozen v1 engine-competition.json bytes, pinned after the Tue 27 Oct 14:30Z slot was finalised (frozen file
# engine-competition.v1-final-1faccee3.json, copy in tests/data). While it is None the bridge is off, and ``release_bridge_problem``
# refuses to run v2 on finney.
ENGINE_V1_BRIDGE_STATE_SHA256: Optional[str] = "1faccee327f80b48427778d0561195c9d2c0110ffc15f897ff7ef5045969c7a6"


def release_bridge_problem() -> Optional[str]:
    """Why a finney process must not run the composite (v2) track yet, or None. Fail closed: with a bridge end time but no pinned
    frozen-state digest the v1 title holder would silently lose its champion share and its former-champion place."""
    if ENGINE_V1_BRIDGE_END_UTC is not None and not ENGINE_V1_BRIDGE_STATE_SHA256:
        return ("ENGINE_V1_BRIDGE_END_UTC is set but ENGINE_V1_BRIDGE_STATE_SHA256 is not pinned: freeze the v1 state after the "
                "27 Oct 14:30Z slot, pin its sha256, then start v2")
    return None

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


# Keep the initial release-constant prelude executable without package context:
# set_generalist_start.py and its validation tests execute that prelude in isolation.
from . import gr2_podium as _gr2_podium
from . import ka_feed as _ka_feed
from .ka_feed import GR2_KA_FEED_START_UTC

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

# Waiting-state keep-alive HOTFIX (Florian 26 Oct 2026 14:5xZ, DECISIONS keepalive-no-dereg-while-waiting: "Miner UIDs
# must not be deregistered just because they are in a waiting state"; job keepalive-hotfix-20261026, SPEC.md). Every wait
# WE cause now keeps the keep-alive it already had until the wait ends, bounded; pay levels, floors, per-coldkey limits,
# KEEPALIVE_MAX_ENTRIES and the share caps are unchanged. False restores the rc11.post4 windows exactly.
WAITING_KA_HOTFIX = True
# Big-undecided fix (lead decision 26 Oct 2026, 21:3xZ): the open-class base floor for entries above KEEPALIVE_BIG_PARAMS is a
# test on a SCORE. An entry still undecided (waiting for its first draw, or owing one more draw) keeps the provisional keep-alive
# (same caps, windows, abuse clause); score-based keep-alive (pending, champion, top-eight) keeps the floor. False = bit-identical to 233dbd2.
KA_BIG_UNDECIDED = True
# Gauntlet round 2 (lead decision 26 Oct 2026, property-test findings F1/F2/F3/F5; needs WAITING_KA_HOTFIX): F1 compute_keepalive no
# longer cuts pending / confirmed / champion-awaiting entries to KEEPALIVE_MAX_ENTRIES (the share cap splits evenly downstream);
# F2 an on-chain commitment the evaluator has not recorded yet gets UNSEEN_GRACE_HOURS from its OWN commitment block whatever
# earlier windows the hotkey used; F3 unseen commitments do not count against PROVISIONAL_KA_UNVERIFIED_MAX; F5 the same-tensors
# "one window per model" minimum applies to the plain window only, never to the slot extension or the hard cap.
# False (with KA_BIG_UNDECIDED False) = bit-identical to 233dbd2.
KA_GAUNTLET_R2 = True
PROVISIONAL_KA_UNSEEN_GRACE_HOURS = 6.0
# Most exposed first (lead decision 26 Oct 2026, needs WAITING_KA_HOTFIX): when keep-alives per coldkey are chosen (generalist
# provisional, engine waiting), the uid that is most exposed to pruning gets it: not immune before immune, then the earliest immunity
# end, then the old order (an unknown registration block counts as exposed, after the known ones). Uids 165 (immune) / 223 (exposed)
# and champion 62 share a coldkey: the older commitment must not push the exposed one to zero.
# False = the previous order (commitment block / first sighting).
KA_EXPOSED_FIRST = True
# Match the published two paid waiting entries per coldkey and track (MAX_OPEN_PER_COLDKEY_TRACK).
# Needs WAITING_KA_HOTFIX; False restores the previous one-entry pick. Share and unverified caps stay unchanged.
KA_PROV_TWO_PER_COLDKEY = True
# Round-2 intake wait (lead decision 22 Nov 2026, board #76664 / #76679; needs WAITING_KA_HOTFIX): the published promise is
# keep-alive for an entry that waits for OUR Round-2 intake, at least GR2_INTAKE_WAIT_KA_DAYS from its own commitment. The
# legacy path (spec.evaluator._committed_waiting, used while the intake feed is not armed) cannot see the intake and the
# Generalist evaluator is paused, so such a commitment was UNVERIFIED and lost its keep-alive after 24 h. With this switch a
# Generalist commitment made at or after GR2_INTAKE_WAIT_KA_START_UTC that is still undecided gets [commitment, commitment +
# GR2_INTAKE_WAIT_KA_DAYS) instead, provided its repo@revision passes a cheap anonymous public check on the hub (a failed check
# falls back to the old windows; a hub error keeps the window). Recommits do not chain (same repo@revision: never; a new one:
# only if the hotkey's previous window started KA_REWINDOW_HOURS earlier). Abuse clause, two per coldkey, share cap, unverified
# cap (a PASSED entry is not counted, an unchecked one is) and the ordering rules are unchanged. Never used while the feed mode
# (ka_feed.active) is on. False = exactly post8.16.
GR2_INTAKE_WAIT_KA = True
GR2_INTAKE_WAIT_KA_START_UTC = datetime(2026, 11, 20, 16, 40, tzinfo=timezone.utc)
GR2_INTAKE_WAIT_KA_DAYS = 10
# public check (ka_intake_wait.py): one anonymous hub request per (repo, revision), at most GR2_INTAKE_WAIT_LOOKUP_TIMEOUT_S
# each, the verdict cached GR2_INTAKE_WAIT_CACHE_HOURS (a hub error only GR2_INTAKE_WAIT_RETRY_MINUTES); per weight-set round at
# most GR2_INTAKE_WAIT_MAX_LOOKUPS new lookups and GR2_INTAKE_WAIT_ROUND_BUDGET_S seconds waiting in total (what is not
# answered in time counts as a hub error for this round and is answered from the cache in a later one; a round never waits for
# a lookup launched by an earlier round). Budget 1 s (v2, critic S2): fail-open keeps the window anyway, so an unanswered lookup
# costs at most one round of being counted against PROVISIONAL_KA_UNVERIFIED_MAX, while the wait blocks the serving event loop.
GR2_INTAKE_WAIT_CACHE_HOURS = 6.0
GR2_INTAKE_WAIT_RETRY_MINUTES = 10.0
GR2_INTAKE_WAIT_LOOKUP_TIMEOUT_S = 4.0
GR2_INTAKE_WAIT_ROUND_BUDGET_S = 1.0
GR2_INTAKE_WAIT_MAX_LOOKUPS = 24


def exposure_key(registered_block, block: int) -> tuple[int, int]:
    """Sort key, smaller = more exposed: (1 if still immune else 0, end of immunity); unknown registration = (0, huge)."""
    if isinstance(registered_block, int) and not isinstance(registered_block, bool) and registered_block > 0:
        end = registered_block + CHAIN_IMMUNITY_BLOCKS
        return (1 if block < end else 0, end)
    return (0, 1 << 62)
# H1: a pending entry (first pass taken) keeps its keep-alive until the slot that decides it has run (slot + SLOT_CLOCK_S),
# not just KEEPALIVE_HOURS after its first pass; hard cap after the first pass: the longest slot gap twice (a voided slot
# re-runs at the next one) plus the confirmation lead.
PENDING_KA_MAX_HOURS = 216.0
# H5: a new champion waiting for the owner's award approval (source + rebuild check + signer card) keeps it this long
# after its crowning; an approved champion is paid by its rank and holds no keep-alive slot anyway.
CHAMPION_APPROVAL_KA_HOURS = 72.0
# G1/H3: every live commitment of an undecided entry gets the provisional keep-alive from the first weight set after its
# on-chain commitment (spec.evaluator._committed_waiting), also before the evaluator saw it and when our first fetch found
# no repo. Verified entries and waits on OUR side (queue, boxes, fetch, base download) keep it until the first draw decides
# them, at most this long (a safety bound; the reliability invariant alerts long before); unverified ones (repo missing or
# private, miner-side base problem; we cannot tell a private repo from a hub/token failure on our side) get
# PROVISIONAL_KA_UNVERIFIED_HOURS from the commitment, one such window per hotkey per KA_REWINDOW_HOURS.
# A waiting hotkey with a small serving weight is topped up to the keep-alive level instead of being left out.
PROVISIONAL_KA_OUR_WAIT_HOURS = 96.0
PROVISIONAL_KA_UNVERIFIED_HOURS = 24.0
# at most this many UNVERIFIED provisional entries per track (oldest commitments first), so dummy commitments cannot
# dilute the others without bound (critic F3; 16 per critic2 N4: the real state had 9 at once); verified / our-wait entries are not limited by count (share cap only).
PROVISIONAL_KA_UNVERIFIED_MAX = 16
# Abuse clause for the unverified window (proposal 26 Oct 2026 17:4xZ, ships only after Florian's yes): a commitment that is
# not a loadable model (FAKE: the committed revision does not exist (HTTP 404), no safetensors listed, weights identical to
# the declared base, hub revision different from the committed one) gets no unverified window, its coldkey gets no
# unverified window for KA_ABUSE_WINDOW_DAYS, and a coldkey with KA_ABUSE_EXCLUDE_AFTER fakes in that time gets no provisional
# keep-alive at all. A private repo (HTTP 401) is not a fake: it keeps its 24 h. False = the clause is off.
KA_ABUSE_CLAUSE = True
KA_ABUSE_WINDOW_DAYS = 30
KA_ABUSE_EXCLUDE_AFTER = 2
# H4: a hotkey's provisional / engine waiting window used to start once and never again, so a former champion or an
# earlier entrant that commits a new entry (track switch, new model) got none. A new entry first seen at least this long
# after the start of the hotkey's previous window opens a new window from its own first sighting (at most one per this
# many hours per hotkey; per-coldkey limits and share caps unchanged).
KA_REWINDOW_HOURS = 72.0
# H6 (engine): an image waiting for its first measurement: this long from its own first sighting (was ENGINE_KA_MAX_HOURS
# per hotkey); the pending challenger (its first session beat the bar; confirming session, Tue/Fri slot or finalize still
# to come, all on our side) keeps it while it is the pending challenger, at most ENGINE_SLOT_KA_MAX_HOURS after its first
# session (critic F6).
ENGINE_KA_WAIT_HOURS = 24.0
ENGINE_SLOT_KA_MAX_HOURS = 120.0
# H6b (engine, 26 Oct 2026, uid 165): an image that has no measurement yet waits for OUR measurement, so its window lasts until
# it is measured, at most ENGINE_SLOT_KA_MAX_HOURS (not ENGINE_KA_WAIT_HOURS), and the window of an immune hotkey opens
# ENGINE_KA_LEAD_BLOCKS before its immunity ends instead of PROVISIONAL_KA_LEAD_BLOCKS: SN92 has commit-reveal, a set counts only
# at the second epoch boundary after its block (up to 2 x 360 blocks), so 600 left the first paying set landing after the
# immunity (uid 165 paid 0 at the end of its immunity). Per-coldkey limit and ENGINE_KA_MAX_SHARE cap still apply. False restores
# the post7.1 windows exactly.
ENGINE_KA_UNTIL_MEASURED = True
ENGINE_KA_LEAD_BLOCKS = 1500

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
# with more than ``KEEPALIVE_BIG_PARAMS`` parameters qualifies for score-based keep-alive only if
# this table has its declared base's tensor hash (``Submission.base_id``) for the track and the score that qualified it (the
# non-proc score, the total for a legacy entry) is above that floor + ``KEEPALIVE_BIG_MARGIN``. A larger model must beat
# its own base, not only a rules-only model. Maps base tensor hash -> {track id: floor}. Ships EMPTY (filled after
# calibration): until then no scored entry above 10 B parameters gets a keep-alive. With KA_BIG_UNDECIDED, undecided
# entries retain provisional keep-alive without a floor or score. A larger scored entry without a recorded base hash gets
# none; an entry without a recorded parameter count (fetched by an earlier release, under the 9 B cap, or not fetched yet) counts as small. An entry with at most ``KEEPALIVE_BIG_PARAMS`` parameters is not affected.
KEEPALIVE_BIG_PARAMS = 10_000_000_000
KEEPALIVE_BIG_MARGIN = 1.5
# REAX 1 rows, on the r8 footing (shipped pplx floor 58.01 + the paired non-proc offset vs pplx on the same 6 draws of the final
# pool, 21 Oct): decisionmodels/reax-1@ca2a0cc5 (4031ce5e..., the official base) 60.43 (+2.42); the earlier public revision
# @ea9d5d8b (572884f6..., readout not rescaled) 55.34 (-2.67).
KEEPALIVE_BASE_FLOOR: dict[str, dict[str, float]] = {
    '0e8d23478b5b6ccefff3136e5e53ecfb7d053451ec405d2b0fe31ce3a49a713e': {'generalist': 39.23},
    '1647929265009944f8157465e27417ec84014893953f3b3ed0874e373bb276fc': {'generalist': 62.23},
    '19d89e343b9109fcb478d685cb08f15914c4aded296d02c68c2c3fa9f0fdb4df': {'generalist': 39.93},
    '4031ce5ec8b6f10e388f09df862810315e4132f429f8cdf308f13566bd159eef': {'generalist': 60.43},
    '564471ef18768dd040eb3ce7effaf0561b65b3b2a50cb10980974bae3ae713dd': {'generalist': 55.01},
    '572884f6e3788b5debd2084e2789a009622e50272fe6c226e1d93dcb24ec6d41': {'generalist': 55.34},
    '81a3eb978ff3c5b90c273a100f36499a86dfec698652c2d181f43e11e89fe08d': {'generalist': 58.01},
    'dac9897fe790514fec2d32f666b5627e098205b051b48fe6630a3699a5138e10': {'generalist': 49.52},
    # Gemma-4-31B leader bases, on the gemma-4-31B-it footing (shipped 62.23 + the paired non-proc offset vs gemma-4-31B-it on the
    # same 6 draws of the final pool df68cdc2, one session on floor-calbox-01, 28 Oct 2026; workers/floor-calib/results/tables.md).
    # Sieve-31B-v3@eb99cd7e: no row (its transformers-5.17 config layout does not serve on vLLM 0.30).
    "8e41636d16fa5874685a38696ee8a5346fe0c03998dc13343d077617ec6034c1": {"generalist": 63.04},   # PelaAI/KnowLine-31B-Gen1@8f052a81 (+0.81 +-2.07)
    "c910628ca23a605b6b2de3b2dabf104140326adf882aa7801f627dd8b1414b7e": {"generalist": 48.15},   # turion-ai/turion-jev-31b@866c0fc6 = @9edd5783 tensors (-14.08 +-1.87)
    "7c8a0695c85afa67f52f834216337990acfa1f42c7db9cc4583c90bb809431f8": {"generalist": 46.32},   # tattrongvu/Jeffan-31b-1710@cebfa86c (-15.91 +-2.65)
}

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
# rc11.post4.5.2 / post8.4 (Florian GO #31857, public in #92, 28 Oct 2026): ONE extra generalist slot Fri 30 Oct 2026 16:30Z
# for every entry confirmed after Thursday's slot; commit cutoff Fri 10:30Z (GENERALIST_COMMIT_CUTOFF_S, the usual 6 h). The
# Mon/Thu generator is unchanged; the list stays sorted (``_cutoff_slot`` takes the first match). Only the generalist track
# is active then. The 2 h confirmation for this slot only: ``spec.machine.confirm_delay_s``.
EXTRA_CROWN_SLOTS_UTC: tuple[datetime, ...] = (datetime(2026, 10, 30, 16, 30, tzinfo=timezone.utc),)
CROWN_SLOTS_UTC: tuple[datetime, ...] = tuple(sorted(set(
    _crown_slots(datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc), datetime(2027, 3, 31, 23, 59, tzinfo=timezone.utc))
    + EXTRA_CROWN_SLOTS_UTC)))


# Engine titles change on Tuesday and Friday, independently of measurement batches. Empty = legacy instant crowns.
ENGINE_CROWN_SLOTS_UTC: tuple[datetime, ...] = _crown_slots(
    datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc),
    datetime(2027, 3, 31, 23, 59, tzinfo=timezone.utc), weekdays=(1, 4))


# Shared leading group at a crown slot (1.2.0rc11.post3; CRITERIA v2 and the public slot rules, reax.co/competitions/faq/#slot-rules;
# Florian's decisions 22 Oct, card 18227 and DECISIONS tie-numbers-k). Every knob is ONE constant with the approved default, so a
# change is a one-liner (it must be announced 48 h before the first slot it applies to, slot rules section 1).
GENERALIST_COMMIT_CUTOFF_S = 6 * 3600
GENERALIST_COMMIT_CUTOFF_START_UTC = datetime(2026, 10, 29, 14, 30, tzinfo=timezone.utc)

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


# Emission follows need (Florian 26 Oct 2026, DECISIONS emission-follows-need-burn-1030, burn-announced-1026 and the standing
# pragmatism rule; spec: docs/EMISSION-NEED.md). From NEED_START_UTC on:
# (1) the inference (serving) part of the miner part is exactly the current inference share, not 1 minus whatever the
#     competitions claim;
# (2) a PAUSED competition track ("engine" or a specialist track id) pays no champion and no rank: those entries get the
#     track's keep-alive, so a pause prunes nobody;
# (3) every part of the miner part that is not paid (the rest above the inference share, a paused or unclaimed competition
#     share, the decayed champion part, an unapproved award) goes to the subnet owner UID, i.e. it is burned; it never goes
#     to another miner.
# The inference share and the pauses are our decisions (no automatic thresholds): release constants below, plus the
# operator file need.json (need.py) for a change between releases. Before NEED_START_UTC, or with NEED_RULE False, the
# weights are exactly the ones without this rule.
NEED_RULE = True
# The first weight set at or after the Fri 30 Oct 14:30Z engine crown slot.
NEED_START_UTC = datetime(2026, 10, 30, 14, 30, tzinfo=timezone.utc)
# (effective from, inference share of the miner part), increasing in time.
INFERENCE_SHARE_STEPS: tuple[tuple[datetime, float], ...] = (
    (NEED_START_UTC, 0.40),
)
# Pause key ("engine" or a specialist track id) -> ((from, until or None), ...). Empty = no track paused.
TRACK_PAUSES: dict[str, tuple[tuple[datetime, Optional[datetime]], ...]] = {}
# Sanity bounds of an inference share (a typo guard for the constants and need.json, not a promise).
INFERENCE_SHARE_SANE = (0.05, 0.80)


def _need_check() -> None:
    last: Optional[datetime] = None
    for at, share in INFERENCE_SHARE_STEPS:
        assert at.tzinfo is not None and (last is None or at > last), "INFERENCE_SHARE_STEPS must be increasing"
        assert INFERENCE_SHARE_SANE[0] <= share <= INFERENCE_SHARE_SANE[1], "inference share outside the sanity bounds"
        last = at


_need_check()


def need_active(now: Optional[datetime]) -> bool:
    return NEED_RULE and now is not None and now >= NEED_START_UTC


def inference_share(now: Optional[datetime],
                    extra: Sequence[tuple[datetime, float]] = ()) -> Optional[float]:
    """The inference share of the miner part at ``now``: the latest step at or before ``now`` of INFERENCE_SHARE_STEPS and
    ``extra`` (need.json rows; a later step wins, on a tie ``extra``); None = the rule is not active (old weights)."""
    if not need_active(now):
        return None
    share: Optional[float] = None
    best: Optional[datetime] = None
    for at, value in [*INFERENCE_SHARE_STEPS, *extra]:
        if at <= now and (best is None or at >= best):
            share, best = value, at
    return share


def track_paused(key: str, now: Optional[datetime],
                 extra: Sequence[tuple[str, datetime, Optional[datetime]]] = ()) -> bool:
    """True when ``key`` ("engine" or a specialist track id) is paused at ``now`` by TRACK_PAUSES or ``extra``
    (need.json rows ``(key, from, until)``); always False while the rule is not active."""
    if not need_active(now):
        return False
    return pause_start(key, now, extra) is not None


def pause_start(key: str, now: Optional[datetime],
                extra: Sequence[tuple[str, datetime, Optional[datetime]]] = ()) -> Optional[datetime]:
    """Start of the continuous pause containing now (overlapping/touching spans do not renew notice)."""
    if not need_active(now):
        return None
    spans = sorted([*TRACK_PAUSES.get(key, ()), *((a, b) for k, a, b in extra if k == key)], key=lambda span: span[0])
    merged: list[tuple[datetime, Optional[datetime]]] = []
    for a, b in spans:
        if merged and (merged[-1][1] is None or a <= merged[-1][1]):
            old_a, old_b = merged[-1]
            merged[-1] = (old_a, None if old_b is None or b is None else max(old_b, b))
        else:
            merged.append((a, b))
    return next((a for a, b in merged if a <= now and (b is None or now < b)), None)


# Older crowns retain their original clock; their approval delay is settled separately.
RUNOUT_APPROVAL_CLOCK_FROM_UTC = datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)


def runout_start(crowned_at: Optional[float], approved_at: Optional[float] = None) -> Optional[float]:
    """None = unstarted/unknown clock (full factor, independently of the award gate)."""
    if crowned_at is None or not math.isfinite(crowned_at):
        return None
    if crowned_at < RUNOUT_APPROVAL_CLOCK_FROM_UTC.timestamp():
        return crowned_at
    if approved_at is None or not math.isfinite(approved_at):
        return None
    return max(crowned_at, approved_at)


def runout_factor(crowned_at: Optional[float], pause_start: Optional[float], now: float,
                  approved_at: Optional[float] = None) -> float:
    """Unix seconds: full through start+3d, linear to zero at start+7d.

    For crowns from RUNOUT_APPROVAL_CLOCK_FROM_UTC, start=max(crown, own approval).
    Before approval the factor is 1; the independent award gate still controls payment.

    Notice is enforced by needctl before scheduling, never by payout weights.
    Missing/non-finite times preserve the full share; before the pause this is a no-op.
    """
    if any(v is None or not math.isfinite(v) for v in (crowned_at, pause_start, now)):
        return 1.0
    if now < pause_start:
        return 1.0
    start = runout_start(crowned_at, approved_at)
    if start is None:
        return 1.0
    day = 86400.0
    return max(0.0, min(1.0, (start + 7 * day - now) / (4 * day)))


class RankedLeaders(list):
    """``[(uid, score), ...]`` already in rank order, rank 1 first (docs/SPECIALIST-TRACKS.md, "Ranks"): the specialist
    evaluator's champion, then its former champions, most recently crowned first. ``track_payout`` pays it in this
    order and never re-sorts it by score; the scores come from different draws and are not comparable.

    ``crowned``: uid -> when that entry was crowned champion (unix seconds), for the entries that have such a time (the
    champion; a former champion keeps the time it was crowned). Used by keep-share and paused-track run-out.
    ``champions``: current title-holder uids (empty for a vacant title); None for legacy callers without this metadata.

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

    champions: Optional[set[int]]
    crowned: dict[int, float]
    cochamps: dict[int, float]
    formset: dict[int, float]
    formsets: list[dict[int, float]]
    cochamps_dynamic: bool

    def __init__(self, *args):
        super().__init__(*args)
        self.champions = None
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


def release_v2_start() -> Optional[datetime]:
    """The composite (engine v2) start on finney; None = no extra start (development constant unset)."""
    return ENGINE_V2_START_UTC


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


def engine_composite_enabled(env: Mapping[str, str]) -> bool:
    """REAX_ENGINE_COMPOSITE: strict on/off switch for engine competition v2, unset = off; an unknown value is refused
    (``0``/``false`` must never switch v2 on)."""
    raw = (env.get("REAX_ENGINE_COMPOSITE") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"REAX_ENGINE_COMPOSITE must be 1/0 (or true/false, yes/no, on/off), got {raw!r}")


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


def _need_weights(base: Mapping[int, float], paid: Mapping[int, float], owner_uid: int, burn: float,
                  target: float, legacy_paid: float, owner_allocation: Optional[Mapping[int, float]] = None) -> dict[int, float]:
    """Emission follows need: the serving part of ``base`` (everything but the owner's burn fraction, including the owner's
    fallback for a pool without an eligible miner) is scaled to ``target`` of the miner part, less the legacy keep-alive
    that is taken from it; ``paid`` (fractions of the miner part) is added; everything else goes to ``owner_uid``."""
    miner_part = 1.0 - burn
    serving = {uid: (w - burn if uid == owner_uid else w) for uid, w in base.items()}
    serving = {uid: w for uid, w in serving.items() if w > 0}
    s_total = math.fsum(serving.values())
    # never more than the room the paid competition parts leave (critic M1: a share above it would be renormalised and
    # silently shrink the competition payouts)
    s_share = max(0.0, min(target, 1.0 - math.fsum(paid.values())) - legacy_paid)
    if s_total <= 0:
        s_share = 0.0
    from .owner_alloc import take
    allocation, taken = take(owner_allocation, s_share)
    s_share = max(0.0, s_share - taken)
    out: dict[int, float] = {}
    for uid, w in serving.items():
        out[uid] = miner_part * s_share * w / s_total
    for uid, f in paid.items():
        out[uid] = out.get(uid, 0.0) + miner_part * f
    for uid, f in allocation.items():
        log.info("owner allocation applied uid %s share %.12f", uid, f)
        out[uid] = out.get(uid, 0.0) + miner_part * f
    used = math.fsum(out.values())
    # Burn allocations consume only the residual owner top-up, never its serving component.
    remainder = max(0.0, 1.0 - used)
    burn_allocation, _ = take(owner_allocation, remainder / miner_part if miner_part > 0 else 0.0,
                                       source="burn")
    if hasattr(owner_allocation, "burn_weights"):
        owner_allocation.miner_part = miner_part
        owner_allocation.burn_weights = {uid: miner_part * f for uid, f in burn_allocation.items()}
    for uid, f in burn_allocation.items():
        log.info("owner allocation applied uid %s share %.12f source burn", uid, f)
        out[uid] = out.get(uid, 0.0) + miner_part * f
    used = math.fsum(out.values())
    out[owner_uid] = out.get(owner_uid, 0.0) + max(0.0, 1.0 - used)
    norm = math.fsum(out.values())
    return {uid: w / norm for uid, w in sorted(out.items()) if w > 0}


def need_only(base: Mapping[int, float], owner_uid: Optional[int], burn_fraction: float,
              now: Optional[datetime]) -> dict[int, float]:
    """The weights when no competition can be paid (an error path or nothing to pay) under emission follows need: the
    serving part scaled to the inference share, the rest to the owner UID. Rule not active or no owner UID: ``base``."""
    target = inference_share(now) if owner_uid is not None else None
    total = math.fsum(w for w in base.values() if w > 0)
    if target is None or total <= 0:
        return dict(base)
    norm = {uid: w / total for uid, w in base.items() if w > 0}
    burn = burn_fraction if owner_uid in norm else 0.0
    return _need_weights(norm, {}, owner_uid, burn, target, 0.0)


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
                    award_approved_at: Optional[Callable[[str, int], Optional[float]]] = None,
                    engine_approved_at: Optional[Mapping[int, float]] = None,
                    keepalive: Optional[Mapping[str, Sequence[tuple[int, float]]]] = None,
                    keepalive_share: Optional[float] = None,
                    provisional: Optional[Mapping[str, Sequence[int]]] = None,
                    engine_waiting: Optional[Sequence[int]] = None,
                    keep_share: bool = False, now: Optional[float] = None,
                    engine_cochampions: Optional[Mapping[int, float]] = None,
                    legacy_keepalive: Optional[Sequence[int]] = None,
                    keepalive_protected: Optional[Mapping[str, Sequence[int]]] = None,
                    paused: Optional[Sequence[str]] = None,
                    inference: Optional[float] = None,
                    pause_starts: Optional[Mapping[str, float]] = None,
                    engine_runout: Optional[Mapping[int, tuple[float, Optional[float]]]] = None,
                    owner_allocation: Optional[Mapping[int, float]] = None,
                    generalist_feed_uids: Optional[Sequence[int]] = None,
                    generalist_lastgood: Optional[Mapping[str, Sequence[int]]] = None,
                    generalist_ka_used: Optional[dict[str, list[int]]] = None,
                    log_generalist_unit: bool = True,
                    generalist_podium: Optional[Any] = None,
                    generalist_podium_used: Optional[dict[str, Any]] = None) -> dict[int, float]:
    """Mix the competitions into the serving weights.

    Emission follows need (``need_active(now)``, docs/EMISSION-NEED.md): the serving uids get exactly
    ``inference_share(now)`` of the miner part, every part that is not paid goes to ``owner_uid`` (burned), and a track in
    ``paused`` ("engine" or a specialist track id) ends former ranks and applies ``runout_factor`` to rank 1.
    Ranked entries retain the existing keep-alive floor. ``pause_starts`` supplies each pause's unix start (otherwise
    release constants are consulted; no start means ignore the pause). ``engine_runout`` contains ONLY title holders:
    uid -> (current engine pot x frozen rank-1 fraction, crowning unix time), already eligibility/approval gated by the caller.
    Missing engine metadata ignores the engine pause, preserving its normal weights. ``inference``: the share to use
    (need.py, with need.json); None = ``inference_share(now)``. Without an owner UID nothing can be burned and the
    weights are the ones without the rule. ``now`` None or before NEED_START_UTC: exactly the old weights.

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

    ``generalist_feed_uids``: authoritative, chain-checked round-2 entry set in commitment order, former champions
    first. Used only with REAX_KA_FEED_PATH after GR2_KA_FEED_START_UTC. It replaces the generalist score/provisional
    lists, has at most 64 entries and tops up each to the same level within the existing 25 percent track cap.
    None in active feed mode is an empty set; it never restores the legacy score path.

    ``generalist_lastgood``: validator-owned KA/provisional slots, only on unusable spec-state and outside feed mode.
    The caller rechecks a <=26 h lease, chain identity and award gate. No rank share is restored. Existing median,
    top-up and share caps still apply. ``generalist_ka_used`` optionally records the selected slots for persistence.

    ``log_generalist_unit`` emits the observational unit once per computation.
    The caller disables it on owner-allocation variants, whose KA inputs are identical.
    ``generalist_podium`` (``gr2_podium.PodiumPlan``; None = the ordinary rank path, exactly the weights without it): the
    Round 2 Generalist podium replaces the 75/17/8 rank shares of the generalist track. Keep-alive and provisional pay are
    settled first; each payable slot then gets ``(pool - keep-alive) x window share`` (``gr2_podium.payout_after_keepalive``).
    The plan's slots must also pass ``award_ok`` (and never the owner UID). An empty, held, private or unapproved slot burns
    its own share: it is neither promoted nor redistributed. ``generalist_podium_used`` (an empty dict) is filled from
    this very computation with ``window``, ``pool``, ``keepalive``, ``eligible`` (slots actually paid), ``podium`` (uid ->
    weight) and ``burn``, all in weight units of the returned vector before the miner-slot factor (the miner part is
    already applied); ``paused`` instead when the generalist track is paused (the ordinary paused path then runs).

    ``legacy_keepalive``: uids of the ended DF/RA tracks' leaders (``legacy_ka_uids``, ``LEGACY_KA_*``). Paid like the
    engine keep-alive (pure entrants only: no serving weight, not the owner, nothing else paid), at most
    ``LEGACY_KA_MAX_ENTRIES`` in the given order, each ``min(LEGACY_KA_FRACTION x per_entry, LEGACY_KA_MAX_SHARE / n)``
    of the miner part, taken from the inference part only (``keep``). None or empty = exactly the weights without it.

    The competition fractions apply to the miner part (the owner's burn stays as it is, as in 1.1). Everything not
    claimed stays with ``serving``. The result is normalised to sum 1; an empty ``serving`` returns ``{}``.
    """
    # Feed authority is dated AND explicitly configured. The legacy path below
    # remains bit-identical until both conditions hold.
    feed_mode = _ka_feed.active(time.time() if now is None else now)
    feed_uids = ([u for u in dict.fromkeys(generalist_feed_uids or ()) if u != owner_uid][:_ka_feed.MAX_ENTRIES]
                 if feed_mode else [])
    if feed_mode:
        provisional = {t: v for t, v in (provisional or {}).items() if t != "generalist"}
    prof = profile_for(profile, datetime.fromtimestamp(now, timezone.utc) if now is not None else None)
    if floors is TRACK_ENTRY_FLOOR and now is not None and TRACK_ENTRY_FLOOR.get("generalist") is not None:
        from .spec import sampling as _sampling    # eval v5: the generalist scores are S_v5.1 (DI scale); the v4-scale release floor is off
        if _sampling.v5_active(now):
            floors = {**TRACK_ENTRY_FLOOR, "generalist": None}
    base_total = math.fsum(w for w in serving.values() if w > 0)
    if base_total <= 0:
        if log_generalist_unit:
            log.info("generalist keep-alive unit: none")
        return {}
    base = {uid: w / base_total for uid, w in serving.items() if w > 0}
    burn = burn_fraction if owner_uid is not None and owner_uid in base else 0.0
    miner_part = 1.0 - burn
    need_at = datetime.fromtimestamp(now, timezone.utc) if now is not None else None
    target = inference_share(need_at) if owner_uid is not None else None
    if target is not None and inference is not None:
        target = inference
    if owner_uid is None and need_active(need_at):
        log.warning("emission follows need: no owner UID on the subnet, nothing can be burned; weights without the rule")
    halted = set(paused or ()) if target is not None else set()
    starts = dict(pause_starts or {})
    for key in list(halted):
        if key not in starts:
            at = pause_start(key, need_at)
            if at is not None:
                starts[key] = at.timestamp()
        if key not in starts or not math.isfinite(starts[key]) or starts[key] > now:
            halted.remove(key)  # missing start must not introduce an unexpected burn
    if "engine" in halted and engine_runout is None:
        halted.remove("engine")

    paid: dict[int, float] = {}  # fractions of the miner part
    halted_engine: list[int] = []  # engine payees of a paused engine track: keep-alive instead of their share
    ordinary_engine_pay = {}
    ordinary_engine_waiting = list(engine_waiting or ()) if "engine" in halted else []
    if "engine" in halted:
        if engine_champion is not None and engine_share_frac > 0:
            ordinary_engine_pay[engine_champion] = prof.engine * min(1.0, engine_share_frac)
        for uid, part in sorted((engine_cochampions or {}).items()):
            if part > 0 and uid != owner_uid:
                ordinary_engine_pay[uid] = ordinary_engine_pay.get(uid, 0.0) + prof.engine * min(1.0, part)
    if "engine" in halted:
        halted_engine = [u for u in dict.fromkeys(([engine_champion] if engine_champion is not None else [])
                                                  + sorted(engine_cochampions or {})) if u != owner_uid]
        engine_champion, engine_cochampions = None, None
        engine_waiting = halted_engine + [u for u in (engine_waiting or ()) if u not in halted_engine]
    if engine_champion is not None and engine_share_frac > 0:
        paid[engine_champion] = prof.engine * min(1.0, engine_share_frac)
    for uid, part in sorted((engine_cochampions or {}).items()):
        if part > 0 and uid != owner_uid:
            paid[uid] = paid.get(uid, 0.0) + prof.engine * min(1.0, part)
    engine_payees = [u for u in sorted(paid)]  # the champion and/or the co-champions
    ka_cap = KEEPALIVE_MAX_SHARE if keepalive_share is None else keepalive_share
    # a uid's own serving weight in the unit of per_entry (docs/DESIGN-MULTIROLE.md section 5): 0 for a pure entrant
    own = {uid: w / miner_part for uid, w in base.items()}
    # NEED_OWN_START
    if target is not None:
        # Credit only serving pay that survives emission-follows-need. Reserve the
        # full competition pots (and any legacy cap), so later awards cannot make
        # this credit exceed the final serving payment. At the release's 0.40
        # inference share, without legacy, this is the exact post-rescale credit.
        reserve = prof.engine + math.fsum(prof.tracks)
        legacy_room = LEGACY_KA_MAX_SHARE if legacy_keepalive and ka_cap > 0 else 0.0
        credit = max(0.0, min(target, 1.0 - reserve) - legacy_room)
        projected = _need_weights(base, {}, owner_uid, burn, credit, 0.0)
        own = {uid: projected.get(uid, 0.0) / miner_part for uid in base if uid != owner_uid}
    # NEED_OWN_END
    cand: dict[str, list[int]] = {}  # keep-alive candidates per track, before the eligibility check and the cut
    if ka_cap > 0:
        for track_id, entries in (keepalive or {}).items():
            if feed_mode and track_id == "generalist":
                continue  # the sidecar replaces score-based generalist eligibility
            uids = keepalive_uids(entries, floors.get(track_id), owner_uid)
            if award_ok is not None and uids:
                # an approved entry is paid by its rank: it does not also hold a keep-alive slot
                ranked = rank_shares(tracks.get(track_id, ()), floors.get(track_id))
                uids = [uid for uid in uids if not (uid in ranked and award_ok(track_id, uid))]
            if uids:
                cand[track_id] = uids
    if feed_mode and ka_cap > 0 and feed_uids:
        ranked = rank_shares(tracks.get("generalist", ()), floors.get("generalist"))
        cand["generalist"] = [uid for uid in feed_uids if not (
            "generalist" not in halted and award_ok is not None and uid in ranked and award_ok("generalist", uid))]
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
    if generalist_lastgood is not None and not feed_mode:
        # Already chain/award checked by the validator. Reuse slots, never stale rank shares.
        taken_elsewhere = set(paid) | {u for t, r in ranks_of.items() if t != "generalist" for u in r}
        taken_elsewhere.update(u for t, us in cand.items() if t != "generalist" for u in us)
        taken_elsewhere.update(u for t, us in prov_cand.items() if t != "generalist" for u in us)
        ka_ids = [u for u in dict.fromkeys(generalist_lastgood.get("keepalive", ()))
                  if u != owner_uid and u not in taken_elsewhere]
        pv_ids = [u for u in dict.fromkeys(generalist_lastgood.get("provisional", ()))
                  if u != owner_uid and u not in taken_elsewhere and u not in ka_ids]
        if ka_cap > 0:
            if ka_ids:
                cand["generalist"] = list(dict.fromkeys(cand.get("generalist", []) + ka_ids))
            if pv_ids and PROVISIONAL_KA:
                prov_cand["generalist"] = list(dict.fromkeys(prov_cand.get("generalist", []) + pv_ids))
    ordinary_cand = dict(cand) if halted else {}
    ordinary_protected = keepalive_protected
    if ka_cap > 0:
        for track_id in sorted(halted & set(ranks_of)):
            if feed_mode and track_id == "generalist":
                continue  # a paused track must not add non-feed entrants
            # paused track: its ranked entries (champion, co-champions, former places) hold keep-alive slots first, so the
            # pause prunes nobody; they are paid no rank share
            # (critic H1) only entries the owner-approval gate allows: a rejected entry gets no keep-alive from the pause
            def _ok(u: int, t: str = track_id) -> bool:
                return u != owner_uid and (award_ok is None or award_ok(t, u))
            top = [u for u in ranks_of[track_id] if _ok(u)]
            tr = tracks[track_id]
            groups = [getattr(tr, "cochamps", None) or {}, getattr(tr, "formset", None) or {},
                      *(getattr(tr, "formsets", None) or ())]
            for u in sorted(set().union(*groups)):
                if _ok(u) and u not in top:
                    top.append(u)
            if top:
                cand[track_id] = top + [u for u in cand.get(track_id, []) if u not in top]
                keepalive_protected = {**(keepalive_protected or {}),
                                       track_id: top + [u for u in (keepalive_protected or {}).get(track_id, ())
                                                        if u not in top]}

    def cut(track_id: str, uids: list[int], protected=None) -> list[int]:
        # rc10.post5 (Florian via CoS #23485): protected uids (open opening-cohort entries of the vacant generalist track,
        # spec.evaluator.keepalive_protected) keep their slot first, beyond KEEPALIVE_MAX_ENTRIES if needed; the others
        # fill what is left of the KEEPALIVE_MAX_ENTRIES slots in the given order
        prot = set(((keepalive_protected if protected is None else protected) or {}).get(track_id, ()))
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
        if feed_mode and track_id == "generalist":
            # critic2 Q7: as on the legacy path, only uids that still gain (own < per_entry) take a slot; feed order kept
            elig = [u for u in cand[track_id] if own.get(u, 0.0) < per_entry]
            if elig:
                alive[track_id] = elig
            continue
        elig = [u for u in cand[track_id] if own.get(u, 0.0) < per_entry]
        elig = [u for u in elig if u not in own] + [u for u in elig if u in own]
        if WAITING_KA_HOTFIX:
            # G1: no waiting entry is cut to zero by KEEPALIVE_MAX_ENTRIES any more; the track's keep-alive share cap
            # (KEEPALIVE_MAX_SHARE) still bounds the total and is split evenly when it binds
            if len(elig) > KEEPALIVE_MAX_ENTRIES:
                log.info("track %s: keep-alive for all %d waiting entries (more than %d), within the share cap",
                         track_id, len(elig), KEEPALIVE_MAX_ENTRIES)
            kept = elig
        else:
            kept = cut(track_id, elig)
        if kept:
            alive[track_id] = kept
    ordinary_alive = {}
    for track_id, candidates in ordinary_cand.items():
        ids = [u for u in candidates if own.get(u, 0.0) < per_entry]
        ids = [u for u in ids if u not in own] + [u for u in ids if u in own]
        ordinary_alive[track_id] = ids if WAITING_KA_HOTFIX else cut(track_id, ids, ordinary_protected or {})
    ordinary_taken = set(paid) | set(ordinary_engine_pay) | {u for r in ranks_of.values() for u in r}
    ordinary_taken.update(u for ids in ordinary_alive.values() for u in ids)
    ordinary_prov = {t: [u for u in ids if u not in ordinary_taken and
                        (own.get(u, 0.0) < PROVISIONAL_KA_FRACTION * per_entry
                         if WAITING_KA_HOTFIX else u not in own)] for t, ids in prov_cand.items()} if halted else {}
    taken = {u for ids in alive.values() for u in ids} | set(paid)
    for ranked_t in ranks_of.values():
        taken.update(ranked_t)
    # provisional pay keeps an UNSERVED waiting entry from being pruned: a hotkey that serves is not at that risk, so only
    # pure entrants get it (critic round 1 F9) and only they make a track hold a slot (round 2 NEW-2)
    if WAITING_KA_HOTFIX:
        # G1: a waiting hotkey that serves a little is still at risk: it is topped up to the provisional level
        prov_all = {t: [u for u in ids if u not in taken and own.get(u, 0.0) < PROVISIONAL_KA_FRACTION * per_entry]
                    for t, ids in prov_cand.items()}
    else:
        prov_all = {t: [u for u in ids if u not in taken and u not in own] for t, ids in prov_cand.items()}
    # which tracks hold a slot: only tracks with a payable entry (or ranks), so an entry that would be paid nothing never
    # pushes another track's entrants out of the profile's slots
    slot_tracks = set(tracks) | {t for t, v in alive.items() if v} | {t for t, v in prov_all.items() if v}
    if generalist_podium is not None:
        slot_tracks.add("generalist")   # the podium pool exists even when the generalist track has no score or keep-alive entry
    slots = list(zip(sorted(slot_tracks), prof.tracks))  # (track id, slot share), as paid
    slot_ids = {t for t, _ in slots}
    prov = {t: ids for t, ids in prov_all.items() if t in slot_ids and ids}
    if generalist_ka_used is not None:
        generalist_ka_used.update(keepalive=list(alive.get("generalist", ())) if "generalist" in slot_ids else [],
                                  provisional=list(prov.get("generalist", ())))
    ewait: list[int] = []
    if ewait_cand:
        # engine keep-alive (miner report 21903): pure entrants only, never a uid paid elsewhere
        taken = {u for t in slot_ids for u in alive.get(t, ())} | {u for ids in prov.values() for u in ids} | set(paid)
        for ranked_t in ranks_of.values():
            taken.update(ranked_t)
        for uid in ewait_cand:
            low = own.get(uid, 0.0) < ENGINE_KA_FRACTION * per_entry if WAITING_KA_HOTFIX else uid not in own
            if uid not in taken and low and (WAITING_KA_HOTFIX or len(ewait) < ENGINE_KA_MAX_ENTRIES + ENGINE_KA_PROTECTED_MAX):
                ewait.append(uid)
    if ewait:
        each_ew = min(ENGINE_KA_FRACTION * per_entry, ENGINE_KA_MAX_SHARE * prof.engine / len(ewait))
        for uid in ewait:
            paid[uid] = max(0.0, each_ew - own.get(uid, 0.0)) if WAITING_KA_HOTFIX else each_ew
        ew_total = math.fsum(paid[u] for u in ewait)
        # taken from the engine share, i.e. from the champion's part of it (split by payment over co-champions)
        if not engine_cochampions:
            if engine_champion in paid:  # exactly the rule before the co-champions
                paid[engine_champion] = max(0.0, paid[engine_champion] - ew_total)
        else:
            held = math.fsum(paid[u] for u in engine_payees)
            if held > 0:
                take = ew_total
                for u in engine_payees:
                    paid[u] = max(0.0, paid[u] - take * paid[u] / held)
        log.info("engine keep-alive for uids %s, each %.5f of the miner part", ewait, each_ew)
    if "engine" in halted:
        # Reconstruct the ordinary engine keep-alive deduction: paused title/former
        # floors are additional payees, not additional waiting entrants in that budget.
        taken_normal = ({u for t in slot_ids for u in alive.get(t, ())}
                        | {u for ids in prov.values() for u in ids} | set(ordinary_engine_pay))
        taken_normal.update(u for r in ranks_of.values() for u in r)
        normal_wait = [u for u in dict.fromkeys(ordinary_engine_waiting)
                       if u != owner_uid and u not in taken_normal
                       and (own.get(u, 0.0) < ENGINE_KA_FRACTION * per_entry
                            if WAITING_KA_HOTFIX else u not in own)] if ka_cap > 0 else []
        if not WAITING_KA_HOTFIX:
            normal_wait = normal_wait[:ENGINE_KA_MAX_ENTRIES + ENGINE_KA_PROTECTED_MAX]
        each = min(ENGINE_KA_FRACTION * per_entry, ENGINE_KA_MAX_SHARE * prof.engine / len(normal_wait)) if normal_wait else 0.0
        deduction = math.fsum(max(0.0, each - own.get(u, 0.0)) if WAITING_KA_HOTFIX else each for u in normal_wait)
        held = math.fsum(ordinary_engine_pay.values())
        room = max(0.0, prof.engine - math.fsum(paid.get(u, 0.0) for u in ewait))
        extras = {}
        for uid, (part, crowned) in sorted((engine_runout or {}).items()):
            if uid != owner_uid and part > 0:
                baseline = prof.engine * min(1.0, part)
                if held > 0:
                    baseline = max(0.0, baseline - deduction * ordinary_engine_pay.get(uid, 0.0) / held)
                amount = baseline * runout_factor(crowned, starts["engine"], now,
                                                    (engine_approved_at or {}).get(uid))
                extras[uid] = max(0.0, amount - paid.get(uid, 0.0))
        total_extra = math.fsum(extras.values())
        scale = min(1.0, room / total_extra) if total_extra > 0 else 1.0
        for uid, extra in extras.items():
            paid[uid] = paid.get(uid, 0.0) + extra * scale
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
    legacy_paid = math.fsum(paid[u] for u in (legacy if legacy_cand else ()) if u in paid)
    generalist_unit_logged = False
    for track_id, slot_share in slots:  # a track without a slot pays nothing
        rank_share = slot_share
        if track_id in alive:
            n = len(alive[track_id])
            # generalist: the keep-alive cap is never above KEEPALIVE_MAX_SHARE even if keepalive_share is raised
            cap = min(ka_cap, KEEPALIVE_MAX_SHARE) if track_id == "generalist" else ka_cap
            room = cap * slot_share
            if WAITING_KA_HOTFIX and track_id == "generalist" and prov.get(track_id):
                # critic2 N1: the provisional share is reserved first, so a crowd of keep-alive entries (no count cut any more)
                # cannot squeeze the waiting committed entries to zero; the keep-alive gets the rest of the 25 %
                room = max(0.0, room - min(PROVISIONAL_KA_MAX_SHARE * slot_share,
                                           len(prov[track_id]) * PROVISIONAL_KA_FRACTION * per_entry))
            each = min(per_entry, room / n)
            if track_id == "generalist" and log_generalist_unit:
                # All amounts except miner_part are fractions of the miner part.
                # Credit is the conservative `own` used for top-ups, not necessarily
                # the final serving pay. No typical miner gives per_entry=inf.
                log.info("generalist keep-alive unit: mode %s, per_entry %.6f, each %.6f, room %.6f, "
                         "n %d, miner_part %.6f; uids [%s]",
                         "feed" if feed_mode else "legacy", per_entry, each, room, n, miner_part,
                         ", ".join(f"{u}:{own.get(u, 0.0):.6f}:{max(0.0, each - own.get(u, 0.0)):.6f}"
                                   for u in alive[track_id]))
                generalist_unit_logged = True
            for uid in alive[track_id]:
                paid[uid] = paid.get(uid, 0.0) + max(0.0, each - own.get(uid, 0.0))
            # The top-ups actually paid; credit serving income as in the legacy path.
            rank_share = slot_share - (each * n - math.fsum(min(each, own.get(u, 0.0)) for u in alive[track_id]))
        ids = prov.get(track_id, [])
        if ids:
            n_prov = len(ids)
            prov_cap = PROVISIONAL_KA_MAX_SHARE * slot_share
            if track_id == "generalist":
                prov_cap = min(prov_cap, max(0.0, min(ka_cap, KEEPALIVE_MAX_SHARE) * slot_share - (slot_share - rank_share)))
            each_prov = min(PROVISIONAL_KA_FRACTION * per_entry, prov_cap / n_prov)
            if WAITING_KA_HOTFIX:   # G1: a top-up (a pure entrant gets all of it, as before)
                top = {uid: max(0.0, each_prov - own.get(uid, 0.0)) for uid in ids}
                for uid in ids:
                    paid[uid] = paid.get(uid, 0.0) + top[uid]
                rank_share -= math.fsum(top.values())
            else:
                for uid in ids:
                    paid[uid] = paid.get(uid, 0.0) + each_prov
                rank_share -= each_prov * n_prov
            log.info("track %s: provisional keep-alive for uids %s, each %.5f of the miner part",
                     track_id, ids, each_prov)
        if track_id in halted:
            if track_id == "generalist" and generalist_podium_used is not None:
                generalist_podium_used["paused"] = True   # the ordinary paused path runs; no podium record is claimed
            # Use the same budget as the unpaused rank path. The additional paused-rank
            # keep-alive floors must not reduce the title's baseline a second time.
            if not (feed_mode and track_id == "generalist"):
                ordinary = ordinary_alive.get(track_id, [])
                normal_prov = ordinary_prov.get(track_id, [])
                cap = min(ka_cap, KEEPALIVE_MAX_SHARE) if track_id == "generalist" else ka_cap
                room = cap * slot_share
                if WAITING_KA_HOTFIX and track_id == "generalist" and normal_prov:
                    room = max(0.0, room - min(PROVISIONAL_KA_MAX_SHARE * slot_share,
                                               len(normal_prov) * PROVISIONAL_KA_FRACTION * per_entry))
                each = min(per_entry, room / len(ordinary)) if ordinary else 0.0
                rank_share = slot_share - (each * len(ordinary) - math.fsum(
                    min(each, own.get(u, 0.0)) for u in ordinary))
                if normal_prov:
                    cap_prov = PROVISIONAL_KA_MAX_SHARE * slot_share
                    if track_id == "generalist":
                        cap_prov = min(cap_prov, max(0.0, cap * slot_share - (slot_share - rank_share)))
                    each_prov = min(PROVISIONAL_KA_FRACTION * per_entry, cap_prov / len(normal_prov))
                    rank_share -= math.fsum(max(0.0, each_prov - own.get(u, 0.0)) for u in normal_prov)
            tr = tracks.get(track_id, ())
            ranks = track_payout(tr, floors.get(track_id))
            if ranks:
                uid, full = next(iter(ranks.items()))
                members = getattr(tr, "cochamps", None) or {uid: 1.0}
                crowned = getattr(tr, "crowned", {})
                for member, part in sorted(members.items()):
                    titles = getattr(tr, "champions", None)
                    if member == owner_uid or (titles is not None and member not in titles) or \
                            (award_ok is not None and not award_ok(track_id, member)):
                        continue
                    amount = rank_share * full * min(1.0, max(0.0, part)) * runout_factor(
                        crowned.get(member), starts[track_id], now,
                        award_approved_at(track_id, member) if award_approved_at is not None else None)
                    paid[member] = max(paid.get(member, 0.0), amount)
            continue
        if track_id == "generalist" and generalist_podium is not None:
            # Round 2 podium (gr2_podium.py): keep-alive and provisional pay are settled above; the podium is paid from the
            # rest of the pool. Nothing is promoted: a slot that is not payable burns its own share.
            ka_paid = min(slot_share, max(0.0, slot_share - rank_share))
            eligible = {slot: uid for slot, uid in generalist_podium.eligible.items()
                        if uid != owner_uid and (award_ok is None or award_ok(track_id, uid))}
            fractions = _gr2_podium.payout_fractions(generalist_podium.window, generalist_podium.shares, eligible) \
                if eligible else {}
            pay = _gr2_podium.payout_after_keepalive(slot_share, ka_paid, fractions)
            for uid, amount in pay.items():
                paid[uid] = paid.get(uid, 0.0) + amount
            if generalist_podium_used is not None:
                base_pool = miner_part * (slot_share - ka_paid)
                generalist_podium_used.update(
                    window=generalist_podium.window, pool=miner_part * slot_share, keepalive=miner_part * ka_paid,
                    eligible=dict(eligible), podium={uid: miner_part * amount for uid, amount in pay.items()},
                    burn=max(0.0, base_pool * (1.0 - math.fsum(fractions.values()))))
            log.debug("generalist podium %s: pool %.6f keep-alive %.6f of the miner part, paid slots %s",
                     generalist_podium.window, slot_share, ka_paid, sorted(eligible))
            continue
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
    if log_generalist_unit and not generalist_unit_logged:
        log.info("generalist keep-alive unit: none")
    paid.pop(owner_uid, None)  # never to the owner UID: its share stays with the inference part
    claimed = math.fsum(paid.values())
    if target is not None:
        return _need_weights(base, paid, owner_uid, burn, target, legacy_paid, owner_allocation)
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
