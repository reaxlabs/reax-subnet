"""Generated waiting-state timelines for the keep-alive gauntlet (job keepalive-gauntlet-20261026, slice proptests).

A *world* is a set of hotkeys. Each hotkey has one or more *episodes* (a commitment and what happens to it: a generalist
entry or an engine digest). An episode is a list of *segments* ``(t_start, patch)``: from ``t_start`` the validator's state
looks like ``patch`` (the fields of a ``spec.state.Submission``, or the engine ``TrackState`` entries of one digest). The
state at a weight set is built from the segments and fed to the REAL code (``compute_keepalive`` / ``compute_provisional`` /
``keepalive_protected`` / ``legacy_leaders`` / ``EngineCompetition.weights`` -> ``combine_weights``); nothing of the share
layer is re-implemented here. Only the weights come out of the code; what *should* happen is decided by the independent
oracle (``episode.excl``: the published exclusions, each with its tag).

Times are unix seconds, blocks are 12 s. Anchor: block 9_369_048 (an epoch boundary: block % 360 == 48) is 26 Oct 2026 17:00Z.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet import engvalidator as EV
from reax_subnet import ka_override as KO
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST
from reax_subnet.spec.commitment import SpecCommitment

H = 3600.0
DAY = 86400.0
INF = math.inf
REV = "c" * 40
GEN, DF, RA, ENG = "generalist", "demand_forecasting", "risk_assessment", "s1-fast"
OWNER_UID = 0
OWNER_HK = "owner"
REF_DIGEST = "ab" * 32

T_REF = datetime(2026, 10, 26, 17, 0, tzinfo=timezone.utc).timestamp()
B_REF = 9_369_048
EPOCH = 360
EPOCH_PHASE = B_REF % EPOCH     # 48


def ts(*a) -> float:
    return datetime(*a, tzinfo=timezone.utc).timestamp()


def block_at(t: float) -> int:
    return B_REF + int((t - T_REF) // 12)


def time_of(block: int) -> float:
    return T_REF + (block - B_REF) * 12.0


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def next_boundary(b: int) -> int:
    """First epoch boundary strictly after block ``b``."""
    return b + EPOCH - ((b - EPOCH_PHASE) % EPOCH)


def effective_block(set_block: int) -> int:
    """Commit-reveal (CommitRevealWeightsEnabled, RevealPeriodEpochs=1): a set at block b counts at the 2nd epoch boundary after b."""
    return next_boundary(set_block) + EPOCH


# The published numbers the oracle uses (SPEC.md of keepalive-hotfix-20261026, section 2), as literals: NOT read from the code
# under test, so a changed constant is caught (``test_release_constants_match_the_spec`` pins the code to the same literals).
SPEC = dict(unverified_h=24.0, rewindow_h=72.0, unverified_max=16, our_wait_h=96.0, pending_cap_h=216.0, approval_h=72.0,
            engine_wait_h=24.0, engine_slot_after_h=4.0, engine_slot_cap_h=120.0, share_ka=0.25, share_prov=0.05,
            share_engine_ka=0.05, share_legacy=0.03)

GEN_SLOTS = [t.timestamp() for t in CS.CROWN_SLOTS_UTC]
ENG_SLOTS = [t.timestamp() for t in CS.ENGINE_CROWN_SLOTS_UTC]


def next_slot(t: float, slots=None) -> float:
    return next(s for s in (slots or GEN_SLOTS) if s >= t)


@dataclass
class Episode:
    """One commitment of one hotkey and its life until it is decided."""
    kind: str                       # "gen" | "eng"
    hk: str
    uid: int
    coldkey: str
    tc: float                       # on-chain commitment time
    reg_block: int
    segs: list = field(default_factory=list)      # [(t_start, patch|None)]
    decided_t: float = INF
    end_t: float = INF              # the commitment is replaced / the hotkey pruned: the episode ends (not "decided")
    excl: list = field(default_factory=list)      # [(t0, t1, tag)] published exclusions (oracle)
    repo: str = ""
    digest: str = ""
    label: str = ""                 # scenario label
    first_pass: Optional[float] = None
    approved_t: float = INF
    first_seen_ts: Optional[float] = None
    reg_t: float = 0.0              # when this uid was registered (the hotkey holds the uid from then on)
    end_kind: str = ""              # how the episode ended: "changed" (recommit / other track) | "pruned"
    crowned_t: float = INF

    @property
    def cb(self) -> int:
        return block_at(self.tc)

    def patch(self, t: float):
        cur = None
        for t0, p in self.segs:
            if t0 <= t:
                cur = p
            else:
                break
        return cur

    def required(self, t: float) -> bool:
        """P1 applies at ``t``: committed, not decided, not replaced, and no published exclusion covers ``t``."""
        return self.tc <= t < min(self.decided_t, self.end_t)

    def excluded(self, t: float) -> Optional[str]:
        for a, b, tag in self.excl:
            if a <= t < b:
                return tag
        return None


# ------------------------------------------------------------------------------------------------ generalist episodes
def _sub_defaults() -> dict:
    return dict(status=ST.QUEUED, reason="", wait_kind="", hash_named="", delta_checked=False, score=None, nonproc=None,
                by_tier={}, history=[], first_pass_ts=None, confirm_after_ts=None, champion_since_ts=None, close_draws=[],
                draw1="", first_gain=None, params=None, base_id="", withdrawn_ts=None, withdrawn_kind="", withdrawn_status="")


def _scored(v5: bool) -> dict:
    return dict(score=44.0, nonproc=44.0, by_tier={"T": 0.5, "N": 0.4}) if v5 else dict(score=50.0, nonproc=50.0, by_tier={})


BIG_SHARE = 0.3          # share of generated generalist entries with 26-36 B parameters (all live entries are in that class)


def _big_info(hk: str, tc: float) -> Optional[dict]:
    """Parameters 26-36 B (open class, above KEEPALIVE_BIG_PARAMS), a base with or without a floor in KEEPALIVE_BASE_FLOOR, and a
    score above / below that floor + KEEPALIVE_BIG_MARGIN. Drawn from its own stream so the other draws of a world do not move."""
    r = random.Random(f"big|{hk}|{int(tc)}")
    if r.random() >= BIG_SHARE:
        return None
    floors = sorted(CS.KEEPALIVE_BASE_FLOOR)
    has_floor = r.random() < 0.6
    base_id = r.choice(floors) if has_floor else "ff" * 32
    floor = CS.KEEPALIVE_BASE_FLOOR[base_id]["generalist"] if has_floor else None
    above = has_floor and r.random() < 0.5
    v5 = r.random() < 0.4
    q = None if floor is None else (floor + r.choice([1.6, 3.0, 6.0]) if above else floor + r.choice([-3.0, 0.5, 1.4, 1.5]))
    return dict(params=int(r.uniform(26e9, 36e9)), base_id=base_id, floor=floor, q=q, v5=v5, above=above)


def _score_of(big: Optional[dict], v5: bool) -> dict:
    if big is None:
        return _scored(v5)
    if big["v5"]:
        return _scored(True)
    q = big["q"] if big["q"] is not None else 52.0
    return dict(score=q, nonproc=q, by_tier={})


def gen_episode(rng: random.Random, hk: str, uid: int, coldkey: str, tc: float, reg_block: int, mode: str, *,
                prior_seen: tuple = (), unverified_hours: float = SPEC["unverified_h"], our_wait_hours: float = SPEC["our_wait_h"],
                scale: float = 1.0) -> Episode:
    """A generalist entry (see ``_gen_episode``); the 26-36 B class adds the published score-based base-floor exclusion."""
    big = _big_info(hk, tc)
    ep = _gen_episode(rng, hk, uid, coldkey, tc, reg_block, mode, prior_seen=prior_seen, unverified_hours=unverified_hours,
                      our_wait_hours=our_wait_hours, scale=scale, big=big)
    if big is not None:
        ep.label += "/big" + ("-v5" if big["v5"] else "-above" if big["above"] else "-nofloor" if big["floor"] is None else "-below")
        beats = big["v5"] or (big["floor"] is not None and big["q"] > big["floor"] + 1.5)
        if not beats and ep.first_pass is not None:
            # SPEC: the base floor stays a test on a SCORE for the pending / champion keep-alive (big entries below base floor + 1.5,
            # or with no floor for their base, get no score-based keep-alive; undecided entries keep the provisional one)
            ep.excl.append((ep.first_pass, ep.decided_t, "E_BIG_BASE_FLOOR"))
    return ep


def _gen_episode(rng: random.Random, hk: str, uid: int, coldkey: str, tc: float, reg_block: int, mode: str, *,
                 prior_seen: tuple = (), unverified_hours: float = SPEC["unverified_h"], our_wait_hours: float = SPEC["our_wait_h"],
                 scale: float = 1.0, big: Optional[dict] = None) -> Episode:
    """A generalist entry. ``mode``: how the fetch goes: ok | f401 | f404 | ftimeout | private | base_miner | base_infra.
    Durations are drawn from ``rng``; ``scale`` > 1 stretches our own waits (beyond-bounds runs)."""
    repo = f"m/{hk[2:7]}-{int(tc) % 100000}"
    ep = Episode("gen", hk, uid, coldkey, tc, reg_block, repo=repo, label=mode)
    d = lambda lo, hi: rng.uniform(lo, hi) * H * scale  # noqa: E731
    t_seen = tc + rng.choice([0.0, 0.2, 1.0, 2.5]) * H
    ep.first_seen_ts = t_seen
    segs: list = [(tc, None)]
    base = _sub_defaults()
    miner_wait_until = None
    t = t_seen
    if mode in ("f401", "f404", "private"):
        reason = E.WAITING_FOR_REPO + (": 404 Client Error: revision not found" if mode == "f404" else ": 401 Client Error: private")
        segs.append((t_seen, {**base, "reason": reason, "wait_kind": ST.WAIT_MINER}))
        t = INF if mode == "private" else t_seen + rng.uniform(0.5, 60.0) * H
        miner_wait_until = t
    elif mode == "ftimeout":
        segs.append((t_seen, {**base, "reason": E.WAITING_FOR_REPO + ": ReadTimeout", "wait_kind": ST.WAIT_INFRA}))
        t = t_seen + d(0.5, 40.0)
    else:
        segs.append((t_seen, dict(base)))                        # queued, not fetched yet: our queue
        t = t_seen + d(0.0, 30.0)
    if t == INF:
        ep.segs = segs
        _gen_excl_unverified(ep, tc, INF, mode, prior_seen, t_seen, unverified_hours)
        return ep
    if mode in ("f401", "f404"):
        _gen_excl_unverified(ep, tc, t, mode, prior_seen, t_seen, unverified_hours)
    fetched = {**base, "hash_named": "h" + hk[2:7], "delta_checked": True}
    if big is not None:
        fetched.update(params=big["params"], base_id=big["base_id"])
    if mode in ("base_miner", "base_infra"):
        kind = ST.WAIT_MINER if mode == "base_miner" else ST.WAIT_INFRA
        segs.append((t, {**base, "status": ST.PENDING_VERIFY, "wait_kind": kind, "hash_named": "h" + hk[2:7],
                         "reason": "base model not on the hub" if kind == ST.WAIT_MINER else "waiting for the base download"}))
        t2 = t + (rng.uniform(1.0, 60.0) * H if kind == ST.WAIT_MINER else d(1.0, 40.0))
        if kind == ST.WAIT_MINER:
            _gen_excl_unverified(ep, tc, t2, mode, prior_seen, t_seen, unverified_hours)
        t = t2
    segs.append((t, fetched))
    t_draw = t + d(0.0, 40.0)
    ours_end = t_draw
    v5 = rng.random() < 0.85
    draw = rng.choice(["pass", "pass", "pass", "owed_pass", "owed_fail", "fail"])
    ep.label += "/" + draw
    sc = _score_of(big, v5)
    if draw == "fail":
        segs.append((t_draw, {**fetched, "status": ST.NOT_BETTER, **{**sc, "score": 12.0, "nonproc": 12.0}}))
        ep.decided_t = t_draw
        return _finish_gen(ep, segs, ours_end, tc, our_wait_hours)
    fp = t_draw
    if draw.startswith("owed"):
        segs.append((t_draw, {**fetched, **sc, "close_draws": ["d1"]}))
        t_draw2 = t_draw + d(0.5, 30.0)
        ours_end = t_draw2
        if draw == "owed_fail":
            segs.append((t_draw2, {**fetched, "status": ST.NOT_BETTER, **{**sc, "score": 20.0, "nonproc": 20.0}}))
            ep.decided_t = t_draw2
            return _finish_gen(ep, segs, ours_end, tc, our_wait_hours)
        fp = t_draw2
    ep.first_pass = fp
    pend = {**fetched, **sc, "status": ST.PENDING_CONFIRM, "first_pass_ts": fp, "confirm_after_ts": fp + 12 * H,
            "draw1": "d1", "first_gain": 3.0}
    segs.append((fp, dict(pend)))
    # excluded by the published hard cap: 216 h after the first pass (SPEC H1)
    t_conf = fp + 12 * H + d(0.0, 40.0)
    hist: list = []
    outcome = rng.choice(["crown", "crown", "notbetter", "void_crown", "void_void_notbetter", "conf_fail"])
    ep.label += "/" + outcome
    if outcome == "conf_fail":
        segs.append((t_conf, {**pend, "status": ST.NOT_BETTER}))
        ep.decided_t = t_conf
        return _finish_gen(ep, segs, ours_end, tc, our_wait_hours, fp=fp)
    voids = {"crown": 0, "notbetter": 0, "void_crown": 1, "void_void_notbetter": 2}[outcome]
    t_dec = None
    for k in range(voids + 1):
        held = rng.random() < 0.3                                  # held by the 6 h cutoff to the following slot
        slot = next_slot(t_conf)
        row = {"kind": "crown_pending", "ts": t_conf}
        if held or rng.random() < 0.25:
            if held and t_conf >= slot - 6 * H:
                slot = next_slot(slot + 1)
            row = {**row, "slot": slot, "k1a": True}
        elif t_conf > slot - 6 * H and t_conf >= ts(2026, 10, 29):
            slot = next_slot(slot + 1)                               # 6 h cutoff from Thu 29 Oct
            row = {**row, "slot": slot, "k1a": True}
        hist = hist + [row]
        segs.append((t_conf, {**pend, "history": list(hist)}))
        t_dec = slot + rng.uniform(0.0, 11.0) * H
        if k < voids:
            hist = hist + [{"kind": "crown_done", "ts": t_dec}]       # slot void / back on the challenger path
            segs.append((t_dec, {**pend, "history": list(hist)}))
            t_conf = t_dec + d(0.0, 30.0)
            continue
    if outcome in ("notbetter", "void_void_notbetter"):
        segs.append((t_dec, {**pend, "status": ST.NOT_BETTER, "history": hist + [{"kind": "crown_done", "ts": t_dec}]}))
        ep.decided_t = t_dec
    else:
        appr = rng.choice([0.2, 1.0, 5.0, 20.0, 40.0, 60.0, 70.0, 100.0]) * H
        segs.append((t_dec, {**pend, "status": ST.CHAMPION, "champion_since_ts": t_dec,
                             "history": hist + [{"kind": "crown_done", "ts": t_dec}]}))
        ep.approved_t = t_dec + appr
        ep.decided_t = ep.approved_t
        ep.crowned_t = t_dec
        if appr > SPEC["approval_h"] * H:
            ep.excl.append((t_dec + SPEC["approval_h"] * H, ep.approved_t, "E_APPROVAL_72H"))
    return _finish_gen(ep, segs, ours_end, tc, our_wait_hours, fp=fp)


def _finish_gen(ep: Episode, segs: list, ours_end: float, tc: float, our_wait_hours: float, fp: Optional[float] = None) -> Episode:
    ep.segs = segs
    if ours_end > tc + our_wait_hours * H:
        ep.excl.append((tc + our_wait_hours * H, min(ours_end, ep.decided_t), "E_OURWAIT_96H"))   # SPEC G1 safety cap
    if fp is not None:
        ep.excl.append((fp + SPEC["pending_cap_h"] * H, ep.decided_t, "E_PENDING_216H"))        # SPEC H1 hard cap
    return ep


def _gen_excl_unverified(ep: Episode, tc: float, until: float, mode: str, prior_seen: tuple, t_seen: float, hours: float) -> None:
    """Miner-side wait (repo missing/private, declared base not on the hub): published bound 24 h from the start of the
    hotkey's unverified window; one such window per hotkey per 72 h (SPEC G1: "repo not found/private" - a 404 included; the
    abuse clause of commit 233dbd2 that takes the window from a 404 is NOT in SPEC.md and waits for Florian's yes: finding F4)."""
    start = tc
    for s in sorted(prior_seen):
        if s <= tc and tc - s < SPEC["rewindow_h"] * H:
            start = None            # inside the 72 h of an earlier window of the same hotkey: no new unverified window
            break
    a = tc + hours * H if start is not None else tc
    ep.excl.append((a, until, "E_UNVERIFIED_24H" if start is not None else "E_REWINDOW_72H"))


# ------------------------------------------------------------------------------------------------ engine episodes
def eng_episode(rng: random.Random, hk: str, uid: int, coldkey: str, tc: float, reg_block: int, *, scale: float = 1.0,
                occupied: Optional[list] = None) -> Episode:
    """An engine digest. The engine state holds ONE challenger at a time (``TrackState.challenger``): a digest that would pass
    while another challenger is pending is measured below the bar instead (``occupied``: [(t_from, t_to)] of the pending ones)."""
    digest = hashlib.sha256(f"{hk}{int(tc)}".encode()).hexdigest()
    ep = Episode("eng", hk, uid, coldkey, tc, reg_block, digest=digest, repo=f"ghcr.io/m/{hk[2:7]}")
    d = lambda lo, hi: rng.uniform(lo, hi) * H * scale  # noqa: E731
    t_seen = tc + rng.choice([0.0, 0.3, 2.0]) * H
    segs: list = [(tc, {"first_seen": None, "owner": None, "result": None, "challenger": None, "champion": False})]
    ep.first_seen_ts = t_seen
    base = {"first_seen": (block_at(t_seen), hk), "owner": None, "result": None, "challenger": None, "champion": False}
    segs.append((t_seen, dict(base)))
    t_bound = t_seen + d(0.0, 12.0)
    bound = {**base, "owner": hk}
    segs.append((t_bound, dict(bound)))
    t_meas = t_bound + d(0.0, 30.0)
    # our wait for the first measurement: published window ENGINE_KA_WAIT_HOURS from the commitment
    if t_meas > tc + SPEC["engine_wait_h"] * H:
        ep.excl.append((tc + SPEC["engine_wait_h"] * H, t_meas, "E_ENGINE_WAIT_24H"))
    outcome = rng.choice(["low", "pass", "pass", "pass", "conf_fail"])
    t_conf = t_meas + d(1.0, 40.0)
    slot = next_slot(t_conf, ENG_SLOTS)
    t_dec = slot + rng.uniform(0.0, 3.0) * H
    busy = (t_meas, t_conf if outcome == "conf_fail" else t_dec)
    if outcome != "low" and occupied is not None and any(a < busy[1] and busy[0] < b_ for a, b_ in occupied):
        outcome = "low"
    elif outcome != "low" and occupied is not None:
        occupied.append(busy)
    ep.label = "eng/" + outcome

    def res(score):
        return EC.EvalResult(digest=digest, track=ENG, ok=True, score=score, items=100, block=block_at(t_meas), session=f"s{hk[2:5]}1")

    if outcome == "low":
        segs.append((t_meas, {**bound, "result": res(0.97)}))
        ep.decided_t = t_meas
        ep.segs = segs
        return ep
    chal = {"digest": digest, "hotkey": hk, "measurements": [[block_at(t_meas), 1.4, f"s{hk[2:5]}1"]]}
    pend = {**bound, "result": res(1.4), "challenger": chal}
    segs.append((t_meas, dict(pend)))
    first_session = t_meas
    if outcome == "conf_fail":
        segs.append((t_conf, {**bound, "result": res(1.4)}))
        ep.decided_t = t_conf
        if t_conf > first_session + SPEC["engine_slot_cap_h"] * H:
            ep.excl.append((first_session + SPEC["engine_slot_cap_h"] * H, t_conf, "E_ENGINE_SLOT_CAP_120H"))
        ep.segs = segs
        return ep
    chal2 = {**chal, "measurements": chal["measurements"] + [[block_at(t_conf), 1.38, f"s{hk[2:5]}2"]],
             "confirmed_at": iso(t_conf), "slot_utc": iso(slot)}
    segs.append((t_conf, {**pend, "challenger": chal2}))
    # published: a confirmed challenger keeps it until slot + 4 h, at most 120 h after its first session
    limit = min(slot + SPEC["engine_slot_after_h"] * H, first_session + SPEC["engine_slot_cap_h"] * H)
    if t_dec > limit:
        ep.excl.append((limit, t_dec, "E_ENGINE_SLOT_CAP_120H"))
    appr = rng.choice([0.5, 3.0, 20.0, 50.0]) * H
    segs.append((t_dec, {**bound, "result": res(1.4), "challenger": None, "champion": True, "crowned_block": block_at(t_dec)}))
    ep.approved_t = t_dec + appr
    ep.decided_t = ep.approved_t
    ep.crowned_t = t_dec
    ep.segs = segs
    return ep


# ------------------------------------------------------------------------------------------------ the world
@dataclass
class Hotkey:
    hk: str
    coldkey: str
    episodes: list = field(default_factory=list)       # in commitment order
    gaps: list = field(default_factory=list)           # [(t_prune, t_reregister, new_uid)] re-registration
    serving: float = 0.0                                # serving weight (dual role)


class Gate:
    """award.AwardGate stand-in (approved (track, hotkey) pairs; the set is rebuilt at every weight set)."""

    def __init__(self):
        self.ok: set = set()
        self.path = ""

    def approved(self, track, hotkey, sub_id):
        return (track, hotkey) in self.ok


class _R:
    binds_hotkey = True


class Recorder:
    """Captures what ``EngineCompetition.weights`` hands to ``combine_weights`` (the real call, only observed)."""

    def __init__(self):
        self.kwargs: dict = {}

    def combine(self, *a, **kw):
        # (serving, engine_champion, engine_share_frac, tracks, ...): keep the positional ones under their names
        self.kwargs = dict(kw, engine_champion=a[1], tracks=a[3])
        return CS.combine_weights(*a, **kw)


class Specialist:
    """The part of the spec evaluator the share layer reads, over an in-memory SpecState (the real compute_* functions)."""

    def __init__(self):
        self.state: ST.SpecState = ST.SpecState()
        self.snap: Optional[Snapshot] = None
        self.now = 0.0

    def track_results(self, ids, now):
        return E.compute_leaders(self.state, ids, self.snap, now=self.now)

    def keepalive(self, ids, now):
        return E.compute_keepalive(self.state, ids, self.snap, now=self.now)

    def provisional(self, ids, now):
        return E.compute_provisional(self.state, ids, self.snap, now=self.now)

    def protected(self, ids, now):
        return E.keepalive_protected(self.state, ids, self.snap)

    def legacy(self, ids, now):
        return E.legacy_leaders(self.state, ids, self.snap)


def ss58(i: int) -> str:
    """A valid-looking ss58 address (base58 alphabet, 47 characters: ka_override accepts it)."""
    return "5W" + "".join("abcdefghij"[int(c)] for c in f"{i:05d}") + "x" * 40


@dataclass
class World:
    seed: int
    t0: float
    horizon: float
    cadence: float
    phase: float
    hotkeys: list
    serving: dict               # uid -> weight (inference miners, incl. owner burn entry)
    burn: float
    incumbent_gen: list         # [(uid, hk, status, score, since)] generalist incumbents: champion + formers (approved)
    legacy: list                # [(uid, hk, track, champion?, crowned)] ended DF/RA leaders
    eng_incumbent: Optional[tuple]      # (uid, hk, digest, crowned_block)
    override: list              # [(hotkey, kind, until)]
    reg_default: int
    reregs: dict = field(default_factory=dict)


def _sub(ep: Episode, patch: dict, t_seen: float) -> ST.Submission:
    key = f"{ep.hk}|{SpecCommitment(GEN, ep.repo, REV).text()}"
    s = ST.Submission(key=key, uid=ep.uid, hotkey=ep.hk, coldkey=ep.coldkey, track_id=GEN, repo=ep.repo, revision=REV,
                      first_seen_block=block_at(t_seen), first_seen_ts=t_seen)
    for k, v in patch.items():
        setattr(s, k, list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
    return s


class Runner:
    """Runs a world hour by hour through the real weight code and records the weights."""

    def __init__(self, world: World, tmpdir: str):
        self.w = world
        self.gate = Gate()
        self.rec = Recorder()
        self.spec = Specialist()
        self.now = [datetime.fromtimestamp(world.t0, timezone.utc)]
        self.tmpdir = tmpdir
        appr = os.path.join(tmpdir, "approvals.json")
        with open(appr, "w") as f:
            f.write("[]")
        self.gate.path = appr
        if world.override:
            with open(os.path.join(tmpdir, KO.FILE_NAME), "w") as f:
                json.dump([{"hotkey": hk, "kind": kind, "until_utc": iso(until), "reason": "gauntlet"}
                           for hk, kind, until in world.override], f)
        self.comp = EV.EngineCompetition(_R(), {ENG: "ghcr.io/r/ref@sha256:" + REF_DIGEST}, os.path.join(tmpdir, "eng.json"),
                                         utcnow=lambda: self.now[0], specialist=self.spec, award_gate=self.gate)
        self.comp.engine_keepalive = True
        self.comp.keep_share = True
        self.sets: list = []        # (t, block, weights dict, kwargs)

    # ---- the state at time t
    def _snapshot(self, t: float, block: int):
        w = self.w
        neurons = [Neuron(OWNER_UID, OWNER_HK, "c-owner", False, 0.0, None, None, registered_block=1)]
        for uid in w.serving:
            if uid != OWNER_UID:
                neurons.append(Neuron(uid, ss58(1000 + uid), f"cs{uid}", False, 0.0, None, None, registered_block=1))
        for uid, hk, status, score, since in w.incumbent_gen:
            neurons.append(Neuron(uid, hk, f"cg{uid}", False, 0.0, None, None,
                                  commitment=SpecCommitment(GEN, f"inc/{uid}", REV).text(), commitment_block=1, registered_block=1))
        for uid, hk, track, champ, crowned in w.legacy:
            c = SpecCommitment(track, f"leg/{uid}", REV).text()
            neurons.append(Neuron(uid, hk, f"cl{uid}", False, 0.0, None, None, commitment=c, commitment_block=1, registered_block=1))
        if w.eng_incumbent:
            uid, hk, dg, cbk = w.eng_incumbent
            neurons.append(Neuron(uid, hk, f"ce{uid}", False, 0.0, None, None,
                                  commitment=EC.commitment_text(ENG, "ghcr.io/inc/e", dg), commitment_block=1, registered_block=1))
        for hko in w.hotkeys:
            # latest episode committed by t (a later episode replaces the earlier commitment)
            latest = None
            for ep in hko.episodes:
                if ep.tc <= t:
                    latest = ep
            uid = None
            for ep in hko.episodes:
                if ep.reg_t <= t:
                    uid = ep.uid
            if uid is None:
                continue
            gone = any(a <= t < b for a, b, _ in hko.gaps)
            if gone:
                continue
            reg = next((ep.reg_block for ep in reversed(hko.episodes) if ep.reg_t <= t), 1)
            commit, cbk = "", 0
            if latest is not None and latest.uid == uid:
                if latest.kind == "gen":
                    commit = SpecCommitment(GEN, latest.repo, REV).text()
                else:
                    commit = EC.commitment_text(ENG, latest.repo, latest.digest)
                cbk = latest.cb
            neurons.append(Neuron(uid, hko.hk, hko.coldkey, False, 0.0, None, None, commitment=commit, commitment_block=cbk,
                                  registered_block=reg))
        return Snapshot(block=block, neurons=tuple(neurons), owner_hotkey=OWNER_HK)

    def _spec_state(self, t: float) -> ST.SpecState:
        w = self.w
        st = ST.SpecState()
        g = st.track(GEN)
        new_champ = any(e.kind == "gen" and e.end_t > t and (e.patch(t) or {}).get("status") == ST.CHAMPION
                        for h in w.hotkeys for e in h.episodes)
        for uid, hk, status, score, since in w.incumbent_gen:
            key = f"{hk}|{SpecCommitment(GEN, f'inc/{uid}', REV).text()}"
            s = ST.Submission(key=key, uid=uid, hotkey=hk, coldkey=f"cg{uid}", track_id=GEN, repo=f"inc/{uid}", revision=REV,
                              first_seen_block=1, first_seen_ts=1.0, status=ST.FORMER if (new_champ and status == ST.CHAMPION) else status, score=score, nonproc=score,
                              by_tier={"T": 0.5, "N": 0.4}, champion_since_ts=since, delta_checked=True, hash_named=f"hi{uid}")
            g.submissions[key] = s
            if status == ST.CHAMPION and not new_champ:
                g.champion = key
        champions = []
        for hko in w.hotkeys:
            for ep in hko.episodes:
                if ep.kind != "gen":
                    continue
                p = ep.patch(t)
                if p is None:
                    continue
                s = _sub(ep, p, ep.first_seen_ts)
                if ep.end_t <= t:
                    s.status, s.withdrawn_ts = ST.WITHDRAWN, ep.end_t
                    s.withdrawn_kind, s.withdrawn_status = ep.end_kind, p.get("status", "")
                g.submissions[s.key] = s
                if s.status == ST.CHAMPION and ep.end_t > t:
                    champions.append((s.champion_since_ts or 0.0, s.key))
        if champions:
            g.champion = max(champions)[1]
            for _, k in champions:
                if k != g.champion:
                    g.submissions[k].status = ST.FORMER
        for uid, hk, track, champ, crowned in w.legacy:
            ts_ = st.track(track)
            key = f"{hk}|{SpecCommitment(track, f'leg/{uid}', REV).text()}"
            ts_.submissions[key] = ST.Submission(key=key, uid=uid, hotkey=hk, coldkey=f"cl{uid}", track_id=track, repo=f"leg/{uid}",
                                                 revision=REV, first_seen_block=1, first_seen_ts=1.0,
                                                 status=ST.CHAMPION if champ else ST.FORMER, score=40.0, nonproc=40.0,
                                                 champion_since_ts=crowned, delta_checked=True)
            if champ:
                ts_.champion = key
        return st

    def _engine_state(self, t: float) -> EC.TrackState:
        w = self.w
        st = EC.TrackState()
        if w.eng_incumbent:
            uid, hk, dg, cbk = w.eng_incumbent
            st.first_seen[dg] = (cbk, hk)
            st.owners[dg] = hk
            st.results[dg] = EC.EvalResult(digest=dg, track=ENG, ok=True, score=1.5, items=100, block=cbk, session="inc")
            st.champion = EC.Champion(dg, hk, uid, 1.5, cbk)
            st.history.append({"block": cbk, "event": "crowned", "digest": dg, "hotkey": hk, "score": 1.5})
        newest = None
        for hko in w.hotkeys:
            for ep in hko.episodes:
                if ep.kind != "eng":
                    continue
                p = ep.patch(t)
                if p is None:
                    continue
                if p["first_seen"] is not None:
                    st.first_seen[ep.digest] = p["first_seen"]
                if p["owner"] is not None:
                    st.owners[ep.digest] = p["owner"]
                if p["result"] is not None:
                    st.results[ep.digest] = p["result"]
                if p["challenger"] is not None:
                    st.challenger = dict(p["challenger"])
                if p["champion"] and ep.end_t > t:
                    if newest is None or p["crowned_block"] > newest[0]:
                        newest = (p["crowned_block"], ep)
        if newest is not None:
            cbk, ep = newest
            old = st.champion
            st.history.append({"block": cbk, "event": "crowned", "digest": ep.digest, "hotkey": ep.hk, "score": 1.4})
            st.champion = EC.Champion(ep.digest, ep.hk, ep.uid, 1.4, cbk)
        return st

    # ---- one weight set
    def step(self, t: float) -> dict:
        w = self.w
        block = block_at(t)
        self.now[0] = datetime.fromtimestamp(t, timezone.utc)
        snap = self._snapshot(t, block)
        self.spec.state, self.spec.snap, self.spec.now = self._spec_state(t), snap, t
        self.comp.states[ENG] = self._engine_state(t)
        self.gate.ok = {(GEN, hk) for _, hk, st_, _, _ in w.incumbent_gen} | {(trk, hk) for _, hk, trk, _, _ in w.legacy}
        if w.eng_incumbent:
            self.gate.ok.add((ENG, w.eng_incumbent[1]))
        for hko in w.hotkeys:
            for ep in hko.episodes:
                if ep.approved_t <= t and ep.end_t > t:
                    self.gate.ok.add((GEN if ep.kind == "gen" else ENG, ep.hk))
        orig = EV.combine_weights
        EV.combine_weights = self.rec.combine
        try:
            out = self.comp.weights(dict(w.serving), snap, {OWNER_UID}, w.burn)
        finally:
            EV.combine_weights = orig
        self.sets.append((t, block, out, self.rec.kwargs))
        return out

    def run(self) -> list:
        t = self.w.t0 + self.w.phase
        end = self.w.t0 + self.w.horizon
        while t < end:
            self.step(t)
            t += self.w.cadence
        return self.sets


# ------------------------------------------------------------------------------------------------ world generator
MODES = [("ok", 50), ("f401", 12), ("ftimeout", 8), ("base_infra", 6), ("base_miner", 4), ("private", 3)]


def pick_mode(rng: random.Random, allow404: bool) -> str:
    items = MODES + ([("f404", 6)] if allow404 else [])
    return rng.choices([m for m, _ in items], [w for _, w in items])[0]


def make_world(seed: int, *, n_hot: Optional[int] = None, allow404: bool = False, scale: float = 0.8,
               horizon_days: float = 9.0, burst: bool = False, t0: Optional[float] = None, n_serv: Optional[int] = None, families: tuple = ("simple", "recommit", "switch", "rereg", "engine", "coldkey", "override"),
               wait_scale: float = 1.0) -> World:
    rng = random.Random(seed)
    t0 = t0 if t0 is not None else ts(2026, 10, 26, 15, 0) + rng.uniform(0, 9 * DAY)
    n = n_hot if n_hot is not None else rng.choice([4, 8, 12, 20, 28, 40])
    cadence = rng.choice([3600.0, 4200.0])
    phase = rng.uniform(0, cadence)
    # serving miners: positive weights, an owner burn entry
    n_serv = n_serv or rng.choice([20, 30, 60, 120])
    burn = rng.choice([0.0, 0.1, 0.3, 0.6])
    raw = {10 + i: rng.lognormvariate(0, 0.8) for i in range(n_serv)}
    tot = sum(raw.values())
    serving = {u: (1 - burn) * v / tot for u, v in raw.items()}
    if burn > 0:
        serving[OWNER_UID] = burn
    incumbent_gen = [(1, ss58(1), ST.CHAMPION, 60.0, t0 - 6 * DAY), (2, ss58(2), ST.FORMER, 58.0, t0 - 9 * DAY),
                     (3, ss58(3), ST.FORMER, 57.0, t0 - 12 * DAY)]
    gstart = CS.GENERALIST_START_UTC.timestamp()
    legacy = [(4, ss58(4), DF, True, gstart - 3 * DAY), (6, ss58(6), DF, False, gstart - 5 * DAY), (8, ss58(8), DF, False, gstart - 6 * DAY),
              (7, ss58(7), RA, True, gstart - 4 * DAY), (9, ss58(9), RA, False, gstart - 6 * DAY)]
    eng_inc = (5, ss58(5), "cd" * 32, block_at(t0 - 5 * DAY))
    hotkeys: list = []
    uid_next = 200
    hk_next = 100
    ov: list = []

    def new_hk(coldkey=None):
        nonlocal hk_next
        hk_next += 1
        return ss58(hk_next), coldkey or f"ck{hk_next}"

    def commit_time():
        if burst:                    # many entries at once: hits the 16-unverified, 25 % and 5 % caps
            return t0 + DAY + rng.uniform(-2.0, 2.0) * H
        return t0 + rng.uniform(-0.5, 3.0) * DAY

    def gen(hk, uid, ck, tc, reg_t, mode=None, prior=()):
        ep = gen_episode(rng, hk, uid, ck, tc, block_at(reg_t), mode or pick_mode(rng, allow404), prior_seen=prior,
                         scale=scale * wait_scale)
        ep.reg_t = reg_t
        return ep

    occupied: list = []

    def eng(hk, uid, ck, tc, reg_t):
        ep = eng_episode(rng, hk, uid, ck, tc, block_at(reg_t), scale=scale * wait_scale, occupied=occupied)
        ep.reg_t = reg_t
        return ep

    reg0 = t0 - 3 * DAY
    for i in range(n):
        fam = rng.choice(families)
        hk, ck = new_hk()
        uid = uid_next
        uid_next += 1
        tc = commit_time()
        hko = Hotkey(hk, ck)
        if fam == "simple":
            hko.episodes.append(gen(hk, uid, ck, tc, reg0))
        elif fam == "engine":
            hko.episodes.append(eng(hk, uid, ck, tc, reg0))
        elif fam == "recommit":
            a = gen(hk, uid, ck, tc, reg0)
            tr = tc + rng.uniform(0.3, 60.0) * H
            if rng.random() < 0.3 and a.first_pass:          # late: after a long wait of the first entry (voided slots)
                tr = rng.uniform(a.first_pass, min(a.decided_t, a.first_pass + 9 * DAY))
            if tr >= a.decided_t:
                tr = a.decided_t - 0.1 * H if a.decided_t - tc > 0.5 * H else tr
            a.end_t, a.end_kind = tr, "changed"
            b = gen(hk, uid, ck, tr + 60.0, reg0, prior=(a.first_seen_ts,))
            if rng.random() < 0.5:                         # same tensors: the new revision of the same weights
                for _, p in b.segs:
                    if p and p.get("hash_named"):
                        p["hash_named"] = next((q["hash_named"] for _, q in a.segs if q and q.get("hash_named")), p["hash_named"])
            hko.episodes += [a, b]
        elif fam == "switch":
            a, b = (gen, eng) if rng.random() < 0.5 else (eng, gen)
            ea = a(hk, uid, ck, tc, reg0)
            tr = tc + rng.uniform(0.3, 60.0) * H
            if tr >= ea.decided_t:
                tr = max(tc + 0.2 * H, ea.decided_t - 0.1 * H)
            ea.end_t, ea.end_kind = tr, "changed"
            eb = b(hk, uid, ck, tr + 60.0, reg0, **({"prior": (ea.first_seen_ts,)} if ea.kind == "gen" and b is gen else {}))
            hko.episodes += [ea, eb]
        elif fam == "rereg":
            a = gen(hk, uid, ck, tc, reg0)
            tp = tc + rng.uniform(1.0, 80.0) * H
            if tp >= a.decided_t:
                tp = max(tc + 0.5 * H, a.decided_t - 0.1 * H)
            a.end_t, a.end_kind = tp, "pruned"
            tr_ = tp + rng.uniform(0.2, 30.0) * H
            nuid = uid_next
            uid_next += 1
            hko.gaps.append((tp, tr_, nuid))
            b = gen(hk, nuid, ck, tr_ + rng.uniform(0.1, 2.0) * H, tr_, prior=(a.first_seen_ts,))
            hko.episodes += [a, b]
        elif fam == "coldkey":
            a = gen(hk, uid, ck, tc, reg0)
            hko.episodes.append(a)
            hotkeys.append(hko)
            hk2, _ = new_hk()
            uid2 = uid_next
            uid_next += 1
            tc2 = tc + rng.uniform(0.1, 20.0) * H
            b = gen(hk2, uid2, ck, tc2, reg0)
            h2 = Hotkey(hk2, ck)
            h2.episodes.append(b)
            hotkeys.append(h2)
            continue
        elif fam == "override":
            a = gen(hk, uid, ck, tc, reg0, mode="private")        # a private repo gets no window: the override carries it
            hko.episodes.append(a)
            ov.append((hk, "waiting", t0 + rng.uniform(1.0, 7.0) * DAY))
        hotkeys.append(hko)
    # SPEC W12 (not changed by the hotfix): a champion dethroned by a newer champion before its approval is a former champion
    # awaiting approval: ranks 2/3 are unpaid without approval and it has no keep-alive any more
    for kind in ("gen", "eng"):
        champs = sorted((e for h in hotkeys for e in h.episodes if e.kind == kind and e.crowned_t < INF), key=lambda e: e.crowned_t)
        for i, e in enumerate(champs):
            nxt = next((f.crowned_t for f in champs[i + 1:] if f.crowned_t < e.approved_t), None)
            if nxt is not None and kind == "gen":
                e.excl.append((nxt, e.approved_t, "E_FORMER_AWAITING_APPROVAL_W12"))
    return World(seed, t0, horizon_days * DAY, cadence, phase, hotkeys, serving, burn, incumbent_gen, legacy, eng_inc, ov[:5], block_at(reg0))


# ------------------------------------------------------------------------------------------------ checks (the properties)
WEIGHT_MIN = 0.0            # P1: weight > 0 (the SPEC.md alert level is 1e-4: tracked as ``min_weight``)
U16 = 65535


def uid_at(w: World, hko: Hotkey, t: float) -> Optional[int]:
    """The uid the hotkey holds at ``t`` (None: not registered / pruned and not yet back)."""
    uid = None
    for ep in hko.episodes:
        if ep.reg_t <= t:
            uid = ep.uid
    if uid is None or any(a <= t < b for a, b, _ in hko.gaps):
        return None
    return uid


def dynamic_exclusions(w: World, t: float) -> dict:
    """Exclusions that depend on the other entries at ``t`` (published caps on the unverified lane): {id(ep): tag}."""
    out: dict = {}
    unver = []
    for hko in w.hotkeys:
        for ep in hko.episodes:
            if ep.kind != "gen" or not (ep.tc <= t < min(ep.decided_t, ep.end_t)):
                continue
            p = ep.patch(t)
            if p is None or t >= ep.tc + SPEC["unverified_h"] * H:
                continue
            if p.get("wait_kind") == ST.WAIT_MINER and not p.get("score"):
                unver.append(ep)
    unver.sort(key=lambda e: (e.cb, e.uid))
    if CS.KA_PROV_TWO_PER_COLDKEY and CS.KA_GAUNTLET_R2:
        # final critic F-A: every coldkey's first entry takes a cap slot before any coldkey's second one. A coldkey's waiting
        # entry outside the cap (checked/our wait, or not seen yet) is its first slot, so its unverified one is a second (#28382)
        counted = {id(e) for e in unver}
        nth: dict = {}
        for hko in w.hotkeys:
            for ep in hko.episodes:
                if ep.kind == "gen" and id(ep) not in counted and ep.tc <= t < min(ep.decided_t, ep.end_t) \
                        and lane(ep, t) == "prov" and not ep.excluded(t):
                    nth[ep.coldkey] = min(1, nth.get(ep.coldkey, 0) + 1)
        slot = {}
        for ep in unver:
            slot[id(ep)] = nth.get(ep.coldkey, 0)
            nth[ep.coldkey] = slot[id(ep)] + 1
        unver.sort(key=lambda e: (slot[id(e)], e.cb, e.uid))
    for ep in unver[SPEC["unverified_max"]:]:
        out[id(ep)] = "E_UNVERIFIED_MAX_16"
    # published (Florian 26 Oct, FAQ 2.16): the abuse clause. A fake (a commitment whose revision answers 404, as long as the state
    # says so) takes the unverified window from every entry of its coldkey that is not ours (not seen yet, repo missing or
    # private, miner-side base problem); two fakes of one coldkey take the provisional keep-alive from all its waiting entries
    fakes: dict = {}
    for hko in w.hotkeys:
        for ep in hko.episodes:
            p = ep.patch(t)
            if ep.kind == "gen" and p is not None and ep.first_seen_ts is not None and ep.first_seen_ts <= t \
                    and p["reason"].startswith(E.WAITING_FOR_REPO) and "404" in p["reason"]:
                fakes[ep.coldkey] = fakes.get(ep.coldkey, 0) + 1
    if fakes:
        for hko in w.hotkeys:
            for ep in hko.episodes:
                if ep.kind != "gen" or not (ep.tc <= t < min(ep.decided_t, ep.end_t)) or not fakes.get(ep.coldkey):
                    continue
                p = ep.patch(t)
                not_ours = (p is None or (p["status"] == ST.QUEUED and p["score"] is None and not p["hash_named"]
                                          and p["reason"].startswith(E.WAITING_FOR_REPO) and p["wait_kind"] != ST.WAIT_INFRA)
                            or (p["status"] == ST.PENDING_VERIFY and p["wait_kind"] != ST.WAIT_INFRA))
                if not_ours or (fakes[ep.coldkey] >= 2 and lane(ep, t) == "prov"):
                    out.setdefault(id(ep), "E_ABUSE_404")
    # Published: two provisional entries per coldkey/track; rollback keeps one. Above the limit the
    # group check below verifies the number covered without predicting the exposure tie-break.
    by_ck: dict = {}
    for hko in w.hotkeys:
        for ep in hko.episodes:
            if ep.kind == "gen" and ep.tc <= t < min(ep.decided_t, ep.end_t) and lane(ep, t) == "prov":
                by_ck.setdefault(ep.coldkey, []).append(ep)
    for ck, eps in by_ck.items():
        limit = 2 if CS.KA_PROV_TWO_PER_COLDKEY else 1
        if len(eps) > limit:
            for ep in eps:
                out.setdefault(id(ep), "E_COLDKEY_LIMIT")
    return out


def coldkey_groups(w: World, t: float) -> list:
    """Coldkeys over the provisional limit at ``t`` (without their own exclusion): the group must fill its slots."""
    by_ck: dict = {}
    dyn = dynamic_exclusions(w, t)
    for hko in w.hotkeys:
        for ep in hko.episodes:
            if ep.kind == "gen" and ep.tc <= t < min(ep.decided_t, ep.end_t) and lane(ep, t) == "prov" and not ep.excluded(t) \
                    and dyn.get(id(ep)) != "E_ABUSE_404":
                by_ck.setdefault(ep.coldkey, []).append(ep)
    limit = 2 if CS.KA_PROV_TWO_PER_COLDKEY else 1
    return [eps for eps in by_ck.values() if len(eps) > limit]


def lane(ep: Episode, t: float) -> str:
    """Which keep-alive lane the generalist entry is in at ``t``: "prov" (waits for its first draw) or "ka" (pending/champion)."""
    p = ep.patch(t)
    if ep.kind != "gen":
        return ""
    if p is None:
        return "prov"
    if p["status"] in (ST.PENDING_CONFIRM, ST.CHAMPION):
        return "ka"
    if p["status"] in (ST.QUEUED, ST.PENDING_VERIFY) and (p["score"] is None or p["close_draws"]):
        return "prov"
    return ""


@dataclass
class Gap:
    ep: Episode
    t: float
    weight: float
    label: str


@dataclass
class Report:
    gaps: list = field(default_factory=list)            # P1 violations
    sets: int = 0
    checked: int = 0                                     # (episode, set) pairs with P1 required
    excluded: dict = field(default_factory=dict)         # tag -> (episode, set) pairs the oracle excluded
    min_weight: float = INF
    cap_violations: list = field(default_factory=list)   # P2
    p3: dict = field(default_factory=dict)                # label -> max zero-incentive span in blocks
    p3_violations: list = field(default_factory=list)
    override_gaps: list = field(default_factory=list)
    max_wait_set: int = 0


def check_world(w: World, sets: list) -> Report:
    rep = Report(sets=len(sets))
    req_sets: list = []
    for t, block, wts, kw in sets:
        dyn = dynamic_exclusions(w, t)
        req_now: set = set()
        req_sets.append(req_now)
        if abs(math.fsum(wts.values()) - 1.0) > 1e-9 or any((not math.isfinite(v)) or v < 0 for v in wts.values()):
            rep.cap_violations.append((t, "weights do not sum to 1 / negative / not finite"))
        _check_caps(w, t, wts, kw, rep)
        for hko in w.hotkeys:
            uid = uid_at(w, hko, t)
            for ep in hko.episodes:
                if uid is None or ep.uid != uid or not ep.required(t):
                    continue
                tag = ep.excluded(t) or dyn.get(id(ep))
                if tag:
                    rep.excluded[tag] = rep.excluded.get(tag, 0) + 1
                    continue
                v = wts.get(uid, 0.0)
                rep.checked += 1
                req_now.add(id(ep))
                rep.min_weight = min(rep.min_weight, v)
                if not v > WEIGHT_MIN:
                    rep.gaps.append(Gap(ep, t, v, ep.label))
        for eps in coldkey_groups(w, t):
            limit = 2 if CS.KA_PROV_TWO_PER_COLDKEY else 1
            if sum(wts.get(e.uid, 0.0) > 0 for e in eps) < limit:
                rep.gaps.append(Gap(eps[0], t, 0.0, eps[0].label + "+coldkey-group"))
        for hk, kind, until in w.override:
            hko = next((h for h in w.hotkeys if h.hk == hk), None)
            uid = uid_at(w, hko, t) if hko else None
            if uid is not None and t < until and not wts.get(uid, 0.0) > 0:
                rep.override_gaps.append((hk, t, until))
    _check_chain(w, sets, rep, req_sets)
    return rep


def _check_caps(w: World, t: float, wts: dict, kw: dict, rep: Report) -> None:
    """P2: the shares stay inside the published caps. Pure entrants (no serving weight) are paid exactly what ``combine_weights``
    gives them, so their weight is their share of the miner part."""
    prof = CS.profile_for(CS.SPLIT_PROFILE, datetime.fromtimestamp(t, timezone.utc))
    miner = 1.0 - (w.burn if OWNER_UID in w.serving else 0.0)
    slot = prof.tracks[0] if prof.tracks else 0.0
    eps = 1e-9
    ka = [u for u, _ in (kw.get("keepalive") or {}).get(GEN, ())]
    pv = list((kw.get("provisional") or {}).get(GEN, ()))
    eng_wait = list(kw.get("engine_waiting") or ())
    legacy = list(kw.get("legacy_keepalive") or ())
    serving_uids = set(w.serving)
    pure = lambda us: [u for u in us if u not in serving_uids and u != OWNER_UID]  # noqa: E731
    ranked = {u for u, _ in ((kw.get("tracks") or {}).get(GEN, ()) or ())}
    gate = kw.get("award_ok")
    rank_paid = {u for u in ranked if gate is None or gate(GEN, u)}      # paid by its rank, not by the keep-alive
    prov_only = [u for u in pure(pv) if u not in ka and u not in rank_paid]
    ka_only = [u for u in pure(ka) if u not in rank_paid]
    s_prov = math.fsum(wts.get(u, 0.0) for u in prov_only)
    s_ka = math.fsum(wts.get(u, 0.0) for u in ka_only)
    if s_prov > SPEC["share_prov"] * slot * miner + eps:
        rep.cap_violations.append((t, f"provisional {s_prov:.6f} > 5% of the track share {SPEC["share_prov"] * slot * miner:.6f}"))
    if s_ka + s_prov > SPEC["share_ka"] * slot * miner + eps:
        rep.cap_violations.append((t, f"generalist keep-alive+provisional {s_ka + s_prov:.6f} > 25% of the track share"))
    eng_paid = {kw.get("engine_champion")} | set(kw.get("engine_cochampions") or {})
    ew = [u for u in pure(eng_wait) if u not in ka and u not in pv and u not in eng_paid and u not in rank_paid]
    s_ew = math.fsum(wts.get(u, 0.0) for u in ew)
    if s_ew > SPEC["share_engine_ka"] * prof.engine * miner + eps:
        rep.cap_violations.append((t, f"engine keep-alive {s_ew:.6f} > 5% of the engine share"))
    lg = [u for u in pure(legacy) if u not in ka and u not in pv and u not in eng_wait]
    s_lg = math.fsum(wts.get(u, 0.0) for u in lg)
    if s_lg > SPEC["share_legacy"] * miner + eps:
        rep.cap_violations.append((t, f"legacy keep-alive {s_lg:.6f} > 3% of the miner part"))
    if wts.get(OWNER_UID, 0.0) + eps < (w.burn if OWNER_UID in w.serving else 0.0) * 0.999999:
        rep.cap_violations.append((t, "owner burn entry shrank"))


def u16(weights: dict, uid: int) -> int:
    top = max(weights.values()) if weights else 0.0
    return int(round(U16 * weights.get(uid, 0.0) / top)) if top > 0 else 0


def _check_chain(w: World, sets: list, rep: Report, req_sets: list) -> None:
    """P3: commit-reveal. A set at block b counts at the 2nd epoch boundary after b; the incentive of a uid at a boundary is
    > 0 iff the newest counting set gives it a non-zero u16 weight (max-normalised) and was set after the uid registered.
    Required from the commitment + 3 epochs until decided/ended, outside the oracle's exclusions."""
    if not sets:
        return
    eff = [(effective_block(b), b, wts, i) for i, (_, b, wts, _) in enumerate(sets)]
    first_b, last_b = sets[0][1], sets[-1][1]
    for hko in w.hotkeys:
        for ep in hko.episodes:
            lo = ep.cb + 3 * EPOCH
            hi_t = min(ep.decided_t, ep.end_t)
            hi = block_at(hi_t) if hi_t < INF else INF
            E0 = next_boundary(lo - 1)
            span = best = 0
            E_ = E0
            last_eff_block = eff[-1][0]
            while E_ <= last_eff_block and E_ <= hi:
                tE = time_of(E_)
                uidE = uid_at(w, hko, tE)
                excl = ep.excluded(tE)
                active = None
                for eb, b, wts, i in eff:
                    if eb <= E_:
                        active = (b, wts, i)
                    else:
                        break
                # the incentive at E comes from the newest counting set: owed only if THAT set owed the uid a weight (P1)
                if uidE is None or uidE != ep.uid or excl or active is None or id(ep) not in req_sets[active[2]]:
                    span = 0
                else:
                    b, wts, _i = active
                    reg = block_at(ep.reg_t)
                    ok = u16(wts, ep.uid) > 0 and b > reg
                    if ok:
                        span = 0
                    else:
                        span += 1
                        best = max(best, span)
                        if span == 1:
                            first_zero = E_
                E_ += EPOCH
            key = ep.label
            rep.p3[key] = max(rep.p3.get(key, 0), best * EPOCH)
            if best:
                rep.p3_violations.append((ep, best * EPOCH))


def state_tag(w: World, ep: Episode, t: float) -> str:
    """A short description of what the validator state says about the episode at ``t`` (for grouping counterexamples)."""
    p = ep.patch(t)
    prior = any(e is not ep and e.tc < ep.tc for h in w.hotkeys if h.hk == ep.hk for e in h.episodes)
    pre = "recommit/" if prior else ""
    if p is None:
        return pre + "unseen"
    if ep.kind == "eng":
        if p["result"] is None:
            return pre + ("eng-unmeasured" + ("" if p["owner"] else "-unbound"))
        if p["challenger"] is not None:
            return pre + ("eng-confirmed-waiting-slot" if p["challenger"].get("slot_utc") else "eng-pending-challenger")
        if p["champion"]:
            return pre + "eng-champion-awaiting-approval"
        return pre + "eng-other"
    if p["status"] == ST.PENDING_CONFIRM:
        return pre + ("pending-confirmed-waiting-slot" if any(h.get("kind") == "crown_pending" for h in p["history"])
                      else "pending-confirm")
    if p["status"] == ST.CHAMPION:
        return pre + "champion-awaiting-approval"
    if p["status"] == ST.PENDING_VERIFY:
        return pre + f"pending-verify/{p['wait_kind']}"
    if p["reason"]:
        return pre + f"fetch-failed/{p['wait_kind']}/{p['reason'][22:32]}"
    return pre + ("queued-fetched" if p["hash_named"] else "queued-unfetched")


# ------------------------------------------------------------------------------------------------ the gauntlet-round-2 switch
# The proposed fixes F1/F2/F3/F5 of the first property run are code now, behind one switch. ``apply_fixes(())`` turns it off
# (with KA_BIG_UNDECIDED left on: the 233dbd2 behaviour of the keep-alive windows), any of the names turns it on. F4 (the abuse
# clause) is policy: it stays on and is an exclusion of the oracle (``dynamic_exclusions``, E_ABUSE_404).
UNSEEN_GRACE_H = 6.0
FIXES = ("F1", "F2", "F3", "F5")


def apply_fixes(names, **extra) -> list:
    """Set ``KA_GAUNTLET_R2`` (on when any fix name is given); ``extra`` = more module switches of ``competition_split``.
    Returns the undo list for ``undo_fixes``."""
    undo = []

    def setattr_(mod, name, val):
        undo.append((mod, name, getattr(mod, name)))
        setattr(mod, name, val)
    setattr_(CS, "KA_GAUNTLET_R2", bool(set(names) & set(FIXES)))
    for k, v in extra.items():
        setattr_(CS, k, v)
    return undo


def undo_fixes(undo: list) -> None:
    for mod, name, val in reversed(undo):
        setattr(mod, name, val)


# ------------------------------------------------------------------------------------------------ batch runner (process pool)
def run_one(args: tuple) -> dict:
    """Run one generated world through the real code and check it. ``args`` = (seed, make_world kwargs, fix names, slots)."""
    import tempfile
    seed, kw, fixes, slots = args[:4]
    extra = args[4] if len(args) > 4 else {}
    saved = (CS.CROWN_SLOTS_UTC, CS.ENGINE_CROWN_SLOTS_UTC, dict(CS.TRACK_ENTRY_FLOOR))
    if slots is not None:        # the child may be a fresh process: restore the release slot calendars (conftest clears them)
        CS.CROWN_SLOTS_UTC, CS.ENGINE_CROWN_SLOTS_UTC = slots
    for k in list(CS.TRACK_ENTRY_FLOOR):
        if k != "generalist":
            CS.TRACK_ENTRY_FLOOR[k] = None
    undo = apply_fixes(fixes, **extra)
    try:
        w = make_world(seed, **kw)
        with tempfile.TemporaryDirectory() as d:
            sets = Runner(w, d).run()
        rep = check_world(w, sets)
    finally:
        undo_fixes(undo)
        CS.CROWN_SLOTS_UTC, CS.ENGINE_CROWN_SLOTS_UTC = saved[0], saved[1]
        CS.TRACK_ENTRY_FLOOR.clear()
        CS.TRACK_ENTRY_FLOOR.update(saved[2])
    gaps = [(g.ep.uid, g.label, round((g.t - g.ep.tc) / H, 2), state_tag(w, g.ep, g.t), g.weight) for g in rep.gaps]
    return {"seed": seed, "episodes": sum(len(h.episodes) for h in w.hotkeys), "hotkeys": len(w.hotkeys), "sets": rep.sets,
            "checked": rep.checked, "gaps": gaps, "cap": rep.cap_violations[:5], "override_gaps": rep.override_gaps[:5],
            "p3": rep.p3, "p3_viol": [(ep.uid, ep.label, span) for ep, span in rep.p3_violations],
            "excluded": rep.excluded, "min_weight": rep.min_weight if rep.min_weight < INF else None,
            "max_simul": max_simultaneous(w, sets), "ka_len": max((len((s[3].get("keepalive") or {}).get(GEN, ())) for s in sets), default=0)}


def max_simultaneous(w: World, sets: list) -> int:
    best = 0
    for t, *_ in sets[::6]:
        n = sum(1 for h in w.hotkeys for e in h.episodes if e.tc <= t < min(e.decided_t, e.end_t))
        best = max(best, n)
    return best


def run_batch(jobs: list, workers: Optional[int] = None) -> list:
    """Run ``jobs`` (tuples for ``run_one``) in a process pool (fork); sequentially when there is one CPU."""
    import concurrent.futures as cf
    workers = workers or min(8, os.cpu_count() or 1)
    if workers <= 1 or len(jobs) < 4:
        return [run_one(j) for j in jobs]
    ctx = __import__("multiprocessing").get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(run_one, jobs, chunksize=max(1, len(jobs) // (workers * 4))))


# ------------------------------------------------------------------------------------------------ hand-built worlds (minimal reproductions)
def hand_world(hotkeys: list, t0: float, hours: float, *, burn: float = 0.0, n_serv: int = 40, cadence: float = 3600.0) -> World:
    """A world with exactly the given hotkeys (no random ones), the default incumbents, and a set every ``cadence`` s."""
    serving = {10 + i: (1 - burn) / n_serv for i in range(n_serv)}
    if burn > 0:
        serving[OWNER_UID] = burn
    gstart = CS.GENERALIST_START_UTC.timestamp()
    return World(0, t0, hours * H, cadence, 0.0, hotkeys, serving, burn,
                 [(1, ss58(1), ST.CHAMPION, 60.0, t0 - 6 * DAY), (2, ss58(2), ST.FORMER, 58.0, t0 - 9 * DAY)], [],
                 (5, ss58(5), "cd" * 32, block_at(t0 - 5 * DAY)), [], block_at(t0 - 3 * DAY))


def hand_gen(hk: str, uid: int, ck: str, tc: float, segs: list, *, decided: float = INF, end: float = INF, end_kind: str = "",
             label: str = "hand", first_seen: Optional[float] = None, excl: Optional[list] = None) -> Episode:
    ep = Episode("gen", hk, uid, ck, tc, block_at(tc - 3 * DAY), label=label, repo=f"m/{hk[2:7]}-{int(tc) % 100000}",
                 decided_t=decided, end_t=end, end_kind=end_kind, reg_t=tc - 3 * DAY, excl=list(excl or []))
    ep.first_seen_ts = first_seen if first_seen is not None else tc
    ep.segs = [(tc, None)] + list(segs)
    return ep


def pending_patch(hk: str, fp: float, hash_named: Optional[str] = None) -> dict:
    return {**_sub_defaults(), **_scored(True), "status": ST.PENDING_CONFIRM, "hash_named": hash_named or "h" + hk[2:7],
            "delta_checked": True, "first_pass_ts": fp, "confirm_after_ts": fp + 12 * H, "draw1": "d1", "first_gain": 3.0}


def queued_patch(hk: str, hash_named: Optional[str] = None, fetched: bool = True) -> dict:
    return {**_sub_defaults(), "hash_named": (hash_named or "h" + hk[2:7]) if fetched else "", "delta_checked": fetched}


def weights_of(w: World, tmp: str) -> list:
    return Runner(w, tmp).run()
