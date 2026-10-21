"""Dual role (docs/DESIGN-MULTIROLE.md section 5): an inference hotkey with one competition entry, keep-alive top-up and
slot rule in ``combine_weights``; the competition-side readers of the commitment fields get the competition view."""

from __future__ import annotations

import math
import random

from typing import Callable, Mapping, Optional, Sequence

import pytest

from reax_subnet import competition_split as cs
from reax_subnet.competition_split import (
    KEEPALIVE_MAX_ENTRIES, PROFILES, PROVISIONAL_KA_FRACTION, PROVISIONAL_KA_MAX_SHARE, KEEPALIVE_MAX_SHARE,
    PROVISIONAL_KA, RankedLeaders, combine_weights, keepalive_uids, keep_share_ranks, track_payout,
    SPLIT_PROFILE, TRACK_ENTRY_FLOOR, TRACK_RANK_SPLIT, log, profile_for,
)

DF, RA = "demand_forecasting", "risk_assessment"
OWNER = 0


@pytest.fixture(autouse=True)
def _floors_off(monkeypatch):
    for k in list(TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(TRACK_ENTRY_FLOOR, k, None)


# ---- frozen copy of combine_weights before the dual-role release (base 592d5f2) ----
def _combine_weights_v1(serving: Mapping[int, float], engine_champion: Optional[int], engine_share_frac: float,
                    tracks: Mapping[str, Mapping[int, float] | Sequence[tuple[int, float]]],
                    profile: str = SPLIT_PROFILE, *, owner_uid: Optional[int] = None,
                    burn_fraction: float = 0.0,
                    floors: Mapping[str, Optional[float]] = TRACK_ENTRY_FLOOR,
                    award_ok: Optional[Callable[[str, int], bool]] = None,
                    keepalive: Optional[Mapping[str, Sequence[tuple[int, float]]]] = None,
                    keepalive_share: Optional[float] = None,
                    provisional: Optional[Mapping[str, Sequence[int]]] = None,
                    keep_share: bool = False, now: Optional[float] = None) -> dict[int, float]:
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

    ``provisional``: per active track, the uids of the entries that wait for their first evaluation
    (``compute_provisional``; ignored with ``PROVISIONAL_KA`` False). Each is paid ``PROVISIONAL_KA_FRACTION`` x the same
    median as the keep-alive entries; a track's provisional total is at most ``PROVISIONAL_KA_MAX_SHARE`` x the track's
    share (then split evenly), taken from the track's share before the ranks, in addition to the keep-alive. It does not
    use a keep-alive slot (``KEEPALIVE_MAX_ENTRIES``); the owner UID, the engine champion and a uid that holds a
    keep-alive slot or a rank of any track are left out. Unclaimed: stays with ``serving``.

    The competition fractions apply to the miner part (the owner's burn stays as it is, as in 1.1). Everything not
    claimed stays with ``serving``. The result is normalised to sum 1; an empty ``serving`` returns ``{}``.
    """
    prof = profile_for(profile)
    base_total = math.fsum(w for w in serving.values() if w > 0)
    if base_total <= 0:
        return {}
    base = {uid: w / base_total for uid, w in serving.items() if w > 0}
    burn = burn_fraction if owner_uid is not None and owner_uid in base else 0.0
    miner_part = 1.0 - burn

    paid: dict[int, float] = {}  # fractions of the miner part
    if engine_champion is not None and engine_share_frac > 0:
        paid[engine_champion] = prof.engine * min(1.0, engine_share_frac)
    ka_cap = KEEPALIVE_MAX_SHARE if keepalive_share is None else keepalive_share
    alive: dict[str, list[int]] = {}
    if ka_cap > 0:
        for track_id, entries in (keepalive or {}).items():
            uids = keepalive_uids(entries, floors.get(track_id), owner_uid)
            if award_ok is not None and uids:
                # an approved entry is paid by its rank: it does not also hold a keep-alive slot
                ranked = track_payout(tracks.get(track_id, ()), floors.get(track_id))
                uids = [uid for uid in uids if not (uid in ranked and award_ok(track_id, uid))]
            if len(uids) > KEEPALIVE_MAX_ENTRIES:
                # ``keepalive`` arrives in rank order (compute_keepalive): entries outside the top ones get nothing
                log.info("track %s: keep-alive for %d entries only, %d qualifying entries get none: uids %s",
                         track_id, KEEPALIVE_MAX_ENTRIES, len(uids) - KEEPALIVE_MAX_ENTRIES,
                         uids[KEEPALIVE_MAX_ENTRIES:])
                uids = uids[:KEEPALIVE_MAX_ENTRIES]
            if uids:
                alive[track_id] = uids
    prov: dict[str, list[int]] = {}
    if PROVISIONAL_KA and ka_cap > 0:
        # keepalive_share=0 turns the provisional slots off too
        taken = {u for ids in alive.values() for u in ids} | set(paid)
        for track_id in tracks:
            taken.update(track_payout(tracks[track_id], floors.get(track_id)))
        for track_id, uids in (provisional or {}).items():
            ids = []
            for uid in uids:
                if uid != owner_uid and uid not in taken and uid not in ids:
                    ids.append(uid)
            if ids:
                prov[track_id] = ids
    slots = list(zip(sorted(set(tracks) | set(alive) | set(prov)), prof.tracks))  # (track id, slot share), as paid
    alive = {t: alive[t] for t, _ in slots if t in alive}  # a track without a slot pays nothing
    prov = {t: prov[t] for t, _ in slots if t in prov}
    per_entry = math.inf
    if alive or prov:
        competing = {u for ids in alive.values() for u in ids} | {u for ids in prov.values() for u in ids} | set(paid)
        for track_id, _ in slots:
            competing.update(track_payout(tracks.get(track_id, ()), floors.get(track_id)))
        competing.add(owner_uid)
        typical = sorted(w for uid, w in base.items() if uid not in competing)
        if typical:
            mid = len(typical) // 2
            median = typical[mid] if len(typical) % 2 else math.fsum((typical[mid - 1], typical[mid])) / 2
            per_entry = median / miner_part
    for track_id, slot_share in slots:
        rank_share = slot_share
        if track_id in alive:
            each = min(per_entry, ka_cap * slot_share / len(alive[track_id]))
            for uid in alive[track_id]:
                paid[uid] = paid.get(uid, 0.0) + each
            rank_share = slot_share - each * len(alive[track_id])
        if track_id in prov:
            n_prov = len(prov[track_id])
            each_prov = min(PROVISIONAL_KA_FRACTION * per_entry, PROVISIONAL_KA_MAX_SHARE * slot_share / n_prov)
            for uid in prov[track_id]:
                paid[uid] = paid.get(uid, 0.0) + each_prov
            rank_share -= each_prov * n_prov
            log.info("track %s: provisional keep-alive for uids %s, each %.5f of the miner part",
                     track_id, prov[track_id], each_prov)
        ranked_pay = track_payout(tracks.get(track_id, ()), floors.get(track_id))
        if keep_share and award_ok is not None and now is not None:
            ranked_pay = keep_share_ranks(ranked_pay, getattr(tracks.get(track_id), "crowned", None), now,
                                          lambda u, t=track_id: award_ok(t, u))
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


# ---------------------------------------------------------------- helpers

def _serving(dual: dict[int, float] | None = None, n_big: int = 0, big: float = 0.006):
    """20 typical miners at 0.004 and one heavy miner, so the median of the typical uids is 0.004; ``dual``: extra uids with
    the given serving weight; ``n_big`` hotkeys (uids 300..) with weight ``big``. Raw weights sum to 1."""
    w = {i: 0.004 for i in range(1, 21)}
    w.update(dual or {})
    w.update({300 + i: big for i in range(n_big)})
    w[200] = 1.0 - math.fsum(w.values())
    return w


def _expect(base: dict[int, float], paid: dict[int, float]) -> dict[int, float]:
    """Serving scaled by 1 - claimed, plus the pay (a base summing to 1, no owner burn)."""
    keep = 1.0 - math.fsum(paid.values())
    out = {u: w * keep for u, w in base.items()}
    for u, f in paid.items():
        out[u] = out.get(u, 0.0) + f
    return out


def _same(got: dict[int, float], want: dict[int, float]):
    assert set(got) == set(want)
    for u in want:
        assert got[u] == pytest.approx(want[u], abs=1e-12), u


# ---------------------------------------------------------------- the worked example of the design

def test_worked_example_pure_dual_and_big_server():
    base = _serving({100: 0.0015, 101: 0.006})
    assert math.fsum(base.values()) == pytest.approx(1.0)
    ka = {DF: [(50, 30.0), (100, 29.0), (101, 28.0)]}
    w = combine_weights(base, None, 0.0, {}, keepalive=ka)
    _same(w, _expect(base, {50: 0.004, 100: 0.0025}))  # the big server (0.006 >= 0.004) is paid nothing and has no slot
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_big_server_takes_no_slot_so_the_cap_is_shared_by_the_paying_entries_only():
    base = _serving({101: 0.006})
    ka = {DF: [(50, 30.0), (101, 29.0)]}
    with_big = combine_weights(base, None, 0.0, {}, keepalive=ka)
    without = combine_weights(base, None, 0.0, {}, keepalive={DF: [(50, 30.0)]})
    assert with_big == without


def test_no_typical_serving_uid_the_cap_decides():
    # the only serving uid is itself a candidate: no typical weight, per_entry is unbounded, each = cap x share / n.
    # Corner case: the serving uid still holds a slot (its own weight is below the unbounded per_entry) and pays 0.
    base = {7: 1.0}
    w = combine_weights(base, None, 0.0, {}, keepalive={DF: [(50, 30.0), (7, 29.0)]})
    each = KEEPALIVE_MAX_SHARE * PROFILES["50/50"].tracks[0] / 2
    _same(w, _expect(base, {50: each}))


def test_top_up_when_the_cap_binds():
    base = {7: 0.01, 8: 0.99}
    # 8 pure entrants + one dual (uid 7, own 0.01): n = 9, each = 0.25 x 0.15 / 9 = 0.0041667 < per_entry (inf: 7 and 8 compete)
    pure = [(50 + i, 30.0 - i) for i in range(8)]
    # uid 8 is not a candidate, so the median is 0.99 (only typical); the cap decides
    w = combine_weights(base, None, 0.0, {}, keepalive={DF: pure + [(7, 20.0)]})
    # pure entrants come first: the 8 slots go to them, the dual-role uid is cut
    each = KEEPALIVE_MAX_SHARE * 0.15 / 8
    _same(w, _expect(base, {50 + i: each for i in range(8)}))


# ---------------------------------------------------------------- slots

def test_crowd_out_attempt_by_big_serving_hotkeys_fails():
    base = _serving(n_big=10)
    big = [(300 + i, 40.0 - i * 0.1) for i in range(10)]  # best scores, first in the evaluator's order
    pure = [(50, 20.0), (51, 19.0), (52, 18.0)]
    w = combine_weights(base, None, 0.0, {}, keepalive={DF: big + pure})
    _same(w, _expect(base, {50: 0.004, 51: 0.004, 52: 0.004}))
    only_pure = combine_weights(base, None, 0.0, {}, keepalive={DF: pure})
    assert w == only_pure


def test_pure_entrants_come_first_then_dual_role_in_the_given_order_cut_to_eight():
    dual = {100 + i: 0.001 for i in range(10)}
    base = _serving(dual)
    ka = [(100 + i, 40.0 - i) for i in range(10)] + [(50, 5.0), (51, 4.0), (52, 3.0)]
    w = combine_weights(base, None, 0.0, {}, keepalive={DF: ka})
    paid = {50: 0.004, 51: 0.004, 52: 0.004}
    paid.update({100 + i: 0.003 for i in range(5)})  # 3 pure + the first 5 dual-role entrants = 8 slots
    _same(w, _expect(base, paid))
    for i in range(5, 10):
        assert w[100 + i] == pytest.approx(base[100 + i] * (1.0 - math.fsum(paid.values())), abs=1e-15)


def test_rank_share_is_reduced_by_the_top_ups_actually_paid():
    base = _serving({100: 0.0015, 101: 0.006})
    ka = {DF: [(50, 30.0), (100, 29.0), (101, 28.0)]}
    tracks = {DF: [(60, 40.0)]}
    w = combine_weights(base, None, 0.0, tracks, keepalive=ka)
    topups = 0.004 + 0.0025
    rank1 = (0.15 - topups) * TRACK_RANK_SPLIT[0]
    _same(w, _expect(base, {50: 0.004, 100: 0.0025, 60: rank1}))
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_approved_ranked_entry_gets_no_keepalive_slot():
    base = _serving({100: 0.001})
    tracks = {DF: [(100, 50.0)]}
    ka = {DF: [(100, 50.0), (50, 30.0)]}
    w = combine_weights(base, None, 0.0, tracks, keepalive=ka, award_ok=lambda t, u: True)
    paid_rank = (0.15 - 0.004) * TRACK_RANK_SPLIT[0]
    _same(w, _expect(base, {50: 0.004, 100: paid_rank}))
    # unapproved: the rank is not paid and the entry keeps its keep-alive slot (top-up 0.003)
    w2 = combine_weights(base, None, 0.0, tracks, keepalive=ka, award_ok=lambda t, u: u != 100)
    _same(w2, _expect(base, {50: 0.004, 100: 0.003}))


# ---------------------------------------------------------------- median

def test_median_ignores_every_keepalive_candidate_even_when_cut_or_ineligible():
    typical_only = _serving()
    pure_pay = combine_weights(typical_only, None, 0.0, {}, keepalive={DF: [(50, 30.0)]})[50]
    assert pure_pay == pytest.approx(0.004)
    # 15 dual-role uids with a tiny serving weight (8 get a slot, 7 are cut) would pull the median down if counted
    dual = {100 + i: 0.0001 for i in range(15)}
    base = _serving(dual)
    ka = [(50, 30.0)] + [(100 + i, 20.0 - i * 0.1) for i in range(15)]
    w = combine_weights(base, None, 0.0, {}, keepalive={DF: ka})
    typical = sorted(base[u] for u in base if u < 100 or u == 200)
    median = typical[len(typical) // 2]  # 21 typical uids
    assert median == pytest.approx(0.004)
    each = min(median, KEEPALIVE_MAX_SHARE * 0.15 / 8)
    assert w[50] == pytest.approx(each, abs=1e-12) or w[50] > 0
    _same(w, _expect(base, {50: each, **{100 + i: each - 0.0001 for i in range(7)}}))


def test_median_ignores_provisional_ranked_and_champion_uids():
    base = _serving({400: 0.0001, 401: 0.0001, 402: 0.0001, 403: 0.0001})
    w = combine_weights(base, 403, 1.0, {DF: [(402, 40.0)]}, provisional={DF: [400, 401]}, keepalive={DF: [(50, 30.0)]})
    # all four small uids are competition-paid and left out of the median: per_entry stays 0.004
    # serving provisional candidates get no provisional pay (critic round 1 F9) but still stay out of the median
    claimed_engine = 0.20
    assert w[50] > 0
    ka_each = 0.004
    rank_share = 0.15 - ka_each
    paid = {50: ka_each, 402: rank_share * 0.75, 403: claimed_engine}
    _same(w, _expect(base, paid))


# ---------------------------------------------------------------- provisional

def test_provisional_goes_to_pure_entrants_only():
    """Critic round 1 F9: provisional pay protects an unserved entry from pruning; a serving hotkey is not at risk."""
    base = _serving({100: 0.001, 101: 0.003})
    w = combine_weights(base, None, 0.0, {}, provisional={DF: [50, 100, 101]})
    each = min(PROVISIONAL_KA_FRACTION * 0.004, PROVISIONAL_KA_MAX_SHARE * 0.15 / 1)  # n_prov = 1 (50 only)
    assert each == pytest.approx(0.002)
    _same(w, _expect(base, {50: each}))


def test_provisional_cap_is_shared_by_the_counted_entries_only():
    base = _serving({100: 0.01})
    w_a = combine_weights(base, None, 0.0, {}, provisional={DF: [50, 100]})
    w_b = combine_weights(base, None, 0.0, {}, provisional={DF: [50]})
    assert w_a == w_b


def test_provisional_uid_with_a_keepalive_slot_is_left_out():
    base = _serving({100: 0.001})
    w = combine_weights(base, None, 0.0, {}, keepalive={DF: [(100, 30.0)]}, provisional={DF: [100]})
    _same(w, _expect(base, {100: 0.003}))


# ---------------------------------------------------------------- owner and totals

def test_owner_is_never_paid_and_totals_sum_to_one():
    base = _serving({100: 0.001})
    base_owner = {**{u: w * 0.95 for u, w in base.items()}, OWNER: 0.05}
    ka = {DF: [(OWNER, 99.0), (100, 30.0), (50, 20.0)]}
    with_owner = combine_weights(base_owner, None, 0.0, {DF: [(OWNER, 90.0)]}, owner_uid=OWNER, burn_fraction=0.05,
                                 keepalive=ka, provisional={RA: [OWNER]})
    without = combine_weights(base_owner, None, 0.0, {}, owner_uid=OWNER, burn_fraction=0.05,
                              keepalive={DF: [(100, 30.0), (50, 20.0)]})
    assert math.fsum(with_owner.values()) == pytest.approx(1.0)
    assert with_owner[OWNER] >= 0.05 - 1e-12
    assert with_owner[50] > 0 and with_owner[100] > base_owner[100] * 0.99
    assert math.fsum(without.values()) == pytest.approx(1.0)


def test_totals_sum_to_one_over_random_dual_role_inputs():
    rng = random.Random(7)
    for _ in range(300):
        base = {u: rng.random() + 1e-3 for u in range(1, rng.randint(2, 30))}
        cands = rng.sample(range(1, 60), rng.randint(0, 14))
        ka = [(u, rng.uniform(1, 50)) for u in cands]
        prov = rng.sample(range(1, 60), rng.randint(0, 5))
        tr = {DF: [(u, rng.uniform(1, 50)) for u in rng.sample(range(1, 60), rng.randint(0, 4))]}
        w = combine_weights(base, rng.choice([None, 3, 45]), 1.0, tr, keepalive={DF: ka}, provisional={RA: prov})
        assert math.fsum(w.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in w.values())


# ---------------------------------------------------------------- bit-identical to the frozen v1 without dual-role uids

def test_bit_identical_to_the_previous_function_without_dual_role_uids():
    rng = random.Random(20261020)
    for it in range(3000):
        owner = rng.choice([None, 0])
        serving = {u: rng.choice([rng.random(), rng.random() * 1e-3, 1.0]) for u in range(0, rng.randint(0, 40))
                   if rng.random() < 0.9}
        if owner is not None and rng.random() < 0.8:
            serving[owner] = rng.random() * 0.1
        pool = [u for u in range(100, 140)]  # entrants never serve: not in ``serving``
        tracks = {}
        for tid in rng.sample([DF, RA, "x_track"], rng.randint(0, 3)):
            # ranked entrants are pure too (a ranked uid that also serves is a dual-role uid: not this test's premise)
            ents = [(rng.choice(pool), rng.uniform(0, 60)) for _ in range(rng.randint(0, 5))]
            tracks[tid] = RankedLeaders(ents) if rng.random() < 0.5 else ents
        champ = rng.choice([None, None, rng.choice(pool), rng.choice(list(serving) or [1])])
        ka = {tid: [(rng.choice(pool + [owner if owner is not None else 101]), rng.uniform(-1, 60))
                    for _ in range(rng.randint(0, 14))]
              for tid in rng.sample([DF, RA, "x_track"], rng.randint(0, 3))}
        prov = {tid: [rng.choice(pool) for _ in range(rng.randint(0, 5))]
                for tid in rng.sample([DF, RA, "x_track"], rng.randint(0, 3))}
        kw = dict(owner_uid=owner, burn_fraction=rng.choice([0.0, 0.05, 0.3]),
                  floors={DF: rng.choice([None, 20.0]), RA: None, "x_track": rng.choice([None, 5.0])},
                  keepalive=ka, keepalive_share=rng.choice([None, None, 0.0, 0.1, 0.5]), provisional=prov)
        if rng.random() < 0.5:
            approved = {(t, u) for t in tracks for u in pool if rng.random() < 0.6}
            kw["award_ok"] = lambda t, u, a=approved: (t, u) in a
            if rng.random() < 0.5:
                kw.update(keep_share=True, now=1000.0)
        for prof in ("50/50", "80/20"):
            a = _combine_weights_v1(serving, champ, rng.choice([0.5, 1.0]) if False else 1.0, tracks, prof, **kw)
            b = combine_weights(serving, champ, 1.0, tracks, prof, **kw)
            assert a == b, (it, prof)
            assert list(a) == list(b)


# ---------------------------------------------------------------- the competition readers get the competition view
# (spec/evaluator.py, engvalidator.py, engcomp.py: commitment = the competition element only, commitment_block = its
# element block, "" / 0 for a pure-inference hotkey)

import dataclasses  # noqa: E402

from reax_subnet import elements as E  # noqa: E402
from reax_subnet.chain import Neuron, Snapshot  # noqa: E402
from reax_subnet.engcomp import observe  # noqa: E402
from reax_subnet.engvalidator import EngineCompetition  # noqa: E402
from reax_subnet.spec import config as C  # noqa: E402
from reax_subnet.spec import evaluator as SE  # noqa: E402
from reax_subnet.spec import state as ST  # noqa: E402
from reax_subnet.spec.commitment import parse_commitment  # noqa: E402

from spec_helpers import HOTKEYS, commit  # noqa: E402
from test_spec_evaluator import TRACK, Env  # noqa: E402

MID = {m: f"{i:x}" * 16 for i, m in enumerate(E.MODEL_ORDER, start=1)}
INF1 = f"reax-s1/1:s1-fast:{MID['s1-fast']}"
INF2 = f"reax-s1/1:s1-fast:{'e' * 16}"
INFM = f"reax-s1m/1:s1-fast:{MID['s1-fast']},s1-pro:{MID['s1-pro']}"
ENG = "reax-eng/1:s1-fast:ghcr.io/alice/engine@sha256:" + "c" * 64
SPEC = commit(TRACK, "a/m0")


@pytest.fixture
def started(monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 100)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 100)


def _neuron(uid, fields, block, hotkey=None, coldkey=None):
    return Neuron(uid=uid, hotkey=hotkey or HOTKEYS[uid], coldkey=coldkey or f"ck{uid}", validator_permit=False, stake=0.0,
                  ip=None, port=None, commitment="".join(fields), commitment_block=block, commitment_fields=tuple(fields))


def _comp_view(chain_block, neurons):
    return E.role_views(Snapshot(chain_block, tuple(neurons), None)).competition


def test_spec_key_stays_when_only_the_inference_element_changes(started):
    v1 = _comp_view(300, [_neuron(1, [INF1, SPEC], 200)])
    v2 = _comp_view(400, [_neuron(1, [INFM, SPEC], 350)])  # the inference element was rewritten at block 350
    n1, n2 = v1.neurons[0], v2.neurons[0]
    assert n1.commitment == n2.commitment == SPEC
    key = lambda n: f"{n.hotkey}|{parse_commitment(n.commitment).text()}"  # noqa: E731
    assert key(n1) == key(n2)
    # the view carries the chain's block (deterministic for every validator, critic round 1 F3): it moves with any write
    assert (n1.commitment_block, n2.commitment_block) == (200, 350)


def test_spec_sync_keeps_the_entry_through_an_inference_rewrite_and_withdraws_when_the_competition_element_goes(
        started, tmp_path):
    env = Env(tmp_path, {})
    tdef = C.DEFAULT_TRACKS[0]
    key = f"{HOTKEYS[1]}|{SPEC}"

    def sync(fields, commit_block, chain_block):
        view = _comp_view(chain_block, [_neuron(1, fields, commit_block)])
        env.ev._sync(tdef, view, 1000.0 + chain_block)
        return env.ev.state.track(TRACK).submissions

    subs = sync([INF1, SPEC], 200, 300)
    assert list(subs) == [key] and subs[key].status == ST.QUEUED
    subs = sync([INF2, SPEC], 350, 400)  # only the inference element changed: same entry, same persisted first sighting
    assert list(subs) == [key] and subs[key].status == ST.QUEUED and subs[key].first_seen_block == 300
    subs = sync([INFM], 450, 500)  # the competition element is removed, the inference element stays
    assert subs[key].status == ST.WITHDRAWN
    subs = sync([INFM, SPEC], 600, 700)  # committed again: a new entry, as after any withdrawal
    assert subs[key].status == ST.QUEUED and any(k.startswith(key + "#withdrawn") for k in subs)


def test_pure_inference_hotkey_is_invisible_to_the_competition_code(started, tmp_path):
    env = Env(tmp_path, {})
    view = _comp_view(300, [_neuron(1, [INF1], 200), _neuron(2, [SPEC], 210)])
    assert [n.commitment for n in view.neurons] == ["", SPEC]
    assert view.neurons[0].commitment_block == 0
    env.ev._sync(C.DEFAULT_TRACKS[0], view, 1000.0)
    assert [s.uid for s in env.ev.state.track(TRACK).submissions.values()] == [2]
    live = SE.compute_leaders(env.ev.state, [TRACK], view)
    assert live[TRACK] == []


def test_keepalive_order_is_deterministic_and_a_rewrite_only_loses_exact_ties(started):
    """compute_keepalive orders equal scores by the view's commitment block = the chain block: every validator orders
    alike. A dual-role entrant that rewrites its inference element looks newer and loses exact ties (accepted)."""
    from test_keepalive import DF as KDF, NOW, _sub
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.PENDING_CONFIRM, 80.0, coldkey="same"), _sub(st, 2, ST.PENDING_CONFIRM, 80.0, coldkey="c2")]
    spec_of = {s.uid: commit(KDF, s.repo) for s in subs}
    before = _comp_view(310, [_neuron(1, [INF1, spec_of[1]], 200, coldkey="same"),
                              _neuron(2, [spec_of[2]], 250, coldkey="c2")])
    assert [u for u, _ in SE.compute_keepalive(st, [KDF], before, {KDF: 0.0}, now=NOW)[KDF]] == [1, 2]
    raw = [_neuron(1, [INF2, spec_of[1]], 390, coldkey="same"), _neuron(2, [spec_of[2]], 250, coldkey="c2")]
    after = _comp_view(400, raw)
    assert [u for u, _ in SE.compute_keepalive(st, [KDF], after, {KDF: 0.0}, now=NOW)[KDF]] == [2, 1]
    # identical to what a one-element commitment written at the same block gets
    raw_old = Snapshot(400, tuple(dataclasses.replace(n, commitment=spec_of[n.uid], commitment_fields=()) for n in raw),
                       None)
    assert [u for u, _ in SE.compute_keepalive(st, [KDF], raw_old, {KDF: 0.0}, now=NOW)[KDF]] == [2, 1]
    # a pure-inference hotkey has no competition commitment in the view: nothing to keep alive
    gone = _comp_view(500, [_neuron(1, [INF2], 390, coldkey="same"), raw[1]])
    assert [u for u, _ in SE.compute_keepalive(st, [KDF], gone, {KDF: 0.0}, now=NOW)[KDF]] == [2]


def test_engine_submissions_come_from_the_competition_element(started, tmp_path):
    comp = EngineCompetition(object(), {"s1-fast": "ref"}, str(tmp_path / "s.json"), start_utc=None)
    view = _comp_view(300, [_neuron(1, [INF1, ENG], 200), _neuron(2, [INF1], 205)])
    assert [(s.uid, s.block) for s in comp.submissions(view, set())["s1-fast"]] == [(1, 200)]


def test_engine_first_committer_survives_an_inference_rewrite_in_the_persisted_state(started, tmp_path):
    from reax_subnet.engcomp import TrackState
    comp = EngineCompetition(object(), {"s1-fast": "ref"}, str(tmp_path / "s.json"), start_utc=None)
    # uid 1 (dual role) commits the engine at 200, uid 2 commits the same digest at 250; uid 1 then rewrites its
    # inference element at 350. The engine state keeps the minimum block per digest, so uid 1 stays the first committer.
    first = _comp_view(260, [_neuron(1, [INF1, ENG], 200), _neuron(2, [ENG], 250)])
    later = _comp_view(400, [_neuron(1, [INFM, ENG], 350), _neuron(2, [ENG], 250)])
    st = TrackState()
    observe(st, comp.submissions(first, set())["s1-fast"], 260)
    digest = next(iter(st.first_seen))
    eligible = observe(st, comp.submissions(later, set())["s1-fast"], 400)
    assert st.first_seen[digest] == (200, HOTKEYS[1]) and [s.uid for s in eligible] == [1]


# ---------------------------------------------------------------- engine keep-alive (release line, miner report 21903)

def test_engine_keepalive_goes_to_pure_entrants_only_and_serving_waiters_stay_out_of_the_median():
    from reax_subnet.competition_split import ENGINE_KA_FRACTION, ENGINE_KA_MAX_SHARE
    base = _serving({100: 0.0001, 101: 0.0001})
    w = combine_weights(base, None, 0.0, {}, engine_waiting=[50, 100, 101])
    each = min(ENGINE_KA_FRACTION * 0.004, ENGINE_KA_MAX_SHARE * profile_for(SPLIT_PROFILE).engine / 1)
    _same(w, _expect(base, {50: each}))  # 100/101 serve: no engine keep-alive, and the median stays 0.004
    # without serving waiters the result is what the release line paid
    assert combine_weights(base, None, 0.0, {}, engine_waiting=[50]) == w


# ---------------------------------------------------------------- critic round 2 NEW-2: slots only for payable entries

def test_a_serving_provisional_candidate_does_not_take_a_track_slot_from_a_pure_entrant():
    base = _serving({100: 0.001})
    prov = {DF: [100], RA: [50], "x_track": [51]}  # 50/50 profile: two slots, three tracks with entries
    with_dual = combine_weights(base, None, 0.0, {}, provisional=prov)
    without = combine_weights(base, None, 0.0, {}, provisional={RA: [50], "x_track": [51]})
    assert with_dual[51] > 0 and with_dual[51] == pytest.approx(without[51])


def test_a_big_server_keepalive_candidate_does_not_take_a_track_slot_from_a_pure_entrant():
    base = _serving({300: 0.05})  # never eligible for a top-up (own >= per_entry)
    ka = {DF: [(300, 40.0)], RA: [(50, 30.0)], "x_track": [(51, 30.0)]}
    w = combine_weights(base, None, 0.0, {}, keepalive=ka)
    assert w[51] > 0 and w[50] > 0
    assert w[300] == pytest.approx(base[300] * (1 - (w[50] + w[51]) / 1.0), rel=0.05)  # serving only, no top-up


def test_a_serving_ranked_uid_of_any_track_stays_out_of_the_median():
    raw = {1: 0.002, 2: 0.002, 3: 0.004, 4: 0.004, 100: 0.0001, 200: 0.9}
    total = math.fsum(raw.values())
    # uid 100 serves and is ranked on a track: left out, the median of the other five is 0.004
    a = combine_weights(raw, None, 0.0, {"x_track": [(100, 40.0)]}, keepalive={DF: [(50, 30.0)]})
    assert a[50] == pytest.approx(0.004 / total)
    # not ranked: an ordinary serving uid, part of the median of six = (0.002 + 0.004) / 2
    b = combine_weights(raw, None, 0.0, {}, keepalive={DF: [(50, 30.0)]})
    assert b[50] == pytest.approx(0.003 / total)


# ---------------------------------------------------------------- critic round 3 NEW-3: the remote competition worker

async def test_the_competition_worker_reads_the_competition_element(started, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from reax_subnet import worker as W

    seen = []

    class Driver:
        def __init__(self, *a, **kw):
            pass

        def due(self):
            return True

        async def run(self, snap, exclude, between=None):
            seen.append(snap)

    class Chain:
        def snapshot(self):
            return Snapshot(500, (_neuron(1, [INF1, ENG], 200), _neuron(2, [INF1, SPEC], 210),
                                  _neuron(3, [SPEC], 220), _neuron(4, [INF1], 230)), None)

    monkeypatch.setattr(W, "CompetitionDriver", Driver)
    wc = SimpleNamespace(state_dir=str(tmp_path / "st"), run_engine=False, heartbeat_s=3600, poll_s=1,
                         common=SimpleNamespace(network="local", netuid=2))
    await W.run_worker(wc, Chain(), once=True, stack=(None, None, None))
    (snap,) = seen
    assert [n.commitment for n in snap.neurons] == [ENG, SPEC, SPEC, ""]
    comp = EngineCompetition(object(), {"s1-fast": "ref"}, str(tmp_path / "s.json"), start_utc=None)
    assert [(s.uid, s.block) for s in comp.submissions(snap, set())["s1-fast"]] == [(1, 200)]
