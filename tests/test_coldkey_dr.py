"""Per-coldkey diminishing returns in the inference pools (tiers.COLDKEY_DR_*; docs/DESIGN-POOLS.md section 4.6).

Pure tests on compute_tiered_weights plus one wiring test through S1Validator.compute_weights. The rule is off until
COLDKEY_DR_START_UTC is set, so most tests set it with monkeypatch.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import elements as E
from reax_subnet import tiers
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.elements import slot_key
from reax_subnet.state import MinerState
from reax_subnet.tiers import (COLDKEY_DR_ORDER_RATIO, TierBudgets, coldkey_dr_active, coldkey_factor,
                               coldkey_factors, compute_tiered_weights)
from test_multimodel import DEV, MID, _validator, role

D = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
AFTER = D + timedelta(days=3)
BEFORE = D - timedelta(microseconds=1)
FULL = {p: tiers.CAPACITY[p].target_dps for p in tiers.CAPACITY}


@pytest.fixture(autouse=True)
def _release_floors_off(monkeypatch):
    from reax_subnet import competition_split as _cs
    for _k in list(_cs.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(_cs.TRACK_ENTRY_FLOOR, _k, None)


@pytest.fixture
def dr_on(monkeypatch):
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", D)


@pytest.fixture
def dr_all_pools(monkeypatch):
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", D)
    monkeypatch.setattr(tiers, "COLDKEY_DR_POOLS", frozenset(tiers.POOL_TIER))


def _caps(ema):
    return {p: {u: FULL[p] for u in m} for p, m in ema.items()}


def _w(ema, coldkey_of=None, now=None, budgets=None, caps=None):
    return compute_tiered_weights(ema, caps if caps is not None else _caps(ema), budgets or TierBudgets(), owner_uid=0,
                                  min_ema=0.1, coldkey_of=coldkey_of, now=now)


# ------------------------------------------------------------------------------------------------ d(k) and grouping
def test_release_constants():
    assert tiers.COLDKEY_DR_START_UTC == datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)  # rc11: active at T1
    assert tiers.COLDKEY_DR_POOLS == frozenset({"s1-lite", "s1-cpu"})
    assert (tiers.COLDKEY_FREE_SLOTS, tiers.COLDKEY_DR_RATIO, COLDKEY_DR_ORDER_RATIO) == (2, 0.7, 1.0)


def test_factor_function():
    assert [coldkey_factor(k) for k in (1, 2)] == [1.0, 1.0]
    assert coldkey_factor(3) == pytest.approx(0.7)
    assert coldkey_factor(4) == pytest.approx(0.49)
    assert coldkey_factor(10) == pytest.approx(0.7 ** 8)
    assert [coldkey_factor(k, 1, 0.5) for k in (1, 2, 3)] == [1.0, 0.5, 0.25]
    assert [coldkey_factor(k, 3, 0.8) for k in (1, 3, 4)] == [1.0, 1.0, pytest.approx(0.8)]
    # whatever the farm size, the effective count stays below free + ratio / (1 - ratio)
    assert math.fsum(coldkey_factor(k) for k in range(1, 500)) == pytest.approx(2 + 0.7 / 0.3, rel=1e-9)


def test_factors_rank_by_score_then_slot_id():
    scores = {5: 1.0, 6: 3.0, 7: 3.0, 8: 2.0, 9: 5.0}
    ck = {s: "A" for s in scores}
    f = coldkey_factors(scores, ck)
    assert f[9] == f[6] == 1.0  # best two
    assert f[7] == pytest.approx(0.7)  # tie 6/7 at 3.0: the lower slot id ranks first
    assert f[8] == pytest.approx(0.49) and f[5] == pytest.approx(0.343)


def test_factors_group_per_coldkey():
    scores = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    f = coldkey_factors(scores, {1: "A", 2: "A", 3: "A", 4: "B", 5: "B"})
    assert f == {1: 1.0, 2: 1.0, 3: pytest.approx(0.7), 4: 1.0, 5: 1.0}


def test_unknown_coldkey_counts_as_its_own():
    scores = {s: 1.0 for s in range(1, 8)}
    assert coldkey_factors(scores, None) == {s: 1.0 for s in scores}
    assert coldkey_factors(scores, {}) == {s: 1.0 for s in scores}
    f = coldkey_factors(scores, {1: "", 2: None, 3: "A", 4: "A", 5: "A"})  # type: ignore[dict-item]
    assert f[1] == f[2] == f[6] == f[7] == 1.0 and f[5] == pytest.approx(0.7)
    # two unknown slots are not one coldkey
    assert coldkey_factors({1: 1.0, 2: 1.0, 3: 1.0}, {}) == {1: 1.0, 2: 1.0, 3: 1.0}


# ------------------------------------------------------------------------------------------------ the gate
def test_gate_instants(monkeypatch):
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", None)
    assert not coldkey_dr_active(AFTER)  # constant None: never
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", D)
    assert not coldkey_dr_active(None)
    assert not coldkey_dr_active(BEFORE)
    assert coldkey_dr_active(D) and coldkey_dr_active(AFTER)


def _farm_fleet():
    ema = {"s1-fast": {u: 1.0 for u in range(10, 40)}, "s1-lite": {u: 1.0 for u in range(100, 130)}}
    ck = {u: f"ck{u}" for p in ema for u in ema[p]}
    ck.update({u: "FARM" for u in range(100, 120)})
    return ema, ck


def test_gate_off_none_before_after_boundary(monkeypatch):
    ema, ck = _farm_fleet()
    base = _w(ema)
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", None)
    # constant None: identical whatever now is
    assert _w(ema, ck, AFTER) == base
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", D)
    assert _w(ema, ck, None) == base      # no clock
    assert _w(ema, None, AFTER) == base   # no coldkeys
    assert _w(ema, ck, BEFORE) == base    # before the date
    at, after = _w(ema, ck, D), _w(ema, ck, AFTER)
    assert at != base                      # the start instant itself is on
    assert at == after
    assert sum(at[u] for u in range(100, 120)) < sum(base[u] for u in range(100, 120))


def test_gate_off_identical_on_random_fixtures(monkeypatch):
    rng = random.Random(92)
    for case in range(200):
        pools = rng.sample(["s1-fast", "s1-pro", "s1-lite", "s1-cpu"], rng.randint(1, 4))
        uid = iter(range(10, 10_000))
        ema = {p: {next(uid): rng.choice([0.05, 0.2, 0.5, 0.9, 1.0]) for _ in range(rng.randint(0, 12))} for p in pools}
        caps = {p: {u: tiers.CAPACITY[p].target_dps * rng.uniform(0, 5) for u in m if rng.random() < 0.9}
                for p, m in ema.items()}
        ck = {u: f"ck{rng.randint(0, 4)}" for m in ema.values() for u in m}
        budgets = TierBudgets(order_ratio=rng.choice([0.5, 1.0]))
        owner, burn = rng.choice([(0, 0.0), (0, 0.1), (None, 0.0)]), None
        owner_uid, burn = owner
        base = compute_tiered_weights(ema, caps, budgets, owner_uid, 0.1, burn)
        for start, now, cmap in ((None, AFTER, ck), (D, BEFORE, ck), (D, None, ck), (D, AFTER, None)):
            monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", start)
            got = compute_tiered_weights(ema, caps, budgets, owner_uid, 0.1, burn, coldkey_of=cmap, now=now)
            assert got == base, (case, start, now)  # bit for bit, not approx


def test_weights_stay_a_distribution_on_random_fixtures(dr_on):
    rng = random.Random(5)
    for _ in range(100):
        uid = iter(range(10, 10_000))
        ema = {p: {next(uid): rng.uniform(0.3, 1.0) for _ in range(rng.randint(1, 15))} for p in ("s1-fast", "s1-lite", "s1-cpu")}
        ck = {u: f"ck{rng.randint(0, 3)}" for m in ema.values() for u in m}
        w = _w(ema, ck, AFTER)
        assert math.fsum(w.values()) == pytest.approx(1.0) and all(x >= 0 for x in w.values())


# ------------------------------------------------------------------------------------------------ the economics
def test_farm_earns_less_than_the_same_miners_on_separate_coldkeys(dr_on):
    ema, ck = _farm_fleet()
    farm = range(100, 120)
    separate = {u: (f"ck{u}" if u in farm else c) for u, c in ck.items()}
    w_farm, w_sep = _w(ema, ck, AFTER), _w(ema, separate, AFTER)
    assert sum(w_farm[u] for u in farm) < 0.5 * sum(w_sep[u] for u in farm)
    # the freed weight stays in the pool (nothing burned): the other lite miners earn more
    others = range(120, 130)
    assert all(w_farm[u] > w_sep[u] for u in others)
    assert math.fsum(w_farm.values()) == pytest.approx(1.0)


def test_pay_per_slot_follows_d_of_k(dr_on):
    ema = {"s1-lite": {u: 1.0 for u in range(10, 18)}}
    w = _w(ema, {u: "A" for u in ema["s1-lite"]}, AFTER)
    for k in range(1, 9):
        assert w[9 + k] / w[10] == pytest.approx(coldkey_factor(k))
    assert math.fsum(w.values()) == pytest.approx(1.0)  # one coldkey in the pool still gets the whole pool share


def test_best_slots_keep_full_pay(dr_on):
    ema = {"s1-lite": {10: 0.5, 11: 1.0, 12: 0.9, 13: 0.4, 20: 1.0}}
    w = _w(ema, {10: "A", 11: "A", 12: "A", 13: "A", 20: "B"}, AFTER)
    # slots 11 (1.0) and 12 (0.9) keep d = 1; 10 and 13 are cut by 0.7 and 0.49
    assert w[11] / w[20] == pytest.approx(1.0) and w[12] / w[20] == pytest.approx(0.9)
    assert w[10] / w[20] == pytest.approx(0.5 * 0.7) and w[13] / w[20] == pytest.approx(0.4 * 0.49)


def test_honest_operators_unchanged_when_no_farm(dr_on):
    rng = random.Random(3)
    uid = iter(range(10, 1000))
    ema = {p: {next(uid): rng.uniform(0.5, 1.0) for _ in range(n)} for p, n in (("s1-fast", 12), ("s1-lite", 9), ("s1-cpu", 3))}
    caps = {p: {u: tiers.CAPACITY[p].target_dps * rng.uniform(0.5, 3) for u in m} for p, m in ema.items()}
    # one hotkey per coldkey, and coldkeys with two hotkeys (both free)
    ck = {}
    for i, u in enumerate(sorted(u for m in ema.values() for u in m)):
        ck[u] = f"ck{i // 2 if i % 3 == 0 else i}"
    assert max(sum(1 for c in ck.values() if c == x) for x in set(ck.values())) <= 2
    ratio1 = TierBudgets(order_ratio=COLDKEY_DR_ORDER_RATIO)  # the DR run uses ratio 1.0 in place of the brake
    plain = compute_tiered_weights(ema, caps, ratio1, 0, 0.1)
    got = compute_tiered_weights(ema, caps, TierBudgets(), 0, 0.1, coldkey_of=ck, now=AFTER)
    assert got.keys() == plain.keys()
    for u in plain:
        assert got[u] == pytest.approx(plain[u], rel=1e-12, abs=1e-15)


def test_brake_replaced_not_stacked(dr_on):
    # 5 fast miners, 5 lite miners on distinct coldkeys: the old brake (0.5) cuts lite to half the per-miner pay of fast;
    # with the rule active the cap is the plain 1.0 ratio, so per-miner pay of lite may equal fast's, never exceed it.
    ema = {"s1-fast": {u: 1.0 for u in range(10, 50)}, "s1-lite": {u: 1.0 for u in range(100, 105)}}
    ck = {u: f"ck{u}" for m in ema.values() for u in m}
    old = _w(ema)
    new = _w(ema, ck, AFTER)
    avg = lambda w, us: math.fsum(w[u] for u in us) / len(us)  # noqa: E731
    lite = list(ema["s1-lite"])
    per_fast = 0.80 / 0.95 / 40  # the fast tier's budget per miner before any excess moves up to it
    assert avg(old, lite) == pytest.approx(0.5 * per_fast)
    assert avg(new, lite) == pytest.approx(per_fast)


def test_tier_cap_counts_effective_miners(dr_on):
    ema = {"s1-fast": {u: 1.0 for u in range(10, 50)}, "s1-lite": {u: 1.0 for u in range(100, 105)}}
    distinct = {u: f"ck{u}" for m in ema.values() for u in m}
    farm = {**distinct, **{u: "FARM" for u in range(100, 105)}}
    w_d, w_f = _w(ema, distinct, AFTER), _w(ema, farm, AFTER)
    lite = list(ema["s1-lite"])
    tier = lambda w: math.fsum(w[u] for u in lite)  # noqa: E731
    per_fast = 0.80 / 0.95 / 40  # the fast tier's budget per miner before any excess moves up to it
    eff = math.fsum(coldkey_factor(k) for k in range(1, 6))  # 2 + 0.7 + 0.49 + 0.343
    # distinct coldkeys: 5 effective miners, the cap binds at 5 x the per-fast-miner pay
    assert tier(w_d) == pytest.approx(5 * per_fast)
    # one farm: the cap follows the effective count (3.533), not the five slots
    assert tier(w_f) == pytest.approx(eff * per_fast)
    assert tier(w_f) < tier(w_d)
    assert math.fsum(w_f.values()) == pytest.approx(1.0)


def test_tier_budgets_stay_the_hard_cap(dr_on):
    ema = {"s1-fast": {u: 1.0 for u in range(10, 14)}, "s1-lite": {u: 1.0 for u in range(100, 110)},
           "s1-cpu": {u: 1.0 for u in range(200, 210)}}
    ck = {u: f"ck{u}" for m in ema.values() for u in m}
    w = _w(ema, ck, AFTER)
    for pool, share in (("s1-fast", 0.80), ("s1-lite", 0.15), ("s1-cpu", 0.05)):
        assert math.fsum(w[u] for u in ema[pool]) <= share + 1e-9


def test_pool_without_other_miners_keeps_its_share(dr_on):
    # the farm is the only lite coldkey: its pool share is not burned, it only pays the farm's best slots less per slot
    ema = {"s1-fast": {u: 1.0 for u in range(10, 20)}, "s1-lite": {u: 1.0 for u in range(100, 110)}}
    ck = {u: ("FARM" if u >= 100 else f"ck{u}") for u in ema["s1-fast"] | ema["s1-lite"]}
    w = _w(ema, ck, AFTER)
    assert math.fsum(w[u] for u in ema["s1-lite"]) == pytest.approx(0.15 / 0.95)  # the pool keeps its budget
    assert w[100] > w[109]
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_splitting_over_coldkeys_is_the_honest_limit(dr_on):
    """The rule cannot see links beyond one coldkey: 20 hotkeys on 10 coldkeys (2 each) keep full pay."""
    ema = {"s1-lite": {u: 1.0 for u in range(10, 50)}}
    one = {u: ("FARM" if u < 30 else f"ck{u}") for u in ema["s1-lite"]}
    ten = {u: (f"F{u % 10}" if u < 30 else f"ck{u}") for u in ema["s1-lite"]}
    ten.update({u: f"F{u % 10}" for u in range(10, 30)})
    pay = lambda w: math.fsum(w[u] for u in range(10, 30))  # noqa: E731
    assert pay(_w(ema, ten, AFTER)) == pytest.approx(pay(_w(ema, {u: f"ck{u}" for u in ema["s1-lite"]}, AFTER)))
    assert pay(_w(ema, one, AFTER)) < pay(_w(ema, ten, AFTER))


def test_pools_are_grouped_separately(dr_all_pools):
    # one hotkey (uid 10) holds a slot in two pools: it is one slot in each pool, never rank 2 of its coldkey
    a, b = slot_key(10, "s1-fast"), slot_key(10, "s1-pro")
    ema = {"s1-fast": {a: 1.0, 20: 1.0}, "s1-pro": {b: 1.0, 21: 1.0}}
    ck = {a: "A", b: "A", 20: "B", 21: "C"}
    w = _w(ema, ck, AFTER)
    assert w[a] == pytest.approx(w[20]) and w[b] == pytest.approx(w[21])


# ------------------------------------------------------------------------------------------------ the pool set
def test_gpu_pools_untouched_by_default(dr_on):
    # a 20-slot coldkey in s1-fast / s1-pro is real capacity: pays per slot as without the rule (only the tier cap ratio moves)
    ema = {"s1-fast": {u: 1.0 for u in range(10, 30)}, "s1-pro": {u: 1.0 for u in range(30, 50)}}
    ck = {u: "BIG" for m in ema.values() for u in m}
    ck.update({60: "x"})
    ema["s1-fast"][60] = 1.0
    w = _w(ema, ck, AFTER)
    assert w == pytest.approx(_w(ema, None, AFTER))
    assert w[10] == pytest.approx(w[29]) and w[10] == pytest.approx(w[60])


def test_all_pools_cuts_the_gpu_farm(dr_all_pools):
    ema = {"s1-fast": {u: 1.0 for u in range(10, 30)} | {60: 1.0}}
    ck = {u: "BIG" for u in range(10, 30)} | {60: "x"}
    w = _w(ema, ck, AFTER)
    assert w[12] / w[60] == pytest.approx(coldkey_factor(3))
    assert w[29] / w[60] == pytest.approx(coldkey_factor(20))


def test_pool_set_picks_the_pools(monkeypatch):
    ema = {"s1-fast": {u: 1.0 for u in range(10, 15)}, "s1-lite": {u: 1.0 for u in range(100, 105)}}
    ck = {u: "A" for u in range(10, 15)} | {u: "B" for u in range(100, 105)}
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", D)
    for pools, cut in ((frozenset({"s1-lite"}), {"s1-lite"}), (frozenset({"s1-fast"}), {"s1-fast"}),
                       (frozenset(), set()), (frozenset(tiers.POOL_TIER), {"s1-fast", "s1-lite"})):
        monkeypatch.setattr(tiers, "COLDKEY_DR_POOLS", pools)
        w = _w(ema, ck, AFTER)
        for pool in ema:
            u0, u4 = min(ema[pool]), max(ema[pool])
            assert (w[u4] < w[u0]) == (pool in cut), (pools, pool)


def test_tier_cap_counts_slots_for_tiers_outside_the_set(monkeypatch):
    # upper tier = s1-fast (not in the set): its 40 slots count as 40 miners even though they share one coldkey
    ema = {"s1-fast": {u: 1.0 for u in range(10, 50)}, "s1-lite": {u: 1.0 for u in range(100, 105)}}
    ck = {u: "F" for u in range(10, 50)} | {u: f"ck{u}" for u in range(100, 105)}
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", D)
    w = _w(ema, ck, AFTER)
    per_fast = 0.80 / 0.95 / 40
    assert math.fsum(w[u] for u in ema["s1-lite"]) == pytest.approx(5 * per_fast)
    monkeypatch.setattr(tiers, "COLDKEY_DR_POOLS", frozenset(tiers.POOL_TIER))  # all pools: fast counts 2 + 0.7/0.3 ~ 4.3
    w = _w(ema, ck, AFTER)
    assert math.fsum(w[u] for u in ema["s1-lite"]) == pytest.approx(0.15 / 0.95)  # cap no longer binds: lite keeps its budget


# ------------------------------------------------------------------------------------------------ validator wiring
def _snap(coldkeys):
    ns = [Neuron(0, "hk0", "owner-ck", False, 0.0, None, None)]
    ns += [Neuron(u, f"hk{u}", ck, False, 0.0, "127.0.0.1", 9000 + u) for u, ck in sorted(coldkeys.items())]
    return Snapshot(100, tuple(ns), "hk0")


def _weights(val, coldkeys, models):
    snap = _snap(coldkeys)
    val.state.miners.clear()
    val.state.sync(snap)
    for u in coldkeys:
        m = models[u]
        val.state.miners[slot_key(u, m)] = MinerState(hotkey=f"hk{u}", ema=1.0, role=role(m),
                                                      capacity=tiers.CAPACITY[m].target_dps)
    return val.compute_weights(snap)


@pytest.mark.usefixtures("rules_v4_gates_off")
def test_validator_wires_coldkeys_from_the_snapshot(monkeypatch, make_wallet, fake_chain_cls, tmp_path):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 0)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 0)
    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    models = {u: "s1-fast" for u in range(10, 20)} | {u: "s1-lite" for u in range(30, 46)}
    farm = {u: "FARM" for u in range(30, 40)}  # ten of the sixteen lite hotkeys
    cks = {u: f"ck{u}" for u in models} | farm
    separate = {u: f"ck{u}" for u in models}

    off = _weights(val, cks, models)
    assert off == _weights(val, separate, models)  # without the date the coldkeys are not read at all

    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", datetime(2020, 1, 1, tzinfo=timezone.utc))  # long passed
    on_farm, on_sep = _weights(val, cks, models), _weights(val, separate, models)
    assert on_farm != on_sep
    assert sum(on_farm[u] for u in farm) < 0.75 * sum(on_sep[u] for u in farm)
    assert math.fsum(on_farm.values()) == pytest.approx(1.0)

    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", datetime(2999, 1, 1, tzinfo=timezone.utc))  # not yet
    assert _weights(val, cks, models) == off


def test_validator_unknown_coldkey_fails_open(monkeypatch, make_wallet, fake_chain_cls, tmp_path):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 0)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 0)
    monkeypatch.setattr(tiers, "COLDKEY_DR_START_UTC", datetime(2020, 1, 1, tzinfo=timezone.utc))
    val = _validator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    models = {u: "s1-lite" for u in range(30, 38)}
    blank = _weights(val, {u: "" for u in models}, models)  # the metagraph row has no coldkey
    separate = _weights(val, {u: f"ck{u}" for u in models}, models)
    assert blank == separate
