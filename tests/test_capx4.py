"""Mechanism v2: the per-hotkey capacity cap of s1-fast / s1-pro is multiplied by CAPACITY_CAP_MULTIPLIER (x2 =
1280/s and 320/s, x4 = 2560/s and 640/s), so one hotkey can stand for a multi-GPU or multi-machine fleet. These tests
build the x1, x2 and x4 tables from tiers.capacity_table, so they hold whatever value the release constant has."""

from __future__ import annotations

import math

import pytest

from reax_subnet import probeshard, s1validator, tiers
from reax_subnet.chain import Neuron
from reax_subnet.s1validator import probe_request_load
from reax_subnet.tiers import TierBudgets, capacity_table, compute_tiered_weights

X1 = capacity_table(1.0)
X2 = capacity_table(2.0)
X4 = capacity_table(4.0)
MULTIPLIERS = [1.0, 2.0, 4.0]


def test_x1_reproduces_the_pre_v2_numbers_exactly():
    fast, pro = X1["s1-fast"], X1["s1-pro"]
    assert (fast.max_factor, pro.max_factor) == (16.0, 16.0)
    assert (fast.cap_dps, pro.cap_dps) == (640.0, 160.0)
    for cap, requests in ((fast, math.ceil(640 * 5 / 16) + 16),
                          (pro, math.ceil(160 * 8 / 8) + 16)):
        assert cap.max_requests == requests  # 216 and 176, as before
        for budget in (cap.target_dps, 4 * cap.target_dps, 100.0, cap.cap_dps):
            assert cap.lanes_for(budget) == 16  # at or below the old cap: today's lane count
            assert cap.requests_for(budget) == math.ceil(budget * cap.window_s / cap.items_per_request) + 16
        for answered in (1, 3, 12, 100, 5000):
            for dps in (None, 1.0, cap.cap_dps):
                assert cap.verified(answered, dps) == cap.verified(answered)
    assert tiers.CAPACITY == capacity_table(tiers.CAPACITY_CAP_MULTIPLIERS)


def test_only_the_gpu_pools_get_the_multiplier():
    assert X4["s1-fast"].cap_dps == 2560.0 and X4["s1-pro"].cap_dps == 640.0
    for pool in ("s1-lite", "s1-cpu"):
        assert X4[pool] == X1[pool]


@pytest.mark.parametrize("mult", MULTIPLIERS)
@pytest.mark.parametrize("pool", ["s1-fast", "s1-pro"])
def test_lanes_scale_with_the_budget_between_today_and_the_bound(pool, mult):
    cap = capacity_table(mult)[pool]
    at_cap = int(mult * 16)  # 16 lanes carry the old cap, so the cap at x2 / x4 needs 32 / 64
    assert cap.lanes_for(cap.target_dps) == 16  # a new UID gets today's lanes
    assert cap.lanes_for(cap.cap_dps) == at_cap
    budgets = [cap.target_dps * 1.5 ** k for k in range(20) if cap.target_dps * 1.5 ** k <= cap.cap_dps]
    lanes = [cap.lanes_for(b) for b in budgets]
    assert lanes == sorted(lanes) and min(lanes) == 16 and max(lanes) <= cap.max_inflight == 64
    # the lanes carry the budget: lanes x items per request / latency >= budget
    for b, n in zip(budgets, lanes):
        assert n * cap.items_per_request / cap.lane_latency_s >= b * (1 - 1e-9)
    assert cap.lanes_for(1e9) == 64  # bounded; the cap itself is reached at x4 (review F5)
    assert cap.max_requests == cap.requests_for(cap.cap_dps)
    assert cap.max_requests == math.ceil(cap.cap_dps * cap.window_s / cap.items_per_request) + at_cap


def test_small_pools_keep_their_fixed_lanes():
    for pool in ("s1-lite", "s1-cpu"):
        cap = X4[pool]
        assert cap.lanes_for(cap.cap_dps) == cap.inflight


@pytest.mark.parametrize("mult", MULTIPLIERS)
@pytest.mark.parametrize("pool", ["s1-fast", "s1-pro"])
def test_verified_sample_grows_with_the_measured_factor(pool, mult):
    cap = capacity_table(mult)[pool]
    answered = 5000
    at_old_cap = cap.verified(answered, 16 * cap.target_dps)
    at_cap = cap.verified(answered, cap.cap_dps)
    assert at_old_cap == cap.verified_max  # up to the pre-v2 cap nothing changes
    assert at_cap == math.ceil(cap.verified_max * mult ** tiers.VERIFIED_EXPONENT)  # exponent 0.75 (review F3)
    assert (at_cap > at_old_cap) == (mult > 1)
    assert cap.verified(answered, 4 * cap.cap_dps) == at_cap  # factor is capped
    assert cap.verified(answered) == cap.verified_max  # no measurement given: unchanged
    assert cap.verified(5, cap.cap_dps) == 3  # min 3, share 25 % unchanged for small answers
    assert cap.verified(2, cap.cap_dps) == 2


def test_detection_per_probe_never_falls_below_today_while_the_checked_fraction_falls():
    for pool in ("s1-fast", "s1-pro"):
        base, cap = X1[pool], X4[pool]
        answered_old = int(base.cap_dps * base.window_s)
        answered_new = int(cap.cap_dps * cap.window_s)
        k_old, k_new = base.verified(answered_old, base.cap_dps), cap.verified(answered_new, cap.cap_dps)
        assert k_new > k_old  # catch chance 1 - (1 - f)^k is higher per probe
        assert k_new / answered_new < k_old / answered_old  # but the checked share of the answers is lower


def _weights(capacities, pool="s1-pro"):
    ema = {pool: {uid: 1.0 for uid in capacities}}
    return compute_tiered_weights(ema, {pool: capacities}, TierBudgets(), owner_uid=None, min_ema=0.1)


def test_one_hotkey_with_four_machines_earns_what_four_hotkeys_with_one_machine_earn(monkeypatch):
    monkeypatch.setattr(tiers, "CAPACITY", X4)
    one = _weights({1: 4 * 160.0, 10: 160.0, 11: 80.0})
    four = _weights({1: 160.0, 2: 160.0, 3: 160.0, 4: 160.0, 10: 160.0, 11: 80.0})
    assert one[1] == pytest.approx(math.fsum(four[u] for u in (1, 2, 3, 4)), rel=0.01)
    assert one[10] == pytest.approx(four[10], rel=1e-9) and one[11] == pytest.approx(four[11], rel=1e-9)
    # linear up to the new cap: 640/s is paid four times 160/s, the old cap (160/s) would have stopped at 1x
    assert one[1] == pytest.approx(4 * one[10], rel=1e-9)
    monkeypatch.setattr(tiers, "CAPACITY", X1)
    old = _weights({1: 4 * 160.0, 10: 160.0, 11: 80.0})
    assert old[1] == pytest.approx(old[10], rel=1e-9)  # at x1 the 640/s hotkey is capped at 160/s


def _targets(n_small):
    out = [Neuron(1, "hk1", "ck-big", False, 0.0, "198.51.100.1", 8001)]
    out += [Neuron(2 + i, f"hk{2 + i}", f"ck{i}", False, 0.0, "198.51.100.2", 8002 + i) for i in range(n_small)]
    return out


def test_shard_planner_places_a_2560_dps_s1_fast_uid_within_a_shard():
    cap = X4["s1-fast"]
    targets = _targets(30)
    budgets = {n.uid: (cap.cap_dps if n.uid == 1 else cap.target_dps) for n in targets}
    lanes = {u: cap.lanes_for(b) for u, b in budgets.items()}
    assert lanes[1] == 64 and lanes[2] == 16
    lanes_total = sum(lanes.values())
    shards = probeshard.shard_count(lanes_total, len(targets), {"REAX_PROBE_SHARDS": "auto"})
    shards = max(shards, math.ceil(lanes_total / probeshard.PROBE_LANES_PER_SHARD))  # the host's cores may be fewer
    assert shards >= 3
    load = lambda n: probe_request_load(cap, budgets[n.uid])  # noqa: E731
    plan = probeshard.plan_shards(targets, load, shards)
    assert sorted(n.uid for s in plan for n in s) == sorted(budgets)
    shard_lanes = [sum(lanes[n.uid] for n in s) for s in plan]
    big = next(s for s in plan if any(n.uid == 1 for n in s))
    # The plan balances the requests a shard sends (its CPU), not its lanes: the big UID's 64 lanes fit a shard, and
    # no shard's lanes stray far above the average either.
    assert sum(lanes[n.uid] for n in big) <= probeshard.PROBE_LANES_PER_SHARD
    assert max(shard_lanes) <= 1.5 * lanes_total / len(plan)
    biggest = load(big[0]) if big[0].uid == 1 else max(load(n) for n in big)
    fair = math.fsum(load(n) for n in targets) / len(plan)
    # balance guard: a shard never carries more than a fair share plus the biggest single UID
    assert max(math.fsum(load(n) for n in s) for s in plan) <= fair + biggest
    # the big UID's own probe load (requests/s) is 864 requests / 5 s, far below what one shard can send
    assert load(targets[0]) == pytest.approx((math.ceil(2560 * 5 / 16) + 64) / 5)


def test_miner_defaults_follow_the_cap():
    """x1: today's 16 / 240 / 8; x2: 32 / 456 / 8; x4: 64 lanes, 864 + 24 requests per minute, queue of 5 x
    max_concurrent holds the lanes."""
    def defaults(table):
        lanes = max(table[p].lanes_for(table[p].cap_dps) for p in ("s1-fast", "s1-pro"))
        reqs = max(table[p].max_requests for p in ("s1-fast", "s1-pro"))
        return lanes, reqs + 24, max(8, -(-lanes // 4))

    assert defaults(X1) == (16, 240, 8)
    assert defaults(X2) == (32, 456, 8)
    assert defaults(X4) == (64, 888, 16)
    from reax_subnet import config
    assert (config.DEFAULT_PER_CALLER_CONCURRENCY, config.DEFAULT_PER_CALLER_PER_MINUTE,
            config.DEFAULT_MAX_CONCURRENT) == defaults(tiers.CAPACITY)


def test_max_pending_per_ip_follows_the_lanes():
    from reax_subnet import config
    assert config.default_max_pending_per_ip(16) == 32  # x1: unchanged
    assert config.default_max_pending_per_ip(32) == 64
    assert config.default_max_pending_per_ip(64) == 128  # all 64 lanes of one probe start from one validator IP
    cfg = config.load_miner({})
    assert cfg.max_pending_per_ip == config.default_max_pending_per_ip(cfg.per_caller_concurrency)
    assert cfg.max_pending_per_ip >= 2 * cfg.per_caller_concurrency
    # an operator's explicit values still win
    cfg = config.load_miner({"REAX_PER_CALLER_CONCURRENCY": "8", "REAX_MAX_PENDING_PER_IP": "5"})
    assert (cfg.per_caller_concurrency, cfg.max_pending_per_ip) == (8, 5)


# --- F1: the probe's preparation ceiling --------------------------------------------------------------------------

def _fleet(table, n_pro, n_fast, mult_of_target):
    caps, budgets = {}, {}
    for i in range(n_pro):
        caps[i], budgets[i] = table["s1-pro"], min(table["s1-pro"].cap_dps, mult_of_target * 10.0)
    for i in range(n_fast):
        caps[1000 + i], budgets[1000 + i] = table["s1-fast"], min(table["s1-fast"].cap_dps, mult_of_target * 40.0)
    return caps, budgets


def _total(caps, budgets):
    return sum(caps[u].requests_for(b) for u, b in budgets.items())


def test_the_live_size_x1_fleet_is_below_the_ceiling_and_unchanged():
    caps, budgets = _fleet(X1, 146, 88, 16.0)  # everybody at the old cap: ~45k requests
    assert 40_000 < _total(caps, budgets) < 50_000
    assert _total(caps, budgets) < s1validator.PROBE_MAX_PREPARED_REQUESTS
    assert tiers.scale_budgets_to_ceiling(budgets, caps, s1validator.PROBE_MAX_PREPARED_REQUESTS) == budgets


@pytest.mark.parametrize("mult", [2.0, 4.0])
def test_a_ramped_fleet_is_scaled_to_the_ceiling_by_one_common_factor(mult):
    table = capacity_table(mult)
    # x4 (both pools, 1.2.0rc11.post7): the live-size fleet fits (see below), so a bigger one (200 s1-pro + 130 s1-fast) shows the scaling
    caps, budgets = _fleet(table, *((146, 88) if mult <= 2.0 else (200, 130)), 16.0 * mult)  # every UID at the new cap
    ceiling = s1validator.PROBE_MAX_PREPARED_REQUESTS
    if mult <= 2.0:
        # the live-size fleet fully ramped at x2 is prepared unscaled
        assert _total(caps, budgets) <= ceiling
        assert tiers.scale_budgets_to_ceiling(budgets, caps, ceiling) == budgets
        return
    assert _total(caps, budgets) > ceiling
    scaled = tiers.scale_budgets_to_ceiling(budgets, caps, ceiling)
    assert _total(caps, scaled) <= ceiling
    assert _total(caps, scaled) > 0.97 * ceiling  # not scaled further than needed
    ratios = {round(scaled[u] / budgets[u], 9) for u in budgets}
    assert len(ratios) == 1 and next(iter(ratios)) < 1.0  # same factor for every UID (proportional)
    assert all(scaled[u] >= caps[u].target_dps for u in budgets)


def test_the_ceiling_never_cuts_below_a_uids_target_and_stays_equal_for_equal_uids():
    caps = {u: X4["s1-fast"] for u in range(100)}
    budgets = {u: (2560.0 if u < 50 else 40.0) for u in range(100)}
    scaled = tiers.scale_budgets_to_ceiling(budgets, caps, 1)  # impossible ceiling: everyone at the target
    assert all(v == 40.0 for v in scaled.values())
    scaled = tiers.scale_budgets_to_ceiling(budgets, caps, 20_000)
    assert len({scaled[u] for u in range(50)}) == 1 and len({scaled[u] for u in range(50, 100)}) == 1
    assert _total(caps, scaled) <= 20_000
    assert tiers.scale_budgets_to_ceiling({}, {}, 10) == {}


def test_a_junk_fast_swarm_cannot_blow_up_the_preparation():
    """300 fresh-but-instantly-answering UIDs all at the x4 cap draw 4x the prepared requests of the live fleet."""
    caps, budgets = _fleet(X4, 0, 300, 64.0)
    assert _total(caps, budgets) > 250_000
    scaled = tiers.scale_budgets_to_ceiling(budgets, caps, s1validator.PROBE_MAX_PREPARED_REQUESTS)
    assert _total(caps, scaled) <= s1validator.PROBE_MAX_PREPARED_REQUESTS


# --- F6: fewer shards than lanes / 192 (small host) ---------------------------------------------------------------

def test_a_small_host_gets_fewer_shards_than_the_lanes_ask_for_and_still_plans_every_uid(monkeypatch):
    cap = X4["s1-fast"]
    targets = _targets(30)
    budgets = {n.uid: (cap.cap_dps if n.uid == 1 else cap.target_dps) for n in targets}
    lanes_total = sum(cap.lanes_for(b) for b in budgets.values())
    wanted = math.ceil(lanes_total / probeshard.PROBE_LANES_PER_SHARD)
    assert wanted >= 3
    monkeypatch.setattr(probeshard.os, "sched_getaffinity", lambda _pid: set(range(4)), raising=False)
    shards = probeshard.shard_count(lanes_total, len(targets), {"REAX_PROBE_SHARDS": "auto"})
    assert shards == 4 - probeshard.PROBE_SHARD_SPARE_CORES < wanted  # the cores cap the shard count
    plan = probeshard.plan_shards(targets, lambda n: probe_request_load(cap, budgets[n.uid]), shards)
    assert sorted(n.uid for s in plan for n in s) == sorted(budgets)  # nobody is lost; the CPU guard (cpu_share >
    # PROBE_CPU_SATURATED, tests/test_probe_shards.py) is what drops a probe that does not fit the cores


# --- F7: per-UID lanes reach the probe window; rollback pays as the x1 cap -----------------------------------------

async def test_probe_window_gets_the_lanes_of_each_uid(make_wallet, fake_chain_cls, tmp_path, monkeypatch):
    """_probe_capacity hands _probe_window the lanes of every UID (tiers.lanes_for of its scaled budget)."""
    from test_tiers import PROBE_V2, _probe_arena
    cap = PROBE_V2.__class__(target_dps=40.0, window_s=2.0, items_per_request=16, inflight=16, max_factor=16.0,
                             alpha=1.0, lane_latency_s=4.0, max_inflight=64)  # slow answers: lanes grow early
    monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 1)
    monkeypatch.setattr(tiers, "CAPACITY", {"s1-fast": cap})
    monkeypatch.setattr(s1validator, "CAPACITY", tiers.CAPACITY)
    seen = []
    window = s1validator.S1Validator._probe_window

    async def spy(self, targets, roles, prepared, ready=None, go=None, lanes=None):
        seen.append(dict(lanes or {}))
        return await window(self, targets, roles, prepared, ready, go, lanes)

    monkeypatch.setattr(s1validator.S1Validator, "_probe_window", spy)
    monkeypatch.setenv("REAX_PROBE_SHARDS", "1")
    from test_tiers import Machine
    val, by_group, _s, _r = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, [(Machine(160.0), [{}])], cap=cap)
    assert seen and all(set(l) == set(by_group[0]) for l in seen)
    first = seen[0][by_group[0][0]]
    assert first == cap.lanes_for(cap.target_dps) == 16  # a new UID: today's lanes
    # once measured (~160/s) the UID is probed at a bigger budget and gets proportionally more lanes in the window
    assert max(l[by_group[0][0]] for l in seen) > 16


def test_rollback_stored_capacity_above_the_x1_cap_pays_as_the_cap(monkeypatch):
    """A stored EMA from x4 (2000/s) after a rollback to x1: pays as the 640/s cap, probes at the cap budget."""
    monkeypatch.setattr(tiers, "CAPACITY", X1)
    cap = X1["s1-fast"]
    ema = {"s1-fast": {1: 1.0, 2: 1.0}}
    caps_at = lambda c: compute_tiered_weights(ema, {"s1-fast": {1: c, 2: 640.0}}, TierBudgets(),  # noqa: E731
                                               owner_uid=None, min_ema=0.1)
    w = caps_at(2000.0)
    assert w[1] == pytest.approx(w[2], rel=1e-9)  # the same as a hotkey measured exactly at the cap
    assert cap.budget_dps(2000.0) == cap.cap_dps == 640.0
    assert cap.factor(2000.0) == cap.max_factor == 16.0
    assert cap.verified(5000, 2000.0) == cap.verified_max  # and checked as before


# --- review-delta 6: budgets recover after a fleet that sat above the preparation ceiling shrinks ----------------------

def _simulate_probes(table, true_dps, probes, ema, last_scale, divide=True):
    """Pure model of the probe loop (no network): a UID shows min(true capacity, scaled budget) and the validator
    folds it into the moving average exactly as state.update_capacity does."""
    caps = {u: table["s1-fast"] for u in true_dps}
    ceiling = s1validator.PROBE_MAX_PREPARED_REQUESTS
    alpha = table["s1-fast"].alpha
    for _ in range(probes):
        prev = {u: ema.get(u) for u in true_dps}
        budgets, last_scale = tiers.probe_budgets(prev, caps, ceiling, last_scale if divide else 1.0)
        for u, c in true_dps.items():
            shown = min(c, budgets[u])
            # divide=False models the plain rule of capx4 (plain average, scaled measurement fed back unchanged)
            ema[u] = tiers.fold_capacity(ema.get(u), shown, budgets[u], last_scale if divide else 1.0, alpha)
    return last_scale


@pytest.mark.parametrize("mult", [2.0, 4.0])
def test_capacity_recovers_within_a_bounded_number_of_probes_after_the_fleet_shrinks(mult):
    """A junk-fast swarm (answers at any budget) pushes the fleet over PROBE_MAX_PREPARED_REQUESTS for 20 probes, so
    every budget is scaled down. Then it leaves. The honest big UID must show its true capacity again: it was never
    measured below the scale it ran at (budget / scale is divided back in), and after the shrink the average climbs
    back to 90 % of the truth within RECOVERY_PROBES (the budget is 4x the average, the average moves 30 % per probe)."""
    table = capacity_table(mult)
    cap = table["s1-fast"].cap_dps
    true = 0.75 * cap
    swarm = {2000 + i: cap for i in range(1000)}  # 1000: deep over the 220k ceiling at x2 as well
    ema = {1: true}  # the honest UID was measured at its true capacity before the swarm came
    scale = _simulate_probes(table, {1: true, **swarm}, 20, ema, 1.0)
    assert scale < 1.0  # the ceiling really cut the budgets
    assert ema[1] >= 0.9 * true  # no ratchet downwards while the fleet is over the ceiling
    for u in swarm:
        ema.pop(u)
    RECOVERY_PROBES = 3
    _simulate_probes(table, {1: true, **{3000 + i: 40.0 for i in range(50)}}, RECOVERY_PROBES, ema, scale)
    assert ema[1] >= 0.9 * true


def test_without_the_scale_handling_the_same_fleet_would_ratchet_down():
    """Documents why probe_budgets / fold_capacity handle the scale: the plain rule (budget from the scaled average)
    drops the honest UID's measured capacity well below its truth while the swarm is there."""
    table = capacity_table(2.0)
    cap = table["s1-fast"].cap_dps
    true = 0.75 * cap
    swarm = {2000 + i: cap for i in range(3000)}   # 3000 (was 500 at the 60k ceiling, 1000 at 90k): deep enough under 220k
    ema = {1: true}
    _simulate_probes(table, {1: true, **swarm}, 20, ema, 1.0, divide=False)
    assert ema[1] < 0.5 * true


def test_probe_budgets_below_the_ceiling_are_the_plain_budget_rule():
    caps = {1: X2["s1-fast"], 2: X2["s1-pro"], 3: X2["s1-fast"]}
    budgets, scale = tiers.probe_budgets({1: 100.0, 2: None, 3: 5000.0}, caps, s1validator.PROBE_MAX_PREPARED_REQUESTS)
    assert scale == 1.0
    assert budgets == {1: caps[1].budget_dps(100.0), 2: caps[2].budget_dps(None), 3: caps[3].budget_dps(5000.0)}
    assert tiers.probe_budgets({}, {}, 10) == ({}, 1.0)
    # a degenerate last scale (0) must not divide by zero
    assert tiers.probe_budgets({1: 100.0}, caps, 10_000, 0.0)[0][1] >= caps[1].target_dps


def test_fold_capacity_holds_a_saturated_scaled_probe_but_lets_a_real_drop_through():
    f = tiers.fold_capacity
    assert f(None, 50.0, 100.0, 0.2, 0.3) == 50.0
    assert f(800.0, 100.0, 100.0, 1.0, 0.3) == pytest.approx(0.7 * 800 + 0.3 * 100)  # unscaled probe: plain average
    assert f(800.0, 100.0, 100.0, 0.2, 0.3) == 800.0  # scaled and saturated: only a floor, never lowers
    assert f(100.0, 100.0, 100.0, 0.2, 0.3) == 100.0  # saturated at the floor: kept
    assert f(100.0, 400.0, 100.0, 0.2, 0.3) == pytest.approx(0.7 * 100 + 0.3 * 400)  # shows more: raises
    assert f(800.0, 40.0, 100.0, 0.2, 0.3) == pytest.approx(0.7 * 800 + 0.3 * 40)  # answered far less: real drop
    assert f(800.0, 0.0, 100.0, 0.2, 0.3) == pytest.approx(0.7 * 800)  # a strike (0) always counts


def test_a_held_average_lasts_at_most_three_scaled_probes_in_a_row():
    """Lead review: while the ceiling binds, a UID that answers most of a small scaled budget may keep its average for
    at most CAPACITY_HOLD_MAX_PROBES probes; then a real capacity drop shows again. An unscaled probe resets the count."""
    f = tiers.fold_capacity_bounded
    ema, held = 600.0, 0
    for _ in range(tiers.CAPACITY_HOLD_MAX_PROBES):
        ema, held = f(ema, 300.0, 320.0, 0.5, 0.3, held)   # 300 >= 90 % of the scaled budget 320: held
        assert ema == 600.0
    ema2, held2 = f(ema, 300.0, 320.0, 0.5, 0.3, held)     # fourth scaled probe: averaged as usual
    assert ema2 == pytest.approx(0.7 * 600.0 + 0.3 * 300.0) and held2 == held
    ema3, held3 = f(ema2, 300.0, 320.0, 1.0, 0.3, held2)   # unscaled probe: plain average, count reset
    assert held3 == 0 and ema3 == pytest.approx(0.7 * ema2 + 0.3 * 300.0)


# --- per-pool multiplier: s1-pro x4 (640/s) with s1-fast staying x2 (1280/s) -------------------------------------
PRO4 = capacity_table({"s1-fast": 2.0, "s1-pro": 4.0})


def test_release_ships_both_gpu_pools_at_x4():
    """The release constants of 1.2.0rc11.post7: s1-fast x4 (2560/s) and s1-pro x4 (640/s), WEIGHTS_VERSION_KEY 1260."""
    assert (tiers.CAPACITY_CAP_MULTIPLIER, tiers.S1_PRO_CAP_MULTIPLIER) == (4.0, 4.0)
    assert tiers.CAPACITY == X4 == capacity_table({"s1-fast": 4.0, "s1-pro": 4.0})
    assert (tiers.CAPACITY["s1-fast"].cap_dps, tiers.CAPACITY["s1-pro"].cap_dps) == (2560.0, 640.0)
    assert tiers.CAPACITY_CAP_MULTIPLIERS == {"s1-fast": 4.0, "s1-pro": 4.0}
    # the previous release tables are still reproducible: the rollback targets (pro-only x4 release, then x2 / x2)
    assert capacity_table({"s1-fast": 2.0, "s1-pro": 2.0}) == X2
    assert tiers.CAPACITY["s1-fast"].cap_dps == 2 * PRO4["s1-fast"].cap_dps and tiers.CAPACITY["s1-pro"] == PRO4["s1-pro"]
    assert tiers.CAPACITY["s1-pro"].cap_dps == 2 * X2["s1-pro"].cap_dps


def test_s1_pro_x4_with_s1_fast_at_x2_is_the_rollback_table_and_leaves_the_small_pools_alone():
    fast, pro = PRO4["s1-fast"], PRO4["s1-pro"]
    assert (fast.cap_dps, pro.cap_dps) == (1280.0, 640.0)
    assert (fast.max_factor, pro.max_factor) == (32.0, 64.0)
    assert fast == X2["s1-fast"]
    for pool in ("s1-lite", "s1-cpu"):
        assert PRO4[pool] == X1[pool]
    # lanes: s1-fast 32 at its cap, s1-pro 64 at its cap (640 x 0.8 s / 8 items)
    assert fast.lanes_for(fast.cap_dps) == 32 and pro.lanes_for(pro.cap_dps) == 64
    # probe sizes: 640 x 8 s / 8 + 64 = 704 requests; s1-fast unchanged 432
    assert pro.max_requests == 704 and fast.max_requests == 432
    assert pro.verified(5000, pro.cap_dps) == math.ceil(10 * 4 ** tiers.VERIFIED_EXPONENT) == 29


def test_s1_fast_x4_numbers():
    fast = tiers.CAPACITY["s1-fast"]
    assert (fast.cap_dps, fast.max_factor) == (2560.0, 64.0)
    assert fast.lanes_for(fast.cap_dps) == fast.max_inflight == 64  # 2560 x 0.4 s / 16 items = 64: the s1-pro ceiling
    assert fast.max_requests == 864  # 2560 x 5 s / 16 + 64
    assert fast.requests_for(fast.cap_dps) == 864
    assert fast.verified(5000 * 16, fast.cap_dps) == math.ceil(40 * 4 ** tiers.VERIFIED_EXPONENT) == 114
    for pool in ("s1-lite", "s1-cpu"):
        assert tiers.CAPACITY[pool] == X1[pool]


def test_capacity_table_validates_its_multipliers():
    assert capacity_table({"s1-pro": 4.0})["s1-fast"].cap_dps == 640.0  # missing pool = x1
    with pytest.raises(ValueError):
        capacity_table({"s1-lite": 2.0})
    with pytest.raises(ValueError):
        capacity_table({"s1-pro": 0.5})
    with pytest.raises(ValueError):
        capacity_table(float("nan"))


def test_miner_defaults_per_model_follow_the_pools_multiplier(monkeypatch):
    from reax_subnet import config
    # at the release (s1-fast x4, s1-pro x4)
    assert config.probe_defaults("s1-fast") == (64, 888, 16)
    assert config.probe_defaults("s1-pro") == (64, 728, 16)
    assert config.probe_defaults("s1-lite") == (32, 456, 8)  # small pools: the s1-fast values (9f23128)
    # the previous release table (x2 / x2): s1-pro needs only 352 + 24 requests per minute at 320/s
    monkeypatch.setattr(config, "CAPACITY", X2)
    assert config.probe_defaults("s1-pro") == (32, 376, 8)
    assert config.probe_defaults("s1-lite") == (32, 456, 8)
    monkeypatch.setattr(config, "CAPACITY", PRO4)
    assert config.probe_defaults("s1-fast") == (32, 456, 8)      # the pro-only x4 release (rollback of s1-fast)
    assert config.probe_defaults("s1-pro") == (64, 728, 16)      # 704 + 24 requests per minute
    # small pools keep the s1-fast values, i.e. exactly the 9f23128 defaults (32 / 456 / 8, pending 64)
    assert config.probe_defaults("s1-lite") == (32, 456, 8) == config.probe_defaults("s1-cpu")
    # the queue (4 x max_concurrent + max_concurrent) holds the lanes of one probe
    for model in ("s1-fast", "s1-pro"):
        lanes, _, mc = config.probe_defaults(model)
        assert 5 * mc >= lanes


def test_load_miner_gives_each_model_its_own_defaults(monkeypatch):
    from reax_subnet import config
    monkeypatch.setattr(config, "CAPACITY", X4)
    env = {"REAX_MECHANISM": "s1", "REAX_MODELS": "s1-fast,s1-pro"}
    try:
        cfg = config.load_miner(env)
    except config.ConfigError:
        pytest.skip("multi-model env needs runtime URLs here")
    by = {m.model: m for m in cfg.models}
    assert (by["s1-fast"].per_caller_concurrency, by["s1-fast"].per_caller_per_minute,
            by["s1-fast"].max_concurrent) == (64, 888, 16)
    assert (by["s1-pro"].per_caller_concurrency, by["s1-pro"].per_caller_per_minute,
            by["s1-pro"].max_concurrent) == (64, 728, 16)
    assert cfg.max_pending_per_ip == 128  # holds the lanes of the widest model


def test_a_fully_ramped_live_size_fleet_with_both_gpu_pools_at_x4_fits_the_preparation_ceiling():
    """Worst case, every slot at its cap at the 25 Oct 2026 counts (workers/cap-headroom Q2): 55 s1-fast slots x 864 = 47.5k,
    168 s1-pro x 704 = 118.3k, plus s1-lite / s1-cpu at their caps (~16k): ~182k; the ceiling (220k) leaves >= 20 % growth
    margin and is prepared without scaling."""
    caps = ({i: X4["s1-pro"] for i in range(168)} | {1000 + i: X4["s1-fast"] for i in range(55)}
            | {2000 + i: X4["s1-lite"] for i in range(148)} | {3000 + i: X4["s1-cpu"] for i in range(161)})
    budgets = {u: c.cap_dps for u, c in caps.items()}
    total = _total(caps, budgets)
    assert 168 * 704 + 55 * 864 == 165_792
    assert 165_792 < total < 190_000
    ceiling = s1validator.PROBE_MAX_PREPARED_REQUESTS
    assert ceiling == 220_000 and total * 1.2 <= ceiling
    assert tiers.scale_budgets_to_ceiling(budgets, caps, ceiling) == budgets
    # at 0.78 ms per request (slowest measured) a full ceiling must still fit the preparation budget
    assert s1validator.PROBE_PREPARE_BUDGET_S >= ceiling * 0.78e-3 and s1validator.PROBE_PREPARE_BUDGET_S == 180.0


def test_small_pool_miner_defaults_are_the_9f23128_ones(monkeypatch):
    """s1-lite / s1-cpu miners: limits 8 / 32 / 456 and pending 64 per IP, as before s1-pro went to x4."""
    from reax_subnet.config import load_miner
    for pool in ("s1-lite", "s1-cpu"):
        cfg = load_miner({"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
                          "REAX_MODEL": pool})
        assert (cfg.max_concurrent, cfg.per_caller_concurrency, cfg.per_caller_per_minute) == (8, 32, 456), pool
        assert cfg.max_pending_per_ip == 64, pool


def test_prepare_budget_env_override(monkeypatch):
    f = s1validator._probe_prepare_budget_s
    monkeypatch.delenv("REAX_PROBE_PREPARE_BUDGET_S", raising=False)
    assert f() == s1validator.PROBE_PREPARE_BUDGET_S == 180.0
    for raw, want in (("200", 200.0), ("0", 180.0), ("-5", 180.0), ("nan", 180.0), ("inf", 180.0), ("x", 180.0), ("", 180.0)):
        monkeypatch.setenv("REAX_PROBE_PREPARE_BUDGET_S", raw)
        assert f() == want, raw
