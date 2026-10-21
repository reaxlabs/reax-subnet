"""Competition share layer (docs/COMPETITIONS.md): profiles, decay, unclaimed -> inference, start time."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.competition_split import (
    COMPETITIONS_START_UTC, GENERALIST_START_UTC, PROFILES, SPLIT_PROFILE, TRACK_SCHEDULE, NoSpecialistEvaluator, RankedLeaders,
    active_tracks, combine_weights, engine_decay_fraction, track_payout,
)
from reax_subnet.engcomp import Champion, EvalResult, RewardParams, TrackState
from reax_subnet.engvalidator import EngineCompetition


@pytest.fixture(autouse=True)
def _release_floors_off_unless_floor_test(request, monkeypatch):
    """The split/payout tests use small synthetic scores; the release entry floors (28.8 / 30.0) are tested in the floor tests."""
    if any(k in request.node.name for k in ("floor_constants", "at_or_below_its_floor")):
        return
    from reax_subnet import competition_split as _cs
    for _k in list(_cs.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(_cs.TRACK_ENTRY_FLOOR, _k, None)


SERVING = {10: 0.5, 11: 0.3, 12: 0.2}
DAY = 86400.0


def total(w):
    return math.fsum(w.values())


def test_release_constants():
    assert SPLIT_PROFILE == "50/50"  # approved by the owner on 15 Oct 2026, 10:10 UTC
    assert COMPETITIONS_START_UTC == datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
    p80, p50 = PROFILES["80/20"], PROFILES["50/50"]
    assert (p80.inference, p80.engine, p80.tracks) == (pytest.approx(0.8), 0.10, (0.10,))
    assert (p50.inference, p50.engine, p50.tracks) == (pytest.approx(0.5), 0.20, (0.15, 0.15))
    assert [t.track_id for t in TRACK_SCHEDULE][:2] == ["demand_forecasting", "risk_assessment"]


def test_profile_80_20_engine_and_one_track():
    out = combine_weights(SERVING, 1, 1.0, {"demand_forecasting": {2: 5.0, 3: 4.0, 4: 1.0}}, "80/20")
    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(0.10 * 0.75) and out[3] == pytest.approx(0.10 * 0.17)
    assert out[4] == pytest.approx(0.10 * 0.08)
    assert out[10] == pytest.approx(0.5 * 0.8) and out[12] == pytest.approx(0.2 * 0.8)
    assert total(out) == pytest.approx(1.0)


def test_profile_50_50_engine_and_two_tracks():
    tr = {"demand_forecasting": {2: 1.0}, "risk_assessment": {3: 1.0, 4: 0.5}}
    out = combine_weights(SERVING, 1, 1.0, tr, "50/50")
    assert out[1] == pytest.approx(0.20)
    assert out[2] == pytest.approx(0.15 * 0.75)
    assert out[3] == pytest.approx(0.15 * 0.75) and out[4] == pytest.approx(0.15 * 0.17)
    # unclaimed ranks (2 and 3 of the first track, 3 of the second) stay with serving
    claimed = 0.20 + 0.15 * 0.75 + 0.15 * (0.75 + 0.17)
    assert out[10] == pytest.approx(0.5 * (1 - claimed))
    assert total(out) == pytest.approx(1.0)


def test_80_20_pays_only_one_track_slot():
    tr = {"demand_forecasting": {2: 1.0}, "risk_assessment": {3: 1.0}}
    out = combine_weights(SERVING, None, 0.0, tr, "80/20")
    assert 2 in out and 3 not in out  # exactly ONE active track gets 0.10


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_no_champion_and_no_track_is_exactly_serving(profile):
    out = combine_weights(SERVING, None, 1.0, {}, profile)
    assert out == pytest.approx(SERVING) and total(out) == pytest.approx(1.0)
    assert set(out) == set(SERVING)  # nothing burned, nobody new


def test_missing_champion_share_goes_to_inference_not_to_tracks_or_owner():
    out = combine_weights({0: 0.2, 10: 0.8}, None, 0.0, {"t": {2: 1.0}}, "80/20", owner_uid=0, burn_fraction=0.2)
    # only the track's first rank (0.10 * 0.75) is paid; the owner keeps exactly its burn, nothing more
    assert out[2] == pytest.approx(0.10 * 0.75 * 0.8)
    assert out[0] == pytest.approx(0.2 + 0.0 * 0.8, abs=1e-12)
    assert total(out) == pytest.approx(1.0)


def test_owner_is_never_a_competition_payee():
    out = combine_weights({0: 0.1, 10: 0.9}, 0, 1.0, {"t": {0: 3.0}}, "80/20", owner_uid=0, burn_fraction=0.1)
    assert out == pytest.approx({0: 0.1, 10: 0.9})


@pytest.mark.parametrize("days,frac", [(0, 1.0), (7, 0.75), (14, 0.5), (30, 0.5)])
def test_engine_decay_at_0_7_14_days(days, frac):
    assert engine_decay_fraction(days * DAY) == pytest.approx(frac)
    out = combine_weights(SERVING, 1, engine_decay_fraction(days * DAY), {}, "80/20")
    assert out[1] == pytest.approx(0.10 * frac)
    assert total(out) == pytest.approx(1.0)  # the decayed part went back to serving
    assert out[10] == pytest.approx(0.5 * (1 - 0.10 * frac))


def test_track_ranks_1_2_3_and_missing_ranks():
    assert track_payout({1: 9.0, 2: 8.0, 3: 7.0, 4: 6.0}) == {1: 0.75, 2: 0.17, 3: 0.08}
    assert track_payout({1: 9.0, 2: 8.0}) == {1: 0.75, 2: 0.17}
    assert track_payout({1: 9.0}) == {1: 0.75}
    assert track_payout([(5, 2.0), (3, 2.0), (4, 1.0)]) == {3: 0.75, 5: 0.17, 4: 0.08}  # ties: lower uid first
    assert track_payout({1: float("nan"), 2: -1.0, 3: 0.0}) == {}
    assert sum((0.75, 0.17, 0.08)) == pytest.approx(1.0)
    for ranks, paid in ((1, 0.75), (2, 0.92), (3, 1.0)):
        out = combine_weights(SERVING, None, 0.0, {"t": {i: 10.0 - i for i in range(ranks)}}, "80/20")
        assert total(out) == pytest.approx(1.0)
        assert sum(out[i] for i in range(ranks)) == pytest.approx(0.10 * paid)


def test_ranked_leaders_are_paid_in_their_order_not_by_score():
    """S1 (review 16 Oct 2026): champion 101 (85, this draw), former 102 (80, this draw), older former 103 (90, an old
    draw). Re-sorting by score paid 103 the 75 %; the evaluator's order is the rank."""
    leaders = RankedLeaders([(101, 85.0), (102, 80.0), (103, 90.0)])
    assert track_payout(leaders) == {101: 0.75, 102: 0.17, 103: 0.08}
    assert track_payout(list(leaders)) == {103: 0.75, 101: 0.17, 102: 0.08}   # a plain list is still ranked by score
    # the floor still applies; the leaders after a floored one move up one rank
    assert track_payout(leaders, floor=82.0) == {101: 0.75, 103: 0.17}
    assert track_payout(leaders, floor=86.0) == {103: 0.75}
    assert track_payout(RankedLeaders([(101, float("nan")), (102, 0.0), (103, 5.0)])) == {103: 0.75}
    # a uid keeps its first place; more than three leaders: only ranks 1-3 are paid
    assert track_payout(RankedLeaders([(7, 1.0), (8, 9.0), (7, 50.0), (9, 2.0), (6, 3.0)])) == {7: 0.75, 8: 0.17, 9: 0.08}
    out = combine_weights(SERVING, None, 0.0, {"t": leaders}, "80/20")
    assert out[101] == pytest.approx(0.10 * 0.75) and out[103] == pytest.approx(0.10 * 0.08)


def test_property_sum_preserved_and_nothing_negative():
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 8)
        serving = {u: rng.random() + 0.01 for u in range(100, 100 + n)}
        s = sum(serving.values())
        owner = rng.choice([None, 100])
        burn = rng.choice([0.0, 0.1, 0.5]) if owner is not None else 0.0
        if owner is not None:
            serving = {u: w / s * (1 - burn) for u, w in serving.items()}
            serving[owner] = serving.get(owner, 0) + burn if owner in serving else burn
        tracks = {f"t{i}": {rng.randint(0, 20): rng.random() for _ in range(rng.randint(0, 5))} for i in range(rng.randint(0, 3))}
        out = combine_weights(serving, rng.choice([None, rng.randint(0, 20)]), rng.random(), tracks,
                              rng.choice(sorted(PROFILES)), owner_uid=owner, burn_fraction=burn)
        assert total(out) == pytest.approx(1.0)
        assert all(w >= 0 for w in out.values())
        if owner is not None and owner in serving:
            assert out[owner] >= burn - 1e-9  # burn is never touched, owner never loses or gains from competitions


def test_empty_serving_sets_nothing():
    assert combine_weights({}, 1, 1.0, {"t": {2: 1.0}}) == {}


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError):
        combine_weights(SERVING, None, 0.0, {}, "90/10")


def test_active_tracks_follow_the_schedule_and_profile():
    start = COMPETITIONS_START_UTC
    assert active_tracks(start - timedelta(seconds=1)) == []
    assert active_tracks(start) == ["demand_forecasting", "risk_assessment"]  # 50/50 (active): two tracks
    assert active_tracks(start, "80/20") == ["demand_forecasting"]  # 80/20 (inactive): one track
    assert active_tracks(start, "50/50") == ["demand_forecasting", "risk_assessment"]
    opening = GENERALIST_START_UTC  # the first pair ends at the generalist opening (release constant)
    assert active_tracks(opening - timedelta(seconds=1), "80/20") == ["demand_forecasting"]
    assert active_tracks(opening - timedelta(seconds=1)) == ["demand_forecasting", "risk_assessment"]
    assert active_tracks(opening) == ["generalist"]
    assert active_tracks(opening + timedelta(days=7)) == ["generalist"]


# ------------------------------------------------------------------------------------------------ EngineCompetition
D = "ab" * 32


def _snap(block=1000):
    neurons = [Neuron(0, "owner", "c0", False, 0.0, None, None),
               Neuron(3, "hk-eng", "c3", False, 0.0, None, None, commitment=f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{D}", commitment_block=5)]
    return Snapshot(block=block, neurons=tuple(neurons), owner_hotkey="owner")


class _Spec:
    def __init__(self, results):
        self.results, self.asked = results, []

    def track_results(self, track_ids, now):
        self.asked.append(list(track_ids))
        return {t: self.results[t] for t in track_ids if t in self.results}


def _comp(tmp_path, now, specialist=None, start=COMPETITIONS_START_UTC, profile="80/20"):
    comp = EngineCompetition(object(), {"s1-fast": "ghcr.io/r/ref@sha256:" + D}, str(tmp_path / "s.json"),
                             start_utc=start, profile=profile, specialist=specialist, utcnow=lambda: now[0])
    st = comp.states["s1-fast"]
    st.first_seen[D] = (5, "hk-eng")
    st.results[D] = EvalResult(digest=D, track="s1-fast", ok=True, score=1.5, block=900)
    st.champion = Champion(D, "hk-eng", 3, 1.5, 900)
    return comp


def test_before_start_time_the_weights_are_exactly_1_1(tmp_path):
    now = [COMPETITIONS_START_UTC - timedelta(seconds=1)]
    spec = _Spec({"demand_forecasting": [(7, 1.0)]})
    comp = _comp(tmp_path, now, spec)
    snap = _snap(900)
    base = {0: 0.0, 10: 0.7, 11: 0.3}
    assert not comp.active() and not comp.due()
    assert comp.weights(base, snap, {0}) == base  # identical object contents, not just close
    assert spec.asked == []
    now[0] = COMPETITIONS_START_UTC  # at the start instant the competitions are on
    assert comp.active() and comp.due()
    out = comp.weights(base, snap, {0})
    assert out[3] == pytest.approx(0.10) and out[7] == pytest.approx(0.10 * 0.75)
    assert total(out) == pytest.approx(1.0)
    assert spec.asked == [["demand_forecasting"]]


def test_engine_weights_decay_over_14_days_and_struck_champion_goes_to_serving(tmp_path):
    now = [COMPETITIONS_START_UTC + timedelta(days=1)]
    comp = _comp(tmp_path, now)
    base = {10: 1.0}
    for blocks, frac in ((0, 1.0), (50_400, 0.75), (100_800, 0.5), (500_000, 0.5)):
        out = comp.weights(base, _snap(900 + blocks), {0})
        assert out[3] == pytest.approx(0.10 * frac) and out[10] == pytest.approx(1 - 0.10 * frac)
    comp.states["s1-fast"].champion = None  # struck / vacated: everything back to inference
    assert comp.weights(base, _snap(), {0}) == base
    assert comp.weights(base, _snap(), {0}) is not base  # a copy


def test_champion_who_withdrew_is_not_paid_and_profile_50_50(tmp_path):
    now = [COMPETITIONS_START_UTC]
    comp = _comp(tmp_path, now, _Spec({"demand_forecasting": [(7, 1.0)], "risk_assessment": [(8, 1.0)]}),
                 profile="50/50")
    out = comp.weights({10: 1.0}, _snap(900), {0})
    assert out[3] == pytest.approx(0.20) and out[7] == pytest.approx(0.15 * 0.75) and out[8] == pytest.approx(0.15 * 0.75)
    gone = Snapshot(block=900, neurons=(Neuron(0, "owner", "c0", False, 0.0, None, None),), owner_hotkey="owner")
    out = comp.weights({10: 1.0}, gone, {0})
    assert 3 not in out and out[7] == pytest.approx(0.15 * 0.75)  # engine share unclaimed -> serving


def test_default_specialist_pays_nothing(tmp_path):
    assert NoSpecialistEvaluator().track_results(["demand_forecasting"], COMPETITIONS_START_UTC) == {}
    comp = _comp(tmp_path, [COMPETITIONS_START_UTC])
    out = comp.weights({10: 1.0}, _snap(900), {0})
    assert set(out) == {3, 10} and total(out) == pytest.approx(1.0)


def test_development_networks_are_active_at_once(tmp_path):
    comp = _comp(tmp_path, [datetime(2020, 1, 1, tzinfo=timezone.utc)], start=None)
    assert comp.active() and comp.due()


async def test_cycle_is_skipped_when_the_shared_gpu_is_too_full(tmp_path):
    from reax_subnet.engrunner import InfraError

    class Full:
        calls = 0

        async def preflight(self):
            self.calls += 1
            raise InfraError("GPU 0 has 1000 MiB free")

    runner = Full()
    comp = EngineCompetition(runner, {"s1-fast": "ghcr.io/r/ref@sha256:" + D}, str(tmp_path / "s.json"), start_utc=None)
    assert await comp.run_cycle(_snap(), {0}) == {}  # nothing measured, nothing recorded, no failed candidate
    assert runner.calls == 1 and comp.states["s1-fast"].results == {}


# ------------------------------------------------------------------------------------------------ keep-share rule (c)
from reax_subnet import competition_split as CS  # noqa: E402

TRACK_ID = "demand_forecasting"
T_NOW = (CS.GENERALIST_START_UTC - timedelta(days=4)).timestamp()


def _gate(approved):
    return lambda track, uid: uid in approved


def _leaders(order, crowned):
    r = RankedLeaders([(u, 50.0 - i) for i, u in enumerate(order)])
    r.crowned = dict(crowned)
    return r


def _pay(order, crowned, approved, *, keep=True, now=T_NOW, with_ts=True):
    leaders = _leaders(order, crowned) if with_ts else [(u, 50.0 - i) for i, u in enumerate(order)]
    return combine_weights(SERVING, None, 0.0, {TRACK_ID: leaders}, "50/50", owner_uid=0, award_ok=_gate(approved),
                           keep_share=keep, now=now)


def test_keep_share_flag_parsing():
    assert CS.keep_share_enabled({}) is False and CS.keep_share_enabled({"REAX_SPEC_NC_KEEP_SHARE": ""}) is False
    assert CS.keep_share_enabled({"REAX_SPEC_NC_KEEP_SHARE": "0"}) is False
    assert CS.keep_share_enabled({"REAX_SPEC_NC_KEEP_SHARE": " On "}) is True and CS.keep_share_enabled({"REAX_SPEC_NC_KEEP_SHARE": "1"}) is True
    with pytest.raises(ValueError):
        CS.keep_share_enabled({"REAX_SPEC_NC_KEEP_SHARE": "maybe"})
    assert CS.APPROVAL_GRACE_S == 48 * 3600


def test_keep_share_ranks_pure_cases():
    ranks = {20: 0.75, 21: 0.17, 22: 0.08}
    ok = lambda s: (lambda u: u in s)  # noqa: E731
    fresh = {20: T_NOW - 3600}
    assert CS.keep_share_ranks(ranks, fresh, T_NOW, ok({21})) == {20: 0.75, 22: 0.08, 21: 0.75}   # 21 takes rank 1's share, not its own
    assert CS.keep_share_ranks(ranks, fresh, T_NOW, ok({22})) == {20: 0.75, 21: 0.17, 22: 0.75}   # first approved one in rank order
    assert CS.keep_share_ranks(ranks, fresh, T_NOW, ok({21, 22}))[21] == 0.75 and 22 in CS.keep_share_ranks(ranks, fresh, T_NOW, ok({21, 22}))
    assert CS.keep_share_ranks(ranks, fresh, T_NOW, ok({20, 21})) == ranks                          # rank 1 approved: unchanged
    assert CS.keep_share_ranks(ranks, fresh, T_NOW, ok(set())) == ranks                             # nobody approved: unchanged
    assert CS.keep_share_ranks(ranks, {20: T_NOW - 48 * 3600}, T_NOW, ok({21})) == ranks            # grace over (exactly 48 h)
    assert CS.keep_share_ranks(ranks, {20: T_NOW - 48 * 3600 + 1}, T_NOW, ok({21}))[21] == 0.75     # one second before
    assert CS.keep_share_ranks(ranks, {}, T_NOW, ok({21})) == ranks and CS.keep_share_ranks(ranks, None, T_NOW, ok({21})) == ranks
    assert CS.keep_share_ranks(ranks, {20: float("nan")}, T_NOW, ok({21})) == ranks
    assert CS.keep_share_ranks(ranks, {20: T_NOW + 601}, T_NOW, ok({21})) == ranks                   # crowned in the future: unknown
    assert CS.keep_share_ranks(ranks, {20: T_NOW + 600}, T_NOW, ok({21}))[21] == 0.75               # within the clock slack
    assert CS.keep_share_ranks({20: 0.75}, fresh, T_NOW, ok(set())) == {20: 0.75}                    # a single rank: nothing to do


def test_keep_share_off_is_bit_identical():
    order, crowned = [10, 11, 12], {10: T_NOW - 3600}
    plain = combine_weights(SERVING, None, 0.0, {TRACK_ID: _leaders(order, crowned)}, "50/50", owner_uid=0, award_ok=_gate({11}))
    assert _pay(order, crowned, {11}, keep=False) == plain
    assert combine_weights(SERVING, None, 0.0, {TRACK_ID: _leaders(order, crowned)}, "50/50", owner_uid=0, award_ok=_gate({11}),
                           keep_share=True, now=None) == plain                                   # no clock: no rule
    assert combine_weights(SERVING, None, 0.0, {TRACK_ID: _leaders(order, crowned)}, "50/50", owner_uid=0, keep_share=True, now=T_NOW) \
        == combine_weights(SERVING, None, 0.0, {TRACK_ID: _leaders(order, crowned)}, "50/50", owner_uid=0)   # no gate: nothing pending
    assert _pay(order, crowned, {11}, with_ts=False) == _pay(order, crowned, {11}, keep=False, with_ts=False)   # no crowning times: off


def test_keep_share_pays_the_approved_former_champion_the_rank_1_share_for_48_hours():
    order, crowned = [20, 21, 22], {20: T_NOW - 3600}        # uids outside SERVING: their weight is the competition payout only
    off = _pay(order, crowned, {21, 22}, keep=False)
    on = _pay(order, crowned, {21, 22})
    assert 20 not in off and 20 not in on                                     # the unapproved crown is not paid either way
    assert off[21] / off[22] == pytest.approx(0.17 / 0.08) and on[21] / on[22] == pytest.approx(0.75 / 0.08)
    assert math.isclose(total(on), 1.0)
    assert off[21] > off[22] and on[21] > off[21]                             # rank 2's own 17 % is not paid on top of the 75 %
    after = _pay(order, crowned, {21, 22}, now=T_NOW + 48 * 3600)             # grace over: today's behaviour
    assert after == _pay(order, crowned, {21, 22}, keep=False, now=T_NOW + 48 * 3600) == off
    assert _pay(order, crowned, {20, 21, 22}) == _pay(order, crowned, {20, 21, 22}, keep=False)   # approval ends it at once
    assert _pay(order, crowned, {21, 22}) == on                               # deterministic


def test_keep_share_leaves_the_other_track_alone():
    r = {TRACK_ID: _leaders([20, 21], {20: T_NOW - 60}), "risk_assessment": _leaders([22, 23], {22: T_NOW - 10 * DAY})}
    gate = _gate({21, 23})
    base = combine_weights(SERVING, None, 0.0, r, "50/50", owner_uid=0, award_ok=gate)
    on = combine_weights(SERVING, None, 0.0, r, "50/50", owner_uid=0, award_ok=gate, keep_share=True, now=T_NOW)
    assert base[21] / base[23] == pytest.approx(1.0)                          # both pay rank 2 (17 %) of an equal track slot
    assert on[21] / on[23] == pytest.approx(0.75 / 0.17)                      # df: 21 takes the fresh crown's 75 %; risk: unchanged
    assert 20 not in on and 22 not in on


def test_ranked_leaders_crowning_times_survive_the_copy_and_come_from_the_state():
    from reax_subnet.competitions import _copy
    r = _leaders([1, 2], {1: 5.0})
    c = _copy(r)
    assert isinstance(c, RankedLeaders) and c.crowned == {1: 5.0} and c.crowned is not r.crowned
    assert _copy([(1, 2.0)]) == [(1, 2.0)]
