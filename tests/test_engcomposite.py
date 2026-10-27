"""Composite engine track (engine competition v2, unreleased; docs/ENGINE-COMPETITION-V2.md): declaration label,
composite score and its gates, the champion rule on composite scores, real measurements of the synthetic engines
through CompositeRunner, and the validator cycle."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import sys

import pytest

from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.engcomp import EvalResult, Submission, TrackParams, TrackState, observe, update_champion
from reax_subnet.engcomposite import (
    COMPOSITE_PROTOCOL_TRACK, CompositeRunner, CompositeTrack, ModelSlot, composite_rules, composite_rules_sha256, composite_score,
    default_track, evaluate_composite, parse_declared, to_eval_result,
)
from reax_subnet.engrunner import DevRunner, InfraError, ReferenceError_, ReferenceFailedWithCandidate
from reax_subnet.engvalidator import EngineCompetition

from test_engcomp import FAKE, FAST_P, ref_of

D1, D2, D3 = (hashlib.sha256(x.encode()).hexdigest() for x in ("c1", "c2", "c3"))


@pytest.fixture(autouse=True)
def _copyline_not_started(monkeypatch):
    """These tests are about the composite rules, not the copy line (ENGINE_COPYLINE_START_UTC, 23 Oct 13:00Z, which is in the
    past now: with it on, an entry is crownable only with a review record read from runner.review_dir, and the fake runners
    here have none). Freezing the gate keeps them independent of the wall clock; the copy line on a composite track is
    tested at the end of this file with a real review directory."""
    from reax_subnet import engcomp

    monkeypatch.setattr(engcomp, "ENGINE_COPYLINE_START_UTC", None)


def _slot(name, weight, noise=0.005, mem=0, params=None):
    p = params or TrackParams(name, pool="s1-fast", temperature=2.07, reference_noise_std=noise)
    return ModelSlot(name, name, weight, p, mem, calibrated=True)


TRACK = CompositeTrack("dm-all", (_slot("a", 0.1), _slot("b", 0.2), _slot("c", 0.2), _slot("d", 0.5)), max_evals=4)
ALL = frozenset("abcd")


def _r(score, ok=True, reason="", mem=-1, items=10):
    return EvalResult(digest="x", track="t", ok=ok, reason=reason, score=score, items=items, cand_memory_mib=mem)


# ------------------------------------------------------------------------------------------------ declaration
def test_declared_models_label():
    assert parse_declared(None, TRACK) == ALL  # no label: the image claims every model
    assert parse_declared(" All ", TRACK) == ALL
    assert parse_declared("d, b", TRACK) == {"b", "d"}
    for bad in ("", "d,,b", "d,x", 7, "a" * 300):
        with pytest.raises(ValueError):
            parse_declared(bad, TRACK)


def test_track_definition_is_checked():
    with pytest.raises(ValueError, match="sum to 1"):
        CompositeTrack("t", (_slot("a", 0.5), _slot("b", 0.6)))
    with pytest.raises(ValueError, match="unique"):
        CompositeTrack("t", (_slot("a", 0.5), _slot("a", 0.5)))
    with pytest.raises(ValueError):
        CompositeTrack("t", (_slot("all", 1.0),))


# ------------------------------------------------------------------------------------------------ the score
def test_composite_is_the_weighted_geometric_mean():
    res = {"a": _r(1.2), "b": _r(1.1), "c": _r(1.0), "d": _r(1.5)}
    cs = composite_score(TRACK, ALL, res)
    want = math.exp(0.1 * math.log(1.2) + 0.2 * math.log(1.1) + 0.5 * math.log(1.5))
    assert cs.ok and cs.score == pytest.approx(want)
    assert cs.parts["d"]["declared"] and cs.parts["d"]["score"] == 1.5


def test_undeclared_models_count_as_the_reference_and_are_not_needed():
    cs = composite_score(TRACK, frozenset({"d"}), {"d": _r(1.21)})
    assert cs.ok and cs.score == pytest.approx(1.21 ** 0.5)
    assert cs.parts["a"] == {"model": "a", "weight": 0.1, "declared": False, "floor": cs.parts["a"]["floor"], "score": 1.0}
    assert not composite_score(TRACK, frozenset(), {}).ok


def test_one_failed_or_missing_model_disqualifies_the_whole_image():
    good = {"a": _r(1.2), "b": _r(1.2), "c": _r(1.2), "d": _r(3.0)}
    for slot, bad in (("b", _r(0.0, ok=False, reason="correctness: mean distance 0.300 > 0.1")),
                      ("c", None)):
        res = dict(good)
        res[slot] = bad
        cs = composite_score(TRACK, ALL, res)
        assert not cs.ok and cs.score == 0.0 and cs.reason.startswith(slot + ":")


def test_no_regression_floor_stops_winning_on_one_model_while_losing_another():
    # +50 % on d3 would carry a 0.90x model in a plain mean (0.90^0.2 * 1.5^0.5 = 1.19); the floor forbids it
    res = {"a": _r(1.0), "b": _r(0.90), "c": _r(1.0), "d": _r(1.5)}
    cs = composite_score(TRACK, ALL, res)
    assert not cs.ok and "below the no-regression floor" in cs.reason and cs.reason.startswith("b:")
    # inside the noise floor (1 - max(2 %, 3 sigma)) a model may be a little slower
    res["b"] = _r(0.985)
    assert composite_score(TRACK, ALL, res).ok
    noisy = CompositeTrack("t", (_slot("a", 0.5, noise=0.02), _slot("b", 0.5)))
    assert noisy.slot("a").floor == pytest.approx(0.94)  # 3 x 2 % noise


def test_memory_budget_per_model():
    track = CompositeTrack("t", (_slot("a", 0.5, mem=1000), _slot("b", 0.5)))
    ok = composite_score(track, frozenset("ab"), {"a": _r(1.2, mem=1000), "b": _r(1.2, mem=99999)})
    assert ok.ok  # b has no budget
    over = composite_score(track, frozenset("ab"), {"a": _r(1.2, mem=1001), "b": _r(1.2)})
    assert not over.ok and "GPU memory 1001 MiB > budget 1000 MiB" in over.reason


def test_composite_noise_and_margin():
    t = CompositeTrack("t", (_slot("a", 0.5, noise=0.02), _slot("b", 0.5, noise=0.02)))
    assert t.noise() == pytest.approx(math.sqrt(2) * 0.01)
    p = t.params()
    assert p.reference_noise_std == pytest.approx(t.noise()) and p.min_gain == 0.03 and p.confirmations == 2
    big = CompositeTrack("t", (_slot("a", 1.0, noise=0.02),))
    from reax_subnet.engconfigs import required_engine_margin

    assert required_engine_margin(big.params().reference_noise_std) == pytest.approx(0.06)


def test_rules_hash_changes_with_any_rule():
    h = composite_rules_sha256(TRACK)
    assert h == composite_rules_sha256(CompositeTrack("dm-all", TRACK.slots))
    moved = CompositeTrack("dm-all", (_slot("a", 0.2), _slot("b", 0.1), _slot("c", 0.2), _slot("d", 0.5)))
    assert composite_rules_sha256(moved) != h
    stricter = dataclasses.replace(TRACK.slots[0], params=dataclasses.replace(TRACK.slots[0].params, mean_tau=0.05))
    assert composite_rules_sha256(CompositeTrack("dm-all", (stricter,) + TRACK.slots[1:])) != h


def test_default_track_is_not_release_ready_until_calibrated(monkeypatch):
    from reax_subnet import engcomposite as ec

    monkeypatch.setattr(ec, "CALIBRATED_ON_RELEASE_REFERENCES", False)  # the pre-release state
    t = default_track(variant="approved-1021")
    assert t.name == COMPOSITE_PROTOCOL_TRACK and [s.name for s in t.slots] == ["plumb", "blink", "reax1", "pplx", "d3"]
    assert math.fsum(s.weight for s in t.slots) == pytest.approx(1.0)
    assert len(t.release_ready()) == 5
    cal = {n: {"noise": 0.004, "temperature": 1.0} for n in ("plumb", "blink", "reax1", "pplx", "d3")}
    ready = default_track(cal, variant="approved-1021")
    assert ready.release_ready() == [] and ready.slot("d3").params.reference_noise_std == 0.004
    assert ready.slot("d3").params.pool == "s1-pro" and ready.slot("plumb").params.pool == "s1-fast"
    gated = default_track({"d3": {"gates": {"mean_tau": 0.15}}}, variant="approved-1021")
    assert gated.slot("d3").params.mean_tau == 0.15 and gated.slot("d3").calibrated and not gated.slot("pplx").calibrated
    assert t.carried_declaration("s1-fast") == {"plumb"} and t.carried_declaration("dm-all") is None
    with pytest.raises(ValueError, match="carried"):
        CompositeTrack("t", (_slot("a", 1.0),), carried=(("s1-fast", ("zzz",)),))


# ------------------------------------------------------------------------------------------------ champion rule
def _comp_result(digest, scores, session, block=100):
    cs = composite_score(TRACK, frozenset(scores), {k: _r(v) for k, v in scores.items()})
    r = to_eval_result(TRACK, digest, block, cs, {k: _r(v) for k, v in scores.items()})
    r.session = session
    return r


def test_champion_rule_runs_unchanged_on_composite_scores():
    params = TRACK.params()
    subs = [Submission(1, "hk1", "dm-all", "ghcr.io/x/e", D1, block=10),
            Submission(2, "hk2", "dm-all", "ghcr.io/x/f", D2, block=20)]
    st = TrackState()
    observe(st, subs, block=100)
    # a d3-only specialist: +8 % on d3 alone = 1.039 composite, clears the 3 % gain; needs a second session
    st.results[D1] = _comp_result(D1, {"d": 1.08}, "s1")
    assert update_champion(st, subs, params, block=100) is None and st.challenger["digest"] == D1
    st.results[D1] = _comp_result(D1, {"d": 1.07}, "s2")
    ch = update_champion(st, subs, params, block=101)
    assert ch is not None and ch.digest == D1 and ch.score == pytest.approx(1.07 ** 0.5)  # held at the lower session
    # a broad engine: +4 % on every model = 1.04 composite, below champion x 1.03 = 1.0656 -> no challenger
    st.results[D2] = _comp_result(D2, {"a": 1.04, "b": 1.04, "c": 1.04, "d": 1.04}, "s3")
    update_champion(st, subs, params, block=102)
    assert st.champion.digest == D1 and st.challenger is None
    # +12 % everywhere = 1.12 >= 1.0656: challenger, crowned after its confirmation
    st.results[D2] = _comp_result(D2, {"a": 1.12, "b": 1.12, "c": 1.12, "d": 1.12}, "s4")
    update_champion(st, subs, params, block=103)
    st.results[D2] = _comp_result(D2, {"a": 1.11, "b": 1.12, "c": 1.12, "d": 1.12}, "s5")
    assert update_champion(st, subs, params, block=104).digest == D2


def test_a_composite_result_with_a_failed_model_never_counts():
    params = TRACK.params()
    subs = [Submission(1, "hk1", "dm-all", "ghcr.io/x/e", D1, block=10)]
    st = TrackState()
    observe(st, subs, block=100)
    res = {"a": _r(2.0), "b": _r(2.0), "c": _r(0, ok=False, reason="determinism"), "d": _r(2.0)}
    r = to_eval_result(TRACK, D1, 100, composite_score(TRACK, ALL, res), res)
    r.session = "s1"
    st.results[D1] = r
    assert update_champion(st, subs, params, block=100) is None and st.challenger is None
    assert st.results[D1].to_json()["parts"]["c"]["reason"] == "determinism"
    assert EvalResult.from_json(json.loads(json.dumps(st.results[D1].to_json()))).parts == r.parts


# ------------------------------------------------------------------------------------------------ the session loop
class _Calls:
    def __init__(self, scores, raise_on=None):
        self.scores, self.raise_on, self.calls, self.finished = scores, raise_on or {}, [], []

    async def __call__(self, runner, reference, ref, track, params, block=0, digest="", progress=None):
        self.calls.append(params.name)
        if params.name in self.raise_on:
            raise self.raise_on[params.name]
        s = self.scores[params.name]
        return EvalResult(digest=digest, track=track, ok=s > 0, reason="" if s > 0 else "correctness", score=s,
                          block=block)


class _Runner:
    def __init__(self, alone_ok=True):
        self.finished, self.alone_ok = [], alone_ok

    async def finish_candidate(self, ref):
        self.finished.append(ref)


async def test_session_loop_order_fail_fast_and_cleanup(monkeypatch):
    import reax_subnet.engrunner as er

    fake = _Calls({"a": 1.1, "b": 0.0, "c": 1.2, "d": 1.3})
    monkeypatch.setattr(er, "evaluate", fake)
    runner = _Runner()
    res = await evaluate_composite(runner, "ref", "img@sha256:" + D1, TRACK, ALL, block=7, digest=D1)
    assert fake.calls == ["a", "b"]  # stops at the first disqualifying model (saves the GPU time of the big ones)
    assert not res.ok and res.reason.startswith("b:") and res.track == "dm-all" and res.block == 7
    assert runner.finished == ["img@sha256:" + D1]
    fake2 = _Calls({"a": 1.1, "b": 1.1, "c": 1.2, "d": 1.3})
    monkeypatch.setattr(er, "evaluate", fake2)
    res = await evaluate_composite(runner, "ref", "img", TRACK, frozenset({"b", "d"}), digest=D1)
    assert fake2.calls == ["b", "d"] and res.ok and res.score == pytest.approx(1.1 ** 0.2 * 1.3 ** 0.5)


async def test_reference_failure_with_candidate_is_decided_per_model(monkeypatch):
    import reax_subnet.engrunner as er

    seen = []

    async def alone(runner, ref, params):
        seen.append(params.name)
        return runner.alone_ok

    monkeypatch.setattr(er, "reference_alone_ok", alone)
    monkeypatch.setattr(er, "evaluate", _Calls({"a": 1.1}, raise_on={"b": ReferenceFailedWithCandidate("timeout")}))
    runner = _Runner(alone_ok=True)
    with pytest.raises(InfraError, match="b: reference failed while this candidate was loaded"):
        await evaluate_composite(runner, "ref", "img", TRACK, ALL)
    assert seen == ["b"] and runner.finished == ["img"]
    monkeypatch.setattr(er, "evaluate", _Calls({"a": 1.1}, raise_on={"b": ReferenceFailedWithCandidate("timeout")}))
    with pytest.raises(ReferenceError_, match="broken"):
        await evaluate_composite(_Runner(alone_ok=False), "ref", "img", TRACK, ALL)


# ------------------------------------------------------------------------------------------------ real fake engines
def _two_model_runner(tmp_path):
    variants = ("reference", "fast", "cache", "broken")
    mapping = {ref_of(v): [sys.executable, FAKE, "--variant", v, "--delay", "0.01", "--jitter", "0.4"] for v in variants}
    return CompositeRunner({m: DevRunner(str(tmp_path / m), mapping) for m in ("m-a", "m-b")})


def _fast_track():
    def slot(name, w):
        return ModelSlot(name, name, w, dataclasses.replace(FAST_P, name=name), 0, calibrated=True)
    return CompositeTrack("dm-all", (slot("m-a", 0.4), slot("m-b", 0.6)), confirmations=1)


async def test_real_sessions_on_two_models(tmp_path):
    runner, track = _two_model_runner(tmp_path), _fast_track()
    both = frozenset({"m-a", "m-b"})
    fast = await evaluate_composite(runner, ref_of("reference"), ref_of("fast"), track, both, digest=D1)
    assert fast.ok and 1.4 < fast.score < 2.6, fast
    assert set(fast.parts) == {"m-a", "m-b"} and all(p["mean_distance"] == 0.0 for p in fast.parts.values())
    cache = await evaluate_composite(runner, ref_of("reference"), ref_of("cache"), track, both, digest=D2)
    assert not cache.ok and cache.reason.startswith("m-a: anti-cache probe")
    assert "score" not in cache.parts["m-b"]  # never measured: the first model already disqualified it


# ------------------------------------------------------------------------------------------------ validator cycle
def _snapshot(entries, block):
    neurons = tuple(Neuron(i, f"hk{i}", f"c{i}", False, 0.0, None, None,
                           commitment=f"reax-eng/1:{track}:ghcr.io/x/e@sha256:{d}", commitment_block=1 + i)
                    for i, (track, d) in enumerate(entries))
    return Snapshot(block, neurons, None)


class _LabelRunner:
    def __init__(self, labels):
        self.labels = labels

    async def models_label(self, ref):
        return self.labels.get(ref.rsplit(":", 1)[-1])


async def test_validator_crowns_on_the_composite_and_ignores_v1_commitments(tmp_path, monkeypatch):
    import reax_subnet.engvalidator as ev

    calls = []

    async def fake_composite(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        calls.append((digest, tuple(sorted(declared))))
        scores = {D1: 1.20, D2: 1.01}[digest]
        res = {s.name: _r(scores) for s in track.slots if s.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    monkeypatch.setattr(ev, "evaluate_composite", fake_composite)
    comp = EngineCompetition(_LabelRunner({D1: "a,d", D3: "d,zzz"}), {"dm-all": "ref"}, str(tmp_path / "eng.json"),
                             composites={"dm-all": TRACK}, start_utc=None)
    assert set(comp.tracks) == {"dm-all"} and comp.tracks["dm-all"].reference_noise_std == pytest.approx(TRACK.noise())
    await comp.run_cycle(_snapshot([("dm-all", D1), ("dm-all", D2), ("s1-fast", D1), ("dm-all", D3)], 500), set())
    st = comp.states["dm-all"]
    assert (D1, ("a", "d")) in calls and (D2, ("a", "b", "c", "d")) in calls  # label read; no label = all models
    assert not st.results[D3].ok and "unknown models: zzz" in st.results[D3].reason  # a bad label fails, unmeasured
    assert st.champion is not None and st.champion.digest == D1  # two sessions in one cycle
    assert st.champion.score == pytest.approx(1.2 ** 0.6)
    saved = json.load(open(tmp_path / "eng.json"))
    assert saved["thresholds_sha256"] == comp.thresholds_sha256
    # migration: a v1 s1-fast commitment is a Plumb-only entry of the composite (no new commitment, label ignored)
    carry = CompositeTrack("dm-all", TRACK.slots, carried=(("s1-fast", ("a",)),))
    calls.clear()
    mig = EngineCompetition(_LabelRunner({D1: "a,d"}), {"dm-all": "ref"}, str(tmp_path / "m.json"),
                            composites={"dm-all": carry}, start_utc=None)
    await mig.run_cycle(_snapshot([("s1-fast", D1), ("s1-pro", D2)], 600), set())
    st = mig.states["dm-all"]
    assert calls == [(D1, ("a",))] and st.results[D1].ok and st.results[D1].score == pytest.approx(1.2 ** 0.1)
    assert st.challenger is None  # 1.2x on Plumb alone = +1.8 % composite: below the 3 % gain, no confirmation session
    assert mig.thresholds_sha256 != comp.thresholds_sha256
    other = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "e2.json"),
                              composites={"dm-all": CompositeTrack("dm-all", (_slot("a", 1.0),))}, start_utc=None)
    assert other.thresholds_sha256 != comp.thresholds_sha256  # other per-model rules = another protocol


# ------------------------------------------------------------------------------------------------ critic fixes (critic1)
async def test_memory_peak_is_sampled_and_a_failed_reading_is_ours(monkeypatch):
    import reax_subnet.engrunner as er

    monkeypatch.setattr(er, "MEM_SAMPLE_S", 0.001)
    seq = iter([1000, 1500, 1500])  # a teardown of the previous model settles, then two equal readings

    async def read():
        return next(seq)

    assert await er._stable_mib(read) == 1500
    values = [2000, 9000, 2100]  # the engine frees memory after its peak: still charged the peak

    async def read2():
        return values.pop(0) if len(values) > 1 else values[0]

    s = er._PeakSampler(read2)
    s.start()
    import asyncio

    await asyncio.sleep(0.05)
    assert await s.stop() == 9000

    async def broken():
        return None

    assert await er._stable_mib(broken) is None
    s = er._PeakSampler(broken)
    s.start()
    await asyncio.sleep(0.05)
    assert await s.stop() is None
    flaky = [3000, None, 3500, None, 3100]  # isolated misses are tolerated

    async def read3():
        return flaky.pop(0) if len(flaky) > 1 else flaky[0]

    s = er._PeakSampler(read3)
    for _ in range(4):
        await s._sample()
    assert await s.stop() == 3500


async def test_unreadable_memory_makes_the_composite_inconclusive(monkeypatch):
    import reax_subnet.engrunner as er

    class Unread(_Calls):
        async def __call__(self, *a, **k):
            r = await super().__call__(*a, **k)
            r.cand_memory_mib = None
            return r

    monkeypatch.setattr(er, "evaluate", Unread({"a": 1.1}))
    budgeted = CompositeTrack("dm-all", tuple(dataclasses.replace(s, memory_budget_mib=1000) for s in TRACK.slots))
    with pytest.raises(InfraError, match="GPU memory could not be read"):
        await evaluate_composite(_Runner(), "ref", "img", budgeted, ALL, digest=D1)
    monkeypatch.setattr(er, "evaluate", _Calls({"a": 1.1}))  # a runner that cannot read memory at all (-1)
    with pytest.raises(InfraError, match="GPU memory could not be read"):
        await evaluate_composite(_Runner(), "ref", "img", budgeted, ALL, digest=D1)


async def test_per_model_budget_fail_vs_inconclusive(monkeypatch):
    import asyncio

    import reax_subnet.engrunner as er

    async def hang(runner, reference, ref, track, params, block=0, digest="", progress=None):
        if params.name == "b":
            progress["measuring"] = True  # inside b's timed phase: a slow candidate
        await asyncio.sleep(10)

    tiny = CompositeTrack("dm-all", tuple(dataclasses.replace(s, params=dataclasses.replace(s.params, eval_budget_s=0.05))
                                          for s in TRACK.slots))
    monkeypatch.setattr(er, "evaluate", hang)
    with pytest.raises(InfraError, match="before the first timed request"):  # model a never got to measure
        await evaluate_composite(_Runner(), "ref", "img", tiny, ALL, digest=D1)
    res = await evaluate_composite(_Runner(), "ref", "img", tiny, frozenset({"b", "d"}), digest=D1)
    assert not res.ok and res.reason == "b: evaluation exceeded its 0 s budget"


def test_required_models_and_runner_gates():
    req = CompositeTrack("dm-all", TRACK.slots, required=("d",))
    assert not composite_score(req, frozenset({"a"}), {"a": _r(2.0)}).ok
    assert composite_score(req, frozenset({"d"}), {"d": _r(1.1)}).ok
    with pytest.raises(ValueError):
        CompositeTrack("dm-all", TRACK.slots, required=("zzz",))

    class R:
        def __init__(self, mode):
            self.release_mode, self.docker_runtime, self.review_dir = mode, "runsc", "/r"

    CompositeRunner({"a": R(True), "b": R(True)})
    with pytest.raises(ValueError, match="release_mode"):
        CompositeRunner({"a": R(True), "b": R(False)})  # a non-first runner without the review gate


async def test_required_model_missing_is_rejected_without_measuring(monkeypatch):
    import reax_subnet.engrunner as er

    fake = _Calls({"a": 2.0})
    monkeypatch.setattr(er, "evaluate", fake)
    req = CompositeTrack("dm-all", TRACK.slots, required=("d",))
    res = await evaluate_composite(_Runner(), "ref", "img", req, frozenset({"a"}), digest=D1)
    assert not res.ok and "must declare d" in res.reason and fake.calls == []


async def test_dm_all_champion_is_paid_through_the_award_gate(tmp_path, monkeypatch):
    import reax_subnet.engvalidator as ev

    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        res = {s.name: _r(1.3) for s in track.slots if s.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    class Gate:
        def __init__(self):
            self.asked, self.ok = [], False

        def approved(self, track, hotkey, sid):
            self.asked.append((track, hotkey, sid))
            return self.ok

    monkeypatch.setattr(ev, "evaluate_composite", fake)
    gate = Gate()
    comp = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "e.json"),
                             composites={"dm-all": TRACK}, start_utc=None, award_gate=gate)
    snap = _snapshot([("dm-all", D1)], 500)
    await comp.run_cycle(snap, set())
    base = {5: 1.0}
    assert comp.weights(base, snap, set()) == base  # not approved yet: nothing moves
    assert gate.asked[-1] == ("dm-all", "hk0", "sha256:" + D1)
    gate.ok = True
    w = comp.weights(base, snap, set())
    assert w[0] > 0 and w[5] < 1.0


def test_award_cli_accepts_the_composite_track():
    from reax_subnet.award import ENGINE_TRACK_IDS

    assert "dm-all" in ENGINE_TRACK_IDS and "s1-fast" in ENGINE_TRACK_IDS


def test_composite_from_env_refuses_an_uncalibrated_track_on_finney(tmp_path, monkeypatch):
    from reax_subnet import engcomposite as ec
    from reax_subnet.config import ConfigError
    from reax_subnet.worker import composite_track_from_env

    monkeypatch.setattr(ec, "CALIBRATED_ON_RELEASE_REFERENCES", False)  # the pre-release state
    with pytest.raises(ConfigError, match="not release-ready"):
        composite_track_from_env({}, release_mode=True)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({n: {"noise": 0.004} for n in ("blink", "vision", "quyet", "reax1", "pplx", "d3", "mammoth", "jeffan")}))
    with pytest.raises(ConfigError, match="not allowed on finney"):
        composite_track_from_env({"REAX_ENGINE_COMPOSITE_CALIBRATION": str(cal)}, release_mode=True)
    dev = composite_track_from_env({"REAX_ENGINE_COMPOSITE_CALIBRATION": str(cal)}, release_mode=False)
    assert [r for r in dev.release_ready() if "calibrated" in r] == []  # only the unpublished reference digests remain


async def test_remote_validator_pays_the_workers_composite_champion(tmp_path, monkeypatch):
    import reax_subnet.engvalidator as ev
    from reax_subnet.remote import RemoteEngineCompetition

    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        res = {s.name: _r(1.3) for s in track.slots if s.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    monkeypatch.setattr(ev, "evaluate_composite", fake)
    path = str(tmp_path / "engine-competition.json")
    worker = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, path, composites={"dm-all": TRACK}, start_utc=None)
    snap = _snapshot([("dm-all", D1)], 500)
    await worker.run_cycle(snap, set())
    remote = RemoteEngineCompetition(path, composites={"dm-all": TRACK}, start_utc=None)
    assert set(remote.tracks) == {"dm-all"}
    w = remote.weights({5: 1.0}, snap, set())
    assert w.get(0, 0) > 0


async def test_composite_cycle_takes_at_most_its_max_evals(tmp_path, monkeypatch):
    import reax_subnet.engvalidator as ev

    seen = []

    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        seen.append(digest)
        res = {s.name: _r(1.0) for s in track.slots if s.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    monkeypatch.setattr(ev, "evaluate_composite", fake)
    two = dataclasses.replace(TRACK, max_evals=2)
    comp = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "e.json"),
                             composites={"dm-all": two}, start_utc=None, max_evals=4)
    await comp.run_cycle(_snapshot([("dm-all", D1), ("dm-all", D2), ("dm-all", D3)], 500), set())
    assert len(seen) == 2 and default_track().max_evals == 2


def test_per_model_reference_images(tmp_path, monkeypatch):
    from reax_subnet.config import ENGINE_COMPOSITE_REFERENCE_IMAGES, ENGINE_REFERENCE_IMAGE, ConfigError
    from reax_subnet.worker import composite_references

    for k in ("blink", "reax1", "pplx", "d3"):  # unpublished references: the track cannot open
        monkeypatch.setitem(ENGINE_COMPOSITE_REFERENCE_IMAGES, k, None)
    monkeypatch.setattr("reax_subnet.engcomposite.COMPOSITE_WEIGHTS_VARIANT", "v22")  # six slots incl. quyet, vision
    assert ENGINE_COMPOSITE_REFERENCE_IMAGES["plumb"] == ENGINE_REFERENCE_IMAGE  # v1's reference carries over
    with pytest.raises(ConfigError, match="missing"):
        composite_references({}, release_mode=True)  # not published yet: the track cannot open
    pinned = ",".join(f"{n}=ghcr.io/reaxlabs/x-{n}@sha256:{D1}" for n in ("blink", "reax1", "pplx", "d3", "quyet", "vision"))
    with pytest.raises(ConfigError, match="not allowed on finney"):
        composite_references({"REAX_ENGINE_COMPOSITE_REFS": pinned}, release_mode=True)
    refs = composite_references({"REAX_ENGINE_COMPOSITE_REFS": pinned}, release_mode=False)
    assert refs["d3"].endswith(D1) and refs["plumb"] == ENGINE_REFERENCE_IMAGE
    with pytest.raises(ConfigError, match="pinned"):
        composite_references({"REAX_ENGINE_COMPOSITE_REFS": pinned + ",d3=ghcr.io/x/y:latest"}, release_mode=False)


async def test_validator_passes_per_model_references(tmp_path, monkeypatch):
    import reax_subnet.engvalidator as ev

    seen = {}

    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        seen.update(reference=reference, references=references)
        res = {s.name: _r(1.0) for s in track.slots if s.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    monkeypatch.setattr(ev, "evaluate_composite", fake)
    comp = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "e.json"), composites={"dm-all": TRACK},
                             start_utc=None, slot_references={"dm-all": {"d": "ref-d"}})
    await comp.run_cycle(_snapshot([("dm-all", D1)], 500), set())
    assert seen == {"reference": "ref", "references": {"d": "ref-d"}}
    assert comp.context["references"]["dm-all"] == {"default": "ref", "d": "ref-d"}
    plain = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "p.json"), composites={"dm-all": TRACK},
                              start_utc=None)
    assert plain.context_sha256 != comp.context_sha256  # another reference = scores must be re-measured


def test_default_track_carries_the_measured_gates():
    t = default_track(variant="approved-1021")
    d3, pplx, plumb = t.slot("d3").params, t.slot("pplx").params, t.slot("plumb").params
    assert (plumb.mean_tau, plumb.min_top_agreement, plumb.determinism_tau) == (0.10, 0.99, 0.05)  # v1 gates
    # release calibration (23 Oct): worst honest pair mean d3 0.062 / probe 0.134, pplx 0.063, reax1 0.092, blink 0
    assert pplx.min_top_agreement < 0.9805 and pplx.mean_tau > 0.063  # an honest copy of the reference passes
    assert (pplx.mean_tau, pplx.item_tau, pplx.min_top_agreement, pplx.determinism_tau) == (0.13, 1.1, 0.98, 0.8)
    assert pplx.reference_noise_std == 0.0051 and t.slot("pplx").memory_budget_mib == 39_936
    assert d3.concurrency == 8 and d3.timed_items == 1024 and d3.determinism_tau == 1.1 and d3.item_tau == 1.1
    assert d3.mean_tau == 0.17 and d3.mean_tau > 0.134 > 0.062 and d3.min_top_agreement == 0.97  # probe 0.134 passes
    bl = t.slot("blink").params
    assert (bl.mean_tau, bl.item_tau, bl.min_top_agreement, bl.determinism_tau) == (0.08, 1.6, 0.97, 0.05)
    # hotfix 26 Oct: the honest start-to-start drift of the Blink reference itself (gpucheck3: mean 0.042, item 1.125, top
    # 98.83 %; raw pair of fresh starts 98.44 %) must pass, with headroom; the old gates (0.05 / 1.0 / 99 %) failed it
    assert 0.042 * 1.5 < bl.mean_tau and 1.125 * 1.4 < bl.item_tau and bl.min_top_agreement <= 0.9844 - 0.01
    assert bl.concurrency == 16 and bl.reference_noise_std == 0.0751 and t.slot("blink").memory_budget_mib == 52_224
    # REAX 1 (23 Oct release calibration): worst honest pair mean 0.092 / item 0.53 (session tail 1.325) / top 98.8 %
    r1 = t.slot("reax1").params
    assert r1.mean_tau > 0.092 and r1.item_tau > 1.325 and r1.min_top_agreement < 0.988 and r1.determinism_tau == 1.15
    assert r1.concurrency == 8 and r1.timed_items == 1024 and r1.reference_noise_std == 0.0325
    assert t.slot("reax1").memory_budget_mib == 39_936 and t.slot("reax1").floor == pytest.approx(1 - 3 * 0.0325)
    assert t.budget_s() == 8400.0 and t.max_evals == 2  # 900 slack + plumb 1200 + blink 1200 + reax1 1500 + pplx 1200 + d3 2400


# ------------------------------------------------------------------------------------------------ REAX 1 slot (21 Oct 2026)
REAX1_WEIGHTS = {"reax1": 0.40, "d3": 0.30, "blink": 0.15, "pplx": 0.10, "plumb": 0.05}


def test_default_weights_put_reax1_first():
    t = default_track(variant="approved-1021")
    assert {s.name: s.weight for s in t.slots} == REAX1_WEIGHTS
    assert math.fsum(s.weight for s in t.slots) == pytest.approx(1.0)
    assert max(t.slots, key=lambda s: s.weight).name == "reax1"
    r1 = t.slot("reax1")
    assert r1.model == "reax-1" and r1.params.pool == "s1-pro" and r1.params.concurrency == 8
    assert r1.params.temperature == pytest.approx(1.9754)  # decision_config.json of decisionmodels/reax-1@ca2a0cc5
    assert parse_declared("reax1,d3", t) == {"reax1", "d3"}
    assert parse_declared(None, t) == {"plumb", "blink", "reax1", "pplx", "d3"}
    with pytest.raises(ValueError):
        parse_declared("reax-1", t)  # the slot is "reax1"; the model id is not a label name


def test_reax1_composite_math_with_five_slots():
    t = default_track(variant="approved-1021")
    every = frozenset(s.name for s in t.slots)
    gains = {"reax1": 1.5, "d3": 1.2, "blink": 1.1, "pplx": 1.05, "plumb": 1.0}
    cs = composite_score(t, every, {k: _r(v) for k, v in gains.items()})
    want = math.exp(sum(REAX1_WEIGHTS[k] * math.log(v) for k, v in gains.items()))
    assert cs.ok and cs.score == pytest.approx(want)
    # a REAX 1-only image: 1.0x everywhere else, so the composite is s ** 0.40 (needs ~1.0777x for a 3 % gain)
    only = composite_score(t, frozenset({"reax1"}), {"reax1": _r(1.5)})
    assert only.ok and only.score == pytest.approx(1.5 ** 0.40)
    # a regression on the 0.05 slot still disqualifies the image (the floor is per model, not weighted)
    bad = composite_score(t, every, {**{k: _r(2.0) for k in gains}, "plumb": _r(0.5)})
    assert not bad.ok


def test_reax1_is_in_the_frozen_rules_hash_and_v1_entries_stay_plumb_only():
    t = default_track(variant="approved-1021")
    rules = composite_rules(t)
    assert [x["name"] for x in rules["slots"]] == ["plumb", "blink", "reax1", "pplx", "d3"]
    assert {x["name"]: x["weight"] for x in rules["slots"]} == REAX1_WEIGHTS
    assert rules["slots"][2]["model"] == "reax-1"
    h = composite_rules_sha256(t)
    assert h == composite_rules_sha256(default_track(variant="approved-1021"))  # deterministic
    # moving a weight, the model id or the REAX 1 gates is a new release (new hash)
    rest = [s for s in t.slots if s.name != "reax1"]
    four = CompositeTrack(t.name, tuple(dataclasses.replace(s, weight=w) for s, w in zip(rest, (0.1, 0.2, 0.2, 0.5))),
                          carried=t.carried)  # the 20 Oct four-slot weight set
    assert composite_rules_sha256(four) != h
    r1 = t.slot("reax1")
    renamed = dataclasses.replace(r1, model="reax-2")
    assert composite_rules_sha256(CompositeTrack(t.name, tuple(renamed if s.name == "reax1" else s for s in t.slots),
                                                  carried=t.carried)) != h
    looser = dataclasses.replace(r1, params=dataclasses.replace(r1.params, mean_tau=r1.params.mean_tau + 0.01))
    assert composite_rules_sha256(CompositeTrack(t.name, tuple(looser if s.name == "reax1" else s for s in t.slots),
                                                  carried=t.carried)) != h
    # migration: v1 s1-fast entries are still Plumb-only (so they score plumb ** 0.05 in the first v2 weight set)
    assert t.carried_declaration("s1-fast") == {"plumb"} and t.carried_declaration("dm-all") is None
    assert t.slot("plumb").weight == 0.05


def test_reax1_reference_image_slot_and_noise_units():
    from reax_subnet import config

    assert set(config.ENGINE_COMPOSITE_REFERENCE_IMAGES) == {"plumb", "blink", "reax1", "pplx", "d3", "quyet", "vision",
                                                              "mammoth", "jeffan"}
    assert "@sha256:" in config.ENGINE_COMPOSITE_REFERENCE_IMAGES["reax1"]  # published (reax-engine #9 @4230e50)
    t = default_track(variant="v21")
    assert t.release_ready() == []
    sigma = math.sqrt(sum((s.weight * s.params.reference_noise_std) ** 2 for s in t.slots))
    assert t.noise() == pytest.approx(sigma)


# ------------------------------------------------------------------------------------------------ v2.1 model list (22 Oct 2026)
V21_WEIGHTS = {"d3": 0.50, "blink": 0.25, "reax1": 0.20, "pplx": 0.05}
V22_WEIGHTS = {"d3": 0.45, "blink": 0.20, "vision": 0.05, "reax1": 0.20, "pplx": 0.05, "quyet": 0.05}
REAX_HIGH_WEIGHTS = {"reax1": 0.40, "d3": 0.35, "blink": 0.20, "pplx": 0.05}


def test_v21_stays_selectable_without_plumb(monkeypatch):
    from reax_subnet import engcomposite as ec

    t = default_track(variant="v21")
    assert [s.name for s in t.slots] == ["blink", "reax1", "pplx", "d3"]  # cheapest first, no plumb, no vision, no quyet
    assert {s.name: s.weight for s in t.slots} == V21_WEIGHTS
    assert t.carried == () and t.carried_declaration("s1-fast") is None  # v1 s1-fast has no v2 entry
    with pytest.raises(ValueError):
        parse_declared("plumb", t)
    assert parse_declared(None, t) == {"blink", "reax1", "pplx", "d3"}
    assert t.release_ready() == []  # calibrated on the release references
    monkeypatch.setattr(ec, "CALIBRATED_ON_RELEASE_REFERENCES", False)
    assert len(default_track(variant="v21").release_ready()) == 4  # the pre-release state refuses all four slots
    for variant, weights in ec.COMPOSITE_WEIGHTS.items():  # one table per variant, each sums to 1
        assert math.fsum(weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert {s.name: s.weight for s in default_track(variant=variant).slots} == weights
    with pytest.raises(ValueError, match="unknown composite weights variant"):
        default_track(variant="v99")


def test_approved_1021_variant_stays_selectable_with_the_plumb_carry_over():
    t = default_track(variant="approved-1021")
    assert [s.name for s in t.slots] == ["plumb", "blink", "reax1", "pplx", "d3"]
    assert {s.name: s.weight for s in t.slots} == REAX1_WEIGHTS
    assert t.carried_declaration("s1-fast") == {"plumb"}
    assert composite_rules_sha256(t) != composite_rules_sha256(default_track())  # the table is part of the rules hash
    # a v1 image alone: plumb ** 0.05 in the approved set, no entry at all in v2.1
    assert composite_score(t, frozenset({"plumb"}), {"plumb": _r(1.81)}).score == pytest.approx(1.81 ** 0.05)


def test_v21_composite_math_and_rules_hash(monkeypatch):
    from reax_subnet import engcomposite as ec

    t = default_track(variant="v21")
    every = frozenset(V21_WEIGHTS)
    gains = {"reax1": 1.5, "d3": 1.2, "blink": 1.1, "pplx": 1.05}
    cs = composite_score(t, every, {k: _r(v) for k, v in gains.items()})
    assert cs.ok and cs.score == pytest.approx(math.exp(sum(V21_WEIGHTS[k] * math.log(v) for k, v in gains.items())))
    only = composite_score(t, frozenset({"d3"}), {"d3": _r(1.5)})
    assert only.ok and only.score == pytest.approx(1.5 ** 0.50)
    h = composite_rules_sha256(t)
    assert h == composite_rules_sha256(default_track(variant="v21"))
    monkeypatch.setitem(ec.COMPOSITE_WEIGHTS, "v21", {"reax1": 0.25, "d3": 0.45, "blink": 0.25, "pplx": 0.05})
    assert composite_rules_sha256(default_track(variant="v21")) != h  # moving a weight is a new release


def test_d3_slot_serves_d3_turbo_and_rolls_back_with_one_constant(monkeypatch):
    from reax_subnet import engcomposite as ec

    t = default_track(variant="v20-d3")  # d3 is a slot of the fallback variant (and of v22, v21, ...), not of v20
    d3 = t.slot("d3")
    assert ec.D3_SLOT_MODEL == "d3-turbo" and d3.model == "d3-turbo" and d3.calibrated
    # release calibration of d3-turbo (23 Oct)
    assert d3.params.concurrency == 8 and d3.params.reference_noise_std == 0.0057 and d3.memory_budget_mib == 45_056
    h = composite_rules_sha256(t)
    monkeypatch.setattr(ec, "D3_SLOT_MODEL", "d3")
    back = default_track(variant="v20-d3")
    assert back.slot("d3").model == "d3" and composite_rules_sha256(back) != h


def test_reference_image_keys_follow_the_slots_and_release_refuses_none(monkeypatch):
    from reax_subnet import config
    from reax_subnet.worker import composite_references

    monkeypatch.setattr("reax_subnet.engcomposite.COMPOSITE_WEIGHTS_VARIANT", "v22")
    assert set(config.ENGINE_COMPOSITE_REFERENCE_IMAGES) == {"plumb", "blink", "reax1", "pplx", "d3", "quyet", "vision",
                                                              "mammoth", "jeffan"}
    assert all("@sha256:" in config.ENGINE_COMPOSITE_REFERENCE_IMAGES[k] for k in ("blink", "reax1", "pplx", "d3", "quyet", "vision"))
    assert set(composite_references({}, release_mode=True)) >= set(V22_WEIGHTS)  # all six are pinned: release mode accepts them
    for k in ("blink", "reax1", "pplx", "d3", "quyet", "vision"):  # an unpublished (None) reference is refused in release mode
        monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, k, None)
    with pytest.raises(config.ConfigError, match="(?=.*blink)(?=.*reax1)(?=.*pplx)(?=.*d3)(?=.*quyet)(?=.*vision)"):
        composite_references({}, release_mode=True)  # plumb is no v2.2 slot: its constant is not demanded
    pinned = ",".join(f"{n}=ghcr.io/reaxlabs/x-{n}@sha256:{D1}" for n in ("blink", "reax1", "pplx", "d3", "quyet", "vision"))
    assert set(composite_references({"REAX_ENGINE_COMPOSITE_REFS": pinned}, release_mode=False)) >= set(V22_WEIGHTS)


def test_noise_and_margin_of_the_variants():
    sigma = {v: default_track(variant=v).noise() for v in ("v21", "reax-high", "approved-1021")}
    v20 = default_track().noise()  # the default (v20, 26 Oct pick): blink's .0751 at weight 0.33 dominates the unit
    assert v20 == pytest.approx(math.sqrt(math.fsum((s.weight * s.params.reference_noise_std) ** 2
                                                    for s in default_track().slots)))
    assert 0.0245 < v20 < 0.0265
    assert sigma["v21"] == pytest.approx(0.020073, abs=1e-6) and sigma["reax-high"] == pytest.approx(0.019966, abs=1e-6)
    assert sigma["approved-1021"] == pytest.approx(0.017298, abs=1e-6)
    assert max(0.03, 3 * sigma["v21"]) == pytest.approx(0.060220, abs=1e-6)  # 3 sigma beats the 3 % floor
    assert max(0.03, 3 * sigma["approved-1021"]) == pytest.approx(0.051893, abs=1e-6)


def test_reax_high_variant_keeps_reax1_on_top_without_plumb():
    t = default_track(variant="reax-high")
    assert [s.name for s in t.slots] == ["blink", "reax1", "pplx", "d3"]
    assert {s.name: s.weight for s in t.slots} == REAX_HIGH_WEIGHTS
    assert t.carried == ()
    assert composite_rules_sha256(t) != composite_rules_sha256(default_track())


# ------------------------------------------------------------------------------------------------ v1 payout bridge
BRIDGE_END = __import__("datetime").datetime(2026, 11, 20, tzinfo=__import__("datetime").timezone.utc)


def _v1_state_file(path, champion_uid=0, hotkey="hk0", digest=D3):
    from reax_subnet.engcomp import Champion

    st = TrackState()
    st.champion = Champion(digest, hotkey, champion_uid, 2.0, 400)
    st.results[digest] = EvalResult(digest, "s1-fast", True, score=2.0, block=399, session="v1")  # a former champion is paid on a valid result
    st.slot_results.append({"slot_utc": "2026-10-27T14:30:00+00:00", "event": "crowned"})  # the switch slot is finalised in the frozen file
    path.write_text(json.dumps({"version": 1, "tracks": {"s1-fast": st.to_json()}}))
    return str(path)


def _pin(path, end=None):
    """(path, end, sha256 of the file): the bridge tuple with the correct pin."""
    return (str(path), end or BRIDGE_END, hashlib.sha256(open(path, "rb").read()).hexdigest())


def _bridge_comp(tmp_path, now, bridge, gate=None, sub=None):
    """Each call gets its own state directory (the bridge-ended marker lives there) unless ``sub`` names one."""
    d = tmp_path / (sub or f"d{next(_BRIDGE_DIRS)}")
    return EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(d / "e.json"), composites={"dm-all": TRACK},
                             start_utc=None, utcnow=lambda: now, award_gate=gate, v1_bridge=bridge)


_BRIDGE_DIRS = __import__("itertools").count()


def test_v1_bridge_pays_the_v1_champion_until_the_end_or_the_first_dm_all_champion(tmp_path):
    from reax_subnet.engcomp import Champion

    when = BRIDGE_END.replace(day=10)
    v1 = _v1_state_file(tmp_path / "v1.json")
    snap = _snapshot([("s1-fast", D3), ("dm-all", D1)], 500)  # uid 0 = v1 champion, uid 1 = a dm-all entry
    base = {5: 1.0}
    off = _bridge_comp(tmp_path, when, None)
    assert off.weights(base, snap, set()) == base  # constant off (None): v2.1 without a champion pays nothing
    on = _bridge_comp(tmp_path, when, _pin(v1))
    paid = on.weights(base, snap, set())
    assert paid[0] > 0 and paid[5] < 1.0 and 1 not in paid
    assert on.context_sha256 == off.context_sha256 and not (tmp_path / "e.json").exists()
    assert not os.path.exists(on._bridge_marker())  # no context, no state touched
    # the v1 weight equals what v1 itself pays: the s1-fast track with the same state file and the same reward rule
    v1_comp = EngineCompetition(_LabelRunner({}), {"s1-fast": "ref"}, str(tmp_path / "v1own.json"), start_utc=None,
                                tracks={"s1-fast": __import__("reax_subnet.engvalidator", fromlist=["TRACKS"]).TRACKS["s1-fast"]},
                                utcnow=lambda: when)
    v1_comp.states["s1-fast"].champion = on_state = Champion(D3, "hk0", 0, 2.0, 400)
    assert on_state is not None and v1_comp.weights(base, snap, set()) == pytest.approx(paid)
    # after the end time the normal no-champion rule applies
    assert _bridge_comp(tmp_path, BRIDGE_END, _pin(v1)).weights(base, snap, set()) == base
    # the first dm-all champion ends it at once: it gets the champion share, the v1 holder moves to the former place
    class Everyone:
        def approved(self, track, hotkey, sid):
            return True

    first = _bridge_comp(tmp_path, when, _pin(v1), Everyone())
    first.states["dm-all"].champion = Champion(D1, "hk1", 1, 1.3, 450)
    w = first.weights(base, snap, set())
    assert w[1] > 0 and w[0] > 0 and w[0] < w[1]  # the v1 holder now sits in the 17 % former-champion place (not the champion share)
    # the v1 champion needs its award approval like in v1 (track s1-fast), and a gone/changed commitment pays nothing
    class Gate:
        asked = []

        def approved(self, track, hotkey, sid):
            self.asked.append((track, hotkey, sid))
            return False

    gate = Gate()
    assert _bridge_comp(tmp_path, when, _pin(v1), gate).weights(base, snap, set()) == base
    assert gate.asked and gate.asked[0] == ("s1-fast", "hk0", "sha256:" + D3)
    gone = _snapshot([("dm-all", D1)], 500)
    assert on.weights(base, gone, set()) == base


def test_v1_bridge_with_a_missing_or_corrupt_v1_file_pays_nothing_and_never_raises(tmp_path):
    when = BRIDGE_END.replace(day=10)
    snap = _snapshot([("s1-fast", D3)], 500)
    base = {5: 1.0}
    assert _bridge_comp(tmp_path, when, (str(tmp_path / "nope.json"), BRIDGE_END, "0" * 64)).weights(base, snap, set()) == base
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert _bridge_comp(tmp_path, when, _pin(bad)).weights(base, snap, set()) == base
    bad.write_text(json.dumps({"version": 7, "tracks": {}}))
    assert _bridge_comp(tmp_path, when, _pin(bad)).weights(base, snap, set()) == base


def test_the_bridge_constants_are_the_published_switch_and_the_pin_is_the_frozen_v1_file():
    from datetime import datetime, timezone

    from reax_subnet import competition_split

    assert competition_split.ENGINE_V1_BRIDGE_END_UTC == datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc)
    assert competition_split.ENGINE_V1_BRIDGE_STATE_SHA256 == FROZEN_V1_SHA256  # pinned after the 27 Oct slot
    assert competition_split.release_bridge_problem() is None


FROZEN_V1_SHA256 = "1faccee327f80b48427778d0561195c9d2c0110ffc15f897ff7ef5045969c7a6"


def test_read_frozen_v1_accepts_the_frozen_file_with_the_pin_and_refuses_other_pins():
    import os

    from reax_subnet import competition_split
    from reax_subnet.engvalidator import read_frozen_v1

    path = os.path.join(os.path.dirname(__file__), "data", "engine-competition.v1-final-1faccee3.json")
    st, binds = read_frozen_v1(path, competition_split.ENGINE_V1_BRIDGE_STATE_SHA256)
    assert st is not None and isinstance(binds, bool)
    with pytest.raises(ValueError, match="not the pinned"):
        read_frozen_v1(path, "0" * 64)


# ------------------------------------------------------------------------------------------------ release state (23 Oct 2026)
def test_release_state_is_consistent():
    from reax_subnet import config
    from reax_subnet import engcomposite as ec

    assert ec.CALIBRATED_ON_RELEASE_REFERENCES is True
    for variant in ec.COMPOSITE_WEIGHTS:
        if variant not in ("v22", "v20", "v20-d3"):  # the four calibrated slots of the earlier variants are release-ready
            assert default_track(variant=variant).release_ready() == []
    refs = config.ENGINE_COMPOSITE_REFERENCE_IMAGES
    for s in default_track(variant="v21").slots:  # every v2.1 slot is digest-pinned
        assert refs[s.name] and re.search(r"@sha256:[0-9a-f]{64}$", refs[s.name]), s.name
    assert refs["reax1"] == refs["pplx"] == refs["d3"]  # one qwen35 image serves the three 27B slots
    assert refs["blink"] != refs["d3"]


# ------------------------------------------------------------------------------------------------ local tool (reax-engine-eval)
def test_reax_engine_eval_runs_dm_all_locally(tmp_path, monkeypatch, capsys):
    """Miners measure a v2.0 image locally with the validator's code path: ``reax-engine-eval --track dm-all``."""
    from reax_subnet import config, engcli
    from reax_subnet import engcomposite as ec

    track = _fast_track()
    monkeypatch.setattr(ec, "default_track", lambda calibration=None: track)
    monkeypatch.setattr(config, "ENGINE_COMPOSITE_REFERENCE_IMAGES", {"m-a": ref_of("reference"), "m-b": ref_of("reference")})
    variants = ("reference", "fast", "cache")
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({ref_of(v): [sys.executable, FAKE, "--variant", v, "--delay", "0.01", "--jitter", "0.4"]
                               for v in variants}))
    out = tmp_path / "out.json"
    engcli.eval_main(["--track", "dm-all", "--candidate", ref_of("fast"), "--candidate", ref_of("cache"),
                      "--models", "m-a,m-b", "--dev-map", str(dev), "--workdir", str(tmp_path / "w"), "--out", str(out)])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]["ok"] and 1.4 < rows[0]["score"] < 2.6 and set(rows[0]["parts"]) == {"m-a", "m-b"}
    assert rows[0]["declared"] == ["m-a", "m-b"]
    assert not rows[1]["ok"] and rows[1]["reason"].startswith("m-a: anti-cache probe")
    saved = json.loads(out.read_text())
    assert saved["track"] == "dm-all" and saved["composite_rules_sha256"] == ec.composite_rules_sha256(track)
    with pytest.raises(SystemExit, match="unknown models"):
        engcli.eval_main(["--track", "dm-all", "--candidate", ref_of("fast"), "--reference", "nope=x@sha256:" + D1,
                          "--dev-map", str(dev)])
    with pytest.raises(SystemExit, match="pin every image by digest"):
        engcli.eval_main(["--track", "dm-all", "--candidate", ref_of("fast"), "--reference", "m-a=ghcr.io/x/e:latest",
                          "--dev-map", str(dev)])
    with pytest.raises(SystemExit, match="--reference is required"):
        engcli.eval_main(["--candidate", ref_of("fast"), "--dev-map", str(dev)])
    # docker path: only declared models need weights; vision-style slots borrow their source's directory
    monkeypatch.setattr(ec, "SLOT_WEIGHTS_FROM", {"m-b": "m-a"})
    with pytest.raises(SystemExit, match="no weights directory.*m-b"):
        engcli.eval_main(["--track", "dm-all", "--candidate", ref_of("fast"), "--models", "m-b",
                          "--weights", f"m-a={tmp_path / 'nope'}", "--workdir", str(tmp_path / "w")])
    monkeypatch.setattr(ec, "evaluate_composite", _fake_composite := _Recorder())
    (tmp_path / "wa").mkdir()
    engcli.eval_main(["--track", "dm-all", "--candidate", ref_of("fast"), "--models", "m-b",
                      "--weights", f"m-a={tmp_path / 'wa'}", "--workdir", str(tmp_path / "w")])
    runner, declared = _fake_composite.calls[0]
    assert declared == frozenset({"m-b"}) and runner.runners["m-b"].weights_dir == str(tmp_path / "wa")
    assert runner.runners["m-b"].model_env == {ec.MODEL_ENV: "m-b"}
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["declared"] == ["m-b"]


class _Recorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, runner, reference, candidate, track, declared, **kw):
        self.calls.append((runner, declared))
        return EvalResult(digest="d", track=track.name, ok=True, score=1.0)


def _blink_pair(n_moved: int, n_flipped: int, shift: float, n: int = 512):
    """Reference answers and a candidate whose first n_moved items move by a logit shift and whose first n_flipped of
    those swap their top choice (3 options, T=1.3 distances)."""
    ref, cand = {}, {}
    for i in range(n):
        p = [0.6, 0.3, 0.1]
        if i < n_flipped:
            q = [0.3, 0.6, 0.1]
        elif i < n_moved:
            w = [math.log(x) for x in p]
            w[0] += shift
            z = sum(math.exp(x) for x in w)
            q = [math.exp(x) / z for x in w]
        else:
            q = list(p)
        ref[f"i{i}"], cand[f"i{i}"] = p, q
    return cand, ref


def test_blink_gates_admit_the_reference_cross_start_drift_and_refuse_a_lossy_engine():
    from reax_subnet.engcomp import check_correctness

    bl = default_track().slot("blink").params
    # ~16 % of 2x256 timed items move (gpucheck3: 42/256), 6 of 512 flip (3/256): an exact copy must pass
    cand, ref = _blink_pair(n_moved=84, n_flipped=6, shift=0.12)
    ok = check_correctness(cand, ref, bl)
    assert ok.ok, ok.reason
    assert not check_correctness(cand, ref, dataclasses.replace(bl, mean_tau=0.05, item_tau=1.0, min_top_agreement=0.99)).ok
    # a lossy engine that moves every item (mean 0.1 at T 1.3) or flips more than 3 % of the tops fails
    assert not check_correctness(*_blink_pair(n_moved=512, n_flipped=0, shift=0.12), bl).ok
    assert not check_correctness(*_blink_pair(n_moved=0, n_flipped=16, shift=0.0), bl).ok
    # the worst measured byte-copy pair (raw fresh starts: 4 of 256 tops flipped, i.e. 8 of 512) passes
    assert check_correctness(*_blink_pair(n_moved=40, n_flipped=8, shift=0.12), bl).ok


def test_v2_docs_gate_table_matches_the_code():
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "ENGINE-COMPETITION-V2.md"), encoding="utf-8").read()
    rows = re.findall(r"^\| (mammoth|blink|reax1|jeffan|vision|d3)(?: \(fallback v20-d3\))? \| ([\d.]+) \| (\d+) \| ([\d.]+) \| "
                      r"([\d.]+) \| (\d+) % \| ([\d.]+) \|$", doc, re.M)
    assert {r[0] for r in rows} == {"mammoth", "blink", "reax1", "jeffan", "vision", "d3"}
    for name, t, items, mean, item, top, det in rows:
        track = default_track(variant="v20-d3") if name == "d3" else default_track()
        p = track.slot(name).params
        assert (p.temperature, p.timed_items, p.mean_tau, p.item_tau, p.min_top_agreement, p.determinism_tau) == pytest.approx(
            (float(t), int(items), float(mean), float(item), int(top) / 100, float(det))), name
