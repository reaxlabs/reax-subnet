"""Critic findings C1-C3 for engine v2 (each failed on the reviewed branch 4593918; fixed after critic1)."""

from __future__ import annotations

import pytest

from reax_subnet.engcomp import EvalResult
from reax_subnet.engcomposite import evaluate_composite
from test_engcomposite import ALL, TRACK, D1, D2, _Calls, _Runner, _LabelRunner, _snapshot, _r, composite_score, to_eval_result
from reax_subnet.engvalidator import EngineCompetition


@pytest.fixture(autouse=True)
def _copyline_not_started(monkeypatch):
    from reax_subnet import engcomp  # see test_engcomposite._copyline_not_started

    monkeypatch.setattr(engcomp, "ENGINE_COPYLINE_START_UTC", None)


async def test_C1_dm_all_champion_gets_no_weight(tmp_path, monkeypatch):
    import reax_subnet.engvalidator as ev

    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        res = {s.name: _r(1.3) for s in track.slots if s.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    monkeypatch.setattr(ev, "evaluate_composite", fake)
    comp = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "e.json"),
                             composites={"dm-all": TRACK}, start_utc=None)
    snap = _snapshot([("dm-all", D1)], 500)
    await comp.run_cycle(snap, set())
    assert comp.states["dm-all"].champion is not None
    w = comp.weights({5: 1.0}, snap, set())  # (lead: an empty base stays empty by v1 design, so a non-empty base here)
    assert w.get(0, 0) > 0  # failed before the fix: weights() only paid track == "s1-fast"


async def test_C2_negative_memory_delta_is_not_a_failed_candidate(monkeypatch):
    import reax_subnet.engrunner as er

    class Fake(_Calls):
        async def __call__(self, *a, **k):
            r = await super().__call__(*a, **k)
            r.cand_memory_mib = 0  # a negative device delta (reference trimmed / teardown) is clamped to 0 in evaluate()
            return r

    monkeypatch.setattr(er, "evaluate", Fake({"a": 1.1, "b": 1.1, "c": 1.1, "d": 1.1}))
    import dataclasses
    from reax_subnet.engcomposite import CompositeTrack
    budgeted = CompositeTrack("dm-all", tuple(dataclasses.replace(s, memory_budget_mib=1000) for s in TRACK.slots))
    res = await evaluate_composite(_Runner(), "ref", "img", budgeted, ALL, digest=D1)
    assert res.ok  # failed before the fix: a negative delta was recorded as a definite failure


async def test_C3_slow_later_model_is_a_candidate_failure_not_inconclusive(tmp_path, monkeypatch):
    import asyncio
    import reax_subnet.engvalidator as ev

    async def slow(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        progress["measuring"] = True  # model 1 measured ...
        await asyncio.sleep(10)  # ... the validator is slow pulling/starting model 2

    monkeypatch.setattr(ev, "evaluate_composite", slow)
    monkeypatch.setattr(type(TRACK), "budget_s", lambda self, declared=None: 0.05)
    comp = EngineCompetition(_LabelRunner({}), {"dm-all": "ref"}, str(tmp_path / "e.json"),
                             composites={"dm-all": TRACK}, start_utc=None)
    await comp.run_cycle(_snapshot([("dm-all", D1)], 500), set())
    assert D1 not in comp.states["dm-all"].results  # failed before the fix: recorded as ok=False 'exceeded its budget'
