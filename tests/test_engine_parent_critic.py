"""Critic pass on the same-session parent rule (workers/parent-critic): attack scenarios against engvalidator._parent_reference."""

import asyncio
from datetime import timedelta

from reax_subnet import engcomp as EC
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import EvalResult, Submission
from test_engcomp import P
from test_engfair import _snap, _sub
from test_engine_copyline import D1, D2, START, _comp, _gate, _label, _record
from test_engine_parent_session import NEAR


class _Refs:
    """Stand-in for engrunner.evaluate that records the image reference of every run."""

    def __init__(self, scores):
        self.scores, self.refs = scores, []

    async def __call__(self, runner, reference, ref, track, params, block=0, digest="", progress=None):
        self.refs.append(ref)
        return EvalResult(digest, track, True, score=self.scores[digest], block=block)


def _cycle(tmp_path, monkeypatch, subs):
    _gate(monkeypatch, START)
    monkeypatch.setattr(EC, "ENGINE_PARENT_SESSION_START_UTC", START + timedelta(hours=1))
    comp = _comp(tmp_path, now=START + timedelta(hours=2))
    st = comp.states["s1-fast"]
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=50, session="p")
    st.hold(EvalResult(D1, "s1-fast", True, score=1.45, block=40, session="p0"), P)
    st.hold(st.results[D1], P)
    st.first_seen[D1] = (50, "pa")
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
    fake = _Refs({D2: NEAR, D1: 1.46})
    monkeypatch.setattr(EV, "evaluate", fake)
    asyncio.run(comp.run_cycle(_snap(subs, block=1000), set()))
    return st, fake


def test_parent_image_comes_from_the_parent_owner_not_a_copycat_commit(tmp_path, monkeypatch):
    """Anybody can commit the parent's digest under another image name (the commitment is ignored for payment, but it sits in
    the snapshot). A lower uid must not decide which image name the reference run pulls: an unpullable name would make the
    parent run fail and push the child back to the stored line."""
    copycat = Submission(0, "evil", "s1-fast", "ghcr.io/evil/gone", D1, 60)
    owner = Submission(1, "pa", "s1-fast", "ghcr.io/a/e", D1, 50)
    st, fake = _cycle(tmp_path, monkeypatch, [copycat, owner, _sub(2, "ch", D2, 200)])
    assert fake.refs and all(r == f"ghcr.io/a/e@sha256:{D1}" for r in fake.refs if r.endswith(D1))
    assert not any("evil" in r for r in fake.refs)
    assert st.champion is not None and st.champion.digest == D2


def test_non_finite_parent_score_is_no_run(tmp_path, monkeypatch):
    """An ok parent run with a non-finite score must not be stored as a valid run of score 0.0 (that would drop the line to
    the floor); it counts as a failed run, so the stored line applies."""
    own = Submission(1, "pa", "s1-fast", "ghcr.io/a/e", D1, 50)
    _gate(monkeypatch, START)
    monkeypatch.setattr(EC, "ENGINE_PARENT_SESSION_START_UTC", START + timedelta(hours=1))
    comp = _comp(tmp_path, now=START + timedelta(hours=2))
    st = comp.states["s1-fast"]
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=50, session="p")
    st.hold(st.results[D1], P)
    st.first_seen[D1] = (50, "pa")
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
    fake = _Refs({D2: NEAR, D1: float("nan")})
    monkeypatch.setattr(EV, "evaluate", fake)
    asyncio.run(comp.run_cycle(_snap([own, _sub(2, "ch", D2, 200)], block=1000), set()))
    runs = list(st.parent_sessions.values())
    assert runs and all(r["ok"] is False for r in runs)
    assert st.champion is None or st.champion.digest != D2
