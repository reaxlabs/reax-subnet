"""Near-champion rule (REAX_SPEC_NC_NEAR_CHAMP, off by default; decided 19 Oct 2026; docs/SPECIALIST-TRACKS.md "Near-duplicates").

A challenger whose weights are close to the incumbent's (delta sketch cos > 0.995 or relative L2 < 0.15) must also answer more
items correctly than the incumbent and gain max(margin, 3.0) points on every draw that decides it. Live shape (DF, 19 Oct): r5 =
x9b + letter rows x0.85 + weight noise, cos 0.998 / rel 0.075 against x9b, gains +2.19 / +1.66 from calibration only."""

import dataclasses
import os

import pytest
from spec_helpers import finetune, write_model
from test_copycheck2 import _base, _server_env, noisy, temperature
from test_nearcopy import _crown, _orth_copy, _run
from test_spec_evaluator import TRACK

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import scoring as SC
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

NEAR = dict(nc_near_champ=True)


@pytest.fixture(autouse=True)
def small_margin(monkeypatch):
    """As in test_nearcopy: the fake copy's calibration gain is small, a low margin lets the old rules crown it."""
    monkeypatch.setattr(M, "MARGIN_FLOOR", 0.3)
    monkeypatch.setattr(M, "SE_MULTIPLE", 1.0)


def _flags(env, **kw):
    env.cfg = dataclasses.replace(env.cfg, **kw)
    env.ev.cfg = env.cfg


async def _scenario(tmp_path, flags, rel=0.08):
    """x9b champion; r5 = x9b weights + orthogonal noise (rel ``rel``), answers = the same hidden model, better calibrated."""
    base = _base()
    env = _server_env(tmp_path, base)
    _flags(env, **flags)
    parent = finetune(base, 10)
    env.server.skills["z/hidden"] = 0.65
    env.server.like["a/x9b"] = ("z/hidden", temperature(0.3))
    await _crown(env, 1, "a/x9b", None, parent)
    assert env.sub(1).status == ST.CHAMPION
    env.add(2, "b/r5", weights=_orth_copy(parent, base, rel, 5))
    env.server.like["b/r5"] = ("z/hidden", noisy(0.05))
    return env, parent


def _rows_of(sub, kind="near_champ"):
    return [h for h in sub.history if h.get("kind") == kind]


def test_flag_default_off_and_parse(tmp_path):
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path)}).nc_near_champ is False
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_NC_NEAR_CHAMP": "1"}).nc_near_champ is True
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_NC_NEAR_CHAMP": ""}).nc_near_champ is False
    with pytest.raises(ValueError):
        C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_NC_NEAR_CHAMP": "maybe"})
    assert (E.NEAR_CHAMP_COS, E.NEAR_CHAMP_REL, E.NEAR_CHAMP_FLOOR) == (0.995, 0.15, 3.0)
    assert (E.NEAR_CHAMP_ALPHA_COS, E.NEAR_CHAMP_ALPHA_TOL) == (0.98, 0.05)


async def test_flag_off_the_near_copy_is_crowned_and_no_rows(tmp_path):
    env, _ = await _scenario(tmp_path, {})
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    assert env.sub(2).status == ST.CHAMPION and not _rows_of(env.sub(2))


@pytest.mark.parametrize("v3", [False, True])
async def test_near_copy_of_the_champion_is_not_crowned(tmp_path, v3):
    env, _ = await _scenario(tmp_path, dict(NEAR, v3_decisions=v3))
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    s, x = env.sub(2), env.sub(1)
    assert env.ev.state.track(TRACK).champion == x.key and s.status != ST.CHAMPION
    rows = _rows_of(s)
    assert rows and rows[0]["vs"] == x.key and not rows[0]["ok"]
    assert rows[0]["cos"] > E.NEAR_CHAMP_COS or rows[0]["rel"] < E.NEAR_CHAMP_REL
    assert rows[0]["need"] >= E.NEAR_CHAMP_FLOOR
    assert "close to the champion's weights" in s.reason


async def test_near_by_relative_l2_alone(tmp_path):
    """cos is high for any small noise; a larger noise (rel 0.12) gives cos ~0.993 < 0.995 but rel < 0.15: still near."""
    env, _ = await _scenario(tmp_path, NEAR, rel=0.12)
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    rows = _rows_of(env.sub(2))
    assert rows and rows[0]["cos"] <= E.NEAR_CHAMP_COS < 1 and rows[0]["rel"] < E.NEAR_CHAMP_REL
    assert env.ev.state.track(TRACK).champion == env.sub(1).key


async def test_far_weights_are_not_near(tmp_path):
    """Noise of 0.5 x the parent's delta: cos ~0.89, rel 0.5 -> not near; the old rules decide (and crown it here)."""
    env, _ = await _scenario(tmp_path, NEAR, rel=0.5)
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    assert not _rows_of(env.sub(2)) and env.sub(2).status == ST.CHAMPION


def test_is_near_thresholds():
    """The three triggers, exactly (the sketch positions are random per validator, so the end-to-end tests do not pin them)."""
    from reax_subnet.spec import tensors as T

    def st(cos, rel, alpha, compared=10, norm_a=1.0):
        return T.DeltaStats(cosine=cos, relative_l2=rel, compared=compared, norm_a=norm_a, norm_b=1.0, alpha=alpha)

    near = E.SpecEvaluator._is_near
    assert near(st(0.9976, 0.075, 0.967))                      # r5 vs x9b (real sketch)
    assert near(st(0.996, 0.30, 0.5)) and near(st(0.90, 0.149, 0.5))
    assert near(st(0.985, 0.17, 1.04)) and not near(st(0.985, 0.17, 1.06))     # alpha band: champion + orthogonal noise
    assert not near(st(0.979, 0.2, 1.0))
    assert not near(st(0.9947, 0.172, 1.126))                  # closest legitimate pair (RA putty77 vs RA160, real sketch)
    assert not near(st(1.0, 0.0, 1.0, compared=0)) and not near(st(1.0, 0.0, 1.0, norm_a=0.0))


async def test_twice_the_noise_is_still_near_by_alpha(tmp_path):
    """critic-nc F2: about twice r5's noise (cos ~0.986, rel ~0.17) is caught (alpha ~1): near, not crowned."""
    env, _ = await _scenario(tmp_path, NEAR, rel=0.17)
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    r = _rows_of(env.sub(2))
    assert r and not r[0]["ok"] and abs(r[0]["alpha"] - 1) < E.NEAR_CHAMP_ALPHA_TOL
    assert env.ev.state.track(TRACK).champion == env.sub(1).key


async def test_declaring_the_champion_as_base_does_not_hide_the_copy(tmp_path):
    """critic-nc F1: the same r5 weights, but the manifest declares the champion's own public repo as its base. The
    challenger is sketched again against the champion's base: near, not crowned."""
    base = _base()
    env = _server_env(tmp_path, base)
    _flags(env, **NEAR)
    parent = finetune(base, 10)
    env.server.skills["z/hidden"] = 0.65
    env.server.like["a/x9b"] = ("z/hidden", temperature(0.3))
    await _crown(env, 1, "a/x9b", None, parent)
    x9b_as_base = "a/x9b@" + "9" * 40
    write_model(str(tmp_path / "bases" / x9b_as_base.replace("/", "__", 1)), parent, dtype="BF16")
    env.add(2, "b/r5", weights=_orth_copy(parent, base, 0.08, 5), base_model=x9b_as_base)
    env.server.like["b/r5"] = ("z/hidden", noisy(0.05))
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    r5 = env.sub(2)
    assert r5.base_id != env.sub(1).base_id
    rows = _rows_of(r5)
    assert rows and rows[0].get("comparable", True) and not rows[0]["ok"] and rows[0]["rel"] < E.NEAR_CHAMP_REL
    assert env.ev.state.track(TRACK).champion == env.sub(1).key


def test_items_right_ignore_letter_rescaling():
    """critic-nc F3: the scored correctness of noul/score items moves with a temperature, ``top_hit`` does not."""
    rows = [{"id": "n", "question": {"type": "noul"}, "labels": ["yes", "no"], "expected": "yes"},
            {"id": "s", "question": {"type": "score"}, "labels": ["0", "1", "2", "3", "4"], "expected": 3},
            {"id": "c", "question": {"type": "choice"}, "labels": ["A", "B", "C"], "expected": "B"}]
    sharp = [{"yes": 0.9, "no": 0.1}, {"0": 0.0, "1": 0.05, "2": 0.15, "3": 0.6, "4": 0.2}, {"A": 0.2, "B": 0.7, "C": 0.1}]

    def soften(p, s):
        w = {k: v ** s if v > 0 else 0.0 for k, v in p.items()}
        t = sum(w.values())
        return {k: v / t for k, v in w.items()}

    for s in (0.3, 0.5, 2.0):
        a = [SC.score_item(r, p) for r, p in zip(rows, sharp)]
        b = [SC.score_item(r, soften(p, s)) for r, p in zip(rows, sharp)]
        assert [x.top_hit for x in a] == [x.top_hit for x in b] == [True, True, True]
    soft = [SC.score_item(r, soften(p, 0.3)) for r, p in zip(rows, sharp)]
    assert not soft[0].correct and soft[0].top_hit        # noul: softened into the abstain band, still the top label


async def test_independent_better_model_is_still_crowned(tmp_path):
    env, _ = await _scenario(tmp_path, NEAR)
    env.add(3, "c/indep", 0.97, weights=finetune(_base(), 77))
    await _run(env, {1: "a/x9b", 3: "c/indep"})
    assert env.sub(3).status == ST.CHAMPION and not _rows_of(env.sub(3))


async def test_near_entry_that_really_answers_better_still_wins(tmp_path):
    """A continued fine-tune of the champion with clearly more items right and a lead above 3 points is crowned."""
    base = _base()
    env = _server_env(tmp_path, base)
    _flags(env, **NEAR)
    parent = finetune(base, 10)
    await _crown(env, 1, "a/x9b", 0.6, parent)
    env.add(2, "b/better", 0.98, weights=_orth_copy(parent, base, 0.08, 5))
    await _run(env, {1: "a/x9b", 2: "b/better"})
    s = env.sub(2)
    rows = _rows_of(s)
    assert s.status == ST.CHAMPION and len(rows) == 2 and all(r["ok"] for r in rows)   # first draw AND confirmation draw
    assert all(r["correct"] > r["inc_correct"] and r["gain"] >= r["need"] >= E.NEAR_CHAMP_FLOOR for r in rows)


async def test_small_real_gain_below_the_floor_is_not_enough(tmp_path, monkeypatch):
    """Near and slightly better: passes the (lowered) normal margin but not the 3-point floor."""
    base = _base()
    env = _server_env(tmp_path, base)
    _flags(env, **NEAR)
    parent = finetune(base, 10)
    await _crown(env, 1, "a/x9b", 0.6, parent)
    monkeypatch.setattr(E, "NEAR_CHAMP_FLOOR", 1000.0)      # any real gain is below this floor
    env.add(2, "b/better", 0.98, weights=_orth_copy(parent, base, 0.08, 5))
    await _run(env, {1: "a/x9b", 2: "b/better"})
    s = env.sub(2)
    assert s.status != ST.CHAMPION and _rows_of(s) and not _rows_of(s)[0]["ok"] and _rows_of(s)[0]["correct"] > _rows_of(s)[0]["inc_correct"]


async def test_missing_champion_sketch_is_recorded_as_not_comparable(tmp_path):
    """The copy check renews the champion's sketch, so this is rare; if it is gone at decision time the draw is recorded as
    ``comparable: false`` (visible in the audit trail) and the old rules decide that draw."""
    env, _ = await _scenario(tmp_path, NEAR)
    k1 = env.sub(1).key
    real = env.ev._near_champ

    def no_champion_sketch(ch, champ, ch_path=""):
        if champ.key == k1:
            os.remove(env.ev._sketch_path(champ))
            env.ev._near_cache.clear()
        return real(ch, champ, ch_path)

    env.ev._near_champ = no_champion_sketch
    await env.cycle({1: "a/x9b", 2: "b/r5"})
    rows = _rows_of(env.sub(2))
    assert rows and rows[0]["comparable"] is False and rows[0]["vs"] == k1


async def test_layout_that_does_not_map_onto_the_champions_base_is_recorded(tmp_path):
    """critic-nc2 #1: a copy with renamed tensors on a renamed own base compares nothing against the champion's base; it is
    recorded as not comparable (audit trail), never silently treated as far."""
    env, _ = await _scenario(tmp_path, NEAR)
    base, parent = _base(), finetune(_base(), 10)
    ren = lambda w: {"x." + k: v for k, v in w.items()}        # noqa: E731
    fake_base = "evil/base@" + "7" * 40
    write_model(str(tmp_path / "bases" / fake_base.replace("/", "__", 1)), ren(base), dtype="BF16")
    env.add(3, "c/renamed", weights=ren(_orth_copy(parent, base, 0.08, 5)), base_model=fake_base)
    env.server.like["c/renamed"] = ("z/hidden", noisy(0.05))
    await env.cycle({1: "a/x9b", 3: "c/renamed"})
    rows = _rows_of(env.sub(3))
    assert rows and rows[0]["comparable"] is False
