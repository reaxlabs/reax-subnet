"""Near-copy rules R1/R2/R3 (REAX_SPEC_NC_R1/R2/R3, off by default; docs/SPECIALIST-TRACKS.md "Near-duplicates").

Regression shape of the live case (DF track, 19 Oct 2026): a champion x9b, a copy of it with the letter rows rescaled (temperature)
and a little orthogonal weight noise, which gains a few points of calibration and used to be crowned over it."""

import dataclasses

import numpy as np
import pytest
from spec_helpers import finetune
from test_copycheck2 import Server, _base, _rows, _server_env, noisy, temperature
from test_spec_evaluator import TRACK

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST
from reax_subnet.spec import tensors as T

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

ALL = dict(nc_r1=True, nc_r2=True, nc_r3=True)


def _flags(env, **kw):
    env.cfg = dataclasses.replace(env.cfg, **kw)
    env.ev.cfg = env.cfg


def _orth_copy(parent, base, rel, seed):
    """The parent's delta plus a noise delta of ``rel`` x its norm (random in a high dimension: orthogonal to it)."""
    rng = np.random.RandomState(seed)
    out = {}
    for k, v in parent.items():
        d = v - base[k]
        n = rng.randn(*d.shape).astype(np.float32)
        out[k] = base[k] + d + n * (rel * np.linalg.norm(d) / np.linalg.norm(n))
    return out


# ------------------------------------------------------------------------------------------------ flags / defaults
def test_flags_default_off_and_parse(tmp_path):
    cfg = C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path)})
    assert (cfg.nc_r1, cfg.nc_r2, cfg.nc_r3) == (False, False, False)
    on = C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_NC_R1": "1", "REAX_SPEC_NC_R2": "on", "REAX_SPEC_NC_R3": "true"})
    assert (on.nc_r1, on.nc_r2, on.nc_r3) == (True, True, True)
    with pytest.raises(ValueError):
        C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_NC_R2": "maybe"})


# ------------------------------------------------------------------------------------------------ R2
def test_r2_temperature_copy_is_a_twin_only_with_the_flag():
    rows = _rows()
    srv = Server({"m": 0.8})
    srv.like = {"c": ("m", temperature(2.0))}
    a, c = srv.answers("m", rows), srv.answers("c", rows)
    off = E.behaviour_stats(rows, a, c)
    on = E.behaviour_stats(rows, a, c, temp_invariant=True, rel_tie=True)
    assert not off.twin() and off.jsd_temp_bits is None and "jsd_temp" not in off.as_dict()
    assert on.twin() and on.jsd_temp_bits < E.TWIN_JSD_BITS / 10 and on.as_dict()["temp_scale"] == pytest.approx(0.5)
    assert on.jsd_bits == off.jsd_bits and on.same_order == off.same_order


def test_r2_independent_finetunes_are_not_twins():
    rows = _rows()
    srv = Server()
    srv.logit = {"m": (1, 0.5), "i03": (2, 0.3), "i05": (3, 0.5)}
    for x, y in (("m", "i03"), ("m", "i05"), ("i03", "i05")):
        st = E.behaviour_stats(rows, srv.answers(x, rows), srv.answers(y, rows), temp_invariant=True, rel_tie=True)
        assert not st.twin() and st.jsd_temp_bits > E.TWIN_JSD_BITS, (x, y, st)


def test_r2_relative_tie_fixes_rounding_induced_order_changes():
    assert E._order([1e-9, 3e-9, 0.5]) == (0, -1, -1)                       # old absolute tie 1e-6: the tail order is hidden
    assert E._order([1e-9, 3e-9, 0.5], rel_tie=True) == (-1, -1, -1)         # R2: 3x is not a tie
    assert E._order([1e-9, 1e-9 * (1 + 1e-8), 0.5], rel_tie=True) == (0, -1, -1)   # a relative difference of 1e-8 is
    assert E._order([0.5, 0.25, 0.25]) == E._order([0.5, 0.25, 0.25], rel_tie=True)  # ordinary values: same verdict


# ------------------------------------------------------------------------------------------------ R3 (weights)
def test_r3_alpha_catches_orthogonal_noise_only_with_the_tolerance():
    rng = np.random.RandomState(0)
    a = {"x": rng.randn(4000).astype(np.float32), "y": rng.randn(4000).astype(np.float32)}

    def noisy(scale, rel):
        return {k: scale * v + rel * np.linalg.norm(v) * rng.randn(*v.shape).astype(np.float32) / np.sqrt(v.size) for k, v in a.items()}

    copy = T.compare_sketches(a, noisy(1.0, 0.15))
    assert copy.cosine > 0.98 and copy.relative_l2 > 0.05 and abs(copy.alpha - 1) < 0.05
    assert not copy.derived() and copy.derived(alpha_tol=0.05)
    cont = T.compare_sketches(a, noisy(1.4, 0.15))               # continued training in the same direction: a different length
    assert cont.cosine > 0.98 and abs(cont.alpha - 1) > 0.3 and not cont.derived(alpha_tol=0.05)
    other = T.compare_sketches(a, noisy(0.0, 1.0))
    assert not other.derived(alpha_tol=0.05)
    assert T.compare_sketches(a, a).alpha == pytest.approx(1.0) and T.compare_sketches(a, a).derived()


# ------------------------------------------------------------------------------------------------ the live case
async def _crown(env, uid, repo, skill, weights):
    env.add(uid, repo, skill, weights=weights)
    await env.cycle({uid: repo})
    env.clock.advance(13 * 3600)
    await env.cycle({uid: repo})


@pytest.fixture(autouse=True)
def small_margin(monkeypatch):
    """The fake models' calibration gains are small: a lower margin floor lets the copy's +0.8 clear the total margin, so only
    the near-copy rules can stop it (the live copy gained +2.19)."""
    monkeypatch.setattr(M, "MARGIN_FLOOR", 0.3)
    monkeypatch.setattr(M, "SE_MULTIPLE", 1.0)


async def _scenario(tmp_path, flags):
    """x9b is the champion; r5 = x9b + rescaled letters + orthogonal weight noise (not a twin by the old test: jsd 0.08, order changed)."""
    base = _base()
    env = _server_env(tmp_path, base)
    _flags(env, **flags)
    parent = finetune(base, 10)
    env.server.skills["z/hidden"] = 0.65     # the answers of x9b: an over-sharpened version of a hidden model; r5's are the same model's own
    env.server.like["a/x9b"] = ("z/hidden", temperature(0.3))
    await _crown(env, 1, "a/x9b", None, parent)
    assert env.sub(1).status == ST.CHAMPION
    env.add(2, "b/r5", weights=_orth_copy(parent, base, 0.15, 5))
    env.server.like["b/r5"] = ("z/hidden", noisy(0.05))
    return env


async def _run(env, commits):
    await env.cycle(commits)
    env.clock.advance(13 * 3600)
    await env.cycle(commits)
    env.clock.advance(3601)
    return await env.cycle(commits)


async def test_live_case_flags_off_the_copy_is_crowned(tmp_path):
    env = await _scenario(tmp_path, {})
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    assert env.sub(2).derived_from == ""            # neither the weights nor the behaviour call it derived: the live failure
    assert env.sub(2).status == ST.CHAMPION and env.ev.state.track(TRACK).champion == env.sub(2).key


async def test_live_case_all_rules_the_copy_is_not_crowned(tmp_path):
    env = await _scenario(tmp_path, ALL)
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    s = env.sub(2)
    assert s.derived_from == env.sub(1).key and s.status != ST.CHAMPION
    assert env.ev.state.track(TRACK).champion == env.sub(1).key


async def test_live_case_r1_alone_stops_the_weight_derived_copy_of_the_incumbent(tmp_path):
    """R1 + R3 (weights) and no R2: the weight parent is the incumbent, which used to be exempt from the parent rule."""
    env = await _scenario(tmp_path, dict(nc_r1=True, nc_r3=True))
    await _run(env, {1: "a/x9b", 2: "b/r5"})
    assert env.ev.state.track(TRACK).champion == env.sub(1).key
    gates = [h for h in env.sub(2).history if h["kind"] == "comp_gate"]
    assert gates and gates[0]["vs"] == env.sub(1).key and gates[0]["comp_gain"] < E.TWIN_COMP_FLOOR


async def test_independent_better_model_is_still_crowned_with_all_rules(tmp_path):
    env = await _scenario(tmp_path, ALL)
    env.server.like.pop("b/r5")
    env.add(3, "c/indep", 0.97, weights=finetune(_base(), 77))
    commits = {1: "a/x9b", 3: "c/indep"}
    await _run(env, commits)
    assert env.sub(3).status == ST.CHAMPION and env.sub(3).derived_from == ""
    assert not [h for h in env.sub(3).history if h["kind"] == "comp_gate" and h["comp_gain"] < E.TWIN_COMP_FLOOR]


async def test_real_improvement_of_the_champions_own_line_still_wins_with_r1(tmp_path):
    """A weight-derived entry that really answers better (competence), not just calibrates, passes the R1 gate."""
    base = _base()
    env = _server_env(tmp_path, base)
    _flags(env, **ALL)
    parent = finetune(base, 10)
    await _crown(env, 1, "a/x9b", 0.6, parent)
    env.add(2, "b/better", 0.98, weights=_orth_copy(parent, base, 0.15, 5))
    await _run(env, {1: "a/x9b", 2: "b/better"})
    assert env.sub(2).derived_from == env.sub(1).key and env.sub(2).status == ST.CHAMPION


# ------------------------------------------------------------------------------------------------ R3 (transitive)
async def test_r3_transitive_lineage_inherits_the_root_parent(tmp_path):
    """r4 is not derived from x9b by itself (noise above the tolerance), r5 is derived from r4 only: with R3 the lineage
    r5 -> r4 is kept, and a derived_from chain r5 -> r4 -> x9b makes x9b a must-beat for r5."""
    base = _base()
    env = _server_env(tmp_path, base)
    _flags(env, **ALL)
    await _crown(env, 1, "a/x9b", 0.7, finetune(base, 10))
    ts = env.ev.state.track(TRACK)
    k1 = env.sub(1).key
    env.add(2, "b/r4", 0.7)
    env.add(3, "c/r5", 0.7)
    await env.cycle({1: "a/x9b", 2: "b/r4", 3: "c/r5"})
    r4, r5 = env.sub(2), env.sub(3)
    r4.derived_from, r4.derived_how = k1, "weights"
    r5.derived_from, r5.derived_how = r4.key, "weights"
    assert env.ev._lineage(ts, r5.key) == [r5.key, r4.key, k1]
    r4.derived_from = r5.key                                # a cycle ends the walk
    assert env.ev._lineage(ts, r5.key) == [r5.key, r4.key]


async def test_r3_transitive_chain_is_beaten_end_to_end(tmp_path):
    """x9b (champion) <- r4 (weights) <- r5 (weights of r4 plus noise; too far from x9b to be derived from it directly): r5 must
    beat x9b as well, on the draw, through the lineage."""
    env = await _scenario(tmp_path, ALL)
    parent = finetune(_base(), 10)
    r4 = _orth_copy(parent, _base(), 0.15, 5)
    env.add(2, "b/r4", weights=r4)
    env.server.like["b/r4"] = ("z/hidden", noisy(0.05, 1))
    env.add(3, "c/r5", weights=_orth_copy(r4, _base(), 0.15, 6))
    env.server.like["c/r5"] = ("z/hidden", noisy(0.05, 2))
    commits = {1: "a/x9b", 2: "b/r4", 3: "c/r5"}
    await _run(env, commits)
    x, r4s, r5s = env.sub(1), env.sub(2), env.sub(3)
    assert r4s.derived_from == x.key and r5s.derived_from != "" and env.ev.state.track(TRACK).champion == x.key
    assert env.ev._lineage(env.ev.state.track(TRACK), r5s.key)[-1] == x.key
    assert x.key in {h["vs"] for h in r5s.history if h["kind"] == "comp_gate"}


async def test_leaders_export_the_crowning_times(tmp_path):
    """(c) needs them: ``RankedLeaders.crowned`` is uid -> champion_since_ts of the entries in the ranks."""
    env = await _scenario(tmp_path, {})
    out = await env.cycle({1: "a/x9b"})
    leaders = out[TRACK]
    assert [u for u, _ in leaders] == [1] and leaders.crowned == {1: env.sub(1).champion_since_ts}
