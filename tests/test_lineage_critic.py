"""rc11.post5 critic fixes (lineage-critic, 23 Oct): robustness and false positives of the reparametrisation checks.

F1 a declared base is any public repo: its config (head size) and shapes are hostile input, and an exception in the plan
   must not abort the track's cycle at one entry for ever.
F2 an honest model with pruned or dead MLP units (all-zero rows) is not a permutation.
F3 a small rescale (below 1.4 % rms) that swamps a 0.2 % fine-tune is undone and flagged.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from lineage_helpers import HD, I, LAYERS, mlp_rescale, save, toy_base, tune

from reax_subnet.spec import lineage as L
from reax_subnet.spec import tensors as T

SEED = b"critic-seed"


def _sk(root, name, w, base_path):
    return T.delta_sketch_ex(save(str(root / name), w), base_path, seed=SEED)


# ---------------------------------------------------------------- F1
@pytest.mark.parametrize("head_dim", [7, 24, 1000, 10 ** 30, "x", -1, 0, None])
def test_hostile_base_head_size_is_ignored(tmp_path, head_dim):
    base = toy_base()
    bpath = save(str(tmp_path / "base"), base)
    cfg = json.load(open(os.path.join(bpath, "config.json")))
    cfg["head_dim"] = head_dim
    json.dump(cfg, open(os.path.join(bpath, "config.json"), "w"))
    sk = _sk(tmp_path, "m", tune(base, 3, 0.01), bpath)
    assert sk.review == ()


def test_hostile_base_q_norm_shape_is_ignored(tmp_path):
    base = toy_base()
    for i in range(LAYERS):
        base[f"model.layers.{i}.self_attn.q_norm.weight"] = np.ones(HD + 8, np.float32)
    bpath = save(str(tmp_path / "base"), base)
    assert _sk(tmp_path, "m", tune(base, 3, 0.01), bpath).review == ()


def test_plan_exception_is_held_not_raised(tmp_path, monkeypatch):
    base = toy_base()
    bpath = save(str(tmp_path / "base"), base)
    m = tune(base, 3, 0.01)

    def boom(*a, **k):
        raise RuntimeError("odd layout")

    plain = _sk(tmp_path, "plain", m, bpath)
    monkeypatch.setattr(L, "plan", boom)
    sk = _sk(tmp_path, "m", m, bpath)
    assert sk.review == ("plan_error:RuntimeError",)
    assert all(np.array_equal(sk.deltas[k], plain.deltas[k]) for k in plain.deltas)    # the raw sketch


# ---------------------------------------------------------------- F2
def _prune_units(w, frac, seed=3):
    rng = np.random.RandomState(seed)
    w = dict(w)
    for i in range(LAYERS):
        p = f"model.layers.{i}.mlp"
        sel = rng.choice(I, int(frac * I), replace=False)
        for k in ("gate", "up"):
            a = np.array(w[f"{p}.{k}_proj.weight"])
            a[sel] = 0
            w[f"{p}.{k}_proj.weight"] = a
        a = np.array(w[f"{p}.down_proj.weight"])
        a[:, sel] = 0
        w[f"{p}.down_proj.weight"] = a
    return w


@pytest.mark.parametrize("frac", [0.02, 0.1, 0.3])
def test_structured_pruning_is_not_a_permutation(tmp_path, frac):
    base = toy_base()
    bpath = save(str(tmp_path / "base"), base)
    assert _sk(tmp_path, "m", _prune_units(tune(base, 3, 0.02), frac), bpath).review == ()


def test_dead_base_units_are_not_a_permutation(tmp_path):
    base = _prune_units(toy_base(), 0.1)
    bpath = save(str(tmp_path / "base"), base)
    assert _sk(tmp_path, "same", dict(base), bpath).review == ()
    assert _sk(tmp_path, "ft", tune(base, 4, 0.02), bpath).review == ()


# ---------------------------------------------------------------- F3
def _small_rescale(eps):
    return lambda rng, n: np.exp2(rng.uniform(-eps, eps, n))


def test_small_rescale_is_undone_and_flagged(tmp_path, monkeypatch):
    """A per-unit rescale of up to 1 % (below the old 1.4 % floor) moves the delta of a 0.4 % fine-tune further than the
    fine-tune itself, and is function-preserving to BF16 precision: it must still be traced back and flagged."""
    base = toy_base()
    bpath = save(str(tmp_path / "base"), base)
    champ = tune(base, 20, 0.004)
    copy = tune(champ, 21, 0.001)
    ck = _sk(tmp_path, "champ", champ, bpath)
    plain = T.compare_sketches(ck.deltas, _sk(tmp_path, "copy", copy, bpath).deltas)
    attack = mlp_rescale(copy, 1, _small_rescale(0.014))
    sk = _sk(tmp_path, "att", attack, bpath)
    st = T.compare_sketches(ck.deltas, sk.deltas)
    assert any(f.startswith("rescale_anticorrelated") for f in sk.review)
    assert st.cosine > 0.97 and st.cosine > plain.cosine - 0.02
    monkeypatch.setattr(L, "ANTI_RMS_MIN", 0.02)     # the rc11.post5 floor: not undone
    old = T.compare_sketches(ck.deltas, _sk(tmp_path, "att-old", attack, bpath).deltas)
    assert old.cosine < 0.95


@pytest.mark.parametrize("rel,seed", [(0.002, 5), (0.05, 6), (0.3, 7)])
def test_honest_fine_tunes_stay_unflagged_at_the_low_floor(tmp_path, rel, seed):
    base = toy_base()
    bpath = save(str(tmp_path / "base"), base)
    assert _sk(tmp_path, "m", tune(base, seed, rel), bpath).review == ()
