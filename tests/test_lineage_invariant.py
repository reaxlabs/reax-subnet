"""rc11.post5: the weight-lineage checks survive function-preserving reparametrisations (integrity report uid 196, 23 Oct).

Toy Qwen-style decoder (RMSNorm, GQA attention, SwiGLU MLP, untied lm_head) in BF16: base; champ = base + 5 % fine-tune;
copy = champ + a 0.1 % fine-tune of its own. Each attack rewrites the copy without changing its function (checked with a
numpy forward pass); after canonicalisation the sketch comparison with the champion gives exactly the numbers of the
unrewritten copy, and the entry carries a review flag. Honest fine-tunes have an empty plan and the 1.2.0rc11.post4 sketch.
"""

from __future__ import annotations

import numpy as np
import pytest
from lineage_helpers import V, everything, forward, mlp_permute, mlp_rescale, norm_fold, save, toy_base, tune, vo_rescale

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import lineage as L
from reax_subnet.spec import tensors as T

SEED = b"lineage-test-seed"


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineage")
    base = toy_base()
    champ = tune(base, 10, 0.05)
    copy = tune(champ, 11, 0.001)
    other = tune(base, 12, 0.05)
    d = {"base": base, "champ": champ, "copy": copy, "other": other}
    paths = {k: save(str(root / k), v) for k, v in d.items()}
    return root, d, paths


def sketch(path, base_path):
    return T.delta_sketch_ex(path, base_path, seed=SEED)


ATTACKS = {"mlp_rescale": mlp_rescale, "mlp_permute": mlp_permute, "mlp_permute_partial": lambda w: mlp_permute(w, share=0.05),
           "vo_rescale": vo_rescale, "norm_fold": norm_fold, "everything": everything}


@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_attack_preserves_function_and_is_undone(models, attack):
    root, d, p = models
    toks = np.arange(12) % V
    rewritten = ATTACKS[attack](d["copy"])
    np.testing.assert_allclose(forward(rewritten, toks), forward(d["copy"], toks), rtol=1e-6, atol=1e-6)
    path = save(str(root / f"att-{attack}"), rewritten)
    champ = sketch(p["champ"], p["base"])
    plain = T.compare_sketches(champ.deltas, sketch(p["copy"], p["base"]).deltas)
    assert plain.derived() and E.SpecEvaluator._is_near(plain)
    sk = sketch(path, p["base"])
    st = T.compare_sketches(champ.deltas, sk.deltas)
    # exactly the unrewritten copy's comparison (the positions are the same, the values are mapped back exactly)
    assert (st.cosine, st.relative_l2, st.alpha, st.compared) == (plain.cosine, plain.relative_l2, plain.alpha, plain.compared)
    assert st.derived() and E.SpecEvaluator._is_near(st)
    assert sk.review, "a rewritten copy is flagged for review"
    # the other direction too: the rewritten entry as the earlier one
    back = T.compare_sketches(sk.deltas, champ.deltas)
    assert back.derived()


def test_without_canonicalisation_the_attack_hides_the_copy(models, monkeypatch):
    """The report's finding, as a regression guard for the test itself: with the plan switched off nothing fires."""
    root, d, p = models
    path = save(str(root / "att-raw"), mlp_rescale(d["copy"]))
    monkeypatch.setattr(L, "plan", lambda *a, **k: L.Plan())
    st = T.compare_sketches(sketch(p["champ"], p["base"]).deltas, sketch(path, p["base"]).deltas)
    assert not st.derived() and not E.SpecEvaluator._is_near(st)


def test_honest_fine_tunes_have_an_empty_plan_and_the_old_sketch(models, monkeypatch):
    root, d, p = models
    for k in ("champ", "copy", "other"):
        pl = L.plan(T._canonical(T.read_model(p[k])), T._canonical(T.read_model(p["base"])), p[k], p["base"])
        assert pl.transforms == {} and pl.flags == [] and pl.pow2_units == 0 and pl.units > 0
    new = {k: sketch(p[k], p["base"]) for k in ("champ", "other")}
    assert all(not s.review for s in new.values())
    monkeypatch.setattr(L, "plan", lambda *a, **k: L.Plan())
    for k, s in new.items():
        old = sketch(p[k], p["base"])
        assert old.deltas.keys() == s.deltas.keys() and old.changed == s.changed
        assert all(old.deltas[n].tobytes() == s.deltas[n].tobytes() for n in old.deltas)
    st = T.compare_sketches(new["champ"].deltas, new["other"].deltas)
    assert not st.derived() and not E.SpecEvaluator._is_near(st)


def test_rewritten_independent_fine_tune_is_not_made_a_copy(models):
    """Canonicalisation must not create false lineage: an independent fine-tune, rescaled, stays independent (flagged)."""
    root, d, p = models
    path = save(str(root / "att-other"), everything(d["other"]))
    sk = sketch(path, p["base"])
    st = T.compare_sketches(sketch(p["champ"], p["base"]).deltas, sk.deltas)
    assert sk.review and not st.derived() and not E.SpecEvaluator._is_near(st)


def test_non_power_of_two_rescale_is_flagged_and_mostly_undone(models):
    """s_i in [1.3, 3] (not exact in BF16, the function moves by rounding only): the anticorrelated rule undoes it with the
    measured factor; the comparison is close to the unrescaled copy's and the entry is flagged."""
    root, d, p = models
    w = mlp_rescale(d["copy"], factors=lambda rng, n: rng.uniform(1.3, 3.0, n))
    path = save(str(root / "att-nonpow2"), w)
    sk = sketch(path, p["base"])
    assert any(f.startswith("rescale_anticorrelated") for f in sk.review)
    st = T.compare_sketches(sketch(p["champ"], p["base"]).deltas, sk.deltas)
    assert st.cosine > 0.98 and E.SpecEvaluator._is_near(st)


def test_large_raw_delta_is_flagged(models):
    root, d, p = models
    sk = sketch(p["base"], save(str(root / "far-base"), toy_base(99)))
    assert any(f.startswith("delta_norm:") for f in sk.review)
    assert not sketch(p["champ"], p["base"]).review


def test_plan_is_memory_bounded(models, monkeypatch):
    """A tiny chunk bound gives the same plan (every reader works a chunk of rows at a time)."""
    root, d, p = models
    path = save(str(root / "att-chunks"), everything(d["copy"]))
    m, b = T._canonical(T.read_model(path)), T._canonical(T.read_model(p["base"]))
    big = L.plan(m, b, path, p["base"])
    monkeypatch.setattr(L, "ROW_CHUNK_ELEMS", 100)
    small = L.plan(m, b, path, p["base"])
    assert big.transforms.keys() == small.transforms.keys() and big.flags == small.flags
    for k, t in big.transforms.items():
        u = small.transforms[k]
        for f in ("rows", "cols", "row_f", "col_f"):
            a, c = getattr(t, f), getattr(u, f)
            assert (a is None) == (c is None) and (a is None or np.array_equal(a, c))
