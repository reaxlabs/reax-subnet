"""Golden tests on three real cached fingerprints (Sample A, seed A0, 4 rows per tensor, Qwen3.5-27B line):
baselabs/mammoth-1 (root), xflud/sn92-decider-1 (uid 114) and isomsom/sn92-gen-m87b (uid 60), both independent
fine-tunes of mammoth-1. Expected numbers: LINEAGE-PROCEDURE.md 6.4 (prototype) and calibration/c1_results.json."""
import glob
import os

import numpy as np
import pytest

from reax_lineage.fp import Fingerprint
from reax_lineage.measure import (Thresholds, aligned_keys, copy_stats, delta_vs, frame_check, grid_shares,
                                  measure_candidate, own_delta)

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
THR = Thresholds()


@pytest.fixture(scope="module")
def fps():
    def load(pat):
        matches = glob.glob(os.path.join(GOLD, pat + "*.npz"))
        if not matches:
            pytest.skip("Sampled third-party fingerprints are omitted from the public distribution")
        return Fingerprint.load(matches[0])
    return {"R": load("baselabs__mammoth-1"), "A": load("xflud__sn92-decider-1"), "B": load("isomsom__sn92-gen-m87b")}


def scan(C, R, P, S=None, PS=None):
    keys = aligned_keys(C, R, P)
    return measure_candidate(own_delta(C, S, R, keys), own_delta(P, PS, R, keys), keys, C.n_layers, THR)


def test_shape_of_the_sample(fps):
    for f in fps.values():
        assert f.meta["n_tensors"] == 400 and f.meta["values"] == 11_599_872 and f.n_layers == 64
        assert f.meta["seed"] == "A0" and f.head["embed"].shape == (1024, 5120)


def test_independent_real_finetunes_are_not_sources(fps):
    s = scan(fps["A"], fps["R"], fps["B"])           # prototype: beta +0.0009 (114 vs 60)
    assert not s["detected"] and abs(s["beta"]) < 0.005 and abs(s["cos"]) < 0.01
    s2 = scan(fps["B"], fps["R"], fps["A"])
    assert not s2["detected"] and abs(s2["beta"]) < 0.005


def test_real_finetune_is_in_the_frame_of_its_start(fps):
    fr = frame_check(fps["A"], fps["R"], THR)        # prototype: share 1.0000, scale 0.0000, min cos 0.9999
    assert fr["ok"] and fr["frame_share"] == 1.0 and fr["scale_share"] == 0.0 and fr["min_cos"] > 0.99


def test_candidate_beta_of_itself_is_one(fps):
    s = scan(fps["A"], fps["R"], fps["A"])
    assert s["detected"] and s["beta"] == pytest.approx(1.0, abs=1e-3) and s["cos"] > 0.999 and s["splice_layers"] == 64


@pytest.mark.parametrize("wa,wb", [(0.5, 0.5), (0.7, 0.7), (1.0, 0.3)])
def test_task_arithmetic_recipe_is_recovered(fps, wa, wb):
    """round-1 recipes (gen-ma 0.5/0.5, gen-mb 0.7/0.7, gen-me 1.0/0.3): the coefficients are read back."""
    R, A, B = fps["R"], fps["A"], fps["B"]
    child = Fingerprint({**R.meta, "repo": "synthetic/merge", "rev": "x"}, {k: R.body[k] + wa * (A.body[k] - R.body[k]) + wb * (B.body[k] - R.body[k]) for k in R.body})
    child.meta["row_starts"] = R.meta["row_starts"]
    sa, sb = scan(child, R, A), scan(child, R, B)
    assert sa["detected"] and sb["detected"]
    assert sa["beta"] == pytest.approx(wa, abs=0.02) and sb["beta"] == pytest.approx(wb, abs=0.02)


def test_real_finetune_is_not_a_copy_of_its_start(fps):
    cs = copy_stats(fps["A"], fps["R"], THR)         # prototype: 0.5098 of the values equal within bf16 rounding
    assert 0.3 < cs["body"] < 0.7 and not cs["is_copy"]


def test_real_models_are_not_on_a_quantisation_grid(fps):
    for f in fps.values():
        assert grid_shares(f)["max"] < 0.01


def test_int4_round_trip_of_a_real_model_is_a_grid_and_still_a_source(fps):
    R, A = fps["R"], fps["A"]

    def q4(x, g=128):
        y = x.reshape(-1, g)
        s = np.abs(y).max(1, keepdims=True) / 7.0 + 1e-12
        return (np.clip(np.round(y / s), -8, 7) * s).reshape(x.shape).astype(np.float32)
    child = Fingerprint({**A.meta, "repo": "synthetic/int4", "rev": "x"}, {k: q4(v) for k, v in A.body.items()})
    assert grid_shares(child)["max"] > 0.95
    s = scan(child, R, A)
    assert s["detected"] and s["beta"] == pytest.approx(1.03, abs=0.06)         # prototype: 1.0337


def test_power_of_two_rescale_is_undone(fps):
    rng = np.random.default_rng(7)
    R, A = fps["R"], fps["A"]
    body = {k: (v * (2.0 ** rng.integers(1, 5, size=(v.shape[0], 1))).astype(np.float32) if ("up_proj" in k or "gate_proj" in k) else v)
            for k, v in A.body.items()}
    child = Fingerprint({**A.meta, "repo": "synthetic/rescale", "rev": "x"}, body)
    s = scan(child, R, A)
    assert s["detected"] and s["beta"] == pytest.approx(1.0, abs=0.01)                 # prototype: 1.0000
    assert frame_check(child, A, THR)["scale_share"] > 0.3 and not frame_check(child, A, THR)["ok"]


def test_four_layer_splice_is_found(fps):
    R, A, B = fps["R"], fps["A"], fps["B"]
    layers = (8, 24, 40, 56)
    body = {k: (A.body[k] if int(k.split(".")[1]) in layers else B.body[k]) for k in B.body}
    child = Fingerprint({**B.meta, "repo": "synthetic/splice", "rev": "x"}, body)
    s = scan(child, R, A, S=None)
    assert s["splice_layers"] == 4 and s["detected"]                                  # prototype: splice 4
