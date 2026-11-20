"""Measurements on synthetic tensors: merges, rescale, noise, quantisation, splices, copy, head swap, own-change frames."""
import numpy as np
import pytest

from reax_lineage.fp import f32_to_bf16_bits, bf16_bits_to_f32
from reax_lineage.measure import (Thresholds, aligned_keys, copy_stats, delta_vs, frame_check, grid_shares, head_source,
                                  measure_candidate, own_delta)
from synth import SPEC, delta, combine, finetune, fp_of, make_base

THR = Thresholds()


@pytest.fixture(scope="module")
def world():
    root = make_base(seed=1)
    a = finetune(root, 11, 0.15)          # independent fine-tune A
    b = finetune(root, 12, 0.15)          # independent fine-tune B
    return {"root": root, "A": a, "B": b, "fR": fp_of(root, "org/root"), "fA": fp_of(a, "org/A"), "fB": fp_of(b, "org/B")}


def scan(C, R, P, PS=None, S=None):
    keys = aligned_keys(C, R, P)
    oc = own_delta(C, S, R, keys)
    op = own_delta(P, PS, R, keys)
    return measure_candidate(oc, op, keys, C.n_layers, THR)


def test_independent_pair_not_detected(world):
    s = scan(world["fB"], world["fR"], world["fA"])
    assert not s["detected"]
    assert abs(s["beta"]) < 0.03 and abs(s["cos"]) < 0.02


def test_self_is_beta_one(world):
    s = scan(world["fA"], world["fR"], world["fA"])
    assert s["detected"] and s["beta"] == pytest.approx(1.0, abs=1e-3) and s["cos"] > 0.99


@pytest.mark.parametrize("share", [0.02, 0.05, 0.10, 0.20, 0.50])
def test_merge_share_is_read_to_within_one_percent_point(world, share):
    """C = B + share * delta(A): beta(A) reads the share; >= 10% always detected, 2% never."""
    c = combine(world["B"], (share, delta(world["A"], world["root"])))
    s = scan(fp_of(c, "org/C"), world["fR"], world["fA"])
    assert s["beta"] == pytest.approx(share, abs=0.012)
    if share >= 0.10:
        assert s["detected"]
    if share <= 0.02:
        assert not s["detected"]


def test_merge_under_declared_start_b_is_still_seen(world):
    """Start B declared: own change of C = share * dA; candidate A is still found."""
    c = combine(world["B"], (0.5, delta(world["A"], world["root"])))
    s = scan(fp_of(c, "org/C"), world["fR"], world["fA"], S=world["fB"])
    assert s["detected"] and s["beta"] == pytest.approx(0.5, abs=0.02)


def test_noise_does_not_hide_a_copy(world):
    rng = np.random.default_rng(5)
    for f in (1.0, 3.0):
        c = {k: v + (rng.standard_normal(v.shape) * (world["A"][k] - world["root"][k]).std() * f).astype(np.float32)
             if v.ndim == 2 and "layers." in k else v for k, v in world["A"].items()}
        s = scan(fp_of(c, "org/C"), world["fR"], world["fA"])
        assert s["detected"] and s["beta"] > 0.95


def test_rescale_is_undone_by_tier_a_and_seen_by_scale_signature(world):
    rng = np.random.default_rng(6)
    c = {}
    for k, v in world["A"].items():
        if v.ndim == 2 and ("up_proj" in k or "gate_proj" in k):
            v = v * (2.0 ** rng.integers(1, 5, size=(v.shape[0], 1))).astype(np.float32)
        c[k] = v
    fc = fp_of(c, "org/C")
    s = scan(fc, world["fR"], world["fA"])
    assert s["detected"] and s["beta"] == pytest.approx(1.0, abs=0.01)
    fr = frame_check(fc, world["fA"], THR)
    assert fr["scale_share"] > 0.2 and not fr["ok"]          # reparametrisation shows in the scale signature
    assert frame_check(world["fA"], world["fR"], THR)["ok"]


def test_frame_check_honest_finetune_passes(world):
    fr = frame_check(world["fA"], world["fR"], THR)
    assert fr["ok"] and fr["frame_share"] == 1.0 and fr["scale_share"] == 0.0


def q_int(x, bits, g=128):
    y = x.reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(y).max(1, keepdims=True) / qmax + 1e-12
    return (np.clip(np.round(y / s), -qmax - 1, qmax) * s).reshape(x.shape).astype(np.float32)


@pytest.mark.parametrize("bits", [4, 8])
def test_int_round_trip_is_a_grid_and_still_a_source(world, bits):
    c = {k: (q_int(v, bits) if v.ndim == 2 and "layers." in k else v) for k, v in world["A"].items()}
    fc = fp_of(c, "org/Q")
    g = grid_shares(fc)
    assert g["max"] >= 0.9
    assert grid_shares(world["fA"])["max"] < 0.05 and grid_shares(world["fR"])["max"] < 0.05
    s = scan(fc, world["fR"], world["fA"])
    assert s["detected"] and s["beta"] > 0.8


def test_fp8_round_trip_is_a_grid():
    from reax_lineage.measure import _e4m3_grid
    rng = np.random.default_rng(3)
    base = make_base(seed=2)
    grid = _e4m3_grid()
    c = {}
    for k, v in base.items():
        if v.ndim == 2 and "layers." in k:
            y = v.reshape(-1, 128)
            amax = np.abs(y).max(1, keepdims=True)
            u = np.abs(y) / (amax / 448.0)
            pos = np.clip(np.searchsorted(grid, u), 1, len(grid) - 1)
            near = np.where(u - grid[pos - 1] < grid[pos] - u, grid[pos - 1], grid[pos])
            v = (np.sign(y) * near * (amax / 448.0)).reshape(v.shape).astype(np.float32)
        c[k] = v
    assert grid_shares(fp_of(c, "org/F8"))["fp8_e4m3"] >= 0.9
    assert grid_shares(fp_of(base, "org/F"))["fp8_e4m3"] < 0.05


def test_splice_of_two_layers_is_detected(world):
    c = {k: (world["A"][k] if k.startswith("model.layers.") and int(k.split(".")[2]) in (2, 5) else v)
         for k, v in world["B"].items()}
    s = scan(fp_of(c, "org/C"), world["fR"], world["fA"])
    assert s["splice_layers"] >= 2 and s["detected"]
    # a single spliced layer is below the splice rule and below the median rule
    c1 = {k: (world["A"][k] if k.startswith("model.layers.") and int(k.split(".")[2]) == 3 else v) for k, v in world["B"].items()}
    s1 = scan(fp_of(c1, "org/C1"), world["fR"], world["fA"])
    assert s1["splice_layers"] == 1 and not s1["detected"]


def test_half_splice(world):
    c = {k: (world["A"][k] if k.startswith("model.layers.") and int(k.split(".")[2]) < 4 else v) for k, v in world["B"].items()}
    s = scan(fp_of(c, "org/C"), world["fR"], world["fA"])
    assert s["detected"] and s["splice_layers"] == 4


def test_copy_resave_and_light_noise_is_copy_real_finetune_is_not(world):
    fA = world["fA"]
    assert copy_stats(fp_of(world["A"], "org/C"), fA, THR)["is_copy"]
    rng = np.random.default_rng(9)
    jitter = {k: (v * (1 + 1e-6 * rng.standard_normal(v.shape)).astype(np.float32) if v.ndim == 2 else v) for k, v in world["A"].items()}
    cs = copy_stats(fp_of(jitter, "org/N"), fA, THR)         # float32-level jitter that vanishes in bf16 rounding
    assert cs["is_copy"] and cs["body"] > 0.999
    noisy = {k: (v * (1 + 1e-3 * rng.standard_normal(v.shape)).astype(np.float32) if v.ndim == 2 else v) for k, v in world["A"].items()}
    cn = copy_stats(fp_of(noisy, "org/N3"), fA, THR)         # 0.1% noise flips bf16 roundings: no longer "equal" ...
    assert not cn["is_copy"] and 0.5 < cn["body"] < 0.999
    assert scan(fp_of(noisy, "org/N3"), world["fR"], fA)["beta"] > 0.99     # ... but beta still reads the source
    cs2 = copy_stats(world["fB"], fA, THR)
    assert not cs2["is_copy"] and cs2["body"] < 0.5
    cs3 = copy_stats(fA, world["fR"], THR)       # fine-tune vs its start: far from 99.9%
    assert not cs3["is_copy"]


def test_copy_needs_identical_other_files(world):
    fA = fp_of(world["A"], "org/A", extra_files={"readout.safetensors": b"aaaa"})
    same = fp_of(world["A"], "org/C", extra_files={"readout.safetensors": b"aaaa"})
    other = fp_of(world["A"], "org/D", extra_files={"readout.safetensors": b"bbbb"})
    assert copy_stats(same, fA, THR)["is_copy"]
    assert not copy_stats(other, fA, THR)["is_copy"]      # same body, different readout: not a copy


def test_scaled_copy_is_still_a_copy(world):
    c = {k: (v * 4.0 if v.ndim == 2 and "mlp" in k else v) for k, v in world["A"].items()}
    assert copy_stats(fp_of(c, "org/S"), world["fA"], THR)["is_copy"]          # Tier A divides the row scale out
    c3 = {k: (v * 3.0 if v.ndim == 2 and "mlp" in k else v) for k, v in world["A"].items()}
    s3 = copy_stats(fp_of(c3, "org/S3"), world["fA"], THR)                      # x3 is not exact in bf16: ~6% of values move one ulp
    assert 0.9 < s3["body"] < 0.999 and not s3["is_copy"]
    assert scan(fp_of(c3, "org/S3"), world["fR"], world["fA"])["beta"] > 0.99   # the source is still read by beta


def test_head_swap_is_head_source(world):
    """Body of B, output head of A (A's head differs from the start's): head source test fires."""
    c = dict(world["B"])
    c["lm_head.weight"] = world["A"]["lm_head.weight"]
    fc = fp_of(c, "org/C")
    assert head_source(fc, world["fR"], world["fA"], THR)
    assert not head_source(world["fB"], world["fR"], world["fA"], THR)
    # but a head equal to the start's head is nothing new
    assert not head_source(fp_of(world["B"], "org/C2"), world["fR"], world["fR"], THR)


def test_row_positions_depend_on_seed_not_on_model(world):
    f1 = fp_of(world["A"], "org/A", seed="A0")
    f2 = fp_of(world["B"], "org/B", seed="A0")
    f3 = fp_of(world["A"], "org/A", seed="B-secret", seed_id="B")
    assert f1.meta["row_starts"] == f2.meta["row_starts"]
    assert f1.meta["row_starts"] != f3.meta["row_starts"]
    assert f3.meta["seed"] is None and f3.meta["seed_sha256"] and f1.meta["seed"] == "A0"


def test_name_normalisation_wrapper_vs_text_only():
    base = make_base(seed=3, layers=3)
    wrapped = {k.replace("model.layers", "model.language_model.layers"): v for k, v in base.items()}
    f1, f2 = fp_of(base, "a/b"), fp_of(wrapped, "a/c")
    assert sorted(f1.body) == sorted(f2.body) and f1.n_layers == 3
    assert all(np.array_equal(f1.body[k], f2.body[k]) for k in f1.body)


def test_fingerprint_roundtrip_bf16_exact(tmp_path, world):
    from reax_lineage.fp import Fingerprint
    f = world["fA"]
    f.save(str(tmp_path / "x.npz"))
    g = Fingerprint.load(str(tmp_path / "x.npz"))
    assert g.meta["rev"] == f.meta["rev"] and all(np.array_equal(f.body[k], g.body[k]) for k in f.body)
    assert all(np.array_equal(f.head[k], g.head[k]) for k in f.head)


def test_unrelated_model_fails_frame_check(world):
    other = fp_of(make_base(seed=99), "org/other")
    assert not frame_check(other, world["fR"], THR)["ok"]


def test_own_change_uses_the_candidates_own_start(world):
    """Candidate P = fine-tune of an instruct model T (T = root + big change). beta must be measured on P's OWN change."""
    t = combine(world["root"], (1.0, finetune_delta(world["root"], 21, 3.0)))        # T: big change from root
    p = finetune(t, 22, 0.15)                                                         # P: small change from T
    c = combine(t, (0.5, delta(p, t)))                                                # C = T + 0.5 dP, start T
    fT, fP, fC = fp_of(t, "org/T"), fp_of(p, "org/P"), fp_of(c, "org/C")
    keys = aligned_keys(fC, world["fR"], fP, fT)
    oc = own_delta(fC, fT, world["fR"], keys)
    right = measure_candidate(oc, own_delta(fP, fT, world["fR"], keys), keys, 8, THR)
    wrong = measure_candidate(oc, own_delta(fP, None, world["fR"], keys), keys, 8, THR)   # the prototype bug
    assert right["beta"] == pytest.approx(0.5, abs=0.02) and right["detected"]
    assert not wrong["detected"] or wrong["beta"] < 0.1


def finetune_delta(root, seed, scale):
    t = finetune(root, seed, scale)
    return delta(t, root)


def test_shipped_thresholds_file_equals_the_code_defaults():
    """thresholds.json is part of the script hash: it must say exactly what the code uses, so a reader of the file reads the rules."""
    import json, os
    import reax_lineage.measure as m
    file = {k: v for k, v in json.load(open(m.THRESHOLDS_FILE)).items() if not k.startswith("_")}
    d = m.Thresholds().to_dict()
    assert {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()} == file
    assert m.Thresholds.load().to_dict() == d
