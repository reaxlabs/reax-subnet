#!/usr/bin/env python3
"""C3 synthetic positives built from REAL fingerprints (rows of real models), for both families.

Usage: c3.py <cache dir> <out dir>
Families: qwen  R = Qwen/Qwen3.5-27B, S = baselabs/mammoth-1, A = xflud/sn92-decider-1 (source), B = isomsom/sn92-gen-m87b (own training)
          gemma R = gemma-4-31B pt,    S = gemma-4-31B-it,   A = PelaAI/KnowLine-31B-Gen1,  B = tattrongvu/Jeffan-31b-1710 (own training, 3.4x |dA|)
Every case: child = S + own change + construction; measured against the source A (and the control B). Raw metrics are stored,
so thresholds can be swept offline (c3_report.py)."""
import glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from reax_lineage.fp import Fingerprint, split_layer_name
from reax_lineage.measure import (Thresholds, aligned_keys, copy_stats, delta_vs, frame_check, grid_shares, head_source,
                                  measure_candidate, own_delta, _e4m3_grid)

cache, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
thr = Thresholds.load()
man = {m["repo"]: m for m in json.load(open(os.path.join(HERE, "c1_manifest.json"))) if m.get("ok")}
def load(repo):
    return Fingerprint.load(glob.glob(os.path.join(cache, "%s@%s.A-*.npz" % (repo.replace("/", "__"), man[repo]["rev"])))[0])

FAM = {
    "qwen": dict(R="Qwen/Qwen3.5-27B", S="baselabs/mammoth-1", A="xflud/sn92-decider-1", B="isomsom/sn92-gen-m87b"),
    "gemma": dict(R="google/gemma-4-31B", S="google/gemma-4-31B-it", A="PelaAI/KnowLine-31B-Gen1", B="tattrongvu/Jeffan-31b-1710"),
}
rng = np.random.default_rng(2026)
results = []


def child_fp(proto, body, head=None):
    f = Fingerprint({**proto.meta, "repo": "synthetic/child", "rev": "x"}, body, head if head is not None else {k: v.copy() for k, v in proto.head.items()})
    return f


def q_int(x, bits, g=128):
    y = x.reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(y).max(1, keepdims=True) / qmax + 1e-12
    return (np.clip(np.round(y / s), -qmax - 1, qmax) * s).reshape(x.shape).astype(np.float32)


def q_fp8(x, g=128):
    grid = _e4m3_grid()
    y = x.reshape(-1, g)
    amax = np.abs(y).max(1, keepdims=True) + 1e-12
    u = np.abs(y) / (amax / 448.0)
    pos = np.clip(np.searchsorted(grid, u), 1, len(grid) - 1)
    near = np.where(u - grid[pos - 1] < grid[pos] - u, grid[pos - 1], grid[pos])
    return (np.sign(y) * near * (amax / 448.0)).reshape(x.shape).astype(np.float32)


def bf16(x):
    from reax_lineage.fp import f32_to_bf16_bits, bf16_bits_to_f32
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


for fam, cfg in FAM.items():
    R, S, A, B = (load(cfg[k]) for k in ("R", "S", "A", "B"))
    keys = aligned_keys(R, S, A, B)
    layers = sorted({split_layer_name(k)[0] for k in keys})
    dA = {k: A.body[k] - S.body[k] for k in keys}
    dB = {k: B.body[k] - S.body[k] for k in keys}
    nA = {k: float(np.linalg.norm(dA[k])) for k in keys}
    print(fam, "tensors", len(keys), "layers", len(layers), "|dB|/|dA| (global) = %.2f" % (np.sqrt(sum(float((v ** 2).sum()) for v in dB.values()) / sum(float((v ** 2).sum()) for v in dA.values()))), flush=True)

    def measure(case, child, extra=None):
        """child: Fingerprint. Measure vs source A (own change measured from S) and vs control B, plus copy / grid / frame."""
        kk = aligned_keys(child, S, R, A, B)
        oc = own_delta(child, S, R, kk)
        row = {"family": fam, "case": case, **(extra or {})}
        for nm, P in (("A", A), ("B", B)):
            op = own_delta(P, S, R, kk)
            m = measure_candidate(oc, op, kk, child.n_layers, thr)
            row[nm] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k not in ("layer_beta", "layer_cos")}
            row[nm]["copy"] = copy_stats(child, P, thr, keys=kk)["body"]
            row[nm]["head_source"] = bool(head_source(child, S, P, thr))
        row["copy_vs_S"] = copy_stats(child, S, thr)["body"]
        g = grid_shares(child)
        row["grid"] = {k: round(v, 4) for k, v in g.items() if k != "groups"}
        fr = frame_check(child, S, thr)
        row["frame"] = {k: fr[k] for k in ("frame_share", "scale_share", "ok")}
        results.append(row)
        return row

    # --- merges: linear / task arithmetic / TIES / DARE at 2..50 %, own training = real dB ---
    def ties(d_src, d_own, density=0.5):
        out = {}
        for k in keys:
            a = d_src[k]
            thr_v = np.quantile(np.abs(a), 1 - density)
            keep = np.abs(a) >= thr_v
            elect = np.sign(d_src[k] + d_own[k])
            out[k] = np.where(keep & (np.sign(a) == elect), a, 0.0).astype(np.float32)
        return out

    def dare(d_src, p=0.5):
        return {k: (d_src[k] * (rng.random(d_src[k].shape) > p) / (1 - p)).astype(np.float32) for k in keys}

    dT, dD = ties(dA, dB), dare(dA)
    for share in (0.02, 0.05, 0.10, 0.20, 0.50):
        t0 = time.time()
        for kind in ("linear", "task_arith", "ties", "dare"):
            if kind == "linear":      # (1-s) B + s A  (= S + (1-s) dB + s dA)
                body = {k: S.body[k] + (1 - share) * dB[k] + share * dA[k] for k in keys}
            elif kind == "task_arith":
                body = {k: S.body[k] + dB[k] + share * dA[k] for k in keys}
            elif kind == "ties":
                body = {k: S.body[k] + dB[k] + share * dT[k] for k in keys}
            else:
                body = {k: S.body[k] + dB[k] + share * dD[k] for k in keys}
            measure("merge/%s" % kind, child_fp(A, {**body}), {"share": share, "own": "real dB"})
        # --- own training modelled as Gaussian change of kappa x |dA| per tensor; source share s of dA ---
        for kappa in (0.0, 1.0, 3.0, 10.0):
            body = {k: S.body[k] + share * dA[k] + (rng.standard_normal(dA[k].shape) * nA[k] / np.sqrt(dA[k].size) * kappa).astype(np.float32) for k in keys}
            measure("own-noise", child_fp(A, body), {"share": share, "kappa": kappa})
        print("  share", share, "done %.0fs" % (time.time() - t0), flush=True)

    # --- copies of the source (declared as copy / hidden): resave, jitter, noise, rescale ---
    measure("copy/resave", child_fp(A, {k: A.body[k] for k in keys}))
    measure("copy/jitter1e-6", child_fp(A, {k: bf16(A.body[k] * (1 + 1e-6 * rng.standard_normal(A.body[k].shape)).astype(np.float32)) for k in keys}))
    for rel in (1e-4, 1e-3, 1e-2):
        measure("copy/noise-rel", child_fp(A, {k: bf16(A.body[k] * (1 + rel * rng.standard_normal(A.body[k].shape)).astype(np.float32)) for k in keys}), {"rel_noise": rel})
    for f in (1.0, 3.0, 10.0):
        body = {k: bf16(A.body[k] + (rng.standard_normal(A.body[k].shape) * nA[k] / np.sqrt(A.body[k].size) * f).astype(np.float32)) for k in keys}
        measure("copy/noise-dA", child_fp(A, body), {"noise_x_dA": f})
    # P01 power-of-two rescale of up/gate rows
    body = {k: (A.body[k] * (2.0 ** rng.integers(1, 5, size=(A.body[k].shape[0], 1))).astype(np.float32) if ("up_proj" in k or "gate_proj" in k) else A.body[k]) for k in keys}
    measure("rescale/pow2 up+gate", child_fp(A, body))
    body = {k: (A.body[k] * 1.37 if "mlp" in k else A.body[k]) for k in keys}
    measure("rescale/x1.37 mlp (non power of two)", child_fp(A, body))
    # P05 head scaling: head rows x1.6 with the body untouched
    head = {g: v * 1.6 for g, v in A.head.items()}
    measure("head-scale/x1.6", child_fp(A, {k: A.body[k] for k in keys}, head))
    # --- quantisation round trips (bf16 re-save) of the source ---
    measure("quant/int4", child_fp(A, {k: bf16(q_int(A.body[k], 4)) for k in keys}))
    measure("quant/int8", child_fp(A, {k: bf16(q_int(A.body[k], 8)) for k in keys}))
    measure("quant/fp8-e4m3", child_fp(A, {k: bf16(q_fp8(A.body[k])) for k in keys}))
    measure("quant/int8 fp32-resave", child_fp(A, {k: q_int(A.body[k], 8) for k in keys}))
    measure("quant/int4 of S+dB (not a source)", child_fp(A, {k: bf16(q_int(S.body[k] + dB[k], 4)) for k in keys}))
    # --- splices of A's layers into B ---
    nL = len(layers)
    for label, sel in (("1 layer", {nL // 2}), ("2 layers", {nL // 4, 3 * nL // 4}), ("4 layers", {nL // 8, 3 * nL // 8, 5 * nL // 8, 7 * nL // 8}), ("first half", set(range(nL // 2)))):
        body = {k: (A.body[k] if split_layer_name(k)[0] in sel else B.body[k]) for k in keys}
        measure("splice/" + label, child_fp(A, body), {"n": len(sel)})
    # --- head-only swap: body of B, head of A ---
    measure("head-swap (body B, head A)", child_fp(B, {k: B.body[k] for k in keys}, {g: v.copy() for g, v in A.head.items()}))
    # --- controls: honest fine-tune = B itself, S + small noise ---
    measure("control/B itself (independent fine-tune)", child_fp(B, {k: B.body[k] for k in keys}))
    measure("control/S + own noise 1x|dA|", child_fp(S, {k: S.body[k] + (rng.standard_normal(dA[k].shape) * nA[k] / np.sqrt(dA[k].size)).astype(np.float32) for k in keys}))

json.dump({"thresholds": thr.to_dict(), "cases": results}, open(os.path.join(out, "c3_results.json"), "w"), indent=0)
print("cases:", len(results))
