"""Measurements on fingerprints (LINEAGE-PROCEDURE.md sections 5-6): Tier A canonical form, frame check,
beta/cos decomposition vs a candidate, null level, splice, copy test, quantisation-grid test."""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from .fp import Fingerprint, split_layer_name

THRESHOLDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json")


@dataclass
class Thresholds:
    # source detection (6.2)
    beta_min: float = 0.05           # detected and reported
    beta_action: float = 0.25        # v1.2.0: no longer used by any verdict (every non-copy detection is a weight overlap); kept as the documented value
    cos_min: float = 0.01
    share_min: float = 0.80
    null_x: float = 5.0
    share_layer_beta: float = 0.025
    z_min: float = 15.0              # beta / standard error across layers
    splice_layers: int = 2
    splice_beta: float = 0.5
    splice_cos: float = 0.5
    splice_min_tensors: int = 4      # a transplanted layer changes (nearly) every tensor of the layer; deterministic edits change 1-2
    # start confirmation / frame check (5, 6.2)
    start_beta: float = 0.5
    frame_cos: float = 0.80
    frame_share: float = 0.97
    scale_tol_log2: float = 0.05
    scale_share_max: float = 0.05
    # copy, head, grid
    copy_share: float = 0.999
    head_equal: float = 0.999
    bf16_tol: float = 2.0 ** -8
    grid_share: float = 0.90
    # v1.2.0: shown on the UID page as the old limit for reference; no verdict (3.8: a statistical share alone is no misdeclaration)
    deliberate_share: float = 0.5
    # v1.3.0 (RULES 3.6 final): at most this many weight overlaps join the bar list. Every detection stays listed on the UID page.
    # v1.4.0: the ones with the largest weight in the joint non-negative fit of the entry's own change (ties: larger cos, then earlier commitment;
    # beta if the fit fails for the entry); v1.3.0 ranked by the layer share, which is 1.0 for almost every overlap.
    bar_list_max_overlaps: int = 3
    # rows per tensor and tier B
    tierb_cos: float = 0.80
    null_shifts: tuple = (1, 2, 3, 5, 8, 13, 21)

    @staticmethod
    def load(path=None):
        t = Thresholds()
        p = path or THRESHOLDS_FILE
        if os.path.exists(p):
            d = json.load(open(p))
            for k, v in d.items():
                if k.startswith("_"):
                    continue
                if hasattr(t, k):
                    setattr(t, k, tuple(v) if isinstance(v, list) else v)
        return t

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- Tier A ----

def row_scale(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """a = <x,r>/<r,r> per row."""
    rr = np.einsum("ij,ij->i", r, r)
    xr = np.einsum("ij,ij->i", x, r)
    return np.where(rr > 0, xr / np.where(rr > 0, rr, 1.0), 1.0)


def ortho_delta(x: np.ndarray, r: np.ndarray):
    """Delta of x against the reference row r after dividing the row by its scale a (undoes per-row rescaling).
    Rows whose scale is outside [0.05, 32] in magnitude are not rescaled (unrelated or degenerate rows)."""
    a = row_scale(x, r)
    ae = np.where((np.abs(a) >= 0.05) & (np.abs(a) <= 32.0), a, 1.0).astype(np.float32)
    return (x / ae[:, None] - r).astype(np.float32), a


def aligned_keys(*fps: Fingerprint, keys=None) -> List[str]:
    """Keys present in all fingerprints with the same shape and the same sampled row positions."""
    base = fps[0]
    out = []
    for k in sorted(keys if keys is not None else base.body):
        ok = True
        for f in fps:
            if k not in f.body or f.body[k].shape != base.body[k].shape or f.row_start(k) != base.row_start(k):
                ok = False
                break
        if ok:
            out.append(k)
    return out


def delta_vs(X: Fingerprint, R: Fingerprint, keys) -> Dict[str, np.ndarray]:
    return {k: ortho_delta(X.body[k], R.body[k])[0] for k in keys}


def own_delta(X: Fingerprint, XS, R: Fingerprint, keys) -> Dict[str, np.ndarray]:
    """Own change of X: delta_X - delta_start(X); delta_X if the start is the root, unknown or none.
    v1.6.0: `XS` may also be a LIST of (fingerprint, weight) pairs, the declared parents of a merge with their shares: own change = delta_X - sum(weight * delta_parent) (a parent that is
    the root contributes nothing). A one-element list [(S, 1.0)] is exactly the single-start case."""
    d = delta_vs(X, R, keys)
    pairs = XS if isinstance(XS, list) else ([(XS, 1.0)] if XS is not None else [])
    for P, w in pairs:
        if P is not None and P is not R and P.name != R.name:
            dp = delta_vs(P, R, keys)
            d = {k: d[k] - (dp[k] if w == 1.0 else w * dp[k]) for k in keys}
    return d


def frame_check(C: Fingerprint, S: Fingerprint, thr: Thresholds, keys=None) -> dict:
    """Is C in the coordinates of its declared start S? (share of rows with cosine >= 0.80, scale signature)."""
    keys = aligned_keys(C, S, keys=keys)
    if not keys:
        return {"keys": 0, "rows": 0, "frame_share": 0.0, "scale_share": 1.0, "min_cos": None, "ok": False,
                "reason": "no common tensor shapes"}
    cosr, loga = [], []
    for k in keys:
        x, s = C.body[k], S.body[k]
        num = np.einsum("ij,ij->i", x, s)
        den = np.sqrt(np.einsum("ij,ij->i", x, x) * np.einsum("ij,ij->i", s, s))
        cosr.append(np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0))
        a = row_scale(x, s)
        loga.append(np.abs(np.log2(np.maximum(np.abs(a), 1e-12))))
    cosr, loga = np.concatenate(cosr), np.concatenate(loga)
    share = float((cosr >= thr.frame_cos).mean())
    scale = float((loga > thr.scale_tol_log2).mean())
    cov = len(keys) / max(1, len(C.body))
    return {"keys": len(keys), "coverage": round(cov, 4), "rows": int(cosr.size), "frame_share": round(share, 6),
            "scale_share": round(scale, 6), "min_cos": round(float(cosr.min()), 4),
            "max_abs_log2a": round(float(loga.max()), 4),
            "ok": bool(share >= thr.frame_share and scale <= thr.scale_share_max and cov >= 0.5)}


# --------------------------------------------------------- beta / cos ----

def _per_tensor(oc: Dict, op: Dict, keys, shift_layers=0, layers_total=None):
    """(layer, beta, cos) per tensor. shift>0 pairs the child tensor with layer (l+shift) mod L of the candidate."""
    by = {}
    if shift_layers:
        for k in op:
            l, post = split_layer_name(k)
            by[(post, l)] = k
    out = []
    for k in keys:
        l, post = split_layer_name(k)
        m = k
        if shift_layers:
            m = by.get((post, (l + shift_layers) % layers_total))
            if m is None or oc[k].shape != op[m].shape:
                continue
        elif k not in op:
            continue
        c, p = oc[k].ravel(), op[m].ravel()
        pp = float(p @ p)
        if pp <= 0:
            continue
        cp = float(c @ p)
        cc = float(c @ c)
        out.append((l, cp / pp, cp / (np.sqrt(cc * pp) + 1e-30)))
    return out


def summarise(rows, min_tensors: int = 4) -> dict:
    if not rows:
        return {"n_tensors": 0, "n_layers": 0, "beta": 0.0, "cos": 0.0, "share": 0.0, "se": 0.0, "beta_q10": 0.0,
                "splice_layers": 0, "layers_ge_05": 0, "z": 0.0, "layer_beta": [], "layer_cos": []}
    layers = sorted(set(r[0] for r in rows))
    lb, lc, ln = [], [], []
    for l in layers:
        sel = [r for r in rows if r[0] == l]
        lb.append(float(np.median([r[1] for r in sel])))
        lc.append(float(np.median([r[2] for r in sel])))
        ln.append(len(sel))
    lb, lc, ln = np.array(lb), np.array(lc), np.array(ln)
    se = float(lb.std(ddof=1) / np.sqrt(len(lb))) if len(lb) > 1 else 0.0
    return {"n_tensors": len(rows), "n_layers": len(layers), "beta": float(np.median(lb)), "cos": float(np.median(lc)),
            "share": float((lb >= 0.025).mean()), "se": se, "beta_q10": float(np.quantile(lb, 0.10)),
            "layers_ge_05": int((lb >= 0.5).sum()), "splice_layers": int(((lb >= 0.5) & (lc >= 0.5) & (ln >= min_tensors)).sum()),
            "z": float(float(np.median(lb)) / se) if se > 0 else (1e9 if float(np.median(lb)) > 0 else 0.0),
            "layer_beta": [round(float(x), 5) for x in lb], "layer_cos": [round(float(x), 5) for x in lc]}


def null_level(oc, op, keys, n_layers, thr: Thresholds):
    best = 0.0
    for k in thr.null_shifts:
        if k % max(n_layers, 1) == 0:
            continue
        s = summarise(_per_tensor(oc, op, keys, k, n_layers), thr.splice_min_tensors)
        best = max(best, abs(s["beta"]))
    return best


def main_detected(s: dict, thr: Thresholds) -> bool:
    """The whole-model test (beta, cos, share of layers, z, null): the pattern of a merge, of a continued training and of same-data siblings,
    as opposed to the splice test (a few transplanted layers)."""
    return bool(s["beta"] >= thr.beta_min and s["cos"] >= thr.cos_min and s["share"] >= thr.share_min
                and s.get("z", 1e9) >= thr.z_min and s["beta"] >= thr.null_x * s.get("null", 0.0))


def is_detected(s: dict, thr: Thresholds) -> bool:
    return bool(main_detected(s, thr) or s["splice_layers"] >= thr.splice_layers)


def measure_candidate(oc: Dict, op: Dict, keys, n_layers: int, thr: Thresholds, with_null=True) -> dict:
    """oc = own change of the entry (vs its declared start), op = own change of candidate P."""
    s = summarise(_per_tensor(oc, op, keys), thr.splice_min_tensors)
    s["null"] = 0.0
    if with_null and (s["beta"] >= thr.beta_min and s["cos"] >= thr.cos_min and s["share"] >= thr.share_min):
        s["null"] = null_level(oc, op, keys, n_layers, thr)          # only needed when the main test could fire
    s["main"] = main_detected(s, thr)
    s["detected"] = is_detected(s, thr)
    s["beta_lb"] = s["beta"] - 2 * s["se"]
    return s


# ------------------------------------------------------ joint share ----

def nnls_gram(G: np.ndarray, b: np.ndarray, sweeps: int = 4000, tol: float = 1e-9) -> np.ndarray:
    """min_w>=0 ||c - P w||^2 from the Gram matrix G = P^T P and b = P^T c (cyclic coordinate descent; k is small)."""
    k = len(b)
    w = np.zeros(k)
    d = np.diag(G).copy()
    d[d <= 0] = 1.0
    for _ in range(sweeps):
        delta = 0.0
        for i in range(k):
            r = (b[i] - G[i] @ w + G[i, i] * w[i]) / d[i]
            r = r if r > 0 else 0.0
            delta = max(delta, abs(r - w[i]))
            w[i] = r
        if delta < tol:
            break
    return w


def joint_fit(oc: Dict[str, np.ndarray], ops: Dict[str, Dict[str, np.ndarray]], keys) -> dict:
    """Joint non-negative fit of the entry's own change on the own changes of several candidates at once (rule 3.8(a): the
    share the sources make up *together*). Per layer: minimise sum over the layer's tensors of ||oc_t - sum_i w_i op_i,t||^2 / ||oc_t||^2
    (every tensor counts equally, like the median over tensors in beta). Sources whose own changes overlap (a composite and its
    parents, same-data siblings) share the explained part instead of each claiming all of it, which a plain sum of betas does.
    Returns {"names", "layer_weights" (layers x k)}."""
    names = list(ops)
    k = len(names)
    layers = {}
    for key in keys:
        layers.setdefault(split_layer_name(key)[0], []).append(key)
    W = []
    for l in sorted(layers):
        G, b = np.zeros((k, k)), np.zeros(k)
        for key in layers[l]:
            c = oc[key].ravel().astype(np.float64)
            nc = float(c @ c)
            if nc <= 0 or any(key not in ops[n] for n in names):
                continue
            P = np.stack([ops[n][key].ravel() for n in names]).astype(np.float32)
            G += (P @ P.T).astype(np.float64) / nc
            b += (P @ c.astype(np.float32)).astype(np.float64) / nc
        W.append(nnls_gram(G, b))
    return {"names": names, "layer_weights": np.array(W) if W else np.zeros((0, k))}


def joint_summary(fit: dict, subset) -> dict:
    """Median over layers of the summed weights of `subset` (names), its standard error across layers and the lower bound."""
    idx = [i for i, n in enumerate(fit["names"]) if n in set(subset)]
    W = fit["layer_weights"]
    if W.shape[0] == 0 or not idx:
        return {"share": 0.0, "se": 0.0, "lower_bound": 0.0, "by_model": {}}
    tot = W[:, idx].sum(1)
    se = float(tot.std(ddof=1) / np.sqrt(len(tot))) if len(tot) > 1 else 0.0
    med = float(np.median(tot))
    return {"share": med, "se": se, "lower_bound": med - 2 * se,
            "by_model": {fit["names"][i]: round(float(np.median(W[:, i])), 4) for i in idx}}


# ------------------------------------------------------------- copy ----

def equal_share(x: np.ndarray, y: np.ndarray, tol: float) -> tuple:
    """Values of x equal to y within bf16 rounding after dividing each row of x by its scale vs y."""
    a = row_scale(x, y)
    ae = np.where((np.abs(a) >= 0.05) & (np.abs(a) <= 32.0), a, 1.0).astype(np.float32)
    xx = x / ae[:, None]
    return int((np.abs(xx - y) <= np.abs(y) * tol).sum()), int(y.size)


def copy_stats(C: Fingerprint, P: Fingerprint, thr: Thresholds, keys=None) -> dict:
    keys = aligned_keys(C, P, keys=keys)
    same = tot = 0
    for k in keys:
        a, b = equal_share(C.body[k], P.body[k], thr.bf16_tol)
        same, tot = same + a, tot + b
    body = same / tot if tot else 0.0
    head = {}
    for g in set(C.head) & set(P.head):
        if C.head[g].shape == P.head[g].shape and C.meta["head_blocks"][g][1] == P.meta["head_blocks"][g][1]:
            a, b = equal_share(C.head[g], P.head[g], thr.bf16_tol)
            head[g] = a / b
    cf, pf = C.meta.get("files_sha256") or {}, P.meta.get("files_sha256") or {}
    files_identical = sorted(cf.values()) == sorted(pf.values())
    is_copy = bool(tot > 0 and body >= thr.copy_share and all(v >= thr.head_equal for v in head.values())
                   and files_identical)
    return {"body": round(body, 6), "head": {g: round(v, 6) for g, v in head.items()},
            "files_identical": files_identical, "is_copy": is_copy, "values": tot}


def head_source(C: Fingerprint, S, P: Fingerprint, thr: Thresholds) -> bool:
    """Head rows of C equal (>= 99.9% bf16) those of P, while P's head differs from the start's. v1.6.0: `S` may be a list of the declared parents' fingerprints: P's head must
    differ from every one of them (a head equal to a declared parent's is not "taken from another model")."""
    refs = S if isinstance(S, list) else ([S] if S is not None else [])
    for g in set(C.head) & set(P.head):
        if C.head[g].shape != P.head[g].shape:
            continue
        a, b = equal_share(C.head[g], P.head[g], thr.bf16_tol)
        if a / b < thr.head_equal:
            continue
        if any(g in R_.head and R_.head[g].shape == P.head[g].shape and
               (lambda ab: ab[0] / ab[1] >= thr.head_equal)(equal_share(R_.head[g], P.head[g], thr.bf16_tol)) for R_ in refs if R_ is not None):
            continue          # P's head is a declared parent's head: nothing new
        return True
    return False


# --------------------------------------------------------- quantisation ----

E4M3 = None


def _e4m3_grid():
    global E4M3
    if E4M3 is None:
        vals = {0.0}
        for e in range(0, 16):
            for m in range(8):
                if e == 0:
                    v = (m / 8.0) * 2.0 ** -6
                else:
                    v = (1 + m / 8.0) * 2.0 ** (e - 7)
                if v <= 448.0:
                    vals.add(v)
        E4M3 = np.array(sorted(vals), dtype=np.float32)
    return E4M3


def _groups(x: np.ndarray, g: int) -> np.ndarray:
    r, c = x.shape
    n = (c // g) * g
    return x[:, :n].reshape(r * (n // g), g)


def grid_shares(X: Fingerprint, g: int = 128, tol_rel: float = 2.0 ** -8) -> dict:
    """Share of 128-value groups that lie on a quantisation grid with one scale per group:
    int4 (<= 16 distinct values; symmetric absmax/7 and /8; asymmetric min-max), int8 (absmax/127; min-max), fp8 e4m3."""
    tot = 0
    cnt = {"int4_distinct": 0, "int4_fit": 0, "int8_fit": 0, "fp8_e4m3": 0}
    grid = _e4m3_grid()
    for k, x in X.body.items():
        G = _groups(x, g)
        if G.size == 0:
            continue
        n = G.shape[0]
        tot += n
        srt = np.sort(G, axis=1)
        distinct = 1 + (np.diff(srt, axis=1) != 0).sum(1)
        cnt["int4_distinct"] += int((distinct <= 16).sum())
        amax = np.abs(G).max(1, keepdims=True)
        amax = np.where(amax > 0, amax, 1.0)
        mn, mx = G.min(1, keepdims=True), G.max(1, keepdims=True)

        def fit(y, mag):
            """Group lies on the grid: among values with |x|/step <= 32 (bf16 rounding is finer than the grid there),
            >= 95% are within 0.02 + 2^-8 |x|/step of an integer (rounding error of a bf16 re-save)."""
            sel = mag <= 32.0
            n = sel.sum(1)
            ok = (np.abs(y - np.round(y)) <= 0.02 + tol_rel * mag) & sel
            return (n >= 24) & (ok.sum(1) >= 0.95 * n)

        s7, s8 = amax / 7.0, amax / 8.0
        y7, y8 = G / s7, G / s8
        step_a = np.where(mx > mn, (mx - mn) / 15.0, 1.0)
        ya = (G - mn) / step_a
        ok4 = fit(y7, np.abs(y7)) | fit(y8, np.abs(y8)) | fit(ya, np.abs(G) / step_a)
        cnt["int4_fit"] += int(ok4.sum())
        s127 = amax / 127.0
        y127 = G / s127
        step_b = np.where(mx > mn, (mx - mn) / 255.0, 1.0)
        yb = (G - mn) / step_b
        ok8 = fit(y127, np.abs(y127)) | fit(yb, np.abs(G) / step_b)
        cnt["int8_fit"] += int(ok8.sum())
        yf = np.abs(G) / (amax / 448.0)
        pos = np.clip(np.searchsorted(grid, yf), 1, len(grid) - 1)
        lo, hi = grid[pos - 1], grid[pos]
        near = np.where(yf - lo < hi - yf, lo, hi)
        spacing = np.maximum(hi - lo, 1e-12)
        okf = ((np.abs(yf - near) <= 0.12 * spacing).mean(1) >= 0.95)
        cnt["fp8_e4m3"] += int(okf.sum())
    out = {k: (v / tot if tot else 0.0) for k, v in cnt.items()}
    out["max"] = max(out.values()) if out else 0.0
    out["groups"] = tot
    return out
