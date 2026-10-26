"""Function-preserving reparametrisations, undone before the weight delta is sketched (rc11.post5, integrity report uid 196).

A transformer computes the same function after several exact rewrites of its weights, and each of them moves the raw delta
from the declared base so far that the delta sketch (``tensors.delta_sketch_ex``) no longer sees the model's parent:

* MLP channel scaling: row i of ``up_proj`` times s_i, column i of ``down_proj`` divided by s_i (SwiGLU is linear in up);
  s_i = +-2^k is exact in BF16. The gate cannot be rescaled (SiLU is not homogeneous), so it is never touched.
* MLP hidden-unit permutation: the same permutation of the gate and up rows and the down columns.
* attention value/output scaling: row c of ``v_proj`` (and its bias) times s_c, the ``o_proj`` columns that read value
  channel c (every query head of its KV group) divided by s_c.
* RMSNorm folding: norm weight j times t_j, column j of the projections that read the normed stream divided by t_j
  (``input_layernorm`` -> q/k/v, ``post_attention_layernorm`` -> gate/up, ``model.norm`` -> an untied ``lm_head``).

``plan`` measures the entry against its declared base, tensor by tensor (memory-bounded row chunks, the whole tensor is
read: the secret-seeded sample positions stay where they are, only the values read at them change), and returns a
``Transform`` per tensor that maps the entry back onto the base's parametrisation. A pair of factors is undone when both
sides agree on an exact signed power of two (``+-2^k``, k != 0 or a sign flip): an honest fine-tune never moves a row
norm by a factor sqrt(2) and the inverse on the paired column, so its plan is empty and its sketch is byte-identical to
1.2.0rc11.post4. A non-power-of-two rescale (inexact in BF16, but close) is undone with the measured factor when the row
and column ratios of a layer are anticorrelated (the signature of a rescale; the scaling symmetry makes the norms of an
honest fine-tune move together, not against each other), and it always raises a review flag. A permutation is applied
when the gate rows of the entry match other base rows than their own.

``Plan.flags`` (empty for an honest fine-tune) names what was found; the evaluator holds a flagged entry for a reviewer
(``review.py``): it is drawn, but never confirmed, crowned or kept alive until the flag is cleared.

Not covered (open, see the rc11.post5 notes): sign/scale tricks on q/k (RoPE pairs, per-head scale under QK-norm), the
residual-stream basis (a rotation of every hidden dimension), fused projections (``gate_up_proj``, ``qkv_proj``), expert
permutations in MoE routers and (1 + w) norms (Gemma); such an entry keeps the raw delta and, when it is far from its
base, the ``delta_norm`` flag.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import tensors as T

ROW_CHUNK_ELEMS = 16 * 1024 * 1024   # values per chunk read (64 MB as float32): the memory bound of the whole plan
PERM_COS = 0.5          # a gate row whose cosine with its own base row is below this is matched against every base row
PERM_SHARE_FLAG = 0.01  # unmatched units above this share (of a layer) that cannot be matched one-to-one: flag
PROJ_DIM = 256          # random projection used to find the best base row of an unmatched gate row
POW2_SHARE_FLAG = 0.01  # share of the units of the model undone as exact +-2^k pairs above which the entry is flagged
ANTI_RMS_MIN = 0.0005   # anticorrelated rescale: rms of log2 row ratio above this (0.035 %; a rescale of 1 % swamps a 0.2 %
                        # fine-tune, and the ratio test below, not this floor, tells a rescale from noise) ...
ANTI_RATIO_MAX = 0.1    # ... and mean((log2 row + log2 col)^2) below this share of mean((log2 row)^2)
NORM_EPS_REL = 0.05     # a norm channel whose base weight is below this share of the rms is never refolded
DELTA_REL_FLAG = 0.5    # ||raw delta|| / ||base|| over the sampled values of the recognised blocks above this: no
                        # plausible fine-tune; flag (a layout without recognised blocks is not judged)


@dataclass
class Transform:
    """Canonical tensor ``C[r, c] = M[rows[r], cols[c]] * row_f[r] * col_f[c]`` (1-D tensors: rows and row_f only)."""
    rows: "np.ndarray | None" = None
    cols: "np.ndarray | None" = None
    row_f: "np.ndarray | None" = None
    col_f: "np.ndarray | None" = None

    def identity(self) -> bool:
        return self.rows is None and self.cols is None and self.row_f is None and self.col_f is None


@dataclass
class Plan:
    transforms: dict[str, Transform] = field(default_factory=dict)   # only tensors with a non-identity transform
    flags: list[str] = field(default_factory=list)                   # review reasons ("" = none)
    units: int = 0              # paired units examined (MLP channels, value channels, norm channels)
    pow2_units: int = 0         # of these, undone as exact +-2^k pairs
    examined: set[str] = field(default_factory=set)   # tensors of the recognised blocks (the ``delta_norm`` flag counts these)


# ------------------------------------------------------------------ reading


def _mm(e: T.TensorEntry) -> np.ndarray:
    dt = {"BF16": "<u2", "F32": "<f4", "F16": "<f2", "F64": "<f8"}[e.dtype]
    return np.memmap(e.path, dtype=dt, mode="r", offset=e.start, shape=e.shape)


def _f32(raw: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(raw).astype(np.float32)


def _chunks(rows: int, cols: int):
    step = max(1, ROW_CHUNK_ELEMS // max(cols, 1))
    for a in range(0, rows, step):
        yield a, min(rows, a + step)


def _vec(e: T.TensorEntry) -> np.ndarray:
    return _f32(_mm(e), e.dtype).astype(np.float64).reshape(-1)


def _row_stats(em: T.TensorEntry, eb: T.TensorEntry, rows: "np.ndarray | None" = None,
               col_f: "np.ndarray | None" = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per base row i: (||M'[i]||^2, ||B[i]||^2, M'[i] . B[i]) with M'[i] = M[rows[i]] * col_f."""
    mm, mb = _mm(em), _mm(eb)
    n, c = eb.shape
    nm, nb, dot = np.zeros(n), np.zeros(n), np.zeros(n)
    for a, b in _chunks(n, c):
        src = slice(a, b) if rows is None else np.sort(rows[a:b])
        xm = _f32(mm[src], em.dtype).astype(np.float64)
        if rows is not None:   # the sorted read, back in the order of the base rows
            xm = xm[np.searchsorted(np.sort(rows[a:b]), rows[a:b])]
        if col_f is not None:
            xm *= col_f
        xb = _f32(mb[a:b], eb.dtype).astype(np.float64)
        nm[a:b] = np.einsum("ij,ij->i", xm, xm)
        nb[a:b] = np.einsum("ij,ij->i", xb, xb)
        dot[a:b] = np.einsum("ij,ij->i", xm, xb)
    return nm, nb, dot


def _col_stats(em: T.TensorEntry, eb: T.TensorEntry, cols: "np.ndarray | None" = None,
               row_f: "np.ndarray | None" = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per base column j: (||M'[:, j]||^2, ||B[:, j]||^2, M'[:, j] . B[:, j]) with M' = (M * row_f)[:, cols]."""
    mm, mb = _mm(em), _mm(eb)
    n, c = eb.shape
    nm, nb, dot = np.zeros(c), np.zeros(c), np.zeros(c)
    for a, b in _chunks(n, c):
        xm = _f32(mm[a:b], em.dtype).astype(np.float64)
        if cols is not None:
            xm = xm[:, cols]
        if row_f is not None:
            xm *= row_f[a:b, None]
        xb = _f32(mb[a:b], eb.dtype).astype(np.float64)
        nm += np.einsum("ij,ij->j", xm, xm)
        nb += np.einsum("ij,ij->j", xb, xb)
        dot += np.einsum("ij,ij->j", xm, xb)
    return nm, nb, dot


# ------------------------------------------------------------------ pairing rules


def _log2_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """log2 sqrt(num / den) (squared norms in, ratio of norms out); NaN where either side is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * np.log2(num / den)
    out[~np.isfinite(out)] = np.nan
    return out


def _pair_factors(lr: np.ndarray, sr: np.ndarray, lc: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Factors that undo a row/column pair rescale: row side log2 ratio ``lr`` and dot sign ``sr`` (row scaled by s), column
    side ``lc`` / ``sc`` (column scaled by 1/s). Returns (s per unit, 1.0 = untouched; units undone as exact +-2^k;
    whether the anticorrelated (non-power-of-two) rule fired). The row is canonicalised by 1/s, the column by s."""
    s = np.ones(len(lr))
    ok = np.isfinite(lr) & np.isfinite(lc)
    kr = np.where(ok, np.rint(np.nan_to_num(lr)), 0.0)
    kc = np.where(ok, np.rint(np.nan_to_num(lc)), 0.0)
    pow2 = ok & (kr == -kc) & (sr == sc) & ((kr != 0) | (sr < 0))
    s[pow2] = sr[pow2] * np.exp2(kr[pow2])
    rr, rc = lr - np.where(pow2, kr, 0.0), lc + np.where(pow2, kr, 0.0)   # what is left after the exact part
    anti = False
    m = ok & (sr == sc)
    if m.sum() >= 8:
        mr = float(np.mean(rr[m] ** 2))
        ms = float(np.mean((rr[m] + rc[m]) ** 2))
        if mr > ANTI_RMS_MIN ** 2 and ms < ANTI_RATIO_MAX * mr:
            anti = True
            s[m] = sr[m] * np.exp2(lr[m])     # the measured row ratio (brief: s_i = ||up_i(entry)|| / ||up_i(base)||)
    return s, int(pow2.sum()), anti


def _sign(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, -1.0, 1.0)


# ------------------------------------------------------------------ the plan


_LAYER_RE = re.compile(r"^(.*\.layers\.\d+)\.")


def _config(model_dir: str) -> dict:
    try:
        with open(os.path.join(model_dir, "config.json")) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    text = cfg.get("text_config")
    return {**cfg, **text} if isinstance(text, dict) else cfg


def _head_dim(cfg: dict, base: dict[str, T.TensorEntry], layer: str) -> int | None:
    qn = base.get(f"{layer}.self_attn.q_norm.weight")
    if qn is not None and len(qn.shape) == 1:
        return int(qn.shape[0])
    hd = cfg.get("head_dim")
    if isinstance(hd, int) and hd > 0:
        return hd
    h, n = cfg.get("hidden_size"), cfg.get("num_attention_heads")
    if isinstance(h, int) and isinstance(n, int) and n > 0 and h % n == 0:
        return h // n
    return None


def _float2d(e: "T.TensorEntry | None") -> bool:
    return e is not None and e.dtype in T.FLOAT_DTYPES and len(e.shape) == 2 and e.numel > 0


def _float1d(e: "T.TensorEntry | None") -> bool:
    return e is not None and e.dtype in T.FLOAT_DTYPES and len(e.shape) == 1 and e.numel > 0


class _Builder:
    def __init__(self, model: dict[str, T.TensorEntry], base: dict[str, T.TensorEntry]):
        self.model, self.base = model, base
        self.plan = Plan()

    def pair(self, name: str) -> "tuple[T.TensorEntry, T.TensorEntry] | None":
        em, eb = self.model.get(name), self.base.get(name)
        if em is None or eb is None or em.shape != eb.shape or em.dtype not in T.FLOAT_DTYPES or eb.dtype not in T.FLOAT_DTYPES:
            return None
        self.plan.examined.add(name)
        return em, eb

    def tf(self, name: str) -> Transform:
        return self.plan.transforms.setdefault(name, Transform())

    def flag(self, why: str) -> None:
        if why not in self.plan.flags:
            self.plan.flags.append(why)

    # RMSNorm folding: norm weight j * t, the reading columns / t
    def norm_fold(self, norm: str, readers: list[str]) -> "np.ndarray | None":
        p = self.pair(norm)
        rs = [self.pair(r) for r in readers]
        if p is None or not _float1d(p[0]) or not rs or any(r is None or not _float2d(r[0]) for r in rs):
            return None
        h = p[0].shape[0]
        if any(r[0].shape[1] != h for r in rs):
            return None
        wm, wb = _vec(p[0]), _vec(p[1])
        rms = math.sqrt(float(np.mean(wb ** 2))) if h else 0.0
        big = np.abs(wb) > NORM_EPS_REL * rms
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(big, wm / np.where(big, wb, 1.0), np.nan)
        lt = np.full(h, np.nan)
        lt[big & (t != 0)] = np.log2(np.abs(t[big & (t != 0)]))
        st = _sign(np.nan_to_num(t, nan=1.0))
        # the reading side: the median over the readers of the column ratio (a row rescale of one reader, e.g. a value
        # scaling of v_proj, moves its columns too; q and k outvote it)
        lcs, votes = [], np.zeros(h)
        for em, eb in rs:
            nm, nb, dot = _col_stats(em, eb)
            lcs.append(_log2_ratio(nm, nb))
            # the sign of a column only where the column still resembles the base's (a permutation of the reader's rows,
            # e.g. the MLP units, leaves the column norms alone but scrambles the column dot): no opinion elsewhere
            with np.errstate(divide="ignore", invalid="ignore"):
                cos = np.nan_to_num(dot / np.sqrt(nm * nb))
            votes += np.where(np.abs(cos) >= PERM_COS, np.sign(cos), 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)    # an all-NaN column (zero norms): stays NaN, never undone
            lc = np.nanmedian(np.vstack(lcs), axis=0) if len(lcs) > 1 else lcs[0]
        sc = np.where(votes < 0, -1.0, np.where(votes > 0, 1.0, st))
        f, n_pow2, anti = _pair_factors(lt, st, lc, sc)
        self.plan.units += int(big.sum())
        self.plan.pow2_units += n_pow2
        if anti:
            self.flag(f"rescale_anticorrelated:{norm}")
        if np.all(f == 1.0):
            return None
        self.tf(norm).row_f = 1.0 / f
        for r in readers:
            self.tf(r).col_f = f.copy()
        return f

    def mlp(self, prefix: str, col_f: "np.ndarray | None") -> None:
        g, u, d = (self.pair(f"{prefix}.{k}_proj.weight") for k in ("gate", "up", "down"))
        if g is None or u is None or d is None or not all(_float2d(x[0]) for x in (g, u, d)):
            return
        n, h = g[0].shape
        if u[0].shape != (n, h) or d[0].shape != (h, n):
            return
        # permutation, anchored on the gate rows (never rescaled)
        nm, nb, dot = _row_stats(g[0], g[1], None, col_f)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = dot / np.sqrt(nm * nb)
        # a unit whose entry row or base row is all zero (a pruned or a dead unit) has no cosine and is no image of a permutation
        cand = np.flatnonzero((nm > 0) & (nb > 0) & ~(cos >= PERM_COS))
        perm = None
        if len(cand):
            perm = self._match(g, cand, col_f)
            if perm is None and len(cand) > PERM_SHARE_FLAG * n:
                self.flag(f"permutation_unmatched:{prefix}")
            elif perm is not None:
                self.flag(f"permuted:{prefix}")
                for k in ("gate", "up"):
                    self.tf(f"{prefix}.{k}_proj.weight").rows = perm
                self.tf(f"{prefix}.down_proj.weight").cols = perm
                for k in ("gate", "up"):
                    if self.pair(f"{prefix}.{k}_proj.bias") is not None:
                        self.tf(f"{prefix}.{k}_proj.bias").rows = perm
        # up rows against down columns
        nm, nb, dot = _row_stats(u[0], u[1], perm, col_f)
        lr, sr = _log2_ratio(nm, nb), _sign(dot)
        nm, nb, dot = _col_stats(d[0], d[1], perm)
        lc, sc = _log2_ratio(nm, nb), _sign(dot)
        s, n_pow2, anti = _pair_factors(lr, sr, lc, sc)
        self.plan.units += n
        self.plan.pow2_units += n_pow2
        if anti:
            self.flag(f"rescale_anticorrelated:{prefix}")
        if not np.all(s == 1.0):
            self.tf(f"{prefix}.up_proj.weight").row_f = 1.0 / s
            self.tf(f"{prefix}.down_proj.weight").col_f = s
            if self.pair(f"{prefix}.up_proj.bias") is not None:
                self.tf(f"{prefix}.up_proj.bias").row_f = 1.0 / s

    def _match(self, g: tuple[T.TensorEntry, T.TensorEntry], cand: np.ndarray, col_f: "np.ndarray | None") -> "np.ndarray | None":
        """Best base gate row for each unmatched entry row (random projection, then the exact cosine); a permutation of the
        unmatched units onto themselves, or None when the matching is not one-to-one or not convincing."""
        em, eb = g
        n, h = eb.shape
        rng = np.random.Generator(np.random.PCG64(20261023))
        proj = rng.standard_normal((h, PROJ_DIM)) / math.sqrt(PROJ_DIM)
        pb = np.zeros((n, PROJ_DIM))
        mm, mb = _mm(em), _mm(eb)
        for a, b in _chunks(n, h):
            pb[a:b] = _f32(mb[a:b], eb.dtype).astype(np.float64) @ proj
        pb /= np.maximum(np.linalg.norm(pb, axis=1, keepdims=True), 1e-30)
        best = np.zeros(len(cand), dtype=np.int64)
        for a, b in _chunks(len(cand), max(h, n)):     # bounds both the rows read and the similarity block
            xm = _f32(mm[cand[a:b]], em.dtype).astype(np.float64)
            if col_f is not None:
                xm *= col_f
            pm = xm @ proj
            pm /= np.maximum(np.linalg.norm(pm, axis=1, keepdims=True), 1e-30)
            best[a:b] = np.argmax(pm @ pb.T, axis=1)
        if len(np.unique(best)) != len(best) or not np.array_equal(np.sort(best), cand):
            return None
        perm = np.arange(n, dtype=np.int64)
        perm[best] = cand     # canonical row best[k] = entry row cand[k]
        nm, nb, dot = _row_stats(em, eb, perm, col_f)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = dot / np.sqrt(nm * nb)
        if not np.all(cos[best] >= PERM_COS):
            return None
        return perm

    def attention(self, layer: str, col_f: "np.ndarray | None", hd: int | None) -> None:
        v, o = self.pair(f"{layer}.self_attn.v_proj.weight"), self.pair(f"{layer}.self_attn.o_proj.weight")
        if v is None or o is None or not _float2d(v[0]) or not _float2d(o[0]):
            return
        nv, h = v[0].shape
        ho, no = o[0].shape
        if ho != h or no % nv:
            return
        g = no // nv
        nm, nb, dot = _row_stats(v[0], v[1], None, col_f)
        lr, sr = _log2_ratio(nm, nb), _sign(dot)
        cm, cb, cd = _col_stats(o[0], o[1])
        # grouped query heads: o_proj column j reads value channel ((j // hd) // g) * hd + j % hd. Without a head size in
        # the config, the usual sizes are tried and the one that pairs the most channels is taken (none pairs: untouched)
        sizes = [nv] if g == 1 else ([hd] if hd else [d for d in (64, 128, 256, 96, 80, 192, 32, 16) if nv % d == 0])
        best = None
        for d in sizes:
            if not isinstance(d, int) or d <= 0 or d > nv:
                continue    # a head size the base's config (or q_norm) gives that cannot index the value channels: no pairing
            j = np.arange(no)
            chan = ((j // d) // g) * d + j % d
            if int(chan.max()) >= nv:
                continue
            lc = _log2_ratio(np.bincount(chan, cm, nv), np.bincount(chan, cb, nv))
            sc = _sign(np.bincount(chan, cd, nv))
            s, n_pow2, anti = _pair_factors(lr, sr, lc, sc)
            if best is None or n_pow2 > best[2]:
                best = (chan, s, n_pow2, anti)
        if best is None:
            return
        chan, s, n_pow2, anti = best
        self.plan.units += nv
        self.plan.pow2_units += n_pow2
        if anti:
            self.flag(f"rescale_anticorrelated:{layer}.self_attn")
        if not np.all(s == 1.0):
            self.tf(f"{layer}.self_attn.v_proj.weight").row_f = 1.0 / s
            self.tf(f"{layer}.self_attn.o_proj.weight").col_f = s[chan]
            if self.pair(f"{layer}.self_attn.v_proj.bias") is not None:
                self.tf(f"{layer}.self_attn.v_proj.bias").row_f = 1.0 / s


def plan(model: dict[str, T.TensorEntry], base: dict[str, T.TensorEntry], model_dir: str = "", base_dir: str = "") -> Plan:
    """The canonicalisation of ``model`` (canonical names, ``tensors._canonical``) onto ``base``. Reads the paired tensors
    in full, a chunk of rows at a time; nothing else."""
    b = _Builder(model, base)
    cfg = _config(base_dir) if base_dir else {}
    gemma = str(cfg.get("model_type", "")).startswith("gemma")
    layers = sorted({m.group(1) for n in base if (m := _LAYER_RE.match(n))}, key=lambda s: (len(s), s))
    for layer in layers:
        a = f"{layer}.self_attn"
        in_f = post_f = None
        if not gemma:   # (1 + w) norms, and post-norms around the blocks: not refolded
            in_f = b.norm_fold(f"{layer}.input_layernorm.weight", [f"{a}.{k}_proj.weight" for k in ("q", "k", "v")])
            if f"{layer}.pre_feedforward_layernorm.weight" not in base:
                post_f = b.norm_fold(f"{layer}.post_attention_layernorm.weight", [f"{layer}.mlp.gate_proj.weight"])
                if post_f is not None:   # the up rows read the same normed stream
                    b.tf(f"{layer}.mlp.up_proj.weight").col_f = post_f.copy()
        b.attention(layer, in_f, _head_dim(cfg, base, layer))
        b.mlp(f"{layer}.mlp", post_f)
        if f"{layer}.mlp.shared_expert.gate_proj.weight" in base:
            b.mlp(f"{layer}.mlp.shared_expert", post_f)
    if not gemma and "lm_head.weight" in model and "lm_head.weight" in base:
        b.norm_fold("model.norm.weight", ["lm_head.weight"])
    p = b.plan
    p.transforms = {k: v for k, v in p.transforms.items() if not v.identity()}
    if p.units and p.pow2_units > POW2_SHARE_FLAG * p.units:
        p.flags.insert(0, f"pow2_rescale:{p.pow2_units}/{p.units}")
    return p


def canonical_values(e: T.TensorEntry, eb: T.TensorEntry, tf: Transform, idx: np.ndarray) -> np.ndarray:
    """The canonical entry tensor at the flat (base-layout) positions ``idx``, as float32."""
    if len(eb.shape) == 1:
        r, c = idx, None
    else:
        r, c = idx // eb.shape[1], idx % eb.shape[1]
    sr = tf.rows[r] if tf.rows is not None else r
    if c is None:
        src = sr
    else:
        sc = tf.cols[c] if tf.cols is not None else c
        src = sr * eb.shape[1] + sc
    order = np.argsort(src, kind="stable")
    raw = T._sample_float(e, T.STRIDE, src[order])
    vals = np.empty_like(raw)
    vals[order] = raw
    x = vals.astype(np.float64)
    if tf.row_f is not None:
        x *= tf.row_f[r]
    if tf.col_f is not None and c is not None:
        x *= tf.col_f[c]
    return x.astype(np.float32)
