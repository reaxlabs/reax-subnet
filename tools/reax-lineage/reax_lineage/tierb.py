"""Tier B (LINEAGE-PROCEDURE.md section 5): undo residual-stream rotation, MLP-neuron and attention-head permutation,
per-input-column scale (norm folding) and per-neuron up/down scale, using 4 full layers and 16,384 shared-token embedding
rows of the entry and of its declared start. Then the frame check is redone in the canonical coordinates.

Row-vector convention: embedding rows E_c = E_s Q.  In-hidden weights W (out x d) become W Q^T, out-hidden weights Q W.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .fp import _h, find_head_tensor, split_layer_name, to_f32
from .hf import DT_BYTES, FLOAT_OK
from .measure import Thresholds

OUT_RE = re.compile(r"(o_proj|out_proj|down_proj|\.wo|\.w2|dense)\.weight$")
ATTN_Q, ATTN_K, ATTN_V, ATTN_O = "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"


@dataclass
class TierBData:
    layers: Dict[int, Dict[str, np.ndarray]]
    embed: np.ndarray
    n_layers: int = 0
    meta: dict = field(default_factory=dict)


def pick_layers(n_layers: int, seed: str, k: int = 4) -> List[int]:
    q = max(1, n_layers // k)
    return sorted({min(n_layers - 1, i * q + _h(seed, "tierb", i) % q) for i in range(k)})


def embed_ids(seed: str, vocab: int, rows=16384, block=1024, pool=100_000):
    p = min(vocab, pool)
    n = rows // block
    return sorted(_h(seed, "tierb-embed", k) % (p - block + 1) for k in range(n)), block


def fetch_tierb(source, seed: str, layers: Optional[List[int]] = None, embed_rows=16384, workers=8) -> TierBData:
    from concurrent.futures import ThreadPoolExecutor
    idx = source.tensor_index()
    lay_names = {}
    nl = 0
    for n, r in idx.items():
        sp = split_layer_name(n)
        if sp and not any(w in n for w in ("visual", "vision", "audio", "mtp.")):
            nl = max(nl, sp[0] + 1)
            if len(r.shape) == 2 and r.dtype in FLOAT_OK:
                lay_names.setdefault(sp[0], {})[sp[1]] = n
    layers = layers or pick_layers(nl, seed)
    jobs = [(l, suf, n) for l in layers for suf, n in lay_names.get(l, {}).items()]

    def read_full(n):
        r = idx[n]
        b = source.read_range(r.file, r.offset, r.offset + r.nbytes - 1)
        return to_f32(b, r.dtype).reshape(r.shape)

    out: Dict[int, Dict[str, np.ndarray]] = {l: {} for l in layers}
    with ThreadPoolExecutor(workers) as ex:
        for (l, suf, _), arr in zip(jobs, ex.map(lambda j: read_full(j[2]), jobs)):
            out[l][suf] = arr
        en = find_head_tensor(idx, "embed")
        er = idx[en]
        starts, block = embed_ids(seed, er.shape[0], embed_rows)

        def rd(s):
            cols, sz = er.shape[1], DT_BYTES[er.dtype]
            a = er.offset + s * cols * sz
            return to_f32(source.read_range(er.file, a, a + block * cols * sz - 1), er.dtype).reshape(block, cols)
        embed = np.concatenate(list(ex.map(rd, starts)), axis=0)
    return TierBData(out, embed, nl, {"layers": layers, "embed_starts": starts})


# ------------------------------------------------------------ transforms ----

def procrustes(Es: np.ndarray, Ec: np.ndarray) -> np.ndarray:
    """Orthogonal Q minimising ||Es Q - Ec||_F."""
    M = (Es.astype(np.float64).T @ Ec.astype(np.float64))
    U, _, Vt = np.linalg.svd(M)
    return (U @ Vt).astype(np.float32)


def _kind(suffix: str, shape, d: int) -> Optional[str]:
    if len(shape) != 2:
        return None
    r, c = shape
    if OUT_RE.search(suffix) and r == d:
        return "out"
    if c == d and not OUT_RE.search(suffix):
        return "in"
    if r == d and c != d:
        return "out"
    return None


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def match_rows(A: np.ndarray, B: np.ndarray, block=2048):
    """For each row of A the index of the most cosine-similar row of B, and that cosine."""
    A, B = _norm_rows(A.astype(np.float32)), _norm_rows(B.astype(np.float32))
    j = np.empty(A.shape[0], dtype=np.int64)
    c = np.empty(A.shape[0], dtype=np.float32)
    for s in range(0, A.shape[0], block):
        sim = A[s:s + block] @ B.T
        j[s:s + block] = sim.argmax(1)
        c[s:s + block] = sim.max(1)
    return j, c


def _perm_from_match(j: np.ndarray):
    """j[i] = start index of entry row i. Returns (inverse-permutation index array `src` with new[j]=old[src[j]], one_to_one share)."""
    n = len(j)
    src = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    for i, t in enumerate(j):
        if src[t] < 0:
            src[t] = i
            used[i] = True
    free = [i for i in range(n) if not used[i]]
    for t in range(n):
        if src[t] < 0:
            src[t] = free.pop()
    return src, float(len(np.unique(j)) / n)


def row_cos_stats(X: np.ndarray, S: np.ndarray, thr: Thresholds):
    num = np.einsum("ij,ij->i", X, S)
    den = np.sqrt(np.einsum("ij,ij->i", X, X) * np.einsum("ij,ij->i", S, S))
    cos = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    ss = np.einsum("ij,ij->i", S, S)
    a = np.where(ss > 0, num / np.where(ss > 0, ss, 1.0), 1.0)
    return cos, np.abs(np.log2(np.maximum(np.abs(a), 1e-12)))


def frame_of(layers_c, layers_s, thr: Thresholds) -> dict:
    cs, ls = [], []
    for l, d in layers_c.items():
        for suf, x in d.items():
            s = layers_s.get(l, {}).get(suf)
            if s is None or s.shape != x.shape:
                continue
            c, a = row_cos_stats(x, s, thr)
            cs.append(c)
            ls.append(a)
    if not cs:
        return {"rows": 0, "frame_share": 0.0, "scale_share": 1.0, "ok": False}
    c, a = np.concatenate(cs), np.concatenate(ls)
    fs, ss = float((c >= thr.tierb_cos).mean()), float((a > thr.scale_tol_log2).mean())
    return {"rows": int(c.size), "frame_share": round(fs, 6), "scale_share": round(ss, 6),
            "ok": bool(fs >= thr.frame_share and ss <= thr.scale_share_max)}


def canonicalise(entry: TierBData, start: TierBData, thr: Thresholds, head_dim: Optional[int] = None, consume: bool = False) -> dict:
    """Transform the entry's layers into the start's coordinates. Returns the report; the transformed layers are in
    report['layers'] (not JSON-serialisable; stripped by the caller).
    consume=True (v1.5.0, memory): `entry.layers` is emptied layer by layer while the transformed copy is built, so a tensor that the later steps replace
    (column scale, permutation) is freed instead of living on in `entry.layers` next to its replacement. Without it the transformed layers are a second
    copy of the entry's four layers (9.9 GB at 31B, CALIBRATION.md 18.4). The report is identical; the caller must not use `entry.layers` afterwards."""
    rep = {"rotation": {}, "col_scale": {}, "mlp_perm": {}, "heads_perm": {}, "notes": []}
    d = entry.embed.shape[1]
    if start.embed.shape != entry.embed.shape:
        rep["available"] = False
        rep["notes"].append("embedding rows of entry and start do not have the same shape")
        return rep
    rep["available"] = True
    before = frame_of(entry.layers, start.layers, thr)
    rep["frame_before"] = before
    # 1. residual rotation
    Es, Ec = start.embed, entry.embed
    res0 = float(np.linalg.norm(Es - Ec) / (np.linalg.norm(Ec) + 1e-30))
    Q = procrustes(Es, Ec)
    res1 = float(np.linalg.norm(Es @ Q - Ec) / (np.linalg.norm(Ec) + 1e-30))
    dev = float(np.linalg.norm(Q - np.eye(d, dtype=np.float32)) / np.sqrt(d))
    rotated = bool(dev > 0.1 and res1 < 0.7 * res0)
    rep["rotation"] = {"residual_before": round(res0, 5), "residual_after": round(res1, 5), "q_minus_identity": round(dev, 5),
                       "found": rotated}
    L: Dict[int, Dict[str, np.ndarray]] = {}
    for l in list(entry.layers):
        tens = entry.layers.pop(l) if consume else entry.layers[l]
        L[l] = {}
        for suf, w in tens.items():
            k = _kind(suf, w.shape, d)
            if rotated and k == "in":
                w = w @ Q.T
            elif rotated and k == "out":
                w = Q @ w
            L[l][suf] = w.astype(np.float32, copy=False)
    # 2-4. iterate twice: permutations (matched on column-standardised rows, so a norm fold does not hide them),
    #      then per-input-column scale (norm folding) and per-neuron up/down scale on the aligned tensors
    mlp = {"layers": 0, "identity_share": [], "one_to_one": [], "matched_cos": []}
    hp = []
    cum_cols = np.zeros((), dtype=np.int64)
    tot_cols = 0
    colscale_share = 0.0

    def colstd(w):
        r = np.sqrt((w.astype(np.float64) ** 2).mean(0, keepdims=True))
        return (w / np.where(r > 0, r, 1.0)).astype(np.float32)

    for it in range(2):
        for l, tens in L.items():
            if l not in start.layers:
                continue
            S_l = start.layers[l]
            # MLP neurons
            if "mlp.gate_proj.weight" in tens and "mlp.up_proj.weight" in tens and "mlp.gate_proj.weight" in S_l:
                g, u, dn = tens["mlp.gate_proj.weight"], tens["mlp.up_proj.weight"], tens.get("mlp.down_proj.weight")
                sg, su = S_l["mlp.gate_proj.weight"], S_l["mlp.up_proj.weight"]
                if sg.shape == g.shape:
                    j, cs = match_rows(np.concatenate([colstd(g), colstd(u)], 1), np.concatenate([colstd(sg), colstd(su)], 1))
                    src, o2o = _perm_from_match(j)
                    if it == 0:
                        mlp["layers"] += 1
                        mlp["identity_share"].append(float((j == np.arange(len(j))).mean()))
                        mlp["one_to_one"].append(o2o)
                        mlp["matched_cos"].append(float(cs.mean()))
                    if o2o >= 0.99 and cs.mean() >= 0.5 and not (j == np.arange(len(j))).all():
                        g, u = g[src], u[src]
                        if dn is not None and dn.shape[1] == len(src):
                            dn = dn[:, src]
                        tens["mlp.gate_proj.weight"], tens["mlp.up_proj.weight"] = g, u
                        if dn is not None:
                            tens["mlp.down_proj.weight"] = dn
            # attention heads (needs head_dim)
            if head_dim and ATTN_Q in tens and ATTN_O in tens and ATTN_Q in S_l and S_l[ATTN_Q].shape == tens[ATTN_Q].shape \
                    and tens[ATTN_Q].shape[0] % head_dim == 0:
                q, o = tens[ATTN_Q], tens[ATTN_O]
                nh = q.shape[0] // head_dim
                j, cs = match_rows(colstd(q).reshape(nh, -1), colstd(S_l[ATTN_Q]).reshape(nh, -1))
                src, o2o = _perm_from_match(j)
                if it == 0:
                    hp.append(float((j == np.arange(nh)).mean()))
                if o2o >= 0.99 and cs.mean() >= 0.5 and not (j == np.arange(nh)).all():
                    tens[ATTN_Q] = q.reshape(nh, head_dim, -1)[src].reshape(q.shape)
                    if o.shape[1] == q.shape[0]:
                        tens[ATTN_O] = o.reshape(o.shape[0], nh, head_dim)[:, src, :].reshape(o.shape)
                    for nm in (ATTN_K, ATTN_V):
                        w, sw = tens.get(nm), S_l.get(nm)
                        if w is not None and sw is not None and w.shape == sw.shape and w.shape[0] % head_dim == 0:
                            nk = w.shape[0] // head_dim
                            jk, _ = match_rows(colstd(w).reshape(nk, -1), colstd(sw).reshape(nk, -1))
                            sk, ok = _perm_from_match(jk)
                            if ok >= 0.99:
                                tens[nm] = w.reshape(nk, head_dim, -1)[sk].reshape(w.shape)
            # per-input-column scale shared by the tensors that read the same norm
            groups: Dict[str, List[str]] = {}
            for suf, w in tens.items():
                if _kind(suf, w.shape, d) == "in" and suf in S_l and S_l[suf].shape == w.shape:
                    groups.setdefault("attn" if ("attn" in suf) else "mlp", []).append(suf)
            for gname, sufs in groups.items():
                num = np.zeros(d, np.float64)
                den = np.zeros(d, np.float64)
                for sname in sufs:
                    num += np.einsum("ij,ij->j", tens[sname], S_l[sname]).astype(np.float64)
                    den += np.einsum("ij,ij->j", S_l[sname], S_l[sname]).astype(np.float64)
                gj = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
                gj = np.clip(np.where(np.abs(gj) < 1e-3, 1.0, gj), -1e3, 1e3).astype(np.float32)
                if it == 0:
                    tot_cols += d
                    colscale_share += float((np.abs(np.log2(np.maximum(np.abs(gj), 1e-12))) > thr.scale_tol_log2).sum())
                for sname in sufs:
                    tens[sname] = tens[sname] / gj[None, :]
            # per-neuron scale between up and down (last iteration, on aligned and column-corrected tensors)
            if it == 1 and "mlp.up_proj.weight" in tens and "mlp.up_proj.weight" in S_l and tens["mlp.up_proj.weight"].shape == S_l["mlp.up_proj.weight"].shape:
                u, su, dn = tens["mlp.up_proj.weight"], S_l["mlp.up_proj.weight"], tens.get("mlp.down_proj.weight")
                ssq = np.einsum("ij,ij->i", su, su)
                sc = np.where(ssq > 0, np.einsum("ij,ij->i", u, su) / np.where(ssq > 0, ssq, 1.0), 1.0)
                sc = np.where((np.abs(sc) > 1e-3) & (np.abs(sc) < 1e3), sc, 1.0).astype(np.float32)
                tens["mlp.up_proj.weight"] = u / sc[:, None]
                if dn is not None and dn.shape[1] == len(sc):
                    tens["mlp.down_proj.weight"] = dn * sc[None, :]
    rep["col_scale"] = {"columns_rescaled_share": round(colscale_share / tot_cols, 5) if tot_cols else 0.0}
    rep["mlp_perm"] = {"layers": mlp["layers"],
                       "identity_share_mean": round(float(np.mean(mlp["identity_share"])), 5) if mlp["layers"] else None,
                       "one_to_one_min": round(float(np.min(mlp["one_to_one"])), 5) if mlp["layers"] else None,
                       "matched_cos_mean": round(float(np.mean(mlp["matched_cos"])), 5) if mlp["layers"] else None}
    rep["heads_perm"] = {"layers": len(hp), "identity_share_mean": round(float(np.mean(hp)), 5) if hp else None}
    after = frame_of(L, start.layers, thr)
    rep["frame_after"] = after
    transform = bool(rotated or rep["col_scale"]["columns_rescaled_share"] > 0.01 or
                     (mlp["layers"] and np.mean(mlp["identity_share"]) < 0.95) or (hp and np.mean(hp) < 0.95))
    rep["transform_found"] = bool(transform and after["ok"])
    rep["start_confirmed"] = bool(after["ok"])
    rep["layers"] = L
    return rep
