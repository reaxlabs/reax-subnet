"""Fingerprint = the weight sample of one model (LINEAGE-PROCEDURE.md section 4).

Body: every 2-D floating tensor with >= min_elems elements inside the repeated language-model layer blocks,
keyed `layers.<l>.<suffix>` (so `model.language_model.layers.N.x` and `language_model.layers.N.x` match);
`rows` consecutive rows at a position fixed by sha256(seed | suffix | layer). MoE: router of every layer in full,
experts only at seeded indices. Head group: 1,024 token rows of the output head / embeddings (32 blocks of 32).
The vision tower, norms, biases are not sampled.
"""
import hashlib
import json
import os
import re
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .hf import DT_BYTES, FLOAT_OK, HFClient, HFSource, LocalSource, file_sha256_map

SEED_A_DEFAULT = "A0"
FP_SCHEMA = "reax-fp/1"
LAYER_RE = re.compile(r"^(?P<pre>.*?language_model\.layers|.*?model\.layers|layers)\.(?P<l>\d+)\.(?P<post>.+)$")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
ROUTER_RE = re.compile(r"(?:^|\.)(?:mlp|moe|block_sparse_moe)\.(?:gate|router)\.weight$|\.router\.weight$")
SKIP_WORDS = ("visual", "vision", "audio", "image", "mtp.", "multi_modal", "projector")
HEAD_RES = {"lm_head": re.compile(r"(?:^|\.)lm_head\.weight$"),
            "embed": re.compile(r"(?:^|\.)(?:embed_tokens|wte)\.weight$")}


def find_head_tensor(index, grp):
    """Name of the output head ('lm_head') or the token embedding ('embed') tensor, or None (tied heads have no lm_head)."""
    hits = [n for n, r in sorted(index.items(), key=lambda kv: (len(kv[0]), kv[0]))
            if HEAD_RES[grp].search(n) and not any(w in n for w in SKIP_WORDS) and len(r.shape) == 2]
    return hits[0] if hits else None


@dataclass
class SampleSpec:
    rows: int = 4
    min_elems: int = 1_000_000
    head_rows: int = 1024
    head_block: int = 32
    head_pool: int = 100_000
    experts_per_layer: int = 4
    workers: int = 8


def seed_sha256(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def split_layer_name(name):
    m = LAYER_RE.match(name)
    return (int(m.group("l")), m.group("post")) if m else None


def _h(seed, *parts) -> int:
    return int(hashlib.sha256(("|".join([seed] + [str(p) for p in parts])).encode()).hexdigest(), 16)


def plan_body(index: Dict, seed: str, spec: SampleSpec, layer_map: Optional[Dict] = None):
    """-> (jobs, warnings, non_float). jobs: sorted list of (key, tensor name, row0, nrows)."""
    jobs, warns, nonfloat = [], [], []
    n_experts = {}
    for name in index:
        sp = split_layer_name(name)
        m = EXPERT_RE.search(name)
        if sp and m:
            n_experts[sp[0]] = max(n_experts.get(sp[0], 0), int(m.group(1)) + 1)
    chosen = {}
    for l, n in n_experts.items():
        hl = (layer_map or {}).get(l, l)
        chosen[l] = {_h(seed, "experts", hl, k) % n for k in range(spec.experts_per_layer)}
    fused_skipped = 0
    for name, ref in sorted(index.items()):
        sp = split_layer_name(name)
        if not sp or any(w in name for w in SKIP_WORDS):
            continue
        l, post = sp
        if len(ref.shape) == 3 and ref.shape[0] * ref.shape[1] * ref.shape[2] >= spec.min_elems:
            fused_skipped += 1
            continue
        if len(ref.shape) != 2:
            continue
        r, c = ref.shape
        if ref.dtype not in FLOAT_OK:
            if r * c >= spec.min_elems:
                nonfloat.append((name, ref.dtype))
            continue
        hl = (layer_map or {}).get(l, l)
        if ROUTER_RE.search(name) and r <= 4096:
            jobs.append(("layers.%d.%s" % (l, post), name, 0, r))
            continue
        if r * c < spec.min_elems:
            continue
        m = EXPERT_RE.search(name)
        if m and int(m.group(1)) not in chosen.get(l, ()):
            continue
        row0 = _h(seed, post, hl) % (r - spec.rows + 1)
        jobs.append(("layers.%d.%s" % (l, post), name, row0, spec.rows))
    if fused_skipped:
        warns.append("fused_3d_expert_tensors_skipped:%d" % fused_skipped)
    return sorted(jobs), warns, nonfloat


def plan_head(index: Dict, seed: str, spec: SampleSpec):
    """-> {group: (tensor name, [row0 of each block])} for the output head and the embeddings."""
    out = {}
    nblocks = spec.head_rows // spec.head_block
    for grp in HEAD_RES:
        n = find_head_tensor(index, grp)
        if n is not None and index[n].dtype in FLOAT_OK:
            pool = min(index[n].shape[0], spec.head_pool)
            starts = sorted(_h(seed, "head", grp, k) % (pool - spec.head_block + 1) for k in range(nblocks))
            out[grp] = (n, starts)
    return out


def to_f32(b: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(b, "<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(b, "<f2").astype(np.float32)
    return np.frombuffer(b, "<f4").astype(np.float32)


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits with round-to-nearest-even (used to store and to build synthetic checkpoints)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    r = u + 0x7FFF + ((u >> 16) & 1)
    return (r >> 16).astype(np.uint16)


def bf16_bits_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


@dataclass
class Fingerprint:
    meta: dict
    body: Dict[str, np.ndarray] = field(default_factory=dict)
    head: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def name(self):
        return "%s@%s" % (self.meta.get("repo"), self.meta.get("rev"))

    @property
    def n_layers(self):
        return 1 + max((split_layer_name(k)[0] for k in self.body), default=-1)

    def row_start(self, key):
        return tuple(self.meta["row_starts"].get(key, (None, None)))

    def save(self, path):
        arrs = {}
        bf = set(self.meta.get("bf16_keys", []))
        for k, v in self.body.items():
            arrs["B:" + k] = f32_to_bf16_bits(v) if k in bf else v.astype(np.float32)
        for k, v in self.head.items():
            arrs["H:" + k] = f32_to_bf16_bits(v) if ("H:" + k) in bf else v.astype(np.float32)
        tmp = path + ".tmp.npz"
        np.savez(tmp, __meta__=np.array(json.dumps(self.meta)), **arrs)
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        fp = Fingerprint(meta)
        for k in z.files:
            if k == "__meta__":
                continue
            a = z[k]
            a = bf16_bits_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32, copy=False)
            (fp.body if k.startswith("B:") else fp.head)[k[2:]] = a
        return fp


# ------------------------------------------------ declared layer changes ----
# RULES-v2 2.4 / 2.5: an entry may physically duplicate, drop or reorder layers of its start if it declares `architecture_changes.layer_map`
# (entry layer -> start layer). The sampled rows of entry layer i are chosen by the hash of the layer it descends from, so they sit at the
# same positions as the start's rows of that layer; the comparison models have to be read in the entry's layer numbering too, or a
# duplicated block shifts every later layer against its start (coverage < 0.5 -> "start not confirmed", or accidental row coincidences).

def root_map(fp: "Fingerprint") -> Dict[int, int]:
    """Layer j of this fingerprint -> the family-root layer whose rows position it samples (empty = identity)."""
    return {int(k): int(v) for k, v in (fp.meta.get("layer_map") or {}).items()}


def effective_map(declared: Optional[Dict[int, int]], start_fp: Optional["Fingerprint"]) -> Optional[Dict[int, int]]:
    """Entry layer -> root layer: the declared entry->start map composed with the start's own map (a start that is itself an entry with a
    layer_map). None when it is the identity (the normal case: nothing changes anywhere)."""
    rs = root_map(start_fp) if start_fp is not None else {}
    if declared:
        eff = {int(i): rs.get(int(j), int(j)) for i, j in declared.items()}
    else:
        eff = dict(rs)
    return eff if any(i != j for i, j in eff.items()) else None


def layer_view(fp: "Fingerprint", rho: Dict[int, int], n_layers: int, direct: Optional[Dict[int, int]] = None) -> "Fingerprint":
    """`fp` (a comparison model: root, start, candidate, a candidate's start) in the numbering of an entry whose rows are positioned by
    `rho` (entry layer -> root layer). Entry layer i takes the layer of `fp` that samples the same root layer (the first one if `fp`
    has copies of it); for the entry's own start pass `direct` (entry layer -> start layer, as declared), which picks the exact copy.
    A model with the same map as the entry is returned unchanged."""
    own = root_map(fp)
    if direct is None and all(own.get(i, i) == rho.get(i, i) for i in range(max(n_layers, fp.n_layers))):
        return fp
    by_layer: Dict[int, Dict[str, str]] = {}
    for key in fp.body:
        l, post = split_layer_name(key)
        by_layer.setdefault(l, {})[post] = key
    first: Dict[int, int] = {}
    for j in sorted(by_layer):
        first.setdefault(own.get(j, j), j)
    body, rows, shapes = {}, {}, {}
    for i in range(n_layers):
        j = direct.get(i, i) if direct is not None else first.get(rho.get(i, i))
        for post, key in by_layer.get(j, {}).items():
            nk = "layers.%d.%s" % (i, post)
            body[nk] = fp.body[key]
            rows[nk] = fp.meta["row_starts"][key]
            if key in (fp.meta.get("shapes") or {}):
                shapes[nk] = fp.meta["shapes"][key]
    meta = dict(fp.meta)
    meta.update(row_starts=rows, shapes=shapes, n_layers=n_layers, layer_view_of=fp.name,
                layer_map={str(i): int(rho.get(i, i)) for i in range(n_layers)})
    return Fingerprint(meta, body, fp.head)


def cache_path(cache_dir, repo, rev, seed_id, seed_hash):
    safe = (repo + "@" + rev).replace("/", "__")
    return os.path.join(cache_dir, "%s.%s-%s.npz" % (safe, seed_id, seed_hash[:12]))


def _model_dims(source):
    """hidden size / vocab from tensor shapes (no config needed)."""
    idx = source.tensor_index()
    vocab = hidden = None
    for grp in ("embed", "lm_head"):
        n = find_head_tensor(idx, grp)
        if n:
            return tuple(idx[n].shape)
    return vocab, hidden


def build_fingerprint(source, seed: str = SEED_A_DEFAULT, seed_id: str = "A", spec: SampleSpec = None,
                      layer_map: Optional[Dict] = None, publish_seed: bool = None, progress=None) -> Fingerprint:
    """Fetch the sample of `source` (HFSource or LocalSource). The sealed seed (B) is never written out:
    only its sha256 (publish_seed defaults to True for seed_id 'A')."""
    spec = spec or SampleSpec()
    t0 = time.time()
    idx = source.tensor_index()
    jobs, warns, nonfloat = plan_body(idx, seed, spec, layer_map)
    hplan = plan_head(idx, seed, spec)
    bf16_keys = []

    def read_rows(name, row0, nrows):
        ref = idx[name]
        cols, sz = ref.shape[1], DT_BYTES[ref.dtype]
        a = ref.offset + row0 * cols * sz
        b = source.read_range(ref.file, a, a + nrows * cols * sz - 1)
        if len(b) != nrows * cols * sz:
            raise RuntimeError("short read for %s" % name)
        return to_f32(b, ref.dtype).reshape(nrows, cols)

    def one_body(j):
        key, name, row0, nrows = j
        return key, read_rows(name, row0, nrows)

    fp = Fingerprint(meta={})
    with ThreadPoolExecutor(spec.workers) as ex:
        for key, arr in ex.map(one_body, jobs):
            fp.body[key] = arr
        head_jobs = [(grp, name, s) for grp, (name, starts) in hplan.items() for s in starts]
        res = list(ex.map(lambda j: (j[0], read_rows(j[1], j[2], spec.head_block)), head_jobs))
    for grp in hplan:
        fp.head[grp] = np.concatenate([a for g, a in res if g == grp], axis=0)
    for key, name, _, _ in jobs:
        if idx[name].dtype == "BF16":
            bf16_keys.append(key)
    bf16_keys += ["H:" + g for g in hplan if idx[hplan[g][0]].dtype == "BF16"]
    vocab, hidden = _model_dims(source)
    try:
        files_sha = file_sha256_map(source)
    except Exception as e:      # tree API failure must not kill the fingerprint
        files_sha, warns = {}, warns + ["file_sha256_failed:%s" % type(e).__name__]
    shapes = {k: list(idx[n].shape) for k, n, _, _ in jobs}
    pub = (seed_id == "A") if publish_seed is None else publish_seed
    fp.meta = {
        "schema": FP_SCHEMA, "repo": source.repo, "rev": source.rev, "source": source.kind,
        "seed_id": seed_id, "seed_sha256": seed_sha256(seed), "seed": seed if pub else None,
        "rows": spec.rows, "min_elems": spec.min_elems, "head_rows": spec.head_rows,
        "n_tensors": len(fp.body), "values": int(sum(a.size for a in fp.body.values())),
        "head_values": int(sum(a.size for a in fp.head.values())),
        "row_starts": {k: [r0, nr] for k, _, r0, nr in jobs}, "shapes": shapes,
        "head_blocks": {g: [n, s] for g, (n, s) in hplan.items()},
        "bf16_keys": bf16_keys, "non_float_body": nonfloat, "vocab": vocab, "hidden": hidden,
        "n_layers": fp.n_layers, "layer_map": {str(k): v for k, v in (layer_map or {}).items()} or None,
        "files_sha256": files_sha, "warnings": warns,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return fp


def get_fingerprint(source, seed=SEED_A_DEFAULT, seed_id="A", cache_dir=None, spec=None, layer_map=None,
                    refresh=False, publish_seed=None) -> Fingerprint:
    """Cached build: key = repo@rev + seed. Local checkpoints are keyed by path + header signature + shard sizes and mtimes."""
    sh = seed_sha256(seed)
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        rev = source.rev
        if source.kind == "local":
            # header signature + size and mtime of every shard + the absolute path: two training checkpoints of one run have the same
            # headers, so the header alone would return the fingerprint of the wrong checkpoint (stale-cache bug found in the finish pass)
            idx = source.tensor_index()
            files = {r.file for r in idx.values()}
            st = {f: [os.path.getsize(os.path.join(source.path, f)), os.stat(os.path.join(source.path, f)).st_mtime_ns] for f in sorted(files)}
            sig = hashlib.sha256(json.dumps({"tensors": {k: [r.file, r.offset, r.dtype, r.shape] for k, r in sorted(idx.items())},
                                             "files": st, "path": source.path}).encode()).hexdigest()[:16]
            rev = "local-" + sig
        if layer_map:
            sh = hashlib.sha256((sh + json.dumps(sorted((layer_map or {}).items()))).encode()).hexdigest()
        path = cache_path(cache_dir, source.repo, rev, seed_id, sh)
        if os.path.exists(path) and not refresh:
            return Fingerprint.load(path)
    fp = build_fingerprint(source, seed, seed_id, spec, layer_map, publish_seed)
    if path:
        fp.save(path)
    return fp
