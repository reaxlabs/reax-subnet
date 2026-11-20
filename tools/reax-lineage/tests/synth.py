"""Synthetic models for the tests: small transformers as dicts of float32 tensors, an in-memory safetensors source,
and helpers to write offline hub repos (config, tokenizer, README, reax-lineage.json) and comparison databases."""
import json
import os
import re
import shutil
import struct

import numpy as np

from reax_lineage.db import ComparisonDB
from reax_lineage.fp import SampleSpec, build_fingerprint, f32_to_bf16_bits, bf16_bits_to_f32
from reax_lineage.hf import FileInfo, TensorRef

SEED_B = "sealed-test-seed"
SPEC = SampleSpec(rows=16, min_elems=20_000, head_rows=256, head_block=16, head_pool=1000, workers=4)
D, FF, L, HEADS, HD, VOCAB = 256, 512, 8, 4, 64, 1200


def make_base(seed=0, layers=L, d=D, ff=FF, vocab=VOCAB, tied=False, std=0.02):
    rng = np.random.default_rng(seed)
    t = {}

    def w(*shape):
        return (rng.standard_normal(shape) * std).astype(np.float32)
    t["model.embed_tokens.weight"] = w(vocab, d)
    for l in range(layers):
        p = "model.layers.%d." % l
        for n in ("q_proj", "k_proj", "v_proj"):
            t[p + "self_attn.%s.weight" % n] = w(HEADS * HD, d)
        t[p + "self_attn.o_proj.weight"] = w(d, HEADS * HD)
        t[p + "mlp.gate_proj.weight"] = w(ff, d)
        t[p + "mlp.up_proj.weight"] = w(ff, d)
        t[p + "mlp.down_proj.weight"] = w(d, ff)
        t[p + "input_layernorm.weight"] = np.ones(d, np.float32)
        t[p + "post_attention_layernorm.weight"] = np.ones(d, np.float32)
    t["model.norm.weight"] = np.ones(d, np.float32)
    if not tied:
        t["lm_head.weight"] = w(vocab, d)
    return t


def finetune(base, seed, scale=0.15, layers=None):
    """base + own change (random, `scale` x the weight std per tensor)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in base.items():
        if v.ndim == 2 and ("layers." in k or k == "lm_head.weight") and (layers is None or k.split(".")[2] in layers or "lm_head" in k):
            out[k] = v + (rng.standard_normal(v.shape) * v.std() * scale).astype(np.float32)
        else:
            out[k] = v.copy()
    return out


def delta(a, b):
    return {k: a[k] - b[k] for k in a if a[k].ndim == 2}


def combine(base, *terms):
    """base + sum(coef * delta)"""
    out = {k: v.copy() for k, v in base.items()}
    for coef, dl in terms:
        for k, v in dl.items():
            out[k] = out[k] + coef * v
    return out


class MemSource:
    """In-memory bf16 safetensors checkpoint (Source interface of reax_lineage.hf)."""
    kind = "mem"

    def __init__(self, tensors, repo="org/model", rev="r0", extra_files=None, dtype="BF16"):
        self.repo, self.rev, self.name = repo, rev, "%s@%s" % (repo, rev)
        self.tensors = tensors
        blob, idx, off = bytearray(), {}, 0
        for k, v in sorted(tensors.items()):
            if dtype == "BF16":
                b = f32_to_bf16_bits(v).tobytes()
            else:
                b = v.astype("<f4").tobytes()
            idx[k] = TensorRef("m.safetensors", off, len(b), dtype, tuple(v.shape))
            blob += b
            off += len(b)
        self.blob = bytes(blob)
        self._idx = idx
        self.extra = extra_files or {}

    def tensor_index(self):
        return self._idx

    def list_files(self):
        fs = {"m.safetensors": FileInfo("m.safetensors", len(self.blob))}
        for k, v in self.extra.items():
            fs[k] = FileInfo(k, len(v), __import__("hashlib").sha256(v).hexdigest())
        return fs

    def read_range(self, path, s, e):
        return self.blob[s:e + 1]

    def read_file(self, path, max_bytes=None):
        return self.extra[path]


def fp_of(tensors, repo="org/model", rev="r0", seed="A0", seed_id="A", spec=SPEC, extra_files=None, layer_map=None):
    return build_fingerprint(MemSource(tensors, repo, rev, extra_files), seed, seed_id, spec, layer_map)


# ------------------------------------------------------- offline hub repos ----

def write_safetensors(path, tensors):
    hdr, blobs, off = {}, [], 0
    for k, v in sorted(tensors.items()):
        b = f32_to_bf16_bits(v).tobytes()
        hdr[k] = {"dtype": "BF16", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


CONFIG = {"architectures": ["TinyForCausalLM"], "model_type": "tiny", "hidden_size": D, "num_attention_heads": HEADS,
          "num_key_value_heads": HEADS, "head_dim": HD, "intermediate_size": FF, "num_hidden_layers": L,
          "vocab_size": VOCAB, "tie_word_embeddings": False}


def tokenizer(extra=()):
    vocab = {"tok%d" % i: i for i in range(VOCAB)}
    added = [{"id": VOCAB + i, "content": t} for i, t in enumerate(extra)]
    return {"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added,
            "normalizer": None, "pre_tokenizer": {"type": "ByteLevel"}}


def write_repo(hub_dir, repo, rev, tensors, config=None, tok=None, readme="# model\n", record=None, extra_files=None):
    p = os.path.join(hub_dir, repo, rev)
    os.makedirs(p, exist_ok=True)
    write_safetensors(os.path.join(p, "model.safetensors"), tensors)
    json.dump(config or CONFIG, open(os.path.join(p, "config.json"), "w"))
    json.dump(tok or tokenizer(), open(os.path.join(p, "tokenizer.json"), "w"))
    open(os.path.join(p, "README.md"), "w").write(readme)
    if record is not None:
        json.dump(record, open(os.path.join(p, "reax-lineage.json"), "w"))
    for k, v in (extra_files or {}).items():
        open(os.path.join(p, k), "wb").write(v)
    return p


HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX"
REV = lambda n: ("%040x" % n)


def record(repo, rev, start_repo=None, start_rev=None, kind="list", uid=None, hotkey=HOTKEY, **kw):
    r = {"schema": "reax-lineage/1", "hotkey": hotkey, "entry": {"repo": repo, "revision": rev},
         "start": {"kind": "none"} if kind == "none" else {"kind": kind, "repo": start_repo, "revision": start_rev},
         "own_models": [], "architecture_changes": {"added_tokens": [], "layer_map": None, "pruned": None, "config_fields": {}},
         "teachers": [], "datasets": [{"name": "synthetic", "source": "https://example.org/data", "licence": "Apache-2.0", "approx_tokens": 1000}],
         "mentioned_not_used": [], "licence": "Apache-2.0", "corrects": None, "signature": None}
    if kind == "entry":
        r["start"]["uid"] = uid if uid is not None else 7
    r.update(kw)
    return r


class World:
    """A small round: official root + list start + entries, an offline hub and a comparison DB (spec SPEC)."""

    def __init__(self, tmp, seed=1):
        self.tmp = str(tmp)
        self.hub = os.path.join(self.tmp, "hub")
        self.dbdir = os.path.join(self.tmp, "db")
        self.listing = {"version": "t1", "models": [], "families": {"tiny": {"root": "org/root@" + REV(1), "config": {"model_type": "tiny", "hidden_size": D}}}}
        self.db = ComparisonDB(self.dbdir)
        self.db.index["families"] = {"tiny": {"root": "org/root@" + REV(1)}}
        self.models = {}
        self.n = 0

    def add(self, repo, tensors, kind="public", start=None, block=None, public_at="2026-01-01T00:00:00Z", uid=None, holder=None,
            flagged=False, result=None, listed=None, rev=None, family="tiny", write=True, record_=None, layer_map=None, private=False, parents=None,
            parent_shares=None, void=False, scored_published_at="default", **wr):
        """`public_at`: when the model / entry became public (an entry too since checker 1.6.0: RULES 2.2 / 4.7 ask whether another holder's entry was public at the commitment; the
        test worlds' entries are public from 2026-01-01 unless `private=True`). `parents` / `parent_shares`: an entry with several declared parents (db row fields)."""
        self.n += 1
        rev = rev or REV(100 + self.n)
        key = "%s@%s" % (repo, rev)
        self.models[repo] = (rev, tensors)
        if write:
            write_repo(self.hub, repo, rev, tensors, record=record_, **wr)
        os.makedirs(os.path.join(self.dbdir, "fp"), exist_ok=True)
        fps = {}
        for sid, seed in (("A", "A0"), ("B", SEED_B)):
            fpa = os.path.join("fp", "%s.%s.npz" % (key.replace("/", "__"), sid))
            fp_of(tensors, repo, rev, seed=seed, seed_id=sid, layer_map=layer_map).save(os.path.join(self.dbdir, fpa))
            fps[sid] = fpa
        meta = {"repo": repo, "revision": rev, "family": family, "kind": kind, "start": start, "public_at": None if private else public_at,
                "block": block, "uid": uid, "holder_ids": [holder or ("hf:" + repo.split("/")[0].lower())], "flagged": flagged,
                "result": result, "official": kind == "official"}
        if scored_published_at == "default":      # v1.6.1 (RULES 2.2 / 4.7, d11): an entry's results were published 2026-01-01 unless a test says otherwise (None = never published)
            scored_published_at = "2026-01-01T00:00:00Z" if kind == "entry" else None
        if scored_published_at:
            meta["scored_published_at"] = scored_published_at
        if private:
            meta["private"] = True
        if void:
            meta["void"] = True
        if parents:
            meta["parents"] = list(parents)
            if parent_shares:
                meta["parent_shares"] = dict(parent_shares)
        self.db.add(meta, fps)
        if listed:
            self.listing["models"].append({"repo": repo, "revision": rev, "family": family, "licence": "Apache-2.0", "use": listed,
                                           "added": "2026-01-01T00:00:00Z", "flagged": flagged, "official": kind == "official"})
        return key

    def finish(self):
        self.db.save()
        lp = os.path.join(self.tmp, "list.json")
        json.dump(self.listing, open(lp, "w"))
        return lp
