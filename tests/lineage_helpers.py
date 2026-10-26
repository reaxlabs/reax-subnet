"""Toy Qwen-style decoder for the rc11.post5 lineage tests (no import of the new modules: the golden test runs on
1.2.0rc11.post4 too)."""

from __future__ import annotations

import json
import os

import numpy as np
from spec_helpers import write_safetensors

from reax_subnet.spec import tensors as T

H, I, NQ, NKV, HD, V, LAYERS = 64, 160, 4, 2, 16, 48, 2
def _bf16(a: np.ndarray) -> np.ndarray:
    return T.as_dtype(np.asarray(a, np.float32), "BF16")


def toy_base(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    w = {"model.embed_tokens.weight": rng.randn(V, H), "lm_head.weight": rng.randn(V, H) / 8,
         "model.norm.weight": 1 + 0.1 * rng.randn(H)}
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        w.update({
            f"{p}.input_layernorm.weight": 1 + 0.1 * rng.randn(H),
            f"{p}.post_attention_layernorm.weight": 1 + 0.1 * rng.randn(H),
            f"{p}.self_attn.q_proj.weight": rng.randn(NQ * HD, H) / 8,
            f"{p}.self_attn.k_proj.weight": rng.randn(NKV * HD, H) / 8,
            f"{p}.self_attn.v_proj.weight": rng.randn(NKV * HD, H) / 8,
            f"{p}.self_attn.o_proj.weight": rng.randn(H, NQ * HD) / 8,
            f"{p}.mlp.gate_proj.weight": rng.randn(I, H) / 8,
            f"{p}.mlp.up_proj.weight": rng.randn(I, H) / 8,
            f"{p}.mlp.down_proj.weight": rng.randn(H, I) / 12,
        })
    return {k: _bf16(v) for k, v in w.items()}


def tune(w: dict[str, np.ndarray], seed: int, rel: float) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {k: _bf16(v + rel * np.sqrt(np.mean(v ** 2)) * rng.randn(*v.shape)) for k, v in w.items()}


def _rms(x, w):
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6) * w


def forward(w: dict[str, np.ndarray], toks: np.ndarray) -> np.ndarray:
    w = {k: v.astype(np.float64) for k, v in w.items()}
    x = w["model.embed_tokens.weight"][toks]
    t = len(toks)
    mask = np.triu(np.full((t, t), -np.inf), 1)
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        h = _rms(x, w[f"{p}.input_layernorm.weight"])
        q = (h @ w[f"{p}.self_attn.q_proj.weight"].T).reshape(t, NQ, HD)
        k = (h @ w[f"{p}.self_attn.k_proj.weight"].T).reshape(t, NKV, HD)
        v = (h @ w[f"{p}.self_attn.v_proj.weight"].T).reshape(t, NKV, HD)
        outs = []
        for hq in range(NQ):
            kv = hq // (NQ // NKV)
            a = q[:, hq] @ k[:, kv].T / np.sqrt(HD) + mask
            a = np.exp(a - a.max(-1, keepdims=True))
            outs.append((a / a.sum(-1, keepdims=True)) @ v[:, kv])
        x = x + np.concatenate(outs, -1) @ w[f"{p}.self_attn.o_proj.weight"].T
        h = _rms(x, w[f"{p}.post_attention_layernorm.weight"])
        g = h @ w[f"{p}.mlp.gate_proj.weight"].T
        u = h @ w[f"{p}.mlp.up_proj.weight"].T
        x = x + (g / (1 + np.exp(-g)) * u) @ w[f"{p}.mlp.down_proj.weight"].T
    return _rms(x, w["model.norm.weight"]) @ w["lm_head.weight"].T


def _pow2(rng, n, lo=1, hi=4):
    return np.exp2(rng.randint(lo, hi + 1, n)) * rng.choice([1.0, 1.0, 1.0, -1.0], n)


def mlp_rescale(w, seed=1, factors=None):
    rng = np.random.RandomState(seed)
    w = dict(w)
    for i in range(LAYERS):
        p = f"model.layers.{i}.mlp"
        s = _pow2(rng, I) if factors is None else factors(rng, I)
        w[f"{p}.up_proj.weight"] = _bf16(w[f"{p}.up_proj.weight"] * s[:, None])
        w[f"{p}.down_proj.weight"] = _bf16(w[f"{p}.down_proj.weight"] / s[None, :])
    return w


def mlp_permute(w, seed=2, share=1.0):
    rng = np.random.RandomState(seed)
    w = dict(w)
    for i in range(LAYERS):
        p = f"model.layers.{i}.mlp"
        perm = np.arange(I)
        sel = rng.choice(I, max(2, int(share * I)), replace=False)
        perm[sel] = rng.permutation(sel)
        for k in ("gate", "up"):
            w[f"{p}.{k}_proj.weight"] = w[f"{p}.{k}_proj.weight"][perm]
        w[f"{p}.down_proj.weight"] = w[f"{p}.down_proj.weight"][:, perm]
    return w


def vo_rescale(w, seed=3):
    rng = np.random.RandomState(seed)
    w = dict(w)
    g = NQ // NKV
    for i in range(LAYERS):
        p = f"model.layers.{i}.self_attn"
        s = _pow2(rng, NKV * HD)
        j = np.arange(NQ * HD)
        chan = ((j // HD) // g) * HD + j % HD
        w[f"{p}.v_proj.weight"] = _bf16(w[f"{p}.v_proj.weight"] * s[:, None])
        w[f"{p}.o_proj.weight"] = _bf16(w[f"{p}.o_proj.weight"] / s[chan][None, :])
    return w


def norm_fold(w, seed=4):
    rng = np.random.RandomState(seed)
    w = dict(w)

    def fold(norm, readers):
        t = _pow2(rng, H, 1, 2)
        w[norm] = _bf16(w[norm] * t)
        for r in readers:
            w[r] = _bf16(w[r] / t[None, :])

    for i in range(LAYERS):
        p = f"model.layers.{i}"
        fold(f"{p}.input_layernorm.weight", [f"{p}.self_attn.{k}_proj.weight" for k in "qkv"])
        fold(f"{p}.post_attention_layernorm.weight", [f"{p}.mlp.{k}_proj.weight" for k in ("gate", "up")])
    fold("model.norm.weight", ["lm_head.weight"])
    return w


def everything(w):
    return mlp_permute(vo_rescale(norm_fold(mlp_rescale(w))))


def save(path, w) -> str:
    os.makedirs(path, exist_ok=True)
    write_safetensors(os.path.join(path, "model.safetensors"), w, "BF16")
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump({"model_type": "qwen3", "hidden_size": H, "num_attention_heads": NQ, "num_key_value_heads": NKV,
                   "head_dim": HD}, fh)
    return path
