"""Reviewed, text-only joint-readout inference. Repository files are data only.

Allowlist pinned against pplx-decider-v1.2@cce0f3f80033e9d89d80831d5281fd72124aacce.
Full-attention layers are bidirectional; recurrent/linear layers retain their native mask.
"""
from __future__ import annotations

import math
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import prompt as P
from . import rules as R
from . import tensors as T

ARCHITECTURES = ("Qwen3_5Model",)
PINNED_REPO = "perplexity-ai/pplx-decider-v1.2"
PINNED_REVISION = "cce0f3f80033e9d89d80831d5281fd72124aacce"
JOINT_PROTOCOL = {
    "version": 3, "template": "joint-question-inline-readout-v1", "marker": "<|fim_suffix|>",
    "readout_offset": 0, "max_questions": None, "max_length": None,
    "usage": "nonpadding_joint_sequence_including_template_and_image_positions",
}


@dataclass(frozen=True)
class JointReadout:
    hidden_size: int
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]


def detect(model_dir: str, *, category: str, now_ts: float | None = None) -> JointReadout | None:
    """Absent markers/outside generalist opening => causal path. Partial/invalid joint data => reject.

    Call only on the pinned downloaded directory, never on the moving Hub branch.
    No author-provided class name, path, temperature or template controls execution.
    """
    if category != "generalist" or not R.open_class_active(now_ts):
        return None
    cfg = R.read_json_file(os.path.join(model_dir, "config.json"))
    if not isinstance(cfg, dict) or cfg.get("architectures") != ["Qwen3_5Model"]:
        return None
    if os.path.lexists(os.path.join(model_dir, "adapter_config.json")):
        raise ValueError("invalid joint readout: adapters are not allowed")
    present = [os.path.lexists(os.path.join(model_dir, f)) for f in
               ("readout.safetensors", "decision_config.json", "joint_config.json")]
    if not all(present[:2]):
        raise ValueError("invalid joint readout: backbone architecture requires a readout head and decision_config.json")
    for name in ("config.json", "decision_config.json", "readout.safetensors", "joint_config.json", "tokenizer.json", "model.safetensors.index.json"):
        if os.path.islink(os.path.join(model_dir, name)):
            raise ValueError("invalid joint readout: symlink")
    index = os.path.join(model_dir, "model.safetensors.index.json")
    referenced = {"model.safetensors", "readout.safetensors"}
    if os.path.lexists(index):
        data = R.read_json_file(index)
        mapping = data.get("weight_map") if isinstance(data, dict) else None
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(v, str) or not v.endswith(".safetensors")
                       or "/" in v or "\\" in v or ".." in v for v in mapping.values())):
            raise ValueError("invalid joint readout: index weight_map paths")
        referenced.update(mapping.values())
    if any(os.path.relpath(f, model_dir) not in referenced for f in T.weight_files(model_dir)):
        raise ValueError("invalid joint readout: unreferenced safetensors file")
    if cfg.get("model_type") != "qwen3_5" or any(k in cfg for k in ("auto_map", "quantization_config")):
        raise ValueError("invalid joint readout: invalid config")
    text = cfg.get("text_config")
    if not isinstance(text, dict) or any(k in text for k in ("auto_map", "quantization_config")) or text.get("model_type") != "qwen3_5_text":
        raise ValueError("invalid joint readout: invalid text config")
    vision = cfg.get("vision_config", {})
    if not isinstance(vision, dict) or any(k in vision for k in ("auto_map", "quantization_config")):
        raise ValueError("invalid joint readout: vision config")
    vocab = text.get("vocab_size")
    if type(vocab) is not int or not 26 <= vocab <= 1_000_000:
        raise ValueError("invalid joint readout: vocabulary size")
    hidden = text.get("hidden_size")
    if type(hidden) is not int or not 1 <= hidden <= 16384:
        raise ValueError("invalid joint readout: hidden size")
    d = R.read_json_file(os.path.join(model_dir, "decision_config.json"), 65536)
    fields = {"format_version", "base_model", "revision", "attention_mode", "pooling", "temperature",
              "max_pixels", "codes", "token_ids"}
    if not isinstance(d, dict) or set(d) != fields:
        raise ValueError("invalid joint readout: decision config schema")
    if (type(d["format_version"]) is not int or d["format_version"] != 2
            or d["attention_mode"] != "noncausal_full_attention" or d["pooling"] != "last"):
        raise ValueError("invalid joint readout: attention mode/pooling/version")
    if (not isinstance(d["base_model"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", d["base_model"])
            or not isinstance(d["revision"], str) or not re.fullmatch(r"[a-f0-9]{40}", d["revision"])):
        raise ValueError("invalid joint readout: base identity (no paths)")
    if (type(d["temperature"]) not in (int, float) or not 0 < d["temperature"] <= 100
            or not math.isfinite(d["temperature"]) or type(d["max_pixels"]) is not int
            or not 1 <= d["max_pixels"] <= 16_777_216):
        raise ValueError("invalid joint readout: numeric metadata")
    codes, ids = d["codes"], d["token_ids"]
    if (not isinstance(codes, list) or not 26 <= len(codes) <= 256
            or any(not isinstance(c, str) or not re.fullmatch(r"[A-Z]{1,2}", c) for c in codes)
            or len(set(codes)) != len(codes) or codes[:26] != list(string.ascii_uppercase)
            or not isinstance(ids, list) or len(ids) != len(codes)
            or any(type(i) is not int or not 0 <= i < vocab for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("invalid joint readout: answer codes/token ids")
    try:
        tokenizer = R.read_json_file(os.path.join(model_dir, "tokenizer.json"), R.MAX_TOKENIZER_JSON_BYTES)
        model_vocab = tokenizer["model"]["vocab"]
        # BPE/WordLevel use a map; Unigram uses [piece, score] pairs.
        vocabulary = dict(model_vocab) if isinstance(model_vocab, dict) else {v[0]: i for i, v in enumerate(model_vocab)}
        vocabulary.update({v["content"]: v["id"] for v in tokenizer.get("added_tokens", [])})
        if any(vocabulary.get(code) != token_id for code, token_id in zip(codes, ids)):
            raise ValueError("answer codes differ from tokenizer")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid joint readout: tokenizer.json / answer codes: {exc}") from exc
    if present[2]:
        j = R.read_json_file(os.path.join(model_dir, "joint_config.json"), 4096)
        # Canonical JSON comparison also rejects bools standing in for integers.
        import json
        if json.dumps(j, sort_keys=True) != json.dumps(JOINT_PROTOCOL, sort_keys=True):
            raise ValueError("invalid joint readout: joint config schema")
    tensors = T.read_header(os.path.join(model_dir, "readout.safetensors"))
    if (len(tensors) != 1 or tensors[0].name != "weight" or tensors[0].shape != (len(codes), hidden)
            or tensors[0].dtype not in ("BF16", "F16", "F32")):
        raise ValueError("invalid joint readout: weight shape/dtype")
    return JointReadout(hidden, tuple(codes), tuple(ids))


def enable_full_attention(text_model: Any) -> None:
    """Own implementation of the pinned mask contract, using standard Transformers utilities."""
    import torch
    from transformers.masking_utils import create_recurrent_attention_mask

    def mask(_module: Any, args: tuple, kwargs: dict) -> tuple:
        if args or kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("joint readout requires keyword inputs without KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = _module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if isinstance(padding, dict):
            raise ValueError("joint readout expects a padding mask")
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=_module.config, inputs_embeds=embeddings, attention_mask=padding),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask, with_kwargs=True)


class JointRunner:
    """Prefill only; small equal-length batches avoid padding/position ambiguity in hybrid layers."""
    def __init__(self, model: Any, tokenizer: Any, weight: Any, *, max_length: int = 8192,
                 batch_tokens: int = 8192, batch_size: int = 4):
        self.model, self.tokenizer, self.weight = model, tokenizer, weight
        self.max_length, self.batch_tokens, self.batch_size = max_length, batch_tokens, batch_size
        enable_full_attention(model.language_model)

    @classmethod
    def load(cls, model_dir: str, *, device: str = "cuda", gpu_fraction: float | None = None) -> JointRunner:
        import torch
        from safetensors.torch import load_file
        from transformers import PreTrainedTokenizerFast
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from .templates import template_path
        info = detect(model_dir, category="generalist", now_ts=float("inf"))
        if info is None:
            raise ValueError("invalid joint readout: missing metadata")
        if device.startswith("cuda") and gpu_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        torch.backends.cuda.enable_cudnn_sdp(False)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        # Rust tokenizer data only: no AutoProcessor/AutoTokenizer dispatch, no repository template.
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
        with open(template_path("qwen.jinja"), encoding="utf-8") as fh:
            tok.chat_template = fh.read()
        for code, token_id in zip(info.codes, info.token_ids):
            if tok.encode(code, add_special_tokens=False) != [token_id]:
                raise ValueError("invalid joint readout: answer codes differ from tokenizer")
        config = Qwen3_5Config.from_dict(R.read_json_file(os.path.join(model_dir, "config.json")))
        model = Qwen3_5Model.from_pretrained(model_dir, config=config, local_files_only=True, trust_remote_code=False,
                                            use_safetensors=True, dtype=dtype, attn_implementation="sdpa",
                                            device_map={"": device}, output_loading_info=True)
        model, loading = model
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("invalid joint readout: missing/unexpected/mismatched weights")
        model.requires_grad_(False).eval()
        weight = load_file(os.path.join(model_dir, "readout.safetensors"))["weight"].to(device=device, dtype=dtype)
        if not torch.isfinite(weight).all():
            raise ValueError("invalid joint readout: nonfinite weights")
        return cls(model, tok, weight)

    def decide(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float] | None]:
        import torch
        results: list[dict[str, float] | None] = [None] * len(rows)
        plans = []
        for i, row in enumerate(rows):
            try:
                r = P.render(row["state"], row["question"])
                text = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": r.system}, {"role": "user", "content": r.user}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if 0 < len(ids) <= self.max_length:
                    plans.append((len(ids), i, ids, r))
            except Exception:
                continue
        plans.sort(key=lambda p: (p[0], p[1]))
        while plans:
            length = plans[0][0]
            count = min(self.batch_size, max(1, self.batch_tokens // length))
            group = []
            while plans and plans[0][0] == length and len(group) < count:
                group.append(plans.pop(0))
            try:
                ids = torch.tensor([p[2] for p in group], device=self.weight.device)
                with torch.inference_mode():
                    hidden = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                    # Same last prompt position and BF16 matmul -> float logits as reax_parity.py.
                    logits = torch.nn.functional.linear(hidden[:, -1], self.weight).float().cpu().tolist()
                for p, values in zip(group, logits):
                    if all(math.isfinite(z) for z in values[:len(p[3].letters)]):
                        results[p[1]] = P.answer_probs(rows[p[1]], p[3], values[:len(p[3].letters)])
            except Exception:
                # A bad item/group, including CUDA OOM, has the causal path's per-item failure semantics.
                continue
        return results
