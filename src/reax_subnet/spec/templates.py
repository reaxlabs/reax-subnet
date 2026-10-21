"""Fixed chat templates of the specialist track (docs/SPECIALIST-TRACKS.md, "Chat templates").

A submission's own ``chat_template.jinja`` / ``tokenizer_config.json["chat_template"]`` is never rendered: a template is
code (Jinja) and the validator host parses it inside the vLLM container that holds the GPU. Every allowed architecture
maps to one template shipped in this package (``templates/*.jinja``, plain ``messages`` rendering, no tools), passed to
vLLM with ``--chat-template``. Architectures without an entry are refused by the submission rules.
"""

from __future__ import annotations

import os
from typing import Mapping

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# architecture (config.json) -> template file. The open-class architectures (``rules.OPEN_CLASS_ARCHS``) have a template here
# too, but only the rules of the open class (``rules.rules_at``, from 1.2.0rc9.post6) allow them.
TEMPLATE_FOR_ARCH: Mapping[str, str] = {
    "Qwen3_5Model": "qwen.jinja",
    "Qwen3ForCausalLM": "qwen.jinja",
    "Qwen3MoeForCausalLM": "qwen.jinja",
    "Qwen3_5ForCausalLM": "qwen.jinja",
    "Qwen3_5ForConditionalGeneration": "qwen.jinja",
    "Qwen3_5MoeForCausalLM": "qwen.jinja",
    "Qwen3_5MoeForConditionalGeneration": "qwen.jinja",
    "Gemma3ForCausalLM": "gemma3.jinja",
    "Gemma3ForConditionalGeneration": "gemma3.jinja",
    "Gemma4ForCausalLM": "gemma4.jinja",
    "Gemma4ForConditionalGeneration": "gemma4.jinja",
    "Gemma4UnifiedForConditionalGeneration": "gemma4.jinja",
    "LlamaForCausalLM": "llama3.jinja",
}

# Special tokens each template writes as text. A tokenizer that does not define them would tokenize the markers as plain
# characters and the model would see a prompt in a format it was never trained on (rules.check_tokenizer refuses it).
TEMPLATE_MARKERS: Mapping[str, tuple[str, ...]] = {
    "qwen.jinja": ("<|im_start|>", "<|im_end|>"),
    "gemma3.jinja": ("<start_of_turn>", "<end_of_turn>"),
    "gemma4.jinja": ("<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"),
    "llama3.jinja": ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"),
}

# Files of a submission that carry a chat template or parse as code-like input and are never used (deleted from the
# validator's copy after the download; see sanitize_dir).
TEMPLATE_FILE_PREFIX = "chat_template."


def template_name(architecture: str) -> str | None:
    return TEMPLATE_FOR_ARCH.get(architecture)


def template_path(name: str) -> str:
    return os.path.join(TEMPLATE_DIR, name)
