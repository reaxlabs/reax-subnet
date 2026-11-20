# Engine templates (Generalist track, round 2)

A **template** is the pair our open engine serves for a model family: the **architecture** (the layer stack the weights must fit) and the
**readout** (the prompt layout, the option codes, and how the model's output becomes option probabilities). Rulebook rules this page backs:

- **2.4:** the architecture and the readout of an entry must be one of the templates our open engine serves. That list includes the readouts
  of the listed starts. Allowed, if declared: added tokens, more or fewer layers, pruning, context settings. A text-only export of a parent
  that also has a vision part needs no declaration. At most 36 billion parameters.
- **2.2:** all parents of an entry are served by the same template (in practice: the same architecture).
- **2.5:** every entry is scored single-pass in its parents' template: each layer runs once, with the template's prompt and readout.
  Layers you add by copying layers must be served by the template at the new layer count.
- **4.2 / 4.3:** a model is a start only if the engine serves it with a reviewed readout template. Other listed models are teachers until
  their template is published.

Engine: <https://github.com/reaxlabs/reax-engine> (registry `engine/registry.py`, per-model notes in `docs/models/`).

## Templates of the listed starts

| Template (engine family / id) | Architecture | Readout | Listed model that uses it | Role today (rule 4.3) |
| --- | --- | --- | --- | --- |
| `jeffan` / `jeffan-31b-1710` | Gemma 4 31B dense, causal | code readout, with conditional re-read (below) | `tattrongvu/Jeffan-31b-1710` | start and teacher |
| `gemma4` / `gemma-4-31b-it` (aliases `gemma-4-31b`, `gemma4-code-v1`) | Gemma 4 31B dense, causal | the same code readout, one read (below) | `google/gemma-4-31B-it` | teacher now; start when this template is published on the list page |
| `qwen38` / `qwen3.8-27b` (aliases `qwen38`, `qwen3.8`) | Qwen3.5 text stack, 27B, causal (`Qwen3_5ForConditionalGeneration`; text-only export allowed) | `d3-code-readout-v1` from the `lm_head` rows (below) | `Qwen/Qwen3.8-27B` | teacher only |

**Served by the public engine today:** `jeffan` only. The `gemma4` and `qwen38` templates are not in the public engine yet; they are listed here so you can see their status. A model becomes a start only when its template is served and published on the list (rule 4.3).

The other models of rule 4.3 (`PelaAI/KnowLine-31B-Gen1` and `-Gen1.5`, `sthanika-ai/Sieve-31B-v3`, `OmniJev/OneJev1.6g-31B`,
`vllm-sr/d3-ultra`, `tattrongvu/Jeffan-27B-1211`) have no published template here. They are teachers until one is published. KnowLine
Gen1 and Gen1.5 use a different prompt layout from the `gemma4` template, so that template does not serve them.

### Readout: code readout (`jeffan`, `gemma4`)

- Prompt: a system line, then one user turn with `State:`, `Question:` and `Options:`. The options are single-token codes `A` to `Z`, then
  `AA`, `AB`, and so on, at most 255. The `gemma4` template uses the wording, codes and renderer of `jeffan`, and the tokenizer and chat
  template of the stock model are byte-identical to Jeffan's.
- Output: the model generates one token. The engine takes the logits of the code tokens only and applies a softmax with temperature 1.
- Limit: the first read may be up to 8,192 tokens; a longer question is refused, never truncated.
- `jeffan` only: a question whose top-1 probability is below 0.9 is read again with the input shown twice, and that distribution replaces
  the first one. If the second prompt would exceed 16,440 tokens, the first read stands. `gemma4` has one read and no re-read.

### Readout: `d3-code-readout-v1` (`qwen38`)

- Prompt: the `d3` prompt contract inside the Qwen chat template, thinking off.
- Output: the readout rows are the rows of `lm_head` for the 255 answer codes, in float32; restricted softmax at temperature 1; causal
  attention. The 255 codes and their token ids equal the `codes` and `token_ids` of `vllm-sr/d3`.

## Parity status

Parity is measured against a Hugging Face reference on the 297 renderable items of the 300-item open-split set (P300); an item is a
"confident flip" when the engine and the reference give opposite answers with high confidence.

| Template | Result |
| --- | --- |
| `jeffan` | unchanged by the new templates: the engine returns identical probabilities on 497 of 497 golden items in a deterministic setting |
| `gemma4`, bf16 (default) | passes. Agreement with a float32 reference 99.66 %, no confident flip, accuracy change +0.34 pp (95 % interval +0.00 to +1.01). Against a bf16 reference: 98.65 % (bar 98.33 %); the one confident flip is a numeric difference of the bf16 reference (float32, the gold label and the engine agree), not a template difference. |
| `gemma4`, FP8 KV | opt-in, not parity-verified (97.31 % agreement, 3 confident flips) |
| `qwen38`, bf16, one request at a time | 98.32 % agreement (bar 96.98 %), accuracy change -0.34 pp (-1.68 to +0.67) |
| `qwen38`, bf16, batched | accuracy change -1.68 pp (-3.37 to -0.34): fails the non-inferiority bar. This is why `qwen38` is teacher only. |
| `qwen38`, FP8 | opt-in, not parity-verified (95.62 % agreement, below the 96.98 % bar) |

bf16 is the default serving path of the new templates; the FP8 presets are opt-in.
