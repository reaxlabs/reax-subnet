# REAX yardstick: measure your model yourself

The yardstick runs your model on the **public part of our preview**. It uses the same prompts, the same readout, the
same parsing and the same scoring code as the preview our validator computes for every entry that passes intake. One
command gives you a score per family and the composite with the preview's weights, each with its standard error.

**If your model does well on the Hugging Face Decision Index, it will do well with us too.**

How to interpret your score

* It is a **proxy on public items**. Official scores come from a separate run on **private items that rotate** every
  window (Rules 5.3). Your yardstick number tells you roughly where you stand. It does not predict your official score
  exactly, and a high yardstick number earns nothing on its own.
* The public items are for orientation, not for memorising. A higher public score alone gives no assurance of a higher private score. Several sources ask that their items never be used for training (see [THIRD-PARTY.md](THIRD-PARTY.md)).
  If you train on any public data, declare it in your lineage record with its source and licence (Rules 4.8).
* Everything runs on your own machine. Nothing is sent to us.

## What it measures

| Family | Items | What it is | Weight | How you get the items |
|---|---:|---|---:|---|
| `PV-HF` | 300 | Tasks from the Hugging Face Decision Index public suite (33 benchmarks in this release), HF-style prompt | 0.4 | ids + hashes here; you rebuild the suite locally (below) |
| `PV-HLE` | 509 | Humanity's Last Exam multiple choice with an "I don't know" option, AA-style prompt | 0.2 | ids + hashes here; loaded from `cais/hle` after you accept its terms |
| `PV-W2C` | 3,423 | When2Call: pick the right next reply (answer, call a tool, ask, or decline) | 0.2 | included (`items/when2call/`) |
| `PV-ONET` | 1,000 | EvalSafe O*NET: judgements about synthetic workplace documents | 0.2 | included (`items/onet/`) |

These four families and weights are the preview's. A preview scores about 1,000 of these items: a fixed part plus a part
that rotates every week. The yardstick scores all of them by default, so its standard errors are smaller than a
preview's. The family means estimate the same thing.

Every item is pinned by a hash of exactly what the model sees plus the gold answer (`items/*/index.jsonl`). The option
order is part of the hash. An item whose text does not match is refused and listed, never scored with different text.

## Requirements

* Python 3.10+ on Linux or macOS. The yardstick itself uses only the standard library.
* To serve the model: one GPU that fits it in BF16 (entries have at most 36B parameters, so one 96 GB GPU) and
  **vLLM's OpenAI-compatible server**. The preview pins **vLLM 0.33.0**. The readout needs `/tokenize` and
  `/v1/completions` with `logprob_token_ids`.
* For `PV-HF`: the Decision Index reproduction kit (MIT), to rebuild the public suite on your machine.
* For `PV-HLE`: a Hugging Face account that accepted the `cais/hle` terms, plus `pyarrow` (to read the parquet) or `datasets`.

## Run it

From this directory (`tools/reax-yardstick/`), with a model that vLLM can serve:

```bash
python3 -m reax_yardstick run --model <org>/<repo>@<revision> --vllm \
    --di-suite <path to the rebuilt suite> --hle <path to cais/hle test parquet>
```

`--vllm` downloads the pinned revision and normalises it to a text-only causal LM, as the preview does. It then serves
it with the preview's vLLM flags (BF16, max model length 16,384, prefix caching, seed 0) and stops the server at the
end. If you already serve the model yourself:

```bash
python3 -m reax_yardstick run --model <org>/<repo>@<revision> --engine-url http://127.0.0.1:8000 \
    --di-suite <suite> --hle <hle.parquet>
```

`--served-model-name` sets the name your server expects; the default is the repo id. Without `--di-suite` or `--hle`, those families are
skipped. The composite then uses the families you ran, with the weights renormalised, and the report says
`complete: false`. For a quick check, `--limit 50` takes 50 items per family (a fixed seeded subsample). A stopped
run resumes where it stopped: run the same command again. Use a separate `--out` directory when changing the model, server, or item selection.

Output in `yardstick-runs/<repo>-<revision>/` (or `--out`):

* `summary.md`: the table below, readable
* `yardstick.json`: family means, SEs and item counts, the composite and its SE, the weights used, loader reports and completeness
* `yardstick.items.json`: the score of every item
* `<family>/results.jsonl`: every answer with the per-option probabilities

```
Composite (weights PV-HF 0.4, PV-HLE 0.2, PV-W2C 0.2, PV-ONET 0.2): 61.23 ± 0.84 (SE)
| family  | score (0-100) | SE   | items scored | errors | not loaded |
```

`python3 -m reax_yardstick verify-items --di-suite ... --hle ...` checks every item hash without a model.
`python3 -m reax_yardstick score --out <dir>` rescores a finished run.

### Getting the PV-HF items (Hugging Face Decision Index public suite)

The Decision Index maintainers do not redistribute their suite: some sources forbid republishing, and others are
non-commercial or share-alike. So we do not ship it either. Rebuild it once with their kit
(<https://github.com/apolinario/decision-index>, "The suite"). That takes about 7 GB of downloads and needs the HLE terms accepted:

```bash
python -m decision_index suite rebuild --work work
python3 -m reax_yardstick run ... --di-suite \
    work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
```

`--di-suite` takes files or a directory (`*.jsonl`, `*.jsonl.gz`, read recursively; naming the three row files is
faster). The yardstick picks its 300 items by run id and checks each hash.

### Getting the PV-HLE items

Accept the terms of <https://huggingface.co/datasets/cais/hle>, download `data/test-00000-of-00001.parquet`, and pass
it with `--hle` (needs `pyarrow`). Alternatively, pass `--hle hub` with the `datasets` package and your token. The
yardstick rebuilds our 509 items from the HLE rows: the question without its answer list, the lettered options, and one
decline option ("I don't know" or "It would be best to consult a human expert") as the next letter. It checks each item's hash.

## Same scoring as the preview

The readout code is the preview's, ported line by line:

* System prompt, user message renderers (`render_prompt` for `PV-HF`, `render_prompt_aa` for the other three) and
  the score-rubric handling are the preview's.
* The fast next-token label readout is the preview's: chat template through vLLM `/tokenize`, label-prefix trie,
  `logprob_token_ids`, softmax over label log-likelihoods, first label wins a tie.
* Scoring is the preview's code. `question_correct`, `item_score`, `score_journal`, `family_summary` and
  `weighted_aggregate` are byte-identical (`tests/test_kit.py` pins their hashes). An item scores the share of its
  questions answered right. A yes/no question counts as yes when p ≥ 0.5. A family score is the mean over items (0-100), with an SE
  clustered by item. The composite is the weighted mean, and its SE treats the families as independent.

We checked this. The preview pipeline and the yardstick read the same mock model on all 5,232 items. Every answer,
every item score, every family mean and SE and the composite came out identical.

Differences in the code path, none of which changes a score:

1. The preview runs each family through its pool client in a subprocess, with receipts and cost accounting. The yardstick runs
   the same per-item readout in a thread pool, with a resumable journal.
2. If any item errors, the preview stops and gives no score. The yardstick scores the items that answered, counts the
   errors, and reports `complete: false` (exit code 3). `--retry-errors` asks failed items again.
3. The preview draws about 1,000 items per week; the yardstick uses all items (or `--limit`).
4. Items come through loaders with hash checks instead of frozen copies, so the content is the same by construction.
5. Your vLLM version and GPU are your own. Log-probabilities can differ slightly across vLLM versions and GPUs, so small
   differences from a preview on our hardware are normal.

Not in the yardstick: the preview's comparison of your entry with its parents on the same items. To get it, run the
yardstick on each parent and compare family by family.

## Tests (CPU, no model)

```bash
python3 -m unittest discover -s tests -v
```

The tests check item hashes and both loaders (with synthetic upstream rows), pin the scoring code, and run end to end
against a mock vLLM server (`tests/mock_server.py`), including a resume.

## Licence

Our code is under Apache-2.0 ([LICENSE](LICENSE)). The items keep their own licences and terms: [THIRD-PARTY.md](THIRD-PARTY.md).
