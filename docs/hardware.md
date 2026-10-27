# Hardware by pool (release 1.2)

One hotkey serves one pool. The validator checks you against the reference model's answers, so what counts is not
the card but that your runtime reproduces the reference (see [CALIBRATION.md](CALIBRATION.md)).

| Pool | Share of the miner part | Model | GPU | Measured | Runtime |
|---|---|---|---|---|---|
| s1-fast | 0.35 | Plumb-4B | one NVIDIA GPU with **≥ 24 GB** | RTX PRO 6000 Blackwell, H100 80 GB, RTX 4090, RTX 5090, RTX 3090, L40S, RTX A6000 | `runtimes/s1-fast` |
| s1-pro | 0.65 | Surogate Rune 26B-A4B v3 | one NVIDIA GPU of the **80–96 GB class** (52 GB bf16 weights + KV cache) | RTX PRO 6000 96 GB and H100 80 GB (bit-identical results). H200 and B200: expected to work, **not yet measured**. A100 80 GB: **not measured, at your own risk** | `runtimes/s1-pro`, deterministic build **required** |

Cards below the class (for example 48 GB for s1-pro) cannot hold the weights plus KV cache; a quantised build
is not the reference and is struck.

## s1-pro pool: Rune until Fri 30 Oct 2026 09:00 UTC, mammoth-1 from then (release 1.2.0rc11.post8.1)

| | Rune (until the switch) | mammoth-1 (from the switch) |
|---|---|---|
| Weights | `surogate/rune-26b-a4b-GGUF@c6b360d4` (gated), 52 GB bf16 | `baselabs/mammoth-1@61cec44c` (public, MIT), 54.7 GB bf16 on disk, anchored FP8 linears at load |
| GPU memory | 80-96 GB class | measured peak **35.2 GB** per engine (two engines together 67.9 GB); one 48 GB-class or larger NVIDIA GPU |
| Reference runtime | `reax-s1-pro-runtime` (vLLM 0.30.0, deterministic) | `reax-engine-ref-mammoth@sha256:3080a593...` (state-once, 16384 token cap) |
| Temperature / tolerances | 2.0 / 0.20, 0.50, 2.0 | 1.0 / 0.20, 0.50, 2.0 |
| `manifest_id` | `5b11440bf57b5434` | `3f3558e583b0a1d6` |

The mammoth calibration (ref-mammoth `3080a593`, 26 Oct 2026): concurrency-1 answers bit-identical to the production
engine on 500 of 500 items; batched answers drift at most 0.087 mean / 0.33 per item (restart drift at most 0.39),
well inside the tolerances; two 96 GB-card release sessions passed (0.980 / 0.996); noise 1 %. What miners do before
the switch: [MINER.md](MINER.md), "s1-pro switches from Rune to mammoth-1". The capacity unit of the pool
(`target_dps` 10 decisions/s, cap 640/s) is unchanged.

**Validator GPU around the switch.** Rune (58.7 GB) and mammoth (about 35 GB) do not fit on one 96 GB card together with
the s1-fast reference (12.3 GB). A validator therefore runs one s1-pro reference at a time: Rune until 09:00 UTC, stop it
at 09:00 UTC, start mammoth on `REAX_S1_PRO_NEXT_RUNTIME_URL` (a separate loopback port). Until the new runtime answers
the pool is left unscored (no strike, no reset). After the switch the card holds s1-fast 12.3 GB + mammoth about 35 GB,
about 49 GB, leaving more room than before for competition measurements.

## Validator GPU budget (one 96 GB card)

Measured on an RTX PRO 6000 Blackwell Server Edition (97,887 MiB):

| Part | VRAM |
|---|---|
| s1-fast reference runtime | 12.3 GB |
| s1-pro reference runtime (`RUNE_GPU_MEM_UTIL=0.60`, `RUNE_MAX_MODEL_LEN=8192`, bit-identical to the calibration runs) | 58.7 GB |
| one competition measurement at a time (engine 2 x 12.5 GB, or specialist up to 25 GB) | up to 25 GB |
| **Total** | **about 96 GB** |

## Confidential-compute GPUs

On confidential-compute GPUs (e.g. some cloud RTX PRO 6000 / H100 CC instances) set `REAX_VLLM_DISABLE_UVA=1` for the
s1-pro runtime. Without it vLLM returns garbage there, because a zero-copy (UVA) GPU view of pinned host memory does not
see later host writes. The flag loads `runtimes/vllm-ccfix/sitecustomize.py`, which switches vLLM to explicit-copy
buffers; results are unchanged. For a miner, add three flags to the s1-pro runtime container:

```bash
docker run ... \
  -v "$(pwd)/runtimes/vllm-ccfix:/opt/reax-ccfix:ro" -e PYTHONPATH=/opt/reax-ccfix -e REAX_VLLM_DISABLE_UVA=1 \
  reax-s1-pro-runtime:1.2.0rc3
```

## s1-pro needs the deterministic runtime

The pool is scored against the deterministic reference build (`RUNE_DETERMINISTIC=1`, the default in
`runtimes/s1-pro`). Other builds are not accepted:

* **Stock vLLM** (compiled, batch-variant kernels) gives different numbers under different load: logit distance
  to the reference 0.146 median with items up to 3.9; it scores only about 65 % of its rounds (CALIBRATION.md).
* **FP8 / online-quantised** builds score about 12 % of rounds and are struck in about 22 % of them
  (CALIBRATION.md), which keeps the moving average below the 0.3 needed for any weight.

Run `reax-selfcheck --model s1-pro` against your running sidecar before you commit on chain; it must print PASS.

## s1-pro miner quickstart

1. **Get the weights with your own Hugging Face account.** The weights are gated: accept the gate for
   `surogate/rune-26b-a4b-GGUF` on huggingface.co, run `hf auth login`, then
   `hf download surogate/rune-26b-a4b-GGUF --revision c6b360d47895bb77bdf3805a13ee5a5557ab1921 --local-dir /srv/reax/rune-v3`
   (about 52 GB). Never mirror the weights for anyone else.
2. **Build and start the sidecar and the miner:** `docker build -t reax:1.2.0rc3 .` and
   `docker build -f runtimes/s1-pro/Dockerfile -t reax-s1-pro-runtime:1.2.0rc3 .`. Run the sidecar on loopback
   `127.0.0.1:8102` with the weights read-only at `/model` and the shared token file, exactly like the s1-fast
   example in `deploy/docker-compose.s1.yml`; the miner uses `REAX_MODEL=s1-pro` and
   `REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102`. Start-up verifies all 18 weight files (a few minutes).
3. **Self-check:** `REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102 REAX_RUNTIME_TOKEN_FILE=<token file> reax-selfcheck --model s1-pro`.
4. **Commitment:** the miner publishes `reax-s1/1:s1-pro:<manifest_id>` on start (`REAX_PUBLISH_COMMITMENT=1`);
   the manifest id of this release is `5b11440bf57b5434`, so the commitment is `reax-s1/1:s1-pro:5b11440bf57b5434`.
   A different manifest id means a different release and the validator does not query you.

More than one GPU or machine for the same hotkey: run one sidecar per GPU and list them in
`REAX_S1_PRO_RUNTIME_URLS` (MINER.md, "Several GPUs or machines behind one hotkey").

Details: [MINER.md](MINER.md). Validators: [VALIDATOR.md](VALIDATOR.md) and `deploy/validator-1.2/README.md`.
