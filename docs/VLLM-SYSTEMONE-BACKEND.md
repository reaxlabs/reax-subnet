# External vLLM System One backend

The miner can obtain decisions from `POST /v1/systemone` instead of a
`POST /internal/v1/decide` runtime sidecar. Validator authentication, hotkeys,
request limits and response signatures remain in the miner process. vLLM never
receives the miner's keys or the validator's authentication headers.

## Release status (timed switch, 30 Oct 2026 09:00 UTC)

Since the timed pool switch (`manifest.POOL_MANIFEST_SWITCH`, 30 Oct 2026 09:00 UTC) the
live `s1-pro` manifest is `s1-pro-v2-mammoth.json`, id **`3f3558e583b0a1d6`** (mammoth-1,
revision `61cec44c281c693992e3d6ebeb44dd3402dfa186`). A vLLM miner declares that live
manifest; there is no separate vLLM manifest and no second validator reference runtime.
The validators keep scoring against the production reference
`ghcr.io/reaxlabs/reax-engine-ref-mammoth@sha256:3080a593…` (anchored FP8 linears).

`src/reax_subnet/manifests/s1-pro-mammoth-v1.json` is an unregistered draft from before
the switch. It is not in `RELEASE_MANIFESTS`, its id is never published, and a miner
must not commit to it.

**Parity prerequisite.** The vLLM path (bf16, vllm-mammoth) is not the reference runtime.
Before the hotkey commits to `3f3558e583b0a1d6`, the exact server it will use must pass
the self-check against the published reference answers (640 items, pool gates
mean 0.20 / item 2.0 at T = 1; `selfcheck_data/s1-pro-3f3558e583b0a1d6.json`):

```sh
REAX_S1_PRO_BACKEND=vllm-systemone \
REAX_S1_PRO_RUNTIME_URL=http://10.0.0.2:8000 \
REAX_S1_PRO_SERVED_MODEL=mammoth-1 \
reax-selfcheck --model s1-pro
```

The same check without environment variables:
`reax-selfcheck --model s1-pro --backend vllm-systemone --url http://10.0.0.2:8000 --served-model mammoth-1`.
No token file is needed. Exit code 0 = PASS (commit and mine), 1 = FAIL (do not mine
with this setup), 2 = cannot run (server not ready, served model not listed by
`GET /v1/models`, bad URL). The check runs through the miner's backend factory
(`s1backends.load_s1_backend`), so it calls the server exactly as the miner will.
Validators strike a miner whose answers fall outside the gates whatever its
self-check said; the check is a pre-flight, not an approval.

## Miner settings

Use the existing miner wallet/network settings and set these environment variables:

```sh
REAX_MODEL=s1-pro
REAX_S1_PRO_BACKEND=vllm-systemone
REAX_S1_PRO_RUNTIME_URL=http://10.0.0.2:8000
REAX_S1_PRO_SERVED_MODEL=mammoth-1
REAX_S1_PRO_RUNTIME_TIMEOUT=60
REAX_RUNTIME_INFLIGHT=16
```

`RUNTIME_URL` is the server's base URL, without `/v1/systemone`. The served name
must equal vLLM's `--served-model-name`. The timeout is in seconds per decision,
starting after a concurrency slot becomes available (range 0.001–3600). The
inflight limit bounds simultaneous upstream decisions across batches; one signed
batch can contain up to 16 decisions. Results retain their original order and ids.
Replace `S1_PRO` with the relevant model prefix for another approved model.
The default backend is `sidecar`, preserving the existing behavior.

No `REAX_RUNTIME_TOKEN_FILE` is required for this backend. It does not send the
sidecar token or implement vLLM API-key authentication. Keep the endpoint on a
trusted private network or VPN: loopback, literal RFC1918, or ULA addresses are
accepted by default, with HTTP or HTTPS. TLS certificates are verified normally.
The existing `REAX_ALLOW_PUBLIC_RUNTIME=1` override permits public hosts and DNS
names. Do not expose an unauthenticated endpoint publicly.
Only one `RUNTIME_URL` is supported per model; `RUNTIME_URLS` pools and multinode
relay mode are separate sidecar features.

## vLLM with Mammoth

Install `vllm-mammoth==0.1.0` (`pip install --no-deps "vllm-mammoth @ git+https://github.com/reaxlabs/vllm-mammoth@v0.1.0"`) into `vllm/vllm-openai:v0.32.0`, mount the weights
from `baselabs/mammoth-1` at the live manifest's revision
`61cec44c281c693992e3d6ebeb44dd3402dfa186` (the safetensors, config and tokenizer files are
byte-identical to the earlier draft pin `03007ab3…`; only the authors' engine helper files
differ, and the plugin does not use them), and start vLLM with:

```sh
VLLM_BATCH_INVARIANT=1 vllm serve /weights/mammoth-1 \
  --served-model-name mammoth-1 --dtype bfloat16 \
  --max-model-len 16384 --no-enable-chunked-prefill \
  --max-num-batched-tokens 16384 --max-num-seqs 256 --logprobs-mode raw_logprobs \
  --no-enable-prefix-caching --attention-backend TRITON_ATTN
```

The plugin supplies the qev-labels-v2 prompt, slot_lm readout at temperature 1,
and block-bidirectional full-attention mask. Prefix caching must stay off (vLLM 0.32
needs chunked prefill for it on this hybrid model). An 80 GB GPU is the minimum; the
weights alone occupy about 51 GB.

GPU check (27 Oct 2026, 1x RTX PRO 6000): this backend against a live vLLM 0.32.0 +
vllm-mammoth server answered 1,000 validator-shaped decisions (s1items) with 0
contract failures, probabilities identical to direct `/v1/systemone` calls,
32.6 decisions/s at 64 in flight. The plugin's parity against the authors' reference
runtime and its batch invariance are documented in the vllm-mammoth README.

## Mapping and failures

Each decision forwards its state and single typed question unchanged, including
criteria order and question id. The adapter uses only returned probabilities:

- Choice: original option ids, first maximum in original order, confidence = top probability.
- Noul: `yes` maps to `true`, `no` to `false`; `noul = P(yes)`.
- Score: expected zero-indexed level, original legend, confidence = top probability.

The current `s1contract.py` requires top probability for choice and score
confidence. The qev confidence formulas and vLLM's `confidence`, `choice`, `score`,
`noul`, and `legend` fields are not authoritative for this contract. Malformed
probabilities, mismatched model/question/type, or non-finite values are refused.

Upstream HTTP 400 is a request refusal (`SidecarError`, not a backend health
fault); 5xx, 408, 429, connection errors and timeouts are backend faults.
Redirects are not followed. A keep-alive connection that the server closed between
requests is retried once on a fresh connection, only before any answer byte
arrived; once HTTP 200 arrived nothing is retried (truncated or malformed bodies,
bodies over 1 MiB and timeouts after the headers fail the item). Readiness
(`ready()`, used by `reax-selfcheck` and at miner start) is `GET /health` = 200
plus the served model in `GET /v1/models`. Only `Content-Type` goes upstream: no
sidecar token, link signature or validator header.
`decide_each` reports exceptions per item using the existing retry interface.
For validator requests, the existing miner returns unsigned HTTP 500
`{"error":"inference failed"}` for a failed batch, with no fabricated or partial
signed answers. No hotkey signature is added to an upstream refusal.

## Draft provenance and validation

The unregistered draft `s1-pro-mammoth-v1.json` loads only when a test registers it
explicitly; the loader checks model/schema and computes the manifest hash, nothing
more. Its file hashes come from the pinned [Hugging Face tree API](https://huggingface.co/api/models/baselabs/mammoth-1/tree/03007ab3c4481bf01c71ef63adbbbc4b2093d71a?recursive=true&limit=1000)
(LFS oids are SHA-256; small files hashed after download). It is kept as provenance
for the parity work only.

CPU-only checks, using an aiohttp stub and throwaway development signing keys:

```sh
python3 -m pytest tests/test_vllm_systemone.py tests/test_sidecar_reconnect.py \
  tests/test_config.py -q
```

`tests/test_vllm_systemone.py` includes an end-to-end `reax-selfcheck` run against a
stand-in vLLM server that answers all 640 reference items (no token file), and the
transport edge cases above (stale keep-alive, connection refused, 408, truncated and
malformed JSON, the 1 MiB cap, redirects, timeout after headers, upstream headers).
GPU parity is the self-check above on the operator's server.
