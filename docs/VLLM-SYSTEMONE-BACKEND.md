# External vLLM System One backend

The miner can obtain decisions from `POST /v1/systemone` instead of a
`POST /internal/v1/decide` runtime sidecar. Validator authentication, hotkeys,
request limits and response signatures remain in the miner process. vLLM never
receives the miner's keys or the validator's authentication headers.

## Release status

This is a transport adapter and a **draft** Mammoth manifest. The registered
`s1-pro` release still pins Rune. Do not advertise Mammoth as that release or
publish a commitment for it: operators must wait for parity checks, approved
manifest registration, reference updates and coordinated validator release.
Changing the served model setting alone does not change the manifest identity.
The example below is for isolated integration testing until that release exists.

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
from `baselabs/mammoth-1` at revision
`03007ab3c4481bf01c71ef63adbbbc4b2093d71a`, and start vLLM with:

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
`decide_each` reports exceptions per item using the existing retry interface.
For validator requests, the existing miner returns unsigned HTTP 500
`{"error":"inference failed"}` for a failed batch, with no fabricated or partial
signed answers. No hotkey signature is added to an upstream refusal.

## Draft provenance and validation

`src/reax_subnet/manifests/s1-pro-mammoth-v1.json` is deliberately absent from
`RELEASE_MANIFESTS`. Its model/schema are accepted by `load_manifest` when the
filename is temporarily registered in an isolated test. The loader checks only
model/schema and computes the manifest hash; it does not verify GPU parity or
validate arbitrary runtime metadata.

File hashes come from the pinned [Hugging Face tree API](https://huggingface.co/api/models/baselabs/mammoth-1/tree/03007ab3c4481bf01c71ef63adbbbc4b2093d71a?recursive=true&limit=1000):
LFS oids are SHA-256; small runtime files were downloaded from that revision and
hashed with SHA-256. Art, documentation and the authors' executable setup files
are unused by this plugin and excluded. The image index digest was resolved via
Docker Hub's anonymous registry API on 2026-10-26; `base_image_digest` stores it
alongside the requested `base_image` tag. The plugin version is a draft pin,
not a package/source integrity proof.

CPU-only checks, using an aiohttp stub and throwaway development signing keys:

```sh
python3 -m pytest tests/test_vllm_systemone.py tests/test_s1.py \
  tests/test_sidecar_reconnect.py tests/test_config.py -q
```
