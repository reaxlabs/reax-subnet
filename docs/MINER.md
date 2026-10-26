# Running a REAX miner

This release ships two mechanisms:

* `REAX_MECHANISM=s1` (default, release 1.1): your miner serves exactly one REAX reference model —
  `s1-fast` (Plumb-4B) or `s1-pro` (Surogate Rune 26B-A4B v3) — in a runtime sidecar container on loopback and
  answers signed validator requests over protocol `reax-s1/1` (`POST /v1/s1/decide`).
* `REAX_MECHANISM=mpm1`: the rc3 mechanism (Qwen3-4B, `POST /v1/decide`), unchanged as a fallback. See
  [the mpm1 section below](#the-mpm1-rc3-miner-fallback) and [MPM.md](MPM.md).

Your score depends on how closely your answers match the validators' reference computation (see
[DESIGN-v1.1.md](DESIGN-v1.1.md)). There is nothing to tune: run the pinned model with the code in this
repository, stay online, answer fast.

## How the s1 miner fits together

Two processes on one machine:

1. the **runtime sidecar** runs the model with exactly the reference engine and serves the reference runtime's
   decision contract (`POST /internal/v1/decide`, header `X-S1M-Internal-Token`) on **loopback only**
   (s1-fast `127.0.0.1:8101`, s1-pro `127.0.0.1:8102`). Never publish this port.
2. the **reax miner** (`reax-miner`) talks to the sidecar over loopback with a shared internal token
   (`REAX_S1_FAST_RUNTIME_URL` / `REAX_S1_PRO_RUNTIME_URL`, `REAX_RUNTIME_TOKEN_FILE`) and serves the public
   signed endpoint (btauth-signed both ways).

Sidecar and reax process share a random internal token: the sidecar reads `S1M_INTERNAL_TOKEN` (env) or
`S1M_INTERNAL_TOKEN_FILE`, the reax process reads `REAX_RUNTIME_TOKEN_FILE`. Keep the token file mode 0600 and
run the sidecar with your uid (`--user "$(id -u):$(id -g)"`, as the compose file does). Both processes
must share one loopback namespace: run them with host networking (as in `deploy/docker-compose.s1.yml`) or in
one network namespace.

Before serving, the sidecar verifies its weights against the release manifest (s1-fast also verifies the
pinned runtime source; s1-pro verifies all 18 weight files, which takes a few minutes) and refuses to run on
any mismatch.

On start the miner also publishes its **commitment** on chain: `reax-s1/1:<model>:<manifest_id>`, signed by
your hotkey (`REAX_PUBLISH_COMMITMENT`, default 1). Validators only query miners whose commitment matches an
enabled pool and this release's manifest; without a matching commitment your miner is not queried and earns
nothing. Changing the commitment (model or manifest) restarts that UID's score from zero.

## Models, licences and hardware (s1)

A hotkey serves one model, or **several models** (below); your commitment says which. Several models on one hotkey and a
competition entry next to the inference element count only from the start blocks of the multirole release
(`elements.MULTIMODEL_START_BLOCK` / `DUALROLE_START_BLOCK`; both are held in a build until the date is announced on
#92). Until then, one model per hotkey.

| Pool | Model | Weights (pinned revision) | Licence | `manifest_id` (this release) | GPU (sidecar) |
|---|---|---|---|---|---|
| s1-fast | Plumb-4B (s1-fast) | `crh225/plumb-4b@55de037801a8a9b9de3db5c0e16cef86210c2186` (public) | Apache-2.0 | `ff6e01b983d48024` | one NVIDIA GPU with ≥ 24 GB; **measured: RTX PRO 6000 Blackwell, H100 80 GB, RTX 4090, RTX 5090, RTX 3090, L40S, RTX A6000** (CALIBRATION.md) |
| s1-pro | Surogate Rune 26B-A4B v3 (s1-pro) | `surogate/rune-26b-a4b-GGUF@c6b360d47895bb77bdf3805a13ee5a5557ab1921` (Hugging Face gated) | Apache-2.0 | `5b11440bf57b5434` | one NVIDIA GPU with 80–96 GB; **measured: RTX PRO 6000 96 GB, H100 80 GB** (bit-identical results) |

For s1-fast, RTX 4090, RTX 5090, RTX 3090, L40S and RTX A6000 are also measured (CALIBRATION.md). Other cards
(L4, A100, H200 …) should work but are **not yet measured**: run
`scripts/calibration_s1` on your GPU first and serve only if it stays inside the release tolerances.

Both models are Apache-2.0. The s1-pro weights are gated on Hugging Face: accept the gate on huggingface.co
with **your own** HF account, log in (`hf auth login`) and download with your own account; never mirror the
weights for anyone else. The s1-pro pool is ON in 1.2 on every network (shares s1-fast 0.35 / s1-pro 0.65; see
[FEATURE-FLAGS.md](FEATURE-FLAGS.md)). Hardware by pool, and the s1-pro quickstart: [hardware.md](hardware.md).

Other requirements: a current NVIDIA driver for the pinned runtime images (CUDA 13 for the s1-fast image),
Docker with the NVIDIA Container Toolkit, 4 cores / 32 GB RAM, disk for the model snapshot (s1-fast about 9 GB
of weights, s1-pro about 52 GB), a public IPv4/IPv6 address with one open TCP port for the miner endpoint (the
sidecar port must **never** be published), and NTP (signed requests older than 10 s or more than 2 s in the
future are refused, in both directions).

## Keys

Use **bittensor-cli 9.x** (`pip install bittensor-cli==9.23.2`) for wallet and registration commands, in a
separate virtual environment from the miner: the `btcli` that comes with `bittensor==11.1.0` (pinned by this
package) has a different command syntax.

The miner needs **only the hotkey** on the server. Create and register keys on a separate machine, then copy
`~/.bittensor/wallets/<name>/hotkeys/<hotkey>` and `coldkeypub.txt` to the server. Never copy the coldkey.
REAX is registered on Finney as netuid 92. Emission start and the TAO emission share are controlled on-chain (subnet owner and root network); check the live chain state. Use a separately configured netuid for testnet or local endpoints.

```bash
btcli wallet new_coldkey --wallet.name reax
btcli wallet new_hotkey  --wallet.name reax --wallet.hotkey miner1
btcli subnets register   --netuid 92 --wallet.name reax --wallet.hotkey miner1   # pays the current burn cost
```

## Get the weights (s1)

The sidecar verifies the weights against the release manifest before serving, so download exactly the pinned
revision:

```bash
# s1-fast (public):
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir /srv/reax/plumb-4b

# s1-pro (gated: accept the gate on huggingface.co with your own HF account, log in once, then download;
# never mirror the weights for anyone else):
hf auth login
hf download surogate/rune-26b-a4b-GGUF --revision c6b360d47895bb77bdf3805a13ee5a5557ab1921 --local-dir /srv/reax/rune-v3
```

Always download with `--local-dir <dir>` and mount **that directory** read-only at `/model` in the sidecar
container. Do not mount a path from the Hugging Face cache (`~/.cache/huggingface/hub/.../snapshots/<rev>`):
there every file is a relative symlink into the cache's blob store, which does not exist inside the container,
so the sidecar fails its weights check and refuses to start. The sidecar verifies every pinned file against
the release manifest (s1-pro verifies all 18 weight files at start, which takes a few minutes) and refuses to
serve on any mismatch.

## Run with Docker (s1, compose)

`deploy/docker-compose.s1.yml` runs an **s1-fast miner**: the runtime sidecar
(`runtimes/s1-fast/Dockerfile`, loopback `127.0.0.1:8101`) plus the `reax-miner` process, both on host
networking so the sidecar is reachable on loopback and unreachable from outside. (The s1-pro sidecar mirrors
this on loopback port 8102 with its own weights and an 80–96 GB GPU.)

```bash
# from the repository root:
docker build -t reax:1.2.0rc3 .
docker build -f runtimes/s1-fast/Dockerfile -t reax-s1-fast-runtime .
cp deploy/miner-s1.env.example deploy/miner.env   # fill in netuid, wallet names, public IP (mechanism/model are set in the compose file)
mkdir -p deploy/secrets && openssl rand -hex 32 > deploy/secrets/runtime-token
chmod 0600 deploy/secrets/runtime-token        # the compose file runs both containers as your uid
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir /srv/reax/plumb-4b
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b \
  docker compose -f deploy/docker-compose.s1.yml up -d --build
```

Details that matter:

* both services use **host networking**, so the sidecar is reachable on `127.0.0.1:8101` from the miner and
  not reachable from outside; never publish the sidecar port. The miner endpoint (default port 8091) binds on
  the host directly and is what you publish on chain.
* the sidecar runs the model on the GPU; the reax miner process itself needs no GPU.
* the compose file mounts the weights directory from your `hf download --local-dir` (env
  `REAX_S1_FAST_WEIGHTS`; never a HF cache snapshot path — its files are symlinks that dangle inside the
  container) read-only at `/model`, plus the shared token from `deploy/secrets/runtime-token` into both
  containers.
* the miner service waits for the sidecar's health check (`depends_on: service_healthy`; the image's
  HEALTHCHECK allows up to 10 minutes for the weights hash and model load).

## Run without Docker (reax process only)

The sidecar always runs as a container (it is the pinned reference runtime). The reax miner process itself
can run from a virtual environment:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements/lock.txt && pip install --no-deps --no-build-isolation .
export REAX_MECHANISM=s1 REAX_MODEL=s1-fast \
  REAX_NETUID=92 REAX_WALLET_NAME=reax REAX_WALLET_HOTKEY=miner1 REAX_EXTERNAL_IP=<public ip> \
  REAX_S1_FAST_RUNTIME_URL=http://127.0.0.1:8101 REAX_RUNTIME_TOKEN_FILE=<path to the shared token>
reax-miner
```

On start the miner refuses to run without a readable, non-empty `REAX_RUNTIME_TOKEN_FILE`, checks the
sidecar with an **authenticated test decision** (if it fails — sidecar down, or the two sides hold different
tokens — the miner exits with an error), checks that the hotkey is registered, publishes
`REAX_EXTERNAL_IP:REAX_EXTERNAL_PORT` on chain if it changed, publishes its commitment
`reax-s1/1:<model>:<manifest_id>` if it changed, and starts serving `POST /v1/s1/decide`. A commitment that
did not land (chain rate limits apply) is retried every 10 minutes until the chain shows it; without a
matching commitment validators never query the miner. `GET /health` returns the protocol, version, model
and manifest_id.

## Check your setup before you register (s1)

`reax-selfcheck` sends 640 published reference decisions through your runtime sidecar exactly as the miner does
and scores your answers with the validator's release constants (per-item logit distance, simulated validator
rounds). It needs no wallet and no chain access. Run it after the sidecar is healthy and before you pay for a
registration, and again after any driver, GPU or image change:

```bash
# Compose: start only the sidecar, then run the check in the miner image with the miner's settings
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b \
  docker compose -f deploy/docker-compose.s1.yml up -d s1-fast-runtime
mkdir -p ~/.bittensor/wallets   # the miner service mounts it; create it as yourself so Docker does not create it as root
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b \
  docker compose -f deploy/docker-compose.s1.yml run --rm --no-deps --entrypoint reax-selfcheck miner
# virtual environment: same variables as reax-miner above
reax-selfcheck            # --json for one machine-readable object; exit code 0 = pass, 1 = fail, 2 = cannot run
```

Individual machines can drift more than their card model (one rented RTX 4090 host scored only about 93 % of rounds
and had items above 2.0, while other cards of the same and other types scored 100 %; docs/CALIBRATION.md), so run the check on each host.
A pass means every item was answered, no simulated round is struck and at least 99 % of simulated rounds
score. It checks fidelity only, not speed: validators also score latency (s1-fast: full speed credit at 0.5 s per
request of 8 decisions, none at 8 s), so compare the printed decisions/s with docs/CALIBRATION.md. On the measured GPUs (docs/CALIBRATION.md) the reference path gives a median distance of about 0.06–0.08; across
25,000 items compared between cards the largest single item was 2.0 and the next 1.4 (batched vLLM bf16: up to about 1.7). Only
items above 2.0 or a failed check matter. A fail means validators would score you 0 or strike you: check that the weights are the pinned
revision, that you run the pinned runtime image (not your own vLLM or an FP8 build) and that the GPU is not
shared with other work. The reference answers ship with the release (`reax_subnet/selfcheck_data/`) and match
its manifest; a release with a new manifest ships new ones.

## What your score looks like (s1)

* Per round the validator sends you 8 fresh typed decisions (a state plus one `choice`, `noul` or `score`
  question, generated fresh per miner and round — there is no answer key to leak) and compares your answers
  with its own reference runtime of the same pinned model in logit space.
* s1-fast: the validator verifies all 8 items. s1-pro: it verifies a secret random 3 of 8, drawn after your
  answers arrive; you cannot tell which items are verified, so you must answer all 8 properly.
* Per verified item the validator computes the per-item distance d: the largest difference between your
  option logits and its own (logit units, recovered from the probabilities, floored at 1e-6 — below that an
  option carries no usable fingerprint). With D = the mean d over the verified items:
  D ≤ 0.20 → the round scores `0.8 + 0.2 × speed`; D > 0.20 → the round is lost (scores 0);
  D > 0.50 or any single item with d > 2.0 → **strike: your moving average resets to 0**.
  The reference runtime on the same GPU type gives D = 0 exactly; across GPU types see CALIBRATION.md.
  Speed comes from the validator's own clock around the signed exchange (bounds s1-fast 0.5–8 s, s1-pro
  1–12 s; beyond the slow bound the request times out). Items are generated fresh per miner and round; exact
  repeats are rare (under 1 %) and a repeat can only replay an answer the real model produced.
* A new miner starts at 0. The moving average of a perfect miner is about 0.34 after 4 rounds (first weight),
  0.64 after 10 rounds, 0.86 after 20 and 0.93 after about 30 rounds (one hour at the default 2-minute
  rounds).
* A wrong or cheaper model, an FP8 build, stock (non-deterministic) vLLM for s1-pro, or a modified prompt: the
  logits drift beyond these limits (measured in CALIBRATION.md) and you lose rounds or get struck. Run the pinned
  runtime images unchanged.
* Slow answers reduce only the 20 % speed term (past the pool's slow-latency bound the request times out and
  the round scores 0).
* Downtime: every missed round counts as zero in the moving average.
* Changing your commitment (model or manifest) restarts that UID's score from zero.

## Several models on one hotkey, and a competition entry next to it

From `MULTIMODEL_START_BLOCK` one hotkey can serve `s1-fast`, `s1-pro`, `s1-lite` and `s1-cpu` (any subset of at least
two; one model keeps the old `REAX_MODEL`). Design: [DESIGN-MULTIROLE.md](DESIGN-MULTIROLE.md).

* **Caps are per model, exactly like separate hotkeys.** Each served model is scored as its own slot with its own pool
  cap, its own score and its own strikes, and gets at most what one hotkey of that model gets; you save the
  registrations, not the hardware. A slot of a model whose manifest id is not this release's is not scored, the other
  models still are.
* **Configure:** `REAX_MODELS=s1-fast,s1-pro,s1-lite` (each at most once, known models only; default: the single
  `REAX_MODEL`, so existing configs are unchanged). The first listed model is the primary one (`REAX_MODEL` may name
  another listed model to change that). Every model has its own sidecar and its own variables, as for a single model:
  `REAX_S1_FAST_RUNTIME_URL(S)`, `REAX_S1_PRO_RUNTIME_URL(S)`, `REAX_S1_LITE_RUNTIME_URL`, ... (two models must not share
  a URL: every answer is verified against that model's own reference). With `REAX_FEATURE_MULTINODE=1` each of
  `s1-fast` / `s1-pro` takes its workers from `REAX_<MODEL>_WORKERS_FILE` (for example `REAX_S1_PRO_WORKERS_FILE`) and
  `REAX_<MODEL>_WORKERS_MIN_HEALTHY` (else `REAX_WORKERS_MIN_HEALTHY`); `s1-lite` / `s1-cpu` stay on one machine
  (`REAX_<MODEL>_RUNTIME_URL`). `REAX_WORKERS_FILE` keeps working with a single model.
* **One endpoint, one lane per model.** The miner routes each signed request by its `model` and `manifest_id` (400 for a
  model it does not serve). Every model has its own runtime slots (`REAX_MAX_CONCURRENT` per model) and its own
  per-validator limits (`REAX_PER_CALLER_CONCURRENCY`, `REAX_PER_CALLER_PER_MINUTE` per model, defaults sized by that
  model's own backends), so a capacity probe of one model never uses up another model's limits and a lost worker of one
  model shrinks only its own limits. `REAX_MAX_PENDING_PER_IP` / `REAX_MAX_PENDING_TOTAL` are multiplied by the number
  of models (the probe windows of all models run together). `GET /health` returns `{"models": {model: manifest_id}}`
  (plus the old `model` / `manifest_id` fields when exactly one model is served). Start-up checks (test decision, pool
  identity, worker checks) run for every model; the miner does not start if one fails.
* **The miner publishes the inference element** `reax-s1/1:<model>:<manifest_id>` for one model,
  `reax-s1m/1:<model>:<id>,<model>:<id>,...` for several (canonical order s1-fast, s1-pro, s1-lite, s1-cpu). Every
  commitment write replaces all fields, so the miner reads its current fields and writes them with **only its inference
  element replaced**: a competition entry on the same hotkey is kept. The 10-minute re-publish compares only the
  inference element, so a competition element you or `reax-commit` wrote never triggers (or loses to) a miner rewrite.
* **Safety rails.** The miner refuses to start with several models before `MULTIMODEL_START_BLOCK` (or while it is
  held) and never writes two elements before `DUALROLE_START_BLOCK` (it logs an error naming the competition element it
  would otherwise drop, and publishes nothing): validators read no role from such a commitment, and a miner must not
  silently delete your competition entry. `REAX_FORCE_MULTIROLE=1` overrides both (nobody scores the result before the
  start; it is for tests and localnets). With `REAX_PUBLISH_COMMITMENT=0` the miner writes nothing and does not check.
* **One competition entry next to inference.** A hotkey can hold one competition entry (use-case/generalist track or
  engine track) next to its inference element from `DUALROLE_START_BLOCK`; the entry is evaluated on our worker, never on
  your serving GPU. Write it with the tools, never with a hand-made `btcli` commit (which replaces all fields):
  * `reax-engine-submit <image>@sha256:<digest>` and `reax-commit set <element>` write `[your inference element, the new
    element]`, keeping the inference element and replacing a competition element of the same kind.
  * `reax-commit show` prints the fields on chain and how this release reads them (elements, held/started, per-model
    manifest match); `reax-commit withdraw competition|inference` removes one element; `--dry-run` prints the fields it
    would write and writes nothing. The tools use the same wallet variables as `reax-miner`.
  * Before `DUALROLE_START_BLOCK`, when the other kind exists, both tools refuse with an explanation unless you pass
    `--replace-all` (write the new element alone, today's behaviour: it drops the other element) or set
    `REAX_FORCE_MULTIROLE=1`.

```bash
# three models on one hotkey (three sidecars on loopback ports 8101-8103), after MULTIMODEL_START_BLOCK:
REAX_MODELS=s1-fast,s1-pro,s1-lite \
REAX_S1_FAST_RUNTIME_URL=http://127.0.0.1:8101 REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102 \
REAX_S1_LITE_RUNTIME_URL=http://127.0.0.1:8103 REAX_RUNTIME_TOKEN_FILE=/srv/reax/runtime-token reax-miner
reax-commit show                      # what is on chain, and how validators read it
reax-commit set "reax-spec/1:demand_forecasting:acme/df-4b@<40 hex>" --dry-run
```

## Several GPUs or machines behind one hotkey

One hotkey can be served by several runtime sidecars of the **same model**. The miner process (one per hotkey, it
alone signs and holds the key) sends each decision to the backend with the fewest requests in flight, never above that
backend's `REAX_RUNTIME_INFLIGHT`. Backends never see the hotkey; they only need the shared token.

* **List them:** `REAX_S1_PRO_RUNTIME_URLS=http://10.0.0.2:8000,http://10.0.0.3:8000` (or `REAX_S1_FAST_RUNTIME_URLS`).
  Each entry is `URL` or `URL|TOKEN_FILE` when a backend has its own token; the others use `REAX_RUNTIME_TOKEN_FILE`.
  The single-URL variable still works and is unchanged (loopback only); set one of the two, not both.
* **Where the backends may be:** loopback or a private network (10/8, 172.16/12, 192.168/16, IPv6 `fc00::/7`), given
  as literal IP addresses. Anything else (public IP, DNS name) aborts start-up unless `REAX_ALLOW_PUBLIC_RUNTIME=1`;
  the miner then warns on every start, because with `http://` the token and every decision travel in clear text. Use a
  private network or VPN, firewall the sidecar ports so only the miner machine reaches them, never publish them.
* **Start-up:** every backend must answer an authenticated test decision, and the miner then sends the first 24
  published reference decisions to each backend. This is a smoke test against published answers, not a proof of the
  backend's weights; the validators' scoring remains the enforcement. A backend whose answers differ from the
  reference model (other weights, other settings) is refused and the miner does not start. A list of several URLs is
  accepted only for models with published reference answers (`s1-fast`, `s1-pro`); `s1-lite` and `s1-cpu` take a single
  URL. Run `reax-selfcheck --model s1-pro` with the same variable
  set: it checks every backend and prints one verdict per URL.
* **Failures:** a backend that fails twice in a row (connection error, timeout, HTTP 5xx) leaves the rotation; the
  miner checks `GET /health` every 5 s and takes it back after two good answers and a repeat of the reference check on that backend alone (while the check
  fails it stays out and the miner logs a warning). A request that failed on one backend
  before any answer byte arrived is retried once on another; never more often, never after the answer had started. With
  every backend out the miner answers 503 instead of hammering them.
* **Limits:** unless you set them, `REAX_MAX_CONCURRENT`, `REAX_PER_CALLER_CONCURRENCY` and
  `REAX_PER_CALLER_PER_MINUTE` default to N times their single-backend values (capped at their maximum: 256 / 64 /
  6000), and the per-validator limits shrink while backends are out, so a capacity probe measures what is really running.

Two machines in one private network (miner on `10.0.0.1`, sidecars on `10.0.0.2` and `10.0.0.3`, each started like the
single-machine sidecar but listening on its private address instead of loopback, same token file on all three):

```
REAX_MODEL=s1-pro
REAX_S1_PRO_RUNTIME_URLS=http://10.0.0.2:8102,http://10.0.0.3:8102
REAX_RUNTIME_TOKEN_FILE=/srv/reax/token
```

One machine with N GPUs: run N sidecars, one per GPU (`CUDA_VISIBLE_DEVICES` or `--gpus device=K`), each on its own
loopback port, and list them:

```
REAX_S1_FAST_RUNTIME_URLS=http://127.0.0.1:8101,http://127.0.0.1:8111,http://127.0.0.1:8121,http://127.0.0.1:8131
```

Adding backends raises throughput and capacity only up to the capacity cap (docs/DESIGN-POOLS.md section 4.1); every
backend must reproduce the reference answers exactly like a single one, so use the same image and weights everywhere.

## Several machines in different places behind one hotkey (multinode, opt-in)

`REAX_FEATURE_MULTINODE=1` (default off; s1-fast and s1-pro only) is for workers that are **not** in one private
network with the miner, and for swapping machines without restarting it. Design and threat analysis:
[DESIGN-MULTINODE.md](DESIGN-MULTINODE.md).

* The **gateway** is `reax-miner` with the hotkey. It needs no GPU: a small, stable server close to the workers
  (within about 20-30 ms; DESIGN-MULTINODE.md section 6) is ideal. It is the only machine that holds the hotkey.
* Every **worker** runs the pinned sidecar exactly as a single-machine miner does (loopback only, never published)
  plus `reax-runtime-relay`, which forwards signed requests from the gateway to the sidecar. Workers never see the
  hotkey. Run `reax-selfcheck` on every worker machine before you add it.

**1. On each worker** (sidecar already running on `127.0.0.1:8101` or `:8102`, token in `/srv/reax/token`):

```bash
openssl rand -hex 32 > /srv/reax/link.key && chmod 0600 /srv/reax/link.key   # one key per worker; copy it to the gateway
# Across the internet: a self-signed certificate (the gateway pins its fingerprint; no CA needed)
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 3650 -subj /CN=reax-relay \
  -keyout /srv/reax/relay.key -out /srv/reax/relay.crt
docker run -d --name reax-relay --restart unless-stopped --network host --user "$(id -u):$(id -g)" \
  --no-healthcheck -v /srv/reax:/srv/reax:ro --entrypoint reax-runtime-relay \
  -e REAX_MODEL=s1-pro -e REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102 \
  -e REAX_RUNTIME_TOKEN_FILE=/srv/reax/token -e REAX_RELAY_KEY_FILE=/srv/reax/link.key \
  -e REAX_RELAY_HOST=0.0.0.0 -e REAX_RELAY_PORT=8190 \
  -e REAX_RELAY_TLS_CERT=/srv/reax/relay.crt -e REAX_RELAY_TLS_KEY=/srv/reax/relay.key \
  reax:<release>
docker logs reax-relay | grep fingerprint     # put this value into the gateway's workers file
```

Firewall the relay port so that only the gateway's IP reaches it. In a WireGuard (or other VPN) network you can
leave out TLS: set `REAX_RELAY_HOST` to the worker's VPN address (e.g. `10.8.0.2`) and drop the two TLS variables;
without TLS the relay refuses to listen on a public or wildcard address.

**2. On the gateway**, one entry per worker in a workers file (key files mode 0600, relative paths are relative to
the file):

```json
{"model": "s1-pro",
 "workers": [
  {"name": "fra-1", "url": "https://203.0.113.5:8190", "fingerprint": "<64 hex digits>", "key_file": "keys/fra-1.key"},
  {"name": "fra-2", "url": "http://10.8.0.3:8190", "key_file": "keys/fra-2.key", "max_inflight": 32}]}
```

```bash
REAX_FEATURE_MULTINODE=1 REAX_MODEL=s1-pro REAX_WORKERS_FILE=/srv/reax/workers.json reax-miner   # plus the usual wallet/axon variables
```

Do not set `REAX_<MODEL>_RUNTIME_URL(S)` on the gateway in this mode (refused); `REAX_RUNTIME_TOKEN_FILE` is not used. It starts when
`REAX_WORKERS_MIN_HEALTHY` (default 1) workers pass the authenticated test decision and the identity check; the
others join when they pass. `max_inflight` (default 16) is that worker's `REAX_RUNTIME_INFLIGHT`.

**3. Swapping a machine without losing the UID.** The gateway re-reads the workers file every 10 s (write it
atomically: write a new file, then `mv` it over the old one).

1. Start the new worker (sidecar, `reax-selfcheck`, relay) and add its entry to the file.
2. Wait for `runtime backend <url> is back in rotation` in the gateway log (about 15 s).
3. Remove the old entry, then switch the old machine off once the log says it left the rotation.

The gateway keeps a removed worker serving while fewer than `REAX_WORKERS_MIN_HEALTHY` of the remaining entries are in
rotation (never more than remain, so scaling down always works), and
an edited entry (new URL, key or limit) keeps serving until its replacement is in rotation, so a typo cannot take
the hotkey offline. To take a machine out at once (compromised, misbehaving), replace its entry with
`{"name": "<worker>", "evict": true}`. To rotate a key, write the new key on the worker, restart its relay and write
the same key into the gateway's key file; the gateway notices the change itself. If a replacement entry never comes
up (a typo), the old entry stays listed and health-checked until you fix or evict the entry. The old key stops working when the
relay restarts and the new entry needs 10-15 s to be admitted, so rotate one worker at a time.

**Moving the gateway** changes the axon IP: start the new gateway (it publishes the new IP), stop the old one when
validators query the new address. Expect a few lost rounds, so keep the gateway on a server that does not move.

The limits (`REAX_MAX_CONCURRENT`, `REAX_PER_CALLER_*`) default to those of the number of workers in the file at
start; restart the gateway or set them after growing the fleet. Capacity caps are the same as for any hotkey
(docs/DESIGN-POOLS.md); multinode does not change what a hotkey can earn, only how many UIDs you need.

## Network protection

The miner endpoint (default port 8091) is public. Application limits (5 s body deadline, 8 unfinished requests
per IP, per-validator rate limits) stop single-source abuse, but not attackers with many addresses. Recommended
for production: put a reverse proxy in front (nginx/caddy: header and body timeouts of a few seconds,
connection limits per IP, `client_max_body_size 256k`) and raise the open-file limit
(`--ulimit nofile=65536:65536` in Docker). The sidecar port must never be published; it binds to loopback and
requires the shared internal token. Note for an s1-pro sidecar: its internal vLLM listens on
`127.0.0.1:8000` **without authentication** — with host networking every local process can reach it (and it
clashes with anything else on :8000, e.g. a second Rune), so keep other local users off the machine.

## Settings

See [CONFIG.md](CONFIG.md). The ones s1 miners usually touch: `REAX_MECHANISM` (default `s1`), `REAX_MODEL`
(`s1-fast` or `s1-pro`), `REAX_S1_FAST_RUNTIME_URL` / `REAX_S1_PRO_RUNTIME_URL` (loopback sidecar URLs; several backends: `..._RUNTIME_URLS`, see "Several GPUs or machines behind one hotkey"),
`REAX_RUNTIME_TOKEN_FILE`, `REAX_PUBLISH_COMMITMENT` (default 1), `REAX_EXTERNAL_IP`, `REAX_EXTERNAL_PORT`,
`REAX_MINER_PORT`, `REAX_MODELS` (several models on one hotkey, see above), `REAX_FORCE_MULTIROLE`. `REAX_MAX_CONCURRENT` (default 8) is the number of requests passed to the runtime at once.
Capacity probes pay verified throughput linearly up to a cap far above one machine, so one hotkey per machine is the
best setup: keep `REAX_PER_CALLER_CONCURRENCY` and `REAX_PER_CALLER_PER_MINUTE` at least at their defaults so a probe
can measure your full speed, and raise `REAX_RUNTIME_INFLIGHT` (default 16) for a batched or multi-GPU runtime.
**Currently s1-fast x4 (2560/s) and s1-pro x4 (640/s) since 1.2.0rc11.post7 (s1-fast was x2 = 1280/s since 1.2.0rc9, s1-pro x2 = 320/s in rc9..rc11.post6 and x1 = 160/s in 1.2.0rc8); the x4 enable has its own dated #92 notice.** The effective miner defaults are s1-fast `REAX_PER_CALLER_CONCURRENCY` 64, `REAX_PER_CALLER_PER_MINUTE` 888, `REAX_MAX_CONCURRENT` 16 and s1-pro 64 / 728 / 16; `REAX_MAX_PENDING_PER_IP` 128 (s1-lite / s1-cpu keep 32 / 456 / 8 and 64).
The defaults are per model and follow that model's pool (since 1.2.0rc11.post7 both are x4: s1-fast 64 / 888 / 16, s1-pro 64 / 728 / 16; `REAX_MAX_PENDING_PER_IP` follows the widest model). The defaults follow the pools' cap (`CAPACITY_CAP_MULTIPLIER`): with the cap at 160/s (s1-pro) / 640/s (s1-fast) they
are 16 / 240 / 8 (`REAX_PER_CALLER_CONCURRENCY` / `REAX_PER_CALLER_PER_MINUTE` / `REAX_MAX_CONCURRENT`); with the
s1-pro cap at x4 (640/s) they are 64 / 728 / 16, because a probe at the new cap sends up to 704 requests
in 8 s over 64 parallel lanes (s1-fast at x4, 2560/s: up to 864 requests in 5 s over 64 lanes, 64 / 888 / 16). Also set `REAX_MAX_PENDING_PER_IP` to at least twice
`REAX_PER_CALLER_CONCURRENCY` (its default now follows it: 32 / 64 / 128): all lanes of one validator start within a few
ms from one IP, and a refused lane ends for the whole probe (behind a reverse proxy all validators share one IP, so
raise it further). A miner still on the old limits is measured at most what its limits allow (the probe's requests
beyond them are refused with 429/503) and never struck; raise the settings to the values above to be measured in full.
The probe's lane count assumes answers of about 0.4 s (s1-fast) / 0.8 s (s1-pro) per request: a miner with a longer
round trip shows less than the cap, as it does today.

**One hotkey, several GPUs or machines (mechanism v2).** With the cap at 2560/s (s1-fast) / 640/s (s1-pro) one hotkey
can stand for a whole fleet: pay is linear in the verified throughput, so one hotkey with 4 × 160/s earns what four
hotkeys with one 160/s machine each earn, and uses one UID instead of four. The miner endpoint of that hotkey must
spread the requests over all machines (the miner-side fan-out; see the miner fan-out section of this guide when it is
released), and the per-validator limits above must be raised as described. Do not consolidate before the higher cap is
live on the stake majority of validators (the release owner announces the date in #92; x1 in 1.2.0rc8, x2 from 1.2.0rc9, s1-fast and s1-pro x4 from
1.2.0rc11.post7 after a live probe bench): until then validators on the old cap pay a consolidated hotkey only up to the old
cap. Registering several hotkeys on one machine
still gains nothing: probes hit all UIDs at the same moment and they share the machine. With `REAX_MECHANISM=s1` the model runs
in the sidecar; `REAX_DEVICE`/`REAX_DTYPE` and `HF_HOME` only apply to the mpm1 (in-process) model.

## Upgrades

Follow the release announcements. When the protocol version or the manifest changes, upgrade before the
announced switch time; requests with an unknown protocol or manifest are refused, and a new manifest means a
new commitment (the score restarts from zero).

## The mpm1 (rc3) miner (fallback)

`REAX_MECHANISM=mpm1` runs the unchanged rc3 miner (`REAX_MODEL=qwen3-4b-instruct-2507`, protocol
`reax-mpm/1`, `POST /v1/decide`): the approved Qwen3-4B model runs **in the miner process** (no sidecar).
Keys, registration and network protection are the same as above; mechanism, wire protocol and scoring are
documented in [MPM.md](MPM.md).

| Part | Minimum (mpm1) | Notes |
|---|---|---|
| GPU | 1× NVIDIA with 16 GB (e.g. RTX 4080/4090, L4, A10) | the model needs about 8 GB in bfloat16 |
| NVIDIA driver | a driver that supports CUDA 13 (R580 or newer) | the pinned `torch==2.14.1` wheel is a CUDA 13 build. With an older driver the miner refuses to start rather than silently running on CPU |
| CPU / RAM / disk | 4 cores / 32 GB / 30 GB | model weights in `HF_HOME` |
| Platform | Linux x86_64, Python 3.12 for the hash-locked install | other platforms: `pip install .[model]` without the lock |

```bash
docker build -t reax:1.2.0rc3 .
cp deploy/miner.env.example deploy/miner.env   # fill in netuid, wallet names, public IP (sets REAX_MECHANISM=mpm1)
docker run -d --name reax-miner --gpus all --init --restart unless-stopped \
  --user "$(id -u):$(id -g)" --ulimit nofile=65536:65536 \
  --env-file deploy/miner.env -e REAX_MECHANISM=mpm1 -p 8091:8091 \
  -v $HOME/.bittensor/wallets:/wallets:ro -v reax-models:/models \
  reax:1.2.0rc3
```

`--user` is required: key files are readable only by their owner (mode 0600), so the container must run with
your host user id. `GET /health` returns protocol `reax-mpm/1`, version and model. Do not change the inference
code path on the mpm1 mechanism: the validator computes its reference with exactly the same prompt,
temperature, batching and dtype; different batching or a different serving engine shifts bfloat16 results
slightly and uses up part of the tolerance.
