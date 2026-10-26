# Engine competition `reax-eng/1` (release 1.2)

Status: **held on finney** in release 1.2.0rc5 (`ENGINE_TRACK_START_UTC = None` in `competition_split.py`): no engine
evaluation runs, no engine submission is published and no engine champion is paid; the engine share goes to inference.
The owner opens the track in a later release; a champion is then paid only with the owner's signed approval
(COMPETITIONS.md, "Owner approval"). On development networks and the testnet `REAX_FEATURE_ENGINE_COMPETITION=1` switches
it on, active at once. The share of the weights and the other
competitions are described in [COMPETITIONS.md](COMPETITIONS.md). Parameters below are release candidates that still need
calibration (section 8).

Goal: make the REAX engine faster. Competitors submit engine images; validators measure them against the reference
engine on their own GPUs; the best verified improvement receives a share of the miner weights.

## 1. What is being optimised

A **track** = one model + one engine contract + one GPU class. The first track is **`s1-fast`**: the model the subnet
already serves (Plumb-4B, pinned weights `crh225/plumb-4b@55de037`, temperature 2.07). The reference engine is the
release's s1-fast runtime (`runtimes/s1-fast`: jevk5, batch-1 CUDA graphs). This is the right first track because:

- every validator already has this model and a 24 GB+ GPU, so evaluation needs no new hardware;
- a winning engine is a drop-in runtime for every serving miner (same contract), so the subnet gets faster as a whole;
- correctness can be checked exactly as serving is checked today (logit distance against the reference).

A second track for the 27B REAX Engine (`reaxlabs/reax-engine`, Perplexity Decider v1.1) fits the same code but needs
96 GB validator GPUs and its own calibration. It is listed as a later step (section 9).

**In scope:** serving code (kernels, batching, scheduling, CUDA graphs, memory layout, tokenisation, process layout,
quantisation only if it stays inside the correctness gate). **Out of scope:** other weights, fine-tuning, changing
the prompt/readout, anything that changes answers beyond the gate, caching across requests, network access.

## 2. How a miner submits

1. Build a container image that serves the engine contract (below), with the label
   `ai.reax.engine.hotkey=<ss58 of the hotkey that will commit it>`, and push it to an allowed public registry
   (ghcr.io, docker.io, quay.io). The label binds the digest to that hotkey: nobody else can claim the same digest.
2. Commit it on chain with a hotkey registered on subnet 92 (one commitment per hotkey):
   `reax-eng/1:<track>:<image>@sha256:<64 hex>` (at most 128 bytes, so the image name is at most ~37 characters),
   e.g. `reax-eng/1:s1-fast:ghcr.io/alice/plumb-fast@sha256:…`. Tool: `reax-engine-submit <image>@sha256:<digest>`.
3. Before submitting, the miner can run the exact validator measurement locally: `reax-engine-eval --reference <ref>
   --candidate <image> --weights <model dir>`.

A competition hotkey does not serve traffic; an operator who wants both registers two UIDs. The image is pinned by
digest, so nothing can change after the commitment. Submission time = the on-chain block of the commitment.

**Engine contract** (the existing runtime sidecar contract, on a unix socket because there is no network):
`POST /internal/v1/decide` with header `X-S1M-Internal-Token` and `GET /health`, served on `$RUNTIME_UDS`
(mode 0666); token from `$S1M_INTERNAL_TOKEN_FILE`; weights read-only at `$PLUMB_MODEL_DIR`; `/tmp` is the only
writable path. The reference runtime got a 5-line change to also serve on `RUNTIME_UDS`.

## 3. How a validator measures, fairly

Every validator measures every new submission itself, on its own GPU, **relative to the reference engine measured in
the same session on the same GPU** (both engines loaded at once, interleaved). Absolute speed never counts, so an H100
validator and an RTX 4090 validator rank submissions the same way.

Per evaluation (`engrunner.evaluate`):

0. **Before anything is pulled:** the registry manifest is read (`docker manifest inspect -v`) and an image above
   **15 GB compressed** is refused (the candidate fails; an unreadable manifest is inconclusive). Every path that does not
   end in a finished evaluation removes the image again (`docker rmi`): not-owner images, failed starts, aborted pulls.
   The reference engine is measured **alone** once, before the candidate starts; this is the session baseline (below).
1. `docker pull <image>@sha256:<digest>` (allowed registries only); the image label must name the committing
   hotkey. `docker run --network none --read-only --cap-drop ALL --security-opt no-new-privileges`, pids, CPU,
   memory (no swap), file and log caps, `--gpus device=N`, weights mounted read-only, fresh socket directory and
   token per engine; the socket must be a real socket (not a symlink). Optional gVisor (`--runtime runsc`).
   **The idle engine is paused (`docker pause`) during every measurement**, so neither engine can load the GPU or
   CPU to slow the other one down.
2. Warm-up, then **timed runs**: 256 fresh decisions at concurrency 16, reference and candidate on the same items,
   two pairs in ABBA order. Score = median of the per-pair throughput ratios (candidate / reference).
3. **Correctness on every timed answer** against the reference engine's answer for the same item:
   mean logit distance ≤ 0.10, every item ≤ 1.0, top choice agrees on ≥ 99 %. This is stricter than serving
   (0.20 / 2.0) so that the serving tolerance never becomes a speed budget; a winning engine is valid for serving.
4. **Anti-cache probes:** items that repeat an earlier state with a different question, and the same question with a
   different state, sent one at a time after their base items. A cache keyed by part of the input answers from the
   cache and fails.
5. **Determinism:** 32 timed items re-sent at concurrency 1 must reproduce the timed answers within 0.05 logit
   (the reference is bit-identical at concurrency 1; honest batched engines move ~0.01).
6. **Latency gate:** 24 items one at a time; candidate p50 ≤ 1.25 × reference p50, and the same distance gates.
7. Nothing the engine reports is trusted: all times are the validator's clock; all answers are recomputed.
8. Budget: one evaluation may take at most 45 minutes; a candidate that fails its first timed run is stopped
   at once. Pull or start trouble that may be ours (registry rate limit, Docker error) is inconclusive and retried
   up to 3 times; a reference failure while a candidate is loaded is checked against the reference alone.

**Fresh, unpredictable items:** the validator generates items after the engine has started (it is already sealed,
without network) with a secret seed, and adds a secret random record id to every state, so no answer table can cover
them. Items never repeat inside the timed runs, so a legitimate exact-match cache cannot help either.

## 4. Anti-cheat summary

| Cheat | What stops it | Shown in the tests |
|---|---|---|
| Answer table / lookup | fresh items + secret record id per item | (by construction) |
| Ask an outside service ("oracle") | `--network none` | GPU test: `phonehome` engine |
| Cache by state or by question | anti-cache probes | local + GPU: `cache` engine |
| Lossy shortcut (cut input, aggressive quantisation, other model) | distance gate on every timed answer | local: `broken`; GPU: `shortcut` |
| Non-deterministic shortcuts | determinism re-run (+ distance gates) | local: `jitter` |
| Fast only when unobserved / slow when checked | correctness is checked on the timed answers themselves | by construction |
| Copy the champion's image | the image label binds the digest to its committer (dev runner: the earliest on-chain commitment) | local: `c_copy`; unit tests |
| Slow the reference down while it is measured (GPU/CPU load when idle) | the idle engine is paused during each measurement | GPU: `hog` engine, paused vs not paused |
| Keep GPU kernels running while paused (a paused container still finishes kernels it queued) | the reference is measured alone at session start; every later reference run must stay within +-12 % of that baseline, else the evaluation is inconclusive and retried; optionally (`REAX_ENGINE_NVML_CHECK=1`) the paused candidate's PIDs are sampled with `nvidia-smi pmon` during each reference window and any SM activity fails the candidate | unit tests (`tests/test_engine_hardening.py`) |
| Pull from an internal address / huge images | registry allowlist, manifest size cap (15 GB compressed) before the pull, pull timeout, image removed after the evaluation and on every failure path | unit tests |
| Rebuild the champion with a trivial change | must beat the champion by the 3 % margin | local: `c_ref`, `c_tiny` |
| Lucky measurement (winner's curse) | the title changes only after a second, fresh session (new containers, new secret items; it may follow the first in the same cycle) also clears the bar; it is held at the lower score, and the champion's own score is held at the minimum of its measurements, so a lucky re-measurement cannot raise the bar | unit tests, local chain |
| Engine that degrades after winning | the champion is re-measured whenever it is challenged; two consecutive invalid measurement sessions vacate the title; the first triggers a re-check | unit tests |
| Escape from the container | no network, no capabilities, read-only, pids/memory caps, optional gVisor; the docker socket lives only in the competition worker, which holds no wallet or token (COMPETITIONS.md, "Where the competitions run") | review item (section 8) |

## 5. Who wins

- A submission counts only if all gates pass **and** it beats the reference by ≥ 3 % (min gain).
- **Champion:** the best valid submission. A challenger takes the title only with ≥ 3 % over the champion's score,
  confirmed by a second, fresh session (same cycle is fine); a pending challenger is displaced only by a submission that beats its pooled score by the same margin, so a later near-copy cannot reset its confirmation; ties go to the earlier commitment.
- **Fixed crowning slots (25 Oct 2026 change, effective Tue 27 Oct 2026 14:30 UTC):** titles change on Tuesdays
  and Fridays at 14:30 UTC. Measurements continue automatically for reviewed entries and in the daily 12:00 UTC batch.
  A challenger that completes its confirmations waits for the first slot at or after confirmation; its confirmation time,
  pooled score and slot are published. It keeps the same displacement protection while waiting. At the slot it must still
  be committed, eligible and above the current champion and parent bars in every confirming session. Its title score is
  the lowest confirming score. Slot finalisation needs no GPU; a missed slot is finalised at the next worker cycle.
  Decay, owner approval, the copy line, parent rule and co-champion bar are unchanged. The published calendar runs through
  March 2027; after it ends, confirmations stay pending until a release extends the calendar. An empty slot calendar
  restores immediate crowning.
- **Frozen thresholds:** margin, min gain, agreement gate, distance budget, items, runs and confirmations live in one block, `src/reax_subnet/engthresholds.py`. Its sha256 is logged when the validator starts and written into the state file (`thresholds_sha256`); changing a value is a new release.
- Each validator keeps its own champion state; Yuma consensus combines them (stake-weighted), as for serving scores.

## 5a. Reusing other entries (copy line)

*From `ENGINE_COPYLINE_START_UTC` in `src/reax_subnet/engcomp.py` (date set in the release notes). Until then the previous
rule applies unchanged: an entry that reuses another miner's files is judged by review alone.*

Recombining what works is a legitimate way to make progress; repackaging is not. The line:

1. **Ideas are free.** Re-implementing a technique you saw in a public entry is always allowed, with no declaration.
2. **Reusing another entry's code, source or binaries, is allowed only if you declare it.** The image carries the label
   `ai.reax.engine.parents`, a JSON list: `[{"uid": 4, "digest": "<64 hex of the parent image>", "files": ["/app/engine.py", "/app/kernels/"]}]`
   (at most 8 parents, at most 2,000 files per parent, label at most 128 KiB; `files` are the paths in YOUR image whose content
   comes from that parent, and an entry ending in `/` is a directory prefix covering everything below it). Such an entry
   counts as **derived**: in every confirming session it must beat **every declared parent** by the full challenger margin (the
   engine track's margin: 3 % or 3x the reference noise, whichever is larger), on top of the normal champion bar.
   - **The parent is its hotkey.** The bar is the best valid score ever measured for ANY digest committed by the hotkey that
     committed the declared parent digest. The declared `uid` must belong to that hotkey: if it does not, the label is a problem
     and that entry is held out of the title until it is fixed. Only when the committer of the digest is not yet known is the
     declared uid's hotkey used, read once and fixed. The set belongs to that one entry: another entry's label can never
     change it, and a uid that is re-registered later changes nothing. Naming an old or weak digest of the parent, or a parent
     whose latest run failed, does not lower it; the best score is kept even if the parent is measured again.
   - **The parent is measured in the same session** (from Tue 27 Oct 2026 14:30 UTC; not
     retroactive). Scores of one image move between sessions by about as much as the margin, so when a derived entry's session
     clears the champion bar (less one margin of slack, for the champion's same-cycle re-measure; crowning itself uses the
     real bars) but is below, or within the margin of, the stored parent line above, the validator measures the binding
     parent image (the digest that gives that line) right after it: same machine, same reference, the same items. For that
     session the line is `max(parent in this session, that image's lowest valid score) x (1 + margin)`. The floor is the
     parent image's lowest **valid** score (ok and at least the gain threshold); a session of the parent that is merely ok
     but not valid never lowers it, and a slow parent run cannot lower the line below it. Each confirming session measures
     the parent again. If the parent image cannot be run (no hotkey commits it any more, not pullable, failed, over budget)
     the stored line applies. Cost bounds: up to 3 parent runs per batch cycle, each limited to 20 minutes, and a parent
     whose run failed or timed out earlier in the cycle is not run again in that cycle (stored line). This run is a reference only: it
     never changes the parent's own score, title or ranking.
   - **No score, no crowning.** A declared parent with no valid score yet holds the entry (never crowned, never on a missing
     bar) until the parent has been measured.
   - **Only declared parents set bars.** The validators build the bar from the label alone. Reuse that review finds but the
     label does not declare is not a bar: it is a copy (point 4).
   - **Read first.** From the start date an entry can be crowned only after its review record has been read: a missing or
     unreadable record, or a label with any problem (bad JSON, nested deeper than 4 levels, more than 8 parents or 2,000 files
     per parent), keeps that entry out of the title until it is fixed. A validator without a review directory crowns nobody.
     A record written before the start date has no label fields: for an unmeasured digest the check runs again before its first
     run; for an already-measured, eligible digest the validators re-read the image labels once (docker image inspect). Until
     then it cannot be crowned, but it stays in the pool and nobody has to recommit. Only an image that is gone (not on
     the host and not pullable by its digest) counts as "no parents declared" (it was measured before the rule; undeclared
     reuse is still caught by a COPY marker). A docker error or timeout changes nothing: the re-read is retried next cycle
     and the entry stays out of the title until it succeeds. A measured digest whose record is missing stays out of the
     title (in release mode every measured digest has a record).
3. **Binaries are treated like source, and closing code gives no protection and no payout.** A closed binary (`.so`, `.cubin`, ...
   without published source; `.safetensors` and `.bin` files inside directories named exactly model, models, weights,
   checkpoint(s) or tokenizer(s) are weights and not counted; any other `.bin` is listed as "weights? (check)") may be reused under the same declaration and the same
   full-margin rule. But a component without published source keeps its entry **unpaid**: the owner's signed award approval is
   withheld until the source is published. This is done by hand (no code gates it): **closed components present -> withhold the
   approval**; it is on the review checklist and printed by the review tool.
4. **Undeclared reuse found by our review is a copy.** The reviewer creates `COPY/<digest>`; **the marker alone is enough**
   (`REVIEWED/<digest>` is not needed and is not created for a copy). In that cycle the commitment fails (it is never run) and
   counts as a strike. The first strike has no further consequence; the **second and every later** one pauses that hotkey for
   **7 days** (50 400 blocks; a later strike restarts the 7 days from its own block, never shortening a longer pause). A marker
   put on an entry that was already measured, a former champion or the champion has the same effect: its result becomes a
   failure, it is vacated if it holds the title, and it leaves the former-champion ranks and the keep-alive.
5. **The same owner may reuse its own code freely**: parents on the same coldkey (chain snapshot), or a signed ownership proof
   (below), are not constraints and are not copies.

**While a hotkey is paused** it is out of the challenger pool, out of confirmation sessions and cannot be crowned; its new
commitments are not evaluated. What it already earns from earlier, non-copy entries continues: the champion share of a
sitting champion and the former-champion ranks (the former-champion slot keep-alive continues); waiting entries of a paused
hotkey get no waiting keep-alive. A copy itself earns nothing.

**Where the declaration lives.** In the image label, not in the chain commitment: the on-chain field is 128 bytes and an engine
commitment already uses most of it. The label is part of the image configuration, which is part of the digest, so the declaration
is bound to exactly the image that is run and cannot be changed afterwards; the pre-run check already reads the labels.

**Ownership proof (different coldkeys, one owner).** The parent's hotkey signs (sr25519) the text
`reax-eng/1 same-owner:<parent hotkey>><child hotkey>:<parent digest>` and the child image carries
`ai.reax.engine.ownership=[{"parent_digest": "...", "parent_hotkey": "...", "signature": "<hex>"}]`. Validators accept it only
when the signature verifies (`bittensor.sp_core.verify`) for exactly that parent digest and child hotkey; without the chain
library the proof is not accepted.

**Review tooling and checklist.** `tools/engine_parents_check.py CHILD_DIR --parent DIGEST=DIR ... --declared <label>` compares the
extracted trees and prints the identical files and binaries, declared vs detected, and the verdict: "derived (declared)", "copy
(undeclared)", "same owner", "independent" or "unverified declaration". It needs no docker. Pass every digest of the parent's
hotkey; a declared digest without a tree is reported (exit 3). Checklist: (1) run the tool against the declared parents' hotkeys;
(2) shared but undeclared files -> create `COPY/<child digest>` (exit 2); (3) closed components listed -> withhold the award
approval until their source is published; (4) otherwise create `REVIEWED/<digest>` as usual.

## 6. Rewards (share of miner emissions)

The split between serving and the competitions is defined once in [COMPETITIONS.md](COMPETITIONS.md)
(`competition_split.py`). For the engine:

- **Active profile `80/20`: 10 %** of the miner part for the engine champion (profile `50/50`: 20 %). A release constant.
- Inside that share, **winner-take-most with decay**: when crowned the champion gets all of it; its part then decays
  linearly to 50 % over 14 days (100,800 blocks), and the decayed part goes back to serving. A new champion starts at
  100 % again, so the prize for beating the champion stays high while a long reign without progress pays less.
- **Runner-ups are off by default** (code supports them: halving by rank).
  Reason (review finding): an engine is an image, not weights, so a rebuilt copy of the champion or of the
  reference has a new digest and could hold a runner-up slot with one lucky measurement. Switch on only together
  with confirmation and similarity checks for runner-ups.
- **No champion, a struck or withdrawn champion, the decayed part = back to serving.** Never burned, never to the owner
  UID. Before the start time the weights are byte-for-byte the 1.1 weights (tested).
- **Opening co-champions (1.2.0rc9.post7, one-time exception).** In the opening round uid 4 (2.823) and uid 198 (2.814)
  measured within the margin of each other, so they share the title. While one of them holds it, the champion's part
  (decaying from the opening crowning, block 9,326,968) is split 50/50. Each half is paid only after that entry's own
  award review, and only while it stays committed by the same hotkey with a valid result. An unpaid half goes back to
  serving, never to the other co-champion. A challenger must beat 2.823 by the margin (2.908). Once a challenger takes the
  title, the normal rule applies again. From 1.2.0rc10.post1 the pair counts as one former champion: each of them is
  paid half of whatever a former champion gets, each half only after that entry's own award review.
- **Former champions (1.2.0rc10.post1).** Of the champion's part (after the decay above), the previous champion gets
  17 % and the one before it 8 %; the champion keeps the rest (75 %, or 83 % when only one former champion holds a slot,
  100 % when none does). A former champion's slot ends 7 days (50,400 blocks) after it lost the title, or earlier when two
  newer champions push it out. It is paid only after its own award review and only while it stays committed by the same
  hotkey with a valid result; an unpaid slot goes back to serving. While a new champion waits for its award review, the
  most recently crowned approved former champion is paid the champion's part (decaying from its own crowning) instead of
  its slot: the whole part when no other former champion holds a slot, otherwise the part minus the other slot.
- **Keep-alive for the champion and the former champions (1.2.0rc9.post8, extended in 1.2.0rc10.post1).** With the
  engine keep-alive on, the small keep-alive that waiting entries get (half the median serving weight each, together at
  most 5 % of the engine share) is also paid, ahead of the waiting entries, to (a) the champion while its award review is
  pending and (b) every former champion that holds a slot (both opening co-champions when the pair holds it), until 7 days
  (50,400 blocks) after it lost the title, while it stays committed with a valid result. This is a keep-alive, not an
  award: it needs no approval, and an entry paid anything else (including its slot) gets none. At most 4 such entries on
  top of the 4 waiting entries. The engine keep-alive is taken from all paid engine shares in proportion.
- The share is taken from the miner part only; the owner's burn fraction is unchanged. If a validator sets nothing (full
  burn without an owner UID), it still sets nothing.

## 7. What the branch implements (flag `REAX_FEATURE_ENGINE_COMPETITION`)

- `engcomp.py` (pure rules), `engrunner.py` (docker + dev runners, measurement), `engvalidator.py` (cycle, state,
  weights), `engcli.py` (`reax-engine-eval`, `reax-engine-submit`), hooks in `s1validator.py` and `cli.py`,
  config in `config.py`, the share layer in `competition_split.py`. On finney the competition is on by release (a
  validator needs `REAX_ENGINE_WEIGHTS_DIR` and `REAX_SPEC_SEALED_DIR`; the reference engine is the release constant
  `config.ENGINE_REFERENCE_IMAGE`; without them it serves only; the flag is refused so that no validator can
  switch it off); elsewhere the flag decides. A validator on a custom endpoint whose genesis hash is finney's refuses to
  start (use `REAX_NETWORK=finney`), so every mainnet validator runs the same rules and start time.
- The evaluation cycle (every 6 h, at most 4 new submissions, oldest first, plus re-measurements) runs in the validator
  process (`REAX_COMPETITIONS_MODE=local`) or in `reax-competition-worker` (`remote`, `worker.py`; the validator then
  reads `engine-competition.json` through `remote.py`). Serving rounds go on during a cycle (the GPU is shared with
  the serving runtimes), and weights are still submitted between evaluations.
- Tests: `tests/test_engcomp.py`; local-chain proof `scripts/localnet/e2e_engine.py` with the synthetic engines in
  `scripts/localnet/fake_engine.py`.

## 8. Before switching on (calibration and review)

1. **Calibrate on the s1-fast GPU classes** (4090, 5090, H100, PRO 6000): honest engines' distance and determinism
   distributions, and the run-to-run noise of the speed ratio (sets min gain and margin; the GPU test measured it once).
2. **Security review of the runner**: docker flags, socket directory, image pull, resource caps;
   decide whether gVisor is required. Validators run third-party images: this must be opt-in and documented.
3. **Validator cost:** ~5 min of GPU per evaluation; at most 4 new submissions per 6 h = ≤ 1.5 GPU-h/day per validator.
   **GPU memory on a shared 96 GB card:** the serving runtimes (s1-fast ~12 GB, s1-pro) stay resident. A measurement adds
   two s1-fast engines (reference + candidate, ~12 GB each, so 28 GB with headroom); the idle one is paused, which stops
   its compute but keeps its memory. Before each cycle the runner reads the free GPU memory and skips the cycle (retried
   next interval, no candidate is marked failed) if less than 28 GB is free. **To check on the real host:** `nvidia-smi`
   with all runtimes up and one `reax-engine-eval` run.
4. Testnet run with at least two independent validators to confirm they agree on the champion.
5. Release: version bump, release notes, miner docs, and a short public page ("make the REAX Engine faster").
6. Licence condition for submissions: entrants license their changes to the licensor of the REAX Engine (the REAX
   Engine License 1.0 grants this on submission); the REAX parts stay under the REAX Engine License, and versions released
   before the licence change stay Apache-2.0. A winner's change may be merged into the reference runtime under these terms;
   the payout itself does not require a merge.

### Open items from the code review (not fixed on the branch)

- The engine's socket directory is a bind mount on the worker's disk; a hostile engine could fill it while it runs
  (the directory is removed afterwards). Fix before release: a size-limited mount for it (XFS project quota or a small
  loop-mounted filesystem); until then keep `REAX_COMPETITION_DIR` on its own filesystem.
- Independent validators measure on different GPUs; the local chain showed two validators agreeing, but the
  real-GPU agreement between validators needs the testnet run (step 4 above).
- Container escape is mitigated, not excluded: decide whether gVisor (`runsc`) is mandatory for validators.
- Runner-up payouts stay off until there is a similarity check for rebuilt copies.

## 9. Later

- Track `pplx-27b` for the 27B REAX Engine (96 GB GPUs; the reference = current reax-engine FP8 build).
- Adopting a champion as the new reference runtime (new manifest; serving miners switch).
- A signed public scoreboard per validator (helps miners and audits).


## Reference determinism release gate

Every Docker engine (reference and candidate, including the separate worker and CLI)
receives `FLA_CACHE_MODE=strict` and `FLA_CONFIG_DIR=/opt/reax/fla-configs/<GPU-class-directory>`.
`engconfigs.GPU_CONFIG_DIRS` maps the exact `nvidia-smi --query-gpu=name` value to a
directory; `PUBLISHED_GPU_CLASSES` admits only classes with published, measured configs.
Unknown/unpublished hardware produces a logged, persisted `session_void` event before
containers start.

**What is pinned.** One class is published: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
(`runtimes/s1-fast/fla-configs/rtx-pro-6000-blackwell-server/`: six kernel config files,
`_meta.json`, `SHA256SUMS`, provenance in the README). They were measured on an idle GPU of that
class with the reference image `ghcr.io/reaxlabs/reax-engine-s1-fast@sha256:9bd41632e33bd90377f7e29bdb9410b18cfe1fb557200160b9ebe9222abcecea`
(fla 0.5.2, Triton 3.7.1). `engconfigs.PINNED_CONFIG_MANIFEST_SHA256` pins the hash of `SHA256SUMS`,
which pins every file. The Workstation Edition has no configs and stays VOID.

**How they reach the engines.** The reference image ships no configs and its entrypoint
does not call `engfla`; strict mode is read by the fla library inside the image itself
(`fla.ops.utils.cache`: exact key match, `FLA_CONFIG_DIR/<kernel>.json`). The runner therefore
bind-mounts the directory read-only at the `FLA_CONFIG_DIR` path into **every** engine container
(reference and candidate: identical files, nothing for a miner to ship or tamper with; the root
filesystem stays read-only). The host directory is `runtimes/s1-fast/fla-configs` of the validator's
checkout, or `REAX_ENGINE_FLA_CONFIG_ROOT`. Before every session `engconfigs.verify_published_configs`
checks `SHA256SUMS` against the pin, every file against `SHA256SUMS`, no extra/missing/symlinked files;
any mismatch makes the session VOID (no result, score, strike or attempt).

**Coverage caveat.** Only the six kernels decorated with `@fla_cache_autotune` consume the pins
(`chunk_local_cumsum_scalar_kernel`, `l2norm_fwd_kernel`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
`chunk_fwd_kernel_o`, `recompute_w_u_fwd_kernel`, `chunk_gated_delta_rule_fwd_kkt_solve_kernel`; seven
measured autotune keys). `l2norm_fwd_kernel` is additionally pinned for every row bucket NB 3-256 (up to
16.7M rows) to BT=16/num_warps=4, the config the reference engine forces itself, because some of its
configs round differently (a minority bit-class) and a candidate without its own pin would otherwise
autotune it for large prefill calls. Of ~196 fla autotuners only these ran in the measured 2k-item workload; plain
`@triton.autotune` kernels would still autotune at runtime if a workload reached them. Strict mode
also falls back to timing autotune on an exact-key miss: `engfla.require_fla_configs` (which forbids
that) is not part of image 9bd41632, so a candidate using other shapes can autotune. The
repeatability self-check below and the reference comparisons are the guards for that.

**Reproducing.** Miners run their engine with the same `-e FLA_CACHE_MODE=strict -e
FLA_CONFIG_DIR=/opt/reax/fla-configs/rtx-pro-6000-blackwell-server` and
`-v <checkout>/runtimes/s1-fast/fla-configs/rtx-pro-6000-blackwell-server:/opt/reax/fla-configs/rtx-pro-6000-blackwell-server:ro`
on a Server Edition GPU, after `sha256sum -c SHA256SUMS` in that directory. Serving runtimes built from
`runtimes/s1-fast/Dockerfile` also contain the directory and the `engfla` wrapper.

Before the candidate is loaded, each session repeats the same public seeded check set
(seed `0x52454158`, `timed_items` items) twice on the reference, using the existing
mean/item distance and top-choice correctness gates. Missing answers or a failing
self-comparison make the session VOID: no result, score, miner strike or infrastructure
attempt is recorded; a `session_void` history event survives worker restarts. This is a
repeatability check, not an independent proof of the reference's accuracy or a
cross-process comparison with a golden answer set. GPU release validation must also
compare answers across fresh idle/loaded starts.

A first invalid champion session retains its held score and schedules a fresh re-check
(in the confirmation loop when available, otherwise next cycle). Only two consecutive
invalid sessions vacate the title; a valid measurement resets the streak. Reprocessing
the same session cannot increment the count, and VOID sessions do not count as invalid.
Withdrawal/change of the champion's committed hotkey still vacates immediately.
The invalid count and last session identity are persisted with backward-compatible defaults.

Both the minimum gain over the reference and the margin over a champion or pending
challenger use at least `max(3%, 3 * reference_noise_std)`. The noise unit is
`sample_std(fresh_process_reference_throughputs) / mean(throughputs)`.
`engconfigs.PINNED_REFERENCE_NOISE_STD` is 0.00167 (11 idle fresh-process reference samples on the
Server Edition: mean 65.70 dec/s), so the margin stays at the 3 % floor; it is part of the frozen rules
hash (`engthresholds.py`). With multiple admitted GPU classes the single release noise constant must
conservatively cover all classes. A second fresh
session is still required to crown a challenger.

Serving verification uses the same s1-fast sidecar/JevK5 FLA kernel path, with broader
0.20 mean / 2.0 per-item tolerances. Its rules and container environment are unchanged
by this engine-only patch; the runtime image gains the config directory for competition
use. Before opening the engine track, the release owner must measure serving-side drift
and decide separately whether to deploy pinned configs to serving runtimes.

## Security: runner hardening and submission review (unreleased)

**Runtime (release mode / finney).** Every engine, reference and candidate, runs under gVisor. `preflight()` and the start
itself refuse unless `REAX_ENGINE_DOCKER_RUNTIME=runsc` is set **and** `docker info` lists a `runsc` runtime; if `docker info`
cannot be read the answer is no (fail closed). The `runsc` entry must also point at a binary named `runsc` and carry
`--network=none`, `--nvproxy=true`, `--nvproxy-allowed-driver-capabilities=compute,utility` and
`--host-uds=create` in its `runtimeArgs` (daemon.json), otherwise it is not accepted. The host UDS creation mode exposes
`RUNTIME_UDS=/run/reax/engine.sock` in the bind-mounted per-run I/O directory so the runner outside gVisor can connect.
It permits creating host sockets in mounts without permitting connections to existing host sockets. Any other
`--host-uds=` value (including `open` or `all`) and the deprecated `--fsgofer-host-uds` flag are refused, even alongside
`--host-uds=create`. A later `--network=` flag must still be `none`. For example:

```json
{"runtimes":{"runsc":{"path":"/usr/local/bin/runsc","runtimeArgs":["--network=none","--nvproxy=true","--nvproxy-allowed-driver-capabilities=compute,utility","--host-uds=create"]}}}
```

The result of a runtime refusal is a `session_void` event: no score, strike or attempt is
recorded and no container is started. Development networks and the testnet keep working without runsc.

**Container flags** (all engines): `--network none --read-only --cap-drop ALL --security-opt no-new-privileges`,
`/tmp` as tmpfs `rw,exec,nosuid,nodev,size=16g` (exec stays for Triton's JIT), and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility` so an image's own ENV cannot widen the driver capabilities.

**Paused-candidate GPU check (`REAX_ENGINE_NVML_CHECK=1`).** The check matches `docker top` PIDs against `nvidia-smi pmon`.
Under runsc `docker top` reports the sandbox's *internal* PIDs, which never equal the host PIDs of the sentry that owns
the GPU, so the check would always say "idle": a claim that looks like protection and is none. It is therefore
**explicitly disabled under runsc** (a warning is logged at start-up and the engine gets no `gpu_busy` probe) instead of
being matched through sandbox host PIDs, which could not be validated without GPU hardware and would break silently when
the runtime changes. What still covers queued GPU work of a paused candidate under runsc: the reference is measured
alone first and every later reference run must stay within the drift bound of that baseline (otherwise the session is
inconclusive), and `docker pause` freezes the sandbox's tasks. The per-PID check remains available under runc (development).

**Cycle gate (optional).** With `REAX_ENGINE_CYCLE_GATE_FILE` set, a cycle starts only while the file exists (regular
file) and its mtime is younger than `REAX_ENGINE_CYCLE_GATE_MAX_AGE_MIN` (default 60 min); the gate is checked again before
every session, so closing it stops the cycle after the running session. A cycle that ran appends a `cycle_end` event
(`evaluated`, `gate_open`) to the state history and the log. An orchestrator touches the file when the reference rotation
allows candidate runs and removes it before the rotation. Unset: behaviour as before, no event.

**Pre-run submission check.** Before the first run of a digest (`engreview.py`, nothing of the image is executed;
`docker image inspect`, then `docker create` + `docker cp` of known files) the validator records
`<review dir>/reviews/<digest>.json` and rejects, with the reason in the result, an image that: runs as root (`User` root,
`0`, or root group; no `User` is only a warning because the runner forces `--user 10002:10002`), exposes ports, declares
`VOLUME`s (anonymous volumes are writable under `--read-only`), sets proxy / `LD_PRELOAD` / `LD_AUDIT` in `Env`, carries
privileged hints (`--privileged`, `docker.sock`, `nsenter`, `cap_add`, ...) in Entrypoint, Cmd, Env or labels, has no
Entrypoint/Cmd, is above 40 GB unpacked, or is not linux/amd64|arm64. Tokenizer and config files
(`tokenizer.json`, `tokenizer_config.json`, `config.json`, `vocab.json`, ...) at the usual image paths (`/`, `/app`,
`/model`, `/models`, `/opt/model`, `/opt/reax`, `/srv`, `/workspace`, the image's `WorkingDir`) must be byte-identical to
the reference image's file at the same path, else to the pinned weights directory's file of that name; a symlink or
oversized file is a violation, a file with nothing to compare to is a warning. A bare file name matches across
directories only for tokenizer and vocabulary files; `config.json` / `generation_config.json` are matched by name only
inside `/model`, `/models`, `/opt/model` (an engine's own `/app/config.json` is not the model's). If the reference image's
copy and the weights directory's copy of the same file disagree, the validator is misconfigured and the session is void
(no record is written). No file is copied out of an image whose configuration is already rejected, and a copied file is cut
off at 64 MiB. A `rejected` record is final for its digest; to re-check a digest after fixing a validator-side cause, delete
`<review dir>/reviews/<digest>.json` (the next cycle checks it again). The list of paths is a screen, not a proof:
an engine can read any path, which is why the review gate below exists.

**Review gate (release mode only).** A digest runs only after the operator created the empty file
`<review dir>/REVIEWED/<digest>` (64 hex characters, no `sha256:`; a symlink does not count) following the isolated
Claude review of the image and the record. Until then the digest is `awaiting review`: it is logged, not run, not recorded,
no infrastructure attempt is counted, and later cycles leave it out so unreviewed submissions cannot use up
`REAX_ENGINE_MAX_EVALS` slots. Outside release mode the static check runs and rejects, but no marker is needed.

## Quantisation and repacking (25 October 2026 decision)

Effective **27 October 2026 at 14:30 UTC** (`ENGINE_QUANT_RULE_START_UTC`), any re-encoding of the pinned model is
allowed: FP8, INT8, NVFP4/MXFP4, mixed precision, lossless repacks, layout changes, fused or packed matrices, and an
approximate path with an exact recheck. The following conditions apply:

- Every tensor the engine uses must be derived at start inside the container from the read-only pinned `/model`, by
  code shipped in the image whose source is published. No weight or calibration files may ship in the image.
- The pre-run inventory screen checks every non-reference layer, including files overwritten in a later layer. It flags
  `.pt`, `.pth`, `.safetensors`, `.bin`, `.npy`, `.npz`, `.gguf`, `.ckpt` and `.pkl` files, and any non-code file over
  64 MiB. Only the reference's exact tokenizer filenames are exceptions (`tokenizer.json` for the pinned s1-fast manifest;
  no binary tokenizer exception). Source files, shared libraries, CUDA kernels and small configuration formats are code
  for this screen; `.so` and `.cubin` are exempt from the size rule. Source and rebuild review still applies.
- Conversion must be calibration-free, or use only activations computed at start from the public seed set
  (`0x52454158`). Shipped scales and calibration statistics count as weight-like data.
- Start-up, including conversion, has the existing **15 minute** readiness budget; the complete evaluation has a
  **45 minute** budget. The gates stay unchanged: mean distance ≤ 0.10, every item ≤ 1.0, top agreement ≥ 99 %, and
  determinism within 0.05. A private 64-item holdout slice is planned for a separate release; this change does not add it.

Current titles and awards stay valid. Existing entries have three days to submit a compliant version, until
**30 October 2026 at 14:30 UTC** (`ENGINE_QUANT_CONVERSION_END_UTC`). From that slot on, only a digest explicitly in
`TrackState.weights_ok` can take the title, including a challenger confirmed before the deadline. Missing compliance
fails closed. Sandy populates that set from explicit compliance reviews; a copy-line `derived` marker alone does not
establish weight compliance. The gate does not remove a sitting champion or reverse an award.

For an offline inventory check, without starting the image:

```bash
python -m reax_subnet.engreview weights --inventory files-223.json
```

Input is a reviewer JSON path-to-metadata map, a layer-to-path-to-metadata map, a list of file rows, or a per-layer
`inv-N.tsv` list (`sha256`, size, path). Supply all non-reference layers, rather than only a merged filesystem. The
command emits JSON findings with reasons and exits 1 for findings, 0 for a clear screen, or 2 for invalid input.
A clear static screen is not itself a compliance approval: published conversion code and calibration provenance still
need review. These conditions make derivation reproducible instead of trusting a shipped file's claim to be the same model.

A pending challenger is re-measured in later batches and keeps its place only while every measurement stays above every bar; the crown score is the lowest of its confirming measurements.

The weight inventory screen must cover all non-reference layers. Its tokenizer exception is only
`/model/tokenizer.json`, SHA256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`,
at most 1.1 × 19,989,325 bytes. The digest is pinned by `src/reax_subnet/manifests/s1-fast-v1.json`;
the size was verified from the corresponding read-only Plumb-4B revision cache (55de0378).
Large JSON and numeric multipart groups above 64 MiB are screened, along with recognized weight suffixes.
This filename screen still needs source/rebuild review: renamed small data, archive containers,
non-numeric splits, hidden files and tensors embedded in source/native code can evade it.
A clear screen alone does not certify calibration provenance or completeness of the supplied inventory.
