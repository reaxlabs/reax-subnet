# The answer test (`reax-answertest`): admission for EU contract operators

Status: proposal + tool (22 Nov 2026). Replaces the strict bit-identity bar (`reax-selfcheck` PASS at >= 99 % simulated
rounds) for EU contract-tier operators. Code: `src/reax_subnet/answertest.py`, tests `tests/test_answertest.py`.
Nothing in the validator or the EU registry changes with this document.

## 1. What `reax-selfcheck` measures today

It sends 640 public reference decisions (`selfcheck_data/<model>-<manifest>.json`, synthetic text, reference answers from
the release reference engine on an RTX PRO 6000) through the sidecar, at ONE client concurrency (default 1), and computes per
item the validator's logit distance `T * max_i |centred ln p_i - centred ln q_i|` (`s1scoring.logit_distance`). It then
draws 20,000 random rounds of `verified_per_round` items (3 for mammoth-1) and counts the share that would score
(mean <= tau_mean 0.20) and strike (mean > 0.50 or any item > 2.0). PASS = 0 strikes and >= 99 % of rounds scoring.

What that really is: a check that the engine, at that concurrency, stays within a few hundredths of the reference *on the
distance metric*. The 99 % bar is our margin, not a contract term: contract 4.2 (EU-TIER.md) pays an hour at >= 90 %
positive validator rounds, no strike, >= 25 decisions/s. An engine whose batched answers drift a little (TNG at c=8: median
0.066, p99 0.369, max 0.442, nothing near the strike lines, 98.40 % of rounds) fails the self-check yet is far inside what the
validators accept. The self-check also has no sustained load and no probability-space view, which is what customers see.

## 2. What the answer test measures instead

Per model: the same fixed public golden set (no customer data; reference answers from our reference engine, manifest-pinned;
for a newly listed model the set is generated and frozen with the model's manifest, same format `reax-selfcheck/1`).

Ladder `c = 1, 2, 4, 8, 12, 16, 24, 32`. At each `c`, for at least `--duration` seconds (default 60; quick read 15-20; binding
run 120) and at least one full pass over the golden set:

- `c` parallel streams, each sending one decision at a time, cycling over the shuffled golden set;
- plus ONE background validator-like stream: rounds of `items_per_round` decisions sent together (8), `verified_per_round`
  of them scored with the validator's own `score_round` (so the engine always sees a validator burst on top of the `c`
  streams; `c` therefore counts customer-like decisions in flight, and the validator burst is part of the load).

Two cold starts (runtime restarted between the two ladder runs: `--cold-starts 2 --restart-cmd ...`, or two runs combined with
`reax-answertest combine a.json b.json`). A concurrency passes only if it passes in BOTH runs (worst case per metric).

Metrics per `c` (all in `levels[]` of the JSON report):

| metric | meaning |
|---|---|
| `top1_agreement` | share of non-tie items whose argmax equals the reference argmax (non-tie: reference top-2 margin >= 0.02) |
| `max_dp`, `p99_dp`, `median_dp` | max absolute difference of any option probability vs the reference (probability space) |
| `distance_mean/median/p99/max` | validator logit distance to the reference |
| `items_over_tau_item` | items above the validator's strike line 2.0 (must be 0) |
| `round_scores`, `round_strikes` | simulated validator rounds from the c-stream answers (selfcheck's method) |
| `validator_stream` | rounds, scored and struck rounds of the real background stream (`score_round`) |
| `decisions_per_s`, `latency_p50_s/p95_s` | speed under that load |
| `failed` | requests that errored or timed out |

## 3. Admission rule (exact thresholds)

### 3.0 Bar in force: `eu-v1` (23 Nov 2026)

The calibration run of 23 Nov 2026 (board #79492, Verda CC box, s1-pro golden 3f3558e5, 640 items) showed that the strict
bar below admits our own production engine only at one request in flight, below the 25 decisions/s pay floor, and does not
admit the UID 240 engine at all. The drift comes from concurrent batching, not from a different model: top-1 agreement stayed
at 0.983-1.0 and max_dp at 0.086-0.135 for c = 2..16 (up to 54 decisions/s), while substitute models score about 86 % top-1
in our template parity runs. The EU bar therefore measures answer quality, not bit-identity:

| criterion | `eu-v1` | `strict` (below) |
|---|---|---|
| top-1 agreement on non-tie items (reference top-2 margin >= 0.02) | >= 0.98 | 1.0 |
| max_dp (largest option-probability difference on any item) | <= 0.20 | <= 0.02 |
| p99_dp | <= 0.10 | not checked |
| mean logit distance | <= 0.20 (the validator pool gate tau_mean) | <= 0.10 |
| answers over the strike line tau_item | <= 2 % | 0 |
| simulated validator-round strikes | reported, not a criterion | 0 |
| failed requests, coverage, two cold starts | as below | as below |

`reax-answertest` uses `eu-v1` by default (`--bar strict` for the old bar). `reax-answertest combine <reports> --bar eu-v1`
re-judges existing reports against a bar, so earlier runs need no GPU to be re-read. The bar is reviewed after the first
operator runs (TNG) and one quantised-substitute control run (FP8 build of the same model must be reported; if it passes
`eu-v1` its answers are as good as the reference by this measure, which is acceptable for contract operators, who are
also bound by the contract's model and data clauses). P2P pots keep the strict check (validator, unchanged).

### 3.1 The strict bar (our node gate)

A concurrency level PASSES only if ALL hold (`Thresholds` in answertest.py, CLI `--max-dp`, `--min-top1`):

| criterion | threshold | why this value |
|---|---|---|
| `complete` | 0 failed requests | a miner that drops requests at c is not serving c |
| `full_pass` | every golden item answered at least once | no cherry-picked partial run |
| `top1` | top-1 agreement = 1.0 on non-tie items | our nodes' gate (1.0 on 500 items). Near-ties (margin < 0.02) are not counted: a flip there is inside the dp budget below and says nothing about answer quality; they are reported (`near_tie_items`) |
| `max_dp` | <= 0.02 | our nodes' gate: no option probability moves by more than 2 percentage points on any golden item |
| `mean_distance` | mean logit distance <= 0.10 | the engine-competition gate (CALIBRATION.md); half the pool's tau_mean 0.20 so the validators keep headroom |
| `no_strikes` | 0 items > tau_item 2.0, 0 simulated strike rounds, 0 strikes in the real validator stream | the contract's hard line: a strike voids the hour |

Informational, never blocking admission, but printed and stored: `contract.rounds_ok` (simulated rounds >= 90 %, the pay
rule) and `contract.throughput_ok` (>= 25 decisions/s, the pay rule). An operator can be admitted at a concurrency where pay
would still not accrue; the tool says so.

**Admitted concurrency** = highest `c` such that it and every lower tested `c` pass. A pass above a failing lower level is not
admitted (warning in the report): non-monotone behaviour usually means a flaky engine.

Where this differs from our own nodes' gate, and why:
- Same: max_dp 0.02, top-1 1.0, 0 strike-line items, two cold starts.
- Different: (a) our gate is a one-off node admission at the concurrency we then run; this one is a ladder under sustained
  load with a validator burst, because the operator picks the concurrency. (b) near-tie items are excluded from top-1 (our gate
  counts them; on 500 items of our own engine that never mattered). (c) the mean-distance line is explicit.
- **Calibration risk, read this before the first binding run.** CALIBRATION.md measured honest non-reference setups at
  top-1 98.4-99.5 % on all items (H100, RTX 4090 vs RTX PRO 6000 reference path, vLLM batched) and max logit distance up to
  0.7-1.7; the dp implied by a logit distance d at T=1 is at most ~0.25*d at p~0.5, so TNG's c=8 max 0.442 could mean
  max_dp up to ~0.1 and p99 0.369 up to ~0.09 on mid-probability items, which would fail the 0.02 bar. The defaults are therefore the
  CONSERVATIVE proposal (our own bar). The tool is how to decide: run it once on our reference engine (c=1 and c=16),
  once on the dm240 engine on an RTX PRO 6000 (ours), then on TNG's, and set `--max-dp` / `--min-top1` from the measured spread
  (candidates: max_dp 0.05 and top-1 >= 0.998 on non-tie items). That is a decision for Florian/lead; the numbers do not
  exist yet. Any engine passes if its answers do, including vLLM + plugin or SGLang.

## 4. Re-test triggers

The report stores the identity of what was tested: model, manifest id, `engine`, `engine_image_digest`, `gpu_name`, `driver`,
tool version, timestamp. A new binding test is needed when any of these changes:
1. engine image digest (or engine/plugin version) or the model weights/manifest;
2. GPU model (or a mixed-GPU box: each model of GPU is tested);
3. NVIDIA driver MAJOR version (or CUDA major in the image);
4. the operator wants more concurrency than `admitted_c` (a test only at/above the new `c`, plus the cold-start pair);
5. the continuous check below steps the concurrency down twice in 7 days;
6. our golden set or thresholds change (new tool major version).
Within a validity period of 90 days without trigger the result stands; the continuous check is what catches drift in between.

## 5. Continuous check (proposal, not built yet)

Reuses the router's shadow mechanism (ROUTER.md): a share of real answered requests (proposal 2 %; today `REAX_ROUTER_SHADOW_RATE`
default 0.05) is recomputed on OUR reference backend (inside the EU for EU traffic; the request already left to the operator
under the DPA, the reference run adds no new recipient). Per (operator, model) keep a rolling window of the last 500 shadowed
decisions / 6 h: `max_dp`, `p99_dp`, top-1 flips on non-tie items, items over tau_item.

- Step-down: any item over tau_item (existing strike path), or any non-tie top-1 flip with dp > 0.05, or `p99_dp` > 0.04 in the window
  -> the router lowers that operator's in-flight cap for the model to the next lower ladder rung (the cap exists today:
  `REAX_ROUTER_INFLIGHT_CAP`, per hotkey, and `max_inflight` per worker in multinode) for 24 h and flags a re-test; two step-downs in
  7 days = trigger 5.
- Raising the cap again only through a new answer-test run.
- Stored per check: request id, model, operator, dp, top-1 match, margin, timestamp. NO request text, no answer text. The recomputed
  answer is discarded after comparison. Aggregates only in reports.
- Failures of our reference backend never penalise the operator (as today).

## 6. Who runs it

- Operator self-run (quick read, any time): `reax-answertest --model s1-pro --url ... --token-file ... --duration 20 --ladder 1,2,4,8,16`.
- Binding run: by us, against the operator's endpoint with the golden set only (no customer traffic), duration 120 s, two cold
  starts (operator restarts on our request, or runs `--restart-cmd`), result stored with the contract file; `admitted_c`
  becomes the operator's concurrency in the registry/router configuration. Runs of different operators can be parallel.
- The golden set is public; an operator could special-case it. The answer test is an admission screen, not a proof: the validators'
  scoring on private items and the continuous check on real requests remain the enforcement (same stance as `reax-selfcheck`).

## 7. Report and usage

```
reax-answertest --model s1-pro --url http://127.0.0.1:8112 --token-file ~/.reax/token \
    --engine vllm --image-digest sha256:... --gpu-name auto --driver 580.xx --duration 120 \
    --cold-starts 2 --restart-cmd "docker restart reax-runtime" --out answertest.json
reax-answertest combine run1.json run2.json --json     # separate cold starts
```

JSON (`reax-answertest/1`): `model, manifest_id, weights_revision, engine, engine_image_digest, gpu_name, driver, golden_items,
level_seconds, cold_starts, timestamp, tool_version, thresholds, levels[] (metrics above + criteria{} + pass + contract{}),
admitted_c, pass, warnings`. Exit code 0 = some concurrency admitted, 1 = none, 2 = cannot run. The human summary is a table per `c`
with the failing criteria named, and a line `ADMITTED concurrency: N in flight per GPU`.
