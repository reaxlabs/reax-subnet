# REAX engine track v2: one engine, every model we serve (design, 20 Oct 2026; REAX 1 slot added 21 Oct; v2.1, 22 Oct; v2.2, 25 Oct; v2.0 lineup, 25 Oct)

Florian, 20 Oct 18:55Z: "We should not only measure s1-fast but all the models we run on decisionmodels.io in the
REAX engine, and build a composite score from that. What we really want is an engine that is faster on all our models."
Florian, 19:10Z (#22407): ship v2 as soon as it is green (tests + Claude critic); about 1 h of #92 notice is enough.
Florian, 21 Oct ~11:10Z: REAX 1 goes into the engine competition track "with especially high weight". v2 is not open yet,
so REAX 1 joins the **first** v2 weight set (five slots) instead of a mid-track change.
Florian, 22 Oct ~20:1xZ (standing, DECISIONS `engine-serves-lineup-principle`): the engine track optimises exactly the models
we serve on decisionmodels.io, weights follow what we serve, and every approved engine gain ships to decisionmodels.io promptly.
**v2.1 (22 Oct):** default model list d3 0.50, blink 0.25, reax1 0.20, pplx 0.05, **no plumb slot** (pending Florian's pick on
card 18258; `reax-high` and `approved-1021` stay selectable),
slot `d3` serves `d3-turbo`; the 21 Oct set stays selectable (`engcomposite.COMPOSITE_WEIGHTS_VARIANT`, section 1); optional
v1 payout bridge (section 6); margins in section 7.

**v2.0 lineup (25 Oct, evening; board #22 #26507, #26739, #26768):**
the scored models are the ones decisionmodels.io serves after its switch. Default weights `v20` (Florian's pick 26 Oct, card 18618 "Floor"; board #22
#27151): **mammoth 0.37, blink 0.33, reax1 0.10, jeffan 0.10, vision 0.10**: a 10 % floor for every served model, the remaining
0.70 split between the two main engines by GPU-cost share since 23 Oct (50.7 : 44.9). Quyet, pplx and d3 are out of the default (their variants stay selectable). Section 10 has the
slot table, the weights and the fallback rule. The v2.2 text below is kept as history.

**v2.2 (25 Oct, earlier that day):** Florian decided that the engine covers **every model we serve on decisionmodels.io**, including
Quyet-1.0-Large and s1-vision (DECISIONS `engine-v2-all-models-tue-switch-fri-crown`, board #22 #26392). Six slots, weights
`v22` (no longer the default): d3 0.45, blink 0.20, vision 0.05, reax1 0.20, pplx 0.05, quyet 0.05. v2 becomes binding right after the Tue 27 Oct
14:30Z crowning slot (the last v1 crowning); the first v2 crowning is Fri 30 Oct 14:30Z. Section 10 has the slots, the image
items, the calendar and the open calibration. The rc11.post6 engine rules (Tue/Fri slots, weight-file screen, parent rule)
apply to `dm-all`.

Status (v2.2c): design + code on branch `fusion/engine-composite-track-20261020` of `reaxlabs/reax-subnet-p2p-wip`
(base rc10 52c8385). The release owner sn92-fast-launch-20261015 integrates; nothing here is deployed.
The measured reference numbers are in section 7 (RunPod RTX PRO 6000, 20 Oct evening).

**Short version.** v2 is coded and tested (Claude critic rounds, blockers fixed). The five `v20` slots are in the code. The track
cannot open until the digests of `mammoth` and `jeffan` are published, the GPU check has calibrated `mammoth`, `jeffan` and `vision`
(release guard, section 10) and the frozen v1 state is pinned after the Tue 27 Oct slot. The calendar is in section 10: v2 starts right after the Tue 27 Oct 14:30Z slot, the first
`dm-all` crowning is Fri 30 Oct 14:30Z. (Earlier drafts of this page named Wed 21 Oct as the target; that date is obsolete.)
Historic notes: d3's reference throughput moves ~4 % from one process start to the next (margin 7 % instead of 3 %, the d3 floor
allows a 14 % regression), so d3's compile/autotune choices are pinned (the l2norm lesson).

## 1. What is measured

| slot | public model on decisionmodels.io | weights (frozen) | engine | share of requests (7 d to 22 Oct, gateway logs) | weight v21 (reax-high / approved-1021) |
|---|---|---|---|---|---|
| `reax1` | `reax-1` (alias of the current REAX 1 release; our own flagship) | `decisionmodels/reax-1@ca2a0cc5` (public, Apache-2.0, 27B; FP8 serving) | REAX native, v1.2 joint path (joint protocol v3; same backbone as pplx v1.1) | 0.1 % (7 d) | **0.20** (0.40) |
| `d3` | s1-pro → s1-pro-3 | v2.1: `vllm-sr/d3-turbo@cfa0d41d…` (model id `d3-turbo`, `engcomposite.D3_SLOT_MODEL`; `d3@5ce995d` = roll-back value; FP8 serving) | REAX native (Qwen3.5 hybrid; d3-turbo: prefix_bidir + state-once) | 49 % (7 d), 76 % of GPU time | **0.50** (0.35 / 0.30) |
| `blink` | s1-fast → s1-fast-2, s1-vision-2 | `PixilabAI/Blink-v0.3-26B-A4B-NVFP4@08caad20` | vLLM 0.30, production preset `ref-prod` (batch-invariant, util 0.45) | 46 % (7 d, incl. vision), 12 % of GPU time | **0.25** (0.20 / 0.15) |
| `pplx` | s1-pro-2 (pinnable until ≥ 11 Nov) | `perplexity-ai/pplx-decider-v1.1-27b@6195aa55` (FP8) | REAX native | 4.8 % (7 d), 11 % of GPU time, falling | **0.05** (0.05 / 0.10) |
| `plumb` | legacy s1-fast-1; the SN92 `s1-fast` serving pool; the v1 track model | `crh225/plumb-4b@55de037` (bf16) | REAX native, exact mode (the v1 reference `9bd41632`) | ~0 % on the API | **removed** (0.05, only in the approved-1021 variant) |

Since v2.2 (25 Oct) `quyet-1-large` and `s1-vision` are slots too (`quyet`, `vision`, section 10). Not measured:
`s1-llm-auto-router` (CPU ONNX classifier, 14 ms, no GPU engine to speed up); the CPU tiers s1-lite / s1-cpu (llama.cpp, no
competition hardware). A new model later = a new slot = new weights = a new release.

Vision: `s1-vision` runs in the same Blink process as `s1-fast`. Until v2.1 only text items were measured. v2.2 adds the
slot `vision`: the Blink engine and weights, every timed item with exactly one picture (section 10).

**Weights.** First proposal (20 Oct): half equal share, half GPU-time share (d3 0.50, pplx 0.20, Blink 0.20, Plumb 0.10),
because what a faster engine saves us is GPU time and that is dominated by d3 (requests × latency in the 30 h log: d3 ~84 %,
pplx ~13 %, Blink ~3 %); pure GPU-time weights would make v2 a d3-only contest.
**Decision of 21 Oct (Florian): REAX 1 gets an especially high weight.** Set: **REAX 1 0.40, d3 0.30, Blink 0.15, pplx 0.10,
Plumb 0.05.** Rationale:
- REAX 1 is our own flagship model, so an engine that is faster on it is what we most want to buy. It is the largest single
  weight, but d3 (today's s1-pro traffic, ~82 % of requests) stays at 0.30, so the composite still tracks production GPU time.
- REAX 1 will be replaced by newer REAX versions over time. A newer version is a new slot with new weights, i.e. a new release
  (`reax1` is a release constant like every other slot). The engine work carries over: REAX 1 has the same backbone and the
  same joint request path as pplx (v1.1 native engine, v1.2 joint path), so a gain on one is largely a gain on the other.
- pplx v1.1 (s1-pro-2) goes from 0.20 to 0.10 and Plumb from 0.10 to 0.05; Plumb keeps a non-zero weight for v1 continuity
  (a v1 `s1-fast` entry is a Plumb-only entry, section 6).
**v2.1 (22 Oct): weights follow what we serve.** Served traffic, 7 days to 22 Oct (gateway request logs; summed latency as the
GPU-time proxy): d3 49 % of requests / 76 % of GPU time, Blink 46 % / 12 %, pplx 4.8 % / 11 % (24 h: 1.5 % / 2 %), REAX 1 0.1 % /
0.1 %, Plumb 0 / 0. (The "~82 % d3" of 20 Oct came from a 17-18 Oct window dominated by one batch account.) d3, pplx and REAX 1 share
the 27B Qwen3.5-hybrid backbone, so kernel gains carry over between them; Blink (MoE NVFP4 on vLLM) is the other stack.
Default `v21`: 27B family 0.75 (≈ its 77-88 % GPU-time share) with d3 0.50 (the s1-pro engine, burst nodes are d3-only),
REAX 1 0.20 (our flagship, forward-looking), pplx 0.05 (pinned until 11 Nov, falling); Blink 0.25 (most requests, shares every
GPU). Plumb 0: not a product, a faster Plumb saves nothing.

**Weight variants (v2.0 to v2.2, `engcomposite.COMPOSITE_WEIGHTS`, one table each, asserted to sum to 1, part of `composite_rules_sha256`):**

| variant | mammoth | reax1 | d3 | blink | pplx | plumb | vision | quyet | jeffan | v1 `s1-fast` commitment | slot order (fail-fast) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v20` (**default**, `COMPOSITE_WEIGHTS_VARIANT`, weights 26 Oct) | 0.37 | 0.10 | not a slot | 0.33 | not a slot | not a slot | 0.10 | not a slot | 0.10 | not carried (no v2 entry) | blink, vision, reax1, jeffan, mammoth |
| `v20-d3` (fallback, section 10) | not a slot | 0.10 | 0.37 | 0.33 | not a slot | not a slot | 0.10 | not a slot | 0.10 | not carried | blink, vision, reax1, jeffan, d3 |
| `v22` (25 Oct, earlier) | not a slot | 0.20 | 0.45 | 0.20 | 0.05 | not a slot | 0.05 | 0.05 | not a slot | not carried (no v2 entry) | blink, vision, quyet, reax1, pplx, d3 |
| `v21` (22 Oct) | not a slot | 0.20 | 0.50 | 0.25 | 0.05 | not a slot | not a slot | not a slot | not a slot | not carried (no v2 entry) | blink, reax1, pplx, d3 |
| `reax-high` (REAX 1 stays on top) | not a slot | 0.40 | 0.35 | 0.20 | 0.05 | not a slot | not a slot | not a slot | not a slot | not carried | blink, reax1, pplx, d3 |
| `approved-1021` (21 Oct set) | not a slot | 0.40 | 0.30 | 0.15 | 0.10 | 0.05 | not a slot | not a slot | not a slot | Plumb-only entry | plumb, blink, reax1, pplx, d3 |

d3 keeps the provisional calibration numbers of model `d3` (d3-turbo measured the same speed and +0.8 GiB memory); the slot
stays uncalibrated until the GPU check. Release constants; recomputed (new release) when the traffic mix changes a lot.

**GPU.** One NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), the only published class (FLA configs and noise
are measured on it). Blink is NVFP4 and needs Blackwell anyway. The models are measured **one at a time** (each
model = its own fresh session); inside a session the v1 layout stays: reference and candidate loaded at once, the
idle one paused. Two engines of every model fit (measured: Plumb 2 × 15 GiB, d3 2 × 35 GiB, pplx 2 × 34.7 GiB, REAX 1 2 × 34.5 GiB,
Blink 2 × 41 GiB with `gpu_memory_utilization` 0.45 each; production Blink at 0.9 takes 83.5 GiB alone, so the Blink
reference preset for the competition must use 0.45). Validators with smaller GPUs cannot
run v2; v2 runs where the engine track already runs today (the competition worker on an ephemeral RTX PRO 6000
candidate box).

## 2. How an image declares support

- One image for all models, pinned by digest, committed exactly as in v1 with the new track name:
  `reax-eng/1:dm-all:<image>@sha256:<digest>` (the 128-byte commitment format and the parser are unchanged).
- The engine learns its model from `REAX_MODEL=<model id>` (`plumb-4b`, `blink-v0.3-26b-a4b`,
  `reax-1`, `pplx-decider-v1.1-27b`, `d3`) and reads that model's frozen weights read-only at `/model`; everything else is the
  v1 contract (UDS socket, token file, `/internal/v1/decide`, `/health`, network none, read-only, runsc).
- Label `ai.reax.engine.models=reax1,d3` (or `all`) names the models the image serves. **No label = all models.**
  An unknown name or an empty entry fails the submission (a typo must not silently drop a model).
- **A model the image does not declare counts as exactly 1.0x and is not measured** (we would keep the reference
  for that model). Proposed instead of disqualification because (a) a real gain on one model is a real gain for us,
  (b) it lets entrants start with one model and grow, (c) the geometric mean already prices breadth: a REAX 1-only
  engine needs +6.1 % on d3 for a +3 % composite in v21 (+10.4 % in approved-1021), a Blink-only engine +12.6 % (+21.8 %), a REAX 1-only engine +15.9 % (+7.7 %).
- **A model the image declares must pass every gate on that model**; one failure disqualifies the whole image
  (otherwise an image could try a lossy shortcut on one model for free).

## 3. Composite score

Per declared model *i* (code `engcomposite.py`): the unchanged v1 session (`engrunner.evaluate`) against that model's
reference — fresh secret items, reference self-check (VOID on failure), solo baseline, 2 ABBA pairs × 256 items,
distance + top-choice gates on every timed answer, anti-cache probes, determinism re-run, latency gate, reference drift,
idle engine paused. Result s_i = median paired throughput ratio.

Additional per-model gates:
- **No-regression floor:** s_i ≥ 1 − max(2 %, 3 σ_i) (σ_i = that model's fresh-process reference noise). This is the
  rule that stops "win on one model, regress on another": a 0.90x model fails the image even if d3 is 1.5x.
- **GPU memory budget:** the GPU memory the candidate adds next to the loaded reference must stay within the
  model's budget (proposed: reference footprint × ~1.3, from section 7). Measured device-level with `nvidia-smi`
  (works under runsc): a stable reading before the candidate starts, then the **peak** sampled every second during
  its whole measurement (an engine that frees memory after its last timed request is still charged its peak).
  A failed reading is never a verdict (inconclusive, retried). This needs a **dedicated GPU** (the candidate box runs
  no serving runtimes), otherwise foreign allocations would be charged to the candidate. A faster engine that needs
  twice the memory fits fewer replicas per card and is not a production win. Limit: an engine could size its pools
  from observed free memory and use more on a production card; the reviewer rule is "fixed memory caps, no sizing from
  free memory" (engines get `REAX_GPU_MEMORY_BUDGET_MIB` from the runner config; to wire in R4).

Composite S = exp(Σ w_i ln s_i) over all slots, undeclared s_i = 1. Option (`CompositeTrack.required`, off by
default): slots every image must declare, e.g. `("d3",)` if Florian wants to force the 82 % model. Valid if S ≥ 1 + max(3 %, 3 σ_S) with
σ_S = sqrt(Σ w_i² σ_i²) (all slots, the conservative release constant).

**Champion rule: unchanged v1 code.** The composite becomes one `EvalResult` per digest (with a per-model breakdown in
`parts`), so `engcomp.update_champion` applies as is: 3 % margin over the champion (or `3 σ_S` if larger), a second
fresh session must confirm (it re-measures every declared model), the title is held at the lower session score, the
champion is re-measured whenever it is challenged, two invalid sessions vacate, earliest commitment wins ties,
image-to-hotkey label binding, review gate, owner-signed award approval, winner-take-most with decay.

**Frozen rules.** Per-model thresholds, weights, budgets, floor and memory budgets are hashed
(`composite_rules_sha256`) into the validator's `thresholds_sha256` and the state file, as v1 does with
`engthresholds.py`. v1's `FROZEN_SHA256` is unchanged.

**Release guard.** `CompositeTrack.release_ready()` lists every slot that is not calibrated on the release
reference image; the track must not open while it is non-empty (sn92 wires this into the start constant check).

## 4. Reference images and pinned kernels (lesson from the l2norm autotune issue)

- **One multi-model reference image** (`reax-engine-all`), pinned by digest, built from reax-engine main with the
  production presets: d3/pplx native FP8, Blink `bi1`, Plumb exact mode. Today the code lives in two image families
  (`docker/qwen35` torch stack, `docker/vllm-readout` vLLM 0.30 stack); unifying them is the main open build item
  (R1, section 8). **Implemented path that needs no unified image:** one reference image per model, release
  constants `config.ENGINE_COMPOSITE_REFERENCE_IMAGES` (Plumb = today's v1 reference `9bd41632`; Blink = the
  `docker/vllm-readout` image with `bi1`, pplx and d3 = the `docker/qwen35` image, digests to publish), part of the
  measurement context (a new reference re-measures everything). Release mode refuses a missing or unpinned slot.
  The candidate is still one image. A unified image later = the same digest in every slot.
- **Pinned autotune.** d3 and pplx use the same Gated-DeltaNet FLA kernels as Plumb (with other shapes), so the
  l2norm lesson applies to them directly: FLA configs generated on the published GPU class for every model's shapes,
  merged into the one `fla-configs/<gpu class>/` directory (strict mode is an exact key match per kernel and shape,
  so one directory serves all models), sha256-pinned (`engconfigs.PINNED_CONFIG_MANIFEST_SHA256`), mounted read-only
  into reference and candidate. Blink's vLLM path: `bi1` = batch-invariant kernels with fixed tiles; Triton/FlashInfer
  autotune must be off or pinned in the reference preset (to verify in R2).
- Every session starts with the v1 reference self-check on that model (same seeded set twice, distance gates);
  failure = VOID (no score, no strike). The fresh-process noise of section 7 sets σ_i.

## 5. Budget, time and cost per evaluation

Per model: two engine starts (reference, candidate), self-check 2 × 256, solo baseline 256 + latency 24, warm-up,
2 ABBA pairs (4 × 256), probes 32, determinism 32, latency 2 × 24. Section 7 gives measured start times and
throughputs; the per-model budgets in `default_track()` are provisional (Plumb 20 min, Blink 20 min, REAX 1 30 min,
pplx 30 min, d3 30 min, + 15 min slack = 2 h 25 min hard cap in approved-1021, 2 h 5 min in v21 without Plumb: these are the v2.1
numbers; **v2.2 had six slots and a hard cap of 2 h 50 min (10 200 s); the v2.0 lineup has five slots and 2 h 41 min (9 660 s), see section 10**; a typical full evaluation is expected well below that, see
section 7). Each model has its own budget: a slow pull/start/load is inconclusive (retried, at most 3 attempts per
digest), a slow measurement fails; the overall cap is ours (inconclusive). Fail-fast order (v2.0 lineup): blink → vision → reax1 → jeffan → mammoth (v2.2: blink → vision → quyet → reax1 → pplx → d3)
(cheapest first; v2.1 had (Plumb →) Blink → REAX 1 → pplx → d3, Plumb only in approved-1021): a broken or cheating image usually dies on Plumb in minutes. At most **2 new submissions per
6 h cycle** (`max_evals`; a full composite is up to ~2 h of GPU), plus confirmation and champion re-measurement.

Measured inputs (section 7): every engine container starts with empty caches, so cold starts count — d3 ~8 min,
pplx ~4 min, Blink ~3.5 min, Plumb 40 s, twice per model (reference + candidate). Expected wall time of one full
evaluation ≈ Plumb 3 + Blink 8 + pplx 15 + d3 21 ≈ **47 min** (cap 1 h 55 min). Cost on one Verda/RunPod
RTX PRO 6000 (USD 2.40-2.50/h): **about USD 2 per full evaluation**, about USD 4-6 for a crowned challenger
(confirmation session + the champion's re-measurement). The weights of the five
models (~180 GB) live on the candidate box's volume (download once per box, ~15-25 min, or a prepared volume).

## 6. Migration from v1

- v1 (`s1-fast`, opened 20 Oct 18:00Z) runs unchanged until v2 opens. v2 is a new track name in the same engine
  share; v1's last crowning is the Tue 27 Oct 14:30Z slot and `dm-all` starts right after it (section 10). v1 does not run in parallel
  (the share is not split); its title holder and former champions keep their rights through the payout bridge and the carried places (below).
- **v2.1 default (`v21`): v1 `s1-fast` commitments are not carried** (`carried = ()`, Plumb is no slot): a v1 image has no v2
  entry and a miner commits to `dm-all` anew. **Variant `approved-1021`: every v1 entry carries over as a v2 entry, not as the
  v2 champion** (implemented: `CompositeTrack.carried`, `s1-fast → plumb`). A `reax-eng/1:s1-fast:…` commitment is read as a Plumb-only `dm-all` entry: no new
  commitment, same commitment block (seniority kept), measured like everyone else, the other models count 1.0x, so
  its composite is s_plumb^0.05 (a v1 engine needs ≥ 1.81x on Plumb to clear the 3 % gain, as Plumb is now the 0.05 slot; today's best v1 entries
  are in that range). A miner who wants the other models re-commits to `dm-all` with a new image.
  Not seeded as champion: a v1 score is a ratio on another rule set; carrying it would pay a number that no v2
  session measured (the v1 state file's `context_changed` logic exists exactly to prevent that).
- **Payout bridge (25 Oct, replaces the earlier "stops paying" rule):** see section 10, "Switching from v1 to v2". The bridge
  keeps paying the v1 title holder from the frozen v1 state until the first `dm-all` crowning (at most until
  `ENGINE_V1_BRIDGE_END_UTC` = Tue 3 Nov 14:30Z); when it ends, the holder moves to the 17 % former-champion place as if beaten.
- Commitments made before the opening count at the opening block (v1's `ENGINE_PRE_OPEN_AS_START` + opening-block
  hash tie order), so a short notice is fair: nobody gains by watching the clock.
- Announcement (we-form, #92, draft in `cards/92-engine-v2-notice.md`): before the switch; the dates are the calendar of section 10
  (Tue 27 Oct 14:30Z last v1 crowning, Fri 30 Oct 14:30Z first `dm-all` crowning). The earlier Wed 21 Oct target is obsolete.

## 7. Measured reference (RunPod SECURE RTX PRO 6000 Server Edition, 20 Oct 2026; REAX 1: 21 Oct 2026)

reax-engine main 69b6dc5 run natively on the pod (torch 2.14 cu130, transformers 5.17, vLLM 0.30 venv for Blink),
production presets, d3 at the production revision 5ce995df. 2 × 256 synthetic items per level (reax-subnet generator,
fixed seeds), 4 fresh process starts per model, validator-side clock. Raw responses and logs:
`jobs/engine-composite-track-20261020/calib/results/`.
REAX 1 (21 Oct) was measured the same way, with the same scripts and item files, on reax-engine f879674 (the v1.2 joint path:
`engine/server.py --linear fp8 --compile 1 --gva 1 --warm-steps 120 --max-length 8192 --v12-split joint --v12-overlength reject`,
served through `/v1/systemone`, see R6): `jobs/reax1-everywhere-20261021/workers/engine-reax1/calib-reax1/`.

| model | GPU mem 1× / 2× GiB | start warm / cold s | dec/s c1 / c8 / c16 | fresh-process noise c8 / c16 | ref vs ref across processes (slot conc.): mean / max distance, top choice | c1 vs batched, same process |
|---|---|---|---|---|---|---|
| REAX 1 (FP8, v1.2 joint) | 34.5 / 69.0 | 171 / 353 | 9 / 49 / 58 | 3.8 % / 6.9 % | 0.056 / 0.42, ≥ 99.2 % (c8); at c16 0.13 / 1.29, ≥ 96.5 % | max 0.70, mean 0.14 (c1 vs c16); c1 bit-identical across processes |
| Plumb (exact) | 14.9 / 29.8 | 38 / 39 | 64 / 99 / 103 | 0.4 % / 0.7 % | 0 / 0, 100 % (c16) | bit-identical |
| Blink (vLLM bi1, util 0.45) | 40.7 / 81.9 | 74 / 210 | 33 / 77 / 109 | 8.6 % / 1.4 % | 0.006 / 0.33, ≥ 99.6 % (c16) | bit-identical (1 of 4 starts moved 6/256 items at c1) |
| pplx v1.1 (FP8) | 33.9 / 67.7 | 175 / 248 | 12 / 48 / 60 | 1.8 % / 7.6 % | 0.089 / 0.53, ≥ 98.05 % (c8) | max 0.44 (c1 vs c16) |
| d3 (FP8) | 34.5 / 69.0 | 177 / 461 | 12 / 48 / 58 | 4.6 % / 9.0 % | 0.043 / 0.24, ≥ 99.6 % (c8) | max 0.30 (c1 vs c16) |

What follows from it (all in `engcomposite._DEFAULT_SLOTS`):
- **The v1 gates do not fit d3/pplx.** d3 and pplx FP8 are not batch-invariant: an exact copy of our own pplx engine
  would fail v1's 99 % top-choice gate (98.05 %) and come close to its mean-distance cap (0.089 vs 0.10). REAX 1 is the same
  kind of engine (c16 across processes: top choice down to 96.5 %).
  Per-model gates are set to the honest worst case with ~1.5-2× headroom: pplx mean 0.18 / item 1.0 / top 96 % /
  determinism 0.9; d3 mean 0.10 / item 1.0 / top 98 % / determinism 0.6; REAX 1 mean 0.12 / item 1.0 / top 98 % /
  determinism 1.4 (worst c1-vs-c16 item 0.70, doubled like pplx and d3; the c8 slot concurrency measured 0.056 / 0.42 / 99.2 %); Blink top 98 %; Plumb keeps v1's gates.
  Looser gates are a bigger "quality budget" for shortcuts on those two models; the mean gate is the main defence
  there. Making d3/pplx/REAX 1 batch-invariant (as Blink bi1 is) is the clean long-term fix.
- **Concurrency per model = the quieter level:** REAX 1, d3 and pplx c8 (their production slot count), Blink and Plumb c16.
  REAX 1/d3/pplx use 1024 timed items per run (a 256-item run takes only ~5 s at 48 dec/s).
- **Noise units:** REAX 1 3.8 %, d3 4.6 %, pplx 1.8 %, Blink 1.4 %, Plumb 0.7 % → composite σ_S = 2.06 % in `approved-1021` (weights 0.40 / 0.30 /
  0.10 / 0.15 / 0.05) → the margin is max(3 %, 3 σ_S) = **6.19 %**; **2.44 % / 7.31 % in `v21`** (d3 0.50 / blink 0.25 / reax1 0.20 / pplx 0.05; 2.22 % / 6.67 % in `reax-high`), and the no-regression floors are 1 − 3 × 3.8 % = 0.89
  (REAX 1) and 1 − 3 × 4.6 % = 0.86 (d3). The noise of both 27B FP8 models is mostly between process starts (REAX 1: std of
  process means 2.6 % at c8, d3 4.1 %), so more items do not help; pinning their torch.compile / FP8 GEMM autotune choices
  (like the FLA configs for Plumb) should bring both near pplx's 1.8 % → margin back to ~3 %, floors ~0.95. This is item R2b.
- **Memory budgets** (candidate next to the loaded reference, peak): Plumb 20 GiB, Blink 48 GiB, REAX 1 45 GiB, pplx 45 GiB,
  d3 45 GiB (≈ 1.3 × the reference, inside what is left on 96 GB).
- Pausing the idle engine (SIGSTOP) did not change the active engine's throughput for any model (within noise).
- Spend: USD 6.24 (20 Oct; includes USD 2.11 for two RunPod pods whose containers never started; deleted) plus USD 1.62 for
  the REAX 1 pod (21 Oct, 0.65 h at 2.49/h, deleted).

## 8. What must be green before opening (owner: sn92 release lane unless noted)

| # | item | state |
|---|---|---|
| C1 | composite scoring + session loop + runner + validator hook, tests | done on the branch (16 new tests; engine suites 63/63; rest of the suite green) |
| C2 | Claude critic on the branch (security + rules) | see section 9 |
| R1 | reference images for blink / reax1 / pplx / d3 built from reax-engine `docker/vllm-readout` and `docker/qwen35` (entrypoints read `REAX_MODEL`; qwen35's `python -m engine.serve` needs a `--model $REAX_MODEL` default), published by digest, set in `config.ENGINE_COMPOSITE_REFERENCE_IMAGES` | open (reax-engine CI builds only s1-fast today; a unified image is optional). REAX 1 is an additional image: the v1.2 path (PR #3) is in the `engine/server.py` family, with no registry entry for `reax-1` (see R6) |
| R2 | GPU check of R1 on the published class, in the real containers under runsc: per model answers equal to section 7 on a fixed set, fresh-process repeatability, FLA configs for all shapes generated + pinned, `bi1` autotune pinned | open |
| R2b | pin d3's, REAX 1's (and pplx's) compile/FP8-GEMM autotune choices so the process-to-process throughput noise drops from 4.6 % / 3.8 % toward ~1.5 % (margin 6.2 % → 3 %) | open; recommended before opening, not a hard blocker (the rules stay fair at 6.2 %, just harder) |
| R3 | re-measure section 7 on the R1 images in containers (noise σ_i, gates, memory, start times), update `_DEFAULT_SLOTS`, set `CALIBRATED_ON_RELEASE_REFERENCES = True` → `release_ready()` empty | numbers from the native run are in the code; the flag stays False |
| R4 | candidate box: 5 weight sets (~180 GB; REAX 1 adds ~49 GB) on the volume, runner config (one DockerRunner per model, `model_env`), worker image with the composite | open (sn92) |
| R5 | #92 notice ~1 h before; reax.co/competitions page line | drafts in `cards/` |
| R6 | REAX 1 under the competition contract: the v1.2 joint path lives in `engine/server.py` (`/v1/systemone`, TCP, no token). The v2 contract is `engine/serve.py` (`/internal/v1/decide`, UDS, `X-S1M-Internal-Token`) with a registry entry per `REAX_MODEL`; `reax-1` has none, and `Qwen35Backend` has no joint v3 readout. Needs a registry entry + a joint-protocol backend (or an adapter in front of the v1.2 server) in reax-engine, then the 21 Oct numbers are re-measured on that image under R3 | open (reax-engine owner; the measured numbers are for the engine core and carry over, HTTP/UDS framing is outside the timed GPU path but not identical) |

## 9. Critic (Claude Sonnet 5.5, two rounds)

- Round 1 (`workers/critic1/RESULT.md`): PASS WITH CONDITIONS. Blocker C1: a `dm-all` champion was never paid
  (weights, award ids, worker + remote validator wiring). Majors: C2 memory reading turned our own nvidia-smi trouble
  into a candidate failure; C3 a slow later model was a final failure; C4 cost/starvation; C5 memory only sampled once.
  All fixed in f5ac4d3 (+ 3 failing tests from the critic, now passing). C9 (breadth): kept "undeclared = 1.0" as the
  default and added the `required` option instead (decision for Florian, see the GO card).
- Round 2 on the fix delta (`workers/critic2/RESULT.md`): PASS WITH CONDITIONS; C1-C3 confirmed fixed in the code.
  M2-M4 fixed in e8431e7; M1 (shared GPU) is covered by the dedicated-GPU rule above; M5 (dev calibration file must
  match on worker and validator) is a dev-network note.

## 10. v2.0 lineup (25 Oct 2026): five slots, image items, calendar

Plain English, short sentences. Release constants live in `engcomposite.py` (slots, weights, order), `s1contract.py` (the
decision contract) and `s1vision.py` (the image generator).

**The lineup.** The scored models are the ones decisionmodels.io serves after its switch. v2 becomes binding right after the
Tue 27 Oct 14:30Z slot. The first v2 crowning is Fri 30 Oct, the v1 bridge ends Tue 3 Nov at the latest. Quyet, pplx and d3 leave the default
list.

**The five slots (variant `v20`, the default).**

| slot | public model on decisionmodels.io | model, revision, licence | engine path | contract pool and limits | weight |
|---|---|---|---|---|---|
| `mammoth` | `s1-pro` (replaces d3-turbo) | `baselabs/mammoth-1@61cec44c281c693992e3d6ebeb44dd3402dfa186`, MIT | REAX Qwen3.5 reference (`ref-qwen35`, `REAX_MODEL=mammoth-1`), state-once production recipe; 35.2 GB peak alone | `s1-pro`: 64 options, one question, text only (the s1-pro public limits, same as d3 today) | **0.37** |
| `blink` | `s1-fast` (Blink text) | `PixilabAI/Blink-v0.3-26B-A4B-NVFP4@08caad20` | vLLM 0.30, preset `ref-prod` | `s1-fast`: 16 options | **0.33** |
| `reax1` | `reax-1` (REAX 1) | `decisionmodels/reax-1@ca2a0cc5`, Apache-2.0 | REAX native, v1.2 joint path | `s1-pro` | **0.10** |
| `jeffan` | `jeffan-31b` | `tattrongvu/Jeffan-31b-1710@cebfa86c2bed2b5c83b12e44de4374b20c87fd79`, Apache-2.0 | vLLM readout reference (`ref-jeffan`, `REAX_MODEL=jeffan-31b-1710`), FP8 weights and FP8 KV, util 0.44, batch invariant; 42.8 GB per engine (pair 83.3 GB peak) | `jeffan-31b`: one question, 2 to 26 options, text only (see "Jeffan's contract") | **0.10** |
| `vision` | `s1-vision` (Blink plus one image) | the same Blink weights as `blink` | the same Blink engine; every timed item carries exactly one picture | `s1-vision`: 26 options, exactly one image | **0.10** |

**Why these weights.** Florian picked the "Floor" option on 26 Oct (card 18618): every served model gets a 10 % floor so that each
one matters (reax1, jeffan, vision sit at the floor); the remaining 0.70 goes to the two main engines in proportion to their
GPU-cost share since 23 Oct (mammoth/s1-pro 50.7, Blink/s1-fast 44.9), giving mammoth 0.37 and blink 0.33. mammoth inherits the
share of the s1-pro engine it replaces. The sum is 1.0 (asserted when the code loads and again for every table).

**Fallback rule (data, not code).** If decisionmodels.io still serves d3-turbo as `s1-pro` at the binding, the variant `v20-d3` is
used: slot `d3` takes mammoth's weight 0.37, with its existing calibrated constants. It is one line in
`COMPOSITE_WEIGHTS`. The code never switches by itself. The lead and the release owner decide, and they announce it before the
binding, because it changes `composite_rules_sha256`.

The fallback also **renames the slot label**: `mammoth` becomes `d3`, so an image labelled `mammoth,blink` fails under `v20-d3`
(and `d3,blink` fails under `v20`). Miners must be told which label applies. Both the label change and the new rules hash
mean the fallback can only be chosen **before the binding, announced**, never mid-cycle: after the binding a switch would
reset the context (open commitments, confirmations and the champion rule all compare against the old rules hash). The same
holds for lowering a cap such as `MAX_OPTIONS["jeffan-31b"]`, which also changes the rules hash.

Evaluation order, cheapest first: blink, vision, reax1, jeffan, mammoth. The rules of section 3 are unchanged: a declared
model must pass every gate, the no-regression floor and the memory budget apply per slot, an undeclared model counts 1.0.
The label `ai.reax.engine.models` takes the names `mammoth`, `blink`, `reax1`, `jeffan`, `vision`; no label means all five.
A name outside the variant (`quyet`, `pplx`, `d3`) is a typo and fails the submission.

**Jeffan's contract.** The contract is the public decisionmodels.io limit for `jeffan-31b`: one question per request, 2 to 26
options, at most 8192 input tokens per question (gateway `MAX_OPTIONS[JEFFAN] = 26`). The contract has its own pool entry,
`MAX_OPTIONS["jeffan-31b"] = 26`. It takes no images. The prompt limits are the shared ones (instructions 8192 bytes, state 16 KiB).
The engine itself takes up to 255 options (25 Oct validation), but the competition follows what the gateway serves. The
generated text items have at most 7 options.

**Quyet's contract (v22 only; Quyet is out of the default).** At most 10 options per question (production `max_options`). The contract has its own pool entry,
`MAX_OPTIONS["quyet-1-large"] = 10`; the engine competition never reuses the 64 of `s1-pro` or the 16 of `s1-fast` for it.
The text items have at most 7 options (checked on 5000 generated items), so every item fits.

**Image items (slot `vision`).** The engine receives the production request: the field `images`, a list of exactly one
`data:image/png;base64,...` URL, only on the `s1-vision` pool. Every other pool rejects `images`. The contract check
(`s1contract.parse_image`) follows the gateway: PNG or JPEG, at most 4 MiB decoded, at most 2,000,000 pixels (and at least
8 x 8), valid base64, and a real PNG/JPEG structure (PNG: every chunk length and CRC, IHDR first, IEND last; JPEG: frame
header and end marker) whose type matches the URL. A winning engine can serve the request shape of `s1-vision` as the contract defines it; whether it may serve production traffic is
decided by the production adoption gate below, not by this score.

**Vision reference runs without CUDA graphs (determinism next to another engine).** In the competition the reference and the
candidate share one GPU. Measured on 25 Oct: two co-resident Blink engines with CUDA graphs answer the same image items
differently across processes (16 of 69 items exact, max |dp| 0.37), also with the vision encoder chunk budget pinned to
`total_bytes // 10`; the same engines with `--enforce-eager` are bit-identical (69/69 at concurrency 1 and 16). So the `vision`
reference runs eager (`REAX_VLLM_EXTRA=--enforce-eager` in the image; the encoder budget is also pinned). Against production
s1-vision it keeps every top choice, max |dp| 0.049. A candidate is free to use CUDA graphs, but its image answers must stay
within the slot's distance gates against the eager reference; an engine whose image numerics depend on what the other engine
holds will fail them. The slot weight is small (0.10), so the eager reference's lower speed (about 4 %) moves the composite by
far less than the winning margin.

The pictures are drawn inside the validator from the secret session seed. There is no image set, no file, no font and no
third-party licence. Nothing is looked up, so a miner cannot precompute, cache or fetch them. The generator is pure Python
with `zlib` (no Pillow); a picture is 320 x 240 pixels and about 1 KB to 3 KB as a PNG, and costs about 2.5 ms.
`vision_item(seed, index)` is a pure function: the same seed and index give the same item on every host. Six families, each
question depends on what the picture shows:

- **count**: coloured circles, squares and triangles on a grid (no overlap): "How many red circles are in the picture?" (options 0 to 6);
- **largest**: shapes of one kind (all circles, all squares or all triangles per picture, so the radius orders the areas) in different
  colours and clearly different sizes: "Which colour is the largest shape?". All six colours are always the options, in random order,
  so the option list does not show which colours are in the picture;
- **present**: "Is there a triangle in the picture?" (yes or no). The answer is decided first (50 % yes) and the picture drawn to it;
- **bars**: a bar chart with 3 to 6 bars of distinct heights: tallest, shortest, "Is the red bar taller than the blue bar?", "How uneven are the bars?";
- **number**: a three-digit number drawn with rectangles (7-segment digits): "Which number is written in the picture?" (4 to 6 options: the
  truth and random three-digit numbers drawn like the truth, independent of it) or "Is the number even?";
- **crowd**: 0 to 70 dots: "How crowded is the picture?" (five levels).

Colours, positions, sizes, counts and the state text vary with the seed; colours also get a small per-item jitter. There is no
answer key. As in the text items the validator compares a candidate with the reference engine's answers for the same
picture. The question text alone does not give the answer away: the options and the yes/no balance are independent of the truth, and
`tests/test_s1vision_leaks.py` checks that text-only heuristics (most central option, largest, first, alphabetical ...) are at chance
over 3000 items. An engine that ignores the picture can still win some single items by luck, but it has to reproduce the reference
distribution on every item, so it does not pass the distance gate. The generator is bound by its output:
`s1vision.GENERATOR_RULE` (label `reax-vision/2` plus a sha256 over the pixels and questions of 16 fixed-seed items) is part of
`composite_rules_sha256`, together with the pool's option cap and the image size caps. Changing a family, a palette or a text
changes the rules hash and starts a new rule set even if nobody bumps the label.

The anti-cache probes cover the pictures: a base item, then a variant with the **same state and the same picture but another
question**, and a variant with **the same question on another picture**. An engine that caches by state, by picture or by
question answers from its cache and fails the distance gate.

**Reference runs and the parent rule.** Each model has its own item tape, so the declared parent of a derived entry is
measured right after it in the same session on **the models the child declared**, with the child's items per model (the rc11.post6
parent rule, now on `dm-all`). A failed or missing model in the parent run is a failed run and the stored line applies.

**Calendar (Florian's lineup switch, board #22 #26507; fixed by the CoS, the release owner and the lead on 25 Oct 13:15Z; this is the
faster alternative to the "switch after Fri 30 Oct" calendar).** The Tue 27 Oct 14:30Z slot runs under engine rules v1 and is
the last v1 crowning. v2.0 is binding right after it. The first v2.0 crowning is Fri 30 Oct. The v1
bridge ends at the first v2 crowning or at the Tue 3 Nov slot at the latest.

| when (UTC) | what |
|---|---|
| Tue 27 Oct 14:30 | last v1 (`s1-fast`) crowning, v1 rules; v2 (`dm-all`) starts right after it (`ENGINE_V2_START_UTC`); also the start of the Option A announcement (`ENGINE_QUANT_RULE_START_UTC`) |
| Fri 30 Oct 14:30 | first v2 crowning; also the end of the weight-file conversion window (Option A, `ENGINE_QUANT_CONVERSION_END_UTC`): a title taken at or after this instant needs the weight-file screen (`weights_ok`), so a `dm-all` title taken at this slot needs it too |
| Tue 3 Nov 14:30 | latest end of the payout bridge (`ENGINE_V1_BRIDGE_END_UTC`) |
| every Tue and Fri 14:30 | crowning slots (`ENGINE_CROWN_SLOTS_UTC`), as on v1: a confirmed challenger waits for the next slot, `finalize_engine_slot` crowns without measuring |

Note: the first v2 crowning (Fri 30 Oct) is at the same instant as the end of the conversion window, so its winner needs `weights_ok`:
that slot is the first one under Option A. The last v1 crowning (Tue 27 Oct) is before the end of the window and needs no screen. Every `dm-all` title (the first is Fri 30 Oct) is therefore Option A compliant from the start;
an unreviewed `dm-all` entry measured after the switch does not even become a challenger.

**Lineup (decision of 25 Oct, board #22 #26507, #26739, #26768).** The slot list follows the models decisionmodels.io serves after its
switch: mammoth 0.37, blink 0.33, reax1 0.10, jeffan 0.10, vision 0.10 (section 10; weights picked by Florian 26 Oct). The code has these five slots (`v20`).
Quyet and pplx are out of the default.

### Validator runbook: weights to stage for the new slots

`weights_identity()` hashes the local weights directories, so every validator needs byte-identical files. Stage exactly these pins
before the binding:

| slot | repo@revision | shard sha256 |
|---|---|---|
| `mammoth` | `baselabs/mammoth-1@61cec44c281c693992e3d6ebeb44dd3402dfa186` | `29701f9a7b131bedb3fe1d565dad69b912d9128c6fc27e3d8fb0aedc47ae0e06`, `4da0ff5476813bbd9898454f0d7f7ba3c82914c42e32e2b87577c6d69daae671` |
| `jeffan` | `tattrongvu/Jeffan-31b-1710@cebfa86c2bed2b5c83b12e44de4374b20c87fd79` | (shard hashes are published with the GPU check) |

`blink`, `reax1` (and `d3` for the fallback) keep their existing pins.

### Measuring an image locally (miners, 26 Oct)

`reax-engine-eval --track dm-all` runs the validator's own session loop (`evaluate_composite`, release track constants,
published reference digest per slot, gates and per-model budgets; no chain, no champion rule) on one GPU:

    reax-engine-eval --track dm-all --candidate ghcr.io/you/engine@sha256:<digest> \
        --weights mammoth=/w/mammoth-1,blink=/w/blink,reax1=/w/reax-1,jeffan=/w/jeffan [--models mammoth] [--out run.json]

- Each `--weights` directory is the pinned Hugging Face snapshot above; it is mounted read-only at `/model`
  (`HF_HUB_OFFLINE=1`) and the engine gets `REAX_MODEL=<model id>`. `vision` uses the `blink` directory.
- Without `--models` the image's `ai.reax.engine.models` label decides, exactly as on a validator; `--models mammoth` measures
  one slot (the others count 1.0x). Only declared models need a weights directory.
- Output: one JSON line per candidate with `score` (weighted geometric mean), `parts` (per-model ratio and gates) and
  `declared`; stderr shows the `composite_rules_sha256` the run used (must equal the release's).
- `--reference slot=<image@sha256:...>` overrides a published reference (development only; listed as
  `overridden_references` in the output).
- A local result is a preview, not a verdict: hardware and noise differ from the validators' (RTX PRO 6000, runsc), the
  item seed differs (no block/digest), and the validator's pre-run submission review and Option A weight-file screen do
  not run locally.

### Switching from v1 to v2 (plain English; the promises of the 25 Oct check, board #22 #26409)

1. **Calendar.** The Tuesday 27 Oct 14:30Z slot runs under the v1 rules (`s1-fast` scoring) and is
   the last v1 crowning. `dm-all` is in force right after it:
   `competition_split.ENGINE_V2_START_UTC` = Tue 27 Oct 14:30Z. Before that instant no `dm-all` cycle runs and `dm-all` pays
   nothing (`EngineCompetition(composite_start_utc=...)`, set from the constant on finney). From it on `dm-all` is measured, but a
   composite is only ever crowned at a slot **strictly after** the start (`finalize_engine_slot`), so the v1 slot and the v2 start
   at the same instant cannot both crown. The first `dm-all` crowning is the Fri 30 Oct 14:30Z slot.
2. **Rights that carry over.** Approved awards, the former-champion places (17 % for the first, 8 % for the second, 7 days each)
   and the 14-day decay of the v1 title carry into v2 unchanged. The v1 title holder at the switch keeps the champion share
   (the payout bridge, paid from the frozen v1 state) until the first `dm-all` crowning, at the latest until Tue 3 Nov 14:30Z
   (`ENGINE_V1_BRIDGE_END_UTC`). When the bridge ends it does **not** drop the holder: the holder moves to the 17 % place as if
   beaten, its 7 days start at the bridge end (the first `dm-all` crowning, or 3 Nov 14:30Z if that comes first), and the v1
   former champions shift down exactly as a normal crowning shifts them (the old 17 % place becomes the 8 % place and keeps its own
   7-day window from its original dethroning; the old 8 % place drops out). Payment still needs each entry's own award approval
   on track `s1-fast` and the commitment still on chain. If the bridge ends without a `dm-all` champion, the carried places are
   paid on the decayed champion part counted from the crowning of the newest of them, and the rest goes to the inference part.
   **The carried place belongs to the hotkey (decision M3).** It stays paid, and during the bridge so does the champion share, while the
   hotkey's current commitment is either its original `reax-eng/1:s1-fast:` commitment or any `reax-eng/1:dm-all:` commitment. So a miner
   can enter v2 with the same hotkey and keep what it earned. It is lost if the hotkey commits something else (another track or
   another `s1-fast` image), commits nothing, or is deregistered (the uid then belongs to another hotkey), as in v1. The payout identity
   and the approvals stay those of the carried v1 image (the `s1-fast` approval of that digest), not those of the new `dm-all` image.
   A carried entry that the frozen v1 state struck as a copy (`copy_strikes`) earns nothing, as on v1.
   **A context change of `dm-all`** (new reference digests, rules, thresholds; `context_changed`) wipes `dm-all`'s own champion and
   its own former champions, as on every track, but never the carried v1 places: `carried_view` takes only the crowned and vacated
   events from the `dm-all` history and reduces what the reset wiped to the one fact the chain needs, the block at which the v1
   holder was first beaten. The v1 title itself follows the v1 rules.
   **The bridge end block.** The first chain block this validator sees at or after the end time is stored (`engine-v1-bridge-end-block.json`
   next to the engine state, never above the 12 s/block estimate) and used from then on, so there is no payout gap if the real chain is
   behind the estimate.
   Mechanics: `engcomp.carried_view` puts the frozen v1 history in front of the `dm-all` history, so one chain feeds
   `former_ranks`; v1-crowned entries are paid with the frozen state's results and the `s1-fast` approvals (`_carry`,
   `_carried_ranks`). The frozen file's sha256 (`ENGINE_V1_BRIDGE_STATE_SHA256`) is pinned by the release owner right after the
   27 Oct slot has been finalised; while it is `None` the bridge and the carried places are off, and a finney process refuses to
   start `dm-all` (`competition_split.release_bridge_problem`, checked in `worker.composite_track_from_env`).
   **Where the frozen file lives.** `REAX_ENGINE_V1_STATE_PATH` (default `engine-competition-v1.json` next to the engine state file).
   **Every validator needs the same bytes** (the sha256 pin is checked on every read); a validator with a different or missing file
   pays no carried place and the weights of the validators split. The file is only usable if it contains the finalisation of the Tue
   27 Oct 14:30Z slot (a slot result or crowning at that slot, a `slot_closed` event, or a slot schedule already past it); a file
   frozen earlier is refused with a loud log line. If nobody is crowned at that slot, the slot finaliser (`engslot finalize`, run
   after the slot as at every slot) appends one `{"event": "slot_closed", "slot_utc": "2026-10-27T14:30:00+00:00", "crowned":
   false}` event to `s1-fast` (`engslot.close_switch_slot`; deterministic, so the `check_state slot` replay accepts it); freeze the
   state after that publication. The event is not written while a confirmed challenger for that slot is still pending (it could
   still be crowned by a rerun or a late recovery), nor by a finaliser run at or after the Fri 30 Oct slot: run it between the two
   slots, and if a pending challenger blocks the freeze, crown it or clear it (tools/engslots/rollback.sh) and rerun. A finney validator (remote or local competitions mode, both build the same bridge) refuses to **start** when
   the file is configured but missing, mismatching or unfinished; at runtime it still fails closed (pays nothing from it).
   A dm-all parent reference run is capped by GPU time: all parent runs of one cycle together use at most one session budget of the
   child's declared models.
3. **One rule set from Fri 30 Oct.** `dm-all` follows exactly the rules of `s1-fast`: the quantisation Option A (a title taken at or
   after Fri 30 Oct 14:30Z needs the weight-file screen, `title_eligible`), the same-session parent rule (a derived entry must beat
   its declared parent by the full margin, measured in the same session) and the Tue/Fri crowning slots (a crown happens only at a
   slot). The same payout rules apply too: former-champion places and the co-champion logic work on `dm-all` as on `s1-fast`.
4. **Commitment format.** `reax-eng/1:dm-all:<image>@sha256:<digest>`. The models are declared by the image label
   `ai.reax.engine.models`; no label means all five. A v1 `reax-eng/1:s1-fast:...` commitment does not carry into `dm-all` (Plumb is
   not a v2 model, `CompositeTrack.carried = ()`): a miner commits to `dm-all` anew. Committing to `dm-all` does **not** cost the
   hotkey its carried place (see item 2, the carried place belongs to the hotkey); to keep it, the hotkey must not commit anything
   else.

**Licence condition.** Unchanged from v2.1 (the section below). It applies to every commitment to `dm-all`, including the
image path (`vision`) and the Jeffan path. (Entries are accepted only with a licence that lets us run them in production.)

**Time budget.** The hard cap of one full composite evaluation is `CompositeTrack.budget_s()` = 900 s slack + blink 1200 +
vision 1500 + reax1 1500 + jeffan 2160 + mammoth 2400 = **9 660 s (2 h 41 min)** in the v2.0 lineup (v2.2: 10 200 s, v2.1: 7 200 s). At the caps one RTX PRO 6000 completes at most 8 full evaluations a day (86 400 / 9 660 = 8.9). A real run is far shorter (v2.1: about 47 min measured), so the real rate is higher. `max_evals` stays
2 new submissions per cycle, so one cycle is at most 5 h 22 min plus confirmation sessions. A typical run is far below the
cap (v2.1: about 47 min measured); the two new slots (mammoth, jeffan) add an engine start each. The batch scripts are not changed.

**What the timed items cover (and what they do not).** The generated text items are short, single-question items: the state is
at most about 300 characters (maximum 307 in 5000 generated items) and there are 2 to 7 options. The score therefore does
**not** cover long contexts, 255 options, the 16,384-token cap, state-once on long states, Jeffan's caps (8192 / 16440 tokens)
or Jeffan's re-read of an unsure first read. An engine can drop any of these and still win. So **a winner must pass the
production adoption gate before it serves any traffic**: golden answers on long DI rows, many options, the token caps and
Jeffan's re-read, checked per model, then a canary and a rollback path. The s1m-supervisor runs that gate; the steps are in
the rollout runbook of the parent job (`RUNBOOK-ROLLOUT.md`). Adding long items to the generator would change the rules and is
not part of v2.0.

**Release calibration of 26 Oct (GPU check).** `mammoth`, `jeffan` and `vision` are calibrated: their rows carry
`release_calibrated: True` and `config.ENGINE_COMPOSITE_CALIBRATED_ON` records the digest each was measured on. The check ran on
one rented 96 GB GPU (Verda, runsc, subnet 57d2774); the full report is `workers/gpucheck-lineup-b/RESULT.md` of the parent job.
`CompositeTrack.release_ready()` is now empty for `v20` and for the fallback `v20-d3`. It still names a slot that is not
calibrated, has no published digest, or was calibrated on a different digest (a digest bump clears the calibration until the
slot is re-measured). Every slot is measured the same way: parity of the reference against the production engine, a fit of two
co-resident engines, two release sessions with an identical candidate, and a noise series of 3 fresh processes x 6 runs.
`blink` and `reax1` keep their calibrated rows and digests. The v1 bridge pin (section 6) is a separate release step.

- **mammoth** (`ref-mammoth@sha256:3080a593…`). At concurrency 1 the reference is bit-identical to the production engine on 500
  items, and a repeat is exact. At concurrency 16 both engines move with the batch composition (tops 500/500, largest
  probability difference 0.018). Two engines fit together (67.9 GB peak, 33.9 GB each; cold start about 467 s). Both
  sessions pass with d3's gates (worst mean distance 0.087, item 0.333, top agreement 99.4 %, probe 0.32, determinism 0.39);
  the slot scores are 0.980 and 0.996, and a session takes about 1086 s. The throughput noise is only 0.33 to 0.36 % (51.9
  decisions/s at concurrency 8). Still, the session score of an identical engine spread by 1.6 points. So the noise is set to
  **1 %**, not 0.37 %: the floor becomes 0.97 instead of 0.98, and an honest equal engine must not fail it. Budget 2400 s
  and memory 45 056 MiB stay.
- **jeffan** (`ref-jeffan@sha256:fdebc6e5…`). On the 500-item golden set all tops agree (500/500); the largest probability
  difference is 0.011, the restart drift that val-jeffan documented, and repeats in one process are exact. Two engines fit (83.3
  GB peak, 42.8 GB each). Both sessions pass (slot 0.960 and 0.956, all distances 0). The throughput noise is 2.17 % over all 18 runs
  (concurrency 8, 19.4 decisions/s), but the session scores of honest engines spread about 9 points (0.956 / 0.960 for the
  reference against itself, 1.042 for another preset), so the row uses **3.5 %** and the floor is 0.895. A session takes about 720 s; the budget stays 2160 s. An honest
  different engine (preset `fp8-kv-colo`, not batch-invariant) passes the same gates against the reference: mean 0.021, item
  max 0.625, top agreement 99.8 %, slot 1.042. The re-read of an unsure first read (top-1 below 0.9) fires on about 79 % of
  the competition items in both, so an engine that skips it fails the item gate. The sample used 2 to 7 options; the
  26-option boundary is not covered by this measurement.
- **vision** (`ref-blink@sha256:97d06a91…`, the Blink engine with the vision encoder; `--enforce-eager` is baked in). The
  reference is **eager**, not CUDA-graph: two co-resident eager engines are bit-identical on 69 image items at concurrency 1
  and 16 (with CUDA graphs only 16 of 69 were). Against the production s1-vision every top choice is equal and the largest
  probability difference is 0.049 (bound 0.05). Two engines fit (89.6 GB peak, 44.9 GB each). Both identical-image sessions
  pass (slot 1.005 and 1.016, all distances 0; about 362 s each). The noise is **0.30 %** (3 fresh processes x 6 runs,
  fresh pictures every run, concurrency 16, 16.9 decisions/s). A repeated picture hits the prefix cache and runs about 6
  times faster (16 to 120 decisions/s), so the noise is taken from the fresh-picture series, and the probe tape uses fresh
  pictures. The row now states `timed_items` 256. Budget 1500 s and memory 52 224 MiB stay.

**Noise and margin (v20, noise measured 26 Oct, weights of 26 Oct).** The unit of the composite is sigma = sqrt(sum (w_i x s_i)^2): mammoth 0.37 x 0.0100,
blink 0.33 x 0.0751, reax1 0.10 x 0.0325, jeffan 0.10 x 0.0350, vision 0.10 x 0.0030. That gives
sigma = sqrt(1.37e-5 + 6.142e-4 + 1.06e-5 + 1.23e-5 + 9e-8) = **2.551 %** and a margin of max(3 %, 3 sigma) = **7.65 %**.
Blink dominates: its bimodal throughput (7.5 %) is 94 % of the variance. (With the 25 Oct placeholder weights .50/.20/.20/.05/.05
it was sigma 1.720 %, margin 5.16 %.) The 1 % mammoth and 3.5 % jeffan row values are deliberately more careful than the
measured 0.37 % and 2.17 %; they barely move the unit at these weights.

## Licence condition and the road to production (v2.1, 22 Oct; unchanged in v2.2)

Florian, 22 Oct: approved engine gains ship to decisionmodels.io promptly; the licence must allow us to run the engine in production.

> **Licence condition.** By committing an engine image to the `dm-all` track, you grant productivity-boost.com Betriebs
> UG (haftungsbeschränkt) & Co. KG, the licensor of the REAX Engine, a perpetual, worldwide, royalty-free, non-exclusive
> licence to use, copy, modify, build and run the engine code in your image, including commercially in production on
> decisionmodels.io, and to merge it into the REAX Engine. An award is paid only after the complete source of your
> changes (including the source of every GPU kernel) is published under the MIT License or Apache-2.0 and our rebuild from
> that source matches your image. Engines built on the REAX Engine remain subject to the REAX Engine License for the parts
> that come from it. (Draft; needs Florian's OK before publication.)

Road to production (runbook: engine-composite-open-20261023 `RUNBOOK-ROLLOUT.md`, owner s1m-supervisor): award signed →
rebuild from published source in reaxlabs CI (we never run the submitted image in production) → golden-answer gate + DI/P300
on a rented GPU → burst node → canary 5/25/50/100 % with rollback → base fleet; target 48 h from the award to 100 % on the
first model. The deployed champion becomes the next reference for that model (new release, ≥ 24 h notice).
