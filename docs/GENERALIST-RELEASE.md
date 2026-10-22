# Generalist release preparation

This branch prepares G3 only. It does not choose the opening date or change the version/weights key. Both the competition
worker and remote validator need this code. Capacity, pool custody, calibration, dated notice and independent critic gates
remain with the release owner; no deployment is authorized by this document.

## Release constants (`competition_split.py`)

| Constant | Default | Required before release |
|---|---|---|
| `GENERALIST_START_UTC` | 2026-10-30T12:00:00Z | Set to the agreed opening; earlier dates use the same exact instant |
| `GENERALIST_COMMIT_FROM_UTC` | `None` (sync disabled) | Set to actual dated rules-notice time, no later than opening |
| `GENERALIST_ENTRY_FLOOR` | `math.inf` | Replace with six-release-draw s1-fast reference + 1.5 |
| `GENERALIST_G_MED` | `None` | Replace with the release's frozen calibrated median gap |
| `GENERALIST_KEEPALIVE_NONPROC_FLOOR` | `math.inf` | Calibrate non-proc reference + 1.5 on the same draws |
| `KEEPALIVE_BASE_FLOOR` | `{}` | Add base tensor hashes -> `{"generalist": nonproc_base_score}` for >10B keep-alive |

The placeholders fail closed: no generalist evaluations until entry floor and G_med are finite and the worker runs
`REAX_SPEC_X_TIER_MODE=binding`. The source `GENERALIST_G_MED` constant is mandatory for both evaluation and
provisional payment; an environment-only G_med cannot enable either. An infinite entry floor also blocks provisional payments.
An unset notice time disables generalist commitment sync, even after opening; import asserts notice <= opening when set. An empty base-floor table denies
keep-alive to entries above 10B (including provisional slots once parameter count is known), as in post6. Generalist has
no legacy-score floor: it has no scores predating this release. All pre-opening entries form one cohort without rewriting
their observed block/time. For these entries, the publication deadline is opening + 24 hours (wall clock),
not their original commitment block. Entries first seen at/after opening retain the existing 7,200-block deadline.
Pre-opening sync never rejects generalist entries for coldkey budgets; admission occurs at opening, ordered by observed
block/time/key, counting only admitted generalist entries (at most two open, four per week). Pre-opening observations
count from opening for that weekly window. Generalist entries never consume the DF/RA weekly budget. The existing cohort wait deadline and playoff rules remain in force.

`REAX_SPEC_G_MED` remains the existing explicit JSON override for the worker; populate the source constant as well,
freeze the same calibrated `generalist` value in both, and retain its receipt. Generalist shadow diagnostics cannot replace binding gap penalties.

## Opening playoff slots (from `OPENING_PLAYOFF_SLOTS_START_UTC`, date set in the release notes)

Decision: Florian, 21 Oct 2026 (board #23036). Only the generalist track's opening cohort: the first cohort playoff on the
vacant track after `GENERALIST_START_UTC` (all entrants committed before the opening). From `OPENING_PLAYOFF_SLOTS_START_UTC`
(`None` = off, the weights are exactly the ones without this rule), playoff places 2 and 3 take the track's two
former-champion ranks, 17 % and 8 % of the generalist share (the same 75/17/8 split as every track; no extra emission,
the share is the existing former-champion share). Place 1 is the champion.

* **Ranking.** The playoff's own order, rebuilt from the stored results (the playoff is found from the first champion's
  history, so it does not matter that old draws are pruned): entrants that took the extension draw first, then the stored
  playoff score, best first, ties to the earlier commitment. Entries the playoff skipped (twins, derived entries, entries
  below the floor after the extension) hold no place. The champion is place 1; places after the winner are ordered by the
  playoff score (mean of both draws when extended); ties by commitment order. Places 2 and 3 are the next two.
* **Qualification** (a place that fails is left empty, a lower place does not move into it; the entries after it in the ranks
  move up as for any unranked entry): playoff score above `TRACK_ENTRY_FLOOR["generalist"]`; the non-proc score of that
  playoff more than `KEEPALIVE_BASE_FLOOR[declared base]["generalist"]` + 1.5 points (strictly above, like the base rule) (`OPENING_PLAYOFF_BASE_MARGIN`; an entry
  whose declared base has no floor in that table does not qualify); not a copy or twin (`derived_from`), not withdrawn,
  pruned or rejected, copy check done, and still committed to that submission; and the normal award approval. An unapproved
  slot keeps its rank and its share goes to the inference part (rank 3 is not promoted), like any unapproved former champion.
* **Duration.** Both slots start at the later of the crowning and `OPENING_PLAYOFF_SLOTS_START_UTC` (so a playoff that ran
  before the release gets its 7 days from the activation instant) and last 7 days (`OPENING_PLAYOFF_SLOT_DAYS`, end
  exclusive). They sit behind the real former champions: when the champion is dethroned it becomes the newest former
  champion and pushes the slots down; a slot pushed out is gone for good: every former champion crowned since the opening crowning removes the lowest place for
  good, even if that former champion later stops committing. A slot is never rank 1. If the champion no longer commits and no
  former champion exists, no slot is paid (the share goes to inference).
* The 7-day gate uses each validator's wall-clock UTC, like the other dated rules.
* Deterministic from the saved state and the clock; no new network inputs, no environment override.

## Schedule and draws

The default `50/50` profile changes exactly at opening from two slots (0.15 DF + 0.15 RA) to one (0.30 generalist).
DF/RA end at min(published 30 Oct 12:00 UTC, opening). Schedule conversion retains exact seconds, including fractional days.
The inactive `80/20` development profile retains its one 0.10 slot. The cancelled DF/RA extension and open class do not apply.

Generalist uses `<sealed_dir>/generalist/sealed/sealed.clean.jsonl` (or `sealed.jsonl`). Every row must have a unique id,
a valid immutable `category` from `sampling.GENERALIST_CATEGORIES`, a choice/noul/score type and a known tier. Previously drawn
ids cannot change category across pool refreshes or worker restarts. Only ids/categories and aggregate diagnostics are
persisted; no item text or predictions are exported.

Draws have exactly 30 items in each of the ten categories. Tier quotas apply per category across all types, with rounding
carried across categories so an unconstrained binding draw has 45 proc and 60 cross-author items. Eligible-pool type
proportions are targets within those tier constraints; a tier may borrow across types within its category. Missing tiers
are filled from non-proc, non-X items of that category; an X deficit is recorded in `Draw.x_shortfall` (category -> count),
never a reason by itself to stop. If those replacements are exhausted, remaining eligible items complete the category,
even if they exceed a tier target. This implements the requested liveness rule: a valid pool fails a draw only when a
category has fewer than 30 eligible items. No borrowing across categories, no partial draws. Review all tier shortfalls
in the release dry-run, especially degenerate proc-only categories; the dry-run is a release gate, not a calibration waiver.
Use counts still retire each item at 24. Seeds use the existing mechanism.

**Rules v4, larger draws (from `DRAWS_V4_START_UTC`, date set in the release notes).** From that instant a generalist draw
has **90 items in each of the ten categories (900 per draw)**, with the same tier shares, type targets, borrowing and
liveness rules per category (so an unconstrained binding draw has 135 proc and 180 cross-author items), and a valid pool
needs at least 90 eligible items per category (and 180 for a first draw plus its disjoint confirmation). Before that
instant, and while the constant is unset, draws stay at 30 per category. The margin floor of the generalist track drops
from 1.5 to **1.0 point** from the same instant (`machine.MARGIN_FLOOR_V4`; see "Decision rule v3" in
SPECIALIST-TRACKS.md); the z-gates are unchanged. Use counts still retire each item at 24, so the pool is used up three
times as fast per draw; watch the "refill needed" warnings, which now count 900-item draws. An entry whose first stage was
taken before the instant keeps its draws: its later draws are 900-item draws, decisions taken after the instant use the
1.0 floor, and draws of different sizes are pooled as before (mean of the gains over `rms(SE) / sqrt(k)`). The entry floor
and keep-alive floors are not changed by this rule (see the release notes for their recalibration, if any). To dry-run the
larger draws on the real pool: `python tools/generalist_draw_dryrun.py /path/to/pool.jsonl --pairs 100 --per-category 90`
(default 30; output counts only, as above).

Run `python tools/generalist_draw_dryrun.py /path/to/pool.jsonl --pairs 100` on the real G1 pool before release. Each pair
excludes the first draw from its confirmation and carries use counts forward across pairs. Output contains only counts:
complete pairs, draws possible, and aggregate shortfalls per category/tier. No item text, ids or predictions are printed.

Category diagnostics use the existing scorer for each draw and pool its sufficient statistics across draws, by primitive,
so a pooled score is the score of the concatenated observations rather than the average of draw scores. These fields
(`Submission.category_draws`, `category_pooled`) never enter a floor, gap penalty or decision. `spec/scoring.py` is unchanged. Per-draw category summaries are pruned with retained draws (pending references plus
the newest 40); pooled sufficient statistics survive pruning. Only drawn ids are bound in `TrackState.item_categories`;
loading or correcting a never-drawn pool row creates no binding.

## Operator resources

Keep post6's pinned serving templates, text-only architecture handling and weight-sized GPU reservation. A 36B/68GiB
entry requires a measured 96GB-class box; set `REAX_SPEC_GPU_BUDGET_GB` >=81 (88 recommended),
`REAX_SPEC_MIN_FREE_GB` >=150, `REAX_SPEC_BASE_MIN_FREE_GB` >=100, `REAX_SPEC_CONTAINER_MEM` about `80g`,
`REAX_SPEC_STARTUP_TIMEOUT_S=1800`, and set evaluation rate from the capacity measurement. The default base download
budget is 40GiB before opening and 80GiB afterward, evaluated per download so an early-started worker switches too.
An explicit `REAX_SPEC_BASE_BUDGET_GB` overrides both. Templates/caps do not prove GPU capacity; G2 still needs measurements.

## Award-time Decision-Index check

Before any generalist award request, the release owner's award pipeline scores the candidate and the official base
(REAX 1, `decisionmodels/reax-1@ca2a0cc5...`) on the frozen house DI-proxy eval: the HF-DI-style composite (0.6 private
same-skill + 0.4 private new-domain) plus the AA-style part, in the model's native protocol where it has one. This is a
GPU run of about 10 minutes per candidate on the gen box and is not part of the validator. It exists so we do not pay for
gains on our own pool that do not hold on the Decision Indices.

`tools/award_di_proxy.py` takes the two per-item result files (`id`, `unit`/`area`, `weight_group`, `score` in [0,1],
`protocol`), `--eval-file` with its expected `--eval-sha` (checked by the tool) and the ids, and prints the `di_proxy`
block `{done, eval_id, eval_sha, protocol, base_id, items, score, base_score, diff, se}` in points (0-100, 4 decimals).
It refuses mismatched item sets, duplicate or missing ids, NaN, out-of-range scores, differing unit/protocol per item and
an eval sha mismatch. Composite weights come from a small JSON (unit -> weight, normalised; default equal over units).

The comparison is paired per item: `diff` = weighted mean of (candidate - base) item scores, with the same weights.
`se` is the larger of a stratified paired bootstrap (2,000 resamples per run, seeded) and the analytic paired SE.
Threshold (non-inferiority): `lower90 = diff - 1.2816 x se >= -0.5` points. If the check fails, or the block is missing
or not `done`, the signer does not release the award automatically and raises a manual card to Florian with the block
and the item-level files. The tool only produces the numbers; the decision stays with the signer.

## Release-owner checklist

- Set the opening and actual notice time together; notice defaults to disabled and must be no later than opening.
- Freeze the entry floor, independent non-proc keep-alive floor, source G_med and any worker override from calibration receipts.
- Run the real-pool first/confirmation dry-run; retain counts and review all tier shortfalls before shipping.
- Set the release version in `version.py` and `pyproject.toml`, and choose `WEIGHTS_VERSION_KEY` deliberately. This slice
  leaves these inherited values unchanged; the release owner owns the final version and weights-key decision.

## Joint-readout entries

From generalist opening, `spec/joint.py` recognizes the pinned pplx v1.2 data format (`Qwen3_5Model`,
`decision_config.json` and `readout.safetensors`; optional `joint_config.json` must match protocol v3 exactly).
Partial metadata, unknown architectures, unsupported dtypes and mismatched head/hidden dimensions are rejected.
The initial allowlist is deliberately only the dense standard Transformers `Qwen3_5Model` class; MoE and future
architectures require a reviewed code change. The observed upstream revision is
`perplexity-ai/pplx-decider-v1.2@cce0f3f80033e9d89d80831d5281fd72124aacce`; its readout is BF16 `[255,5120]`.
Derived entries keep their own committed revision, checked by the existing fetcher. All safetensors, including the
head, enter parameter/byte limits and full canonical hashes; matching backbone tensors enter delta sketches. Draw construction and scoring are unchanged.

The reviewed Transformers worker runs in the same pinned image with the same sandbox flags and cleanup as vLLM.
It uses `REAX_SPEC_DOCKER_RUNTIME`, including the Docker default when unset, just like causal entries. It never imports
candidate source, uses a repo template, enables remote code, loads pickle, or fetches a config's base-model path. Full-attention
layers use a bidirectional padding mask; hybrid linear layers retain the native recurrent mask. This reproduces the
house parity script's semantics, not an all-layers bidirectional rewrite. The last prompt-position hidden vector goes
through readout rows A..Z, with the existing REAX prompt, fixed qwen template, T=2 and answer post-processing.
The native joint markers, saved temperature and multi-question protocol are metadata only in this serving mode.

Inference is prefill-only and cache-free. Equal-length groups batch up to four sequences under an 8,192-token budget;
each item appears in one forward pass. Unequal lengths run separately to preserve position/mask parity with unpadded
single-item measurements. No truncation: prompts above the existing 8,192-token cap fail as items. The CUDA allocator
has a per-process limit from the same weight-sized GPU reservation. Usual host RAM/time limits still apply.

Before release, prove the pinned image has the declared Transformers Qwen class, safetensors, tokenizer and aiohttp
modules; check the configured Docker runtime/NVIDIA combination on the actual host. Capacity and numerical GPU parity are still release
gates. With the pinned public snapshot downloaded separately, run inside a network-disabled GPU environment:

```
python tools/joint_readout_gpu_check.py /path/to/cce0f3f80033e9d89d80831d5281fd72124aacce items.jsonl --limit 300
```

For a non-Hub-cache directory, supply `--revision cce0f3f80033e9d89d80831d5281fd72124aacce` from the pinned download
receipt. The script never downloads or rents hardware; it reports per-item timing plus accuracy and raw multiclass
Brier, with REAX T=2. It accepts single-question house JSONL (`request`/`gold`) or rows with `state`/`question`/`gold`.
Use public approved smoke inputs first. The runtime does not expose predictions in release artifacts by itself.

### Joint loader and failure attribution (critic follow-up)

Only `architectures == ["Qwen3_5Model"]` selects joint serving; causal entries ignore stray joint files.
Joint entries require every safetensors file to appear in the index or be `model.safetensors`/`readout.safetensors`.
They require `tokenizer.json` with matching answer-code IDs, reject adapters, and require flat safetensors
index paths. The fixed loader streams weights with `device_map={"": device}` and rejects missing, unexpected or
mismatched weights. Both complete tensor hashes remain recorded; generalist exact-copy checks also record a canonical
backbone hash excluding `readout.safetensors` and `lm_head.weight`. Qwen backbone names with and without `model.` map
to the same sketch names in either direction. Sketch rule 4 invalidates sketches sampled under the old name mapping.

Before load, the backend requires container memory of at least tensor bytes plus max(4 GiB, 10%). This is a conservative
host admission check even though device-map loading reduces CPU peaks. A cached per-image/runtime/GPU preflight checks
fixed imports and CUDA access in the same network-disabled sandbox. Only that environment check and Docker/socket
faults are host outcomes. Once preflight passes, every candidate-load exception is a model failure; CUDA OOM after the
free-memory check and container OOM-kill after the weights-plus-margin check are model budget failures. Joint entries
receive at most three consecutive infrastructure outcomes without a strike; subsequent failures count as model strikes
until a successful serve or a direct model outcome resets the persisted counter.
Preflight success is cached only in process memory; failures are retried. The runtime checks dependencies on each start.

The HTTP server accepts up to 64 MiB and eight rows; the host sends four rows per 120-second request. Rendering, tokenizer,
forward and CUDA OOM errors produce per-item `None` for the affected group, and later groups continue. HTTP errors or
request timeouts fail the affected four rows. GPU parity, BF16 batch-composition effects, pinned-image compatibility and
measured 27B throughput still require the release-owner GPU gate.
