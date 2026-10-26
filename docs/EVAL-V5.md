# Eval v5.1 (generalist, HF Decision Index 0.4 structure) — build notes

Status: built and tested offline behind `sampling.EVAL_V5_START_UTC = None` (off). Not activated, not wired into the evaluator's
serving/decision path (`v5.EVALUATOR_WIRED = False`; `SpecEvaluator.__init__` raises if the gate is set). Spec: board #22
`handoff/SPEC-DELTA-v5.1.md` and PROPOSAL §1.1–1.6 (sn92-eval-di-alignment-20261022). This replaces the v5 M/P design (cap, D/SE_D).

## Pieces
| what | where |
|---|---|
| gate, `v5_active`, typed-file loading, draw routing | `spec/sampling.py` (unchanged) |
| draw (T 540 + N 360 by task type), item scoring, type / area aggregation, stock-panel scaling, `score_v5`, gap check, overlap rule, prompts / codes, migration helpers | `spec/v5.py` |
| offline scorer | `tools/score_v5.py` |
| tests | `tests/test_eval_v5.py`, synthetic fixtures `tests/data/v5/` (`gen_synthetic.py`) |

## Score
```
T_eq = mB + sB (T - m_T) / s_T          N_eq = mB + sB (N - m_N) / s_N        mB = 23.4973, sB = 16.539081
S_v5.1 = (1 - w_N) T_eq + w_N N_eq      w_N = 0.4 by default, 0 = twins-only fallback (S = T_eq)
```
* `m_X`, `s_X` are config per pool version (`PanelStats`, `--panel-stats`), from the stock-panel run; never computed per draw.
  `v5.PANEL_STATS` is empty until that run delivers numbers. Without stats the CLI needs an explicit `--unscaled` (X_eq = X, flagged).
* No cap `min(P, ...)`, no D / SE_D terms.
* **Task type score** (0 = random or best constant answer, 100 = perfect): `100 (acc - c) / (1 - c)`, clipped to 0..100. `c` is the random
  accuracy, raised to the most frequent gold key's share when all items of the type have one question (`const_baseline`, on by default;
  `--no-const-baseline`). Exact ties score 1/k (S3). Omniscience (task_type `omniscience`): `(100 + correct% - incorrect%) / 2`, IDK = 0 (S1).
  WildGuard (`wildguard`): per label F1 chance-corrected against the best constant answer (S2). Both keep their S1/S2 scale.
* **T**: type scores averaged inside each area, gold types x1.2; areas knowledge .2585, language .2585, retrieval .2002, tools .1828,
  arts .10, renormalised over the areas present. **N**: plain mean over its task types (every task counts the same, no areas).
* **Items** carry `part` (`T` / `N` / `pub`), `area`, `task_type`, `gold` (bool). A task type has one area, one gold flag, one part.
* **Weight 0, reported**: typed public items (`part: "pub"`), v4-format public tiers (`same_family`, `proc_sealed`) and the old own tiers
  (`new_family`, `new_domain`, `cross_author`; `award_realdata` is an alias) -> `detail.internal.tiers`.
* **Gap check (report only)**: `Pub` = item-count-weighted mean of the typed public score (area aggregation) and the v4-format public
  composite (`scoring.combine`), scaled with `Pub` stats if the panel has them. `G = (Pub - Pub_base) - (S - S_base)`, `SE_G` = paired
  bootstrap over the items of all parts (same resample for entry and base), `A = max(a_floor, a_se_mult SE_G)` (1.5, 2),
  `gap_flag = G > A`. The flag holds awards and keep-alive for a human overlap check; it never changes S. No public items: G, SE_G, A, flag = null.
* **Overlap exclusion** (`--exclude-types`): for each listed T or N task type the entry's type score becomes `min(entry, base)` on the same
  items (applied inside every bootstrap resample). Output `overlap_excluded: true` only; the ids stay in `detail.internal`.
  `v5.public_view(out)` gives the public subset (no type ids, no per-area / per-type scores).

## Draw
`draw_v5`: 540 T items by area weights, inside an area by type weight (gold x1.2), every type >= 12 items; 360 N items equal per task
type, >= 12 each; `V5Params.pub_items` (default 0, e.g. 120) public items for the gap check on top. The cells come from all typed rows
of the pool, so a type whose items are used up fails the draw instead of silently giving its share away. Retirement after 12 uses.

## Offline scorer
```
PYTHONPATH=/path/to/eval-v51/wt/src python tools/score_v5.py --items typed.jsonl [--items public_and_old_tiers.jsonl] \
    --responses model.jsonl --base-responses base.jsonl --panel-stats panel.json [--wN 0.4] [--exclude-types types.json] \
    [--seed 0] [--min-type-items 12] [--allow-partial]
```
Responses: JSONL `{"id": ..., "probs": {key: p}}` (a list of such dicts for multi-question items). Panel stats:
`{"pool_version": "v0", "T": {"mean": m, "sd": s}, "N": {...}, "Pub": {...}}` (N needed while w_N > 0; Pub optional).
Output: JSON `S_v5.1, T, N, T_eq, N_eq, Pub, G, SE_G, A, gap_flag, overlap_excluded` and a `detail` block (base values, coverage,
params, and `internal`: per-area, per-type, per-tier scores, type item counts, excluded ids). Rules kept from v5: coverage >= 95 % for
candidate and base (`--allow-partial` overrides; a missing answer counts as wrong), clean exit-2 errors, `--max-rows`, parameter
validation. New: every T / N type needs `--min-type-items` items (default 12 per draw; give 12 x draws for pooled draws; 0 disables).

Run from the repository root with the checkout's absolute `src` path in `PYTHONPATH`; do not use `python -I`.

## Scoring decisions S1–S3 (spec owner, 2026-10-22; `SPEC-DECISIONS-S1-S3.md`) — unchanged
**S1** Omniscience `aa_index` (default) / `half_half` (gate-only, `omni_acc_weight`). **S2** WildGuard `const_corrected` (default) / `raw`.
**S3** exact ties earn 1/k if gold is among the k tied top options (mirror-format items, WildGuard counts, v4-format `choice` rows via
`v5.pool_credit`). The v4 `scoring.py` is byte-identical.

## Second serving slot and evaluator wiring (implemented, behind `EVAL_V5_START_UTC`; `spec/v5wire.py`)
Once `sampling.v5_active(now)` the generalist track decides on S_v5.1 (the gate `None` keeps rc11 bit for bit; there is no new state
field: an entry scored under v5 has `by_tier == {"T": .., "N": ..}` and its history rows carry `scoring_regime = "v5"`).

- **Pool.** `REAX_SPEC_V5_POOL_DIR` holds `v51.T.jsonl` and `v51.N.jsonl` (mode 0600, anything group/other readable is refused;
  typed rows as above, tier `mirror`, `part` T / N). `N` may be absent only with `REAX_SPEC_V5_W_N=0` (the T-only fallback: no N
  items are drawn or served). The public items of the gap check are the `same_family` / `proc_sealed` rows of the ordinary sealed
  generalist pool. `REAX_SPEC_V5_PANEL` is the JSON of `v5.panel_from_json` (stock-panel mean / SD per pool version), unset = unscaled.
- **Draw.** `v5.draw_v5` with `pub_items = 90`: 540 T + 360 N items by task-type quota, at least 12 items per type, fresh and least
  used first, retired after `MIRROR_MAX_USES` uses, no borrowing across types (a short type raises `SampleError`: no draw), plus 90
  public items at weight 0. Confirmation draws exclude the first draw's items as before.
- **Serving.** Every question of a T/N item is its own `choice` row (`v5wire.flatten`, IDK an extra last option); public rows
  are served unchanged. The champion, every seated co-champion (`cochamp_members`, rc11.post1) and the declared base of every
  challenger are served on the same items of every draw. Per (model key, draw id) the result is cached, so a base shared by
  entries is served once. A co-champion or base that cannot be served abandons the group (re-measure next cycle).
- **Decision.** `S_v5.1` (`v5.score_v5_detail`) replaces the v4 score everywhere in `_decide`. The reference of a challenger on a draw is the best of
  the seated co-champions and its declared base; gain, paired SE (bootstrap over the T, N and public items), margin, first stage,
  close call and the pooled confirmation are the existing v3 machinery with that reference (stage rows pool only draws against
  the same reference set, `inc_key = v5|<co-champion keys>|<base>`). `gap_penalty` / the binding mode are not applied.
- **Published numbers** (in the history rows, `scoring_regime = "v5"`): `s_v51`, `t`, `n`, `pub`, `ref_best`, `ref_totals`,
  and the gap check `gap` (G), `se_gap`, `gap_allowance` (A = max(1.5, 2 SE_G)), `gap_flag` (G > A; `v5wire.gap_flagged(history)`).
  The flag does not change the score; the hold of awards and keep-alive promotion for the overlap check is the share layer's.
- **Floors.** The v4-scale entry / non-proc / base floors are not applied to an entry whose score is S_v5.1 **and** the gate is open
  at the time of the check (`_v51_scored(s, now)`; the marker is `by_tier == {T, N}`, written only for challengers, the champion and the
  seated co-champions of a draw, never for a parent served as a reference). `_big_ok` keeps its parameter checks (invalid count: no
  slot; unknown count: eligible) and skips only the v4-scale base-floor comparison. The share layer drops the v4 generalist floor while
  v5 is active (`combine_weights`), so `compute_leaders` applies it to entries without the marker (critic F2/F6). A v5 score may be
  zero or negative and still qualifies. For the keep-alive top-eight cut, a v5 entry ranks by its average `S_v5.1 - incumbent`
  margin over its v5 first and confirmation decision rows, rather than its raw DI score; v4 entries keep their existing
  score-based qualification key. An imported v5 marker with no decision rows uses a neutral margin of zero.
- **D4.** From the gate on, the provisional half keep-alive starts at the entry's on-chain commitment block (age from
  `commitment_block`, 12 s per block) and no longer waits for the immunity lead; it still ends 12 h after that start, after a
  decided first draw, and the queue prepares entries whose immunity ends soonest first (`queue_order`, unchanged). An entry that is
  not fetched yet gets its window only after a repo probe succeeded (`SpecEvaluator._d4_probe`: the repo exists, is public, the
  committed revision is there and a `.safetensors` file is listed; metadata only, `WAIT_REPO_PROBES_PER_CYCLE` per track and cycle, a
  failed probe again after `RETRY_UNAVAILABLE_S`; the result is kept in memory, `provisional_since_ts` is the stored record). A
  missing or dummy repo gets no slot (critic F5).
- **Options above 26** are capped in the draw (`v5wire.cap_item`, applied by `v5wire.draw`): every question with more than 26 options
  (25 with the IDK option, which is added when the item is served) keeps the gold option plus distractors sampled from
  `(draw seed, item id, question index)`, the keys re-lettered A..Z in their original order, and a number in the task text equal to
  the original option count followed by candidate(s)/answer(s)/tool(s)/option(s)/operation(s)/document(s)/choice(s)/label(s)/move(s)/
  chord(s)/agency/agencies/title(s)/division(s) is replaced by 26 (`COUNT_RE`, as `scripts/merge_pools.py`). The subset is the same
  for every model of the draw and new for every draw; chance-correction and the readout only see the capped rows. The pool keeps the
  full questions. The two-letter-code design below stays for a later release.
- **Joint packages** (`spec/joint.detect`: Qwen3_5Model + readout head + `decision_config.json`) are served on v5 draws exactly as
  on v4 draws: `_serve` inspects every served model directory (co-champions, challengers, bases) and routes a joint package to
  `JointDockerBackend`, never to the causal readout. A multi-question item is served as one row per question (`<item>#<j>`, shared
  state repeated, same template, one marker per call).
- **Failure scope.** A challenger whose declared base is not on this host, whose base cannot be served (including a `Qwen3_5Model`
  base without the required readout head), or whose S_v5.1 (or its
  base's / parent's) is not finite waits alone (`reason` starts with `waiting`, a queued entry is tried again after
  `RETRY_UNAVAILABLE_S`); the group goes on with the others. A seated co-champion / champion that cannot be served or has no finite
  score abandons the group (re-measured next cycle). The served-result cache is keyed by `sha(draw id + sorted row ids)`, checked for
  its length on a hit and cleared every cycle. The pool is read sorted by item id.
- **Several crownings in one cycle.** An entry crowned earlier in a `_decide` loop is part of the bar of the later challengers of the
  group (max(reference, the new champion on this draw), SE against it, stage rows labelled with the extended reference so they are never
  pooled with draws against another reference): a weaker challenger cannot take the title from a stronger one crowned in the same cycle.
- **Entries that straddle the gate** (critic F7): an entry that passed its first stage under v4 before the gate is restarted by rule A
  (below) unless it is confirmed; if it is somehow confirmed under v5 against a first stage measured under another reference, the
  stage rows are not pooled (different `inc_key`) and its confirmation must reach the crowning z on a single v5 draw: stricter, not
  weaker.
- **Vacant track on v5.1** (rc11.post5, #25186): with crowning slots configured, an entry that confirms on v5.1 on a track without
  a champion (its bar: its declared base, by the margin, on both draws) is stored for the next slot (`crown_pending`) exactly as on an
  occupied track; it is never crowned ad hoc. At the slot one such entry is crowned alone, several play the v5.1 slot rounds
  (rounds 1-3, `slot_eval_ts`, separation by the margin, G test): the cohort wait is replaced by the slot. A vacant-track
  confirmation taken under v4 that still waits for its cohort when the gate is open is restarted as a v5.1 first stage (rule A).
- Not wired: a vacant-track cohort playoff under v5 (it fails closed with a log line). It is reached only without a slot ahead
  (after the last instant of `CROWN_SLOTS_UTC`) or with `CROWN_SLOTS_UTC` empty: do not run v5.1 without slots on a vacant track.
- Crowning slots on v5.1 (rc11.post4): every slot after the flip slot (from Thu 29 Oct; `slot_eval_ts(slot)` = slot - 1 s is at or
  after the gate) decides its rounds on v5.1: a v5.1 draw per round, each member's S_v5.1 per round (`v5wire.slot_context`), the
  pooled mean, the v5.1 paired SE pooled over the rounds (`v5wire.slot_pair_se`), margin floor 1.0 and G 0.5, no v4-scale entry
  floor. Fail closed without a usable pool / config / panel: no slot round, nothing crowned, the slot runs when the pool is back.
  A crowning earned under v4 is not measured in a v5.1 slot (rule A: it restarts as a v5.1 first stage); the flip slot itself
  stays v4 and refuses entries decided under v5.1.

### Transition rule A (Florian, 22 Oct 2026, binding; `EVAL_V5_START_UTC` = the crown slot 2026-10-26T14:30:00Z)
An entry that confirms under the current (v4) evaluation before the flip is crowned at the slot as earned; afterwards titles are
defended on v5 and new entries are scored on v5. Wiring (all behind the gate, `evaluator.py`):
- A v4 confirmation before the flip is stored as the existing `crown_pending` row (CROWN_SLOTS_UTC, rc11.post1). `_finalise_crown`
  crowns it at the slot without looking at a score again, and runs **before** `_v5_restart_v4_stages` and before the cycle's draw:
  at exactly 14:30:00Z the crown comes first, then v5 (`now >= slot` and `v5_active(now)` are both inclusive).
- An entry whose v4 first stage (pending confirmation, or a queued v4 close call) has not confirmed and has no stored crowning at the
  flip is reset to QUEUED once (`_v5_restart_v4_stages`; history row `v5_restart`, log line, `close_draws`/`confirm_draws`/`draw1`/
  `confirm_after_ts` cleared) and takes a v5.1 first stage. The same for all entries; it never confirms under v4 after the flip.
  A v5.1 stage (first-draw row with `scoring_regime == "v5"`) and a vacant-track confirmation taken on v5.1 are left alone; a
  vacant-track confirmation taken under v4 (waiting for its cohort playoff) restarts too (rc11.post5).
- After the flip every challenger draw uses v5.1 against the CURRENT title holders and the challenger's own base: while the seated
  co-champions hold the title, all of them (`cochamp_members`); after a rule-A crowning the new champion alone. The former champions
  (the co-champions are made FORMER by the crowning, rc11.post2) and the pending members are not in the bar and are drawable and
  servable as normal entries (`v5_bar_members` == `cochamp_members`).
- rc11.post2 (crown slot without freeze, slot playoff): several entries that confirmed under v4 play the slot playoff on shared v4
  draws at the flip slot, although the gate opens at that very instant (`_v4_draw_now`: the playoff draws at one second before the
  gate); afterwards v5.1 applies. The other entries of the field are NOT_BETTER or back on the challenger path and restart as v5.1
  first stages. The slot code itself is the release owner's.

### Options above 26 with two-letter codes (design only; v0 caps instead, see above)
Without the cap: `prompt.options` raises `PromptError` for more than 26 options; `vllm.VllmDockerBackend._one` (the vLLM/docker path;
`_load_letter_ids` knows only A..Z) and `joint.JointReadout.decide` (the joint path) both catch it and return `None` for the row, so
the question is wrong with zero credit for every model alike. `v5.render_question` already renders two-letter codes (`v5.option_codes`,
up to 255) but no backend calls it. Not implemented because it is not contained:
1. **Prefix problem.** `option_codes` keeps A..Z as single-letter codes and adds AA.. so code "A" is a prefix of "AA".. A sequence
   log-prob of a code then needs the end-of-sequence probability after a single letter. Proposal: for items above 26 options use
   prefix-free two-letter codes for **all** options (AA..), a change to `option_codes` for k > 26 only (the joint `detect` rule about
   the first 26 codes applies to the joint path, which does not serve these items).
2. **Readout (causal path).** Call 1: distribution of the first letter over the 26 letters (`allowed_token_ids`, `top_logprobs` 26).
   Then one continuation call per first letter that has a code (`ceil(k/26)`, assistant prefix = the letter, `continue_final_message`),
   distribution over the second letter. Code log-prob = lp1 + lp2; answer probabilities = `softmax_t` over the k code log-probs at the
   v4 temperature. Cost: 1 + ceil(k/26) requests per row (k = 255: 11) instead of 1.
3. **Tokenizer risk (needs a GPU probe).** A tokenizer may merge "AA" into one token, so the two-step path differs from the path a model
   was trained on and the artifact is not the same for every model. Probe the tokenizers of the allowed base families (the two letters
   must tokenize as two tokens, or the code must be read as one token id) before enabling it.
4. **Where it plugs in.** `vllm.VllmDockerBackend._one` (branch on `len(question criteria) > 26`: `v5.render_question`, two-step
   call, `P.softmax_t`); `v5wire.flatten` rows are already `choice` rows with IDK as the last option. Unit tests need only a fake
   backend (prompt/code rendering, log-prob aggregation); the joint backend keeps returning `None` for such rows or routes them
   to the causal path in `_serve` (`evaluator.py`, near `factory = self._joint_backend`).

### gap_flag hold
`v5wire.gap_flagged(history)` (public, reads the newest `gap_flag` in an entry's history) is the reader. This repo has no award
*request* payload: `award.py` only verifies owner-signed approvals (`_KEYS` = track_id, hotkey, submission_id, approved_at,
expires_at; a field cannot be added without breaking the signature format), so there is no `hold_reason` field to add here. The
auto-approve policy that decides which awards are signed (outside this repo) must call `gap_flagged(submission.history)` and hold the
award with `hold_reason="gap_review"` (and not promote the keep-alive) while it is true; the evaluator only records the flag.
