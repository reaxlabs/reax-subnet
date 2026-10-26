# Changelog

## 1.2.0rc11.post7 — s1-pro cap x4 (640/s) and s1-fast cap x4 (2560/s)
Validator + miner release, `WEIGHTS_VERSION_KEY` 1260 (was 1250), shipped as its own consensus release after a live probe bench
(Florian, card 17931 "All 3", board #21849; s1-fast x4: board #26839 "way higher caps"). Start: Tue 27 Oct 2026 16:00 UTC (dated #92 notice, >= 24 h). Rollback: `tiers.S1_PRO_CAP_MULTIPLIER = 2.0` and / or `tiers.CAPACITY_CAP_MULTIPLIER = 2.0` (+ key bump).
- `tiers.S1_PRO_CAP_MULTIPLIER` 2.0 -> 4.0: s1-pro cap 640/s, 64 lanes, 704 requests per probe, verified sample 29.
- `tiers.CAPACITY_CAP_MULTIPLIER` 2.0 -> 4.0 (s1-fast): cap 2560/s, 64 lanes (`max_inflight`), 864 requests per probe, verified sample 68 -> 114
  at the cap (17 of 54 s1-fast slots sat at 1280/s on 25 Oct 2026, 25 % of the pool weight). s1-lite / s1-cpu: every number as before,
  including their miner defaults (32 / 456 / 8, pending 64).
- `s1validator.PROBE_MAX_PREPARED_REQUESTS` 90 000 -> 220 000 and `PROBE_PREPARE_BUDGET_S` 60 -> 180: worst case, every slot at its x4 cap
  (55 s1-fast x 864 + 168 s1-pro x 704 + s1-lite / s1-cpu 16.2k = ~182k, x 1.2 growth margin = 218.5k) is prepared without scaling;
  at 0.34-0.78 ms per request a full ceiling is 75-172 s. The budget only pads the probe's outer timeout (preparation +
  window + references must fit in it with the rest of the probe's budget); it is not a per-phase limit, and the validator operator
  can change it with `REAX_PROBE_PREPARE_BUDGET_S` (float > 0, validator-local). Re-size from the shadow bench (`prepbench`).
- Miner defaults per model: s1-pro 64 / 728 / 16 (`REAX_PER_CALLER_CONCURRENCY` / `_PER_MINUTE` / `REAX_MAX_CONCURRENT`), s1-fast
  64 / 888 / 16, `REAX_MAX_PENDING_PER_IP` 128. Miners that consolidated hotkeys can show up to 2560/s per s1-fast and 640/s per
  s1-pro hotkey once the stake majority runs this release.

## 1.2.0rc11.post6 (validator, operational)
- Capacity probe verification off the round cadence (board #22 #26732). On mainnet a probe round took ~340 s instead of
  ~120 s (~21.5 rounds/h instead of ~29.5): the probe's s1-cpu references (2 per slot, the floor, for ~162 slots = ~320
  decisions, twice the pool's `REFERENCE_CAP`) ran on the single-lock llama.cpp runtime for ~130 s inside the round. Now
  the window and the signature check stay in the round; the reference verification of the sampled answers and the
  capacity fold run as one background task while the next rounds go on (`REAX_PROBE_BACKGROUND`, default on; `0` = the
  synchronous probe of rc11.post5).
  - Same items, same scoring, same fold: the sample is drawn before the task starts; budgets, scale and holds are those
    of the probe; only the time of the fold moves (typically 1-3 rounds later). Its report (incl. probe strikes for the
    EU ledger) appears in the summary of the first round after it finished.
  - The rounds' own references go first: the probe's references go to a runtime in chunks (16 decisions for s1-cpu /
    s1-lite, 256 for the GPU pools), and a chunk starts only while no round reference of that pool is in flight. The
    reference budget counts the chunks' own time; the wall clock is bounded at 4x that budget (then that pool's slots
    stay unmeasured by this probe, as after a reference timeout today).
  - One verification at a time: a probe that falls due meanwhile runs in the first round after it (no probe is lost).
    A slot whose hotkey or role changed, or that was dropped, before the fold is not folded.
  - No scoring constant, `REFERENCE_CAP`, tiers constant or `WEIGHTS_VERSION_KEY` changed. Expected probe round at
    today's slot counts ~215 s from the previous round's end (was ~340 s), ~25.5 rounds/h. tests/test_probe_cadence.py.

## 1.2.0rc11.post5 (unreleased) — EU tier: Scheme A x1.5 + EU contract tier, INACTIVE (empty registry)
- `eutier.py` + `eu_registry.json` (docs/EU-TIER.md): Scheme A (x1.5 on the in-pool score of verified EU hotkeys in
  s1-fast / s1-pro, capped per operator at `cap_units`) and the EU contract tier (EUR 1.00 per verified GPU-hour,
  `H_o = min(slots x 24, passed hours)`, daily alpha price, 5 % cap of the inference share with a pro rata cut, carved
  out of the miner part, no double pay, daily public receipt). Fail closed: no operator with a signed contract + DPA,
  no fresh EU verification, no price or no inference-share estimate -> no effect. The shipped registry lists nobody:
  weights byte-identical to rc11.post4 (tests/test_eutier.py against a b510d79 golden). `WEIGHTS_VERSION_KEY` unchanged.
- Vacant generalist track on eval v5.1 (#25186/#25193): with `CROWN_SLOTS_UTC` set, a v5.1 confirmation on a track without a
  champion is stored for the next slot (`crown_pending`) instead of waiting for the cohort playoff (not wired for v5.1: the track
  stayed uncrowned and blocked; a lone confirmation was crowned ad hoc). The slot crowns one entry alone or runs the v5.1 slot
  rounds for several, as on an occupied v5.1 track. Rule A: a vacant-track confirmation taken under v4 that still waits for its
  cohort after the flip restarts as a v5.1 first stage. Pre-flip (v4) vacant-track behaviour unchanged. tests/test_v51_vacant.py.
- Engine copy line, parent measured in the same session (Florian 24 Oct 13:02Z, card 18502, option A;
  `engcomp.ENGINE_PARENT_SESSION_START_UTC` = Tue 27 Oct 14:30Z, release constant): a derived entry whose session clears the
  champion bar but is below, or within the margin of, its stored parent line gets the binding parent image measured right after
  it (same box, same reference, the same items replayed via `engrunner.ItemTape`). That session's line =
  `max(parent in session, parent image's lowest valid score) x (1 + margin)` (`parent_bar(..., session)`); no valid parent run =
  the stored line. Reference runs never touch the parent's results/held score. New add-only state fields `low_scores`,
  `parent_sessions` (absent while empty; rc11.post4 ignores them). Golden tests unchanged; tests/test_engine_parent_session.py.
- Parent same-session fixes (parent-critic F3/F4/F5): the floor is the parent image's lowest VALID score (`hold(result, params)`
  records `low_scores` for valid results only; a state without it: min(held best, latest valid result)); a parent run is also
  due when the child scores >= `champion_bar / (1 + margin)` (slack for the champion's same-cycle re-measure; crowning uses the
  real bars); at most `ENGINE_PARENT_RUNS_PER_CYCLE` = 3 parent runs per engine cycle, each limited to
  min(`eval_budget_s`, `ENGINE_PARENT_RUN_BUDGET_S` = 1200 s), and a parent digest whose run failed or timed out earlier in the
  cycle is not run again in that cycle. The parent is measured right after the child on the same machine with the same items.
- New env (all optional): `REAX_EU_REGISTRY_FILE`, `REAX_EU_VERIFICATION_FILE`, `REAX_EU_MARKET_FILE`,
  `REAX_EU_TIER_STATE_PATH`, `REAX_EU_TIER_RECEIPTS_DIR`; `tools/eu_tier_quote.py` writes the daily market file.

## 1.2.0rc11.post4 — crowning slots on eval v5.1 from Thu 29 Oct; a shared group outlives its leader
- post4 critic fixes: after a new champion is crowned, every entry that has not confirmed yet (pending with a first draw, or a queued close call) also restarts with a new first draw (SLOT-RULES s.2 / FAQ); eval v5.1 slot rounds never reuse items: when no disjoint draw is left the rounds end early and the slot is decided on the completed rounds (SLOT-RULES s.4).

`WEIGHTS_VERSION_KEY` unchanged (1250). On top of rc11.post3 (slot rules + eval v5.1 armed for Mon 26 Oct 14:30Z).
- v5 slot hook (rc11p3d item F, slot rules section 3): `slot_eval_ts(slot)` = slot - 1 s for every slot (the rc11.post3
  `_v4_draw_now` pin is gone). The flip slot Mon 26 Oct 14:30Z (slot - 1 s before `EVAL_V5_START_UTC`) stays on v4 exactly as
  in rc11.post3; every later slot (`_slot_v5`: from Thu 29 Oct) is decided on eval v5.1:
  - each round is a v5.1 draw (`V5W.draw`: 540 T + 360 N + the public gap-check items, disjoint from the slot's earlier
    rounds, least-used items logged if the pool cannot give that), served question by question (`V5W.flatten`, `_serve_v5`);
  - `_slot_totals` returns the per-round S_v5.1 (`V5W.slot_context`: each member's absolute S_v5.1, no base, no reference)
    and their mean; the gap penalty is retired; the v4-scale entry floor does not apply (rule 4.2 is a no-op on v5.1, as the
    floor is in `_decide`);
  - rule 4.5 uses `_slot_pair_se_v5` = `V5W.slot_pair_se`: the v5.1 paired bootstrap SE per round (`DrawContext.pair_se`),
    pooled like the mean (`v5.pool_se`); margin = `required_margin` at `slot_eval_ts` (floor 1.0), G = 0.5 x that floor = 0.5;
  - rule 4.3 (parents and twins) compares S_v5.1 with the v5.1 pair SE (`_playoff_short(pse=)`), as `_decide` does;
  - members' scores, `by_tier {T, N}` (the v5.1 marker) and the slot `playoff` history rows carry `scoring_regime: "v5"`;
  - a member with no finite S_v5.1 on a round counts as failed on its own side (rule 4.1);
  - the evidence record says `evaluation: "v5.1"`, `pool_version` (v51.config.json), `panel` (panel stats version) and the
    sha256 of the v5.1 T/N item ids.
  - Fail closed (as rc11.post3's v5.1 arm): no usable v5.1 pool, config or panel stats -> no slot draw, nothing served,
    decided or crowned, `eval v5.1 FAIL CLOSED at the slot` in the log; the stored crownings stay and the slot runs when the
    pool is back (our side: slot rules section 7), also after the slot clock.
  - A stored crowning earned under v4 is never measured in a v5.1 slot: it restarts as a v5.1 first stage (rule A). A v4 slot
    still refuses entries decided under v5.1.
- K1-a: the confirmations run before a slot are measured on the slot's evaluation (`slot_eval_ts`). At the flip slot the
  rc11.post3 critic behaviour stays (M1): a due-but-unrun confirmation of a v4 first stage is not restarted by rule A while the
  slot is open; it runs on v4 at `slot_eval_ts` and may join the slot. From Thursday (v5.1 at slot - 1 s and now) K1-a runs on
  v5.1 and `_decide` keeps the `crown_pending` `k1a` marker; a v5.1-staged entry is never confirmed on v4.
- Void round 1 (rc11.post3 critic S4) holds for v4 and v5.1 slots alike: a round 1 in which every member fails (on v5.1 also:
  no finite S_v5.1 for any member) is void and re-run with the same entries, never released; after the slot clock an error is
  logged.
- CRITERIA 6 (rc11p3d section 4 item 3): a shared group crowned at a slot stays the champion when its leader withdraws or is
  rejected while other members remain. Validator: `cochamp_members` keeps the group in force while the track has no champion
  seat; worker: `_reanchor_cochamps` (after every sync and at the start of a track step) gives the seat to the first remaining
  member (status champion, `champion_since_ts` = the group's crowning, history row `cochamp_reanchor`). The group keeps its
  keys, so each member's part stays 1/n of the crowned group and a departed member's part goes to inference. The title is vacant
  only when the last member has left (the set is then cleared). The hard-coded opening set (116/199/250) is unchanged. This
  replaces the rc11.post3 critic's inline re-anchor in `_sync` (S1): one implementation, same result.
- Slot rules section 2 ("if a new champion is crowned, it has to beat the new champion, starting with a new first draw"):
  when a slot crowns (one champion or a shared group, also a single stored crowning), every entry of that slot that is not
  crowned and every entry whose crowning is stored for a later slot is queued for a new first draw against the new title
  holders (`_slot_restart`: stage dropped as for a requeue, history row `slot_restart`; history, commitment order and checks
  kept, no recommit). Before, they kept their first draw and confirmed against the new champion. A slot that crowns nobody
  changes nothing: its entries go back to the challenger path with their first draw, later-slot entries keep waiting. At the
  flip slot the restart comes from the crowning, so rule A finds no v4 stage left on these entries (one restart, first draw
  on v5.1). Payout effect: a requeued entry is QUEUED, so it no longer holds a pending-confirmation keep-alive.
- Tests: `tests/test_slot_v5.py` (incl. a v5.1 void round 1); three tests of the v5.1 wiring written against rc11.post2 slot semantics now follow the
  rc11.post3 rules (a slot entry not crowned goes back to the challenger path instead of not-better; an unloadable slot field
  is recorded and stays pending).

## 1.2.0rc11.post3 — crowning slot rules: clear winner or shared leading group (public slot rules, FAQ v2)

`WEIGHTS_VERSION_KEY` unchanged (1250). Implements reax.co/competitions/faq/#slot-rules sections 1-9 (CRITERIA v2, Florian's
decisions of 22 Oct: card 18227, tie-numbers-k). Every knob is one constant in `competition_split` (`SLOT_*`).
- `CROWN_SLOTS_UTC`: every Monday and Thursday 14:30Z from Mon 26 Oct 2026, generated through 31 Mar 2027.
- Slot rounds (worker): every confirmed entry is in (no budget cut). Up to `SLOT_MAX_ROUNDS` = 3 shared fresh rounds, all on the
  slot's pinned evaluation time (`slot_eval_ts` = slot - 1 s). After every round, in this order: serving (a model failure is
  retried once on the same items; a second failure while others were served: the entry leaves), entry floor (pooled mean at or
  below it), parents/twins and one entry per holder (same coldkey or same HF account/org, transitively; best round-1 score
  stays, exact tie: earlier commitment) on round 1 only, separation (leader ahead by at least `required_margin(pooled paired
  SE)` = max(1.0, 2 SE) on v4). One entry left: clear winner, no further round. After round 3 (or fewer when the slot clock /
  void retries / pool end the rounds): every remaining entry within G = 0.5 x margin floor of the leader (unrounded, no
  chaining) shares; nobody within G: the leader alone (K8b). The rc11.post2 highest-mean tiebreak is removed.
- A round where every member fails or a host fault is flagged is void and re-run on new items (rounds 2-3: at most
  `SLOT_VOID_RETRIES` = 2 times; round 1: next cycle). Slot clock `SLOT_CLOCK_S` = 12 h.
- K1-a: a confirmation due (first pass + confirmation delay) at or before slot - 1 h that has not run by the slot runs before
  the slot rounds (the slot waits, up to the slot clock) and, if it passes, joins this slot (`crown_pending` row `k1a`).
- Entries not crowned go back to the challenger path (they confirm against the new champion with the full margin); no place.
  Nobody crowned: only the slot's own entries are released, entries stored for a later slot keep waiting. A field whose
  every entry fails to load no longer blocks the track (rc11.post2 critic F2).
- A shared group is stored in the state (`TrackState.cochamps`, `cochamps_since`; add-only, an older loader ignores it) and
  is the co-champion set in force (served as the bar, best member on the challenger's draw; 1/n of rank 1 each). Beaten, it
  becomes ONE former champion (`TrackState.former_sets`; validator `RankedLeaders.formsets`, 1/n of the place each).
- Keep-share (slot rules section 8): for up to 48 h the beaten champion (or beaten set, in its own split) receives the parts
  of rank 1 that belong to unapproved members, instead of its own place share. Fixes the rc11.post2 path that gave a beaten
  set's whole keep-share to one member. The opening set at rank 1 keeps the rc11.post2 behaviour (no keep-share).
- Public evidence per slot in `TrackState.slot_results` (aggregates only: entries, how each got in, round scores, pooled
  means, gap / paired SE / margin against the leader, G, the rule each entry left by, draw ids, evaluation, release,
  pool sha256, result, shares).

## 1.2.0rc11.post2 — crown slot without freeze, slot playoff, the beaten co-champion set is one former champion

`WEIGHTS_VERSION_KEY` unchanged (1250). Florian's rule A (22 Oct): an entry that confirms under the current evaluation before
Mon 26 Oct 14:30Z is crowned at that slot as earned; afterwards titles are defended on v5. Ties: more draws, never commit order.
- Worker, no freeze: a stored crowning (`crown_pending` history row) no longer stops the other challengers. Each keeps being
  evaluated against the current bar (the best co-champion on its own draw); every entry that passes is stored. A stored entry is
  not re-challenged and stays keep-alive eligible as a pending entry.
- Worker, at the slot: one earned entry is crowned as earned. Several: a slot playoff on ONE shared fresh draw (the champion and
  the co-champions are not in it), same preparation / budget / infra-abort / behaviour-twin rules as the cohort playoff, which
  waits for the full budget. If the best two are within `PLAYOFF_MIN_GAP_SE` paired SE (pooled over the draws taken), the tied
  group takes an extension draw, and again a second one; after three draws the highest mean wins. All three draws run in one
  call. Every score and SE is in the public event log. Others are not better (served) or back on the challenger path.
- Crowning a challenger makes all listed co-champions FORMER (not only the state champion).
- Validator (`GENERALIST_CO_CHAMPIONS`): the beaten set is ONE former champion. It takes one former-champion place (ordered by
  `GENERALIST_CO_CHAMPIONS_CROWNED`) and that place's share (17 % / 8 %) is split 1/3 over its committed, approved members
  (`RankedLeaders.formset`, `rank_shares`); a dropped member's third is not redistributed. Also read from a state where only
  the champion member is former and the others are still pending (rc11.post1 worker). Empty constant = unchanged.

## Unreleased — eval v5.1 evaluator wiring (behind `EVAL_V5_START_UTC = None`, rc11 bit for bit while it is `None`)

New `spec/v5wire.py`; `v5.EVALUATOR_WIRED = True` (the evaluator may start with the gate set). With the gate reached the generalist
track draws 540 T + 360 N private items (+ 90 public items at weight 0) from `REAX_SPEC_V5_POOL_DIR`, serves the seated
co-champions and every challenger's declared base on the same items (a base shared by entries once per draw), decides on S_v5.1
against the best of them (D3), records S_v5.1 / T / N / the gap check in the history (`scoring_regime = "v5"`, no gap penalty),
and starts the provisional half keep-alive at the on-chain commitment (D4). No new state field (an entry scored under v5 has `by_tier == {T, N}`).
See docs/EVAL-V5.md, "Second serving slot and evaluator wiring".
Transition rule A (Florian, 22 Oct): a v4 confirmation before the flip is crowned at the crown slot (not re-judged; crown first, then v5;
several earned entries play the rc11.post2 slot playoff, which still draws under v4 at the flip slot: `_v4_draw_now`); v4 first
stages unconfirmed at the flip restart as v5.1 first stages (`_v5_restart_v4_stages`, history row `v5_restart`). The v5.1 bar is the
CURRENT title holders only (all seated co-champions while they hold the title; after a rule-A crowning the new champion alone) plus
the challenger's own base. `v5.paired_s_se` is public.
wiring-3: questions above 26 options are capped in the v5 draw (`v5wire.cap_item`: gold + 25 distractors from the draw seed, item id
and question index, re-lettered A..Z, the option count in the task text replaced by 26; with IDK 25 + IDK; chance-correction uses the
capped count); joint-readout packages go through `JointDockerBackend` on v5 draws (tests with fake backends); a crowning inside one
`_decide` loop becomes part of the bar of the later challengers of the group; the v5 floor exemption needs the open gate and a v5
marker that only challengers, the champion and the co-champions get (never a parent), `_big_ok` keeps its parameter checks; an
unavailable or unservable base, or a non-finite S, holds back only its own challenger; the served-result cache is keyed by the item
set and cleared every cycle; the v4 generalist floor still holds for entries without the v5 marker; the pool is sorted by id; the
half keep-alive from the commitment needs a successful repo probe first (exists, public, revision, safetensors listed).

## 1.2.0rc11.post1 — generalist opening co-champions 116 / 199 / 250 (validator + competition worker)

`WEIGHTS_VERSION_KEY` is unchanged (1250). One source for the validator and the competition worker. Florian's decision
(22 Oct 2026): uid 116, 199 and 250 are co-champions of the generalist opening, equal thirds of the champion share until a
new entry beats the best of them; places 2 and 3 of the opening are uid 160 (17 %) and uid 113 (8 %) from 23 Oct 13:00 UTC.
- `GENERALIST_CO_CHAMPIONS` (the three `hotkey|commitment` keys of the state) and `GENERALIST_CO_CHAMPIONS_CROWNED`: while the
  generalist champion is one of them, each member that is still committed, not withdrawn or rejected and has its own owner
  approval gets one third of rank 1's share (`rank_shares`, `RankedLeaders.cochamps`). An unapproved or dropped member's
  third goes to the inference part. Members take no keep-alive slot while paid (an unapproved member keeps the champion-like
  slot). Empty constant = the weights are exactly the rc11 ones.
- Opening-playoff places 2 and 3 (rules v4, from T1) skip the co-champions: the next two in the playoff's own order.
- Worker: a challenger's bar is the best co-champion on the challenger's own draw (all still-eligible members are served on
  it; the extra serves count against `max_evals_per_hour`). Beating it, with the normal margin and confirmation, makes the
  challenger champion and ends the set. Members are never served as challengers. The event log and the history rows show
  "vs best co-champion uid X", the bar, the gain and the SE.
- `CROWN_SLOTS_UTC` (ships empty; stored as a history row, no new state field): when set, a decision that would crown is stored (scores published, champion unchanged,
  the track's challengers frozen) and finalised at the first slot instant at or after the decision.
- `tools/cochamp_dryrun.py`: prints the generalist payout for a state file and a snapshot JSON.

## 1.2.0rc11 — rules v4 active from 23 Oct 2026 13:00 UTC; rc10.post2-5 hotfixes folded in (validator + competition worker)

`WEIGHTS_VERSION_KEY` is unchanged (1250). One source for the validator and the competition worker. Release constants
`CONFIRM_BEFORE_CLOSE_START_UTC`, `DRAWS_V4_START_UTC`, `COLDKEY_DR_START_UTC`, `ENGINE_COPYLINE_START_UTC` and
`OPENING_PLAYOFF_SLOTS_START_UTC` are all set to 2026-10-23T13:00:00Z (before that instant the weights are exactly the
rc10 ones). Includes rc10.post2 (nonce retention), post3/post4 (cohort-first queue, re-entry of pruned opening-cohort
entries), post5 (cohort keep-alive protection) and `certifi==2026.10.19` in `requirements/lock.txt` (eMudhra CA, #22935).
The reax-1 pool is not part of this release.

## 1.2.0rc10.post2 — capacity probe: no false 'stale' answers with many scoring slots (validator)

`WEIGHTS_VERSION_KEY` is unchanged (1250). The validator checks the miners' signed probe answers after the probe window, one
scoring slot after the other, and accepted an answer only while it was younger than its nonce store's retention
(request timeout + 60 s). With ~310 slots (multi-model hotkeys) the check took longer than that, so the slots checked
last were rejected as stale and their capacity probe counted 0. The nonce store now keeps nonces for at least 600 s.
Miners need no update.

## 1.2.0rc10.post1 — engine former champions: 75/17/8 with 7-day slots, co-champions as one former champion (validator)

`WEIGHTS_VERSION_KEY` is unchanged (1250). Owner decisions 21 Oct 2026. Of the engine champion's part (after the decay),
the previous champion gets 17 % and the one before 8 % while their slot runs (7 days = 50,400 blocks from the block they
lost the title, or until two newer champions push them out); the champion keeps the rest (no former champion in a slot =
exactly the rule before). Each former champion needs its own owner approval, must still be committed by its hotkey and
hold a valid result; an unpaid slot goes to the inference part. While a new champion waits for its approval, the most
recently crowned approved former champion takes the champion's part (its own decay; its own slot is not paid on top and
not held against it, so with no other slot held it gets the whole part). The opening co-champions (uid 4, uid 198) count
as one former champion, each half with its own approval. The protected engine keep-alive of 1.2.0rc9.post8
(`REAX_ENGINE_KEEPALIVE`) now covers the champion awaiting approval and every former champion in its slot (at most 4,
each until 7 days after it lost the title); a uid paid anything gets none. Validator-only: miners need no update.

## 1.2.0rc10 — one hotkey, several roles: all inference models + one competition entry (consensus; start block <START_BLOCK>)

Validator + miner release. Both halves count only from release constants `elements.MULTIMODEL_START_BLOCK` /
`DUALROLE_START_BLOCK` (None = held); before that every commitment is read exactly as today. `WEIGHTS_VERSION_KEY` 1250. Start block: <START_BLOCK>. Design: docs/DESIGN-MULTIROLE.md.

* Cap multiplier per pool (`tiers.CAPACITY_CAP_MULTIPLIERS`): s1-fast stays x2 (1,280 decisions/s); s1-pro has its own
  switch `tiers.S1_PRO_CAP_MULTIPLIER`, shipped at 2.0 (320/s, every weight identical to before) and flipped to 4.0 (640/s,
  64 lanes, 704 requests per probe, verified sample 29) by the release owner after the shadow bench, with a
  `WEIGHTS_VERSION_KEY` bump and a dated #92 notice. Miner defaults follow each model's own pool (`config.probe_defaults`):
  s1-pro at x4 gives 64 / 728 / 16 (`REAX_PER_CALLER_CONCURRENCY` / `_PER_MINUTE` / `REAX_MAX_CONCURRENT`), s1-fast 32 / 456 / 8.
* Commitments: up to two Raw fields, one inference element (`reax-s1/1`, or new `reax-s1m/1` for 2-4 models) and one
  competition element (`reax-spec/1`, `reax-eng/1`). Strict parser in `elements.py`; one-field commitments unchanged.
* Validator: one scoring slot per (uid, model) (`uid + 65536 x model index`; s1-fast slot = uid) with the same cap per
  model as a separate hotkey; all slots probed in the one common window; slot weights summed per uid. Legacy state
  migrates on load without a reset; `scripts/state_unslot.py` writes it back before a rollback to an older release.
  The competition code gets the competition element only (`role_views`), so an inference rewrite never re-queues an
  entry and a new entry never resets the inference score.
* Pay: keep-alive is a top-up for entrants that also serve (`max(0, each - own serving)`), pure entrants take the 8
  slots first, keep-alive candidates stay out of the median; provisional keep-alive for pure entrants only. Without a
  serving entrant the weights are bit-identical to before (golden test).
* Miner: `REAX_MODELS=s1-fast,s1-pro,s1-lite` serves several models behind one endpoint (per-model slots and
  per-validator limits); commitment publishing touches only the inference element, keeps any other (also unknown
  `reax-`) element and writes only after a confirming second read. New `reax-commit show|set|withdraw` (always merges);
  `reax-engine-submit` merges too.
* Router: routes a multi-model hotkey for each of its models (score at the slot key).
* Legacy keep-alive (Florian 21 Oct 07:05Z): for 7 days from the generalist start, the ended demand_forecasting and
  risk_assessment tracks' champions and owner-approved former champions that still commit to their entry get 0.5 x the
  median serving weight (pure entrants only, at most 12 uids, together at most 3 % of the miner part, from the inference
  part; `LEGACY_KA_*`). Read from the frozen DF/RA part of the spec state; no new state key. Off or empty: weights
  byte-identical (fuzz 3 x 20k). Tests: test_legacyka.
* Tests: test_elements, test_multimodel, test_dualrole, test_miner_multimodel, test_state_unslot; localnet
  `scripts/localnet/e2e_multirole.py`.


## Registration-sniping protection (part of 1.2.0rc10; spec tracks, all tracks' waiting entries)

Validator/competition-worker change (DF/RA), reg-sniping review 20 Oct 2026. `WEIGHTS_VERSION_KEY` unchanged.
* Near-duplicate check: withdrawn entries (pruned or replaced) of another coldkey and repo are compared too; a derived
  entry gets them as its parent (must beat them). Their sketches are renewed like live entries' after a seed change.
  The author's own withdrawn entries are skipped (re-registration, carry-over).
* Exact check: an earlier original of another coldkey that is checked after its exact copy was crowned rejects the copy
  (title cleared, the cycle treats the track as vacant); before, the crowned copy kept the title and the original was
  refused.
* Provisional keep-alive: a QUEUED close call waiting for its extra draw (scored, `close_draws`) stays eligible until its
  verdict (uid 85 on 20 Oct earned nothing between its first draw and its verdict).
* `IMMUNITY_BLOCKS_ASSUMED` 5000 -> 7200 (the chain's ImmunityPeriod since ~block 9,305,000, 17 Oct): the provisional
  window opened ~2,200 blocks too early and expired ~2.7 h after the real immunity end; the queue order used the
  wrong end too.

* Engine keep-alive (s1-fast, `ENGINE_KA_*`, off unless `engine_keepalive`): also covers an unpulled entry whose hotkey is the
  digest's first committer and the pending challenger between its first and confirming session; the window is per hotkey
  (first sighting of any of its digests) and starts at the end of the immunity (`CHAIN_IMMUNITY_BLOCKS` 7200 minus the lead).
* Engine fairness: candidates within the displacement margin of the best score are a tie on merit and the earliest
  commitment (persisted first sighting) wins, also against a pending later challenger; `pending()` orders by commitment
  order; the first inconclusive (infra) retry keeps its place, from the second one on the entry goes behind the untried
  entries, so a few slow or oversized images cannot hold the head of the queue for `max_infra_attempts` cycles. State
  format unchanged.
* One clock read per `EngineCompetition.weights` computation (track list, profile and payout agree at the
  `GENERALIST_START_UTC` switch). `scripts/set_generalist_start.py` keeps seconds and checks `COMMIT_FROM <= START` also
  when only `COMMIT_FROM` is given.
* Generalist: the spec-track rules above (copy registry incl. withdrawn entrants, crowned-copy rejection, close-call
  provisional keep-alive, 7200 immunity) apply to it unchanged; its entry-floor/G_med fail-closed guard still comes first.

## 1.2.0rc9.post7 — engine opening co-champions (validator)

`WEIGHTS_VERSION_KEY` is unchanged (1240). One-time exception for the engine opening round (owner decision 20 Oct 2026):
uid 4 and uid 198 measured within the margin of each other and share the s1-fast title. While one of them holds the
title, the champion's part of the engine share (decaying from the crowning) is split 50/50; each half is paid only to an
entry that is still committed by its hotkey, has a valid result and its own owner approval. An unpaid half goes to the
inference part. A challenger must beat 2.8233 (uid 4's opening score) by the margin. Once a challenger takes the title
the normal rule applies again. Miners need no update.

## 1.2.0rc9.post6 — s1-lite brake: tier-order cap order_ratio 1.0 -> 0.5 (validator)

`WEIGHTS_VERSION_KEY` is unchanged (1240). The average weight per miner in a tier may again be at most half of the average
per miner in the tier above (as before 1.2.0rc7.post4); the excess moves up to the tier above, never to a lower tier and
never burned. Tier budgets are unchanged. Miners need no update.

## Unreleased — several machines behind one inference hotkey (miner-only, `REAX_FEATURE_MULTINODE`, default off)

Validators, scoring, weights and `WEIGHTS_VERSION_KEY` are unchanged; with the flag unset the miner behaves as in
1.2.0rc9.post5 (one fix below applies to `REAX_<MODEL>_RUNTIME_URLS` lists too). Design: docs/DESIGN-MULTINODE.md.

* New `reax-runtime-relay` for worker machines: forwards the gateway's signed requests to the loopback sidecar.
  Per-request HMAC-SHA256 with a per-worker link key (10 s freshness, replay cache), every answer signed and bound
  to its request, TLS with a self-signed certificate pinned by fingerprint on the gateway, or plain http only on
  private/VPN addresses (10/8, 172.16/12, 192.168/16, 100.64/10, fc00::/7). Workers never see the hotkey.
* Gateway (`reax-miner`, `REAX_FEATURE_MULTINODE=1`, s1-fast and s1-pro only): workers from `REAX_WORKERS_FILE`,
  re-read every 10 s. New workers take validator traffic only after two good health checks and the identity check;
  removed or edited workers are retired make-before-break (an edited entry serves until its replacement is in
  rotation; a removed one while fewer than `REAX_WORKERS_MIN_HEALTHY` others serve) and drained. Key files
  rewritten in place are noticed. `{"name": ..., "evict": true}` takes a worker out at once. The miner starts when
  `REAX_WORKERS_MIN_HEALTHY` (default 1) workers pass.
* Link 401/403, unsigned or changed answers count as worker faults and fail over to another worker.
* Fix (also for `REAX_<MODEL>_RUNTIME_URLS`): a backend taken out of rotation while one of its requests was in flight
  is no longer re-admitted by that request's late success; it returns only through the health and identity checks.
* Docs: MINER.md "Several machines in different places behind one hotkey", CONFIG.md, FEATURE-FLAGS.md,
  THREAT-MODEL.md S15. Tests: `tests/test_multinode.py`; localnet `scripts/localnet/e2e_multinode.py`.

## Generalist release preparation (not deployed)

* Merge the live rc9p5-final engine/validator, FLA publication and queue-retry fixes, plus required host UDS creation.
* Replace DF/RA with one generalist slot at `GENERALIST_START_UTC` (target 30 Oct 12:00 UTC, movable earlier).
  Cancel the 6 Nov extension and DF/RA open class; larger models belong to generalist only.
* Add 300-item category/type/tier draws, immutable category bindings, and diagnostic category scores pooled per entry.
  Scoring and decision formulas are unchanged. Generalist waits for release calibration; both images must be updated.
* See `docs/GENERALIST-RELEASE.md` for the required settings, defaults and remaining release gates.

## Specialist open class (labelled 1.2.0rc9.post6 on the generalist branch; part of 1.2.0rc10) from 22 Oct 18:00 UTC (bigger bases), first track pair runs to 6 Nov

`WEIGHTS_VERSION_KEY` is set by the release (1250 in rc10). **Operator note:** the open-class box needs `REAX_SPEC_GPU_BUDGET_GB` >= 81,
`REAX_SPEC_MIN_FREE_GB` >= 150, `REAX_SPEC_BASE_BUDGET_GB` 80, `REAX_SPEC_CONTAINER_MEM` about `72g` and
`REAX_SPEC_STARTUP_TIMEOUT_S` 1800, and **validators must run this release too** (keep-alive base floor and the schedule to 6 Nov are
computed by the validator). Before `SPEC_OPEN_CLASS_START_UTC` (22 Oct 2026, 18:00 UTC, `competition_split.py`) the
specialist caps, architecture allowlist, base caps, serving flags and keep-alive are those of 1.2.0rc9.post5.

* **Caps from the switch.** Up to 36 B parameters, 68 GiB of weights, 72 GiB in total (before: 9 B, 20 GiB, 22 GiB). A declared
  base has the same explicit caps (before: twice a submission's). `spec.rules.rules_at(now)` returns the rules in force; the
  evaluator uses its clock, the base fetcher and the miner CLI the wall clock. `SpecConfig.rules` is now `None` by default
  (the release rules at the time of the check); an explicit `TrackRules` still wins.
* **Architectures from the switch.** `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM`,
  `Gemma4UnifiedForConditionalGeneration` (new fixed template `gemma4.jinja`, derived from the s1-pro Gemma 4 template with
  thinking off, a fixed `<bos>` instead of the tokenizer's `bos_token`; the tokenizer must define `<bos>`, `<|turn>`, `<turn|>`,
  `<|channel>`, `<channel|>`) and
  `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM` (`qwen.jinja`). Multimodal checkpoints of these new
  architectures are served text-only (`--limit-mm-per-prompt '{"image":0}'`). An architecture allowed before (e.g.
  `Qwen3_5ForConditionalGeneration`) keeps exactly its old command line up to 20 GiB of weights and is served text-only above that.
* **Serving resources.** `REAX_SPEC_CONTAINER_MEM` (default `48g`, `<int>g`, 8..512) sets the container's `--memory` and
  `--memory-swap`; `REAX_SPEC_STARTUP_TIMEOUT_S` (default 900, 60..1800) the start-up deadline. A bad value stops the worker
  at start-up.
* **Keep-alive base floor.** From the switch an entry above 10 B parameters gets a keep-alive only if
  `KEEPALIVE_BASE_FLOOR` (base tensor hash -> floor per track) has its declared base and its qualifying score is above the
  floor + 1.5; a fetched entry above 10 B gets no provisional keep-alive either. The table ships empty, so for now no entry above 10 B gets either. The parameter
  count (`Submission.params`, new) is recorded at the fetch; an entry without one (fetched by an earlier release under the 9 B
  cap, or not fetched yet) counts as small, and a larger entry without a recorded base hash gets nothing. Entries of 10 B or
  less are unchanged.
* **Host misfits are never the miner's.** Weights above 20 GiB that do not fit `REAX_SPEC_GPU_BUDGET_GB` raise a host outcome
  (the entry waits, no strike, no rejection) instead of a model start-up failure; up to 20 GiB (25 GB needed, the default budget)
  the classification is unchanged. A warning is logged once per cycle from 24 h before the switch while the budget is below
  80.2 GB. Before a submission is downloaded in the open class, its listed size must leave `REAX_SPEC_MIN_FREE_GB` free, otherwise
  it waits for this host (no strike) and no further download starts that cycle. Before a base is refused for disk, cached bases
  that no live submission declares (and that are not preseeded) are removed, oldest first, when that makes room.
* **Schedule.** The first track pair (`demand_forecasting`, `risk_assessment`) now ends **6 Nov 2026, 12:00 UTC** (was 30 Oct).

## Unreleased — specialist queue

* Specialist queue: check waiting repositories with metadata-only probes each cycle (up to 32 per track, oldest
  commitment first). Public revisions return to their original queue priority; unavailable repos stay skipped without
  spending the preparation skip allowance. State format, evaluation budgets and scoring rules are unchanged.

## 1.2.0rc9.post4 — engine track hardening (engine stays HELD), near-champion rule and keep-share (flags off)

`WEIGHTS_VERSION_KEY` is unchanged (1240). On finney the engine track stays held in code, the specialist path is unchanged, and
with the new flags unset the decisions and weights are those of 1.2.0rc9.post3.

### Engine track (held; no effect on weights while it is held)
* Engine runner hardening. Release mode (finney) refuses to start any engine unless `REAX_ENGINE_DOCKER_RUNTIME=runsc` and
  `docker info` lists the runtime (void session, never a miner strike). Engine containers get `/tmp` with `nosuid,nodev`
  (exec kept) and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. `REAX_ENGINE_NVML_CHECK` is ignored under runsc (it cannot
  match the sandbox's PIDs). Optional `REAX_ENGINE_CYCLE_GATE_FILE` / `REAX_ENGINE_CYCLE_GATE_MAX_AGE_MIN` bracket cycles
  and a cycle writes a `cycle_end` event. New pre-run submission check (`engreview.py`): image config and tokenizer/config
  files are screened once per digest and recorded in `<review dir>/reviews/<digest>.json`; in release mode a digest runs
  only after the operator created `<review dir>/REVIEWED/<digest>`.
* Engine containers receive `FLA_CACHE_MODE=strict` and a fixed per-GPU-class `FLA_CONFIG_DIR`. The measured configs of the
  RTX PRO 6000 Blackwell Server Edition are published (`runtimes/s1-fast/fla-configs/`, pinned by SHA256SUMS hash) and
  bind-mounted read-only into every engine (reference image 9bd41632 ships none); checksum mismatch or any other GPU
  class makes the session VOID. `PINNED_REFERENCE_NOISE_STD` = 0.00167 (new frozen rules hash). The Workstation Edition
  stays unpublished.
* Strict-mode reference startup/request paths reject missing FLA config keys instead of falling back to autotuning.
* Before a candidate starts, the reference answers a fixed check set twice. Failed self-checks produce a persisted
  `session_void` event and no miner score, strike or infrastructure-attempt increment.
* One invalid champion measurement triggers a fresh-session re-check; only two consecutive invalid sessions
  vacate the title. A valid session resets the streak; a duplicate session does not increment it.
* Minimum gain and champion/challenger displacement margin are at least `max(0.03, 3 * reference_noise_std)`.
  Noise is the measured relative standard deviation of fresh-process pinned-reference throughput. It is included
  in the frozen rules hash; now measured (0.00167) and published. Fresh-session confirmation stays.
* Engine track code gates: a digest's image owner is recorded when it is first verified and a second hotkey committing a cached
  digest is excluded (the champion is matched by digest and hotkey); an owner-lookup timeout is inconclusive, not a failed
  session. Saved results and champions carry a measurement context (reference image, weights file manifest, frozen track
  parameters, chain/netuid); a changed or missing context drops them so they are re-measured and re-confirmed before they pay.
  Cancelling an evaluation reaps the subprocess and removes a half-started container and its session directory. Track
  parameter overrides reject zero counts, non-positive and non-finite timeouts. The engine track stays HELD.
* Critic fixes. An image that waits for its `REVIEWED/<digest>` marker is removed again (the record keeps the verdict; it is
  pulled again once the marker exists); an `OSError` of the submission check (review directory, disk) is a host fault and counts
  nothing against the miner; release mode without a review directory voids instead of skipping the gate. A missing, empty or
  unreadable weights directory no longer looks like a context change: the saved state is kept, no cycle runs and no engine
  share is paid until it is readable. The weights identity also covers the content of the small JSON files and the safetensors
  headers plus a 64 KiB sample of the tensor data at both ends. The chain id is normalised like the config (every mainnet
  spelling is `finney`, whitespace stripped). `runsc` is accepted only when the daemon's runtime path is a `runsc` binary with
  `--network=none` and `--nvproxy=true` in its `runtimeArgs`. In the static check a bare file name is compared only in model
  directories (`/model`, `/models`, `/opt/model`), a reference image that disagrees with the weights directory voids the
  session, no files are extracted from an image whose configuration is already rejected, and a copied file is cut off at the
  size cap. A rejected record can be cleared by deleting `<review dir>/reviews/<digest>.json`. `cycle_end` and `session_void`
  events live in their own capped list (`cycle_events`), so they cannot push `crowned`/`vacated` out of the history. A
  whole-budget timeout before the first timed request is an inconclusive attempt (bounded by `max_infra_attempts`), not a
  failed measurement; after measurement began it still fails as before.

### Near-champion rule and keep-share (`REAX_SPEC_NC_NEAR_CHAMP`, `REAX_SPEC_NC_KEEP_SHARE`; default off)
- New `REAX_SPEC_NC_NEAR_CHAMP` (competition worker, default off): a challenger whose weights are close to the incumbent's
  (delta sketch cos > 0.995, or relative L2 < 0.15, or cos > 0.98 with alpha within 0.05 of 1) must, on every deciding draw
  against it, have more items whose most probable label is the gold one (`ItemScore.top_hit`, rescaling-proof) than the
  incumbent and gain at least max(margin, 3.0) points (`NEAR_CHAMP_*`). History rows `kind: near_champ`. A challenger on
  another declared base is sketched again from its served weights against the champion's base (critic-nc F1). If no
  comparison can be made (champion sketch or weights missing) the draw is recorded `comparable: false` and the old rules
  decide it. Vacant-track playoffs are unaffected.
- New `REAX_SPEC_NC_KEEP_SHARE` (validator, default off; consensus-relevant, same value on every validator): while a track's
  new champion has no owner approval and was crowned less than 48 h ago, its rank-1 share goes to the most recently crowned
  approved former champion, who gives up its own rank share. `RankedLeaders.crowned` carries the crowning times.
- Near-copy rules R1/R2/R3 (`REAX_SPEC_NC_R1/R2/R3`, default off) are in the code for later use; not part of the 19 Oct decision.
- All flags off: decisions and weights are bit for bit those of 1.2.0rc9.post3.

## 1.2.0rc9.post3 — X tier binding mode and capped generalisation penalty (competition worker only; unreleased)

- New `REAX_SPEC_X_TIER_MODE=binding` (default stays `off`): 20 % of each 300-item draw (including playoff draws) comes from
  the cross-author tier (`cross_author`, and `award_realdata` rows count as the same tier) and counts in the score; the
  reuse cap and disjointness rules are unchanged, a shortage is filled from the normal tiers and logged, and the refill
  alarm counts cross-author items separately.
- Every decision uses `S_eff = S - P` with `P = min(CAP, 0.5 x max(0, gap - G_med - A))`, `A = max(8, 2 x SE)` from one
  draw (not pooled), `G_med` frozen per track in `REAX_SPEC_G_MED` (JSON; a track without one runs as shadow) and
  `CAP` from `REAX_SPEC_GAP_PENALTY_CAP` (default 10, `none`/`inf` lifts it). Each history row records `gap_binding`.
  `scoring.gap_penalty` now takes the allowance from one draw and has an optional `cap`; the shadow report follows it.
- `REAX_SPEC_V3_DECISIONS` keeps the strict parsing of rc9.post2 (empty = off, unknown value stops the worker at start-up).
- Fix (critic-v3int #3): a recommit that takes over a pruned entry waiting for its extra confirmation now keeps `confirm_draws` and the confirmation rows (gap fields included), so the extra-confirmation allowance, the draw exclusions and the binding penalty are unchanged by the recommit.
- Fix (critic-v3int #4): an entry that is held (PENDING_VERIFY) and queued again starts a fresh stage: `close_draws`/`confirm_draws` are cleared, so it gets no close-call priority and no endless close-draw appends.
- Fix (critic-v3int #4): v3 pools only the rows of the current stage (first stage: `draw1` + `close_draws`; confirmation: `confirm_draw` + `confirm_draws`), and `machine.confirm_v3` takes (gain, SE) pairs so SEs and gains cannot misalign.
- Fix (critic-v3int #2): with `REAX_SPEC_V3_DECISIONS` off, a draw still excludes the items of `close_draws`/`confirm_draws` already in the state, so turning v3 off mid-flight never reuses them.
- Off and shadow behave as before for new draws; state files with the new history fields load in old and new builds.

## 1.2.0rc9.post2 — evaluation v3 staged (competition worker only; weights version 1240 unchanged)

- Integrate v3 stage 1 behind `REAX_SPEC_V3_DECISIONS=0` and the cross-author plumbing behind
  `REAX_SPEC_X_TIER_MODE=off`, both off by default. The current decision rule remains in effect until activation.
- If the incumbent cannot start, including after its immediate retry, abandon the comparison without charging
  item uses or recording a consumed draw for challengers. Serving attempts and round outcomes still count.
- Preserve the rc9.post1 start-up window, champion retry and host probe, infrastructure playoff bound and liveness signal.

* **Decision rule v3** (`REAX_SPEC_V3_DECISIONS`, default off; `0` preserves the old rule exactly, no state migration):
  a race is decided on evidence in standard errors pooled over the draws against the same incumbent. First draw `z >= 2`
  and gain >= 1.5 passes as before; `1 <= z < 2` with gain >= 0.75 gets ONE more disjoint first-stage draw (the 12 h wait still counts from
  the first draw); confirmation crowns at pooled `z >= 2.75` with confirmation gain at least 0.5 SE; positive confirmation gain and pooled z >= 1.5 get ONE
  extra confirmation draw when the crown conditions are not yet met. Against a new incumbent (crowned meanwhile or earlier in the same round) only the draws against
  it count, same thresholds, so a tie inside the noise goes to the earlier commitment. Copy/parent/twin checks unchanged.
  Entries that passed their first draw before the switch are confirmed under the old rule (`decision_rule` in the state).
* State: new optional submission fields `decision_rule`, `close_draws`, `confirm_draws`; with v3 enabled, history rows gain `se`, `inc_key`,
  `z`, `pooled_z`. Old states load; an older worker ignores the new fields.
* **Stage 2 plumbing, off by default:** optional `cross_author` item tier (`REAX_SPEC_X_TIER_MODE` off|shadow|scored,
  `REAX_SPEC_X_TIER_SHARE` 0.20; `scored` currently falls back to `shadow`), per-draw `s_gen`/`s_x`, a pure `gap_penalty` function and a `gap_shadow` report. Nothing is
  applied to any score; a pool without the tier draws exactly the same items as before.
- `REAX_SPEC_V3_DECISIONS` parses strictly: unset or empty means off (the release default), 1/0, true/false, yes/no and on/off are accepted, anything else stops the worker at start-up.

## 1.2.0rc9.post1 — start-up reliability (competition worker only; weights version 1240 unchanged)

- Model start-up failures count over the last six scheduled rounds: three model failures reject the entry and cost
  a champion its crown. A successful serve no longer resets the counter. The last twelve round outcomes persist.
- A champion gets one immediate retry on the same draw. If it still cannot start, the comparison is abandoned and
  it is re-measured next cycle, including when no challenger remains. The retry counts toward the hourly budget.
- Hub/network/download failures, Docker errors, busy GPUs, disk errors and whole-group failures are infrastructure
  failures, never model strikes. Missing/private repos are skipped and retried. Hash-verified entries, including
  champions, are rejected as no longer available after three 401/403/404 or revision-missing refetches at least ten
  minutes apart; this separate bound never counts network errors, timeouts or 5xx. Waiting repos cannot block
  confirmation groups.
- After three infrastructure playoff aborts, decide among entries served. Others keep their first draw and return
  to the challenger path without a strike. If nobody serves, there is no decision.
- After a failed champion retry, serve at most one other model as a host probe. GPU preflight waits up to 60 seconds
  for memory to clear; a retry blocked only by preflight keeps the first attempt's verdict. Lone champion failures
  need model-specific evidence to count as strikes. GPU host-error matching is anchored to the error line.
- Record a champion award-suspension signal after 72 hours without a successful round and at least three attempted
  rounds; clear it on success, keeping the crown and adding no strike. Validator 1.2.0rc9 ignores the new field:
  enforcing suspension requires a separate validator payment-gate change, which is not included here.
- Successful serves clear stale start-up failure details; recent model failures appear as a neutral window count.
  State remains version 1, with defaults for old files and unknown-field filtering for rollback to rc8.post2.
- Start-up classification reads the last 200 log lines (vLLM's free-GPU-memory error sits above a ~40-line traceback), strips colour codes and accepts `[rankN]:` prefixes, so a busy GPU is recognised as infrastructure.

## 1.2.0rc9 — mechanism v2, release B: per-hotkey cap x2 for s1-fast and s1-pro (weights version 1240; validator and miner)

Announced in #92 at least 24 hours ahead. In plain words:

* The per-hotkey speed cap doubles: s1-fast 640 -> 1,280 decisions/s, s1-pro 160 -> 320/s (`CAPACITY_CAP_MULTIPLIER`
  1.0 -> 2.0). s1-lite and s1-cpu are unchanged. Pay stays linear in verified throughput, so one hotkey standing for
  several GPUs or machines (miner fan-out, 1.2.0rc8) earns what the same hardware earns spread over several hotkeys, with
  fewer UIDs; splitting one machine across hotkeys still gains nothing.
* Validators probe big hotkeys with more parallel requests (up to 32 at this step) and verify more of their answers
  (sample grows as factor^0.75); probe preparation stays within `PROBE_MAX_PREPARED_REQUESTS` = 90,000 with one common
  scale factor.
* Miner defaults follow the cap: per-caller concurrency 32, 456 requests/min, max pending per validator IP 64 (`REAX_MAX_CONCURRENT`
  stays 8). A miner on the old limits is measured at most what its limits allow, never struck.
* Nothing else changes (competitions, split, floors, keep-alive as in 1.2.0rc8.post2). x4 needs a live probe bench first.

## 1.2.0rc8.post2 — queue liveness + first-draw carry-over (competition worker only; weights version 1230 unchanged)

* Count the incumbent, distinct parents and challengers as a complete group, serving each submission once. One group
  per cycle may exceed its track share when it fits the remaining global hourly budget; unused shares get a second pass.
* Retry the oldest due waiting repo once per track and cycle even with fresh arrivals; recovered entries regain their
  normal queue priority. Fetch retries have a separate persisted timestamp; preparation never updates last evaluation.
* Defer entry-local fetch I/O failures without rejecting the miner; full/read-only filesystems still stop the track.
  Enforce the fixed four-skip allowance across both scheduling passes.
* Rule 17 Oct: a pruned entry waiting for confirmation can reclaim its first draw once within 72 hours when the same
  tensors are recommitted by the same coldkey or HF repo. Earlier prunes have until 20 Oct 18:25 UTC. Copy checks,
  confirmation, submission limits and the original model keep-alive window still apply; registration fees are not refunded.
* Carry-over review fixes: a carried entry pulled in as another entry's parent gets no second first draw; the 72 h
  count to when the recommit is first seen; a hotkey pruned and registered again counts as pruned; the recommit keeps
  the original's derivation (a near-copy committed while the original was pruned does not become its parent); a
  withdrawn entry served as another entry's parent is never re-checked, rejected, charged attempts or re-scored; an
  exact copy committed during the prune window is refused against the carried recommit; a host failure is not retried
  in the same cycle's second pass.

## 1.2.0rc8.post1 — evaluation queue hotfix (competition worker only; weights version 1230 unchanged)

* An entry whose repo or revision cannot be fetched (401/404, e.g. still private) no longer takes an evaluation slot:
  the worker prepares the next entry of the queue instead and fetches the unavailable one again on every later cycle.
  Before, unavailable entries at the head of the queue could fill a cycle's slots and nobody behind them was evaluated.
  At most 4 unusable entries are passed over per track and cycle, so a hub outage cannot reject the whole queue at once.
* The hourly evaluation budget is shared between the tracks in every cycle: while another track has work, a track
  takes at most its share of what is left (at least 2, the champion and one entry). Before, both tracks tied on the
  cycle time, the first track of the schedule took the whole budget and the second track was served only every second
  cycle, where its pending confirmations went first, so its first-pass queue did not move (risk_assessment, 17 Oct).
* Rules, margins, floors, payouts and the validator are unchanged; validators keep running 1.2.0rc8.

## 1.2.0rc8 — mechanism v2, release A: bigger keep-alive rules, miner fan-out, cap plumbing at x1 (weights version 1230; validator, competition worker and miner)

Validator and competition worker must ship together (the provisional keep-alive window is written by the worker's state
file); validators on different versions pay different weights until all run rc8, hence the new weights version 1230. In
plain words:

* Faster evaluation queue: the competition worker polls every 30 minutes instead of every hour and runs up to 8 model
  evaluations per hour instead of 6 (the incumbent included), so each track can evaluate two groups per hour; a
  confirmation backlog clears about twice as fast. Rules, margins and floors are unchanged.
* "Refill needed" now warns when the LLM-written items of a track have fewer than 48 draws' worth of uses left (about a
  day at the new rate); before, it counted items and could stay silent until draws failed.
* Keep-alive is paid to at most the 8 best waiting entries per track, not counting the champion (an approved champion is
  paid by its rank, so 8 pending entries stay paid); a champion awaiting owner approval is never pushed out.
* The first-pass queue puts entries closest to the end of their immunity first (those already past it go first of all).
* An entry still waiting for its first evaluation gets a provisional keep-alive: 0.5 x the median, from 600 blocks before
  immunity ends, for at most 12 h per hotkey, one per coldkey and track, together at most 5 % of the track share; not
  while the repo cannot be fetched.
* Sealed items are used at most 24 times (`ITEM_MAX_USES`); the validator logs "refill needed" when a track runs low.
* Close first-champion playoffs (gap below `PLAYOFF_MIN_GAP_SE` = 1.0 paired standard errors) get one more shared draw.
* Miners can put several GPUs or machines behind one hotkey with `REAX_S1_FAST_RUNTIME_URLS` / `REAX_S1_PRO_RUNTIME_URLS`.
* The per-hotkey capacity cap multiplier (`tiers.CAPACITY_CAP_MULTIPLIER`) is in the code but stays at 1.0: **no capacity
  change** (s1-fast 640/s, s1-pro 160/s, every capacity number identical to 1.2.0rc7.post4). Raising it to 2 is a later release
  with its own weights version and a dated #92 notice; the "x2" text below describes that later step.

Details per change:

* **Provisional keep-alive for entries waiting for their first evaluation** (`PROVISIONAL_KA`, release constants). An
  entry that is queued (or held for this host's infrastructure), still committed, has no score, and whose
  hotkey's immunity ends within `PROVISIONAL_KA_LEAD_BLOCKS` = 600 blocks (or is over) is paid
  `PROVISIONAL_KA_FRACTION` = 0.5 x the keep-alive median for at most `PROVISIONAL_KA_MAX_HOURS` = 12 h (one per coldkey
  and track), all together at most `PROVISIONAL_KA_MAX_SHARE` = 0.05 of the track's share, from the track's share before the
  ranks, outside the top-8 keep-alive slots. Fixes entries pruned while waiting for their first draw. New
  `Submission.provisional_since_ts` (optional; older state files load), `compute_provisional`, `mark_provisional`,
  `combine_weights(provisional=...)`. An entry still queued and **not fetched yet** is eligible too (valid current
  commitment equal to the submission's, never rejected, within the open-entry limit); fetched entries keep the
  stricter checks, and a fetched entry that fails one is rejected and drops out at once.
* **Provisional keep-alive: the 12 h window is per hotkey** (review fixes). It is counted from the earliest
  `provisional_since_ts` over all of the hotkey's entries in all tracks, so a recommit (or a second track) never
  restarts it; `mark_provisional` gives a later entry of the hotkey that earlier start. An entry not fetched yet whose
  last fetch attempt left it "waiting for the repo" (missing or private repo) is not eligible until a fetch succeeds.
  `evaluate_cycle` also marks right after the chain sync, so a long cycle does not delay the window start. Weights
  change: roll out together with the keep-alive top-8 change.
* **Keep-alive goes to the 8 strongest waiting entries per track** (`KEEPALIVE_MAX_ENTRIES = 8`, release constant). When
  more entries qualify, a crowned champion awaiting approval keeps its slot, then the highest qualifying (non-proc) scores are paid (tie: earlier on-chain commitment
  block, unknown last, then uid), each `min(median, cap / n_paid)`; the 2-per-coldkey limit is applied first, the others are logged and
  get 0. Champion and rank payouts are unchanged. Cheap extra entries can no longer dilute honest waiting entries.
* **First evaluation by immunity.** Queued first-pass entries whose hotkey is already past its immunity period go first
  (commitment order), then those whose immunity ends soonest (registration block + `IMMUNITY_BLOCKS_ASSUMED` = 5000).
  `Neuron.registered_block` (metagraph `block_at_registration`) is new in the snapshot; unknown = 0 = last.
* **Sealed items are reused by rule.** `ITEM_MAX_USES = 24`: an item used 24 times is retired (never drawn again, eligible for
  later publication); below that the least-used-first order is unchanged, and a confirmation draw stays disjoint from the
  entry's own first draw. The "sealed pool exhausted" event is gone (reuse is normal); the only refill trigger is a
  WARNING "refill needed" when the LLM-written items of a track have fewer than `REFILL_DRAWS` = 48 draws' worth of uses
  left (about a day at two draws per hour).
* **Close first-champion playoffs get one more draw.** `PLAYOFF_MIN_GAP_SE = 1.0`: if the best and second-best playoff
  scores differ by less than 1.0 paired SE, one more shared draw (disjoint where the pool allows) is run for the tied top
  group, the decision is on the mean of both draws, and a remaining tie goes to the earlier commitment. One extension per
  playoff, run right after the first draw and charged to the rolling evaluation budget (it is not skipped for lack of
  budget). The tied group is formed among all served entries; the entry floor applies to the two-draw means. Stored
  score, non-proc and per-tier scores of extended entries are the two-draw means (per-draw values kept in history).
### Miner: several GPUs or machines behind one hotkey (miner only; no scoring or protocol change)

* `REAX_S1_FAST_RUNTIME_URLS` / `REAX_S1_PRO_RUNTIME_URLS`: a list of runtime sidecars of one model (each `URL` or
  `URL|TOKEN_FILE`). Each decision goes to the backend with the fewest requests in flight (respecting
  `REAX_RUNTIME_INFLIGHT`); `/health` checks remove a failing backend and re-admit it after two good answers plus the reference check on that backend; a request
  that failed before any answer byte is retried once on another backend, never more. The single-URL variable is unchanged.
* Backends may be on loopback or a private network (RFC1918, ULA); public addresses and DNS names are refused unless
  `REAX_ALLOW_PUBLIC_RUNTIME=1` (warns at start: token in clear text unless `https`). Signing stays in the miner.
* Start-up and `reax-selfcheck` check every backend against the published reference decisions (a smoke test, not a proof
  of the weights; validator scoring remains the enforcement); a backend that does not reproduce them is refused at start
  and reported per URL by the selfcheck. A URL list is refused for `s1-lite` and `s1-cpu` (no published references).
* Defaults of `REAX_MAX_CONCURRENT`, `REAX_PER_CALLER_CONCURRENCY` and `REAX_PER_CALLER_PER_MINUTE` scale with the number
  of backends (explicit settings win); the per-validator limits shrink while backends are out.
### Validator: per-hotkey capacity cap plumbing (x1 in this release; x2 later, x4 after a probe bench) for s1-fast and s1-pro

* **Capacity recovery under the preparation ceiling, docs for x2 (slice capx4b, review-delta 6 and 8).**
  A probe scaled down by `tiers.scale_budgets_to_ceiling` can only show the scale factor times the unscaled budget,
  which used to feed a ratchet (scaled budget -> lower measurement -> lower average -> lower budget) while a fleet sat
  above the ceiling. Now `tiers.probe_budgets` divides the last scale out of the previous measurement, and
  `tiers.fold_capacity` never lowers the average from a scaled probe in which the UID showed (nearly) all of its budget
  (that number is only a floor); a UID that answers less than its budget, or a strike, is averaged as before. Same
  factor for everybody, deterministic, so still sybil-neutral. Tests simulate 20 probes with a 500-UID junk-fast swarm
  and the swarm leaving. docs/MINER.md and docs/DESIGN-POOLS.md now say plainly: this release stays at x1 (s1-fast 640/s, s1-pro 160/s); x2 (1280/s, 320/s) is a later release, x4 after a live probe bench; effective miner defaults at x2: concurrency 32, 456 per minute,
  max_concurrent 8, max pending per IP 64. Griefing note (a swarm of really fast UIDs lowers everybody's scaled budget)
  in DESIGN-POOLS.md.
* **Per-hotkey capacity cap x2 (later x4) for s1-fast and s1-pro (validator and miner defaults; slice capx4).** New release
  constant `tiers.CAPACITY_CAP_MULTIPLIER` multiplies the cap of the two GPU pools (s1-fast 640 -> 1280 decisions/s at x2 and 2560 at x4,
  s1-pro 160 -> 320/s and 640/s) so one hotkey can stand for several GPUs or machines; pay stays linear in verified throughput
  (sybil-neutral). s1-lite and s1-cpu are unchanged. The plumbing ships at 1.0, where every
  number is identical to 1.2.0rc7.post4; the enable (x2 first, x4 only after a live probe bench) is its own release with
  a `WEIGHTS_VERSION_KEY` bump by the release owner and a dated #92 notice, and miners should not consolidate hotkeys
  before it is live on the stake majority. With it the probe scales: lanes per UID follow its budget (16 up to 64,
  `CapacityParams.lanes_for`), requests per probe at the x4 cap are 864 / 704, the verified sample grows by
  (factor / 16) ** 0.75 above the old cap, a release ceiling `PROBE_MAX_PREPARED_REQUESTS` (90 000) scales all budgets
  down by one common factor if a probe would prepare more requests, and the shard plan and skew guards are
  unchanged. Miner defaults follow the cap
  (x4: `REAX_PER_CALLER_CONCURRENCY` 64, `REAX_PER_CALLER_PER_MINUTE` 888, `REAX_MAX_CONCURRENT` 16; x2: 32 / 456 / 8;
  x1: 16 / 240 / 8 as before) and `REAX_MAX_PENDING_PER_IP` defaults to `max(32, 2 x REAX_PER_CALLER_CONCURRENCY)`.
  Old miners with old limits are measured at most what those allow, never struck. Validators must all run the
  same value; see docs/DESIGN-POOLS.md and docs/MINER.md. While the ceiling binds, a UID that answers nearly all of its scaled budget keeps its capacity average for at most
  `CAPACITY_HOLD_MAX_PROBES` (3) probes in a row; an unscaled probe resets the count.

## 1.2.0rc7.post4 — small-tier pay cap raised (validator; weights version unchanged)

Miners need no update.

* **Tier-order cap raised from 0.5 to 1.0.** The average reward per miner in a tier used to be capped at half of the
  average per miner of the tier above. It is now capped at the full average of the tier above
  (`TierBudgets.order_ratio = 1.0`), so a small-tier (s1-lite / s1-cpu) miner can earn up to what the average miner of the
  tier above earns. The fixed tier budgets (GPU 80 / small GPU 15 / CPU 5 % of the inference part) are unchanged: the
  small and CPU tiers together never take more than 20 % (the small tier can absorb excess moved up from the CPU tier),
  and excess only ever moves up. The cap applies to a tier's average, so while the small pool has only a few miners a
  single fast small-pool hotkey can earn more than a GPU miner at target (bounded by the small tier's budget and the
  per-hotkey unit cap, as before). Nothing else changes (scoring, floors, probes, weights version 1220).
* Example (from `tests/test_tiers.py`): one healthy s1-lite miner next to 30 GPU miners. Before, it was held at half the GPU
  average (1.40 % of the inference part, GPU miners about 3.29 % each); now it gets the GPU average (2.81 % vs 3.24 % per GPU
  miner), still within the 15 % small-tier budget. Fleets where the tier-order rule was not binding (e.g. 20 / 50 / 200 miners) are
  unchanged.
## 1.2.0rc7.post3 — owner award key (validator; weights version unchanged)

Miners need no update.

* **Award approval key replaced.** `AWARD_APPROVAL_PUBKEY` is now the subnet owner's own Ed25519 key
  (`0cadd19fccadd3b00de072d1e518a6263671f8c106bc8201d02a86af7b1f96e5`), created on the owner's machine and
  passphrase-encrypted. The 16 Oct interim key is retired and its private key deleted; approvals signed with it no
  longer verify (none were issued). Scoring, floors and the award rules are unchanged.

## 1.2.0rc7.post2 — refilled sealed pools and re-measured floors (validator and competition worker; weights version unchanged)

Miners need no update.

* **Sealed pools refilled.** Both pools were used up (every demand forecasting item had been drawn at least once, risk
  assessment 98 %), so draws reused items. 1,116 (demand forecasting) and 1,095 (risk assessment) new LLM-written sealed
  items, three agreeing blind labelers each, split evenly over same-family, new-domain and new-family (four new families
  per track), are added to the pools. Draws take unused items first, so the next few draws are mostly new items.
* **Floors re-measured on the refilled pools.** The s1-fast reference model (Plumb-4B) on six consecutive release draws
  (300 items, 15 % procedural, live use counts): demand forecasting 27.32 (non-proc 31.10), risk assessment 28.54
  (non-proc 32.95). `TRACK_ENTRY_FLOOR` 39.7 / 33.2 -> **28.8 / 30.0**, `KEEPALIVE_NONPROC_FLOOR` 49.8 / 37.8 ->
  **32.6 / 34.5** (reference + 1.5 points, as before). The new items are harder for the reference model, so the floors
  are lower; a model that is perfect on the procedural items and uniform elsewhere reaches at most 20.7 / 21.6 on these
  draws, below both floors. `KEEPALIVE_LEGACY_FLOOR` is unchanged.

## 1.2.0rc7.post1 — keep-alive pays the median inference weight (validator; weights version unchanged)

Miners need no update.

* **Keep-alive per entry = median serving weight.** At a fixed 5 % of a track share split evenly, an entry earned about
  a quarter of the median inference miner (0.0011 demand forecasting / 0.0013 risk assessment against 0.0045 in the
  23:40Z weight set), the keep-alive entries were 13 of the 21 lowest weights, and two entries (UIDs 26 and 27) were
  deregistered while holding keep-alive weight. Now each qualifying entry is paid the median of the positive serving
  weights (miner part, without the owner UID and without competition-paid UIDs; the mean of the two middle values for
  an even count). The keep-alive total of a track is bounded by `KEEPALIVE_MAX_SHARE = 0.25` of the track's share
  (replaces `KEEPALIVE_SHARE = 0.05`): above it, each entry gets `0.25 x share / n`. The ranks are paid from the share
  minus the keep-alive paid. Eligibility, the owner-approval exemption, 2 per coldkey and the 24 h window are
  unchanged; an approved ranked entry still has no keep-alive slot. The `keepalive_share` argument of `combine_weights`
  is now that cap fraction (0 = off).
- Keep-alive window per model: a pending entry's 24 h window starts at the earliest first pass of the same model
  (`hash_named`) in the track, withdrawn records included, so committing the same tensors again (same coldkey, any
  hotkey) does not open a second window. A crowned champion keeps its 24 h from crowning.

## 1.2.0rc7 — capacity probe shards (validator) and a fair first crowning on specialist tracks (competition worker; weights version unchanged)

Miners need no update.

* **Keep-alive for entries scored before 1.2.0rc6.** An entry whose first pass was scored under rc5 has no non-proc
  score, so the rc6 keep-alive left it out until its next draw; on demand forecasting three such entries (scores 79.5
  to 89.5) were deregistered while pending confirmation. Such an entry now qualifies when its total score is above
  `KEEPALIVE_LEGACY_FLOOR` (60.0 demand forecasting, 50.0 risk assessment: above the 49.8 / 31.8 a rules-only model
  reaches under the rc5 mix, plus a margin). Every other keep-alive condition is unchanged (copy-checked, current
  commitment, champion or pending confirmation, 24 hours from the first pass, 2 per coldkey, 5 % of the track share).
  An rc6+ draw without non-proc items is not treated as legacy.
* **Re-registration with the same model.** A withdrawn entry (the hotkey was deregistered or changed its commitment)
  no longer blocks its own model: the same coldkey may commit the same tensors again, from the same hotkey on a new uid
  or from a new hotkey, and the exact-hash check lets it pass (the hash passes to the new entry). Another coldkey re-using
  the hash is still refused. The new commitment is a new entry in a later cohort; the old first pass and score are not
  restored. Before, the same hotkey re-committing the same text stayed withdrawn forever, and a new hotkey was refused as
  `identical tensors to an earlier commitment`. The withdrawn entry stays in the state under `<key>#withdrawn@<block>`.
  Event: `... committed again after its withdrawal (uid N), a new entry`.
* Advice for entrants (docs/SPECIALIST-TRACKS.md): register the hotkey shortly before you commit; immunity (~16.7 h)
  starts at registration and the first weight after your first pass takes up to ~1-2 h to count.
* `reax-award-approve sign` accepts a passphrase-encrypted PEM key and asks for the passphrase on the terminal (never from an
  argument or the environment).
* **A cohort playoff that waits for a full hour's budget holds it.** Its track goes first in the next cycle and the other
  tracks wait while it still lacks budget, so it is not starved by single freed slots.
* **The first champion of a vacant specialist track is decided per cohort** (docs/SPECIALIST-TRACKS.md, "First
  champion of a track"). On the DF track all eleven opening entries were first seen in the same block, but they reached
  their confirmation draws in three groups hours apart, and the first group to confirm would have been crowned; the
  later ones then needed the full margin over it. Now an entry that passes its confirmation on a track without a
  champion waits until every entry first seen in the same block or earlier is decided (queued entries, pending
  confirmations and waits of the validator's own, such as a base download deferred by its budget or the disk, hold it
  back; a private repo or another wait on the miner's side does not). The cohort's confirmed entries then meet on one
  shared fresh draw (the playoff, at most the hour's remaining evaluations, best confirmation scores first); the best
  score there that reaches the entry floor is crowned, the others stay pending and are confirmed against the new
  champion with the full margin. The playoff draw may reuse items (the pools are small); only aggregate scores leave
  the validator. A lone confirmed entry is crowned at its confirmation, as before. Later cohorts are unaffected.
  Events: `passed the confirmation draw (...); waits for its cohort`, `cohort playoff on draw ...`, `playoff draw ...:
  uid ... <score>`, `is the new champion (...): best of the cohort playoff`, `back on the challenger path`.
* **Draws a pending entry needs are never pruned.** The state kept the 40 newest draws; a pending entry whose first
  draw fell out could never be confirmed (1.2.0rc6 made it wait with an error instead of confirming without the
  exclusion). Now its first draw and its passed confirmation draw are always kept, plus the 40 newest other draws.
* **Cohort review fixes.**
  * A wait of the validator's own is recognised by a field set where the wait starts, no longer by words in the reason.
    The reason can contain the miner's base model id, so a base named `.../deferred-...` could hold a cohort forever.
  * A cohort waits at most 12 hours after its first passed confirmation (`COHORT_MAX_WAIT_S`). Then the open entries
    no longer hold it back, e.g. a legal base larger than the 40 GiB download budget per cycle, which is deferred on
    every cycle.
  * An entry whose first draw was lost before this release no longer heads the confirmation queue. Before, its group
    was fetched and dropped every cycle, and no later group was confirmed.
  * After 3 playoffs abandoned in a row because a participant could not start on the host, the playoff is decided among
    the entries served.
  * The playoff runs only with a full field (every confirmed entry, or a whole hour's evaluations). It is no longer cut
    by what is left of the current hour.
  Event: `cohort playoff: 3 playoffs abandoned for a host reason; decided among the N entries served`.
* State: new fields with defaults. Submissions: `confirm_draw`, `confirm_score`, `wait_kind`. Tracks: `playoff_aborts`.
  A confirmed entry waiting for its playoff keeps the status `pending_confirm`. Rollback-safe both ways: an older
  release ignores the fields and confirms such an entry again on its next draw (the 1.2.0rc6 rule). A `pending_verification`
  entry without `wait_kind` counts as the miner's wait until its next cycle re-checks the base and sets it.

* **The capacity probe is no longer bound by one validator core.** On mainnet the validator's event loop sat at 100 %
  of one core for the whole probe window (~800 requests/s in total, however many UIDs answered), so the best s1-fast
  miners were clipped at 55-70/s and s1-pro at 37-45/s. The probe window now runs in forked shard processes, one per
  ~12 UIDs (`REAX_PROBE_SHARDS`, default `auto`, at most the cores minus 2; `1` = the 1.2.0rc4 single-process probe).
  All shards open their windows at one common moment and the 250 ms send-skew guard covers all shards together
  (siblings in different shards measured split/solo median 1.000, max 1.052 on the live validator host). A shard that fails or times out drops the whole probe. Shards
  inherit the loaded wallet through `fork`; no key file is written and no key material crosses a pipe. Answers are
  still verified in the validator process. Weight math is unchanged.
* **Shard plan cannot be steered by declared IPs; big coldkey groups are split** (gate review H1). UIDs of one
  coldkey share a shard only while the group fits a fair share of the probe's prepared requests; a heavier group is
  split over shards (the live metagraph has a 33-UID coldkey). Axon IPs no longer group UIDs: they are declared on
  chain by anyone, and sybil UIDs declaring competitors' IPs could glue those competitors onto one saturated shard (a
  simulated 118-UID shard on the live metagraph). The shard count is bounded by the UIDs, not the coldkeys.
* **A saturated probe loop drops the probe** (gate review M2). `cpu_saturated` (the busiest window loop, the
  validator's or one shard's, above 90 % of a core) is now a drop reason like the lag and skew guards: capacities are
  kept, nobody is recorded at what the loop left. On the live host a saturated loop read s1-fast ~0.00 with a loop
  lag p95 of 1-2 ms (the lag guard cannot see it); sharded probes run at 13-29 % of a core. With
  `REAX_PROBE_SHARDS=1` on a mainnet-sized metagraph this means probes are dropped instead of stored near 0.
* **Shards are planned by the requests the validator may send, not by expected throughput** (gate review N1). A UID
  is prepared requests for up to 4x its last measurement and gets all of them if it answers instantly (the answers
  are checked after the window). Planned by expected requests/s, a coldkey with a high past verified capacity could
  answer junk at loopback speed, saturate its own shard on purpose and so drop every probe: capacities frozen, new
  UIDs at 0, its own capacity kept without hardware. Each UID now weighs `requests_for(budget) / window` (all its
  prepared requests inside its window). In the test simulation (240 UIDs, 24 s1-fast attackers with a moving average
  of 48/s and a last probe of 160/s, 20 shards) the busiest shard falls from ~434 to ~176 requests/s (saturation is
  extrapolated at ~470). Honest probes are measured as before; only the assignment of UIDs to shards changes.
* **Probes dropped in a row are counted** (gate review N2). Any dropped probe (a guard, a failed or hanging shard, the
  outer time budget, an error) counts; a kept probe resets the count. `last_probe_stats["drops_in_row"]` carries it,
  and from the third drop in a row every drop logs one ERROR line `capacity probe: N probes dropped in a row; ...
  (reasons of the last 3, oldest first: ...)` for an external alert. Unmeasured UIDs earn nothing while probes are
  dropped.
* **Probe responses are capped at 32 kB and never decompressed** (gate review M1; validator challenge findings 6/7).
  The probe session sends `Accept-Encoding: identity` and does not decompress; an answer above 32 kB or gzip-encoded
  does not count for that UID (no strike; honest answers are 3.5-5 kB). This removes the gzip-bomb CPU lever on a
  shard and cuts the worst-case memory of hostile answers ~8x. The probe report names lanes a miner refused with
  429/503 (`refused_lanes`, also from shards; observability only).
* **Probe diagnostics.** Every probe (kept or dropped) logs requests/s, decisions/s, the busiest loop's CPU share,
  loop lag p95 and send skew, and with shards one line with every shard's UID count, share of the prepared requests
  and CPU share; `last_probe_stats` carries the same fields (`shard_uids`, `shard_load`, `shard_cpu`, `drops_in_row`).
* Shards set `RLIMIT_CORE` 0 and are not dumpable (no core file with the hotkey in memory, no ptrace); the compose
  file sets `ulimits: core: 0` and `mem_limit: 8g` for the validator (measured peak below 2 GiB with all shards). A
  shard that exits between its ready message and the window's start is reported as such; the probe's outer time
  budget includes the shards' start and result slack, so a hanging shard is named by its own timeout.
* **Fix: s1-fast answers no longer turn stale while the validator checks them.** Probe answers are checked after the
  slowest pool's window (s1-lite: 16 s); 1.2.0rc4 allowed each pool's own bounds only, and on a busy host whole s1-fast
  UIDs measured 0 (`invalid answer (auth: StaleRequest)`). The allowed age now covers the probe's duration (bounded by
  the nonce store's retention).
* **Fix: a connection warm-up that overruns is a dropped probe with a clear reason** (`the connection warm-up did not
  finish within 3 s`), counted like the other guards; 1.2.0rc4 logged a misleading `capacity probe did not finish
  within N s`.
* A probe cancelled by its time budget while it is still forking its shards also kills and reaps the shard forked
  last (before, that one process stayed behind as a zombie with an open pipe).
* `scripts/bench_probe.py`: `--lite-uids/--lite-dps`, `--shards`, `--extra-cpu-us` (emulates a slower validator host,
  e.g. 900 for mainnet), `--warmup-s`, `--uvloop`; prints requests/s, the busiest loop's CPU share and shard count.
  `--sibling-sets 1,2,4` (with `--sibling-dps`, `--sibling-rtt-ms`, `--siblings-coldkey`) compares one machine served
  by 1, 2 and 4 hotkeys inside the same probe; the peak memory of the shards is printed. A fake machine's GPU now
  finishes its requests at absolute times, so late timer wake-ups of a busy bench host no longer slow the fake GPU
  itself (a sleep per request read ~20 % low at 640/s).

## 1.2.0rc6 — keep-alive weight for specialist entrants (validator) and copy-check fixes (competition worker)

Miners need no update. Validators and the competition worker update (the state file stays compatible both ways).

* **Keep-alive window counts from the first pass** (`confirm_after_ts`), not from the last fetch, so an entry whose
  confirmation keeps failing loses its keep-alive after 24 h. A confirmation whose first draw is no longer in the state
  waits with an ERROR log instead of running without the exclusion.
* **Both tracks share the hourly evaluation budget fairly.** The track evaluated least recently goes first in a cycle;
  before, demand_forecasting always went first and could use the whole budget while risk_assessment waited.
* **Keep-alive for competition entrants.** A competition hotkey has one commitment and cannot serve inference, so in
  rc5 it earned 0 while queued, evaluated, waiting for its confirmation draw or for the owner's approval, and was
  deregistered as the lowest-emission UID when its immunity (5000 blocks) ended, which also dropped the submission.
  Now `KEEPALIVE_SHARE = 0.05` (`competition_split.py`, release constant) of each active track's share is split evenly
  among the track's qualifying entries before the ranks are paid from the other 95 %: per entry `0.15 x 0.05 / n` of
  the miner weights in the 50/50 profile (0.00039 with 19 entries in one track; the lowest non-zero live incentive is
  about 0.00008). A qualifying entry is the hotkey's current commitment, copy-checked (exact hashes and the
  near-duplicate check: not pending verification, not rejected, a near copy only once it beat its parent), with a
  latest score on a shared draw above the entry floor, waiting for a decision: pending confirmation (at most 24 h after
  its first pass) or a new champion waiting for the owner's approval (at most 24 h after crowning). Decided entries
  (former champions, not better) do not qualify, so the keep-alive is never a standing payment. Queued, unevaluated, below-floor and copied entries get nothing. At most 2
  entries per coldkey and track. It is a participation keep-alive, not an award: the owner-approval gate does not apply
  to it, and ranks still need the approval. Local and remote mode (`spec-state.json`, read once per round) compute it
  the same way (`spec.evaluator.compute_keepalive`). With no qualifying entry the weights are bit-identical to rc5.
* **Keep-alive needs the non-proc tiers.** On the rc5 mix a model that only learned the five public procedural
  families scored about 49.8 on a DF draw (entry floor 24.8), so "above the floor" alone was farmable. An entry now
  also needs a score above `KEEPALIVE_NONPROC_FLOOR` (`competition_split.py`, release constant: DF 49.8, RA 37.8) on
  the non-proc items of its latest draw (same_family, new_domain, new_family). The value is the reference model's
  (Plumb-4B = s1-fast) score on the non-proc items of six 300-item release draws + 1.5 (DF 48.34, RA 36.27; 1530 items
  each, see "Release mix" below). An entry last scored before this build has no non-proc score and does not qualify
  until its next draw (fail closed).
* **Scores per tier in the history.** For every model served on a draw the evaluator stores the non-proc score and the
  score per tier (`Submission.nonproc`, `Submission.by_tier`, and `nonproc` / `by_tier` in each first and confirmation
  history entry) and logs one `draw <id> non-proc: ...` event line per draw. New keys only: the state stays version 1
  and an older build ignores them.
* **Tier shares hold as the pools are used (sampler).** A tier with too few unused items now fills its `TIER_MIX`
  quota with its own least-used items before any other tier's fresh items; only a tier with too few items at all hands
  the rest to the other non-proc tiers. Before, the shortfall went to any fresh item, so proc grew from 50 % to 62-70 %
  of the DF draws from the second draw on (6 simulated draws on the deployed pool sizes). Reuse is still reported
  (`fresh = False`, "sealed pool exhausted" event).
* **Release mix: 15 % proc, 300 items (owner decision 16 Oct 2026).** `TIER_MIX` is proc_sealed 0.15, same_family
  0.425, new_domain 0.2125, new_family 0.2125 (the LLM-written 85 % split 50/25/25) and `SAMPLE_SIZE` 300 (was 600), so
  a first draw and its disjoint confirmation draw fit the pools. The proc tier never takes another tier's share and
  keeps at most its share of the draw that results: without enough non-proc items the draw is smaller, and below
  `min_n` (200) there is no draw (`SampleError`); a smaller `REAX_SPEC_SAMPLE_SIZE` that is drawn in full still works.
  Six release draws on the deployed pools hold 45 proc and 255 non-proc items each.
* **Entry floors measured on the release mix.** `TRACK_ENTRY_FLOOR` DF 39.7, RA 33.2 (was 24.8 / 33.7): the reference
  model's (Plumb-4B = s1-fast) score with the validator's evaluator on six consecutive 300-item release draws from the
  deployed pools (writer-note rows removed, live use counts of 16 Oct 14:05Z; 1800 items per track) + 1.5: DF 38.17,
  RA 31.66. A model that only learned the procedural families no longer clears the DF entry floor.
* **Model files only for champions.** The evaluator keeps a model directory between rounds only for a champion (served
  every round). An entry pending confirmation is fetched again for its confirmation draw (the pinned revision) and must
  hash to the weights of its first check (`hash_named`); other weights reject it ("fetched again: other weights than at
  its first check"). Sketch renewal of an earlier entry whose files are gone re-fetches and hash-checks it as before.

Copy-check fixes (competition worker; the award gate is unchanged):

* **Rounding a copy to BF16 no longer hides it (R1).** The delta sketch rounds the model's sampled values to the base
  tensor's dtype (round to nearest even, as torch) before the difference. An F32 parent saved as BF16 by a copier now
  has exactly the parent's rounded delta (real Qwen3-4B, review measurement: rel_L2 0.77 before, 0.00 now). An
  independent fine-tune stays not derived. "Identical to the declared base" is still decided on the unrounded values,
  so an F32 fine-tune whose change vanishes in BF16 is not refused.
* **Secret sample positions (R2).** The sampled positions come from a per-validator secret seed
  (`REAX_SPEC_SKETCH_SEED`, or `sketch-seed` in the work directory, created once with `os.urandom`, mode 0600, never
  logged): one random position in each block of 4093 values, and at least 64 values of small tensors (norms), which
  are down-weighted to their share of the elements. rc5 sampled every 4093rd value from 0, so re-tuning exactly those
  0.024 % of the weights made a copy look independent. Every stored sketch records the seed id
  (`tensors.seed_id`); a sketch of another seed or of rc5 is recomputed (weights re-fetched if needed) before a later
  submission is compared with it. `SKETCH_RULE` 3: pending verdicts of older rules are re-checked.
* **Behavioural near-duplicate check (R4).** On every shared draw the evaluator compares each challenger's per-item
  option probabilities with those of every earlier submission served on the same draw (incumbent, parents, other
  challengers). Mean Jensen-Shannon divergence below 0.005 bits, or the same option order on >= 99 % of >= 50 items,
  marks it derived from that submission (`derived_how = "behaviour"`), whatever its weights look like (permuted
  hidden units, rescaled layers). A twin must beat each twin by the full margin, on the first and on the confirmation
  draw, so a behavioural twin of the champion cannot be crowned. Each measured pair is kept in the challenger's
  history (`kind: behaviour`) for later calibration on real outputs.
* **Coverage reason (R5).** The pending reason names int-quantized tensors (GPTQ/AWQ) as well as FP8. A
  `base_model.model.` tensor-name prefix (PEFT export) is stripped before the names are matched with the base.
* New state field `Submission.derived_how` (default "", rollback-safe).

Review fixes (review-rc6 F1, F2, F3, F5, F8):

* **An approved champion holds no keep-alive slot (F1).** `combine_weights` drops a keep-alive uid that the track's
  ranks pay and `award_ok` approves: it is paid by its rank only, and the other entries share the keep-alive part. A
  champion still awaiting approval keeps its slot. With the approved champion as the only entry, the weights are
  bit-identical to rc5.
* **A far-future keep-alive start gives no slot (F2).** `compute_keepalive` refuses a window start more than
  `KEEPALIVE_FUTURE_S = 3600` s after `now` (fail closed); before, a clock that jumped forward kept the slot for good.
* **Sketch renewal cannot block a track (F3).** Any error while renewing one earlier submission's sketch (ENOSPC, a
  damaged download, a failed sketch write) is logged and that submission is skipped, instead of failing the track
  every cycle. A damaged sketch file (`zipfile.BadZipFile`, ...) counts as not current and is renewed. Before each
  renewal download the free space is checked; below `min_free_bytes` the candidate waits in PENDING_VERIFY ("renewal
  waits for disk", its download removed) and is queued again next cycle, instead of being checked against an
  incomplete set of sketches. An error in the behavioural check no longer discards the group's scores (no twins then).
* **Behavioural twins on yes/no rows (F5).** The order rule now counts only items with at least 3 labels (on a yes/no
  item the order is just the argmax, which two independent fine-tunes share on nearly every item) and also needs the
  mean JSD below 0.05 bits. The rule: twin = >= 50 valid items and (JSD < 0.005 bits, or >= 20 items with >= 3 labels,
  >= 99 % of them in the same order, and JSD < 0.05 bits). Two independent binary-heavy fine-tunes (probability noise
  sd 0.08, same argmax on >= 99 % of the yes/no items) are no longer twins; a temperature copy (T 0.7 to 2.5 on the
  test draws) still is. The history entries record `order_items`.

Review fixes (review-rc6b F1, F2, F3, F4):

* **One first draw per confirmation group (F1).** A confirmation draw excludes the first draws of its whole group.
  Entries from two first draws that came due in the same cycle were one group, and both first draws together left
  too few non-proc items on the deployed DF pool (a 300-item draw takes 255 of its 520): `SampleError` on every
  cycle, nothing confirmed and the queue behind it starved. Now a group holds only the due entries that share the first
  draw of the earliest-committed due entry; the next cycle confirms the next first draw, then the queue goes on. A draw that still cannot be
  made (`SampleError`) is one `no draw: ...; refill the sealed pool` event and a warning without a traceback, and the
  track waits for the next cycle.
* **A transient re-fetch error does not reject a pending entry (F2).** A pending entry is fetched again at its
  confirmation (its files are not kept), more than 7200 blocks after its commitment. A hub 5xx, a download timeout or a
  full disk then rejected it for good ("not available within 7200 blocks"). For an entry fetched before
  (`hash_named` set) the publish deadline no longer applies: it waits, and only `MAX_ATTEMPTS` (3) failed fetches in a
  row reject it ("no longer available (3 fetches failed)"). An entry never fetched still expires at the deadline.
* **A re-fetched pending entry keeps its first check (F3).** When the re-fetched weights hash to those of its first
  check, a copy-checked pending entry is served at once, as if its files had been kept: no second copy check and no
  sketch renewal for it, so a renewal that waits for disk can no longer send it back to the queue. Other weights are
  still rejected.
* **Keep-alive tests on the evaluation clock (F4, tests only).** The keep-alive tests stamped their entries with the
  wall clock and two of them read the state at a fixed time (`NOW0`), so they failed once the wall clock was more than
  `KEEPALIVE_FUTURE_S` past `NOW0`. All keep-alive tests now stamp and evaluate on `NOW0` (`test_keepalive.NOW`).

## 1.2.0rc5 — engine track held; owner approval before any competition award (weights version 1220; validator and worker)

Miners need no update. Validators and the competition worker must update before the engine track could pay.

* **The engine track is held in code.** `ENGINE_TRACK_START_UTC = None` (`competition_split.py`), separate from the
  use-case tracks, which still start at 2026-10-16 12:00 UTC. While it is held: no engine cycle runs in the validator or
  in `reax-competition-worker`, `reax-engine-submit` refuses on finney, and the validator pays no engine champion, also
  in remote mode when the synced `engine-competition.json` names one. The engine share goes to inference.
* **Owner approval before payment.** A winning entry is paid only after the subnet owner has checked and approved it:
  the engine champion and every paid rank of a use-case track need an Ed25519-signed approval for exactly (track,
  hotkey, image digest or `repo@revision`). Without it the share goes to inference (never burned; lower ranks do not
  move up) and the validator logs `awaiting owner approval`. Approvals are read from `REAX_AWARD_APPROVALS` (validator
  host only, never the worker) and checked against `award.AWARD_APPROVAL_PUBKEY` or `REAX_AWARD_APPROVAL_PUBKEY`. A
  missing key or file, a bad signature or an expired approval pays nothing. Always on for finney.
* New tool `reax-award-approve` (`keygen`, `sign`, `verify`). `pycryptodome` (already installed through bittensor) is now
  a declared dependency.
* With no champion and no approved leader the weights are exactly the rc4 weights.
* **Specialist tracks: payout ranks follow the champion order.** Ranks 1-3 of a track (75 / 17 / 8 %) are the
  champion, then the former champions, most recently crowned first, as SPECIALIST-TRACKS.md describes. 1.2.0rc4
  re-sorted them by their latest scores, which come from different draws: a former champion holding 90 from an old
  draw outranked the champion that beat it with 85 on a newer draw and took 75 % of the track share. The evaluator
  now hands out `competition_split.RankedLeaders` and `track_payout` pays them in that order; entry floors still apply
  (the leaders after a floored one move up). Plain score lists are still ranked by score.
* **Specialist tracks: no award without the near-duplicate check.** When the declared base model is not on the
  validator (an unknown name or revision), or shares no weight with the model, the submission is *pending
  verification*: kept, not evaluated, ranked or paid, and checked again every cycle until the base is there.
  1.2.0rc4 let it pass with only the exact hash check. A champion that was never checked is held the same way. Near
  duplicates are now compared by the tensor hash of the resolved base, not the declared `base_model` text, so a
  mirror or another name of the same base does not hide a copy. Exact copies are refused as before, with or without
  the base. Operators: `REAX_SPEC_BASE_DIR` must hold the bases miners declare (`pending verification` in the log).
* **Specialist tracks: the worker fetches declared base models itself.** A missing base is downloaded into
  `REAX_SPEC_BASE_DIR` (default `${REAX_COMPETITION_DIR}/spec-bases` in both compose files) with the submission download
  rules (pinned revision, public, safetensors only, size caps, allowlisted architecture, no `auto_map`; any licence),
  into a temp directory renamed into place. Gated, private, missing or oversize bases stay pending with the reason and
  are retried every 6 cycles; at most 40 GB of bases per cycle and 50 GB must stay free. `REAX_SPEC_BASE_PRESEED`
  fetches a list in the background at start (SPECIALIST-TRACKS.md, "Base models").
* **Spec state rollback compatibility.** The state file stays at version 1; new fields (`first_incumbent`, `base_id`)
  have defaults and unknown keys are ignored, so an older release loads a newer file after a rollback and the other
  way round.
* **Specialist tracks: confirmation against a champion crowned after the first draw** (contributed by Ihor-52466,
  reaxlabs/reax-subnet#3). A challenger whose first draw was measured against another incumbent, or against a vacant
  track, now has to beat the incumbent it meets on the confirmation draw by the full margin `max(1.5, 2 x SE)`. Before, it needed half of its first-draw delta, and on a
  vacant track that delta is the entry floor itself: a challenger that cleared the floor one cycle after another one
  needed +16.85 (risk_assessment) or +12.4 (demand_forecasting) over it once that one had been crowned. The submission
  state records the incumbent of the first pass (`first_incumbent`); state files written before fall back to the
  first-pass history. Scoring and the wire protocol are unchanged.

* **The candidate's vLLM share leaves room for what vLLM uses outside it.** vLLM 0.30 takes its memory snapshot after
  the CUDA context exists and allocates more after its memory profile (kernel warm-up, first batches), so a model capped
  at the 25 GB budget really took about 26.2 GB. Measured with Qwen3.5-9B text-only bf16 (8.95 B parameters) on an RTX PRO
  6000 Blackwell: 1.22 GiB outside the share, which explains the reported CUDA OOM a few seconds into scoring with
  26,605 MiB free. The cap is now `budget - 1.5 GiB` (`GPU_OVERHEAD_GB`): the same model gets 0.245 instead of 0.261,
  peaks at 24.46 GiB and scores all 513 public dev items. Models below the cap (4 B and smaller) are unchanged.
* **The free-memory gate before a specialist cycle uses `REAX_SPEC_GPU_BUDGET_GB`** instead of the release default, so
  an operator who lowers the budget is not still asked for 25,600 MiB free.

* **A declared base may be up to twice a submission's size caps** (18 B parameters, 40 GiB of weights): it is only read for
  the copy check, and a text-only 9 B fine-tune of Qwen3.5-9B declares the full 9.65 B model as its base.

Review fixes (rc5 review F1-F6):

* **Specialist near-duplicate check works across dtypes, with a coverage rule.** The delta sketch compares every float
  tensor whose name and shape match the base, whatever its dtype (both sides are decoded to float). An honest
  fine-tune saved in F32 or F16 of a BF16 base is now verified (rc5 left it *pending verification* forever), and a
  copy that upcasts the parent and keeps one small tensor in the base dtype with noise is now *derived*. The compared
  tensors must hold at least 90 % of the base's float parameters, else the submission is pending with the coverage in
  its reason. FP8 tensors are never compared and say so in the reason. A submission held under the rc5 rule while
  its base is on the host is queued once more for the check (`sketch_rule` in the state, default 0).
* **Former champions that were never delta-checked are not ranked** (`compute_leaders`).
* **The competition worker refuses a custom finney endpoint** (the validator's genesis check), so it can never run in
  development mode against mainnet.
* **The award key:** the compose comment now says that an empty `REAX_AWARD_APPROVAL_PUBKEY` means the release key, and
  the validator logs an ERROR at start on finney if neither key is valid. `AwardGate.approved` never raises (returns
  False).
* **A candidate whose vLLM container stops at start-up because the GPU has too little free memory** ("Free memory on
  device ... is less than desired GPU memory utilization") is a host problem: it does not count toward the model's
  `MAX_ATTEMPTS` and is retried next cycle.

Engine-track rules, prepared for its opening (the track stays held; nothing here pays out):

* **A pending challenger is protected.** It is displaced only by a submission that beats its pooled score by the same
  3 % margin required against the champion. A later near-copy at noise level, or one that misses the champion's bar,
  no longer resets its confirmation (`update_champion` used to clear the challenger whenever the top contender did not
  clear the margin).
* **The champion's score is held at the minimum of its measurements** instead of the latest one, so one lucky
  re-measurement cannot raise the bar.
* **Confirmation is a second fresh session, not a later cycle.** New containers and new secret items, run right after
  the first in the same cycle (`confirm_gap_blocks` 1800 -> 0); the champion is measured again with the challenger.
  Each measurement carries a session id, so the same one is never counted twice. Crowned at the lower of the two;
  the history records the pooled score and the number of sessions.
* **Frozen thresholds in one block** (`engthresholds.py`: margin, min gain, agreement gate, distance budget, items,
  runs, confirmations). Its sha256 is logged at start and written into the state file as `thresholds_sha256`; a state
  written under other thresholds logs a warning. A test pins the digest.
* Acceptance test: 1000 rebuilt copies of the reference with seeded measurement noise are never crowned; a real +10 %
  engine in the same run is.

## 1.2.0rc4 — capacity probe timing (weights version 1220; validator-only, miners need no update)

The wire protocol and the weights are computed as in 1.2.0rc3; only when a UID's probe window opens changes.

* **A UID's probe window opens at the validator's first send to it.** 1.2.0rc3 opened one window for every UID and then
  opened all 16 lanes of each UID in turn (~0.2 ms of validator CPU per lane): at ~175 UIDs the last UIDs' first
  requests left 0.6 s (bench) to 2.5 s (mainnet report) after their window had opened, so fast miners were measured at
  75-90 % of their speed. Now each UID's window opens at its own first send and has the pool's length, and lanes open
  one per UID per loop pass, so every UID's first request leaves within ~30 ms (175 UIDs, bench: measured/true 0.95-0.97
  median, worst 0.89, was 0.82-0.92 median, worst 0.73).
* **Siblings still share one machine.** All UIDs' windows open within the send skew; a probe whose first sends spread
  over more than 250 ms is dropped like a lagging one (`capacity probe skipped: ... first sends spread over`).
* `scripts/bench_probe.py` emulates the internet (`--rtt-ms`, per-machine speed `--dps LO-HI`, `--pro-uids`,
  `--siblings`, `--loop-load`, `--gil-load`) and prints per-lane timing with `--trace` (`REAX_PROBE_TRACE=1`).

## 1.2.0rc3 — unreleased: capacity scoring v2

Weights version **1210** (scoring changes; 1.2.0rc2 sent 1200). The wire protocol is unchanged.

* **Capacity is paid in a straight line with verified speed**. 1.2.0rc2 paid `ema x min(1, capacity /
  target)`: a GPU faster than the s1-fast target (40/s) earned once per hotkey when it was split, because every share
  still measured at least 40/s. Now the factor is `min(max_factor, capacity / target)`:
  * s1-fast and s1-pro: up to 16 units (640/s and 160/s), far above one machine. Several hotkeys on one machine measure
    and earn at most what one hotkey on it does, and a hotkey with more GPUs earns more.
  * s1-lite and s1-cpu: up to 4 units (32/s and 4/s), near the best honest node of the tier. Their small models run
    10–100x faster on a big GPU; a higher cap would let one GPU take most of a cheap pool.
* **The tier-order rule is unchanged**: the average reward per miner in a tier is at most half of the tier above.
  The 80 / 15 / 5 tables hold however fast GPU miners get.
* **The probe can measure fast machines.**
  * All of a pool's queryable UIDs are probed at the same instant, including UIDs the reference-cap rotation rests and
    UIDs that did not answer this round (a UID that is down measures 0).
  * Requests are prepared and serialised before the window, and the lanes' connections are opened before it.
  * Only answers inside the common window count, so a hotkey that stops early cannot free the machine for its
    siblings.
  * 16 lanes per UID for s1-fast and s1-pro (8 s1-lite, 4 s1-cpu), requests for 4x the UID's previous measurement
    (at least the target, at most the cap). Verification stays at most `verified_max` decisions per UID and probe.
  * s1-fast window 3 -> 5 s, s1-pro 5 -> 8 s, s1-cpu 4 decisions per request.
* **The validator's CPU stays out of the measurement.** Inside the window it only sends and records. Signatures and
  answers are checked after the window; one forged or malformed answer sets that UID's probe to 0. If the validator's
  event loop still lags (p95 above 50 ms) or the probe reaches `REAX_PROBE_CEILING_DPS`, the probe is dropped with no
  capacity update. `scripts/bench_probe.py` measures what a validator host can take.
* **Validator deploy:** `ulimits: nofile 65536` for the validator service (a probe holds about 16 sockets per s1-fast /
  s1-pro UID). The reference budget of a probe starts after its requests are prepared.
* **Miners: new defaults, and how to get measured fully.**
  * Update to 1.2.0rc3, or set `REAX_PER_CALLER_CONCURRENCY=16` and `REAX_PER_CALLER_PER_MINUTE=240` now (old
    defaults: 2 and 20).
  * Batched (vLLM) or multi-GPU setups also raise `REAX_RUNTIME_INFLIGHT` (new, default 16: decisions in flight to the
    runtime) and `REAX_MAX_CONCURRENT` (default 4 -> 8).
  * `REAX_MAX_PENDING_PER_IP` 8 -> 32.
  * The miner's queue is served round-robin per validator, so one validator's burst cannot starve another validator's
    probe.
* **Transition.** Miners on the old defaults keep working: extra probe lanes are refused (no penalty) and such a miner
  is measured up to about 55/s. Faster miners are measured in full only after they update or set the two settings.
  Capacity is a moving average (α 0.3) and probes run about every 20 minutes, so a node reaches its new level within
  1–2 hours. Pay inside s1-fast moves from "equal per hotkey" to "in proportion to verified speed".

## 1.2.0rc2 — unreleased

* Entry floors per use-case track: the s1-fast model's score on the sealed pool + 1.5 points (demand_forecasting 24.8, risk_assessment 33.7).
Weights version stays 1200 (no scoring change). The wire protocol is unchanged. Security hardening of the competitions and a
way to run them outside the validator process, from the security review of 15 October 2026.

* **Review fixes (release review of 15 October 2026).** *Engine:* a worker container without `nvidia-smi` no longer aborts the
  cycle (the free-memory preflight and the optional per-PID check warn and skip); the compose worker service now gets
  `nvidia-smi` through a `utility`-only GPU reservation. OCI manifests and image indexes (buildx default) are sized
  (`unknown/unknown` attestations skipped); the speed denominator is `max(reference run, solo baseline)` per sample, the
  drift check is one-sided (only a slower reference is inconclusive) and also covers the latency-gate reference runs; the
  runner removes only images it pulled itself and never the reference or vLLM image. *Specialist:* vLLM starts through
  `sh -c 'umask 000; exec vllm serve "$@"'` so the worker uid can connect to its socket; a failure that is the host's (docker
  refused, socket not connectable, never healthy, every model of a group of several failing) no longer counts toward
  `MAX_ATTEMPTS`; the tokenizer must define the markers of the architecture's fixed chat template (refused before any GPU run
  otherwise), `{{ bos_token }}` renders empty when a tokenizer has none, Gemma 4 left the architecture allowlist. *Remote
  mode:* the engine weights and the spec leaders never raise on a corrupt state file (outage warning, inference only; a
  corrupt spec file no longer keeps paying the last leaders); non-finite or overflowing `saved_at` is rejected; champion
  fields are type-checked. *Config:* on finney `REAX_COMPETITIONS_MODE=local` must be set explicitly (unset = serving-only,
  with an error if rc1-style in-process variables are present). Worker `HOME=/tmp`. Documented: the heartbeat shows
  liveness, not freshness, and the worker parses miner input while holding the docker socket.

* **Competitions out of the validator process.** New `REAX_COMPETITIONS_MODE` = `local` (today's behaviour, the default) |
  `remote` | `off`. In `remote` the validator runs no competition cycle, needs no docker socket and no sealed data, and each
  round reads `engine-competition.json` and `spec-state.json` from `REAX_COMPETITIONS_STATE_DIR` (read-only). It pays
  exactly like local mode (engine champion, 14-day decay, confirmations; track leaders, entry floors, 75/17/8;
  withdrawn or deregistered competitors drop out to inference) and re-checks on every read that the hotkey still holds its uid
  and the commitment still matches. A missing, invalid or stale (> 26 h) file pays that competition's share to inference,
  with one warning. New CLI `reax-competition-worker [--engine] [--spec]`: only the cycles, chain read-only, no wallet, no
  weights, same GPU lock, preflights and anti-cheat, state files written atomically with a heartbeat. It runs as a container
  next to the validator without wallets or tokens, or unchanged on a separate keyless GPU box. `deploy/validator-1.2` now has
  a `validator` service without the docker socket and a `competition-worker` service with it; `deploy/competition-worker`
  is the compose file for the keyless box. New Dockerfile target `competition-worker`; the plain `validator` target is used
  by the compose file (no docker CLI).
* **Specialist tracks (security review F3, F6, F9, F10, F13).** The vLLM socket directory is created under
  `REAX_SPEC_WORK_DIR/sock/` (the same-path mounted, host-visible work directory), not in the container's `/tmp` (every
  specialist evaluation would have failed in the compose deployment). The near-duplicate check (`delta_sketch`, compare,
  save) no longer runs on the event loop. A submission's chat template is never rendered: a fixed template per architecture
  family ships in the package (`spec/templates/`), `*.jinja` is not downloaded, `chat_template.*` and the `chat_template`
  keys are removed from the validator's copy, `tokenizer.model`/`*.tiktoken` are removed when `tokenizer.json` exists;
  Gemma 4 is refused until its template is shipped. The vLLM container runs as `REAX_SPEC_UID` (default 10003:10003, not the
  validator's uid) with `--cpus 8 --memory-swap 48g --ulimit nofile=4096:4096`, local log driver capped at 2 x 10 MB, and
  `REAX_SPEC_DOCKER_RUNTIME`; downloaded model directories are made readable for that uid. vLLM responses are capped at 1 MiB.
  A specialist cycle is skipped (nothing recorded) with less than 60 GB free on the work directory's filesystem
  (`REAX_SPEC_MIN_FREE_GB`). One mechanism for the confidential-compute UVA fix: `REAX_VLLM_DISABLE_UVA=1` with the
  packaged `spec/ccfix/sitecustomize.py` (copied into the container's run directory); `REAX_CCFIX_DIR` is only for the
  s1-pro runtime in the compose file. The track entry floor of the evaluator now also honours
  `competition_split.TRACK_ENTRY_FLOOR` (`None` for now, so nothing changes yet).
* **Engine competition (F4, F5).** An image above 15 GB compressed (registry manifest, read before the pull) is refused;
  every path that does not end in a finished evaluation removes the image again (not-owner images, failed starts, aborted
  pulls). Every session measures the reference alone first; later reference runs more than 12 % away from that baseline
  (`TrackParams.reference_drift`) make the evaluation inconclusive. `REAX_ENGINE_NVML_CHECK=1` additionally fails a
  candidate that keeps using the GPU while paused. A candidate whose socket is replaced or that uses the GPU while paused
  is now a failed evaluation, not an internal error. The GPU free-memory preflight (2 x 12.5 GB) now actually runs for
  the docker runner (it was defined on the runner protocol instead of `DockerRunner`).
* `config.ENGINE_REFERENCE_IMAGE` is the published 1.2.0rc1 s1-fast runtime (`ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:d857a90e…`):
  a finney validator with the host side set up now runs the competitions instead of serving only.
* Docs: COMPETITIONS.md ("Where the competitions run"), ENGINE-COMPETITION.md, SPECIALIST-TRACKS.md, CONFIG.md, and the
  Security section of `deploy/validator-1.2/README.md` (the socket now lives only in the worker container).

## 1.2.0rc1 — unreleased

Weights version 1200 (scoring changes: validators on an older release are rejected by the chain once the owner raises
`weights_version`). The wire protocol is unchanged.

**What this release does, in plain English.** From **16 October 2026, 12:00 UTC** the miner weights are split 50/50:
half goes to serving (inference), half to competitions. Before that instant a 1.2 validator sets exactly the 1.1
weights.

* **50 % inference, 50 % competitions (profile `50/50`, active).** Inference 0.50; the engine champion 0.20; two use-case
  tracks 0.15 each (`demand_forecasting` and `risk_assessment`, 16–30 Oct 12:00 UTC; the next pair follows with a later
  release). Inside a track ranks 1–3 get 75 / 17 / 8 %. The engine champion's share halves linearly over 14 days. Any
  competition share nobody earns (no champion, a missing rank, a score at or below a track's entry floor, a struck
  champion, the decayed part) goes back to the inference part: never burned, never paid to the owner UID. The older
  `80/20` profile ships as an inactive profile. New share layer `competition_split.py`, see docs/COMPETITIONS.md.
* **The s1-pro pool is ON by default on every network, finney included.** Inside the inference part the shares stay
  s1-fast 0.35 / s1-pro 0.65. `REAX_FEATURE_S1_PRO_POOL` stays a fixed release variable: operators cannot set it on
  finney; development networks may still switch it off with `REAX_ALLOW_DEV_MODELS=1`. Every validator needs an
  80–96 GB GPU for the s1-pro reference runtime.
* **An unclaimed pool share is no longer burned.** A pool with no miner at or above `REAX_MIN_EMA` passes its share
  to the other enabled pools pro rata (s1-pro empty → s1-fast gets the whole miner part; same for tiered pools and tiers). Only when no pool has a
  miner does the miner part go to the owner UID, as in 1.1. `REAX_BURN_FRACTION` is unchanged.
* **Engine competition `reax-eng/1`** (fixed on for finney from the start time): validators measure submitted engine
  images against the reference engine in the same session (no network, read-only, fresh perturbed items, correctness on
  every timed answer, anti-cache probes, determinism re-run, latency gate, idle engine paused). The reference engine is a
  release constant (`config.ENGINE_REFERENCE_IMAGE`); `REAX_ENGINE_REFERENCE` is a development-network override. New
  tools `reax-engine-eval` and `reax-engine-submit`; the s1-fast runtime can also serve on a unix socket
  (`RUNTIME_UDS`). A validator that uses a custom endpoint of finney refuses to start (set `REAX_NETWORK=finney`).
  See docs/ENGINE-COMPETITION.md.
* **Specialist (use-case) tracks** `reax_subnet.spec`, commitment `reax-spec/1:<track_id>:<repo>@<revision>`: weights-only
  submission rules, a pinned network-less vLLM evaluation on fresh sealed draws, the 0.75 competence + 0.25 calibration
  score, the margin/second-draw confirmation, tensor-level hashes and weight-delta near-duplicate checks, persisted state,
  and the `reax-spec-eval` miner CLI (docs/SPECIALIST-TRACKS.md). The validator runs it next to the engine competition;
  its schedule is the one in `competition_split.py`. A per-track entry floor (`TRACK_ENTRY_FLOOR`, none yet) keeps
  weak submissions from being paid.
* **One GPU, one measurement at a time.** Engine measurements and specialist evaluations share one GPU lock, run as a
  background task (serving rounds are never paused) and each skips its cycle cleanly when the card has too little free
  memory (engine 2 x 12.5 GB, specialist up to 25 GB; measured: s1-fast 12.3 GB + s1-pro 58.7 GB).
* **A finney validator without the competition setup does not crash.** If the sealed pools, the engine weights or the
  published reference image are missing, it logs one warning and serves only (100 % inference weights).
* `deploy/validator-1.2/`: compose file and README for both reference runtimes (s1-fast on 127.0.0.1:8101, s1-pro on
  127.0.0.1:8102) on one 96 GB GPU, plus everything the competitions need: docker socket (with an honest security
  note), the engine and sealed-data directories, the spec work/state directories and the Hugging Face cache. New
  Dockerfile target `validator-competitions` (validator + docker CLI).
* `docs/hardware.md`: hardware per pool and an s1-pro miner quickstart. `scripts/make_selfcheck_reference.py` writes
  the `reax-selfcheck` reference answers of a pool from its reference runtime.
* Confidential-compute GPUs: `REAX_VLLM_DISABLE_UVA=1` loads `runtimes/vllm-ccfix/sitecustomize.py` (vLLM without UVA
  buffers, numerics unchanged); wired into the validator compose file and the specialist evaluator's vLLM containers.
  The s1-pro validator defaults are now `RUNE_GPU_MEM_UTIL=0.60` and `RUNE_MAX_MODEL_LEN=8192` (58.7 GB, bit-identical).

* **Tiered pools and capacity scoring are ON by default on every network, finney included** (like s1-pro;
  docs/DESIGN-POOLS.md). Two new pools next to s1-fast and s1-pro:
  * `s1-lite`: small GPUs incl. NVIDIA Jetson Orin. It serves the s1-fast model (Plumb-4B v5) as the author's Q8_0 GGUF on
    llama.cpp CUDA.
  * `s1-cpu`: CPU-only machines incl. ARM. It serves Gutsy 0.8B v0.4 Q8_0 GGUF on llama.cpp CPU.
  * Tier budgets inside the **inference** part: GPU 80 % (s1-fast 35 / s1-pro 65 inside it), small GPU 15 %, CPU 5 %. A
    tier-order rule keeps the average reward per miner in a tier at most half of the tier above it (the excess moves up).
    The order of the weight computation is: pools and tiers, then the serving vector, then the 50/50 competition layer,
    then owner burn as in 1.1. A pool or tier without an eligible miner passes its share pro rata to the other pools
    (never burned).
  * Inside each pool, weight follows the correctness-gated score times verified throughput, up to a per-pool target.
    Capacity probes load every UID of a pool at the same time, so hotkeys that share one machine share its throughput.
  * Reference decisions per round are capped per pool (`s1-lite` 240, `s1-cpu` 160 decisions, fair rotation; validator
    local `REAX_REFERENCE_CAP_S1_LITE` / `_S1_CPU`). The s1-lite tolerance (`tau_mean` 0.30) is set for a bf16 reference, so
    a validator verifies s1-lite against its s1-fast runtime (`REAX_S1_LITE_RUNTIME_URL`, default `127.0.0.1:8101`, no
    extra GPU memory) and s1-cpu against a CPU reference service (`127.0.0.1:8104`, 8 threads) that
    `deploy/validator-1.2` now starts. A validator without a tiered reference does not crash: it logs a warning and
    leaves that pool unscored.
  * `REAX_FEATURE_TIERED_POOLS` is a fixed release variable like `REAX_FEATURE_S1_PRO_POOL`: operators cannot set it on
    finney (development networks may switch it off). `REAX_ALLOW_DEV_MODELS=1` is now refused on every spelling of
    finney's endpoint.
  * New runtime sidecar `runtimes/s1-gguf` (CPU image for linux/amd64 and linux/arm64, CUDA image, Jetson recipe), slim
    multi-architecture miner image (Dockerfile target `miner-s1`, `requirements/lock-s1.txt`), cross-backend calibration
    scripts (`scripts/calibration_gguf`) and a tiered localnet end-to-end run (`scripts/localnet/e2e_tiered.py`).
* **Reference isolation (all pools):** each pool's references run concurrently with their own time budget and failure
  share; a hung or failing reference runtime leaves only that pool's miners unscored for the round instead of skipping
  the round for everyone. The round is skipped only when every pool's reference fails.
* Specialist evaluator, first run against the real pinned vLLM 0.30.0 image on an RTX 5090: the container's `/tmp` tmpfs
  is now mounted `exec` (Triton could not load its helper modules, vLLM never started), start-up uses `--enforce-eager`,
  the candidate's GPU budget is 25 GB computed from the detected card size, `REAX_VLLM_DISABLE_UVA=1` mounts a
  read-only `sitecustomize.py` that disables vLLM's UVA buffers (confidential-compute GPUs), and `reax-spec-eval`
  reports start-up time, scoring time and items per second.
* Optional `reax-router` service: a client-facing router that forwards signed requests to miners (read-only
  validator score selection, sequential failover, circuit breakers, bounded shadow verification with a separate
  strike ledger). Disabled until an operator configures and enables it; see docs/ROUTER.md.
* CPU-only pinned Plumb prompt/token accounting in the router; token counts are never taken from miner-supplied usage.
* The runtime client retries once on a dropped sidecar connection.
* The repository moved to the `reaxlabs` organisation; source and GHCR references use `reaxlabs`.
* MIT licence, `CONTRIBUTING.md`, issue and pull-request templates.
* Validator robustness: every chain read has a deadline (a stalled RPC skips the round instead of freezing the loop,
  and a hung metagraph read is not started twice); late `set_weights` outcomes are logged; the sidecar client reads
  runtime responses split across TCP segments in full; integers too large for a float are a contract error; a
  committed miner that is not queried (other manifest, disabled pool, malformed commitment) is explained once in the log;
  a recycled UID no longer inherits the previous hotkey's role-change shield cooldown.
* Compose files cap Docker logs (json-file, 5 x 50 MB per service).
* `reax-selfcheck`: checks a miner's runtime sidecar against 640 published reference answers with the release
  scoring constants before registration (docs/MINER.md). No wallet or chain access.
* Calibration addendum: the s1-fast reference path on RTX A6000, L40S and RTX 5090 stays inside the release tolerances.

## 1.1.0rc2 — 2026-10-05

Weights version stays 1100; the wire protocol is unchanged, so rc1 and rc2 validators and miners interoperate.

* **Security fix:** a request carrying the `X-Bittensor-Hotkey` header twice could pass the miner's validator
  allow-list with one value and the signature check with another. Miners now refuse duplicate `X-Bittensor-*`
  headers and serve only when the verified signer is the allow-listed validator. Upgrade miners.
* A UID that keeps changing its commitment no longer keeps a frozen score; weights count only roles on the current manifest.
* The reference step of a validator round is time-bounded, so a hung sidecar skips the round with an error instead
  of stalling the validator.
* Runtime URLs must parse as loopback; `REAX_ALLOW_PRIVATE_IPS` is refused on every mainnet network name.
* Docker: targets `miner` (default) and `validator`; caches live under /tmp so the GPU path works for any `--user`.
  CI builds, smoke-tests and then pushes `reax-miner`, `reax-validator` and `reax-s1-fast-runtime` images to GHCR.
* Docker: CI also builds, smoke-tests and publishes the tiered-pool runtime images `reax-s1-cpu-runtime` (one index,
  linux/amd64 + linux/arm64, built natively per architecture and merged), `reax-s1-lite-runtime` (CUDA, amd64) and
  `reax-s1-lite-runtime-jetson` (Jetson Orin, arm64; build and file/import-metadata check only, no GPU smoke). Same
  rules as the other images: push only on a manual dispatch with `publish_images`, `sha-` tags written once, a `v*` tag
  promotes the `sha-` manifest byte for byte.
* Docs: rollback uses the 1.1 image with `REAX_MECHANISM=mpm1`; emission-brake caveats; per-UID role confirmation.

## 1.1.0rc1 — 2026-10-03

Weights version 1100.

* **New default mechanism `reax-s1` (`REAX_MECHANISM=s1`).** A miner commits on chain to one role,
  `reax-s1/1:<model>:<manifest_id>`, and serves `s1-fast` (Plumb-4B) or `s1-pro` (Surogate Rune 26B-A4B v3)
  with the reference prompt, readout and typed-decision answer schema (choice / noul / score with probabilities).
* Versioned model manifests pin weights (every file's sha256), runtime image digests, package pins and the readout.
* Runtime sidecars (`runtimes/s1-fast`, `runtimes/s1-pro`) run the reference readout on loopback.
* Verification in logit space: per-item distance, round mean ≤ 0.20 scores `0.8 + 0.2 × speed`, mean > 0.20 loses
  the round, mean > 0.50 or any item > 2.0 is a strike. Thresholds and latency bounds come from GPU calibration
  (docs/CALIBRATION.md).
* Pools with fixed shares; the s1-pro pool is OFF by default (development networks only in this release).
* Role changes take effect only when two consecutive metagraph reads agree for that UID.
* The runtime token is required in s1 mode and verified with an authenticated test decision at startup; the miner
  retries its commitment every 10 minutes until the chain shows it; a metagraph read that changes more than 20 %
  of committed roles is skipped until confirmed; a full burn without an owner UID sets no weights.
* The `mpm1` mechanism stays available unchanged as `REAX_MECHANISM=mpm1`.
* Fixed: the Docker image crashed on GPUs because torch compiles Triton helpers with a C compiler the slim base
  did not have; the image now installs gcc.

### Upgrading from 1.0.0rc3

1. Upgrade validators first (they understand both mechanisms; `REAX_MECHANISM` selects which one runs).
2. Then miners start their runtime sidecar and publish their commitment `reax-s1/1:<model>:<manifest_id>`
   (docs/MINER.md). A miner without a matching commitment is not queried while validators run s1.
3. Then the subnet owner raises `weights_version` to 1100.

Rollback: run the 1.1 images with `REAX_MECHANISM=mpm1` set explicitly, lower `weights_version` to 1000 and move the
validator state file aside (or use a fresh `REAX_STATE_PATH`).

## 1.0.0rc3 — 2026-10-03

Weights version 1000.

* Probabilities are read at protocol temperature T = 4 (`softmax(logits / T)`), which defeats a "copy the top choice" cheat.
* Rounds use 12 items with a 10-of-12 gate.
* Fixed: the metagraph guard no longer stops a validator forever after an owner trim; the global cap on unfinished
  requests is 4096, the per-IP cap stays 8.
* Development mode is refused on finney; a full burn without an owner UID sets no weights.
* `scripts/calibrate.py` reports round-pass probabilities and adds an "oracle top choice" cheat and `--cheat-model`.

## 1.0.0rc2 — 2026-10-03

* Fixed: a miner answering with a huge integer probability made every validator round abort. Parsing now rejects it.
* Fixed: slow-drip request bodies could hold miner connections forever (5 s body deadline, per-IP and total caps).
* Validators score 6 rounds before their first weights.
* Docker: run with `--user`; CPU fallback is refused; bounded `set_weights` wait; directory fsync for state.

## 1.0.0rc1 — 2026-10-03

First release candidate of the Minimal Production Mechanism (`reax-mpm/1`, weights version 1000).

* Miner: aiohttp endpoint `POST /v1/decide`, request verification, validator-permit and stake check, per-validator
  concurrency and rate limits, signed responses.
* Validator: fresh items per miner per round, local reference computation, total-variation scoring with a fidelity
  gate and a speed term, persisted EMA, owner-UID fallback, SSRF-safe outbound requests.
* Features from the long-term design are disabled behind `REAX_FEATURE_*` (docs/FEATURE-FLAGS.md).
* Local subtensor end-to-end harness, adversarial miners and a threat model.
