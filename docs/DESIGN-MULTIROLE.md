# One hotkey, several roles: all inference models + one competition entry (design, 20 Oct 2026)

Status: design + implementation on branch `fusion/sn92-hotkey-dualrole-allmodels-20261020` (from the multinode branch,
793fe94). Consensus change: validator release with a `WEIGHTS_VERSION_KEY` bump, plus the miner/submit tooling. Two release
constants in `elements.py`, `MULTIMODEL_START_BLOCK` and `DUALROLE_START_BLOCK` (None = held, like
`ENGINE_TRACK_START_UTC`): from that block on, validators accept multi-model inference elements / a second (competition)
field; the miner and submit tools refuse to write them earlier unless `--force`. Constants, not environment flags,
because every validator must apply the same rule.

Two requests, one mechanism:

* **All models on one hotkey** (cxmplex, #92/DM, 20 Oct; our decision 20 Oct 13:05Z): one hotkey serves s1-fast, s1-pro,
  s1-lite and s1-cpu, each with the SAME per-model cap that a separate hotkey of that model gets. Why: the smaller tiers
  earn too little per hotkey to pay back a registration, so today they either stay empty or are farmed with many cheap
  hotkeys.
* **Inference + one competition on one hotkey** (#92, 19 Oct; card 17931 "All 3"): a hotkey serves inference and holds
  one competition entry (use-case/generalist track or engine track) at the same time, so moving into a competition needs
  no second registration and the entry is not pruned while it is evaluated. Assessment: DESIGN-MULTINODE.md section 11.

Both reduce UID pressure (MECHANISM-V2 P1/P2) and neither pays a hotkey more than the same work spread over separate
hotkeys gets today.

## 1. Commitment format

Facts (finney, block 9,324,949, 20 Oct): Commitments pallet `MaxFields = 3`, `MaxSpace = 3100` bytes per rate-limit
interval, `InitialDeposit = FieldDeposit = 0`. A `Raw` field holds at most 128 bytes. Every `set_commitment` replaces all
fields and carries ONE block number for the whole commitment. The SDK's `NeuronCommitment.data` concatenates the Raw
fields without a separator; the per-field list is in `NeuronCommitment.fields`.

**Elements.** A commitment is 1-3 Raw fields; each field is exactly one element:

| Element | Text | Max bytes |
|---|---|---|
| inference, one model (unchanged) | `reax-s1/1:<model>:<manifest_id>` | 33 |
| inference, several models (new) | `reax-s1m/1:<model>:<manifest_id>,<model>:<manifest_id>[,...]` | 11 + 4 × 24 + 3 = 110 |
| use-case / generalist track (unchanged) | `reax-spec/1:<track>:<repo>@<40 hex>` | ≤ 128 |
| engine track (unchanged) | `reax-eng/1:<track>:<image>@sha256:<64 hex>` | ≤ 128 |

Rules (strict, like every parser in this repo; anything else = the whole commitment is invalid and the hotkey has no
role, exactly as a malformed commitment today):

1. At most one **inference** element (`reax-s1/1` or `reax-s1m/1`) and at most one **competition** element (any
   competition prefix this release knows: `reax-spec/1`, `reax-eng/1`, and the generalist prefix when it ships). At least
   one element. No two fields of the same kind, no unknown field, no empty field, no non-UTF-8 field.
2. `reax-s1m/1` lists 2-4 distinct models, each with this release's manifest id for it; a model listed twice, an unknown
   model or a malformed manifest id invalidates the commitment. Order does not matter (the canonical form sorts by the
   pool order s1-fast, s1-pro, s1-lite, s1-cpu; validators accept any order). A one-model list must use `reax-s1/1`.
3. A model whose manifest id is not this release's is not scored (the other listed models still are), as a single-model
   hotkey on an old manifest today. A model whose pool is disabled on this validator is ignored the same way.
4. **Old single-field commitments keep working unchanged**: one field holding one element is exactly today's format.

Before the start blocks (or while held) a validator reads only single-field commitments, so a miner that
writes a multi-element commitment early simply is not scored until the start. The tooling refuses to write one before the
start unless `--force`.

**Writing.** Every write replaces all fields, so every writer merges. A writer keeps an element it cannot read (an
unknown `reax-` prefix of a later release, e.g. the generalist track) verbatim, compares fields stripped as validators
read them, refuses when the current commitment is ambiguous, and the miner writes only after a second fresh read showed
the same fields (one bad metagraph read must never make it write over an entrant's element). In detail: the miner (`reax-miner`, which re-publishes its
commitment when it differs from what it serves) writes `[inference element, current competition element]`; the submit
tools (`reax-spec-submit`, `reax-engine-submit`) write `[current inference element, new competition element]`. A writer
never drops an element it did not create unless told to (`--drop-inference` / `--withdraw`). The miner compares only its
own element with the chain, so an entrant's competition element is never overwritten by the miner's 10-minute re-publish.

## 2. Which block orders a competition entry

The chain keeps one block per commitment, and a rewrite of one element moves it for both. Competition ordering must not
let a dual-role entrant who updates its inference element look newer than a copier who committed the same model later.

**Rule: the order that decides priority comes from the competition worker's own persisted first sighting, never from
the commitment block.** That is already how the competition code works:
* use-case / generalist tracks: `Submission.first_seen_block` (the block at which the worker first saw the submission
  key `hotkey|<element text>`), used for the cohort order, the exact tensor-hash copy check, the near-duplicate
  ("derived from") check and playoff ties. The key is the competition element only, so an inference rewrite neither
  changes the key nor the stored block;
* engine track: `first_seen[digest]` = the minimum block over every sighting, so a later rewrite never moves it.
Both live in the worker's state, which validators in remote mode read: one writer, the same for everybody.

The competition view (section 3) carries the **chain's commitment block**, as today. It is read only for exact-score
ties and for the order inside one coldkey (keep-alive per-coldkey limit, provisional one-per-coldkey). There a rewrite
can only cost the rewriting hotkey an exact tie or its coldkey-internal order: deterministic on every validator, and
nobody else can use it. (The first draft kept a validator-local "element block" here; critic round 1 found that
validators with different histories would then pay different UIDs, so it was removed.)

## 3. Views: each subsystem sees one element

`role_views(snapshot)` returns two snapshots with the same neurons:

* **inference view**: `commitment` = the inference element (or ""), `commitment_block` = the chain block (0 if none);
* **competition view**: `commitment` = the competition element (or ""), `commitment_block` = the chain block (0 if none).

The serving rounds and weights use the inference view; the engine, specialist and generalist code (unchanged) gets the
competition view. Consequences, all by construction:

* A new competition element does not reset the inference score (the inference role string is unchanged), and a new
  inference element does not re-queue the entry (the spec submission key is `hotkey|<element text>`).
* Old single-element commitments produce the same views as today's raw snapshot (one view has the element, the other
  "").

## 4. All models on one hotkey (validator scoring)

**Scoring slots.** Everything that is per UID today (moving average, capacity, strikes, rotation, probe budget, lanes,
held-average count) becomes per **slot** = (uid, model). Slot key `uid + SLOT_STRIDE × index(model)` with
`SLOT_STRIDE = 1 << 16` and index s1-fast 0, s1-pro 1, s1-lite 2, s1-cpu 3, so the existing int-keyed code is reused
and the slot of a one-model s1-fast hotkey keeps its UID as key. State migration on load: a stored entry with role
`<model>:<manifest>` for a model with index > 0 is moved to its slot key (no score reset for anyone). The role string per
slot is `<model>:<manifest_id>` as today, so a manifest change resets only that model's slot.

**Queries.** A slot is queried like a separate hotkey of that model: own request, own verification sample, own score,
own strike (a strike zeroes only that model's slot; it is a different backend). Rotation and the reference cap count
slots per pool.

**Probes.** All slots of all pools are probed in the one common window that already covers every UID of every pool
(review H2: "a probe measures every queryable UID of each pool at once"). So one GPU serving two models of one hotkey
splits its throughput between the two slots exactly as it would between two hotkeys today. Probe shards group slots
of one UID together (same IP/coldkey: `probeshard.sibling_groups`).

**Weights.** `compute_tiered_weights` runs on slots: per pool, each slot is a miner of that pool with that pool's cap
(`CAPACITY[pool].multiplier(capacity of the slot)`). The tier-order cap counts slots, i.e. a hotkey with s1-fast and
s1-pro counts as two GPU-tier miners, exactly as two hotkeys. The slot weights are then summed per UID before the
competition layer. **Caps are per hotkey and model, never shared and never multiplied**: a hotkey can hold at most one
slot per model, so it gets at most what one hotkey per model gets today.

**Anti-farming.**
* No cap multiplication: max 4 slots per hotkey = max what 4 separate hotkeys get, and fewer UIDs used.
* s1-lite farm (MECHANISM-V2 3.6): a lite slot on a big GPU hotkey is paid exactly like a separate lite hotkey on the
  same box (same cap 32/s, same common probe window). Today the same operator registers extra lite hotkeys; this only
  saves the registrations. The contest_watch lite alert counts slots instead of hotkeys. The order-ratio question is
  unchanged by this design.
* Siblings / same IP: nothing new to detect. Slots of one hotkey are by definition one operator on one endpoint, probed
  together; the existing rule (no IP rules, Florian 16 Oct) stands.
* Shared backend across models is impossible to fake: every slot's answers are verified against that model's own
  reference; a s1-pro slot answered by s1-fast weights fails the distance check and strikes.

**Miner.** `reax-miner` serves several models behind one endpoint: `REAX_MODELS=s1-fast,s1-pro,s1-lite` (default: the
existing single `REAX_MODEL`), one backend per model (each with its own `REAX_<MODEL>_RUNTIME_URLS` and, with
multinode, its own workers file), requests routed by `model` + `manifest_id`, per model its own concurrency slots and
its own per-validator limits (a model is not slowed by another model's probe lanes, exactly as separate hotkeys),
`/health` lists all models. Commitment: `reax-s1/1` for one model, `reax-s1m/1` for several.

## 5. Inference + one competition entry (dual role)

**Evaluation.** Unchanged: entries are evaluated on our competition worker (use-case/generalist: the model in our
sandbox on sealed items; engine: the image on our candidate boxes). The miner's serving GPU is never the one measured,
so running both cannot slow either measurement, and the copy checks, near-champion rule, per-coldkey limits and the
owner approval gate apply exactly as today.

**Pay.** Serving and competition pay come from separate budgets and add up per UID (DESIGN-MULTINODE §11: every
competition payment comes out of the competition share and scales the serving vector by `1 − claimed`). Changes in
`combine_weights`:

1. **Keep-alive is a top-up.** Keep-alive replaces the serving income an entrant gives up. A dual-role entrant still
   serves, so its keep-alive is `max(0, each − own)`, where `each = min(per_entry, cap × share / n)` as today and `own` =
   the UID's serving weight in the same unit (`base[uid] / miner_part`, fraction of the miner part before competition
   shares). The provisional keep-alive (paid to an entry that waits for its first evaluation, so that it is not pruned)
   goes to pure entrants only: a hotkey that serves is not at that risk.
2. **Slots are not crowded out.** Only entries whose top-up is positive (`own < per_entry`) compete for the
   `KEEPALIVE_MAX_ENTRIES = 8` slots, and pure entrants (no serving weight) come first in the slot order; within each
   group the order is today's (score, then commitment block, then uid). So big serving hotkeys cannot fill the 8 slots with
   cheap above-floor entries at no cost to themselves and push pure entrants out.
3. **The median is untouched by dual-role UIDs.** Every keep-alive candidate (before the cut to 8), every provisional
   entry, every rank and the engine champion are left out of the median (today only the paid ones are; for pure entrants
   this is the same, because they have no serving weight).
4. Ranks and the engine champion share are paid on top of serving, unchanged: they reward a winning model/engine, and
   a separate hotkey gets the same today.

**Keep-alive interplay, worked example** (per_entry = 0.004 of the miner part, cap not binding): a pure entrant gets
0.004 as today; a dual-role entrant with serving 0.0015 gets 0.0025 top-up (total about 0.004: its serving part is scaled
by `1 − claimed` like everybody's); a big s1-pro hotkey serving 0.006 gets 0 and does not take a slot.

**Engine / generalist tracks.** The competition element can be any competition prefix; one per hotkey. The engine
track's first-committer rule (engcomp `first_seen` = min block per digest) and, in the release lineage that has it,
opening rule A (pre-opening commits count as made at the opening) apply unchanged to the competition element. The generalist track (G3) adds its prefix to the competition prefix
list in its release; nothing else here changes.

**Bound per UID.** Inference share (pool caps) + one competition share (rank share or engine share, per-coldkey limits).
One operator reaches the same today with two UIDs.

## 6. Migration

* Old single-element commitments: identical behaviour (views equal today's raw snapshot; slot key = UID for s1-fast;
  s1-pro/s1-lite/s1-cpu state entries migrate to their slot key on load without reset).
* Before the start blocks: validators ignore multi-field commitments (no role), so nobody gains by writing early;
  miners keep their single-element commitment until the start.
* After the start: a miner adds models by writing `reax-s1m/1` (its existing models' slots keep their scores: the role
  string per model is unchanged); an entrant adds `reax-s1/1`/`reax-s1m/1` next to its entry (its entry keeps its key and
  the worker's first sighting); a server adds a competition element (its inference score is unchanged).
* Holding or rolling back multimodel never voids a competition element next to a multi-model element: that hotkey keeps
  its entry and only loses the inference role.
* Old validators (not upgraded) see the concatenated text, cannot parse it and do not score multi-element hotkeys. The
  `WEIGHTS_VERSION_KEY` bump makes upgrading mandatory, as for every consensus release.
* Rollback: to hold either half, set its start constant back to None (rebuild): multi-field / multi-model commitments
  have no role again and their slots decay. Holding dual role again WITHDRAWS every dual-role competition entry (the
  competition code sees no element, a later re-commit is a new entry in a later cohort): announce it to entrants first. To go back to a release OLDER than multirole, stop the validator and run
  `scripts/state_unslot.py <state> <copy>` first, then start the old release on the copy: an older release does not
  know slot keys and would otherwise reset every s1-pro / s1-lite / s1-cpu score.

## 7. Release plan and risk

**One combined validator release** ("multirole"), after the multinode miner release (Mon 26 Oct), with two independent
start constants (`MULTIMODEL_START_BLOCK`, `DUALROLE_START_BLOCK`), so either half can be held in the build (None) without
touching the other. Why one release: both need the same commitment parser, the same views, the same state
migration and one weights-version bump; two releases would mean two consensus switches and two migrations for miners in
one week.

Risk call: medium. The weight math is reused (slots are just more int keys); the new consensus surfaces are the parser,
the slot migration and the keep-alive top-up/slot rule, all covered by tests and a localnet run (multi-model hotkey vs
separate hotkeys: equal pay within noise; dual-role entrant keeps its score through an inference rewrite and vice
versa). Earliest realistic dates (proposal; the date is set by us only after Florian's GO): build + CI + localnet by
Thu 22 Oct, #92 notice Fri 23 Oct, validator deploy Tue 27 Oct with both start blocks ≈ 24 h after the deploy
(Wed 28 Oct), so every validator has upgraded before the first multi-element commitment counts.
