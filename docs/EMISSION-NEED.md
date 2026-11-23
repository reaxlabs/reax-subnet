# Emission follows need

Decision: Florian, 26 Oct 2026 (DECISIONS `emission-follows-need-burn-1030`, `burn-announced-1026`, standing rule
`pragmatism-simple-rules`); agreed with the subnet owner. Announced in #92 on 26 Oct 2026, 21:0x UTC. Effective from the
first weight set at or after **Fri 30 Oct 2026, 14:30 UTC** (the engine crown slot). Code: `competition_split.py`
(`NEED_*`, `INFERENCE_SHARE_STEPS`, `TRACK_PAUSES`, `combine_weights(..., paused=, inference=)`), `need.py` (operator
file), tests `tests/test_emission_need.py` and `tests/test_pause_runout.py`.

The principle: SN92 pays for what it needs. The inference share follows measured customer demand, competitions are paid
while they bring real improvements, and emission that is not needed is not paid out (burned). The numbers below are our
current decisions, not guarantees; we change them at our discretion.

## 1. What "burned" means on SN92

* Our validator (UID 3, the subnet owner hotkey `5H3xdf9f...`, about 95 % of the validator stake; the other three
  validators copy its vector) puts every unneeded part of the miner weight vector on **UID 3**.
* Subtensor (runtime 486) pays no miner incentive to an owner hotkey. SN92's `RecycleOrBurn` is `Burn`, so that alpha
  is destroyed. Validator dividends of UID 3 are not affected.
* Weight that is simply left out is **not** burned: the chain renormalises each validator row, so it would flow to the
  other miners. Burning needs explicit weight on the owner UID. This is the same path as `REAX_BURN_FRACTION` (still 0).
* **Chain side effect (check before raising the burn).** Since subtensor PR #3071 (Aug 2026), the chain records
  `MinerBurned[92]` (burned fraction of miner incentive, Burn and Recycle alike) and scales SN92's share of the network TAO
  block emission by `(1 - MinerBurned)`. Simulation from live storage (26 Oct): 10 % burned -> about 0.89 x TAO into SN92's
  pool, 20 % -> 0.78 x, 30 % -> 0.67 x. The alpha issued per block does not change. Evidence:
  `jobs/emission-follows-need-20261026/research/chain/NOTES.md`.

## 2. Inference share

* From Fri 30 Oct 14:30Z: **0.40** of the miner part (was 0.50 nominal; it was 0.50 to 0.57 in practice because unclaimed
  competition shares flowed to inference).
* Inside inference nothing changes (pools, caps x4, tier budgets, per-coldkey rules): `combine_weights` scales the serving
  vector to exactly the share; the legacy keep-alive and the EU carve-out stay inside it.
* **Measured demand (guidance for our review, not an automatic rule).** D = highest one-hour average of customer decisions
  per second served successfully by the System1 gateway over the last 7 days, all tiers and models, without our own
  tests/probes (VictoriaMetrics). Needed capacity ~ 10 x D. Supply = verified s1-fast + s1-pro miner capacity
  from the validator probes. On 26 Oct: D ~ 35-42/s, supply ~ 90,000/s (more than 2,000 x the need), so demand does not
  argue for more than 40 %. We review it when we look at the slots and change the share when it makes sense.
* A change is one row in `need.json` (below) or in `INFERENCE_SHARE_STEPS` at the next release. Sanity bounds 0.05-0.80
  (typo guard only).

## 3. Pausing a competition track

* **Our call, no automatic threshold.** When progress flattens (for example a crowning round brings no meaningful
  improvement), we pause the track. We restart it when there is new work (a new model in our lineup, a new task).
* **During a pause** (`paused` contains `"engine"` or the track id): the track stays open (entries are evaluated and
  measured, results published, the champion keeps the title). One run-out rule applies to every track (Florian,
  28 Oct 2026; public wording is still a draft for review on 29 Oct): the champion's full rank-1 share holds through
  **three days from crowning**, then decays **linearly to zero at seven days**. This always uses title age:
  a five-day-old title starts the pause at half its champion entitlement; a title at least seven days old starts
  at keep-alive only. There is no full-payout notice period. Co-champions use each member's crowning time and
  fraction; an absent or unapproved member's fraction is never redistributed. Former-champion places end immediately
  with the pause; they cannot be restored as champion while paused (including a vacant title).
* **Keep-alive stays.** The existing paused-track protection, waiting, provisional, override and legacy paths and their
  caps are unchanged (engine: 0.5 x median, within 5 % of the engine share; generalist: 1 x median, within 25 % of its
  share). The run-out tops up the champion to its remaining entitlement only when that exceeds its keep-alive; it
  never takes away keep-alive or pays it twice. Generalist rank 1 is 75 %, split equally over co-champions.
  Engine entitlement is its current pot (the existing 100 % to 50 % decay over 14 days) times the champion fraction
  frozen at pause start, times the title-age run-out factor. Former slot expiry during a pause cannot increase
  that fraction; former places' fractions burn rather than moving to the champion.
  Engine top-ups fit inside the share remaining after its existing keep-alive payments, split by rank-1 fraction.
  Every unpaid part goes to the owner UID through the existing burn path.
* A pause starts/ends with the first weight set after the time written in `need.json` (or `TRACK_PAUSES`).

## 4. How we change it: `need.json`

File next to the award approvals on the validator node (`<dir of REAX_AWARD_APPROVALS>/need.json`, i.e.
`/w/approvals/need.json`, the same place as `ka-override.json`). Read on every weight set; no restart.

```json
{"inference": [{"from_utc": "2026-11-03T14:30:00+00:00", "share": 0.35, "reason": "demand review 2 Nov"}],
 "pauses":    [{"track": "generalist", "from_utc": "2026-11-03T03:00:00+00:00", "until_utc": null,
                "reason": "progress flattened (slot 2 Nov)"}]}
```

* Strict: exactly these keys, ISO times with timezone, not before NEED_START_UTC, `until_utc` null or after `from_utc`,
  at most 20 rows each, non-empty reason. A refused or half-written file keeps the **last good content** of the running
  validator in force (else the release constants), with a warning. **Write it atomically** (temp file + `mv`). A pause
  key that is not "engine" or an active track id is logged and ignored. To end a pause, set `until_utc`.
* The run-out clock uses the pause row's **`from_utc`**, or the corresponding release `TRACK_PAUSES` start, never
  the file mtime, validator restart time or first read. Overlapping or touching intervals form one continuous pause;
  extending one preserves its start. A real gap followed by another pause gets a new pause start; title age never resets. Malformed
  rows (including a missing `from_utc`) retain the last good file, as above; without one the release defaults apply.
  A direct `combine_weights` caller lacking a finite, active pause start leaves that track's normal payout unchanged.
* Generalist `RankedLeaders.crowned` carries each member's unix crowning time, and `champions` distinguishes current
  holders from former places even when the title is vacant. Engine holders are passed separately from former payees;
  their crowning time is the matching digest/hotkey crowning event's `slot_utc`. Legacy v1 block-only crownings
  (including UID 62 while the bridge holds) use the release engine opening UTC/block anchor and the existing 12 s
  block clock, the inverse of `block_at_utc`. Without either UTC or an anchor, the operator tool refuses scheduling
  until the timestamp is recovered; the weights retain the existing missing-time fallback.
* **Operator notice guard (not a payout rule):** run `python -m reax_subnet.needctl` on the validator host.
  Supply `--track engine|generalist`, `--state <validator state.json>` (repeat for an active frozen v1 bridge),
  `--from-utc <UTC>`, and `--reason <text>`. Default path is `/w/approvals/need.json`.
  If any title holder will be at least seven days old at pause start, `--announced-at <UTC>` is mandatory and
  pause start must be at least 24 h after that announcement. Unknown holder times or missing track state refuse.
  The default is a dry run printing the exact step; `--write` takes an operator lock, backs up the existing file,
  validates the entire result with `need.parse`, and atomically replaces it with fsync. No extra keys enter need.json.
  The operator must supply the same current snapshots as the validator, including its bridge if active.
* Procedure: decision (Florian or by his delegation) -> release owner (`sn92-fast-launch-20261015`) writes the file,
  checks the next weight set (UID 3 weight, champion run-out, former payees at keep-alive level), posts the receipt to board #22, and the
  FAQ/#92 get a short note.

## 4a. Edge cases (critic round 1)

* The inference share never takes more than the room the paid competition parts leave (a share above it is cut, the
  competition payouts are not squeezed).
* A pool without an eligible miner leaves its part on the owner UID inside the serving vector; that part counts inside
  the inference share and is burned with the rest.
* A rejected or unapproved ranked entry of a paused track gets no keep-alive from the pause (the existing protected /
  provisional paths still apply).
* The keep-alive unit is the median of the unscaled serving vector, as before; with inference at 0.40 a keep-alive is
  about 2.5 x a median server's real weight (was about 2 x).
* A paused engine's payees share the engine keep-alive cap (5 % of the engine share) with the waiting engine entries.
* Not covered: a validator without the competition layer (`engine is None`, dev setups) sets the old serving weights;
  the remote fallback after an exception uses the release constants only (not need.json).

## 5. Interaction with live and planned changes

| Change | Interaction |
| --- | --- |
| caps x4 (post7.1, live since 26 Oct 20:08Z) | none: caps act inside the serving vector, which is scaled as a whole |
| waiting keep-alive hotfix + abuse clause | unchanged; `combine_weights` keeps every keep-alive path, paused-track champions are added as protected keep-alive entries first |
| keep-alive override (`ka-override.json`) | unchanged, separate file |
| commit-reveal (worker Tue 20:00Z, validator readers post7.2) | none on weights; run-out also reads each champion's recorded crowning time |
| engine v2 (Fri 30 Oct crowning, branch rc11p8-v2) | integrate this patch into the v2 image; `paused` uses the key "engine" for every engine track id; the v1 bridge payee is an engine payee like any other |
| generalist cutoff (gen post4.2, Thu 29 Oct) | none |
| REAX_BURN_FRACTION | kept on top (owner burn fraction + unneeded part) |

## 6. Rollback

`NEED_RULE = False` (release) restores the previous weights exactly. Between releases: a `need.json` row with share 0.50
and no pauses restores the old inference share but still burns unclaimed competition shares.


### Approval clock for future crowns (staged 2 November 2026)

Only crownings at or after `2026-11-02T13:00:00Z` use
`start = max(crowned_at, own signed award approved_at)`: full through start + 3 days,
linear decline to zero at start + 7 days. Each co-champion uses its own approval
for its exact track, hotkey and submission. No approval (or unknown/non-finite time)
returns a full run-out factor; the independent award gate still decides payment.
Every earlier crown retains the previous clock exactly, including UIDs 83, 114 and
240; their separate make-good must not shift this clock or pay the delay twice.

The notice scheduler uses this same effective start for the seven-day notice test;
a new-rule holder with no verified approval time is skipped with a stderr warning.
Its clock has not started; a later approval begins at full share. Old crowns keep the previous notice test. Keep-share and
frozen relay stamps are unchanged. The existing paused-track keep-alive floor is
unchanged: it can already top up a declining title share, and remains after the
run-out ends at start + 7 days. There is no new seven-day keep-alive expiry.


### Allocation delivery estimates (staged 2 November 2026)

The monitor's emission parameter is `133.2 / 360` alpha per block per unit sent weight
(measured on chain 2 November 2026, after the miner slot), not the nominal `2950 / 7200`.
It multiplies each committed row's actual post-miner-slot vector weight by the
observed blocks since the previous set and this configurable alpha-per-block
parameter, and keeps a running estimate per canonical row hash. Unknown intervals
are reported, never replaced by 360 blocks. Scaled shares warn without failing.
These interval estimates are not chain-payment receipts; the final true-up uses
chain emission readback. Pin the monitor's approval public key to the validator's
reviewed key and provide the matching `reax_subnet.award` verifier; otherwise its
engine run-out approval check warns "unknown" and skips the factor check.
