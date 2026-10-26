# Engine slots — release-owner installation notes

These are review artifacts. No script was installed outside this worktree, no timer was created and no state was
published. The release owner integrates and deploys them together with the candidate worker image.

## Repository code

Integrate the `rc11p6-engslots` commits onto the rc11.post6 candidate. `competition_split.ENGINE_CROWN_SLOTS_UTC` is the
single engine calendar: Tuesdays and Fridays 14:30 UTC, 27 October 2026 through 30 March 2027. Empty restores the
old immediate-crown path. The generalist calendar is unchanged.

`update_champion(..., now=...)` must receive aware UTC cycle time in production; the omitted-time default is the
fixed Unix epoch for deterministic callers/tests. `EngineCompetition` passes its injected UTC clock. Completed
confirmations are persisted in `challenger.confirmed_at` and `challenger.slot_utc`; missing fields in an old state
mean it is not confirmed pending. Track JSON also exposes `next_slot_utc`, a public `pending` summary, and
`slot_results`. The exported `crown_slots_utc` calendar lets the collector advance its next-slot display even
when a slot has no winner and accepted state stays unchanged. A slot can be consumed only once.

`finalize_engine_slot(state, eligible, params, now, block)` mutates only a successful crown. Its block argument is
the current chain snapshot block; decay starts at that actual finalisation block, including recovery of a missed
slot. Its history/slot-result event retains the originally scheduled `slot_utc`. At worker startup finalisation
runs before GPU preflight; it needs only accepted measurements and the current eligible commitments.

## Sandy batch and no-GPU wrapper

1. Back up the deployed `engbox/hardening/candidate-batch.sh`. From that directory, run
   `patch --dry-run -p1 < <this-worktree>/tools/engslots/candidate-batch.patch`, then apply it after review.
   The patch base SHA256 is recorded in `tests/fixtures/README.md`.
2. Install `<this-worktree>/tools/engslot-finalize.sh` as `engbox/hardening/engslot-finalize.sh` (mode 0755).
3. Set **both** the daily batch's `CHAIN_SRC` and the wrapper's `ENGSLOT_SRC` to the reviewed release checkout's
   `src` directory. The old default `workers/engopen/wt/src` may contain pre-slot code and must not be used.
   Keep `CHAIN_PY` pointing at the release virtualenv interpreter. The check imports fail closed with stale code.
4. Install `engslot-retry.sh` beside the wrapper (0755). Review/install `sn92fl-engslot.service` and
   `sn92fl-engslot.timer` from this directory into `~/.config/systemd/user/`. Create
   `engbox/hardening/engslot-release.conf` with absolute `ENGSLOT_SRC`, `CHAIN_SRC` (same release `src`) and
   `CHAIN_PY` paths. These values must also be in the daily/auto batch service environment. The service sets
   `DRY_RUN=0`; direct wrapper invocation still defaults to a plan. Run `systemctl --user daemon-reload`,
   then `systemctl --user enable --now sn92fl-engslot.timer` only after the release owner's notice/review gate.
   The timer is `Tue,Fri *-*-* 14:30:00 UTC`, `Persistent=true`. It invokes the runner, which derives the latest
   due slot from release code and retries failures every 300 seconds until slot+4h. A stale persistent firing
   writes a missed receipt immediately; no retry continues past its deadline. Missed slots recover next worker cycle.
   Scheduling/announcements and the required notice period belong to the lead; this worker has installed nothing.
5. Back up `bin/engbatch-auto.sh`, then review/dry-run/apply `engbatch-auto.patch` there with `patch -p1`.
   It uses `$CHAIN_PY -I` plus `$CHAIN_SRC` to read `ENGINE_CROWN_SLOTS_UTC`, never a weekday table. Every mode
   defers starts from slot-75min until a matching successful publication/defer receipt exists or slot+4h has passed.
   The normal daily 12:00Z start is 150 minutes before a slot and proceeds. If still live at 14:30Z, the wrapper
   defers instead of changing its seed. The existing auto launch lock/caps/review gates remain in force.
6. Review the receipt and retained chain snapshot in `engbox/hardening/batches/engslot-*/`, and the history pair in
   `engbox/hardening/state/history/`. The wrapper shares `.engine-state.lock` with batch collect/recheck, runs the
   existing plausibility/review checks plus an exact deterministic replay, then uses the existing `publish_state`.
   It retains exit code, changed/published flags and state/snapshot hashes even on failure. A no-op can republish
   accepted state, allowing a failed publication to be retried safely.

The wrapper waits up to 300 seconds for `.engine-state.lock`; timeout writes `deferred: state publication locked`
and exits 75. Under that lock it checks all `batches/*/box_id`: a box without `deleted_at` (confirmed teardown)
or `ledger-release.txt` (guard release receipt) causes `deferred: batch <id> in flight`, receipt, exit 75.
`delete.json` alone is only the delete request and does not prove teardown. Both success and defer receipts carry
`slot_utc` from the retry runner, so the auto gate releases only the corresponding slot. A give-up receipt is not
mistaken for a successful/deferred attempt. No accepted-state write or publication happens during a defer.

`check_state slot` accepts only the exact pending-to-champion output from the configured slot instant through
slot+4h, matching the bounded retry window (no early publication). The CLI itself can recover a later missed slot,
but the wrapper deliberately refuses to publish a new crown outside that window. The worker recovery path handles it in the next batch. Normal batch mode
rejects an unscheduled crown; a delayed scheduled crown must replay the seed's already-confirmed pending entry,
with the same event and slot-result receipt. A later remeasurement may lower that champion's held score.

No GPU invocation, spending approval or credential is needed by the wrapper.

## Offline CLI

With a wrapper snapshot (same chain rows as the existing engreview snapshot plus the head block):

```
python -m reax_subnet.engslot finalize --state accepted.json --commitments chain-slot.json \
  --now 2026-10-27T14:30:00Z --out final.json
```

The envelope is `{"block": HEAD, "rows": [{"uid": 7, "hotkey": "...", "block": COMMIT_BLOCK,
"text": "reax-eng/1:..."}]}`. The original `workers/engreview-open17/scripts/chain_snapshot.py` produces a **list**
without the current head block. That unchanged format is supported with `--block HEAD`. The block cannot be
inferred from commitment blocks without changing decay; the CLI therefore refuses to invent it. No network call
is made by the CLI. The Sandy snapshot helper is extended to retain both its old `commits` map and these rows/head.

## Collector and site patches

- From `/home/flori/jobs/reax-miner-dashboard-20261015`, review/apply `tools/engslots/collector.patch` with `patch -p1`.
  It reads the validator's engine state through the existing read-only fetch and carries the s1-fast summary into
  `/data/miners/competitions.json` as `engine.next_slot_utc` and `engine.pending`. Only uid, digest12, pooled score
  and slot leave the collector. Run `python3 -I collector/test_collect.py`.
- Apply `tools/engslots/site.patch` to a fresh reax-site checkout at main `b079c1559a564f8f9c8893be0ba3fc33af98c011`
  or rebase it as needed. The job's site checkout was `57a37c1`; its `origin/main` ref was `abe3ed6`, both stale.
  The actual remote main was fetched into a temporary job-owned bare repository to prepare the patch. It adds one
  UTC line to the engine competition card: `Next engine crowning slot: Tue 27 Oct 14:30 UTC`.

The collector patch is based on the read-only source present during this job. Re-run patch dry-runs against live
files before integration; other release work may have changed them.

## rc11.post6 critic fixes: installation and compliance

Ship the updated patch and source together. `check_state` now uses **`"$CHAIN_PY" -I`** and
explicit `CHAIN_SRC`; both wrappers run an import smoke and retain failure receipts. Run the shell tests with
the release virtualenv interpreter. Batch time comes from Sandy's wall clock, never the box's `saved_at`.
New confirmations older than Sandy batch `created_at - 600s` are rejected; established seed confirmations remain valid.
The box cannot change `weights_ok` (including silently losing it on old images), rewrite slot results, or swap
confirmed challenger identity/measurements. Additional measurements must match this batch's results/measurement
receipts; confirmation is replayed. These receipts do not make untrusted benchmark results trusted; existing
review, chain, context and measurement checks still apply.

Install `tools/engweights-ok.sh` beside `candidate-batch.sh`, mode 0755, before the Fri 30 October slot.
It defaults to DRY_RUN. For an explicit digest review set `ENGSLOT_SRC`, `DRY_RUN=0`, `WEIGHTS_ACTION=add`,
`WEIGHTS_DIGEST=<64hex>` and `WEIGHTS_INVENTORY=<complete non-reference layer inventory>`. The operator must verify
that this inventory covers ALL non-reference layers and that the derivation/calibration review is complete;
a filename screen alone is insufficient. The CLI requires `state/review/REVIEWED/<digest>` plus a clear screen.
It retains inventory bytes, a compliance JSONL line (who/UTC/inventory SHA256), seed/final history, a weights-only
acceptance check and publication receipt, under the same state lock and `publish_state` path as finalization.
Use `WEIGHTS_ACTION=remove` for revocation. Approval changes do not affect the existing champion.
If publication fails after local acceptance, rerun the wrapper to republish the approved set; previous history
and the failed publication receipt remain. The standalone CLI changes only `--out` and prints the compliance log:

```
python -m reax_subnet.engslot weights-ok add --state accepted.json --digest <64hex> \
  --review-dir state/review --inventory non-reference-layer-files.json --out approved.json
python -m reax_subnet.engslot weights-ok list --state approved.json
```

## Safe worker rollback

Never reuse a post6 confirmed pending challenger with post4/post5 worker code: it crowns immediately off-slot.
Install/review `tools/engslots/rollback.sh`, then set `CANDIDATE`, `CANDIDATE_BACKUP` (the pre-post6 script including
its check_state), `PREVIOUS_WORKER_TAR`, `PREVIOUS_WORKER_IMAGE`, `PREVIOUS_WORKER_ID` and release `CHAIN_PY`.
Dry-run is default. With `DRY_RUN=0`, under the same lock it backs up state/script, strips **every** challenger
carrying `confirmed_at` plus its public pending summary, records `pending_cleared_rollback`, and restores the
old check_state/script with previous worker tar/tag/ID pinned. It sanitizes state before enabling the old worker.
It calls the installed candidate's existing `publish_state` while holding the lock, **before** replacing the
script/defaults. Publication failure leaves the stripped local seed and current post6 script, and exits nonzero;
resolve publication before resuming. DRY_RUN's last line prints the exact source-and-publish command without executing it.
The owner must stop any in-flight post6 batch before rollback. No worker process is started. Alternatively
restore the pre-post6 accepted-state backup, taking care not to discard newer compliance decisions unnoticed.
Clearing pending costs re-confirmation. Emptying the slot calendar alone restores immediate crowns but does
**not** disable the independent weights policy; reverting that policy requires a separate owner decision.

## Keep-alive decision required (unchanged)

`competition_split.ENGINE_KA_MAX_HOURS = 12.0`; `engcomp.waiting_uids` measures from the challenger's first
measurement (or, if later, near the end of registration immunity). It does not reset at confirmation.
It cannot cover Tue 12:00→Fri 14:30 (74.5 h) or Fri 12:00→Tue 14:30 (98.5 h). Proposed owner change:
cover at least 98.5 h **plus the confirmation duration**, or give confirmed pending challengers a separate
bounded window through their scheduled slot and missed-slot recovery. Merely setting 98.5 h from the first
measurement is insufficient when confirmation is delayed. No keep-alive constant was changed here.
