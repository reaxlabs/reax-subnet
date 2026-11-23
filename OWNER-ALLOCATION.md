# One-off owner allocations (staged post8.9)

`owner-allocation.json` is next to `REAX_AWARD_APPROVALS`, like `ka-override.json`.
It is an array, at most six strict rows (required keys below, optional `source`):

```json
[{"hotkey":"5D5goWB72691wTobFgrT961FKJ2JVGDnNDgm9iaGrn3uifpp","share":0.06101694915254237,"sets":6,"start_block":0,"reason":"uid 114 merged-work credit: 54 alpha over 6 sets"}]
```

This example is a proposal, not activation authority. Start block 0 means the first
successful set after installation; use a reviewed future block if needed. A share
is a fraction of the MINER weight, not the inference pot. All active allocations
default to `"source":"inference"` when the field is absent. Inference rows are taken
proportionally from inference after every protected payout has been assembled.
`"source":"burn"` rows instead use only the final residual top-up to the owner UID:
inference miners, competition and keep-alive parts stay exactly as in the vector
without that burn row. The owner's own serving component is protected too.
Mixed sources (including the same recipient) are allowed; each source scales its
rows proportionally when its own budget is too small. No negative owner weight,
no borrowing from the other source. A zero-room source consumes no set; a partial
payment counts as one set, as with existing inference rows. Unknown source values
refuse the entire file and weights use the plain vector. The owner's own UID is refused as
a recipient. Apply only after NEED_START with an owner UID. A valid hotkey with no
UID consumes no set. No config means the old vectors exactly.

`sets` is an integer from 1 through 24 (previously 12); share <= 0.10 per row,
active total <= 0.25. Parser defaults do not add keys to rows, so existing rows
keep their hashes and completed rows never replay. Adding an explicit `source`
to an old row changes its identity: retain the old journal and review any new row.

Counts follow the SHA256 of each complete canonical row, independently of row order.
Do not change the reason/share/sets/start block of an allocation in progress: that
is a NEW allocation. Removing/readding an identical completed row cannot replay it
as long as progress is retained. Keep `allocation-progress.json` across restarts,
upgrades, removal of the config and rollback. Never delete it to reset an allocation.
Install configuration with atomic replacement, not in-place edits. One validator
process owns the file and journal; they must be on a durable writable volume.
Allocation counting needs a writable volume; weight setting never does.

Weight setting NEVER halts or waits because of this module. Weight computation and
dry runs do not touch progress. Immediately before an actual send the validator
fsyncs a pending journal; only then does the vector WITH the allocation go out, and
the success callback (also late success after the wait timeout) increments the
selected rows exactly once and fsyncs again.

On ANY uncertainty the validator sends, for that set, the plain vector (identical
to post8.8, computed in the same pass), does NOT count the set, and logs one ERROR
line `owner allocation NEEDS-RECONCILE: weight set at block <B> rows <ids> ...`:

- journal cannot be written before the send (read-only mount, disk error, journal
  changed since the computation): plain vector sent, nothing on chain to reconcile;
- the allocation-bearing send raised or its outcome is unknown: retried next round
  like post8.8, plain from then on;
- restart with a pending entry (crash during a send): plain vector sent;
- corrupt/unreadable journal: plain vector sent, the journal is never overwritten.

Before any burn send, the validator also verifies that only the residual owner
top-up moved. An inconsistent vector (including a changed approval/state between
computations) triggers the existing plain-vector fallback; no allocation is counted.

Burn sends additionally log `owner allocation burn vector block ...` with exact
full-precision vectors from the same pass: the inference-only allocation vector,
the mixed vector, actual burn allocation weights, and miner-part factor. The staged
money monitor requires this proof for a committed burn receipt, checks every UID
to 1e-12, and checks the raw sent vector to the log's 4-decimal precision against the
expected miner-slot-scaled vector (without amplifying rounding by de-scaling). Missing proof raises I-OA-BURN. Computation-only logs never count
as payment; committed lines include the source before the reason.

An exception while computing the allocation-bearing vector itself also sends the
plain vector for that set (one ERROR line `owner allocation: <Exception> at block
<B>; weights without allocation`); nothing is journalled or flagged, the next set
tries again.

The rows go into `needs_reconcile` of `allocation-progress.json` (pending cleared).
While `needs_reconcile` is non-empty, every set goes out without any allocation.
If the flag cannot be written (read-only or corrupt journal) it is latched in memory
until the validator restarts after the repair. A local file cannot commit atomically
with a chain transaction, so exactly-once is guaranteed only for sends whose success
was observed; ambiguous sends are left to the operator, never guessed.

Operator reconciliation: look up the validator's weight extrinsic around the named
block. If the finalized vector carried the allocation (recipient weight above the
plain vector), increment `counts[<row hash>]` by one for each flagged row; otherwise
leave counts. Then set `needs_reconcile` to `{}` atomically (write a new file and
rename). Allocation resumes with the next set; no restart needed unless the flag was
only latched in memory (then fix the mount/journal and restart). Keep the old/new
journal hashes and the receipt.

Rollback: remove/rename the operator file first, retain progress, restore post8.8
using the staged switch script. Post8.8 ignores both files; it does not decrement
or reset counts. Reinstalling post8.9 resumes only the unconsumed sets. Do not
reinstall an allocation file during rollback or before the payout decision.

At 2950 miner alpha/day, 7200 blocks/day and 360 blocks/set, one set is 147.5 alpha
and about 72 minutes. 54 alpha / 6 sets = 9 alpha/set = share
0.06101694915254237 (0.061 would deliver 53.985 alpha). A make-good of M alpha per
champion over 12 sets requires share M/1770 each. Together with the credit this
stays under the active 0.25 cap when M <= 111.5 alpha (12-set duration; first six
sets overlap). Verify both the 0.10 per-row cap and active 0.25 cap for the FINAL
amount; exact chain incentive/emission realization can differ from this estimate,
and early weight sets alter interval length. There is no fixed-alpha guarantee.


## Reviewed run-out make-good and rollout order (2 November 2026)

Staged rows (not installed until the post8.13 deploy): burn shares 0.0680 for UID 83
and UID 114 and 0.0334 for UID 240, 24 sets each. `miner_slot.apply` runs after
allocations, so one share unit pays `factor x alpha per unit sent weight`. Measured
2 November: factor 1.242431, 133.2 alpha per 360-block epoch per unit sent weight,
about 369.6 blocks per set. That gives about 277 alpha for 0.0680 and 136 for 0.0334,
below the gross targets of 279 and 137.3 (0.0788 would have paid about 321).
A set can be short or scaled; the monitor's weight x actual
block interval x alpha/block tally is an estimate, not proof of chain payment.
After the 24 sets, read back actual chain emissions. Pay the remaining gap to
279 each (and UID 240's recomputed gross) with NEW reviewed true-up rows. Never
edit installed rows, including reason, source or set count: any edit changes the
row hash and can replay payment. Keep the journal and receipts across all changes.

Step 2 and owner allocations must never be active together. Remove the allocation
file after the last set, even when all rows have completed; keep the journal.
Only then enable step 2. Do not bypass its compatibility check. On rollback,
remove the allocation file FIRST, keep the journal, then restore the old code.

Approval renewal restarts the approval-based clock for crowns at/after the cutoff:
the first currently valid approval wins. Do not re-sign approvals of a champion
already in run-out; review the payout consequences before any necessary renewal.
