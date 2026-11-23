# Miner slot factor (post8.10)

When the subnet owner moves part of the MINER emission slot to the owner slot (`AdminUtils.sudo_set_subnet_emission_weights`,
lock instead of burn; Florian #35779), every miner would get proportionally less alpha. UID 3 is the only weight setter,
so the validator compensates: every weight except the owner UID's is multiplied by

    factor = old miner share / new miner share      (41 % -> 33 %: 0.41 / 0.33 = 1.2424...)

and the owner UID (the burn) takes the rest. Every miner, competition payee, keep-alive and owner-allocation recipient
keeps the same absolute alpha; only the burn shrinks (UID 3 ~0.37 -> ~0.22 of the vector). The old split is pinned in
`miner_slot.BASE_EMISSION_WEIGHTS` (read at block 9399298, owner 18 % / validator 41 % / miner 41 %).

The scaling is the last step before sending. If the scaled non-owner weights would exceed the total, they are scaled to
exactly the total and the owner UID gets 0 (warning `CLAMPED`; miners then get less than their old absolute alpha).

## Operator file

`miner-slot.json` next to `REAX_AWARD_APPROVALS` (on the validator host: `/w/approvals/miner-slot.json`, in the container
`/approvals/miner-slot.json`):

```json
{"factor": 1.2424242424242424, "from_utc": "2026-10-30T23:00:00+00:00", "reason": "SN92 miner slot 33 % (was 41 %)"}
```

Exactly these keys; factor a number in [1.0, 1.5]; `from_utc` an ISO time with a timezone; reason 1..300 characters;
duplicate keys refused; at most 4096 bytes. Absent file, before `from_utc`, a refused file or any error: factor 1.0 and
the vector is sent unchanged (warning in the log; the money monitor reports `I-MS0` for a refused file). The file is read
at every weight set; no restart is needed. Write it atomically (temp file + rename).

## Steps

1. After the owner's split change is on chain, read it (read-only):
   `pod.sh "docker exec nodes-validator-1 python -m reax_subnet.miner_slot"`.
   It prints the split, the factor and the exact JSON line. Exit 2 = factor outside [1.0, 1.5]: do not write the file.
2. Write that line to `/w/approvals/miner-slot.json.tmp`, then `mv` it to `/w/approvals/miner-slot.json`.
3. Check: `pod.sh "docker exec nodes-validator-1 python -m reax_subnet.miner_slot --check /approvals/miner-slot.json"`
   prints `OK matches chain`.
4. Next weight set: the log has `miner slot factor F effective F at block B: owner uid 3 x -> y` followed by
   `weights set at block B`; UID 3 is lower by about `(F - 1) x (1 - x)`, every other weight is x F. The money monitor
   de-scales the set before its invariants and reports `I-MS` if the logged factor differs from the file.

Undo: move the file away (factor 1.0 from the next set). Rollback to post8.9 ignores the file.
