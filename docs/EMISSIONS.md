# Emissions, owner share and burn

## What the chain does (dynamic TAO)

Each subnet's emission is split by the chain, not by this code: the **subnet owner cut** (18 % by default; check the live chain) goes
to the owner, and the rest is shared between miners (by incentive, from validator weights) and validators
(by dividends, from stake and bonds). This code only decides the **miner weights** each validator submits.

## Roles

* **Subnet owner** = the account that registered the subnet and holds the owner coldkey. The chain pays the owner cut to that coldkey.
* **Operator** = the REAX maintainers, who develop and maintain this code and may run a reference
  validator. Nothing in this repository moves TAO or pays anyone.

The owner-cut percentage is a chain parameter and can change; verify its current value on the live chain.

## The one knob in code: `REAX_BURN_FRACTION`

Validators can give a fixed share of every weight vector to the owner UID (the UID of the subnet owner hotkey).
Weight on the owner UID earns no miner reward for anyone except the owner hotkey; whether the chain pays it to
the owner or recycles it depends on the runtime version, so check on testnet before relying on either effect.

* Default `0.0`: all miner weight goes to miners that pass the fidelity check.
* When **no** miner qualifies (start-up, outage), validators put all weight on the owner UID instead of setting
  nothing, so they keep their validator trust. This is automatic and independent of the setting.
* Any non-zero value must be agreed among validators and set identically on every validator; otherwise
  validators disagree and lose trust.
* `REAX_BURN_FRACTION=1` is also the emergency brake: it stops all miner rewards once validators holding most of
  the stake have switched and submitted new weights (consensus is stake-weighted).
* With a partial burn (0 < value < 1) and no owner UID on the subnet, the burn has no recipient and is ignored:
  miners are paid in full. The owner hotkey is registered when the subnet is created, so this only matters if
  that UID was deregistered.

## Per coldkey inside the inference pools (from `COLDKEY_DR_START_UTC`, date set in the release notes)

The pool shares (80 / 15 / 5 per tier) do not change. In the small pools (s1-lite and s1-cpu; `COLDKEY_DR_POOLS`), the miner part of one coldkey has diminishing returns:
its first two hotkeys count in full, every further hotkey 0.7 times the one before (3rd 70 %, 4th 49 %, 5th 34 %), so any
number of hotkeys on one coldkey counts at most 4.33 full hotkeys. s1-fast and s1-pro are not changed (pay there stays linear
in verified throughput). The weight this frees stays in the same pool and goes to
the other miners of the pool pro rata, never to the owner or the burn. The lite brake (tier-order ratio 0.5) is replaced by the same
ratio at 1.0 applied to these effective miner counts. Honest operators with one or two hotkeys per coldkey are not reduced.
Numbers and variants: docs/DESIGN-POOLS.md section 4.6.
