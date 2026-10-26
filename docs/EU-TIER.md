# EU tier: Scheme A (x1.5) and the EU contract tier (1.2.0rc11.post5, inactive)

Decisions: Scheme A (card 18283, 23 Oct 2026) and the EU contract tier, 16 RTX PRO 6000 slots at EUR 1.00 per verified
GPU-hour (card 18296, 23 Oct 2026). Code: `src/reax_subnet/eutier.py`, hooked into `S1Validator.compute_weights` and
`run_round`. **The shipped registry lists no operator: nothing changes** (tests: weights byte-identical to rc11.post4).

## Registry (`src/reax_subnet/eu_registry.json`, override `REAX_EU_REGISTRY_FILE`)

Versioned JSON, reloaded every round, strictly schema-checked. An invalid file (bad JSON, unknown key, a hotkey under two
operators, a cap above 5 %, ...) is logged at ERROR and treated as the empty registry; weight setting never stops.

```json
{"version": 1, "revision": "2026-11-02-tng",
 "scheme_a": {"operators": [
   {"id": "teemo", "contract": "<id>", "dpa": "<id>", "datacenter": "<name, city, EU>", "asn": "AS<n>",
    "cap_units": 32, "hotkeys": ["5F..."]}]},
 "eu_contract_tier": {"version": 1, "eur_per_gpu_hour": 1.00, "gpu_class": "rtx-pro-6000-96gb",
   "cap_pct_of_inference_share": 5.0, "review_due": "2026-11-22",
   "operators": [{"id": "tng", "slots": 8, "contract": "<id>", "dpa": "<id>", "datacenter": "<...>", "asn": "AS<n>",
                  "hotkeys": ["5G...", {"hotkey": "5H...", "gpus": 2}]}]}}
```

An operator is **active** only with a signed contract id, a DPA id, a datacenter, an ASN and hotkeys (placeholders
`<...>` count as missing), plus `slots > 0` (contract tier) or `cap_units > 0` (Scheme A). A hotkey belongs to at most one
operator, so contract-tier hotkeys never get Scheme A.

## EU verification (`REAX_EU_VERIFICATION_FILE`)

`{"version": 1, "hotkeys": {hotkey: {"rtt_ok": bool, "asn_ok": bool, "checked_at": iso8601}}}`, written by the EU
probe job (multi-site RTT Helsinki/Katowice/Frankfurt as anomaly evidence + ASN/contract match). A hotkey is verified
only with both checks true and a check younger than 6 h. No file = nobody verified. **The probe job itself does not
exist yet**: until it writes this file, Scheme A pays x1.0 and the contract tier counts no hours (fail closed).

## Scheme A

In s1-fast and s1-pro a verified hotkey's in-pool score (`ema x capacity factor`) is multiplied by
`1 + 0.5 x min(1, cap_units / F)`, F = the operator's summed capacity factor (units) over its verified hotkeys in that
pool: at most `cap_units` units of an operator earn x1.5, splitting into more hotkeys gains nothing. Pool budgets
are unchanged (the premium is paid by the other miners of the pool through normalisation).

## Contract tier (pay rule 1-6)

1. Per hotkey and UTC hour the validator records: rounds queried, rounds answered without strike, any strike (round or
   capacity probe), EU verification at every round. An hour passes if the hotkey answered at least 90 % of the rounds
   it was queried in that hour (0 queried = not passed), no strike, verified throughout. A strike voids the hour;
   3 struck hours within 24 h suspend the hotkey until a clean hour (that hour itself is unpaid).
   `H_o = min(slots x 24, sum of passed hours x gpus per hotkey)`. The declared `gpus` per hotkey are not cross-checked
   against measured capacity by the validator: they are checked at contract review.
2. `P = alpha_TAO x median(TAO_EUR)` from `REAX_EU_MARKET_FILE` (tools/eu_tier_quote.py), quotes within 2 h of 00:00Z
   of the day and at least 2 TAO/EUR sources; otherwise the previous P (logged); no P ever -> the tier pays nothing.
3. `A_o = H_o x eur_per_gpu_hour / P`.
4. `sum A_o <= cap_pct % x inference_share_alpha` of the day (from the market file; missing -> nothing paid); above
   it all A_o are cut pro rata (logged, in the receipt).
5. The day after, each paid operator's hotkeys get `A_o / inference_share_alpha` of the miner part, split by their
   verified GPU-hours; those hotkeys leave the pools (no global inference pay on top); everyone else is scaled by
   `1 - sum`. The competition layer scales the whole miner part, so the fraction stays a fraction of the inference
   share. An operator the tier does not pay (no price, no hours) keeps the normal rule.
6. Receipt per settled day: `REAX_EU_TIER_RECEIPTS_DIR/YYYY-MM-DD.json` (operator, slots, H_o, P, A_o, cap cut,
   hotkeys' GPU-hours; no contract ids, no IPs). Ledger: `REAX_EU_TIER_STATE_PATH` (default next to the validator state).

If the market entry or inference share of a day is missing while contract operators exist, the settlement is not cached
and no receipt is written: every round retries it. Still missing at 23:00Z of the next day, the day is settled as
"no price" (pays nothing), cached and logged at ERROR.

A registry revision after a day was settled pays nothing until the next settlement under the new revision.
