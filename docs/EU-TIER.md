# EU tier: Scheme A (x1.5) and the EU contract tier (staged quality update, inactive)

Decisions: Scheme A (card 18283, 23 Oct 2026) and the EU contract tier, 16 RTX PRO 6000 slots at EUR 1.00 per verified
GPU-hour (card 18296, 23 Oct 2026). Code: `src/reax_subnet/eutier.py`, hooked into `S1Validator.compute_weights` and
`run_round`. **The shipped registry lists no operator: nothing changes** (tests: weights byte-identical to live 04eefe3).

## Registry (`src/reax_subnet/eu_registry.json`, override `REAX_EU_REGISTRY_FILE`)

Versioned JSON, reloaded every round, strictly schema-checked. An invalid file (bad JSON, unknown key, a hotkey under two
operators, a cap above 50 %, ...) is logged at ERROR and treated as the empty registry; weight setting never stops.

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

Contract operators need no EU probe job or `VerificationSource` row. Florian's manual review and the signed
contract replace RTT/ASN probes (4 Nov 2026 decision). Signed contract id, DPA id, datacenter and ASN remain required
in the registry; missing or placeholder fields still make the operator inactive.

**Scheme A is unchanged:** `{"version": 1, "hotkeys": {hotkey: {"rtt_ok": bool, "asn_ok": bool,
"checked_at": iso8601}}}` is written by the EU probe job (multi-site RTT Helsinki/Katowice/Frankfurt as anomaly evidence
+ ASN/contract match). Both checks must be true and younger than 6 h. No file means no Scheme A premium. The contract
recording hook no longer consults this source; Scheme A retains its separate multiplier/verification path.

## Scheme A

In s1-fast and s1-pro a verified hotkey's in-pool score (`ema x capacity factor`) is multiplied by
`1 + 0.5 x min(1, cap_units / F)`, F = the operator's summed capacity factor (units) over its verified hotkeys in that
pool: at most `cap_units` units of an operator earn x1.5, splitting into more hotkeys gains nothing. Pool budgets
are unchanged (the premium is paid by the other miners of the pool through normalisation).

## Contract tier (pay rule 1-6)

1. Per hotkey and UTC hour the validator counts informative rounds (`q`), clean rounds (`k`),
   any strike (`f`, including capacity probes), answered zero-score rounds (`w`), capacity per declared GPU
   at the last counted round (`c`), the declaration (`g`), and active contract admission (`v`).
   - At least **10 counted rounds** and **90% clean rounds**, with admission throughout.
     Clean means answered, no strike and score greater than zero. Latency reductions of a positive score
     do not reduce the clean share. CPU/lite scores do not qualify GPU contracts. No EMA gate remains.
   - A validator reference failure is uncounted. A non-serving, unregistered or not-queried contract
     hotkey counts as unanswered. An informative GPU outcome or capacity strike takes precedence.
   - At least **25 decisions/s per declared GPU at the last counted round**. Use only the **s1-pro**
     pool (RTX PRO 6000 contract slots): the best of its last two folded probes, each at most **3 hours**
     old and matching the declared GPU count. Persist both reports across restart. A probe arriving
     later in the hour can qualify it; no fresh report at the last counted round fails the hour.
     Other pools cannot qualify capacity. One low probe is tolerated until both retained probes are low.
   - A strike always voids the hour. Three correctness-failed hours in the trailing 24 h suspend the hotkey until
     an hour satisfying all gates; that recovery hour is unpaid. Correctness-failed means a strike, or at least
     two wrong rounds and more than 10% wrong. This is compatible with the clean-share gate: a lone wrong round
     can withhold pay when clean share falls below 90%, without creating a suspension failure.
     `H_o = min(slots x 24, sum of passed hours x declared gpus per hotkey)`.

   Hours recorded under another GPU count cannot be multiplied by a later registry increase. Old buckets load
   safely; prior unscoped capacity histories are discarded, and their old answered count is not trusted as a clean
   count. Old cached settlements cannot start
   pay without the current `quality_version` (3). The next day's settlement applies the new rules.

   Threshold basis: the captured UID 148 folded capacity is 38.77-40.39 decisions/s; the reference is about 47/s.
   The 25/s floor is about half the reference, below UID 148 and the estimated 26.5/s with an overlapping customer
   call (that overlap estimate has not been measured). The capture's longest probe gap is 150.65 minutes, below
   the 3-hour freshness limit. UID 148 and six peers are replayed in the worker's final hour table. Capacity is a
   GPU quality condition, not a guarantee of a particular model's customer throughput.
2. `P = alpha_TAO x median(TAO_EUR)` from `REAX_EU_MARKET_FILE` (tools/eu_tier_quote.py), quotes within 2 h of 00:00Z
   of the day and at least 2 TAO/EUR sources; otherwise the previous P (logged); no P ever -> the tier pays nothing.
3. `A_o = H_o x eur_per_gpu_hour / P`.
4. `sum A_o <= cap_pct % x inference_share_alpha` of the day (from the market file; missing -> nothing paid); above
   it all A_o are cut pro rata (logged, in the receipt). The shipped registry stays at **5%**. `MAX_CAP_PCT = 50`
   permits later reviewed registry changes without another code release. At about 885 alpha/day and EUR 4.53/alpha,
   this covers 64 slots x 24 h x EUR 1 (EUR 1,536) while retaining a bound that catches a large cap typo.
   No cap is raised here.
5. The day after, each paid operator's hotkeys get `A_o / inference_share_alpha` of the miner part, split by their
   verified GPU-hours; those hotkeys leave the pools (no global inference pay on top); everyone else is scaled by
   `1 - sum`. The engine scales this vector to the inference share; competition payments, legacy keep-alive and
   owner allocations can reduce it further. The resulting contract payment is at most the owed fraction at the
   estimated share and price, rather than a guaranteed EUR floor. Miner-slot compensation applies last.
   An operator the tier does not pay (no price, no hours) keeps the normal rule.
6. Receipt per settled day: `REAX_EU_TIER_RECEIPTS_DIR/YYYY-MM-DD.json` (operator, slots, H_o, P, A_o, cap cut,
   hotkeys' GPU-hours; no contract ids, no IPs). Ledger: `REAX_EU_TIER_STATE_PATH` (default next to the validator state).

If the market entry or inference share of a day is missing while contract operators exist, the settlement is not cached
and no receipt is written: every round retries it. Still missing at 23:00Z of the next day, the day is settled as
"no price" (pays nothing), cached and logged at ERROR.

A registry revision after a day was settled pays nothing until the next settlement under the new revision.

## Keep-alive overlap (existing limitation)

`contract_hotkeys` controls ledger admission; paid operators' hotkeys leave ordinary inference pools and `carve_out`
assigns their contract fractions. None of these excludes a `ka-override.json` waiting row passed later to the competition
layer. That layer credits serving pay and tops up only below its waiting level: sufficiently large contract pay gets no
extra, but small contract pay can still receive a waiting top-up. Tests exercise both with an actual waiting override.
**Do not leave a paid contract hotkey in the waiting override.** The staged onboarding renewal job removes a roster
hotkey only after its operator has a positive fraction in the latest settled receipt. No receipt keeps the row; a
receipt read failure renews conservatively and alerts the owner. The next renewal removes it after pay starts, so an
overlap until that renewal remains possible. This change installs no configuration.
