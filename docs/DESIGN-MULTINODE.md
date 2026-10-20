# Several machines behind one inference hotkey (multinode)

Status: implemented behind `REAX_FEATURE_MULTINODE` (default off), miner-only. Validators, scoring, weights and the
`WEIGHTS_VERSION_KEY` do not change. Pools: s1-fast and s1-pro only.

## 1. Why

Miners asked to run several GPU machines behind one hotkey. A miner on #92 lost a UID to deregistration while moving
machines: with one machine per hotkey, a move means the UID answers nothing for a while, its moving average falls,
and it becomes the lowest-ranked UID when the next registration comes in.

Since 1.2.0rc8 the miner can already send decisions to several runtime sidecars (`REAX_<MODEL>_RUNTIME_URLS`:
least-in-flight dispatch, removal after two failures, re-admission after health and identity checks). That list is
enough for one machine with several GPUs, or for machines in one private network. Three things are missing for the
cases miners describe:

| Gap in 1.2.0rc8 | Effect | Multinode |
|---|---|---|
| The sidecar token travels in clear text; across the internet only `REAX_ALLOW_PUBLIC_RUNTIME=1` helps, and the sidecar has no TLS | Machines in different data centres cannot be linked safely | `reax-runtime-relay` on every worker: per-request HMAC signature, TLS with a pinned certificate or a VPN |
| The backend list is an environment variable | Adding or swapping a machine needs a miner restart (start-up checks, axon/commitment checks) | `REAX_WORKERS_FILE` is re-read every 10 s; machines join and leave while the miner keeps answering |
| Every listed backend must pass at start | One dead machine stops the whole hotkey from starting | The miner starts when `REAX_WORKERS_MIN_HEALTHY` workers pass; the rest join when they pass |

## 2. Shape

```
validators ──signed reax-s1/1──▶ GATEWAY (reax-miner, holds the hotkey, signs every answer; CPU only)
                                   │  per decision: the worker with the fewest decisions in flight
                                   │  link: HMAC per request + TLS (pinned fingerprint) or WireGuard
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
                 WORKER A      WORKER B      WORKER C      each: reax-runtime-relay ─loopback─▶ pinned sidecar (GPU)
```

* **Gateway.** The normal `reax-miner` with `REAX_FEATURE_MULTINODE=1`. It holds the hotkey, publishes the axon
  and commitment, checks validator signatures and signs every response, exactly as today. It needs no GPU, so it
  can sit on a small, stable server that never moves.
* **Worker.** A GPU machine with the unchanged pinned sidecar on loopback (as for a single-machine miner) plus
  `reax-runtime-relay`. The relay holds the sidecar token and one link key; it never sees the hotkey, so a worker
  (also a rented one) cannot sign as the miner or take the UID.

## 3. Link between gateway and worker

* **Authentication: HMAC per request.** Header `X-Reax-Link: <unix ms>.<nonce>.<sha256 of the body>.<hmac>`,
  HMAC-SHA256 with the worker's link key (32 random bytes, `openssl rand -hex 32`, one per worker) over the method,
  path, time, nonce and body hash. The key never travels. Because the body hash is inside the signed header, the
  relay checks the signature before it reads any body, then checks the body against the hash. The relay refuses signatures older than 10 s or more than 2 s in the
  future (the same NTP requirement as the validator exchange) and replayed nonces (30 s window). `/health` is
  signed too, so an unauthenticated scanner learns nothing.
* **Transport: one of two.**
  * TLS on any network: the relay serves a self-signed certificate; the gateway pins its SHA-256 fingerprint (printed
    by the relay at start). No CA is involved; another certificate (a man in the middle) is refused.
  * Plain http inside a private network or VPN (WireGuard, or a CGNAT-range VPN such as Tailscale): only private
    addresses (10/8, 172.16/12, 192.168/16, 100.64/10, fc00::/7). The relay refuses to listen on a public or
    wildcard address without TLS. A provider LAN in these ranges is often unencrypted; the reply signature below
    keeps answers intact there, but decisions are readable on it, so prefer WireGuard or TLS.
* **Integrity of answers.** Every answer of the relay to an authenticated request carries `X-Reax-Link-Reply`:
  HMAC over the request's link signature, the HTTP status and the SHA-256 of the body. The gateway refuses an
  answer without it or with a wrong one (before the hotkey signs anything) and fails the decision over to another
  worker. This holds on any transport, also plain http on an unencrypted LAN.
* **Model check.** At every health check the relay reports the model and `manifest_id` of its installed release
  (signed); a relay of another model or release is not used. This checks the relay package, not the sidecar's
  weights; the identity check at admission and the validators' scoring cover the weights.
* **Relay limits.** A request without a valid signature is refused before any body byte is read, so it never takes
  one of the body slots (at most 32 unfinished bodies per address and 256 in total, 256 KiB, 5 s), and outsiders
  cannot fill them. Every worker has its own key (the gateway refuses two workers with one key).

## 4. Health, removal and swaps

| Event | What happens |
|---|---|
| Start | Every worker gets an authenticated test decision and the identity check (24 published reference decisions within the release tolerances). Passing workers are in rotation. The miner starts if at least `REAX_WORKERS_MIN_HEALTHY` (default 1) pass, otherwise it exits with the names of the failing workers. |
| A worker fails (2 faults in a row: connection error, timeout, HTTP 5xx, a refused link signature, an unsigned or changed answer; or 2 failed `/health` checks) | Out of rotation. A decision that failed before any answer byte is retried once on another worker (as in rc8). The relay's `/health` also turns 503 for 30 s after its sidecar refused the relay's token. |
| It recovers | Back after 2 good health answers in a row (5 s apart) **and** a fresh identity check; a request that was in flight when it was removed and succeeds late does not bring it back. |
| Worker added to the file | Joins **out** of rotation; validator traffic only after 2 good health answers and the identity check (about 10-15 s). |
| Worker removed from the file | Make before break: it keeps serving while fewer than `REAX_WORKERS_MIN_HEALTHY` of the remaining entries are in rotation (at most as many as remain, so scaling down always works); then it leaves the rotation and its decisions in flight finish (drain, at most 60 s). A dead worker leaves at once. |
| URL, key, fingerprint or `max_inflight` of an entry changed, or its key file rewritten in place | Remove + add under the same name: the old entry stays until the new one is in rotation, also through a transient fault of its own (it is still health-checked). A wrong edit (typo in the URL, a key the relay does not have) therefore never takes the worker out; fixing the file drops the failed replacement. |
| Entry `{"name": "<worker>", "evict": true}` | The worker leaves the rotation at once, without make-before-break (a compromised or misbehaving machine). |
| The file is broken (JSON error, empty list, bad entry, two workers with one key) | Nothing changes; one error line in the log. |
| Every worker is out | The miner answers 503 at once (unchanged from rc8). |

**Machine swap without dropping the UID (the #92 case).** Keep the old machine running, start the new one, add it
to the file, wait for `runtime backend <url> is back in rotation` in the log, remove the old one, switch it off. The UID answers
throughout. Tested in `tests/test_multinode.py::test_machine_swap_under_load_loses_no_request` (4 concurrent
validator streams, add C, remove A, switch A off: 0 failed requests) and in the localnet run in section 9.

**Rotating a key** on a worker restarts its relay with the new key, so the old entry stops working at that moment and
the new one needs one admission (10-15 s). With one worker that is a short gap; with two or more, rotate one worker
at a time.

**Moving the gateway itself** is a different case: it holds the axon IP. Start the new gateway with the same hotkey
on the new IP; it publishes the new axon on start; stop the old one once validators query the new IP (one metagraph
refresh, a few minutes). Rounds in between can be lost, so move the gateway rarely; it is small and needs no GPU,
which is why it should be a stable server.

## 5. Load spreading and limits

* Per decision (not per request): every one of a request's 8 (round) or 16 (s1-fast probe) decisions goes to the
  worker with the fewest decisions in flight, never above that worker's `max_inflight` unless all are full. A
  request is spread over several machines, so its answer time falls as workers are added.
* Limits default to the same values as N runtime URLs in rc8, with N = workers in the file at start:
  `REAX_MAX_CONCURRENT` (requests passed to the workers at once, all validators together) and the per-validator
  `REAX_PER_CALLER_CONCURRENCY` / `REAX_PER_CALLER_PER_MINUTE`. The two per-validator limits shrink while workers
  are out (in rotation / N), so a capacity probe measures what really runs; a worker that is joining during a swap
  does not shrink them, and more workers than at start never raise any limit above its configured value (set the
  variables or restart the gateway after growing the fleet).

## 6. Latency budget

The extra cost of multinode is one gateway-to-worker round trip per request (the decisions of a request go out in
parallel over kept-alive connections; TLS adds no handshake per request).

* **Round score.** The speed term is 20 % of the round score and falls linearly from the fast bound to the slow
  bound (s1-fast 0.5 s → 8 s, s1-pro 1 s → 12 s). An answer inside the fast bound loses nothing. Beyond it, an extra
  round trip of r costs `0.2 × r / 7.5` (s1-fast) or `0.2 × r / 11` (s1-pro) of the round score:
  r = 20 ms → 0.05 % / 0.04 %; 50 ms → 0.13 % / 0.09 %; 150 ms → 0.40 % / 0.27 %.
* **Capacity probes matter more.** A probe offers a fixed number of request lanes per UID (sized for an answer time
  of about 0.4 s for s1-fast and 0.8 s for s1-pro), so a hotkey that is lane-bound shows about
  `t / (t + r)` of its throughput: r = 20 ms → 95 % (s1-fast) / 98 % (s1-pro); 50 ms → 89 % / 94 %; 150 ms → 73 % / 84 %.
  This applies only near the lane limit; below it the extra lanes cover the round trip.
* **Recommendation:** workers within about 20-30 ms of the gateway (same region, ideally the same provider or city).
  Distance between the gateway and validators is the same for every miner and is not part of this budget.

## 7. Caps and pools

* **Caps unchanged by this feature.** Pay inside a pool stays `ema × min(cap, capacity / target)`; a multinode
  hotkey is paid like any other hotkey with the same verified throughput.
* **Only s1-fast and s1-pro.** s1-lite (32/s) and s1-cpu (4/s) caps are low on purpose (DESIGN-POOLS.md 4.3); the
  miner refuses `REAX_FEATURE_MULTINODE=1` with those models, and the relay refuses to serve them.

### 7.1 Should the caps rise with multinode? (input from #92, 19 Oct)

Measured per-GPU throughput (CALIBRATION.md, deterministic reference builds, concurrency 64):

| Pool | Per GPU | One 8-GPU box | Cap today (x2) | Cap at x4 | GPUs at the cap today |
|---|---|---|---|---|---|
| s1-fast | 55-74/s (reference path) | ~450-600/s | 1,280/s | 2,560/s | ~17-23 |
| s1-pro | 93/s RTX PRO 6000, 108/s H100 | ~740-860/s | 320/s | 640/s | ~3-3.5 |

* **s1-fast:** the x2 cap is already above two full 8-GPU boxes. No change needed.
* **s1-pro:** 320/s is about 3.5 GPUs, so an 8-GPU machine is split into 2-3 hotkeys today, and live data (17 Oct)
  showed many s1-pro hotkeys sitting exactly at the old cap. The x4 step already planned in DESIGN-POOLS.md (640/s)
  covers about 7 GPUs: most of one 8-GPU box behind one hotkey. Going beyond x4 for s1-pro (e.g. 1,280/s) would need
  more than the 64 lanes a probe can offer today (64 lanes × 8 decisions / 0.8 s = 640/s visible), so it is a probe
  change, not just a constant.
* **Recommendation:** raise s1-pro to x4 (640/s) as its own validator release after the live probe bench that
  DESIGN-POOLS.md already requires, ideally in the same week as multinode; keep s1-fast at x2. That release is a
  consensus change (its own `WEIGHTS_VERSION_KEY` bump and #92 notice), separate from this miner-only feature.
* **Registration fees and UID pressure.** Pay is linear in verified throughput, so a higher cap is sybil-neutral: it
  does not change what a machine earns, only how many UIDs it needs. Fewer split hotkeys free UIDs (256 slots,
  many held as same-machine siblings today), so registration burn falls for new miners and fewer honest UIDs are
  deregistered. The registration cost is recycled by the chain, not paid to the subnet owner, so the owner has no
  revenue effect either way.

## 8. Abuse analysis

* **Copying and forwarding (DESIGN-POOLS.md 4.2) stay unprofitable.** Items are fresh and verification is secret.
  Workers speak only the internal decision contract; they cannot answer validators. A gateway that forwards to
  another miner's machines shares that capacity in the simultaneous probe exactly as before (4.1), and the
  responses still have to be signed by the gateway's own hotkey.
* **Concentration is bounded by the per-hotkey cap.** Multinode does not raise what one hotkey can earn: the cap
  is unchanged. One operator could already run many hotkeys; multinode only means fewer registrations for the
  same hardware. Pool budgets, tier budgets and the tier-order rule are unchanged.
* **A bad worker.** A worker that answers with other weights is refused at admission by the identity check (a smoke
  test on published items, not a proof). The validators' scoring stays the enforcement: a bad worker that slips
  through gets the hotkey struck. Operators must run the pinned runtime on every worker and run `reax-selfcheck`
  on each worker machine.
* **A rented worker.** Never sees the hotkey; at worst it serves wrong answers (strikes, see above) or goes offline
  (removed from rotation). It cannot take the UID.
* **A stolen link key.** Lets the thief send decisions to that one worker (free inference on that GPU), or, on an
  unencrypted network, forge that worker's answers, until the key is rotated (new key on the relay and in the key
  file; handled as a swap). Keys are per worker and the gateway refuses shared keys, so one leak does not open the
  others.
* **The relay port.** In VPN mode it is not reachable from the internet. In TLS mode it is public, but every request
  without a valid signature is refused before its body is read, so it takes no body slot. What remains is plain
  connection and TLS-handshake load, as on any public port. Firewall the relay port to the gateway's IP (the
  documented default setup). A request's nonce is spent only after its body matched the signed hash, so a copy of
  a seen header with another body cannot use it up. On plain http an on-path observer can still read decisions and
  hold body slots with copies of seen headers: prefer TLS or WireGuard over an unencrypted provider LAN.
* **A rented or compromised worker redirecting the gateway.** The gateway does not follow redirects on the link.
* **Validators.** No change: no new message, no new scoring input, nothing to agree on. A multinode hotkey looks
  exactly like a single fast machine.

## 9. Tests

* `tests/test_multinode.py` (46 tests): link signatures (wrong key, changed body/path/method, stale, future,
  replay, flood), key files, workers-file validation (incl. shared keys), relay over http and TLS (pinned fingerprint
  accepted, other certificate refused, unsigned or wrongly signed requests refused, the sidecar token alone is not
  enough, stale signature refused before the body is read), answers changed on the way refused (tampering proxy),
  model/manifest check at health, sidecar token mismatch reported, partial start, admission only after health +
  identity, drain on removal, make before break (wrong key, typo URL, limit change, min healthy), in-place key
  rotation noticed, a broken or empty file keeps the workers, dead worker and wrong-key failover, no re-admission by
  a late success, forged signatures take no relay body slot, a leaving worker survives its own blip while its
  replacement is broken, scaling below `REAX_WORKERS_MIN_HEALTHY`, eviction, a file written during a reload,
  a copied header with another body cannot spend the nonce, flag-off health check unchanged, limits during swaps, config with the flag on and off.
* `tests/test_runtime_fanout.py` (rc8) unchanged and passing: with the flag off nothing changes.
* `scripts/localnet/e2e_multinode.py`: a local subtensor, two validators, one multinode hotkey with 2 workers
  (each fake sidecar + relay), a single-machine control hotkey; during a running capacity probe cycle worker A is
  replaced by worker C. Pass = the multinode UID never stops answering, keeps a non-zero capacity, and earns about
  twice the control hotkey (2 machines vs 1).

## 10. Rollout

Miner-only, flag default off; validators need nothing. Target: release after Thu 22 Oct (e.g. Mon 26 Oct with the
review), handed to the release owner. #92 notice draft: `notice/N-multinode.md` in the job folder.

## 11. Related request: inference and one competition on the same hotkey (assessment, not implemented)

Miners on #92 (19 Oct) asked to let one hotkey serve inference **and** take part in one competition at the same
time (SN44-style), so that moving from inference into a competition does not need a second registration and does
not risk deregistration while the entry is evaluated.

**Today.** A hotkey has one on-chain commitment, and its prefix decides its role: `reax-s1/1:<pool>:<manifest>`
(inference), `reax-spec/1:<track>:<repo>@<rev>` (use-case track) or `reax-eng/1:<track>:<image>@sha256:<digest>`
(engine). Switching role means a new commitment, so the inference score restarts from zero. Keep-alive pay exists
precisely to keep competition entrants registered: an entry above its floor is paid the median positive serving
weight (capped), computed without the competition-paid UIDs.

**Proposal.** One commitment with up to two elements, one per type: an inference element and at most one
competition element (the commitments pallet stores several fields per hotkey; each element in its own field, 128
bytes each). Validators parse each field by its prefix and keep the latest element per type. Changing one element
must not reset the other: a new competition entry keeps the inference moving average, and a new inference element
does not re-queue the competition entry.

**Weight interaction.**
* The two payments come from separate budgets (the inference part and the competition share), each earned by
  separate work (verified throughput; a model or engine that wins on sealed items). Adding them up is fair, and
  nothing is counted twice.
* **"Separate budgets", precisely:** every competition payment comes out of the competition share and scales the
  whole serving vector by `1 − claimed`. So the two payments add up per UID, but they are not independent pots.
* **Keep-alive becomes a top-up.** Keep-alive replaces the inference income a competition entrant gives up. A
  hotkey that still serves should not get it in full on top of its serving weight. Pay
  `max(0, per_entry − own serving weight)`, both in the units `per_entry` already uses: the fraction of the miner
  part, before competition shares are applied. Otherwise a dual-role hotkey would earn inference pay plus
  inference-sized keep-alive for one entry above its floor. Rank shares stay as they are.
* **Keep-alive slots must not be crowded out.** A track pays at most `KEEPALIVE_MAX_ENTRIES` (8) keep-alive entries,
  chosen by score before any pay is computed. Large inference UIDs could commit cheap entries above the floor that
  cost them nothing under the top-up, and take all 8 slots, so pure entrants would get nothing and be deregistered.
  Rule: only entries with a positive top-up count toward the 8, and pure entrants are ranked first.
* **The keep-alive median.** Competition-paid UIDs are already left out of the median; a dual-role UID is
  competition-paid, so it stays out. The median then rests only on pure inference UIDs, and dual-role UIDs cannot
  move it.

**Abuse and fairness.**
* No new copying path. Inference answers are still verified fresh, and competition entries still go through the
  copy checks, the near-champion rule and the per-coldkey limits.
* The real gain for a miner is one registration less and no score reset. That is the stated goal, and it eases UID
  pressure in the same way as multinode.
* The GPU that serves inference is not the one that is measured: competition entries run on the validator's or
  worker's box. So running both does not slow either measurement.
* Concentration: one UID can now hold an inference share plus a competition rank. Both are already bounded (the
  pool cap; the rank share and the per-coldkey limit), so the upper bound per UID is their sum, which one operator
  can reach today with two UIDs.

**Commitment mechanics to settle first.** Ties and copy-check priority use the commitment block. Rewriting one
field re-commits the whole commitment, which moves the block of the other element too. So validators need a
first-seen block **per element**, not only per-element change detection. Every write replaces all fields, and the
commitments pallet rate-limits writes; the tooling must always write both elements.

**Recommendation.** Do it, but as its own validator release after multinode. It is a consensus change, because
every validator must parse the elements and apply the keep-alive top-up the same way: it needs a
`WEIGHTS_VERSION_KEY` bump and a dated #92 notice. Before implementing, read `MaxFields` of the commitments pallet
on finney. Effort is about 2-3 days with tests and a localnet run (commitment parsing, per-element change
detection, the top-up rule, miner and competition-submit tooling writing two fields).
