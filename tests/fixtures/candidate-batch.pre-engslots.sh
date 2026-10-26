#!/bin/bash
# candidate-batch.sh: one daily engine-competition batch on a FRESH, EPHEMERAL Verda RTX PRO 6000 (RUNBOOK-P.md, path P,
# s1m-supervisor #21620). DRAFT, never executed (19 Oct 2026): spend is not approved yet.
#
#   candidate-batch.sh plan                     print every step (default; no API call, no ssh, nothing written)
#   candidate-batch.sh run   [--gonogo]         create -> provision -> (gonogo) -> cycle -> collect -> delete -> receipt
#   candidate-batch.sh <phase>                  one phase: preflight create provision gonogo cycle collect delete orphans receipt
#   candidate-batch.sh deadman                  exact-ID delete of this batch's box (started by the deadman timer)
#
# Every real step needs EXECUTE=1 AND SPEND_CARD=<approved card id>; otherwise the script only prints what it would do.
# Trust direction: Sandy pulls from the box, the box never pushes and never holds a credential (H1). Everything the box
# returns (state, review records, logs) is UNTRUSTED data: it is hashed, size-capped and parsed only by check_state below
# (python3 -I); no agent reads it with tools enabled (RUNBOOK-P.md P6).
set -euo pipefail
umask 077

JOB=sn92-fast-launch-20261015
H=/home/flori/jobs/$JOB/engbox/hardening
V=/home/flori/jobs/reax-validator-host-20261011/bin/verda.py
DAY=$(date -u +%Y%m%d)
BATCH=${BATCH:-cand-$DAY}                            # one batch per UTC day; the id names the box, its tag and its folder
B=$H/batches/$BATCH
UNTRUSTED=$B/untrusted                               # everything copied off the box
MAX_EVALS=${MAX_EVALS:-8}                               # new entries per batch cycle (REAX_ENGINE_MAX_EVALS 1..32); opening 20 Oct: 8
CAP_H=${CAP_H:-3}                                     # hard wall-clock cap from create to delete (deadman)
REG_HOSTS=${REG_HOSTS:-ghcr.io registry-1.docker.io auth.docker.io quay.io}   # worker -> registry API (manifest + label check), 443 only
ALLOWED_LOCS=${ALLOWED_LOCS:-FIN-02 FIN-03}                # never FIN-01: Hetzner AS24940 (crypto rule)
TYPE=${TYPE:-1RTXPRO6000.30V}                         # non-CC, same class as engbox-01
IMAGE=${IMAGE:-24.04.cuda13.4.docker}                 # same OS image as engbox-01 (driver 580.178.04 there, VERIFY per box)
OS_GB=${OS_GB:-200}
PRICE=${PRICE:-2.395}                                 # Verda list price 19 Oct 2026 (GET /instance-types)
PRICE_MAX=${PRICE_MAX:-2.60}
COST_CAP=${COST_CAP:-8}                               # CAP_H x PRICE_MAX, rounded up
SSH_KEY_ID=${SSH_KEY_ID:-e1097703-a557-408c-9061-ae2904817d88}   # public key only; the private key never leaves Sandy
SSH_KEY=/home/flori/.ssh/reax_node_ed25519
H1_STOCK='^(/opt/nvidia/nsight-(compute|systems)/[0-9.]+/[^ ]+|/etc/pki/fwupd(-metadata)?/LVFS-CA[A-Za-z0-9-]*\.pem|/etc/request-key\.d/id_resolver\.conf|/etc/nvidia-container-toolkit/nvidia-cdi-refresh\.env)$'
REAX_NODE_FP=SHA256:RtDVfboGrPEwxnsKpifkDsfG6SI3c4OyKh2dNAfkcdM   # ssh-keygen -lf $SSH_KEY.pub
SANDY_IP=65.109.49.103
DRIVER=580.178.04
RUNSC_REL=${RUNSC_REL:-20261012.0}
CACHE=${CACHE:-/home/flori/jobs/$JOB/engbox/hardening/cache}   # Sandy-side, built once per release (RUNBOOK-P.md P0)
WORKER_TAR=${WORKER_TAR:-$CACHE/worker-1.2.0rc9.post4.tar}     # docker save of the reviewed worker image
WORKER_IMAGE=${WORKER_IMAGE:-reax-competition-worker:1.2.0rc9.post4}
WORKER_ID=${WORKER_ID:-}                              # expected image ID (sha256:...), recorded when the tarball was made
WEIGHTS=${WEIGHTS:-$CACHE/plumb-4b@55de037801a8a9b9de3db5c0e16cef86210c2186}
FLA_CONFIGS=${FLA_CONFIGS:-$CACHE/fla-configs}   # = runtimes/s1-fast/fla-configs of the deployed release (engconfigs.PUBLISHED_GPU_CLASSES)
SEED=$H/state/accepted/engine-competition.json        # last state Sandy accepted (input of this batch)
REVIEWED=$H/state/review/REVIEWED                     # operator markers REVIEWED/<digest>, written on Sandy only
SUBTENSOR_HOST=entrypoint-finney.opentensor.ai
OWN_NETS="65.109.49.103, 95.133.253.35, 86.38.238.226, 65.108.15.165, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, 100.64.0.0/10"
MAX_STATE_BYTES=$((64 * 1024 * 1024))
# R7: the measurement context Sandy expects the seed to carry (engvalidator.measurement_context; ENGINE_REFERENCE_IMAGE of the shipped worker)
EXPECT_REFERENCE=${EXPECT_REFERENCE:-ghcr.io/reaxlabs/reax-engine-s1-fast@sha256:9bd41632e33bd90377f7e29bdb9410b18cfe1fb557200160b9ebe9222abcecea}
EXPECT_THRESHOLDS=${EXPECT_THRESHOLDS:-c973e35f186a49e081322b0432f58d566f68c843030451c139266be4412eb320}   # engthresholds.FROZEN_SHA256
EXPECT_CHAIN=${EXPECT_CHAIN:-finney:92}
EXPECT_TRACK=s1-fast
# R2: Sandy-side chain reads go through the repo's own SdkChain (release checkout + its venv; the system python's bittensor is broken)
CHAIN_SRC=${CHAIN_SRC:-/home/flori/jobs/$JOB/workers/engopen/wt/src}
CHAIN_PY=${CHAIN_PY:-/home/flori/jobs/$JOB/repos/release-wt/.venv/bin/python}
CHAIN_NETWORK=${CHAIN_NETWORK:-finney}
CHAIN_NETUID=${CHAIN_NETUID:-92}
ENGINE_START_BLOCK=${ENGINE_START_BLOCK:-}   # = competition_split.ENGINE_TRACK_START_BLOCK of the deployed release (empty = raw blocks)
FIRST_SEEN_OLDER_OK=${FIRST_SEEN_OLDER_OK:-1}         # 1 (default, lead 19 Oct): a new first_seen may predate head_at_create; the EARLIEST on-chain commitment match stays mandatory. 0 = verbatim R2b
SYNC_SH=/home/flori/jobs/$JOB/engbox/sync.sh          # R8: single writer of the engine state file on the validator

MODE=${1:-plan}; shift || true
GONOGO=0; [ "${1:-}" = --gonogo ] && GONOGO=1
LIVE=0; [ "${EXECUTE:-0}" = 1 ] && [ -n "${SPEND_CARD:-}" ] && [ "$MODE" != plan ] && LIVE=1
if [ "$LIVE" = 1 ]; then mkdir -p "$B" "$UNTRUSTED"; fi

log() { echo "$(date -u +%FT%TZ) [$BATCH] $*" | { if [ "$LIVE" = 1 ]; then tee -a "$B/batch.log"; else cat; fi; }; }
die() {
  log "ABORT: $*"
  if [ "$LIVE" = 1 ] && [ -s "$B/box_id" ] && [ "$MODE" != delete ] && [ "$MODE" != deadman ] && [ "$MODE" != recheck ]; then phase_delete || true; phase_orphans || true; phase_receipt || true; fi
  exit 1
}
# R5: an exit outside die() (set -e) in a full run still deletes the box instead of waiting for the 3 h deadman
on_exit() {
  local rc=$?
  if [ "$LIVE" = 1 ] && [ "$MODE" = run ] && [ -s "$B/box_id" ] && ! is_done delete; then
    log "ABORT: unexpected exit (rc $rc), deleting the box"; phase_delete || true; phase_orphans || true; phase_receipt || true
  fi
}
trap on_exit EXIT
# run: print in plan mode, execute otherwise
run() { if [ "$LIVE" = 1 ]; then "$@"; else printf '  PLAN: %s\n' "$*"; fi; }
# feed: producer on the left of a pipe into E (plan: prints to stderr, outputs nothing); save: sink (plan: drains stdin)
feed() { if [ "$LIVE" = 1 ]; then "$@"; else printf '  PLAN: %s |\n' "$*" >&2; fi; }
save() { if [ "$LIVE" = 1 ]; then tee "$1"; else cat >/dev/null; printf '  PLAN: > %s\n' "$1" >&2; fi; }
api() { if [ "$LIVE" = 1 ]; then python3 "$V" "$@"; else printf '  PLAN: verda %s\n' "$*" >&2; echo '{"status": 0, "body": null}'; fi; }
status_of() { python3 -I -c 'import json,sys; print(json.load(sys.stdin)["status"])'; }
host_ip() { cat "$B/ip"; }
# E: root ssh to the candidate box. No agent forwarding, no port forwarding, our key only, per-batch known_hosts.
_ssh() {
  ssh -q -a -x -o ClearAllForwardings=yes -o IdentitiesOnly=yes -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
      -o UserKnownHostsFile="$B/known_hosts" -o ServerAliveInterval=30 -o ConnectTimeout=20 "root@$(host_ip)" "$@"
}
E() {
  if [ "$LIVE" != 1 ]; then printf '  PLAN: E %s\n' "$(printf '%s' "$*" | tr -s ' \n' ' ' | head -c 170)" >&2; return 0; fi
  _ssh "$@"
}
# R3: EU = E for everything after miner code ran on the box (cycle, collect, delete). The remote command's stderr is
# UNTRUSTED: capped on the box (head -c 1 MiB, the box shell is bash) and appended to $UNTRUSTED/stderr.txt on Sandy, never
# to the terminal or batch.log. ssh's own errors (-q: only fatal ones) land in that file too; callers log their own rc.
EU() {
  if [ "$LIVE" != 1 ]; then printf '  PLAN: EU %s\n' "$(printf '%s' "$*" | tr -s ' \n' ' ' | head -c 170)" >&2; return 0; fi
  _ssh 'exec 2> >(head -c 1048576 >&2);' "$@" 2>> "$UNTRUSTED/stderr.txt"
}
receipt() { [ "$LIVE" = 1 ] && printf '%s\n' "$1" >> "$B/receipt-parts.jsonl"; log "receipt: $1"; }
done_mark() { [ "$LIVE" = 1 ] && touch "$B/.done-$1"; return 0; }
is_done() { [ -e "$B/.done-$1" ]; }

# R2: Sandy-side chain read (public finney entrypoint via the repo's SdkChain, as the validator does). Prints one JSON line
# {"block": <head>, "commits": {<digest>: [[hotkey, commitment_block], ...]}} for reax-eng/1 commitments. Fails closed.
sandy_chain_snapshot() {
  [ -x "$CHAIN_PY" ] || { echo "sandy_chain_snapshot: no python at $CHAIN_PY" >&2; return 1; }
  timeout 180 "$CHAIN_PY" -I - "$CHAIN_SRC" "$CHAIN_NETWORK" "$CHAIN_NETUID" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from reax_subnet.chain import SdkChain
from reax_subnet.engcomp import parse_engine_commitment
snap = SdkChain(sys.argv[2], int(sys.argv[3]), None, commitments=True).snapshot()
commits = {}
def eng(text):   # dual-role commitments (since DUALROLE_START_BLOCK) carry other role fields before the reax-eng/1 one
    t = text if isinstance(text, str) else ""
    i = t.find("reax-eng/1:")
    return parse_engine_commitment(t[i:]) if i >= 0 else None
for n in snap.neurons:
    p = eng(n.commitment)
    if p:
        commits.setdefault(p[2], []).append([n.hotkey, n.commitment_block])
print(json.dumps({"block": snap.block, "commits": commits}))
PY
}
sandy_chain_head() { sandy_chain_snapshot | python3 -I -c 'import json,sys; print(int(json.load(sys.stdin)["block"]))'; }

# R7: the context fields the seed must carry, computed on Sandy: references/thresholds/tracks/chain from the release constants,
# weights = engrunner.weights_identity() mirrored over $WEIGHTS (same walk, same samples), context_sha256 = engvalidator.context_sha256.
# Prints the expected context_sha256 and weights identity; exit 1 and a reason on any mismatch with the seed.
check_seed_context() {
  python3 -I - "$SEED" "$WEIGHTS" "$EXPECT_REFERENCE" "$EXPECT_THRESHOLDS" "$EXPECT_CHAIN" "$EXPECT_TRACK" <<'PY'
import hashlib, json, os, sys
seed_path, wdir, ref, thr, chain, track = sys.argv[1:7]
JSON_MAX, HDR_MAX, SAMPLE = 32 * 2**20, 128 * 2**20, 64 * 2**10   # engrunner IDENTITY_* constants
def fail(why): print("check_seed_context:", why); sys.exit(1)
def weights_identity(d):
    h, n = hashlib.sha256(), 0
    def sample(fh, at):
        fh.seek(at); h.update(hashlib.sha256(fh.read(SAMPLE)).digest())
    for root, dirs, files in os.walk(d, onerror=lambda e: (_ for _ in ()).throw(e)):
        dirs.sort()
        for f in sorted(files):
            path = os.path.join(root, f); size = os.path.getsize(path)
            h.update(f"{os.path.relpath(path, d)}\0{size}\n".encode())
            if f.endswith(".json") and size <= JSON_MAX:
                h.update(hashlib.sha256(open(path, "rb").read()).digest())
            elif f.endswith(".safetensors"):
                with open(path, "rb") as fh:
                    head = fh.read(8); h.update(head)
                    if len(head) == 8:
                        hlen = min(int.from_bytes(head, "little"), HDR_MAX)
                        h.update(hashlib.sha256(fh.read(hlen)).digest())
                        sample(fh, 8 + hlen)
                        if size > 8 + hlen + SAMPLE: sample(fh, size - SAMPLE)
            n += 1
    return h.hexdigest() if n else fail("weights directory empty")
if not os.path.isfile(seed_path): fail(f"no seed at {seed_path} (the box would stamp an unverified context)")
try: seed = json.load(open(seed_path, encoding="utf-8"))
except ValueError: fail("seed is not JSON")
ctx = {"references": {track: ref}, "weights": weights_identity(wdir), "thresholds": thr, "tracks": [track], "chain": chain}
want = hashlib.sha256(json.dumps(ctx, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if seed.get("context") != ctx: fail("seed context fields differ from the Sandy-computed expectation")
if seed.get("context_sha256") != want: fail("seed context_sha256 != sha256(expected context)")
if seed.get("thresholds_sha256") != thr: fail("seed thresholds_sha256 differs")
print(json.dumps({"context_sha256": want, "weights": ctx["weights"], "saved_at": seed.get("saved_at")}))
PY
}

# R8: single writer. engbox/sync.sh copies engbox-01's engine state file to the validator; while it still does, a
# published batch state would be overwritten (or overwrite). PUBLISH=1 is refused until sync.sh no longer names the file.
check_single_writer() {
  if [ -e "$SYNC_SH" ] && grep -v '^[[:space:]]*#' "$SYNC_SH" | grep -q 'engine-competition\.json'; then
    receipt "{\"phase\":\"publish_guard\",\"result\":\"refused\",\"why\":\"$SYNC_SH still writes engine-competition.json\"}"
    die "PUBLISH=1 refused: $SYNC_SH still syncs engine-competition.json to the validator (two writers). Remove that sync (or the engine file from it) first."
  fi
  receipt "{\"phase\":\"publish_guard\",\"result\":\"ok\",\"sync_sh\":\"does not write engine-competition.json\"}"
}

# ------------------------------------------------------------------------------------------------ P1 preflight (Sandy only)
phase_preflight() {
  log "preflight: host, lock, approvals, Sandy-side inputs"
  [ "$(hostname)" = Ubuntu-2204-jammy-amd64-base ] || die "not on Sandy"
  if [ "$LIVE" = 1 ]; then exec 9>"$B/.lock"; flock -n 9 || { LIVE=0; log "ABORT: another run of $BATCH holds the lock (that run owns the box; nothing deleted)"; exit 1; }; fi
  [ "$LIVE" = 1 ] || log "plan mode: EXECUTE=1 and SPEND_CARD=<card id> are both required for a real run"
  for f in "$WORKER_TAR" "$WEIGHTS/config.json" "$CACHE/runsc-$RUNSC_REL/gvisor.tar.zstd" "$CACHE/runsc-$RUNSC_REL/gvisor.tar.zstd.sha512" \
           "$CACHE/runsc-$RUNSC_REL/gvisor.tar" "$CACHE/runsc-$RUNSC_REL/gvisor.tar.sha512" \
           "$WEIGHTS.manifest.sha256"; do
    if [ "$LIVE" = 1 ]; then [ -e "$f" ] || die "missing Sandy input $f (RUNBOOK-P.md P0)"; else log "needs $f"; fi
  done
  [ "$LIVE" = 1 ] && [ -z "$WORKER_ID" ] && die "WORKER_ID (expected image ID of $WORKER_TAR) not set"
  # upstream bundle (sha512 published with the release) -> plain tar decompressed on Sandy; the box checks the tar's sha512
  run bash -c "cd '$CACHE/runsc-$RUNSC_REL' && sha512sum -c gvisor.tar.zstd.sha512 && cmp <(zstd -dc gvisor.tar.zstd) gvisor.tar && sha512sum -c gvisor.tar.sha512"
  run bash -c "cd '$WEIGHTS' && sha256sum -c --quiet '$WEIGHTS.manifest.sha256'"
  # the subtensor IPs the worker bridge may reach, resolved on Sandy at batch time (the box resolves nothing for the worker)
  if [ "$LIVE" = 1 ]; then
    getent ahostsv4 "$SUBTENSOR_HOST" | awk '{print $1}' | sort -u > "$B/subtensor4"; [ -s "$B/subtensor4" ] || die "no subtensor IPs"
  fi
  # R7: the seed (produced on Sandy by release code) must carry the context Sandy expects; the box only ever sees a seed that passed
  if [ "$LIVE" = 1 ]; then
    local ctx; ctx=$(check_seed_context) || die "seed context check failed: $ctx"
    receipt "{\"phase\":\"preflight\",\"seed_context\":$ctx,\"expect_reference\":\"$EXPECT_REFERENCE\",\"note\":\"seed produced on Sandy by release code at saved_at\"}"
    [ "${PUBLISH:-0}" = 1 ] && check_single_writer
  else
    log "needs: seed context (reference $EXPECT_REFERENCE, weights identity of $WEIGHTS, thresholds, chain $EXPECT_CHAIN) == $SEED; PUBLISH=1 additionally needs $SYNC_SH free of engine-competition.json"
  fi
  run ~/bin/gpu-pod-guard status
  done_mark preflight
}

# ------------------------------------------------------------------------------------------------ P2 create (keyless, exact ID)
phase_create() {
  if [ -s "$B/box_id" ] && [ -s "$B/ip" ]; then log "create: box exists ($(cat "$B/box_id")), not creating again"; return 0; fi
  local loc rid out id
  if [ -s "$B/box_id" ]; then
    # a box whose create answer was lost and that was adopted by exact ID: finish attach/deadman/wait only
    id=$(cat "$B/box_id"); rid=$(cat "$B/reservation_id" 2>/dev/null || true); log "create: resuming adopted box $id"
  else
  # idempotency: a box carrying this batch's tag (a create whose answer got lost) is adopted, never duplicated
  local seen
  seen=$(api GET "/instances?tag=reax-batch=$BATCH" | python3 -I -c '
import json,sys
b=json.load(sys.stdin)["body"] or []
print(" ".join(x["id"] for x in b if isinstance(x,dict) and x.get("status") not in ("discontinued","deleting")))')
  [ -n "$seen" ] && die "a live box with tag reax-batch=$BATCH exists ($seen) but no box_id file: reconcile by hand"
  loc=$(api GET /instance-availability | python3 -I -c "
import json,sys
for x in json.load(sys.stdin)['body'] or []:
    if '$TYPE' in x['availabilities'] and x['location_code'] in '$ALLOWED_LOCS'.split(): print(x['location_code']); break")
  [ "$LIVE" = 1 ] && [ -z "$loc" ] && { receipt "{\"phase\":\"create\",\"result\":\"no_capacity\",\"type\":\"$TYPE\"}"; log "no $TYPE available: batch skipped (inconclusive)"; exit 3; }
  # R2: chain head as Sandy sees it just before the box exists; check_state compares NEW first_seen entries against it.
  # Best effort: if the read fails the file says "unavailable" and a batch that reports a new first_seen entry is rejected.
  if [ "$LIVE" = 1 ]; then
    sandy_chain_head > "$B/head_at_create" 2>"$B/head_at_create.err" || { echo unavailable > "$B/head_at_create"; log "WARNING: chain head unavailable (see head_at_create.err)"; }
    # 20 Oct: commitments as Sandy sees them at create; check_state also accepts a first_seen that matched a commitment then
    # (an entrant pruned during the batch must not void the whole batch; the validator pays only live commitments anyway).
    sandy_chain_snapshot > "$B/chain-create.json" 2>"$B/chain-create.json.err" || rm -f "$B/chain-create.json"
  else log "chain head read on Sandy -> $B/head_at_create"; fi
  rid=$(run ~/bin/gpu-pod-guard reserve --job "$JOB" --provider verda --name "$BATCH" --hourly-price "$PRICE" \
          --max-hourly-price "$PRICE_MAX" --cost-cap "$COST_CAP" --runtime-hours "$CAP_H" \
        | python3 -I -c 'import json,sys
try: print(json.load(sys.stdin)["reservation_id"])
except Exception: pass')
  [ "$LIVE" = 1 ] && { [ -n "$rid" ] || die "gpu-pod-guard reserve refused"; echo "$rid" > "$B/reservation_id"; }
  out=$(api POST /instances "{\"instance_type\":\"$TYPE\",\"image\":\"$IMAGE\",\"ssh_key_ids\":[\"$SSH_KEY_ID\"],
    \"location_code\":\"${loc:-PLAN}\",\"hostname\":\"$BATCH\",\"description\":\"REAX SN92 ephemeral engine candidate box $BATCH\",
    \"tags\":[{\"key\":\"reax-batch\",\"value\":\"$BATCH\"},{\"key\":\"owner\",\"value\":\"$JOB\"}],
    \"os_volume\":{\"name\":\"$BATCH-os\",\"size\":$OS_GB},\"is_spot\":false,\"contract\":\"PAY_AS_YOU_GO\"}") \
    || out='{"status": 0, "body": null}'   # client timeout: the box may exist -> reconciled below by hostname, never re-created
  if [ "$LIVE" = 1 ] && [ "$(printf '%s' "$out" | status_of)" = 0 ]; then
    sleep 30
    id=$(api GET /instances | python3 -I -c "
import json,sys
b=json.load(sys.stdin)['body'] or []
m=[x['id'] for x in b if isinstance(x,dict) and x.get('hostname')=='$BATCH' and x.get('status') not in ('discontinued','deleting')]
print(m[0] if len(m)==1 else '')")
    [ -n "$id" ] && out="{\"status\": 202, \"body\": \"$id\"}"
  fi
  [ "$LIVE" = 1 ] && printf '%s\n' "$out" > "$B/create.json"
  if [ "$LIVE" = 1 ] && [ "$(printf '%s' "$out" | status_of)" != 202 ]; then
    ~/bin/gpu-pod-guard release --job "$JOB" --reservation-id "$rid" --reason "create failed" || true
    die "create failed: $(printf '%s' "$out" | head -c 200)"
  fi
  id=$(printf '%s' "$out" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["body"] or "PLAN-ID")')
  [ "$LIVE" = 1 ] && { echo "$id" > "$B/box_id"; date -u +%FT%TZ > "$B/created_at"; }
  fi
  run ~/bin/gpu-pod-guard attach --job "$JOB" --provider verda --reservation-id "${rid:-PLAN}" --pod-id "$id" --name "$BATCH" --hourly-price "$PRICE"
  # deadman: independent of this process; gpu-reaper does not list Verda pods (provider_pods: lium/runpod only)
  run systemd-run --user --unit "cand-deadman-$BATCH" --on-active="${CAP_H}h" --setenv=BATCH="$BATCH" --setenv=EXECUTE=1 \
      --setenv=SPEND_CARD="${SPEND_CARD:-}" "$H/candidate-batch.sh" deadman
  receipt "{\"phase\":\"create\",\"box_id\":\"$id\",\"location\":\"${loc:-}\",\"type\":\"$TYPE\",\"price\":$PRICE,\"head_at_create\":\"$(cat "$B/head_at_create" 2>/dev/null || true)\",\"reservation_id\":\"${rid:-}\",\"created_at\":\"$(date -u +%FT%TZ)\",\"deadline_h\":$CAP_H}"
  # wait for running + IP (exact ID)
  local st ip rate
  for _ in $(seq 1 60); do
    read -r st ip rate < <(api GET "/instances/$id" | python3 -I -c '
import json,sys
b=json.load(sys.stdin)["body"]; b=b if isinstance(b,dict) else {}
print(b.get("status","?"), b.get("ip") or "-", b.get("price_per_hour") or 0)')
    [ "$st" = running ] && [ "$ip" != - ] && break; [ "$LIVE" = 1 ] || break
    case "$st" in error|no_capacity|installation_failed|discontinued|notfound) die "box status $st";; esac
    sleep 20
  done
  if [ "$LIVE" = 1 ]; then
    [ "$st" = running ] || die "box not running after 20 min (status $st)"
    python3 -I -c 'import sys; sys.exit(float(sys.argv[1]) > float(sys.argv[2]))' "$rate" "$PRICE_MAX" || die "provider rate $rate > $PRICE_MAX"
    echo "$ip" > "$B/ip"; receipt "{\"phase\":\"running\",\"ip\":\"$ip\",\"provider_rate\":$rate}"
    # No SN92 workload on Hetzner (crypto rule): Verda FIN-01 runs on Hetzner AS24940 (17 + 20 Oct). Check the ASN, delete on match.
    asn=$(curl -s --max-time 10 "https://ipinfo.io/$ip/org" || true)
    case "$asn" in AS24940*|*Hetzner*) receipt "{\"phase\":\"running\",\"result\":\"hetzner_refused\",\"asn\":\"${asn%% *}\"}"; die "box $ip is on Hetzner ($asn): refused";; esac
    [ -n "$asn" ] || log "WARN: ASN lookup for $ip failed (continuing; location $(cat "$B/location" 2>/dev/null) is allow-listed)"
  fi
  for _ in $(seq 1 30); do E true && break; [ "$LIVE" = 1 ] || break; sleep 10; done
  done_mark create
}

# ------------------------------------------------------------------------------------------------ P3 provision (H0-H5)
phase_provision() {
  log "provision H0: baseline + driver gate"
  E 'set -e; hostname; uname -r; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader;
     nvidia-smi conf-compute -f; docker version --format "{{.Server.Version}}"; runc --version | head -1;
     nvidia-ctk --version | head -1; modinfo nvidia | grep -i ^license; date +%s' | save "$UNTRUSTED/h0.txt" >/dev/null
  if [ "$LIVE" = 1 ]; then
    grep -q ", $DRIVER," "$UNTRUSTED/h0.txt" || die "driver is not $DRIVER (H0): inconclusive batch"
    grep -qi "CC status: OFF" "$UNTRUSTED/h0.txt" || die "confidential computing not OFF"
    local skew; skew=$(( $(tail -1 "$UNTRUSTED/h0.txt") - $(date +%s) )); [ "${skew#-}" -le 5 ] || die "clock skew ${skew}s"
  fi

  log "provision H1: credential inventory (names only; must be empty)"
  E 'ls -la /root/.docker 2>&1; echo "== AUTHS $(grep -c "\"auth" /root/.docker/config.json 2>/dev/null || echo 0)"; echo "== FIND";
     find / -xdev \( -name "id_*" -o -name "*.pem" -o -name "*.key" -o -iname "*token*" -o -iname "*secret*" -o -name "*.env" -o -name credentials \) \
       -not -path "/proc/*" -not -path "/sys/*" -not -path "/usr/*" -not -path "/var/lib/docker/*" -not -path "/etc/ssl/*" \
       -not -name "*.pub" -not -path "/etc/ssh/ssh_host_*" 2>/dev/null; echo "== KEYS $(grep -c . /root/.ssh/authorized_keys)";
     ssh-keygen -lf /root/.ssh/authorized_keys' | save "$UNTRUSTED/h1.txt" >/dev/null
  if [ "$LIVE" = 1 ]; then
    grep -qx "== AUTHS 0" "$UNTRUSTED/h1.txt" || die "H1: docker config.json carries auth entries"
    # stock-image files that match the name patterns (Verda 24.04.cuda13.4.docker, seen 19 Oct): exact allowlist, anything else dies
    local stray; stray=$(sed -n '/^== FIND$/,/^== KEYS /p' "$UNTRUSTED/h1.txt" | grep -v '^== ' | grep -Ev "$H1_STOCK" || true)
    [ -z "$stray" ] || die "H1: credential-like files found on the box: $(printf '%s' "$stray" | head -3 | tr '\n' ' ' | tr -cd '[:alnum:]/._ -')"
    grep -qx "== KEYS 1" "$UNTRUSTED/h1.txt" || die "H1: authorized_keys is not exactly one key"
    grep -q " $REAX_NODE_FP " "$UNTRUSTED/h1.txt" || die "H1: authorized_keys is not the reax_node key"
  fi

  log "provision H2: pinned runsc from Sandy (sha512 checked on both ends; no apt source)"
  # the full release bundle: runsc + shim + gvisor-bin/ sidecars (gvisor_sentry etc.; runsc >= 2026 refuses to start without them)
  feed tar -C "$CACHE/runsc-$RUNSC_REL" -cf - gvisor.tar gvisor.tar.sha512 \
    | E 'set -e; rm -rf /tmp/runsc; mkdir -p /tmp/runsc && tar -C /tmp/runsc -xf - && cd /tmp/runsc && sha512sum -c gvisor.tar.sha512 &&
         tar -C /usr/local/bin --no-same-owner -xf gvisor.tar && test -x /usr/local/bin/gvisor-bin/gvisor_sentry && test ! -e /etc/apt/sources.list.d/gvisor.list &&
         runsc --version && runsc nvproxy list-supported-drivers | tr " " "\n" | grep -qx '"$DRIVER"' && echo DRIVER_SUPPORTED' \
    | save "$UNTRUSTED/h2.txt" >/dev/null
  [ "$LIVE" = 1 ] && { grep -qx DRIVER_SUPPORTED "$UNTRUSTED/h2.txt" || die "runsc $RUNSC_REL does not list $DRIVER: NO-GO"; }

  log "provision H3: runsc docker runtime (args = engrunner RELEASE_RUNTIME_ARGS + overlay/platform + host-uds=create: the engine creates /run/reax/engine.sock for the runc-side runner; create-only, the sandbox cannot connect to host sockets)"
  E 'set -e; cp -p /etc/docker/daemon.json /etc/docker/daemon.json.pre-runsc 2>/dev/null || echo "{}" > /etc/docker/daemon.json.pre-runsc;
     jq ".runtimes.runsc = {\"path\": \"/usr/local/bin/runsc\", \"runtimeArgs\": [\"--nvproxy=true\",
          \"--nvproxy-allowed-driver-capabilities=compute,utility\", \"--overlay2=root:memory\", \"--platform=systrap\",
          \"--network=none\", \"--host-uds=create\"]}" /etc/docker/daemon.json.pre-runsc > /etc/docker/daemon.json;
     dockerd --validate --config-file /etc/docker/daemon.json; systemctl restart docker;
     docker info --format "{{json .Runtimes}}" | jq -c ".runsc"; sha256sum /etc/docker/daemon.json'

  log "provision H5: DNS root-only to fixed resolvers, no NTP, worker bridge -> subtensor only, ingress ssh from Sandy only"
  local sub; sub=$( [ "$LIVE" = 1 ] && paste -sd, "$B/subtensor4" | sed 's/,/, /g' || echo "<subtensor IPs>")
  E "set -e; systemctl disable --now systemd-timesyncd 2>/dev/null || true; timedatectl set-ntp false 2>/dev/null || true
     systemctl disable --now systemd-resolved; rm -f /etc/resolv.conf; printf 'nameserver 8.8.8.8\nnameserver 8.8.4.4\n' > /etc/resolv.conf
     docker network inspect reaxw >/dev/null 2>&1 || docker network create -o com.docker.network.bridge.name=br-reaxw reaxw >/dev/null
     NETUID=\$(id -u systemd-network 2>/dev/null || echo 65534)
     # 20 Oct: the worker reads image manifests itself (owner label check) -> worker bridge may reach the registries' API on 443.
     mkdir -p /w; : > /w/reghosts; for h in $REG_HOSTS; do getent ahostsv4 \$h | awk '{print \$1}' | sort -u > /tmp/reg.\$h; test -s /tmp/reg.\$h; echo \"\$h \$(head -1 /tmp/reg.\$h)\" >> /w/reghosts; done
     REG4=\$(cat /tmp/reg.* | sort -u | paste -sd, | sed 's/,/, /g'); test -n \"\$REG4\"
     cat > /etc/reax-cand.nft <<NFT
table inet reaxcand {
  set own4 { type ipv4_addr; flags interval; elements = { $OWN_NETS } }
  set subtensor4 { type ipv4_addr; elements = { $sub } }
  set registry4 { type ipv4_addr; elements = { \$REG4 } }
  chain input {
    type filter hook input priority 10; policy drop;
    iifname \"lo\" accept
    ct state established,related accept
    ip saddr $SANDY_IP tcp dport 22 accept
    udp sport 67 udp dport 68 accept
    counter drop
  }
  chain output {
    type filter hook output priority 10; policy drop;
    oifname \"lo\" accept
    ct state established,related accept
    meta nfproto ipv6 counter drop
    # DHCP: broadcast only (a unicast renewal to the DHCP server is dropped: VERIFY the lease is longer than the 3 h cap)
    meta skuid \$NETUID ip daddr 255.255.255.255 udp sport 68 udp dport 67 accept
    meta skuid 0 ip daddr { 8.8.8.8, 8.8.4.4 } meta l4proto { udp, tcp } th dport 53 accept
    meta skuid 0 ip daddr @own4 counter drop
    meta skuid 0 tcp dport 443 accept
    counter log prefix \"cand-out-drop \" limit rate 6/minute drop
  }
  chain forward {
    type filter hook forward priority 10; policy drop;
    ct state established,related accept
    meta nfproto ipv6 counter drop
    iifname \"br-reaxw\" ip daddr @subtensor4 tcp dport 443 accept
    iifname \"br-reaxw\" ip daddr @registry4 tcp dport 443 accept
    # 20 Oct: registry blobs redirect to rotating CDN hosts (pkg-containers.githubusercontent.com, Docker CDN) -> the worker gets
    # DNS to the two fixed resolvers and outbound 443, never to own/private ranges. Miner code still runs under runsc --network=none.
    iifname \"br-reaxw\" ip daddr { 8.8.8.8, 8.8.4.4 } meta l4proto { udp, tcp } th dport 53 accept
    iifname \"br-reaxw\" ip daddr @own4 counter drop
    iifname \"br-reaxw\" ip daddr { 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10, 169.254.0.0/16, 127.0.0.0/8 } counter drop
    iifname \"br-reaxw\" tcp dport 443 accept
    counter log prefix \"cand-fwd-drop \" limit rate 6/minute drop
  }
}
NFT
     nft -c -f /etc/reax-cand.nft && nft -f /etc/reax-cand.nft && nft list table inet reaxcand | sha256sum"
  # verify from the box (expected: BLOCKED_USER, PULL_OK, NO_NET, BLOCKED_BRIDGE, BLOCKED_V6)
  E 'timeout 5 curl -6 -s -o /dev/null https://[2a01:4f9:5a:26c0::2]/ 2>/dev/null || echo BLOCKED_V6
     timeout 8 setpriv --reuid=65534 --regid=65534 --clear-groups getent hosts example.com >/dev/null || echo BLOCKED_USER
     timeout 120 docker pull -q busybox:1.36 >/dev/null && echo PULL_OK
     docker run --rm --runtime=runsc --network none busybox:1.36 wget -T 5 -q -O- http://1.1.1.1 >/dev/null 2>&1 || echo NO_NET
     docker run --rm --network reaxw busybox:1.36 wget -T 5 -q -O- http://1.1.1.1 >/dev/null 2>&1 || echo BLOCKED_BRIDGE' \
    | save "$UNTRUSTED/h5.txt" >/dev/null
  if [ "$LIVE" = 1 ]; then for w in BLOCKED_USER PULL_OK NO_NET BLOCKED_BRIDGE BLOCKED_V6; do grep -qx "$w" "$UNTRUSTED/h5.txt" || die "H5 verify: $w missing"; done; fi

  log "provision inputs: worker image, weights, seeded state, REVIEWED markers (all streamed from Sandy, nothing pulled with creds)"
  feed cat "$WORKER_TAR" | E "docker load -q >/dev/null && docker image inspect --format '{{.Id}}' $WORKER_IMAGE" | save "$UNTRUSTED/worker_id" >/dev/null
  [ "$LIVE" = 1 ] && { [ "$(cat "$UNTRUSTED/worker_id")" = "$WORKER_ID" ] || die "worker image ID mismatch"; }
  feed tar -C "$WEIGHTS" -cf - . | E 'set -e; mkdir -p /w/plumb && tar -C /w/plumb -xf - && chown -R 0:0 /w/plumb && chmod -R a+rX,go-w /w/plumb'
  feed cat "$WEIGHTS.manifest.sha256" | E 'cd /w/plumb && sha256sum -c --quiet - && echo WEIGHTS_OK' | save "$UNTRUSTED/weights.txt" >/dev/null
  [ -f "$FLA_CONFIGS/rtx-pro-6000-blackwell-server/SHA256SUMS" ] || die "FLA configs missing at $FLA_CONFIGS"
  feed tar -C "$FLA_CONFIGS" -cf - . | E 'set -e; rm -rf /w/fla-configs; mkdir -p /w/fla-configs && tar -C /w/fla-configs -xf - && chown -R 0:0 /w/fla-configs && chmod -R a+rX,go-w /w/fla-configs && cd /w/fla-configs/rtx-pro-6000-blackwell-server && sha256sum -c --quiet SHA256SUMS && echo FLA_OK' | save "$UNTRUSTED/fla.txt" >/dev/null
  [ "$LIVE" = 1 ] && { grep -qx FLA_OK "$UNTRUSTED/fla.txt" || die "FLA configs verify failed on the box"; }
  E 'mkdir -p /w/comp/state /w/comp/engines/review/REVIEWED /w/gate && chown -R 4321:4321 /w/comp && chmod 0755 /w/gate'
  install_worker_sh
  if [ "$LIVE" = 1 ] && [ -s "$SEED" ]; then
    E 'cat > /w/comp/state/engine-competition.json && chown 4321:4321 /w/comp/state/engine-competition.json' < "$SEED"
  fi
  if [ "$LIVE" = 1 ] && [ -d "$REVIEWED" ]; then
    tar -C "$REVIEWED" -cf - . | E 'tar -C /w/comp/engines/review/REVIEWED -xf - && chown -R 0:0 /w/comp/engines/review/REVIEWED && chmod -R a+r,go-w /w/comp/engines/review/REVIEWED'
  fi
  if [ "$LIVE" = 1 ]; then sha256sum "$SEED" > "$B/seed.sha256" 2>/dev/null || echo "none (first batch, empty state)" > "$B/seed.sha256"; fi
  receipt "{\"phase\":\"provision\",\"driver\":\"$DRIVER\",\"runsc\":\"$RUNSC_REL\",\"worker_image_id\":\"${WORKER_ID:-}\",\"seed_sha256\":\"$(cut -d' ' -f1 "$B/seed.sha256" 2>/dev/null || true)\"}"
  done_mark provision
}

# /w/worker.sh on the box: the worker container (our image, runc + nvidia utility only, uid 4321, docker socket, no DNS,
# bridge reaxw -> subtensor only, gate mounted ro). Usage on the box: [ENTRY=<entrypoint>] [NAME=<name>] /w/worker.sh <args>.
install_worker_sh() {
  E "cat > /w/worker.sh && chmod 0700 /w/worker.sh" <<WSH
#!/bin/sh
set -eu
SUB=\$(head -1 /w/subtensor4)
exec docker run --rm --name "\${NAME:-reax-cand-worker}" --network reaxw --dns 8.8.8.8 --dns 8.8.4.4 --add-host "$SUBTENSOR_HOST:\$SUB" \\
  --user 4321:4321 --group-add "\$(stat -c %g /var/run/docker.sock)" --init --read-only \\
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g --cap-drop ALL --security-opt no-new-privileges:true \\
  \$(awk '{printf "--add-host %s:%s ", \$1, \$2}' /w/reghosts) \\
  --gpus device=0 -e NVIDIA_DRIVER_CAPABILITIES=utility --ulimit nofile=65536:65536 --log-driver json-file --log-opt max-size=50m \\
  -e HOME=/tmp -e REAX_NETWORK=finney -e REAX_NETUID=92 -e REAX_MECHANISM=s1 -e REAX_LOG_JSON=1 -e REAX_COMPETITIONS_MODE=local \\
  -e REAX_ENGINE_WEIGHTS_DIR=/w/plumb -e REAX_ENGINE_WORKDIR=/w/comp/engines -e REAX_COMPETITIONS_STATE_DIR=/w/comp/state \\
  -e REAX_ENGINE_MAX_EVALS=$MAX_EVALS -e REAX_ENGINE_GPU=0 -e REAX_ENGINE_NVML_CHECK=0 -e REAX_ENGINE_DOCKER_RUNTIME=runsc -e REAX_ENGINE_FLA_CONFIG_ROOT=/w/fla-configs \\
  -e REAX_ENGINE_CYCLE_GATE_FILE=/w/gate/OPEN -e REAX_ENGINE_CYCLE_GATE_MAX_AGE_MIN=20 \\
  -v /var/run/docker.sock:/var/run/docker.sock -v /w/plumb:/w/plumb:ro -v /w/fla-configs:/w/fla-configs:ro -v /w/comp:/w/comp -v /w/gate:/w/gate:ro \\
  --entrypoint "\${ENTRY:-reax-competition-worker}" $WORKER_IMAGE "\$@"
WSH
  if [ "$LIVE" = 1 ]; then head -1 "$B/subtensor4" | E 'cat > /w/subtensor4'; else E 'cat > /w/subtensor4'; fi
}

# ------------------------------------------------------------------------------------------------ P4 GO/NO-GO (first box, H2/H4)
phase_gonogo() {
  log "gonogo H4: reference vs reference under runsc (reference image only, no miner code)"
  E "set -e; REF=\$(ENTRY=python3 NAME=reax-cand-ref /w/worker.sh -c 'from reax_subnet.config import ENGINE_REFERENCE_IMAGE as R; print(R)')
     echo \"reference \$REF\"; docker pull -q \"\$REF\" >/dev/null
     docker run --rm --runtime=runsc --gpus device=0 --network none --read-only --cap-drop ALL --user 10002:10002 \
       -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --entrypoint nvidia-smi \"\$REF\" -L
     ENTRY=reax-engine-eval NAME=reax-cand-gonogo /w/worker.sh --reference \"\$REF\" --candidate \"\$REF\" --weights /w/plumb \
       --docker-runtime runsc --workdir /w/comp/gonogo --out /w/comp/gonogo.json
     head -c 1048576 /w/comp/gonogo.json" | save "$UNTRUSTED/gonogo.json" >/dev/null
  log "gonogo H4b: docker pause/unpause of a runsc container (the engine runner pauses sandboxes between measurements)"
  E 'docker rm -f reax-cand-pause >/dev/null 2>&1
     docker run -d --name reax-cand-pause --runtime=runsc --network none --read-only --cap-drop ALL --user 10002:10002 busybox:1.36 sleep 60 >/dev/null || exit 1
     st() { docker inspect -f "{{.State.Status}} {{.State.Paused}}" reax-cand-pause; }
     [ "$(st)" = "running false" ] && docker pause reax-cand-pause >/dev/null && [ "$(st)" = "paused true" ] &&
       docker unpause reax-cand-pause >/dev/null && [ "$(st)" = "running false" ] && echo PAUSE_OK
     rc=$?; docker rm -f reax-cand-pause >/dev/null 2>&1; exit $rc' | save "$UNTRUSTED/pause.txt" >/dev/null || true
  local pause=null   # plan: not tested
  if [ "$LIVE" = 1 ]; then grep -qx PAUSE_OK "$UNTRUSTED/pause.txt" && pause=true || die "gonogo NO-GO: docker pause/unpause under runsc did not pass (PAUSE_OK missing)"; fi
  log "gonogo pass criteria (the lead reads the numbers, RUNBOOK-P.md P4): completes, distance 0 / top agreement 1.0, pause works (PAUSE_OK, checked above), dec/s recorded"
  receipt "{\"phase\":\"gonogo\",\"pause_ok\":$pause,\"out_sha256\":\"$(sha256sum "$UNTRUSTED/gonogo.json" 2>/dev/null | cut -d' ' -f1)\"}"
  done_mark gonogo
}

# ------------------------------------------------------------------------------------------------ P5 one gated cycle
phase_cycle() {
  log "cycle step 1: dry start with the gate CLOSED (context check before any candidate code runs)"
  EU "rm -f /w/gate/OPEN; /w/worker.sh --engine --once >/w/comp/worker-dry.log 2>&1; head -c $MAX_STATE_BYTES /w/comp/state/engine-competition.json" \
    | save "$UNTRUSTED/state-dry.json" >/dev/null
  if [ "$LIVE" = 1 ]; then check_state "$UNTRUSTED/state-dry.json" dry || die "context check failed (weights/reference/thresholds/chain differ from the seed)"; fi
  log "cycle step 2: gate OPEN, one cycle (--once), gate refreshed every 10 min from Sandy, hard stop before the deadline"
  EU 'touch /w/gate/OPEN; rm -f /w/comp/worker.rc
     setsid sh -c "/w/worker.sh --engine --once >/w/comp/worker.log 2>&1; echo \$? >/w/comp/worker.rc" </dev/null >/dev/null 2>&1 &'
  local end=$(( $(date +%s) + CAP_H * 3600 - 1800 ))   # 30 min of the cap stay for collect + delete
  while [ "$LIVE" = 1 ]; do
    if EU 'test -e /w/comp/worker.rc'; then break; fi
    if [ "$(date +%s)" -ge "$end" ]; then log "cycle over budget: gate closed, candidates killed (inconclusive)"; break; fi
    EU 'touch /w/gate/OPEN' || log "gate refresh failed (the gate closes by itself after 20 min)"
    sleep 600
  done
  # shellcheck disable=SC2016  # expands on the box
  EU 'rm -f /w/gate/OPEN; docker rm -f reax-cand-worker >/dev/null 2>&1 || true
     docker ps -aq --filter name=reax-eng- | xargs -r docker rm -fv >/dev/null
     echo "rc=$(cat /w/comp/worker.rc 2>/dev/null || echo none) containers=$(docker ps -aq | wc -l) sandboxes=$(pgrep -c -f "runsc-(sandbox|gofer)" || true)"' \
    | save "$UNTRUSTED/cycle-end.txt" >/dev/null
  done_mark cycle
}

# ------------------------------------------------------------------------------------------------ P6 collect (pull, hash, check)
phase_collect() {
  log "collect: state + review records + worker log tail, size-capped, hashed; all UNTRUSTED"
  EU "head -c $MAX_STATE_BYTES /w/comp/state/engine-competition.json" | save "$UNTRUSTED/engine-competition.json" >/dev/null
  EU 'nft list chain inet reaxcand forward 2>&1 | head -c 65536; nft list chain inet reaxcand output 2>&1 | head -c 65536' | save "$UNTRUSTED/nft-counters.txt" >/dev/null   # #22442 (2)
  EU 'mkdir -p /w/comp/engines/review && cd /w/comp/engines/review && tar --exclude=./REVIEWED -cf - . | head -c 16777216' | save "$UNTRUSTED/review-records.tar" >/dev/null
  EU 'tail -c 4194304 /w/comp/worker.log 2>/dev/null; cd /w/comp/engines/logs 2>/dev/null && tail -c 262144 ./* 2>/dev/null; true' \
    | save "$UNTRUSTED/logs.txt" >/dev/null
  [ "$LIVE" = 1 ] && printf 'UNTRUSTED: copied off a box that ran miner code. Do not open with an agent that has tools enabled.\n' > "$UNTRUSTED/README"
  if [ "$LIVE" = 1 ]; then ( cd "$UNTRUSTED" && sha256sum ./* ) > "$B/collected.sha256"; else log "sha256 of every collected file -> collected.sha256"; fi
  local verdict=rejected
  if [ "$LIVE" = 1 ] && check_state "$UNTRUSTED/engine-competition.json" final; then verdict=accepted; fi
  receipt "{\"phase\":\"collect\",\"state_sha256\":\"$(sha256sum "$UNTRUSTED/engine-competition.json" 2>/dev/null | cut -d' ' -f1)\",\"verdict\":\"$verdict\"}"
  if [ "$verdict" = accepted ]; then
    mkdir -p "$H/state/accepted" "$H/state/history"
    cp "$SEED" "$H/state/history/$BATCH-seed.json" 2>/dev/null || true
    cp "$UNTRUSTED/engine-competition.json" "$H/state/history/$BATCH.json"
    install -m 0644 "$UNTRUSTED/engine-competition.json" "$SEED.tmp" && mv "$SEED.tmp" "$SEED"
    # relay to the validator is the lead's step (PUBLISH=1, RUNBOOK-P.md P7); off while the engine track is held
    if [ "${PUBLISH:-0}" = 1 ]; then publish_state; fi
  fi
  done_mark collect
}

# publish_state: Sandy -> rxnode-03, atomic, the same path as engbox/sync.sh; the hash sidecar names the accepted batch.
# Off by default (PUBLISH=1): while ENGINE_TRACK_START_UTC is None the validator pays no engine share anyway.
publish_state() {
  local X=/home/flori/jobs/$JOB/node/pod.sh sum
  check_single_writer   # R8 (also checked in preflight): never publish while sync.sh still writes the same file
  sum=$(sha256sum "$SEED" | cut -d' ' -f1)
  timeout 60 "$X" 'cat > /w/comp/state/.engine-competition.json.tmp && chown 4321:4321 /w/comp/state/.engine-competition.json.tmp &&
    chmod 644 /w/comp/state/.engine-competition.json.tmp && mv /w/comp/state/.engine-competition.json.tmp /w/comp/state/engine-competition.json' < "$SEED"
  printf '%s  engine-competition.json  batch=%s\n' "$sum" "$BATCH" | timeout 60 "$X" 'cat > /w/comp/state/engine-competition.json.sha256'
  receipt "{\"phase\":\"publish\",\"sha256\":\"$sum\",\"to\":\"rxnode-03\"}"
}

archive_commits() {   # <untrusted state> <out json>
  timeout 600 "$CHAIN_PY" -I - "$CHAIN_SRC" "$1" "$CHAIN_NETUID" <<'PY' > "$2.tmp" && mv "$2.tmp" "$2"
import json, re, sys
sys.path.insert(0, sys.argv[1])
import bittensor as bt
from reax_subnet.chain import _commitment_text, _commitment_block
from reax_subnet.engcomp import parse_engine_commitment
st = json.load(open(sys.argv[2], encoding="utf-8"))
want = set()
for t in (st.get("tracks") or {}).values():
    for d, fs in ((t or {}).get("first_seen") or {}).items():
        if isinstance(fs, list) and len(fs) == 2 and isinstance(fs[0], int) and isinstance(fs[1], str) \
           and re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{47,48}", fs[1]) and 0 < fs[0] < 10**9:
            want.add((fs[0], fs[1]))
sub = bt.Subtensor(network="archive"); out = {}
for blk, hk in sorted(want)[:32]:
    mg = sub.subnets.metagraph(int(sys.argv[3]), commitments=True, block=blk + 1)
    for n in mg.neurons:
        if str(n.hotkey) == hk:
            t = _commitment_text(getattr(n, "commitment", None)); i = t.find("reax-eng/1:")
            p = parse_engine_commitment(t[i:]) if i >= 0 else None   # dual-role commitments: the reax-eng/1 field
            if p: out.setdefault(p[2], []).append([hk, _commitment_block(getattr(n, "commitment", None))])
print(json.dumps({"source": "archive", "commits": out}))
PY
}

# check_state <file> dry|final: Sandy-side plausibility checks against the seed. Not a proof (the box is untrusted); it
# blocks a replaced/rewound/foreign state, unreviewed digests/champions and forged first_seen (who gets paid). Exit 0 = acceptable.
# A NEW first_seen entry needs the chain: Sandy reads head + reax-eng/1 commitments itself (sandy_chain_snapshot, fail closed).
# Everything it prints from the file is sanitised (digests ^[0-9a-f]+$ only, names [a-z0-9-]).
check_state() {
  local commits="" rc=0
  chk_py() {
    python3 -I - "$1" "$2" "$SEED" "$REVIEWED" "$(cat "$B/created_at" 2>/dev/null || echo 1970-01-01T00:00:00Z)" "$MAX_STATE_BYTES" \
      "$(cat "$B/head_at_create" 2>/dev/null || true)" "$FIRST_SEEN_OLDER_OK" "$3" "$ENGINE_START_BLOCK" <<'PY'
import json, os, re, sys, time, datetime
path, kind, seed_path, reviewed_dir, created, cap, head_s, older_ok, commits_path, start_s = sys.argv[1:11]
def dg(d):   # an untrusted digest, printable only if it is hex (sha256: prefix stripped)
    d = str(d).split(":")[-1]
    return d[:16] if re.fullmatch(r"[0-9a-f]+", d) else "<non-hex>"
def nm(t): return t if re.fullmatch(r"[a-z0-9-]{1,16}", str(t)) else "<odd-name>"
def fail(why): print("check_state:", why); sys.exit(1)
if os.path.getsize(path) == 0 or os.path.getsize(path) >= int(cap): fail("empty or at the size cap")
try: new = json.load(open(path, encoding="utf-8"))
except ValueError: fail("not JSON")
if not isinstance(new, dict) or not isinstance(new.get("tracks"), dict): fail("not a state object")
seed = json.load(open(seed_path, encoding="utf-8")) if os.path.exists(seed_path) else None
t0 = datetime.datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
sa = new.get("saved_at")
if not isinstance(sa, int) or not (t0 - 600 <= sa <= time.time() + 600): fail("saved_at outside the batch window")
if seed:
    for k in ("version", "thresholds_sha256", "context_sha256"):
        if new.get(k) != seed.get(k): fail(f"{k} differs from the seed")
reviewed = set(os.listdir(reviewed_dir)) if os.path.isdir(reviewed_dir) else set()
chain = None
if os.path.exists(commits_path):
    chain = json.load(open(commits_path, encoding="utf-8"))   # written by Sandy (sandy_chain_snapshot), trusted
    for cc in [os.path.join(os.path.dirname(commits_path), f) for f in ("chain-create.json", "chain-archive.json")]:   # Sandy reads (trusted)
      if os.path.exists(cc):
        for d, lst in json.load(open(cc, encoding="utf-8")).get("commits", {}).items():
            cur = chain["commits"].setdefault(d, [])
            for c in lst:
                if c not in cur: cur.append(c)
need_chain = False
for t, st in new["tracks"].items():
    if not isinstance(st, dict): fail(f"{nm(t)}: track is not an object")
    old = (seed or {}).get("tracks", {}).get(t, {})
    oh, nh = old.get("history", []), st.get("history", [])
    if oh and not any(oh[s:] == nh[:len(oh) - s] for s in range(len(oh))): fail(f"{nm(t)}: history rewritten (no overlap with the seed)")
    for d, fs in old.get("first_seen", {}).items():
        if st.get("first_seen", {}).get(d) != fs: fail(f"{nm(t)}: first_seen of {dg(d)} changed")
    for d in st.get("results", {}):
        if d not in old.get("results", {}) and d.split(":")[-1] not in reviewed: fail(f"{nm(t)}: result for unreviewed {dg(d)}")
    # champion (R2a): must be a measured result and reviewed on Sandy, or be the seed's own champion
    ch = (st.get("champion") or {}).get("digest")
    och = (old.get("champion") or {}).get("digest")
    if ch:
        if not isinstance(ch, str) or ch not in st.get("results", {}): fail(f"{nm(t)}: champion {dg(ch)} is not in results")
        if ch != och and ch.split(":")[-1] not in reviewed: fail(f"{nm(t)}: champion {dg(ch)} has no REVIEWED marker and is not the seed's champion")
        # co-champions (rc9.post7, Florian #22468): while uid 198 / uid 4 hold the title, a new champion must reach the
        # decided bar 2.82332 x (1 + 0.03); the batch worker images before p7 still use the champion's own score (critic F1b)
        co = ("576fea84b347fc85160f788d737db616be4b5ca4cde19268e0ce1d1f5cdfe12f", "ec2ee6ed3df69903bf9df5420cde8f869280647ce7d23a3837a2397b2ce34272")
        if ch != och and och in co and ch not in co:
            sc = (st.get("champion") or {}).get("score")
            if not isinstance(sc, (int, float)) or sc < 2.82332066708034 * 1.03:
                fail(f"{nm(t)}: new champion {dg(ch)} score {sc} is below the co-champion bar 2.90802 (#22468)")
    # first_seen (R2b): digest -> (block, hotkey) decides who is paid; a NEW entry is checked against the chain as Sandy sees it
    for d, fs in st.get("first_seen", {}).items():
        if d in old.get("first_seen", {}): continue
        if not (isinstance(fs, list) and len(fs) == 2 and isinstance(fs[0], int) and not isinstance(fs[0], bool) and isinstance(fs[1], str)):
            fail(f"{nm(t)}: first_seen of {dg(d)} is malformed")
        blk, hk = fs
        if chain is None: need_chain = True; continue
        if not head_s.isdigit(): fail(f"{nm(t)}: new first_seen {dg(d)} but no chain head was recorded at create ({head_s[:12] or 'none'!r})")
        if blk >= int(chain["block"]) + 1: fail(f"{nm(t)}: first_seen {dg(d)} block is in the future")
        if blk < int(head_s) and older_ok != "1": fail(f"{nm(t)}: new first_seen {dg(d)} block {blk} < head at create {head_s}")
        com = chain["commits"].get(d.split(":")[-1])
        if not com: fail(f"{nm(t)}: no on-chain commitment for new first_seen {dg(d)}")
        # pre-opening commitments count at the opening block (validator engcomp.effective_block, #21905)
        eff = lambda b: max(int(b), int(start_s)) if start_s.isdigit() else int(b)
        first = min(eff(c[1]) for c in com)
        if [hk, blk] not in [[c[0], eff(c[1])] for c in com] or blk != first or blk <= 0:
            fail(f"{nm(t)}: first_seen {dg(d)} does not match the earliest on-chain commitment (hotkey/block)")
        # same-block tie (supervisor #21796 cond 2): more than one hotkey at the earliest block -> no automatic owner
        if len({c[0] for c in com if eff(c[1]) == first}) > 1:
            fail(f"{nm(t)}: first_seen {dg(d)}: several hotkeys committed it at the earliest block {first} (manual review)")
    if kind == "final":
        for d, r in sorted(st.get("results", {}).items()):
            if not isinstance(r, dict): continue
            kv = " ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}={v}"
                          for k, v in sorted(r.items()) if re.fullmatch(r"[a-z_]{1,24}", str(k)) and
                          (isinstance(v, (int, float, bool)) or (isinstance(v, str) and re.fullmatch(r"[a-z_0-9.-]{0,24}", v))))
            print(f"check_state: result {nm(t)} {dg(d)} {kv[:400]}")
    if kind == "final" and ch and ch != och:
        print(f"check_state: NOTE {nm(t)}: new champion {dg(ch)} (award gate: owner-signed approval before any payment)")
if need_chain: sys.exit(9)
print("check_state: ok", kind)
PY
  }
  chk_py "$1" "$2" "" || rc=$?
  if [ "$rc" = 9 ]; then
    commits="$B/chain-$2.json"
    sandy_chain_snapshot > "$commits" 2>"$commits.err" || { echo "check_state: new first_seen entries but the chain read on Sandy failed"; return 1; }
    chk_py "$1" "$2" "$commits" && return 0
    # 20 Oct: an entrant can commit and be pruned between create and collect. For each new first_seen (block, hotkey) that
    # neither view has, Sandy asks the ARCHIVE node for that hotkey's commitment at that block (only ints/ss58 are read from
    # the untrusted file) and re-checks once. A forged first_seen still fails: the chain must show that exact commitment.
    archive_commits "$1" "$B/chain-archive.json" 2>"$B/chain-archive.err" || return 1
    chk_py "$1" "$2" "$commits"; return
  fi
  return "$rc"
}

# ------------------------------------------------------------------------------------------------ P7 delete (exact ID) + ledger
phase_delete() {
  local id; id=$(cat "$B/box_id" 2>/dev/null || echo PLAN-ID)
  if [ "$LIVE" = 1 ] && [ ! -s "$B/box_id" ]; then log "delete: no box_id for $BATCH, nothing to delete"; return 0; fi
  local vols; vols=$(api GET "/instances/$id" | python3 -I -c '
import json,sys
b=json.load(sys.stdin)["body"]; b=b if isinstance(b,dict) else {}
print(json.dumps(sorted({v for v in [b.get("os_volume_id")] + list(b.get("volume_ids") or []) if v})))')
  log "delete: exact ID $id, volumes $vols, permanently"
  local out; out=$(api PUT /instances "{\"action\":\"delete\",\"id\":\"$id\",\"volume_ids\":$vols,\"delete_permanently\":true}")
  [ "$LIVE" = 1 ] && printf '%s\n' "$out" > "$B/delete.json"
  local st="?"
  for _ in $(seq 1 30); do
    st=$(api GET "/instances/$id" | python3 -I -c '
import json,sys
d=json.load(sys.stdin); b=d["body"]
st=b.get("status") if d["status"]==200 and isinstance(b,dict) else "gone"
print("gone" if st=="notfound" else st)')
    { [ "$st" = gone ] || [ "$st" = discontinued ]; } && break; [ "$LIVE" = 1 ] || break; sleep 10
  done
  if [ "$LIVE" = 1 ] && { [ "$st" = gone ] || [ "$st" = discontinued ]; }; then
    ~/bin/gpu-pod-guard release --job "$JOB" --pod-id "$id" --reason "batch $BATCH done ($st)" | tee "$B/ledger-release.txt"
    systemctl --user stop "cand-deadman-$BATCH.timer" 2>/dev/null || true
    # the OS volume survives delete_permanently (seen 19 Oct): delete each volume by exact ID and confirm
    local v vst
    for v in $(printf '%s' "$vols" | python3 -I -c 'import json,sys; print(" ".join(json.load(sys.stdin)))'); do
      api PUT /volumes "{\"action\":\"delete\",\"id\":[\"$v\"],\"is_permanent\":true}" > "$B/delete-volume-$v.json" || true
      for _ in $(seq 1 12); do
        vst=$(api GET "/volumes/$v" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); b=d["body"]; print(b.get("status") if d["status"]==200 and isinstance(b,dict) else "gone")')
        { [ "$vst" = deleted ] || [ "$vst" = gone ]; } && break; sleep 10
      done
      log "delete: volume $v -> $vst"
    done
    date -u +%FT%TZ > "$B/deleted_at"
  elif [ "$LIVE" = 1 ]; then
    log "delete NOT confirmed (status $st): deadman stays armed; post a #22 warning (lead)"; return 1
  fi
  receipt "{\"phase\":\"delete\",\"box_id\":\"$id\",\"status\":\"$st\",\"volumes\":$vols,\"deleted_at\":\"$(cat "$B/deleted_at" 2>/dev/null || true)\",\"ledger_release\":\"$(head -c 300 "$B/ledger-release.txt" 2>/dev/null | tr -d '"\n' || true)\"}"
  done_mark delete
}

# ------------------------------------------------------------------------------------------------ P8 orphan check
phase_orphans() {
  log "orphans: no live instance with our tag/name, no volume of ours left, no detached volume at all (report only)"
  local inst vols
  inst=$(api GET /instances | python3 -I -c "
import json,sys
b=json.load(sys.stdin)['body'] or []
print(len([x for x in b if isinstance(x,dict) and str(x.get('hostname','')).startswith('cand-') and x.get('status') not in ('discontinued',)]))")
  vols=$(api GET /volumes | python3 -I -c "
import json,sys
b=json.load(sys.stdin)['body'] or []
ours=[x for x in b if isinstance(x,dict) and str(x.get('name','')).startswith('cand-') and x.get('status') not in ('deleted',)]
det=[x for x in b if isinstance(x,dict) and x.get('status')=='detached']
print(len(ours), len(det))")
  local trash; trash=$(api GET /volumes/trash | python3 -I -c "
import json,sys
b=json.load(sys.stdin)['body'] or []
print(len([x for x in b if isinstance(x,dict) and str(x.get('name','')).startswith('$BATCH')]))")
  # Verda API v1 has no separate public-IP resource (openapi.json, 19 Oct 2026): the IP goes with the instance (VERIFY P8)
  receipt "{\"phase\":\"orphans\",\"live_cand_instances\":${inst:-0},\"cand_volumes_left\":${vols%% *},\"detached_volumes_any\":${vols##* },\"cand_volumes_in_trash\":${trash:-0},\"ip_resource\":\"none in API v1\"}"
  [ "$LIVE" = 1 ] && [ "${inst:-0}" != 0 ] && log "WARNING: live cand-* instance(s) remain"
  [ "$LIVE" = 1 ] && [ "${vols%% *}" != 0 ] && log "WARNING: cand-* volume(s) remain"
  done_mark orphans
}

# ------------------------------------------------------------------------------------------------ P9 receipt (one JSON line)
phase_receipt() {
  [ "$LIVE" = 1 ] || { log "receipt: merges receipt-parts.jsonl into $H/batches.jsonl (one line per batch)"; return 0; }
  python3 -I - "$B/receipt-parts.jsonl" "$BATCH" "$PRICE" >> "$H/batches.jsonl" <<'PY'
import json, sys, datetime
parts, batch, price = sys.argv[1], sys.argv[2], float(sys.argv[3])
out = {"batch": batch, "phases": {}}
for line in open(parts):
    p = json.loads(line); out["phases"][p.pop("phase")] = p
c, d = out["phases"].get("create", {}), out["phases"].get("delete", {})
if c.get("created_at") and d.get("deleted_at"):
    f = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    h = (f(d["deleted_at"]) - f(c["created_at"])).total_seconds() / 3600
    out["box_hours"], out["cost_usd_est"] = round(h, 2), round(h * price, 2)
print(json.dumps(out, sort_keys=True))
PY
  log "receipt appended to $H/batches.jsonl; summary to #22 @s1m-supervisor-20260927 is the lead's step"
}

case "$MODE" in
  plan)
    log "PLAN ONLY (nothing is created, no ssh, no files written)"
    for p in preflight create provision gonogo cycle collect delete orphans receipt; do echo "== $p"; "phase_$p"; done ;;
  run)
    for p in preflight create provision; do if [ "$LIVE" = 1 ] && is_done "$p" && [ "$p" != preflight ]; then continue; fi; "phase_$p"; done
    [ "$GONOGO" = 1 ] && phase_gonogo
    for p in cycle collect delete orphans receipt; do if [ "$LIVE" = 1 ] && is_done "$p"; then continue; fi; "phase_$p"; done ;;
  preflight|create|provision|gonogo|cycle|collect|delete|orphans|receipt) "phase_$MODE" ;;
  deadman) log "DEADMAN: cap of $CAP_H h reached"; phase_delete; phase_orphans; phase_receipt ;;
  recheck)   # 20 Oct: re-run check_state on an already collected batch (box gone) and accept/publish like phase_collect
    [ -s "$UNTRUSTED/engine-competition.json" ] || die "recheck: no collected state"
    verdict=rejected; if check_state "$UNTRUSTED/engine-competition.json" final; then verdict=accepted; fi
    receipt "{\"phase\":\"recheck\",\"state_sha256\":\"$(sha256sum "$UNTRUSTED/engine-competition.json" | cut -d' ' -f1)\",\"verdict\":\"$verdict\"}"
    if [ "$verdict" = accepted ]; then
      mkdir -p "$H/state/accepted" "$H/state/history"
      cp "$SEED" "$H/state/history/$BATCH-seed.json" 2>/dev/null || true
      cp "$UNTRUSTED/engine-competition.json" "$H/state/history/$BATCH.json"
      install -m 0644 "$UNTRUSTED/engine-competition.json" "$SEED.tmp" && mv "$SEED.tmp" "$SEED"
      if [ "${PUBLISH:-0}" = 1 ]; then publish_state; fi
    fi ;;
  *) echo "usage: $0 plan | run [--gonogo] | <phase> | deadman | recheck"; exit 2 ;;
esac
