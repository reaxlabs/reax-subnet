#!/bin/bash
# Install beside candidate-batch.sh after applying tools/engslots/candidate-batch.patch.
# Defaults to a plan. ENGSLOT_SRC must name the reviewed release's src directory.
set -euo pipefail
umask 077
if [ "${DRY_RUN:-1}" != 0 ]; then
  echo 'DRY_RUN: lock accepted state, fresh Sandy snapshot, offline finalise, check_state slot, history, publish, receipt'
  exit 0
fi
: "${ENGSLOT_SRC:?Set ENGSLOT_SRC to the reviewed release src directory}"
CANDIDATE=${CANDIDATE:-$(dirname "$(readlink -f "$0")")/candidate-batch.sh}
CANDIDATE_BATCH_LIBRARY=1 source "$CANDIDATE" library
CHAIN_SRC=$ENGSLOT_SRC
LIVE=1
BATCH=engslot-$(date -u +%Y%m%dT%H%M%SZ)-$$
B=$H/batches/$BATCH
mkdir -p "$B" "$H/state/history"
SLOT_SNAPSHOT=$B/chain-slot.json
changed=0
published=0
blocked=0
slot_now=$(date -u +%FT%TZ)
reason=""
slot_utc=${ENGSLOT_SLOT_UTC:-}
slot_receipt() {
  local slot_rc=$?
  "$CHAIN_PY" -I - "$BATCH" "$changed" "$published" "$SEED" "$SLOT_SNAPSHOT" "$slot_now" "$slot_rc" "$blocked" "$reason" "$slot_utc" <<'PYRECEIPT' > "$B/receipt.json"
import hashlib,json,sys
from pathlib import Path
batch, changed, published, state, snapshot, now, rc, blocked, reason, slot = sys.argv[1:]
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
print(json.dumps(dict(batch=batch, changed=bool(int(changed)), published=bool(int(published)), utc=now,
                     exit_code=int(rc), reason=reason, slot_utc=slot or None, deferred=int(rc)==75, blocked_by_weights_ok=int(blocked), state_sha256=h(state), snapshot_sha256=h(snapshot)), sort_keys=True))
PYRECEIPT
  return "$slot_rc"
}
trap slot_receipt EXIT
exec 8>"$H/state/.engine-state.lock"
if ! flock -w 300 8; then
  reason="deferred: state publication locked"
  echo "$reason" >&2
  exit 75
fi
# delete.json is only a deletion request. deleted_at records confirmed teardown;
# ledger-release.txt is the guard's teardown receipt. Be conservative if neither exists.
for box in "$H"/batches/*/box_id; do
  [ -e "$box" ] || continue
  batch_dir=${box%/box_id}
  if [ ! -s "$batch_dir/deleted_at" ] && [ ! -s "$batch_dir/ledger-release.txt" ]; then
    reason="deferred: batch ${batch_dir##*/} in flight"
    echo "$reason" >&2
    exit 75
  fi
done
if ! "$CHAIN_PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import reax_subnet.engslot' "$CHAIN_SRC" 2>"$B/engslot-preflight.log"; then
  echo 'engine slot import failed; check release CHAIN_PY/ENGSLOT_SRC' >&2
  exit 1
fi
sandy_chain_snapshot > "$SLOT_SNAPSHOT"
date -u +%FT%TZ > "$B/created_at"
slot_now=$(cat "$B/created_at")
cp "$SEED" "$B/seed.json"
# Import reviewed code explicitly even when the interpreter has an older editable installation.
"$CHAIN_PY" -I - "$CHAIN_SRC" "$SEED" "$SLOT_SNAPSHOT" "$slot_now" "$B/final.json" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from reax_subnet.engslot import main
main(['finalize', '--state', sys.argv[2], '--commitments', sys.argv[3], '--now', sys.argv[4], '--out', sys.argv[5]])
PY
blocked=$("$CHAIN_PY" -I - "$B/seed.json" "$B/final.json" <<'PYBLOCK'
import json,sys
old,new = [json.load(open(p)) for p in sys.argv[1:]]
print(sum(sum(e.get('event') == 'blocked_noncompliant' for e in st.get('history', [])) -
          sum(e.get('event') == 'blocked_noncompliant' for e in old['tracks'][t].get('history', []))
          for t,st in new['tracks'].items()))
PYBLOCK
)
echo "$blocked contenders blocked by weights_ok" | tee "$B/weights-ok-summary.txt"
# A no-op keeps accepted state intact. Re-publish it so a retry recovers a prior publication failure.
changed=$("$CHAIN_PY" -I - "$B/seed.json" "$B/final.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1])) != json.load(open(sys.argv[2]))))
PY
)
if [ "$changed" = 1 ]; then
  check_state "$B/final.json" slot
  cp "$SEED" "$H/state/history/$BATCH-seed.json"
  cp "$B/final.json" "$H/state/history/$BATCH.json"
  install -m 0644 "$B/final.json" "$SEED.tmp"
  mv "$SEED.tmp" "$SEED"
fi
publish_state
published=1
