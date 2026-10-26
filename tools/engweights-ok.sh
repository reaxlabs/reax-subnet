#!/bin/bash
# Install beside the patched candidate-batch.sh. Inventory must cover ALL non-reference layers.
set -euo pipefail
umask 077
if [ "${DRY_RUN:-1}" != 0 ]; then
  echo 'DRY_RUN: lock accepted state, review marker + inventory screen, weights-only check_state, history, publish, receipt'
  exit 0
fi
: "${ENGSLOT_SRC:?Set ENGSLOT_SRC to the reviewed release src directory}"
: "${WEIGHTS_ACTION:?Set WEIGHTS_ACTION to add or remove}"
: "${WEIGHTS_DIGEST:?Set WEIGHTS_DIGEST to the reviewed digest}"
: "${WEIGHTS_INVENTORY:?Set WEIGHTS_INVENTORY to the complete non-reference layer inventory}"
CANDIDATE=${CANDIDATE:-$(dirname "$(readlink -f "$0")")/candidate-batch.sh}
CANDIDATE_BATCH_LIBRARY=1 source "$CANDIDATE" library
CHAIN_SRC=$ENGSLOT_SRC
LIVE=1
BATCH=engweights-$(date -u +%Y%m%dT%H%M%SZ)-$$
B=$H/batches/$BATCH
mkdir -p "$B" "$H/state/history"
exec 8>"$H/state/.engine-state.lock"
flock -w 180 8 || { echo 'state publication locked' >&2; exit 1; }
changed=0
published=0
weights_receipt() {
  local weights_rc=$?
  "$CHAIN_PY" -I - "$BATCH" "$changed" "$published" "$SEED" "$weights_rc" <<'PY' > "$B/receipt.json"
import hashlib,json,sys
from pathlib import Path
batch,changed,published,state,rc=sys.argv[1:]
print(json.dumps(dict(batch=batch,changed=bool(int(changed)),published=bool(int(published)),exit_code=int(rc),
    state_sha256=hashlib.sha256(Path(state).read_bytes()).hexdigest()),sort_keys=True))
PY
  return "$weights_rc"
}
trap weights_receipt EXIT
if ! "$CHAIN_PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import reax_subnet.engslot' "$CHAIN_SRC" 2>"$B/engslot-preflight.log"; then
  echo 'engine slot import failed; check release CHAIN_PY/ENGSLOT_SRC' >&2
  exit 1
fi
date -u +%FT%TZ > "$B/created_at"
cp "$SEED" "$B/seed.json"
# Bind the screen and compliance log to the same retained inventory bytes.
cp "$WEIGHTS_INVENTORY" "$B/inventory"
"$CHAIN_PY" -I - "$CHAIN_SRC" "$WEIGHTS_ACTION" "$SEED" "$WEIGHTS_DIGEST" "$H/state/review" "$B/inventory" "$B/final.json" <<'PY' > "$B/compliance.jsonl"
import sys
sys.path.insert(0,sys.argv[1])
from reax_subnet.engslot import main
main(['weights-ok',sys.argv[2],'--state',sys.argv[3],'--digest',sys.argv[4],
      '--review-dir',sys.argv[5],'--inventory',sys.argv[6],'--out',sys.argv[7]])
PY
check_state "$B/final.json" weights
cp "$SEED" "$H/state/history/$BATCH-seed.json"
cp "$B/final.json" "$H/state/history/$BATCH.json"
install -m 0644 "$B/final.json" "$SEED.tmp"
mv "$SEED.tmp" "$SEED"
changed=1
publish_state
published=1
