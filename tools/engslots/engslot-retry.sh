#!/bin/bash
# Timer entry point. Never retries past the scheduled slot's four-hour deadline.
set -euo pipefail
umask 077
: "${ENGSLOT_SRC:?Set ENGSLOT_SRC to reviewed release src}"
CANDIDATE=${CANDIDATE:-/home/flori/jobs/sn92-fast-launch-20261015/engbox/hardening/candidate-batch.sh}
CANDIDATE_BATCH_LIBRARY=1 source "$CANDIDATE" library
CHAIN_SRC=$ENGSLOT_SRC
ENGSLOT_SLOT_UTC=$("$CHAIN_PY" -I - "$CHAIN_SRC" <<'PY'
import sys
from datetime import datetime,timezone
sys.path.insert(0,sys.argv[1])
from reax_subnet.competition_split import ENGINE_CROWN_SLOTS_UTC
slots=[s for s in ENGINE_CROWN_SLOTS_UTC if s<=datetime.now(timezone.utc)]
print(max(slots).isoformat() if slots else '')
PY
)
[ -n "$ENGSLOT_SLOT_UTC" ] || { echo 'No engine slot due yet'; exit 0; }
export ENGSLOT_SLOT_UTC CANDIDATE ENGSLOT_SRC CHAIN_PY
end=$(date -u -d "$ENGSLOT_SLOT_UTC + 4 hours" +%s)
wrapper=$(dirname "$(readlink -f "$CANDIDATE")")/engslot-finalize.sh
while :; do
  remaining=$((end - $(date -u +%s)))
  [ "$remaining" -gt 0 ] || break
  if timeout "$remaining" "$wrapper"; then exit 0; fi
  remaining=$((end - $(date -u +%s)))
  [ "$remaining" -gt 0 ] || break
  pause=300; [ "$remaining" -ge 300 ] || pause=$remaining
  sleep "$pause"
done
B=$H/batches/engslot-missed-$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$B"
"$CHAIN_PY" -I - "$ENGSLOT_SLOT_UTC" <<'PY' > "$B/receipt.json"
import json,sys
from datetime import datetime,timezone
print(json.dumps(dict(slot_utc=sys.argv[1],utc=datetime.now(timezone.utc).isoformat(),
    published=False,deferred=False,exit_code=75,reason='missed: four-hour slot retry deadline expired; next worker cycle recovers')))
PY
cat "$B/receipt.json"
exit 75
