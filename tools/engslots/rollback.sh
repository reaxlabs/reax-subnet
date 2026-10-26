#!/bin/bash
# Sandy rollback: strip and publish state before enabling previous worker defaults.
set -euo pipefail
umask 077
if [ "${DRY_RUN:-1}" != 0 ]; then
  echo 'DRY_RUN: lock state, back up state/script, clear confirmed pending, restore candidate check_state and pinned previous worker tar/image/ID'
  printf 'DRY_RUN: CANDIDATE_BATCH_LIBRARY=1 source %q library; LIVE=1; publish_state\n' "${CANDIDATE:-<installed candidate-batch.sh>}"
  exit 0
fi
: "${CANDIDATE:?Set CANDIDATE to installed candidate-batch.sh}"
: "${CANDIDATE_BACKUP:?Set CANDIDATE_BACKUP to the pre-post6 candidate-batch backup}"
: "${PREVIOUS_WORKER_TAR:?Set PREVIOUS_WORKER_TAR to retained previous worker tar}"
: "${PREVIOUS_WORKER_IMAGE:?Set PREVIOUS_WORKER_IMAGE to previous worker tag}"
: "${PREVIOUS_WORKER_ID:?Set PREVIOUS_WORKER_ID to recorded sha256 image ID}"
: "${CHAIN_PY:?Set CHAIN_PY to the release virtualenv interpreter}"
rollback_candidate=$CANDIDATE
rollback_h=${H:-$(dirname "$(readlink -f "$CANDIDATE")")}
rollback_seed=${SEED:-$rollback_h/state/accepted/engine-competition.json}
CANDIDATE_BATCH_LIBRARY=1 source "$CANDIDATE" library
CANDIDATE=$rollback_candidate
H=$rollback_h
SEED=$rollback_seed
LIVE=1
B=$H/batches/engslot-rollback-$(date -u +%Y%m%dT%H%M%SZ)-$$
BATCH=${B##*/}
mkdir -p "$B" "$H/state/history"
exec 8>"$H/state/.engine-state.lock"
flock -w 180 8
bash -n "$CANDIDATE_BACKUP"
cp "$SEED" "$B/seed.json"
cp "$CANDIDATE" "$B/candidate-before.sh"
"$CHAIN_PY" -I - "$SEED" "$CANDIDATE_BACKUP" "$PREVIOUS_WORKER_TAR" "$PREVIOUS_WORKER_IMAGE" "$PREVIOUS_WORKER_ID" "$B" <<'PY'
import hashlib,json,re,shlex,sys
from datetime import datetime,timezone
from pathlib import Path
state,backup,tar,image,ident,directory=sys.argv[1:]
b=Path(directory)
if not Path(tar).is_file() or not re.fullmatch(r'sha256:[0-9a-f]{64}',ident):
    raise SystemExit('previous tar missing or previous image ID invalid')
s=Path(backup).read_text()
# Pin the restored defaults; a normal batch still verifies WORKER_ID when loading the tar.
for name,value in [('WORKER_TAR',tar),('WORKER_IMAGE',image),('WORKER_ID',ident)]:
    s,n=re.subn(r'^'+name+r'=.*$',lambda _: name+'='+shlex.quote(value),s,count=1,flags=re.M)
    if n!=1: raise SystemExit('missing previous worker setting '+name)
# Keep library sourcing available for a repeated rollback/publication retry on the old script.
if 'if [ "${CANDIDATE_BATCH_LIBRARY:-0}" = 1 ]; then return 0; fi' not in s:
    marker='\ncase "$MODE" in\n'
    if s.count(marker)!=1: raise SystemExit('cannot locate previous script dispatch')
    s=s.replace(marker,'\nif [ "${CANDIDATE_BATCH_LIBRARY:-0}" = 1 ]; then return 0; fi\n'+marker,1)
(b/'candidate-restored.sh').write_text(s)
raw=json.loads(Path(state).read_text()); cleared=0
for st in raw['tracks'].values():
    c=st.get('challenger')
    if c and c.get('confirmed_at'):
        st['challenger']=None
        if 'pending' in st: st['pending']=None
        st.setdefault('history',[]).append(dict(event='pending_cleared_rollback',digest=c['digest'],
                                               utc=datetime.now(timezone.utc).isoformat()))
        cleared+=1
(b/'final.json').write_text(json.dumps(raw,sort_keys=True,indent=1,allow_nan=False)+'\n')
print(json.dumps(dict(pending_cleared=cleared,previous_worker_tar=tar,previous_worker_image=image,
                     previous_worker_id=ident,check_state_backup_sha256=hashlib.sha256(Path(backup).read_bytes()).hexdigest())))
PY
bash -n "$B/candidate-restored.sh"
cp "$B/seed.json" "$H/state/history/$(basename "$B")-seed.json"
cp "$B/final.json" "$H/state/history/$(basename "$B").json"
# Fail-safe order: sanitize the seed before enabling previous worker defaults/check_state.
install -m 0644 "$B/final.json" "$SEED.tmp"
mv "$SEED.tmp" "$SEED"
# Fail closed before enabling the previous worker if publication fails.
publish_state
install -m 0755 "$B/candidate-restored.sh" "$CANDIDATE.tmp"
mv "$CANDIDATE.tmp" "$CANDIDATE"
"$CHAIN_PY" -I - "$B" "$SEED" "$CANDIDATE" <<'PY' > "$B/receipt.json"
import hashlib,json,sys
from pathlib import Path
b,state,script=map(Path,sys.argv[1:])
print(json.dumps(dict(restored=True,state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
                     candidate_sha256=hashlib.sha256(script.read_bytes()).hexdigest()),sort_keys=True))
PY
