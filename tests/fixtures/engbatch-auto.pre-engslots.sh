#!/bin/bash
# Engine batches without a lead pass (Florian 23 Oct 16:5xZ, board #25088):
#   engbatch-auto.sh auto    every 15 min (timer sn92fl-engbatch-auto): reviewed + committed + unmeasured entries -> start a batch
#   engbatch-auto.sh daily   12:00Z (timer sn92fl-engbatch-daily): the guaranteed daily batch; with nothing pending a receipt only
# Same gates/caps as by hand: REVIEWED markers are the lead's gate, run-batch-any-rc11.sh (ephemeral Verda box, card 17879,
# COST_CAP per batch, gpu-pod-guard). At most AUTO_MAX_PER_DAY started batches per UTC day; a batch is never started while one runs.
J=/home/flori/jobs/sn92-fast-launch-20261015; H=$J/engbox/hardening; W=$J/watch/engbatch; mkdir -p $W
MODE=${1:-auto}; AUTO_MAX_PER_DAY=${AUTO_MAX_PER_DAY:-6}
exec 9>$W/.lock; flock -n 9 || exit 0
log() { echo "$(date -u +%FT%TZ) $MODE $*" >> $W/auto.log; }
post() { /home/flori/bin/agent-board post 22 "$1" --kind note --to sn92-fast-launch-20261015 >/dev/null 2>&1 || true; }
if pgrep -f "^[^ ]*bash .*(run-batch-any-rc11\.sh|candidate-batch\.sh run)" >/dev/null; then
  log "batch already running"
  if [ "$MODE" = daily ]; then  # the daily receipt must exist even when an auto batch occupies the slot
    run=$(tail -1 $W/started.log 2>/dev/null | awk '{print $1}')
    echo "{\"batch\": \"daily-$(date -u +%F)\", \"mode\": \"daily\", \"covered_by_running\": \"$run\", \"box\": null, \"at\": \"$(date -u +%FT%TZ)\"}" >> $H/batches.jsonl
    post "Daily engine batch $(date -u +%d\ %b) 12:00Z: batch $run is still running and measures the pending entries -> no second box (receipt in engbox/hardening/batches.jsonl); its result follows."
  fi
  exit 0
fi
# fresh chain list (the watcher's file if < 30 min old, else list now)
C=$J/watch/engcommits/latest.txt
if [ ! -s "$C" ] || [ $(( $(date +%s) - $(stat -c %Y "$C") )) -gt 1800 ]; then
  timeout 300 $J/venv-tools/bin/python -I $J/bin/engcommits.py $J/workers/mnpub/wt/src 2>/dev/null | grep -v -i warn > $W/commits.txt || { log "chain list failed"; exit 0; }
  C=$W/commits.txt
fi
timeout 90 $J/node/pod.sh 'cat /w/comp/state/engine-competition.json' > $W/published.json 2>/dev/null || { log "cannot read published state"; exit 0; }
pending=$(python3 -I $J/bin/engbatch-pending.py "$C" $H/state/review $W/published.json 2>>$W/auto.log)
day=$(date -u +%F)
if [ -z "$pending" ]; then
  if [ "$MODE" = daily ]; then
    echo "{\"batch\": \"daily-$day\", \"mode\": \"daily\", \"pending\": [], \"box\": null, \"at\": \"$(date -u +%FT%TZ)\"}" >> $H/batches.jsonl
    log "daily: nothing pending (receipt written)"
    post "Daily engine batch $(date -u +%d\ %b) 12:00Z: no reviewed, unmeasured engine entries on chain -> no box rented (receipt in engbox/hardening/batches.jsonl)."
  fi
  exit 0
fi
n=$(grep -c " $day" $W/started.log 2>/dev/null || true); n=${n:-0}
if [ "$n" -ge "$AUTO_MAX_PER_DAY" ]; then log "cap: $n batches started today, pending: $(echo $pending | tr '\n' ' ')"; exit 0; fi
unit=sn92fl-engbatch-$MODE-$(date -u +%Y%m%d%H%M)
echo "$unit $day $(echo "$pending" | tr '\n' ' ')" >> $W/started.log
log "start $unit for: $(echo "$pending" | tr '\n' ' ')"
systemd-run --user --unit "$unit" --collect --property=WorkingDirectory=$H \
  bash -c "WAIT_S=3600 ./run-batch-any-rc11.sh > batches/$unit.log 2>&1; $J/bin/engbatch-report.sh $unit" >/dev/null 2>&1 \
  || { log "systemd-run failed"; exit 0; }
post "Engine batch $unit started automatically ($MODE) for reviewed, unmeasured entries: $(echo "$pending" | awk '{printf "uid %s (%s) ", $1, substr($2,1,12)}'). Result follows."
