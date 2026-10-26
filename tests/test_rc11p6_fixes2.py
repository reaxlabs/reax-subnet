"""Local drills for delta critic N1-N4; no box, network or installed-script writes."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_rc11p6_fixes import patched_candidate, ROOT
from test_engslot import D, P, SUB, SLOT, BEFORE, document, ready, snapshot, slots
from reax_subnet.engcomp import EvalResult, TrackState, update_champion
from reax_subnet.engslot import check_transition, finalize_document


def executable(path, content):
    path.write_text(content); path.chmod(0o755)
    return path


@pytest.mark.parametrize('offset,accepted', [(0, True), (600, True), (601, False), (7200, False)])
def test_confirmation_cannot_predate_batch(offset, accepted):
    # Critic2 sim.py "backdate": move a real Tue15 confirmation to Tue13,
    # including its otherwise consistent earlier calendar slot.
    old = document(ready()); old['tracks'][P.name]['challenger'] = None
    claimed = SLOT-timedelta(hours=1,minutes=30)
    st = TrackState.from_json(old['tracks'][P.name])
    for i in range(P.confirmations):
        r = EvalResult(D, P.name, True, score=2, block=200+i, session=f'n2-{i}')
        st.results[D] = r
        st.add_cycle_event(dict(event='measurement',digest=D,block=r.block,score=r.score,session=r.session,ok=True))
        update_champion(st,[SUB],P,r.block,now=claimed)
    new=deepcopy(old); new['tracks'][P.name]=st.to_json()
    args=dict(created_at=claimed+timedelta(seconds=offset))
    if accepted: check_transition(old,new,SLOT+timedelta(hours=1),'final',**args)
    else:
        with pytest.raises(ValueError,match='predates batch'): check_transition(old,new,SLOT+timedelta(hours=1),'final',**args)


def test_existing_confirmation_may_precede_batch():
    old=document(ready())
    check_transition(old,deepcopy(old),SLOT,created_at=SLOT)


def test_candidate_patch_passes_creation_time():
    patch=(ROOT/'tools/engslots/candidate-batch.patch').read_text()
    assert 'created_at=datetime.datetime.fromtimestamp(t0, datetime.timezone.utc)' in patch


@pytest.mark.parametrize('marker,deferred', [(None,True), ('delete.json',True), ('deleted_at',False), ('ledger-release.txt',False)])
def test_wrapper_live_batch_markers(tmp_path,marker,deferred):
    hard,seed,script,config=patched_candidate(tmp_path)
    live=hard/'batches/real-batch'; live.mkdir(parents=True); (live/'box_id').write_text('test-id\n')
    if marker: (live/marker).write_text('{}\n')
    # A nondeferred test stops at import (all chain calls are avoided).
    config.update(ENGSLOT_SRC=str(tmp_path/'missing'),ENGSLOT_SLOT_UTC=SLOT.isoformat())
    fake=executable(tmp_path/'python-stub', '#!/bin/bash\nif [[ "$*" == *"import reax_subnet.engslot"* ]]; then exit 1; fi\nexec '+sys.executable+' "$@"\n')
    config['CHAIN_PY']=str(fake)
    result=subprocess.run([str(ROOT/'tools/engslot-finalize.sh')],env=config,capture_output=True,text=True)
    receipt=json.loads(next((hard/'batches').glob('engslot-*/receipt.json')).read_text())
    assert receipt['slot_utc']==SLOT.isoformat() and not receipt['published']
    assert receipt['deferred'] is deferred
    if deferred:
        assert result.returncode==75 and receipt['reason']=='deferred: batch real-batch in flight'
    else: assert result.returncode==1 and receipt['reason']==''


def test_lock_timeout_has_defer_receipt(tmp_path):
    hard,seed,script,config=patched_candidate(tmp_path)
    bindir=tmp_path/'bin'; bindir.mkdir()
    executable(bindir/'flock','#!/bin/bash\n[ "$*" = "-w 300 8" ] || exit 99\nexit 1\n')
    config.update(PATH=str(bindir)+os.pathsep+os.environ['PATH'],ENGSLOT_SLOT_UTC=SLOT.isoformat())
    result=subprocess.run([str(ROOT/'tools/engslot-finalize.sh')],env=config,capture_output=True,text=True)
    assert result.returncode==75
    r=json.loads(next((hard/'batches').glob('*/receipt.json')).read_text())
    assert r['deferred'] and r['reason']=='deferred: state publication locked'


@pytest.mark.parametrize('hours', [0,1,3.999,4])
def test_slot_retry_can_finalize_through_four_hours(hours):
    old=document(ready()); now=SLOT+timedelta(hours=hours)
    new=finalize_document(old,snapshot(),now)
    check_transition(old,new,now,'slot',snapshot())


@pytest.mark.parametrize('before', [True,False])
def test_slot_retry_rejects_outside_window(before):
    old=document(ready()); now=SLOT+timedelta(seconds=-1 if before else 14401)
    new=finalize_document(old,snapshot(),now)
    with pytest.raises(ValueError,match='clock allowance'): check_transition(old,new,now,'slot',snapshot())


@pytest.mark.parametrize('at,receipt,blocked', [
    ('2026-10-27T12:00:00+00:00',None,False),
    ('2026-10-27T13:14:59+00:00',None,False),
    ('2026-10-27T13:15:00+00:00',None,True),
    ('2026-10-27T14:30:00+00:00',None,True),
    ('2026-10-27T18:29:59+00:00',None,True),
    ('2026-10-27T18:30:00+00:00',None,False),
    ('2026-10-28T13:15:00+00:00',None,False),
    ('2026-10-30T13:15:00+00:00',None,True),
    ('2026-10-27T15:00:00+00:00','success',False),
    ('2026-10-27T15:00:00+00:00','defer',False),
    ('2026-10-27T15:00:00+00:00','failure',True),
    ('2026-10-27T15:00:00+00:00','wrong-slot',True),
])
def test_auto_patch_slot_gate(tmp_path,at,receipt,blocked):
    script=tmp_path/'engbatch-auto.sh'
    script.write_text((ROOT/'tests/fixtures/engbatch-auto.pre-engslots.sh').read_text())
    subprocess.run(['patch','--silent',str(script),str(ROOT/'tools/engslots/engbatch-auto.patch')],check=True)
    job=tmp_path/'job'; hard=job/'engbox/hardening'; hard.mkdir(parents=True)
    source=script.read_text().replace('J=/home/flori/jobs/sn92-fast-launch-20261015;', 'J='+str(job)+';')
    source=source.replace('now=datetime.now(timezone.utc)', 'now=datetime.fromisoformat('+repr(at)+')')
    script.write_text(source)
    if receipt:
        dest=hard/'batches/engslot-test'; dest.mkdir(parents=True)
        r=dict(slot_utc=SLOT.isoformat(),published=receipt=='success',deferred=receipt=='defer',exit_code=75 if receipt=='defer' else 1)
        if receipt=='wrong-slot': r.update(slot_utc='2026-10-23T14:30:00+00:00',published=True)
        (dest/'receipt.json').write_text(json.dumps(r))
    bindir=tmp_path/'bin'; bindir.mkdir()
    executable(bindir/'pgrep','#!/bin/bash\nexit 1\n')
    executable(bindir/'timeout','#!/bin/bash\nexit 99\n')
    config=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],CHAIN_PY=sys.executable,CHAIN_SRC=str(ROOT/'src'))
    # daily exercises the exact same gate; timeout stub prevents any external command.
    result=subprocess.run(['bash',str(script),'daily'],env=config,capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    log=(job/'watch/engbatch/auto.log').read_text()
    assert ('deferred: engine slot' in log) is blocked
    assert ('chain list failed' in log) is (not blocked)


@pytest.mark.parametrize('succeed', [True,False])
def test_retry_runner_is_bounded_and_retries(tmp_path,succeed):
    hard=tmp_path/'hard'; hard.mkdir(); bindir=tmp_path/'bin'; bindir.mkdir()
    start=int(SLOT.timestamp()); clock=tmp_path/'clock'; clock.write_text(str(start))
    # Freeze only the calendar lookup. All receipt Python still runs normally.
    py=executable(tmp_path/'clock-python', '#!'+sys.executable+'\nimport sys,subprocess\ncode=sys.stdin.read().replace("datetime.now(timezone.utc)","datetime.fromisoformat('+repr(SLOT.isoformat())+')")\nraise SystemExit(subprocess.run(['+repr(sys.executable)+',"-I","-",*sys.argv[3:]],input=code,text=True).returncode)\n')
    candidate=tmp_path/'candidate-batch.sh'
    candidate.write_text('H='+str(hard)+'\nCHAIN_PY='+str(py)+'\n')
    executable(bindir/'date', '#!/bin/bash\nif [ "$*" = "-u +%s" ]; then cat '+str(clock)+'; else exec /bin/date "$@"; fi\n')
    executable(bindir/'sleep', '#!'+sys.executable+'\nfrom pathlib import Path\nimport sys\np=Path('+repr(str(clock))+'); p.write_text(str(int(p.read_text())+int(sys.argv[1])))\n')
    counter=tmp_path/'attempts'
    executable(tmp_path/'engslot-finalize.sh','#!/bin/bash\nn=$(cat '+str(counter)+' 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > '+str(counter)+'\n'+('if [ "$n" = 2 ]; then exit 0; fi\n' if succeed else '')+'exit 75\n')
    config=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],CANDIDATE=str(candidate),ENGSLOT_SRC=str(ROOT/'src'))
    result=subprocess.run([str(ROOT/'tools/engslots/engslot-retry.sh')],env=config,capture_output=True,text=True,timeout=30)
    if succeed:
        assert result.returncode==0,result.stderr
        assert counter.read_text().strip()=='2'
    else:
        assert result.returncode==75,result.stderr
        assert counter.read_text().strip()=='48'
        r=json.loads(next(hard.glob('batches/*/receipt.json')).read_text())
        assert not r['deferred'] and not r['published'] and 'deadline expired' in r['reason']


def test_timer_install_artifacts():
    service=(ROOT/'tools/engslots/sn92fl-engslot.service').read_text()
    timer=(ROOT/'tools/engslots/sn92fl-engslot.timer').read_text()
    assert 'engslot-retry.sh' in service and 'TimeoutStartSec=4h5min' in service
    assert 'OnCalendar=Tue,Fri *-*-* 14:30:00 UTC' in timer and 'Persistent=true' in timer


def test_rollback_dry_run_last_line_is_publication_command(tmp_path):
    path=tmp_path/'candidate with spaces.sh'
    result=subprocess.run([str(ROOT/'tools/engslots/rollback.sh')],env=dict(os.environ,DRY_RUN='1',CANDIDATE=str(path)),capture_output=True,text=True)
    assert result.returncode==0
    assert result.stdout.splitlines()[-1] == 'DRY_RUN: CANDIDATE_BATCH_LIBRARY=1 source '+str(path).replace(' ','\\ ')+' library; LIVE=1; publish_state'


def test_rollback_failed_publication_does_not_enable_old_script(tmp_path):
    hard,seed,script,config=patched_candidate(tmp_path)
    before=script.read_bytes()
    backup=tmp_path/'pre.sh'; backup.write_text((ROOT/'tests/fixtures/candidate-batch.pre-engslots.sh').read_text())
    tar=tmp_path/'old.tar'; tar.touch(); (hard/'fail-publish').touch()
    config.update(CANDIDATE_BACKUP=str(backup),PREVIOUS_WORKER_TAR=str(tar),PREVIOUS_WORKER_IMAGE='worker:post4',
                  PREVIOUS_WORKER_ID='sha256:'+'9'*64,H=str(hard),SEED=str(seed))
    result=subprocess.run([str(ROOT/'tools/engslots/rollback.sh')],env=config,capture_output=True,text=True)
    assert result.returncode!=0
    assert script.read_bytes()==before and not (hard/'published.json').exists()
    assert json.loads(seed.read_text())['tracks'][P.name]['challenger'] is None
