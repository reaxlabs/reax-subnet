"""Regression drills for critic forge.py/pay.py/compat.py cases; local state and stub publication only."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_engslot import D, E, P, SUB, BEFORE, SLOT, ready, document, snapshot, slots
from reax_subnet.engslot import check_transition, finalize_document, main, weights_document
from reax_subnet.engcomp import EvalResult, TrackState, update_champion, finalize_engine_slot
from reax_subnet.engreview import (MAX_CHECKED_FILE_BYTES as M, REFERENCE_TOKENIZER_PATH,
    REFERENCE_TOKENIZER_SIZE, REFERENCE_TOKENIZER_SHA256, weight_file_screen)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['final', 'slot'])
@pytest.mark.parametrize('mutation', ['weights_add', 'weights_remove', 'slot_results', 'identity', 'measurements'])
def test_forge_cases_rejected(mode, mutation):
    old = document(ready())
    old['tracks'][P.name]['weights_ok'] = [D]
    new = deepcopy(old) if mode == 'final' else finalize_document(old, snapshot(), SLOT)
    st = new['tracks'][P.name]
    if mutation == 'weights_add': st['weights_ok'].append(E)
    elif mutation == 'weights_remove': st.pop('weights_ok')  # old image silently drops it
    elif mutation == 'slot_results': st.setdefault('slot_results', []).append(dict(event='crowned', slot_utc='future'))
    elif mutation == 'identity':
        st['challenger'] = deepcopy(old['tracks'][P.name]['challenger']); st['challenger']['digest'] = E
    else:
        st['challenger'] = deepcopy(old['tracks'][P.name]['challenger']); st['challenger']['measurements'][0][1] = 9
    with pytest.raises(ValueError): check_transition(old, new, SLOT, mode, snapshot())


def test_confirmed_pending_can_only_append_batch_result():
    old = document(ready()); new = deepcopy(old)
    st = new['tracks'][P.name]
    r = EvalResult(D, P.name, True, score=2.1, block=120, session='later')
    st['results'][D] = r.to_json()
    st['challenger']['measurements'].append([120, 2.1, 'later'])
    check_transition(old, new, BEFORE)
    st['challenger']['measurements'][-1][1] = 8
    with pytest.raises(ValueError): check_transition(old, new, BEFORE)


def test_new_confirmation_replayed_from_batch_results():
    old = document(ready()); old['tracks'][P.name]['challenger'] = None
    st = TrackState.from_json(old['tracks'][P.name])
    for i in range(P.confirmations):
        r = EvalResult(D, P.name, True, score=2, block=200+i, session=f'new{i}')
        st.results[D] = r
        st.add_cycle_event(dict(event='measurement', digest=D, block=r.block, score=r.score, session=r.session, ok=True))
        update_champion(st, [SUB], P, r.block, now=BEFORE)
    new = deepcopy(old); new['tracks'][P.name] = st.to_json()
    check_transition(old, new, BEFORE)
    new['tracks'][P.name]['challenger']['confirmed_at'] = (SLOT+timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError): check_transition(old, new, BEFORE)


def test_lazy_import_needs_no_chain_dependencies():
    result = subprocess.run(['/usr/bin/python3', '-I', '-c',
        'import sys; sys.path.insert(0,sys.argv[1]); import reax_subnet.engslot; '
        'assert "reax_subnet.engvalidator" not in sys.modules', str(ROOT/'src')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def review_inputs(tmp_path):
    review = tmp_path/'review'; (review/'REVIEWED').mkdir(parents=True)
    inventory = tmp_path/'inventory.json'; inventory.write_text(json.dumps([dict(path='/app/runtime.py', size=20)]))
    return review, inventory


def test_compliance_cli_requires_marker_screen_and_logs(tmp_path, capsys):
    review, inv = review_inputs(tmp_path); old = document(ready())
    with pytest.raises(ValueError, match='REVIEWED'): weights_document(old, 'add', D, review, inv)
    (review/'REVIEWED'/D).touch()
    inv.write_text(json.dumps([dict(path='/hidden/x.pt', size=1)]))
    with pytest.raises(ValueError, match='screen'): weights_document(old, 'add', D, review, inv)
    inv.write_text('[]')
    state = tmp_path/'seed.json'; state.write_text(json.dumps(old)); out = tmp_path/'out.json'
    main(['weights-ok','add','--state',str(state),'--digest',D,'--review-dir',str(review),'--inventory',str(inv),'--out',str(out)])
    event = json.loads(capsys.readouterr().out)
    assert event['who'] and event['utc'] and event['inventory_sha256'] == hashlib.sha256(inv.read_bytes()).hexdigest()
    new = json.loads(out.read_text()); check_transition(old, new, BEFORE, 'weights')
    assert json.loads(state.read_text()) == old
    main(['weights-ok','list','--state',str(out)])
    assert json.loads(capsys.readouterr().out)[P.name] == [D]
    removed, _ = weights_document(new, 'remove', D, review, inv)
    check_transition(new, removed, BEFORE, 'weights')
    assert removed['tracks'][P.name]['weights_ok'] == []
    historic = deepcopy(old); historic['tracks'][P.name]['history'] = [{'event':'seed_event'}]
    updated, _ = weights_document(historic, 'add', D, review, inv)
    updated['tracks'][P.name]['history'][0] = {'event':'forged'}
    with pytest.raises(ValueError): check_transition(historic, updated, BEFORE, 'weights')
    bad = deepcopy(new); bad['saved_at'] += 1
    with pytest.raises(ValueError): check_transition(old, bad, BEFORE, 'weights')


def patched_candidate(tmp_path):
    hard = tmp_path/'hard'; (hard/'state/accepted').mkdir(parents=True)
    (hard/'state/review/REVIEWED').mkdir(parents=True)
    (hard/'state/review/REVIEWED'/D).touch()
    seed = hard/'state/accepted/engine-competition.json'; seed.write_text(json.dumps(document(ready())))
    script = tmp_path/'candidate-batch.sh'
    script.write_text((ROOT/'tests/fixtures/candidate-batch.pre-engslots.sh').read_text())
    subprocess.run(['patch','--silent',str(script),str(ROOT/'tools/engslots/candidate-batch.patch')],check=True)
    source = script.read_text().replace('H=/home/flori/jobs/$JOB/engbox/hardening', 'H=$TEST_H')
    source = source.replace('# Source only the definitions',
        'publish_state() { [ ! -e "$H/fail-publish" ] || return 1; cp "$SEED" "$H/published.json"; }\n# Source only the definitions')
    script.write_text(source)
    config = dict(os.environ, DRY_RUN='0', ENGSLOT_SRC=str(ROOT/'src'), CHAIN_SRC=str(ROOT/'src'),
                  CHAIN_PY=sys.executable, CANDIDATE=str(script), TEST_H=str(hard))
    return hard, seed, script, config


def test_weights_wrapper_drill_add_seed_batch_publish_and_old_image_rejection(tmp_path):
    hard, seed, script, config = patched_candidate(tmp_path)
    inv = tmp_path/'inventory'; inv.write_text('[]')
    config.update(WEIGHTS_ACTION='add', WEIGHTS_DIGEST=D, WEIGHTS_INVENTORY=str(inv))
    # A failed publication keeps accepted compliance + a failure receipt; retry publishes.
    (hard/'fail-publish').touch()
    first = subprocess.run([str(ROOT/'tools/engweights-ok.sh')],env=config,capture_output=True,text=True)
    assert first.returncode != 0, first.stdout
    assert D in json.loads(seed.read_text())['tracks'][P.name]['weights_ok']
    (hard/'fail-publish').unlink()
    second = subprocess.run([str(ROOT/'tools/engweights-ok.sh')],env=config,capture_output=True,text=True)
    assert second.returncode == 0, second.stderr+second.stdout
    receipts = [json.loads(p.read_text()) for p in (hard/'batches').glob('*/receipt.json')]
    assert len(receipts) == 2 and any(r['published'] for r in receipts) and any(not r['published'] for r in receipts)
    assert json.loads((hard/'published.json').read_text()) == json.loads(seed.read_text())
    accepted = json.loads(seed.read_text())
    # Simulated worker round-trip preserves approvals; real patched shell accepts, then publication uses same path.
    batch = deepcopy(accepted); batch['saved_at'] = int(__import__('time').time())
    target = tmp_path/'batch.json'; target.write_text(json.dumps(batch))
    run = subprocess.run(['bash','-c', '''CANDIDATE_BATCH_LIBRARY=1 source "$CANDIDATE" library
B="$TEST_H/batches/drill"; mkdir -p "$B"; date -u +%FT%TZ > "$B/created_at"
check_state "$1" final && cp "$1" "$SEED" && publish_state
''','drill',str(target)],env=config,capture_output=True,text=True)
    assert run.returncode == 0, run.stderr+run.stdout
    assert D in json.loads((hard/'published.json').read_text())['tracks'][P.name]['weights_ok']
    old_image = deepcopy(batch); old_image['tracks'][P.name].pop('weights_ok')
    with pytest.raises(ValueError, match='weights_ok'): check_transition(batch, old_image, BEFORE)


def test_preflight_failure_has_receipt(tmp_path):
    hard, seed, script, config = patched_candidate(tmp_path)
    config['ENGSLOT_SRC'] = str(tmp_path/'missing')
    # Block fallback to an installed editable package as well.
    fake = tmp_path/'bad-python'; fake.write_text('#!/bin/bash\nif [[ "$*" == *"import reax_subnet.engslot"* ]]; then exit 1; fi\nexec '+sys.executable+' "$@"\n'); fake.chmod(0o755)
    config['CHAIN_PY'] = str(fake)
    result = subprocess.run([str(ROOT/'tools/engslot-finalize.sh')],env=config,capture_output=True,text=True)
    assert result.returncode != 0
    receipt = json.loads(next((hard/'batches').glob('*/receipt.json')).read_text())
    assert receipt['exit_code'] != 0 and not receipt['published']


def test_rollback_strips_pending_and_post4_cannot_crown(tmp_path):
    hard, seed, script, config = patched_candidate(tmp_path)
    backup = tmp_path/'pre-post6.sh'; backup.write_text((ROOT/'tests/fixtures/candidate-batch.pre-engslots.sh').read_text())
    tar = tmp_path/'old-worker.tar'; tar.write_bytes(b'local test tar placeholder')
    old_id = 'sha256:'+'9'*64
    config.update(CANDIDATE_BACKUP=str(backup),PREVIOUS_WORKER_TAR=str(tar),PREVIOUS_WORKER_IMAGE='worker:post4',
                  PREVIOUS_WORKER_ID=old_id,H=str(hard),SEED=str(seed))
    result = subprocess.run([str(ROOT/'tools/engslots/rollback.sh')],env=config,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr+result.stdout
    assert json.loads((hard/'published.json').read_text()) == json.loads(seed.read_text())
    stripped = json.loads(seed.read_text())['tracks'][P.name]
    assert stripped['challenger'] is None and stripped['pending'] is None
    assert stripped['history'][-1]['event'] == 'pending_cleared_rollback'
    assert 'from reax_subnet.engslot import check_transition' not in script.read_text()
    assert old_id in script.read_text() and str(tar) in script.read_text()
    # Publication happens before enabling old code; the library guard supports repeat rollback.
    script.write_text(script.read_text().replace('if [ "${CANDIDATE_BATCH_LIBRARY:-0}" = 1 ]; then return 0; fi', 'publish_state() { cp "$SEED" "$H/published.json"; }\nif [ "${CANDIDATE_BATCH_LIBRARY:-0}" = 1 ]; then return 0; fi'))
    second = subprocess.run([str(ROOT/'tools/engslots/rollback.sh')],env=config,capture_output=True,text=True)
    assert second.returncode == 0, second.stderr+second.stdout
    # Same read-only post4 code used by critic pay.py. One old-code update can only start a NEW confirmation.
    old_source = Path('/home/flori/jobs/sn92-fast-launch-20261015/workers/rc11p4f/wt/src/reax_subnet/engcomp.py')
    assert old_source.is_file()
    import types
    module = types.ModuleType('reax_subnet._post4_rollback_test'); sys.modules[module.__name__] = module
    try:
        exec(compile(old_source.read_text(),str(old_source),'exec'),module.__dict__)
        st = module.TrackState.from_json(stripped)
        assert module.update_champion(st,[SUB],P,1000) is None
        assert st.champion is None
    finally: sys.modules.pop(module.__name__,None)


@pytest.mark.parametrize('path', ['x.onnx','x.h5','x.tflite','x.msgpack','x.pte','x.safetensors.bak','x.pt.1'])
def test_added_weight_suffixes(path):
    assert weight_file_screen([dict(path='/app/'+path,size=1)])


def test_json_tokenizer_and_numeric_multipart_screen():
    ref = dict(path=REFERENCE_TOKENIZER_PATH,size=REFERENCE_TOKENIZER_SIZE,sha256=REFERENCE_TOKENIZER_SHA256)
    assert not weight_file_screen([ref])
    for key,value in [('path','/other/tokenizer.json'),('size',REFERENCE_TOKENIZER_SIZE*2),('sha256',E)]:
        assert weight_file_screen([{**ref,key:value}])
    assert weight_file_screen([dict(path='/app/data.json',size=M+1)])
    assert weight_file_screen([dict(path='/app/part-001',size=M//2+1),dict(path='/app/part-002',size=M//2+1,layer='different')])
    assert not weight_file_screen([dict(path='/app/part-001',size=M//2),dict(path='/app/part-002',size=M//2)])


def test_noncompliant_logged_by_update_and_finalize(caplog):
    import logging
    from reax_subnet import competition_split as cs
    caplog.set_level(logging.INFO)
    st = ready(); late = cs.ENGINE_QUANT_CONVERSION_END_UTC
    assert finalize_engine_slot(st,[SUB],P,late,1000) is None
    assert st.history[-1]['event'] == 'blocked_noncompliant'
    update_champion(st,[SUB],P,1000,now=late)
    assert st.challenger is None
    assert st.history[-1]['digest12'] == D[:12]
    assert 'blocked by weights_ok' in caplog.text


def test_finalize_document_keeps_blocked_receipt_without_previous_slot_results():
    from reax_subnet import competition_split as cs
    old = document(ready())
    assert 'slot_results' not in old['tracks'][P.name]
    new = finalize_document(old, snapshot(), cs.ENGINE_QUANT_CONVERSION_END_UTC)
    assert new['tracks'][P.name]['history'][-1]['event'] == 'blocked_noncompliant'
    assert new['tracks'][P.name]['champion'] is None
    check_transition(old, new, cs.ENGINE_QUANT_CONVERSION_END_UTC, 'slot', snapshot())


def test_slot_wrapper_counts_blocked_contender(tmp_path):
    from reax_subnet import competition_split as cs
    hard, seed, script, config = patched_candidate(tmp_path)
    (hard/'snapshot.json').write_text(json.dumps(snapshot()))
    source = script.read_text().replace('import json, os, re, sys, time, datetime',
        'import json, os, re, sys, time, datetime\ntime.time=lambda: '+str(cs.ENGINE_QUANT_CONVERSION_END_UTC.timestamp()))
    source = source.replace('# Source only the definitions',
        'sandy_chain_snapshot() { cat "$H/snapshot.json"; }\n# Source only the definitions')
    script.write_text(source)
    bindir=tmp_path/'bin'; bindir.mkdir(); clock=bindir/'date'
    clock.write_text('#!/bin/bash\ncase "$*" in\n*%FT%TZ*) echo 2026-10-30T14:30:00Z;;\n*%Y%m%dT%H%M%SZ*) echo 20261030T143000Z;;\n*) echo 20261030;;\nesac\n')
    clock.chmod(0o755)
    config['PATH']=str(bindir)+os.pathsep+os.environ['PATH']
    result=subprocess.run([str(ROOT/'tools/engslot-finalize.sh')],env=config,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr+result.stdout
    assert '1 contenders blocked by weights_ok' in result.stdout
    receipt=json.loads(next((hard/'batches').glob('*/receipt.json')).read_text())
    assert receipt['blocked_by_weights_ok'] == 1 and receipt['changed'] and receipt['published']
    assert json.loads(seed.read_text())['tracks'][P.name]['history'][-1]['event'] == 'blocked_noncompliant'
