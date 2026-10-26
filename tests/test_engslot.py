"""Fixed engine slots: pure rules, offline publication and the actual patched check_state shell."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from conftest import RELEASE_ENGINE_CROWN_SLOTS_UTC
from reax_subnet import competition_split as cs
from reax_subnet.engcomp import (Champion, EvalResult, Submission, TrackState, finalize_engine_slot,
                                 observe, update_champion)
from reax_subnet.engslot import check_transition, finalize_document, main
from reax_subnet.engvalidator import TRACKS

P = TRACKS['s1-fast']
SLOT = datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc)
BEFORE = SLOT - timedelta(hours=1)
D, E = 'a' * 64, 'b' * 64
SUB = Submission(7, 'hk', P.name, 'repo/image', D, 50)


@pytest.fixture(autouse=True)
def slots(monkeypatch):
    monkeypatch.setattr(cs, 'ENGINE_CROWN_SLOTS_UTC', RELEASE_ENGINE_CROWN_SLOTS_UTC)


def ready():
    st = TrackState(derived={D: []})
    observe(st, [SUB], 100)
    for i in range(P.confirmations):
        st.results[D] = EvalResult(D, P.name, True, score=2, block=100+i, session=f's{i}')
        assert update_champion(st, [SUB], P, 100+i, now=BEFORE) is None
    return st


def document(st):
    return {'version': 1, 'saved_at': int(BEFORE.timestamp()), 'tracks': {P.name: st.to_json()},
            'binds_hotkey': False, 'thresholds_sha256': 'fixed', 'context_sha256': 'fixed', 'context': {}}


def snapshot():
    return {'block': 1000, 'rows': [{'uid': SUB.uid, 'hotkey': SUB.hotkey, 'block': SUB.block,
                                    'text': f'reax-eng/1:{P.name}:{SUB.image}@sha256:{D}'}]}


def test_calendar():
    assert RELEASE_ENGINE_CROWN_SLOTS_UTC[0] == SLOT
    assert {s.weekday() for s in RELEASE_ENGINE_CROWN_SLOTS_UTC} == {1, 4}
    assert all((s.hour, s.minute) == (14, 30) for s in RELEASE_ENGINE_CROWN_SLOTS_UTC)
    assert RELEASE_ENGINE_CROWN_SLOTS_UTC[-1].date().isoformat() == '2027-03-30'


def test_pending_crown_roundtrip_and_finalize():
    st = ready()
    c = st.challenger
    assert c['confirmed_at'] == BEFORE.isoformat()
    assert c['slot_utc'] == SLOT.isoformat()
    raw = st.to_json()
    assert raw['pending']['uid'] == 7
    assert raw['pending']['pooled_score'] == 2
    st = TrackState.from_json(json.loads(json.dumps(raw)))
    prior = deepcopy(st.to_json())
    assert finalize_engine_slot(st, [SUB], P, SLOT - timedelta(seconds=1), 1000) is None
    assert st.to_json() == prior
    ch = finalize_engine_slot(st, [SUB], P, SLOT, 1000)
    assert (ch.digest, ch.score, ch.crowned_block) == (D, 2, 1000)
    assert st.challenger is None
    assert st.history[-1]['event'] == 'crowned'
    assert st.history[-1]['slot_utc'] == SLOT.isoformat()
    assert st.slot_results[-1] == st.history[-1]
    assert finalize_engine_slot(st, [SUB], P, SLOT, 1000) is None


def test_confirmation_at_slot_and_after():
    for now, expected in [(SLOT, SLOT), (SLOT + timedelta(microseconds=1), RELEASE_ENGINE_CROWN_SLOTS_UTC[1])]:
        st = TrackState(derived={D: []})
        observe(st, [SUB], 100)
        st.results[D] = EvalResult(D, P.name, True, score=2, block=100)
        update_champion(st, [SUB], replace(P, confirmations=1), 100, now=now)
        assert st.challenger['slot_utc'] == expected.isoformat()


def test_displacement():
    st = ready()
    other = Submission(8, 'later', P.name, 'repo/other', E, 80)
    observe(st, [SUB, other], 200)
    st.derived[E] = []
    st.results[E] = EvalResult(E, P.name, True, score=2.01, block=200)
    update_champion(st, [SUB, other], P, 200, now=BEFORE)
    assert st.challenger['digest'] == D
    st.results[E].score = 2.2
    update_champion(st, [SUB, other], P, 200, now=BEFORE)
    assert st.challenger['digest'] == E
    assert 'confirmed_at' not in st.challenger


@pytest.mark.parametrize('why', ['withdrawn', 'invalid', 'bar', 'parent', 'paused', 'copy', 'owner'])
def test_invalid_pending_cannot_crown(why):
    st = ready()
    eligible = [SUB]
    if why == 'withdrawn': eligible = []
    if why == 'invalid': st.results[D].ok = False
    if why == 'bar': st.champion = Champion(E, 'other', 8, 3, 100)
    if why == 'parent':
        st.derived[D] = [E]
        st.held_scores[E] = 3
    if why == 'paused': st.copy_paused_until['hk'] = 2000
    if why == 'copy': st.copy_strikes['hk'] = [D]
    if why == 'owner': st.owners[D] = 'someone-else'
    raw = deepcopy(st.to_json())
    assert finalize_engine_slot(st, eligible, P, SLOT, 1000) is None
    assert st.to_json() == raw


@pytest.mark.parametrize('why', ['withdrawn', 'bar'])
def test_update_removes_pending(why):
    st = ready()
    eligible = [] if why == 'withdrawn' else [SUB]
    if why == 'bar': st.champion = Champion(E, 'other', 8, 3, 100); eligible += [Submission(8,'other',P.name,'repo/o',E)]
    update_champion(st, eligible, P, 200, now=BEFORE)
    assert st.challenger is None


def test_offline_cli_and_strict_transition(tmp_path):
    old = document(ready())
    snap = snapshot()
    new = finalize_document(old, snap, SLOT)
    assert new != old
    check_transition(old, new, SLOT, 'slot', snap)
    check_transition(old, new, SLOT, 'final')
    bad = deepcopy(new); bad['context'] = {'forged': True}
    with pytest.raises(ValueError): check_transition(old, bad, SLOT, 'slot', snap)
    with pytest.raises(ValueError): check_transition(old, new, SLOT+timedelta(hours=4,seconds=1), 'slot', snap)
    assert finalize_document(old, snap, BEFORE) == old
    snap['rows'] = []
    assert finalize_document(old, snap, SLOT) == old
    for name, data in [('in.json',old), ('snapshot.json',snapshot()['rows'])]:
        (tmp_path/name).write_text(json.dumps(data))
    main(['finalize','--state',str(tmp_path/'in.json'),'--commitments',str(tmp_path/'snapshot.json'),
          '--block','1000','--now',SLOT.isoformat(),'--out',str(tmp_path/'out.json')])
    assert json.loads((tmp_path/'out.json').read_text()) == new


def test_batch_rejects_off_slot_crown_and_accepts_missed_slot():
    old = document(ready())
    new = finalize_document(old, snapshot(), SLOT+timedelta(hours=2))
    check_transition(old, new, SLOT+timedelta(hours=2), 'final')
    forged = deepcopy(new)
    forged['tracks'][P.name]['history'][-1]['slot_utc'] = BEFORE.isoformat()
    with pytest.raises(ValueError): check_transition(old, forged, SLOT, 'final')
    old['tracks'][P.name]['challenger'].pop('confirmed_at')
    with pytest.raises(ValueError): check_transition(old, new, SLOT+timedelta(hours=2), 'final')


def test_real_old_state_roundtrip(monkeypatch):
    path = Path(__file__).parent/'fixtures'/'engine-accepted-pre-slots.json'
    raw = json.loads(path.read_text())
    monkeypatch.setattr(cs, 'ENGINE_CROWN_SLOTS_UTC', ())
    for track in raw['tracks'].values():
        st = TrackState.from_json(track)
        assert st.slot_results == []
        assert not (st.challenger or {}).get('confirmed_at')
        # Old optional fields may already have been normalised by the baseline reader.
        assert TrackState.from_json(st.to_json()).to_json() == st.to_json()


def test_legacy_instant_crown(monkeypatch):
    monkeypatch.setattr(cs, 'ENGINE_CROWN_SLOTS_UTC', ())
    st = TrackState(); observe(st,[SUB],100)
    st.results[D] = EvalResult(D,P.name,True,score=2,block=100)
    assert update_champion(st,[SUB],replace(P,confirmations=1),100).digest == D
    assert 'next_slot_utc' not in st.to_json()


@pytest.mark.asyncio
async def test_worker_finalizes_missed_slot_before_gpu_preflight(tmp_path):
    from reax_subnet.chain import Neuron, Snapshot
    from reax_subnet.engrunner import InfraError
    from reax_subnet.engvalidator import EngineCompetition
    class BrokenGPU:
        async def preflight(self): raise InfraError('GPU unavailable')
    comp = EngineCompetition(BrokenGPU(), {P.name:'ref'}, str(tmp_path/'state.json'),
                             utcnow=lambda:SLOT+timedelta(hours=2),start_utc=None)
    comp.states[P.name] = ready()
    n = Neuron(uid=7,hotkey='hk',coldkey='ck',validator_permit=False,stake=0,ip='0.0.0.0',port=0,commitment=f'reax-eng/1:{P.name}:{SUB.image}@sha256:{D}')
    await comp._run_cycle(Snapshot(block=1000,neurons=(n,),owner_hotkey=None),set())
    assert comp.states[P.name].champion.digest == D
    assert comp.states[P.name].history[-1]['slot_utc'] == SLOT.isoformat()


def test_slots_off_matches_baseline_byte_for_byte(monkeypatch):
    """Replay the same state transitions in the release base and candidate, serialising identically."""
    import sys
    import types
    from reax_subnet import engcomp
    monkeypatch.setattr(cs, 'ENGINE_CROWN_SLOTS_UTC', ())
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git', 'show', '7d22efb:src/reax_subnet/engcomp.py'], cwd=root, text=True)
    baseline = types.ModuleType('reax_subnet._engcomp_baseline')
    sys.modules[baseline.__name__] = baseline
    try:
        exec(compile(source, '<7d22efb-engcomp>', 'exec'), baseline.__dict__)
        a, b = baseline.TrackState(), engcomp.TrackState()
        subs = [SUB, Submission(8, 'other', P.name, 'repo/o', E, 80)]
        for mod, st in ((baseline, a), (engcomp, b)):
            mod.observe(st, subs, 100)
        for i, score in enumerate([2, 2.1, 2, 2.08, 2.2, 2.3, 1.1, 2.4]):
            dg = D if i < 3 else E
            for mod, st in ((baseline, a), (engcomp, b)):
                st.results[dg] = mod.EvalResult(dg,P.name,True,score=score,block=100+i,session=str(i))
                mod.update_champion(st, subs, P, 100+i)
            assert json.dumps(a.to_json(), sort_keys=True) == json.dumps(b.to_json(), sort_keys=True)
    finally:
        sys.modules.pop(baseline.__name__, None)


def test_patched_check_state_accepts_slot_rejects_other_mutations(tmp_path):
    root = Path(__file__).resolve().parents[1]
    reference = root/'tests/fixtures/candidate-batch.pre-engslots.sh'
    script = tmp_path/'candidate-batch.sh'
    script.write_text(reference.read_text())
    subprocess.run(['patch', '--silent', str(script), str(root/'tools/engslots/candidate-batch.patch')], check=True)
    subprocess.run(['bash', '-n', str(script)], check=True)
    source = script.read_text().split('check_state() {',1)[1].split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
    old = document(ready())
    good = finalize_document(old,snapshot(),SLOT)
    (tmp_path/'seed.json').write_text(json.dumps(old))
    (tmp_path/'chain.json').write_text(json.dumps(snapshot()))
    review = tmp_path/'review'; review.mkdir(); (review/D).touch()
    def check(new, mode='slot'):
        (tmp_path/'new.json').write_text(json.dumps(new))
        args = [str(tmp_path/'new.json'),mode,str(tmp_path/'seed.json'),str(review),BEFORE.isoformat(),
                str(64*1024*1024),'100','1','','',str(root/'src'),str(tmp_path/'chain.json')]
        # Pin only the system clock used by the existing batch-window guard; all slot code uses release constants.
        return subprocess.run([__import__('sys').executable,'-I','-',*args],input='import time\ntime.time = lambda: '+str(SLOT.timestamp())+'\n'+source,
                              text=True,capture_output=True)
    got = check(good)
    assert got.returncode == 0, got.stderr+got.stdout
    assert 'check_state: ok slot' in got.stdout
    for key, value in [('results', {}),('first_seen',{}),('champion_invalid',2)]:
        bad = deepcopy(good); bad['tracks'][P.name][key] = value
        got = check(bad)
        assert got.returncode != 0, (key,got.stdout)
    bad = deepcopy(good); bad['tracks'][P.name]['history'][-1].pop('slot_utc')
    assert check(bad,'final').returncode != 0
    # Exact slot match still requires the existing Sandy review marker.
    (review/D).unlink()
    assert check(good).returncode != 0


def test_next_slot_does_not_repeat_consumed_slot(tmp_path):
    from reax_subnet.engcomp import next_engine_slot
    st = ready()
    finalize_engine_slot(st,[SUB],P,SLOT,1000)
    assert next_engine_slot(SLOT, st) == RELEASE_ENGINE_CROWN_SLOTS_UTC[1].isoformat()
    # Even a restored challenger cannot crown a second time in a consumed slot.
    st.challenger = ready().challenger
    assert finalize_engine_slot(st,[SUB],P,SLOT,1000) is None


def test_wrapper_dry_run_and_retry_after_publication_failure(tmp_path):
    """Execute the real wrapper and patched check_state; stub only chain read, clock and remote publication."""
    import os
    import sys
    root = Path(__file__).resolve().parents[1]
    wrapper = root/'tools/engslot-finalize.sh'
    out = subprocess.run([str(wrapper)],capture_output=True,text=True,check=True)
    assert out.stdout.startswith('DRY_RUN:')
    hard = tmp_path/'hard'; (hard/'state/accepted').mkdir(parents=True)
    (hard/'state/review/REVIEWED').mkdir(parents=True)
    (hard/'state/review/REVIEWED'/D).touch()
    seed = hard/'state/accepted/engine-competition.json'; seed.write_text(json.dumps(document(ready())))
    (hard/'snapshot.json').write_text(json.dumps(snapshot()))
    script = tmp_path/'candidate-batch.sh'
    script.write_text((root/'tests/fixtures/candidate-batch.pre-engslots.sh').read_text())
    subprocess.run(['patch','--silent',str(script),str(root/'tools/engslots/candidate-batch.patch')],check=True)
    s = script.read_text().replace('H=/home/flori/jobs/$JOB/engbox/hardening', 'H=$TEST_H')
    s = s.replace('import json, os, re, sys, time, datetime',
                  'import json, os, re, sys, time, datetime\ntime.time = lambda: '+str(SLOT.timestamp()))
    stubs = '''sandy_chain_snapshot() { cat "$H/snapshot.json"; }
publish_state() { [ ! -e "$H/fail-publish" ] || return 1; cp "$SEED" "$H/published.json"; }
'''
    s = s.replace('# Source only the definitions', stubs+'\n# Source only the definitions')
    script.write_text(s)
    bindir = tmp_path/'bin'; bindir.mkdir()
    clock = bindir/'date'
    clock.write_text('#!/bin/bash\ncase "$*" in\n*%FT%TZ*) echo 2026-10-27T14:30:00Z;;\n*%Y%m%dT%H%M%SZ*) echo 20261027T143000Z;;\n*) echo 20261027;;\nesac\n')
    clock.chmod(0o755)
    config = dict(os.environ, DRY_RUN='0', ENGSLOT_SRC=str(root/'src'), CANDIDATE=str(script), TEST_H=str(hard),
                  CHAIN_PY=sys.executable, PATH=str(bindir)+os.pathsep+os.environ['PATH'])
    (hard/'fail-publish').touch()
    first = subprocess.run([str(wrapper)],env=config,capture_output=True,text=True)
    assert first.returncode != 0
    receipts = [json.loads(p.read_text()) for p in (hard/'batches').glob('*/receipt.json')]
    assert len(receipts) == 1 and receipts[0]['changed'] and not receipts[0]['published'], first.stderr
    assert json.loads(seed.read_text())['tracks'][P.name]['champion']['digest'] == D
    (hard/'fail-publish').unlink()
    second = subprocess.run([str(wrapper)],env=config,capture_output=True,text=True)
    assert second.returncode == 0, second.stderr+second.stdout
    receipts = [json.loads(p.read_text()) for p in (hard/'batches').glob('*/receipt.json')]
    assert len(receipts) == 2 and any(r['published'] and not r['changed'] for r in receipts)
    assert json.loads((hard/'published.json').read_text()) == json.loads(seed.read_text())


def test_batch_crown_needs_seed_pending_even_at_slot():
    old = document(ready())
    new = finalize_document(old,snapshot(),SLOT)
    old['tracks'][P.name]['challenger'] = None
    with pytest.raises(ValueError): check_transition(old,new,SLOT,'final')


def test_batch_can_lower_new_champions_held_score():
    old = document(ready())
    new = finalize_document(old,snapshot(),SLOT)
    new['tracks'][P.name]['champion']['score'] = 1.9
    check_transition(old,new,SLOT+timedelta(hours=1),'final')
    new['tracks'][P.name]['champion']['score'] = 2.1
    with pytest.raises(ValueError): check_transition(old,new,SLOT+timedelta(hours=1),'final')
