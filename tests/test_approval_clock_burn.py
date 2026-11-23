"""Approval clock cutoff, source budgets and unchanged journal semantics."""
import hashlib
import json
import math
from types import SimpleNamespace

import pytest
from Crypto.PublicKey import ECC

from reax_subnet import award, competition_split as cs, owner_alloc as oa
from test_pause_runout import BASE, DAY, leaders, weights
from test_owner_alloc import HK, N, PLAIN, FULL, mix, send, journal, write, row

CROWN = cs.RUNOUT_APPROVAL_CLOCK_FROM_UTC.timestamp()


@pytest.fixture(autouse=True)
def active(monkeypatch):
    monkeypatch.setattr(cs, 'NEED_RULE', True)
    monkeypatch.setattr(oa, '_LATCHED', {})


@pytest.mark.parametrize('uid', [83, 114, 240])
def test_old_crowns_ignore_late_approval_byte_exact(uid):
    crown = CROWN - uid * 3600
    approval = crown + 25 * 3600
    for now in (crown, crown+3*DAY, crown+5*DAY, crown+7*DAY, crown+9*DAY):
        old = max(0., min(1., (crown + 7*DAY - now)/(4*DAY)))
        assert repr(cs.runout_factor(crown, crown, now, approval)) == repr(old)
        tr = cs.RankedLeaders([(uid, 50.)]); tr.crowned = {uid: crown}
        kwargs = dict(tr=tr, pause_starts={'generalist': crown})
        assert repr(weights(now, **kwargs)) == repr(weights(now, **kwargs,
                      award_approved_at=lambda *a: approval))


@pytest.mark.parametrize('delta', [0, -25*3600, 25*3600])
def test_new_clock(delta):
    approved = CROWN + delta
    start = max(CROWN, approved)
    for days, factor in [(0, 1.), (3, 1.), (5, .5), (7, 0.), (9, 0.)]:
        assert cs.runout_factor(CROWN, CROWN, start+days*DAY, approved) == factor


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -float('inf')])
def test_unknown_approval_preserves_full_and_gate_still_controls_pay(value):
    assert cs.runout_factor(CROWN, CROWN, CROWN+99*DAY, value) == 1.
    w = weights(CROWN+9*DAY, leaders({20: CROWN}), award_ok=lambda *a: False,
                award_approved_at=lambda *a: value, pause_starts={'generalist': CROWN})
    assert w.get(20, 0.) < .1  # at most the unchanged waiting keep-alive, never title pay


def test_cochampions_own_approvals_and_engine_clock():
    tr = leaders({20: CROWN, 24: CROWN}); tr.cochamps = {20: .5, 24: .5}
    approvals = {20: CROWN, 24: CROWN+25*3600}
    now = CROWN+5*DAY
    w = weights(now, tr, award_ok=lambda *a: True,
                award_approved_at=lambda t,u: approvals[u], pause_starts={'generalist': CROWN})
    ka = .075/4
    assert w[20] == pytest.approx(max(ka, .225*.5*.5))
    assert w[24] == pytest.approx(max(ka, .225*.5*cs.runout_factor(CROWN,CROWN,now,approvals[24])))
    eng = weights(now, paused=['engine'], pause_starts={'engine': CROWN},
                  engine_runout={30: (1., CROWN)}, engine_approved_at={30: CROWN+25*3600})
    assert eng[30] == pytest.approx(.2*cs.runout_factor(CROWN,CROWN,now,CROWN+25*3600))


def test_signed_timestamp_is_same_record_as_award_gate(tmp_path):
    key = ECC.generate(curve='ed25519')
    payload = dict(track_id='generalist', hotkey=HK, submission_id='repo/model@'+'a'*40,
                   approved_at=int(CROWN+25*3600), expires_at=None)
    bad = award.sign(key, payload); bad['payload']['approved_at'] += 100
    good = award.sign(key, payload)
    expired = award.sign(key, dict(payload, approved_at=int(CROWN), expires_at=int(CROWN+DAY)))
    path = tmp_path/'approvals.json'
    path.write_text(json.dumps(dict(version=1,approvals=[bad,expired,good])))
    gate = award.AwardGate(str(path), award.public_hex(key), wall=lambda: CROWN+2*DAY)
    ident = ('generalist', HK, payload['submission_id'])
    assert gate.approved(*ident) and gate.approved_at(*ident) == payload['approved_at']
    for ident2 in [('engine',HK,ident[2]),('generalist','other',ident[2]),('generalist',HK,'other')]:
        assert not gate.approved(*ident2) and gate.approved_at(*ident2) is None
    gate._wall = lambda: CROWN-DAY
    assert not gate.approved(*ident) and gate.approved_at(*ident) is None


def burn_plan(tmp_path, share=.07, **kw):
    r = dict(row(), source='burn', share=share, **kw)
    return oa.Plan(write(tmp_path, [r]), 100, N), r


@pytest.mark.parametrize('burn', [0., .1])
def test_burn_preserves_all_other_uids_and_inference(tmp_path, burn):
    plan,_ = burn_plan(tmp_path)
    kw = dict(serving={10:.3,11:.3,12:.2,13:.1,3:burn},burn_fraction=burn,
              provisional={'generalist':[42]}, legacy_keepalive=[41],engine_waiting=[43])
    before = mix(**kw); after = mix(**kw,owner_allocation=plan.shares)
    for u in set(before)-{3,114}:
        assert abs(before[u]-after[u]) <= 1e-12
    assert abs((before[3]-after[3])-(1-burn)*.07) <= 1e-12
    assert after[114] == pytest.approx((1-burn)*.07)
    assert min(after.values()) >= 0 and abs(math.fsum(after.values())-1) < 1e-12


def test_burn_scales_only_residual_and_mixed_sources(tmp_path):
    a=dict(row(), source='burn',share=.10)
    b=dict(row(), source='inference',share=.04,reason='separate')
    plan=oa.Plan(write(tmp_path,[a,b]),100,N)
    before=mix(inference=.85)
    inf=mix(inference=.85,owner_allocation={114:.04})
    after=mix(inference=.85,owner_allocation=plan.shares)
    for u in set(before)-{3,114}:
        assert abs(after[u]-inf[u]) <= 1e-12
    assert after.get(3,0.) >= 0 and after.get(3,0.) < 1e-12
    assert after[114] == pytest.approx(.04+before.get(3,0.))


def test_old_hash_and_strict_source(tmp_path):
    old=row()
    assert oa.row_id(old)==hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert oa.parse([old])==[old] and 'source' not in old
    for source in ('Burn',None,[],{},1,True,''):
        assert not oa.Plan(write(tmp_path,[dict(old,source=source)]),100,N).shares


@pytest.mark.parametrize('source', ['burn','inference'])
def test_24_set_journal_restart_remove_readd_pending_reconcile(tmp_path, source):
    r=dict(row(),source=source,sets=24)
    path=write(tmp_path,[r])
    for i in range(24):
        plan=oa.Plan(path,100+i,N)
        assert plan.shares
        send(plan)
    assert journal(tmp_path)['counts'][oa.row_id(r)] == 24
    write(tmp_path,[])
    assert not oa.Plan(path,200,N).shares
    write(tmp_path,[r])
    assert not oa.Plan(path,201,N).shares
    r=dict(r,reason='new pending award')
    write(tmp_path,[r]); plan=oa.Plan(path,202,N); plan.attach(FULL,PLAIN)
    assert plan.begin(FULL)[1] is plan
    assert not oa.Plan(path,203,N).shares
    assert journal(tmp_path)['needs_reconcile']
    assert not oa.Plan(path,204,N).shares


def test_burn_dry_run_and_unwritable_fallback(tmp_path, monkeypatch):
    plan,_=burn_plan(tmp_path)
    plain=mix(); full=mix(owner_allocation=plan.shares); plan.attach(full,plain)
    assert not (tmp_path/oa.STATE_NAME).exists()
    monkeypatch.setattr(oa,'_write',lambda *a: (_ for _ in ()).throw(OSError('read only')))
    assert plan.begin(full)==(plain,None)
    assert not oa.Plan(plan.path.replace(oa.STATE_NAME,oa.FILE_NAME),101,N).shares


def test_new_engine_remote_path_and_notice_start(tmp_path, monkeypatch, capsys):
    from datetime import datetime, timezone
    from test_remote_competitions import Clock, SERVING, snap, stamp, write_engine_state
    from reax_subnet import remote, needctl
    from reax_subnet.config import ENGINE_STATE_FILE
    crown = datetime.fromtimestamp(CROWN, timezone.utc)
    approval = CROWN+25*3600
    now = datetime.fromtimestamp(approval+5*DAY, timezone.utc)
    path = tmp_path/ENGINE_STATE_FILE
    write_engine_state(path)
    raw=json.loads(path.read_text()); eng=raw['tracks']['s1-fast']
    eng['history']=[dict(event='crowned',digest=eng['champion']['digest'],hotkey='hk-eng',slot_utc=crown.isoformat())]
    path.write_text(json.dumps(raw)); stamp(path,now.timestamp())
    (tmp_path/'need.json').write_text(json.dumps({'pauses':[dict(track='engine',from_utc=crown.isoformat(),until_utc=None,reason='test')]}))
    identities=[]
    def approved_at(*ident):
        identities.append(ident)
        return approval
    gate=SimpleNamespace(path=str(tmp_path/'approvals.json'),approved=lambda *a:True,approved_at=approved_at)
    comp=remote.RemoteEngineCompetition(str(path),utcnow=lambda:now,wall=Clock(now.timestamp()),award_gate=gate)
    w=comp.weights(SERVING,snap(),{0},0.)
    assert w[3] == pytest.approx(.1) and w[11]+w[12] == pytest.approx(.4)
    assert ('s1-fast','hk-eng','sha256:'+eng['champion']['digest']) in identities
    assert needctl.holder_times('engine',[str(path)],gate)==[approval]
    assert needctl.holder_times('engine',[str(path)]) == []
    assert 'WARNING: skipping title holder' in capsys.readouterr().err
    assert needctl.holder_times('engine',[str(path)],SimpleNamespace(approved_at=lambda *a: None)) == []
    assert 'no verified approval time' in capsys.readouterr().err
    from reax_subnet.award import AwardGate
    monkeypatch.delenv('REAX_AWARD_APPROVALS', raising=False)
    assert needctl.holder_times('engine',[str(path)],AwardGate.from_env({})) == []
    assert 'no verified approval time' in capsys.readouterr().err
    # Crown+7d is only approval+5d23h, so no old-title notice is required yet.
    from datetime import timedelta
    needctl.schedule(tmp_path/'schedule.json','engine',crown+timedelta(days=7),None,'test',[approval],None)
    with pytest.raises(ValueError,match='24 hours notice'):
        needctl.schedule(tmp_path/'schedule.json','engine',datetime.fromtimestamp(approval+7*DAY,timezone.utc),None,'test',[approval],None)


def test_burn_remote_fallback_has_no_plan(tmp_path, monkeypatch):
    from reax_subnet.remote import RemoteEngineCompetition
    from reax_subnet.engvalidator import EngineCompetition
    eng=RemoteEngineCompetition(str(tmp_path/'remote.json'))
    monkeypatch.setattr(eng,'engine_active',lambda:False)
    def fails(self,*a,**kw):
        self._owner_alloc_plan=burn_plan(tmp_path)[0]
        raise RuntimeError('payment code')
    monkeypatch.setattr(EngineCompetition,'weights',fails)
    assert eng.weights({3:1.},SimpleNamespace(owner_uid=3),set())
    assert eng.allocation_plan() is None


def test_burn_zero_room_rows_do_not_consume_mixed_plan(tmp_path):
    rows=[dict(row(),source='burn',reason='burn'),dict(row(),source='inference',reason='inference')]
    plan=oa.Plan(write(tmp_path,rows),100,N)
    oa.take(plan.shares,.4)
    oa.take(plan.shares,0.,source='burn')
    plan.attach(FULL,PLAIN)
    plan.begin(FULL)[1].finish(True)
    assert oa.row_id(rows[0]) not in journal(tmp_path)['counts']
    assert journal(tmp_path)['counts'][oa.row_id(rows[1])]==1


def test_burn_attach_refuses_changed_protected_vector(tmp_path):
    plan,_=burn_plan(tmp_path)
    plain=mix()
    full=mix(owner_allocation=plan.shares)
    plan.owner_uid=3
    changed=dict(full)
    changed[10]+=.000001
    changed[3]-=.000001
    with pytest.raises(ValueError,match='protected component'):
        plan.attach(changed,plain,plain)
    assert not (tmp_path/oa.STATE_NAME).exists()


@pytest.mark.parametrize("source", ["inference", "burn"])
@pytest.mark.parametrize("available", [0., 1.1102230246251565e-16, 1e-9, 1.01e-9])
def test_allocation_dust_does_not_consume_source(source, available):
    from reax_subnet.owner_alloc import Shares, take
    shares = Shares()
    getattr(shares, source)[100] = .07
    paid, total = take(shares, available, source)
    assert (source in shares.funded) == (available > 1e-9)
    assert bool(paid) == (available > 1e-9)
    assert total == pytest.approx(available if available > 1e-9 else 0., abs=1e-20)
