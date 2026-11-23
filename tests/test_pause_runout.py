"""Paused-track title-age run-out: real weight vectors, owner burn and keep-alive floors."""
import json
import math
from datetime import timedelta

import pytest

from reax_subnet import competition_split as cs, need

DAY = 86400.
START = cs.NEED_START_UTC.timestamp()
BASE = {3: 0., 10: .4, 11: .3, 12: .2, 13: .1}


@pytest.fixture(autouse=True)
def active(monkeypatch):
    monkeypatch.setattr(cs, 'NEED_RULE', True)


def leaders(crowns):
    result = cs.RankedLeaders([(20, 50.), (21, 45.), (22, 40.)])
    result.crowned = crowns
    return result


def weights(now, tr=None, **kw):
    args = dict(owner_uid=3, now=now, paused=['generalist'], pause_starts={'generalist': START})
    args.update(kw)
    return cs.combine_weights(BASE, 30, 1., {'generalist': tr or leaders({20: START})}, **args)


@pytest.mark.parametrize('days,factor', [(0, 1.), (3, 1.), (4, .75), (5, .5), (6, .25), (7, 0.), (8, 0.)])
def test_generalist_runout_and_former_places(days, factor):
    w = weights(START + days * DAY)
    ka = .30 * .25 / 3
    assert w[20] == pytest.approx(max(ka, .30 * .75 * factor))
    assert w[21] == pytest.approx(ka) and w[22] == pytest.approx(ka)
    assert math.fsum(w[u] for u in (10, 11, 12, 13)) == pytest.approx(.4)
    assert math.fsum(w.values()) == pytest.approx(1.)
    assert w[3] == pytest.approx(1 - .4 - .2 - w[20] - 2 * ka)


@pytest.mark.parametrize('age', [5, 7, 8])
def test_existing_crown_uses_only_title_age(age):
    crowned = START - age * DAY
    for elapsed in (0, .5, 1, 2):
        expected = max(0., (7 - age - elapsed) / 4)
        assert cs.runout_factor(crowned, START, START + elapsed * DAY) == expected
        w = weights(START + elapsed * DAY, leaders({20: crowned}))
        assert w[20] == pytest.approx(max(.025, .225 * expected))


def test_cochampions_use_their_own_crowning_times_and_do_not_inherit_a_former_share():
    tr = leaders({20: START, 24: START - 8 * DAY})
    tr.cochamps = {20: .5, 24: .5}
    tr.formsets = [{21: .5, 25: .5}]
    w = weights(START + 4 * DAY, tr)
    ka = .075 / 5
    assert w[20] == pytest.approx(.3 * .75 * .5 * .75)
    for uid in (24, 21, 22, 25):
        assert w[uid] == pytest.approx(ka)
    assert sum(w.values()) == pytest.approx(1.)


@pytest.mark.parametrize('days,factor', [(1, 1.), (4, .75), (6, .25), (8, 0.)])
def test_engine_title_only_and_keepalive(days, factor):
    w = weights(START + days * DAY, paused=['engine'], pause_starts={'engine': START},
                engine_runout={30: (1., START)}, engine_waiting=[31])
    assert w[31] == pytest.approx(.005)
    assert w[30] == pytest.approx(max(.005, .19 * factor))
    assert sum(w.values()) == pytest.approx(1.)


def test_engine_former_places_end_even_when_former_is_an_ordinary_share_payee():
    w = cs.combine_weights(
        BASE, 30, .75, {}, owner_uid=3, now=START + 4 * DAY, paused=['engine'],
        pause_starts={'engine': START}, engine_cochampions={31: .17, 32: .08}, engine_runout={30: (.75, START)})
    assert w[31] == pytest.approx(.01 / 3)
    assert w[32] == pytest.approx(.01 / 3)
    assert w[30] == pytest.approx(.2 * .75 * .75)
    assert sum(w.values()) == pytest.approx(1.)


def test_missing_pause_start_does_not_burn_and_missing_crown_does_not_decay():
    assert weights(START + 8 * DAY, pause_starts={}) == weights(START + 8 * DAY, paused=[])
    w = weights(START + 8 * DAY, leaders({}))
    assert w[20] == pytest.approx(.225)
    assert cs.runout_factor(None, START, START + 8 * DAY) == 1.
    assert cs.runout_factor(START, None, START + 8 * DAY) == 1.
    assert cs.runout_factor(float('nan'), START, START + 8 * DAY) == 1.


def test_before_activation_ignores_every_runout_input():
    assert weights(START - 1) == weights(START - 1, paused=[])


def test_operator_from_time_and_continuous_pause_union(tmp_path):
    a = cs.NEED_START_UTC
    path = tmp_path / 'need.json'
    rows = [('engine', a, a + timedelta(days=2)), ('engine', a + timedelta(days=1), None),
            ('engine', a + timedelta(days=1), a + timedelta(days=4))]
    path.write_text(json.dumps({'pauses': [dict(track=k, from_utc=t.isoformat(),
                    until_utc=u.isoformat() if u else None, reason='test') for k, t, u in rows]}))
    share, starts = need.decide_with_starts(a + timedelta(days=3), str(path), ['generalist'])
    assert share == .4 and starts == {'engine': START}
    assert need.decide(a + timedelta(days=3), str(path), ['generalist']) == (.4, ['engine'])
    assert need.decide_with_starts(a - timedelta(seconds=1), str(path), ['generalist']) == (None, {})
    path.write_text('{"pauses": [{"track": "engine"}]}')
    assert need.decide_with_starts(a + timedelta(days=3), str(path), []) == (share, starts)


def test_evaluator_exposes_every_cochampion_time_and_copy_keeps_title_metadata(monkeypatch):
    from test_cochampions import case, key_of, UIDS
    from test_opening_playoff_slots import TRACK, view
    from reax_subnet.spec.evaluator import compute_leaders
    from reax_subnet.competitions import _copy
    st = case(monkeypatch)
    for uid, since in [(1, START), (5, START - 8 * DAY), (6, START - 2 * DAY)]:
        st.tracks[TRACK].submissions[key_of(uid)].champion_since_ts = since
    res = compute_leaders(st, [TRACK], view(UIDS), now=START + DAY)[TRACK]
    assert res.champions == {1, 5, 6}
    assert res.crowned[1] == START
    assert res.crowned[5] == START - 8 * DAY
    assert res.crowned[6] == START - 2 * DAY
    copied = _copy(res)
    assert copied.champions == res.champions and copied.crowned == res.crowned
    assert copied.champions is not res.champions


def test_vacant_generalist_does_not_pay_the_first_former_as_champion(monkeypatch):
    from test_opening_playoff_slots import make_state, view, TRACK
    from reax_subnet.spec import state as st
    from reax_subnet.spec.evaluator import compute_leaders
    from reax_subnet.competitions import _copy
    state = make_state()
    track = state.tracks[TRACK]
    old = track.submissions[track.champion]
    old.status, old.champion_since_ts = st.FORMER, START
    track.champion = None
    res = _copy(compute_leaders(state, [TRACK], view(), now=START + DAY)[TRACK])
    assert res and res.champions == set()
    w = cs.combine_weights(BASE, None, 0., {TRACK: res}, owner_uid=3, now=START + DAY,
                           paused=[TRACK], pause_starts={TRACK: START})
    assert w[1] == pytest.approx(.075 / len(res))


@pytest.mark.parametrize('age,elapsed,expected', [(1, 0, .2), (1, 3, .15), (5, 0, .1), (5, 1, .05), (7, 0, .01), (8, 0, .01), (8, 1, .01)])
def test_remote_engine_reads_pause_start_and_exact_crowning(tmp_path, age, elapsed, expected):
    from types import SimpleNamespace
    from test_remote_competitions import Clock, SERVING, snap, stamp, write_engine_state
    from reax_subnet import remote
    from reax_subnet.config import ENGINE_STATE_FILE
    a = cs.NEED_START_UTC
    now = a + timedelta(days=elapsed)
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    raw = json.loads(path.read_text())
    eng = raw['tracks']['s1-fast']
    eng['history'] = [dict(event='crowned', digest=eng['champion']['digest'], hotkey='hk-eng',
                           slot_utc=(a - timedelta(days=age)).isoformat())]
    path.write_text(json.dumps(raw))
    stamp(path, now.timestamp())
    (tmp_path / 'need.json').write_text(json.dumps({'pauses': [dict(track='engine', from_utc=a.isoformat(),
                          until_utc=None, reason='test run-out')]}))
    gate = SimpleNamespace(path=str(tmp_path / 'approvals.json'), approved=lambda *a, **kw: True)
    comp = remote.RemoteEngineCompetition(str(path), utcnow=lambda: now, wall=Clock(now.timestamp()), award_gate=gate)
    w = comp.weights(SERVING, snap(), {0}, 0.)
    assert w[3] == pytest.approx(expected)
    assert w[11] + w[12] == pytest.approx(.4)
    assert w[0] == pytest.approx(1 - .4 - expected)


@pytest.mark.parametrize('elapsed', [0, 1, 2])
def test_engine_pot_times_frozen_fraction_no_move_up(tmp_path, monkeypatch, elapsed):
    from types import SimpleNamespace
    from test_remote_competitions import Clock, SERVING, snap, stamp, write_engine_state
    from reax_subnet import remote
    from reax_subnet.config import ENGINE_STATE_FILE
    from reax_subnet.engcomp import champion_fraction, RewardParams
    # Crown was five days old at pause start. Former slot expires one day into pause.
    crown_block = 900
    pause_block = crown_block + 5 * 7200
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_UTC', cs.NEED_START_UTC)
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_BLOCK', pause_block)
    monkeypatch.setattr(cs, 'ENGINE_FORMER_SLOT_BLOCKS', 6 * 7200)
    a = cs.NEED_START_UTC
    now = a + timedelta(days=elapsed)
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    raw = json.loads(path.read_text())
    eng = raw['tracks']['s1-fast']
    eng['history'] = [dict(event='crowned', digest='cd' * 32, hotkey='old', block=100, score=1.1),
                      dict(event='crowned', digest=eng['champion']['digest'], hotkey='hk-eng',
                           block=crown_block, slot_utc=(a - timedelta(days=5)).isoformat())]
    path.write_text(json.dumps(raw))
    stamp(path, now.timestamp())
    (tmp_path / 'need.json').write_text(json.dumps({'pauses': [dict(track='engine', from_utc=a.isoformat(),
                                                       until_utc=None, reason='test')]}))
    gate = SimpleNamespace(path=str(tmp_path / 'approvals.json'), approved=lambda *a, **k: True)
    comp = remote.RemoteEngineCompetition(str(path), utcnow=lambda: now, wall=Clock(now.timestamp()), award_gate=gate)
    w = comp.weights(SERVING, snap(block=pause_block + elapsed * 7200), {0})
    frozen = 1 - cs.ENGINE_RANK_SPLIT[1]
    pot = champion_fraction(RewardParams(), (5 + elapsed) * 7200)
    r = max(0, (2 - elapsed) / 4)
    assert w[3] == pytest.approx(max(.01, .2 * pot * frozen * r))
    assert sum(w.values()) == pytest.approx(1)
    assert w[0] == pytest.approx(.6 - w[3])


def test_generalist_unapproved_coholder_does_not_move_up():
    tr = leaders({20: START, 24: START})
    tr.cochamps = {20: .5, 24: .5}
    w = weights(START, tr, award_ok=lambda track, uid: uid != 24)
    assert w[20] == pytest.approx(.225 / 2)
    assert 24 not in w
    assert sum(w.values()) == pytest.approx(1)


@pytest.mark.parametrize('age', [5, 7, 8])
def test_legacy_v1_uid62_block_crown_runs_out(tmp_path, monkeypatch, age):
    from types import SimpleNamespace
    from dataclasses import replace
    from test_remote_competitions import Clock, SERVING, snap, stamp, write_engine_state
    from reax_subnet import remote, needctl
    from reax_subnet.config import ENGINE_STATE_FILE
    a = cs.NEED_START_UTC
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_UTC', a)
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_BLOCK', 900 + age * 7200)
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)  # legacy: no UTC event, champion.crowned_block only
    raw = json.loads(path.read_text())
    raw['tracks']['s1-fast']['champion']['uid'] = 62
    path.write_text(json.dumps(raw))
    stamp(path, a.timestamp())
    (tmp_path / 'need.json').write_text(json.dumps({'pauses': [dict(track='engine', from_utc=a.isoformat(),
                                                       until_utc=None, reason='test')]}))
    assert needctl.holder_times('engine', [str(path)]) == [(a - timedelta(days=age)).timestamp()]
    gate = SimpleNamespace(path=str(tmp_path / 'approvals.json'), approved=lambda *a, **k: True)
    comp = remote.RemoteEngineCompetition(str(path), utcnow=lambda: a, wall=Clock(a.timestamp()), award_gate=gate)
    snapshot = snap(block=900 + age * 7200)
    snapshot = replace(snapshot, neurons=tuple(replace(n, uid=62) if n.uid == 3 else n for n in snapshot.neurons))
    w = comp.weights(SERVING, snapshot, {0})
    expected = max(.01, .2 * cs.engine_decay_fraction(age * DAY) * cs.runout_factor(START - age * DAY, START, START))
    assert w[62] == pytest.approx(expected)
    assert sum(w.values()) == pytest.approx(1)


@pytest.mark.parametrize('days,factor', [(0, 1.), (5, .5)])
@pytest.mark.parametrize('waiting', [0, 3])
def test_paused_title_uses_unpaused_rank_budget(days, factor, waiting):
    ka = {'generalist': [(40 + i, 30 - i) for i in range(waiting)]}
    normal = weights(START, paused=[], keepalive=ka, floors={})
    paused = weights(START + days * DAY, keepalive=ka, floors={})
    assert normal[20] == pytest.approx(.16875 if waiting else .225)
    assert paused[20] == pytest.approx(normal[20] * factor)
    assert sum(paused.values()) == pytest.approx(1.)


@pytest.mark.parametrize('known', [True, False])
def test_opening_coholder_without_own_time_uses_group_or_fails_open(monkeypatch, known):
    from test_cochampions import case, key_of, UIDS
    from test_opening_playoff_slots import TRACK, view
    from reax_subnet.spec.evaluator import compute_leaders
    st = case(monkeypatch)
    monkeypatch.setattr(cs, 'GENERALIST_CO_CHAMPIONS_CROWNED',
                        cs.NEED_START_UTC if known else None)
    for uid in (1, 5, 6):
        st.tracks[TRACK].submissions[key_of(uid)].champion_since_ts = None
    tr = compute_leaders(st, [TRACK], view(UIDS), now=START + DAY)[TRACK]
    assert tr.crowned.get(5) == (START if known else None)
    paused = cs.combine_weights(BASE, None, 0., {TRACK: tr}, owner_uid=3,
                                now=START + 5 * DAY, paused=[TRACK], pause_starts={TRACK: START})
    normal = cs.combine_weights(BASE, None, 0., {TRACK: tr}, owner_uid=3, now=START + 5 * DAY, paused=[])
    assert paused[5] == pytest.approx(normal[5] * (.5 if known else 1.))


@pytest.mark.parametrize('elapsed', [0, 5, 6.5, 7])
def test_dm_all_pause_freezes_carried_v1_former_ranks(tmp_path, monkeypatch, elapsed):
    monkeypatch.setattr(cs, "ENGINE_TRACK_START_BLOCK", 1000000)
    from test_engine_switch import _carry_comp, _chain, _snapshot, blk, FRI
    from reax_subnet.engcomp import champion_fraction
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_UTC', cs.NEED_START_UTC)
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_BLOCK', 1000000)
    now = FRI + timedelta(days=elapsed)
    comp = _carry_comp(tmp_path, now, 'carry', crowned_at=FRI)
    import hashlib
    from pathlib import Path
    v1_path, end, _ = comp.v1_bridge
    raw = json.loads(Path(v1_path).read_text())
    v1 = raw['tracks']['s1-fast']
    for event, days in zip(v1['history'], (3, 2, 1)):
        event['block'] = blk(FRI - timedelta(days=days))
    v1['champion']['crowned_block'] = blk(FRI - timedelta(days=1))
    Path(v1_path).write_text(json.dumps(raw))
    comp.v1_bridge = (v1_path, end, hashlib.sha256(Path(v1_path).read_bytes()).hexdigest())
    comp.award_gate.path = str(tmp_path / 'approvals.json')
    (tmp_path / 'need.json').write_text(json.dumps({'pauses': [dict(
        track='engine', from_utc=FRI.isoformat(), until_utc=None, reason='carry runout')]}))
    # The two carried former places hold 17% + 8% at pause start. They expire
    # during the pause, but their freed shares never move up to the v2 title.
    from dataclasses import replace
    snapshot = _snapshot(_chain(), blk(now))
    owner = replace(snapshot.neurons[0], uid=99, hotkey='owner', commitment=None)
    snapshot = replace(snapshot, owner_hotkey='owner', neurons=(*snapshot.neurons, owner))
    w = comp.weights({5: 1.0}, snapshot, {99}, 0.)
    pot = champion_fraction(comp.reward, elapsed * 7200)
    floor = .01 / (3 if elapsed < 6 else 2 if elapsed < 7 else 1)
    assert w[1] == pytest.approx(max(floor, .2 * pot * .75 * cs.runout_factor(
        FRI.timestamp(), FRI.timestamp(), now.timestamp())))
    assert w.get(0, 0.) == pytest.approx(floor if elapsed < 7 else 0.)
    assert w.get(2, 0.) == pytest.approx(floor if elapsed < 6 else 0.)
    assert sum(w.values()) == pytest.approx(1.)


@pytest.mark.parametrize('fraction', [1., .8, .75])
@pytest.mark.parametrize('days,factor', [(0, 1.), (5, .5)])
def test_engine_runout_uses_ordinary_waiting_deduction(fraction, days, factor):
    # combine_weights takes the engine fraction positionally.
    normal = cs.combine_weights(BASE, 30, fraction, {}, owner_uid=3, now=START, engine_waiting=[31])
    paused = cs.combine_weights(BASE, 30, fraction, {}, owner_uid=3, now=START + days * DAY,
                                engine_waiting=[31], paused=['engine'], pause_starts={'engine': START},
                                engine_runout={30: (fraction, START)})
    assert paused[30] == pytest.approx(normal[30] * factor)
    assert sum(paused.values()) == pytest.approx(1.)


@pytest.mark.parametrize('days,factor', [(0, 1.), (5, .5)])
def test_engine_coholder_runout_uses_ordinary_waiting_deduction(days, factor):
    parts = {30: .4, 32: .4}
    normal = cs.combine_weights(BASE, None, 0., {}, owner_uid=3, now=START,
                                engine_cochampions=parts, engine_waiting=[31])
    paused = cs.combine_weights(BASE, None, 0., {}, owner_uid=3, now=START + days * DAY,
                                engine_cochampions=parts, engine_waiting=[31], paused=['engine'],
                                pause_starts={'engine': START}, engine_runout={u: (p, START) for u, p in parts.items()})
    for uid in parts:
        assert paused[uid] == pytest.approx(normal[uid] * factor)
    assert sum(paused.values()) == pytest.approx(1.)


@pytest.mark.parametrize('dynamic', [False, True])
def test_generalist_group_fallback_preserves_a_members_own_time(monkeypatch, dynamic):
    from test_cochampions import case, key_of, UIDS
    from test_opening_playoff_slots import TRACK, view
    from reax_subnet.spec.evaluator import compute_leaders
    st = case(monkeypatch)
    ts = st.tracks[TRACK]
    monkeypatch.setattr(cs, 'GENERALIST_CO_CHAMPIONS_CROWNED', cs.NEED_START_UTC)
    if dynamic:
        ts.cochamps = [key_of(u) for u in (1, 5, 6)]
        ts.cochamps_since = START
    ts.submissions[key_of(1)].champion_since_ts = START - 2 * DAY
    ts.submissions[key_of(5)].champion_since_ts = START - DAY
    ts.submissions[key_of(6)].champion_since_ts = None
    tr = compute_leaders(st, [TRACK], view(UIDS), now=START + DAY)[TRACK]
    assert tr.crowned[1] == START - 2 * DAY
    assert tr.crowned[5] == START - DAY
    assert tr.crowned[6] == START


def test_engine_opening_member_uses_group_crown_and_needctl_names_unknown(tmp_path, monkeypatch):
    from reax_subnet.engcomp import Champion, TrackState
    from reax_subnet.engvalidator import EngineCompetition
    from reax_subnet import needctl
    monkeypatch.setattr(cs, 'ENGINE_CO_CHAMPIONS', {'s1-fast': (('aa', 'holder'), ('bb', 'opening'))})
    monkeypatch.setattr(cs, 'ENGINE_CO_CHAMPION_CROWNED', {'s1-fast': 100})
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_BLOCK', 100)
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_UTC', cs.NEED_START_UTC)
    ts = TrackState(champion=Champion('aa', 'holder', 1, 2., 110))
    assert EngineCompetition._crowned_time(ts, 'bb', 'opening') == START
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'tracks': {'s1-fast': ts.to_json()}}))
    assert needctl.holder_times('engine', [str(path)]) == [START + 120, START]
    monkeypatch.setattr(cs, 'ENGINE_TRACK_START_BLOCK', None)
    assert EngineCompetition._crowned_time(ts, 'bb', 'opening') is None
    with pytest.raises(ValueError, match=r's1-fast hotkey=holder digest=aa missing crowning time'):
        needctl.holder_times('engine', [str(path)])


@pytest.mark.parametrize('before', [True, False])
def test_engine_no_pause_never_reads_runout_crown(tmp_path, monkeypatch, before):
    from test_remote_competitions import Clock, SERVING, snap, stamp, write_engine_state
    from reax_subnet import remote
    from reax_subnet.config import ENGINE_STATE_FILE
    from types import SimpleNamespace
    now = cs.NEED_START_UTC + timedelta(seconds=-1 if before else DAY)
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    stamp(path, now.timestamp())
    gate = SimpleNamespace(path=str(tmp_path / 'approvals.json'), approved=lambda *a, **k: True)
    comp = remote.RemoteEngineCompetition(str(path), utcnow=lambda: now, wall=Clock(now.timestamp()), award_gate=gate)
    def forbidden(*args):
        raise AssertionError('run-out crown read without an active pause')
    monkeypatch.setattr(comp, '_crowned_time', forbidden)
    assert comp.weights(SERVING, snap(), {0})[3] > 0


@pytest.mark.parametrize('provisional', [[], [45], [45, 46]])
@pytest.mark.parametrize('waiting', [0, 1, 3])
def test_paused_generalist_preserves_budget_with_provisional_entrants(waiting, provisional):
    ka = {'generalist': [(40 + i, 30 - i) for i in range(waiting)]}
    kw = dict(keepalive=ka, floors={}, provisional={'generalist': provisional})
    normal = weights(START, paused=[], **kw)
    paused = weights(START, **kw)
    assert paused[20] == pytest.approx(normal[20])
    assert sum(paused.values()) == pytest.approx(1.)
