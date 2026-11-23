import json
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import competition_split as cs, need, needctl

START = cs.NEED_START_UTC + timedelta(days=10)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return START + timedelta(days=1)
    monkeypatch.setattr(needctl, 'datetime', Clock)



@pytest.mark.parametrize('age', [7, 8])
@pytest.mark.parametrize('notice', [None, 23.999])
def test_old_holder_refused(tmp_path, age, notice):
    announced = START - timedelta(hours=notice) if notice is not None else None
    with pytest.raises(ValueError, match='24 hours notice'):
        needctl.schedule(tmp_path / 'need.json', 'engine', START, None, 'pause',
                         [(START - timedelta(days=age)).timestamp()], announced)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('age,notice', [(0, None), (5, None), (7, 24), (8, 25)])
def test_dry_run_and_atomic_backup(tmp_path, age, notice):
    path = tmp_path / 'need.json'
    original = b'{"inference": []}\n'
    path.write_bytes(original)
    crowns = [(START - timedelta(days=age)).timestamp()]
    announced = START - timedelta(hours=notice) if notice is not None else None
    step = needctl.schedule(path, 'engine', START, None, 'pause', crowns, announced)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert needctl.schedule(path, 'engine', START, None, 'pause', crowns, announced, True) == step
    assert len(list(tmp_path.glob('*.bak'))) == 1
    assert next(tmp_path.glob('*.bak')).read_bytes() == original
    assert json.loads(path.read_text()) == {'inference': [], 'pauses': [step]}
    assert need.parse(json.loads(path.read_text()))[2] == ''


def test_cli_checks_all_members_and_prints_step(tmp_path, capsys):
    from test_cochampions import case, key_of
    from reax_subnet.spec.state import _to_json
    # Native evaluator state, including a co-holder older than seven days.
    with pytest.MonkeyPatch.context() as mp:
        state = case(mp)
        # START is after the approval-clock cutoff. Supply a verified-gate-shaped
        # reader with approvals before each new crown; old crowns still ignore it.
        from types import SimpleNamespace
        from reax_subnet.award import AwardGate
        mp.setattr(AwardGate, 'from_env', classmethod(lambda cls, env: SimpleNamespace(
            approved_at=lambda *identity: (START - timedelta(days=1)).timestamp())))
        track = state.tracks['generalist']
        for sub in track.submissions.values():
            sub.champion_since_ts = START.timestamp()
        track.submissions[key_of(5)].champion_since_ts = (START - timedelta(days=8)).timestamp()
        state_path = tmp_path / 'spec.json'
        state_path.write_text(json.dumps(_to_json(state)))
        path = tmp_path / 'need.json'
        args = ['--path', str(path), '--track', 'generalist', '--state', str(state_path),
                '--from-utc', START.isoformat(), '--reason', 'test']
        with pytest.raises(SystemExit) as exc:
            needctl.main(args)
        assert exc.value.code == 2
        args += ['--announced-at', (START - timedelta(days=1)).isoformat()]
        assert needctl.main(args) == 0
        assert not path.exists()
        assert 'Dry run' in capsys.readouterr().out
        assert needctl.main(args + ['--write']) == 0
        assert path.exists()


def test_refuses_invalid_existing_file_and_naive_utc(tmp_path):
    path = tmp_path / 'need.json'
    path.write_text('{"unknown": []}')
    with pytest.raises(ValueError):
        needctl.schedule(path, 'engine', START, None, 'pause', [], None, True)
    assert path.read_text() == '{"unknown": []}'
    with pytest.raises(ValueError):
        needctl.utc('2026-10-01T00:00:00')


def test_refuses_full_row_budget_and_oversized_existing_file(tmp_path):
    path = tmp_path / 'need.json'
    step = dict(track='engine', from_utc=START.isoformat(), until_utc=None, reason='pause')
    path.write_text(json.dumps({'pauses': [step] * need.MAX_ROWS}))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='at most'):
        needctl.schedule(path, 'engine', START, None, 'pause', [], None, True)
    assert path.read_bytes() == original
    path.write_text('{}' + ' ' * need.MAX_BYTES)
    with pytest.raises(ValueError, match='byte limit'):
        needctl.schedule(path, 'engine', START, None, 'pause', [], None, True)
    assert not list(tmp_path.glob('*.bak'))


def test_missing_state_or_crowning_refuses(tmp_path):
    from test_cochampions import case
    from reax_subnet.spec.state import _to_json
    with pytest.MonkeyPatch.context() as mp:
        state = case(mp)
        mp.setattr(cs, 'GENERALIST_CO_CHAMPIONS_CROWNED', None)
        for sub in state.tracks['generalist'].submissions.values():
            sub.champion_since_ts = None
        path = tmp_path / 'state.json'
        path.write_text(json.dumps(_to_json(state)))
        with pytest.raises(ValueError, match='missing crowning time'):
            needctl.holder_times('generalist', [str(path)])
        with pytest.raises(ValueError, match='track missing'):
            needctl.holder_times('unknown', [str(path)])


def test_opening_members_use_group_time_and_refusal_names_member(tmp_path, monkeypatch, capsys):
    from test_cochampions import case, key_of
    from reax_subnet.spec.state import _to_json
    state = case(monkeypatch)
    track = state.tracks['generalist']
    for uid in (1, 5, 6):
        track.submissions[key_of(uid)].champion_since_ts = None
    monkeypatch.setattr(cs, 'GENERALIST_CO_CHAMPIONS_CROWNED', START)
    path = tmp_path / 'state.json'
    path.write_text(json.dumps(_to_json(state)))
    from types import SimpleNamespace
    gate = SimpleNamespace(approved_at=lambda *identity: START.timestamp())
    assert needctl.holder_times('generalist', [str(path)]) == []
    assert 'no verified approval time' in capsys.readouterr().err
    assert needctl.holder_times('generalist', [str(path)], gate) == [START.timestamp()] * 3
    monkeypatch.setattr(cs, 'GENERALIST_CO_CHAMPIONS_CROWNED', None)
    track.submissions[key_of(1)].champion_since_ts = START.timestamp()
    path.write_text(json.dumps(_to_json(state)))
    with pytest.raises(ValueError, match=r'generalist uid=5 key=.*missing crowning time'):
        needctl.holder_times('generalist', [str(path)], gate)
