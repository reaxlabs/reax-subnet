"""Reference determinism, release admission and two-session champion re-checks."""

import dataclasses
from types import SimpleNamespace

import pytest

from reax_subnet import engrunner as ER
from reax_subnet import engvalidator as EV
from reax_subnet.engconfigs import required_engine_margin
from reax_subnet.engcomp import (
    Champion, EvalResult, Submission, TrackState, confirmation_due, needs_remeasurement, observe, update_champion,
)
from test_engcomp import D1, P, FAST_P, ref_of
from test_engine_hardening import FakeDocker, REF, _snap


def test_noise_margin_is_computed_and_cannot_be_lowered():
    assert required_engine_margin(0.005) == 0.03
    assert required_engine_margin(0.02) == 0.06
    for x in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            required_engine_margin(x)
    sub = Submission(1, 'hk', P.name, 'ghcr.io/a/e', D1)
    st = TrackState()
    observe(st, [sub], 10)
    params = dataclasses.replace(P, reference_noise_std=0.02, min_gain=0, margin=0, confirmations=1)
    st.results[D1] = EvalResult(D1, P.name, True, score=1.04, session='one')
    assert update_champion(st, [sub], params, 10) is None
    st.results[D1].score = 1.07
    assert update_champion(st, [sub], params, 10) is not None
    from reax_subnet.engcomp import track_thresholds
    from reax_subnet.engthresholds import thresholds_sha256
    assert thresholds_sha256(track_thresholds(P)) != thresholds_sha256(track_thresholds(params))


def test_champion_needs_two_distinct_consecutive_invalid_sessions_and_persists():
    sub = Submission(1, 'hk', P.name, 'ghcr.io/a/e', D1)
    st = TrackState(champion=Champion(D1, 'hk', 1, 1.5, 1))
    observe(st, [sub], 10)
    st.results[D1] = EvalResult(D1, P.name, False, session='bad1')
    assert update_champion(st, [sub], P, 10) is not None
    assert confirmation_due(st, [sub], P)
    assert needs_remeasurement(st, [sub]) == [sub]
    st = TrackState.from_json(st.to_json())
    assert update_champion(st, [sub], P, 10) is not None  # same result cannot count twice
    st.results[D1] = EvalResult(D1, P.name, True, score=1.4, session='good')
    assert update_champion(st, [sub], P, 11).score == 1.4
    assert st.champion_invalid == 0
    st.results[D1] = EvalResult(D1, P.name, False, session='bad2')
    assert update_champion(st, [sub], P, 12) is not None
    st.results[D1] = EvalResult(D1, P.name, False, session='bad3')
    assert update_champion(st, [sub], P, 13) is None


async def test_unknown_or_unpublished_gpu_voids_before_any_container(tmp_path, monkeypatch):
    docker = FakeDocker(present=True)
    monkeypatch.setattr(ER, '_run', docker)
    monkeypatch.setattr(ER, 'PUBLISHED_GPU_CLASSES', frozenset())
    runner = ER.DockerRunner(str(tmp_path), str(tmp_path))
    with pytest.raises(ER.ReferenceSelfCheckFailed, match='no published FLA config'):
        await runner.start(REF, P.pool, P)
    assert docker.cmds('docker', 'run') == []
    comp = EV.EngineCompetition(runner, {P.name: REF}, str(tmp_path / 'state.json'), tracks={P.name: P})
    assert await comp.run_cycle(_snap(f'reax-eng/1:{P.name}:{REF}'), {1}) == {}
    st = comp.states[P.name]
    assert st.cycle_events[-1]['event'] == 'session_void'
    assert not st.results and not st.attempts


async def test_both_containers_receive_strict_fixed_config_env(tmp_path, monkeypatch):
    docker = FakeDocker(present=True)
    monkeypatch.setattr(ER, '_run', docker)
    monkeypatch.setattr(ER, 'PUBLISHED_GPU_CLASSES', frozenset(ER.GPU_CONFIG_DIRS))
    runner = ER.DockerRunner(str(tmp_path), str(tmp_path))
    for keep in (True, False):
        eng = await runner.start(REF, P.pool, P, keep_image=keep)
        await eng.stop()
        await eng.client.close()
    runs = docker.cmds('docker', 'run')
    assert len(runs) == 2
    for argv in runs:
        assert 'FLA_CACHE_MODE=strict' in argv
        assert 'FLA_CONFIG_DIR=/opt/reax/fla-configs/rtx-pro-6000-blackwell' in argv


async def test_reference_repeat_set_is_fixed_and_instability_voids(monkeypatch):
    seen = []

    async def answers(client, decisions):
        seen.append([d.ident for d in decisions])
        probs = (0.9, 0.1) if len(seen) == 1 else (0.1, 0.9)
        return {d.ident: probs for d in decisions}

    monkeypatch.setattr(ER, '_answers', answers)
    with pytest.raises(ER.ReferenceSelfCheckFailed, match='self-check'):
        await ER.reference_self_check(SimpleNamespace(client=None), FAST_P)
    assert len(seen) == 2 and seen[0] == seen[1]


async def test_self_check_failure_is_never_a_miner_strike(tmp_path, monkeypatch):
    class Runner:
        binds_hotkey = False

    async def broken(*args, **kwargs):
        raise ER.ReferenceSelfCheckFailed('reference differs')

    monkeypatch.setattr(EV, 'evaluate', broken)
    comp = EV.EngineCompetition(Runner(), {P.name: ref_of('reference')}, str(tmp_path / 's.json'),
                                tracks={P.name: P}, interval_s=0)
    snap = _snap(f'reax-eng/1:{P.name}:{REF}')
    for _ in range(P.max_infra_attempts + 1):
        await comp.run_cycle(snap, set())
    st = comp.states[P.name]
    assert not st.results and not st.attempts
    assert all(h['event'] == 'session_void' for h in st.cycle_events)
    assert EV.EngineCompetition(Runner(), {P.name: ref_of('reference')}, str(tmp_path / 's.json'),
                                tracks={P.name: P}).states[P.name].cycle_events == st.cycle_events


def test_strict_reference_forbids_fla_missing_key_fallback(monkeypatch):
    import sys
    from reax_subnet.engfla import require_fla_configs

    calls = []
    config = {"kwargs": {}, "num_warps": 4}
    fake = SimpleNamespace(load_cached_config=lambda kernel, key: calls.append((kernel, key)) or
                           (config if key == "published" else None))
    monkeypatch.setitem(sys.modules, "fla.ops.utils", SimpleNamespace(cache=fake))
    monkeypatch.setenv("FLA_CACHE_MODE", "strict")
    require_fla_configs()
    assert fake.load_cached_config("kernel", "published") is config
    with pytest.raises(RuntimeError, match="unpublished FLA config"):
        fake.load_cached_config("kernel", "missing")
    assert len(calls) == 2


def test_serving_default_does_not_install_competition_fla_guard(monkeypatch):
    from reax_subnet.engfla import require_fla_configs
    monkeypatch.delenv("FLA_CACHE_MODE", raising=False)
    require_fla_configs()  # no FLA/GPU imports in the unchanged serving path
