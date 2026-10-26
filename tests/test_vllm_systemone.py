"""CPU-only transport and signed miner tests against an in-process vLLM stub."""
import asyncio
import json
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import auth
from reax_subnet.cli import _s1_backend
from reax_subnet.config import ConfigError, load_common
from reax_subnet.manifest import RELEASE_MANIFESTS, load_manifest
from reax_subnet.miner import S1Miner
from reax_subnet.s1backends import SidecarError, VllmSystemOneBackend
from reax_subnet.s1contract import check_answer, parse_decision
from reax_subnet.s1protocol import S1Request, S1_DECIDE_PATH, parse_s1_response


@asynccontextmanager
async def serve(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    try:
        yield f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    finally:
        await runner.cleanup()


@pytest.fixture
async def upstream():
    control = {'status': 200, 'delay': 0.005, 'calls': [], 'active': 0, 'peak': 0, 'mutate': lambda a: a, 'mutate_response': lambda r: r}

    async def handle(request):
        body = await request.json()
        control['calls'].append(body)
        control['active'] += 1
        control['peak'] = max(control['peak'], control['active'])
        try:
            await asyncio.sleep(control['delay'])
            if control['status'] != 200:
                return web.json_response({'error': 'refused'}, status=control['status'])
            qid, q = next(iter(body['questions'].items()))
            kind = q['type']
            probs = {'yes': .8, 'no': .2} if kind == 'noul' else (
                {'0': .2, '1': .3, '2': .5} if kind == 'score' else {'z': .25, 'a': .75})
            answer = {'type': kind, 'probabilities': probs, 'confidence': .001}
            answer.update({'noul': .01} if kind == 'noul' else (
                {'score': 999, 'legend': {}} if kind == 'score' else {'choice': 'z'}))
            return web.json_response(control['mutate_response']({'id': 'upstream', 'object': 'structured_decision', 'model': body['model'],
                                     'answers': {qid: control['mutate'](answer)},
                                     'usage': {'input_tokens': 12, 'output_tokens': 0},
                                     'diagnostics': {qid: {'label_mass': .1, 'argmax_is_label': True}}}))
        finally:
            control['active'] -= 1

    app = web.Application()
    app.router.add_post('/v1/systemone', handle)
    async with serve(app) as url:
        control['url'] = url
        yield control


def decision(kind, ident='d'):
    q = {'type': kind, 'instructions': 'Apply this to the state.'}
    if kind == 'choice':
        q['criteria'] = {'z': 'first', 'a': 'second'}
    elif kind == 'score':
        q['criteria'] = ['low', 'medium', 'high']
    else:
        q['criteria'] = {'true': 'matches', 'false': 'does not match'}
    return parse_decision({'id': ident, 'state': {'text': 'the shared state'}, 'questions': {'typed_q': q}}, 's1-pro')


@pytest.mark.parametrize('kind,values', [('choice', (.25, .75)), ('noul', (.8, .2)), ('score', (.2, .3, .5))])
async def test_roundtrip(upstream, kind, values):
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1')
    d = decision(kind)
    try:
        answer, = await backend.decide([d])
        assert check_answer(d.question, answer) == pytest.approx(values)
        assert list(answer['probabilities']) == list(d.question.keys)
        assert upstream['calls'] == [{'model': 'mammoth-1', 'state': d.state,
                                      'questions': {'typed_q': d.question.author_dict()}}]
        if kind in ('choice', 'score'):
            assert answer['confidence'] == max(values)
        if kind == 'score':
            assert answer['score'] == pytest.approx(1.3)
        if kind == 'noul':
            assert answer['noul'] == .8
    finally:
        await backend.close()


async def test_batch_16_preserves_order_and_bounds_concurrency(upstream):
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1', max_inflight=3)
    decisions = [decision(('choice', 'noul', 'score')[i % 3], str(i)) for i in range(16)]
    try:
        answers = await backend.decide(decisions)
        assert len(answers) == len(upstream['calls']) == 16
        assert 1 < upstream['peak'] <= 3
        assert backend.inflight == 0
        for d, a in zip(decisions, answers):
            check_answer(d.question, a)
    finally:
        await backend.close()


@pytest.mark.parametrize('status,fault', [(400, False), (503, True), (429, True)])
async def test_http_failure(upstream, status, fault):
    upstream['status'] = status
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1')
    try:
        errors = await backend.decide_each([decision('noul')], retries=0)
        assert isinstance(errors[0], SidecarError)
        assert errors[0].backend_fault is fault
        assert errors[0].answer_started is False
        assert not await backend.probe()
    finally:
        await backend.close()


async def test_timeout(upstream):
    upstream['delay'] = .05
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1', timeout_s=.005)
    try:
        with pytest.raises(SidecarError, match='transport failed') as error:
            await backend.decide([decision('noul')])
        assert error.value.backend_fault and not error.value.answer_started
    finally:
        await backend.close()


@pytest.mark.parametrize('bad', [
    {'yes': .9, 'no': .9}, {'yes': True, 'no': 0}, {'yes': .8},
    {'yes': '0.8', 'no': .2}, {'yes': float('nan'), 'no': .2},
    {'yes': -0.1, 'no': 1.1}, {'yes': .8, 'no': .2, 'extra': 0},
])
async def test_invalid_probabilities_are_refused(upstream, bad):
    upstream['mutate'] = lambda answer: {**answer, 'probabilities': bad}
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1')
    try:
        with pytest.raises(SidecarError, match='invalid answer'):
            await backend.decide([decision('noul')])
    finally:
        await backend.close()


@pytest.mark.parametrize('status', [200, 400, 503, 'timeout'])
async def test_signed_miner_path(upstream, make_wallet, monkeypatch, status):
    # Only this isolated test registers the candidate; the release stays on Rune.
    monkeypatch.setitem(RELEASE_MANIFESTS, "s1-pro", "s1-pro-mammoth-v1.json")
    mw, vw = make_wallet('//M'), make_wallet('//V')
    if status == 'timeout':
        upstream['delay'] = .05
    else:
        upstream['status'] = status
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1',
                                 timeout_s=.005 if status == 'timeout' else 2)
    miner = S1Miner(mw, backend)
    miner.set_validators(frozenset({vw.hotkey.ss58_address}))
    req = S1Request('signed', 's1-pro', backend.manifest.manifest_id,
                    tuple(decision(('choice', 'noul', 'score')[i % 3], str(i)) for i in range(16)))
    body = req.to_bytes()
    headers = auth.sign_request(vw, body, mw.hotkey.ss58_address, path=S1_DECIDE_PATH)
    try:
        async with serve(miner.app()) as url, aiohttp.ClientSession() as session:
            async with session.post(url + S1_DECIDE_PATH, data=body, headers=headers) as response:
                raw = await response.read()
                if status == 200:
                    assert response.status == 200
                    auth.verify_response(response.headers, raw, req.request_id, vw.hotkey.ss58_address,
                                         mw.hotkey.ss58_address, auth.NonceStore(), 30, path=S1_DECIDE_PATH)
                    assert len(parse_s1_response(raw, req)) == 16
                else:
                    assert response.status == 500
                    assert json.loads(raw) == {'error': 'inference failed'}
                    assert not any(k.lower().startswith('x-bittensor-') for k in response.headers)
    finally:
        await backend.close()


def env():
    return {'REAX_MODEL': 's1-pro', 'REAX_S1_PRO_BACKEND': 'vllm-systemone',
            'REAX_S1_PRO_RUNTIME_URL': 'http://10.0.0.2:8000', 'REAX_S1_PRO_SERVED_MODEL': 'mammoth-1',
            'REAX_S1_PRO_RUNTIME_TIMEOUT': '12.5', 'REAX_RUNTIME_INFLIGHT': '3'}


def test_config_factory_without_sidecar_token():
    cfg = load_common(env())
    backend = _s1_backend(cfg, 's1-pro')
    assert isinstance(backend, VllmSystemOneBackend)
    assert backend.url == 'http://10.0.0.2:8000'
    assert backend.served_model == 'mammoth-1'
    assert backend.max_inflight == 3 and backend._timeout == 12.5
    assert cfg.runtime_token_file == ''


@pytest.mark.parametrize('change', [
    {'REAX_S1_PRO_BACKEND': 'unknown'}, {'REAX_S1_PRO_SERVED_MODEL': ''},
    {'REAX_S1_PRO_RUNTIME_TIMEOUT': 'nan'}, {'REAX_S1_PRO_RUNTIME_TIMEOUT': '0'},
    {'REAX_S1_PRO_RUNTIME_URL': 'https://example.com'},
    {'REAX_S1_PRO_RUNTIME_URL': 'http://10.0.0.2/v1/systemone'},
    {'REAX_S1_PRO_RUNTIME_URLS': 'http://10.0.0.3:8000'},
])
def test_bad_config(change):
    with pytest.raises(ConfigError):
        load_common({**env(), **change})


def test_draft_manifest_loads_only_with_explicit_test_registration(monkeypatch):
    release = load_manifest('s1-pro')
    assert RELEASE_MANIFESTS['s1-pro'] == 's1-pro-v1.json'
    monkeypatch.setitem(RELEASE_MANIFESTS, 's1-pro', 's1-pro-mammoth-v1.json')
    draft = load_manifest('s1-pro')
    assert draft.manifest_id != release.manifest_id
    assert draft.temperature == 1
    assert len(draft.weight_files) == 10
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in draft.weight_files.values())
    assert draft.data['weights']['revision'] == '03007ab3c4481bf01c71ef63adbbbc4b2093d71a'


@pytest.mark.parametrize('change', [
    {'model': 'wrong-model'}, {'answers': {}},
    {'answers': {'wrong-id': {'type': 'noul', 'probabilities': {'yes': .8, 'no': .2}}}},
    {'answers': {'typed_q': {'type': 'choice', 'probabilities': {'yes': .8, 'no': .2}}}},
])
async def test_mismatched_upstream_response(upstream, change):
    upstream['mutate_response'] = lambda result: {**result, **change}
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1')
    try:
        with pytest.raises(SidecarError, match='invalid answer'):
            await backend.decide([decision('noul')])
    finally:
        await backend.close()


async def test_tie_uses_original_option_order(upstream):
    upstream['mutate'] = lambda answer: {**answer, 'probabilities': {'a': .5, 'z': .5}}
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1')
    try:
        answer, = await backend.decide([decision('choice')])
        assert list(answer['probabilities']) == ['z', 'a']
        assert answer['choice'] == 'z'
        assert answer['confidence'] == .5
    finally:
        await backend.close()


async def test_operator_config_runs_startup_probe(upstream):
    from types import SimpleNamespace
    from reax_subnet.cli import _single_machine_backend

    cfg = SimpleNamespace(common=load_common({**env(), 'REAX_S1_PRO_RUNTIME_URL': upstream['url']}))
    backend = await _single_machine_backend(cfg, 's1-pro')
    try:
        assert isinstance(backend, VllmSystemOneBackend)
        assert len(upstream['calls']) == 1
        assert upstream['calls'][0]['model'] == 'mammoth-1'
    finally:
        await backend.close()
