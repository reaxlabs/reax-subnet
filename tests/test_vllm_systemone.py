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
    control = {'status': 200, 'delay': 0.005, 'calls': [], 'headers': [], 'active': 0, 'peak': 0,
               'mutate': lambda a: a, 'mutate_response': lambda r: r, 'models': ['mammoth-1'], 'health': 200}

    async def handle(request):
        body = await request.json()
        control['calls'].append(body)
        control['headers'].append(dict(request.headers))
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

    async def health(request):
        return web.Response(status=control['health'])

    async def models(request):
        return web.json_response({'object': 'list', 'data': [{'id': m, 'object': 'model'} for m in control['models']]})

    app = web.Application()
    app.router.add_post('/v1/systemone', handle)
    app.router.add_get('/health', health)
    app.router.add_get('/v1/models', models)
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


@pytest.mark.parametrize('status,fault', [(400, False), (503, True), (429, True), (408, True)])
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
async def test_signed_miner_path(upstream, make_wallet, status):
    # The backend answers for the pool's current manifest (the live mammoth-1 manifest since the 30 Oct switch).
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
                    # Nothing of the validator's request (signature headers) or of the sidecar contract goes upstream.
                    for h in upstream['headers']:
                        names = {k.lower() for k in h}
                        assert not any(k.startswith(('x-bittensor', 'x-s1m', 'x-reax')) for k in names), names
                        assert 'authorization' not in names and 'cookie' not in names
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


# --- readiness, selfcheck through the backend factory, transport edge cases (review of PR #8, 30 Oct 2026) ---

async def test_ready_is_vllm_health_and_served_model(upstream):
    backend = VllmSystemOneBackend('s1-pro', upstream['url'], 'mammoth-1')
    try:
        assert await backend.ready()
        upstream['models'] = ['some-other-model']
        assert not await backend.ready()
        upstream['models'] = ['mammoth-1']
        upstream['health'] = 503
        assert not await backend.ready()
    finally:
        await backend.close()


def test_factory_keeps_sidecar_and_pool_behaviour():
    from reax_subnet.s1backends import PooledBackend, SidecarBackend, load_s1_backend

    single = load_s1_backend('s1-pro', 'http://127.0.0.1:9000', 'tok', False)
    assert type(single) is SidecarBackend and single._headers['X-S1M-Internal-Token'] == 'tok'
    pool = load_s1_backend('s1-pro', None, 'tok', False, endpoints=(('http://127.0.0.1:9000', ''),
                                                                     ('http://127.0.0.1:9001', 'other')))
    assert isinstance(pool, PooledBackend)
    with pytest.raises(ValueError, match='TOKEN_FILE'):
        load_s1_backend('s1-pro', 'http://127.0.0.1:9000', '', False)
    vllm = load_s1_backend('s1-pro', 'http://10.0.0.2:8000', '', False, 4, kind='vllm-systemone',
                           served_model='mammoth-1', timeout_s=7)
    assert isinstance(vllm, VllmSystemOneBackend) and vllm.max_inflight == 4 and vllm._timeout == 7
    assert 'X-S1M-Internal-Token' not in vllm._headers
    assert vllm.manifest.manifest_id == load_manifest('s1-pro').manifest_id


class _ReferenceVllm:
    """A thread-run stand-in for vLLM + vllm-mammoth that answers every reference item with its published
    probabilities (the selfcheck must then pass with distance 0)."""

    def __init__(self, served='mammoth-1', listed=('mammoth-1',), skew=0.0):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from reax_subnet.selfcheck import load_reference

        ref = {(json.dumps(it['state'], sort_keys=True), it['question']['instructions']): it
               for it in load_reference('s1-pro')['items']}
        outer = self
        self.headers = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == '/health':
                    self._json(200, {})
                elif self.path == '/v1/models':
                    self._json(200, {'data': [{'id': m} for m in listed]})
                else:
                    self._json(404, {})

            def do_POST(self):
                outer.headers.append(dict(self.headers))
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                qid, q = next(iter(body['questions'].items()))
                it = ref[(json.dumps(body['state'], sort_keys=True), q['instructions'])]
                p = list(it['p'])
                if skew:
                    p = [x + skew if i == 0 else x for i, x in enumerate(p)]
                    p = [x / sum(p) for x in p]
                keys = ('yes', 'no') if q['type'] == 'noul' else tuple(
                    q['criteria']) if q['type'] == 'choice' else tuple(str(i) for i in range(len(q['criteria'])))
                self._json(200, {'model': served, 'answers': {qid: {'type': q['type'],
                                                                     'probabilities': dict(zip(keys, p))}}})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _selfcheck(monkeypatch, capsys, url, *extra, env_kind=True):
    from reax_subnet import manifest as manifest_mod
    from reax_subnet.selfcheck import main

    # --at mutates the switch table for the process; monkeypatch restores it.
    monkeypatch.setattr(manifest_mod, 'POOL_MANIFEST_SWITCH', dict(manifest_mod.POOL_MANIFEST_SWITCH))
    for k in ('REAX_RUNTIME_TOKEN_FILE', 'REAX_S1_PRO_RUNTIME_URLS'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('REAX_S1_PRO_RUNTIME_URL', url)
    if env_kind:
        monkeypatch.setenv('REAX_S1_PRO_BACKEND', 'vllm-systemone')
        monkeypatch.setenv('REAX_S1_PRO_SERVED_MODEL', 'mammoth-1')
    code = main(['--model', 's1-pro', '--at', '2026-10-31T00:00:00Z', '--json', '--concurrency', '16', *extra])
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1]) if out else {}


@pytest.mark.real_switch
def test_selfcheck_vllm_end_to_end_without_token(monkeypatch, capsys):
    with _ReferenceVllm() as fake:
        code, report = _selfcheck(monkeypatch, capsys, fake.url)
    assert code == 0, report
    assert report['pass'] and report['manifest_id'] == '3f3558e583b0a1d6'
    assert report['answered'] == report['items'] == 640 and report['distance_max'] == 0
    assert all(not any(k.lower().startswith(('x-s1m', 'x-bittensor')) for k in h) for h in fake.headers)


@pytest.mark.real_switch
def test_selfcheck_vllm_flags_instead_of_env(monkeypatch, capsys):
    with _ReferenceVllm(served='m1', listed=('m1',)) as fake:
        code, report = _selfcheck(monkeypatch, capsys, 'http://10.9.9.9:1', '--url', fake.url,
                                  '--backend', 'vllm-systemone', '--served-model', 'm1', '--limit', '64',
                                  env_kind=False)
    assert code == 0 and report['pass'] and report['items'] == 64


@pytest.mark.real_switch
def test_selfcheck_vllm_fails_on_wrong_probabilities(monkeypatch, capsys):
    with _ReferenceVllm(skew=0.5) as fake:
        code, report = _selfcheck(monkeypatch, capsys, fake.url)
    assert code == 1 and not report['pass']


@pytest.mark.real_switch
@pytest.mark.parametrize('listed,extra', [(('other',), ()), (('mammoth-1',), ('--served-model', '')),
                                          (('mammoth-1',), ('--timeout', '0'))])
def test_selfcheck_vllm_cannot_run(monkeypatch, capsys, listed, extra):
    with _ReferenceVllm(listed=listed) as fake:
        code, report = _selfcheck(monkeypatch, capsys, fake.url, *extra)
    assert code == 2 and report['pass'] is False and report['error']


def test_selfcheck_vllm_refuses_public_url_and_lists(monkeypatch, capsys):
    code, report = _selfcheck(monkeypatch, capsys, 'http://8.8.8.8:8000')
    assert code == 2 and 'private network' in report['error']
    code, report = _selfcheck(monkeypatch, capsys, 'http://10.0.0.2:8000', '--url', 'http://10.0.0.2:1,http://10.0.0.3:1')
    assert code == 2 and 'one base URL' in report['error']


def test_selfcheck_sidecar_still_needs_token(monkeypatch, capsys):
    code, report = _selfcheck(monkeypatch, capsys, 'http://127.0.0.1:9', env_kind=False)
    assert code == 2 and 'token' in report['error']


@asynccontextmanager
async def raw_server(handler):
    """A bare TCP server: ``handler(reader, writer, n)`` gets the n-th connection (for broken HTTP replies)."""
    count = [0]

    async def on_conn(reader, writer):
        count[0] += 1
        try:
            await handler(reader, writer, count[0])
        finally:
            writer.close()

    server = await asyncio.start_server(on_conn, '127.0.0.1', 0)
    try:
        yield f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}', count
    finally:
        server.close()
        await server.wait_closed()


async def _read_request(reader):
    head = await reader.readuntil(b'\r\n\r\n')
    length = next(int(l.split(b':')[1]) for l in head.split(b'\r\n') if l.lower().startswith(b'content-length'))
    return json.loads(await reader.readexactly(length))


def _http(body: bytes, status=b'200 OK', length=None, extra=b''):
    return (b'HTTP/1.1 ' + status + b'\r\nContent-Type: application/json\r\nContent-Length: '
            + str(len(body) if length is None else length).encode() + b'\r\n' + extra + b'\r\n' + body)


def _noul_answer(body):
    qid = next(iter(body['questions']))
    return json.dumps({'model': body['model'], 'answers': {qid: {'type': 'noul',
                                                                 'probabilities': {'yes': .8, 'no': .2}}}}).encode()


@pytest.mark.parametrize('drops', [1, 2])
async def test_stale_keepalive_is_retried_once(drops):
    async def handler(reader, writer, n):
        while True:
            try:
                body = await _read_request(reader)
            except (asyncio.IncompleteReadError, ConnectionError):
                return
            if n <= drops:
                return  # close without answering: what a keep-alive timeout on the server looks like
            writer.write(_http(_noul_answer(body)))
            await writer.drain()

    async with raw_server(handler) as (url, count):
        backend = VllmSystemOneBackend('s1-pro', url, 'mammoth-1')
        try:
            if drops == 1:
                answer, = await backend.decide([decision('noul')])
                assert answer['noul'] == .8 and count[0] == 2
            else:
                err, = await backend.decide_each([decision('noul')], retries=0)
                assert isinstance(err, SidecarError) and 'transport failed' in str(err)
                assert err.answer_started is False and err.backend_fault and count[0] == 2
        finally:
            await backend.close()


async def test_connection_refused():
    import socket

    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    backend = VllmSystemOneBackend('s1-pro', f'http://127.0.0.1:{port}', 'mammoth-1')
    try:
        err, = await backend.decide_each([decision('noul')], retries=0)
        assert isinstance(err, SidecarError) and err.backend_fault and err.answer_started is False
        assert not await backend.ready()
    finally:
        await backend.close()


@pytest.mark.parametrize('case', ['truncated', 'malformed', 'oversize', 'stall_after_headers'])
async def test_broken_bodies(case):
    async def handler(reader, writer, n):
        body = await _read_request(reader)
        good = _noul_answer(body)
        if case == 'truncated':
            writer.write(_http(good[:20], length=len(good)))
        elif case == 'malformed':
            writer.write(_http(good[:-3] + b'   '))
        elif case == 'oversize':
            big = b'{"pad": "' + b'x' * (1 << 20) + b'"}'
            writer.write(_http(big))
        else:
            writer.write(_http(b'', length=len(good)))
            await writer.drain()
            await asyncio.sleep(1)
        await writer.drain()

    async with raw_server(handler) as (url, count):
        backend = VllmSystemOneBackend('s1-pro', url, 'mammoth-1', timeout_s=.3)
        try:
            err, = await backend.decide_each([decision('noul')], retries=0)
            assert isinstance(err, SidecarError), err
            assert err.answer_started is True  # HTTP 200 arrived: never failed over or retried
            assert count[0] == 1
            expected = {'truncated': 'transport failed', 'malformed': 'invalid answer',
                        'oversize': 'too large', 'stall_after_headers': 'transport failed'}[case]
            assert expected in str(err)
        finally:
            await backend.close()


async def test_redirect_is_not_followed(upstream):
    async def redirect(request):
        return web.Response(status=307, headers={'Location': upstream['url'] + '/v1/systemone'})

    app = web.Application()
    app.router.add_post('/v1/systemone', redirect)
    async with serve(app) as url:
        backend = VllmSystemOneBackend('s1-pro', url, 'mammoth-1')
        try:
            err, = await backend.decide_each([decision('noul')], retries=0)
            assert isinstance(err, SidecarError) and 'HTTP 307' in str(err) and not err.backend_fault
            assert upstream['calls'] == []
        finally:
            await backend.close()
