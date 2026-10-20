"""Several worker machines behind one hotkey (REAX_FEATURE_MULTINODE): signed link, TLS pinning, relay, workers file,
hot add / remove / swap without dropping requests, partial start, limits, config."""
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time

import pytest
from aiohttp import web

from reax_subnet import multinode as mn
from reax_subnet.config import ConfigError, load_miner
from reax_subnet.miner import S1Miner
from reax_subnet.s1backends import PooledBackend, S1HashBackend, SidecarBackend, SidecarError
from reax_subnet.s1contract import Decision, parse_decision, parse_question
from reax_subnet.selfcheck import check_backend_identity

TOKEN = "s" * 32
KEY_HEX = "ab" * 32


def decision(i: int = 0):
    return parse_decision({"id": f"d{i}", "state": f"Probe {i}.",
                           "questions": {"q": {"type": "noul", "instructions": "Late?"}}}, "s1-fast")


# --- link signatures -------------------------------------------------------------------------------------------------

def test_link_signature_roundtrip_and_refusals():
    key = bytes.fromhex(KEY_HEX)
    nonces = mn.NonceCache()
    h = mn.sign_link(key, "POST", "/internal/v1/decide", b"{}")
    mn.verify_link(key, h, "POST", "/internal/v1/decide", b"{}", nonces)
    with pytest.raises(mn.LinkError, match="replayed"):
        mn.verify_link(key, h, "POST", "/internal/v1/decide", b"{}", nonces)
    cases = [
        (bytes.fromhex("cd" * 32), mn.sign_link(key, "POST", "/p", b"x"), "POST", "/p", b"x", "bad"),  # other key
        (key, mn.sign_link(key, "POST", "/p", b"x"), "POST", "/p", b"y", "bad"),  # body changed
        (key, mn.sign_link(key, "POST", "/p", b"x"), "POST", "/q", b"x", "bad"),  # path changed
        (key, mn.sign_link(key, "GET", "/p"), "POST", "/p", b"", "bad"),  # method changed
        (key, mn.sign_link(key, "POST", "/p", b"", now=time.time() - 11), "POST", "/p", b"", "stale"),
        (key, mn.sign_link(key, "POST", "/p", b"", now=time.time() + 3), "POST", "/p", b"", "stale"),
        (key, None, "POST", "/p", b"", "missing"),
        (key, "1.2", "POST", "/p", b"", "missing"),
        (key, "12.zz.ff.ee", "POST", "/p", b"", "malformed"),
        (key, (lambda h: h[:-1] + ("1" if h[-1] == "0" else "0"))(mn.sign_link(key, "POST", "/p", b"x")),
         "POST", "/p", b"x", "bad"),
    ]
    for k, header, method, path, body, msg in cases:
        with pytest.raises(mn.LinkError, match=msg):
            mn.verify_link(k, header, method, path, body, mn.NonceCache())


def test_nonce_cache_expires_and_refuses_when_flooded():
    c = mn.NonceCache(retention_s=30, max_size=3)
    assert c.add("a", 0) and not c.add("a", 1)
    assert c.add("b", 1) and c.add("c", 2)
    assert not c.add("d", 3)  # full of live nonces: refuse rather than forget
    assert c.add("d", 31)  # "a" expired
    assert c.add("a", 40)


def test_read_key(tmp_path):
    p = tmp_path / "k"
    p.write_text(KEY_HEX + "\n")
    assert mn.read_key(str(p)) == bytes.fromhex(KEY_HEX)
    p.write_text("ab" * 16)
    with pytest.raises(ValueError, match="at least 32"):
        mn.read_key(str(p))
    p.write_text("not hex")
    with pytest.raises(ValueError, match="hex"):
        mn.read_key(str(p))


# --- workers file ------------------------------------------------------------------------------------------------

@pytest.fixture
def keyfile(tmp_path):
    p = tmp_path / "gpu-a.key"
    p.write_text(KEY_HEX)
    return str(p)


@pytest.fixture
def keyfiles(tmp_path):
    """A different link key file per worker name."""
    def make(name):
        p = tmp_path / f"{name}.key"
        p.write_text(hashlib.sha256(name.encode()).hexdigest())
        return str(p)
    return make


def doc(*workers, model="s1-fast"):
    return {"model": model, "workers": list(workers)}


def test_parse_workers_accepts_tls_with_fingerprint_and_vpn_http(keyfiles):
    fp = "AA:" * 31 + "AA"
    ws = mn.parse_workers(doc({"name": "gpu-a", "url": "https://203.0.113.5:8190/", "key_file": keyfiles("a"),
                               "fingerprint": fp},
                              {"name": "gpu-b", "url": "http://10.8.0.3:8190", "key_file": keyfiles("b"),
                               "max_inflight": 32},
                              {"name": "gpu-c", "url": "http://100.101.1.2:8190", "key_file": keyfiles("c")}),
                          "s1-fast")
    assert [w.url for w in ws] == ["https://203.0.113.5:8190", "http://10.8.0.3:8190", "http://100.101.1.2:8190"]
    assert ws[0].fingerprint == "aa" * 32 and ws[1].max_inflight == 32


@pytest.mark.parametrize("worker,msg", [
    ({"name": "a", "url": "https://203.0.113.5:8190", "key_file": "K"}, "fingerprint"),
    ({"name": "a", "url": "http://203.0.113.5:8190", "key_file": "K"}, "private or VPN"),
    ({"name": "a", "url": "http://gpu.example.com:8190", "key_file": "K"}, "private or VPN"),
    ({"name": "a", "url": "http://10.0.0.2:8190/x", "key_file": "K"}, "must look like"),
    ({"name": "a", "url": "http://user@10.0.0.2:8190", "key_file": "K"}, "must look like"),
    ({"name": "a", "url": "ftp://10.0.0.2:8190", "key_file": "K"}, "must look like"),
    ({"name": "a b", "url": "http://10.0.0.2:8190", "key_file": "K"}, "name"),
    ({"name": "a", "url": "http://10.0.0.2:8190", "key_file": "/nonexistent"}, "not readable"),
    ({"name": "a", "url": "http://10.0.0.2:8190", "key_file": "K", "max_inflight": 0}, "max_inflight"),
    ({"name": "a", "url": "http://10.0.0.2:8190", "key_file": "K", "token": "x"}, "unknown"),
    ({"name": "a", "url": "http://10.0.0.2:8190", "key_file": "K", "fingerprint": "aa" * 32}, "only applies"),
])
def test_parse_workers_refuses(keyfile, worker, msg):
    w = {k: (keyfile if v == "K" else v) for k, v in worker.items()}
    with pytest.raises(ValueError, match=msg):
        mn.parse_workers(doc(w), "s1-fast")


def test_parse_workers_document_rules(keyfile, keyfiles):
    w = {"name": "a", "url": "http://10.0.0.2:8190", "key_file": keyfile}
    with pytest.raises(ValueError, match="share one link key"):
        mn.parse_workers(doc(w, {**w, "name": "b", "url": "http://10.0.0.3:8190"}), "s1-fast")
    with pytest.raises(ValueError, match="for model"):
        mn.parse_workers(doc(w, model="s1-pro"), "s1-fast")
    with pytest.raises(ValueError, match="no workers"):
        mn.parse_workers(doc(), "s1-fast")
    with pytest.raises(ValueError, match="name twice"):
        mn.parse_workers(doc(w, {**w, "url": "http://10.0.0.3:8190", "key_file": keyfiles("x")}), "s1-fast")
    with pytest.raises(ValueError, match="same URL"):
        mn.parse_workers(doc(w, {**w, "name": "b", "key_file": keyfiles("x")}), "s1-fast")
    with pytest.raises(ValueError, match="at most"):
        mn.parse_workers(doc(*[{**w, "name": f"w{i}", "url": f"http://10.0.0.{i}:8190", "key_file": keyfiles(f"w{i}")}
                               for i in range(65)]), "s1-fast")


def test_relative_key_file_is_relative_to_the_workers_file(tmp_path):
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "a.key").write_text(KEY_HEX)
    f = tmp_path / "workers.json"
    f.write_text(json.dumps(doc({"name": "a", "url": "http://10.0.0.2:8190", "key_file": "keys/a.key"})))
    (w,) = mn.load_workers_file(str(f), "s1-fast")
    assert w.key_file == str(tmp_path / "keys" / "a.key")


# --- a worker: hash-model sidecar + relay ------------------------------------------------------------------------

class HashSidecar:
    """A loopback runtime sidecar answering with the hash test model (the validators' ``hash`` reference)."""

    def __init__(self, salt="reference", delay=0.0):
        self.model = S1HashBackend("s1-fast", salt=salt)
        self.delay = delay
        self.decides = 0
        self.healthy = True

    async def start(self):
        async def health(_r):
            return web.json_response({"ok": True}, status=200 if self.healthy else 503)

        async def decide(request):
            if request.headers.get("X-S1M-Internal-Token") != TOKEN:
                return web.json_response({"error": "unauthorized"}, status=401)
            body = await request.json()
            (qid, q), = body["questions"].items()
            d = Decision(qid, body["state"], parse_question(qid, q, "s1-fast"))
            if self.delay:
                await asyncio.sleep(self.delay)
            (a,) = await self.model.decide([d])
            self.decides += 1
            return web.json_response({"answers": {qid: a}})

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_post("/internal/v1/decide", decide)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    async def stop(self):
        await self.runner.cleanup()


class Worker:
    """One worker machine: sidecar + reax-runtime-relay (optionally with TLS)."""

    def __init__(self, name, key_hex=None, tls=None, salt="reference", delay=0.0):
        self.name, self.tls = name, tls
        self.key_hex = key_hex or hashlib.sha256(name.encode()).hexdigest()  # every worker its own key
        self.sidecar = HashSidecar(salt, delay)
        import tempfile

        self.key_file = os.path.join(tempfile.mkdtemp(prefix="reax-mn-"), f"{name}.key")
        with open(self.key_file, "w") as fh:
            fh.write(self.key_hex)

    async def start(self):
        await self.sidecar.start()
        cfg = mn.RelayConfig("127.0.0.1", 0, bytes.fromhex(self.key_hex), self.sidecar.url, TOKEN, "s1-fast",
                             *(self.tls or ("", "")))
        self.runner = web.AppRunner(mn.build_relay_app(cfg), access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0, ssl_context=mn.relay_ssl_context(cfg))
        await site.start()
        scheme = "https" if self.tls else "http"
        self.url = f"{scheme}://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    async def stop(self):
        if self.runner is not None:
            await self.runner.cleanup()
            await self.sidecar.stop()
            self.runner = None


@pytest.fixture
async def workers():
    made = []

    async def make(*a, **kw):
        w = await Worker(*a, **kw).start()
        made.append(w)
        return w

    yield make
    for w in made:
        await w.stop()


@pytest.fixture
def tls_cert(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl not available")
    cert, key = tmp_path / "relay.crt", tmp_path / "relay.key"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                    "-nodes", "-days", "2", "-subj", "/CN=reax-relay", "-keyout", str(key), "-out", str(cert)],
                   check=True, capture_output=True)
    return str(cert), str(key)


def link_backend(w, key_hex=None, fingerprint="", **kw):
    return SidecarBackend("s1-fast", w.url, "", link_key=bytes.fromhex(key_hex or w.key_hex), fingerprint=fingerprint,
                          name=w.name, timeout_s=5, **kw)


async def test_relay_answers_signed_requests_like_the_sidecar(workers):
    w = await workers("gpu-a")
    b = link_backend(w)
    try:
        assert await b.ready()
        got = await b.decide([decision(1), decision(2)])
        assert got == await S1HashBackend("s1-fast").decide([decision(1), decision(2)])
        assert await check_backend_identity(b, "s1-fast", reference_backend=S1HashBackend("s1-fast")) == []
    finally:
        await b.close()


async def test_relay_refuses_a_wrong_key_and_unsigned_requests(workers):
    import aiohttp

    w = await workers("gpu-a")
    b = link_backend(w, key_hex="cd" * 32)
    try:
        assert not await b.ready()
        with pytest.raises(SidecarError) as ei:
            await b.decide([decision()])
        assert ei.value.answer_started is False
        async with aiohttp.ClientSession() as s:
            async with s.post(w.url + "/internal/v1/decide", data=b"{}",
                              headers={"X-S1M-Internal-Token": TOKEN}) as r:  # the sidecar token is not a link key
                assert r.status == 401
            async with s.get(w.url + "/health") as r:
                assert r.status == 401
        assert w.sidecar.decides == 0
    finally:
        await b.close()


async def test_relay_health_reports_its_model_and_the_gateway_checks_it(workers):
    w = await workers("gpu-a")
    b = link_backend(w)
    try:
        assert await b.ready()
        b2 = SidecarBackend("s1-pro", w.url, "", link_key=bytes.fromhex(w.key_hex), name="x", timeout_s=5)
        assert not await b2.ready()  # the relay serves s1-fast: an s1-pro gateway must not use it
        await b2.close()
        w.sidecar.healthy = False
        assert not await b.ready()
    finally:
        await b.close()


async def test_tls_link_with_pinned_fingerprint(workers, tls_cert):
    w = await workers("gpu-a", tls=tls_cert)
    fp = mn.cert_fingerprint(tls_cert[0])
    good = link_backend(w, fingerprint=fp)
    bad = link_backend(w, fingerprint="00" * 32)
    try:
        assert await good.ready()
        assert len(await good.decide([decision(3)])) == 1
        assert not await bad.ready()  # another certificate (a man in the middle) is refused
        with pytest.raises(Exception):
            await bad.decide([decision(3)])
    finally:
        await good.close()
        await bad.close()
    with pytest.raises(ValueError, match="fingerprint"):
        SidecarBackend("s1-fast", "https://127.0.0.1:1", "", link_key=b"k" * 32)


def test_relay_config(tmp_path):
    tok = tmp_path / "tok"
    tok.write_text(TOKEN)
    key = tmp_path / "key"
    key.write_text(KEY_HEX)
    base = {"REAX_MODEL": "s1-fast", "REAX_S1_FAST_RUNTIME_URL": "http://127.0.0.1:8101",
            "REAX_RUNTIME_TOKEN_FILE": str(tok), "REAX_RELAY_KEY_FILE": str(key), "REAX_RELAY_HOST": "10.8.0.2"}
    cfg = mn.load_relay_config(base)
    assert (cfg.host, cfg.port, cfg.sidecar_token) == ("10.8.0.2", mn.DEFAULT_RELAY_PORT, TOKEN)
    for extra, msg in [({"REAX_RELAY_HOST": "0.0.0.0"}, "without TLS"),
                       ({"REAX_RELAY_HOST": "203.0.113.5"}, "without TLS"),
                       ({"REAX_S1_FAST_RUNTIME_URL": "http://10.0.0.5:8101"}, "loopback"),
                       ({"REAX_MODEL": "s1-lite"}, "one machine per hotkey"),
                       ({"REAX_RELAY_TLS_CERT": "/x"}, "both"),
                       ({"REAX_RELAY_KEY_FILE": str(tok)}, "REAX_RELAY_KEY_FILE"),
                       ({"REAX_RELAY_HOST": ""}, "REAX_RELAY_HOST")]:
        with pytest.raises(ConfigError, match=msg):
            mn.load_relay_config({**base, **extra})
    cfg = mn.load_relay_config({**base, "REAX_RELAY_HOST": "0.0.0.0", "REAX_RELAY_TLS_CERT": "/c",
                                "REAX_RELAY_TLS_KEY": "/k"})
    assert cfg.tls_cert == "/c"


# --- the gateway's pool: start, add, remove, swap ------------------------------------------------------------------

def spec_for(w, keyfile=None, **kw):
    return mn.WorkerSpec(w.name, w.url, keyfile or w.key_file, **kw)


def maker(fingerprints=None):
    def make(sp):
        return SidecarBackend("s1-fast", sp.url, "", max_inflight=sp.max_inflight,
                              link_key=mn.read_key(sp.key_file), fingerprint=sp.fingerprint, name=sp.name, timeout_s=5)
    return make


def identity_check():
    ref = S1HashBackend("s1-fast")
    return lambda b: check_backend_identity(b, "s1-fast", reference_backend=ref)


async def test_partial_start_keeps_failing_workers_out(workers, keyfile):
    a = await workers("gpu-a")
    wrong = await workers("gpu-wrong", salt="other-weights")  # answers differ from the reference model
    dead = mn.WorkerSpec("gpu-dead", "http://127.0.0.1:9", keyfile)
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(wrong), dead], maker(),
                                      health_interval_s=0.05)
    pool.readmit_check = identity_check()
    try:
        assert await pool.startup(1) == 1
        assert [d["healthy"] for d in pool.describe()] == [True, False, False]
        await pool.decide([decision(i) for i in range(8)])
        # the wrong worker got only the start-up test decision and the 24 identity items, no validator traffic
        assert a.sidecar.decides == 1 + 24 + 8 and wrong.sidecar.decides == 1 + 24
        for _ in range(4):
            await pool.check_health_once()
        assert pool.healthy_count == 1  # the wrong-weights worker stays out however healthy it looks
    finally:
        await pool.close()


async def test_added_worker_takes_traffic_only_after_health_and_identity(workers, keyfile):
    a = await workers("gpu-a")
    b = await workers("gpu-b")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a)], maker())
    pool.readmit_check = identity_check()
    try:
        assert await pool.startup(1) == 1
        assert await pool.apply([spec_for(a), spec_for(b)], maker())
        assert not await pool.apply([spec_for(a), spec_for(b)], maker())  # unchanged: no-op
        assert (pool.size, pool.healthy_count) == (2, 1)
        await pool.decide([decision(i) for i in range(16)])
        assert b.sidecar.decides == 0
        await pool.check_health_once()
        assert pool.healthy_count == 1
        await pool.check_health_once()  # second good health answer -> identity check -> in rotation
        assert pool.healthy_count == 2 and b.sidecar.decides == 24
        await pool.decide([decision(i) for i in range(16)])
        assert b.sidecar.decides > 24
    finally:
        await pool.close()


async def test_removed_worker_is_drained_not_cut(workers, keyfile):
    a = await workers("gpu-a")
    b = await workers("gpu-b", delay=0.3)
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(b)], maker())
    try:
        assert await pool.startup(1) == 2
        task = asyncio.ensure_future(pool.decide([decision(i) for i in range(8)]))
        await asyncio.sleep(0.1)  # some decisions are in flight on the slow worker b
        old_b = next(m.backend for m in pool._members if m.backend.name == "gpu-b")
        assert old_b.inflight > 0
        assert await pool.apply([spec_for(a)], maker())
        assert len(await task) == 8  # the in-flight decisions on b still finished
        before = b.sidecar.decides
        await pool.decide([decision(i) for i in range(8)])
        assert b.sidecar.decides == before and pool.size == 1
    finally:
        await pool.close()


async def test_a_wrong_edit_of_the_only_worker_keeps_it_serving(workers, tmp_path):
    """Make before break: changing the key, URL or limit of a worker is remove + add, but the old entry keeps serving
    until its same-name replacement is in rotation, so a valid but wrong edit cannot take the hotkey offline."""
    a = await workers("gpu-a")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a)], maker())
    pool.readmit_check = identity_check()
    try:
        await pool.startup(1)
        new_key = tmp_path / "new.key"
        new_key.write_text("cd" * 32)  # the relay still has the old key: the replacement can never pass
        typo = mn.WorkerSpec(a.name, "http://127.0.0.1:9", a.key_file)
        for wrong in (spec_for(a, str(new_key)), typo):
            assert await pool.apply([wrong], maker())
            for _ in range(3):
                await pool.check_health_once()
            assert pool.healthy_count == 1 and len(await pool.decide([decision(i) for i in range(8)])) == 8
            assert await pool.apply([spec_for(a)], maker())  # the operator fixes the file
            assert (pool.size, pool.healthy_count) == (1, 1)  # the failed replacement is dropped at once
            assert [d["url"] for d in pool.describe()] == [a.url]
        # a correct edit (higher limit) admits the new entry, then retires the old one
        assert await pool.apply([spec_for(a, max_inflight=32)], maker())
        await pool.check_health_once()
        await pool.check_health_once()
        assert [d["max_inflight"] for d in pool.describe()] == [32] and pool.healthy_count == 1
    finally:
        await pool.close()


async def test_removal_keeps_min_healthy_in_rotation(workers):
    a = await workers("gpu-a")
    b = await workers("gpu-b")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(b)], maker())
    pool.readmit_check = identity_check()
    pool.min_healthy = 2
    try:
        await pool.startup(2)
        c_spec = mn.WorkerSpec("gpu-c", "http://127.0.0.1:9", b.key_file + ".c")
        with open(c_spec.key_file, "w") as fh:
            fh.write("ef" * 32)
        assert await pool.apply([spec_for(a), c_spec], maker())  # b removed, its replacement c is dead
        assert pool.healthy_count == 2  # b stays: otherwise fewer than REAX_WORKERS_MIN_HEALTHY would serve
        await pool.check_health_once()
        assert [d["url"] for d in pool.describe() if d["healthy"]] == [a.url, b.url]
    finally:
        await pool.close()


async def test_machine_swap_under_load_loses_no_request(workers, keyfile, tmp_path):
    """Charlie's case: machine A is replaced by machine C while validators keep sending. With B serving throughout,
    no request fails and the UID never stops answering."""
    a = await workers("gpu-a", delay=0.01)
    b = await workers("gpu-b", delay=0.01)
    wf = tmp_path / "workers.json"

    def write(*ws):
        wf.write_text(json.dumps(doc(*[{"name": w.name, "url": w.url, "key_file": w.key_file} for w in ws])))

    write(a, b)
    specs = mn.load_workers_file(str(wf), "s1-fast")
    pool = PooledBackend.from_workers("s1-fast", specs, maker(), health_interval_s=0.05)
    pool.readmit_check = identity_check()
    watcher = mn.WorkersWatcher(str(wf), "s1-fast", pool, maker(), interval_s=0.05)
    failures, answered = [], [0]
    stop = asyncio.Event()

    async def validator_traffic():
        i = 0
        while not stop.is_set():
            try:
                answered[0] += len(await pool.decide([decision(i + k) for k in range(8)]))
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)
            i += 8
            await asyncio.sleep(0.005)

    try:
        assert await pool.startup(1) == 2
        await pool.start()
        watcher.start()
        traffic = [asyncio.ensure_future(validator_traffic()) for _ in range(4)]
        await asyncio.sleep(0.3)
        c = await workers("gpu-c", delay=0.01)
        write(a, b, c)  # 1. add the new machine
        for _ in range(100):
            await asyncio.sleep(0.05)
            if pool.healthy_count == 3:
                break
        await asyncio.sleep(0.3)
        assert pool.healthy_count == 3 and c.sidecar.decides > 24  # the new machine serves validator traffic
        write(b, c)  # 2. remove the old one
        await asyncio.sleep(0.3)
        await a.stop()  # 3. the old machine is switched off
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.gather(*traffic)
        assert failures == [] and answered[0] > 200
        assert [d["url"] for d in pool.describe()] == [b.url, c.url]
    finally:
        watcher.stop()
        await pool.close()


async def test_watcher_keeps_workers_on_a_broken_file(workers, keyfile, tmp_path, caplog):
    a = await workers("gpu-a")
    wf = tmp_path / "workers.json"
    wf.write_text(json.dumps(doc({"name": "gpu-a", "url": a.url, "key_file": a.key_file})))
    pool = PooledBackend.from_workers("s1-fast", mn.load_workers_file(str(wf), "s1-fast"), maker())
    watcher = mn.WorkersWatcher(str(wf), "s1-fast", pool, maker())
    try:
        await pool.startup(1)
        assert not await watcher.check_once()  # the same workers: nothing changes
        wf.write_text("{not json")
        assert not await watcher.check_once()
        wf.write_text(json.dumps(doc()))  # an empty list is refused, the UID keeps its worker
        assert not await watcher.check_once()
        assert pool.size == 1 and pool.healthy_count == 1
        assert "not applied" in caplog.text
    finally:
        await pool.close()


async def test_a_dead_worker_leaves_rotation_and_the_rest_serve(workers, keyfile):
    a = await workers("gpu-a")
    b = await workers("gpu-b")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(b)], maker())
    pool.readmit_check = identity_check()
    try:
        await pool.startup(2)
        await b.stop()
        for i in range(4):  # requests that hit b fail over to a before any answer byte
            assert len(await pool.decide([decision(8 * i + k) for k in range(8)])) == 8
        workers_out = [d["healthy"] for d in pool.describe()]
        assert workers_out == [True, False]
    finally:
        await pool.close()


# --- limits and config ---------------------------------------------------------------------------------------------

class _Stub:
    model = "s1-fast"

    def __init__(self, size, healthy, nominal):
        self.size, self.healthy_count, self.nominal_size = size, healthy, nominal


def _miner(pool):
    m = S1Miner.__new__(S1Miner)
    m.backend, m._per_caller, m._per_minute = pool, 64, 900
    return m


def test_limits_follow_workers_in_rotation_against_the_start_count():
    assert _miner(_Stub(2, 2, 2))._caller_limits() == (64, 900)
    assert _miner(_Stub(3, 2, 2))._caller_limits() == (64, 900)  # a joining worker during a swap does not shrink them
    assert _miner(_Stub(2, 1, 2))._caller_limits() == (32, 450)
    assert _miner(_Stub(4, 4, 2))._caller_limits() == (64, 900)  # never above what was configured


@pytest.fixture
def mn_env(tmp_path, keyfiles):
    wf = tmp_path / "workers.json"
    wf.write_text(json.dumps(doc({"name": "a", "url": "http://10.8.0.2:8190", "key_file": keyfiles("a")},
                                 {"name": "b", "url": "http://10.8.0.3:8190", "key_file": keyfiles("b")})))
    return {"REAX_NETUID": "7", "REAX_FEATURE_MULTINODE": "1", "REAX_WORKERS_FILE": str(wf), "REAX_MODEL": "s1-fast"}


def test_config_flag_off_is_unchanged(mn_env, tmp_path):
    tok = tmp_path / "tok"
    tok.write_text(TOKEN)
    env = {"REAX_NETUID": "7", "REAX_S1_FAST_RUNTIME_URL": "http://127.0.0.1:8101", "REAX_RUNTIME_TOKEN_FILE": str(tok),
           "REAX_WORKERS_FILE": mn_env["REAX_WORKERS_FILE"]}
    cfg = load_miner(env)
    assert not cfg.multinode and cfg.workers == () and cfg.per_caller_concurrency == load_miner(
        {k: v for k, v in env.items() if k != "REAX_WORKERS_FILE"}).per_caller_concurrency


def test_config_multinode(mn_env):
    cfg = load_miner(mn_env)
    assert cfg.multinode and [w.name for w in cfg.workers] == ["a", "b"] and cfg.workers_min_healthy == 1
    single = load_miner({**mn_env, "REAX_WORKERS_FILE": ""} | {"REAX_FEATURE_MULTINODE": "0",
                         "REAX_S1_FAST_RUNTIME_URLS": "http://10.8.0.2:8101,http://10.8.0.3:8101",
                         "REAX_RUNTIME_TOKEN_FILE": cfg.workers[0].key_file})
    assert (cfg.max_concurrent, cfg.per_caller_concurrency, cfg.per_caller_per_minute) == (
        single.max_concurrent, single.per_caller_concurrency, single.per_caller_per_minute)  # same defaults as 2 URLs
    assert load_miner({**mn_env, "REAX_WORKERS_MIN_HEALTHY": "2"}).workers_min_healthy == 2
    for extra, msg in [({"REAX_WORKERS_MIN_HEALTHY": "3"}, "REAX_WORKERS_MIN_HEALTHY"),
                       ({"REAX_MODEL": "s1-lite"}, "one machine per hotkey"),
                       ({"REAX_MODEL": "s1-cpu"}, "one machine per hotkey"),
                       ({"REAX_WORKERS_FILE": ""}, "needs REAX_WORKERS_FILE"),
                       ({"REAX_WORKERS_FILE": "/nonexistent.json"}, "not readable"),
                       ({"REAX_MODEL": "s1-pro"}, "for model"),
                       ({"REAX_S1_FAST_RUNTIME_URL": "http://127.0.0.1:8101",
                         "REAX_RUNTIME_TOKEN_FILE": mn_env["REAX_WORKERS_FILE"]}, "unset")]:
        with pytest.raises(ConfigError, match=msg):
            load_miner({**mn_env, **extra})


# --- critic round 1: integrity, failover, key rotation, flag-off headers ---------------------------------------------

class TamperingProxy:
    """A man in the middle on an unencrypted private network: forwards to the relay, rewrites the answer."""

    def __init__(self, target):
        self.target = target

    async def start(self):
        import aiohttp

        async def any_(request):
            body = await request.read()
            hdrs = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
            async with aiohttp.ClientSession() as s:
                async with s.request(request.method, self.target + request.path, data=body, headers=hdrs) as r:
                    raw = await r.read()
                    out = {k: v for k, v in r.headers.items() if k.lower().startswith("x-reax")}
            raw = raw.replace(b'"true": 0.', b'"true": 0.0').replace(b'"noul": 0.', b'"noul": 0.0') + b" "
            return web.Response(body=raw, status=r.status, headers=out, content_type="application/json")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", any_)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self


async def test_a_changed_answer_on_the_way_is_refused(workers):
    w = await workers("gpu-a")
    proxy = await TamperingProxy(w.url).start()
    b = SidecarBackend("s1-fast", proxy.url, "", link_key=bytes.fromhex(w.key_hex), name="via-proxy", timeout_s=5)
    try:
        with pytest.raises(SidecarError, match="signature does not match") as ei:
            await b.decide([decision(1)])
        assert ei.value.backend_fault and not ei.value.answer_started  # fails over to another worker
        assert not await b.ready()  # the health answer is signed too
    finally:
        await b.close()
        await proxy.runner.cleanup()
    with pytest.raises(mn.LinkError, match="not signed"):
        mn.verify_reply(b"k" * 32, "x", 200, b"{}", None)


async def test_a_worker_with_a_wrong_key_fails_over_to_the_others(workers, tmp_path):
    a = await workers("gpu-a")
    b = await workers("gpu-b")
    wrong = tmp_path / "wrong.key"
    wrong.write_text("ee" * 32)
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(b, str(wrong))], maker())
    try:
        await pool.startup(1)
        pool._members[1].healthy = True  # e.g. the relay was restarted with a new key after the gateway admitted it
        for i in range(3):
            assert len(await pool.decide([decision(8 * i + k) for k in range(8)])) == 8  # 401 -> failover
        assert [d["healthy"] for d in pool.describe()] == [True, False]
    finally:
        await pool.close()


async def test_a_key_rewritten_in_place_is_noticed(workers, tmp_path):
    a = await workers("gpu-a")
    wf = tmp_path / "workers.json"
    wf.write_text(json.dumps(doc({"name": a.name, "url": a.url, "key_file": a.key_file})))
    pool = PooledBackend.from_workers("s1-fast", mn.load_workers_file(str(wf), "s1-fast"), maker())
    watcher = mn.WorkersWatcher(str(wf), "s1-fast", pool, maker())
    try:
        await pool.startup(1)
        assert not await watcher.check_once()
        with open(a.key_file, "w") as fh:
            fh.write("12" * 32)  # key rotation: same file name, new content
        assert await watcher.check_once()
        assert pool.size == 2  # the new key is a replacement; the old entry serves until it is in rotation
    finally:
        await pool.close()


async def test_late_success_does_not_readmit_a_removed_backend(workers):
    a = await workers("gpu-a", delay=0.3)
    b = await workers("gpu-b")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(b)], maker())
    pool.readmit_check = identity_check()
    try:
        await pool.startup(2)
        task = asyncio.ensure_future(pool.decide([decision(k) for k in range(8)]))
        await asyncio.sleep(0.1)
        pool._mark_bad(pool._members[0])
        pool._mark_bad(pool._members[0])  # removed while its requests are still in flight
        assert len(await task) == 8
        assert [d["healthy"] for d in pool.describe()] == [False, True]
    finally:
        await pool.close()


async def test_relay_reports_a_sidecar_token_mismatch_as_unhealthy(workers):
    w = await workers("gpu-a")
    b = link_backend(w)
    try:
        assert await b.ready()
        w_token = TOKEN
        globals()["TOKEN"] = "other-token-" + "x" * 20  # the sidecar's token was rotated, the relay's was not
        try:
            with pytest.raises(SidecarError) as ei:
                await b.decide([decision(1)])
            assert ei.value.backend_fault
            assert not await b.ready()
        finally:
            globals()["TOKEN"] = w_token
    finally:
        await b.close()


async def test_relay_refuses_a_stale_signature_before_reading_the_body(workers):
    import aiohttp

    w = await workers("gpu-a")
    stale = mn.sign_link(bytes.fromhex(w.key_hex), "POST", "/internal/v1/decide", b"x", now=time.time() - 60)
    async with aiohttp.ClientSession() as s:
        async with s.post(w.url + "/internal/v1/decide", data=b"x" * 1000, headers={mn.LINK_HEADER: stale}) as r:
            assert r.status == 401
    assert w.sidecar.decides == 0


async def test_flag_off_health_check_sends_no_headers():
    seen = []

    async def health(request):
        seen.append({k.lower() for k in request.headers})
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/health", health)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    b = SidecarBackend("s1-fast", f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", TOKEN)
    try:
        assert await b.ready()
        assert "x-s1m-internal-token" not in seen[0] and mn.LINK_HEADER.lower() not in seen[0]
    finally:
        await b.close()
        await runner.cleanup()


# --- critic round 2: pre-auth slots, retirement edge cases, eviction, reload race --------------------------------------

async def test_forged_signatures_take_no_body_slot(workers, monkeypatch):
    """An outsider without the key cannot hold the relay's unfinished-body slots: the signature (with the signed body
    hash) is checked before any body is read."""
    monkeypatch.setattr(mn, "RELAY_MAX_PENDING_TOTAL", 1)
    w = await workers("gpu-a")
    host, port = w.url.split("//")[1].split(":")
    forged = f"{int(time.time() * 1000)}.{'ab' * 12}.{'cd' * 32}.{'ef' * 32}"  # fresh-looking, wrong MAC
    readers = []
    for _ in range(3):  # three slow-drip clients that never send their body
        r, wr = await asyncio.open_connection(host, int(port))
        wr.write(f"POST /internal/v1/decide HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n{mn.LINK_HEADER}: {forged}\r\n"
                 "Content-Type: application/json\r\n\r\n".encode())
        await wr.drain()
        readers.append((r, wr))
    for r, wr in readers:
        status = await asyncio.wait_for(r.readline(), 2)
        assert b" 401 " in status  # refused at once, before any body byte
        wr.close()
    b = link_backend(w)
    try:
        assert len(await b.decide([decision(1)])) == 1  # the real gateway still gets through
    finally:
        await b.close()


async def test_a_leaving_worker_with_a_blip_waits_for_its_broken_replacement(workers):
    a = await workers("gpu-a")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a)], maker())
    pool.readmit_check = identity_check()
    try:
        await pool.startup(1)
        typo = mn.WorkerSpec(a.name, "http://127.0.0.1:9", a.key_file)
        assert await pool.apply([typo], maker())
        old = pool._members[0]
        pool._mark_bad(old)
        pool._mark_bad(old)  # a transient fault of the old worker while its replacement can never pass
        await pool.check_health_once()
        assert old in pool._members  # kept (health-checked), not retired
        await pool.check_health_once()
        await pool.check_health_once()
        assert old.healthy and pool.healthy_count == 1  # back after two good checks + identity
    finally:
        await pool.close()


async def test_scaling_down_below_min_healthy_still_retires(workers):
    a, b, c = await workers("gpu-a"), await workers("gpu-b"), await workers("gpu-c")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(b), spec_for(c)], maker())
    pool.min_healthy = 2
    try:
        await pool.startup(3)
        assert await pool.apply([spec_for(a)], maker())  # 3 -> 1 entries with REAX_WORKERS_MIN_HEALTHY 2
        assert [d["url"] for d in pool.describe()] == [a.url]
    finally:
        await pool.close()


async def test_evict_takes_a_worker_out_at_once(workers, tmp_path):
    a, b = await workers("gpu-a"), await workers("gpu-b")
    pool = PooledBackend.from_workers("s1-fast", [spec_for(a), spec_for(b)], maker())
    pool.min_healthy = 2
    try:
        await pool.startup(2)
        assert await pool.apply([spec_for(a), mn.WorkerSpec(b.name, "", "", evict=True)], maker())
        assert [d["url"] for d in pool.describe()] == [a.url]  # no min-healthy hold for an eviction
        before = b.sidecar.decides
        await pool.decide([decision(i) for i in range(8)])
        assert b.sidecar.decides == before
    finally:
        await pool.close()
    ws = mn.parse_workers(doc({"name": "a", "url": "http://10.0.0.2:8190", "key_file": a.key_file},
                              {"name": "b", "evict": True}), "s1-fast")
    assert [w.evict for w in ws] == [False, True]
    for bad in ({"name": "b", "evict": False}, {"name": "b", "evict": True, "url": "http://10.0.0.3:8190"}):
        with pytest.raises(ValueError, match="eviction entry"):
            mn.parse_workers(doc({"name": "a", "url": "http://10.0.0.2:8190", "key_file": a.key_file}, bad),
                             "s1-fast")
    with pytest.raises(ValueError, match="no workers"):
        mn.parse_workers(doc({"name": "b", "evict": True}), "s1-fast")


async def test_a_file_written_while_it_is_applied_is_applied_next_pass(workers, tmp_path, monkeypatch):
    a, b = await workers("gpu-a"), await workers("gpu-b")
    wf = tmp_path / "workers.json"
    version = lambda *ws: json.dumps(doc(*[{"name": w.name, "url": w.url, "key_file": w.key_file} for w in ws]))
    wf.write_text(version(a))
    pool = PooledBackend.from_workers("s1-fast", mn.load_workers_file(str(wf), "s1-fast"), maker())
    watcher = mn.WorkersWatcher(str(wf), "s1-fast", pool, maker())
    real = mn.load_workers_file

    def racing(path, model, raw=None):
        out = real(path, model, raw)
        wf.write_text(version(a, b))  # the operator writes the next version right now
        monkeypatch.setattr(mn, "load_workers_file", real)
        return out

    try:
        await pool.startup(1)
        monkeypatch.setattr(mn, "load_workers_file", racing)
        await watcher.check_once()
        assert pool.size == 1
        assert await watcher.check_once()  # the version written during the first pass is not lost
        assert pool.size == 2
    finally:
        await pool.close()


async def test_a_copied_header_with_another_body_does_not_spend_the_nonce(workers):
    """Critic round 3: an on-path observer (plain http) races a seen header with a junk body; the real request with
    that header must still be accepted."""
    import aiohttp

    w = await workers("gpu-a")
    body = json.dumps({"request_id": "r1", "deadline_unix_ms": int(time.time() * 1000) + 5000, "state": "Probe 1.",
                       "questions": {"q": {"type": "noul", "instructions": "Late?"}}}).encode()
    header = mn.sign_link(bytes.fromhex(w.key_hex), "POST", "/internal/v1/decide", body)
    async with aiohttp.ClientSession() as s:
        async with s.post(w.url + "/internal/v1/decide", data=b'{"junk": 1}', headers={mn.LINK_HEADER: header}) as r:
            assert r.status == 401
        async with s.post(w.url + "/internal/v1/decide", data=body, headers={mn.LINK_HEADER: header}) as r:
            assert r.status == 200
        async with s.post(w.url + "/internal/v1/decide", data=body, headers={mn.LINK_HEADER: header}) as r:
            assert r.status == 401  # an exact replay is still refused
