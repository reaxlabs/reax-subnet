"""Several worker machines behind one miner hotkey (``REAX_FEATURE_MULTINODE=1``; docs/DESIGN-MULTINODE.md).

One *gateway* (``reax-miner``) holds the hotkey, answers the validators and signs every response. Each *worker* machine
runs the pinned runtime sidecar on loopback, exactly as a single-machine miner does, plus ``reax-runtime-relay``: a
small proxy that exposes the sidecar's decision contract to the gateway over an authenticated link. Workers never see
the hotkey.

The link between gateway and relay:

* every request carries ``X-Reax-Link: <unix ms>.<nonce>.<hmac>``: HMAC-SHA256 with a per-worker link key over the
  method, path, time, nonce and the SHA-256 of the body. The key itself never travels. The relay refuses requests
  older than ``LINK_MAX_AGE_S`` or replayed nonces;
* the transport is either TLS with the relay's self-signed certificate pinned by its SHA-256 fingerprint on the
  gateway (any network, e.g. two data centres), or plain http inside a private network or VPN such as WireGuard
  (private addresses only; the relay refuses to listen on a public address without TLS).

The gateway reads its workers from ``REAX_WORKERS_FILE`` (JSON) and re-reads it while running, so a machine can be
added, swapped or removed without restarting the gateway (the UID keeps answering on the remaining workers). New
workers go through the health and identity checks before they receive validator traffic; removed workers are drained.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import ssl
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

log = logging.getLogger("reax.multinode")

LINK_HEADER = "X-Reax-Link"
REPLY_HEADER = "X-Reax-Link-Reply"
LINK_MAX_AGE_S = 10.0  # same freshness as the validator <-> miner exchange
LINK_MAX_FUTURE_S = 2.0
NONCE_RETENTION_S = 30.0
MIN_KEY_BYTES = 32
MAX_WORKERS = 64
RELOAD_INTERVAL_S = 10.0
DRAIN_TIMEOUT_S = 60.0
RELAY_MAX_BODY = 256 * 1024
RELAY_MAX_RESPONSE = 1 << 20
RELAY_MAX_PENDING_PER_IP = 32  # request bodies being read before authentication, per remote address
RELAY_MAX_PENDING_TOTAL = 256
DEFAULT_RELAY_PORT = 8190
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FP_RE = re.compile(r"^[0-9a-f]{64}$")


class LinkError(Exception):
    """A request on the gateway <-> worker link failed authentication."""


# --- link authentication --------------------------------------------------------------------------------------------
def read_key(path: str) -> bytes:
    """A link key file: hex text (``openssl rand -hex 32``), at least 32 bytes of key."""
    with open(path, encoding="ascii") as fh:
        raw = fh.read().strip()
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        raise ValueError(f"link key file {path!r} must contain hex text (openssl rand -hex 32)") from None
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"link key file {path!r} holds {len(key)} bytes; at least {MIN_KEY_BYTES} are required")
    return key


def _mac(key: bytes, method: str, path: str, ts_ms: str, nonce: str, body_sha: str) -> str:
    msg = "\n".join(("reax-link/1", method.upper(), path, ts_ms, nonce, body_sha))
    return hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()


def sign_link(key: bytes, method: str, path: str, body: bytes = b"", now: Optional[float] = None) -> str:
    """``<unix ms>.<nonce>.<sha256(body)>.<hmac>``: the body hash travels in the signed header, so the relay can check
    the signature before it reads (or reserves room for) any body."""
    ts_ms = str(int((time.time() if now is None else now) * 1000))
    nonce = secrets.token_hex(12)
    body_sha = hashlib.sha256(body).hexdigest()
    return f"{ts_ms}.{nonce}.{body_sha}.{_mac(key, method, path, ts_ms, nonce, body_sha)}"


class NonceCache:
    def __init__(self, retention_s: float = NONCE_RETENTION_S, max_size: int = 200_000):
        self._seen: dict[str, float] = {}
        self._retention = retention_s
        self._max = max_size

    def seen(self, nonce: str, now: float) -> bool:
        t = self._seen.get(nonce)
        return t is not None and t >= now - self._retention

    def add(self, nonce: str, now: float) -> bool:
        """False if the nonce was seen within the retention."""
        cutoff = now - self._retention
        while self._seen:  # insertion order = time order: drop expired nonces from the front
            oldest = next(iter(self._seen))
            if self._seen[oldest] >= cutoff:
                break
            del self._seen[oldest]
        if len(self._seen) >= self._max:
            return False  # flooded with valid signatures: refuse rather than forget
        if nonce in self._seen:
            return False
        self._seen[nonce] = now
        return True


def verify_link_head(key: bytes, header: Optional[str], method: str, path: str, nonces: NonceCache,
                     now: Optional[float] = None) -> tuple[str, str]:
    """Check a link signature WITHOUT the body (shape, freshness, HMAC, not yet used). Returns (signed body hash,
    nonce); the caller compares the hash with the body it then reads and only then spends the nonce
    (``check_body``), so a copy of a seen header with another body cannot use up the real request's nonce."""
    if not header or header.count(".") != 3:
        raise LinkError("missing or malformed link signature")
    ts_ms, nonce, body_sha, mac = header.split(".")
    if not ts_ms.isdigit() or len(ts_ms) > 16 or not re.fullmatch(r"[0-9a-f]{24}", nonce) \
            or not _FP_RE.fullmatch(body_sha) or not _FP_RE.fullmatch(mac):
        raise LinkError("malformed link signature")
    now = time.time() if now is None else now
    age = now - int(ts_ms) / 1000.0
    if age > LINK_MAX_AGE_S or age < -LINK_MAX_FUTURE_S:
        raise LinkError("stale link signature (check NTP on both machines)")
    if not hmac.compare_digest(mac, _mac(key, method, path, ts_ms, nonce, body_sha)):
        raise LinkError("bad link signature")
    if nonces.seen(nonce, now):
        raise LinkError("replayed link signature")
    return body_sha, nonce


def check_body(head: tuple[str, str], body: bytes, nonces: NonceCache, now: Optional[float] = None) -> None:
    body_sha, nonce = head
    if not hmac.compare_digest(body_sha, hashlib.sha256(body).hexdigest()):
        raise LinkError("bad link signature (body)")
    if not nonces.add(nonce, time.time() if now is None else now):
        raise LinkError("replayed link signature")


def verify_link(key: bytes, header: Optional[str], method: str, path: str, body: bytes, nonces: NonceCache,
                now: Optional[float] = None) -> None:
    check_body(verify_link_head(key, header, method, path, nonces, now), body, nonces, now)


def _reply_mac(key: bytes, request_signature: str, status: int, body: bytes) -> str:
    msg = "\n".join(("reax-link-reply/1", request_signature, str(int(status)), hashlib.sha256(body).hexdigest()))
    return hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()


def sign_reply(key: bytes, request_signature: str, status: int, body: bytes) -> str:
    """The relay's signature over its answer, bound to the request it answers (its link signature)."""
    return _reply_mac(key, request_signature, status, body)


def verify_reply(key: bytes, request_signature: str, status: int, body: bytes, header: Optional[str]) -> None:
    if not header or not _FP_RE.fullmatch(header):
        raise LinkError("the worker's answer is not signed")
    if not hmac.compare_digest(header, _reply_mac(key, request_signature, status, body)):
        raise LinkError("the worker's answer signature does not match (changed on the way?)")


def cert_fingerprint(cert_pem_path: str) -> str:
    """SHA-256 of the certificate's DER bytes, as the gateway pins it (lower-case hex)."""
    with open(cert_pem_path, encoding="ascii") as fh:
        der = ssl.PEM_cert_to_DER_cert(fh.read())
    return hashlib.sha256(der).hexdigest()


def _private_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # RFC1918, ULA, loopback, and 100.64/10 (CGNAT range that Tailscale-style VPNs use).
    return ip.is_loopback or any(ip in ipaddress.ip_network(n) for n in
                                 ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7", "100.64.0.0/10"))


# --- worker file ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkerSpec:
    name: str
    url: str
    key_file: str
    fingerprint: str = ""  # required for https
    max_inflight: int = 16
    evict: bool = False  # {"name": ..., "evict": true}: take this worker out at once, without make-before-break

    def identity(self) -> tuple:
        """What makes a worker the same worker: a change of any of these is a swap (drain the old, admit the new)."""
        try:
            key_id = hashlib.sha256(read_key(self.key_file)).hexdigest()
        except (OSError, ValueError):
            key_id = "unreadable"
        return (self.name, self.url, self.fingerprint, key_id, self.max_inflight)


def parse_workers(doc: Any, model: str, base_dir: str = "") -> list[WorkerSpec]:
    """Validate a workers document: ``{"model": "s1-pro", "workers": [{"name", "url", "key_file", "fingerprint"?,
    "max_inflight"?}, ...]}``. Raises ValueError with a readable reason."""
    if not isinstance(doc, dict) or set(doc) - {"model", "workers"}:
        raise ValueError('the workers file must be a JSON object with "model" and "workers"')
    if doc.get("model") != model:
        raise ValueError(f"the workers file is for model {doc.get('model')!r}, this miner serves {model!r}")
    raw = doc.get("workers")
    if not isinstance(raw, list) or not raw:
        raise ValueError("the workers file lists no workers (at least one is required)")
    if len(raw) > MAX_WORKERS:
        raise ValueError(f"the workers file lists {len(raw)} workers; at most {MAX_WORKERS}")
    out: list[WorkerSpec] = []
    for i, w in enumerate(raw):
        where = f"worker #{i + 1}"
        if not isinstance(w, dict) or set(w) - {"name", "url", "key_file", "fingerprint", "max_inflight", "evict"}:
            raise ValueError(f"{where}: unknown fields (allowed: name, url, key_file, fingerprint, max_inflight, "
                             "evict)")
        name = w.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"{where}: name must be 1-64 letters, digits, '.', '_' or '-'")
        where = f"worker {name!r}"
        if "evict" in w:
            if w["evict"] is not True or set(w) - {"name", "evict"}:
                raise ValueError(f'{where}: an eviction entry is exactly {{"name": ..., "evict": true}}')
            out.append(WorkerSpec(name, "", "", evict=True))
            continue
        url = w.get("url")
        if not isinstance(url, str):
            raise ValueError(f"{where}: url missing")
        url = url.rstrip("/")
        try:
            parts = urlsplit(url)
            host, _port = parts.hostname, parts.port
        except ValueError:
            raise ValueError(f"{where}: url {url!r} is not valid") from None
        if parts.scheme not in ("http", "https") or not host or "@" in parts.netloc or parts.path or parts.query \
                or parts.fragment:
            raise ValueError(f"{where}: url must look like https://203.0.113.5:8190 or http://10.8.0.2:8190")
        fp = str(w.get("fingerprint", "")).lower().replace(":", "")
        if parts.scheme == "https":
            if not _FP_RE.match(fp):
                raise ValueError(f"{where}: an https worker needs the relay certificate's SHA-256 fingerprint "
                                 "(64 hex digits, printed by reax-runtime-relay at start)")
        else:
            if fp:
                raise ValueError(f"{where}: a fingerprint only applies to https URLs")
            if not _private_host(host):
                raise ValueError(f"{where}: plain http is allowed only to private or VPN addresses (10/8, 172.16/12, "
                                 "192.168/16, 100.64/10, fc00::/7); use https with a pinned fingerprint across the "
                                 "internet")
        key_file = w.get("key_file")
        if not isinstance(key_file, str) or not key_file:
            raise ValueError(f"{where}: key_file missing")
        key_file = os.path.expanduser(key_file)
        if base_dir and not os.path.isabs(key_file):
            key_file = os.path.join(base_dir, key_file)
        try:
            read_key(key_file)
        except OSError as exc:
            raise ValueError(f"{where}: key file {key_file!r} not readable ({exc.strerror})") from None
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
        mi = w.get("max_inflight", 16)
        if not isinstance(mi, int) or isinstance(mi, bool) or not 1 <= mi <= 1024:
            raise ValueError(f"{where}: max_inflight must be an integer 1-1024")
        out.append(WorkerSpec(name, url, key_file, fp, mi))
    if len({w.name for w in out}) != len(out):
        raise ValueError("the workers file uses a worker name twice")
    live = [w for w in out if not w.evict]
    if not live:
        raise ValueError("the workers file lists no workers (at least one that is not evicted is required)")
    if len({w.url for w in live}) != len(live):
        raise ValueError("the workers file lists the same URL twice")
    keys = [read_key(w.key_file) for w in live]
    if len(set(keys)) != len(keys):
        raise ValueError("two workers share one link key; give every worker its own key (openssl rand -hex 32)")
    return out


def load_workers_file(path: str, model: str, raw: Optional[bytes] = None) -> list[WorkerSpec]:
    """Parse the workers file (or ``raw``, its bytes as already read)."""
    try:
        if raw is None:
            with open(path, "rb") as fh:
                raw = fh.read()
        doc = json.loads(raw.decode("utf-8"))
    except OSError as exc:
        raise ValueError(f"workers file {path!r} not readable ({exc.strerror})") from None
    except UnicodeDecodeError:
        raise ValueError(f"workers file {path!r} is not UTF-8") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"workers file {path!r} is not valid JSON ({exc.msg} at line {exc.lineno})") from None
    return parse_workers(doc, model, os.path.dirname(os.path.abspath(path)))


class WorkersWatcher:
    """Re-reads the workers file and applies changes to the pool. An invalid file keeps the current workers."""

    def __init__(self, path: str, model: str, pool: Any, make_backend: Any, interval_s: float = RELOAD_INTERVAL_S):
        self.path = path
        self.model = model
        self.pool = pool
        self.make_backend = make_backend
        self.interval = interval_s
        self._last_sig: Optional[tuple] = None
        self._last_error = ""
        self._key_files: tuple[str, ...] = ()
        self._task: Optional[asyncio.Task] = None

    def _read(self) -> tuple[Optional[bytes], Optional[tuple]]:
        """The workers file's bytes and a signature of exactly those bytes plus the content of every key file it
        named last time (a key rewritten in place is noticed like an edit). The bytes read here are the ones that are
        parsed, so a file written while it is being applied is seen again on the next pass."""
        try:
            with open(self.path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return None, None
        parts = [hashlib.sha256(raw).hexdigest()]
        for kf in self._key_files:
            try:
                with open(kf, "rb") as fh:
                    parts.append(hashlib.sha256(fh.read()).hexdigest())
            except OSError:
                parts.append("unreadable")
        return raw, tuple(parts)

    async def check_once(self) -> bool:
        """Apply the file if it changed. Returns True when the pool changed."""
        raw, sig = self._read()
        if sig is not None and sig == self._last_sig:
            return False
        try:
            specs = load_workers_file(self.path, self.model, raw)
        except ValueError as exc:
            if str(exc) != self._last_error:
                log.error("workers file not applied, the current workers stay: %s", exc)
                self._last_error = str(exc)
            return False
        key_files = tuple(sp.key_file for sp in specs if not sp.evict)
        if key_files != self._key_files:
            self._key_files = key_files
            _raw, sig = self._read()  # include the key files of this version; a change after this is seen next pass
            if _raw != raw:
                sig = None  # the file changed meanwhile: apply this version now, the new one on the next pass
        self._last_error = ""
        changed = await self.pool.apply(specs, self.make_backend)
        # A version with an entry the pool refused (e.g. a backend it could not build) is tried again next pass.
        self._last_sig = None if getattr(self.pool, "refused_specs", 0) else sig
        return changed

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.check_once()
            except Exception:  # noqa: BLE001 - the loop must survive
                log.exception("workers file reload failed")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


# --- the relay on each worker machine ------------------------------------------------------------------------------
@dataclass(frozen=True)
class RelayConfig:
    host: str
    port: int
    link_key: bytes
    sidecar_url: str
    sidecar_token: str
    model: str
    tls_cert: str = ""
    tls_key: str = ""
    max_inflight: int = 64


def load_relay_config(env: Mapping[str, str] = os.environ) -> RelayConfig:
    from .config import S1_MODELS, ConfigError, _is_loopback_url

    def get(name: str, default: str = "") -> str:
        return (env.get(name) or default).strip()

    model = get("REAX_MODEL", "s1-fast")
    if model not in ("s1-fast", "s1-pro"):
        raise ConfigError("reax-runtime-relay serves s1-fast or s1-pro (REAX_MODEL); s1-lite and s1-cpu keep one "
                          "machine per hotkey") if model in S1_MODELS else ConfigError(f"unknown REAX_MODEL {model!r}")
    url_var = f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URL"
    sidecar_url = get(url_var).rstrip("/")
    if not sidecar_url or not _is_loopback_url(sidecar_url):
        raise ConfigError(f"{url_var} must be the loopback URL of this machine's runtime sidecar")
    token_file = os.path.expanduser(get("REAX_RUNTIME_TOKEN_FILE"))
    try:
        with open(token_file, encoding="utf-8") as fh:
            token = fh.read().strip()
    except OSError:
        token = ""
    if not token:
        raise ConfigError("REAX_RUNTIME_TOKEN_FILE must name the sidecar's non-empty token file")
    try:
        key = read_key(os.path.expanduser(get("REAX_RELAY_KEY_FILE")))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"REAX_RELAY_KEY_FILE: {exc}") from None
    host = get("REAX_RELAY_HOST", "")
    if not host:
        raise ConfigError("REAX_RELAY_HOST is required: the address the relay listens on (the VPN address, or the "
                          "public address with TLS)")
    try:
        port = int(get("REAX_RELAY_PORT", str(DEFAULT_RELAY_PORT)))
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise ConfigError("REAX_RELAY_PORT must be 1-65535")
    cert, tkey = os.path.expanduser(get("REAX_RELAY_TLS_CERT")), os.path.expanduser(get("REAX_RELAY_TLS_KEY"))
    if bool(cert) != bool(tkey):
        raise ConfigError("set both REAX_RELAY_TLS_CERT and REAX_RELAY_TLS_KEY, or neither")
    if not cert and (host in ("0.0.0.0", "::") or not _private_host(host)):
        raise ConfigError(f"without TLS the relay listens only on a private or VPN address (got {host!r}); set "
                          "REAX_RELAY_TLS_CERT / REAX_RELAY_TLS_KEY to listen on a public address")
    try:
        mi = int(get("REAX_RELAY_MAX_INFLIGHT", "64"))
    except ValueError:
        mi = 0
    if not 1 <= mi <= 1024:
        raise ConfigError("REAX_RELAY_MAX_INFLIGHT must be 1-1024")
    return RelayConfig(host, port, key, sidecar_url, token, model, cert, tkey, mi)


def build_relay_app(cfg: RelayConfig):
    """The relay: authenticated pass-through to the loopback sidecar. Holds no hotkey."""
    import aiohttp
    from aiohttp import web

    from .manifest import load_manifest

    manifest_id = load_manifest(cfg.model).manifest_id
    nonces = NonceCache()
    sem = asyncio.Semaphore(cfg.max_inflight)
    state: dict[str, Any] = {"session": None, "ok": 0, "refused": 0, "sidecar_auth_failed_at": -1e9,
                             "pending": {}, "pending_total": 0}
    last_warn = [0.0]

    async def session() -> aiohttp.ClientSession:
        if state["session"] is None or state["session"].closed:
            state["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), trust_env=False,
                                                     connector=aiohttp.TCPConnector(limit=cfg.max_inflight + 2))
        return state["session"]

    def refuse(reason: str, status: int = 401) -> web.Response:
        state["refused"] += 1
        now = time.monotonic()
        if now - last_warn[0] > 60:
            last_warn[0] = now
            log.warning("relay refused a request: %s (further ones this minute are counted only)", reason)
        return web.json_response({"error": "unauthorized" if status == 401 else reason}, status=status)

    def signed(request: web.Request, status: int, body: bytes) -> web.Response:
        """Every answer to an authenticated request carries the reply signature (gateway checks it)."""
        headers = {REPLY_HEADER: sign_reply(cfg.link_key, request.headers[LINK_HEADER], status, body)}
        return web.Response(body=body, status=status, content_type="application/json", headers=headers)

    def as_json(obj: Any) -> bytes:
        return json.dumps(obj).encode()

    async def health(request: web.Request) -> web.Response:
        try:
            verify_link(cfg.link_key, request.headers.get(LINK_HEADER), "GET", request.path, b"", nonces)
        except LinkError as exc:
            return refuse(str(exc))
        try:
            s = await session()
            async with s.get(cfg.sidecar_url + "/health", headers={"X-S1M-Internal-Token": cfg.sidecar_token},
                             timeout=aiohttp.ClientTimeout(total=5), allow_redirects=False) as r:
                ok = r.status == 200
        except Exception:  # noqa: BLE001 - reported as not ready
            ok = False
        # The sidecar's /health needs no token; a token mismatch shows only on decisions: report it here, so the
        # gateway takes this worker out instead of sending it decisions that fail.
        token_ok = time.monotonic() - state["sidecar_auth_failed_at"] > 30
        ok = ok and token_ok
        info = {"ok": ok, "model": cfg.model, "manifest_id": manifest_id, "served": state["ok"],
                "refused": state["refused"]}
        if not token_ok:
            info["error"] = "the sidecar refused the relay's token (REAX_RUNTIME_TOKEN_FILE)"
        return signed(request, 200 if ok else 503, as_json(info))

    async def decide(request: web.Request) -> web.Response:
        if request.content_length is None or request.content_length > RELAY_MAX_BODY:
            return refuse("body missing or too large", 413)
        try:
            # Signature (with the signed body hash) checked before any body is read: a request without the key
            # never takes one of the slots below, so outsiders cannot fill them.
            head = verify_link_head(cfg.link_key, request.headers.get(LINK_HEADER), "POST", request.path, nonces)
        except LinkError as exc:
            return refuse(str(exc))
        peer = request.remote or "?"
        pending = state["pending"]
        if pending.get(peer, 0) >= RELAY_MAX_PENDING_PER_IP or state["pending_total"] >= RELAY_MAX_PENDING_TOTAL:
            return refuse("too many unfinished requests", 503)
        pending[peer] = pending.get(peer, 0) + 1
        state["pending_total"] += 1
        try:
            body = await asyncio.wait_for(request.read(), timeout=5.0)
        except asyncio.TimeoutError:
            return refuse("request body too slow", 408)
        finally:
            state["pending_total"] -= 1
            pending[peer] -= 1
            if pending[peer] <= 0:
                del pending[peer]
        try:
            check_body(head, body, nonces)
        except LinkError as exc:
            return refuse(str(exc))
        async with sem:
            try:
                s = await session()
                async with s.post(cfg.sidecar_url + "/internal/v1/decide", data=body, allow_redirects=False,
                                  headers={"X-S1M-Internal-Token": cfg.sidecar_token,
                                           "Content-Type": "application/json"}) as r:
                    raw = b""
                    async for chunk in r.content.iter_chunked(64 * 1024):
                        raw += chunk
                        if len(raw) > RELAY_MAX_RESPONSE:
                            return signed(request, 502, as_json({"error": "runtime response too large"}))
                    status = r.status
            except Exception as exc:  # noqa: BLE001
                log.warning("sidecar request failed: %s", type(exc).__name__)
                return signed(request, 502, as_json({"error": "runtime unavailable"}))
        if status == 200:
            state["ok"] += 1
        elif status in (401, 403):
            state["sidecar_auth_failed_at"] = time.monotonic()
            log.error("the sidecar refused the relay's token (HTTP %d): check REAX_RUNTIME_TOKEN_FILE", status)
        return signed(request, status, raw)

    async def on_cleanup(_app):
        if state["session"] is not None:
            await state["session"].close()

    app = web.Application(client_max_size=RELAY_MAX_BODY)
    app.router.add_get("/health", health)
    app.router.add_post("/internal/v1/decide", decide)
    app.on_cleanup.append(on_cleanup)
    return app


def relay_ssl_context(cfg: RelayConfig) -> Optional[ssl.SSLContext]:
    if not cfg.tls_cert:
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cfg.tls_cert, cfg.tls_key)
    return ctx


def relay_main() -> None:
    from aiohttp import web

    from .config import ConfigError

    logging.basicConfig(level=os.environ.get("REAX_LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_relay_config()
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    ctx = relay_ssl_context(cfg)
    if ctx is not None:
        log.info("relay TLS certificate fingerprint (put it into the gateway's workers file): %s",
                 cert_fingerprint(cfg.tls_cert))
    else:
        log.info("relay without TLS on %s: only for a private network or VPN (WireGuard)", cfg.host)
    log.info("reax-runtime-relay for %s on %s:%d -> %s", cfg.model, cfg.host, cfg.port, cfg.sidecar_url)
    web.run_app(build_relay_app(cfg), host=cfg.host, port=cfg.port, ssl_context=ctx, print=None)
