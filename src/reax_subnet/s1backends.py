"""Model backends for ``reax-s1/1``.

Miners and validators do not load the reference models in this process. The
model runs in a *runtime sidecar* container that serves the reference runtime's
decision contract on loopback (``POST /internal/v1/decide`` with the
``X-S1M-Internal-Token`` header):

* s1-fast: ``runtimes/s1-fast`` = the reference fast runtime (pytorch 2.13 image digest,
  jevk5 at the pinned commit) serving the batch-1 CUDA-graph readout of the reference engine;
* s1-pro: ``runtimes/s1-pro`` = the reference Rune image recipe (vLLM 0.30.0 digest,
  deterministic flags) with the reference decision adapter, unchanged.

An optional ``vllm-systemone`` backend instead calls an external vLLM
``POST /v1/systemone`` endpoint and derives the subnet answer fields from its
probabilities. It does not register or approve a new model manifest.

This module sends each decision to the sidecar exactly as the reference engine
does (``Question.author_dict``) and applies the reference ``normalize_answer``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import time
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

from .manifest import Manifest, load_manifest
from .s1contract import Decision, check_answer, normalize_answer, parse_decision


class S1Backend(Protocol):
    model: str
    manifest: Manifest

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]: ...

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]: ...

    async def probe(self) -> bool: ...

    async def close(self) -> None: ...


class SidecarError(RuntimeError):
    """A runtime sidecar failure. ``answer_started``: the runtime had already begun sending its answer (a request that
    failed before that point is safe to try on another backend). ``backend_fault``: the failure says something about
    the backend's health (a 4xx caused by one bad request does not)."""

    def __init__(self, message: str, *, answer_started: bool = True, backend_fault: bool = True):
        super().__init__(message)
        self.answer_started = answer_started
        self.backend_fault = backend_fault


MAX_RUNTIME_RESPONSE_BYTES = 1 << 20
# Engine-competition pools without a release manifest of their own (s1contract.ENGINE_MAX_OPTIONS) borrow the one of the
# reference model they extend; the engine measurement never reads it (it matters for the miner mechanism only).
ENGINE_POOL_MANIFEST = {"s1-vision": "s1-fast", "quyet-1-large": "s1-pro", "jeffan-31b": "s1-pro"}

PROBE_DECISION = parse_decision(
    {"id": "probe", "state": "Probe: the parcel arrived two days late.",
     "questions": {"q": {"type": "noul", "instructions": "Was the delivery late?"}}},
    "s1-fast",
)


async def _read_capped(stream: Any, limit: int) -> bytes:
    """The whole body up to ``limit`` bytes. ``StreamReader.read(n)`` alone returns only what has arrived
    so far, so a body split across TCP segments would be cut short and fail to parse."""
    chunks = []
    size = 0
    async for chunk in stream.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > limit:
            raise SidecarError("runtime response too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _manifest_for(model: str) -> Manifest:
    """A pool model's manifest follows the pool's timed switch; an engine model that only BORROWS a pool's manifest
    (ENGINE_POOL_MANIFEST) keeps the release manifest, so a pool switch never changes what an engine client carries (critic-post81 F1)."""
    if model in ENGINE_POOL_MANIFEST:
        return load_manifest(ENGINE_POOL_MANIFEST[model], switches={})
    return load_manifest(model)


class SidecarBackend:
    """Client for a runtime sidecar on loopback, or (``link_key`` set) for a worker's ``reax-runtime-relay``
    (multinode.py): then every request is HMAC-signed with the worker's link key instead of carrying the sidecar
    token, and an https URL is accepted only with the relay certificate's pinned SHA-256 ``fingerprint``."""

    def __init__(self, model: str, url: str, token: str, max_inflight: int = 16, timeout_s: float = 60.0, *,
                 link_key: Optional[bytes] = None, fingerprint: str = "", name: str = "",
                 manifest: Optional[Manifest] = None):
        self.model = model
        self.manifest = manifest or _manifest_for(model)
        self.url = url.rstrip("/")
        self.name = name or self.url
        self._link_key = link_key
        if link_key is None:
            self._headers = {"X-S1M-Internal-Token": token, "Content-Type": "application/json"}
        else:
            self._headers = {"Content-Type": "application/json"}
        if self.url.startswith("https://") and link_key is not None and not fingerprint:
            raise ValueError(f"worker {self.name}: https needs the relay certificate fingerprint")
        self._fingerprint = fingerprint
        self._sem = asyncio.Semaphore(max_inflight)
        self.max_inflight = max_inflight
        self.inflight = 0  # requests of this process on this backend, running or waiting for a slot
        self._timeout = timeout_s
        self._session = None
        self._health_session = None

    async def _client(self):
        import aiohttp

        if self._session is None or self._session.closed:
            if self._link_key is None:  # flag off: unchanged
                connector = aiohttp.TCPConnector(limit=64)
            else:
                # Enough connections for every slot, so no signed request waits for a connection and goes stale;
                # https only with the pinned self-signed relay certificate.
                connector = aiohttp.TCPConnector(limit=max(64, self.max_inflight), ssl=self._ssl_arg())
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout), trust_env=False, connector=connector,
            )
        return self._session

    def _ssl_arg(self) -> Any:
        import aiohttp

        return aiohttp.Fingerprint(bytes.fromhex(self._fingerprint)) if self._fingerprint else None

    async def _health_client(self):
        """Link mode: health checks get their own connection, so a worker saturated by validator traffic (exactly
        during a capacity probe) is not removed because its health check waited for a pooled connection."""
        import aiohttp

        if self._health_session is None or self._health_session.closed:
            self._health_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0), trust_env=False,
                connector=aiohttp.TCPConnector(limit=2, ssl=self._ssl_arg()))
        return self._health_session

    def _signed(self, method: str, path: str, body: bytes = b"") -> dict[str, str]:
        if self._link_key is None:
            return self._headers
        from .multinode import LINK_HEADER, sign_link

        return {**self._headers, LINK_HEADER: sign_link(self._link_key, method, path, body)}

    def _check_reply(self, request_headers: dict[str, str], response: Any, body: bytes) -> None:
        """Link mode: the relay signs status and body bound to our request signature, so an answer changed on the way
        (e.g. on an unencrypted provider LAN) is refused before the hotkey signs it."""
        from .multinode import LINK_HEADER, REPLY_HEADER, LinkError, verify_reply

        try:
            verify_reply(self._link_key, request_headers[LINK_HEADER], response.status, body,
                         response.headers.get(REPLY_HEADER))
        except LinkError as exc:
            raise SidecarError(f"worker {self.name}: {exc}", answer_started=False, backend_fault=True) from None

    async def ready(self) -> bool:
        if self._link_key is not None:
            return await self._link_ready()
        try:
            s = await self._client()
            async with s.get(self.url + "/health") as r:
                return r.status == 200
        except Exception:
            return False

    async def _link_ready(self) -> bool:
        try:
            s = await self._health_client()
            headers = self._signed("GET", "/health")
            async with s.get(self.url + "/health", headers=headers, allow_redirects=False) as r:
                raw = await _read_capped(r.content, 64 * 1024)
                self._check_reply(headers, r, raw)
                if r.status != 200:
                    return False
                info = json.loads(raw)
            # A relay in front of another model or release must never take this hotkey's traffic.
            return info.get("model") == self.model and info.get("manifest_id") == self.manifest.manifest_id
        except Exception:
            return False

    async def _one(self, d: Decision) -> dict[str, Any]:
        self.inflight += 1
        started = [False]  # set once the runtime answered HTTP 200: from then on answer bytes may exist
        try:
            return await self._send(d, started)
        except Exception as exc:
            # Failover (PooledBackend) needs to know whether the runtime had begun to answer.
            if not hasattr(exc, "answer_started"):
                exc.answer_started = started[0]  # type: ignore[attr-defined]
            raise
        finally:
            self.inflight -= 1

    async def _send(self, d: Decision, started: list) -> dict[str, Any]:
        async with self._sem:
            # The deadline is set once a slot is free, so queued items never reach the runtime already expired.
            body = {
                "request_id": secrets.token_hex(12),
                "deadline_unix_ms": int(time.time() * 1000) + int(self._timeout * 1000),
                "state": d.state,
                "questions": {d.question.ident: d.question.author_dict()},
            }
            if d.images:  # s1-vision: the production field, a list of exactly one data URL
                body["images"] = list(d.images)
            import aiohttp

            s = await self._client()
            data = json.dumps(body, ensure_ascii=False).encode()
            link = self._link_key is not None
            kw = {"allow_redirects": False} if link else {}
            # A pooled keep-alive connection can be closed by the runtime between requests; the first
            # write then fails with ServerDisconnectedError (seen in the 10 Oct GPU rehearsal: 2 of 5
            # sparse requests). Decisions are pure, so one retry on a fresh connection is safe.
            for attempt in (0, 1):
                try:
                    headers = self._signed("POST", "/internal/v1/decide", data)  # fresh nonce per attempt
                    async with s.post(self.url + "/internal/v1/decide", data=data, headers=headers, **kw) as r:
                        if r.status != 200:
                            # Link mode: 401 means the link (key, clock) or the sidecar token behind the relay is
                            # wrong for this worker, not that the request was bad: fail over and count it.
                            fault = r.status >= 500 or r.status in (408, 429) or (link and r.status in (401, 403))
                            raise SidecarError(f"runtime returned HTTP {r.status}", answer_started=False,
                                               backend_fault=fault)
                        if link:
                            raw = await _read_capped(r.content, MAX_RUNTIME_RESPONSE_BYTES)
                            self._check_reply(headers, r, raw)
                            started[0] = True
                        else:
                            started[0] = True
                            raw = await _read_capped(r.content, MAX_RUNTIME_RESPONSE_BYTES)
                    break
                except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
                    if attempt or started[0]:
                        raise
        return normalize_answer(d.question, json.loads(raw)["answers"][d.question.ident])

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(self._one(d) for d in decisions)))

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]:
        """Like ``decide`` but per item: each entry is an answer or the exception of its last attempt."""

        async def attempt(d: Decision) -> Any:
            last: Exception = SidecarError("not attempted")
            for _ in range(retries + 1):
                try:
                    return await self._one(d)
                except Exception as exc:  # noqa: BLE001 - reported per item
                    last = exc
            return last

        return list(await asyncio.gather(*(attempt(d) for d in decisions)))

    async def probe(self) -> bool:
        """Authenticated end-to-end check: one fixed decision through the runtime."""
        try:
            await self._one(PROBE_DECISION)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
        if self._health_session is not None:
            await self._health_session.close()


class VllmSystemOneBackend(SidecarBackend):
    """External vLLM decision source; hotkeys and validator signatures stay in the miner.

    The release manifest is deliberately unchanged. This transport does not approve
    new weights for a production pool.
    """

    def __init__(self, model: str, url: str, served_model: str, max_inflight: int = 16,
                 timeout_s: float = 60.0):
        if not served_model.strip():
            raise ValueError("vllm-systemone needs a served model name")
        if max_inflight < 1 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("vllm-systemone needs positive concurrency and timeout")
        super().__init__(model, url, "", max_inflight, timeout_s)
        self.served_model = served_model

    async def _send(self, d: Decision, started: list) -> dict[str, Any]:
        import aiohttp

        async with self._sem:
            body = {"model": self.served_model, "state": d.state,
                    "questions": {d.question.ident: d.question.author_dict()}}
            try:
                client = await self._client()
                async with client.post(self.url + "/v1/systemone", json=body, allow_redirects=False) as response:
                    if response.status != 200:
                        raise SidecarError(f"vllm-systemone returned HTTP {response.status}",
                                           answer_started=False,
                                           backend_fault=response.status >= 500 or response.status in (408, 429))
                    started[0] = True
                    raw = await _read_capped(response.content, MAX_RUNTIME_RESPONSE_BYTES)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise SidecarError("vllm-systemone transport failed", answer_started=started[0]) from exc
        try:
            result = json.loads(raw)
            if result["model"] != self.served_model or set(result["answers"]) != {d.question.ident}:
                raise ValueError("response model or question ids do not match")
            answer = result["answers"][d.question.ident]
            q = d.question
            if answer["type"] != q.kind:
                raise ValueError("answer type does not match")
            given = answer["probabilities"]
            keys = ("yes", "no") if q.kind == "noul" else q.keys
            if not isinstance(given, dict) or set(given) != set(keys):
                raise ValueError("probabilities must cover exactly the option keys")
            # Validate the raw values before normalization (which coerces numbers).
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                   not math.isfinite(v) or not 0 <= v <= 1 for v in given.values()):
                raise ValueError("invalid probability")
            if abs(math.fsum(given.values()) - 1) > 1e-6:
                raise ValueError("probabilities must sum to 1")
            # Ignore all upstream derived fields, including confidence. The wire
            # contract requires top probability, rather than qev or vLLM confidence.
            source = {"noul": given["yes"]} if q.kind == "noul" else {"probabilities": given}
            normalized = normalize_answer(q, source)
            check_answer(q, normalized)
            return normalized
        except (KeyError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
            raise SidecarError("vllm-systemone returned an invalid answer") from exc


def aiohttp_timeout(total: float):
    import aiohttp

    return aiohttp.ClientTimeout(total=total)


FAILURES_TO_REMOVE = 2  # consecutive backend faults (connection, timeout, HTTP 5xx) before a backend is taken out
SUCCESSES_TO_READMIT = 2  # consecutive good /health answers before a removed backend takes requests again
RETRY_AFTER_S = 10.0  # a removed backend is re-checked (health loop) or tried once (nothing else left) after this long
HEALTH_INTERVAL_S = 5.0

log = logging.getLogger("reax.s1backends")


class _Member:
    def __init__(self, backend: SidecarBackend, healthy: bool = True, spec_id: Optional[tuple] = None):
        self.backend = backend
        self.healthy = healthy
        self.spec_id = spec_id  # multinode: the worker-file entry this member was built from
        self.leaving = False  # multinode: no longer in the file; keeps serving until it can be retired safely
        self.kept_logged = False
        self.failures = 0
        self.good_checks = 0
        self.retry_at = 0.0
        self.trial = False  # a single request is being tried on this removed backend


class PooledBackend:
    """Several runtime sidecars of the SAME model behind one hotkey (docs/MINER.md, "Several GPUs or machines").

    Every decision goes to the healthy backend with the fewest requests in flight (ties go round-robin), never beyond
    what a backend's own ``max_inflight`` allows unless all of them are full. A backend that fails twice in a row is
    removed; the health loop (``GET /health``) re-admits it after it answered twice and, when ``readmit_check`` is set, passed
    the identity check against the release references. A request that failed on one
    backend before any answer byte arrived is tried once on another; it is never tried a third time, and never again
    once the runtime had begun to answer, so a sick fleet cannot be flooded with retries. Signing happens in the miner
    process only: the backends never see the hotkey.
    """

    def __init__(self, backends: Sequence[SidecarBackend], *, health_interval_s: float = HEALTH_INTERVAL_S,
                 retry_after_s: float = RETRY_AFTER_S):
        if not backends:
            raise ValueError("a backend pool needs at least one backend")
        if len({b.model for b in backends}) != 1 or len({b.manifest.manifest_id for b in backends}) != 1:
            raise ValueError("all backends of a pool must serve the same model manifest")
        self.model = backends[0].model
        self.manifest = backends[0].manifest
        self._members = [_Member(b) for b in backends]
        # The number of backends the miner's per-validator limits were configured for (multinode: workers at start).
        self.nominal_size = len(self._members)
        self.min_healthy = 1  # multinode: REAX_WORKERS_MIN_HEALTHY, kept in rotation while workers are swapped
        self._draining: set[asyncio.Task] = set()
        self._drained: list[SidecarBackend] = []
        self._evict: set[str] = set()  # names the workers file evicts at once
        self.refused_specs = 0  # entries of the last applied workers file that could not be built
        self._health_interval = health_interval_s
        self._retry_after = retry_after_s
        self._turn = 0
        self._monitor: Optional[asyncio.Task] = None
        # Set by the miner CLI: ``await readmit_check(backend)`` returns the identity problems of one backend (empty =
        # matches the release references). With it, a removed backend returns only after passing it, and never
        # through a trial request.
        self.readmit_check: Optional[Callable[[SidecarBackend], Awaitable[list[str]]]] = None

    # --- state -----------------------------------------------------------------------------------------------
    @property
    def backends(self) -> list[SidecarBackend]:
        return [m.backend for m in self._members]

    @property
    def size(self) -> int:
        return len(self._members)

    @property
    def healthy_count(self) -> int:
        return sum(1 for m in self._members if m.healthy)

    def describe(self) -> list[dict[str, Any]]:
        return [{"url": m.backend.url, "healthy": m.healthy, "inflight": m.backend.inflight,
                 "max_inflight": m.backend.max_inflight} for m in self._members]

    def _mark_bad(self, m: _Member) -> None:
        m.failures += 1
        m.good_checks = 0
        if m.healthy and m.failures >= FAILURES_TO_REMOVE:
            m.healthy = False
            log.warning("runtime backend %s removed from rotation after %d failures (%d of %d backends healthy)",
                        m.backend.url, m.failures, self.healthy_count, self.size)
        if not m.healthy:
            m.retry_at = time.monotonic() + self._retry_after

    def _mark_good(self, m: _Member) -> None:
        m.failures = 0
        if not m.healthy:
            m.healthy = True
            log.info("runtime backend %s is back in rotation (%d of %d backends healthy)",
                     m.backend.url, self.healthy_count, self.size)

    # --- dispatch --------------------------------------------------------------------------------------------
    def _pick(self, exclude: set[int]) -> Optional[_Member]:
        live = [m for m in self._members if m.healthy and id(m) not in exclude]
        if live:
            free = [m for m in live if m.backend.inflight < m.backend.max_inflight]
            pool = free or live  # everything full: queue at the least loaded one
            low = min(m.backend.inflight for m in pool)
            tied = [m for m in pool if m.backend.inflight == low]
            self._turn += 1
            return tied[self._turn % len(tied)]
        # Nothing healthy: let one request at a time test a removed backend once its wait is over, so the miner
        # recovers even without the health loop, but a dead fleet sees at most one request per backend per wait.
        if self.readmit_check is not None:
            return None  # an unchecked backend must not get validator traffic; the health loop re-admits it
        now = time.monotonic()
        for m in self._members:
            if id(m) not in exclude and not m.trial and now >= m.retry_at:
                m.trial = True
                return m
        return None

    async def _one(self, d: Decision) -> dict[str, Any]:
        tried: set[int] = set()
        for attempt in (0, 1):
            m = self._pick(tried)
            if m is None:
                raise SidecarError("no healthy runtime backend", answer_started=False, backend_fault=False)
            trial = m.trial
            try:
                answer = await m.backend._one(d)
            except Exception as exc:
                fault = getattr(exc, "backend_fault", True)
                if fault:
                    self._mark_bad(m)
                if trial:
                    m.trial = False
                    m.retry_at = time.monotonic() + self._retry_after
                if attempt or not fault or getattr(exc, "answer_started", True):
                    raise
                tried.add(id(m))
                continue
            if trial:
                m.trial = False
            if m.healthy or trial or self.readmit_check is None:
                # A backend taken out while this request was in flight comes back only through the health loop
                # (two good checks and the identity check), never through a late success.
                self._mark_good(m)
            return answer
        raise AssertionError("unreachable")  # pragma: no cover

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(self._one(d) for d in decisions)))

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]:
        async def attempt(d: Decision) -> Any:
            last: Exception = SidecarError("not attempted")
            for _ in range(retries + 1):
                try:
                    return await self._one(d)
                except Exception as exc:  # noqa: BLE001 - reported per item
                    last = exc
            return last

        return list(await asyncio.gather(*(attempt(d) for d in decisions)))

    # --- health ----------------------------------------------------------------------------------------------
    async def check_health_once(self) -> None:
        """One round of ``GET /health`` on every backend: removes failing ones, re-admits recovered ones."""
        members = list(self._members)  # the worker file may change the list while this round awaits
        results = await asyncio.gather(*(m.backend.ready() for m in members))
        for m, ok in zip(members, results):
            if m not in self._members:
                continue
            if ok:
                if m.healthy:
                    m.failures = 0
                else:
                    m.good_checks += 1
                    if m.good_checks >= SUCCESSES_TO_READMIT:
                        m.good_checks = 0
                        if await self._passes_identity(m):
                            self._mark_good(m)
            else:
                self._mark_bad(m)
        self._retire_leaving()

    async def _passes_identity(self, m: _Member) -> bool:
        if self.readmit_check is None:
            return True
        try:
            problems = await self.readmit_check(m.backend)
        except Exception as exc:  # noqa: BLE001 - fail closed
            problems = [f"identity check failed ({type(exc).__name__}: {exc})"]
        if problems:
            log.warning("runtime backend %s is healthy again but stays out of rotation: %s", m.backend.url,
                        "; ".join(problems))
        return not problems

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(self._health_interval)
            try:
                await self.check_health_once()
            except Exception:  # noqa: BLE001 - the loop must survive
                log.exception("backend health check failed")

    async def start(self) -> None:
        if self._monitor is None:
            self._monitor = asyncio.ensure_future(self._monitor_loop())

    async def probe(self) -> bool:
        """Start-up check: every backend answers an authenticated test decision (a misconfigured one is an error)."""
        results = await asyncio.gather(*(m.backend.probe() for m in self._members))
        for m, ok in zip(self._members, results):
            if not ok:
                log.error("runtime backend %s did not answer an authenticated test decision", m.backend.url)
        return all(results)

    async def close(self) -> None:
        if self._monitor is not None:
            self._monitor.cancel()
            self._monitor = None
        for t in list(self._draining):
            t.cancel()
        await asyncio.gather(*(m.backend.close() for m in self._members), *(b.close() for b in self._drained))

    # --- multinode: workers from a file, changed while running ----------------------------------------------------
    @classmethod
    def from_workers(cls, model: str, specs: Sequence[Any], make_backend: Callable[[Any], SidecarBackend],
                     **kw: Any) -> "PooledBackend":
        specs = [sp for sp in specs if not getattr(sp, "evict", False)]
        pool = cls([make_backend(sp) for sp in specs], **kw)
        for m, sp in zip(pool._members, specs):
            m.spec_id = sp.identity()
        if pool.model != model:
            raise ValueError(f"workers serve {pool.model}, the miner serves {model}")
        return pool

    async def startup(self, min_healthy: int) -> int:
        """Multinode start: every worker gets the authenticated test decision and the identity check; the ones that
        fail stay out of rotation (the health loop admits them once they pass) instead of stopping the miner.
        Returns the number of workers in rotation; the caller refuses to start below ``min_healthy``."""
        async def check(m: _Member) -> list[str]:
            if not await m.backend.probe():
                return [f"{m.backend.name}: no answer to an authenticated test decision"]
            if self.readmit_check is None:
                return []
            try:
                return await self.readmit_check(m.backend)
            except Exception as exc:  # noqa: BLE001 - fail closed
                return [f"{m.backend.name}: identity check failed ({type(exc).__name__})"]

        results = await asyncio.gather(*(check(m) for m in self._members))
        for m, problems in zip(self._members, results):
            if problems:
                m.healthy = False
                m.retry_at = time.monotonic() + self._retry_after
                log.warning("worker %s is not in rotation yet: %s", m.backend.name, "; ".join(problems))
            else:
                m.healthy = True
                log.info("worker %s in rotation (%s)", m.backend.name, m.backend.url)
        return self.healthy_count

    async def apply(self, specs: Sequence[Any], make_backend: Callable[[Any], SidecarBackend]) -> bool:
        """Bring the pool in line with the worker file, make before break:

        * new entries join OUT of rotation and get validator traffic only after the health loop saw them healthy
          twice and they passed the identity check;
        * entries that are gone (or whose URL, key, fingerprint or limit changed: remove + add) are marked leaving.
          A leaving worker keeps serving while its same-name replacement is not yet in rotation, or while fewer than
          ``min_healthy`` of the remaining entries are; then it leaves the rotation and is drained. A dead leaving
          worker goes at once unless its same-name replacement is still pending. So a valid but wrong edit (a typo in
          a URL) cannot take the hotkey offline.

        Returns True if anything changed."""
        evict = {sp.name for sp in specs if getattr(sp, "evict", False)}
        wanted = {sp.identity(): sp for sp in specs if not getattr(sp, "evict", False)}
        changed = evict != self._evict
        self._evict = evict
        for m in self._members:
            if m.spec_id in wanted and m.leaving:  # back in the file before it was retired
                m.leaving, m.kept_logged, changed = False, False, True
                log.info("worker %s is in the workers file again", m.backend.name)
            elif m.spec_id not in wanted and not m.leaving:
                m.leaving, changed = True, True
                log.info("worker %s is no longer in the workers file; it leaves once that is safe", m.backend.name)
        known = {m.spec_id for m in self._members}
        self.refused_specs = 0
        for sid, sp in wanted.items():
            if sid in known:
                continue
            try:
                b = make_backend(sp)
            except ValueError as exc:
                self.refused_specs += 1
                log.error("worker %s not added: %s", getattr(sp, "name", "?"), exc)
                continue
            if b.model != self.model or b.manifest.manifest_id != self.manifest.manifest_id:
                self.refused_specs += 1
                log.error("worker %s not added: it is not a %s backend", b.name, self.model)
                continue
            self._members.append(_Member(b, healthy=False, spec_id=sid))
            changed = True
            log.info("worker %s added by the workers file; it takes traffic after the health and identity checks",
                     b.name)
        if changed:
            self._retire_leaving()
            log.info("workers: %d configured, %d in rotation", self.size, self.healthy_count)
        return changed

    def _retire_leaving(self) -> None:
        """Retire leaving members once that is safe. Kept: a member whose same-name replacement is not in rotation yet
        (even while it is itself out after a transient fault: it is still health-checked and may come back), and a
        healthy member while fewer than ``min_healthy`` of the entries still in the file are in rotation (never more
        than there are such entries). Evicted names (``"evict": true``) go at once."""
        staying = [m for m in self._members if not m.leaving]
        staying_healthy = sum(1 for m in staying if m.healthy)
        need = min(self.min_healthy, len(staying))
        for m in [m for m in self._members if m.leaving]:
            replacement_pending = any(not o.healthy and o.backend.name == m.backend.name for o in staying)
            if m.backend.name in self._evict:
                keep = False
            elif replacement_pending:
                keep = True
            else:
                keep = m.healthy and staying_healthy < need
            if keep:
                if not m.kept_logged:
                    m.kept_logged = True
                    log.warning("worker %s stays in rotation although it left the workers file: %s", m.backend.name,
                                "its replacement is not in rotation yet" if replacement_pending else
                                f"only {staying_healthy} other workers are in rotation (REAX_WORKERS_MIN_HEALTHY "
                                f"{self.min_healthy})")
                continue
            self._members.remove(m)
            t = asyncio.ensure_future(self._drain(m))
            self._draining.add(t)
            t.add_done_callback(self._draining.discard)
            log.info("worker %s left the rotation; draining %d requests in flight", m.backend.name, m.backend.inflight)

    async def _drain(self, m: _Member) -> None:
        from .multinode import DRAIN_TIMEOUT_S

        self._drained.append(m.backend)
        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        while m.backend.inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        await m.backend.close()
        self._drained.remove(m.backend)


class UnavailableBackend:
    """Placeholder reference of a pool whose runtime is not configured (a timed manifest switch without its
    ``REAX_<MODEL>_NEXT_RUNTIME_URL``): every decision fails, so the pool is left unscored (no strike, no reset)."""

    def __init__(self, model: str, manifest: Manifest, why: str = "no reference runtime configured"):
        self.model = model
        self.manifest = manifest
        self.why = why

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]:
        raise SidecarError(f"{self.model}: {self.why}", answer_started=False, backend_fault=True)

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]:
        return [SidecarError(f"{self.model}: {self.why}", answer_started=False, backend_fault=True) for _ in decisions]

    async def probe(self) -> bool:
        return False

    async def close(self) -> None:
        pass


class S1HashBackend:
    """Deterministic pseudo-model for tests: ``salt`` plays the role of the weights."""

    def __init__(self, model: str = "s1-fast", salt: str = "reference", logit_noise: float = 0.0, seed: int = 0):
        self.model = model
        self.manifest = _manifest_for(model)
        self.salt = salt
        self.noise = logit_noise
        self._rng_seed = seed
        self.calls = 0

    def logits(self, d: Decision) -> list[float]:
        base = json.dumps([d.state, d.question.author_dict(), *([list(d.images)] if d.images else [])],
                          sort_keys=True, ensure_ascii=False)
        out = []
        for k in d.question.keys:
            h = hashlib.sha256(f"{self.salt}\x00{base}\x00{k}".encode()).digest()
            z = int.from_bytes(h[:4], "big") / 2**32 * 12.0
            if self.noise:
                n = hashlib.sha256(f"noise{self._rng_seed}\x00{base}\x00{k}".encode()).digest()
                z += (int.from_bytes(n[:4], "big") / 2**32 - 0.5) * 2 * self.noise
            out.append(z)
        return out

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]:
        self.calls += len(decisions)
        T = self.manifest.temperature
        answers = []
        for d in decisions:
            z = self.logits(d)
            m = max(z)
            e = [math.exp((x - m) / T) for x in z]
            s = math.fsum(e)
            probs = {k: v / s for k, v in zip(d.question.keys, e)}
            raw: dict[str, Any] = {"type": d.question.kind, "probabilities": probs}
            if d.question.kind == "noul":
                raw = {"type": "noul", "noul": probs["true"]}
            answers.append(normalize_answer(d.question, raw))
        return answers

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]:
        return await self.decide(decisions)

    async def probe(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def load_s1_backend(model: str, url: Optional[str], token: str, allow_dev: bool, max_inflight: int = 16,
                    endpoints: Sequence[tuple[str, str]] = (), manifest: Optional[Manifest] = None, *,
                    kind: str = "sidecar", served_model: str = "", timeout_s: float = 60.0) -> S1Backend:
    """One sidecar from ``url``, or a ``PooledBackend`` over ``endpoints`` ((url, token) pairs; an empty token falls
    back to ``token``). A single endpoint is a plain ``SidecarBackend``.
    ``kind="vllm-systemone"`` selects a single external, token-free endpoint."""
    if kind == "vllm-systemone":
        if not url or endpoints:
            raise ValueError("vllm-systemone needs one runtime URL (runtime URLS pools are unsupported)")
        return VllmSystemOneBackend(model, url, served_model, max_inflight, timeout_s)
    if kind != "sidecar":
        raise ValueError(f"unknown s1 backend kind {kind!r}")
    if endpoints:
        members = []
        for ep_url, ep_token in endpoints:
            if not (ep_token or token):
                raise ValueError("REAX_RUNTIME_TOKEN_FILE must name a non-empty token file shared with the runtime sidecar")
            members.append(SidecarBackend(model, ep_url, ep_token or token, max_inflight=max_inflight, manifest=manifest))
        return members[0] if len(members) == 1 else PooledBackend(members)
    if url == "hash" and allow_dev:
        return S1HashBackend(model)
    if not url:
        var = f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URL"
        raise ValueError(f"{var} is required (the loopback URL of the {model} runtime sidecar)")
    if not token:
        raise ValueError("REAX_RUNTIME_TOKEN_FILE must name a non-empty token file shared with the runtime sidecar")
    return SidecarBackend(model, url, token, max_inflight=max_inflight, manifest=manifest)
