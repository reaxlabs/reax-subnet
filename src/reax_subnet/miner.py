"""The REAX miner: an HTTP endpoint that answers signed validator requests.

Request handling order (cheapest checks first):
1. size limit, 2. claimed sender must be a current validator (permit and
minimum stake), 3. btauth signature, freshness and replay, 4. per-validator
concurrency, 5. strict payload parsing, 6. inference under a global
concurrency limit, shared out round-robin between validators, 7. signed response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import defaultdict, deque
from typing import Any, Callable, Mapping, Optional, Sequence

from aiohttp import web

from . import auth
from .backends import Backend
from .chain import Chain, Snapshot
from .protocol import MAX_REQUEST_BYTES, DecideResponse, ProtocolError, parse_request
from .s1protocol import MAX_REQUEST_BYTES as S1_MAX_REQUEST_BYTES
from .s1protocol import PROTOCOL_S1, S1_DECIDE_PATH, S1Response, parse_s1_request
from .version import PROTOCOL, __version__

log = logging.getLogger("reax.miner")

DEFAULT_MAX_PENDING_PER_IP = 8
DEFAULT_MAX_PENDING_TOTAL = 4096


class Overloaded(Exception):
    """No slot and no room in the queue for this caller."""


class FairSlots:
    """Runtime slots shared out round-robin between callers (validator hotkeys).

    A plain FIFO semaphore lets one staked caller that bursts many requests fill the queue in front of another
    validator's capacity probe, so the probe measures a fraction of the machine (capacity review M1). Here every caller
    with waiting requests gets the next free slot in turn, and a full queue makes room for a caller below its fair
    share by refusing the newest request of the caller with the most waiting ones.
    """

    def __init__(self, slots: int, max_waiting: int):
        self._free = slots
        self._max_waiting = max_waiting
        self._waiting: dict[str, deque[asyncio.Future]] = {}  # insertion order = round-robin order
        self.waiting = 0

    def _drop(self, caller: str, fut: asyncio.Future) -> None:
        q = self._waiting.get(caller)
        if q is not None and fut in q:
            q.remove(fut)
            self.waiting -= 1
            if not q:
                del self._waiting[caller]

    async def acquire(self, caller: str) -> None:
        if self._free > 0 and not self._waiting:
            self._free -= 1
            return
        if self.waiting >= self._max_waiting:
            if not self._waiting:
                raise Overloaded()
            mine = len(self._waiting.get(caller, ()))
            heaviest = max(self._waiting, key=lambda c: len(self._waiting[c]))
            if len(self._waiting[heaviest]) <= mine + 1:
                raise Overloaded()
            victim = self._waiting[heaviest][-1]
            self._drop(heaviest, victim)
            victim.set_exception(Overloaded())
        fut = asyncio.get_running_loop().create_future()
        self._waiting.setdefault(caller, deque()).append(fut)
        self.waiting += 1
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self.release()  # the slot was handed over just as the caller went away
            else:
                self._drop(caller, fut)
            raise

    def release(self) -> None:
        while self._waiting:
            caller = next(iter(self._waiting))
            q = self._waiting.pop(caller)
            fut = q.popleft()
            self.waiting -= 1
            if q:
                self._waiting[caller] = q  # to the back of the round-robin order
            if not fut.done():
                fut.set_result(None)
                return
        self._free += 1


class _Lane:
    """The per-model state of a hotkey that serves several models: its own runtime slots, per-validator concurrency and
    per-minute window, as if the model had a hotkey of its own. The miner object itself is the lane of its first model
    (the attribute names are the same), so a single-model miner is exactly what it was."""

    def __init__(self, backend: Backend, max_concurrent: int, per_caller_concurrency: int, per_caller_per_minute: int,
                 max_queue: Optional[int] = None):
        self.backend = backend
        self._slots = FairSlots(max_concurrent, max_queue if max_queue is not None else 4 * max_concurrent)
        self._per_caller = per_caller_concurrency
        self._per_minute = per_caller_per_minute
        self._inflight: dict[str, int] = defaultdict(int)
        self._recent: dict[str, deque[float]] = defaultdict(deque)


def allowed_validators(snapshot: Snapshot, min_stake: float) -> frozenset[str]:
    return frozenset(n.hotkey for n in snapshot.neurons if n.validator_permit and n.stake >= min_stake)


class Miner:
    path = auth.DECIDE_PATH
    max_request_bytes = MAX_REQUEST_BYTES

    def __init__(
        self,
        wallet: Any,
        backend: Backend,
        *,
        max_concurrent: int = 4,
        per_caller_concurrency: int = 2,
        per_caller_per_minute: int = 20,
        max_queue: Optional[int] = None,
        body_timeout_s: float = 5.0,
        max_pending_per_ip: int = DEFAULT_MAX_PENDING_PER_IP,
        max_pending_total: int = DEFAULT_MAX_PENDING_TOTAL,
    ):
        self.wallet = wallet
        self.hotkey: str = wallet.hotkey.ss58_address
        self.backend = backend
        self.validators: Optional[frozenset[str]] = None  # None = not loaded yet
        self.nonce_store = auth.NonceStore(retention=60.0)
        self._max_queue = max_queue if max_queue is not None else 4 * max_concurrent
        self._slots = FairSlots(max_concurrent, self._max_queue)
        self._per_caller = per_caller_concurrency
        self._inflight: dict[str, int] = defaultdict(int)
        # Sliding one-minute window of accepted request times per validator.
        self._per_minute = per_caller_per_minute
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        # Bodies are read under a deadline and with a cap on unfinished reads, so
        # slow-drip clients cannot hold connections open.
        self._body_timeout = body_timeout_s
        self._max_pending_ip = max_pending_per_ip
        self._max_pending_total = max_pending_total
        self._pending: dict[str, int] = defaultdict(int)
        self._pending_total = 0
        self._last_warning: dict[int, float] = {}
        self.stats = {"ok": 0, "rejected": 0, "errors": 0}
        self.rejections: dict[int, int] = defaultdict(int)

    def _capacity_scale(self, lane: Any = None) -> float:
        """Share of the configured runtime backends that are healthy (1.0 for a single backend). The per-validator
        limits are configured for all of them; while one is out the limits shrink with it, so a capacity probe
        measures what the machines can really do."""
        pool = (lane or self).backend
        size = getattr(pool, "size", 1)
        healthy = getattr(pool, "healthy_count", size)
        # Multinode: limits were configured for the workers at start; a worker that is joining during a swap does
        # not shrink them, and more workers than at start never raise them above the configured values.
        nominal = getattr(pool, "nominal_size", size)
        return min(1.0, max(healthy, 1) / max(nominal, 1))

    def _caller_limits(self, lane: Any = None) -> tuple[int, int]:
        lane = lane or self
        scale = self._capacity_scale(lane)
        return max(1, math.ceil(lane._per_caller * scale)), max(1, math.ceil(lane._per_minute * scale))

    def _lane_for(self, body: bytes) -> tuple[Any, Optional[web.Response]]:
        """The lane (per-model limits) a verified request counts against, or the rejection to answer instead."""
        return self, None

    def set_validators(self, hotkeys: frozenset[str]) -> None:
        self.validators = hotkeys

    def app(self) -> web.Application:
        app = web.Application(client_max_size=self.max_request_bytes)
        app.router.add_post(self.path, self.handle_decide)
        app.router.add_get("/health", self.handle_health)
        return app

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(self.health_info())

    def health_info(self) -> dict:
        return {"protocol": PROTOCOL, "version": __version__, "model": self.backend.model_alias}

    def _reject(self, status: int, reason: str) -> web.Response:
        self.stats["rejected"] += 1
        self.rejections[status] += 1
        now = time.monotonic()
        if status in (401, 503) and now - self._last_warning.get(status, -1e9) > 60.0:
            # Rate-limited so floods cannot fill the log, but operators see e.g. clock-skew problems.
            self._last_warning[status] = now
            log.warning("rejected a request with %d: %s (further ones this minute are counted only)", status, reason)
        else:
            log.debug("rejected with %d: %s", status, reason)
        return web.json_response({"error": reason}, status=status)

    async def handle_decide(self, request: web.Request) -> web.Response:
        if request.content_length is None or request.content_length > self.max_request_bytes:
            return self._reject(413, "body missing or too large")
        if self.validators is None:
            return self._reject(503, "metagraph not loaded yet")
        claimed = auth.claimed_hotkey(request.headers)
        if not claimed or claimed not in self.validators:
            return self._reject(403, "sender is not a validator on this subnet")
        peer = request.remote or "?"
        if self._pending[peer] >= self._max_pending_ip or self._pending_total >= self._max_pending_total:
            return self._reject(503, "too many unfinished requests")
        self._pending[peer] += 1
        self._pending_total += 1
        try:
            body = await asyncio.wait_for(request.read(), timeout=self._body_timeout)
        except asyncio.TimeoutError:
            return self._reject(408, "request body too slow")
        finally:
            self._pending_total -= 1
            self._pending[peer] -= 1
            if self._pending[peer] <= 0:
                del self._pending[peer]
        try:
            caller = auth.verify_request(request.headers, body, self.hotkey, self.nonce_store, path=self.path)
        except auth.AuthError as exc:
            return self._reject(401, f"authentication failed: {type(exc).__name__}")
        # The verified signer must be the sender that passed the allow-list.
        if caller != claimed or caller not in self.validators:
            return self._reject(403, "sender is not a validator on this subnet")
        lane, refused = self._lane_for(body)
        if refused is not None:
            return refused
        per_caller, per_minute = self._caller_limits(lane)
        if lane._inflight[caller] >= per_caller:
            return self._reject(429, "too many concurrent requests from this validator")
        now = time.monotonic()
        recent = lane._recent[caller]
        while recent and recent[0] <= now - 60.0:
            recent.popleft()
        if len(recent) >= per_minute:
            return self._reject(429, "request rate limit for this validator exceeded")
        recent.append(now)
        lane._inflight[caller] += 1
        try:
            return await self._decide(caller, body)
        finally:
            lane._inflight[caller] -= 1
            if lane._inflight[caller] <= 0:
                del lane._inflight[caller]

    async def _decide(self, caller: str, body: bytes) -> web.Response:
        try:
            req = parse_request(body)
        except ProtocolError as exc:
            return self._reject(400, str(exc))
        if req.model != self.backend.model_alias:
            return self._reject(400, f"this miner serves {self.backend.model_alias}")
        try:
            await self._slots.acquire(caller)
        except Overloaded:
            return self._reject(503, "miner is overloaded")
        started = time.monotonic()
        # The slot is released when inference really finishes, even if the
        # validator disconnects and this handler is cancelled meanwhile.
        task = asyncio.ensure_future(asyncio.to_thread(self.backend.predict, list(req.items)))
        task.add_done_callback(lambda _t: self._slots.release())
        try:
            probs = await asyncio.shield(task)
            if len(probs) != len(req.items):
                raise RuntimeError("backend returned the wrong number of results")
        except Exception:
            self.stats["errors"] += 1
            log.exception("inference failed")
            return web.json_response({"error": "inference failed"}, status=500)
        resp = DecideResponse(
            request_id=req.request_id,
            results=tuple((it.item_id, tuple(p)) for it, p in zip(req.items, probs)),
            miner_version=__version__,
        )
        raw = resp.to_bytes()
        headers = auth.sign_response(self.wallet, raw, req.request_id, caller)
        headers["Content-Type"] = "application/json"
        self.stats["ok"] += 1
        log.debug("answered %d items for %s in %.2fs", len(req.items), caller[:8], time.monotonic() - started)
        return web.Response(body=raw, headers=headers)


class S1Miner(Miner):
    """reax-s1 miner: answers typed decisions with the pinned reference model in its runtime sidecar.

    ``backend`` is one backend, or a ``{model: backend}`` mapping for a hotkey that serves several models
    (docs/DESIGN-MULTIROLE.md section 4): requests are routed by ``model`` and ``manifest_id``; every model has its own
    runtime slots and its own per-validator limits (``model_limits``: model -> (max_concurrent, per-validator
    concurrency, per-validator per minute); the first model uses the keyword arguments), so a probe of one model never
    uses up another model's limits, exactly as with separate hotkeys. The unfinished-body limits scale with the number
    of models (the probe windows of all models run together); the total stays as configured."""

    path = S1_DECIDE_PATH
    max_request_bytes = S1_MAX_REQUEST_BYTES

    def __init__(self, wallet: Any, backend: Any, *, model_limits: Optional[Mapping[str, tuple[int, int, int]]] = None,
                 **kw: Any):
        backends = dict(backend) if isinstance(backend, Mapping) else {backend.model: backend}
        if not backends:
            raise ValueError("a miner needs at least one model")
        n = len(backends)
        # a validator sends the probe lanes of every model from one address; the total of unauthenticated bodies a miner
        # buffers stays as configured (critic round 1 F6)
        kw["max_pending_per_ip"] = kw.get("max_pending_per_ip", DEFAULT_MAX_PENDING_PER_IP) * n
        first, *rest = backends
        super().__init__(wallet, backends[first], **kw)
        self.backends = backends
        self._lanes: dict[str, Any] = {first: self}
        defaults = (kw.get("max_concurrent", 4), kw.get("per_caller_concurrency", 2), kw.get("per_caller_per_minute", 20))
        for model in rest:
            conc, per_caller, per_minute = (model_limits or {}).get(model, defaults)
            self._lanes[model] = _Lane(backends[model], conc, per_caller, per_minute)

    def health_info(self) -> dict:
        models = {m: b.manifest.manifest_id for m, b in self.backends.items()}
        info = {"protocol": PROTOCOL_S1, "version": __version__, "models": models}
        if len(models) == 1:
            info.update(model=self.backend.model, manifest_id=self.backend.manifest.manifest_id)
        return info

    def _serves(self) -> str:
        return ", ".join(f"{b.model} manifest {b.manifest.manifest_id}" for b in self.backends.values())

    def _lane_for(self, body: bytes) -> tuple[Any, Optional[web.Response]]:
        if len(self._lanes) == 1:
            return self, None
        # Several models: the limits depend on the model, which is only known from the body. It is signed and from an
        # allowed validator by now; a body that is not for a served model is refused without using any model's limits.
        try:
            model = json.loads(body).get("model")
        except (ValueError, AttributeError):
            model = None
        lane = self._lanes.get(model) if isinstance(model, str) else None
        if lane is None:
            return None, self._reject(400, f"this miner serves {self._serves()}")
        return lane, None

    async def _decide(self, caller: str, body: bytes) -> web.Response:
        try:
            req = parse_s1_request(body)
        except ProtocolError as exc:
            return self._reject(400, str(exc))
        lane = self._lanes.get(req.model, self)
        backend = lane.backend
        if req.model != backend.model or req.manifest_id != backend.manifest.manifest_id:
            return self._reject(400, f"this miner serves {self._serves()}")
        try:
            await lane._slots.acquire(caller)
        except Overloaded:
            return self._reject(503, "miner is overloaded")
        started = time.monotonic()
        task = asyncio.ensure_future(backend.decide(list(req.decisions)))
        task.add_done_callback(lambda _t: lane._slots.release())
        try:
            answers = await asyncio.shield(task)
            if len(answers) != len(req.decisions):
                raise RuntimeError("backend returned the wrong number of answers")
        except Exception:
            self.stats["errors"] += 1
            log.exception("inference failed")
            return web.json_response({"error": "inference failed"}, status=500)
        resp = S1Response(
            request_id=req.request_id,
            miner_version=__version__,
            results=tuple((d.ident, {d.question.ident: a}) for d, a in zip(req.decisions, answers)),
        )
        raw = resp.to_bytes()
        headers = auth.sign_response(self.wallet, raw, req.request_id, caller, path=self.path)
        headers["Content-Type"] = "application/json"
        self.stats["ok"] += 1
        log.debug("answered %d decisions for %s in %.2fs", len(req.decisions), caller[:8], time.monotonic() - started)
        return web.Response(body=raw, headers=headers)


def multimodel_refusal(n_models: int, block: Optional[int], force: bool) -> str:
    """Why a miner with ``n_models`` models must not start (it would publish a commitment validators do not read), or "".

    ``block`` None = not known yet: only the held constant (MULTIMODEL_START_BLOCK None) refuses."""
    from . import elements as E

    if n_models < 2 or force:
        return ""
    start = E.MULTIMODEL_START_BLOCK
    if start is None:
        return (f"{n_models} models on one hotkey are not accepted by validators in this release yet "
                "(elements.MULTIMODEL_START_BLOCK is held); serve one model, or REAX_FORCE_MULTIROLE=1")
    if block is not None and block < start:
        return (f"{n_models} models on one hotkey count from block {start}, the chain is at block {block}: "
                "start this miner then, or REAX_FORCE_MULTIROLE=1")
    return ""


class CommitmentPublisher:
    """Keeps the inference element of this hotkey's commitment at ``{model: manifest_id}`` (docs/DESIGN-MULTIROLE.md
    section 1). Every write replaces all fields, so the new commitment is the current one with only the inference element
    replaced: a competition element the operator or an entrant tool put there stays. The miner compares only its own
    element with the chain; it never writes two elements before DUALROLE_START_BLOCK (validators read no role from them,
    and the alternative would delete the competition element) unless ``force``."""

    def __init__(self, chain: Any, models: Mapping[str, str], *, hotkey: Optional[str], force: bool = False,
                 retry_s: float = 600.0):
        from . import elements as E

        self.chain = chain
        self.wanted = E.inference_text(models)
        self.force = force
        self.retry_s = retry_s
        # Every write is preceded by a fresh read of this hotkey's commitment that must show the same fields (one bad
        # metagraph read must never make the miner write over an entrant's element; critic round 1 F1). ``hotkey`` is
        # required; None (tests only) skips the second read.
        self.hotkey = hotkey
        self._last_try = -1e9
        self._warned: tuple[str, ...] = ()

    def published(self, fields: Sequence[str]) -> bool:
        """The wanted inference element is on chain and nothing voids the commitment: every field is an element (of this
        or a later release: an unknown ``reax-`` element is not this miner's business) and no kind repeats."""
        from . import elements as E

        stripped = [f.strip() for f in fields if isinstance(f, str)]
        kinds = [E.loose_kind(f) for f in stripped]
        if len(stripped) != len(fields) or None in kinds or len(set(kinds)) != len(kinds):
            return False
        return self.wanted in stripped

    def _own_fields(self) -> Optional[tuple[str, ...]]:
        from . import elements as E

        snap = self.chain.snapshot()
        mine = next((n for n in snap.neurons if n.hotkey == self.hotkey), None)
        return None if mine is None else E.fields_of(mine)

    def plan(self, current_fields: Sequence[str], block: int) -> Optional[list[str]]:
        """The fields to write, or None (already published, or refused: the reason is logged)."""
        from .commitcli import Refused, write_plan

        if self.published(current_fields):
            return None
        try:
            return write_plan(current_fields, self.wanted, block, force=self.force)
        except Refused as exc:
            log.error("NOT publishing commitment %s: %s", self.wanted, exc)
            return None

    async def publish(self, current_fields: Sequence[str], block: int) -> None:
        self._last_try = time.monotonic()
        fields = self.plan(current_fields, block)
        if fields is None:
            return
        if self.hotkey is not None:
            try:
                fresh = await asyncio.to_thread(self._own_fields)
            except Exception as exc:  # noqa: BLE001 - chain unavailable: try again at the next refresh
                log.warning("not publishing commitment %s now: could not re-read it (%s)", self.wanted, exc)
                return
            if fresh is None or [f.strip() for f in fresh] != [f.strip() for f in current_fields]:
                log.warning("not publishing commitment %s now: the commitment read differently on a second read; "
                            "retrying at the next refresh", self.wanted)
                return
        try:
            await asyncio.to_thread(self.chain.set_commitment_fields, fields)
            log.info("published commitment %s", " + ".join(fields))
        except Exception as exc:
            log.error("could not publish commitment %s (validators will not query this miner until it is on chain; "
                      "retrying every %d minutes): %s", self.wanted, self.retry_s // 60, exc)

    def on_snapshot(self, hotkey: str) -> Callable[[Snapshot], None]:
        """Callback for ``refresh_validators_forever``: re-publish (at most every ``retry_s``) when the inference
        element on chain differs from the wanted one; a different competition element is none of its business."""
        from . import elements as E

        def check(snap: Snapshot) -> None:
            mine = next((n for n in snap.neurons if n.hotkey == hotkey), None)
            if mine is None or time.monotonic() - self._last_try <= self.retry_s:
                return
            fields = E.fields_of(mine)
            if not self.published(fields):
                asyncio.get_running_loop().create_task(self.publish(fields, snap.block))
            elif E.split_elements(fields, snap.block) is None and fields != self._warned:
                # e.g. next to an element of a later release, or before a start block: not this miner's to change
                self._warned = tuple(fields)
                log.warning("validators of this release read no role from this hotkey's commitment %s at block %d; "
                            "check it with reax-commit show", " + ".join(f[:40] for f in fields), snap.block)

        return check


async def refresh_validators_forever(
    miner: Miner, chain: Chain, min_stake: float, interval_s: int, on_snapshot: Optional[Callable[[Snapshot], None]] = None
) -> None:
    while True:
        try:
            snap = await asyncio.to_thread(chain.snapshot)
            miner.set_validators(allowed_validators(snap, min_stake))
            if snap.uid_of(miner.hotkey) is None:
                log.warning("hotkey %s is not registered on this subnet", miner.hotkey)
            if on_snapshot is not None:
                on_snapshot(snap)
            log.info(
                "metagraph block %d: %d neurons, %d accepted validators; served ok=%d rejected=%d %s errors=%d",
                snap.block, len(snap.neurons), len(miner.validators or ()), miner.stats["ok"],
                miner.stats["rejected"], dict(sorted(miner.rejections.items())), miner.stats["errors"],
            )
        except Exception as exc:  # keep serving with the last known validator set
            log.warning("metagraph refresh failed: %s", exc)
        await asyncio.sleep(interval_s)
