"""Public-repo check behind the Round-2 intake-wait keep-alive (``competition_split.GR2_INTAKE_WAIT_KA``).

One anonymous request per (repo, revision) to the hub's model-info endpoint (no token, no write path): the revision must
exist, be public (not private, not gated), be the committed revision and list at least one ``.safetensors`` file. The
verdict is PASS, FAIL (HTTP 401 / 403 / 404, not public, no safetensors: definite) or UNKNOWN (timeout, network error, hub
5xx / 429: OUR problem, the caller keeps the miner's window). Verdicts are cached per (repo, revision) in this process.

``Round`` is the per weight-set round budget: at most ``GR2_INTAKE_WAIT_MAX_LOOKUPS`` new lookups, at most
``GR2_INTAKE_WAIT_ROUND_BUDGET_S`` seconds waiting in total, each request at most ``GR2_INTAKE_WAIT_LOOKUP_TIMEOUT_S``. Lookups
run on two daemon worker threads (they never delay process exit) so a hung socket or DNS call cannot hold the weight set
longer than the budget; an answer that arrives late is cached and used by a later round. A lookup that is not answered in
time is UNKNOWN for that round (or its last definite verdict). A round waits only for lookups it launched itself: one still
in flight from an earlier round answers at once from the cache (stale verdict or UNKNOWN), so a hung lookup costs one
bounded wait in total, not one per round.

``first_seen`` / ``save_first_seen`` keep the earliest on-chain commitment per repo@revision (critic S-A / S-B, v2.1) in a small
JSON file next to the validator's state (``FIRST_FILE``), so a restart or deploy does not hand "first commitment" to a copy.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import math
import queue
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

from . import competition_split as split

log = logging.getLogger(__name__)

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"
DEFAULT_ENDPOINT = "https://huggingface.co"
MAX_BODY_BYTES = 2 * 1024 * 1024
_urlopen = urllib.request.urlopen          # tests replace this; production never touches the network anywhere else

_LOCK = threading.RLock()
_CACHE: dict[tuple[str, str], tuple[str, float, str]] = {}        # (repo, revision) -> (verdict, expires, detail)
_INFLIGHT: dict[tuple[str, str], concurrent.futures.Future] = {}
_QUEUE: "queue.SimpleQueue[tuple]" = queue.SimpleQueue()
_WORKERS: list[threading.Thread] = []
_N_WORKERS = 2
_CACHE_MAX = 4096
FIRST_FILE = "intake-wait-first.json"
FIRST_KEEP_DAYS = 30.0
_FIRST_MAX_BYTES = 4 * 1024 * 1024
_FIRST_MAX_ENTRIES = 16_000                       # an entry is at most ~230 bytes (128-byte commitment, 48-char hotkey)
_FIRST: dict[str, tuple[float, int, str]] = {}    # "repo.casefold()@revision" -> (earliest on-chain time, uid, hotkey)
_FIRST_LOADED: set[str] = set()


def _endpoint() -> str:
    ep = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    return ep if ep.startswith(("https://", "http://")) else DEFAULT_ENDPOINT


def hub_check(repo: str, revision: str, timeout: float) -> tuple[str, str]:
    """One anonymous model-info request for ``repo@revision`` -> (verdict, short detail). Never raises."""
    timeout = min(float(timeout), 10.0)
    url = f"{_endpoint()}/api/models/{urllib.parse.quote(repo, safe='/')}/revision/{urllib.parse.quote(revision, safe='')}"
    req = urllib.request.Request(url, headers={"User-Agent": "reax-validator", "Accept": "application/json"})
    try:
        with _urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        if status in (401, 403, 404):
            return FAIL, f"hub answered HTTP {status}: repo or revision not public"
        return UNKNOWN, f"hub answered HTTP {status}"
    except Exception as exc:  # noqa: BLE001 - timeouts, DNS, TLS, resets: our side, never the miner's
        return UNKNOWN, f"hub unreachable ({type(exc).__name__})"
    if status != 200:
        return UNKNOWN, f"hub answered HTTP {status}"
    if len(body) > MAX_BODY_BYTES:      # far beyond any accepted repo (spec.rules.MAX_FILES = 256 files): the miner's, not a hub fault
        return FAIL, "file listing far too large for a model repo"
    try:
        info = json.loads(body)
    except ValueError:
        return UNKNOWN, "hub answer is not JSON"
    if not isinstance(info, dict):
        return UNKNOWN, "hub answer is not an object"
    if info.get("private") or info.get("gated"):
        return FAIL, "repo is private or gated"
    if info.get("sha") != revision:
        return FAIL, "hub revision does not match the committed revision"
    siblings = info.get("siblings")
    names = [s.get("rfilename") for s in siblings if isinstance(s, dict)] if isinstance(siblings, list) else []
    if not any(isinstance(x, str) and x.endswith(".safetensors") for x in names):
        return FAIL, "no safetensors weights listed"
    return PASS, "public, safetensors listed"


def reset() -> None:
    """Forget every cached verdict (tests)."""
    with _LOCK:
        _CACHE.clear()


def reset_first_seen() -> None:
    """Forget the first-commitment memory and which files were read (tests: a new process)."""
    _FIRST.clear()
    _FIRST_LOADED.clear()


def first_seen(path: str | None) -> dict[str, tuple[float, int, str]]:
    """The process memory of the earliest commitment per repo@revision; ``path`` (None: memory only) is merged in once per
    process (earlier time wins). A missing file is an empty start; an unreadable or invalid one is ignored with a warning (it is
    replaced at the next save); invalid entries are skipped."""
    if path is None or path in _FIRST_LOADED:
        return _FIRST
    _FIRST_LOADED.add(path)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read(_FIRST_MAX_BYTES + 1)
    except FileNotFoundError:
        log.info("intake-wait: no first-commitment file %s yet, starting empty", path)
        return _FIRST
    except (OSError, ValueError) as exc:
        log.warning("intake-wait: first-commitment file %s unreadable (%r), starting empty", path, exc)
        return _FIRST
    try:
        if len(raw) > _FIRST_MAX_BYTES:
            raise ValueError("too large")
        doc = json.loads(raw)
        if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("entries"), dict):
            raise ValueError("not a version 1 document")
    except Exception as exc:  # noqa: BLE001 - ValueError, RecursionError (deep nesting): a bad file must not stop the weight set
        log.warning("intake-wait: first-commitment file %s invalid (%r), starting empty", path, exc)
        return _FIRST
    bad = 0
    for k, v in doc["entries"].items():
        if (isinstance(k, str) and "@" in k and isinstance(v, list) and len(v) == 3 and isinstance(v[0], (int, float))
                and not isinstance(v[0], bool) and abs(v[0]) < 1e12 and math.isfinite(v[0])     # abs first: no OverflowError
                and type(v[1]) is int and isinstance(v[2], str)):
            rec = (float(v[0]), v[1], v[2])
            _FIRST[k] = min(_FIRST.get(k, rec), rec)
        else:
            bad += 1
    if bad:
        log.warning("intake-wait: skipped %d invalid entries of %s", bad, path)
    return _FIRST


def save_first_seen(path: str | None, now: float, live: set[str], changed: bool) -> None:
    """Forget entries whose time is more than ``FIRST_KEEP_DAYS`` before ``now`` unless a live commitment still names that
    repo@revision; then, if anything changed and ``path`` is set, replace the file atomically (tmp + rename). Never raises."""
    old = now - FIRST_KEEP_DAYS * 86400.0
    for k in [k for k, v in _FIRST.items() if v[0] < old and k not in live]:
        del _FIRST[k]
        changed = True
    if len(_FIRST) > _FIRST_MAX_ENTRIES:     # stay readable (< _FIRST_MAX_BYTES): drop the newest entries no live commitment names
        for k in sorted((k for k in _FIRST if k not in live), key=lambda k: _FIRST[k], reverse=True)[:len(_FIRST) - _FIRST_MAX_ENTRIES]:
            del _FIRST[k]
        changed = True
    if not changed or path is None:
        return
    tmp = None
    try:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".intake-wait-first-", dir=d)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "entries": {k: list(v) for k, v in sorted(_FIRST.items())}}, fh, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
    except Exception as exc:  # noqa: BLE001 - the memory still holds; the next change retries the write
        log.warning("intake-wait: first-commitment file %s not written (%r)", path, exc)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _stale(key: tuple[str, str], why: str) -> tuple[str, str]:
    """The last definite verdict of ``key`` even if expired (the hub failing must not flip a known answer), else UNKNOWN."""
    with _LOCK:
        old = _CACHE.get(key)
    if old is not None and old[0] != UNKNOWN:
        return old[0], f"{old[2]} (stale; {why})"
    return UNKNOWN, why


def _run(key: tuple[str, str], now: float, fetch: Callable[[str, str, float], tuple[str, str]], timeout: float) -> tuple[str, str]:
    try:
        verdict, detail = fetch(key[0], key[1], timeout)
    except Exception as exc:  # noqa: BLE001 - a bug in the check must not stop the weight set
        verdict, detail = UNKNOWN, f"lookup failed ({type(exc).__name__})"
    if verdict not in (PASS, FAIL):
        verdict = UNKNOWN
        log.warning("intake-wait public check of %s@%s: %s; keeping the miner's window, retry in %g min",
                    key[0], key[1][:12], detail, split.GR2_INTAKE_WAIT_RETRY_MINUTES)
    with _LOCK:
        if verdict == UNKNOWN:
            old = _CACHE.get(key)
            if old is not None and old[0] != UNKNOWN:
                verdict, detail = old[0], f"{old[2]} (stale; {detail})"
            ttl = split.GR2_INTAKE_WAIT_RETRY_MINUTES * 60.0
        else:
            ttl = split.GR2_INTAKE_WAIT_CACHE_HOURS * 3600.0
        _CACHE[key] = (verdict, now + ttl, detail)
        _INFLIGHT.pop(key, None)
        if len(_CACHE) > _CACHE_MAX:
            for k in [k for k, v in _CACHE.items() if v[1] < now]:
                del _CACHE[k]
    return verdict, detail


def _worker() -> None:
    while True:
        fut, args = _QUEUE.get()
        if fut.set_running_or_notify_cancel():
            fut.set_result(_run(*args))       # _run never raises


def _submit(*args) -> concurrent.futures.Future:
    """Queue one ``_run`` on the daemon workers (started on first use). Call with ``_LOCK`` held."""
    fut: concurrent.futures.Future = concurrent.futures.Future()
    if len(_WORKERS) < _N_WORKERS:
        t = threading.Thread(target=_worker, name=f"ka-intake-wait-{len(_WORKERS)}", daemon=True)
        t.start()
        _WORKERS.append(t)
    _QUEUE.put((fut, args))
    return fut


class Round:
    """Budget of one weight-set round: ``check`` answers from the cache, else starts one lookup and waits for it only as long
    as the round's budget lasts. A lookup launched by an earlier round is never waited for (critic S2)."""

    def __init__(self, now: float):
        self.now = now
        self.deadline = time.monotonic() + split.GR2_INTAKE_WAIT_ROUND_BUDGET_S
        self.launched = 0
        self._own: set[tuple[str, str]] = set()

    def check(self, repo: str, revision: str) -> tuple[str, str]:
        key = (repo, revision)
        with _LOCK:
            hit = _CACHE.get(key)
            if hit is not None and self.now < hit[1]:
                return hit[0], hit[2]
            fut = _INFLIGHT.get(key)
            if fut is not None and key not in self._own:
                return _stale(key, "lookup from an earlier round still unanswered")
            if fut is None:
                if self.launched >= split.GR2_INTAKE_WAIT_MAX_LOOKUPS:
                    return _stale(key, "lookup limit of this round reached")
                self.launched += 1
                self._own.add(key)
                fut = _INFLIGHT[key] = _submit(key, self.now, hub_check, min(split.GR2_INTAKE_WAIT_LOOKUP_TIMEOUT_S, 10.0))
        try:
            return fut.result(timeout=max(0.0, self.deadline - time.monotonic()))
        except concurrent.futures.TimeoutError:
            return _stale(key, "no answer within the round's lookup budget")
