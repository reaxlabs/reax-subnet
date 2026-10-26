"""Start competition engines and measure them against the reference engine (docs/ENGINE-COMPETITION.md).

Engine contract: the image serves the runtime sidecar contract (``POST /internal/v1/decide`` with the
``X-S1M-Internal-Token`` header, ``GET /health``) on the unix socket named by ``RUNTIME_UDS`` (mode 0666), reads the
token from ``S1M_INTERNAL_TOKEN_FILE`` and the pinned weights from ``PLUMB_MODEL_DIR`` (read-only). It runs with no
network, a read-only root filesystem and no capabilities; ``/tmp`` is the only writable path. A submitted image
carries the label ``ai.reax.engine.hotkey=<ss58 of the committing hotkey>``, so a digest cannot be claimed by
anybody else.

Two runners:

* ``DockerRunner`` (real validators): ``docker pull <image>@sha256:<digest>`` from an allowed registry, then
  ``docker run --network none`` with one GPU, CPU/memory/pids/log caps, the weights mounted read-only and a fresh
  socket directory per engine. The idle engine is paused (``docker pause``) while the other one is measured;
* ``DevRunner`` (local test chains and unit tests only): a JSON map from image reference to a local command.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

from .engconfigs import (FLA_CONFIG_ROOT, GPU_CONFIG_DIRS, PUBLISHED_GPU_CLASSES, ConfigVerificationError,
                         host_config_root, verify_published_configs)
from . import engreview
from .engcomp import EvalResult, TrackParams, check_correctness, copyline_active, max_distance, median, speed_score
from .s1backends import SidecarBackend
from .s1contract import Decision, parse_decision
from .s1items import S1Item, generate_s1_items

log = logging.getLogger("reax.engines")

SOCKET_NAME = "engine.sock"
HOTKEY_LABEL = "ai.reax.engine.hotkey"
DEFAULT_REGISTRIES = ("ghcr.io", "docker.io", "quay.io")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean(text: str, limit: int = 200) -> str:
    """Engine-influenced text for logs and state: no control characters, bounded length."""
    return _CONTROL.sub(" ", text)[-limit:]


IDENTITY_JSON_MAX_BYTES = 32 * 2 ** 20  # config/tokenizer/index files whose content is part of the weights identity
IDENTITY_HEADER_MAX_BYTES = 128 * 2 ** 20
IDENTITY_SAMPLE_BYTES = 64 * 2 ** 10
CP_TOO_LARGE = -2  # _run: the watched output path grew past its cap and the process was killed


class EngineError(RuntimeError):
    """The engine under test failed (it loses this evaluation)."""


class InfraError(RuntimeError):
    """Pull/start failed for a reason that may be ours (registry rate limit, docker hiccup): inconclusive."""


class AwaitingReview(RuntimeError):
    """Release mode: the digest has no REVIEWED/<digest> marker yet. Not the miner's fault: nothing is recorded or counted."""


RELEASE_RUNTIME = "runsc"
# What the daemon's ``runsc`` entry must carry (daemon.json "runtimes"): the binary is gVisor's runsc, its sandbox has no
# network stack of its own, it has the NVIDIA proxy for the GPU and it can create the engine socket in its host mount.
RELEASE_RUNTIME_ARGS = ("--network=none", "--nvproxy=true", "--nvproxy-allowed-driver-capabilities=compute,utility",
                        "--host-uds=create")
ENGINE_GPU_MIB = 12_500  # one s1-fast engine (Plumb-4B, 12.3 GB measured on the validator card)
MAX_IMAGE_COMPRESSED_BYTES = 15 * 10 ** 9  # refused before the pull (a decompression bomb is small compressed, but this caps the download)


class ReferenceError_(RuntimeError):
    """The reference engine failed before the candidate ran: the evaluation is inconclusive and is not recorded."""


class ReferenceSelfCheckFailed(ReferenceError_):
    """VOID session: reference repeatability failed; never charged to a miner."""


class ReferenceFailedWithCandidate(ReferenceError_):
    """The reference failed while the candidate was loaded (the candidate may have starved it)."""


class UdsSidecarBackend(SidecarBackend):
    """The sidecar client over a unix socket."""

    def __init__(self, model: str, socket_path: str, token: str, max_inflight: int, timeout_s: float):
        super().__init__(model, "http://engine", token, max_inflight=max_inflight, timeout_s=timeout_s)
        self.socket_path = socket_path

    def socket_ok(self) -> bool:
        """The path is a real socket, not a symlink to some other socket on the host."""
        try:
            return stat.S_ISSOCK(os.lstat(self.socket_path).st_mode)
        except OSError:
            return False

    async def _client(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout), trust_env=False,
                connector=aiohttp.UnixConnector(path=self.socket_path, limit=256),
            )
        return self._session


@dataclass
class Engine:
    ref: str
    client: UdsSidecarBackend
    stop: Callable[[], Awaitable[None]]
    pause: Callable[[], Awaitable[None]]
    resume: Callable[[], Awaitable[None]]
    gpu_busy: Optional[Callable[[], Awaitable[bool]]] = None  # per-PID GPU activity check (REAX_ENGINE_NVML_CHECK=1)


class Runner(Protocol):
    binds_hotkey: bool

    async def preflight(self) -> None: ...

    async def owner_of(self, ref: str) -> Optional[str]: ...

    async def start(self, ref: str, model: str, params: TrackParams, keep_image: bool = False) -> Engine: ...

    async def discard(self, ref: str) -> None: ...


def _tree_bytes(path: str) -> int:
    """Bytes below ``path`` (a file or a directory; links are not followed). Best effort: 0 when it does not exist yet."""
    total = 0
    try:
        st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode):
            return st.st_size
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.lstat(os.path.join(root, f)).st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


# Docker messages that mean "this image does not exist" (locally or in the registry), as opposed to a daemon error or a
# timeout. Only these let the copy-line re-check count an image as "no parents declared" (critic H1, fail closed).
_GONE_MARKERS = ("no such image", "no such object", "no such manifest", "manifest unknown")


def _image_gone(message: str) -> bool:
    m = message.lower()
    return any(k in m for k in _GONE_MARKERS)


async def _run(argv: Sequence[str], timeout_s: float, watch: Optional[tuple[str, int]] = None) -> tuple[int, str, str]:
    """A subprocess without blocking the event loop; output capped at 64 KiB per stream. ``watch=(path, max_bytes)``: the
    process is killed once ``path`` (what it writes, e.g. a ``docker cp`` destination) grows past ``max_bytes``; the
    result is then ``(CP_TOO_LARGE, "", ...)``."""
    proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    exceeded = False

    async def guard() -> None:
        nonlocal exceeded
        while True:
            await asyncio.sleep(0.1)
            if _tree_bytes(watch[0]) > watch[1]:
                exceeded = True
                proc.kill()
                return

    watcher = asyncio.ensure_future(guard()) if watch else None
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"timeout after {timeout_s:.0f} s"
    except BaseException:
        # Cancelled by an outer budget (or interrupted): the child must not outlive the call that started it.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):  # noqa: BLE001 - reaping is best effort
            pass
        raise
    finally:
        if watcher:
            watcher.cancel()
    if exceeded:
        return CP_TOO_LARGE, "", f"output larger than {watch[1]} bytes"
    return proc.returncode or 0, out[-65536:].decode(errors="replace"), err[-65536:].decode(errors="replace")


async def _abort_start(name: str, d: str) -> None:
    """Remove the container of a start that was interrupted and its session directory. The removal is shielded, so a
    second cancellation cannot abandon it half way; the directory goes in every case."""
    try:
        await asyncio.shield(_run(["docker", "rm", "-fv", name], 120))
    except Exception as exc:  # noqa: BLE001 - best effort; a cancellation still propagates from the caller's raise
        log.warning("could not remove %s after an interrupted start: %r", name, exc)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _session_dir(root: str) -> tuple[str, str]:
    os.makedirs(root, mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    d = tempfile.mkdtemp(prefix="eng-", dir=root)
    io = os.path.join(d, "io")
    os.mkdir(io)
    os.chmod(io, 0o777)  # the engine runs as its own uid and must create its socket here
    token = secrets.token_hex(24)
    with open(os.path.join(io, "token"), "w", encoding="ascii") as fh:
        fh.write(token)
    os.chmod(os.path.join(io, "token"), 0o644)
    return d, token


def _keep_log(root: str, d: str, name: str, text: str) -> None:
    logs = os.path.join(root, "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, f"{name}.log"), "w") as fh:
        fh.write(text[-100_000:])
    shutil.rmtree(d, ignore_errors=True)


def manifest_compressed_size(doc: Any) -> int:
    """Compressed bytes of the image to be pulled, from ``docker manifest inspect -v`` output (one object, or a list with
    one object per platform). Docker-format (``SchemaV2Manifest``) and OCI (``OCIManifest``, the buildx default)
    manifests are both read; attestation entries of an image index (platform ``unknown/unknown``) are skipped. With
    several platforms the one this host would pull counts (linux/amd64 unless the host is arm64); if none matches
    the largest counts. A manifest without sizes raises ValueError."""
    entries = doc if isinstance(doc, list) else [doc]
    sizes: dict[str, int] = {}
    for e in entries:
        plat = (e.get("Descriptor") or {}).get("platform") or {}
        if plat.get("os") == "unknown" and plat.get("architecture") == "unknown":
            continue  # buildx provenance/SBOM attestation, not an image
        m = e.get("SchemaV2Manifest") or e.get("OCIManifest") or e
        if "manifests" in m and "layers" not in m:
            raise ValueError("image index without per-platform manifests")
        layers = m["layers"]
        total = int(m["config"]["size"]) + sum(int(layer["size"]) for layer in layers)
        key = f"{plat.get('os', '')}/{plat.get('architecture', '')}"
        sizes[key] = max(total, sizes.get(key, 0))  # a variant list (arm v6/v7) collapses to its largest
    if not sizes:
        raise ValueError("no manifest")
    import platform

    want = "linux/arm64" if platform.machine().lower() in ("aarch64", "arm64") else "linux/amd64"
    return sizes[want] if want in sizes else max(sizes.values())


async def candidate_gpu_busy(container: str, gpu: str) -> bool:
    """True when a process of the (paused) candidate container shows SM activity on the GPU: queued kernels that keep
    running after ``docker pause`` would slow the reference run that is being timed (docs/ENGINE-COMPETITION.md).
    Without ``nvidia-smi`` in this container the check is skipped (False, with a warning)."""
    code, out, _ = await _run(["docker", "top", container, "-eo", "pid"], 30)
    pids = {ln.strip() for ln in out.splitlines()[1:] if ln.strip().isdigit()} if code == 0 else set()
    if not pids:
        return False
    try:
        code, out, _ = await _run(["nvidia-smi", "pmon", "-i", gpu, "-c", "2", "-s", "u"], 30)
    except OSError as exc:
        log.warning("REAX_ENGINE_NVML_CHECK is on but nvidia-smi cannot be run (%r): the per-PID GPU check is skipped", exc)
        return False
    if code != 0:
        return False
    for ln in out.splitlines():
        cols = ln.split()
        if len(cols) >= 4 and not ln.lstrip().startswith("#") and cols[1] in pids and cols[3].isdigit() and int(cols[3]) > 0:
            return True
    return False


def registry_of(image: str) -> str:
    first, _, rest = image.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        return first
    return "docker.io"


class DockerRunner:
    binds_hotkey = True

    def __init__(self, workdir: str, weights_dir: str, gpu: str = "0", memory: str = "64g", cpus: str = "8",
                 user: str = "10002:10002", docker_runtime: str = "", pull: bool = True,
                 registries: Sequence[str] = DEFAULT_REGISTRIES, pull_timeout_s: float = 1800.0,
                 nvml_check: bool = False, max_compressed_bytes: int = MAX_IMAGE_COMPRESSED_BYTES,
                 protected_images: Sequence[str] = (), release_mode: bool = False, review_dir: str = "",
                 reference_ref: str = "", fla_config_root: str = ""):
        """``release_mode`` (finney): no engine starts unless ``docker_runtime`` is ``runsc`` and ``docker info`` lists it, and
        a digest only runs once ``<review_dir>/REVIEWED/<digest>`` exists. ``review_dir``: where the pre-run submission
        check keeps its records (empty = no check). ``reference_ref``: the reference image, the source of the expected
        tokenizer/config files when the weights directory has none. ``fla_config_root``: host directory with the published
        FLA configs (default engconfigs.host_config_root()); mounted read-only into every engine."""
        self.workdir = workdir
        self.fla_config_root = os.path.abspath(fla_config_root or host_config_root())
        self.weights_dir = os.path.abspath(weights_dir)
        self.gpu = gpu
        self.memory = memory
        self.cpus = cpus
        self.user = user
        self.docker_runtime = docker_runtime
        self.pull = pull
        self.registries = tuple(registries)
        self.pull_timeout_s = pull_timeout_s
        self.nvml_check = nvml_check
        self.max_compressed_bytes = max_compressed_bytes
        self.protected_images = frozenset(protected_images)  # the release reference, the vLLM image: never removed
        self._pulled: set[str] = set()  # images THIS runner pulled (only these are ever removed)
        self.release_mode = release_mode
        self.review_dir = review_dir
        self.reference_ref = reference_ref
        self._reference_files: Optional[dict[str, str]] = None
        if nvml_check and docker_runtime == RELEASE_RUNTIME:
            log.warning("REAX_ENGINE_NVML_CHECK is ignored under %s: docker top lists the sandbox's internal PIDs, which "
                        "never match the host PIDs nvidia-smi reports, so the check would be blind. The reference "
                        "baseline drift check and the runtime's pause semantics cover this case "
                        "(docs/ENGINE-COMPETITION.md, security)", RELEASE_RUNTIME)

    async def config_dir(self) -> str:
        try:
            code, out, _ = await _run(["nvidia-smi", f"--id={self.gpu}", "--query-gpu=name",
                                       "--format=csv,noheader"], 30)
        except OSError:
            raise ReferenceSelfCheckFailed("GPU class unavailable; no published FLA config") from None
        gpu_class = out.strip()
        if code != 0 or gpu_class not in PUBLISHED_GPU_CLASSES or gpu_class not in GPU_CONFIG_DIRS:
            raise ReferenceSelfCheckFailed(f"no published FLA config for GPU class {clean(gpu_class)!r}")
        try:  # checked again before every session: an altered or missing config set voids it, never a miner strike
            verify_published_configs(gpu_class, os.path.join(self.fla_config_root, GPU_CONFIG_DIRS[gpu_class]))
        except ConfigVerificationError as exc:
            raise ReferenceSelfCheckFailed(clean(str(exc))) from None
        return FLA_CONFIG_ROOT + "/" + GPU_CONFIG_DIRS[gpu_class]

    async def require_runtime(self) -> None:
        """C1: in release mode every engine runs under gVisor. Anything else voids the session (never a miner strike):
        the runner is configured without it, or the daemon does not list it, or that cannot be read (fail closed)."""
        if not self.release_mode:
            return
        if self.docker_runtime != RELEASE_RUNTIME:
            raise ReferenceSelfCheckFailed(f"release mode needs REAX_ENGINE_DOCKER_RUNTIME={RELEASE_RUNTIME} "
                                           f"(configured: {clean(self.docker_runtime) or 'none'})")
        try:
            code, out, err = await _run(["docker", "info", "--format", "{{json .Runtimes}}"], 60)
        except OSError as exc:
            raise ReferenceSelfCheckFailed(f"docker info cannot be run ({type(exc).__name__}): runtime unverified") from None
        try:
            runtimes = json.loads(out) if code == 0 else None
        except ValueError:
            runtimes = None
        if not isinstance(runtimes, dict):
            raise ReferenceSelfCheckFailed(f"docker info did not list the runtimes ({clean(err) or 'unreadable'}): "
                                           "runtime unverified")
        if RELEASE_RUNTIME not in runtimes:
            raise ReferenceSelfCheckFailed(f"the docker daemon has no {RELEASE_RUNTIME} runtime")
        entry = runtimes[RELEASE_RUNTIME]
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or os.path.basename(path) != RELEASE_RUNTIME:
            raise ReferenceSelfCheckFailed(f"the docker daemon's {RELEASE_RUNTIME} runtime does not point at a "
                                           f"{RELEASE_RUNTIME} binary (path {clean(str(path))!r})")
        args = entry.get("runtimeArgs")
        # runsc boolean flags may be written "--nvproxy" or "--nvproxy=true" (the runbook's daemon.json uses the bare form)
        have = {a[:-5] if a.endswith("=true") else a for a in args if isinstance(a, str)} if isinstance(args, list) else set()
        missing = [a for a in RELEASE_RUNTIME_ARGS if (a[:-5] if a.endswith("=true") else a) not in have]
        if missing:
            raise ReferenceSelfCheckFailed(f"the docker daemon's {RELEASE_RUNTIME} runtime lacks the required "
                                           f"runtimeArgs {' '.join(missing)}")
        # for runsc the last occurrence of a flag wins: a later --network=host would undo --network=none
        nets = [a for a in args if isinstance(a, str) and a.startswith("--network=")]
        if nets and nets[-1] != "--network=none":
            raise ReferenceSelfCheckFailed(f"the docker daemon's {RELEASE_RUNTIME} runtime ends with {clean(nets[-1])!r}, "
                                           "not --network=none")

        # Reject every wider UDS mode, even if a later flag would override it, and the deprecated alias. runsc parses Go
        # flags: one or two leading dashes, and "--host-uds all" as two arguments, all mean the same flag.
        for arg in args:
            if not isinstance(arg, str):
                continue
            name = "--" + arg.lstrip("-")
            if (name == "--host-uds" or name.startswith("--host-uds=")) and name != "--host-uds=create":
                raise ReferenceSelfCheckFailed(f"the docker daemon's {RELEASE_RUNTIME} runtime has {clean(arg)!r}, "
                                               "not --host-uds=create")
            if name == "--fsgofer-host-uds" or name.startswith("--fsgofer-host-uds="):
                raise ReferenceSelfCheckFailed(f"the docker daemon's {RELEASE_RUNTIME} runtime has deprecated "
                                               f"{clean(arg)!r}; use --host-uds=create")

    async def preflight(self) -> None:
        """Both engines (reference + candidate) are loaded at once. On a shared 96 GB GPU the serving runtimes stay
        resident, so check the free memory first: too little = inconclusive (cycle skipped), never a failed candidate.
        If nvidia-smi cannot be read the memory check is skipped (config admission still fails closed)."""
        await self.require_runtime()
        await self.config_dir()
        try:
            code, out, _ = await _run(["nvidia-smi", f"--id={self.gpu}", "--query-gpu=memory.free",
                                       "--format=csv,noheader,nounits"], 30)
        except OSError as exc:  # no nvidia-smi in this container: the free-memory guard cannot run
            log.warning("GPU free-memory check skipped, nvidia-smi cannot be run (%r)", exc)
            return
        try:
            free_mib = int(out.strip().splitlines()[0])
        except (ValueError, IndexError):
            return
        need = 2 * ENGINE_GPU_MIB
        if code == 0 and free_mib < need:
            raise InfraError(f"GPU {self.gpu} has {free_mib} MiB free, an evaluation needs about {need} MiB "
                             "(two s1-fast engines next to the serving runtimes)")

    async def discard(self, ref: str) -> None:
        """Remove an image this runner pulled (never raises): not-owner images, failed starts and aborted pulls must not
        fill the disk. An image that was already on the host, the configured reference and the spec image stay."""
        if ref in self.protected_images or ref not in self._pulled:
            return
        self._pulled.discard(ref)
        try:
            await _run(["docker", "rmi", ref], 120)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not remove image %s: %r", ref[:80], exc)

    async def compressed_size(self, ref: str) -> int:
        """Total compressed size (config + layers) of the image for this host's platform, read from the registry
        manifest before anything is pulled. Raises InfraError when the manifest cannot be read."""
        code, out, err = await _run(["docker", "manifest", "inspect", "-v", ref], 120)
        if code != 0:
            raise InfraError(f"could not read the image manifest: {clean(err)}")
        try:
            return manifest_compressed_size(json.loads(out))
        except (ValueError, TypeError, KeyError) as exc:
            raise InfraError(f"unreadable image manifest ({type(exc).__name__})") from None

    async def _ensure_image(self, ref: str) -> None:
        if "@sha256:" not in ref:
            raise EngineError("engine images are run by digest only")
        if registry_of(ref.split("@", 1)[0]) not in self.registries:
            raise EngineError(f"registry {registry_of(ref.split('@', 1)[0])!r} is not allowed")
        code, _, _ = await _run(["docker", "image", "inspect", "--format", "{{.Id}}", ref], 60)
        if code == 0 or not self.pull:
            return
        size = await self.compressed_size(ref)
        if size > self.max_compressed_bytes:
            raise EngineError(f"image is {size / 1e9:.1f} GB compressed, the limit is {self.max_compressed_bytes / 1e9:.0f} GB")
        self._pulled.add(ref)  # not on the host before this call, so a half-finished pull is ours to clean up
        code, _, err = await _run(["docker", "pull", "-q", ref], self.pull_timeout_s)
        if code != 0:
            await self.discard(ref)  # a pull that timed out half-way may leave layers behind
            raise InfraError(f"docker pull failed: {clean(err)}")

    async def review(self, ref: str) -> None:
        """Pre-run submission check (engreview.py): once per digest, before its first run. Raises EngineError with the
        reason for an obvious violation and AwaitingReview (release mode) until ``REVIEWED/<digest>`` exists. No-op
        without a review directory."""
        if not self.review_dir:
            if self.release_mode:  # the REVIEWED gate must never be skipped on finney
                raise ReferenceSelfCheckFailed("release mode needs a review directory")
            return
        digest = engreview.digest_of(ref)
        if not digest:
            raise EngineError("engine images are run by digest only")
        rec = engreview.load_record(self.review_dir, digest)
        if rec is None or (copyline_active(datetime.now(timezone.utc)) and not rec.get("copyline")):
            # (a record written before the copy-line start has no label fields: checked again once)
            rec = await self._static_check(ref, digest)
            engreview.write_record(self.review_dir, digest, rec)
            log.info("engine submission check %s: %s%s", digest[:12], rec["status"],
                     (" - " + "; ".join(rec["violations"])) if rec["violations"] else "")
        if rec.get("violations"):
            raise EngineError(clean("submission check: " + "; ".join(str(v) for v in rec["violations"]), 400))
        if self.release_mode and not engreview.is_reviewed(self.review_dir, digest):
            raise AwaitingReview(f"{digest} has a static check record but no REVIEWED/{digest} marker yet")

    def awaiting_review(self, ref: str) -> bool:
        """Release mode: the static check of this digest passed (its record exists) and the operator's marker is still
        missing. The cycle leaves such a digest out, so unreviewed submissions cannot use up the evaluation slots."""
        digest = engreview.digest_of(ref)
        if not (self.release_mode and self.review_dir and digest):
            return False
        rec = engreview.load_record(self.review_dir, digest)
        return bool(rec) and not rec.get("violations") and not engreview.is_reviewed(self.review_dir, digest)

    async def _inspect(self, ref: str) -> dict[str, Any]:
        code, out, err = await _run(["docker", "image", "inspect", "--format", "{{json .}}", ref], 60)
        if code != 0:
            raise InfraError(f"docker inspect failed: {clean(err)}")
        try:
            info = json.loads(out)
        except ValueError:
            raise InfraError("unreadable docker inspect output") from None
        if isinstance(info, list):
            info = info[0] if info else {}
        if not isinstance(info, dict):
            raise InfraError("unreadable docker inspect output")
        return info

    async def recheck_copyline(self, ref: str) -> None:
        """Copy line, once per already-measured digest: a review record written before the start has no ``copyline`` mark.
        Read the labels from ``docker image inspect`` as ``review`` does and add them to the record (the files list and the
        verdict of the earlier check stay). Only an image that is gone (docker: no such image / manifest
        unknown, after a digest-pinned pull attempt when pulls are on) counts as "no parents declared": the record is marked
        read without labels (undeclared reuse is still caught by a COPY marker). Any other docker error is raised, so the
        digest stays not crownable and is retried next cycle. No-op when the record is missing (the digest then stays
        not crownable: in release mode every measured digest has a record) or already marked."""
        digest = engreview.digest_of(ref)
        rec = engreview.load_record(self.review_dir, digest) if self.review_dir and digest else None
        if rec is None or rec.get("copyline"):
            return
        try:
            info = await self._inspect(ref)
        except InfraError as exc:
            if not _image_gone(str(exc)):
                raise   # daemon down, timeout, unreadable output: retried next cycle, not crownable meanwhile (critic H1)
            info = None
            if getattr(self, "pull", False):
                try:   # a digest-pinned pull: its labels are the authentic ones of the measured image
                    await self._ensure_image(ref)
                    info = await self._inspect(ref)
                except InfraError as exc2:
                    if not _image_gone(str(exc2)):
                        raise
        if info is None:
            log.warning("engine copy line: %s is gone (not on the host, not pullable); counted as no parents declared",
                        digest[:12])
        else:
            rec.update(engreview.copyline_fields(info))
        rec["copyline"] = 1
        engreview.write_record(self.review_dir, digest, rec)

    async def _static_check(self, ref: str, digest: str) -> dict[str, Any]:
        info = await self._inspect(ref)
        violations, warnings, summary = engreview.check_image_config(info)
        if violations:  # already rejected: do not copy anything out of an image that cannot run
            return engreview.new_record(ref, digest, violations, warnings, summary, {})
        files = await self.image_files(ref, summary.get("workdir", ""))
        expected = dict(await self._expected_files())
        v2, w2 = engreview.compare_files(files, expected)
        rec = engreview.new_record(ref, digest, violations + v2, warnings + w2, summary, files)
        if copyline_active(datetime.now(timezone.utc)):  # ENGINE_COPYLINE_START_UTC; records stay as before while off
            rec.update(engreview.copyline_fields(info))  # parents / ownership labels, only when the image has them
            rec["copyline"] = 1  # this record was written with the copy-line fields (the validator reads only these)
        return rec

    async def _expected_files(self) -> dict[str, str]:
        """Tokenizer/config files the candidate must match: the reference image's copy at the same path, else the pinned
        weights directory's file of the same name."""
        exp = {}
        if self._reference_files is None:
            self._reference_files = ({} if not self.reference_ref
                                     else await self.image_files(self.reference_ref, ""))
        weights = engreview.weights_file_hashes(self.weights_dir)
        ref_files = {k: v for k, v in self._reference_files.items() if not v.startswith("!")}
        # The reference image and the pinned weights directory describe the same model. If they disagree, this validator
        # is misconfigured (wrong weights directory for this reference): every candidate would be rejected for a fault
        # that is not theirs, and the rejection is cached. Void instead.
        for path, sha in sorted(ref_files.items()):
            want = weights.get(os.path.basename(path))
            if want is not None and not want.startswith("!") and engreview.model_like(path) and want != sha:
                raise ReferenceSelfCheckFailed(
                    f"the reference image's {path} differs from the weights directory's {os.path.basename(path)}: "
                    "weights directory and reference image do not belong together")
        exp.update(weights)  # bare file names
        exp.update(ref_files)  # full paths, looked up first
        return exp

    async def image_files(self, ref: str, workdir: str) -> dict[str, str]:
        """sha256 of every tokenizer/config file found at the checked paths of the image (path -> hash). The image is only
        created, never started (``docker create``); each file is copied to a private temporary directory."""
        dirs = list(engreview.CHECKED_DIRS) + ([workdir.rstrip("/")] if workdir.startswith("/") and workdir != "/" else [])
        name = "reax-chk-" + secrets.token_hex(6)
        code, _, err = await _run(["docker", "create", "--name", name, "--network", "none", "--entrypoint", "/nonexistent",
                                   ref], 120)
        if code != 0:
            raise InfraError(f"docker create for the submission check failed: {clean(err)}")
        os.makedirs(self.review_dir, mode=0o700, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="chk-", dir=self.review_dir)
        found: dict[str, str] = {}
        try:
            for d in dict.fromkeys(dirs):
                for fname in engreview.CHECKED_FILE_NAMES:
                    path = (d.rstrip("/") + "/" + fname)
                    dest = os.path.join(tmp, "f")
                    c, _, _ = await _run(["docker", "cp", f"{name}:{path}", dest], 60,
                                         watch=(dest, engreview.MAX_CHECKED_FILE_BYTES))
                    if c == CP_TOO_LARGE:
                        found[path] = "!file larger than the check limit"
                    elif c == 0:
                        found[path] = engreview.hash_regular_file(dest)
                    if os.path.lexists(dest):
                        if os.path.isdir(dest) and not os.path.islink(dest):
                            shutil.rmtree(dest, ignore_errors=True)
                        else:
                            os.unlink(dest)
        finally:
            await _run(["docker", "rm", "-fv", name], 120)
            shutil.rmtree(tmp, ignore_errors=True)
        return found

    def weights_identity(self) -> str:
        """sha256 over the relative path and size of every file under the weights directory, the content of the small
        JSON files (config, tokenizer, index) and the header bytes of every safetensors file (tensor names, dtypes,
        shapes, offsets) plus a 64 KiB sample of the tensor data at both ends. Cheap enough to take at every start (a
        content hash of the weights themselves would not be). ``unreadable:...`` when it cannot be read at all (missing,
        empty): that is not a change of the weights."""
        h = hashlib.sha256()
        n = 0

        def fail(exc: OSError) -> None:
            raise exc

        def sample(fh, at: int) -> None:
            fh.seek(at)
            h.update(hashlib.sha256(fh.read(IDENTITY_SAMPLE_BYTES)).digest())

        try:
            if not os.path.isdir(self.weights_dir):
                return "unreadable:missing"
            for root, dirs, files in os.walk(self.weights_dir, onerror=fail):
                dirs.sort()
                for f in sorted(files):
                    path = os.path.join(root, f)
                    size = os.path.getsize(path)
                    h.update(f"{os.path.relpath(path, self.weights_dir)}\0{size}\n".encode())
                    if f.endswith(".json") and size <= IDENTITY_JSON_MAX_BYTES:
                        with open(path, "rb") as fh:
                            h.update(hashlib.sha256(fh.read()).digest())
                    elif f.endswith(".safetensors"):
                        with open(path, "rb") as fh:
                            head = fh.read(8)
                            h.update(head)
                            if len(head) == 8:
                                hlen = min(int.from_bytes(head, "little"), IDENTITY_HEADER_MAX_BYTES)
                                h.update(hashlib.sha256(fh.read(hlen)).digest())
                                # two fine-tunes of one architecture share the header: add a sample of the tensor data
                                sample(fh, 8 + hlen)
                                if size > 8 + hlen + IDENTITY_SAMPLE_BYTES:
                                    sample(fh, size - IDENTITY_SAMPLE_BYTES)
                    n += 1
        except OSError as exc:
            return f"unreadable:{type(exc).__name__}"
        return h.hexdigest() if n else "unreadable:empty"

    async def owner_of(self, ref: str) -> Optional[str]:
        """The hotkey the image is bound to (its label), or "" if none. On any failure (also a cancelled pull) the image
        is removed again; the caller discards it when the label names another hotkey."""
        try:
            return await self._owner_of(ref)
        except BaseException:
            await self.discard(ref)
            raise

    async def _owner_of(self, ref: str) -> Optional[str]:
        await self._ensure_image(ref)
        code, out, err = await _run(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", ref], 60)
        if code != 0:
            raise InfraError(f"docker inspect failed: {clean(err)}")
        try:
            labels = json.loads(out) or {}
        except ValueError:
            labels = {}
        value = labels.get(HOTKEY_LABEL, "") if isinstance(labels, dict) else ""
        return value if isinstance(value, str) else ""

    async def start(self, ref: str, model: str, params: TrackParams, keep_image: bool = False) -> Engine:
        try:
            return await self._start(ref, model, params, keep_image)
        except AwaitingReview:
            raise  # the image stays for the operator's review
        except BaseException:
            if not keep_image:  # every failed start (pull, docker run, cancellation) gives the image back
                await self.discard(ref)
            raise

    async def _start(self, ref: str, model: str, params: TrackParams, keep_image: bool) -> Engine:
        if self.release_mode and self.docker_runtime != RELEASE_RUNTIME:  # also for callers that skip preflight
            raise ReferenceSelfCheckFailed(f"release mode needs REAX_ENGINE_DOCKER_RUNTIME={RELEASE_RUNTIME}")
        config_dir = await self.config_dir()
        await self._ensure_image(ref)
        if ref not in self.protected_images:
            await self.review(ref)  # the candidate; the release reference and the vLLM image are ours
        d, token = _session_dir(self.workdir)
        name = "reax-eng-" + secrets.token_hex(6)
        cmd = ["docker", "run", "-d", "--name", name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=16g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", "1024", "--ulimit", "nofile=4096:4096", "--memory", self.memory,
               "--memory-swap", self.memory, "--cpus", self.cpus, "--shm-size", "2g", "--user", self.user,
               "--log-driver", "local", "--log-opt", "max-size=10m", "--log-opt", "max-file=2",
               "--gpus", f"device={self.gpu}",
               "-v", f"{self.weights_dir}:/model:ro", "-v", f"{os.path.join(d, 'io')}:/run/reax",
               "-v", f"{os.path.join(self.fla_config_root, os.path.basename(config_dir))}:{config_dir}:ro",
               "-e", "RUNTIME_UDS=/run/reax/" + SOCKET_NAME, "-e", "S1M_INTERNAL_TOKEN_FILE=/run/reax/token",
               "-e", "FLA_CACHE_MODE=strict", "-e", f"FLA_CONFIG_DIR={config_dir}",
               "-e", "PLUMB_MODEL_DIR=/model", "-e", "HF_HUB_OFFLINE=1", "-e", "OPENAI_API_KEY=",
               "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
        if self.docker_runtime:
            cmd += ["--runtime", self.docker_runtime]
        cmd.append(ref)
        try:
            code, _, err = await _run(cmd, 300)
        except BaseException:
            await _abort_start(name, d)  # cancelled before an Engine (and its stop handle) exists
            raise
        if code != 0:
            await _run(["docker", "rm", "-fv", name], 120)
            shutil.rmtree(d, ignore_errors=True)
            raise InfraError(f"docker run failed: {clean(err)}")

        async def stop() -> None:
            text = ""
            try:
                _, out, err = await _run(["docker", "logs", "--tail", "200", name], 60)
                text = out + err
            finally:
                await _run(["docker", "rm", "-fv", name], 120)
                _keep_log(self.workdir, d, name, text)
                if not keep_image:
                    await self.discard(ref)

        async def pause() -> None:
            await _run(["docker", "pause", name], 60)

        async def resume() -> None:
            await _run(["docker", "unpause", name], 60)

        async def gpu_busy() -> bool:
            return await candidate_gpu_busy(name, self.gpu)

        client = UdsSidecarBackend(model, os.path.join(d, "io", SOCKET_NAME), token, params.concurrency,
                                   params.request_timeout_s)
        return Engine(ref, client, stop, pause, resume, gpu_busy if self.nvml_check and self.docker_runtime != RELEASE_RUNTIME else None)


class DevRunner:
    """Local test chains only: ``{"<image>@sha256:<digest>": ["python", "fake_engine.py", "--variant", "fast"]}``.

    No hotkey binding (copy protection falls back to the first committer of a digest)."""

    binds_hotkey = False

    def __init__(self, workdir: str, mapping: dict[str, list[str]]):
        self.workdir = workdir
        self.mapping = mapping

    @classmethod
    def from_file(cls, workdir: str, path: str) -> "DevRunner":
        with open(path, encoding="utf-8") as fh:
            return cls(workdir, json.load(fh))

    async def owner_of(self, ref: str) -> Optional[str]:
        return None

    async def discard(self, ref: str) -> None:
        return None

    async def start(self, ref: str, model: str, params: TrackParams, keep_image: bool = False) -> Engine:
        argv = self.mapping.get(ref)
        if not argv:
            raise EngineError("image not available to the dev runner")
        d, token = _session_dir(self.workdir)
        sock = os.path.join(d, "io", SOCKET_NAME)
        env = {**os.environ, "RUNTIME_UDS": sock, "S1M_INTERNAL_TOKEN_FILE": os.path.join(d, "io", "token"),
               "OPENAI_API_KEY": ""}
        log_path = os.path.join(d, "engine.log")
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(argv, env=env, stdout=log_fh, stderr=subprocess.STDOUT)

        async def stop() -> None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGCONT)
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, 10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            log_fh.close()
            with open(log_path, errors="replace") as fh:
                text = fh.read()
            _keep_log(self.workdir, d, os.path.basename(d), text)

        async def pause() -> None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGSTOP)

        async def resume() -> None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGCONT)

        client = UdsSidecarBackend(model, sock, token, params.concurrency, params.request_timeout_s)
        return Engine(ref, client, stop, pause, resume)


# ------------------------------------------------------------------------------------------------ items
def _perturb_state(state: Any, nonce: str) -> Any:
    """A secret record id makes every item unpredictable, so no answer table can cover it."""
    if isinstance(state, dict):
        return {"ref": nonce, **state}
    if isinstance(state, list):
        return {"ref": nonce, "entries": state}
    return f"[ref {nonce}] {state}"


def fresh_decisions(n: int, model: str) -> list[Decision]:
    out = []
    for it in generate_s1_items(n):
        it = S1Item(it.item_id, _perturb_state(it.state, secrets.token_hex(4)), it.kind, it.instructions, it.criteria)
        out.append(parse_decision(it.decision(), model))
    return out


class ItemTape:
    """The timed, probe and latency items of one session, so a second session can replay exactly the same items (the
    declared parent's reference run right after a derived entry's session; docs/ENGINE-COMPETITION.md, "Reusing other
    entries"). The first session records; ``replay()`` gives a reader that returns the recorded items in the same order
    and falls back to fresh items once the recording runs out (a session that ended early)."""

    def __init__(self) -> None:
        self._calls: list[tuple[tuple[Any, ...], Any]] = []
        self._pos: Optional[int] = None  # None: recording

    def replay(self) -> "ItemTape":
        out = ItemTape()
        out._calls, out._pos = self._calls, 0
        return out

    @property
    def recorded(self) -> int:
        return len(self._calls)

    def _take(self, key: tuple[Any, ...], make: Callable[[], Any]) -> Any:
        if self._pos is None:
            value = make()
            self._calls.append((key, value))
            return value
        if self._pos < len(self._calls) and self._calls[self._pos][0] == key:
            self._pos += 1
            return self._calls[self._pos - 1][1]
        self._pos = len(self._calls)  # out of step: everything from here on is fresh
        return make()

    def fresh(self, n: int, model: str) -> list[Decision]:
        return self._take(("fresh", n, model), lambda: fresh_decisions(n, model))

    def probes(self, pairs: int, model: str) -> tuple[list[Decision], list[Decision]]:
        return self._take(("probes", pairs, model), lambda: probe_decisions(pairs, model))


def probe_decisions(pairs: int, model: str) -> tuple[list[Decision], list[Decision]]:
    """Base items, then variants that reuse a base item's state with another item's question, or its question with
    another item's state. An engine that caches by state (or by question) answers the variant from its cache."""
    a = generate_s1_items(pairs)
    b = generate_s1_items(pairs)
    base, variants = [], []
    for i, (x, y) in enumerate(zip(a, b)):
        nonce = secrets.token_hex(4)
        state = _perturb_state(x.state, nonce)
        base.append(parse_decision(S1Item(x.item_id, state, x.kind, x.instructions, x.criteria).decision(), model))
        if i % 2 == 0:  # same state, other question
            v = S1Item(secrets.token_hex(16), state, y.kind, y.instructions, y.criteria)
        else:  # same question, other state
            v = S1Item(secrets.token_hex(16), _perturb_state(y.state, nonce), x.kind, x.instructions, x.criteria)
        variants.append(parse_decision(v.decision(), model))
    return base, variants


# ------------------------------------------------------------------------------------------------ measuring
Answers = dict[str, Optional[tuple[float, ...]]]


async def _answers(client: UdsSidecarBackend, decisions: Sequence[Decision]) -> Answers:
    results = await client.decide_each(decisions, retries=0)
    out: Answers = {}
    for d, a in zip(decisions, results):
        if isinstance(a, BaseException):
            out[d.ident] = None
            continue
        try:
            out[d.ident] = tuple(float(a["probabilities"][k]) for k in d.question.keys)
        except (KeyError, TypeError, ValueError):
            out[d.ident] = None
    return out


async def timed_run(client: UdsSidecarBackend, decisions: Sequence[Decision]) -> tuple[float, Answers]:
    """Decisions per second over the whole batch (validator clock), with every answer."""
    started = time.monotonic()
    answers = await _answers(client, decisions)
    elapsed = time.monotonic() - started
    return len(decisions) / elapsed if elapsed > 0 else 0.0, answers


async def latencies(client: UdsSidecarBackend, decisions: Sequence[Decision]) -> tuple[list[float], Answers]:
    lat, out = [], {}
    for d in decisions:
        started = time.monotonic()
        out.update(await _answers(client, [d]))
        lat.append(time.monotonic() - started)
    return lat, out


PAUSE_DRAIN_S = 0.5
PROBE_GRACE_S = 120.0  # after /health is up, the authenticated test decision must succeed within this time


async def wait_ready(engine: Engine, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    healthy_since = None
    while time.monotonic() < deadline:
        if engine.client.socket_ok() and await engine.client.ready():
            if await engine.client.probe():
                return True
            healthy_since = healthy_since or time.monotonic()
            if time.monotonic() - healthy_since > PROBE_GRACE_S:
                return False
        await asyncio.sleep(1.0)
    return False


def _complete(answers: Answers) -> bool:
    return all(v is not None for v in answers.values())


class _Solo:
    """Measure one engine while the other one is paused, so neither can disturb the other's timing."""

    def __init__(self, active: Engine, idle: Optional[Engine], pause: bool = True):
        self.active, self.idle = active, (idle if pause else None)

    async def __aenter__(self):
        if not self.active.client.socket_ok():
            raise EngineError("engine socket replaced")
        if self.idle is not None:
            await self.idle.pause()
            await asyncio.sleep(PAUSE_DRAIN_S)  # GPU work the idle engine queued before the pause drains first
            if self.idle.gpu_busy is not None and await self.idle.gpu_busy():
                await self.idle.resume()
                raise EngineError("the candidate kept using the GPU while it was paused")
        return self.active.client

    async def __aexit__(self, *exc):
        if self.idle is not None:
            await self.idle.resume()


async def evaluate(runner: Runner, reference_ref: str, candidate_ref: str, track: str, params: TrackParams,
                   block: int = 0, digest: str = "", progress: Optional[dict[str, Any]] = None) -> EvalResult:
    """Measure one candidate against the reference in one session (both engines loaded on the same GPU; the idle
    one is paused during every measurement).

    Raises ReferenceError_ when the reference engine fails before the candidate starts (inconclusive),
    ReferenceFailedWithCandidate when it fails while the candidate is loaded, and InfraError when the candidate
    image cannot be pulled or started for infrastructure reasons. Wrap it in ``asyncio.wait_for`` with
    ``params.eval_budget_s``. ``progress``: set to ``{"measuring": True}`` when the first timed request of the pair
    measurement starts, so a caller whose budget expires can tell "never got to measure" from "measured too slowly".
    ``progress["items"]`` (optional ItemTape): the measured items are recorded on it, or replayed from it.
    """
    model = params.pool
    res = EvalResult(digest=digest or candidate_ref.rsplit(":", 1)[-1], track=track, ok=False, block=block)
    ref = cand = None
    try:
        try:
            ref = await runner.start(reference_ref, model, params, keep_image=True)
        except (EngineError, InfraError) as exc:
            raise ReferenceError_(f"reference engine did not start: {exc}") from None
        if not await wait_ready(ref, params.start_timeout_s):
            raise ReferenceError_("reference engine not ready")
        await reference_self_check(ref, params)
        baseline = await reference_baseline(ref, params)  # the reference ALONE on the GPU, before the candidate exists
        lat_baseline = await reference_latency_baseline(ref, params)
        try:
            cand = await runner.start(candidate_ref, model, params)
        except EngineError as exc:
            res.reason = clean(f"start: {exc}")
            return res
        if not await wait_ready(cand, params.start_timeout_s):
            res.reason = "not ready within the start timeout (or failed the authenticated test decision)"
            return res
        try:
            return await _measure(ref, cand, params, res, baseline, lat_baseline, progress)
        except ReferenceError_ as exc:
            raise ReferenceFailedWithCandidate(str(exc)) from None
        except EngineError as exc:  # replaced socket, GPU use while paused: the candidate cheats or is broken
            res.reason = clean(str(exc))
            return res
    finally:
        for eng in (cand, ref):
            if eng is None:
                continue
            try:
                await eng.resume()
                await eng.client.close()
            except Exception:  # noqa: BLE001 - cleanup of one engine must not skip the other
                pass
            try:
                await eng.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("could not stop engine %s: %r", eng.ref[:60], exc)


async def reference_self_check(ref: Engine, params: TrackParams) -> None:
    """Repeat a fixed, public check set before any candidate is loaded (not a scoring set)."""
    decisions = [parse_decision(it.decision(), params.pool)
                 for it in generate_s1_items(params.timed_items, seed=0x52454158)]
    first = await _answers(ref.client, decisions)
    second = await _answers(ref.client, decisions)
    if len(first) != len(decisions) or len(second) != len(decisions) or not _complete(first) or not _complete(second):
        raise ReferenceSelfCheckFailed("reference self-check missing answers")
    check = check_correctness(second, first, params)
    if not check.ok:
        raise ReferenceSelfCheckFailed("reference self-check: " + check.reason)


async def reference_baseline(ref: Engine, params: TrackParams) -> float:
    """Throughput of the reference engine alone on the GPU (warm-up first, then one timed run). Every later reference
    run of the session must stay within ``params.reference_drift`` of it, else a candidate is slowing the reference
    (queued GPU kernels keep running while a container is paused). Raises ReferenceError_ if the reference fails."""
    model = params.pool
    await _answers(ref.client, fresh_decisions(min(32, params.timed_items), model))
    tp, ans = await timed_run(ref.client, fresh_decisions(params.timed_items, model))
    if not _complete(ans) or tp <= 0:
        raise ReferenceError_("reference engine failed on the baseline run")
    return tp


async def reference_latency_baseline(ref: Engine, params: TrackParams) -> float:
    """Median one-at-a-time latency of the reference alone on the GPU (the latency gate's counterpart of the
    throughput baseline). Raises ReferenceError_ if the reference fails."""
    lat, ans = await latencies(ref.client, fresh_decisions(params.latency_items, params.pool))
    if not _complete(ans) or not lat:
        raise ReferenceError_("reference engine failed on the latency baseline")
    return median(lat)


def _check_drift(tp: float, baseline: Optional[float], params: TrackParams) -> None:
    """One-sided: only a reference that is SLOWER than its solo baseline is suspicious (a candidate's queued kernels
    slow it down). A faster one cannot help a candidate, because the baseline is the floor of the speed denominator."""
    if baseline and tp < baseline * (1 - params.reference_drift):
        raise ReferenceError_(f"reference throughput {tp:.1f} dec/s is more than {params.reference_drift:.0%} below its solo "
                              f"baseline {baseline:.1f} dec/s (something slowed it down)")


def _check_latency_drift(p50: float, baseline: Optional[float], params: TrackParams) -> None:
    if baseline and p50 > baseline * (1 + params.reference_drift):
        raise ReferenceError_(f"reference latency p50 {p50 * 1000:.1f} ms is more than {params.reference_drift:.0%} above its "
                              f"solo baseline {baseline * 1000:.1f} ms (something slowed it down)")


async def _measure(ref: Engine, cand: Engine, params: TrackParams, res: EvalResult,
                   baseline: Optional[float] = None, lat_baseline: Optional[float] = None,
                   progress: Optional[dict[str, Any]] = None) -> EvalResult:
    model = params.pool
    tape = (progress or {}).get("items")
    if not isinstance(tape, ItemTape):
        tape = ItemTape()  # records, nobody replays it
    warm = tape.fresh(min(32, params.timed_items), model)
    async with _Solo(ref, cand, params.pause_idle) as c:
        await _answers(c, warm)
    async with _Solo(cand, ref, params.pause_idle) as c:
        await _answers(c, warm)

    ref_tp, cand_tp, ref_ans, cand_ans = [], [], {}, {}
    last_items: list[Decision] = []
    for run in range(params.runs):
        items = tape.fresh(params.timed_items, model)
        order = [(ref, cand), (cand, ref)]
        if run % 2:
            order.reverse()  # ABBA: neither engine always runs first
        for eng, other in order:
            async with _Solo(eng, other, params.pause_idle) as c:
                if progress is not None:
                    progress["measuring"] = True
                tp, ans = await timed_run(c, items)
            if eng is ref:
                if not _complete(ans):
                    raise ReferenceError_("reference engine failed on timed items")
                _check_drift(tp, baseline, params)
                ref_tp.append(tp)
                ref_ans.update(ans)
            else:
                cand_tp.append(tp)
                cand_ans.update(ans)
                if not _complete(ans):  # stop early: no need to sit through more timeouts
                    missing = sum(v is None for v in ans.values())
                    res.reason = f"{missing} of {len(ans)} timed items failed"
                    return res
        last_items = items
    res.items = len(ref_ans)
    res.throughput = median(cand_tp)
    res.ref_throughput = median(ref_tp)
    corr = check_correctness(cand_ans, ref_ans, params)  # type: ignore[arg-type]
    res.mean_distance, res.max_distance, res.top_agreement = corr.mean_distance, corr.max_distance, corr.top_agreement
    if not corr.ok:
        res.reason = f"correctness: {corr.reason}"
        return res

    base, variants = tape.probes(params.probe_pairs, model)
    async with _Solo(ref, cand, params.pause_idle) as c:
        p_ref = await _answers(c, base + variants)
    if not _complete(p_ref):
        raise ReferenceError_("reference engine failed on probe items")
    async with _Solo(cand, ref, params.pause_idle) as c:
        _, p_base = await latencies(c, base)  # one at a time: a cache sees every base item first
        _, p_var = await latencies(c, variants)
    probe = check_correctness({**p_base, **p_var}, p_ref, params, top_agreement=False)  # type: ignore[arg-type]
    res.probe_max_distance = probe.max_distance
    if not probe.ok:
        res.reason = f"anti-cache probe: {probe.reason}"
        return res

    again = last_items[: params.determinism_items]
    async with _Solo(cand, ref, params.pause_idle) as c:
        _, rep = await latencies(c, again)
    res.determinism_max_distance = max_distance(rep, {d.ident: cand_ans[d.ident] for d in again},  # type: ignore[misc]
                                                params.temperature)
    if res.determinism_max_distance > params.determinism_tau:
        res.reason = (f"determinism: a repeated item moved by {res.determinism_max_distance:.3f} "
                      f"> {params.determinism_tau}")
        return res

    lat_items = tape.fresh(params.latency_items, model)
    async with _Solo(ref, cand, params.pause_idle) as c:
        ref_lat, ref_l_ans = await latencies(c, lat_items)
    async with _Solo(cand, ref, params.pause_idle) as c:
        cand_lat, cand_l_ans = await latencies(c, lat_items)
    if not _complete(ref_l_ans):
        raise ReferenceError_("reference engine failed on latency items")
    lat_corr = check_correctness(cand_l_ans, ref_l_ans, params, top_agreement=False)  # type: ignore[arg-type]
    if not lat_corr.ok:
        res.reason = f"correctness at concurrency 1: {lat_corr.reason}"
        return res
    res.latency_p50_s, res.ref_latency_p50_s = median(cand_lat), median(ref_lat)
    _check_latency_drift(res.ref_latency_p50_s, lat_baseline, params)
    ref_p50 = min(res.ref_latency_p50_s, lat_baseline) if lat_baseline else res.ref_latency_p50_s  # a slowed reference loosens nothing
    if res.latency_p50_s > params.max_latency_ratio * ref_p50:
        res.reason = (f"latency p50 {res.latency_p50_s * 1000:.1f} ms > {params.max_latency_ratio} x reference "
                      f"{ref_p50 * 1000:.1f} ms")
        return res
    # the denominator is never below the reference's solo baseline: slowing the reference after the baseline gains nothing
    res.score = speed_score(cand_tp, [max(r, baseline) for r in ref_tp] if baseline else ref_tp)
    res.ok = True
    return res


async def reference_alone_ok(runner: Runner, reference_ref: str, params: TrackParams) -> bool:
    """After a reference failure with a candidate loaded: does the reference work on its own?"""
    eng = None
    try:
        eng = await runner.start(reference_ref, params.pool, params, keep_image=True)
        if not await wait_ready(eng, params.start_timeout_s):
            return False
        return _complete(await _answers(eng.client, fresh_decisions(16, params.pool)))
    except Exception:  # noqa: BLE001
        return False
    finally:
        if eng is not None:
            try:
                await eng.client.close()
            finally:
                await eng.stop()
