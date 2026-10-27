"""``reax-competition-worker``: runs ONLY the competition cycles (docs/COMPETITIONS.md, "Remote mode").

* Reads the chain read-only (no wallet, no hotkey, no weights, no runtime or router tokens).
* Runs the engine competition (``--engine``) and/or the specialist tracks (``--spec``; both when neither is given) with
  docker, on one GPU under one lock, with the same preflights and anti-cheat as a validator in local mode.
* Writes ``engine-competition.json`` and ``spec-state.json`` atomically into ``REAX_COMPETITIONS_STATE_DIR``; a heartbeat
  refreshes their ``saved_at`` while a cycle runs, so the validator (REAX_COMPETITIONS_MODE=remote) can tell a live worker
  from a dead one (26 h limit).

It runs unchanged (a) as a container on the validator host WITHOUT wallets or tokens mounted, and (b) on a separate
keyless GPU box. Whoever reads its state files pays; this process never sets weights.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from . import competition_split as _csplit
from . import elements
from .competition_split import COMPETITIONS_START_UTC, ENGINE_HELD, SPLIT_PROFILE, release_engine_start, release_v2_start
from .competitions import CompetitionDriver, SpecialistAdapter
from .config import (ENGINE_STATE_FILE, LOCAL_NETWORK_NAMES, MAINNET_NAMES, SPEC_STATE_FILE, CommonConfig, ConfigError,
                     EngineConfig, _bool, _engine, _int, _is_mainnet, load_common)
from .engcomp import RewardParams, sync_start_block_hash
from .engvalidator import TRACKS, EngineCompetition
from .engcomposite import composite_rules_sha256

log = logging.getLogger("reax.worker")


# ------------------------------------------------------------------------------------------------ building the stack
def engine_params(e: EngineConfig):
    """Track and reward parameters: the release constants; development networks may override them (JSON)."""
    tracks, reward = dict(TRACKS), RewardParams()
    if e.dev_params:
        over = json.loads(e.dev_params)
        tracks = {t: dataclasses.replace(p, **over.get("track", {})) for t, p in tracks.items()}
        reward = dataclasses.replace(reward, **over.get("reward", {}))
    return tracks, reward


def build_specialist(sealed_dir: str, e: EngineConfig, env: Mapping[str, str], *, state_path: str = "") -> SpecialistAdapter:
    """The specialist-track evaluator (reax_subnet.spec) behind the adapter the share layer reads. Its schedule is
    competition_split.TRACK_SCHEDULE; REAX_SPEC_TRACKS_FILE is a development-network override only."""
    from .spec import config as spec_config
    from .spec.evaluator import SpecEvaluator

    senv = dict(env)
    senv["REAX_SPEC_SEALED_DIR"] = sealed_dir
    senv.setdefault("REAX_SPEC_GPU_DEVICE", e.gpu)  # one GPU for both
    if state_path:
        senv["REAX_SPEC_STATE"] = state_path
    if e.release_mode and senv.pop("REAX_SPEC_TRACKS_FILE", None):
        log.warning("REAX_SPEC_TRACKS_FILE ignored on finney: the track schedule is a release constant")
    spec_cfg = spec_config.from_env(senv)
    # the free-memory gate uses the configured budget (REAX_SPEC_GPU_BUDGET_GB), not the release default
    return SpecialistAdapter(SpecEvaluator(spec_cfg), gpu=spec_cfg.gpu_device, need_mib=int(spec_cfg.gpu_budget_gb * 1024))


def _chain_id(env: Mapping[str, str]) -> str:
    """``network:netuid`` as config.load_* parses them: whitespace stripped, every mainnet spelling is ``finney``
    (finney defaults to netuid 92), the netuid an integer."""
    network = (env.get("REAX_NETWORK") or "").strip() or "finney"
    if _is_mainnet(network):
        network = "finney"
    netuid = (env.get("REAX_NETUID") or "").strip()
    if netuid.isdigit():
        netuid = str(int(netuid))
    elif not netuid and network == "finney":
        netuid = "92"
    return f"{network}:{netuid}"


def build_local_stack(e: EngineConfig, sealed_dir: str, env: Mapping[str, str], *, run_engine: bool = True,
                      state_dir: str = "", award_gate: Any = None):
    """``(engine, adapter, spec_evaluator)`` for a process that runs the cycles itself (a validator in local mode, or the
    worker). ``engine`` is an ``IdleEngine`` and its file untouched when ``run_engine`` is False (``--spec`` only).
    On finney the engine track starts at ``release_engine_start()``: while it is held no engine cycle ever runs.
    ``award_gate``: the validator's owner-approval gate (award.py); the worker pays nothing and passes none."""
    from .engrunner import DevRunner, DockerRunner
    from .spec.vllm import VLLM_IMAGE

    start = COMPETITIONS_START_UTC if e.release_mode else None
    engine_start = release_engine_start() if e.release_mode else None
    adapter = build_specialist(sealed_dir, e, env, state_path=os.path.join(state_dir, SPEC_STATE_FILE) if state_dir else "") \
        if sealed_dir else None
    if not run_engine:
        return IdleEngine(start), adapter, adapter.evaluator if adapter else None
    if e.runner == "dev":
        runner = DevRunner.from_file(e.workdir, e.dev_map)
    else:
        runner = DockerRunner(e.workdir, e.weights_dir, gpu=e.gpu, docker_runtime=e.docker_runtime,
                              registries=e.registries, memory=e.memory, cpus=e.cpus, nvml_check=e.nvml_check,
                              protected_images=(e.reference, env.get("REAX_SPEC_VLLM_IMAGE", VLLM_IMAGE)),
                              release_mode=e.release_mode, review_dir=e.review_dir, reference_ref=e.reference)
    tracks, reward = engine_params(e)
    composite = composite_from_env(env, e, runner) if _csplit.engine_composite_enabled(env) else None
    if composite is not None:
        runner, track = composite
        log.info("engine competition: composite track %s (%s), rules %s", track.name,
                 ", ".join(f"{s.name} {s.weight:g}" for s in track.slots), composite_rules_sha256(track))
        refs = composite_references(env, e.release_mode)
        engine = EngineCompetition(runner, {track.name: e.reference}, e.state_path, reward=reward,
                                   slot_references={track.name: refs},
                                   interval_s=e.interval_s, max_evals=e.max_evals, start_utc=start, specialist=adapter,
                                   engine_start_utc=engine_start, award_gate=award_gate,
                                   chain_id=_chain_id(env), cycle_gate_file=e.cycle_gate_file,
                                   cycle_gate_max_age_s=float(e.cycle_gate_max_age_s), composites={track.name: track},
                                   v1_bridge=(v1_bridge_from_env(env, e.state_path, {track.name: track}, e.release_mode)
                                              if award_gate is not None else None),  # the worker pays nothing
                                   composite_start_utc=release_v2_start() if e.release_mode else None)
        return engine, adapter, adapter.evaluator if adapter else None
    log.info("engine competition: %s runner, reference %s, split profile %s, starts %s", e.runner, e.reference,
             SPLIT_PROFILE, "NEVER: the engine track is HELD (no engine cycle runs)" if engine_start == ENGINE_HELD
             else (engine_start or start).isoformat() if (engine_start or start) else "immediately (development network)")
    engine = EngineCompetition(runner, {"s1-fast": e.reference}, e.state_path, tracks=tracks, reward=reward,
                               interval_s=e.interval_s, max_evals=e.max_evals, start_utc=start, specialist=adapter,
                               engine_start_utc=engine_start, award_gate=award_gate,
                               chain_id=_chain_id(env), cycle_gate_file=e.cycle_gate_file,
                               cycle_gate_max_age_s=float(e.cycle_gate_max_age_s))
    return engine, adapter, adapter.evaluator if adapter else None


def v1_bridge_from_env(env: Mapping[str, str], state_path: str, composites: Any, release_mode: bool = False):
    """The v1 payout bridge tuple ``(frozen v1 state path, end time, pinned sha256)`` or None, the same for the remote validator
    and the local competitions mode (both must pay the same weights). The frozen file is ``REAX_ENGINE_V1_STATE_PATH`` (default
    ``engine-competition-v1.json`` next to the engine state); every validator needs the same bytes. On finney a configured but
    missing, mismatching or unfinished file refuses the start (ConfigError); at runtime an unusable file still pays nothing."""
    if not (composites and _csplit.ENGINE_V1_BRIDGE_END_UTC is not None and _csplit.ENGINE_V1_BRIDGE_STATE_SHA256):
        return None
    bridge = (env.get("REAX_ENGINE_V1_STATE_PATH")
              or os.path.join(os.path.dirname(os.path.abspath(state_path)), "engine-competition-v1.json"),
              _csplit.ENGINE_V1_BRIDGE_END_UTC, _csplit.ENGINE_V1_BRIDGE_STATE_SHA256)
    if release_mode:
        from .engvalidator import read_frozen_v1

        try:
            read_frozen_v1(bridge[0], bridge[2])
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"frozen v1 engine state {bridge[0]} is unusable ({type(exc).__name__}: {exc}); every validator needs "
                              "the pinned bytes (REAX_ENGINE_V1_STATE_PATH), else the v1 rights are not paid") from exc
    return bridge


def composite_track_from_env(env: Mapping[str, str], release_mode: bool):
    """The composite track (worker and remote validator use the same definition). ``REAX_ENGINE_COMPOSITE_CALIBRATION``:
    JSON file with per-slot calibration, development networks only; on finney the track is the release constant and
    must be release-ready, otherwise this refuses (ConfigError)."""
    from .engcomposite import default_track

    cal = None
    path = env.get("REAX_ENGINE_COMPOSITE_CALIBRATION", "")
    if path:
        if release_mode:
            raise ConfigError("REAX_ENGINE_COMPOSITE_CALIBRATION is not allowed on finney (release constant)")
        with open(path, encoding="utf-8") as fh:
            cal = json.load(fh)
    track = default_track(cal)
    if release_mode and track.release_ready():
        raise ConfigError("composite engine track is not release-ready: " + "; ".join(track.release_ready()))
    if release_mode:
        from .competition_split import release_bridge_problem

        problem = release_bridge_problem()
        if problem:
            raise ConfigError("composite engine track: " + problem)
    return track


def composite_references(env: Mapping[str, str], release_mode: bool) -> dict[str, str]:
    """Reference image per slot: the release constants (config.ENGINE_COMPOSITE_REFERENCE_IMAGES); development networks
    may override them with ``REAX_ENGINE_COMPOSITE_REFS=slot=<image@sha256:...>,...``. Every slot of the active weights
    variant needs one (a constant of a slot the variant does not use, e.g. plumb in v2.1, may stay unset)."""
    from .config import ENGINE_COMPOSITE_REFERENCE_IMAGES
    from .engcomposite import COMPOSITE_WEIGHTS, COMPOSITE_WEIGHTS_VARIANT

    refs = {k: v for k, v in ENGINE_COMPOSITE_REFERENCE_IMAGES.items() if v}
    over = env.get("REAX_ENGINE_COMPOSITE_REFS", "")
    if over:
        if release_mode:
            raise ConfigError("REAX_ENGINE_COMPOSITE_REFS is not allowed on finney (release constants)")
        for part in over.split(","):
            name, _, ref = part.strip().partition("=")
            if name:
                refs[name.strip()] = ref.strip()
    active = COMPOSITE_WEIGHTS[COMPOSITE_WEIGHTS_VARIANT]
    missing = [k for k in (*ENGINE_COMPOSITE_REFERENCE_IMAGES, *active) if k in active and not refs.get(k)]  # every slot of the variant, also one without a dict key
    missing = list(dict.fromkeys(missing))
    bad = [k for k, v in refs.items() if "@sha256:" not in v]
    if missing or bad:
        raise ConfigError("composite reference images missing or not pinned by digest: " + ", ".join(missing + bad))
    return refs


def composite_from_env(env: Mapping[str, str], e: EngineConfig, base: Any):
    """(CompositeRunner, CompositeTrack) for engine competition v2 (REAX_ENGINE_COMPOSITE=1; docs/ENGINE-COMPETITION-V2.md).

    ``REAX_ENGINE_COMPOSITE_WEIGHTS``: ``slot=/weights/dir,...`` for every slot of the track (``vision`` may be left out:
    it uses the ``blink`` directory; one DockerRunner per model,
    same GPU, gates and image settings as ``base``; the engine gets ``REAX_MODEL=<model id>``).
    ``REAX_ENGINE_COMPOSITE_CALIBRATION``: JSON file with per-slot calibration (development networks only); on finney the
    track is the release constant and must be release-ready, otherwise this refuses (ConfigError)."""
    from .engcomposite import MODEL_ENV, SLOT_WEIGHTS_FROM, CompositeRunner
    from .engrunner import DockerRunner

    track = composite_track_from_env(env, e.release_mode)
    dirs = {}
    for part in (env.get("REAX_ENGINE_COMPOSITE_WEIGHTS") or "").split(","):
        name, _, d = part.strip().partition("=")
        if name:
            dirs[name.strip()] = d.strip()
    for slot, source in SLOT_WEIGHTS_FROM.items():  # vision runs the Blink weights: no second directory to configure
        if source in dirs and not dirs.get(slot):
            dirs[slot] = dirs[source]
    missing = [s.name for s in track.slots if not dirs.get(s.name)]
    if missing:
        raise ConfigError("REAX_ENGINE_COMPOSITE_WEIGHTS lacks a weights directory for: " + ", ".join(missing))
    if not isinstance(base, DockerRunner):
        raise ConfigError("the composite engine track needs the docker runner")
    protected = tuple(base.protected_images) + tuple(composite_references(env, e.release_mode).values())
    runners = {}
    for s in track.slots:
        runners[s.params.name] = DockerRunner(
            e.workdir, dirs[s.name], gpu=base.gpu, docker_runtime=base.docker_runtime, registries=base.registries,
            memory=base.memory, cpus=base.cpus, nvml_check=base.nvml_check, protected_images=protected,
            release_mode=base.release_mode, review_dir=base.review_dir, reference_ref=base.reference_ref,
            fla_config_root=base.fla_config_root, model_env={MODEL_ENV: s.model})
    return CompositeRunner(runners), track


class IdleEngine:
    """Placeholder engine of a ``--spec`` only worker: never due, so the driver only runs the specialist cycles."""

    def __init__(self, start_utc: Optional[datetime]):
        self.start_utc = start_utc

    def due(self) -> bool:
        return False

    async def run_cycle(self, snap: Any, exclude: set[int], between: Any = None) -> dict[str, list[dict]]:
        return {}


# ------------------------------------------------------------------------------------------------ state files
def touch_state(path: str, now: Optional[float] = None) -> bool:
    """Refresh ``saved_at`` of an existing state file atomically, without re-serialising live objects (a cycle may be
    changing them in an executor thread). False if the file is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict):
        return False
    raw["saved_at"] = int(time.time() if now is None else now)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".touch-", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(raw, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)  # the validator container reads it with its own uid
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return True


# ------------------------------------------------------------------------------------------------ config
@dataclass(frozen=True)
class WorkerConfig:
    common: CommonConfig
    engine: EngineConfig            # also describes the GPU / release mode when only --spec runs
    spec_sealed_dir: str            # "" = no specialist tracks
    state_dir: str
    run_engine: bool
    run_spec: bool
    poll_s: int = 30
    heartbeat_s: int = 300


def _inside(child: str, parent: str) -> bool:
    child, parent = os.path.abspath(child), os.path.abspath(parent)
    return os.path.commonpath([child, parent]) == parent


def load_worker(env: Mapping[str, str] = os.environ, *, engine: bool = True, spec: bool = True) -> WorkerConfig:
    if not (engine or spec):
        raise ConfigError("nothing to do: ask for --engine and/or --spec")
    common = load_common(env)
    state_dir = os.path.expanduser((env.get("REAX_COMPETITIONS_STATE_DIR") or "").strip())
    if not state_dir or not os.path.isabs(state_dir):
        raise ConfigError("REAX_COMPETITIONS_STATE_DIR (absolute path of the directory the validator reads) is required")
    cfg, sealed = _engine(env, common, want_engine=engine, want_spec=spec, strict=True)
    assert cfg is not None
    cfg = dataclasses.replace(cfg, state_path=os.path.join(state_dir, ENGINE_STATE_FILE))
    spec_work = os.path.expanduser(env.get("REAX_SPEC_WORK_DIR") or "~/.reax/spec-work")
    private = [("REAX_ENGINE_WORKDIR", cfg.workdir), ("REAX_SPEC_WORK_DIR", spec_work), ("REAX_ENGINE_WEIGHTS_DIR", cfg.weights_dir),
               ("REAX_SPEC_SEALED_DIR", sealed)]
    for name, path in private:
        if path and _inside(path, state_dir):
            raise ConfigError(f"{name} ({path}) must not be inside REAX_COMPETITIONS_STATE_DIR: that directory is shown to "
                              "the validator")
    return WorkerConfig(common=common, engine=cfg, spec_sealed_dir=sealed if spec else "", state_dir=state_dir,
                        run_engine=engine, run_spec=spec, poll_s=_int(env, "REAX_WORKER_POLL_S", 30, 1, 3600),
                        heartbeat_s=_int(env, "REAX_WORKER_HEARTBEAT_S", 300, 1, 3600))


# ------------------------------------------------------------------------------------------------ run
async def run_worker(wc: WorkerConfig, chain: Any, env: Mapping[str, str] = os.environ, *, once: bool = False,
                     stack: Optional[tuple] = None) -> None:
    """Poll, and run what is due. ``once``: one pass over everything that is due, then return."""
    os.makedirs(wc.state_dir, mode=0o755, exist_ok=True)
    os.chmod(wc.state_dir, 0o755)
    engine, adapter, spec_eval = stack or build_local_stack(wc.engine, wc.spec_sealed_dir, env, run_engine=wc.run_engine,
                                                            state_dir=wc.state_dir)
    spec_path = os.path.join(wc.state_dir, SPEC_STATE_FILE)
    engine_path = os.path.join(wc.state_dir, ENGINE_STATE_FILE)

    def write_initial() -> None:
        if wc.run_engine:
            engine.save()
            os.chmod(engine_path, 0o644)
        if spec_eval is not None:
            from .spec import state as ST

            ST.save(spec_eval.state, spec_path)
            os.chmod(spec_path, 0o644)

    write_initial()

    def heartbeat() -> None:
        for p in ([engine_path] if wc.run_engine else []) + ([spec_path] if spec_eval is not None else []):
            touch_state(p)

    async def beat() -> None:
        while True:
            await asyncio.sleep(wc.heartbeat_s)
            heartbeat()

    async def after_step() -> None:
        heartbeat()

    driver = CompetitionDriver(engine, adapter, background=False)
    log.info("competition worker: engine=%s spec=%s, state dir %s, network %s netuid %d", wc.run_engine,
             spec_eval is not None, wc.state_dir, wc.common.network, wc.common.netuid)
    hb = asyncio.ensure_future(beat())
    try:
        while True:
            if driver.due():
                try:
                    # the competition code reads the competition element only (docs/DESIGN-MULTIROLE.md section 3),
                    # exactly as the validators do: a dual-role hotkey's entry is evaluated (critic round 3 NEW-3)
                    snap = elements.role_views(await asyncio.to_thread(chain.snapshot)).competition
                    await sync_start_block_hash(chain, snap.block, 180.0)  # same tie order as the validators
                    exclude = {snap.owner_uid} if snap.owner_uid is not None else set()
                    await driver.run(snap, exclude, between=after_step)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("competition cycle failed; retrying at the next poll")
                heartbeat()
            if once:
                return
            await asyncio.sleep(wc.poll_s)
    finally:
        hb.cancel()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reax-competition-worker", description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", action="store_true", help="run the engine competition")
    ap.add_argument("--spec", action="store_true", help="run the specialist tracks")
    ap.add_argument("--once", action="store_true", help="run what is due once and exit (smoke test)")
    return ap


def refuse_custom_finney(network: str, chain: Any) -> str:
    """The validator's genesis check (cli.py): a custom endpoint of finney would run the worker in development mode (the
    engine track open at once). Returns the reason to refuse, "" when the network name is fine."""
    if network in MAINNET_NAMES or network in LOCAL_NETWORK_NAMES:
        return ""
    from .chain import FINNEY_GENESIS

    try:
        genesis = chain.genesis_hash()
    except Exception as exc:  # noqa: BLE001 - fail closed
        return f"cannot read the chain's genesis hash ({exc}); refusing to start"
    if str(genesis).lower() == FINNEY_GENESIS:
        return ("REAX_NETWORK is a custom endpoint of finney (mainnet): set REAX_NETWORK=finney so that the "
                "release-fixed engine competition and split are used (docs/COMPETITIONS.md)")
    return ""


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    both = not (args.engine or args.spec)
    try:
        wc = load_worker(engine=args.engine or both, spec=args.spec or both)
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    from .cli import setup_logging

    setup_logging(wc.common)
    from .spec.review import log_config
    log_config()
    from .chain import SdkChain

    chain = SdkChain(wc.common.network, wc.common.netuid, None, commitments=True)  # read-only: no wallet
    why = refuse_custom_finney(wc.common.network, chain)
    if why:
        sys.exit(why)
    try:
        asyncio.run(run_worker(wc, chain, once=args.once))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
