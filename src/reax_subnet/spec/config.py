"""Configuration of the specialist evaluator (environment variables ``REAX_SPEC_*``, see docs/SPECIALIST-TRACKS.md)."""

from __future__ import annotations

import calendar
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .. import competition_split as split
from ..competition_split import ROTATION_DAYS
from . import machine, rules, sampling, scoring, vllm
from .commitment import _valid_repo

log = logging.getLogger("reax.spec.config")

MAX_EVALS_PER_HOUR = 8            # served-model evaluations per hour, all tracks together (incumbent runs count); 6 before rc8
MAX_ATTEMPTS = 3
START_FAIL_WINDOW = 6
START_FAIL_LIMIT = 3
START_FAIL_LOG_SIZE = 12
PLAYOFF_MAX_ABORTS = 3
CHAMPION_LIVENESS_S = 72 * 3600
PUBLISH_DEADLINE_BLOCKS = 7200    # the repo must be public within ~24 h of the commitment (docs/SPECIALIST-TRACKS.md)
RETRY_UNAVAILABLE_S = 600
WAIT_REPO_PROBES_PER_CYCLE = 32    # metadata only, per track, in commitment order
MAX_OPEN_PER_COLDKEY_TRACK = 2
MAX_NEW_PER_COLDKEY_WEEK = 4
MIN_FREE_GB = 60                  # free space on the work directory's filesystem before a submission is downloaded
MAX_LEADERS = 3                   # champion + two former champions (the split layer decides the shares)
BASE_RETRY_CYCLES = 6             # a declared base that could not be fetched is tried again after this many cycles
BASE_BUDGET_GB = 40               # base model downloads per cycle (spec/bases.py)
BASE_MIN_FREE_GB = 50             # free space kept on the base directory's filesystem after a base download


@dataclass(frozen=True)
class TrackDef:
    track_id: str
    category: str
    start_ts: float
    days: float = ROTATION_DAYS
    end_ts: float | None = None

    def active(self, now: float) -> bool:
        return self.start_ts <= now < (self.end_ts if self.end_ts is not None else self.start_ts + self.days * 86400)


def _ts(text: str) -> float:
    return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))


def _from_schedule() -> tuple[TrackDef, ...]:
    """The release schedule (competition_split.TRACK_SCHEDULE) is the one source of truth; the sealed category of a
    track is its id. REAX_SPEC_TRACKS_FILE (development only) replaces it."""
    return tuple(TrackDef(t.track_id, t.track_id, t.start.timestamp(),
                          (t.end - t.start).total_seconds() / 86400, t.end.timestamp())
                 for t in split.release_schedule())


DEFAULT_TRACKS = _from_schedule()


@dataclass
class SpecConfig:
    sealed_dir: str
    work_dir: str
    state_path: str
    tracks: tuple[TrackDef, ...] = field(default_factory=_from_schedule)
    rules: rules.TrackRules | None = None   # None = the release rules at the time of each check (``rules_for``); an explicit object always wins
    sample_size: int = sampling.SAMPLE_SIZE
    max_evals_per_hour: int = MAX_EVALS_PER_HOUR
    confirm_delay_s: float = machine.CONFIRM_DELAY_S
    entry_floor: float = 0.0
    v3_decisions: bool = machine.V3_DECISIONS   # decision rule v3 (machine.py); False = the pre-v3 rule, bit for bit
    x_tier_mode: str = sampling.X_TIER_MODE     # cross-author tier: "off" | "shadow" | "scored" | "binding" (sampling.py)
    x_tier_share: float = sampling.X_TIER_SHARE
    g_med: dict[str, float] = field(default_factory=lambda: {"generalist": split.GENERALIST_G_MED}
                                            if split.GENERALIST_G_MED is not None else {})   # binding mode: frozen median gap per track id (REAX_SPEC_G_MED)
    gap_penalty_cap: float = scoring.GAP_PENALTY_CAP        # binding mode: largest penalty in points (math.inf = no cap)
    nc_r1: bool = False                # near-copy rule R1: twins / weight parents (incumbent included) must also win on competence
    nc_r2: bool = False                # R2: temperature-invariant twin test + relative order tie
    nc_r3: bool = False                # R3: weight check robust to orthogonal noise (alpha) + transitive lineage
    nc_near_champ: bool = False        # near-champion rule: a challenger close to the incumbent's weights needs more items right + 3 points
    gpu_device: str = "0"
    gpu_total_gb: float = 0.0          # 0 = read the card size from nvidia-smi
    gpu_budget_gb: float = vllm.GPU_BUDGET_GB
    image: str = vllm.VLLM_IMAGE
    base_dir: str = ""
    user: str = vllm.SPEC_USER         # uid:gid of the vLLM container (not the validator's uid)
    docker_runtime: str = ""           # e.g. runsc (gVisor); same idea as REAX_ENGINE_DOCKER_RUNTIME
    cpus: str = "8"
    min_free_bytes: int = MIN_FREE_GB * 1024 ** 3
    base_fetch: bool = True            # download a missing declared base into base_dir (spec/bases.py)
    base_retry_cycles: int = BASE_RETRY_CYCLES
    base_budget_bytes: int = BASE_BUDGET_GB * 1024 ** 3
    generalist_base_budget_bytes: int = 80 * 1024 ** 3
    base_min_free_bytes: int = BASE_MIN_FREE_GB * 1024 ** 3
    base_preseed: tuple[str, ...] = ()  # "<hf id>@<revision>" fetched in the background when the evaluator starts
    container_mem: str = vllm.CONTAINER_MEM                 # docker --memory / --memory-swap of the candidate container
    startup_timeout_s: float = vllm.STARTUP_TIMEOUT_S       # how long the candidate's vLLM may take to become healthy
    # eval v5.1 (spec/v5wire.py; read only once sampling.EVAL_V5_START_UTC is reached)
    v5_pool_dir: str = ""              # private T/N pool files (mode 0600), REAX_SPEC_V5_POOL_DIR
    v5_w_n: float = 0.4                # weight of the N part; 0 = the T-only fallback (REAX_SPEC_V5_W_N)
    v5_panel: Any = None               # v5.PanelStats (stock-panel mean / SD of T, N, Pub for the pool version), REAX_SPEC_V5_PANEL = its JSON

    def base_budget_for(self, now: float) -> int:
        return self.generalist_base_budget_bytes if split.spec_open_class_active(now) else self.base_budget_bytes

    def rules_for(self, now: float, category: str | None = None) -> rules.TrackRules:
        """The submission rules for a check at ``now``: the injected ``rules`` object if there is one, else the release rules
        of that time (``rules.rules_at``: the open class from competition_split.SPEC_OPEN_CLASS_START_UTC)."""
        return self.rules if self.rules is not None else rules.rules_at(now, category=category)

    def budget_warning(self, now: float) -> str:
        """Non-empty when the open class is active (or starts within 24 h) and ``gpu_budget_gb`` is too small for its largest
        accepted submission: weights + 15 % + 2 GiB (``vllm.gpu_memory_utilization``). Never a reason to refuse to start: the
        box that runs this image before the switch keeps the default budget."""
        need = rules.OPEN_CLASS_MAX_WEIGHT_BYTES / rules.GIB * 1.15 + 2.0
        if self.gpu_budget_gb >= need or not rules.open_class_active(now + 86400):
            return ""
        return (f"REAX_SPEC_GPU_BUDGET_GB={self.gpu_budget_gb:g} is below the {need:.1f} GB the largest open-class submission needs "
                f"({rules.OPEN_CLASS_MAX_WEIGHT_BYTES / rules.GIB:.0f} GiB of weights): entries above "
                f"{(self.gpu_budget_gb - 2.0) / 1.15:.0f} GiB will wait for this host (no strike) until the budget is raised")

    def active_tracks(self, now: float) -> list[TrackDef]:
        return [t for t in self.tracks if t.active(now)]


def load_tracks(path: str) -> tuple[TrackDef, ...]:
    """JSON list of {"track_id", "category", "start": "YYYY-MM-DDTHH:MM:SSZ", "days": 14}."""
    with open(path) as fh:
        raw = json.load(fh)
    return tuple(TrackDef(str(t["track_id"]), str(t["category"]), _ts(t["start"]), int(t.get("days", ROTATION_DAYS))) for t in raw)


def from_env(env: Mapping[str, str] = os.environ, state_dir: str = "") -> SpecConfig:
    sealed = env.get("REAX_SPEC_SEALED_DIR", "")
    if not sealed:
        raise ValueError("REAX_SPEC_SEALED_DIR (host path of the sealed pool, mounted read-only) is required")
    base = state_dir or env.get("REAX_SPEC_STATE_DIR", "") or os.path.expanduser("~/.reax")
    tracks = load_tracks(env["REAX_SPEC_TRACKS_FILE"]) if env.get("REAX_SPEC_TRACKS_FILE") else _from_schedule()
    return SpecConfig(
        sealed_dir=sealed,
        work_dir=env.get("REAX_SPEC_WORK_DIR", os.path.join(base, "spec-work")),
        state_path=env.get("REAX_SPEC_STATE", os.path.join(base, "spec-state.json")),
        tracks=tracks,
        sample_size=int(env.get("REAX_SPEC_SAMPLE_SIZE", sampling.SAMPLE_SIZE)),
        max_evals_per_hour=int(env.get("REAX_SPEC_MAX_EVALS_PER_HOUR", MAX_EVALS_PER_HOUR)),
        v3_decisions=_flag(env, "REAX_SPEC_V3_DECISIONS", machine.V3_DECISIONS),
        nc_r1=_flag(env, "REAX_SPEC_NC_R1", False),
        nc_r2=_flag(env, "REAX_SPEC_NC_R2", False),
        nc_r3=_flag(env, "REAX_SPEC_NC_R3", False),
        nc_near_champ=_flag(env, "REAX_SPEC_NC_NEAR_CHAMP", False),
        x_tier_mode=_x_mode(env.get("REAX_SPEC_X_TIER_MODE", sampling.X_TIER_MODE)),
        x_tier_share=float(env.get("REAX_SPEC_X_TIER_SHARE", sampling.X_TIER_SHARE)),
        g_med={**({"generalist": split.GENERALIST_G_MED} if split.GENERALIST_G_MED is not None else {}),
               **_g_med(env.get("REAX_SPEC_G_MED", ""))},
        gap_penalty_cap=_penalty_cap(env.get("REAX_SPEC_GAP_PENALTY_CAP", "")),
        gpu_device=env.get("REAX_SPEC_GPU_DEVICE", "0"),
        gpu_total_gb=float(env.get("REAX_SPEC_GPU_TOTAL_GB", 0)),
        gpu_budget_gb=float(env.get("REAX_SPEC_GPU_BUDGET_GB", vllm.GPU_BUDGET_GB)),
        container_mem=vllm.parse_container_mem(env.get("REAX_SPEC_CONTAINER_MEM", "")),
        startup_timeout_s=vllm.parse_startup_timeout(env.get("REAX_SPEC_STARTUP_TIMEOUT_S", "")),
        image=env.get("REAX_SPEC_VLLM_IMAGE", vllm.VLLM_IMAGE),
        base_dir=env.get("REAX_SPEC_BASE_DIR", ""),
        user=_user(env.get("REAX_SPEC_UID", vllm.SPEC_USER)),
        docker_runtime=env.get("REAX_SPEC_DOCKER_RUNTIME", "").strip(),
        cpus=env.get("REAX_SPEC_CPUS", "8").strip() or "8",
        min_free_bytes=int(float(env.get("REAX_SPEC_MIN_FREE_GB", MIN_FREE_GB)) * 1024 ** 3),
        base_fetch=env.get("REAX_SPEC_BASE_FETCH", "1").strip().lower() not in ("0", "false", "no", "off"),
        base_retry_cycles=max(1, int(env.get("REAX_SPEC_BASE_RETRY_CYCLES", BASE_RETRY_CYCLES))),
        base_budget_bytes=int(float(env.get("REAX_SPEC_BASE_BUDGET_GB", BASE_BUDGET_GB)) * 1024 ** 3),
        generalist_base_budget_bytes=int(float(env.get("REAX_SPEC_BASE_BUDGET_GB", 80)) * 1024 ** 3),
        base_min_free_bytes=int(float(env.get("REAX_SPEC_BASE_MIN_FREE_GB", BASE_MIN_FREE_GB)) * 1024 ** 3),
        base_preseed=tuple(x.strip() for x in env.get("REAX_SPEC_BASE_PRESEED", "").split(",") if x.strip()),
        v5_pool_dir=env.get("REAX_SPEC_V5_POOL_DIR", "").strip(),
        v5_w_n=_unit_float(env.get("REAX_SPEC_V5_W_N", ""), 0.4, "REAX_SPEC_V5_W_N"),
        v5_panel=_v5_panel(env.get("REAX_SPEC_V5_PANEL", "")),
    )


def _unit_float(value: str, default: float, name: str) -> float:
    """A number in [0, 1]; empty = the default; anything else refuses to start (a wrong weight must not be scored)."""
    value = value.strip()
    if not value:
        return default
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return x


def _v5_panel(value: str):
    """``REAX_SPEC_V5_PANEL``: the JSON of ``v5.panel_from_json`` ({"pool_version", "T": {"mean", "sd"}, "N": .., "Pub": ..}). Empty =
    unscaled; unusable refuses to start (a wrong scale must not decide)."""
    if not value.strip():
        return None
    from . import v5
    return v5.panel_from_json(json.loads(value))


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Strict on/off switch: unset or empty means the release default, an unknown value refuses to start (critic-v3int #1)."""
    raw = (env.get(name) or "").strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be 1/0 (or true/false, yes/no, on/off), got {raw!r}")


def _x_mode(value: str) -> str:
    """An unknown mode is "off" (the safe value), not an error that would stop the validator."""
    value = value.strip().lower()
    if value == "scored":
        # fail closed (critic H2): the gap penalty is not applied in this release, so a scored X tier would count the
        # private items without the penalty. "scored" stays "shadow"; the penalty is applied in "binding" mode only.
        return "shadow"
    return value if value in sampling.X_TIER_MODES else "off"


def _g_med(value: str) -> dict[str, float]:
    """``REAX_SPEC_G_MED``: JSON object {track id: median gap in points}. Anything unusable gives {} (no track has a G_med,
    so binding mode behaves as shadow: no penalty), never an error that would stop the validator; entries that are not
    finite numbers are dropped one by one."""
    if not value.strip():
        return {}
    try:
        raw = json.loads(value)
    except ValueError:
        log.error("REAX_SPEC_G_MED is not valid JSON; no G_med configured, binding mode applies no penalty")
        return {}
    if not isinstance(raw, dict):
        log.error("REAX_SPEC_G_MED must be a JSON object {track id: gap}; no G_med configured")
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            log.error("REAX_SPEC_G_MED: ignoring %r (not a finite number)", k)
            continue
        out[str(k)] = float(v)
    return out


def _penalty_cap(value: str) -> float:
    """``REAX_SPEC_GAP_PENALTY_CAP``: points, or "none"/"inf" for no cap. Empty, negative or unparseable: the default
    (the first-week cap), the safe side."""
    v = value.strip().lower()
    if not v:
        return scoring.GAP_PENALTY_CAP
    if v in ("none", "inf", "infinity", "+inf"):
        return math.inf
    try:
        cap = float(v)
    except ValueError:
        cap = math.nan
    if math.isnan(cap) or cap < 0:
        log.error("REAX_SPEC_GAP_PENALTY_CAP=%r is not usable; using the default %.1f", value, scoring.GAP_PENALTY_CAP)
        return scoring.GAP_PENALTY_CAP
    return cap


def _user(value: str) -> str:
    """``uid:gid`` of the vLLM container; it must not be root."""
    value = value.strip()
    parts = value.split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts) or int(parts[0]) == 0:
        raise ValueError(f"REAX_SPEC_UID must be <uid>:<gid> with a non-root uid, got {value!r}")
    return value


_REVISION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


def base_dir_name(base_model: str) -> str | None:
    """``<hf id>@<revision>`` -> its directory name ``<hf id with / -> __>@<revision>``; None if it is not a well-formed
    name (it comes from a miner's manifest: no path separators, no ``..``, nothing that starts with a dot)."""
    repo, sep, rev = base_model.partition("@")
    if not sep or not _valid_repo(repo) or not _REVISION_NAME.fullmatch(rev) or ".." in rev:
        return None
    return repo.replace("/", "__", 1) + "@" + rev


def dir_base_provider(base_dir: str):
    """Base models prepared by the operator (or fetched by spec/bases.py) as ``<base_dir>/<hf id with / -> __>@<revision>/``."""
    def provider(base_model: str) -> str | None:
        name = base_dir_name(base_model) if base_dir else None
        if name is None:
            return None
        p = os.path.join(base_dir, name)
        return p if os.path.isdir(p) else None
    return provider
