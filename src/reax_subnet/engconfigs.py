"""Published engine FLA configurations: the GPU-class release gate and the checksum check that guards every session."""

import hashlib
import math
import os
import re
from pathlib import Path

FLA_CONFIG_ROOT = "/opt/reax/fla-configs"
GPU_CONFIG_DIRS = {
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "rtx-pro-6000-blackwell-server",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "rtx-pro-6000-blackwell",
}
# Add a GPU class only together with its idle-GPU measured configs, their provenance/checksums and the measured
# relative run-to-run throughput standard deviation. The reference image (9bd41632) does not contain the configs:
# the runner mounts the verified directory read-only into every engine container (see verify_published_configs).
PUBLISHED_GPU_CLASSES: frozenset[str] = frozenset({"NVIDIA RTX PRO 6000 Blackwell Server Edition"})
# sha256 of each published class's SHA256SUMS file; the SHA256SUMS file in turn pins every config file.
PINNED_CONFIG_MANIFEST_SHA256 = {
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "2afd76102700d3a26673cd4bcc4372bbf8cf0bc548580aa5da6b4560eab0fc9d",
}
# 11 idle fresh-process reference samples on that class: mean 65.70 dec/s, sample std / mean = 0.00167 (board #21985).
PINNED_REFERENCE_NOISE_STD = 0.00167
ENGINE_MARGIN_FLOOR = 0.03
ENGINE_NOISE_MULTIPLIER = 3.0


def required_engine_margin(noise_std: float) -> float:
    """Noise is dimensionless: sample std(throughputs) / mean(throughputs)."""
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("reference noise must be finite and nonnegative")
    return max(ENGINE_MARGIN_FLOOR, ENGINE_NOISE_MULTIPLIER * noise_std)


class ConfigVerificationError(Exception):
    """The published FLA config directory is missing, altered or incomplete: the session is VOID, never a miner strike."""


def host_config_root() -> str:
    """Host directory holding one sub-directory per GPU class (the repo's runtimes/s1-fast/fla-configs by default;
    a deployment may point REAX_ENGINE_FLA_CONFIG_ROOT at another checkout)."""
    env = os.environ.get("REAX_ENGINE_FLA_CONFIG_ROOT", "")
    return env or str(Path(__file__).resolve().parents[2] / "runtimes" / "s1-fast" / "fla-configs")


_SUMS_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


def verify_published_configs(gpu_class: str, host_dir: str) -> None:
    """Check ``host_dir`` is byte-for-byte the published config set of ``gpu_class``.

    SHA256SUMS must hash to the pinned value, every listed file must match, and nothing else except README.md may be
    present (fla loads ``<kernel>.json`` by name, so a stray file would change what the engines run). Files are
    regular files, no symlinks. Raises ConfigVerificationError.
    """
    pinned = PINNED_CONFIG_MANIFEST_SHA256.get(gpu_class)
    if pinned is None:
        raise ConfigVerificationError(f"no pinned FLA config checksum for GPU class {gpu_class!r}")
    root = Path(host_dir)
    try:
        if root.is_symlink() or not root.is_dir():
            raise ConfigVerificationError(f"FLA config directory {host_dir} is missing")
        sums = root / "SHA256SUMS"
        if sums.is_symlink() or not sums.is_file():
            raise ConfigVerificationError("FLA config SHA256SUMS is missing")
        raw = sums.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pinned:
            raise ConfigVerificationError("FLA config SHA256SUMS does not match the pinned checksum")
        listed: dict[str, str] = {}
        for line in raw.decode("utf-8").splitlines():
            m = _SUMS_LINE.match(line)
            if not m or m.group(2) in listed or m.group(2) in ("SHA256SUMS", "README.md"):
                raise ConfigVerificationError("FLA config SHA256SUMS is malformed")
            listed[m.group(2)] = m.group(1)
        if not listed:
            raise ConfigVerificationError("FLA config SHA256SUMS lists no files")
        present = {e.name for e in root.iterdir()} - {"SHA256SUMS", "README.md"}
        if present != set(listed):
            raise ConfigVerificationError("FLA config directory content differs from SHA256SUMS: "
                                          f"{sorted(present ^ set(listed))}")
        for name, digest in listed.items():
            f = root / name
            if f.is_symlink() or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
                raise ConfigVerificationError(f"FLA config file {name} does not match SHA256SUMS")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigVerificationError(f"FLA config directory unreadable: {exc.__class__.__name__}") from None
