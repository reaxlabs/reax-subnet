"""Published FLA configs of the engine track: checksum verification, GPU-class gate, strict env + read-only mount."""

import hashlib
import shutil
from pathlib import Path

import pytest

from reax_subnet import config, engconfigs
from reax_subnet import engrunner as ER
from reax_subnet import engvalidator as EV
from reax_subnet.engconfigs import ConfigVerificationError, required_engine_margin, verify_published_configs
from test_engine_hardening import FakeDocker, REF, P, _snap

pytestmark = pytest.mark.fla_checksum

SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
REPO_ROOT = Path(__file__).resolve().parents[1] / "runtimes" / "s1-fast" / "fla-configs"
SERVER_DIR = REPO_ROOT / engconfigs.GPU_CONFIG_DIRS[SERVER]
CONTAINER_DIR = "/opt/reax/fla-configs/rtx-pro-6000-blackwell-server"


class ServerDocker(FakeDocker):
    def __init__(self, gpu=SERVER, **kw):
        super().__init__(**kw)
        self.gpu = gpu

    async def __call__(self, argv, timeout_s, **kw):
        if argv[0] == "nvidia-smi" and "--query-gpu=name" in argv:
            self.calls.append(list(argv))
            return 0, self.gpu, ""
        return await super().__call__(argv, timeout_s, **kw)


@pytest.fixture
def config_root(tmp_path):
    root = tmp_path / "cfgroot"
    shutil.copytree(REPO_ROOT, root)
    return root


def test_only_the_measured_server_edition_is_published_with_the_measured_noise():
    assert engconfigs.PUBLISHED_GPU_CLASSES == frozenset({SERVER})
    assert engconfigs.PINNED_REFERENCE_NOISE_STD == 0.00167
    assert required_engine_margin(engconfigs.PINNED_REFERENCE_NOISE_STD) == 0.03
    assert "9bd41632e33bd90377f7e29bdb9410b18cfe1fb557200160b9ebe9222abcecea" in config.ENGINE_REFERENCE_IMAGE


def test_the_committed_configs_match_the_pinned_checksums_and_cover_the_six_kernels():
    verify_published_configs(SERVER, str(SERVER_DIR))
    names = {p.name for p in SERVER_DIR.glob("*.json")}
    assert len(names) == 7 and "_meta.json" in names and "l2norm_fwd_kernel.json" in names


def test_l2norm_is_pinned_to_the_majority_class_for_every_published_row_bucket():
    """No strict-mode miss for l2norm up to NB 256: a candidate engine without its own pin cannot autotune it (#22151)."""
    import json
    data = json.loads((SERVER_DIR / "l2norm_fwd_kernel.json").read_text())
    entries = list(data["autotune_entries"].values())
    by_nb = {e["autotune_key"][1]: e for e in entries}
    assert sorted(by_nb) == list(range(1, 257)) and len(entries) == 256
    minority = {(8, 8), (16, 16), (8, 16)}  # measured bit-classes other than the majority (diag/l2norm classmap)
    for nb, e in by_nb.items():
        assert e["autotune_key"] == [128, nb, "torch.bfloat16", "torch.bfloat16", "torch.float32"]
        c = e["config"]
        assert (c["kwargs"]["BT"], c["num_warps"]) not in minority and c["num_stages"] == 3
        if nb >= 3:
            assert (c["kwargs"]["BT"], c["num_warps"]) == (16, 4)  # = the reference engine's pin_fla_autotune()
    for h, e in data["autotune_entries"].items():
        key = json.dumps(e["autotune_key"], separators=(",", ":"), sort_keys=True)
        assert hashlib.md5(key.encode()).hexdigest() == h  # fla's AutotuneKey.key_hash


@pytest.mark.parametrize("mutate", ["edit", "extra", "missing", "sums", "symlink"])
def test_any_change_to_the_config_directory_is_rejected(config_root, mutate):
    d = config_root / engconfigs.GPU_CONFIG_DIRS[SERVER]
    if mutate == "edit":
        f = d / "l2norm_fwd_kernel.json"
        f.write_text(f.read_text().replace('"BT": 8', '"BT": 16'))
    elif mutate == "extra":
        (d / "causal_conv1d_fwd_kernel.json").write_text("{}")
    elif mutate == "missing":
        (d / "chunk_fwd_kernel_o.json").unlink()
    elif mutate == "sums":
        (d / "SHA256SUMS").write_text((d / "SHA256SUMS").read_text() + "\n")
    else:
        f = d / "l2norm_fwd_kernel.json"
        content = f.read_bytes()
        (d / "other.json").write_bytes(content)
        f.unlink()
        f.symlink_to("other.json")
    with pytest.raises(ConfigVerificationError):
        verify_published_configs(SERVER, str(d))


def test_unpinned_class_and_missing_directory_are_rejected(tmp_path):
    with pytest.raises(ConfigVerificationError, match="no pinned"):
        verify_published_configs("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", str(SERVER_DIR))
    with pytest.raises(ConfigVerificationError, match="missing"):
        verify_published_configs(SERVER, str(tmp_path / "nothing"))


async def test_checksum_mismatch_voids_the_session_before_any_container(config_root, tmp_path, monkeypatch):
    f = config_root / engconfigs.GPU_CONFIG_DIRS[SERVER] / "recompute_w_u_fwd_kernel.json"
    f.write_text(f.read_text() + " ")
    docker = ServerDocker(present=True)
    monkeypatch.setattr(ER, "_run", docker)
    runner = ER.DockerRunner(str(tmp_path), str(tmp_path), fla_config_root=str(config_root))
    with pytest.raises(ER.ReferenceSelfCheckFailed, match="does not match"):
        await runner.start(REF, P.pool, P)
    assert docker.cmds("docker", "run") == []
    comp = EV.EngineCompetition(runner, {P.name: REF}, str(tmp_path / "state.json"), tracks={P.name: P})
    assert await comp.run_cycle(_snap(f"reax-eng/1:{P.name}:{REF}"), {1}) == {}
    st = comp.states[P.name]
    assert st.cycle_events[-1]["event"] == "session_void"
    assert not st.results and not st.attempts  # VOID: no result, no attempt, no miner strike


async def test_unpublished_gpu_class_voids_even_with_valid_files(config_root, tmp_path, monkeypatch):
    docker = ServerDocker(gpu="NVIDIA RTX PRO 6000 Blackwell Workstation Edition", present=True)
    monkeypatch.setattr(ER, "_run", docker)
    runner = ER.DockerRunner(str(tmp_path), str(tmp_path), fla_config_root=str(config_root))
    with pytest.raises(ER.ReferenceSelfCheckFailed, match="no published FLA config"):
        await runner.start(REF, P.pool, P)
    assert docker.cmds("docker", "run") == []


async def test_every_engine_gets_strict_env_and_the_verified_read_only_mount(config_root, tmp_path, monkeypatch):
    docker = ServerDocker(present=True)
    monkeypatch.setattr(ER, "_run", docker)
    runner = ER.DockerRunner(str(tmp_path), str(tmp_path), fla_config_root=str(config_root))
    for keep in (True, False):  # reference, then candidate
        eng = await runner.start(REF, P.pool, P, keep_image=keep)
        await eng.stop()
        await eng.client.close()
    runs = docker.cmds("docker", "run")
    assert len(runs) == 2
    host = str(config_root / engconfigs.GPU_CONFIG_DIRS[SERVER])
    for argv in runs:
        assert "FLA_CACHE_MODE=strict" in argv and f"FLA_CONFIG_DIR={CONTAINER_DIR}" in argv
        assert f"{host}:{CONTAINER_DIR}:ro" in argv
        assert "--read-only" in argv
