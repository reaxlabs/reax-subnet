"""Engine runner hardening: release runtime (C1), tmpfs flags (C4), NVIDIA capabilities (C5), paused-candidate GPU check
under runsc (C9), cycle gate (C10) and the pre-run submission check with the REVIEWED gate."""

import hashlib
import json
import os

import pytest

from reax_subnet import engreview
from reax_subnet import engrunner as ER
from reax_subnet import engvalidator as EV
from reax_subnet.engrunner import AwaitingReview, DockerRunner, EngineError, ReferenceSelfCheckFailed
from test_engfix import _snapshot
from test_engine_hardening import FakeDocker, P, manifest

D1, D2 = (hashlib.sha256(x.encode()).hexdigest() for x in "ab")
REF = f"ghcr.io/a/e@sha256:{D1}"
CAND = f"ghcr.io/a/c@sha256:{D2}"


@pytest.fixture(autouse=True)
def published_test_config(monkeypatch):
    monkeypatch.setattr(ER, "PUBLISHED_GPU_CLASSES", frozenset(ER.GPU_CONFIG_DIRS))


RUNSC = {"path": "/usr/local/bin/runsc", "runtimeArgs": list(ER.RELEASE_RUNTIME_ARGS)}


class Docker(FakeDocker):
    def __init__(self, info=None, files=None, runtimes=None, ref_files=None, **kw):
        super().__init__(manifest([1]), present=True, **kw)
        self.info = info if info is not None else good_image()
        self.files, self.ref_files, self.owner = files or {}, ref_files or {}, {}
        self.runtimes = {"runc": {"path": "runc"}, "runsc": RUNSC} if runtimes is None else runtimes

    async def __call__(self, argv, timeout_s, **kw):
        a = list(argv)
        if a[:3] == ["docker", "image", "inspect"] and "{{json .}}" in a:
            self.calls.append(a)
            return 0, json.dumps(self.info), ""
        if a[:2] == ["docker", "info"]:
            self.calls.append(a)
            return (0, json.dumps(self.runtimes), "") if self.runtimes is not False else (1, "", "down")
        if a[:2] == ["docker", "create"]:
            self.calls.append(a)
            self.owner[a[a.index("--name") + 1]] = a[-1]
            return 0, "id", ""
        if a[:2] == ["docker", "cp"]:
            self.calls.append(a)
            name, path = a[2].split(":", 1)
            files = self.ref_files if self.owner.get(name) == REF else self.files
            if path in files:
                with open(a[3], "wb") as fh:
                    fh.write(files[path])
                return 0, "", ""
            return 1, "", "no such file"
        return await super().__call__(argv, timeout_s)


def good_image(**over):
    cfg = {"User": "10001", "Entrypoint": ["python", "-m", "engine"], "Cmd": [], "Env": ["PATH=/usr/bin"],
           "ExposedPorts": None, "Labels": {"ai.reax.engine.hotkey": "hk-a"}, "Volumes": None, "WorkingDir": "/app"}
    cfg.update(over)
    return {"Config": cfg, "Size": 5 * 10 ** 9, "Os": "linux", "Architecture": "amd64"}


def runner_for(tmp_path, **kw):
    kw.setdefault("review_dir", str(tmp_path / "review"))
    return DockerRunner(str(tmp_path / "w"), str(tmp_path / "weights"), **kw)


def cmd_of(docker):
    return docker.cmds("docker", "run")


# ------------------------------------------------------------------------------------------------ C1
async def test_release_mode_without_runsc_voids_the_session(tmp_path, monkeypatch):
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    for runtime in ("", "runc"):
        r = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime=runtime, release_mode=True)
        with pytest.raises(ReferenceSelfCheckFailed, match="runsc"):
            await r.preflight()
        with pytest.raises(ReferenceSelfCheckFailed):
            await r.start(CAND, "s1-fast", P)
    assert cmd_of(d) == []  # nothing was started


@pytest.mark.parametrize("runtimes", [{"runc": {}}, False, "not json"])
async def test_release_mode_needs_runsc_listed_by_docker_info(tmp_path, monkeypatch, runtimes):
    d = Docker(runtimes=runtimes)
    monkeypatch.setattr(ER, "_run", d)
    r = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runsc", release_mode=True)
    with pytest.raises(ReferenceSelfCheckFailed):
        await r.preflight()


async def test_release_mode_with_runsc_passes_and_runs_under_it(tmp_path, monkeypatch):
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    r = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runsc", release_mode=True,
                     protected_images=[REF])
    await r.preflight()
    await r.start(REF, "s1-fast", P, keep_image=True)
    argv = cmd_of(d)[0]
    assert argv[argv.index("--runtime") + 1] == "runsc"


async def test_release_mode_without_a_review_dir_never_runs_a_candidate(tmp_path, monkeypatch):
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    r = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runsc", release_mode=True)
    with pytest.raises(ReferenceSelfCheckFailed, match="review directory"):
        await r.start(CAND, "s1-fast", P)
    assert cmd_of(d) == []


async def test_development_mode_keeps_working_without_runsc(tmp_path, monkeypatch):
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    r = DockerRunner(str(tmp_path / "w"), str(tmp_path))
    await r.preflight()
    await r.start(REF, "s1-fast", P, keep_image=True)
    assert "--runtime" not in cmd_of(d)[0]


def test_config_and_worker_pass_release_mode_to_the_runner(tmp_path):
    from reax_subnet.config import load_validator
    from test_engcomp import _env

    cfg = load_validator(_env(REAX_ENGINE_CYCLE_GATE_FILE="~/gate", REAX_ENGINE_CYCLE_GATE_MAX_AGE_MIN="5",
                              REAX_ENGINE_WORKDIR=str(tmp_path / "wd"))).engine
    assert cfg.cycle_gate_file == os.path.expanduser("~/gate") and cfg.cycle_gate_max_age_s == 300
    assert cfg.review_dir == str(tmp_path / "wd" / "review")
    off = load_validator(_env()).engine
    assert off.cycle_gate_file == "" and off.review_dir.endswith("/review")


# ------------------------------------------------------------------------------------------------ C4 C5
async def test_candidate_and_reference_get_nosuid_nodev_tmpfs_and_limited_nvidia_capabilities(tmp_path, monkeypatch):
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    r = DockerRunner(str(tmp_path / "w"), str(tmp_path))
    await r.start(REF, "s1-fast", P, keep_image=True)   # reference
    await r.start(CAND, "s1-fast", P)                  # candidate
    runs = cmd_of(d)
    assert len(runs) == 2
    for argv in runs:
        opts = argv[argv.index("--tmpfs") + 1].split(":", 1)[1].split(",")
        assert {"rw", "exec", "nosuid", "nodev"} <= set(opts) and "noexec" not in opts
        env = [argv[i + 1] for i, x in enumerate(argv) if x == "-e"]
        assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in env


# ------------------------------------------------------------------------------------------------ C9
async def test_pid_check_is_off_under_runsc_even_when_requested(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ER, "_run", Docker())
    with caplog.at_level("WARNING", logger="reax.engines"):
        r = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runsc", nvml_check=True)
    assert "ignored under runsc" in caplog.text
    assert (await r.start(REF, "s1-fast", P, keep_image=True)).gpu_busy is None
    runc = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runc", nvml_check=True)
    assert (await runc.start(REF, "s1-fast", P, keep_image=True)).gpu_busy is not None


# ------------------------------------------------------------------------------------------------ C10
def competition(tmp_path, **kw):
    class Stub:
        binds_hotkey = False

    return EV.EngineCompetition(Stub(), {"s1-fast": REF}, str(tmp_path / "state.json"), start_utc=None, **kw)


def test_without_a_gate_file_nothing_changes(tmp_path):
    c = competition(tmp_path)
    assert c.gate_open() and c.due()


async def test_gate_must_exist_and_be_fresh(tmp_path):
    gate = tmp_path / "gate"
    c = competition(tmp_path, cycle_gate_file=str(gate), cycle_gate_max_age_s=600)
    assert not c.gate_open() and not c.due()
    assert await c.run_cycle(_snapshot([]), set()) == {} and c._last_cycle is None  # closed: no cycle at all
    gate.write_text("x")
    assert c.gate_open() and c.due()
    old = os.stat(gate).st_mtime - 3600
    os.utime(gate, (old, old))
    assert not c.gate_open() and not c.due()
    gate.unlink()
    gate.mkdir()  # a directory is not a gate
    assert not c.gate_open()


async def test_cycle_writes_cycle_end_and_it_survives_a_restart(tmp_path):
    gate = tmp_path / "gate"
    gate.write_text("x")
    c = competition(tmp_path, cycle_gate_file=str(gate))
    await c.run_cycle(_snapshot([]), set())
    assert [h["event"] for h in c.states["s1-fast"].cycle_events] == ["cycle_end"] and not c.states["s1-fast"].history
    assert json.load(open(tmp_path / "state.json"))["tracks"]["s1-fast"]["cycle_events"][-1]["event"] == "cycle_end"
    plain = competition(tmp_path / "plain")
    await plain.run_cycle(_snapshot([]), set())
    assert plain.states["s1-fast"].history == []   # default: no event, behaviour as before


async def test_gate_closing_stops_further_sessions(tmp_path):
    from test_engfix import _sub

    gate = tmp_path / "gate"
    gate.write_text("x")
    c = competition(tmp_path, cycle_gate_file=str(gate))
    seen = []

    async def evaluate_one(st, track, params, sub, block):
        seen.append(sub.uid)
        gate.unlink()
        return None

    c._evaluate_one = evaluate_one
    stopped = await c._run_sessions(c.states["s1-fast"], "s1-fast", c.tracks["s1-fast"],
                                    [_sub(1, "a"), _sub(2, "b")], _snapshot([]), [], None)
    assert stopped and seen == [1]


# ------------------------------------------------------------------------------------------------ submission check
def test_image_config_violations():
    ok = engreview.check_image_config(good_image())
    assert ok[0] == [] and ok[1] == []
    cases = {
        "root": good_image(User="root"), "uid0": good_image(User="0:0"), "rootgroup": good_image(User="1000:0"),
        "ports": good_image(ExposedPorts={"8080/tcp": {}}), "volume": good_image(Volumes={"/data": {}}),
        "proxy": good_image(Env=["HTTPS_PROXY=http://x"]), "preload": good_image(Env=["LD_PRELOAD=/x.so"]),
        "priv": good_image(Entrypoint=["sh", "-c", "mount /var/run/docker.sock"]),
        "label": good_image(Labels={"x": "run with --privileged"}), "empty": good_image(Entrypoint=[], Cmd=[]),
    }
    for name, info in cases.items():
        assert engreview.check_image_config(info)[0], name
    big = good_image()
    big["Size"] = 90 * 10 ** 9
    assert engreview.check_image_config(big)[0]
    win = good_image()
    win["Os"] = "windows"
    assert engreview.check_image_config(win)[0]
    assert engreview.check_image_config(good_image(User=""))[1]  # no User: a warning only


def test_file_comparison():
    v, w = engreview.compare_files({"/app/tokenizer.json": "aa", "/m/vocab.json": "bb", "/x/config.json": "cc",
                                    "/a/merges.txt": "!not a regular file"}, {"tokenizer.json": "aa", "vocab.json": "zz"})
    assert any("vocab.json" in x and "mismatch" in x for x in v) and any("merges.txt" in x for x in v)
    assert not any("tokenizer.json" in x for x in v) and any("config.json" in x for x in w)


def test_digest_helpers_reject_path_tricks(tmp_path):
    assert engreview.digest_of(CAND) == D2 and engreview.digest_of("x/y:latest") == ""
    for bad in ("../x", D2[:-1], D2.upper(), "sha256:" + D2):
        assert not engreview.valid_digest(bad) and not engreview.is_reviewed(str(tmp_path), bad)


async def test_a_clean_image_gets_a_record_and_no_gate_outside_release_mode(tmp_path, monkeypatch):
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    r = runner_for(tmp_path)
    await r.review(CAND)
    rec = engreview.load_record(r.review_dir, D2)
    assert rec["status"] == "static_ok" and rec["violations"] == [] and rec["image"]["user"] == "10001"
    n = len(d.calls)
    await r.review(CAND)                      # second time: the record is used, nothing is inspected again
    assert len(d.calls) == n
    await r.start(CAND, "s1-fast", P)          # and the start works without a REVIEWED marker


@pytest.mark.usefixtures("rules_v4_gates_off")
async def test_obvious_violation_is_rejected_with_a_reason_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(ER, "_run", Docker(info=good_image(User="root", ExposedPorts={"80/tcp": {}})))
    r = runner_for(tmp_path)
    with pytest.raises(EngineError, match="root.*ports|ports.*root"):
        await r.review(CAND)
    assert engreview.load_record(r.review_dir, D2)["status"] == "rejected"
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(EngineError):          # the stored verdict stands for this digest
        await r.start(CAND, "s1-fast", P)
    assert cmd_of(d) == []


async def test_tokenizer_mismatch_against_the_weights_and_the_reference(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "tokenizer.json").write_bytes(b"real tokenizer")
    d = Docker(files={"/app/tokenizer.json": b"tampered", "/app/config.json": b"engine cfg"})
    monkeypatch.setattr(ER, "_run", d)
    r = runner_for(tmp_path, reference_ref=REF)
    with pytest.raises(EngineError, match="tokenizer.json"):
        await r.review(CAND)
    rec = engreview.load_record(r.review_dir, D2)
    assert rec["files"]["/app/tokenizer.json"] == hashlib.sha256(b"tampered").hexdigest()
    assert d.cmds("docker", "create") and not cmd_of(d)         # created for the copy, never run
    assert any(c[-1].startswith("reax-chk-") for c in d.cmds("docker", "rm"))     # the check container is removed

    d2 = Docker(files={"/app/tokenizer.json": b"real tokenizer"})
    monkeypatch.setattr(ER, "_run", d2)
    r2 = runner_for(tmp_path / "second", reference_ref=REF)
    (tmp_path / "second").mkdir()
    r2.weights_dir = str(weights)
    await r2.review(CAND)                                       # identical copy: passes


async def test_symlinked_checked_file_is_a_violation(tmp_path):
    link = tmp_path / "l"
    os.symlink("/etc/passwd", link)
    assert engreview.hash_regular_file(str(link)).startswith("!")


async def test_release_mode_waits_for_the_reviewed_marker(tmp_path, monkeypatch):
    d = Docker()
    monkeypatch.setattr(ER, "_run", d)
    r = runner_for(tmp_path, docker_runtime="runsc", release_mode=True)
    with pytest.raises(AwaitingReview):
        await r.start(CAND, "s1-fast", P)
    assert cmd_of(d) == [] and r.awaiting_review(CAND)         # recorded, not run, left out of the next cycle
    os.makedirs(os.path.join(r.review_dir, "REVIEWED"))
    marker = os.path.join(r.review_dir, "REVIEWED", D2)
    os.symlink("/etc/hostname", marker)
    assert r.awaiting_review(CAND)                              # a symlink is not a marker
    os.unlink(marker)
    open(marker, "w").close()
    assert not r.awaiting_review(CAND)
    await r.start(CAND, "s1-fast", P)
    assert len(cmd_of(d)) == 1
    # the release reference is ours: no review needed
    r.protected_images = frozenset([REF])
    await r.start(REF, "s1-fast", P, keep_image=True)


async def test_rejected_digest_is_not_waiting_for_review(tmp_path, monkeypatch):
    monkeypatch.setattr(ER, "_run", Docker(info=good_image(User="root")))
    r = runner_for(tmp_path, release_mode=True, docker_runtime="runsc")
    with pytest.raises(EngineError):
        await r.review(CAND)
    assert not r.awaiting_review(CAND)


async def test_awaiting_review_costs_the_miner_nothing(tmp_path):
    from test_engfix import _sub

    class Waiting:
        binds_hotkey = False

        async def review(self, ref):
            raise AwaitingReview("x")

    c = EV.EngineCompetition(Waiting(), {"s1-fast": REF}, str(tmp_path / "s.json"), start_utc=None)
    st = c.states["s1-fast"]
    out = await c._evaluate_one(st, "s1-fast", c.tracks["s1-fast"], _sub(1, "a"), 5)
    assert out is None and st.attempts == {} and st.results == {}

    class Bad(Waiting):
        async def review(self, ref):
            raise EngineError("submission check: image runs as root")

    c.runner = Bad()
    out = await c._evaluate_one(st, "s1-fast", c.tracks["s1-fast"], _sub(1, "a"), 5)
    assert out.ok is False and "root" in out.reason


async def test_tokenizer_compared_with_the_reference_image_when_the_weights_have_none(tmp_path, monkeypatch):
    d = Docker(files={"/app/tokenizer.json": b"tampered"}, ref_files={"/app/tokenizer.json": b"ref tokenizer"})
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(EngineError, match="reference copy"):
        await runner_for(tmp_path, reference_ref=REF).review(CAND)
    d2 = Docker(files={"/app/tokenizer.json": b"ref tokenizer"}, ref_files={"/app/tokenizer.json": b"ref tokenizer"})
    monkeypatch.setattr(ER, "_run", d2)
    await runner_for(tmp_path / "ok", reference_ref=REF).review(CAND)
