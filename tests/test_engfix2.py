"""Critic findings of the engine release delta (workers/critic-engine): fault handling in the submission check, the
measurement context (unreadable/swapped weights, chain id spelling), the reference pin, runsc verification, the static
check (basename scope, reference/weights disagreement, extraction cap) and the budget timeout before measurement."""

import asyncio
import hashlib
import json
import os

import pytest

from reax_subnet import config
from reax_subnet import engreview
from reax_subnet import engrunner as ER
from reax_subnet import engvalidator as EV
from reax_subnet import worker as W
from reax_subnet.engcomp import Champion, EvalResult, Submission, TrackState
from reax_subnet.engrunner import AwaitingReview, DockerRunner, ReferenceSelfCheckFailed
from test_engfix import DA, _res, _snapshot, _sub
from test_engharden import CAND, REF, RUNSC, Docker, cmd_of, good_image
from test_engine_hardening import P


@pytest.fixture(autouse=True)
def published_test_config(monkeypatch):
    monkeypatch.setattr(ER, "PUBLISHED_GPU_CLASSES", frozenset(ER.GPU_CONFIG_DIRS))


# ------------------------------------------------------------------------------------------------ 1, 2 (probe p2)
class ReviewRunner:
    binds_hotkey = False

    def __init__(self, exc):
        self.exc, self.discarded = exc, []

    async def review(self, ref):
        raise self.exc

    async def discard(self, ref):
        self.discarded.append(ref)


def _comp(tmp_path, runner):
    return EV.EngineCompetition(runner, {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": P}, start_utc=None)


async def test_image_awaiting_review_is_discarded_and_counts_nothing(tmp_path):
    r = ReviewRunner(AwaitingReview("x"))
    c = _comp(tmp_path, r)
    out = await c._evaluate_one(c.states["s1-fast"], "s1-fast", P, _sub(1, "hk"), 5)
    assert out is None and r.discarded == [_sub(1, "hk").ref] and not c.states["s1-fast"].attempts


async def test_host_fault_in_the_submission_check_is_never_a_miner_failure(tmp_path):
    c = _comp(tmp_path, ReviewRunner(PermissionError("review dir")))
    st = c.states["s1-fast"]
    for i in range(P.max_infra_attempts + 2):
        assert await c._evaluate_one(st, "s1-fast", P, _sub(1, "hk"), 5 + i) is None
    assert not st.attempts and not st.results


# ------------------------------------------------------------------------------------------------ 3 (probe p1)
def _weights(tmp_path, content=b"x" * 100, config_json=b'{"a": 1}'):
    w = tmp_path / "weights"
    w.mkdir(exist_ok=True)
    (w / "model.safetensors").write_bytes(content)
    (w / "config.json").write_bytes(config_json)
    return w


def _champion_file(tmp_path, w):
    r = DockerRunner(str(tmp_path / "wd"), str(w))
    c = EV.EngineCompetition(r, {"s1-fast": REF}, str(tmp_path / "state.json"), start_utc=None, chain_id="finney:92")
    c.states["s1-fast"].champion = Champion(DA, "hk", 1, 1.5, 10)
    c.states["s1-fast"].results[DA] = _res(DA, 1.5)
    c.save()
    return c


def test_missing_or_empty_weights_dir_does_not_wipe_the_champion(tmp_path):
    w = _weights(tmp_path)
    _champion_file(tmp_path, w)
    kept = EV.EngineCompetition(DockerRunner(str(tmp_path / "wd"), str(w)), {"s1-fast": REF},
                                str(tmp_path / "state.json"), start_utc=None, chain_id="finney:92")
    assert kept.states["s1-fast"].champion and not kept.context_reset and kept.context_ok
    for state in ("missing", "empty"):
        gone = tmp_path / "weights"
        if state == "missing":
            os.rename(gone, tmp_path / "off")
        else:
            os.makedirs(tmp_path / "weights", exist_ok=True)
            for f in list(gone.iterdir()):
                f.unlink()
        r = DockerRunner(str(tmp_path / "wd"), str(gone))
        assert r.weights_identity() == f"unreadable:{state}"
        c = EV.EngineCompetition(r, {"s1-fast": REF}, str(tmp_path / "state.json"), start_utc=None, chain_id="finney:92")
        assert c.states["s1-fast"].champion is not None and not c.context_reset and not c.context_ok
        assert not c.due()
        c.save()  # keeps the context of the loaded file: the next start with readable weights still matches
        if state == "missing":
            os.rename(tmp_path / "off", gone)
    again = EV.EngineCompetition(DockerRunner(str(tmp_path / "wd"), str(_weights(tmp_path))), {"s1-fast": REF},
                                 str(tmp_path / "state.json"), start_utc=None, chain_id="finney:92")
    assert again.states["s1-fast"].champion is not None and not again.context_reset


# ------------------------------------------------------------------------------------------------ 4
def test_weights_identity_sees_content_of_small_files_and_safetensors_headers(tmp_path):
    def ident(**kw):
        return DockerRunner(str(tmp_path / "wd"), str(_weights(tmp_path, **kw))).weights_identity()

    header = (20).to_bytes(8, "little") + b'{"t":{"dtype":"F16"}}  '
    base = ident(content=header + b"d" * 500)
    assert base == ident(content=header + b"d" * 500)                                 # stable
    assert ident(content=header + b"d" * 500, config_json=b'{"a": 2}') != base        # same size, other config content
    other_header = (20).to_bytes(8, "little") + b'{"t":{"dtype":"BF16"}} '
    assert ident(content=other_header + b"d" * 500) != base                           # same size, other tensor header
    assert ident(content=header + b"e" * 500) != base                                 # same header, other tensor data (sample)


def test_weights_identity_reads_a_truncated_safetensors_file(tmp_path):
    for body in (b"", b"abc", (10 ** 12).to_bytes(8, "little")):
        r = DockerRunner(str(tmp_path / "wd"), str(_weights(tmp_path, content=body)))
        assert not r.weights_identity().startswith("unreadable")


# ------------------------------------------------------------------------------------------------ 5
@pytest.mark.parametrize("env", [{"REAX_NETWORK": "archive"}, {"REAX_NETWORK": " finney "}, {"REAX_NETWORK": "Finney"},
                                 {"REAX_NETWORK": "wss://entrypoint-finney.opentensor.ai:443/"},
                                 {"REAX_NETWORK": "wss://entrypoint-finney.opentensor.ai:443", "REAX_NETUID": "92"},
                                 {"REAX_NETUID": " 92 "}, {}])
def test_chain_id_is_the_same_for_every_spelling_of_finney(env):
    assert W._chain_id(env) == "finney:92"


def test_chain_id_of_development_networks():
    assert W._chain_id({"REAX_NETWORK": "test", "REAX_NETUID": "287"}) == "test:287"
    assert W._chain_id({"REAX_NETWORK": "local", "REAX_NETUID": "1"}) != W._chain_id({"REAX_NETWORK": "test", "REAX_NETUID": "1"})


# ------------------------------------------------------------------------------------------------ 6
PRE_STRICT_FLA_REFERENCE = "d857a90eaf168d1db3ca982c5111b10fb61a65cc5f7113ea9e391fe6fca17b71"


def test_no_gpu_class_is_published_while_the_reference_predates_strict_fla():
    """The reference image d857a90... has neither engfla.py nor /opt/reax/fla-configs: FLA_CACHE_MODE=strict does nothing
    on it. Publishing a GPU class needs the rebuilt reference (new digest in config.ENGINE_REFERENCE_IMAGE) first.
    That is 9bd41632 (fla 0.5.2 reads strict mode itself); it ships no configs, so the runner mounts the verified
    directory (tests/test_engfla_publish.py)."""
    from reax_subnet import engconfigs

    if engconfigs.PUBLISHED_GPU_CLASSES:
        assert PRE_STRICT_FLA_REFERENCE not in config.ENGINE_REFERENCE_IMAGE, (
            "PUBLISHED_GPU_CLASSES is set but the reference is still the pre-strict-FLA image d857a90: rebuild the "
            "reference image with the measured FLA configs and pin its new digest first")
    else:
        assert not engconfigs.PUBLISHED_GPU_CLASSES


def test_the_guard_fails_for_a_published_class_with_the_old_reference(monkeypatch):
    from reax_subnet import engconfigs

    monkeypatch.setattr(engconfigs, "PUBLISHED_GPU_CLASSES", frozenset(engconfigs.GPU_CONFIG_DIRS))
    monkeypatch.setattr(config, "ENGINE_REFERENCE_IMAGE",
                        f"ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:{PRE_STRICT_FLA_REFERENCE}")
    with pytest.raises(AssertionError, match="pre-strict-FLA|rebuild"):
        test_no_gpu_class_is_published_while_the_reference_predates_strict_fla()


# ------------------------------------------------------------------------------------------------ 8
@pytest.mark.parametrize("entry,match", [
    ({"path": "/usr/bin/runc", "runtimeArgs": list(ER.RELEASE_RUNTIME_ARGS)}, "binary"),
    ({"runtimeArgs": list(ER.RELEASE_RUNTIME_ARGS)}, "binary"),
    ({"path": "/usr/local/bin/runsc"}, "runtimeArgs"),
    ({"path": "/usr/local/bin/runsc", "runtimeArgs": ["--network=none"]}, "--nvproxy=true"),
    ({"path": "/usr/local/bin/runsc", "runtimeArgs": ["--network=none", "--nvproxy"]}, "allowed-driver-capabilities"),
    ({"path": "/usr/local/bin/runsc", "runtimeArgs": [*ER.RELEASE_RUNTIME_ARGS, "--network=host"]}, "network=host"),
    ({"path": "/usr/local/bin/runsc", "runtimeArgs": ["--nvproxy-allowed-driver-capabilities=all", "--nvproxy", "--network=none"]}, "allowed-driver-capabilities"),
    ({"path": "/usr/local/bin/runsc", "runtimeArgs": ["--nvproxy=true"]}, "--network=none"),
    ({"path": "/usr/local/bin/runsc", "runtimeArgs": "--network=none --nvproxy=true"}, "runtimeArgs"),
    ({}, "binary"), ("runsc", "binary"),
])
async def test_runsc_must_be_the_runsc_binary_with_its_required_flags(tmp_path, monkeypatch, entry, match):
    monkeypatch.setattr(ER, "_run", Docker(runtimes={"runc": {}, "runsc": entry}))
    r = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runsc", release_mode=True)
    with pytest.raises(ReferenceSelfCheckFailed, match=match):
        await r.preflight()


async def test_a_correct_runsc_entry_passes(tmp_path, monkeypatch):
    for args in (list(ER.RELEASE_RUNTIME_ARGS), ["--host-uds=create", "--debug", "--nvproxy=true", "--network=host", "--nvproxy-allowed-driver-capabilities=compute,utility", "--network=none"], ["--host-uds=create", "--nvproxy", "--nvproxy-allowed-driver-capabilities=compute,utility", "--network=none"]):
        monkeypatch.setattr(ER, "_run", Docker(runtimes={"runsc": {"path": "/usr/bin/runsc", "runtimeArgs": args}}))
        await DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runsc", release_mode=True).require_runtime()


@pytest.mark.parametrize("uds_args", [
    [],
    ["--host-uds=open"],
    ["--host-uds=all"],
    ["--host-uds=create", "--host-uds=open"],
    ["--host-uds=create", "--host-uds=all"],
    ["--host-uds=all", "--host-uds=create"],
    ["--host-uds=create", "--host-uds="],
    ["--host-uds=create", "--fsgofer-host-uds"],
    ["--host-uds=create", "--fsgofer-host-uds=true"],
    ["--host-uds=create", "--fsgofer-host-uds=false"],
    ["--host-uds=create", "-host-uds=all"],
    ["--host-uds=create", "--host-uds", "all"],
    ["--host-uds=create", "-fsgofer-host-uds"],
])
async def test_runsc_requires_create_only_host_uds(tmp_path, monkeypatch, uds_args):
    # Keep the other required flags valid so each refusal exercises the UDS policy alone.
    args = [arg for arg in ER.RELEASE_RUNTIME_ARGS if not arg.startswith("--host-uds=")]
    args.extend(uds_args)
    monkeypatch.setattr(ER, "_run", Docker(runtimes={"runsc": {"path": "/usr/bin/runsc", "runtimeArgs": args}}))
    runner = DockerRunner(str(tmp_path / "w"), str(tmp_path), docker_runtime="runsc", release_mode=True)
    with pytest.raises(ReferenceSelfCheckFailed, match="host-uds"):
        await runner.require_runtime()


# ------------------------------------------------------------------------------------------------ 9
TOK = hashlib.sha256(b"model tokenizer").hexdigest()
OWN = hashlib.sha256(b"engine settings").hexdigest()


def test_bare_config_names_are_compared_only_in_model_directories():
    expected = {"config.json": TOK, "tokenizer.json": TOK}
    v, w = engreview.compare_files({"/app/config.json": OWN, "/srv/config.json": OWN}, expected)
    assert v == [] and len(w) == 2                              # the engine's own config.json: no reference copy, a warning
    v, _ = engreview.compare_files({"/model/config.json": OWN, "/models/config.json": OWN, "/opt/model/config.json": OWN},
                                   expected)
    assert len(v) == 3                                           # the model's config.json must be the pinned one
    v, _ = engreview.compare_files({"/app/config.json": OWN}, {"/app/config.json": TOK})
    assert len(v) == 1                                           # the same full path is still compared
    v, _ = engreview.compare_files({"/app/tokenizer.json": OWN}, expected)
    assert len(v) == 1                                           # a tokenizer is compared wherever it is


async def _static(tmp_path, monkeypatch, files, ref_files, weights=None, info=None):
    d = Docker(files=files, ref_files=ref_files, info=info)
    monkeypatch.setattr(ER, "_run", d)
    w = tmp_path / "weights"
    w.mkdir(exist_ok=True)
    for n, b in (weights or {}).items():
        (w / n).write_bytes(b)
    r = DockerRunner(str(tmp_path / "w"), str(w), release_mode=False, review_dir=str(tmp_path / "review"),
                     reference_ref=REF, docker_runtime="runsc")
    return d, r


async def test_engine_with_its_own_config_json_is_not_rejected(tmp_path, monkeypatch):
    _, r = await _static(tmp_path, monkeypatch, {"/app/config.json": b"server settings"}, {}, {"config.json": b"model"})
    await r.review(CAND)
    assert engreview.load_record(r.review_dir, engreview.digest_of(CAND))["status"] == "static_ok"


async def test_reference_image_and_weights_directory_that_disagree_void_the_session(tmp_path, monkeypatch):
    _, r = await _static(tmp_path, monkeypatch, {"/model/config.json": b"model"}, {"/model/config.json": b"other model"},
                         {"config.json": b"model"})
    with pytest.raises(ReferenceSelfCheckFailed, match="do not belong together"):
        await r.review(CAND)
    assert engreview.load_record(r.review_dir, engreview.digest_of(CAND)) is None   # nothing cached against the miner


async def test_matching_reference_and_weights_still_reject_a_swapped_candidate_tokenizer(tmp_path, monkeypatch):
    _, r = await _static(tmp_path, monkeypatch, {"/model/config.json": b"evil"}, {"/model/config.json": b"model"},
                         {"config.json": b"model"})
    with pytest.raises(ER.EngineError, match="mismatch"):
        await r.review(CAND)


# ------------------------------------------------------------------------------------------------ 10
async def test_no_files_are_extracted_from_an_image_that_is_already_rejected(tmp_path, monkeypatch):
    d, r = await _static(tmp_path, monkeypatch, {"/model/config.json": b"x"}, {}, info=good_image(User="root"))
    with pytest.raises(ER.EngineError, match="root"):
        await r.review(CAND)
    assert d.cmds("docker", "create") == [] and d.cmds("docker", "cp") == []


async def test_a_huge_file_is_cut_off_while_it_is_copied(tmp_path):
    out = tmp_path / "dest"
    script = f"head -c 3000000 /dev/zero > {out}; sleep 30"
    code, _, err = await asyncio.wait_for(ER._run(["bash", "-c", script], 30, watch=(str(out), 1000000)), 20)
    assert code == ER.CP_TOO_LARGE and "larger" in err


async def test_a_small_copy_is_not_interrupted(tmp_path):
    out = tmp_path / "dest"
    code, _, _ = await ER._run(["bash", "-c", f"head -c 1000 /dev/zero > {out}"], 30, watch=(str(out), 1000000))
    assert code == 0


async def test_an_oversized_file_in_an_image_is_recorded_as_such(tmp_path, monkeypatch):
    d, r = await _static(tmp_path, monkeypatch, {}, {})
    real = ER._run

    async def run(argv, timeout_s, **kw):
        if argv[:2] == ["docker", "cp"] and argv[2].endswith(":/model/config.json"):
            return ER.CP_TOO_LARGE, "", "big"
        return await real(argv, timeout_s, **kw)

    monkeypatch.setattr(ER, "_run", lambda a, t, **kw: run(a, t, **kw) if a[:2] == ["docker", "cp"] else d(a, t))
    files = await r.image_files(CAND, "")
    assert files == {"/model/config.json": "!file larger than the check limit"}


# ------------------------------------------------------------------------------------------------ 11
async def test_cycle_events_do_not_push_title_changes_out_of_the_history(tmp_path):
    gate = tmp_path / "gate"
    gate.write_text("x")

    class Stub:
        binds_hotkey = False

    c = EV.EngineCompetition(Stub(), {"s1-fast": REF}, str(tmp_path / "state.json"), start_utc=None,
                             cycle_gate_file=str(gate))
    st = c.states["s1-fast"]
    st.history.append({"block": 1, "event": "crowned", "digest": DA})
    for _ in range(150):
        await c.run_cycle(_snapshot([]), set())
    assert st.history == [{"block": 1, "event": "crowned", "digest": DA}]
    assert len(st.cycle_events) == 100 and all(e["event"] == "cycle_end" for e in st.cycle_events)
    back = TrackState.from_json(json.loads(json.dumps(st.to_json())))
    assert back.cycle_events == st.cycle_events and back.history == st.history


# ------------------------------------------------------------------------------------------------ 14
class SlowEvaluate:
    """evaluate() that never finishes; ``measuring`` says whether it got as far as the first timed request."""

    def __init__(self, measuring):
        self.measuring = measuring

    async def __call__(self, runner, reference, ref, track, params, block=0, digest="", progress=None):
        if self.measuring:
            progress["measuring"] = True
        await asyncio.sleep(3600)


class Plain:
    binds_hotkey = False


async def test_budget_expiry_before_any_timed_request_is_an_infra_attempt(tmp_path, monkeypatch):
    import dataclasses

    p = dataclasses.replace(P, eval_budget_s=0.05)
    monkeypatch.setattr(EV, "evaluate", SlowEvaluate(measuring=False))
    c = EV.EngineCompetition(Plain(), {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": p}, start_utc=None)
    st = c.states["s1-fast"]
    sub = _sub(1, "hk")
    for n in range(1, p.max_infra_attempts):
        assert await c._evaluate_one(st, "s1-fast", p, sub, 5) is None and st.attempts[sub.digest] == n
    last = await c._evaluate_one(st, "s1-fast", p, sub, 5)       # bounded: after max_infra_attempts it is final
    assert last is not None and not last.ok and "inconclusive" in last.reason


async def test_budget_expiry_after_measurement_began_is_still_a_failed_measurement(tmp_path, monkeypatch):
    import dataclasses

    p = dataclasses.replace(P, eval_budget_s=0.05)
    monkeypatch.setattr(EV, "evaluate", SlowEvaluate(measuring=True))
    c = EV.EngineCompetition(Plain(), {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": p}, start_utc=None)
    res = await c._evaluate_one(c.states["s1-fast"], "s1-fast", p, _sub(1, "hk"), 5)
    assert res is not None and not res.ok and "exceeded its" in res.reason and not c.states["s1-fast"].attempts


# ------------------------------------------------------------------------------------------------ critic-rc9p4 #1
async def test_unreadable_file_in_the_image_is_a_violation_not_a_host_fault(tmp_path, monkeypatch):
    """A checked file with mode 000 in the image (docker cp keeps the mode bits) must not raise OSError out of review():
    a7a5ef6 treats that as a host fault, so the digest would be re-checked every cycle, never counted and never removed."""
    from test_engharden import Docker, runner_for

    class Mode000(Docker):
        async def __call__(self, argv, timeout_s, **kw):
            a = list(argv)
            if a[:2] == ["docker", "cp"] and a[2].endswith(":/app/tokenizer.json"):
                self.calls.append(a)
                with open(a[3], "wb") as fh:
                    fh.write(b"{}")
                os.chmod(a[3], 0)
                return 0, "", ""
            return await super().__call__(argv, timeout_s, **kw)

    if os.getuid() == 0:
        pytest.skip("root reads mode-000 files")
    monkeypatch.setattr(ER, "_run", Mode000())
    r = runner_for(tmp_path)
    with pytest.raises(ER.EngineError, match="unreadable"):
        await r.review(CAND)


async def test_host_fault_in_the_submission_check_gives_the_disk_back(tmp_path):
    r = ReviewRunner(PermissionError("review dir"))
    c = _comp(tmp_path, r)
    assert await c._evaluate_one(c.states["s1-fast"], "s1-fast", P, _sub(1, "hk"), 5) is None
    assert r.discarded == [_sub(1, "hk").ref]
