"""Joint detection, sandbox command and real tiny hybrid-backbone CPU parity."""
import copy
import json
import string
import time
from pathlib import Path

import numpy as np
import pytest

from reax_subnet.spec import joint as J, prompt as P, rules as R, tensors as T
from reax_subnet.spec.fetch import inspect_dir
from reax_subnet.spec.joint_docker import build_joint_command
from reax_subnet.spec.vllm import ServeSpec, build_docker_command
from spec_helpers import write_model, write_safetensors, make_weights

OPEN = 2_000_000_000.0


def package(tmp_path):
    path = Path(write_model(str(tmp_path / "model"), make_weights(1), category="generalist", arch="Qwen3_5Model",
                           config_extra={"model_type": "qwen3_5", "text_config": {
                               "model_type": "qwen3_5_text", "hidden_size": 16, "vocab_size": 256}}))
    (path / "model-00000-of-00001.safetensors").rename(path / "model.safetensors")
    d = {"format_version": 2, "base_model": "Qwen/Qwen3.8-27B", "revision": "a" * 40,
         "attention_mode": "noncausal_full_attention", "pooling": "last", "temperature": 1.38,
         "max_pixels": 1638400, "codes": list(string.ascii_uppercase), "token_ids": list(range(26))}
    (path / "decision_config.json").write_text(json.dumps(d))
    (path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": dict(zip(d["codes"], d["token_ids"]))}, "added_tokens": [{"content": t, "id": i + 100} for i, t in enumerate(("<|im_start|>", "<|im_end|>"))]}))
    (path / "joint_config.json").write_text(json.dumps(J.JOINT_PROTOCOL))
    write_safetensors(str(path / "readout.safetensors"), {"weight": np.ones((26, 16), np.float32)})
    return path


def detect(path):
    return J.detect(str(path), category="generalist", now_ts=OPEN)


def test_positive_and_optional_joint_config(tmp_path, monkeypatch):
    path = package(tmp_path)
    assert detect(path).hidden_size == 16
    monkeypatch.setattr(R, "_wall_clock", lambda: OPEN)
    ins = inspect_dir(str(path), category="generalist", hotkey=None)
    assert ins.violations == []
    assert ins.hashes.params == sum(e.numel for e in T.read_model(str(path)).values())
    original_hash = ins.hashes.named
    write_safetensors(str(path / "readout.safetensors"), {"weight": np.zeros((26, 16), np.float32)})
    assert T.canonical_hashes(str(path)).named != original_hash
    (path / "joint_config.json").unlink()
    assert detect(path) is not None


@pytest.mark.parametrize("category,now", [("generalist", 0), ("demand_forecasting", OPEN), ("risk_assessment", OPEN)])
def test_track_and_opening_gate(tmp_path, category, now):
    path = package(tmp_path)
    assert J.detect(str(path), category=category, now_ts=now) is None
    assert R.check_config(json.loads((path / "config.json").read_text()), R.rules_at(now, category=category))


@pytest.mark.parametrize("missing", ["readout.safetensors", "decision_config.json", "config.json"])
def test_missing_files(tmp_path, missing):
    path = package(tmp_path)
    (path / missing).unlink()
    with pytest.raises((ValueError, OSError)):
        detect(path)


@pytest.mark.parametrize("shape,dtype,name", [((25, 16), "F32", "weight"), ((26, 17), "F32", "weight"),
    ((26, 16), "I32", "weight"), ((26, 16), "F32", "evil"), ((26, 16, 1), "F32", "weight")])
def test_head_negative(tmp_path, shape, dtype, name):
    path = package(tmp_path)
    write_safetensors(str(path / "readout.safetensors"), {name: np.ones(shape, np.float32)}, dtype)
    with pytest.raises(ValueError, match="shape/dtype"):
        detect(path)


@pytest.mark.parametrize("file,key,value", [
    ("config.json", "auto_map", {}), ("config.json", "model_type", "other"),
    ("decision_config.json", "pooling", "mean"), ("decision_config.json", "attention_mode", "causal"),
    ("decision_config.json", "temperature", float("nan")), ("decision_config.json", "temperature", True),
    ("decision_config.json", "temperature", 10 ** 400),
    ("config.json", "quantization_config", {"quant_method": "unknown"}),
    ("decision_config.json", "base_model", "/tmp/source.py"),
    ("decision_config.json", "format_version", True), ("decision_config.json", "revision", "main"),
    ("decision_config.json", "codes", ["A"] * 26), ("decision_config.json", "token_ids", [0] * 26),
    ("decision_config.json", "max_pixels", -1), ("decision_config.json", "source_path", "evil.py"),
    ("joint_config.json", "version", True), ("joint_config.json", "readout_offset", 1),
    ("joint_config.json", "template", "../../evil.jinja"),
])
def test_config_negative(tmp_path, file, key, value, monkeypatch):
    path = package(tmp_path)
    data = json.loads((path / file).read_text()); data[key] = value
    (path / file).write_text(json.dumps(data))
    with pytest.raises(ValueError):
        detect(path)
    monkeypatch.setattr(R, "_wall_clock", lambda: OPEN)
    assert inspect_dir(str(path), category="generalist", hotkey=None).violations


def test_no_head_backbone_rejected(tmp_path, monkeypatch):
    path = package(tmp_path)
    for name in ("readout.safetensors", "decision_config.json", "joint_config.json"):
        (path / name).unlink()
    monkeypatch.setattr(R, "_wall_clock", lambda: OPEN)
    assert any("requires a readout" in x for x in inspect_dir(str(path), category="generalist", hotkey=None).violations)


@pytest.mark.parametrize("runtime", ["", "runc", "runsc"])
def test_causal_no_detection_and_identical_command(tmp_path, runtime):
    path = write_model(str(tmp_path / "causal"), make_weights(2))
    assert detect(path) is None
    # Actual source function body must remain byte-identical to the base commit; test avoids golden noise.
    import subprocess
    before = subprocess.check_output(["git", "show", "98c96a2:src/reax_subnet/spec/vllm.py"], text=True)
    current = Path("src/reax_subnet/spec/vllm.py").read_text()
    assert current == before
    assert Path("src/reax_subnet/spec/scoring.py").read_bytes() == subprocess.check_output(
        ["git", "show", "98c96a2:src/reax_subnet/spec/scoring.py"])
    spec = ServeSpec("name", "/model", "/socket", .8, aux_dir="/aux", runtime=runtime, user="10003:10003")
    causal = build_docker_command(spec)
    joint = build_joint_command(spec)
    idx = causal.index(spec.image)
    assert joint[:idx] == causal[:idx]
    assert joint[idx:idx+2] == ["-e", "PYTHONPATH=/aux"]
    assert "--network" in joint and joint[joint.index("--network")+1] == "none"
    if runtime:
        assert joint[joint.index("--runtime") + 1] == runtime
    else:
        assert "--runtime" not in joint


def tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    torch.manual_seed(17)
    torch.set_num_threads(1)
    cfg = Qwen3_5Config(text_config={
        "vocab_size": 128, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"], "linear_num_key_heads": 2,
        "linear_num_value_heads": 2, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4, "max_position_embeddings": 8192,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                            "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 16, "num_position_embeddings": 16, "deepstack_visual_indexes": []})
    cfg._attn_implementation = "sdpa"
    return Qwen3_5Model(cfg).eval()


class TinyTokenizer:
    """Use the real fixed template with a deterministic tiny vocabulary, no checkpoint download."""
    def apply_chat_template(self, messages, **kwargs):
        return ("".join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in messages)
                + '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    def encode(self, text, **kwargs):
        return [ord(c) % 120 for c in text]


def test_tiny_real_hybrid_cpu_vs_independent_reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.masking_utils import create_recurrent_attention_mask
    model = tiny_model()
    ref = copy.deepcopy(model)
    weight = torch.randn(26, 16)
    tok = TinyTokenizer()
    rows = [{"state": "item " + s, "question": {"type": "choice", "instructions": "Pick", "criteria": {
        "one": "first", "two": "second", "three": "third"}}} for s in ("1", "2", "longer")]
    runner = J.JointRunner(model, tok, weight)
    start = time.perf_counter()
    actual = runner.decide(rows)
    timing = (time.perf_counter() - start) / len(rows)
    expected = []
    # Independent reference: call the text model with masks directly (no production hook/runner).
    for row in rows:
        rendered = P.render(row["state"], row["question"])
        text = tok.apply_chat_template([{"role":"system", "content":rendered.system},
                                       {"role":"user", "content":rendered.user}])
        ids = torch.tensor([tok.encode(text)])
        emb = ref.language_model.embed_tokens(ids)
        padding = torch.ones_like(ids)
        masks = {"full_attention": torch.ones((1, 1, 1, ids.shape[1]), dtype=torch.bool),
                 "linear_attention": create_recurrent_attention_mask(
                     config=ref.language_model.config, inputs_embeds=emb, attention_mask=padding)}
        with torch.inference_mode():
            hidden = ref.language_model(input_ids=ids, attention_mask=masks, use_cache=False).last_hidden_state[0, -1]
            z = (weight @ hidden).float()[:3]
            probs = torch.softmax(z / 2, dim=-1).tolist()
        expected.append(dict(zip(rendered.keys, probs)))
    for got, want in zip(actual, expected):
        assert list(got) == list(want)
        assert list(got.values()) == pytest.approx(list(want.values()), abs=2e-6)
    # Sensitivity: this fixture must actually distinguish noncausal from causal hybrid attention.
    ids = torch.tensor([tok.encode(text)])
    with torch.inference_mode():
        causal = ref(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        noncausal = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    assert not torch.allclose(causal[:, -1], noncausal[:, -1], atol=1e-6)
    assert runner.decide([{"state": "x", "question": {"type": "bad"}}]) == [None]
    print(f"tiny hybrid CPU: {timing:.6f} seconds/item ({len(rows)} items)")


def test_tiny_safetensors_loader_cpu_end_to_end(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from reax_subnet.spec.templates import template_path
    model = tiny_model()
    path = package(tmp_path)
    # Replace synthetic header fixtures with actual standard Transformers safetensors.
    for file in path.glob("*.safetensors"):
        file.unlink()
    model.save_pretrained(path, safe_serialization=True)
    weight = torch.randn(26, 16)
    save_file({"weight": weight}, str(path / "readout.safetensors"))
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(string.ascii_uppercase)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.save(str(path / "tokenizer.json"))
    data = json.loads((path / "decision_config.json").read_text())
    data["token_ids"] = list(range(1, 27))
    (path / "decision_config.json").write_text(json.dumps(data))
    # A hostile author template is inert; the loader supplies our fixed qwen template.
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ 1 / 0 }}", "auto_map": {"evil": "source.py"}}))
    loaded = J.JointRunner.load(str(path), device="cpu")
    assert loaded.tokenizer.chat_template == Path(template_path("qwen.jinja")).read_text()
    rows = [
        {"state": "x", "question": {"type": "noul", "criteria": {}}},
        {"state": "y", "question": {"type": "score", "criteria": ["bad", "ok", "good"]}},
    ]
    output = loaded.decide(rows)
    assert set(output[0]) == {"yes", "no"}
    assert set(output[1]) == {"0", "1", "2"}
    assert all(sum(p.values()) == pytest.approx(1) for p in output)
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k, v in model.state_dict().items())
    assert torch.equal(weight, loaded.weight)
    with pytest.raises(ValueError, match="KV cache"):
        loaded.model.language_model(input_ids=torch.tensor([[1, 2]]), use_cache=True)


async def test_evaluator_selects_only_generalist_after_opening(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from reax_subnet.spec import evaluator as E, config as C, state as ST
    path = package(tmp_path)
    evaluator = E.SpecEvaluator(C.SpecConfig(sealed_dir=str(tmp_path / "sealed"), work_dir=str(tmp_path / "work"),
                                             state_path=str(tmp_path / "state.json")))
    calls = []

    @asynccontextmanager
    async def causal(*_args):
        calls.append("causal")
        yield Backend()

    @asynccontextmanager
    async def joint(*_args):
        calls.append("joint")
        yield Backend()

    class Backend:
        async def decide(self, rows):
            return [{"yes": .7, "no": .3} for _ in rows]

    monkeypatch.setattr(evaluator, "_docker_backend", causal)
    monkeypatch.setattr(evaluator, "_joint_backend", joint)
    evaluator.backend_factory = evaluator._docker_backend
    # Construction helper from the existing test suite preserves all submission fields.
    from test_keepalive import _sub
    sub = _sub(evaluator.state, 1, ST.QUEUED, None, track="generalist")
    prepared = E._Prepared(sub, str(path), 1000)
    await evaluator._serve(prepared, [], OPEN)
    assert calls == ["joint"]
    calls.clear()
    await evaluator._serve(prepared, [], 0)
    assert calls == ["causal"]
    calls.clear()
    sub.track_id = "risk_assessment"
    await evaluator._serve(prepared, [], OPEN)
    assert calls == ["causal"]


def test_size_and_symlink_rejections(tmp_path, monkeypatch):
    path = package(tmp_path)
    monkeypatch.setattr(R, "_wall_clock", lambda: OPEN)
    assert any("parameters" in x for x in inspect_dir(str(path), category="generalist", hotkey=None,
        rules=R.TrackRules(architectures=R.OPEN_CLASS_RULES.architectures, max_params=1)).violations)
    assert any("weight" in x for x in inspect_dir(str(path), category="generalist", hotkey=None,
        rules=R.TrackRules(architectures=R.OPEN_CLASS_RULES.architectures, max_weight_bytes=1)).violations)
    file = path / "readout.safetensors"
    file.rename(path / "saved")
    file.symlink_to(path / "saved")
    with pytest.raises(ValueError, match="symlink"):
        detect(path)


@pytest.mark.parametrize("file", ["decision_config.json", "joint_config.json"])
def test_oversized_metadata_rejected(tmp_path, file):
    path = package(tmp_path)
    (path / file).write_text(" " * 65537)
    with pytest.raises(ValueError, match="larger"):
        detect(path)


@pytest.mark.parametrize("arch", [["Qwen3_5ForConditionalGeneration"], ["Qwen3_5Model", "other"], []])
def test_stray_joint_files_are_ignored(tmp_path, arch):
    path = package(tmp_path)
    cfg = json.loads((path / "config.json").read_text())
    cfg["architectures"] = arch
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "decision_config.json").write_text("invalid ignored data")
    assert detect(path) is None


@pytest.mark.parametrize("fault", ["adapter", "tokenizer_missing", "tokenizer_codes", "index_absolute", "index_parent", "index_nested", "index_windows"])
def test_data_preflight_rejects(tmp_path, fault):
    path = package(tmp_path)
    if fault == "adapter":
        (path / "adapter_config.json").write_text("{}")
    elif fault == "tokenizer_missing":
        (path / "tokenizer.json").unlink()
    elif fault == "tokenizer_codes":
        (path / "tokenizer.json").write_text('{"model":{"vocab":{"A":99}}}')
    else:
        value = {"index_absolute": "/model.safetensors", "index_parent": "..safetensors",
                 "index_nested": "sub/model.safetensors", "index_windows": "sub\\model.safetensors"}[fault]
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": value}}))
    with pytest.raises(ValueError, match="invalid joint readout"):
        detect(path)


def test_inspect_uses_one_clock(tmp_path, monkeypatch):
    path = package(tmp_path)
    monkeypatch.setattr(R, "_wall_clock", lambda: 0)
    (path / "tokenizer.json").unlink()
    ins = inspect_dir(str(path), category="generalist", hotkey=None, rules=R.OPEN_CLASS_RULES, now_ts=OPEN)
    assert any("joint readout" in v and "tokenizer" in v for v in ins.violations)


@pytest.mark.parametrize("reverse", [False, True])
def test_evaluator_joint_layout_comparable_and_near(tmp_path, reverse):
    from reax_subnet.spec import evaluator as E, config as C, state as ST
    from test_keepalive import _sub
    base_w = {"model.language_model.layers.0.weight": np.ones((32, 32), np.float32),
              "model.visual.weight": np.ones((32, 32), np.float32)}
    causal_w = {k: v + .2 for k, v in base_w.items()}
    joint_w = {k.removeprefix("model."): v for k, v in causal_w.items()}
    if reverse:
        base_w = {k.removeprefix("model."): v for k, v in base_w.items()}
        causal_w, joint_w = joint_w, causal_w
    base = write_model(str(tmp_path / "base"), base_w)
    champion = write_model(str(tmp_path / "champ"), causal_w)
    challenger = write_model(str(tmp_path / "challenger"), joint_w)
    ev = E.SpecEvaluator(C.SpecConfig(sealed_dir=str(tmp_path / "sealed"), work_dir=str(tmp_path / "work"),
                                     state_path=str(tmp_path / "state.json")), base_provider=lambda _: base)
    ts = ev.state.track("generalist")
    champ = _sub(ev.state, 1, ST.CHAMPION, None, track="generalist")
    ch = _sub(ev.state, 2, ST.QUEUED, None, track="generalist")
    champ.base_model = ch.base_model = "test/base"
    assert ev._near_check(ts, champ, champion) == ("", "")
    assert ev._near_check(ts, ch, challenger)[0] != ST.PENDING_VERIFY
    assert ch.delta_checked
    stats = ev._near_champ(ch, champ, challenger)
    assert stats is not None and ev._is_near(stats)
    assert stats.relative_l2 == 0


def test_champion_new_head_is_exact_twin(tmp_path):
    from reax_subnet.spec import evaluator as E, config as C, state as ST
    from test_keepalive import _sub
    path = package(tmp_path)
    first = T.canonical_hashes(str(path))
    write_safetensors(str(path / "readout.safetensors"), {"weight": np.zeros((26, 16), np.float32)})
    second = T.canonical_hashes(str(path))
    assert first.named != second.named and first.backbone == second.backbone
    ev = E.SpecEvaluator(C.SpecConfig(sealed_dir=str(tmp_path / "sealed"), work_dir=str(tmp_path / "work"),
                                     state_path=str(tmp_path / "state.json")))
    ts = ev.state.track("generalist")
    champ = _sub(ev.state, 1, ST.CHAMPION, None, track="generalist")
    ch = _sub(ev.state, 2, ST.QUEUED, None, track="generalist")
    champ.hash_named, champ.hash_unnamed, champ.hash_backbone = first.named, first.unnamed, first.backbone
    ch.hash_named, ch.hash_unnamed, ch.hash_backbone = second.named, second.unnamed, second.backbone
    assert ev._exact_check(ts, champ) == ""
    assert "identical tensors" in ev._exact_check(ts, ch)


@pytest.mark.parametrize("fault", ["missing", "unexpected", "mismatched"])
def test_strict_real_loader_rejects(tmp_path, fault):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    model = tiny_model()
    path = package(tmp_path)
    for file in path.glob("*.safetensors"):
        file.unlink()
    model.save_pretrained(path, safe_serialization=True)
    weights = model.state_dict()
    key = next(k for k in weights if "layers.1." in k)
    if fault == "missing":
        weights.pop(key)
    elif fault == "unexpected":
        weights["extra.ignored"] = torch.zeros(1)
    else:
        weights[key] = torch.zeros(1)
    save_file(weights, str(path / "model.safetensors"))
    save_file({"weight": torch.zeros(26, 16)}, str(path / "readout.safetensors"))
    Tokenizer(models.WordLevel({c: i for i, c in enumerate(string.ascii_uppercase)}, unk_token="A")).save(str(path / "tokenizer.json"))
    with pytest.raises((ValueError, RuntimeError)):
        J.JointRunner.load(str(path), device="cpu")


@pytest.mark.parametrize("fault", [RuntimeError("tokenizer error"), RuntimeError("CUDA out of memory.")])
def test_runner_group_and_tokenizer_failure_is_per_item(fault):
    torch = pytest.importorskip("torch")
    class Model:
        def __call__(self, **kw):
            raise fault
    runner = object.__new__(J.JointRunner)
    runner.model, runner.weight, runner.tokenizer = Model(), torch.zeros(26, 16), TinyTokenizer()
    runner.max_length, runner.batch_size, runner.batch_tokens = 8192, 4, 8192
    row = {"state": "x", "question": {"type": "noul", "criteria": {}}}
    assert runner.decide([row, row]) == [None, None]
    runner.tokenizer.encode = lambda *a, **kw: (_ for _ in ()).throw(fault)
    assert runner.decide([row]) == [None]


async def test_http_large_request_and_26_options():
    from aiohttp.test_utils import TestClient, TestServer
    from reax_subnet.spec.joint_server import create_app, MAX_REQUEST_BYTES
    class Runner:
        def decide(self, rows):
            return [{"yes": .5, "no": .5} for _ in rows]
    async with TestClient(TestServer(create_app(Runner()))) as client:
        row = {"state": "x" * (300 << 10), "question": {"type": "noul", "criteria": {}}}
        response = await client.post("/decide", json=[row] * 4)
        assert response.status == 200
        assert len(await response.json()) == 4
        assert MAX_REQUEST_BYTES == 64 << 20
        assert (await client.post("/decide", json=[row] * 9)).status == 400
        assert (await client.get("/health")).status == 200
    row = {"state": "x", "question": {"type": "choice", "criteria": {str(i): str(i) for i in range(26)}}}
    r = P.render(row["state"], row["question"])
    assert r.letters == tuple(string.ascii_uppercase) or list(r.letters) == list(string.ascii_uppercase)
    assert len(P.answer_probs(row, r, list(range(26)))) == 26


@pytest.mark.parametrize("marker,oom,expected", [("REAX-JOINT-ENV-FAIL: import", False, "host"),
    ("REAX-JOINT-MODEL-FAIL: bad shape", False, "model"), ("", True, "oom"),
    ("", False, "model"), ("CUDA out of memory.", False, "oom"),
    ("REAX-JOINT-MODEL-FAIL: CUDA initialization failed", False, "model")])
async def test_startup_fault_attribution(tmp_path, monkeypatch, marker, oom, expected):
    from reax_subnet.spec import joint_docker as D, vllm as V
    backend = D.JointDockerBackend(str(tmp_path), 1000, gpu_total_gb=96)
    backend._gpu_preflight_ok = True
    async def parent(self):
        raise V.ModelUnavailable("container exited")
    async def run(*args, **kw):
        if args[1] == "inspect":
            return 0, str(oom).lower()
        return 0, marker
    monkeypatch.setattr(V.VllmDockerBackend, "_wait_healthy", parent)
    monkeypatch.setattr(V, "_run", run)
    with pytest.raises(V.ModelUnavailable) as e:
        await backend._wait_healthy()
    assert isinstance(e.value, V.HostUnavailable) == (expected == "host")
    if expected == "oom":
        assert isinstance(e.value, V.ModelBudgetOOM)
    elif expected == "model":
        assert isinstance(e.value, D.JointModelUnavailable)


async def test_memory_refusal_before_docker(tmp_path):
    from reax_subnet.spec import joint_docker as D, vllm as V
    backend = D.JointDockerBackend(str(tmp_path), 46 * V.GIB, gpu_total_gb=96, memory="48g", budget_gb=90)
    with pytest.raises(V.HostUnavailable, match="load margin"):
        await backend.__aenter__()


@pytest.mark.parametrize("runtime", ["", "runc", "runsc"])
@pytest.mark.parametrize("failure", [False, True])
async def test_image_preflight_cached_and_cleaned(tmp_path, monkeypatch, failure, runtime):
    from reax_subnet.spec import joint_docker as D, vllm as V
    D._PREFLIGHT_OK.clear()
    backend = D.JointDockerBackend(str(tmp_path), 1000, gpu_total_gb=96)
    calls = []
    async def run(*args, **kw):
        calls.append(args)
        return (1, "missing library") if failure and args[1] == "run" else (0, "")
    monkeypatch.setattr(V, "_run", run)
    spec = V.ServeSpec("test", str(tmp_path), "/socket", .8, runtime=runtime, aux_dir="/aux")
    if failure:
        with pytest.raises(V.HostUnavailable):
            await backend._image_preflight(spec)
        assert not D._PREFLIGHT_OK
    else:
        await backend._image_preflight(spec)
        await backend._image_preflight(spec)
        assert len(calls) == 2
    assert calls[-1][:3] == ("docker", "rm", "-f")
    command = calls[0]
    assert "--preflight" in command and "-d" not in command
    assert command[command.index("--network") + 1] == "none"
    if runtime:
        assert command[command.index("--runtime") + 1] == runtime
    else:
        assert "--runtime" not in command
    D._PREFLIGHT_OK.clear()


@pytest.mark.parametrize("field,value", [("image", "other-image"), ("runtime", "runc"), ("gpu_device", "1")])
async def test_image_preflight_cache_separates_image_runtime_gpu(tmp_path, monkeypatch, field, value):
    from dataclasses import replace
    from reax_subnet.spec import joint_docker as D, vllm as V
    D._PREFLIGHT_OK.clear()
    backend = D.JointDockerBackend(str(tmp_path), 1000, gpu_total_gb=96)
    calls = []
    async def run(*args, **kw):
        calls.append(args)
        return 0, ""
    monkeypatch.setattr(V, "_run", run)
    spec = V.ServeSpec("test", str(tmp_path), "/socket", .8, aux_dir="/aux")
    try:
        await backend._image_preflight(spec)
        await backend._image_preflight(replace(spec, **{field: value}))
        await backend._image_preflight(spec)
        assert len(calls) == 4  # Two distinct probes, each followed by cleanup.
        assert len(D._PREFLIGHT_OK) == 2
    finally:
        D._PREFLIGHT_OK.clear()


@pytest.mark.parametrize("runtime", ["", "runc", "runsc"])
async def test_backend_cleans_startup_error(tmp_path, monkeypatch, runtime):
    from reax_subnet.spec import joint_docker as D, vllm as V
    backend = D.JointDockerBackend(str(tmp_path), 1000, gpu_total_gb=96, runtime=runtime, work_dir=str(tmp_path))
    async def gpu(*a): pass
    async def preflight(self, spec): pass
    async def health(self): raise V.HostUnavailable("test startup")
    calls = []
    async def run(*args, **kw):
        calls.append(args)
        return 0, ""
    monkeypatch.setattr(V, "check_gpu_free", gpu)
    monkeypatch.setattr(V, "_run", run)
    monkeypatch.setattr(D.JointDockerBackend, "_image_preflight", preflight)
    monkeypatch.setattr(D.JointDockerBackend, "_wait_healthy", health)
    with pytest.raises(V.HostUnavailable):
        await backend.__aenter__()
    assert backend._http is None and not backend.run_dir
    if runtime:
        assert calls[0][calls[0].index("--runtime") + 1] == runtime
    else:
        assert "--runtime" not in calls[0]
    assert calls[-1][:3] == ("docker", "rm", "-f")
    assert not list(tmp_path.glob("reax-spec-*"))


async def test_client_chunks_four_and_recovers_group(tmp_path, monkeypatch):
    from reax_subnet.spec import joint_docker as D, vllm as V
    backend = D.JointDockerBackend(str(tmp_path), 1000, gpu_total_gb=96)
    sizes = []
    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
    class Http:
        def post(self, url, json):
            sizes.append(len(json))
            response = Response()
            response.count = len(json)
            response.status = 500 if len(sizes) == 1 else 200
            return response
    async def read(response): return [{"yes": 1.}] * response.count
    monkeypatch.setattr(V, "_read_json", read)
    backend._http = Http()
    result = await backend.decide([{}] * 9)
    assert sizes == [4, 4, 1]
    assert result[:4] == [None] * 4 and result[4:] == [{"yes": 1.}] * 5


@pytest.mark.parametrize("fault,expected", [(ImportError("late import during load"), "REAX-JOINT-MODEL-FAIL"),
    (RuntimeError("CUDA initialization failed"), "REAX-JOINT-MODEL-FAIL"),
    (ValueError("invalid joint readout: missing/unexpected weights"), "REAX-JOINT-MODEL-FAIL"),
    (Exception("invalid pre_tokenizer/normalizer"), "REAX-JOINT-MODEL-FAIL"),
    (OSError("missing shard"), "REAX-JOINT-MODEL-FAIL"),
    (ValueError("truncated shard"), "REAX-JOINT-MODEL-FAIL"),
    (RuntimeError("CUDA out of memory."), "REAX-JOINT-MODEL-FAIL")])
def test_server_markers(monkeypatch, capsys, fault, expected):
    from reax_subnet.spec import joint_server as S
    pytest.importorskip("safetensors")
    monkeypatch.setattr(S, "environment_check", lambda: None)
    monkeypatch.setattr(J.JointRunner, "load", lambda *a, **kw: (_ for _ in ()).throw(fault))
    with pytest.raises(SystemExit):
        S.load_runner(.8)
    assert expected in capsys.readouterr().out


def test_server_import_marker(monkeypatch, capsys):
    from reax_subnet.spec import joint_server as S
    monkeypatch.setattr(S, "environment_check", lambda: (_ for _ in ()).throw(ImportError("missing Qwen")))
    with pytest.raises(SystemExit):
        S.load_runner(.8)
    assert S.ENV_FAIL in capsys.readouterr().out


@pytest.mark.parametrize("indexed,nested", [(False, False), (True, False), (True, True)])
def test_unreferenced_safetensors_rejected(tmp_path, indexed, nested):
    path = package(tmp_path)
    if indexed:
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    extra = path / "sub" / "pad.safetensors" if nested else path / "pad.safetensors"
    extra.parent.mkdir(exist_ok=True)
    write_safetensors(str(extra), {"pad": np.ones((1,), np.float32)})
    with pytest.raises(ValueError, match="unreferenced safetensors"):
        detect(path)
    inspected = inspect_dir(str(path), category="generalist", hotkey=None, require_manifest=False, now_ts=OPEN)
    assert any("unreferenced safetensors" in v for v in inspected.violations)


def test_indexed_safetensors_allowed(tmp_path):
    path = package(tmp_path)
    (path / "model.safetensors").rename(path / "shard.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "shard.safetensors"}}))
    assert detect(path) is not None


@pytest.mark.parametrize("operation", ["inspect", "logs", "socket"])
async def test_joint_docker_socket_faults_remain_host(tmp_path, monkeypatch, operation):
    from reax_subnet.spec import joint_docker as D, vllm as V
    backend = D.JointDockerBackend(str(tmp_path), 1000, gpu_total_gb=96)
    async def parent(self):
        if operation == "socket":
            raise V.HostUnavailable("socket permission denied")
        raise V.ModelUnavailable("container exited")
    async def run(*args, **kw):
        return (1, "docker unavailable") if args[1] == operation else (0, "false")
    monkeypatch.setattr(V.VllmDockerBackend, "_wait_healthy", parent)
    monkeypatch.setattr(V, "_run", run)
    with pytest.raises(V.HostUnavailable):
        await backend._wait_healthy()


@pytest.mark.parametrize("joint", [True, False])
def test_joint_infra_retry_bound_persists_and_rejects(joint):
    from dataclasses import asdict
    from reax_subnet.spec import evaluator as E, state as ST
    sub = ST.Submission("entry", 1, "hot", "cold", "generalist", "a/b", "a" * 40, 1, OPEN)
    ts = ST.TrackState(track_id="generalist")
    for n in range(1, 7):
        # Simulate restarting the evaluator between every scheduled round.
        sub = ST.Submission(**asdict(sub))
        E.SpecEvaluator._serve_round(ts, sub, "infra", OPEN + n, joint=joint)
        assert sub.serve_log[-1][1] == ("model" if joint and n > 3 else "infra")
    assert sub.attempts == (3 if joint else 0)
    assert (sub.status == ST.REJECTED) == joint


def test_joint_infra_bound_resets_after_success():
    from reax_subnet.spec import evaluator as E, state as ST
    sub = ST.Submission("entry", 1, "hot", "cold", "generalist", "a/b", "a" * 40, 1, OPEN)
    ts = ST.TrackState(track_id="generalist")
    for n, outcome in enumerate(["infra"] * 3 + ["ok"] + ["infra"] * 3):
        E.SpecEvaluator._serve_round(ts, sub, outcome, OPEN + n, joint=True)
    assert sub.attempts == 0 and sub.joint_infra_outcomes == 3


@pytest.mark.parametrize("fault", ["pre_tokenizer", "normalizer", "missing_shard", "truncated_shard"])
def test_real_candidate_load_errors_get_model_marker(tmp_path, monkeypatch, capsys, fault):
    from reax_subnet.spec import joint_server as S
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    import torch
    model = tiny_model()
    path = package(tmp_path)
    for file in path.glob("*.safetensors"):
        file.unlink()
    model.save_pretrained(path, safe_serialization=True)
    save_file({"weight": torch.zeros(26, 16)}, str(path / "readout.safetensors"))
    Tokenizer(models.WordLevel({c: i for i, c in enumerate(string.ascii_uppercase)}, unk_token="A")).save(str(path / "tokenizer.json"))
    if fault in ("pre_tokenizer", "normalizer"):
        data = json.loads((path / "tokenizer.json").read_text())
        data[fault] = {"type": "InvalidCandidateComponent"}
        (path / "tokenizer.json").write_text(json.dumps(data))
    else:
        shard = path / "model.safetensors"
        shard.rename(path / "shard.safetensors")
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            name: "shard.safetensors" for name in model.state_dict()}}))
        if fault == "missing_shard":
            (path / "shard.safetensors").unlink()
        else:
            (path / "shard.safetensors").write_bytes(b"truncated")
    assert detect(path) is not None  # Metadata admission does not prove that loading succeeds.
    real_load = J.JointRunner.load
    monkeypatch.setattr(S, "environment_check", lambda: None)
    monkeypatch.setattr(J.JointRunner, "load", lambda *_a, **_kw: real_load(str(path), device="cpu"))
    with pytest.raises(SystemExit) as exc:
        S.load_runner(.8)
    assert exc.value.code == 65
    assert S.MODEL_FAIL in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["model", "cuda_oom", "host"])
async def test_joint_attribution_reaches_evaluator(tmp_path, monkeypatch, fault):
    from contextlib import asynccontextmanager
    from reax_subnet.spec import evaluator as E, config as C, state as ST, joint_docker as D, vllm as V
    from test_keepalive import _sub
    path = package(tmp_path)
    ev = E.SpecEvaluator(C.SpecConfig(sealed_dir=str(tmp_path / "sealed"), work_dir=str(tmp_path / "work"),
                                    state_path=str(tmp_path / "state.json")))
    error = {"model": D.JointModelUnavailable("CUDA initialization failed during candidate load"),
             "cuda_oom": V.ModelBudgetOOM("CUDA out of memory."),
             "host": V.HostUnavailable("docker unavailable")}[fault]
    @asynccontextmanager
    async def failed(*_args):
        raise error
        yield
    monkeypatch.setattr(ev, "_joint_backend", failed)
    sub = _sub(ev.state, 1, ST.CHAMPION, None, track="generalist")
    prepared = E._Prepared(sub, str(path), 1000)
    assert await ev._serve(prepared, [], OPEN) is None
    assert prepared.joint
    assert prepared.infra_failure == (fault == "host")
    assert prepared.model_specific_failure == (fault != "host")
