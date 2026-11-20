"""RULES-v2 2.4: a text-only export of a start that also has a vision part passes the config check without a declaration.
Real configs (tests/golden/configs/PROVENANCE.txt): Gemma-4-31B-it and two real public text-only exports of it."""
import copy
import json
import os

import pytest

from reax_lineage.evidence import config_diff, text_only_export

G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "configs")


def cfg(name):
    return json.load(open(os.path.join(G, name)))


START = "gemma-4-31B-it.json"
EXPORTS = ["termgrade-gemma4-31b-merged.text-only.json", "Jeff-1.0-Large.text-only.json"]


def fields(problems):
    return sorted(p.field for p in problems)


@pytest.mark.parametrize("name", EXPORTS)
def test_real_gemma4_text_only_export_passes_without_declaration(name):
    e, s = cfg(name), cfg(START)
    assert e["architectures"] == ["Gemma4ForCausalLM"] and s["architectures"] == ["Gemma4ForConditionalGeneration"]
    assert "text_config" in s and "vision_config" in s and "text_config" not in e and "vision_config" not in e
    assert config_diff(e, s, {}) == []
    t = text_only_export(e, s)
    assert t and t["start_architecture"] == "Gemma4ForConditionalGeneration" and t["entry_architecture"] == "Gemma4ForCausalLM"
    assert t["model_type"] == "gemma4_text"


def test_before_the_change_the_only_flagged_field_was_architectures():
    """What the first checker reported for these two real exports: exactly one difference, the wrapper class name."""
    s = cfg(START)
    for name in EXPORTS:
        e = cfg(name)
        e["architectures"] = ["Gemma4ForConditionalGeneration"]          # undo the wrapper difference: nothing else differs
        assert config_diff(e, s, {}) == []


@pytest.mark.parametrize("name", EXPORTS)
@pytest.mark.parametrize("field,value", [("hidden_size", 4096), ("num_hidden_layers", 48), ("vocab_size", 200000), ("intermediate_size", 1000),
                                         ("tie_word_embeddings", False), ("sliding_window", 512)])
def test_a_real_change_in_the_text_part_is_still_flagged(name, field, value):
    e, s = cfg(name), cfg(START)
    e[field] = value
    assert field in fields(config_diff(e, s, {})), field
    assert "architectures" not in fields(config_diff(e, s, {}))               # the allowance covers the wrapper name only


@pytest.mark.parametrize("name", EXPORTS)
def test_declared_change_still_works_next_to_the_export(name):
    e, s = cfg(name), cfg(START)
    e["rope_theta"] = 123.0
    assert fields(config_diff(e, s, {})) == ["rope_theta"]
    assert config_diff(e, s, {"config_fields": {"rope_theta": 123.0}}) == []


def test_other_class_with_remote_code_is_not_the_documented_export():
    """mente-ai/uyu-2-28B: Uyu2ForCausalLM + auto_map + pruned shapes: a different architecture, not 'ForConditionalGeneration -> ForCausalLM'."""
    e, s = cfg("uyu-2-28B.other-class.json"), cfg(START)
    assert text_only_export(e, s) is None
    assert {"architectures", "auto_map"} <= set(fields(config_diff(e, s, {})))


def test_wrong_family_class_name_is_flagged():
    e, s = cfg(EXPORTS[0]), cfg(START)
    e["architectures"] = ["Gemma3ForCausalLM"]
    assert "architectures" in fields(config_diff(e, s, {}))
    e["architectures"] = ["Gemma4ForCausalLM", "Gemma4Model"]
    assert "architectures" in fields(config_diff(e, s, {}))


def test_export_with_the_wrong_text_model_type_is_flagged():
    e, s = cfg(EXPORTS[0]), cfg(START)
    e["model_type"] = "gemma3_text"
    out = fields(config_diff(e, s, {}))
    assert "architectures" in out and "model_type" in out


def test_entry_that_still_has_a_vision_part_is_not_a_text_only_export():
    """Round-1 style `Qwen3_5Model` wrappers keep text_config + vision_config: the class name still has to match or be declared."""
    e, s = cfg("sn92-gen-ma.wrapper-with-vision.json"), cfg("Qwen3.8-27B.json")
    assert text_only_export(e, s) is None
    assert "architectures" in fields(config_diff(e, s, {}))
    assert "architectures" not in fields(config_diff(e, s, {"config_fields": {"architectures": e["architectures"]}}))
    e2 = cfg(EXPORTS[0])
    e2["vision_config"] = {"model_type": "gemma4_vision"}                   # a CausalLM class with a vision part left in the config
    assert text_only_export(e2, cfg(START)) is None and "architectures" in fields(config_diff(e2, cfg(START), {}))


def test_start_without_a_vision_part_gets_no_allowance():
    s = cfg(START)
    for k in ("vision_config", "audio_config"):
        s.pop(k, None)
    e = cfg(EXPORTS[0])
    assert text_only_export(e, s) is None
    assert "architectures" in fields(config_diff(e, s, {}))


def test_multimodal_fine_tune_of_the_start_is_unchanged():
    """Jeffan-31b-1710 (the round-2 list start's own config) against Gemma-4-31B-it: nothing flagged, no allowance involved."""
    e, s = cfg("Jeffan-31b-1710.multimodal.json"), cfg(START)
    assert text_only_export(e, s) is None and config_diff(e, s, {}) == []


def test_qwen_pair_class_name_is_allowed_other_findings_stay():
    """Jeffan-27B-1211 is a real text-only export of a Qwen3.8-27B-like start: the class name passes; its remote code (auto_map) is still refused."""
    e, s = cfg("Jeffan-27B-1211.text-only.json"), cfg("Qwen3.8-27B.json")
    assert text_only_export(e, s) is not None
    assert fields(config_diff(e, s, {})) == ["auto_map"]
    e = copy.deepcopy(e)
    e.pop("auto_map")
    assert config_diff(e, s, {}) == []


# ------------------------------------------------------- whole check (synthetic world) ----

def _wrapper_start_config():
    """A multimodal wrapper around the synthetic text model, shaped like the real Gemma-4 config: top-level wrapper keys + text_config + vision_config."""
    from synth import CONFIG
    text = {k: v for k, v in CONFIG.items() if k != "architectures"}
    text["model_type"] = "tiny_text"
    return {"architectures": ["TinyForConditionalGeneration"], "model_type": "tiny", "text_config": text, "vision_config": {"model_type": "tiny_vision"},
            "image_token_id": 7, "tie_word_embeddings": False}


def _text_only_export_config(**changes):
    from synth import CONFIG
    c = dict(CONFIG, architectures=["TinyForCausalLM"], model_type="tiny_text")
    c.update(changes)
    return c


@pytest.fixture(scope="module")
def mmworld(tmp_path_factory):
    from synth import REV, World, finetune, make_base
    w = World(tmp_path_factory.mktemp("mm"))
    root = make_base(1)
    w.add("org/root", root, kind="official", rev=REV(1), public_at="2025-06-01T00:00:00Z")
    start = finetune(root, 31, 1.0)
    w.add("org/start", start, kind="list", start="org/root@" + REV(1), listed="both", rev=REV(2), public_at="2025-07-01T00:00:00Z", config=_wrapper_start_config())
    w.tensors = {"start": start}
    w.list_path = w.finish()
    return w


def test_text_only_export_passes_the_whole_check_without_a_declaration(mmworld):
    from synth import finetune
    from test_pipeline import codes, rec_for, run
    c = finetune(mmworld.tensors["start"], 51, 0.15)
    res = run(mmworld, "carol/textonly", c, rec_for(mmworld, "carol/textonly"), config=_text_only_export_config())
    assert res["result"] == "PASS", res["reasons"]
    assert res["config_text_only_export"]["entry_architecture"] == "TinyForCausalLM"
    assert any("text-only export" in n for n in res["notes"]) and codes(res, "N") == []
    # The context-setting difference is retained as information.
    res2 = run(mmworld, "carol/textonly2", c, rec_for(mmworld, "carol/textonly2"), config=_text_only_export_config(rope_theta=500000.0))
    assert res2["result"] == "PASS" and [r["field"] for r in res2["reasons"] if r["code"] == "config" and r["class"] == "I"] == ["rope_theta"]
    # and the old way (declaring the class change) still works
    rec = rec_for(mmworld, "carol/textonly3")
    rec["architecture_changes"]["config_fields"] = {"architectures": ["TinyForCausalLM"]}
    assert run(mmworld, "carol/textonly3", c, rec, config=_text_only_export_config())["result"] == "PASS"
