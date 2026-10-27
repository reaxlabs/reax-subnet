"""Engine competition v2.2 (25 Oct 2026): every decisionmodels.io model is a slot. The two new slots (quyet = Quyet-1.0-Large,
vision = s1-vision), the image field of the decision contract, the procedural image items, weights variant v22 as the
default, the release guard for the uncalibrated slots, and the rc11p6 engine rules (copy line, parent run, Tue/Fri slots)
on the composite track."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import math
import struct
import sys
import time
import zlib
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import competition_split as cs
from reax_subnet import engcomp as EC
from reax_subnet import engcomposite as ecmp
from reax_subnet import engreview
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import Champion, EvalResult, Submission, TrackParams, TrackState, observe, update_champion
from reax_subnet.engcomposite import (CompositeRunner, CompositeTrack, SlotTapes, composite_rules, composite_rules_sha256,
                                      composite_score, default_track, evaluate_composite, to_eval_result)
from reax_subnet.engrunner import DevRunner, fresh_decisions, probe_decisions
from reax_subnet.s1contract import (ENGINE_MAX_OPTIONS, IMAGE_POOLS, MAX_IMAGE_BYTES, MAX_OPTIONS, MODELS, ContractError,
                                    parse_decision, parse_image)
from reax_subnet.s1items import generate_s1_items, items_for_pool
from reax_subnet.s1vision import GENERATOR_RULE, data_url, generate_vision_items, vision_item

from test_engcomp import FAKE, FAST_P, ref_of
from test_engcomposite import ALL, D1, D2, TRACK, _r, _snapshot

V22 = {"d3": 0.45, "blink": 0.20, "vision": 0.05, "reax1": 0.20, "pplx": 0.05, "quyet": 0.05}


@pytest.fixture(autouse=True)
def _v22_variant(monkeypatch):
    """This file tests the v2.2 table; v2.0 (the lineup after the 25 Oct decisionmodels.io switch) is the default now
    and is tested in test_engcomposite_v20.py. v22 stays selectable."""
    monkeypatch.setattr(ecmp, "COMPOSITE_WEIGHTS_VARIANT", "v22")


@pytest.fixture(autouse=True)
def _copyline_not_started(monkeypatch):
    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", None)  # tests that need it set it themselves


# ------------------------------------------------------------------------------------------------ png helpers
def _png(width=16, height=16, *, crc_ok=True, idat=True, iend=True, trailing=b"") -> bytes:
    def chunk(kind, payload, ok=True):
        crc = zlib.crc32(kind + payload) if ok else 0
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    body = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0), crc_ok)
    if idat:
        body += chunk(b"IDAT", zlib.compress(b"\x00" * (1 + 3 * width) * min(height, 4)))
    if iend:
        body += chunk(b"IEND", b"")
    return body + trailing


def _url(raw: bytes, kind="png") -> str:
    return f"data:image/{kind};base64," + base64.b64encode(raw).decode()


def _jpeg(width=16, height=16) -> bytes:
    return (b"\xff\xd8\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00" + b"\xff\xd9")


def _decision(images=None, n_options=3):
    d = {"id": "d", "state": "s", "questions": {"q": {"type": "choice", "instructions": "?",
                                                      "criteria": {f"k{i}": None for i in range(n_options)}}}}
    if images is not None:
        d["images"] = images
    return d


# ------------------------------------------------------------------------------------------------ contract: images
def test_the_vision_pool_takes_exactly_one_valid_image():
    item = vision_item(1, 0)
    d = parse_decision(item.decision(), "s1-vision")
    assert len(d.images) == 1 and d.images[0] == item.images[0] and d.body()["images"] == [item.images[0]]
    info = parse_image(item.images[0])
    assert (info.mime, info.width, info.height) == ("image/png", 320, 240) and info.size < 50_000
    assert parse_decision(_decision(), "s1-vision").images == ()  # text only is allowed in the contract (as in production)
    assert "images" not in parse_decision(_decision(), "s1-vision").body()
    assert parse_image(_url(_jpeg(40, 30), "jpeg")).mime == "image/jpeg"


@pytest.mark.parametrize("pool", ["s1-fast", "s1-pro", "s1-lite", "s1-cpu", "quyet-1-large"])
def test_every_other_pool_rejects_images(pool):
    with pytest.raises(ContractError, match="text only"):
        parse_decision(_decision([_url(_png())]), pool)
    with pytest.raises(ContractError, match="text only"):
        parse_decision(_decision([]), pool)  # even an empty list: the field does not exist there


def test_exactly_one_image():
    ok = _url(_png())
    for bad in ([], [ok, ok], "x", {"url": ok}):
        with pytest.raises(ContractError, match="exactly one image"):
            parse_decision(_decision(bad), "s1-vision")
    with pytest.raises(ContractError, match="exactly the fields"):
        d = _decision([ok])
        d["extra"] = 1
        parse_decision(d, "s1-vision")


@pytest.mark.parametrize("url,why", [
    (123, "data URL"),
    ("https://example.com/a.png", "data:image"),
    ("data:image/webp;base64,AAAA", "data:image"),
    ("data:image/png;base64,@@@@", "data:image"),
    ("data:image/png;base64,AAA", "base64"),  # length not a multiple of 4
    ("data:image/png;base64,", "data:image"),
    (_url(b"this is not an image at all"), "not a PNG"),
    (_url(b"\x89PNG\r\n\x1a\n" + b"junk"), "truncated"),
    (_url(_png(crc_ok=False)), "checksum"),
    (_url(_png(idat=False)), "no image data"),
    (_url(_png(iend=False)), "truncated"),
    (_url(_png(trailing=b"x")), "trailing"),
    (_url(_png(4, 4)), "at least 8"),
    (_url(_png(2000, 1001)), "pixels"),
    (_url(_png(), "jpeg"), "not a JPEG"),  # PNG bytes under a JPEG URL
    (_url(_jpeg(), "png"), "not a PNG"),  # JPEG bytes under a PNG URL
    (_url(b"\xff\xd8\xff\xd9", "jpeg"), "frame header"),
])
def test_bad_images_are_refused(url, why):
    with pytest.raises(ContractError, match=why):
        parse_decision(_decision([url]), "s1-vision")


def test_oversize_images_are_refused_before_decoding():
    big = _png() + b"\x00" * (MAX_IMAGE_BYTES + 1)  # decoded size above the gateway's 4 MiB
    with pytest.raises(ContractError):
        parse_image(_url(big))
    with pytest.raises(ContractError, match="size limit"):
        parse_image("data:image/png;base64," + "A" * (5 * 1024 * 1024 * 4 // 3))


def test_the_engine_pools_do_not_open_the_miner_protocol():
    assert set(MODELS) == {"s1-fast", "s1-pro", "s1-lite", "s1-cpu"}  # s1-vision / quyet are no reax-s1/1 models
    assert IMAGE_POOLS == {"s1-vision"} and ENGINE_MAX_OPTIONS == {"quyet-1-large": 10, "jeffan-31b": 26, "s1-vision": 26}
    from reax_subnet.s1protocol import ProtocolError, parse_s1_request

    with pytest.raises(ProtocolError):
        parse_s1_request(json.dumps({"protocol": "reax-s1/1", "request_id": "r1", "model": "s1-vision",
                                     "manifest_id": "0" * 16, "decisions": [_decision()]}).encode())


# ------------------------------------------------------------------------------------------------ contract: quyet
def test_quyet_has_its_own_ten_option_contract():
    assert MAX_OPTIONS["quyet-1-large"] == 10
    parse_decision(_decision(n_options=10), "quyet-1-large")
    with pytest.raises(ContractError, match="option count"):
        parse_decision(_decision(n_options=11), "quyet-1-large")
    parse_decision(_decision(n_options=11), "s1-pro")  # the s1-pro pool keeps its 64
    with pytest.raises(ContractError, match="option count"):
        parse_decision(_decision(n_options=27), "s1-vision")
    for it in generate_s1_items(300, seed=5):  # every text item the generator can make fits Quyet's 10 options
        parse_decision(it.decision(), "quyet-1-large")


# ------------------------------------------------------------------------------------------------ generator
def test_items_are_deterministic_per_seed_and_index():
    a, b = vision_item(42, 7), vision_item(42, 7)
    assert a == b and a.images == b.images
    assert [i.item_id for i in generate_vision_items(5, seed=9)] == [i.item_id for i in generate_vision_items(5, seed=9)]
    assert vision_item(42, 7) != vision_item(42, 8) and vision_item(42, 7) != vision_item(43, 7)
    assert vision_item(42, 7).images != vision_item(43, 7).images
    fresh = [generate_vision_items(2) for _ in range(3)]  # no seed: a secret one every time
    assert len({f[0].item_id for f in fresh}) == 3


def test_items_vary_and_every_item_has_one_valid_image():
    items = generate_vision_items(240, seed=2026)
    assert len({i.images[0] for i in items}) == 240  # no two pictures alike
    assert len({i.instructions for i in items}) > 40 and {i.kind for i in items} == {"choice", "noul", "score"}
    families = {i.instructions.split()[0:3].__str__() for i in items}
    assert len(families) >= 8
    for i in items:
        d = parse_decision(i.decision(), "s1-vision")
        assert len(d.images) == 1 and len(d.question.keys) >= 2
        assert parse_image(d.images[0]).size < 20_000  # small pictures: cheap to generate and to send
    for kind, text in (("count", "How many"), ("bars", "bar"), ("number", "number"), ("crowd", "crowded")):
        assert any(text in str(i.instructions) for i in items), kind


def test_generation_is_cheap():
    vision_item(0, 0)  # warm-up
    t = time.perf_counter()
    for i in range(100):
        vision_item(11, i)
    per_item = (time.perf_counter() - t) / 100
    assert per_item < 0.020, f"{per_item * 1000:.1f} ms per item"  # brief: < 20 ms per image


def test_the_image_decodes_and_carries_the_content():
    PIL = pytest.importorskip("PIL.Image")
    import io

    seen_colours = set()
    for i in range(30):
        img = PIL.open(io.BytesIO(base64.b64decode(vision_item(3, i).images[0].split(",", 1)[1]))).convert("RGB")
        assert img.size == (320, 240)
        seen_colours |= {c for _n, c in img.getcolors(maxcolors=100_000) if True}
    assert len(seen_colours) > 100  # shapes, bars and digits in many colours


def test_every_item_depends_on_its_image():
    """Same state, same question, other picture: the pseudo model (which hashes the image) answers differently, and the
    engine contract carries the image under the production field name."""
    from reax_subnet.s1backends import S1HashBackend

    be = S1HashBackend("s1-vision")
    base = parse_decision(vision_item(1, 0).decision(), "s1-vision")
    other = dataclasses.replace(base, images=vision_item(1, 1).images)
    text_only = dataclasses.replace(base, images=())
    assert be.logits(base) != be.logits(other) and be.logits(base) != be.logits(text_only)
    assert be.logits(base) == be.logits(dataclasses.replace(base))


def test_fresh_and_probe_items_for_the_pools():
    assert all(len(d.images) == 1 for d in fresh_decisions(6, "s1-vision"))
    assert all(not d.images for d in fresh_decisions(6, "s1-fast")) and all(not d.images for d in fresh_decisions(6, "quyet-1-large"))
    assert items_for_pool("s1-pro", 3, 1) == generate_s1_items(3, 1)
    base, variants = probe_decisions(8, "s1-vision")
    assert len(base) == len(variants) == 8 and all(len(d.images) == 1 for d in base + variants)
    for i, (b, v) in enumerate(zip(base, variants)):
        if i % 2 == 0:  # same state and picture, another question: a cache by state or by picture is caught
            assert (v.state, v.images) == (b.state, b.images) and v.question != b.question
        else:  # the same question on another picture: a cache by question is caught
            assert v.question == b.question and v.images != b.images
    tb, tv = probe_decisions(4, "s1-fast")
    assert all(not d.images for d in tb + tv)


# ------------------------------------------------------------------------------------------------ the engine wire
async def test_a_real_vision_session_sends_the_image_and_the_anti_cache_probe_covers_it(tmp_path):
    vision = dataclasses.replace(FAST_P, name="s1-vision", pool="s1-vision")
    track = CompositeTrack("dm-all", (ecmp.ModelSlot("vision", "blink-v0.3-26b-a4b", 1.0, vision, 0, calibrated=True),),
                           confirmations=1)

    def runner(variant):
        argv = [sys.executable, FAKE, "--variant", variant, "--delay", "0.01", "--jitter", "0.4", "--model", "s1-vision"]
        return CompositeRunner({"s1-vision": DevRunner(str(tmp_path / variant), {ref_of("reference"): argv[:3] + ["reference"] + argv[4:],
                                                                                  ref_of(variant): argv})})

    fast = await evaluate_composite(runner("fast"), ref_of("reference"), ref_of("fast"), track, frozenset({"vision"}), digest=D1)
    assert fast.ok and fast.parts["vision"]["mean_distance"] == 0.0, fast  # images arrive: the engine parsed them
    cache = await evaluate_composite(runner("cache"), ref_of("reference"), ref_of("cache"), track, frozenset({"vision"}), digest=D2)
    assert not cache.ok and "anti-cache probe" in cache.reason, cache


# ------------------------------------------------------------------------------------------------ slots and weights
def test_v22_stays_selectable_with_six_slots():
    assert ecmp.COMPOSITE_WEIGHTS_VARIANT == "v22" and ecmp.COMPOSITE_WEIGHTS["v22"] == V22
    assert math.fsum(V22.values()) == pytest.approx(1.0, abs=1e-12)
    t = default_track()
    assert [s.name for s in t.slots] == ["blink", "vision", "quyet", "reax1", "pplx", "d3"]  # cheapest first
    assert {s.name: s.weight for s in t.slots} == V22 and t.carried == ()
    assert {"v21", "reax-high", "approved-1021"} <= set(ecmp.COMPOSITE_WEIGHTS)  # earlier sets stay selectable
    for variant, weights in ecmp.COMPOSITE_WEIGHTS.items():
        assert math.fsum(weights.values()) == pytest.approx(1.0, abs=1e-12) and set(weights) <= set(ecmp.COMPOSITE_EVAL_ORDER)
        assert {s.name for s in default_track(variant=variant).slots} == set(weights)
    assert "vision" not in {s.name for s in default_track(variant="v21").slots}
    assert min(V22.values()) == 0.05  # a 5 % floor for every served model


def test_the_two_new_slots():
    t = default_track()
    q, v, b = t.slot("quyet"), t.slot("vision"), t.slot("blink")
    assert q.model == "quyet-1-large" and q.params.pool == "quyet-1-large" and MAX_OPTIONS[q.params.pool] == 10
    assert q.params.concurrency == 8 and q.params.temperature == pytest.approx(1.3007)
    pplx = t.slot("pplx").params
    assert (q.params.timed_items, q.params.mean_tau, q.params.item_tau, q.params.min_top_agreement,
            q.params.determinism_tau) == (pplx.timed_items, pplx.mean_tau, pplx.item_tau, pplx.min_top_agreement,
                                          pplx.determinism_tau)  # provisional gates: like pplx
    # vision = the Blink engine and weights with images: same model id for the engine, own runner key and pool
    assert v.model == b.model == "blink-v0.3-26b-a4b" and v.params.name == "s1-vision" and v.params.pool == "s1-vision"
    assert (v.params.concurrency, v.params.temperature) == (b.params.concurrency, b.params.temperature)
    # vision's eager reference is bit-identical across starts: strict gates; Blink text (CUDA graphs) got wider ones (26 Oct)
    assert (v.params.mean_tau, v.params.item_tau, v.params.min_top_agreement) == (0.05, 1.0, 0.99)
    assert v.params.mean_tau < b.params.mean_tau and v.params.item_tau < b.params.item_tau
    assert v.memory_budget_mib == b.memory_budget_mib and v.params.eval_budget_s >= b.params.eval_budget_s
    assert ecmp.SLOT_WEIGHTS_FROM == {"vision": "blink"}
    assert ecmp.parse_declared("vision,quyet", t) == {"vision", "quyet"}
    assert ecmp.parse_declared(None, t) == set(V22)


def test_composite_math_with_the_new_slots():
    t = default_track()
    gains = {"d3": 1.2, "blink": 1.1, "vision": 1.1, "reax1": 1.5, "pplx": 1.05, "quyet": 1.3}
    cs_ = composite_score(t, frozenset(gains), {k: _r(v) for k, v in gains.items()})
    assert cs_.ok and cs_.score == pytest.approx(math.exp(sum(V22[k] * math.log(v) for k, v in gains.items())))
    only = composite_score(t, frozenset({"quyet"}), {"quyet": _r(2.0)})
    assert only.ok and only.score == pytest.approx(2.0 ** 0.05)  # a Quyet-only engine needs a lot for 3 %
    bad = composite_score(t, frozenset(gains), {**{k: _r(2.0) for k in gains}, "vision": _r(0.5)})
    assert not bad.ok and bad.reason.startswith("vision:")  # the no-regression floor applies per slot


def test_the_rules_hash_follows_the_new_slots():
    t = default_track()
    h = composite_rules_sha256(t)
    assert h == composite_rules_sha256(default_track()) and h != composite_rules_sha256(default_track(variant="v21"))
    rules = composite_rules(t)
    by = {s["name"]: s for s in rules["slots"]}
    assert by["vision"]["pool"] == "s1-vision" and by["vision"]["image_items"] == GENERATOR_RULE
    assert by["quyet"]["pool"] == "quyet-1-large" and "image_items" not in by["quyet"] and "image_items" not in by["d3"]

    def swapped(name, **kw):
        s = t.slot(name)
        return CompositeTrack(t.name, tuple(dataclasses.replace(x, **kw) if x.name == name else x for x in t.slots))

    assert composite_rules_sha256(swapped("quyet", model="quyet-2")) != h
    assert composite_rules_sha256(swapped("vision", memory_budget_mib=1)) != h
    q = t.slot("quyet")
    looser = dataclasses.replace(q, params=dataclasses.replace(q.params, mean_tau=q.params.mean_tau + 0.01))
    assert composite_rules_sha256(CompositeTrack(t.name, tuple(looser if x.name == "quyet" else x for x in t.slots))) != h


def test_the_image_generator_is_part_of_the_rules(monkeypatch):
    h = composite_rules_sha256(default_track())
    monkeypatch.setattr(ecmp, "IMAGE_ITEMS_RULE", GENERATOR_RULE + " v2")
    assert composite_rules_sha256(default_track()) != h  # a new generator is a new rule set
    assert composite_rules_sha256(default_track(variant="v21")) == composite_rules_sha256(default_track(variant="v21"))


# ------------------------------------------------------------------------------------------------ release guard
def test_release_ready_names_the_uncalibrated_slots_and_the_missing_digests(monkeypatch):
    from reax_subnet import config
    from reax_subnet.config import ENGINE_COMPOSITE_REFERENCE_IMAGES as refs

    t = default_track()
    assert t.release_ready() == []  # digests are pinned; quyet calibrated 25 Oct, vision 26 Oct
    assert default_track(variant="v21").release_ready() == []  # the four calibrated slots stay ready
    assert all("@sha256:" in refs[k] for k in ("quyet", "vision"))
    with monkeypatch.context() as m:  # vision uncalibrated (the state before the 26 Oct check)
        m.setattr(ecmp, "_DEFAULT_SLOTS", tuple({**r, "release_calibrated": False} if r["name"] == "vision" else r
                                                 for r in ecmp._DEFAULT_SLOTS))
        assert default_track().release_ready() == ["slot vision is not calibrated"]
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "quyet", None)
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "vision", None)
    assert t.release_ready() == ["slot vision has no published reference image digest",
                                 "slot quyet has no published reference image digest"]


def test_the_per_slot_marker_wins_over_the_global_flag(monkeypatch):
    from reax_subnet import config

    assert ecmp.CALIBRATED_ON_RELEASE_REFERENCES is True  # the global flag stays True: it still covers the four slots
    t = default_track()
    assert [s.name for s in t.slots if s.calibrated] == ["blink", "vision", "quyet", "reax1", "pplx", "d3"]
    monkeypatch.setattr(ecmp, "CALIBRATED_ON_RELEASE_REFERENCES", False)  # global flag off: the four lose it; the own markers of vision and quyet stay
    assert [s.name for s in default_track().slots if s.calibrated] == ["vision", "quyet"]
    monkeypatch.setattr(ecmp, "CALIBRATED_ON_RELEASE_REFERENCES", True)
    cal = {n: {"noise": 0.01} for n in ("quyet", "vision")}  # development: a calibration entry marks a slot calibrated
    ready = default_track(cal)
    assert [r for r in ready.release_ready() if "calibrated" in r] == []
    assert ready.release_ready() == []  # the digests are pinned in config
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "quyet", None)
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "vision", None)
    assert len(default_track(cal).release_ready()) == 2  # a missing digest is still named
    digest = "ghcr.io/reaxlabs/x@sha256:" + "a" * 64
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "quyet", digest)
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "vision", digest)
    assert default_track(cal).release_ready() == []
    # a calibration entry bound the dev slots to the digest of that moment; the bare release table was calibrated on the real one
    assert default_track().release_ready() == ["slot vision calibrated on a different digest",
                                               "slot quyet calibrated on a different digest"]


def test_finney_refuses_the_v22_track_until_it_is_ready(monkeypatch):
    from reax_subnet import competition_split as _cs
    monkeypatch.setattr(_cs, "ENGINE_V1_BRIDGE_STATE_SHA256", None)  # before the pin of the frozen v1 file (27 Oct slot)
    from reax_subnet import config
    from reax_subnet.config import ConfigError
    from reax_subnet.worker import composite_references, composite_track_from_env

    with monkeypatch.context() as m:  # vision uncalibrated (the state before the 26 Oct check)
        m.setattr(ecmp, "_DEFAULT_SLOTS", tuple({**r, "release_calibrated": False} if r["name"] == "vision" else r
                                                 for r in ecmp._DEFAULT_SLOTS))
        with pytest.raises(ConfigError, match="not release-ready: slot vision is not calibrated$"):
            composite_track_from_env({}, release_mode=True)
    with pytest.raises(ConfigError, match="ENGINE_V1_BRIDGE_STATE_SHA256"):  # ready now; only the v1 bridge pin refuses
        composite_track_from_env({}, release_mode=True)
    assert set(composite_references({}, release_mode=True)) >= {"quyet", "vision"}  # pinned: accepted
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "quyet", None)
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "vision", None)
    with pytest.raises(ConfigError, match="missing(?=.*quyet)(?=.*vision)"):
        composite_references({}, release_mode=True)


def test_vision_runs_on_the_blink_weights_directory(tmp_path, monkeypatch):
    from reax_subnet import worker

    made = []

    class Made:  # stands in for DockerRunner: records the weights directory and the model each runner is built for
        protected_images, gpu, docker_runtime, registries = (), "0", "", ()
        memory, cpus, nvml_check, release_mode, review_dir, reference_ref, fla_config_root = "1g", 1, False, False, "", "", ""

        def __init__(self, workdir, weights, **kw):
            if "model_env" in kw:  # the base runner passed to composite_from_env is not one of the per-model runners
                made.append((kw["model_env"][ecmp.MODEL_ENV], weights))

    class E:
        release_mode, workdir = False, str(tmp_path)

    monkeypatch.setattr("reax_subnet.engrunner.DockerRunner", Made)
    monkeypatch.setattr(worker, "composite_references", lambda env, release_mode: {})
    dirs = ",".join(f"{n}=/w/{n}" for n in ("blink", "quyet", "reax1", "pplx", "d3"))  # no vision entry
    runner, track = worker.composite_from_env({"REAX_ENGINE_COMPOSITE_WEIGHTS": dirs}, E, Made("/w", "/b"))
    assert set(runner.runners) == {"blink-v0.3-26b-a4b", "s1-vision", "quyet-1-large", "reax-1", "pplx-decider-v1.1-27b", "d3-turbo"}
    assert sorted(made) == sorted([("blink-v0.3-26b-a4b", "/w/blink"), ("blink-v0.3-26b-a4b", "/w/blink"),
                                        ("quyet-1-large", "/w/quyet"), ("reax-1", "/w/reax1"),
                                        ("pplx-decider-v1.1-27b", "/w/pplx"), ("d3-turbo", "/w/d3")])  # vision = blink's weights
    with pytest.raises(Exception, match="lacks a weights directory for: quyet"):
        worker.composite_from_env({"REAX_ENGINE_COMPOSITE_WEIGHTS": dirs.replace("quyet=/w/quyet,", "")}, E, Made("/w", "/b"))


# ------------------------------------------------------------------------------------------------ time budget
def test_the_time_budget_of_six_slots():
    t = default_track()
    assert t.budget_s() == 10_200.0 and t.max_evals == 2  # 900 slack + blink 1200 + vision 1500 + quyet 1500 + reax1 1500 + pplx 1200 + d3 2400
    assert default_track(variant="v21").budget_s() == 7_200.0
    assert t.budget_s(frozenset({"vision", "quyet"})) == 900.0 + 1500.0 + 1500.0


# ------------------------------------------------------------------------------------------------ item tapes
async def test_every_model_records_its_own_tape_and_a_parent_run_replays_them(monkeypatch):
    import reax_subnet.engrunner as er

    seen = {}

    async def fake(runner, reference, ref, track, params, block=0, digest="", progress=None):
        tape = progress["items"]
        seen.setdefault(digest + "/" + ref, {})[params.name] = tape._take(("fresh", 4, params.name), lambda: object())
        return EvalResult(digest=digest, track=track, ok=True, score=1.1, block=block)

    class Runner:
        async def finish_candidate(self, ref):
            pass

    monkeypatch.setattr(er, "evaluate", fake)
    first = SlotTapes()
    await evaluate_composite(Runner(), "r", "child", TRACK, ALL, digest="c", progress={"items": first})
    assert first.recorded == 4  # one item set per model
    replay = first.replay()
    await evaluate_composite(Runner(), "r", "parent", TRACK, ALL, digest="p", progress={"items": replay})
    child, parent = seen["c/child"], seen["p/parent"]
    assert set(child) == set(parent) == set("abcd") and all(child[m] is parent[m] for m in child)  # the same items
    assert len({id(v) for v in child.values()}) == 4  # one tape per model, not shared between models
    assert SlotTapes().recorded == 0


# ------------------------------------------------------------------------------------------------ copy line on dm-all
NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
COPYLINE = datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)


class _Reviewed:
    """What a DockerRunner offers the copy line: ``review_dir``, a label reader, an owner lookup."""

    release_mode = docker_runtime = user = pull = registries = None

    def __init__(self, review_dir):
        self.review_dir = review_dir
        self.rechecked = []

    async def models_label(self, ref):
        return None

    async def owner_of(self, ref):
        return "hk0"

    async def discard(self, ref):
        pass

    async def recheck_copyline(self, ref):
        self.rechecked.append(ref)


def _fast_composite(scores, calls=None):
    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        if calls is not None:
            calls.append(digest)
        s = scores[digest]
        res = {x.name: _r(s) for x in track.slots if x.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    return fake


def _record(review_dir, digest, parents_label=None):
    rec = engreview.new_record("img@sha256:" + digest, digest, [], [], {}, {})
    rec["copyline"] = 1
    if parents_label is not None:
        rec["parents_label"] = parents_label
    engreview.write_record(str(review_dir), digest, rec)


def test_the_composite_runner_forwards_the_copy_line_hooks(tmp_path):
    first = _Reviewed(str(tmp_path / "rev"))
    runner = CompositeRunner({"a": first, "b": _Reviewed(str(tmp_path / "rev"))})
    assert runner.review_dir == str(tmp_path / "rev")
    asyncio.run(runner.recheck_copyline("img@sha256:" + D1))
    assert first.rechecked == ["img@sha256:" + D1]
    assert CompositeRunner({"a": object()}).review_dir == ""  # a runner without a review directory: no copy-line records
    with pytest.raises(ValueError, match="review_dir"):
        CompositeRunner({"a": _Reviewed("x"), "b": _Reviewed("y")})


async def test_a_dm_all_champion_is_crowned_with_the_copy_line_on(tmp_path, monkeypatch):
    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", COPYLINE)
    monkeypatch.setattr(EV, "evaluate_composite", _fast_composite({D1: 1.2}))
    runner = CompositeRunner({"a": _Reviewed(str(tmp_path / "rev"))})
    comp = EV.EngineCompetition(runner, {"dm-all": "ref"}, str(tmp_path / "e.json"), composites={"dm-all": TRACK},
                                start_utc=None, utcnow=lambda: NOW)
    snap = _snapshot([("dm-all", D1)], 500)
    await comp.run_cycle(snap, set())
    assert comp.states["dm-all"].champion is None  # no review record: not crownable (fail closed)
    _record(tmp_path / "rev", D1)
    await comp.run_cycle(snap, set())
    ch = comp.states["dm-all"].champion
    assert ch is not None and ch.digest == D1 and D1 in comp.states["dm-all"].derived


async def test_the_parent_is_measured_in_the_same_session_on_the_childs_models(tmp_path, monkeypatch):
    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", COPYLINE)
    now = EC.ENGINE_PARENT_SESSION_START_UTC + timedelta(hours=2)
    assert now < cs.ENGINE_QUANT_CONVERSION_END_UTC
    calls, items = [], {}

    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        tapes = progress["items"]
        assert isinstance(tapes, SlotTapes)
        got = {s.name: tapes.for_slot(s.name)._take(("fresh", 4, "m"), lambda: object()) for s in track.slots if s.name in declared}
        calls.append((digest, declared))
        items.setdefault(digest, []).append(got)
        s = {D2: 1.52, D1: 1.46}[digest]
        res = {x.name: _r(s) for x in track.slots if x.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    monkeypatch.setattr(EV, "evaluate_composite", fake)
    runner = CompositeRunner({"a": _Reviewed(str(tmp_path / "rev"))})
    runner.first.owner_of = lambda ref: _owner(ref)

    async def _owner(ref):
        return "hk0" if D2 in ref else "hk1"

    comp = EV.EngineCompetition(runner, {"dm-all": "ref"}, str(tmp_path / "e.json"), composites={"dm-all": TRACK},
                                start_utc=None, utcnow=lambda: now)
    st = comp.states["dm-all"]
    params = comp.tracks["dm-all"]
    st.results[D1] = EvalResult(D1, "dm-all", True, score=1.5, block=50, session="p")
    st.hold(EvalResult(D1, "dm-all", True, score=1.45, block=40, session="p0"), params)
    st.hold(st.results[D1], params)
    st.first_seen[D1] = (50, "hk1")
    st.owners[D1] = "hk1"
    label = json.dumps([{"uid": 1, "digest": D1, "files": ["/app/a.py"]}])
    _record(tmp_path / "rev", D2, label)
    _record(tmp_path / "rev", D1)
    await comp.run_cycle(_snapshot([("dm-all", D2), ("dm-all", D1)], 1000), set())
    assert [d for d, _ in calls] == [D2, D1, D2, D1]  # child, its parent right after it, in both sessions
    assert all(declared == ALL for _, declared in calls)  # the parent is measured on the models the child declared
    assert all(a is b for a, b in zip((v for g in items[D1] for v in g.values()), (v for g in items[D2] for v in g.values())))
    events = [e for e in st.cycle_events if e.get("event") == "parent_reference"]
    assert len(events) == 2 and all(e["ok"] and e["parent"] == D1 for e in events)
    assert st.results[D1].score == 1.5 and st.champion is not None and st.champion.digest == D2


async def test_the_parent_runs_of_a_cycle_use_at_most_one_session_budget_of_the_childs_models(tmp_path, monkeypatch):
    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", COPYLINE)
    now = EC.ENGINE_PARENT_SESSION_START_UTC + timedelta(hours=2)
    clock = [0.0]
    calls = []

    async def fake(runner, reference, ref, track, declared, block=0, digest="", progress=None, references=None):
        calls.append(digest)
        if digest == D1:  # the parent run takes the whole session budget of the declared models
            clock[0] += track.budget_s(declared)
        s = {D2: 1.52, D1: 1.46}[digest]
        res = {x.name: _r(s) for x in track.slots if x.name in declared}
        return to_eval_result(track, digest, block, composite_score(track, declared, res), res)

    monkeypatch.setattr(EV, "evaluate_composite", fake)
    runner = CompositeRunner({"a": _Reviewed(str(tmp_path / "rev"))})

    async def _owner(ref):
        return "hk0" if D2 in ref else "hk1"

    runner.first.owner_of = lambda ref: _owner(ref)
    comp = EV.EngineCompetition(runner, {"dm-all": "ref"}, str(tmp_path / "e.json"), composites={"dm-all": TRACK},
                                start_utc=None, utcnow=lambda: now, clock=lambda: clock[0])
    st = comp.states["dm-all"]
    params = comp.tracks["dm-all"]
    st.results[D1] = EvalResult(D1, "dm-all", True, score=1.5, block=50, session="p")
    st.hold(EvalResult(D1, "dm-all", True, score=1.45, block=40, session="p0"), params)
    st.hold(st.results[D1], params)
    st.first_seen[D1] = (50, "hk1")
    st.owners[D1] = "hk1"
    _record(tmp_path / "rev", D2, json.dumps([{"uid": 1, "digest": D1, "files": ["/app/a.py"]}]))
    _record(tmp_path / "rev", D1)
    await comp.run_cycle(_snapshot([("dm-all", D2), ("dm-all", D1)], 1000), set())
    assert calls.count(D1) == 1 and calls.count(D2) >= 2  # two child sessions, one parent run: the second would exceed the budget
    assert len([e for e in st.cycle_events if e.get("event") == "parent_reference"]) == 1


# ------------------------------------------------------------------------------------------------ Tue/Fri slots, Option A
TUE, FRI = datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc), datetime(2026, 10, 30, 14, 30, tzinfo=timezone.utc)


def _slot_state(track, slot):
    p = track.params()
    sub = Submission(7, "hk", p.name, "repo/image", D1, 50)
    st = TrackState(derived={D1: []})
    observe(st, [sub], 100)
    for i in range(p.confirmations):
        st.results[D1] = EvalResult(D1, p.name, True, score=1.3, block=100 + i, session=f"s{i}")
        update_champion(st, [sub], p, 100 + i, now=slot - timedelta(hours=1))
    return st, sub, p


def test_dm_all_follows_the_tuesday_friday_slots(monkeypatch):
    from conftest import RELEASE_ENGINE_CROWN_SLOTS_UTC

    monkeypatch.setattr(cs, "ENGINE_CROWN_SLOTS_UTC", RELEASE_ENGINE_CROWN_SLOTS_UTC)
    assert {s.weekday() for s in RELEASE_ENGINE_CROWN_SLOTS_UTC} == {1, 4} and RELEASE_ENGINE_CROWN_SLOTS_UTC[0] == TUE
    st, sub, p = _slot_state(TRACK, TUE)
    assert st.challenger["slot_utc"] == TUE.isoformat() and st.champion is None  # confirmed: the title waits for the slot
    assert EC.finalize_engine_slot(st, [sub], p, TUE - timedelta(seconds=1), 1000) is None
    # the Tuesday 27 Oct slot is the last v1 crowning (ENGINE_V2_START_UTC): dm-all is crowned from the Friday slot on
    assert EC.finalize_engine_slot(st, [sub], p, TUE, 1000) is None and st.champion is None
    st, sub, p = _slot_state(TRACK, FRI)
    st.weights_ok.add(D1)
    ch = EC.finalize_engine_slot(st, [sub], p, FRI, 1000)
    assert ch is not None and ch.digest == D1 and st.history[-1]["slot_utc"] == FRI.isoformat()


def test_option_a_applies_to_dm_all_from_the_friday_slot(monkeypatch):
    """The conversion window ends at the Friday 30 Oct slot itself (ENGINE_QUANT_CONVERSION_END_UTC = 14:30Z): a title taken at
    that slot needs the weights screen (``weights_ok``), exactly as on the v1 track."""
    from conftest import RELEASE_ENGINE_CROWN_SLOTS_UTC

    monkeypatch.setattr(cs, "ENGINE_CROWN_SLOTS_UTC", RELEASE_ENGINE_CROWN_SLOTS_UTC)
    assert cs.ENGINE_QUANT_CONVERSION_END_UTC == FRI
    st, sub, p = _slot_state(TRACK, FRI)
    assert EC.finalize_engine_slot(st, [sub], p, FRI, 1000) is None and st.history[-1]["event"] == "blocked_noncompliant"
    st.weights_ok.add(D1)
    assert EC.finalize_engine_slot(st, [sub], p, FRI, 1000).digest == D1
    st1 = TrackState(derived={D1: []})  # the v1 track: a title taken before the window ends needs no screen (Tue 27 Oct slot)
    p1 = EV.TRACKS["s1-fast"]
    sub1 = Submission(7, "hk", p1.name, "repo/image", D1, 50)
    observe(st1, [sub1], 100)
    for i in range(p1.confirmations):
        st1.results[D1] = EvalResult(D1, p1.name, True, score=3.0, block=100 + i, session=f"s{i}")
        update_champion(st1, [sub1], p1, 100 + i, now=TUE - timedelta(hours=1))
    assert EC.finalize_engine_slot(st1, [sub1], p1, TUE, 1000).digest == D1


def test_the_offline_finaliser_knows_the_composite_track(monkeypatch):
    from conftest import RELEASE_ENGINE_CROWN_SLOTS_UTC
    from reax_subnet.engslot import finalize_document, track_params

    monkeypatch.setattr(cs, "ENGINE_CROWN_SLOTS_UTC", RELEASE_ENGINE_CROWN_SLOTS_UTC)
    assert set(track_params()) == {"s1-fast", "dm-all"} and track_params()["dm-all"] == default_track().params()
    st, sub, p = _slot_state(default_track(), FRI)  # the first dm-all crowning is the Friday slot
    st.weights_ok.add(D1)
    raw = {"version": 1, "saved_at": 1, "binds_hotkey": False, "tracks": {"dm-all": st.to_json()}}
    snapshot = {"block": 1000, "rows": [{"uid": 7, "hotkey": "hk", "block": 50, "text": f"reax-eng/1:dm-all:repo/image@sha256:{D1}"}]}
    out = finalize_document(raw, snapshot, FRI)
    assert out["tracks"]["dm-all"]["champion"]["digest"] == D1


# ------------------------------------------------------------------------------------------------ v1 stays as on rc11p6
def test_a_v1_state_serialises_without_the_composite_fields():
    r = EvalResult("d", "s1-fast", True, score=1.2, block=3, session="x")
    assert "parts" not in r.to_json() and "cand_memory_mib" not in r.to_json()
    c = EvalResult("d", "dm-all", True, score=1.2, block=3, session="x", cand_memory_mib=900, parts={"a": {"score": 1.0}})
    assert c.to_json()["parts"] == {"a": {"score": 1.0}} and c.to_json()["cand_memory_mib"] == 900
    assert EvalResult.from_json(c.to_json()) == c and EvalResult.from_json(r.to_json()) == r
