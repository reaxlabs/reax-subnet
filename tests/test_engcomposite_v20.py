"""Engine competition v2.0 lineup (25 Oct 2026): the models decisionmodels.io serves after its switch. Default variant
``v20`` (mammoth 0.37, blink 0.33, reax1 0.10, jeffan 0.10, vision 0.10), the data-only fallback ``v20-d3``, the slots
mammoth and jeffan, Jeffan's own contract and the release guard (since the GPU check of 26 Oct all five slots are calibrated;
the guard tests make a slot uncalibrated, unpublished or re-digested by monkeypatch)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from reax_subnet import config
from reax_subnet import engcomposite as ecmp
from reax_subnet.engcomposite import (COMPOSITE_WEIGHTS, CompositeTrack, composite_rules, composite_rules_sha256,
                                      composite_score, default_track, parse_declared)
from reax_subnet.s1contract import (ENGINE_MAX_OPTIONS, IMAGE_POOLS, MAX_OPTIONS, MODELS, ContractError, parse_decision)
from reax_subnet.s1items import generate_s1_items

from test_engcomposite import _r

V20 = {"mammoth": 0.37, "blink": 0.33, "reax1": 0.10, "jeffan": 0.10, "vision": 0.10}
LABELS = ["mammoth", "blink", "reax1", "jeffan", "vision"]


def _uncalibrated(monkeypatch, *names):
    """Rows of the slot table with ``release_calibrated`` False for the named slots (what the rows were before the check)."""
    rows = tuple({**r, "release_calibrated": False} if r["name"] in names else r for r in ecmp._DEFAULT_SLOTS)
    monkeypatch.setattr(ecmp, "_DEFAULT_SLOTS", rows)


def _decision(n_options=3, questions=1, images=None):
    qs = {f"q{i}": {"type": "choice", "instructions": "?", "criteria": {f"k{j}": None for j in range(n_options)}}
          for i in range(questions)}
    d = {"id": "d", "state": "s", "questions": qs}
    if images is not None:
        d["images"] = images
    return d


# ------------------------------------------------------------------------------------------------ weights
def test_v20_is_the_default_and_sums_to_one():
    assert ecmp.COMPOSITE_WEIGHTS_VARIANT == "v20" and COMPOSITE_WEIGHTS["v20"] == V20
    assert math.fsum(V20.values()) == pytest.approx(1.0, abs=1e-12)
    t = default_track()
    assert [s.name for s in t.slots] == ["blink", "vision", "reax1", "jeffan", "mammoth"]  # cheapest first
    assert {s.name: s.weight for s in t.slots} == V20 and t.carried == ()
    assert min(V20.values()) == 0.10  # a 10 % floor for every served model
    assert not {"quyet", "pplx", "d3", "plumb"} & {s.name for s in t.slots}
    assert {"v22", "v21", "reax-high", "approved-1021"} <= set(COMPOSITE_WEIGHTS)  # earlier sets stay selectable
    for variant, weights in COMPOSITE_WEIGHTS.items():
        assert math.fsum(weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert {s.name for s in default_track(variant=variant).slots} == set(weights)


def test_the_commitment_label_list_is_the_five_slots():
    t = default_track()
    assert set(LABELS) == {s.name for s in t.slots}
    assert parse_declared(None, t) == set(LABELS) == parse_declared("all", t)
    assert parse_declared(",".join(LABELS), t) == set(LABELS)
    assert parse_declared("mammoth,jeffan", t) == {"mammoth", "jeffan"}
    for gone in ("quyet", "pplx", "d3", "plumb"):  # out of the default: a typo or an old label must fail
        with pytest.raises(ValueError, match="unknown"):
            parse_declared(f"mammoth,{gone}", t)


def test_the_fallback_is_a_one_line_variant_d3_for_mammoth():
    assert COMPOSITE_WEIGHTS["v20-d3"] == {**{k: v for k, v in V20.items() if k != "mammoth"}, "d3": V20["mammoth"]}
    t, fb = default_track(), default_track(variant="v20-d3")
    assert [s.name for s in fb.slots] == ["blink", "vision", "reax1", "jeffan", "d3"] and fb.carried == ()
    # d3 keeps its existing calibrated constants and is calibrated; the other four slots are the same rows
    d3 = fb.slot("d3")
    assert d3.weight == 0.37 and d3.model == ecmp.D3_SLOT_MODEL and d3.calibrated
    assert (d3.params.concurrency, d3.params.reference_noise_std, d3.memory_budget_mib) == (8, 0.0057, 45_056)
    for name in ("blink", "vision", "reax1", "jeffan"):
        assert fb.slot(name) == t.slot(name)
    assert composite_rules_sha256(fb) != composite_rules_sha256(t)  # a different rule set, never by accident
    assert not any("mammoth" in r for r in fb.release_ready())  # the fallback has no mammoth slot to name


# ------------------------------------------------------------------------------------------------ slots
def test_the_mammoth_slot_uses_the_s1_pro_pool_like_d3():
    t = default_track()
    m, d3 = t.slot("mammoth"), default_track(variant="v20-d3").slot("d3")
    assert m.model == "mammoth-1" and m.params.name == "mammoth-1" and m.params.pool == "s1-pro" == d3.params.pool
    assert MAX_OPTIONS[m.params.pool] == 64
    assert (m.params.concurrency, m.memory_budget_mib, m.params.eval_budget_s) == (
        d3.params.concurrency, d3.memory_budget_mib, d3.params.eval_budget_s)  # noise is the only measured difference
    assert (m.params.mean_tau, m.params.item_tau, m.params.min_top_agreement, m.params.determinism_tau) == (
        d3.params.mean_tau, d3.params.item_tau, d3.params.min_top_agreement, d3.params.determinism_tau)
    assert m.calibrated and m.calibrated_on == config.ENGINE_COMPOSITE_CALIBRATED_ON["mammoth"]  # GPU check 26 Oct
    assert m.params.reference_noise_std == 0.0100 and d3.params.reference_noise_std == 0.0057  # 1 %, not d3's 0.57 %
    assert m.params.name != d3.params.name  # own runner key and weights directory


def test_the_jeffan_slot():
    t = default_track()
    j, q = t.slot("jeffan"), default_track(variant="v22").slot("quyet")
    assert j.model == "jeffan-31b-1710" and j.params.pool == "jeffan-31b" and MAX_OPTIONS["jeffan-31b"] == 26
    assert (j.params.concurrency, j.params.reference_noise_std, j.params.temperature) == (8, 0.0350, 1.0)  # T=1.0 = JEFFAN_DEFAULTS, not the Quyet value
    assert (j.params.mean_tau, j.params.item_tau, j.params.min_top_agreement, j.params.determinism_tau) == (
        q.params.mean_tau, q.params.item_tau, q.params.min_top_agreement, q.params.determinism_tau)
    assert j.params.eval_budget_s > q.params.eval_budget_s  # 2.2x the cost per decision
    assert j.calibrated and j.calibrated_on == config.ENGINE_COMPOSITE_CALIBRATED_ON["jeffan"] and j.weight == 0.10


def test_vision_stays_the_blink_engine_with_images():
    t = default_track()
    v, b = t.slot("vision"), t.slot("blink")
    assert v.model == b.model and v.params.pool == "s1-vision" and v.calibrated and b.calibrated
    assert ecmp.SLOT_WEIGHTS_FROM == {"vision": "blink"}
    assert v.params.reference_noise_std == 0.0030 and v.params.timed_items == 256  # fresh-image noise, eager reference


# ------------------------------------------------------------------------------------------------ Jeffan's contract
def test_jeffan_contract_accepts_one_question_up_to_26_options():
    assert ENGINE_MAX_OPTIONS["jeffan-31b"] == 26 and "jeffan-31b" not in MODELS  # no reax-s1/1 model
    parse_decision(_decision(n_options=2), "jeffan-31b")
    parse_decision(_decision(n_options=11), "jeffan-31b")  # Quyet's limit (10) does not apply
    parse_decision(_decision(n_options=26), "jeffan-31b")  # the public gateway limit (#26863)
    for it in generate_s1_items(300, seed=5):
        parse_decision(it.decision(), "jeffan-31b")  # every generated text item fits


def test_jeffan_contract_rejects_what_the_engine_refuses():
    with pytest.raises(ContractError, match="option count"):
        parse_decision(_decision(n_options=27), "jeffan-31b")
    with pytest.raises(ContractError, match="option count"):
        parse_decision(_decision(n_options=1), "jeffan-31b")
    with pytest.raises(ContractError, match="exactly one question"):
        parse_decision(_decision(questions=2), "jeffan-31b")
    with pytest.raises(ContractError, match="text only"):
        parse_decision(_decision(images=["data:image/png;base64,AAAA"]), "jeffan-31b")
    with pytest.raises(ContractError, match="text only"):
        parse_decision(_decision(images=[]), "jeffan-31b")
    with pytest.raises(ContractError, match="exactly the fields"):
        d = _decision()
        d["extra"] = 1
        parse_decision(d, "jeffan-31b")
    assert "jeffan-31b" not in IMAGE_POOLS
    parse_decision(_decision(n_options=64), "s1-pro")  # the mammoth pool keeps s1-pro's 64
    with pytest.raises(ContractError, match="option count"):
        parse_decision(_decision(n_options=65), "s1-pro")


def test_jeffan_has_a_release_manifest_alias_and_is_no_miner_model():
    from reax_subnet.s1backends import ENGINE_POOL_MANIFEST

    assert ENGINE_POOL_MANIFEST["jeffan-31b"] == "s1-pro"


# ------------------------------------------------------------------------------------------------ rules and math
def test_the_rules_hash_follows_the_slot_set():
    t = default_track()
    h = composite_rules_sha256(t)
    assert h == composite_rules_sha256(default_track())
    assert h not in {composite_rules_sha256(default_track(variant=v)) for v in COMPOSITE_WEIGHTS if v != "v20"}
    by = {s["name"]: s for s in composite_rules(t)["slots"]}
    assert by["jeffan"]["pool"] == "jeffan-31b" and by["jeffan"]["max_options"] == 26
    assert by["mammoth"]["pool"] == "s1-pro" and by["mammoth"]["max_options"] == 64 and by["mammoth"]["model"] == "mammoth-1"
    assert "image_items" not in by["jeffan"] and "image_items" in by["vision"]

    def swapped(name, **kw):
        return CompositeTrack(t.name, tuple(dataclasses.replace(x, **kw) if x.name == name else x for x in t.slots))

    assert composite_rules_sha256(swapped("mammoth", model="mammoth-2")) != h
    assert composite_rules_sha256(swapped("jeffan", memory_budget_mib=1)) != h
    j = t.slot("jeffan")
    looser = dataclasses.replace(j, params=dataclasses.replace(j.params, mean_tau=j.params.mean_tau + 0.01))
    assert composite_rules_sha256(CompositeTrack(t.name, tuple(looser if x.name == "jeffan" else x for x in t.slots))) != h


def test_composite_math_with_the_lineup():
    t = default_track()
    gains = {"mammoth": 1.2, "blink": 1.1, "vision": 1.1, "reax1": 1.5, "jeffan": 1.3}
    cs_ = composite_score(t, frozenset(gains), {k: _r(v) for k, v in gains.items()})
    assert cs_.ok and cs_.score == pytest.approx(math.exp(sum(V20[k] * math.log(v) for k, v in gains.items())))
    only = composite_score(t, frozenset({"mammoth"}), {"mammoth": _r(2.0)})
    assert only.ok and only.score == pytest.approx(2.0 ** 0.37)  # a mammoth-only engine: 37 % of the weight
    bad = composite_score(t, frozenset(gains), {**{k: _r(2.0) for k in gains}, "jeffan": _r(0.5)})
    assert not bad.ok and bad.reason.startswith("jeffan:")


def test_budget_and_noise_of_v20():
    t = default_track()
    assert t.budget_s() == 900 + 1200 + 1500 + 1500 + 2160 + 2400 == 9660  # 2 h 41 min hard cap per full evaluation
    assert 86400 / t.budget_s() == pytest.approx(8.94, abs=0.01)  # at most 8 complete evaluations a day at the caps
    # measured noise (26 Oct): mammoth 1 % (set above the 0.33 % throughput noise), blink .0751, reax1 .0325, jeffan .0350 (set above the .0217 throughput noise), vision .0030
    sigma = math.sqrt((0.37 * 0.0100) ** 2 + (0.33 * 0.0751) ** 2 + (0.10 * 0.0325) ** 2 + (0.10 * 0.0350) ** 2
                      + (0.10 * 0.0030) ** 2)
    # = sqrt(1.369e-05 + 6.1420e-04 + 1.05625e-05 + 1.225e-05 + 9e-08) = sqrt(6.5079e-04)
    assert t.noise() == pytest.approx(sigma) and sigma == pytest.approx(0.025511, abs=1e-6)
    assert t.params().margin == 0.03  # the track's own margin; the one a challenger needs is max(3 %, 3 sigma):
    assert max(0.03, 3 * t.noise()) == pytest.approx(0.076532, abs=1e-6)  # 3 x 0.025511
    assert t.slot("mammoth").floor == pytest.approx(0.97)  # 1 - max(2 %, 3 x .0100) = 0.97
    assert t.slot("jeffan").floor == pytest.approx(0.895) and t.slot("vision").floor == pytest.approx(0.98)


# ------------------------------------------------------------------------------------------------ release guard
def test_release_ready_is_empty_after_the_release_calibration():
    t = default_track()
    assert t.release_ready() == []  # GPU check 26 Oct: mammoth, jeffan and vision are calibrated and published
    for new in ("mammoth", "jeffan", "vision"):
        assert "@sha256:" in config.ENGINE_COMPOSITE_REFERENCE_IMAGES[new]
        assert t.slot(new).calibrated_on == config.ENGINE_COMPOSITE_REFERENCE_IMAGES[new]
    assert default_track(variant="v20-d3").release_ready() == []  # the fallback keeps vision and jeffan: ready as well


def test_release_ready_names_every_slot_that_is_not_ready(monkeypatch):
    _uncalibrated(monkeypatch, "mammoth", "jeffan", "vision")
    t = default_track()
    out = t.release_ready()
    assert out == ["slot vision is not calibrated", "slot jeffan is not calibrated", "slot mammoth is not calibrated"]
    for calibrated in ("blink", "reax1"):  # their calibrated rows and digests are unchanged
        assert not any(calibrated in r for r in out)
        assert "@sha256:" in config.ENGINE_COMPOSITE_REFERENCE_IMAGES[calibrated]
    # the fallback is blocked only by the two new slots that it keeps
    assert default_track(variant="v20-d3").release_ready() == [
        "slot vision is not calibrated", "slot jeffan is not calibrated"]


def test_an_unpublished_digest_is_a_release_reason(monkeypatch):
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "jeffan", None)
    assert "slot jeffan has no published reference image digest" in default_track().release_ready()


def test_a_digest_and_a_calibration_clear_one_reason_each(monkeypatch):
    _uncalibrated(monkeypatch, "mammoth", "jeffan", "vision")
    cal = {"mammoth": {"noise": 0.006}}  # a calibration entry marks the slot calibrated (development networks)
    out = default_track(cal).release_ready()
    assert not any("mammoth" in r for r in out) and any("jeffan" in r for r in out) and any("vision" in r for r in out)


def test_worker_refuses_the_v20_track_until_digests_and_calibration_exist(monkeypatch):
    from reax_subnet import competition_split as _cs
    monkeypatch.setattr(_cs, "ENGINE_V1_BRIDGE_STATE_SHA256", None)  # before the pin of the frozen v1 file (27 Oct slot)
    from reax_subnet.config import ConfigError
    from reax_subnet.worker import composite_references, composite_track_from_env

    with monkeypatch.context() as m:  # a slot that is not calibrated: refused before anything else
        _uncalibrated(m, "mammoth", "jeffan", "vision")
        with pytest.raises(ConfigError, match="not release-ready"):
            composite_track_from_env({}, release_mode=True)
    # calibrated: the track itself is ready; only the v1 bridge pin (a separate, later release step) still refuses
    with pytest.raises(ConfigError, match="ENGINE_V1_BRIDGE_STATE_SHA256") as exc:
        composite_track_from_env({}, release_mode=True)
    assert "not release-ready" not in str(exc.value)
    assert composite_references({}, release_mode=True)["mammoth"].endswith("3080a593060542bccbdaf7cf0c66593fc182d9753b3facd50206df86d93b8588")
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "mammoth", None)
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, "jeffan", None)
    with pytest.raises(ConfigError, match="mammoth.*jeffan|jeffan.*mammoth|missing"):
        composite_references({}, release_mode=True)


# ------------------------------------------------------------------------------------------------ M5: calibration bound to the digest
def test_every_calibrated_slot_records_the_digest_it_was_calibrated_on():
    for variant in ("v20", "v20-d3", "v21", "v22", "approved-1021"):
        t = default_track(variant=variant)
        for s in t.slots:
            if s.calibrated:
                assert s.calibrated_on == config.ENGINE_COMPOSITE_REFERENCE_IMAGES[s.name], (variant, s.name)
            else:
                assert s.calibrated_on == ""
        assert not any("different digest" in r for r in t.release_ready())


@pytest.mark.parametrize("slot", ["blink", "reax1", "quyet"])
def test_a_digest_bump_clears_the_calibration(monkeypatch, slot):
    t = default_track(variant="v22")  # the only variant with all three slots
    assert t.slot(slot).calibrated
    monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, slot, "ghcr.io/reaxlabs/x@sha256:" + "b" * 64)
    out = t.release_ready()
    assert f"slot {slot} calibrated on a different digest" in out
    assert [r for r in out if "different digest" in r] == [f"slot {slot} calibrated on a different digest"]
    # the rebuilt track reads the same explicit calibration digests: the bump is still not calibrated
    assert default_track(variant="v22").slot(slot).calibrated_on != config.ENGINE_COMPOSITE_REFERENCE_IMAGES[slot]


def test_a_shared_digest_bump_names_every_slot_on_it(monkeypatch):
    for n in ("reax1", "pplx", "d3"):
        monkeypatch.setitem(config.ENGINE_COMPOSITE_REFERENCE_IMAGES, n, "ghcr.io/reaxlabs/x@sha256:" + "c" * 64)
    out = default_track(variant="v21").release_ready()
    assert sorted(out) == sorted(f"slot {n} calibrated on a different digest" for n in ("reax1", "pplx", "d3"))


def test_v20_names_exactly_vision_jeffan_mammoth_as_not_calibrated(monkeypatch):
    assert [r for r in default_track().release_ready() if "not calibrated" in r] == []  # all calibrated on 26 Oct
    _uncalibrated(monkeypatch, "vision", "jeffan", "mammoth")
    out = [r for r in default_track().release_ready() if "not calibrated" in r]
    assert sorted(out) == sorted(f"slot {n} is not calibrated" for n in ("vision", "jeffan", "mammoth"))


@pytest.mark.parametrize("slot", ["mammoth", "jeffan", "vision"])
def test_a_missing_calibration_digest_clears_the_new_slots_calibration(monkeypatch, slot):
    monkeypatch.delitem(config.ENGINE_COMPOSITE_CALIBRATED_ON, slot)  # marker True but no recorded digest
    assert default_track().release_ready() == [f"slot {slot} calibrated on a different digest"]
