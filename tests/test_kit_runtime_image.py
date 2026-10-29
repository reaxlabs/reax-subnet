"""Pool runtime images: release constant, validation, reax-kit and the selfcheck relationship."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reax_subnet import config as C
from reax_subnet import kitcli, selfcheck
from reax_subnet.engthresholds import THRESHOLDS
from reax_subnet.s1validator import POOL_PARAMS

DIGEST = "ghcr.io/example/engine@sha256:" + "a" * 64
GPUS = {"compute_capability": "12.0", "name_contains": "RTX PRO 6000"}
CHAMP = {"image": DIGEST, "source": "champion", "engine_uid": 7, "award": "board#1", "licence": "MIT", "gpus": GPUS}
PRO6000 = ["NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0"]
REF = {"image": None, "source": "reference", "engine_uid": None, "award": None, "licence": "Apache-2.0"}


def _images(**pool):
    return {"s1-fast": pool.get("fast", REF), "s1-pro": pool.get("pro", REF)}


def test_release_constant_is_valid_s1_fast_champion_uid62():
    C.validate_pool_runtime_images()
    assert {p: e["source"] for p, e in C.POOL_RUNTIME_IMAGES.items()} == {"s1-fast": "champion", "s1-pro": "reference"}
    e = C.POOL_RUNTIME_IMAGES["s1-fast"]
    assert e["engine_uid"] == 62 and e["image"].endswith("@sha256:8467e642812347a6c34f5c3f26eb7dd42678e4a4a532aad09f3d22d9fa0138a7")
    assert set(C.REFERENCE_RUNTIME_TAGS) == set(C.POOL_MODELS)
    # uid 62's native kernels are sm_120a only: champion on the measured RTX PRO 6000, reference everywhere else
    assert C.pool_runtime_image("s1-fast", {}, PRO6000) == e["image"]
    for other in (["NVIDIA H100 80GB HBM3, 9.0"], ["NVIDIA GeForce RTX 4090, 8.9"], ["NVIDIA GeForce RTX 5090, 12.0"], None, []):
        assert C.pool_runtime_image("s1-fast", {}, other) == C.REFERENCE_RUNTIME_TAGS["s1-fast"]


def test_champion_entry_valid():
    C.validate_pool_runtime_images(_images(fast=CHAMP))


@pytest.mark.parametrize("field,value", [("award", None), ("award", ""), ("licence", "GPL-3.0"), ("licence", None),
                                         ("engine_uid", None), ("image", None), ("gpus", None),
                                         ("gpus", {"compute_capability": "12.0"})])
def test_champion_refused_without_award_licence_uid_or_image(field, value):
    with pytest.raises(C.ConfigError):
        C.validate_pool_runtime_images(_images(fast={**CHAMP, field: value}))


@pytest.mark.parametrize("image", ["ghcr.io/example/engine:latest", "engine", "ghcr.io/x/e@sha256:abc",
                                   "ghcr.io/x/e@sha256:" + "A" * 64])
def test_non_digest_refs_refused(image):
    with pytest.raises(C.ConfigError, match="digest"):
        C.validate_pool_runtime_images(_images(fast={**CHAMP, "image": image}))
    with pytest.raises(C.ConfigError, match="digest"):
        C.validate_pool_runtime_images(_images(pro={**REF, "image": image}))


def test_unknown_source_and_missing_pool_refused():
    with pytest.raises(C.ConfigError):
        C.validate_pool_runtime_images(_images(fast={**REF, "source": "mine"}))
    with pytest.raises(C.ConfigError):
        C.validate_pool_runtime_images({"s1-fast": REF})


def test_override_reference_or_digest(monkeypatch):
    assert C.pool_runtime_image("s1-fast", {}) == C.REFERENCE_RUNTIME_TAGS["s1-fast"]
    assert C.pool_runtime_image("s1-fast", {"REAX_S1_FAST_RUNTIME_IMAGE": "reference"}) == C.REFERENCE_RUNTIME_TAGS["s1-fast"]
    assert C.pool_runtime_image("s1-pro", {"REAX_S1_PRO_RUNTIME_IMAGE": DIGEST}) == DIGEST
    with pytest.raises(C.ConfigError):
        C.pool_runtime_image("s1-fast", {"REAX_S1_FAST_RUNTIME_IMAGE": "evil:latest"})
    with pytest.raises(C.ConfigError):
        C.pool_runtime_image("s1-lite", {})


def test_champion_default_is_used_only_on_matching_gpus(monkeypatch):
    monkeypatch.setitem(C.POOL_RUNTIME_IMAGES, "s1-fast", CHAMP)
    assert C.pool_runtime_image("s1-fast", {}, PRO6000) == DIGEST
    assert C.pool_runtime_image("s1-fast", {}, PRO6000 * 2) == DIGEST
    ref = C.REFERENCE_RUNTIME_TAGS["s1-fast"]
    assert C.pool_runtime_image("s1-fast", {"REAX_S1_FAST_RUNTIME_IMAGE": "reference"}, PRO6000) == ref
    assert C.pool_runtime_image("s1-fast", {}, PRO6000 + ["NVIDIA L40S, 8.9"]) == ref  # mixed box: every GPU must match
    assert C.pool_runtime_image("s1-fast", {}, None) == ref


def test_kit_gpu_matches_parses_nvidia_smi_lines():
    assert C.kit_gpu_matches(GPUS, ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 12.0"])
    assert not C.kit_gpu_matches(GPUS, ["NVIDIA RTX PRO 6000 Blackwell, 10.0"])
    assert not C.kit_gpu_matches(GPUS, ["garbage"])
    assert not C.kit_gpu_matches(GPUS, ["NVIDIA RTX PRO 6000 Blackwell, [N/A]"])  # driver without compute_cap
    assert not C.kit_gpu_matches(GPUS, [])
    # an explicit override wins on any GPU (at the miner's own risk)
    assert C.pool_runtime_image("s1-fast", {"REAX_S1_FAST_RUNTIME_IMAGE": DIGEST}, ["NVIDIA L40S, 8.9"]) == DIGEST
    assert not C.kit_gpu_matches(GPUS, None)


def test_kit_cli(capsys, monkeypatch):
    monkeypatch.setattr(C, "local_gpus", lambda: None)  # a machine without a matching GPU -> reference
    monkeypatch.delenv("REAX_S1_FAST_RUNTIME_IMAGE", raising=False)
    monkeypatch.delenv("REAX_S1_PRO_RUNTIME_IMAGE", raising=False)
    assert kitcli.main(["image", "s1-fast"]) == 0
    assert capsys.readouterr().out.strip() == C.REFERENCE_RUNTIME_TAGS["s1-fast"]
    assert kitcli.main(["env"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [f"REAX_S1_FAST_RUNTIME_IMAGE={C.REFERENCE_RUNTIME_TAGS['s1-fast']}",
                   f"REAX_S1_PRO_RUNTIME_IMAGE={C.REFERENCE_RUNTIME_TAGS['s1-pro']}"]
    monkeypatch.setattr(C, "local_gpus", lambda: PRO6000)
    assert kitcli.main(["image", "s1-fast"]) == 0
    assert capsys.readouterr().out.strip() == C.POOL_RUNTIME_IMAGES["s1-fast"]["image"]
    assert kitcli.main(["show"]) == 0
    assert "else reference" in capsys.readouterr().out
    monkeypatch.setenv("REAX_S1_FAST_RUNTIME_IMAGE", "bad:tag")
    assert kitcli.main(["image", "s1-fast"]) == 2


def test_compose_default_matches_release_default():
    compose = (Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.s1.yml").read_text()
    m = re.search(r"image: \$\{REAX_S1_FAST_RUNTIME_IMAGE:-([^}]+)\}", compose)
    assert m and m.group(1) == C.pool_runtime_image("s1-fast", {})


def test_kit_default_is_what_selfcheck_documents():
    # selfcheck scores any image against the published reference answers of the pool, with the pool gates.
    for pool, entry in C.POOL_RUNTIME_IMAGES.items():
        assert selfcheck.load_reference(pool)["model"] == pool
        params = POOL_PARAMS[pool]
        # engine gates (engthresholds) are stricter than the pool gates (s1validator): a champion that passed them passes selfcheck
        assert THRESHOLDS["mean_tau"] < params.tau_mean
        assert THRESHOLDS["item_tau"] < params.tau_item
    assert "reference answers" in selfcheck.__doc__ and "reax-kit" in selfcheck.__doc__
    miner = (Path(__file__).resolve().parents[1] / "docs" / "MINER.md").read_text()
    assert "run the pinned runtime images unchanged" not in miner and "Which runtime image" in miner


# --- timed kit pins: the kit default follows the pool's manifest switch (manifest.POOL_MANIFEST_SWITCH) ---

from datetime import datetime, timezone  # noqa: E402

from reax_subnet import manifest  # noqa: E402

MAMMOTH = "ghcr.io/reaxlabs/reax-engine-ref-mammoth@sha256:3080a593060542bccbdaf7cf0c66593fc182d9753b3facd50206df86d93b8588"
BLINK = "ghcr.io/reaxlabs/reax-engine-ref-blink@sha256:ab02e141ffb5672dc68e137736be0a6fa15445c7cc824a51f6ad2a6664c0e10a"
SWITCH = datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc)


def test_timed_pins_are_the_published_reference_digests():
    C.validate_pool_runtime_images()
    assert C.POOL_RUNTIME_IMAGES_NEXT["s1-pro"]["image"] == MAMMOTH == C.ENGINE_COMPOSITE_REFERENCE_IMAGES["mammoth"]
    assert C.POOL_RUNTIME_IMAGES_NEXT["s1-fast"]["image"] == BLINK == C.ENGINE_COMPOSITE_REFERENCE_IMAGES["blink"]
    assert all(e["source"] == "reference" for e in C.POOL_RUNTIME_IMAGES_NEXT.values())
    with pytest.raises(C.ConfigError):
        C.validate_pool_runtime_images(next_images={"s1-lite": C.POOL_RUNTIME_IMAGES_NEXT["s1-pro"]})
    with pytest.raises(C.ConfigError):
        C.validate_pool_runtime_images(next_images={"s1-pro": {**C.POOL_RUNTIME_IMAGES_NEXT["s1-pro"], "image": None}})
    with pytest.raises(C.ConfigError):
        C.validate_pool_runtime_images(next_images={"s1-pro": {**C.POOL_RUNTIME_IMAGES_NEXT["s1-pro"], "image": "x:latest"}})


@pytest.mark.real_switch
def test_s1_pro_kit_pin_switches_with_the_validators():
    assert manifest.POOL_MANIFEST_SWITCH["s1-pro"].at == SWITCH == C.pool_runtime_switch_at("s1-pro")
    before, at = datetime(2026, 10, 30, 8, 59, 59, tzinfo=timezone.utc), SWITCH
    assert C.pool_runtime_image("s1-pro", {}, None, before) == C.REFERENCE_RUNTIME_TAGS["s1-pro"]
    assert C.pool_runtime_image("s1-pro", {}, None, at) == MAMMOTH
    # "reference" means the reference of the model that is current at that time, never the old Rune runtime
    assert C.pool_runtime_image("s1-pro", {"REAX_S1_PRO_RUNTIME_IMAGE": "reference"}, None, at) == MAMMOTH
    assert C.pool_runtime_image("s1-pro", {"REAX_S1_PRO_RUNTIME_IMAGE": DIGEST}, None, at) == DIGEST
    # the manifest the miner commits and the image the kit starts change at the same second
    assert manifest.manifest_name("s1-pro", before) != manifest.manifest_name("s1-pro", at)


@pytest.mark.real_switch
def test_s1_fast_blink_pin_stays_inert_until_the_release_arms_the_switch(monkeypatch):
    assert "s1-fast" not in manifest.POOL_MANIFEST_SWITCH and C.pool_runtime_switch_at("s1-fast") is None
    late = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert C.pool_runtime_image("s1-fast", {}, PRO6000, late) == C.POOL_RUNTIME_IMAGES["s1-fast"]["image"]
    sw = dict(manifest.POOL_MANIFEST_SWITCH)
    sw["s1-fast"] = manifest.PoolSwitch("s1-fast", datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc), "s1-fast-blink.json")
    monkeypatch.setattr(manifest, "POOL_MANIFEST_SWITCH", sw)
    # armed: the Plumb champion (uid 62) is dropped at the switch even on its own GPU, Blink reference everywhere
    assert C.pool_runtime_image("s1-fast", {}, PRO6000, late) == BLINK
    assert C.pool_runtime_image("s1-fast", {}, None, late) == BLINK
    assert C.pool_runtime_image("s1-fast", {"REAX_S1_FAST_RUNTIME_IMAGE": "reference"}, None, late) == BLINK


@pytest.mark.real_switch
def test_kit_cli_at(capsys, monkeypatch):
    monkeypatch.setattr(C, "local_gpus", lambda: None)
    monkeypatch.delenv("REAX_S1_FAST_RUNTIME_IMAGE", raising=False)
    monkeypatch.delenv("REAX_S1_PRO_RUNTIME_IMAGE", raising=False)
    assert kitcli.main(["env", "s1-pro", "--at", "2026-10-30T09:00Z"]) == 0
    assert capsys.readouterr().out.strip() == f"REAX_S1_PRO_RUNTIME_IMAGE={MAMMOTH}"
    assert kitcli.main(["env", "s1-pro", "--at", "2026-10-30T08:59Z"]) == 0
    assert capsys.readouterr().out.strip() == f"REAX_S1_PRO_RUNTIME_IMAGE={C.REFERENCE_RUNTIME_TAGS['s1-pro']}"
    assert kitcli.main(["show", "--at", "2026-10-29T00:00Z"]) == 0
    out = capsys.readouterr().out
    assert "from 2026-10-30T09:00Z" in out and MAMMOTH in out and ">= 48 GB" in out
