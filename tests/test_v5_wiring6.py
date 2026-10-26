"""Per-pool-version area weights (``<pool dir>/v51.config.json``, spec/v5wire.py): absent = defaults bit for bit, 0-weight areas
are skipped by the draw, a bad or readable-by-others config means no draw, scoring reads the same weights."""
from __future__ import annotations

import json
import os
from collections import Counter

import pytest
from test_v5_wiring import _answers, pub_rows, v51_rows, write_v51

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import v5, v5wire as W

THREE = {"knowledge": 0.2585, "language": 0.0, "retrieval": 0.2002, "tools": 0.1828, "arts": 0.0}


def write_config(d, obj=None, mode=0o600, raw=None):
    p = os.path.join(d, W.CONFIG_FILE)
    with open(p, "w") as fh:
        fh.write(raw if raw is not None else json.dumps(obj if obj is not None else {"pool_version": "v0.1", "hf_areas": THREE}))
    os.chmod(p, mode)
    return p


@pytest.fixture(scope="module")
def three_area_rows():
    t, n = v51_rows()
    keep = lambda r: r["area"] not in ("language", "arts")      # noqa: E731
    return [r for r in t if keep(r)], [r for r in n if keep(r)]


def test_absent_config_is_the_default(tmp_path, three_area_rows):
    t, n = v51_rows()
    d = write_v51(tmp_path / "p", t, n)
    assert W.load_config(d) is None and W.load_config("") is None
    assert W.params(pool_dir=d) == W.params() and dict(W.params(pool_dir=d).hf_areas) == v5.HF_AREAS
    pool = W.load_pool(d)
    pub = W.public_rows(pub_rows())
    assert W.draw(pool, pub, seed=7, use_counts={}, p=W.params(pool_dir=d)) == W.draw(pool, pub, seed=7, use_counts={})


def test_zero_weight_areas_draw_540_t_over_three_areas(tmp_path, three_area_rows, caplog):
    t, n = three_area_rows
    d = write_v51(tmp_path / "p", t, n)
    write_config(d)
    caplog.set_level("INFO", logger="reax.spec")
    p = W.params(pool_dir=d)
    assert dict(p.hf_areas) == THREE and p.t_items == 540
    with pytest.raises(S.SampleError, match="language"):                       # default weights: the draw fails closed
        W.draw(W.load_pool(d), W.public_rows(pub_rows()), seed=7, use_counts={})
    got, fresh = W.draw(W.load_pool(d), W.public_rows(pub_rows()), seed=7, use_counts={}, p=p)
    typed = [r for r in got if r["tier"] == "mirror"]
    t_area = Counter(r["area"] for r in typed if r["part"] == "T")
    assert sum(t_area.values()) == 540 and set(t_area) == {"knowledge", "retrieval", "tools"}
    assert dict(t_area) == S._quotas(540, {a: w for a, w in THREE.items() if w > 0})      # renormalised quotas
    assert Counter(r["part"] for r in typed) == {"T": 540, "N": 360}
    W.params(pool_dir=d)
    assert sum("pool_version=v0.1" in r.getMessage() for r in caplog.records) == 1        # logged once per load


@pytest.mark.parametrize("obj", [
    {"hf_areas": THREE},                                                           # no pool_version
    {"pool_version": "", "hf_areas": THREE},
    {"pool_version": "v0.1", "hf_areas": {k: v for k, v in THREE.items() if k != "arts"}},      # four keys
    {"pool_version": "v0.1", "hf_areas": {**THREE, "chess": 1}},
    {"pool_version": "v0.1", "hf_areas": {**THREE, "tools": -0.1}},
    {"pool_version": "v0.1", "hf_areas": {**THREE, "tools": "0.2"}},
    {"pool_version": "v0.1", "hf_areas": {**THREE, "tools": True}},
    {"pool_version": "v0.1", "hf_areas": {k: 0 for k in THREE}},                   # sum 0
    {"pool_version": "v0.1", "hf_areas": THREE, "extra": 1},
    [1, 2],
])
def test_malformed_config_fails_closed(tmp_path, three_area_rows, obj, caplog):
    d = write_v51(tmp_path / "p", *three_area_rows)
    write_config(d, obj)
    with pytest.raises(S.SampleError, match="invalid"):
        W.params(pool_dir=d)
    assert "refused" in caplog.text


def test_unparseable_and_world_readable_config_fails_closed(tmp_path, three_area_rows):
    d = write_v51(tmp_path / "p", *three_area_rows)
    write_config(d, raw="{not json")
    with pytest.raises(S.SampleError, match="invalid"):
        W.params(pool_dir=d)
    for mode in (0o644, 0o640, 0o604):
        write_config(d, mode=mode)
        with pytest.raises(S.SampleError, match="0600"):
            W.params(pool_dir=d)


def test_evaluator_params_read_the_config_and_no_draw_when_bad(tmp_path, three_area_rows):
    d = write_v51(tmp_path / "p", *three_area_rows)
    write_config(d)
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    ev.cfg = C.from_env({"REAX_SPEC_SEALED_DIR": "/pool", "REAX_SPEC_V5_POOL_DIR": d})
    ev._v5_config = None                                       # what evaluate_cycle resets at the start of every cycle
    assert dict(ev._v5_params().hf_areas) == THREE and ev._v5_params().panel is v5.PANEL_STATS["v0.1"]
    write_config(d, mode=0o644)
    assert dict(ev._v5_params().hf_areas) == THREE             # read once per cycle (round2 #24356): the cycle keeps what it read
    ev._v5_config = None                                       # the next cycle reads the file again and refuses it
    with pytest.raises(S.SampleError):
        ev._v5_params()


def test_scoring_uses_the_same_weights(tmp_path, three_area_rows):
    t, n = v51_rows()
    d = write_v51(tmp_path / "p", t, n)          # a pool of all five areas: the weights decide what language/arts count for
    write_config(d)
    pc = W.params(pool_dir=d, resamples=40)
    items, _ = W.draw(W.load_pool(d), W.public_rows(pub_rows()), seed=3, use_counts={}, p=W.params(resamples=40))
    good = _answers(items, .95, "m")
    bad = _answers(items, .3, "m")
    # a model strong on knowledge only: its score depends on how knowledge is weighted against retrieval and tools
    resp = {r["id"]: (good if r.get("area") == "knowledge" else bad)[r["id"]] for r in items}
    base = _answers(items, .5, "base")
    cfg_s = v5.score_v5_detail(resp, items, base, pc, 1).s
    default_s = v5.score_v5_detail(resp, items, base, W.params(resamples=40), 1).s
    explicit_s = v5.score_v5_detail(resp, items, base, v5.V5Params(resamples=40, pub_items=90, hf_areas=THREE), 1).s
    assert cfg_s == explicit_s and cfg_s != default_s
    ctx = W.build_context("d", items, {"m": resp, "b": base}, {"m": "b"}, [], 1, pc)
    assert ctx.parts["m"].s == cfg_s and ctx.params.hf_areas == pc.hf_areas
