"""Award-time DI-proxy check: paired diff, SE, refusals, determinism, weighting."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("award_di_proxy", Path(__file__).resolve().parents[1] / "tools" / "award_di_proxy.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def items(scores, unit_of=lambda i: "a" if i % 2 == 0 else "b", protocol="native"):
    return {f"i{i}": {"stratum": unit_of(i), "score": float(s), "protocol": protocol} for i, s in enumerate(scores)}


def write(path, d, field="unit"):
    path.write_text("\n".join(json.dumps({"id": k, field: v["stratum"], "weight_group": "g", "score": v["score"],
                                          "protocol": v["protocol"]}) for k, v in d.items()) + "\n")


def test_identical_inputs_zero_diff():
    c = items([1, 0, 1, 1, 0, 1, 0, 0])
    r = P.compute(c, dict(c))
    assert r["diff"] == 0 and r["se"] == 0 and r["score"] == r["base_score"]


def test_known_shift_exact():
    base = items([0.5] * 10)
    cand = items([0.6] * 10)
    r = P.compute(cand, base)
    assert r["diff"] == 10.0 and r["score"] == 60.0 and r["base_score"] == 50.0 and r["items"] == 10


def test_se_matches_analytic_on_binary():
    rng = np.random.default_rng(1)
    b = rng.integers(0, 2, 400)
    c = np.where(rng.random(400) < 0.3, 1 - b, b)
    cand, base = items(c, lambda i: "a"), items(b, lambda i: "a")
    d = (c - b).astype(float)
    expect = 100 * math.sqrt(d.var(ddof=1) / 400)
    r = P.compute(cand, base)
    assert r["se"] == pytest.approx(expect, abs=0.01)
    boot = 100 * P.bootstrap_se(["a"] * 400, d, {"a": 1.0})
    assert boot == pytest.approx(expect, rel=0.1)
    assert r["se"] >= round(min(expect, boot), 4)


def test_se_is_larger_of_two(monkeypatch):
    base, cand = items([0, 1, 0, 1]), items([1, 1, 0, 0])
    monkeypatch.setattr(P, "bootstrap_se", lambda *a, **k: 0.9)
    assert P.compute(cand, base)["se"] == 90.0
    monkeypatch.setattr(P, "bootstrap_se", lambda *a, **k: 0.0)
    assert P.compute(cand, base)["se"] > 0


def test_deterministic_with_seed_and_seed_matters():
    rng = np.random.default_rng(2)
    base = items(rng.random(60))
    cand = items(np.clip(rng.random(60), 0, 1))
    assert P.compute(cand, base, seed=7) == P.compute(cand, base, seed=7)
    d = np.array([cand[k]["score"] - base[k]["score"] for k in sorted(cand)])
    s = [cand[k]["stratum"] for k in sorted(cand)]
    assert P.bootstrap_se(s, d, {"a": .5, "b": .5}, seed=1) != P.bootstrap_se(s, d, {"a": .5, "b": .5}, seed=2)


def test_weighting():
    # unit a: +0.2 on every item, unit b: no change
    base = items([0.5] * 8)
    cand = items([0.7 if i % 2 == 0 else 0.5 for i in range(8)])
    assert P.compute(cand, base)["diff"] == 10.0  # equal weights
    assert P.compute(cand, base, weights={"a": 0.6, "b": 0.4})["diff"] == 12.0
    assert P.compute(cand, base, weights={"a": 6, "b": 4})["diff"] == 12.0  # normalised
    assert P.compute(cand, base, weights={"a": 1, "b": 1})["diff"] == 10.0
    with pytest.raises(P.ProxyError):
        P.compute(cand, base, weights={"a": 1})
    with pytest.raises(P.ProxyError):
        P.compute(cand, base, weights={"a": 1, "b": 0})


def test_weighted_se_uses_same_weights():
    base = items([0.5] * 8)
    cand = items([0.9, 0.5, 0.1, 0.5, 0.7, 0.5, 0.3, 0.5])
    w = {"a": 0.9, "b": 0.1}
    d = np.array([cand[k]["score"] - base[k]["score"] for k in sorted(cand)])
    s = [cand[k]["stratum"] for k in sorted(cand)]
    assert P.analytic_se(s, d, w) == pytest.approx(math.sqrt(0.81 * np.var([.4, -.4, .2, -.2], ddof=1) / 4))


@pytest.mark.parametrize("mutate", ["missing", "extra", "unit", "protocol"])
def test_pairing_refusals(mutate):
    base = items([1, 0, 1, 0])
    cand = dict(base)
    if mutate == "missing":
        cand.pop("i3")
    elif mutate == "extra":
        cand["i9"] = dict(base["i0"])
    elif mutate == "unit":
        cand["i0"] = dict(base["i0"], stratum="b")
    else:
        cand["i0"] = dict(base["i0"], protocol="other")
    with pytest.raises(P.ProxyError):
        P.compute(cand, base)


def test_loader_refusals(tmp_path):
    f = tmp_path / "x.jsonl"
    row = {"id": "a", "unit": "u", "score": 1, "protocol": "p"}
    for bad in [dict(row, score=float("nan")), dict(row, score=1.5), dict(row, score="x"), dict(row, id=""),
                {k: v for k, v in row.items() if k != "unit"}, {k: v for k, v in row.items() if k != "protocol"}]:
        f.write_text(json.dumps(bad) + "\n")
        with pytest.raises(P.ProxyError):
            P.load_items(f)
    f.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(P.ProxyError, match="duplicate"):
        P.load_items(f)
    f.write_text("")
    with pytest.raises(P.ProxyError):
        P.load_items(f)
    f.write_text(json.dumps({"id": "a", "area": "u", "score": 0.5, "protocol": "p"}) + "\n")
    assert P.load_items(f)["a"]["stratum"] == "u"


def test_unit_with_one_item_refused():
    with pytest.raises(P.ProxyError, match="at least 2"):
        P.compute(items([1, 0, 1], lambda i: "a" if i < 2 else "b"), items([0, 0, 0], lambda i: "a" if i < 2 else "b"))


IDS8 = "\n".join(json.dumps({"id": f"i{i}"}) for i in range(8)) + "\n"


def run_cli(tmp_path, capsys, sha=None, eval_text=IDS8, cand_items=None, extra=("--min-items", "1"), weights=True):
    ev = tmp_path / "eval.jsonl"
    ev.write_text(eval_text)
    base = items([0, 1, 0, 1, 1, 0, 1, 0])
    cand = cand_items or items([1, 1, 0, 1, 1, 1, 1, 0])
    write(tmp_path / "c.jsonl", cand)
    write(tmp_path / "b.jsonl", base)
    argv = ["--candidate", str(tmp_path / "c.jsonl"), "--base", str(tmp_path / "b.jsonl"), "--eval-file", str(ev),
            "--eval-id", "di-proxy-v1", "--eval-sha", sha or hashlib.sha256(eval_text.encode()).hexdigest(),
            "--base-id", "reax-1", "--candidate-id", "cand", *extra]
    if weights:
        (tmp_path / "w.json").write_text(json.dumps({"a": 1, "b": 1}))
        argv += ["--weights", str(tmp_path / "w.json")]
    rc = P.main(argv)
    return rc, capsys.readouterr()


def test_cli_block_and_summary(tmp_path, capsys):
    rc, out = run_cli(tmp_path, capsys)
    assert rc == 0
    block = json.loads(out.out)
    assert list(sorted(block)) == sorted(["done", "eval_id", "eval_sha", "protocol", "base_id", "items", "score",
                                          "base_score", "diff", "se"])
    assert block["done"] is True and block["items"] == 8 and block["base_id"] == "reax-1" and block["protocol"] == "native"
    assert block["diff"] == 25.0 and "lower90" in out.err


def test_cli_eval_sha_mismatch(tmp_path, capsys):
    rc, out = run_cli(tmp_path, capsys, sha="0" * 64)
    assert rc == 2 and out.out == "" and "sha mismatch" in out.err


def test_cli_eval_ids_must_match(tmp_path, capsys):
    text = "\n".join(json.dumps({"id": f"i{i}"}) for i in range(7)) + "\n"
    rc, out = run_cli(tmp_path, capsys, eval_text=text)
    assert rc == 2 and "frozen eval" in out.err
    assert run_cli(tmp_path, capsys, eval_text=IDS8)[0] == 0


def test_cli_refuses_silent_defaults(tmp_path, capsys):
    # critic F6: an eval file without ids, too few items, or no weights file are refused, never silently accepted
    rc, out = run_cli(tmp_path, capsys, eval_text="frozen\n")
    assert rc == 2 and "cannot be bound" in out.err
    rc, out = run_cli(tmp_path, capsys, extra=())
    assert rc == 2 and "needs >= 1000" in out.err
    with pytest.raises(SystemExit):
        run_cli(tmp_path, capsys, weights=False)


def test_mixed_protocol_label_passes_the_signer_id_pattern():
    # reaxsign autopolicy accepts ids matching [A-Za-z0-9._:@/-]{1,200} only (lead review: no commas)
    import re
    c = {"i0": {"stratum": "a", "score": 1.0, "protocol": "reax"}, "i1": {"stratum": "a", "score": 0.0, "protocol": "reax"},
         "i2": {"stratum": "b", "score": 1.0, "protocol": "joint-v3"}, "i3": {"stratum": "b", "score": 1.0, "protocol": "joint-v3"}}
    r = P.compute(c, dict(c))
    assert r["protocol"] == "mixed:joint-v3/reax"
    assert re.fullmatch(r"[A-Za-z0-9._:@/-]{1,200}", r["protocol"])
