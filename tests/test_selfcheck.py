"""reax-selfcheck: the published reference answers and the pass/fail verdict against a runtime sidecar."""

from __future__ import annotations

import json
import math

import pytest
from aiohttp import web

from reax_subnet import selfcheck
from reax_subnet.s1contract import parse_decision
from reax_subnet.s1validator import POOL_PARAMS


def test_reference_matches_release_manifest_and_contract():
    ref = selfcheck.load_reference("s1-fast")
    assert len(ref["items"]) >= 500
    for it in ref["items"]:
        d = parse_decision({"id": it["id"], "state": it["state"], "questions": {"q": it["question"]}}, "s1-fast")
        assert len(d.question.keys) == len(it["p"])
        assert abs(math.fsum(it["p"]) - 1.0) < 1e-3


def test_simulated_rounds_follow_release_constants():
    params = POOL_PARAMS["s1-fast"]
    assert selfcheck.simulate_rounds([0.05] * 100, params, rounds=500) == (1.0, 0.0)
    assert selfcheck.simulate_rounds([0.3] * 100, params, rounds=500) == (0.0, 0.0)  # round lost, no strike
    assert selfcheck.simulate_rounds([0.6] * 100, params, rounds=500) == (0.0, 1.0)
    assert selfcheck.simulate_rounds([0.05] * 99 + [5.0], params, rounds=2000)[1] > 0  # one item over tau_item


async def _health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _fake_runtime(ref, noise: float):
    def key(state, question):
        return json.dumps([state, question], sort_keys=True)

    by_key = {key(it["state"], it["question"]): it for it in ref["items"]}

    async def decide(request: web.Request) -> web.Response:
        if request.headers.get("X-S1M-Internal-Token") != "t" * 32:
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        it = by_key[key(body["state"], body["questions"]["q"])]
        keys = list(parse_decision({"id": "x", "state": it["state"], "questions": {"q": it["question"]}},
                                   "s1-fast").question.keys)
        logits = [math.log(max(p, 1e-9)) + (noise if i == 0 else 0.0) for i, p in enumerate(it["p"])]
        z = math.fsum(math.exp(x) for x in logits)
        probs = {k: math.exp(x) / z for k, x in zip(keys, logits)}
        answer = {"noul": probs["true"]} if it["question"]["type"] == "noul" else {"probabilities": probs}
        return web.json_response({"answers": {"q": answer}})

    app = web.Application()
    app.router.add_post("/internal/v1/decide", decide)
    app.router.add_get("/health", _health)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.mark.parametrize("noise,expect", [(0.0, True), (0.6, False)])
async def test_selfcheck_against_runtime(noise, expect):
    ref = selfcheck.load_reference("s1-fast")
    runner, url = await _fake_runtime(ref, noise)
    try:
        report = await selfcheck.run("s1-fast", url, "t" * 32, limit=120, concurrency=8)
    finally:
        await runner.cleanup()
    assert report["answered"] == 120 and report["failed"] == 0
    assert report["pass"] is expect
    if expect:
        assert report["distance_max"] < 1e-3


async def test_selfcheck_reports_unreachable_runtime():
    with pytest.raises(ConnectionError):
        await selfcheck.run("s1-fast", "http://127.0.0.1:9", "t" * 32, limit=5, concurrency=1)


def test_cli_cannot_run_without_token(tmp_path, capsys):
    assert selfcheck.main(["--url", "http://127.0.0.1:9", "--token-file", str(tmp_path / "missing"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["pass"] is False


async def test_selfcheck_s1_pro_path(monkeypatch):
    """The s1-pro model uses the s1-pro pool constants and its own manifest/reference file (synthetic here)."""
    fast = selfcheck.load_reference("s1-fast")
    pro = dict(fast, model="s1-pro", manifest_id=selfcheck.load_manifest("s1-pro").manifest_id)
    monkeypatch.setattr(selfcheck, "load_reference", lambda model: pro if model == "s1-pro" else fast)
    runner, url = await _fake_runtime(fast, 0.0)
    try:
        report = await selfcheck.run("s1-pro", url, "t" * 32, limit=60, concurrency=4)
    finally:
        await runner.cleanup()
    assert report["model"] == "s1-pro" and report["manifest_id"] == pro["manifest_id"]
    assert report["answered"] == 60 and report["pass"] is True


def test_cli_s1_pro_without_reference_file_cannot_run(tmp_path, capsys, monkeypatch):
    def missing(model):
        raise ValueError(f"no reference answers for {model} manifest x in this release")

    monkeypatch.setattr(selfcheck, "load_reference", missing)
    tok = tmp_path / "tok"
    tok.write_text("t" * 32)
    assert selfcheck.main(["--model", "s1-pro", "--url", "http://127.0.0.1:9", "--token-file", str(tok), "--json"]) == 2


@pytest.mark.real_switch
def test_pool_params_follow_the_manifest_switch():
    from datetime import datetime, timezone

    from reax_subnet.manifest import POOL_MANIFEST_SWITCH
    from reax_subnet.s1validator import POOL_SWITCH_PARAMS, pool_params

    at = POOL_MANIFEST_SWITCH["s1-pro"].at
    before = datetime(at.year, at.month, at.day, tzinfo=timezone.utc).replace(hour=0)
    assert pool_params("s1-pro", before) is POOL_PARAMS["s1-pro"]
    assert pool_params("s1-pro", at) is POOL_SWITCH_PARAMS["s1-pro"]
    assert pool_params("s1-pro", at).temperature == 1.0
    assert pool_params("s1-fast", at) is POOL_PARAMS["s1-fast"]  # no switch armed for s1-fast


@pytest.mark.real_switch
def test_reference_files_parse_and_match_their_manifest():
    """Every shipped reference file belongs to one of the model's manifests (current or the timed next one)."""
    from importlib import resources

    from reax_subnet.manifest import POOL_MANIFEST_SWITCH, load_manifest

    for path in resources.files("reax_subnet.selfcheck_data").iterdir():
        if not path.name.endswith(".json"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        model = doc["model"]
        ids = {load_manifest(model).manifest_id}
        if model in POOL_MANIFEST_SWITCH:
            ids.add(load_manifest(model, name=POOL_MANIFEST_SWITCH[model].manifest_name).manifest_id)
        assert doc["schema"] == selfcheck.SCHEMA
        assert doc["manifest_id"] in ids, path.name
        assert path.name == f"{model}-{doc['manifest_id']}.json"
        for it in doc["items"]:
            d = parse_decision({"id": it["id"], "state": it["state"], "questions": {"q": it["question"]}}, model)
            assert len(d.question.keys) == len(it["p"])
            assert abs(math.fsum(it["p"]) - 1.0) < 1e-3


@pytest.mark.real_switch
def test_at_view_selects_the_switched_manifest_and_params(monkeypatch):
    from datetime import datetime, timezone

    from reax_subnet import manifest
    from reax_subnet.s1validator import POOL_SWITCH_PARAMS, pool_params

    sw = manifest.POOL_MANIFEST_SWITCH["s1-pro"]
    monkeypatch.setattr(manifest, "POOL_MANIFEST_SWITCH", dict(manifest.POOL_MANIFEST_SWITCH))
    selfcheck._view_switches_at(sw.at)
    nxt = manifest.load_manifest("s1-pro", name=sw.manifest_name).manifest_id
    assert selfcheck.load_reference("s1-pro")["manifest_id"] == nxt
    assert pool_params("s1-pro") is POOL_SWITCH_PARAMS["s1-pro"]
    monkeypatch.setattr(manifest, "POOL_MANIFEST_SWITCH", {"s1-pro": sw})
    selfcheck._view_switches_at(datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert manifest.POOL_MANIFEST_SWITCH == {}
    assert selfcheck.load_reference("s1-pro")["manifest_id"] != nxt
