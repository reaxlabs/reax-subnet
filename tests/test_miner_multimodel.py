"""One hotkey, several models: the multi-model miner gateway and the commitment-merging tools
(docs/DESIGN-MULTIROLE.md sections 1 and 4)."""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import auth
from reax_subnet import elements as E
from reax_subnet import commitcli
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.config import ConfigError, load_miner
from reax_subnet.manifest import load_manifest
from reax_subnet.miner import CommitmentPublisher, S1Miner, multimodel_refusal
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1contract import parse_decision
from reax_subnet.s1items import generate_s1_items
from reax_subnet.s1protocol import S1_DECIDE_PATH, S1Request, parse_s1_response

MID = {m: load_manifest(m).manifest_id for m in E.MODEL_ORDER}
FAST = f"reax-s1/1:s1-fast:{MID['s1-fast']}"
PRO = f"reax-s1/1:s1-pro:{MID['s1-pro']}"
FAST_PRO = E.inference_text({m: MID[m] for m in ("s1-fast", "s1-pro")})
SPEC = "reax-spec/1:demand_forecasting:alice/df-model@" + "a" * 40
ENG = "reax-eng/1:s1-fast:ghcr.io/alice/engine@sha256:" + "c" * 64
ENG2 = "reax-eng/1:s1-fast:ghcr.io/alice/engine@sha256:" + "d" * 64


class Slow(S1HashBackend):
    delay = 0.4

    async def decide(self, decisions):
        await asyncio.sleep(self.delay)
        return await super().decide(decisions)


@pytest.fixture
def started(monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 100)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 100)


# ---------------------------------------------------------------------------------------------------- the gateway
def _body(model, manifest_id=None, n=2, rid="r1"):
    ds = tuple(parse_decision(i.decision(), model) for i in generate_s1_items(n, seed=5))
    return S1Request(rid, model, manifest_id or MID[model], ds).to_bytes()


class Env:
    def __init__(self, miner, url, vw, mw, session):
        self.miner, self.url, self.vw, self.mw, self.session = miner, url, vw, mw, session

    async def post(self, body):
        h = auth.sign_request(self.vw, body, self.mw.hotkey.ss58_address, path=S1_DECIDE_PATH)
        h["Content-Type"] = "application/json"
        async with self.session.post(self.url + S1_DECIDE_PATH, data=body, headers=h) as r:
            return r.status, await r.read()


async def _env(make_wallet, backends, **kw):
    mw, vw = make_wallet("//MM"), make_wallet("//MV")
    miner = S1Miner(mw, backends, **kw)
    miner.set_validators(frozenset({vw.hotkey.ss58_address}))
    runner = web.AppRunner(miner.app(), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    session = aiohttp.ClientSession()
    return Env(miner, url, vw, mw, session), runner, session


@pytest.fixture
async def multi(make_wallet):
    backends = {"s1-fast": S1HashBackend("s1-fast"), "s1-pro": S1HashBackend("s1-pro")}
    env, runner, session = await _env(make_wallet, backends, max_concurrent=2, per_caller_concurrency=1,
                                      per_caller_per_minute=600)
    env.backends = backends
    yield env
    await session.close()
    await runner.cleanup()


async def test_requests_are_routed_by_model(multi):
    for model in ("s1-fast", "s1-pro", "s1-fast"):
        status, raw = await multi.post(_body(model))
        assert status == 200
        assert len(parse_s1_response(raw, parse_s1_request_of(_body(model)))) == 2
    assert multi.backends["s1-fast"].calls == 4 and multi.backends["s1-pro"].calls == 2


def parse_s1_request_of(body):
    from reax_subnet.s1protocol import parse_s1_request

    return parse_s1_request(body)


async def test_unserved_model_and_wrong_manifest_are_400(multi):
    assert (await multi.post(_body("s1-lite")))[0] == 400
    assert (await multi.post(_body("s1-fast", manifest_id=MID["s1-pro"])))[0] == 400  # pro's id on the fast lane
    assert (await multi.post(b'{"protocol":"reax-s1/1"}'))[0] == 400
    assert (await multi.post(b"[1]"))[0] == 400
    assert multi.backends["s1-fast"].calls == multi.backends["s1-pro"].calls == 0
    # refusals use up no model's limits
    assert (await multi.post(_body("s1-fast")))[0] == 200


async def test_limits_are_per_model(make_wallet):
    backends = {"s1-fast": Slow("s1-fast"), "s1-pro": S1HashBackend("s1-pro")}
    env, runner, session = await _env(make_wallet, backends, max_concurrent=1, per_caller_concurrency=1,
                                      per_caller_per_minute=600)
    try:
        # model A (fast): two at once from one validator -> the second is refused; model B (pro) is not affected
        a = [asyncio.ensure_future(env.post(_body("s1-fast", rid=f"a{i}"))) for i in range(2)]
        await asyncio.sleep(0.1)
        status_b, _ = await env.post(_body("s1-pro", rid="b1"))
        statuses_a = sorted(s for s, _ in await asyncio.gather(*a))
        assert status_b == 200 and statuses_a == [200, 429]
    finally:
        await session.close()
        await runner.cleanup()


async def test_per_minute_window_is_per_model(make_wallet):
    backends = {"s1-fast": S1HashBackend("s1-fast"), "s1-pro": S1HashBackend("s1-pro")}
    env, runner, session = await _env(make_wallet, backends, max_concurrent=4, per_caller_concurrency=4,
                                      per_caller_per_minute=2, model_limits={"s1-pro": (4, 4, 5)})
    try:
        assert [(await env.post(_body("s1-fast", rid=f"f{i}")))[0] for i in range(3)] == [200, 200, 429]
        assert [(await env.post(_body("s1-pro", rid=f"p{i}")))[0] for i in range(6)] == [200] * 5 + [429]
    finally:
        await session.close()
        await runner.cleanup()


async def test_runtime_slots_are_per_model(make_wallet):
    """A saturated fast lane (1 slot, long answer) does not delay the pro lane's runtime slot."""
    backends = {"s1-fast": Slow("s1-fast"), "s1-pro": S1HashBackend("s1-pro")}
    env, runner, session = await _env(make_wallet, backends, max_concurrent=1, per_caller_concurrency=8,
                                      per_caller_per_minute=600)
    try:
        slow = asyncio.ensure_future(env.post(_body("s1-fast", rid="s1")))
        await asyncio.sleep(0.05)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        assert (await env.post(_body("s1-pro", rid="p1")))[0] == 200
        assert loop.time() - t0 < Slow.delay  # did not wait for the fast lane's slot
        assert (await slow)[0] == 200
        assert env.miner._lanes["s1-fast"]._slots is not env.miner._lanes["s1-pro"]._slots
    finally:
        await session.close()
        await runner.cleanup()


async def test_health_lists_all_models(multi):
    async with multi.session.get(multi.url + "/health") as r:
        info = await r.json()
    assert info["models"] == {"s1-fast": MID["s1-fast"], "s1-pro": MID["s1-pro"]}
    assert "model" not in info and "manifest_id" not in info


async def test_single_model_health_and_limits_are_as_before(make_wallet):
    env, runner, session = await _env(make_wallet, S1HashBackend("s1-fast"), max_concurrent=4,
                                      per_caller_concurrency=2)
    try:
        info = env.miner.health_info()
        assert info["model"] == "s1-fast" and info["manifest_id"] == MID["s1-fast"]
        assert info["models"] == {"s1-fast": MID["s1-fast"]}
        assert list(env.miner._lanes) == ["s1-fast"] and env.miner._lanes["s1-fast"] is env.miner
        assert (env.miner._max_pending_ip, env.miner._max_pending_total) == (8, 4096)
        assert env.miner._caller_limits() == (2, 20)
        assert (await env.post(_body("s1-pro")))[0] == 400  # not served: the old message names the model
        assert (await env.post(_body("s1-fast")))[0] == 200
    finally:
        await session.close()
        await runner.cleanup()


def test_pending_body_limits_per_ip_scale_with_the_models_the_total_does_not(make_wallet):
    w = make_wallet("//MM")
    two = S1Miner(w, {"s1-fast": S1HashBackend("s1-fast"), "s1-pro": S1HashBackend("s1-pro")},
                  max_pending_per_ip=32, max_pending_total=1000)
    assert (two._max_pending_ip, two._max_pending_total) == (64, 1000)  # critic round 1 F6


def test_capacity_scale_is_per_model(make_wallet):
    class Pool:
        model = "s1-pro"
        size, healthy_count, nominal_size = 2, 1, 2

        class manifest:
            manifest_id = MID["s1-pro"]

    m = S1Miner(make_wallet("//MM"), {"s1-fast": S1HashBackend("s1-fast"), "s1-pro": Pool()},
                per_caller_concurrency=8, per_caller_per_minute=100,
                model_limits={"s1-pro": (4, 8, 100)})
    assert m._caller_limits(m._lanes["s1-fast"]) == (8, 100)  # fast backend healthy
    assert m._caller_limits(m._lanes["s1-pro"]) == (4, 50)  # pro lost half of its workers: only its limits shrink


# ---------------------------------------------------------------------------------------------------------- config
def _env_vars(tmp_path, **extra):
    token = tmp_path / "token"
    token.write_text("t" * 32)
    return {"REAX_NETUID": "7", "REAX_RUNTIME_TOKEN_FILE": str(token), **extra}


def test_models_default_to_the_single_model(tmp_path):
    cfg = load_miner(_env_vars(tmp_path, REAX_MODEL="s1-pro", REAX_S1_PRO_RUNTIME_URL="http://127.0.0.1:8102"))
    assert [m.model for m in cfg.served] == ["s1-pro"] and cfg.common.model == "s1-pro"
    assert cfg.served[0].max_concurrent == cfg.max_concurrent
    assert cfg.served[0].per_caller_per_minute == cfg.per_caller_per_minute


def test_models_list(tmp_path):
    env = _env_vars(tmp_path, REAX_MODELS="s1-fast, s1-pro,s1-lite",
                    REAX_S1_FAST_RUNTIME_URL="http://127.0.0.1:8101", REAX_S1_PRO_RUNTIME_URL="http://127.0.0.1:8102",
                    REAX_S1_LITE_RUNTIME_URL="http://127.0.0.1:8103")
    cfg = load_miner(env)
    assert [m.model for m in cfg.served] == ["s1-fast", "s1-pro", "s1-lite"] and cfg.common.model == "s1-fast"
    # the first model's numbers are the scalar fields (a single-model config reads the same)
    assert cfg.served[0].max_concurrent == cfg.max_concurrent
    assert cfg.common.endpoints("s1-lite") == (("http://127.0.0.1:8103", ""),)
    assert load_miner({**env, "REAX_MODEL": "s1-pro"}).common.model == "s1-pro"  # REAX_MODEL names the primary


@pytest.mark.parametrize("extra,msg", [
    ({"REAX_MODELS": "s1-fast,s1-fast"}, "twice"),
    ({"REAX_MODELS": "s1-fast,s1-huge"}, "unknown"),
    ({"REAX_MODELS": "s1-fast,,s1-pro"}, "unknown"),
    ({"REAX_MODELS": "s1-fast,s1-pro", "REAX_MODEL": "s1-lite"}, "not in REAX_MODELS"),
    ({"REAX_MODELS": "s1-fast,s1-pro", "REAX_MECHANISM": "mpm1"}, "REAX_MECHANISM=s1"),
    ({"REAX_MODELS": "s1-fast,s1-pro", "REAX_S1_PRO_RUNTIME_URL": "http://127.0.0.1:8101"}, "share a runtime URL"),
])
def test_models_list_errors(tmp_path, extra, msg):
    env = _env_vars(tmp_path, REAX_S1_FAST_RUNTIME_URL="http://127.0.0.1:8101",
                    REAX_S1_PRO_RUNTIME_URL="http://127.0.0.1:8102")
    with pytest.raises(ConfigError, match=msg):
        load_miner({**env, **extra})


def _workers_doc(model, key_file):
    return {"model": model, "workers": [{"name": f"{model}-a", "url": "http://10.8.0.2:8190", "key_file": key_file}]}


def test_multinode_has_a_workers_file_per_model(tmp_path):
    key = tmp_path / "k.key"
    key.write_text("ab" * 32)
    files = {}
    for m in ("s1-fast", "s1-pro"):
        f = tmp_path / f"{m}.json"
        f.write_text(json.dumps(_workers_doc(m, str(key))))
        files[m] = f
    env = _env_vars(tmp_path, REAX_FEATURE_MULTINODE="1", REAX_MODELS="s1-fast,s1-pro,s1-lite",
                    REAX_S1_FAST_WORKERS_FILE=str(files["s1-fast"]), REAX_S1_PRO_WORKERS_FILE=str(files["s1-pro"]),
                    REAX_S1_LITE_RUNTIME_URL="http://127.0.0.1:8103")
    cfg = load_miner(env)
    by = {m.model: m for m in cfg.served}
    assert [w.name for w in by["s1-fast"].workers] == ["s1-fast-a"] and by["s1-fast"].workers_file == str(files["s1-fast"])
    assert [w.name for w in by["s1-pro"].workers] == ["s1-pro-a"]
    assert by["s1-lite"].workers == () and by["s1-lite"].workers_file == ""  # lite keeps one machine
    assert cfg.workers_file == str(files["s1-fast"])  # the single-model fields follow the first model
    with pytest.raises(ConfigError, match="REAX_S1_PRO_WORKERS_FILE"):
        load_miner({k: v for k, v in env.items() if k != "REAX_S1_PRO_WORKERS_FILE"})
    # several models do not fall back to the shared REAX_WORKERS_FILE (it names one model's workers)
    with pytest.raises(ConfigError, match="REAX_S1_PRO_WORKERS_FILE"):
        load_miner({**{k: v for k, v in env.items() if k != "REAX_S1_PRO_WORKERS_FILE"},
                    "REAX_WORKERS_FILE": str(files["s1-fast"])})
    # the single-model variable keeps working
    single = load_miner(_env_vars(tmp_path, REAX_FEATURE_MULTINODE="1", REAX_WORKERS_FILE=str(files["s1-fast"])))
    assert [w.name for w in single.workers] == ["s1-fast-a"]


# ---------------------------------------------------------------------------------------------- publishing
class Chain:
    def __init__(self):
        self.writes = []

    def set_commitment_fields(self, fields):
        self.writes.append(list(fields))


def _publisher(models=("s1-fast",), force=False):
    chain = Chain()
    return chain, CommitmentPublisher(chain, {m: MID[m] for m in models}, force=force, hotkey=None)


def test_publisher_writes_the_inference_element():
    chain, p = _publisher()
    assert p.plan([], 50) == [FAST]
    assert p.plan(["reax-s1/1:s1-fast:0123456789abcdef"], 50) == [FAST]  # stale manifest -> replaced alone
    assert p.plan([FAST], 50) is None  # nothing to do
    assert p.plan([f" {FAST} "], 50) is None


def test_publisher_keeps_a_competition_field(started):
    chain, p = _publisher()
    old = "reax-s1/1:s1-fast:0123456789abcdef"
    assert p.plan([old, SPEC], 150) == [FAST, SPEC]
    assert p.plan([SPEC, old], 150) == [FAST, SPEC]
    assert p.plan([SPEC], 150) == [FAST, SPEC]  # an entrant that did not serve yet gains its inference element


def test_publisher_ignores_a_competition_difference(started):
    chain, p = _publisher()
    assert p.plan([FAST, SPEC], 150) is None
    assert p.plan([FAST, ENG], 150) is None
    assert p.plan([ENG, FAST], 150) is None


def test_publisher_cleans_a_commitment_that_voids_itself(started):
    chain, p = _publisher()
    assert p.plan([FAST, FAST], 150) == [FAST]  # two inference fields = invalid
    assert p.plan([FAST, "junk"], 150) == [FAST]


def test_dual_role_write_is_refused_before_the_start_block(caplog, monkeypatch):
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", None)  # held (the committed default; the release sets a block)
    chain, p = _publisher()
    stale = "reax-s1/1:s1-fast:0123456789abcdef"
    with caplog.at_level("ERROR"):
        assert p.plan([stale, SPEC], 10**9) is None
    assert "competition" in caplog.text and "NOT publishing" in caplog.text
    assert p.plan([SPEC], 10**9) is None  # an entrant-only commitment is not overwritten either
    assert p.plan([stale], 10**9) == [FAST]  # plain single-field rewrite stays possible


def test_dual_role_write_before_start_block_with_a_start_in_the_future(monkeypatch):
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 1000)
    _, p = _publisher()
    assert p.plan(["reax-s1/1:s1-fast:0123456789abcdef", SPEC], 999) is None
    assert p.plan(["reax-s1/1:s1-fast:0123456789abcdef", SPEC], 1000) == [FAST, SPEC]


def test_force_writes_both_before_the_start():
    _, p = _publisher(force=True)
    assert p.plan(["reax-s1/1:s1-fast:0123456789abcdef", SPEC], 10) == [FAST, SPEC]


def test_several_models_refused_while_held_or_before_the_start(monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", None)  # held (the committed default; the release sets a block)
    assert "MULTIMODEL_START_BLOCK is held" in multimodel_refusal(2, None, False)
    assert multimodel_refusal(2, None, True) == "" and multimodel_refusal(1, 5, False) == ""
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", 500)
    assert multimodel_refusal(3, None, False) == ""  # block unknown yet
    assert "count from block 500" in multimodel_refusal(3, 499, False)
    assert multimodel_refusal(3, 500, False) == ""


def test_multimodel_element_is_not_written_before_the_start(monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", None)  # held (the committed default; the release sets a block)
    _, p = _publisher(("s1-fast", "s1-pro"))
    assert p.wanted == FAST_PRO
    assert p.plan([], 10**9) is None  # held: refused even if the caller forgot the startup check
    _, forced = _publisher(("s1-fast", "s1-pro"), force=True)
    assert forced.plan([], 10) == [FAST_PRO]


def test_multimodel_publish_keeps_competition(started):
    _, p = _publisher(("s1-fast", "s1-pro"))
    assert p.plan([FAST, ENG], 150) == [FAST_PRO, ENG]
    assert p.plan([FAST_PRO, ENG], 150) is None


async def test_publish_writes_via_set_commitment_fields(started):
    chain, p = _publisher()
    await p.publish([SPEC], 150)
    assert chain.writes == [[FAST, SPEC]]
    await p.publish([FAST, SPEC], 150)
    assert len(chain.writes) == 1


def _snap(fields, block=150, hotkey="me"):
    n = Neuron(0, hotkey, "c", False, 0.0, None, None, commitment="".join(fields), commitment_block=5,
               commitment_fields=tuple(fields))
    return Snapshot(block, (n,), None)


async def test_republish_compares_only_the_inference_element(started):
    chain, p = _publisher()
    check = p.on_snapshot("me")
    check(_snap([FAST, SPEC]))  # in sync (competition element is not ours): no task
    await asyncio.sleep(0.05)
    assert chain.writes == []
    check(_snap([ENG]))  # inference element missing -> written next to the engine element
    await asyncio.sleep(0.05)
    assert chain.writes == [[FAST, ENG]]
    check(_snap([ENG]))  # within 10 minutes: no second attempt
    await asyncio.sleep(0.05)
    assert len(chain.writes) == 1
    p._last_try = -1e9
    check(_snap([FAST, SPEC], hotkey="other"))  # not our hotkey in the snapshot: nothing
    await asyncio.sleep(0.05)
    assert len(chain.writes) == 1


# ---------------------------------------------------------------------------------------------------- the tools
def test_write_plan_merges_and_refuses(started):
    assert commitcli.write_plan([FAST], ENG, 150) == [FAST, ENG]
    assert commitcli.write_plan([FAST, SPEC], ENG, 150) == [FAST, ENG]  # replaces the competition element
    assert commitcli.write_plan([SPEC], FAST_PRO, 150) == [FAST_PRO, SPEC]
    assert commitcli.write_plan([FAST, SPEC], FAST, 150, replace_all=True) == [FAST]
    with pytest.raises(commitcli.Refused, match="not a valid element"):
        commitcli.write_plan([FAST], "reax-eng/1:junk", 150)


def test_write_plan_before_the_start(monkeypatch):
    # held: a plain single element (no other kind present) is today's behaviour
    assert commitcli.write_plan([], ENG, 10) == [ENG]
    assert commitcli.write_plan([SPEC], ENG, 10) == [ENG]  # same kind replaced
    assert commitcli.write_plan([FAST], FAST_PRO, 10, force=True) == [FAST_PRO]
    with pytest.raises(commitcli.Refused, match="--replace-all"):
        commitcli.write_plan([FAST], ENG, 10)
    with pytest.raises(commitcli.Refused, match="multi-model"):
        commitcli.write_plan([], FAST_PRO, 10)
    assert commitcli.write_plan([FAST], ENG, 10, replace_all=True) == [ENG]
    assert commitcli.write_plan([FAST], ENG, 10, force=True) == [FAST, ENG]
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", 200)
    with pytest.raises(commitcli.Refused, match="counts from block 200"):
        commitcli.write_plan([FAST], ENG, 199)
    assert commitcli.write_plan([FAST], ENG, 200) == [FAST, ENG]


def test_withdraw_plan():
    assert commitcli.withdraw_plan([FAST, SPEC], "competition") == [FAST]
    assert commitcli.withdraw_plan([SPEC, FAST], "inference") == [SPEC]
    with pytest.raises(commitcli.Refused, match="only one"):
        commitcli.withdraw_plan([SPEC], "competition")
    with pytest.raises(commitcli.Refused, match="no competition"):
        commitcli.withdraw_plan([FAST], "competition")


class FakeChain:
    def __init__(self, fields, block=150, hotkey="me"):
        self.neuron = Neuron(3, hotkey, "c", False, 0.0, None, None, commitment="".join(fields), commitment_block=77,
                             commitment_fields=tuple(fields))
        self.block, self.writes = block, []

    def snapshot(self):
        return Snapshot(self.block, (self.neuron,), None)

    def set_commitment_fields(self, fields):
        self.writes.append(list(fields))


@pytest.fixture
def commit_env(monkeypatch):
    def use(fields, block=150):
        chain = FakeChain(fields, block)
        monkeypatch.setattr(commitcli, "_open_chain", lambda args, need_wallet: (chain, "me"))
        return chain

    monkeypatch.delenv("REAX_FORCE_MULTIROLE", raising=False)
    return use


def test_reax_commit_show(commit_env, started, capsys):
    commit_env([FAST_PRO.replace(MID["s1-pro"], "0123456789abcdef"), SPEC])
    assert commitcli.main(["show"]) == 0
    out = capsys.readouterr().out
    assert "hotkey me uid 3" in out and "inference element" in out and "competition element" in out
    assert f"model s1-fast: manifest {MID['s1-fast']} matches this release" in out
    assert "model s1-pro: manifest 0123456789abcdef STALE" in out
    assert f"read at block 150: inference={FAST_PRO.replace(MID['s1-pro'], '0123456789abcdef')}" in out


def test_reax_commit_show_held_and_invalid(commit_env, capsys, monkeypatch):
    monkeypatch.setattr(E, "MULTIMODEL_START_BLOCK", None)  # held (the committed default; the release sets a block)
    monkeypatch.setattr(E, "DUALROLE_START_BLOCK", None)
    commit_env([FAST, SPEC])  # two fields while held
    assert commitcli.main(["show"]) == 0
    out = capsys.readouterr().out
    assert "read NO role" in out and "is held in this release" in out
    commit_env(["junk"])
    commitcli.main(["show"])
    assert "NOT a valid element" in capsys.readouterr().out


def test_reax_commit_set_merges(commit_env, started, capsys):
    chain = commit_env([FAST])
    assert commitcli.main(["set", SPEC]) == 0
    assert chain.writes == [[FAST, SPEC]]
    chain = commit_env([FAST, SPEC])
    assert commitcli.main(["set", ENG]) == 0
    assert chain.writes == [[FAST, ENG]]
    chain = commit_env([FAST, ENG])
    assert commitcli.main(["set", FAST_PRO]) == 0
    assert chain.writes == [[FAST_PRO, ENG]]
    chain = commit_env([FAST, ENG])
    assert commitcli.main(["set", ENG]) == 0 and chain.writes == []  # unchanged: no write
    assert "nothing to change" in capsys.readouterr().out


def test_reax_commit_set_refuses_before_the_start(commit_env, capsys, monkeypatch):
    chain = commit_env([FAST])
    assert commitcli.main(["set", SPEC]) == 1
    assert chain.writes == [] and "--replace-all" in capsys.readouterr().err
    assert commitcli.main(["set", SPEC, "--replace-all"]) == 0
    assert chain.writes == [[SPEC]]
    chain = commit_env([FAST])
    monkeypatch.setenv("REAX_FORCE_MULTIROLE", "1")
    assert commitcli.main(["set", SPEC]) == 0
    assert chain.writes == [[FAST, SPEC]]
    monkeypatch.delenv("REAX_FORCE_MULTIROLE")
    chain = commit_env([])
    assert commitcli.main(["set", SPEC]) == 0 and chain.writes == [[SPEC]]  # nothing to lose: today's behaviour


def test_reax_commit_withdraw(commit_env, capsys):
    chain = commit_env([FAST, SPEC])  # withdraw works at any time (a single element is always valid)
    assert commitcli.main(["withdraw", "competition"]) == 0
    assert chain.writes == [[FAST]]
    chain = commit_env([FAST, SPEC])
    assert commitcli.main(["withdraw", "inference"]) == 0
    assert chain.writes == [[SPEC]]
    chain = commit_env([SPEC])
    assert commitcli.main(["withdraw", "competition"]) == 1 and chain.writes == []
    assert "only one" in capsys.readouterr().err


def test_reax_commit_dry_run_writes_nothing(commit_env, started, capsys):
    chain = commit_env([FAST])
    assert commitcli.main(["set", SPEC, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"[0] {FAST}" in out and f"[1] {SPEC}" in out and "dry run" in out and chain.writes == []
    chain = commit_env([FAST, SPEC])
    assert commitcli.main(["withdraw", "competition", "--dry-run"]) == 0
    assert chain.writes == [] and f"[0] {FAST}" in capsys.readouterr().out


def test_reax_commit_unregistered_hotkey(commit_env, capsys, monkeypatch):
    chain = FakeChain([FAST], hotkey="someone")
    monkeypatch.setattr(commitcli, "_open_chain", lambda args, need_wallet: (chain, "me"))
    assert commitcli.main(["show"]) == 1 and "not registered" in capsys.readouterr().err


def test_commit_entry_point_is_declared():
    text = open(__file__.replace("tests/test_miner_multimodel.py", "pyproject.toml")).read()
    assert 'reax-commit = "reax_subnet.commitcli:main"' in text


# --------------------------------------------------------------------------------------- reax-engine-submit
@pytest.fixture
def engine_env(monkeypatch, make_wallet):
    import reax_subnet.chain as chain_mod
    import reax_subnet.cli as cli_mod

    monkeypatch.setenv("REAX_NETWORK", "ws://127.0.0.1:9951")
    monkeypatch.setenv("REAX_NETUID", "7")
    monkeypatch.setenv("REAX_FEATURE_ENGINE_COMPETITION", "1")
    monkeypatch.delenv("REAX_FORCE_MULTIROLE", raising=False)
    monkeypatch.setattr(cli_mod, "_wallet", lambda common: (None, "me"))

    def use(fields, block=150):
        chain = FakeChain(fields, block)
        monkeypatch.setattr(chain_mod, "SdkChain", lambda *a, **k: chain)
        return chain

    return use


def test_engine_submit_keeps_the_inference_element(engine_env, started, capsys):
    from reax_subnet.engcli import submit_main

    chain = engine_env([FAST_PRO])
    submit_main([ENG.split(":", 2)[2], "--track", "s1-fast"])
    assert chain.writes == [[FAST_PRO, ENG]]
    chain = engine_env([FAST, ENG2])
    submit_main([ENG.split(":", 2)[2]])
    assert chain.writes == [[FAST, ENG]]
    chain = engine_env([FAST, ENG])
    submit_main([ENG.split(":", 2)[2]])  # identical: nothing written
    assert chain.writes == [] and "nothing written" in capsys.readouterr().out


def test_engine_submit_before_the_start(engine_env, capsys):
    from reax_subnet.engcli import submit_main

    image = ENG.split(":", 2)[2]
    chain = engine_env([FAST])
    with pytest.raises(SystemExit, match="--replace-all"):
        submit_main([image])
    assert chain.writes == []
    submit_main([image, "--replace-all"])
    assert chain.writes == [[ENG]]
    chain = engine_env([])  # nothing else on the hotkey: exactly today's single-field write
    submit_main([image])
    assert chain.writes == [[ENG]]
    chain = engine_env([ENG2])
    submit_main([image])
    assert chain.writes == [[ENG]]


# ---------------------------------------------------------------------------------------------- critic round 1 (F1, F2)
class RereadChain(Chain):
    """A chain whose fresh read (``snapshot``) returns ``reread`` for the miner's own hotkey."""

    def __init__(self, reread):
        super().__init__()
        self.reread = reread

    def snapshot(self):
        return _snap(self.reread, hotkey="me")


def test_publisher_keeps_an_element_of_a_later_release(started):
    gen = "reax-gen/1:generalist:alice/model@" + "d" * 40
    _, p = _publisher()
    assert p.published([FAST, gen])  # not this miner's business: nothing to write
    assert p.plan(["reax-s1/1:s1-fast:0123456789abcdef", gen], 150) == [FAST, gen]


def test_publisher_keeps_a_whitespace_padded_entry(started):
    _, p = _publisher()
    assert p.plan(["reax-s1/1:s1-fast:0123456789abcdef", " " + SPEC], 150) == [FAST, SPEC]


async def test_publisher_writes_only_after_a_second_read_agrees(started):
    # first read shows the commitment without its competition element (a flaky read); the fresh read shows it: no write
    chain = RereadChain([FAST.replace(MID["s1-fast"], "0123456789abcdef"), SPEC])
    p = CommitmentPublisher(chain, {"s1-fast": MID["s1-fast"]}, hotkey="me")
    await p.publish([], 150)
    assert chain.writes == []
    # both reads agree: written, the competition element kept
    await p.publish(list(chain.reread), 150)
    assert chain.writes == [[FAST, SPEC]]


async def test_publisher_does_not_write_when_the_reread_fails(started):
    class Down(Chain):
        def snapshot(self):
            raise RuntimeError("node down")

    chain = Down()
    p = CommitmentPublisher(chain, {"s1-fast": MID["s1-fast"]}, hotkey="me")
    await p.publish([SPEC], 150)
    assert chain.writes == []


def test_withdraw_keeps_an_element_of_a_later_release(started):
    from reax_subnet.commitcli import withdraw_plan

    gen = "reax-gen/1:generalist:alice/model@" + "d" * 40
    with pytest.raises(Exception):
        withdraw_plan([gen], "competition")  # this release cannot read it as its competition element
    assert withdraw_plan([FAST, SPEC], "competition") == [FAST]




# ---------------------------------------------------------------------------------------------- critic round 2 (NEW-1)
@pytest.mark.parametrize("bad", ["reax-spec/1:broken", "reax-eng/1:", "reax-s1/1:s1-fast:NOTHEX", "reax-s1m/1:s1-fast:x"])
def test_malformed_elements_of_known_protocols_are_garbage_not_future_elements(started, bad):
    assert E.loose_kind(bad) is None
    _, p = _publisher()
    assert not p.published([FAST, bad])  # validators read no role from it: the miner repairs it
    assert p.plan([FAST, bad], 150) == [FAST]


def test_submit_tools_never_keep_a_malformed_inference_element(started):
    from reax_subnet.commitcli import write_plan

    assert write_plan(["reax-s1/1:s1-fast:NOTHEX"], SPEC, 150) == [SPEC]


def test_publisher_requires_the_hotkey_argument():
    with pytest.raises(TypeError):
        CommitmentPublisher(Chain(), {"s1-fast": MID["s1-fast"]})
