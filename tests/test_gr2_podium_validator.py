"""The validator writes the set-bound Round 2 podium record from the SAME computation as the vector it submits (pytest)."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from reax_subnet import gr2_podium_record as REC
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.config import load_validator
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1validator import S1Validator

import test_gr2_podium_runtime as T

LAST = 900


@pytest.fixture(autouse=True)
def release_need(monkeypatch):
    from reax_subnet import competition_split as C
    monkeypatch.setattr(C, "NEED_RULE", True)     # the release rule (conftest switches it off)
    monkeypatch.delenv("REAX_KA_FEED_PATH", raising=False)


def snap(block=1000):
    me = Neuron(3, "validator", "vc", True, 100., None, None, last_update=LAST)
    return Snapshot(block, tuple(T.neurons()) + (me,), "owner")


def make(tmp_path, fake_chain_cls, harness):
    chain = fake_chain_cls(list(snap().neurons), owner_hotkey="owner", rate_limit=100, since_update=10_000)
    cfg = load_validator({"REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_NETUID": "92", "REAX_ALLOW_DEV_MODELS": "1",
                          "REAX_STATE_PATH": str(tmp_path / "validator-state.json"), "REAX_WARMUP_ROUNDS": "0"})
    val = S1Validator(cfg, SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator")), chain,
                      {"s1-fast": S1HashBackend("s1-fast")}, shares={"s1-fast": 1.})
    val.engine = harness.engine
    val.compute_weights = lambda s: harness.engine.weights(dict(T.BASE), s, set(), T.BURN_FRACTION)
    return chain, val


def podium_harness():
    net = T.Net()
    net.publish(T.OPENING, T.fragment(T.OPENING, T.rec(15), place2=T.rec(25), place3=T.rec(35)))
    return T.Harness(net), net


def lines(tmp_path):
    path = tmp_path / "gr2-podium-sets.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


async def test_record_is_bound_to_the_vector_that_was_sent(tmp_path, fake_chain_cls):
    harness, net = podium_harness()
    chain, val = make(tmp_path, fake_chain_cls, harness)
    assert await val.maybe_set_weights(snap(1000))
    await asyncio.sleep(0)
    sent_uids, sent_weights, _ = chain.set_calls[-1]
    sent = dict(zip(sent_uids, sent_weights))
    line, = lines(tmp_path)
    assert line["vector_sha256"] == REC.vector_sha256(sent)
    assert (line["v"], line["set_block"], line["source_block"], line["status"]) == (1, 1000, 1000, "paid")
    assert (line["window"], line["eligible"]) == (T.OPENING, {"champion": 15, "former[0]": None, "former[1]": None, "place2": 25, "place3": 35})
    assert line["submitted"] is True
    assert set(line) == {"v", "set_block", "window", "fragment_sha256", "source_block", "pool", "keepalive", "eligible",
                         "podium", "burn", "status", "vector_sha256", "submitted"}
    for uid, amount in line["podium"].items():                    # the vector is never below the recorded podium
        assert sent[int(uid)] == pytest.approx(amount, abs=1e-12)
    assert line["pool"] == pytest.approx(T.MINER_PART * T.POOL)
    assert len(net.fragment_calls()) == 1                          # the thread prefetch served compute_weights


async def test_record_amounts_are_before_the_miner_slot_factor_and_the_hash_is_after_it(tmp_path, fake_chain_cls):
    harness, _ = podium_harness()
    chain, val = make(tmp_path, fake_chain_cls, harness)
    val._miner_slot = lambda w, s: {u: (x if u == 0 else x * 1.25) for u, x in w.items()}
    assert await val.maybe_set_weights(snap(1000))
    await asyncio.sleep(0)
    sent = dict(zip(*chain.set_calls[-1][:2]))
    line, = lines(tmp_path)
    assert line["vector_sha256"] == REC.vector_sha256(sent)
    assert sent[15] == pytest.approx(line["podium"]["15"] * 1.25)  # recorded de-scaled, sent scaled
    assert line["pool"] == pytest.approx(T.MINER_PART * T.POOL)


async def test_failed_set_is_marked_submitted_false(tmp_path, fake_chain_cls):
    harness, _ = podium_harness()
    chain, val = make(tmp_path, fake_chain_cls, harness)
    chain.fail_set_weights = True
    assert not await val.maybe_set_weights(snap(1000))
    await asyncio.sleep(0)
    line, = lines(tmp_path)
    assert line["submitted"] is False and line["set_block"] == 1000 and line["status"] == "paid"


async def test_burned_podium_is_recorded_and_the_set_still_goes_out(tmp_path, fake_chain_cls):
    harness, net = podium_harness()
    net.row(T.OPENING, "0" * 64)                                   # the row's sha does not match the file
    chain, val = make(tmp_path, fake_chain_cls, harness)
    assert await val.maybe_set_weights(snap(1000))
    await asyncio.sleep(0)
    sent = dict(zip(*chain.set_calls[-1][:2]))
    line, = lines(tmp_path)
    assert line["status"] == "burned:sha_mismatch" and line["podium"] == {} and line["fragment_sha256"] is None
    assert line["burn"] == pytest.approx(T.MINER_PART * T.REST)
    assert line["vector_sha256"] == REC.vector_sha256(sent)
    assert sent[T.KA_UID] > 0 and sent[T.SERVING_UID] > 0


async def test_flag_off_sends_the_same_vector_and_writes_nothing(tmp_path, fake_chain_cls):
    off = T.Harness(None)
    chain, val = make(tmp_path, fake_chain_cls, off)
    assert await val.maybe_set_weights(snap(1000))
    await asyncio.sleep(0)
    assert not (tmp_path / "gr2-podium-sets.jsonl").exists()
    assert chain.set_calls[-1][1] == [off.engine.weights(dict(T.BASE), snap(1000), set(), T.BURN_FRACTION)[u]
                                      for u in chain.set_calls[-1][0]]
    assert val.podium_prefetcher() is None and val.podium_record() is None


async def test_a_broken_record_never_stops_the_set(tmp_path, fake_chain_cls):
    harness, _ = podium_harness()
    chain, val = make(tmp_path, fake_chain_cls, harness)
    (tmp_path / "gr2-podium-sets.jsonl").mkdir()                    # the record cannot be opened as a file
    assert await val.maybe_set_weights(snap(1000))
    assert len(chain.set_calls) == 1


async def test_an_exception_while_resolving_burns_the_podium_and_the_set_still_goes_out(tmp_path, fake_chain_cls, monkeypatch):
    from reax_subnet import gr2_podium_reader as P

    harness, _ = podium_harness()
    chain, val = make(tmp_path, fake_chain_cls, harness)

    def boom(*args, **kwargs):
        raise KeyError("unexpected")

    monkeypatch.setattr(P, "resolve", boom)
    assert await val.maybe_set_weights(snap(1000))
    await asyncio.sleep(0)
    assert len(chain.set_calls) == 1
    line, = lines(tmp_path)
    assert line["status"] == "burned:internal_error" and line["podium"] == {} and line["submitted"] is True


@pytest.mark.parametrize("failure", ["unreachable", "row_404", "fragment_404", "resolve_raises"])
async def test_before_the_opening_window_a_failed_read_sends_the_flag_off_vector(tmp_path, fake_chain_cls, monkeypatch,
                                                                                failure):
    from datetime import datetime, timezone
    from reax_subnet import gr2_podium_reader as P

    before = datetime(2026, 11, 24, 13, tzinfo=timezone.utc)
    ranks = [(15, 80.0), (25, 70.0), (35, 60.0)]
    net = T.Net()
    if failure == "unreachable":
        net.files[P.COMPETITIONS_URL] = P.FetchError("unavailable")
    elif failure == "fragment_404":
        net.row(T.OPENING, "a" * 64)
    elif failure == "resolve_raises":
        net.row(None, None)
        monkeypatch.setattr(P, "resolve", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    sent = {}
    for name, h in (("off", T.Harness(None, ranks=ranks, now=before)), ("on", T.Harness(net, ranks=ranks, now=before))):
        (tmp_path / name).mkdir()
        chain, val = make(tmp_path / name, fake_chain_cls, h)
        assert await val.maybe_set_weights(snap(1000))
        await asyncio.sleep(0)
        sent[name] = chain.set_calls[-1][:2]
    assert sent["on"] == sent["off"]
    assert sent["off"][0] and 15 in sent["off"][0]          # the ordinary rank path pays the generalist leaders
    assert lines(tmp_path / "on") == []
