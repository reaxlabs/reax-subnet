"""Engine-track code gates: image ownership of cached digests, measurement context, conservative champion score,
cancellation-safe subprocess/Docker start, parameter validation and the engine hold."""

import ast
import asyncio
import dataclasses
import hashlib
import os
import sys

import pytest

from reax_subnet import engrunner as ER
from reax_subnet import engvalidator as EV
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.engcomp import (
    Champion, EvalResult, Submission, TrackParams, TrackState, check_correctness, observe, update_champion,
)
from test_engcomp import FAST_P, P, ref_of

DA = hashlib.sha256(b"a").hexdigest()
DB = hashlib.sha256(b"b").hexdigest()
REF = ref_of("reference")


def _sub(uid, hotkey, digest=DA, block=100):
    return Submission(uid, hotkey, "s1-fast", "ghcr.io/a/e", digest, block)


def _snapshot(subs, block=2000):
    return Snapshot(block, tuple(Neuron(s.uid, s.hotkey, "cold", False, 0, None, None, "reax-eng/1:s1-fast:" + s.ref,
                                        s.block) for s in subs), None)


def _res(digest, score, block=100, session="", ok=True):
    return EvalResult(digest, "s1-fast", ok, score=score, block=block, session=session or f"s{block}")


class _Binding:
    """Image-to-hotkey binding runner: the owner of every digest is ``owners``; lookups are counted."""
    binds_hotkey = True

    def __init__(self, owners=None, fail=None):
        self.owners, self.calls, self.fail = owners or {}, [], fail

    async def owner_of(self, ref):
        self.calls.append(ref)
        if self.fail:
            raise self.fail
        return self.owners[ref.rsplit(":", 1)[1]]

    async def discard(self, ref):
        pass


def _champion_state(owner="owner", record_owner=True):
    st = TrackState(results={DA: _res(DA, 1.5)}, champion=Champion(DA, owner, 2, 1.5, 100))
    if record_owner:
        st.owners[DA] = owner
    return st


# ------------------------------------------------------------------------------------------------ F1
@pytest.mark.parametrize("order", ["owner-first", "squatter-first"])
async def test_cached_digest_requires_owner_validation(tmp_path, order):
    owner, squatter = _sub(2, "owner"), _sub(3, "squatter", block=150)
    subs = [owner, squatter] if order == "owner-first" else [squatter, owner]
    runner = _Binding(fail=AssertionError("a verified digest needs no lookup"))
    path = str(tmp_path / "s.json")
    comp = EV.EngineCompetition(runner, {"s1-fast": REF}, path, tracks={"s1-fast": P}, start_utc=None)
    comp.states["s1-fast"] = _champion_state()
    await comp.run_cycle(_snapshot(subs), set())
    st = comp.states["s1-fast"]
    assert st.champion is not None and st.champion.hotkey == "owner" and st.champion.uid == 2
    assert f"{DA}:squatter" in st.not_owner and st.challenger is None
    assert runner.calls == []
    w = comp.weights({9: 1.0}, _snapshot(subs), set())
    assert w[2] > 0 and 3 not in w
    # restart: the squatter stays excluded and the owner keeps the title
    again = EV.EngineCompetition(_Binding(fail=AssertionError("no lookup")), {"s1-fast": REF}, path,
                                 tracks={"s1-fast": P}, start_utc=None)
    await again.run_cycle(_snapshot(subs), set())
    assert again.states["s1-fast"].champion.hotkey == "owner"
    w = again.weights({9: 1.0}, _snapshot(subs), set())
    assert w[2] > 0 and 3 not in w


async def test_weights_ignore_a_second_hotkey_on_a_cached_digest_before_the_next_cycle(tmp_path):
    comp = EV.EngineCompetition(_Binding(), {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": P},
                                start_utc=None)
    comp.states["s1-fast"] = _champion_state()
    snap = _snapshot([_sub(3, "squatter", block=150), _sub(2, "owner")])
    assert comp.weights({9: 1.0}, snap, set())[2] > 0


async def test_legacy_cached_digest_without_a_recorded_owner_is_looked_up_once(tmp_path):
    runner = _Binding({DA: "owner"})
    comp = EV.EngineCompetition(runner, {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": P},
                                start_utc=None)
    st = comp.states["s1-fast"]
    st.results[DA] = _res(DA, 1.01, ok=False)  # cached, not the champion, owner never recorded
    await comp.run_cycle(_snapshot([_sub(3, "squatter", block=150), _sub(2, "owner")]), set())
    assert f"{DA}:squatter" in st.not_owner and st.owners[DA] == "owner"
    assert len(runner.calls) == 2  # one lookup per hotkey, not one per cycle
    await comp.run_cycle(_snapshot([_sub(3, "squatter", block=150), _sub(2, "owner")]), set())
    assert len(runner.calls) == 2


def test_champion_is_matched_by_digest_and_hotkey():
    owner, squatter = _sub(2, "owner"), _sub(3, "squatter", block=150)
    st = _champion_state()
    observe(st, [squatter, owner], 2000, first_committer_only=False)
    ch = update_champion(st, [squatter, owner], P, 2000)
    assert ch is not None and ch.uid == 2 and ch.hotkey == "owner"
    gone = update_champion(st, [squatter], P, 2001)  # the owner really withdrew: the squatter does not inherit
    assert gone is None


# ------------------------------------------------------------------------------------------------ F2 (engdet) + timeout
async def test_owner_lookup_timeout_is_inconclusive(tmp_path):
    class Slow(_Binding):
        async def owner_of(self, ref):
            await asyncio.sleep(10)

    tight = dataclasses.replace(P, eval_budget_s=0.01)
    comp = EV.EngineCompetition(Slow(), {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": tight},
                                start_utc=None)
    st = comp.states["s1-fast"]
    st.results[DA] = _res(DA, 1.5)
    st.champion = Champion(DA, "owner", 2, 1.5, 100)
    res = await comp._evaluate_one(st, "s1-fast", tight, _sub(2, "owner"), 2000)
    assert res is None and st.attempts == {DA: 1}  # retried later, no failed result, no invalid session
    observe(st, [_sub(2, "owner")], 2000)
    update_champion(st, [_sub(2, "owner")], tight, 2000)
    assert st.champion is not None and st.champion_invalid == 0


# ------------------------------------------------------------------------------------------------ F3
def _write(tmp_path, ref=REF, chain="finney:92", params=P, weights="w1", with_champion=True):
    class Runner:
        binds_hotkey = False

        def weights_identity(self):
            return weights

    path = str(tmp_path / "s.json")
    comp = EV.EngineCompetition(Runner(), {"s1-fast": ref}, path, tracks={"s1-fast": params}, start_utc=None,
                                chain_id=chain)
    return Runner, path, comp


@pytest.mark.parametrize("change", ["reference", "chain", "weights", "params"])
async def test_reference_context_change_invalidates_saved_scores(tmp_path, change):
    Runner, path, comp = _write(tmp_path)
    st = comp.states["s1-fast"]
    st.results[DA] = _res(DA, 1.5)
    st.champion = Champion(DA, "owner", 2, 1.5, 100)
    st.owners[DA] = "owner"
    comp.save()
    kw = {"reference": dict(ref=ref_of("new-reference")), "chain": dict(chain="test:92"),
          "weights": dict(weights="w2"), "params": dict(params=dataclasses.replace(P, mean_tau=0.09))}[change]
    runner = Runner()
    runner.weights_identity = lambda w=kw.get("weights", "w1"): w
    comp2 = EV.EngineCompetition(runner, {"s1-fast": kw.get("ref", REF)}, path,
                                 tracks={"s1-fast": kw.get("params", P)}, start_utc=None,
                                 chain_id=kw.get("chain", "finney:92"))
    st2 = comp2.states["s1-fast"]
    assert comp2.context_reset and st2.champion is None and not st2.results
    assert st2.owners == {DA: "owner"}  # what was verified about the image is not a measurement
    assert st2.history[-1]["event"] == "context_changed" and st2.history[-1]["dropped_champion"] == DA
    # the saved champion pays nothing: the idle cycle has no result and no champion
    assert comp2.weights({9: 1.0}, _snapshot([_sub(2, "owner")]), set()) == {9: 1.0}


async def test_same_context_keeps_the_saved_champion_and_legacy_state_is_not_trusted(tmp_path):
    Runner, path, comp = _write(tmp_path)
    comp.states["s1-fast"].champion = Champion(DA, "owner", 2, 1.5, 100)
    comp.save()
    assert EV.EngineCompetition(Runner(), {"s1-fast": REF}, path, tracks={"s1-fast": P}, start_utc=None,
                                chain_id="finney:92").states["s1-fast"].champion is not None
    import json
    raw = json.load(open(path))
    del raw["context_sha256"], raw["context"]
    json.dump(raw, open(path, "w"))
    legacy = EV.EngineCompetition(Runner(), {"s1-fast": REF}, path, tracks={"s1-fast": P}, start_utc=None,
                                  chain_id="finney:92")
    assert legacy.context_reset and legacy.states["s1-fast"].champion is None


@pytest.mark.usefixtures("rules_v4_gates_off")
async def test_reevaluation_after_a_context_change_needs_the_confirmations(tmp_path, monkeypatch):
    Runner, path, comp = _write(tmp_path)
    comp.states["s1-fast"].champion = Champion(DA, "owner", 2, 1.5, 100)
    comp.save()
    calls = []

    async def evaluate(runner, reference, ref, track, params, block=0, digest="", progress=None):
        calls.append(digest)
        return _res(digest, 1.5, block)

    monkeypatch.setattr(EV, "evaluate", evaluate)
    comp2 = EV.EngineCompetition(Runner(), {"s1-fast": ref_of("new")}, path, tracks={"s1-fast": P}, start_utc=None,
                                 chain_id="finney:92")
    await comp2.run_cycle(_snapshot([_sub(2, "owner")]), set())
    assert calls == [DA] * P.confirmations  # measured again, in separate sessions, before it is crowned
    assert comp2.states["s1-fast"].champion is not None


def test_docker_weights_identity_follows_the_files(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "a.bin").write_bytes(b"1234")
    runner = ER.DockerRunner(str(tmp_path / "work"), str(tmp_path / "w"))
    first = runner.weights_identity()
    assert first == runner.weights_identity()
    (tmp_path / "w" / "a.bin").write_bytes(b"12345")
    assert runner.weights_identity() != first


# ------------------------------------------------------------------------------------------------ F4
def test_confirmed_minimum_survives_idle_cycle_and_restart():
    a, b = _sub(2, "owner"), _sub(4, "other", digest=DB, block=300)
    st = TrackState()
    observe(st, [a, b], 100)
    st.results[DA] = _res(DA, 1.4, block=100, session="s1")
    update_champion(st, [a], P, 100)
    st.results[DA] = _res(DA, 1.6, block=2000, session="s2")
    assert update_champion(st, [a], P, 2000).score == 1.4  # crowned at the lowest confirmation
    for block in (2100, 2200):  # idle cycles apply no new measurement
        assert update_champion(st, [a], P, block).score == 1.4
    st = TrackState.from_json(st.to_json())
    assert update_champion(st, [a], P, 2300).score == 1.4
    # a challenger 3.5 % above the confirmed 1.4 (1.449) clears the margin although 1.6 would not hold it
    st.results[DB] = _res(DB, 1.45, block=2300, session="c1")
    observe(st, [a, b], 2300)
    update_champion(st, [a, b], P, 2300)
    assert st.challenger is not None and st.challenger["digest"] == DB


# ------------------------------------------------------------------------------------------------ F5
async def test_run_cancellation_reaps_child(tmp_path):
    pidfile = tmp_path / "pid"
    code = "import os,time,pathlib;pathlib.Path(%r).write_text(str(os.getpid()));time.sleep(60)" % str(pidfile)
    task = asyncio.create_task(ER._run([sys.executable, "-c", code], 60))
    for _ in range(300):
        if pidfile.exists() and pidfile.read_text():
            break
        await asyncio.sleep(0.01)
    pid = int(pidfile.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # reaped, not merely signalled


async def test_cancel_during_runner_start_cleans_resources(tmp_path, monkeypatch):
    calls = []

    async def fake_run(argv, timeout):
        calls.append(list(argv[:3]))
        if argv[:2] == ["docker", "run"]:
            await asyncio.sleep(60)
        if argv[:3] == ["docker", "image", "inspect"]:
            return 0, "{}", ""
        return 0, "", ""

    monkeypatch.setattr(ER, "_run", fake_run)
    monkeypatch.setattr(ER, "PUBLISHED_GPU_CLASSES", frozenset({"x"}), raising=False)

    async def ensure(self, ref):
        return None

    async def config_dir(self):
        return "/cfg"

    monkeypatch.setattr(ER.DockerRunner, "_ensure_image", ensure)
    monkeypatch.setattr(ER.DockerRunner, "config_dir", config_dir)
    runner = ER.DockerRunner(str(tmp_path / "sessions"), str(tmp_path))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(runner.start(REF, "s1-fast", P), 0.05)
    assert ["docker", "rm", "-fv"] in calls
    assert not list((tmp_path / "sessions").glob("eng-*"))


# ------------------------------------------------------------------------------------------------ parameters, statistics
@pytest.mark.parametrize("bad", [
    {"determinism_items": 0}, {"determinism_items": -1}, {"start_timeout_s": -1}, {"request_timeout_s": -1},
    {"start_timeout_s": float("inf")}, {"request_timeout_s": float("nan")}, {"eval_budget_s": float("inf")},
    {"mean_tau": float("nan")}, {"temperature": float("inf")}, {"latency_items": 0},
])
def test_track_overrides_validate_counts_and_timeouts(bad):
    with pytest.raises(ValueError):
        dataclasses.replace(P, **bad)


def test_pooled_top_choice_agreement_boundary():
    refs = {f"i{i}": (0.5004, 0.4996) for i in range(512)}
    for flipped, ok in ((5, True), (6, False)):
        got = {k: (0.4996, 0.5004) if i < flipped else v for i, (k, v) in enumerate(refs.items())}
        assert check_correctness(got, refs, P).ok is ok


# ------------------------------------------------------------------------------------------------ engine hold
def test_engine_track_opens_only_at_the_release_date(tmp_path):
    from datetime import datetime, timedelta, timezone

    from reax_subnet import competition_split as CS
    from reax_subnet import worker as W

    start = CS.ENGINE_TRACK_START_UTC
    far = datetime(2100, 1, 1, tzinfo=timezone.utc)
    # 1. the shared start constant is a fixed release date; opening the track is an explicit release edit
    assert start == datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
    assert CS.release_engine_start() == start
    assert not CS.engine_track_open(start - timedelta(seconds=1)) and CS.engine_track_open(start)
    # 2. the finney worker (default configuration) builds the engine with that start
    (tmp_path / "w").mkdir()
    env = {"REAX_NETUID": "92", "REAX_NETWORK": "finney", "REAX_COMPETITIONS_STATE_DIR": str(tmp_path / "state"),
           "REAX_ENGINE_WEIGHTS_DIR": str(tmp_path / "w"), "REAX_ENGINE_WORKDIR": str(tmp_path / "e")}
    engine, _, _ = W.build_local_stack(W.load_worker(env, spec=False).engine, "", env)
    engine._utcnow = lambda: start - timedelta(hours=1)
    assert engine.engine_start_utc == start and not engine.engine_active() and not engine.due()
    # 3. nothing is paid from a champion in the state before the start
    engine.states["s1-fast"].champion = Champion(DA, "owner", 2, 1.5, 100)
    engine.states["s1-fast"].results[DA] = _res(DA, 1.5)
    assert engine.weights({9: 1.0}, _snapshot([_sub(2, "owner")]), set()) == {9: 1.0}
    # 4. past the start the same engine is live (a champion gets paid from then on)
    engine._utcnow = lambda: far
    assert engine.engine_active()
    # 5. every production construction site passes the start explicitly (a new site that forgets it would open the track)
    root = os.path.join(os.path.dirname(__file__), "..", "src", "reax_subnet")
    sites = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(open(os.path.join(root, name)).read())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in (
                    "EngineCompetition", "RemoteEngineCompetition"):
                sites.append((name, node.lineno, {k.arg for k in node.keywords}))
    assert sites, "no construction site found"
    for name, line, kwargs in sites:
        assert "engine_start_utc" in kwargs, f"{name}:{line} constructs the engine competition without engine_start_utc"
