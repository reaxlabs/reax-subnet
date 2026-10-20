"""Waiting repositories get bounded metadata probes before queue selection."""

import dataclasses
import json

import pytest
from spec_helpers import REV, snapshot
from test_spec_evaluator import Env, TRACK

from reax_subnet.spec import config as C
from reax_subnet.spec import fetch as F
from reax_subnet.spec import state as ST
from reax_subnet.spec.commitment import SpecCommitment

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


async def test_newly_public_entry_precedes_a_later_commitment(tmp_path):
    env = Env(tmp_path, {}, max_evals=1)
    env.add(1, "m/old", 0.9)
    env.hub.unavailable.add("m/old")
    await env.cycle({1: "m/old"}, block=1000)
    assert env.sub(1).reason.startswith("waiting for the repo")
    env.hub.unavailable.clear()
    env.add(2, "m/new", 0.9)
    # Even inside the old 600-second retry interval, metadata is probed on every cycle.
    await env.cycle({1: "m/old", 2: "m/new"}, block=1010)
    assert env.hub.probes == ["m/old"]
    assert env.sub(1).status == ST.PENDING_CONFIRM
    assert env.sub(2).status == ST.QUEUED
    assert env.server.served == ["m/old"]


def _waiting(env, count, track=TRACK):
    ts = env.ev.state.track(track)
    for i in reversed(range(count)):
        sub = ST.Submission(key=f"k{i:03}", uid=i, hotkey=f"h{i}", coldkey=f"c{i}", track_id=track,
                            repo=f"r/{i}", revision=REV, first_seen_block=i, first_seen_ts=0,
                            reason="waiting for the repo: 401", last_fetch_ts=env.clock.t)
        ts.submissions[sub.key] = sub
    return ts


@pytest.mark.parametrize("count", [7, C.WAIT_REPO_PROBES_PER_CYCLE + 5])
async def test_all_waiters_are_probed_up_to_cap_without_downloads_or_skips(tmp_path, count):
    env = Env(tmp_path, {}, max_evals=1)
    _waiting(env, count)
    # A second track has its own independent allowance.
    _waiting(env, count, "risk_assessment")
    for tdef in C.DEFAULT_TRACKS[:2]:
        await env.ev._retry_waiting(tdef, snapshot(1000, []), env.clock.t)
        await env.ev._retry_waiting(tdef, snapshot(1000, []), env.clock.t)
    expected = [f"r/{i}" for i in range(min(count, C.WAIT_REPO_PROBES_PER_CYCLE))]
    assert env.hub.probes == expected * 2
    assert env.hub.fetches == []
    assert env.ev._cycle_skips == {}
    assert env.ev._cycle_prepared == {}


@pytest.mark.parametrize("expired", [False, True])
async def test_still_private_waiters_do_not_block_a_fresh_entry(tmp_path, expired):
    env = Env(tmp_path, {}, max_evals=1)
    commits = {u: f"private/{u}" for u in range(1, 9)}
    # Simulate a saved queue of more 401s than the preparation skip allowance.
    for u, repo in commits.items():
        env.hub.unavailable.add(repo)
    from spec_helpers import commit, neuron
    snap = snapshot(1000, [neuron(u, commit(TRACK, repo)) for u, repo in commits.items()])
    env.ev._sync(env.cfg.tracks[0], snap, env.clock.t)
    for sub in env.ev.state.track(TRACK).submissions.values():
        sub.reason = "waiting for the repo: 401"
    env.add(9, "m/public", 0.9)
    commits[9] = "m/public"
    env.clock.advance(C.RETRY_UNAVAILABLE_S)
    block = 1000 + C.PUBLISH_DEADLINE_BLOCKS + 1 if expired else 1010
    await env.cycle(commits, block=block)
    assert len(env.hub.probes) == 8
    assert env.hub.fetches == ["m/public"]
    assert env.sub(9).status == ST.PENDING_CONFIRM
    assert env.ev._cycle_skips == {}
    rejected = sum(env.sub(u).status == ST.REJECTED for u in range(1, 9))
    assert rejected == (4 if expired else 0)
    assert all(env.sub(u).status in (ST.QUEUED, ST.REJECTED) for u in range(1, 9))


async def test_old_state_loads_and_recovers_without_new_fields(tmp_path):
    env = Env(tmp_path, {}, max_evals=1)
    env.add(1, "m/old", 0.9)
    env.hub.unavailable.add("m/old")
    await env.cycle({1: "m/old"})
    raw = json.loads((tmp_path / "spec.json").read_text())
    # Older releases also lack last_fetch_ts; _retry_at used last_eval_ts as fallback.
    for sub in raw["tracks"][TRACK]["submissions"].values():
        sub.pop("last_fetch_ts", None)
    (tmp_path / "spec.json").write_text(json.dumps(raw))
    loaded = ST.load(env.cfg.state_path)
    assert loaded.track(TRACK).submissions[env.sub(1).key].first_seen_block == 1000
    env.ev.state = loaded
    env.hub.unavailable.clear()
    await env.cycle({1: "m/old"})
    assert env.sub(1).status == ST.PENDING_CONFIRM
    # Serialization still uses precisely the existing state dataclasses.
    assert set(dataclasses.asdict(env.sub(1))) == set(raw["tracks"][TRACK]["submissions"][env.sub(1).key]) | {"last_fetch_ts"}


@pytest.mark.parametrize("private,gated", [(False, False), (True, False), (False, True)])
async def test_real_fetcher_probe_uses_listing_and_never_downloads(private, gated):
    class Hub:
        calls = []

        def listing(self, repo, revision):
            self.calls.append((repo, revision))
            return F.RepoListing(REV, {}, private, gated)

        async def download(self, *args):
            pytest.fail("availability probe downloaded files")

    hub = Hub()
    fetcher = F.HfFetcher(hub=hub)
    commitment = SpecCommitment(TRACK, "m/public", REV)
    if private or gated:
        with pytest.raises(F.FetchError, match="gated or private"):
            await fetcher.probe(commitment)
    else:
        await fetcher.probe(commitment)
    assert hub.calls == [("m/public", REV)]


async def test_available_waiters_are_only_probed_when_no_budget_or_disk(tmp_path):
    env = Env(tmp_path, {}, max_evals=0)
    commits = {u: f"m/{u}" for u in range(1, 6)}
    for u, repo in commits.items():
        env.add(u, repo, 0.9)
    from spec_helpers import commit, neuron
    snap = snapshot(1000, [neuron(u, commit(TRACK, repo)) for u, repo in commits.items()])
    env.ev._sync(env.cfg.tracks[0], snap, env.clock.t)
    for sub in env.ev.state.track(TRACK).submissions.values():
        sub.reason = "waiting for the repo: 401"
    env.cfg.min_free_bytes = 10**30
    await env.ev.evaluate_cycle(snap)
    assert env.hub.probes == list(commits.values())
    assert env.hub.fetches == []
    assert all(env.sub(u).reason == "" for u in commits)
    assert env.ev.state.eval_times == []


@pytest.mark.parametrize("hash_named", [False, True])
async def test_hub_infrastructure_errors_stop_probing_and_count_nothing(tmp_path, hash_named):
    """A 429/5xx/timeout from the hub is infrastructure: the miner is not charged, and the cycle stops hitting the hub."""
    env = Env(tmp_path, {}, max_evals=1)
    ts = _waiting(env, 10)
    _waiting(env, 10, "risk_assessment")
    for sub in ts.submissions.values():
        sub.hash_named = hash_named
        sub.last_fetch_ts = env.clock.t - C.RETRY_UNAVAILABLE_S    # every retry is due
    calls = []

    async def probe(commitment):
        calls.append(commitment.repo)
        raise F.FetchError("infra", "hub error: 429 Too Many Requests")

    env.hub.probe = probe
    snap = snapshot(1000 + C.PUBLISH_DEADLINE_BLOCKS + 1, [])    # past the publish deadline too
    for tdef in C.DEFAULT_TRACKS[:2]:
        await env.ev._retry_waiting(tdef, snap, env.clock.t)
    assert calls == ["r/0"]    # one request in the whole cycle, not 20
    for sub in ts.submissions.values():
        assert sub.status == ST.QUEUED and sub.unavailable_fetches == 0 and sub.reason.startswith("waiting for the repo")
    # the next cycle probes again from the oldest entry
    env.ev._retry_tracks.clear()
    env.ev._probes_halted = False
    await env.ev._retry_waiting(C.DEFAULT_TRACKS[0], snap, env.clock.t)
    assert calls == ["r/0", "r/0"]


async def test_a_local_download_error_keeps_its_retry_delay(tmp_path):
    """A waiting entry parked by a local staging/disk error (WAIT_INFRA) stays parked for RETRY_UNAVAILABLE_S even when
    the hub listing is fine; a hub that does not have it yet (401) cannot shorten or lengthen anything."""
    env = Env(tmp_path, {}, max_evals=1)
    ts = _waiting(env, 1)
    sub = ts.submissions["k000"]
    sub.wait_kind = ST.WAIT_INFRA
    env.add(0, "r/0", 0.5)
    await env.ev._retry_waiting(C.DEFAULT_TRACKS[0], snapshot(1000, []), env.clock.t)
    assert sub.reason.startswith("waiting for the repo")
    env.ev._retry_tracks.clear()
    await env.ev._retry_waiting(C.DEFAULT_TRACKS[0], snapshot(1000, []), env.clock.t + C.RETRY_UNAVAILABLE_S)
    assert sub.reason == ""


async def test_unavailable_attempt_and_deadline_rules_are_unchanged(tmp_path):
    """Probes that find the repo still private count one attempt per 600 s for a hash-named entry (three reject it) and
    reject a name-less entry only after the publish deadline; probes inside the delay count nothing."""
    env = Env(tmp_path, {}, max_evals=1)
    ts = _waiting(env, 2)
    named, plain = ts.submissions["k000"], ts.submissions["k001"]
    named.hash_named, plain.hash_named = True, False
    tdef = C.DEFAULT_TRACKS[0]
    for i in range(C.MAX_ATTEMPTS):
        env.ev._retry_tracks.clear()
        env.clock.advance(C.RETRY_UNAVAILABLE_S)
        await env.ev._retry_waiting(tdef, snapshot(1000, []), env.clock.t)
        env.ev._retry_tracks.clear()
        await env.ev._retry_waiting(tdef, snapshot(1000, []), env.clock.t)    # same instant: inside the delay
    assert named.status == ST.REJECTED and "no longer available" in named.reason
    assert plain.status == ST.QUEUED
    env.ev._retry_tracks.clear()
    env.clock.advance(C.RETRY_UNAVAILABLE_S)
    await env.ev._retry_waiting(tdef, snapshot(1 + C.PUBLISH_DEADLINE_BLOCKS + 1, []), env.clock.t)
    assert plain.status == ST.REJECTED and "not available within" in plain.reason
