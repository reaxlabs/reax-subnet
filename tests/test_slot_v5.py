"""rc11.post4: crowning slots on eval v5.1 from Thu 29 Oct 2026 (slot rules section 3; rc11p3d item F) and a shared co-champion
group that stays the champion when its leader leaves (CRITERIA 6). Fake models and the fixture v5.1 pool only."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import pytest
from conftest import RELEASE_CROWN_SLOTS_UTC, RELEASE_EVAL_V5_START_UTC
from test_v5_wiring import BASE_REPO, Gen, v51_rows

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import state as ST
from reax_subnet.spec import v5, v5wire as W

MON = datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc)
THU = datetime(2026, 10, 29, 14, 30, tzinfo=timezone.utc)
T_MON, T_THU = MON.timestamp(), THU.timestamp()


@pytest.fixture(scope="module")
def rows():
    return v51_rows()


def _write_config(d, version="v0.1"):
    p = os.path.join(d, W.CONFIG_FILE)
    with open(p, "w") as fh:
        json.dump({"pool_version": version, "hf_areas": dict(v5.HF_AREAS)}, fh)
    os.chmod(p, 0o600)


def _release_gen(tmp_path, monkeypatch, rows):
    """The release slots and gate, the real v5.1 parameter resolution (config v0.1 + PANEL_STATS), fewer resamples for speed."""
    g = Gen(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", RELEASE_CROWN_SLOTS_UTC)
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)
    _write_config(g.env.cfg.v5_pool_dir)
    real = E.SpecEvaluator._v5_params

    def params(self_):
        p = real(self_)
        return v5.V5Params(**{**{f: getattr(p, f) for f in p.__dataclass_fields__}, "resamples": 40})

    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", params)
    return g


async def _v5_confirmed(g, entries, *, start=T_MON + 3600):
    """Champion uid 1 crowned under v4 before the gate; ``entries`` ({uid: (repo, skill)}) take their v5.1 first stage at
    ``start`` and confirm 13 h later (after the Monday slot clock): each stores a crowning for the Thursday slot."""
    g.env.clock.t = T_MON - 29 * 3600
    await g.crown(1, "c1/champ", .60)
    g.seat(1)
    g.env.clock.t = start
    for u, (repo, skill) in entries.items():
        g.add(u, repo, skill)
    await g.cycle()
    await g.cycle(13 * 3600)
    for u in entries:
        s = g.sub(u)
        assert E._crown_pending(s) is not None and E._decided_under_v5(s), (u, s.status, s.reason)
        assert E.SpecEvaluator._slot_of(s) == T_THU


# ------------------------------------------------------------------------------------------------ the pinned instant
def test_slot_eval_ts_flip_slot_v4_thursday_v5(monkeypatch):
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    assert ev.slot_eval_ts(T_MON) == T_MON - 1 and ev.slot_eval_ts(T_THU) == T_THU - 1
    assert not ev._slot_v5("generalist", T_MON)                      # the flip slot: slot - 1 s is before the gate -> v4
    assert ev._slot_v5("generalist", T_THU)                          # every later slot -> v5.1
    assert all(ev._slot_v5("generalist", t.timestamp()) for t in RELEASE_CROWN_SLOTS_UTC if t > MON)
    assert not ev._slot_v5("demand_forecasting", T_THU)
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", None)                # gate off: never v5
    assert not ev._slot_v5("generalist", T_THU) and ev.slot_eval_ts(T_THU) == T_THU - 1


# --------------------------------------------------------------------------------------- Thursday slot end to end
@pytest.fixture(scope="module")
def deep_rows():
    return v51_rows(per_type=260)         # three disjoint 900-item rounds fit (60 N items per task type a round)


async def test_thursday_slot_is_decided_on_v51(tmp_path, monkeypatch, deep_rows):
    rows = deep_rows
    g = _release_gen(tmp_path, monkeypatch, rows)
    await _v5_confirmed(g, {3: ("n3/strong", .95), 5: ("n5/weak", .70), 6: ("n6/close", .945)})
    pair_calls = []
    real_pair = W.slot_pair_se
    monkeypatch.setattr(W, "slot_pair_se", lambda c, a, b: pair_calls.append(len(c)) or real_pair(c, a, b))
    await g.cycle(T_THU - 1 - g.env.clock.t)                         # Thu 14:29:59: nothing yet
    assert g.ts.champion == g.sub(1).key and not g.ts.slot_results
    g.env.server.served.clear()
    await g.cycle(1)                                                 # Thu 14:30:00: the slot, on v5.1
    rec = g.ts.slot_results[-1]
    assert rec["slot"] == T_THU and rec["eval_ts"] == T_THU - 1 and rec["evaluation"] == "v5.1"
    assert rec["pool_version"] == "v0.1" and rec["panel"] == "v0.1" and len(rec["pool_sha256"]) == 64
    assert rec["g"] == pytest.approx(0.5) and rec["margin_floor"] == pytest.approx(1.0)
    served = g.env.server.served
    assert "c1/champ" not in served and BASE_REPO not in served          # members only: no champion, no base at a slot
    assert served.count("n3/strong") == rec["rounds"]
    pool = {r["id"]: r for r in [*rows[0], *rows[1]]}
    for d in rec["draws"]:                                           # every round a v5.1 draw: 540 T + 360 N (+ public)
        ids = g.ts.draws[d].item_ids
        assert sum(pool.get(i, {}).get("part") == "T" for i in ids) == 540
        assert sum(pool.get(i, {}).get("part") == "N" for i in ids) == 360
    seen = [set(g.ts.draws[d].item_ids) & set(pool) for d in rec["draws"]]
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1:])     # rounds share no T/N item
    assert not any("least-used items taken" in e for e in g.ts.events)
    weak = next(e for e in rec["entries"] if e["uid"] == 5)
    assert weak["left"]["rule"] == "4.5" and weak["left"]["round"] == 1 and weak["gap"] >= weak["margin"] >= 1.0
    assert pair_calls                                                # the separation SE is the v5.1 pooled pair SE
    strong = g.sub(3)
    assert rec["result"] in ("clear_winner", "shared_group", "leader_alone_beyond_g") and 3 in rec["crowned"]
    assert g.ts.champion == strong.key and strong.status == ST.CHAMPION and g.sub(1).status == ST.FORMER
    row = [h for h in strong.history if h["kind"] == "playoff" and h.get("slot")][-1]
    assert row["scoring_regime"] == "v5" and W.scored_under_v5(strong.by_tier) and E._v51_scored(strong, g.env.clock.t)
    assert strong.score == pytest.approx(sum(row["scores"]) / len(row["scores"]), abs=1e-4)
    e3 = next(e for e in rec["entries"] if e["uid"] == 3)
    assert e3["scores"] == row["scores"] and e3["mean"] == pytest.approx(strong.score)
    assert not E._crown_pending(g.sub(5)) and g.sub(5).status == ST.QUEUED and not g.sub(5).draw1    # a new first draw (rc11.post4)
    # the validator: the new champion is rank 1 on its S_v5.1 (the v4-scale floor does not drop it), the old one place 2
    lead = (await g.cycle(60))["generalist"]
    assert lead[0][0] == 3 and 1 in [u for u, _ in lead]


async def test_flip_slot_stays_v4_and_records_it(tmp_path, monkeypatch, rows):
    """The Monday slot (= the gate) keeps the post3 v4 slot playoff; its record says v4; no v5.1 slot code runs for it."""
    g = _release_gen(tmp_path, monkeypatch, rows)
    called = []
    monkeypatch.setattr(W, "slot_context", lambda *a, **k: called.append(1))
    g.env.clock.t = T_MON - 29 * 3600
    await g.crown(1, "c1/champ", .60)
    g.seat(1)
    g.add(3, "n3/e1", .90)
    g.add(5, "n5/e2", .95)
    await g.cycle()
    await g.cycle(13 * 3600)
    await g.cycle(T_MON - g.env.clock.t)
    rec = g.ts.slot_results[-1]
    assert rec["slot"] == T_MON and rec["evaluation"] == "v4" and rec["eval_ts"] == T_MON - 1 and "pool_version" not in rec
    assert not called and g.ts.champion in (g.sub(3).key, g.sub(5).key)
    assert all(len(g.ts.draws[d].item_ids) == 900 for d in rec["draws"])          # v4 draws: 90 per category


# ---------------------------------------------------------------------------------------------------- K1-a on v5.1
async def test_k1a_confirmation_runs_on_v51_and_joins_the_thursday_slot(tmp_path, monkeypatch, rows):
    g = _release_gen(tmp_path, monkeypatch, rows)
    await _v5_confirmed(g, {3: ("n3/a", .95)})
    g.env.clock.t = T_THU - 13.5 * 3600 - 60                         # first stage Thu 00:59: due 12:59, before 13:30
    g.add(5, "n5/b", .96)
    await g.cycle()
    late = g.sub(5)
    assert late.status == ST.PENDING_CONFIRM and late.confirm_after_ts <= T_THU - 3600
    g.env.clock.t = T_THU + 60                                       # our queue did not run it before the slot
    await g.cycle()
    assert not g.ts.slot_results and g.ts.champion == g.sub(1).key   # the slot waited for the K1-a confirmation
    conf = [h for h in late.history if h["kind"] == "confirm"][-1]
    assert conf["scoring_regime"] == "v5"                            # on the slot's evaluation: v5.1
    pend = [h for h in late.history if h["kind"] == "crown_pending"][-1]
    assert pend["k1a"] and pend["slot"] == T_THU
    await g.cycle(60)
    rec = g.ts.slot_results[-1]
    assert rec["evaluation"] == "v5.1" and {e["uid"]: e["in_by"] for e in rec["entries"]} == {3: "confirmed", 5: "k1a"}


async def test_k1a_on_a_v51_stage_is_never_confirmed_on_v4(tmp_path, monkeypatch, rows):
    """The flip-slot v4 path (critic M1, tests/test_k1a_flip_slot.py) applies to v4 first stages only: a K1-a entry staged on
    v5.1 confirms on v5.1 even if the slot's evaluation were v4 (no reverse hybrid)."""
    g = _release_gen(tmp_path, monkeypatch, rows)
    await _v5_confirmed(g, {3: ("n3/a", .95)})
    g.env.clock.t = T_THU - 13.5 * 3600 - 60
    g.add(5, "n5/b", .96)
    await g.cycle()
    g.env.clock.t = T_THU + 60
    monkeypatch.setattr(E.SpecEvaluator, "slot_eval_ts", lambda self_, slot: T_MON - 1)     # as if the slot were on v4
    await g.cycle()
    conf = [h for h in g.sub(5).history if h["kind"] == "confirm"]
    assert conf and all(h["scoring_regime"] == "v5" for h in conf)


# ----------------------------------------------------------------------------------------- void round 1 at a v5.1 slot
async def test_v51_slot_round_one_every_member_failing_is_void_and_re_run(tmp_path, monkeypatch, rows, caplog):
    """Slot rules section 7 (critic S4) on a v5.1 slot too: a round 1 in which every member fails on its own side is void and
    re-run with the same entries; nobody is released, also after the slot clock (an error is logged for a human)."""
    g = _release_gen(tmp_path, monkeypatch, rows)
    await _v5_confirmed(g, {3: ("n3/a", .95), 5: ("n5/b", .90)})
    real = E.SpecEvaluator._serve_v5

    async def fail(self_, p, *a, **k):
        return None if p.sub.key in (g.sub(3).key, g.sub(5).key) else await real(self_, p, *a, **k)

    monkeypatch.setattr(E.SpecEvaluator, "_serve_v5", fail)
    await g.cycle(T_THU - g.env.clock.t)
    assert not g.ts.slot_results and any("void" in e for e in g.ts.events)
    g.env.clock.t = T_THU + CS.SLOT_CLOCK_S + 60
    with caplog.at_level("ERROR"):
        await g.cycle()
    assert not g.ts.slot_results and g.ts.champion == g.sub(1).key
    assert all(E._crown_pending(g.sub(u)) is not None for u in (3, 5))
    assert any("still void after" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(E.SpecEvaluator, "_serve_v5", real)         # our side works again: the slot is decided on v5.1
    await g.cycle(3600)
    rec = g.ts.slot_results[-1]
    assert rec["evaluation"] == "v5.1" and rec["result"] != "none" and rec["crowned"]


# ---------------------------------------------------------------------------------------------- fail closed at the slot
@pytest.mark.parametrize("broken", ["no_pool_dir", "no_config"])
async def test_v5_slot_fails_closed_without_pool_or_panel(tmp_path, monkeypatch, rows, caplog, broken):
    g = _release_gen(tmp_path, monkeypatch, rows)
    await _v5_confirmed(g, {3: ("n3/a", .95), 5: ("n5/b", .70)})
    pool_dir = g.env.cfg.v5_pool_dir
    if broken == "no_pool_dir":
        g.env.cfg.v5_pool_dir = ""
    else:
        os.rename(os.path.join(pool_dir, W.CONFIG_FILE), os.path.join(pool_dir, "moved.json"))
    caplog.set_level(logging.ERROR, logger="reax.spec")
    g.env.clock.t = T_THU
    g.env.server.served.clear()
    await g.cycle()
    for t in (T_THU + 3600, T_THU + 13 * 3600):                      # within the slot clock and after it: still nobody crowned
        g.env.clock.t = t
        await g.cycle()
        assert not g.ts.slot_results and g.ts.champion == g.sub(1).key and g.sub(1).status == ST.CHAMPION
        assert all(E._crown_pending(g.sub(u)) is not None for u in (3, 5))      # nobody removed for our failure
    assert not g.env.server.served                                   # nothing served: the draw is refused first
    assert any("FAIL CLOSED" in r.getMessage() for r in caplog.records)
    assert any("no slot round draw" in e and "our side" in e for e in g.ts.events)
    g.env.cfg.v5_pool_dir = pool_dir                                 # the pool is back: the slot is run on v5.1
    if broken == "no_config":
        os.rename(os.path.join(pool_dir, "moved.json"), os.path.join(pool_dir, W.CONFIG_FILE))
    await g.cycle(3600)
    rec = g.ts.slot_results[-1]
    assert rec["slot"] == T_THU and rec["evaluation"] == "v5.1" and g.ts.champion == g.sub(3).key


async def test_v4_crowning_in_a_v51_slot_restarts_on_v51(tmp_path, monkeypatch, rows):
    """Rule A at a v5.1 slot: a stored crowning earned under v4 is not measured on v5.1; it restarts as a v5.1 first stage."""
    g = _release_gen(tmp_path, monkeypatch, rows)
    await _v5_confirmed(g, {3: ("n3/a", .95), 5: ("n5/b", .96)})
    old = g.sub(5)
    for h in old.history:
        h.pop("scoring_regime", None)                                # as if its confirmation had been measured on v4
    calls = []
    monkeypatch.setattr(E.SpecEvaluator, "_prepare_cycle", lambda *a, **k: calls.append(a[2].uid) or _none())
    await g.env.ev._run_playoff(g.env.cfg.tracks[-1], g.ts, [g.sub(3), old], None, T_THU + 10, 10, slot=True)
    assert old.status == ST.QUEUED and old.history[-1]["kind"] == "v5_restart" and E._crown_pending(old) is None
    assert calls == [3]


async def _none():
    return None


# ------------------------------------------------------------------- CRITERIA 6: the group outlives its leader
async def _shared_group(tmp_path, monkeypatch, uids=(2, 3, 4)):
    from test_slot_rules import _script, _slot
    _script(monkeypatch, {u: [74.5] for u in uids})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, list(uids))
    await env.cycle(allc)
    assert ts.slot_results[-1]["result"] == "shared_group" and len(ts.cochamps) == len(uids)
    return env, T, allc, ts


async def test_group_stays_champion_when_its_leader_withdraws(tmp_path, monkeypatch):
    env, T, allc, ts = await _shared_group(tmp_path, monkeypatch)
    keys = list(ts.cochamps)
    leader = ts.submissions[keys[0]]
    assert ts.champion == leader.key
    allc.pop(leader.uid)                                             # the leader's hotkey leaves
    lead = (await env.cycle(allc))[T]
    assert leader.status == ST.WITHDRAWN and ts.cochamps == keys     # the group keeps its keys: 1/3 each stays 1/3
    new = ts.submissions[ts.champion]
    assert new.key == keys[1] and new.status == ST.CHAMPION and new.champion_since_ts == ts.cochamps_since
    assert any(h["kind"] == "cochamp_reanchor" for h in new.history)
    assert [m.key for m in E.cochamp_members(ts)] == keys[1:]
    others = {ts.submissions[k].uid for k in keys[1:]}
    assert lead.cochamps == {u: pytest.approx(1 / 3) for u in others} and lead.cochamps_dynamic   # the leader's part: inference
    assert lead[0][0] in others and leader.uid not in [u for u, _ in lead]
    assert lead.crowned[lead[0][0]] == ts.cochamps_since
    # the next member leaves too; then the last one: vacant only now
    allc.pop(new.uid)
    await env.cycle(allc)
    assert ts.champion == keys[2] and ts.cochamps == keys
    allc.pop(ts.submissions[keys[2]].uid)
    lead = (await env.cycle(allc))[T]
    assert ts.champion == "" and ts.cochamps == [] and not lead.cochamps
    assert any("the title is vacant" in e for e in ts.events)


async def test_group_stays_champion_when_its_leader_is_rejected(tmp_path, monkeypatch):
    env, T, allc, ts = await _shared_group(tmp_path, monkeypatch, uids=(2, 3))
    keys = list(ts.cochamps)
    leader = ts.submissions[keys[0]]
    leader.status, leader.reason, ts.champion = ST.REJECTED, "identical tensors to an earlier commitment", ""
    # validator side, before any worker cycle re-anchors the seat: the remaining member still holds the group's rank 1
    snap_leads = E.compute_leaders(env.ev.state, [T], env_snapshot(allc), now=env.clock())
    assert snap_leads[T].cochamps == {ts.submissions[keys[1]].uid: pytest.approx(0.5)} and snap_leads[T].cochamps_dynamic
    await env.cycle(allc)
    assert ts.champion == keys[1] and ts.submissions[keys[1]].status == ST.CHAMPION and ts.cochamps == keys
    # a challenger who beats the re-anchored group retires the whole group as ONE former champion
    env.ev._retire_champions(ts, ts.submissions[ts.champion])
    assert ts.former_sets[-1]["keys"] == keys and ts.cochamps == []


def env_snapshot(allc):
    from spec_helpers import commit, neuron, snapshot
    from test_spec_evaluator import TRACK
    return snapshot(1000, [neuron(u, commit(TRACK, r)) for u, r in allc.items()])


def test_opening_set_keeps_the_rc11_rule_when_its_champion_leaves(monkeypatch):
    """Only a group crowned at a slot outlives its leader; the hard-coded opening set is unchanged (payouts before Mon)."""
    ts = ST.TrackState("generalist")
    for k in ("a", "b"):
        ts.submissions[k] = ST.Submission(key=k, uid=1 if k == "a" else 2, hotkey=k, coldkey=k, track_id="generalist",
                                          repo=f"{k}/m", revision="r" * 40, first_seen_block=1, first_seen_ts=0.0)
    ts.submissions["b"].status, ts.submissions["b"].delta_checked = ST.PENDING_CONFIRM, True
    monkeypatch.setattr(CS, "GENERALIST_CO_CHAMPIONS", ("a", "b"))
    ts.champion = ""
    assert E.cochamp_members(ts) == []
    E.SpecEvaluator._reanchor_cochamps(ts, 0.0)
    assert ts.champion == "" and ts.submissions["b"].status == ST.PENDING_CONFIRM


async def test_v51_slot_rounds_end_early_instead_of_reusing_items(tmp_path, monkeypatch):
    """SLOT-RULES s.4 'rounds share no items' (post4 critic S2): when the pool has no disjoint draw for the next round, the rounds
    end there and the slot is decided on the completed rounds; items are never reused within a slot."""
    rows = v51_rows(per_type=130)          # two disjoint 900-item rounds fit (60 N items per task type a round), a third does not
    g = _release_gen(tmp_path, monkeypatch, rows)
    await _v5_confirmed(g, {3: ("n3/strong", .95), 6: ("n6/close", .949)})
    await g.cycle(T_THU - g.env.clock.t)
    rec = g.ts.slot_results[-1]
    assert rec["evaluation"] == "v5.1" and rec["rounds"] < 3
    pool = {r["id"] for r in [*rows[0], *rows[1]]}
    seen = [set(g.ts.draws[d].item_ids) & pool for d in rec["draws"]]
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1:])
    assert not any("least-used items taken" in e for e in g.ts.events)
    assert any("rounds end early" in e for e in g.ts.events)
