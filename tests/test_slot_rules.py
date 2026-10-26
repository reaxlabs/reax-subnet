"""rc11.post3: crowning slot rules (reax.co/competitions/faq/#slot-rules sections 1-9; CRITERIA v2 section 8.8)."""

from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as EV
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import _crown_pending, cochamp_members, compute_leaders
from spec_helpers import commit, neuron, snapshot


@pytest.fixture(autouse=True)
def _floor_one(monkeypatch):
    # the published numbers of the current (v4) generalist evaluation: margin floor 1.0, so G = 0.5
    monkeypatch.setattr(M, "margin_floor", lambda now=None: 1.0)


def _script(monkeypatch, means, ses=None):
    """Scripted pooled means: ``means[uid][r]`` after round r+1 (a shorter list repeats its last value); paired SE of a
    pair after round r+1 = ``ses[r]`` (default 0.75 -> margin 1.5)."""
    real = EV.SpecEvaluator._slot_totals
    rounds = {"n": 1}

    def totals(self, ts, tdef, draws):
        mean, per, pen, pdet = real(self, ts, tdef, draws)
        rounds["n"] = len(draws)
        for k in mean:
            uid = ts.submissions[k].uid
            if uid in means:
                seq = means[uid]
                mean[k] = seq[min(len(draws), len(seq)) - 1]
        return mean, per, pen, pdet

    monkeypatch.setattr(EV.SpecEvaluator, "_slot_totals", totals)
    seq = ses or [0.75, 0.53, 0.45]
    monkeypatch.setattr(EV.SC, "paired_se", lambda *a, **k: seq[min(rounds["n"], len(seq)) - 1])


async def _slot(tmp_path, monkeypatch, uids, *, slot_in=20 * 3600, repos=None):
    """Champion uid 1; every uid in ``uids`` passes first draw + confirmation before the slot (distinct holders unless
    ``repos`` says otherwise). Returns at the slot instant + 1 h, before the slot cycle."""
    from test_spec_evaluator import TRACK as T, Env, crown
    env = Env(tmp_path, {})
    await crown(env, 1, "c1/champ", 0.65)
    allc = {1: "c1/champ"}
    for u in uids:
        repo = (repos or {}).get(u, f"n{u}/e{u}")
        env.add(u, repo, 0.95)
        allc[u] = repo
    slot = datetime.fromtimestamp(env.clock() + slot_in, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot, slot + timedelta(days=3)))
    await env.cycle(allc)
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    ts = env.ev.state.track(T)
    assert all(_crown_pending(env.sub(u)) is not None for u in uids), [env.sub(u).reason for u in uids]
    env.clock.advance(slot_in - 13 * 3600 + 3600)
    return env, T, allc, ts, slot


def _rec(ts):
    return ts.slot_results[-1]


def _entry(rec, uid):
    return next(e for e in rec["entries"] if e["uid"] == uid)


# ---------------------------------------------------------------- the published examples

async def test_example_clear_winner_after_round_two(tmp_path, monkeypatch):
    # A 75.40, B 74.20, D 73.60, E 72.90, margin about 1.50; after round 2 A 75.25, B 73.85, margin 1.06
    _script(monkeypatch, {2: [75.40, 75.25], 3: [74.20, 73.85], 4: [73.60], 5: [72.90]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3, 4, 5])
    env.server.served.clear()
    await env.cycle(allc)
    assert ts.champion == env.sub(2).key and env.sub(1).status == ST.FORMER and ts.cochamps == []
    rec = _rec(ts)
    assert rec["result"] == "clear_winner" and rec["rounds"] == 2 and rec["crowned"] == [2]
    assert _entry(rec, 4)["left"] == {"round": 1, "rule": "4.5", "why": _entry(rec, 4)["left"]["why"]}
    assert _entry(rec, 5)["left"]["rule"] == "4.5" and _entry(rec, 3)["left"] == {**_entry(rec, 3)["left"], "round": 2, "rule": "4.5"}
    assert _entry(rec, 3)["margin"] == pytest.approx(1.06) and _entry(rec, 3)["gap"] == pytest.approx(1.40)
    # round 2 served only A and B; the champion is never re-measured at the slot
    assert env.server.served.count("n2/e2") == 2 and env.server.served.count("n4/e4") == 1 and "c1/champ" not in env.server.served
    for u in (3, 4, 5):     # not crowned: no place, a new first draw against the new champion (slot rules section 2, rc11.post4)
        s = env.sub(u)
        assert s.status == ST.QUEUED and _crown_pending(s) is None and "a new first draw against the new champion" in s.reason
        assert not s.draw1 and s.confirm_after_ts is None and s.history[-1]["kind"] == "slot_restart"


async def test_example_practically_tied_shares(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0, 74.9, 74.78], 3: [74.6, 74.7, 74.71]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    await env.cycle(allc)
    rec = _rec(ts)
    assert rec["result"] == "shared_group" and rec["rounds"] == 3 and sorted(rec["crowned"]) == [2, 3]
    assert rec["g"] == pytest.approx(0.5) and rec["share_each"] == pytest.approx(0.5)
    assert ts.champion == env.sub(2).key and ts.cochamps == [env.sub(2).key, env.sub(3).key]
    assert [m.uid for m in cochamp_members(ts)] == [2, 3] and env.sub(3).status == ST.PENDING_CONFIRM
    assert env.sub(1).status == ST.FORMER and _crown_pending(env.sub(3)) is None
    # the validator: the group shares rank 1 (37.5 % of the 75 % ... i.e. half of rank 1 each), the old champion is place 2
    lead = (await env.cycle(allc))[T]
    assert [u for u, _ in lead][:2] == [2, 1] and lead.cochamps == {2: 0.5, 3: 0.5} and lead.cochamps_dynamic
    shares = CS.rank_shares(lead)
    assert shares[2] == pytest.approx(shares[3]) == pytest.approx(CS.TRACK_RANK_SPLIT[0] / 2)


async def test_group_leader_withdraws_the_group_stays_the_champion(tmp_path, monkeypatch):
    """CRITERIA v2 section 6 (critic rc11p3): the leader of a shared group recommits; its part goes to inference, the other
    member keeps the title (anchor of the group), its share stays 1/n, the beaten champion stays place 2."""
    _script(monkeypatch, {2: [75.0, 74.9, 74.78], 3: [74.6, 74.7, 74.71]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    await env.cycle(allc)
    assert ts.champion == env.sub(2).key and ts.cochamps == [env.sub(2).key, env.sub(3).key]
    since = ts.cochamps_since
    n_hist = len(env.sub(3).history)
    left = {u: r for u, r in allc.items() if u != 2}          # uid 2 no longer commits this entry
    lead = (await env.cycle(left))[T]
    assert env.sub(2).status == ST.WITHDRAWN and ts.champion == env.sub(3).key and env.sub(3).status == ST.CHAMPION
    assert ts.cochamps == [env.sub(2).key, env.sub(3).key] and ts.cochamps_since == since
    assert env.sub(3).champion_since_ts == since and not any(   # no re-measurement (on the generalist track under eval v5.1 the
        # vacant-track path would restart it as a v5.1 first stage, and v5.1 has no vacant-track playoff): it keeps the title
        h["kind"] in ("first", "confirm", "v5_restart") for h in env.sub(3).history[n_hist:])
    assert [m.uid for m in cochamp_members(ts)] == [3]
    assert [u for u, _ in lead][:2] == [3, 1] and lead.cochamps == {3: 0.5} and lead.cochamps_dynamic
    shares = CS.rank_shares(lead)
    assert shares[3] == pytest.approx(CS.TRACK_RANK_SPLIT[0] / 2) and 2 not in shares   # the leaver's half is not paid


async def test_example_not_separated_but_beyond_g_leader_alone(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0, 74.9, 74.87], 3: [74.6, 74.3, 74.10]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    await env.cycle(allc)
    rec = _rec(ts)
    assert rec["result"] == "leader_alone_beyond_g" and rec["crowned"] == [2] and rec["rounds"] == 3
    assert _entry(rec, 3)["left"]["rule"] == "5" and ts.cochamps == []
    assert any("not practically tied" in e for e in ts.events)


async def test_k8b_off_nobody_is_crowned(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "SLOT_K8B_LEADER_ALONE", False)
    _script(monkeypatch, {2: [75.0, 74.9, 74.87], 3: [74.6, 74.3, 74.10]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    await env.cycle(allc)
    assert _rec(ts)["result"] == "none_unresolved" and ts.champion == env.sub(1).key


async def test_exactly_equal_means_share_whatever_the_commit_order(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [74.5], 3: [74.5], 4: [74.5]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3, 4])
    await env.cycle(allc)
    rec = _rec(ts)
    assert rec["result"] == "shared_group" and sorted(rec["crowned"]) == [2, 3, 4] and rec["rounds"] == 3
    assert rec["share_each"] == pytest.approx(1 / 3)


async def test_entry_floor_at_or_below_leaves(tmp_path, monkeypatch):
    env0 = None
    _script(monkeypatch, {2: [75.0], 3: [74.8]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    monkeypatch.setattr(EV.SpecEvaluator, "_entry_floor", lambda self, tdef: 74.8)
    await env.cycle(allc)
    rec = _rec(ts)
    assert rec["result"] == "clear_winner" and _entry(rec, 3)["left"]["rule"] == "4.2" and rec["rounds"] == 1


async def test_one_holder_two_coldkeys_same_hf_account_keeps_the_best_round_one(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [74.0], 3: [74.9], 4: [74.95]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3, 4], repos={2: "team/a", 3: "Team/b", 4: "other/c"})
    assert env.sub(2).coldkey != env.sub(3).coldkey
    await env.cycle(allc)
    rec = _rec(ts)
    assert _entry(rec, 2)["left"]["rule"] == "4.4" and "uid 3" in _entry(rec, 2)["left"]["why"]
    assert rec["result"] == "shared_group" and sorted(rec["crowned"]) == [3, 4]


def test_holder_groups_are_transitive_and_coldkey_only_with_k4b(monkeypatch):
    mk = lambda uid, ck, repo: ST.Submission(key=f"k{uid}", uid=uid, hotkey=f"h{uid}", coldkey=ck, track_id="t", repo=repo,  # noqa: E731
                                             revision="r", first_seen_block=uid, first_seen_ts=0.0)
    a, b, c, d = mk(1, "c1", "x/m"), mk(2, "c1", "y/m"), mk(3, "c3", "Y/n"), mk(4, "c4", "z/m")
    groups = EV.SpecEvaluator._holder_groups([a, b, c, d])
    assert sorted(sorted(s.uid for s in g) for g in groups) == [[1, 2, 3], [4]]
    monkeypatch.setattr(CS, "SLOT_OWNER_COLDKEY_OR_HF", False)
    groups = EV.SpecEvaluator._holder_groups([a, b, c, d])
    assert sorted(sorted(s.uid for s in g) for g in groups) == [[1, 2], [3], [4]]


# ---------------------------------------------------------------- serving, void rounds, clock

def _failing(env, fail):
    """Backend factory that raises for the repos ``fail(repo, n)`` names (n = that repo's serve count so far)."""
    from reax_subnet.spec.vllm import HostUnavailable, ModelUnavailable
    real = env.ev.backend_factory
    count: dict[str, int] = {}

    def factory(model_dir, weight_bytes):
        import os
        repo = open(os.path.join(model_dir, "skill_key.txt")).read()
        count[repo] = count.get(repo, 0) + 1
        kind = fail(repo, count[repo])
        if kind == "model":
            raise ModelUnavailable("model refused")
        if kind == "infra":
            raise HostUnavailable("host down")
        return real(model_dir, weight_bytes)

    env.ev.backend_factory = factory
    return count


async def test_member_failing_twice_leaves_others_continue(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0, 74.9, 74.8], 3: [74.9, 74.8, 74.7], 4: [74.0]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3, 4])
    count = _failing(env, lambda repo, n: "model" if repo == "n3/e3" and n in (2, 3) else None)
    await env.cycle(allc)
    rec = _rec(ts)
    assert count["n3/e3"] == 3                      # round 1 ok, round 2 failed + one retry on the same items
    assert _entry(rec, 3)["left"] == {"round": 2, "rule": "4.1", "why": "model failed twice on the round's items"}
    assert rec["crowned"] == [2] or sorted(rec["crowned"]) == [2, 4]


async def test_member_failing_once_is_retried_on_the_same_items(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0], 3: [70.0]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    count = _failing(env, lambda repo, n: "model" if repo == "n3/e3" and n == 1 else None)
    await env.cycle(allc)
    rec = _rec(ts)
    assert count["n3/e3"] == 2 and _entry(rec, 3)["served"] and _entry(rec, 3)["left"]["rule"] == "4.5"


async def test_void_round_is_rerun_on_new_items_and_counts_for_nobody(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0, 74.9, 74.8], 3: [74.9, 74.8, 74.7]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    count = _failing(env, lambda repo, n: "infra" if n == 2 else None)     # round 2: host down for everyone
    await env.cycle(allc)
    rec = _rec(ts)
    assert rec["void_rounds"] == 1 and rec["rounds"] == 3 and len(set(rec["draws"])) == 3
    assert rec["result"] == "shared_group" and any("void" in e for e in ts.events)
    assert count["n2/e2"] == 4                       # rounds 1, 2 (void), 2 again, 3
    assert env.sub(2).status == ST.CHAMPION and env.sub(3).status == ST.PENDING_CONFIRM   # nobody punished


async def test_void_round_one_waits_for_the_next_cycle(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0], 3: [70.0]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    _failing(env, lambda repo, n: "infra" if n == 1 else None)
    await env.cycle(allc)
    assert ts.slot_results == [] and ts.champion == env.sub(1).key and _crown_pending(env.sub(2)) is not None
    env.clock.advance(3600)
    await env.cycle(allc)
    assert _rec(ts)["result"] == "clear_winner" and ts.champion == env.sub(2).key


async def test_clock_runs_out_after_round_two_decided_on_two_rounds(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0, 74.9], 3: [74.9, 74.6]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    real = EV.SpecEvaluator._slot_serve_round

    calls = {"n": 0}

    async def serve(self, ts_, live, rows, now, **kw):
        out = await real(self, ts_, live, rows, now, **kw)
        calls["n"] += 1
        if calls["n"] == 2:
            env.clock.advance(12 * 3600)    # round 2 ends after the slot clock
        return out

    monkeypatch.setattr(EV.SpecEvaluator, "_slot_serve_round", serve)
    await env.cycle(allc)
    rec = _rec(ts)
    assert rec["rounds"] == 2 and "clock" in rec["ended_early"] and rec["result"] == "shared_group"   # gap 0.3 <= G


async def test_all_entries_unloadable_nobody_crowned_track_continues(tmp_path, monkeypatch):
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])

    async def nope(self, tdef, sub, snap, now):
        sub.wait_kind, sub.reason = ST.WAIT_MINER, "repo gone"
        return None

    monkeypatch.setattr(EV.SpecEvaluator, "_prepare_cycle", nope)
    await env.cycle(allc)
    assert _rec(ts)["result"] == "none" and ts.champion == env.sub(1).key
    assert _crown_pending(env.sub(2)) is None and _crown_pending(env.sub(3)) is None


# ---------------------------------------------------------------- dethroning a group, former group, keep-share

async def test_group_dethroned_via_best_member_then_one_former_champion(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.0], 3: [74.9]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    await env.cycle(allc)
    assert ts.cochamps == [env.sub(2).key, env.sub(3).key]
    monkeypatch.undo()                                   # real scores from here on
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", ())
    env.add(6, "n6/strong", 0.999)
    allc[6] = "n6/strong"
    env.server.served.clear()
    await env.cycle(allc)
    assert {"n2/e2", "n3/e3", "n6/strong"} <= set(env.server.served)      # every member is served as the bar
    h = [x for x in env.sub(6).history if x["kind"] == "first"][-1]
    assert set(h["bar_scores"]) == {"2", "3"} and h["incumbent"] == pytest.approx(max(h["bar_scores"].values()))
    env.clock.advance(13 * 3600)
    await env.cycle(allc)
    assert ts.champion == env.sub(6).key, env.sub(6).reason
    assert env.sub(2).status == env.sub(3).status == ST.FORMER and ts.cochamps == []
    assert ts.former_sets and ts.former_sets[-1]["keys"] == [env.sub(2).key, env.sub(3).key]
    lead = (await env.cycle(allc))[T]
    # rank 1 = 6, place 2 = the group (ONE former champion, uid 2 stands for it), place 3 = the old champion uid 1
    assert [u for u, _ in lead] == [6, 2, 1] and lead.formsets == [{2: 0.5, 3: 0.5}]
    shares = CS.rank_shares(lead)
    assert shares[2] == pytest.approx(shares[3]) == pytest.approx(CS.TRACK_RANK_SPLIT[1] / 2)
    assert shares[1] == pytest.approx(CS.TRACK_RANK_SPLIT[2])
    # the state round trip keeps the group
    st2 = ST._from_json(ST._to_json(env.ev.state))
    assert st2.track(T).former_sets == ts.former_sets


def _ranked(order, crowned, cochamps=None, formsets=(), dynamic=True):
    r = CS.RankedLeaders([(u, 70.0) for u in order])
    r.crowned = dict(crowned)
    r.cochamps = dict(cochamps or {})
    r.cochamps_dynamic = dynamic
    r.formsets = [dict(f) for f in formsets]
    return r


def test_keep_share_one_unapproved_group_member_pro_rata():
    now = 1_000_000.0
    lead = _ranked([2, 1], {2: now - 3600}, cochamps={2: 0.5, 3: 0.5})
    ranks = CS.rank_shares(lead)
    out = CS.keep_share_set_ranks(ranks, lead, None, now, lambda u: u != 3)
    r1 = CS.TRACK_RANK_SPLIT[0]
    # uid 3 is not approved: its half of rank 1 goes to the beaten champion uid 1, instead of uid 1's own place share
    assert out[2] == pytest.approx(r1 / 2) and out[1] == pytest.approx(r1 / 2)
    after = CS.keep_share_set_ranks(ranks, lead, None, now + CS.APPROVAL_GRACE_S, lambda u: u != 3)
    assert after == ranks                                # after 48 h: the beaten champion is paid its place again


def test_keep_share_beaten_set_receives_in_equal_parts():
    now = 1_000_000.0
    lead = _ranked([9, 2], {9: now - 60}, formsets=[{2: 0.5, 3: 0.5}])
    ranks = CS.rank_shares(lead)
    assert ranks[2] == pytest.approx(CS.TRACK_RANK_SPLIT[1] / 2)
    out = CS.keep_share_set_ranks(ranks, lead, None, now, lambda u: u != 9)
    r1 = CS.TRACK_RANK_SPLIT[0]
    assert out[2] == pytest.approx(r1 / 2) and out[3] == pytest.approx(r1 / 2) and out[9] == pytest.approx(r1)
    # the old rc11.post2 path gave the whole 75 % to one member; the combine_weights path now uses the set rule
    w = CS.combine_weights({50: 1.0}, None, 0.0, {"generalist": lead}, award_ok=lambda t, u: u != 9, keep_share=True, now=now)
    assert w[2] == pytest.approx(w[3]) and w.get(9, 0.0) == 0.0


def test_opening_set_at_rank_one_has_no_keep_share_as_in_post2():
    now = 1_000_000.0
    lead = _ranked([2, 1], {2: now - 60}, cochamps={2: 1 / 3, 3: 1 / 3, 4: 1 / 3}, dynamic=False)
    a = CS.combine_weights({50: 1.0}, None, 0.0, {"generalist": lead}, award_ok=lambda t, u: u != 3, keep_share=True, now=now)
    b = CS.combine_weights({50: 1.0}, None, 0.0, {"generalist": lead}, award_ok=lambda t, u: u != 3, keep_share=False, now=now)
    assert a == b


# ---------------------------------------------------------------- K1-a

async def test_k1a_confirmation_due_before_the_slot_runs_first_and_joins_it(tmp_path, monkeypatch):
    """uid 3's confirmation was due at slot - 2 h but our queue had not run it by the slot: it runs before the slot rounds,
    on the same evaluation, passes, and is in this slot (not the next)."""
    from test_spec_evaluator import Env, crown
    _script(monkeypatch, {2: [75.0], 3: [74.9]})
    env = Env(tmp_path, {})
    await crown(env, 1, "c1/champ", 0.65)
    t0 = env.clock()
    slot = datetime.fromtimestamp(t0 + 16 * 3600, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot, slot + timedelta(days=3)))
    allc = {1: "c1/champ", 2: "n2/e2"}
    env.add(2, "n2/e2", 0.95)
    await env.cycle(allc)                                  # uid 2: first pass at t0
    env.clock.advance(2 * 3600)
    env.add(3, "n3/e3", 0.95)
    allc[3] = "n3/e3"
    await env.cycle(allc)                                  # uid 3: first pass at t0 + 2 h, due at slot - 2 h
    env.clock.advance(11 * 3600)
    await env.cycle(allc)                                  # uid 2 confirmed (t0 + 13 h), stored for the slot
    assert _crown_pending(env.sub(2)) is not None
    T = next(iter(env.ev.state.tracks))
    ts = env.ev.state.track(T)
    assert env.sub(3).status == ST.PENDING_CONFIRM and env.sub(3).confirm_after_ts <= slot.timestamp() - 3600
    env.clock.t = slot.timestamp() + 600                   # the queue did not run it before the slot
    assert [s.uid for s in env.ev._k1a_outstanding(ts, env.clock())] == [3]
    await env.cycle(allc)                                  # its confirmation runs first, the slot waits
    rows = [h for h in env.sub(3).history if h.get("kind") == "crown_pending"]
    assert rows and rows[-1]["k1a"] and rows[-1]["slot"] == slot.timestamp()
    if not ts.slot_results:
        env.clock.advance(60)
        await env.cycle(allc)
    rec = _rec(ts)
    assert rec["slot"] == slot.timestamp() and {e["uid"] for e in rec["entries"]} == {2, 3}
    assert _entry(rec, 3)["in_by"] == "k1a" and _entry(rec, 2)["in_by"] == "confirmed"


async def test_k1b_late_confirmation_goes_to_the_next_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "SLOT_K1_A", False)
    from test_spec_evaluator import Env, crown
    env = Env(tmp_path, {})
    await crown(env, 1, "c1/champ", 0.65)
    allc = {1: "c1/champ", 3: "n3/e3"}
    env.add(3, "n3/e3", 0.95)
    await env.cycle(allc)
    slot = datetime.fromtimestamp(env.clock() + 12 * 3600 + 7200, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot, slot + timedelta(days=3)))
    env.clock.t = slot.timestamp() + 600
    await env.cycle(allc)
    rows = [h for h in env.sub(3).history if h.get("kind") == "crown_pending"]
    assert rows and "k1a" not in rows[-1] and rows[-1]["slot"] == (slot + timedelta(days=3)).timestamp()


# ---------------------------------------------------------------- state compatibility

def test_state_round_trip_old_new_and_post2_loader(tmp_path):
    import importlib.util
    import json
    import subprocess
    st = ST.SpecState()
    ts = st.track("generalist")
    raw_old = ST._to_json(st)
    for k in ("cochamps", "cochamps_since", "former_sets", "slot_results"):
        raw_old["tracks"]["generalist"].pop(k)
    assert ST._from_json(raw_old).track("generalist").cochamps == []          # an old file loads unchanged
    ts.cochamps, ts.cochamps_since = ["a|x", "b|y"], 5.0
    ts.former_sets = [{"keys": ["c|z"], "since": 4.0}]
    ts.slot_results = [{"slot": 1.0, "result": "shared_group"}]
    raw = ST._to_json(st)
    back = ST._from_json(json.loads(json.dumps(raw))).track("generalist")
    assert (back.cochamps, back.cochamps_since, back.former_sets, back.slot_results) == (
        ts.cochamps, ts.cochamps_since, ts.former_sets, ts.slot_results)
    # rc11.post2's loader (the rollback image) reads the new file without crashing
    path = tmp_path / "state.json"
    path.write_text(json.dumps(raw))
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    src = subprocess.run(["git", "-C", str(repo), "show", "f1af7bd:src/reax_subnet/spec/state.py"], capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip("rc11.post2 commit not in this checkout")
    mod_path = tmp_path / "state_post2.py"
    mod_path.write_text(src.stdout.replace("from . import config as C", "from reax_subnet.spec import config as C"))
    import sys
    spec = importlib.util.spec_from_file_location("state_post2", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["state_post2"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("state_post2", None)
    old = mod.load(str(path))
    assert "generalist" in old.tracks and not hasattr(old.tracks["generalist"], "cochamps")


async def test_round_one_every_model_failing_stays_void_and_is_re_run_after_the_slot_clock(tmp_path, monkeypatch, caplog):
    """Slot rules section 7 (critic S4): a round where every entry fails is void and re-run with the same entries; nobody
    is released for it, also after the slot clock (then an error is logged for a human)."""
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    real = env.ev.backend_factory
    _failing(env, lambda repo, n: "model" if repo in ("n2/e2", "n3/e3") else None)
    await env.cycle(allc)
    assert ts.slot_results == [] and _crown_pending(env.sub(2)) is not None and any("void" in e for e in ts.events)
    env.clock.t = slot.timestamp() + CS.SLOT_CLOCK_S + 60
    with caplog.at_level("ERROR"):
        await env.cycle(allc)
    assert ts.slot_results == [] and ts.champion == env.sub(1).key
    assert _crown_pending(env.sub(2)) is not None and _crown_pending(env.sub(3)) is not None
    assert any("still void after" in r.getMessage() for r in caplog.records)
    env.ev.backend_factory = real           # our side works again: the same entries are served and the slot is decided
    env.clock.t += 3600
    await env.cycle(allc)
    assert ts.slot_results and _rec(ts)["result"] != "none"


# ---------------------------------------------------------------- rc11.post4: after a new champion, a new first draw (section 2)

def _to_next_slot(env, slot, uid):
    """``uid`` confirmed after this slot's cutoff: its stored crowning is for the next slot."""
    row = [h for h in env.sub(uid).history if h.get("kind") == "crown_pending"][-1]
    row["ts"] = row["slot"] = slot.timestamp() + 60


def _restarted_then_first(s, old_draw1):
    """Requeued once by the crowning (``slot_restart``), then a new first draw, never a confirmation of the old stage."""
    i = next(i for i, h in enumerate(s.history) if h.get("kind") == "slot_restart")
    after = [h for h in s.history[i + 1:] if h.get("kind") in ("first", "confirm")]
    assert after and after[0]["kind"] == "first" and after[0]["draw"] != old_draw1, after
    assert sum(h.get("kind") == "slot_restart" for h in s.history) == 1


def _fresh(s, why):
    return (s.status == ST.QUEUED and _crown_pending(s) is None and why in s.reason and not s.draw1 and not s.confirm_draw
            and s.confirm_after_ts is None and s.first_pass_ts is None and s.first_incumbent is None and not s.close_draws
            and s.history[-1]["kind"] == "slot_restart" and s.history[-1]["was"] == ST.PENDING_CONFIRM)


async def test_new_champion_requeues_slot_losers_and_later_slot_entries_for_a_new_first_draw(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.40, 75.25], 3: [74.20, 73.85], 4: [73.60]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3, 4, 5])
    _to_next_slot(env, slot, 5)
    before = {u: (env.sub(u).first_seen_block, len(env.sub(u).history)) for u in (3, 4, 5)}
    old = {u: env.sub(u).draw1 for u in (3, 4, 5)}
    await env.cycle(allc)
    rec = _rec(ts)
    assert rec["crowned"] == [2] and ts.champion == env.sub(2).key and sorted(e["uid"] for e in rec["entries"]) == [2, 3, 4]
    for u in (3, 4):        # in the slot, not crowned
        assert _fresh(env.sub(u), "a new first draw against the new champion"), env.sub(u)
    assert _fresh(env.sub(5), "a new champion was crowned at the slot: a new first draw against it")   # stored for the next slot
    for u in (3, 4, 5):     # history and commitment order are kept, nothing is recommitted
        assert env.sub(u).first_seen_block == before[u][0] and len(env.sub(u).history) > before[u][1]
    assert not any(h["kind"] == "v5_restart" for u in (3, 4, 5) for h in env.sub(u).history)
    # the next cycles: each one is measured again from a first draw (against the new champion), not confirmed on the old one
    for _ in range(3):
        env.clock.advance(3600)
        await env.cycle(allc)
    for u in (3, 4, 5):
        _restarted_then_first(env.sub(u), old[u])
    assert ts.champion == env.sub(2).key


async def test_single_entry_crowning_requeues_the_next_slots_entries(tmp_path, monkeypatch):
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3])
    _to_next_slot(env, slot, 3)
    old = env.sub(3).draw1
    await env.cycle(allc)
    assert _rec(ts)["result"] == "single_entry" and ts.champion == env.sub(2).key
    _restarted_then_first(env.sub(3), old)      # requeued, then (budget left) its new first draw in the same cycle


async def test_nobody_crowned_later_slot_entries_and_slot_entries_keep_their_first_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "SLOT_K8B_LEADER_ALONE", False)
    _script(monkeypatch, {2: [75.0, 74.9, 74.87], 3: [74.6, 74.3, 74.10]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3, 4])
    _to_next_slot(env, slot, 4)
    draw1 = {u: env.sub(u).draw1 for u in (2, 3, 4)}
    pending4 = _crown_pending(env.sub(4))
    await env.cycle(allc)
    assert _rec(ts)["result"] == "none_unresolved" and ts.champion == env.sub(1).key
    s4 = env.sub(4)     # waits for the next slot as before
    assert s4.status == ST.PENDING_CONFIRM and _crown_pending(s4) == pending4 and s4.draw1 == draw1[4]
    assert EV.SpecEvaluator._slot_of(s4) == (slot + timedelta(days=3)).timestamp()
    for u in (2, 3):    # the slot's own entries: back on the challenger path with their first draw (unchanged)
        s = env.sub(u)
        assert s.status == ST.PENDING_CONFIRM and _crown_pending(s) is None and s.draw1 == draw1[u]
        assert "confirms against the champion" in s.reason
    assert not any(h["kind"] == "slot_restart" for u in (2, 3, 4) for h in env.sub(u).history)
