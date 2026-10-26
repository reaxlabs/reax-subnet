"""rc11.post5 (board #25186/#25193): a vacant generalist track on eval v5.1 is crowned at the next crowning slot from the
entries that confirmed on v5.1, on the same slot machinery as an occupied v5.1 slot (rounds 1-3, pinned ``slot_eval_ts``).
Before: a v5.1 confirmation on a vacant track waited for the cohort playoff, which fails closed under v5.1 (the track stayed
uncrowned and blocked); a lone one was crowned at once, outside the slots. Fake models and the fixture v5.1 pool only."""
from __future__ import annotations

import pytest
from test_slot_v5 import T_MON, T_THU, _release_gen, _shared_group
from test_v5_wiring import v51_rows

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST
from reax_subnet.spec import v5wire as W


@pytest.fixture(scope="module")
def deep_rows():
    return v51_rows(per_type=260)         # three disjoint 900-item rounds fit


async def _vacant_after_flip(g):
    """Champion uid 1 crowned under v4 before the gate and seated alone; at Mon 15:30Z (after the flip) it recommits from the
    same hotkey: the old entry is withdrawn, the track is vacant, the recommit is a new (weak) entry."""
    g.env.clock.t = T_MON - 29 * 3600
    await g.crown(1, "c1/champ", .60)
    g.seat(1)
    g.env.clock.t = T_MON + 3600
    g.add(1, "c1/recommit", .30)                                     # below its base (.45): never better
    await g.cycle()
    old = next(s for s in g.ts.submissions.values() if s.repo == "c1/champ")
    assert old.status == ST.WITHDRAWN and g.ts.champion == "" and not E.cochamp_members(g.ts)


async def test_vacant_v51_track_crowns_the_best_at_the_next_slot(tmp_path, monkeypatch, deep_rows):
    """(a) The sole champion withdraws after the flip; two entries confirm on v5.1 (against their declared base): both wait
    for the Thursday slot (no cohort playoff, no ad hoc crowning) and the v5.1 slot rounds crown the best by the margin."""
    g = _release_gen(tmp_path, monkeypatch, deep_rows)
    await _vacant_after_flip(g)
    g.add(3, "n3/strong", .95)
    g.add(5, "n5/weak", .70)
    await g.cycle()
    await g.cycle(13 * 3600)                                         # Tue 05:30Z: both confirm on v5.1
    for u in (3, 5):
        s = g.sub(u)
        assert E._crown_pending(s) is not None and E._decided_under_v5(s), (u, s.status, s.reason)
        assert E.SpecEvaluator._slot_of(s) == T_THU and not s.confirm_draw
        assert s.first_incumbent is not None and "takes effect at 2026-10-29 14:30Z" in s.reason
    assert g.ts.champion == "" and not any("not wired" in e for e in g.ts.events)
    await g.cycle(T_THU - 1 - g.env.clock.t)                         # Thu 14:29:59: still nobody
    assert g.ts.champion == "" and not g.ts.slot_results
    await g.cycle(1)                                                 # Thu 14:30:00: the slot, on v5.1
    rec = g.ts.slot_results[-1]
    assert rec["slot"] == T_THU and rec["eval_ts"] == T_THU - 1 and rec["evaluation"] == "v5.1"
    assert rec["margin_floor"] == pytest.approx(1.0) and rec["g"] == pytest.approx(0.5)
    weak = next(e for e in rec["entries"] if e["uid"] == 5)
    assert weak["left"]["rule"] == "4.5" and weak["gap"] >= weak["margin"] >= 1.0      # separated by the margin
    strong = g.sub(3)
    assert rec["crowned"] == [3] and g.ts.champion == strong.key and strong.status == ST.CHAMPION
    row = [h for h in strong.history if h["kind"] == "playoff" and h.get("slot")][-1]
    assert row["scoring_regime"] == "v5" and W.scored_under_v5(strong.by_tier)
    assert g.sub(5).status == ST.QUEUED and not g.sub(5).draw1      # a new first draw against the new champion
    assert not any("not wired" in e for e in g.ts.events)


async def test_vacant_v51_track_lone_confirmation_waits_for_the_slot(tmp_path, monkeypatch, deep_rows):
    """One v5.1 confirmation on a vacant track is not crowned ad hoc: it waits for the slot and is crowned there alone."""
    g = _release_gen(tmp_path, monkeypatch, deep_rows)
    await _vacant_after_flip(g)
    g.add(3, "n3/only", .90)
    await g.cycle()
    await g.cycle(13 * 3600)
    s = g.sub(3)
    assert E._crown_pending(s) is not None and s.status == ST.PENDING_CONFIRM and g.ts.champion == ""
    await g.cycle(3600)
    assert g.ts.champion == ""                                       # still no crowning between the slots
    await g.cycle(T_THU - g.env.clock.t)
    assert g.ts.champion == s.key and s.status == ST.CHAMPION
    assert g.ts.slot_results[-1]["result"] == "single_entry" and g.ts.slot_results[-1]["evaluation"] == "v5.1"


async def test_vacant_v51_track_only_v4_decided_entries_restart_on_v51(tmp_path, monkeypatch, deep_rows):
    """(b) Entries that confirmed under v4 on a vacant track and wait for the cohort playoff when the gate opens: nothing is
    crowned from them (no v4 crowning after the flip, rule A); each restarts as a v5.1 first stage, and the cohort playoff
    (not wired for v5.1) never runs."""
    g = _release_gen(tmp_path, monkeypatch, deep_rows)
    g.env.clock.t = T_MON - 29 * 3600
    await g.crown(1, "c1/champ", .60)
    g.seat(1)
    g.env.clock.t = T_MON - 20 * 3600
    for u, repo, skill in ((3, "n3/a", .90), (5, "n5/b", .95)):
        g.add(u, repo, skill)
    await g.cycle()
    for u in (3, 5):
        g.sub(u).first_incumbent = ""                                # as if the track had been vacant at their first draw
    g.ts.champion, g.sub(1).status = "", ST.WITHDRAWN               # the champion left before their confirmations
    g.env.clock.t = T_MON - 6 * 3600
    for u in (3, 5):                                                 # their v4 confirmations, on a vacant track
        s = g.sub(u)
        s.confirm_draw, s.confirm_score = "d-v4-" + str(u), .9
        s.history.append({"draw": s.confirm_draw, "kind": "confirm", "score": .9})
    ran = []
    monkeypatch.setattr(E.SpecEvaluator, "_run_playoff", lambda *a, **k: ran.append(1) or _none())
    g.env.clock.t = T_MON + 3600                                     # after the flip
    ev, ts = g.env.ev, g.ts
    ev._v5_restart_v4_stages(ts, g.env.clock.t)
    for u in (3, 5):
        s = g.sub(u)
        assert s.status == ST.QUEUED and not s.confirm_draw and s.history[-1]["kind"] == "v5_restart", (u, s.status)
    assert ev._cohort_step(ts, g.env.clock.t) == [] and ts.champion == "" and not ran
    # a vacant-track confirmation taken on v5.1 is left alone by rule A
    s = g.sub(3)
    s.status, s.confirm_draw = ST.PENDING_CONFIRM, "d-v5"
    s.history.append({"draw": "d-v5", "kind": "confirm", "score": .9, "scoring_regime": W.REGIME})
    ev._v5_restart_v4_stages(ts, g.env.clock.t)
    assert s.status == ST.PENDING_CONFIRM and s.confirm_draw == "d-v5"


async def _none():
    return None


async def test_shared_group_with_a_leaver_is_unchanged(tmp_path, monkeypatch):
    """(c) A shared group whose leader leaves keeps the title (``_reanchor_cochamps``); the vacant-track slot path is not
    taken while a member remains."""
    env, T, allc, ts = await _shared_group(tmp_path, monkeypatch)
    keys = list(ts.cochamps)
    allc.pop(ts.submissions[keys[0]].uid)
    await env.cycle(allc)
    assert ts.champion == keys[1] and ts.cochamps == keys and ts.submissions[keys[1]].status == ST.CHAMPION


async def test_pre_flip_vacant_track_keeps_the_v4_cohort_playoff(tmp_path, monkeypatch, deep_rows):
    """(d) Before the flip (gate and slots set as released) a vacant track is unchanged: v4 confirmations wait for their
    cohort and the cohort playoff crowns at once, not at a slot; no crowning is stored for the Monday slot."""
    g = _release_gen(tmp_path, monkeypatch, deep_rows)
    g.env.clock.t = T_MON - 60 * 3600
    g.add(3, "n3/a", .95)
    g.add(5, "n5/b", .70)
    await g.cycle()
    await g.cycle(13 * 3600)
    await g.cycle(60)
    assert g.env.clock.t < T_MON
    assert g.ts.champion == g.sub(3).key and g.sub(3).status == ST.CHAMPION
    assert not g.ts.slot_results and not any(E._crown_pending(s) for s in g.ts.submissions.values())
    assert any(h["kind"] == "playoff" and not h.get("slot") for h in g.sub(3).history)       # the v4 cohort playoff
    assert not any(h.get("scoring_regime") == W.REGIME for s in g.ts.submissions.values() for h in s.history)


@pytest.mark.parametrize("source", ["hold", "flag"])
@pytest.mark.parametrize("when", ["before_confirm", "after_confirm"])
async def test_vacant_v51_review_excludes_entry_without_blocking_other_crown(
        tmp_path, monkeypatch, deep_rows, source, when):
    """Both review sources gate the vacant path and a previously stored slot decision."""
    from reax_subnet.spec import review as RV
    monkeypatch.delenv(RV.HOLD_ENV, raising=False)
    monkeypatch.delenv(RV.CLEAR_ENV, raising=False)
    g = _release_gen(tmp_path, monkeypatch, deep_rows)
    await _vacant_after_flip(g)
    g.add(3, "n3/reviewed", .95)
    g.add(5, "n5/eligible", .90)
    await g.cycle()
    s = g.sub(3)

    def review():
        if source == "hold":
            monkeypatch.setenv(RV.HOLD_ENV, f"{s.hotkey[:10]}@{s.revision[:8]}")
        else:
            RV.record(s, ["rescale_anticorrelated:test"], g.env.clock.t)
        assert RV.under_review(s)

    if when == "before_confirm":
        review()
    await g.cycle(13 * 3600)
    assert E._crown_pending(g.sub(5)) is not None
    if when == "after_confirm":
        assert E._crown_pending(s) is not None
        review()
    else:
        assert E._crown_pending(s) is None and not s.confirm_draw
    assert not E.SpecEvaluator._cohort_open(g.ts, s)
    await g.cycle(T_THU - g.env.clock.t)
    assert s.status != ST.CHAMPION
    assert g.ts.champion == g.sub(5).key
    assert g.ts.slot_results[-1]["crowned"] == [5]
