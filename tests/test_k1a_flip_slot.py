"""rc11.post3 critic: K1-a at the flip slot (CROWN slot = EVAL_V5_START_UTC, Monday 26 Oct 14:30Z). An entry whose v4 confirmation
was due at or before slot - 1 h but had not run when the slot opened is confirmed on the slot's evaluation (v4, ``slot_eval_ts``)
before the slot rounds, joins the slot when it passes (slot rules section 2/3), and is not restarted on v5.1 by rule A first."""
from __future__ import annotations

from datetime import datetime, timezone

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E, sampling as S, state as ST
from test_v5_wiring import Gen, gen, rows  # noqa: F401 - the fake pool, the Gen harness and its fixtures
from test_v5_wiring3 import FAST_P


async def test_k1a_confirmation_due_before_the_flip_slot_runs_on_v4_and_joins_the_slot(gen, monkeypatch):
    g = gen
    await g.crown(1, "a/old", .60)
    g.seat(1)
    t0 = g.env.clock.t
    slot = datetime.fromtimestamp(t0 + 30 * 3600, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot,))
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", slot)
    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", lambda self_: FAST_P)
    g.add(3, "b/e1", .90)
    await g.cycle()                                   # e1: v4 first pass
    await g.cycle(13 * 3600)                          # e1: v4 confirmation, crowning stored for the slot
    g.add(5, "c/e2", .95)
    await g.cycle(3600)                               # e2: v4 first pass at t0 + 14 h, confirmation due t0 + 26 h <= slot - 1 h
    e1, e2 = g.sub(3), g.sub(5)
    assert E._crown_pending(e1) is not None and e2.status == ST.PENDING_CONFIRM
    assert e2.confirm_after_ts <= slot.timestamp() - CS.SLOT_K1_DUE_LEAD_S
    await g.cycle(slot.timestamp() + 60 - g.env.clock.t)   # the slot opened; our queue never ran e2's confirmation before it
    assert S.v5_active(g.env.clock.t)
    assert not any(h["kind"] == "v5_restart" for h in e2.history)                 # rule A does not restart a K1-a entry
    conf = [h for h in e2.history if h["kind"] == "confirm"]
    assert conf and conf[-1].get("scoring_regime") is None                         # confirmed on v4, not on v5.1
    assert g.ts.draws[conf[-1]["draw"]].item_ids and len(g.ts.draws[conf[-1]["draw"]].item_ids) == E.S.draw_size(
        g.env.cfg.sample_size, True, E.SpecEvaluator.slot_eval_ts(g.env.ev, slot.timestamp()))
    assert E._crown_pending(e2) is not None and E.SpecEvaluator._slot_of(e2) == slot.timestamp()   # joins this slot
    assert not g.ts.slot_results                                                   # the slot waited for it
    await g.cycle(600)
    rec = g.ts.slot_results[-1]
    assert rec["slot"] == slot.timestamp() and {e["uid"] for e in rec["entries"]} == {3, 5}
    assert rec["eval_ts"] == slot.timestamp() - 1 and rec["rounds"] >= 1           # one v4 slot playoff with both entries
    assert {e["in_by"] for e in rec["entries"] if e["uid"] == 5} == {"k1a"}
    # rc11.post4 (slot rules section 2): a slot entry not crowned (here possibly the K1-a one) is queued once for a new first
    # draw, which is on v5.1; rule A does not restart it again
    await g.cycle(3600)
    await g.cycle(3600)
    losers = [s for s in (e1, e2) if s.uid not in rec["crowned"]]
    assert losers or rec["result"] == "shared_group"
    for s in losers:
        assert [h["kind"] for h in s.history].count("slot_restart") == 1 and not any(h["kind"] == "v5_restart" for h in s.history)
        i = next(i for i, h in enumerate(s.history) if h["kind"] == "slot_restart")
        after = [h for h in s.history[i + 1:] if h["kind"] in ("first", "confirm")]
        assert after and after[0]["kind"] == "first" and after[0]["scoring_regime"] == "v5"
