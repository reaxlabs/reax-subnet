"""rc11p4-critic: SLOT-RULES s.2 / FAQ ("an entry that is not in the slot starts again with a new first draw against it")
also for an entry whose first stage was taken against the former champion and had not confirmed at the crowning."""
from test_slot_rules import _fresh, _restarted_then_first, _script, _slot, _rec


async def test_mid_stage_entry_restarts_with_a_new_first_draw_after_a_crowning(tmp_path, monkeypatch):
    _script(monkeypatch, {2: [75.40, 75.25], 3: [74.20, 73.85]})
    env, T, allc, ts, slot = await _slot(tmp_path, monkeypatch, [2, 3, 4])
    s4 = env.sub(4)         # first pass before the slot, confirmation due only after it: not in the slot, no stored crowning
    s4.history = [h for h in s4.history if h.get("kind") not in ("crown_pending", "confirm")]
    s4.confirm_draw, s4.confirm_score, s4.reason = "", None, ""
    s4.confirm_after_ts = slot.timestamp() + 6 * 3600
    old, n = s4.draw1, len(s4.history)
    await env.cycle(allc)
    assert ts.champion == env.sub(2).key and 4 not in [e["uid"] for e in _rec(ts)["entries"]]
    assert len(s4.history) > n
    assert _fresh(s4, "before this entry confirmed") or any(h["kind"] == "slot_restart" for h in s4.history)
    assert env.sub(2).status == "champion"      # the new holder is not touched
    for _ in range(3):
        env.clock.advance(3600)
        await env.cycle(allc)
    _restarted_then_first(s4, old)
