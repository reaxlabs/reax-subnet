"""Engine copy line: the declared parent measured in the same session (Florian 24 Oct 2026 13:02Z, card 18502, option A;
gate ``engcomp.ENGINE_PARENT_SESSION_START_UTC``; docs/ENGINE-COMPETITION.md, "Reusing other entries").

A derived entry whose session clears the champion bar but is below, or within the margin of, its stored parent line (the
parent's best valid score ever x (1 + margin)) gets the binding parent image measured right after it with the same items. For
that session the line is max(parent in this session, the parent image's lowest valid score) x (1 + margin); without a valid
same-session run the stored line applies. The reference run never changes the parent's own scores."""

import asyncio
import json
from datetime import timedelta

import pytest

from reax_subnet import engcomp as EC
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import Champion, EvalResult, TrackState
from reax_subnet.engrunner import ItemTape
from test_engcomp import P
from test_engfair import DA, _snap, _sub
from test_engine_copyline import D1, D2, MARGIN, START, SEC, _comp, _gate, _label, _record

LINE = 1.5 * (1 + MARGIN)          # stored line: parent D1's best ever 1.5
NEAR = 1.52                        # near-miss child: clears no champion bar issue, below LINE
SUBS = [_sub(1, "pa", D1, 100), _sub(2, "ch", D2, 200)]


def _state(child=NEAR, low=1.45, champion=None):
    st = TrackState()
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=100, session="p")
    st.hold(EvalResult(D1, "s1-fast", True, score=low, block=90, session="p0"), P)
    st.hold(st.results[D1], P)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=child, block=200, session="c1")
    st.first_seen = {D1: (100, "pa"), D2: (200, "ch")}
    st.derived[D2] = [D1]
    st.champion = champion
    return st


def _run(st, session, score, ok=True, parent=D1, child=D2):
    st.parent_sessions[session] = {"child": child, "parent": parent, "ok": ok, "score": score, "block": 1,
                                   "session": "x", "items": "same"}


def _two_sessions(st, parent_scores, copyline=True):
    """Child measured in sessions c1 and c2 at its score; ``parent_scores``: same-session parent run per session (None: none)."""
    score = st.results[D2].score
    if parent_scores[0] is not None:
        _run(st, "c1", parent_scores[0])
    EC.update_champion(st, SUBS, P, 3000, copyline=copyline)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=score, block=300, session="c2")
    if parent_scores[1] is not None:
        _run(st, "c2", parent_scores[1])
    return EC.update_champion(st, SUBS, P, 3100, copyline=copyline)


# ------------------------------------------------------------------------------------------------ the line
def test_floor_is_the_lowest_valid_score():
    st = _state()
    assert st.low_scores[D1] == 1.45 and st.held_scores[D1] == 1.5
    assert EC.parent_floor(st, D1) == 1.45
    assert EC.binding_parent(st, D2) == D1


def test_session_line_uses_parent_in_session_with_floor():
    st = _state()
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(LINE)               # stored line
    _run(st, "c1", 1.47)
    assert EC.parent_bar(st, D2, MARGIN, "c1") == pytest.approx(1.47 * (1 + MARGIN))
    _run(st, "c1", 1.0)                                                       # deliberately slow parent run
    assert EC.parent_bar(st, D2, MARGIN, "c1") == pytest.approx(1.45 * (1 + MARGIN))   # the floor holds
    _run(st, "c1", 1.6)                                                       # a faster parent raises the line
    assert EC.parent_bar(st, D2, MARGIN, "c1") == pytest.approx(1.6 * (1 + MARGIN))
    assert EC.parent_bar(st, D2, MARGIN, "other") == pytest.approx(LINE)      # another session: stored line


def test_near_miss_child_passes_with_lower_same_session_parent():
    st = _state()
    ch = _two_sessions(st, (1.46, 1.46))                       # line 1.46 x 1.03 = 1.504 < 1.52
    assert ch is not None and ch.digest == D2


def test_slow_parent_run_cannot_drop_below_the_floor():
    st = _state(low=1.49)                                       # floor 1.49 x 1.03 = 1.535 > 1.52
    ch = _two_sessions(st, (1.0, 1.0))
    assert ch is None or ch.digest != D2


def test_no_same_session_result_keeps_the_stored_line():
    st = _state()
    ch = _two_sessions(st, (None, None))
    assert ch is None or ch.digest != D2


@pytest.mark.parametrize("bad", [
    {"ok": False},                       # the parent run failed
    {"child": DA},                        # a run of another child
    {"parent": DA},                       # not the binding parent image
])
def test_unusable_same_session_runs_keep_the_stored_line(bad):
    st = _state()
    entry = {"child": D2, "parent": D1, "ok": True, "score": 1.46, "block": 1, "session": "x", "items": "same", **bad}
    st.parent_sessions["c1"] = entry
    assert EC.parent_bar(st, D2, MARGIN, "c1") == pytest.approx(LINE)


def test_every_confirmation_session_needs_its_own_parent_run():
    st = _state()
    ch = _two_sessions(st, (1.46, None))                        # second session falls back to the stored line
    assert ch is None or ch.digest != D2
    assert st.challenger is None or st.challenger["digest"] != D2


def test_earlier_measurement_keeps_its_own_session_line():
    st = _state()
    _run(st, "c1", 1.46)
    EC.update_champion(st, SUBS, P, 3000, copyline=True)
    assert st.challenger is not None and st.challenger["digest"] == D2
    # the second session is clear of the stored line by itself; the first still counts at its own session line
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=1.56, block=300, session="c2")
    ch = EC.update_champion(st, SUBS, P, 3100, copyline=True)
    assert ch is not None and ch.digest == D2 and ch.score == pytest.approx(NEAR)


def test_non_derived_entries_unaffected():
    st = _state()
    st.derived[D2] = []
    _run(st, "c1", 0.5)
    assert EC.parent_bar(st, D2, MARGIN, "c1") == 0.0
    assert EC.parent_reference_due(st, D2, P) is None
    st.derived.pop(D2)
    assert EC.parent_reference_due(st, D2, P) is None


def test_copyline_off_ignores_session_runs():
    st = _state(low=1.6)                       # a session run would set a line above the child if it were read
    st.champion = Champion(DA, "ha", 9, 1.3, 100)
    a = _two_sessions(st, (1.7, 1.7), copyline=False)
    ref = _state(low=1.6)
    ref.champion = Champion(DA, "ha", 9, 1.3, 100)
    b = _two_sessions(ref, (None, None), copyline=False)
    assert (a.digest if a else None) == (b.digest if b else None)
    assert st.challenger == ref.challenger and st.history == ref.history


# ------------------------------------------------------------------------------------------------ when a parent run is due
def test_due_only_for_a_valid_near_miss_that_clears_the_champion_bar():
    st = _state()
    assert EC.parent_reference_due(st, D2, P) == D1                       # below the stored line
    st.results[D2].score = LINE + 0.01                                    # above, but within the margin of it
    assert EC.parent_reference_due(st, D2, P) == D1
    st.results[D2].score = LINE * (1 + MARGIN) + 0.01                     # clear of the line by the margin
    assert EC.parent_reference_due(st, D2, P) is None
    st.results[D2].score = NEAR
    st.champion = Champion(DA, "ha", 9, 1.6, 100)                         # champion bar 1.648; child below it by more than a margin
    assert EC.parent_reference_due(st, D2, P) is None
    st.champion = Champion(DA, "ha", 9, 1.3, 100)
    assert EC.parent_reference_due(st, D2, P) == D1
    st.results[D2].ok = False
    assert EC.parent_reference_due(st, D2, P) is None


def test_not_due_while_a_parent_has_no_valid_score():
    st = _state()
    st.results[D1].ok = False
    st.held_scores.clear()
    assert EC.parent_bar(st, D2, MARGIN) == float("inf")
    assert EC.parent_reference_due(st, D2, P) is None


def test_binding_parent_is_the_best_digest_of_the_parent_family():
    st = _state()
    d3 = "3" * 64
    st.first_seen[d3] = (150, "pa")                                       # same hotkey, a later and better image
    st.results[d3] = EvalResult(d3, "s1-fast", True, score=1.55, block=150, session="p3")
    assert EC.binding_parent(st, D2) == d3
    _run(st, "c1", 1.4, parent=D1)                                        # a run of the weaker image does not count
    assert EC.parent_bar(st, D2, MARGIN, "c1") == pytest.approx(1.55 * (1 + MARGIN))


# ------------------------------------------------------------------------------------------------ state
def test_state_roundtrip_add_only_and_old_state():
    st = TrackState()
    assert not {"low_scores", "parent_sessions"} & set(st.to_json())       # absent while empty: rollback-safe
    st = _state()
    _run(st, "c1", 1.46)
    back = TrackState.from_json(json.loads(json.dumps(st.to_json())))
    assert back.low_scores == st.low_scores and back.parent_sessions == st.parent_sessions
    old = TrackState.from_json({"results": {}, "held_scores": {D1: 1.5}})  # a file from rc11.post4
    assert old.low_scores == {} and old.parent_sessions == {}
    raw = st.to_json()
    raw["parent_sessions"]["bad"] = {"child": D2, "parent": D1, "ok": "yes", "score": float("nan")}
    assert "bad" not in TrackState.from_json(raw).parent_sessions


def test_old_state_floor_without_low_scores_is_never_too_low():
    st = _state()
    st.low_scores.clear()                                                  # state written before the field existed
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.48, block=400, session="p9")
    assert EC.parent_floor(st, D1) == 1.48                                 # min(held 1.5, latest 1.48)


def test_reset_drops_the_new_fields_and_prune_keeps_live_sessions():
    st = _state()
    _run(st, "c1", 1.46)
    _run(st, "gone", 1.46)
    EC.prune_parent_sessions(st)
    assert set(st.parent_sessions) == {"c1"}
    st.reset_measurements(1, "test")
    assert st.low_scores == {} and st.parent_sessions == {}


# ------------------------------------------------------------------------------------------------ item replay
def test_item_tape_replays_in_order_and_falls_back():
    tape = ItemTape()
    a = tape._take(("fresh", 4, "m"), lambda: ["a"])
    b = tape._take(("probes", 2, "m"), lambda: ["b"])
    rp = tape.replay()
    assert rp._take(("fresh", 4, "m"), lambda: ["X"]) is a
    assert rp._take(("probes", 2, "m"), lambda: ["X"]) is b
    assert rp._take(("fresh", 4, "m"), lambda: ["new"]) == ["new"]          # recording ran out: fresh items
    rp2 = tape.replay()
    assert rp2._take(("fresh", 8, "m"), lambda: ["new"]) == ["new"]         # out of step: fresh from here on
    assert rp2._take(("probes", 2, "m"), lambda: ["new2"]) == ["new2"]
    assert tape.recorded == 2


# ------------------------------------------------------------------------------------------------ validator wiring
class _Eval:
    """Stand-in for engrunner.evaluate: records the items of a child session and checks the parent replays them."""

    def __init__(self, scores, parent_ok=True):
        self.scores, self.parent_ok, self.calls, self.items = scores, parent_ok, [], {}

    async def __call__(self, runner, reference, ref, track, params, block=0, digest="", progress=None):
        tape = progress["items"]
        got = tape._take(("fresh", 4, "m"), lambda: object())
        self.calls.append(digest)
        self.items.setdefault(digest, []).append(got)
        ok = self.parent_ok or digest != D1
        return EvalResult(digest, track, ok, score=self.scores[digest] if ok else 0.0, block=block)


def _wired(tmp_path, monkeypatch, now, scores, parent_committed=True, parent_ok=True):
    _gate(monkeypatch, START)
    monkeypatch.setattr(EC, "ENGINE_PARENT_SESSION_START_UTC", START + timedelta(hours=1))
    comp = _comp(tmp_path, now=now)
    st = comp.states["s1-fast"]
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=50, session="p")
    st.hold(EvalResult(D1, "s1-fast", True, score=1.45, block=40, session="p0"), P)
    st.hold(st.results[D1], P)
    st.first_seen[D1] = (50, "pa")
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
    fake = _Eval(scores, parent_ok)
    monkeypatch.setattr(EV, "evaluate", fake)
    subs = [_sub(2, "ch", D2, 200)] + ([_sub(1, "pa", D1, 50)] if parent_committed else [])
    asyncio.run(comp.run_cycle(_snap(subs, block=1000), set()))
    return comp, st, fake


def test_wired_parent_measured_after_each_child_session_with_the_same_items(tmp_path, monkeypatch):
    comp, st, fake = _wired(tmp_path, monkeypatch, START + timedelta(hours=2), {D2: NEAR, D1: 1.46})
    # parent D1 is committed and unmeasured-pending? no: it has a result, so only the child is new
    assert fake.calls == [D2, D1, D2, D1]                                  # first session + confirmation, parent after each
    assert fake.items[D1] == fake.items[D2]                                # the same items, replayed
    assert st.results[D1].score == 1.5 and st.held_scores[D1] == 1.5 and st.low_scores[D1] == 1.45   # untouched
    assert st.champion is not None and st.champion.digest == D2
    events = [e for e in st.cycle_events if e.get("event") == "parent_reference"]
    assert len(events) == 2 and all(e["ok"] and e["parent"] == D1 for e in events)


def test_wired_off_before_the_start_instant(tmp_path, monkeypatch):
    comp, st, fake = _wired(tmp_path, monkeypatch, START + timedelta(minutes=30), {D2: NEAR, D1: 1.46})
    assert D1 not in fake.calls and st.parent_sessions == {}
    assert st.champion is None or st.champion.digest != D2                 # stored line, as before


def test_wired_parent_not_committed_keeps_the_stored_line(tmp_path, monkeypatch):
    comp, st, fake = _wired(tmp_path, monkeypatch, START + timedelta(hours=2), {D2: NEAR, D1: 1.46},
                            parent_committed=False)
    assert D1 not in fake.calls
    assert st.champion is None or st.champion.digest != D2
    ev = [e for e in st.cycle_events if e.get("event") == "parent_reference"]
    assert ev and not ev[0]["ok"] and "not committed" in ev[0]["reason"]


def test_wired_failed_parent_run_keeps_the_stored_line(tmp_path, monkeypatch):
    comp, st, fake = _wired(tmp_path, monkeypatch, START + timedelta(hours=2), {D2: NEAR, D1: 1.46}, parent_ok=False)
    assert fake.calls == [D2, D1]
    assert ("s1-fast", D1) in comp._parent_failed
    assert st.results[D1].ok and st.results[D1].score == 1.5               # the failed reference run recorded nothing
    assert st.champion is None or st.champion.digest != D2


def test_wired_clear_child_needs_no_parent_run(tmp_path, monkeypatch):
    comp, st, fake = _wired(tmp_path, monkeypatch, START + timedelta(hours=2), {D2: 1.7, D1: 1.46})
    assert D1 not in fake.calls
    assert st.champion is not None and st.champion.digest == D2


# ------------------------------------------------------------------------------------------------ F3: floor = lowest VALID score
def test_ok_but_invalid_parent_session_does_not_lower_the_floor():
    st = _state()
    assert EC.valid_min(P) > 1.0
    st.hold(EvalResult(D1, "s1-fast", True, score=1.0, block=95, session="p1"), P)    # ok, but not a valid score
    assert st.low_scores[D1] == 1.45 and EC.parent_floor(st, D1) == 1.45
    _run(st, "c1", 1.0)
    assert EC.parent_bar(st, D2, MARGIN, "c1") == pytest.approx(1.45 * (1 + MARGIN))
    st.hold(EvalResult(D1, "s1-fast", True, score=1.44, block=96, session="p2"), P)   # a valid lower one does lower it
    assert st.low_scores[D1] == 1.44
    st.hold(EvalResult(D1, "s1-fast", True, score=1.0, block=95, session="p1"))         # no params: no floor recorded
    assert st.low_scores[D1] == 1.44


def test_old_state_floor_ignores_an_invalid_latest_result():
    st = _state()
    st.low_scores.clear()
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.0, block=400, session="p9")
    assert EC.parent_floor(st, D1, EC.valid_min(P)) == 1.5                 # held best only; the invalid 1.0 is not a floor
    assert EC.parent_floor(st, D1) == 1.0                                  # (without a threshold the old behaviour)


# ------------------------------------------------------------------------------------------------ F4: one margin of slack
def test_due_in_the_band_below_the_champion_bar_by_less_than_a_margin():
    st = _state(child=1.56)
    st.champion = Champion(DA, "ha", 9, 1.55, 100)
    bar = EC.champion_bar(st, P)                                           # 1.55 x (1 + margin)
    margin = bar / 1.55 - 1.0
    assert bar > 1.56 >= bar / (1 + margin) and EC.parent_reference_due(st, D2, P) == D1   # inside the band
    st.results[D2].score = 1.55 - 0.005                                    # below the band
    assert EC.parent_reference_due(st, D2, P) is None
    st.results[D2].score = 1.56                                            # crowning still uses the real bar
    EC.update_champion(st, SUBS, P, 3000, copyline=True)
    assert st.champion is None or st.champion.digest != D2


# ------------------------------------------------------------------------------------------------ F5: bounded cost
def test_parent_run_limits():
    assert EC.ENGINE_PARENT_RUNS_PER_CYCLE == 3 and EC.ENGINE_PARENT_RUN_BUDGET_S == 1200.0


def test_wired_parent_runs_per_cycle_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(EV, "ENGINE_PARENT_RUNS_PER_CYCLE", 1)
    comp, st, fake = _wired(tmp_path, monkeypatch, START + timedelta(hours=2), {D2: NEAR, D1: 1.46})
    assert fake.calls == [D2, D1, D2]                                      # session 1 got its parent run, the confirmation did not
    ev = [e for e in st.cycle_events if e.get("event") == "parent_reference"]
    assert len(ev) == 1 and ev[0]["ok"]
    assert st.champion is None or st.champion.digest != D2                 # confirmation falls back to the stored line


def test_wired_parent_run_budget_is_at_most_1200_s(tmp_path, monkeypatch):
    seen = []
    real = asyncio.wait_for

    async def spy(aw, timeout=None):
        seen.append(timeout)
        return await real(aw, timeout)

    monkeypatch.setattr(EV.asyncio, "wait_for", spy)
    _wired(tmp_path, monkeypatch, START + timedelta(hours=2), {D2: NEAR, D1: 1.46})
    assert P.eval_budget_s > 1200 and 1200.0 in seen
    assert seen.count(P.eval_budget_s) >= 2                                # the child sessions keep the full budget


def test_wired_failed_parent_is_not_run_again_in_the_same_cycle(tmp_path, monkeypatch):
    comp, st, fake = _wired(tmp_path, monkeypatch, START + timedelta(hours=2), {D2: NEAR, D1: 1.46}, parent_ok=False)
    assert fake.calls == [D2, D1]
    res = st.results[D2]
    res.session = "again"
    asyncio.run(comp._parent_reference(st, "s1-fast", P, _sub(2, "ch", D2, 200), res,
                                       _snap([_sub(2, "ch", D2, 200), _sub(1, "pa", D1, 50)], block=1000), None))
    assert fake.calls == [D2, D1]                                          # no second parent run
    e = st.parent_sessions["again"]
    assert not e["ok"] and "earlier in this cycle" in e["reason"]
    comp._parent_failed.clear()                                            # a new cycle starts clean
    fake.parent_ok = True
    asyncio.run(comp._parent_reference(st, "s1-fast", P, _sub(2, "ch", D2, 200), res,
                                       _snap([_sub(2, "ch", D2, 200), _sub(1, "pa", D1, 50)], block=1000), None))
    assert fake.calls == [D2, D1, D1]
