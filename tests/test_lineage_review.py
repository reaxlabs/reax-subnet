"""rc11.post5 evaluator wiring of the lineage canonicalisation and the review state (spec/review.py).

* a function-preserving rewrite of the champion's weights is linked to the champion (``derived_from``), flagged and never
  crowned or kept alive; ``REAX_SPEC_REVIEW_CLEAR`` lifts the flag;
* ``REAX_SPEC_HOLD``: a held entry is drawn but not confirmed, crowned or kept alive; lifting the hold lets it go on;
* empty lists and no flag: the evaluator's state and outputs are those of 1.2.0rc11.post4 (test_lineage_golden.py).
"""

from __future__ import annotations

import pytest
from spec_helpers import HOTKEYS, REV, commit, neuron, snapshot
from test_copycheck2 import _server_env, noisy, temperature
from lineage_helpers import everything, toy_base, tune
from test_nearcopy import _crown
from test_spec_evaluator import TRACK

from reax_subnet.spec import config as C
from reax_subnet.spec import machine as M
from reax_subnet.spec import review as RV
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


@pytest.fixture(autouse=True)
def small_margin(monkeypatch):
    monkeypatch.setattr(M, "MARGIN_FLOOR", 0.3)
    monkeypatch.setattr(M, "SE_MULTIPLE", 1.0)


@pytest.fixture(autouse=True)
def no_lists(monkeypatch):
    monkeypatch.delenv(RV.HOLD_ENV, raising=False)
    monkeypatch.delenv(RV.CLEAR_ENV, raising=False)


def _snap(env, commits):
    return snapshot(1000, [neuron(u, commit(TRACK, r)) for u, r in commits.items()])


def _ka(env, commits):
    return [u for u, _ in env.ev.keepalive(C.DEFAULT_TRACKS[:1], _snap(env, commits)).get(TRACK, [])]


async def _copy_scenario(tmp_path):
    base = toy_base()
    env = _server_env(tmp_path, base)
    parent = tune(base, 10, 0.05)
    env.server.skills["z/hidden"] = 0.65
    env.server.like["a/x9b"] = ("z/hidden", temperature(0.3))
    await _crown(env, 1, "a/x9b", None, parent)
    assert env.sub(1).status == ST.CHAMPION
    env.add(2, "b/copy", weights=everything(tune(parent, 11, 0.001)))
    env.server.like["b/copy"] = ("z/hidden", noisy(0.05))
    return env


def test_review_lists_parse():
    assert RV.parse("") == () and RV.parse(" , ") == ()
    assert RV.parse("5Hot01xx@abcdef12, 5Hot02xx@ABCDEF") == (("5Hot01xx", "abcdef12"), ("5Hot02xx", "abcdef"))
    assert RV.parse("5Hot@abcdef,5Hot01xx,5Hot01xx@abc") == ()     # too short / no revision: ignored, never "everything"


async def test_rewritten_copy_of_the_champion_is_linked_flagged_and_not_crowned(tmp_path):
    env = await _copy_scenario(tmp_path)
    commits = {1: "a/x9b", 2: "b/copy"}
    await env.cycle(commits)
    s2 = env.sub(2)
    assert s2.derived_from == env.sub(1).key and s2.derived_how == "weights"
    assert RV.flag_of(s2).startswith("pow2_rescale:") and RV.under_review(s2)
    assert any("under review" in e for e in env.ev.state.track(TRACK).events)
    for _ in range(3):
        env.clock.advance(13 * 3600)
        await env.cycle(commits)
        assert env.sub(2).status != ST.CHAMPION and env.ev.state.track(TRACK).champion == env.sub(1).key
        assert 2 not in _ka(env, commits)


async def test_operator_hold_blocks_confirmation_crown_and_keepalive_then_lifts(tmp_path, monkeypatch):
    base = toy_base()
    env = _server_env(tmp_path, base)
    env.add(1, "a/m1", 0.9, weights=tune(base, 20, 0.05))
    commits = {1: "a/m1"}
    monkeypatch.setenv(RV.HOLD_ENV, f"{HOTKEYS[1][:10]}@{REV[:8]}")
    await env.cycle(commits)
    s = env.sub(1)
    assert s.status == ST.PENDING_CONFIRM and not RV.flag_of(s)    # drawn: the first draw runs and passes
    assert RV.under_review(s).startswith("operator hold")
    assert _ka(env, commits) == []                                 # no keep-alive while held
    served = len(env.server.served)
    env.clock.advance(13 * 3600)                                   # the confirmation is due
    await env.cycle(commits)
    assert env.sub(1).status == ST.PENDING_CONFIRM and env.ev.state.track(TRACK).champion == ""
    assert len(env.server.served) == served                        # no confirmation draw while held
    assert env.sub(1).reason.startswith("under review")
    assert any("under review, its confirmation waits" in e for e in env.ev.state.track(TRACK).events)
    monkeypatch.setenv(RV.HOLD_ENV, "")
    assert _ka(env, commits) == [1]
    env.clock.advance(60)
    await env.cycle(commits)
    assert env.sub(1).status == ST.CHAMPION


async def test_reviewer_clears_a_flag(tmp_path, monkeypatch):
    env = await _copy_scenario(tmp_path)
    await env.cycle({1: "a/x9b", 2: "b/copy"})
    s2 = env.sub(2)
    assert RV.under_review(s2)
    monkeypatch.setenv(RV.CLEAR_ENV, f"{s2.hotkey[:10]}@{s2.revision[:8]}")
    assert RV.under_review(s2) == "" and RV.flag_of(s2)            # the record stays, the review is over
    monkeypatch.setenv(RV.CLEAR_ENV, f"{s2.hotkey[:10]}@{'f' * 8}")  # another revision: still held
    assert RV.under_review(s2)


def test_hold_keeps_a_stored_crowning_and_a_cohort_entry_back(tmp_path, monkeypatch):
    """The crowning paths that do not run through a confirmation draw: a crowning stored for its slot and a confirmed
    entry waiting for its cohort's playoff. Held: neither is due, and neither holds the others back."""
    from reax_subnet import competition_split as split
    env = _server_env(tmp_path, toy_base())
    ts = env.ev.state.track(TRACK)
    now = env.clock()
    monkeypatch.setattr(split, "CROWN_SLOTS_UTC", ())

    def sub(uid, **kw):
        s = ST.Submission(key=f"{HOTKEYS[uid]}|{uid}", uid=uid, hotkey=HOTKEYS[uid], coldkey=f"c{uid}", track_id=TRACK,
                          repo=f"r/{uid}", revision=REV, first_seen_block=1000, first_seen_ts=now, status=ST.PENDING_CONFIRM, **kw)
        ts.submissions[s.key] = s
        return s

    stored = sub(1, history=[{"kind": "crown_pending", "ts": now - 10}])
    waiting = sub(2, confirm_draw="d1", confirm_score=50.0)
    queued = sub(3)
    queued.status = ST.QUEUED
    assert env.ev._cohort_open(ts, queued)
    assert env.ev._slot_due(ts, now) == [stored]
    assert env.ev._cohort(ts, 1000)[0] == [waiting]
    monkeypatch.setenv(RV.HOLD_ENV, ",".join(f"{HOTKEYS[u][:10]}@{REV[:8]}" for u in (1, 2, 3)))
    assert env.ev._slot_due(ts, now) == []
    assert env.ev._cohort(ts, 1000)[0] == []
    assert env.ev._cohort_step(ts, now) == [] and ts.champion == ""
    assert not env.ev._cohort_open(ts, queued)


def test_startup_receipt_logs_the_parsed_hold_list(monkeypatch, caplog):
    import logging
    monkeypatch.setenv(RV.HOLD_ENV, "5GYXd1n3uQMo@3EC2C313")
    monkeypatch.delenv(RV.CLEAR_ENV, raising=False)
    with caplog.at_level(logging.INFO, logger="reax.spec.review"):
        RV.log_config()
    assert "REAX_SPEC_HOLD parsed: 5GYXd1n3uQMo@3ec2c313" in caplog.text
    assert "REAX_SPEC_REVIEW_CLEAR parsed: (empty)" in caplog.text
