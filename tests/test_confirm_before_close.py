"""Every pending confirmation gets its draw before the track closes (M.CONFIRM_BEFORE_CLOSE_START_UTC, 21 Oct 2026)."""

import dataclasses
import math
from datetime import datetime, timedelta, timezone

import pytest
from test_spec_evaluator import TRACK, Env, T0
from test_v3_decisions import champ_and_challenger, zs  # noqa: F401 - zs is a fixture

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

H = 3600.0
LEAD = M.CONFIRM_CLOSE_LEAD_S
GATE = datetime.fromtimestamp(T0 - 24 * H, tz=timezone.utc)      # open for the whole test unless a test says otherwise
FOREVER = datetime.max.replace(tzinfo=timezone.utc).timestamp()


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(M, "CONFIRM_BEFORE_CLOSE_START_UTC", GATE)


def closing_env(tmp_path, close_in_h):
    """An Env whose only track closes ``close_in_h`` hours after T0."""
    env = Env(tmp_path, {})
    env.cfg.tracks = (dataclasses.replace(C.DEFAULT_TRACKS[0], end_ts=T0 + close_in_h * H),)
    env.ev = env.make()
    return env


# ------------------------------------------------------------------ the pure function

def test_release_constant_is_t1():
    assert M.CONFIRM_BEFORE_CLOSE_START_UTC == datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc) and M.CONFIRM_CLOSE_LEAD_S == 6 * 3600


def test_confirm_due_ts_gate_off_is_unchanged(monkeypatch):
    monkeypatch.setattr(M, "CONFIRM_BEFORE_CLOSE_START_UTC", None)
    assert M.confirm_due_ts(1000.0, 12 * H, 1000.0 + 8 * H, 1000.0) == 1000.0 + 12 * H


def test_confirm_due_ts_before_at_after_the_gate(monkeypatch):
    g = datetime(2026, 11, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(M, "CONFIRM_BEFORE_CLOSE_START_UTC", g)
    t, end = g.timestamp(), g.timestamp() + 8 * H
    assert M.confirm_due_ts(t, 12 * H, end, t - 1) == t + 12 * H           # before the date: old
    assert M.confirm_due_ts(t, 12 * H, end, t) == end - LEAD               # the boundary instant: new
    assert M.confirm_due_ts(t, 12 * H, end, t + 5) == end - LEAD           # after


def test_confirm_due_ts_shapes(gate):
    now = GATE.timestamp() + H
    assert M.confirm_due_ts(now, 12 * H, now + 30 * H, now) == now + 12 * H               # close far away: normal delay
    assert M.confirm_due_ts(now, 12 * H, now + 18 * H, now) == now + 12 * H               # exactly lead + delay: equal
    assert M.confirm_due_ts(now, 12 * H, now + 8 * H, now) == now + 2 * H                 # 8 h before the close
    assert M.confirm_due_ts(now, 12 * H, now + 2 * H, now) == now                         # inside the lead: at once
    assert M.confirm_due_ts(now, 12 * H, now - H, now) == now                             # after the close: never earlier than the first pass
    for end in (None, math.inf, math.nan, FOREVER):                                       # no close
        assert M.confirm_due_ts(now, 12 * H, end, now) == now + 12 * H


# ------------------------------------------------------------------ the evaluator

async def test_first_pass_8h_before_the_close_is_confirmed_before_it_and_crowned(tmp_path, gate):
    env = closing_env(tmp_path, 8)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    assert s.status == ST.PENDING_CONFIRM and s.first_pass_ts == T0 and s.confirm_after_ts == T0 + 2 * H
    env.clock.advance(H)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_CONFIRM and len(env.server.served) == 1      # not yet due
    env.clock.advance(1.5 * H)                                                          # T0 + 2.5 h, close at T0 + 8 h
    out = await env.cycle({1: "a/m1"})
    assert [u for u, _ in out[TRACK]] == [1] and env.sub(1).status == ST.CHAMPION
    assert env.clock.t < T0 + 8 * H


async def test_first_pass_2h_before_the_close_is_confirmed_at_once(tmp_path, gate):
    env = closing_env(tmp_path, 2)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).confirm_after_ts == T0
    env.clock.advance(60)
    out = await env.cycle({1: "a/m1"})
    assert [u for u, _ in out[TRACK]] == [1] and env.sub(1).status == ST.CHAMPION


async def test_gate_off_keeps_the_12_hour_delay(tmp_path):
    env = closing_env(tmp_path, 8)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).confirm_after_ts == T0 + 12 * H and env.sub(1).first_pass_ts is None     # state as before the gate


async def test_track_without_a_close_is_unchanged(tmp_path, gate):
    env = closing_env(tmp_path, 8)
    env.cfg.tracks = (dataclasses.replace(C.DEFAULT_TRACKS[0], end_ts=FOREVER),)
    env.ev = env.make()
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).confirm_after_ts == T0 + 12 * H


async def test_due_confirmation_runs_before_a_new_first_stage_entry_inside_the_lead(tmp_path, gate):
    env = closing_env(tmp_path, 20)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).confirm_after_ts == T0 + 12 * H
    env.clock.advance(14.5 * H)                         # 5.5 h before the close: inside the lead, m1 is due
    env.add(2, "a/m2", 0.95)
    await env.cycle({1: "a/m1", 2: "a/m2"})
    # vacant track: the passed confirmation waits for its cohort (crowning is the cohort's step), but it was drawn first ...
    assert env.sub(1).confirm_draw and env.sub(1).status == ST.PENDING_CONFIRM
    assert env.sub(2).status == ST.QUEUED and not env.sub(2).draw1      # ... the new entry got no first draw this cycle
    assert len(env.server.served) == 2


async def test_v3_extra_confirmation_draw_is_due_at_once_inside_the_lead(tmp_path, gate, zs):
    """The extra draw keeps ``confirm_after_ts`` (a past value): due on the next cycle, with no second delay."""
    env = closing_env(tmp_path, 17)
    env.cfg.v3_decisions = True
    sub, ts = await champ_and_challenger(env, zs, 2.05)          # first pass at T0 + 13 h: inside the last 6 h of the track
    assert sub.status == ST.PENDING_CONFIRM and sub.confirm_after_ts == env.clock.t
    env.clock.advance(60)
    zs[:] = [1.5]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.PENDING_CONFIRM and len(sub.confirm_draws) == 1            # "one more draw"
    assert sub.confirm_after_ts <= env.clock.t                                         # due at once
    env.clock.advance(60)
    zs[:] = [8.0]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.CHAMPION and env.clock.t < T0 + 17 * H


async def test_pull_forward_of_entries_pending_when_the_gate_opens_is_idempotent_and_logged_once(tmp_path, monkeypatch):
    env = closing_env(tmp_path, 8)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})                                           # gate still None: due at T0 + 12 h
    s = env.sub(1)
    s.first_pass_ts = None                                                 # a state written before the field
    old = s.confirm_after_ts
    assert old == T0 + 12 * H
    monkeypatch.setattr(M, "CONFIRM_BEFORE_CLOSE_START_UTC", datetime.fromtimestamp(T0 + 60, tz=timezone.utc))
    env.clock.advance(120)
    ts = env.ev.state.track(TRACK)
    env.ev._confirm_before_close(env.cfg.tracks, env.clock.t)
    assert s.confirm_after_ts == T0 + 2 * H and s.first_pass_ts == T0     # pulled to close - lead; first pass kept
    lines = [e for e in ts.events if "pulled forward" in e]
    assert len(lines) == 1 and str(int(old)) in lines[0]
    env.ev._confirm_before_close(env.cfg.tracks, env.clock.t)
    env.ev._confirm_before_close(env.cfg.tracks, env.clock.t + 3 * H)      # due by now: left alone
    assert s.confirm_after_ts == T0 + 2 * H and len([e for e in ts.events if "pulled forward" in e]) == 1


async def test_pull_forward_inside_the_lead_is_due_now_and_never_before_the_first_pass(tmp_path, monkeypatch):
    env = closing_env(tmp_path, 8)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    monkeypatch.setattr(M, "CONFIRM_BEFORE_CLOSE_START_UTC", datetime.fromtimestamp(T0, tz=timezone.utc))
    now = T0 + 3 * H                                                       # inside the lead already
    env.ev._confirm_before_close(env.cfg.tracks, now)
    assert s.confirm_after_ts == now


async def test_pull_forward_pass_is_off_with_the_gate_none(tmp_path):
    env = closing_env(tmp_path, 8)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    env.ev._confirm_before_close(env.cfg.tracks, T0 + 7 * H)
    assert env.sub(1).confirm_after_ts == T0 + 12 * H


async def test_closing_with_a_pending_entry_is_logged_once_and_nobody_is_crowned(tmp_path, gate):
    env = closing_env(tmp_path, 8)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    env.clock.advance(9 * H)                                               # after the close, the draw never ran
    out = await env.cycle({1: "a/m1"})
    out = await env.cycle({1: "a/m1"})
    assert out == {} or all(not v for v in out.values())
    assert env.sub(1).status == ST.PENDING_CONFIRM
    assert len([e for e in env.ev.state.track(TRACK).events if "closed with the confirmation still pending" in e]) == 1


async def test_keepalive_window_counts_from_the_first_pass_not_from_the_pulled_in_due_time(tmp_path, gate):
    env = closing_env(tmp_path, 8)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    assert s.confirm_after_ts - s.first_pass_ts == 2 * H
    assert E._keepalive_since(s) == T0                                     # not T0 - 10 h
    s.first_pass_ts = None                                                 # old state: derived from confirm_after_ts as before
    assert E._keepalive_since(s) == s.confirm_after_ts - M.CONFIRM_DELAY_S
