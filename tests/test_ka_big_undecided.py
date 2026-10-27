"""KA_BIG_UNDECIDED: the open-class base floor (> KEEPALIVE_BIG_PARAMS) is a test on a SCORE; an undecided generalist entry
(waiting for its first draw, or owing one more) keeps the provisional keep-alive."""
import pytest
from spec_helpers import HOTKEYS
from test_keepalive import _snap, _sub
from test_provisional_ka import _p
from test_waiting_ka_hotfix import THU, _commit_block, _prov, _queued, release_slots  # noqa: F401

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import compute_keepalive

pytestmark = pytest.mark.usefixtures("rc5_entry_floors", "release_slots")
GEN = "generalist"
BIG = 27_400_000_000


def _gen(st, uid, *, params=BIG, **kw):
    s = _queued(st, uid, track=GEN, **kw)
    s.params = params
    return s


def _gprov(st, subs, hours=3, **kw):
    cb = {s.uid: _commit_block(hours) for s in subs}
    return _prov(st, subs, THU, track=GEN, cblocks=cb, regs={s.uid: 1 for s in subs}, **kw)


def test_a_big_fetched_unscored_entry_is_kept_until_its_first_draw_then_pending_or_nothing():
    st = ST.SpecState()
    s = _gen(st, 1)
    assert CS.spec_open_class_active(THU)
    assert _gprov(st, [s], hours=40) == [1]
    s.score, s.status = 12.0, ST.NOT_BETTER                  # failed first draw: the window is revoked
    assert _gprov(st, [s], hours=40) == []
    s.status = ST.REJECTED                                 # copy / rules rejection also revokes it
    assert _gprov(st, [s], hours=40) == []
    s.score, s.status, s.nonproc = 31.0, ST.PENDING_CONFIRM, 99.0   # passed: provisional ends, the score-based keep-alive decides
    assert _gprov(st, [s], hours=40) == []


def test_a_big_unfetched_and_private_entries_also_keep_their_windows():
    st = ST.SpecState()
    a = _gen(st, 1, fetched=False)
    b = _gen(st, 2, fetched=False, private=True)
    assert _gprov(st, [a, b], hours=10) == [1, 2]
    assert _gprov(st, [b], hours=25) == []                   # unverified: still 24 h


def test_b_draw_owed_big_entry_without_base_floor_keeps_it():
    st = ST.SpecState()
    s = _gen(st, 1, params=26_100_000_000)
    s.score, s.close_draws, s.base_id = 20.0, 1, ""
    assert _gprov(st, [s], hours=30) == [1]


def test_invalid_params_still_get_nothing():
    st = ST.SpecState()
    for bad in (-1, "x", True):
        s = _gen(st, 1, params=bad)
        assert _gprov(st, [s]) == [], bad


def test_c_scored_entry_under_its_base_floor_gets_no_score_based_keepalive():
    st = ST.SpecState()
    floor = 40.0
    table = {"bh": {GEN: floor}}
    s = _sub(st, 1, ST.PENDING_CONFIRM, floor + 1.5, track=GEN, nonproc=floor + 1.5)   # exactly floor + margin: not above it
    s.params, s.base_id = BIG, "bh"
    s.first_pass_ts = THU - 3600.0
    s.confirm_after_ts = s.first_pass_ts + 3600.0
    snap = _snap([s])
    res = compute_keepalive(st, [GEN], snap, {GEN: 0.0}, now=THU, base_floors=table)
    assert res[GEN] == []
    s.nonproc = floor + 2.0                                          # control: above floor + margin is kept
    assert [u for u, _ in compute_keepalive(st, [GEN], snap, {GEN: 0.0}, now=THU, base_floors=table)[GEN]] == [1]
    assert not E._big_ok(s, GEN, THU, table, floor + 1.5)
    assert E._big_ok(s, GEN, THU, table, floor + 1.6)
    assert not E._big_ok(s, GEN, THU, table, None)                  # no score: still never, without the undecided flag


def test_d_switch_off_restores_the_old_gate(monkeypatch):
    monkeypatch.setattr(CS, "KA_BIG_UNDECIDED", False)
    st = ST.SpecState()
    s = _gen(st, 1)
    assert _gprov(st, [s], hours=40) == []
    assert not E._big_ok(s, GEN, THU, undecided=True)


def test_small_and_unknown_params_unchanged():
    st = ST.SpecState()
    a, b = _gen(st, 1, params=9_000_000_000), _gen(st, 2, params=None)
    assert _gprov(st, [a, b]) == [1, 2]
