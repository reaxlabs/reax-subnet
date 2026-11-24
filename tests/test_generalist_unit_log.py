"""Observational generalist unit: exact internal credit/top-up, once per vector."""
import logging
import math
import re
import sys

import pytest
from reax_subnet import competition_split as C

NOW = 1795359600.0  # 2026-11-22, feed activation passed
UNIT_RE = re.compile(
    r"^generalist keep-alive unit: mode (feed|legacy), per_entry (inf|\d+\.\d{6}), "
    r"each (\d+\.\d{6}), room (\d+\.\d{6}), n (\d+), miner_part (\d+\.\d{6}); "
    r"uids \[(\d+:\d+\.\d{6}:\d+\.\d{6}(?:, \d+:\d+\.\d{6}:\d+\.\d{6})*)\]$")


def compute(caplog, monkeypatch, mode, base, candidates, **kwargs):
    if mode == 'feed':
        monkeypatch.setenv('REAX_KA_FEED_PATH', '/offline/not-read')
    else:
        monkeypatch.delenv('REAX_KA_FEED_PATH', raising=False)
    monkeypatch.setattr(C, 'NEED_RULE', True)
    captured = {}
    # Observe the original frame without substituting any payout expression.
    def trace(frame, event, arg):
        if frame.f_code is C.combine_weights.__code__ and event == 'return':
            captured.update(frame.f_locals)
        return trace
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=C.log.name):
        sys.settrace(trace)
        try:
            vector = C.combine_weights(base, None, 0., {}, owner_uid=3, floors={}, now=NOW,
                keepalive={'generalist': [(u, 50.) for u in candidates]},
                generalist_feed_uids=candidates, **kwargs)
        finally:
            sys.settrace(None)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('generalist keep-alive unit:')]
    assert len(lines) == 1
    return vector, lines[0], captured


@pytest.mark.parametrize('mode', ['feed', 'legacy'])
@pytest.mark.parametrize('shape', ['dual', 'cap', 'unbounded', 'provisional', 'paused', 'burn', 'above_each'])
def test_logged_values_equal_finalized_internal_amounts(caplog, monkeypatch, mode, shape):
    candidates = [83, 114, 116, 199, 250, 19, 24]
    base = {3: .4, 10: .002, 11: .004, 12: .006, 13: .582, 19: .006}
    kwargs = {'inference': .3}
    if shape == 'cap':
        base = {3: .44, 10: .01, 11: .02, 12: .03, 13: .5}
    if shape == 'unbounded':
        base = {3: 1.}
    if shape == 'provisional':
        kwargs['provisional'] = {'generalist': [29, 30]}
    if shape == 'paused':
        kwargs.update(paused=['generalist'], pause_starts={'generalist': NOW - 86400})
    if shape == 'burn':
        kwargs['burn_fraction'] = .1
    if shape == 'above_each':
        base = {3: .4, 10: .1, 11: .1, 12: .1, 13: .1, 19: .2}
    vector, line, internal = compute(caplog, monkeypatch, mode, base, candidates, **kwargs)
    match = UNIT_RE.fullmatch(line)
    assert match, line
    got_mode, per, each, room, n, part, rows = match.groups()
    assert got_mode == mode
    # In paused mode later rank-baseline calculations overwrite each/room.
    # The log is taken at the payout loop, so independently use its finalized
    # room and the actual paid[uid] for top-ups; the median/own remain internal.
    assert float(per) == float(f"{internal['per_entry']:.6f}")
    assert float(part) == float(f"{internal['miner_part']:.6f}")
    ids = internal['alive']['generalist']
    assert int(n) == len(ids)
    expected_room = min(internal['ka_cap'], C.KEEPALIVE_MAX_SHARE) * dict(internal['slots'])['generalist']
    if C.WAITING_KA_HOTFIX and internal['prov'].get('generalist'):
        expected_room = max(0., expected_room - min(C.PROVISIONAL_KA_MAX_SHARE * dict(internal['slots'])['generalist'],
            len(internal['prov']['generalist']) * C.PROVISIONAL_KA_FRACTION * internal['per_entry']))
    expected_each = min(internal['per_entry'], expected_room / len(ids))
    assert float(each) == float(f'{expected_each:.6f}')
    assert float(room) == float(f'{expected_room:.6f}')
    assert [int(row.split(':')[0]) for row in rows.split(', ')] == ids
    for row in rows.split(', '):
        uid, credit, topup = row.split(':')
        own = internal['own'].get(int(uid), 0.)
        assert float(credit) == float(f'{own:.6f}')
        assert float(topup) == float(f"{internal['paid'][int(uid)]:.6f}")
        assert float(topup) == float(f'{max(0., expected_each-own):.6f}')
    assert math.isclose(sum(vector.values()), 1.)


@pytest.mark.parametrize('mode', ['feed', 'legacy'])
@pytest.mark.parametrize('case', ['empty', 'no_candidate', 'off', 'no_gain', 'no_slot'])
def test_no_generalist_unit_is_explicit(caplog, monkeypatch, mode, case):
    base, candidates, kwargs = {3:.4, 10:.6}, [19], {}
    if case == 'empty':
        base = {}
    if case == 'no_candidate':
        candidates = []
    if case == 'off':
        kwargs['keepalive_share'] = 0.
    if case == 'no_gain':
        base = {3:.4, 10:.001, 19:.599}
    if case == 'no_slot':
        monkeypatch.setattr(C, 'profile_for', lambda *a: C.SplitProfile('none', .5, ()))
    _, line, _ = compute(caplog, monkeypatch, mode, base, candidates, **kwargs)
    assert line == 'generalist keep-alive unit: none'


@pytest.mark.parametrize('mode', ['feed', 'legacy'])
@pytest.mark.parametrize('burn_allocation', [False, True])
def test_engine_allocation_variants_log_once(tmp_path, caplog, monkeypatch, mode, burn_allocation):
    from test_ka_feed import remote_stack, row, feed, NOW as FEED_NOW
    from reax_subnet import engvalidator as E
    from reax_subnet.owner_alloc import Shares
    path = tmp_path/'feed.json'
    path.write_text(__import__('json').dumps(feed([row(1)])))
    if mode == 'feed':
        monkeypatch.setenv('REAX_KA_FEED_PATH', str(path))
    else:
        monkeypatch.delenv('REAX_KA_FEED_PATH', raising=False)
    monkeypatch.setattr(C, 'NEED_RULE', True)
    engine, _, snap = remote_stack(tmp_path, [row(1)])
    monkeypatch.setattr(E._need, 'decide_with_starts', lambda *a: (.4, {'generalist': FEED_NOW-20*86400}))
    class Plan:
        def __init__(self, *a):
            self.shares = Shares()
            self.shares.inference = {44:.01}
            self.shares.update(self.shares.inference)
            if burn_allocation:
                self.shares.burn = {45:.01}
                self.shares.update(self.shares.burn)
        def attach(self, *vectors):
            self.vectors = vectors
    monkeypatch.setattr(E._owner_alloc, 'Plan', Plan)
    original = E.combine_weights
    calls = []
    def combine(*args, **kwargs):
        calls.append(kwargs['log_generalist_unit'])
        return original(*args, **kwargs)
    monkeypatch.setattr(E, 'combine_weights', combine)
    with caplog.at_level(logging.INFO, logger=C.log.name):
        weights = engine.weights({0:.1,250:.9}, snap, set(), .1)
    assert calls == ([True, False, False] if burn_allocation else [True, False])
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('generalist keep-alive unit:')]
    assert len(lines) == 1
    assert UNIT_RE.fullmatch(lines[0])
    assert weights[44] > 0
    if burn_allocation:
        assert weights[45] > 0
