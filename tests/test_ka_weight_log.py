"""post8.11: each weight submission logs the exact compute_keepalive entries inside its vector (rr #36471 (1)), and
compute_keepalive follows the snapshot it is given when a commitment changes or a uid is briefly absent (#36471 (2)).
Observational only: no selection, payout or state change."""
import copy
import logging
from types import SimpleNamespace

import pytest

from reax_subnet import engvalidator as EV
from reax_subnet.spec import evaluator as E, state as ST
from spec_helpers import commit, neuron, snapshot as spec_snapshot
from test_generalist_commit_cutoff import CUT, SLOT, NEXT, slots, sub  # noqa: F401
from test_weight_timing import LAST, arena, snapshot  # noqa: F401

THU = SLOT + 30 * 60
VECTOR = {0: .999, 109: .001}
RECORD = (9385800, {"generalist": [(109, 80.0), (7, 0.123456)]}, {"generalist": [7]})


def ka_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("keep-alive in weight set")]


async def test_each_submission_logs_the_exact_entries_and_sends_the_same_vector(arena, caplog):
    chain, val = arena
    caplog.set_level(logging.INFO)
    val.engine = SimpleNamespace(keepalive_used=lambda: RECORD)
    assert await val.maybe_set_weights(snapshot(LAST + 100))
    line, = ka_lines(caplog)
    assert line == (f"keep-alive in weight set at block {LAST + 100} (computed at block 9385800): "
                    "entries generalist=[109:80.0, 7:0.123456]; protected generalist=[7]")  # exact round-trip scores
    sent = chain.set_calls[-1]

    def broken():
        raise RuntimeError("record")
    val.engine = SimpleNamespace(keepalive_used=broken)
    caplog.clear()
    chain.set_calls.clear()
    val._weights_task = None
    chain.since_update = 10 ** 6  # past the interval again (the fake chain resets it on each set)
    assert await val.maybe_set_weights(snapshot(LAST + 100))
    assert chain.set_calls[-1] == sent  # identical submission with or without a usable record
    assert ka_lines(caplog) == [f"keep-alive in weight set at block {LAST + 100}: none computed"]
    assert any("keep-alive record unavailable" in r.getMessage() for r in caplog.records)


async def test_record_is_taken_with_the_vector_not_after_the_rate_limit_await(arena, caplog):
    chain, val = arena
    caplog.set_level(logging.INFO)
    current = {"record": RECORD}
    val.engine = SimpleNamespace(keepalive_used=lambda: current["record"])
    real = val.read_rate_limit

    async def racing(uid):  # a commitment changes between snapshot and send; the engine record moves on
        current["record"] = (9385999, {"generalist": []}, {})
        return await real(uid)
    val.read_rate_limit = racing
    assert await val.maybe_set_weights(snapshot(LAST + 100))
    line, = ka_lines(caplog)
    assert "(computed at block 9385800): entries generalist=[109:80.0, 7:0.123456]" in line


def test_engine_record_resets_when_weights_returns_early():
    eng = object.__new__(EV.EngineCompetition)
    eng._keepalive_used = RECORD  # left over from an earlier set
    eng.active = lambda: False
    assert eng.weights({0: 1.0}, spec_snapshot(10000, []), set()) == {0: 1.0}
    assert eng.keepalive_used() is None  # never a stale list attributed to this vector


def held_state():
    s = sub(committed=CUT + 12)
    s.cutoff_hold_until_ts = NEXT
    s.hash_named = "checked-model"
    s.first_pass_ts = SLOT - 500 * 3600
    st = ST.SpecState()
    st.track("generalist").submissions[s.key] = s
    return st, s


def keepalive(st, neurons, now=THU):
    snap = spec_snapshot(10000, neurons)
    return E.compute_keepalive(st, ["generalist"], snap, floors={"generalist": 0},
                               nonproc_floors={"generalist": 0}, now=now)["generalist"]


@pytest.mark.usefixtures("slots")
def test_commitment_change_between_snapshot_and_set_follows_the_given_snapshot():
    st, s = held_state()
    before = copy.deepcopy(st)
    assert keepalive(st, [neuron(s.uid, commitment=commit("generalist", s.repo))]) == [(s.uid, 80.0)]
    # Same uid, new commitment by the time of the next snapshot: no keep-alive for the old entry from that snapshot.
    assert keepalive(st, [neuron(s.uid, commitment=commit("generalist", s.repo + "-other"))]) == []
    assert st == before  # pure read: the stored entry is untouched


@pytest.mark.usefixtures("slots")
def test_briefly_absent_uid_returns_with_the_same_entry():
    st, s = held_state()
    before = copy.deepcopy(st)
    present = [neuron(s.uid, commitment=commit("generalist", s.repo))]
    assert keepalive(st, present) == [(s.uid, 80.0)]
    assert keepalive(st, []) == []  # absent from this snapshot: nothing paid for it in this set
    assert keepalive(st, present) == [(s.uid, 80.0)]  # back: same entry, nothing reset
    assert st == before
