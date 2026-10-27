"""Keep-alive gauntlet, commit-reveal cases (#29715/#29724/#29731): generated reveal states over the reader paths.

Oracle: a non-public entry keeps prune protection (keep-alive) until its reveal deadline unless void or re-gated;
shares (leaders) need the reveal receipt. Late receipt, void or deadline passed: 0 everywhere."""

import random

import pytest
from test_commit_reveal_readers import DUE, _pay, _private
from test_keepalive import NOW

from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
H = 3600.0


def _oracle(nonpublic, due, revealed, void, lost, now):
    if void:
        return False, False
    if not nonpublic or due is None:
        return True, True
    share = revealed is not None and revealed <= due and lost is None
    return share or (revealed is None and now <= due), share


def _public_twin(status, now):
    st = ST.SpecState()
    lead, ka = _pay(st, [_private(st, 1, status, 40.0, nonpublic=False)], now=now)
    return bool(lead), bool(ka)


@pytest.mark.parametrize("seed", range(4))
def test_generated_reveal_states_keepalive_and_share(seed):
    """Differential: the reveal gate may only remove what the same entry, public, gets at the same clock."""
    rng = random.Random(seed)
    protected = 0
    for _ in range(400):
        st = ST.SpecState()
        status = rng.choice([ST.PENDING_CONFIRM, ST.CHAMPION])
        nonpublic = rng.random() < 0.85
        due = rng.choice([None, DUE, NOW + rng.uniform(-30, 30) * H])
        revealed = rng.choice([None, None, NOW - H, (due or NOW) + rng.choice([-1.0, 1.0])])
        void, lost = rng.random() < 0.15, rng.choice([None, None, None, NOW])
        # the clock never runs before the fixture's NOW (entries' windows start there; future timestamps fail closed)
        now = rng.choice([NOW, NOW + 1.0] if due is None or due < NOW + 1.0 else [NOW, due - 1.0, due + 1.0])
        s = _private(st, 1, status, 40.0, reveal_due_ts=due, revealed_ts=revealed, reveal_void=void, public_lost_ts=lost)
        s.nonpublic = nonpublic
        want_ka, want_share = _oracle(nonpublic, due, revealed, void, lost, now)
        lead, ka = _pay(st, [s], now=now)
        twin_lead, twin_ka = _public_twin(status, now)
        case = (status, nonpublic, due, revealed, void, lost, now)
        assert ST.reveal_allows_keepalive(s, now) is want_ka, case
        assert ST.reveal_allows_payment(s, now) is want_share, case
        assert bool(ka) is (want_ka and twin_ka), case   # the waiting entry is never at 0 before its deadline
        assert bool(lead) is (want_share and twin_lead), case   # no share without the receipt
        protected += bool(ka and not want_share)
    assert protected >= 20                               # the generator really reaches the slot..deadline window
