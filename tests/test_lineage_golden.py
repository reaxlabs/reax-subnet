"""rc11.post5 golden: with REAX_SPEC_HOLD / REAX_SPEC_REVIEW_CLEAR empty and honest fine-tunes of a structured base (the
lineage plan runs and is empty), the evaluator's cycle outputs, keep-alive and state file are those of 1.2.0rc11.post4.

``GOLDEN`` is the digest this file produced on b510d79 + 74b4c2b (rc11.post4 with the clock pins), where neither
spec/lineage.py nor spec/review.py exists; this file imports neither, so it runs unchanged on both.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from lineage_helpers import toy_base, tune
from spec_helpers import commit, neuron, snapshot
from test_copycheck2 import _server_env
from test_spec_evaluator import TRACK

from reax_subnet.spec import config as C
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

GOLDEN = "787d8f0d790ce1636d2f21c13585dbfaf8292331e5ea0f09f408b2eb78c559bc"


@pytest.fixture(autouse=True)
def small_margin(monkeypatch):
    monkeypatch.setattr(M, "MARGIN_FLOOR", 0.3)
    monkeypatch.setattr(M, "SE_MULTIPLE", 1.0)
    monkeypatch.delenv("REAX_SPEC_HOLD", raising=False)
    monkeypatch.delenv("REAX_SPEC_REVIEW_CLEAR", raising=False)


async def _golden_run(tmp_path) -> str:
    """A champion, an honest better fine-tune that takes the title, a weaker one; digest of every cycle's output, the
    keep-alive after it, and the final state file (all of it but its write time)."""
    base = toy_base()
    env = _server_env(tmp_path, base)
    h = hashlib.sha256()
    env.add(1, "a/m1", 0.70, weights=tune(base, 30, 0.05))
    env.add(2, "b/m2", 0.90, weights=tune(base, 31, 0.05))
    env.add(3, "c/m3", 0.60, weights=tune(base, 32, 0.05))
    one, three = {1: "a/m1"}, {1: "a/m1", 2: "b/m2", 3: "c/m3"}
    for commits, adv in ((one, 0), (one, 13 * 3600), (three, 60), (three, 13 * 3600), (three, 3601)):
        env.clock.advance(adv)
        out = await env.cycle(commits)
        snap = snapshot(1000, [neuron(u, commit(TRACK, r)) for u, r in commits.items()])
        ka = env.ev.keepalive(C.DEFAULT_TRACKS[:1], snap)
        h.update(json.dumps([out, ka], sort_keys=True, default=str).encode())
    assert [env.sub(u).status for u in (1, 2, 3)] == [ST.FORMER, ST.CHAMPION, ST.NOT_BETTER]   # the path it pins
    with open(env.cfg.state_path) as fh:
        state = json.load(fh)
    state.pop("saved_at", None)    # the wall clock of the write
    h.update(json.dumps(state, sort_keys=True).encode())
    return h.hexdigest()


async def test_golden_empty_lists_and_honest_entries_match_rc11_post4(tmp_path):
    got = await _golden_run(tmp_path)
    assert got == GOLDEN, got
