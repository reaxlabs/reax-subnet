"""Differential: with no commit-reveal records the readers change nothing. ``tests/data/post7_1_ka2_golden.json`` was
generated on 035897d (reax-validator 1.2.0rc11.post7.1-ka2) with ``_scenario()``; this branch must reproduce it exactly.
Regenerate only on the base commit: ``GOLDEN_OUT=<out.json> pytest tests/test_commit_reveal_differential.py``."""
import json
import os

import pytest
from test_keepalive import DF, NOW, _snap, _sub
from test_provisional_ka import _p, _snapr

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "post7_1_ka2_golden.json")


def _state_bytes(st, tmp_path):
    ST.save(st, str(tmp_path / "state.json"))
    d = json.loads((tmp_path / "state.json").read_text())
    d.pop("saved_at", None)    # wall clock
    return json.dumps(d, sort_keys=True)


def _scenario(tmp_path):
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0), _sub(st, 2, ST.PENDING_CONFIRM, 30.0), _sub(st, 3, ST.FORMER, 35.0),
            _sub(st, 4, ST.NOT_BETTER, 28.0), _sub(st, 5, ST.QUEUED, None), _sub(st, 6, ST.PENDING_VERIFY, None, checked=False),
            _sub(st, 7, ST.REJECTED, 50.0), _sub(st, 8, ST.PENDING_CONFIRM, 32.0, derived="y"),
            _sub(st, 9, ST.PENDING_CONFIRM, 31.0, coldkey="c8"), _sub(st, 10, ST.QUEUED, None, coldkey="c8")]
    snap = _snap(subs)
    leaders = E.compute_leaders(st, [DF], snap, now=NOW)
    ka = E.compute_keepalive(st, [DF], snap, {DF: 24.8}, now=NOW)
    st2 = ST.SpecState()
    w = [_p(st2, 1), _p(st2, 2, status=ST.PENDING_VERIFY, checked=False), _p(st2, 3, coldkey="cw"), _p(st2, 4, coldkey="cw"),
         _p(st2, 5, coldkey="cw")]
    for s in w:
        s.provisional_since_ts = None
    psnap = _snapr(w, cblocks={s.uid: 19990 - s.uid for s in w})
    cw = E._committed_waiting(st2, [DF], psnap, NOW)
    marked = E.mark_provisional(st2, [DF], psnap, NOW)
    since = {s.uid: s.provisional_since_ts for s in w}
    prov = E.compute_provisional(st2, [DF], psnap, now=NOW)
    return json.loads(json.dumps({"leaders": leaders, "keepalive": ka, "committed_waiting": cw, "marked": marked,
                                  "since": since, "provisional": prov, "state_bytes": _state_bytes(st, tmp_path)},
                                 sort_keys=True, default=str))


def test_weights_without_commit_reveal_records_equal_post7_1_ka2(tmp_path):
    got = _scenario(tmp_path)
    if os.environ.get("GOLDEN_OUT"):
        with open(os.environ["GOLDEN_OUT"], "w") as f:
            json.dump(got, f, sort_keys=True, indent=1)
        return
    with open(GOLDEN) as f:
        assert got == json.load(f)
