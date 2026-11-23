"""Validator reads the worker's optional cutoff hold without duplicating restart payments."""
import json
from datetime import datetime, timezone

import pytest

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E, review as RV, state as ST
from spec_helpers import commit, neuron, snapshot
from test_generalist_commit_cutoff import CUT, SLOT, NEXT, slots, sub  # noqa: F401

THU = SLOT + 30 * 60
SUN = datetime(2026, 11, 1, 12, tzinfo=timezone.utc).timestamp()


def paths(st, subs, now):
    snap = snapshot(10000, [neuron(s.uid, commitment=commit("generalist", s.repo)) for s in subs])
    return (E.compute_keepalive(st, ["generalist"], snap, floors={"generalist": 0},
                               nonproc_floors={"generalist": 0}, now=now)["generalist"],
            E.compute_provisional(st, ["generalist"], snap, now=now)["generalist"])


@pytest.mark.usefixtures("slots")
@pytest.mark.parametrize("restart", [False, True])
def test_worker_hold_state_loaded_by_validator_through_sunday(tmp_path, restart):
    s = sub(committed=CUT + 12)
    s.cutoff_hold_until_ts = NEXT
    s.hash_named = "checked-model"
    # Exhaust ordinary pending windows so that only the held exception can pay.
    s.first_pass_ts = SLOT - 500 * 3600
    if restart:
        E.SpecEvaluator._slot_restart(ST.TrackState(track_id="generalist"), s, SLOT, "new champion")
    st = ST.SpecState()
    st.track("generalist").submissions[s.key] = s
    p = tmp_path / "worker.json"
    ST.save(st, p)
    st = ST.load(p)
    s = st.track("generalist").submissions[s.key]
    for now in (THU, SUN):
        assert paths(st, [s], now) == ([(s.uid, 80.0)], [])
    assert not E._cutoff_held(s, NEXT + 60)
    assert paths(st, [s], NEXT + 60)[0] == []


@pytest.mark.usefixtures("slots")
@pytest.mark.parametrize("gate", ["review", "copy", "reveal", "score"])
def test_held_restart_preserves_validator_gates(gate):
    s = sub(committed=CUT + 12)
    s.cutoff_hold_until_ts, s.hash_named = NEXT, "checked-model"
    E.SpecEvaluator._slot_restart(ST.TrackState(track_id="generalist"), s, SLOT, "new champion")
    if gate == "review":
        s.history.insert(0, {"kind": RV.FLAG_KIND, "flags": ["reparametrised"]})
    elif gate == "copy":
        s.delta_checked = False
    elif gate == "reveal":
        s.nonpublic, s.reveal_due_ts = True, SLOT - 1
    else:
        s.score = None
    st = ST.SpecState()
    st.track("generalist").submissions[s.key] = s
    assert paths(st, [s], THU) == ([], [])


@pytest.mark.usefixtures("slots")
def test_old_worker_state_without_optional_fields_loads_unchanged(tmp_path):
    s = sub()
    s.first_pass_ts = THU - 3600
    st = ST.SpecState()
    st.track("generalist").submissions[s.key] = s
    p = tmp_path / "old-worker.json"
    ST.save(st, p)
    data = json.loads(p.read_text())
    row = data["tracks"]["generalist"]["submissions"][s.key]
    for field in ("chain_commit_block", "chain_commit_ts", "cutoff_hold_until_ts"):
        row.pop(field)
    p.write_text(json.dumps(data))
    loaded = ST.load(p)
    old = loaded.track("generalist").submissions[s.key]
    assert (old.chain_commit_block, old.chain_commit_ts, old.cutoff_hold_until_ts) == (None, None, None)
    # Golden bytes independently checked against base 7f81cfd on this same fixture.
    assert json.dumps(paths(loaded, [old], THU)[0]).encode() == b'[[2, 80.0]]'
    assert paths(loaded, [old], THU) == paths(st, [s], THU)


@pytest.mark.usefixtures("slots")
def test_held_restarts_keep_share_caps():
    st = ST.SpecState()
    subs = [sub(u, CUT + 12) for u in range(2, 14)]
    for s in subs:
        s.cutoff_hold_until_ts, s.hash_named = NEXT, f"checked-{s.uid}"
        E.SpecEvaluator._slot_restart(st.track("generalist"), s, SLOT, "new champion")
        st.track("generalist").submissions[s.key] = s
    ka, provisional = paths(st, subs, SUN)
    assert provisional == []
    weights = CS.combine_weights({100: 1.0}, None, 0.0, {}, keepalive={"generalist": ka})
    share = CS.profile_for().tracks[0]
    paid = sum(weights.get(s.uid, 0) for s in subs)
    assert all(weights.get(s.uid, 0) > 0 for s in subs)
    assert 0 < paid <= CS.KEEPALIVE_MAX_SHARE * share + 1e-12
