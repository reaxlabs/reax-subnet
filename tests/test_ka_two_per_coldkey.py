"""Two paid waiting entries per coldkey/track, with bounded caps and exposure ordering."""
from dataclasses import replace

from spec_helpers import snapshot
from test_ka_exposed_first import IMMUNE_REG, OUT_REG, OUT_REG_OLDER, _gen
from test_keepalive import DF
from test_waiting_ka_hotfix import B0, CB_NOW, DA, DB, THU, _esub, _neuron, _commit_block, _queued, no_opening  # noqa: F401

from reax_subnet import competition_split as CS, elements, engcomp as EC
from reax_subnet.engvalidator import EngineCompetition
from reax_subnet.s1validator import S1Validator
from reax_subnet.spec import config as C, state as ST
from reax_subnet.spec.evaluator import compute_provisional


def test_generalist_two_infra_waits_on_the_same_coldkey_are_paid(monkeypatch):
    st, snap = _gen([IMMUNE_REG, OUT_REG], [5, 2])
    for s in st.tracks[DF].submissions.values():
        s.coldkey = "ck"
        s.wait_kind = ST.WAIT_INFRA
    assert CS.KA_PROV_TWO_PER_COLDKEY is True
    assert C.MAX_OPEN_PER_COLDKEY_TRACK == 2
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [2, 1]
    monkeypatch.setattr(CS, "KA_PROV_TWO_PER_COLDKEY", False)
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [2]


def test_generalist_three_candidates_keep_two_most_exposed_and_other_coldkeys():
    st, snap = _gen([IMMUNE_REG, OUT_REG, OUT_REG_OLDER], [5, 2, 3])
    ns = list(snap.neurons) + [_neuron(4, cb=_commit_block(1), coldkey="other", reg=OUT_REG)]
    _queued(st, 4)
    assert compute_provisional(st, [DF], snapshot(CB_NOW, ns), now=THU)[DF] == [3, 4, 2]   # final critic F-A: first slots of all coldkeys before second slots


def test_generalist_unverified_cap_still_binds_with_two_slots(monkeypatch):
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 1)
    st = ST.SpecState()
    for uid in (1, 2):
        _queued(st, uid, fetched=False, private=True, first_seen=THU - 3 * 3600)
    ns = [_neuron(uid, cb=_commit_block(2), coldkey="ck", reg=OUT_REG) for uid in (1, 2)]
    assert compute_provisional(st, [DF], snapshot(CB_NOW, ns), now=THU)[DF] == [1]



def test_second_slot_does_not_jump_the_global_unverified_queue(monkeypatch):
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 2)
    st = ST.SpecState()
    for uid in (1, 2, 3):
        _queued(st, uid, fetched=False, private=True, first_seen=THU - 5 * 3600)
    ns = [_neuron(1, cb=_commit_block(4), coldkey="ck", reg=OUT_REG),
          _neuron(2, cb=_commit_block(3), coldkey="other", reg=OUT_REG),
          _neuron(3, cb=_commit_block(2), coldkey="ck", reg=OUT_REG)]
    assert compute_provisional(st, [DF], snapshot(CB_NOW, ns), now=THU)[DF] == [1, 2]

def test_engine_two_most_exposed_per_coldkey_with_switch_rollback(no_opening, monkeypatch):
    subs = [_esub(i, f"w{i}", d, B0 + i * 100) for i, d in enumerate((DA, DB, "c" * 64, "d" * 64), 1)]
    now = B0 + 3000
    ck = {1: "ck", 2: "ck", 3: "ck", 4: "other"}
    reg = {1: now - 6000, 2: now - 20_000, 3: now - 30_000, 4: now - 20_000}
    assert EC.waiting_uids(EC.TrackState(), subs, now, ck, reg) == [3, 2, 4]
    monkeypatch.setattr(CS, "KA_EXPOSED_FIRST", False)
    assert EC.waiting_uids(EC.TrackState(), subs, now, ck, reg) == [1, 2, 4]
    monkeypatch.setattr(CS, "KA_PROV_TWO_PER_COLDKEY", False)
    assert EC.waiting_uids(EC.TrackState(), subs, now, ck, reg) == [1, 4]
    monkeypatch.setattr(CS, "KA_PROV_TWO_PER_COLDKEY", True)
    monkeypatch.setattr(CS, "WAITING_KA_HOTFIX", False)
    legacy = EC.TrackState()
    legacy.first_seen = {s.digest: (s.block, s.hotkey) for s in subs}
    # Older window is shorter; isolate its one-slot limit while all candidates are inside it.
    monkeypatch.setattr(CS, "ENGINE_KA_MAX_HOURS", 120)
    assert EC.waiting_uids(legacy, subs, now, ck) == [1, 4]


def test_dual_role_engine_reaches_real_weights_waiting_path(tmp_path, monkeypatch, no_opening):
    """Round splits Raw fields; compute_weights uses _competition_view before EngineCompetition.weights.
    The candidate needs no serving weight (e.g. its serving failed); another uid supplies the serving pool."""
    now = 9_400_000
    inf = "reax-s1/1:s1-fast:" + "a" * 16
    eng = "reax-eng/1:s1-fast:ghcr.io/r/e@sha256:" + DA
    n = replace(_neuron(1, cb=now - 100, coldkey="ck", reg=now - 20_000),
                commitment=inf + eng, commitment_fields=(inf, eng))
    raw = snapshot(now, [n])
    views = elements.role_views(raw)
    # The real round caches both views; exercise that route and the fallback for callers with a raw snapshot.
    validator = object.__new__(S1Validator)
    validator._latest_snap, validator._latest_comp = views.inference, views.competition
    assert validator._competition_view(views.inference) is views.competition
    csnap = validator._competition_view(raw)
    comp = EngineCompetition(object(), {"s1-fast": "ref"}, str(tmp_path / "eng.json"), start_utc=None)
    comp.context_ok = True
    comp.engine_keepalive = True
    monkeypatch.setattr(comp, "engine_active", lambda: True)
    assert comp.submissions(raw, set())["s1-fast"] == []   # raw concatenation is intentionally not parsed
    assert [s.uid for s in comp.submissions(csnap, set())["s1-fast"]] == [1]
    weights = comp.weights({9: 1.0}, csnap, set())
    assert weights[1] > 0
