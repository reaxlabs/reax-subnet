"""Most exposed first (switch ``CS.KA_EXPOSED_FIRST``): where one keep-alive per coldkey is chosen, the uid closest to pruning gets it.

Live case 26 Oct 2026: engine uids 165 (immune to 9372809) and 223 (out of immunity, waiting) and champion 62 share a coldkey; when
165's window opened the older commitment won and the exposed 223 dropped to 0. The limit (one per coldkey) is unchanged."""

import pytest
from spec_helpers import snapshot
from test_keepalive import DF
from test_waiting_ka_hotfix import B0, CB_NOW, DA, DB, THU, _commit_block, _esub, _neuron, _queued, no_opening  # noqa: F401

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet.engcomp import TrackState
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import compute_provisional

@pytest.fixture(autouse=True)
def one_per_coldkey(monkeypatch):
    """These regressions isolate the previous one-entry exposure pick."""
    monkeypatch.setattr(CS, "KA_PROV_TWO_PER_COLDKEY", False)


H = 3600.0
IMMUNE_REG = CB_NOW - 1000            # immune for 6200 more blocks
IMMUNE_REG_LATE = CB_NOW - 200        # immune longer
OUT_REG = CB_NOW - 20_000             # out of immunity
OUT_REG_OLDER = CB_NOW - 30_000       # out of immunity for longer


def _gen(regs, cbs):
    """Generalist snapshot: uids 1.. share coldkey ``ck``; ``regs``/``cbs`` per uid (hours ago for the commitment)."""
    st = ST.SpecState()
    ns = []
    for uid, (reg, cb) in enumerate(zip(regs, cbs), start=1):
        _queued(st, uid, first_seen=THU - 30 * H)
        ns.append(_neuron(uid, cb=_commit_block(cb), coldkey="ck", reg=reg))
    return st, snapshot(CB_NOW, ns)


def test_generalist_exposed_beats_the_older_immune_commitment():
    st, snap = _gen([IMMUNE_REG, OUT_REG], [5.0, 2.0])                   # uid 1: older commitment, immune; uid 2: exposed
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [2]
    st, snap = _gen([OUT_REG, IMMUNE_REG], [5.0, 2.0])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [1]


def test_generalist_both_immune_the_earliest_immunity_end_first():
    st, snap = _gen([IMMUNE_REG_LATE, IMMUNE_REG], [5.0, 2.0])           # uid 2 registered earlier: its immunity ends first
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [2]


def test_generalist_both_exposed_the_longest_exposed_first_then_the_old_order():
    st, snap = _gen([OUT_REG, OUT_REG_OLDER], [5.0, 2.0])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [2]
    st, snap = _gen([OUT_REG, OUT_REG], [5.0, 2.0])                      # a tie: the old order (earlier commitment block)
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [1]


def test_generalist_unknown_registration_counts_as_exposed_after_the_known_ones():
    st, snap = _gen([0, IMMUNE_REG], [5.0, 2.0])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [1]       # unknown = exposed, beats the immune one
    st, snap = _gen([0, OUT_REG], [2.0, 5.0])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [2]       # known exposed before unknown exposed


def test_generalist_the_limit_stays_one_per_coldkey_and_other_coldkeys_are_untouched():
    st, snap = _gen([IMMUNE_REG, OUT_REG, IMMUNE_REG], [5.0, 2.0, 4.0])
    ns = list(snap.neurons)
    ns[2] = _neuron(3, cb=_commit_block(4.0), coldkey="other", reg=IMMUNE_REG)
    assert compute_provisional(st, [DF], snapshot(CB_NOW, ns), now=THU)[DF] == [2, 3]       # uid 2 for "ck" (keeps the place of the older commitment), uid 3 alone on "other"


def test_generalist_switch_off_is_the_old_order(monkeypatch):
    monkeypatch.setattr(CS, "KA_EXPOSED_FIRST", False)
    st, snap = _gen([IMMUNE_REG, OUT_REG], [5.0, 2.0])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [1]


def test_generalist_unverified_cap_falls_back_to_the_next_of_the_coldkey(monkeypatch):
    """The exposed entry is unverified and the cap is full: a verified entry of the same coldkey is still paid."""
    monkeypatch.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 0)
    st = ST.SpecState()
    _queued(st, 1, first_seen=THU - 30 * H)                              # verified, immune
    _queued(st, 2, fetched=False, private=True, first_seen=THU - 30 * H)  # unverified, exposed
    snap = snapshot(CB_NOW, [_neuron(1, cb=_commit_block(5.0), coldkey="ck", reg=IMMUNE_REG),
                             _neuron(2, cb=_commit_block(2.0), coldkey="ck", reg=OUT_REG)])
    assert compute_provisional(st, [DF], snap, now=THU)[DF] == [1]


# ------------------------------------------------------------------------------------------------ engine
def test_engine_exposed_uid_is_kept_when_the_immune_one_shares_its_coldkey(no_opening):
    """165 (immune, older commitment) and 223 (exposed): both inside their windows, one coldkey -> 223."""
    a, b = _esub(165, "w165", DA, B0), _esub(223, "w223", DB, B0 + 100)
    st = TrackState()
    ck = {165: "ck", 223: "ck"}
    now = B0 + 3000                                                  # the immune one is inside the 1500-block lead
    reg = {165: now - 6000, 223: now - 20_000}
    assert EC.waiting_uids(st, [a, b], now, ck, reg) == [223]
    reg = {165: now - 20_000, 223: now - 6000}
    assert EC.waiting_uids(st, [a, b], now, ck, reg) == [165]


def test_engine_both_immune_earliest_end_first_and_old_order_without_registration(no_opening):
    a, b = _esub(54, "w54", DA, B0), _esub(173, "w173", DB, B0 + 100)
    now = B0 + 3000
    ck = {54: "ck", 173: "ck"}
    assert EC.waiting_uids(TrackState(), [a, b], now, ck, {54: now - 6000, 173: now - 6900}) == [173]
    assert EC.waiting_uids(TrackState(), [a, b], now, ck, {54: now - 6900, 173: now - 6000}) == [54]
    assert EC.waiting_uids(TrackState(), [a, b], now, ck, None) == [54]               # no snapshot data: the old order
    assert EC.waiting_uids(TrackState(), [a, b], now, ck, {54: 0, 173: 0}) == [54]


def test_engine_other_coldkeys_keep_their_order_and_the_switch_off_is_the_old_order(no_opening, monkeypatch):
    a, b, c = _esub(1, "w1", DA, B0), _esub(2, "w2", DB, B0 + 100), _esub(3, "w3", "d" * 64, B0 + 200)
    now = B0 + 3000
    ck = {1: "x", 2: "x", 3: "y"}
    reg = {1: now - 6000, 2: now - 20_000, 3: now - 6000}
    assert EC.waiting_uids(TrackState(), [a, b, c], now, ck, reg) == [2, 3]
    monkeypatch.setattr(CS, "KA_EXPOSED_FIRST", False)
    assert EC.waiting_uids(TrackState(), [a, b, c], now, ck, reg) == [1, 3]


def test_exposure_key_and_switch_default():
    assert CS.KA_EXPOSED_FIRST is True
    assert CS.exposure_key(100, 100 + CS.CHAIN_IMMUNITY_BLOCKS - 1)[0] == 1          # immune
    assert CS.exposure_key(100, 100 + CS.CHAIN_IMMUNITY_BLOCKS)[0] == 0              # immunity is over
    for bad in (None, 0, -5, True, "x"):
        assert CS.exposure_key(bad, 1000) == (0, 1 << 62)
