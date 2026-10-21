"""Protected engine keep-alive (rc9.post8 / rc10, Florian 21 Oct 05:25Z/05:30Z): the bounded engine keep-alive also pays the
champion that awaits its award approval and every former champion that holds a slot (rank 2 / 3, the whole opening
co-champion set) for ENGINE_FORMER_SLOT_BLOCKS after it was dethroned (rc10.post1: one mechanism with the slots). A
keep-alive, not an award: a uid paid anything else (incl. its slot share) gets none."""

import math

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import EvalResult
from test_engcomp import P
from test_engfair import DA, DB, DC, Gate, _comp, _ok, _res, _snap, _sub

SERVE = {u: 1.0 for u in range(10, 20)}
SUBS3 = [_sub(2, "old", DA, 100), _sub(3, "new", DB, 1500), _sub(4, "x", DC, 200)]


def _st(comp):
    return comp.states["s1-fast"]


@pytest.fixture
def ka(monkeypatch):
    monkeypatch.setenv("REAX_ENGINE_KEEPALIVE", "1")


def _comp3(tmp_path, gate):
    comp = _comp(tmp_path, gate, SUBS3)
    _st(comp).results[DC] = _res(DC, 1.6, 200)
    _st(comp).first_seen[DC] = (200, "x")
    return comp


# ------------------------------------------------------------------------------------------------ pure helper
def test_protected_uids_champion_and_newest_former(tmp_path):
    st = _st(_comp3(tmp_path, None))
    assert EC.protected_uids(st, SUBS3, P, 2000, champion_approved=False) == [3, 2]
    assert EC.protected_uids(st, SUBS3, P, 2000, champion_approved=True) == [2]


def test_protected_uids_window_withdrawn_invalid(tmp_path):
    st = _st(_comp3(tmp_path, None))
    end = 1500 + CS.ENGINE_FORMER_SLOT_BLOCKS  # rc10.post1: the slot end (first block without the slot)
    assert EC.protected_uids(st, SUBS3, P, end - 1, champion_approved=True) == [2]
    assert EC.protected_uids(st, SUBS3, P, end, champion_approved=True) == []              # 7 days after dethroning
    assert EC.protected_uids(st, SUBS3, P, end, champion_approved=False) == [3]
    assert EC.protected_uids(st, SUBS3[1:], P, 2000, champion_approved=True) == []         # former withdrew
    st.results[DA] = EvalResult(DA, "s1-fast", False, score=0.0, block=100)
    assert EC.protected_uids(st, SUBS3, P, 2000, champion_approved=True) == []             # former's result invalid
    st.champion = None
    assert EC.protected_uids(st, SUBS3, P, 2000, champion_approved=False) == []


def test_protected_uids_co_set_and_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPIONS", {"s1-fast": ((DA, "old"), (DC, "x"))})
    st = _st(_comp3(tmp_path, None))
    assert EC.protected_uids(st, SUBS3, P, 2000, champion_approved=False) == [3, 2, 4]
    monkeypatch.setattr(CS, "ENGINE_KA_PROTECTED_MAX", 2)
    assert EC.protected_uids(st, SUBS3, P, 2000, champion_approved=False) == [3, 2]


# ------------------------------------------------------------------------------------------------ validator weights
def test_live_situation_nobody_approved_pays_keepalive_to_champion_and_co_set(tmp_path, monkeypatch, ka):
    """uid 56-like champion unapproved, co set {4,198}-like former, nobody approved: today 0 each, now a keep-alive each."""
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPIONS", {"s1-fast": ((DA, "old"), (DC, "x"))})
    comp = _comp3(tmp_path, Gate())
    w = comp.weights(SERVE, _snap(SUBS3), set())
    assert w[2] > 0 and w[3] > 0 and w[4] > 0 and w[2] == pytest.approx(w[3]) == pytest.approx(w[4])
    assert w[2] + w[3] + w[4] <= CS.ENGINE_KA_MAX_SHARE * CS.profile_for().engine * 1.01
    assert math.isclose(sum(w.values()), 1.0)


def test_flag_off_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("REAX_ENGINE_KEEPALIVE", "0")
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPIONS", {"s1-fast": ((DA, "old"), (DC, "x"))})
    comp = _comp3(tmp_path, Gate())
    assert comp.weights(SERVE, _snap(SUBS3), set()) == SERVE


def test_approved_champion_is_paid_and_former_gets_keepalive(tmp_path, ka):
    comp = _comp3(tmp_path, Gate(_ok(("new", DB))))
    w = comp.weights(SERVE, _snap(SUBS3), set())
    # rc10.post1: the unapproved former's 17 % slot goes to inference; it gets the keep-alive instead
    plain_frac = CS.profile_for().engine * EC.champion_fraction(comp.reward, 2000 - 1500) * (1 - CS.ENGINE_RANK_SPLIT[1])
    cap = CS.ENGINE_KA_MAX_SHARE * CS.profile_for().engine
    assert 0 < w[2] <= cap * (1 + 1e-9) and w[3] > w[2]
    assert math.isclose(w[3] + w[2], plain_frac, rel_tol=1e-6)      # the champion pays the former's keep-alive
    assert 4 not in w


def test_former_paid_by_restore_previous_gets_no_keepalive(tmp_path, ka):
    comp = _comp3(tmp_path, Gate(_ok(("old", DA))))
    w = comp.weights(SERVE, _snap(SUBS3), set())
    restored = CS.profile_for().engine * EC.champion_fraction(comp.reward, 2000 - 100)
    cap = CS.ENGINE_KA_MAX_SHARE * CS.profile_for().engine
    assert 0 < w[3] <= cap * (1 + 1e-9)                             # unapproved champion: keep-alive only
    assert math.isclose(w[2] + w[3], restored, rel_tol=1e-6)        # former holds the share and pays the keep-alive


def test_protected_go_first_then_waiting_and_total_cap(tmp_path, monkeypatch, ka):
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPIONS", {"s1-fast": ((DA, "old"), (DC, "x"))})
    comp = _comp3(tmp_path, Gate())
    st = _st(comp)
    import hashlib
    waiters = [_sub(30 + i, f"w{i}", hashlib.sha256(bytes([i])).hexdigest(), 1990) for i in range(6)]
    for s in waiters:
        st.first_seen[s.digest], st.owners[s.digest] = (1990, s.hotkey), s.hotkey
    subs = SUBS3 + waiters
    w = comp.weights(SERVE, _snap(subs), set())
    paid_ka = [u for u in w if u not in SERVE]
    assert {2, 3, 4} <= set(paid_ka)
    assert len(paid_ka) == 3 + CS.ENGINE_KA_MAX_ENTRIES       # 3 protected + the 4 waiting entries (waiting_uids cap)
    assert sum(w[u] for u in paid_ka) <= CS.ENGINE_KA_MAX_SHARE * CS.profile_for().engine * 1.01
    monkeypatch.setattr(CS, "ENGINE_KA_PROTECTED_MAX", 2)    # total cap: ENGINE_KA_MAX_ENTRIES + ENGINE_KA_PROTECTED_MAX
    w = comp.weights(SERVE, _snap(subs), set())
    paid_ka = [u for u in w if u not in SERVE]
    assert {2, 3} <= set(paid_ka) and len(paid_ka) == CS.ENGINE_KA_MAX_ENTRIES + 2
