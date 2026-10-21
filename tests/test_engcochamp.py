"""Engine opening co-champions (1.2.0rc9.post7, ENGINE_CO_CHAMPIONS): the champion's part is split evenly over the listed
entries, each half needs its own approval, an unpaid half goes to inference, and the challenger bar uses the higher score."""

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet.engcomp import Champion, EvalResult, TrackState
from test_engfair import BASE, DA, DB, DC, SUBS, Gate, _comp, _ok, _res, _snap, _sub


@pytest.fixture
def co(monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPIONS", {"s1-fast": ((DB, "new"), (DA, "old"))})
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPION_BAR", {"s1-fast": 1.9})


def _full(comp):
    return CS.profile_for().engine * EC.champion_fraction(comp.reward, 2000 - 1500)


def test_both_approved_split_the_champion_part(tmp_path, co):
    comp = _comp(tmp_path, Gate(_ok(("old", DA), ("new", DB))), SUBS)
    w = comp.weights(BASE, _snap(SUBS), set())
    assert w[2] == pytest.approx(w[3]) and w[2] == pytest.approx(_full(comp) / 2, rel=1e-9)


def test_one_approved_gets_half_the_other_half_goes_to_inference(tmp_path, co):
    for ok, paid, unpaid in ((("old", DA), 2, 3), (("new", DB), 3, 2)):
        comp = _comp(tmp_path, Gate(_ok(ok)), SUBS)
        w = comp.weights(BASE, _snap(SUBS), set())
        assert unpaid not in w and w[paid] == pytest.approx(_full(comp) / 2, rel=1e-9)
        assert w[9] == pytest.approx(w[8]) and w[9] == pytest.approx((1 - _full(comp) / 2) / 2, rel=1e-9)


def test_nobody_approved_is_unchanged(tmp_path, co):
    comp = _comp(tmp_path, Gate(), SUBS)
    assert comp.weights(BASE, _snap(SUBS), set()) == BASE


def test_withdrawn_or_invalid_co_champion_is_not_paid(tmp_path, co):
    comp = _comp(tmp_path, Gate(_ok(("old", DA), ("new", DB))), SUBS)
    w = comp.weights(BASE, _snap(SUBS[1:]), set())       # uid 2 (DA) withdrew
    assert 2 not in w and w[3] == pytest.approx(_full(comp) / 2, rel=1e-9)
    comp.states["s1-fast"].results[DA] = EvalResult(DA, "s1-fast", False, score=0.0, block=100)
    w = comp.weights(BASE, _snap(SUBS), set())           # DA's last result invalid
    assert 2 not in w and w[3] == pytest.approx(_full(comp) / 2, rel=1e-9)


def test_no_gate_pays_both(tmp_path, co):
    comp = _comp(tmp_path, None, SUBS)
    w = comp.weights(BASE, _snap(SUBS), set())
    assert w[2] == pytest.approx(_full(comp) / 2, rel=1e-9) and w[3] == pytest.approx(w[2])


def test_champion_outside_the_set_is_the_normal_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPIONS", {"s1-fast": ((DC, "x"), (DA, "old"))})
    comp = _comp(tmp_path, Gate(_ok(("old", DA), ("new", DB))), SUBS)
    w = comp.weights(BASE, _snap(SUBS), set())
    # rc10.post1: uid 2 (DA, crowned 100) is a former champion in its rank-2 slot: 17 % of the part, its half of the set's
    assert w[3] == pytest.approx(_full(comp) * (1 - CS.ENGINE_RANK_SPLIT[1]), rel=1e-9)
    assert w[2] == pytest.approx(_full(comp) * CS.ENGINE_RANK_SPLIT[1] / 2, rel=1e-9)


def test_hotkey_must_match(tmp_path, co):
    subs = [_sub(2, "thief", DA, 100), SUBS[1]]
    comp = _comp(tmp_path, Gate(_ok(("old", DA), ("thief", DA), ("new", DB))), subs)
    w = comp.weights(BASE, _snap(subs), set())
    assert 2 not in w and w[3] == pytest.approx(_full(comp) / 2, rel=1e-9)


def test_bar_uses_the_higher_co_champion_score(co):
    p1 = EC.TrackParams("s1-fast", pool="s1-fast", temperature=2.07, confirmations=1)
    m = max(p1.margin, EC.required_engine_margin(p1.reference_noise_std))
    between = 1.85 * (1 + m) if 1.85 * (1 + m) < 1.9 * (1 + m) else None
    assert between is not None
    ch = _sub(3, "new", DB, 1500)
    chal = _sub(5, "c", DC, 1600)
    st = TrackState(results={DB: _res(DB, 1.85, 1500), DC: _res(DC, (1.85 * (1 + m) + 1.9 * (1 + m)) / 2, 1600)},
                    first_seen={DB: (1500, "new"), DC: (1600, "c")}, champion=Champion(DB, "new", 3, 1.85, 1500))
    EC.update_champion(st, [ch, chal], p1, 2000)
    assert st.champion.digest == DB and st.challenger is None  # beats 1.85 x (1+m) but not 1.9 x (1+m)
    st.results[DC] = _res(DC, 1.9 * (1 + m) + 0.01, 1700)
    EC.update_champion(st, [ch, chal], p1, 2100)
    assert st.champion.digest == DC  # crowned (confirmations=1); the co-champion rule ends with it
    assert EC.co_champion_set(st.champion) is None


def test_engine_keepalive_is_taken_from_both_halves():
    base = {i: 1.0 for i in range(10, 20)}
    w0 = CS.combine_weights(base, None, 0.0, {}, engine_cochampions={2: 0.5, 3: 0.5}, engine_waiting=[7])
    w1 = CS.combine_weights(base, 3, 1.0, {}, engine_waiting=[7])
    assert w0[2] == pytest.approx(w0[3]) and w0[2] + w0[3] == pytest.approx(w1[3], rel=1e-12)
    assert w0[7] == pytest.approx(w1[7], rel=1e-12)


def test_cochampions_equal_a_single_champion_in_total():
    base = {i: 1.0 for i in range(10, 20)}
    w0 = CS.combine_weights(base, None, 0.0, {}, engine_cochampions={2: 0.4, 3: 0.4})
    w1 = CS.combine_weights(base, 3, 0.8, {})
    assert w0[2] + w0[3] == pytest.approx(w1[3], rel=1e-12) and w0[10] == pytest.approx(w1[10], rel=1e-12)


def test_release_constants():
    pairs = CS.ENGINE_CO_CHAMPIONS["s1-fast"]
    assert ("576fea84b347fc85160f788d737db616be4b5ca4cde19268e0ce1d1f5cdfe12f",
            "5FLhkmTrKBUCJWnmCEY7xzcuVe6Mpk7FUgLdwfj7SbH33wo2") in pairs
    assert ("ec2ee6ed3df69903bf9df5420cde8f869280647ce7d23a3837a2397b2ce34272",
            "5H6iSGe3t74QSUq2RgKtA4sET9EauSYsNBkEjGynW4jJ9htc") in pairs
    assert len(pairs) == 2 and CS.ENGINE_CO_CHAMPION_BAR["s1-fast"] == pytest.approx(2.8233, abs=1e-4)


def test_decay_runs_from_the_opening_crowning_after_a_recrowning(tmp_path, co, monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPION_CROWNED", {"s1-fast": 1500})
    comp = _comp(tmp_path, Gate(_ok(("old", DA), ("new", DB))), SUBS)
    comp.states["s1-fast"].champion = Champion(DA, "old", 2, 1.5, 1900)  # uid 2 re-crowned later (e.g. after 3 left)
    w = comp.weights(BASE, _snap(SUBS), set())
    assert w[2] == pytest.approx(_full(comp) / 2, rel=1e-9) and w[3] == pytest.approx(w[2])


def test_co_champion_is_never_paid_the_full_share_as_former_champion(tmp_path, co):
    subs = SUBS + [_sub(5, "x", DC, 1800)]
    comp = _comp(tmp_path, Gate(_ok(("old", DA), ("new", DB))), subs)
    st = comp.states["s1-fast"]
    st.results[DC] = _res(DC, 2.5, 1800)
    st.first_seen[DC] = (1800, "x")
    st.champion = Champion(DC, "x", 5, 2.5, 1800)  # an unapproved challenger took the title
    st.history.append({"block": 1800, "event": "crowned", "digest": DC, "hotkey": "x", "score": 2.5})
    # rc10.post1: the set is the former champion, each member its own half of rank 1's share (its own 17 % slot is not
    # paid on top; tests/test_engcofallback.py), never more
    w = comp.weights(BASE, _snap(subs), set())
    half = CS.profile_for().engine * EC.champion_fraction(comp.reward, 2000 - 1500) / 2
    assert 5 not in w and w[2] == pytest.approx(half, rel=1e-9) and w[3] == pytest.approx(half, rel=1e-9)
    assert comp.weights(BASE, _snap(SUBS[1:] + subs[2:]), set())[3] == pytest.approx(half, rel=1e-9)  # uid 2 gone


def test_without_cochampions_the_keepalive_rule_is_exactly_the_old_one():
    base = {i: 1.0 + i / 100 for i in range(10, 20)}
    for frac in (1.0, 0.73, 0.5):
        w = CS.combine_weights(base, 3, frac, {}, engine_waiting=[7, 8])
        assert w[3] > 0 and w[7] == w[8]
