"""Engine former champions (1.2.0rc10.post1; Florian 21 Oct 2026, DECISIONS 05:25Z + 05:30Z "B").

Decision 1: while a champion outside the opening co-champion set waits for its approval (ENGINE_RESTORE_PREVIOUS), the set
counts as ONE former champion (ranked by its crowning block): each listed entry that is still committed, valid and approved
gets 1/n of rank 1's part, decaying from ENGINE_CO_CHAMPION_CROWNED; an unpaid part goes to the inference part; a single
member is never paid the full share (post7 F2). No member qualifies: the walk goes on to older former champions. The
restored payee's own slot is not held against rank 1 (lead call on doubt 2): with no other slot held it gets the whole
share of its pot, so nothing jumps when its slot ends.

Decision 2: of the title holder's (decayed) part, the previous champion gets 17 % and the one before 8 % while their slot
runs (ENGINE_FORMER_SLOT_BLOCKS = 7 days from the block they were dethroned, or until pushed below rank 3); rank 1 keeps
the rest. Each former champion needs its own approval; an unpaid slot goes to the inference part. A restored former
champion takes rank 1's share (1 - the other held slots). A former champion in its slot gets the protected engine
keep-alive (engcomp.protected_uids) unless it is paid."""

import hashlib

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import Champion, EvalResult, TrackState
from test_engfair import BASE, DA, DB, DC, REF, SUBS, Gate, _comp, _ok, _res, _snap, _sub

DD, DE = (hashlib.sha256(x.encode()).hexdigest() for x in "de")
ENG = CS.profile_for().engine
R2, R3 = CS.ENGINE_RANK_SPLIT[1], CS.ENGINE_RANK_SPLIT[2]
SLOT = CS.ENGINE_FORMER_SLOT_BLOCKS


@pytest.fixture
def co(monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPIONS", {"s1-fast": ((DB, "new"), (DA, "old"))})
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPION_BAR", {"s1-fast": 1.9})
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPION_CROWNED", {"s1-fast": 1500})


def _crown(comp, subs, uid, hotkey, digest, block, score=2.5):
    st = comp.states["s1-fast"]
    st.results[digest] = _res(digest, score, block)
    st.first_seen[digest] = (block, hotkey)
    st.champion = Champion(digest, hotkey, uid, score, block)
    st.history.append({"block": block, "event": "crowned", "digest": digest, "hotkey": hotkey, "score": score})
    return subs + [_sub(uid, hotkey, digest, block)]


def _live(tmp_path, gate):
    """co set {uid 3 (DB, crowned 1500), uid 2 (DA)}; uid 5 (DC) crowned 1800 after them = the current champion."""
    comp = _comp(tmp_path, gate, SUBS)
    return comp, _crown(comp, SUBS, 5, "x", DC, 1800)


def _f(comp, age):
    return EC.champion_fraction(comp.reward, age)


def _w(comp, subs, block=2000, base=BASE):
    return comp.weights(base, _snap(subs, block=block), set())


# ------------------------------------------------------------------ (a)-(d): the live situation in miniature
def test_a_nobody_approved_is_unchanged(tmp_path, co):
    comp, subs = _live(tmp_path, Gate())
    assert _w(comp, subs) == BASE


def test_a_an_approval_for_another_digest_does_not_count(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("x", DD), ("old", DC))))
    assert _w(comp, subs) == BASE


def test_b_one_approved_gets_half_of_rank_1_the_other_half_goes_to_inference(tmp_path, co):
    for ok, paid, unpaid in ((("old", DA), 2, 3), (("new", DB), 3, 2)):
        comp, subs = _live(tmp_path, Gate(_ok(ok)))
        w = _w(comp, subs)
        half = ENG * _f(comp, 500) / 2
        assert unpaid not in w and 5 not in w and w[paid] == pytest.approx(half, rel=1e-12)
        assert w[9] == pytest.approx(w[8]) and w[9] == pytest.approx((1 - half) / 2, rel=1e-12)


def test_c_both_approved_split_rank_1_without_their_own_slot(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    w = _w(comp, subs)
    assert 5 not in w and w[2] == pytest.approx(w[3]) and w[2] == pytest.approx(ENG * _f(comp, 500) / 2, rel=1e-12)
    assert w[2] + w[3] == pytest.approx(ENG * _f(comp, 500), rel=1e-12)  # one share, never more


def test_d_approved_champion_75_plus_unused_8_and_the_set_17_split(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC))))
    w = _w(comp, subs)
    assert w[5] == pytest.approx(ENG * (1 - R2) * _f(comp, 200), rel=1e-12)
    assert w[2] == pytest.approx(ENG * R2 * _f(comp, 200) / 2, rel=1e-12) and w[3] == pytest.approx(w[2], rel=1e-12)


def test_d_one_member_unapproved_its_half_slot_goes_to_inference(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("new", DB), ("x", DC))))
    w = _w(comp, subs)
    assert 2 not in w and w[3] == pytest.approx(ENG * R2 * _f(comp, 200) / 2, rel=1e-12)
    assert w[5] == pytest.approx(ENG * (1 - R2) * _f(comp, 200), rel=1e-12)  # not promoted into the champion


def test_d_slot_ends_seven_days_after_the_dethroning(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC))))
    last = 1800 + SLOT - 1
    w = _w(comp, subs, block=last)
    assert w[2] == pytest.approx(ENG * R2 * _f(comp, last - 1800) / 2, rel=1e-12)
    w = _w(comp, subs, block=last + 1)
    assert 2 not in w and 3 not in w and w[5] == pytest.approx(ENG * _f(comp, SLOT), rel=1e-12)
    assert SLOT == 7 * 86400 // 12


# ------------------------------------------------------------------ newer champions
def test_challenger_after_the_champion_pushes_the_set_to_rank_3(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC))))
    subs = _crown(comp, subs, 6, "y", DD, 1900, score=2.7)  # crowned after uid 5, not approved
    w = _w(comp, subs)
    # uid 5 (rank 2, approved) is restored: rank 1's share less the set's 8 % slot, with its own decay
    assert 6 not in w and w[5] == pytest.approx(ENG * (1 - R3) * _f(comp, 200), rel=1e-12)
    assert w[2] == pytest.approx(ENG * R3 * _f(comp, 100) / 2, rel=1e-12) and w[3] == pytest.approx(w[2], rel=1e-12)


def test_unapproved_newer_former_falls_through_to_the_set(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    subs = _crown(comp, subs, 6, "y", DD, 1900, score=2.7)
    w = _w(comp, subs)
    assert 5 not in w and 6 not in w  # uid 5's 17 % slot is unapproved: inference
    assert w[2] == pytest.approx(ENG * (1 - R2) * _f(comp, 500) / 2, rel=1e-12) and w[3] == pytest.approx(w[2])


def test_two_newer_champions_push_the_set_out(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC), ("y", DD), ("z", DE))))
    subs = _crown(comp, subs, 6, "y", DD, 1900, score=2.7)
    subs = _crown(comp, subs, 7, "z", DE, 1950, score=2.9)
    w = _w(comp, subs)
    pot = ENG * _f(comp, 50)
    assert 2 not in w and 3 not in w
    assert w[7] == pytest.approx(pot * 0.75, rel=1e-12) and w[6] == pytest.approx(pot * R2, rel=1e-12)
    assert w[5] == pytest.approx(pot * R3, rel=1e-12)


# ------------------------------------------------------------------ member checks (F2)
def test_withdrawn_member_never_gives_the_other_the_full_share(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    for gone in (2, 3):  # either member, incl. uid 3 whose crowning ranks the set
        left = [s for s in subs if s.uid != gone]
        w = _w(comp, left)
        assert gone not in w and w[5 - gone] == pytest.approx(ENG * _f(comp, 500) / 2, rel=1e-12)


def test_withdrawn_member_in_the_slot_is_half_a_slot(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC))))
    w = _w(comp, [s for s in subs if s.uid != 2])
    assert 2 not in w and w[3] == pytest.approx(ENG * R2 * _f(comp, 200) / 2, rel=1e-12)


def test_invalid_member_is_not_paid(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    comp.states["s1-fast"].results[DB] = EvalResult(DB, "s1-fast", False, score=0.0, block=1950)
    w = _w(comp, subs)
    assert 3 not in w and w[2] == pytest.approx(ENG * _f(comp, 500) / 2, rel=1e-12)


def test_hotkey_must_match(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("thief", DA), ("new", DB))))
    subs = [_sub(2, "thief", DA, 100) if s.uid == 2 else s for s in subs]
    w = _w(comp, subs)
    assert 2 not in w and w[3] == pytest.approx(ENG * _f(comp, 500) / 2, rel=1e-12)


def test_uid_renumbering_pays_the_hotkey_at_its_new_uid(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    subs = [_sub(7, "old", DA, 100) if s.uid == 2 else _sub(11, "new", DB, 1500) if s.uid == 3 else s for s in subs]
    w = _w(comp, subs)
    assert 2 not in w and 3 not in w and w[7] == pytest.approx(ENG * _f(comp, 500) / 2, rel=1e-12)
    assert w[11] == pytest.approx(w[7])


def test_no_gate_pays_no_former_champion(tmp_path, co):
    comp, subs = _live(tmp_path, None)  # without a gate the champion is paid as before, nobody is "approved"
    w = _w(comp, subs)
    assert 2 not in w and 3 not in w and w[5] == pytest.approx(ENG * (1 - R2) * _f(comp, 200), rel=1e-12)


# ------------------------------------------------------------------ decay (F3)
def test_decay_continues_from_the_opening_crowning(tmp_path, co, monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_CO_CHAMPION_CROWNED", {"s1-fast": 1200})
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    w = _w(comp, subs)
    assert w[2] == pytest.approx(ENG * _f(comp, 800) / 2, rel=1e-12) and w[3] == pytest.approx(w[2])
    for block in (1800 + SLOT - 1, 1800 + SLOT, 60_000, 500_000):  # no jump at the slot end, no restart, floor
        w = _w(comp, subs, block=block)
        assert w[2] == pytest.approx(ENG * _f(comp, block - 1200) / 2, rel=1e-12) and w[3] == pytest.approx(w[2])
    assert _w(comp, subs, block=500_000)[2] == pytest.approx(ENG * comp.reward.champion_floor / 2, rel=1e-12)


def test_slots_decay_with_the_title_holder(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC))))
    for block in (2000, 30_000, 52_000):
        w = _w(comp, subs, block=block)
        assert w[2] / w[5] == pytest.approx(R2 / 2 / (1 - R2), rel=1e-9)


# ------------------------------------------------------------------ walk order, flag
def test_set_counts_once_and_no_member_qualifying_continues_the_walk(tmp_path, co):
    comp = _comp(tmp_path, Gate(_ok(("z", DE))), SUBS)
    st = comp.states["s1-fast"]
    st.results[DE] = _res(DE, 1.2, 50)
    st.first_seen[DE] = (50, "z")
    st.history.insert(0, {"block": 50, "event": "crowned", "digest": DE, "hotkey": "z", "score": 1.2})
    subs = _crown(comp, SUBS + [_sub(4, "z", DE, 50)], 5, "x", DC, 1800)
    # formers: DB@1500 (the set, rank 2), DA@100 (the set again: skipped), DE@50 (rank 3, dethroned at 100)
    assert [f["hotkey"] for f in EC.former_champions(st)] == ["new", "old", "z"]
    ranks = EC.former_ranks(st, "s1-fast", 2000)
    assert [(g["co"], g["share"]) for g in ranks] == [(True, R2), (False, R3)]
    w = _w(comp, subs)  # uid 4 restored: rank 1's share less the set's (unpaid) 17 % slot
    assert 2 not in w and 3 not in w and w[4] == pytest.approx(ENG * (1 - R2) * _f(comp, 1950), rel=1e-12)
    comp.award_gate = Gate(_ok(("z", DE), ("old", DA)))  # one member approved: the set is restored, the walk stops
    w = _w(comp, subs)
    assert 3 not in w and w[2] == pytest.approx(ENG * (1 - R3) * _f(comp, 500) / 2, rel=1e-12)
    assert w[4] == pytest.approx(ENG * R3 * _f(comp, 200), rel=1e-12)  # uid 4 in its own rank-3 slot


def test_restore_flag_off_pays_the_slots_only(tmp_path, co, monkeypatch):
    monkeypatch.setattr(EV, "ENGINE_RESTORE_PREVIOUS", False)
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    w = _w(comp, subs)
    assert 5 not in w and w[2] == pytest.approx(ENG * R2 * _f(comp, 200) / 2, rel=1e-12) and w[3] == pytest.approx(w[2])


def test_dethroned_blocks():
    st = TrackState(history=[
        {"block": 1, "event": "context_changed"},
        {"block": 100, "event": "crowned", "digest": DA, "hotkey": "a"},
        {"block": 200, "event": "crowned", "digest": DB, "hotkey": "b"},
        {"block": 250, "event": "crowned", "digest": DB, "hotkey": "b"},   # re-crowned: dethrones nobody
        {"block": 300, "event": "vacated", "digest": DB, "hotkey": "b"},
        {"block": 400, "event": "crowned", "digest": DC, "hotkey": "c"},   # after a vacated title: dethrones nobody
        {"block": 500, "event": "crowned", "digest": DA, "hotkey": "a"},   # a crowned again: its old entry is gone
        {"block": 600, "event": "crowned", "digest": DD, "hotkey": "d"}])
    assert EC.dethroned_blocks(st) == {(DC, "c"): 500, (DA, "a"): 600}
    st.history.append({"block": 700, "event": "context_changed"})
    assert EC.dethroned_blocks(st) == {}


def test_malformed_dethroning_block_holds_no_slot(tmp_path, co):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC))))
    comp.states["s1-fast"].history[-1]["block"] = "9328728"  # string: the champion's crowning cannot date the slot
    comp.states["s1-fast"].champion.crowned_block = 1800
    w = _w(comp, subs)
    assert 2 not in w and 3 not in w and w[5] == pytest.approx(ENG * _f(comp, 200), rel=1e-12)


# ------------------------------------------------------------------ engine keep-alive
def test_engine_keepalive_comes_out_of_all_paid_engine_shares(tmp_path, co, monkeypatch):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB), ("x", DC))))
    monkeypatch.setattr(EV, "waiting_uids", lambda *a, **k: [12])
    comp.engine_keepalive = True
    base = {i: 1.0 for i in range(20, 30)}
    w = _w(comp, subs, base=base)
    f = _f(comp, 200)
    ref = CS.combine_weights(base, 5, (1 - R2) * f, {}, engine_cochampions={2: R2 * f / 2, 3: R2 * f / 2},
                             engine_waiting=[12])
    assert w[12] > 0 and all(w[u] == pytest.approx(ref[u], rel=1e-12) for u in (2, 3, 5, 12))
    assert w[2] + w[3] + w[5] + w[12] == pytest.approx(ENG * f, rel=1e-9)  # the keep-alive is inside the engine share


def test_unpaid_former_champion_in_its_slot_gets_the_keepalive(tmp_path, co, monkeypatch):
    comp, subs = _live(tmp_path, Gate())  # nobody approved: without the keep-alive the set earns 0 and is pruned
    monkeypatch.setattr(EV, "waiting_uids", lambda *a, **k: [])
    comp.engine_keepalive = True
    base = {i: 1.0 for i in range(20, 30)}
    w = _w(comp, subs, base=base)
    each = min(CS.ENGINE_KA_FRACTION * 0.1, CS.ENGINE_KA_MAX_SHARE * ENG / 3)  # champion 5 (unapproved) + the set
    assert all(w[u] == pytest.approx(each, rel=1e-9) for u in (5, 2, 3))
    w = _w(comp, subs, block=1800 + SLOT, base=base)  # slot over: only the unapproved champion keeps it
    assert 2 not in w and 3 not in w and w[5] == pytest.approx(min(CS.ENGINE_KA_FRACTION * 0.1, CS.ENGINE_KA_MAX_SHARE * ENG))


def test_protected_uids_cover_both_slots_in_rank_order_each_from_its_own_dethroning(tmp_path, co, monkeypatch):
    comp = _comp(tmp_path, Gate(), SUBS)
    st = comp.states["s1-fast"]
    st.results[DE] = _res(DE, 1.2, 50)
    st.first_seen[DE] = (50, "z")
    st.history.insert(0, {"block": 50, "event": "crowned", "digest": DE, "hotkey": "z", "score": 1.2})
    subs = _crown(comp, SUBS + [_sub(4, "z", DE, 50)], 5, "x", DC, 1800)
    # champion 5 (unapproved), the set (rank 2, listed order uid 3, uid 2), uid 4 (rank 3, dethroned at block 100)
    assert EC.protected_uids(st, subs, comp.tracks["s1-fast"], 2000, False) == [5, 3, 2, 4]
    assert EC.protected_uids(st, subs, comp.tracks["s1-fast"], 2000, True) == [3, 2, 4]
    assert EC.protected_uids(st, subs, comp.tracks["s1-fast"], 100 + SLOT, True) == [3, 2]  # uid 4's own slot ended
    assert EC.protected_uids(st, subs, comp.tracks["s1-fast"], 1800 + SLOT, True) == []
    assert CS.ENGINE_KA_PROTECTED_MAX == 4
    monkeypatch.setattr(CS, "ENGINE_KA_PROTECTED_MAX", 3)
    assert EC.protected_uids(st, subs, comp.tracks["s1-fast"], 2000, False) == [5, 3, 2]


def test_paid_former_champion_gets_no_keepalive_on_top(tmp_path, co, monkeypatch):
    comp, subs = _live(tmp_path, Gate(_ok(("old", DA), ("new", DB))))
    monkeypatch.setattr(EV, "waiting_uids", lambda *a, **k: [])
    comp.engine_keepalive = True
    base = {i: 1.0 for i in range(20, 30)}
    w = _w(comp, subs, base=base)
    # 2 and 3 are paid (restored set): no keep-alive; the unapproved champion 5 gets one, out of their pay
    ref = CS.combine_weights(base, None, 0.0, {}, engine_cochampions={2: _f(comp, 500) / 2, 3: _f(comp, 500) / 2},
                             engine_waiting=[5, 2, 3])
    assert w == pytest.approx(ref, rel=1e-12) and w[5] > 0 and w[2] == pytest.approx(w[3])
    assert w[2] + w[3] + w[5] == pytest.approx(ENG * _f(comp, 500), rel=1e-9)


# ------------------------------------------------------------------ the live state (real constants, no monkeypatch)
D56, H56 = "23dcdecc1519edf813699ab48adadd1df07a269016da3816bee37cba84e8a89d", "5HC3sybJYnAUcEcWkKZnaT6f4fNXFQ5Q42EcZXpEKb8UN7Wr"
(D198, H198), (D4, H4) = CS.ENGINE_CO_CHAMPIONS["s1-fast"]
LIVE_BLOCK = 9330448  # ~08:00Z 21 Oct 2026


def _live_comp(tmp_path, ok):
    comp = EV.EngineCompetition(object(), {"s1-fast": REF}, str(tmp_path / "s.json"), start_utc=None,
                                award_gate=Gate(_ok(*ok)))
    comp.states["s1-fast"] = TrackState(
        results={D56: _res(D56, 3.0066, 9328728), D198: _res(D198, 2.8976, 9328728), D4: _res(D4, 2.8233, 9326968)},
        first_seen={D56: (9328000, H56), D198: (9326382, H198), D4: (9326382, H4)},
        champion=Champion(D56, H56, 56, 3.0066, 9328728),
        history=[{"block": 0, "event": "context_changed"},
                 {"block": 9326968, "event": "crowned", "digest": D198, "hotkey": H198, "score": 2.8136},
                 {"block": 9328728, "event": "crowned", "digest": D56, "hotkey": H56, "score": 3.0066}])
    return comp


LIVE_SUBS = [_sub(56, H56, D56, 9328000), _sub(198, H198, D198, 9326382), _sub(4, H4, D4, 9326382)]


@pytest.mark.parametrize("ok, want", [
    ((), {56: 0.0, 198: 0.0, 4: 0.0}),
    (((H4, D4),), {56: 0.0, 198: 0.0, 4: 0.491369}),
    (((H4, D4), (H198, D198)), {56: 0.0, 198: 0.491369, 4: 0.491369}),
    (((H4, D4), (H198, D198), (H56, D56)), {56: 0.822918, 198: 0.084275, 4: 0.084275}),
    (((H56, D56),), {56: 0.822918, 198: 0.0, 4: 0.0}),
])
def test_live_situation_numbers(tmp_path, ok, want):
    """Fractions of the engine share at block 9330448 (RESULT.md table): co set crowned 9326968 (age 3480 blocks,
    champion_fraction 0.982738), uid 56 crowned 9328728 (age 1720, 0.991468), the set's slot runs to block 9379128."""
    comp = _live_comp(tmp_path, ok)
    base = {u: 1.0 for u in range(100, 140)}
    w = comp.weights(base, _snap(LIVE_SUBS, block=LIVE_BLOCK), set())
    got = {u: w.get(u, 0.0) / ENG for u in (56, 198, 4)}
    assert got == pytest.approx(want, abs=1e-6)
    assert CS.ENGINE_FORMER_SLOT_BLOCKS + 9328728 == 9379128


def test_live_situation_no_jump_at_the_slot_end(tmp_path):
    """Case (c) at the co set's last slot block 9379127 and at 9379128: 0.5 x its decayed pot on both (lead call on doubt
    2); case (d): the 17 % slot ends at 9379128 and uid 56 gets its whole pot."""
    base = {u: 1.0 for u in range(100, 140)}
    for block, d in ((9379127, (0.622504, 0.06375)), (9379128, (0.75, 0.0))):
        comp = _live_comp(tmp_path, ((H4, D4), (H198, D198)))
        w = comp.weights(base, _snap(LIVE_SUBS, block=block), set())
        want = _f(comp, block - 9326968) / 2
        assert w[4] / ENG == pytest.approx(want, abs=1e-6) and w[198] / ENG == pytest.approx(want, abs=1e-6)
        assert want == pytest.approx({9379127: 0.370637, 9379128: 0.370635}[block], abs=1e-6)
        comp = _live_comp(tmp_path, ((H4, D4), (H198, D198), (H56, D56)))
        w = comp.weights(base, _snap(LIVE_SUBS, block=block), set())
        assert w[56] / ENG == pytest.approx(d[0], abs=1e-6) and w.get(4, 0.0) / ENG == pytest.approx(d[1], abs=1e-6)
