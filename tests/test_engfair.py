"""Engine opening fairness (pre-opening commitments count at the opening block, hash tie-break), the previous-champion
fallback when the new champion is not approved, and the engine keep-alive (REAX_ENGINE_KEEPALIVE, default off)."""

import hashlib
import math

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet import engvalidator as EV
from reax_subnet.award import engine_submission_id
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.engcomp import Champion, EvalResult, Submission, TrackState
from test_engcomp import P, ref_of

DA, DB, DC = (hashlib.sha256(x.encode()).hexdigest() for x in "abc")
REF = ref_of("reference")
START, BH = 1000, "ab" * 32


def _sub(uid, hotkey, digest=DA, block=100):
    return Submission(uid, hotkey, "s1-fast", "ghcr.io/a/e", digest, block)


def _snap(subs, block=2000, coldkeys=None):
    return Snapshot(block, tuple(Neuron(s.uid, s.hotkey, (coldkeys or {}).get(s.uid, f"cold{s.uid}"), False, 0, None,
                                        None, "reax-eng/1:s1-fast:" + s.ref, s.block) for s in subs), None)


def _res(digest, score=1.5, block=100):
    return EvalResult(digest, "s1-fast", True, score=score, block=block, session=f"s{block}")


class Gate:
    def __init__(self, ok=()):
        self.ok = set(ok)

    def approved(self, track, hotkey, sid):
        return (hotkey, sid) in self.ok


def _ok(*pairs):
    return {(hk, engine_submission_id(d)) for hk, d in pairs}


@pytest.fixture(autouse=True)
def _reset_start_hash():
    EC.set_start_block_hash(None)
    yield
    EC.set_start_block_hash(None)


@pytest.fixture
def opening(monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_TRACK_START_BLOCK", START)
    EC.set_start_block_hash(BH)


# ------------------------------------------------------------------------------------------------ A: opening fairness
@pytest.mark.real_opening
def test_opening_block_is_the_predicted_finney_block():
    assert CS.ENGINE_TRACK_START_BLOCK == 9326382 and CS.ENGINE_PRE_OPEN_AS_START


def test_inactive_without_an_opening_block(monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_TRACK_START_BLOCK", None)
    assert EC.effective_block(5) == 5 and EC.tie_key(5, "hk", DA) == ""


def test_pre_opening_commitments_count_at_the_opening_block(opening):
    assert EC.effective_block(10) == START and EC.effective_block(START) == START
    assert EC.effective_block(START + 7) == START + 7 and EC.effective_block(0) == 0


def test_flag_off_keeps_raw_blocks(opening, monkeypatch):
    monkeypatch.setattr(CS, "ENGINE_PRE_OPEN_AS_START", False)
    assert EC.effective_block(10) == 10 and EC.tie_key(10, "hk", DA) == ""


def test_tie_key_is_pinned_and_only_for_the_opening_group(opening):
    want = hashlib.sha256(f"reax-eng-tie/1|{BH}|hkA|{DA}".encode()).hexdigest()
    assert EC.tie_key(5, "hkA", DA) == want and EC.tie_key(START, "hkA", DA) == want
    assert EC.tie_key(START + 1, "hkA", DA) == "" and EC.tie_key(0, "hkA", DA) == ""


def test_early_commitment_buys_no_order_advantage(opening):
    early, late = _sub(7, "hk7", DA, block=5), _sub(3, "hk3", DB, block=900)
    after = _sub(1, "hk1", DC, block=START + 5)
    st = TrackState()
    todo = EC.pending(st, EC.observe(st, [after, late, early], 2000), 10)
    keys = {s.uid: EC.tie_key(s.block, s.hotkey, s.digest) for s in (early, late)}
    first_two = sorted((early, late), key=lambda s: keys[s.uid])
    assert todo[:2] == first_two and todo[2] == after  # hash order inside the group, then the later commitment
    assert st.first_seen[DA][0] == START and st.first_seen[DB][0] == START
    # the order does not depend on how the validator happened to list them
    st2 = TrackState()
    assert EC.pending(st2, EC.observe(st2, [early, after, late], 2000), 10) == todo


def test_without_a_block_hash_the_uid_decides_inside_the_group(opening, monkeypatch):
    EC.set_start_block_hash(None)
    early, late = _sub(7, "hk7", DA, block=5), _sub(3, "hk3", DB, block=900)
    st = TrackState()
    assert EC.pending(st, EC.observe(st, [early, late], 2000), 10) == [late, early]


def test_identical_digest_goes_to_the_hotkey_with_the_lower_tie_hash(opening):
    a, b = _sub(7, "hkA", DA, block=5), _sub(3, "hkB", DA, block=900)
    st = TrackState()
    eligible = EC.observe(st, [a, b], 2000)
    winner = min((a, b), key=lambda s: EC.tie_key(s.block, s.hotkey, s.digest))
    assert eligible == [winner] and st.first_seen[DA] == (START, winner.hotkey)


def test_post_opening_commitment_does_not_beat_the_opening_group(opening):
    early, after = _sub(7, "hk7", DA, block=5), _sub(3, "hk3", DA, block=START + 1)
    st = TrackState()
    assert EC.observe(st, [after, early], 2000) == [early]


def test_champion_tie_on_score_uses_the_same_order(opening):
    early, late = _sub(7, "hk7", DA, block=5), _sub(3, "hk3", DB, block=900)
    st = TrackState(results={DA: _res(DA, 1.6), DB: _res(DB, 1.6)})
    elig = EC.observe(st, [early, late], 2000)
    p1 = EC.TrackParams("s1-fast", pool="s1-fast", temperature=2.07, confirmations=1)
    EC.update_champion(st, elig, p1, 2000)
    want = min((early, late), key=lambda s: EC.tie_key(s.block, s.hotkey, s.digest))
    assert st.champion is not None and st.champion.digest == want.digest


# ------------------------------------------------------------------------------------------------ C: previous champion
def _comp(tmp_path, gate, subs, monkeypatch=None):
    comp = EV.EngineCompetition(object(), {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": P},
                                start_utc=None, award_gate=gate)
    st = comp.states["s1-fast"]
    st.results = {DA: _res(DA, 1.5), DB: _res(DB, 1.8, block=1500)}
    st.first_seen = {DA: (100, "old"), DB: (1500, "new")}
    st.champion = Champion(DB, "new", 3, 1.8, 1500)
    st.history = [{"block": 100, "event": "crowned", "digest": DA, "hotkey": "old", "score": 1.5},
                  {"block": 1500, "event": "crowned", "digest": DB, "hotkey": "new", "score": 1.8}]
    return comp


SUBS = [_sub(2, "old", DA, 100), _sub(3, "new", DB, 1500)]
BASE = {9: 1.0, 8: 1.0}


def test_former_champions_from_history():
    st = TrackState(champion=Champion(DB, "new", 3, 1.8, 1500), history=[
        {"block": 100, "event": "crowned", "digest": DA, "hotkey": "old"},
        {"block": 700, "event": "crowned", "digest": DC, "hotkey": "mid"},
        {"block": 800, "event": "vacated", "digest": DC, "hotkey": "mid"},
        {"block": 1500, "event": "crowned", "digest": DB, "hotkey": "new"}])
    assert [f["hotkey"] for f in EC.former_champions(st)] == ["old"]
    st.history.append({"block": 1600, "event": "context_changed"})
    assert EC.former_champions(st) == []


def test_unapproved_new_champion_restores_the_previous_approved_one(tmp_path):
    comp = _comp(tmp_path, Gate(_ok(("old", DA))), SUBS)
    w = comp.weights(BASE, _snap(SUBS), set())
    assert 3 not in w and w[2] > 0.15  # about the engine share (0.20), decayed a little from block 100
    age_frac = EC.champion_fraction(comp.reward, 2000 - 100)
    # rc10.post1: "old" holds the rank-2 slot (dethroned at 1500); it takes rank 1's whole share (its own slot is not
    # held against it), the same as after its slot ended
    assert w[2] == pytest.approx(CS.profile_for().engine * age_frac, rel=1e-9)
    comp.states["s1-fast"].history[1]["block"] = 2000 - CS.ENGINE_FORMER_SLOT_BLOCKS  # slot over: the same share
    w = comp.weights(BASE, _snap(SUBS), set())
    assert w[2] == pytest.approx(CS.profile_for().engine * age_frac, rel=1e-9)


def test_approved_new_champion_keeps_the_share(tmp_path):
    comp = _comp(tmp_path, Gate(_ok(("old", DA), ("new", DB))), SUBS)
    w = comp.weights(BASE, _snap(SUBS), set())
    assert w[3] > 0.15 and w[2] == pytest.approx(w[3] * CS.ENGINE_RANK_SPLIT[1] / (1 - CS.ENGINE_RANK_SPLIT[1]))
    comp.award_gate = Gate(_ok(("new", DB)))  # rc10.post1: the former champion's slot needs its own approval
    w = comp.weights(BASE, _snap(SUBS), set())
    assert w[3] > 0.15 and 2 not in w


def test_nobody_approved_the_share_goes_to_inference(tmp_path):
    comp = _comp(tmp_path, Gate(), SUBS)
    assert comp.weights(BASE, _snap(SUBS), set()) == BASE


def test_previous_withdrawn_or_invalid_is_not_restored(tmp_path):
    comp = _comp(tmp_path, Gate(_ok(("old", DA))), SUBS)
    assert comp.weights(BASE, _snap(SUBS[1:]), set()) == BASE          # no longer committed
    comp.states["s1-fast"].results[DA] = EvalResult(DA, "s1-fast", False, score=0.0, block=100)
    assert comp.weights(BASE, _snap(SUBS), set()) == BASE              # last result invalid


def test_restore_flag_off_is_the_old_behaviour(tmp_path, monkeypatch):
    monkeypatch.setattr(EV, "ENGINE_RESTORE_PREVIOUS", False)
    comp = _comp(tmp_path, Gate(_ok(("old", DA))), SUBS)
    comp.states["s1-fast"].history[1]["block"] = 2000 - CS.ENGINE_FORMER_SLOT_BLOCKS  # rc10.post1: no former slot
    assert comp.weights(BASE, _snap(SUBS), set()) == BASE
    comp.states["s1-fast"].history[1]["block"] = 1500  # in its slot: only the slot (17 % of rank 1's decayed part)
    w = comp.weights(BASE, _snap(SUBS), set())
    assert 3 not in w and w[2] == pytest.approx(CS.profile_for().engine * CS.ENGINE_RANK_SPLIT[1]
                                                * EC.champion_fraction(comp.reward, 500), rel=1e-9)


# ------------------------------------------------------------------------------------------------ B: engine keep-alive
def _waiting_state(block=1990):
    return TrackState(first_seen={DA: (block, "w1"), DB: (block, "w2")}, owners={DA: "w1", DB: "w2"})


def test_waiting_uids_guards():
    w1, w2 = _sub(4, "w1", DA, 1990), _sub(5, "w2", DB, 1991)
    st = _waiting_state()
    assert EC.waiting_uids(st, [w1, w2], 2000, {4: "ck1", 5: "ck2"}) == [4, 5]
    assert EC.waiting_uids(st, [w1, w2], 2000, {4: "ck", 5: "ck"}) == [4]               # one per coldkey
    st.results[DA] = _res(DA)
    assert EC.waiting_uids(st, [w1, w2], 2000, {}) == [5]                               # measured: no longer waiting
    st.owners.pop(DB)
    assert EC.waiting_uids(st, [w1, w2], 2000, {}) == [5]                               # not pulled yet: first committer (reg-sniping 20 Oct)
    st.owners[DB] = "someone-else"
    assert EC.waiting_uids(st, [w1, w2], 2000, {}) == []                                # image bound to another hotkey
    st2 = _waiting_state(block=2000 - int(CS.ENGINE_KA_MAX_HOURS * 300) - 1)
    assert EC.waiting_uids(st2, [w1], 2000, {}) == []                                   # window over
    many = [_sub(10 + i, f"h{i}", hashlib.sha256(bytes([i])).hexdigest(), 1990) for i in range(9)]
    st3 = TrackState(first_seen={s.digest: (1990, s.hotkey) for s in many}, owners={s.digest: s.hotkey for s in many})
    assert len(EC.waiting_uids(st3, many, 2000, {})) == CS.ENGINE_KA_MAX_ENTRIES


def test_combine_weights_pays_waiting_entries_from_the_engine_share():
    serving = {u: 1.0 for u in range(10, 20)}
    plain = CS.combine_weights(serving, 1, 1.0, {})
    ka = CS.combine_weights(serving, 1, 1.0, {}, engine_waiting=[2, 3])
    assert CS.combine_weights(serving, 1, 1.0, {}, engine_waiting=[]) == plain
    assert ka[2] == ka[3] > 0 and ka[1] < plain[1]
    assert ka[2] + ka[3] <= CS.ENGINE_KA_MAX_SHARE * CS.profile_for().engine * 1.01   # capped at 5 % of the engine share
    assert math.isclose(plain[1] - ka[1], ka[2] + ka[3], rel_tol=0.05)                 # the champion pays for it
    assert math.isclose(sum(ka.values()), 1.0)


def test_combine_weights_waiting_without_champion_and_owner_excluded():
    serving = {u: 1.0 for u in range(10, 20)}
    ka = CS.combine_weights(serving, None, 0.0, {}, engine_waiting=[2, 99], owner_uid=99)
    assert 2 in ka and 99 not in ka
    # a uid that already holds the champion share is left out
    assert CS.combine_weights(serving, 1, 1.0, {}, engine_waiting=[1]) == CS.combine_weights(serving, 1, 1.0, {})


def test_keepalive_flag_parse():
    assert CS.engine_keepalive_enabled({}) is False and CS.engine_keepalive_enabled({"REAX_ENGINE_KEEPALIVE": "1"})
    with pytest.raises(ValueError):
        CS.engine_keepalive_enabled({"REAX_ENGINE_KEEPALIVE": "maybe"})


@pytest.mark.parametrize("flag", ["0", "1"])
def test_validator_pays_waiting_entry_only_with_the_flag(tmp_path, monkeypatch, flag):
    monkeypatch.setenv("REAX_ENGINE_KEEPALIVE", flag)
    comp = EV.EngineCompetition(object(), {"s1-fast": REF}, str(tmp_path / "s.json"), tracks={"s1-fast": P},
                                start_utc=None)
    st = comp.states["s1-fast"]
    w1 = _sub(4, "w1", DA, 1990)
    st.first_seen[DA], st.owners[DA] = (1990, "w1"), "w1"
    base = {u: 1.0 for u in range(10, 20)}
    w = comp.weights(base, _snap([w1]), set())
    if flag == "1":
        assert w[4] > 0 and math.isclose(sum(w.values()), 1.0)
    else:
        assert 4 not in w


# ------------------------------------------------------------------------------------------------ opening hash lookup
def test_hash_unknown_means_uid_order_known_means_pinned_hash_order(opening):
    EC.set_start_block_hash(None)
    a, b = _sub(1, "hkA", DA, block=5), _sub(2, "hkB", DA, block=6)
    assert EC.tie_key(5, "hkA", DA) == "" and EC.commit_order(a) < EC.commit_order(b)
    EC.set_start_block_hash("0x" + BH.upper())
    assert EC.start_block_hash() == BH
    assert EC.tie_key(5, "hkA", DA) == hashlib.sha256(f"reax-eng-tie/1|{BH}|hkA|{DA}".encode()).hexdigest()


class _Info:
    def __init__(self, h):
        self.hash = h


def _sdk_chain(fn):
    from reax_subnet.chain import SdkChain

    c = SdkChain.__new__(SdkChain)
    c._block_hashes = {}
    c.subtensor = type("S", (), {"block_info": staticmethod(fn)})()
    return c


def test_sdk_block_hash_none_when_block_not_produced_and_on_errors():
    assert _sdk_chain(lambda n: None).block_hash(10) is None

    def boom(n):
        raise RuntimeError("rpc down")

    assert _sdk_chain(boom).block_hash(10) is None
    assert _sdk_chain(lambda n: _Info(None)).block_hash(10) is None


def test_sdk_block_hash_is_cached():
    calls = []

    def info(n):
        calls.append(n)
        return _Info("0x" + BH)

    c = _sdk_chain(info)
    assert c.block_hash(10) == "0x" + BH and c.block_hash(10) == "0x" + BH and calls == [10]


def _validator(chain):
    from reax_subnet.validator import Validator

    v = Validator.__new__(Validator)
    v.chain = chain
    return v


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_validator_fills_the_hash_once_the_block_exists(opening):
    EC.set_start_block_hash(None)
    seen = []

    class C:
        def block_hash(self, n):
            seen.append(n)
            return None if len(seen) == 1 else "0x" + BH

    v, snap = _validator(C()), Snapshot(START + EC.START_HASH_CONFIRM_BLOCKS, (), None)
    _run(v.sync_engine_start_hash(Snapshot(START - 1, (), None)))
    _run(v.sync_engine_start_hash(Snapshot(START + EC.START_HASH_CONFIRM_BLOCKS - 1, (), None)))
    assert seen == [] and EC.start_block_hash() is None  # block not reached / not yet confirmed: no query
    _run(v.sync_engine_start_hash(snap))
    assert seen == [START] and EC.start_block_hash() is None  # not produced yet / unreadable
    _run(v.sync_engine_start_hash(snap))
    assert EC.start_block_hash() == BH
    _run(v.sync_engine_start_hash(snap))
    assert seen == [START, START]  # known: no more queries


def test_validator_survives_lookup_error_and_chains_without_block_hash(opening):
    EC.set_start_block_hash(None)

    class Bad:
        def block_hash(self, n):
            raise RuntimeError("down")

    _run(_validator(Bad()).sync_engine_start_hash(Snapshot(START + EC.START_HASH_CONFIRM_BLOCKS, (), None)))
    _run(_validator(object()).sync_engine_start_hash(Snapshot(START + EC.START_HASH_CONFIRM_BLOCKS, (), None)))
    assert EC.start_block_hash() is None


def test_worker_uses_the_same_hash_sync(opening):
    """Remote mode: the keyless competition worker computes first_seen and the champion, so it must read the hash too."""
    import inspect

    from reax_subnet import worker

    assert "sync_start_block_hash(chain, snap.block" in inspect.getsource(worker.run_worker)

    class C:
        def block_hash(self, n):
            return "0x" + BH

    EC.set_start_block_hash(None)
    _run(EC.sync_start_block_hash(C(), START + EC.START_HASH_CONFIRM_BLOCKS, 5.0))
    assert EC.start_block_hash() == BH


def test_tie_order_follows_the_first_sighting_not_a_later_rewrite(opening):
    """Critic round 3 NEW-4: a pre-opening entrant whose commitment is rewritten after the opening (e.g. the inference
    element of a dual-role hotkey) keeps its opening tie hash: the chain block of the rewrite does not change the order."""
    early, late = _sub(7, "hk7", DA, block=5), _sub(3, "hk3", DB, block=900)
    st = TrackState(results={DA: _res(DA, 1.6), DB: _res(DB, 1.6)})
    EC.observe(st, [early, late], 2000)
    p1 = EC.TrackParams("s1-fast", pool="s1-fast", temperature=2.07, confirmations=1)
    want = min((early, late), key=lambda s: EC.tie_key(s.block, s.hotkey, s.digest))
    rewritten = [_sub(s.uid, s.hotkey, s.digest, block=2500 if s is want else s.block) for s in (early, late)]
    elig = EC.observe(st, rewritten, 2600)
    EC.update_champion(st, elig, p1, 2600)
    assert st.champion is not None and st.champion.digest == want.digest
    assert [s.uid for s in EC.pending(TrackState(first_seen=dict(st.first_seen)), rewritten, 5)] == \
        [s.uid for s in sorted(rewritten, key=lambda s: EC.tie_key(st.first_seen[s.digest][0], s.hotkey, s.digest))]
