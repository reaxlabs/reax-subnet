"""Reg-sniping review (20 Oct 2026): pruning an entrant must not let a copy of its public work win, and an undecided
entry keeps its provisional keep-alive until its verdict."""
import pytest
from spec_helpers import commit, make_weights, neuron, snapshot, finetune
from test_carryover import H, pruned
from test_provisional_ka import BLOCK, NOW, _p, _snapr, _uids
from test_spec_evaluator import TRACK, Env

from reax_subnet.spec import evaluator as E
from reax_subnet.spec.evaluator import compute_provisional
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


def test_immunity_matches_the_chain():
    assert E.IMMUNITY_BLOCKS_ASSUMED == 7200


# ------------------------------------------------------------------ near copies of a pruned entrant
async def test_near_copy_of_a_pruned_entrant_by_another_coldkey_is_refused(tmp_path):
    env, old = await pruned(tmp_path)
    env.add(3, "m/copy", 0.99, weights=finetune(make_weights(7), 3, scale=0.001))
    await env.ev.evaluate_cycle(snapshot(300, [neuron(3, commit(TRACK, "m/copy"), coldkey="mallory")]))
    m = env.sub(3)
    assert m.derived_from == old.key and m.derived_how == "weights"
    assert m.status == ST.REJECTED and "withdrawn commitment" in m.reason, m.reason
    assert old.status == ST.WITHDRAWN and old.attempts == 0


async def test_own_near_copy_of_the_own_pruned_entry_is_not_a_copy(tmp_path):
    env, old = await pruned(tmp_path)
    env.add(2, "a/new", 0.99, weights=finetune(make_weights(7), 2, scale=0.001))
    await env.ev.evaluate_cycle(snapshot(300, [neuron(2, commit(TRACK, "a/new"), coldkey=old.coldkey)]))
    assert env.sub(2).derived_from != old.key


async def test_far_entry_is_not_linked_to_a_pruned_entrant(tmp_path):
    env, old = await pruned(tmp_path)
    env.add(3, "m/own", 0.99, weights=make_weights(11))
    await env.ev.evaluate_cycle(snapshot(300, [neuron(3, commit(TRACK, "m/own"), coldkey="mallory")]))
    assert env.sub(3).derived_from == ""


# ------------------------------------------------------------------ exact copy crowned before the earlier original
async def test_crowned_exact_copy_loses_to_the_earlier_original_of_another_coldkey(tmp_path):
    env = Env(tmp_path, {})
    w = make_weights(7)
    env.add(1, "a/orig", 0.9, weights=w)
    env.add(3, "m/copy", 0.9, weights=w)
    env.hub.unavailable.add("a/orig")            # the original waits (e.g. queued behind the copy)
    a_n = neuron(1, commit(TRACK, "a/orig"), coldkey="alice")
    m_n = neuron(3, commit(TRACK, "m/copy"), coldkey="mallory")
    await env.ev.evaluate_cycle(snapshot(100, [a_n]))
    await env.ev.evaluate_cycle(snapshot(150, [a_n, m_n]))
    env.clock.advance(13 * H)                    # confirmation delay
    await env.ev.evaluate_cycle(snapshot(160, [a_n, m_n]))
    a, m = env.sub(1), env.sub(3)
    ts = env.ev.state.track(TRACK)
    assert m.status == ST.CHAMPION and ts.champion == m.key, (m.status, m.reason)
    assert a.first_seen_block < m.first_seen_block
    env.hub.unavailable.clear()
    env.clock.advance(2 * H)
    await env.ev.evaluate_cycle(snapshot(170, [a_n, m_n]))
    assert m.status == ST.REJECTED and "identical tensors" in m.reason and ts.champion != m.key
    assert a.status != ST.REJECTED, a.reason


async def test_crowned_exact_copy_still_wins_over_a_later_commitment(tmp_path):
    env = Env(tmp_path, {})
    w = make_weights(7)
    env.add(1, "a/first", 0.9, weights=w)
    a_n = neuron(1, commit(TRACK, "a/first"), coldkey="alice")
    for b in (100, 110):
        await env.ev.evaluate_cycle(snapshot(b, [a_n]))
        env.clock.advance(13 * H)
    assert env.sub(1).status == ST.CHAMPION
    env.add(3, "m/later", 0.9, weights=w)
    await env.ev.evaluate_cycle(snapshot(200, [a_n, neuron(3, commit(TRACK, "m/later"), coldkey="mallory")]))
    assert env.sub(3).status == ST.REJECTED and env.sub(1).status == ST.CHAMPION


# ------------------------------------------------------------------ provisional keep-alive until the verdict
def test_close_call_waiting_for_its_extra_draw_keeps_the_provisional_slot():
    st = ST.SpecState()
    s = _p(st, 1)
    s.score, s.delta_checked, s.close_draws = 61.0, True, ["d1"]
    assert _uids(st, _snapr([s])) == [1]
    s.close_draws = []                     # scored and not waiting for an extra draw: the normal keep-alive decides
    assert _uids(st, _snapr([s])) == []
    s.close_draws, s.status = ["d1"], ST.NOT_BETTER
    assert _uids(st, _snapr([s])) == []
    s.status, s.delta_checked = ST.QUEUED, False
    assert _uids(st, _snapr([s])) == []


def test_close_call_still_needs_the_window_and_the_immunity_condition():
    st = ST.SpecState()
    s = _p(st, 1, since=0.0)               # 12-hour window long over
    s.score, s.delta_checked, s.close_draws = 61.0, True, ["d1"]
    assert _uids(st, _snapr([s])) == []
    s2 = _p(st, 2)
    s2.score, s2.delta_checked, s2.close_draws = 61.0, True, ["d1"]
    assert _uids(st, _snapr([s2], regs={2: BLOCK - 10})) == []   # freshly registered: immune


def test_generalist_keeps_the_reg_sniping_rules_and_stays_closed_until_calibrated(monkeypatch):
    """The provisional keep-alive loop and its close-call rule are track-generic (rc10h merge); the generalist guard in
    ``_provisional_ok`` (finite entry floor and frozen G_med) still comes first, so nothing is paid before calibration."""
    from generalist_defaults import defaults
    from reax_subnet import competition_split as cs
    monkeypatch.setitem(cs.TRACK_ENTRY_FLOOR, "generalist", defaults()["GENERALIST_ENTRY_FLOOR"])  # uncalibrated: the release
    monkeypatch.setattr(cs, "GENERALIST_G_MED", defaults()["GENERALIST_G_MED"])                     # sets both constants
    st = ST.SpecState()
    s = _p(st, 1, track="generalist")
    s.score, s.delta_checked, s.close_draws = 61.0, True, ["d1"]
    snap = _snapr([s])
    assert compute_provisional(st, ["generalist"], snap, now=NOW)["generalist"] == []      # fail-closed defaults
    monkeypatch.setitem(cs.TRACK_ENTRY_FLOOR, "generalist", 50.0)
    monkeypatch.setattr(cs, "GENERALIST_G_MED", 1.0)
    assert compute_provisional(st, ["generalist"], snap, now=NOW)["generalist"] == [1]    # calibrated: close call keeps the slot
    s.close_draws = []
    assert compute_provisional(st, ["generalist"], snap, now=NOW)["generalist"] == []


# ------------------------------------------------------------------ engine keep-alive: every waiting state (Florian 18:55Z)
from test_engfair import DA, DB, DC, _res, _sub   # noqa: E402

from reax_subnet import competition_split as CS   # noqa: E402
from reax_subnet import engcomp as EC   # noqa: E402
from reax_subnet.engcomp import Champion, TrackState   # noqa: E402

B = 20000


def test_engine_entry_not_pulled_yet_is_covered_for_its_first_committer_only():
    a, copy = _sub(4, "w1", DA, B - 10), _sub(5, "w2", DA, B - 5)
    st = TrackState(first_seen={DA: (B - 10, "w1")})
    assert EC.waiting_uids(st, [a, copy], B, {4: "c1", 5: "c2"}) == [4]
    st.not_owner.add(f"{DA}:w2")
    st.owners[DA] = "w9"                                   # bound to another hotkey after the pull
    assert EC.waiting_uids(st, [a], B, {4: "c1"}) == []


def test_engine_pending_challenger_keeps_the_slot_until_its_confirmation():
    ch, other = _sub(4, "w1", DA, B - 100), _sub(5, "w2", DB, B - 100)
    st = TrackState(first_seen={DA: (B - 100, "w1"), DB: (B - 100, "w2")}, owners={DA: "w1", DB: "w2"})
    st.results[DA], st.results[DB] = _res(DA, block=B - 50), _res(DB, block=B - 50)
    assert EC.waiting_uids(st, [ch, other], B, {}) == []               # measured, nothing pending
    st.challenger = {"digest": DA, "hotkey": "w1", "measurements": [[B - 50, 1.5, "s1"]]}
    assert EC.waiting_uids(st, [ch, other], B, {}) == [4]
    st.champion = Champion(DA, "w1", 4, 1.5, B)                        # crowned: the champion share pays it
    st.challenger = None
    assert EC.waiting_uids(st, [ch, other], B, {}) == []


def test_engine_window_starts_at_the_end_of_the_immunity():
    w = _sub(4, "w1", DA, B - 10)
    st = TrackState(first_seen={DA: (B - 10, "w1")}, owners={DA: "w1"})
    lead = CS.CHAIN_IMMUNITY_BLOCKS - CS.PROVISIONAL_KA_LEAD_BLOCKS
    assert EC.waiting_uids(st, [w], B, {}, {4: B - 20}) == []          # freshly registered: immune, nothing yet
    assert EC.waiting_uids(st, [w], B, {}, {4: B - lead}) == [4]       # immunity ends within the lead
    max_blocks = int(CS.ENGINE_KA_MAX_HOURS * 300)
    late = B + max_blocks - 1                                           # 12 h after the commitment, but the window
    assert EC.waiting_uids(st, [w], late, {}, {4: late - lead - 100}) == [4]   # started at the immunity end
    assert EC.waiting_uids(st, [w], late, {}, {4: 1}) == []             # long registered: 12 h from the commitment
    assert EC.waiting_uids(st, [w], late, {}) == []                     # registration unknown: old rule


async def test_a_tie_in_first_sighting_never_dethrones_a_champion(tmp_path):
    # critic #2: same sync (equal first_seen_block), lower uid: the uid tie-break must not reject a crowned entry
    env = Env(tmp_path, {})
    w = make_weights(7)
    env.add(5, "a/champ", 0.9, weights=w)
    env.add(1, "m/same-sync", 0.9, weights=w)
    env.hub.unavailable.add("m/same-sync")
    a_n = neuron(5, commit(TRACK, "a/champ"), coldkey="alice")
    m_n = neuron(1, commit(TRACK, "m/same-sync"), coldkey="mallory")
    await env.ev.evaluate_cycle(snapshot(100, [a_n, m_n]))
    env.clock.advance(13 * H)
    await env.ev.evaluate_cycle(snapshot(110, [a_n, m_n]))
    a, m = env.sub(5), env.sub(1)
    assert a.status == ST.CHAMPION and a.first_seen_block == m.first_seen_block, (a.status, a.reason)
    env.hub.unavailable.clear()
    env.clock.advance(2 * H)
    await env.ev.evaluate_cycle(snapshot(120, [a_n, m_n]))
    assert a.status == ST.CHAMPION and m.status == ST.REJECTED


def test_engine_new_digest_does_not_restart_the_hotkey_window():
    max_blocks = int(CS.ENGINE_KA_MAX_HOURS * 300)
    old_b = B - max_blocks - 10
    w = _sub(4, "w1", DB, B - 5)
    st = TrackState(first_seen={DA: (old_b, "w1"), DB: (B - 5, "w1")})
    assert EC.waiting_uids(st, [w], B, {}) == []           # the hotkey's window opened with its first digest
    st2 = TrackState(first_seen={DB: (B - 5, "w1")})
    assert EC.waiting_uids(st2, [w], B, {}) == [4]


@pytest.mark.parametrize("bad", [[], [["x"]], [None], "junk"])
def test_engine_malformed_challenger_fails_closed(bad):
    w = _sub(4, "w1", DA, B - 10)
    st = TrackState(first_seen={DA: (B - 10, "w1")}, owners={DA: "w1"})
    st.results[DA] = _res(DA, block=B - 5)
    st.challenger = {"digest": DA, "hotkey": "w1", "measurements": bad}
    assert EC.waiting_uids(st, [w], B, {}) == []
