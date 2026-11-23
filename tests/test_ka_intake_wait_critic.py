"""Critic probes for GR2_INTAKE_WAIT_KA (ka-intake-wait-critic-1122), updated for v2: S1 (first commitment wins per
repo@revision), S2 (a round never waits for a lookup launched by an earlier round; budget 1 s) and N2 (daemon lookup threads)
now assert the fixed behaviour. The 72 h rewindow chaining (critic S3, lead decision: unchanged) and the restart note (N1) still
assert the behaviour as briefed."""
import dataclasses
import os
import subprocess
import sys
import textwrap
import time

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import ka_intake_wait as IW
from reax_subnet.spec import evaluator as E
from reax_subnet.spec.commitment import SpecCommitment
from tests.test_ka_intake_wait import C49, D, H, T0, _n, _paid, _snap, hub  # noqa: F401  (hub = autouse fixture)


@pytest.fixture(autouse=True)
def _drain(hub):
    """Hung lookups of one test must not occupy the two lookup workers (or land in the cache) during the next one."""
    def wait():
        end = time.monotonic() + 10.0
        while IW._INFLIGHT and time.monotonic() < end:
            time.sleep(0.05)
    wait()
    IW.reset()
    yield
    wait()


def _copy(uid, ck, commit_ts, of, repo=None):
    """uid committing ``of``'s repo@revision (optionally under another spelling of the repo) at ``commit_ts``."""
    n = _n(uid, ck, commit_ts)
    c = E.parse_commitment(of.commitment)
    return dataclasses.replace(n, commitment=SpecCommitment("generalist", repo or c.repo, c.revision).text())


def _flag_off(fn):
    old = CS.GR2_INTAKE_WAIT_KA
    try:
        CS.GR2_INTAKE_WAIT_KA = False
        return fn()
    finally:
        CS.GR2_INTAKE_WAIT_KA = old


def test_S1_only_the_first_commitment_of_a_public_repo_revision_gets_the_window(hub):
    # 40 hotkeys on 20 coldkeys all commit the SAME public repo@revision: only the earliest on-chain commitment gets the
    # 10-day window; the 39 copies keep the post8.16 windows (none paid at day 9), one lookup in total.
    first = _n(0, "ck0", C49)
    ns = [first] + [_copy(uid, f"ck{uid // 2}", C49 + uid * 60, first) for uid in range(1, 40)]
    assert len({n.commitment for n in ns}) == 1
    now = C49 + 9 * D
    assert _paid(_snap(now, *ns), now) == [0]
    assert len(hub.calls) == 1
    assert _flag_off(lambda: _paid(_snap(now, *ns), now)) == []


def test_S1_copies_get_exactly_the_flag_off_treatment(hub):
    first = _n(0, "ck0", C49)
    copies = [_copy(uid, f"ck{uid}", C49 + uid * 600, first) for uid in range(1, 4)]
    for dt in (1 * H, 5 * H, 7 * H, 20 * H, 30 * H, 3 * D, 9 * D):
        now = C49 + dt
        on = _paid(_snap(now, first, *copies), now)
        off = _flag_off(lambda: _paid(_snap(now, first, *copies), now))
        assert 0 in on                                                     # the first one is inside its 10 days
        assert [u for u in on if u != 0] == [u for u in off if u != 0], dt  # the copies: exactly the post8.16 result


def test_S1_tie_on_the_same_block_goes_to_the_lower_uid(hub):
    a = _n(31, "ckA", C49)
    b = _copy(17, "ckB", C49, a)
    now = C49 + 5 * D
    assert _paid(_snap(now, a, b), now) == [17]


def test_S1_an_original_from_before_the_start_blocks_later_copies(hub):
    orig = _n(5, "ck5", T0 - 2 * D)                  # committed before 20 Nov 16:40Z: old rules, and copies made after it get none
    cp = _copy(6, "ck6", C49, orig)
    now = C49 + 3 * D
    assert _paid(_snap(now, orig, cp), now) == []
    assert hub.calls == []


def test_S1_repo_spelling_does_not_make_a_new_first(hub):
    a = _n(3, "ckA", C49, repo="Org/Model")
    b = _copy(4, "ckB", C49 + H, a, repo="org/model")
    now = C49 + 5 * D
    assert _paid(_snap(now, a, b), now) == [3]


def test_S1_a_copy_on_another_revision_or_repo_is_independent(hub):
    a = _n(3, "ckA", C49)
    b = _n(4, "ckB", C49 + H)                         # own repo, own revision
    now = C49 + 5 * D
    assert _paid(_snap(now, a, b), now) == [3, 4]


def test_S1_first_holder_keeps_it_after_recommitting_the_same_revision(hub):
    # A commits X, B copies X, A recommits X later (newer block): A's remembered earliest time still wins
    a1 = _n(1, "ckA", C49)
    b = _copy(2, "ckB", C49 + 2 * H, a1)
    now = C49 + 3 * H
    assert _paid(_snap(now, a1, b), now) == [1, 2]   # B: still inside its own 6 h unseen grace (old rules)
    a2 = _copy(1, "ckA", C49 + 4 * H, a1)
    now = C49 + 3 * D
    assert _paid(_snap(now, a2, b), now) == [1]


def test_S1_a_first_holder_that_left_still_blocks_the_copy(hub):
    # v2.1 (critic S-B): the remembered first holder keeps first place after it deregistered or recommitted something else
    a = _n(1, "ckA", C49)
    b = _copy(2, "ckB", C49 + 2 * H, a)
    now = C49 + 3 * D
    assert _paid(_snap(now, a, b), now) == [1]
    assert _paid(_snap(now, b), now) == []           # was [2] in v2: the copy inherited the window
    assert _paid(_snap(now, _n(1, "ckA", C49 + 2 * D, rev_i=1), b), now) == [1]   # A recommits Y: A keeps Y's window, B gets nothing


def test_S3_new_revision_every_72h_still_chains_keep_alive_as_briefed(hub):
    # lead decision: unchanged (one hotkey, a new public revision every 72 h: paid without a gap for 30 days)
    gaps = []
    t = C49
    while t < C49 + 30 * D:
        k = int((t - C49) // (72 * H))
        n = _n(7, "ck7", C49 + k * 72 * H, rev_i=k)
        if _paid(_snap(t, n), t) != [7]:
            gaps.append(t)
        t += 6 * H
    assert gaps == []


def test_N1_restart_plus_same_revision_recommit_restarts_the_ten_days(hub):
    n1 = _n(9, "ck9", C49)
    now = C49 + 9.5 * D
    assert _paid(_snap(now, n1), now) == [9]
    n2 = _n(9, "ck9", C49 + 9.5 * D)
    now = C49 + 10.5 * D
    assert _paid(_snap(now, n2), now) == []
    E._IW_TIMES.clear()                              # validator restart: documented limitation, unchanged
    assert _paid(_snap(now, n2), now) == [9]


def test_S2_default_round_budget_is_one_second():
    assert CS.GR2_INTAKE_WAIT_ROUND_BUDGET_S == 1.0


def test_S2_a_hung_lookup_costs_one_bounded_wait_not_one_per_round(hub):
    hub.delay = 3.5                                  # a hung lookup (DNS / socket) that outlives several rounds
    n = _n(11, "ck11", C49)
    waits, paid = [], []
    for i in range(3):
        t0 = time.monotonic()
        paid.append(_paid(_snap(C49 + 30 * H + i, n), C49 + 30 * H + i))
        waits.append(time.monotonic() - t0)
    assert 0.9 <= waits[0] < 1.5                     # round 1 launched it: waits its budget (1 s) once
    assert waits[1] < 0.2 and waits[2] < 0.2         # rounds 2 and 3 do not wait for round 1's lookup
    assert paid == [[11]] * 3                        # unanswered = UNKNOWN: window kept (counted against the cap)
    assert len(hub.calls) == 1


def test_S2_a_new_lookup_in_a_later_round_may_still_wait_within_that_rounds_budget(hub):
    hub.delay = 3.0
    a = _n(11, "ck11", C49)
    _paid(_snap(C49 + 30 * H, a), C49 + 30 * H)      # round 1: lookup of a hangs
    hub.delay = 0.0
    b = _n(12, "ck12", C49)
    t0 = time.monotonic()
    assert _paid(_snap(C49 + 30 * H + 1, a, b), C49 + 30 * H + 1) == [11, 12]
    assert time.monotonic() - t0 < 0.5               # b's own lookup ran on the second worker and was answered at once
    assert IW._CACHE[(E.parse_commitment(b.commitment).repo, E.parse_commitment(b.commitment).revision)][0] == IW.PASS


def test_S2_two_rounds_in_one_weight_set_round_wait_only_once(hub):
    # serving refresh and a background run_cycle -> _refresh_keepalive each build their own Round
    hub.delay = 2.5
    n = _n(11, "ck11", C49)
    now = C49 + 30 * H
    t0 = time.monotonic()
    _paid(_snap(now, n), now)
    _paid(_snap(now, n), now)
    assert time.monotonic() - t0 < 1.5


def test_N2_daemon_lookup_threads_do_not_delay_interpreter_exit(tmp_path):
    code = textwrap.dedent("""
        import time
        from reax_subnet import ka_intake_wait as IW, competition_split as CS
        CS.GR2_INTAKE_WAIT_ROUND_BUDGET_S = 0.2
        IW.hub_check = lambda r, v, t: (time.sleep(5), (IW.PASS, "x"))[1]
        print(IW.Round(0.0).check("a/b", "0" * 40))
    """)
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-c", code], check=True, timeout=30, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")})
    assert time.monotonic() - t0 < 3.0               # was >= 4.5 s with the non-daemon ThreadPoolExecutor
    assert "unknown" in p.stdout
