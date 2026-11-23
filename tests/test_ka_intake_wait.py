"""Round-2 intake-wait keep-alive (GR2_INTAKE_WAIT_KA, reax_subnet/ka_intake_wait.py, spec/evaluator.py::_committed_waiting).

A Generalist commitment made on chain at or after 2026-11-20T16:40Z that is still undecided keeps the provisional keep-alive
for 10 days from its own commitment (legacy path only; the hub is faked, no test touches the network)."""
import logging
import random
import time
import urllib.error
from datetime import datetime, timezone

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import ka_intake_wait as IW
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.spec import evaluator as E, state as ST
from reax_subnet.spec.commitment import SpecCommitment

H, D = 3600.0, 86400.0
T0 = datetime(2026, 11, 20, 16, 40, tzinfo=timezone.utc).timestamp()      # = GR2_INTAKE_WAIT_KA_START_UTC; block B0 is at T0
B0 = 9_400_000
C49 = T0 + 4800.0                  # uid 49: 20 Nov 18:00Z
C59 = T0 + 58800.0                 # uid 59: 21 Nov 09:00Z (same coldkey)
FEED_ON = datetime(2026, 11, 21, 15, tzinfo=timezone.utc).timestamp()      # ka_feed.GR2_KA_FEED_START_UTC
REAL_HUB_CHECK = IW.hub_check      # the autouse fixture replaces IW.hub_check by a fake


def _blk(ts):
    return B0 + int(round((ts - T0) / 12.0))


def _hk(uid):
    return f"5Hk{uid:03d}" + "x" * 40


def _n(uid, ck, commit_ts, rev_i=0, repo=None):
    rev = f"{uid * 100 + rev_i:040x}"
    com = SpecCommitment("generalist", repo or f"m/m{uid}", rev).text()
    return Neuron(uid, _hk(uid), ck, False, 0.0, None, None, commitment=com, commitment_block=_blk(commit_ts),
                  registered_block=_blk(commit_ts) - 20000)


def _snap(now, *ns):
    return Snapshot(block=_blk(now), neurons=tuple(ns), owner_hotkey=None)


def _paid(snap, now, st=None):
    return sorted(E._committed_waiting(st or ST.SpecState(), ["generalist"], snap, now)["generalist"])


class Hub:
    """Fake for ``IW.hub_check``: PASS unless told otherwise; records every lookup."""

    def __init__(self):
        self.calls, self.by_repo, self.default, self.delay = [], {}, (IW.PASS, "public, safetensors listed"), 0.0

    def __call__(self, repo, revision, timeout):
        self.calls.append((repo, revision, timeout))
        if self.delay:
            time.sleep(self.delay)
        v = self.by_repo.get(repo, self.default)
        if isinstance(v, Exception):
            raise v
        return v


@pytest.fixture(autouse=True)
def hub(monkeypatch):
    for k in ("WAITING_KA_HOTFIX", "KA_BIG_UNDECIDED", "KA_GAUNTLET_R2", "KA_EXPOSED_FIRST", "KA_PROV_TWO_PER_COLDKEY",
              "KA_ABUSE_CLAUSE", "GR2_INTAKE_WAIT_KA"):
        monkeypatch.setattr(CS, k, True)
    monkeypatch.delenv("REAX_KA_FEED_PATH", raising=False)
    h = Hub()
    monkeypatch.setattr(IW, "hub_check", h)
    IW.reset()
    IW.reset_first_seen()
    E._IW_TIMES.clear()
    yield h
    IW.reset()
    IW.reset_first_seen()
    E._IW_TIMES.clear()


def _replay(flag, monkeypatch, hub):
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", flag)
    ns = (_n(49, "X", C49), _n(59, "X", C59))
    out = []
    for now in (C49 + 30 * H, C59 + 30 * H, C49 + 9.9 * D, C49 + 10.1 * D, C59 + 9.9 * D, C59 + 10.1 * D):
        out.append(_paid(_snap(now, *ns), now))
    return out


def test_uid49_uid59_replay_never_seen_keep_alive_for_ten_days_from_each_own_commitment(monkeypatch, hub):
    # +30 h after uid 49 (uid 59 is 15 h in), +30 h after uid 59, +9.9 d / +10.1 d after uid 49, +9.9 d / +10.1 d after uid 59
    assert _replay(True, monkeypatch, hub) == [[49, 59], [49, 59], [49, 59], [59], [59], []]


def test_flag_false_is_the_post816_behaviour(monkeypatch, hub):
    assert _replay(False, monkeypatch, hub) == [[59], [], [], [], [], []]      # uid 49 at +30 h: none; nothing is ever looked up
    assert hub.calls == []


def test_window_is_half_open(hub):
    ns = (_n(49, "X", C49),)
    now = C49 + 10 * D - 12
    assert _paid(_snap(now, *ns), now) == [49]
    now = C49 + 10 * D
    assert _paid(_snap(now, *ns), now) == []


def test_commitment_before_the_start_is_unchanged(monkeypatch, hub):
    ns = (_n(49, "X", T0 - 12),)           # 12 s before 20 Nov 16:40Z
    for now in (T0 - 12 + 10 * H, T0 - 12 + 30 * H):
        on = _paid(_snap(now, *ns), now)
        monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", False)
        assert on == _paid(_snap(now, *ns), now)
        monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", True)
    assert _paid(_snap(T0 - 12 + 10 * H, *ns), T0 - 12 + 10 * H) == [49] and _paid(_snap(T0 - 12 + 30 * H, *ns), T0 - 12 + 30 * H) == []
    assert hub.calls == []
    ns = (_n(49, "X", T0),)                # exactly at the start: in
    assert _paid(_snap(T0 + 30 * H, *ns), T0 + 30 * H) == [49]


def test_before_start_random_worlds_are_identical_with_the_flag_on_and_off(monkeypatch, hub):
    rnd = random.Random(7)
    for _ in range(40):
        st, ns = ST.SpecState(), []
        for uid in range(rnd.randint(1, 12)):
            ct = T0 - rnd.randint(1, 20 * 86400) // 12 * 12
            nr = _n(uid, f"ck{rnd.randint(0, 5)}", ct)
            ns.append(nr)
            kind = rnd.choice(("unseen", "private", "ours"))
            if kind != "unseen":
                rev = nr.commitment.rsplit("@", 1)[1]
                s = ST.Submission(key=f"{nr.hotkey}|{nr.commitment}", uid=uid, hotkey=nr.hotkey, coldkey=nr.coldkey, track_id="generalist",
                                  repo=f"m/m{uid}", revision=rev, first_seen_block=nr.commitment_block, first_seen_ts=ct, status=ST.QUEUED)
                if kind == "ours":
                    s.hash_named, s.delta_checked = "h" * 64, True
                else:
                    s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 401", ST.WAIT_MINER
                st.track("generalist").submissions[s.key] = s
        for now in (T0 - 1 * H, T0, T0 + 30 * H):
            monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", True)
            on = _paid(_snap(now, *[x for x in ns if x.commitment_block <= _blk(now)]), now, st)
            monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", False)
            assert on == _paid(_snap(now, *[x for x in ns if x.commitment_block <= _blk(now)]), now, st)
    assert hub.calls == []


def test_a_third_hotkey_of_the_same_coldkey_is_not_paid(hub):
    ns = (_n(49, "X", C49), _n(59, "X", C49 + 60), _n(60, "X", C49 + 120))
    now = C49 + 30 * H
    assert len(_paid(_snap(now, *ns), now)) == 2      # MAX_OPEN_PER_COLDKEY_TRACK, unchanged


@pytest.mark.parametrize("detail", ["hub answered HTTP 404: repo or revision not public", "hub answered HTTP 401: repo or revision not public",
                                    "hub answered HTTP 403: repo or revision not public", "no safetensors weights listed",
                                    "repo is private or gated"])
def test_failed_public_check_gives_no_ten_day_window(hub, detail):
    hub.default = (IW.FAIL, detail)
    ns = (_n(49, "X", C49),)
    assert _paid(_snap(C49 + 10 * H, *ns), C49 + 10 * H) == [49]        # the old 24 h window (unseen) still applies
    assert _paid(_snap(C49 + 30 * H, *ns), C49 + 30 * H) == []
    assert _paid(_snap(C49 + 3 * D, *ns), C49 + 3 * D) == []
    n_calls = len(hub.calls)
    assert _paid(_snap(C49 + 3 * D + H, *ns), C49 + 3 * D + H) == [] and len(hub.calls) == n_calls     # a definite FAIL is cached too


def test_a_repo_that_turns_public_gets_the_window_after_the_cache_expires(hub):
    hub.default = (IW.FAIL, "hub answered HTTP 401: repo or revision not public")
    ns = (_n(49, "X", C49),)
    assert _paid(_snap(C49 + 30 * H, *ns), C49 + 30 * H) == []
    hub.default = (IW.PASS, "public, safetensors listed")
    now = C49 + 30 * H + 5 * H
    assert _paid(_snap(now, *ns), now) == [] and len(hub.calls) == 1   # still cached
    now = C49 + 30 * H + 6.1 * H
    assert _paid(_snap(now, *ns), now) == [49] and len(hub.calls) == 2


@pytest.mark.parametrize("err", [OSError("network down"), TimeoutError("slow"), RuntimeError("bug in the check")])
def test_hub_error_keeps_the_window_but_never_beyond_ten_days(hub, err):
    hub.default = err
    ns = (_n(49, "X", C49),)
    for now, want in ((C49 + 30 * H, [49]), (C49 + 9.9 * D, [49]), (C49 + 10.1 * D, [])):
        assert _paid(_snap(now, *ns), now) == want


def test_hub_answer_is_cached_six_hours_and_errors_are_retried_after_ten_minutes(hub):
    ns = (_n(49, "X", C49),)
    for h in (1, 2, 5, 6.9):                       # first lookup at +1 h, answer good for 6 h
        now = C49 + h * H
        assert _paid(_snap(now, *ns), now) == [49]
    assert len(hub.calls) == 1
    now = C49 + 7.1 * H
    _paid(_snap(now, *ns), now)
    assert len(hub.calls) == 2
    IW.reset()
    hub.calls.clear()
    hub.default = OSError("down")
    base = C49 + 2 * D
    for dt, calls in ((0, 1), (5 * 60, 1), (9 * 60, 1), (11 * 60, 2)):
        _paid(_snap(base + dt, *ns), base + dt)
        assert len(hub.calls) == calls


def test_a_known_verdict_survives_a_later_hub_outage(hub):
    ns = (_n(49, "X", C49), _n(59, "Y", C49))
    hub.by_repo = {"m/m59": (IW.FAIL, "hub answered HTTP 404: repo or revision not public")}
    now = C49 + 30 * H
    assert _paid(_snap(now, *ns), now) == [49]
    hub.by_repo = {"m/m49": OSError("down"), "m/m59": OSError("down")}
    hub.default = OSError("down")
    now = C49 + 37 * H                       # both cached verdicts expired, the hub is down: the last answers stand
    assert _paid(_snap(now, *ns), now) == [49]


def test_lookups_per_round_are_limited_and_the_rest_is_kept_and_checked_in_later_rounds(monkeypatch, hub):
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_MAX_LOOKUPS", 2)
    ns = tuple(_n(uid, f"ck{uid}", C49) for uid in range(5))
    now = C49 + 30 * H
    assert _paid(_snap(now, *ns), now) == [0, 1, 2, 3, 4]        # unchecked ones keep the window (hub limit is our problem)
    assert len(hub.calls) == 2
    now += 60
    _paid(_snap(now, *ns), now)
    assert len(hub.calls) == 4
    now += 60
    _paid(_snap(now, *ns), now)
    assert len(hub.calls) == 5 and len({c[:2] for c in hub.calls}) == 5      # each (repo, revision) looked up once


def test_round_never_waits_longer_than_its_budget_and_the_late_answer_is_used_later(monkeypatch, hub):
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_ROUND_BUDGET_S", 0.3)
    hub.delay = 1.2
    ns = tuple(_n(uid, f"ck{uid}", C49) for uid in range(3))
    now = C49 + 30 * H
    t = time.monotonic()
    assert _paid(_snap(now, *ns), now) == [0, 1, 2]
    assert time.monotonic() - t < 0.9
    time.sleep(2.0)                                  # the late answers land in the cache
    calls = len(hub.calls)
    t = time.monotonic()
    assert _paid(_snap(now + 60, *ns), now + 60) == [0, 1, 2]
    assert time.monotonic() - t < 0.3 and len(hub.calls) == calls


def test_lookup_timeout_handed_to_the_hub_is_at_most_ten_seconds(hub):
    assert CS.GR2_INTAKE_WAIT_LOOKUP_TIMEOUT_S <= 10 and CS.GR2_INTAKE_WAIT_ROUND_BUDGET_S <= 10
    now = C49 + 30 * H
    _paid(_snap(now, _n(49, "X", C49)), now)
    assert hub.calls and all(0 < c[2] <= 10 for c in hub.calls)


def test_fake_coldkey_gets_nothing_by_the_abuse_clause(hub):
    st = ST.SpecState()
    fake = ST.Submission(key="5Fake|x", uid=7, hotkey="5Fake", coldkey="X", track_id="generalist", repo="m/old", revision="a" * 40,
                         first_seen_block=B0 - 1000, first_seen_ts=T0 - 3 * D, status=ST.REJECTED)
    fake.reason = "no safetensors weights listed"
    st.track("generalist").submissions[fake.key] = fake
    ns = (_n(49, "X", C49), _n(59, "Y", C49))
    now = C49 + 30 * H
    assert _paid(_snap(now, *ns), now, st) == [59]               # coldkey X: one fake within 30 days: no unverified window


def test_two_fakes_exclude_the_coldkey_even_when_the_check_passes(hub):
    st = ST.SpecState()
    for i in range(2):
        f = ST.Submission(key=f"5Fake{i}|x", uid=7 + i, hotkey=f"5Fake{i}", coldkey="X", track_id="generalist", repo="m/old",
                          revision=f"{i + 1:040x}", first_seen_block=B0 - 1000, first_seen_ts=T0 - 3 * D, status=ST.REJECTED)
        f.reason = "weights are identical to the declared base model"
        st.track("generalist").submissions[f.key] = f
    now = C49 + 2 * H
    assert _paid(_snap(now, _n(49, "X", C49)), now, st) == []


def test_a_seen_entry_with_a_404_is_a_fake_and_gets_nothing(hub):
    st = ST.SpecState()
    n = _n(49, "X", C49)
    s = ST.Submission(key=f"{n.hotkey}|{n.commitment}", uid=49, hotkey=n.hotkey, coldkey="X", track_id="generalist", repo="m/m49",
                      revision=n.commitment.rsplit("@", 1)[1], first_seen_block=n.commitment_block, first_seen_ts=C49, status=ST.QUEUED)
    s.reason, s.wait_kind = E.WAITING_FOR_REPO + ": 404", ST.WAIT_MINER
    st.track("generalist").submissions[s.key] = s
    now = C49 + 2 * H
    assert _paid(_snap(now, n), now, st) == [] and hub.calls == []


def _seen(n, **kw):
    s = ST.Submission(key=f"{n.hotkey}|{n.commitment}", uid=n.uid, hotkey=n.hotkey, coldkey=n.coldkey, track_id="generalist", repo="m/x",
                      revision=n.commitment.rsplit("@", 1)[1], first_seen_block=n.commitment_block,
                      first_seen_ts=T0 + (n.commitment_block - B0) * 12.0, status=ST.QUEUED)
    for k, v in kw.items():
        setattr(s, k, v)
    st = ST.SpecState()
    st.track("generalist").submissions[s.key] = s
    return st


def test_a_seen_private_repo_entry_gets_ten_days_once_the_repo_is_public(hub):
    n = _n(49, "X", C49)
    st = _seen(n, reason=E.WAITING_FOR_REPO + ": 401", wait_kind=ST.WAIT_MINER)
    hub.default = (IW.FAIL, "hub answered HTTP 401: repo or revision not public")
    assert _paid(_snap(C49 + 30 * H, n), C49 + 30 * H, st) == []                   # still private: 24 h as before
    hub.default = (IW.PASS, "public, safetensors listed")
    now = C49 + 40 * H
    assert _paid(_snap(now, n), now, st) == [49]                                   # public now: the window of its own commitment
    now = C49 + 10.1 * D
    assert _paid(_snap(now, n), now, st) == []


def test_an_entry_the_evaluator_verified_is_never_looked_up_and_never_gets_less_than_ten_days(hub):
    n = _n(49, "X", C49)
    st = _seen(n, hash_named="h" * 64, delta_checked=True)
    for now, want in ((C49 + 5 * D, [49]), (C49 + 9.9 * D, [49]), (C49 + 10.1 * D, [])):     # our-wait was 96 h
        assert _paid(_snap(now, n), now, st) == want
    assert hub.calls == []


def test_unchecked_entries_count_against_the_unverified_cap_passed_ones_do_not(monkeypatch, hub):
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_MAX_LOOKUPS", 100)
    ns = tuple(_n(uid, f"ck{uid}", C49 + uid * 12) for uid in range(20))
    now = C49 + 30 * H
    assert _paid(_snap(now, *ns), now) == list(range(20))                   # all passed: not counted
    IW.reset()
    E._IW_TIMES.clear()
    hub.default = OSError("down")
    assert _paid(_snap(now, *ns), now) == list(range(CS.PROVISIONAL_KA_UNVERIFIED_MAX))   # unchecked: oldest 16 only, as today


def test_recommit_of_the_same_repo_and_revision_never_starts_a_fresh_window(hub):
    n0 = _n(49, "X", C49)
    for now in (C49 + 1 * H, C49 + 8 * D):
        assert _paid(_snap(now, n0), now) == [49]
    again = _n(49, "X", C49 + 9 * D)                                           # same repo@revision, new commitment block
    assert again.commitment == n0.commitment
    for now, want in ((C49 + 9 * D + H, [49]), (C49 + 9.9 * D, [49]), (C49 + 10.5 * D, [])):
        assert _paid(_snap(now, again), now) == want


def test_a_new_revision_inside_the_rewindow_shares_the_running_window(hub):
    assert CS.KA_REWINDOW_HOURS == 72.0
    n0 = _n(49, "X", C49)
    assert _paid(_snap(C49 + H, n0), C49 + H) == [49]
    n1 = _n(49, "X", C49 + 1 * D, rev_i=1)
    for now, want in ((C49 + 1 * D + H, [49]), (C49 + 9.9 * D, [49]), (C49 + 10.1 * D, [])):    # not until day 11
        assert _paid(_snap(now, n1), now) == want


def test_a_new_revision_after_the_rewindow_gets_its_own_window_from_its_commitment(hub):
    n0 = _n(49, "X", C49)
    assert _paid(_snap(C49 + H, n0), C49 + H) == [49]
    n1 = _n(49, "X", C49 + 4 * D, rev_i=1)                                     # more than 72 h after the first window started
    for now, want in ((C49 + 4 * D + H, [49]), (C49 + 10.5 * D, [49]), (C49 + 13.9 * D, [49]), (C49 + 14.1 * D, [])):
        assert _paid(_snap(now, n1), now) == want
    n2 = _n(49, "X", C49 + 5 * D, rev_i=2)                                     # inside the second window's rewindow: shares it
    for now, want in ((C49 + 13.9 * D, [49]), (C49 + 14.1 * D, [])):
        assert _paid(_snap(now, n2), now) == want
    n1_back = _n(49, "X", C49 + 6 * D, rev_i=1)                               # the first new revision again: its own earlier time stands
    assert _paid(_snap(C49 + 14.1 * D, n1_back), C49 + 14.1 * D) == []


def test_a_recommit_with_a_failing_repo_gets_no_ten_day_window(hub):
    hub.by_repo = {"m/m49": (IW.FAIL, "hub answered HTTP 404: repo or revision not public")}
    ns = (_n(49, "X", C49),)
    assert _paid(_snap(C49 + 30 * H, *ns), C49 + 30 * H) == []


def test_feed_mode_active_leaves_the_legacy_result_untouched_and_looks_nothing_up(monkeypatch, hub):
    monkeypatch.setenv("REAX_KA_FEED_PATH", "/nonexistent/feed.json")
    ns = (_n(49, "X", C49), _n(59, "X", C59))
    now = C49 + 30 * H                                                            # 21 Nov 24:00Z, after the feed start
    assert now >= FEED_ON
    on = _paid(_snap(now, *ns), now)
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", False)
    assert on == _paid(_snap(now, *ns), now) == [59]
    assert hub.calls == []
    # before the feed start the same environment variable changes nothing: the patch applies
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", True)
    now = C49 + 10 * H
    assert now < FEED_ON and _paid(_snap(now, *ns[:1]), now) == [49] and hub.calls


def test_ordinary_tracks_and_other_track_commitments_are_not_touched(hub):
    com = SpecCommitment("demand_forecasting", "m/m9", "9" * 40).text()
    n = Neuron(9, _hk(9), "X", False, 0.0, None, None, commitment=com, commitment_block=_blk(C49), registered_block=_blk(C49) - 20000)
    now = C49 + 30 * H
    out = E._committed_waiting(ST.SpecState(), ["demand_forecasting"], _snap(now, n), now)
    assert out == {"demand_forecasting": []} and hub.calls == []


def test_one_info_line_per_applied_entry(hub, caplog):
    ns = (_n(49, "X", C49), _n(59, "Y", C59))
    now = C49 + 30 * H
    with caplog.at_level(logging.INFO, logger="reax.spec"):
        _paid(_snap(now, *ns), now)
    lines = [r.getMessage() for r in caplog.records if "intake-wait keep-alive" in r.getMessage()]
    assert len(lines) == 2
    assert "uid 49" in lines[0] and _hk(49)[:8] in lines[0] and "m/m49@" in lines[0] and "window ends 2026-11-30T18:00Z" in lines[0] \
        and "public check pass" in lines[0]
    assert "uid 59" in lines[1] and "window ends 2026-12-01T09:00Z" in lines[1]


def test_compute_provisional_is_the_public_entry_point(hub):
    ns = (_n(49, "X", C49),)
    now = C49 + 30 * H
    assert E.compute_provisional(ST.SpecState(), ["generalist"], _snap(now, *ns), now=now) == {"generalist": [49]}


# ----------------------------------------------------------------------------------------------- the hub request itself


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]


def _info(**kw):
    import json
    d = {"sha": "a" * 40, "private": False, "gated": False, "siblings": [{"rfilename": "config.json"}, {"rfilename": "model-00001.safetensors"}]}
    d.update(kw)
    return json.dumps(d).encode()


def _http(code):
    return urllib.error.HTTPError("https://huggingface.co/x", code, "x", {}, None)


@pytest.fixture
def urlopen(monkeypatch):
    seen = []

    def fake(req, timeout=None):
        seen.append((req, timeout))
        out = fake.answer
        if isinstance(out, BaseException):
            raise out
        return out
    fake.answer, fake.seen = _Resp(_info()), seen
    monkeypatch.setattr(IW, "_urlopen", fake)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    return fake


def test_hub_check_pass_is_anonymous_and_bounded(urlopen):
    assert REAL_HUB_CHECK("owner/model", "a" * 40, 99)[0] == IW.PASS
    req, timeout = urlopen.seen[0]
    assert req.full_url == "https://huggingface.co/api/models/owner/model/revision/" + "a" * 40
    assert timeout == 10.0                                                             # never above ten seconds
    assert not any(k.lower() in ("authorization", "cookie") for k in req.headers)       # anonymous


@pytest.mark.parametrize("answer,verdict", [
    (_http(404), IW.FAIL), (_http(401), IW.FAIL), (_http(403), IW.FAIL),
    (_http(429), IW.UNKNOWN), (_http(500), IW.UNKNOWN), (_http(503), IW.UNKNOWN), (_http(418), IW.UNKNOWN),
    (urllib.error.URLError("dns"), IW.UNKNOWN), (TimeoutError("t"), IW.UNKNOWN), (OSError("reset"), IW.UNKNOWN),
    (_Resp(b"not json"), IW.UNKNOWN), (_Resp(b"[]"), IW.UNKNOWN), (_Resp(_info(), status=204), IW.UNKNOWN),
    (_Resp(b"x" * (IW.MAX_BODY_BYTES + 5)), IW.FAIL),
    (_Resp(_info(private=True)), IW.FAIL), (_Resp(_info(gated="auto")), IW.FAIL), (_Resp(_info(sha="b" * 40)), IW.FAIL),
    (_Resp(_info(siblings=[{"rfilename": "pytorch_model.bin"}])), IW.FAIL), (_Resp(_info(siblings=[])), IW.FAIL),
    (_Resp(_info(siblings=None)), IW.FAIL), (_Resp(_info(siblings=[{"rfilename": "weights.safetensors.index.json"}])), IW.FAIL),
    (_Resp(_info()), IW.PASS)])
def test_hub_check_verdicts(urlopen, answer, verdict):
    urlopen.answer = answer
    got, detail = REAL_HUB_CHECK("owner/model", "a" * 40, 4)
    assert got == verdict and isinstance(detail, str) and detail


def test_hub_endpoint_override_only_accepts_http_urls(urlopen, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")
    REAL_HUB_CHECK("o/m", "a" * 40, 4)
    assert urlopen.seen[-1][0].full_url.startswith("https://mirror.example/api/models/")
    monkeypatch.setenv("HF_ENDPOINT", "file:///etc")
    REAL_HUB_CHECK("o/m", "a" * 40, 4)
    assert urlopen.seen[-1][0].full_url.startswith("https://huggingface.co/api/models/")
