"""Emission follows need (DECISIONS 26 Oct 2026 emission-follows-need-burn-1030, burn-announced-1026; docs/EMISSION-NEED.md).

From NEED_START_UTC (Fri 30 Oct 2026 14:30Z) the serving uids get exactly the current inference share (0.40), a track we
paused ends former ranks and runs out its champion share (then keep-alive only), and everything unpaid goes to the owner UID (burned). The share and the
pauses are our decisions (release constants + need.json). Before the start, with the switch off or without an owner UID:
exactly the old weights."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import need as N

OWNER = 3
START = CS.NEED_START_UTC
AFTER = (START + timedelta(hours=1)).timestamp()
BEFORE = (START - timedelta(hours=1)).timestamp()
SERVING = {OWNER: 0.0, 10: 0.4, 11: 0.3, 12: 0.2, 13: 0.1}
GEN = {"generalist": [(20, 50.0), (21, 45.0), (22, 40.0)]}


@pytest.fixture
def need_on(monkeypatch):
    monkeypatch.setattr(CS, "NEED_RULE", True)


def _w(now, **kw):
    args = dict(owner_uid=OWNER, now=now, pause_starts={"engine": START.timestamp(), "generalist": START.timestamp()})
    args.update(kw)
    serving = args.pop("serving", SERVING)
    return CS.combine_weights(serving, args.pop("champion", 30), args.pop("frac", 1.0), args.pop("tracks", GEN), "50/50",
                              **args)


def _serving_sum(w, uids=(10, 11, 12, 13)):
    return math.fsum(w.get(u, 0.0) for u in uids)


def test_release_constants():
    import conftest
    assert conftest.RELEASE_NEED_RULE is True
    assert CS.NEED_START_UTC == datetime(2026, 10, 30, 14, 30, tzinfo=timezone.utc)
    assert CS.INFERENCE_SHARE_STEPS[0] == (CS.NEED_START_UTC, 0.40)
    assert CS.TRACK_PAUSES == {}                          # no track is paused by the release; pauses are our decisions


def test_switch_off_and_before_start_are_exactly_the_old_weights(need_on, monkeypatch):
    before = _w(BEFORE)
    assert CS.inference_share(datetime.fromtimestamp(BEFORE, timezone.utc)) is None
    assert before.get(OWNER, 0.0) == 0.0 and pytest.approx(_serving_sum(before)) == 0.5
    monkeypatch.setattr(CS, "NEED_RULE", False)
    assert _w(BEFORE) == before
    assert CS.inference_share(datetime.fromtimestamp(AFTER, timezone.utc)) is None
    off = _w(AFTER)
    assert off.get(OWNER, 0.0) == 0.0
    assert pytest.approx(_serving_sum(off)) == 0.5


def test_inference_is_exactly_the_scheduled_share_and_the_rest_is_burned(need_on):
    w = _w(AFTER)
    assert pytest.approx(sum(w.values())) == 1.0
    assert pytest.approx(_serving_sum(w)) == 0.40
    assert pytest.approx(w[10] / w[13]) == 4.0                    # the serving order is unchanged
    assert pytest.approx(w[30]) == 0.20                            # engine champion, full share
    assert pytest.approx(w[20] + w[21] + w[22]) == 0.30            # generalist ranks
    assert pytest.approx(w[OWNER]) == 0.10                         # 1 - 0.40 - 0.20 - 0.30


def test_unclaimed_and_decayed_competition_shares_are_burned_not_paid_to_inference(need_on):
    w = _w(AFTER, frac=0.6, tracks={})
    assert pytest.approx(_serving_sum(w)) == 0.40
    assert pytest.approx(w[30]) == 0.12
    assert pytest.approx(w[OWNER]) == 1 - 0.40 - 0.12
    none = _w(AFTER, champion=None, tracks={})
    assert pytest.approx(_serving_sum(none)) == 0.40 and pytest.approx(none[OWNER]) == 0.60


def test_unapproved_award_is_burned(need_on):
    w = _w(AFTER, award_ok=lambda t, u: u != 20)
    assert 20 not in w or w[20] == 0.0
    assert pytest.approx(w[OWNER]) == 0.10 + 0.30 * 0.75


def test_owner_burn_fraction_is_kept_on_top(need_on):
    w = _w(AFTER, burn_fraction=0.1, serving={**SERVING, OWNER: 0.1})
    assert pytest.approx(_serving_sum(w)) == 0.9 * 0.40
    assert pytest.approx(w[OWNER]) == 0.1 + 0.9 * 0.10


def test_no_owner_uid_means_no_burn_and_the_old_weights(need_on, monkeypatch):
    serving = {u: v for u, v in SERVING.items() if u != OWNER}
    on = _w(AFTER, owner_uid=None, serving=serving, paused=["generalist"])
    monkeypatch.setattr(CS, "NEED_RULE", False)
    assert on == _w(AFTER, owner_uid=None, serving=serving)


def test_paused_generalist_pays_no_ranks_and_keeps_its_ranked_entries_alive(need_on):
    tr = CS.RankedLeaders(GEN["generalist"])
    tr.crowned = {20: START.timestamp() - 8 * 86400}
    w = _w(AFTER + 86400, paused=["generalist"], tracks={"generalist": tr})
    median = 0.25  # of the serving uids 10..13 (0.4/0.3/0.2/0.1): mean of 0.3 and 0.2, unit = miner part
    each = min(median, 0.25 * 0.30 / 3)
    for u in (20, 21, 22):
        assert pytest.approx(w[u]) == each
    assert pytest.approx(_serving_sum(w)) == 0.40
    assert pytest.approx(w[30]) == 0.20
    assert pytest.approx(w[OWNER]) == 1 - 0.40 - 0.20 - 3 * each


def test_paused_engine_pays_the_champion_the_engine_keepalive_only(need_on):
    w = _w(AFTER + 86400, paused=["engine"], engine_runout={30: (1., START.timestamp() - 8 * 86400)})
    each = min(CS.ENGINE_KA_FRACTION * 0.25, CS.ENGINE_KA_MAX_SHARE * 0.20)
    assert pytest.approx(w[30]) == each
    assert pytest.approx(w[20] + w[21] + w[22]) == 0.30
    assert pytest.approx(w[OWNER]) == 1 - 0.40 - 0.30 - each


def test_pause_is_ignored_before_the_start(need_on):
    assert _w(BEFORE, paused=["engine", "generalist"]) == _w(BEFORE)


def test_need_only_fallback(need_on):
    at = datetime.fromtimestamp(AFTER, timezone.utc)
    w = CS.need_only({10: 0.6, 11: 0.4}, OWNER, 0.0, at)
    assert pytest.approx(w[10] + w[11]) == 0.40 and pytest.approx(w[OWNER]) == 0.60
    assert CS.need_only({10: 0.6, 11: 0.4}, None, 0.0, at) == {10: 0.6, 11: 0.4}
    assert CS.need_only({10: 0.6}, OWNER, 0.0, datetime.fromtimestamp(BEFORE, timezone.utc)) == {10: 0.6}


# ---- our decisions: release constants and need.json ----------------------------------------------------------------------

def _at(**kw):
    return START + timedelta(**kw)


def test_track_paused_from_constants(need_on, monkeypatch):
    monkeypatch.setattr(CS, "TRACK_PAUSES", {"generalist": ((_at(days=3), _at(days=5)),), "engine": ((_at(days=4), None),)})
    assert not CS.track_paused("generalist", _at(days=3) - timedelta(seconds=1))
    assert CS.track_paused("generalist", _at(days=3)) and CS.track_paused("generalist", _at(days=4))
    assert not CS.track_paused("generalist", _at(days=5))
    assert CS.track_paused("engine", _at(days=40))
    monkeypatch.setattr(CS, "NEED_RULE", False)
    assert not CS.track_paused("engine", _at(days=40))


def test_inference_share_steps_latest_wins(need_on):
    assert CS.inference_share(_at(seconds=0)) == 0.40
    assert CS.inference_share(_at(days=2), [(_at(days=1), 0.30)]) == 0.30
    assert CS.inference_share(_at(hours=12), [(_at(days=1), 0.30)]) == 0.40
    assert CS.inference_share(_at(days=3), [(_at(days=1), 0.30), (_at(days=2), 0.45)]) == 0.45


def _need_file(tmp_path, obj):
    p = tmp_path / N.FILE_NAME
    p.write_text(__import__("json").dumps(obj))
    return str(p)


GOOD = {"inference": [{"from_utc": "2026-11-03T14:30:00+00:00", "share": 0.35, "reason": "demand review 2 Nov"}],
        "pauses": [{"track": "engine", "from_utc": "2026-11-04T00:00:00+00:00", "until_utc": None, "reason": "progress flattened"},
                   {"track": "generalist", "from_utc": "2026-11-04T00:00:00+00:00", "until_utc": "2026-11-06T00:00:00+00:00",
                    "reason": "test"}]}


def test_need_file_decides_share_and_pauses(need_on, tmp_path):
    path = _need_file(tmp_path, GOOD)
    t = datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert N.decide(t, path, ["generalist"]) == (0.35, ["engine", "generalist"])
    assert N.decide(datetime(2026, 11, 7, tzinfo=timezone.utc), path, ["generalist"]) == (0.35, ["engine"])
    assert N.decide(_at(hours=1), path, ["generalist"]) == (0.40, [])
    assert N.decide(_at(hours=-1), path, ["generalist"]) == (None, [])
    assert N.decide(t, "", ["generalist"]) == (0.40, [])                       # no file: the release constants
    assert N.decide(t, str(tmp_path / "absent.json"), ["generalist"]) == (0.40, [])


@pytest.mark.parametrize("bad", [
    [],                                                                          # not an object
    {"inference": [], "other": []},                                              # unknown key
    {"inference": [{"from_utc": "2026-11-03T14:30:00", "share": 0.35, "reason": "x"}]},           # no timezone
    {"inference": [{"from_utc": "2026-10-29T14:30:00+00:00", "share": 0.35, "reason": "x"}]},     # before the start
    {"inference": [{"from_utc": "2026-11-03T14:30:00+00:00", "share": 3.5, "reason": "x"}]},      # typo
    {"inference": [{"from_utc": "2026-11-03T14:30:00+00:00", "share": True, "reason": "x"}]},
    {"inference": [{"from_utc": "2026-11-03T14:30:00+00:00", "share": 0.35, "reason": ""}]},
    {"pauses": [{"track": "engine", "from_utc": "2026-11-04T00:00:00+00:00", "until_utc": "2026-11-03T00:00:00+00:00",
                 "reason": "x"}]},                                               # until before from
    {"pauses": [{"track": "engine", "from_utc": "2026-11-04T00:00:00+00:00", "reason": "x"}]},   # missing key
])
def test_a_bad_need_file_is_ignored_whole(need_on, tmp_path, bad, caplog):
    path = _need_file(tmp_path, bad)
    assert N.decide(datetime(2026, 11, 5, tzinfo=timezone.utc), path, ["generalist"]) == (0.40, [])
    assert "refused" in caplog.text and "release constants apply" in caplog.text


def test_garbage_need_file_is_ignored(need_on, tmp_path):
    p = tmp_path / N.FILE_NAME
    p.write_text("{nope")
    assert N.decide(datetime(2026, 11, 5, tzinfo=timezone.utc), str(p), []) == (0.40, [])


def test_explicit_inference_share_is_used(need_on):
    w = _w(AFTER, inference=0.30)
    assert pytest.approx(_serving_sum(w)) == 0.30 and pytest.approx(w[OWNER]) == 0.20
    assert _w(BEFORE, inference=0.30) == _w(BEFORE)                              # before the start: ignored


# ---- through the validator's remote competition path (engvalidator.weights) ---------------------------------------------

def test_remote_validator_path_caps_inference_burns_the_rest_and_pauses_the_engine(need_on, tmp_path, monkeypatch):
    from test_remote_competitions import Clock, SERVING as RSERV, snap, stamp, write_engine_state
    from reax_subnet import remote as RM
    from reax_subnet.config import ENGINE_STATE_FILE

    tue = datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(CS, "TRACK_PAUSES", {"engine": ((tue + timedelta(hours=6), None),)})
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    import json
    raw = json.loads(path.read_text())
    raw["tracks"]["s1-fast"]["history"] = [{"event": "crowned", "digest": raw["tracks"]["s1-fast"]["champion"]["digest"],
        "hotkey": "hk-eng", "slot_utc": (START - timedelta(days=8)).isoformat()}]
    path.write_text(json.dumps(raw))
    now = [START + timedelta(hours=1)]
    clock = Clock(now[0].timestamp())
    stamp(path, clock.t)
    comp = RM.RemoteEngineCompetition(str(path), utcnow=lambda: now[0], wall=clock)
    s = snap()                                  # owner hotkey "owner" = uid 0, engine champion uid 3, serving 11 / 12
    w = comp.weights(RSERV, s, {0}, 0.0)
    assert pytest.approx(w[11] + w[12]) == 0.40 and pytest.approx(w[11] / w[12]) == 1.5
    assert pytest.approx(w[3], rel=1e-3) == 0.20
    assert pytest.approx(w[0], rel=1e-3) == 0.40          # no generalist result: its 0.30 and the 0.10 rest are burned
    # Old crown: full for 24h notice from the pause start, then keep-alive only.
    now[0] = tue + timedelta(hours=31)
    clock.t = now[0].timestamp()
    stamp(path, clock.t)
    w = comp.weights(RSERV, s, {0}, 0.0)
    each = min(CS.ENGINE_KA_FRACTION * 0.5, CS.ENGINE_KA_MAX_SHARE * 0.20)
    assert pytest.approx(w[3], rel=1e-3) == each
    assert pytest.approx(w[11] + w[12]) == 0.40
    assert pytest.approx(w[0], rel=1e-3) == 1 - 0.40 - each


def test_remote_validator_reads_need_json_next_to_the_award_approvals(need_on, tmp_path):
    import json
    from types import SimpleNamespace
    from test_remote_competitions import Clock, SERVING as RSERV, snap, stamp, write_engine_state
    from reax_subnet import remote as RM
    from reax_subnet.config import ENGINE_STATE_FILE

    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    raw = json.loads(path.read_text())
    raw["tracks"]["s1-fast"]["history"] = [{"event": "crowned", "digest": raw["tracks"]["s1-fast"]["champion"]["digest"],
        "hotkey": "hk-eng", "slot_utc": (START - timedelta(days=8)).isoformat()}]
    path.write_text(json.dumps(raw))
    now = [datetime(2026, 11, 5, tzinfo=timezone.utc)]
    clock = Clock(now[0].timestamp())
    stamp(path, clock.t)
    gate = SimpleNamespace(path=str(tmp_path / "approvals.json"), approved=lambda *a, **k: True)
    (tmp_path / N.FILE_NAME).write_text(json.dumps(GOOD))
    comp = RM.RemoteEngineCompetition(str(path), utcnow=lambda: now[0], wall=clock, award_gate=gate)
    w = comp.weights(RSERV, snap(), {0}, 0.0)
    assert pytest.approx(w[11] + w[12]) == 0.35                       # need.json step from Tue 3 Nov
    each = min(CS.ENGINE_KA_FRACTION * 0.5, CS.ENGINE_KA_MAX_SHARE * 0.20)
    assert pytest.approx(w[3], rel=1e-3) == each                      # engine paused by need.json
    assert pytest.approx(w[0], rel=1e-3) == 1 - 0.35 - each


# ---- critic round 1 ------------------------------------------------------------------------------------------------------

def test_paused_track_gives_no_keepalive_to_a_rejected_entry(need_on):
    w = _w(AFTER, paused=["generalist"], award_ok=lambda t, u: u != 20)
    assert w.get(20, 0.0) == 0.0
    each = min(0.25, 0.25 * 0.30 / 2)
    assert pytest.approx(w[21]) == each and pytest.approx(w[22]) == each


def test_inference_share_never_squeezes_the_paid_competition_parts(need_on):
    w = _w(AFTER, inference=0.80)
    assert pytest.approx(w[30]) == 0.20 and pytest.approx(w[20] + w[21] + w[22]) == 0.30
    assert pytest.approx(_serving_sum(w)) == 0.50 and w.get(OWNER, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_owner_pool_fallback_is_part_of_the_inference_share_and_burned(need_on):
    # a pool without an eligible miner leaves its part on the owner UID in the serving vector: it counts inside the
    # inference share (and is burned with the rest); the serving miners keep their proportions
    w = _w(AFTER, serving={OWNER: 0.3, 10: 0.4, 11: 0.3})
    assert pytest.approx(w[10] + w[11]) == 0.40 * 0.7
    assert pytest.approx(w[OWNER]) == 0.40 * 0.3 + 0.10


def test_a_refused_need_file_keeps_the_last_good_content(need_on, tmp_path, caplog):
    path = _need_file(tmp_path, GOOD)
    t = datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert N.decide(t, path, ["generalist"]) == (0.35, ["engine", "generalist"])
    (tmp_path / N.FILE_NAME).write_text('{"inference": [')                    # half written
    assert N.decide(t, path, ["generalist"]) == (0.35, ["engine", "generalist"])
    assert "last good content stays in force" in caplog.text


def test_a_pause_for_an_unknown_track_is_logged(need_on, tmp_path, caplog):
    bad = {"pauses": [{"track": "genralist", "from_utc": "2026-11-04T00:00:00+00:00", "until_utc": None, "reason": "typo"}]}
    path = _need_file(tmp_path, bad)
    assert N.decide(datetime(2026, 11, 5, tzinfo=timezone.utc), path, ["generalist"]) == (0.40, [])
    assert "genralist" in caplog.text


def test_remote_last_resort_fallback_caps_inference_and_never_raises(need_on, tmp_path, monkeypatch):
    from test_remote_competitions import Clock, SERVING as RSERV, snap, stamp, write_engine_state
    from reax_subnet import remote as RM
    from reax_subnet.config import ENGINE_STATE_FILE
    from reax_subnet.engvalidator import EngineCompetition

    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    comp = RM.RemoteEngineCompetition(str(path), utcnow=lambda: START + timedelta(hours=1), wall=Clock())
    monkeypatch.setattr(EngineCompetition, "weights", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(RM, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: START + timedelta(hours=1))}))
    w = comp.weights(RSERV, snap(), {0}, 0.0)
    assert pytest.approx(w[11] + w[12]) == 0.40 and pytest.approx(w[0]) == 0.60
    monkeypatch.setattr(RM, "need_only", lambda *a, **k: (_ for _ in ()).throw(ValueError("y")))
    assert comp.weights(RSERV, snap(), {0}, 0.0) == dict(RSERV)
