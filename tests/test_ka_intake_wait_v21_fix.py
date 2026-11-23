"""Tests for the critic's proposed SHOULD diffs (run in the fix tree only)."""
import dataclasses
import json

from reax_subnet import ka_intake_wait as IW
from reax_subnet.spec import evaluator as E, state as ST
from reax_subnet.spec.commitment import SpecCommitment
from tests.test_ka_intake_wait import C49, D, H, _blk, _n, _snap, hub  # noqa: F401


def _copy(uid, ck, ts, of):
    n = _n(uid, ck, ts)
    c = E.parse_commitment(of.commitment)
    return dataclasses.replace(n, commitment=SpecCommitment("generalist", c.repo, c.revision).text())


def _sub(n, seen_ts, chain_ts):
    c = E.parse_commitment(n.commitment)
    return ST.Submission(key=f"{n.hotkey}|{c.text()}", uid=n.uid, hotkey=n.hotkey, coldkey=n.coldkey, track_id="generalist",
                         repo=c.repo, revision=c.revision, first_seen_block=_blk(seen_ts), first_seen_ts=seen_ts,
                         chain_commit_block=None if chain_ts is None else _blk(chain_ts),
                         chain_commit_ts=None if chain_ts is None else chain_ts, status=ST.WITHDRAWN)


def _state(*subs):
    st = ST.SpecState()
    tr = st.track("generalist")
    for s in subs:
        tr.submissions[s.key + "#w"] = s           # withdrawn record: never matches a live key, so only the seed uses it
    return st


def _paid(snap, now, path, st):
    return sorted(E._committed_waiting(st, ["generalist"], snap, now, str(path))["generalist"])


def test_F1_P1b_fresh_process_is_covered_by_the_chain_commit_block_seed(hub, tmp_path):
    a1 = _n(1, "ckA", C49)
    b = _copy(2, "ckB", C49 + 2 * H, a1)
    a2 = dataclasses.replace(a1, commitment_block=_snap(C49 + 4 * H).block)
    now = C49 + 3 * D
    st = _state(_sub(a1, C49 + 600, C49), _sub(b, C49 + 2 * H + 600, C49 + 2 * H))
    assert _paid(_snap(now, a2, b), now, tmp_path / IW.FIRST_FILE, st) == [1]
    assert _paid(_snap(now, a2, b), now, tmp_path / "other" / IW.FIRST_FILE, ST.SpecState()) == [1]  # memory now holds A


def test_F2_a_copy_seed_never_beats_a_truly_earlier_original(hub, tmp_path):
    a = _n(1, "ckA", C49)
    b = _copy(2, "ckB", C49 + 60, a)                  # 5 blocks later; both seen in the same worker poll
    now = C49 + 3 * D
    st = _state(_sub(b, C49 + 600, C49 + 60), _sub(a, C49 + 600, None))   # A's chain block was never captured
    a_live = dataclasses.replace(a, commitment_block=_blk(C49))
    got = _paid(_snap(now, a_live, b), now, tmp_path / IW.FIRST_FILE, st)
    assert got == [1]


def test_F3_seed_older_than_keep_days_is_ignored_and_does_not_churn(hub, tmp_path, monkeypatch):
    f = tmp_path / IW.FIRST_FILE
    a = _n(1, "ckA", C49)
    st = _state(_sub(a, C49, C49))
    now = C49 + 40 * D
    other = _n(9, "ck9", now - H)
    _paid(_snap(now, other), now, f, st)
    assert not any(k.startswith("m/m1@") for k in json.loads(f.read_text())["entries"])
    writes = []
    monkeypatch.setattr(IW.os, "replace", lambda *a: writes.append(a))
    _paid(_snap(now + 600, other), now + 600, f, st)
    assert writes == []


def test_F4_huge_integer_is_an_invalid_entry_not_an_exception(hub, tmp_path):
    f = tmp_path / IW.FIRST_FILE
    f.write_text('{"version": 1, "entries": {"x@y": [1' + "0" * 400 + ', 1, "hk"], "a@b": [5.0, 1, "hk"]}}')
    assert IW.first_seen(str(f)) == {"a@b": (5.0, 1, "hk")}


def test_F5_entry_cap_keeps_live_and_oldest_and_the_file_stays_readable(hub, tmp_path):
    f = tmp_path / IW.FIRST_FILE
    for i in range(24000):
        IW._FIRST[f"org{i:05d}/{'m' * 60}@{i:040x}"] = (C49 + i, i % 256, "5" + "H" * 47)
    live = {f"org23999/{'m' * 60}@{23999:040x}"}
    IW.save_first_seen(str(f), C49 + D, live, True)
    assert len(IW._FIRST) == IW._FIRST_MAX_ENTRIES and live <= set(IW._FIRST) and f"org00000/{'m' * 60}@{0:040x}" in IW._FIRST
    assert f.stat().st_size < IW._FIRST_MAX_BYTES
    IW.reset_first_seen()
    assert len(IW.first_seen(str(f))) == IW._FIRST_MAX_ENTRIES
