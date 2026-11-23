"""GR2_INTAKE_WAIT_KA v2.1 (critic S-A, S-B of the v2 review): the earliest commitment per repo@revision is persisted
(``ka_intake_wait.first_seen`` / ``save_first_seen``, ``intake-wait-first.json`` next to the validator state), so a restart, a
recommit or an element update of the original does not hand "first commitment" to a copy, and a first holder that left still
blocks copies (for ``FIRST_KEEP_DAYS``)."""
import dataclasses
import json
import os
import subprocess
import sys
import textwrap

from reax_subnet import competition_split as CS
from reax_subnet import ka_intake_wait as IW
from reax_subnet.spec import evaluator as E, state as ST
from reax_subnet.spec.commitment import SpecCommitment
from tests.test_ka_intake_wait import C49, D, H, T0, _n, _snap, hub  # noqa: F401  (hub = autouse fixture)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _copy(uid, ck, ts, of):
    n = _n(uid, ck, ts)
    c = E.parse_commitment(of.commitment)
    return dataclasses.replace(n, commitment=SpecCommitment("generalist", c.repo, c.revision).text())


def _paid(snap, now, path):
    return sorted(E._committed_waiting(ST.SpecState(), ["generalist"], snap, now, str(path))["generalist"])


def _restart():
    """A new process: nothing in memory, the file is read again."""
    IW.reset_first_seen()
    E._IW_TIMES.clear()
    IW.reset()


def test_restart_keeps_the_original_after_its_block_moved(hub, tmp_path):
    f = tmp_path / IW.FIRST_FILE
    a1 = _n(1, "ckA", C49)
    b = _copy(2, "ckB", C49 + 2 * H, a1)
    now = C49 + 3 * H
    _paid(_snap(now, a1, b), now, f)
    rec = json.loads(f.read_text())
    assert rec["version"] == 1 and list(rec["entries"].values()) == [[C49, 1, a1.hotkey]]
    a2 = dataclasses.replace(a1, commitment_block=_snap(C49 + 4 * H).block)      # recommit / other element updated
    _restart()
    now = C49 + 3 * D
    assert _paid(_snap(now, a2, b), now, f) == [1]      # v2: [2]


def test_restart_in_a_new_process_with_the_same_file_keeps_the_original(tmp_path):
    # the brief's restart test, with real processes: process 1 sees A and B; A's block moves; process 2 (fresh interpreter,
    # same file) still pays A only. Also the critic's P1b with the history a previous process left in the file.
    f = tmp_path / IW.FIRST_FILE
    code = textwrap.dedent("""
        import dataclasses, sys
        from reax_subnet import competition_split as CS, ka_intake_wait as IW
        from reax_subnet.spec import evaluator as E, state as ST
        from tests.test_ka_intake_wait import C49, D, H, Hub, _n, _snap
        from tests.test_ka_intake_wait_v21 import _copy
        for k in ("WAITING_KA_HOTFIX", "KA_BIG_UNDECIDED", "KA_GAUNTLET_R2", "KA_EXPOSED_FIRST", "KA_PROV_TWO_PER_COLDKEY",
                  "KA_ABUSE_CLAUSE", "GR2_INTAKE_WAIT_KA"):
            setattr(CS, k, True)
        IW.hub_check = Hub()
        a1 = _n(1, "ckA", C49)
        b = _copy(2, "ckB", C49 + 2 * H, a1)
        if sys.argv[2] == "1":
            now = C49 + 3 * H
            snap = _snap(now, a1, b)
        else:
            now = C49 + 3 * D
            snap = _snap(now, dataclasses.replace(a1, commitment_block=_snap(C49 + 4 * H).block), b)
        print(sorted(E._committed_waiting(ST.SpecState(), ["generalist"], snap, now, sys.argv[1])["generalist"]))
    """)
    env = {k: v for k, v in os.environ.items() if k != "REAX_KA_FEED_PATH"}
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), ROOT])
    run = lambda step: subprocess.run([sys.executable, "-c", code, str(f), step], check=True, timeout=60, capture_output=True,
                                      text=True, env=env, cwd=ROOT).stdout.strip()
    assert run("1") == "[1, 2]"                          # B still inside its own 6 h unseen grace (old rules)
    assert f.exists()
    assert run("2") == "[1]"                             # new process, same file: A keeps first place (v2: [2])


def test_corrupt_or_invalid_file_starts_empty_with_a_warning_and_is_replaced(hub, tmp_path, caplog):
    f = tmp_path / IW.FIRST_FILE
    a = _n(1, "ckA", C49)
    b = _copy(2, "ckB", C49 + 2 * H, a)
    now = C49 + 3 * D
    for bad in ("{not json", json.dumps({"version": 2, "entries": {}}), json.dumps([1, 2]), "[" * 200000, "\udcff"):
        f.write_text(bad, errors="surrogateescape")
        _restart()
        caplog.clear()
        assert _paid(_snap(now, a, b), now, f) == [1]       # empty start: the live chain decides as in v2
        assert any("first-commitment file" in r.message and r.levelname == "WARNING" for r in caplog.records)
        assert json.loads(f.read_text())["version"] == 1    # replaced by a valid file
    # invalid entries are skipped, valid ones kept
    k = f"m/m1@{a.commitment.rsplit('@', 1)[1]}"
    f.write_text(json.dumps({"version": 1, "entries": {"x@y": [True, 1, "h"], "z": [1.0, 1, "h"], k: [C49 - 60, 7, "5Old"]}}))
    _restart()
    assert _paid(_snap(now, a, b), now, f) == []            # the remembered holder 5Old (not live) blocks both
    assert IW.first_seen(str(f)) == {k: (C49 - 60, 7, "5Old")}   # b names the same repo@revision as a: one key


def test_missing_file_and_unwritable_directory_never_raise(hub, tmp_path):
    a = _n(1, "ckA", C49)
    now = C49 + 3 * D
    assert _paid(_snap(now, a), now, tmp_path / "new" / IW.FIRST_FILE) == [1]       # directory created
    assert (tmp_path / "new" / IW.FIRST_FILE).exists()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    _restart()
    assert _paid(_snap(now, a), now, blocker / IW.FIRST_FILE) == [1]                 # cannot write: memory only, same answer


def test_repo_spelling_is_one_key_in_the_file_and_in_the_window_memory(hub, tmp_path):
    f = tmp_path / IW.FIRST_FILE
    a = _n(1, "ckA", C49, repo="Org/Model")
    now = C49 + 3 * H
    assert _paid(_snap(now, a), now, f) == [1]
    c = E.parse_commitment(a.commitment)
    k = f"org/model@{c.revision}"
    assert list(json.loads(f.read_text())["entries"]) == [k] and list(E._IW_TIMES[a.hotkey]) == [k]   # critic N-c
    b = _copy(2, "ckB", C49 + H, dataclasses.replace(a, commitment=SpecCommitment("generalist", "org/MODEL", c.revision).text()))
    a2 = dataclasses.replace(a, commitment=SpecCommitment("generalist", "ORG/model", c.revision).text(),
                             commitment_block=_snap(C49 + 4 * H).block)
    _restart()
    now = C49 + 3 * D
    assert _paid(_snap(now, a2, b), now, f) == [1]


def test_entries_are_pruned_30_days_after_unless_still_live(hub, tmp_path):
    f = tmp_path / IW.FIRST_FILE
    a = _n(1, "ckA", C49)
    old = _n(3, "ck3", C49)
    now = C49 + 3 * H
    _paid(_snap(now, a, old), now, f)
    assert len(json.loads(f.read_text())["entries"]) == 2
    now = C49 + 31 * D                                   # uid 3 left long ago, A is still live with the same commitment
    _paid(_snap(now, a), now, f)
    keys = list(json.loads(f.read_text())["entries"])
    assert keys == [f"m/m1@{a.commitment.rsplit('@', 1)[1]}"]
    assert IW.FIRST_KEEP_DAYS == 30.0


def test_file_is_written_only_when_something_changed(hub, tmp_path):
    f = tmp_path / IW.FIRST_FILE
    a = _n(1, "ckA", C49)
    now = C49 + 3 * H
    _paid(_snap(now, a), now, f)
    os.utime(f, (1, 1))
    for i in range(1, 4):
        _paid(_snap(now + i * 12, a), now + i * 12, f)  # same block offset: same time, nothing new
    assert os.stat(f).st_mtime == 1


def test_flag_off_and_feed_mode_never_touch_the_file(hub, tmp_path, monkeypatch):
    f = tmp_path / IW.FIRST_FILE
    a = _n(1, "ckA", C49)
    now = C49 + 3 * H
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", False)
    _paid(_snap(now, a), now, f)
    assert not f.exists() and IW._FIRST == {} and hub.calls == []
    monkeypatch.setattr(CS, "GR2_INTAKE_WAIT_KA", True)
    monkeypatch.setattr(E._ka_feed, "active", lambda now: True)
    _paid(_snap(now, a), now, f)
    assert not f.exists() and IW._FIRST == {} and hub.calls == []


def test_remote_evaluator_passes_its_file_through(hub, tmp_path):
    from reax_subnet.remote import RemoteSpecEvaluator
    f = tmp_path / "validator" / IW.FIRST_FILE
    r = RemoteSpecEvaluator(str(tmp_path / "ro" / "spec-state.json"), iw_path=str(f), clock=lambda: C49 + 3 * H)
    r._state = ST.SpecState()
    a = _n(1, "ckA", C49)
    assert r.provisional(r.cfg.active_tracks(C49 + 3 * H), _snap(C49 + 3 * H, a)).get("generalist") == [1]
    assert f.exists()
