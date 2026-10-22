"""Engine copy line (gate ``engcomp.ENGINE_COPYLINE_START_UTC``; docs/ENGINE-COMPETITION.md, "Reusing other entries").

Rule points: (1) ideas are free = no effect; (2) declared reuse -> derived, must beat every parent by the full margin;
(3) closed binaries are treated like source but stay award-gated; (4) undeclared reuse found by review = copy, strike, the
second pauses the hotkey 7 days; (5) same owner reuses freely. Each behaviour is checked off (None), before, after, and at
the boundary instant."""

import asyncio
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from reax_subnet import engcomp as EC
from reax_subnet import engcopyline as CL
from reax_subnet import engreview
from reax_subnet import engvalidator as EV
from reax_subnet.engcomp import Champion, EvalResult, TrackState
from test_engcomp import P
from test_engfair import DA, DB, DC, REF, Gate, _ok, _res, _snap, _sub

START = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
SEC = timedelta(seconds=1)
D1, D2 = "1" * 64, "2" * 64
MARGIN = max(P.margin, EC.required_engine_margin(P.reference_noise_std))


def _gate(monkeypatch, value):
    monkeypatch.setattr(EC, "ENGINE_COPYLINE_START_UTC", value)


# ------------------------------------------------------------------------------------------------ gate
@pytest.mark.parametrize("start,now,expected", [
    (None, START, False),
    (START, START - SEC, False),
    (START, START, True),            # the boundary instant counts
    (START, START + SEC, True),
])
def test_gate(monkeypatch, start, now, expected):
    _gate(monkeypatch, start)
    assert EC.copyline_active(now) is expected


def test_release_constant_is_t1():
    assert EC.ENGINE_COPYLINE_START_UTC == datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)  # rc11


# ------------------------------------------------------------------------------------------------ label parsing (point 2)
def _label(*items):
    return json.dumps([{"uid": u, "digest": d, "files": f} for u, d, f in items])


def test_parse_parents_ok():
    parents, problems = CL.parse_parents(_label((4, D1, ["/app/b.py", "/app/a.so", "/app/a.so"])))
    assert problems == [] and parents == [CL.Parent(4, D1, ("/app/a.so", "/app/b.py"))]


@pytest.mark.parametrize("bad", ["not json", "{}", '[{"uid": 4}]', _label((4, "xx", ["/a"])), _label((4, D1, [])),
                                 _label((4, D1, ["rel/path"])), _label((4, D1, ["/a/../b"])),
                                 json.dumps([{"uid": True, "digest": D1, "files": ["/a"]}])])
def test_parse_parents_rejects_malformed(bad):
    parents, problems = CL.parse_parents(bad)
    assert parents == [] and problems


def test_parse_parents_absent_is_no_declaration():
    assert CL.parse_parents(None) == ([], []) and CL.parse_parents("") == ([], [])


def test_parse_parents_caps():
    items = [(i, f"{i:064x}", ["/a"]) for i in range(CL.MAX_PARENTS + 2)]
    parents, problems = CL.parse_parents(_label(*items))
    assert len(parents) == CL.MAX_PARENTS and problems


# ------------------------------------------------------------------------------------------------ verdicts (points 2, 3, 5)
def test_verdict_independent_when_nothing_shared():
    assert CL.verdict([], {D1: []}) == (CL.VERDICT_INDEPENDENT, {D1: CL.VERDICT_INDEPENDENT})


def test_verdict_declared_reuse_is_derived_source_and_binary():
    v, per = CL.verdict([D1], {D1: ["/app/a.py", "/app/lib.so"]}, declared_files={D1: ["/app/a.py", "/app/lib.so"]})
    assert v == CL.VERDICT_DERIVED and per[D1] == CL.VERDICT_DERIVED  # binaries are treated like source


def test_verdict_undeclared_is_copy_and_beats_derived():
    assert CL.verdict([], {D1: ["/app/lib.so"]})[0] == CL.VERDICT_COPY
    v, per = CL.verdict([D1], {D1: ["/a.py"], D2: ["/b.so"]}, declared_files={D1: ["/a.py"]})
    assert v == CL.VERDICT_COPY and per == {D1: CL.VERDICT_DERIVED, D2: CL.VERDICT_COPY}


def test_verdict_declared_parent_but_extra_files_is_copy():
    assert CL.verdict([D1], {D1: ["/a.py", "/b.so"]}, declared_files={D1: ["/a.py"]})[0] == CL.VERDICT_COPY


def test_verdict_same_owner_reuses_freely():
    assert CL.verdict([], {D1: ["/a.so"]}, same_owner=[D1]) == (CL.VERDICT_SAME_OWNER, {D1: CL.VERDICT_SAME_OWNER})


def test_constraint_parents_leave_out_same_owner():
    assert CL.constraint_parents([D2, D1], [D1], [D2]) == [D1]


def test_closed_components_by_missing_source():
    files = ["/app/engine.py", "/app/fast.so", "/app/fast.cu", "/app/closed.so", "/app/k.cubin", "/app/README.md"]
    assert CL.closed_components(files) == ["/app/closed.so", "/app/k.cubin"]


# ------------------------------------------------------------------------------------------------ same owner
def test_same_coldkey():
    ck = {"hk1": "ck", "hk2": "ck", "hk3": "other"}
    assert CL.same_coldkey("hk1", "hk2", ck) and not CL.same_coldkey("hk1", "hk3", ck)
    assert not CL.same_coldkey("hk1", "unknown", ck) and not CL.same_coldkey("x", "y", {})


def test_ownership_proof_binds_parent_digest_and_child_hotkey():
    msg = CL.ownership_message("parent_hk", "child_hk", D1)
    assert msg == f"reax-eng/1 same-owner:parent_hk>child_hk:{D1}"
    seen = []

    def verify(addr, message, sig):
        seen.append((addr, message))
        return sig == "good"
    label = json.dumps([{"parent_digest": D1, "parent_hotkey": "parent_hk", "signature": "good"}])
    assert CL.valid_ownership_proof(label, D1, "parent_hk", "child_hk", verify)
    assert seen == [("parent_hk", msg)]
    bad = json.dumps([{"parent_digest": D1, "parent_hotkey": "parent_hk", "signature": "forged"}])
    assert not CL.valid_ownership_proof(bad, D1, "parent_hk", "child_hk", verify)
    assert not CL.valid_ownership_proof(label, D2, "parent_hk", "child_hk", verify)      # another parent digest
    assert not CL.valid_ownership_proof(label, D1, "parent_hk", "child_hk", None)        # no chain library: no proof
    assert not CL.valid_ownership_proof("junk", D1, "parent_hk", "child_hk", verify)


# ------------------------------------------------------------------------------------------------ margin vs parents (point 2)
def _state(parent_score, child_score, champion=None, parent_ok=True):
    st = TrackState()
    st.results[D1] = EvalResult(D1, "s1-fast", parent_ok, score=parent_score, block=100, session="p")
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=child_score, block=200, session="c")
    st.first_seen = {D1: (100, "pa"), D2: (200, "ch")}
    st.champion = champion
    return st


SUBS = [_sub(1, "pa", D1, 100), _sub(2, "ch", D2, 200)]


def _two_sessions(st, copyline):
    EC.update_champion(st, SUBS, P, 3000, copyline=copyline)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=st.results[D2].score, block=300, session="c2")
    return EC.update_champion(st, SUBS, P, 3100, copyline=copyline)


def test_parent_bar_values():
    st = _state(1.5, 1.6)
    assert EC.parent_bar(st, D2, MARGIN) == 0.0                  # no parents
    st.derived[D2] = [D1]
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(1.5 * (1 + MARGIN))
    st.hold(st.results[D1])
    st.results[D1] = EvalResult(D1, "s1-fast", False, score=9.0, block=100)
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(1.5 * (1 + MARGIN))   # a failed re-run does not lower the bar
    st.held_scores[D1] = 1.7                                     # nor does a lower later score (held survives)
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(1.7 * (1 + MARGIN))


def test_derived_entry_must_beat_parent_by_full_margin():
    # no champion: the child clears the reference bar but not parent x (1 + margin)
    st = _state(1.5, 1.5 * (1 + MARGIN) - 0.001)
    st.derived[D2] = [D1]
    ch = _two_sessions(st, True)
    assert ch is None or ch.digest != D2


def test_derived_entry_crowned_when_it_beats_parent_by_margin():
    st = _state(1.5, 1.5 * (1 + MARGIN) + 0.01)
    st.derived[D2] = [D1]
    ch = _two_sessions(st, True)
    assert ch is not None and ch.digest == D2


def test_parent_rule_ignored_before_gate():
    # same numbers, copyline False (gate off / before the date): the derived record changes nothing
    st = _state(1.5, 1.5 * (1 + MARGIN) - 0.001)
    st.derived[D2] = [D1]
    off = _two_sessions(st, False)
    st2 = _state(1.5, 1.5 * (1 + MARGIN) - 0.001)
    ref = _two_sessions(st2, False)
    assert (off.digest if off else None) == (ref.digest if ref else None)


def test_confirmation_must_clear_parent_bar_in_every_session():
    st = _state(1.5, 1.5 * (1 + MARGIN) + 0.01)
    st.derived[D2] = [D1]
    EC.update_champion(st, SUBS, P, 3000, copyline=True)
    assert st.challenger is not None and st.challenger["digest"] == D2
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=1.5 * (1 + MARGIN) - 0.001, block=300, session="c2")
    ch = EC.update_champion(st, SUBS, P, 3100, copyline=True)
    assert ch is None or ch.digest != D2
    assert st.challenger is None or st.challenger["digest"] != D2    # the parent may be the challenger, the child no longer


# ------------------------------------------------------------------------------------------------ strikes and pause (point 4)
def test_first_strike_no_pause_second_pauses_seven_days():
    st = TrackState()
    assert st.record_copy("hk", D1, 1000) == 1
    assert not st.copy_paused("hk", 1000) and st.copy_paused_until == {}
    assert st.record_copy("hk", D1, 1001) == 1                   # same digest: counted once
    assert st.record_copy("hk", D2, 2000) == 2
    until = 2000 + EC.ENGINE_COPYLINE_PAUSE_BLOCKS
    assert EC.ENGINE_COPYLINE_PAUSE_BLOCKS == 50400 and st.copy_paused_until == {"hk": until}
    assert st.copy_paused("hk", until - 1) and not st.copy_paused("hk", until)   # boundary block
    assert not st.copy_paused("other", 2001)
    assert st.record_copy("hk", DC, 3000) == 3                   # every later one pauses again (F6)
    assert st.copy_paused_until == {"hk": 3000 + EC.ENGINE_COPYLINE_PAUSE_BLOCKS}
    st.copy_paused_until["hk"] = 10 ** 9                         # never shortens a longer running pause
    st.record_copy("hk", "9" * 64, 3100)
    assert st.copy_paused_until == {"hk": 10 ** 9}


def test_state_roundtrip_add_only():
    st = TrackState()
    assert not {"copy_strikes", "copy_paused_until", "derived", "derived_hk", "held_scores"} & set(st.to_json())   # absent while empty: rollback-safe
    st.record_copy("hk", D1, 10)
    st.record_copy("hk", D2, 20)
    st.derived[DC] = [D1]
    st.derived_hk[DC] = {D1: ["pa"]}
    st.hold(EvalResult(D1, "s1-fast", True, score=1.5, block=1))
    back = TrackState.from_json(json.loads(json.dumps(st.to_json())))
    assert back.copy_strikes == {"hk": [D1, D2]} and back.copy_paused_until == st.copy_paused_until
    assert back.derived == {DC: [D1]} and back.derived_hk == {DC: {D1: ["pa"]}} and back.held_scores == {D1: 1.5}
    old = TrackState.from_json({"results": {}})                   # a file from the live build
    assert old.copy_strikes == {} and old.derived == {}


# ------------------------------------------------------------------------------------------------ validator wiring
class FakeRunner:
    binds_hotkey = False

    def __init__(self, review_dir):
        self.review_dir = review_dir


def _comp(tmp_path, runner=None, now=START + SEC):
    comp = EV.EngineCompetition(runner or FakeRunner(str(tmp_path / "rev")), {"s1-fast": REF}, str(tmp_path / "s.json"),
                                tracks={"s1-fast": P}, start_utc=None, utcnow=lambda: now)
    return comp


def _mark_copy(tmp_path, digest):
    os.makedirs(tmp_path / "rev" / "COPY", exist_ok=True)
    (tmp_path / "rev" / "COPY" / digest).write_text("")


def test_copy_marker_is_regular_file_only(tmp_path):
    rd = str(tmp_path / "rev")
    assert not engreview.copy_marker(rd, D1)
    _mark_copy(tmp_path, D1)
    assert engreview.copy_marker(rd, D1) and not engreview.copy_marker(rd, D2) and not engreview.copy_marker(rd, "bad")
    os.symlink("/etc/passwd", tmp_path / "rev" / "COPY" / D2)
    assert not engreview.copy_marker(rd, D2)


@pytest.mark.parametrize("start,now,counts", [
    (None, START, False), (START, START - SEC, False), (START, START, True), (START, START + SEC, True)])
def test_copy_marker_effect_follows_gate(monkeypatch, tmp_path, start, now, counts):
    _gate(monkeypatch, start)
    _mark_copy(tmp_path, D1)
    comp = _comp(tmp_path, now=now)
    st = comp.states["s1-fast"]
    res = asyncio.run(comp._evaluate_one(st, "s1-fast", P, _sub(1, "hk", D1, 100), 500))
    if counts:
        assert isinstance(res, EvalResult) and not res.ok and res.reason.startswith("copy:")
        assert st.copy_strikes == {"hk": [D1]}
    else:
        assert st.copy_strikes == {} and not (isinstance(res, EvalResult) and res.reason.startswith("copy:"))


def test_second_copy_pauses_and_paused_hotkey_is_not_evaluated(monkeypatch, tmp_path):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    for d in (D1, D2):
        _mark_copy(tmp_path, d)
        asyncio.run(comp._evaluate_one(st, "s1-fast", P, _sub(1, "hk", d, 100), 500))
    assert st.copy_paused("hk", 501)
    tried = []

    async def fake_eval(st_, track, params, sub, block):
        tried.append(sub.uid)
        return None
    comp._evaluate_one = fake_eval
    subs = [_sub(1, "hk", DA, 100), _sub(2, "other", DB, 100)]
    asyncio.run(comp.run_cycle(_snap(subs, block=1000), set()))
    assert tried == [2]                                         # uid 1's hotkey is paused
    tried.clear()
    asyncio.run(comp.run_cycle(_snap(subs, block=500 + EC.ENGINE_COPYLINE_PAUSE_BLOCKS), set()))
    assert sorted(tried) == [1, 2]                              # the pause ended at its boundary block


def test_pause_not_applied_when_gate_off(monkeypatch, tmp_path):
    _gate(monkeypatch, None)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.copy_paused_until["hk"] = 10 ** 9                        # even a leftover record changes nothing while off
    tried = []

    async def fake_eval(st_, track, params, sub, block):
        tried.append(sub.uid)
    comp._evaluate_one = fake_eval
    asyncio.run(comp.run_cycle(_snap([_sub(1, "hk", DA, 100)], block=1000), set()))
    assert tried == [1]


def _record(tmp_path, digest, parents_label, ownership_label=None):
    rec = engreview.new_record("img@sha256:" + digest, digest, [], [], {}, {})
    rec["copyline"] = 1
    if parents_label is not None:
        rec["parents_label"] = parents_label
    if ownership_label:
        rec["ownership_label"] = ownership_label
    engreview.write_record(str(tmp_path / "rev"), digest, rec)


def test_refresh_derived_declared_parent_and_same_coldkey(tmp_path):
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.first_seen = {D1: (100, "pa"), D2: (200, "ch"), DC: (300, "pb")}
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"]), (3, DC, ["/app/c.so"])))
    subs = [_sub(1, "pa", D1, 100), _sub(2, "ch", D2, 200), _sub(3, "pb", DC, 300)]
    snap = _snap(subs, coldkeys={1: "ckA", 2: "ckB", 3: "ckB"})   # uid 3 belongs to the child's owner
    comp._refresh_derived(st, subs, snap)
    assert st.derived == {D2: [D1]}                              # DC is the same owner: not a constraint
    snap = _snap(subs, coldkeys={1: "ckB", 2: "ckB", 3: "ckB"})
    comp._refresh_derived(st, subs, snap)
    assert st.derived == {D2: []}                                # all parents are the same owner: read, no constraint


def test_refresh_derived_signed_proof(tmp_path):
    comp = _comp(tmp_path)
    comp._verify_ownership = lambda addr, msg, sig: addr == "pa" and msg == CL.ownership_message("pa", "ch", D1)
    st = comp.states["s1-fast"]
    st.first_seen = {D1: (100, "pa")}
    proof = json.dumps([{"parent_digest": D1, "parent_hotkey": "pa", "signature": "00"}])
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])), proof)
    subs = [_sub(1, "pa", D1, 100), _sub(2, "ch", D2, 200)]
    comp._refresh_derived(st, subs, _snap(subs, coldkeys={1: "ckA", 2: "ckB"}))
    assert st.derived == {D2: []}


def test_review_record_keeps_labels_complete():
    long = _label(*[(i, f"{i:064x}", ["/app/" + "x" * 150 + ".py"]) for i in range(3)])
    info = {"Config": {"Labels": {CL.PARENTS_LABEL: long, "other": "v"}}}
    assert engreview.copyline_fields(info) == {"parents_label": long}
    assert engreview.copyline_fields({"Config": {"Labels": {"other": "v"}}}) == {}
    assert engreview.copyline_fields({}) == {}


# ------------------------------------------------------------------------------------------------ operator tool
def _tool():
    spec = importlib.util.spec_from_file_location(
        "engine_parents_check", os.path.join(os.path.dirname(__file__), "..", "tools", "engine_parents_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(root, files):
    for rel, data in files.items():
        p = root / rel.lstrip("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _trees(tmp_path, child, parent, base=None):
    for name, files in (("child", child), ("parent", parent), ("base", base or {})):
        (tmp_path / name).mkdir()
        _write(tmp_path / name, files)
    return [str(tmp_path / n) for n in ("child", "parent", "base")]


def test_tool_declared_reuse_is_derived(tmp_path, capsys):
    c, p, b = _trees(tmp_path, {"/app/e.py": b"mine", "/app/lib.so": b"\x7fELFx", "/app/n.txt": b"n"},
                     {"/x/lib2.so": b"\x7fELFx", "/y.py": b"other"})
    label = _label((1, D1, ["/app/lib.so"]))
    rc = _tool().main([c, "--parent", f"{D1}={p}", "--declared", label, "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["verdict"] == CL.VERDICT_DERIVED
    assert rep["parents"][D1]["shared_binaries"] == ["/app/lib.so"] and rep["must_beat"] == [D1]
    assert rep["closed_components"] == ["/app/lib.so"]            # no source of that stem: still award-gated


def test_tool_undeclared_reuse_is_copy_exit_2(tmp_path, capsys):
    c, p, b = _trees(tmp_path, {"/app/lib.so": b"\x7fELFx"}, {"/z/lib.so": b"\x7fELFx"})
    rc = _tool().main([c, "--parent", f"{D1}={p}"])
    assert rc == 2 and "copy (undeclared)" in capsys.readouterr().out


def test_tool_same_owner_and_independent_and_ignore_tree(tmp_path, capsys):
    c, p, b = _trees(tmp_path, {"/app/lib.so": b"\x7fELFx", "/app/base.py": b"common"},
                     {"/z/lib.so": b"\x7fELFx", "/z/base.py": b"common"}, {"/b/base.py": b"common"})
    t = _tool()
    assert t.main([c, "--parent", f"{D1}={p}", "--same-owner", D1, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == CL.VERDICT_SAME_OWNER
    # only the shared base file: ignored through --ignore-tree -> independent
    (tmp_path / "child" / "app" / "lib.so").write_bytes(b"different")
    assert t.main([c, "--parent", f"{D1}={p}", "--ignore-tree", b, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == CL.VERDICT_INDEPENDENT


def test_tool_partial_declaration_is_copy(tmp_path, capsys):
    c, p, b = _trees(tmp_path, {"/app/a.py": b"A", "/app/b.so": b"B"}, {"/q/a.py": b"A", "/q/b.so": b"B"})
    rc = _tool().main([c, "--parent", f"{D1}={p}", "--declared", _label((1, D1, ["/app/a.py"])), "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 2 and rep["parents"][D1]["undeclared_files"] == ["/app/b.so"]


# the case where the rule bites: the parent is a former entry (no longer committed), the champion is weaker than the parent
def _former_parent_state(child_score):
    st = TrackState()
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=100, session="p")     # parent, withdrawn
    st.results[DA] = EvalResult(DA, "s1-fast", True, score=1.0, block=100, session="a")
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=child_score, block=200, session="c")
    st.first_seen = {D1: (100, "pa"), DA: (100, "ha"), D2: (200, "ch")}
    st.champion = Champion(DA, "ha", 9, 1.0, 100)
    st.derived[D2] = [D1]
    return st


SUBS_FP = [_sub(9, "ha", DA, 100), _sub(2, "ch", D2, 200)]


@pytest.mark.parametrize("copyline,crowned", [(False, True), (True, False)])
def test_rule_bites_when_parent_scored_higher_than_champion(copyline, crowned):
    st = _former_parent_state(1.52)               # clears the champion bar (1.03) but not the parent bar (1.5 x 1.03)
    EC.update_champion(st, SUBS_FP, P, 3000, copyline=copyline)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=1.52, block=300, session="c2")
    ch = EC.update_champion(st, SUBS_FP, P, 3100, copyline=copyline)
    assert (ch.digest == D2) is crowned


def test_rule_lets_a_real_improvement_through():
    st = _former_parent_state(1.5 * (1 + MARGIN) + 0.02)
    EC.update_champion(st, SUBS_FP, P, 3000, copyline=True)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=1.5 * (1 + MARGIN) + 0.02, block=300, session="c2")
    assert EC.update_champion(st, SUBS_FP, P, 3100, copyline=True).digest == D2


# ================================================================================================ round 2 (critic F1-F12)
def _cycle(comp, subs, block=1000):
    asyncio.run(comp.run_cycle(_snap(subs, block=block), set()))


# ---- F1: hostile labels never stop the cycle
@pytest.mark.parametrize("bad", ["[" * 16000, "[" * 130000, "[" * 5 + "]" * 5, '{"a":' * 3000, '[[[[[{"uid": 1}]]]]]'])
def test_f1_deeply_nested_labels_are_rejected_not_raised(bad):
    assert CL.parse_parents(bad) == ([], ["parents label is not valid JSON"])
    assert not CL.valid_ownership_proof(bad, D1, "pa", "ch", lambda *a: True)


def test_f1_nesting_check_ignores_brackets_inside_strings():
    ok = _label((1, D1, ["/app/[[[[[[[[[[x]]]]]]]]]].py", '/app/a\\"[[[[[[.py']))
    parents, problems = CL.parse_parents(ok)
    assert problems == [] and len(parents[0].files) == 2


def test_f1_bad_record_makes_only_that_digest_not_crownable(tmp_path):
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.first_seen = {D1: (100, "pa"), D2: (200, "ch"), DC: (300, "pc")}
    _record(tmp_path, D2, "[" * 16000)
    _record(tmp_path, DC, _label((1, D1, ["/app/a.py"])))
    st.derived[D2] = ["stale"]
    subs = [_sub(1, "pa", D1, 100), _sub(2, "ch", D2, 200), _sub(3, "pc", DC, 300)]
    comp._refresh_derived(st, subs, _snap(subs))                # does not raise
    assert D2 not in st.derived and st.derived[DC] == [D1]


def test_f1_record_that_raises_is_contained(tmp_path, monkeypatch):
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
    _record(tmp_path, DC, None)
    real = CL.parse_parents
    monkeypatch.setattr(CL, "parse_parents", lambda v: real(v) if v is None else (_ for _ in ()).throw(RuntimeError("boom")))
    st.derived[D2] = ["stale"]
    subs = [_sub(2, "ch", D2, 200), _sub(3, "pc", DC, 300)]
    comp._refresh_derived(st, subs, _snap(subs))
    assert D2 not in st.derived and st.derived[DC] == []      # no label = read, no parents (parser is not even needed)


# ---- F2: records are read before any crowning
def _run(tmp_path, monkeypatch, record_before):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    # parent D1 (hotkey pa) measured earlier at 2.0, now withdrawn (pa moved on); child D2 declares it
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=2.0, block=50, session="p")
    st.first_seen[D1] = (50, "pa")
    if record_before:
        _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))

    async def fake_eval(st_, track, params, sub, block):
        if not record_before:   # the static check writes the record during the first evaluation
            _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
        return EvalResult(sub.digest, track, True, score=2.0 * (1 + MARGIN) - 0.01, block=block)
    comp._evaluate_one = fake_eval
    _cycle(comp, [_sub(2, "ch", D2, 200)])
    return st


def test_f2_control_record_exists_parent_bar_holds(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, True).champion is None


def test_f2_record_written_in_the_same_cycle_still_meets_the_parent_bar(tmp_path, monkeypatch):
    st = _run(tmp_path, monkeypatch, False)
    assert st.champion is None and st.derived == {D2: [D1]}


def test_f2_entry_without_a_read_record_is_not_crowned(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]

    async def fake_eval(st_, track, params, sub, block):
        return EvalResult(sub.digest, track, True, score=2.0, block=block)      # strong, but no review record at all
    comp._evaluate_one = fake_eval
    _cycle(comp, [_sub(2, "ch", D2, 200)])
    assert st.champion is None and D2 not in st.derived
    _record(tmp_path, D2, None)                                                  # record read: "no parents"
    _cycle(comp, [_sub(2, "ch", D2, 200)], block=1010)
    assert st.derived[D2] == [] and st.champion is not None and st.champion.digest == D2


def test_f2_record_without_copyline_marker_is_not_read(tmp_path):
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    rec = engreview.new_record("img@sha256:" + D2, D2, [], [], {}, {})           # written before the start: no labels in it
    engreview.write_record(str(tmp_path / "rev"), D2, rec)
    comp._refresh_derived(st, [_sub(2, "ch", D2, 200)], _snap([_sub(2, "ch", D2, 200)]))
    assert D2 not in st.derived


# ---- F3: COPY alone is enough, also in release mode
def test_f3_copy_marker_strikes_without_reviewed_marker(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    comp.runner.awaiting_review = lambda ref: True       # release mode: record exists, no REVIEWED marker
    st = comp.states["s1-fast"]
    _mark_copy(tmp_path, D1)
    _cycle(comp, [_sub(1, "hk", D1, 100)])
    assert st.copy_strikes == {"hk": [D1]} and not st.results[D1].ok
    _cycle(comp, [_sub(1, "hk", D1, 100)], block=1012)    # counted once
    assert st.copy_strikes == {"hk": [D1]}


def test_f3_copy_marker_ignored_before_start(tmp_path, monkeypatch):
    _gate(monkeypatch, None)
    comp = _comp(tmp_path)
    comp.runner.awaiting_review = lambda ref: True
    _mark_copy(tmp_path, D1)
    _cycle(comp, [_sub(1, "hk", D1, 100)])
    assert comp.states["s1-fast"].copy_strikes == {} and comp.states["s1-fast"].results == {}


# ---- F4: real sr25519 keys
def test_f4_real_sr25519_proof_verifies():
    sp_core = pytest.importorskip("bittensor").sp_core
    parent = sp_core.Keypair.create_from_mnemonic(sp_core.Keypair.generate_mnemonic())
    child = sp_core.Keypair.create_from_mnemonic(sp_core.Keypair.generate_mnemonic())
    verify = CL.sr25519_verifier()
    assert verify is not None
    msg = CL.ownership_message(parent.ss58_address, child.ss58_address, D1)
    sig = parent.sign(msg.encode("utf-8")).hex()
    label = json.dumps([{"parent_digest": D1, "parent_hotkey": parent.ss58_address, "signature": sig}])
    ok = lambda ph=parent.ss58_address, ch=child.ss58_address, d=D1, lab=label: CL.valid_ownership_proof(lab, d, ph, ch, verify)
    assert ok() and CL.valid_ownership_proof(label.replace(sig, "0x" + sig), D1, parent.ss58_address, child.ss58_address, verify)
    assert not ok(ch=parent.ss58_address)                    # another child hotkey
    assert not ok(d=D2) and not ok(lab=label.replace(D1, D2))   # another digest
    assert not ok(ph=child.ss58_address)                     # signed by somebody else
    assert not CL.valid_ownership_proof(label.replace(sig, "00" * 64), D1, parent.ss58_address, child.ss58_address, verify)
    assert not CL.valid_ownership_proof(label.replace(sig, "zz"), D1, parent.ss58_address, child.ss58_address, verify)


# ---- F5: the parent bar
def test_f5_failed_latest_run_does_not_remove_the_bar():
    st = EC.TrackState()
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=2.0, block=100, session="a")
    st.first_seen[D1] = (100, "pa")
    st.derived[D2] = [D1]
    st.hold(st.results[D1])
    st.results[D1] = EvalResult(D1, "s1-fast", False, score=0.0, block=200, reason="inconclusive 3 times")
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(2.0 * (1 + MARGIN))


def test_f5_parent_never_measured_holds_the_child():
    st = EC.TrackState()
    st.derived[D2] = [D1]
    assert EC.parent_bar(st, D2, MARGIN) == float("inf")
    st.results[D1] = EvalResult(D1, "s1-fast", False, score=0.0, block=100, reason="inconclusive 3 times")
    assert EC.parent_bar(st, D2, MARGIN) == float("inf")      # failed, never a valid score: still held


def test_f5_bar_is_the_best_digest_of_the_parent_hotkey():
    st = EC.TrackState()
    old, new = D1, DC
    st.first_seen = {old: (100, "pa"), new: (300, "pa"), D2: (400, "ch")}
    st.results[old] = EvalResult(old, "s1-fast", True, score=1.2, block=100, session="a")     # weak digest named by the child
    st.results[new] = EvalResult(new, "s1-fast", True, score=2.0, block=300, session="b")     # strong newer digest, same hotkey
    st.derived[D2] = [old]
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(2.0 * (1 + MARGIN))
    st2 = EC.TrackState()                                       # the declared uid's hotkey counts too (cross-check)
    st2.first_seen = {old: (100, "liar"), new: (300, "pa")}
    st2.results[old] = EvalResult(old, "s1-fast", True, score=1.2, block=100, session="a")
    st2.results[new] = EvalResult(new, "s1-fast", True, score=2.0, block=300, session="b")
    st2.derived[D2], st2.derived_hk[D2] = [old], {old: ["liar", "pa"]}
    assert EC.parent_bar(st2, D2, MARGIN) == pytest.approx(2.0 * (1 + MARGIN))


def test_f5_refresh_uses_committer_only_and_flags_a_foreign_uid(tmp_path):
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.first_seen = {D1: (100, "pa")}
    _record(tmp_path, D2, _label((7, D1, ["/app/a.py"])))
    subs = [_sub(2, "ch", D2, 200), _sub(7, "px", DC, 150)]
    comp._refresh_derived(st, subs, _snap(subs))
    assert D2 not in st.derived and D2 not in st.derived_hk      # uid 7 is not the committer of D1: label problem, fail closed
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
    subs = [_sub(1, "pa", D1, 100), _sub(2, "ch", D2, 200)]
    comp._refresh_derived(st, subs, _snap(subs))
    assert st.derived[D2] == [D1] and st.derived_hk[D2] == {D1: ["pa"]}


def test_f5_child_of_unmeasured_parent_is_held_then_crowned(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))

    async def fake_eval(st_, track, params, sub, block):
        return EvalResult(sub.digest, track, True, score=2.0, block=block)
    comp._evaluate_one = fake_eval
    _cycle(comp, [_sub(2, "ch", D2, 200)])
    assert st.champion is None                                  # parent D1 was never measured
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.0, block=50, session="p")
    st.first_seen[D1] = (50, "pa")
    _cycle(comp, [_sub(2, "ch", D2, 200)], block=1010)
    assert st.champion is not None and st.champion.digest == D2


# ---- F6/F7: pause
def test_f7_paused_hotkey_is_not_crowned_via_confirmation(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    _record(tmp_path, D2, None)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=1.5, block=200, session="a")
    st.first_seen[D2] = (200, "ch")
    st.challenger = {"digest": D2, "hotkey": "ch", "measurements": [[200, 1.5, "a"]]}
    st.copy_strikes["ch"] = ["3" * 64]
    st.record_copy("ch", "4" * 64, 900)
    assert st.copy_paused("ch", 1000)

    async def fake_eval(st_, track, params, sub, block):
        return EvalResult(sub.digest, track, True, score=1.5, block=block)
    comp._evaluate_one = fake_eval
    _cycle(comp, [_sub(2, "ch", D2, 200)])
    assert st.champion is None and st.challenger is None
    _cycle(comp, [_sub(2, "ch", D2, 200)], block=900 + EC.ENGINE_COPYLINE_PAUSE_BLOCKS)    # pause over: back in the pool
    assert st.champion is not None and st.champion.digest == D2    # confirmed in that cycle, as before the pause


def test_f7_paused_hotkey_keeps_champion_entry_and_payout_inputs(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.champion = Champion(D2, "ch", 2, 1.5, 100)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=1.5, block=100, session="a")
    st.first_seen[D2] = (100, "ch")
    st.copy_paused_until["ch"] = 10 ** 9
    subs = [_sub(2, "ch", D2, 100), _sub(3, "ch", DC, 100)]
    assert [s.digest for s in comp._unpaused(st, subs, 1000)] == [D2]            # the sitting champion stays, new entries do not


# ---- F8: a COPY marker on a measured digest
def test_f8_copy_marker_on_measured_digest_fails_and_strikes(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=100, session="a")
    st.first_seen[D1] = (100, "hk")
    st.challenger = {"digest": D1, "hotkey": "hk", "measurements": [[100, 1.5, "a"]]}
    _mark_copy(tmp_path, D1)
    _cycle(comp, [_sub(1, "hk", D1, 100)])
    assert not st.results[D1].ok and st.results[D1].reason.startswith("copy:")
    assert st.copy_strikes == {"hk": [D1]} and st.challenger is None


def test_f8_copy_marker_vacates_the_champion(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.champion = Champion(D1, "hk", 1, 1.5, 100)
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=100, session="a")
    st.first_seen[D1] = (100, "hk")
    _mark_copy(tmp_path, D1)
    _cycle(comp, [_sub(1, "hk", D1, 100)])
    assert st.champion is None and st.history[-1]["event"] == "vacated"


def test_f8_copy_marker_off_before_start_changes_nothing(tmp_path, monkeypatch):
    _gate(monkeypatch, None)
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=100, session="a")
    st.first_seen[D1] = (100, "hk")
    _mark_copy(tmp_path, D1)
    _cycle(comp, [_sub(1, "hk", D1, 100)])
    assert st.results[D1].ok and st.copy_strikes == {}


def test_f8_copies_leave_former_ranks_and_keepalive():
    st = EC.TrackState()
    st.history = [{"block": 10, "event": "crowned", "digest": D1, "hotkey": "a", "score": 1.0},
                  {"block": 20, "event": "crowned", "digest": D2, "hotkey": "b", "score": 1.1},
                  {"block": 30, "event": "crowned", "digest": DC, "hotkey": "c", "score": 1.2}]
    st.champion = Champion(DC, "c", 3, 1.2, 30)
    base = EC.former_ranks(st, "s1-fast", 40)
    assert [g["pairs"][0][0] for g in base] == [D2, D1]
    out = EC.former_ranks(st, "s1-fast", 40, frozenset({D2}))
    assert [g["pairs"][0][0] for g in out] == [D1]
    st.copy_strikes = {"b": [D2]}
    assert EC.copy_digests(st) == frozenset({D2})
    subs = [_sub(1, "a", D1, 10), _sub(2, "b", D2, 20), _sub(3, "c", DC, 30)]
    for s in subs:
        st.results[s.digest] = EvalResult(s.digest, "s1-fast", True, score=1.1, block=40, session="x")
    prot = EC.protected_uids(st, subs, P, 40, True, "s1-fast", frozenset({D2}))
    assert 2 not in prot


# ---- F9: no new fields while off; F10: tz guard
def test_f9_static_check_adds_label_fields_only_when_active(tmp_path, monkeypatch):
    from reax_subnet import engrunner as ER
    info = {"Config": {"Labels": {CL.PARENTS_LABEL: _label((1, D1, ["/app/a.py"]))}}}

    async def fake_run(args, timeout):
        return 0, json.dumps(info), ""
    monkeypatch.setattr(ER, "_run", fake_run)
    monkeypatch.setattr(ER.engreview, "check_image_config", lambda i: ([], [], {"workdir": "/app"}))

    async def files(self, ref, wd):
        return {}

    async def expected(self):
        return {}
    monkeypatch.setattr(ER.DockerRunner, "image_files", files)
    monkeypatch.setattr(ER.DockerRunner, "_expected_files", expected)
    runner = ER.DockerRunner.__new__(ER.DockerRunner)
    for start, now_active in ((None, False), (datetime(2020, 1, 1, tzinfo=timezone.utc), True)):
        _gate(monkeypatch, start)
        rec = asyncio.run(runner._static_check("img@sha256:" + D1, D1))
        assert ("parents_label" in rec) is now_active and bool(rec.get("copyline")) is now_active


def test_f10_start_constant_must_be_timezone_aware():
    src = open(EC.__file__, encoding="utf-8").read()
    line = next(l for l in src.splitlines() if l.startswith("assert ENGINE_COPYLINE_START_UTC"))
    exec(line, {"ENGINE_COPYLINE_START_UTC": None})
    exec(line, {"ENGINE_COPYLINE_START_UTC": START})
    with pytest.raises(AssertionError):
        exec(line, {"ENGINE_COPYLINE_START_UTC": datetime(2026, 10, 25, 12, 0)})


# ---- F11: file cap, directory prefixes, weights
def test_f11_many_files_and_directory_prefix():
    many = [f"/app/k/f{i}.py" for i in range(CL.MAX_FILES_PER_PARENT)]
    parents, problems = CL.parse_parents(_label((1, D1, many)))
    assert problems == [] and len(parents[0].files) == CL.MAX_FILES_PER_PARENT == 2000
    parents, problems = CL.parse_parents(_label((1, D1, many + ["/app/x.py"])))
    assert parents == [] and problems                              # over the cap: reported, never cut silently
    parents, problems = CL.parse_parents(_label((1, D1, ["/app/kernels/"])))
    assert problems == [] and parents[0].files == ("/app/kernels/",)
    assert CL.parse_parents(_label((1, D1, ["/"])))[0] == []       # the whole filesystem is no declaration
    decl = {D1: ["/app/kernels/", "/app/a.py"]}
    assert CL.verdict([D1], {D1: ["/app/kernels/x.so", "/app/kernels/sub/y.cu", "/app/a.py"]}, (), decl)[0] == CL.VERDICT_DERIVED
    assert CL.verdict([D1], {D1: ["/app/kernelsX/x.so"]}, (), decl)[0] == CL.VERDICT_COPY
    assert CL.verdict([D1], {D1: ["/app/other.so"]}, (), decl)[0] == CL.VERDICT_COPY


def test_f11_weights_and_tokenizers_are_not_closed_components():
    files = ["/app/model/pytorch_model.bin", "/app/weights/w.bin", "/app/w.safetensors", "/app/tokenizer/t.bin",
             "/app/fast.so", "/app/blob.bin"]
    assert CL.closed_components(files) == ["/app/blob.bin", "/app/fast.so"]


# ---- F12: tool
def test_f12_tool_directory_prefix_and_unverified_declaration(tmp_path, capsys):
    c, p, b = _trees(tmp_path, {"/app/k/a.so": b"A", "/app/k/b.so": b"B"}, {"/q/a.so": b"A", "/q/b.so": b"B"})
    t = _tool()
    rc = t.main([c, "--parent", f"{D1}={p}", "--declared", _label((1, D1, ["/app/k/"])), "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["verdict"] == CL.VERDICT_DERIVED and rep["parents"][D1]["undeclared_files"] == []
    # the child declares a second digest of the parent hotkey that was not given as a tree: cannot be verified
    rc = t.main([c, "--parent", f"{D1}={p}", "--declared", _label((1, D1, ["/app/k/"]), (1, DC, ["/app/k/"])), "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 3 and rep["verdict"] == "unverified declaration" and rep["unverified_declared"] == [DC]
    rc = t.main([c, "--parent", f"{D1}={p}", "--declared", _label((1, D1, ["/app/k/"]), (1, DC, ["/app/k/"]))])
    out = capsys.readouterr().out
    assert rc == 3 and "cannot verify" in out and "WITHHOLD the award approval" in out


# ================================================================================================ round 3 (critic G1, G3, G5)
D3, D4, D5 = "3" * 64, "4" * 64, "5" * 64


def _g1_setup(tmp_path):
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    st.results[D1] = EvalResult(D1, "s1-fast", True, score=1.5, block=50, session="p")   # parent pa (uid 1)
    st.first_seen[D1] = (50, "pa")
    st.results[D4] = EvalResult(D4, "s1-fast", True, score=3.0, block=60, session="s")   # strong unrelated hotkey (uid 4)
    st.first_seen[D4] = (60, "strong")
    return comp, st


def test_g1_third_party_label_cannot_change_another_childs_bar(tmp_path):
    comp, st = _g1_setup(tmp_path)
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
    subs = [_sub(1, "pa", D1, 50), _sub(4, "strong", D4, 60), _sub(2, "c", D2, 200)]
    comp._refresh_derived(st, subs, _snap(subs, block=1000))
    honest = EC.parent_bar(st, D2, MARGIN)
    assert honest == pytest.approx(1.5 * (1 + MARGIN))
    _record(tmp_path, D3, _label((4, D1, ["/app/z.py"])))        # same parent digest, uid of the strong hotkey
    subs2 = subs + [_sub(3, "x", D3, 300)]
    comp._refresh_derived(st, subs2, _snap(subs2, block=1000))
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(honest)
    assert D3 not in st.derived                                   # the foreign uid is that child's own label problem
    assert st.derived_hk == {D2: {D1: ["pa"]}}


def test_g1_uid_reregistration_cannot_ratchet_a_bar(tmp_path):
    comp, st = _g1_setup(tmp_path)
    _record(tmp_path, D2, _label((1, D1, ["/app/a.py"])))
    subs = [_sub(1, "pa", D1, 50), _sub(2, "c", D2, 200)]
    comp._refresh_derived(st, subs, _snap(subs, block=1000))
    before = EC.parent_bar(st, D2, MARGIN)
    subs_b = [_sub(1, "strong", D4, 60), _sub(2, "c", D2, 200)]   # uid 1 re-registered by 'strong'
    comp._refresh_derived(st, subs_b, _snap(subs_b, block=2000))
    assert EC.parent_bar(st, D2, MARGIN) == pytest.approx(before) and st.derived[D2] == [D1]


def test_g1_uid_hotkey_used_only_when_committer_unknown_and_fixed_at_first_reading(tmp_path):
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    _record(tmp_path, D2, _label((7, D1, ["/app/a.py"])))
    subs = [_sub(2, "ch", D2, 200), _sub(7, "px", DC, 150)]
    comp._refresh_derived(st, subs, _snap(subs))
    assert st.derived_hk == {D2: {D1: ["px"]}}
    subs = [_sub(2, "ch", D2, 200), _sub(7, "py", DC, 150)]       # uid 7 changes hands: the first reading stays
    comp._refresh_derived(st, subs, _snap(subs))
    assert st.derived_hk == {D2: {D1: ["px"]}}


# ---- G3: measured digests whose record predates the start are re-checked once, not dropped
class RecheckRunner(FakeRunner):
    def __init__(self, review_dir):
        super().__init__(review_dir)
        self.calls = []

    async def recheck_copyline(self, ref):
        self.calls.append(ref)
        rec = engreview.load_record(self.review_dir, engreview.digest_of(ref))
        if rec and not rec.get("copyline"):
            rec["copyline"] = 1
            engreview.write_record(self.review_dir, engreview.digest_of(ref), rec)


def _old_record(tmp_path, digest):
    rec = engreview.new_record("img@sha256:" + digest, digest, [], [], {}, {})   # no copyline mark: written before the start
    engreview.write_record(str(tmp_path / "rev"), digest, rec)


def test_g3_measured_old_record_is_rechecked_once_and_not_crownable_before(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    runner = RecheckRunner(str(tmp_path / "rev"))
    comp = _comp(tmp_path, runner)
    st = comp.states["s1-fast"]
    _old_record(tmp_path, D2)
    st.results[D2] = EvalResult(D2, "s1-fast", True, score=2.0, block=100, session="a")
    subs = [_sub(2, "ch", D2, 100)]
    st.first_seen[D2] = (100, "ch")
    comp._refresh_derived(st, subs, _snap(subs))
    assert D2 not in st.derived                                   # not read yet: not crownable (fail closed)
    assert EC.update_champion(st, subs, P, 1000, copyline=True) is None
    assert any(s.digest == D2 for s in subs)                      # still in the pool

    async def fake_eval(st_, track, params, sub, block):
        return EvalResult(sub.digest, track, True, score=2.0, block=block)
    comp._evaluate_one = fake_eval
    _cycle(comp, subs)
    assert len(runner.calls) == 1 and st.derived[D2] == [] and st.champion.digest == D2
    assert engreview.load_record(str(tmp_path / "rev"), D2)["copyline"] == 1   # marked: the runner's re-check is a no-op from now on


def test_g3_gate_off_does_no_recheck(tmp_path, monkeypatch):
    _gate(monkeypatch, None)
    runner = RecheckRunner(str(tmp_path / "rev"))
    comp = _comp(tmp_path, runner)
    comp.states["s1-fast"].results[D2] = EvalResult(D2, "s1-fast", True, score=2.0, block=100, session="a")
    _old_record(tmp_path, D2)
    _cycle(comp, [_sub(2, "ch", D2, 100)])
    assert runner.calls == []


def _docker_runner(monkeypatch, tmp_path, info, code=0, err="no such image"):
    from reax_subnet import engrunner as ER

    async def fake_run(args, timeout):
        return code, json.dumps(info), err if code else ""
    monkeypatch.setattr(ER, "_run", fake_run)
    runner = ER.DockerRunner.__new__(ER.DockerRunner)
    runner.review_dir = str(tmp_path / "rev")
    return runner


def test_g3_docker_recheck_reads_labels_into_the_old_record(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    label = _label((1, D1, ["/app/a.py"]))
    runner = _docker_runner(monkeypatch, tmp_path, {"Config": {"Labels": {CL.PARENTS_LABEL: label}}})
    _old_record(tmp_path, D2)
    asyncio.run(runner.recheck_copyline("img@sha256:" + D2))
    rec = engreview.load_record(str(tmp_path / "rev"), D2)
    assert rec["copyline"] == 1 and rec["parents_label"] == label and rec["status"] == "static_ok"
    asyncio.run(runner.recheck_copyline("img@sha256:" + D2))      # marked: nothing runs again
    assert engreview.load_record(str(tmp_path / "rev"), D2) == rec


def test_g3_docker_recheck_uninspectable_image_counts_as_no_parents(tmp_path, monkeypatch):
    _gate(monkeypatch, START)
    runner = _docker_runner(monkeypatch, tmp_path, {}, code=1)
    _old_record(tmp_path, D2)
    asyncio.run(runner.recheck_copyline("img@sha256:" + D2))
    rec = engreview.load_record(str(tmp_path / "rev"), D2)
    assert rec["copyline"] == 1 and "parents_label" not in rec
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    subs = [_sub(2, "ch", D2, 100)]
    comp._refresh_derived(st, subs, _snap(subs))
    assert st.derived[D2] == []                                   # crownable; a COPY marker still strikes it
    _mark_copy(tmp_path, D2)
    comp._apply_copy_markers(st, subs, 1000)
    assert st.copy_strikes == {"ch": [D2]}


@pytest.mark.parametrize("code,err", [(1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"),
                                      (-1, "timeout"), (1, "error during connect: context deadline exceeded")])
def test_h1_transient_docker_error_keeps_the_record_unread(tmp_path, monkeypatch, code, err):
    # critic H1: only a gone image counts as "no parents"; a daemon error or timeout leaves the record unmarked
    from reax_subnet.engrunner import InfraError
    _gate(monkeypatch, START)
    runner = _docker_runner(monkeypatch, tmp_path, {}, code=code, err=err)
    _old_record(tmp_path, D2)
    before = engreview.load_record(str(tmp_path / "rev"), D2)
    with pytest.raises(InfraError):
        asyncio.run(runner.recheck_copyline("img@sha256:" + D2))
    assert engreview.load_record(str(tmp_path / "rev"), D2) == before and "copyline" not in before
    comp = _comp(tmp_path)
    st = comp.states["s1-fast"]
    subs = [_sub(2, "ch", D2, 100)]
    comp._refresh_derived(st, subs, _snap(subs))
    assert D2 not in st.derived                                   # not crownable until the re-check succeeds


@pytest.mark.parametrize("pull_err,gone", [("docker pull failed: manifest for img not found: manifest unknown", True),
                                           ("docker pull failed: dial tcp: connection refused", False)])
def test_h1_gone_locally_tries_a_pinned_pull_first(tmp_path, monkeypatch, pull_err, gone):
    from reax_subnet.engrunner import InfraError
    _gate(monkeypatch, START)
    runner = _docker_runner(monkeypatch, tmp_path, {}, code=1, err="Error: No such image: img@sha256:" + D2)
    runner.pull = True
    calls = []

    async def fake_ensure(ref):
        calls.append(ref)
        raise InfraError(pull_err)
    runner._ensure_image = fake_ensure
    _old_record(tmp_path, D2)
    if gone:
        asyncio.run(runner.recheck_copyline("img@sha256:" + D2))
        assert engreview.load_record(str(tmp_path / "rev"), D2)["copyline"] == 1
    else:
        with pytest.raises(InfraError):
            asyncio.run(runner.recheck_copyline("img@sha256:" + D2))
        assert "copyline" not in engreview.load_record(str(tmp_path / "rev"), D2)
    assert calls == ["img@sha256:" + D2]


def test_h1_unreadable_inspect_output_is_not_gone(tmp_path, monkeypatch):
    from reax_subnet import engrunner as ER
    _gate(monkeypatch, START)

    async def fake_run(args, timeout):
        return 0, "not json", ""
    monkeypatch.setattr(ER, "_run", fake_run)
    runner = ER.DockerRunner.__new__(ER.DockerRunner)
    runner.review_dir = str(tmp_path / "rev")
    _old_record(tmp_path, D2)
    with pytest.raises(ER.InfraError):
        asyncio.run(runner.recheck_copyline("img@sha256:" + D2))
    assert "copyline" not in engreview.load_record(str(tmp_path / "rev"), D2)


# ---- G5: weights are matched by whole directory names
def test_g5_weight_dirs_are_whole_names():
    assert CL.is_weight_file("/app/model/x.bin") and CL.is_weight_file("/opt/Weights/sub/x.bin") and CL.is_weight_file("/a/b.safetensors")
    assert not CL.is_weight_file("/app/mymodelkernels/fast.bin") and not CL.is_weight_file("/app/models_cuda/k.bin")
    assert not CL.is_weight_file("/app/tokenizer.bin")            # a file name is not a directory name
    assert CL.closed_components(["/app/mymodelkernels/fast.bin", "/app/model/w.bin"]) == ["/app/mymodelkernels/fast.bin"]
    assert CL.needs_weights_check("/app/blob.bin") and not CL.needs_weights_check("/app/fast.so")
