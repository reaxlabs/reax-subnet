"""Property tests over generated waiting-state timelines (job keepalive-gauntlet-20261026, slice proptests; bar (b)).

What is generated (tests/ka_gauntlet_model.py): per hotkey a timeline of hourly-ish weight sets (every 3600 s or 4200 s) with
commitment -> evaluator has not seen it -> fetch fails (401 / 404 / timeout / private forever / base missing) -> repo public ->
queued for a box -> first draw (pass / owed / fail) -> pending confirmation -> confirmed, waiting for the Mon/Thu slot (also
held by the 6 h cutoff) -> void slot -> next slot -> decided (crowned, then awaiting approval 0.2 .. 100 h, or not better);
re-commit (new revision, same tensors), track switch (generalist <-> engine), re-registration of the same hotkey on a new uid,
engine digests (unbound, waiting for a measurement, pending challenger, confirmed challenger waiting for the Tue/Fri slot,
champion awaiting approval), up to 40 simultaneous waiting entries, former champions, the DF/RA legacy keep-alive and the
ka-override row. Every set goes through the REAL code (``compute_keepalive``, ``compute_provisional`` / ``_committed_waiting``,
``keepalive_protected``, ``legacy_leaders``, ``EngineCompetition.weights`` -> ``combine_weights``); only the validator's state
(what the evaluator would have written) is generated.

P1 every set from the first one after the commitment until the entry is decided gives the uid weight > 0, unless a PUBLISHED
   exclusion applies (SPEC.md of keepalive-hotfix-20261026; each is a tag, counted and printed with -s; see ``model.SPEC``).
P2 shares stay inside the caps (25 % generalist keep-alive + provisional, 5 % provisional, engine 5 % of the engine share,
   legacy 3 %), weights sum to 1, no negative / non-finite weight, the owner burn entry does not shrink.
P3 (chain) with commit-reveal modelled (a set at block b counts at the 2nd epoch boundary after b; u16 max-normalised; a set
   older than the uid's registration is masked), the uid's incentive is > 0 at every epoch boundary from commitment + 3 epochs
   until decided, outside the same exclusions.

Counterexamples were findings F1 .. F5. F1/F2/F3/F5 are fixed in the code behind ``CS.KA_GAUNTLET_R2`` (default True); F4 (the
abuse clause: a 404 earns no window) is approved policy and an exclusion of the oracle (E_ABUSE_404):
 * ``test_p1_p2_p3_with_fixes``    the properties hold on thousands of generated timelines (``model.apply_fixes`` = the switch);
 * ``test_finding_*``              one minimal reproduction each: passes with the switch on, and the old failure is shown with it off;
 * ``test_unpatched_census``       the same generator with the switch off: counts the counterexamples;
 * ``test_mutants_are_caught``     the properties fail when a published number, the override or KA_BIG_UNDECIDED is broken.
The generator includes 26-36 B entries (fetched, with and without a floor for their base, scored above and below floor + 1.5).

Run:  PYTHONPATH=src:tests <venv>/bin/python -m pytest -q -p no:cacheprovider tests/test_ka_gauntlet_props.py
Scale the number of generated worlds with KA_GAUNTLET_SCALE (default 1.0 = about 6000 timelines in the main run); write the JSON
report with KA_GAUNTLET_OUT=/path/report.json.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

import pytest
from ka_gauntlet_model import (DAY, ENG, GEN, H, INF, REV, SPEC, ST, apply_fixes, block_at, hand_gen, hand_world, pending_patch,
                               queued_patch, run_batch, ss58, ts, undo_fixes, Hotkey, Runner)

from reax_subnet import competition_split as CS
from reax_subnet import engcomp as EC
from reax_subnet import ka_override as KO
from reax_subnet.spec import evaluator as E

SCALE = float(os.environ.get("KA_GAUNTLET_SCALE", "1.0"))
ALL_FIXES = ("F1", "F2", "F3", "F5")       # = the switch KA_GAUNTLET_R2 on (the release default)


@pytest.fixture(autouse=True)
def _release_state(monkeypatch):
    """conftest clears the crown slots for the older tests; the release has them (Mon/Thu generalist, Tue/Fri engine)."""
    import conftest
    logging.disable(logging.CRITICAL)             # the share layer logs every set; restored below (the global switch is process wide)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", conftest.RELEASE_CROWN_SLOTS_UTC)
    monkeypatch.setattr(CS, "ENGINE_CROWN_SLOTS_UTC", conftest.RELEASE_ENGINE_CROWN_SLOTS_UTC)
    from reax_subnet.spec import sampling      # the release arms eval v5.1 (26 Oct 14:30Z); conftest turns the gate off for older tests
    monkeypatch.setattr(sampling, "EVAL_V5_START_UTC", conftest.RELEASE_EVAL_V5_START_UTC)
    for k in list(CS.TRACK_ENTRY_FLOOR):          # the ended DF/RA tracks' floors are not what is tested here (as test_legacyka)
        if k != GEN:
            monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, k, None)
    yield
    logging.disable(logging.NOTSET)


def _slots():
    return CS.CROWN_SLOTS_UTC, CS.ENGINE_CROWN_SLOTS_UTC


def _jobs(seeds, kw=None, fixes=()):
    return [(s, dict(kw or {}), tuple(fixes), _slots()) for s in seeds]


def _n(k):
    return max(2, int(round(k * SCALE)))


def _main_jobs(fixes):
    """The generated worlds of the main run: general mix, bursts of 40, long waits (beyond the bounds), 404s."""
    return (_jobs(range(0, _n(130)), {}, fixes)
            + _jobs(range(5000, 5000 + _n(14)), dict(burst=True, n_hot=40), fixes)
            + _jobs(range(6000, 6000 + _n(20)), dict(wait_scale=3.0), fixes)
            + _jobs(range(7000, 7000 + _n(14)), dict(allow404=True, n_hot=20), fixes))


def _show(res, limit=6):
    rows = []
    for r in res:
        for g in r["gaps"][:2]:
            rows.append(f"seed {r['seed']} uid {g[0]} {g[1]} +{g[2]}h after the commitment, state {g[3]}, weight {g[4]}")
    return "\n".join(rows[:limit])


# ------------------------------------------------------------------------------------------------ constants
def test_release_constants_match_the_spec():
    """The oracle uses the SPEC.md numbers as literals; the code must carry the same ones."""
    assert CS.WAITING_KA_HOTFIX is True
    assert CS.PROVISIONAL_KA_UNVERIFIED_HOURS == SPEC["unverified_h"] and CS.KA_REWINDOW_HOURS == SPEC["rewindow_h"]
    assert CS.PROVISIONAL_KA_UNVERIFIED_MAX == SPEC["unverified_max"] and CS.PROVISIONAL_KA_OUR_WAIT_HOURS == SPEC["our_wait_h"]
    assert CS.PENDING_KA_MAX_HOURS == SPEC["pending_cap_h"] and CS.CHAMPION_APPROVAL_KA_HOURS == SPEC["approval_h"]
    assert CS.ENGINE_KA_WAIT_HOURS == SPEC["engine_wait_h"] and CS.ENGINE_SLOT_KA_MAX_HOURS == SPEC["engine_slot_cap_h"]
    assert (CS.KEEPALIVE_MAX_SHARE, CS.PROVISIONAL_KA_MAX_SHARE, CS.ENGINE_KA_MAX_SHARE, CS.LEGACY_KA_MAX_SHARE) == \
        (SPEC["share_ka"], SPEC["share_prov"], SPEC["share_engine_ka"], SPEC["share_legacy"])
    assert CS.profile_for(CS.SPLIT_PROFILE, CS.GENERALIST_START_UTC).tracks == (0.30,)


# ------------------------------------------------------------------------------------------------ the properties, fixes applied
def test_p1_p2_p3_with_fixes():
    """P1 + P2 + P3 on thousands of generated timelines, the gauntlet-round-2 switch on."""
    res = run_batch(_main_jobs(ALL_FIXES))
    episodes = sum(r["episodes"] for r in res)
    checked = sum(r["checked"] for r in res)
    excluded = Counter()
    for r in res:
        excluded.update(r["excluded"])
    print(f"\n[P1-P3 with fixes] worlds {len(res)}, timelines (episodes) {episodes}, (episode, set) pairs checked {checked}, "
          f"max simultaneous waiting entries {max(r['max_simul'] for r in res)}, "
          f"longest keep-alive list {max(r['ka_len'] for r in res)}, min weight {min(r['min_weight'] for r in res if r['min_weight']):.3e}")
    print("[exclusions used, (episode, set) pairs] " + ", ".join(f"{k}={v}" for k, v in sorted(excluded.items())))
    if SCALE >= 1.0:
        assert episodes >= 5000
    bad = [r for r in res if r["gaps"] or r["cap"] or r["override_gaps"] or r["p3_viol"]]
    assert not bad, "\n" + _show(bad) + f"\nP2: {[r['cap'] for r in bad if r['cap']][:2]}\nP3: {[r['p3_viol'][:2] for r in bad if r['p3_viol']][:2]}"
    # the weights are not merely positive: they survive the u16 normalisation and the SPEC.md alert level of 1e-4 is cleared
    assert min(r["min_weight"] for r in res if r["min_weight"]) > 1e-4


def test_generator_covers_the_scenario_space():
    """A broken generator must not pass silently: every scenario of the brief shows up in a batch (labels of the episodes)."""
    from collections import defaultdict
    import ka_gauntlet_model as M
    seen = defaultdict(int)
    for seed in range(60):
        w = M.make_world(seed, allow404=True, n_hot=28)
        for h in w.hotkeys:
            for i, e in enumerate(h.episodes):
                seen["mode:" + e.label.split("/")[0]] += 1
                for part in e.label.split("/")[1:]:
                    seen["step:" + part] += 1
                seen["kind:" + e.kind] += 1
                seen["episodes_per_hotkey:%d" % len(h.episodes)] += 1
            if h.gaps:
                seen["reregistration"] += 1
            if len({e.kind for e in h.episodes}) == 2:
                seen["track_switch"] += 1
        seen["override"] += bool(w.override)
        seen["legacy"] += bool(w.legacy)
        seen["former_champions"] += len(w.incumbent_gen) > 1
    need = ["mode:ok", "mode:f401", "mode:f404", "mode:ftimeout", "mode:private", "mode:base_miner", "mode:base_infra", "mode:eng",
            "step:pass", "step:owed_pass", "step:owed_fail", "step:fail", "step:crown", "step:notbetter", "step:void_crown",
            "step:void_void_notbetter", "step:conf_fail", "step:low", "kind:eng", "kind:gen", "episodes_per_hotkey:2",
            "reregistration", "track_switch", "override", "legacy", "former_champions"]
    missing = [k for k in need if not seen[k]]
    assert not missing, missing


# ------------------------------------------------------------------------------------------------ census of the unpatched code
def _gapset(res):
    return {(r["seed"], g[0], g[2]) for r in res for g in r["gaps"]}


def test_unpatched_census(tmp_path):
    """The same generator with KA_GAUNTLET_R2 off (the 233dbd2 windows): reports the counterexamples (gaps by validator state) and
    the longest zero-incentive span (P3) per scenario, and then with the switch on (nothing)."""
    kinds = (_jobs(range(100, 100 + _n(8)), dict(n_hot=20)) + _jobs(range(300, 300 + _n(4)), dict(burst=True, n_hot=40))
             + _jobs(range(400, 400 + _n(6)), dict(allow404=True, n_hot=20)) + _jobs(range(500, 500 + _n(8)), dict(n_hot=12, wait_scale=1.5)))
    base = run_batch(kinds)
    with_all = run_batch([(s, kw, ALL_FIXES, sl) for s, kw, _, sl in kinds])
    assert not _gapset(with_all) and not any(r["cap"] or r["p3_viol"] for r in with_all)
    unpatched = _gapset(base)
    classes = Counter(g[3] for r in base for g in r["gaps"])
    print("\n[census, switch off] gaps:", len(unpatched), "by validator state:", dict(classes.most_common()))
    p3 = {}
    for r in base:
        for k, v in r["p3"].items():
            p3[k] = max(p3.get(k, 0), v)
    print("[P3, switch off] max zero-incentive span in blocks (360 = one epoch) by scenario, worst 12:",
          sorted(p3.items(), key=lambda kv: -kv[1])[:12])
    p3b = {}
    for r in with_all:
        for k, v in r["p3"].items():
            p3b[k] = max(p3b.get(k, 0), v)
    print("[P3, switch on] max zero-incentive span in blocks over all scenarios:", max(p3b.values(), default=0))
    out = os.environ.get("KA_GAUNTLET_OUT")
    if out:
        with open(out, "w") as f:
            json.dump({"gaps_switch_off": len(unpatched), "gap_states": dict(classes), "p3_max_span_blocks_by_scenario_off": p3,
                       "p3_max_span_blocks_by_scenario_on": p3b, "episodes": sum(r["episodes"] for r in base)}, f, indent=1, sort_keys=True)
    assert unpatched      # the switch matters: F1 alone is thousands of gaps


# ------------------------------------------------------------------------------------------------ mutants
@pytest.mark.parametrize("name,patch,expect", [
    ("hotfix switch off", lambda m: m.setattr(CS, "WAITING_KA_HOTFIX", False), "gaps"),
    ("pending cap 216 h -> 48 h", lambda m: m.setattr(CS, "PENDING_KA_MAX_HOURS", 48.0), "gaps"),
    ("champion approval 72 h -> 24 h", lambda m: m.setattr(CS, "CHAMPION_APPROVAL_KA_HOURS", 24.0), "gaps"),
    ("our-wait 96 h -> 12 h", lambda m: m.setattr(CS, "PROVISIONAL_KA_OUR_WAIT_HOURS", 12.0), "gaps"),
    ("engine wait 24 h -> 6 h", lambda m: (m.setattr(CS, "ENGINE_KA_WAIT_HOURS", 6.0), m.setattr(CS, "ENGINE_KA_MAX_HOURS", 6.0),
                                           m.setattr(CS, "ENGINE_SLOT_KA_MAX_HOURS", 6.0)), "gaps"),     # until-measured cap is the 120 h one
    ("provisional cap 5 % -> 20 %", lambda m: m.setattr(CS, "PROVISIONAL_KA_MAX_SHARE", 0.20), "cap"),
    ("engine keep-alive cap 5 % -> 30 %", lambda m: m.setattr(CS, "ENGINE_KA_MAX_SHARE", 0.30), "cap"),
    ("legacy cap 3 % -> 30 %", lambda m: m.setattr(CS, "LEGACY_KA_MAX_SHARE", 0.30), "cap"),
    ("override not applied", lambda m: m.setattr(KO, "apply", lambda rows, neurons, now: ([], [])), "override_gaps"),
    ("big undecided entries zeroed (KA_BIG_UNDECIDED off)", lambda m: m.setattr(CS, "KA_BIG_UNDECIDED", False), "gaps"),
    ("unverified cap 16 -> 2", lambda m: m.setattr(CS, "PROVISIONAL_KA_UNVERIFIED_MAX", 2), "gaps"),
])
def test_mutants_are_caught(monkeypatch, name, patch, expect):
    """Break a published number (or the override): the properties on the fixed code must fail."""
    patch(monkeypatch)
    seeds = list(range(8000, 8000 + _n(12)))
    kw = dict(n_hot=28) if expect != "override_gaps" else dict(n_hot=12, families=("override", "simple"))
    if name.startswith("legacy"):         # the DF/RA legacy keep-alive runs until 28 Oct 13:00Z only: put the world inside it
        kw = dict(n_hot=4, t0=ts(2026, 10, 26, 16, 0), n_serv=20, horizon_days=1.0)
    res = run_batch(_jobs(seeds, kw, ALL_FIXES))
    assert any(r[expect] for r in res), f"mutant not caught: {name}"


def test_generator_has_the_26_36b_class():
    """The class of gap the first generator missed: fetched 26-36 B entries, a base with and without a floor, scores around floor + 1.5."""
    import ka_gauntlet_model as M
    seen = Counter()
    for seed in range(40):
        w = M.make_world(seed, n_hot=28)
        for h in w.hotkeys:
            for e in h.episodes:
                if e.kind == "gen":
                    for part in e.label.split("/"):
                        if part.startswith("big"):
                            seen[part] += 1
                    ps = [p for _, p in e.segs if p and p.get("params")]
                    assert all(26e9 <= p["params"] <= 36e9 for p in ps)
    assert {"big-v5", "big-above", "big-below", "big-nofloor"} <= set(seen), seen


def test_big_undecided_mutant_fails_the_properties_even_with_the_gauntlet_fixes():
    """With KA_BIG_UNDECIDED off every fetched 26-36 B entry waiting for its first draw is zero: the extended generator must see it."""
    res = run_batch(_jobs(range(8100, 8100 + _n(10)), dict(n_hot=20), ALL_FIXES) and
                    [(s, kw, f, sl, {"KA_BIG_UNDECIDED": False}) for s, kw, f, sl in _jobs(range(8100, 8100 + _n(10)), dict(n_hot=20), ALL_FIXES)])
    gaps = [g for r in res for g in r["gaps"]]
    assert gaps and any("big" in g[1] for g in gaps), gaps[:3]


# ------------------------------------------------------------------------------------------------ minimal reproductions
T0 = ts(2026, 10, 27, 9, 0)          # a Tuesday; the next generalist slot is Thu 29 Oct 14:30Z


def _final(world, tmp_path):
    return Runner(world, str(tmp_path)).run()


def _weights_at(sets, t):
    return min(sets, key=lambda s: abs(s[0] - t))[2]


def _pending_world(n, fixes=()):
    hks = []
    for i in range(n):
        hk = ss58(700 + i)
        tc = T0 - 6 * H - i * 60.0
        ep = hand_gen(hk, 400 + i, f"ck{700 + i}", tc, [(tc + 0.5 * H, queued_patch(hk)), (T0 + i * 60.0, pending_patch(hk, T0 + i * 60.0))])
        h = Hotkey(hk, f"ck{700 + i}")
        h.episodes.append(ep)
        hks.append(h)
    return hand_world(hks, T0, 5.0)


def _run_with(fixes, world, tmp_path):
    undo = apply_fixes(fixes)
    try:
        return _final(world, tmp_path)
    finally:
        undo_fixes(undo)


def test_finding_f1_ninth_pending_entry_is_cut(tmp_path):
    """10 honest entries, each past its first draw (pending confirmation, distinct coldkeys): every one must be paid."""
    world = _pending_world(10)
    sets = _final(world, tmp_path)
    w = _weights_at(sets, T0 + 3 * H)
    unpaid = [400 + i for i in range(10) if not w.get(400 + i, 0.0) > 0]
    assert not unpaid, f"unpaid pending entries (uids): {unpaid}"


def test_finding_f1_switch_off_shows_the_old_cut(tmp_path):
    """KA_GAUNTLET_R2 off = 233dbd2: the 9th and 10th pending entry get 0."""
    sets = _run_with((), _pending_world(10), tmp_path)
    w = _weights_at(sets, T0 + 3 * H)
    assert [400 + i for i in range(10) if not w.get(400 + i, 0.0) > 0] == [400, 401]


def test_finding_f1_many_entries_split_evenly_inside_the_cap(tmp_path):
    """44 pending entries: all paid, the 25 % keep-alive cap holds, nobody is cut by a count anywhere."""
    sets = _final(_pending_world(44), tmp_path)
    t, b, w, kw = min(sets, key=lambda s: abs(s[0] - (T0 + 3 * H)))
    assert all(w.get(400 + i, 0.0) > 0 for i in range(44))
    assert len((kw.get("keepalive") or {}).get(GEN, ())) == 44


def _recommit_world(seen_h=2.0):
    hk = ss58(800)
    ta = T0 - 30 * H                                   # first revision, seen and fetched, past its first draw 10 h before the recommit
    fp = T0 - 10 * H
    tb = T0 + 0.3 * H                                  # the miner pushes a new revision; our evaluator has not seen it yet
    a = hand_gen(hk, 500, "ck800", ta, [(ta + 1 * H, queued_patch(hk)), (fp, pending_patch(hk, fp))], end=tb, end_kind="changed",
                 first_seen=ta + 1 * H, label="first revision")
    b = hand_gen(hk, 500, "ck800", tb, [(T0 + seen_h * H, queued_patch(hk, "hnew", fetched=False))], first_seen=T0 + seen_h * H, label="new revision")
    h = Hotkey(hk, "ck800")
    h.episodes += [a, b]
    return hand_world([h], T0, 4.0 if seen_h <= 4 else 14.0), b


def test_finding_f2_recommit_of_a_waiting_hotkey_is_unpaid_until_seen(tmp_path):
    world, b = _recommit_world()
    sets = _final(world, tmp_path)
    w = _weights_at(sets, T0 + 1 * H)            # 0.7 h after the new commitment, before our evaluator has seen it
    assert w.get(500, 0.0) > 0


def test_finding_f2_switch_off_shows_the_old_gap(tmp_path):
    world, b = _recommit_world()
    sets = _run_with((), world, tmp_path)
    assert _weights_at(sets, T0 + 1 * H).get(500, 0.0) == 0.0


def test_finding_f2_grace_is_short_and_from_the_own_commitment_block(tmp_path):
    """The grace is 6 h from the NEW commitment: still paid 5 h after it, nothing owed to an unseen commitment 7 h after it."""
    world, b = _recommit_world(seen_h=10.0)           # the evaluator records the new revision only 9.7 h after its commitment
    sets = _final(world, tmp_path)
    assert _weights_at(sets, T0 + 0.3 * H + 5 * H).get(500, 0.0) > 0
    assert _weights_at(sets, T0 + 0.3 * H + 7 * H).get(500, 0.0) == 0.0


def _burst_world(n=20):
    hks = []
    for i in range(n):
        hk = ss58(900 + i)
        tc = T0 + 0.05 * H + i * 30.0
        ep = hand_gen(hk, 600 + i, f"ck{900 + i}", tc, [(T0 + 2.5 * H, queued_patch(hk, fetched=False))], first_seen=T0 + 2.5 * H)
        h = Hotkey(hk, f"ck{900 + i}")
        h.episodes.append(ep)
        hks.append(h)
    return hand_world(hks, T0, 3.0), n


def test_finding_f3_seventeenth_fresh_commitment_is_unpaid(tmp_path):
    world, n = _burst_world(20)
    sets = _final(world, tmp_path)
    w = _weights_at(sets, T0 + 1 * H)
    unpaid = [600 + i for i in range(n) if not w.get(600 + i, 0.0) > 0]
    assert not unpaid, f"unpaid fresh commitments (uids): {unpaid}"


def test_finding_f3_switch_off_shows_the_old_cap(tmp_path):
    world, n = _burst_world(20)
    sets = _run_with((), world, tmp_path)
    w = _weights_at(sets, T0 + 1 * H)
    assert sum(1 for i in range(n) if not w.get(600 + i, 0.0) > 0) >= 1


def _f404_world():
    hk = ss58(1000)
    tc = T0
    reason = E.WAITING_FOR_REPO + ": 404 Client Error: revision not found"
    ep = hand_gen(hk, 700, "ck1000", tc, [(tc + 0.2 * H, {**queued_patch(hk, fetched=False), "reason": reason, "wait_kind": ST.WAIT_MINER})],
                  first_seen=tc + 0.2 * H)
    h = Hotkey(hk, "ck1000")
    h.episodes.append(ep)
    return hand_world([h], T0, 4.0)


def test_f4_404_earns_no_window_and_is_an_excluded_case_of_the_oracle(tmp_path):
    """Approved policy (Florian 26 Oct, FAQ 2.16): a 404 revision has no window. The model excludes it (E_ABUSE_404), the code pays 0."""
    import ka_gauntlet_model as M
    world = _f404_world()
    sets = _final(world, tmp_path)
    if not hasattr(CS, "KA_ABUSE_CLAUSE"):          # the Tue (p8) line has no abuse clause: the 404 simply gets the plain 24 h
        assert _weights_at(sets, T0 + 2 * H).get(700, 0.0) > 0.0
        return
    assert CS.KA_ABUSE_CLAUSE is True
    assert _weights_at(sets, T0 + 2 * H).get(700, 0.0) == 0.0
    rep = M.check_world(world, sets)
    assert not rep.gaps and rep.excluded.get("E_ABUSE_404", 0) > 0, (rep.gaps, rep.excluded)


def _f5_world():
    hk = ss58(1100)
    t_old = T0 - 200 * H                                  # first revision's first pass 8.3 days ago, withdrawn at the recommit
    t_rec = T0 - 11 * H
    fp_b = T0 - 9 * H
    a = hand_gen(hk, 800, "ck1100", t_old - 5 * H, [(t_old - 4 * H, queued_patch(hk, "hsame")), (t_old, pending_patch(hk, t_old, "hsame"))],
                 end=t_rec, end_kind="changed", first_seen=t_old - 4 * H, label="old revision")
    b = hand_gen(hk, 800, "ck1100", t_rec, [(t_rec + 1 * H, queued_patch(hk, "hsame")), (fp_b, pending_patch(hk, fp_b, "hsame"))],
                 first_seen=t_rec + 1 * H, label="same tensors, new revision")
    h = Hotkey(hk, "ck1100")
    h.episodes += [a, b]
    return hand_world([h], T0, 24.0)


def test_finding_f5_same_tensors_recommit_inherits_the_old_first_pass(tmp_path):
    sets = _final(_f5_world(), tmp_path)
    assert _weights_at(sets, T0 + 20 * H).get(800, 0.0) > 0         # its own confirmation is due T0 + 3 h, the slot is Thu 29 Oct


def test_finding_f5_switch_off_shows_the_old_cap(tmp_path):
    sets = _run_with((), _f5_world(), tmp_path)
    assert _weights_at(sets, T0 + 20 * H).get(800, 0.0) == 0.0


def _engine_long_wait_world(measure_after_h):
    """An engine digest that is bound and waits for OUR first measurement for ``measure_after_h`` hours (live case: uid 165, 26 Oct)."""
    import ka_gauntlet_model as M
    hk, uid, tc = ss58(1200), 900, T0
    digest = "ee" * 32
    ep = M.Episode("eng", hk, uid, "ck1200", tc, block_at(tc - 3 * DAY), digest=digest, repo="ghcr.io/m/long", label="eng/long-wait")
    t_meas = tc + measure_after_h * H
    base = {"first_seen": (block_at(tc + 0.3 * H), hk), "owner": hk, "result": None, "challenger": None, "champion": False}
    ep.first_seen_ts = tc + 0.3 * H
    ep.reg_t = tc - 3 * DAY
    ep.segs = [(tc, {**base, "first_seen": None, "owner": None}), (tc + 0.3 * H, base),
               (t_meas, {**base, "result": EC.EvalResult(digest=digest, track=ENG, ok=True, score=0.97, items=100, block=block_at(t_meas), session="s1")})]
    ep.decided_t = t_meas
    if measure_after_h > SPEC["engine_wait_h"]:
        ep.excl.append((tc + SPEC["engine_wait_h"] * H, t_meas, "E_ENGINE_WAIT_24H"))
    h = Hotkey(hk, "ck1200")
    h.episodes.append(ep)
    return hand_world([h], T0, 50.0), ep


def test_engine_digest_waiting_longer_than_24h_for_our_measurement(tmp_path):
    """Published bound E_ENGINE_WAIT_24H: an engine digest keeps its keep-alive 24 h from its commitment while WE have not measured
    it. Beyond that nothing is owed (SPEC); the test reports what the code does (engine code is not changed by this slice)."""
    world, ep = _engine_long_wait_world(40.0)
    sets = _final(world, tmp_path)
    at = lambda h: _weights_at(sets, T0 + h * H).get(900, 0.0)    # noqa: E731
    print(f"\n[engine wait > 24 h, measured at +40 h] weight of uid 900 at +1 h {at(1):.5f}, +12 h {at(12):.5f}, +23 h {at(23):.5f}, "
          f"+25 h {at(25):.5f}, +30 h {at(30):.5f}, +39 h {at(39):.5f}")
    assert all(at(h) > 0 for h in (1, 12, 23))                 # inside the published window
    import ka_gauntlet_model as M
    rep = M.check_world(world, sets)
    assert not rep.gaps, rep.gaps[:2]        # beyond 24 h only the published exclusion applies


# ------------------------------------------------------------------------------------------------ a few fixed points of the model
def test_chain_model_commit_reveal_lag():
    """A set at block b counts at the 2nd epoch boundary after b; epoch boundaries are block % 360 == 48."""
    import ka_gauntlet_model as M
    b = M.B_REF + 10
    assert (M.next_boundary(b) - 48) % 360 == 0 and 0 < M.next_boundary(b) - b <= 360
    assert M.effective_block(b) == M.next_boundary(b) + 360
    assert M.effective_block(M.B_REF) == M.B_REF + 720          # a set exactly on a boundary counts two epochs later
    assert M.u16({1: 0.6, 2: 1e-5}, 2) == 1 and M.u16({1: 0.6, 2: 1e-6}, 2) == 0
