"""wiring-3: joint packages on the v5 path, the 26-option cap, critic fixes F1-F9, the rule-A bar."""
from __future__ import annotations

import pytest

from reax_subnet.spec import state as ST
from test_v5_wiring import Gen, gen, rows  # noqa: F401 - the fake pool, the Gen harness and its fixtures


@pytest.mark.parametrize("order", [(3, 4), (4, 3)])
async def test_f1_two_confirmed_in_one_group_the_stronger_keeps_the_title(gen, order):
    g = gen
    await g.crown(1, "a/champ", .50)
    g.seat(1)
    g.gate_on()
    skills = {3: .99, 4: .80}
    for u in order:
        g.add(u, f"a/c{u}", skills[u])
    await g.cycle()
    assert {g.sub(u).status for u in (3, 4)} == {ST.PENDING_CONFIRM}
    await g.cycle(13 * 3600)
    assert g.ts.champion == g.sub(3).key and g.sub(3).status == ST.CHAMPION
    other = g.sub(4)
    assert other.status in (ST.NOT_BETTER, ST.FORMER), other.reason
    if other.status == ST.FORMER:   # the weaker one was decided first and crowned: the stronger then faced it on that draw
        row = next(h for h in g.sub(3).history if h["kind"] == "confirm")
        assert row["incumbent"] == pytest.approx(other.score) and row["bar_key"] == other.key


# ---------------------------------------------------------------------------------------------------------------------
import json
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import fetch as F
from reax_subnet.spec import sampling as S
from reax_subnet.spec import v5, v5wire as W
from spec_helpers import FakeHub, make_weights, write_model
from test_v5_wiring import BASE_REPO, DEFAULT_BASE_DIR, P51, _answers, _row, pub_rows, write_v51


def many_rows(options=40, per_type=170):
    """The fake pool with the ``hle`` and the first ``new domain`` type asking 40 options per question (the omniscience type gets 30
    plus the IDK option, wildguard stays yes/no)."""
    from test_v5_wiring import v51_rows
    t_rows, n_rows = v51_rows(per_type)

    def widen(r, options):
        out = json.loads(json.dumps(r))
        for q in out["questions"]:
            keys = [f"k{i:02d}" for i in range(options)]
            gold = random.Random(r["id"]).choice(keys)
            q["criteria"] = {k: k.upper() for k in keys}
            q["expected"] = gold
            q["instructions"] = f"{r['id']}: which of the {options} candidates is right?"
        return out

    t_rows = [widen(r, 40) if r["task_type"] == "hle" else widen(r, 30) if r["task_type"] == "omniscience" else r for r in t_rows]
    n_rows = [widen(r, 40) if r["task_type"] == "newdom-0" else r for r in n_rows]
    return t_rows, n_rows


# ------------------------------------------------------------------------------------------------ item 2: the 26-option cap
def _wide_item(n=40, nq=2, idk=False, text="Pick one of the {n} candidates (1 of {n}); {m} documents are unrelated"):
    qs = []
    for j in range(nq):
        keys = [f"x{i}" for i in range(n)]
        qs.append({"instructions": text.format(n=n, m=n - 1) + f" q{j}", "criteria": {k: f"text {k}" for k in keys}, "expected": keys[7 + j]})
    return {"id": "item-1", "tier": "mirror", "part": "T", "area": "knowledge", "task_type": "hle", "gold": False, "state": {"a": 1},
            "idk": idk, "questions": qs}


def test_cap_keeps_gold_plus_25_in_original_order_and_relabels():
    it = _wide_item()
    snapshot_before = json.dumps(it, sort_keys=True)
    out = W.cap_item(it, 123)
    assert json.dumps(it, sort_keys=True) == snapshot_before                         # the pool row is not modified
    assert out["capped_from"] == 40
    for j, (q, o) in enumerate(zip(it["questions"], out["questions"])):
        assert list(o["criteria"]) == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        kept = [q["criteria"][k] for k in q["criteria"] if q["criteria"][k] in o["criteria"].values()]
        assert list(o["criteria"].values()) == kept                                 # original order, nothing else changed
        gold_text = q["criteria"][q["expected"]]
        assert o["criteria"][o["expected"]] == gold_text and len(set(o["criteria"].values())) == 26
        assert "26 candidates" in o["instructions"] and "(1 of 40)" in o["instructions"] and "39 documents" in o["instructions"]
        assert "40 candidates" not in o["instructions"]


def test_cap_is_deterministic_per_draw_item_and_question_and_resampled_per_draw():
    it = _wide_item(nq=2)
    a, b = W.cap_item(it, 1), W.cap_item(it, 1)
    assert a == b                                                                    # same draw: the same subset for every model
    c = W.cap_item(it, 2)
    assert a["questions"][0]["criteria"] != c["questions"][0]["criteria"]            # another draw: another sample
    assert list(a["questions"][0]["criteria"].values()) != list(a["questions"][1]["criteria"].values())   # another question index
    other = dict(it, id="item-2")
    assert W.cap_item(other, 1)["questions"][0]["criteria"] != a["questions"][0]["criteria"]               # another item id
    # distractors are sampled from all 39, not from a prefix
    seen = set()
    for seed in range(40):
        seen |= set(W.cap_item(it, seed)["questions"][0]["criteria"].values())
    assert len(seen) == 40


def test_cap_leaves_small_questions_and_counts_without_a_noun_alone():
    small = _wide_item(n=26)
    assert W.cap_item(small, 5) is small                                             # exactly 26: untouched (same object)
    it = _wide_item(text="Number 40 is in the text, but also 40 apples and {n} tools")
    out = W.cap_item(it, 5)["questions"][0]["instructions"]
    assert "Number 40 is" in out and "40 apples" in out and "26 tools" in out
    mixed = _wide_item(n=40)
    mixed["questions"].append({"instructions": "yes or no?", "criteria": {"yes": "Yes", "no": "No"}, "expected": "no"})
    o = W.cap_item(mixed, 5)
    assert o["questions"][2] == mixed["questions"][2] and len(o["questions"][0]["criteria"]) == 26


def test_cap_counts_the_idk_option_against_the_26():
    it = _wide_item(n=30, nq=1, idk=True)
    o = W.cap_item(it, 9)
    assert len(o["questions"][0]["criteria"]) == 25 and "26 candidates" in o["questions"][0]["instructions"]
    flat = W.flatten([o])
    assert len(flat[0]["labels"]) == 26 and flat[0]["labels"][-1] == v5.IDK_KEY       # 25 + IDK: the single-letter readout fits
    from reax_subnet.spec import prompt as P
    P.options(flat[0]["question"])                                                   # no PromptError
    ok26 = _wide_item(n=25, nq=1, idk=True)
    assert W.cap_item(ok26, 9) is ok26


def test_draw_applies_the_cap_and_chance_uses_the_capped_count(rows):
    pool = list(many_rows()[0]) + list(many_rows()[1])
    items, _ = W.draw(pool, W.public_rows(pub_rows()), seed=11, use_counts={}, p=FAST_P)
    typed = [i for i in items if "questions" in i]
    for it in typed:
        for q in it["questions"]:
            n = len(q["criteria"]) + (1 if it.get("idk") else 0)
            assert n <= 26, (it["id"], n)
    capped = [i for i in typed if i.get("capped_from")]
    assert capped and {i["task_type"] for i in capped} == {"hle", "omniscience", "newdom-0"}
    again, _ = W.draw(pool, W.public_rows(pub_rows()), seed=11, use_counts={}, p=FAST_P)
    assert again == items                                                            # the draw (seed) fixes the subset
    other, _ = W.draw(pool, W.public_rows(pub_rows()), seed=12, use_counts={}, p=FAST_P)
    assert [i["questions"][0]["criteria"] for i in other if i.get("capped_from")][:5] != \
        [i["questions"][0]["criteria"] for i in capped][:5]
    # chance-correction reads the served (capped) rows: 1/26 for a hle question, 1/26 for omniscience with IDK
    resp = {"m": _answers(items, .7, "m")}
    table = v5.item_table([i for i in typed if i["task_type"] == "hle"], resp["m"])
    assert set(round(c, 6) for c in table.chance) == {round(1 / 26, 6)}
    omni = v5.item_table([i for i in typed if i["task_type"] == "omniscience"], resp["m"])
    assert set(round(c, 6) for c in omni.chance) == {round(1 / 26, 6)}
    # nothing else of the pool row changed: the same ids as an uncapped draw would pick
    assert {i["id"] for i in items} == {i["id"] for i in W.v5.draw_v5([*pool, *W.public_rows(pub_rows())], n=W.draw_size(FAST_P),
                                                                        seed=11, use_counts={}, params=FAST_P)[0]}


FAST_P = W.params(resamples=40)


def _record_rows(g):
    """Wrap the fake backend so that the rows each model is served are kept (repo -> rows)."""
    inner = g.env.server
    seen: dict[str, list[dict]] = {}

    def factory(path, weight_bytes):
        repo = open(os.path.join(path, "skill_key.txt")).read()
        cm = inner(path, weight_bytes)

        @asynccontextmanager
        async def ctx():
            async with cm as backend:
                class B:
                    async def decide(self, rows):
                        seen.setdefault(repo, []).extend(rows)
                        return await backend.decide(rows)
                yield B()
        return ctx()

    g.env.ev.backend_factory = factory
    return seen


async def test_every_model_of_a_draw_is_served_the_same_capped_questions(tmp_path, monkeypatch):
    g = Gen(tmp_path, monkeypatch, many_rows())
    await g.crown(1, "a/champ", .50)
    g.seat(1)
    g.gate_on()
    g.add(3, "a/one", .95)
    g.add(4, "a/two", .30)
    seen = _record_rows(g)
    await g.cycle()
    assert set(seen) == {"a/champ", "a/one", "a/two", BASE_REPO}
    sig = {repo: {r["id"]: (json.dumps(r["question"], sort_keys=True), r["expected"]) for r in rs} for repo, rs in seen.items()}
    ref = sig["a/champ"]
    assert all(s == ref for s in sig.values())                                       # identical subset, keys, gold for every model
    assert any(len(r["labels"]) == 26 for r in seen["a/champ"]) and all(len(r["labels"]) <= 26 for r in seen["a/champ"])
    assert g.sub(3).status == ST.PENDING_CONFIRM


# ------------------------------------------------------------------------------------------------ item 1: joint packages
JOINT = {"a/champ", "a/joint"}


async def test_a_joint_package_never_reaches_the_causal_readout_in_a_v5_draw(tmp_path, monkeypatch):
    from reax_subnet.spec import joint as J
    g = Gen(tmp_path, monkeypatch, many_rows())
    ev = g.env.ev
    joint_calls: list[tuple[str, list[dict]]] = []
    causal_calls: list[str] = []
    detected: list[str] = []

    def key_of(path):
        return open(os.path.join(path, "skill_key.txt")).read()

    def fake_detect(path, *, category, now_ts=None):
        detected.append(key_of(path))
        return J.JointReadout(16, tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), tuple(range(26))) if key_of(path) in JOINT else None

    def wrap(kind):
        def factory(path, weight_bytes):
            repo = key_of(path)
            assert (kind == "joint") == (repo in JOINT), f"{repo} served on the {kind} path"
            cm = g.env.server(path, weight_bytes)

            @asynccontextmanager
            async def ctx():
                async with cm as backend:
                    class B:
                        async def decide(self, rows):
                            (joint_calls.append((repo, list(rows))) if kind == "joint" else causal_calls.append(repo))
                            return await backend.decide(rows)
                    yield B()
            return ctx()
        return factory

    monkeypatch.setattr(J, "detect", fake_detect)
    monkeypatch.setattr(ev, "_docker_backend", wrap("causal"))
    monkeypatch.setattr(ev, "_joint_backend", wrap("joint"))
    ev.backend_factory = ev._docker_backend                       # the production default: detection decides
    await g.crown(1, "a/champ", .50)                               # (rules v4: the same routing)
    g.seat(1)
    g.gate_on()
    g.add(3, "a/joint", .95)
    g.add(4, "a/plain", .40)
    joint_calls.clear(), causal_calls.clear()
    await g.cycle()
    assert sorted(r for r, _ in joint_calls) == ["a/champ", "a/joint"]               # co-champion and challenger: joint, once each
    assert sorted(causal_calls) == sorted(["a/plain", BASE_REPO])                    # everything else: causal
    rows = dict(joint_calls)["a/joint"]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) and all(isinstance(r["question"], dict) for r in rows)
    multi = [i for i in ids if i.endswith("#2")]                                      # wildguard: three questions
    assert multi and all(i.rsplit("#", 1)[0] + "#0" in ids and i.rsplit("#", 1)[0] + "#1" in ids for i in multi)   # one call row per question
    assert all(len(r["labels"]) <= 26 for r in rows) and any("#" in i for i in ids)  # the cap: the single-letter readout fits
    assert g.sub(3).status == ST.PENDING_CONFIRM, g.sub(3).reason                    # decided on S_v5.1 from the joint answers
    assert detected.count("a/joint") >= 1 and "a/plain" in detected                  # every served model was inspected first


# ------------------------------------------------------------------------------------------------ critic F2: the floor exemption
async def test_f2_a_parent_served_as_a_reference_is_not_marked_v5(gen):
    g = gen
    await g.crown(1, "a/champ", .50)
    g.seat(1)
    g.add(3, "a/par", .70)
    await g.cycle()                                              # v4 first stage: pending, scored on the v4 scale
    par = g.sub(3)
    assert par.status == ST.PENDING_CONFIRM and not W.scored_under_v5(par.by_tier)
    old_score = par.score
    par.status = ST.NOT_BETTER                                   # (a parent that is no challenger: it is served as a reference only)
    g.gate_on()
    from spec_helpers import finetune
    g.env.add(4, "a/child", .95, weights=finetune(make_weights(3 + 100), 7), category="generalist")   # a fine-tune of the parent's weights
    g.commits[4] = "a/child"
    await g.cycle(3600)
    assert g.sub(4).derived_from == par.key
    assert par.score is not None and par.score != old_score      # it was served and re-measured on the v5 draw ...
    assert not W.scored_under_v5(par.by_tier)                     # ... but it is not floor-exempt because of that
    assert W.scored_under_v5(g.sub(1).by_tier)                    # the champion, a reference of the draw, is


def test_f2_big_ok_keeps_the_size_checks_and_skips_only_the_base_floor(monkeypatch):
    from test_keepalive import NOW, _sub
    st = ST.SpecState()
    s = _sub(st, 1, ST.PENDING_CONFIRM, 30.0, track="generalist")
    s.params, s.base_id = 14_000_000_000, "unknown-base"
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", None)
    at = CS.SPEC_OPEN_CLASS_START_UTC.timestamp() + 3600
    assert not E._big_ok(s, "generalist", at, {}, 30.0)                # v4: a big entry without a base floor gets none
    s.by_tier = {"T": 30.0, "N": 20.0}
    assert not E._big_ok(s, "generalist", at, {}, 30.0)                # the marker alone changes nothing while the gate is closed
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(at - 60, timezone.utc))
    assert E._big_ok(s, "generalist", at, {}, 30.0)                     # gate open + v5 score: the v4-scale base floor is skipped
    s.params = -5
    assert not E._big_ok(s, "generalist", at, {}, 30.0)                # an invalid parameter count still fails closed
    s.params = None
    assert E._big_ok(s, "generalist", at, {}, 30.0)                     # unknown count = eligible, as before


# ------------------------------------------------------------------------------------------------ critic F6: the generalist floor
def test_f6_the_v4_floor_still_holds_for_entries_without_the_v5_marker(monkeypatch):
    from test_cochampions import NOW as CNOW, TRACK, case, leaders
    st = case(monkeypatch, co=())
    monkeypatch.setattr(CS, "OPENING_PLAYOFF_SLOTS_START_UTC", None)
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, TRACK, 36.8)
    ts = st.track(TRACK)
    for s in ts.submissions.values():
        s.score, s.by_tier, s.status = 20.0, {}, ST.FORMER if s.status != ST.CHAMPION else s.status
    ts.submissions[ts.champion].score = 20.0
    before = [u for u, _ in leaders(st)]
    assert before                                                           # gate off: listed, the share layer applies the floor
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(CNOW - 60, timezone.utc))
    assert [u for u, _ in leaders(st)] == []                                # gate on, nobody scored under v5.1: the v4 floor drops all
    keep = next(s for s in ts.submissions.values() if s.uid == 2)
    keep.by_tier = {"T": 20.0, "N": 18.0}
    assert [u for u, _ in leaders(st)] == [2]                               # the v5.1-scored former champion is ranked, rank 1 of the list
    paid = CS.combine_weights({99: 1.0}, None, 0, {TRACK: leaders(st)}, now=CNOW)
    assert paid[2] > 0 and not {1, 3, 4} & set(paid)


# ------------------------------------------------------------------------------------------------ critic F9: pool order
def test_f9_load_pool_sorts_by_id_and_the_draw_does_not_depend_on_file_order(tmp_path, rows):
    t_rows, n_rows = rows
    a = write_v51(tmp_path / "a", t_rows, n_rows)
    shuffled_t, shuffled_n = list(t_rows), list(n_rows)
    random.Random(4).shuffle(shuffled_t), random.Random(5).shuffle(shuffled_n)
    b = write_v51(tmp_path / "b", shuffled_t, shuffled_n)
    la, lb = W.load_pool(a, FAST_P), W.load_pool(b, FAST_P)
    assert [r["id"] for r in la] == sorted(r["id"] for r in la) == [r["id"] for r in lb]
    pub = W.public_rows(pub_rows())
    da, _ = W.draw(la, pub, seed=3, use_counts={}, p=FAST_P)
    db, _ = W.draw(lb, pub, seed=3, use_counts={}, p=FAST_P)
    assert [r["id"] for r in da] == [r["id"] for r in db]


# ------------------------------------------------------------------------------------------------ critic F4: the served cache
async def test_f4_served_results_are_keyed_by_the_item_set(gen):
    ev = gen.env.ev
    calls = []

    async def fake_serve(p, rows, now):
        calls.append([r["id"] for r in rows])
        p.probs = [{"a": 1.0}] * len(rows)
        return [object()] * len(rows)

    ev._serve = fake_serve
    p = SimpleNamespace(sub=SimpleNamespace(key="m"), probs=None, joint=False)
    flat_a, flat_b = [{"id": "1#0"}, {"id": "2#0"}], [{"id": "1#0"}, {"id": "3#0"}]
    first = await ev._serve_v5(p, flat_a, "draw", 0)
    assert await ev._serve_v5(p, list(reversed(flat_a)), "draw", 0) is first and len(calls) == 1      # same set: cached
    assert await ev._serve_v5(p, flat_b, "draw", 0) is not first and len(calls) == 2                  # same id, other items: served again
    key_a = next(k for k, v in ev._v5_served.items() if len(v[1]) == 2 and v[0] is first)
    ev._v5_served[key_a] = (first, [{"a": 1.0}], False)                                                      # a corrupted entry is never used
    n = len(calls)
    assert await ev._serve_v5(p, flat_a, "draw", 0) is not first and len(calls) == n + 1


async def test_f4_a_cache_hit_keeps_the_joint_flag(gen):
    ev = gen.env.ev

    async def fake_serve(p, rows, now):
        p.joint = p.is_joint
        p.probs = [{"a": 1.0}] * len(rows)
        return [object()] * len(rows)

    ev._serve = fake_serve
    flat = [{"id": "1#0"}, {"id": "2#0"}]
    joint = SimpleNamespace(sub=SimpleNamespace(key="co"), probs=None, joint=False, is_joint=True)
    causal = SimpleNamespace(sub=SimpleNamespace(key="base"), probs=None, joint=False, is_joint=False)
    for p in (joint, causal):
        await ev._serve_v5(p, flat, "draw", 0)
    for p, want in ((joint, True), (causal, False)):
        p.joint = not want                      # a stale flag on the object that gets the hit
        await ev._serve_v5(p, flat, "draw", 0)
        assert p.joint is want


async def test_f4_the_cache_is_cleared_every_cycle(gen):
    g = gen
    await g.crown(1, "a/champ", .50)
    g.env.ev._v5_served[("x", "y")] = ([], [], False)
    await g.cycle(3600)
    assert ("x", "y") not in g.env.ev._v5_served


# ------------------------------------------------------------------------------------------------ critic F3: per-challenger waits
BAD_BASE_REPO, BAD_BASE_MODEL = "base/unservable", "Qwen/Qwen3-8B@" + "c" * 40


def _bad_base_world(g, monkeypatch):
    """A second declared base that is on this host but cannot be served (the backend raises for it)."""
    from reax_subnet.spec import vllm as V
    write_model(os.path.join(g.env.cfg.base_dir, "Qwen__Qwen3-8B@" + "c" * 40), make_weights(998), skill_key=BAD_BASE_REPO,
                category="generalist")
    inner = g.env.server

    def factory(path, weight_bytes):
        if open(os.path.join(path, "skill_key.txt")).read() == BAD_BASE_REPO:
            raise V.ModelUnavailable("invalid weights: the base cannot be served")
        return inner(path, weight_bytes)

    g.env.ev.backend_factory = factory


async def test_f3_an_unservable_base_holds_back_only_its_own_challenger(gen, monkeypatch):
    g = gen
    await g.crown(1, "a/champ", .50)
    g.seat(1)
    g.gate_on()
    _bad_base_world(g, monkeypatch)
    g.add(3, "a/good1", .95)
    g.env.add(4, "a/badbase", .99, category="generalist", base_model=BAD_BASE_MODEL)
    g.commits[4] = "a/badbase"
    g.add(5, "a/good2", .20)
    await g.cycle()
    good1, bad, good2 = g.sub(3), g.sub(4), g.sub(5)
    assert good1.status == ST.PENDING_CONFIRM and good2.status == ST.NOT_BETTER      # the group went on and decided the others
    assert bad.status in (ST.QUEUED, ST.PENDING_CONFIRM) and bad.score is None and bad.reason.startswith("waiting")
    assert "could not be served" in bad.reason and not any(h.get("kind") == "first" for h in bad.history)
    assert "generalist" not in g.env.ev._host_failed                                  # no abandoned group
    # a queued entry that waits is not picked again before RETRY_UNAVAILABLE_S, so it does not starve the others
    await g.cycle(60)
    assert bad.status == ST.QUEUED and bad.score is None


async def test_f3_v5_bases_reports_the_missing_base_per_challenger(gen):
    g = gen
    ev = g.env.ev
    ts = g.ts
    await g.crown(1, "a/champ", .50)
    g.add(3, "a/one", .60)
    await g.cycle()
    good, other = g.sub(3), g.sub(1)
    other.base_model = "Qwen/Qwen3-70B@" + "9" * 40
    p_good = E._Prepared(good, "/nowhere", 1)
    p_miss = E._Prepared(other, "/nowhere", 1)
    bases, missing = await ev._v5_bases(ts, [p_good, p_miss], g.env.clock.t)
    assert missing == [p_miss] and list(bases) == [good.base_model]
    assert other.reason.startswith("waiting") and "not available" in other.reason and not good.reason.startswith("waiting")


def test_f3_a_model_without_a_finite_score_is_marked_bad_and_gets_no_reference(rows):
    items, _ = W.draw(rows[0] + rows[1], W.public_rows(pub_rows()), seed=3, use_counts={}, p=FAST_P)
    resp = {"ch": _answers(items, .9, "ch"), "ch2": _answers(items, .8, "ch2"), "base": _answers(items, .5, "base"),
            "cc": _answers(items, .6, "cc")}
    real = v5.score_v5_detail

    def nan_for_ch2(responses, its, base_responses, params=v5.DEFAULT, seed=0, **kw):
        res = real(responses, its, base_responses, params, seed, **kw)
        if responses is resp["ch2"] or responses == resp["ch2"]:
            from dataclasses import replace
            return replace(res, s=float("nan"))
        return res

    v5_mod = W.v5
    orig = v5_mod.score_v5_detail
    v5_mod.score_v5_detail = nan_for_ch2
    try:
        ctx = W.build_context("d", items, resp, {"ch": "base", "ch2": "base"}, ["cc"], 1, FAST_P)
    finally:
        v5_mod.score_v5_detail = orig
    assert ctx.bad == {"ch2"} and "ch" in ctx.ref and "ch2" not in ctx.ref


async def test_f3_a_challenger_without_a_finite_score_waits_and_the_rest_is_decided(gen, monkeypatch):
    g = gen
    await g.crown(1, "a/champ", .50)
    g.seat(1)
    g.gate_on()
    g.add(3, "a/fine", .95)
    g.add(4, "a/nan", .96)
    real = E.V5W.build_context

    def build(draw_id, items, resp, challengers, seated, seed, p=None):
        ctx = real(draw_id, items, resp, challengers, seated, seed, p)
        nan_key = next(k for k in challengers if g.ts.submissions[k].uid == 4)
        ctx.bad.add(nan_key)
        ctx.ref.pop(nan_key, None)
        return ctx

    monkeypatch.setattr(E.V5W, "build_context", build)
    await g.cycle()
    assert g.sub(3).status == ST.PENDING_CONFIRM
    nan = g.sub(4)
    assert nan.score is None and nan.reason.startswith("waiting") and "finite" in nan.reason


# ------------------------------------------------------------------------------------------------ critic F5: D4 needs a repo probe
def _d4_unfetched_world(monkeypatch):
    from test_keepalive import DF, NOW
    from test_provisional_ka import BLOCK, _p, _snapr
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(NOW - 3600, timezone.utc))
    st = ST.SpecState()
    s = _p(st, 1, since=None)
    s.provisional_since_ts, s.hash_named = None, ""              # committed, not fetched yet
    snap = _snapr([s], regs={1: BLOCK - 500}, cblocks={1: BLOCK - 1800})
    return st, s, snap, NOW, DF


@pytest.mark.usefixtures("rc5_entry_floors")
def test_f5_an_unfetched_entry_gets_no_slot_before_its_repo_was_probed(monkeypatch):
    st, s, snap, now, df = _d4_unfetched_world(monkeypatch)
    assert not E.mark_provisional(st, [df], snap, now) and s.provisional_since_ts is None
    assert E.compute_provisional(st, [df], snap, now=now)[df] == []
    assert E.mark_provisional(st, [df], snap, now, probed={"someone-else"}) is False
    assert E.mark_provisional(st, [df], snap, now, probed={s.key})
    assert s.provisional_since_ts == pytest.approx(now - 6 * 3600) and E.compute_provisional(st, [df], snap, now=now)[df] == [1]
    # a fetched entry has passed the fetch (the repo exists): no probe needed; gate off: rc11, no probe either
    st2, s2, snap2, now, df = _d4_unfetched_world(monkeypatch)
    s2.hash_named = "h1"
    assert E.mark_provisional(st2, [df], snap2, now)
    # (gate off with an unfetched entry is the rc11 rule; covered by test_d4_gate_off_is_rc11_no_slot_before_the_immunity_lead)


class _Listing:
    def __init__(self, files):
        self.files, self.sha = files, "x"


class _ProbeHub(FakeHub):
    def __init__(self, workdir, listings):
        super().__init__(workdir)
        self.listings = listings

    async def probe_listing(self, commitment):
        self.probes.append(commitment.repo)
        item = self.listings.get(commitment.repo)
        if isinstance(item, F.FetchError):
            raise item
        if item is None:
            raise F.FetchError("unavailable", "not public yet")
        return _Listing(item)


@pytest.mark.usefixtures("rc5_entry_floors")
async def test_f5_the_probe_checks_repo_public_and_safetensors_and_retries_later(tmp_path, monkeypatch):
    from test_spec_evaluator import Env
    st, s, snap, now, df = _d4_unfetched_world(monkeypatch)
    ev = Env(tmp_path, {}).ev
    ev.state = st
    s.repo = "a/real"
    hub = ev.fetcher = _ProbeHub(str(tmp_path), {"a/real": {"x.json": 10}})
    clock = ev.clock
    ev.clock = lambda: now
    snap2 = snap
    await ev._d4_probe([df], snap2, now)
    assert s.key not in ev._d4_probed                                              # a repo without safetensors: no slot
    hub.listings["a/real"] = {"model.safetensors": 100, "config.json": 5}
    await ev._d4_probe([df], snap2, now + 1)
    assert s.key not in ev._d4_probed                                              # failed probes are retried after RETRY_UNAVAILABLE_S only
    await ev._d4_probe([df], snap2, now + 700)
    assert s.key in ev._d4_probed and hub.probes == ["a/real", "a/real"]
    assert E.mark_provisional(st, [df], snap2, now, ev._d4_probed) and E.compute_provisional(st, [df], snap2, now=now)[df] == [1]


@pytest.mark.usefixtures("rc5_entry_floors")
async def test_f5_a_missing_repo_or_an_infra_error_gives_no_slot_and_the_gate_off_never_probes(tmp_path, monkeypatch):
    from test_spec_evaluator import Env
    st, s, snap, now, df = _d4_unfetched_world(monkeypatch)
    ev = Env(tmp_path, {}).ev
    ev.state = st
    hub = ev.fetcher = _ProbeHub(str(tmp_path), {"a/m1": F.FetchError("infra", "hub down")})
    s.repo = "a/m1"
    await ev._d4_probe([df], snap, now)
    assert s.key not in ev._d4_probed and s.key not in ev._d4_probe_failed         # infrastructure says nothing about the repo
    hub.listings.clear()                                                           # the repo does not exist
    await ev._d4_probe([df], snap, now + 1)
    assert s.key not in ev._d4_probed and s.key in ev._d4_probe_failed
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", None)
    n = len(hub.probes)
    await ev._d4_probe([df], snap, now + 5000)
    assert len(hub.probes) == n                                                    # gate off: nothing is probed


# ------------------------------------------------------------------------------------------------ rc11.post2 + rule A
async def test_rule_a_two_earned_entries_play_the_v4_slot_playoff_at_the_flip_then_v5_follows(gen, monkeypatch):
    """Two entries confirm under v4 before the flip: at the slot (= the gate) the release owner's slot playoff decides between
    them under v4, one is crowned, the other is queued for a new first draw (rc11.post4, slot rules section 2; post2 made it
    not better) whose new first draw is on v5.1 (rule A does not restart it again); an unconfirmed v4 stage restarts as a
    v5.1 first stage and is decided against the new champion alone (and its base)."""
    g = gen
    await g.crown(1, "a/old", .60)
    g.seat(1)
    t0 = g.env.clock.t
    slot = datetime.fromtimestamp(t0 + 16 * 3600, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot,))
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", slot)
    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", lambda self_: FAST_P)
    g.add(3, "a/e1", .90)
    g.add(5, "a/e2", .95)
    await g.cycle()
    g.add(4, "a/late", .99)
    await g.cycle(5 * 3600)
    await g.cycle(8 * 3600)                                                  # 13 h: both confirm under v4, both crowning stored
    e1, e2, late = g.sub(3), g.sub(5), g.sub(4)
    assert all(any(h["kind"] == "crown_pending" for h in s.history) for s in (e1, e2)) and g.ts.champion == g.sub(1).key
    await g.cycle(slot.timestamp() - g.env.clock.t)
    assert g.env.clock.t == slot.timestamp() and S.v5_active(g.env.clock.t)
    crowned = [s for s in (e1, e2) if s.status == ST.CHAMPION]
    assert len(crowned) == 1 and g.ts.champion == crowned[0].key
    assert any(h["kind"] == "playoff" and h.get("slot") for s in (e1, e2) for h in s.history)    # the v4 slot playoff ran
    assert not any(h.get("scoring_regime") == "v5" for s in (e1, e2) for h in s.history if h["kind"] in ("playoff", "crown_done"))
    loser = e1 if crowned[0] is e2 else e2
    assert E._crown_pending(loser) is None and loser.history[-1]["kind"] == "slot_restart" and g.sub(1).status == ST.FORMER
    assert loser.status == ST.QUEUED and not loser.draw1             # rc11.post4 (slot rules section 2): a new first draw
    rec = g.ts.slot_results[-1]
    assert rec["evaluation"] in ("v3", "v4") and rec["eval_ts"] == slot.timestamp() - 1 and rec["crowned"] == [crowned[0].uid]   # pinned before the gate
    assert any(h["kind"] == "v5_restart" for h in late.history)             # the unconfirmed v4 stage restarts as a v5.1 first stage
    n_conf = sum(h["kind"] == "confirm" for h in loser.history)
    await g.cycle(3600)
    first = [h for h in late.history if h["kind"] == "first"][-1]
    assert first["scoring_regime"] == "v5" and set(first["ref_totals"]) == {crowned[0].key, W.base_key(late.base_model)}   # bar: the new champion alone
    assert not any(h["kind"] == "v5_restart" for h in loser.history)          # restarted once, by the slot, not again by rule A
    first = [h for h in loser.history if h["kind"] == "first"][-1]           # the slot loser's new first draw is on v5.1
    assert first["scoring_regime"] == "v5" and set(first["ref_totals"]) == {crowned[0].key, W.base_key(loser.base_model)}
    assert all(h.get("scoring_regime") == "v5" for h in [h for h in loser.history if h["kind"] == "confirm"][n_conf:])


async def test_flip_slot_crowning_restarts_a_later_slot_v4_entry_once_on_v51(gen, monkeypatch):
    """rc11.post4 (slot rules section 2) with rule A at the Monday flip slot: an entry that confirmed under v4 after the cutoff
    (stored for Thursday) is queued for a new first draw when the flip slot crowns a new champion; rule A does not restart it
    a second time, and its new first draw is on v5.1 against the new champion."""
    g = gen
    await g.crown(1, "a/old", .60)
    g.seat(1)
    t0 = g.env.clock.t
    slot = datetime.fromtimestamp(t0 + 16 * 3600, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot, datetime.fromtimestamp(slot.timestamp() + 3 * 86400, timezone.utc)))
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", slot)
    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", lambda self_: FAST_P)
    g.add(3, "a/e1", .90)
    g.add(5, "b/e2", .95)
    await g.cycle()
    await g.cycle(13 * 3600)                                                 # both confirm under v4, both crownings stored
    e1, later = g.sub(3), g.sub(5)
    assert E._crown_pending(e1) is not None and E._crown_pending(later) is not None
    row = [h for h in later.history if h["kind"] == "crown_pending"][-1]     # later confirmed after the cutoff: Thursday's
    row["ts"] = row["slot"] = slot.timestamp() + 60
    old = later.draw1
    await g.cycle(slot.timestamp() - g.env.clock.t)
    assert g.ts.champion == e1.key and g.ts.slot_results[-1]["result"] == "single_entry"
    assert [h["kind"] for h in later.history].count("slot_restart") == 1 and E._crown_pending(later) is None
    assert not any(h["kind"] == "v5_restart" for h in later.history)          # rule A runs after the crown: nothing left to restart
    for _ in range(3):
        await g.cycle(3600)
    assert not any(h["kind"] == "v5_restart" for h in later.history)
    i = next(i for i, h in enumerate(later.history) if h["kind"] == "slot_restart")
    after = [h for h in later.history[i + 1:] if h["kind"] in ("first", "confirm")]
    assert after and after[0]["kind"] == "first" and after[0]["draw"] != old and after[0]["scoring_regime"] == "v5"
    assert set(after[0]["ref_totals"]) == {e1.key, W.base_key(later.base_model)}     # against the new champion alone


async def test_the_v4_slot_playoff_does_not_run_for_entries_decided_under_v5(gen, monkeypatch):
    g = gen
    await g.crown(1, "a/old", .60)
    g.seat(1)
    g.gate_on()
    g.add(3, "a/x", .95)
    await g.cycle()
    await g.cycle(13 * 3600)
    x = g.sub(3)
    ts = g.ts
    called = []
    async def prepare(*a, **k):
        called.append(1)

    monkeypatch.setattr(E.SpecEvaluator, "_prepare_cycle", prepare)
    # a slot field always has a stored crowning (_finalise_crown); here for the slot AT the gate instant: that slot is decided on
    # v4 (slot_eval_ts = slot - 1 s, rc11.post4)
    x.history.append({"kind": "crown_pending", "ts": g.env.clock.t - 60, "slot": S.EVAL_V5_START_UTC.timestamp(), "k1a": True})
    await g.env.ev._run_playoff(E.C.DEFAULT_TRACKS[-1], ts, [x, g.sub(1)], None, g.env.clock.t, 10, slot=True)
    assert not called                                                        # fail closed: nothing prepared, nothing served
    x.history[:] = [h for h in x.history if h.get("scoring_regime") != "v5"]    # decided under v4: the slot playoff may run
    await g.env.ev._run_playoff(E.C.DEFAULT_TRACKS[-1], ts, [x], None, g.env.clock.t, 10, slot=True)
    assert called


@pytest.mark.usefixtures("rc5_entry_floors")
@pytest.mark.parametrize("gate", ["off", "armed"])     # armed: the release gate (Mon 26 Oct 14:30Z), not reached at NOW
def test_gate_off_the_new_pieces_change_nothing(monkeypatch, gate):
    from conftest import RELEASE_EVAL_V5_START_UTC
    from test_keepalive import DF, NOW
    from test_provisional_ka import _p, _snapr
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC if gate == "armed" else None)
    assert not S.v5_active(NOW)
    st = ST.SpecState()
    s = _p(st, 1, since=None)
    s.provisional_since_ts, s.hash_named = None, ""                          # unfetched, immunity due: the rc11 rule needs no probe
    assert E.mark_provisional(st, [DF], _snapr([s]), NOW) and s.provisional_since_ts == NOW
    assert E.SpecEvaluator._v4_draw_now(NOW) == NOW                         # the slot playoff draws at the time it is asked
    assert not E._v51_scored(s, NOW) and E.v5_bar_members(ST.TrackState("generalist"), NOW) == []
    s.by_tier = {"T": 1.0, "N": 1.0}
    assert not E._v51_scored(s, NOW)                                         # the marker without the gate: not exempt
