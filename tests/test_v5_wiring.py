"""Eval v5.1 evaluator wiring (spec/v5wire.py + the gated hooks in spec/evaluator.py): pool, draw, base on every draw,
co-champion re-score (D3), provisional keep-alive from the commitment (D4), and gate-off equivalence. Fake models only."""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from spec_helpers import (HOTKEYS, FakeHub, FakeServer, commit, make_weights, neuron, snapshot, synthetic_pool, write_model,
                          write_pool)

from reax_subnet import competition_split as CS
from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import state as ST
from reax_subnet.spec import v5, v5wire as W

GATE = datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc)
FAST = W.params(resamples=40)


# ------------------------------------------------------------------------------------------------------ fake pool
AREA_OF = {"omniscience": "knowledge", "hle": "knowledge", "when2call": "tools", "wildguard": "language"}


def _row(rid, part, task_type, gold, *, area, options=4, idk=False, nq=1, yesno=False):
    rng = random.Random(rid)
    qs = []
    for j in range(nq):
        keys = ["yes", "no"] if yesno else [chr(97 + i) for i in range(options)]
        qs.append({"instructions": f"{rid} q{j}", "criteria": {k: k.upper() for k in keys}, "expected": rng.choice(keys)})
    return {"id": rid, "tier": "mirror", "part": part, "area": area, "task_type": task_type, "gold": gold, "source": "fake",
            "collected": "2026-10-20", "public": False, "state": {"id": rid}, "idk": idk, "questions": qs}


def v51_rows(per_type=170, n_types=6):
    t_rows, n_rows = [], []
    for area in v5.HF_AREAS:
        for k in range(2):
            tt = f"{area}-{k}"
            t_rows += [_row(f"T-{tt}-{i}", "T", tt, k == 0, area=area) for i in range(per_type)]
    t_rows += [_row(f"T-omni-{i}", "T", "omniscience", False, area="knowledge", idk=True, options=3) for i in range(per_type)]
    t_rows += [_row(f"T-hle-{i}", "T", "hle", True, area="knowledge") for i in range(per_type)]
    t_rows += [_row(f"T-w2c-{i}", "T", "when2call", False, area="tools") for i in range(per_type)]
    t_rows += [_row(f"T-wg-{i}", "T", "wildguard", False, area="language", nq=3, yesno=True) for i in range(per_type)]
    areas = list(v5.HF_AREAS)
    for k in range(n_types):
        tt = f"newdom-{k}"
        n_rows += [_row(f"N-{tt}-{i}", "N", tt, False, area=areas[k % 5]) for i in range(per_type)]
    return t_rows, n_rows


def write_v51(dirpath, t_rows, n_rows, mode=0o600):
    os.makedirs(dirpath, exist_ok=True)
    for name, rows in (("v51.T.jsonl", t_rows), ("v51.N.jsonl", n_rows)):
        p = os.path.join(dirpath, name)
        with open(p, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows))
        os.chmod(p, mode)
    return str(dirpath)


@pytest.fixture(scope="module")
def rows():
    return v51_rows()


def pub_rows(n=600):
    out = []
    for cat in S.GENERALIST_CATEGORIES:
        for i, r in enumerate(synthetic_pool(n)):
            out.append(dict(r, id=f"{cat}-{r['id']}", category=cat))
    return [r for r in out if r["tier"] in v5.PUB_TIERS]


P51 = W.params(resamples=40)


# ------------------------------------------------------------------------------------------------------- W1: pool
def test_loader_reads_the_typed_files_and_refuses_bad_ones(tmp_path, rows):
    t, n = rows
    d = write_v51(tmp_path / "p", t, n)
    pool = W.load_pool(d)
    assert len(pool) == len(t) + len(n) and {r["part"] for r in pool} == {"T", "N"} and all(r["tier"] == "mirror" for r in pool)
    meta = v5.type_meta(pool)
    assert meta["knowledge-0"] == ("knowledge", True, "T") and meta["omniscience"][2] == "T" and meta["newdom-0"][2] == "N"
    for bad in ({"gold": "yes"}, {"task_type": ""}, {"part": "X"}, {"area": "chess"}):
        with pytest.raises(S.SampleError):
            v5.validate_mirror_row(dict(t[0], **bad))
    d = write_v51(tmp_path / "tier", [dict(t[0], tier="same_family")] + t[1:], n)
    with pytest.raises(S.SampleError, match="tier"):
        W.load_pool(d)


def test_loader_fails_closed(tmp_path, rows):
    t, n = rows
    with pytest.raises(S.SampleError, match="REAX_SPEC_V5_POOL_DIR"):
        W.load_pool("")
    for mode in (0o644, 0o640, 0o604):
        d = write_v51(tmp_path / f"m{mode:o}", t, n, mode=mode)
        with pytest.raises(S.SampleError, match="0600"):
            W.load_pool(d)
    d = write_v51(tmp_path / "wrong", n, t)                      # N rows in the T file
    with pytest.raises(S.SampleError, match="part"):
        W.load_pool(d)
    d = write_v51(tmp_path / "dup", t + t[:1], n)
    with pytest.raises(S.SampleError, match="duplicate"):
        W.load_pool(d)
    d = write_v51(tmp_path / "noN", t, n)
    os.remove(os.path.join(d, "v51.N.jsonl"))
    with pytest.raises(S.SampleError, match="v51.N.jsonl"):
        W.load_pool(d)
    assert {r["part"] for r in W.load_pool(d, W.params(w_n=0))} == {"T"}          # the T-only fallback needs no N file
    os.remove(os.path.join(d, "v51.T.jsonl"))
    with pytest.raises(S.SampleError, match="v51.T.jsonl"):
        W.load_pool(d, W.params(w_n=0))


def test_draw_composition_coverage_and_use_counts(rows):
    pool = rows[0] + rows[1]
    pub = W.public_rows(pub_rows())
    got, fresh = W.draw(pool, pub, seed=7, use_counts={})
    assert fresh and len(got) == 990 and len({r["id"] for r in got}) == 990
    assert Counter(("pub" if r["tier"] != "mirror" else r["part"]) for r in got) == {"T": 540, "N": 360, "pub": 90}
    per_type = Counter(r["task_type"] for r in got if r["tier"] == "mirror")
    assert min(per_type.values()) >= 12                                                  # at least 12 items per type per draw
    assert all(per_type[f"newdom-{k}"] == 60 for k in range(6))                          # N: equal per task
    area = Counter(r["area"] for r in got if r["tier"] == "mirror" and r["part"] == "T")
    assert [area[a] for a in v5.HF_AREAS] == pytest.approx([540 * w for w in v5.HF_AREAS.values()], abs=1.5)     # DI area weights
    assert per_type["knowledge-0"] > per_type["knowledge-1"] and per_type["hle"] > per_type["omniscience"]       # gold types weigh x1.2
    assert W.draw(pool, pub, seed=7, use_counts={}) == (got, fresh)                         # deterministic (for a given pool order)
    again, _ = W.draw(pool, pub, seed=8, use_counts={r["id"]: 1 for r in got}, exclude={r["id"] for r in got})
    assert not {r["id"] for r in got} & {r["id"] for r in again}                         # a disjoint confirmation draw fits
    # use counts: fresh and least used first; a row at MIRROR_MAX_USES is retired
    used = {r["id"]: v5.MIRROR_MAX_USES for r in pool if r["task_type"] == "knowledge-1"}
    with pytest.raises(S.SampleError, match="knowledge-1"):
        W.draw(pool, pub, seed=7, use_counts=used)                                       # every row of that type retired: fail closed
    assert W.draw(pool, pub, seed=7, use_counts={r["id"]: 1 for r in pool})[1] is False
    only_t = W.draw(pool, pub, seed=7, use_counts={}, p=W.params(w_n=0))[0]
    assert Counter(("pub" if r["tier"] != "mirror" else r["part"]) for r in only_t) == {"T": 540, "pub": 90}


def test_draw_fails_closed_without_borrowing(rows):
    pool = rows[0] + rows[1]
    pub = W.public_rows(pub_rows())
    thin = [r for r in pool if not (r["task_type"] == "newdom-0" and int(r["id"].rsplit("-", 1)[1]) >= 40)]
    with pytest.raises(S.SampleError, match="newdom-0"):
        W.draw(thin, pub, seed=1, use_counts={})                                         # 40 of the 60 it needs: no borrowing
    with pytest.raises(S.SampleError, match="public"):
        W.draw(pool, pub[:30], seed=1, use_counts={})                                    # too few public items
    assert W.draw_size(W.params()) == 990 and W.draw_size(W.params(w_n=0)) == 630


def test_flatten_makes_one_choice_row_per_question(rows):
    t = rows[0]
    omni = next(r for r in t if r["task_type"] == "omniscience")
    wg = next(r for r in t if r["task_type"] == "wildguard")
    pub = pub_rows(5)[0]
    flat = W.flatten([omni, wg, pub])
    assert [r["id"] for r in flat] == [f"{omni['id']}#0", f"{wg['id']}#0", f"{wg['id']}#1", f"{wg['id']}#2", pub["id"]]
    assert flat[0]["labels"] == ["a", "b", "c", "idk"] and flat[0]["question"]["criteria"]["idk"] == v5.IDK_TEXT   # IDK is an option
    assert flat[1]["labels"] == ["yes", "no"] and flat[1]["question"]["type"] == "choice"
    assert flat[0]["tier"] == "T" and flat[-1] == pub
    probs = [{"a": .7, "b": .2, "c": .05, "idk": .05}, {"yes": .9, "no": .1}, None, {"yes": .4, "no": .6}, {"x": 1}]
    resp = W.responses([omni, wg, pub], flat, probs)
    assert resp[omni["id"]] == [probs[0]] and resp[wg["id"]] == probs[1:4] and resp[pub["id"]] == probs[4]
    assert v5.item_table([wg], resp).correct[0] == 0.0                                    # a failed question makes the item wrong


# --------------------------------------------------------------------------------------------- scoring numbers
def _answers(items, skill, tag):
    """Answers of a fake model: right with probability ``skill`` (a hash of tag and item), confident either way."""
    import hashlib
    out = {}
    for it in items:
        h = int(hashlib.sha256(f"{tag}|{it['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if "questions" not in it:
            labels = [str(x) for x in it["labels"]]
            gold = str(it["expected"])
            pick = gold if h < skill else next(x for x in labels if x != gold)
            out[it["id"]] = {k: (0.85 if k == pick else 0.15 / (len(labels) - 1)) for k in labels}
            continue
        res = []
        for q in it["questions"]:
            keys = list(q["criteria"]) + ([v5.IDK_KEY] if it.get("idk") else [])
            gold = str(q["expected"])
            pick = gold if h < skill else next(k for k in keys if k != gold)
            res.append({k: (0.85 if k == pick else 0.15 / (len(keys) - 1)) for k in keys})
        out[it["id"]] = res
    return out


def test_context_reference_gap_and_parts(rows):
    pool = rows[0] + rows[1]
    items, _ = W.draw(pool, W.public_rows(pub_rows()), seed=3, use_counts={})
    resp = {"ch": _answers(items, .9, "ch"), "base": _answers(items, .5, "base"), "cc1": _answers(items, .6, "cc1"),
            "cc2": _answers(items, .75, "cc2"), "parent": _answers(items, .7, "par")}
    ctx = W.build_context("d", items, resp, {"ch": "base"}, ["cc1", "cc2", "parent"], 1, P51)
    s = {k: p.s for k, p in ctx.parts.items()}
    assert s["ch"] > s["cc2"] > s["parent"] > s["cc1"] > s["base"]
    ref = ctx.ref["ch"]
    assert ref.best == "cc2" and ref.total == s["cc2"] and ref.key == "v5|cc1+cc2+parent|base"
    assert ref.se > 0 and ctx.pair_se("ch", "cc2") == ref.se                              # D3: the best of the co-champions and the base
    ex = v5.score_v5_detail(resp["ch"], items, resp["base"], P51, 1)
    assert ctx.parts["ch"].s == ex.s and ctx.parts["ch"].t == ex.t_eq and ctx.parts["base"].s == ex.detail["S_v5.1_base"]
    assert ctx.gap["ch"] == W.Gap(ex.g, ex.se_g, ex.a, ex.gap_flag) and ctx.gap["ch"].allowance >= 1.5
    # an absolute S for a seated model does not depend on which base it is "scored against"
    assert ctx.parts["cc2"].s == pytest.approx(v5.score_v5_detail(resp["cc2"], items, resp["cc1"], P51, 1).s)
    # the base is the bar when it is better than every seated model
    ctx2 = W.build_context("d", items, resp, {"cc1": "base", "ch": "cc2"}, ["cc1"], 1, P51)
    assert ctx2.ref["ch"].best == "cc2"
    # the gap check reads the public items only through G = (Pub - Pub_base) - (S - S_base); it never changes S
    pub_only = {"ch": dict(resp["base"]), "base": resp["base"]}
    for it in items:
        if it["tier"] != "mirror":
            pub_only["ch"][it["id"]] = _answers([it], 1.0, "x")[it["id"]]                  # perfect on public, base-like in private
    c3 = W.build_context("d", items, pub_only, {"ch": "base"}, [], 1, P51)
    assert c3.parts["ch"].s == pytest.approx(c3.parts["base"].s) and c3.gap["ch"].g > c3.gap["ch"].allowance and c3.gap["ch"].flag
    assert W.build_context("d", items, resp, {"ch": "base"}, [], 1, P51).gap["ch"].flag is False   # a real private gain is not flagged
    no_pub = [i for i in items if i["tier"] == "mirror"]
    assert v5.score_v5_detail(resp["ch"], no_pub, resp["base"], P51, 1).s == pytest.approx(ctx.parts["ch"].s)   # public items carry weight 0
    assert v5.score_v5_detail(resp["ch"], no_pub, resp["base"], P51, 1).gap_flag is None                      # no public items: no gap check


def test_paired_s_se_matches_the_scorers_own_bootstrap(rows):
    pool = rows[0] + rows[1]
    items, _ = W.draw(pool, W.public_rows(pub_rows()), seed=3, use_counts={})
    a, b = _answers(items, .8, "a"), _answers(items, .6, "b")
    se = W.paired_s_se(items, a, b, P51, 1)
    assert W.paired_s_se is v5.paired_s_se and v5.paired_s_se(items, a, b, P51, 1) == se       # public in the scorer module
    assert 0 < se < 5 and W.paired_s_se(items, a, b, P51, 1) == se                         # deterministic
    assert W.paired_s_se(items, a, a, P51, 1) == 0.0                                       # identical models: no difference at all
    assert W.paired_s_se(items, b, a, P51, 1) == pytest.approx(se)                         # symmetric


def test_panel_scaling_and_w_n(rows):
    pool = rows[0] + rows[1]
    items, _ = W.draw(pool, [], seed=3, use_counts={}, p=v5.V5Params(resamples=40))
    resp, base = _answers(items, .8, "m"), _answers(items, .5, "b")
    plain = v5.score_v5_detail(resp, items, base, v5.V5Params(resamples=40), 1)
    assert plain.s == pytest.approx(.6 * plain.t_eq + .4 * plain.n_eq) and plain.t_eq == plain.t
    panel = v5.PanelStats("v1", t=(10.0, 5.0), n=(20.0, 10.0))
    scaled = v5.score_v5_detail(resp, items, base, v5.V5Params(resamples=40, panel=panel), 1)
    assert scaled.t_eq == pytest.approx(v5.DI_MEAN + v5.DI_SD * (plain.t - 10) / 5)
    assert scaled.n_eq == pytest.approx(v5.DI_MEAN + v5.DI_SD * (plain.n - 20) / 10)
    ctx = W.build_context("d", items, {"c": resp, "b": base}, {"c": "b"}, [], 1, v5.V5Params(resamples=40, panel=panel, pub_items=0))
    assert ctx.parts["c"].t == pytest.approx(scaled.t_eq)                                  # the history numbers are the scaled ones


def test_config_reads_the_v51_settings():
    panel = {"pool_version": "v0", "T": {"mean": 20, "sd": 8}, "N": {"mean": 30, "sd": 9}}
    cfg = C.from_env({"REAX_SPEC_SEALED_DIR": "/pool", "REAX_SPEC_V5_POOL_DIR": "/v51", "REAX_SPEC_V5_W_N": "0",
                      "REAX_SPEC_V5_PANEL": json.dumps(panel)})
    assert (cfg.v5_pool_dir, cfg.v5_w_n) == ("/v51", 0.0) and cfg.v5_panel == v5.panel_from_json(panel)
    dflt = C.from_env({"REAX_SPEC_SEALED_DIR": "/pool"})
    assert (dflt.v5_pool_dir, dflt.v5_w_n, dflt.v5_panel) == ("", 0.4, None)
    for bad in ({"REAX_SPEC_V5_W_N": "1.5"}, {"REAX_SPEC_V5_PANEL": json.dumps({"pool_version": "v", "T": {"mean": 1, "sd": 0}})}):
        with pytest.raises(ValueError):
            C.from_env({"REAX_SPEC_SEALED_DIR": "/pool", **bad})


# ------------------------------------------------------------------------------------ end to end (evaluate_cycle)
BASE_REPO = "base/qwen"
DEFAULT_BASE_DIR = "Qwen__Qwen3-4B@" + "b" * 40


class Gen:
    """A generalist track on fake models: the real evaluator, FakeHub / FakeServer, the v5.1 pool in ``tmp_path/v51``."""

    def __init__(self, tmp_path, monkeypatch, rows, skills=None):
        from test_spec_evaluator import Env
        bases = tmp_path / "bases"
        write_model(str(bases / DEFAULT_BASE_DIR), make_weights(999), skill_key=BASE_REPO, category="generalist")
        self.env = env = Env(tmp_path, {BASE_REPO: 0.45, **(skills or {})}, sample=300, base_dir=str(bases))
        env.cfg.tracks = C._from_schedule()[-1:]
        env.cfg.x_tier_mode = "binding"
        env.cfg.g_med = {"generalist": 0.0}
        env.cfg.base_fetch = False
        monkeypatch.setattr(CS, "GENERALIST_G_MED", 0.0)
        monkeypatch.setattr(CS, "GENERALIST_COMMIT_FROM_UTC", CS.GENERALIST_START_UTC)
        monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", 0.0)
        monkeypatch.setitem(CS.KEEPALIVE_NONPROC_FLOOR, "generalist", 0.0)
        env.clock.t = CS.GENERALIST_START_UTC.timestamp() + 3600
        cats = []
        for category in S.GENERALIST_CATEGORIES:
            for i, row in enumerate(synthetic_pool(600)):
                row["id"], row["category"] = f"{category}-{row['id']}", category
                cats.append(row)
        write_pool(str(tmp_path / "sealed"), "generalist", cats)
        env.cfg.v5_pool_dir = write_v51(tmp_path / "v51", *rows)
        self.commits: dict[int, str] = {}
        self.block = 1000
        self.mp = monkeypatch

    @property
    def ts(self):
        return self.env.ev.state.track("generalist")

    def add(self, uid, repo, skill):
        self.env.add(uid, repo, skill, category="generalist")
        self.commits[uid] = repo

    def sub(self, uid):
        return next(s for s in self.ts.submissions.values() if s.uid == uid)

    async def cycle(self, advance=0.0):
        self.env.clock.advance(advance)
        self.block += 100
        snap = snapshot(self.block, [neuron(u, commit("generalist", r)) for u, r in self.commits.items()])
        return await self.env.ev.evaluate_cycle(snap)

    async def crown(self, uid, repo, skill):
        """Crown an entry under rules v4 (gate off): first draw, then the confirmation draw 13 h later."""
        self.add(uid, repo, skill)
        await self.cycle()
        assert self.sub(uid).status == ST.PENDING_CONFIRM, self.sub(uid).reason
        await self.cycle(13 * 3600)
        assert self.sub(uid).status == ST.CHAMPION, (self.sub(uid).status, self.sub(uid).reason)

    def gate_on(self):
        self.mp.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(self.env.clock.t - 60, timezone.utc))
        self.mp.setattr(E.SpecEvaluator, "_v5_params", lambda self_: FAST)
        self.mp.setattr(CS, "GENERALIST_START_UTC", CS.GENERALIST_START_UTC)

    def seat(self, champion_uid, *co_uids):
        """Seat the given entries as the opening co-champion set (rc11.post1 ``GENERALIST_CO_CHAMPIONS``): ``champion_uid`` holds
        the champion seat, the others are pending, copy-checked members."""
        for s in self.ts.submissions.values():
            if s.status == ST.CHAMPION:
                s.status = ST.FORMER
        champ = self.sub(champion_uid)
        self.ts.champion, champ.status = champ.key, ST.CHAMPION
        for u in co_uids:
            self.sub(u).status, self.sub(u).delta_checked = ST.PENDING_CONFIRM, True
        self.mp.setattr(CS, "GENERALIST_CO_CHAMPIONS", (champ.key, *(self.sub(u).key for u in co_uids)))


@pytest.fixture
def gen(tmp_path, monkeypatch, rows):
    return Gen(tmp_path, monkeypatch, rows)


def draw_parts(g, draw_id):
    pool = {r["id"]: r for r in [*g.rows_pool]}
    return Counter(pool[i]["part"] for i in g.ts.draws[draw_id].item_ids)


async def test_v51_draw_serves_champions_and_base_and_decides_on_s_v51(gen, rows):
    g = gen
    await g.crown(1, "a/old", .55)
    await g.crown(2, "a/co", .85)
    g.seat(1, 2)                                   # the weak entry holds the champion seat, the strong one is a co-champion
    g.gate_on()
    assert E.SpecEvaluator._v5_on(g.env.ev, "generalist", g.env.clock.t) and v5.EVALUATOR_WIRED
    g.add(3, "a/mid", .70)                         # beats the champion (.55) but not the best co-champion (.85)
    g.add(4, "a/top", .99)
    g.env.server.served.clear()
    await g.cycle()
    served = g.env.server.served
    assert sorted(served) == sorted(["a/old", "a/co", "a/mid", "a/top", BASE_REPO])        # W2: every model once, the base once
    mid, top = g.sub(3), g.sub(4)
    ts = g.ts
    assert top.status == ST.PENDING_CONFIRM and mid.status == ST.NOT_BETTER, (top.reason, mid.reason)
    # W3: decided on S_v5.1 against the best of the co-champions and the base
    first = next(h for h in top.history if h["kind"] == "first")
    assert first["scoring_regime"] == "v5" and W.scored_under_v5(top.by_tier) and W.scored_under_v5(g.sub(1).by_tier)
    assert first["ref_best"] == g.sub(2).key and first["incumbent"] == first["ref_totals"][g.sub(2).key]
    assert first["gain"] == pytest.approx(top.score - first["incumbent"]) and first["gain"] > 1.0
    assert first["s_v51"] == top.score and first["t"] is not None and first["n"] is not None
    assert first["gap_flag"] in (True, False) and "se_gap" in first and top.nonproc == top.score and set(top.by_tier) == {"T", "N"}
    mfirst = next(h for h in mid.history if h["kind"] == "first")
    assert mfirst["incumbent"] > mid.score and mfirst["ref_best"] == g.sub(2).key        # below the best co-champion, whatever the champion seat
    assert mid.score > g.sub(1).score                                                    # ... although it beats the seated champion
    # the draw: 540 T, 360 N and the public items (weight 0)
    pool = {r["id"]: r for r in [*rows[0], *rows[1], *pub_rows()]}
    d = ts.draws[top.draw1]
    assert Counter(pool[i].get("part", "pub") for i in d.item_ids) == {"T": 540, "N": 360, "pub": 90}
    assert ts.used[d.item_ids[0]] >= 1
    assert not any("gap_penalty" in h for h in top.history)
    # confirmation 13 h later on a disjoint draw: the challenger takes the title, the co-champions are former champions
    await g.cycle(13 * 3600)
    assert top.status == ST.CHAMPION and ts.champion == top.key and E.cochamp_members(ts) == []   # the set has ended
    assert g.sub(1).status == ST.FORMER
    conf = next(h for h in top.history if h["kind"] == "confirm")
    assert conf["scoring_regime"] == "v5" and not set(ts.draws[conf["draw"]].item_ids) & set(d.item_ids)


async def test_base_shared_by_two_entries_is_served_once_per_draw_and_cached(gen):
    g = gen
    await g.crown(1, "a/champ", .60)
    g.seat(1)
    g.gate_on()
    g.add(3, "a/one", .95)
    g.add(4, "a/two", .96)
    g.env.server.served.clear()
    await g.cycle()
    assert g.env.server.served.count(BASE_REPO) == 1 and sorted(g.env.server.served) == sorted(["a/champ", "a/one", "a/two", BASE_REPO])
    ev = g.env.ev
    assert len({k[1] for k in ev._v5_served}) == 1 and any(k[0] == E.V5W.base_key("Qwen/Qwen3-4B@" + "b" * 40) for k in ev._v5_served)


async def test_challenger_must_also_beat_its_base(gen):
    g = gen
    await g.crown(1, "a/champ", .50)
    g.seat(1)
    g.gate_on()
    g.env.server.skills[BASE_REPO] = .90            # the declared base is better than the champion
    g.add(3, "a/weak", .75)                         # beats the champion, not the base
    await g.cycle()
    h = next(x for x in g.sub(3).history if x["kind"] == "first")
    assert g.sub(3).status == ST.NOT_BETTER and h["ref_best"].startswith("base:") and h["gain"] < 0


async def test_unusable_co_champion_or_pool_waits_and_decides_nothing(gen, tmp_path):
    g = gen
    await g.crown(1, "a/champ", .60)
    await g.crown(2, "a/co", .70)
    g.seat(2, 1)
    g.gate_on()
    g.add(3, "a/new", .95)
    g.env.hub.unavailable.add("a/champ")           # a seated entry cannot be fetched now
    await g.cycle()
    assert g.sub(3).status == ST.QUEUED and not g.sub(3).history
    g.env.hub.unavailable.clear()
    os.chmod(os.path.join(g.env.cfg.v5_pool_dir, "v51.T.jsonl"), 0o644)      # the pool is readable by others: no draw
    await g.cycle(C.RETRY_UNAVAILABLE_S + 1)
    assert g.sub(3).status == ST.QUEUED and not g.sub(3).history
    assert any("refill the sealed pool" in e for e in g.ts.events), g.ts.events[-8:]
    os.chmod(os.path.join(g.env.cfg.v5_pool_dir, "v51.T.jsonl"), 0o600)
    await g.cycle(C.RETRY_UNAVAILABLE_S + 1)
    assert g.sub(3).status == ST.PENDING_CONFIRM


# ----------------------------------------------------------------------------------------- gate off = rc11
@pytest.mark.parametrize("gate", ["off", "armed"])
async def test_gate_off_never_touches_v51(gen, monkeypatch, gate):
    """With ``EVAL_V5_START_UTC = None`` (or the armed release gate, not reached on this clock) the v5.1 code is never reached:
    no v5.1 draw, no base served, v4 state only."""
    from conftest import RELEASE_EVAL_V5_START_UTC
    g = gen
    assert S.EVAL_V5_START_UTC is None
    if gate == "armed":
        monkeypatch.setattr(S, "EVAL_V5_START_UTC", RELEASE_EVAL_V5_START_UTC)

    def boom(*a, **k):
        raise AssertionError("v5.1 code reached with the gate off")

    for name in ("draw", "load_pool", "flatten", "build_context", "responses"):
        monkeypatch.setattr(W, name, boom)
    monkeypatch.setattr(E.SpecEvaluator, "_v5_bases", boom)
    await g.crown(1, "a/champ", .60)
    g.add(2, "a/new", .95)
    g.env.server.served.clear()
    await g.cycle()
    assert BASE_REPO not in g.env.server.served and sorted(g.env.server.served) == ["a/champ", "a/new"]
    sub = g.sub(2)
    assert not W.scored_under_v5(sub.by_tier) and not W.gap_flagged(sub.history)
    assert all("scoring_regime" not in h and "s_v51" not in h for s in g.ts.submissions.values() for h in s.history)
    d = g.ts.draws[sub.draw1]
    assert len(d.item_ids) == 300 and set(d.item_categories.values()) == set(S.GENERALIST_CATEGORIES)    # a v4 draw: 30 per category
    assert g.env.ev._v5_ctx is None and not g.env.ev._v5_served
    assert not S.v5_active(g.env.clock.t)


def test_gate_off_with_a_set_but_future_gate_is_v4(monkeypatch):
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime(2030, 1, 1, tzinfo=timezone.utc))
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)
    assert not ev._v5_on("generalist", 1.0e9) and not ev._v5_on("demand_forecasting", 2.0e9) and ev._v5_on("generalist", 2.0e9)


def test_no_new_state_fields_and_the_v5_marker_survives_a_reload(tmp_path):
    assert not {"scoring_regime", "gap_flag", "cochampions"} & {f.name for f in __import__("dataclasses").fields(ST.Submission)}
    assert "cochampions" not in {f.name for f in __import__("dataclasses").fields(ST.TrackState)}
    st = ST.SpecState()
    sub = ST.Submission("hk|c", 1, "hk", "ck", "generalist", "a/m", "r" * 40, 5, 1.0)
    st.track("generalist").submissions[sub.key] = sub
    sub.by_tier = {"T": 24.0, "N": 20.0}
    sub.history.append({"kind": "first", "scoring_regime": "v5", "gap_flag": True})
    path = str(tmp_path / "s.json")
    ST.save(st, path)
    back = ST.load(path).track("generalist").submissions[sub.key]
    assert W.scored_under_v5(back.by_tier) and W.gap_flagged(back.history)
    assert not W.scored_under_v5({"proc_sealed": 50.0, "new_family": 40.0}) and not W.scored_under_v5({}) and not W.gap_flagged([{"kind": "x"}])


# ------------------------------------------------------------------ W4: D4 provisional half keep-alive from the commitment
def _d4_world(monkeypatch, gate_on, *, commit_age_blocks=1800, registered_age_blocks=500, track=None):
    """One queued, copy-checked, unscored entry committed ``commit_age_blocks`` blocks ago, registered
    ``registered_age_blocks`` blocks ago (its immunity of 7200 blocks is far from over)."""
    from test_keepalive import DF, NOW
    from test_provisional_ka import BLOCK, _p, _snapr
    if gate_on:
        monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(NOW - 3600, timezone.utc))
    st = ST.SpecState()
    s = _p(st, 1, since=None)
    s.provisional_since_ts = None
    snap = _snapr([s], regs={1: BLOCK - registered_age_blocks}, cblocks={1: BLOCK - commit_age_blocks})
    return st, s, snap, NOW, DF


@pytest.mark.usefixtures("rc5_entry_floors")
def test_d4_gate_off_is_rc11_no_slot_before_the_immunity_lead(monkeypatch):
    st, s, snap, now, df = _d4_world(monkeypatch, False)
    assert not E.mark_provisional(st, [df], snap, now) and s.provisional_since_ts is None       # immunity far from its end: nothing
    assert E.compute_provisional(st, [df], snap, now=now)[df] == []


@pytest.mark.usefixtures("rc5_entry_floors")
def test_d4_slot_starts_at_the_onchain_commitment(monkeypatch):
    st, s, snap, now, df = _d4_world(monkeypatch, True, commit_age_blocks=1800)               # committed 1800 x 12 s = 6 h ago
    assert E.mark_provisional(st, [df], snap, now)
    assert s.provisional_since_ts == pytest.approx(now - 6 * 3600)                              # not "now": the commitment block
    assert E.compute_provisional(st, [df], snap, now=now)[df] == [1]                            # paid although the immunity has 6700 blocks left
    assert E.compute_provisional(st, [df], snap, now=now + 5.9 * 3600)[df] == [1]
    assert E.compute_provisional(st, [df], snap, now=now + 6.1 * 3600)[df] == []                # 12 h after the commitment, not after the check


@pytest.mark.usefixtures("rc5_entry_floors")
def test_d4_old_commitment_has_no_time_left_and_unknown_block_counts_from_now(monkeypatch):
    st, s, snap, now, df = _d4_world(monkeypatch, True, commit_age_blocks=3700)               # 12.3 h ago: the half is over
    E.mark_provisional(st, [df], snap, now)
    assert E.compute_provisional(st, [df], snap, now=now)[df] == []
    st2, s2, snap2, now, df = _d4_world(monkeypatch, True, commit_age_blocks=0)                # commitment block unknown (0)
    E.mark_provisional(st2, [df], snap2, now)
    assert s2.provisional_since_ts == now and E.compute_provisional(st2, [df], snap2, now=now)[df] == [1]
    assert E._commitment_ts(type("N", (), {"commitment_block": 10 ** 9})(), 100, now) == now   # a block in the future is not trusted


@pytest.mark.usefixtures("rc5_entry_floors")
@pytest.mark.parametrize("status", [ST.NOT_BETTER, ST.REJECTED, ST.WITHDRAWN])
def test_d4_slot_is_revoked_when_the_first_draw_fails(monkeypatch, status):
    st, s, snap, now, df = _d4_world(monkeypatch, True)
    E.mark_provisional(st, [df], snap, now)
    assert E.compute_provisional(st, [df], snap, now=now)[df] == [1]
    s.status, s.score = status, 12.0                      # the first draw did not pass (scored, not better)
    assert E.compute_provisional(st, [df], snap, now=now)[df] == []
    s.status, s.score = ST.QUEUED, None
    assert E.compute_provisional(st, [df], snap, now=now)[df] == [1]


@pytest.mark.usefixtures("rc5_entry_floors")
def test_d4_close_call_keeps_the_slot_until_its_verdict_and_a_recommit_does_not_restart_it(monkeypatch):
    from test_provisional_ka import _p
    st, s, snap, now, df = _d4_world(monkeypatch, True)
    E.mark_provisional(st, [df], snap, now)
    s.score, s.close_draws = 20.0, ["d1"]                  # close call: undecided
    assert E.compute_provisional(st, [df], snap, now=now)[df] == [1]
    first = s.provisional_since_ts
    s2 = _p(st, 1, since=None, repo="m/second")            # the same hotkey commits again: it inherits the window start
    s2.provisional_since_ts = None
    E.mark_provisional(st, [df], snap, now + 3600)
    assert s2.provisional_since_ts in (None, first)


def test_d4_pay_is_half_the_median():
    from reax_subnet import competition_split as cs
    assert cs.PROVISIONAL_KA_FRACTION == 0.5 and cs.PROVISIONAL_KA_MAX_HOURS == 12.0     # "half", unchanged by D4


def test_queue_prepares_the_entry_whose_immunity_ends_soonest_first():
    from reax_subnet.chain import Neuron
    subs, ns = [], []
    block = 20000
    for uid, reg in ((1, block - 100), (2, block - 7000), (3, block - 3000), (4, block - 8000), (5, 0)):
        s = ST.Submission(f"hk{uid}|c", uid, f"hk{uid}", "ck", "generalist", "a/m", "r" * 40, uid, 1.0)
        subs.append(s)
        ns.append(Neuron(uid, f"hk{uid}", "ck", False, 0.0, None, None, commitment="", registered_block=reg))
    order = [s.uid for s in E.queue_order(subs, snapshot(block, ns))]
    assert order == [4, 2, 3, 1, 5]        # past its immunity (4: ended 800 blocks ago) first, then by end: 2, 3, 1; unknown last


async def test_d4_queue_fetches_in_immunity_order_in_a_v51_cycle(gen):
    g = gen
    await g.crown(1, "a/champ", .60)
    g.seat(1)
    g.gate_on()
    g.add(3, "a/late", .95)
    g.add(4, "a/soon", .96)
    g.env.hub.fetches.clear()
    g.block = 20000
    regs = {3: g.block - 100, 4: g.block - 7000}            # uid 4's immunity ends first
    snap = snapshot(g.block, [neuron(u, commit("generalist", r), registered_block=regs.get(u, g.block - 9000)) for u, r in g.commits.items()])
    await g.env.ev.evaluate_cycle(snap)
    fetched = [r for r in g.env.hub.fetches if r in ("a/late", "a/soon")]
    assert fetched == ["a/soon", "a/late"]


# ----------------------------------------------------------- v4-scale floors do not judge a DI-scale S_v5.1
def test_v4_floors_do_not_drop_a_v51_scored_entry(monkeypatch):
    from test_cochampions import case, leaders
    st = case(monkeypatch)
    ts = st.track(next(iter(st.tracks)))
    assert getattr(leaders(st), "cochamps", None)                       # the three opening co-champions are listed
    for s in ts.submissions.values():
        s.score = 24.0                       # a DI-scale score, far below the v4 entry floor
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, ts.track_id, 36.8)
    assert not getattr(leaders(st), "cochamps", None)                   # scored under v4 and below the floor: dropped (rc11 behaviour)
    for s in ts.submissions.values():
        s.by_tier = {"T": 24.0, "N": 20.0}
    assert not getattr(leaders(st), "cochamps", None)                   # the marker alone does nothing while the gate is closed (F2)
    from test_cochampions import NOW as CNOW
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(CNOW - 60, timezone.utc))
    assert len(leaders(st).cochamps) == 3                               # re-scored under v5.1 and the gate open: the thirds stay


def test_keepalive_floors_skip_v51_scored_entries(monkeypatch):
    from test_keepalive import DF, NOW, _sub, _snap
    st = ST.SpecState()
    s = _sub(st, 1, ST.PENDING_CONFIRM, 24.0, track="generalist", nonproc=24.0)
    snap = _snap([s])
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", 36.8)
    monkeypatch.setitem(CS.KEEPALIVE_NONPROC_FLOOR, "generalist", 37.4)
    assert E.compute_keepalive(st, ["generalist"], snap, now=NOW)["generalist"] == []            # a v4 score of 24: below the floors
    s.by_tier = {"T": 24.0, "N": 20.0}
    assert E.compute_keepalive(st, ["generalist"], snap, now=NOW)["generalist"] == []            # marker without the gate (F2)
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(NOW - 60, timezone.utc))
    assert [u for u, _ in E.compute_keepalive(st, ["generalist"], snap, now=NOW)["generalist"]] == [1]


def test_share_layer_does_not_apply_the_v4_generalist_floor_to_s_v51(monkeypatch):
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, "generalist", 36.8)
    at = CS.GENERALIST_START_UTC.timestamp() + 3600
    tracks = {"generalist": CS.RankedLeaders([(10, 24.5), (11, 23.0)])}
    keep = {"generalist": [(12, 22.0)]}
    kw = dict(now=at, keepalive=keep)
    off = CS.combine_weights({1: 1.0}, None, 0, tracks, **kw)
    assert 10 not in off and 11 not in off and 12 not in off                  # gate off (rc11): below the v4 floor, unpaid
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", datetime.fromtimestamp(at - 60, timezone.utc))
    on = CS.combine_weights({1: 1.0}, None, 0, tracks, **kw)
    assert on[10] > on[11] > 0 and 12 in on                                    # v5 active: ranked and kept alive on S_v5.1
    explicit = CS.combine_weights({1: 1.0}, None, 0, tracks, floors={"generalist": 36.8}, **kw)
    assert 10 not in explicit                                                  # an explicitly passed floor is respected
    assert CS.TRACK_ENTRY_FLOOR["generalist"] == 36.8


async def test_declared_bases_count_against_the_hourly_serving_budget(gen):
    g = gen
    await g.crown(1, "a/champ", .60)
    g.seat(1)
    g.gate_on()
    g.env.cfg.max_evals_per_hour = 3                        # champion + one challenger + its base; not two challengers
    g.add(3, "a/one", .95)
    g.add(4, "a/two", .96)
    g.env.server.served.clear()
    await g.cycle(3 * 3600)                                  # the crowning serves have left the rolling hour
    served = g.env.server.served
    assert sorted(served) == sorted(["a/champ", "a/one", BASE_REPO])                      # the second challenger waits for the next hour
    assert g.sub(3).status == ST.PENDING_CONFIRM and g.sub(4).status == ST.QUEUED


async def test_v51_crowning_waits_for_the_crown_slot(gen, monkeypatch):
    """Crowns only at crowning slots (rc11.post1 code): a v5.1 decision is stored and takes effect at the slot."""
    g = gen
    await g.crown(1, "a/champ", .60)
    g.seat(1)
    g.gate_on()
    g.add(3, "a/top", .99)
    await g.cycle()
    assert g.sub(3).status == ST.PENDING_CONFIRM
    slot = datetime.fromtimestamp(g.env.clock.t + 20 * 3600, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot,))
    await g.cycle(13 * 3600)                                   # the confirmation passes before the slot
    top = g.sub(3)
    assert top.status == ST.PENDING_CONFIRM and any(h["kind"] == "crown_pending" for h in top.history)
    assert g.ts.champion == g.sub(1).key and "takes effect" in top.reason
    assert next(h for h in top.history if h["kind"] == "confirm")["scoring_regime"] == "v5"
    await g.cycle(8 * 3600)                                    # past the slot
    assert top.status == ST.CHAMPION and g.ts.champion == top.key and g.sub(1).status == ST.FORMER


# ------------------------------------------------------------------ transition rule A (Florian, 22 Oct): crown at the slot, then v5.1
async def _rule_a_world(g, monkeypatch):
    """Champion 1 (seated, v4). Entry 3 passes its v4 first stage, entry 4 a later one. The crown slot = the v5.1 start, 16 h on.
    Returns the slot (unix seconds). The gate is in the future: every draw until the slot is v4."""
    await g.crown(1, "a/old", .60)
    g.seat(1)
    t0 = g.env.clock.t
    slot = datetime.fromtimestamp(t0 + 16 * 3600, timezone.utc)
    monkeypatch.setattr(CS, "CROWN_SLOTS_UTC", (slot,))
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", slot)
    monkeypatch.setattr(E.SpecEvaluator, "_v5_params", lambda self_: FAST)
    g.add(3, "a/early", .90)
    await g.cycle()
    assert g.sub(3).status == ST.PENDING_CONFIRM
    g.add(4, "a/late", .99)
    await g.cycle(5 * 3600)
    assert g.sub(4).status == ST.PENDING_CONFIRM and g.sub(4).confirm_after_ts > slot.timestamp()
    await g.cycle(8 * 3600)                                            # t0 + 13 h: entry 3 confirms under v4, before the slot
    return slot.timestamp()


async def test_rule_a_v4_confirmation_before_the_flip_is_crowned_at_the_slot_then_v5(gen, monkeypatch):
    g = gen
    slot = await _rule_a_world(g, monkeypatch)
    early, late, old = g.sub(3), g.sub(4), g.sub(1)
    assert early.status == ST.PENDING_CONFIRM and any(h["kind"] == "crown_pending" for h in early.history)
    assert g.ts.champion == old.key
    assert next(h for h in early.history if h["kind"] == "confirm").get("scoring_regime") is None        # decided under v4
    assert late.status == ST.PENDING_CONFIRM and late.draw1                                                # (b) still a v4 stage
    old_draw1 = late.draw1
    await g.cycle(slot - 1 - g.env.clock.t)                                                                 # one second before the flip
    assert g.env.clock.t == slot - 1 and early.status == ST.PENDING_CONFIRM and late.draw1 == old_draw1
    assert not any(h["kind"] == "v5_restart" for h in late.history)
    # (d) exactly at 14:30:00: crown first, then v5.1
    await g.cycle(1.0)
    assert g.env.clock.t == slot
    assert early.status == ST.CHAMPION and g.ts.champion == early.key and old.status == ST.FORMER
    crowned = [h for h in early.history if h["kind"] == "crown_done"]
    assert len(crowned) == 1 and crowned[0]["ts"] == slot
    assert sum(h["kind"] == "confirm" for h in early.history) == 1                     # the crowning is not re-judged under v5.1
    assert "scoring_regime" not in next(h for h in early.history if h["kind"] == "confirm")
    # (b) the unconfirmed v4 stage restarted and took a v5.1 first stage in the same cycle (log + history row)
    # (rc11.post4, slot rules s.2: the crowning at the flip restarts it first; rule A then has nothing left to restart)
    assert [h["kind"] for h in late.history if h["kind"] in ("v5_restart", "slot_restart")] == ["slot_restart"]
    assert next(h for h in late.history if h["kind"] == "slot_restart")["was"] == ST.PENDING_CONFIRM
    assert not any(h["kind"] == "confirm" for h in late.history)                       # never confirmed under v4
    first = [h for h in late.history if h["kind"] == "first"]
    assert first[0].get("scoring_regime") is None and first[-1]["scoring_regime"] == "v5"
    assert late.draw1 != old_draw1 and late.status == ST.PENDING_CONFIRM and W.scored_under_v5(late.by_tier)
    # (c) the bar: the CURRENT title holder only (the new rule-A champion) and the challenger's own base; the title holder it
    # replaced is a former champion and no longer part of it
    assert E.cochamp_members(g.ts) == []                                               # the rc11 set has ended (payments untouched)
    assert E.v5_bar_members(g.ts, slot) == []
    ref = first[-1]["ref_totals"]
    assert set(ref) == {early.key, W.base_key(late.base_model)} and old.key not in ref
    assert first[-1]["incumbent"] == max(ref.values())
    # its confirmation (v5.1, 13 h later) crowns a v5.1 champion and ends the bar for good
    await g.cycle(13 * 3600)
    assert late.status == ST.CHAMPION and E.v5_bar_members(g.ts, g.env.clock.t) == []
    assert early.status == ST.FORMER


async def test_rule_a_without_the_gate_nothing_restarts(gen, monkeypatch):
    g = gen
    await _rule_a_world(g, monkeypatch)
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", None)
    late = g.sub(4)
    g.env.ev._v5_restart_v4_stages(g.ts, g.env.clock.t + 10 * 86400)
    assert late.status == ST.PENDING_CONFIRM and not any(h["kind"] == "v5_restart" for h in late.history)
    assert E.v5_bar_members(g.ts, g.env.clock.t + 10 * 86400) == E.cochamp_members(g.ts)    # gate off: the rc11 set, nothing more


async def test_rule_a_a_v5_stage_or_a_stored_crowning_is_not_restarted(gen, monkeypatch):
    g = gen
    slot = await _rule_a_world(g, monkeypatch)
    early = g.sub(3)
    g.env.ev._v5_restart_v4_stages(g.ts, slot + 1)                     # entry 3 has a stored crowning: untouched
    assert early.status == ST.PENDING_CONFIRM and not any(h["kind"] == "v5_restart" for h in early.history)
    await g.cycle(slot - g.env.clock.t)
    late = g.sub(4)
    n = sum(h["kind"] == "v5_restart" for h in late.history)
    assert n == 1
    g.env.ev._v5_restart_v4_stages(g.ts, slot + 5)                     # idempotent: its stage is v5.1 now
    assert sum(h["kind"] == "v5_restart" for h in late.history) == 1 and late.status == ST.PENDING_CONFIRM
