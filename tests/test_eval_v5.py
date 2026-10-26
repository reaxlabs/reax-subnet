"""Eval v5 (docs/EVAL-V5.md): formulas with hand-computed numbers, code paths, the draw, the gate and the migration rule."""
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from reax_subnet.spec import evaluator as E, prompt as P, sampling as S, scoring as SC, v5

T = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
CATS = S.GENERALIST_CATEGORIES


# ------------------------------------------------------------------------------------------------ fixtures
NC = v5.V5Params(const_baseline=False)   # constant-gold fixtures below: the best-constant zero point has its own tests


def onehot(keys, key):
    return {k: (1.0 if k == key else 0.0) for k in keys}


def q(keys, expected, idk=False):
    return {"instructions": "pick", "criteria": {k: k.upper() for k in keys}, "expected": expected}


def mirror(iid, *, ttype="t1", area="tools", part="T", gold=False, questions=None, idk=False):
    return {"id": iid, "tier": "mirror", "part": part, "area": area, "task_type": ttype, "gold": gold, "state": {"x": 1},
            "idk": idk, "questions": questions or [q(["a", "b"], "a")]}


def mscore(table, w=None, params=NC, details=None):
    """(None, {task type: score}) -- the type scores of an item table."""
    return None, v5.type_scores(table, w, params, details)


def answers(items, right):
    """One-hot answers: item i is answered with its gold key if right[i] else with a wrong key."""
    out = {}
    for it, ok in zip(items, right):
        res = []
        for qq in it["questions"]:
            keys = list(qq["criteria"]) + ([v5.IDK_KEY] if it.get("idk") and v5.IDK_KEY not in qq["criteria"] else [])
            gold = str(qq["expected"])
            pick = gold if ok else next(k for k in keys if k != gold and k != v5.IDK_KEY)
            res.append(onehot(keys, pick))
        out[it["id"]] = res[0] if len(res) == 1 else res
    return out


def held(iid, tier="new_family", cat="math_quant"):
    return {"id": iid, "tier": tier, "category": cat, "question": {"type": "choice"}, "labels": ["a", "b"], "expected": "a"}


def held_answers(rows, right):
    return {r["id"]: onehot(["a", "b"], "a" if ok else "b") for r, ok in zip(rows, right)}


def typed(prefix, n, *, ttype, area="tools", part="T", gold=False):
    """n two-option items with balanced gold keys (best constant answer = 50 %): right-answer score = 100 (2a - 1)."""
    return [mirror(f"{prefix}{i}", ttype=ttype, area=area, part=part, gold=gold, questions=[q(["a", "b"], "ab"[i % 2])])
            for i in range(n)]


# ------------------------------------------------------------------------------------- codes and prompts
def test_option_codes():
    assert v5.option_codes(3) == ["A", "B", "C"]
    c26 = v5.option_codes(26)
    assert c26 == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    c27 = v5.option_codes(27)
    assert c27[:26] == c26 and c27[26] == "AA"
    c255 = v5.option_codes(255)
    assert len(set(c255)) == 255 and c255[26 + 27] == "BB" and c255[-1] == "IU"   # index 254: j = 228 = 8 x 26 + 20
    for bad in (1, 256):
        with pytest.raises(P.PromptError):
            v5.option_codes(bad)


def test_render_is_v4_byte_for_byte_up_to_26_options():
    question = {"type": "choice", "instructions": "which?", "criteria": {"x": "ex", "y": "why", "z": 3}}
    old, new = P.render({"s": 1}, question), v5.render_question({"s": 1}, question)
    assert (new.system, new.user, new.codes, new.keys) == (old.system, old.user, old.letters, old.keys)


def test_render_many_options_and_idk():
    crit = {f"k{i}": f"opt {i}" for i in range(30)}
    r = v5.render_question({}, {"instructions": "?", "criteria": crit})
    assert r.codes[26] == "AA" and "AA: opt 26" in r.user and r.user.endswith("Answer with one option code only.")
    assert "one option code" in r.system
    r2 = v5.render_question({}, {"instructions": "?", "criteria": {"a": "x", "b": "y"}}, idk=True)
    assert r2.keys == ("a", "b", "idk") and "C: I don't know" in r2.user


def test_render_joint_has_one_marker_per_question_and_joint_probs():
    item = mirror("m", questions=[q(["a", "b"], "a"), q(["a", "b", "c"], "b")])
    text, per = v5.render_joint(item)
    assert text.count(v5.JOINT_MARKER) == 2 and "QUESTION 2:" in text and [len(p.codes) for p in per] == [2, 3]
    # hidden [2 questions, H=1]; readout weight [codes=3, H=1]; logits q1 = [2, 0], q2 = [0, 4, 0]
    w = np.array([[1.0], [0.0], [0.0]])
    out = v5.joint_probs(item, np.array([[2.0], [0.0]]), w)
    assert out[0]["a"] == pytest.approx(1 / (1 + math.exp(-1)))      # softmax at temperature 2 of [2, 0]
    assert out[1] == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    with pytest.raises(ValueError):
        v5.joint_probs(item, np.array([[1.0]]), w)
    assert [r["id"] for r in v5.causal_rows(item)] == ["m#0", "m#1"]


# ------------------------------------------------------------------------------------- item outcomes
def test_question_outcome():
    yn = q(["yes", "no"], "yes")
    assert v5.question_outcome(yn, {"yes": 0.51, "no": 0.49}) == ("right", "yes")
    assert v5.question_outcome(yn, {"yes": 0.49, "no": 0.51}) == ("wrong", "no")
    ch = q(["a", "b", "c"], "b")
    assert v5.question_outcome(ch, {"a": 0.2, "b": 0.5, "c": 0.3}) == ("right", "b")
    assert v5.question_outcome(ch, {"a": 0.4, "b": 0.4, "c": 0.2}) == ("tie", None)        # exact tie: no single prediction
    assert v5.question_outcome(ch, {"a": 1.0}) == ("wrong", None)                          # unusable = wrong
    assert v5.question_outcome(ch, None) == ("wrong", None)
    assert v5.question_outcome(q(["a", "b"], "a"), {"a": 0.2, "b": 0.2, "idk": 0.6}, idk=True) == ("idk", "idk")
    assert v5.question_outcome(q(["a", "b"], "a"), {"a": 0.2, "b": 0.2, "idk": 0.6}) == ("wrong", None)   # no idk offered


def test_s3_exact_ties_earn_one_over_k_credit_if_gold_is_among_them():
    cr = lambda qq, pr, idk=False: v5.question_credit(qq, pr, idk).right   # noqa: E731
    abc = lambda gold: q(["a", "b", "c"], gold)   # noqa: E731
    assert cr(abc("a"), {"a": 0.4, "b": 0.4, "c": 0.2}) == 0.5 and cr(abc("b"), {"a": 0.4, "b": 0.4, "c": 0.2}) == 0.5
    assert cr(abc("c"), {"a": 0.4, "b": 0.4, "c": 0.2}) == 0.0                              # gold not among the tied top options
    third = 1 / 3
    assert cr(abc("c"), {"a": third, "b": third, "c": third}) == pytest.approx(third)      # k = 3
    assert cr(abc("a"), {"a": 0.5, "b": 0.3, "c": 0.2}) == 1.0 and cr(abc("b"), {"a": 0.5, "b": 0.3, "c": 0.2}) == 0.0
    yes_no = lambda gold: q(["yes", "no"], gold)   # noqa: E731
    for gold in ("yes", "no"):                      # yes/no at exactly 0.5 is a 2-way tie
        assert cr(yes_no(gold), {"yes": 0.5, "no": 0.5}) == 0.5
    assert cr(yes_no("yes"), {"yes": 0.5000001, "no": 0.4999999}) == 1.0
    # the key order does not matter (no first-key advantage)
    assert cr(q(["z", "a"], "z"), {"z": 0.5, "a": 0.5}) == cr(q(["z", "a"], "a"), {"z": 0.5, "a": 0.5}) == 0.5


def test_mirror_item_tie_splits_credit_without_first_key_advantage():
    for gold in ("a", "z"):
        item = mirror("tie", questions=[q(["z", "a"], gold)])
        _, parts = mscore(v5.item_table([item], {"tie": {"z": 0.5, "a": 0.5}}))
        assert parts["t1"] == pytest.approx(100 * (0.5 - 0.5) / 0.5)   # credit 1/2 = chance 1/2 -> 0
    items = [mirror(f"t{i}", questions=[q(list("abcd"), "a")]) for i in range(2)]
    ans = {"t0": {"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0}, "t1": onehot(list("abcd"), "a")}
    _, parts = mscore(v5.item_table(items, ans))
    assert parts["t1"] == pytest.approx(100 * ((0.5 + 1) / 2 - 0.25) / 0.75)
    # a multi-question item: credits multiply (1/2 x 1 = 1/2)
    both = mirror("two", questions=[q(["a", "b"], "a"), q(["a", "b"], "a")])
    t = v5.item_table([both], {"two": [{"a": 0.5, "b": 0.5}, onehot(["a", "b"], "a")]})
    assert t.correct[0] == 0.5 and t.wrong[0] == 0.5


def test_held_out_choice_ties_use_the_same_rule_and_v4_scoring_is_unchanged():
    from reax_subnet.spec import scoring as SC
    row = held("h")
    tie = {"a": 0.5, "b": 0.5}
    assert not hasattr(SC.score_item(row, tie), "credit") and SC.score_item(row, tie).correct is True    # v4: first label wins
    assert v5.pool_scores([row], {"h": tie})[0].credit == 0.5
    assert v5.pool_scores([dict(row, expected="b")], {"h": tie})[0].credit == 0.5
    assert v5.pool_scores([row], {"h": {"a": 0.7, "b": 0.3}})[0].credit == 1.0
    assert v5.pool_scores([row], {"h": None})[0].credit == 0.0


# ------------------------------------------------------------------------------------- components
def test_generic_component_is_chance_corrected():
    items = [mirror(f"i{i}", area="tools", questions=[q(list("abcd"), "a")]) for i in range(4)]
    m, parts = mscore(v5.item_table(items, answers(items, [1, 1, 1, 0])))
    assert parts["t1"] == pytest.approx(100 * (0.75 - 0.25) / 0.75)
    assert mscore(v5.item_table(items, answers(items, [0, 0, 0, 0])))[1]["t1"] == 0.0   # clipped at 0


def _omni(c, i, n_idk, total=10):
    """Omniscience items answered c right, i wrong, the rest IDK (4 options + IDK)."""
    items = [mirror(f"o{k}", ttype="omniscience", idk=True, questions=[q(list("abcd"), "a")]) for k in range(total)]
    ans = {it["id"]: onehot(["a", "b", "c", "d", "idk"], "a" if k < c else "b" if k < c + i else "idk") for k, it in enumerate(items)}
    return items, ans


def test_s1_omniscience_aa_index_default():
    omni = lambda items, ans, params=v5.DEFAULT: mscore(v5.item_table(items, ans), None, params)[1]["omniscience"]   # noqa: E731
    assert omni(*_omni(0, 0, 10)) == pytest.approx(50.0)       # always IDK
    assert omni(*_omni(6, 3, 1)) == pytest.approx(65.0)        # 60 / 30 / 10
    assert omni(*_omni(10, 0, 0)) == pytest.approx(100.0)      # perfect
    assert omni(*_omni(0, 10, 0)) == pytest.approx(0.0)        # always wrong
    assert omni(*_omni(0, 0, 10)) < omni(*_omni(6, 3, 1))      # abstaining no longer beats an honest 60 / 30 / 10 model
    items, ans = _omni(5, 3, 2)
    assert omni(items, ans) == pytest.approx(50 * (1 + 0.5 - 0.3))
    # blind guessing among 4 options: 25 % right, 75 % wrong -> 25
    blind = [mirror(f"b{k}", ttype="omniscience", idk=True, questions=[q(list("abcd"), "a")]) for k in range(4)]
    bans = {it["id"]: onehot(["a", "b", "c", "d", "idk"], "abcd"[k]) for k, it in enumerate(blind)}
    assert omni(blind, bans) == pytest.approx(25.0)
    # an unusable answer is wrong, never IDK
    assert omni(blind[:1], {"b0": None}) == pytest.approx(0.0)
    # an exact tie of gold with IDK: 1/2 right + 1/2 IDK -> (100 + 50 - 0) / 2
    assert omni(blind[:1], {"b0": {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.0, "idk": 0.5}}) == pytest.approx(75.0)
    # an exact tie of gold with a wrong answer: 1/2 right, 1/2 wrong -> 50 (1 + 0.5 - 0.5)
    assert omni(blind[:1], {"b0": {"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0, "idk": 0.0}}) == pytest.approx(50.0)


def test_s1_omniscience_half_half_stays_selectable():
    items, ans = _omni(5, 3, 2)
    hh = v5.V5Params(omniscience_mode="half_half")
    assert mscore(v5.item_table(items, ans), None, hh)[1]["omniscience"] == pytest.approx(100 * (0.5 * 0.5 + 0.5 * 2 / 5))   # 45
    always_idk = _omni(0, 0, 10)
    assert mscore(v5.item_table(*always_idk), None, hh)[1]["omniscience"] == pytest.approx(50.0)
    p2 = v5.V5Params(omniscience_mode="half_half", omni_acc_weight=0.8)
    assert mscore(v5.item_table(items, ans), None, p2)[1]["omniscience"] == pytest.approx(100 * (0.8 * 0.5 + 0.2 * 0.4))
    # omni_acc_weight does not matter for the default AA index
    assert (mscore(v5.item_table(items, ans), None, v5.V5Params(omni_acc_weight=0.9))[1]["omniscience"]
            == mscore(v5.item_table(items, ans))[1]["omniscience"])


def test_wildguard_mean_of_three_f1():
    yn = lambda gold: q(["yes", "no"], gold)   # noqa: E731
    gold = [("yes", "yes", "no"), ("yes", "no", "no"), ("no", "no", "no"), ("no", "no", "no")]
    pred = [("yes", "yes", "no"), ("no", "no", "no"), ("yes", "no", "no"), ("no", "no", "no")]
    items = [mirror(f"w{i}", ttype="wildguard", questions=[yn(g) for g in gs]) for i, gs in enumerate(gold)]
    ans = {it["id"]: [onehot(["yes", "no"], p) for p in ps] for it, ps in zip(items, pred)}
    # q1: tp 1, fp 1, fn 1 -> 0.5; q2: tp 1 -> 1; q3: no positives at all -> 0 (zero-division convention)
    _, parts = mscore(v5.item_table(items, ans), params=v5.V5Params(wildguard_mode="raw"))
    assert parts["wildguard"] == pytest.approx(100 * (0.5 + 1 + 0) / 3)


def _wg(gold_rows, pred_rows):
    items = [mirror(f"wc{i}", ttype="wildguard", questions=[q(["yes", "no"], g) for g in gs])
             for i, gs in enumerate(gold_rows)]
    ans = {it["id"]: [onehot(["yes", "no"], p) for p in ps] for it, ps in zip(items, pred_rows)}
    return items, ans


def _wg_score(items, ans, **kw):
    extra = {}
    sc = mscore(v5.item_table(items, ans), None, v5.V5Params(**kw), extra)[1]["wildguard"]
    return sc, extra.get("wildguard")


def test_s2_wildguard_const_corrected_default():
    # 4 items, gold yes in 3 of them for every label: P = 3, N = 4 -> always-yes F1 = 2*3/(3+4) = 6/7
    gold = [("yes",) * 3] * 3 + [("no",) * 3]
    f1_const = 6 / 7
    always_yes = [("yes",) * 3] * 4
    always_no = [("no",) * 3] * 4
    perfect = gold
    sc, ex = _wg_score(*_wg(gold, always_yes))
    assert sc == pytest.approx(0.0) and ex["raw_f1"] == pytest.approx(100 * f1_const) and ex["f1_const"] == pytest.approx(100 * f1_const)
    assert _wg_score(*_wg(gold, always_no))[0] == pytest.approx(0.0)
    assert _wg_score(*_wg(gold, perfect))[0] == pytest.approx(100.0)
    # tp 2, fn 1, fp 0 per label: F1 = 4/5 < 6/7 -> clipped to 0 (worse than the constant answer)
    worse = [("yes",) * 3] * 2 + [("no",) * 3] * 2
    assert _wg_score(*_wg(gold, worse))[0] == pytest.approx(0.0)
    # a better-than-constant model: 3 tp, 1 fp -> F1 = 6/7 (== const -> 0); tp 3, fp 0 -> 100
    # balanced draw, 2 yes / 2 no: always-yes F1 = 2*2/(2+4) = 2/3; tp 2, fp 1, fn 0 -> F1 = 0.8 -> (0.8-2/3)/(1/3) = 0.4
    bal = [("yes",) * 3] * 2 + [("no",) * 3] * 2
    one_fp = [("yes",) * 3] * 3 + [("no",) * 3]
    sc, ex = _wg_score(*_wg(bal, one_fp))
    assert sc == pytest.approx(100 * 0.4) and ex["raw_f1"] == pytest.approx(80.0) and ex["f1_const"] == pytest.approx(100 * 2 / 3)
    # labels are scored separately and averaged: label 1 perfect, labels 2 and 3 always-yes
    mixed_pred = [("yes", "yes", "yes")] * 2 + [("no", "yes", "yes")] * 2
    assert _wg_score(*_wg(bal, mixed_pred))[0] == pytest.approx(100 / 3)
    # the raw F1 stays selectable
    assert _wg_score(*_wg(gold, always_yes), wildguard_mode="raw")[0] == pytest.approx(100 * f1_const)
    # unusable answers count as "no": every gold-yes becomes a false negative
    items, _ = _wg(gold, perfect)
    assert _wg_score(items, {it["id"]: None for it in items})[0] == pytest.approx(0.0)


def test_s2_wildguard_f1_const_one_edge_case():
    # every gold answer is yes: always-yes has F1 = 1, so the score is 0 unless the model reaches F1 = 1, then 100
    gold = [("yes",) * 3] * 4
    assert _wg_score(*_wg(gold, gold))[0] == pytest.approx(100.0)
    assert _wg_score(*_wg(gold, [("yes",) * 3] * 3 + [("no",) * 3]))[0] == pytest.approx(0.0)
    assert _wg_score(*_wg(gold, [("no",) * 3] * 4))[0] == pytest.approx(0.0)
    # no gold positives at all: always-no is the best constant (F1_const 0 by the zero-division convention), nothing to hit
    none = [("no",) * 3] * 4
    assert _wg_score(*_wg(none, none))[0] == pytest.approx(0.0)
    assert _wg_score(*_wg(none, [("yes",) * 3] * 4))[0] == pytest.approx(0.0)


def test_s3_wildguard_yes_no_tie_splits_the_counts():
    gold = [("yes",) * 3] * 2 + [("no",) * 3] * 2
    items, _ = _wg(gold, gold)
    tie = {it["id"]: [{"yes": 0.5, "no": 0.5}] * 3 for it in items}
    t = v5.item_table(items, tie)
    assert t.tp[:, 0].tolist() == [0.5, 0.5, 0, 0] and t.fn[:, 0].tolist() == [0.5, 0.5, 0, 0] and t.fp[:, 0].tolist() == [0, 0, 0.5, 0.5]
    # tp 1, fp 1, fn 1 -> F1 = 0.5 < const 2/3 -> corrected 0; raw F1 = 50
    assert _wg_score(items, tie)[0] == pytest.approx(0.0)
    assert _wg_score(items, tie, wildguard_mode="raw")[0] == pytest.approx(50.0)


def test_new_mode_parameters_are_validated():
    for kw in ({"omniscience_mode": "x"}, {"wildguard_mode": "x"}, {"wildguard_mode": True}):
        with pytest.raises(ValueError):
            v5.V5Params(**kw)
    assert v5.DEFAULT.omniscience_mode == "aa_index" and v5.DEFAULT.wildguard_mode == "const_corrected"


def test_all_fields_right_and_many_labels():
    items = [mirror(f"f{i}", questions=[q(["a", "b"], "a"), q(["a", "b"], "a")]) for i in range(4)]
    ans = answers(items, [1, 1, 0, 0])
    ans["f2"] = [onehot(["a", "b"], "a"), onehot(["a", "b"], "b")]   # one field wrong still fails the item
    _, parts = mscore(v5.item_table(items, ans))
    assert parts["t1"] == pytest.approx(100 * (0.5 - 0.25) / 0.75)            # chance 1/2 x 1/2
    keys = [f"k{i}" for i in range(30)]
    big = [mirror(f"b{i}", questions=[q(keys, "k29")]) for i in range(2)]
    ans = {"b0": onehot(keys, "k29"), "b1": onehot(keys, "k3")}
    assert mscore(v5.item_table(big, ans))[1]["t1"] == pytest.approx(100 * (0.5 - 1 / 30) / (1 - 1 / 30))


def test_best_constant_answer_is_the_zero_point_of_a_type():
    # 4 yes/no items, gold "no" three times: always-no gets 75 % right; random chance alone would score it 50
    items = [mirror(f"c{i}", ttype="cls", questions=[q(["yes", "no"], g)]) for i, g in enumerate(["no", "no", "no", "yes"])]
    always_no = {it["id"]: onehot(["yes", "no"], "no") for it in items}
    assert v5.type_scores(v5.item_table(items, always_no), None, v5.DEFAULT)["cls"] == pytest.approx(0.0)
    assert v5.type_scores(v5.item_table(items, always_no), None, NC)["cls"] == pytest.approx(50.0)
    perfect = {it["id"]: onehot(["yes", "no"], it["questions"][0]["expected"]) for it in items}
    assert v5.type_scores(v5.item_table(items, perfect), None, v5.DEFAULT)["cls"] == pytest.approx(100.0)
    # better than the constant: 3 of 4 right with the minority class caught: (1.0 - 0.75) / 0.25 stays 100 only if perfect
    one_off = dict(perfect, c0=onehot(["yes", "no"], "yes"))
    assert v5.type_scores(v5.item_table(items, one_off), None, v5.DEFAULT)["cls"] == pytest.approx(0.0)   # 3/4 == the constant 3/4
    # multi-question types fall back to random chance
    two = [mirror(f"m{i}", ttype="mq", questions=[q(["a", "b"], "a"), q(["a", "b"], "a")]) for i in range(2)]
    ans = {"m0": [onehot(["a", "b"], "a")] * 2, "m1": [onehot(["a", "b"], "a"), onehot(["a", "b"], "b")]}
    assert v5.type_scores(v5.item_table(two, ans), None, v5.DEFAULT)["mq"] == pytest.approx(100 * (0.5 - 0.25) / 0.75)


# ------------------------------------------------------------------------------------- aggregation (DI style)
def test_area_aggregation_gold_weight_and_area_weights():
    meta = {"k_gold": ("knowledge", True, "T"), "k_plain": ("knowledge", False, "T"), "l_plain": ("language", False, "T")}
    ts = {"k_gold": 80.0, "k_plain": 50.0, "l_plain": 20.0}
    score, areas = v5.aggregate_areas(ts, meta)
    k = (1.2 * 80 + 50) / 2.2                      # gold x1.2 inside its area
    assert areas == pytest.approx({"knowledge": k, "language": 20.0})
    # areas absent from the draw are left out and the weights of the rest renormalised
    assert score == pytest.approx((0.2585 * k + 0.2585 * 20) / (0.2585 + 0.2585))
    full = {**meta, "r": ("retrieval", False, "T"), "t": ("tools", False, "T"), "a": ("arts", False, "T")}
    s_full, _ = v5.aggregate_areas({**ts, "r": 10.0, "t": 30.0, "a": 90.0}, full)
    assert s_full == pytest.approx(0.2585 * k + 0.2585 * 20 + 0.2002 * 10 + 0.1828 * 30 + 0.10 * 90)
    assert v5.aggregate_areas({}, {}) == (None, {})
    # a gold weight of 1 removes the factor
    s1, a1 = v5.aggregate_areas(ts, meta, v5.V5Params(gold_weight=1.0))
    assert a1["knowledge"] == pytest.approx(65.0)


def test_n_is_weighted_per_task_not_per_item_or_area():
    assert v5.aggregate_tasks({"x": 80.0, "y": 20.0, "z": 50.0}) == pytest.approx(50.0) and v5.aggregate_tasks({}) is None
    big, small = typed("b", 30, ttype="big", part="N"), typed("s", 10, ttype="small", part="N")
    ans = {**answers(big, [1] * 30), **answers(small, [0] * 10)}
    t = v5.item_table([*big, *small], ans)
    assert v5.aggregate_tasks(v5.type_scores(t, None, NC)) == pytest.approx(50.0)    # (100 + 0) / 2, not 75 per item


# ------------------------------------------------------------------------------------- S_v5.1
PANEL_NOPUB = v5.PanelStats("synthetic-v0", t=(50.0, 20.0), n=(40.0, 10.0))
# identity Pub stats (m = mB, s = sB): Pub keeps its raw value but counts as scaled (gap_scaled), see critic r1 F1
PANEL = v5.PanelStats("synthetic-v0", t=(50.0, 20.0), n=(40.0, 10.0), pub=(23.4973, 16.539081))


def eq(x, m, s):
    return 23.4973 + 16.539081 * (x - m) / s


def scenario(t_right, n_right, pub_right=None, bt=6, bn=6, bp=6, n=10, legacy=False, legacy_right=None):
    """T type tt1 (tools), N type nn1, optional typed public type pp1 (knowledge), each ``n`` balanced 2-option items.
    Type score = 100 (2a - 1) for a fraction a right."""
    ti, ni = typed("t", n, ttype="tt1"), typed("n", n, ttype="nn1", part="N")
    items = ti + ni
    cand = {**answers(ti, [1] * t_right + [0] * (n - t_right)), **answers(ni, [1] * n_right + [0] * (n - n_right))}
    base = {**answers(ti, [1] * bt + [0] * (n - bt)), **answers(ni, [1] * bn + [0] * (n - bn))}
    if pub_right is not None:
        pi = typed("p", n, ttype="pp1", area="knowledge", part="pub")
        items += pi
        cand.update(answers(pi, [1] * pub_right + [0] * (n - pub_right)))
        base.update(answers(pi, [1] * bp + [0] * (n - bp)))
    if legacy:
        lr = [held(f"l{i}", tier="same_family") for i in range(n)]
        items += lr
        cand.update(held_answers(lr, [1] * (legacy_right if legacy_right is not None else n) + [0] * (n - (legacy_right or n))))
        base.update(held_answers(lr, [1] * 5 + [0] * (n - 5)))
    return items, cand, base


def test_s_v51_hand_computed_with_scaling_and_no_cap():
    items, cand, base = scenario(9, 8, pub_right=9)
    params = v5.V5Params(panel=PANEL, a_floor=3.0, a_se_mult=0.0)
    res = v5.score_v5_detail(cand, items, base, params)
    assert (res.t, res.n) == pytest.approx((80, 60))                       # 100 (2 x 0.9 - 1), 100 (2 x 0.8 - 1)
    assert res.detail["T_base"] == pytest.approx(20) and res.detail["N_base"] == pytest.approx(20)
    te, ne = eq(80, 50, 20), eq(60, 40, 10)
    assert (res.t_eq, res.n_eq) == pytest.approx((te, ne)) and te == pytest.approx(48.3059215) and ne == pytest.approx(56.575462)
    assert res.s == pytest.approx(0.6 * te + 0.4 * ne) == pytest.approx(51.6137377, abs=1e-6)
    s_b = 0.6 * eq(20, 50, 20) + 0.4 * eq(20, 40, 10)
    assert res.detail["S_v5.1_base"] == pytest.approx(s_b)
    # no cap and no D / SE_D / P terms anywhere in the result
    assert not {"cap_binding", "D", "SE_D", "P_eff", "M", "P"} & (set(res.as_dict()) | set(res.detail))
    # a huge N gain is paid in full: N all right, T unchanged
    hi = v5.score_v5_detail(scenario(9, 10)[1], scenario(9, 10)[0], base, params)
    assert hi.s == pytest.approx(0.6 * te + 0.4 * eq(100, 40, 10))


def test_w_n_zero_is_the_twins_only_fallback_and_w_n_default_is_point_four():
    assert v5.DEFAULT.w_n == 0.4 and v5.DEFAULT.a_floor == 1.5 and v5.DEFAULT.a_se_mult == 2.0
    items, cand, base = scenario(9, 3)
    params = v5.V5Params(panel=PANEL, w_n=0.0)
    res = v5.score_v5_detail(cand, items, base, params)
    assert res.s == pytest.approx(res.t_eq) == pytest.approx(eq(80, 50, 20))      # N does not enter
    only_t = [i for i in items if i["part"] == "T"]                                   # N items may be absent when w_N = 0
    res2 = v5.score_v5_detail(cand, only_t, base, params)
    assert res2.s == pytest.approx(res.s) and res2.n is None and res2.n_eq is None
    with pytest.raises(ValueError, match="at least two N items"):
        v5.score_v5_detail(cand, only_t, base, v5.V5Params(panel=PANEL))
    with pytest.raises(ValueError, match="lack N"):
        v5.score_v5_detail(cand, items, base, v5.V5Params(panel=v5.PanelStats("v", t=(50, 20))))
    assert v5.score_v5_detail(cand, items, base, v5.V5Params(panel=v5.PanelStats("v", t=(50, 20)), w_n=0.0)).s == pytest.approx(res.s)


def test_unscaled_is_flagged_and_scale_is_linear_in_the_panel_stats():
    items, cand, base = scenario(9, 8)
    raw = v5.score_v5_detail(cand, items, base)
    assert raw.detail["scaled"] is False and raw.t_eq == raw.t and raw.s == pytest.approx(0.6 * 80 + 0.4 * 60)
    sc = v5.score_v5_detail(cand, items, base, v5.V5Params(panel=PANEL))
    assert sc.detail["scaled"] is True and sc.detail["pool_version"] == "synthetic-v0"
    # DI constants: mB 23.4973, sB 16.539081; a panel with mean mB's source and sd sB's source maps X to itself shifted
    ident = v5.V5Params(panel=v5.PanelStats("id", t=(23.4973, 16.539081), n=(23.4973, 16.539081)))
    assert v5.score_v5_detail(cand, items, base, ident).t_eq == pytest.approx(80)


def test_base_against_itself():
    items, _, base = scenario(8, 8, pub_right=8)
    res = v5.score_v5_detail(base, items, base, v5.V5Params(panel=PANEL))
    assert res.s == pytest.approx(res.detail["S_v5.1_base"]) and res.g == 0 and res.se_g == 0
    assert res.a == 1.5 and res.gap_flag is False


# ------------------------------------------------------------------------------------- gap check (report only)
def test_gap_check_formula_and_flag():
    items, cand, base = scenario(8, 8, pub_right=10, bt=6, bn=6, bp=6)
    # identity Pub stats (m = mB, s = sB): Pub stays 100 / 20 but counts as scaled, so G compares like with like
    panel = v5.PanelStats("synthetic-v0", t=(50.0, 20.0), n=(40.0, 10.0), pub=(23.4973, 16.539081))
    params = v5.V5Params(panel=panel, a_floor=3.0, a_se_mult=0.0)
    res = v5.score_v5_detail(cand, items, base, params)
    d_pub = 100 - 20
    d_s = res.s - res.detail["S_v5.1_base"]
    assert res.pub == pytest.approx(100) and res.detail["Pub_base"] == pytest.approx(20) and res.detail["gap_scaled"] is True
    assert res.g == pytest.approx(d_pub - d_s) and res.g > 3.0
    assert res.a == 3.0 and res.gap_flag is True and res.se_g > 0
    # A = max(1.5, 2 SE_G)
    p2 = v5.V5Params(panel=panel, a_floor=0.0, a_se_mult=2.0)
    r2 = v5.score_v5_detail(cand, items, base, p2)
    assert r2.a == pytest.approx(2 * r2.se_g) and r2.gap_flag == bool(r2.g > r2.a)
    assert v5.score_v5_detail(cand, items, base, v5.V5Params(panel=panel, a_floor=1000.0)).gap_flag is False   # floor above G
    # report only: the score does not move with the public part
    lo = scenario(8, 8, pub_right=3)
    assert v5.score_v5_detail(lo[1], lo[0], lo[2], params).s == pytest.approx(res.s)
    # a scaled Pub uses its own stats
    pp = v5.V5Params(panel=v5.PanelStats("v", t=(50, 20), n=(40, 10), pub=(60, 30)), a_floor=3.0, a_se_mult=0.0)
    r3 = v5.score_v5_detail(cand, items, base, pp)
    assert r3.pub == pytest.approx(eq(100, 60, 30)) and r3.detail["pub_scaled"] is True
    # critic r1 F1: a panel without Pub stats mixes units -> G reported, gap_flag None, gap_scaled False
    r4 = v5.score_v5_detail(cand, items, base, v5.V5Params(panel=PANEL_NOPUB, a_floor=3.0, a_se_mult=0.0))
    assert r4.detail["pub_scaled"] is False and r4.detail["gap_scaled"] is False and r4.gap_flag is None and r4.g is not None


def test_gap_check_without_public_items_is_none():
    items, cand, base = scenario(8, 8)
    res = v5.score_v5_detail(cand, items, base, v5.V5Params(panel=PANEL))
    assert (res.pub, res.g, res.se_g, res.a, res.gap_flag) == (None, None, None, None, None)


def test_v4_format_public_rows_feed_pub_and_the_old_tiers_have_weight_zero():
    items, cand, base = scenario(8, 8, legacy=True, legacy_right=10)
    old = [held(f"o{i}", tier="new_domain") for i in range(6)]
    pf = [held(f"f{i}", tier="proc_sealed") for i in range(6)]
    params = v5.V5Params(panel=PANEL)
    a = v5.score_v5_detail({**cand, **held_answers(old, [1] * 6)}, items + old + pf,
                           {**base, **held_answers(old, [0] * 6), **held_answers(pf, [0] * 6)} | held_answers(pf, [0] * 6), params, allow_partial=True)
    b = v5.score_v5_detail({**cand, **held_answers(old, [0] * 6)}, items + old + pf,
                           {**base, **held_answers(old, [1] * 6), **held_answers(pf, [0] * 6)}, params, allow_partial=True)
    assert a.s == pytest.approx(b.s) and a.t_eq == b.t_eq                       # old own tiers: reported, weight 0
    assert a.detail["internal"]["tiers"]["new_domain"]["score"] > b.detail["internal"]["tiers"]["new_domain"]["score"]
    assert set(a.detail["internal"]["tiers"]) == {"same_family", "proc_sealed", "new_domain"}
    assert a.pub is not None and a.g is not None and a.detail["n_pub"] == 16 and a.detail["n_old"] == 6
    # an award_realdata row counts as the cross_author tier
    ra = dict(held("x1", tier="award_realdata"))
    r = v5.score_v5_detail({**cand, "x1": onehot(["a", "b"], "a")}, items + [ra], {**base, "x1": onehot(["a", "b"], "a")}, params)
    assert "cross_author" in r.detail["internal"]["tiers"]


# ------------------------------------------------------------------------------------- overlap exclusion
def test_overlap_exclusion_scores_the_type_at_min_of_entry_and_base():
    ta, tb = typed("a", 10, ttype="ta"), typed("b", 10, ttype="tb", area="language")
    nn = typed("n", 10, ttype="nn1", part="N")
    items = ta + tb + nn
    cand = {**answers(ta, [1] * 10), **answers(tb, [1] * 10), **answers(nn, [1] * 8 + [0] * 2)}   # ta 100, tb 100, N 60
    base = {**answers(ta, [1] * 6 + [0] * 4), **answers(tb, [1] * 9 + [0]), **answers(nn, [1] * 6 + [0] * 4)}   # ta 20, tb 80, N 20
    params = v5.V5Params(w_n=0.0)
    plain = v5.score_v5_detail(cand, items, base, params)
    assert plain.t == pytest.approx(100.0) and plain.overlap_excluded is False
    ex = v5.score_v5_detail(cand, items, base, params, exclude_types=["ta"])
    assert ex.detail["internal"]["types_T"]["ta"] == pytest.approx(20)               # min(100, 20)
    assert ex.t == pytest.approx((0.1828 * 20 + 0.2585 * 100) / (0.1828 + 0.2585)) and ex.overlap_excluded is True
    # the base itself is unchanged, and an excluded type where the entry is BELOW base keeps the entry value (min cannot raise it)
    assert ex.detail["T_base"] == pytest.approx(plain.detail["T_base"])
    low = v5.score_v5_detail(base, items, cand, params, exclude_types=["ta", "tb"])
    assert low.t == pytest.approx(v5.score_v5_detail(base, items, cand, params).t)
    # exclusion also applies to N tasks
    w4 = v5.V5Params(w_n=0.4)
    ne = v5.score_v5_detail(cand, items, base, w4, exclude_types=["nn1"])
    assert ne.n == pytest.approx(20) and v5.score_v5_detail(cand, items, base, w4).n == pytest.approx(60)
    # unknown ids are ignored in the score and listed internally; the flag is just "a list was given"
    un = v5.score_v5_detail(cand, items, base, params, exclude_types=["nope"])
    assert un.t == plain.t and un.overlap_excluded is True and un.detail["internal"]["overlap_unmatched_types"] == ["nope"]
    with pytest.raises(ValueError, match="exclude_types"):
        v5.score_v5_detail(cand, items, base, params, exclude_types="ta")


def test_public_view_has_no_type_ids_or_internal_scores():
    items, cand, base = scenario(9, 8, pub_right=9)
    out = v5.score_v5_detail(cand, items, base, v5.V5Params(panel=PANEL), exclude_types=["tt1"]).as_dict()
    pub = v5.public_view(out)
    assert pub["overlap_excluded"] is True and set(pub) <= set(v5.PUBLIC_KEYS)
    assert "tt1" not in json.dumps(pub) and "detail" not in pub and "internal" in out["detail"]


# ------------------------------------------------------------------------------------- inputs and parameters
def test_type_metadata_must_be_consistent():
    a, b = mirror("x", ttype="tt"), mirror("y", ttype="tt", area="language")
    with pytest.raises(ValueError, match="conflicting"):
        v5.type_meta([a, b])
    items, cand, base = scenario(8, 8)
    bad = [dict(items[0], area="arts")] + items[1:]
    with pytest.raises(ValueError, match="conflicting"):
        v5.score_v5_detail(cand, bad, base)


def test_type_coverage_rule():
    items, _, _ = scenario(8, 8, n=12)
    sp = v5.split_items(items)
    assert v5.type_coverage(sp, 12) == {}
    thin = v5.split_items(items[:-1])
    assert v5.type_coverage(thin, 12) == {"nn1": 11} and v5.type_coverage(thin, 0) == {}
    pubs = v5.split_items(scenario(8, 8, pub_right=8, n=3)[0])
    assert v5.type_coverage(pubs, 4) == {"nn1": 3, "tt1": 3}          # public types are not checked


def test_inputs_are_checked():
    items, cand, base = scenario(9, 9)
    with pytest.raises(ValueError, match="base responses missing"):
        v5.score_v5(cand, items, {k: v for k, v in base.items() if k != "t0"})
    with pytest.raises(ValueError, match="required"):
        v5.score_v5_detail(cand, items, None)
    with pytest.raises(ValueError, match="not part of the v5.1 score"):
        v5.score_v5(cand, items + [held("sf", tier="mystery")], {**base, "sf": onehot(["a", "b"], "a")})
    res = v5.score_v5_detail(cand, items + [held("sf", tier="mystery")], {**base, "sf": onehot(["a", "b"], "a")}, strict=False)
    assert res.detail["ignored_items"] == 1
    with pytest.raises(ValueError, match="part"):
        v5.score_v5(cand, items + [dict(items[0], id="bad", part="X")], base)
    missing = {k: v for k, v in cand.items() if k != "t0"}   # an unanswered item is a failed item, not an error
    assert v5.score_v5(missing, items, base)[1] < v5.score_v5(cand, items, base)[1]
    with pytest.raises(ValueError, match="two T items"):
        v5.score_v5(cand, items[:1] + items[10:], base)


@pytest.mark.parametrize("kwargs", [
    {"w_n": float("nan")},
    {"w_n": -0.1},
    {"w_n": 1.1},
    {"omni_acc_weight": 1.1},
    {"hf_areas": {**v5.HF_AREAS, "tools": float("inf")}},
    {"hf_areas": {a: 0.0 for a in v5.HF_AREAS}},
    {"hf_areas": {"knowledge": 1.0}},
    {"gold_weight": 0.0},
    {"gold_weight": float("inf")},
    {"m_b": float("nan")},
    {"s_b": 0.0},
    {"panel": {"T": (0, 1)}},
    {"const_baseline": 1},
    {"a_floor": -0.1},
    {"a_floor": float("nan")},
    {"a_se_mult": float("inf")},
    {"a_se_mult": -1.0},
    {"resamples": 1},
    {"resamples": 2.5},
    {"t_items": -1},
    {"n_items": 1.5},
    {"t_items": 0, "n_items": 0},
    {"omniscience_mode": "x"},
    {"wildguard_mode": "x"},
])
def test_v5_params_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        v5.V5Params(**kwargs)


def test_v5_params_accept_boundary_values():
    params = v5.V5Params(w_n=0.0, omni_acc_weight=1.0, a_floor=0.0, a_se_mult=0.0, resamples=2, pub_items=0)
    assert params.w_n == 0.0 and params.resamples == 2 and v5.V5Params(w_n=1.0).w_n == 1.0


def test_panel_stats_parsing_and_validation():
    p = v5.panel_from_json({"pool_version": "v0", "T": {"mean": 30, "sd": 10}, "N": {"mean": 20, "sd": 5}})
    assert p.t == (30.0, 10.0) and p.n == (20.0, 5.0) and p.pub is None and p.version == "v0"
    for bad in ([], {"pool_version": "v", "N": {"mean": 1, "sd": 1}}, {"pool_version": "v", "T": {"mean": 1}},
                {"pool_version": "v", "T": {"mean": 1, "sd": 0}}, {"pool_version": "v", "T": {"mean": float("nan"), "sd": 1}},
                {"pool_version": "v", "T": {"mean": 1, "sd": 1}, "X": 1}, {"T": {"mean": 1, "sd": 1}},
                {"pool_version": "v", "T": {"mean": True, "sd": 1}}):
        with pytest.raises(ValueError):
            v5.panel_from_json(bad)
    assert set(v5.PANEL_STATS) == {"v0.1"}        # pool v0.1 MAIN, stock panel at w_N 0.4 (round2-v5-monday ON-REPLY step 3)
    v01 = v5.PANEL_STATS["v0.1"]
    assert (v01.version, v01.t, v01.n, v01.pub) == ("v0.1", (58.443470442129176, 19.08197856407689),
                                                     (28.490328496088534, 8.20230568361892), (46.68485197351358, 19.22683663229349))


def test_pooled_se():
    assert v5.pool_se([1.4, 1.4]) == pytest.approx(1.4 / math.sqrt(2))
    assert v5.pool_se([3.0]) == 3.0 and math.isinf(v5.pool_se([]))


def test_bootstrap_is_deterministic_for_a_seed():
    items, cand, base = scenario(9, 8, pub_right=9)
    a = v5.score_v5_detail(cand, items, base, v5.V5Params(panel=PANEL, resamples=50), seed=3)
    b = v5.score_v5_detail(cand, items, base, v5.V5Params(panel=PANEL, resamples=50), seed=3)
    c = v5.score_v5_detail(cand, items, base, v5.V5Params(panel=PANEL, resamples=50), seed=4)
    assert a.se_g == b.se_g and a.se_g != c.se_g


# ------------------------------------------------------------------------------------- validation
def test_validate_mirror_row():
    ok = mirror("v")
    v5.validate_mirror_row(ok)
    for bad in (dict(ok, area="nope"), dict(ok, part="X"), dict(ok, questions=[]), dict(ok, task_type=""), dict(ok, gold="yes"),
                dict(ok, idk=True),                                                       # idk needs omniscience
                dict(ok, questions=[q(["a", "b"], "c")]),
                mirror("w", ttype="wildguard", questions=[q(["a", "b"], "a")]),
                mirror("o", ttype="omniscience", idk=False),
                mirror("o2", ttype="omniscience", idk=True, questions=[q(["a", "b"], "idk")])):
        with pytest.raises(S.SampleError):
            v5.validate_mirror_row(bad)
    v5.validate_mirror_row(mirror("o3", ttype="omniscience", idk=True))
    v5.validate_mirror_row(mirror("w3", ttype="wildguard", questions=[q(["yes", "no"], "yes")] * 3))
    for part in v5.PARTS:
        v5.validate_mirror_row(mirror("p", part=part))


# ------------------------------------------------------------------------------------- the draw
T_TYPES = {"knowledge": [("k_gold", True), ("k_plain", False)], "language": [("l_gold", True), ("l_plain", False)],
           "retrieval": [("r_gold", True), ("r_plain", False)], "tools": [("t_gold", True), ("t_plain", False)],
           "arts": [("a_plain", False)]}
N_NAMES = ["n_triage", "n_claims", "n_clauses", "n_moderation"]


def v5_pool(per_type=100, per_n=100, per_cat=15, pub=0):
    rows = []
    for area, types in T_TYPES.items():
        for tt, g in types:
            rows += [mirror(f"{tt}-{i}", ttype=tt, area=area, gold=g) for i in range(per_type)]
    for tt in N_NAMES:
        rows += [mirror(f"{tt}-{i}", ttype=tt, area="tools", part="N") for i in range(per_n)]
    rows += [mirror(f"pub-{i}", ttype="pub_mc", area="knowledge", part="pub") for i in range(pub)]
    for tier in ("same_family", "proc_sealed", "new_family", "new_domain", S.X_TIER):
        for c in CATS:
            rows += [dict(held(f"{tier}-{c}-{i}", tier=tier, cat=c)) for i in range(per_cat)]
    return rows


def test_type_cells():
    cells = v5.type_cells(v5_pool())
    t = {k[1]: n for k, n in cells.items() if k[0] == "T"}
    nn = {k[1]: n for k, n in cells.items() if k[0] == "N"}
    assert sum(t.values()) == 540 and sum(nn.values()) == 360 and set(nn.values()) == {90}
    area = {a: sum(t[tt] for tt, _ in types) for a, types in T_TYPES.items()}
    assert sum(area.values()) == 540 and abs(area["knowledge"] - 139.59) < 1 and abs(area["arts"] - 54) < 1e-9
    assert abs(area["retrieval"] - 108.1) < 1 and abs(area["tools"] - 98.7) < 1
    for a, types in T_TYPES.items():
        if len(types) == 2:
            assert t[types[0][0]] > t[types[1][0]]                     # a gold type gets 1.2 / 2.2 of its area
            assert abs(t[types[0][0]] / area[a] - 1.2 / 2.2) < 0.02
    assert min(t.values()) >= 12 and min(nn.values()) >= 12


def test_type_cells_enforce_the_per_type_floor_and_cover_every_area():
    tiny = v5.V5Params(t_items=30, n_items=24, min_type_items=12)
    rows = [r for r in v5_pool(per_type=40, per_n=40) if r.get("part") in ("T", "N")]
    with pytest.raises(S.SampleError, match="cannot give"):
        v5.type_cells(rows, tiny)                                     # 9 T types x 12 = 108 > 30
    only_k = [r for r in rows if r["part"] == "N" or r["area"] == "knowledge"]
    with pytest.raises(S.SampleError, match="no task type in area"):
        v5.type_cells(only_k)
    with pytest.raises(S.SampleError, match="no N task types"):
        v5.type_cells([r for r in rows if r["part"] == "T"])
    # a floor lifts a small type to 12 and pays from the largest type of its area
    skew = v5.V5Params(t_items=200, n_items=0, min_type_items=12, gold_weight=100.0)
    cells = v5.type_cells([r for r in rows if r["part"] == "T"], skew)
    assert cells[("T", "k_plain")] == 12 and cells[("T", "k_gold")] > 30 and sum(cells.values()) == 200


def test_draw_composition():
    pool = v5_pool()
    rows, fresh = v5.draw_v5(pool, n=900, seed=7, use_counts={})
    assert len(rows) == 900 and fresh and len({r["id"] for r in rows}) == 900
    parts = {p: sum(r["part"] == p for r in rows) for p in v5.PARTS}
    assert parts == {"T": 540, "N": 360, "pub": 0} and all(r["tier"] == "mirror" for r in rows)
    got = {}
    for r in rows:
        got[(r["part"], r["task_type"])] = got.get((r["part"], r["task_type"]), 0) + 1
    assert got == v5.type_cells(pool) and min(got.values()) >= 12
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    assert v5.draw_v5(pool, n=900, seed=7, use_counts={})[0] == rows
    assert v5.draw_v5(pool, n=900, seed=8, use_counts={})[0] != rows
    with pytest.raises(S.SampleError, match="exactly 900"):
        v5.draw_v5(pool, n=300, seed=1, use_counts={})


def test_draw_with_public_items_for_the_gap_check():
    pool = v5_pool(pub=40)
    params = v5.V5Params(pub_items=60)
    rows, _ = v5.draw_v5(pool, n=960, seed=2, use_counts={}, params=params)
    n_pub = sum(1 for r in rows if v5._row_part(r) == "pub")
    assert len(rows) == 960 and n_pub == 60 and sum(r.get("part") == "T" for r in rows) == 540
    with pytest.raises(S.SampleError, match="exactly 960"):
        v5.draw_v5(pool, n=900, seed=2, use_counts={}, params=params)
    with pytest.raises(S.SampleError, match="public items"):
        v5.draw_v5([r for r in pool if v5._row_part(r) != "pub"], n=960, seed=2, use_counts={}, params=params)
    # old own tiers are never drawn
    assert not any(v5._row_part(r) == "old" for r in rows)


def test_draw_uses_least_used_first_exclude_and_retirement():
    pool = v5_pool(per_type=100, per_n=100)
    use = {r["id"]: 5 for r in pool if r["tier"] == "mirror" and r["id"].endswith("-0")}   # one used item per type
    rows, fresh = v5.draw_v5(pool, n=900, seed=1, use_counts=use)
    assert not any(r["id"] in use for r in rows) and fresh
    ids = [r["id"] for r in pool if r.get("task_type") == "n_claims"]
    assert len(ids) == 100
    rows, _ = v5.draw_v5(pool, n=900, seed=1, use_counts={}, exclude=set(ids[:10]))      # exactly 90 left
    assert not set(ids[:10]) & {r["id"] for r in rows}
    with pytest.raises(S.SampleError, match="type N/n_claims"):
        v5.draw_v5(pool, n=900, seed=1, use_counts={}, exclude=set(ids[:11]))            # one short
    retired = {i: v5.MIRROR_MAX_USES for i in ids}
    with pytest.raises(S.SampleError, match="type N/n_claims"):
        v5.draw_v5(pool, n=900, seed=1, use_counts=retired)
    heavy = {r["id"]: 1 for r in pool if r["tier"] == "mirror"}
    assert v5.draw_v5(pool, n=900, seed=1, use_counts=heavy)[1] is False


def test_sampling_routes_by_gate(monkeypatch):
    pool = v5_pool(per_cat=40)
    now = T.timestamp()
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", T)
    assert S.v5_active(now) and S.v5_active(T) and not S.v5_active(now - 1) and not S.v5_active(None)
    rows, _ = S.draw(pool, n=900, seed=3, generalist=True, now=now)
    assert sum(r["part"] == "T" for r in rows) == 540 and sum(r["part"] == "N" for r in rows) == 360
    monkeypatch.setattr(S, "DRAWS_V4_START_UTC", datetime(2026, 10, 23, 13, tzinfo=timezone.utc))
    old, _ = S.draw(pool, n=900, seed=3, generalist=True, now=now - 10)   # before T: the v4 draw, typed rows never in it
    assert not any(r["tier"] == "mirror" for r in old) and len(old) == 900


# ------------------------------------------------------------------------------------- gate off
V4_DIGESTS = ['9ce93dfb1129e50f', '6c7f9cc7045a6340', '6689481676cf2cdc', 'c14cd3dedf9bfde2', 'a58409e3286ff58a', '8c98d531b4ce2935',
              '7591ef5062b7bee4', 'a33e08ef2379d651', '0d5d2ffe783b1872', 'f1496a915cdd08db', 'dbcb634ccc6072b3', '2af6f56c58ecbe86']


def _v4_pool():
    rows = []
    for cat in CATS:
        for i in range(200):
            kind = ("choice", "choice", "noul", "score")[i % 4] if i % 2 else ("choice", "noul", "score")[i % 3]
            rows.append({"id": f"{cat}-{i}", "category": cat, "question": {"type": kind},
                         "tier": ("proc_sealed", "same_family", "new_domain", "new_family", S.X_TIER)[i % 5]})
    return rows


def _digests(pool, monkeypatch):
    gate = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(S, "DRAWS_V4_START_UTC", gate)
    use = {r["id"]: (i % 7) for i, r in enumerate(_v4_pool())}
    out = []
    for seed in (1, 2, 3):
        for xm in ("off", "binding"):
            for now in (gate.timestamp() + 5, gate.timestamp() - 5):
                rows, fresh = S.draw(pool, n=S.draw_size(S.SAMPLE_SIZE, True, now), seed=seed, use_counts=use, x_mode=xm,
                                     generalist=True, now=now)
                out.append(hashlib.sha256((",".join(r["id"] for r in rows) + str(fresh)).encode()).hexdigest()[:16])
    return out


def test_gate_defaults_to_none():
    assert S.EVAL_V5_START_UTC is None and v5.EVALUATOR_WIRED is True   # wired (tests/test_v5_wiring.py); the gate stays off
    assert not S.v5_active(10 ** 12) and v5.scoring_regime(10 ** 12) == "v4"


def test_gate_off_draws_equal_rc11_bit_for_bit(monkeypatch):
    """The digests were produced by the rc11 tree (5424beb) for this exact pool, use counts and seeds."""
    assert _digests(_v4_pool(), monkeypatch) == V4_DIGESTS


def test_gate_off_ignores_mirror_rows_in_the_pool(monkeypatch):
    mixed = _v4_pool() + [r for r in v5_pool() if r["tier"] == "mirror"]
    assert _digests(mixed, monkeypatch) == V4_DIGESTS
    assert S.eligible_count(mixed, {}) == len(mixed)                      # counts unchanged for non-mirror users of the pool
    assert S.nonproc_draws_left(mixed, {}) == S.nonproc_draws_left(_v4_pool(), {})


def test_mirror_file_is_loaded_only_with_the_gate(tmp_path, monkeypatch):
    d = tmp_path / "generalist" / "sealed"
    d.mkdir(parents=True)
    held_rows = [dict(held("h1"), category="math_quant")]
    (d / "sealed.clean.jsonl").write_text("\n".join(json.dumps(r) for r in held_rows))
    (d / S.MIRROR_FILE).write_text(json.dumps(mirror("m1")))
    assert [r["id"] for r in S.load_pool(str(tmp_path), "generalist")] == ["h1"]
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", T)
    assert [r["id"] for r in S.load_pool(str(tmp_path), "generalist")] == ["h1", "m1"]
    (d / S.MIRROR_FILE).write_text(json.dumps(dict(mirror("h1"))))
    with pytest.raises(S.SampleError, match="duplicate"):
        S.load_pool(str(tmp_path), "generalist")


def test_evaluator_refuses_to_start_with_the_gate_set_while_not_wired(monkeypatch):
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", T)
    monkeypatch.setattr(v5, "EVALUATOR_WIRED", False)
    with pytest.raises(RuntimeError, match="not wired"):
        E.SpecEvaluator(None)


# ------------------------------------------------------------------------------------- migration
def test_scoring_regime(monkeypatch):
    t = T.timestamp()
    assert v5.scoring_regime(t + 1) == "v4"                       # gate off
    monkeypatch.setattr(S, "EVAL_V5_START_UTC", T)
    assert (v5.scoring_regime(t - 1), v5.scoring_regime(t), v5.scoring_regime(t + 86400)) == ("v4", "v5", "v5")


def test_challenger_vs_seated_champion():
    champ, base = {"d1": 60.0, "d2": 62.0}, {"d1": 58.0, "d2": 59.0}
    assert v5.beats_seated(champ, base, {"d1": 63.0, "d2": 65.0}, 1.0) == (True, 62.0, 64.0)   # max(61, 58.5) + 1
    assert v5.beats_seated(champ, base, {"d1": 62.0, "d2": 62.0}, 1.0)[0] is True               # exactly threshold: gain >= margin
    assert v5.beats_seated(champ, base, {"d1": 61.9, "d2": 61.9}, 1.0)[0] is False
    weak = {"d1": 55.0, "d2": 55.0}                                                              # the base is the bar
    ok, thr, _ = v5.beats_seated(weak, base, {"d1": 60.0, "d2": 60.0}, 1.0)
    assert thr == pytest.approx(59.5) and ok
    with pytest.raises(ValueError):
        v5.beats_seated(champ, base, {"d1": 70.0}, 1.0)
    with pytest.raises(ValueError):
        v5.beats_seated(champ, {"d1": 1.0, "d3": 2.0}, {"d1": 70.0, "d2": 70.0}, 1.0)



# ------------------------------------------------------------------------------------- the CLI
def _score_cli():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("score_v5_cli", pathlib.Path(__file__).parent.parent / "tools" / "score_v5.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def _cli_files(tmp_path, **kw):
    items, cand, base = scenario(9, 8, **kw)
    _write_jsonl(tmp_path / "items.jsonl", items)
    for name, resp in (("cand", cand), ("base", base)):
        _write_jsonl(tmp_path / f"{name}.jsonl", ({"id": k, "probs": v} for k, v in resp.items()))
    (tmp_path / "panel.json").write_text(json.dumps({"pool_version": "synthetic-v0", "T": {"mean": 50, "sd": 20},
                                                     "N": {"mean": 40, "sd": 10}, "Pub": {"mean": 23.4973, "sd": 16.539081}}))
    return items, cand, base


def _args(tmp_path, *extra):
    return ["--items", str(tmp_path / "items.jsonl"), "--responses", str(tmp_path / "cand.jsonl"),
            "--base-responses", str(tmp_path / "base.jsonl"), "--min-type-items", "0", *extra]


def test_cli_matches_the_library(tmp_path, capsys):
    cli = _score_cli()
    items, cand, base = _cli_files(tmp_path, pub_right=10)
    with (tmp_path / "cand.jsonl").open("a") as fh:   # The extra candidate id is reported, but never contributes to the score.
        fh.write(json.dumps({"id": "outside", "probs": {"a": 1.0}}) + "\n")
    (tmp_path / "excl.json").write_text('["nn1"]')
    rc = cli.main(_args(tmp_path, "--panel-stats", str(tmp_path / "panel.json"), "--wN", "0.25", "--a-floor", "3", "--a-se-mult", "0",
                        "--seed", "0", "--exclude-types", str(tmp_path / "excl.json"), "--gold-weight", "1.5",
                        "--omniscience-mode", "half_half", "--omni-acc-weight", "0.7", "--wildguard-mode", "raw"))
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    params = v5.V5Params(w_n=0.25, a_floor=3.0, a_se_mult=0.0, gold_weight=1.5, panel=PANEL, omniscience_mode="half_half",
                         omni_acc_weight=0.7, wildguard_mode="raw")
    ref = v5.score_v5_detail(cand, items, base, params, exclude_types=["nn1"]).as_dict()
    for k in ("S_v5.1", "T", "N", "T_eq", "N_eq", "Pub", "G", "SE_G", "A", "gap_flag", "overlap_excluded"):
        assert out[k] == pytest.approx(ref[k]) if isinstance(ref[k], float) else out[k] == ref[k]
    assert out["N"] == pytest.approx(20) and out["overlap_excluded"] is True and out["gap_flag"] is True    # nn1: min(60, 20)
    assert out["detail"]["coverage"]["responses"] == {"answered": 30, "expected": 30, "fraction": 1.0, "extra_ids": ["outside"]}
    assert out["detail"]["params"]["omniscience_mode"] == "half_half" and out["detail"]["params"]["w_n"] == 0.25
    assert out["detail"]["internal"]["overlap_excluded_types"] == ["nn1"]


def test_cli_needs_panel_stats_or_unscaled_and_wn_zero_runs_on_twins(tmp_path, capsys):
    cli = _score_cli()
    _cli_files(tmp_path)
    assert cli.main(_args(tmp_path)) == 2
    assert "exactly one of --panel-stats and --unscaled" in capsys.readouterr().err
    assert cli.main(_args(tmp_path, "--unscaled", "--panel-stats", str(tmp_path / "panel.json"))) == 2
    capsys.readouterr()
    assert cli.main(_args(tmp_path, "--unscaled", "--wN", "0")) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["S_v5.1"] == pytest.approx(80) == pytest.approx(out["T_eq"]) and out["detail"]["scaled"] is False
    # inline JSON panel stats; panel stats without N and w_N > 0 are a clean error
    assert cli.main(_args(tmp_path, "--panel-stats", '{"pool_version": "v", "T": {"mean": 1, "sd": 1}}')) == 2
    assert "lack N" in capsys.readouterr().err


@pytest.mark.parametrize("flag,content,reason", [
    ("--panel-stats", "{bad json", "invalid JSON"),
    ("--panel-stats", '{"pool_version": "v", "T": {"mean": 1, "sd": 0}}', "sd must be > 0"),
    ("--panel-stats", "/nonexistent/panel.json", "panel.json"),
    ("--exclude-types", '{"a": 1}', "expected a JSON list"),
    ("--exclude-types", '[1]', "expected a JSON list"),
    ("--wN", "1.5", "w_n must be in [0, 1]"),
])
def test_cli_bad_stats_exclusions_and_params_are_clean_errors(tmp_path, capsys, flag, content, reason):
    cli = _score_cli()
    _cli_files(tmp_path)
    extra = ["--panel-stats", str(tmp_path / "panel.json")] if flag != "--panel-stats" else []
    assert cli.main(_args(tmp_path, *extra, flag, content)) == 2 if flag != "--panel-stats" else cli.main(_args(tmp_path, flag, content)) == 2
    out, err = capsys.readouterr()
    assert reason in err and "Traceback" not in err and not out


def test_cli_min_type_items_rule(tmp_path, capsys):
    cli = _score_cli()
    _cli_files(tmp_path)
    base_args = ["--items", str(tmp_path / "items.jsonl"), "--responses", str(tmp_path / "cand.jsonl"),
                 "--base-responses", str(tmp_path / "base.jsonl"), "--unscaled"]
    assert cli.main(base_args) == 2                                   # default 12 per type; the fixture has 10
    err = capsys.readouterr().err
    assert "task types below 12 items: nn1 (10), tt1 (10)" in err and "Traceback" not in err
    assert cli.main([*base_args, "--min-type-items", "10"]) == 0
    capsys.readouterr()
    assert cli.main([*base_args, "--min-type-items", "-1"]) == 2


@pytest.mark.parametrize("target,content,reason", [
    ("responses", "{bad json", "Expecting property name"),
    ("responses", "[]", "expected a JSON object"),
    ("responses", '{"probs": {}}', "missing id"),
    ("responses", '{"id": "x"}', "missing probs"),
    ("responses", '{"id": ["x"], "probs": {}}', "id is not hashable"),
    ("items", "{bad json", "Expecting property name"),
    ("items", "[]", "expected a JSON object"),
    ("items", '{"tier": "mirror"}', "missing id"),
    ("items", '{"id": ["x"], "tier": "mirror"}', "id is not hashable"),
])
def test_cli_malformed_jsonl_is_a_line_error_with_exit_2(tmp_path, capsys, target, content, reason):
    cli = _score_cli()
    _cli_files(tmp_path)
    bad = tmp_path / f"bad-{target}.jsonl"
    bad.write_text(content + "\n")
    args = ["--items", str(bad if target == "items" else tmp_path / "items.jsonl"),
            "--responses", str(bad if target == "responses" else tmp_path / "cand.jsonl"),
            "--base-responses", str(tmp_path / "base.jsonl"), "--unscaled"]
    rc = cli.main(args)
    out, err = capsys.readouterr()
    assert rc == 2
    assert f"{bad}:line 1:" in err and reason in err
    assert "Traceback" not in err and not out


def test_cli_invalid_utf8_is_a_line_error_with_exit_2(tmp_path, capsys):
    cli = _score_cli()
    _cli_files(tmp_path)
    bad = tmp_path / "bad-encoding.jsonl"
    bad.write_bytes(b"\xff\n")
    rc = cli.main(["--items", str(tmp_path / "items.jsonl"), "--responses", str(bad),
                   "--base-responses", str(tmp_path / "base.jsonl"), "--unscaled"])
    out, err = capsys.readouterr()
    assert rc == 2 and f"{bad}:line 1: invalid UTF-8" in err
    assert "Traceback" not in err and not out


def test_cli_checks_coverage_for_both_models_and_allows_explicit_partial(tmp_path, capsys):
    cli = _score_cli()
    items, cand, base = _cli_files(tmp_path)
    partial_cand = dict(list(cand.items())[:-2])
    partial_cand["extra"] = {"a": 1.0}
    _write_jsonl(tmp_path / "cand.jsonl", ({"id": k, "probs": v} for k, v in partial_cand.items()))
    args = _args(tmp_path, "--unscaled")
    assert cli.main(args) == 2
    out, err = capsys.readouterr()
    assert "responses coverage 18/20 (90.0%)" in err and 'extra ids: ["extra"]' in err
    assert "Traceback" not in err and not out

    assert cli.main([*args, "--allow-partial"]) == 0
    scored = json.loads(capsys.readouterr().out)
    assert scored["detail"]["coverage"]["responses"] == {"answered": 18, "expected": 20, "fraction": 0.9, "extra_ids": ["extra"]}
    assert scored["detail"]["coverage"]["base_responses"]["answered"] == 20

    partial_base = dict(list(base.items())[:-2])
    _write_jsonl(tmp_path / "cand.jsonl", ({"id": k, "probs": v} for k, v in cand.items()))
    _write_jsonl(tmp_path / "base.jsonl", ({"id": k, "probs": v} for k, v in partial_base.items()))
    assert cli.main(args) == 2
    _, err = capsys.readouterr()
    assert "base responses coverage 18/20 (90.0%)" in err
    assert "Traceback" not in err
    assert cli.main([*args, "--allow-partial"]) == 0
    scored = json.loads(capsys.readouterr().out)
    assert scored["detail"]["coverage"]["base_responses"]["answered"] == 18


def test_cli_accepts_exactly_95_percent_coverage(tmp_path, capsys):
    cli = _score_cli()
    items, cand, base = _cli_files(tmp_path)
    cand.pop(next(iter(cand)))
    _write_jsonl(tmp_path / "cand.jsonl", ({"id": k, "probs": v} for k, v in cand.items()))
    assert cli.main(_args(tmp_path, "--unscaled")) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["detail"]["coverage"]["responses"]["fraction"] == 0.95


def test_cli_max_rows_is_per_file_and_invalid_params_exit_2(tmp_path, capsys):
    cli = _score_cli()
    _cli_files(tmp_path)
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    one.write_text('{"id": "one"}\n')
    two.write_text('{"id": "two"}\n')
    assert len(cli.read_jsonl([one, two], max_rows=1)) == 2

    rc = cli.main(_args(tmp_path, "--unscaled", "--max-rows", "19"))
    _, err = capsys.readouterr()
    assert rc == 2 and "items.jsonl:line 20: exceeds --max-rows=19" in err
    assert "Traceback" not in err

    rc = cli.main(_args(tmp_path, "--unscaled", "--resamples", "1"))
    _, err = capsys.readouterr()
    assert rc == 2 and "resamples must be an integer >= 2" in err
    assert "Traceback" not in err


def test_cli_end_to_end_on_a_synthetic_set_prints_json():
    # synthetic data from tests/data/v5/gen_synthetic.py: 9 T types + 3 N types + 2 typed public types + v4-format public / old tiers
    import os, subprocess, sys
    d = os.path.join(os.path.dirname(__file__), "data", "v5")
    root = os.path.dirname(os.path.dirname(__file__))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
    out = subprocess.run([sys.executable, os.path.join(root, "tools", "score_v5.py"),
                          "--items", os.path.join(d, "items_typed.jsonl"), "--items", os.path.join(d, "items_legacy.jsonl"),
                          "--responses", os.path.join(d, "cand.jsonl"), "--base-responses", os.path.join(d, "base.jsonl"),
                          "--panel-stats", os.path.join(d, "panel.json"), "--exclude-types", os.path.join(d, "exclude.json")],
                         capture_output=True, text=True, env=env, timeout=240)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert set(res) >= {"S_v5.1", "T", "N", "T_eq", "N_eq", "Pub", "G", "SE_G", "gap_flag"}
    assert isinstance(res["gap_flag"], bool) and res["overlap_excluded"] is True and res["detail"]["scaled"] is True
    assert res["S_v5.1"] == pytest.approx(0.6 * res["T_eq"] + 0.4 * res["N_eq"])
    assert set(res["detail"]["internal"]["tiers"]) == {"same_family", "proc_sealed", "new_family", "new_domain"}
    assert res["detail"]["n_T"] == 135 and res["detail"]["n_N"] == 45 and res["detail"]["n_pub"] == 30 + 60
    flat = subprocess.run([sys.executable, os.path.join(root, "tools", "score_v5.py"),
                           "--items", os.path.join(d, "items_typed.jsonl"), "--items", os.path.join(d, "items_legacy.jsonl"),
                           "--responses", os.path.join(d, "flat.jsonl"), "--base-responses", os.path.join(d, "base.jsonl"), "--unscaled"],
                          capture_output=True, text=True, env=env, timeout=240)
    assert flat.returncode == 0, flat.stderr
    assert json.loads(flat.stdout)["T"] < 5            # an all-uniform answerer sits at the zero point
