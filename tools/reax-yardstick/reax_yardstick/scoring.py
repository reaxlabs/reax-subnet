# Ported verbatim from the preview scorer (lib/scoring.py of the REAX preview pipeline): question_correct,
# item_score, score_journal, mean, sd, family_summary and weighted_aggregate are byte-identical to the
# functions the preview uses. Do not edit them here without changing the preview too (tests/test_kit.py pins
# their sha256).
"""Per-item scoring of journals against gold, and aggregates.

Journal row (written by reax_yardstick.run):
  {"run_id": ..., "status": "ok"|"error", "response": {"answers": {qid: {...}}}, ...}
Request row: {"questions": {qid: {"type": "choice"|"score"|"noul", ...}}, "_evaluation": {"run_id": ...}}
Gold row:    {"id": run_id, "answers": {qid: str | bool | None}}

Correctness of one question:
  choice / score : chosen key == gold key (as text)
  noul           : (p >= 0.5) == gold bool          (gold None -> question not scored)
Item score = mean over scored questions (0..1). Family score = mean over items,
SE = cluster-robust over items (items are the independent units).
Pure stdlib.
"""
import math

NOUL_THRESHOLD = 0.5


def question_correct(qtype, answer, gold):
    """True/False, or None when the question cannot be scored."""
    if gold is None or answer is None:
        return None
    if qtype == "noul":
        if not isinstance(gold, bool):
            return None
        p = answer.get("noul")
        if p is None:
            return None
        return (p >= NOUL_THRESHOLD) == gold
    chosen = answer.get("choice", answer.get("score"))
    if chosen is None:
        return None
    return str(chosen) == str(gold)


def item_score(request_row, gold_row, journal_row):
    """Return (score in 0..1 or None, n_scored_questions). None = errored / nothing scorable."""
    if journal_row.get("status") != "ok":
        return None, 0
    answers = journal_row["response"]["answers"]
    gold = gold_row["answers"]
    marks = []
    for qid, q in request_row["questions"].items():
        mark = question_correct(q["type"], answers.get(qid), gold.get(qid))
        if mark is not None:
            marks.append(1.0 if mark else 0.0)
    if not marks:
        return None, 0
    return sum(marks) / len(marks), len(marks)


def score_journal(requests, golds, journal):
    """requests: list of rows; golds: {id: row}; journal: list of journal rows.
    Returns {"items": {run_id: score}, "n_error": int, "n_unscorable": int, "n_missing": int}."""
    by_id = {r["_evaluation"]["run_id"]: r for r in requests}
    items, n_error, n_unscorable = {}, 0, 0
    seen = set()
    for j in journal:
        rid = j["run_id"]
        if rid not in by_id:
            continue
        seen.add(rid)
        if j.get("status") != "ok":
            n_error += 1
            continue
        g = golds.get(rid)
        if g is None:
            n_unscorable += 1
            continue
        s, n = item_score(by_id[rid], g, j)
        if s is None:
            n_unscorable += 1
        else:
            items[rid] = s
    return {"items": items, "n_error": n_error, "n_unscorable": n_unscorable, "n_missing": len(by_id) - len(seen)}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs, ddof=1):
    xs = list(xs)
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def family_summary(item_scores):
    """item_scores: {run_id: score}. Mean on a 0-100 scale, SE (cluster = item), n."""
    xs = list(item_scores.values())
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    return {"mean": 100.0 * mean(xs), "se": 100.0 * sd(xs) / math.sqrt(n) if n > 1 else float("nan"), "n": n}


def weighted_aggregate(family_stats, weights):
    """Weighted mean of family means; SE assuming independent families. Weights are renormalised
    over the families present (and reported), never silently dropped."""
    present = {k: w for k, w in weights.items() if k in family_stats and family_stats[k]["n"] > 0}
    total = sum(present.values())
    if total <= 0:
        return {"mean": float("nan"), "se": float("nan"), "weights_used": {}}
    w = {k: v / total for k, v in present.items()}
    m = sum(w[k] * family_stats[k]["mean"] for k in w)
    var = sum((w[k] * family_stats[k]["se"]) ** 2 for k in w if not math.isnan(family_stats[k]["se"]))
    return {"mean": m, "se": math.sqrt(var), "weights_used": w}
