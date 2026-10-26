"""Eval v5.1: the HF Decision Index 0.4 structure (docs/EVAL-V5.md; spec: handoff/SPEC-DELTA-v5.1.md, PROPOSAL.md 1.1-1.6).

Everything here is behind ``sampling.EVAL_V5_START_UTC`` (``None`` = off: rules v4 bit for bit). The module is pure and
importable without chain, validator state or GPU, so the same ``score_v5`` runs offline on a rented pod.

Draw (900 private items): T = 540 same-skill twins (weight ``1 - w_N``) + N = 360 new-domain tasks (weight ``w_N``, default 0.4,
fallback 0 = twins only). Public items (part ``pub``) and the old own-authored tiers are scored and reported with weight 0.

Score: ``S = (1 - w_N) T_eq + w_N N_eq``, ``X_eq = mB + sB (X - m_X) / s_X`` (m_X, s_X per pool version from the stock-panel run;
config, never computed per draw). T: per task type chance-corrected (0 = random or best constant answer, 100 = perfect), gold types
x1.2 inside their area, areas weighted ``HF_AREAS``. N: per task type, equal weight per task. No cap, no D terms. Gap check
(report only): ``G = (Pub - Pub_base) - (S - S_base)``, ``A = max(a_floor, a_se_mult SE_G)``, ``gap_flag = G > A``.

Item row schema (one JSON object per line; tier ``mirror`` = the private typed format)::

    {"id": str, "tier": "mirror", "part": "T"|"N"|"pub", "area": <area>, "task_type": str, "gold": bool, "source": str,
     "collected": "YYYY-MM-DD", "public": bool, "state": any,
     "idk": bool,                                    # fact items: an "I don't know" option (key ``IDK_KEY``) is offered
     "questions": [{"instructions": str, "criteria": {key: text, ...}, "expected": key}, ...]}

Task types ``omniscience`` (AA index, IDK offered) and ``wildguard`` (three yes/no questions) keep their S1-S2 scoring. A two-option
question with keys ``yes``/``no`` is a yes/no question (decided at p(yes) >= 0.5). An item with several questions is right only if
ALL its questions are right. A response is ``{key: prob}`` for one question or a list of them (one per question). Held-out rows in the
v4 format (``question.type``, ``labels``, ``expected``, tiers same_family / proc_sealed = public, new_family / new_domain /
cross_author = old own tiers) are scored by ``scoring.score_item`` and reported with weight 0.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field, fields, replace
from typing import Any, Collection, Iterable, Mapping, Sequence

import numpy as np

from . import prompt as P
from . import sampling as S
from . import scoring as SC

# The evaluator serves the typed T/N draw, the seated co-champions and every challenger's declared base on the same items and
# decides on S_v5.1 (spec/v5wire.py, SpecEvaluator._run_track / _decide); tests/test_v5_wiring.py runs a full simulated draw with fake
# models. ``Evaluator.__init__`` still refuses to start while the gate is set and this is False.
EVALUATOR_WIRED = True

IDK_KEY = "idk"
IDK_TEXT = "I don't know"
MAX_LABELS = 255
LETTERS = string.ascii_uppercase
JOINT_MARKER = "<|fim_suffix|>"   # joint-question-inline-readout-v1 (joint.JOINT_PROTOCOL): one marker per question

HF_AREAS = {"knowledge": 0.2585, "language": 0.2585, "retrieval": 0.2002, "tools": 0.1828, "arts": 0.10}
PARTS = ("T", "N", "pub")
OMNI_TYPE = "omniscience"
WILDGUARD_TYPE = "wildguard"
PUB_TIERS = ("same_family", "proc_sealed")                 # v4-format public tiers (weight 0, reported)
OLD_TIERS = ("new_family", "new_domain", S.X_TIER)         # v4-format old own-authored tiers (weight 0, reported)
OMNISCIENCE_MODES = ("aa_index", "half_half")
WILDGUARD_MODES = ("const_corrected", "raw")
MIRROR_MAX_USES = 12   # PROPOSAL 1.6: an item retires after 12 uses
MIN_TYPE_ITEMS = 12    # PROPOSAL 1.4: at least 12 items per task type per draw
DI_MEAN = 23.4973      # mB: DI's public-scale constants (SPEC-DELTA 3)
DI_SD = 16.539081      # sB


@dataclass(frozen=True)
class PanelStats:
    """Stock-panel mean and SD of T and N (and optionally Pub) on one pool version: config from the panel run, never per draw."""
    version: str
    t: tuple[float, float]
    n: tuple[float, float] | None = None
    pub: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("panel stats need a pool version")
        for name in ("t", "n", "pub"):
            ms = getattr(self, name)
            if ms is None:
                continue
            if (not isinstance(ms, (tuple, list)) or len(ms) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float, np.number))
                                                                          or not math.isfinite(float(v)) for v in ms)):
                raise ValueError(f"panel stats {name}: need a finite (mean, sd)")
            if float(ms[1]) <= 0:
                raise ValueError(f"panel stats {name}: sd must be > 0")
            object.__setattr__(self, name, (float(ms[0]), float(ms[1])))
        if self.t is None:
            raise ValueError("panel stats need T")


PANEL_STATS: dict[str, PanelStats] = {}   # pool version -> stats from the stock-panel run (filled below ``panel_from_json``)


def panel_from_json(obj: Any) -> PanelStats:
    """``{"pool_version": str, "T": {"mean": m, "sd": s}, "N": {...}, "Pub": {...}}``; N and Pub are optional."""
    if not isinstance(obj, Mapping):
        raise ValueError("panel stats must be a JSON object")
    def ms(key: str):
        v = obj.get(key)
        if v is None:
            return None
        if not isinstance(v, Mapping) or set(v) != {"mean", "sd"}:
            raise ValueError(f"panel stats {key}: need exactly mean and sd")
        return (v["mean"], v["sd"])
    if "T" not in obj:
        raise ValueError("panel stats need T")
    unknown = set(obj) - {"pool_version", "T", "N", "Pub"}
    if unknown:
        raise ValueError(f"panel stats: unknown keys {sorted(unknown)}")
    return PanelStats(obj.get("pool_version"), ms("T"), ms("N"), ms("Pub"))


# Pool v0.1 MAIN (round2-v5-monday ON-REPLY step 3, stock panel at w_N 0.4; Florian GO 23 Oct 2026). The evaluator takes the stats
# of the pool version named in ``<v5 pool dir>/v51.config.json`` and decides nothing without them (``v5wire.active_panel``).
PANEL_STATS["v0.1"] = panel_from_json({"pool_version": "v0.1", "T": {"mean": 58.443470442129176, "sd": 19.08197856407689},
                                       "N": {"mean": 28.490328496088534, "sd": 8.20230568361892},
                                       "Pub": {"mean": 46.68485197351358, "sd": 19.22683663229349}})


@dataclass(frozen=True)
class V5Params:
    """Weights, gap allowance and scale constants are parameters, fixed at the validation gate (PROPOSAL 1.1-1.5)."""
    w_n: float = 0.4                    # weight of N; 0 = the twins-only fallback (T = 1.0)
    a_floor: float = 1.5
    a_se_mult: float = 2.0
    hf_areas: Mapping[str, float] = field(default_factory=lambda: dict(HF_AREAS))
    gold_weight: float = 1.2            # gold task types weigh x1.2 inside their area
    m_b: float = DI_MEAN
    s_b: float = DI_SD
    panel: PanelStats | None = None     # None = unscaled (X_eq = X), flagged in the detail
    const_baseline: bool = True         # type zero point = max(random chance, best constant answer) for single-question types
    omniscience_mode: str = "aa_index"  # "aa_index": (100 + correct% - incorrect%) / 2, IDK = 0; "half_half": gate-only alternative
    omni_acc_weight: float = 0.5        # half_half only: weight of accuracy, the rest IDK/(IDK+wrong)
    wildguard_mode: str = "const_corrected"   # per label F1 chance-corrected against the best constant answer; "raw": plain F1
    t_items: int = 540
    n_items: int = 360
    pub_items: int = 0                  # public items run per draw for the gap check (weight 0); 0 = none drawn
    min_type_items: int = MIN_TYPE_ITEMS
    resamples: int = SC.BOOTSTRAP_RESAMPLES
    mirror_max_uses: int = MIRROR_MAX_USES

    def __post_init__(self) -> None:
        """Reject invalid gate parameters before they can affect an evaluation."""
        def finite_number(name: str, value: Any) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{name} must be a finite number")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            return number

        def share(name: str, value: Any) -> float:
            number = finite_number(name, value)
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            return number

        share("w_n", self.w_n)
        share("omni_acc_weight", self.omni_acc_weight)
        values = self.hf_areas
        if not isinstance(values, Mapping) or set(values) != set(HF_AREAS):
            raise ValueError(f"hf_areas must contain exactly {sorted(HF_AREAS)}")
        weights = [share(f"hf_areas[{key!r}]", value) for key, value in values.items()]
        if math.fsum(weights) <= 0:
            raise ValueError("hf_areas weights must sum to more than zero")
        for name in ("a_floor", "a_se_mult"):
            if finite_number(name, getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if finite_number("gold_weight", self.gold_weight) <= 0:
            raise ValueError("gold_weight must be > 0")
        finite_number("m_b", self.m_b)
        if finite_number("s_b", self.s_b) <= 0:
            raise ValueError("s_b must be > 0")
        if self.panel is not None and not isinstance(self.panel, PanelStats):
            raise ValueError("panel must be a PanelStats")
        if not isinstance(self.const_baseline, bool):
            raise ValueError("const_baseline must be a bool")
        for name in ("t_items", "n_items", "pub_items", "min_type_items"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if self.t_items + self.n_items <= 0:
            raise ValueError("t_items and n_items must sum to more than zero")
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int) or self.resamples < 2:
            raise ValueError("resamples must be an integer >= 2")
        if self.omniscience_mode not in OMNISCIENCE_MODES:
            raise ValueError(f"omniscience_mode must be one of {OMNISCIENCE_MODES}")
        if self.wildguard_mode not in WILDGUARD_MODES:
            raise ValueError(f"wildguard_mode must be one of {WILDGUARD_MODES}")


DEFAULT = V5Params()


# ------------------------------------------------------------------------------------------------------------ labels

# ------------------------------------------------------------------------------------------------------------ labels

def option_codes(k: int) -> list[str]:
    """Answer codes for ``k`` options: A..Z, then AA, AB, ... (two letters, 26 x 26 of them), at most 255. The first 26 are the
    single letters, as ``joint.detect`` requires of a readout's code list."""
    if not 2 <= k <= MAX_LABELS:
        raise P.PromptError(f"option count {k} outside 2..{MAX_LABELS}")
    return [LETTERS[i] if i < 26 else LETTERS[(i - 26) // 26] + LETTERS[(i - 26) % 26] for i in range(k)]


def question_keys(question: Mapping[str, Any], idk: bool = False) -> list[str]:
    crit = question.get("criteria")
    if not isinstance(crit, Mapping) or not crit:
        raise P.PromptError("question needs criteria")
    keys = [str(k) for k in crit]
    if idk and IDK_KEY not in keys:
        keys.append(IDK_KEY)
    return keys


def is_yesno(question: Mapping[str, Any]) -> bool:
    return set(map(str, question.get("criteria", {}))) == {"yes", "no"}


@dataclass(frozen=True)
class RenderedV5:
    system: str
    user: str
    codes: tuple[str, ...]
    keys: tuple[str, ...]


def _option_lines(question: Mapping[str, Any], idk: bool) -> tuple[list[str], list[str], list[str]]:
    keys = question_keys(question, idk)
    crit = question["criteria"]
    texts = [P._text(crit[k]) if k in crit else IDK_TEXT for k in keys]
    return keys, option_codes(len(keys)), texts


def render_question(state: Any, question: Mapping[str, Any], idk: bool = False) -> RenderedV5:
    """The v4 prompt (``prompt.render``) for up to 26 options without IDK, byte for byte; more options use two-letter codes
    (the causal path reads them as a code, not a single letter), IDK is an extra last option."""
    import json
    keys, codes, texts = _option_lines(question, idk)
    many = len(keys) > 26
    user = "SHARED STATE (JSON string):\n" + json.dumps(state, ensure_ascii=False) + "\n\n"
    user += "QUESTION:\n" + P._text(question.get("instructions", "")) + "\nOPTIONS:\n"
    user += "\n".join(f"{c}: {t}" for c, t in zip(codes, texts))
    user += "\nAnswer with one option code only." if many else "\nAnswer with one option letter only."
    system = P.SYSTEM.replace("one option letter", "one option code") if many else P.SYSTEM
    return RenderedV5(system, user, tuple(codes), tuple(keys))


def render_joint(item: Mapping[str, Any]) -> tuple[str, list[RenderedV5]]:
    """One prompt for ALL questions of an item (joint readout). Returns (user text, per-question option lists); the text has
    ``JOINT_MARKER`` after each question's options, where the joint model reads that question's answer."""
    import json
    idk = bool(item.get("idk"))
    parts = ["SHARED STATE (JSON string):\n" + json.dumps(item.get("state"), ensure_ascii=False) + "\n"]
    per: list[RenderedV5] = []
    for n, q in enumerate(item["questions"], 1):
        keys, codes, texts = _option_lines(q, idk)
        parts.append(f"QUESTION {n}:\n" + P._text(q.get("instructions", "")) + "\nOPTIONS:\n"
                     + "\n".join(f"{c}: {t}" for c, t in zip(codes, texts)) + "\n" + JOINT_MARKER)
        per.append(RenderedV5(P.SYSTEM, "", tuple(codes), tuple(keys)))
    return "\n\n".join(parts), per


def joint_probs(item: Mapping[str, Any], hidden: np.ndarray, weight: np.ndarray) -> list[dict[str, float]]:
    """Answer distributions of every question of an item from the joint model's hidden states at the markers
    (``hidden`` [n_questions, H], ``weight`` the readout [n_codes, H]): the first ``len(options)`` logits, the v4 temperature."""
    _, per = render_joint(item)
    if hidden.shape[0] != len(per):
        raise ValueError("one hidden state per question")
    out = []
    for h, r in zip(hidden, per):
        z = (weight[: len(r.codes)] @ h).tolist()
        out.append(dict(zip(r.keys, P.softmax_t(z))))
    return out


def causal_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The causal path answers each question of a multi-question item in its own call (shared state repeated)."""
    return [{"id": f"{item['id']}#{j}", "state": item.get("state"), "question": q, "idk": bool(item.get("idk"))}
            for j, q in enumerate(item["questions"])]



# ------------------------------------------------------------------------------------------------- item validation

def validate_mirror_row(row: Mapping[str, Any]) -> None:
    """Validate one typed (tier ``mirror``) row: part, area, task_type, gold, questions."""
    from .sampling import SampleError
    rid = row.get("id")
    if not isinstance(rid, str) or not rid:
        raise SampleError("typed row without an id")
    qs = row.get("questions")
    if not isinstance(qs, list) or not qs:
        raise SampleError(f"typed row {rid!r}: no questions")
    if row.get("part") not in PARTS:
        raise SampleError(f"typed row {rid!r}: part must be one of {PARTS}")
    if row.get("area") not in HF_AREAS:
        raise SampleError(f"typed row {rid!r}: invalid area")
    ttype = row.get("task_type")
    if not isinstance(ttype, str) or not ttype:
        raise SampleError(f"typed row {rid!r}: task_type must be a non-empty string")
    if not isinstance(row.get("gold"), bool):
        raise SampleError(f"typed row {rid!r}: gold must be a bool")
    idk = bool(row.get("idk"))
    if idk and (len(qs) != 1 or ttype != OMNI_TYPE):
        raise SampleError(f"typed row {rid!r}: IDK is only for single-question omniscience items")
    for q in qs:
        try:
            keys = question_keys(q, idk)
            if len(set(keys)) != len(keys) or len(keys) > MAX_LABELS or len(keys) < 2:
                raise P.PromptError("label count or duplicate keys")
        except P.PromptError as exc:
            raise SampleError(f"typed row {rid!r}: {exc}") from exc
        exp = str(q.get("expected"))
        if exp not in keys or (exp == IDK_KEY and idk):
            raise SampleError(f"typed row {rid!r}: expected answer not among the options")
    if ttype == WILDGUARD_TYPE and not (len(qs) == 3 and all(is_yesno(q) for q in qs)):
        raise SampleError(f"typed row {rid!r}: wildguard rows need exactly three yes/no questions")
    if ttype == OMNI_TYPE and not idk:
        raise SampleError(f"typed row {rid!r}: omniscience items offer IDK")


# ------------------------------------------------------------------------------------------------------------ draw

def _cell_take(rows: list[dict], count: int, rng: random.Random, use: Mapping[str, int], what: str) -> list[dict]:
    """``count`` rows of one cell: fresh first, least used next, random inside equal use counts. Fail closed on a shortage."""
    if len(rows) < count:
        raise S.SampleError(f"eval v5 draw: {what}: only {len(rows)} eligible items, need {count}")
    rows = list(rows)
    rng.shuffle(rows)
    rows.sort(key=lambda r: use.get(r["id"], 0))
    return rows[:count]


def _with_floor(quota: dict[str, int], floor: int, what: str) -> dict[str, int]:
    """Raise every quota below ``floor`` to it, paying from the largest quotas above the floor."""
    quota = dict(quota)
    if sum(quota.values()) < floor * len(quota):
        raise S.SampleError(f"eval v5 draw: {what}: {sum(quota.values())} items cannot give {len(quota)} types {floor} each")
    for t in sorted(quota):
        while quota[t] < floor:
            donor = max((k for k in quota if quota[k] > floor), key=lambda k: (quota[k], k))
            quota[donor] -= 1
            quota[t] += 1
    return quota


def type_cells(rows: Sequence[Mapping[str, Any]], params: V5Params = DEFAULT) -> dict[tuple, int]:
    """Items per (part, task_type) cell of one draw from the typed rows' types. T: area quotas by ``hf_areas``, inside an area
    by type weight (gold x ``gold_weight``), every type at least ``min_type_items``; N: equal per task type, same floor."""
    cells: dict[tuple, int] = {}
    meta = type_meta([r for r in rows if r.get("part") in ("T", "N")])
    t_types = {t: m for t, m in meta.items() if m[2] == "T"}
    n_types = sorted(t for t, m in meta.items() if m[2] == "N")
    if params.t_items:
        by_area: dict[str, dict[str, float]] = {}
        for t, (area, gold, _) in t_types.items():
            by_area.setdefault(area, {})[t] = params.gold_weight if gold else 1.0
        missing = [a for a in params.hf_areas if params.hf_areas[a] > 0 and a not in by_area]
        if missing:
            raise S.SampleError(f"eval v5 draw: T has no task type in area(s) {missing}")
        for area, n in S._quotas(params.t_items, {a: params.hf_areas[a] for a in by_area}).items():
            for t, k in _with_floor(S._quotas(n, by_area[area]), params.min_type_items, f"T/{area}").items():
                cells[("T", t)] = k
    if params.n_items:
        if not n_types:
            raise S.SampleError("eval v5 draw: no N task types in the pool")
        for t, k in _with_floor(S._quotas(params.n_items, {t: 1.0 for t in n_types}), params.min_type_items, "N").items():
            cells[("N", t)] = k
    return cells


def _row_part(row: Mapping[str, Any]) -> str | None:
    """"T" / "N" / "pub" for typed rows, "pub" for v4-format public tiers, "old" for the old own tiers, else None."""
    tier = row.get("tier")
    if tier == S.MIRROR_TIER:
        return row.get("part")
    if tier in PUB_TIERS:
        return "pub"
    if tier in OLD_TIERS or tier in S.X_TIERS:
        return "old"
    return None


def draw_v5(pool: Sequence[dict[str, Any]], *, n: int, seed: int, use_counts: Mapping[str, int],
            exclude: Collection[str] = frozenset(), params: V5Params = DEFAULT) -> tuple[list[dict], bool]:
    """900 private items: T 540 and N 360 by task type (``type_cells``), plus ``params.pub_items`` public items for the gap check.
    Items in ``exclude`` and retired items are never picked; a type with too few eligible items raises ``SampleError`` (fail closed:
    no silent borrowing across types). Returns (rows sorted by id, fresh)."""
    total = params.t_items + params.n_items + params.pub_items
    if n != total:
        raise S.SampleError(f"eval v5 draws require exactly {total} items")
    S._validate_generalist(pool)
    rng = random.Random(seed)
    use = use_counts
    ok = [r for r in sorted(pool, key=lambda r: r["id"]) if r["id"] not in exclude]
    typed = [r for r in ok if r.get("tier") == S.MIRROR_TIER]
    chosen: list[dict] = []
    elig = [r for r in typed if use.get(r["id"], 0) < params.mirror_max_uses]
    by_cell: dict[tuple, list[dict]] = {}
    for r in elig:
        if r["part"] in ("T", "N"):
            by_cell.setdefault((r["part"], r["task_type"]), []).append(r)
    # the cells come from ALL typed rows of the pool: a type whose items are retired or excluded fails the draw (it must not
    # silently vanish and hand its share to the other types)
    for cell, count in type_cells(pool, params).items():
        chosen += _cell_take(by_cell.get(cell, []), count, rng, use, f"type {cell[0]}/{cell[1]}")
    if params.pub_items:
        pub = [r for r in ok if _row_part(r) == "pub"
               and use.get(r["id"], 0) < (params.mirror_max_uses if r.get("tier") == S.MIRROR_TIER else S.ITEM_MAX_USES)]
        chosen += _cell_take(pub, params.pub_items, rng, use, "public items")
    chosen.sort(key=lambda r: r["id"])
    return chosen, all(use.get(r["id"], 0) == 0 for r in chosen)



# ---------------------------------------------------------------------------------------------------- item scoring

def _as_list(resp: Any, n_q: int) -> list[Any]:
    if n_q == 1 and isinstance(resp, Mapping):
        return [resp]
    if isinstance(resp, (list, tuple)) and len(resp) == n_q:
        return list(resp)
    return [None] * n_q


def top_distribution(p: Mapping[str, float]) -> dict[str, float]:
    """The prediction of a cleaned answer distribution as probability mass over its top options: the single top option, or
    1/k on each of k options tied EXACTLY at the maximum (no first-key advantage; the expected value of a random tie-break)."""
    best = max(p.values())
    tied = [k for k, v in p.items() if v == best]
    return {k: 1.0 / len(tied) for k in tied}


@dataclass(frozen=True)
class QuestionCredit:
    right: float          # 1/k if gold is among the k exactly tied top options, else 0 (k = 1: plain right / wrong)
    idk: float            # same for the IDK option (0 when none is offered)
    yes: float            # mass on "yes" (yes/no questions; the WildGuard counts), else 0
    valid: bool


def question_credit(question: Mapping[str, Any], probs: Any, idk: bool = False) -> QuestionCredit:
    """Credit of one answer. An unusable answer (not exactly over the question's keys, non-finite, not summing to ~1) is wrong
    with zero credit, never IDK. Otherwise the top option(s): exact ties among k options earn 1/k if gold is among them, so a
    yes/no answer at exactly p(yes) = 0.5 is a 2-way tie worth 1/2 (gold is always yes or no)."""
    p = SC.clean_probs(probs, question_keys(question, idk))
    if p is None:
        return QuestionCredit(0.0, 0.0, 0.0, False)
    top = top_distribution(p)
    return QuestionCredit(top.get(str(question["expected"]), 0.0), top.get(IDK_KEY, 0.0) if idk else 0.0,
                          top.get("yes", 0.0), True)


def question_outcome(question: Mapping[str, Any], probs: Any, idk: bool = False) -> tuple[str, str | None]:
    """(outcome, predicted key) of an answer WITHOUT an exact tie, for reporting: right / wrong / idk. An exact tie has no single
    prediction: it returns ("tie", None); the score uses ``question_credit``."""
    c = question_credit(question, probs, idk)
    if not c.valid:
        return "wrong", None
    p = SC.clean_probs(probs, question_keys(question, idk))
    top = top_distribution(p)
    if len(top) > 1:
        return "tie", None
    pred = next(iter(top))
    if idk and pred == IDK_KEY:
        return "idk", pred
    return ("right" if pred == str(question["expected"]) else "wrong"), pred


@dataclass(frozen=True)
class CreditItem(SC.ItemScore):
    """A v4 ``ItemScore`` plus the v5 fractional credit of tied top options (``correct`` stays the v4 boolean). The v4
    scoring module is not touched (it must stay byte-identical, tests/test_joint_readout.py); ``_arrays`` below applies it."""
    credit: float | None = None


def _arrays(items: Sequence[SC.ItemScore]) -> dict[str, dict[str, np.ndarray]]:
    """``scoring._arrays`` with the v5 tie credit in place of the 0/1 ``correct`` where an item carries one."""
    arrs = SC._arrays(items)
    for kind, a in arrs.items():
        for j, idx in enumerate(a["idx"].astype(int)):
            c = getattr(items[idx], "credit", None)
            if c is not None:
                a["correct"][j] = float(c)
    return arrs


def pool_credit(row: Mapping[str, Any], probs: Any, item: SC.ItemScore) -> SC.ItemScore:
    """v5 tie rule for held-out ``choice`` items: exact ties among the top labels earn 1/k if gold is among them (v4 gives the
    lexicographically first label everything). Other kinds and the v4 scorer itself are unchanged."""
    if item.kind != "choice":
        return item
    p = SC.clean_probs(probs, [str(x) for x in row["labels"]])
    credit = top_distribution(p).get(str(row["expected"]), 0.0) if p else 0.0
    return CreditItem(**{f.name: getattr(item, f.name) for f in fields(SC.ItemScore)}, credit=credit)


# ------------------------------------------------------------------------------------------------------ type tables

def type_meta(rows: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, bool, str]]:
    """task_type -> (area, gold, part) of typed rows; a type must be one area, one gold flag and one part."""
    meta: dict[str, tuple[str, bool, str]] = {}
    for r in rows:
        t = r["task_type"]
        m = (r["area"], bool(r["gold"]), r["part"])
        if meta.setdefault(t, m) != m:
            raise ValueError(f"task type {t!r} has conflicting area / gold / part metadata ({meta[t]} vs {m}, item {r.get('id')!r})")
    return meta


@dataclass
class _Table:
    """Per-item arrays of one model's run on typed items (indices follow the item list)."""
    task: list[str]
    correct: np.ndarray
    chance: np.ndarray
    idk: np.ndarray
    wrong: np.ndarray
    tp: np.ndarray                # [n, 3] wildguard counts (zero elsewhere)
    fp: np.ndarray
    fn: np.ndarray
    gold_code: np.ndarray         # int code of the gold key of single-question items, -1 for multi-question items
    idx: dict = field(default_factory=dict)   # task type -> item indices


def item_table(items: Sequence[Mapping[str, Any]], responses: Mapping[str, Any]) -> _Table:
    """Per-item expected outcomes. ``correct`` / ``idk`` / ``wrong`` are fractions (exact ties split 1/k, see
    ``question_credit``) and sum to 1; the WildGuard counts are the expected tp / fp / fn of the same tie split."""
    n = len(items)
    t = _Table([], np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)),
               np.full(n, -1, dtype=int))
    codes: dict[str, int] = {}
    for i, it in enumerate(items):
        qs = it["questions"]
        idk = bool(it.get("idk"))
        cr = [question_credit(q, r, idk) for q, r in zip(qs, _as_list(responses.get(it["id"]), len(qs)))]
        t.task.append(it["task_type"])
        right = math.prod(c.right for c in cr)
        t.correct[i] = right
        t.idk[i] = cr[0].idk if idk else 0.0
        t.wrong[i] = 1.0 - right - t.idk[i]
        chance = 1.0
        for q in qs:
            chance *= 1.0 / len(question_keys(q, idk))
        t.chance[i] = chance
        if len(qs) == 1:
            t.gold_code[i] = codes.setdefault(str(qs[0]["expected"]), len(codes))
        if it["task_type"] == WILDGUARD_TYPE:
            for j, (q, c) in enumerate(zip(qs, cr)):
                if str(q["expected"]) == "yes":
                    t.tp[i, j], t.fn[i, j] = c.yes, 1.0 - c.yes
                else:
                    t.fp[i, j] = c.yes
    for i, g in enumerate(t.task):
        t.idx.setdefault(g, []).append(i)
    t.idx = {g: np.array(v) for g, v in t.idx.items()}
    return t


def wildguard_label_scores(tp: float, fp: float, fn: float, total: float, mode: str) -> tuple[float, float, float]:
    """(score, raw F1, F1_const) of one WildGuard label, all in 0..1, from the (weighted) expected counts over ``total`` items.
    F1_const is the F1 of the better of always-yes (F1 = 2P / (P + total), P = gold positives) and always-no (F1 = 0, no yes
    predicted). const_corrected: max(0, (F1 - F1_const) / (1 - F1_const)); when F1_const = 1 (every gold answer is yes) the
    score is 1 if F1 = 1 and 0 otherwise. raw: the F1."""
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom > 0 else 0.0
    pos = tp + fn
    f1_const = max(2 * pos / (pos + total) if pos + total > 0 else 0.0, 0.0)
    if mode == "raw":
        return f1, f1, f1_const
    if f1_const >= 1.0 - 1e-12:
        return (1.0 if f1 >= 1.0 - 1e-12 else 0.0), f1, f1_const
    return max(0.0, min(1.0, (f1 - f1_const) / (1.0 - f1_const))), f1, f1_const


def _clip(x: float) -> float:
    return max(0.0, min(100.0, x))


def type_scores(t: _Table, w: np.ndarray | None = None, params: V5Params = DEFAULT,
                details: dict[str, Any] | None = None) -> dict[str, float]:
    """Score 0..100 of every task type present, with item weights ``w`` (None = ones). 0 = random or the best constant answer,
    100 = perfect. ``details`` (optional) receives extras: WildGuard raw F1 and F1_const.

    Generic type: ``100 (acc - c) / (1 - c)`` clipped, c = the random-answer accuracy, raised to the best constant answer's accuracy
    when every item has one question and ``const_baseline``. Omniscience (aa_index): (100 + correct% - incorrect%) / 2 with IDK = 0
    (always-IDK 50, 60/30/10 -> 65, blind 25/75 -> 25, perfect 100; S1 keeps its own scale); half_half keeps the old accuracy /
    IDK-share mix. WildGuard: see ``wildguard_label_scores`` (S2)."""
    w = np.ones(len(t.task)) if w is None else w
    scores: dict[str, float] = {}
    for name in sorted(t.idx):
        idx = t.idx[name]
        ww = w[idx]
        tot = float(ww.sum())
        if tot <= 0:
            continue
        if name == OMNI_TYPE:
            acc = float((ww * t.correct[idx]).sum() / tot)
            idk, wr = float((ww * t.idk[idx]).sum()), float((ww * t.wrong[idx]).sum())
            if params.omniscience_mode == "aa_index":   # AA Omniscience index: correct% - incorrect%, IDK = 0, mapped to 0..100
                sc = 50 * (1 + acc - wr / tot)
            else:
                nh = idk / (idk + wr) if idk + wr > 0 else 1.0
                sc = 100 * (params.omni_acc_weight * acc + (1 - params.omni_acc_weight) * nh)
        elif name == WILDGUARD_TYPE:
            lab, raw, const = [], [], []
            for j in range(3):
                tp, fp, fn = (float((ww * a[idx, j]).sum()) for a in (t.tp, t.fp, t.fn))
                sj, rj, cj = wildguard_label_scores(tp, fp, fn, tot, params.wildguard_mode)
                lab.append(sj), raw.append(rj), const.append(cj)
            sc = 100 * sum(lab) / 3
            if details is not None:
                details[name] = {"raw_f1": 100 * sum(raw) / 3, "f1_const": 100 * sum(const) / 3}
        else:
            acc = float((ww * t.correct[idx]).sum() / tot)
            c = float((ww * t.chance[idx]).sum() / tot)
            codes = t.gold_code[idx]
            if params.const_baseline and (codes >= 0).all():   # best constant answer: the most frequent gold key
                c = max(c, float(np.bincount(codes, weights=ww).max() / tot))
            sc = _clip(100 * (acc - c) / (1 - c)) if c < 1 - 1e-12 else 0.0
        scores[name] = sc
    return scores


def aggregate_areas(ts: Mapping[str, float], meta: Mapping[str, tuple], params: V5Params = DEFAULT) -> tuple[float | None, dict[str, float]]:
    """DI aggregation: the area score is the mean of its task types (gold x ``gold_weight``); the score is the area-weighted mean
    over the areas present (weights renormalised). Returns (score or None if no type, area scores)."""
    areas: dict[str, float] = {}
    for area in HF_AREAS:
        num = den = 0.0
        for t, s in ts.items():
            a, gold, _ = meta[t]
            if a == area:
                wt = params.gold_weight if gold else 1.0
                num += wt * s
                den += wt
        if den > 0:
            areas[area] = num / den
    wsum = sum(params.hf_areas[a] for a in areas)
    if wsum <= 0:
        return None, areas
    return sum(params.hf_areas[a] * s for a, s in areas.items()) / wsum, areas


def aggregate_tasks(ts: Mapping[str, float]) -> float | None:
    """N: every task (task type) counts the same."""
    return sum(ts.values()) / len(ts) if ts else None


# ------------------------------------------------------------------------------------------------------ the score

@dataclass
class Split:
    """Items of one run by role. ``pub_typed`` / ``pub_legacy`` and ``old`` have weight 0 (reported)."""
    t: list[Mapping[str, Any]]
    n: list[Mapping[str, Any]]
    pub_typed: list[Mapping[str, Any]]
    pub_legacy: list[Mapping[str, Any]]
    old: list[Mapping[str, Any]]
    ignored: int = 0

    @property
    def scored(self) -> list[Mapping[str, Any]]:
        return [*self.t, *self.n, *self.pub_typed, *self.pub_legacy, *self.old]


def split_items(items: Iterable[Mapping[str, Any]], strict: bool = True) -> Split:
    """Items by role. Typed rows go by ``part``; v4-format rows by tier (public tiers, old own tiers). Unknown tiers raise unless
    ``strict`` is False, in which case they are dropped and counted."""
    sp = Split([], [], [], [], [])
    for it in items:
        tier = it.get("tier")
        if tier == S.MIRROR_TIER:
            part = it.get("part")
            if part not in PARTS:
                raise ValueError(f"item {it.get('id')!r}: part {part!r} is not one of {PARTS}")
            {"T": sp.t, "N": sp.n, "pub": sp.pub_typed}[part].append(it)
        elif tier in PUB_TIERS:
            sp.pub_legacy.append(it)
        elif tier in OLD_TIERS or tier in S.X_TIERS:
            sp.old.append(it)
        elif strict:
            raise ValueError(f"item {it.get('id')!r}: tier {tier!r} is not part of the v5.1 score")
        else:
            sp.ignored += 1
    return sp


def type_coverage(split: Split, min_items: int = MIN_TYPE_ITEMS) -> dict[str, int]:
    """Task types of T and N with fewer than ``min_items`` items (type -> count). Empty = fine."""
    counts: dict[str, int] = {}
    for it in (*split.t, *split.n):
        counts[it["task_type"]] = counts.get(it["task_type"], 0) + 1
    return {t: c for t, c in sorted(counts.items()) if c < min_items}


def pool_scores(rows: Sequence[Mapping[str, Any]], responses: Mapping[str, Any]) -> list[SC.ItemScore]:
    out = []
    for r in rows:
        probs = responses.get(r["id"])
        out.append(pool_credit(r, probs, SC.score_item(r, probs)))
    return out


def _p_total(arrs: dict, w: np.ndarray | None) -> float:
    comp, cal, _ = SC._score_weighted(arrs, w)
    return SC.combine(comp, cal)


@dataclass
class _Side:
    """One typed part (T, N or typed public) of an entry and its base."""
    kind: str
    n: int
    te: _Table
    tb: _Table
    meta: dict


@dataclass
class _Legacy:
    """v4-format public rows of an entry and its base."""
    n: int
    ae: dict
    ab: dict


def _part_scores(side: _Side, w: np.ndarray | None, params: V5Params, excl: Collection[str],
                 de: dict | None = None, db: dict | None = None) -> dict[str, Any]:
    """Raw part scores of entry and base. Overlap exclusion: an excluded type scores min(entry, base) for the entry."""
    ts_e, ts_b = type_scores(side.te, w, params, de), type_scores(side.tb, w, params, db)
    for t in excl:
        if t in ts_e and t in ts_b:
            ts_e[t] = min(ts_e[t], ts_b[t])
    if side.kind == "N":
        xe, xb, ae, ab = aggregate_tasks(ts_e), aggregate_tasks(ts_b), {}, {}
    else:
        (xe, ae), (xb, ab) = aggregate_areas(ts_e, side.meta, params), aggregate_areas(ts_b, side.meta, params)
    return {"x": xe, "x_b": xb, "types": ts_e, "types_b": ts_b, "areas": ae, "areas_b": ab}


def _scale(x: float | None, ms: tuple[float, float] | None, params: V5Params) -> float | None:
    """``X_eq = mB + sB (X - m_X) / s_X``; no stats for X (or no panel at all): X unchanged."""
    if x is None or ms is None:
        return x
    return params.m_b + params.s_b * (x - ms[0]) / ms[1]


def _stats(sides: Mapping[str, _Side | None], leg: _Legacy | None, w: Mapping[str, np.ndarray | None], params: V5Params,
           excl: Collection[str], extra: dict | None = None) -> dict[str, Any]:
    """All headline numbers of entry and base for item weights ``w`` (per part; missing = ones)."""
    panel = params.panel
    out: dict[str, Any] = {}
    parts = {}
    for k in ("T", "N", "pub"):
        side = sides.get(k)
        parts[k] = _part_scores(side, w.get(k), params, excl,
                                extra.setdefault(f"{k}_e", {}) if extra is not None else None,
                                extra.setdefault(f"{k}_b", {}) if extra is not None else None) if side else None
    t, n, pt = parts["T"], parts["N"], parts["pub"]
    for key, p, ms in (("T", t, panel.t if panel else None), ("N", n, panel.n if panel else None)):
        out[key] = p["x"] if p else None
        out[key + "_b"] = p["x_b"] if p else None
        out[key + "_eq"] = _scale(p["x"], ms, params) if p else None
        out[key + "_eq_b"] = _scale(p["x_b"], ms, params) if p else None
    pub_ms = panel.pub if panel else None
    pubs, pubs_b, counts = [], [], []
    if pt:
        pubs.append(pt["x"]), pubs_b.append(pt["x_b"]), counts.append(sides["pub"].n)
    if leg:
        wl = w.get("legacy")
        pubs.append(_p_total(leg.ae, wl)), pubs_b.append(_p_total(leg.ab, wl)), counts.append(leg.n)
    if counts and all(v is not None for v in (*pubs, *pubs_b)):
        tot = sum(counts)
        pub = sum(c * v for c, v in zip(counts, pubs)) / tot
        pub_b = sum(c * v for c, v in zip(counts, pubs_b)) / tot
        out["Pub"], out["Pub_b"] = _scale(pub, pub_ms, params), _scale(pub_b, pub_ms, params)
    else:
        out["Pub"] = out["Pub_b"] = None
    wn = params.w_n
    for suf in ("", "_b"):
        te, ne = out["T_eq" + suf], out["N_eq" + suf]
        out["S" + suf] = (1 - wn) * te + (wn * ne if wn else 0.0)
    out["G"] = ((out["Pub"] - out["Pub_b"]) - (out["S"] - out["S_b"])) if out["Pub"] is not None else None
    out["_parts"] = parts
    return out


@dataclass
class V5Result:
    s: float
    t: float
    n: float | None
    t_eq: float
    n_eq: float | None
    pub: float | None
    g: float | None
    se_g: float | None
    a: float | None
    gap_flag: bool | None
    overlap_excluded: bool
    detail: dict[str, Any]

    def as_tuple(self) -> tuple[float, float, float | None, float | None, float | None]:
        return self.s, self.t, self.n, self.g, self.se_g

    def as_dict(self) -> dict[str, Any]:
        """The CLI output: headline numbers plus the internal detail (per area / type scores, coverage, params)."""
        return {"S_v5.1": self.s, "T": self.t, "N": self.n, "T_eq": self.t_eq, "N_eq": self.n_eq, "Pub": self.pub, "G": self.g,
                "SE_G": self.se_g, "A": self.a, "gap_flag": self.gap_flag, "overlap_excluded": self.overlap_excluded,
                "detail": self.detail}


PUBLIC_KEYS = ("S_v5.1", "T_eq", "N_eq", "Pub", "gap_flag", "overlap_excluded")


def public_view(out: Mapping[str, Any]) -> dict[str, Any]:
    """The part of ``V5Result.as_dict`` fit for public surfaces: no per-area / per-type scores, no type ids (overlap is only the
    flag ``overlap_excluded``)."""
    return {k: out[k] for k in PUBLIC_KEYS if k in out}


def _check_base(base: Mapping[str, Any] | None, items: Sequence[Mapping[str, Any]], allow_partial: bool = False) -> None:
    if base is None:
        raise ValueError("base responses are required: S_v5.1 is measured against the declared base on the same items")
    missing = [it["id"] for it in items if it["id"] not in base]
    if missing and not allow_partial:
        raise ValueError(f"base responses missing for {len(missing)} item(s), e.g. {missing[:3]}")


def response_coverage(responses: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage of the scored item IDs, with extra response IDs listed for audit."""
    expected_ids = {item["id"] for item in items}
    answered = sum(item_id in responses for item_id in expected_ids)
    extras = set(responses) - expected_ids
    return {
        "answered": answered,
        "expected": len(expected_ids),
        "fraction": answered / len(expected_ids) if expected_ids else 0.0,
        "extra_ids": sorted(extras, key=lambda item_id: (type(item_id).__name__, repr(item_id))),
    }


def _f(x: Any) -> Any:
    """JSON-safe: numpy numbers and bools to Python ones."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return float(x)


def paired_s_se(items: Sequence[Mapping[str, Any]], a: Mapping[str, Any], b: Mapping[str, Any], p: V5Params,
                seed: int = 0) -> float:
    """Paired bootstrap SE of ``S_v5.1(a) - S_v5.1(b)`` over the items of every part (the same resample for both models, like
    ``score_v5_detail``'s SE_G), with ``b`` in the place of the base. ``inf`` when fewer than 2 T items. Deterministic in ``seed``."""
    sp = split_items(items)
    sides: dict[str, Any] = {}
    for kind, rows in (("T", sp.t), ("N", sp.n), ("pub", sp.pub_typed)):
        sides[kind] = (_Side(kind, len(rows), item_table(rows, a), item_table(rows, b), type_meta(rows)) if rows else None)
    leg = (_Legacy(len(sp.pub_legacy), _arrays(pool_scores(sp.pub_legacy, a)), _arrays(pool_scores(sp.pub_legacy, b)))
           if sp.pub_legacy else None)
    sizes = {k: s.n for k, s in sides.items() if s}
    if leg:
        sizes["legacy"] = leg.n
    if sizes.get("T", 0) < 2:
        return math.inf
    rng = np.random.RandomState((seed + len(items)) % (2 ** 32))
    ds = np.empty(p.resamples)
    for r in range(p.resamples):
        w = {k: np.bincount(rng.randint(0, n, size=n), minlength=n).astype(float) for k, n in sizes.items()}
        st = _stats(sides, leg, w, p, ())
        ds[r] = st["S"] - st["S_b"]
    return float(ds.std(ddof=1))


def score_v5_detail(responses: Mapping[str, Any], items: Sequence[Mapping[str, Any]], base_responses: Mapping[str, Any],
                    params: V5Params = DEFAULT, seed: int = 0, strict: bool = True, allow_partial: bool = False,
                    exclude_types: Collection[str] = ()) -> V5Result:
    """``S_v5.1`` of one model against its declared base on ``items`` (T, N, public and old-tier items of one or more pooled
    draws). ``responses`` / ``base_responses``: item id -> answer (see the module docstring). A missing candidate answer is a failed
    item; a missing base answer is an error unless ``allow_partial`` is explicitly enabled. ``exclude_types``: task types whose score
    is min(entry, base) for this entry (overlap rule); the result carries only ``overlap_excluded``, the ids stay in the detail."""
    sp = split_items(items, strict)
    wn = params.w_n
    if len(sp.t) < 2:
        raise ValueError("v5.1 needs at least two T items")
    if wn > 0 and len(sp.n) < 2:
        raise ValueError("v5.1 needs at least two N items while w_N > 0")
    if not isinstance(exclude_types, (list, tuple, set, frozenset)) or not all(isinstance(t, str) for t in exclude_types):
        raise ValueError("exclude_types must be a list of task type ids")
    excl = sorted(set(exclude_types))
    if params.panel is not None and wn > 0 and params.panel.n is None:
        raise ValueError("panel stats lack N while w_N > 0")
    scored_items = sp.scored
    _check_base(base_responses, scored_items, allow_partial=allow_partial)
    meta_all = type_meta([*sp.t, *sp.n, *sp.pub_typed])
    sides: dict[str, _Side | None] = {}
    for kind, rows in (("T", sp.t), ("N", sp.n), ("pub", sp.pub_typed)):
        sides[kind] = (_Side(kind, len(rows), item_table(rows, responses), item_table(rows, base_responses),
                             type_meta(rows)) if rows else None)
    leg = None
    if sp.pub_legacy:
        leg = _Legacy(len(sp.pub_legacy), _arrays(pool_scores(sp.pub_legacy, responses)),
                      _arrays(pool_scores(sp.pub_legacy, base_responses)))
    extra: dict[str, Any] = {}
    pt = _stats(sides, leg, {}, params, excl, extra)
    # paired bootstrap over the items of every part: the same resample for the entry and its base
    rng = np.random.RandomState((seed + len(items)) % (2 ** 32))
    sizes = {k: s.n for k, s in sides.items() if s}
    if leg:
        sizes["legacy"] = leg.n
    gs = np.empty(params.resamples)
    has_pub = pt["G"] is not None
    for r in range(params.resamples):
        w = {k: np.bincount(rng.randint(0, n, size=n), minlength=n).astype(float) for k, n in sizes.items()}
        g = _stats(sides, leg, w, params, excl)["G"]
        gs[r] = g if g is not None else 0.0
    se_g = float(gs.std(ddof=1)) if has_pub else None
    allowance = max(params.a_floor, params.a_se_mult * se_g) if has_pub else None
    # critic r1 F1 / spec owner #23830: G compares Pub with S, so it is only meaningful when both are on the same scale (both scaled by
    # the stock panel, or both raw); with a panel but no Pub stats G mixes units: reported, but gap_flag is None and gap_scaled False
    gap_scaled = params.panel is None or bool(params.panel.pub)
    gap_flag = bool(pt["G"] > allowance) if has_pub and gap_scaled else None
    # old own tiers and public tiers: reported with weight 0
    tier_scores = {}
    for tier in (*PUB_TIERS, *OLD_TIERS):
        rows = [r for r in (*sp.pub_legacy, *sp.old) if (S.X_TIER if r["tier"] in S.X_TIERS else r["tier"]) == tier]
        if rows:
            tier_scores[tier] = {"n": len(rows), "score": _f(_p_total(_arrays(pool_scores(rows, responses)), None)),
                                 "base": _f(_p_total(_arrays(pool_scores(rows, base_responses)), None))}
    parts = pt.pop("_parts")
    panel = params.panel
    detail = {
        "S_v5.1_base": _f(pt["S_b"]), "T_base": _f(pt["T_b"]), "N_base": _f(pt["N_b"]), "T_eq_base": _f(pt["T_eq_b"]),
        "N_eq_base": _f(pt["N_eq_b"]), "Pub_base": _f(pt["Pub_b"]), "w_N": wn,
        "n_T": len(sp.t), "n_N": len(sp.n), "n_pub": len(sp.pub_typed) + len(sp.pub_legacy), "n_old": len(sp.old),
        "ignored_items": sp.ignored,
        "scaled": panel is not None, "pool_version": panel.version if panel else None,
        "pub_scaled": bool(panel and panel.pub), "gap_scaled": gap_scaled, "gap_check": "report-only (no score effect)",
        "coverage": {"responses": response_coverage(responses, scored_items),
                     "base_responses": response_coverage(base_responses, scored_items)},
        "internal": {
            "areas_T": {k: _f(v) for k, v in parts["T"]["areas"].items()} if parts["T"] else {},
            "areas_T_base": {k: _f(v) for k, v in parts["T"]["areas_b"].items()} if parts["T"] else {},
            "areas_pub": {k: _f(v) for k, v in parts["pub"]["areas"].items()} if parts["pub"] else {},
            "types_T": {k: _f(v) for k, v in parts["T"]["types"].items()} if parts["T"] else {},
            "types_T_base": {k: _f(v) for k, v in parts["T"]["types_b"].items()} if parts["T"] else {},
            "types_N": {k: _f(v) for k, v in parts["N"]["types"].items()} if parts["N"] else {},
            "types_N_base": {k: _f(v) for k, v in parts["N"]["types_b"].items()} if parts["N"] else {},
            "types_pub": {k: _f(v) for k, v in parts["pub"]["types"].items()} if parts["pub"] else {},
            "type_meta": {k: {"area": a, "gold": g, "part": p} for k, (a, g, p) in sorted(meta_all.items())},
            "type_items": {k: len(v) for side in (sides["T"], sides["N"]) if side for k, v in sorted(side.te.idx.items())},
            "overlap_excluded_types": excl,
            "overlap_unmatched_types": sorted(t for t in excl if t not in meta_all or meta_all[t][2] == "pub"),
            "tiers": tier_scores,
            "wildguard_extra": {k: {kk: _f(vv) for kk, vv in v.items()} for k, v in (extra.get("T_e") or {}).items()},
        },
        "params": {"w_n": params.w_n, "a_floor": params.a_floor, "a_se_mult": params.a_se_mult, "gold_weight": params.gold_weight,
                   "m_b": params.m_b, "s_b": params.s_b, "const_baseline": params.const_baseline,
                   "omniscience_mode": params.omniscience_mode, "omni_acc_weight": params.omni_acc_weight,
                   "wildguard_mode": params.wildguard_mode, "resamples": params.resamples, "seed": seed,
                   "panel": ({"T": panel.t, "N": panel.n, "Pub": panel.pub} if panel else None)},
    }
    return V5Result(_f(pt["S"]), _f(pt["T"]), _f(pt["N"]), _f(pt["T_eq"]), _f(pt["N_eq"]), _f(pt["Pub"]), _f(pt["G"]), se_g,
                    _f(allowance), gap_flag, bool(excl), detail)


def score_v5(responses: Mapping[str, Any], items: Sequence[Mapping[str, Any]], base_responses: Mapping[str, Any],
             params: V5Params = DEFAULT, seed: int = 0, allow_partial: bool = False,
             exclude_types: Collection[str] = ()) -> tuple[float, float, float | None, float | None, float | None]:
    """(S_v5.1, T, N, G, SE_G). See ``score_v5_detail``."""
    return score_v5_detail(responses, items, base_responses, params, seed, allow_partial=allow_partial,
                           exclude_types=exclude_types).as_tuple()


def pool_se(ses: Sequence[float]) -> float:
    """SE of the mean of independent per-draw estimates: sqrt(sum se^2) / k (two equal draws: SE / sqrt(2))."""
    return math.sqrt(sum(s * s for s in ses)) / len(ses) if ses else float("inf")


# --------------------------------------------------------------------------------------------------------- migration

def scoring_regime(started_ts: float) -> str:
    """"v4" or "v5" for an evaluation (first + confirmation draw) STARTED at ``started_ts``: before T it finishes under v4,
    after T it is v5. Gate off: always v4. A later decision of an evaluation started before T stays v4."""
    if S.EVAL_V5_START_UTC is None:
        return "v4"
    return "v5" if float(started_ts) >= S.EVAL_V5_START_UTC.timestamp() else "v4"


def beats_seated(champion: Mapping[str, float], base: Mapping[str, float], challenger: Mapping[str, float],
                 margin: float) -> tuple[bool, float, float]:
    """A v5 challenger takes a seat held since before T only if its S_v5 beats max(champion S_v5, base S_v5) + margin on the
    SAME draws. Arguments: draw id -> S_v5 of each. Returns (beats, threshold, challenger mean)."""
    ids = set(challenger)
    if not ids or set(champion) != ids or set(base) != ids:
        raise ValueError("champion, base and challenger must be scored on the same draws")
    mean = lambda d: sum(d[i] for i in ids) / len(ids)   # noqa: E731
    threshold = max(mean(champion), mean(base)) + margin
    c = mean(challenger)
    return c >= threshold, threshold, c
