"""Eval v5.1 evaluator wiring: the private T/N pool files, the draw the evaluator takes, the per-question rows a backend serves,
and the per-draw numbers (``S_v5.1``, T, N, the gap check) a decision reads (docs/EVAL-V5.md, PROPOSAL.md 1.1-1.6).

The scoring and the draw themselves are ``spec/v5.py`` (``draw_v5``, ``score_v5_detail``: the eval-v5.1 scorer delta); this module
only connects them to ``evaluator.SpecEvaluator``. It is reached only when ``sampling.v5_active(now)`` (``EVAL_V5_START_UTC`` set and
reached); with the gate ``None`` nothing here runs and the evaluator is rc11 bit for bit.

Pool: ``<v5_pool_dir>/v51.T.jsonl`` and ``v51.N.jsonl`` (mode 0600, owner only: anything group/other readable is refused), typed rows
of ``v5`` (tier ``mirror``, ``part`` T / N, ``area``, ``task_type``, ``gold``, ``questions``). The public items of the gap check
(weight 0) are the ``same_family`` / ``proc_sealed`` rows of the ordinary sealed generalist pool (v4 row format).
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Collection, Mapping, Sequence

import numpy as np

from . import sampling as S
from . import v5

log = logging.getLogger("reax.spec")

PUB_ITEMS = 90           # public items per draw (weight 0): only the gap check reads them
POOL_FILES = {"T": "v51.T.jsonl", "N": "v51.N.jsonl"}
REGIME = "v5"            # ``scoring_regime`` of the history rows of decisions taken under this module
TIER_KEYS = frozenset({"T", "N"})   # ``Submission.by_tier`` of an entry whose latest score is S_v5.1 (v4 keys are tier names); no new state field


def scored_under_v5(by_tier: Mapping[str, Any]) -> bool:
    """True if ``by_tier`` (``Submission.by_tier``) was written by a v5.1 decision: the keys are exactly T and N."""
    return set(by_tier) == TIER_KEYS


def gap_flagged(history: Sequence[Mapping[str, Any]]) -> bool:
    """The entry's newest gap check flagged it (G > A): its awards and keep-alive promotion wait for the overlap check."""
    for h in reversed(history):
        if "gap_flag" in h:
            return bool(h["gap_flag"])
    return False


CONFIG_FILE = "v51.config.json"
_logged_configs: set[tuple] = set()


def load_config(pool_dir: str) -> tuple[str, dict[str, float]] | None:
    """The optional pool-version config ``<pool_dir>/v51.config.json`` (mode 0600 like the pools):
    ``{"pool_version": str, "hf_areas": {<the five areas>: weight >= 0}}``, weights summing to more than 0. Returns
    (pool_version, hf_areas), or None when there is no pool directory or no file (the default ``v5.HF_AREAS`` then). A malformed or
    readable-by-others file raises ``SampleError`` (fail closed: no draw). The pool version and weights are logged once per change."""
    import json
    if not pool_dir:
        return None
    path = os.path.join(pool_dir, CONFIG_FILE)
    if not os.path.lexists(path):
        return None
    try:
        check_file_mode(path)
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        if not isinstance(obj, dict) or set(obj) != {"pool_version", "hf_areas"}:
            raise ValueError('keys must be exactly "pool_version" and "hf_areas"')
        version, areas = obj["pool_version"], obj["hf_areas"]
        if not isinstance(version, str) or not version:
            raise ValueError("pool_version must be a non-empty string")
        if not isinstance(areas, dict) or set(areas) != set(v5.HF_AREAS):
            raise ValueError(f"hf_areas must contain exactly {sorted(v5.HF_AREAS)}")
        weights: dict[str, float] = {}
        for k, x in areas.items():
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f"hf_areas[{k!r}] must be a finite number >= 0")
            weights[k] = float(x)
        if math.fsum(weights.values()) <= 0:
            raise ValueError("hf_areas weights must sum to more than zero")
    except S.SampleError as exc:
        log.error("v5.1 pool config %s refused: %s", CONFIG_FILE, exc)
        raise
    except (OSError, ValueError) as exc:     # json.JSONDecodeError is a ValueError
        log.error("v5.1 pool config %s refused: %s", CONFIG_FILE, exc)
        raise S.SampleError(f"v5.1 pool config {CONFIG_FILE} is invalid: {exc}") from exc
    key = (path, version, tuple(sorted(weights.items())))
    if key not in _logged_configs:
        _logged_configs.add(key)
        log.info("v5.1 pool config: pool_version=%s hf_areas=%s", version, weights)
    return version, weights


def active_panel(config: tuple[str, dict[str, float]] | None, panel: "v5.PanelStats | None") -> "v5.PanelStats":
    """The panel stats the evaluator scales with: those of the pool version named in ``config`` (``load_config``), from
    ``REAX_SPEC_V5_PANEL`` (``panel``) if set, else ``v5.PANEL_STATS``. Fail closed: no config (pool version unknown), a
    ``panel`` for another pool version, or no stats for the version raise ``SampleError`` (no v5.1 draw, nothing decided)."""
    version = config[0] if config is not None else None
    if version is None:
        why = f"pool version unknown ({CONFIG_FILE} missing in the v5.1 pool directory)"
    elif panel is not None:
        if panel.version == version:
            return panel
        why = f"REAX_SPEC_V5_PANEL is for pool version {panel.version!r}, the pool is {version!r}"
    elif version in v5.PANEL_STATS:
        return v5.PANEL_STATS[version]
    else:
        why = f"no panel stats for pool version {version!r} (v5.PANEL_STATS, REAX_SPEC_V5_PANEL)"
    log.error("eval v5.1 FAIL CLOSED: %s; no v5.1 draw, nothing is decided or crowned", why)
    raise S.SampleError(f"v5.1 panel stats: {why}")


_READ = object()


def params(w_n: float = 0.4, panel: "v5.PanelStats | None" = None, resamples: int | None = None,
           pool_dir: str = "", config: Any = _READ) -> v5.V5Params:
    """The scorer parameters the evaluator uses: 540 T + 360 N + ``PUB_ITEMS`` public items per draw. With ``w_n`` = 0 (the T-only
    fallback) no N items are drawn or served. With ``pool_dir`` and a ``v51.config.json`` in it, its area weights replace the
    default ``v5.HF_AREAS`` (draw and scoring both build their params here); an invalid config raises ``SampleError``.
    ``config``: an already loaded ``load_config`` result (the evaluator's per-cycle cache) instead of reading ``pool_dir``."""
    kw: dict[str, Any] = {} if resamples is None else {"resamples": resamples}
    cfg = load_config(pool_dir) if config is _READ else config
    if cfg is not None:
        kw["hf_areas"] = cfg[1]
    return v5.V5Params(w_n=w_n, panel=panel, pub_items=PUB_ITEMS, n_items=360 if w_n > 0 else 0, **kw)


def draw_size(p: v5.V5Params) -> int:
    return p.t_items + p.n_items + p.pub_items


# ------------------------------------------------------------------------------------------------------------- pool

def check_file_mode(path: str) -> None:
    """Private pool files are owner-only (0600 or stricter); a readable-by-others pool is treated as leaked."""
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise S.SampleError(f"v5.1 pool file {os.path.basename(path)} has mode {mode:03o}; it must be 0600 (owner only)")


def load_pool(pool_dir: str, p: v5.V5Params | None = None) -> list[dict[str, Any]]:
    """The typed T and N rows. A missing T file, or a missing N file while ``w_n`` > 0, is an error (fail closed); with ``w_n`` = 0
    (the T-only fallback) the N file may be absent. Every row is validated (``v5.validate_mirror_row``) and must carry the part of
    its file; ids are unique."""
    import json
    p = p or v5.DEFAULT
    if not pool_dir:
        raise S.SampleError("no v5.1 pool directory configured (REAX_SPEC_V5_POOL_DIR)")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for part, name in POOL_FILES.items():
        path = os.path.join(pool_dir, name)
        if not os.path.exists(path):
            if part == "N" and p.w_n == 0:
                continue
            raise S.SampleError(f"v5.1 pool file {name} not found under {pool_dir}")
        check_file_mode(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("tier") != S.MIRROR_TIER:
                    raise S.SampleError(f"row {row.get('id')!r} in {name}: tier must be {S.MIRROR_TIER!r}")
                v5.validate_mirror_row(row)
                if row["part"] != part:
                    raise S.SampleError(f"row {row['id']!r}: part {row['part']!r} in the {part} file")
                if row["id"] in ids:
                    raise S.SampleError(f"duplicate v5.1 item id {row['id']!r}")
                ids.add(row["id"])
                rows.append(row)
    rows.sort(key=lambda r: r["id"])    # critic F9: the draw must not depend on the order of the lines in the files
    return rows


def public_rows(generalist_pool: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Public items of the gap check: the ordinary sealed generalist rows of the public tiers (v4 row format)."""
    return [dict(r) for r in generalist_pool if r.get("tier") in v5.PUB_TIERS]


def draw(pool: Sequence[dict[str, Any]], public: Sequence[dict[str, Any]], *, seed: int, use_counts: Mapping[str, int],
         exclude: Collection[str] = frozenset(), p: v5.V5Params | None = None) -> tuple[list[dict], bool]:
    """One draw (``v5.draw_v5``): T 540 and N 360 rows by task type (fresh and least used first, retired after ``mirror_max_uses``
    uses, a short type raises ``SampleError``: no borrowing), plus ``PUB_ITEMS`` public rows. Every typed question is capped at 26 options (``cap_item``). Returns (rows sorted by id, fresh)."""
    p = p or params()
    rows, fresh = v5.draw_v5([*pool, *public], n=draw_size(p), seed=seed, use_counts=use_counts, exclude=exclude, params=p)
    return [cap_item(r, seed) if "questions" in r else r for r in rows], fresh


# ------------------------------------------------------------------------------------------ the 26-option cap

OPTION_CAP = 26     # the single-letter readout (A..Z) every backend supports; a question with more options is reduced in the draw
# the option count a task text may name ("... one of the 40 candidates"): replaced by the capped count (as scripts/merge_pools.py)
COUNT_RE = (r"\b{n}\b(?=\s+(?:candidate|candidates|answers|answer|tools|tool|options|option|operations|documents|choices|labels"
            r"|moves|chords|agencies|titles|divisions))")


def cap_item(item: dict[str, Any], seed: int) -> dict[str, Any]:
    """The item as served in the draw of ``seed``: every question with more than ``OPTION_CAP`` options (an IDK option counts
    against the cap) keeps the gold option plus distractors sampled from ``(seed, item id, question index)``, the keys re-lettered
    A..Z in their original order, and a task text that names the original option count names the capped count. The subset
    depends only on the draw, so every model of the draw sees the same question; a question within the cap is unchanged; the input
    is not modified. The capped count is what chance-correction and the readout see (they only ever see the returned item)."""
    idk = bool(item.get("idk"))
    cap = OPTION_CAP - 1 if idk else OPTION_CAP        # the IDK option is added when the item is served
    if all(len(q["criteria"]) <= cap for q in item["questions"]):
        return item
    out = dict(item)
    out["questions"] = []
    for j, q in enumerate(item["questions"]):
        keys = [str(k) for k in q["criteria"]]
        if len(keys) <= cap:
            out["questions"].append(q)
            continue
        gold = str(q["expected"])
        rng = random.Random(f"v5cap|{seed}|{item['id']}|{j}")
        keep = set(rng.sample([k for k in keys if k != gold], cap - 1)) | {gold}
        kept = [k for k in keys if k in keep]
        relabel = dict(zip(kept, v5.LETTERS))
        crit = {str(k): v for k, v in q["criteria"].items()}
        out["questions"].append(dict(q, instructions=re.sub(COUNT_RE.format(n=len(keys)), str(OPTION_CAP), str(q.get("instructions", ""))),
                                     criteria={relabel[k]: crit[k] for k in kept}, expected=relabel[gold]))
    out["capped_from"] = max(len(q["criteria"]) for q in item["questions"])
    return out


# ----------------------------------------------------------------------------------------------- serving: flat rows

def flatten(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The rows a backend serves: every question of a typed item as its own ``choice`` row (the causal path; shared state
    repeated, IDK an extra last option), v4-format public rows unchanged. Row ids are ``<item id>#<j>``; ``labels`` / ``expected``
    make each row scorable by the v4 per-item scorer (competence gate, near-copy and twin checks) as well."""
    out: list[dict[str, Any]] = []
    for it in items:
        if "questions" not in it:
            out.append(dict(it))
            continue
        idk = bool(it.get("idk"))
        for j, q in enumerate(it["questions"]):
            crit = {str(k): v for k, v in q["criteria"].items()}
            if idk:
                crit.setdefault(v5.IDK_KEY, v5.IDK_TEXT)
            out.append({"id": f"{it['id']}#{j}", "state": it.get("state"), "tier": it["part"], "category": "", "item": it["id"],
                        "question": {"type": "choice", "instructions": q.get("instructions", ""), "criteria": crit},
                        "labels": list(crit), "expected": str(q["expected"]), "idk": idk})
    return out


def responses(items: Sequence[Mapping[str, Any]], flat: Sequence[Mapping[str, Any]], probs: Sequence[Any]) -> dict[str, Any]:
    """Item id -> response for ``v5``: a list with one answer per question for typed items, the answer dict for a public item.
    A question the backend failed on (None) stays None, i.e. wrong with zero credit for every model alike."""
    by_row = {r["id"]: p for r, p in zip(flat, probs)}
    out: dict[str, Any] = {}
    for it in items:
        if "questions" in it:
            out[it["id"]] = [by_row.get(f"{it['id']}#{j}") for j in range(len(it["questions"]))]
        else:
            out[it["id"]] = by_row.get(it["id"])
    return out


# ------------------------------------------------------------------------------------------------------------ scoring

@dataclass(frozen=True)
class PartScores:
    """Headline numbers of one model on one draw (``t`` / ``n`` are the scaled T_eq / N_eq, 0.0 when N does not count)."""
    s: float
    t: float
    n: float
    pub: float | None


@dataclass(frozen=True)
class Gap:
    g: float | None
    se_g: float | None
    allowance: float | None
    flag: bool | None       # G > A: awards and keep-alive promotion wait for the overlap check; the score is unchanged. None = not measurable


paired_s_se = v5.paired_s_se    # public since the scorer branch gained it (v5.paired_s_se); kept here for the evaluator's callers


def base_key(base_model: str) -> str:
    return f"base:{base_model}"


@dataclass
class Ref:
    """What a challenger must beat on a draw: the best of the seated co-champions and its declared base."""
    total: float          # S_v5.1 of the best of them on this draw
    se: float             # paired SE of (challenger - that reference)
    key: str              # stable id of the reference SET (pools draws against the same set), not of the argmax
    best: str             # who was best on this draw
    totals: dict[str, float]


@dataclass
class DrawContext:
    """Per-draw v5.1 numbers for ``SpecEvaluator._decide``: S_v5.1 of every served model, each challenger's reference and gap."""
    draw_id: str
    items: list[dict]
    params: v5.V5Params
    seed: int
    resp: dict[str, dict[str, Any]]
    parts: dict[str, PartScores]
    ref: dict[str, Ref] = field(default_factory=dict)
    gap: dict[str, Gap] = field(default_factory=dict)
    detail: dict[str, v5.V5Result] = field(default_factory=dict)
    _pair: dict[tuple[str, str], float] = field(default_factory=dict)
    bad: set = field(default_factory=set)    # models whose S_v5.1 is not a finite number (not servable); no reference is built for them

    def totals(self) -> dict[str, float]:
        return {k: p.s for k, p in self.parts.items()}

    def pair_se(self, a: str, b: str) -> float:
        """Paired SE of S_a - S_b on this draw (cached)."""
        if (a, b) not in self._pair:
            self._pair[(a, b)] = paired_s_se(self.items, self.resp[a], self.resp[b], self.params, self.seed)
        return self._pair[(a, b)]


def _parts(res: v5.V5Result, base: bool) -> PartScores:
    d = res.detail
    if not base:
        return PartScores(res.s, res.t_eq, res.n_eq or 0.0, res.pub)
    return PartScores(d["S_v5.1_base"], d["T_eq_base"], d["N_eq_base"] or 0.0, d["Pub_base"])


def build_context(draw_id: str, items: Sequence[Mapping[str, Any]], resp: Mapping[str, Mapping[str, Any]],
                  challengers: Mapping[str, str], seated: Sequence[str], seed: int, p: v5.V5Params | None = None) -> DrawContext:
    """``resp``: model key -> responses (``responses``), covering the challengers, every seated co-champion and every base
    (``base_key``). ``challengers``: challenger key -> base key. ``seated``: keys of the seated co-champions (served on this draw).
    Each challenger is scored against its own base (``v5.score_v5_detail``: S, T_eq, N_eq, the gap check); a seated model or a base
    gets its absolute S_v5.1 on the same items. The reference of a challenger = the argmax of S_v5.1 over ``seated`` and its base;
    its SE is the paired bootstrap against that argmax."""
    p = p or params()
    items = list(items)
    ctx = DrawContext(draw_id, items, p, seed, {k: dict(r) for k, r in resp.items()}, {})
    cheap = replace(p, resamples=2)            # point estimates only: no bootstrap needed for a seated model or a base
    for ck, bk in challengers.items():
        res = v5.score_v5_detail(resp[ck], items, resp[bk], p, seed)
        ctx.detail[ck] = res
        ctx.parts[ck] = _parts(res, False)
        ctx.parts.setdefault(bk, _parts(res, True))
        ctx.gap[ck] = Gap(res.g, res.se_g, res.a, res.gap_flag)
    any_base = next(iter(challengers.values()), None)
    for k in resp:     # seated co-champions, parents: their absolute S_v5.1 on the same items
        if k not in ctx.parts and any_base is not None:
            ctx.parts[k] = _parts(v5.score_v5_detail(resp[k], items, resp[any_base], cheap, seed), False)
    fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)   # noqa: E731
    ctx.bad = {k for k, part in ctx.parts.items() if not fin(part.s)}
    for ck, bk in challengers.items():
        members = [*seated, bk]
        if ck in ctx.bad or any(m in ctx.bad for m in members):
            continue
        totals = {m: ctx.parts[m].s for m in members}
        best = max(members, key=lambda m: (totals[m], m))
        ctx.ref[ck] = Ref(ctx.parts[best].s, ctx.pair_se(ck, best), "v5|" + "+".join(sorted(seated)) + "|" + bk, best, totals)
    return ctx


def slot_context(draw_id: str, items: Sequence[Mapping[str, Any]], resp: Mapping[str, Mapping[str, Any]], seed: int,
                 p: v5.V5Params | None = None) -> DrawContext:
    """The per-round numbers of a crowning slot round on eval v5.1 (rc11.post4, slot rules section 4): the absolute S_v5.1 of
    every member on the round's items (``resp``: member key -> ``responses``) and, through ``DrawContext.pair_se``, the paired
    SE of any two members. No base and no reference: slot members are compared with the slot leader only, never with a
    champion or a base (they all beat the bar on their own confirmation). S_v5.1 of an entry depends only on its own answers
    (the base enters the gap check alone), so each member is scored with itself in the place of the base. A member whose
    S_v5.1 is not a finite number is in ``bad``."""
    p = p or params()
    items = list(items)
    ctx = DrawContext(draw_id, items, p, seed, {k: dict(r) for k, r in resp.items()}, {})
    cheap = replace(p, resamples=2)            # point estimates only; the pair SE runs its own bootstrap (``p.resamples``)
    for k, r in ctx.resp.items():
        ctx.parts[k] = _parts(v5.score_v5_detail(r, items, r, cheap, seed), False)
    fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)   # noqa: E731
    ctx.bad = {k for k, part in ctx.parts.items() if not fin(part.s)}
    return ctx


def slot_pair_se(ctxs: Sequence[DrawContext], a: str, b: str) -> float:
    """Paired SE of the pooled slot mean of ``a`` minus that of ``b`` over the rounds both took (``ctxs``, one per round): the
    rounds are independent draws and the pooled score is their mean, so ``v5.pool_se`` of the per-round paired SEs."""
    return v5.pool_se([c.pair_se(a, b) for c in ctxs])
