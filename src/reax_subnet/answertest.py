"""``reax-answertest``: the answer test for EU operators (replaces the strict bit-identity bar of ``reax-selfcheck``).

Runs the public golden set of ``reax-selfcheck`` (synthetic items with reference answers, no customer data) through a
runtime sidecar at a ladder of concurrencies under sustained load: ``c`` parallel streams plus one background
validator-like stream (rounds of ``items_per_round`` decisions, scored with the validator's own ``score_round``).
Per concurrency it measures answer quality against the reference (top-1 agreement, max / p99 probability difference,
logit distance, strike-line items, simulated validator rounds) and speed, and applies the admission criteria of
docs/ANSWER-TEST.md. The admitted concurrency is the highest ``c`` such that it and every lower ``c`` pass.

Uses the miner's settings like ``reax-selfcheck`` (``--url``, ``--token-file``). Exit code 0 = at least one concurrency
admitted, 1 = none admitted, 2 = cannot run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from .s1backends import SidecarBackend
from .s1contract import parse_decision
from .s1scoring import PoolParams, logit_distance, score_round
from .selfcheck import SIMULATED_ROUNDS, load_reference, simulate_rounds, _at, _view_switches_at

SCHEMA = "reax-answertest/1"
TOOL_VERSION = "1"
LADDER = (1, 2, 4, 8, 12, 16, 24, 32)


class Thresholds:
    """Admission criteria. Defaults = the bar of our own production nodes (node gate on 500 golden items) plus the
    validator strike lines; see docs/ANSWER-TEST.md section 4 for the reasoning."""

    def __init__(self, max_dp: float = 0.02, min_top1: float = 1.0, tie_margin: float = 0.02,
                 max_mean_distance: float = 0.10, contract_round_pass: float = 0.90,
                 contract_dps: float = 25.0, max_failed: int = 0) -> None:
        self.max_dp = max_dp  # largest |p_i - q_i| of any option probability
        self.min_top1 = min_top1  # top-1 agreement on items whose reference top-2 margin exceeds tie_margin
        self.tie_margin = tie_margin  # below this margin a flip is inside the dp budget: not counted
        self.max_mean_distance = max_mean_distance  # mean logit distance (engine gate; pool gate is tau_mean 0.20)
        self.contract_round_pass = contract_round_pass  # contract 4.2: >= 90 % rounds (informational)
        self.contract_dps = contract_dps  # contract: >= 25 decisions/s (informational)
        self.max_failed = max_failed

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def max_dp(p: Sequence[float], q: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(p, q))


def top1_margin(q: Sequence[float]) -> float:
    s = sorted(q, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def argmax(p: Sequence[float]) -> int:
    return max(range(len(p)), key=lambda i: p[i])


def _keyed(decision: Any, answer: Any) -> list[float]:
    return [float(answer["probabilities"][k]) for k in decision.question.keys]


def judge(m: dict[str, Any], th: Thresholds) -> dict[str, bool]:
    """Pass/fail per admission criterion for one concurrency's metrics ``m``."""
    answered = m["answered"] > 0
    return {
        "complete": m["failed"] <= th.max_failed and answered,
        "top1": answered and m["top1_agreement"] >= th.min_top1,
        "max_dp": answered and m["max_dp"] <= th.max_dp,
        "mean_distance": answered and m["distance_mean"] <= th.max_mean_distance,
        "no_strikes": answered and m["items_over_tau_item"] == 0 and m["round_strikes"] == 0.0
                      and m["validator_stream"]["strikes"] == 0,
        "full_pass": m["items_covered"] >= m["items"],
    }


async def run_level(backend: Any, decisions: Sequence[Any], reference_items: Sequence[dict[str, Any]],
                    params: PoolParams, c: int, duration_s: float, th: Thresholds, seed: int = 1) -> dict[str, Any]:
    """One concurrency level: ``c`` parallel streams cycling over the golden set until ``duration_s`` has passed and
    every item was answered once, plus one background stream of validator-like rounds."""
    n = len(decisions)
    rng = random.Random(seed * 1000 + c)
    dist: list[float] = []
    dps: list[float] = []
    lat: list[float] = []
    failures: list[str] = []
    flips = ties = non_tie = 0
    covered: set[int] = set()
    deadline = time.monotonic() + duration_s
    stop = asyncio.Event()
    counter = {"next": 0}
    order = list(range(n))
    rng.shuffle(order)

    def next_index() -> Optional[int]:
        i = counter["next"]
        if i >= n and time.monotonic() >= deadline:
            return None
        counter["next"] = i + 1
        return order[i % n]

    async def stream() -> None:
        nonlocal flips, ties, non_tie
        while (i := next_index()) is not None:
            d, ref = decisions[i], reference_items[i]
            t0 = time.monotonic()
            (ans,) = await backend.decide_each([d], retries=0)
            lat.append(time.monotonic() - t0)
            if isinstance(ans, BaseException):
                failures.append(f"{ref['id']}: {type(ans).__name__}: {ans}"[:200])
                continue
            covered.add(i)
            p = _keyed(d, ans)
            dist.append(logit_distance(p, ref["p"], params.temperature))
            dps.append(max_dp(p, ref["p"]))
            if top1_margin(ref["p"]) < th.tie_margin:
                ties += 1
            else:
                non_tie += 1
                flips += argmax(p) != argmax(ref["p"])

    vrounds: list[Any] = []
    vfail = {"failed": 0}

    async def validator_stream() -> None:
        k, per = params.verified_per_round, params.items_per_round
        vrng = random.Random(seed * 7 + c)
        while not stop.is_set():
            idx = vrng.sample(range(n), per)
            t0 = time.monotonic()
            answers = await backend.decide_each([decisions[i] for i in idx], retries=0)
            took = time.monotonic() - t0
            if any(isinstance(a, BaseException) for a in answers):
                vfail["failed"] += 1
                vrounds.append(score_round(None, {"x": [1.0]}, took, params))
                continue
            got = {reference_items[i]["id"]: _keyed(decisions[i], a) for i, a in zip(idx, answers)}
            verified = vrng.sample(idx, k)
            refs = {reference_items[i]["id"]: reference_items[i]["p"] for i in verified}
            vrounds.append(score_round(got, refs, took, params))

    started = time.monotonic()
    vtask = asyncio.create_task(validator_stream())
    try:
        await asyncio.gather(*(stream() for _ in range(c)))
    finally:
        stop.set()
        await vtask
    elapsed = time.monotonic() - started
    answered = len(dist)
    m: dict[str, Any] = {"c": c, "items": n, "items_covered": len(covered), "answered": answered,
                         "failed": len(failures), "failures": failures[:5], "seconds": round(elapsed, 1),
                         "decisions_per_s": round(answered / elapsed, 1) if elapsed > 0 else None}
    if answered:
        sd, sp, sl = sorted(dist), sorted(dps), sorted(lat)
        scored, struck = simulate_rounds(dist, params, rounds=SIMULATED_ROUNDS) if answered >= params.verified_per_round \
            else (0.0, 1.0)
        m.update({
            "top1_agreement": round((non_tie - flips) / non_tie, 5) if non_tie else 1.0,
            "top1_flips": flips, "top1_items": non_tie, "near_tie_items": ties,
            "max_dp": round(sp[-1], 5), "p99_dp": round(_pct(sp, 0.99), 5), "median_dp": round(sp[len(sp) // 2], 5),
            "distance_mean": round(math.fsum(dist) / answered, 4), "distance_median": round(sd[len(sd) // 2], 4),
            "distance_p99": round(_pct(sd, 0.99), 4), "distance_max": round(sd[-1], 4),
            "items_over_tau_item": sum(1 for d in sd if d > params.tau_item),
            "round_scores": round(scored, 5), "round_strikes": round(struck, 5),
            "latency_p50_s": round(_pct(sl, 0.50), 3), "latency_p95_s": round(_pct(sl, 0.95), 3),
        })
    vs = len(vrounds)
    m["validator_stream"] = {"rounds": vs, "failed": vfail["failed"],
                             "scored": sum(1 for r in vrounds if r.score > 0),
                             "strikes": sum(1 for r in vrounds if r.strike)}
    crit = judge(m, th) if answered else {k: False for k in
                                         ("complete", "top1", "max_dp", "mean_distance", "no_strikes", "full_pass")}
    m["criteria"] = crit
    m["pass"] = all(crit.values())
    m["contract"] = {  # informational: pay needs >= 90 % positive rounds and >= 25 decisions/s (EU-TIER.md)
        "rounds_ok": bool(answered) and m.get("round_scores", 0.0) >= th.contract_round_pass,
        "throughput_ok": bool(m["decisions_per_s"]) and m["decisions_per_s"] >= th.contract_dps}
    return m


def admitted_concurrency(levels: Sequence[dict[str, Any]]) -> tuple[int, list[str]]:
    """Highest ``c`` such that it and every lower tested ``c`` pass (0 = none), and warnings (a pass above a fail)."""
    adm, broken, warnings = 0, False, []
    for lv in sorted(levels, key=lambda x: x["c"]):
        if lv["pass"] and not broken:
            adm = lv["c"]
        elif lv["pass"]:
            warnings.append(f"c={lv['c']} passes above a failing lower level; not admitted")
        else:
            broken = True
    return adm, warnings


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip().splitlines()
        return out[0].strip() if out else ""
    except (OSError, subprocess.SubprocessError):
        return ""


async def _wait_ready(backend: Any, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if await backend.ready():
            return True
        await asyncio.sleep(2.0)
    return False


async def run(model: str, url: str, token: str, *, ladder: Sequence[int] = LADDER, duration_s: float = 60.0,
              cold_starts: int = 1, restart_cmd: str = "", ready_timeout_s: float = 1800.0, limit: Optional[int] = None,
              th: Optional[Thresholds] = None, reference: Optional[dict[str, Any]] = None, engine: str = "",
              image_digest: str = "", gpu_name: str = "", driver: str = "", stop_after_fails: int = 2,
              backend: Any = None, progress: Any = None) -> dict[str, Any]:
    """Run the ladder ``cold_starts`` times (restarting between runs with ``restart_cmd``) and report the worst case per
    concurrency. ``backend`` is for tests (any object with ``decide_each`` / ``ready`` / ``close``)."""
    from .s1validator import pool_params

    th = th or Thresholds()
    reference = reference or load_reference(model)
    items = reference["items"][:limit] if limit else reference["items"]
    params = pool_params(model)
    decisions = [parse_decision({"id": it["id"], "state": it["state"], "questions": {"q": it["question"]}}, model)
                 for it in items]
    ladder = sorted(set(int(x) for x in ladder if int(x) >= 1))
    if cold_starts > 1 and not restart_cmd:
        raise ValueError("cold_starts > 1 needs restart_cmd (or run the tool once per cold start and use 'combine')")
    own = backend is None
    if own:
        backend = SidecarBackend(model, url, token, max_inflight=max(ladder) + 1, timeout_s=120.0)
    runs: list[list[dict[str, Any]]] = []
    try:
        if not await backend.ready():
            raise ConnectionError(f"runtime sidecar at {url} is not healthy")
        for cs in range(cold_starts):
            if cs > 0:
                subprocess.run(shlex.split(restart_cmd), check=True, timeout=600)
                await asyncio.sleep(5.0)
                if not await _wait_ready(backend, ready_timeout_s):
                    raise ConnectionError("runtime did not become healthy after the restart command")
            levels: list[dict[str, Any]] = []
            fails = 0
            for c in ladder:
                lv = await run_level(backend, decisions, items, params, c, duration_s, th, seed=cs + 1)
                levels.append(lv)
                if progress:
                    progress(cs, lv)
                fails = 0 if lv["pass"] else fails + 1
                if stop_after_fails and fails >= stop_after_fails:
                    break
            runs.append(levels)
    finally:
        if own:
            await backend.close()
    return build_report(model, reference["manifest_id"], runs, th, engine=engine, image_digest=image_digest,
                        gpu_name=gpu_name, driver=driver, items=len(items), duration_s=duration_s,
                        weights_revision=reference.get("weights_revision", ""))


def build_report(model: str, manifest_id: str, runs: Sequence[Sequence[dict[str, Any]]], th: Thresholds, *,
                 engine: str = "", image_digest: str = "", gpu_name: str = "", driver: str = "", items: int = 0,
                 duration_s: float = 0.0, weights_revision: str = "") -> dict[str, Any]:
    """Merge the per-cold-start ladders: a concurrency passes only if it passed in EVERY run (worst case); its
    reported numbers are the worst per metric."""
    cs = [sorted(r, key=lambda x: x["c"]) for r in runs]
    cvals = sorted({lv["c"] for r in cs for lv in r})
    merged: list[dict[str, Any]] = []
    for c in cvals:
        have = [lv for r in cs for lv in r if lv["c"] == c]
        if len(have) < len(cs):  # a run stopped before this c: it did not pass there
            continue
        worst = dict(have[0])
        for key, hi in (("max_dp", True), ("p99_dp", True), ("distance_max", True), ("distance_mean", True),
                        ("distance_p99", True), ("latency_p95_s", True), ("round_strikes", True),
                        ("items_over_tau_item", True), ("top1_flips", True), ("failed", True),
                        ("top1_agreement", False), ("round_scores", False), ("decisions_per_s", False)):
            vals = [lv[key] for lv in have if key in lv and lv[key] is not None]
            if vals:
                worst[key] = max(vals) if hi else min(vals)
        worst["criteria"] = {k: all(lv["criteria"][k] for lv in have) for k in have[0]["criteria"]}
        worst["pass"] = all(lv["pass"] for lv in have)
        worst["contract"] = {k: all(lv["contract"][k] for lv in have) for k in have[0]["contract"]}
        merged.append(worst)
    adm, warnings = admitted_concurrency(merged)
    return {
        "schema": SCHEMA, "tool_version": TOOL_VERSION, "model": model, "manifest_id": manifest_id,
        "weights_revision": weights_revision, "engine": engine, "engine_image_digest": image_digest,
        "gpu_name": gpu_name, "driver": driver, "golden_items": items, "level_seconds": duration_s,
        "cold_starts": len(cs), "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thresholds": th.as_dict(), "levels": merged, "admitted_c": adm, "pass": adm > 0, "warnings": warnings,
        "runs_c_tested": [[lv["c"] for lv in r] for r in cs],
    }


def combine(reports: Sequence[dict[str, Any]], th: Optional[Thresholds] = None) -> dict[str, Any]:
    """Combine separate runs (one cold start each) of the same model/manifest into one worst-case report."""
    first = reports[0]
    for r in reports:
        if r.get("schema") != SCHEMA or (r["model"], r["manifest_id"]) != (first["model"], first["manifest_id"]):
            raise ValueError("reports must be reax-answertest/1 reports of the same model and manifest")
    th = th or Thresholds(**first["thresholds"])
    out = build_report(first["model"], first["manifest_id"], [r["levels"] for r in reports], th,
                       engine=first.get("engine", ""), image_digest=first.get("engine_image_digest", ""),
                       gpu_name=first.get("gpu_name", ""), driver=first.get("driver", ""),
                       items=first.get("golden_items", 0), duration_s=first.get("level_seconds", 0.0),
                       weights_revision=first.get("weights_revision", ""))
    out["warnings"] += [f"engine/GPU differ between runs: {r.get('engine_image_digest')}/{r.get('gpu_name')}"
                        for r in reports[1:] if (r.get("engine_image_digest"), r.get("gpu_name"))
                        != (first.get("engine_image_digest"), first.get("gpu_name"))]
    return out


def print_report(r: dict[str, Any]) -> None:
    print(f"answer test: {r['model']} manifest {r['manifest_id']}, {r['golden_items']} golden items, "
          f"{r['cold_starts']} cold start(s), {r['timestamp']}")
    if r.get("gpu_name") or r.get("engine_image_digest"):
        print(f"  GPU {r.get('gpu_name') or '?'}, engine {r.get('engine') or '?'} "
              f"{r.get('engine_image_digest') or ''}, driver {r.get('driver') or '?'}")
    print("  c   verdict  top1     max_dp   p99_dp   dist(mean/max)  over  sim.rounds  dec/s   p50/p95 s")
    for lv in r["levels"]:
        if not lv["answered"]:
            print(f"  {lv['c']:<3} FAIL     no answers")
            continue
        bad = [k for k, v in lv["criteria"].items() if not v]
        print(f"  {lv['c']:<3} {'PASS' if lv['pass'] else 'FAIL':<8} {lv['top1_agreement']:<8.4f} {lv['max_dp']:<8.4f} "
              f"{lv['p99_dp']:<8.4f} {lv['distance_mean']:.3f}/{lv['distance_max']:<7.3f} {lv['items_over_tau_item']:<5} "
              f"{lv['round_scores']:<11.2%} {lv['decisions_per_s']:<7} {lv['latency_p50_s']}/{lv['latency_p95_s']}"
              + (f"   failed: {', '.join(bad)}" if bad else ""))
    for w in r["warnings"]:
        print(f"  warning: {w}")
    if r["admitted_c"]:
        c = r["admitted_c"]
        lv = next(x for x in r["levels"] if x["c"] == c)
        notes = [] if lv["contract"]["rounds_ok"] else ["fewer than 90 % simulated rounds score (pay risk)"]
        notes += [] if lv["contract"]["throughput_ok"] else ["below 25 decisions/s (pay needs 25)"]
        print(f"ADMITTED concurrency: {c} in flight per GPU" + (f" -- note: {'; '.join(notes)}" if notes else ""))
    else:
        print("NOT ADMITTED: no concurrency met the answer criteria; see docs/ANSWER-TEST.md")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="reax-answertest", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    cmb = sub.add_parser("combine", help="combine reports of separate cold starts into one worst-case report")
    cmb.add_argument("reports", nargs="+")
    cmb.add_argument("--json", action="store_true")
    ap.add_argument("--model", default=os.environ.get("REAX_MODEL", "s1-fast"),
                    choices=("s1-fast", "s1-pro", "s1-lite", "s1-cpu"))
    ap.add_argument("--url", help="runtime sidecar URL (default: REAX_<MODEL>_RUNTIME_URL)")
    ap.add_argument("--token-file", default=os.environ.get("REAX_RUNTIME_TOKEN_FILE", ""))
    ap.add_argument("--ladder", default=",".join(map(str, LADDER)), help="concurrencies to test, comma-separated")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds of sustained load per concurrency (at least one full pass of the golden set is "
                         "always run); 15-20 gives a quick read, the binding run uses 120")
    ap.add_argument("--cold-starts", type=int, default=1, help="ladder runs, restarting the runtime between them")
    ap.add_argument("--restart-cmd", default="", help="command that restarts the runtime (needed for --cold-starts > 1)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N golden items (quick read)")
    ap.add_argument("--reference", help="reference JSON file (reax-selfcheck/1 format) instead of the shipped one")
    ap.add_argument("--stop-after-fails", type=int, default=2, help="stop the ladder after N failing levels in a row (0 = never)")
    ap.add_argument("--engine", default="", help="engine name for the report (reax-engine, vllm, sglang, ...)")
    ap.add_argument("--image-digest", default="", help="engine image digest for the report")
    ap.add_argument("--gpu-name", default="", help="GPU name for the report; 'auto' asks nvidia-smi on this machine")
    ap.add_argument("--driver", default="", help="NVIDIA driver version for the report")
    ap.add_argument("--max-dp", type=float, default=Thresholds().max_dp)
    ap.add_argument("--min-top1", type=float, default=Thresholds().min_top1)
    ap.add_argument("--out", help="also write the JSON report to this file")
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    ap.add_argument("--at", type=_at, default=None, help="check as of this UTC time (pool manifest switches, as reax-selfcheck)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "combine":
            docs = []
            for path in args.reports:
                with open(path, encoding="utf-8") as fh:
                    docs.append(json.load(fh))
            report = combine(docs)
        else:
            if args.at is not None:
                _view_switches_at(args.at)
            var = f"REAX_{args.model.upper().replace('-', '_')}_RUNTIME_URL"
            url = args.url or os.environ.get(var, "")
            if not url:
                raise ValueError(f"no runtime URL: pass --url or set {var}")
            if not args.token_file:
                raise ValueError("no runtime token: pass --token-file or set REAX_RUNTIME_TOKEN_FILE")
            with open(os.path.expanduser(args.token_file), encoding="ascii") as fh:
                token = fh.read().strip()
            if not token:
                raise ValueError("runtime token file is empty")
            reference = None
            if args.reference:
                with open(args.reference, encoding="utf-8") as fh:
                    reference = json.load(fh)
                if reference.get("schema") != "reax-selfcheck/1" or reference.get("model") != args.model:
                    raise ValueError("reference file is not a reax-selfcheck/1 file for this model")
            ladder = [int(x) for x in args.ladder.split(",") if x.strip()]
            report = asyncio.run(run(
                args.model, url, token, ladder=ladder, duration_s=args.duration, cold_starts=max(1, args.cold_starts),
                restart_cmd=args.restart_cmd, limit=args.limit or None, reference=reference, engine=args.engine,
                image_digest=args.image_digest, gpu_name=_gpu_name() if args.gpu_name == "auto" else args.gpu_name,
                driver=args.driver, stop_after_fails=args.stop_after_fails,
                th=Thresholds(max_dp=args.max_dp, min_top1=args.min_top1),
                progress=None if args.json else (lambda cs, lv: print(
                    f"  cold start {cs + 1}, c={lv['c']}: {'PASS' if lv['pass'] else 'FAIL'}", file=sys.stderr))))
    except (OSError, ValueError, ConnectionError, subprocess.SubprocessError) as exc:
        if args.json:
            print(json.dumps({"pass": False, "error": str(exc)}))
        else:
            print(f"reax-answertest: cannot run: {exc}", file=sys.stderr)
        return 2
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, sort_keys=True, indent=1)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_report(report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
