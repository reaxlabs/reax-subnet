"""Run the families against a served model, journal every answer (resumable), score and write the report."""
import concurrent.futures
import datetime
import json
import math
import os
import time
from pathlib import Path

from . import __version__, items as itemlib, scoring
from .common import atomic_write_json, atomic_write_text, read_json, read_jsonl_tolerant
from .readout import Client, answer_row


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def run_family(fam, pairs, out_dir, base_url, served_model, concurrency=48, timeout=120, api_key="", retry_errors=False, progress=None):
    """Answer every request of one family; append to <out>/<fam>/results.jsonl. Finished rows are not asked again;
    error rows are asked again only with retry_errors (the preview stops on errors; see README)."""
    style = itemlib.FAMILIES[fam]["style"]
    fdir = Path(out_dir) / fam
    fdir.mkdir(parents=True, exist_ok=True)
    journal = fdir / "results.jsonl"
    prior = read_jsonl_tolerant(journal)
    if prior and journal.read_bytes()[-1:] != b"\n":  # torn tail from a killed run: rewrite the readable rows
        atomic_write_text(journal, "".join(json.dumps(r, allow_nan=False) + "\n" for r in prior))
    done = {r["run_id"] for r in prior if r.get("status") == "ok" or not retry_errors}
    todo = [req for req, _ in pairs if req["_evaluation"]["run_id"] not in done]

    def one(row):
        client = Client(base_url, served_model, timeout=timeout, api_key=api_key)
        t = time.monotonic()
        result = {"run_id": row["_evaluation"]["run_id"], "started_utc": utc_now()}
        try:
            result.update(status="ok", response=answer_row(client, row, style))
        except Exception as e:  # noqa: BLE001 - recorded per row, like the preview
            result.update(status="error", error=str(e)[:500], exception=type(e).__name__)
        result.update(http_calls=client.calls, wall_seconds=round(time.monotonic() - t, 4))
        return result

    t0 = time.monotonic()
    n = 0
    with open(journal, "a", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for result in pool.map(one, todo):
            f.write(json.dumps(result, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n += 1
            if progress and (n % 100 == 0 or n == len(todo)):
                progress(f"{fam}: {n}/{len(todo)} new answers")
    return {"family": fam, "new_rows": n, "wall_seconds": round(time.monotonic() - t0, 2)}


def latest_rows(journal_rows):
    """Last row per run id (a retried error is superseded by its retry)."""
    last = {}
    for r in journal_rows:
        last[r["run_id"]] = r
    return list(last.values())


def score_family(pairs, out_dir, fam):
    requests = [req for req, _ in pairs]
    golds = {g["id"]: g for _, g in pairs}
    journal = latest_rows(read_jsonl_tolerant(Path(out_dir) / fam / "results.jsonl"))
    res = scoring.score_journal(requests, golds, journal)
    return res, scoring.family_summary(res["items"])


def _clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def build_report(loaded, out_dir, model, revision, engine, loader_reports, items_manifest_sha256):
    fam_stats, items_by_family, quality = {}, {}, {}
    for fam, pairs in loaded.items():
        res, summ = score_family(pairs, out_dir, fam)
        fam_stats[fam] = summ
        items_by_family[fam] = res["items"]
        quality[fam] = {k: res[k] for k in ("n_error", "n_unscorable", "n_missing")}
        quality[fam]["n_planned"] = len(pairs)
    agg = scoring.weighted_aggregate(fam_stats, itemlib.WEIGHTS)
    aa_fams = [f for f in fam_stats if f != "PV-HF"]
    aa = scoring.weighted_aggregate(fam_stats, {f: 1.0 for f in aa_fams})
    complete = all(len(loaded.get(f, [])) == loader_reports.get(f, {}).get("indexed", -1) and quality.get(f, {}).get("n_error", 1) == 0
                   and quality.get(f, {}).get("n_missing", 1) == 0 for f in itemlib.FAMILIES)
    body = {
        "kit": "reax-yardstick", "kit_version": __version__, "created_utc": utc_now(),
        "model": model, "revision": revision, "engine": engine,
        "items_manifest_sha256": items_manifest_sha256,
        "families": fam_stats, "quality": quality, "loader": loader_reports,
        "composite": agg["mean"], "composite_se": agg["se"], "weights_declared": itemlib.WEIGHTS, "weights_used": agg["weights_used"],
        "hf_public": fam_stats.get("PV-HF", {}).get("mean"), "aa_public": aa["mean"],
        "complete": complete,
        "note": "Proxy on public items, read like the REAX preview. Official scores use rotating private items.",
    }
    body = _clean(body)
    atomic_write_json(Path(out_dir) / "yardstick.json", body)
    atomic_write_json(Path(out_dir) / "yardstick.items.json", items_by_family)
    atomic_write_text(Path(out_dir) / "summary.md", summary_text(body))
    return body


def _fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def summary_text(b):
    lines = [f"# REAX yardstick: {b['model']}" + (f"@{b['revision']}" if b.get("revision") else ""), "",
             f"Composite (weights {', '.join(f'{k} {v}' for k, v in b['weights_used'].items())}): "
             f"**{_fmt(b['composite'])}** ± {_fmt(b['composite_se'])} (SE)", "",
             "| family | score (0-100) | SE | items scored | errors | not loaded |", "|---|---:|---:|---:|---:|---:|"]
    for fam in itemlib.FAMILIES:
        s = b["families"].get(fam)
        q = b["quality"].get(fam, {})
        lr = b["loader"].get(fam, {})
        not_loaded = len(lr.get("missing", [])) + len(lr.get("hash_mismatch", []))
        if s is None:
            lines.append(f"| {fam} | not run | | 0 | | {not_loaded} |")
        else:
            lines.append(f"| {fam} | {_fmt(s['mean'])} | {_fmt(s['se'])} | {s['n']} | {q.get('n_error', 0)} | {not_loaded} |")
    lines += ["", f"HF public: {_fmt(b['hf_public'])} · AA-style public (mean of three families): {_fmt(b['aa_public'])}", "",
              "Complete run (every item loaded and answered): " + ("yes" if b["complete"] else "**no** (composite uses the families present, weights renormalised)"),
              "", b["note"]]
    for fam, lr in b["loader"].items():
        if lr.get("note"):
            lines.append(f"- {fam}: {lr['note']}")
    return "\n".join(lines) + "\n"


def load_all(families, di_suite, hle_source, limit, items_dir=itemlib.ITEMS_DIR):
    loaded, reports = {}, {}
    for fam in families:
        ids = [r["id"] for r in itemlib.load_index(fam, items_dir)]
        wanted = set(itemlib.subsample(sorted(ids), limit))
        pairs, rep = itemlib.load_family(fam, wanted, di_suite=di_suite, hle_source=hle_source, items_dir=items_dir)
        rep["indexed"] = len(wanted)
        reports[fam] = rep
        if pairs:
            loaded[fam] = pairs
    return loaded, reports


def items_manifest_sha256(items_dir=itemlib.ITEMS_DIR):
    m = read_json(Path(items_dir) / "MANIFEST.json") or {}
    return m.get("manifest_sha256")
