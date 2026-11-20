"""CLI.

  python3 -m reax_yardstick run --model <org/repo>@<revision> (--engine-url URL | --vllm) [options]
  python3 -m reax_yardstick score --out DIR            # rescore an existing run directory
  python3 -m reax_yardstick verify-items [--di-suite PATH ...] [--hle PATH|hub]   # check every item hash, no model
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import re
import sys
from pathlib import Path

from . import items as itemlib
from . import run as runlib
from .common import atomic_write_json, read_json


def parse_model(spec):
    m = re.fullmatch(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:@([0-9a-f]{7,40}|[A-Za-z0-9_.\-/]+))?", spec)
    if not m:
        raise SystemExit(f"--model must be org/repo@revision: {spec}")
    return m.group(1), m.group(2)


def add_item_args(p):
    p.add_argument("--di-suite", nargs="*", default=[], help="rebuilt Decision Index public suite files or directory (PV-HF)")
    p.add_argument("--hle", default=None, help="cais/hle test split: .parquet, .jsonl export, or 'hub' (PV-HLE)")
    p.add_argument("--families", default=",".join(itemlib.FAMILIES), help="comma list, default all four")
    p.add_argument("--limit", type=int, default=0, help="at most N items per family (seeded, for a quick check)")


@contextmanager
def output_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another yardstick command is using {out}") from None
        yield


def cmd_run(a):
    repo, revision = parse_model(a.model)
    if a.vllm and (not revision or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise SystemExit("--vllm needs a pinned revision: --model org/repo@<commit sha>")
    if bool(a.engine_url) == bool(a.vllm):
        raise SystemExit("give exactly one of --engine-url URL or --vllm")
    out = Path(a.out or f"yardstick-runs/{repo.replace('/', '--')}-{(revision or 'unpinned')[:12]}")
    with output_lock(out):
        return run_locked(a, repo, revision, out)


def run_locked(a, repo, revision, out):
    families = [f for f in a.families.split(",") if f]
    served = a.served_model_name or repo
    identity = {"model": repo, "revision": revision, "families": families, "limit": a.limit,
                "engine_url": a.engine_url, "vllm": bool(a.vllm), "served_model": served,
                "max_model_len": a.max_model_len if a.vllm else None,
                "items_manifest_sha256": runlib.items_manifest_sha256()}
    meta_path = out / "run.json"
    prior = read_json(meta_path)
    if prior != identity and (prior is not None or any(out.glob("*/results.jsonl"))):
        raise SystemExit(f"{out} contains results for a different or unverified run; use a new --out directory")
    loaded, reports = runlib.load_all(families, a.di_suite, a.hle, a.limit)
    for fam in families:
        r = reports[fam]
        print(f"{fam}: {r['loaded']}/{r['indexed']} items loaded" + (f" — {r['note']}" if r.get("note") else "")
              + (f"; {len(r['hash_mismatch'])} hash mismatches" if r["hash_mismatch"] else ""), file=sys.stderr)
    if not loaded:
        raise SystemExit("no items loaded")
    atomic_write_json(meta_path, identity)
    engine = {"mode": "engine-url" if a.engine_url else "vllm", "readout": "fast next-token label logprobs", "concurrency": a.concurrency}

    def go(base_url):
        for fam, pairs in loaded.items():
            print(runlib.run_family(fam, pairs, out, base_url, served, a.concurrency, a.timeout, a.api_key, a.retry_errors,
                                    progress=lambda s: print(s, file=sys.stderr)), file=sys.stderr)

    if a.engine_url:
        go(a.engine_url)
    else:
        from . import serve
        mdir = serve.stage(repo, revision, out / "work", log=lambda s: print(s, file=sys.stderr))
        port = serve.free_port()
        cmd = serve.serve_command(a.vllm_bin, mdir, served, port, max_model_len=a.max_model_len,
                                  multimodal_arch=serve.is_multimodal_arch(mdir))
        engine.update(vllm_pin_expected=serve.VLLM_PIN, max_model_len=a.max_model_len)
        with serve.OwnedServer(cmd, out / "server", port) as srv:
            srv.wait_ready(a.load_timeout)
            go(f"http://127.0.0.1:{port}")
    body = runlib.build_report(loaded, out, repo, revision, engine, reports, runlib.items_manifest_sha256())
    print((out / "summary.md").read_text())
    print(f"written: {out / 'yardstick.json'}", file=sys.stderr)
    return 0 if body["complete"] else 3


def cmd_score(a):
    out = Path(a.out)
    with output_lock(out):
        meta = read_json(out / "run.json")
        manifest_sha = runlib.items_manifest_sha256()
        if not meta or meta.get("items_manifest_sha256") != manifest_sha:
            raise SystemExit(f"{out} has no run for this item manifest")
        loaded, reports = runlib.load_all(meta["families"], a.di_suite, a.hle, meta["limit"])
        runlib.build_report(loaded, out, meta["model"], meta.get("revision"), {"mode": "rescore"}, reports, manifest_sha)
        print((out / "summary.md").read_text())
    return 0


def cmd_verify(a):
    loaded, reports = runlib.load_all([f for f in a.families.split(",") if f], a.di_suite, a.hle, a.limit)
    print(json.dumps({f: {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()} for f, r in reports.items()}, indent=2))
    optional_unsupplied = {"PV-HF": not a.di_suite, "PV-HLE": not a.hle}
    bad = any(r["hash_mismatch"] or (r["missing"] and not optional_unsupplied.get(fam))
              for fam, r in reports.items())
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m reax_yardstick")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure one model")
    r.add_argument("--model", required=True, help="org/repo@revision (revision = commit sha, as in a REAX commitment)")
    r.add_argument("--engine-url", help="base URL of a running vLLM OpenAI-compatible server serving this model")
    r.add_argument("--vllm", action="store_true", help="download, normalise and serve the model with vLLM like the preview")
    r.add_argument("--vllm-bin", default="vllm")
    r.add_argument("--max-model-len", type=int, default=16384)
    r.add_argument("--load-timeout", type=float, default=1800)
    r.add_argument("--served-model-name", default=None, help="model name the server expects (default: the repo id)")
    r.add_argument("--api-key", default="", help="bearer token for --engine-url, if your server needs one")
    r.add_argument("--concurrency", type=int, default=48)
    r.add_argument("--timeout", type=float, default=120)
    r.add_argument("--retry-errors", action="store_true", help="ask failed items again on resume")
    r.add_argument("--out", default=None)
    add_item_args(r)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("score", help="rescore an existing run directory")
    s.add_argument("--out", required=True)
    s.add_argument("--di-suite", nargs="*", default=[])
    s.add_argument("--hle", default=None)
    s.set_defaults(fn=cmd_score)
    v = sub.add_parser("verify-items", help="load every item and check its hash (no model needed)")
    add_item_args(v)
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
