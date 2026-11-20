"""reax-lineage command line: check | dry-run | fingerprint."""
import argparse
import json
import os
import sys

from . import __version__
from .fp import SEED_A_DEFAULT, SampleSpec, get_fingerprint, seed_sha256
from .hf import HFClient, HFError, LocalSource, scrub_url
from .sources import Hub

# Exit codes: 0 = done (a result was written / printed); 2 = the run ended without a result and said why: one JSON line on stdout,
# {"error": <message>, "error_kind": <kind>, "http_status": <int or null>}; never a traceback. Anything else (exit 1, traceback) is a bug of the checker.
#   error_kind "repo_unreadable": a repo, revision or file cannot be read (HTTP 401 / 403 / 404 and other 4xx except 408 / 425 / 429: private, gated, missing,
#                                 removed; or not a readable model repo). Retrying does not help until somebody changes something. Never a reason to refuse.
#   error_kind "transient":       HTTP 429 / 5xx / 408 / 425, timeouts, connection errors, a server that ignores Range requests. Try again later.
#   error_kind "check_error":     the checker refused to run on its input (CheckError).
EXIT_ERROR = 2


def error_result(kind, msg, status=None):
    return {"error": scrub_url(msg), "error_kind": kind, "http_status": status}


def seed_for(args, seed_id):
    if seed_id == "B":
        s = os.environ.get("REAX_LINEAGE_SEED_B")
        if not s:
            raise SystemExit("Sample B needs the sealed seed in the environment variable REAX_LINEAGE_SEED_B")
        return s
    return getattr(args, "seed", None) or os.environ.get("REAX_LINEAGE_SEED_A") or SEED_A_DEFAULT


def cmd_fingerprint(args):
    spec = SampleSpec(rows=args.rows, min_elems=args.min_elems)
    if args.local:
        src = LocalSource(args.local)
        if args.name:
            src.repo = args.name
    else:
        hub = Hub(hub_dir=args.hub_dir)
        rev = hub.resolve_rev(args.repo, args.rev or "main")
        src = hub.source(args.repo, rev)
    seed = seed_for(args, args.seed_id)
    fp = get_fingerprint(src, seed, args.seed_id, cache_dir=args.cache, spec=spec, refresh=args.refresh)
    if args.out:
        fp.save(args.out)
    m = fp.meta
    print(json.dumps({"repo": m["repo"], "rev": m["rev"], "seed_id": m["seed_id"], "seed_sha256": m["seed_sha256"],
                      "n_tensors": m["n_tensors"], "values": m["values"], "seconds": m["seconds"],
                      "peak_rss_mb": m["peak_rss_mb"], "warnings": m["warnings"]}))
    return 0


def cmd_check(args):
    from .check import Check, CheckError
    from .measure import Thresholds
    if not args.repo and not args.local:
        raise SystemExit("check needs --repo and --rev (or --local DIR for a local checkpoint)")
    hub = Hub(hub_dir=args.hub_dir)
    thr = Thresholds.load(args.thresholds)
    ck = Check(args.repo, args.rev or "main", hotkey=args.hotkey, block=args.block, extrinsic=args.extrinsic,
               commit_time=args.commit_time, record=args.record, list_path=args.list, db_path=args.comparison_db,
               cache_dir=args.cache, hub=hub, thr=thr, dry_run=(args.cmd == "dry-run"), seed_a=args.seed, jobs=args.jobs,
               tier_b=not args.no_tier_b, holder_ids=args.holder, local_dir=args.local, auto_hf_holder=args.auto_hf_holder)
    try:
        res = ck.run()
    except CheckError as e:
        print(json.dumps(error_result("check_error", str(e))))
        return EXIT_ERROR
    txt = json.dumps(res, indent=1, sort_keys=False, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    print(json.dumps({"result": res["result"], "reasons": [r["code"] for r in res["reasons"]],
                      "sources": [(d.get("public_name") or d["model"], d["beta"]) for d in res["sources_detected"]],
                      "seconds": res["total_seconds"], "out": args.out}))
    return 0


def cmd_export_db(args):
    """Write a copy of the comparison DB that holds only public rows (RULES 2.7 / 3.2: the rows and fingerprints of entries that are still private are confidential and are never exported)."""
    from .db import ComparisonDB
    out = ComparisonDB(args.comparison_db).export_public(args.out)
    print(json.dumps(out))
    return 0


def _check_args(c, dry):
    c.add_argument("--repo")
    c.add_argument("--rev")
    c.add_argument("--local", help="check a local safetensors dir as the entry (offline; --repo is then only its name)")
    c.add_argument("--hotkey")
    c.add_argument("--block", type=int)
    c.add_argument("--extrinsic", type=int, default=0)
    c.add_argument("--commit-time", help="UTC ISO time of the commitment block (public models are compared by their HF commit time)")
    c.add_argument("--record", help="record file (default: reax-lineage.json of the repo at the revision)")
    c.add_argument("--list", help="list.json (the List at the commitment time)")
    c.add_argument("--comparison-db", help="comparison database directory (index.json + fingerprints)")
    c.add_argument("--out", help="result.json")
    c.add_argument("--cache", default=os.environ.get("REAX_LINEAGE_CACHE"))
    c.add_argument("--hub-dir", help="offline hub: DIR/<org>/<name>/<revision>/ holds the repos")
    c.add_argument("--jobs", type=int, default=None, help="worker processes for the candidate scan")
    c.add_argument("--seed", help="Sample A seed (default %s)" % SEED_A_DEFAULT)
    c.add_argument("--thresholds", help="thresholds.json (default: the calibrated file shipped with the checker)")
    c.add_argument("--no-tier-b", action="store_true", help="do not run Tier B (then a failed frame check ends as NEEDS_CORRECTION, manual)")
    c.add_argument("--holder", action="append", help="holder id of the entry (coldkey or hf:<account>); repeatable. RULES 2.9: the caller passes the coldkey and the accounts linked through "
                                                     "passed entries; the only id the checker adds itself is hf:<account of the entry's repo id>, and only when the record names "
                                                     "the committing hotkey (--hotkey)")
    c.add_argument("--auto-hf-holder", action="store_true",
                   help="MANUAL USE ONLY (default off; the intake service never sets it): also count hf:<owner of the repo id> as a holder id of the entry and of every "
                        "comparison model, as v1.4.1 did. Wrong under RULES 2.9 for a commitment of someone else's repo")


def build_parser():
    p = argparse.ArgumentParser(prog="reax-lineage", description="REAX stage-0 lineage checker")
    p.add_argument("--version", action="version", version="reax-lineage " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fingerprint", help="fetch (or load from cache) the weight sample of one model")
    f.add_argument("--repo")
    f.add_argument("--rev")
    f.add_argument("--local", help="local safetensors directory instead of a Hub repo")
    f.add_argument("--name", help="display name for a local checkpoint")
    f.add_argument("--seed-id", choices=["A", "B"], default="A")
    f.add_argument("--seed", help="Sample A seed (default %s)" % SEED_A_DEFAULT)
    f.add_argument("--rows", type=int, default=4)
    f.add_argument("--min-elems", type=int, default=1_000_000)
    f.add_argument("--cache", default=os.environ.get("REAX_LINEAGE_CACHE"))
    f.add_argument("--out")
    f.add_argument("--hub-dir")
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(fn=cmd_fingerprint)

    c = sub.add_parser("check", help="stage-0 intake check (needs the sealed Sample B seed in REAX_LINEAGE_SEED_B for the full check)")
    _check_args(c, False)
    c.set_defaults(fn=cmd_check)
    d = sub.add_parser("dry-run", help="the same check on Sample A only (what the MCP's lineage_dry_run calls); no sealed seed; compares only with public models and revealed entries (RULES 3.2)")
    _check_args(d, True)
    d.set_defaults(fn=cmd_check)
    e = sub.add_parser("export-db", help="copy the comparison DB without its private rows (entries that are still private are confidential and never exported)")
    e.add_argument("--comparison-db", required=True)
    e.add_argument("--out", required=True, help="a new or empty directory")
    e.set_defaults(fn=cmd_export_db)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd == "fingerprint" and not (a.local or a.repo):
        p.error("fingerprint needs --repo (and --rev) or --local")
    if a.cmd in ("check", "dry-run") and not (a.local or a.repo):
        p.error("%s needs --repo (and --rev) or --local" % a.cmd)
    try:
        return a.fn(a)
    except HFError as e:                 # a private / gated / missing repo, a failing Hugging Face: clean JSON and exit 2, never a traceback
        print(json.dumps(error_result(e.kind, str(e), e.status)))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
