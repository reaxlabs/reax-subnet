"""``reax-kit``: which runtime image the miner kit starts for a pool (config.POOL_RUNTIME_IMAGES, docs/MINER.md).

    reax-kit image s1-fast     # the image reference for the sidecar of that pool on this machine's GPUs
    reax-kit env [pool ...]    # REAX_<POOL>_RUNTIME_IMAGE=... lines for deploy/runtime-image.env (default: every pool)
    reax-kit show              # source, award and licence per pool, plus a timed pin (pool switch) if one is armed
    reax-kit env --at 2026-10-30T09:00Z   # the same lines as they will be at a later time (prepare a pool switch)

``REAX_<POOL>_RUNTIME_IMAGE=reference`` selects the reference runtime instead of the release default.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Sequence

from . import config as C


def _var(pool: str) -> str:
    return f"REAX_{pool.upper().replace('-', '_')}_RUNTIME_IMAGE"


def _at(text: str) -> datetime:
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="reax-kit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_image = sub.add_parser("image", help="print the sidecar image of a pool")
    p_image.add_argument("pool", choices=C.POOL_MODELS)
    p_env = sub.add_parser("env", help="print env lines for the compose file")
    p_env.add_argument("pools", nargs="*", metavar="pool")
    p_show = sub.add_parser("show", help="print the release default per pool")
    for p in (p_image, p_env, p_show):
        p.add_argument("--at", type=_at, default=None, help="evaluate at this UTC time (ISO 8601) instead of now")
    args = ap.parse_args(argv)
    now = args.at or datetime.now(timezone.utc)
    try:
        if args.cmd == "image":
            print(C.pool_runtime_image(args.pool, os.environ, C.local_gpus(), now))
        elif args.cmd == "env":
            bad = [p for p in args.pools if p not in C.POOL_MODELS]
            if bad:
                ap.error(f"unknown pool {bad[0]!r}; expected one of {', '.join(C.POOL_MODELS)}")
            gpus = C.local_gpus()
            for pool in args.pools or C.POOL_MODELS:
                print(f"{_var(pool)}={C.pool_runtime_image(pool, os.environ, gpus, now)}")
        else:
            C.validate_pool_runtime_images()
            for pool in C.POOL_RUNTIME_IMAGES:
                e, reference = C.pool_runtime_entry(pool, now)
                g = e.get("gpus")
                only = f" gpus={g['name_contains']} (cc {g['compute_capability']}), else reference" if g else ""
                note = f" gpus: {e['gpus_note']}" if e.get("gpus_note") else ""
                print(f"{pool}: {e['source']} {e['image'] or reference} award={e['award'] or '-'} "
                      f"licence={e['licence']}{only}{note}")
                at = C.pool_runtime_switch_at(pool)
                if at is not None and now < at:
                    n = C.POOL_RUNTIME_IMAGES_NEXT[pool]
                    print(f"  from {at.strftime('%Y-%m-%dT%H:%MZ')}: {n['source']} {n['image']} ({n.get('model', '')}) "
                          f"licence={n['licence']} gpus: {n.get('gpus_note', '-')}; rerun `reax-kit env` at that time")
    except C.ConfigError as exc:
        print(f"reax-kit: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
