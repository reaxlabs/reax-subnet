#!/usr/bin/env python3
"""Write one UTC day's EU contract-tier market inputs into the validator's market file (REAX_EU_MARKET_FILE).

Runs on Sandy (or the validator host) once per day shortly after 00:00Z; the validator itself never makes a network call.
Inputs (pay rule 2 of the decided spec): the alpha price in TAO from the on-chain pool, TAO/EUR from at least two
sources (CoinGecko + one more; the validator takes the median), and the day's inference-share alpha estimate (for the
5 % cap). Each quote carries the time it was taken; the validator ignores quotes more than 2 h away from 00:00Z.

    python tools/eu_tier_quote.py --file /srv/reax/eu-market.json --day 2026-10-26 \
        --alpha-tao 0.031057 --tao-eur coingecko=233.70 --tao-eur kraken=233.95 --inference-share-alpha 1681

``--fetch-coingecko`` fetches the CoinGecko TAO/EUR spot instead of passing it (network; not used in tests).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import urllib.request
from datetime import date, datetime, timezone

COINGECKO = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=eur"


def _pos(v: float, what: str) -> float:
    if not math.isfinite(v) or v <= 0:
        raise SystemExit(f"{what} must be a positive number")
    return v


def fetch_coingecko(timeout: float = 15.0) -> float:
    with urllib.request.urlopen(COINGECKO, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
        return float(json.load(r)["bittensor"]["eur"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", required=True)
    ap.add_argument("--day", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--alpha-tao", type=float, required=True)
    ap.add_argument("--tao-eur", action="append", default=[], metavar="SOURCE=VALUE")
    ap.add_argument("--fetch-coingecko", action="store_true")
    ap.add_argument("--inference-share-alpha", type=float)
    ap.add_argument("--at", help="quote time (ISO 8601, default now)")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.day).isoformat()
    at = a.at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    tao: dict[str, dict] = {}
    for item in a.tao_eur:
        src, _, val = item.partition("=")
        tao[src.strip()] = {"value": _pos(float(val), f"tao-eur {src}"), "at": at}
    if a.fetch_coingecko:
        tao["coingecko"] = {"value": _pos(fetch_coingecko(), "coingecko"), "at": at}
    if len(tao) < 2:
        print("warning: fewer than 2 TAO/EUR sources: the validator treats the price as missing", file=sys.stderr)
    entry = {"alpha_tao": {"value": _pos(a.alpha_tao, "alpha-tao"), "at": at}, "tao_eur": tao}
    if a.inference_share_alpha is not None:
        entry["inference_share_alpha"] = _pos(a.inference_share_alpha, "inference-share-alpha")
    try:
        with open(a.file, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        doc = {"version": 1, "days": {}}
    doc.setdefault("days", {})[day] = entry
    d = os.path.dirname(os.path.abspath(a.file))
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, a.file)
    print(json.dumps({"day": day, **entry}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
