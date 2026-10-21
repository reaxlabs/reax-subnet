"""``reax-commit``: read and edit the elements of this hotkey's on-chain commitment (docs/DESIGN-MULTIROLE.md section 1).

Every write to the Commitments pallet replaces all fields, so a hand-made ``btcli`` commit of one element silently drops the
other one (an inference element or a competition entry). This tool always merges: the element you set replaces the one of
its kind, the other kind stays.

    reax-commit show [--hotkey <ss58>]                 # fields on chain and how this release reads them
    reax-commit set <element> [--dry-run]              # any valid element: inference, use-case/generalist track, engine track
    reax-commit withdraw competition|inference [--dry-run]

Wallet, network and netuid come from the same environment variables as ``reax-miner`` (REAX_WALLET_NAME, REAX_WALLET_HOTKEY,
REAX_WALLET_PATH, REAX_NETWORK, REAX_NETUID) or the options of the same names. Two elements in one commitment, and several
models in one inference element, are read by validators only from the start blocks of this release
(``elements.DUALROLE_START_BLOCK`` / ``MULTIMODEL_START_BLOCK``); before that the tool refuses unless ``--replace-all``
(write the new element alone, what a plain commit does) or REAX_FORCE_MULTIROLE=1.

The planning functions (``write_plan``, ``withdraw_plan``) are shared with ``reax-miner`` and ``reax-engine-submit``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Mapping, Optional, Sequence

from . import elements as E


class Refused(Exception):
    """The write would not do what the caller expects (or would delete an element); the message says why."""


def force_multirole(env: Mapping[str, str] = os.environ) -> bool:
    return (env.get("REAX_FORCE_MULTIROLE") or "").strip().lower() in ("1", "true", "yes", "on")


def _since(start: Optional[int]) -> str:
    return "is held in this release" if start is None else f"counts from block {start}"


def elements_of(fields: Sequence[str]) -> dict[str, str]:
    """{kind: text} of the valid elements among ``fields`` (unparseable fields and repeated kinds are ignored)."""
    out: dict[str, str] = {}
    for f in fields:
        f = f.strip() if isinstance(f, str) else ""
        kind = E.kind_of(f)
        if kind is not None and kind not in out:
            out[kind] = f
    return out


def write_plan(current: Sequence[str], element: str, block: int, *, replace_all: bool = False,
               force: bool = False) -> list[str]:
    """The fields to write so that ``element`` is on chain next to the other kind's current element.

    Raises ``Refused`` where validators would not read the result yet (a second element before DUALROLE_START_BLOCK, a
    multi-model element before MULTIMODEL_START_BLOCK) unless ``force``; ``replace_all`` writes the element alone."""
    try:
        merged = E.merge_fields(current, element)
    except ValueError as exc:
        raise Refused(str(exc)) from None
    if replace_all:
        merged = [element]
    if E.is_multimodel(element) and not force and not E._started(E.MULTIMODEL_START_BLOCK, block):
        raise Refused(f"refusing to write a multi-model inference element: it {_since(E.MULTIMODEL_START_BLOCK)} "
                      f"(chain block now {block}); validators would not score this hotkey at all. "
                      "REAX_FORCE_MULTIROLE=1 writes it anyway")
    if len(merged) > 1 and not force and not E._started(E.DUALROLE_START_BLOCK, block):
        other = next(f for f in merged if f != element)
        raise Refused(f"refusing to write two elements: a commitment with an inference and a competition element "
                      f"{_since(E.DUALROLE_START_BLOCK)} (chain block now {block}); before that validators read no role "
                      f"from it. Writing only the new element would drop your other element ({other[:60]}...). "
                      "Use --replace-all to write the new element alone (a plain commit does this), or "
                      "REAX_FORCE_MULTIROLE=1 to write both anyway")
    if not force and all(E.kind_of(f) for f in merged) and E.split_elements(merged, block) is None:
        # safety net: never write (or report as published) a commitment validators of this release read no role from
        raise Refused(f"refusing to write {' + '.join(f[:40] for f in merged)}: validators of this release would read "
                      f"no role from it at block {block}")
    return merged


def withdraw_plan(current: Sequence[str], kind: str) -> list[str]:
    """The fields that remain when the element of ``kind`` is removed; raises ``Refused`` if none would remain."""
    have = elements_of(current)
    if kind not in have:
        raise Refused(f"the commitment has no {kind} element")
    # the other fields stay verbatim, also an element of a later release this one cannot read (never delete it)
    rest = [f.strip() for f in current if isinstance(f, str) and E.loose_kind(f.strip()) not in (None, kind)]
    if not rest:
        raise Refused(f"the {kind} element is the only one on chain; an empty commitment cannot be written. "
                      "Set the replacement element instead (reax-commit set ...)")
    return rest


def same_elements(current: Sequence[str], fields: Sequence[str]) -> bool:
    """True if writing ``fields`` would change nothing (same elements; no unreadable field on chain to clean up)."""
    have = elements_of(current)
    return have == elements_of(fields) and len(current) == len(have)


# ------------------------------------------------------------------------------------------------------------- show
def describe(fields: Sequence[str], block: int) -> list[str]:
    """Human-readable lines: the fields and how this release parses them at ``block``."""
    from .manifest import load_manifest

    if not fields:
        return ["no commitment on chain"]
    lines = [f"{len(fields)} field(s) on chain:"]
    for i, f in enumerate(fields):
        kind = E.kind_of(f.strip()) if isinstance(f, str) else None
        lines.append(f"  [{i}] {f!r}  -> {kind + ' element' if kind else 'NOT a valid element of this release'}")
    parsed = E.split_elements(tuple(fields), block)
    if parsed is None:
        lines.append(f"validators of this release read NO role from this commitment at block {block} "
                     f"(invalid, or multi-model {_since(E.MULTIMODEL_START_BLOCK)}, or two elements "
                     f"{_since(E.DUALROLE_START_BLOCK)})")
    else:
        lines.append(f"read at block {block}: inference={parsed.inference or '-'} competition={parsed.competition or '-'}")
    models = E.parse_inference(elements_of(fields).get(E.INFERENCE, ""))
    for model, mid in (models or {}).items():
        try:
            current = load_manifest(model).manifest_id
        except Exception as exc:  # noqa: BLE001 - a model without a manifest in this build
            lines.append(f"  model {model}: manifest {mid} (cannot compare: {exc})")
            continue
        verdict = "matches this release" if mid == current else f"STALE, this release has {current} (not scored)"
        lines.append(f"  model {model}: manifest {mid} {verdict}")
    lines.append(f"multi-model element {_since(E.MULTIMODEL_START_BLOCK)}; two elements {_since(E.DUALROLE_START_BLOCK)}")
    return lines


# ------------------------------------------------------------------------------------------------------------- main
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reax-commit", description=__doc__.split("\n\n")[0])
    for name in ("wallet-name", "wallet-hotkey", "wallet-path", "network", "netuid"):
        ap.add_argument(f"--{name}", default="", help=f"default: REAX_{name.upper().replace('-', '_')}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="print the fields and how this release parses them")
    show.add_argument("--hotkey", default="", help="any hotkey's ss58 address (default: the wallet's hotkey)")
    for name in ("set", "withdraw"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true", help="print the fields that would be written, write nothing")
        if name == "set":
            p.add_argument("element")
            p.add_argument("--replace-all", action="store_true",
                           help="write the element alone (drops the other kind), as before this release")
        else:
            p.add_argument("kind", choices=("competition", "inference"))
    return ap


def _open_chain(args: argparse.Namespace, *, need_wallet: bool) -> tuple[Any, str]:
    """(chain, hotkey ss58). Reads are possible without the wallet when ``--hotkey`` is given."""
    from .chain import SdkChain
    from .cli import _wallet
    from .config import ConfigError, load_common

    env = dict(os.environ)
    for name in ("wallet-name", "wallet-hotkey", "wallet-path", "network", "netuid"):
        value = getattr(args, name.replace("-", "_"))
        if value:
            env[f"REAX_{name.upper().replace('-', '_')}"] = value
    try:
        common = load_common(env)
        if not need_wallet and getattr(args, "hotkey", ""):
            return SdkChain(common.network, common.netuid, None, commitments=True), args.hotkey
        wallet, hotkey = _wallet(common)
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    return SdkChain(common.network, common.netuid, wallet, commitments=True), hotkey


def _mine(chain: Any, hotkey: str) -> tuple[Any, int]:
    snap = chain.snapshot()
    neuron = next((n for n in snap.neurons if n.hotkey == hotkey), None)
    if neuron is None:
        raise Refused(f"hotkey {hotkey} is not registered on this subnet")
    return neuron, snap.block


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "show":
            chain, hotkey = _open_chain(args, need_wallet=False)
            neuron, block = _mine(chain, hotkey)
            print(f"hotkey {hotkey} uid {neuron.uid}, chain block {block}, commitment block {neuron.commitment_block or '-'}")
            print("\n".join(describe(E.fields_of(neuron), block)))
            return 0
        chain, hotkey = _open_chain(args, need_wallet=True)
        neuron, block = _mine(chain, hotkey)
        current = E.fields_of(neuron)
        if args.cmd == "set":
            fields = write_plan(current, args.element.strip(), block, replace_all=args.replace_all,
                                force=force_multirole())
        else:
            fields = withdraw_plan(current, args.kind)
    except Refused as exc:
        print(f"reax-commit: {exc}", file=sys.stderr)
        return 1
    if same_elements(current, fields):
        print(f"nothing to change: {hotkey} already has exactly these elements")
        return 0
    for i, f in enumerate(fields):
        print(f"[{i}] {f}")
    if args.dry_run:
        print(f"dry run: would write {len(fields)} field(s) for {hotkey}")
        return 0
    chain.set_commitment_fields(fields)
    print(f"published {len(fields)} field(s) for {hotkey}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
