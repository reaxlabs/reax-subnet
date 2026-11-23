"""Check that the tables the gen worker and the validator must share are byte-identical between two checkouts.

Usage: python scripts/check_shared_tables.py <worker checkout> <validator checkout>
Compares the source text of the shared ``competition_split`` names (slot list, commit cutoff, keep-alive base floors) and
``spec/machine.py`` as a whole (``FAST_CONFIRM_*``, ``confirm_delay_s``). Exit 0 = identical, 1 = a difference (printed).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SHARED_NAMES = ("_crown_slots", "EXTRA_CROWN_SLOTS_UTC", "CROWN_SLOTS_UTC", "GENERALIST_COMMIT_CUTOFF_S",
                "TRACK_ENTRY_FLOOR", "KEEPALIVE_BIG_PARAMS", "KEEPALIVE_BIG_MARGIN", "KEEPALIVE_BASE_FLOOR")
SPLIT = Path("src/reax_subnet/competition_split.py")
MACHINE = Path("src/reax_subnet/spec/machine.py")


def segments(path: Path) -> dict[str, str]:
    """Top-level name -> exact source text of its definition."""
    src = path.read_text()
    out: dict[str, str] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            continue
        if name in SHARED_NAMES:
            # end_lineno covers a trailing comment line inside a literal (e.g. the KEEPALIVE_BASE_FLOOR marker)
            out[name] = "\n".join(src.splitlines()[node.lineno - 1:node.end_lineno])
    return out


def diff(worker: Path, validator: Path) -> list[str]:
    problems = []
    w, v = segments(worker / SPLIT), segments(validator / SPLIT)
    for name in SHARED_NAMES:
        if name not in w or name not in v:
            problems.append(f"{name}: missing in {'worker' if name not in w else 'validator'}")
        elif w[name] != v[name]:
            problems.append(f"{name}: source differs")
    if (worker / MACHINE).read_bytes() != (validator / MACHINE).read_bytes():
        problems.append("spec/machine.py differs")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    problems = diff(Path(argv[1]), Path(argv[2]))
    for p in problems:
        print("DIFF", p)
    if not problems:
        print("OK: shared tables identical (" + ", ".join(SHARED_NAMES) + ", spec/machine.py)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
