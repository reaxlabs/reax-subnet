#!/usr/bin/env python3
"""Set the multirole release start block in the build (release constants of src/reax_subnet/elements.py).

    scripts/set_multirole_start.py <block>        # both constants = <block>
    scripts/set_multirole_start.py none           # held again (committed default)

Rewrites ``MULTIMODEL_START_BLOCK`` and ``DUALROLE_START_BLOCK`` and asserts the result by re-reading the file.
Proposal (docs/DESIGN-MULTIROLE.md): deploy block + 7,200 (~24 h). Choosing the block is the release owner's decision.
"""
from __future__ import annotations

import pathlib
import re
import sys

TARGET = pathlib.Path(__file__).resolve().parent.parent / "src" / "reax_subnet" / "elements.py"
NAMES = ("MULTIMODEL_START_BLOCK", "DUALROLE_START_BLOCK")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    arg = argv[1].strip().lower()
    if arg == "none":
        value = None
    elif arg.isdigit() and int(arg) > 0:
        value = int(arg)
    else:
        print(f"block must be a positive integer or 'none', got {argv[1]!r}", file=sys.stderr)
        return 2
    text = TARGET.read_text()
    for name in NAMES:
        text, n = re.subn(rf"^{name}: Optional\[int\] = .*$", f"{name}: Optional[int] = {value}", text, flags=re.M)
        if n != 1:
            print(f"{name}: expected exactly one definition, found {n}", file=sys.stderr)
            return 1
    TARGET.write_text(text)
    after = TARGET.read_text()
    for name in NAMES:
        assert re.findall(rf"^{name}: Optional\[int\] = (.*)$", after, flags=re.M) == [str(value)], name
    print(f"{TARGET}: {' = '.join(NAMES)} = {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
