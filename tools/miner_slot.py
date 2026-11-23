#!/usr/bin/env python3
"""Read SN92's emission split (SubtensorModule.SubnetEmissionWeights[92]) from chain and print the exact miner-slot.json
factor (read-only; never writes). The logic lives in the package so it also runs inside the validator image, where
tools/ is not copied:

    docker exec nodes-validator-1 python -m reax_subnet.miner_slot                    # on rxnode-03
    docker exec nodes-validator-1 python -m reax_subnet.miner_slot --check /approvals/miner-slot.json
    python tools/miner_slot.py --weights 0,0,<owner>,<validator>,<miner>              # offline, from a repo checkout

See MINER-SLOT.md.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from reax_subnet.miner_slot import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
