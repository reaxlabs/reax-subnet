"""Recorded validator inputs for the EU tier no-op test (tests/test_eutier.py).

``recorded_weights`` builds a tiered S1Validator (all four pools), fills its state with a fixed, seeded miner set
(emas, capacities, roles, coldkeys incl. a coldkey farm, the owner's burn) and returns ``compute_weights`` at a fixed
clock. ``eutier_golden_b510d79.json`` holds its output computed by rc11.post4 (b510d79) itself:

    git archive b510d79 src | tar -x -C /tmp/eutier-base
    PYTHONPATH=/tmp/eutier-base/src python tests/data/eutier_recorded.py > tests/data/eutier_golden_b510d79.json

Only modules that exist in b510d79 are imported here.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)  # after COLDKEY_DR_START_UTC
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def fake_hotkey(rng: random.Random) -> str:
    return "5" + "".join(rng.choice(_B58) for _ in range(47))


def recorded_weights(tmp_dir: str, wallet, monkeypatch=None, env_extra=None, tweak=None) -> dict[int, float]:
    from reax_subnet import s1validator
    from reax_subnet.chain import Neuron, Snapshot
    from reax_subnet.config import load_validator
    from reax_subnet.elements import slot_key
    from reax_subnet.manifest import load_manifest
    from reax_subnet.s1backends import S1HashBackend
    from reax_subnet.state import MinerState

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    env = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
           "REAX_FEATURE_TIERED_POOLS": "1", "REAX_FEATURE_S1_PRO_POOL": "1", "REAX_BURN_FRACTION": "0.1",
           "REAX_STATE_PATH": f"{tmp_dir}/state.json"}

    class _Chain:
        def __init__(self, neurons, owner):
            self.neurons, self.owner = neurons, owner

    rng = random.Random(9213)
    owner_hk = fake_hotkey(rng)
    neurons = [Neuron(0, owner_hk, "cold-owner", False, 0.0, None, None),
               Neuron(1, wallet.hotkey.ss58_address, "cold-val", True, 5000.0, None, None)]
    pools = ["s1-fast", "s1-pro", "s1-lite", "s1-cpu"]
    miners: dict[int, MinerState] = {}
    for uid in range(2, 74):
        hk = fake_hotkey(rng)
        ck = "cold-farm" if 50 <= uid < 62 else f"cold-{uid}"
        neurons.append(Neuron(uid, hk, ck, False, 0.0, "127.0.0.1", 9000 + uid))
        for pool in rng.sample(pools, rng.choice([1, 1, 2])):
            role = f"{pool}:{load_manifest(pool).manifest_id}"
            ema = rng.choice([0.0, rng.uniform(0.0, 0.05), rng.uniform(0.3, 1.0), rng.uniform(0.8, 1.0)])
            cap = None if rng.random() < 0.1 else rng.uniform(0.0, 3.0) * {"s1-fast": 400, "s1-pro": 100,
                                                                          "s1-lite": 20, "s1-cpu": 3}[pool]
            miners[slot_key(uid, pool)] = MinerState(hotkey=hk, ema=ema, role=role, capacity=cap)
    if tweak is not None:
        tweak(neurons, miners)
    if monkeypatch is not None:
        for k, v in (env_extra or {}).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(s1validator, "datetime", _Clock)
    else:
        s1validator.datetime = _Clock
    val = s1validator.S1Validator(load_validator(env), wallet, _Chain(neurons, owner_hk),
                                  {m: S1HashBackend(m) for m in pools})
    val.state.miners = miners
    snap = Snapshot(block=1000, neurons=tuple(neurons), owner_hotkey=owner_hk)
    return val.compute_weights(snap)


if __name__ == "__main__":
    import tempfile

    from bittensor.keyfiles import Keypair

    class _W:
        hotkey = Keypair.create_from_uri("//Validator")

    with tempfile.TemporaryDirectory() as d:
        w = recorded_weights(d, _W())
    print(json.dumps({str(k): v for k, v in sorted(w.items())}, indent=0))
