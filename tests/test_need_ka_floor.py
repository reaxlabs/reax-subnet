"""Bit-exact compatibility vectors captured from 0e5ef36, not rounded decimal weights."""
import hashlib

import pytest

from reax_subnet import competition_split as CS


@pytest.mark.parametrize("case,expected", [
    ('before', 'f21e50346406198480b93fc0272091a196ad5d6fbfa802288ba384c434be62ba'),
    ('pure', 'd6ab23605c3b99578ff562114eb7aa69a624fc495f7eee4c4b66dc2ec3d9998c'),
    ('scored', '4667a11ae2676735e217106a42c8bce1032495cfc09c002235e919f9f16b9dc3'),
])
def test_need_floor_compatibility_with_0e5ef36(monkeypatch, case, expected):
    monkeypatch.setattr(CS, "NEED_RULE", True)
    base = {0: .1, 1: .009, 2: .009, 3: .009, 4: .0027, 99: .8703}
    args = dict(owner_uid=0, burn_fraction=.1, floors={}, profile="50/50",
                keepalive={"generalist": [(4, 50), (8, 45)]},
                provisional={"generalist": [9]}, engine_waiting=[10])
    now = CS.NEED_START_UTC.timestamp() + (-1 if case == "before" else 1)
    if case == "pure":
        args["keepalive"] = {"generalist": [(8, 45)]}
    if case == "scored":
        for key in ("keepalive", "provisional", "engine_waiting"):
            args.pop(key)
    weights = CS.combine_weights(base, 20, 1, {"generalist": [(4 if case == "scored" else 6, 50), (7, 45)]}, now=now, **args)
    exact = repr([(uid, weight.hex()) for uid, weight in weights.items()]).encode()
    assert hashlib.sha256(exact).hexdigest() == expected
