"""rc10g golden: with the generalist track not started and multirole held, the competition split is bit-identical to
1.2.0rc9.post6 (bbec981) for the same inputs.

``EXPECTED`` is the digest this fuzz produced on bbec981 (public 666bf7a + order_ratio 0.5); it was computed by running
``_digest()`` there. Any change of ``combine_weights`` / ``active_tracks`` / ``profile_for`` before the generalist start
that moves a single weight changes it.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone

from reax_subnet import competition_split as CS
from reax_subnet.competition_split import active_tracks, combine_weights, profile_for

DF, RA = "demand_forecasting", "risk_assessment"
EXPECTED = "02ba1f4457fc42ae4d4882edbf6f2f239e1eb25511e7ede3d2a7f411f104f896"
# The same fuzz in two parts, each computed on bbec981 (rc9.post6) and identical on eededa1 and with the rc10 release
# constants: the schedule over instants before the release opening (2026-10-21T13:00Z), and the weight mixing alone.
RELEASE_INSTANTS = ((2026, 10, 16, 11, 59), (2026, 10, 16, 12), (2026, 10, 20, 18), (2026, 10, 21, 12, 59, 59))
EXPECTED_RELEASE_SCHEDULE = "d9cf6d5e66675832c32dde375a1299b82e3ecb0423547b974c41662c8e6c2dca"
EXPECTED_WEIGHTS = "004a200aef3adf9150ed48b4ad0c4dc421c57d72682492970fea6c8b68b62ee6"


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def _digest(instants=None, fuzz=True) -> str:
    h = hashlib.sha256()

    def feed(obj):
        h.update(json.dumps(obj, sort_keys=True).encode())

    # schedule: every instant between the competition start and the generalist start (pair end 30 Oct 12:00Z)
    for now in instants if instants is not None else (
            _utc(2026, 10, 16, 11, 59), _utc(2026, 10, 16, 12), _utc(2026, 10, 20, 18), _utc(2026, 10, 25),
            _utc(2026, 10, 29, 23, 59), _utc(2026, 10, 30, 11, 59, 59)):
        for prof in ("50/50", "80/20"):
            p = profile_for(prof, now) if "now" in profile_for.__code__.co_varnames[:2] else profile_for(prof)
            feed([now.isoformat(), prof, active_tracks(now, prof), p.engine, list(p.tracks)])
    rng = random.Random(20261020)
    for it in range(3000 if fuzz else 0):
        owner = rng.choice([None, 0])
        serving = {u: rng.choice([rng.random(), rng.random() * 1e-3, 1.0]) for u in range(0, rng.randint(0, 40))
                   if rng.random() < 0.9}
        if owner is not None and rng.random() < 0.8:
            serving[owner] = rng.random() * 0.1
        pool = list(range(100, 140))
        tracks = {t: [(rng.choice(pool), round(rng.uniform(0, 60), 3)) for _ in range(rng.randint(0, 5))]
                  for t in rng.sample([DF, RA], rng.randint(0, 2))}
        ka = {t: [(rng.choice(pool + [101]), round(rng.uniform(-1, 60), 3)) for _ in range(rng.randint(0, 14))]
              for t in rng.sample([DF, RA], rng.randint(0, 2))}
        prov = {t: [rng.choice(pool) for _ in range(rng.randint(0, 5))] for t in rng.sample([DF, RA], rng.randint(0, 2))}
        kw = dict(owner_uid=owner, burn_fraction=rng.choice([0.0, 0.05, 0.3]), keepalive=ka,
                  keepalive_share=rng.choice([None, None, 0.0, 0.1, 0.5]), provisional=prov)
        if rng.random() < 0.5:
            kw["floors"] = {DF: rng.choice([None, 20.0]), RA: None}
        if rng.random() < 0.5:
            approved = {(t, u) for t in tracks for u in pool if rng.random() < 0.6}
            kw["award_ok"] = lambda t, u, a=approved: (t, u) in a
        champ = rng.choice([None, None, rng.choice(pool)])
        frac = rng.choice([0.5, 0.75, 1.0])
        for prof in ("50/50", "80/20"):
            w = combine_weights(serving, champ, frac, tracks, prof, **kw)
            feed([it, prof, [[u, v.hex()] for u, v in w.items()]])
    return h.hexdigest()


def test_generalist_defaults_are_fail_closed():
    from generalist_defaults import defaults  # what scripts/set_generalist_start.py default restores
    d = defaults()
    assert d["GENERALIST_START_UTC"] == _utc(2026, 10, 30, 12)
    assert d["GENERALIST_COMMIT_FROM_UTC"] is None
    assert d["GENERALIST_ENTRY_FLOOR"] == float("inf")
    assert d["GENERALIST_KEEPALIVE_NONPROC_FLOOR"] == float("inf")
    assert d["GENERALIST_G_MED"] is None


def test_multirole_start_blocks_are_the_release_block():
    from reax_subnet import elements
    assert elements.MULTIMODEL_START_BLOCK == 9331987 and elements.DUALROLE_START_BLOCK == 9331987  # rc10 release constants


def test_competition_split_before_generalist_is_bit_identical_to_rc9_post6(monkeypatch):
    monkeypatch.setattr(CS, "GENERALIST_START_UTC", _utc(2026, 10, 30, 12))  # the fuzz instants run to 30 Oct 12:00Z
    assert _digest() == EXPECTED


def test_release_schedule_before_the_opening_is_bit_identical_to_rc9_post6():
    assert CS.GENERALIST_START_UTC == _utc(2026, 10, 21, 13)
    assert _digest([_utc(*a) for a in RELEASE_INSTANTS], fuzz=False) == EXPECTED_RELEASE_SCHEDULE


def test_weight_mixing_with_the_release_constants_is_bit_identical_to_rc9_post6():
    assert _digest([], fuzz=True) == EXPECTED_WEIGHTS


if __name__ == "__main__":
    print(_digest())
