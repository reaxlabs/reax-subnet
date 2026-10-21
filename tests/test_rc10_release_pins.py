"""rc10 release constants (commit 87679c7): pins of the values the release owner set. A change of any of them is a
consensus change and must be deliberate. Tests of the fail-closed behaviour use the defaults (generalist_defaults.py)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import elements
from reax_subnet.spec import config as C

BASE_FLOORS = {
    "0e8d23478b5b6ccefff3136e5e53ecfb7d053451ec405d2b0fe31ce3a49a713e": {"generalist": 39.23},
    "1647929265009944f8157465e27417ec84014893953f3b3ed0874e373bb276fc": {"generalist": 62.23},
    "19d89e343b9109fcb478d685cb08f15914c4aded296d02c68c2c3fa9f0fdb4df": {"generalist": 39.93},
    "564471ef18768dd040eb3ce7effaf0561b65b3b2a50cb10980974bae3ae713dd": {"generalist": 55.01},
    "81a3eb978ff3c5b90c273a100f36499a86dfec698652c2d181f43e11e89fe08d": {"generalist": 58.01},
    "572884f6e3788b5debd2084e2789a009622e50272fe6c226e1d93dcb24ec6d41": {"generalist": 55.34},  # REAX 1 @ea9d5d8b (paired offset)
    "4031ce5ec8b6f10e388f09df862810315e4132f429f8cdf308f13566bd159eef": {"generalist": 60.43},  # REAX 1 @ca2a0cc5 (official base)
    "dac9897fe790514fec2d32f666b5627e098205b051b48fe6630a3699a5138e10": {"generalist": 49.52},
}


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_generalist_release_constants():
    assert CS.GENERALIST_START_UTC == _utc(2026, 10, 21, 13, 0)
    assert CS.SPEC_OPEN_CLASS_START_UTC == CS.GENERALIST_START_UTC
    assert CS.GENERALIST_COMMIT_FROM_UTC == _utc(2026, 10, 20, 19, 9, 28)
    assert CS.GENERALIST_COMMIT_FROM_UTC <= CS.GENERALIST_START_UTC
    assert CS.GENERALIST_ENTRY_FLOOR == 36.8 and CS.TRACK_ENTRY_FLOOR["generalist"] == 36.8
    assert CS.GENERALIST_KEEPALIVE_NONPROC_FLOOR == 37.4 and CS.KEEPALIVE_NONPROC_FLOOR["generalist"] == 37.4
    assert CS.GENERALIST_G_MED == 14.84


def test_keepalive_base_floor_table():
    assert CS.KEEPALIVE_BASE_FLOOR == BASE_FLOORS


def test_multirole_start_block():
    assert elements.MULTIMODEL_START_BLOCK == 9331987 and elements.DUALROLE_START_BLOCK == 9331987


def test_first_pair_is_shortened_to_the_opening():
    assert [t.end for t in CS.TRACK_SCHEDULE[:2]] == [CS.GENERALIST_START_UTC] * 2
    assert [t.days for t in C.DEFAULT_TRACKS[:2]] == [pytest.approx(5 + 1 / 24)] * 2


def test_spec_config_carries_the_frozen_g_med():
    assert C.SpecConfig("s", "w", "p").g_med == {"generalist": 14.84}
