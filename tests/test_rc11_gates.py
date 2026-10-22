"""rc11 activates the rules-v4 gates at one instant (T1 = 2026-10-23T13:00:00Z): off one second before, on at T1."""
from datetime import datetime, timedelta, timezone

from reax_subnet import competition_split, engcomp, tiers
from reax_subnet.spec import machine, sampling

T1 = datetime(2026, 10, 23, 13, 0, tzinfo=timezone.utc)
BEFORE = T1 - timedelta(seconds=1)


def test_five_gates_equal_t1():
    assert machine.CONFIRM_BEFORE_CLOSE_START_UTC == T1
    assert sampling.DRAWS_V4_START_UTC == T1
    assert tiers.COLDKEY_DR_START_UTC == T1
    assert engcomp.ENGINE_COPYLINE_START_UTC == T1
    assert competition_split.OPENING_PLAYOFF_SLOTS_START_UTC == T1


def test_gates_off_one_second_before_and_on_at_t1():
    assert not machine.confirm_before_close_active(BEFORE.timestamp())
    assert machine.confirm_before_close_active(T1.timestamp())
    assert not sampling.v4_active(BEFORE) and not sampling.v4_active(BEFORE.timestamp())
    assert sampling.v4_active(T1) and sampling.v4_active(T1.timestamp())
    assert not tiers.coldkey_dr_active(BEFORE) and tiers.coldkey_dr_active(T1)
    assert not engcomp.copyline_active(BEFORE) and engcomp.copyline_active(T1)
