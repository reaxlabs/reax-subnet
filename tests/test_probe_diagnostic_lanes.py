"""Lane failures remain visible across the real in-process and forked probe paths."""
import logging

import pytest

from test_probe_responses import _OldLimitsMiner
from test_tiers import Machine, _probe_arena, probe_v2, _release_floors_off  # noqa: F401 (fixtures)


@pytest.mark.parametrize("shards", ["1", "2"])
async def test_lane_failures_reach_slot_log(shards, caplog, monkeypatch, probe_v2,
                                           make_wallet, fake_chain_cls, tmp_path):
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_SHARDS", shards)
    caplog.set_level(logging.INFO)
    groups = [(Machine(160.0), [{}]), (Machine(160.0), [{"miner_cls": _OldLimitsMiner}])]
    _val, by_group, _summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    uid = by_group[1][0]
    lines = [r.getMessage() for r in caplog.records if "probe diagnostic " in r.getMessage()
             and r.getMessage().startswith(f"uid {uid} [")]
    assert len(lines) == 1
    assert "lane_errors={'http_429':" in lines[0]
    assert "folded=True" in lines[0]
