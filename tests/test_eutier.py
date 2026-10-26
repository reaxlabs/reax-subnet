"""EU tier (eutier.py): registry, Scheme A x1.5, contract tier pay rule, receipts, and the no-op guarantee."""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from reax_subnet import eutier
from reax_subnet.eutier import (ContractLedger, DailyMarket, FileMarket, FileVerification, Quote, carve_out,
                                fresh_price, parse_registry, scheme_a_multipliers)

DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(DATA))
import eutier_recorded  # noqa: E402

GOLDEN = {int(k): v for k, v in json.loads((DATA / "eutier_golden_b510d79.json").read_text()).items()}
RNG = random.Random(4)
HK = [eutier_recorded.fake_hotkey(RNG) for _ in range(8)]
DAY = date(2026, 10, 24)
T0 = eutier.day_start(DAY)


def _op(oid, hotkeys, **kw):
    d = {"id": oid, "contract": "C-2026-001", "dpa": "DPA-2026-001", "datacenter": "Hetzner FSN1, Falkenstein, DE",
         "asn": "AS24940", "hotkeys": hotkeys}
    d.update(kw)
    return d


def _reg(scheme_a=(), contract=(), **ct):
    tier = {"version": 1, "eur_per_gpu_hour": 1.0, "gpu_class": "rtx-pro-6000-96gb", "cap_pct_of_inference_share": 5.0,
            "review_due": "2026-11-22", "operators": list(contract)}
    tier.update(ct)
    return {"version": 1, "revision": "t1", "scheme_a": {"operators": list(scheme_a)}, "eu_contract_tier": tier}


class _Verified:
    def __init__(self, hotkeys=(), ok=True):
        self.hotkeys, self.ok = set(hotkeys), ok

    def verified(self, hotkey, now):
        return self.ok and hotkey in self.hotkeys


class _Market:
    def __init__(self, days):
        self.days = days

    def day(self, d):
        return self.days.get(d)


def _market(d, alpha_tao=0.03, eur=(240.0, 230.0), share=1700.0, at=None):
    at = at or eutier.day_start(d) + timedelta(minutes=5)
    return DailyMarket(alpha_tao=Quote(alpha_tao, at), tao_eur={f"s{i}": Quote(v, at) for i, v in enumerate(eur)},
                       inference_share_alpha=share)


def _fill(ledger, hotkey, d, hours, ok=True, failed=False, verified=True, start=0):
    for h in range(start, start + hours):
        ledger.record(eutier.day_start(d) + timedelta(hours=h, minutes=10), {hotkey: (ok, failed)}, {hotkey: verified})


# ----------------------------------------------------------------- registry
def test_shipped_registry_is_valid_and_lists_nobody():
    reg = eutier.load_registry(eutier.REGISTRY_FILE)
    assert reg is not eutier.EMPTY and reg.empty and not reg.scheme_a and not reg.contract
    assert reg.params.eur_per_gpu_hour == 1.0 and reg.params.cap_pct == 5.0
    assert reg.params.gpu_class == "rtx-pro-6000-96gb" and reg.params.review_due == "2026-11-22"


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(version=2),
    lambda d: d.update(extra=1),
    lambda d: d["eu_contract_tier"].update(cap_pct_of_inference_share=6.0),  # above the decided 5 % hard cap
    lambda d: d["eu_contract_tier"]["operators"].append(_op("b", [HK[0]], slots=1)),  # HK[0] also in scheme_a
    lambda d: d["scheme_a"]["operators"].append(_op("a", [HK[5]], cap_units=1)),  # operator id twice
    lambda d: d["scheme_a"]["operators"][0].update(hotkeys=["not-a-hotkey"]),
    lambda d: d["scheme_a"]["operators"][0].update(slots=3),  # contract field on a Scheme A operator
    lambda d: d["eu_contract_tier"].update(eur_per_gpu_hour=float("nan")),
])
def test_invalid_registries_are_rejected(mutate):
    d = _reg(scheme_a=[_op("a", [HK[0]], cap_units=10)])
    parse_registry(d)
    mutate(d)
    with pytest.raises(eutier.RegistryError):
        parse_registry(d)


def test_invalid_or_missing_file_loads_as_empty_and_logs(tmp_path, caplog):
    p = tmp_path / "r.json"
    p.write_text("{not json")
    with caplog.at_level(logging.ERROR, "reax.eutier"):
        assert eutier.load_registry(str(p)) is eutier.EMPTY
        assert eutier.load_registry(str(tmp_path / "missing.json")) is eutier.EMPTY
    assert sum("EU tier off" in r.message for r in caplog.records) == 2


@pytest.mark.parametrize("word", ["TBD", " pending ", "N/A", "none", "TODO", "-"])
def test_placeholder_words_count_as_missing(word):
    for field in ("contract", "dpa", "datacenter", "asn"):
        reg = parse_registry(_reg(scheme_a=[_op("a", [HK[0]], cap_units=10, **{field: word})],
                                  contract=[_op("c", [HK[1]], slots=8, **{field: word})]))
        assert reg.empty, (field, word)


def test_operator_without_contract_or_dpa_is_inactive():
    reg = parse_registry(_reg(
        scheme_a=[_op("a", [HK[0]], cap_units=10, contract="<signed contract id>"), _op("b", [HK[1]], cap_units=0)],
        contract=[_op("c", [HK[2]], slots=8, dpa=""), _op("d", [HK[3]], slots=0), _op("e", [], slots=8)]))
    assert reg.empty and not reg.active_scheme_a() and not reg.active_contract()
    reg = parse_registry(_reg(contract=[_op("c", [HK[2]], slots=8)], eur_per_gpu_hour=0.0))
    assert reg.empty  # no price per hour = no tier


def test_eutier_reloads_the_registry_when_the_file_changes(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_reg()))
    eu = eutier.EUTier(str(p), _Verified(), _Market({}), str(tmp_path / "s.json"), str(tmp_path / "rc"))
    assert eu.registry.empty
    p.write_text(json.dumps({**_reg(contract=[_op("c", [HK[2]], slots=8)]), "revision": "t2"}))
    assert not eu.refresh().empty and eu.contract_hotkeys() == {HK[2]}
    p.write_text("[]")  # invalid -> empty again, no exception
    assert eu.refresh().empty


def test_any_load_error_gives_the_empty_registry_and_is_retried(tmp_path, monkeypatch, caplog):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_reg(contract=[_op("c", [HK[2]], slots=8)])))
    eu = eutier.EUTier(str(p), _Verified(), _Market({}), str(tmp_path / "s.json"), str(tmp_path / "rc"))
    assert eu.contract_hotkeys() == {HK[2]}
    p.write_text(json.dumps({**_reg(contract=[_op("c", [HK[2]], slots=8)]), "revision": "t2-xx"}))

    def boom(doc):
        raise RecursionError("deep")

    monkeypatch.setattr(eutier, "parse_registry", boom)
    with caplog.at_level(logging.ERROR, "reax.eutier"):
        assert eu.refresh().empty  # not the previous registry
    assert any("RecursionError" in r.message for r in caplog.records)
    # A load that raises outright leaves the stamp unset, so the next round tries again.
    monkeypatch.setattr(eutier, "load_registry", boom)
    p.write_text(json.dumps({**_reg(contract=[_op("c", [HK[2]], slots=8)]), "revision": "t3-longer"}))
    stamp = eu._stamp
    with pytest.raises(RecursionError):
        eu.refresh()
    assert eu._stamp == stamp
    monkeypatch.undo()
    eu.refresh()
    assert eu.contract_hotkeys() == {HK[2]}


# ----------------------------------------------------------------- Scheme A
def test_scheme_a_multiplier_cap_and_fail_closed():
    reg = parse_registry(_reg(scheme_a=[_op("a", [HK[0], HK[1]], cap_units=20), _op("b", [HK[2]], cap_units=100)]))
    hk = {10: HK[0], 11: HK[1], 12: HK[2], 13: HK[3]}
    factors = {"s1-fast": {10: 10.0, 11: 30.0, 12: 5.0, 13: 5.0}, "s1-lite": {10: 2.0}}
    m = scheme_a_multipliers(reg, _Verified(HK[:3]), T0, factors, hk)
    # operator a: 40 units, cap 20 -> half of it earns x1.5 -> 1.25 each; b below its cap -> 1.5; HK[3] not listed;
    # s1-lite is not a Scheme A pool.
    assert m == {"s1-fast": {10: 1.25, 11: 1.25, 12: 1.5}}
    # Splitting operator a's 40 units over more hotkeys does not raise its bonus: sum(bonus x factor) stays 0.5 x 20.
    assert math.fsum((m["s1-fast"][s] - 1) * factors["s1-fast"][s] for s in (10, 11)) == pytest.approx(10.0)
    # Fail closed: an unverified hotkey falls back to x1.0, nobody verified -> nothing.
    # (the cap then covers the verified 30 units only: 1 + 0.5 x 20 / 30)
    assert scheme_a_multipliers(reg, _Verified(HK[1:3]), T0, factors, hk) == {"s1-fast": {11: 1 + 0.5 * 20 / 30, 12: 1.5}}
    assert scheme_a_multipliers(reg, eutier.NoVerification(), T0, factors, hk) == {}


def test_file_verification_needs_both_checks_and_freshness(tmp_path):
    p = tmp_path / "v.json"
    now = T0 + timedelta(hours=10)
    p.write_text(json.dumps({"version": 1, "hotkeys": {
        HK[0]: {"rtt_ok": True, "asn_ok": True, "checked_at": (now - timedelta(hours=1)).isoformat()},
        HK[1]: {"rtt_ok": True, "asn_ok": False, "checked_at": now.isoformat()},
        HK[2]: {"rtt_ok": True, "asn_ok": True, "checked_at": (now - timedelta(hours=7)).isoformat()},
        HK[3]: {"rtt_ok": "yes", "asn_ok": True, "checked_at": now.isoformat()}}}))
    v = FileVerification(str(p))
    assert [v.verified(h, now) for h in HK[:5]] == [True, False, False, False, False]
    assert not FileVerification(str(tmp_path / "none.json")).verified(HK[0], now)
    p.write_text("garbage")
    assert not FileVerification(str(p)).verified(HK[0], now)


# ----------------------------------------------------------------- price (pay rule 2)
def test_price_needs_two_fresh_sources():
    assert fresh_price(_market(DAY, 0.031057, (233.70, 233.70)), DAY) == pytest.approx(7.258, rel=1e-3)
    assert fresh_price(_market(DAY, 0.03, (240.0, 230.0)), DAY) == pytest.approx(0.03 * 235.0)
    assert fresh_price(_market(DAY, eur=(240.0,)), DAY) is None  # one source
    assert fresh_price(_market(DAY, at=T0 + timedelta(hours=2, minutes=1)), DAY) is None  # stale > 2 h
    assert fresh_price(None, DAY) is None


def test_file_market_reads_the_sandy_quote_file(tmp_path):
    p = tmp_path / "m.json"
    at = (T0 + timedelta(minutes=3)).isoformat()
    p.write_text(json.dumps({"version": 1, "days": {DAY.isoformat(): {
        "alpha_tao": {"value": 0.03, "at": at},
        "tao_eur": {"coingecko": {"value": 240.0, "at": at}, "kraken": {"value": 230.0, "at": at}},
        "inference_share_alpha": 1681.0}}}))
    m = FileMarket(str(p)).day(DAY)
    assert fresh_price(m, DAY) == pytest.approx(7.05) and m.inference_share_alpha == 1681.0
    assert FileMarket(str(p)).day(DAY + timedelta(days=1)) is None
    assert FileMarket(str(tmp_path / "none.json")).day(DAY) is None


# ----------------------------------------------------------------- contract tier (pay rule 1, 3-6)
def _ledger(tmp_path):
    return ContractLedger(str(tmp_path / "ledger.json"), str(tmp_path / "receipts"))


def _settle(ledger, reg, market, d=DAY):
    return ledger.settle(reg, market, eutier.day_start(d) + timedelta(days=1, minutes=1))


def test_slot_cap_and_gpu_hours(tmp_path):
    reg = parse_registry(_reg(contract=[_op("tng", [{"hotkey": HK[0], "gpus": 2}, HK[1]], slots=2)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 24)
    _fill(led, HK[1], DAY, 24)
    r = _settle(led, reg, _Market({DAY: _market(DAY)}))
    row = r["operators"][0]
    assert row["verified_gpu_hours"] == 72 and row["H_o"] == 48  # min(slots x 24, verified GPU-hours)
    assert row["A_o"] == pytest.approx(48 * 1.0 / (0.03 * 235.0)) and r["price_source"] == "fresh"
    assert r["cap_cut_factor"] == 1.0 and row["cap_cut_alpha"] == 0.0


def test_failed_correctness_voids_the_hour_and_unverified_or_unanswered_hours_do_not_count(tmp_path):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0]], slots=8)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 10)                                   # 10 good hours
    led.record(T0 + timedelta(hours=10, minutes=1), {HK[0]: (True, False)}, {HK[0]: True})
    led.record(T0 + timedelta(hours=10, minutes=30), {HK[0]: (True, True)}, {HK[0]: True})  # a strike voids hour 10
    _fill(led, HK[0], DAY, 2, verified=False, start=11)          # EU verification failed: not paid
    _fill(led, HK[0], DAY, 2, ok=False, start=13)                # no answer: not paid
    led.record(T0 + timedelta(hours=15, minutes=1), {HK[0]: (True, False)}, {HK[0]: True})
    led.record(T0 + timedelta(hours=15, minutes=2), {HK[0]: (True, False)}, {HK[0]: False})  # one failed check voids
    assert led.passed_hours(HK[0], DAY) == 10
    assert _settle(led, reg, _Market({DAY: _market(DAY)}))["operators"][0]["H_o"] == 10


def test_three_failed_hours_suspend_until_a_clean_hour(tmp_path):
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 2)                          # hours 0-1 paid
    _fill(led, HK[0], DAY, 3, failed=True, start=2)    # hours 2-4 failed -> suspended
    _fill(led, HK[0], DAY, 1, ok=False, start=5)       # still suspended (not answered: no clean re-probe)
    _fill(led, HK[0], DAY, 1, start=6)                 # the clean re-probe hour lifts it, unpaid
    _fill(led, HK[0], DAY, 3, start=7)                 # hours 7-9 paid
    assert led.passed_hours(HK[0], DAY) == 5
    # Two failed hours only void themselves.
    led2 = _ledger(tmp_path / "b")
    _fill(led2, HK[0], DAY, 2, failed=True)
    _fill(led2, HK[0], DAY, 4, start=2)
    assert led2.passed_hours(HK[0], DAY) == 4


def test_hour_needs_90_percent_of_queried_rounds_answered(tmp_path):
    led = _ledger(tmp_path)

    def rounds(h, answered, queried):
        for i in range(queried):
            led.record(T0 + timedelta(hours=h, minutes=i), {HK[0]: (i < answered, False)}, {HK[0]: True})

    rounds(0, 9, 10)   # 90 % -> passes
    rounds(1, 8, 10)   # 80 % -> does not
    rounds(2, 1, 20)   # one answer in an hour of queries -> does not
    rounds(3, 1, 1)    # 1 of 1 -> passes
    assert led.passed_hours(HK[0], DAY) == 2
    assert led.passed_hours(HK[0] + "x", DAY) == 0  # never queried = not passed


def test_old_ledger_state_without_counts_still_loads(tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps({"version": 1, "last_price": None, "settled": {}, "hours": {
        HK[0]: {eutier._hour_key(T0): {"a": True, "f": False, "v": True},
                eutier._hour_key(T0 + timedelta(hours=1)): {"a": False, "f": False, "v": True}}}}))
    assert _ledger(tmp_path).passed_hours(HK[0], DAY) == 1


def test_suspension_carries_over_midnight(tmp_path):
    led = _ledger(tmp_path)
    prev = DAY - timedelta(days=1)
    _fill(led, HK[0], prev, 3, failed=True, start=21)  # 21-23 the day before
    _fill(led, HK[0], DAY, 1, ok=False)                # 00: not answered, still suspended
    _fill(led, HK[0], DAY, 3, start=1)                 # 01 lifts (unpaid), 02-03 paid
    assert led.passed_hours(HK[0], DAY) == 2


def test_suspension_survives_a_restart_late_in_the_next_day(tmp_path):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0]], slots=8)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY - timedelta(days=2), 3, failed=True, start=1)  # suspended from D-2 01:00
    _fill(led, HK[0], DAY, 10)                                          # 00 lifts (unpaid), 01-09 paid
    late = eutier.day_start(DAY) + timedelta(days=1, hours=22)
    led.record(late, {HK[0]: (True, False)}, {HK[0]: True})            # prunes with KEEP_HOURS
    led.save()
    led2 = _ledger(tmp_path)                                            # restart
    r = led2.settle(reg, _Market({DAY: _market(DAY)}), late + timedelta(minutes=30))
    assert r["operators"][0]["H_o"] == 9 and eutier.KEEP_HOURS == 120


def test_stale_price_reuses_previous_and_no_price_ever_pays_nothing(tmp_path, caplog):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0]], slots=8)]))
    led = _ledger(tmp_path)
    prev = DAY - timedelta(days=1)
    _fill(led, HK[0], prev, 4)
    _fill(led, HK[0], DAY, 4)
    with caplog.at_level(logging.WARNING, "reax.eutier"):
        r1 = _settle(led, reg, _Market({prev: _market(prev, 0.03, (200.0, 200.0))}), prev)
        stale = _market(DAY, 0.06, (300.0, 300.0), at=T0 + timedelta(hours=3))
        r2 = _settle(led, reg, _Market({DAY: stale}))
    assert r1["price_eur_per_alpha"] == pytest.approx(6.0) and r2["price_eur_per_alpha"] == pytest.approx(6.0)
    assert r2["price_source"] == "previous" and any("previous P" in r.message for r in caplog.records)
    led3 = _ledger(tmp_path / "c")
    _fill(led3, HK[0], DAY, 4)
    r3 = _settle(led3, reg, _Market({}))
    assert r3["price_source"] == "none" and r3["operators"][0]["A_o"] == 0.0
    assert led3.pay_fractions(reg, _Market({}), T0 + timedelta(days=1, hours=1)) == ({}, set())


def test_cap_cuts_all_operators_pro_rata(tmp_path):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0]], slots=8), _op("teemo", [HK[1]], slots=8)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 24)
    _fill(led, HK[1], DAY, 12)
    # P = 0.01 x 100 = EUR 1/alpha: 36 alpha owed; cap = 5 % of 400 = 20 alpha.
    r = _settle(led, reg, _Market({DAY: _market(DAY, 0.01, (100.0, 100.0), share=400.0)}))
    a = {row["operator"]: row for row in r["operators"]}
    assert r["owed_alpha"] == pytest.approx(36.0) and r["cap_alpha"] == pytest.approx(20.0)
    assert a["tng"]["A_o"] + a["teemo"]["A_o"] == pytest.approx(20.0)
    assert a["tng"]["A_o"] / a["teemo"]["A_o"] == pytest.approx(2.0)
    assert a["tng"]["cap_cut_alpha"] == pytest.approx(24.0 - 20.0 * 24 / 36)
    pay, excluded = led.pay_fractions(reg, _Market({}), T0 + timedelta(days=1, hours=2))
    assert math.fsum(pay.values()) == pytest.approx(0.05) and excluded == {HK[0], HK[1]}
    # Below the cap nothing is cut.
    led2 = _ledger(tmp_path / "b")
    _fill(led2, HK[0], DAY, 1)
    r2 = _settle(led2, reg, _Market({DAY: _market(DAY, 0.01, (100.0, 100.0), share=400.0)}))
    assert r2["cap_cut_factor"] == 1.0 and r2["operators"][0]["fraction_of_inference_share"] == pytest.approx(1 / 400)


def test_no_inference_share_estimate_pays_nothing(tmp_path):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0]], slots=8)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 5)
    r = _settle(led, reg, _Market({DAY: _market(DAY, share=None)}))
    assert r["operators"][0]["A_o"] == 0.0 and r["operators"][0]["A_o_uncapped"] > 0


def test_missing_market_data_is_retried_then_settled_as_no_price_after_the_cutoff(tmp_path, caplog):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0]], slots=8)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 5)
    key = DAY.isoformat()
    t = eutier.day_start(DAY) + timedelta(days=1, minutes=1)
    r = led.settle(reg, _Market({}), t)  # entry missing: receipt returned, nothing cached or written
    assert r["operators"][0]["A_o"] == 0.0 and key not in led.settled
    assert not (tmp_path / "receipts" / f"{key}.json").exists()
    r = led.settle(reg, _Market({DAY: _market(DAY, share=None)}), t + timedelta(hours=5))  # no share: same
    assert key not in led.settled
    r = led.settle(reg, _Market({DAY: _market(DAY)}), t + timedelta(hours=6))  # data arrived: settled and cached
    assert r["price_source"] == "fresh" and r["operators"][0]["A_o"] > 0 and key in led.settled
    assert (tmp_path / "receipts" / f"{key}.json").exists()
    assert led.settle(reg, _Market({}), t + timedelta(hours=7)) is led.settled[key]
    # Still missing at 23:00Z of the next day: settled as "no price" (pays nothing), cached, logged at ERROR.
    led2 = _ledger(tmp_path / "b")
    _fill(led2, HK[0], DAY, 5)
    led2.last_price = 7.0  # even a previous price is not reused then
    with caplog.at_level(logging.ERROR, "reax.eutier"):
        r2 = led2.settle(reg, _Market({}), eutier.day_start(DAY) + timedelta(days=1, hours=23))
    assert r2["price_source"] == "none" and r2["operators"][0]["A_o"] == 0.0 and key in led2.settled
    assert any("still missing" in x.message and x.levelno == logging.ERROR for x in caplog.records)
    assert led2.pay_fractions(reg, _Market({}), eutier.day_start(DAY) + timedelta(days=1, hours=23)) == ({}, set())


def test_inactive_tier_settles_and_caches_without_data(tmp_path):
    led = _ledger(tmp_path)
    t = eutier.day_start(DAY) + timedelta(days=1, minutes=1)
    led.settle(parse_registry(_reg()), _Market({}), t)
    assert DAY.isoformat() in led.settled and not (tmp_path / "receipts").exists()


def test_receipt_is_public_and_written_once(tmp_path):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0], HK[1]], slots=8)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 6)
    _settle(led, reg, _Market({DAY: _market(DAY)}))
    path = tmp_path / "receipts" / f"{DAY.isoformat()}.json"
    doc = json.loads(path.read_text())
    assert doc["day"] == DAY.isoformat() and doc["operators"][0]["operator"] == "tng"
    assert {"H_o", "A_o", "cap_cut_alpha"} <= set(doc["operators"][0]) and "price_eur_per_alpha" in doc
    text = path.read_text()
    for secret in ("C-2026-001", "DPA-2026-001", "127.0.0.1", "AS24940", "Falkenstein"):
        assert secret not in text
    # A second call returns the stored settlement (other market data no longer matters).
    assert _settle(led, reg, _Market({}))["price_source"] == "fresh"
    # The ledger survives a restart.
    led2 = _ledger(tmp_path)
    led.save()
    led2 = _ledger(tmp_path)
    assert led2.passed_hours(HK[0], DAY) == 6 and DAY.isoformat() in led2.settled


def test_pay_is_split_by_hours_and_stops_for_removed_or_inactive_operators(tmp_path):
    reg = parse_registry(_reg(contract=[_op("tng", [HK[0], HK[1]], slots=8)]))
    led = _ledger(tmp_path)
    _fill(led, HK[0], DAY, 6)
    _fill(led, HK[1], DAY, 3)
    now = T0 + timedelta(days=1, hours=1)
    pay, excluded = led.pay_fractions(reg, _Market({DAY: _market(DAY)}), now)
    assert pay[HK[0]] == pytest.approx(2 * pay[HK[1]]) and excluded == {HK[0], HK[1]}
    # The operator loses its DPA id in a new revision: inactive -> no pay, normal rule again.
    reg2 = parse_registry({**_reg(contract=[_op("tng", [HK[0], HK[1]], slots=8, dpa="")]), "revision": "t2"})
    assert led.pay_fractions(reg2, _Market({}), now) == ({}, set())


# ----------------------------------------------------------------- carve-out (pay rule 5)
def test_carve_out_takes_the_fractions_from_the_miner_part_only():
    base = {0: 0.1, 2: 0.45, 3: 0.45}  # owner burn 0.1
    out = carve_out(base, {7: 0.04}, owner_uid=0)
    assert out[0] == 0.1 and out[7] == pytest.approx(0.04 * 0.9)
    assert out[2] == pytest.approx(0.45 * 0.96) and math.fsum(out.values()) == pytest.approx(1.0)
    assert carve_out(base, {}, 0) == base and carve_out(base, {0: 0.04}, 0) == base
    assert carve_out({0: 1.0}, {7: 0.04}, 0) == {0: 1.0}  # no other miner: nothing carved


# ----------------------------------------------------------------- the validator: no-op and live paths
def _write_registry(tmp_path, doc):
    p = tmp_path / "registry.json"
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return str(p)


def _weights(tmp_path, make_wallet, monkeypatch, env=None, tweak=None):
    return eutier_recorded.recorded_weights(str(tmp_path), make_wallet("//Validator"), monkeypatch,
                                            {} if env is None else env, tweak)


def test_default_registry_weights_are_byte_identical_to_rc11_post4(tmp_path, make_wallet, monkeypatch):
    monkeypatch.delenv("REAX_EU_REGISTRY_FILE", raising=False)
    w = _weights(tmp_path, make_wallet, monkeypatch)
    assert w == GOLDEN and [repr(v) for v in w.values()] == [repr(v) for v in GOLDEN.values()]
    assert not (tmp_path / "eu-tier-state.json").exists() and not (tmp_path / "eu-tier-receipts").exists()


def _hotkeys_of(neurons, miners, pool):
    from reax_subnet.elements import slot_key, slot_uid
    by_uid = {n.uid: n.hotkey for n in neurons}
    return [by_uid[slot_uid(s)] for s, m in sorted(miners.items())
            if m.role.startswith(pool + ":") and m.ema > 0.5 and (m.capacity or 0) > 0 and slot_key(slot_uid(s), pool) == s]


@pytest.mark.parametrize("case", ["placeholders", "invalid", "missing_file", "scheme_a_unverified",
                                  "contract_no_price", "contract_no_dpa"])
def test_inactive_configurations_are_byte_identical(case, tmp_path, make_wallet, monkeypatch):
    env = {"REAX_EU_TIER_STATE_PATH": str(tmp_path / "ledger.json"),
           "REAX_EU_TIER_RECEIPTS_DIR": str(tmp_path / "receipts")}

    def tweak(neurons, miners):
        hks = _hotkeys_of(neurons, miners, "s1-fast")
        if case == "placeholders":
            doc = _reg(scheme_a=[_op("a", hks[:2], cap_units=10, contract="<signed contract id>")],
                       contract=[_op("c", hks[2:4], slots=8, dpa="<AVV/DPA id>")])
        elif case == "invalid":
            doc = json.dumps(_reg(scheme_a=[_op("a", hks[:2], cap_units=10)],
                                  contract=[_op("c", hks[1:3], slots=8)]))  # hks[1] twice
        elif case == "missing_file":
            env["REAX_EU_REGISTRY_FILE"] = str(tmp_path / "nope.json")
            return
        elif case == "scheme_a_unverified":
            doc = _reg(scheme_a=[_op("a", hks[:3], cap_units=100)])
            env["REAX_EU_VERIFICATION_FILE"] = str(tmp_path / "no-verification.json")
        else:
            doc = _reg(contract=[_op("c", hks[:2], slots=8, **({"dpa": ""} if case == "contract_no_dpa" else {}))])
            led = ContractLedger(env["REAX_EU_TIER_STATE_PATH"], env["REAX_EU_TIER_RECEIPTS_DIR"])
            for h in hks[:2]:
                _fill(led, h, DAY - timedelta(days=1), 20)  # verified hours yesterday, but no market file / no DPA
            led.save()
        env["REAX_EU_REGISTRY_FILE"] = _write_registry(tmp_path, doc)

    w = _weights(tmp_path, make_wallet, monkeypatch, env, tweak)
    assert w == GOLDEN and [repr(v) for v in w.values()] == [repr(v) for v in GOLDEN.values()]


def test_scheme_a_verified_raises_the_operator_weight(tmp_path, make_wallet, monkeypatch):
    seen = {}

    def tweak(neurons, miners):
        hks = _hotkeys_of(neurons, miners, "s1-fast")
        seen["hk"] = hks[0]
        seen["uid"] = next(n.uid for n in neurons if n.hotkey == hks[0])
        env["REAX_EU_REGISTRY_FILE"] = _write_registry(tmp_path, _reg(scheme_a=[_op("a", [hks[0]], cap_units=1000)]))
        v = tmp_path / "v.json"
        v.write_text(json.dumps({"version": 1, "hotkeys": {hks[0]: {
            "rtt_ok": True, "asn_ok": True, "checked_at": eutier_recorded.NOW.isoformat()}}}))
        env["REAX_EU_VERIFICATION_FILE"] = str(v)

    env: dict[str, str] = {}
    w = _weights(tmp_path, make_wallet, monkeypatch, env, tweak)
    uid = seen["uid"]
    assert w[uid] > GOLDEN[uid] * 1.3  # x1.5 in-pool score, renormalised over the pool
    assert w[0] == GOLDEN[0] and math.fsum(w.values()) == pytest.approx(1.0)
    others = [u for u in GOLDEN if u not in (0, uid)]
    assert all(w[u] <= GOLDEN[u] for u in others)


def test_contract_tier_pays_carve_out_and_no_double_pay(tmp_path, make_wallet, monkeypatch):
    env = {"REAX_EU_TIER_STATE_PATH": str(tmp_path / "ledger.json"),
           "REAX_EU_TIER_RECEIPTS_DIR": str(tmp_path / "receipts"), "REAX_EU_MARKET_FILE": str(tmp_path / "m.json")}
    seen = {}
    yday = eutier_recorded.NOW.date() - timedelta(days=1)

    def tweak(neurons, miners):
        hks = _hotkeys_of(neurons, miners, "s1-fast")[:2]
        seen["uids"] = [next(n.uid for n in neurons if n.hotkey == h) for h in hks]
        env["REAX_EU_REGISTRY_FILE"] = _write_registry(tmp_path, _reg(contract=[_op("tng", hks, slots=8)]))
        led = ContractLedger(env["REAX_EU_TIER_STATE_PATH"], env["REAX_EU_TIER_RECEIPTS_DIR"])
        _fill(led, hks[0], yday, 24)
        _fill(led, hks[1], yday, 12)
        led.save()
        at = (eutier.day_start(yday) + timedelta(minutes=2)).isoformat()
        (tmp_path / "m.json").write_text(json.dumps({"version": 1, "days": {yday.isoformat(): {
            "alpha_tao": {"value": 0.031057, "at": at},
            "tao_eur": {"coingecko": {"value": 233.7, "at": at}, "kraken": {"value": 233.7, "at": at}},
            "inference_share_alpha": 1681.0}}}))

    w = _weights(tmp_path, make_wallet, monkeypatch, env, tweak)
    a, b = seen["uids"]
    p = 0.031057 * 233.7
    fa, fb = 24 / p / 1681.0, 12 / p / 1681.0
    miner = 1.0 - GOLDEN[0]
    assert w[0] == GOLDEN[0]
    # The contracted hotkeys get exactly A_o / inference share of the miner part (no global inference pay on top).
    assert w[a] == pytest.approx(fa * miner) and w[b] == pytest.approx(fb * miner)
    # Everyone else keeps their relative weights, scaled by 1 - sum of the fractions (their slots' pool budget is
    # shared without the contracted hotkeys).
    rest = [u for u in w if u not in (0, a, b)]
    assert math.fsum(w[u] for u in rest) == pytest.approx(miner * (1 - fa - fb))
    assert math.fsum(w.values()) == pytest.approx(1.0)
    receipt = json.loads((tmp_path / "receipts" / f"{yday.isoformat()}.json").read_text())
    assert receipt["operators"][0]["H_o"] == 36 and receipt["cap_cut_factor"] == 1.0


def test_round_outcomes_feed_the_ledger(tmp_path, make_wallet, fake_chain_cls, monkeypatch):
    from reax_subnet import s1validator
    from reax_subnet.config import load_validator
    from reax_subnet.elements import slot_key
    from reax_subnet.s1backends import S1HashBackend
    from reax_subnet.state import MinerState

    env = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
           "REAX_FEATURE_TIERED_POOLS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json")}
    monkeypatch.setenv("REAX_EU_REGISTRY_FILE",
                       _write_registry(tmp_path, _reg(contract=[_op("tng", [HK[0], HK[1], HK[2]], slots=8)])))
    val = s1validator.S1Validator(load_validator(env), make_wallet("//Validator"), fake_chain_cls(neurons=[]),
                                  {m: S1HashBackend(m) for m in s1validator.POOL_PARAMS})
    assert val.eu is not None and val.eu.contract_hotkeys() == {HK[0], HK[1], HK[2]}
    val.state.miners = {2: MinerState(hotkey=HK[0]), slot_key(2, "s1-pro"): MinerState(hotkey=HK[0]),
                        3: MinerState(hotkey=HK[1]), 4: MinerState(hotkey=HK[3])}
    val.eu.verifier = _Verified({HK[0], HK[1]})
    summary = {"scores": {2: {"score": 0.9, "strike": False, "error": None},
                          slot_key(2, "s1-pro"): {"score": 0.8, "strike": False, "error": None},
                          3: {"score": 0.0, "strike": False, "error": None},
                          4: {"score": 0.9, "strike": False, "error": None}},
               "capacity": {3: {"strike": True}}}
    val._eu_record(summary)
    hours = val.eu.ledger.hours
    assert set(hours) == {HK[0], HK[1], HK[2]}  # only tier hotkeys; HK[2] did not take part (not answered)
    (b0,), (b1,), (b2,) = (list(hours[h].values()) for h in HK[:3])
    assert b0 == {"q": 1, "k": 1, "f": False, "v": True}
    assert b1 == {"q": 1, "k": 0, "f": True, "v": True}  # the capacity probe's strike voids the hour
    assert b2 == {"q": 1, "k": 0, "f": False, "v": False}
    assert (tmp_path / "eu-tier-state.json").exists()
    # A broken registry turns recording off without raising.
    Path(val.eu.registry_path).write_text("{")
    val._eu_record(summary)
    assert val.eu.registry.empty


def test_a_late_capacity_strike_goes_to_the_hotkey_that_was_checked(tmp_path, make_wallet, fake_chain_cls, monkeypatch):
    from reax_subnet import s1validator
    from reax_subnet.config import load_validator
    from reax_subnet.s1backends import S1HashBackend
    from reax_subnet.state import MinerState

    env = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
           "REAX_FEATURE_TIERED_POOLS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json")}
    monkeypatch.setenv("REAX_EU_REGISTRY_FILE",
                       _write_registry(tmp_path, _reg(contract=[_op("tng", [HK[0], HK[1], HK[2]], slots=8)])))
    val = s1validator.S1Validator(load_validator(env), make_wallet("//Validator"), fake_chain_cls(neurons=[]),
                                  {m: S1HashBackend(m) for m in s1validator.POOL_PARAMS})
    val.eu.verifier = _Verified({HK[0], HK[1]})
    # Slot 3 was HK[0] at the probe window and was re-registered to HK[1] before the round that records the report.
    val.state.miners = {3: MinerState(hotkey=HK[1])}
    val._eu_record({"scores": {}, "capacity": {3: {"strike": True, "hotkey": HK[0]}}})
    hours = val.eu.ledger.hours
    assert [b["f"] for b in hours[HK[0]].values()] == [True]
    assert [b["f"] for b in hours[HK[1]].values()] == [False]  # the new owner is not struck


def test_quote_tool_writes_what_the_validator_reads(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("eu_tier_quote", DATA.parent.parent / "tools" / "eu_tier_quote.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    f = tmp_path / "m.json"
    at = (T0 + timedelta(minutes=4)).isoformat()
    assert tool.main(["--file", str(f), "--day", DAY.isoformat(), "--alpha-tao", "0.031057", "--tao-eur",
                      "coingecko=233.7", "--tao-eur", "kraken=233.9", "--inference-share-alpha", "1681", "--at", at]) == 0
    m = FileMarket(str(f)).day(DAY)
    assert fresh_price(m, DAY) == pytest.approx(0.031057 * 233.8) and m.inference_share_alpha == 1681.0
