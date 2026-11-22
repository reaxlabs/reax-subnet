"""reax-answertest: ladder, criteria, admitted concurrency and report, against a fake runtime (no GPU)."""

from __future__ import annotations

import json
import math

import pytest

from reax_subnet import answertest as at
from reax_subnet import selfcheck
from reax_subnet.s1contract import parse_decision
from reax_subnet.s1validator import POOL_PARAMS
from tests.test_selfcheck import _fake_runtime


class FakeBackend:
    """In-process backend: answers = reference + a logit shift on option 0 that depends on the concurrency in flight."""

    def __init__(self, ref, noise_at):
        self.by_id = {it["id"]: it for it in ref["items"]}
        self.noise_at = noise_at  # callable(inflight) -> logit shift
        self.inflight = 0
        self.peak = 0
        self.fail_ids: set[str] = set()

    async def ready(self):
        return True

    async def close(self):
        pass

    async def decide_each(self, decisions, retries=0):
        import asyncio

        self.inflight += len(decisions)
        self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(0.001)
            shift = self.noise_at(self.inflight)
            out = []
            for d in decisions:
                if d.ident in self.fail_ids:
                    out.append(RuntimeError("boom"))
                    continue
                it = self.by_id[d.ident]
                logits = [math.log(max(p, 1e-9)) + (shift if i == 0 else 0.0) for i, p in enumerate(it["p"])]
                z = math.fsum(math.exp(x) for x in logits)
                out.append({"probabilities": {k: math.exp(x) / z for k, x in zip(d.question.keys, logits)}})
            return out
        finally:
            self.inflight -= len(decisions)


@pytest.fixture(scope="module")
def ref():
    r = selfcheck.load_reference("s1-fast")
    return dict(r, items=r["items"][:96])


async def _run(ref, backend, ladder=(1, 2, 4, 8), **kw):
    return await at.run("s1-fast", "http://unused", "t" * 32, ladder=ladder, duration_s=0.0, reference=ref,
                        backend=backend, **kw)


def test_helpers():
    assert at.max_dp([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.1)
    assert at.top1_margin([0.7, 0.3]) == pytest.approx(0.4)
    assert at.argmax([0.1, 0.9]) == 1


async def test_exact_engine_admits_every_level(ref):
    rep = await _run(ref, FakeBackend(ref, lambda n: 0.0))
    assert rep["schema"] == at.SCHEMA and rep["admitted_c"] == 8 and rep["pass"] is True
    for lv in rep["levels"]:
        assert lv["pass"] and lv["max_dp"] < 1e-6 and lv["top1_agreement"] == 1.0
        assert lv["items_covered"] == lv["items"] == 96
        assert lv["validator_stream"]["strikes"] == 0
        assert all(lv["criteria"].values())
    assert rep["levels"][0]["c"] == 1 and rep["model"] == "s1-fast"


async def test_load_dependent_drift_caps_the_admitted_concurrency(ref):
    # no drift up to 10 decisions in flight (c streams + the validator stream's round of 8), a logit shift of 0.5 beyond
    # (like an engine that is batch-sensitive)
    backend = FakeBackend(ref, lambda n: 0.0 if n <= 10 else 0.5)
    rep = await _run(ref, backend, ladder=(1, 2, 4, 8, 16), stop_after_fails=2)
    by_c = {lv["c"]: lv for lv in rep["levels"]}
    assert by_c[1]["pass"] and by_c[2]["pass"]
    assert not by_c[8]["pass"] and by_c[8]["criteria"]["max_dp"] is False
    assert not by_c[4]["pass"] and rep["admitted_c"] == 2
    assert 16 not in by_c  # ladder stopped after two failing levels in a row


async def test_small_drift_passes_dp_bar_but_big_drift_does_not(ref):
    ok = await _run(ref, FakeBackend(ref, lambda n: 0.005), ladder=(2,))
    assert ok["levels"][0]["pass"] and 0 < ok["levels"][0]["max_dp"] <= 0.02
    bad = await _run(ref, FakeBackend(ref, lambda n: 0.3), ladder=(2,))
    assert not bad["levels"][0]["pass"] and bad["admitted_c"] == 0 and bad["pass"] is False


async def test_failed_requests_fail_the_level(ref):
    be = FakeBackend(ref, lambda n: 0.0)
    be.fail_ids = {ref["items"][3]["id"]}
    rep = await _run(ref, be, ladder=(2,))
    lv = rep["levels"][0]
    assert lv["failed"] >= 1 and lv["criteria"]["complete"] is False and lv["criteria"]["full_pass"] is False
    assert rep["admitted_c"] == 0


async def test_top1_flip_on_clear_item_fails_but_near_tie_is_not_counted(ref):
    th = at.Thresholds()
    # a huge shift flips the clear items
    rep = await _run(ref, FakeBackend(ref, lambda n: 6.0), ladder=(1,))
    lv = rep["levels"][0]
    assert lv["top1_flips"] > 0 and lv["criteria"]["top1"] is False
    assert lv["top1_items"] + lv["near_tie_items"] >= 96 and th.tie_margin == 0.02


def test_admitted_concurrency_is_contiguous_prefix():
    lv = lambda c, ok: {"c": c, "pass": ok}
    assert at.admitted_concurrency([lv(1, True), lv(2, True), lv(4, False), lv(8, True)]) == \
        (2, ["c=8 passes above a failing lower level; not admitted"])
    assert at.admitted_concurrency([lv(1, False)]) == (0, [])


async def test_two_cold_starts_take_the_worst_case(ref):
    good = await _run(ref, FakeBackend(ref, lambda n: 0.0), ladder=(1, 2))
    worse = await _run(ref, FakeBackend(ref, lambda n: 0.0 if n < 10 else 0.3), ladder=(1, 2))
    comb = at.combine([good, worse])
    assert comb["cold_starts"] == 2 and comb["admitted_c"] == 1
    assert next(l for l in comb["levels"] if l["c"] == 2)["max_dp"] >= worse["levels"][1]["max_dp"] > 0.02
    other = dict(good, model="s1-pro")
    with pytest.raises(ValueError):
        at.combine([good, other])


async def test_cold_starts_need_a_restart_command(ref):
    with pytest.raises(ValueError):
        await _run(ref, FakeBackend(ref, lambda n: 0.0), cold_starts=2)


async def test_report_is_json_with_identity_fields(ref):
    rep = await _run(ref, FakeBackend(ref, lambda n: 0.0), ladder=(1,), engine="vllm", image_digest="sha256:abc",
                     gpu_name="RTX PRO 6000", driver="580.1")
    doc = json.loads(json.dumps(rep))
    for k in ("model", "manifest_id", "engine_image_digest", "gpu_name", "timestamp", "tool_version", "admitted_c",
              "thresholds", "levels"):
        assert k in doc
    assert doc["engine_image_digest"] == "sha256:abc" and doc["gpu_name"] == "RTX PRO 6000"
    assert set(doc["levels"][0]["criteria"]) == {"complete", "top1", "max_dp", "mean_distance", "no_strikes", "full_pass"}


@pytest.mark.parametrize("noise,expect", [(0.0, True), (0.6, False)])
async def test_against_http_fake_runtime(noise, expect):
    full = selfcheck.load_reference("s1-fast")
    ref = dict(full, items=full["items"][:40])
    runner, url = await _fake_runtime(full, noise)
    try:
        rep = await at.run("s1-fast", url, "t" * 32, ladder=(1, 4), duration_s=0.0, reference=ref)
    finally:
        await runner.cleanup()
    assert rep["pass"] is expect
    assert all(lv["answered"] >= 40 for lv in rep["levels"])


def test_cli_cannot_run_without_token(tmp_path, capsys):
    assert at.main(["--url", "http://127.0.0.1:9", "--token-file", str(tmp_path / "missing"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["pass"] is False


def test_print_report_runs(ref, capsys):
    import asyncio

    rep = asyncio.run(_run(ref, FakeBackend(ref, lambda n: 0.0), ladder=(1, 2)))
    at.print_report(rep)
    out = capsys.readouterr().out
    assert "ADMITTED concurrency: 2" in out and "PASS" in out
