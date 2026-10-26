from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from reax_subnet.chain import Neuron, Snapshot


class DevWallet:
    """Wallet-shaped object holding throwaway dev keypairs (tests only)."""

    def __init__(self, uri: str):
        Keypair = pytest.importorskip("bittensor.keyfiles", reason="wallet tests require bittensor.keyfiles").Keypair
        self.hotkey = Keypair.create_from_uri(uri)
        self.coldkey = Keypair.create_from_uri(uri + "/cold")
        self.coldkeypub = self.coldkey


@pytest.fixture(autouse=True)
def _engine_opening_off(request, monkeypatch):
    """The release sets an engine opening block (competition_split.ENGINE_TRACK_START_BLOCK); tests use small block
    numbers, so the rule is off unless a test enables it (tests/test_engfair.py)."""
    from reax_subnet import competition_split as cs

    if request.node.get_closest_marker("real_opening") is None:
        monkeypatch.setattr(cs, "ENGINE_TRACK_START_BLOCK", None)
    from reax_subnet import engcomp

    engcomp.set_start_block_hash(None)  # module-global state must not leak between tests


@pytest.fixture
def make_wallet():
    return DevWallet


@dataclass
class FakeChain:
    neurons: list[Neuron]
    owner_hotkey: str | None = None
    block: int = 1000
    rate_limit: int = 0
    since_update: int = 10_000
    fail_snapshot: bool = False
    fail_set_weights: bool = False
    set_calls: list[tuple[list[int], list[float], int]] = field(default_factory=list)

    def snapshot(self) -> Snapshot:
        if self.fail_snapshot:
            raise ConnectionError("node down")
        self.block += 10
        return Snapshot(block=self.block, neurons=tuple(self.neurons), owner_hotkey=self.owner_hotkey)

    def blocks_since_last_update(self, uid: int) -> int:
        return self.since_update

    def weights_rate_limit(self) -> int:
        return self.rate_limit

    def set_weights(self, uids: Sequence[int], weights: Sequence[float], version_key: int) -> None:
        if self.fail_set_weights:
            raise RuntimeError("pool rejected")
        self.set_calls.append((list(uids), list(weights), version_key))

    def serve_axon(self, ip: str, port: int) -> None:  # pragma: no cover - unused
        pass


@pytest.fixture
def fake_chain_cls():
    return FakeChain


# The entry floors the evaluator mechanism tests were calibrated on (1.2.0rc5 release values). The fake models' scores
# do not follow the real models' scale, so tests of the state machine and the copy checks pin these instead of following
# every re-measured release floor; the release values have their own tests (test_competitions.py, test_ka_nonproc.py).
RC5_ENTRY_FLOOR = {"demand_forecasting": 24.8, "risk_assessment": 33.7}


@pytest.fixture
def rc5_entry_floors(monkeypatch):
    from reax_subnet import competition_split as cs
    for k, v in RC5_ENTRY_FLOOR.items():
        monkeypatch.setitem(cs.TRACK_ENTRY_FLOOR, k, v)


# The specialist rules depend on the wall clock (``spec.rules.rules_at``: the open class starts on
# competition_split.SPEC_OPEN_CLASS_START_UTC). Tests that were written for the rules before it pin the clock to a time
# before the switch, so that they do not change on 22 Oct 2026; tests/test_open_class.py sets its own clock.
# Only the wall clock of the rules is pinned. The evaluator and the remote reader take a ``clock=`` argument, and every test that
# decides something with it passes a fake one (test_spec_evaluator.Env: Clock(T0); the remote tests: NOW0). The few that keep
# the real ``time.time`` only read leaders (test_remote_competitions: ``SpecEvaluator(cfg)``) or evaluate entries of at most 9 B,
# which the open class does not change (``_big_ok``: small or unknown size). Checked by running the spec, keep-alive, provisional,
# competition and remote test files with ``time.time`` shifted six days ahead (after the switch): no failure came from the switch
# (the ones left were file mtimes / signed approvals dated "in the future" and the known DevRunner flakes).
PRE_OPEN_CLASS_TS = 1792454400.0   # 2026-10-20T00:00:00Z


@pytest.fixture(autouse=True)
def _wall_clock_before_the_open_class(monkeypatch):
    from reax_subnet.spec import rules as R
    monkeypatch.setattr(R, "_wall_clock", lambda: PRE_OPEN_CLASS_TS)


@pytest.fixture(autouse=True)
def _synchronous_probe(monkeypatch):
    """Probe tests written before 1.2.0rc11.post6 read a probe's capacities right after ``run_round``: they run the
    synchronous probe (REAX_PROBE_BACKGROUND=0). tests/test_probe_cadence.py runs the background probe (the release
    default) and checks that its later fold equals the synchronous one."""
    monkeypatch.setenv("REAX_PROBE_BACKGROUND", "0")


@pytest.fixture(autouse=True)
def _no_fla_checksum_on_fake_docker(request, monkeypatch):
    """Runner tests drive fake docker/nvidia-smi without the published config files on disk; tests marked
    ``fla_checksum`` (tests/test_engfla_publish.py) run the real engconfigs.verify_published_configs."""
    if request.node.get_closest_marker("fla_checksum"):
        return
    from reax_subnet import engrunner

    monkeypatch.setattr(engrunner, "verify_published_configs", lambda gpu_class, host_dir: None)


# rc11.post1: the release ships CROWN_SLOTS_UTC = (Mon 26 Oct 2026 14:30Z,) (Florian #23798). Tests that crown on a crowned track run
# with "now" before that slot, so every test sees the pre-slot behaviour (crown at once) unless it sets the constant itself.
from reax_subnet import competition_split as _CS_RELEASE   # noqa: E402
RELEASE_CROWN_SLOTS_UTC = _CS_RELEASE.CROWN_SLOTS_UTC
RELEASE_ENGINE_CROWN_SLOTS_UTC = _CS_RELEASE.ENGINE_CROWN_SLOTS_UTC


@pytest.fixture(autouse=True)
def _no_crown_slots_unless_set(monkeypatch):
    monkeypatch.setattr(_CS_RELEASE, "CROWN_SLOTS_UTC", ())
    monkeypatch.setattr(_CS_RELEASE, "ENGINE_CROWN_SLOTS_UTC", ())


# The release arms eval v5.1 (sampling.EVAL_V5_START_UTC = Mon 26 Oct 2026 14:30Z, pool v0.1). Tests run with the gate off unless
# they set it themselves, as with the crown slots above: fake clocks (and the real one after Monday) would otherwise cross it in
# tests written for the rules before it. tests/test_v5_arm.py checks the armed release value and the behaviour around it.
from reax_subnet.spec import sampling as _S_RELEASE   # noqa: E402
RELEASE_EVAL_V5_START_UTC = _S_RELEASE.EVAL_V5_START_UTC


@pytest.fixture(autouse=True)
def _v5_gate_off_unless_set(monkeypatch):
    monkeypatch.setattr(_S_RELEASE, "EVAL_V5_START_UTC", None)


# rules-v4 (rc11): the gates below switched on at 2026-10-23 13:00Z. Tests written for the rules before them request this
# fixture (``@pytest.mark.usefixtures("rules_v4_gates_off")``) so that they do not depend on the wall clock; the gate behaviour
# itself is tested in test_coldkey_dr / test_draws_v4 / test_confirm_before_close / test_engine_copyline / test_opening_playoff_slots.
@pytest.fixture
def rules_v4_gates_off(monkeypatch):
    from reax_subnet import competition_split, engcomp, tiers
    from reax_subnet.spec import machine, sampling
    for mod, name in ((tiers, "COLDKEY_DR_START_UTC"), (sampling, "DRAWS_V4_START_UTC"),
                      (machine, "CONFIRM_BEFORE_CLOSE_START_UTC"), (engcomp, "ENGINE_COPYLINE_START_UTC"),
                      (competition_split, "OPENING_PLAYOFF_SLOTS_START_UTC")):
        monkeypatch.setattr(mod, name, None)
