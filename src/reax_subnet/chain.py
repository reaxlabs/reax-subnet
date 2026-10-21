"""The small slice of the chain that miners and validators use.

``Chain`` is an interface so the validator loop can be tested without a
node. ``SdkChain`` implements it with the Bittensor SDK (v11). Every method
may raise; callers treat any exception as "chain unavailable this round" and
retry later instead of crashing.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neuron:
    uid: int
    hotkey: str
    coldkey: str
    validator_permit: bool
    stake: float  # subnet alpha stake
    ip: Optional[str]
    port: Optional[int]
    # Plaintext on-chain commitment (Commitments pallet), "" if none. reax-s1 miners publish
    # "reax-s1/1:<model>:<manifest_id>" here.
    commitment: str = ""
    commitment_block: int = 0  # block of that commitment (0 = unknown)
    registered_block: int = 0  # block this hotkey registered on the subnet (metagraph block_at_registration; 0 = unknown)
    # The commitment's Raw fields, one element each (elements.py, docs/DESIGN-MULTIROLE.md); () = not read separately
    # (then ``commitment`` is the only field). A field that is not plain UTF-8 Raw data reads as INVALID_FIELD.
    commitment_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    block: int
    neurons: tuple[Neuron, ...]
    owner_hotkey: Optional[str]

    @property
    def owner_uid(self) -> Optional[int]:
        for n in self.neurons:
            if n.hotkey == self.owner_hotkey:
                return n.uid
        return None

    def uid_of(self, hotkey: str) -> Optional[int]:
        for n in self.neurons:
            if n.hotkey == hotkey:
                return n.uid
        return None


class Chain(Protocol):
    def snapshot(self) -> Snapshot: ...

    def blocks_since_last_update(self, uid: int) -> int: ...

    def weights_rate_limit(self) -> int: ...

    def set_weights(self, uids: Sequence[int], weights: Sequence[float], version_key: int) -> None: ...

    def serve_axon(self, ip: str, port: int) -> None: ...

    def set_commitment(self, text: str) -> None: ...

    def set_commitment_fields(self, texts: Sequence[str]) -> None: ...

    # Optional (callers use getattr): ``def block_hash(self, block: int) -> Optional[str]`` - the hash of a block,
    # None when it does not exist yet or cannot be read.


def parse_axon(axon: Any) -> tuple[Optional[str], Optional[int]]:
    """Parse the SDK's ``"ip:port"`` axon string. Returns (None, None) when unset or invalid."""
    if not axon or not isinstance(axon, str):
        return None, None
    host, sep, port_s = axon.rpartition(":")
    if not sep:
        return None, None
    host = host.strip("[]")
    try:
        ip = ipaddress.ip_address(host)
        port = int(port_s)
    except ValueError:
        return None, None
    if not 0 < port < 65536 or ip.is_unspecified:
        return None, None
    return str(ip), port


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class SdkChain:
    def __init__(self, network: str, netuid: int, wallet: Any, commitments: bool = False):
        import bittensor as bt

        self._bt = bt
        self.netuid = netuid
        self.wallet = wallet
        self.subtensor = bt.Subtensor(network=network)
        self.read_commitments = commitments
        self._block_hashes: dict[int, str] = {}

    def snapshot(self) -> Snapshot:
        mg = self.subtensor.subnets.metagraph(self.netuid, commitments=self.read_commitments)
        if mg is None:
            raise RuntimeError(f"subnet {self.netuid} does not exist")
        neurons = []
        for n in mg.neurons:
            ip, port = parse_axon(n.axon)
            neurons.append(
                Neuron(
                    uid=int(n.uid),
                    hotkey=str(n.hotkey),
                    coldkey=str(n.coldkey),
                    validator_permit=bool(n.validator_permit),
                    stake=float(n.total_stake.amount),
                    ip=ip,
                    port=port,
                    commitment=_commitment_text(getattr(n, "commitment", None)),
                    commitment_block=_commitment_block(getattr(n, "commitment", None)),
                    commitment_fields=_commitment_fields(getattr(n, "commitment", None)),
                    registered_block=_int_or_zero(getattr(n, "block_at_registration", 0)),
                )
            )
        return Snapshot(block=int(mg.block), neurons=tuple(neurons), owner_hotkey=mg.owner_hotkey)

    def blocks_since_last_update(self, uid: int) -> int:
        value = self.subtensor.epochs.blocks_since_last_update(self.netuid, uid)
        return int(value) if value is not None else 1 << 30

    def weights_rate_limit(self) -> int:
        return int(self.subtensor.hyperparameters.weights_rate_limit(self.netuid))

    def set_weights(self, uids: Sequence[int], weights: Sequence[float], version_key: int) -> None:
        bt = self._bt
        result = self.subtensor.execute(
            bt.SetWeights(self.netuid, uids=list(uids), weights=list(weights), version_key=version_key),
            self.wallet,
        )
        if not result.success:
            raise RuntimeError(f"set_weights failed: {result.message}")

    def set_commitment(self, text: str) -> None:
        self.set_commitment_fields([text])

    def set_commitment_fields(self, texts: Sequence[str]) -> None:
        """Replace the whole commitment with these fields (one element each; at most 3, 1..128 bytes each)."""
        from bittensor._generated import calls

        datas = [t.encode("utf-8") for t in texts]
        if not 1 <= len(datas) <= 3 or not all(0 < len(d) <= 128 for d in datas):
            raise ValueError("a commitment is 1..3 fields of 1..128 bytes")
        info = {"fields": [[{f"Raw{len(d)}": d} for d in datas]]}
        result = self.subtensor.submit_call(
            calls.Commitments.set_commitment(netuid=self.netuid, info=info), self.wallet, signer="hotkey"
        )
        if not result.success:
            raise RuntimeError(f"set_commitment failed: {result.message}")

    def genesis_hash(self) -> str:
        """Hash of block 0. ``block_info(0)`` would also read state at block 0, which only archive nodes keep."""
        import inspect

        client = getattr(self.subtensor, "_client", None)
        if client is not None and hasattr(client, "_block_hash"):
            r = client._block_hash(0)
            value = self.subtensor._call(r) if inspect.iscoroutine(r) else r
        else:  # pragma: no cover - other SDK layouts
            value = self.subtensor.block_info(0).hash
        if not isinstance(value, str) or not value.startswith("0x"):
            raise RuntimeError("could not read the genesis hash")
        return value

    def block_hash(self, block: int) -> Optional[str]:
        """Hash of ``block`` (0x...), cached once known. None when the block does not exist yet or the read fails."""
        cached = self._block_hashes.get(block)
        if cached is not None:
            return cached
        try:
            info = self.subtensor.block_info(int(block))
        except Exception as exc:  # noqa: BLE001
            log.warning("block hash of %s not readable: %r", block, exc)
            return None
        value = getattr(info, "hash", None) if info is not None else None
        if not isinstance(value, str) or not value.startswith("0x"):
            return None
        self._block_hashes[block] = value
        return value

    def serve_axon(self, ip: str, port: int) -> None:
        bt = self._bt
        result = self.subtensor.execute(bt.ServeAxon(self.netuid, ip, port), self.wallet)
        if not result.success:
            raise RuntimeError(f"serve_axon failed: {result.message}")


FINNEY_GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"


def _commitment_block(c: Any) -> int:
    value = getattr(c, "block", 0) if c is not None else 0
    return value if isinstance(value, int) and value > 0 else 0


# Stands for a field that is not plain UTF-8 Raw data (a hash or timelock variant, bad UTF-8): no element parses it, so it
# voids a commitment that contains it.
INVALID_FIELD = "\x00"


def _commitment_fields(c: Any) -> tuple[str, ...]:
    """The commitment's fields as text, one per field (SDK ``NeuronCommitment.fields``: [{variant: 0x-hex}, ...])."""
    fields = getattr(c, "fields", None) if c is not None else None
    if not isinstance(fields, (list, tuple)):
        return ()
    out = []
    for entry in fields[:8]:
        if not isinstance(entry, dict) or len(entry) != 1:
            out.append(INVALID_FIELD)
            continue
        (variant, value), = entry.items()
        if not (isinstance(variant, str) and variant.startswith("Raw") and isinstance(value, str)):
            out.append(INVALID_FIELD)
            continue
        try:
            out.append(bytes.fromhex(value.removeprefix("0x")).decode("utf-8"))
        except ValueError:
            out.append(INVALID_FIELD)
    return tuple(out)


def _commitment_text(c: Any) -> str:
    """Visible plaintext of an SDK NeuronCommitment (or None) as a short string."""
    if c is None:
        return ""
    value = getattr(c, "value", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(c, "data", "")
    return value[:256] if isinstance(value, str) else ""
