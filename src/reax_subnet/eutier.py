"""EU tier of the inference pools (Scheme A x1.5 and the EU contract tier; docs/EU-TIER.md). INACTIVE by default.

Two lanes for EU-verified capacity, both driven by one versioned registry file (``REGISTRY_FILE`` shipped with the
package, ``REAX_EU_REGISTRY_FILE`` overrides it; reloaded every round):

* **Scheme A** (Florian, card 18283, 23 Oct 2026): a verified EU hotkey's in-pool score is multiplied by
  ``SCHEME_A_MULTIPLIER`` (1.5) in the GPU pools, capped per operator: of an operator's summed capacity factor in a pool
  at most ``cap_units`` units earn the bonus (splitting into more hotkeys gains nothing). Any failed check -> x1.0.
* **Contract tier** (card 18296): operators with contracted slots earn EUR ``eur_per_gpu_hour`` per verified GPU-hour,
  converted to alpha at the daily price and carved out of the inference share (pay rule 1-6 of the decided spec):
  ``H_o = min(slots x 24, verified GPU-hours)``, ``A_o = H_o x eur / P``, ``sum A_o <= cap_pct % x inference share``
  (pro rata cut), contracted hotkeys earn no normal inference pay while their operator is paid (no double pay), and a
  public receipt per UTC day.

Fail closed everywhere: no file, an invalid file (schema, duplicate hotkey, bad number), an operator without a signed
contract id + DPA id + datacenter + ASN, a hotkey without a fresh passed EU verification, no price, no inference-share
estimate -> that part does nothing, and with nothing active the weights are exactly the weights without this module.
Nothing here may ever stop weight setting: callers wrap every hook (s1validator).

Inputs that live outside the validator are injectable interfaces with file-backed defaults:
``VerificationSource`` (RTT/ASN checks; ``REAX_EU_VERIFICATION_FILE``) and ``MarketData`` (alpha price, TAO/EUR quotes,
the day's inference-share alpha; ``REAX_EU_MARKET_FILE``). No network call is made from the validator.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
import statistics
import tempfile
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Protocol, Sequence

log = logging.getLogger("reax.eutier")

REGISTRY_VERSION = 1
# The shipped registry: no operators (inactive). Entering a contract = a reviewed registry revision.
REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "eu_registry.json")
MAX_FILE_BYTES = 1024 * 1024

# Scheme A: x1.5 on the in-pool score (ema x capacity factor) of verified EU hotkeys, in these pools only.
SCHEME_A_MULTIPLIER = 1.5
SCHEME_A_POOLS: frozenset[str] = frozenset({"s1-fast", "s1-pro"})

# Contract tier limits a registry may not exceed (the registry carries the decided values).
MAX_SLOTS_PER_OPERATOR = 64
MAX_GPUS_PER_HOTKEY = 16
MAX_CAP_PCT = 5.0
# An EU verification (RTT probes + ASN match) counts only this long after it was made.
VERIFICATION_MAX_AGE_S = 6 * 3600
CLOCK_SKEW_S = 300
# A price quote counts for day D only if taken within this many seconds of D 00:00Z (pay rule 2).
PRICE_MAX_AGE_S = 2 * 3600
# TAO/EUR needs at least this many independent sources (pay rule 2: CoinGecko + one second source).
MIN_TAO_EUR_SOURCES = 2
# A correctness failure in this many hours of the trailing 24 suspends a hotkey until a clean hour (pay rule 1).
SUSPEND_FAILED_HOURS = 3
# Hour buckets and settlements kept in the ledger state.
# An hour passes only if the hotkey answered at least this share of the rounds it was queried in that hour (pay rule 1).
HOUR_PASS_RATIO = 0.9
# 48 h suspension look-back + the settled day + the hours until its latest settlement (D+1 23:00Z) + a restart margin.
KEEP_HOURS = 120
# Hour (UTC) of the day after the settled day from which a still missing market entry / inference share is settled as
# "no price" (pays nothing) instead of being retried.
SETTLE_CUTOFF_HOUR = 23
KEEP_DAYS = 45

_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_HOTKEY = re.compile(r"[1-9A-HJ-NP-Za-km-z]{46,48}")  # ss58
_PLACEHOLDER = re.compile(r"^\s*<.*>\s*$")
# Words a registry author may leave in a field before the paper exists: they count as missing, like <...>.
_PLACEHOLDER_WORDS = frozenset({"tbd", "tba", "todo", "n/a", "na", "none", "null", "pending", "unknown", "xxx", "-", "?"})


class RegistryError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Operator:
    id: str
    lane: str  # "scheme_a" | "contract"
    contract: str
    dpa: str
    datacenter: str
    asn: str
    hotkeys: Mapping[str, int]  # hotkey -> GPUs behind it (contract tier; Scheme A: 1)
    slots: int = 0  # contract tier
    cap_units: float = 0.0  # Scheme A

    @property
    def active(self) -> bool:
        """Signed contract id, DPA id, named datacenter and ASN present (placeholders count as missing)."""
        ok = all(_filled(v) for v in (self.contract, self.dpa, self.datacenter, self.asn)) and bool(self.hotkeys)
        if self.lane == "contract":
            return ok and self.slots > 0
        return ok and self.cap_units > 0


@dataclasses.dataclass(frozen=True)
class ContractParams:
    eur_per_gpu_hour: float = 1.00
    cap_pct: float = 5.0
    gpu_class: str = ""
    review_due: str = ""


@dataclasses.dataclass(frozen=True)
class Registry:
    version: int = REGISTRY_VERSION
    revision: str = "empty"
    scheme_a: tuple[Operator, ...] = ()
    contract: tuple[Operator, ...] = ()
    params: ContractParams = ContractParams()

    def active_scheme_a(self) -> list[Operator]:
        return [o for o in self.scheme_a if o.active]

    def active_contract(self) -> list[Operator]:
        return [o for o in self.contract if o.active and self.params.eur_per_gpu_hour > 0 and self.params.cap_pct > 0]

    @property
    def empty(self) -> bool:
        return not self.active_scheme_a() and not self.active_contract()


EMPTY = Registry()


def _filled(v: str) -> bool:
    return (isinstance(v, str) and bool(v.strip()) and not _PLACEHOLDER.match(v)
            and v.strip().lower() not in _PLACEHOLDER_WORDS)


def _str(d: Mapping[str, Any], key: str, where: str) -> str:
    v = d.get(key, "")
    if not isinstance(v, str) or len(v) > 200:
        raise RegistryError(f"{where}: {key} must be a string (<= 200 chars)")
    return v


def _num(v: Any, where: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise RegistryError(f"{where}: must be a number in [{lo}, {hi}]")
    return float(v)


def _int(v: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise RegistryError(f"{where}: must be an integer in [{lo}, {hi}]")
    return v


def _hotkeys(raw: Any, where: str, contract: bool) -> dict[str, int]:
    if not isinstance(raw, list):
        raise RegistryError(f"{where}: hotkeys must be a list")
    out: dict[str, int] = {}
    for i, h in enumerate(raw):
        gpus = 1
        if isinstance(h, Mapping) and contract:
            if set(h) - {"hotkey", "gpus"}:
                raise RegistryError(f"{where}.hotkeys[{i}]: unknown keys {sorted(set(h) - {'hotkey', 'gpus'})}")
            gpus = _int(h.get("gpus", 1), f"{where}.hotkeys[{i}].gpus", 1, MAX_GPUS_PER_HOTKEY)
            h = h.get("hotkey")
        if not isinstance(h, str) or not _HOTKEY.fullmatch(h):
            raise RegistryError(f"{where}.hotkeys[{i}]: not an ss58 hotkey")
        if h in out:
            raise RegistryError(f"{where}: hotkey {h} listed twice")
        out[h] = gpus
    return out


_OP_KEYS = {"id", "contract", "dpa", "datacenter", "asn", "hotkeys", "slots", "cap_units", "note"}


def _operator(raw: Any, lane: str, where: str) -> Operator:
    if not isinstance(raw, Mapping):
        raise RegistryError(f"{where}: must be an object")
    extra = set(raw) - _OP_KEYS
    if extra:
        raise RegistryError(f"{where}: unknown keys {sorted(extra)}")
    oid = raw.get("id")
    if not isinstance(oid, str) or not _ID.fullmatch(oid):
        raise RegistryError(f"{where}: id must match {_ID.pattern}")
    common = dict(id=oid, lane=lane, contract=_str(raw, "contract", where), dpa=_str(raw, "dpa", where),
                  datacenter=_str(raw, "datacenter", where), asn=_str(raw, "asn", where),
                  hotkeys=_hotkeys(raw.get("hotkeys", []), where, lane == "contract"))
    if lane == "contract":
        if "cap_units" in raw:
            raise RegistryError(f"{where}: cap_units belongs to scheme_a operators")
        return Operator(**common, slots=_int(raw.get("slots", 0), f"{where}.slots", 0, MAX_SLOTS_PER_OPERATOR))
    if "slots" in raw:
        raise RegistryError(f"{where}: slots belongs to contract-tier operators")
    return Operator(**common, cap_units=_num(raw.get("cap_units", 0), f"{where}.cap_units", 0.0, 1e6))


def parse_registry(doc: Any) -> Registry:
    """Strict schema check; any violation raises RegistryError (the caller then uses the empty registry)."""
    if not isinstance(doc, Mapping):
        raise RegistryError("registry must be a JSON object")
    extra = set(doc) - {"version", "revision", "scheme_a", "eu_contract_tier", "note"}
    if extra:
        raise RegistryError(f"unknown top-level keys {sorted(extra)}")
    if doc.get("version") != REGISTRY_VERSION:
        raise RegistryError(f"version must be {REGISTRY_VERSION}")
    revision = doc.get("revision", "")
    if not isinstance(revision, str) or not revision or len(revision) > 64:
        raise RegistryError("revision must be a non-empty string")
    sa = doc.get("scheme_a", {"operators": []})
    ct = doc.get("eu_contract_tier", {"operators": []})
    if not isinstance(sa, Mapping) or set(sa) - {"operators", "note"}:
        raise RegistryError("scheme_a must be {operators: [...]}")
    if not isinstance(ct, Mapping) or set(ct) - {"version", "eur_per_gpu_hour", "gpu_class",
                                                  "cap_pct_of_inference_share", "review_due", "operators", "note"}:
        raise RegistryError("eu_contract_tier has unknown keys")
    if not isinstance(sa.get("operators", []), list) or not isinstance(ct.get("operators", []), list):
        raise RegistryError("operators must be lists")
    scheme_a = tuple(_operator(o, "scheme_a", f"scheme_a.operators[{i}]") for i, o in enumerate(sa.get("operators", [])))
    contract = tuple(_operator(o, "contract", f"eu_contract_tier.operators[{i}]")
                     for i, o in enumerate(ct.get("operators", [])))
    params = ContractParams(
        eur_per_gpu_hour=_num(ct.get("eur_per_gpu_hour", 1.0), "eur_per_gpu_hour", 0.0, 100.0),
        cap_pct=_num(ct.get("cap_pct_of_inference_share", 5.0), "cap_pct_of_inference_share", 0.0, MAX_CAP_PCT),
        gpu_class=_str(ct, "gpu_class", "eu_contract_tier"), review_due=_str(ct, "review_due", "eu_contract_tier"))
    seen_ids: set[str] = set()
    seen_hk: dict[str, str] = {}
    for op in (*scheme_a, *contract):
        if op.id in seen_ids:
            raise RegistryError(f"operator id {op.id} used twice")
        seen_ids.add(op.id)
        for h in op.hotkeys:  # a hotkey belongs to at most one operator (and so to one lane)
            if h in seen_hk:
                raise RegistryError(f"hotkey {h} listed under {seen_hk[h]} and {op.id}")
            seen_hk[h] = op.id
    return Registry(version=REGISTRY_VERSION, revision=revision, scheme_a=scheme_a, contract=contract, params=params)


def load_registry(path: str) -> Registry:
    """The registry at ``path``; a missing, unreadable or invalid file is logged at ERROR and gives the empty one."""
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise RegistryError("file too large")
        with open(path, encoding="utf-8") as fh:
            reg = parse_registry(json.load(fh))
    except FileNotFoundError:
        log.error("EU registry %s not found: EU tier off (no multiplier, no contract pay)", path)
        return EMPTY
    except (OSError, ValueError) as exc:  # json.JSONDecodeError and RegistryError are ValueErrors
        log.error("EU registry %s invalid (%s): EU tier off (no multiplier, no contract pay)", path, exc)
        return EMPTY
    except Exception as exc:  # e.g. RecursionError on deeply nested JSON: fail closed
        log.error("EU registry %s failed to load (%s: %s): EU tier off (no multiplier, no contract pay)", path,
                  type(exc).__name__, exc)
        return EMPTY
    inactive = [o.id for o in (*reg.scheme_a, *reg.contract) if not o.active]
    if inactive:
        log.warning("EU registry %s: operators %s lack a signed contract / DPA id / datacenter / ASN / slots or cap: "
                    "inactive", reg.revision, inactive)
    return reg


# ------------------------------------------------------------------ verification (RTT probes + ASN/contract match)
class VerificationSource(Protocol):
    def verified(self, hotkey: str, now: datetime) -> bool: ...


class NoVerification:
    """No probe infrastructure: nobody is verified (fail closed)."""

    def verified(self, hotkey: str, now: datetime) -> bool:
        return False


class FileVerification:
    """``{"version": 1, "hotkeys": {hotkey: {"rtt_ok": bool, "asn_ok": bool, "checked_at": iso8601}}}``, written by
    the EU probe job (multi-site RTT from Helsinki/Katowice/Frankfurt + ASN match). A hotkey is verified only with both
    checks true and ``checked_at`` within VERIFICATION_MAX_AGE_S; anything else (no file, bad entry) = not verified."""

    def __init__(self, path: str):
        self.path = path
        self._stamp: Optional[tuple[float, int]] = None
        self._data: dict[str, tuple[bool, Optional[datetime]]] = {}

    def _reload(self) -> None:
        try:
            st = os.stat(self.path)
            stamp = (st.st_mtime, st.st_size)
            if stamp == self._stamp:
                return
            self._stamp = stamp
            self._data = {}
            if st.st_size > MAX_FILE_BYTES:
                raise ValueError("file too large")
            with open(self.path, encoding="utf-8") as fh:
                doc = json.load(fh)
            if not isinstance(doc, Mapping) or doc.get("version") != 1 or not isinstance(doc.get("hotkeys"), Mapping):
                raise ValueError("schema")
            for hk, e in doc["hotkeys"].items():
                if isinstance(e, Mapping):
                    ok = e.get("rtt_ok") is True and e.get("asn_ok") is True
                    self._data[hk] = (ok, _parse_ts(e.get("checked_at")))
        except FileNotFoundError:
            self._stamp, self._data = None, {}
        except (OSError, ValueError) as exc:
            log.error("EU verification file %s invalid (%s): nobody verified", self.path, exc)
            self._data = {}

    def verified(self, hotkey: str, now: datetime) -> bool:
        self._reload()
        ok, at = self._data.get(hotkey, (False, None))
        if not ok or at is None:
            return False
        age = (now - at).total_seconds()
        return -CLOCK_SKEW_S <= age <= VERIFICATION_MAX_AGE_S


def _parse_ts(v: Any) -> Optional[datetime]:
    if not isinstance(v, str):
        return None
    try:
        t = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


# ------------------------------------------------------------------ Scheme A
def scheme_a_multipliers(reg: Registry, verifier: VerificationSource, now: datetime,
                         factors_by_pool: Mapping[str, Mapping[int, float]],
                         hotkey_of: Mapping[int, str]) -> dict[str, dict[int, float]]:
    """``{pool: {slot: multiplier}}`` for verified Scheme A slots (only entries != 1). ``factors_by_pool[pool][slot]``
    is the slot's capacity factor in units (what the pool pays it for); per operator and pool the bonus covers at most
    ``cap_units`` units: each verified slot gets ``1 + (M - 1) x min(1, cap_units / sum of the operator's factors)``."""
    out: dict[str, dict[int, float]] = {}
    ops = reg.active_scheme_a()
    if not ops:
        return out
    owner = {h: op for op in ops for h in op.hotkeys}
    for pool, factors in factors_by_pool.items():
        if pool not in SCHEME_A_POOLS:
            continue
        groups: dict[str, list[int]] = {}
        for slot in factors:
            op = owner.get(hotkey_of.get(slot, ""))
            if op is not None and verifier.verified(hotkey_of[slot], now):
                groups.setdefault(op.id, []).append(slot)
        for oid, slots in sorted(groups.items()):
            op = next(o for o in ops if o.id == oid)
            total = math.fsum(max(0.0, factors[s]) for s in slots)
            if total <= 0:
                continue
            m = 1.0 + (SCHEME_A_MULTIPLIER - 1.0) * min(1.0, op.cap_units / total)
            for s in slots:
                out.setdefault(pool, {})[s] = m
    return out


# ------------------------------------------------------------------ market data (pay rule 2)
@dataclasses.dataclass(frozen=True)
class Quote:
    value: float
    at: datetime


@dataclasses.dataclass(frozen=True)
class DailyMarket:
    alpha_tao: Optional[Quote]  # alpha price in TAO (on-chain pool)
    tao_eur: Mapping[str, Quote]  # TAO/EUR per source
    inference_share_alpha: Optional[float]  # the day's inference-share alpha (estimate)


class MarketData(Protocol):
    def day(self, d: date) -> Optional[DailyMarket]: ...


class NoMarket:
    def day(self, d: date) -> Optional[DailyMarket]:
        return None


class FileMarket:
    """``{"version": 1, "days": {"YYYY-MM-DD": {"alpha_tao": {"value", "at"}, "tao_eur": {source: {"value", "at"}},
    "inference_share_alpha": float}}}`` written by tools/eu_tier_quote.py on Sandy (no network call in the validator)."""

    def __init__(self, path: str):
        self.path = path

    def day(self, d: date) -> Optional[DailyMarket]:
        try:
            if os.path.getsize(self.path) > MAX_FILE_BYTES:
                raise ValueError("file too large")
            with open(self.path, encoding="utf-8") as fh:
                doc = json.load(fh)
            if not isinstance(doc, Mapping) or doc.get("version") != 1 or not isinstance(doc.get("days"), Mapping):
                raise ValueError("schema")
            e = doc["days"].get(d.isoformat())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.error("EU market file %s invalid (%s): no price", self.path, exc)
            return None
        if not isinstance(e, Mapping):
            return None
        tao = e.get("tao_eur") if isinstance(e.get("tao_eur"), Mapping) else {}
        share = e.get("inference_share_alpha")
        return DailyMarket(alpha_tao=_quote(e.get("alpha_tao")),
                           tao_eur={k: q for k, v in tao.items() if (q := _quote(v)) is not None},
                           inference_share_alpha=float(share) if _pos(share) else None)


def _pos(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _quote(v: Any) -> Optional[Quote]:
    if not isinstance(v, Mapping) or not _pos(v.get("value")):
        return None
    at = _parse_ts(v.get("at"))
    return Quote(float(v["value"]), at) if at is not None else None


def day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def fresh_price(m: Optional[DailyMarket], d: date) -> Optional[float]:
    """P (EUR per alpha) at D 00:00Z from quotes taken within PRICE_MAX_AGE_S of it: alpha_TAO x median(TAO_EUR over
    >= MIN_TAO_EUR_SOURCES sources). None = missing or stale."""
    if m is None or m.alpha_tao is None:
        return None
    t0 = day_start(d)

    def ok(q: Quote) -> bool:
        return abs((q.at - t0).total_seconds()) <= PRICE_MAX_AGE_S

    eur = [q.value for q in m.tao_eur.values() if ok(q)]
    if not ok(m.alpha_tao) or len(eur) < MIN_TAO_EUR_SOURCES:
        return None
    p = m.alpha_tao.value * statistics.median(eur)
    return p if math.isfinite(p) and p > 0 else None


# ------------------------------------------------------------------ contract-tier ledger (pay rule 1, 3-6)
def _hour_key(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")


class ContractLedger:
    """Per hotkey and UTC hour: availability (an answered, non-struck round), correctness failure (any strike in a
    round or capacity probe), EU verification (must hold at every record in the hour). Settles each finished UTC day
    once into a receipt; the settlement of the last finished day decides the pay fractions until the next one.

    State is JSON at ``state_path`` (written atomically); receipts go to ``receipts_dir/YYYY-MM-DD.json``."""

    def __init__(self, state_path: str, receipts_dir: str):
        self.state_path = state_path
        self.receipts_dir = receipts_dir
        # hotkey -> hour -> {"q": rounds queried, "k": answered without strike, "f": correctness failure,
        # "v": verified throughout}
        self.hours: dict[str, dict[str, dict[str, bool]]] = {}
        self.settled: dict[str, dict[str, Any]] = {}  # day -> receipt
        self.last_price: Optional[float] = None
        self._load()

    # persistence
    def _load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.hours = {str(h): {str(k): self._bucket(v) for k, v in hs.items()}
                          for h, hs in doc.get("hours", {}).items()}
            self.settled = dict(doc.get("settled", {}))
            lp = doc.get("last_price")
            self.last_price = float(lp) if _pos(lp) else None
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            log.error("EU tier ledger %s unreadable (%s): starting empty", self.state_path, exc)
            self.hours, self.settled, self.last_price = {}, {}, None

    @staticmethod
    def _bucket(v: Mapping[str, Any]) -> dict[str, Any]:
        if "q" in v:
            q, k = max(0, int(v["q"])), max(0, int(v.get("k", 0)))
        else:  # an hour recorded before the answered/queried counts: one round, answered or not
            q, k = 1, 1 if v.get("a") else 0
        return {"q": q, "k": min(k, q), "f": bool(v.get("f")), "v": bool(v.get("v"))}

    def save(self) -> None:
        doc = {"version": 1, "hours": self.hours, "settled": self.settled,
               "last_price": self.last_price}
        _atomic_write(self.state_path, doc)

    # recording
    def record(self, now: datetime, outcomes: Mapping[str, tuple[bool, bool]], verified: Mapping[str, bool]) -> None:
        """``outcomes[hotkey] = (answered_ok, correctness_failed)`` of one round (or probe), for contract-tier hotkeys
        only; ``verified[hotkey]`` its EU verification at this moment."""
        hk = _hour_key(now)
        for h, (ok, failed) in outcomes.items():
            b = self.hours.setdefault(h, {}).setdefault(hk, {"q": 0, "k": 0, "f": False, "v": True})
            b["q"] += 1
            b["k"] += 1 if (ok and not failed) else 0
            b["f"] = b["f"] or failed
            b["v"] = b["v"] and bool(verified.get(h, False))
        self._prune(now)

    def _prune(self, now: datetime) -> None:
        cut = _hour_key(now - timedelta(hours=KEEP_HOURS))
        for h in list(self.hours):
            self.hours[h] = {k: v for k, v in self.hours[h].items() if k >= cut}
            if not self.hours[h]:
                del self.hours[h]
        if len(self.settled) > KEEP_DAYS:
            for d in sorted(self.settled)[:-KEEP_DAYS]:
                del self.settled[d]

    def passed_hours(self, hotkey: str, d: date) -> int:
        """Hours of day ``d`` that pass for ``hotkey``, walking the hours in order with the suspension rule: a failed
        hour voids itself; 3 failed hours in the trailing 24 suspend the hotkey; a clean hour (answered in >= 90 % of its
        queried rounds, no failure, verified) lifts the suspension but is itself not paid."""
        hs = self.hours.get(hotkey, {})
        start = day_start(d) - timedelta(hours=48)  # the two days before carry a suspension into the day
        suspended = False
        failed: list[datetime] = []
        n = 0
        for i in range(72):
            t = start + timedelta(hours=i)
            b = hs.get(_hour_key(t))
            in_day = t >= day_start(d)
            if b is None:
                continue
            failed = [x for x in failed if t - x < timedelta(hours=24)]
            if b["f"]:
                failed.append(t)
                if len(failed) >= SUSPEND_FAILED_HOURS:
                    suspended = True
                continue
            clean = b["q"] > 0 and b["k"] >= HOUR_PASS_RATIO * b["q"] and b["v"]
            if suspended:
                if clean:
                    suspended = False  # the clean re-probe hour itself is not paid
                continue
            if clean and in_day:
                n += 1
        return n

    def _receipt(self, reg: Registry, key: str, m: Optional[DailyMarket], p: Optional[float], price_source: str,
                 ops: Sequence[Operator], quiet: bool = False) -> dict[str, Any]:
        d = date.fromisoformat(key)
        share = m.inference_share_alpha if m is not None else None
        params = reg.params
        rows = []
        for op in ops:
            per_hk = {h: self.passed_hours(h, d) * g for h, g in sorted(op.hotkeys.items())}
            raw = float(sum(per_hk.values()))
            h_o = min(float(op.slots * 24), raw)
            a_o = h_o * params.eur_per_gpu_hour / p if p else 0.0
            rows.append({"operator": op.id, "slots": op.slots, "verified_gpu_hours": raw, "H_o": h_o,
                         "A_o_uncapped": a_o, "hotkey_gpu_hours": per_hk})
        total = math.fsum(r["A_o_uncapped"] for r in rows)
        cap = (params.cap_pct / 100.0) * share if share is not None else 0.0
        if share is None and total > 0 and not quiet:
            log.error("EU contract tier %s: no inference-share estimate for the day: the tier pays nothing", key)
        cut = 1.0 if total <= cap else (cap / total if total > 0 else 0.0)
        if cut < 1.0 and share is not None:
            log.warning("EU contract tier %s: %.4f alpha owed > cap %.4f (%.1f %% of %.2f): all A_o cut to %.6f",
                        key, total, cap, params.cap_pct, share, cut)
        for r in rows:
            r["A_o"] = r["A_o_uncapped"] * cut
            r["cap_cut_alpha"] = r["A_o_uncapped"] - r["A_o"]
            r["fraction_of_inference_share"] = r["A_o"] / share if share else 0.0
        return {"version": 1, "day": key, "registry_revision": reg.revision,
                "eur_per_gpu_hour": params.eur_per_gpu_hour, "gpu_class": params.gpu_class,
                "price_eur_per_alpha": p, "price_source": price_source,
                "alpha_tao": m.alpha_tao.value if m is not None and m.alpha_tao is not None else None,
                "tao_eur": ({k: q.value for k, q in sorted(m.tao_eur.items())} if m is not None else {}),
                "inference_share_alpha": share, "cap_pct": params.cap_pct, "cap_alpha": cap,
                "owed_alpha": total, "cap_cut_factor": cut, "operators": rows}

    def settle(self, reg: Registry, market: MarketData, now: datetime) -> Optional[dict[str, Any]]:
        """Settle the last finished UTC day (yesterday) once; returns its receipt (also for an earlier settlement)."""
        d = (now.astimezone(timezone.utc) - timedelta(days=1)).date()
        key = d.isoformat()
        if key in self.settled:
            return self.settled[key]
        ops = reg.active_contract()
        m = market.day(d)
        no_data = m is None or m.inference_share_alpha is None
        p = fresh_price(m, d)
        late = now.astimezone(timezone.utc) >= day_start(d + timedelta(days=1)) + timedelta(hours=SETTLE_CUTOFF_HOUR)
        if ops and no_data and not late:
            # Contract operators are waiting for pay: do not cache an empty settlement, the next call retries.
            log.warning("EU contract tier %s: market entry / inference share missing: settlement retried", key)
            return self._receipt(reg, key, m, p, "fresh" if p else "none", ops, quiet=True)
        price_source = "fresh"
        if ops and no_data:
            p, price_source = None, "none"
            log.error("EU contract tier %s: market data still missing at %02d:00Z of the next day: settled as no "
                      "price, the tier pays nothing", key, SETTLE_CUTOFF_HOUR)
        elif p is None:
            if self.last_price is not None:
                p, price_source = self.last_price, "previous"
                log.warning("EU contract tier %s: alpha/EUR price missing or stale (> %d s): previous P %.6f reused",
                            key, PRICE_MAX_AGE_S, p)
            else:
                price_source = "none"
                log.error("EU contract tier %s: no price ever: the tier pays nothing", key)
        else:
            self.last_price = p
        receipt = self._receipt(reg, key, m, p, price_source, ops)
        self.settled[key] = receipt
        if ops:  # an inactive tier writes no receipts (byte-identical deployments stay silent)
            try:
                _atomic_write(os.path.join(self.receipts_dir, f"{key}.json"), receipt)
            except OSError as exc:
                log.error("EU contract tier: could not write receipt %s: %s", key, exc)
        return receipt

    def pay_fractions(self, reg: Registry, market: MarketData, now: datetime) -> tuple[dict[str, float], set[str]]:
        """``({hotkey: fraction of the inference share}, hotkeys without normal inference pay)`` from the last
        finished day's settlement, for operators still active in ``reg``; split inside an operator by its hotkeys'
        verified GPU-hours. Every hotkey of a paid operator loses its normal inference pay (no double pay); an operator
        the tier does not pay (no price, no hours, no estimate) keeps the normal rule."""
        if not reg.active_contract():
            return {}, set()
        r = self.settle(reg, market, now)
        if r is None or r.get("registry_revision") != reg.revision:
            # A registry revision after the settlement pays nothing until the next settlement under it (fail closed).
            return {}, set()
        active = {o.id: o for o in reg.active_contract()}
        out: dict[str, float] = {}
        excluded: set[str] = set()
        for row in r.get("operators", []):
            f = float(row.get("fraction_of_inference_share", 0.0))
            hk = row.get("hotkey_gpu_hours", {})
            tot = math.fsum(hk.values())
            op = active.get(row.get("operator"))
            if op is None or not math.isfinite(f) or f <= 0 or tot <= 0:
                continue
            excluded |= set(op.hotkeys)
            for h, g in hk.items():
                if g > 0 and h in op.hotkeys:
                    out[h] = out.get(h, 0.0) + f * g / tot
        return out, excluded


def _atomic_write(path: str, doc: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def carve_out(base: Mapping[int, float], pay: Mapping[int, float], owner_uid: Optional[int]) -> dict[int, float]:
    """Carve ``pay`` (uid -> fraction of the inference share) out of the miner part of ``base`` (uid -> weight, sum 1,
    the owner entry = burn): the paid uids get ``fraction x miner part``, every other miner is scaled by
    ``1 - sum(pay)``; the owner entry is untouched. Fractions summing above 1 (impossible with the 5 % cap) or no
    other miner -> ``base`` unchanged. The competition layer later scales the whole miner part pro rata, so the
    fractions stay fractions of the inference share."""
    pay = {u: f for u, f in pay.items() if f > 0 and u != owner_uid}
    total = math.fsum(pay.values())
    if not pay or total >= 1.0:
        return dict(base)
    others = {u: w for u, w in base.items() if u != owner_uid and u not in pay}
    miner = math.fsum(w for u, w in base.items() if u != owner_uid)
    if miner <= 0 or not others:
        return dict(base)
    out = {u: w * (1.0 - total) for u, w in others.items()}
    for u, f in pay.items():
        out[u] = f * miner
    if owner_uid is not None and owner_uid in base:
        out[owner_uid] = base[owner_uid]
    return {u: out[u] for u in sorted(out)}


# ------------------------------------------------------------------ the validator's hook
class EUTier:
    """One per validator. ``refresh()`` once per round reloads the registry; ``record_round`` feeds the ledger;
    ``slot_multipliers`` / ``contract_pay`` are read by compute_weights. With the empty registry all of them return
    nothing and nothing is written."""

    def __init__(self, registry_path: str, verifier: VerificationSource, market: MarketData, state_path: str,
                 receipts_dir: str):
        self.registry_path = registry_path
        self.verifier = verifier
        self.market = market
        self.ledger_path, self.receipts_dir = state_path, receipts_dir
        self._ledger: Optional[ContractLedger] = None
        self._stamp: Optional[tuple[float, int]] = None
        self.registry: Registry = EMPTY
        self.refresh()

    @classmethod
    def from_env(cls, env: Mapping[str, str], validator_state_path: str) -> "EUTier":
        base = os.path.dirname(os.path.abspath(os.path.expanduser(validator_state_path)))
        reg = os.path.expanduser(env.get("REAX_EU_REGISTRY_FILE", "") or REGISTRY_FILE)
        ver = env.get("REAX_EU_VERIFICATION_FILE", "")
        mkt = env.get("REAX_EU_MARKET_FILE", "")
        return cls(reg, FileVerification(os.path.expanduser(ver)) if ver else NoVerification(),
                   FileMarket(os.path.expanduser(mkt)) if mkt else NoMarket(),
                   os.path.expanduser(env.get("REAX_EU_TIER_STATE_PATH", "") or os.path.join(base, "eu-tier-state.json")),
                   os.path.expanduser(env.get("REAX_EU_TIER_RECEIPTS_DIR", "") or os.path.join(base, "eu-tier-receipts")))

    @property
    def ledger(self) -> ContractLedger:
        if self._ledger is None:
            self._ledger = ContractLedger(self.ledger_path, self.receipts_dir)
        return self._ledger

    def refresh(self) -> Registry:
        try:
            st = os.stat(self.registry_path)
            stamp: Optional[tuple[float, int]] = (st.st_mtime, st.st_size)
        except OSError:
            stamp = None
        if stamp != self._stamp or stamp is None:
            new = load_registry(self.registry_path)
            self._stamp = stamp  # only after a successful load: an exception keeps the retry (and the old registry)
            if new.revision != self.registry.revision or new != self.registry:
                log.info("EU registry %s: revision %s, %d Scheme A / %d contract-tier operators active",
                         self.registry_path, new.revision, len(new.active_scheme_a()), len(new.active_contract()))
            self.registry = new
        return self.registry

    def contract_hotkeys(self) -> set[str]:
        return {h for op in self.registry.active_contract() for h in op.hotkeys}

    def record_round(self, now: datetime, outcomes: Mapping[str, tuple[bool, bool]]) -> None:
        hks = self.contract_hotkeys()
        mine = {h: o for h, o in outcomes.items() if h in hks}
        if not hks:
            return
        # Hotkeys of the tier that were not in this round at all count as not answered (availability), no failure.
        for h in hks:
            mine.setdefault(h, (False, False))
        self.ledger.record(now, mine, {h: self.verifier.verified(h, now) for h in mine})
        self.ledger.save()

    def slot_multipliers(self, now: datetime, factors_by_pool: Mapping[str, Mapping[int, float]],
                         hotkey_of: Mapping[int, str]) -> dict[str, dict[int, float]]:
        return scheme_a_multipliers(self.registry, self.verifier, now, factors_by_pool, hotkey_of)

    def contract_pay(self, now: datetime) -> tuple[dict[str, float], set[str]]:
        """``({hotkey: fraction of the inference share}, hotkeys excluded from normal pay)``; empty = the tier pays
        nobody right now."""
        if not self.registry.active_contract():
            return {}, set()
        out = self.ledger.pay_fractions(self.registry, self.market, now)
        self.ledger.save()
        return out
