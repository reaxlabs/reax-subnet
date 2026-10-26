"""Entries under review (rc11.post5): drawn as usual, but never confirmed, crowned or kept alive while the review lasts.

Two sources, both fail closed:

* the copy check's reparametrisation flag (``lineage.Plan.flags``, ``tensors.Sketch.review``): recorded once as a
  ``review_flag`` history row (no new state field, so the state file of an unflagged entry is unchanged), lifted only by
  ``REAX_SPEC_REVIEW_CLEAR``;
* the operator hold list ``REAX_SPEC_HOLD``.

Both variables are comma lists of ``<hotkey prefix>@<revision prefix>`` (each part at least ``MIN_PREFIX`` characters;
malformed items are ignored and logged). Empty or unset (the default): nothing is held and nothing changes. The lists
are read from the environment at each use, so they take effect without a state change; set them on every container
that runs the spec evaluator or computes the keep-alive (the validator reads the worker's state, ``remote.py``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

log = logging.getLogger("reax.spec.review")

HOLD_ENV = "REAX_SPEC_HOLD"
CLEAR_ENV = "REAX_SPEC_REVIEW_CLEAR"
MIN_PREFIX = 6
FLAG_KIND = "review_flag"
_warned: set[str] = set()


def parse(text: str) -> tuple[tuple[str, str], ...]:
    out = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        hk, sep, rev = item.partition("@")
        hk, rev = hk.strip(), rev.strip().lower()
        if not sep or len(hk) < MIN_PREFIX or len(rev) < MIN_PREFIX:
            if item not in _warned:
                _warned.add(item)
                log.warning("review list item %r ignored: <hotkey prefix>@<revision prefix>, each >= %d characters", item, MIN_PREFIX)
            continue
        out.append((hk, rev))
    return tuple(out)


def _matches(s: Any, items: tuple[tuple[str, str], ...]) -> bool:
    hk, rev = str(getattr(s, "hotkey", "")), str(getattr(s, "revision", "")).lower()
    return any(hk.startswith(h) and rev.startswith(r) for h, r in items)


def flag_of(s: Any) -> str:
    """The recorded reparametrisation flag of ``s`` ("" = none)."""
    for h in reversed(getattr(s, "history", None) or ()):
        if isinstance(h, dict) and h.get("kind") == FLAG_KIND:
            return ", ".join(str(f) for f in h.get("flags") or ()) or "flagged"
    return ""


def under_review(s: Any, env: "Mapping[str, str] | None" = None) -> str:
    """Why ``s`` is under review ("" = it is not)."""
    env = os.environ if env is None else env
    hold = env.get(HOLD_ENV, "")
    if hold and _matches(s, parse(hold)):
        return f"operator hold ({HOLD_ENV})"
    flag = flag_of(s)
    if flag:
        clear = env.get(CLEAR_ENV, "")
        if not (clear and _matches(s, parse(clear))):
            return f"weights reparametrised against the declared base ({flag}); a reviewer must clear it ({CLEAR_ENV})"
    return ""


def record(s: Any, flags: "list[str] | tuple[str, ...]", now: float) -> bool:
    """Add a ``review_flag`` history row unless the latest one names the same flags. True if added."""
    flags = [str(f) for f in flags]
    if not flags:
        return False
    for h in reversed(s.history):
        if isinstance(h, dict) and h.get("kind") == FLAG_KIND:
            if list(h.get("flags") or ()) == flags:
                return False
            break
    s.history.append({"kind": FLAG_KIND, "ts": now, "flags": flags})
    return True


def log_config() -> None:
    """Startup receipt on state producer and payment reader; report parsed lists."""
    for name in (HOLD_ENV, CLEAR_ENV):
        log.info("%s parsed: %s", name, ",".join(f"{h}@{r}" for h, r in parse(os.environ.get(name, ""))) or "(empty)")
