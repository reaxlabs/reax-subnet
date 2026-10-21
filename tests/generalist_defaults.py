"""The committed fail-closed generalist defaults, read from ``scripts/set_generalist_start.py`` (``default`` resets the
build to them), so tests do not depend on the release constants that are set in ``competition_split.py``."""
from __future__ import annotations

import importlib.util
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "set_generalist_start.py"
_NAMES = ("GENERALIST_START_UTC", "GENERALIST_COMMIT_FROM_UTC", "GENERALIST_ENTRY_FLOOR",
          "GENERALIST_KEEPALIVE_NONPROC_FLOOR", "GENERALIST_G_MED")


def _script():
    spec = importlib.util.spec_from_file_location("set_generalist_start_for_tests", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_source() -> str:
    """competition_split.py with the script's fail-closed defaults applied."""
    mod = _script()
    text = (ROOT / "src" / "reax_subnet" / "competition_split.py").read_text()
    for name, (ann, line) in mod.build(dict(mod.DEFAULTS)).items():
        text, n = re.subn(rf"^{name}{ann} ?= .*$", line, text, flags=re.M)
        assert n == 1, name
    return text


def defaults() -> dict:
    """name -> value of the five generalist constants under the fail-closed defaults."""
    text = default_source()
    env = {"datetime": datetime, "timezone": timezone, "timedelta": timedelta, "math": math, "Optional": Optional}
    out = {}
    for name in _NAMES:
        m = re.search(rf"^{name}(?:: [^=]+)? = (.*)$", text, flags=re.M)
        out[name] = eval(m.group(1), env)
    return out
