"""Gen worker and validator must ship byte-identical shared tables (slot list, commit cutoff, keep-alive base floors,
spec/machine.py with FAST_CONFIRM_*). This file is identical on both release branches: SHARED_DIGEST pins the shared
source, so a change on one side only fails here. Changing a shared table: update SHARED_DIGEST on BOTH branches and run
``python scripts/check_shared_tables.py <worker checkout> <validator checkout>``."""
import hashlib
import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_DIGEST = "5b09e5ec10e473e6953616b23148d723541d35e2ed02c3f48582101fb64089e7"


def _checker():
    spec = importlib.util.spec_from_file_location("check_shared_tables", ROOT / "scripts" / "check_shared_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _digest(c, root: Path) -> str:
    seg = c.segments(root / c.SPLIT)
    h = hashlib.sha256()
    for name in c.SHARED_NAMES:
        h.update(name.encode() + b"\0" + seg[name].encode() + b"\0")
    h.update((root / c.MACHINE).read_bytes())
    return h.hexdigest()


def test_shared_tables_pinned():
    assert _digest(_checker(), ROOT) == SHARED_DIGEST


def test_gemma_marker_inside_base_floor():
    seg = _checker().segments(ROOT / "src/reax_subnet/competition_split.py")["KEEPALIVE_BASE_FLOOR"]
    # 29 Oct: the Gemma rows were filled from workers/floor-calib (KnowLine, Turion, Jeffan; Sieve does not serve on vLLM 0.30)
    for h in ("8e41636d16fa5874685a38696ee8a5346fe0c03998dc13343d077617ec6034c1", "c910628ca23a605b6b2de3b2dabf104140326adf882aa7801f627dd8b1414b7e",
              "7c8a0695c85afa67f52f834216337990acfa1f42c7db9cc4583c90bb809431f8"):
        assert h in seg


def test_checker_detects_difference(tmp_path):
    c = _checker()
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / c.MACHINE).parent.mkdir(parents=True)
        shutil.copy(ROOT / c.SPLIT, d / c.SPLIT)
        shutil.copy(ROOT / c.MACHINE, d / c.MACHINE)
    assert c.diff(a, b) == []
    src = (b / c.SPLIT).read_text().replace("{'generalist': 39.23}", "{'generalist': 39.24}", 1)
    (b / c.SPLIT).write_text(src)
    (b / c.MACHINE).write_text((b / c.MACHINE).read_text().replace("FAST_CONFIRM_DELAY_S = 2 * 3600",
                                                                   "FAST_CONFIRM_DELAY_S = 3 * 3600"))
    assert c.diff(a, b) == ["KEEPALIVE_BASE_FLOOR: source differs", "spec/machine.py differs"]
