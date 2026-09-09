"""The UI can call the public runner but cannot reach engine implementation code."""

import ast
from importlib.util import find_spec, resolve_name
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]


ROOT = Path(__file__).resolve().parents[1]


def forbidden_imports(source: str, package: str) -> list[str]:
    forbidden = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            module = resolve_name(name, package) if node.level else name
            names = [module]
            if module == "gammaforge":
                names += [f"{module}.{alias.name}" for alias in node.names]
        else:
            continue
        forbidden += [name for name in names
                      if (name == "gammaforge.engines" or name.startswith("gammaforge.engines."))
                      and name not in {"gammaforge.engines.base", "gammaforge.engines.runner"}]
    return forbidden


def test_gui_engine_import_boundary():
    failures = []
    for path in (ROOT / "src/gammaforge/gui").rglob("*.py"):
        package = ".".join(path.parent.relative_to(ROOT / "src").parts)
        failures.extend(f"{path.name}: {name}" for name in forbidden_imports(path.read_text(), package))
    assert not failures, failures


def test_boundary_rejects_relative_and_absolute_engine_internals():
    for source in ("from ..engines.xigma.collision import Collision",
                   "import gammaforge.engines.xigma.stages",
                   "from gammaforge import engines"):
        assert forbidden_imports(source, "gammaforge.gui")
    assert not forbidden_imports("from ..engines.runner import LocalRunner", "gammaforge.gui")


def test_importing_gui_does_not_start_or_import_web_server():
    subprocess.run([sys.executable, "-c",
                    "import sys; import gammaforge.gui; assert 'nicegui' not in sys.modules"],
                   check=True, cwd=ROOT, timeout=10)


def test_gui_extra_imports_when_installed():
    if find_spec("nicegui") is None:
        pytest.skip("requires the gui extra (nicegui)")
    from nicegui import ui

    assert ui is not None
