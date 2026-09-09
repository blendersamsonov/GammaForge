"""The supported dependency and local-check contracts remain explicit."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]


ROOT = Path(__file__).resolve().parents[1]


def test_numpy_floor_covers_the_trapezoid_api_and_test_tiers_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "numpy>=2.0" in project["dependencies"]
    assert {"gui", "browser", "symbolic", "dev"} <= set(project["optional-dependencies"])


def test_make_check_is_the_documented_local_core_entry_point():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "$(PYTHON) -m pytest" in makefile
    assert "make check" in readme
    assert (ROOT / "requirements/developer.lock").is_file()
