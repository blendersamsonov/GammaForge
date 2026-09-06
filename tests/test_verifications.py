"""The symbolic-verification wrapper must turn claimed failures into process failures."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts" / "verifications"


def _wrapper():
    return runpy.run_path(str(SCRIPTS / "verify_all.py"), run_name="verification_wrapper")


def test_a_false_symbolic_identity_fails_the_verifier(tmp_path):
    source = (SCRIPTS / "verify_der004.py").read_text(encoding="utf-8")
    wrong = tmp_path / "wrong_der004.py"
    wrong.write_text(source.replace("target = 1 -", "target = 2 -", 1), encoding="utf-8")

    result = subprocess.run([sys.executable, str(wrong)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not match its boxed formula" in result.stderr


def test_wrapper_propagates_a_failing_child_process(tmp_path, capsys):
    failing = tmp_path / "fails.py"
    failing.write_text("raise RuntimeError('expected child failure')\n", encoding="utf-8")

    assert not _wrapper()["run_verification"](failing, "deliberately failing verifier")
    assert "FAILED (exit code 1)" in capsys.readouterr().out


def test_wrapper_selects_a_derivation_by_name_and_rejects_unknown_selector(capsys):
    wrapper = _wrapper()
    assert wrapper["main"](["unknown"]) == 1
    assert "Available: der004, der005, der006, der007" in capsys.readouterr().out

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_all.py"), "der004"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "DER004: Ellipticity in polarization factor" in result.stdout

    upper = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_all.py"), "DER004"], capture_output=True, text=True
    )
    assert upper.returncode == 0, upper.stderr
    assert "DER004: Ellipticity in polarization factor" in upper.stdout


def test_wrapper_rejects_more_than_one_selector(capsys):
    assert _wrapper()["main"](["der004", "der005"]) == 2
    assert "usage: verify_all.py" in capsys.readouterr().out


def test_wrapper_marks_the_author_approved_der006_application_as_passed():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_all.py"), "der006"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "PASS: DER006" in result.stdout
    assert "BLOCKED" not in result.stdout
