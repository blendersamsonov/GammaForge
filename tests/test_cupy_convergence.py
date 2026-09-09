from __future__ import annotations

import json

import numpy as np
import pytest

from gammaforge.engines.xigma.stages import Table
from gammaforge.validation.cupy_convergence import run_convergence_checks


def _table() -> Table:
    edges = np.linspace(-0.02, 0.02, 5)
    gamma = np.linspace(100.0, 140.0, 5)
    ahat = np.linspace(0.0, 0.08, 5)
    H = np.ones((4, 4, 4, 4))
    return Table(gamma, edges, edges, ahat, H, float(H.sum()), "test")


def _cube(table, x, y, s, **kwargs):
    return (1.0 + x[:, None, None] ** 2 + y[None, :, None] ** 2) * (1.0 + s[None, None, :] / 10_000.0)


def test_convergence_report_is_json_serializable_and_passes_with_matching_stub():
    report = run_convergence_checks(
        [{"name": "stub", "table": _table()}], cpu_runner=_cube, gpu_runner=_cube
    )
    assert report["pass"] is True
    assert len(report["checks"]) == 1 + 3 + 3 + 1 + 2
    json.dumps(report)


def test_gpu_unavailable_is_inconclusive_not_green_skip(monkeypatch):
    import gammaforge.engines.xigma.spectrum_sampler as sampler
    monkeypatch.setattr(sampler, "is_gpu_available", lambda: False)
    report = run_convergence_checks([{"name": "stub", "table": _table()}], cpu_runner=_cube, gpu_runner=None)
    assert report["pass"] is False
    gpu_checks = [check for check in report["checks"] if check["kind"] == "gpu"]
    assert gpu_checks and gpu_checks[0]["status"] == "inconclusive"


def test_gpu_schedule_includes_default_and_refinements():
    calls = []
    def gpu(table, x, y, s, **kwargs):
        calls.append((kwargs["rings"], kwargs["subsampling"]))
        return _cube(table, x, y, s)
    run_convergence_checks([{"name": "stub", "table": _table()}], cpu_runner=_cube, gpu_runner=gpu)
    assert {(32, 32), (32, 256), (64, 256), (64, 128)} <= set(calls)


def test_cpu_failure_inhibits_gpu():
    calls = []
    def bad(*args, **kwargs):
        raise RuntimeError("bad cpu")
    def gpu(*args, **kwargs):
        calls.append(1); return _cube(*args[:4])
    report = run_convergence_checks([{"name": "stub", "table": _table()}], cpu_runner=bad, gpu_runner=gpu)
    assert report["pass"] is False and not calls


def test_gpu_scale_error_is_failed():
    def gpu(table, x, y, s, **kwargs):
        return _cube(table, x, y, s) * 2.0
    report = run_convergence_checks([{"name": "stub", "table": _table()}], cpu_runner=_cube, gpu_runner=gpu)
    assert report["pass"] is False


def test_zero_reference_is_inconclusive():
    def zero(table, x, y, s, **kwargs):
        return np.zeros((x.size, y.size, s.size))

    report = run_convergence_checks([{"name": "zero", "table": _table()}], cpu_runner=zero, gpu_runner=zero)
    assert report["pass"] is False
    assert all(check["status"] == "inconclusive" for check in report["checks"])
