"""Permanent CUDA smoke gate for the public crossed-Gaussian Xigma path."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios


gpu = pytest.mark.skipif(not is_gpu_available(), reason="CuPy or CUDA GPU unavailable")


@pytest.fixture(scope="module")
def crossed_interaction():
    laser = replace(
        scenarios.BASELINE.laser,
        theta_xz=Quantity(0.3, "rad"),
        theta_yz=Quantity(0.2, "rad"),
        ellipticity=0.4,
        psi_pol=Quantity(0.37, "rad"),
    )
    target = replace(
        scenarios.BASELINE.target,
        theta_x_col=Quantity(0.0003, "rad"),
        theta_y_col=Quantity(0.0003, "rad"),
        outputs=(OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(9, 9, 16)),),
    )
    return scenarios.build(
        replace(scenarios.BASELINE, laser=laser, target=target),
        SamplingSpec(n_particles=4_000, seed=20260721, prefilter=1e-3),
    )


@gpu
@pytest.mark.parametrize("rings,n_steps", [(32, 32), (64, 64)])
def test_public_engine_crossed_release_smoke(crossed_interaction, rings, n_steps):
    params = XigmaEngine.schema.with_values(
        n_steps=n_steps,
        n_bins_gamma=12,
        n_bins_theta_x=12,
        n_bins_theta_y=12,
        n_bins_a0_shape=12,
        n_bins_ahat=12,
        backend="cupy",
        sampler_rings=rings,
        sampler_subsampling=32,
    )
    results = XigmaEngine().run(crossed_interaction, params)
    slice_ = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    assert slice_.distr.shape == (9, 9, 16)
    assert np.all(np.isfinite(slice_.distr))
    assert np.all(slice_.distr >= 0.0)
    assert np.any(slice_.distr > 0.0)
    assert results.model_specific["stage2_backend"] == "cupy"
    metadata = results.model_specific["stage2_sampler"]
    assert metadata["rings"] == rings
    assert metadata["subsampling"] == 32


def test_explicit_cupy_missing_device_contract(monkeypatch):
    from gammaforge.engines.xigma import spectrum_sampler
    from gammaforge.engines.xigma.stages import stage2_backend

    monkeypatch.setattr(spectrum_sampler, "is_gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        stage2_backend("cupy", ellipticity=0.4, theta_xz=0.3, theta_yz=0.2)


def test_release_runner_reports_failure_instead_of_all_skipped(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_cupy_release.py"
    spec = importlib.util.spec_from_file_location("validate_cupy_release", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    monkeypatch.setattr(runner, "_cuda_info", lambda: (_ for _ in ()).throw(
        runner.ReleaseGateError("simulated missing CUDA")
    ))
    output = tmp_path / "release.json"
    assert runner.main(["--output", str(output)]) == 1
    assert '"status": "failed"' in output.read_text(encoding="utf-8")
    assert "simulated missing CUDA" in output.read_text(encoding="utf-8")


def test_convergence_report_requires_true_pass(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_cupy_release.py"
    spec = importlib.util.spec_from_file_location("validate_cupy_release_report", path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr("gammaforge.validation.cupy_convergence.run_convergence_checks", lambda: {"pass": False, "checks": [{"kind": "cpu"}]})
    report = runner._convergence_report()
    assert report["status"] == "failed" and report["result"]["checks"]

    monkeypatch.setattr("gammaforge.validation.cupy_convergence.run_convergence_checks", lambda: {"checks": []})
    assert runner._convergence_report()["status"] == "failed"


def test_release_fingerprint_change_fails_without_discarding_runs(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_cupy_release.py"
    spec = importlib.util.spec_from_file_location("validate_cupy_release_stub", path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    import types
    import gammaforge.engines.xigma.engine as engine
    monkeypatch.setattr(runner, "_cuda_info", lambda: {"device": "stub"})
    monkeypatch.setattr("gammaforge.engines.xigma.spectrum_sampler.is_gpu_available", lambda: True)
    monkeypatch.setattr(runner, "_convergence_report", lambda: {"status": "passed", "result": {"pass": True, "checks": []}})
    monkeypatch.setattr(runner, "_source_fingerprints", lambda: {"x": "stable"})
    from gammaforge.io.target import OutputKind
    fake = types.SimpleNamespace(photon_slices={OutputKind.COLLIMATED_SPECTRUM: types.SimpleNamespace(distr=np.ones((9, 9, 16)))}, model_specific={"stage2_backend": "cupy", "stage2_sampler": {"rings": 32, "subsampling": 32}})
    monkeypatch.setattr(engine.XigmaEngine, "run", lambda *a, **k: fake)
    report = runner.run_release(rings=(32,))
    assert report["status"] == "passed" and report["runs"]

    values = iter(({"sampler": "old"}, {"sampler": "new"}))
    monkeypatch.setattr(runner, "_source_fingerprints", lambda: next(values))
    changed = runner.run_release(rings=(32,))
    assert changed["status"] == "failed" and changed["source_unchanged"] is False
    assert changed["runs"] and "source files changed" in changed["error"]
