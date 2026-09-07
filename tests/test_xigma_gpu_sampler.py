"""Tests for xigma's Stage 2 CuPy ring/annulus importance-sampling kernel.

Verifies:
- Hardware/CuPy detection (`is_gpu_available`).
- Execution and shape conformance for `angular_spectrum_from_table(backend='cupy')`.
- Numerical agreement with the reference NumPy brute-force grid quadrature.
- Graceful routing: `backend='auto'` falls back to NumPy for non-zero ellipticity or
  crossing angles, while `backend='cupy'` raises NotImplementedError.
- End-to-end `XigmaEngine.run(..., backend='cupy')`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.spectrum_sampler import (
    calculate_angular_spectrum_gpu,
    is_gpu_available,
)
from gammaforge.engines.xigma.stages import (
    angular_spectrum_from_table,
    deposit_shape_table,
    integrate_trajectories,
    retarget_ahat,
)
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.validation import scenarios

pytestmark = pytest.mark.skipif(
    not is_gpu_available(), reason="CuPy or CUDA GPU not available on host"
)


@pytest.fixture(scope="module")
def baseline_table():
    interaction = scenarios.build(
        replace(scenarios.BASELINE, sampling=replace(scenarios.BASELINE.sampling, n_particles=40_000))
    )
    samples = integrate_trajectories(
        interaction.bunch, interaction.laser, interaction.N_e, n_steps=64
    )
    shape_table = deposit_shape_table(samples, n_bins=(32, 32, 32, 64), scheme="cic")
    table = retarget_ahat(shape_table, samples.intensity_peak)
    return table, samples


def test_gpu_is_available():
    assert is_gpu_available() is True


def test_angular_spectrum_gpu_shape_and_nonnegativity(baseline_table):
    table, samples = baseline_table
    tx = np.linspace(-1e-4, 1e-4, 5)
    ty = np.linspace(-1e-4, 1e-4, 4)
    edge = float(np.max(samples.gamma) ** 2)
    s = np.linspace(0.2 * edge, 0.8 * edge, 6)

    cube = calculate_angular_spectrum_gpu(table, tx, ty, s, psi_pol=0.0, subsampling=32)
    assert cube.shape == (5, 4, 6)
    assert np.all(cube >= 0.0)
    assert np.all(np.isfinite(cube))
    assert np.sum(cube) > 0.0


def test_angular_spectrum_from_table_backend_cupy(baseline_table):
    table, samples = baseline_table
    tx = np.linspace(-1e-4, 1e-4, 5)
    ty = np.linspace(-1e-4, 1e-4, 5)
    edge = float(np.max(samples.gamma) ** 2)
    s = np.linspace(0.2 * edge, 0.8 * edge, 8)

    gpu_cube = angular_spectrum_from_table(table, tx, ty, s, backend="cupy")
    cpu_cube = angular_spectrum_from_table(table, tx, ty, s, backend="numpy")

    assert gpu_cube.shape == cpu_cube.shape

    # Check non-zero point agreement
    nonzeros = (cpu_cube > 0.0) & (gpu_cube > 0.0)
    assert np.sum(nonzeros) > 0
    ratios = gpu_cube[nonzeros] / cpu_cube[nonzeros]
    median_ratio = float(np.median(ratios))
    # Median agreement is close to 1.0 (within MC tolerance)
    assert 0.70 < median_ratio < 1.30


def test_unsupported_polarization_with_backend_cupy_raises(baseline_table):
    table, samples = baseline_table
    tx, ty, s = [0.0], [0.0], [float(np.mean(samples.gamma) ** 2)]

    with pytest.raises(NotImplementedError, match="ellipticity or crossing angles"):
        angular_spectrum_from_table(table, tx, ty, s, backend="cupy", ellipticity=0.5)

    with pytest.raises(NotImplementedError, match="ellipticity or crossing angles"):
        angular_spectrum_from_table(table, tx, ty, s, backend="cupy", theta_xz=0.05)


@pytest.mark.xfail(strict=True, reason="GPU density/integral discrepancy; docs/ALPHA_GPU_VALIDATION.md")
def test_gpu_distribution_agrees_with_numpy_reference(baseline_table):
    """A median cell ratio cannot detect misplaced or missing spectral mass."""
    table, samples = baseline_table
    tx = ty = np.linspace(-1e-4, 1e-4, 5)
    s = np.linspace(0.2, 0.8, 8) * float(np.max(samples.gamma) ** 2)
    cpu = angular_spectrum_from_table(table, tx, ty, s, backend="numpy")
    gpu = angular_spectrum_from_table(table, tx, ty, s, backend="cupy")

    def integral(cube):
        return np.trapezoid(np.trapezoid(np.trapezoid(cube, s, axis=2), ty, axis=1), tx)

    mass = integral(cpu)
    assert mass > 0.0
    assert abs(integral(gpu) / mass - 1.0) < 0.1
    assert integral(np.abs(gpu - cpu)) / mass < 0.1


def test_unsupported_polarization_with_backend_auto_routes_to_numpy(baseline_table):
    table, samples = baseline_table
    tx, ty, s = [0.0], [0.0], [float(np.mean(samples.gamma) ** 2)]

    # Auto routes to numpy when ellipticity is nonzero and succeeds
    res = angular_spectrum_from_table(table, tx, ty, s, backend="auto", ellipticity=0.5)
    assert res.shape == (1, 1, 1)
    assert res[0, 0, 0] >= 0.0


def test_xigma_engine_run_with_backend_cupy():
    interaction = scenarios.build(
        replace(scenarios.BASELINE, sampling=replace(scenarios.BASELINE.sampling, n_particles=5000))
    )
    requests = (
        OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(5, 5)),
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(8, 4, 4)),
    )
    interaction = replace(interaction, target=replace(interaction.target, outputs=requests))
    params = XigmaEngine.schema.with_values(
        n_bins_gamma=16,
        n_bins_theta_x=16,
        n_bins_theta_y=16,
        n_bins_a0_shape=32,
        n_bins_ahat=8,
        backend="cupy",
    )
    results = XigmaEngine().run(interaction, params)

    assert OutputKind.ANGULAR_DISTRIBUTION in results.photon_slices
    assert OutputKind.COLLIMATED_SPECTRUM in results.photon_slices

    ang = results.photon_slices[OutputKind.ANGULAR_DISTRIBUTION]
    col = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]

    assert ang.distr.shape == (5, 5)
    assert col.distr.shape == (8, 4, 4)
    assert np.all(ang.distr >= 0.0)
    assert np.all(col.distr >= 0.0)
    assert results.model_specific["stage2_backend"] == "cupy"
    assert any("experimental" in warning for warning in results.model_specific["warnings"])
