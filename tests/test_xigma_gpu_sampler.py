"""CuPy agreement with the NumPy quadrature and a public-engine smoke path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast, pytest.mark.gpu]

from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
from gammaforge.engines.xigma.stages import (
    angular_spectrum_from_table,
    deposit_shape_table,
    integrate_trajectories,
    retarget_ahat,
)
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.validation import scenarios

gpu = pytest.mark.skipif(not is_gpu_available(), reason="CuPy or CUDA GPU not available on host")


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


@gpu
def test_angular_spectrum_from_table_backend_cupy_supports_geometry(baseline_table):
    table, samples = baseline_table
    tx, ty, s = [0.0], [0.0], [0.6 * float(np.mean(samples.gamma) ** 2)]

    gpu_cube = angular_spectrum_from_table(
        table, tx, ty, s, backend="cupy", ellipticity=0.4, theta_xz=0.05, theta_yz=-0.03
    )
    cpu_cube = angular_spectrum_from_table(
        table, tx, ty, s, backend="numpy", ellipticity=0.4, theta_xz=0.05, theta_yz=-0.03
    )
    assert gpu_cube.shape == (1, 1, 1)
    assert np.all(np.isfinite(gpu_cube))
    assert np.all(gpu_cube >= 0.0)
    assert np.all(np.isfinite(cpu_cube))
    assert float(gpu_cube[0, 0, 0]) == pytest.approx(float(cpu_cube[0, 0, 0]), rel=0.1)


@gpu
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


@gpu
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
    assert results.model_specific["stage2_sampler"]["cdf_inversion"] == "exact_binary_search"
    assert results.model_specific["stage2_sampler"]["proposal_floor_fraction"] == 1e-3
    assert any("production-ready" in warning for warning in results.model_specific["warnings"])
