"""Scenario-bank integration checks for the experimental CuPy angular sampler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma.spectrum_sampler import calculate_angular_spectrum_gpu, is_gpu_available
from gammaforge.engines.xigma.stages import (
    angular_spectrum_from_table,
    deposit_shape_table,
    integrate_trajectories,
    retarget_ahat,
)
from gammaforge.io.interaction import SamplingSpec
from gammaforge.validation import scenarios


pytestmark = pytest.mark.skipif(not is_gpu_available(), reason="CuPy or CUDA GPU unavailable")


@pytest.fixture(scope="module")
def scenario_tables():
    tables = []
    for scenario in scenarios.SCENARIOS:
        interaction = scenarios.build(
            scenario,
            SamplingSpec(n_particles=4_000, seed=20260721, prefilter=1e-3),
        )
        samples = integrate_trajectories(
            interaction.bunch, interaction.laser, interaction.N_e, n_steps=32
        )
        shape = deposit_shape_table(samples, n_bins=(12, 12, 12, 12), scheme="cic")
        tables.append((scenario.name, retarget_ahat(shape, samples.intensity_peak), samples))
    return tables


def _integral(cube: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(np.trapezoid(cube, s, axis=2), y, axis=1), x))


def _refine_angular_quadrature(table, factor):
    """Sample the same clamped, bilinearly interpolated H at finer cell centers."""
    density = table.H
    new_edges = []
    for axis, edges in ((1, table.theta_x_edges), (2, table.theta_y_edges)):
        n = len(edges) - 1
        refined = np.linspace(edges[0], edges[-1], n * factor + 1)
        centers = 0.5 * (refined[:-1] + refined[1:])
        coordinate = np.clip((centers - edges[0]) / (edges[1] - edges[0]) - 0.5, 0, n - 1)
        lo = np.clip(np.floor(coordinate).astype(int), 0, n - 2)
        shape = [1] * density.ndim
        shape[axis] = centers.size
        weight = (coordinate - lo).reshape(shape)
        density = np.take(density, lo, axis=axis) * (1 - weight) + np.take(density, lo + 1, axis=axis) * weight
        new_edges.append(refined)
    return replace(table, H=density, theta_x_edges=new_edges[0], theta_y_edges=new_edges[1])


@pytest.mark.parametrize("psi_pol", [0.0, np.pi / 2.0], ids=["pol-x", "pol-y"])
def test_scenario_bank_gpu_mass_and_density_match_refined_cpu(scenario_tables, psi_pol):
    for name, table, samples in scenario_tables:
        gamma_max = float(np.max(samples.gamma))
        x = np.linspace(float(table.theta_x_edges[1]), float(table.theta_x_edges[-2]), 5)
        y = np.linspace(float(table.theta_y_edges[1]), float(table.theta_y_edges[-2]), 5)
        s = np.linspace(0.25 * gamma_max**2, 0.85 * gamma_max**2, 10)

        gpu = calculate_angular_spectrum_gpu(table, x, y, s, psi_pol=psi_pol, subsampling=256)
        # Refine the input angular integration, never the output observation grid.
        # Gamma/ahat axes and the interpolated physical density remain unchanged.
        cpu_coarse = angular_spectrum_from_table(
            _refine_angular_quadrature(table, 8), x, y, s, psi_pol=psi_pol, backend="numpy"
        )
        cpu_fine = angular_spectrum_from_table(
            _refine_angular_quadrature(table, 16), x, y, s, psi_pol=psi_pol, backend="numpy"
        )
        coarse_mass = _integral(cpu_coarse, x, y, s)
        reference = _integral(cpu_fine, x, y, s)
        assert reference > 0.0, name
        assert coarse_mass == pytest.approx(reference, rel=0.02), name
        assert _integral(np.abs(cpu_fine - cpu_coarse), x, y, s) / reference < 0.03, name
        gpu_mass = _integral(gpu, x, y, s)
        assert gpu_mass == pytest.approx(reference, rel=0.03), name

        # Density error is integrated on the common coarse grid, avoiding a pixelwise
        # norm while still catching displaced spectral/angular mass.
        l1 = _integral(np.abs(gpu - cpu_fine), x, y, s) / reference
        assert l1 < 0.05, f"{name}, psi={psi_pol}: integrated L1={l1:.3g}"
