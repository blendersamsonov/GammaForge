"""Scenario-bank integration checks for the experimental CuPy angular sampler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma.spectrum_sampler import calculate_angular_spectrum_gpu, is_gpu_available
from gammaforge.engines.xigma.stages import (
    Table,
    angular_spectrum_from_table,
    deposit_shape_table,
    integrate_trajectories,
    retarget_ahat,
)
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios


gpu = pytest.mark.skipif(not is_gpu_available(), reason="CuPy or CUDA GPU unavailable")
pytestmark = [pytest.mark.tier3, pytest.mark.heavy]


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
@gpu
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


@pytest.fixture(scope="module")
def crossed_gaussian_table():
    """A real deposited Gaussian table with both crossing planes populated."""
    crossed = scenarios.build(
        replace(
            scenarios.BASELINE,
            laser=replace(
                scenarios.BASELINE.laser,
                theta_xz=Quantity(0.02, "rad"),
                theta_yz=Quantity(-0.015, "rad"),
                ellipticity=0.4,
                psi_pol=Quantity(0.37, "rad"),
            ),
        ),
        SamplingSpec(n_particles=4_000, seed=20260721, prefilter=1e-3),
    )
    samples = integrate_trajectories(
        crossed.bunch, crossed.laser, crossed.N_e, n_steps=32
    )
    shape = deposit_shape_table(samples, n_bins=(12, 12, 12, 12), scheme="cic")
    return retarget_ahat(shape, samples.intensity_peak), samples


@gpu
def test_crossed_gaussian_table_matches_refined_cpu(crossed_gaussian_table):
    table, samples = crossed_gaussian_table
    gamma_max = float(np.max(samples.gamma))
    x = np.linspace(float(table.theta_x_edges[1]), float(table.theta_x_edges[-2]), 5)
    y = np.linspace(float(table.theta_y_edges[1]), float(table.theta_y_edges[-2]), 5)
    s = np.linspace(0.25 * gamma_max**2, 0.85 * gamma_max**2, 10)
    kwargs = dict(
        psi_pol=0.37, ellipticity=0.4, theta_xz=0.02, theta_yz=-0.015,
    )
    gpu_cube = calculate_angular_spectrum_gpu(table, x, y, s, subsampling=256, **kwargs)
    cpu_coarse = angular_spectrum_from_table(
        _refine_angular_quadrature(table, 8), x, y, s, backend="numpy", **kwargs
    )
    cpu_fine = angular_spectrum_from_table(
        _refine_angular_quadrature(table, 16), x, y, s, backend="numpy", **kwargs
    )
    coarse_mass = _integral(cpu_coarse, x, y, s)
    reference = _integral(cpu_fine, x, y, s)
    assert reference > 0.0
    assert coarse_mass == pytest.approx(reference, rel=0.03)
    assert _integral(np.abs(cpu_fine - cpu_coarse), x, y, s) / reference < 0.05
    assert _integral(gpu_cube, x, y, s) == pytest.approx(reference, rel=0.03)
    assert _integral(np.abs(gpu_cube - cpu_fine), x, y, s) / reference < 0.05


def _nonuniform_ahat_table() -> Table:
    gamma_edges = np.linspace(1800.0, 2200.0, 9)
    # Keep the input box resolved against the relativistic annulus (1/gamma), while
    # retaining enough cells for the bilinear interpolation/refinement check.
    x_edges = np.linspace(-0.002, 0.002, 25)
    y_edges = np.linspace(-0.002, 0.002, 25)
    ahat_edges = np.array([0.0, 0.001, 0.003, 0.008, 0.02, 0.05, 0.09])
    g = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    a = 0.5 * (ahat_edges[:-1] + ahat_edges[1:])
    G, X, Y, A = np.meshgrid(g, x, y, a, indexing="ij")
    H = 2.0 + 0.002 * (G - 1800.0) + 2.5 * X - 1.5 * Y + 4.0 * A
    return Table(
        gamma_edges=gamma_edges, theta_x_edges=x_edges, theta_y_edges=y_edges,
        ahat_edges=ahat_edges, H=H, total_weight=float(np.sum(H)), scheme="synthetic-nonuniform-ahat",
    )


@gpu
@pytest.mark.parametrize(
    "kwargs",
    [
        {"psi_pol": 0.37, "ellipticity": 0.4, "theta_xz": 0.0, "theta_yz": 0.0},
        {"psi_pol": 0.37, "ellipticity": 1.0, "theta_xz": 0.0, "theta_yz": 0.0},
        {"psi_pol": 0.37, "ellipticity": 0.4, "theta_xz": 0.02, "theta_yz": -0.015},
    ],
    ids=["head-on-elliptical", "head-on-circular", "crossed-elliptical"],
)
def test_synthetic_nonuniform_ahat_table_matches_refined_cpu(kwargs):
    table = _nonuniform_ahat_table()
    x = np.linspace(-0.0008, 0.0008, 5)
    y = np.linspace(-0.0008, 0.0008, 5)
    s = np.linspace(0.25 * 2000.0**2, 0.85 * 2000.0**2, 10)
    gpu_cube = calculate_angular_spectrum_gpu(table, x, y, s, subsampling=256, **kwargs)
    cpu_coarse = angular_spectrum_from_table(
        _refine_angular_quadrature(table, 8), x, y, s, backend="numpy", **kwargs
    )
    cpu_fine = angular_spectrum_from_table(
        _refine_angular_quadrature(table, 16), x, y, s, backend="numpy", **kwargs
    )
    reference = _integral(cpu_fine, x, y, s)
    assert reference > 0.0
    assert _integral(cpu_coarse, x, y, s) == pytest.approx(reference, rel=0.03)
    assert _integral(np.abs(cpu_fine - cpu_coarse), x, y, s) / reference < 0.05
    assert _integral(gpu_cube, x, y, s) == pytest.approx(reference, rel=0.03)
    assert _integral(np.abs(gpu_cube - cpu_fine), x, y, s) / reference < 0.05
