"""Small, deterministic regressions for the experimental CuPy angular sampler.

These tests deliberately use hand-built smooth tables.  The NumPy stage-2 quadrature is
the independent reference; no sampling or CDF implementation is reproduced here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast, pytest.mark.gpu]

from gammaforge.engines.xigma.spectrum_sampler import (
    calculate_angular_spectrum_gpu,
    is_gpu_available,
)
from gammaforge.engines.xigma.stages import Table, angular_spectrum_from_table


gpu = pytest.mark.skipif(not is_gpu_available(), reason="CuPy or CUDA GPU unavailable")


def _smooth_table(
    *, gamma0: float = 80.0, shifted: bool = False, thin: bool = False,
    theta_bins: int = 6, zero: bool = False,
) -> Table:
    """Create a positive smooth density with enough support for ring sampling."""
    gamma_edges = np.linspace(gamma0, gamma0 + 40.0, 5)
    if thin:
        x_edges = np.linspace(0.014, 0.018, theta_bins + 1)
        y_edges = np.linspace(-0.002, 0.002, theta_bins + 1)
    elif shifted:
        x_edges = np.linspace(0.012, 0.052, theta_bins + 1)
        y_edges = np.linspace(-0.035, 0.005, theta_bins + 1)
    else:
        x_edges = np.linspace(-0.03, 0.03, theta_bins + 1)
        y_edges = np.linspace(-0.03, 0.03, theta_bins + 1)
    ahat_edges = np.linspace(0.0, 0.08, 5)
    g = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    a = 0.5 * (ahat_edges[:-1] + ahat_edges[1:])
    G, X, Y, A = np.meshgrid(g, x, y, a, indexing="ij")
    H = np.zeros_like(G) if zero else 1.0 + 0.002 * (G - gamma0) + 2.0 * X + 1.5 * Y + 3.0 * A
    return Table(
        gamma_edges=gamma_edges,
        theta_x_edges=x_edges,
        theta_y_edges=y_edges,
        ahat_edges=ahat_edges,
        H=H,
        total_weight=float(np.sum(H)),
        scheme="synthetic-smooth",
    )


def _integral(cube: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(np.trapezoid(cube, s, axis=2), y, axis=1), x))


@gpu
def test_smooth_table_gpu_mass_and_density_match_converged_numpy_reference():
    # Refine the *input angular table* while preserving the same affine H and boundaries.
    # Both CPU references are evaluated on exactly the GPU output grid.
    coarse = _smooth_table(theta_bins=128)
    table = _smooth_table(theta_bins=256)
    x = y = np.linspace(-0.005, 0.005, 3)
    s = np.array([5_000., 7_000., 9_000., 11_000.])
    gpu = calculate_angular_spectrum_gpu(_smooth_table(theta_bins=12), x, y, s, subsampling=128)
    cpu_coarse = angular_spectrum_from_table(coarse, x, y, s, backend="numpy")
    cpu = angular_spectrum_from_table(table, x, y, s, backend="numpy")
    coarse_mass = _integral(cpu_coarse, x, y, s)
    reference = _integral(cpu, x, y, s)
    assert reference > 0.0
    assert reference == pytest.approx(coarse_mass, rel=0.02)
    assert _integral(np.abs(cpu - cpu_coarse), x, y, s) / reference < 0.02
    assert _integral(gpu, x, y, s) == pytest.approx(reference, rel=0.03)
    l1 = _integral(np.abs(gpu - cpu), x, y, s) / reference
    assert l1 < 0.05


@gpu
def test_gamma_10000_remains_finite():
    table = _smooth_table(gamma0=9_980.0)
    s = np.linspace(0.35 * 10_000.0**2, 0.85 * 10_000.0**2, 6)
    cube = calculate_angular_spectrum_gpu(table, [0.0], [0.0], s, subsampling=64)
    assert np.all(np.isfinite(cube))
    assert np.all(cube >= 0.0)
    assert float(np.sum(cube)) > 0.0


@gpu
@pytest.mark.parametrize("table", [_smooth_table(shifted=True), _smooth_table(thin=True)])
def test_shifted_and_thin_boxes_have_finite_positive_mass(table):
    x = np.linspace(float(table.theta_x_edges[0]), float(table.theta_x_edges[-1]), 5)
    y = np.linspace(float(table.theta_y_edges[0]), float(table.theta_y_edges[-1]), 5)
    s = np.linspace(0.3 * 120.0**2, 0.8 * 120.0**2, 8)
    gpu = calculate_angular_spectrum_gpu(table, x, y, s, subsampling=128)
    assert np.all(np.isfinite(gpu))
    assert np.all(gpu >= 0.0)
    assert float(np.sum(gpu)) > 0.0
