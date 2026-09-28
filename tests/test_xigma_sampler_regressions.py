"""Small, deterministic regressions for the experimental CuPy angular sampler.

These tests deliberately use hand-built smooth tables.  The NumPy stage-2 quadrature is
the independent reference; no sampling or CDF implementation is reproduced here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast, pytest.mark.gpu]

from gammaforge.engines.xigma.spectrum_sampler import (
    calculate_angular_spectral_moments_gpu,
    calculate_angular_spectrum_gpu,
    is_gpu_available,
)
from gammaforge.engines.xigma.stages import (
    SpectralMoments,
    Table,
    angular_spectrum_from_table,
    query_spectral_moments,
    reconstruct_second_order,
)


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
    H = H[..., None]
    return Table(
        gamma_edges=gamma_edges,
        theta_x_edges=x_edges,
        theta_y_edges=y_edges,
        ahat_edges=ahat_edges,
        chirp_edges=np.array([0.5, 1.5]),
        H=H,
        H_var_a=np.zeros_like(H),
        H_var_chirp=np.zeros_like(H),
        H_cov_a_chirp=np.zeros_like(H),
        total_weight=float(np.sum(H)),
        scheme="synthetic-smooth",
    )


def _integral(cube: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(np.trapezoid(cube, s, axis=2), y, axis=1), x))


def _moment_table(chirp_bins: int, theta_bins: int) -> Table:
    gamma_edges = np.linspace(900.0, 1100.0, 9)
    x_edges = np.linspace(-0.004, 0.004, theta_bins + 1)
    y_edges = np.linspace(-0.004, 0.004, theta_bins + 1)
    ahat_edges = np.array([0.0, 0.01, 0.04, 0.1])
    chirp_edges = np.linspace(0.85, 1.15, chirp_bins + 1)
    g = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    a = 0.5 * (ahat_edges[:-1] + ahat_edges[1:])
    c = 0.5 * (chirp_edges[:-1] + chirp_edges[1:])
    G, X, Y, A, C = np.meshgrid(g, x, y, a, c, indexing="ij")
    H = 2.0 + 0.001 * (G - 900.0) + 4.0 * X - 3.0 * Y + 2.0 * A + 0.5 * C
    return Table(
        gamma_edges=gamma_edges,
        theta_x_edges=x_edges,
        theta_y_edges=y_edges,
        ahat_edges=ahat_edges,
        chirp_edges=chirp_edges,
        H=H,
        H_var_a=0.01 * H,
        H_var_chirp=0.05 * H,
        H_cov_a_chirp=0.018 * H,
        total_weight=float(np.sum(H)),
        scheme="synthetic-5d-moments",
    )


def _cpu_moment_cubes(table, x, y, s, **geometry):
    outputs = [np.empty((len(x), len(y), len(s))) for _ in range(3)]
    for i, theta_x in enumerate(x):
        for j, theta_y in enumerate(y):
            moments = query_spectral_moments(
                table, theta_x, theta_y, s, **geometry
            )
            outputs[0][i, j] = moments.rho0
            outputs[1][i, j] = moments.rho1
            outputs[2][i, j] = moments.rho2
    return tuple(outputs)


@gpu
@pytest.mark.parametrize("chirp_bins", [1, 3])
def test_five_dimensional_raw_moments_match_cpu_with_exact_crossed_geometry(chirp_bins):
    gpu_table = _moment_table(chirp_bins, theta_bins=16)
    cpu_table = _moment_table(chirp_bins, theta_bins=128)
    x = np.linspace(-4e-4, 4e-4, 3)
    y = np.linspace(-4e-4, 4e-4, 3)
    s = np.linspace(0.45e6, 0.9e6, 7)
    geometry = dict(psi_pol=0.37, ellipticity=0.4, theta_xz=0.08, theta_yz=-0.05)
    gpu_moments = calculate_angular_spectral_moments_gpu(
        gpu_table, x, y, s, rings=64, subsampling=256, **geometry
    )
    cpu_moments = _cpu_moment_cubes(cpu_table, x, y, s, **geometry)

    for actual, expected in zip(gpu_moments, cpu_moments):
        scale = _integral(np.abs(expected), x, y, s)
        assert scale > 0.0
        assert _integral(np.abs(actual - expected), x, y, s) / scale < 0.1
    assert _integral(gpu_moments[1], x, y, s) < 0.0


@gpu
def test_moment_channels_do_not_change_the_positive_base_proposal(monkeypatch):
    from dataclasses import replace

    table = _moment_table(chirp_bins=3, theta_bins=16)
    zero_moments = replace(
        table,
        H_var_a=np.zeros_like(table.H),
        H_var_chirp=np.zeros_like(table.H),
        H_cov_a_chirp=np.zeros_like(table.H),
    )
    kwargs = dict(theta_xz=0.08, theta_yz=-0.05, rings=32, subsampling=64)
    first = calculate_angular_spectral_moments_gpu(
        table, [0.0], [0.0], [0.5e6, 0.7e6], **kwargs
    )
    second = calculate_angular_spectral_moments_gpu(
        zero_moments, [0.0], [0.0], [0.5e6, 0.7e6], **kwargs
    )
    # Float atomic-add order can shift by a few ulps as the other output channels execute.
    np.testing.assert_allclose(first[0], second[0], rtol=1e-6, atol=0.0)
    np.testing.assert_array_equal(second[1], np.zeros_like(second[1]))
    np.testing.assert_array_equal(second[2], np.zeros_like(second[2]))

    s = np.array([0.4e6, 0.5e6, 0.6e6, 0.7e6, 0.8e6])
    raw = calculate_angular_spectral_moments_gpu(
        table, [0.0], [0.0], s, **kwargs
    )
    expected = reconstruct_second_order(SpectralMoments(
        s=s, rho0=raw[0][0, 0], rho1=raw[1][0, 0], rho2=raw[2][0, 0]
    ))
    import gammaforge.engines.xigma.spectrum_sampler as sampler
    monkeypatch.setattr(sampler, "calculate_angular_spectral_moments_gpu", lambda *a, **k: raw)
    actual = sampler.calculate_angular_spectrum_gpu(
        table, [0.0], [0.0], s, line_model="moment2", **kwargs
    )
    np.testing.assert_array_equal(actual[0, 0], expected)


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
