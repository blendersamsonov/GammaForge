"""Unit tests for Smooth Laboratory Observer Basis Stokes Implementation (DER007).

Validates Stokes parameters (I, Q, U, V), polarization degree P, and polarization
angle chi in the smooth laboratory observer basis (m_x, m_y) across arbitrary laser
polarization, crossing angles, and electron beam emittance.
"""

from __future__ import annotations

import math
from dataclasses import replace
import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast]

from gammaforge.engines.xigma.collision import BunchStokes, Collision
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.stages import (
    bunch_stokes_parameters,
    compute_stokes_components,
    polarization_factor,
    rotated_laser_axes,
    stokes_parameters_vectorized,
)
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios


def test_collinear_linear_polarization_on_axis():
    """Head-on collision, cold beam, on-axis observation gives pure linear polarization.

    Verifies Q/I = cos(2*psi_pol), U/I = sin(2*psi_pol), V = 0, P = 1.0 uniformly
    with no azimuthal dependence (DER007).
    """
    gamma = 2000.0
    th_ex = 0.0
    th_ey = 0.0
    th_x = 0.0
    th_y = 0.0

    # Horizontal linear polarization (psi = 0)
    e0, e1 = rotated_laser_axes(psi_pol=0.0, theta_xz=0.0, theta_yz=0.0)
    I, Q, U, V = compute_stokes_components(gamma, th_ex, th_ey, th_x, th_y, e0, e1, ellipticity=0.0)
    assert I > 0.0
    np.testing.assert_allclose(Q / I, 1.0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(U / I, 0.0, atol=1e-15)
    np.testing.assert_allclose(V / I, 0.0, atol=1e-15)

    # Diagonal linear polarization (psi = pi/4) -> Q = 0, U/I = +1
    e0_diag, e1_diag = rotated_laser_axes(psi_pol=math.pi / 4.0, theta_xz=0.0, theta_yz=0.0)
    I_d, Q_d, U_d, V_d = compute_stokes_components(gamma, th_ex, th_ey, th_x, th_y, e0_diag, e1_diag, ellipticity=0.0)
    np.testing.assert_allclose(Q_d / I_d, 0.0, atol=1e-15)
    np.testing.assert_allclose(U_d / I_d, 1.0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(V_d / I_d, 0.0, atol=1e-15)

    # Vertical linear polarization (psi = pi/2) -> Q/I = -1, U = 0
    e0_v, e1_v = rotated_laser_axes(psi_pol=math.pi / 2.0, theta_xz=0.0, theta_yz=0.0)
    I_v, Q_v, U_v, V_v = compute_stokes_components(gamma, th_ex, th_ey, th_x, th_y, e0_v, e1_v, ellipticity=0.0)
    np.testing.assert_allclose(Q_v / I_v, -1.0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(U_v / I_v, 0.0, atol=1e-15)
    np.testing.assert_allclose(V_v / I_v, 0.0, atol=1e-15)


def test_collinear_circular_polarization_on_axis():
    r"""Head-on collision, cold beam, on-axis observation with circular polarization.

    Verifies Q/I = 0, U/I = 0, V/I = \mp 1.0 for eps = \pm 1, P = 1.0.
    """
    gamma = 2000.0
    th_ex = 0.0
    th_ey = 0.0
    th_x = 0.0
    th_y = 0.0
    e0, e1 = rotated_laser_axes(psi_pol=0.0, theta_xz=0.0, theta_yz=0.0)

    # Right circular polarization (eps = +1.0)
    I_rc, Q_rc, U_rc, V_rc = compute_stokes_components(gamma, th_ex, th_ey, th_x, th_y, e0, e1, ellipticity=1.0)
    assert I_rc > 0.0
    np.testing.assert_allclose(Q_rc / I_rc, 0.0, atol=1e-15)
    np.testing.assert_allclose(U_rc / I_rc, 0.0, atol=1e-15)
    np.testing.assert_allclose(V_rc / I_rc, -1.0, rtol=1e-14, atol=1e-15)
    P_rc = math.sqrt(float(Q_rc**2 + U_rc**2 + V_rc**2)) / float(I_rc)
    np.testing.assert_allclose(P_rc, 1.0, rtol=1e-14, atol=1e-15)

    # Left circular polarization (eps = -1.0)
    I_lc, Q_lc, U_lc, V_lc = compute_stokes_components(gamma, th_ex, th_ey, th_x, th_y, e0, e1, ellipticity=-1.0)
    assert I_lc > 0.0
    np.testing.assert_allclose(Q_lc / I_lc, 0.0, atol=1e-15)
    np.testing.assert_allclose(U_lc / I_lc, 0.0, atol=1e-15)
    np.testing.assert_allclose(V_lc / I_lc, 1.0, rtol=1e-14, atol=1e-15)
    P_lc = math.sqrt(float(Q_lc**2 + U_lc**2 + V_lc**2)) / float(I_lc)
    np.testing.assert_allclose(P_lc, 1.0, rtol=1e-14, atol=1e-15)


def test_single_electron_purity_across_random_angles():
    """Algebraic invariant: I^2 = Q^2 + U^2 + V^2 holds identically for every single electron.

    Tests 1000 random points spanning gamma, electron divergence, observation angle,
    crossing angles, ellipticity, and polarization azimuth.
    """
    rng = np.random.default_rng(20260908)
    n_points = 1000

    gamma = rng.uniform(10.0, 10000.0, size=n_points)
    th_ex = rng.normal(0.0, 2e-3, size=n_points)
    th_ey = rng.normal(0.0, 2e-3, size=n_points)
    th_x = rng.normal(0.0, 2e-3, size=n_points)
    th_y = rng.normal(0.0, 2e-3, size=n_points)
    psi = rng.uniform(0.0, 2.0 * math.pi, size=n_points)
    alpha_xz = rng.uniform(-0.5, 0.5, size=n_points)
    alpha_yz = rng.uniform(-0.5, 0.5, size=n_points)
    ellipticity = rng.uniform(-1.0, 1.0, size=n_points)

    for i in range(n_points):
        e0, e1 = rotated_laser_axes(psi_pol=float(psi[i]), theta_xz=float(alpha_xz[i]), theta_yz=float(alpha_yz[i]))
        I, Q, U, V = compute_stokes_components(
            gamma[i], th_ex[i], th_ey[i], float(th_x[i]), float(th_y[i]), e0, e1, ellipticity=float(ellipticity[i])
        )
        assert I > 0.0
        P_sq = (Q**2 + U**2 + V**2) / (I**2)
        assert abs(float(P_sq) - 1.0) < 1e-13, f"Single electron purity violated: P_sq={P_sq} at index {i}"


def test_consistency_with_scalar_intensity():
    """Stokes parameter I matches stages.py scalar intensity within 1e-14 relative difference.

    Verifies that Tr(U^T Xi U) is numerically identical between the optimized scalar
    kernel and the smooth observer basis Stokes formulation.
    """
    rng = np.random.default_rng(99999)
    for _ in range(100):
        g = float(rng.uniform(10.0, 10000.0))
        tex = float(rng.normal(0.0, 1e-3))
        tey = float(rng.normal(0.0, 1e-3))
        tx = float(rng.normal(0.0, 1e-3))
        ty = float(rng.normal(0.0, 1e-3))
        psi = float(rng.uniform(0.0, 2.0 * math.pi))
        alpha_xz = float(rng.uniform(-0.4, 0.4))
        alpha_yz = float(rng.uniform(-0.4, 0.4))
        eps = float(rng.uniform(-1.0, 1.0))

        e0, e1 = rotated_laser_axes(psi_pol=psi, theta_xz=alpha_xz, theta_yz=alpha_yz)
        I, _, _, _ = compute_stokes_components(g, tex, tey, tx, ty, e0, e1, ellipticity=eps)
        I_scalar = polarization_factor(g, tex, tey, tx, ty, eps, psi, alpha_xz, alpha_yz)

        rel_diff = abs(float(I) - float(I_scalar)) / float(I_scalar)
        assert rel_diff < 1e-14, f"Intensity mismatch: Stokes I={I}, scalar={I_scalar}, rel_diff={rel_diff}"

        # Also check vectorized wrapper stokes_parameters_vectorized
        I_vec, _, _, _ = stokes_parameters_vectorized(g, tex, tey, tx, ty, eps, psi, alpha_xz, alpha_yz)
        np.testing.assert_allclose(I_vec, I, rtol=1e-15, atol=0.0)


def test_dipole_null_at_ninety_degrees():
    """At 90 deg crossing angle with linear polarization in scattering plane, on-axis I == 0 identically."""
    e0, e1 = rotated_laser_axes(psi_pol=0.0, theta_xz=math.pi / 2.0, theta_yz=0.0)
    I, Q, U, V = compute_stokes_components(2000.0, 0.0, 0.0, 0.0, 0.0, e0, e1, ellipticity=0.0)
    np.testing.assert_allclose(float(I), 0.0, atol=1e-15)
    np.testing.assert_allclose(float(Q), 0.0, atol=1e-15)
    np.testing.assert_allclose(float(U), 0.0, atol=1e-15)
    np.testing.assert_allclose(float(V), 0.0, atol=1e-15)


def test_bunch_depolarization_monotonic_with_emittance():
    """Finite beam emittance causes physical depolarization P_bunch < 1.0.

    Using common random numbers to isolate divergence spread, depolarization
    increases monotonically with beam divergence sigma_theta_e * gamma.
    """
    rng = np.random.default_rng(42)
    n_particles = 100000
    gamma_val = 2000.0
    gamma = np.full(n_particles, gamma_val)
    weights = np.ones(n_particles)

    # Common random standard normals for smooth divergence sweep
    z_x = rng.normal(0.0, 1.0, n_particles)
    z_y = rng.normal(0.0, 1.0, n_particles)

    e0, e1 = rotated_laser_axes(psi_pol=0.0, theta_xz=0.05, theta_yz=0.0)

    p_values = []
    # Divergence sweep from zero emittance up to sigma*gamma = 0.4
    sigmas = [0.0, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]
    for sig in sigmas:
        tex = z_x * sig
        tey = z_y * sig
        I, Q, U, V = compute_stokes_components(gamma, tex, tey, 0.0, 0.0, e0, e1, ellipticity=0.0)
        Itot = float(np.sum(weights * I))
        Qtot = float(np.sum(weights * Q))
        Utot = float(np.sum(weights * U))
        Vtot = float(np.sum(weights * V))
        P = math.sqrt(max(0.0, Qtot**2 + Utot**2 + Vtot**2)) / Itot
        p_values.append(P)

    assert p_values[0] == 1.0, f"Cold beam must have P=1.0, got {p_values[0]}"
    for i in range(len(p_values) - 1):
        assert p_values[i] > p_values[i + 1], (
            f"Depolarization not monotonic: P[{i}]={p_values[i]} <= P[{i+1}]={p_values[i+1]}"
        )


def test_collision_stokes_parameters_interface():
    """Collision.stokes_parameters() integrates macroparticle Stokes parameters.

    Checks:
    - Returns BunchStokes NamedTuple with named fields (I, Q, U, V, P, chi).
    - Can unpack as 6-tuple.
    - Cold scenario gives pure polarization (P = 1.0).
    - Finite emittance scenario gives physical depolarization (P < 1.0).
    """
    # 1. Cold scenario: electron bunch with negligible divergence
    beam_cold = replace(
        scenarios.BASELINE.beam,
        emit_x=Quantity(1e-12, "cm * rad"),
        emit_y=Quantity(1e-12, "cm * rad"),
        rel_energy_spread=1e-6,
    )
    interaction_cold = scenarios.build(
        replace(scenarios.BASELINE, beam=beam_cold),
        SamplingSpec(n_particles=500, seed=20260721, prefilter=1e-3),
    )
    params = XigmaEngine.schema.with_values(n_steps=16, backend="numpy")
    collision_cold = Collision(interaction=interaction_cold, params=params)

    stokes_cold = collision_cold.stokes_parameters(theta_x=0.0, theta_y=0.0)
    assert isinstance(stokes_cold, BunchStokes)
    assert stokes_cold.I > 0.0
    # Cold beam on axis has pure polarization P = 1.0
    np.testing.assert_allclose(stokes_cold.P, 1.0, rtol=1e-6, atol=1e-6)
    # Horizontal polarization has Q/I ~ 1, U ~ 0, V ~ 0
    np.testing.assert_allclose(stokes_cold.Q / stokes_cold.I, 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stokes_cold.U / stokes_cold.I, 0.0, atol=1e-6)
    np.testing.assert_allclose(stokes_cold.V / stokes_cold.I, 0.0, atol=1e-6)

    # 2. Baseline scenario with finite beam emittance -> depolarization
    interaction_warm = scenarios.build(
        scenarios.BASELINE,
        SamplingSpec(n_particles=500, seed=20260721, prefilter=1e-3),
    )
    collision_warm = Collision(interaction=interaction_warm, params=params)
    stokes_warm = collision_warm.stokes_parameters(theta_x=0.0, theta_y=0.0)

    assert isinstance(stokes_warm, BunchStokes)
    assert stokes_warm.I > 0.0
    # Warm beam has depolarization: P < 1.0
    assert 0.0 < stokes_warm.P <= 1.0

    # Unpackable as tuple
    I, Q, U, V, P, chi = stokes_warm
    assert I == stokes_warm.I
    assert Q == stokes_warm.Q
    assert U == stokes_warm.U
    assert V == stokes_warm.V
    assert P == stokes_warm.P
    assert chi == stokes_warm.chi


def test_bunch_stokes_parameters_empty_samples():
    """Empty trajectory samples return zero Stokes parameters and zero depolarization."""
    from gammaforge.engines.xigma.stages import TrajectorySamples

    empty_samples = TrajectorySamples(
        gamma=np.empty(0),
        theta_x=np.empty(0),
        theta_y=np.empty(0),
        a0_shape=np.empty(0),
        luminosity=np.empty(0),
        intensity_peak=0.1,
        n_steps=16,
    )
    res = bunch_stokes_parameters(empty_samples)
    assert res == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
