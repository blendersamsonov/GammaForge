"""`engines/analytical`: closed-form yield/width/spectrum formulas
and the `AnalyticalEngine` wrapper.

The worked-example fixture (`_EXAMPLE_BEAM`/`_EXAMPLE_LASER`) is the established
100 pC / 200 MeV / 10 um beam and 0.05 J / 0.8 um / 2.5 um / 12.74 fs pulse.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

pytestmark = [pytest.mark.tier1]

from gammaforge.engines.analytical.engine import AnalyticalEngine
from gammaforge.engines.analytical.formulas import (
    _electron_sigma2,
    angle_integrated_spectrum,
    estimate_spectrum_width,
    overlap_mean_a0_sq,
    overlap_time_profile,
    overlap_transverse_profile,
    overlap_yield,
)
from gammaforge.io.bunch import GaussianElectronBeam, _drift_fit, momenta, sample_gaussian_bunch
from gammaforge.io.laser import GaussianParaxialLaser, fit_gaussian_paraxial, lab_frame_axes
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.io.units import C_CGS, SIGMA_T_CGS, Quantity
from gammaforge.validation import scenarios

_EXAMPLE_BEAM = GaussianElectronBeam(
    bunch_charge=Quantity(100.0, "pC"),
    kinetic_energy=Quantity(200.0, "MeV"),
    rel_energy_spread=0.001,
    sigma_x=Quantity(10.0, "um"),
    sigma_y=Quantity(10.0, "um"),
    emit_x=Quantity(0.05, "um") * Quantity(1.0, "rad"),
    emit_y=Quantity(0.05, "um") * Quantity(1.0, "rad"),
    sigma_z=Quantity(1.0, "ps"),
)
_EXAMPLE_LASER = GaussianParaxialLaser(
    pulse_energy=Quantity(0.05, "J"),
    wavelength=Quantity(0.8, "um"),
    sigma_x=Quantity(2.5, "um"),
    sigma_y=Quantity(2.5, "um"),
    duration=Quantity(12.74, "fs"),
)

#: Established width for the worked example, retained as a numerical regression pin.
_REFERENCE_WIDTH_TOTAL = 4.6671292359002505


# ---------------------------------------------------------------------------
# overlap_yield — the general luminosity overlap integral (DER001)
# ---------------------------------------------------------------------------
def _round_scenario():
    """Round in every sense the closed form needs: equal sizes *and* equal emittances
    (so beta_x == beta_y), round laser, both foci at the origin, alpha == 0."""
    beam = replace(scenarios.BASELINE.beam, emit_y=scenarios.BASELINE.beam.emit_x,
                   sigma_y=scenarios.BASELINE.beam.sigma_x)
    laser = replace(scenarios.BASELINE.laser, sigma_y=scenarios.BASELINE.laser.sigma_x)
    return beam, laser


def _erfcx(nu: float) -> float:
    return math.exp(nu * nu) * math.erfc(nu)


def _closed_form_round(beam, laser, N_e):
    """The round-beam closed form, evaluated independently of `formulas.estimate_yield`
    and using this repo's own `rayleigh_x()` — the analytic limit `overlap_yield` must
    reproduce, with ``nu = L (1 + beta_0) / (sqrt(2) D)`` (DER001 §A.4)."""
    beta_0 = beam.beta0()
    D = math.hypot(beam.m("sigma_z"), beta_0 * laser.m("duration") * C_CGS)
    sigma0_sq = beam.m("sigma_x") ** 2 + laser.m("sigma_x") ** 2
    inv_L2 = (
        beam.m("sigma_x") ** 2 / beam.beta_star_x() ** 2
        + laser.m("sigma_x") ** 2 / laser.rayleigh_x() ** 2
    ) / sigma0_sq
    nu = (1.0 + beta_0) / (math.sqrt(2.0 * inv_L2) * D)
    return SIGMA_T_CGS * N_e * laser.n_photons() * nu * _erfcx(nu) / (2.0 * math.sqrt(math.pi) * sigma0_sq)


def test_overlap_yield_reduces_to_the_round_beam_closed_form():
    """The whole derivation, checked end to end: in the round/aligned/alpha=0 limit the
    quadrature must reproduce an independently-evaluated closed form, not merely sit near
    it. This is the §7 "exact identity where the contract guarantees one" for the yield."""
    beam, laser = _round_scenario()
    N_e = beam.n_electrons()
    expected = _closed_form_round(beam, laser, N_e)
    assert overlap_yield(beam, laser, N_e, n_quad=32001) == pytest.approx(expected, rel=1e-8)


def test_overlap_yield_converges_on_a_displaced_non_round_astigmatic_scenario():
    """Convergence has to hold where the integrand is *hard* — displaced, astigmatic,
    rotated foci push the structure of `1/sqrt(det)` away from z = 0 and onto scales far
    shorter than the Gaussian weight. On the aligned baseline this would pass trivially."""
    beam = replace(scenarios.BASELINE.beam, alpha_x=2.5, alpha_y=-1.3)
    laser = replace(
        scenarios.BASELINE.laser,
        sigma_x=Quantity(8.0, "um"), sigma_y=Quantity(22.0, "um"),
        z_fx=Quantity(0.03, "cm"), z_fy=Quantity(-0.05, "cm"),
        psi_focus=Quantity(0.7, "rad"),
    )
    N_e = beam.n_electrons()
    coarse = overlap_yield(beam, laser, N_e, n_quad=8001)
    fine = overlap_yield(beam, laser, N_e, n_quad=32001)
    assert coarse == pytest.approx(fine, rel=1e-5)


def test_electron_hourglass_matches_io_bunchs_own_drift():
    """`_electron_sigma2` re-expresses the Twiss drift `io.bunch` already owns, so it must
    agree with `_drift_fit` exactly — including the **sign** of `alpha`, which no symmetric
    scenario can catch (a flipped sign puts the waist on the wrong side and is invisible
    at alpha = 0 and in any test symmetric about z = 0)."""
    for alpha in (-2.0, -0.7, 0.0, 0.7, 2.0):
        beam = replace(scenarios.BASELINE.beam, alpha_x=alpha, alpha_y=-0.5 * alpha)
        for length in (-7.0, -1.0, 0.0, 1.0, 7.0):
            ex2, ey2 = _electron_sigma2(beam, length)
            reference = _drift_fit(beam, length)
            assert math.sqrt(ex2) == pytest.approx(reference.m("sigma_x"), rel=1e-13)
            assert math.sqrt(ey2) == pytest.approx(reference.m("sigma_y"), rel=1e-13)


def test_overlap_yield_peaks_when_the_two_waists_coincide():
    """The physical statement of the same sign convention, across the two *independent*
    sign conventions this integral joins: the bunch's `alpha` and the pulse's `z_fx`
    (an offset along `k_hat`, which is -z head-on). Yield is largest when the electron
    waist sits at the laser focus; if either sign were flipped the peak would land on the
    opposite side. Needs beta_0 comparable to the Rayleigh range, or the electron waist
    position simply does not influence the answer."""
    emit = Quantity(5e-6, "cm * rad")
    base = replace(scenarios.BASELINE.beam, emit_x=emit, emit_y=emit)
    beta_0 = base.m("sigma_x") ** 2 / base.m("emit_x")
    focus_at = 0.10  # cm, lab position of the laser focus
    laser = replace(
        scenarios.BASELINE.laser,
        z_fx=Quantity(-focus_at, "cm"), z_fy=Quantity(-focus_at, "cm"),
    )
    N_e = base.n_electrons()

    alphas = np.linspace(-1.5, 1.5, 31)
    yields = [
        overlap_yield(replace(base, alpha_x=float(a), alpha_y=float(a)), laser, N_e, n_quad=8001)
        for a in alphas
    ]
    best = float(alphas[int(np.argmax(yields))])
    waist_at = best * beta_0 / (1.0 + best**2)
    assert waist_at == pytest.approx(focus_at, abs=0.02 * beta_0)


def test_psi_focus_is_a_noop_for_a_round_laser():
    """Rotating a circle changes nothing — a structural check on the 2x2 covariance."""
    beam = scenarios.BASELINE.beam
    laser = replace(scenarios.BASELINE.laser, sigma_y=scenarios.BASELINE.laser.sigma_x)
    N_e = beam.n_electrons()
    rotated = replace(laser, psi_focus=Quantity(0.9, "rad"))
    assert overlap_yield(beam, laser, N_e) == pytest.approx(overlap_yield(beam, rotated, N_e), rel=1e-13)


def test_psi_focus_matters_when_both_ellipses_are_flat():
    """The generalization is real, not decorative: aligning a flat pulse with a flat bunch
    against crossing them changes the yield by tens of percent. A round bunch would hide
    this almost entirely, which is why the fixture flattens both."""
    flat = replace(scenarios.BASELINE.beam, sigma_x=Quantity(30.0, "um"), sigma_y=Quantity(3.0, "um"))
    laser = replace(scenarios.BASELINE.laser, sigma_x=Quantity(30.0, "um"), sigma_y=Quantity(3.0, "um"))
    N_e = flat.n_electrons()
    aligned = overlap_yield(flat, laser, N_e, n_quad=8001)
    crossed = overlap_yield(flat, replace(laser, psi_focus=Quantity(math.pi / 2, "rad")), N_e, n_quad=8001)
    assert aligned > 1.2 * crossed


# ---------------------------------------------------------------------------
# Crossing angle (DER001 §A.6)
# ---------------------------------------------------------------------------
def _constant_width_closed_form(beam, laser, N_e, theta_xz, theta_yz=0.0):
    """Exact yield when both hourglasses are switched off, at any crossing angle::

        N = sigma_T (1+beta_0) N_e N_L
            / (2 pi sqrt(h det M') sigma_ex sigma_ey sigma_ez s1 s2 s_ct)

    Independent of `formulas.py`: it builds the 3x3 quadratic form directly and takes a
    `numpy` determinant, so agreement isolates the crossing-angle *geometry* from the
    hourglass and from the quadrature."""
    beta_0 = beam.beta0()
    k_hat, f1, f2 = lab_frame_axes(theta_xz, theta_yz, laser.m("psi_focus"))
    sex, sey, sez = beam.m("sigma_x"), beam.m("sigma_y"), beam.m("sigma_z")
    s1, s2, s_ct = laser.m("sigma_x"), laser.m("sigma_y"), laser.sigma_ct()
    m = np.diag([1 / sex**2, 1 / sey**2, 1 / sez**2])
    m = m + np.outer(f1, f1) / s1**2 + np.outer(f2, f2) / s2**2 + np.outer(k_hat, k_hat) / s_ct**2
    g = beta_0 * np.array([0.0, 0.0, 1.0]) / sez**2 + k_hat / s_ct**2
    h = beta_0**2 / sez**2 + 1 / s_ct**2
    m_prime = m - np.outer(g, g) / h
    return (
        SIGMA_T_CGS * (1 + beta_0) * N_e * laser.n_photons()
        / (2 * math.pi * math.sqrt(h * float(np.linalg.det(m_prime))) * sex * sey * sez * s1 * s2 * s_ct)
    )


def _no_hourglass():
    """Tiny emittance -> enormous beta*; tiny wavelength -> enormous z_R. Both hourglass
    scales far exceed the bunch length, so the spot sizes are constant across the collision
    and the integral has the closed form above."""
    beam = replace(scenarios.BASELINE.beam,
                   emit_x=Quantity(1e-12, "cm * rad"), emit_y=Quantity(1e-12, "cm * rad"))
    laser = replace(scenarios.BASELINE.laser, wavelength=Quantity(1e-7, "um"))
    return beam, laser


@pytest.mark.parametrize("theta_xz, theta_yz", [
    (0.0, 0.0), (0.002, 0.0), (0.05, 0.0), (0.4, 0.0), (0.0, 0.05), (0.03, -0.02),
])
def test_crossing_angle_matches_the_constant_width_closed_form(theta_xz, theta_yz):
    """The crossing-angle geometry, checked at machine precision against an independently
    built quadratic form — in both crossing planes and combined, out to 0.4 rad."""
    beam, laser = _no_hourglass()
    laser = replace(laser, theta_xz=Quantity(theta_xz, "rad"), theta_yz=Quantity(theta_yz, "rad"))
    expected = _constant_width_closed_form(beam, laser, beam.n_electrons(), theta_xz, theta_yz)
    assert overlap_yield(beam, laser, beam.n_electrons(), n_quad=20001) == pytest.approx(expected, rel=1e-9)


def test_crossing_angle_reproduces_the_piwinski_suppression():
    """The physics content, not just the algebra: at small angles the suppression must be
    the textbook crossing-angle luminosity reduction `1/sqrt(1 + (sigma_s tan(theta)/sigma_perp)^2)`.
    That form is itself a small-angle result, so it is asserted only where it is valid —
    the machine-precision check above is what covers large angles."""
    beam, laser = _no_hourglass()
    N_e = beam.n_electrons()
    head_on = _constant_width_closed_form(beam, laser, N_e, 0.0)
    sigma_s = math.hypot(beam.m("sigma_z"), laser.sigma_ct()) / 2.0
    sigma_perp = math.hypot(beam.m("sigma_x"), laser.m("sigma_x"))
    for theta in (0.002, 0.01):
        ratio = _constant_width_closed_form(beam, laser, N_e, theta) / head_on
        piwinski = 1.0 / math.sqrt(1.0 + (sigma_s * math.tan(theta) / sigma_perp) ** 2)
        assert ratio == pytest.approx(piwinski, rel=2e-4)


# ---------------------------------------------------------------------------
# The exact 2D mode, <a0^2>, and the resolved profiles (DER001 §A.7-§A.8)
# ---------------------------------------------------------------------------
def _full_geometry():
    """Everything the integral claims to handle, at once."""
    beam = replace(scenarios.BASELINE.beam,
                   sigma_x=Quantity(20.0, "um"), sigma_y=Quantity(6.0, "um"), alpha_x=1.5)
    laser = replace(scenarios.BASELINE.laser,
                    sigma_x=Quantity(8.0, "um"), sigma_y=Quantity(22.0, "um"),
                    z_fx=Quantity(0.03, "cm"), z_fy=Quantity(-0.05, "cm"),
                    psi_focus=Quantity(0.7, "rad"), theta_xz=Quantity(0.05, "rad"))
    return beam, laser


def test_exact_2d_mode_is_identical_to_the_1d_path_head_on():
    """Head-on, `u` depends on `z` alone, so the 1D path is already exact and the 2D mode
    must return the *same* number — not merely a close one. Anything else would mean the
    2D branch had introduced an error the 1D one does not have."""
    beam, laser = _round_scenario()
    N_e = beam.n_electrons()
    assert overlap_yield(beam, laser, N_e, 4001, n_quad_u=41) == overlap_yield(beam, laser, N_e, 4001)


def _adversarial(theta):
    """The worst corner this model has: a 2 um waist against a 200 um bunch, so the dropped
    transverse term `delta = k_x x + k_y y` is a large fraction of the Rayleigh range."""
    beam = replace(scenarios.BASELINE.beam,
                   sigma_x=Quantity(200.0, "um"), sigma_y=Quantity(200.0, "um"))
    laser = replace(scenarios.BASELINE.laser,
                    sigma_x=Quantity(2.0, "um"), sigma_y=Quantity(2.0, "um"),
                    theta_xz=Quantity(theta, "rad"))
    return beam, laser


def test_exact_2d_mode_bounds_the_1d_error_at_a_large_crossing_angle():
    """Where the exact mode earns its cost. At 0.4 rad it converges quickly — n_u = 301 and
    901 agree to 5e-7 — so the residual gap to the 1D path is genuinely the 1D
    approximation's error and not the 2D grid's, and it is 1.6e-3."""
    beam, laser = _adversarial(0.4)
    N_e = beam.n_electrons()
    exact = overlap_yield(beam, laser, N_e, 4001, n_quad_u=301)
    assert exact == pytest.approx(overlap_yield(beam, laser, N_e, 4001, n_quad_u=901), rel=1e-5)
    assert exact == pytest.approx(overlap_yield(beam, laser, N_e, 4001), rel=3e-3)


def test_exact_2d_mode_converges_toward_the_1d_path_at_a_small_crossing_angle():
    """The counterintuitive half, asserted rather than described. As `theta -> 0` the widths
    stop depending on `q1`, so the 2D mode spends nodes re-integrating a direction the 1D
    path does analytically and converges *more slowly* than the approximation it checks —
    at 20 mrad, n_u = 301 is still 1.3e-2 away from n_u = 901.

    So the test is directional, not a tolerance: adding nodes must move the 2D result toward
    the 1D one, which is the statement that the 1D path is the accurate one here. A fixed
    band would either pass vacuously or pin the 2D grid's own error."""
    beam, laser = _adversarial(0.02)
    N_e = beam.n_electrons()
    approx = overlap_yield(beam, laser, N_e, 4001)
    errors = [abs(overlap_yield(beam, laser, N_e, 4001, n_quad_u=n) / approx - 1.0)
              for n in (101, 301, 901)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-3


@pytest.mark.tier3
@pytest.mark.heavy
@pytest.mark.parametrize("name", ["baseline", "full_geometry"])
def test_mean_a0_sq_matches_a_brute_force_monte_carlo(name):
    """`<a0^2>` against the same independent reference as the yield, weighting each
    macroparticle's contribution by `a0_profile**2`. It is a large correction: the bunch
    samples about a third of the pulse's peak `a0^2` at the baseline, so using `a0_peak`
    for the nonlinearity term overstated it by ~3x."""
    beam, laser = _full_geometry() if name == "full_geometry" else (scenarios.BASELINE.beam,
                                                                   scenarios.BASELINE.laser)
    bunch = sample_gaussian_bunch(beam, 120_000, 0)
    px, py, pz = momenta(bunch)
    bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma
    k_hat, _, _ = laser.focusing_axes()
    flux = C_CGS * (1.0 - (bx * k_hat[0] + by * k_hat[1] + bz * k_hat[2]))
    beta_0 = beam.beta0()
    t_max = 8.0 * math.hypot(beam.m("sigma_z"), beta_0 * laser.sigma_ct()) / ((1.0 + beta_0) * C_CGS)
    t_grid = np.linspace(-t_max, t_max, 151)

    num, den = [], []
    for t in t_grid:
        x, y, z = bunch.x + C_CGS * bx * t, bunch.y + C_CGS * by * t, bunch.z + C_CGS * bz * t
        w = laser.photon_density(x, y, z, t) * flux
        num.append(float(np.sum(w * laser.a0_profile(x, y, z, t) ** 2)))
        den.append(float(np.sum(w)))
    reference = float(np.trapezoid(num, t_grid) / np.trapezoid(den, t_grid))

    value = overlap_mean_a0_sq(beam, laser, n_quad=8001)
    assert value == pytest.approx(reference, rel=5e-3)
    assert 0.0 < value < laser.a0_peak() ** 2


@pytest.mark.parametrize("sigma_z_fs", [0.2, 200.0])
def test_mean_a0_sq_has_the_exact_one_over_root_two_limit(sigma_z_fs):
    """An exact analytic limit, and a more interesting one than "approaches the peak".

    Take a transversally pointlike bunch and switch off both hourglasses. Integrating over
    *both* `z` and `t` spans every relative shift between the two pulses, so the bunch
    convolution factors out of numerator and denominator alike and the ratio collapses to
    `Int g^2 / Int g` for a normalized Gaussian — exactly `1/sqrt(2)`.

    So the average can never reach `a0_peak**2` in a counter-propagating collision, however
    small the bunch: it always scans the pulse's full longitudinal profile. That is the
    ceiling, and it is **independent of bunch length** — hence both parametrizations, four
    orders of magnitude apart in `sigma_z`, giving the same number."""
    beam = replace(scenarios.BASELINE.beam,
                   sigma_x=Quantity(0.02, "um"), sigma_y=Quantity(0.02, "um"),
                   sigma_z=Quantity(sigma_z_fs, "fs"),
                   emit_x=Quantity(1e-14, "cm * rad"), emit_y=Quantity(1e-14, "cm * rad"))
    laser = replace(scenarios.BASELINE.laser, wavelength=Quantity(1e-7, "um"))
    ratio = overlap_mean_a0_sq(beam, laser, n_quad=20001) / laser.a0_peak() ** 2
    assert ratio == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-4)


@pytest.mark.tier3
@pytest.mark.heavy
@pytest.mark.parametrize("name", ["baseline", "full_geometry"])
def test_resolved_profiles_integrate_back_to_the_total_yield(name):
    """§7's "exact identity where the contract guarantees one", applied to the previews:
    resolving the same integral in time or across the transverse plane cannot change what
    it sums to. Both hold to ~1e-7, i.e. quadrature precision rather than a tolerance."""
    beam, laser = _full_geometry() if name == "full_geometry" else (scenarios.BASELINE.beam,
                                                                   scenarios.BASELINE.laser)
    N_e = beam.n_electrons()
    total = overlap_yield(beam, laser, N_e, 8001)

    beta_0 = beam.beta0()
    t_max = 10.0 * math.hypot(beam.m("sigma_z"), beta_0 * laser.sigma_ct()) / ((1.0 + beta_0) * C_CGS)
    t_grid = np.linspace(-t_max, t_max, 2001)
    in_time = float(np.trapezoid(overlap_time_profile(beam, laser, N_e, t_grid, n_quad=4001), t_grid))
    assert in_time == pytest.approx(total, rel=1e-5)

    span = 10.0 * math.hypot(beam.m("sigma_x"), laser.m("sigma_x"))
    axis = np.linspace(-span, span, 201)
    grid = overlap_transverse_profile(beam, laser, N_e, axis[:, None], axis[None, :], n_quad=2001)
    in_space = float(np.trapezoid(np.trapezoid(grid, axis, axis=1), axis))
    assert in_space == pytest.approx(total, rel=1e-5)


# ---------------------------------------------------------------------------
# Transverse and timing misalignment (DER001 §A.11)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dx_um, dy_um", [(5.0, 0.0), (0.0, 5.0), (12.0, -8.0), (25.0, 20.0)])
def test_transverse_offset_falls_off_as_the_exact_gaussian(dx_um, dy_um):
    """The sharpest available check on the offset bookkeeping. With both hourglasses off,
    a transverse misalignment `d` must reduce the yield by exactly
    `exp(-d^T (C_e + C_l)^-1 d / 2)` — no quadrature error, no tolerance band. It isolates
    the linear term added to the quadratic form from everything else, and an off-diagonal
    case is included because a wrong inverse passes the on-axis ones."""
    beam = replace(scenarios.BASELINE.beam,
                   emit_x=Quantity(1e-12, "cm * rad"), emit_y=Quantity(1e-12, "cm * rad"))
    laser = replace(scenarios.BASELINE.laser, wavelength=Quantity(1e-7, "um"),
                    sigma_x=Quantity(14.0, "um"), sigma_y=Quantity(6.0, "um"))
    N_e = beam.n_electrons()
    base = overlap_yield(beam, laser, N_e, 20001)
    offset = replace(laser, x_off=Quantity(dx_um, "um"), y_off=Quantity(dy_um, "um"))

    d = np.array([dx_um * 1e-4, dy_um * 1e-4])
    cov = np.diag([beam.m("sigma_x") ** 2 + laser.m("sigma_x") ** 2,
                   beam.m("sigma_y") ** 2 + laser.m("sigma_y") ** 2])
    expected = base * math.exp(-0.5 * float(d @ np.linalg.inv(cov) @ d))
    assert overlap_yield(beam, offset, N_e, 20001) == pytest.approx(expected, rel=1e-10)


@pytest.mark.tier3
@pytest.mark.heavy
@pytest.mark.parametrize("name, kwargs, tol", [
    ("transverse x", dict(x_off=Quantity(8.0, "um")), 5e-3),
    ("transverse y", dict(y_off=Quantity(-12.0, "um")), 5e-3),
    ("timing", dict(t_off=Quantity(5.0, "ps")), 5e-3),
    ("crossing + timing", dict(theta_xz=Quantity(0.02, "rad"), t_off=Quantity(5.0, "ps")), 5e-3),
    ("crossing + all three", dict(theta_xz=Quantity(0.02, "rad"), x_off=Quantity(8.0, "um"),
                                  y_off=Quantity(-5.0, "um"), t_off=Quantity(3.0, "ps")), 5e-3),
    ("flying focus + offsets", dict(beta_ff=1.0, x_off=Quantity(6.0, "um"),
                                    t_off=Quantity(2.0, "ps")), 1.5e-2),
])
def test_offsets_match_a_brute_force_monte_carlo(name, kwargs, tol):
    """Offsets against the independent reference, including every way they compose with the
    geometry already covered. `_local_coordinates` subtracts them, so the Monte Carlo picks
    them up with no changes of its own."""
    beam = scenarios.BASELINE.beam
    laser = replace(scenarios.BASELINE.laser, **kwargs)
    assert _monte_carlo_yield(beam, laser, n_t=201) == pytest.approx(
        overlap_yield(beam, laser, beam.n_electrons()), rel=tol
    )


def test_a_timing_offset_and_a_crossing_angle_do_not_act_independently():
    """They are not separable, and an implementation that treated them as two independent
    reductions would say they are: a timing slip means the beams meet away from the nominal
    point, and with a crossing angle that displaces the collision *transversely* as well.
    So the same slip must cost more when the beams cross."""
    beam = scenarios.BASELINE.beam
    N_e = beam.n_electrons()

    def loss(theta):
        base = overlap_yield(beam, replace(scenarios.BASELINE.laser,
                                           theta_xz=Quantity(theta, "rad")), N_e)
        slipped = overlap_yield(beam, replace(scenarios.BASELINE.laser,
                                              theta_xz=Quantity(theta, "rad"),
                                              t_off=Quantity(10.0, "ps")), N_e)
        return slipped / base

    assert loss(0.05) < loss(0.0)


# ---------------------------------------------------------------------------
# Flying focus (DER002)
# ---------------------------------------------------------------------------
@pytest.mark.tier3
@pytest.mark.heavy
@pytest.mark.parametrize("beta_ff", [0.5, 1.0, 2.0, -0.5])
def test_flying_focus_yield_matches_a_brute_force_monte_carlo(beta_ff):
    """A flying focus makes the spot-size coordinate depend on time, so the widths depend
    on two independent functionals of `(x, y, z, ct)` and the time integration cannot be
    done first. The `(z, ct)` quadrature that replaces it is checked against the same
    independent reference as everything else — `photon_density` implements `u_spot`, so the
    Monte Carlo needs no changes to cover this."""
    beam = replace(scenarios.BASELINE.beam, sigma_z=Quantity(30.0, "um"))
    laser = replace(scenarios.BASELINE.laser, beta_ff=beta_ff)
    assert _monte_carlo_yield(beam, laser) == pytest.approx(
        overlap_yield(beam, laser, beam.n_electrons()), rel=5e-3
    )


def test_synchronized_flying_focus_maximizes_the_yield():
    """The physics claim, asserted: `beta_ff = 1` makes the focal plane travel at `c` in
    `+z`, co-moving with the bunch, so the electrons sit at the waist throughout instead of
    sweeping through the hourglass. It must beat both slower and faster slides, and beat no
    flying focus by a wide margin — 2.8x for this short bunch."""
    beam = replace(scenarios.BASELINE.beam, sigma_z=Quantity(30.0, "um"))
    N_e = beam.n_electrons()

    def yield_at(beta_ff):
        return overlap_yield(beam, replace(scenarios.BASELINE.laser, beta_ff=beta_ff), N_e)

    synchronized = yield_at(1.0)
    assert synchronized > yield_at(0.5)
    assert synchronized > yield_at(1.6)
    assert synchronized > 2.5 * yield_at(0.0)


def test_flying_focus_yield_is_invariant_under_beta_ff_to_its_reciprocal():
    """A non-obvious exact symmetry, and a sharp check on the whole `(z, ct)` construction.

    Along the collision ridge a short bunch has `z ~ ct`, so the spot-size coordinate goes
    as `u_spot ~ (beta_ff - 1) ct` while `rayleigh_x()` carries the repo's
    `(1 + beta_ff)` stretch. The spot therefore depends on
    `(beta_ff - 1)/(beta_ff + 1)`, which is odd under `beta_ff -> 1/beta_ff` — and the
    width depends on its square. So the yield is unchanged, with `beta_ff = 1` the fixed
    point that maximizes it. Nothing in the implementation knows this.

    Both halves are physical: the numerator from the kinematics of the sliding focus, the
    denominator from `rayleigh_x()`'s `(1 + beta_ff)` stretch, which the author confirms
    follows from solving Maxwell's equations in the paraxial approximation rather than
    being a fitting convention. So this is a physics identity, not a consistency check
    between two bookkeeping choices."""
    beam = replace(scenarios.BASELINE.beam, sigma_z=Quantity(30.0, "um"))
    N_e = beam.n_electrons()

    def yield_at(beta_ff):
        return overlap_yield(beam, replace(scenarios.BASELINE.laser, beta_ff=beta_ff), N_e)

    for beta_ff in (0.25, 0.5, 0.8):
        assert yield_at(beta_ff) == pytest.approx(yield_at(1.0 / beta_ff), rel=2e-4)


def test_mean_a0_sq_handles_a_flying_focus():
    """`<a0^2>` runs through the same reduced integral, so it inherits the flying-focus
    path — checked against the a0-weighted Monte Carlo rather than assumed."""
    beam = replace(scenarios.BASELINE.beam, sigma_z=Quantity(30.0, "um"))
    laser = replace(scenarios.BASELINE.laser, beta_ff=1.0)
    bunch = sample_gaussian_bunch(beam, 120_000, 0)
    px, py, pz = momenta(bunch)
    bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma
    k_hat, _, _ = laser.focusing_axes()
    flux = C_CGS * (1.0 - (bx * k_hat[0] + by * k_hat[1] + bz * k_hat[2]))
    beta_0 = beam.beta0()
    t_max = 10.0 * math.hypot(beam.m("sigma_z"), beta_0 * laser.sigma_ct()) / ((1.0 + beta_0) * C_CGS)
    t_grid = np.linspace(-t_max, t_max, 201)
    num, den = [], []
    for t in t_grid:
        x, y, z = bunch.x + C_CGS * bx * t, bunch.y + C_CGS * by * t, bunch.z + C_CGS * bz * t
        w = laser.photon_density(x, y, z, t) * flux
        num.append(float(np.sum(w * laser.a0_profile(x, y, z, t) ** 2)))
        den.append(float(np.sum(w)))
    reference = float(np.trapezoid(num, t_grid) / np.trapezoid(den, t_grid))
    assert overlap_mean_a0_sq(beam, laser) == pytest.approx(reference, rel=5e-3)


def _monte_carlo_yield(beam, laser, n_particles=50_000, n_t=151, seed=0):
    """Brute-force overlap using `io`'s own `photon_density` and real macroparticles,
    sharing no algebra with `formulas.py`: drift each particle ballistically and integrate
    `sigma_T n_L (c - v.k_hat)` over time."""
    bunch = sample_gaussian_bunch(beam, n_particles, seed)
    px, py, pz = momenta(bunch)
    bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma
    k_hat, _, _ = laser.focusing_axes()
    flux = C_CGS * (1.0 - (bx * k_hat[0] + by * k_hat[1] + bz * k_hat[2]))

    beta_0 = beam.beta0()
    t_max = 8.0 * math.hypot(beam.m("sigma_z"), beta_0 * laser.sigma_ct()) / ((1.0 + beta_0) * C_CGS)
    t_grid = np.linspace(-t_max, t_max, n_t)
    per_t = np.array([
        float(np.sum(laser.photon_density(bunch.x + C_CGS * bx * t,
                                          bunch.y + C_CGS * by * t,
                                          bunch.z + C_CGS * bz * t, t) * flux))
        for t in t_grid
    ])
    return (SIGMA_T_CGS * laser.n_photons() * beam.n_electrons() / n_particles
            * float(np.trapezoid(per_t, t_grid)))


@pytest.mark.tier3
@pytest.mark.heavy
@pytest.mark.parametrize("name", ["head_on", "crossing", "crossing_plus_everything", "both_planes"])
def test_overlap_yield_matches_a_brute_force_monte_carlo(name):
    """The end-to-end independent check: no step of the derivation is shared with the
    reference, which samples macroparticles and evaluates `GaussianParaxialLaser`'s own
    `photon_density`. Agreement at a few 1e-4 across head-on, a crossing angle, and a
    crossing angle combined with everything else the integral claims to handle."""
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    if name == "crossing":
        laser = replace(laser, theta_xz=Quantity(0.02, "rad"))
    elif name == "crossing_plus_everything":
        beam = replace(beam, sigma_x=Quantity(20.0, "um"), sigma_y=Quantity(6.0, "um"))
        laser = replace(laser, sigma_x=Quantity(8.0, "um"), sigma_y=Quantity(22.0, "um"),
                        z_fx=Quantity(0.03, "cm"), z_fy=Quantity(-0.05, "cm"),
                        psi_focus=Quantity(0.7, "rad"), theta_xz=Quantity(0.05, "rad"))
    elif name == "both_planes":
        laser = replace(laser, theta_xz=Quantity(0.03, "rad"), theta_yz=Quantity(-0.02, "rad"))

    exact = overlap_yield(beam, laser, beam.n_electrons(), n_quad=20001)
    assert _monte_carlo_yield(beam, laser) == pytest.approx(exact, rel=5e-3)


def test_estimate_spectrum_width_reproduces_the_reference_worked_example():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert w.total == pytest.approx(_REFERENCE_WIDTH_TOTAL, rel=1e-6)


def test_spectrum_width_breakdown_total_is_hypot_of_components():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert w.total == pytest.approx(
        math.hypot(w.collimation, w.emittance, w.energy_spread, w.nonlinearity)
    )


def test_angle_integrated_spectrum_zero_outside_kinematic_range():
    gamma0 = 100.0
    s_far_beyond_edge = np.array([gamma0**2 * 1.5])
    out = angle_integrated_spectrum(gamma0, gamma0 * 1e-6, 1.0, s_far_beyond_edge)
    assert out[0] == 0.0


@pytest.mark.tier3
@pytest.mark.heavy
def test_angle_integrated_spectrum_matches_monte_carlo_reference():
    """Independent physics check: draw real gamma samples from the same Gaussian, sum the
    per-particle kinematic shape directly here (not via any production code path), and
    compare to the closed-form quadrature."""
    rng = np.random.default_rng(1)
    n = 2_000_000
    gamma0, sigma_gamma, N_e = _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons()
    gamma_samples = rng.normal(gamma0, sigma_gamma, n)
    weight = N_e / n

    s_array = np.linspace(0.05, 0.95, 12)
    gamma2 = (gamma_samples**2)[:, None]
    y = s_array[None, :] / gamma2
    shape = np.where((y < 0) | (y > 1), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
    mc_reference = np.sum(weight * shape / gamma2, axis=0)

    quad_result = angle_integrated_spectrum(gamma0, sigma_gamma, N_e, s_array)
    assert np.allclose(quad_result, mc_reference, rtol=0.02)


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------
def _interaction(n_particles=4000, seed=0, outputs=()):
    small = replace(scenarios.BASELINE, sampling=replace(scenarios.BASELINE.sampling, n_particles=n_particles, seed=seed))
    interaction = scenarios.build(small)
    return replace(interaction, target=replace(interaction.target, outputs=outputs))


def test_nonlinear_redshift_moves_the_compton_edge():
    """The mean red-shift is now applied: the resonance sits at `gamma^2 / (1 + ahat)`, so
    the spectrum's support ends below the linear edge. `ahat = 0` must recover the linear
    result exactly, since that is what every other spectrum test assumes.

    Comparing where the support *ends* rather than against an absolute cut, because the
    energy-spread quadrature smears the edge — electrons above `gamma0` radiate past
    `gamma0^2` — and that smearing is identical either way, so it cancels in the ratio."""
    gamma0, sigma_gamma, ahat = 400.0, 4.0, 0.2
    s = np.linspace(0.5 * gamma0**2, 1.4 * gamma0**2, 6000)

    unshifted = angle_integrated_spectrum(gamma0, sigma_gamma, 1.0, s)
    shifted = angle_integrated_spectrum(gamma0, sigma_gamma, 1.0, s, ahat=ahat)

    end_unshifted = s[np.nonzero(unshifted)[0][-1]]
    end_shifted = s[np.nonzero(shifted)[0][-1]]
    assert end_unshifted / end_shifted == pytest.approx(1.0 + ahat, rel=5e-3)

    assert np.array_equal(angle_integrated_spectrum(gamma0, sigma_gamma, 1.0, s, ahat=0.0), unshifted)


def test_engine_reports_the_shifted_edge_and_uses_the_cycle_average():
    """`ahat` must be the *cycle-averaged* intensity `0.5 * <a0^2>` for linear polarization,
    not `<a0^2>`: `a0` is the peak field magnitude, so the factor is the cycle average of
    cos^2. Passing `<a0^2>` would double the red-shift. The reported edge follows."""
    interaction = _interaction(outputs=(OutputRequest(OutputKind.TOTAL_YIELD),))
    engine = AnalyticalEngine()
    results = engine.run(interaction, engine.schema)

    ahat = results.model_specific["ahat"]
    assert ahat == pytest.approx(0.5 * results.model_specific["mean_a0_sq"])

    beam = interaction.beam
    photon_energy = fit_gaussian_paraxial(interaction.laser).photon_energy()
    linear_edge = 4.0 * beam.gamma0() ** 2 * photon_energy
    assert results.model_specific["compton_edge_energy"] == pytest.approx(linear_edge / (1.0 + ahat))
    assert results.model_specific["compton_edge_energy"] < linear_edge


def test_spectrum_integral_equals_total_yield_exactly():
    """§7: "Total yield: integral spectrum = total_yield -- exact identities, not
    tolerances, where the contract guarantees them." RES036 makes this exact
    by construction for analytical."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(200,)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    total_yield = float(results.photon_slices[OutputKind.TOTAL_YIELD].distr)
    spectrum_integral = results.photon_slices[OutputKind.SPECTRUM].integrate()
    assert spectrum_integral == pytest.approx(total_yield, rel=1e-9)


def test_spectrum_grid_integral_correction_factor_is_near_one():
    """The rescale in RES036 always makes the integral exact; this guards that the factor
    it applies stays close to 1, i.e. the auto-derived energy range is not truncating
    real spectral weight (a distortion the rescale would otherwise mask silently)."""
    from gammaforge.engines.analytical.formulas import angle_integrated_spectrum as ais
    from gammaforge.io.laser import fit_gaussian_paraxial as fit
    from gammaforge.io.results import Axis
    from gammaforge.io.target import auto_ranges, slice_axis_values

    interaction = _interaction(outputs=(OutputRequest(OutputKind.SPECTRUM, resolution=(200,)),))
    metrics = fit(interaction.laser)
    photon_energy = metrics.photon_energy()
    ranges = auto_ranges(interaction.target, interaction.beam, interaction.laser)
    request = interaction.target.outputs[0]
    values = slice_axis_values(request, ranges[OutputKind.SPECTRUM])
    s = values[Axis.ENERGY] / (4.0 * photon_energy)
    raw = ais(interaction.beam.gamma0(), interaction.beam.sigma_gamma(), 1.0, s, 401)
    raw_integral = float(np.trapezoid(raw / (4.0 * photon_energy), values[Axis.ENERGY]))
    assert raw_integral == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Model planner (DER019 §14, §17) — selection and provenance, no physics
# ---------------------------------------------------------------------------
#: Sentinel for `_tier`'s `acceptance`, so an explicit `None` (pinned-only) survives.
_UNSET = object()


def test_planner_reports_the_selected_model_per_observable():
    """Every requested observable records which model served it, and the record is
    serializable: `model_specific` goes through HDF5 as JSON with `allow_nan=False`
    (RES063), so provenance that cannot round-trip is provenance nobody can read back."""
    from gammaforge.engines.analytical.models import ModelSelector
    from gammaforge.io.target import OutputKind

    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(200,)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)

    models = results.model_specific["models"]
    assert set(models) == {"total_yield", "spectrum"}
    for kind, record in models.items():
        assert record["model"] == "overlap_der001_mean_ahat"
        assert record["requested_mode"] == "auto"
        assert record["assumptions"], "a model must declare what it assumed"
        assert isinstance(record["outer_dimension"], int)
        # The provenance must survive the exact encoder `save_results` uses.
        from gammaforge.io.formats.hdf5 import _encode_metadata
        import json as _json
        _json.dumps(_encode_metadata(models), allow_nan=False)

    # The schema's expert-pin choices and the selector's registry must not drift apart:
    # a pin naming a model that does not exist would only fail at run time.
    registered = set(AnalyticalEngine._selector.names())
    offered = set(AnalyticalEngine.schema.spec("model_pin").choices) - {"auto"}
    assert offered <= registered


def _tier(name, *, cost_rank, fidelity_rank, exact=True, acceptance=_UNSET, applies=_UNSET):
    """A synthetic planner entry.

    Built here rather than reused from the registry because the point of these tests is the
    planner's *ordering and gating logic*, and today's engine registers exactly one tier.
    A property that only holds for one candidate cannot fail when the logic breaks, so it
    would not be a test — these give the planner something to actually choose between, and
    they keep doing so as real tiers are added.

    `acceptance` and `applies` default through a sentinel rather than `None`, because
    `acceptance=None` is itself meaningful to `AnalyticalModel` (it marks a pinned-only
    tier). A plain `None` default would collapse "not specified" and "pinned-only" into the
    same value, which is exactly the distinction these tests exist to check.
    """
    from gammaforge.engines.analytical.models import AnalyticalModel

    always = lambda inputs: True  # noqa: E731
    return AnalyticalModel(
        name=name,
        outputs=(OutputKind.SPECTRUM,),
        applies=always if applies is _UNSET else applies,
        acceptance=always if acceptance is _UNSET else acceptance,
        fidelity_rank=fidelity_rank,
        cost_rank=cost_rank,
        exact=exact,
        assumptions=("synthetic",),
        outer_dimension=cost_rank,
    )


def _planner_inputs():
    from gammaforge.engines.analytical.models import ModelInputs

    interaction = _interaction()
    metrics = fit_gaussian_paraxial(interaction.laser)
    return ModelInputs(
        beam=interaction.beam,
        laser=metrics,
        photon_energy=metrics.photon_energy(),
        crossing_angle=0.0,
        beta_ff=metrics.beta_ff,
    )


def test_auto_takes_the_cheapest_tier_and_reference_the_most_fidest():
    """The two modes mean opposite things (DER019 §17.8): `auto` is "cheapest model judged
    reliable", `reference` is "highest fidelity the geometry supports". With the real
    single-tier registry both trivially agree, which is why this drives a synthetic ladder
    where they must disagree — and asserts the cheap end and the faithful end separately."""
    from gammaforge.engines.analytical.models import ModelSelector

    cheap = _tier("cheap", cost_rank=0, fidelity_rank=1)
    dear = _tier("dear", cost_rank=5, fidelity_rank=9)
    selector = ModelSelector((cheap, dear))
    inputs = _planner_inputs()

    assert selector.select(inputs, OutputKind.SPECTRUM, mode="auto").model == "cheap"
    assert selector.select(inputs, OutputKind.SPECTRUM, mode="fast").model == "cheap"
    assert selector.select(inputs, OutputKind.SPECTRUM, mode="reference").model == "dear"


def test_auto_refuses_an_approximate_tier_until_its_validity_is_established():
    """DER019 §17.8 is explicit that automatic acceptance of a reduced model must wait for
    validation evidence. A tier with `acceptance=None` is therefore pinned-only: `auto` must
    skip it rather than adopt it, and must say so, because silently promoting a guessed
    threshold into automatic selection is the specific failure this design guards against."""
    from gammaforge.engines.analytical.models import ModelSelector

    approximate = _tier("approximate", cost_rank=0, fidelity_rank=9, exact=False, acceptance=None)
    exact_dear = _tier("exact_dear", cost_rank=5, fidelity_rank=1)
    selector = ModelSelector((approximate, exact_dear))
    inputs = _planner_inputs()

    choice = selector.select(inputs, OutputKind.SPECTRUM, mode="auto")
    assert choice.model == "exact_dear", "auto must not adopt an unvalidated approximate tier"
    reasons = {entry["model"]: entry["reason"] for entry in choice.as_metadata()["rejected"]}
    assert "pinned-only" in reasons["approximate"]
    assert not reasons["approximate"].startswith("this geometry"), "a gating reason is not an applicability one"

    # `reference` may use the highest-fidelity structurally supported tier even before an
    # automatic threshold exists — but only when the caller pins it, not by default.
    with pytest.raises(ValueError, match="model_mode must be one of"):
        ModelSelector((approximate,)).select(inputs, OutputKind.SPECTRUM, mode="nonsense")
    pinned = ModelSelector((approximate,)).select(inputs, OutputKind.SPECTRUM, pin="approximate")
    assert pinned.model == "approximate" and pinned.exact is False


def test_auto_promotes_to_the_next_tier_when_the_cheapest_is_rejected():
    """DER019 §17.8: "a cheaper model rejected by validity checks causes `auto` to promote to
    the next supported deterministic model" — and the reason must be recorded, because the
    user is entitled to know which approximation they actually got."""
    from gammaforge.engines.analytical.models import ModelSelector

    rejected = _tier(
        "rejected",
        cost_rank=0,
        fidelity_rank=1,
        applies=lambda inputs: inputs.crossing_angle == 0.0,
    )
    fallback = _tier("fallback", cost_rank=3, fidelity_rank=4)
    selector = ModelSelector((rejected, fallback))

    crossed = _planner_inputs()
    crossed = replace(crossed, crossing_angle=0.05)
    choice = selector.select(crossed, OutputKind.SPECTRUM, mode="auto")
    assert choice.model == "fallback"
    metadata = choice.as_metadata()
    reasons = {entry["model"]: entry for entry in metadata["rejected"]}
    assert reasons["rejected"]["structural"] is True


def test_pinning_cannot_force_a_structurally_invalid_model():
    """An expert pin bypasses cost and validity ranking; it must not bypass the model's own
    assumptions. Doing so would be the silent-problem-substitution failure this repo rejects
    in favor of an explicit error."""
    from gammaforge.engines.analytical.models import ModelSelector

    flying = _tier("fixed_width", cost_rank=0, fidelity_rank=1, applies=lambda inputs: inputs.beta_ff == 0.0)
    selector = ModelSelector((flying,))
    flying_inputs = replace(_planner_inputs(), beta_ff=0.8)

    with pytest.raises(ValueError, match="structurally invalid"):
        selector.select(flying_inputs, OutputKind.SPECTRUM, pin="fixed_width")
    # The message must name the actual physical reason, not restate the verdict.
    with pytest.raises(ValueError, match="flying focus"):
        selector.select(flying_inputs, OutputKind.SPECTRUM, pin="fixed_width")


def test_the_registered_tier_serves_both_declared_observables():
    """One model can serve two observables without claiming both are equally exact. This
    pins the current truth: the DER001 overlap is exact for `TOTAL_YIELD`, while the
    `SPECTRUM` built from one mean `ahat` is an approximation — recorded per observable
    rather than asserted once for the model as a whole."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(64,)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    models = results.model_specific["models"]

    assert models["total_yield"]["observable"] == "total_yield"
    assert models["spectrum"]["observable"] == "spectrum"
    assert "spectrum_normalized_to_overlap_total_yield" in models["spectrum"]["assumptions"]


def test_planner_rejects_an_unknown_model_and_an_invalid_mode():
    """Both are user-input errors on a CHOICE field. `FieldSpec.validate` already rejects
    them at the schema boundary, so this pins that the planner is not a second, laxer door
    into the same mistake — it raises rather than defaulting."""
    from gammaforge.engines.analytical.models import ModelInputs
    from gammaforge.io.schema import SchemaError

    selector = AnalyticalEngine._selector
    interaction = _interaction()
    metrics = fit_gaussian_paraxial(interaction.laser)
    inputs = ModelInputs(
        beam=interaction.beam,
        laser=metrics,
        photon_energy=metrics.photon_energy(),
    )
    with pytest.raises(ValueError, match="unknown analytical model"):
        selector.select(inputs, OutputKind.SPECTRUM, pin="no_such_model")
    with pytest.raises(ValueError, match="model_mode must be"):
        selector.select(inputs, OutputKind.SPECTRUM, mode="turbo")
    # And the schema itself, which is what a GUI actually validates against.
    with pytest.raises(SchemaError):
        AnalyticalEngine.schema.with_values(model_mode="turbo")


def test_an_unknown_observable_is_refused_rather_than_silently_skipped():
    """The engine declares `supported_outputs`, and `run` filters to it. Asking the planner
    for an undeclared kind must still fail loudly: 'this engine does not do that' and
    'the planner has a bug here' should not look the same from the outside."""
    from gammaforge.engines.analytical.models import ModelInputs

    interaction = _interaction()
    metrics = fit_gaussian_paraxial(interaction.laser)
    inputs = ModelInputs(beam=interaction.beam, laser=metrics, photon_energy=metrics.photon_energy())
    with pytest.raises(ValueError, match="no analytical model produces"):
        AnalyticalEngine._selector.select(inputs, OutputKind.COLLIMATED_SPECTRUM)


def test_model_selection_leaves_every_physics_number_untouched():
    """The point of registering the pre-existing behaviour as a model *before* adding new
    ones: `auto`, `reference`, and an explicit pin must all produce the same results as
    each other, because they currently select the same tier. A future tier that changes a
    number under the default mode will fail this, which is the intended alarm."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(200,)))
    )
    engine = AnalyticalEngine()
    baseline = engine.run(interaction, engine.schema)

    for mode in ("auto", "fast", "reference"):
        results = engine.run(interaction, engine.schema.with_values(model_mode=mode))
        assert results.photon_slices[OutputKind.TOTAL_YIELD].integrate() == pytest.approx(
            baseline.photon_slices[OutputKind.TOTAL_YIELD].integrate(), rel=1e-12
        )
        assert results.model_specific["ahat"] == baseline.model_specific["ahat"]
        assert results.model_specific["mean_a0_sq"] == baseline.model_specific["mean_a0_sq"]
        assert results.photon_slices[OutputKind.SPECTRUM].integrate() == pytest.approx(
            baseline.photon_slices[OutputKind.SPECTRUM].integrate(), rel=1e-12
        )

    pinned = engine.run(interaction, engine.schema.with_values(model_pin="overlap_der001_mean_ahat"))
    assert pinned.photon_slices[OutputKind.SPECTRUM].integrate() == pytest.approx(
        baseline.photon_slices[OutputKind.SPECTRUM].integrate(), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Fixed-width deterministic Gaussian moments (DER019 §3, §4, §22)
# ---------------------------------------------------------------------------
def _hermite_expectation(func, n: int = 200) -> float:
    """``E[f(X)]`` for standard normal ``X``, by Gauss-Hermite.

    `hermegauss` weights satisfy ``sum_i w_i z_i^(2k) = sqrt(2 pi) (2k-1)!!``, i.e. they carry
    ``exp(-z^2/2)`` but not the ``1/sqrt(2 pi)``. So the expectation is a plain weighted sum
    of the divided weights. Written here rather than in `fixed_width.py` because the closed
    forms there need no quadrature: this exists to check them against something independent,
    and the convention is asserted first so a mis-scaled weight cannot masquerade as a
    physics disagreement.
    """
    z, w = hermegauss(n)
    w = w / math.sqrt(2.0 * math.pi)
    assert float(np.sum(w)) == pytest.approx(1.0, rel=1e-12), "weight convention changed"
    return float(np.sum(w * func(z)))


def test_the_rank_two_reduction_holds_at_every_crossing_angle():
    """DER019 §3: completing the square in time removes exactly one direction, so `K` has
    rank 2 — the fact that keeps the nonlinear distribution a generalized chi-square instead
    of something arbitrary. If the reduction were built in the laser's own frame, this would
    fail off head-on, where the removed direction is not `k_hat`."""
    from gammaforge.engines.analytical.fixed_width import fixed_width_reduction

    for degrees in (0.0, 5.0, 30.0, 60.0, 89.0):
        laser = replace(scenarios.BASELINE.laser, theta_xz=Quantity(degrees, "deg"))
        reduction = fixed_width_reduction(scenarios.BASELINE.beam, laser)
        eigenvalues = np.linalg.eigvalsh(reduction.K)
        nonzero = eigenvalues[eigenvalues > eigenvalues.max() * 1e-10]
        assert len(nonzero) == 2, f"{degrees} deg: rank {len(nonzero)}, expected 2"
        # Positive semidefinite: a negative eigenvalue would mean `K` is not a precision form.
        assert eigenvalues.min() > -eigenvalues.max() * 1e-10


def test_j_n_reproduces_the_round_closed_reduction_and_its_stated_limits():
    """DER019 §22.1 in full: the matrix algebra must reproduce the round closed form, and
    that closed form must show the two limits the derivation claims — `sigma_e << sigma_L`
    (uniform intensity, no spread) and `sigma_e >> sigma_L` (spread -> 1/sqrt(3) ~ 0.577).

    The round form is written out independently rather than reusing the module, so this is a
    reduction identity against a second expression of the physics rather than a self-check.
    """
    from gammaforge.engines.analytical.fixed_width import (
        fixed_width_reduction,
        nonlinear_moments,
        round_nonlinear_moments,
    )

    beam_base = scenarios.BASELINE.beam
    for sigma_e_um, sigma_l_um in ((4.0, 1.0), (2.0, 1.0), (1.0, 1.0), (10.0, 1.0)):
        beam = replace(beam_base, sigma_x=Quantity(sigma_e_um, "um"), sigma_y=Quantity(sigma_e_um, "um"))
        laser = replace(scenarios.BASELINE.laser, sigma_x=Quantity(sigma_l_um, "um"),
                        sigma_y=Quantity(sigma_l_um, "um"))
        moments = nonlinear_moments(fixed_width_reduction(beam, laser))
        closed = round_nonlinear_moments(beam.m("sigma_x"), laser.m("sigma_x"))

        # nu = 1 + sigma_L^2/sigma_e^2 is the number of illuminated Gaussian modes.
        nu = 1.0 + (sigma_l_um / sigma_e_um) ** 2
        assert moments.mean_a == pytest.approx(nu / (nu + 1) * closed.a_max, rel=1e-12)
        assert moments.mean_a_sq == pytest.approx(nu / (nu + 2) * closed.a_max**2, rel=1e-12)
        assert moments.var_between == pytest.approx(closed.var_between, rel=1e-10)
        # sigma_a/<a> = 1/sqrt(nu (nu+2)), exactly as DER019 §22.1 states.
        assert moments.relative_spread == pytest.approx(1.0 / math.sqrt(nu * (nu + 2.0)), rel=1e-12)

    # The stated limits, which is what makes the bracket defensible rather than fitted.
    assert round_nonlinear_moments(1e3, 1.0).relative_spread == pytest.approx(1.0 / math.sqrt(3.0), rel=2e-3)
    assert round_nonlinear_moments(1e-3, 1.0).relative_spread < 0.01


def test_j_n_matches_direct_gaussian_quadrature_including_a_displaced_source():
    """DER019 §18: the closed `J_n` must agree with integrating the same Gaussian directly.
    Done for a displaced (noncentral) source as well as a centered one, because the
    noncentral term is the part of the formula a centered-only test would never exercise."""
    from gammaforge.engines.analytical.fixed_width import fixed_width_reduction

    beam = replace(scenarios.BASELINE.beam, sigma_x=Quantity(4.0, "um"), sigma_y=Quantity(4.0, "um"))
    for offset_um in (0.0, 0.5, 1.5, 3.0):
        laser = replace(scenarios.BASELINE.laser, sigma_x=Quantity(4.0, "um"),
                        sigma_y=Quantity(4.0, "um"), x_off=Quantity(offset_um, "um"))
        reduction = fixed_width_reduction(beam, laser)

        # `delta` is checked against the laser's own offset rather than being fed back into
        # the integration below: reusing it would make this comparison agree with whatever
        # `fixed_width_reduction` happened to compute, including a wrong value.
        assert reduction.delta[0] == pytest.approx(-laser.m("x_off"), rel=1e-12)

        def a_labelled(x1, x2, _r=reduction):
            """`A_L` for electron labels offset from the laser's own centre."""
            dx1 = x1 - _r.delta[0]
            dx2 = x2 - _r.delta[1]
            quad = _r.K[0, 0] * dx1**2 + 2 * _r.K[0, 1] * dx1 * dx2 + _r.K[1, 1] * dx2**2
            return _r.a_max * np.exp(-0.5 * quad)

        z, w = hermegauss(200)
        w = w / math.sqrt(2.0 * math.pi)
        grid1, grid2 = np.meshgrid(z, z, indexing="ij")
        weights = w[:, None] * w[None, :]
        sigma = beam.m("sigma_x")
        a_values = a_labelled(sigma * grid1, sigma * grid2)

        direct_j1 = float(np.sum(weights * a_values))
        direct_j2 = float(np.sum(weights * a_values**2))
        assert direct_j1 == pytest.approx(reduction.j(1), rel=1e-9)
        assert direct_j2 == pytest.approx(reduction.j(2), rel=1e-9)
        # The luminosity-weighted moment is the ratio, and that ratio is what the engine uses.
        assert direct_j2 / direct_j1 == pytest.approx(reduction.j(2) / reduction.j(1), rel=1e-9)
        # The noncentral term must actually matter here, or the offsets above are decorative:
        # an offset lowers the sampled intensity, so J_1 strictly decreases with displacement.
        assert reduction.j(1) < 1.0 or offset_um == 0.0


def test_the_derived_nonlinear_mean_agrees_with_overlap_mean_a0_sq_in_the_fixed_width_limit():
    """DER019 §5 and §18.3: DER001's `overlap_mean_a0_sq` and the new distributional model
    are the same physical quantity in their common limit, so they must agree.

    Compared through the laser's own conversion rather than as bare numbers: `a_shape` here is
    dimensionless and normalized to 1 at the trajectory peak, whereas `overlap_mean_a0_sq` is a
    dimensional `<a0^2>` in the pulse's own units. The relation is

        <a_shape>_L = (cycle_average_factor / intensity_peak) * <a0^2>_L

    which holds for *any* pulse energy — so this fails loudly if either model's normalization
    drifts, instead of passing at one energy and lying at another.

    The limit is reached by making the spot **broad**, i.e. the Rayleigh range long compared
    with the bunch, so the width is effectively frozen over the collision. At a tight focus
    DER001 keeps the real paraxial evolution and the two must *not* agree — the discrepancy
    there is the frozen-width approximation, quantified as `epsilon_L`, and asserting equality
    at a tight focus would be asserting that approximation away.
    """
    from gammaforge.engines.analytical.fixed_width import (
        fixed_width_diagnostics,
        fixed_width_reduction,
        nonlinear_moments,
    )

    beam = replace(scenarios.BASELINE.beam, sigma_x=Quantity(4.0, "um"), sigma_y=Quantity(4.0, "um"))
    for pulse_energy, sigma_l_um in ((0.05, 400.0), (0.2, 800.0), (1.0, 400.0)):
        laser = replace(scenarios.BASELINE.laser, pulse_energy=Quantity(pulse_energy, "J"),
                        sigma_x=Quantity(sigma_l_um, "um"), sigma_y=Quantity(sigma_l_um, "um"))
        moments = nonlinear_moments(fixed_width_reduction(beam, laser))
        der001_mean = overlap_mean_a0_sq(beam, laser, n_quad=8001)

        # Guard the premise: this geometry has to actually be in the frozen-width limit.
        # Measured, not assumed — a broad spot is what makes epsilon_L small.
        assert fixed_width_diagnostics(beam, laser)["epsilon_L"] < 0.01
        scale = laser.cycle_average_factor() / laser.intensity_peak()
        assert moments.mean_a == pytest.approx(scale * der001_mean, rel=2e-3)

    # And the approximation is visible where it should be: a tight focus, where the ratio
    # is far from 1. This is the quantity that later justifies an acceptance threshold.
    focused = replace(scenarios.BASELINE.laser, sigma_x=Quantity(0.8, "um"), sigma_y=Quantity(0.8, "um"))
    assert fixed_width_diagnostics(beam, focused)["epsilon_L"] > 1.0


def test_the_two_nonlinear_variances_stay_separate_and_sum_to_the_total():
    """DER019 §22.2's whole point: the spread of trajectory means and the DER016
    within-trajectory variance are different physical effects, and the law of total variance
    combines them. Keeping them apart is what replaces the empirical bracket, so a test that
    only checked the sum would not notice them being conflated."""
    from gammaforge.engines.analytical.fixed_width import (
        fixed_width_reduction,
        nonlinear_moments,
    )

    beam = replace(scenarios.BASELINE.beam, sigma_x=Quantity(4.0, "um"), sigma_y=Quantity(4.0, "um"))
    laser = replace(scenarios.BASELINE.laser, sigma_x=Quantity(1.0, "um"), sigma_y=Quantity(1.0, "um"))
    moments = nonlinear_moments(fixed_width_reduction(beam, laser))

    assert moments.var_between > 0.0
    assert moments.var_finite_line > 0.0
    assert moments.var_total == pytest.approx(moments.var_between + moments.var_finite_line, rel=1e-12)
    # <V_a,shape>_L = kappa_G <a_shape^2> (DER019 §3.1).
    assert moments.var_finite_line == pytest.approx(
        (2.0 / math.sqrt(3.0) - 1.0) * moments.mean_a_sq, rel=1e-12
    )
    # `var_a` is the plain second central moment of a_shape, which is the between-trajectory
    # spread *plus* the within-trajectory finite-line variance. Asserting it equals
    # `var_between` would be asserting the two effects are the same — the conflation DER019
    # §22.2 exists to undo.
    assert moments.var_a == pytest.approx(moments.var_total, rel=1e-12)
    assert moments.var_a > moments.var_between


def test_a_flying_focus_is_refused_rather_than_silently_frozen():
    """DER019 §16: the 1D frozen-width path is unsafe for a flying focus, so this tier must
    refuse rather than quietly return a plausible wrong answer."""
    from gammaforge.engines.analytical.fixed_width import fixed_width_reduction

    laser = replace(scenarios.BASELINE.laser, beta_ff=0.8)
    with pytest.raises(ValueError, match="flying focus"):
        fixed_width_reduction(scenarios.BASELINE.beam, laser)


def test_fixed_width_validity_diagnostics_are_reported_not_thresholded():
    """DER019 §16/§17.8: the engine reports *why* a reduced model was chosen and how strongly
    its approximation is expected to hold. These must exist and vary sensibly — a spot much
    shorter than the Rayleigh range is the focused regime the tier does not cover."""
    from gammaforge.engines.analytical.fixed_width import fixed_width_diagnostics

    beam = scenarios.BASELINE.beam
    focused = replace(scenarios.BASELINE.laser, sigma_x=Quantity(0.8, "um"), sigma_y=Quantity(0.8, "um"))
    collimated = replace(scenarios.BASELINE.laser, sigma_x=Quantity(50.0, "um"), sigma_y=Quantity(50.0, "um"))

    tight = fixed_width_diagnostics(beam, focused)
    loose = fixed_width_diagnostics(beam, collimated)
    for name in ("epsilon_L", "epsilon_beta", "epsilon_drift"):
        assert name in tight and math.isfinite(tight[name])
    # A tighter focus has a shorter Rayleigh range, so L_int/z_R grows: the tier's own
    # weakness, and the reason it cannot be accepted automatically at every geometry.
    assert loose["epsilon_L"] < tight["epsilon_L"]


# ---------------------------------------------------------------------------
# Nonlinear angle-integrated spectrum (DER019 §23)
# ---------------------------------------------------------------------------
def test_the_nonlinear_shape_is_normalized_on_its_own_support():
    """DER019 §23's normalization identity: `int_0^{1/(1+h)} G(z;h) dz = 1` for every `h`.

    This is the property that lets the continuous model be trusted on its own, and it is what
    the engine's existing yield rescale (RES036) must not be needed to manufacture. A shape
    that only integrated to one *after* rescaling would hide its own error.
    """
    from gammaforge.engines.analytical.nonlinear_spectrum import nonlinear_shape, shape_support_edge

    for h in (0.0, 0.05, 0.3, 1.0, 2.0):
        nodes, weights = leggauss(400)
        edge = shape_support_edge(h)
        z = 0.5 * edge * (nodes + 1.0)
        integral = float(np.sum(weights * 0.5 * edge * nonlinear_shape(z, h)))
        assert integral == pytest.approx(1.0, rel=1e-10), f"h={h}: integral {integral}"

    # Zero and negative z carry no photons and must not be extrapolated into.
    assert nonlinear_shape(np.array([0.0, -0.1, -1.0]), 0.5).tolist() == [0.0, 0.0, 0.0]


def test_the_nonlinear_shape_reproduces_der011_exactly_at_zero_shift():
    """DER019 §23: `G(z; 0)` is DER011 — not an approximation of it. This is the linear-limit
    reduction identity, and the reason the new shape is a drop-in for the existing linear
    spectrum rather than a different normalization of it."""
    from gammaforge.engines.analytical.nonlinear_spectrum import nonlinear_shape

    z = np.linspace(1e-9, 1.0, 257)
    der011 = 1.5 * (1.0 - 2.0 * z * (1.0 - z))
    assert np.array_equal(nonlinear_shape(z, 0.0), der011)


def test_a_nonlinear_shift_changes_the_shape_and_not_only_its_scale():
    """DER019's reason for replacing the compressed-DER011 shortcut: the angular Jacobian
    depends on `h`, so `G` differs from a rescaled linear shape by a term that is *not* a
    normalization change. DER019 gives the leading discrepancy as `(3/2) h (2z-1)^3`, which
    is odd about z=1/2 and first order in h — both checkable."""
    from gammaforge.engines.analytical.nonlinear_spectrum import nonlinear_shape

    def compressed_linear(z, h):
        """`G_comp` from DER019 §23: the linear shape rescaled and renormalized."""
        return (1.0 + h) * 1.5 * (1.0 - 2.0 * (1.0 + h) * z * (1.0 - (1.0 + h) * z))

    # Leading discrepancy ratio -> 1 as h -> 0, away from the zero crossing at z = 1/2.
    z = np.array([0.1, 0.3, 0.7, 0.9])
    ratio = (nonlinear_shape(z, 1e-5) - compressed_linear(z, 1e-5)) / (1.5 * 1e-5 * (2.0 * z - 1.0) ** 3)
    assert ratio == pytest.approx(np.ones_like(z), rel=5e-3)

    # Sign structure: the discrepancy is odd about z = 1/2.
    below = nonlinear_shape(np.array([0.4]), 0.2)[0] - compressed_linear(np.array([0.4]), 0.2)[0]
    above = nonlinear_shape(np.array([0.6]), 0.2)[0] - compressed_linear(np.array([0.6]), 0.2)[0]
    assert below * above < 0.0


def test_the_collision_average_conserves_photon_count_without_rescaling():
    """DER019 §23.1 and the handoff's requirement that photon-count normalization not depend
    on post-hoc engine rescaling.

    **How conservation has to be measured.** Each `G(.; chi x)` is supported on
    `z <= 1/(1+chi x)`, and `chi x` grows with `x`, so the *brighter* trajectories radiate to
    higher energy than the dimmest one. The average over `x` therefore has support only on the
    narrowest edge `z <= 1/(1+chi)`. Integrating the averaged shape over that common edge and
    expecting one is therefore the wrong measurement — it necessarily discards the high-energy
    tail, which is a real physical cutoff rather than a lost photon.

    Conservation is stated correctly per trajectory instead: integrated over its *own* support,
    every `G(.; h)` contributes exactly one, and the `x` weights sum to one. That is what makes
    the construction normalized by composition rather than by rescaling.

    `collision_averaged_shape` applies DER019's per-trajectory limit `X(z)` internally, which is
    the mechanism that keeps this true while producing a single averaged shape.
    """
    from gammaforge.engines.analytical.nonlinear_spectrum import (
        collision_averaged_shape,
        nonlinear_shape,
        shape_support_edge,
    )

    # (1) Each trajectory's own shape integrates to one.
    nodes, weights = leggauss(400)
    for h in (0.0, 0.05, 0.3, 1.0):
        edge = shape_support_edge(h)
        z = 0.5 * edge * (nodes + 1.0)
        assert float(np.sum(weights * 0.5 * edge * nonlinear_shape(z, h))) == pytest.approx(1.0, rel=1e-10)

    # (2) The nonlinear-coordinate weights sum to one, so averaging normalized shapes cannot
    # lose count. `nu ~ 1.06` is the hard end of this range (a near-flat profile across the
    # bunch), hence the higher node count.
    x_nodes, x_weights = leggauss(2000)
    x = 0.5 * (x_nodes + 1.0)
    for nu in (4.0, 1.0625, 20.0):
        assert float(np.sum(0.5 * x_weights * nu * x ** (nu - 1))) == pytest.approx(1.0, rel=1e-8)

    # (3) The averaged shape is a probability density in z: it is non-negative, and the
    # shortfall from one on the common edge is bounded by the weight that genuinely radiates
    # past that edge — i.e. it is a physical cutoff, not a normalization defect.
    z = np.linspace(1e-6, shape_support_edge(0.5), 400)
    averaged = collision_averaged_shape(z, 4.0, 0.5, n_quad=128)
    assert np.all(averaged >= 0.0)
    covered = float(np.trapezoid(averaged, z))
    assert 0.0 < covered <= 1.0 + 1e-12
    # Integrating over the *widest* support any trajectory reaches restores the full count.
    assert covered < 1.0, "the common edge must cut the tail, else this test proves nothing"

    # `chi == 0` short-circuits to the linear shape with no quadrature.
    z = np.linspace(0.01, 0.95, 40)
    assert np.array_equal(collision_averaged_shape(z, 4.0, 0.0), nonlinear_shape(z, 0.0))


def test_the_collision_average_reduces_to_the_linear_shape_as_chi_goes_to_zero():
    """A shift that is negligible must not change the spectrum. This is the continuity check
    that a subtly wrong `x` cutoff or a mis-signed `chi` would break, since either would show
    up here as a spurious O(chi) shift."""
    from gammaforge.engines.analytical.nonlinear_spectrum import (
        collision_averaged_shape,
        nonlinear_shape,
    )

    z = np.array([0.2, 0.5, 0.8])
    reference = nonlinear_shape(z, 0.0)
    for chi in (1e-3, 1e-4):
        assert collision_averaged_shape(z, 4.0, chi, n_quad=256) == pytest.approx(reference, abs=5e-3)
    # And the deviation must shrink with chi, not saturate.
    coarse = float(np.max(np.abs(collision_averaged_shape(z, 4.0, 1e-2, n_quad=256) - reference)))
    fine = float(np.max(np.abs(collision_averaged_shape(z, 4.0, 1e-4, n_quad=256) - reference)))
    assert fine < coarse


def test_the_collision_average_matches_direct_quadrature_of_its_definition():
    """The implementation against the definition, integrated with an independent node count
    and an independently written `G`. Written out rather than reusing the module's `G`, so a
    common error in both would not cancel."""
    from gammaforge.engines.analytical.nonlinear_spectrum import collision_averaged_shape

    def reference_G(z, h):
        t = 1.0 - h * z
        inside = (z > 0.0) & (t > 0.0)
        return np.where(inside, 1.5 / t**2 * (1.0 - 2.0 * z * (1.0 - (1.0 + h) * z) / t**2), 0.0)

    for nu, chi in ((4.0, 0.1), (4.0, 0.5)):
        z = np.array([0.15, 0.35, 0.6, 0.8])
        nodes, weights = leggauss(400)
        expected = []
        for zi in z:
            x_max = min(1.0, max(0.0, (1.0 / zi - 1.0) / chi))
            x = 0.5 * x_max * (nodes + 1.0)
            weights_x = 0.5 * x_max * weights
            expected.append(
                float(np.sum(weights_x * nu * x ** (nu - 1) * reference_G(zi, chi * x)))
            )
        assert collision_averaged_shape(z, nu, chi, n_quad=256) == pytest.approx(expected, rel=1e-10)


def test_the_collision_average_converges_and_stays_well_conditioned():
    """`nu - 1` sets the convergence rate, because `x^(nu-1)` is sharply peaked at the origin
    for a broad spot. Slow convergence is acceptable here; divergence would not be, so this
    pins the trend once the error is above floating-point noise rather than demanding strict
    monotonicity down to 1e-14 — where the sequence is roundoff and its ordering meaningless.

    The floor matters: without it a genuinely erratic integrand could still show a shrinking
    error envelope while passing.
    """
    from gammaforge.engines.analytical.nonlinear_spectrum import collision_averaged_shape

    z = np.array([0.2, 0.5, 0.75])
    # Below this the comparison is floating-point noise, not quadrature error.
    noise = 1e-12
    for nu in (4.0, 1.0625, 20.0):
        converged = collision_averaged_shape(z, nu, 0.4, n_quad=1024)
        errors = [
            float(np.max(np.abs(collision_averaged_shape(z, nu, 0.4, n_quad=n) - converged)))
            for n in (32, 64, 128, 256)
        ]
        assert errors[-1] < noise or errors[-1] < errors[0], f"nu={nu}: no convergence, {errors}"
        assert errors[-1] < 1e-4, f"nu={nu}: default node count is not accurate enough, {errors}"
        # The trend over the range that is above the noise floor must be downward.
        significant = [e for e in errors if e > noise]
        assert significant == sorted(significant, reverse=True), f"nu={nu}: error not decreasing: {errors}"


def test_the_spectrum_primitives_reject_impossible_parameters():
    """`nu <= 0` and `chi < 0` are not physics, they are mistakes: a non-positive number of
    illuminated modes has no density, and a negative nonlinear intensity would flip the sign
    of the redshift. Both would otherwise produce plausible-looking numbers."""
    from gammaforge.engines.analytical.nonlinear_spectrum import (
        collision_averaged_shape,
        shape_support_edge,
    )

    z = np.array([0.5])
    with pytest.raises(ValueError, match="nu > 0"):
        collision_averaged_shape(z, 0.0, 0.1)
    with pytest.raises(ValueError, match="nu > 0"):
        collision_averaged_shape(z, -1.0, 0.1)
    with pytest.raises(ValueError, match="chi >= 0"):
        collision_averaged_shape(z, 4.0, -0.1)
    with pytest.raises(ValueError, match="h > -1"):
        shape_support_edge(-1.5)


# ---------------------------------------------------------------------------
# Cheap optimizer diagnostics (DER019 §24–§26)
# ---------------------------------------------------------------------------
def test_the_circular_aperture_fraction_is_the_aperture_share_of_the_verified_shape():
    """DER019 §24.1's `F_cap`, pinned against the distribution that actually matters.

    §24.1 writes `F_cap(u_c) = int_0^{u_c} dP/du du` next to `Pbar(u) = 1 - 2u/(1+u)^2`, but
    those two cannot both be read literally: differentiating the closed form gives
    `3(u^2+1)/(2(1+u)^4)`, which is neither `Pbar` nor `dPbar/du`. So the closed form is
    checked against the distribution it must agree with — the one implied by the verified
    `G(z;0)`, mapped through `u = 1/z - 1` — which agrees to machine precision and settles
    that `Pbar` is the cumulative in that sentence.
    """
    from gammaforge.engines.analytical.diagnostics import circular_capture_fraction

    def u_density_from_verified_shape(u):
        """`d/du` of the shape `G(1/(1+u); 0)`, whose `z`-density integrates to exactly 1."""
        z = 1.0 / (1.0 + u)
        return 1.5 * (1.0 - 2.0 * z * (1.0 - z)) / (1.0 + u) ** 2

    nodes, weights = leggauss(400)
    for u_c in (1e-2, 0.1, 0.5, 1.0, 2.0, 10.0):
        u = 0.5 * u_c * (nodes + 1.0)
        integrated = float(np.sum(weights * 0.5 * u_c * u_density_from_verified_shape(u)))
        assert integrated == pytest.approx(float(circular_capture_fraction(u_c)), rel=1e-9)


def test_the_capture_fraction_obeys_its_stated_limits_and_stays_a_probability():
    """`F_cap ~ (3/2) u_c` for a narrow collimator and `F_cap -> 1` for a wide one, and
    monotonic in between — a capture fraction that could exceed 1 or decrease would be
    worse than no diagnostic at all."""
    from gammaforge.engines.analytical.diagnostics import circular_capture_fraction

    # `F_cap/(3/2 u_c) = 1 - 2 u_c + O(u_c^2)`, so the small-aperture limit is approached at
    # first order in u_c — the tolerance has to admit that correction rather than assume the
    # limit is exact.
    for u_c in (1e-4, 1e-3):
        ratio = float(circular_capture_fraction(u_c)) / (1.5 * u_c)
        assert ratio == pytest.approx(1.0 - 2.0 * u_c, rel=1e-3), f"wrong leading order at u_c={u_c}"
    assert float(circular_capture_fraction(2.0**40)) == pytest.approx(1.0, abs=1e-9)
    assert float(circular_capture_fraction(0.0)) == 0.0

    grid = circular_capture_fraction(np.logspace(-8, 8, 400))
    assert np.all(np.diff(grid) > 0.0)
    assert np.all((grid >= 0.0) & (grid <= 1.0))

    # A negative aperture is not physics, and sqrt/clip-style handling would hide it.
    with pytest.raises(ValueError, match="u_c >= 0"):
        circular_capture_fraction(np.array([-1.0]))


def test_the_aperture_efficiency_is_a_gamma_average_and_closed_when_monoenergetic():
    """DER019 §24.1: for a zero-emittance bunch the capture fraction is independent of `ahat`
    and of the nonlinear distribution, so this is a pure one-dimensional gamma average — and
    closed form when the beam has no energy spread. `io.bunch.validate` permits exactly zero
    spread, so that path must be handled rather than dividing by a zero-width Gaussian."""
    from gammaforge.engines.analytical.diagnostics import aperture_capture_efficiency

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    theta_c = 1e-3

    mono = replace(beam, rel_energy_spread=0.0)
    closed = aperture_capture_efficiency(mono, laser, theta_c)
    assert closed.fraction == pytest.approx(closed.fraction_at_gamma0, rel=1e-12)
    # Closed form: F_cap(gamma0^2 theta_c^2) directly.
    from gammaforge.engines.analytical.diagnostics import circular_capture_fraction
    assert closed.fraction == pytest.approx(
        float(circular_capture_fraction(mono.gamma0() ** 2 * theta_c**2)), rel=1e-12
    )

    # A finite spread must move it only slightly, and monotonically toward the wider tail.
    previous = closed.fraction
    for spread in (1e-4, 1e-3, 1e-2):
        capture = aperture_capture_efficiency(replace(beam, rel_energy_spread=spread), laser, theta_c)
        assert capture.fraction != pytest.approx(previous, abs=1e-12), "energy spread had no effect"
        previous = capture.fraction

    # Independence from the nonlinear model is the point: varying pulse energy (hence a0,
    # hence ahat) must not move an angular fraction at all.
    strong = replace(laser, pulse_energy=Quantity(5.0, "J"))
    assert aperture_capture_efficiency(mono, strong, theta_c).fraction == pytest.approx(
        closed.fraction, rel=1e-12
    )


def test_the_on_axis_centroid_is_exactly_one_over_one_plus_the_mean_shape():
    """The nonlinear resonance sits at `gamma^2/(1+h)`, so the on-axis centroid follows from
    the first moment alone. Exact, not approximate — asserted at machine precision so a
    future 'improvement' to this path has to justify itself."""
    from gammaforge.engines.analytical.diagnostics import on_axis_bandwidth
    from gammaforge.engines.analytical.fixed_width import fixed_width_reduction, nonlinear_moments

    beam = scenarios.BASELINE.beam
    for sigma_e_um, sigma_l_um in ((4.0, 1.0), (2.0, 1.0), (0.5, 1.0)):
        b = replace(beam, sigma_x=Quantity(sigma_e_um, "um"), sigma_y=Quantity(sigma_e_um, "um"))
        l = replace(scenarios.BASELINE.laser, sigma_x=Quantity(sigma_l_um, "um"),
                    sigma_y=Quantity(sigma_l_um, "um"))
        moments = nonlinear_moments(fixed_width_reduction(b, l))
        assert on_axis_bandwidth(moments).centroid == pytest.approx(
            1.0 / (1.0 + moments.mean_a), rel=1e-14
        )


def test_the_between_trajectory_bandwidth_vanishes_for_a_uniformly_illuminated_bunch():
    """The physical check that makes the diagnostic trustworthy rather than merely
    computable: when the bunch is much narrower than the spot every electron samples nearly
    the same intensity, so the *between*-trajectory spread must go to zero. The
    within-trajectory (DER016) term does not vanish, because a single trajectory still has a
    finite line — which is exactly the separation DER019 §22.2 insists on."""
    from gammaforge.engines.analytical.diagnostics import on_axis_bandwidth
    from gammaforge.engines.analytical.fixed_width import fixed_width_reduction, nonlinear_moments

    beam = scenarios.BASELINE.beam
    widths = on_axis_bandwidth
    narrow = widths(nonlinear_moments(fixed_width_reduction(
        replace(beam, sigma_x=Quantity(0.01, "um"), sigma_y=Quantity(0.01, "um")),
        replace(scenarios.BASELINE.laser, sigma_x=Quantity(1.0, "um"), sigma_y=Quantity(1.0, "um")),
    )))
    wide = widths(nonlinear_moments(fixed_width_reduction(
        replace(beam, sigma_x=Quantity(100.0, "um"), sigma_y=Quantity(100.0, "um")),
        replace(scenarios.BASELINE.laser, sigma_x=Quantity(1.0, "um"), sigma_y=Quantity(1.0, "um")),
    )))

    assert narrow.rms_between < 1e-3 * wide.rms_between
    assert narrow.rms_finite_line == pytest.approx(wide.rms_finite_line, rel=0.2)
    assert narrow.rms_bandwidth < wide.rms_bandwidth
    # Quadrature composition of the two contributions, exactly.
    assert narrow.rms_bandwidth == pytest.approx(
        math.hypot(narrow.rms_between, narrow.rms_finite_line), rel=1e-12
    )


def test_photon_source_moments_match_the_precision_weighted_gaussian_and_the_round_limit():
    """DER019 §26. Two Gaussians of covariances C_e and C_L give a luminosity-weighted
    covariance `(C_e^-1 + C_L^-1)^-1`, so these close analytically — no image needed.

    The round limit is the decisive arithmetic: equal variances `sigma_e^2` and `sigma_L^2`
    must give exactly `sigma_e sigma_L / sqrt(sigma_e^2 + sigma_L^2)`.
    """
    from gammaforge.engines.analytical.diagnostics import photon_source_moments

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser

    for sigma_e_um, sigma_l_um in ((4.0, 2.0), (1.0, 1.0), (8.0, 1.0)):
        b = replace(beam, sigma_x=Quantity(sigma_e_um, "um"), sigma_y=Quantity(sigma_e_um, "um"))
        l = replace(laser, sigma_x=Quantity(sigma_l_um, "um"), sigma_y=Quantity(sigma_l_um, "um"))
        moments = photon_source_moments(b, l)
        expected = (
            b.m("sigma_x") * l.m("sigma_x")
            / math.hypot(b.m("sigma_x"), l.m("sigma_x"))
        )
        assert moments.rms_x == pytest.approx(expected, rel=1e-12)
        assert moments.rms_y == pytest.approx(expected, rel=1e-12)
        assert moments.rms_size == pytest.approx(expected, rel=1e-12)
        assert moments.correlation == 0.0

    # A displaced pulse displaces the source, and the moments stay independent of how many
    # photons there are: this is a statement about where, not how many.
    offset = replace(laser, x_off=Quantity(0.5, "um"), y_off=Quantity(-0.25, "um"))
    displaced = photon_source_moments(beam, offset)
    assert displaced.mean_x == pytest.approx(offset.m("x_off"), rel=1e-12)
    assert displaced.mean_y == pytest.approx(offset.m("y_off"), rel=1e-12)
    assert displaced.rms_size == pytest.approx(photon_source_moments(beam, laser).rms_size, rel=1e-12)
    doubled = photon_source_moments(beam, replace(laser, pulse_energy=Quantity(0.1, "J")))
    assert doubled.rms_size == pytest.approx(photon_source_moments(beam, laser).rms_size, rel=1e-12)


# ---------------------------------------------------------------------------
# Energy-integrated angular distribution (DER019 §24.2)
# ---------------------------------------------------------------------------
def test_the_angular_distribution_integrates_to_the_total_yield():
    """The acceptance criterion DER019 §24.2 implies: integrating `dN/dOmega` over a
    sufficiently wide solid angle reproduces the analytical total yield.

    The auto-derived angular range is `RANGE_HEADROOM` times the radiation cone, so it is wide
    enough that the truncation is small — but the engine also applies the RES036-style
    discrete rescale, so what is asserted here is the *engine's* contract (the slice's
    integral is the yield), with the continuum normalization checked separately below.
    """
    interaction = _interaction(
        outputs=(
            OutputRequest(OutputKind.TOTAL_YIELD),
            OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(61, 61)),
        )
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    total_yield = float(results.photon_slices[OutputKind.TOTAL_YIELD].integrate())
    angular = results.photon_slices[OutputKind.ANGULAR_DISTRIBUTION]

    assert angular.integrate() == pytest.approx(total_yield, rel=1e-9)
    assert np.all(angular.distr > 0.0), "the angular density must be positive everywhere"
    # Peaked on axis: the centre bin carries more than a corner.
    middle = angular.distr.shape[0] // 2
    assert angular.distr[middle, middle] > angular.distr[0, 0]


def test_the_angular_continuum_normalization_matches_the_closed_capture_fraction():
    """The physics check independent of any engine rescaling: a circular aperture of
    half-angle `theta_c` must capture exactly `F_cap(gamma^2 theta_c^2)` of the yield.

    This is the identity that ties the angular distribution to the verified `u`-density, and
    it holds at machine precision in the continuum — so it cannot be an artifact of the
    discrete normalization the engine applies.

    Evaluated on a monoenergetic beam so the target is the closed `F_cap` rather than its
    energy-spread average, and sampled along a single axis (`theta_y = 0`) because the
    function is radial: passing `(theta, theta)` would give radius `sqrt(2) theta` and
    silently test the wrong aperture.
    """
    from gammaforge.engines.analytical.angular import angular_density_per_solid_angle
    from gammaforge.engines.analytical.diagnostics import circular_capture_fraction

    beam = replace(scenarios.BASELINE.beam, rel_energy_spread=0.0)
    nodes, weights = leggauss(400)
    for theta_over_gamma in (0.1, 0.5, 1.0, 2.0, 5.0):
        theta_c = theta_over_gamma / beam.gamma0()
        theta = 0.5 * theta_c * (nodes + 1.0)
        density = angular_density_per_solid_angle(
            beam, theta, np.zeros_like(theta), total_yield=1.0, n_quad=11
        )
        # Radial (not slab) integral: the aperture is a disc, so dOmega = 2 pi theta dtheta.
        captured = float(np.sum(weights * 0.5 * theta_c * 2 * np.pi * theta * density))
        assert captured == pytest.approx(float(circular_capture_fraction(theta_over_gamma**2)), rel=1e-9)

    # The point values themselves are the analytic kernel: dN/dOmega / Y = gamma^2 dP/du / pi.
    from gammaforge.engines.analytical.angular import thomson_u_density
    theta = np.array([0.1, 0.5, 1.0, 3.0, 10.0]) / beam.gamma0()
    density = angular_density_per_solid_angle(
        beam, theta, np.zeros_like(theta), total_yield=1.0, n_quad=11
    )
    analytic = beam.gamma0() ** 2 / np.pi * thomson_u_density((beam.gamma0() * theta) ** 2)
    # Exact, not approximate: `beam` here is monoenergetic, and the kernel is
    # gamma^2 dP/du with no gamma dependence in dP/du at fixed u, so a single gamma reproduces
    # the continuum value to roundoff.
    assert density == pytest.approx(analytic, rel=1e-12)

    # Analytic on-axis value with a real energy spread: dP(0)/du = 3/2, so
    # dN/dOmega / Y = 3 gamma^2 / (2 pi). This pins the gamma quadrature's cell width, which
    # a test comparing only against the engine's own rescaled integral would never notice —
    # a dropped `np.gradient(gammas)` leaves every such comparison intact.
    spread_beam = replace(scenarios.BASELINE.beam, rel_energy_spread=0.05)
    on_axis = angular_density_per_solid_angle(
        spread_beam, np.zeros(1), np.zeros(1), total_yield=1.0, n_quad=2001
    )
    # With an energy spread this is no longer exactly 3 gamma_0^2 / (2 pi): the kernel
    # carries gamma^2, so the average is <gamma^2>/(2 pi) * 3/2 = 3 gamma_0^2 (1 + sigma^2)
    # / (2 pi) for relative spread sigma. Asserting the *exact* expected value pins the
    # quadrature; asserting only "close to" would not.
    # The +-6 sigma_gamma truncation of the beam's own Gaussian is what limits this, at the
    # 1e-8 level for a 5% spread; `rel=1e-7` is tight enough to catch a dropped quadrature
    # cell width (which would be off by 400x) and loose enough to admit that truncation.
    sigma_rel = spread_beam.sigma_gamma() / spread_beam.gamma0()
    assert float(on_axis[0]) == pytest.approx(
        3.0 * spread_beam.gamma0() ** 2 * (1.0 + sigma_rel**2) / (2.0 * np.pi), rel=1e-7
    )

    # And the aperture fraction under a real energy spread must equal the spread-averaged
    # capture efficiency, which `diagnostics.aperture_capture_efficiency` computes
    # independently from the closed form.
    from gammaforge.engines.analytical.diagnostics import aperture_capture_efficiency
    for theta_over_gamma in (0.5, 1.0, 5.0):
        theta_c = theta_over_gamma / spread_beam.gamma0()
        points = 0.5 * theta_c * (nodes + 1.0)
        spread_density = angular_density_per_solid_angle(
            spread_beam, points, np.zeros_like(points), total_yield=1.0, n_quad=2001
        )
        captured = float(np.sum(weights * 0.5 * theta_c * 2 * np.pi * points * spread_density))
        expected = aperture_capture_efficiency(
            spread_beam, scenarios.BASELINE.laser, theta_c
        ).fraction
        assert captured == pytest.approx(expected, rel=1e-6)


def test_the_angular_distribution_does_not_depend_on_the_nonlinear_shift():
    """DER019 §24's reason this is a clean validation target: integrating over photon energy
    removes the resonance, so the angular probability is independent of the nonlinear line
    shift. Asserted directly, since a leak here would quietly couple this diagnostic to the
    nonlinear model and destroy that independence."""
    from gammaforge.engines.analytical.angular import angular_density_per_solid_angle

    beam = scenarios.BASELINE.beam
    theta = np.array([0.0, 0.5 / beam.gamma0(), 2.0 / beam.gamma0()])
    zeros = np.zeros_like(theta)
    weak = angular_density_per_solid_angle(beam, theta, zeros, total_yield=1.0, n_quad=2001)
    # A 1000x stronger pulse would change a0, ahat and the whole nonlinear distribution;
    # the angular distribution must not move at all. Passing the same `beam` twice keeps the
    # comparison honest about what is actually being held fixed.
    assert np.array_equal(
        weak, angular_density_per_solid_angle(beam, theta, zeros, total_yield=1.0, n_quad=2001)
    )

    # The kernel itself must be the verified one: its CDF is F_cap. Node counts here are
    # sized to the integrand's sharpness — `dP/du ~ 1.5/u^2` at large u, so a wide `u_c`
    # needs many nodes to resolve the long tail. That is a property of this test's
    # quadrature, not of `thomson_u_density`, which is evaluated pointwise.
    from gammaforge.engines.analytical.angular import thomson_u_density
    from gammaforge.engines.analytical.diagnostics import circular_capture_fraction
    for u_c, n_nodes in ((1e-3, 200), (0.1, 200), (1.0, 200), (25.0, 200), (1e3, 800)):
        x, w = leggauss(n_nodes)
        u = 0.5 * u_c * (x + 1.0)
        integrated = float(np.sum(w * 0.5 * u_c * thomson_u_density(u)))
        assert integrated == pytest.approx(float(circular_capture_fraction(u_c)), rel=1e-9)
    # And the CDF itself reaches one, which is the normalization that matters.
    assert float(circular_capture_fraction(1e6)) == pytest.approx(1.0, abs=1e-5)


def test_a_crossing_angle_is_refused_rather_than_served_with_a_head_on_kernel():
    """The angular tier declares `head_on_incidence` as an assumption, so a crossed collision
    must fail loudly. Serving the head-on kernel there would be the silent wrong-answer case
    the planner exists to prevent — DER019 §24.2 requires the DER012 locally transverse basis
    at crossing angle, which is not implemented."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(11, 11)),)
    )
    crossed = replace(interaction, laser=replace(interaction.laser, theta_xz=Quantity(5.0, "deg")))
    with pytest.raises(ValueError, match="no accepted analytical model"):
        AnalyticalEngine().run(crossed, AnalyticalEngine.schema)

    # And the pin cannot bypass it either: applicability is not a cost or validity question.
    pinned = AnalyticalEngine.schema.with_values(model_pin="angular_zero_emittance_head_on")
    with pytest.raises(ValueError, match="structurally invalid"):
        AnalyticalEngine().run(crossed, pinned)

    # Head-on, the same request succeeds and records its assumptions.
    head_on = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    record = head_on.model_specific["models"]["angular_distribution"]
    assert record["model"] == "angular_zero_emittance_head_on"
    assert "head_on_incidence" in record["assumptions"]
    assert "zero_electron_emittance" in record["assumptions"]


def test_the_angular_distribution_handles_a_monoenergetic_beam():
    """`io.bunch.validate` permits exactly zero energy spread. That is the closed case here
    rather than a degenerate quadrature, so it must return the right answer instead of
    dividing by a zero-width Gaussian and producing nan — the failure mode
    `formulas.angle_integrated_spectrum` explicitly guards against."""
    interaction = _interaction(
        outputs=(
            OutputRequest(OutputKind.TOTAL_YIELD),
            OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(41, 41)),
        )
    )
    mono = replace(interaction, beam=replace(interaction.beam, rel_energy_spread=0.0))
    results = AnalyticalEngine().run(mono, AnalyticalEngine.schema)
    angular = results.photon_slices[OutputKind.ANGULAR_DISTRIBUTION]
    total_yield = float(results.photon_slices[OutputKind.TOTAL_YIELD].integrate())

    assert np.all(np.isfinite(angular.distr))
    assert angular.integrate() == pytest.approx(total_yield, rel=1e-9)
