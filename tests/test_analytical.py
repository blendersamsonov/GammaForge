"""`engines/analytical` (GRAND_PLAN.md §4.3): closed-form yield/width/spectrum formulas
and the `AnalyticalEngine` wrapper.

The worked-example fixture (`_EXAMPLE_BEAM`/`_EXAMPLE_LASER`) is the predecessor's own
test scenario (`ComptonSuite/tests/test_analytical.py`), re-parametrized from SI/pint
`CollisionParams` onto this repo's CGS `GaussianElectronBeam`/`GaussianParaxialLaser` —
100 pC / 200 MeV / 10 um beam, 0.05 J / 0.8 um / 2.5 um / 12.74 fs pulse.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.analytical.engine import AnalyticalEngine
from gammaforge.engines.analytical.formulas import (
    SpectrumWidthBreakdown,
    _electron_sigma2,
    _erfcx,
    _overlap_grid,
    _overlap_quadratic_form,
    angle_integrated_spectrum,
    estimate_spectrum_width,
    estimate_yield,
    overlap_det,
    overlap_yield,
)
from gammaforge.engines.base import Engine
from gammaforge.io.bunch import GaussianElectronBeam, _drift_fit, momenta, sample_gaussian_bunch
from gammaforge.io.laser import GaussianParaxialLaser, lab_frame_axes
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

#: `ComptonSuite/models/analytical.py`'s own `estimate_yield`/`estimate_spectrum_width`,
#: evaluated on the identical worked example (SI/pint `CollisionParams`, verified by
#: actually running that repo). The yield differs at ~1e-6 relative rather than float
#: precision because it is the only quantity here that involves `SIGMA_T`, and the two
#: repos are separately-installed `pint` environments with (very slightly) different
#: CODATA constant tables — not a unit-conversion defect in this port. The width uses no
#: such constant and matches to ~1e-11.
_PREDECESSOR_YIELD = 6644238.68637256
_PREDECESSOR_WIDTH_TOTAL = 4.6671292359002505


def test_estimate_yield_is_positive_finite():
    y = estimate_yield(_EXAMPLE_BEAM, _EXAMPLE_LASER, _EXAMPLE_BEAM.n_electrons())
    assert math.isfinite(y) and y > 0


def test_estimate_yield_reproduces_the_predecessors_worked_example():
    y = estimate_yield(_EXAMPLE_BEAM, _EXAMPLE_LASER, _EXAMPLE_BEAM.n_electrons())
    assert y == pytest.approx(_PREDECESSOR_YIELD, rel=1e-4)


def test_estimate_yield_matches_the_thomson_limit_closed_form():
    """§7's own anchor: "Thomson limit: zero-a0 yield ~ N_e . sigma_T . (overlap) closed
    form." As `nu -> infinity`, `nu * erfcx(nu) -> 1/sqrt(pi)` (the standard asymptotic
    limit), collapsing `estimate_yield`'s full expression to the textbook head-on Gaussian
    luminosity `N_e * n_photons * sigma_T / (2 pi (sigma_ex^2 + sigma_lr0^2))`. A short
    bunch/pulse relative to the transverse sizes drives `nu` large without needing the
    predecessor repo at all.
    """
    from gammaforge.io.units import SIGMA_T_CGS

    beam = GaussianElectronBeam(
        bunch_charge=Quantity(100.0, "pC"),
        kinetic_energy=Quantity(200.0, "MeV"),
        rel_energy_spread=0.001,
        sigma_x=Quantity(100.0, "um"),
        sigma_y=Quantity(100.0, "um"),
        emit_x=Quantity(1.0, "um") * Quantity(1.0, "rad"),
        emit_y=Quantity(1.0, "um") * Quantity(1.0, "rad"),
        sigma_z=Quantity(0.03, "um"),
    )
    laser = GaussianParaxialLaser(
        pulse_energy=Quantity(0.05, "J"),
        wavelength=Quantity(0.8, "um"),
        sigma_x=Quantity(100.0, "um"),
        sigma_y=Quantity(100.0, "um"),
        duration=Quantity(0.1, "fs"),
    )
    N_e = beam.n_electrons()
    y = estimate_yield(beam, laser, N_e)

    sigma_ex, sigma_ey = beam.m("sigma_x"), beam.m("sigma_y")
    sigma_lr0 = math.sqrt(laser.m("sigma_x") * laser.m("sigma_y"))
    thomson_limit = N_e * laser.n_photons() * SIGMA_T_CGS / (2.0 * math.pi * (sigma_ex**2 + sigma_lr0**2))
    assert y == pytest.approx(thomson_limit, rel=1e-6)


# ---------------------------------------------------------------------------
# overlap_yield — the general luminosity overlap integral (docs/DERIVATIONS.md §A)
# ---------------------------------------------------------------------------
def _round_scenario():
    """Round in every sense the closed form needs: equal sizes *and* equal emittances
    (so beta_x == beta_y), round laser, both foci at the origin, alpha == 0."""
    beam = replace(scenarios.BASELINE.beam, emit_y=scenarios.BASELINE.beam.emit_x,
                   sigma_y=scenarios.BASELINE.beam.sigma_x)
    laser = replace(scenarios.BASELINE.laser, sigma_y=scenarios.BASELINE.laser.sigma_x)
    return beam, laser


def _closed_form_round(beam, laser, N_e):
    """The round-beam closed form, evaluated independently of `formulas.estimate_yield`
    and using this repo's own `rayleigh_x()` — the analytic limit `overlap_yield` must
    reproduce, with ``nu = L (1 + beta_0) / (sqrt(2) D)`` (docs/DERIVATIONS.md §A.4)."""
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


def test_overlap_det_is_sigma0_squared_for_a_round_aligned_collision_at_the_origin():
    beam, laser = _round_scenario()
    expected = (beam.m("sigma_x") ** 2 + laser.m("sigma_x") ** 2) ** 2
    assert float(overlap_det(beam, laser, 0.0)) == pytest.approx(expected, rel=1e-13)


def test_overlap_yield_refuses_geometries_its_derivation_does_not_cover():
    """A flying focus makes the spot-size evaluation point time-dependent, which breaks the
    analytic time integration. Refusing beats returning a plausible wrong number (P14c).
    A crossing angle is *not* on this list any more — see the crossing-angle tests below."""
    beam, laser = _round_scenario()
    with pytest.raises(ValueError, match="flying focus"):
        overlap_yield(beam, replace(laser, beta_ff=0.2), beam.n_electrons())


# ---------------------------------------------------------------------------
# Crossing angle (docs/DERIVATIONS.md §A.6)
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


def test_crossing_angle_width_sampling_approximation_is_negligible():
    """Measures the single approximation `overlap_yield` makes with a crossing angle —
    sampling the slowly varying spot sizes at `u = (k.z) z`, dropping `delta = k_x x + k_y y`.

    Deliberately adversarial: a 0.4 rad crossing, a 2 um waist and a 200 um bunch push
    `delta/z_R` past 1, well outside any regime where the naive bound is small. The yield
    still moves by <1e-3 under a *coherent* `+/- delta` shift, because the dropped term
    enters only an even, slowly varying prefactor while the exponent — which carries the
    whole Piwinski suppression — stays exact. The real error is smaller still, since the
    true `delta` averages to zero and this probe does not."""
    beam = replace(scenarios.BASELINE.beam,
                   sigma_x=Quantity(200.0, "um"), sigma_y=Quantity(200.0, "um"))
    laser = replace(scenarios.BASELINE.laser,
                    sigma_x=Quantity(2.0, "um"), sigma_y=Quantity(2.0, "um"),
                    theta_xz=Quantity(0.4, "rad"))
    N_e = beam.n_electrons()

    k_hat, _, _ = laser.focusing_axes()
    delta = math.hypot(beam.m("sigma_x"), laser.m("sigma_x")) * math.hypot(k_hat[0], k_hat[1])
    assert delta / laser.rayleigh_x() > 1.0, "fixture is meant to be adversarial"

    def shifted(u_shift):
        z = _overlap_grid(beam, laser, math.hypot(beam.m("sigma_z"), beam.beta0() * laser.sigma_ct())
                          / (1.0 + beam.beta0()), 20001)
        schur, det_a, sex, sey, s1, s2, _ = _overlap_quadratic_form(beam, laser, z, u_shift=u_shift)
        return float(np.trapezoid(np.exp(-0.5 * schur * z**2) / (sex * sey * s1 * s2 * np.sqrt(det_a)), z))

    base = shifted(0.0)
    assert abs(shifted(+delta) / base - 1.0) < 1e-3
    assert abs(shifted(-delta) / base - 1.0) < 1e-3


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


def test_overlap_yield_differs_from_the_legacy_closed_form_by_the_rayleigh_convention():
    """Pins the size of the `estimate_yield` laser-divergence discrepancy (D040) so it stays
    visible and cannot drift silently. The baseline's hourglass is almost entirely
    laser-driven (laser divergence 3e-2 rad against the bunch's 5e-6), so the factor-4 error
    in `lambda / (pi sigma)` vs `sigma / z_R = lambda / (4 pi sigma)` shows up nearly in
    full.

    3.285 is **not** a physical constant: it is the ratio *at* `scenarios.BASELINE`, and it
    depends on that scenario's laser waist, wavelength and duration through how strongly the
    hourglass suppresses the yield. So if this fails, check whether `BASELINE` moved before
    concluding either formula did."""
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    N_e = beam.n_electrons()
    ratio = overlap_yield(beam, laser, N_e, n_quad=32001) / estimate_yield(beam, laser, N_e)
    assert ratio == pytest.approx(3.285, rel=1e-3)


def test_engine_reports_that_a_crossed_spectrum_has_head_on_shape():
    """The yield accounts for the crossing angle; the spectrum's shape does not. Since the
    engine normalizes SPECTRUM to that yield, the slice's integral is right while its shape
    is not — it must say so rather than looking correct."""
    outputs = (OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(64,)))
    interaction = _interaction(outputs=outputs)
    crossed = replace(interaction, laser=replace(interaction.laser, theta_xz=Quantity(0.02, "rad")))
    engine = AnalyticalEngine()

    assert engine.run(interaction, engine.schema).model_specific["warnings"] == ()
    warned = engine.run(crossed, engine.schema).model_specific["warnings"]
    assert len(warned) == 1 and "head-on" in warned[0]


def test_engine_total_yield_tracks_the_crossing_angle():
    """A 20 mrad crossing more than halves the baseline yield — the engine must carry that
    through, not report the head-on number."""
    interaction = _interaction(outputs=(OutputRequest(OutputKind.TOTAL_YIELD),))
    engine = AnalyticalEngine()
    crossed = replace(interaction, laser=replace(interaction.laser, theta_xz=Quantity(0.02, "rad")))
    head_on_yield = float(engine.run(interaction, engine.schema).photon_slices[OutputKind.TOTAL_YIELD].distr)
    crossed_yield = float(engine.run(crossed, engine.schema).photon_slices[OutputKind.TOTAL_YIELD].distr)
    assert crossed_yield < 0.6 * head_on_yield


def test_estimate_spectrum_width_is_positive_finite():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert isinstance(w, SpectrumWidthBreakdown)
    assert math.isfinite(w.total) and w.total > 0


def test_estimate_spectrum_width_reproduces_the_predecessors_worked_example():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert w.total == pytest.approx(_PREDECESSOR_WIDTH_TOTAL, rel=1e-6)


def test_estimate_spectrum_width_grows_with_larger_collimation_angle():
    narrow = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-4)
    wide = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-2)
    assert wide.total > narrow.total


def test_spectrum_width_breakdown_total_is_hypot_of_components():
    w = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-3)
    assert w.total == pytest.approx(
        math.hypot(w.collimation, w.emittance, w.energy_spread, w.nonlinearity)
    )


def test_spectrum_width_breakdown_components_move_independently():
    narrow = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-4)
    wide = estimate_spectrum_width(_EXAMPLE_BEAM, _EXAMPLE_LASER, theta_col=1e-2)
    assert wide.collimation > narrow.collimation
    for field in ("emittance", "energy_spread", "nonlinearity"):
        assert getattr(wide, field) == pytest.approx(getattr(narrow, field))


def test_angle_integrated_spectrum_shape_and_scalar_input():
    s_array = np.linspace(0.01, 0.99, 16)
    out_array = angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons(), s_array
    )
    assert out_array.shape == s_array.shape
    assert np.all(np.isfinite(out_array)) and np.all(out_array >= 0)

    out_scalar = angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons(), 0.5
    )
    assert np.ndim(out_scalar) == 0 or isinstance(out_scalar, float)


def test_angle_integrated_spectrum_zero_outside_kinematic_range():
    gamma0 = 100.0
    s_far_beyond_edge = np.array([gamma0**2 * 1.5])
    out = angle_integrated_spectrum(gamma0, gamma0 * 1e-6, 1.0, s_far_beyond_edge)
    assert out[0] == 0.0


def test_angle_integrated_spectrum_rejects_zero_energy_spread():
    """`io.bunch.validate` permits `rel_energy_spread == 0` (only `< 0` raises), so
    `sigma_gamma == 0` is a legal beam — but it makes this function's quadrature grid
    degenerate (a zero-width Gaussian divided by its own zero width), which would
    otherwise return `nan` silently. This repo's convention is an explicit error over a
    silent fallback."""
    with pytest.raises(ValueError, match="sigma_gamma"):
        angle_integrated_spectrum(100.0, 0.0, 1.0, 0.5)


def test_angle_integrated_spectrum_fast_at_reported_scale():
    """Regression guard for the OOM bug class the predecessor's refactor fixed (see
    `formulas.py`'s module docstring): no argument here scales with `n_particles`."""
    s_array = np.linspace(1e-3, 1.0 - 1e-3, 2048)
    out = angle_integrated_spectrum(
        _EXAMPLE_BEAM.gamma0(), _EXAMPLE_BEAM.sigma_gamma(), _EXAMPLE_BEAM.n_electrons(), s_array
    )
    assert out.shape == s_array.shape
    assert np.all(np.isfinite(out))


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


def test_analytical_engine_conforms_to_the_engine_protocol():
    assert isinstance(AnalyticalEngine(), Engine)


def test_supported_outputs_matches_what_run_actually_fills():
    requests = (
        OutputRequest(OutputKind.TOTAL_YIELD),
        OutputRequest(OutputKind.SPECTRUM, resolution=(64,)),
    )
    interaction = _interaction(outputs=requests)
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert set(results.photon_slices) == set(AnalyticalEngine.supported_outputs)


def test_unsupported_output_kinds_are_silently_omitted_not_errored():
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(3, 3)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert set(results.photon_slices) == {OutputKind.TOTAL_YIELD}


def test_unsupported_temporal_envelope_request_does_not_crash():
    """`io.target.auto_ranges`'s `TEMPORAL_ENVELOPE` branch requires a bunch and raises
    without one; a `Target` requesting it alongside a supported output must not reach
    that branch just because this engine happens to skip the kind (a regression this
    engine's own review caught: filtering unsupported requests must happen before
    `auto_ranges` runs, not after)."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.TEMPORAL_ENVELOPE, resolution=(8,)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert set(results.photon_slices) == {OutputKind.TOTAL_YIELD}


def test_spectrum_integral_equals_total_yield_exactly():
    """§7: "Total yield: integral spectrum = total_yield -- exact identities, not
    tolerances, where the contract guarantees them." `DECISIONS.md` D036 makes this exact
    by construction for analytical."""
    interaction = _interaction(
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(200,)))
    )
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    total_yield = float(results.photon_slices[OutputKind.TOTAL_YIELD].distr)
    spectrum_integral = results.photon_slices[OutputKind.SPECTRUM].integrate()
    assert spectrum_integral == pytest.approx(total_yield, rel=1e-9)


def test_spectrum_grid_integral_correction_factor_is_near_one():
    """The rescale in D036 always makes the integral exact; this guards that the factor
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


def test_analytical_engine_is_independent_of_n_particles():
    """analytical never touches `interaction.bunch` — results must be bit-identical
    across `n_particles`, the structural version of "the old OOM bug class must not
    return" (§4.3)."""
    outputs = (OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, resolution=(64,)))
    small = AnalyticalEngine().run(_interaction(n_particles=100, outputs=outputs), AnalyticalEngine.schema)
    large = AnalyticalEngine().run(_interaction(n_particles=100_000, outputs=outputs), AnalyticalEngine.schema)
    for request in outputs:
        assert np.array_equal(
            small.photon_slices[request.kind].distr, large.photon_slices[request.kind].distr
        )


def test_model_specific_carries_the_width_breakdown_and_is_charge_independent():
    """`Results.scaled()` copies `model_specific` verbatim (unscaled) — these three
    values must actually be charge-independent, or a charge-rescaled `Results` would
    silently carry a stale number."""
    interaction = _interaction(outputs=(OutputRequest(OutputKind.TOTAL_YIELD),))
    results = AnalyticalEngine().run(interaction, AnalyticalEngine.schema)
    assert isinstance(results.model_specific["spectrum_width_fwhm"], SpectrumWidthBreakdown)

    doubled = replace(interaction, N_e=interaction.N_e * 2.0)
    results_doubled = AnalyticalEngine().run(doubled, AnalyticalEngine.schema)
    assert results_doubled.model_specific["spectrum_width_fwhm"] == results.model_specific["spectrum_width_fwhm"]
    assert results_doubled.model_specific["a0_peak"] == results.model_specific["a0_peak"]
    assert results_doubled.model_specific["n_photons"] == results.model_specific["n_photons"]
    # ... while the yield itself does scale.
    y = float(results.photon_slices[OutputKind.TOTAL_YIELD].distr)
    y_doubled = float(results_doubled.photon_slices[OutputKind.TOTAL_YIELD].distr)
    assert y_doubled == pytest.approx(2.0 * y)
