"""Beam description, sampling, prefilter and propagation (GRAND_PLAN.md §3.2, Phase 1 exit).

The statistical assertions use a large fixed-seed bunch and tolerances set by
``1/sqrt(n)``, not by trial and error: a second moment estimated from n samples has a
relative error of order ``1/sqrt(2n)``, so 2e5 particles justify the ~1% bounds below.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gammaforge.io.bunch import (
    Bunch,
    GaussianElectronBeam,
    chirp_to_correlation,
    dispersion_to_correlation,
    luminosity_weights,
    illumination_window,
    peak_illumination,
    prefilter_by_illumination,
    prefilter_by_luminosity,
    drift,
    fit_gaussian,
    momenta,
    overlap_time_window,
    prefilter_bunch,
    propagate,
    sample_gaussian_bunch,
    stream,
    validate,
)
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.units import C_CGS, EV_CGS, E_ESU, MEC2_CGS, Quantity as Q

N = 200_000
TOL = 0.02  # ~4 sigma on a second moment estimated from N samples


def make_beam(**overrides) -> GaussianElectronBeam:
    defaults = dict(
        bunch_charge=Q(100, "pC"),
        kinetic_energy=Q(100, "MeV"),
        rel_energy_spread=0.01,
        sigma_x=Q(20, "um"),
        sigma_y=Q(30, "um"),
        emit_x=Q(1e-7, "cm * rad"),
        emit_y=Q(2e-7, "cm * rad"),
        sigma_z=Q(100, "um"),
    )
    return GaussianElectronBeam(**{**defaults, **overrides})


def make_laser(**overrides) -> GaussianParaxialLaser:
    defaults = dict(pulse_energy=Q(1, "J"), wavelength=Q(800, "nm"),
                    sigma_x=Q(10, "um"), sigma_y=Q(10, "um"), duration=Q(30, "fs"))
    return GaussianParaxialLaser(**{**defaults, **overrides})


# -- derived scalars ---------------------------------------------------------
def test_gamma0_from_kinetic_energy():
    beam = make_beam()
    assert beam.gamma0() == pytest.approx(1.0 + 100e6 * EV_CGS / MEC2_CGS)
    assert beam.beta0() == pytest.approx(math.sqrt(1.0 - beam.gamma0() ** -2))


def test_n_electrons_from_charge():
    assert make_beam().n_electrons() == pytest.approx(100e-12 * C_CGS / 10.0 / E_ESU)


def test_divergence_includes_the_twiss_tilt():
    beam = make_beam(alpha_x=1.0)
    assert beam.divergence_x() == pytest.approx(beam.m("emit_x") * math.sqrt(2.0) / beam.m("sigma_x"))


def test_correlation_helpers_invert_the_dimensional_forms():
    beam = make_beam()
    rho = chirp_to_correlation(3.0, beam.m("sigma_z"), beam.sigma_gamma())
    assert rho == pytest.approx(3.0 * beam.m("sigma_z") / beam.sigma_gamma())
    assert dispersion_to_correlation(0.0, beam.m("sigma_x"), beam.sigma_gamma()) == 0.0


# -- sampling ----------------------------------------------------------------
def test_sampled_moments_match_the_beam():
    beam = make_beam()
    bunch = sample_gaussian_bunch(beam, N, seed=1)
    assert bunch.n_particles == N
    assert np.std(bunch.x) == pytest.approx(beam.m("sigma_x"), rel=TOL)
    assert np.std(bunch.y) == pytest.approx(beam.m("sigma_y"), rel=TOL)
    assert np.std(bunch.z) == pytest.approx(beam.m("sigma_z"), rel=TOL)
    assert np.std(bunch.thx) == pytest.approx(beam.divergence_x(), rel=TOL)
    assert np.std(bunch.gamma) == pytest.approx(beam.sigma_gamma(), rel=TOL)
    assert np.mean(bunch.gamma) == pytest.approx(beam.gamma0(), rel=1e-4)


def test_weights_are_relative_and_sum_to_one():
    bunch = sample_gaussian_bunch(make_beam(), 1000, seed=1)
    assert bunch.weight.sum() == pytest.approx(1.0)
    assert np.allclose(bunch.weight, 1.0 / 1000)
    # §3.2: the physical electron count is not recoverable from a Bunch.
    assert not hasattr(bunch, "n_electrons")


def test_mass_shell_holds_identically_by_construction():
    bunch = sample_gaussian_bunch(make_beam(rel_energy_spread=0.2), 10_000, seed=3)
    px, py, pz = momenta(bunch)
    residual = np.abs(bunch.gamma**2 - (1.0 + px**2 + py**2 + pz**2)) / bunch.gamma**2
    assert residual.max() < 1e-13


def test_same_seed_and_parameters_reproduce_the_bunch_exactly():
    first = sample_gaussian_bunch(make_beam(), 5000, seed=11)
    second = sample_gaussian_bunch(make_beam(), 5000, seed=11)
    for a, b in zip(first.arrays(), second.arrays()):
        assert np.array_equal(a, b)


def test_a_different_seed_gives_a_different_bunch():
    assert not np.array_equal(
        sample_gaussian_bunch(make_beam(), 5000, seed=11).x,
        sample_gaussian_bunch(make_beam(), 5000, seed=12).x,
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(kinetic_energy=Q(200, "MeV")),
        dict(rel_energy_spread=0.05),
        dict(rho_z_gamma=0.4),
        dict(rho_x_gamma=0.3),
    ],
)
def test_energy_side_edits_leave_position_and_angle_draws_bit_identical(change):
    """The property the §5 REUSE_INTERMEDIATES tier rests on (§3.2, pinned RNG design).

    Changing gamma0, energy spread, chirp or dispersion must perturb only the gamma draw.
    If it also moved the positions, Stage 0's cache would be silently invalid while the
    cost table still claimed it was reusable.
    """
    base = sample_gaussian_bunch(make_beam(), 20_000, seed=5)
    changed = sample_gaussian_bunch(make_beam(**change), 20_000, seed=5)
    for name in ("x", "y", "z", "thx", "thy"):
        assert np.array_equal(getattr(base, name), getattr(changed, name)), name
    assert not np.array_equal(base.gamma, changed.gamma)


def test_correlations_are_reproduced_and_do_not_inflate_the_energy_spread():
    beam = make_beam(rho_x_gamma=0.3, rho_y_gamma=-0.2, rho_z_gamma=0.5,
                     rho_thx_gamma=0.25, rho_thy_gamma=-0.15, alpha_x=0.6, alpha_y=-0.4)
    bunch = sample_gaussian_bunch(beam, N, seed=7)
    for name, expected in [("x", 0.3), ("y", -0.2), ("z", 0.5),
                           ("thx", 0.25), ("thy", -0.15)]:
        observed = np.corrcoef(getattr(bunch, name), bunch.gamma)[0, 1]
        assert observed == pytest.approx(expected, abs=0.01), name
    # The marginal spread is preserved whatever the correlations are.
    assert np.std(bunch.gamma) == pytest.approx(beam.sigma_gamma(), rel=TOL)


def test_angle_energy_correlation_is_independent_of_the_twiss_tilt():
    """The dispersion derivative is a free parameter, not a consequence of alpha.

    With a tilt and a position-energy correlation but no dispersion derivative, the angle
    must come out *uncorrelated* with the energy — the physical case of a bunch created at
    a waist with dispersion and then drifted, which is exactly the configuration the old
    "gamma couples to the angle only through the position" assumption got wrong.
    """
    beam = make_beam(rho_x_gamma=0.4, alpha_x=1.2, rho_thx_gamma=0.0)
    bunch = sample_gaussian_bunch(beam, N, seed=8)
    assert np.corrcoef(bunch.thx, bunch.gamma)[0, 1] == pytest.approx(0.0, abs=0.01)
    assert np.corrcoef(bunch.x, bunch.gamma)[0, 1] == pytest.approx(0.4, abs=0.01)


def test_twiss_tilt_correlates_position_and_angle():
    beam = make_beam(alpha_x=1.5)
    bunch = sample_gaussian_bunch(beam, N, seed=9)
    expected = -beam.alpha_x / math.sqrt(1.0 + beam.alpha_x**2)
    assert np.corrcoef(bunch.x, bunch.thx)[0, 1] == pytest.approx(expected, abs=0.01)
    # The emittance is what alpha must not change.
    assert np.sqrt(np.linalg.det(np.cov(np.stack([bunch.x, bunch.thx]), bias=True))) == pytest.approx(
        beam.m("emit_x"), rel=TOL
    )


def test_sampling_validates_the_beam():
    with pytest.raises(ValueError):
        sample_gaussian_bunch(make_beam(sigma_x=Q(-1.0, "cm")), 10, seed=0)
    with pytest.raises(ValueError, match="n_particles"):
        sample_gaussian_bunch(make_beam(), 0, seed=0)


# -- fitting (P8: same type in and out) --------------------------------------
def test_fit_recovers_the_beam_it_was_sampled_from():
    beam = make_beam(rho_x_gamma=0.25, rho_z_gamma=0.4, alpha_x=0.8, alpha_y=-0.3)
    bunch = sample_gaussian_bunch(beam, N, seed=13)
    fitted = fit_gaussian(bunch, bunch_charge=beam.bunch_charge)
    assert isinstance(fitted, GaussianElectronBeam)
    for name in ("sigma_x", "sigma_y", "sigma_z", "emit_x", "emit_y"):
        assert fitted.m(name) == pytest.approx(beam.m(name), rel=TOL), name
    assert fitted.rel_energy_spread == pytest.approx(beam.rel_energy_spread, rel=TOL)
    for name in ("alpha_x", "alpha_y", "rho_x_gamma", "rho_z_gamma"):
        assert _value(fitted, name) == pytest.approx(_value(beam, name), abs=0.02), name
    assert fitted.gamma0() == pytest.approx(beam.gamma0(), rel=1e-4)


def test_fit_quality_is_populated_only_by_a_fit():
    beam = make_beam()
    assert beam.fit_quality is None
    fitted = fit_gaussian(sample_gaussian_bunch(beam, 20_000, seed=2), bunch_charge=beam.bunch_charge)
    assert fitted.fit_quality is not None
    # Gaussian data should be noise-limited: the KS statistic matches its own baseline.
    assert abs(fitted.fit_quality["ks_excess"]) < 0.02


def test_fit_quality_flags_non_gaussian_data():
    beam = make_beam()
    bunch = sample_gaussian_bunch(beam, 20_000, seed=4)
    rng = np.random.default_rng(0)
    uniform = dataclasses.replace(bunch, x=rng.uniform(-3e-3, 3e-3, bunch.n_particles))
    fitted = fit_gaussian(uniform, bunch_charge=beam.bunch_charge)
    assert fitted.fit_quality["ks_excess"] > 0.02


def test_fit_quality_is_reproducible():
    beam = make_beam()
    bunch = sample_gaussian_bunch(beam, 5000, seed=6)
    first = fit_gaussian(bunch, bunch_charge=beam.bunch_charge).fit_quality
    second = fit_gaussian(bunch, bunch_charge=beam.bunch_charge).fit_quality
    assert first == second


# -- propagation -------------------------------------------------------------
# The central claim of `drift`/`propagate` is that the attached `GaussianElectronBeam`
# can be carried analytically *in lockstep* with the macroparticles, so nothing ever needs
# refitting. Every test below checks that by refitting the moved bunch and comparing.
#
# The comparison is made against the bunch's **own** fitted moments, not against the beam
# it was sampled from. That distinction is what makes the tests sharp: sampling noise is
# then common to both sides and cancels, exposing three genuinely different precision
# classes that a comparison to the parent beam would blur into one loose tolerance.

CORRELATED_BEAM = dict(rho_x_gamma=0.3, rho_y_gamma=-0.25, rho_z_gamma=0.4,
                       rho_thx_gamma=0.2, rho_thy_gamma=-0.1,
                       alpha_x=0.5, alpha_y=-0.2)


def _value(beam, name: str) -> float:
    """A beam attribute as a plain number, whether it is dimensioned or not.

    Dimensioned fields are Quantities now; correlations and Twiss tilts are bare floats.
    One accessor keeps the comparison loops below readable instead of branching per name.
    """
    return beam.m(name) if name in beam.UNITS else getattr(beam, name)


def _with_own_moments(bunch, charge):
    """The same bunch, described by its own sample moments rather than its parent beam."""
    return dataclasses.replace(bunch, gaussian_fit=fit_gaussian(bunch, bunch_charge=charge))


@pytest.mark.parametrize("length", [5.0, -12.0, 0.0])
def test_drift_transports_sizes_twiss_and_emittance_exactly(length):
    """Class 1: exact to machine precision.

    The Twiss drift is an algebraic identity on *second moments* — `Var(x + Lx')` and
    `Cov(x + Lx', x')` expand exactly — not merely a statement about the parent Gaussian.
    So this must hold to round-off for any bunch, however noisy, and a tolerance anywhere
    near sampling noise would hide a real error in the formula.
    """
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(sample_gaussian_bunch(beam, N, seed=17), beam.bunch_charge)
    drifted = drift(bunch, length)
    refitted = fit_gaussian(drifted, bunch_charge=beam.bunch_charge)
    for name in ("sigma_x", "sigma_y", "alpha_x", "alpha_y", "emit_x", "emit_y"):
        assert _value(drifted.gaussian_fit, name) == pytest.approx(
            _value(refitted, name), rel=1e-11
        ), name


@pytest.mark.parametrize("length", [5.0, -12.0])
def test_drift_transports_the_energy_correlation_within_sampling_noise(length):
    """Class 2: exact in the model, not on the sample.

    Transporting `rho_x_gamma` needs `cov(x', gamma)`, which the beam description does not
    store. The model supplies it — gamma couples to the angle *only* through the position,
    via the Twiss tilt — and that is exactly true of the parent distribution but only
    true to sampling noise of any finite draw. Hence a looser tolerance than the sizes
    above, and deliberately so: this one cannot be tightened by fixing the formula.
    """
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(sample_gaussian_bunch(beam, N, seed=17), beam.bunch_charge)
    drifted = drift(bunch, length)
    refitted = fit_gaussian(drifted, bunch_charge=beam.bunch_charge)
    for name in ("rho_x_gamma", "rho_y_gamma"):
        assert _value(drifted.gaussian_fit, name) == pytest.approx(
            _value(refitted, name), abs=5e-3
        ), name
    # z and gamma are untouched by a drift, so their correlation is not merely close.
    assert drifted.gaussian_fit.rho_z_gamma == refitted.rho_z_gamma


def test_drift_of_the_input_beam_matches_a_refit():
    """The practical case: a freshly sampled bunch still carries its parent beam.

    Looser than the exactness test above because the bunch's own moments differ from the
    beam's by sampling noise before the drift even starts — which is precisely why the
    other tests re-anchor on the bunch's own fit.
    """
    beam = make_beam(**CORRELATED_BEAM)
    bunch = sample_gaussian_bunch(beam, N, seed=17)
    drifted = drift(bunch, 8.0)
    refitted = fit_gaussian(drifted, bunch_charge=beam.bunch_charge)
    for name in ("sigma_x", "sigma_y", "emit_x", "emit_y"):
        assert _value(drifted.gaussian_fit, name) == pytest.approx(
            _value(refitted, name), rel=TOL
        ), name
    for name in ("alpha_x", "alpha_y", "rho_x_gamma", "rho_y_gamma"):
        assert _value(drifted.gaussian_fit, name) == pytest.approx(
            _value(refitted, name), abs=0.02
        ), name


def test_drift_composes():
    """Two drifts must equal one of the combined length — the regression this cost.

    An earlier transport re-derived `cov(x', gamma)` from `alpha` at every step, on the
    assumption that gamma couples to the angle only through the position. That holds for a
    freshly sampled bunch and is destroyed by the first drift, so a *single* drift agreed
    with a refit while two consecutive ones silently did not. Storing `rho_thx_gamma`
    (which a drift leaves untouched) is what makes this exact.
    """
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(sample_gaussian_bunch(beam, 20_000, seed=19), beam.bunch_charge)
    stepwise = drift(drift(bunch, 4.0), 7.0)
    single = drift(bunch, 11.0)
    assert np.allclose(stepwise.x, single.x, rtol=1e-12)
    for name in ("sigma_x", "alpha_x", "sigma_y", "alpha_y",
                 "rho_x_gamma", "rho_y_gamma", "emit_x", "emit_y"):
        assert _value(stepwise.gaussian_fit, name) == pytest.approx(
            _value(single.gaussian_fit, name), rel=1e-10
        ), name


def test_drift_preserves_dispersion_when_there_is_no_dispersion_derivative():
    """A physical invariant that pins the transport's direction, not just its consistency.

    With `rho_thx_gamma = 0`, `cov(x, gamma)` cannot change under a drift at all — the
    angle carries no energy correlation to mix in. `rho_x_gamma` still changes, but only
    because `sigma_x` does, so the *covariance* is what must stay put.
    """
    # Anchored on the input beam, whose rho_thx_gamma is exactly zero -- a *fit* of any
    # finite sample has it only to within noise, which would put a floor under the check.
    beam = make_beam(rho_x_gamma=0.4, alpha_x=0.0, rho_thx_gamma=0.0)
    bunch = sample_gaussian_bunch(beam, 20_000, seed=20)
    start = bunch.gaussian_fit
    for length in (10.0, 40.0, -25.0):
        moved = drift(bunch, length).gaussian_fit
        covariance = moved.rho_x_gamma * moved.m("sigma_x")
        assert covariance == pytest.approx(start.rho_x_gamma * start.m("sigma_x"), rel=1e-10)
        assert moved.rho_thx_gamma == start.rho_thx_gamma


def test_drift_is_reversible():
    # Drifting back must return the description to where it started, not merely close —
    # the transport has no dissipation to lose information to.
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(sample_gaussian_bunch(beam, 20_000, seed=21), beam.bunch_charge)
    there_and_back = drift(drift(bunch, 25.0), -25.0)
    assert np.allclose(there_and_back.x, bunch.x, rtol=1e-10)
    for name in ("sigma_x", "alpha_x", "sigma_y", "alpha_y", "rho_x_gamma", "rho_y_gamma"):
        assert _value(there_and_back.gaussian_fit, name) == pytest.approx(
            _value(bunch.gaussian_fit, name), rel=1e-9, abs=1e-12
        ), name


@pytest.mark.parametrize("dt", [1e-10, -3e-10])
def test_propagate_transports_the_description_alongside_the_particles(dt):
    """Class 3: exact up to the ensemble-reference approximation.

    `propagate` pushes each particle by its own `vz = 1 / sqrt(1 + thx^2 + thy^2)` but
    carries the description by the ensemble step `c * dt`. The two differ at higher order
    in the divergence, so this agrees very well but not to round-off — see the next test,
    which pins the discrepancy on that approximation rather than on a wrong formula.
    """
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(sample_gaussian_bunch(beam, N, seed=23), beam.bunch_charge)
    moved = propagate(bunch, dt)
    refitted = fit_gaussian(moved, bunch_charge=beam.bunch_charge)
    for name in ("sigma_x", "sigma_y", "alpha_x", "alpha_y", "emit_x", "emit_y"):
        assert _value(moved.gaussian_fit, name) == pytest.approx(
            _value(refitted, name), rel=1e-5
        ), name
    for name in ("rho_x_gamma", "rho_y_gamma"):
        assert _value(moved.gaussian_fit, name) == pytest.approx(_value(refitted, name), abs=5e-3)


def test_propagate_description_error_vanishes_with_divergence():
    """The residual above is the ensemble-reference approximation, nothing else.

    Shrinking the divergence must shrink the disagreement monotonically towards zero. The
    exponent is deliberately not asserted: it is 2 or 4 depending on whether `sigma_x` or
    `L * sigma_theta` dominates the drifted size, so pinning it would test the regime
    rather than the approximation.
    """
    errors = []
    for emit in (4e-7, 2e-7, 1e-7, 5e-8):
        beam = make_beam(emit_x=Q(emit, "cm * rad"), emit_y=Q(emit, "cm * rad"))
        bunch = _with_own_moments(sample_gaussian_bunch(beam, 50_000, seed=17), beam.bunch_charge)
        moved = propagate(bunch, 1e-10)
        refitted = fit_gaussian(moved, bunch_charge=beam.bunch_charge)
        errors.append(abs(moved.gaussian_fit.m("sigma_x") - refitted.m("sigma_x")) / refitted.m("sigma_x"))
    assert all(later < earlier for earlier, later in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-10


def test_stream_snapshots_carry_a_consistent_description():
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(sample_gaussian_bunch(beam, 20_000, seed=25), beam.bunch_charge)
    for snapshot in stream(bunch, [0.0, 1e-11, 5e-11]):
        refitted = fit_gaussian(snapshot, bunch_charge=beam.bunch_charge)
        assert snapshot.gaussian_fit.m("sigma_x") == pytest.approx(refitted.m("sigma_x"), rel=1e-5)
        assert snapshot.gaussian_fit.alpha_x == pytest.approx(refitted.alpha_x, rel=1e-5)


def test_drift_moves_particles_ballistically():
    bunch = sample_gaussian_bunch(make_beam(), 100, seed=21)
    drifted = drift(bunch, 3.0)
    assert np.allclose(drifted.x, bunch.x + 3.0 * bunch.thx)
    assert np.array_equal(drifted.z, bunch.z)
    assert np.array_equal(drifted.gamma, bunch.gamma)


def test_propagate_advances_z_at_close_to_the_speed_of_light():
    bunch = sample_gaussian_bunch(make_beam(), 1000, seed=23)
    dt = 1e-11
    moved = propagate(bunch, dt)
    assert np.allclose(moved.z - bunch.z, C_CGS * dt, rtol=1e-6)
    assert np.array_equal(moved.gamma, bunch.gamma)


def test_stream_snapshots_are_independent_of_grid_spacing():
    bunch = sample_gaussian_bunch(make_beam(), 500, seed=25)
    uneven = list(stream(bunch, [0.0, 1e-12, 5e-12]))
    assert np.allclose(uneven[-1].z, propagate(bunch, 5e-12).z)
    assert np.allclose(uneven[0].z, bunch.z)


# -- prefilter (§3.2: a pure optimization) -----------------------------------
def test_prefilter_keeps_every_particle_that_can_reach_the_pulse():
    beam, laser = make_beam(), make_laser()
    bunch = sample_gaussian_bunch(beam, 20_000, seed=27)
    kept = prefilter_bunch(bunch, laser, 1e-3)
    assert 0 < kept.n_particles <= bunch.n_particles

    # Every discarded particle must miss the active region at every time it could be
    # inside: that is what makes the filter safe to apply (§3.2).
    t0, t1 = overlap_time_window(bunch, laser, 1e-3)
    dropped = t0 > t1
    region = laser.active_region(1e-3)
    times = np.linspace(-3e-12, 3e-12, 61)
    for t in times:
        moved = propagate(bunch, t)
        inside = region.contains(moved.x, moved.y, moved.z, t)
        assert not np.any(inside & dropped)


def test_prefilter_never_renormalizes_weights():
    bunch = sample_gaussian_bunch(make_beam(), 20_000, seed=29)
    kept = prefilter_bunch(bunch, make_laser(), 1e-3)
    assert np.allclose(kept.weight, 1.0 / bunch.n_particles)
    assert kept.weight.sum() < 1.0  # honest record of what was dropped


def test_a_looser_prefilter_threshold_keeps_more_particles():
    bunch = sample_gaussian_bunch(make_beam(), 20_000, seed=31)
    counts = [prefilter_bunch(bunch, make_laser(), t).n_particles for t in (1e-1, 1e-3, 1e-8)]
    assert counts[0] <= counts[1] <= counts[2]


def test_overlap_window_is_finite_for_a_head_on_particle_on_axis():
    bunch = Bunch(
        x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
        thx=np.zeros(1), thy=np.zeros(1), gamma=np.full(1, 200.0), weight=np.ones(1),
    )
    t0, t1 = overlap_time_window(bunch, make_laser(), 1e-3)
    assert np.all(np.isfinite(t0)) and np.all(np.isfinite(t1))
    assert t0[0] < t1[0]


def test_overlap_window_handles_a_crossing_angle():
    bunch = sample_gaussian_bunch(make_beam(), 5000, seed=33)
    t0, t1 = overlap_time_window(bunch, make_laser(theta_xz=Q(0.5, "rad")), 1e-3)
    assert np.any(t0 <= t1)


# -- validation --------------------------------------------------------------
def test_validate_rejects_impossible_values():
    for bad in [dict(bunch_charge=Q(0.0, "pC")), dict(kinetic_energy=Q(-1.0, "MeV")),
                dict(sigma_x=Q(0.0, "um")), dict(emit_y=Q(0.0, "cm * rad")),
                dict(sigma_z=Q(0.0, "um")), dict(rel_energy_spread=-0.1)]:
        with pytest.raises(ValueError):
            validate(make_beam(**bad))


def test_validate_rejects_correlations_that_cannot_coexist():
    with pytest.raises(ValueError, match="cannot coexist"):
        validate(make_beam(rho_x_gamma=0.7, rho_y_gamma=0.7, rho_z_gamma=0.7))
    # The bound is on all five jointly, not pairwise: each of these is admissible alone.
    validate(make_beam(rho_x_gamma=0.75))
    validate(make_beam(rho_thx_gamma=0.75))
    with pytest.raises(ValueError, match="cannot coexist"):
        validate(make_beam(rho_x_gamma=0.75, rho_thx_gamma=-0.75))


def test_validate_rejects_a_correlation_outside_the_unit_interval():
    for name in ("rho_x_gamma", "rho_z_gamma", "rho_thx_gamma"):
        with pytest.raises(ValueError, match=f"{name} must be in"):
            validate(make_beam(**{name: 1.0}))


def test_validate_warns_about_a_units_mix_up():
    warnings = validate(make_beam(emit_x=Q(1.0, "cm * rad")))
    assert any("units mix-up" in warning for warning in warnings)


# ---------------------------------------------------------------------------
# Dimensional typing at the engine boundary (GRAND_PLAN.md §2.1, DECISIONS D013)
# ---------------------------------------------------------------------------
# The point of typing these fields is that a unit mistake at an engine boundary becomes an
# exception instead of a silent factor-of-100 error in the answer. These tests assert that
# the mistakes really are caught, not merely that correct code works.
def test_a_bare_number_is_refused():
    with pytest.raises(TypeError, match="must be a pint Quantity"):
        make_beam(sigma_x=20e-4)


def test_a_wrongly_dimensioned_value_is_refused():
    with pytest.raises(TypeError, match="convertible to 'cm'"):
        make_beam(sigma_x=Q(20, "fs"))
    with pytest.raises(TypeError, match="convertible to 'erg'"):
        make_beam(kinetic_energy=Q(100, "um"))


def test_the_same_physical_value_in_different_units_gives_the_same_beam():
    assert make_beam(sigma_x=Q(20, "um")) == make_beam(sigma_x=Q(2e-3, "cm"))
    assert make_beam(bunch_charge=Q(100, "pC")) == make_beam(bunch_charge=Q(1e-10, "C"))


def test_charge_crosses_the_si_gaussian_divide_that_pint_alone_refuses():
    # pint cannot convert coulombs to statcoulombs unaided (§2.1); the `gaussian_charge`
    # context is what makes a charge in pC acceptable where statC is stored.
    assert make_beam(bunch_charge=Q(100, "pC")).m("bunch_charge") == pytest.approx(
        100e-12 * C_CGS / 10.0
    )


def test_a_bunch_length_can_be_given_as_a_duration():
    # `light_time` again, on a dimensioned field rather than through the schema.
    assert make_beam(sigma_z=Q(1, "ps")).m("sigma_z") == pytest.approx(1e-12 * C_CGS)


def test_bunch_arrays_declare_their_units_and_convert_without_copying():
    bunch = sample_gaussian_bunch(make_beam(), 1000, seed=1)
    assert bunch.UNITS["x"] == "cm"
    # Asking for the stored unit costs nothing at all — the array itself comes back.
    assert bunch.get("x", "cm") is bunch.x
    # A real conversion is one scale factor, applied once.
    assert np.allclose(bunch.get("x", "m"), bunch.x * 0.01, rtol=0, atol=0)
    assert np.allclose(bunch.get("thx", "mrad"), bunch.thx * 1e3)


def test_bunch_conversion_checks_the_dimension():
    bunch = sample_gaussian_bunch(make_beam(), 100, seed=1)
    with pytest.raises(Exception):  # pint raises its own DimensionalityError
        bunch.get("x", "s")
    with pytest.raises(KeyError, match="not a dimensioned Bunch array"):
        bunch.get("meta", "cm")


def test_a_laser_angle_can_be_given_in_degrees():
    import math as _math

    assert make_laser(theta_xz=Q(90, "degree")).m("theta_xz") == pytest.approx(_math.pi / 2)


# ---------------------------------------------------------------------------
# Luminosity-weight prefilter (a tolerance filter, unlike prefilter_bunch)
# ---------------------------------------------------------------------------
def _wide_mismatched():
    """Where a geometric cone is loose: a wide bunch against a small, short pulse whose
    foci are displaced, so the cone has to widen conservatively."""
    from gammaforge.validation import scenarios

    beam = dataclasses.replace(scenarios.BASELINE.beam, sigma_x=Q(400.0, "um"),
                               sigma_y=Q(400.0, "um"), alpha_x=4.0)
    laser = dataclasses.replace(scenarios.BASELINE.laser, sigma_x=Q(4.0, "um"),
                                sigma_y=Q(4.0, "um"), duration=Q(1.0, "ps"),
                                z_fx=Q(0.4, "cm"), z_fy=Q(0.4, "cm"))
    return beam, laser


def test_luminosity_weights_rank_particles_by_actual_relevance():
    """The weight must fall off away from the pulse — a particle far off axis contributes
    less than one on it. That is the whole basis for using it as a filter."""
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    w = luminosity_weights(bunch, laser)
    assert w.shape == (bunch.n_particles,)
    assert np.all(w >= 0.0) and np.all(np.isfinite(w))
    radius = np.hypot(bunch.x, bunch.y)
    assert w[radius < np.percentile(radius, 5)].mean() > 10.0 * w[radius > np.percentile(radius, 95)].mean()


def test_prefilter_by_luminosity_keeps_far_fewer_particles_than_the_cone():
    """The measured claim, on the scenario the cone is worst at: a displaced focus forces
    the cone to keep ~94% of the bunch, while ranking by relevance keeps about a third."""
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 60_000, 0)
    cone = prefilter_bunch(bunch, laser, 1e-6)
    weighted = prefilter_by_luminosity(bunch, laser, 1e-4)
    assert cone.n_particles > 0.8 * bunch.n_particles
    assert weighted.n_particles < 0.5 * cone.n_particles


def test_prefilter_by_luminosity_is_monotone_in_epsilon():
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    counts = [prefilter_by_luminosity(bunch, laser, eps).n_particles for eps in (1e-2, 1e-4, 1e-6)]
    assert counts[0] < counts[1] < counts[2] <= bunch.n_particles


def test_prefilter_by_luminosity_selects_without_reweighting():
    """Like `prefilter_bunch` it is a selection, not a reweighting — `N_e` lives on
    `InteractionParameters`, so a caller cannot accidentally rescale the beam by filtering."""
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    filtered = prefilter_by_luminosity(bunch, laser, 1e-3)
    assert filtered.n_particles < bunch.n_particles
    assert np.all(np.isin(filtered.weight, bunch.weight))


def test_prefilter_by_luminosity_rejects_a_nonsense_epsilon():
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 100, 0)
    for bad in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError, match="epsilon"):
            prefilter_by_luminosity(bunch, laser, bad)


def test_peak_illumination_is_a_fraction_of_the_pulse_peak():
    """Dimensionless and bounded: 1 means a particle passes through the focus at the peak
    of the pulse, and nothing can exceed that."""
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    frac = peak_illumination(bunch, laser)
    # >= 0 rather than > 0: a particle far enough out underflows to exactly zero, which is
    # the correct answer for one that never meaningfully meets the pulse.
    assert np.all(frac >= 0.0) and np.all(np.isfinite(frac))
    assert frac.max() <= 1.0 + 1e-9
    assert frac.max() > 1e-3  # some particle does get reasonably well illuminated


def test_peak_illumination_sees_the_dimming_the_cone_ignores():
    """The point of the whole idea. `active_region` bounds the pulse geometrically and
    deliberately ignores the `1/(s1 s2)` amplitude decay, so it keeps particles sitting in
    the large-but-dim diverged beam. Peak illumination includes that decay, so a particle
    far from focus scores far below one near it even when both are 'inside' the cone."""
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 40_000, 0)
    inside_cone = prefilter_bunch(bunch, laser, 1e-6)
    frac = peak_illumination(inside_cone, laser)
    # the cone keeps most of the bunch, yet most of what it keeps is barely illuminated
    assert inside_cone.n_particles > 0.8 * bunch.n_particles
    assert np.median(frac) < 1e-6


def test_prefilter_by_illumination_matches_the_cone_when_nothing_should_be_dropped():
    """On a well-matched collision every electron passes through the pulse, so a correct
    filter keeps everything — exactly as the cone does. A filter that trims here would be
    discarding real signal."""
    from gammaforge.validation import scenarios

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    assert prefilter_bunch(bunch, laser, 1e-6).n_particles == bunch.n_particles
    assert prefilter_by_illumination(bunch, laser, 1e-6).n_particles == bunch.n_particles


def test_prefilter_by_illumination_is_far_tighter_where_the_cone_is_loose():
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 60_000, 0)
    cone = prefilter_bunch(bunch, laser, 1e-6)
    bright = prefilter_by_illumination(bunch, laser, 1e-6)
    assert bright.n_particles < 0.4 * cone.n_particles


def test_prefilter_by_illumination_is_monotone_and_validated():
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    counts = [prefilter_by_illumination(bunch, laser, thr).n_particles
              for thr in (1e-2, 1e-4, 1e-6)]
    assert counts[0] < counts[1] < counts[2]
    for bad in (0.0, 1.0, -1.0):
        with pytest.raises(ValueError, match="threshold"):
            prefilter_by_illumination(bunch, laser, bad)


def test_illumination_filter_follows_a_misaligned_pulse():
    """Offsets must reach the filter, or it would keep the wrong particles entirely: with
    the pulse displaced transversely the surviving particles sit around the new axis."""
    beam, laser = _wide_mismatched()
    shifted = dataclasses.replace(laser, x_off=Q(300.0, "um"))
    bunch = sample_gaussian_bunch(beam, 40_000, 0)
    kept = prefilter_by_illumination(bunch, shifted, 1e-6)
    assert kept.n_particles > 100
    assert kept.x.mean() > 0.5 * shifted.m("x_off")


def _bunch_yield(bunch, laser, beam, n_full, n_t=121):
    """Brute-force yield from a (possibly filtered) bunch, normalized by the ORIGINAL
    particle count — so a filter that drops real signal shows up as a deficit."""
    px, py, pz = momenta(bunch)
    bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma
    k_hat, _, _ = laser.focusing_axes()
    flux = C_CGS * (1.0 - (bx * k_hat[0] + by * k_hat[1] + bz * k_hat[2]))
    b0 = beam.beta0()
    t_max = 10.0 * math.hypot(beam.m("sigma_z"), b0 * laser.sigma_ct()) / ((1.0 + b0) * C_CGS)
    grid = np.linspace(-t_max, t_max, n_t)
    per_t = [float(np.sum(laser.photon_density(bunch.x + C_CGS * bx * ti,
                                               bunch.y + C_CGS * by * ti,
                                               bunch.z + C_CGS * bz * ti, ti) * flux))
             for ti in grid]
    return float(np.trapezoid(per_t, grid)) / n_full


@pytest.mark.parametrize("beta_ff", [0.0, 1.0])
def test_illumination_filter_keeps_what_actually_contributes(beta_ff):
    """The correctness statement, rather than a guess about which way the particle count
    moves: whatever the filter discards must not have mattered. Checked with a flying focus
    too, since that is where the frozen-width approximation was shown to fail badly for the
    *yield* (34% at beta_ff = 1) — for a filter it only has to rank, but that deserves a
    check rather than an assumption."""
    beam, laser = _wide_mismatched()
    laser = dataclasses.replace(laser, beta_ff=beta_ff)
    bunch = sample_gaussian_bunch(beam, 30_000, 0)
    kept = prefilter_by_illumination(bunch, laser, 1e-6)
    assert 0 < kept.n_particles < bunch.n_particles

    full = _bunch_yield(bunch, laser, beam, bunch.n_particles)
    filtered = _bunch_yield(kept, laser, beam, bunch.n_particles)
    assert filtered == pytest.approx(full, rel=5e-3)


def test_illumination_default_threshold_is_the_measured_safe_one():
    """Pins the default at the value measured safe (3.5e-4 induced error), not the value
    that looks analogous to `prefilter_bunch`'s and induces 44%."""
    import inspect

    assert inspect.signature(prefilter_by_illumination).parameters["threshold"].default == 1e-6


def test_illumination_window_is_empty_exactly_when_the_filter_drops():
    """The window and the filter are one computation: a particle is worth keeping precisely
    when there is an interval during which it is above threshold."""
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    t0, t1 = illumination_window(bunch, laser, 1e-6)
    assert np.array_equal(t0 <= t1, peak_illumination(bunch, laser) >= 1e-6)


def test_illumination_window_edges_are_dark_and_conservatively_so():
    """Checks the boundary rather than the interior, which is what catches a wrong root of
    the quadratic. At `t0`/`t1` the *actual* density — from the laser itself, not the
    frozen-width algebra — must already be at or below `threshold` times the pulse peak.

    It comes out about 100x below, and in the safe direction by construction: away from the
    closest approach the true spot is *larger* than the frozen value, so the real intensity
    falls faster than the model and the window errs towards being too wide. For a window
    that is exactly the right way to be wrong — a too-narrow one would truncate the
    interaction, which no step budget could then recover."""
    from gammaforge.validation import scenarios

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 4_000, 0)
    threshold = 1e-6
    t0, t1 = illumination_window(bunch, laser, threshold)
    live = t0 <= t1
    bunch, t0, t1 = bunch.select(live), t0[live], t1[live]

    peak_density = 1.0 / ((2.0 * math.pi) ** 1.5 * laser.m("sigma_x") * laser.m("sigma_y")
                          * laser.sigma_ct())
    px, py, pz = momenta(bunch)
    bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma
    for edge in (t0, t1):
        density = laser.photon_density(bunch.x + C_CGS * bx * edge,
                                       bunch.y + C_CGS * by * edge,
                                       bunch.z + C_CGS * bz * edge, edge)
        # frozen widths, so allow a factor rather than demanding equality
        ratio = density / (threshold * peak_density)
        assert float(np.median(ratio)) < 1.0        # already dark at the edge
        assert float(np.median(ratio)) > 1e-4       # but not absurdly over-wide


def test_illumination_window_is_narrower_than_the_geometric_one():
    from gammaforge.validation import scenarios

    for beam, laser in (_wide_mismatched(), (scenarios.BASELINE.beam, scenarios.BASELINE.laser)):
        bunch = sample_gaussian_bunch(beam, 20_000, 0)
        c0, c1 = overlap_time_window(bunch, laser, 1e-6)
        i0, i1 = illumination_window(bunch, laser, 1e-6)
        live = (i0 <= i1) & (c0 <= c1)
        assert np.median((i1 - i0)[live]) < np.median((c1 - c0)[live])


def test_illumination_window_width_grows_only_logarithmically():
    """Why a generous margin is affordable: the half-width is
    `sqrt(2 ln(peak/threshold) / a)`, so nine decades of threshold buy roughly a doubling
    of the window — the cost of being safe is logarithmic, not linear."""
    from gammaforge.validation import scenarios

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 20_000, 0)
    widths = []
    for thr in (1e-3, 1e-6, 1e-9, 1e-12):
        t0, t1 = illumination_window(bunch, laser, thr)
        widths.append(float(np.median((t1 - t0)[t0 <= t1])))
    assert widths[0] < widths[1] < widths[2] < widths[3]
    assert widths[3] < 3.0 * widths[0]


def test_illumination_window_rejects_a_nonsense_threshold():
    beam, laser = _wide_mismatched()
    bunch = sample_gaussian_bunch(beam, 100, 0)
    for bad in (0.0, 1.0, -1.0):
        with pytest.raises(ValueError, match="threshold"):
            illumination_window(bunch, laser, bad)


def test_illumination_window_spends_a_fixed_step_budget_better():
    """The payoff. With the same particles and the same number of steps per trajectory, the
    tighter window resolves the interaction better because no steps are spent where nothing
    happens."""
    from gammaforge.validation import scenarios

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 8_000, 0)
    c0, c1 = overlap_time_window(bunch, laser, 1e-6)
    i0, i1 = illumination_window(bunch, laser, 1e-6)

    def integrate(t0, t1, n_steps):
        px, py, pz = momenta(bunch)
        bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma
        k_hat, _, _ = laser.focusing_axes()
        flux = C_CGS * (1.0 - (bx * k_hat[0] + by * k_hat[1] + bz * k_hat[2]))
        total = np.zeros(bunch.n_particles)
        for i in range(n_steps):
            ti = t0 + (t1 - t0) * i / (n_steps - 1)
            weight = (1.0 if 0 < i < n_steps - 1 else 0.5) * (t1 - t0) / (n_steps - 1)
            total += laser.photon_density(bunch.x + C_CGS * bx * ti, bunch.y + C_CGS * by * ti,
                                          bunch.z + C_CGS * bz * ti, ti) * flux * weight
        return float(np.sum(total))

    reference = integrate(c0, c1, 2001)
    coarse_cone = abs(integrate(c0, c1, 33) / reference - 1)
    coarse_illum = abs(integrate(i0, i1, 33) / reference - 1)
    assert coarse_illum < 0.5 * coarse_cone
