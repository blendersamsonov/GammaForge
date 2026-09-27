"""Sampling, transport, unit-boundary, and prefilter invariants for electron bunches."""

from dataclasses import replace
import math

import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast]

from gammaforge.io.bunch import (
    GaussianElectronBeam,
    drift,
    fit_gaussian,
    illumination_window,
    momenta,
    overlap_time_window,
    prefilter_bunch,
    prefilter_by_illumination,
    propagate,
    sample_gaussian_bunch,
)
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.units import C_CGS, Quantity as Q

N = 100_000
TOL = 0.03


def make_beam(**overrides):
    values = dict(
        bunch_charge=Q(100, "pC"),
        kinetic_energy=Q(100, "MeV"),
        rel_energy_spread=0.01,
        sigma_x=Q(20, "um"),
        sigma_y=Q(30, "um"),
        emit_x=Q(1e-7, "cm * rad"),
        emit_y=Q(2e-7, "cm * rad"),
        sigma_z=Q(100, "um"),
    )
    return GaussianElectronBeam(**{**values, **overrides})


def make_laser(**overrides):
    values = dict(
        pulse_energy=Q(1, "J"),
        wavelength=Q(800, "nm"),
        sigma_x=Q(10, "um"),
        sigma_y=Q(10, "um"),
        duration=Q(30, "fs"),
    )
    return GaussianParaxialLaser(**{**values, **overrides})


def test_sampled_moments_weights_and_mass_shell():
    beam = make_beam()
    bunch = sample_gaussian_bunch(beam, N, seed=1)

    for actual, expected in (
        (np.std(bunch.x), beam.m("sigma_x")),
        (np.std(bunch.y), beam.m("sigma_y")),
        (np.std(bunch.z), beam.m("sigma_z")),
        (np.std(bunch.thx), beam.divergence_x()),
        (np.std(bunch.gamma), beam.sigma_gamma()),
    ):
        assert actual == pytest.approx(expected, rel=TOL)
    assert bunch.weight.sum() == pytest.approx(1.0)

    px, py, pz = momenta(bunch)
    residual = np.abs(bunch.gamma**2 - (1.0 + px**2 + py**2 + pz**2)) / bunch.gamma**2
    assert residual.max() < 1e-13


def test_sampling_reproduces_correlations_and_is_deterministic():
    beam = make_beam(
        rho_x_gamma=0.3,
        rho_y_gamma=-0.2,
        rho_z_gamma=0.5,
        rho_thx_gamma=0.25,
        rho_thy_gamma=-0.15,
        alpha_x=0.6,
        alpha_y=-0.4,
    )
    first = sample_gaussian_bunch(beam, N, seed=7)
    second = sample_gaussian_bunch(beam, N, seed=7)
    for a, b in zip(first.arrays(), second.arrays()):
        assert np.array_equal(a, b)

    for name, expected in (
        ("x", 0.3),
        ("y", -0.2),
        ("z", 0.5),
        ("thx", 0.25),
        ("thy", -0.15),
    ):
        assert np.corrcoef(getattr(first, name), first.gamma)[0, 1] == pytest.approx(
            expected, abs=0.01
        )
    assert np.std(first.gamma) == pytest.approx(beam.sigma_gamma(), rel=TOL)


@pytest.mark.parametrize(
    "change",
    [
        dict(kinetic_energy=Q(200, "MeV")),
        dict(rel_energy_spread=0.05),
        dict(rho_z_gamma=0.4),
        dict(rho_x_gamma=0.3),
    ],
)
def test_energy_edits_change_only_energy_draws(change):
    base = sample_gaussian_bunch(make_beam(), 20_000, seed=5)
    changed = sample_gaussian_bunch(make_beam(**change), 20_000, seed=5)

    for name in ("x", "y", "z", "thx", "thy"):
        assert np.array_equal(getattr(base, name), getattr(changed, name))
    assert not np.array_equal(base.gamma, changed.gamma)


def test_fit_recovers_the_sampled_beam():
    beam = make_beam(rho_x_gamma=0.25, rho_z_gamma=0.4, alpha_x=0.8, alpha_y=-0.3)
    fitted = fit_gaussian(
        sample_gaussian_bunch(beam, N, seed=13), bunch_charge=beam.bunch_charge
    )

    for name in ("sigma_x", "sigma_y", "sigma_z", "emit_x", "emit_y"):
        assert fitted.m(name) == pytest.approx(beam.m(name), rel=TOL)
    assert fitted.rel_energy_spread == pytest.approx(beam.rel_energy_spread, rel=TOL)
    for name in ("alpha_x", "alpha_y", "rho_x_gamma", "rho_z_gamma"):
        assert getattr(fitted, name) == pytest.approx(getattr(beam, name), abs=0.02)


CORRELATED_BEAM = dict(
    rho_x_gamma=0.3,
    rho_y_gamma=-0.25,
    rho_z_gamma=0.4,
    rho_thx_gamma=0.2,
    rho_thy_gamma=-0.1,
    alpha_x=0.5,
    alpha_y=-0.2,
)


def _with_own_moments(bunch, charge):
    return replace(bunch, gaussian_fit=fit_gaussian(bunch, bunch_charge=charge))


@pytest.mark.parametrize("length", [5.0, -12.0, 0.0])
def test_drift_transports_second_moments_and_emittance(length):
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(
        sample_gaussian_bunch(beam, N, seed=17), beam.bunch_charge
    )
    moved = drift(bunch, length)
    refitted = fit_gaussian(moved, bunch_charge=beam.bunch_charge)

    for name in ("sigma_x", "sigma_y", "emit_x", "emit_y"):
        assert moved.gaussian_fit.m(name) == pytest.approx(
            refitted.m(name), rel=1e-11
        )
    for name in ("alpha_x", "alpha_y"):
        assert getattr(moved.gaussian_fit, name) == pytest.approx(
            getattr(refitted, name), rel=1e-11
        )


def test_drift_composes_and_is_reversible():
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(
        sample_gaussian_bunch(beam, 20_000, seed=19), beam.bunch_charge
    )
    stepwise = drift(drift(bunch, 4.0), 7.0)
    single = drift(bunch, 11.0)
    returned = drift(drift(bunch, 25.0), -25.0)

    for a, b in zip(stepwise.arrays(), single.arrays()):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
    for a, b in zip(returned.arrays(), bunch.arrays()):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-15)


def test_zero_dispersion_derivative_preserves_position_energy_covariance():
    beam = make_beam(rho_x_gamma=0.4, alpha_x=0.0, rho_thx_gamma=0.0)
    bunch = sample_gaussian_bunch(beam, 20_000, seed=20)
    start = bunch.gaussian_fit

    for length in (10.0, 40.0, -25.0):
        moved = drift(bunch, length).gaussian_fit
        covariance = moved.rho_x_gamma * moved.m("sigma_x")
        assert covariance == pytest.approx(
            start.rho_x_gamma * start.m("sigma_x"), rel=1e-10
        )


def test_propagation_matches_refitted_description():
    beam = make_beam(**CORRELATED_BEAM)
    bunch = _with_own_moments(
        sample_gaussian_bunch(beam, N, seed=23), beam.bunch_charge
    )
    moved = propagate(bunch, 1e-10)
    refitted = fit_gaussian(moved, bunch_charge=beam.bunch_charge)

    for name in ("sigma_x", "sigma_y", "emit_x", "emit_y"):
        assert moved.gaussian_fit.m(name) == pytest.approx(
            refitted.m(name), rel=1e-5
        )
    assert np.mean(moved.z - bunch.z) == pytest.approx(C_CGS * 1e-10, rel=1e-6)


def test_prefilter_drops_only_unreachable_particles_without_renormalizing():
    beam, laser = make_beam(), make_laser()
    bunch = sample_gaussian_bunch(beam, 20_000, seed=27)
    kept = prefilter_bunch(bunch, laser, 1e-3)
    t0, t1 = overlap_time_window(bunch, laser, 1e-3)
    dropped = t0 > t1
    region = laser.active_region(1e-3)

    for t in np.linspace(-3e-12, 3e-12, 61):
        moved = propagate(bunch, t)
        assert not np.any(region.contains(moved.x, moved.y, moved.z, t) & dropped)
    assert np.allclose(kept.weight, 1.0 / bunch.n_particles)
    assert kept.weight.sum() < 1.0


def test_cgs_si_and_light_time_boundaries_preserve_physical_values():
    assert make_beam(sigma_x=Q(20, "um")) == make_beam(sigma_x=Q(2e-3, "cm"))
    assert make_beam(bunch_charge=Q(100, "pC")) == make_beam(
        bunch_charge=Q(1e-10, "C")
    )
    assert make_beam(sigma_z=Q(1, "ps")).m("sigma_z") == pytest.approx(1e-12 * C_CGS)

    bunch = sample_gaussian_bunch(make_beam(), 1000, seed=1)
    np.testing.assert_allclose(bunch.get("x", "m"), bunch.x * 0.01, rtol=0, atol=0)
    np.testing.assert_allclose(bunch.get("thx", "mrad"), bunch.thx * 1e3)


def _wide_mismatched():
    from gammaforge.validation import scenarios

    beam = replace(
        scenarios.BASELINE.beam,
        sigma_x=Q(400.0, "um"),
        sigma_y=Q(400.0, "um"),
        alpha_x=4.0,
    )
    laser = replace(
        scenarios.BASELINE.laser,
        sigma_x=Q(4.0, "um"),
        sigma_y=Q(4.0, "um"),
        duration=Q(1.0, "ps"),
        z_fx=Q(0.4, "cm"),
        z_fy=Q(0.4, "cm"),
    )
    return beam, laser


def _bunch_yield(bunch, laser, beam, n_full, n_t=121):
    px, py, pz = momenta(bunch)
    bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma
    k_hat, _, _ = laser.focusing_axes()
    flux = C_CGS * (1.0 - (bx * k_hat[0] + by * k_hat[1] + bz * k_hat[2]))
    t_max = 10.0 * math.hypot(
        beam.m("sigma_z"), beam.beta0() * laser.sigma_ct()
    ) / ((1.0 + beam.beta0()) * C_CGS)
    grid = np.linspace(-t_max, t_max, n_t)
    values = [
        np.sum(
            laser.photon_density(
                bunch.x + C_CGS * bx * t,
                bunch.y + C_CGS * by * t,
                bunch.z + C_CGS * bz * t,
                t,
            )
            * flux
        )
        for t in grid
    ]
    return float(np.trapezoid(values, grid)) / n_full


@pytest.mark.tier2
@pytest.mark.parametrize("beta_ff", [0.0, 1.0])
def test_illumination_filter_preserves_the_actual_overlap_yield(beta_ff):
    beam, laser = _wide_mismatched()
    laser = replace(laser, beta_ff=beta_ff)
    bunch = sample_gaussian_bunch(beam, 30_000, 0)
    kept = prefilter_by_illumination(bunch, laser, 1e-6)

    assert 0 < kept.n_particles < bunch.n_particles
    assert _bunch_yield(kept, laser, beam, bunch.n_particles) == pytest.approx(
        _bunch_yield(bunch, laser, beam, bunch.n_particles), rel=5e-3
    )


def test_illumination_window_edges_are_conservatively_dark():
    from gammaforge.validation import scenarios

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 4_000, 0)
    threshold = 1e-6
    t0, t1 = illumination_window(bunch, laser, threshold)
    live = t0 <= t1
    bunch, t0, t1 = bunch.select(live), t0[live], t1[live]
    peak_density = 1.0 / (
        (2.0 * math.pi) ** 1.5
        * laser.m("sigma_x")
        * laser.m("sigma_y")
        * laser.sigma_ct()
    )
    px, py, pz = momenta(bunch)
    bx, by, bz = px / bunch.gamma, py / bunch.gamma, pz / bunch.gamma

    for edge in (t0, t1):
        density = laser.photon_density(
            bunch.x + C_CGS * bx * edge,
            bunch.y + C_CGS * by * edge,
            bunch.z + C_CGS * bz * edge,
            edge,
        )
        ratio = density / (threshold * peak_density)
        assert 1e-4 < float(np.median(ratio)) < 1.0
