"""Independent checks for DER023 geometry and the retarded-time laser envelope."""

from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma.gaussian_bound import (luminosity_lower_bound,
                                                    luminosity_upper_bound,
                                                    trajectory_bound_geometry)
from gammaforge.engines.xigma.stages import integrate_trajectories, photon_density_scale
from gammaforge.io.laser import GaussianParaxialLaser, PulseTrainParaxialLaser
from gammaforge.io.units import C_CGS, SIGMA_T_CGS, Quantity
from gammaforge.validation import scenarios

pytestmark = [pytest.mark.tier1, pytest.mark.fast]


def test_envelope_uses_retarded_time_while_carrier_keeps_paraxial_phase():
    laser = GaussianParaxialLaser(
        pulse_energy=Quantity(1.0, "J"), wavelength=Quantity(800, "nm"),
        sigma_x=Quantity(20, "um"), sigma_y=Quantity(20, "um"),
        duration=Quantity(30, "fs"),
    )
    x = laser.m("sigma_x")
    z = -laser.rayleigh_x()
    u = -z
    t = u / C_CGS
    xi1, xi2, local_u, u_spot, ct = laser._local_coordinates(x, 0.0, z, t)
    s1, s2 = laser.spot_sizes(u_spot)
    transverse = np.exp(-0.5 * ((xi1 / s1)**2 + (xi2 / s2)**2)) / (2 * np.pi * s1 * s2)
    expected = transverse * laser.temporal_envelope.peak_value(np) / C_CGS
    np.testing.assert_allclose(laser.photon_density(x, 0.0, z, t), expected, rtol=1e-14)
    assert abs(float(laser._paraxial_phase(xi1, xi2, local_u, u_spot, ct, np))) > 0.1

    train = PulseTrainParaxialLaser(
        pulse_energy=Quantity(1.0, "J"), wavelength=Quantity(800, "nm"),
        sigma_x=Quantity(20, "um"), sigma_y=Quantity(20, "um"),
        subpulse_duration=Quantity(10, "fs"), repetition_period=Quantity(100, "fs"),
        n_subpulses=2,
    )
    delays = train.subpulse_delays()
    for delay in delays:
        density = train.photon_density(0.0, 0.0, 0.0, delay)
        expected = train.temporal_envelope.envelope(delay, np) / (2 * np.pi * train.m("sigma_x")**2 * C_CGS)
        np.testing.assert_allclose(density, expected, rtol=1e-14)


@pytest.mark.parametrize("theta_xz,theta_yz", [(0.0, 0.0), (0.05, -0.03)])
def test_der023_geometry_and_upper_bound_against_direct_trajectory_integral(theta_xz, theta_yz):
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=64))
    interaction = scenarios.build(scenario)
    bunch = interaction.bunch
    laser = replace(interaction.laser, theta_xz=Quantity(theta_xz, "rad"),
                    theta_yz=Quantity(theta_yz, "rad"))
    f, d2, eta_min, curvature = trajectory_bound_geometry(bunch, laser)
    assert np.all(f > 0)
    assert np.all(d2 >= 0)
    assert np.all(curvature >= C_CGS / (2 * laser.rayleigh_x()) * (1 - 1e-12))
    k, f1, f2 = laser.focusing_axes()
    for eta in (-laser.m("duration"), 0.0, laser.m("duration")):
        norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
        direction = np.stack((bunch.thx / norm, bunch.thy / norm, 1.0 / norm), axis=1)
        initial = np.stack((bunch.x - laser.m("x_off"),
                            bunch.y - laser.m("y_off"), bunch.z), axis=1)
        time = (eta + laser.m("t_off") + (initial @ k) / C_CGS) / f
        position = initial + C_CGS * direction * time[:, None] - laser.m("z_fx") * k
        metric = ((position @ f1)**2 + (position @ f2)**2) / (2 * laser.m("sigma_x")**2)
        metric += (position @ k)**2 / laser.rayleigh_x()**2
        np.testing.assert_allclose(metric, d2 + curvature**2 * (eta - eta_min)**2,
                                   rtol=1e-11, atol=1e-10)

    upper = luminosity_upper_bound(bunch, laser)
    lower = luminosity_lower_bound(bunch, laser)
    direct = integrate_trajectories(bunch, laser, interaction.N_e, n_steps=4096)
    common = (interaction.N_e * bunch.weight * photon_density_scale(laser) *
              C_CGS * SIGMA_T_CGS * laser.intensity_peak() /
              laser.temporal_envelope.peak_value(np))
    assert np.all(direct.luminosity <= common * upper * (1 + 1e-9))
    assert np.all(common * lower <= direct.luminosity * (1 + 1e-9))


def test_weighted_rule_converges_to_direct_moments():
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=32))
    interaction = scenarios.build(scenario)
    direct = integrate_trajectories(interaction.bunch, interaction.laser,
                                    interaction.N_e, n_steps=2048)
    coarse = integrate_trajectories(interaction.bunch, interaction.laser,
                                    interaction.N_e, quadrature="auto", gaussian_order=32)
    fine = integrate_trajectories(interaction.bunch, interaction.laser,
                                  interaction.N_e, quadrature="auto", gaussian_order=256)
    for name in ("luminosity", "a0_shape", "var_a_shape"):
        reference = getattr(direct, name)
        scale = max(float(np.max(np.abs(reference))), 1e-20)
        coarse_error = np.max(np.abs(getattr(coarse, name) - reference)) / scale
        fine_error = np.max(np.abs(getattr(fine, name) - reference)) / scale
        assert fine_error < coarse_error / 4
    chunked = integrate_trajectories(interaction.bunch, interaction.laser,
                                     interaction.N_e, quadrature="auto",
                                     gaussian_order=256, chunk=7)
    np.testing.assert_allclose(chunked.luminosity, fine.luminosity, rtol=1e-14)
    np.testing.assert_allclose(chunked.var_a_shape, fine.var_a_shape, rtol=1e-14)


def test_cumulative_discard_bound_exceeds_measured_luminosity_loss():
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=32))
    interaction = scenarios.build(scenario)
    bunch = replace(interaction.bunch,
                    x=np.r_[interaction.bunch.x[:16], np.full(16, 0.01)])
    full = integrate_trajectories(bunch, interaction.laser, interaction.N_e,
                                  quadrature="auto", gaussian_order=64)
    filtered = integrate_trajectories(bunch, interaction.laser, interaction.N_e,
                                      quadrature="auto", gaussian_order=64,
                                      discard_tolerance=0.1)
    assert 0 < filtered.discard_certificate <= 0.1
    measured_loss = (full.total_yield() - filtered.total_yield()) / full.total_yield()
    assert measured_loss <= filtered.discard_certificate


def test_unsupported_astigmatic_laser_keeps_generic_midpoint_path():
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=16))
    interaction = scenarios.build(scenario)
    laser = replace(interaction.laser, sigma_y=Quantity(27, "um"))
    midpoint = integrate_trajectories(interaction.bunch, laser, interaction.N_e, n_steps=64)
    selected = integrate_trajectories(interaction.bunch, laser, interaction.N_e,
                                      n_steps=64, quadrature="auto")
    np.testing.assert_array_equal(selected.luminosity, midpoint.luminosity)
    np.testing.assert_array_equal(selected.a0_shape, midpoint.a0_shape)


def test_separated_pulse_train_composite_rule_matches_direct_integral():
    scenario = replace(scenarios.BASELINE,
                       sampling=replace(scenarios.BASELINE.sampling, n_particles=32))
    interaction = scenarios.build(scenario)
    laser = interaction.laser
    train = PulseTrainParaxialLaser(
        pulse_energy=laser.pulse_energy, wavelength=laser.wavelength,
        sigma_x=laser.sigma_x, sigma_y=laser.sigma_y,
        subpulse_duration=Quantity(30, "fs"), repetition_period=Quantity(500, "fs"),
        n_subpulses=3,
    )
    direct = integrate_trajectories(interaction.bunch, train, interaction.N_e,
                                    n_steps=4096)
    composite = integrate_trajectories(interaction.bunch, train, interaction.N_e,
                                       quadrature="auto", gaussian_order=32)
    np.testing.assert_allclose(composite.total_yield(), direct.total_yield(), rtol=1e-6)
    common = (interaction.N_e * interaction.bunch.weight * photon_density_scale(train) *
              C_CGS * SIGMA_T_CGS * train.intensity_peak() /
              train.temporal_envelope.peak_value(np))
    upper = luminosity_upper_bound(interaction.bunch, train)
    lower = luminosity_lower_bound(interaction.bunch, train)
    assert np.all(common * lower <= direct.luminosity * (1 + 1e-9))
    assert np.all(direct.luminosity <= common * upper * (1 + 1e-9))
