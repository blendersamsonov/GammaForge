"""Tests for PulseTrainParaxialLaser (handoff: temporal-modulation-laser-2026-09-08.md).

Covers:
- Protocol adherence (LaserField)
- Single-subpulse equivalence to GaussianParaxialLaser (Np = 1) to machine precision
- 3D spatial photon density normalization (integral = 1.0) across multiple time points
- Total photon count invariance across Np and repetition period
- Active region enclosure of all sub-pulse centers
- Exact cycle-averaged intensity superposition over constituent sub-pulses
- End-to-end Xigma Stage 0 interaction: train yield vs sum of sub-pulse yields
- Yield decay as duty cycle D -> 0 (hourglass geometric divergence)
"""

import math
import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast]

from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.bunch import GaussianElectronBeam, sample_gaussian_bunch
from gammaforge.io.laser import (
    GaussianParaxialLaser,
    LaserField,
    PulseTrainParaxialLaser,
    fit_gaussian_paraxial,
    validate,
)
from gammaforge.io.units import C_CGS, HBAR_CGS, Quantity


def test_protocol_adherence():
    laser = PulseTrainParaxialLaser(
        pulse_energy=Quantity(0.5, "J"),
        wavelength=Quantity(800.0, "nm"),
        sigma_x=Quantity(20.0, "um"),
        sigma_y=Quantity(20.0, "um"),
        subpulse_duration=Quantity(30.0, "fs"),
        repetition_period=Quantity(300.0, "fs"),
        n_subpulses=5,
    )
    assert isinstance(laser, LaserField)
    assert hasattr(laser, "intensity_profile")
    assert hasattr(laser, "a0_profile")
    assert hasattr(laser, "field")
    assert hasattr(laser, "active_region")
    assert hasattr(laser, "omega0")
    assert hasattr(laser, "photon_energy")
    assert hasattr(laser, "intensity_peak")


def test_single_subpulse_equivalence_to_gaussian():
    """When N_p = 1, PulseTrainParaxialLaser must match GaussianParaxialLaser to 1e-14."""
    energy = Quantity(1.2, "J")
    wavelength = Quantity(800.0, "nm")
    sigma_x = Quantity(22.0, "um")
    sigma_y = Quantity(28.0, "um")
    duration = Quantity(35.0, "fs")
    period = Quantity(1.0, "ps")
    z_fx = Quantity(0.5, "mm")
    z_fy = Quantity(-0.3, "mm")
    x_off = Quantity(4.0, "um")
    y_off = Quantity(-2.0, "um")
    t_off = Quantity(12.0, "fs")
    theta_xz = Quantity(0.08, "rad")
    theta_yz = Quantity(-0.04, "rad")
    psi_focus = Quantity(0.15, "rad")
    psi_pol = Quantity(-0.25, "rad")
    ellipticity = 0.35
    beta_ff = 0.12

    train = PulseTrainParaxialLaser(
        pulse_energy=energy,
        wavelength=wavelength,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        subpulse_duration=duration,
        repetition_period=period,
        n_subpulses=1,
        z_fx=z_fx,
        z_fy=z_fy,
        x_off=x_off,
        y_off=y_off,
        t_off=t_off,
        theta_xz=theta_xz,
        theta_yz=theta_yz,
        psi_focus=psi_focus,
        psi_pol=psi_pol,
        ellipticity=ellipticity,
        beta_ff=beta_ff,
    )
    single = GaussianParaxialLaser(
        pulse_energy=energy,
        wavelength=wavelength,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        duration=duration,
        z_fx=z_fx,
        z_fy=z_fy,
        x_off=x_off,
        y_off=y_off,
        t_off=t_off,
        theta_xz=theta_xz,
        theta_yz=theta_yz,
        psi_focus=psi_focus,
        psi_pol=psi_pol,
        ellipticity=ellipticity,
        beta_ff=beta_ff,
    )

    assert math.isclose(train.omega0(), single.omega0(), rel_tol=1e-15)
    assert math.isclose(train.photon_energy(), single.photon_energy(), rel_tol=1e-15)
    assert math.isclose(train.n_photons(), single.n_photons(), rel_tol=1e-15)
    assert math.isclose(train.rayleigh_x(), single.rayleigh_x(), rel_tol=1e-15)
    assert math.isclose(train.rayleigh_y(), single.rayleigh_y(), rel_tol=1e-15)
    assert math.isclose(train.sigma_ct(), single.sigma_ct(), rel_tol=1e-15)
    assert math.isclose(train.a0_peak(), single.a0_peak(), rel_tol=1e-14)
    assert math.isclose(train.intensity_peak(), single.intensity_peak(), rel_tol=1e-14)

    # Active region
    reg_train = train.active_region(1e-3)
    reg_single = single.active_region(1e-3)
    assert np.allclose(reg_train.axis, reg_single.axis, atol=1e-15)
    assert np.allclose(reg_train.origin, reg_single.origin, atol=1e-15)
    assert math.isclose(reg_train.radius, reg_single.radius, rel_tol=1e-14)
    assert math.isclose(reg_train.radius_slope, reg_single.radius_slope, rel_tol=1e-14)
    assert math.isclose(reg_train.half_length, reg_single.half_length, rel_tol=1e-14)

    # Field profiles on a 3D grid
    x = np.linspace(-40e-4, 40e-4, 5)
    y = np.linspace(-40e-4, 40e-4, 5)
    z = np.linspace(-100e-4, 100e-4, 5)
    t = 10e-15
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    dens_train = train.photon_density(X, Y, Z, t)
    dens_single = single.photon_density(X, Y, Z, t)
    np.testing.assert_allclose(dens_train, dens_single, rtol=1e-14, atol=1e-25)

    int_train = train.intensity_profile(X, Y, Z, t)
    int_single = single.intensity_profile(X, Y, Z, t)
    np.testing.assert_allclose(int_train, int_single, rtol=1e-14, atol=1e-25)

    a0_train = train.a0_profile(X, Y, Z, t)
    a0_single = single.a0_profile(X, Y, Z, t)
    np.testing.assert_allclose(a0_train, a0_single, rtol=1e-14, atol=1e-25)

    field_train = train.field(X, Y, Z, t)
    field_single = single.field(X, Y, Z, t)
    np.testing.assert_allclose(field_train, field_single, rtol=1e-14, atol=1e-25)

    # fit_gaussian_paraxial identity for Np=1
    fit = fit_gaussian_paraxial(train)
    assert isinstance(fit, GaussianParaxialLaser)
    assert math.isclose(fit.m("pulse_energy"), train.m("pulse_energy"), rel_tol=1e-15)


def test_spatial_normalization():
    """Numerically integrate photon_density over 3D space at various t. Integral must be 1.0."""
    laser = PulseTrainParaxialLaser(
        pulse_energy=Quantity(0.1, "J"),
        wavelength=Quantity(800.0, "nm"),
        sigma_x=Quantity(20.0, "um"),
        sigma_y=Quantity(20.0, "um"),
        subpulse_duration=Quantity(20.0, "fs"),
        repetition_period=Quantity(100.0, "fs"),
        n_subpulses=3,
    )
    t_rep = laser.m("repetition_period")

    # Grid covering transverse 5 sigma and longitudinal full burst + 5 sigma_ct
    sx = laser.m("sigma_x")
    sy = laser.m("sigma_y")
    s_ct = laser.sigma_ct()
    burst_c = C_CGS * (laser.n_subpulses - 1) * t_rep

    x = np.linspace(-4.5 * sx, 4.5 * sx, 35)
    y = np.linspace(-4.5 * sy, 4.5 * sy, 35)

    for t_eval in [-t_rep, 0.0, t_rep]:
        # Laser propagates towards -z (head-on), so center at t is at z = -c*t
        z_center = -C_CGS * t_eval
        z_span = 0.5 * burst_c + 4.5 * s_ct
        z = np.linspace(z_center - z_span, z_center + z_span, 101)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        dens = laser.photon_density(X, Y, Z, t_eval)

        # 3D trapezoidal integration
        integral = np.trapezoid(np.trapezoid(np.trapezoid(dens, z, axis=2), y, axis=1), x, axis=0)
        assert math.isclose(integral, 1.0, rel_tol=1e-3), f"Integral at t={t_eval} was {integral}"


def test_total_photon_count_invariance():
    """laser.n_photons() equals E_tot / (hbar * omega0) regardless of N_p or T_rep."""
    e_tot = 0.5  # J
    wl = 800e-9  # m
    omega0 = 2.0 * math.pi * C_CGS / (wl * 100.0)
    expected_photons = (e_tot * 1e7) / (HBAR_CGS * omega0)

    for n_p in [1, 5, 20]:
        for t_rep_fs in [50.0, 200.0, 1000.0]:
            laser = PulseTrainParaxialLaser(
                pulse_energy=Quantity(e_tot, "J"),
                wavelength=Quantity(wl * 1e9, "nm"),
                sigma_x=Quantity(15.0, "um"),
                sigma_y=Quantity(15.0, "um"),
                subpulse_duration=Quantity(25.0, "fs"),
                repetition_period=Quantity(t_rep_fs, "fs"),
                n_subpulses=n_p,
            )
            assert math.isclose(laser.n_photons(), expected_photons, rel_tol=1e-14)


def test_active_region_enclosure():
    """Every sub-pulse center (0, 0, -c*t_k) at t=0 is inside active_region."""
    laser = PulseTrainParaxialLaser(
        pulse_energy=Quantity(1.0, "J"),
        wavelength=Quantity(800.0, "nm"),
        sigma_x=Quantity(20.0, "um"),
        sigma_y=Quantity(20.0, "um"),
        subpulse_duration=Quantity(25.0, "fs"),
        repetition_period=Quantity(200.0, "fs"),
        n_subpulses=7,
    )
    reg = laser.active_region(threshold=1e-4)
    delays = laser.subpulse_delays()

    for tk in delays:
        # At t=0, sub-pulse center along -z is at z = -c * tk (since u = -z = c*tk)
        z_center = -C_CGS * tk
        assert reg.contains(0.0, 0.0, z_center, 0.0)


def test_subpulse_decomposition_matches_train_intensity():
    """train.intensity_profile equals the sum of its sub-pulse intensity profiles."""
    laser = PulseTrainParaxialLaser(
        pulse_energy=Quantity(0.8, "J"),
        wavelength=Quantity(800.0, "nm"),
        sigma_x=Quantity(25.0, "um"),
        sigma_y=Quantity(30.0, "um"),
        subpulse_duration=Quantity(30.0, "fs"),
        repetition_period=Quantity(150.0, "fs"),
        n_subpulses=4,
        x_off=Quantity(2.0, "um"),
        y_off=Quantity(-3.0, "um"),
        t_off=Quantity(5.0, "fs"),
    )
    subpulses = laser.subpulses()
    assert len(subpulses) == 4

    x = np.linspace(-30e-4, 30e-4, 5)
    y = np.linspace(-30e-4, 30e-4, 5)
    z = np.linspace(-150e-4, 150e-4, 7)
    t = 20e-15
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    train_int = laser.intensity_profile(X, Y, Z, t)
    sub_int_sum = sum(sub.intensity_profile(X, Y, Z, t) for sub in subpulses)
    np.testing.assert_allclose(train_int, sub_int_sum, rtol=1e-13, atol=1e-25)

    train_dens = laser.photon_density(X, Y, Z, t)
    sub_dens_sum = sum(sub.photon_density(X, Y, Z, t) for sub in subpulses) / laser.n_subpulses
    np.testing.assert_allclose(train_dens, sub_dens_sum, rtol=1e-13, atol=1e-25)


def test_duty_cycle_and_validation():
    laser = PulseTrainParaxialLaser(
        pulse_energy=Quantity(1.0, "J"),
        wavelength=Quantity(800.0, "nm"),
        sigma_x=Quantity(20.0, "um"),
        sigma_y=Quantity(20.0, "um"),
        subpulse_duration=Quantity(20.0, "fs"),
        repetition_period=Quantity(100.0, "fs"),
        n_subpulses=5,
    )
    assert math.isclose(laser.duty_cycle(), 0.2, rel_tol=1e-12)
    assert validate(laser) == []

    # Invalid n_subpulses
    with pytest.raises(ValueError, match="n_subpulses"):
        PulseTrainParaxialLaser(
            pulse_energy=Quantity(1.0, "J"),
            wavelength=Quantity(800.0, "nm"),
            sigma_x=Quantity(20.0, "um"),
            sigma_y=Quantity(20.0, "um"),
            subpulse_duration=Quantity(20.0, "fs"),
            repetition_period=Quantity(100.0, "fs"),
            n_subpulses=0,
        )

    # Invalid negative energy
    with pytest.raises(ValueError, match="pulse_energy"):
        PulseTrainParaxialLaser(
            pulse_energy=Quantity(-1.0, "J"),
            wavelength=Quantity(800.0, "nm"),
            sigma_x=Quantity(20.0, "um"),
            sigma_y=Quantity(20.0, "um"),
            subpulse_duration=Quantity(20.0, "fs"),
            repetition_period=Quantity(100.0, "fs"),
            n_subpulses=2,
        )

    # fit_gaussian_paraxial raises for N_p > 1
    with pytest.raises(NotImplementedError, match="fit_gaussian_paraxial"):
        fit_gaussian_paraxial(laser)


def test_xigma_stage0_interaction_train_vs_subpulses():
    """End-to-end integration: train yield matches sum of sub-pulse yields."""
    beam = GaussianElectronBeam(
        bunch_charge=Quantity(100.0, "pC"),
        kinetic_energy=Quantity(100.0, "MeV"),
        rel_energy_spread=0.005,
        sigma_x=Quantity(10.0, "um"),
        sigma_y=Quantity(10.0, "um"),
        emit_x=Quantity(1.0, "mm*mrad"),
        emit_y=Quantity(1.0, "mm*mrad"),
        sigma_z=Quantity(30.0, "um"),
    )
    bunch = sample_gaussian_bunch(beam, n_particles=80, seed=42)

    # Compact burst where sub-pulses are close enough to be well within Rayleigh range
    train = PulseTrainParaxialLaser(
        pulse_energy=Quantity(0.5, "J"),
        wavelength=Quantity(800.0, "nm"),
        sigma_x=Quantity(20.0, "um"),
        sigma_y=Quantity(20.0, "um"),
        subpulse_duration=Quantity(25.0, "fs"),
        repetition_period=Quantity(50.0, "fs"),
        n_subpulses=3,
    )

    # Method 1: Train simulation (using fine n_steps to resolve the burst)
    samples_train = integrate_trajectories(
        bunch, train, n_electrons=beam.n_electrons(), n_steps=256
    )
    yield_train = samples_train.total_yield()
    assert yield_train > 0.0

    # Method 2: Sum of individual sub-pulses (each evaluated in its own active region)
    subpulses = train.subpulses()
    sub_yields = [
        integrate_trajectories(
            bunch, sub, n_electrons=beam.n_electrons(), n_steps=64
        ).total_yield()
        for sub in subpulses
    ]
    yield_sub_sum = sum(sub_yields)

    # Agreement within 2% with 256 steps over the compact burst
    rel_diff = abs(yield_train - yield_sub_sum) / yield_sub_sum
    assert rel_diff < 0.02, f"Yield train ({yield_train:.4e}) vs sub sum ({yield_sub_sum:.4e}), diff = {rel_diff:.2%}"


def test_yield_decays_with_decreasing_duty_cycle():
    """As duty cycle D drops, inter-pulse spacing pushes outer collisions outside Rayleigh range."""
    beam = GaussianElectronBeam(
        bunch_charge=Quantity(100.0, "pC"),
        kinetic_energy=Quantity(100.0, "MeV"),
        rel_energy_spread=0.005,
        sigma_x=Quantity(10.0, "um"),
        sigma_y=Quantity(10.0, "um"),
        emit_x=Quantity(1.0, "mm*mrad"),
        emit_y=Quantity(1.0, "mm*mrad"),
        sigma_z=Quantity(20.0, "um"),
    )
    bunch = sample_gaussian_bunch(beam, n_particles=50, seed=123)

    yields = []
    for d_val in [0.5, 0.05, 0.005]:
        t_rep_fs = 20.0 / d_val
        train = PulseTrainParaxialLaser(
            pulse_energy=Quantity(0.5, "J"),
            wavelength=Quantity(800.0, "nm"),
            sigma_x=Quantity(15.0, "um"),
            sigma_y=Quantity(15.0, "um"),
            subpulse_duration=Quantity(20.0, "fs"),
            repetition_period=Quantity(t_rep_fs, "fs"),
            n_subpulses=5,
        )
        # Using subpulses sum for maximum trajectory accuracy across widely spaced collisions
        y_sum = sum(
            integrate_trajectories(bunch, sub, n_electrons=beam.n_electrons(), n_steps=64).total_yield()
            for sub in train.subpulses()
        )
        yields.append(y_sum)

    # Monotonic decay: yield(compact) > yield(intermediate) > yield(sparse)
    assert yields[0] > yields[1] > yields[2]
    assert yields[2] < 0.8 * yields[0], f"Sparse burst yield {yields[2]} should drop relative to compact {yields[0]}"
