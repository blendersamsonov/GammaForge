"""Independent numerical checks of DER025's chromatic pulse-front correction."""

import numpy as np
import pytest

from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.units import C_CGS, Quantity


pytestmark = [pytest.mark.tier1, pytest.mark.fast]


def _correction(q, p2, gf, omega):
    return q / (omega * (1 + q * q)) * (
        gf + 0.5 * p2 * (1 - 2 * gf / (1 + q * q))
    )


def _chromatic_phase(omega, omega0, z_r, u, rho, gf):
    """Paraxial carrier with an explicitly chromatic Rayleigh range."""
    chromatic_z_r = z_r * (omega / omega0) ** gf
    return (
        omega * u / C_CGS
        - np.arctan(u / chromatic_z_r)
        + omega * rho * rho * u / (2 * C_CGS * (u * u + chromatic_z_r**2))
    )


@pytest.mark.parametrize("gf", [-1.0, 0.0, 1.0, 2.0])
def test_der025_delay_matches_chromatic_phase_derivative(gf):
    laser = GaussianParaxialLaser(
        pulse_energy=Quantity(1, "J"), wavelength=Quantity(800, "nm"),
        sigma_x=Quantity(20, "um"), sigma_y=Quantity(20, "um"),
        duration=Quantity(30, "fs"),
    )
    omega0, z_r, sigma = laser.omega0(), laser.rayleigh_x(), laser.m("sigma_x")
    step = omega0 * 1e-5
    for q in [-3.0, -0.5, 0.0, 0.7, 2.5]:
        for p2 in [0.0, 0.2, 2.0, 5.0]:
            u, rho = q * z_r, np.sqrt(2 * p2) * sigma
            numerical = (
                _chromatic_phase(omega0 + step, omega0, z_r, u, rho, gf)
                - _chromatic_phase(omega0 - step, omega0, z_r, u, rho, gf)
            ) / (2 * step) - u / C_CGS
            np.testing.assert_allclose(
                numerical, _correction(q, p2, gf, omega0), atol=3e-20
            )
            if gf == 0.0 and p2 == 0.0:
                # Frequency-independent Gouy phase cannot move the pulse front.
                assert abs(numerical) < 3e-20


def test_der023_group_delay_gate_bounds_gaussian_envelope_change():
    laser = GaussianParaxialLaser(
        pulse_energy=Quantity(1, "J"), wavelength=Quantity(800, "nm"),
        sigma_x=Quantity(20, "um"), sigma_y=Quantity(20, "um"),
        duration=Quantity(30, "fs"),
    )
    omega0 = laser.omega0()
    spatial_floor = 0.01
    q_max = np.sqrt(1 / spatial_floor - 1)
    q = np.linspace(-q_max, q_max, 201)
    p2_max = np.maximum(0, -(1 + q * q) * np.log(spatial_floor * (1 + q * q)))
    p_fraction = np.linspace(0, 1, 21)
    eta_fraction = np.linspace(-4, 4, 81)

    for gf in [-1.0, 0.0, 1.0, 2.0]:
        # DER023's laser-only bound, maximized independently on a dense q grid.
        q_bound = np.linspace(0, q_max, 20001)
        p_bound = np.maximum(
            0, -(1 + q_bound**2) * np.log(spatial_floor * (1 + q_bound**2))
        )
        delta_star = np.max(
            q_bound / (omega0 * (1 + q_bound**2))
            * (abs(gf) + p_bound / 2 * (1 + 2 * abs(gf) / (1 + q_bound**2)))
        )
        actual_delay = _correction(
            q[:, None], p2_max[:, None] * p_fraction[None, :], gf, omega0
        )
        assert np.max(np.abs(actual_delay)) <= delta_star * (1 + 1e-6)

        for duration_fs in [5, 10, 30, 100]:
            sigma_t = duration_fs * 1e-15
            # Gaussian peak-normalized envelope has global slope <= 1/(sqrt(e) sigma).
            gate = delta_star / (np.sqrt(np.e) * sigma_t)
            eta = eta_fraction[:, None, None] * sigma_t
            shifted = np.exp(-0.5 * ((eta - actual_delay[None, :, :]) / sigma_t)**2)
            baseline = np.exp(-0.5 * (eta / sigma_t)**2)
            measured = np.max(np.abs(shifted - baseline))
            assert measured <= gate * (1 + 1e-12)
