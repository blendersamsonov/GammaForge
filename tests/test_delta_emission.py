import numpy as np
import pytest

from gammaforge.engines.xigma.stages import TrajectorySamples
from gammaforge.validation.references.delta_emission import bin_emission, emission_lines


def samples(gamma=2000.0, tx=0.0, ty=0.0, lum=1.0, ahat=0.0, chirp=1.0):
    gamma = np.atleast_1d(gamma)
    zeros = np.zeros_like(gamma)
    return TrajectorySamples(
        gamma, np.atleast_1d(tx), np.atleast_1d(ty), np.atleast_1d(ahat),
        np.atleast_1d(lum), 1.0, 1, np.broadcast_to(chirp, gamma.shape), zeros, zeros, zeros,
    )


def test_headon_resonance_and_signed_ellipticity():
    a = emission_lines(samples(), 0.0, 0.0, photon_energy=1.0, ellipticity=.4)
    b = emission_lines(samples(), 0.0, 0.0, photon_energy=1.0, ellipticity=-.4)
    np.testing.assert_allclose(a[0], [4 * 2000.0**2])
    np.testing.assert_allclose(a[1], b[1])


def test_carrier_rate_scales_the_resonance_energy():
    unchirped, _ = emission_lines(samples(), 0.0, 0.0, photon_energy=1.0)
    chirped, _ = emission_lines(samples(chirp=1.08), 0.0, 0.0, photon_energy=1.0)
    np.testing.assert_allclose(chirped / unchirped, [1.08])


@pytest.mark.parametrize("gamma", [2.0, 2000.0, 10000.0])
def test_particle_doppler_factor(gamma):
    nominal, _ = emission_lines(samples(gamma=gamma), .1, .0, photon_energy=2., doppler="nominal")
    particle, _ = emission_lines(samples(gamma=gamma), .1, .0, photon_energy=2., doppler="particle")
    expected = (1 + np.sqrt(np.longdouble(1) - np.longdouble(gamma) ** -2)) / 2
    np.testing.assert_allclose(particle[0] / nominal[0], expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("gamma", [2000.0, 10000.0])
@pytest.mark.parametrize("theta_xz,theta_yz", [(0.3, 0.0), (0.0, 0.3), (0.3, 0.2)])
def test_crossed_collinear_weight_is_unity(gamma, theta_xz, theta_yz):
    _, weights = emission_lines(samples(gamma=gamma), 0., 0., photon_energy=1., theta_xz=theta_xz, theta_yz=theta_yz)
    ratio = weights[0] / ((3 / (2 * np.pi)) * gamma**2)
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-9)


@pytest.mark.parametrize('gamma', [2000., 10000.])
@pytest.mark.parametrize('electron_angles', [(0., 0.), (.001, -.0006)])
@pytest.mark.parametrize('ellipticity', [-.4, 1.])
def test_independent_lines_conserve_angular_photon_weight(gamma, electron_angles, ellipticity):
    sample = samples(gamma=gamma, tx=electron_angles[0], ty=electron_angles[1], lum=2.)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    integral = 0.
    # Reduced small-angle measure; no production polarization or expected-curve helper.
    for t, weight in zip((nodes + 1) / 2, weights / 2):
        radius = np.sqrt(t / (1 - t)) / gamma
        for phi in 2 * np.pi * np.arange(12) / 12:
            _, lines = emission_lines(
                sample, electron_angles[0] + radius * np.cos(phi),
                electron_angles[1] + radius * np.sin(phi), photon_energy=1.,
                theta_xz=.02, theta_yz=-.015, psi_pol=.37, ellipticity=ellipticity,
            )
            integral += float(lines[0]) * weight * np.pi / (12 * gamma**2 * (1 - t)**2)
    assert integral == pytest.approx(2., rel=2e-5)


@pytest.mark.parametrize("tx", [-.01, .01])
def test_tilted_particle_doppler_uses_explicit_laser_axis(tx):
    gamma = np.longdouble(2000.)
    nominal, _ = emission_lines(samples(gamma=float(gamma), tx=tx), 0., 0., photon_energy=2.,
                                 theta_xz=.3, theta_yz=.2, doppler="nominal")
    particle, _ = emission_lines(samples(gamma=float(gamma), tx=tx), 0., 0., photon_energy=2.,
                                 theta_xz=.3, theta_yz=.2, doppler="particle")
    beta = np.sqrt(np.longdouble(1) - gamma**-2)
    v = beta * np.array([tx, 0., 1.], dtype=np.longdouble) / np.sqrt(1 + np.longdouble(tx)**2)
    n0 = np.array([-np.sin(.3)*np.cos(.2), np.sin(.2), -np.cos(.3)*np.cos(.2)], dtype=np.longdouble)
    expected = (1 - np.dot(v, n0)) / (1 + np.cos(.3)*np.cos(.2))
    np.testing.assert_allclose(particle[0] / nominal[0], expected, rtol=1e-12, atol=1e-15)


def test_nonlinear_redshift_uses_exact_observer_incidence_ratio():
    gamma, tx, ty, ahat = 2000., .12, -.08, .7
    theta_xz, theta_yz = .6, -.25
    observer_x, observer_y = .121, -.079
    energies, _ = emission_lines(
        samples(gamma=gamma, tx=tx, ty=ty, ahat=ahat),
        observer_x,
        observer_y,
        photon_energy=2.,
        theta_xz=theta_xz,
        theta_yz=theta_yz,
        doppler="direction",
    )
    electron = np.array([tx, ty, 1.], dtype=np.longdouble)
    electron /= np.linalg.norm(electron)
    n0 = np.array(
        [-np.sin(theta_xz)*np.cos(theta_yz), np.sin(theta_yz),
         -np.cos(theta_xz)*np.cos(theta_yz)],
        dtype=np.longdouble,
    )
    encounter = 1 - electron @ n0
    observer = np.array([observer_x, observer_y, 1.], dtype=np.longdouble)
    observer /= np.linalg.norm(observer)
    q_incidence = (1 - observer @ n0) / encounter
    r2 = (np.longdouble(tx)-observer_x)**2 + (np.longdouble(ty)-observer_y)**2
    expected = 4*np.longdouble(1.)*encounter*gamma**2 / (
        1 + q_incidence*ahat + gamma**2*r2
    )
    np.testing.assert_allclose(energies[0], expected, rtol=1e-14)

    nominal, _ = emission_lines(
        samples(gamma=gamma, tx=tx, ty=ty, ahat=ahat),
        observer_x,
        observer_y,
        photon_energy=2.,
        theta_xz=theta_xz,
        theta_yz=theta_yz,
        doppler="nominal",
    )
    nominal_encounter = 1 + np.cos(theta_xz)*np.cos(theta_yz)
    expected_nominal = 4*np.longdouble(1.)*nominal_encounter*gamma**2 / (
        1 + q_incidence*ahat + gamma**2*r2
    )
    np.testing.assert_allclose(nominal[0], expected_nominal, rtol=1e-14)


def test_binning_edges_tails_and_mass():
    result = bin_emission([0., 1., 2., 3.], [1., 2., 3., 4.], [1., 2., 3.])
    np.testing.assert_allclose(result["bin_mass"], [2., 7.])
    assert result["underflow"] == 1 and result["overflow"] == 0
    assert result["total_weight"] == 10


def test_nonuniform_bins_bookkeeping_and_empty():
    result = bin_emission([0., .5, 1., 3., 6., 7.], [1., 2., 3., 4., 5., 6.], [0., 1., 3., 6.])
    np.testing.assert_allclose(result["bin_mass"], [3., 3., 9.])
    np.testing.assert_allclose(result["density"], [3., 1.5, 3.])
    assert result["overflow"] == 6 and result["total_weight"] == 21
    empty = bin_emission([], [], [0., 1.])
    assert empty["total_weight"] == 0 and empty["underflow"] == empty["overflow"] == 0


def test_zero_luminosity_is_retained():
    energies, weights = emission_lines(samples(lum=0.), 0., 0., photon_energy=1.)
    assert energies.size == 1 and weights[0] == 0


def test_longdouble_precision_guard(monkeypatch):
    real_finfo = np.finfo
    monkeypatch.setattr(np, "finfo", lambda dtype: real_finfo(np.float64) if dtype is np.longdouble else real_finfo(dtype))
    with pytest.raises(RuntimeError, match="extended longdouble"):
        emission_lines(samples(), 0., 0., photon_energy=1.)


def test_reference_does_not_use_production_polarization(monkeypatch):
    import gammaforge.engines.xigma.stages as stages
    monkeypatch.setattr(stages, "polarization_factor", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(stages, "rotated_laser_axes", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(stages, "polarization_factor_vectorized", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(stages, "physical_transverse_axes", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert emission_lines(samples(), 0., 0., photon_energy=1.)[0].size == 1


def test_headon_offaxis_transverse_weight_matches_closed_form():
    """For a cold head-on electron, evaluate the transverse dipole factor directly."""
    gamma = np.longdouble(2000.0)
    tx, ty = np.longdouble("2e-4"), np.longdouble("-1e-4")
    n = np.array([tx, ty, 1], dtype=np.longdouble)
    n /= np.linalg.norm(n)
    beta = np.sqrt(1 - gamma**-2)
    v = np.array([0, 0, beta], dtype=np.longdouble)
    d = 1 - np.dot(v, n)
    e0 = np.array([1, 0, 0], dtype=np.longdouble)
    e1 = np.array([0, 1, 0], dtype=np.longdouble)
    u0 = np.cross(n, np.cross(n - v, e0)) / d
    u1 = np.cross(n, np.cross(n - v, e1)) / d
    expected_pol = np.dot(u0, u0)
    _, weights = emission_lines(samples(gamma=float(gamma)), float(tx), float(ty), photon_energy=1.0)
    r2 = tx * tx + ty * ty
    expected_weight = (3 / (2 * np.pi)) * gamma**2 * expected_pol / (1 + gamma**2 * r2)**2
    np.testing.assert_allclose(weights[0], float(expected_weight), rtol=1e-12)


def test_nominal_headon_matches_legacy_delta_histogram():
    from gammaforge.validation.references.delta import resonance_spectrum

    sample_set = TrajectorySamples(
        np.array([1200.0, 2000.0, 2800.0]), np.array([0.0, 0.001, -0.0007]),
        np.array([0.0, -0.0004, 0.0006]), np.zeros(3), np.array([1.0, 0.7, 1.3]), 1.0, 1,
        np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3),
    )
    laser = 2.5e-7
    s_edges = np.array([0.0, 0.4e6, 1.0e6, 2.0e6, 4.0e6, 9.0e6])
    energy_edges = 4.0 * laser * s_edges
    legacy_density = resonance_spectrum(sample_set, s_edges, 0.0, 0.0)
    legacy_mass = legacy_density * np.diff(s_edges)
    energies, weights = emission_lines(sample_set, 0.0, 0.0, photon_energy=laser)
    current = bin_emission(energies, weights, energy_edges)
    np.testing.assert_allclose(current["bin_mass"], legacy_mass, rtol=1e-8, atol=1e-20)
