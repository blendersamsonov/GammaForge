"""Actual-CUDA checks against independent long-double emission, not xigma helpers."""
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma import stages
from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
from gammaforge.validation import scenarios
from gammaforge.validation.references import delta, delta_cupy, delta_emission

pytestmark = [pytest.mark.tier1, pytest.mark.fast, pytest.mark.gpu]
gpu = pytest.mark.skipif(not is_gpu_available(), reason="CUDA unavailable")


def samples(gamma=2000., tx=0., ty=0., ahat=0., lum=1.):
    arrays = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, dtype=float))
                                   for x in (gamma, tx, ty, ahat, lum)])
    zeros = np.zeros_like(arrays[0])
    return stages.TrajectorySamples(
        *arrays, 1., 1, np.ones_like(arrays[0]), zeros, zeros, zeros
    )


@gpu
@pytest.mark.parametrize("ellipticity", [0., -.4, 1.])
def test_collinear_polarization_is_one(ellipticity):
    g = np.array([2., 2000., 10000.])
    e, w = delta_cupy.emission_lines(samples(gamma=g), 0., 0., photon_energy=2.,
                                    ellipticity=ellipticity, chunk=2)
    np.testing.assert_allclose(e, 8*g*g, rtol=1e-14)
    np.testing.assert_allclose(w, 3/(2*np.pi)*g*g, rtol=1e-14)
    assert isinstance(e, np.ndarray) and isinstance(w, np.ndarray)


@gpu
@pytest.mark.parametrize("mode", ["nominal", "particle", "direction"])
@pytest.mark.parametrize("gamma", [2., 2000., 10000.])
@pytest.mark.parametrize("geometry", [(0., 0., 0., 0.), (.3, -.2, .37, -.4), (-.03, .01, .71, 1.)])
def test_lines_match_independent_vectors(mode, gamma, geometry):
    rng = np.random.default_rng(20260914)
    s = samples(gamma=gamma*(1+rng.uniform(0, .1, 43)), tx=rng.normal(0, .003, 43),
                ty=rng.normal(0, .002, 43), ahat=rng.uniform(0, .3, 43),
                lum=rng.uniform(0, 2, 43))
    s = replace(s, chirp_mean=np.linspace(0.7, 1.3, s.n_particles))
    xz, yz, psi, eps = geometry
    kwargs = dict(photon_energy=2e-12, theta_xz=xz, theta_yz=yz,
                  psi_pol=psi, ellipticity=eps, doppler=mode)
    expected = delta_emission.emission_lines(s, .001, -.0006, **kwargs)
    actual = delta_cupy.emission_lines(s, .001, -.0006, chunk=7, **kwargs)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-13)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-8, atol=1e-12*float(expected[1].max()))
    assert np.all(actual[1] >= 0)


@gpu
def test_dark_particle_with_zero_carrier_has_zero_weight():
    s = replace(samples(gamma=[2000., 2100.], lum=[1., 0.]),
                chirp_mean=np.array([1.25, 0.]))
    expected = delta_emission.emission_lines(s, 0., 0., photon_energy=1.)
    actual = delta_cupy.emission_lines(s, 0., 0., photon_energy=1.)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-13)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-13)
    assert actual[1][1] == 0.0


@gpu
@pytest.mark.parametrize("scenario", scenarios.SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize("crossed", [False, True])
def test_scenario_lines_bins_and_chunk_invariance(scenario, crossed):
    from gammaforge.io.units import Quantity

    laser = replace(scenario.laser, theta_xz=Quantity(.03 if crossed else 0., "rad"),
                    theta_yz=Quantity(-.02 if crossed else 0., "rad"), ellipticity=.4)
    interaction = scenarios.build(replace(scenario, laser=laser),
                                  replace(scenario.sampling, n_particles=257))
    s = stages.integrate_trajectories(interaction.bunch, laser, interaction.N_e, n_steps=32)
    kwargs = dict(photon_energy=float(laser.photon_energy()), theta_xz=laser.m("theta_xz"),
                  theta_yz=laser.m("theta_yz"), ellipticity=.4, doppler="direction")
    e, w = delta_emission.emission_lines(s, .0002, -.0001, **kwargs)
    edges = np.geomspace(float(e.min())*.9, float(e.max())*1.1, 33)
    expected = delta_emission.bin_emission(e, w, edges)
    for chunk in (17, 10000):
        de, dw = delta_cupy.emission_lines(s, .0002, -.0001, chunk=chunk, **kwargs)
        actual = delta_cupy.bin_emission(de, dw, edges, chunk=chunk)
        np.testing.assert_allclose(actual["bin_mass"], expected["bin_mass"], rtol=3e-8,
                                   atol=1e-12*float(w.sum()))
        assert actual["total_weight"] == pytest.approx(float(w.sum()), rel=3e-8)
        scale = 2*kwargs["photon_energy"]*(1+np.cos(kwargs["theta_xz"])*np.cos(kwargs["theta_yz"]))
        normalized = delta_cupy.resonance_spectrum(s, edges/scale, .0002, -.0001,
            theta_xz=kwargs["theta_xz"], theta_yz=kwargs["theta_yz"], ellipticity=.4,
            doppler="direction", chunk=chunk)
        np.testing.assert_allclose(normalized*np.diff(edges/scale), actual["bin_mass"],
                                   rtol=1e-12, atol=1e-12*float(w.sum()))


@gpu
def test_histogram_edges_tails_nonuniform_bins_zero_weights_and_empty():
    for e, w in [([0., .5, 1., 3., 6., 7.], [1., 2., 3., 0., 5., 6.]), ([], [])]:
        edges = [1., 3., 6.]
        actual = delta_cupy.bin_emission(e, w, edges, chunk=2)
        expected = delta_emission.bin_emission(e, w, edges)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-14)
        assert actual["bin_mass"].sum()+actual["underflow"]+actual["overflow"] == actual["total_weight"]
    empty = samples(gamma=[], tx=[], ty=[], ahat=[], lum=[])
    e, w = delta_cupy.emission_lines(empty, 0., 0., photon_energy=1.)
    assert e.size == w.size == 0
    np.testing.assert_array_equal(delta_cupy.angle_integrated_spectrum(empty, [0., 1.]), [0.])


@gpu
def test_linear_anchor_scalar_vector_and_normalization():
    s = samples(gamma=[20., 30.], lum=[1., 2.])
    for grid in [100., np.array([100., 300., 700.]), np.array([])]:
        actual = delta_cupy.single_electron_spectrum(s, grid, chunk=3)
        expected = delta.single_electron_spectrum(s, grid)
        assert np.shape(actual) == np.shape(grid)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    for g in (20., 2000.):
        grid = g*g*(nodes+1)/2
        spectrum = delta_cupy.single_electron_spectrum(samples(gamma=g, lum=3.), grid)
        assert np.dot(spectrum, weights)*g*g/2 == pytest.approx(3., rel=1e-13)


@gpu
def test_angular_integral_matches_same_cpu_midpoint_rule():
    s = samples(gamma=[1800., 2200.], tx=[0., .0001], ty=[0., -.0001], lum=[1., .7])
    edges = np.linspace(0., 5e6, 33)
    kw = dict(n_angles=9, cone_factor=3., psi_pol=.37, ellipticity=.4, theta_xz=.02, theta_yz=-.015)
    expected = delta.angle_integrated_spectrum(s, edges, **kw)
    for chunk in (1, 7):
        actual = delta_cupy.angle_integrated_spectrum(s, edges, chunk=chunk, **kw)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-18)


@gpu
@pytest.mark.parametrize("eps", [0., -.4, 1.])
def test_angular_photon_conservation(eps):
    g = 2000.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    total = 0.
    for t, w in zip((nodes+1)/2, weights/2):
        r = np.sqrt(t/(1-t))/g
        for phi in 2*np.pi*np.arange(12)/12:
            _, mass = delta_cupy.emission_lines(samples(gamma=g, lum=2.), r*np.cos(phi), r*np.sin(phi),
                photon_energy=1., theta_xz=.02, theta_yz=-.015, psi_pol=.37, ellipticity=eps,
                doppler="direction")
            total += mass[0]*w*np.pi/(12*g*g*(1-t)**2)
    assert total == pytest.approx(2., rel=2e-5)


@gpu
def test_reference_does_not_call_production_or_cpu_emission_helpers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("production/CPU emission helper called")

    for name in ("polarization_factor", "polarization_factor_vectorized", "rotated_laser_axes",
                 "physical_transverse_axes", "direction_doppler_factor"):
        monkeypatch.setattr(stages, name, forbidden)
    monkeypatch.setattr(delta_emission, "emission_lines", forbidden)
    monkeypatch.setattr(delta_emission, "bin_emission", forbidden)
    result = delta_cupy.resonance_spectrum(samples(), [0., 5e6], 0., 0.)
    assert result[0] > 0
