"""Direction-Doppler regression checks independent of the production root/Jacobian."""
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma import stages
from gammaforge.engines.xigma.stages import Table, TrajectorySamples
from gammaforge.io.bunch import overlap_time_window
from gammaforge.io.units import C_CGS, SIGMA_T_CGS
from gammaforge.validation import scenarios
from gammaforge.validation.references.delta_emission import emission_lines

pytestmark = [pytest.mark.tier1, pytest.mark.fast]


def laser_axis(txz, tyz):
    return np.array([-np.sin(txz)*np.cos(tyz), np.sin(tyz), -np.cos(txz)*np.cos(tyz)])


def test_direction_factor_is_unit_speed_and_broadcasts():
    tx, ty = np.array([-.2, .0, .3])[:, None], np.array([-.1, .2])[None, :]
    x, y = np.broadcast_arrays(tx, ty)
    vectors = np.stack((x, y, np.ones_like(x)), axis=-1)
    vectors /= np.linalg.norm(vectors, axis=-1)[..., None]
    n0 = laser_axis(.7, -.3)
    expected = (1 - vectors @ n0) / (1 - n0[2])
    np.testing.assert_allclose(stages.direction_doppler_factor(tx, ty, .7, -.3), expected, rtol=1e-14)
    np.testing.assert_allclose(stages.direction_doppler_factor(tx, ty, k_hat=n0), expected, rtol=1e-14)
    assert stages.direction_doppler_factor(0., 0.) == 1


def test_stage0_flux_uses_each_direction_without_changing_ballistic_window():
    interaction = scenarios.build(replace(scenarios.BASELINE,
        sampling=replace(scenarios.BASELINE.sampling, n_particles=4)))
    bunch = replace(interaction.bunch, thx=np.array([-.15, 0., .1, .2]), thy=np.zeros(4))
    n0 = laser_axis(.6, -.2)
    base = interaction.laser

    class ConstantField:
        def __getattr__(self, name):
            return getattr(base, name)

        def intensity_profile(self, x, y, z, t):
            return np.ones(np.broadcast_shapes(x.shape, y.shape, z.shape, t.shape))

        def focusing_axes(self):
            return n0, None, None

    laser = ConstantField()
    t0, t1 = overlap_time_window(bunch, laser)
    v = np.stack((bunch.thx, bunch.thy, np.ones(4)), axis=-1)
    v /= np.linalg.norm(v, axis=-1)[:, None]
    expected = ((1 - v @ n0) * C_CGS * SIGMA_T_CGS * stages.photon_density_scale(laser)
                * interaction.N_e * bunch.weight * np.maximum(0, t1-t0))
    result = stages.integrate_trajectories(bunch, laser, interaction.N_e, n_steps=4, chunk=1)
    np.testing.assert_allclose(result.luminosity, expected, rtol=1e-13)


@pytest.mark.parametrize('crossing', [(0., 0.), (.8, -.3)])
def test_table_jacobian_preserves_independent_line_mass_and_centroid(crossing):
    tx, ty, ahat = .12, -.08, .02
    gamma_edges = np.linspace(600., 1400., 49)
    centers = (gamma_edges[:-1] + gamma_edges[1:])/2
    ahat_edges = np.array([0., 2*ahat])
    # Single bin [0, 0.04] should be evaluated at its center (0.02), not at 0
    ahat_eval_points = np.array([ahat])
    table = Table(gamma_edges, np.array([tx-1e-5, tx+1e-5]),
                  np.array([ty-1e-5, ty+1e-5]), ahat_edges,
                  (1 + (centers/1000)**2)[:, None, None, None], 1., 'doppler-check',
                  _ahat_eval_points=ahat_eval_points)
    nodes, quad = np.polynomial.legendre.leggauss(12)
    g = ((centers[:-1, None]+centers[1:, None])/2
         + np.diff(centers)[:, None]*nodes/2).ravel()
    wg = (np.diff(centers)[:, None]*quad/2).ravel()
    density = np.interp(g, centers, table.H[:, 0, 0, 0])
    area = 4e-10
    samples = TrajectorySamples(g, np.full_like(g, tx), np.full_like(g, ty),
                                np.full_like(g, ahat), density*wg*area*(2*ahat), 1., 1)
    n0 = laser_axis(*crossing)
    photon_scale = 2*(1-n0[2])  # E_laser=1 erg, same normalized s as production
    energies, weights = emission_lines(samples, tx, ty, photon_energy=1.,
        theta_xz=crossing[0], theta_yz=crossing[1], ellipticity=.4, doppler='direction')
    # Break energy quadrature at mapped table interpolation knots, independently from the kernel.
    unit_v = np.array([tx, ty, 1.]); unit_v /= np.linalg.norm(unit_v)
    D = (1-unit_v@n0)/(1-n0[2])
    knots = D*centers**2/(1+ahat)
    energy = ((knots[:-1, None]+knots[1:, None])/2 + np.diff(knots)[:, None]*nodes/2).ravel()
    ws = (np.diff(knots)[:, None]*quad/2).ravel()
    spectrum = stages.spectrum_from_table(table, tx, ty, energy,
        theta_xz=crossing[0], theta_yz=crossing[1], ellipticity=.4)
    assert np.sum(spectrum*ws) == pytest.approx(float(weights.sum()), rel=2e-9)
    centroid = float(np.sum(energies*weights)/weights.sum()/photon_scale)
    assert np.sum(spectrum*ws*energy)/np.sum(spectrum*ws) == pytest.approx(centroid, rel=2e-9)
    assert np.ndim(stages.spectrum_from_table(table, tx, ty, float(energy[0]))) == 0
    assert stages.spectrum_from_table(table, tx, ty, energy[:1]).shape == (1,)


def test_linear_spectrum_uses_direction_energy_scale_and_conserves_photons():
    sample = TrajectorySamples(np.array([1000.]), np.array([.2]), np.array([-.1]),
                                np.array([0.]), np.array([7.]), 1., 1)
    n0 = laser_axis(.8, .2)
    v = np.array([.2, -.1, 1.]); v /= np.linalg.norm(v)
    D = (1-v@n0)/(1-n0[2])
    nodes, weights = np.polynomial.legendre.leggauss(16)
    energy = (nodes+1)*D*1000**2/2
    values = stages.angle_integrated_spectrum(sample, energy, theta_xz=.8, theta_yz=.2)
    assert np.sum(values*weights)*D*1000**2/2 == pytest.approx(7., rel=1e-14)
    headon = replace(sample, theta_x=np.zeros(1), theta_y=np.zeros(1))
    np.testing.assert_allclose(values, stages.angle_integrated_spectrum(headon, energy/D)/D, rtol=1e-14)
    assert stages.angle_integrated_spectrum(sample, 1.001*D*1000**2, theta_xz=.8, theta_yz=.2) == 0


@pytest.mark.parametrize('crossing', [(0., 0.), (.8, -.3), (-.8, .3)])
def test_gpu_bounds_enclose_all_directions(crossing):
    from gammaforge.engines.xigma.spectrum_sampler import _doppler_bounds
    for xlo, xhi, ylo, yhi in [(-.2, .3, -.1, .2), (.1, .4, -.2, -.1), (-.4, -.1, .1, .2)]:
        n0 = laser_axis(*crossing)
        low, high = _doppler_bounds([xlo, xhi], [ylo, yhi], n0)
        x, y = np.meshgrid(np.linspace(xlo, xhi, 31), np.linspace(ylo, yhi, 31))
        v = np.stack((x, y, np.ones_like(x)), axis=-1)
        v /= np.linalg.norm(v, axis=-1)[..., None]
        exact = (1-v@n0)/(1-n0[2])
        assert low <= exact.min() <= exact.max() <= high


@pytest.mark.gpu
def test_cuda_keeps_support_above_the_old_nominal_edge():
    from gammaforge.engines.xigma import spectrum_sampler as sampler
    if not sampler.is_gpu_available():
        pytest.skip('requires actual CUDA')
    # An amplified direction stress case: old nominal radial bounds reject every sample.
    tx, ty, cross = .4, .0, 1.
    gamma_edges = np.linspace(990., 1010., 17)
    x = np.linspace(tx-.0004, tx+.0004, 97)
    y = np.linspace(-.0004, .0004, 97)
    table = Table(gamma_edges, x, y, np.array([0., .0001, .0002]),
                  np.ones((16, 96, 96, 2)), 1., 'doppler-support-check')
    n0 = laser_axis(cross, 0.)
    v = np.array([tx, ty, 1.]); v /= np.linalg.norm(v)
    D = (1-v@n0)/(1-n0[2])
    energy = np.array([D*1000**2*.995, D*1000**2])
    assert np.all(energy > gamma_edges[-1]**2)
    cpu = stages.angular_spectrum_from_table(table, [tx], [ty], energy, theta_xz=cross, backend='numpy')
    gpu = stages.angular_spectrum_from_table(table, [tx], [ty], energy, theta_xz=cross, backend='cupy', rings=64, subsampling=128)
    assert np.all(gpu > 0)
    np.testing.assert_allclose(gpu, cpu, rtol=.05)


@pytest.mark.symbolic
@pytest.mark.tier3
def test_direction_jacobian_symbolic():
    sympy = pytest.importorskip('sympy')
    D, A, s, r = sympy.symbols('D A s r', positive=True)
    gamma = sympy.sqrt(A/(D/s-r*r))
    assert sympy.simplify(sympy.diff(gamma, s) - D*gamma**3/(2*A*s*s)) == 0
