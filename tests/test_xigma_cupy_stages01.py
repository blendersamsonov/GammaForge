"""Observe real device execution while preserving the host stage contracts."""
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma import stages, spectrum_sampler
from gammaforge.io.bunch import overlap_time_window
from gammaforge.io.laser import PulseTrainParaxialLaser
from gammaforge.io.units import Quantity as Q
from gammaforge.validation import scenarios

pytestmark = [pytest.mark.tier1, pytest.mark.fast, pytest.mark.gpu]
gpu = pytest.mark.skipif(not spectrum_sampler.is_gpu_available(), reason='CUDA unavailable')


def interaction():
    return scenarios.build(replace(scenarios.BASELINE, sampling=replace(scenarios.BASELINE.sampling, n_particles=96)))


@gpu
@pytest.mark.parametrize('pulse_train', [False, True])
@pytest.mark.parametrize('chunk', [7, 96])
def test_stage0_runs_on_device_and_matches_source_diagnostics(pulse_train, chunk):
    cp = spectrum_sampler.cp
    inputs = interaction()
    laser = inputs.laser
    if pulse_train:
        laser = PulseTrainParaxialLaser(pulse_energy=laser.pulse_energy, wavelength=laser.wavelength,
            sigma_x=laser.sigma_x, sigma_y=laser.sigma_y, n_subpulses=3,
            subpulse_duration=Q(1,'ps'), repetition_period=Q(3,'ps'), theta_xz=Q(.02,'rad'))
    t0,t1 = overlap_time_window(inputs.bunch, laser)
    edges = t0.min()+(t1.max()-t0.min())*np.linspace(0,1,17)**2
    spatial = (np.array([-1.,-.001,0.,.001,1.]), np.array([-1.,0.,.002,1.]))
    seen=[]

    class ObservedLaser:
        def __getattr__(self, name):
            return getattr(laser, name)

        def intensity_profile(self, x,y,z,t):
            assert all(isinstance(a, cp.ndarray) for a in (x,y,z,t))
            seen.append(x.shape[0])
            return laser.intensity_profile(x,y,z,t)

    kwargs = dict(n_steps=32,t_edges=edges,spatial_edges=spatial)
    cpu = stages.integrate_trajectories(inputs.bunch,laser,inputs.N_e,**kwargs)
    actual = stages.integrate_trajectories(inputs.bunch,ObservedLaser(),inputs.N_e,
                                         backend='cupy',chunk=chunk,**kwargs)
    assert seen and max(seen)<=chunk
    for a,b in [
        (actual.luminosity, cpu.luminosity),
        (actual.a0_shape, cpu.a0_shape),
        (actual.chirp_mean, cpu.chirp_mean),
        (actual.var_a_shape, cpu.var_a_shape),
        (actual.var_chirp, cpu.var_chirp),
        (actual.cov_a_chirp_shape, cpu.cov_a_chirp_shape),
        (actual.diagnostics.time_envelope, cpu.diagnostics.time_envelope),
        (actual.diagnostics.spatial_envelope, cpu.diagnostics.spatial_envelope),
    ]:
        assert isinstance(a,np.ndarray)
        np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-14*np.max(abs(b)))
    assert isinstance(inputs.bunch.x,np.ndarray)


@gpu
def test_carrier_phase_gradient_broadcasts_on_device():
    cp = spectrum_sampler.cp
    laser = interaction().laser
    gradient = laser.carrier_phase_four_gradient(
        cp.zeros((2, 1)), cp.zeros((1, 3)), 0.0, cp.arange(3)[None, :]
    )
    for component in gradient:
        assert isinstance(component, cp.ndarray)
        cp.testing.assert_array_equal(component, cp.zeros((2, 3)))


@gpu
@pytest.mark.parametrize('scheme',['nearest','cic'])
def test_stage1_device_deposition_and_retry_preserve_mass(scheme, monkeypatch):
    cp = spectrum_sampler.cp
    inputs = interaction()
    samples = stages.integrate_trajectories(inputs.bunch,inputs.laser,inputs.N_e,n_steps=16)
    kwargs = dict(n_bins=(8,7,6,5),scheme=scheme)
    expected = stages.deposit_shape_table(samples,**kwargs)
    name = '_deposit_nearest' if scheme=='nearest' else '_deposit_cic'
    original = getattr(stages,name)
    calls=[]

    def observed(coords,weight,bins):
        assert all(isinstance(a,cp.ndarray) for a in (*coords,weight))
        calls.append(weight.size)
        result = original(coords,weight,bins)
        if len(calls)==1:
            raise MemoryError('force retry after device deposition')
        return result

    monkeypatch.setattr(stages,name,observed)
    actual = stages.deposit_shape_table(samples,backend='cupy',chunk=96,**kwargs)
    assert calls==[96,48,48]
    assert isinstance(actual.H,np.ndarray)
    np.testing.assert_allclose(actual.H,expected.H,rtol=3e-12,atol=1e-14*expected.H.max())
    assert actual.total_weight==pytest.approx(samples.total_yield(),rel=1e-13)


@gpu
def test_device_stage0_retry_preserves_temporal_mass(monkeypatch):
    cp = spectrum_sampler.cp
    inputs = interaction()
    kwargs = dict(n_steps=16,t_edges=np.linspace(-1e-9,1e-9,17))
    expected = stages.integrate_trajectories(inputs.bunch,inputs.laser,inputs.N_e,**kwargs)
    original = cp.bincount
    calls = []

    def fail_once(*args,**kwargs):
        calls.append(args[0].size)
        if len(calls)==1:
            raise MemoryError('forced device histogram allocation failure')
        return original(*args,**kwargs)

    monkeypatch.setattr(cp,'bincount',fail_once)
    actual = stages.integrate_trajectories(inputs.bunch,inputs.laser,inputs.N_e,
                                         backend='cupy',chunk=96,**kwargs)
    assert len(calls)==3
    np.testing.assert_allclose(actual.diagnostics.time_envelope,expected.diagnostics.time_envelope,
                               rtol=2e-11,atol=1e-14*expected.diagnostics.time_envelope.max())
