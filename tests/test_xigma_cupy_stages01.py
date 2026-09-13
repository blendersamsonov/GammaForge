"""Observe real device execution while preserving the host stage contracts."""
from dataclasses import replace

import numpy as np
import pytest

from gammaforge.engines.xigma import stages, spectrum_sampler
from gammaforge.engines.xigma.collision import Collision
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io.bunch import overlap_time_window
from gammaforge.io.laser import PulseTrainParaxialLaser
from gammaforge.io.units import Quantity as Q
from gammaforge.io.target import OutputKind, OutputRequest
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
    for a,b in [(actual.luminosity,cpu.luminosity),(actual.a0_shape,cpu.a0_shape),
                (actual.diagnostics.time_envelope,cpu.diagnostics.time_envelope),
                (actual.diagnostics.spatial_envelope,cpu.diagnostics.spatial_envelope)]:
        assert isinstance(a,np.ndarray)
        np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-14*np.max(abs(b)))
    assert isinstance(inputs.bunch.x,np.ndarray)


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


def test_auto_falls_back_and_explicit_cuda_does_not(monkeypatch):
    monkeypatch.setattr(spectrum_sampler,'is_gpu_available',lambda:False)
    inputs = interaction()
    samples = stages.integrate_trajectories(inputs.bunch,inputs.laser,inputs.N_e,n_steps=8,backend='auto')
    assert isinstance(stages.deposit_shape_table(samples,n_bins=(4,4,4,4),backend='auto').H,np.ndarray)
    inputs = replace(inputs, target=replace(inputs.target, outputs=(
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, (3,2,2)),)))
    result = XigmaEngine().run(inputs, XigmaEngine.schema.with_values(backend='auto', n_steps=8,
        n_bins_gamma=4, n_bins_theta_x=4, n_bins_theta_y=4, n_bins_a0_shape=4))
    assert all(result.model_specific[f'stage{stage}_backend']=='numpy' for stage in (0,1,2))
    with pytest.raises(RuntimeError,match='CUDA device'):
        stages.integrate_trajectories(inputs.bunch,inputs.laser,inputs.N_e,backend='cupy')
    with pytest.raises(RuntimeError,match='CUDA device'):
        stages.deposit_shape_table(samples,backend='cupy')


@gpu
def test_collision_routes_both_stages_and_keeps_cache_host_readonly(monkeypatch):
    cp = spectrum_sampler.cp
    inputs = interaction()
    original = stages._deposit_nearest
    seen=[]

    def observed(coords,weight,bins):
        seen.append(isinstance(weight,cp.ndarray))
        return original(coords,weight,bins)

    monkeypatch.setattr(stages,'_deposit_nearest',observed)
    collision = Collision(inputs,XigmaEngine.schema.with_values(backend='cupy',n_steps=8,
                          n_bins_gamma=4,n_bins_theta_x=4,n_bins_theta_y=4,n_bins_a0_shape=4))
    shape = collision._shape()
    assert seen==[True]
    assert collision._overlap_backend=='cupy'
    assert collision._shape() is shape
    assert not collision.build_overlap().luminosity.flags.writeable


@gpu
def test_public_engine_reports_all_device_stages():
    inputs = interaction()
    requests = (OutputRequest(OutputKind.COLLIMATED_SPECTRUM,(3,2,2)),)
    inputs = replace(inputs,target=replace(inputs.target,outputs=requests))
    result = XigmaEngine().run(inputs,XigmaEngine.schema.with_values(backend='cupy',n_steps=8,
        n_bins_gamma=4,n_bins_theta_x=4,n_bins_theta_y=4,n_bins_a0_shape=4))
    for stage in (0,1,2):
        assert result.model_specific[f'stage{stage}_backend']=='cupy'
    assert isinstance(result.photon_slices[OutputKind.COLLIMATED_SPECTRUM].distr,np.ndarray)


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
