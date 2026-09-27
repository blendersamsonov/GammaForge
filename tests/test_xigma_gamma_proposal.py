"""Regression for resonance proposals retaining direction-corrected support."""
import numpy as np
import pytest

from gammaforge.engines.xigma import spectrum_sampler as sampler
from gammaforge.engines.xigma.stages import Table, angular_spectrum_from_table

pytestmark = [pytest.mark.tier1, pytest.mark.fast, pytest.mark.gpu]


@pytest.mark.skipif(not sampler.is_gpu_available(), reason='CUDA unavailable')
def test_proposal_preserves_emission_beyond_nominal_inverse_resonance():
    # Amplified algorithmic stress: every electron lies beyond nominal r² < 1/s.
    electron, observer, crossing, gamma = .4, .397, 1., 1200.
    x = np.linspace(electron-.000025, electron+.000025, 65)
    y = np.linspace(-.000025, .000025, 65)
    g = np.linspace(1100.,1300.,25)
    H = np.ones((24,64,64,2,1))
    table = Table(g, x, y, np.array([0.,.0001,.0002]), np.array([.5,1.5]), H,
                  np.zeros_like(H), np.zeros_like(H), np.zeros_like(H), 1., 'proposal-support')
    axis = np.array([-np.sin(crossing),0.,-np.cos(crossing)])
    v = np.array([electron,0.,1.]); v /= np.linalg.norm(v)
    D = (1-v@axis)/(1-axis[2])
    energy = np.array([D*gamma**2/(1+(electron-observer)**2*gamma**2)])
    assert (x[0]-observer)**2 > 1/energy[0]
    cpu = angular_spectrum_from_table(table,[observer],[0.],energy,theta_xz=crossing,backend='numpy')
    gpu = sampler.calculate_angular_spectrum_gpu(table,[observer],[0.],energy,theta_xz=crossing,rings=64,subsampling=128)
    assert cpu.item() > 0
    np.testing.assert_allclose(gpu,cpu,rtol=.05)
