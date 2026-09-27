"""Numerical invariants used by the report's independent comparisons."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from report_figure_common import errors, integrate_values, quadrature_nodes
from report_sampling_ablation import load_variants
from gammaforge.engines.xigma import spectrum_sampler as sampler
from gammaforge.engines.xigma.stages import Table, angular_spectrum_from_table
from gammaforge.validation.cupy_convergence import _refine_table

pytestmark = [pytest.mark.tier1, pytest.mark.fast]
gpu = pytest.mark.skipif(not sampler.is_gpu_available(), reason="CUDA unavailable")


def test_bin_mass_conversion_is_exact_and_not_divided_twice():
    edges = np.array([1., 1.1, 2., 4.])
    x, w = quadrature_nodes(edges, 3)
    actual = integrate_values(2+3*x, w)
    expected = 2*np.diff(edges)+1.5*np.diff(edges**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    np.testing.assert_allclose(integrate_values(np.stack([2+3*x, 4+6*x]), w), [expected, 2*expected])
    assert errors(2*actual, actual)["l1"] == pytest.approx(1.)
    assert errors(2*actual, actual)["yield_error"] == pytest.approx(1.)


def test_dark_bins_retain_spurious_signal():
    assert errors([1., 1.], [1., 0.])["l1"] == 1.


@gpu
@pytest.mark.gpu
def test_real_sampler_ablations_preserve_integrand_and_do_not_patch_production():
    g = np.linspace(1800.,2200.,9)
    x = y = np.linspace(-.0015,.0015,9)
    a = np.array([0.,.01,.03,.08])
    G,X,Y,A = np.meshgrid((g[:-1]+g[1:])/2,(x[:-1]+x[1:])/2,
                          (y[:-1]+y[1:])/2,(a[:-1]+a[1:])/2,indexing="ij")
    h = np.exp(-.5*((G-2000)/80)**2-.5*(X/.0006)**2-.5*(Y/.0006)**2)*(1+A)
    h = h[..., None]
    table = Table(g, x, y, a, np.array([.5,1.5]), h, np.zeros_like(h),
                  np.zeros_like(h), np.zeros_like(h), float(h.sum()), "report-ablation-test")
    nodes = np.array([.35,.5,.7])*2000**2
    kwargs = dict(psi_pol=.37,ellipticity=.4,theta_xz=.02,theta_yz=-.015)
    original_kernel = sampler._kernel
    original_source = Path(sampler.__file__).read_bytes()
    modules, _ = load_variants()
    expected = sampler.calculate_angular_spectrum_gpu(table,[0.],[0.],nodes,subsampling=64,**kwargs)
    control = modules["importance_qmc"].calculate_angular_spectrum_gpu(table,[0.],[0.],nodes,subsampling=64,**kwargs)
    np.testing.assert_allclose(control,expected,rtol=3e-6)
    reference = angular_spectrum_from_table(_refine_table(table,16),[0.],[0.],nodes,backend="numpy",**kwargs)
    for mode,module in modules.items():
        estimates = []
        for seed in (711,812,913,1014):
            module.REPORT_SEED = seed
            estimates.append(module.calculate_angular_spectrum_gpu(table,[0.],[0.],nodes,subsampling=64,**kwargs))
        np.testing.assert_allclose(np.mean(estimates,axis=0),reference,rtol=.04)
        if mode == "importance_mc":
            assert not np.array_equal(estimates[0],estimates[1])
            module.REPORT_SEED = 711
            repeated = module.calculate_angular_spectrum_gpu(table,[0.],[0.],nodes,subsampling=64,**kwargs)
            np.testing.assert_allclose(repeated,estimates[0],rtol=3e-6)
    assert sampler._kernel is original_kernel
    assert Path(sampler.__file__).read_bytes() == original_source
