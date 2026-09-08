"""Exercise the production device CDF search independently of emission physics."""
import numpy as np
import pytest

from gammaforge.engines.xigma import spectrum_sampler as sampler

pytestmark = pytest.mark.skipif(not sampler.is_gpu_available(), reason="CUDA unavailable")


def test_exact_cdf_search_and_weights_integrate_constant_with_peaked_proposal():
    cp, jit = sampler.cp, sampler.jit
    cdf_cell = sampler._cdf_cell

    @jit.rawkernel()
    def evaluate(cumulative, output):
        i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        if i < output.size:
            total = cumulative[31]
            target = (cp.float32(i) + cp.float32(0.5)) / cp.float32(output.size) * total
            cell = cdf_cell(cumulative, cp.uint32(0), target)
            mass = cumulative[cell + 1] - cumulative[cell]
            output[i] = total / (cp.float32(31) * mass)

    weights = np.ones(31, dtype=np.float32)
    weights[15] = 1000.0
    cumulative = cp.asarray(np.r_[np.float32(0), np.cumsum(weights)])
    output = cp.empty(1_000_000, dtype=cp.float32)
    evaluate[((output.size + 127) // 128,), (128,)](cumulative, output)
    assert float(cp.mean(output, dtype=cp.float64)) == pytest.approx(1.0, abs=1e-4)


def test_cdf_search_skips_zero_probability_cells():
    cp, jit = sampler.cp, sampler.jit
    cdf_cell = sampler._cdf_cell

    @jit.rawkernel()
    def locate(cumulative, quantiles, output):
        i = jit.threadIdx.x
        if i < output.size:
            output[i] = cdf_cell(cumulative, cp.uint32(0), quantiles[i])

    weights = np.zeros(31, dtype=np.float32)
    weights[[3, 17, 28]] = [1, 2, 1]
    cumulative = cp.asarray(np.r_[np.float32(0), np.cumsum(weights)])
    quantiles = cp.asarray([0., .5, 1., 2., 3., 3.99], dtype=cp.float32)
    output = cp.empty(6, dtype=cp.uint32)
    locate[1, 32](cumulative, quantiles, output)
    np.testing.assert_array_equal(output.get(), [3, 3, 17, 17, 28, 28])
