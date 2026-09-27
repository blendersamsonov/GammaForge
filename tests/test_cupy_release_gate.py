"""Permanent CUDA smoke gate for the public crossed-Gaussian Xigma path."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast, pytest.mark.gpu]

from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios


gpu = pytest.mark.skipif(not is_gpu_available(), reason="CuPy or CUDA GPU unavailable")


@pytest.fixture(scope="module")
def crossed_interaction():
    laser = replace(
        scenarios.BASELINE.laser,
        theta_xz=Quantity(0.3, "rad"),
        theta_yz=Quantity(0.2, "rad"),
        ellipticity=0.4,
        psi_pol=Quantity(0.37, "rad"),
    )
    target = replace(
        scenarios.BASELINE.target,
        theta_x_col=Quantity(0.0003, "rad"),
        theta_y_col=Quantity(0.0003, "rad"),
        outputs=(OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(9, 9, 16)),),
    )
    return scenarios.build(
        replace(scenarios.BASELINE, laser=laser, target=target),
        SamplingSpec(n_particles=4_000, seed=20260721, prefilter=1e-3),
    )


@gpu
@pytest.mark.parametrize("rings,n_steps", [(32, 32), (64, 64)])
def test_public_engine_crossed_release_smoke(crossed_interaction, rings, n_steps):
    params = XigmaEngine.schema.with_values(
        n_steps=n_steps,
        n_bins_gamma=12,
        n_bins_theta_x=12,
        n_bins_theta_y=12,
        n_bins_a0_shape=12,
        n_bins_ahat=12,
        backend="cupy",
        sampler_rings=rings,
        sampler_subsampling=32,
    )
    results = XigmaEngine().run(crossed_interaction, params)
    slice_ = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    assert slice_.distr.shape == (9, 9, 16)
    assert np.all(np.isfinite(slice_.distr))
    assert np.all(slice_.distr >= 0.0)
    assert np.any(slice_.distr > 0.0)
    assert results.model_specific["stage2_backend"] == "cupy"
