"""One representative request-to-engine execution path."""

from dataclasses import replace

import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast]

from gammaforge.engines.runner import LocalRunner
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io.calculation import CalculationRequest
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.validation import scenarios


def test_small_real_xigma_execution():
    scenario = scenarios.BASELINE
    request = CalculationRequest(
        beam=scenario.beam,
        laser=scenario.laser,
        target=replace(
            scenario.target,
            outputs=(OutputRequest(OutputKind.TOTAL_YIELD),),
        ),
        sampling=SamplingSpec(n_particles=64, seed=0, prefilter=0.0),
        engine_params={
            "xigma": XigmaEngine.schema.with_values(
                n_bins_gamma=8,
                n_bins_theta_x=8,
                n_bins_theta_y=8,
                n_bins_a0_shape=8,
                n_bins_ahat=4,
            )
        },
    )

    results = LocalRunner().calculate(request)

    assert results["xigma"].photon_slices[OutputKind.TOTAL_YIELD].integrate() > 0.0
