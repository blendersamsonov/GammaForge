"""Headless result metadata and replayable Gaussian request regressions."""

from dataclasses import asdict, replace

import numpy as np
import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.engines.analytical.engine import AnalyticalEngine
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io.calculation import CalculationRequest
from gammaforge.io.fields import BEAM_FIELDS, LASER_FIELDS, beam_from_parameters, laser_from_parameters
from gammaforge.io.formats.hdf5 import load_request, load_results, save_results
from gammaforge.io.interaction import SamplingSpec, build_interaction
from gammaforge.io.results import Axis, PhasespaceSlice, Results
from gammaforge.io.schema import Parameters
from gammaforge.io.target import OutputKind, OutputRequest, Target
from gammaforge.io.units import Quantity as Q


def request():
    return CalculationRequest(
        beam_from_parameters(Parameters.from_specs(BEAM_FIELDS)),
        laser_from_parameters(Parameters.from_specs(LASER_FIELDS, theta_xz=0.01)),
        Target(Q(1, "mrad"), Q(2, "mrad"), (
            OutputRequest(OutputKind.TOTAL_YIELD), OutputRequest(OutputKind.SPECTRUM, (24,)),
            OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, (2, 3), {Axis.X: (-0.1, 0.2)}),
        )),
        SamplingSpec(n_particles=128, seed=1729, prefilter=0),
        {"analytical": AnalyticalEngine.schema.with_values(n_quad=64),
         "xigma": XigmaEngine.schema.with_values(n_steps=32)},
    )


def test_actual_analytical_result_and_request_can_be_replayed(tmp_path):
    submitted = request()
    interaction = build_interaction(submitted.beam, submitted.laser, submitted.target, submitted.sampling)
    result = AnalyticalEngine().run(interaction, submitted.engine_params["analytical"])
    path = tmp_path / "run.h5"
    save_results(result, path, request=submitted)
    loaded = load_results(path)
    assert loaded.model_specific["warnings"] == result.model_specific["warnings"]
    assert loaded.model_specific["spectrum_width_fwhm"] == asdict(result.model_specific["spectrum_width_fwhm"])
    restored = load_request(path, {"analytical": AnalyticalEngine.schema, "xigma": XigmaEngine.schema})
    assert restored == submitted
    rebuilt = build_interaction(restored.beam, restored.laser, restored.target, restored.sampling)
    np.testing.assert_array_equal(rebuilt.bunch.x, interaction.bunch.x)
    replay = AnalyticalEngine().run(rebuilt, restored.engine_params["analytical"])
    for kind, original in result.photon_slices.items():
        np.testing.assert_array_equal(loaded.photon_slices[kind].distr, original.distr)
        np.testing.assert_array_equal(replay.photon_slices[kind].distr, original.distr)
    with pytest.raises(ValueError, match="missing engine schemas"):
        load_request(path, {})


def test_metadata_arrays_tuples_and_custom_slice_keys(tmp_path):
    result = Results({"custom": PhasespaceSlice({}, np.asarray(2.0))}, model_specific={
        "array": np.array([1, 2], dtype=np.int32), "empty": np.empty((0, 2)),
        "warnings": ("limitation",), "seed": np.int64(12), "items": [True, None],
    })
    path = tmp_path / "custom.h5"
    save_results(result, path)
    loaded = load_results(path)
    np.testing.assert_array_equal(loaded.model_specific["array"], result.model_specific["array"])
    assert loaded.model_specific["array"].dtype == np.int32
    assert loaded.model_specific["empty"].shape == (0, 2)
    assert loaded.model_specific["warnings"] == ("limitation",)
    assert loaded.photon_slices["custom"].integrate() == 2.0
    with pytest.raises(ValueError, match="unknown provenance"):
        load_request(path, {})


def test_existing_bunch_result_payload_survives_without_new_electron_types(tmp_path):
    submitted = request()
    bunch = build_interaction(submitted.beam, submitted.laser, submitted.target, submitted.sampling).bunch
    bunch = replace(bunch, meta={"last_time": np.linspace(-1e-12, 1e-12, bunch.n_particles)})
    path = tmp_path / "bunch.h5"
    save_results(Results({}, electrons=bunch), path)
    restored = load_results(path).electrons
    assert restored.gaussian_fit == bunch.gaussian_fit
    np.testing.assert_array_equal(restored.gamma, bunch.gamma)
    np.testing.assert_array_equal(restored.meta["last_time"], bunch.meta["last_time"])
