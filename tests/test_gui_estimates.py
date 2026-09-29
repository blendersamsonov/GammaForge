"""The analytical preview depends only on physical inputs used by its formulas."""

import pytest

from gammaforge.engines.runner import LocalRunner
from gammaforge.gui.state import InputState
from gammaforge.io.target import OutputKind


pytestmark = [pytest.mark.tier1, pytest.mark.fast]


def test_estimate_survives_unrelated_draft_errors_and_tracks_beam_charge():
    runner = LocalRunner()
    state = InputState(runner.engines)
    first = runner.estimate(state.estimate_request())
    initial_yield = float(first.photon_slices[OutputKind.TOTAL_YIELD].distr)

    state.set_value("engine:xigma", "n_bins_gamma", "invalid")
    state.set_value("sampling", "n_particles", "invalid")
    state.errors["outputs.SPECTRUM"] = "invalid output resolution"
    state.set_selected_engine(None)
    assert float(runner.estimate(state.estimate_request()).photon_slices[OutputKind.TOTAL_YIELD].distr) == initial_yield

    state.set_value("beam", "bunch_charge", "2")
    updated_yield = float(runner.estimate(state.estimate_request()).photon_slices[OutputKind.TOTAL_YIELD].distr)
    assert updated_yield != initial_yield

    state.set_value("beam", "bunch_charge", "invalid")
    with pytest.raises(ValueError, match="Correct the beam"):
        state.estimate_request()
