"""The analytical preview depends only on physical inputs used by its formulas."""

import numpy as np
import pytest

from dataclasses import replace

from gammaforge.engines.runner import LocalRunner
from gammaforge.io.calculation import CalculationRequest
from gammaforge.io.target import OutputRequest, Target
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


# ---------------------------------------------------------------------------
# Analytical estimate: full outputs, and appearing as a run of its own
# ---------------------------------------------------------------------------
def test_the_narrow_preview_still_only_asks_for_the_yield():
    """The always-current panel must stay cheap: it re-runs on every keystroke, so it keeps
    narrowing to `TOTAL_YIELD`. This is the regression check that adding an on-demand figure
    did not quietly move a 3D grid evaluation onto the keystroke path."""
    runner = LocalRunner()
    state = InputState(runner.engines)
    state.set_requested(OutputKind.SPECTRUM, True, (101,))
    state.set_requested(OutputKind.COLLIMATED_SPECTRUM, True, (21, 7, 7))

    narrow = state.estimate_request()
    assert [request.kind for request in narrow.target.outputs] == [OutputKind.TOTAL_YIELD]
    result = runner.estimate(narrow)
    assert set(result.photon_slices) == {OutputKind.TOTAL_YIELD}

    # Asserted on the *result*, not only the request: `runner.estimate` can narrow its own
    # target independently, so the request alone does not prove the runner stays cheap.
    assert OutputKind.COLLIMATED_SPECTRUM not in result.photon_slices

    # And the runner's own narrowing is asserted directly, with a request that *does* ask for
    # the expensive outputs. Relying on `estimate_request()` to have already narrowed hides
    # a regression in the runner: both layers narrow today, so removing either one alone
    # changes nothing observable.
    narrow_request = state.estimate_request()
    wide_request = CalculationRequest(
        beam=narrow_request.beam,
        laser=narrow_request.laser,
        sampling=narrow_request.sampling,
        engine_params={},
        target=Target(
            narrow_request.target.theta_x_col,
            narrow_request.target.theta_y_col,
            (
                OutputRequest(OutputKind.TOTAL_YIELD),
                OutputRequest(OutputKind.COLLIMATED_SPECTRUM, (21, 7, 7)),
            ),
        ),
    )
    cheap = runner.estimate(wide_request)
    assert set(cheap.photon_slices) == {OutputKind.TOTAL_YIELD}, (
        "runner.estimate must narrow to TOTAL_YIELD even when asked for more: the live panel "
        "re-runs it on every keystroke"
    )
    assert "models" in result.model_specific

    full = state.estimate_request(full_outputs=True)
    assert {request.kind for request in full.target.outputs} == {
        OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM, OutputKind.COLLIMATED_SPECTRUM
    }


def test_the_full_estimate_returns_the_requested_spectra():
    """`estimate_requested` is the on-demand counterpart to the cheap preview: it keeps the
    user's own output resolutions, so the figure shows what a saved run would contain."""
    runner = LocalRunner()
    state = InputState(runner.engines)
    state.set_requested(OutputKind.COLLIMATED_SPECTRUM, True, (21, 7, 7))
    result = runner.estimate_requested(state.estimate_request(full_outputs=True))

    collimated = result.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    assert collimated.distr.shape == (21, 7, 7)
    assert result.model_specific["collimated_backend"] in {"numpy", "cupy"}
    # The provenance that badges the figure is present, not reconstructed at the frontend.
    assert "collimated_spectrum" in result.model_specific["models"]


def test_calculate_records_the_estimate_as_its_own_badged_run():
    """The point of the change: the analytical result is no longer computed and discarded, so
    it can be compared against the calculated engine in the same plot, exported, and kept.

    Asserts both halves — the engine run still exists, and the estimate arrives separately
    with its own identity — because folding the two into one `Results` would make the
    estimate indistinguishable from a calculation.
    """
    import asyncio

    from gammaforge.engines.catalog import estimate_engine
    from gammaforge.gui.controller import Workspace

    async def run():
        runner = LocalRunner()
        workspace = Workspace(runner)
        workspace.inputs.set_selected_engine("xigma")
        workspace.inputs.set_requested(OutputKind.TOTAL_YIELD, True)
        workspace.inputs.set_requested(OutputKind.COLLIMATED_SPECTRUM, True, (21, 7, 7))
        await workspace.calculate()
        return runner, workspace

    runner, workspace = asyncio.run(run())

    engines = {run.engine_name for run in workspace.runs}
    assert "xigma" in engines
    assert estimate_engine().name in engines, "no run for the analytical estimate"
    assert len(workspace.runs) == 2

    estimate_run = next(r for r in workspace.runs if r.engine_name == estimate_engine().name)
    assert estimate_run.results is not None
    assert OutputKind.COLLIMATED_SPECTRUM in estimate_run.results.photon_slices
    assert estimate_run.status == "completed"

    # The badge text must name the approximation, and it must come from the result's own
    # provenance rather than a hardcoded tier.
    from gammaforge.gui.app import _estimate_tooltip, estimate_name

    assert estimate_name == estimate_engine().name
    tooltip = _estimate_tooltip(estimate_run)
    assert "Analytical estimate" in tooltip
    assert "collimated_fixed_width_zero_emittance" in tooltip


def test_the_estimate_run_does_not_displace_the_calculated_run():
    """Appending the estimate must leave the calculated run selected and intact — the
    estimate is an addition, not a replacement, and selecting it by accident would hide the
    result the user actually asked for."""
    import asyncio

    from gammaforge.engines.catalog import estimate_engine
    from gammaforge.gui.controller import Workspace

    async def run():
        workspace = Workspace()
        workspace.inputs.set_selected_engine("xigma")
        workspace.inputs.set_requested(OutputKind.TOTAL_YIELD, True)
        await workspace.calculate()
        return workspace

    workspace = asyncio.run(run())
    calculated = workspace.current_run
    assert calculated is not None
    assert calculated.engine_name == "xigma"
    assert float(calculated.results.photon_slices[OutputKind.TOTAL_YIELD].distr) > 0.0
    # Run ids stay unique, so selection and deletion cannot collide.
    ids = [r.id for r in workspace.runs]
    assert len(set(ids)) == len(ids)
    assert estimate_engine().name in {r.engine_name for r in workspace.runs}


def test_the_estimate_engine_is_selectable_without_duplicating_its_run():
    """Making the semi-analytical engine selectable must not double-count it.

    Selecting it produces its own run through the normal path, and the always-on overlay
    would otherwise append a second, identical run — two history entries that look like
    different results and double the plotted yield.
    """
    import asyncio

    from gammaforge.engines.catalog import estimate_engine
    from gammaforge.gui.controller import Workspace

    async def select_and_run(engine):
        workspace = Workspace()
        workspace.inputs.set_selected_engine(engine)
        workspace.inputs.set_requested(OutputKind.TOTAL_YIELD, True)
        workspace.inputs.set_requested(OutputKind.COLLIMATED_SPECTRUM, True, (21, 7, 7))
        await workspace.calculate()
        return workspace

    estimate_name = estimate_engine().name
    chosen = asyncio.run(select_and_run(estimate_name))
    assert [r.engine_name for r in chosen.runs] == [estimate_name], "the estimate ran twice"

    # Selecting a calculation engine still yields both, as before.
    both = asyncio.run(select_and_run("xigma"))
    assert {r.engine_name for r in both.runs} == {"xigma", estimate_name}
    assert len(both.runs) == 2


def test_selecting_the_estimate_still_produces_all_its_outputs():
    """Being selectable must not reduce it to a yield: the whole reason to select it is the
    collimated spectrum and angular distribution."""
    import asyncio

    from gammaforge.gui.controller import Workspace

    async def run():
        workspace = Workspace()
        workspace.inputs.set_selected_engine("analytical")
        workspace.inputs.set_requested(OutputKind.TOTAL_YIELD, True)
        workspace.inputs.set_requested(OutputKind.COLLIMATED_SPECTRUM, True, (21, 7, 7))
        await workspace.calculate()
        return workspace

    workspace = asyncio.run(run())
    slices = set(workspace.runs[0].results.photon_slices)
    assert OutputKind.TOTAL_YIELD in slices
    assert OutputKind.COLLIMATED_SPECTRUM in slices
    collimated = workspace.runs[0].results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
    assert collimated.distr.shape == (21, 7, 7)
    assert np.all(np.isfinite(collimated.distr))


def test_the_estimate_overlay_survives_the_estimate_being_selectable():
    """RES058's always-on behaviour is independent of selectability: the estimates panel must
    still work with no engine selected at all, which is the state a user is in while typing."""
    from gammaforge.engines.catalog import selectable_engines
    from gammaforge.gui.state import InputState

    assert "analytical" in selectable_engines()
    state = InputState(selectable_engines())
    state.set_selected_engine(None)
    result = LocalRunner().estimate(state.estimate_request())
    assert set(result.photon_slices) == {OutputKind.TOTAL_YIELD}
