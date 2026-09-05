from __future__ import annotations

import pytest

from gammaforge.engines.analytical.engine import AnalyticalEngine
from gammaforge.gui.state import InputState
from gammaforge.gui.inputs import _field_editor
from gammaforge.io.target import OutputKind
from gammaforge.io.units import WidthConvention


def test_input_state_uses_canonical_values_and_separate_unit_preference():
    state = InputState({"analytical": AnalyticalEngine()})
    before = state.groups["beam"].get_float("sigma_x")

    state.set_display("beam", "sigma_x", unit="mm")
    state.set_display("beam", "sigma_x", convention=WidthConvention.FWHM_INTENSITY)

    assert state.groups["beam"].get_float("sigma_x") == before
    assert state.set_value("beam", "sigma_x", str(state.groups["beam"].display("sigma_x", "mm", WidthConvention.FWHM_INTENSITY)))
    assert state.groups["beam"].get_float("sigma_x") == pytest.approx(before)


def test_request_is_a_widget_free_snapshot():
    state = InputState({"analytical": AnalyticalEngine()})

    request = state.request()

    assert set(request.engine_params) == {"analytical"}
    assert request.target.kinds == (OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM)


def test_invalid_value_is_visible_and_blocks_request():
    state = InputState({"analytical": AnalyticalEngine()})

    assert not state.set_value("beam", "sigma_x", "not a number")
    assert "beam.sigma_x" in state.errors
    with pytest.raises(ValueError, match="Input errors"):
        state.request()


def test_invalid_raw_text_survives_a_pane_rebuild():
    state = InputState({"analytical": AnalyticalEngine()})

    state.set_value("beam", "sigma_x", "unparseable")

    assert state.raw[("beam", "sigma_x")] == "unparseable"
    assert state.errors["beam.sigma_x"]


def test_invalid_text_is_retained_with_a_display_convention():
    state = InputState({"analytical": AnalyticalEngine()})
    state.set_display("beam", "sigma_x", unit="mm", convention=WidthConvention.FWHM_INTENSITY)

    assert not state.set_value("beam", "sigma_x", "not a width")

    assert state.raw[("beam", "sigma_x")] == "not a width"
    assert "beam.sigma_x" in state.errors


def test_requested_outputs_need_a_selected_supporting_engine():
    state = InputState({"analytical": AnalyticalEngine()})

    assert not state.set_requested(OutputKind.ANGULAR_DISTRIBUTION, True, (32, 32))
    assert OutputKind.ANGULAR_DISTRIBUTION not in state.requested


def test_previously_requested_output_can_be_removed_after_engine_selection_changes():
    state = InputState({"analytical": AnalyticalEngine()})
    state.selected.clear()

    assert state.set_requested(OutputKind.SPECTRUM, False)
    assert OutputKind.SPECTRUM not in state.requested


def test_invalid_output_resolution_is_retained_and_blocks_request():
    state = InputState({"analytical": AnalyticalEngine()})

    assert not state.set_requested(OutputKind.SPECTRUM, True, ("",))
    assert state.output_raw[OutputKind.SPECTRUM] == ("",)
    with pytest.raises(ValueError, match="Input errors"):
        state.request()


def test_correcting_a_cross_field_error_clears_the_request_blocker():
    state = InputState({"analytical": AnalyticalEngine()})

    assert not state.set_value("target", "theta_x_col", "0")
    assert "target.physical" in state.errors
    assert state.set_value("target", "theta_x_col", "1")
    assert "target.physical" not in state.errors
    assert state.request().target.m("theta_x_col") == pytest.approx(1e-3)


def test_unit_event_reformats_value_without_physical_change_callback():
    state = InputState({"analytical": AnalyticalEngine()})
    calls: list[tuple[str, str]] = []

    editor = _field_editor(state, "beam", "sigma_x", lambda group, key: calls.append((group, key)))
    value, unit, convention = editor.widgets
    canonical = state.groups["beam"].get_float("sigma_x")

    unit.value = "mm"
    convention.value = WidthConvention.FWHM_INTENSITY.value

    assert calls == [("display", "beam.sigma_x"), ("display", "beam.sigma_x")]
    assert state.groups["beam"].get_float("sigma_x") == canonical
    value.value = "0.02"
    assert calls[-1] == ("beam", "sigma_x")
