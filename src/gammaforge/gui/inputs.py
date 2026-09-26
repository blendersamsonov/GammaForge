"""NiceGUI field editors built directly from public parameter schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..io.schema import FieldKind
from ..io.plotting import display_scale, display_unit
from ..io.results import Axis
from ..io.target import OutputKind, SLICE_AXES
from .state import InputState

try:  # State tests and non-GUI users must not need the optional browser dependency.
    from nicegui import ui
except ImportError:  # pragma: no cover - exercised on installations without NiceGUI
    ui = None  # type: ignore[assignment]

ChangeCallback = Callable[[str, str], None]

# Laser type specific field keys
_GAUSSIAN_LASER_KEYS = ("duration",)
_PULSE_TRAIN_LASER_KEYS = ("subpulse_duration", "repetition_period", "n_subpulses")
_COMMON_LASER_KEYS = (
    "pulse_energy", "wavelength", "laser_type", "sigma_x", "sigma_y",
    "z_fx", "z_fy", "x_off", "y_off", "t_off",
    "theta_xz", "theta_yz", "psi_focus", "psi_pol",
    "ellipticity", "beta_ff",
)

_CONVENTION_LABELS = {
    "sigma_intensity_rms": "RMS intensity",
    "sigma_field_rms": "Field RMS",
    "fwhm_intensity": "FWHM intensity",
    "w0_1e2": "1/e² intensity radius",
}
_AXIS_BIN_LABELS = {
    "energy": "E bins", "time": "t bins", "x": "x bins", "y": "y bins",
    "theta_x": "θx bins", "theta_y": "θy bins",
}
_AXIS_LIMIT_LABELS = {
    "energy": "Energy", "time": "Time", "x": "x", "y": "y",
    "theta_x": "θx", "theta_y": "θy",
}


def _require_ui() -> None:
    if ui is None:
        raise RuntimeError("NiceGUI is required to render GammaForge inputs")


@dataclass
class Editor:
    """Small public handle enabling the page coordinator to lock an input group."""

    group: str
    key: str
    widgets: list[Any] = field(default_factory=list)
    enabled: list[bool] | None = None
    enabled_provider: Callable[[], list[bool]] | None = None
    state_refresher: Callable[[], None] | None = None

    def set_locked(self, locked: bool) -> None:
        if self.state_refresher is not None:
            self.state_refresher()
        if self.enabled_provider is not None:
            self.enabled = self.enabled_provider()
        for index, widget in enumerate(self.widgets):
            enabled = self.enabled is None or self.enabled[index]
            (widget.disable if locked or not enabled else widget.enable)()


def _number_text(state: InputState, group: str, key: str) -> str:
    if state.error_key(group, key) in state.errors:
        return state.raw.get((group, key), "")
    preference = state.preference(group, key)
    return format(state.groups[group].display(key, preference.unit, preference.convention), ".12g")


def _field_editor(state: InputState, group: str, key: str, on_change: ChangeCallback) -> Editor:
    _require_ui()
    params, spec = state.groups[group], state.groups[group].spec(key)
    widgets: list[Any] = []
    with ui.column().classes("gf-field gap-1"):
        if spec.kind is FieldKind.CHOICE:
            control = ui.select(list(spec.choices), value=params.get_choice(key), label=spec.label).props(
                f'dense aria-label="{spec.label}" data-field="{group}.{key}"'
            )
            widgets.append(control)
            control.on_value_change(lambda event, g=group, k=key: (state.set_value(g, k, event.value), on_change(g, k)))
        else:
            presentation_update = False
            with ui.row().classes("gf-field-row w-full items-start no-wrap"):
                control = ui.input(label=spec.label, value=_number_text(state, group, key)).classes("flex-grow").props(
                    f'dense inputmode="decimal" aria-label="{spec.label}" data-field="{group}.{key}"'
                )
                widgets.append(control)
                preference = state.preference(group, key)
                units = tuple(dict.fromkeys((spec.unit, *spec.display_units)))
                if len(units) > 1:
                    unit = ui.select(list(units), value=preference.unit, label="Unit").classes("gf-unit-select").props(f'dense aria-label="{spec.label} unit"')
                    widgets.append(unit)

                if spec.convention is not None:
                    conventions = type(spec.convention)
                    convention = ui.select(
                        {entry.value: _CONVENTION_LABELS[entry.value] for entry in conventions},
                        value=preference.convention.value,
                        label="Convention",
                    ).classes("gf-convention-select").props(f'dense aria-label="{spec.label} convention"')
                    widgets.append(convention)
            error = ui.label(state.errors.get(state.error_key(group, key), "")).classes("text-negative text-caption")

            def change_value(event, g=group, k=key, error_label=error):
                if presentation_update:
                    return
                state.set_value(g, k, event.value)
                error_label.set_text(
                    state.errors.get(state.error_key(g, k), state.errors.get(state.error_key(g, "physical"), ""))
                )
                on_change(g, k)

            control.on_value_change(change_value)
            if len(units) > 1:
                def change_unit(event, g=group, k=key, field=control):
                    nonlocal presentation_update
                    state.set_display(g, k, unit=event.value)
                    presentation_update = True
                    try:
                        field.value = _number_text(state, g, k)
                    finally:
                        presentation_update = False
                    on_change("display", f"{g}.{k}")

                unit.on_value_change(change_unit)
            if spec.convention is not None:
                def change_convention(event, g=group, k=key, field=control):
                    nonlocal presentation_update
                    state.set_display(g, k, convention=conventions(event.value))
                    presentation_update = True
                    try:
                        field.value = _number_text(state, g, k)
                    finally:
                        presentation_update = False
                    on_change("display", f"{g}.{k}")

                convention.on_value_change(change_convention)
    return Editor(group, key, widgets)


def _render_fields(state: InputState, group: str, on_change: ChangeCallback, keys: tuple[str, ...] | None = None) -> list[Editor]:
    selected = set(keys) if keys is not None else None
    return [_field_editor(state, group, spec.key, on_change) for spec in state.groups[group].specs if selected is None or spec.key in selected]


def _section_title(label: str, section: str, on_save_default: Callable[[str], None] | None) -> None:
    with ui.row().classes("w-full items-center justify-between"):
        ui.label(label)
        if on_save_default is not None:
            ui.button(
                "Save as default",
                icon="save",
                on_click=lambda s=section: on_save_default(s),
            ).props("flat dense size=sm")


def render_input_columns(
    state: InputState,
    on_change: ChangeCallback,
    render_geometry: Callable[[], None],
    on_save_default: Callable[[str], None] | None = None,
) -> list[Editor]:
    """Render desktop A/B/C as an explicit equal-height grid (which stacks narrowly)."""
    _require_ui()
    editors: list[Editor] = []
    angle_keys = ("theta_xz", "theta_yz", "psi_focus", "psi_pol")
    
    # Get laser field keys based on current laser_type
    laser_type = state.groups["laser"].get_choice("laser_type")
    if laser_type == "pulse_train":
        laser_keys = tuple(k for k in _COMMON_LASER_KEYS if k not in angle_keys) + _PULSE_TRAIN_LASER_KEYS
    else:
        laser_keys = tuple(k for k in _COMMON_LASER_KEYS if k not in angle_keys) + _GAUSSIAN_LASER_KEYS
    
    with ui.element("div").classes("gf-input-pane"):
        with ui.element("div").classes("gf-input-columns"):
            with ui.card().classes("gf-input-column"):
                _section_title("Electrons", "electrons", on_save_default)
                editors += _render_fields(state, "beam", on_change)
                ui.separator()
                _section_title("Sampling", "sampling", on_save_default)
                editors += _render_fields(state, "sampling", on_change)
            with ui.card().classes("gf-input-column"):
                _section_title("Laser", "laser", on_save_default)
                editors += _render_fields(state, "laser", on_change, laser_keys)
            with ui.card().classes("gf-input-column"):
                _section_title("Geometry", "geometry", on_save_default)
                editors += _render_fields(state, "laser", on_change, angle_keys)
                render_geometry()
    
    ui.add_head_html("""
    <style>
      .gf-input-pane { width:100%; container-type:inline-size; }
      .gf-input-columns { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; align-items:stretch; }
      .gf-input-column { height:100%; max-height:680px; overflow:auto; }
      .gf-field-row { gap:.25rem; }
      .gf-unit-select { width:7rem; }
      .gf-convention-select { width:10rem; }
      .gf-target-angles { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.5rem 1rem; }
      .gf-output-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.5rem; }
      .gf-output-card { padding:.5rem .7rem; border:1px solid #dce4ed; border-radius:6px; }
      .gf-output-resolutions { gap:.25rem; }
      .gf-output-resolutions .q-field { min-width:0; flex:1 1 4.5rem; }
      .gf-output-range { border-top:1px solid #edf1f5; padding-top:.35rem; }
      .gf-output-range .q-field { min-width:0; flex:1 1 5rem; }
      .gf-engine-fields { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.5rem 1rem; }
      .gf-engine-fields .gf-field { min-width:0; }
      @container (max-width: 900px) { .gf-input-columns { grid-template-columns:1fr; } }
      @container (max-width: 760px) { .gf-output-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .gf-engine-fields { grid-template-columns:repeat(2,minmax(0,1fr)); } }
      @container (max-width: 480px) { .gf-target-angles, .gf-output-grid, .gf-engine-fields { grid-template-columns:1fr; } }
    </style>""")
    
    return editors


def render_target(
    state: InputState,
    on_change: ChangeCallback,
    on_save_default: Callable[[str], None] | None = None,
) -> list[Editor]:
    """Render target angles and output choices; estimates and Calculate live in app.py."""
    _require_ui()
    with ui.element("div").classes("gf-target-angles w-full"):
        editors = _render_fields(state, "target", on_change)
    with ui.row().classes("w-full items-center justify-between"):
        ui.label("Requested outputs")
        if on_save_default is not None:
            ui.button(
                "Save outputs as default",
                icon="save",
                on_click=lambda: on_save_default("outputs"),
            ).props("flat dense size=sm")
    with ui.element("div").classes("gf-output-grid w-full"):
        for kind in OutputKind:
            with ui.element("div").classes("gf-output-card"):
                enabled = kind is OutputKind.TOTAL_YIELD or state.supports(kind)
                checked = kind in state.requested
                title = kind.value.replace("_", " ").title()
                checkbox = ui.checkbox(title, value=checked).props(
                    f'dense aria-label="Request {title}" data-field="outputs.{kind.name}"'
                )
                if kind is OutputKind.TOTAL_YIELD:
                    checkbox.disable()
                elif not enabled and not checked:
                    checkbox.disable()
                    checkbox.tooltip("No selected engine supports this output")
                elif not enabled:
                    checkbox.tooltip("Unavailable with the selected engine. Uncheck to remove this request.")
                error = ui.label(state.errors.get(state.error_key("outputs", kind.name), "")).classes("text-negative text-caption")
                widgets = [checkbox]
                axes = SLICE_AXES[kind]
                resolution = state.output_raw.get(kind, tuple(str(value) for value in state.requested.get(kind, (64,) * len(axes or ()))))
                with ui.row().classes("gf-output-resolutions w-full no-wrap"):
                    for index, axis in enumerate(axes or ()):
                        box = ui.input(label=_AXIS_BIN_LABELS[axis.key], value=str(resolution[index])).props(
                            f'dense inputmode="decimal" data-field="outputs.{kind.name}.resolution.{index}"'
                        )
                        visible = checked and enabled
                        box.set_visibility(visible)
                        if not visible:
                            box.disable()
                        widgets.append(box)

                resolution_boxes = widgets[1:].copy()
                range_axes = axes or ()
                range_controls: list[tuple[Axis, Any, Any, Any, Any]] = []
                auto = None
                try:
                    auto = state.auto_range(kind)
                except ValueError:
                    auto = None
                for axis in range_axes:
                    manual = state.manual_ranges.get(kind, {}).get(axis)
                    is_manual = (kind, axis) in state.manual_axes
                    shown = state.range_raw.get((kind, axis))
                    if shown is None:
                        canonical = manual or ((auto or {}).get(axis) if auto is not None else None)
                        shown = (
                            (format(canonical[0] * display_scale(axis), ".8g"),
                             format(canonical[1] * display_scale(axis), ".8g"))
                            if canonical is not None else ("", "")
                        )
                    with ui.column().classes("gf-output-range w-full gap-1"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(f"{_AXIS_LIMIT_LABELS[axis.key]} limits").classes("text-caption")
                            auto_box = ui.checkbox("Auto", value=not is_manual).props("dense")
                        if auto is not None:
                            low_auto, high_auto = auto[axis]
                            scale = display_scale(axis)
                            auto_label = ui.label(
                                f"Auto: {low_auto * scale:.6g} to {high_auto * scale:.6g} {display_unit(axis)}"
                            ).classes("text-caption text-grey-7")
                        if auto is None:
                            message = (
                                "Auto: sampled overlap at Calculate"
                                if kind is OutputKind.TEMPORAL_ENVELOPE
                                else "Auto: unavailable until inputs are valid"
                            )
                            auto_label = ui.label(message).classes(
                                "text-caption text-grey-7"
                            )
                        with ui.row().classes("w-full no-wrap gap-1"):
                            low_box = ui.input("Min", value=shown[0]).props("dense inputmode=decimal")
                            high_box = ui.input("Max", value=shown[1]).props("dense inputmode=decimal")
                    widgets.extend((auto_box, low_box, high_box))
                    range_controls.append((axis, auto_label, auto_box, low_box, high_box))

                    def update_auto(
                        event,
                        output=kind,
                        output_axis=axis,
                        low=low_box,
                        high=high_box,
                    ):
                        if event.value:
                            state.set_output_auto(output, output_axis, True)
                        else:
                            state.set_output_range(output, output_axis, low.value, high.value)
                        on_change("outputs", output.name)

                    def update_range(
                        _event,
                        output=kind,
                        output_axis=axis,
                        low=low_box,
                        high=high_box,
                        auto_control=auto_box,
                        error_label=error,
                    ):
                        if not auto_control.value:
                            state.set_output_range(output, output_axis, low.value, high.value)
                            error_label.set_text(
                                state.errors.get(
                                    state.error_key("outputs", f"{output.name}.{output_axis.key}.range"), ""
                                )
                            )
                            on_change("outputs", output.name)

                    auto_box.on_value_change(update_auto)
                    low_box.on_value_change(update_range)
                    high_box.on_value_change(update_range)

                def refresh_ranges(output=kind, controls=range_controls):
                    try:
                        current_auto = state.auto_range(output)
                    except ValueError:
                        current_auto = None
                    for output_axis, label, auto_control, low, high in controls:
                        if current_auto is None:
                            label.set_text(
                                "Auto: sampled overlap at Calculate"
                                if output is OutputKind.TEMPORAL_ENVELOPE
                                else "Auto: unavailable until inputs are valid"
                            )
                            continue
                        low_auto, high_auto = current_auto[output_axis]
                        scale = display_scale(output_axis)
                        label.set_text(
                            f"Auto: {low_auto * scale:.6g} to {high_auto * scale:.6g} "
                            f"{display_unit(output_axis)}"
                        )
                        if auto_control.value:
                            low.value = format(low_auto * scale, ".8g")
                            high.value = format(high_auto * scale, ".8g")

                def enabled_now(output=kind, resolution_axes=range_axes, output_axes=range_axes):
                    active = state.supports(output) and output in state.requested
                    flags = [output is not OutputKind.TOTAL_YIELD and (state.supports(output) or output in state.requested)]
                    flags.extend(active for _ in resolution_axes)
                    for output_axis in output_axes:
                        manual_axis = (output, output_axis) in state.manual_axes
                        flags.extend((active, active and manual_axis, active and manual_axis))
                    return flags

                editor = Editor(
                    "outputs", kind.name, widgets,
                    enabled_provider=enabled_now,
                    state_refresher=refresh_ranges,
                )
                editor.enabled = enabled_now()
                editors.append(editor)

                def update_output(event, output=kind, boxes=resolution_boxes, error_label=error):
                    accepted = state.set_requested(output, bool(event.value), tuple(box.value for box in boxes))
                    error_label.set_text(state.errors.get(state.error_key("outputs", output.name), ""))
                    visible = accepted and bool(event.value) and state.supports(output)
                    for box in boxes:
                        box.set_visibility(visible)
                        (box.enable if visible else box.disable)()
                    on_change("outputs", output.name)

                checkbox.on_value_change(update_output)
                for box in resolution_boxes:
                    def update_resolution(_event, output=kind, check=checkbox, boxes=resolution_boxes, error_label=error):
                        state.set_requested(output, bool(check.value), tuple(item.value for item in boxes))
                        error_label.set_text(state.errors.get(state.error_key("outputs", output.name), ""))
                        on_change("outputs", output.name)
                    box.on_value_change(update_resolution)
    return editors


def render_engine_selector(
    state: InputState,
    on_change: ChangeCallback,
    on_save_default: Callable[[str], None] | None = None,
) -> tuple[Any, list[Editor]]:
    """Render engine selection dropdown and return (select_widget, editors_for_selected_engine)."""
    _require_ui()
    editors: list[Editor] = []
    
    engine_names = list(state.engines.keys())
    current = state.selected_engine or (engine_names[0] if engine_names else None)
    
    with ui.row().classes("w-full items-center gap-4"):
        ui.label("Engine").classes("text-weight-medium")
        select = ui.select(
            engine_names,
            value=current,
            label="Calculation engine",
        ).props('dense').classes("w-64")
        
        def on_engine_change(event):
            engine_name = event.value
            if state.set_selected_engine(engine_name):
                on_change("engine", "selected")
                # Refresh will be triggered by the page
        select.on_value_change(on_engine_change)
        if on_save_default is not None:
            ui.button(
                "Save engine defaults",
                icon="save",
                on_click=lambda: on_save_default("engine"),
            ).props("flat dense size=sm")
    
    # Render fields for the currently selected engine
    if current and f"engine:{current}" in state.groups:
        with ui.element("div").classes("gf-engine-fields w-full mt-4"):
            editors += _render_fields(state, f"engine:{current}", on_change)
    
    return select, editors


def render_selected_engine_fields(state: InputState, on_change: ChangeCallback) -> list[Editor]:
    """Render parameter fields for the currently selected engine."""
    _require_ui()
    editors: list[Editor] = []
    current = state.selected_engine
    if current and f"engine:{current}" in state.groups:
        with ui.element("div").classes("gf-engine-fields w-full"):
            editors += _render_fields(state, f"engine:{current}", on_change)
    return editors
