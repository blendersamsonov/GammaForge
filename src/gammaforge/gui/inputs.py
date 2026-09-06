"""NiceGUI field editors built directly from public parameter schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..io.schema import FieldKind
from ..io.target import OutputKind, SLICE_AXES
from .state import InputState

try:  # State tests and non-GUI users must not need the optional browser dependency.
    from nicegui import ui
except ImportError:  # pragma: no cover - exercised on installations without NiceGUI
    ui = None  # type: ignore[assignment]

ChangeCallback = Callable[[str, str], None]

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

    def set_locked(self, locked: bool) -> None:
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


def render_input_columns(state: InputState, on_change: ChangeCallback, render_geometry: Callable[[], None]) -> list[Editor]:
    """Render desktop A/B/C as an explicit equal-height grid (which stacks narrowly)."""
    _require_ui()
    editors: list[Editor] = []
    angle_keys = ("theta_xz", "theta_yz", "psi_focus", "psi_pol")
    with ui.element("div").classes("gf-input-pane"):
        with ui.element("div").classes("gf-input-columns"):
            with ui.card().classes("gf-input-column"):
                ui.label("Electrons")
                editors += _render_fields(state, "beam", on_change)
                ui.separator()
                ui.label("Sampling")
                editors += _render_fields(state, "sampling", on_change)
            with ui.card().classes("gf-input-column"):
                ui.label("Laser")
                editors += _render_fields(state, "laser", on_change, tuple(k for k in state.groups["laser"] if k not in angle_keys))
            with ui.card().classes("gf-input-column"):
                ui.label("Geometry")
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
      .gf-engine-fields { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.5rem 1rem; }
      .gf-engine-fields .gf-field { min-width:0; }
      @container (max-width: 900px) { .gf-input-columns { grid-template-columns:1fr; } }
      @container (max-width: 760px) { .gf-output-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .gf-engine-fields { grid-template-columns:repeat(2,minmax(0,1fr)); } }
      @container (max-width: 480px) { .gf-target-angles, .gf-output-grid, .gf-engine-fields { grid-template-columns:1fr; } }
    </style>""")
    return editors


def render_target(state: InputState, on_change: ChangeCallback) -> list[Editor]:
    """Render target angles and output choices; estimates and Calculate live in app.py."""
    _require_ui()
    with ui.element("div").classes("gf-target-angles w-full"):
        editors = _render_fields(state, "target", on_change)
    ui.label("Requested outputs")
    with ui.element("div").classes("gf-output-grid w-full"):
        for kind in OutputKind:
            with ui.element("div").classes("gf-output-card"):
                enabled = kind is OutputKind.TOTAL_YIELD or state.supports(kind)
                checked = kind in state.requested
                title = kind.value.replace("_", " ").title()
                checkbox = ui.checkbox(title, value=checked).props(
                    f'dense aria-label="Request {title}" data-field="outputs.{kind.name}"'
                )
                recoverable = kind is not OutputKind.TOTAL_YIELD and (enabled or checked)
                if kind is OutputKind.TOTAL_YIELD:
                    checkbox.disable()
                elif not enabled and not checked:
                    checkbox.disable()
                    checkbox.tooltip("No selected engine supports this output")
                elif not enabled:
                    checkbox.tooltip("Unavailable with the selected engines. Uncheck to remove this request.")
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
                editor = Editor(
                    "outputs", kind.name, widgets,
                    [recoverable, *(enabled and checked for _ in widgets[1:])],
                )
                editors.append(editor)

                def update_output(event, output=kind, boxes=widgets[1:], output_editor=editor, error_label=error):
                    accepted = state.set_requested(output, bool(event.value), tuple(box.value for box in boxes))
                    error_label.set_text(state.errors.get(state.error_key("outputs", output.name), ""))
                    visible = accepted and bool(event.value) and state.supports(output)
                    for box in boxes:
                        box.set_visibility(visible)
                        (box.enable if visible else box.disable)()
                    output_editor.enabled = [output is not OutputKind.TOTAL_YIELD and state.supports(output), *(visible for _ in boxes)]
                    on_change("outputs", output.name)

                checkbox.on_value_change(update_output)
                for box in widgets[1:]:
                    def update_resolution(_event, output=kind, check=checkbox, boxes=widgets[1:], error_label=error):
                        state.set_requested(output, bool(check.value), tuple(item.value for item in boxes))
                        error_label.set_text(state.errors.get(state.error_key("outputs", output.name), ""))
                        on_change("outputs", output.name)
                    box.on_value_change(update_resolution)
    return editors


def render_engines(state: InputState, on_change: ChangeCallback) -> list[Editor]:
    """Render only concrete public engine schemas, with selection in each engine tab."""
    _require_ui()
    editors: list[Editor] = []
    with ui.tabs().classes("w-full") as tabs:
        for name in state.engines:
            ui.tab(name, label=name)
    with ui.tab_panels(tabs, value=next(iter(state.engines), None)).classes("w-full"):
        for name in state.engines:
            with ui.tab_panel(name):
                selected = ui.checkbox("Use for calculation", value=name in state.selected).props(
                    f'aria-label="Use {name} for calculation" data-field="engine:{name}.use"'
                )
                editor = Editor(f"engine:{name}", "use", [selected])
                editors.append(editor)

                def update_selection(event, engine_name=name):
                    enabled = bool(event.value)
                    if (engine_name in state.selected) == enabled:
                        return
                    _set_engine_selected(state, engine_name, enabled)
                    on_change(f"engine:{engine_name}", "use")

                selected.on_value_change(update_selection)
                with ui.element("div").classes("gf-engine-fields w-full"):
                    editors += _render_fields(state, f"engine:{name}", on_change)
    return editors


def _set_engine_selected(state: InputState, name: str, selected: bool) -> None:
    if selected:
        state.selected.add(name)
    else:
        state.selected.discard(name)
