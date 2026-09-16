"""Local browser workspace: page layout and presentation of the public runner."""

from __future__ import annotations

import asyncio
import io
import time
import zipfile
from typing import Any

import yaml

try:
    from nicegui import ui
except ModuleNotFoundError:  # Physics/documentation-only installations omit the GUI extra.
    ui = None

from ..io.bunch import validate as validate_beam
from ..io.fields import beam_from_parameters, laser_from_parameters, to_parameters
from ..io.formats.yaml_spec import SPEC_VERSION, parameters_to_yaml_dict
from ..io.laser import validate as validate_laser
from ..io.target import OutputKind
from .controller import Workspace
from .inputs import (
    render_engine_selector,
    render_input_columns,
    render_selected_engine_fields,
    render_target,
)
from .outputs import render_geometry, render_results


_CSS = """
body { background: #f3f6fa; color: #26364a; }
.q-page { min-width: 0; }
.nicegui-content { padding: 0; gap: 0; }
.gf-header { background: #152c43; color: white; padding: 12px 24px; }
.gf-workspace { width: 100%; padding: 16px; }
.gf-pane { min-width: 0; container-type: inline-size; }
.gf-pane > .q-tabs { background: white; border-radius: 8px 8px 0 0; }
.gf-pane > .q-tab-panels { background: transparent; }
.gf-pane .q-tab-panel { padding: 12px 0; }
.gf-section { width: 100%; padding: 18px; border: 1px solid #dce4ed;
  border-radius: 10px; box-shadow: none; }
.gf-input-columns { width: 100%; }
.gf-input-column { border: 1px solid #dce4ed; box-shadow: none; border-radius: 10px; }
.gf-field { width: 100%; }
.gf-field .q-field { min-width: 0; }
.gf-section-title { font-size: 17px; font-weight: 600; }
.gf-status { padding: 8px 12px; background: #e8eef5; border-radius: 6px; }
.gf-error { white-space: pre-wrap; color: #a22232; }
.gf-estimate { min-width: 130px; padding-right: 20px; }
.gf-estimate-value { font-size: 21px; font-variant-numeric: tabular-nums; }
.q-splitter__separator { background: #c8d6e3; width: 5px; }
.q-splitter__before, .q-splitter__after { min-width: 0; overflow: auto; }
.gf-pane .q-tab-panel > .nicegui-column { width: 100%; }
/* Run history panel */
.gf-run-panel { width: 240px; min-width: 240px; max-width: 240px; height: 100%;
  overflow-y: auto; border-right: 1px solid #dce4ed; padding: 12px; background: #f8fafc; }
.gf-run-item { padding: 8px 10px; margin-bottom: 6px; border: 1px solid #dce4ed;
  border-radius: 6px; cursor: pointer; transition: all 0.15s ease; }
.gf-run-item:hover { background: #e8eef5; }
.gf-run-item.selected { background: #d0e1ed; border-color: #256782; }
.gf-run-item-header { display: flex; justify-content: space-between; align-items: center; }
.gf-run-item-name { font-weight: 500; font-size: 13px; }
.gf-run-item-engine { font-size: 11px; color: #6b7280; margin-top: 2px; }
.gf-run-item-status { font-size: 11px; }
.gf-run-status-completed { color: #16a34a; }
.gf-run-status-running { color: #d97706; }
.gf-run-status-failed { color: #dc2626; }
.gf-run-status-pending { color: #6b7280; }
.gf-run-actions { display: flex; gap: 4px; margin-top: 6px; }
.gf-input-layout { display: flex; width: 100%; height: 100%; }
.gf-input-main { flex: 1; overflow-y: auto; min-width: 0; }
"""


class BrowserWorkspace:
    """Views of one page-owned Workspace, including two optional independent panes."""

    def __init__(self) -> None:
        self.model = Workspace()
        self.client = ui.context.client
        self.split = False
        self.active_tabs = {0: "inputs", 1: "results"}
        self.panes: list[Pane] = []
        self.view_states: dict[int, dict] = {0: {}, 1: {}}
        self.preview = None
        self.preview_revision = -1
        self.preview_running = False
        self.preview_error = ""
        self.last_result_state = None
        self.last_status_state = None
        # Settings
        self.show_debug_warnings = True
        with ui.row().classes("gf-header w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("GammaForge").classes("text-h5 font-medium")
                ui.label("Electron–laser interaction").classes("text-caption opacity-80")
            ui.switch("Split view", value=False, on_change=self._toggle_split)
        self.content = ui.column().classes("gf-workspace")
        self._layout()
        ui.timer(0.2, self._tick)

    def _toggle_split(self, event) -> None:
        self.split = event.value
        self._layout()

    def _layout(self) -> None:
        self.content.clear()
        self.panes.clear()
        with self.content:
            if self.split:
                with ui.splitter(value=50, limits=(25, 75)).classes("w-full") as splitter:
                    with splitter.before:
                        self.panes.append(Pane(self, 0))
                    with splitter.after:
                        self.panes.append(Pane(self, 1))
            else:
                self.panes.append(Pane(self, 0))
        self._update_locks()

    def changed(self, source: int, group: str, key: str) -> None:
        if group == "display":
            for pane in self.panes:
                if pane.index != source:
                    pane.inputs.refresh()
            self._update_locks()
            return
        if group == "engine" and key == "selected":
            # Engine selection changed — refresh engine fields and outputs
            for pane in self.panes:
                pane.engine_fields.refresh()
                pane.inputs.refresh()
            self.preview_error = ""
            for pane in self.panes:
                pane.estimates.refresh()
                pane.status.refresh()
                pane.results.refresh()
            self._update_locks()
            return
        self.model.changed(group, key)
        self.preview_error = ""
        for pane in self.panes:
            if pane.index != source:
                pane.inputs.refresh()
            pane.estimates.refresh()
            pane.status.refresh()
            pane.results.refresh()
        self._update_locks()

    def _update_locks(self) -> None:
        state = self.model.inputs
        for pane in self.panes:
            for editor in pane.editors:
                if editor.group == "outputs":
                    kind = OutputKind[editor.key]
                    supported = state.supports(kind)
                    requested = kind in state.requested
                    editor.enabled = [
                        kind is not OutputKind.TOTAL_YIELD and (supported or requested),
                        *(supported and requested for _ in editor.widgets[1:]),
                    ]
                editor.set_locked(self.model.busy)

    def _refresh_run_panels(self) -> None:
        """Refresh the run history panel on all panes."""
        for pane in self.panes:
            pane.run_panel.refresh()
            pane.inputs.refresh()
            pane.status.refresh()
            pane.results.refresh()

    async def calculate(self) -> None:
        if self.model.busy:
            return
        task = asyncio.create_task(self.model.calculate())
        # Let the controller capture the draft and set busy before refreshing controls.
        await asyncio.sleep(0)
        self._update_locks()
        for pane in self.panes:
            pane.status.refresh()
        await task
        if not self.client.is_deleted:
            self._update_locks()
            self._refresh_run_panels()

    async def _tick(self) -> None:
        if self.client.is_deleted:
            return
        status_state = (self.model.busy, self.model.stale, self.model.error,
                        tuple(self.model.statuses.items()), tuple(self.model.inputs.errors.items()))
        if status_state != self.last_status_state:
            self.last_status_state = status_state
            for pane in self.panes:
                pane.status.refresh()
        result_state = (id(self.model.runs), self.model.stale)
        if result_state != self.last_result_state:
            self.last_result_state = result_state
            for pane in self.panes:
                pane.results.refresh()
        if self.preview_running or self.preview_revision == self.model.revision:
            return
        revision = self.model.revision
        try:
            request = self.model.inputs.request()
        except ValueError as exc:
            self.preview_error = str(exc)
            self.preview_revision = revision
            for pane in self.panes:
                pane.estimates.refresh()
            return
        self.preview_running = True
        try:
            result = await asyncio.to_thread(self.model.runner.estimate, request)
            if revision == self.model.revision:
                self.preview = result
                self.preview_error = ""
        except Exception as exc:
            if revision == self.model.revision:
                self.preview_error = str(exc)
        finally:
            self.preview_revision = revision
            self.preview_running = False
        if not self.client.is_deleted:
            for pane in self.panes:
                pane.estimates.refresh()
                pane.geometry.refresh()

    def download_snapshot(self) -> None:
        current = self.model.current_run
        request = current.request if current else None
        if request is None:
            return
        groups = self.model.inputs.groups
        inputs = {"version": SPEC_VERSION}
        for name in ("beam", "laser", "sampling"):
            params = to_parameters(getattr(request, name), groups[name].specs)
            inputs[name] = parameters_to_yaml_dict(params)
        settings = {
            "target": {key: {"value": request.target.m(key), "unit": "rad"}
                       for key in request.target.UNITS},
            "outputs": [{"kind": output.kind.value, "resolution": list(output.resolution)}
                        for output in request.target.outputs],
            "engines": {name: parameters_to_yaml_dict(params)
                        for name, params in request.engine_params.items()},
        }
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("inputs.yaml", yaml.safe_dump(inputs, sort_keys=False))
            bundle.writestr("calculation.yaml", yaml.safe_dump(settings, sort_keys=False))
        ui.download(archive.getvalue(), "gammaforge-inputs.zip", "application/zip")


class Pane:
    def __init__(self, page: BrowserWorkspace, index: int) -> None:
        self.page, self.index = page, index
        self.editors = []

        # Each refreshable belongs to one pane, so an edit can refresh its sibling
        # without rebuilding the focused form and losing the cursor position.
        self.run_panel = ui.refreshable(self._run_panel)
        self.inputs = ui.refreshable(self._inputs)
        self.engine_fields = ui.refreshable(self._engine_fields)
        self.results = ui.refreshable(self._results)
        self.status = ui.refreshable(self._status)
        self.estimates = ui.refreshable(self._estimates)
        self.geometry = ui.refreshable(self._geometry)
        self.settings = ui.refreshable(self._settings)
        with ui.column().classes("gf-pane w-full gap-0"):
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("inputs", label="Inputs", icon="tune")
                ui.tab("results", label="Results", icon="show_chart")
                ui.tab("settings", label="Settings", icon="settings")
            with ui.tab_panels(tabs, value=page.active_tabs[index],
                               on_change=lambda e: page.active_tabs.__setitem__(index, e.value),
                               animated=False).classes("w-full"):
                with ui.tab_panel("inputs"):
                    self._inputs_with_panel()
                with ui.tab_panel("results"):
                    self.results()
                with ui.tab_panel("settings"):
                    self.settings()

    def _inputs_with_panel(self) -> None:
        """Render inputs tab with run history panel on the left."""
        with ui.element("div").classes("gf-input-layout w-full h-full"):
            with ui.element("div").classes("gf-run-panel"):
                self.run_panel()
            with ui.element("div").classes("gf-input-main"):
                self.inputs()

    def _run_panel(self) -> None:
        """Render the run history panel on the left side of Inputs tab."""
        model = self.page.model

        ui.label("Run History").classes("text-subtitle2 font-medium mb-2")
        ui.button("New Run", icon="add", on_click=self._new_run).props("flat dense color=primary")
        ui.separator().classes("my-2")

        if not model.runs:
            ui.label("No runs yet. Click Calculate to start.").classes("text-grey text-caption")
            return

        for run in reversed(model.runs):  # Show newest first
            is_selected = run.id == model.current_run_id
            status_class = f"gf-run-status-{run.status}"

            with ui.element("div").classes(
                f"gf-run-item {'selected' if is_selected else ''}"
            ):
                with ui.element("div").classes("gf-run-item-header").on(
                    "click", lambda rid=run.id: self._select_run(rid)
                ):
                    ui.label(run.name).classes("gf-run-item-name")
                    with ui.row().classes("items-center gap-1"):
                        icon = {"completed": "check_circle", "running": "pending",
                                "failed": "error", "pending": "schedule"}.get(run.status, "help")
                        ui.icon(icon, size="sm").classes(status_class)
                ui.label(run.engine_name).classes("gf-run-item-engine").on(
                    "click", lambda rid=run.id: self._select_run(rid)
                )
                ts = time.strftime("%H:%M:%S", time.localtime(run.timestamp))
                ui.label(ts).classes("text-caption text-grey").on(
                    "click", lambda rid=run.id: self._select_run(rid)
                )

                # Action buttons
                with ui.row().classes("w-full mt-1"):
                    ui.button(icon="edit", on_click=lambda rid=run.id: self._rename_run(rid)).props(
                        "flat dense size=xs color=grey-7"
                    )
                    ui.button(icon="content_copy", on_click=lambda rid=run.id: self._fork_run(rid)).props(
                        "flat dense size=xs color=grey-7"
                    )
                    ui.button(icon="delete", on_click=lambda rid=run.id: self._delete_run(rid)).props(
                        "flat dense size=xs color=negative"
                    )

    def _new_run(self) -> None:
        """Clear the draft for a fresh calculation."""
        model = self.page.model
        model.select_run(None)
        self.page._refresh_run_panels()

    def _select_run(self, run_id: int) -> None:
        """Select a run in the history panel."""
        model = self.page.model
        model.select_run(run_id)
        self.page._refresh_run_panels()

    def _fork_run(self, run_id: int) -> None:
        """Load a historical run's inputs into the draft for editing."""
        model = self.page.model
        if model.fork_run(run_id):
            self.page._refresh_run_panels()

    def _delete_run(self, run_id: int) -> None:
        """Delete a run from history."""
        model = self.page.model
        if model.delete_run(run_id):
            self.page._refresh_run_panels()

    def _rename_run(self, run_id: int) -> None:
        """Open a dialog to rename a run."""
        model = self.page.model
        run = next((r for r in model.runs if r.id == run_id), None)
        if run is None:
            return

        dialog = ui.dialog()
        with dialog:
            with ui.card():
                ui.label("Rename Run").classes("text-h6")
                name_input = ui.input("Run name", value=run.name).classes("w-full")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Save", on_click=lambda: self._save_rename(run_id, name_input.value, dialog)).props("flat color=primary")
        dialog.open()

    def _save_rename(self, run_id: int, new_name: str, dialog: Any) -> None:
        """Save the new run name."""
        model = self.page.model
        if model.rename_run(run_id, new_name):
            dialog.close()
            self.page._refresh_run_panels()

    def _inputs(self) -> None:
        model = self.page.model
        change = lambda group, key: self.page.changed(self.index, group, key)
        with ui.column().classes("w-full gap-4"):
            self.editors = render_input_columns(model.inputs, change, self.geometry)
            with ui.card().classes("gf-section"):
                ui.label("Target and outputs").classes("gf-section-title")
                self.editors += render_target(model.inputs, change)
            with ui.card().classes("gf-section"):
                self.estimates()
            with ui.card().classes("gf-section"):
                ui.label("Calculation engine").classes("gf-section-title")
                select, engine_editors = render_engine_selector(model.inputs, change)
                self.editors += engine_editors
                self.engine_fields.refresh()
                self.status()

    def _engine_fields(self) -> None:
        """Refreshable engine fields for the currently selected engine."""
        model = self.page.model
        change = lambda group, key: self.page.changed(self.index, group, key)
        self.editors = [e for e in self.editors if not e.group.startswith("engine:")]
        self.editors += render_selected_engine_fields(model.inputs, change)

    def _geometry(self) -> None:
        state = self.page.model.inputs
        try:
            beam = beam_from_parameters(state.groups["beam"])
            laser = laser_from_parameters(state.groups["laser"])
            render_geometry(beam, laser, view_state=self.page.view_states[self.index])
        except ValueError as exc:
            ui.label(f"Sketch unavailable: {exc}").classes("gf-error")

    def _estimates(self) -> None:
        page = self.page
        ui.label("Analytical estimates").classes("gf-section-title")
        if page.preview_error:
            ui.label(page.preview_error).classes("gf-error")
            return
        if page.preview_revision != page.model.revision or page.preview is None:
            ui.label("Updating estimates…").classes("text-grey-7")
            return
        result = page.preview
        total = result.photon_slices.get(OutputKind.TOTAL_YIELD)
        width = result.model_specific.get("spectrum_width_fwhm")
        with ui.row().classes("w-full gap-4"):
            if total is not None:
                self._metric("Total yield", f"{float(total.distr):.5g} photons")
            if width is not None:
                for label, attr in (("Total width", "total"), ("Collimation", "collimation"),
                                    ("Emittance", "emittance"), ("Energy spread", "energy_spread"),
                                    ("Nonlinearity", "nonlinearity")):
                    self._metric(label, f"{getattr(width, attr):.3%}")
        if width is not None:
            ui.label("Widths are FWHM relative to the Compton edge. "
                     f"Nonlinear broadening bracket: {width.nonlinearity_lo:.3%}–"
                     f"{width.nonlinearity_hi:.3%}.").classes("text-caption text-grey-7")

    @staticmethod
    def _metric(label: str, value: str) -> None:
        with ui.column().classes("gf-estimate gap-1"):
            ui.label(label).classes("text-caption text-grey-7")
            ui.label(value).classes("gf-estimate-value")

    def _status(self) -> None:
        model = self.page.model
        with ui.row().classes("items-center gap-3"):
            button = ui.button("Calculate", icon="play_arrow", on_click=self.page.calculate)
            if model.busy or model.inputs.errors or not model.inputs.selected_engine:
                button.disable()
            if model.busy:
                ui.spinner(size="sm")
            ui.label("Calculating…" if model.busy else "Results outdated — Calculate to update"
                     if model.stale else "Ready").classes("gf-status")
        for name, status in model.statuses.items():
            ui.label(f"{name}: {status}").classes("text-caption")
        if model.error:
            ui.label(model.error).classes("gf-error")
        for error in dict.fromkeys(model.inputs.errors.values()):
            ui.label(error).classes("gf-error")
        if self.page.show_debug_warnings:
            try:
                state = model.inputs
                warnings = validate_beam(beam_from_parameters(state.groups["beam"]))
                warnings += validate_laser(laser_from_parameters(state.groups["laser"]))
                for warning in warnings:
                    ui.label(warning).classes("text-amber-10 text-caption")
            except ValueError:
                pass  # The field errors already identify invalid physical inputs.

    def _results(self) -> None:
        model = self.page.model
        completed_runs = model.completed_runs
        requested = tuple(model.inputs.requested)
        if model.busy:
            ui.label("Calculation in progress; completed results appear below.").classes("gf-status")
        if model.error:
            ui.label(model.error).classes("gf-error")
        if completed_runs:
            ui.button("Download input snapshot", icon="download", on_click=self.page.download_snapshot).props("outline")
        render_results(
            completed_runs,
            requested,
            stale=model.stale,
            view_state=self.page.view_states[self.index],
        )

    def _settings(self) -> None:
        """Render the Settings tab with program info and debug options."""
        with ui.column().classes("w-full gap-4"):
            with ui.card().classes("gf-section"):
                ui.label("About GammaForge").classes("gf-section-title")
                ui.label(
                    "GammaForge computes properties of Compton photons produced by "
                    "an electron-bunch / laser-pulse interaction."
                ).classes("text-grey-7")
                ui.label(
                    "Version: 0.1.0 (development)  |  "
                    "Physics: Compton scattering in CGS-Gaussian units  |  "
                    "Engines: analytical, xigma (GPU), kascade"
                ).classes("text-caption text-grey-7")
                ui.separator()
                ui.label("Documentation:").classes("text-subtitle1")
                with ui.column().classes("gap-1"):
                    ui.label("• GRAND_PLAN.md — Architecture and phase plan")
                    ui.label("• PROGRESS.md — Current state and open threads")
                    ui.label("• docs/decisions/ — Implementation decisions (RESNNN)")
                    ui.label("• docs/derivations/ — Physics derivations (DERNNN)")

            with ui.card().classes("gf-section"):
                ui.label("Debug & Display").classes("gf-section-title")
                ui.switch(
                    "Show debug warnings in status",
                    value=self.page.show_debug_warnings,
                    on_change=lambda e: setattr(self.page, "show_debug_warnings", e.value)
                ).props("dense")
                ui.label(
                    "When disabled, validation warnings (e.g., focus position, "
                    "paraxial approximation limits) are hidden from the status panel."
                ).classes("text-caption text-grey-7")


def index() -> None:
    ui.colors(primary="#256782", secondary="#51778e", accent="#a67635")
    ui.add_css(_CSS)
    BrowserWorkspace()


def start(*, port: int = 8080, show: bool = True) -> None:
    if ui is None:
        raise ModuleNotFoundError("Install the GUI extra to launch NiceGUI", name="nicegui")
    ui.page("/")(index)
    ui.run(host="127.0.0.1", port=port, show=show, reload=False, title="GammaForge")