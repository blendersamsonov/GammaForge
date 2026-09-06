"""Local browser workspace: page layout and presentation of the public runner."""

from __future__ import annotations

import asyncio
import io
import zipfile

import yaml

try:
    from nicegui import ui
except ModuleNotFoundError:  # Physics/documentation-only installations omit the GUI extra.
    ui = None

from ..engines.base import RecomputeCost
from ..io.bunch import validate as validate_beam
from ..io.fields import beam_from_parameters, laser_from_parameters, to_parameters
from ..io.formats.yaml_spec import SPEC_VERSION, parameters_to_yaml_dict
from ..io.laser import validate as validate_laser
from ..io.target import OutputKind
from .controller import Workspace
from .inputs import render_engines, render_input_columns, render_target
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
        self.model.changed(group, key)
        self.preview_error = ""
        for pane in self.panes:
            if key == "use":
                for editor in pane.editors:
                    if (editor.group, editor.key) == (group, key):
                        name = group.split(":", 1)[1]
                        editor.widgets[0].set_value(name in self.model.inputs.selected)
            elif pane.index != source:
                pane.inputs.refresh()
            pane.estimates.refresh()
            pane.status.refresh()
            pane.results.refresh()
        self._update_locks()

    def _update_locks(self) -> None:
        state = self.model.inputs
        for pane in self.panes:
            for editor in pane.editors:
                if editor.key == "use":
                    editor.set_locked(self.model.busy)
                    continue
                if editor.group == "outputs":
                    kind = OutputKind[editor.key]
                    supported = state.supports(kind)
                    requested = kind in state.requested
                    editor.enabled = [
                        kind is not OutputKind.TOTAL_YIELD and (supported or requested),
                        *(supported and requested for _ in editor.widgets[1:]),
                    ]
                field_key = "n_e" if (editor.group, editor.key) == ("beam", "bunch_charge") else editor.key
                engines = [state.engines[name] for name in state.selected]
                if editor.group.startswith("engine:"):
                    name = editor.group.split(":", 1)[1]
                    engines = [state.engines[name]] if name in state.selected else []
                locked = self.model.locked and any(
                    engine.recompute_costs.get(field_key, RecomputeCost.FULL_RERUN)
                    is RecomputeCost.FULL_RERUN for engine in engines
                )
                editor.set_locked(locked)

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
            for pane in self.panes:
                pane.status.refresh()
                pane.results.refresh()

    def release(self) -> None:
        self.model.locked = False
        self._update_locks()
        for pane in self.panes:
            pane.status.refresh()

    async def _tick(self) -> None:
        if self.client.is_deleted:
            return
        status_state = (self.model.busy, self.model.stale, self.model.error,
                        tuple(self.model.statuses.items()), tuple(self.model.inputs.errors.items()))
        if status_state != self.last_status_state:
            self.last_status_state = status_state
            for pane in self.panes:
                pane.status.refresh()
        result_state = (id(self.model.results), self.model.stale)
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
        request = self.model.completed_request
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
        self.inputs = ui.refreshable(self._inputs)
        self.results = ui.refreshable(self._results)
        self.status = ui.refreshable(self._status)
        self.estimates = ui.refreshable(self._estimates)
        self.geometry = ui.refreshable(self._geometry)
        with ui.column().classes("gf-pane w-full gap-0"):
            with ui.tabs().classes("w-full") as tabs:
                ui.tab("inputs", label="Inputs", icon="tune")
                ui.tab("results", label="Results", icon="show_chart")
            with ui.tab_panels(tabs, value=page.active_tabs[index],
                               on_change=lambda e: page.active_tabs.__setitem__(index, e.value),
                               animated=False).classes("w-full"):
                with ui.tab_panel("inputs"):
                    self.inputs()
                with ui.tab_panel("results"):
                    self.results()

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
                ui.label("Calculation engines").classes("gf-section-title")
                self.editors += render_engines(model.inputs, change)
                self.status()

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
            if model.busy or model.inputs.errors or not model.inputs.selected:
                button.disable()
            release = ui.button("Release inputs", on_click=self.page.release).props("outline")
            if not model.locked or model.busy:
                release.disable()
            if model.busy:
                ui.spinner(size="sm")
            ui.label("Calculating…" if model.busy else "Results outdated — Calculate to update"
                     if model.stale else "Inputs locked after calculation" if model.locked
                     else "Ready").classes("gf-status")
        for name, status in model.statuses.items():
            ui.label(f"{name}: {status}").classes("text-caption")
        if model.error:
            ui.label(model.error).classes("gf-error")
        for error in dict.fromkeys(model.inputs.errors.values()):
            ui.label(error).classes("gf-error")
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
        requested = tuple(model.inputs.requested)
        if model.completed_request is not None:
            requested = tuple(dict.fromkeys(
                [output.kind for output in model.completed_request.target.outputs] + list(requested)))
        if model.busy:
            ui.label("Calculation in progress; completed results appear below.").classes("gf-status")
        if model.error:
            ui.label(model.error).classes("gf-error")
        if model.completed_request is not None:
            ui.button("Download input snapshot", icon="download", on_click=self.page.download_snapshot).props("outline")
        render_results(model.results, requested, stale=model.stale,
                       view_state=self.page.view_states[self.index])


def index() -> None:
    ui.colors(primary="#256782", secondary="#51778e", accent="#a67635")
    ui.add_css(_CSS)
    BrowserWorkspace()


def start(*, port: int = 8080, show: bool = True) -> None:
    if ui is None:
        raise ModuleNotFoundError("Install the GUI extra to launch NiceGUI", name="nicegui")
    ui.page("/")(index)
    ui.run(host="127.0.0.1", port=port, show=show, reload=False, title="GammaForge")
