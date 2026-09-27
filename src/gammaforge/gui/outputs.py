"""NiceGUI result and geometry panes, kept independent of engine implementations."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ..io.bunch import GaussianElectronBeam
from ..io.drawing import plot_geometry
from ..io.formats.hdf5 import save_results
from ..io.laser import LaserField
from ..io.plotting import (
    _density_in_display_units,
    _density_label,
    _LABELS,
    collimated_projections,
    display_unit,
    display_values,
    export_overlay,
    export_plot,
    plot_slice,
)
from ..io.results import PhasespaceSlice, Results
from ..io.target import OutputKind

if TYPE_CHECKING:
    from .run import Run

__all__ = ["particle_summary", "render_results", "render_geometry"]

_RUN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


def _ui():
    try:
        from nicegui import ui
    except ImportError as exc:
        raise ImportError("NiceGUI is required to render GammaForge GUI outputs") from exc
    return ui


def _run_color(index: int) -> str:
    """A stable display-only color for a run by index."""
    return _RUN_COLORS[index % len(_RUN_COLORS)]


def _normalize_to_peak(values):
    """Return display values scaled to unit peak, preserving an all-zero curve."""
    peak = float(values.max())
    return values / peak if peak > 0.0 else values


def _collimated_yield_rows(
    available: dict[str, tuple[int, PhasespaceSlice]],
) -> tuple[tuple[str, float], ...]:
    """Run names and photons inside each collimated spectrum's target acceptance."""
    return tuple((name, slice_.integrate()) for name, (_, slice_) in available.items())


def _plot(figure, *, height: int = 500) -> None:
    """Embed a split-pane-safe Plotly chart without NiceGUI's hidden-pane resize bug."""
    ui = _ui()
    options = figure.to_plotly_json()
    options["config"] = {**options.get("config", {}), "responsive": False}
    with ui.element("div").classes("w-full").style(f"height: {height}px"):
        chart = ui.plotly(options).classes("w-full h-full")

        def relayout(event) -> None:
            args = event.args
            if not isinstance(args, dict):
                return
            width, observed_height = args.get("width", 0), args.get("height", 0)
            if width > 0 and observed_height > 0 and not chart.is_deleted:
                chart.run_plot_method("relayout", {"width": width, "height": observed_height})

        ui.element("q-resize-observer").props("debounce=100").on("resize", relayout)


def _download_plot(slice_, suffix: str, *, density_axes=None) -> None:
    ui = _ui()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
    try:
        export_plot(slice_, path, density_axes=density_axes)
        ui.download(path.read_bytes(), filename=f"gammaforge-plot{suffix}")
    finally:
        path.unlink(missing_ok=True)


def _download_overlay(slices, suffix: str) -> None:
    ui = _ui()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
    try:
        colors = {name: _run_color(i) for i, name in enumerate(slices)}
        export_overlay(slices, path, colors=colors)
        ui.download(path.read_bytes(), filename=f"gammaforge-overlay{suffix}")
    finally:
        path.unlink(missing_ok=True)


def _download_hdf5(results: Results, name: str) -> None:
    ui = _ui()
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
        path = Path(handle.name)
    try:
        save_results(results, path)
        ui.download(path.read_bytes(), filename=f"gammaforge-{name}.h5")
    finally:
        path.unlink(missing_ok=True)


def particle_summary(result: Results) -> tuple[tuple[str, str], ...]:
    """Return engine-agnostic display rows for an MC result."""
    rows: list[tuple[str, str]] = []
    if result.photons is not None:
        rows.extend((
            ("Photon macroparticles", f"{result.photons.n_macroparticles:,}"),
            ("Weighted photons", f"{float(result.photons.weight.sum()):.6g}"),
        ))
    electron_count = getattr(result.electrons, "n_particles", None)
    if electron_count is not None:
        rows.append(("Final electrons", f"{int(electron_count):,}"))
    for key, value in result.model_specific.items():
        if key == "warnings" or not isinstance(value, (bool, int, float, str)):
            continue
        label = key.replace("_", " ").title()
        rows.append((label, str(value) if isinstance(value, (bool, str)) else f"{value:,}"))
    return tuple(rows)


def _download_buttons(runs: list[Run]) -> None:
    ui = _ui()
    with ui.row():
        for run in runs:
            if run.results is not None:
                ui.button(
                    f"Download {run.name} HDF5",
                    on_click=lambda r=run.results, n=run.name: _download_hdf5(r, n),
                ).props("outline")


def _render_particles(runs: list[Run]) -> None:
    ui = _ui()
    runs_with_particles = [r for r in runs if r.results and (r.results.photons is not None or r.results.electrons is not None)]
    if not runs_with_particles:
        ui.label("Requested, but no completed run returned macroparticles.").classes("text-grey")
        return
    with ui.row().classes("w-full items-stretch"):
        for run in runs_with_particles:
            with ui.card().classes("min-w-64"):
                ui.label(run.name).classes("text-subtitle1 font-medium")
                ui.label(f"Engine: {run.engine_name}").classes("text-caption text-grey-7")
                for label, value in particle_summary(run.results):
                    with ui.row().classes("w-full justify-between gap-8"):
                        ui.label(label).classes("text-grey-7")
                        ui.label(value).classes("font-mono")
    ui.label(
        "HDF5 includes result slices and photon macroparticles; final-electron export "
        "remains a Phase 5 format item."
    ).classes("text-caption text-grey-7")
    _download_buttons(runs)


def render_results(
    runs: list[Run],
    requested: tuple[OutputKind, ...],
    stale: bool = False,
    view_state: dict | None = None,
) -> None:
    """Render results using run history and optional caller-owned view state."""
    ui = _ui()
    view_state = {} if view_state is None else view_state

    if not runs:
        ui.label("Calculate to populate results.").classes("text-grey")
        return

    if stale:
        ui.label("Results are from an earlier input snapshot.").classes("text-warning")

    requested = tuple(dict.fromkeys(requested))
    if not requested:
        ui.label("No outputs were requested.").classes("text-grey")
        return

    selected_kind = view_state.get("output_kind")
    if selected_kind not in {kind.value for kind in requested}:
        selected_kind = requested[0].value
    view_state["output_kind"] = selected_kind

    with ui.tabs(value=selected_kind, on_change=lambda event: view_state.__setitem__("output_kind", event.value)).classes("w-full") as tabs:
        tab_by_kind = {kind: ui.tab(kind.value, label=kind.value.replace("_", " ").title()) for kind in requested}
    with ui.tab_panels(tabs, value=selected_kind).classes("w-full"):
        for kind, tab in tab_by_kind.items():
            with ui.tab_panel(tab):
                if kind is OutputKind.MACROPARTICLE_DUMP:
                    _render_particles(runs)
                    continue

                # Collect available runs for this output kind
                available: dict[str, tuple[int, PhasespaceSlice]] = {}
                for run in runs:
                    if run.results and kind in run.results.photon_slices:
                        available[run.name] = (run.id, run.results.photon_slices[kind])

                if not available:
                    ui.label("Requested, but no completed run returned this output.").classes("text-grey")
                    continue

                if kind in {OutputKind.SPECTRUM, OutputKind.TEMPORAL_ENVELOPE}:
                    _render_overlaid_lines(available, view_state)
                elif kind is OutputKind.COLLIMATED_SPECTRUM:
                    _render_collimated(available, view_state)
                elif kind is OutputKind.TOTAL_YIELD:
                    _render_total_yield(available, runs)
                else:
                    _render_picker(available, view_state)
                _download_buttons(runs)


def _render_overlaid_lines(available: dict[str, tuple[int, PhasespaceSlice]], view_state: dict) -> None:
    """Render line plots with run selection checkboxes."""
    ui = _ui()

    all_names = list(available.keys())
    if "selected_runs" not in view_state:
        view_state["selected_runs"] = all_names.copy()
    view_state["selected_runs"] = [n for n in view_state["selected_runs"] if n in available]

    def redraw(selected_names: list[str]) -> None:
        holder.clear()
        with holder:
            if not selected_names:
                ui.label("Select at least one run to plot.").classes("text-grey")
                return

            figure = None
            for name in selected_names:
                if name not in available:
                    continue
                run_id, slice_ = available[name]
                axes = slice_.axis_order
                if not axes or len(axes) != 1:
                    continue

                axis = axes[0]
                x = display_values(axis, slice_.axes[axis])
                y = _density_in_display_units(slice_.distr, axes)
                color = _run_color(all_names.index(name))

                if figure is None:
                    figure = go.Figure(
                        go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color})
                    )
                    figure.update_layout(
                        xaxis_title=f"{_LABELS[axis]} [{display_unit(axis)}]",
                        yaxis_title=_density_label(axes),
                    )
                else:
                    figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color}))

            if figure is None:
                ui.label("No valid spectra found.").classes("text-grey")
            else:
                _plot(figure)

    holder = ui.column().classes("w-full")

    ui.label("Select runs to overlay").classes("text-caption text-grey-7 mb-1")
    with ui.row().classes("w-full flex-wrap gap-2 mb-2"):
        for i, name in enumerate(all_names):
            checked = name in view_state["selected_runs"]
            cb = ui.checkbox(name, value=checked).props("dense")
            def on_toggle(event, n=name):
                if event.value:
                    if n not in view_state["selected_runs"]:
                        view_state["selected_runs"].append(n)
                else:
                    view_state["selected_runs"] = [x for x in view_state["selected_runs"] if x != n]
                redraw(view_state["selected_runs"])
            cb.on_value_change(on_toggle)

    redraw(view_state["selected_runs"])

    with ui.row():
        ui.button("PNG", on_click=lambda: _download_overlay(
            {n: available[n][1] for n in view_state["selected_runs"] if n in available}, ".png"
        )).props("outline")
        ui.button("PDF", on_click=lambda: _download_overlay(
            {n: available[n][1] for n in view_state["selected_runs"] if n in available}, ".pdf"
        )).props("outline")


def _render_picker(available: dict[str, tuple[int, PhasespaceSlice]], view_state: dict) -> None:
    """Render 2D plots with run selection checkboxes."""
    ui = _ui()
    holder = ui.column().classes("w-full")

    all_names = list(available.keys())

    if "picker_selected_runs" not in view_state:
        view_state["picker_selected_runs"] = all_names.copy()
    view_state["picker_selected_runs"] = [n for n in view_state["picker_selected_runs"] if n in available]

    def redraw(selected_names: list[str]) -> None:
        holder.clear()
        with holder:
            if not selected_names:
                ui.label("Select at least one run to view.").classes("text-grey")
                return

            run_selector = ui.select(selected_names, value=selected_names[0], label="Run to display")

            def update_plot(name):
                plot_holder.clear()
                with plot_holder:
                    if name not in available:
                        ui.label("Run not available.").classes("text-grey")
                        return
                    run_id, slice_ = available[name]
                    _plot(plot_slice(slice_, name=name))
                    with ui.row():
                        ui.button("PNG", on_click=lambda: _download_plot(slice_, ".png")).props("outline")
                        ui.button("PDF", on_click=lambda: _download_plot(slice_, ".pdf")).props("outline")

            run_selector.on_value_change(lambda e: update_plot(e.value))
            plot_holder = ui.column().classes("w-full")
            update_plot(selected_names[0])

    holder_checkboxes = ui.column().classes("w-full")

    ui.label("Select runs to include").classes("text-caption text-grey-7 mb-1")
    with holder_checkboxes:
        with ui.row().classes("w-full flex-wrap gap-2 mb-2"):
            for i, name in enumerate(all_names):
                checked = name in view_state["picker_selected_runs"]
                cb = ui.checkbox(name, value=checked).props("dense")
                def on_toggle(event, n=name):
                    if event.value:
                        if n not in view_state["picker_selected_runs"]:
                            view_state["picker_selected_runs"].append(n)
                    else:
                        view_state["picker_selected_runs"] = [x for x in view_state["picker_selected_runs"] if x != n]
                    redraw(view_state["picker_selected_runs"])
                cb.on_value_change(on_toggle)

    redraw(view_state["picker_selected_runs"])


def _render_collimated(available: dict[str, tuple[int, PhasespaceSlice]], view_state: dict) -> None:
    """Render collimated results with run selection checkboxes and view tabs."""
    ui = _ui()

    all_names = list(available.keys())

    with ui.card().classes("w-full"):
        with ui.column().classes("w-full gap-2"):
            ui.label("Collimated Yield by Run").classes("text-subtitle1 font-medium")
            ui.label("Number of photons inside the requested target acceptance.").classes(
                "text-caption text-grey-7"
            )
            with ui.row().classes("w-full items-center gap-8 text-caption text-grey-7 mb-1"):
                ui.label("Run").classes("w-32")
                ui.label("Photons on target").classes("w-40")
            for name, yield_value in _collimated_yield_rows(available):
                with ui.row().classes("w-full items-center gap-8"):
                    ui.label(name).classes("w-32 font-medium")
                    ui.label(f"{yield_value:.6g}").classes("w-40 font-mono")

    if "collimated_selected_runs" not in view_state:
        view_state["collimated_selected_runs"] = all_names.copy()
    view_state["collimated_selected_runs"] = [n for n in view_state["collimated_selected_runs"] if n in available]
    view_state["collimated_normalize_spectrum"] = bool(
        view_state.get("collimated_normalize_spectrum", False)
    )

    holder = ui.column().classes("w-full")

    ONE_D_VIEWS = {"spectrum", "energy_at_theta_x_zero", "energy_at_theta_y_zero"}

    def redraw(selected_names: list[str]) -> None:
        holder.clear()
        with holder:
            if not selected_names:
                ui.label("Select at least one run to view.").classes("text-grey")
                return

            valid_runs = {}
            for name in selected_names:
                if name in available:
                    run_id, slice_ = available[name]
                    try:
                        projections = collimated_projections(slice_)
                        valid_runs[name] = (run_id, projections)
                    except ValueError:
                        pass

            if not valid_runs:
                ui.label("No selected runs have valid collimated spectrum data.").classes("text-grey")
                return

            valid_names = list(valid_runs.keys())

            all_views = set()
            for name, (_, projections) in valid_runs.items():
                all_views.update(projections.keys())
            ordered_views = [v for v in ("energy_theta_x", "energy_theta_y", "energy_at_theta_x_zero", "energy_at_theta_y_zero", "spectrum") if v in all_views]

            current_view = view_state.get("collimated_view", "spectrum")
            if current_view not in ordered_views:
                current_view = ordered_views[0] if ordered_views else "spectrum"
            view_state["collimated_view"] = current_view

            def update_view(view):
                view_state["collimated_view"] = view
                plot_holder.clear()
                with plot_holder:
                    if view in ONE_D_VIEWS:
                        figure = None
                        for name in valid_names:
                            run_id, projections = valid_runs[name]
                            if view not in projections:
                                continue
                            slice_ = projections[view]
                            axes = slice_.axis_order
                            if not axes or len(axes) != 1:
                                continue

                            axis = axes[0]
                            x = display_values(axis, slice_.axes[axis])
                            y = _density_in_display_units(slice_.distr, axes)
                            normalize = (
                                view == "spectrum"
                                and view_state["collimated_normalize_spectrum"]
                            )
                            if normalize:
                                y = _normalize_to_peak(y)
                            color = _run_color(all_names.index(name))

                            if figure is None:
                                figure = go.Figure(
                                    go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color})
                                )
                                figure.update_layout(
                                    xaxis_title=f"{_LABELS[axis]} [{display_unit(axis)}]",
                                    yaxis_title=(
                                        "Normalized spectral density [a.u.]"
                                        if normalize else _density_label(axes)
                                    ),
                                    title=view.replace("_", " ").title(),
                                )
                            else:
                                figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color}))

                        if figure is None:
                            ui.label("No valid 1D data found.").classes("text-grey")
                        else:
                            _plot(figure)
                    else:
                        run_selector = ui.select(valid_names, value=valid_names[0], label="Run to display")

                        def update_2d_plot(name, v=view):
                            plot_2d_holder.clear()
                            with plot_2d_holder:
                                if name not in valid_runs:
                                    ui.label("Run not available.").classes("text-grey")
                                    return
                                _, projections = valid_runs[name]
                                if v not in projections:
                                    ui.label(f"View {v} not available.").classes("text-grey")
                                    return
                                slice_ = projections[v]
                                _plot(plot_slice(slice_, name=f"{name}: {v}"))
                                with ui.row():
                                    ui.button("PNG", on_click=lambda: _download_plot(slice_, ".png")).props("outline")
                                    ui.button("PDF", on_click=lambda: _download_plot(slice_, ".pdf")).props("outline")

                        run_selector.on_value_change(lambda e: update_2d_plot(e.value))
                        plot_2d_holder = ui.column().classes("w-full")
                        update_2d_plot(valid_names[0])

            view_tabs = ui.tabs(value=current_view, on_change=lambda e: update_view(e.value)).classes("w-full")
            with view_tabs:
                for v in ordered_views:
                    ui.tab(v, label=v.replace("_", " ").title())

            plot_holder = ui.column().classes("w-full")
            update_view(current_view)

    holder_checkboxes = ui.column().classes("w-full")

    ui.label("Select runs to include").classes("text-caption text-grey-7 mb-1")
    with holder_checkboxes:
        with ui.row().classes("w-full flex-wrap gap-2 mb-2"):
            for i, name in enumerate(all_names):
                checked = name in view_state["collimated_selected_runs"]
                cb = ui.checkbox(name, value=checked).props("dense")
                def on_toggle(event, n=name):
                    if event.value:
                        if n not in view_state["collimated_selected_runs"]:
                            view_state["collimated_selected_runs"].append(n)
                    else:
                        view_state["collimated_selected_runs"] = [x for x in view_state["collimated_selected_runs"] if x != n]
                    redraw(view_state["collimated_selected_runs"])
                cb.on_value_change(on_toggle)

    normalize_checkbox = ui.checkbox(
        "Normalize spectra to maximum",
        value=view_state["collimated_normalize_spectrum"],
    ).props("dense")

    def on_normalize(event):
        view_state["collimated_normalize_spectrum"] = bool(event.value)
        redraw(view_state["collimated_selected_runs"])

    normalize_checkbox.on_value_change(on_normalize)
    normalize_checkbox.tooltip("Scale each run's angle-integrated spectrum to a peak of 1")

    redraw(view_state["collimated_selected_runs"])


def _render_total_yield(available: dict[str, tuple[int, PhasespaceSlice]], runs: list[Run]) -> None:
    """Render total yield as a table of runs."""
    ui = _ui()
    run_engine = {run.id: run.engine_name for run in runs}
    with ui.card().classes("w-full"):
        with ui.column().classes("w-full gap-2"):
            ui.label("Total Yield by Run").classes("text-subtitle1 font-medium")
            with ui.row().classes("w-full items-center gap-8 text-caption text-grey-7 mb-1"):
                ui.label("Run").classes("w-32")
                ui.label("Engine").classes("w-24")
                ui.label("Yield (photons)").classes("w-40")
            for name, (run_id, slice_) in available.items():
                if slice_ is not None:
                    yield_val = slice_.integrate() if slice_.axis_order else float(slice_.distr)
                    with ui.row().classes("w-full items-center gap-8"):
                        ui.label(name).classes("w-32 font-medium")
                        ui.label(run_engine.get(run_id, "")).classes("w-24")
                        ui.label(f"{yield_val:.6g}").classes("w-40 font-mono")


def render_geometry(beam: GaussianElectronBeam, laser: LaserField, view_state: dict | None = None) -> None:
    """Render a schematic using ``view_state['geometry_mode']`` when provided."""
    ui = _ui()
    view_state = {} if view_state is None else view_state
    mode = view_state.get("geometry_mode", "3D")
    if mode not in {"2D", "3D"}:
        mode = "3D"
    view_state["geometry_mode"] = mode

    def redraw(mode):
        view_state["geometry_mode"] = mode
        holder.clear()
        with holder:
            _plot(plot_geometry(beam, laser, three_d=(mode == "3D")), height=400)
    ui.toggle(["2D", "3D"], value=mode, on_change=lambda e: redraw(e.value))
    holder = ui.column().classes("w-full")
    redraw(mode)
