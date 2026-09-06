"""NiceGUI result and geometry panes, kept independent of engine implementations."""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..io.bunch import GaussianElectronBeam
from ..io.drawing import plot_geometry
from ..io.formats.hdf5 import save_results
from ..io.laser import LaserField
from ..io.plotting import collimated_projections, export_overlay, export_plot, plot_slice
from ..io.results import Results
from ..io.target import OutputKind

__all__ = ["particle_summary", "render_results", "render_geometry"]

_ENGINE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def _ui():
    try:
        from nicegui import ui
    except ImportError as exc:
        raise ImportError("NiceGUI is required to render GammaForge GUI outputs") from exc
    return ui


def _engine_color(name: str) -> str:
    """A stable display-only color, independent of which outputs an engine returned."""
    return _ENGINE_COLORS[sum(name.encode("utf-8")) % len(_ENGINE_COLORS)]


def _plot(figure, *, height: int = 384) -> None:
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
        export_overlay(slices, path, colors={name: _engine_color(name) for name in slices})
        ui.download(path.read_bytes(), filename=f"gammaforge-overlay{suffix}")
    finally:
        path.unlink(missing_ok=True)


def _download_hdf5(results: Results, engine: str) -> None:
    ui = _ui()
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
        path = Path(handle.name)
    try:
        save_results(results, path)
        ui.download(path.read_bytes(), filename=f"gammaforge-{engine}.h5")
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


def _download_buttons(results: dict[str, Results]) -> None:
    ui = _ui()
    with ui.row():
        for engine, result in results.items():
            ui.button(
                f"Download {engine} HDF5",
                on_click=lambda r=result, n=engine: _download_hdf5(r, n),
            ).props("outline")


def _render_particles(results: dict[str, Results]) -> None:
    ui = _ui()
    if not results:
        ui.label("Requested, but no completed engine returned macroparticles.").classes("text-grey")
        return
    with ui.row().classes("w-full items-stretch"):
        for engine, result in results.items():
            with ui.card().classes("min-w-64"):
                ui.label(engine).classes("text-subtitle1 font-medium")
                for label, value in particle_summary(result):
                    with ui.row().classes("w-full justify-between gap-8"):
                        ui.label(label).classes("text-grey-7")
                        ui.label(value).classes("font-mono")
    ui.label(
        "HDF5 includes result slices and photon macroparticles; final-electron export "
        "remains a Phase 5 format item."
    ).classes("text-caption text-grey-7")
    _download_buttons(results)


def render_results(
    results: dict[str, Results],
    requested: tuple[OutputKind, ...],
    stale: bool = False,
    view_state: dict | None = None,
) -> None:
    """Render results using optional caller-owned view state.

    ``view_state`` is mutated only for display choices: ``output_kind`` (the enum value),
    ``two_d_engine``, ``collimated_engine``, and ``collimated_view``.  It is deliberately
    separate from calculation state, so a refresh cannot make data look current.
    """
    ui = _ui()
    view_state = {} if view_state is None else view_state
    if not results:
        ui.label("Calculate to populate results.").classes("text-grey")
        return
    if stale:
        ui.label("Results are from an earlier input snapshot.").classes("text-warning")
    for engine, result in results.items():
        warnings = result.model_specific.get("warnings", ())
        for warning in warnings if isinstance(warnings, (tuple, list)) else (warnings,):
            ui.label(f"{engine}: {warning}").classes("text-warning")
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
                    particles = {
                        name: result for name, result in results.items()
                        if result.photons is not None or result.electrons is not None
                    }
                    _render_particles(particles)
                    continue
                available = {name: result.photon_slices[kind] for name, result in results.items()
                             if kind in result.photon_slices}
                if not available:
                    ui.label("Requested, but no completed engine returned this output.").classes("text-grey")
                    continue
                if kind in {OutputKind.SPECTRUM, OutputKind.TEMPORAL_ENVELOPE}:
                    _render_overlaid_lines(available)
                elif kind is OutputKind.COLLIMATED_SPECTRUM:
                    _render_collimated(available, view_state)
                elif kind is OutputKind.TOTAL_YIELD:
                    for engine, slice_ in available.items():
                        ui.label(f"{engine}: {slice_.integrate() if slice_.axis_order else float(slice_.distr):.6g} photons")
                else:
                    _render_picker(available, view_state)
                _download_buttons(results)


def _render_overlaid_lines(available):
    ui = _ui()
    # Plotly combines compatible curves; independently toggled traces remain available in its legend.
    first_name, first = next(iter(available.items()))
    figure = plot_slice(first, name=first_name)
    figure.update_traces(line={"color": _engine_color(first_name)})
    for name, slice_ in list(available.items())[1:]:
        other = plot_slice(slice_, name=name)
        for trace in other.data:
            trace.line.color = _engine_color(name)
            figure.add_trace(trace)
    _plot(figure)
    with ui.row():
        ui.button("PNG", on_click=lambda: _download_overlay(available, ".png")).props("outline")
        ui.button("PDF", on_click=lambda: _download_overlay(available, ".pdf")).props("outline")


def _render_picker(available, view_state):
    ui = _ui()
    holder = ui.column().classes("w-full")
    selected = view_state.get("two_d_engine")
    if selected not in available:
        selected = next(iter(available))
    view_state["two_d_engine"] = selected

    def redraw(name):
        view_state["two_d_engine"] = name
        holder.clear()
        with holder:
            _plot(plot_slice(available[name], name=name))
            with ui.row():
                ui.button("PNG", on_click=lambda: _download_plot(available[name], ".png")).props("outline")
                ui.button("PDF", on_click=lambda: _download_plot(available[name], ".pdf")).props("outline")
    ui.select(list(available), value=selected, label="Engine", on_change=lambda e: redraw(e.value))
    redraw(selected)


def _render_collimated(available, view_state):
    ui = _ui()
    holder = ui.column().classes("w-full")
    engine = view_state.get("collimated_engine")
    if engine not in available:
        engine = next(iter(available))
    engine_select = ui.select(list(available), value=engine, label="Engine")
    view_select = ui.select([], label="Collimated view")

    def redraw(engine, view):
        try:
            projections = collimated_projections(available[engine])
        except ValueError as exc:
            holder.clear()
            with holder:
                ui.label(f"This collimated result cannot be projected: {exc}").classes("text-warning")
            return
        view_state["collimated_engine"] = engine
        view_state["collimated_view"] = view
        density_axes = available[engine].axis_order if view.startswith("energy_at_theta_") else None
        holder.clear()
        with holder:
            _plot(plot_slice(projections[view], name=f"{engine}: {view}", density_axes=density_axes))
            with ui.row():
                ui.button("PNG", on_click=lambda: _download_plot(projections[view], ".png", density_axes=density_axes)).props("outline")
                ui.button("PDF", on_click=lambda: _download_plot(projections[view], ".pdf", density_axes=density_axes)).props("outline")
    def select_engine(engine):
        try:
            projections = collimated_projections(available[engine])
        except ValueError:
            redraw(engine, "spectrum")
            return
        view = view_state.get("collimated_view")
        if view not in projections:
            view = "spectrum"
        view_select.options = list(projections)
        view_select.value = view
        view_select.update()
        redraw(engine, view)
    engine_select.on_value_change(lambda e: select_engine(e.value))
    view_select.on_value_change(lambda e: redraw(engine_select.value, e.value))
    select_engine(engine_select.value)


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
            _plot(plot_geometry(beam, laser, three_d=(mode == "3D")), height=300)
    ui.toggle(["2D", "3D"], value=mode, on_change=lambda e: redraw(e.value))
    holder = ui.column().classes("w-full")
    redraw(mode)
