"""Result projections and plots.

This module is deliberately the only place where result axes leave CGS.  In
particular, a density is transformed by the reciprocal coordinate scale, so a
plot in eV or mrad still integrates to the same number of photons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .results import Axis, PhasespaceSlice, Results

__all__ = [
    "display_unit", "display_scale", "display_values", "project_slice",
    "collimated_projections", "plot_slice", "plot_collimated", "matplotlib_figure",
    "export_plot", "export_overlay",
]

_DISPLAY_UNITS = {Axis.ENERGY: "eV", Axis.TIME: "fs", Axis.X: "µm", Axis.Y: "µm",
                  Axis.THETA_X: "mrad", Axis.THETA_Y: "mrad"}
_SCALES = {Axis.ENERGY: 1.0 / 1.602176634e-12, Axis.TIME: 1e15,
           Axis.X: 1e4, Axis.Y: 1e4, Axis.THETA_X: 1e3, Axis.THETA_Y: 1e3}
_LABELS = {Axis.ENERGY: "Photon energy", Axis.TIME: "Time", Axis.X: "x", Axis.Y: "y",
           Axis.THETA_X: "θx", Axis.THETA_Y: "θy"}


def display_unit(axis: Axis) -> str:
    return _DISPLAY_UNITS[axis]


def display_scale(axis: Axis) -> float:
    """Display-coordinate units per canonical CGS unit."""
    return _SCALES[axis]


def display_values(axis: Axis, values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * display_scale(axis)


def _density_in_display_units(density: np.ndarray, axes: tuple[Axis, ...]) -> np.ndarray:
    return np.asarray(density, dtype=float) / np.prod([display_scale(a) for a in axes])


def project_slice(slice_: PhasespaceSlice, keep: tuple[Axis, ...]) -> PhasespaceSlice:
    """Marginalize a density, with the slice's trapezoidal quadrature convention.

    ``keep`` also fixes the output axis order.  Plain sums are intentionally never
    used: nonuniform output grids are a supported part of the result contract.
    """
    original = slice_.axis_order
    if len(set(keep)) != len(keep) or not set(keep).issubset(original):
        raise ValueError("project_slice: kept axes must be distinct slice axes")
    values = slice_.distr
    order = list(original)
    for axis in reversed(original):
        if axis not in keep:
            index = order.index(axis)
            grid = slice_.axes[axis]
            if grid.size < 2:
                raise ValueError(f"project_slice: cannot integrate {axis.name} with fewer than 2 samples")
            values = np.trapezoid(values, grid, axis=index)
            order.pop(index)
    if tuple(order) != keep:
        permutation = [order.index(axis) for axis in keep]
        values = np.transpose(values, permutation)
    return PhasespaceSlice({axis: slice_.axes[axis] for axis in keep}, values)


def _zero_slice(slice_: PhasespaceSlice, angle: Axis) -> PhasespaceSlice:
    """Interpolate an angular coordinate at physical zero without inventing a bin."""
    index = slice_.axis_order.index(angle)
    grid = slice_.axes[angle]
    if grid.size < 2:
        raise ValueError(f"zero-angle slice needs at least 2 {angle.name} samples")
    if not grid[0] <= 0.0 <= grid[-1]:
        raise ValueError(f"zero-angle slice requires {angle.name} range to contain zero")
    moved = np.moveaxis(slice_.distr, index, -1)
    flat = moved.reshape(-1, grid.size)
    values = np.array([np.interp(0.0, grid, row) for row in flat]).reshape(moved.shape[:-1])
    axes = {a: v for a, v in slice_.axes.items() if a is not angle}
    return PhasespaceSlice(axes, values)


def collimated_projections(slice_: PhasespaceSlice) -> Mapping[str, PhasespaceSlice]:
    """The four useful views of an ``(E, theta_x, theta_y)`` collimated result."""
    required = (Axis.ENERGY, Axis.THETA_X, Axis.THETA_Y)
    if set(slice_.axis_order) != set(required):
        raise ValueError("collimated_projections requires ENERGY, THETA_X and THETA_Y axes")
    return {
        "energy_at_theta_x_zero": _zero_slice(slice_, Axis.THETA_X),
        "energy_at_theta_y_zero": _zero_slice(slice_, Axis.THETA_Y),
        "energy_theta_x": project_slice(slice_, (Axis.ENERGY, Axis.THETA_X)),
        "energy_theta_y": project_slice(slice_, (Axis.ENERGY, Axis.THETA_Y)),
        "spectrum": project_slice(slice_, (Axis.ENERGY,)),
    }


def _title(slice_: PhasespaceSlice) -> str:
    return " × ".join(_LABELS[a] for a in slice_.axis_order) or "Total yield"


def _density_label(axes: tuple[Axis, ...]) -> str:
    return "Photons" + (" / " + " / ".join(display_unit(axis) for axis in axes) if axes else "")


def plot_slice(
    slice_: PhasespaceSlice,
    *,
    name: str | None = None,
    density_axes: tuple[Axis, ...] | None = None,
):
    """Return an interactive Plotly figure, importing Plotly only on demand."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # keep ``import gammaforge.io`` lightweight
        raise ImportError("Plotly is needed to create interactive plots; install gammaforge GUI dependencies") from exc
    axes = slice_.axis_order
    density_axes = axes if density_axes is None else density_axes
    density = _density_in_display_units(slice_.distr, density_axes)
    title = name or _title(slice_)
    if not axes:
        figure = go.Figure(go.Indicator(mode="number", value=float(density), title={"text": title}))
        figure.update_layout(uirevision=title)
        return figure
    if len(axes) == 1:
        axis = axes[0]
        figure = go.Figure(go.Scatter(x=display_values(axis, slice_.axes[axis]), y=density,
                                      mode="lines", name=name or "result"))
        figure.update_layout(xaxis_title=f"{_LABELS[axis]} [{display_unit(axis)}]",
                             yaxis_title=_density_label(density_axes), title=title, uirevision=title)
        return figure
    if len(axes) == 2:
        x, y = axes
        figure = go.Figure(go.Heatmap(x=display_values(x, slice_.axes[x]), y=display_values(y, slice_.axes[y]),
                                      z=density.T, colorbar_title=_density_label(density_axes)))
        figure.update_layout(xaxis_title=f"{_LABELS[x]} [{display_unit(x)}]",
                             yaxis_title=f"{_LABELS[y]} [{display_unit(y)}]", title=title,
                             uirevision=title)
        return figure
    raise ValueError("plot_slice: plot a 3D result through collimated_projections")


def plot_collimated(slice_: PhasespaceSlice, view: str = "spectrum"):
    projections = collimated_projections(slice_)
    if view not in projections:
        raise ValueError(f"unknown collimated view {view!r}; choose from {tuple(projections)}")
    density_axes = slice_.axis_order if view.startswith("energy_at_theta_") else None
    return plot_slice(projections[view], name=view.replace("_", " "), density_axes=density_axes)


def matplotlib_figure(
    slice_: PhasespaceSlice,
    *,
    name: str | None = None,
    density_axes: tuple[Axis, ...] | None = None,
):
    """Build a headless matplotlib figure from exactly the browser plot data."""
    import matplotlib.pyplot as plt
    axes = slice_.axis_order
    density_axes = axes if density_axes is None else density_axes
    density = _density_in_display_units(slice_.distr, density_axes)
    fig, ax = plt.subplots()
    if not axes:
        ax.text(.5, .5, f"{float(density):.6g} photons", ha="center", va="center")
        ax.set_axis_off()
    elif len(axes) == 1:
        a = axes[0]; ax.plot(display_values(a, slice_.axes[a]), density)
        ax.set_xlabel(f"{_LABELS[a]} [{display_unit(a)}]"); ax.set_ylabel(_density_label(density_axes))
    elif len(axes) == 2:
        x, y = axes
        mesh = ax.pcolormesh(display_values(x, slice_.axes[x]), display_values(y, slice_.axes[y]), density.T,
                             shading="auto")
        fig.colorbar(mesh, ax=ax, label=_density_label(density_axes))
        ax.set_xlabel(f"{_LABELS[x]} [{display_unit(x)}]"); ax.set_ylabel(f"{_LABELS[y]} [{display_unit(y)}]")
    else:
        raise ValueError("matplotlib_figure: plot a 3D result through collimated_projections")
    ax.set_title(name or _title(slice_)); fig.tight_layout()
    return fig


def export_plot(
    slice_: PhasespaceSlice,
    path: str | Path,
    *,
    name: str | None = None,
    density_axes: tuple[Axis, ...] | None = None,
) -> Path:
    path = Path(path)
    if path.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("export_plot path must end in .png or .pdf")
    figure = matplotlib_figure(slice_, name=name, density_axes=density_axes)
    try:
        figure.savefig(path)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
    return path


def export_overlay(slices: Mapping[str, PhasespaceSlice], path: str | Path, *, colors: Mapping[str, str] | None = None) -> Path:
    """Export compatible 1D curves together, matching an engine-overlay browser plot."""
    if not slices:
        raise ValueError("export_overlay needs at least one slice")
    first = next(iter(slices.values()))
    if len(first.axis_order) != 1 or any(slice_.axis_order != first.axis_order for slice_ in slices.values()):
        raise ValueError("export_overlay requires compatible one-dimensional slices")
    import matplotlib.pyplot as plt
    axis = first.axis_order[0]
    fig, ax = plt.subplots()
    for name, slice_ in slices.items():
        ax.plot(display_values(axis, slice_.axes[axis]), _density_in_display_units(slice_.distr, (axis,)),
                label=name, color=None if colors is None else colors.get(name))
    ax.set_xlabel(f"{_LABELS[axis]} [{display_unit(axis)}]")
    ax.set_ylabel(_density_label((axis,)))
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path
