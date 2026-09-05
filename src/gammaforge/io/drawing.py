"""Schematic beam/laser geometry; this contains no emission physics."""
from __future__ import annotations

import numpy as np

from .bunch import GaussianElectronBeam
from .laser import LaserField, fit_gaussian_paraxial
from .plotting import display_scale
from .results import Axis

__all__ = ["geometry_model", "plot_geometry", "matplotlib_geometry"]


def geometry_model(beam: GaussianElectronBeam, laser: LaserField) -> dict:
    """Return drawing primitives in lab coordinates, using the laser's pinned rotation."""
    laser = fit_gaussian_paraxial(laser)
    k, f1, f2 = laser.focusing_axes()
    _, p1, p2 = laser.polarization_axes()
    scale = max(beam.m("sigma_x"), beam.m("sigma_y"), beam.m("sigma_z"), laser.m("sigma_x"), laser.m("sigma_y"))
    focus = np.array([laser.m("x_off"), laser.m("y_off"), 0.0]) + k * 0.5 * (laser.m("z_fx") + laser.m("z_fy"))
    return {"scale": scale, "bunch_radii": np.array([beam.m("sigma_x"), beam.m("sigma_y"), beam.m("sigma_z")]),
            "laser_origin": focus, "k_hat": k, "focus_axes": (f1, f2), "polarization_axes": (p1, p2),
            "laser_radii": np.array([laser.m("sigma_x"), laser.m("sigma_y")]),
            "ellipticity": laser.ellipticity,
            "foci": (focus + k * (laser.m("z_fx") - .5*(laser.m("z_fx") + laser.m("z_fy"))),
                     focus + k * (laser.m("z_fy") - .5*(laser.m("z_fx") + laser.m("z_fy"))))}


def plot_geometry(beam: GaussianElectronBeam, laser: LaserField, *, three_d: bool = True):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Plotly is needed for interactive geometry") from exc
    m = geometry_model(beam, laser)
    unit_scale = display_scale(Axis.X)
    s = m["scale"] * unit_scale
    origin = m["laser_origin"] * unit_scale
    foci = tuple(focus * unit_scale for focus in m["foci"])
    fig = go.Figure()
    if three_d:
        u, v = np.mgrid[0:2*np.pi:28j, 0:np.pi:16j]
        r = m["bunch_radii"] * unit_scale
        fig.add_surface(x=r[0]*np.cos(u)*np.sin(v), y=r[1]*np.sin(u)*np.sin(v), z=r[2]*np.cos(v),
                        opacity=.25, showscale=False, name="bunch")
        for vector, label in zip(np.eye(3), ("lab x", "lab y", "lab z")):
            q = vector * s
            fig.add_scatter3d(x=[0, q[0]], y=[0, q[1]], z=[0, q[2]], mode="lines", name=label,
                              line={"color": "gray", "dash": "dot"}, showlegend=False)
        end = origin + m["k_hat"] * 3*s
        fig.add_scatter3d(x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]], mode="lines", name="laser direction")
        for axis, label in zip(m["focus_axes"] + m["polarization_axes"], ("focus 1", "focus 2", "polarization", "polarization minor")):
            q = origin + axis * s; fig.add_scatter3d(x=[origin[0],q[0]], y=[origin[1],q[1]], z=[origin[2],q[2]], mode="lines", name=label)
        for index, focus in enumerate(foci):
            fig.add_scatter3d(x=[focus[0]], y=[focus[1]], z=[focus[2]], mode="markers", marker={"color":"gray"},
                              name="astigmatic focus", showlegend=index == 0)
        phase = np.linspace(0, 2*np.pi, 80)
        focus = foci[0]
        radii = m["laser_radii"] * unit_scale
        ellipse = focus[:, None] + m["focus_axes"][0][:, None] * radii[0] * np.cos(phase) + m["focus_axes"][1][:, None] * radii[1] * np.sin(phase)
        fig.add_scatter3d(x=ellipse[0], y=ellipse[1], z=ellipse[2], mode="lines", name="focusing ellipse")
        polarization = origin[:, None] + m["polarization_axes"][0][:, None] * .35*s * np.cos(phase) + m["polarization_axes"][1][:, None] * .35*s * m["ellipticity"] * np.sin(phase)
        fig.add_scatter3d(x=polarization[0], y=polarization[1], z=polarization[2], mode="lines", name="polarization ellipse")
        fig.update_layout(scene={"aspectmode":"data", "xaxis_title":"x [µm]", "yaxis_title":"y [µm]", "zaxis_title":"z [µm]"},
                          margin={"l": 0, "r": 0, "t": 25, "b": 0}, legend={"orientation": "h", "y": -0.12})
    else:
        # x-z projection is intentionally simple: it is a schematic, not a field plot.
        r = m["bunch_radii"] * unit_scale
        t = np.linspace(0, 2*np.pi, 100)
        fig.add_scatter(x=r[0]*np.cos(t), y=r[2]*np.sin(t), fill="toself", name="bunch")
        o = origin
        e = o + m["k_hat"] * 3*s
        fig.add_scatter(x=[o[0],e[0]], y=[o[2],e[2]], mode="lines", name="laser direction")
        fig.add_scatter(x=[0, s], y=[0, 0], mode="lines", name="lab x", line={"dash": "dot"})
        fig.add_scatter(x=[0, 0], y=[0, s], mode="lines", name="lab z", line={"dash": "dot"})
        fig.update_layout(xaxis_title="x [µm]", yaxis_title="z [µm]", yaxis_scaleanchor="x",
                          margin={"l": 0, "r": 0, "t": 25, "b": 0}, legend={"orientation": "h", "y": -0.12})
    return fig


def matplotlib_geometry(beam: GaussianElectronBeam, laser: LaserField, *, three_d: bool = True):
    import matplotlib.pyplot as plt
    m = geometry_model(beam, laser)
    s = m["scale"]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d" if three_d else None)
    if three_d:
        u, v = np.mgrid[0:2*np.pi:28j,0:np.pi:16j]
        r = m["bunch_radii"]
        ax.plot_wireframe(r[0]*np.cos(u)*np.sin(v), r[1]*np.sin(u)*np.sin(v), r[2]*np.cos(v), color="C0", alpha=.35)
        o = m["laser_origin"]
        e = o + m["k_hat"] * 3*s
        ax.plot([o[0],e[0]], [o[1],e[1]], [o[2],e[2]], color="C1")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_zlabel("z [cm]")
    else:
        t = np.linspace(0, 2*np.pi, 100)
        r = m["bunch_radii"]
        ax.plot(r[0]*np.cos(t), r[2]*np.sin(t))
    ax.set_title("Beam–laser geometry (schematic)"); return fig
