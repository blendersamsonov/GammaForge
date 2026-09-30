"""DeltaEngine: validation reference as a GUI-accessible engine.

Runs xigma's Stage 0 (trajectory integration) then delta's brute-force
per-macroparticle resonance binning. Not a production engine — shares
Stage 0 with xigma, so trajectory errors are not independently checked.
Use for validation only.
"""

from __future__ import annotations

import numpy as np

from ...io.interaction import InteractionParameters
from ...io.results import Axis, PhasespaceSlice, Results
from ...io.schema import Parameters
from ...io.target import OutputKind, OutputRequest
from ...validation.references.delta import (
    angle_integrated_spectrum,
    resonance_spectrum,
    DEFAULT_CONE_FACTOR,
)
from ..base import RecomputeCost
from ..xigma.collision import Collision
from ..xigma.schema import default_parameters as xigma_default_parameters


__all__ = ["DeltaEngine"]


# Delta reuses xigma's schema (same numeric knobs for Stage 0)
# but only uses n_steps, threshold, backend from it
DELTA_SPECS = (
    # Stage 0 parameters (from xigma)
    *tuple(s for s in xigma_default_parameters().specs
           if s.key in ("n_steps", "threshold", "backend")),
    # Delta-specific parameters
    # Note: n_angles and cone_factor are delta-specific
)

# For now, just reuse xigma's full schema but delta only uses Stage 0 params
# The GUI will show all xigma params but delta only uses Stage 0 ones


class DeltaEngine:
    """Validation reference engine: xigma Stage 0 + delta resonance binning."""

    name = "delta"
    schema: Parameters = xigma_default_parameters()
    supported_outputs: tuple[OutputKind, ...] = (
        OutputKind.TOTAL_YIELD,
        OutputKind.SPECTRUM,
        OutputKind.TEMPORAL_ENVELOPE,
        OutputKind.SPATIAL_DISTRIBUTION,
        OutputKind.ANGULAR_DISTRIBUTION,
        OutputKind.COLLIMATED_SPECTRUM,
    )
    recompute_costs: dict[str, RecomputeCost] = {
        "n_e": RecomputeCost.QUERY_ONLY,
    }

    def run(self, interaction: InteractionParameters, params: Parameters) -> Results:
        # Get requested outputs from interaction.target.outputs
        requested = {req.kind for req in interaction.target.outputs}
        supported_requests = tuple(req for req in interaction.target.outputs
                                   if req.kind in self.supported_outputs)
        if not supported_requests:
            return Results(photon_slices={}, model_specific={"warnings": ()})

        # Determine ranges for all requested outputs (same as xigma)
        from ...io.target import auto_ranges
        ranges = auto_ranges(interaction.target, interaction.beam, interaction.laser, interaction.bunch)

        # Build diagnostic edges if temporal/spatial outputs are requested
        t_edges = spatial_edges = None
        for request in supported_requests:
            if request.kind is OutputKind.TEMPORAL_ENVELOPE:
                t_edges = np.linspace(*ranges[request.kind][Axis.TIME], request.resolution[0] + 1)
            elif request.kind is OutputKind.SPATIAL_DISTRIBUTION:
                spatial_edges = tuple(np.linspace(*ranges[request.kind][axis], n + 1)
                                      for axis, n in zip((Axis.X, Axis.Y), request.resolution))

        # Build xigma Collision to get Stage 0 TrajectorySamples
        collision = Collision(interaction=interaction, params=params)
        samples = collision.build_overlap(t_edges=t_edges, spatial_edges=spatial_edges)

        # Extract laser geometry for delta
        geom = collision._laser_polarization_geometry()
        theta_xz = geom["theta_xz"]
        theta_yz = geom["theta_yz"]
        psi_pol = geom["psi_pol"]
        ellipticity = geom["ellipticity"]

        # Get target collimation angles (convert to plain floats in rad)
        theta_x_col = float(interaction.target.theta_x_col.to("rad").magnitude)
        theta_y_col = float(interaction.target.theta_y_col.to("rad").magnitude)

        # Determine energy grid from target auto-ranges or use default
        # For delta, we need s_edges (normalized energy s = E / (4 * hbar * omega0))
        # Use the same approach as xigma: auto-range based on beam/laser.
        # SPECTRUM may not be requested, but COLLIMATED_SPECTRUM/ANGULAR_DISTRIBUTION
        # still need the s grid — use whichever requested output carries an energy axis.
        energy_range = None
        for kind in (OutputKind.SPECTRUM, OutputKind.COLLIMATED_SPECTRUM):
            if kind in ranges and Axis.ENERGY in ranges[kind]:
                energy_range = ranges[kind][Axis.ENERGY]
                break
        if energy_range is None:
            # No energy-axis output requested; fall back to a probe request.
            from dataclasses import replace
            from ...io.target import auto_ranges as _auto_ranges
            probe = replace(interaction.target,
                            outputs=(OutputRequest(OutputKind.SPECTRUM, resolution=(64,)),))
            energy_range = _auto_ranges(probe, interaction.beam,
                                        interaction.laser, interaction.bunch)[OutputKind.SPECTRUM][Axis.ENERGY]
        energy_max_erg = energy_range[1]
        photon_energy = interaction.laser.photon_energy()
        s_max = energy_max_erg / (4.0 * photon_energy)
        s_edges = np.linspace(0.0, s_max, 257)  # 256 bins

        # Prepare results container
        photon_slices = {}
        model_specific = {}

        # TOTAL_YIELD: angle-integrated spectrum integral
        if OutputKind.TOTAL_YIELD in requested:
            spec = angle_integrated_spectrum(
                samples, s_edges,
                psi_pol=psi_pol, ellipticity=ellipticity,
                theta_xz=theta_xz, theta_yz=theta_yz,
                n_angles=33, cone_factor=DEFAULT_CONE_FACTOR,
            )
            total = float(np.sum(spec * np.diff(s_edges)))
            photon_slices[OutputKind.TOTAL_YIELD] = PhasespaceSlice(
                axes={}, distr=np.array(total)
            )

        # SPECTRUM: angle-integrated dN/dE (axis in erg, like xigma)
        if OutputKind.SPECTRUM in requested:
            spec = angle_integrated_spectrum(
                samples, s_edges,
                psi_pol=psi_pol, ellipticity=ellipticity,
                theta_xz=theta_xz, theta_yz=theta_yz,
                n_angles=33, cone_factor=DEFAULT_CONE_FACTOR,
            )
            s_centres = 0.5 * (s_edges[:-1] + s_edges[1:])
            # Convert from dN/ds to dN/dE: E = 4*hbar*omega0*s, so dE = 4*hbar*omega0*ds
            energy_centres = s_centres * 4.0 * photon_energy
            photon_slices[OutputKind.SPECTRUM] = PhasespaceSlice(
                axes={Axis.ENERGY: energy_centres},
                distr=spec / (4.0 * photon_energy),
            )

        # TEMPORAL_ENVELOPE / SPATIAL_DISTRIBUTION: from Stage 0 diagnostics
        if OutputKind.TEMPORAL_ENVELOPE in requested or OutputKind.SPATIAL_DISTRIBUTION in requested:
            diagnostics = samples.diagnostics
            if OutputKind.TEMPORAL_ENVELOPE in requested:
                edges = {Axis.TIME: diagnostics.t_edges}
                density = diagnostics.time_envelope
                photon_slices[OutputKind.TEMPORAL_ENVELOPE] = PhasespaceSlice(
                    axes={axis: 0.5 * (edge[:-1] + edge[1:]) for axis, edge in edges.items()},
                    widths={axis: np.diff(edge) for axis, edge in edges.items()}, distr=density)
            if OutputKind.SPATIAL_DISTRIBUTION in requested:
                edges = dict(zip((Axis.X, Axis.Y), diagnostics.spatial_edges))
                density = diagnostics.spatial_envelope
                photon_slices[OutputKind.SPATIAL_DISTRIBUTION] = PhasespaceSlice(
                    axes={axis: 0.5 * (edge[:-1] + edge[1:]) for axis, edge in edges.items()},
                    widths={axis: np.diff(edge) for axis, edge in edges.items()}, distr=density)

        # COLLIMATED_SPECTRUM: delta's resonance_spectrum at multiple angles
        if OutputKind.COLLIMATED_SPECTRUM in requested:
            # Build angular grid within collimation cone
            n_angles = 33
            half_x = theta_x_col
            half_y = theta_y_col
            step_x = 2.0 * half_x / n_angles
            step_y = 2.0 * half_y / n_angles
            offsets_x = -half_x + step_x * (np.arange(n_angles) + 0.5)
            offsets_y = -half_y + step_y * (np.arange(n_angles) + 0.5)

            # Compute 3D histogram: (s, theta_x, theta_y)
            spec_3d = np.zeros((len(s_edges) - 1, n_angles, n_angles), dtype=float)
            for i, tx in enumerate(offsets_x):
                for j, ty in enumerate(offsets_y):
                    spec_1d = resonance_spectrum(
                        samples, s_edges, tx, ty,
                        psi_pol=psi_pol, ellipticity=ellipticity,
                        theta_xz=theta_xz, theta_yz=theta_yz,
                    )
                    spec_3d[:, i, j] = spec_1d

            # Create PhasespaceSlice with 3 axes (energy in erg, like xigma)
            s_centres = 0.5 * (s_edges[:-1] + s_edges[1:])
            energy_centres = s_centres * 4.0 * photon_energy
            photon_slices[OutputKind.COLLIMATED_SPECTRUM] = PhasespaceSlice(
                axes={
                    Axis.ENERGY: energy_centres,
                    Axis.THETA_X: offsets_x,
                    Axis.THETA_Y: offsets_y,
                },
                distr=spec_3d / (4.0 * photon_energy),
            )

        # ANGULAR_DISTRIBUTION: integrate resonance_spectrum over s at each angle
        if OutputKind.ANGULAR_DISTRIBUTION in requested:
            n_angles = 33
            half_x = theta_x_col
            half_y = theta_y_col
            step_x = 2.0 * half_x / n_angles
            step_y = 2.0 * half_y / n_angles
            offsets_x = -half_x + step_x * (np.arange(n_angles) + 0.5)
            offsets_y = -half_y + step_y * (np.arange(n_angles) + 0.5)
            distr_2d = np.zeros((n_angles, n_angles), dtype=float)
            for i, tx in enumerate(offsets_x):
                for j, ty in enumerate(offsets_y):
                    spec_1d = resonance_spectrum(
                        samples, s_edges, tx, ty,
                        psi_pol=psi_pol, ellipticity=ellipticity,
                        theta_xz=theta_xz, theta_yz=theta_yz,
                    )
                    # Integrate over s (trapezoid over bin centres)
                    distr_2d[i, j] = float(np.sum(spec_1d * np.diff(s_edges)))

            photon_slices[OutputKind.ANGULAR_DISTRIBUTION] = PhasespaceSlice(
                axes={
                    Axis.THETA_X: offsets_x,
                    Axis.THETA_Y: offsets_y,
                },
                distr=distr_2d,
            )

        return Results(
            photon_slices=photon_slices,
            model_specific=model_specific,
        )