"""DeltaEngine: validation reference as a GUI-accessible engine.

Runs xigma's Stage 0 (trajectory integration) then delta's brute-force
per-macroparticle resonance binning. Not a production engine — shares
Stage 0 with xigma, so trajectory errors are not independently checked.
Use for validation only (§4.5, GRAND_PLAN.md).
"""

from __future__ import annotations

import numpy as np

from ...io.interaction import InteractionParameters
from ...io.results import Axis, PhasespaceSlice, Results
from ...io.schema import Parameters
from ...io.target import OutputKind
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
        OutputKind.COLLIMATED_SPECTRUM,
        OutputKind.ANGULAR_DISTRIBUTION,
    )
    recompute_costs: dict[str, RecomputeCost] = {
        "n_e": RecomputeCost.QUERY_ONLY,
    }

    def run(self, interaction: InteractionParameters, params: Parameters) -> Results:
        # Build xigma Collision to get Stage 0 TrajectorySamples
        collision = Collision(interaction=interaction, params=params)
        samples = collision.build_overlap()

        # Extract laser geometry for delta
        geom = collision._laser_polarization_geometry()
        theta_xz = geom["theta_xz"]
        theta_yz = geom["theta_yz"]
        psi_pol = geom["psi_pol"]
        ellipticity = geom["ellipticity"]

        # Get target collimation angles
        theta_x_col = interaction.target.theta_x_col
        theta_y_col = interaction.target.theta_y_col

        # Determine energy grid from target auto-ranges or use default
        # For delta, we need s_edges (normalized energy)
        # Use the same approach as xigma: auto-range based on beam/laser
        from ...io.target import auto_ranges
        ranges = auto_ranges(interaction.target, interaction.beam, interaction.laser, interaction.bunch)
        s_max = ranges[OutputKind.SPECTRUM][Axis.ENERGY][1]  # energy_max
        s_edges = np.linspace(0.0, s_max, 257)  # 256 bins

        # Prepare results container
        photon_slices = {}
        model_specific = {}

        # Get requested outputs from interaction.target.outputs
        requested = {req.kind for req in interaction.target.outputs}

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

        # SPECTRUM: angle-integrated dN/ds
        if OutputKind.SPECTRUM in requested:
            spec = angle_integrated_spectrum(
                samples, s_edges,
                psi_pol=psi_pol, ellipticity=ellipticity,
                theta_xz=theta_xz, theta_yz=theta_yz,
                n_angles=33, cone_factor=DEFAULT_CONE_FACTOR,
            )
            s_centres = 0.5 * (s_edges[:-1] + s_edges[1:])
            photon_slices[OutputKind.SPECTRUM] = PhasespaceSlice(
                axes={Axis.ENERGY: s_centres},
                distr=spec,
            )

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

            # For each angle, compute resonance spectrum
            theta_x_grid, theta_y_grid = np.meshgrid(offsets_x, offsets_y, indexing='ij')
            theta_x_flat = theta_x_grid.ravel()
            theta_y_flat = theta_y_grid.ravel()

            # Compute 3D histogram: (s, theta_x, theta_y)
            # We'll build it by summing resonance_spectrum at each angle
            spec_3d = np.zeros((len(s_edges) - 1, n_angles, n_angles), dtype=float)
            for i, tx in enumerate(offsets_x):
                for j, ty in enumerate(offsets_y):
                    spec_1d = resonance_spectrum(
                        samples, s_edges, tx, ty,
                        psi_pol=psi_pol, ellipticity=ellipticity,
                        theta_xz=theta_xz, theta_yz=theta_yz,
                    )
                    spec_3d[:, i, j] = spec_1d

            # Create PhasespaceSlice with 3 axes
            s_centres = 0.5 * (s_edges[:-1] + s_edges[1:])
            photon_slices[OutputKind.COLLIMATED_SPECTRUM] = PhasespaceSlice(
                axes={
                    Axis.ENERGY: s_centres,
                    Axis.THETA_X: offsets_x,
                    Axis.THETA_Y: offsets_y,
                },
                distr=spec_3d,
            )

        # ANGULAR_DISTRIBUTION: at a specific energy (use peak or middle)
        if OutputKind.ANGULAR_DISTRIBUTION in requested:
            # Use middle energy bin
            s_mid = s_edges[len(s_edges) // 2]
            s_edges_fine = np.array([s_mid - 1e-6, s_mid + 1e-6])
            spec_3d = np.zeros((1, n_angles, n_angles), dtype=float)
            for i, tx in enumerate(offsets_x):
                for j, ty in enumerate(offsets_y):
                    spec_1d = resonance_spectrum(
                        samples, s_edges_fine, tx, ty,
                        psi_pol=psi_pol, ellipticity=ellipticity,
                        theta_xz=theta_xz, theta_yz=theta_yz,
                    )
                    spec_3d[0, i, j] = spec_1d[0] if len(spec_1d) > 0 else 0.0

            photon_slices[OutputKind.ANGULAR_DISTRIBUTION] = PhasespaceSlice(
                axes={
                    Axis.ENERGY: np.array([s_mid]),
                    Axis.THETA_X: offsets_x,
                    Axis.THETA_Y: offsets_y,
                },
                distr=spec_3d,
            )

        return Results(
            photon_slices=photon_slices,
            model_specific=model_specific,
        )