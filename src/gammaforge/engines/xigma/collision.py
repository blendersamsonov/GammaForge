"""The `Collision` facade: the one stateful object in xigma.

Owns one fixed `InteractionParameters` + xigma `Parameters` pair and memoizes what its
stages produce from them — Stage 0's `TrajectorySamples`, Stage 1's `ShapeTable` (at most
once, peak-a0-agnostic), Stage 1.5's retargeted `Table` per requested peak a0 (RES032),
and prepared Stage-2 contexts per observation/geometry setup — so repeated or adaptively
refined queries reuse both intermediate tables and individual spectral samples.
The cached table and prepared contexts implement DER015–DER017/RES090: raw ``ahat`` plus
carrier moments stay observation-independent, while exact incidence and reconstruction
remain query operations.
`XigmaEngine.run()` (`engine.py`) builds one `Collision` per call; notebooks may hold one
across several queries.

**What this does not do.** It does not detect "only field X changed" across *different*
`InteractionParameters` instances. A `Collision` is cheap to reuse, not smart about being
replaced (RES030).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from typing import NamedTuple

import math
import numpy as np
from ...io.interaction import InteractionParameters
from ...io.laser import fit_gaussian_paraxial
from ...io.results import Axis, PhasespaceSlice, Results
from ...io.schema import Parameters
from ...io.target import OutputKind, OutputRequest, auto_ranges, slice_axis_values
from .stages import (
    PreparedQuery,
    ShapeTable,
    Table,
    TrajectorySamples,
    angle_integrated_spectrum,
    bunch_stokes_parameters,
    deposit_shape_table,
    direction_doppler_factor,
    _check_backend,
    integrate_trajectories,
    prepare_query as prepare_stage2_query,
    retarget_ahat,
    stage2_backend,
)


class BunchStokes(NamedTuple):
    """Bunch-integrated Stokes parameters in the smooth laboratory observer basis (DER007)."""

    I: float
    Q: float
    U: float
    V: float
    P: float
    chi: float


__all__ = ["Collision", "SUPPORTED_OUTPUTS", "BunchStokes"]

#: `OutputKind`s this Collision can fill today. `MACROPARTICLE_DUMP` has
#: no photon-macroparticle population to dump (xigma is a tabulated-density engine, not an
#: MC one). It is omitted — `run()` fills only what a
#: request asks for and this set covers, same contract as any engine (P10).
SUPPORTED_OUTPUTS: tuple[OutputKind, ...] = (
    OutputKind.TOTAL_YIELD,
    OutputKind.SPECTRUM,
    OutputKind.TEMPORAL_ENVELOPE,
    OutputKind.SPATIAL_DISTRIBUTION,
    OutputKind.ANGULAR_DISTRIBUTION,
    OutputKind.COLLIMATED_SPECTRUM,
)
_SUPPORTED = frozenset(SUPPORTED_OUTPUTS)


@dataclass(frozen=True)
class Collision:
    """One fixed interaction/parameter snapshot and the stages it produces.

    Public inputs cannot be replaced after memoization begins. The private cache fields
    remain the facade's only mutable state (RES030).
    """

    interaction: InteractionParameters
    params: Parameters

    _samples: TrajectorySamples | None = field(default=None, init=False, repr=False)
    _overlap_backend: str | None = field(default=None, init=False, repr=False)
    _shape_table: ShapeTable | None = field(default=None, init=False, repr=False)
    _tables: dict[float, Table] = field(default_factory=dict, init=False, repr=False)
    _prepared_queries: dict[tuple[object, ...], PreparedQuery] = field(
        default_factory=dict, init=False, repr=False
    )

    def build_overlap(
        self, *, t_edges: np.ndarray | None = None,
        spatial_edges: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TrajectorySamples:
        """Memoize Stage 0; requesting a new diagnostic grid requires another integration."""
        diagnostics = None if self._samples is None else self._samples.diagnostics
        needs_time = t_edges is not None and (
            diagnostics is None or not np.array_equal(t_edges, diagnostics.t_edges))
        needs_space = spatial_edges is not None and (
            len(spatial_edges) != 2 or diagnostics is None or diagnostics.spatial_edges is None
            or any(not np.array_equal(a, b) for a, b in zip(spatial_edges, diagnostics.spatial_edges)))
        if self._samples is None or needs_time or needs_space:
            backend = self._overlap_backend or _check_backend(self.params.get_choice("backend"))
            samples = integrate_trajectories(
                self.interaction.bunch,
                self.interaction.laser,
                self.interaction.N_e,
                n_steps=self.params.get_int("n_steps"),
                backend=backend,
                threshold=self.params.get_float("threshold"),
                t_edges=t_edges,
                spatial_edges=spatial_edges,
            )
            for values in (
                samples.gamma,
                samples.theta_x,
                samples.theta_y,
                samples.a0_shape,
                samples.luminosity,
                samples.chirp_mean,
                samples.var_a_shape,
                samples.var_chirp,
                samples.cov_a_chirp_shape,
            ):
                values.setflags(write=False)
            if samples.diagnostics is not None:
                diagnostic = samples.diagnostics
                arrays = (diagnostic.t_edges, diagnostic.time_envelope, diagnostic.spatial_envelope,
                          *(diagnostic.spatial_edges or ()))
                for values in arrays:
                    if values is not None:
                        values.setflags(write=False)
            object.__setattr__(self, "_samples", samples)
            object.__setattr__(self, "_overlap_backend", backend)
        return self._samples

    def _shape(self) -> ShapeTable:
        """Stage 1, memoized: peak-a0-agnostic, so this runs at most once per `Collision`
        regardless of how many distinct peak a0 values `_table()` is asked for."""
        if self._shape_table is None:
            object.__setattr__(self, "_shape_table", deposit_shape_table(
                self.build_overlap(),
                n_bins=(
                    self.params.get_int("n_bins_gamma"),
                    self.params.get_int("n_bins_theta_x"),
                    self.params.get_int("n_bins_theta_y"),
                    self.params.get_int("n_bins_a0_shape"),
                    self.params.get_int("n_bins_chirp"),
                ),
                scheme=self.params.get_choice("scheme"),
                backend=self._overlap_backend,
            ))
        return self._shape_table

    def _table(self, intensity_peak: float | None = None) -> Table:
        """Stage 1.5, memoized per requested peak ``<a^2>`` (``None`` means the pulse's
        own). Cheap regardless of ``n_particles`` — a small regrid, not a re-deposit —
        because the expensive part, Stage 1's shape deposit, runs at most once via
        `_shape` no matter how many pulse strengths are retargeted from it.

        The key is a cycle-averaged **intensity**, not an amplitude, so no polarization
        convention crosses this boundary (RES054)."""
        key = self.build_overlap().intensity_peak if intensity_peak is None else intensity_peak
        if key not in self._tables:
            self._tables[key] = retarget_ahat(
                self._shape(),
                key,
                ahat_min=self.params.get_float("ahat_min"),
                ahat_max=self.params.get_float("ahat_max"),
                n_bins=self.params.get_int("n_bins_ahat"),
                decades=self.params.get_float("ahat_decades"),
            )
        return self._tables[key]

    # -- queries --------------------------------------------------------
    def spectrum(self, s) -> np.ndarray:
        """``dN/ds``, table-free (§4.3-adjacent — this is Stage 0's own closed form)."""
        geom = self._laser_polarization_geometry()
        return angle_integrated_spectrum(self.build_overlap(), s,
                                         theta_xz=geom["theta_xz"], theta_yz=geom["theta_yz"])

    def angular_spectrum(
        self,
        s,
        theta_x,
        theta_y,
        *,
        psi_pol: float = 0.0,
        ellipticity: float = 0.0,
        theta_xz: float = 0.0,
        theta_yz: float = 0.0,
        backend: str | None = None,
        rings: int | None = None,
        subsampling: int | None = None,
        line_model: str | None = None,
    ) -> np.ndarray:
        """Stage 2, at the pulse's own peak a0: ``d3N / (ds dtheta_x dtheta_y)``."""
        model = line_model or self.params.get_choice("line_model")
        return self.prepare_query(
            theta_x,
            theta_y,
            psi_pol=psi_pol, ellipticity=ellipticity,
            theta_xz=theta_xz, theta_yz=theta_yz,
            backend=backend,
            rings=rings if rings is not None else self.params.get_int("sampler_rings"),
            subsampling=subsampling if subsampling is not None else self.params.get_int("sampler_subsampling"),
        ).evaluate(s, line_model=model)

    def prepare_query(
        self,
        theta_x,
        theta_y,
        *,
        psi_pol: float = 0.0,
        ellipticity: float = 0.0,
        theta_xz: float = 0.0,
        theta_yz: float = 0.0,
        backend: str | None = None,
        rings: int | None = None,
        subsampling: int | None = None,
    ) -> PreparedQuery:
        """Return the persistent Stage-2 context for this observation setup."""
        requested_backend = backend or (
            self.params.get_choice("backend") if "backend" in self.params else "cupy"
        )
        selected_backend = stage2_backend(
            requested_backend,
            ellipticity=ellipticity,
            theta_xz=theta_xz,
            theta_yz=theta_yz,
        )
        sampler_rings = rings if rings is not None else self.params.get_int("sampler_rings")
        sampler_subsampling = (
            subsampling
            if subsampling is not None
            else self.params.get_int("sampler_subsampling")
        )
        axis_dtype = np.float32 if selected_backend == "cupy" else float
        tx = np.atleast_1d(np.asarray(theta_x, dtype=axis_dtype))
        ty = np.atleast_1d(np.asarray(theta_y, dtype=axis_dtype))
        table = self._table()
        key = (
            id(table),
            selected_backend,
            tx.shape,
            tx.tobytes(),
            ty.shape,
            ty.tobytes(),
            float(psi_pol),
            float(ellipticity),
            float(theta_xz),
            float(theta_yz),
            sampler_rings,
            sampler_subsampling,
        )
        if key not in self._prepared_queries:
            self._prepared_queries[key] = prepare_stage2_query(
                table,
                tx,
                ty,
                psi_pol=psi_pol,
                ellipticity=ellipticity,
                theta_xz=theta_xz,
                theta_yz=theta_yz,
                backend=selected_backend,
                rings=sampler_rings,
                subsampling=sampler_subsampling,
            )
        return self._prepared_queries[key]

    def spectrum_in_angular_range(
        self,
        theta_x_range,
        theta_y_range,
        s_edges,
        *,
        resolution=(33, 33),
        psi_pol: float = 0.0,
        ellipticity: float = 0.0,
        theta_xz: float = 0.0,
        theta_yz: float = 0.0,
        backend: str | None = None,
        rings: int | None = None,
        subsampling: int | None = None,
        line_model: str | None = None,
    ):
        """The windowed on-demand query (§4.2) — cheap once `build_overlap`/`_table` ran."""
        model = line_model or self.params.get_choice("line_model")
        tx = np.linspace(theta_x_range[0], theta_x_range[1], resolution[0])
        ty = np.linspace(theta_y_range[0], theta_y_range[1], resolution[1])
        s_edges = np.asarray(s_edges, dtype=float)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        cube = self.angular_spectrum(
            s_centers,
            tx,
            ty,
            psi_pol=psi_pol, ellipticity=ellipticity,
            theta_xz=theta_xz, theta_yz=theta_yz,
            backend=backend,
            rings=rings if rings is not None else self.params.get_int("sampler_rings"),
            subsampling=subsampling if subsampling is not None else self.params.get_int("sampler_subsampling"),
            line_model=model,
        )
        dN_ds = np.trapezoid(np.trapezoid(cube, ty, axis=1), tx, axis=0)
        n_photons = float(np.trapezoid(dN_ds, s_centers))
        return cube, dN_ds, n_photons

    def _laser_polarization_geometry(self) -> dict[str, float]:
        """Extract polarization geometry and base photon energy from the laser."""
        laser = self.interaction.laser
        if (
            hasattr(laser, "photon_energy")
            and hasattr(laser, "m")
            and hasattr(laser, "theta_xz")
            and hasattr(laser, "theta_yz")
            and hasattr(laser, "psi_pol")
            and hasattr(laser, "ellipticity")
        ):
            return {
                "photon_energy": float(laser.photon_energy()),
                "theta_xz": float(laser.m("theta_xz")),
                "theta_yz": float(laser.m("theta_yz")),
                "psi_pol": float(laser.m("psi_pol")),
                "ellipticity": float(laser.ellipticity),
            }
        elif (
            hasattr(laser, "photon_energy")
            and hasattr(laser, "polarization_axes")
            and hasattr(laser, "ellipticity")
        ):
            k_hat, _, _ = laser.polarization_axes()
            theta_yz = math.asin(np.clip(k_hat[1], -1.0, 1.0))
            cos_yz = math.cos(theta_yz)
            theta_xz = math.atan2(-k_hat[0], -k_hat[2]) if abs(cos_yz) > 1e-12 else 0.0
            return {
                "photon_energy": float(laser.photon_energy()),
                "theta_xz": float(theta_xz),
                "theta_yz": float(theta_yz),
                "psi_pol": float(getattr(laser, "psi_pol", 0.0)),
                "ellipticity": float(laser.ellipticity),
            }
        else:
            metrics = fit_gaussian_paraxial(self.interaction.laser)
            return {
                "photon_energy": float(metrics.photon_energy()),
                "theta_xz": float(metrics.m("theta_xz")),
                "theta_yz": float(metrics.m("theta_yz")),
                "psi_pol": float(metrics.m("psi_pol")),
                "ellipticity": float(metrics.ellipticity),
            }

    def stokes_parameters(
        self,
        theta_x: float = 0.0,
        theta_y: float = 0.0,
        *,
        psi_pol: float | None = None,
        ellipticity: float | None = None,
        theta_xz: float | None = None,
        theta_yz: float | None = None,
    ) -> BunchStokes:
        """Bunch-integrated Stokes parameters in the smooth laboratory observer basis (DER007).

        Computes (I, Q, U, V, P, chi) in the non-singular laboratory basis (m_x, m_y)
        at observation angle (theta_x, theta_y), summing over macroparticles weighted
        by emission luminosity.
        """
        geom = self._laser_polarization_geometry()
        psi = psi_pol if psi_pol is not None else geom["psi_pol"]
        eps = ellipticity if ellipticity is not None else geom["ellipticity"]
        txz = theta_xz if theta_xz is not None else geom["theta_xz"]
        tyz = theta_yz if theta_yz is not None else geom["theta_yz"]

        res = bunch_stokes_parameters(
            self.build_overlap(),
            theta_x,
            theta_y,
            psi_pol=psi,
            ellipticity=eps,
            theta_xz=txz,
            theta_yz=tyz,
        )
        return BunchStokes(*res)

    # -- Results assembly -------------------------------------------------
    def run(self, requests: tuple[OutputRequest, ...]) -> Results:
        """Fill every requested output this Collision supports; skip the rest (P10)."""
        target = self.interaction.target
        supported_requests = tuple(request for request in requests if request.kind in _SUPPORTED)
        if not supported_requests:
            return Results(photon_slices={}, model_specific={"warnings": ()})
        ranges = auto_ranges(
            replace(target, outputs=supported_requests),
            self.interaction.beam,
            self.interaction.laser,
            self.interaction.bunch,
        )
        t_edges = spatial_edges = None
        for request in supported_requests:
            if request.kind is OutputKind.TEMPORAL_ENVELOPE:
                t_edges = np.linspace(*ranges[request.kind][Axis.TIME], request.resolution[0] + 1)
            elif request.kind is OutputKind.SPATIAL_DISTRIBUTION:
                spatial_edges = tuple(np.linspace(*ranges[request.kind][axis], n + 1)
                                      for axis, n in zip((Axis.X, Axis.Y), request.resolution))
        self.build_overlap(t_edges=t_edges, spatial_edges=spatial_edges)
        geom = self._laser_polarization_geometry()
        photon_energy = geom["photon_energy"]
        theta_xz = geom["theta_xz"]
        theta_yz = geom["theta_yz"]
        psi_pol = geom["psi_pol"]
        ellipticity = geom["ellipticity"]

        # Photon energy with crossing angle factor cos²(α/2) per DER005 §2.2 (RES067)
        cos_alpha = math.cos(theta_xz) * math.cos(theta_yz)
        cos_alpha_half_sq = (1.0 + cos_alpha) * 0.5  # cos²(α/2) = (1 + cos α)/2
        photon_energy *= cos_alpha_half_sq

        # The shared auto range uses the reference laser frequency. A chirped xigma
        # trajectory can radiate above that range; manual energy bounds remain authoritative.
        samples = self.build_overlap()
        active = samples.luminosity > 0.0
        if np.any(active):
            carrier_max = float(np.max(samples.chirp_mean[active]))
            request = next((r for r in supported_requests
                            if r.kind is OutputKind.COLLIMATED_SPECTRUM), None)
            if request is not None and Axis.ENERGY not in (request.manual_ranges or {}):
                current_low, current_high = ranges[request.kind][Axis.ENERGY]
                ranges[request.kind][Axis.ENERGY] = (
                    current_low, max(current_high, carrier_max * current_high)
                )

        slices: dict[OutputKind, PhasespaceSlice] = {}
        if self.build_overlap().n_particles == 0:
            for request in supported_requests:
                if request.kind in (OutputKind.TEMPORAL_ENVELOPE, OutputKind.SPATIAL_DISTRIBUTION):
                    slices[request.kind] = self._fill(
                        request, ranges[request.kind], photon_energy, psi_pol, ellipticity,
                        theta_xz, theta_yz)
                    continue
                values = slice_axis_values(request, ranges[request.kind])
                slices[request.kind] = PhasespaceSlice(
                    axes=values,
                    distr=np.zeros(tuple(value.size for value in values.values())),
                )
        else:
            for request in supported_requests:
                slices[request.kind] = self._fill(
                    request, ranges[request.kind], photon_energy, psi_pol, ellipticity,
                    theta_xz, theta_yz
                )
        warnings = ()
        if OutputKind.SPECTRUM in slices:
            warnings = (
                "SPECTRUM uses xigma's table-free linear-Compton shape and omits the "
                "nonlinear redshift, carrier-rate shift, and finite-line corrections "
                "carried by the tabulated angular kernel.",
            )
        line_model = self.params.get_choice("line_model")
        model_specific: dict[str, object] = {
            "warnings": warnings,
            "doppler": {"convention": "direction", "beta": 1.0},
            "line_model": line_model,
            "chirp_treatment": (
                "trajectory_mean_and_second_moments"
                if line_model == "moment2"
                else "trajectory_mean"
            ),
        }
        model_specific["stage0_backend"] = self._overlap_backend
        if self._shape_table is not None:
            model_specific["stage1_backend"] = self._overlap_backend
        captured = {}
        total = self.build_overlap().total_yield()
        for kind in (OutputKind.TEMPORAL_ENVELOPE, OutputKind.SPATIAL_DISTRIBUTION):
            if kind in slices:
                count = slices[kind].integrate()
                fraction = count / total if total > 0 else 0.0
                captured[kind.value] = {"captured_fraction": fraction,
                                        "outside_fraction": max(0.0, 1.0 - fraction) if total > 0 else 0.0}
        if captured:
            model_specific["stage0_diagnostics"] = captured
            if total == 0 and OutputKind.TEMPORAL_ENVELOPE in slices:
                warnings += ("No photons were emitted; the temporal axis is a display interval.",)
            if any(item["outside_fraction"] > 1e-12 for item in captured.values()):
                warnings += ("Stage-0 diagnostic windows exclude some photons; captured and outside "
                             "fractions are recorded in stage0_diagnostics.",)
            model_specific["warnings"] = warnings
        if self.build_overlap().n_particles and (
            OutputKind.ANGULAR_DISTRIBUTION in slices
            or OutputKind.COLLIMATED_SPECTRUM in slices
        ):
            requested_backend = self.params.get_choice("backend") if "backend" in self.params else "cupy"
            selected_backend = stage2_backend(
                requested_backend,
                ellipticity=ellipticity, theta_xz=theta_xz, theta_yz=theta_yz,
            )
            model_specific["stage2_backend"] = selected_backend
            if selected_backend == "cupy":
                from .spectrum_sampler import PHI_CELLS, PROPOSAL_FLOOR_FRACTION, SAMPLES_TOTAL
                sampler_rings = self.params.get_int("sampler_rings")
                sampler_subsampling = self.params.get_int("sampler_subsampling")
                model_specific["stage2_sampler"] = {
                    "samples_total": SAMPLES_TOTAL,
                    "subsampling": sampler_subsampling,
                    "rings": sampler_rings,
                    "phi_cells": PHI_CELLS,
                    "proposal_floor_fraction": PROPOSAL_FLOOR_FRACTION,
                    "cdf_inversion": "exact_binary_search",
                }
                model_specific["warnings"] = (*warnings, (
                    "CuPy Stage 2 has numerical CPU/GPU agreement checks; "
                    "independent arbitrary-angle scientific acceptance remains open."
                ))
        return Results(photon_slices=slices, model_specific=model_specific)


    def _fill(
        self,
        request: OutputRequest,
        ranges: dict[Axis, tuple[float, float]],
        photon_energy: float,
        psi_pol: float,
        ellipticity: float,
        theta_xz: float,
        theta_yz: float,
    ) -> PhasespaceSlice:
        kind = request.kind
        if kind is OutputKind.TOTAL_YIELD:
            return PhasespaceSlice(axes={}, distr=np.asarray(self.build_overlap().total_yield()))

        if kind is OutputKind.SPECTRUM:
            values = slice_axis_values(request, ranges)
            s = values[Axis.ENERGY] / (4.0 * photon_energy)
            dN_ds = self.spectrum(s)
            return PhasespaceSlice(axes=values, distr=dN_ds / (4.0 * photon_energy))

        if kind in (OutputKind.TEMPORAL_ENVELOPE, OutputKind.SPATIAL_DISTRIBUTION):
            diagnostics = self.build_overlap().diagnostics
            if kind is OutputKind.TEMPORAL_ENVELOPE:
                edges = {Axis.TIME: diagnostics.t_edges}
                density = diagnostics.time_envelope
            else:
                edges = dict(zip((Axis.X, Axis.Y), diagnostics.spatial_edges))
                density = diagnostics.spatial_envelope
            return PhasespaceSlice(
                axes={axis: 0.5 * (edge[:-1] + edge[1:]) for axis, edge in edges.items()},
                widths={axis: np.diff(edge) for axis, edge in edges.items()}, distr=density)

        if kind is OutputKind.ANGULAR_DISTRIBUTION:
            values = slice_axis_values(request, ranges)
            s = self._energy_quadrature_grid()
            cube = self.angular_spectrum(
                s, values[Axis.THETA_X], values[Axis.THETA_Y],
                psi_pol=psi_pol, ellipticity=ellipticity,
                theta_xz=theta_xz, theta_yz=theta_yz
            )
            distr = np.trapezoid(cube, s, axis=-1)
            return PhasespaceSlice(axes=values, distr=distr)

        if kind is OutputKind.COLLIMATED_SPECTRUM:
            values = slice_axis_values(request, ranges)
            s = values[Axis.ENERGY] / (4.0 * photon_energy)
            cube = self.angular_spectrum(
                s, values[Axis.THETA_X], values[Axis.THETA_Y],
                psi_pol=psi_pol, ellipticity=ellipticity,
                theta_xz=theta_xz, theta_yz=theta_yz
            )
            # angular_spectrum_from_table returns (theta_x, theta_y, s); §3.4/`SLICE_AXES`
            # orders this output (energy, theta_x, theta_y).
            distr = np.moveaxis(cube, 2, 0) / (4.0 * photon_energy)
            return PhasespaceSlice(axes=values, distr=distr)

        raise AssertionError(f"Collision._fill: {kind} is in _SUPPORTED but has no branch")

    def _energy_quadrature_grid(self, n: int = 64) -> np.ndarray:
        """An ``s`` grid spanning the populated resonance, for the outputs that integrate
        over energy rather than slicing it (`ANGULAR_DISTRIBUTION`). ``s = D*C*gamma**2``
        is the carrier-corrected linear edge in these units (DER013/DER017), so this needs no
        photon energy to convert anything — unlike `SPECTRUM`'s axis, which is stored in
        erg and does."""
        samples = self.build_overlap()
        geom = self._laser_polarization_geometry()
        doppler = direction_doppler_factor(samples.theta_x, samples.theta_y,
                                           geom["theta_xz"], geom["theta_yz"])
        active = samples.luminosity > 0.0
        if np.any(active):
            edge = float(np.max(
                doppler[active] * samples.chirp_mean[active] * samples.gamma[active] ** 2
            ))
        else:
            edge = float(np.max(doppler * samples.gamma**2))
        if self.params.get_choice("line_model") == "moment2":
            edges = np.linspace(0.0, 1.2 * edge, n + 1)
            return 0.5 * (edges[:-1] + edges[1:])
        return np.linspace(0.0, 1.2 * edge, n)
