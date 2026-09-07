"""The `Collision` facade (GRAND_PLAN.md §4.2): the one stateful object in xigma.

Owns one fixed `InteractionParameters` + xigma `Parameters` pair and memoizes what its
stages produce from them — Stage 0's `TrajectorySamples`, Stage 1's `ShapeTable` (at most
once, peak-a0-agnostic), and Stage 1.5's retargeted `Table` per requested peak a0
(RES032) — so calling `spectrum`/`angular_spectrum`/`spectrum_in_angular_range`
more than once, or asking `run()` for several outputs that all need the same table, does
the expensive work exactly once. `XigmaEngine.run()` (`engine.py`) builds one `Collision`
per call; notebooks may hold one across several queries.

**What this does not do.** It does not detect "only field X changed" across *different*
`InteractionParameters` instances. A `Collision` is cheap to reuse, not smart about being
replaced (RES030).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import math
import numpy as np
from ...io.interaction import InteractionParameters
from ...io.laser import fit_gaussian_paraxial
from ...io.results import Axis, PhasespaceSlice, Results
from ...io.schema import Parameters
from ...io.target import OutputKind, OutputRequest, auto_ranges, slice_axis_values
from .stages import (
    ShapeTable,
    Table,
    TrajectorySamples,
    angle_integrated_spectrum,
    angular_spectrum_from_table,
    deposit_shape_table,
    integrate_trajectories,
    retarget_ahat,
    stage2_backend,
    spectrum_in_angular_range as _spectrum_in_angular_range,
)

__all__ = ["Collision", "SUPPORTED_OUTPUTS"]

#: `OutputKind`s this Collision can fill today. `TEMPORAL_ENVELOPE`/`SPATIAL_DISTRIBUTION`
#: need Stage 0 diagnostics `TrajectorySamples` does not carry (per-step position/time,
#: not just the trajectory-averaged quantities it keeps, §4.2); `MACROPARTICLE_DUMP` has
#: no photon-macroparticle population to dump (xigma is a tabulated-density engine, not an
#: MC one). All three are omitted, not silently approximated — `run()` fills only what a
#: request asks for and this set covers, same contract as any engine (P10).
SUPPORTED_OUTPUTS: tuple[OutputKind, ...] = (
    OutputKind.TOTAL_YIELD,
    OutputKind.SPECTRUM,
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
    _shape_table: ShapeTable | None = field(default=None, init=False, repr=False)
    _tables: dict[float, Table] = field(default_factory=dict, init=False, repr=False)

    def build_overlap(self) -> TrajectorySamples:
        """Stage 0, memoized: every other method funnels through this."""
        if self._samples is None:
            samples = integrate_trajectories(
                self.interaction.bunch,
                self.interaction.laser,
                self.interaction.N_e,
                n_steps=self.params.get_int("n_steps"),
                threshold=self.params.get_float("threshold"),
            )
            for values in (samples.gamma, samples.theta_x, samples.theta_y, samples.a0_shape, samples.luminosity):
                values.setflags(write=False)
            object.__setattr__(self, "_samples", samples)
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
                ),
                scheme=self.params.get_choice("scheme"),
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
        return angle_integrated_spectrum(self.build_overlap(), s)

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
    ) -> np.ndarray:
        """Stage 2, at the pulse's own peak a0: ``d3N / (ds dtheta_x dtheta_y)``."""
        b = backend or (self.params.get_choice("backend") if "backend" in self.params else "numpy")
        return angular_spectrum_from_table(
            self._table(), theta_x, theta_y, s,
            psi_pol=psi_pol, ellipticity=ellipticity,
            theta_xz=theta_xz, theta_yz=theta_yz,
            backend=b,
        )

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
    ):
        """The windowed on-demand query (§4.2) — cheap once `build_overlap`/`_table` ran."""
        b = backend or (self.params.get_choice("backend") if "backend" in self.params else "numpy")
        return _spectrum_in_angular_range(
            self._table(),
            theta_x_range, theta_y_range, s_edges,
            resolution=resolution,
            psi_pol=psi_pol, ellipticity=ellipticity,
            theta_xz=theta_xz, theta_yz=theta_yz,
            backend=b,
        )

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
        laser = self.interaction.laser
        if (
            hasattr(laser, "photon_energy")
            and hasattr(laser, "m")
            and hasattr(laser, "theta_xz")
            and hasattr(laser, "theta_yz")
            and hasattr(laser, "psi_pol")
            and hasattr(laser, "ellipticity")
        ):
            photon_energy = laser.photon_energy()
            theta_xz = laser.m("theta_xz")
            theta_yz = laser.m("theta_yz")
            psi_pol = laser.m("psi_pol")
            ellipticity = float(laser.ellipticity)
        elif (
            hasattr(laser, "photon_energy")
            and hasattr(laser, "polarization_axes")
            and hasattr(laser, "ellipticity")
        ):
            photon_energy = laser.photon_energy()
            k_hat, _, _ = laser.polarization_axes()
            theta_yz = math.asin(np.clip(k_hat[1], -1.0, 1.0))
            cos_yz = math.cos(theta_yz)
            theta_xz = math.atan2(-k_hat[0], -k_hat[2]) if abs(cos_yz) > 1e-12 else 0.0
            psi_pol = float(getattr(laser, "psi_pol", 0.0))
            ellipticity = float(laser.ellipticity)
        else:
            metrics = fit_gaussian_paraxial(self.interaction.laser)
            photon_energy = metrics.photon_energy()
            theta_xz = metrics.m("theta_xz")
            theta_yz = metrics.m("theta_yz")
            psi_pol = metrics.m("psi_pol")
            ellipticity = metrics.ellipticity

        # Photon energy with crossing angle factor cos²(α/2) per DER005 §2.2 (RES067)
        cos_alpha = math.cos(theta_xz) * math.cos(theta_yz)
        cos_alpha_half_sq = (1.0 + cos_alpha) * 0.5  # cos²(α/2) = (1 + cos α)/2
        photon_energy *= cos_alpha_half_sq

        slices: dict[OutputKind, PhasespaceSlice] = {}
        if self.build_overlap().n_particles == 0:
            for request in supported_requests:
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
                "nonlinear redshift carried by the tabulated angular kernel.",
            )
        model_specific: dict[str, object] = {"warnings": warnings}
        if self.build_overlap().n_particles and (
            OutputKind.ANGULAR_DISTRIBUTION in slices
            or OutputKind.COLLIMATED_SPECTRUM in slices
        ):
            requested_backend = self.params.get_choice("backend") if "backend" in self.params else "numpy"
            selected_backend = stage2_backend(
                requested_backend,
                ellipticity=ellipticity, theta_xz=theta_xz, theta_yz=theta_yz,
            )
            model_specific["stage2_backend"] = selected_backend
            if selected_backend == "cupy":
                model_specific["stage2_sampler"] = {
                    "samples_total": 256,
                    "subsampling": 32,
                }
                model_specific["warnings"] = (*warnings, (
                    "CuPy Stage 2 is experimental: use backend='numpy' for validated "
                    "results; GPU-versus-NumPy agreement remains an alpha blocker."
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
        over energy rather than slicing it (`ANGULAR_DISTRIBUTION`). ``s = gamma**2`` is
        already the Compton edge in these units (§9.1's convention), so this needs no
        photon energy to convert anything — unlike `SPECTRUM`'s axis, which is stored in
        erg and does."""
        edge = float(np.max(self.build_overlap().gamma) ** 2)
        return np.linspace(0.0, 1.2 * edge, n)
