"""`AnalyticalEngine`: the thin `Engine` wrapper for the closed-form estimates (§4.3).

Unlike xigma there is no `Collision`-style stateful facade — every quantity here is
cheap (`O(1)`/`O(n_quad)`), so nothing needs memoizing across queries within one `run()`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ...io.interaction import InteractionParameters
from ...io.laser import CYCLE_AVERAGE_FACTOR, fit_gaussian_paraxial
from ...io.results import Axis, PhasespaceSlice, Results
from ...io.schema import Parameters
from ...io.target import OutputKind, OutputRequest, auto_ranges, slice_axis_values
from ..base import RecomputeCost
from .formulas import (
    angle_integrated_spectrum,
    estimate_spectrum_width,
    overlap_mean_a0_sq,
    overlap_yield,
)
from .schema import default_parameters

__all__ = ["AnalyticalEngine"]

#: analytical produces only the 0D total yield and the 1D angle-integrated spectrum
#: (§4.3) — never `COLLIMATED_SPECTRUM` (the 3D (E, θx, θy) slice), which the GUI is
#: expected to overlay this 1D estimate onto rather than receive from this engine.
SUPPORTED_OUTPUTS: tuple[OutputKind, ...] = (OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM)

#: Bunch charge is exactly linear in `N_e` for every engine and is handled entirely at
#: the `io` level (`InteractionParameters.with_charge`/`Results.scaled`) without an
#: engine run at all (§5) — matches `XigmaEngine`'s own declaration for the same reason.
RECOMPUTE_COSTS: dict[str, RecomputeCost] = {
    "n_e": RecomputeCost.QUERY_ONLY,
    # Every quadrature knob re-runs the integrals. That is affordable — the default 1D
    # path is ~2 ms — which is what keeps analytical the one real-time engine (§4.3).
    # `n_quad_u > 1` is the exception: the exact 2D mode costs ~40-800 ms and is a
    # deliberate semi-analytical tier, not something to re-trigger per keystroke (D043).
    "n_quad": RecomputeCost.FULL_RERUN,
    "n_quad_overlap": RecomputeCost.FULL_RERUN,
    "n_quad_u": RecomputeCost.FULL_RERUN,
}


class AnalyticalEngine:
    """The closed-form estimate engine, behind the uniform `Engine` protocol (`base.py`)."""

    name = "analytical"
    schema: Parameters = default_parameters()
    supported_outputs: tuple[OutputKind, ...] = SUPPORTED_OUTPUTS
    recompute_costs: dict[str, RecomputeCost] = RECOMPUTE_COSTS

    def run(self, interaction: InteractionParameters, params: Parameters) -> Results:
        target = interaction.target
        beam = interaction.beam
        metrics = fit_gaussian_paraxial(interaction.laser)
        photon_energy = metrics.photon_energy()
        n_quad = params.get_int("n_quad")

        # The general overlap integral, not `formulas.estimate_yield`'s round-beam closed
        # form: it keeps the per-axis sizes, the Twiss `alpha` (electron waist offset), the
        # astigmatic laser waists and the `psi_focus` rotation that this collision geometry
        # actually has, and it uses this repository's own Rayleigh-range convention
        # (`DECISIONS.md` D039/D040). The closed form stays available as the reduction
        # anchor and port-fidelity pin, but the engine does not ship its approximations.
        n_quad_overlap = params.get_int("n_quad_overlap")
        n_quad_u = params.get_int("n_quad_u")
        total_yield = overlap_yield(beam, metrics, interaction.N_e, n_quad_overlap, n_quad_u)
        # Target already owns the two collimation half-angles separately; a single
        # scalar theta_col for the width breakdown is their geometric mean, the same
        # x/y-combining convention `formulas.py` uses for the laser waist (D038).
        theta_col = math.sqrt(target.m("theta_x_col") * target.m("theta_y_col"))
        # The a0 the bunch actually samples, not the pulse's own peak: electrons arriving
        # off-focus or off-peak scatter at lower intensity, and this weights each by the
        # rate at which it does so. Closes the last part of the foci-displacement growth
        # item for the width, which `estimate_spectrum_width` alone could not (D042).
        mean_a0_sq = overlap_mean_a0_sq(beam, metrics, n_quad_overlap, n_quad_u)
        # The cycle-averaged normalized intensity. `a0` is the peak field magnitude, and
        # `io.laser._a0_from_density` builds it through the *linear*-polarization chain, so
        # the cycle average carries C = 1/2. (Circular would be C = 1 — the same factor by
        # which it holds twice the energy density at fixed a0; that is where `ellipticity`
        # enters, §9.2.) Passing `mean_a0_sq` here instead would double the red-shift.
        # `CYCLE_AVERAGE_FACTOR` rather than a literal, so this engine and xigma's
        # `stages.ahat_from_shape` cannot drift apart (D053) — the disagreement between
        # them is what identified the missing factor in the first place.
        ahat = CYCLE_AVERAGE_FACTOR * mean_a0_sq
        width = estimate_spectrum_width(beam, metrics, theta_col, mean_a0_sq)

        # auto_ranges builds a range for every request up front, including kinds this
        # engine will go on to skip — and its TEMPORAL_ENVELOPE branch requires a bunch
        # (raises without one). Filtering to what this engine actually supports before
        # calling it, rather than after, avoids both the crash and passing the bunch just
        # to satisfy a branch never taken (which would cost O(n_particles) for a range
        # this engine discards, defeating the whole point of being bunch-independent).
        supported_requests = tuple(r for r in target.outputs if r.kind in SUPPORTED_OUTPUTS)
        ranges = auto_ranges(replace(target, outputs=supported_requests), beam, interaction.laser)
        slices: dict[OutputKind, PhasespaceSlice] = {}
        for request in supported_requests:
            slices[request.kind] = self._fill(
                request, ranges[request.kind], beam, total_yield, photon_energy, n_quad, ahat
            )

        return Results(
            photon_slices=slices,
            model_specific={
                "spectrum_width_fwhm": width,
                "a0_peak": metrics.a0_peak(),
                "mean_a0_sq": mean_a0_sq,
                "ahat": ahat,
                "compton_edge_energy": 4.0 * beam.gamma0() ** 2 * photon_energy / (1.0 + ahat),
                "n_photons": metrics.n_photons(),
                "warnings": self._geometry_warnings(metrics, slices),
            },
        )

    @staticmethod
    def _geometry_warnings(metrics, slices) -> tuple[str, ...]:
        """Where this engine's answer is only partly covered by its own derivation.

        `overlap_yield` handles a crossing angle exactly, so `TOTAL_YIELD` is right. The
        emitted *spectrum* is a different question — `GRAND_PLAN.md` §9.3's open item is
        the polarization structure of the emission kernel — so `SPECTRUM`'s **shape** is
        still head-on while its integral is correct. That combination looks more right than
        it is, which is exactly why it is reported rather than left to
        `io.laser.validate()`: nothing forces a caller to run that, and this engine is the
        one that knows both quantities were just mixed. Returned as strings on `Results`,
        following this repo's convention that validation reports rather than raises
        (`io.laser.validate`/`io.bunch.validate` both return `list[str]`).
        """
        if metrics.m("theta_xz") == 0.0 and metrics.m("theta_yz") == 0.0:
            return ()
        if OutputKind.SPECTRUM not in slices:
            return ()
        theta = math.hypot(metrics.m("theta_xz"), metrics.m("theta_yz"))
        shift = 1.0 - math.cos(theta / 2.0) ** 2
        return (
            f"SPECTRUM was computed with a {theta * 1e3:.1f} mrad crossing angle: its integral "
            "(the total yield) accounts for the crossing geometry exactly, but its shape is "
            "still the head-on kinematics. The magnitude is quoted so this is actionable "
            f"rather than alarming — the photon energy scale is off by about {shift:.2e} "
            "relative, since it enters as cos^2(theta/2), while the yield changed by far "
            "more. GRAND_PLAN.md §9.3's emission-kernel derivation is what would close it.",
        )

    def _fill(
        self,
        request: OutputRequest,
        ranges: dict[Axis, tuple[float, float]],
        beam,
        total_yield: float,
        photon_energy: float,
        n_quad: int,
        ahat: float,
    ) -> PhasespaceSlice:
        kind = request.kind
        if kind is OutputKind.TOTAL_YIELD:
            return PhasespaceSlice(axes={}, distr=np.asarray(total_yield))

        if kind is OutputKind.SPECTRUM:
            values = slice_axis_values(request, ranges)
            s = values[Axis.ENERGY] / (4.0 * photon_energy)
            # angle_integrated_spectrum's raw shape (at N_e=1) integrates to "one
            # scattering attempt per electron," not a photon count. SPECTRUM is defined
            # as total_yield times that shape's normalized density — not two
            # independently-estimated quantities reconciled after the fact — so that
            # PhasespaceSlice.integrate() reproduces total_yield exactly (§7), matching
            # against the grid's own discrete integral rather than the analytic value of
            # 1 (DECISIONS.md D036).
            raw = angle_integrated_spectrum(beam.gamma0(), beam.sigma_gamma(), 1.0, s, n_quad, ahat)
            raw_dN_dE = raw / (4.0 * photon_energy)
            raw_integral = float(np.trapezoid(raw_dN_dE, values[Axis.ENERGY]))
            dN_dE = raw_dN_dE * (total_yield / raw_integral) if raw_integral > 0 else raw_dN_dE
            return PhasespaceSlice(axes=values, distr=dN_dE)

        raise AssertionError(f"AnalyticalEngine._fill: {kind} is in SUPPORTED_OUTPUTS but has no branch")
