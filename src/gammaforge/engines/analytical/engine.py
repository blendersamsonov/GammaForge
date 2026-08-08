"""`AnalyticalEngine`: the thin `Engine` wrapper for the closed-form estimates (§4.3).

Unlike xigma there is no `Collision`-style stateful facade — every quantity here is
cheap (`O(1)`/`O(n_quad)`), so nothing needs memoizing across queries within one `run()`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ...io.interaction import InteractionParameters
from ...io.laser import fit_gaussian_paraxial
from ...io.results import Axis, PhasespaceSlice, Results
from ...io.schema import Parameters
from ...io.target import OutputKind, OutputRequest, auto_ranges, slice_axis_values
from ..base import RecomputeCost
from .formulas import angle_integrated_spectrum, estimate_spectrum_width, estimate_yield
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

        total_yield = estimate_yield(beam, metrics, interaction.N_e)
        # Target already owns the two collimation half-angles separately; a single
        # scalar theta_col for the width breakdown is their geometric mean, the same
        # x/y-combining convention `formulas.py` uses for the laser waist (D038).
        theta_col = math.sqrt(target.m("theta_x_col") * target.m("theta_y_col"))
        width = estimate_spectrum_width(beam, metrics, theta_col)

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
                request, ranges[request.kind], beam, total_yield, photon_energy, n_quad
            )

        return Results(
            photon_slices=slices,
            model_specific={
                "spectrum_width_fwhm": width,
                "a0_peak": metrics.a0_peak(),
                "n_photons": metrics.n_photons(),
            },
        )

    def _fill(
        self,
        request: OutputRequest,
        ranges: dict[Axis, tuple[float, float]],
        beam,
        total_yield: float,
        photon_energy: float,
        n_quad: int,
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
            raw = angle_integrated_spectrum(beam.gamma0(), beam.sigma_gamma(), 1.0, s, n_quad)
            raw_dN_dE = raw / (4.0 * photon_energy)
            raw_integral = float(np.trapezoid(raw_dN_dE, values[Axis.ENERGY]))
            dN_dE = raw_dN_dE * (total_yield / raw_integral) if raw_integral > 0 else raw_dN_dE
            return PhasespaceSlice(axes=values, distr=dN_dE)

        raise AssertionError(f"AnalyticalEngine._fill: {kind} is in SUPPORTED_OUTPUTS but has no branch")
