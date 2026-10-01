"""`AnalyticalEngine`: the thin `Engine` wrapper for the closed-form estimates (§4.3).

Unlike xigma there is no `Collision`-style stateful facade — every quantity here is
cheap (`O(1)`/`O(n_quad)`), so nothing needs memoizing across queries within one `run()`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ...io.interaction import InteractionParameters
from ...io.laser import GaussianParaxialLaser
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
from .models import AnalyticalModel, ModelChoice, ModelInputs, ModelSelector
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
    # path is ~2 ms, keeping analytical the one real-time engine (§4.3). `n_quad_u > 1`
    # is the exception: the exact 2D mode costs ~40-800 ms (RES043).
    "n_quad": RecomputeCost.FULL_RERUN,
    "n_quad_overlap": RecomputeCost.FULL_RERUN,
    "n_quad_u": RecomputeCost.FULL_RERUN,
    # Model selection is a handful of comparisons over a short registry, with no integral
    # of its own. A different pin can select a different model, so it cannot be assumed
    # QUERY_ONLY — but it is never more expensive than re-running the model it selects.
    "model_mode": RecomputeCost.FULL_RERUN,
    "model_pin": RecomputeCost.FULL_RERUN,
}

#: The DER001/DER002 tier, which is what this engine computed before the DER019 hierarchy
#: existed. Registered as a model so the planner has something truthful to select and so
#: higher tiers can be added as siblings rather than as branches inside `run`.
#:
#: `exact` is True for `TOTAL_YIELD` in the sense DER019 means it: the general Gaussian
#: overlap is exact for the geometry, including a crossing angle and the flying-focus path
#: (RES039/RES041/RES043). The same model serves `SPECTRUM`, where it is **not** exact — the
#: spectrum is built from one luminosity-weighted mean `ahat` and normalized against the
#: yield, which is a documented approximation. One model, two observables, one shared truth
#: about its assumptions; the per-observable exactness is recorded in the provenance below
#: rather than smuggled into the model's own flag.
_OVERLAP_TIER = AnalyticalModel(
    name="overlap_der001_mean_ahat",
    outputs=(OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM),
    # Structurally universal within this engine's own domain: `run` has already rejected
    # every laser that is not a GaussianParaxialLaser (RES067), and the DER001/DER002 overlap
    # covers crossing angle, offsets, astigmatism and flying focus exactly.
    applies=lambda inputs: True,
    # Always accepted automatically: it is the current, tested behaviour of this engine.
    acceptance=lambda inputs: True,
    fidelity_rank=1,
    cost_rank=0,
    exact=True,
    assumptions=(
        "gaussian_beam_and_paraxial_gaussian_laser",
        "luminosity_weighted_mean_ahat_for_spectrum_shape",
        "spectrum_normalized_to_overlap_total_yield",
    ),
    outer_dimension=1,
    trajectory_quadrature=False,
)


class AnalyticalEngine:
    """The closed-form estimate engine, behind the uniform `Engine` protocol (`base.py`)."""

    name = "analytical"
    schema: Parameters = default_parameters()
    supported_outputs: tuple[OutputKind, ...] = SUPPORTED_OUTPUTS
    recompute_costs: dict[str, RecomputeCost] = RECOMPUTE_COSTS

    #: Single planner instance per engine. `ModelSelector` is immutable after construction,
    #: so sharing one across calls keeps selection allocation-free on the real-time path.
    _selector = ModelSelector((_OVERLAP_TIER,))

    def run(self, interaction: InteractionParameters, params: Parameters) -> Results:
        if not isinstance(interaction.laser, GaussianParaxialLaser):
            raise TypeError(
                f"AnalyticalEngine requires a GaussianParaxialLaser, got {type(interaction.laser).__name__}. "
                "Closed-form overlap integrals assume an astigmatic paraxial Gaussian pulse (RES067)."
            )
        target = interaction.target
        beam = interaction.beam
        metrics = interaction.laser
        photon_energy = metrics.photon_energy()
        n_quad = params.get_int("n_quad")

        # The general overlap integral (`overlap_yield`), not `formulas.estimate_yield`'s
        # round-beam closed form — the latter stays only as a reduction anchor and
        # port-fidelity pin (RES039/RES040).
        n_quad_overlap = params.get_int("n_quad_overlap")
        n_quad_u = params.get_int("n_quad_u")
        total_yield = overlap_yield(beam, metrics, interaction.N_e, n_quad_overlap, n_quad_u)
        # Target owns theta_x_col/theta_y_col separately; theta_col here is their
        # geometric mean, matching `formulas.py`'s x/y-combining convention (RES038).
        theta_col = math.sqrt(target.m("theta_x_col") * target.m("theta_y_col"))
        # The a0 the bunch actually samples (luminosity-weighted), not the pulse's own
        # peak — the width's remaining foci-displacement gap (RES042).
        mean_a0_sq = overlap_mean_a0_sq(beam, metrics, n_quad_overlap, n_quad_u)
        # The cycle-averaged normalized intensity `<a^2>`: `overlap_mean_a0_sq` returns a
        # mean of *peak*-amplitude-squared, so the cycle-average factor is applied here.
        # Read from the laser, not hardcoded, to stay pinned to xigma's own definition
        # (RES053/RES054).
        ahat = metrics.cycle_average_factor() * mean_a0_sq
        width = estimate_spectrum_width(beam, metrics, theta_col, mean_a0_sq)

        # auto_ranges builds a range for every request up front, including kinds this
        # engine will go on to skip — and its TEMPORAL_ENVELOPE branch requires a bunch
        # (raises without one). Filtering to what this engine actually supports before
        # calling it, rather than after, avoids both the crash and passing the bunch just
        # to satisfy a branch never taken (which would cost O(n_particles) for a range
        # this engine discards, defeating the whole point of being bunch-independent).
        supported_requests = tuple(r for r in target.outputs if r.kind in SUPPORTED_OUTPUTS)
        ranges = auto_ranges(replace(target, outputs=supported_requests), beam, interaction.laser)

        # Model selection happens per observable, before anything is filled, so the choice is
        # reported even for a requested kind this engine ultimately produces no slice for.
        # A structurally invalid pin raises here rather than deep inside a formula.
        choices = self._select_models(params, interaction, photon_energy, supported_requests)

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
                "models": {choice.observable: choice.as_metadata() for choice in choices},
            },
        )

    def _select_models(
        self, params: Parameters, interaction: InteractionParameters, photon_energy: float, requests
    ) -> list[ModelChoice]:
        """Ask the planner for a model per requested observable.

        Every observable routes through the planner even though only one tier is registered
        yet. That is the point of doing it now: registering the current behaviour as a model
        before adding new ones is what makes the refactor provably inert, and it means a
        later tier is a new registry entry rather than a rewrite of `run`.
        """
        metrics = interaction.laser
        inputs = ModelInputs(
            beam=interaction.beam,
            laser=metrics,
            photon_energy=photon_energy,
            crossing_angle=math.hypot(metrics.m("theta_xz"), metrics.m("theta_yz")),
            beta_ff=metrics.beta_ff,
        )
        mode = str(params["model_mode"])
        pin = str(params["model_pin"])
        pin = None if pin == "auto" else pin
        # TOTAL_YIELD always participates: it is this engine's own normalization anchor, and
        # selecting a spectrum model without selecting what normalizes it would leave the
        # provenance claiming an independence the arithmetic does not have.
        kinds = tuple(dict.fromkeys(r.kind for r in requests)) or SUPPORTED_OUTPUTS
        return [self._selector.select(inputs, kind, mode=mode, pin=pin) for kind in kinds]

    @staticmethod
    def _geometry_warnings(metrics, slices) -> tuple[str, ...]:
        """Where this engine's answer is only partly covered by its own derivation.

        `overlap_yield` handles a crossing angle exactly, so `TOTAL_YIELD` is right. The
        emitted *spectrum* has a separate emission-kernel treatment, so `SPECTRUM`'s **shape** is
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
            "more. The crossed-spectrum shape requires a separate analytical model.",
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
            # angle_integrated_spectrum's raw shape (at N_e=1) integrates to one
            # scattering attempt per electron, not a photon count. SPECTRUM = total_yield
            # times that shape's normalized density, matched against the grid's own
            # discrete integral so PhasespaceSlice.integrate() reproduces total_yield
            # exactly (RES036).
            raw = angle_integrated_spectrum(beam.gamma0(), beam.sigma_gamma(), 1.0, s, n_quad, ahat)
            raw_dN_dE = raw / (4.0 * photon_energy)
            raw_integral = float(np.trapezoid(raw_dN_dE, values[Axis.ENERGY]))
            dN_dE = raw_dN_dE * (total_yield / raw_integral) if raw_integral > 0 else raw_dN_dE
            return PhasespaceSlice(axes=values, distr=dN_dE)

        raise AssertionError(f"AnalyticalEngine._fill: {kind} is in SUPPORTED_OUTPUTS but has no branch")
