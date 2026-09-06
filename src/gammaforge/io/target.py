"""Target: collimation window and requested outputs (GRAND_PLAN.md §3.4).

A first-class concept, where the predecessor scattered "target" between GUI fields and
adapter methods.

**Ranges are auto-derived; the user never enters them.** An `OutputRequest` carries only
a resolution. Every range comes from :func:`auto_ranges` — Compton-edge kinematics for
energy, the radiation cone for angles, beam and laser sizes for space, the actual
beam-laser overlap for time. The one exception is `OutputKind.COLLIMATED_SPECTRUM`, whose
angular ranges *are* the target's collimation window by definition. GUI plots offer zoom
instead of manual range entry.

**Autoranging never samples raw fields** (§3.4/P15): it reads descriptive laser metrics
through :func:`~gammaforge.io.laser.fit_gaussian_paraxial`, which is what keeps it
correct when a non-Gaussian `LaserField` arrives.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .bunch import GaussianElectronBeam, overlap_time_window
from .laser import LaserField, fit_gaussian_paraxial
from .results import Axis
from .units import MEC2_CGS, Quantity, as_canonical_quantity

__all__ = [
    "OutputKind",
    "OutputRequest",
    "Target",
    "auto_ranges",
    "compton_edge_energy",
    "slice_axis_values",
    "slice_axis_widths",
    "SLICE_AXES",
    "RANGE_HEADROOM",
]


class OutputKind(Enum):
    """The engine-facing output vocabulary (§3.4).

    Engines declare which of these they can produce as plain ``supported_outputs`` data
    (P10) — no registry, no capability-negotiation protocol.
    """

    TOTAL_YIELD = "0d_yield"
    SPECTRUM = "spectrum"
    TEMPORAL_ENVELOPE = "temporal_envelope"
    SPATIAL_DISTRIBUTION = "spatial_distribution"
    ANGULAR_DISTRIBUTION = "angular_distribution"
    COLLIMATED_SPECTRUM = "collimated_spectrum"
    MACROPARTICLE_DUMP = "macroparticle_dump"


#: Which slice axes each output kind spans. `MACROPARTICLE_DUMP` is not a slice at all,
#: and `TOTAL_YIELD` is the 0D slice — both are in the table so a caller never has to
#: special-case membership.
SLICE_AXES: dict[OutputKind, tuple[Axis, ...] | None] = {
    OutputKind.TOTAL_YIELD: (),
    OutputKind.SPECTRUM: (Axis.ENERGY,),
    OutputKind.TEMPORAL_ENVELOPE: (Axis.TIME,),
    OutputKind.SPATIAL_DISTRIBUTION: (Axis.X, Axis.Y),
    OutputKind.ANGULAR_DISTRIBUTION: (Axis.THETA_X, Axis.THETA_Y),
    OutputKind.COLLIMATED_SPECTRUM: (Axis.ENERGY, Axis.THETA_X, Axis.THETA_Y),
    OutputKind.MACROPARTICLE_DUMP: None,
}


@dataclass(frozen=True)
class OutputRequest:
    """One requested observable and how finely to resolve it.

    ``resolution`` gives the bin count per axis, in :data:`SLICE_AXES` order.
    ``manual_ranges`` is the advanced-option override the plan allows for
    `SPATIAL_DISTRIBUTION` only; everywhere else a range is derived, never entered.
    """

    kind: OutputKind
    resolution: tuple[int, ...] = ()
    manual_ranges: dict[Axis, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        axes = SLICE_AXES[self.kind]
        if axes is None:
            if self.resolution or self.manual_ranges:
                raise ValueError(f"{self.kind.name} is not a slice — it takes no resolution or ranges")
            return
        if len(self.resolution) != len(axes):
            raise ValueError(
                f"{self.kind.name} needs {len(axes)} resolution value(s) for axes "
                f"{tuple(a.name for a in axes)}, got {self.resolution}"
            )
        if any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1
               for n in self.resolution):
            raise ValueError(f"{self.kind.name}: resolutions must be positive integers, got {self.resolution}")
        if self.manual_ranges is not None:
            if self.kind is not OutputKind.SPATIAL_DISTRIBUTION:
                raise ValueError(
                    f"{self.kind.name}: manual ranges are an advanced override for "
                    f"SPATIAL_DISTRIBUTION only — every other range is auto-derived (§3.4)"
                )
            unknown = set(self.manual_ranges) - set(axes)
            if unknown:
                raise ValueError(f"{self.kind.name}: {sorted(a.name for a in unknown)} is not one of its axes")
            for axis, values in self.manual_ranges.items():
                try:
                    low, high = values
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{self.kind.name}: {axis.name} range must be a (low, high) pair") from exc
                if not (np.isfinite(low) and np.isfinite(high) and low < high):
                    raise ValueError(f"{self.kind.name}: {axis.name} range must be finite and increasing")


@dataclass(frozen=True)
class Target:
    """Collimation window plus the requested outputs.

    ``theta_x_col``/``theta_y_col`` are **half-angles** in rad. They define the angular
    ranges of `OutputKind.COLLIMATED_SPECTRUM` — the "spectrum on target" — and nothing
    else; the fully angle-integrated `SPECTRUM` ignores them.
    """

    theta_x_col: Quantity  # angle
    theta_y_col: Quantity  # angle
    outputs: tuple[OutputRequest, ...] = ()

    UNITS = {"theta_x_col": "rad", "theta_y_col": "rad"}

    #: Longitudinal extents, which §2.1 allows to be quoted as either a length or a
    #: duration. Only these opt into the `light_time` equivalence — a transverse size
    #: given in femtoseconds is a mistake, not a unit choice.
    LIGHT_TIME_FIELDS = frozenset(set())

    def __post_init__(self) -> None:
        for name, unit in self.UNITS.items():
            object.__setattr__(
                self,
                name,
                as_canonical_quantity(
                    getattr(self, name), unit, name, light_time=name in self.LIGHT_TIME_FIELDS
                ),
            )
        if self.m("theta_x_col") <= 0 or self.m("theta_y_col") <= 0:
            raise ValueError("Target: collimation half-angles must be > 0")
        kinds = [request.kind for request in self.outputs]
        if len(set(kinds)) != len(kinds):
            raise ValueError("Target: an output kind may be requested only once")

    def m(self, name: str) -> float:
        """Magnitude of a dimensioned field in its canonical CGS unit."""
        return float(getattr(self, name).magnitude)

    def request(self, kind: OutputKind) -> OutputRequest | None:
        for output in self.outputs:
            if output.kind is kind:
                return output
        return None

    @property
    def kinds(self) -> tuple[OutputKind, ...]:
        return tuple(output.kind for output in self.outputs)


# ---------------------------------------------------------------------------
# Auto-ranging
# ---------------------------------------------------------------------------
#: How much headroom an auto-derived range leaves beyond the physical scale it is built
#: from. Generous on purpose: an output clipped at its edge is a silently wrong result,
#: while an over-wide range only costs empty bins. It also absorbs effects that can only
#: move a boundary *inwards*, such as the nonlinear a0 redshift of the Compton edge.
RANGE_HEADROOM = 1.2

#: Angular half-width of the auto angular window, in units of the 1/gamma radiation cone.
_ANGULAR_CONE_FACTOR = 4.0

#: Transverse half-width of the auto spatial window, in units of the source RMS size.
_SPATIAL_SIGMA_FACTOR = 4.0


def compton_edge_energy(beam: GaussianElectronBeam, photon_energy: float) -> float:
    """Maximum backscattered photon energy for a head-on collision, erg.

    ``E_max = E_L (1 + beta)^2 gamma^2 / (1 + 2 gamma E_L (1 + beta) / (m c^2))``, the
    exact linear-Compton edge — reducing to the familiar ``4 gamma^2 E_L`` when the
    recoil term is negligible, which is xigma's regime (§2.3) but not every engine's.

    The nonlinear ``a0`` redshift is deliberately not applied: it only moves the edge
    *down*, so a range built on this value with :data:`RANGE_HEADROOM` still contains it,
    and applying a redshift formula the paper has not derived would be the silent
    approximation P14 forbids.
    """
    gamma = beam.gamma0()
    beta = beam.beta0()
    return photon_energy * (1.0 + beta) ** 2 * gamma**2 / (
        1.0 + 2.0 * gamma * photon_energy * (1.0 + beta) / MEC2_CGS
    )


def auto_ranges(
    target: Target,
    beam: GaussianElectronBeam,
    laser: LaserField,
    bunch=None,
) -> dict[OutputKind, dict[Axis, tuple[float, float]]]:
    """Derive the ``(low, high)`` range of every axis of every requested output.

    ``bunch`` is required only for `OutputKind.TEMPORAL_ENVELOPE`, whose window comes
    from the actual per-particle beam-laser overlap rather than an estimate — the
    generalized replacement for the predecessor's head-on-only
    ``laser_overlap_time_window``, correct for a crossing angle without a special case.
    """
    metrics = fit_gaussian_paraxial(laser)
    edge = compton_edge_energy(beam, metrics.photon_energy())

    energy_range = (0.0, RANGE_HEADROOM * edge)
    # The radiation cone is ~1/gamma; a divergent beam smears it by its own spread, and
    # the two add in quadrature since they are independent.
    cone_x = _ANGULAR_CONE_FACTOR * math.hypot(1.0 / beam.gamma0(), beam.divergence_x())
    cone_y = _ANGULAR_CONE_FACTOR * math.hypot(1.0 / beam.gamma0(), beam.divergence_y())
    angular_range = {
        Axis.THETA_X: (-RANGE_HEADROOM * cone_x, RANGE_HEADROOM * cone_x),
        Axis.THETA_Y: (-RANGE_HEADROOM * cone_y, RANGE_HEADROOM * cone_y),
    }
    # Photons are emitted where electrons and laser overlap, so the source is no larger
    # than the smaller of the two transverse sizes.
    source_x = _SPATIAL_SIGMA_FACTOR * min(beam.m("sigma_x"), metrics.m("sigma_x"))
    source_y = _SPATIAL_SIGMA_FACTOR * min(beam.m("sigma_y"), metrics.m("sigma_y"))
    spatial_range = {
        Axis.X: (-RANGE_HEADROOM * source_x, RANGE_HEADROOM * source_x),
        Axis.Y: (-RANGE_HEADROOM * source_y, RANGE_HEADROOM * source_y),
    }
    # Ranges are plain CGS floats keyed by `Axis`, matching how a `PhasespaceSlice`
    # stores its own axis values: the unit is declared by the axis, not carried per value.
    theta_x_col, theta_y_col = target.m("theta_x_col"), target.m("theta_y_col")
    collimation_range = {
        Axis.THETA_X: (-theta_x_col, theta_x_col),
        Axis.THETA_Y: (-theta_y_col, theta_y_col),
    }

    ranges: dict[OutputKind, dict[Axis, tuple[float, float]]] = {}
    for request in target.outputs:
        kind = request.kind
        if SLICE_AXES[kind] is None:
            continue
        if kind is OutputKind.TOTAL_YIELD:
            ranges[kind] = {}
        elif kind is OutputKind.SPECTRUM:
            ranges[kind] = {Axis.ENERGY: energy_range}
        elif kind is OutputKind.ANGULAR_DISTRIBUTION:
            ranges[kind] = dict(angular_range)
        elif kind is OutputKind.COLLIMATED_SPECTRUM:
            ranges[kind] = {Axis.ENERGY: energy_range, **collimation_range}
        elif kind is OutputKind.SPATIAL_DISTRIBUTION:
            derived = dict(spatial_range)
            derived.update(request.manual_ranges or {})
            ranges[kind] = derived
        elif kind is OutputKind.TEMPORAL_ENVELOPE:
            if bunch is None:
                raise ValueError(
                    "auto_ranges: TEMPORAL_ENVELOPE needs the bunch — its window is the "
                    "actual beam-laser overlap, not an estimate (§3.4)"
                )
            ranges[kind] = {Axis.TIME: _overlap_window(bunch, laser)}
        else:  # pragma: no cover — the table above is exhaustive over slice kinds
            raise AssertionError(f"no auto-range rule for {kind}")
    return ranges


def _overlap_window(bunch, laser) -> tuple[float, float]:
    t0, t1 = overlap_time_window(bunch, laser)
    overlapping = t0 <= t1
    if not np.any(overlapping):
        raise ValueError(
            "auto_ranges: no macroparticle ever enters the laser's active region — "
            "there is no temporal window to plot. Check the collision geometry."
        )
    low = float(np.min(t0[overlapping]))
    high = float(np.max(t1[overlapping]))
    centre = 0.5 * (low + high)
    half = 0.5 * (high - low) * RANGE_HEADROOM
    return centre - half, centre + half


def slice_axis_values(
    request: OutputRequest, ranges: dict[Axis, tuple[float, float]]
) -> dict[Axis, np.ndarray]:
    """Bin-centre values for one output's axes, from its resolution and derived ranges.

    Bin *centres*, not edges: `PhasespaceSlice` stores densities sampled at points, and
    centres are what make the trapezoid integral of a smooth density accurate.
    """
    axes = SLICE_AXES[request.kind]
    if axes is None:
        raise ValueError(f"{request.kind.name} is not a slice")
    values = {}
    for axis, n_bins in zip(axes, request.resolution):
        low, high = ranges[axis]
        width = (high - low) / n_bins
        values[axis] = low + width * (np.arange(n_bins) + 0.5)
    return values


def slice_axis_widths(
    request: OutputRequest, ranges: dict[Axis, tuple[float, float]]
) -> dict[Axis, np.ndarray]:
    """Constant histogram cell widths for an output request's axis ranges.

    This is separate from :func:`slice_axis_values`: centre coordinates identify where a
    histogram is plotted, while widths define its integration measure.
    """
    axes = SLICE_AXES[request.kind]
    if axes is None:
        raise ValueError(f"{request.kind.name} is not a slice")
    return {
        axis: np.full(n_bins, (ranges[axis][1] - ranges[axis][0]) / n_bins)
        for axis, n_bins in zip(axes, request.resolution)
    }
