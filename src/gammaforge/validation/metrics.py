"""How two results are compared (GRAND_PLAN.md §7).

Cross-engine legs are **tolerance-based**, and Monte-Carlo legs are compared with a
*statistical* tolerance rather than a tight absolute bound. That makes the choice of
metric load-bearing: a pointwise max-relative-error on a noisy spectrum reports the
noisiest empty bin, not the disagreement anyone cares about.

The window-integrated metric solves that. Integrate the density into windows of a stated
reporting width, then report two numbers against a
reference on the same grid:

* the reference-flux-weighted L1 relative deviation — the headline number, representative
  of the whole spectrum, and
* the maximum relative deviation over windows — sensitive to a localized failure such as a
  misplaced Compton edge, which the weighted number would average away.

Windows with negligible reference flux are down-weighted rather than dropped, so a few
empty bins at a grid edge cannot manufacture an infinity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.results import PhasespaceSlice

__all__ = [
    "Deviation",
    "resample_to",
    "window_integrated_deviation",
    "compare_slices",
    "relative_error",
]


@dataclass(frozen=True)
class Deviation:
    """The outcome of one comparison. ``weighted_l1``/``max_window`` are relative."""

    weighted_l1: float
    max_window: float
    n_windows: int
    yield_error: float

    def worst(self) -> float:
        return max(self.weighted_l1, self.max_window, abs(self.yield_error))


def relative_error(value: float, reference: float) -> float:
    """Signed relative error, falling back to the absolute difference at zero reference."""
    if reference == 0.0:
        return float(value)
    return float((value - reference) / reference)


def resample_to(x_ref, x_src, y_src):
    """Linear-interpolate ``y_src(x_src)`` onto ``x_ref``; zero outside ``x_src``'s range.

    Zero rather than edge-extension on purpose: outside the source grid the source made no
    statement, and extending its last value would invent flux the engine never produced.
    """
    return np.interp(np.asarray(x_ref, float), np.asarray(x_src, float),
                     np.asarray(y_src, float), left=0.0, right=0.0)


#: A window carrying less than this fraction of the total flux is not a spectral feature —
#: it is a grid edge or numerical residue, and a relative error on it says nothing. At or
#: above it, the window is compared, whichever side the flux is on.
#:
#: Deliberately a fraction of the **total**, not of the mean window, so the threshold does
#: not move when the binning is refined. And deliberately small: this number is the
#: metric's *sensitivity*, so raising it buys quiet at the cost of blindness. At 1e-3 —
#: where this briefly sat — a candidate that dropped a real feature worth 0.07% of the
#: yield scored exactly zero on both reported numbers, which is the failure RES023 was
#: written to remove, reintroduced on the reference side.
SIGNIFICANT_FLUX_FRACTION = 1e-4


def window_integrated_deviation(x, y, y_ref, window, significance=SIGNIFICANT_FLUX_FRACTION):
    """Compare two densities sampled on the same 1D grid ``x``.

    ``window`` is the reporting resolution, in ``x``'s own units: the width over which
    flux is integrated before comparing. Returns ``(weighted_l1, max_window, n_windows)``.

    **A window counts if either side has flux in it**, not only the reference. Restricting
    the comparison to the reference's own support is the natural-looking choice and it
    blinds the metric to the failure this whole module exists to catch: photons appearing
    where the reference has none — a Compton edge in the wrong place — contribute nothing
    to a reference-weighted average and would be excluded from the maximum as well.
    ``weighted_l1`` stays reference-weighted, since that is what it means; ``max_window``
    is what reports flux that should not be there.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    if not (x.shape == y.shape == y_ref.shape):
        raise ValueError(
            f"window_integrated_deviation: shapes differ — x{x.shape}, y{y.shape}, "
            f"y_ref{y_ref.shape}; resample onto a common grid first"
        )
    order = np.argsort(x)
    x, y, y_ref = x[order], y[order], y_ref[order]

    span = x[-1] - x[0]
    n_windows = max(1, int(round(span / window))) if window > 0 else 1
    edges = np.linspace(x[0], x[-1], n_windows + 1)
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_windows - 1)

    dx = np.gradient(x)
    flux = np.bincount(index, weights=y * dx, minlength=n_windows)
    flux_ref = np.bincount(index, weights=y_ref * dx, minlength=n_windows)

    # The scale is the larger of the two totals, so an all-zero reference does not make
    # every candidate compare as a perfect match — it makes the candidate's own flux the
    # thing being measured, which is the honest answer.
    total_ref = float(flux_ref.sum())
    scale = max(total_ref, float(np.abs(flux).sum()))
    if scale <= 0.0:
        return 0.0, 0.0, n_windows
    threshold = significance * scale
    # The same number guards the division and decides significance, which keeps the
    # reported deviation interpretable at both ends: a window that is missing entirely
    # reports 1.0, and spurious flux against a zero reference reports how many
    # significance-units of it there are — never a ratio against an arbitrary epsilon.
    deviation = np.abs(flux - flux_ref) / np.maximum(np.abs(flux_ref), threshold)

    counted = (np.abs(flux_ref) > threshold) | (np.abs(flux) > threshold)
    if not np.any(counted):
        return 0.0, 0.0, n_windows
    max_window = float(deviation[counted].max())

    weights = np.where(flux_ref > threshold, flux_ref, 0.0)
    total_weight = weights.sum()
    if total_weight <= 0.0:  # nothing in the reference to weight by; the maximum says it all
        return max_window, max_window, n_windows
    return float(np.sum(weights / total_weight * deviation)), max_window, n_windows


def compare_slices(
    slice_: PhasespaceSlice,
    reference: PhasespaceSlice,
    *,
    windows: int = 16,
) -> Deviation:
    """Compare two slices of the same axis grouping, resampling onto the reference's grid.

    Only a **1D slice with a resolvable axis** gets a window-integrated shape metric; a 0D
    total yield, a multi-dimensional slice and a single-bin axis all report the yield error
    instead, since a windowing scheme for a 3D density is a reporting decision nobody has
    needed yet (P6). Every grouping still gets its integrated-yield comparison, which is
    the number the identity tests of §7 are written against.

    ``windows`` is how many reporting windows to divide the reference's span into.
    """
    if frozenset(slice_.axes) != frozenset(reference.axes):
        raise ValueError(
            f"compare_slices: {sorted(a.name for a in slice_.axes)} vs "
            f"{sorted(a.name for a in reference.axes)} — different observables"
        )

    yield_error = relative_error(_total(slice_), _total(reference))
    (axis,) = tuple(reference.axes) if len(reference.axes) == 1 else (None,)
    # A shape metric needs an axis with a width. A 0D yield, a multi-dimensional slice, and
    # a one-bin spectrum all lack one, so all three report the yield error — which is a
    # real comparison, where crashing on the third would only be an accident of `np.gradient`.
    if axis is None or reference.axes[axis].size < 2 or slice_.axes[axis].size < 2:
        magnitude = abs(yield_error)
        return Deviation(magnitude, magnitude, 1, yield_error)

    x_ref = reference.axes[axis]
    y = resample_to(x_ref, slice_.axes[axis], slice_.distr)
    span = float(x_ref[-1] - x_ref[0]) if x_ref.size > 1 else 0.0
    weighted_l1, max_window, n_windows = window_integrated_deviation(
        x_ref, y, reference.distr, window=span / windows if span > 0 else 0.0
    )
    return Deviation(weighted_l1, max_window, n_windows, yield_error)


def _total(slice_: PhasespaceSlice) -> float:
    """Integrated yield, tolerating the single-sample axis `PhasespaceSlice` rejects."""
    if any(values.size < 2 for values in slice_.axes.values()):
        return float(np.sum(slice_.distr))
    return slice_.integrate()
