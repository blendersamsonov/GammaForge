"""How two results are compared (GRAND_PLAN.md §7).

Cross-engine legs are **tolerance-based**, and Monte-Carlo legs are compared with a
*statistical* tolerance rather than a tight absolute bound. That makes the choice of
metric load-bearing: a pointwise max-relative-error on a noisy spectrum reports the
noisiest empty bin, not the disagreement anyone cares about.

The window-integrated metric ported from the predecessor solves that. Integrate the
density into windows of a stated reporting width, then report two numbers against a
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


def window_integrated_deviation(x, y, y_ref, window, floor_fraction=1e-6):
    """Compare two densities sampled on the same 1D grid ``x``.

    ``window`` is the reporting resolution, in ``x``'s own units: the width over which
    flux is integrated before comparing. Returns ``(weighted_l1, max_window, n_windows)``.
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

    total_ref = flux_ref.sum()
    floor = floor_fraction * total_ref / n_windows if total_ref > 0 else 0.0
    deviation = np.abs(flux - flux_ref) / np.maximum(np.abs(flux_ref), max(floor, 1e-300))

    significant = flux_ref > floor
    if not np.any(significant):
        return 0.0, 0.0, n_windows

    weights = np.where(significant, flux_ref, 0.0)
    weights = weights / weights.sum()
    return float(np.sum(weights * deviation)), float(deviation[significant].max()), n_windows


def compare_slices(
    slice_: PhasespaceSlice,
    reference: PhasespaceSlice,
    *,
    windows: int = 16,
) -> Deviation:
    """Compare two slices of the same axis grouping, resampling onto the reference's grid.

    Only the **1D** case gets a window-integrated shape metric; for a 0D total yield or a
    multi-dimensional slice, the shape numbers are the yield error, since a windowing
    scheme for a 3D density is a reporting decision nobody has needed yet (P6). Every
    grouping still gets its integrated-yield comparison, which is the number the identity
    tests of §7 are written against.

    ``windows`` is how many reporting windows to divide the reference's span into.
    """
    if frozenset(slice_.axes) != frozenset(reference.axes):
        raise ValueError(
            f"compare_slices: {sorted(a.name for a in slice_.axes)} vs "
            f"{sorted(a.name for a in reference.axes)} — different observables"
        )

    yield_error = relative_error(_total(slice_), _total(reference))
    if len(reference.axes) != 1:
        magnitude = abs(yield_error)
        return Deviation(magnitude, magnitude, 1, yield_error)

    (axis,) = tuple(reference.axes)
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
