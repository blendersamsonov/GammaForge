"""Smooth Stage-0 observables: the diagnostic that separates source integration from Stage 1.

Phase I measured only *histogrammed* outputs, which conflate two very different things. A
spectrum L1 error is a sum of per-cell Poisson noise: reallocating particles between regions
does not change how many land in a given cell, so an allocation rule can improve the
integral it optimizes while leaving the spectrum flat or worse. That is what was measured, and
it is why the luminosity term looked regime-dependent rather than useful.

The Stage-0 map ``d -> Y(d)`` from six standard-normal latents to
``(gamma, theta_x, theta_y, a0_shape, chirp_mean, luminosity)`` is smooth almost everywhere.
So before asking whether any source rule helps, this module measures the error on *smooth
functionals of that map* with no deposition at all:

- ``M0 = E[L]``
- ``M1_i = E[L y_i]`` and ``M2_ij = E[L y_i y_j]`` for the five coordinates
- nonlinear smooth probes ``E[L a0_shape^3]`` and ``E[L exp(-(thx/th*)^2 - (thy/th*)^2)]``
- characteristic-function probes ``Phi(k) = E[L exp(i k.y~)]``

The characteristic-function probes are the load-bearing addition. A histogram cell indicator is
a *discontinuous* test function, so it samples the push-forward's roughness regardless of how
smooth the map is; a characteristic function is smooth, and how fast ``Phi(k)`` converges says
whether the push-forward's *shape* is being integrated well. Comparing the two rates is what
localizes the loss of high-order convergence (Phase-II handoff §7).

Every observable is a plain weighted sum ``sum(L_i * f_i)`` with **no extra weight factor**:
``TrajectorySamples.luminosity`` already carries the per-particle weight and ``N_e`` (Stage 0
multiplies by ``n_electrons * bunch.weight`` internally, which is why the Phase-I harness can
compute a yield as a bare ``np.sum(samples.luminosity)``). Multiplying by the weight again would
not merely double-count: for IID, where ``L_i is proportional to w_i``, ``sum(w_i L_i)`` scales
as ``1/N``, so every "observable" would shrink with the trajectory count and the fitted
exponents would be meaningless. The weight enters once, inside Stage 0.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

#: The five Stage-1 coordinates the smooth observables are built from.
COORDINATES = ("gamma", "theta_x", "theta_y", "a0_shape", "chirp_mean")


@dataclasses.dataclass(frozen=True)
class Standardization:
    """Per-coordinate location and scale, so probes are dimensionless and scenario-agnostic.

    Taken from a large IID run of the same scenario. Using the *source's own* spread rather
    than a fixed constant is what lets one probe set and one fitted exponent be compared
    across scenarios with very different geometry: ``wide_bunch`` and ``tight_focus`` have
    angular widths orders of magnitude apart, and a hard-coded ``theta_star`` would make one
    of them a trivial or a divergent probe.
    """

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_samples(cls, samples, weights) -> "Standardization":
        y = coordinate_matrix(samples)
        w = np.asarray(weights, dtype=float)
        mean = np.sum(w[:, None] * y, axis=0)
        var = np.sum(w[:, None] * (y - mean) ** 2, axis=0)
        scale = np.sqrt(np.maximum(var, 1e-300))
        return cls(center=mean, scale=scale)

    def apply(self, y: np.ndarray) -> np.ndarray:
        return (y - self.center) / self.scale


def coordinate_matrix(samples) -> np.ndarray:
    """``(n, 5)`` array of the five Stage-0 coordinates, in :data:`COORDINATES` order."""
    return np.column_stack([
        np.asarray(getattr(samples, name), dtype=float) for name in COORDINATES
    ])


def probe_wavenumbers(dim: int = len(COORDINATES)) -> np.ndarray:
    """A small fixed set of dimensionless ``k`` vectors: each axis, then mixed directions.

    Deliberately fixed rather than random, so two methods are probed at *identical* ``k`` and
    the comparison is paired. Mixed directions are included because single-axis probes alone
    would miss anisotropy -- exactly the structure a sparse grid would exploit, so leaving it
    out would bias the answer toward "the tensor rule is fine".
    """
    k = [np.eye(dim)[i] for i in range(dim)]
    k.append(np.full(dim, 0.6))
    k.append(np.array([0.4, -0.5, 0.3, 0.5, -0.4, 0.2, 0.3, -0.3][:dim]))
    return np.array(k, dtype=float)


def evaluate(samples, weights, standardization: Standardization) -> dict:
    """All smooth observables for one Stage-0 run.

    Returns a flat dict of scalars and small arrays keyed by name, so two runs can be compared
    elementwise and a reference is just another such dict. Keys are stable and are what the
    results JSON and the fitted exponents are indexed by.
    """
    L = np.asarray(samples.luminosity, dtype=float)
    y = coordinate_matrix(samples)
    # `weights` is accepted and validated but deliberately unused: luminosity is already
    # weighted (see the module docstring). Summing it again is the bug this note prevents.
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != L.shape[0]:
        raise ValueError(
            f"evaluate: {weights.shape[0]} weights for {L.shape[0]} samples -- the prefilter "
            "drops particles, so the weights must be the post-prefilter ones"
        )
    yt = standardization.apply(y)
    gamma, thx, thy, a0, _chirp = (y[:, i] for i in range(5))
    out: dict = {}

    out["M0"] = float(np.sum(L))
    out["M1"] = np.sum(L[:, None] * y, axis=0)
    out["M2"] = np.einsum("i,ij,ik->jk", L, y, y)

    # Nonlinear but smooth: a cubic response and a Gaussian angular window. The angular
    # window uses the source's own angular scale, which is the same device as the
    # standardization: dimensionless, and comparable across geometries.
    out["a0_cubed"] = float(np.sum(L * a0 ** 3))
    # The source's own angular scale, by the same device as `Standardization`: dimensionless,
    # so one probe is comparable across geometries.
    th_var = float(np.sum(L * (thx ** 2 + thy ** 2)) / max(float(np.sum(L)), 1e-300))
    th_scale = math.sqrt(max(th_var, 1e-300))
    out["angular_window"] = float(np.sum(L * np.exp(-(thx / th_scale) ** 2
                                                  - (thy / th_scale) ** 2)))

    phase = yt @ probe_wavenumbers().T                      # (n, n_k)
    lum = L[:, None]                                       # broadcast over the k probes
    out["charfun_real"] = np.sum(lum * np.cos(phase), axis=0)
    out["charfun_imag"] = np.sum(lum * np.sin(phase), axis=0)
    return out


def relative_error(value: dict, reference: dict) -> dict:
    """Error of each observable against a reference, on a scale that is comparable across them.

    Scalar observables use the relative error. **Array observables use an L2 norm ratio**,
    ``||got - ref|| / ||ref||``, not a maximum of elementwise relatives. The elementwise form
    is unusable here: ``M1`` and ``M2`` mix components spanning many orders of magnitude --
    ``gamma`` is O(1e6) while ``theta`` is O(1e-3) -- and some components of a centred moment
    pass through zero, where a relative error is not a large number but a meaningless one. The
    measured symptom of getting this wrong is a "construction gap" of order 1e17, which reads
    like a divergence and is really one near-zero component.
    """
    out: dict = {}
    for key, ref in reference.items():
        got = value[key]
        ref_a = np.atleast_1d(np.asarray(ref, dtype=float))
        got_a = np.atleast_1d(np.asarray(got, dtype=float))
        denom = float(np.linalg.norm(ref_a))
        if denom <= 0.0:
            out[key] = float(np.linalg.norm(got_a))
        else:
            out[key] = float(np.linalg.norm(got_a - ref_a) / denom)
    return out


def scalar_errors(value: dict, reference: dict) -> dict[str, float]:
    """Per-observable error, as a flat float keyed by observable name.

    :func:`relative_error` already reduces array observables to a scalar norm ratio, so this
    is a naming pass; it exists so the results JSON and the fitted exponents have one stable
    key set to index by.
    """
    return {key: float(np.atleast_1d(np.asarray(err, dtype=float)).ravel()[0])
            if np.asarray(err).ndim == 0 else float(np.asarray(err, dtype=float))
            for key, err in relative_error(value, reference).items()}


def aggregate(errors: dict[str, float]) -> float:
    """One representative error, as the worst of the observable *groups*.

    Kept as a max deliberately: a rule must not look converged because it got the easy moment
    right. The cost is that a single intrinsically hard observable can then dominate, and
    ``E[L a0_shape^3]`` is exactly that -- a third moment of a heavy-tailed quantity that no
    method in the bank converges at these budgets. So this number is reported *together with*
    :func:`group_errors`, never alone, and the headline to read is how many groups are small,
    not what the worst one is.
    """
    groups = group_errors(errors)
    return float(max(groups.values())) if groups else float("nan")


def group_errors(errors: dict[str, float]) -> dict[str, float]:
    """Worst error within each observable group.

    Separating the groups is what makes a headline number readable. Without it, an unconverged
    ``a0_cubed`` masks the fact that ``M0`` is at 1e-2 and makes every arm look equally bad.
    """
    groups = {
        "total": ["M0"],
        "first_moment": ["M1"],
        "second_moment": ["M2"],
        "nonlinear": ["a0_cubed", "angular_window"],
        "shape_cf": ["charfun_real", "charfun_imag"],
    }
    out: dict[str, float] = {}
    for name, names in groups.items():
        present = [errors[n] for n in names if n in errors]
        if present:
            out[name] = float(max(present))
    return out


def effective_sample_fraction(weights) -> float:
    """``N_eff / N`` for a weight vector -- recorded because it is the explanatory variable.

    A deterministic rule can have a pathological weight distribution while being an excellent
    integrator: tensor Gauss-Hermite in six dimensions has weights spanning ``1e21`` and
    ``N_eff/N ~ 5e-3`` at order 8, yet integrates smooth observables spectrally. The same rule
    is a terrible *particle* rule, because deposition needs particles to land in distinct
    cells. Recording this number alongside the histogrammed errors is what distinguishes "the
    rule is inaccurate" from "the rule is fine and deposition is the problem" (handoff §7).
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return float("nan")
    return float((w.sum() ** 2 / np.sum(w ** 2)) / w.size)


def resolved_mask(errors, floor, *, margin: float = 3.0):
    """Indices where the measurement is comfortably above the reference's own error.

    This is the guard the earlier runs lacked, and its absence is why their fitted slopes were
    meaningless: at a 400k reference the spectral floor is ~3e-3, IID reached 3.6e-3 at 262k,
    and the curve was flattening on the *reference*, not converging. A point whose error is
    within ``margin`` times the floor measures the reference, whatever its trend does.
    """
    floor = float(floor)
    if not np.isfinite(floor) or floor <= 0.0:
        return np.ones(len(errors), dtype=bool)
    return np.asarray(errors, dtype=float) > margin * floor


def fit_exponent(n_values, errors, *, floor: float = 0.0, margin: float = 3.0,
                 n_min: float = 2.0) -> dict:
    """Fit ``error ~ N**(-alpha)`` over the part of the curve that is actually measurable.

    Points enter the fit only when they are at or above ``n_min`` trajectories *and* their
    error stands clear of the reference floor. ``n_points`` is reported because a slope from
    three points is a very different claim from a slope from eight, and ``resolved`` is False
    rather than a number when fewer than three survive.
    """
    n_arr = np.asarray(list(n_values), dtype=float)
    e_arr = np.asarray(list(errors), dtype=float)
    finite = np.isfinite(e_arr) & (e_arr > 0)
    keep = finite & (n_arr >= n_min) & resolved_mask(e_arr, floor, margin=margin)
    if keep.sum() < 3:
        return {"alpha": float("nan"), "n_points": int(keep.sum()), "resolved": False,
                "floor": float(floor)}
    slope, intercept = np.polyfit(np.log(n_arr[keep]), np.log(e_arr[keep]), 1)
    return {"alpha": float(-slope), "n_points": int(keep.sum()), "resolved": True,
            "prefactor": float(np.exp(intercept)), "floor": float(floor),
            "n_used": [float(v) for v in n_arr[keep]]}


def trajectories_for_error(n_values, errors, target: float, *, floor: float = 0.0,
                           margin: float = 3.0) -> float:
    """Stage-0 trajectories needed to reach ``target`` relative error, from the fitted slope.

    Returns ``nan`` when the fit did not resolve or the extrapolated count is absurd, rather
    than quoting a number. A trajectory count extrapolated from two points is how a 100x claim
    gets made, and an "unresolved" is more useful than a number that cannot be defended.
    """
    fit = fit_exponent(n_values, errors, floor=floor, margin=margin)
    if not fit["resolved"] or not math.isfinite(fit["alpha"]) or fit["alpha"] <= 0:
        return float("nan")
    count = (fit["prefactor"] / target) ** (1.0 / fit["alpha"])
    if not np.isfinite(count) or count > 1e12:
        return float("nan")
    return float(count)


def reference_from_runs(values: list) -> tuple[dict, float]:
    """Mean of independent reference constructions, and a floor from their disagreement.

    The floor is the largest per-observable disagreement between the runs. That is a *measured*
    limit on the reference, not an assumed one, and it is what :func:`fit_exponent` gates on --
    without it the fits silently reported the reference's own convergence.
    """
    mean = {k: np.mean([np.atleast_1d(np.asarray(v[k], dtype=float)) for v in values], axis=0)
            for k in values[0]}
    if len(values) < 2:
        return mean, 0.0
    per_obs = {}
    for k in values[0]:
        stack = np.stack([np.atleast_1d(np.asarray(v[k], dtype=float)) for v in values])
        denom = float(np.max(np.abs(np.mean(stack, axis=0))))
        spread = float(np.max(np.abs(stack - np.mean(stack, axis=0))))
        per_obs[k] = spread / denom if denom > 0 else spread
    return mean, float(max(per_obs.values()))
