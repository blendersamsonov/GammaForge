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
    """One representative error per observable group, then the worst of them.

    Groups are the physically distinct questions: the total, the first and second moments of
    the push-forward, the nonlinear probes, and the characteristic function. The reported
    number is the maximum over groups, so a rule cannot look converged because it got the
    easy moment right while the shape probes were still wrong.
    """
    groups = {
        "total": ["M0"],
        "first_moment": ["M1"],
        "second_moment": ["M2"],
        "nonlinear": ["a0_cubed", "angular_window"],
        "shape_cf": ["charfun_real", "charfun_imag"],
    }
    worst = 0.0
    for names in groups.values():
        present = [errors[n] for n in names if n in errors]
        if present:
            worst = max(worst, max(present))
    return float(worst)


def fit_exponent(n_values, errors, *, n_min: float = 2.0) -> dict:
    """Fit ``error ~ N**(-alpha)`` over the resolved part of a convergence curve.

    Only points at or above ``n_min`` trajectories enter the fit: at tiny ``N`` the error is
    dominated by the discrete support not yet covering the source, not by the quadrature's
    asymptotic rate, and including those points biases ``alpha`` downwards. The fit is
    reported with the number of contributing points, because a slope from three points is a
    claim with much less behind it than a slope from eight.
    """
    n_arr = np.asarray(list(n_values), dtype=float)
    e_arr = np.asarray(list(errors), dtype=float)
    keep = (n_arr >= n_min) & np.isfinite(e_arr) & (e_arr > 0)
    if keep.sum() < 3:
        return {"alpha": float("nan"), "n_points": int(keep.sum()), "resolved": False}
    slope, intercept = np.polyfit(np.log(n_arr[keep]), np.log(e_arr[keep]), 1)
    return {"alpha": float(-slope), "n_points": int(keep.sum()), "resolved": True,
            "prefactor": float(np.exp(intercept))}


def trajectories_for_error(n_values, errors, target: float) -> float:
    """Stage-0 trajectories needed to reach ``target`` relative error, from the fitted slope.

    Returns ``nan`` when the fit did not resolve, rather than extrapolating: quoting a
    trajectory count from an unresolved slope is how a 100x claim gets made from three points.
    """
    fit = fit_exponent(n_values, errors)
    if not fit["resolved"] or not math.isfinite(fit["alpha"]) or fit["alpha"] <= 0:
        return float("nan")
    # error = prefactor * N**-alpha  ->  N = (prefactor / target) ** (1/alpha)
    return float((fit["prefactor"] / target) ** (1.0 / fit["alpha"]))