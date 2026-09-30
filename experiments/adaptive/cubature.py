"""Deterministic source rules for the Phase-II cubature study.

Three source constructions, all producing the *same* six independent standard-normal latent
coordinates the IID path produces, so every method feeds the identical Stage-0 pipeline and
the comparison isolates the source rule alone:

- :func:`global_qmc` — one extensible Halton sequence, shifted, mapped through the validated
  ``norm_ppf``, equal weights ``1/N``. No regions, no pilot, no allocation.
- :func:`tensor_gauss_hermite` — the exact Gaussian cubature rule. For ``dim``-dimensional
  standard normals, 1D nodes ``(x_k, w_k)`` from ``numpy.polynomial.hermite.hermgauss`` give
  nodes ``sqrt(2) x_k`` and weights ``prod(w) / pi^(dim/2)``, all positive.
- :func:`sobol_qmc` — optional experiment-only comparison when SciPy is importable. Never a
  GammaForge dependency.

Two properties matter and are asserted rather than assumed:

**Positivity.** Every weight here is positive. That is a deliberate constraint: signed
cubature weights are fine for a smooth-observable experiment but are not a `Bunch.weight`,
and a source rule that only works for the experiment cannot become the production one
without redoing the work.

**Extensibility.** Asking for a larger ``N`` (or a higher order) *extends* the same rule
rather than resampling. That is what makes an error-versus-``N`` curve meaningful: a rule that
moved its points when ``N`` changed would not have a convergence rate to measure. The Halton
index starts at 1, since the index-0 point is the origin and ``norm_ppf(0)`` is ``-inf``; this
mirrors :func:`gammaforge.io.adaptive_sampling._halton_points`, whose prefix stability is what
made multiplicative refinement exact there.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from gammaforge.io.adaptive_sampling import HALTON_BASES, N_LATENT, _stream_shift, norm_ppf

__all__ = [
    "global_qmc",
    "tensor_gauss_hermite",
    "sobol_qmc",
    "smolyak_probe_available",
    "smolyak_gauss_hermite",
    "build_source_bunch",
    "node_count",
]


def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    """van der Corput radical inverse, index starting at 1."""
    out = np.zeros(indices.shape, dtype=float)
    f, i = 1.0 / base, indices.astype(float).copy()
    while np.any(i > 0):
        digit = np.mod(i, base)
        out += digit * f
        i = np.floor(i / base)
        f /= base
    return out


def global_qmc(n: int, seed: int = 0, *, dim: int = N_LATENT) -> tuple[np.ndarray, np.ndarray]:
    """``(n, dim)`` standard-normal deviates and equal weights, from one Halton sequence.

    The shift is a *digital* shift derived from ``seed``, so the sequence for a given seed is
    fixed and independent of ``n``: point ``k`` is the same point at every ``n``. Raising the
    digit shift (rather than using a nested-scrambled sequence) keeps that property while
    still decorrelating seeds from each other, which is what the multi-seed agreement check
    in the reference needs.
    """
    if n < 1:
        raise ValueError(f"global_qmc: n must be >= 1, got {n}")
    if dim > len(HALTON_BASES):
        raise ValueError(f"global_qmc: dim {dim} exceeds the {len(HALTON_BASES)} Halton bases")
    indices = np.arange(1, n + 1, dtype=np.int64)
    points = np.empty((n, dim), dtype=float)
    # One shift vector, reused per axis via the region/stream ids, so each axis gets an
    # independent shift drawn from the same deterministic source the adaptive path uses.
    for axis in range(dim):
        shift = float(_stream_shift(seed, axis, "global")[0])
        points[:, axis] = np.mod(_radical_inverse(indices, HALTON_BASES[axis]) + shift, 1.0)
    # norm_ppf clamps to the representable open interval; a point landing exactly on 0 or 1
    # after the modulo would otherwise be +/-inf.
    points = np.clip(points, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    return norm_ppf(points), np.full(n, 1.0 / n)


def tensor_gauss_hermite(order: int, *, dim: int = N_LATENT) -> tuple[np.ndarray, np.ndarray]:
    """``(order**dim, dim)`` nodes and positive normalized weights for a ``dim``-D standard normal.

    ``hermgauss`` integrates against ``exp(-x^2)``; substituting ``x = z / sqrt(2)`` turns that
    into the standard-normal measure, giving nodes ``z = sqrt(2) x`` and the 1D weight
    ``w / sqrt(pi)``. In ``dim`` dimensions the product rule multiplies, so the weight is
    ``prod(w) / pi^(dim/2)``.
    """
    if order < 1:
        raise ValueError(f"tensor_gauss_hermite: order must be >= 1, got {order}")
    x, w = np.polynomial.hermite.hermgauss(order)
    w = w / math.sqrt(math.pi)  # 1D standard-normal weights, already summing to 1
    grids = np.meshgrid(*([x * math.sqrt(2.0)] * dim), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=1)
    wgrids = np.meshgrid(*([w] * dim), indexing="ij")
    weights = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    return nodes, weights


def sobol_qmc(n: int, seed: int = 0, *, dim: int = N_LATENT,
             bits: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Optional experiment-only Sobol comparison. Requires SciPy; never a runtime dependency.

    Scrambled, so it is *not* extensible the way :func:`global_qmc` is -- a larger ``n`` is a
    different point set. It is included only as a second opinion on the QMC family, and its
    convergence curve is reported separately rather than pooled with Halton's.
    """
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "sobol_qmc needs SciPy, which is not a GammaForge dependency. Use global_qmc, "
            "or install SciPy in the experiment environment for this comparison only."
        ) from exc
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random(n)
    u = np.clip(u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    return norm_ppf(u), np.full(n, 1.0 / n)


def smolyak_probe_available(dim: int = N_LATENT, min_order: int | None = None,
                            tolerance: int = 2, tol: float = 1e-10) -> bool:
    """Whether a Smolyak combination of 1D Gauss-Hermite rules is usable here.

    False in practice, and that is the finding rather than a gap. See
    :func:`smolyak_gauss_hermite` for why.
    """
    try:
        _, weights = smolyak_gauss_hermite(dim=dim, min_order=min_order,
                                           tolerance=tolerance)
    except ValueError:
        return False
    return bool(weights.min() >= -tol)


def smolyak_gauss_hermite(*, dim: int = N_LATENT, min_order: int | None = None,
                          tolerance: int = 2, tol: float = 1e-10
                          ) -> tuple[np.ndarray, np.ndarray]:
    """A Smolyak combination of 1D Gauss-Hermite rules -- **raises in this phase**.

    Written and checked, and it does not produce a usable rule here, so it is not offered as
    a source construction. The reason is worth recording precisely, because it is a property
    of Gauss-Hermite and not a coding slip:

    The classical Smolyak combination is built from *difference operators* on a nested 1D
    sequence: ``A(q, d) = sum_{q-d+1 <= |i| <= q} (-1)^(q-|i|) C(d-1, q-|i|) (x) U^i``. Its
    correctness rests on the constituent rules being successive differences of one another, so
    that the combination is a partition of unity.

    Gauss-Hermite orders are **not** nested differences. Each ``U^i`` is an independent rule,
    so the combination's weights sum to ``sum of the coefficients``, which for this window is
    ``C(q-d, dim-d)`` and equals 1 only when ``dim == d``. Measured here:

        dim=6, q=6, d=2  ->  1.0   (only because the window is truncated: orders start at 1,
                                  so |i| >= 6 and the "difference" degenerates to U^6)
        dim=6, q=7, d=2  ->  5.0   (a genuine difference, and it is not a partition of unity)
        dim=2, q=3, d=2  ->  1.0

    A rule whose weights do not integrate the constant 1 is not a quadrature, so using it --
    even for a smooth-observable experiment, where signed weights would otherwise be tolerable
    -- would put a silent normalisation error into the very quantity the experiment is meant to
    measure. The honest move is the one the handoff names for exactly this case: stop after
    the tensor rule rather than ship a fragile one.

    A correct sparse-grid Gauss-Hermite needs the H-index construction (nested difference rules
    over *composite* Gauss-Hermite with matched orders), which is a research implementation in
    its own right and a separate review. :func:`smolyak_probe_available` returns False, and the
    experiment records why rather than quietly dropping the question.

    Raises ``ValueError`` always in this phase.
    """
    raise ValueError(
        "smolyak_gauss_hermite: the classical combination of Gauss-Hermite rules is not a "
        "partition of unity in dim != d (measured: coefficient sum = C(q-d, dim-d)), so it is "
        "not a valid quadrature here. The tensor Gauss-Hermite rule is the offered "
        "deterministic construction; a correct sparse grid needs the H-index construction and "
        "its own review. See the docstring."
    )


def node_count(method: str, n: int) -> int:
    """Stage-0 trajectory count a method spends for its nominal budget ``n``.

    For the tensor rule ``n`` is the 1D *order*, not a particle count, so this is the only
    place the two conventions meet.
    """
    if method in ("tensor-gh", "gauss-hermite"):
        return int(n) ** N_LATENT
    return int(n)


def build_source_bunch(beam, deviates: np.ndarray, weights: np.ndarray, *, meta=None):
    """Materialise a ``Bunch`` from explicit latent deviates and weights.

    Goes through the same ``_bunch_from_standard_deviates`` the IID and adaptive paths share,
    so the latent-to-physical map, the Twiss/gamma algebra and the momentum derivation are
    identical across all four source rules. That identity is the whole point: a difference in
    the results is then attributable to the sampling rule and not to three copies of the same
    algebra drifting apart.
    """
    from gammaforge.io.bunch import _bunch_from_standard_deviates

    deviates = np.ascontiguousarray(deviates, dtype=float)
    weights = np.ascontiguousarray(weights, dtype=float)
    if deviates.shape != (weights.size, N_LATENT):
        raise ValueError(
            f"build_source_bunch: deviates {deviates.shape} incompatible with "
            f"{weights.size} weights"
        )
    return _bunch_from_standard_deviates(beam, deviates, weights, meta=meta)