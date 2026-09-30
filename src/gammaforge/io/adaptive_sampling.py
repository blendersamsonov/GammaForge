"""Luminosity-aware adaptive sampling: the same Gaussian bunch, more cheaply (RES094).

**The problem.** :func:`~gammaforge.io.bunch.sample_gaussian_bunch` draws macroparticles
IID from the beam's six latent standard normals and then runs Xigma Stage 0 — one
trajectory integration per particle — on every one of them. Stage 0 is the expensive stage
by a wide margin: `n_steps` field evaluations per particle, times every particle. IID
sampling is a poor use of that budget twice over: it crowds the dense core, where
neighbouring samples often land in the same Stage-1 cell and are nearly redundant once
deposited, and it starves the tails, which is where the `P_m` mass and much of the
spectrum's shape live.

**The method.** Represent the *same* physical Gaussian by a stratified estimator instead:

1. A broad reference Gaussian :math:`q_s(d) = N(0, s^2 I_6)` supplies a **coverage
   geometry** — a coordinate system in which the target's support becomes the unit cube
   :math:`u_j = \\Phi(d_j / s)`. The reference is *not* the physical beam.
2. That cube is partitioned into axis-aligned boxes (:func:`_subdivide`). They tile it
   exactly, so the partition introduces no cutoff — no finite Gaussian truncation anywhere.
3. Each box has an **exact** target mass :math:`P_m = \\prod_j [\\Phi(h_{mj}) -
   \\Phi(\\ell_{mj})]`, computed from the latent bounds :math:`s\\Phi^{-1}` of its cube
   edges. Because the boxes tile the cube, :math:`\\sum_m P_m = 1` identically.
4. Production particles are drawn from the **target conditional**
   :math:`p(d \\mid R_m)`, not from :math:`q_s` with pointwise importance weights. Since
   the target factorizes and each box is axis-aligned in latent space,
   :math:`d_j = \\Phi^{-1}[a_{mj} + v_j (b_{mj} - a_{mj})]` is *exactly* that conditional
   for any :math:`v_j` uniform on :math:`(0,1)` — so a low-discrepancy :math:`v` gives a
   stratified sample for free, with no variance from the weight ratio at all.
5. Every particle in a box therefore carries the **exact constant** weight
   :math:`w_{mi} = P_m / n_m`, and :math:`\\sum_i w_i = \\sum_m P_m = 1`. No post-hoc
   renormalization is needed or performed, and a change in :math:`n_m` changes only
   numerical resolution, never the represented physical mass.

**Why the luminosity is only a *priority*.** A cheap Stage-0-like pilot
(:func:`trajectory_luminosity_predictor`) estimates :math:`\\ell(X) = F \\int C(t) I\\,dt`
per region, and the production budget is allocated in proportion to
:math:`A_m = P_m \\sqrt{M_{2,m}}` rather than :math:`P_m \\sigma_m`. Two deliberate
conservatisms: the second moment rather than the standard deviation, because GammaForge
needs the *distribution* deposited into the 5D Stage-1 table and not only the total yield
(a nearly-constant-luminosity region can still need many particles because its samples
spread across cells); and an explicit broad-coverage floor,
:math:`Q_m = (1-\\lambda) B_m + \\lambda L_m`, so no region can be starved by a pilot
estimate.

**No pilot-based false negatives.** :math:`Q_m \\ge (1-\\lambda) B_m` and
:math:`B_m > 0` for every box, so every region receives particles no matter what the pilot
says. Since :math:`p/q_s \\le s^6` for :math:`s \\ge 1`, the implied weight ratio is
bounded by :math:`P_m / Q_m \\le s^6 / (1 - \\lambda)` — 32 for the defaults
:math:`s=\\sqrt 2, \\lambda=0.75`. Regions are never *deleted*; the pilot only decides how
many particles each already-existing region receives, and how deep the partition is cut.

**The architectural guarantee.** The pilot may be arbitrarily bad and the represented
distribution is unchanged: :math:`P_m` is exact regardless, and the weights are
:math:`P_m / n_m` regardless. Pilot quality therefore buys *efficiency*, never
*correctness* — which is what makes it safe to run an approximate, cheap pilot at
:math:`O(\\text{regions} \\times 8 \\times 64)` field evaluations against a Stage 0 that
costs :math:`O(N \\times 200)`.

**Observer independence.** Nothing here knows about Stage 2. The plan depends on the beam,
the laser and the seed only; the exact observer-dependent ponderomotive coefficient
:math:`Q` stays in Stage 2, so one plan serves every observation direction (RES094).

**Defaults are numerical, not physical.** The values in :data:`DEFAULT_PILOT_CONFIG` are
starting points to be tuned against measured Stage-1/Stage-2 convergence, not physics
constants; every one is a named field of :class:`PilotConfig` so a benchmark can sweep it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .bunch import Bunch, GaussianElectronBeam, _bunch_from_standard_deviates, overlap_time_window
from .laser import laser_propagation_direction
from .units import C_CGS

__all__ = [
    "SamplingRegion",
    "PilotConfig",
    "AdaptiveSamplingPlan",
    "DEFAULT_PILOT_CONFIG",
    "norm_ppf",
    "trajectory_luminosity_predictor",
    "build_adaptive_plan",
    "build_adaptive_bunch",
]

#: The number of latent dimensions, fixed by ``SAMPLED_VARIABLES`` (§3.2).
N_LATENT = 6

#: Halton bases for the six latent coordinates. Distinct small primes, as a scrambled-free
#: Halton construction requires; 2, 3, 5 and 7 correlate badly in low dimensions, which is
#: exactly why the 6D case needs a wider base set than 2/3 alone would give.
HALTON_BASES = (2, 3, 5, 7, 11, 13)

_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Normal CDF / inverse CDF
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The quantile function is **inverted by iteration against libm's `erfc`**, not by a
# published rational approximation with transcribed constants.
#
# That is a deliberate choice, arrived at by trying the alternative first. Wichura's
# AS 241 and Acklam's algorithm are both the textbook answers, and both are one
# transcription slip away from being catastrophically wrong *and still looking plausible*:
#
# * a wrongly-ordered coefficient list makes the map non-monotone, and the resulting
#   negative regional masses only surface once adaptive refinement produces thin boxes;
# * swapping a polynomial's argument (``q`` vs ``q^2``) is a ~100x error that is invisible
#   in the bulk and obvious only in the tails;
# * a denominator that silently drops its constant term is likewise monotone-looking.
#
# Critically, *none of these are caught by `sum(P_m) == 1`*, because that identity only
# requires the map to be monotone, not correct. Inverting `erfc` by Halley iteration has
# no coefficients to mistype, and being an iteration on the actual function it is accurate
# to floating-point rounding everywhere rather than to the quality of a rational fit.
#
# It costs a few `erfc` evaluations per point instead of a handful of multiplies. That is
# irrelevant here: the regional masses are computed once for a few hundred regions, and a
# production bunch's `N x 6` quantiles are a fraction of the cost of the Stage-0 trajectory
# integrations they exist to avoid.
_SQRT_2PI = math.sqrt(2.0 * math.pi)

#: Halley steps. Cubic convergence from a start that is uniformly within ~0.5 takes four;
#: the fifth is what pins the deep tail, where `erfc` and `exp` themselves run out of
#: relative precision. Measured against `statistics.NormalDist().inv_cdf` over
#: `p in [1e-300, 1-1e-300]`, five steps give a maximum absolute error of 2.8e-14 and are
#: strictly increasing; four leave 6.8e-10 at `|x| > 20`.
_PPF_ITERATIONS = 5

#: Probabilities are clamped strictly inside ``(0, 1)`` so the inverse is always finite
#: rather than handing back ``+/-inf`` from a ``log(0)``.
#:
#: The clamp is **asymmetric, and unavoidably so**: the lower tail reaches ``1e-300``
#: (quantile ~ -37), but ``1 - 1e-300`` is not representable — the nearest double below 1.0
#: is ``1 - 1.1e-16`` — so the upper tail stops at a quantile of about 8.2. Nothing depends
#: on the deeper half: the unbounded *edges* of the outer regions are handled in latent
#: space by :func:`_region_target_mass` and :meth:`SamplingRegion.latent_bounds`, where
#: ``+/-inf`` is passed to ``erfc`` directly and the mass is exact.
_PPF_FLOOR = 1e-300
_PPF_CEIL = float(np.nextafter(1.0, 0.0))


def _standard_normal_cdf(x: np.ndarray) -> np.ndarray:
    """``Phi(x)`` from libm's ``erfc``, for the iteration in :func:`norm_ppf`.

    ``erfc`` is accurate to about one ulp *relatively*, which is exactly the property the
    Halley step needs: the error in the residual then scales with the residual instead of
    with 1, so the tail is not degraded relative to the bulk.
    """
    flat = np.ravel(np.asarray(x, dtype=float))
    values = np.fromiter((0.5 * math.erfc(-v / _SQRT2) for v in flat), dtype=float, count=flat.size)
    return values.reshape(np.shape(x))


def norm_ppf(p):
    """Vectorized standard-normal quantile function, ``Phi^-1``, accurate to ~1e-14.

    Solved as ``Phi(x) = p`` by Halley iteration,

        x <- x - 2u / (2 + x u),      u = (Phi(x) - p) / phi(x),

    from the analytic start ``x0 = -sign(1/2 - p) * sqrt(-2 ln(2 min(p, 1-p)))``, whose error
    is bounded by roughly 0.5 everywhere over the whole open interval.

    **The upper half is solved as a mirrored lower half.** For ``p > 0.5``, ``Phi(x)`` is a
    number just below 1, so forming ``Phi(x) - p`` subtracts two nearly-equal values and
    throws away every significant digit — at ``p = 1 - 1e-16`` it retains none. Inverting
    ``1 - p`` and negating keeps the residual a difference of two small numbers throughout,
    which is the difference between 1e-14 and no accuracy at all in the upper tail.

    Implemented here rather than imported because SciPy is not a dependency of
    `gammaforge.io`, and `statistics.NormalDist.inv_cdf` — which *is* correct, and is what
    this is validated against — is scalar-only.
    """
    p = np.clip(np.asarray(p, dtype=float), _PPF_FLOOR, _PPF_CEIL)
    lower = p <= 0.5
    # Solve only ever for a tail probability <= 0.5, then mirror.
    x = -np.sqrt(-2.0 * np.log(2.0 * np.where(lower, p, 1.0 - p)))
    for _ in range(_PPF_ITERATIONS):
        u = (_standard_normal_cdf(x) - np.where(lower, p, 1.0 - p)) / (
            np.exp(-0.5 * x * x) / _SQRT_2PI
        )
        x = x - 2.0 * u / (2.0 + x * u)
    return np.where(lower, x, -x)


def _normal_interval_mass(lo: float, hi: float) -> float:
    """``Phi(hi) - Phi(lo)`` for standard normals, evaluated without catastrophic cancellation.

    ``erfc`` is accurate to about one ulp *relatively* for arguments of either sign, but
    ``Phi(x) = erfc(-x/sqrt2)/2`` and its complement ``Q(x) = erfc(x/sqrt2)/2`` are each
    accurate only on **one** side: ``Phi`` on ``x <= 0``, ``Q`` on ``x >= 0``. The
    difference of the wrong pair is the difference of two numbers both near 2, which is where
    all the digits go:

    * bounds wholly in the **upper** tail need the ``Q`` pair, ``Q(lo) - Q(hi)``;
    * bounds wholly in the **lower** tail need the ``Phi`` pair — using ``Q`` there returns
      **exactly zero** for, say, ``(-38, -36)``, whose true mass is 4e-284, because
      ``erfc(-26.9)`` and ``erfc(-25.5)`` are both 2.0 to the last bit.

    The stratifier's outer regions live in exactly those tails, and a silent zero there is a
    region silently contributing no mass — the one failure mode with no compensating
    invariant. The branch costs one comparison.
    """
    if lo >= 0.0:
        return 0.5 * (math.erfc(lo / _SQRT2) - math.erfc(hi / _SQRT2))
    return 0.5 * (math.erfc(-hi / _SQRT2) - math.erfc(-lo / _SQRT2))


# ---------------------------------------------------------------------------
# Low-discrepancy sequence
# ---------------------------------------------------------------------------
def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    """The van der Corput radical inverse in ``base``, vectorized over ``indices``.

    ``phi_b(i)`` writes ``i`` in base ``b``, reflects the digits, and reads them back as a
    fraction. The digit count is fixed at the width of the largest index rather than being
    found per element, so this is one pass of vectorized arithmetic per base.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.zeros(0, dtype=float)
    max_index = int(indices.max())
    digits = 1
    while base**digits <= max_index:
        digits += 1
    result = np.zeros(indices.shape, dtype=float)
    work = indices.copy()
    factor = 1.0 / base
    for _ in range(digits):
        result += factor * (work % base)
        work //= base
        factor /= base
    return result


def _stream_shift(seed: int, region_id: int, stream: str, n_dims: int = N_LATENT) -> np.ndarray:
    """A deterministic Cranley-Patterson shift in ``[0, 1)`` per dimension.

    The shift depends on the global seed, the region id and the stream tag, so the pilot
    and the production draw in one region are *not* the same points — a pilot that reused
    production points would bias the allocation towards regions whose points happened to
    land well. ``SeedSequence`` takes a list of integers as entropy, which makes this a
    pure function of the three values with no shared mutable RNG state anywhere.
    """
    tag = 0 if stream == "pilot" else 1
    sequence = np.random.SeedSequence([int(seed), int(region_id), tag])
    return np.random.default_rng(sequence).random(n_dims)


def _halton_points(count: int, seed: int, region_id: int, stream: str) -> np.ndarray:
    """``(count, 6)`` low-discrepancy points in the open unit cube, shifted modulo one.

    **Index starts at 1.** The index-0 point of a Halton sequence is the origin, and
    mapping it through the conditional inverse-CDF would put a particle on the region's
    lower edge in every coordinate at once — a point of the target that carries strictly
    less than a full share of the box's mass.

    **Prefix stability.** Point ``k`` does not depend on ``count``, so asking a region for
    more particles later *extends* its sequence instead of moving every existing point.
    That is what makes multiplicative refinement ``n'_m = K n_m`` exact rather than
    approximate (§19).
    """
    indices = np.arange(1, count + 1, dtype=np.int64)
    points = np.empty((count, N_LATENT), dtype=float)
    for axis, base in enumerate(HALTON_BASES):
        points[:, axis] = _radical_inverse(indices, base)
    shift = _stream_shift(seed, region_id, stream)
    # Modulo one maps the [0,1) van der Corput values plus a fixed shift onto [0,1),
    # keeping the point in the open interval as long as the sum is not an exact integer.
    return np.mod(points + shift, 1.0)


def _latent_bounds(lo: np.ndarray, hi: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """A cube box's edges in **latent** coordinates: ``s * Phi^-1`` of each cube coordinate.

    The cube's own faces are the CDF endpoints 0 and 1, whose quantiles are exact infinities;
    they become exact infinities here rather than the finite clip :func:`norm_ppf` applies,
    so the outermost regions keep their true unbounded extent and their exact mass.
    """
    lo_latent = scale * norm_ppf(lo)
    hi_latent = scale * norm_ppf(hi)
    return (
        np.where(lo <= 0.0, -np.inf, lo_latent),
        np.where(hi >= 1.0, np.inf, hi_latent),
    )


def _conditional_deviates(
    lo: np.ndarray,
    hi: np.ndarray,
    scale: float,
    count: int,
    seed: int,
    region_id: int,
    stream: str,
) -> np.ndarray:
    """``(count, 6)`` deviates drawn exactly from the target conditional ``p(d | R)``.

    **The cube coordinates are not the target's CDF coordinates, and using them as if they
    were is a silent bias.** A box is axis-aligned in the *reference* CDF coordinate
    ``u = Phi(d / s)``, so in target-CDF coordinates its edges are

        A_j = Phi(s Phi^-1(a_j)),      H_j = Phi(s Phi^-1(b_j)),

    and these equal ``a_j``/``b_j`` only when ``s == 1``. Since the shipped default is
    ``s = sqrt(2)``, drawing ``d_j = Phi^-1[a_j + v_j (b_j - a_j)]`` samples from the
    *reference* conditional instead of the target one. The symptom is not a wrong weight sum
    -- ``sum(P_m)`` is computed from the correct latent bounds and stays 1 -- but a
    represented distribution that is biased in a way no invariant test catches, because the
    error is a distortion of *where* the mass sits rather than of how much of it there is.

    Given a low-discrepancy ``v``, the map ``d_j = Phi^-1[A_j + v_j (H_j - A_j)]`` is an
    exact stratified draw from ``p(d | R_m)``: the target factorizes, the box is
    axis-aligned in latent space, and no importance weighting is involved at all — so the
    density ratio ``p / q_s`` never contributes variance. The region therefore carries the
    single exact constant weight ``P_m / n_m``.

    ``A`` and ``H`` are formed from ``erfc`` rather than by pushing the cube coordinates
    through ``Phi(s Phi^-1(.))``, which would cancel catastrophically in the tails.
    """
    latent_lo, latent_hi = _latent_bounds(lo, hi, scale)
    # `erfc` of an infinite argument gives exactly 0 and 2, so a box face maps to exactly 0
    # and 1 without a special case.
    lower_cdf = _standard_normal_cdf(latent_lo)
    upper_cdf = _standard_normal_cdf(latent_hi)
    local = _halton_points(count, seed, region_id, stream)
    return norm_ppf(np.clip(lower_cdf + local * (upper_cdf - lower_cdf), _PPF_FLOOR, _PPF_CEIL))


# ---------------------------------------------------------------------------
# Reference-cube partitioning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Node:
    """A cube region together with how many leaves it must become."""

    lo: np.ndarray
    hi: np.ndarray
    leaves: int


def _split_node(node: _Node) -> tuple[_Node, _Node]:
    """Split a box's longest side so the left child gets ``floor(m/2)`` of the leaves.

    ``np.argmax`` breaks a tie toward the lowest axis index, so a box with two equal
    longest sides always splits the same way for the same input — the partition has to be
    a function of the requested leaf count alone, or "same plan, same result" fails.
    """
    left_leaves = node.leaves // 2
    widths = node.hi - node.lo
    axis = int(np.argmax(widths))
    cut = node.lo[axis] + (left_leaves / node.leaves) * widths[axis]
    lo_l, hi_l = node.lo.copy(), node.hi.copy()
    hi_l[axis] = cut
    lo_r, hi_r = node.lo.copy(), node.hi.copy()
    lo_r[axis] = cut
    return _Node(lo_l, hi_l, left_leaves), _Node(lo_r, hi_r, node.leaves - left_leaves)


def _subdivide(node: _Node) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leaves of the balanced binary partition of ``node`` into exactly ``node.leaves`` boxes.

    A node assigned ``m`` descendants splits into ``floor(m/2)`` and ``m - floor(m/2)``.
    That is what makes an **arbitrary** leaf count work without overlap: a recursive
    longest-side split at a fixed fraction is exact for any ``m``, not just powers of two.
    """
    if node.leaves == 1:
        return [(node.lo, node.hi)]
    left, right = _split_node(node)
    return _subdivide(left) + _subdivide(right)


def _reference_partition(n_leaves: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The deterministic partition of ``[0, 1]^6`` into ``n_leaves`` disjoint boxes."""
    if n_leaves < 1:
        raise ValueError(f"partition leaf count must be >= 1, got {n_leaves}")
    return _subdivide(_Node(np.zeros(N_LATENT), np.ones(N_LATENT), n_leaves))


def _split_box(lo: np.ndarray, hi: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Halve a leaf box, for the adaptive refinement of §20.2.

    This is :func:`_subdivide` at ``m = 2``: one longest-side cut at the midpoint. The two
    children tile the parent exactly, so splitting never changes the represented support
    or the total target mass — only the resolution within it.
    """
    return _subdivide(_Node(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), 2))


# ---------------------------------------------------------------------------
# Stage-0-like luminosity predictor
# ---------------------------------------------------------------------------
#: Gauss-Legendre nodes and weights on [-1, 1], cached. The rule is fixed, so recomputing
#: it per region (the plan pilots one region at a time) would be pure waste.
_LEGENDRE_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _legendre_rule(n_quad: int, panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes and weights on the unit interval, with ``dt/dx`` folded in.

    **Why composite rather than a single high-order rule.** The integration domain is
    `overlap_time_window`, a *conservative geometric* bound: it is deliberately much wider
    than the stretch in which a particle actually sees the pulse, because a bound may only
    ever err towards keeping particles. So a fixed ``n``-node rule spreads its nodes over a
    window the pulse occupies only a small part of, and under-resolves exactly the peak
    that dominates the answer. Measured against over-resolved Stage 0, a single 64-node
    rule reproduces the *coarse* luminosity ordering well but the ordering **among the
    brightest particles** — which is what the allocation actually keys on — barely at all.

    Splitting the same node budget into a few panels fixes the resolution without shrinking
    the domain, so truncation stays impossible: a composite rule over a conservative bound
    is still a bound. Two panels at 64 nodes each takes that ordering from 0.35 to 0.84
    (mean Spearman over five geometries), and to 0.94 at 128 nodes.

    ``panels`` trades against nodes per panel: past roughly four panels each panel gets too
    few nodes and accuracy falls again, so the default is deliberately small.
    """
    if panels < 1:
        raise ValueError(f"_legendre_rule: panels must be >= 1, got {panels}")
    per_panel = n_quad // panels
    if per_panel < 4:
        raise ValueError(
            f"_legendre_rule: n_quad={n_quad} over {panels} panels leaves {per_panel} nodes per "
            f"panel, too few for Gauss-Legendre to mean anything; use fewer panels or more nodes"
        )
    key = (n_quad, panels)
    if key not in _LEGENDRE_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(per_panel)
        # A tiny floor keeps a zero weight from ever producing 0 * inf = NaN where the
        # trajectory leaves the modelled region. Legendre weights are strictly positive, so
        # this can only clip a node that is already 1e-17 wide.
        weights = np.maximum(weights, np.finfo(float).tiny)
        edges = np.linspace(0.0, 1.0, panels + 1)
        centers, scaled = [], []
        for index in range(panels):
            low, high = edges[index], edges[index + 1]
            midpoint, half = 0.5 * (low + high), 0.5 * (high - low)
            centers.append(midpoint + half * nodes)
            scaled.append(half * weights)
        _LEGENDRE_CACHE[key] = (np.concatenate(centers), np.concatenate(scaled))
    return _LEGENDRE_CACHE[key]


#: Particles per batch in :func:`trajectory_luminosity_predictor`. The quadrature evaluates
#: an ``(n_particles, n_quad)`` field grid, which at 100k particles and 64 nodes is several
#: hundred MB across the handful of intermediates. Batching bounds the peak at a few MB
#: whatever the caller passes, so the pilot is safe on a full bunch and not just on its own
#: small subsets.
_PREDICT_BATCH = 16384


def _encounter_factor(thx, thy, n0):
    """``F = 1 - e.n0_hat`` for ultrarelativistic particles, the Stage-0 encounter factor.

    Stage 0 reaches this as ``(1 - k_hat[2]) * direction_doppler_factor(...)``; the
    nominal-axis factor cancels, leaving exactly this. Written directly so the pilot and
    the stage cannot disagree about which of the two factors is which.
    """
    norm = np.sqrt(1.0 + thx**2 + thy**2)
    return 1.0 - (n0[0] * thx + n0[1] * thy + n0[2]) / norm


def _luminosity_block(x, y, z, thx, thy, laser, n0, omega0, offsets, weights, t0, span):
    """``F * Int C(t) I(t) dt`` for one batch, matching Stage 0's luminosity definition.

    The integrand and the carrier ratio are Stage 0's own: the cycle-averaged
    ``<a^2>`` from ``intensity_profile``, the encounter ratio
    ``C = 1 + (d_t + v.grad deltaPhi) / (omega0 F)`` from the four-gradient, and the same
    "non-contributing samples get C = 1" and positivity conventions. What is *not*
    reproduced is the quadrature rule — Gauss-Legendre here against Stage 0's fixed
    midpoint — and the common factors ``density_scale * c * sigma_T``, which are the same
    for every particle and so carry no information about where to spend the budget.
    """
    encounter = _encounter_factor(thx, thy, n0)
    if np.any(encounter <= 64.0 * np.finfo(float).eps):
        raise ValueError(
            "electron-laser encounter factor is too small; co-propagation is outside xigma's regime"
        )
    # An empty window (t0 = +inf, t1 = -inf) is anchored at a finite time; its zero span
    # then makes every dt zero, so the particle contributes nothing — the same treatment
    # Stage 0 gives it, for the same reason.
    start = np.where(span > 0.0, t0, 0.0)
    times = start[:, None] + offsets * span[:, None]
    dts = weights * (0.5 * span)[:, None]

    norm = np.sqrt(1.0 + thx**2 + thy**2)
    vx, vy, vz = (C_CGS * thx / norm, C_CGS * thy / norm, C_CGS / norm)
    positions = (
        x[:, None] + vx[:, None] * times,
        y[:, None] + vy[:, None] * times,
        z[:, None] + vz[:, None] * times,
    )
    intensity = np.asarray(laser.intensity_profile(*positions, times), dtype=float)
    gradient = laser.carrier_phase_four_gradient(*positions, times)
    if not isinstance(gradient, (tuple, list)) or len(gradient) != 4:
        raise TypeError("carrier_phase_four_gradient must return (d_t, d_x, d_y, d_z)")
    d_t, d_x, d_y, d_z = np.broadcast_arrays(
        *(np.asarray(component, dtype=float) for component in gradient)
    )
    carrier = 1.0 + (
        d_t + vx[:, None] * d_x + vy[:, None] * d_y + vz[:, None] * d_z
    ) / (omega0 * encounter[:, None])
    contributing = (intensity > 0.0) & (dts > 0.0)
    if np.any(contributing & ((carrier <= 0.0) | ~np.isfinite(carrier))):
        raise ValueError(
            "encountered carrier phase ratio C must be finite and positive over contributing samples"
        )
    carrier = np.where(contributing, carrier, 1.0)
    return encounter * np.sum(dts * carrier * intensity, axis=1)


def trajectory_luminosity_predictor(
    bunch: Bunch,
    laser,
    *,
    n_quad: int = 128,
    panels: int = 2,
    threshold: float = 1e-8,
) -> np.ndarray:
    """Per-particle score proportional to the Stage-0 luminosity, without weights or ``N_e``.

    Returns :math:`\\ell(X) = F \\int C(t) I(\\mathbf r(t), t)\\,dt` for each particle —
    the same physical relevance Stage 0 integrates, evaluated on a cheap Gauss-Legendre
    rule over :func:`~gammaforge.io.bunch.overlap_time_window` instead of Stage 0's fixed
    midpoint rule.

    **The carrier factor is not optional.** For a chirped or otherwise phase-structured
    pulse, :math:`C(t) \\ne 1` and an intensity-only score systematically mis-ranks regions;
    the built-in unchirped fields return a zero four-gradient, so there it reduces to
    :math:`F \\int I\\,dt` with no change in behaviour.

    **Why `overlap_time_window` and not `illumination_window`.** The former is a *conservative
    bound* — every node outside the pulse simply evaluates to zero intensity. The latter is
    an estimate of where illumination actually happens, and an estimate is not a safe
    integration domain: using it would bias the integral towards whichever particles the
    frozen-width approximation happened to bracket well.

    Being approximate is allowed and expected — it costs :math:`O(n \\times 64)` field
    evaluations against Stage 0's :math:`O(n \\times 200)`, and its error changes only how
    much budget a region receives, never the mass the region represents.
    """
    if n_quad < 2:
        raise ValueError(f"trajectory_luminosity_predictor: n_quad must be >= 2, got {n_quad}")
    t0, t1 = overlap_time_window(bunch, laser, threshold)
    span = np.maximum(0.0, t1 - t0)
    offsets, weights = _legendre_rule(n_quad, panels)
    n0 = laser_propagation_direction(laser)
    omega0 = float(laser.omega0())

    n_particles = bunch.n_particles
    out = np.zeros(n_particles, dtype=float)
    for begin in range(0, n_particles, _PREDICT_BATCH):
        stop = min(begin + _PREDICT_BATCH, n_particles)
        out[begin:stop] = _luminosity_block(
            bunch.x[begin:stop], bunch.y[begin:stop], bunch.z[begin:stop],
            bunch.thx[begin:stop], bunch.thy[begin:stop],
            laser, n0, omega0, offsets, weights, t0[begin:stop], span[begin:stop],
        )
    return out


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PilotConfig:
    """Numerical knobs of the plan build. **Starting values, not physics constants.**

    Every field here is a number a benchmark can defend, and §20 asks for exactly these
    starting points. They are named rather than hard-coded so a sweep is a dict, and they
    live here rather than on the plan so there is one place a value can come from.
    """

    #: Regions in the initial balanced partition, before any adaptive refinement.
    initial_regions: int = 64
    #: Ceiling on the leaf count. Refinement stops here; it never removes a leaf.
    max_regions: int = 256
    #: Pilot points drawn per region to estimate its luminosity moments.
    pilot_points_per_region: int = 8
    #: Total Gauss-Legendre nodes per pilot trajectory. 128 rather than 64 because the
    #: pilot only ever evaluates ``regions x points_per_region`` particles (~2e3), so the
    #: extra nodes cost ~0.1 s against Stage 0's per-particle cost, and they measurably
    #: improve the ordering among the brightest particles -- the allocation's input.
    pilot_quad_nodes: int = 128
    #: Composite panels the node budget is split over; see `_legendre_rule`.
    pilot_quad_panels: int = 2
    #: Active-region threshold (a fraction of peak a0) bounding the pilot's window.
    pilot_window_threshold: float = 1e-8
    #: Broad reference Gaussian's scale, ``q_s = N(0, s^2 I_6)``. ``s >= 1`` is what makes
    #: the reference cover the target; the weight-ratio bound is ``s^6 / (1 - lambda)``.
    proposal_scale: float = math.sqrt(2.0)
    #: Blend of broad coverage against luminosity-driven allocation, §17.
    luminosity_fraction: float = 0.75

    def __post_init__(self) -> None:
        if self.initial_regions < 1:
            raise ValueError(f"PilotConfig: initial_regions must be >= 1, got {self.initial_regions}")
        if self.max_regions < self.initial_regions:
            raise ValueError(
                f"PilotConfig: max_regions ({self.max_regions}) must be >= initial_regions "
                f"({self.initial_regions})"
            )
        if self.pilot_points_per_region < 1:
            raise ValueError(
                f"PilotConfig: pilot_points_per_region must be >= 1, got {self.pilot_points_per_region}"
            )
        if self.pilot_quad_nodes < 2:
            raise ValueError(f"PilotConfig: pilot_quad_nodes must be >= 2, got {self.pilot_quad_nodes}")
        if self.pilot_quad_panels < 1:
            raise ValueError(
                f"PilotConfig: pilot_quad_panels must be >= 1, got {self.pilot_quad_panels}"
            )
        if self.pilot_quad_nodes // self.pilot_quad_panels < 4:
            raise ValueError(
                f"PilotConfig: pilot_quad_nodes={self.pilot_quad_nodes} over "
                f"pilot_quad_panels={self.pilot_quad_panels} leaves too few nodes per panel"
            )
        if self.proposal_scale < 1.0:
            raise ValueError(
                f"PilotConfig: proposal_scale must be >= 1 so the reference Gaussian covers the "
                f"target, got {self.proposal_scale}"
            )
        if not 0.0 <= self.luminosity_fraction <= 1.0:
            raise ValueError(
                f"PilotConfig: luminosity_fraction must be in [0, 1], got {self.luminosity_fraction}"
            )


#: The shipped defaults. A plan records the config it was built with, so this is the value
#: a plan actually used even if the default is later retuned.
DEFAULT_PILOT_CONFIG = PilotConfig()


@dataclass(frozen=True)
class SamplingRegion:
    """One box of the reference cube, with its exact masses and pilot statistics.

    ``reference_mass`` is :math:`B_m`, the cube volume, i.e. the region's probability under
    the broad reference Gaussian :math:`q_s`. ``target_mass`` is :math:`P_m`, its **exact**
    probability under the physical target — the fraction of the electron bunch the region
    represents, and the only quantity the production weights depend on.

    ``allocation_probability`` is :math:`Q_m`, the mixture the integer apportionment draws
    from. It is strictly positive for every region (``Q_m >= (1 - lambda) B_m``), which is
    the mechanical reason a pilot can never delete a region.

    ``proposal_scale`` is stored on the region rather than passed to :meth:`deviates`
    because it is needed to interpret the bounds at all: ``lo``/``hi`` are in the *broad
    reference* Gaussian's CDF coordinates, and the target conditional in latent space is
    only defined once the scale that maps between the two is known. Keeping it here makes
    the region self-contained, so a caller cannot accidentally sample one region's points
    with another region's scale.
    """

    id: int
    lo: np.ndarray
    hi: np.ndarray
    proposal_scale: float
    reference_mass: float
    target_mass: float
    pilot_mean: float
    pilot_second_moment: float
    pilot_std: float
    allocation_probability: float

    def __post_init__(self) -> None:
        for name in ("lo", "hi"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.shape != (N_LATENT,):
                raise ValueError(f"SamplingRegion: {name} must have shape ({N_LATENT},), got {values.shape}")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def latent_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The region's box in latent coordinates, ``s * Phi^-1`` of the cube edges.

        Unbounded at the cube's faces, which is what keeps the partition free of any finite
        Gaussian cutoff: the outermost regions reach infinitely far into the tails and still
        carry their exact, non-zero mass.
        """
        return _latent_bounds(self.lo, self.hi, self.proposal_scale)

    def target_mass_exact(self, proposal_scale: float | None = None) -> float:
        """``P_m`` recomputed from the latent bounds, as the independent check on the field.

        :attr:`target_mass` is what the plan uses; this recomputes it the long way, from the
        actual latent bounds rather than reusing the stored reference mass, so a test can
        confirm the two agree rather than merely restating one in terms of the other.
        """
        lo_bound, hi_bound = self.latent_bounds
        return float(
            np.prod([_normal_interval_mass(float(a), float(b)) for a, b in zip(lo_bound, hi_bound)])
        )

    def deviates(self, count: int, seed: int, stream: str) -> np.ndarray:
        """``(count, 6)`` latent deviates drawn from the **target conditional** ``p(d|R)``.

        Delegates to :func:`_conditional_deviates`, which is the only place the target
        conditional is formed. See there for why the cube coordinates may **not** be used as
        the target's CDF coordinates — the difference is silent at ``s = 1`` and is a real
        bias for every other value, including the shipped default.
        """
        if count < 1:
            raise ValueError(f"SamplingRegion.deviates: count must be >= 1, got {count}")
        return _conditional_deviates(self.lo, self.hi, self.proposal_scale, count, seed, self.id, stream)


@dataclass(frozen=True)
class AdaptiveSamplingPlan:
    """A reusable, observer-independent sampling plan for an analytic beam.

    Built once from ``(beam, laser, seed)`` and then **reusable for any particle count**:
    :meth:`allocate` re-derives the integer apportionment from the stored ``Q_m`` without
    touching the pilot again, and :meth:`AdaptiveSamplingPlan.deviates` gives each region a
    prefix-stable low-discrepancy sequence, so asking for ``K`` times as many particles
    extends the existing points instead of moving them.

    Depends on the beam, the laser and the seed only. The observer-dependent ponderomotive
    coefficient :math:`Q` lives in Stage 2, so one plan serves every observation direction.
    """

    seed: int
    regions: tuple[SamplingRegion, ...]
    pilot_config: PilotConfig

    @property
    def proposal_scale(self) -> float:
        """The reference Gaussian's scale ``s`` (§5)."""
        return self.pilot_config.proposal_scale

    @property
    def luminosity_fraction(self) -> float:
        """The coverage/luminosity blend ``lambda`` (§17)."""
        return self.pilot_config.luminosity_fraction

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def allocate(self, n_particles: int) -> np.ndarray:
        """Per-region production counts summing to exactly ``n_particles``, each ``>= 1``.

        Largest-remainder apportionment of the ``n_particles - M`` surplus over ``Q_m``.
        Every region gets one particle unconditionally, so the budget floor is
        ``M <= n_particles``; asking for fewer particles than regions is an error rather
        than a silent truncation, because a region with no particle would silently
        contribute zero instead of its (small) exact mass.
        """
        if isinstance(n_particles, bool) or not isinstance(n_particles, (int, np.integer)):
            raise ValueError(f"allocate: n_particles must be an integer, got {n_particles!r}")
        n_particles = int(n_particles)
        n_regions = len(self.regions)
        if n_particles < n_regions:
            raise ValueError(
                f"allocate: n_particles ({n_particles}) must be >= the number of regions "
                f"({n_regions}) — every region needs at least one particle to represent its mass"
            )
        share = np.array([r.allocation_probability for r in self.regions], dtype=float)
        share /= share.sum()
        surplus = n_particles - n_regions
        exact = surplus * share
        counts = np.floor(exact).astype(np.int64)
        # Flooring loses at most one particle per region, so this is a small non-negative
        # integer; the stable sort hands the remainder to the largest fractional parts with
        # index order as the tie-break, which keeps the apportionment a pure function of
        # (plan, n_particles) — no dependence on numpy's internal ordering of equal keys.
        remainder = int(surplus - counts.sum())
        if remainder:
            fractions = exact - counts
            order = np.argsort(-fractions, kind="stable")[:remainder]
            counts[order] += 1
        return counts + 1

    def deviates(self, n_particles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(deviates, weight, region_index)`` for exactly ``n_particles`` production particles.

        Each particle's weight is :math:`P_m / n_m`, so the weights sum to
        :math:`\\sum_m P_m = 1` **by construction and with no renormalization pass** —
        which is the property that keeps this bunch interchangeable with an IID one, and
        that a self-normalizing implementation would quietly destroy.
        """
        counts = self.allocate(n_particles)
        per_region = [
            region.deviates(int(count), self.seed, "production")
            for region, count in zip(self.regions, counts)
        ]
        target_mass = np.array([r.target_mass for r in self.regions], dtype=float)
        return (
            np.concatenate(per_region, axis=0),
            np.repeat(target_mass / counts, counts),
            np.repeat(np.arange(len(self.regions), dtype=np.int64), counts),
        )

    def diagnostics(self, n_particles: int, counts: np.ndarray | None = None) -> dict:
        """Enough to reproduce and explain a run's sampling, per §25.

        Deliberately aggregate: the pilot particles themselves are not dumped, since a
        256-region plan's raw pilot output would dominate the metadata it is meant to
        summarize.
        """
        if counts is None:
            counts = self.allocate(n_particles)
        per_particle = np.repeat(
            np.array([r.target_mass for r in self.regions], dtype=float) / counts, counts
        )
        total = float(per_particle.sum())
        # N_eff = (sum w)^2 / sum w^2. It is the honest measure of how many equally-weighted
        # particles this set is worth, and is the number to watch when a very uneven
        # allocation costs more variance than it saves.
        effective = total**2 / float(np.sum(per_particle**2)) if per_particle.size else 0.0
        return {
            "strategy": "adaptive",
            "seed": self.seed,
            "n_particles": int(n_particles),
            "proposal_scale": self.proposal_scale,
            "luminosity_fraction": self.luminosity_fraction,
            "initial_regions": self.pilot_config.initial_regions,
            "final_regions": self.n_regions,
            "pilot_points_per_region": self.pilot_config.pilot_points_per_region,
            "pilot_quad_nodes": self.pilot_config.pilot_quad_nodes,
            "pilot_quad_panels": self.pilot_config.pilot_quad_panels,
            "pilot_threshold": self.pilot_config.pilot_window_threshold,
            "region_counts_min": int(counts.min()),
            "region_counts_max": int(counts.max()),
            "region_counts_median": float(np.median(counts)),
            "weight_min": float(per_particle.min()),
            "weight_max": float(per_particle.max()),
            "weight_quantiles": [float(q) for q in np.quantile(per_particle, (0.05, 0.5, 0.95))],
            "weight_sum": total,
            "n_eff": effective,
            "target_mass_sum": float(sum(r.target_mass for r in self.regions)),
            "reference_mass_sum": float(sum(r.reference_mass for r in self.regions)),
            "luminosity_fraction_estimate": {
                r.id: r.pilot_mean * r.target_mass for r in self.regions
            },
        }


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def _region_target_mass(lo: np.ndarray, hi: np.ndarray, scale: float) -> tuple[float, float]:
    """``(B_m, P_m)`` for one box: its cube volume and its **exact** target mass.

    ``B_m`` is the cube volume because the reference CDF coordinate is uniform under
    ``q_s`` — the ``s`` never enters it.

    ``P_m`` is the product of six exact normal interval masses over the box's latent
    bounds ``s * Phi^-1(edge)``. Because the boxes tile the cube, and ``u -> s Phi^-1(u)``
    is a monotone bijection onto the real line, the boxes tile all of latent space and the
    masses necessarily sum to one. That is an identity of the construction, not a
    numerical approximation — which is exactly why the test can assert ``sum(P_m) == 1``.
    """
    reference = float(np.prod(hi - lo))
    lo_latent, hi_latent = _latent_bounds(
        np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), scale
    )
    target = float(
        np.prod([_normal_interval_mass(float(a), float(b)) for a, b in zip(lo_latent, hi_latent)])
    )
    return reference, target


def _pilot_moments(
    region_lo: np.ndarray,
    region_hi: np.ndarray,
    region_id: int,
    beam: GaussianElectronBeam,
    laser,
    config: PilotConfig,
    seed: int,
) -> tuple[float, float, float]:
    """``(mean, second moment, std)`` of the pilot luminosity inside one region.

    The pilot points are drawn from the **target conditional** in that region, with the
    ``"pilot"`` stream tag so they never coincide with production points. An IID draw here
    would have been defensible too — the moments are unbiased either way — but reusing the
    production sequence would make a region look good or bad according to how its *chosen*
    points happened to fall, which is precisely the bias the allocation is supposed to
    avoid.
    """
    count = config.pilot_points_per_region
    probe = _bunch_from_standard_deviates(
        beam,
        _conditional_deviates(
            np.asarray(region_lo, dtype=float), np.asarray(region_hi, dtype=float),
            config.proposal_scale, count, seed, region_id, "pilot",
        ),
        np.full(count, 1.0 / count),
    )
    luminosity = trajectory_luminosity_predictor(
        probe, laser, n_quad=config.pilot_quad_nodes, panels=config.pilot_quad_panels,
        threshold=config.pilot_window_threshold,
    )
    mean = float(np.mean(luminosity)) if luminosity.size else 0.0
    second = float(np.mean(luminosity**2)) if luminosity.size else 0.0
    return mean, second, math.sqrt(max(second - mean * mean, 0.0))


def _make_region(
    region_id: int, lo, hi, scale: float, moments: tuple[float, float, float], share: float
) -> SamplingRegion:
    reference, target = _region_target_mass(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), scale)
    mean, second, std = moments
    return SamplingRegion(
        id=region_id, lo=lo, hi=hi, proposal_scale=scale,
        reference_mass=reference, target_mass=target,
        pilot_mean=mean, pilot_second_moment=second, pilot_std=std, allocation_probability=share,
    )


def _allocation_shares(regions: list[SamplingRegion], luminosity_fraction: float) -> np.ndarray:
    """``Q_m = (1 - lambda) B_m + lambda L_m``, normalized, for the given regions.

    ``L_m`` uses the **second moment** rather than the standard deviation. Plain Neyman
    allocation ``n_m ~ P_m sigma_m`` is optimal only for the variance of the *total yield*,
    and the total yield is not what GammaForge needs: a region whose luminosity is nearly
    constant can still require many particles because its samples land in different
    ``(gamma, theta_x, theta_y, a0_shape, chirp_mean)`` Stage-1 cells. ``P_m sqrt(M_2)``
    charges for that possibility, at the price of not being provably optimal for anything
    in particular.

    The ``(1 - lambda) B_m`` floor is the no-false-negatives guarantee. Because
    ``B_m > 0`` and ``L_m >= 0``, every ``Q_m > 0`` whenever ``lambda < 1``, and since
    ``P_m / B_m <= s^6`` the implied per-particle weight ratio stays bounded by
    ``s^6 / (1 - lambda)`` — 32 at the shipped defaults. If the pilot is so wrong that
    every luminosity vanishes, the allocation degrades to pure broad coverage rather than
    to nothing.
    """
    reference = np.array([r.reference_mass for r in regions], dtype=float)
    importance = np.array([r.target_mass * math.sqrt(r.pilot_second_moment) for r in regions])
    total = float(importance.sum())
    luminosity = reference.copy() if total <= 0.0 else importance / total
    share = (1.0 - luminosity_fraction) * reference + luminosity_fraction * luminosity
    return share / share.sum()


def build_adaptive_plan(
    beam: GaussianElectronBeam,
    laser,
    *,
    seed: int,
    config: PilotConfig | None = None,
) -> AdaptiveSamplingPlan:
    """Build the plan: partition, pilot, refine, and compute the allocation shares.

    Refinement (§20.2) repeatedly splits the leaf with the largest
    :math:`R_m = P_m \\sigma_m` — the scalar-yield unresolved contribution — and pilots only
    the two new children, since the surviving leaves' statistics do not change when a
    *sibling* is split. It stops at ``max_regions`` and **never removes a leaf**.

    .. warning::

       ``R_m`` measures luminosity variation only. It does not know whether a region spreads
       across many Stage-1 cells, so a region with a flat but broadly-spreading luminosity
       can be judged less urgent than it is. This is a known limitation of the first
       implementation, not a rounding detail: the pilot already computes the Stage-1
       coordinates (§14) and a later refinement criterion should use them.
    """
    from .bunch import validate

    config = config or DEFAULT_PILOT_CONFIG
    validate(beam)
    scale = config.proposal_scale

    boxes = _reference_partition(config.initial_regions)
    regions: list[SamplingRegion] = []
    moments: dict[int, tuple[float, float, float]] = {}
    next_id = 0
    for lo, hi in boxes:
        region_id = next_id
        next_id += 1
        regions.append(_make_region(region_id, lo, hi, scale, (0.0, 0.0, 0.0), 0.0))
        moments[region_id] = _pilot_moments(np.asarray(lo, float), np.asarray(hi, float), region_id, beam, laser, config, seed)

    while len(regions) < config.max_regions:
        # `np.argmax` breaks ties toward the lowest position, and `regions` is kept ordered
        # by id, so which leaf gets split is a pure function of the inputs.
        priority = np.array([r.target_mass * moments[r.id][2] for r in regions])
        victim = regions[int(np.argmax(priority))]
        children = _split_box(victim.lo, victim.hi)
        regions.remove(victim)
        for child_lo, child_hi in children:
            region_id = next_id
            next_id += 1
            child = _make_region(region_id, child_lo, child_hi, scale, (0.0, 0.0, 0.0), 0.0)
            regions.append(child)
            moments[region_id] = _pilot_moments(
                np.asarray(child_lo, float), np.asarray(child_hi, float), region_id,
                beam, laser, config, seed,
            )

    # Write the pilot moments onto the regions *before* the allocation is computed.
    # `_allocation_shares` reads `pilot_second_moment` off the regions, so computing the
    # shares first and patching only the probability afterwards would leave every region
    # reporting a zero second moment — which makes the importance score identically zero,
    # sends `_allocation_shares` down its `total <= 0` fallback, and quietly reduces the whole
    # scheme to uniform allocation over `B_m` for every `luminosity_fraction`. The refinement
    # priority below reads the same dict, so only the *splitting* was ever luminosity-driven.
    regions = [
        replace(
            region,
            pilot_mean=moments[region.id][0],
            pilot_second_moment=moments[region.id][1],
            pilot_std=moments[region.id][2],
        )
        for region in regions
    ]
    share = _allocation_shares(regions, config.luminosity_fraction)
    final = tuple(
        replace(region, allocation_probability=float(share[i]))
        for i, region in enumerate(regions)
    )
    return AdaptiveSamplingPlan(seed=int(seed), regions=final, pilot_config=config)


def build_adaptive_bunch(
    beam: GaussianElectronBeam,
    laser,
    n_particles: int,
    seed: int,
    *,
    plan: AdaptiveSamplingPlan | None = None,
    config: PilotConfig | None = None,
) -> tuple[Bunch, AdaptiveSamplingPlan]:
    """Sample ``n_particles`` macroparticles by stratified adaptive sampling.

    Pass an existing ``plan`` to skip the pilot entirely — the plan depends only on
    ``(beam, laser, seed)``, so reusing it across a sweep of particle counts is the
    intended path, and it is what makes "how many particles does this accuracy need?"
    answerable without paying for a fresh pilot at every point. The returned plan is
    returned again whether or not one was passed, so a caller can capture it on the first
    call and hand it back on the rest.

    The result is an ordinary :class:`~gammaforge.io.bunch.Bunch` with per-particle
    relative weights; nothing downstream needs to know which strategy produced it.
    """
    started = time.perf_counter()
    if plan is None:
        plan = build_adaptive_plan(beam, laser, seed=seed, config=config)
    elif plan.seed != int(seed):
        raise ValueError(
            f"build_adaptive_bunch: plan was built for seed {plan.seed}, got {seed}. A plan's "
            f"low-discrepancy shifts are seeded, so reusing it across seeds is not defined."
        )
    build_seconds = time.perf_counter() - started

    deviates, weight, region_index = plan.deviates(n_particles)
    meta = plan.diagnostics(n_particles)
    meta["pilot_build_seconds"] = build_seconds
    bunch = _bunch_from_standard_deviates(beam, deviates, weight, meta=meta)
    return bunch, plan
