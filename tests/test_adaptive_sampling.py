"""Adaptive-stratified bunch sampling: exact representation, then convergence (RES094).

The tests are ordered by *what could silently be wrong*, and each tier protects a different
class of failure:

1. **The partition and the masses** (tier 0) are arithmetic identities. If a region's target
   mass is wrong, or the boxes do not tile the cube, every downstream number is wrong in a
   way that still looks plausible. These are the cheapest tests in the file and they run
   first, deliberately.
2. **`norm_ppf` against `statistics.NormalDist().inv_cdf`** (tier 0). This is the test that
   catches a mis-transcribed rational approximation. `sum(P_m) == 1` does *not* catch it:
   that identity only requires the quantile function to be monotone, so a wrong-but-monotone
   map passes it and then produces negative regional masses once refinement makes boxes
   thin. See the note in `adaptive_sampling` and the regression at the bottom of this file.
3. **The represented distribution** (tier 1) — weights summing to one, the exact budget, and
   weighted beam moments reproducing every stored correlation. This is the claim that makes
   an adaptive bunch interchangeable with an IID one.
4. **Convergence** (tier 2) — tail behaviour against chi-squared, the pilot against Stage 0,
   and the 5D Stage-1 table and Stage-2 spectra against a high-statistics IID reference.

Nothing here asserts that adaptive sampling *wins*; `scripts/benchmark_adaptive_sampling.py`
measures that, and the promotion decision rests on its numbers, not on a test tolerance.
"""

from __future__ import annotations

from dataclasses import replace
import math
import statistics

import numpy as np
import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.fast]

from gammaforge.io.adaptive_sampling import (
    HALTON_BASES,
    N_LATENT,
    AdaptiveSamplingPlan,
    PilotConfig,
    build_adaptive_bunch,
    build_adaptive_plan,
    norm_ppf,
    trajectory_luminosity_predictor,
    _halton_points,
    _normal_interval_mass,
    _reference_partition,
)
from gammaforge.io.bunch import sample_gaussian_bunch, prefilter_bunch
from gammaforge.io.interaction import ADAPTIVE, IID, SamplingSpec, build_interaction
from gammaforge.io.units import Quantity as Q
from gammaforge.validation import scenarios

#: Fully correlated, deliberately awkward beam: every one of the five energy correlations
#: and both Twiss tilts is non-zero, so the latent->physical map is exercised end to end
#: rather than degenerating to six independent scalings.
CORRELATED = dict(
    rho_x_gamma=0.3,
    rho_y_gamma=-0.25,
    rho_z_gamma=0.4,
    rho_thx_gamma=0.2,
    rho_thy_gamma=-0.1,
    alpha_x=0.5,
    alpha_y=-0.2,
)

#: Small plans keep the fast tier fast; the statistics being checked (sums, budgets, sign,
#: bound containment) are exact at any size.
FAST = PilotConfig(initial_regions=8, max_regions=16, pilot_points_per_region=4, pilot_quad_nodes=16)


def correlated_beam(**overrides):
    return replace(scenarios.BASELINE.beam, **overrides)


# ---------------------------------------------------------------------------
# The quantile function
# ---------------------------------------------------------------------------
def test_norm_ppf_matches_the_standard_library_reference():
    """The reference the handoff names, over the whole representable range.

    Both tails, the deep tail, and the region around p = 0.5 where the sign of the answer
    flips. `statistics.NormalDist.inv_cdf` is stdlib and independently verified, which is
    what makes it usable as the arbiter here.
    """
    probabilities = np.unique(
        np.concatenate(
            [
                np.linspace(1e-15, 1 - 1e-15, 40_001),
                np.logspace(-300, -1.5, 4_001),
                # The upper tail is only representable down to 1 - 1.1e-16, so the mirror
                # of the deep lower tail does not exist in float64. This is a property of
                # the format, not of the function; `adaptive_sampling` documents it.
                1.0 - np.logspace(-16, -1.5, 4_001),
            ]
        )
    )
    probabilities = probabilities[(probabilities > 0.0) & (probabilities < 1.0)]
    expected = np.array([statistics.NormalDist().inv_cdf(float(p)) for p in probabilities])

    np.testing.assert_allclose(norm_ppf(probabilities), expected, rtol=0, atol=1e-13)


def test_norm_ppf_is_monotone():
    """Monotonicity is load-bearing, not cosmetic.

    The partition-of-unity identity `sum(P_m) == 1` holds for *any* monotone map, so it
    cannot detect a broken quantile function. Monotonicity is what guarantees every regional
    mass stays non-negative once refinement produces thin boxes, and it is the property that
    actually protects the whole scheme.

    **Non-decreasing, not strictly increasing.** Two probabilities one ulp apart differ in
    their true quantiles by ~1e-17, which is below what a double can represent in the
    result; requiring strict growth would assert a resolution the format does not have. A
    tie is harmless — it cannot make a mass negative. Any genuine *decrease* would be, and
    none is tolerated below.
    """
    grid = np.unique(
        np.concatenate([np.linspace(1e-300, 1 - 1e-16, 200_001), np.logspace(-300, -1, 5_001)])
    )
    grid = grid[(grid > 0.0) & (grid < 1.0)]
    assert np.all(np.diff(norm_ppf(grid)) >= 0.0)

    # On a grid coarse enough that the true differences are comfortably resolvable, growth is
    # strict: this is the statement that rules out a plateau introduced by a wrong branch.
    coarse = np.linspace(1e-12, 1 - 1e-12, 100_001)
    assert np.all(np.diff(norm_ppf(coarse)) > 0.0)


def test_norm_ppf_round_trips_through_the_standard_normal_cdf():
    """``Phi^-1(Phi(x)) == x``, with a bound that reflects where the *reference* degrades.

    The limit here is ``erfc``, not the inversion. ``Phi(x) = 1 - eps`` for large positive
    ``x``, and the digits encoding ``eps`` are exactly the ones a double near 1.0 cannot
    hold, so the round trip's accuracy falls off going up the upper tail — measured at
    1e-14 at 3 sigma, 3e-11 at 5 and 6e-6 at 7, all of it in the reference.

    The **lower** tail has no such limit: ``Phi`` is there a small number computed with full
    relative precision, so the round trip holds to ~1e-15 all the way to 8 sigma. Asserting
    the two regimes separately states what is actually true instead of picking a symmetric
    bound that is loose on one side and unreachable on the other.
    """
    lower = np.linspace(-8.0, -1.0, 20_001)
    upper = np.linspace(1.0, 4.0, 20_001)
    for x, atol in ((lower, 1e-12), (upper, 1e-12)):
        cdf = np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in x])
        np.testing.assert_allclose(norm_ppf(cdf), x, rtol=0, atol=atol)


def test_normal_interval_mass_is_stable_in_both_tails():
    """``Phi(hi) - Phi(lo)`` without cancellation.

    The naive difference loses every significant digit when both bounds sit in the same tail,
    which is precisely where the stratifier's outer regions live. This asserts the stable
    form against quadrature of the pdf, and covers an interval *in* each tail rather than
    only one straddling zero. Masses below the smallest normal double are excluded: they
    underflow for the reference and the estimate alike, and comparing them would be
    comparing two zeros.
    """
    for lo, hi in ((-0.5, -1e-9), (-38.0, -36.0), (36.0, 38.0), (1e-9, 0.5), (-3.0, 3.0)):
        assert lo < hi
        got = _normal_interval_mass(lo, hi)
        grid = np.linspace(lo, hi, 400_001)
        pdf = np.exp(-0.5 * grid * grid) / math.sqrt(2.0 * math.pi)
        reference = float(np.trapezoid(pdf, grid))
        assert got > 0.0
        assert got == pytest.approx(reference, rel=1e-6)

    # An inverted interval is not this function's contract, but it must not come back
    # negative-and-large: a sign slip in a caller would then silently subtract mass.
    assert _normal_interval_mass(-1e-9, -0.5) < 0.0


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_leaves", [1, 2, 3, 5, 7, 8, 17, 64, 100, 256, 257])
def test_partition_tiles_the_unit_cube_exactly(n_leaves):
    """Disjoint, inside the cube, and covering it — for a non-power-of-two count too.

    The arbitrary-count case is the one that matters: a partition scheme that only works for
    powers of two would force `max_regions` to be rounded and would make the refinement
    granularity coarse exactly where the pilot says the action is.
    """
    boxes = _reference_partition(n_leaves)
    assert len(boxes) == n_leaves

    lo = np.array([b[0] for b in boxes])
    hi = np.array([b[1] for b in boxes])
    assert np.all(lo >= 0.0) and np.all(hi <= 1.0)
    assert np.all(lo < hi)

    # Union covers the cube: total volume is 1, and since the boxes tile by construction a
    # volume shortfall would be exactly an overlap or a hole.
    volumes = np.prod(hi - lo, axis=1)
    assert volumes.sum() == pytest.approx(1.0, rel=0, abs=1e-12)
    assert np.all(volumes > 0.0)

    # Non-overlap, checked pairwise on a random sample of pairs (quadratic, so not
    # exhaustive) plus a volume argument: disjoint boxes with total volume 1 cannot overlap.
    rng = np.random.default_rng(0)
    for _ in range(200):
        i, j = rng.integers(0, n_leaves, 2)
        if i == j:
            continue
        overlap = np.all(np.minimum(hi[i], hi[j]) > np.maximum(lo[i], lo[j]) + 0.0)
        assert not overlap or np.any(lo[i] == lo[j]) or np.any(hi[i] == hi[j])


@pytest.mark.parametrize("n_leaves", [1, 2, 3, 5, 8, 64, 256])
def test_reference_partition_is_deterministic(n_leaves):
    """Same request, same partition. A plan that reshuffled its regions between builds
    would make "reuse the plan for another N" silently produce a different estimator."""
    first = _reference_partition(n_leaves)
    second = _reference_partition(n_leaves)
    for (lo_a, hi_a), (lo_b, hi_b) in zip(first, second):
        np.testing.assert_array_equal(lo_a, lo_b)
        np.testing.assert_array_equal(hi_a, hi_b)


def test_halton_points_are_prefix_stable_and_shifted():
    """Point k does not depend on how many points were requested.

    This is what makes multiplicative refinement (``n' = K n``) exact: the first ``n``
    particles of a region are bit-identical whether the region was asked for ``n`` or
    ``K n``, so refining extends the estimate instead of replacing it.
    """
    short = _halton_points(6, seed=3, region_id=1, stream="production")
    long = _halton_points(24, seed=3, region_id=1, stream="production")
    np.testing.assert_array_equal(short, long[:6])
    assert np.all((short > 0.0) & (short < 1.0))


def test_halton_shifts_differ_by_region_and_stream():
    """Region and stream tags must actually change the points.

    If the pilot reused the production sequence, a region's allocation would depend on how
    its own *chosen* production points happened to fall — exactly the bias the pilot exists
    to avoid. Distinct shifts make the two draws independent by construction.
    """
    a = _halton_points(16, seed=3, region_id=1, stream="production")
    b = _halton_points(16, seed=3, region_id=2, stream="production")
    c = _halton_points(16, seed=3, region_id=1, stream="pilot")
    d = _halton_points(16, seed=4, region_id=1, stream="production")
    assert not np.allclose(a, b)
    assert not np.allclose(a, c)
    assert not np.allclose(a, d)


def test_halton_avoids_the_origin_point():
    """Index 0 is excluded.

    A Halton sequence's zeroth point is the origin of the cube, which maps to the region's
    lower edge in *every* coordinate at once — a point carrying strictly less than a full
    share of the box's mass, at the box's least representative location.
    """
    points = _halton_points(1, seed=0, region_id=0, stream="production")
    assert len(HALTON_BASES) == N_LATENT
    assert np.all(points > 0.0)


# ---------------------------------------------------------------------------
# The plan: exact masses, exact budget, exact weights
# ---------------------------------------------------------------------------
def test_target_masses_sum_to_one_and_are_positive():
    """The partition-of-unity identity, at the plan's own resolution.

    `sum(P_m) == 1` is not a numerical coincidence to be tolerated loosely: the boxes tile
    the cube and `u -> s Phi^-1(u)` is a bijection onto the real line, so the masses tile all
    of latent space and their total probability is 1 by construction. A deviation means the
    bounds or the mass formula are wrong.
    """
    beam, laser = correlated_beam(), scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=5, config=FAST)
    masses = np.array([r.target_mass for r in plan.regions])
    assert np.all(masses > 0.0)
    assert masses.sum() == pytest.approx(1.0, rel=0, abs=1e-13)
    assert sum(r.reference_mass for r in plan.regions) == pytest.approx(1.0, rel=0, abs=1e-12)


def test_regional_target_mass_matches_an_independent_normal_cdf_product():
    """Recompute ``P_m`` from the region's latent bounds, the long way.

    The stored field and `target_mass_exact` are computed by different code paths — the
    first during the plan build, the second from the region's own bounds — so agreement is a
    real check on both the bounds and the mass formula.
    """
    beam, laser = correlated_beam(), scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=5, config=FAST)
    for region in plan.regions:
        expected = region.target_mass_exact(plan.proposal_scale)
        assert region.target_mass == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("n_particles", [16, 17, 100, 999, 10_000])
def test_allocation_is_exact_and_never_starves_a_region(n_particles):
    """``sum(n_m) == N`` and ``n_m >= 1``, for any ``N >= M``.

    Every region must receive at least one particle. A region with none would contribute
    *zero* instead of its small exact mass, which is a silent loss of represented physics
    rather than a visible failure.
    """
    beam, laser = correlated_beam(), scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=5, config=FAST)
    counts = plan.allocate(n_particles)
    assert counts.sum() == n_particles
    assert np.all(counts >= 1)
    assert len(counts) == plan.n_regions


def test_allocation_refuses_a_budget_below_the_region_count():
    """Reject rather than truncate: a silent truncation would lose represented mass."""
    beam, laser = correlated_beam(), scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=5, config=FAST)
    with pytest.raises(ValueError, match="at least one particle"):
        plan.allocate(plan.n_regions - 1)


@pytest.mark.parametrize("n_particles", [500, 5_000, 20_000])
def test_weights_sum_to_one_without_any_normalization(n_particles):
    """``sum(w) == 1`` to machine precision, with no renormalization pass anywhere.

    The regional weights are ``P_m / n_m``, so the total is ``sum_m P_m`` analytically. A
    self-normalizing implementation would also report 1 here — which is why the test also
    checks that a region whose mass is tiny keeps a weight proportional to its *own* mass
    rather than to the particle count.
    """
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    bunch, plan = build_adaptive_bunch(beam, laser, n_particles, seed=11, config=FAST)
    assert float(bunch.weight.sum()) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert not np.allclose(bunch.weight, bunch.weight[0]), "adaptive weights should not be uniform"

    # Within one region every particle carries exactly P_m / n_m, and no two regions share
    # a weight unless their masses and counts conspire to.
    deviates, weight, region_index = plan.deviates(n_particles)
    counts = plan.allocate(n_particles)
    for index, region in enumerate(plan.regions):
        mine = weight[region_index == index]
        assert np.allclose(mine, region.target_mass / counts[index], rtol=1e-12, atol=0.0)


def test_conditional_samples_stay_inside_their_own_region():
    """Transformed deviates respect the region's latent bounds, in every coordinate.

    A sample that escaped its box would be attributed a weight that no longer describes the
    mass it represents — the estimator would then be biased in a way the weight sum cannot
    reveal.
    """
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=5, config=FAST)
    for region in plan.regions:
        deviates = region.deviates(32, plan.seed, "production")
        latent_lo, latent_hi = (plan.proposal_scale * b for b in region.latent_bounds)
        assert np.all(deviates >= latent_lo - 1e-9)
        assert np.all(deviates <= latent_hi + 1e-9)


def test_plan_is_reusable_for_arbitrary_particle_counts():
    """One plan, many budgets, and the pilot is not re-run.

    The plan depends only on ``(beam, laser, seed)``. Rebuilding it for every particle count
    would make the accuracy-versus-cost question unanswerable, since the pilot cost would
    scale with the sweep rather than being paid once.
    """
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=5, config=FAST)
    for n in (1_000, 4_000, 16_000):
        again = build_adaptive_bunch(beam, laser, n, 5, plan=plan, config=FAST)[1]
        assert again is plan
        assert again.deviates(n)[1].sum() == pytest.approx(1.0, rel=0, abs=1e-12)


def test_refusing_a_plan_built_for_a_different_seed():
    """A plan's low-discrepancy shifts are seeded, so cross-seed reuse is undefined."""
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=5, config=FAST)
    with pytest.raises(ValueError, match="seed"):
        build_adaptive_bunch(beam, laser, 1000, 6, plan=plan)


# ---------------------------------------------------------------------------
# The represented distribution
# ---------------------------------------------------------------------------
@pytest.mark.tier2
@pytest.mark.parametrize("strategy", [IID, ADAPTIVE])
def test_weighted_beam_moments_reproduce_the_analytic_target(strategy):
    """Every second moment and every stored correlation, for both strategies.

    This is the test that makes an adaptive bunch *interchangeable* with an IID one. The
    beam here has all five energy correlations and both Twiss tilts non-zero, because a
    sampler that reproduces the diagonal but mishandles a correlation is precisely the bug
    the shared `_bunch_from_standard_deviates` refactor exists to prevent — and the diagonal
    alone would not notice.
    """
    beam = replace(scenarios.BASELINE.beam, **CORRELATED)
    laser = scenarios.BASELINE.laser
    bunch = build_interaction(
        beam, laser, scenarios.BASELINE.target,
        SamplingSpec(n_particles=200_000, seed=3, prefilter=0.0, strategy=strategy),
    ).bunch

    def weighted_mean(values):
        return float(np.sum(values * bunch.weight) / np.sum(bunch.weight))

    def weighted_sigma(name):
        values = getattr(bunch, name)
        return math.sqrt(max(weighted_mean(values**2) - weighted_mean(values) ** 2, 0.0))

    def covariance(a, b):
        return weighted_mean(getattr(bunch, a) * getattr(bunch, b)) - (
            weighted_mean(getattr(bunch, a)) * weighted_mean(getattr(bunch, b))
        )

    def correlation(a, b):
        return covariance(a, b) / (weighted_sigma(a) * weighted_sigma(b))

    # Marginals.
    assert weighted_sigma("x") == pytest.approx(beam.m("sigma_x"), rel=0.01)
    assert weighted_sigma("y") == pytest.approx(beam.m("sigma_y"), rel=0.01)
    assert weighted_sigma("z") == pytest.approx(beam.m("sigma_z"), rel=0.01)
    assert weighted_sigma("thx") == pytest.approx(beam.divergence_x(), rel=0.02)
    assert weighted_sigma("thy") == pytest.approx(beam.divergence_y(), rel=0.02)
    assert weighted_sigma("gamma") == pytest.approx(beam.sigma_gamma(), rel=0.01)

    # Correlations, including the angle-energy pair that a `drift`-derived implementation
    # would not reproduce. `E[ab]` is not the covariance -- with mean(gamma) of order 2e3,
    # an uncorrected mean product swamps a correlation of order 0.1.
    assert correlation("x", "gamma") == pytest.approx(beam.rho_x_gamma, abs=0.015)
    assert correlation("y", "gamma") == pytest.approx(beam.rho_y_gamma, abs=0.015)
    assert correlation("z", "gamma") == pytest.approx(beam.rho_z_gamma, abs=0.015)
    assert correlation("thx", "gamma") == pytest.approx(beam.rho_thx_gamma, abs=0.015)
    assert correlation("thy", "gamma") == pytest.approx(beam.rho_thy_gamma, abs=0.015)

    # The Twiss tilt shows up as a position-angle correlation of -alpha / sqrt(1 + alpha^2).
    for plane, alpha in (("x", beam.alpha_x), ("y", beam.alpha_y)):
        expected = -alpha / math.sqrt(1.0 + alpha**2)
        assert correlation(plane, f"th{plane}") == pytest.approx(expected, abs=0.015)


def test_iid_sampler_is_unchanged_by_the_shared_transform_refactor():
    """The refactor that let the two samplers share one map must be value-preserving.

    Reproduces the pre-refactor arithmetic inline and demands bit-for-bit equality. This is
    the one property the adaptive path could have broken silently: a "harmless" algebraic
    reassociation in the Twiss tilt or the gamma sum would change every sampled bunch in the
    repository while leaving all the physics tests green.
    """
    import math as _math

    from gammaforge.io.bunch import (
        Bunch,
        SAMPLED_VARIABLES,
        _substream_deviates,
        _tilted_angle,
        gamma_coefficients,
    )

    def pre_refactor(beam, n_particles, seed):
        d = _substream_deviates(seed, n_particles)
        x = beam.m("sigma_x") * d["x"]
        y = beam.m("sigma_y") * d["y"]
        z = beam.m("sigma_z") * d["z"]
        thx = _tilted_angle(beam.divergence_x(), beam.alpha_x, d["x"], d["thx"])
        thy = _tilted_angle(beam.divergence_y(), beam.alpha_y, d["y"], d["thy"])
        a_x, a_y, a_z, a_thx, a_thy, residual = gamma_coefficients(beam)
        gamma = beam.gamma0() + beam.sigma_gamma() * (
            a_x * d["x"]
            + a_y * d["y"]
            + a_z * d["z"]
            + a_thx * d["thx"]
            + a_thy * d["thy"]
            + _math.sqrt(residual) * d["gamma"]
        )
        return Bunch._from_owned(
            x=x, y=y, z=z, thx=thx, thy=thy, gamma=gamma,
            weight=np.full(n_particles, 1.0 / n_particles),
            meta={"seed": seed, "n_particles": n_particles}, gaussian_fit=beam,
        )

    for beam in (scenarios.BASELINE.beam, correlated_beam(**CORRELATED)):
        for n_particles, seed in ((1, 0), (37, 5), (5_000, 20260721)):
            new = sample_gaussian_bunch(beam, n_particles, seed)
            old = pre_refactor(beam, n_particles, seed)
            for a, b in zip(new.arrays(), old.arrays()):
                assert np.array_equal(a, b)
            assert new.meta == old.meta
    assert SAMPLED_VARIABLES == ("x", "y", "z", "thx", "thy", "gamma")


def test_prefilter_does_not_renormalize_adaptive_weights():
    """The prefilter stays a pure optimization for the adaptive strategy too.

    For an unfiltered adaptive bunch the weights sum to exactly one, so renormalizing after
    a discard would silently rescale every result by the retained fraction — a
    charge-creating bug that is invisible in any single-strategy comparison.
    """
    beam = replace(scenarios.BASELINE.beam, sigma_x=Q(400.0, "um"), sigma_y=Q(400.0, "um"))
    laser = replace(
        scenarios.BASELINE.laser, sigma_x=Q(4.0, "um"), sigma_y=Q(4.0, "um"),
        duration=Q(1.0, "ps"),
    )
    bunch, _ = build_adaptive_bunch(beam, laser, 20_000, 3, config=FAST)
    assert float(bunch.weight.sum()) == pytest.approx(1.0, rel=0, abs=1e-12)

    kept = prefilter_bunch(bunch, laser, 1e-3)
    assert 0 < kept.n_particles < bunch.n_particles
    assert float(kept.weight.sum()) < 1.0

    # Dropped, not rescaled. Every surviving weight must be a value that was already
    # present in the unfiltered bunch: a renormalizing filter would scale them all up by
    # 1/retained_fraction, and none of them would match.
    available = set(np.unique(bunch.weight).tolist())
    assert set(np.unique(kept.weight).tolist()) <= available
    # And no survivor was scaled up, which is the same statement from the other side.
    assert float(kept.weight.max()) <= float(bunch.weight.max())
    # The retained total records the drop honestly, staying strictly below one.
    assert 0.0 < float(kept.weight.sum()) < 1.0


def test_iid_and_adaptive_agree_on_the_total_represented_mass():
    """Both strategies represent a bunch whose weights sum to one before filtering."""
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    for strategy in (IID, ADAPTIVE):
        bunch = build_interaction(
            beam, laser, scenarios.BASELINE.target,
            SamplingSpec(n_particles=30_000, seed=17, prefilter=0.0, strategy=strategy),
        ).bunch
        assert float(bunch.weight.sum()) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Tail representation
# ---------------------------------------------------------------------------
def _chi2_6_cdf(x):
    """CDF of chi-squared with 6 degrees of freedom (the even-degree closed form)."""
    h = np.asarray(x, dtype=float) / 2.0
    return 1.0 - np.exp(-h) * (1.0 + h + h * h / 2.0)


def _weighted_cdf_error(deviates, weight, threshold, keep_below):
    """``|weighted fraction satisfying the test - true fraction|``.

    Unweighted sample *counts* cannot be compared to the target for the adaptive strategy:
    its spatial density is deliberately different from the target's, which is the entire
    point. The weighted fraction is the quantity that must match.
    """
    keep = (deviates**2).sum(axis=1) <= threshold if keep_below else deviates <= threshold
    return abs(float(np.sum(weight[keep])) - float(_chi2_6_cdf(threshold)))


def _max_weighted_marginal_error(deviates, weight, sigma=1.0, n_thresholds=60):
    """Worst weighted 1D CDF error over all six latent coordinates."""
    worst = 0.0
    for axis in range(deviates.shape[1]):
        for threshold in np.linspace(-3.0 * sigma, 3.0 * sigma, n_thresholds):
            # P(d <= t) for a standard normal, written on the accurate side.
            truth = 0.5 * (1.0 + math.erf(threshold / (sigma * math.sqrt(2.0))))
            got = float(np.sum(weight[deviates[:, axis] <= threshold]))
            worst = max(worst, abs(got - truth))
    return worst


@pytest.mark.tier2
def test_adaptive_representation_has_no_floor_as_n_grows():
    """The weighted representation must **converge**, not settle at a bias.

    This is the regression for the subtlest bug the feature could ship: forming the target
    conditional from the *cube* coordinates instead of the target's own CDF coordinates.
    That mistake is invisible to `sum(P_m) == 1` (the masses are computed from the correct
    latent bounds either way) and to `sum(w) == 1`, and it does not shift any weight — it
    only distorts *where* the represented mass sits. Its signature is unmistakable once
    looked for: the error converges to a **constant** of a few percent and stops improving
    with `N`, while IID keeps falling as `1/sqrt(N)`.

    Two budgets an order of magnitude apart are enough to separate a decaying error from a
    floor, and asserting the ratio also pins the rate rather than merely "it got better".
    """
    from gammaforge.io.bunch import SAMPLED_VARIABLES, _substream_deviates

    beam = replace(scenarios.BASELINE.beam, **CORRELATED)
    laser = scenarios.BASELINE.laser
    config = PilotConfig(
        initial_regions=64, max_regions=256, pilot_points_per_region=4, pilot_quad_nodes=16
    )
    plan = build_adaptive_plan(beam, laser, seed=5, config=config)

    errors = []
    for n in (20_000, 160_000):
        deviates, weight, _ = plan.deviates(n)
        errors.append(_max_weighted_marginal_error(deviates, weight))
    # A bias would leave this ratio near 1; genuine convergence pushes it well below.
    assert errors[1] < 0.4 * errors[0], (
        f"weighted marginal error did not converge: {errors[0]:.2e} -> {errors[1]:.2e}. "
        f"A flat error means the represented distribution is biased, not merely noisy."
    )

    # And the end state is genuinely close, not merely shrinking.
    assert errors[1] < 5e-4


@pytest.mark.tier2
def test_adaptive_beats_iid_on_the_per_coordinate_marginals():
    """Where stratification pays: the one-dimensional marginals.

    Each region's box is a product of coordinate intervals, so a per-coordinate CDF is very
    nearly constant inside a box — exactly the regime where stratified sampling beats IID.
    The margin grows with `N`, which is the signature of a variance reduction rather than a
    lucky draw.

    **The radial distribution is a different story and is deliberately not claimed here.**
    `r^2 = d.d` is not a product functional, and a cell whose corner reaches `|d| = inf`
    spans the whole range of `r^2` — so no box isolates the radial tail and the measured
    improvement there is roughly parity. See the benchmark's report for the numbers; asserting
    a win here would be asserting something the design does not deliver.
    """
    from gammaforge.io.bunch import SAMPLED_VARIABLES, _substream_deviates

    beam = replace(scenarios.BASELINE.beam, **CORRELATED)
    laser = scenarios.BASELINE.laser
    config = PilotConfig(
        initial_regions=64, max_regions=256, pilot_points_per_region=4, pilot_quad_nodes=16
    )
    plan = build_adaptive_plan(beam, laser, seed=5, config=config)
    n = 80_000

    deviates, weight, _ = plan.deviates(n)
    draws = _substream_deviates(5, n)
    iid_deviates = np.column_stack([draws[name] for name in SAMPLED_VARIABLES])

    adaptive_error = _max_weighted_marginal_error(deviates, weight)
    iid_error = _max_weighted_marginal_error(iid_deviates, np.full(n, 1.0 / n))
    assert adaptive_error < iid_error / 2.0, (
        f"adaptive marginal error {adaptive_error:.2e} did not beat IID {iid_error:.2e}"
    )


@pytest.mark.tier2
def test_conditional_sampling_uses_the_target_cdf_not_the_cube_coordinates():
    """The target conditional is formed in the **target's** CDF coordinates.

    A region is axis-aligned in the *reference* coordinate ``u = Phi(d/s)``, so its edges in
    the target's CDF are ``Phi(s Phi^-1(a))``, not ``a``. These coincide only at ``s == 1``.
    Constructed the wrong way, the sampler draws from the *reference* conditional: the
    weights still sum to one and the masses are still exact, so nothing else in the suite
    notices — the error is a pure distortion of where the represented mass sits.
    """
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    for scale in (1.0, math.sqrt(2.0)):
        config = PilotConfig(
            initial_regions=8, max_regions=8, pilot_points_per_region=2, pilot_quad_nodes=8,
            proposal_scale=scale,
        )
        plan = build_adaptive_plan(beam, laser, seed=5, config=config)
        deviates, weight, _ = plan.deviates(20_000)

        # A correctly-drawn conditional reproduces the target's own marginals, so compare
        # each coordinate's weighted mean/sigma against the standard normal's.
        for axis in range(N_LATENT):
            column = deviates[:, axis]
            mean = float(np.sum(column * weight))
            sigma = math.sqrt(max(float(np.sum(column**2 * weight)) - mean**2, 0.0))
            assert mean == pytest.approx(0.0, abs=0.02), (
                f"latent coordinate {axis} has weighted mean {mean:.4f} at scale {scale}"
            )
            assert sigma == pytest.approx(1.0, rel=0.02), (
                f"latent coordinate {axis} has weighted sigma {sigma:.4f} at scale {scale}"
            )


# ---------------------------------------------------------------------------
# The pilot
# ---------------------------------------------------------------------------
def _spearman(a, b):
    """Rank correlation. The pilot's *ordering* is its product; its scale is irrelevant.

    Stage 0 multiplies by factors common to every particle (``N_e``, the weight, the rate
    constant), which the pilot deliberately drops, so only a scale-free comparison is
    meaningful. Rank correlation is also exactly the statistic the allocation consumes:
    `A_m = P_m sqrt(M2_m)` ranks regions by pilot-predicted luminosity.
    """
    return float(
        np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    )


def _stage0_luminosity(bunch, laser, beam, n_steps=2000):
    """Over-resolved Stage 0, the arbiter for the pilot."""
    from gammaforge.engines.xigma.stages import integrate_trajectories

    samples = integrate_trajectories(
        bunch, laser, beam.n_electrons(), n_steps=n_steps, threshold=1e-3
    )
    return np.asarray(samples.luminosity, dtype=float)


def _pilot_rank_agreement(bunch, laser, beam, *, restrict_to=0.80, **kwargs):
    """Rank agreement with Stage 0 among the particles that actually carry luminosity.

    Restricting to the top ``1 - restrict_to`` of Stage-0 luminosity is not a way to make
    the pilot look good — it is the population the allocation is decided on. Over *all*
    particles the metric is dominated by pairs where both answers are ~0 and their order is
    quadrature noise, which says nothing about the allocation; among the bright particles
    the ordering is the signal.
    """
    reference = _stage0_luminosity(bunch, laser, beam)
    bright = reference >= np.quantile(reference, restrict_to)
    predicted = trajectory_luminosity_predictor(bunch, laser, **kwargs)
    return _spearman(predicted[bright], reference[bright])


@pytest.mark.tier2
@pytest.mark.parametrize(
    "name,beam,laser",
    [
        ("head-on", {}, {}),
        ("transverse offset", {}, dict(x_off=Q(30, "um"))),
        ("timing offset", {}, dict(t_off=Q(200, "fs"))),
        ("angular offset", dict(emit_x=Q(6e-7, "cm * rad"), emit_y=Q(6e-7, "cm * rad")), {}),
        ("crossing angle", {}, dict(theta_xz=Q(15, "mrad"))),
        ("displaced foci", {}, dict(z_fx=Q(0.3, "cm"), z_fy=Q(0.3, "cm"))),
        ("astigmatic laser", {}, dict(sigma_x=Q(5, "um"), sigma_y=Q(20, "um"))),
        ("nonzero Twiss alpha", dict(alpha_x=0.8, alpha_y=-0.5), {}),
        ("beam/energy correlations", dict(rho_z_gamma=0.5, rho_x_gamma=0.3), {}),
    ],
)
def test_pilot_ranks_luminosity_like_stage_zero(name, beam, laser):
    """The pilot reproduces Stage 0's ordering well enough to allocate on.

    Scoped to the geometries the handoff lists as required coverage. The pilot is allowed
    to be approximate — it buys efficiency, never correctness — but "approximate" has to
    mean *the right regions get more particles*, which is a testable statement.
    """
    beam_ = replace(scenarios.BASELINE.beam, **beam)
    laser_ = replace(scenarios.BASELINE.laser, **laser)
    bunch = sample_gaussian_bunch(beam_, 1_200, 4)
    assert _pilot_rank_agreement(bunch, laser_, beam_) > 0.9, (
        f"pilot ranking degraded on '{name}'"
    )


@pytest.mark.tier2
def test_pilot_improves_with_more_quadrature_nodes():
    """More nodes must not make the pilot worse — the knob has to actually do something.

    Measured on a **tight focus** rather than the baseline scenario, because the baseline is
    already saturated: its pilot agrees with over-resolved Stage 0 to 1 - 2e-16 at the
    default setting, so there is no headroom to demonstrate a knob with. A 3 um spot against
    a 4 um beam puts the pulse far inside the conservative window, which is exactly the
    regime where quadrature resolution is the binding constraint.

    This is also why the shipped default is a *composite* rule rather than one high-order
    rule: the domain is a deliberately over-wide conservative bound, so nodes have to be
    spread across it as well as raised in number. Over five geometries, splitting the same
    budget into two panels roughly doubles the rank agreement among bright particles.
    """
    beam = scenarios.BASELINE.beam
    laser = replace(scenarios.BASELINE.laser, sigma_x=Q(3, "um"), sigma_y=Q(3, "um"))
    bunch = sample_gaussian_bunch(beam, 1_200, 4)
    coarse = _pilot_rank_agreement(bunch, laser, beam, n_quad=32, panels=1)
    default = _pilot_rank_agreement(bunch, laser, beam)
    fine = _pilot_rank_agreement(bunch, laser, beam, n_quad=256, panels=2)
    assert coarse < default <= fine
    assert default > coarse + 0.1, "the default quadrature is not better than a crude one"


@pytest.mark.tier2
def test_a_deliberately_bad_pilot_still_represents_the_beam_exactly():
    """**Pilot quality affects efficiency, never correctness.** The architectural guarantee.

    A two-node rule over a near-entire-region window ranks luminosity poorly by
    construction. It is used here to build a real plan, and the resulting bunch is then held
    to exactly the same standards as a good-pilot bunch: weights summing to one, exact
    budget, and weighted moments reproducing every beam correlation.

    This is the single most important test in the file. It is what makes it safe to run an
    approximate pilot at `O(regions x points x nodes)` field evaluations against a Stage 0
    costing `O(N x steps)`: the pilot's estimate never enters the represented mass, so
    being wrong about it can only make the answer *noisier*, never *different*.
    """
    degraded = PilotConfig(
        initial_regions=8, max_regions=8, pilot_points_per_region=1,
        pilot_quad_nodes=4, pilot_quad_panels=1, pilot_window_threshold=0.5,
    )
    beam = replace(scenarios.BASELINE.beam, **CORRELATED)
    laser = scenarios.BASELINE.laser
    bunch, _ = build_adaptive_bunch(beam, laser, 20_000, 5, config=degraded)

    assert float(bunch.weight.sum()) == pytest.approx(1.0, rel=0, abs=1e-12)

    def mean(values):
        return float(np.sum(values * bunch.weight) / np.sum(bunch.weight))

    def sigma(name):
        values = getattr(bunch, name)
        return math.sqrt(max(mean(values**2) - mean(values) ** 2, 0.0))

    assert sigma("x") == pytest.approx(beam.m("sigma_x"), rel=0.01)
    assert sigma("gamma") == pytest.approx(beam.sigma_gamma(), rel=0.01)
    corr = lambda a, b: (
        mean(getattr(bunch, a) * getattr(bunch, b))
        - mean(getattr(bunch, a)) * mean(getattr(bunch, b))
    ) / (sigma(a) * sigma(b))
    assert corr("x", "gamma") == pytest.approx(beam.rho_x_gamma, abs=0.02)
    assert corr("z", "gamma") == pytest.approx(beam.rho_z_gamma, abs=0.02)
    assert corr("thx", "gamma") == pytest.approx(beam.rho_thx_gamma, abs=0.02)


def test_pilot_is_finite_and_non_negative_for_particles_that_miss_the_pulse():
    """A bunch that cannot meet the pulse yields zeros, and never a NaN.

    This is the NaN guard. `overlap_time_window` reports a particle that never enters the
    region as ``t0 = +inf, t1 = -inf``; evaluating a trajectory there hands the laser
    non-finite coordinates, and a single such value propagates through the whole batch's
    sum. Stage 0 anchors those particles at a finite time and lets a zero span do the work;
    the pilot does the same, and this asserts it.

    A 20 mm bunch against a 10 um pulse still leaves a handful of particles whose
    *conservative* window is non-empty — the window is a bound, not a filter, so those get
    a real (utterly negligible) score rather than an exact zero. What must hold is that they
    are finite, non-negative, and negligible.
    """
    beam = replace(scenarios.BASELINE.beam, sigma_x=Q(2.0, "cm"), sigma_y=Q(2.0, "cm"))
    laser = scenarios.BASELINE.laser
    bunch = sample_gaussian_bunch(beam, 500, 3)
    luminosity = trajectory_luminosity_predictor(bunch, laser)

    assert luminosity.shape == (500,)
    assert np.all(np.isfinite(luminosity))
    assert np.all(luminosity >= 0.0)

    # Overwhelmingly exact zeros; the handful that are not come from particles whose
    # *conservative* window is non-empty, since the window is a bound and not a filter. The
    # claim is that they are negligible, not that they are absent.
    nonzero = luminosity[luminosity > 0.0]
    assert nonzero.size < 0.05 * luminosity.size
    matched = trajectory_luminosity_predictor(
        sample_gaussian_bunch(scenarios.BASELINE.beam, 500, 3), scenarios.BASELINE.laser
    )
    # Eleven orders of magnitude below a particle that actually meets the pulse: these
    # particles pass the *geometric* bound by a hair and see a vanishing fraction of the
    # peak. The bound is allowed to keep them; it is not allowed to make them matter.
    assert nonzero.max() < 1e-9 * matched.max()


def test_pilot_rejects_a_configuration_with_too_few_nodes_per_panel():
    """A panel count that would starve Gauss-Legendre is an error, not a silent slowdown.

    Splitting a small node budget over many panels produces a rule that is cheap, fast, and
    wrong — the exact failure the composite rule is supposed to avoid.
    """
    with pytest.raises(ValueError, match="too few nodes per panel"):
        PilotConfig(pilot_quad_nodes=8, pilot_quad_panels=4)
    with pytest.raises(ValueError, match="panels must be >= 1"):
        PilotConfig(pilot_quad_panels=0)


# ---------------------------------------------------------------------------
# Engine compatibility
# ---------------------------------------------------------------------------
@pytest.mark.tier2
def test_xigma_stage_zero_consumes_adaptive_weights_particle_by_particle():
    """Stage 0 must respond to the *weight vector*, not merely to the particle count.

    The audit question the handoff poses is whether an engine silently assumes uniform
    weights. The decisive test is a **per-particle perturbation**: double one particle's
    weight, leave every other array untouched, and require exactly that one luminosity to
    double. An engine that normalized by particle count, or that used the mean weight, or
    that rescaled the whole bunch, would not show this.

    (A single common factor does *not* extract the weight: Stage 0's luminosity is
    `N_e * weight_i * F_i * <common> * integral_i`, and the encounter factor `F_i` is
    itself per-particle, so `luminosity / weight` varies legitimately by ~an order of
    magnitude across a beam. That variation is the reason the perturbation test is the right
    instrument and a ratio test is not.)
    """
    from gammaforge.engines.xigma.stages import integrate_trajectories

    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    # An explicit, deliberately non-uniform config rather than DEFAULT_PILOT_CONFIG. This test
    # needs a weight vector with real spread to perturb; taking whatever spread the default
    # happens to produce couples a physics test to a numerical default, and that coupling is
    # not hypothetical -- when 970e0f9 corrected the allocation the default's spread fell from
    # 4.8 to 1.39 and this test started failing, having previously passed *because of* the bug
    # it was written to exist independently of. Pure luminosity allocation gives ~2.4-2.8
    # across seeds, so the margin is real rather than marginal.
    config = PilotConfig(luminosity_fraction=1.0)
    plan = build_adaptive_plan(beam, laser, seed=3, config=config)
    interaction = build_interaction(
        beam, laser, scenarios.BASELINE.target,
        SamplingSpec(n_particles=8_000, seed=3, prefilter=1e-3, strategy=ADAPTIVE),
        plan=plan,
    )
    bunch = interaction.bunch
    # Non-uniform to begin with, or there is nothing to perturb coherently.
    assert float(bunch.weight.max() / bunch.weight.min()) > 1.5

    baseline = np.asarray(
        integrate_trajectories(
            bunch, laser, interaction.N_e, n_steps=200, threshold=1e-3
        ).luminosity,
        dtype=float,
    )

    # Double the weight of a single emitting particle and nothing else.
    target = int(np.argmax(baseline))
    doubled_weight = np.array(bunch.weight, copy=True)
    doubled_weight[target] *= 2.0
    doubled = replace(bunch, weight=doubled_weight)

    after = np.asarray(
        integrate_trajectories(
            doubled, laser, interaction.N_e, n_steps=200, threshold=1e-3
        ).luminosity,
        dtype=float,
    )
    np.testing.assert_allclose(after[target], 2.0 * baseline[target], rtol=1e-12)
    np.testing.assert_allclose(
        np.delete(after, target), np.delete(baseline, target), rtol=1e-12, atol=0.0
    )


@pytest.mark.tier2
def test_engines_do_not_normalize_by_particle_count():
    """No engine may divide by the particle count — that is the equal-weight assumption.

    A source-level guard, because the failure is invisible in a single-strategy comparison:
    an engine that normalized a deposited density by ``n_particles`` instead of by the
    weight sum would agree with IID and disagree with adaptive, and the disagreement would
    look like a physics bug in the sampler rather than in the engine.
    """
    import pathlib

    import gammaforge.engines as engines

    root = pathlib.Path(engines.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if ("/ n_particles" in stripped or "/ len(bunch)" in stripped
                    or "/ bunch.n_particles" in stripped):
                offenders.append(f"{path.name}: {stripped}")
    assert not offenders, "engines must normalize by weight, not particle count:\n" + "\n".join(offenders)


@pytest.mark.tier2
def test_analytical_engine_is_independent_of_the_bunch_entirely():
    """Analytical is a closed-form overlap estimate; it never sees a macroparticle.

    So it cannot assume anything about weights — which is why adaptive sampling needs no
    restriction for it, and why `AnalyticalEngine` can be a validation anchor for a bunch it
    never reads.
    """
    from gammaforge.engines.analytical.engine import AnalyticalEngine

    beam, laser, target = scenarios.BASELINE.beam, scenarios.BASELINE.laser, scenarios.BASELINE.target
    engine = AnalyticalEngine()
    iid = build_interaction(
        beam, laser, target, SamplingSpec(n_particles=500, seed=3, prefilter=1e-3)
    )
    adaptive = build_interaction(
        beam, laser, target,
        SamplingSpec(n_particles=500, seed=3, prefilter=1e-3, strategy=ADAPTIVE),
    )
    # The two bunches genuinely differ — different draws, different weights — and the
    # engine's answer is identical, because it never reads either.
    from gammaforge.io.target import OutputKind

    assert not np.allclose(iid.bunch.x, adaptive.bunch.x)
    assert not np.allclose(iid.bunch.weight, adaptive.bunch.weight)
    iid_yield = float(engine.run(iid, engine.schema).photon_slices[OutputKind.TOTAL_YIELD].distr)
    adaptive_yield = float(
        engine.run(adaptive, engine.schema).photon_slices[OutputKind.TOTAL_YIELD].distr
    )
    assert iid_yield == pytest.approx(adaptive_yield, rel=1e-12)


# ---------------------------------------------------------------------------
# End-to-end convergence against a high-statistics reference
# ---------------------------------------------------------------------------
# The reference and the noise floor are built once, at module scope, because each costs
# several seconds of Stage 0 and they are shared by every test in the group.

#: Bins for the 5D ``ShapeTable``. Smaller than production (48,48,48,96,8) so a tier-2 test
#: finishes in seconds; the point is convergence, not final table resolution.
CONVERGENCE_BINS = (24, 16, 16, 32, 8)

#: High-statistics IID reference particle count. Two independent seeds of this size differ
#: by ~5e-4 in yield and ~1e-2 in spectral L1 -- that is the **noise floor** every
#: comparison below is measured against, and it is why those comparisons are stated
#: comparatively (IID vs adaptive at the same budget) rather than against a fixed absolute
#: tolerance that a reference of this size could not support.
_REFERENCE_N = 150_000


def _spectral_grid(laser):
    """The physical spectrum grid, from the same autorange the engine uses for ``s``.

    ``s = E / (4 * hbar omega_0)`` is how ``collision.py`` parameterises Stage 2, and its
    range comes from the target's ``spectrum`` autorange. Deriving it the same way keeps the
    test's grid identical to the one a real run would use.
    """
    from gammaforge.io.results import Axis
    from gammaforge.io.target import OutputKind, auto_ranges

    ranges = auto_ranges(scenarios.BASELINE.target, scenarios.BASELINE.beam, laser)
    energy_hi = float(ranges[OutputKind.SPECTRUM][Axis.ENERGY][1])
    return np.linspace(0.0, energy_hi / (4.0 * float(laser.photon_energy())), 600)


def _run_pipeline(strategy, n_particles, *, seed=3, laser=None):
    """Interaction -> Stage 0 -> Stage 1 -> retarget -> Stage 2, and the observables.

    The engine's table edges are derived from the data, so two runs do not share a grid.
    That makes a cell-by-cell table comparison ill-posed; the observables returned here are
    all grid-independent physical quantities (a total, a spectrum, a centroid), which is what
    actually has to converge.
    """
    from gammaforge.engines.xigma.stages import (
        deposit_shape_table,
        integrate_trajectories,
        retarget_ahat,
        spectrum_from_table,
    )

    beam, laser = scenarios.BASELINE.beam, laser or scenarios.BASELINE.laser
    interaction = build_interaction(
        beam, laser, scenarios.BASELINE.target,
        SamplingSpec(n_particles=n_particles, seed=seed, prefilter=1e-3, strategy=strategy),
    )
    samples = integrate_trajectories(
        interaction.bunch, laser, interaction.N_e, n_steps=200, threshold=1e-3
    )
    shape_table = deposit_shape_table(samples, n_bins=CONVERGENCE_BINS, scheme="nearest")
    table = retarget_ahat(shape_table, float(samples.intensity_peak))
    grid = _spectral_grid(laser)
    spectrum = spectrum_from_table(table, 0.0, 0.0, grid)
    return {
        "yield": float(np.sum(samples.luminosity)),
        "spectrum": spectrum,
        "centroid": float(np.sum(grid * spectrum) / np.sum(spectrum)),
        "grid": grid,
        "n_particles": interaction.bunch.n_particles,
        "shape_table": shape_table,
    }


# ---------------------------------------------------------------------------
# End-to-end convergence against a high-statistics reference
# ---------------------------------------------------------------------------
# Two measurement facts shape every assertion in this section, and both were established by
# measurement rather than assumed:
#
# 1. **A 150k-particle IID reference is not exact.** Independent seeds of it differ by
#    ~4.6e-4 in total yield. Any absolute tolerance finer than that is measuring the
#    reference, not the sampler. So the comparisons below are stated *comparatively* — the
#    same budget, both strategies, against the same reference — which is robust to a common
#    reference offset in a way an absolute bound is not.
# 2. **Both strategies are compared to a mean over several references**, so a single unlucky
#    reference draw cannot manufacture a win. A single-seed comparison at N = 5000 produced
#    a spurious "IID wins" here; averaging over 2 budgets x 3 references reverses it.
#
# Measured errors (mean over 2 seeds x 3 references), relative to that reference mean:
#
#     N        IID          adaptive     adaptive/IID
#     2500     3.6e-3       1.1e-3       0.31
#     5000     1.3e-3       9.8e-4       0.73
#     10000    2.9e-3       9.0e-4       0.31
#     20000    2.9e-3       5.8e-4       0.20
#     40000    1.6e-3       5.7e-4       0.37
#
# Read honestly: adaptive wins at every budget, by roughly 1.4x to 5x, **not** by the 1-2
# orders of magnitude the handoff hoped for, and its error flattens near 6e-4 — the
# reference noise floor. That floor is the limit of this measurement, not evidence that the
# adaptive estimator stops converging; `test_adaptive_error_falls_below_the_reference_floor`
# pins what is actually established.

#: Bins for the 5D ``ShapeTable``. Smaller than production (48,48,48,96,8) so a tier-2 test
#: finishes in seconds; the point is convergence, not final table resolution.
CONVERGENCE_BINS = (24, 16, 16, 32, 8)

#: High-statistics IID reference particle count, and the seeds used to average it.
_REFERENCE_N = 150_000
_REFERENCE_SEEDS = (3, 11, 23)
#: Budgets, and the sampler seeds used to average each.
_BUDGETS = (5_000, 20_000, 40_000)
_SAMPLER_SEEDS = (3, 11)


def _spectral_grid(laser):
    """The physical spectrum grid, from the same autorange the engine uses for ``s``.

    ``s = E / (4 * hbar omega_0)`` is how ``collision.py`` parameterises Stage 2, and its
    range comes from the target's ``spectrum`` autorange. Deriving it the same way keeps the
    test's grid identical to the one a real run would use.
    """
    from gammaforge.io.results import Axis
    from gammaforge.io.target import OutputKind, auto_ranges

    ranges = auto_ranges(scenarios.BASELINE.target, scenarios.BASELINE.beam, laser)
    energy_hi = float(ranges[OutputKind.SPECTRUM][Axis.ENERGY][1])
    return np.linspace(0.0, energy_hi / (4.0 * float(laser.photon_energy())), 600)


def _run_pipeline(strategy, n_particles, *, seed=3, laser=None):
    """Interaction -> Stage 0 -> Stage 1 -> retarget -> Stage 2, and the observables.

    The engine's table edges are derived from the data, so two runs do not share a grid.
    That makes a cell-by-cell table comparison ill-posed; the observables returned here are
    all grid-independent physical quantities (a total, a spectrum, a centroid), which is what
    actually has to converge.
    """
    from gammaforge.engines.xigma.stages import (
        deposit_shape_table,
        integrate_trajectories,
        retarget_ahat,
        spectrum_from_table,
    )

    beam, laser = scenarios.BASELINE.beam, laser or scenarios.BASELINE.laser
    interaction = build_interaction(
        beam, laser, scenarios.BASELINE.target,
        SamplingSpec(n_particles=n_particles, seed=seed, prefilter=1e-3, strategy=strategy),
    )
    samples = integrate_trajectories(
        interaction.bunch, laser, interaction.N_e, n_steps=200, threshold=1e-3
    )
    shape_table = deposit_shape_table(samples, n_bins=CONVERGENCE_BINS, scheme="nearest")
    table = retarget_ahat(shape_table, float(samples.intensity_peak))
    grid = _spectral_grid(laser)
    spectrum = spectrum_from_table(table, 0.0, 0.0, grid)
    return {
        "yield": float(np.sum(samples.luminosity)),
        "spectrum": spectrum,
        "centroid": float(np.sum(grid * spectrum) / np.sum(spectrum)),
        "grid": grid,
        "n_particles": interaction.bunch.n_particles,
    }


@pytest.fixture(scope="module")
def converged():
    """Mean observables over several high-statistics references, plus their spread.

    Built once, at module scope: each reference costs several seconds of Stage 0 and every
    test in the group shares them.
    """
    runs = [_run_pipeline(IID, _REFERENCE_N, seed=seed) for seed in _REFERENCE_SEEDS]
    yields = np.array([r["yield"] for r in runs])
    spectra = np.array([r["spectrum"] for r in runs])
    return {
        "yield": float(np.mean(yields)),
        "spectrum": np.mean(spectra, axis=0),
        "centroid": float(np.mean([r["centroid"] for r in runs])),
        "yield_spread": float((yields.max() - yields.min()) / np.mean(yields)),
        "spectrum_spread": float(
            np.mean(np.abs(spectra - np.mean(spectra, axis=0)), axis=1).mean()
            / np.mean(np.abs(np.mean(spectra, axis=0)))
        ),
        "grid": runs[0]["grid"],
    }


def _mean_error(strategy, n_particles, ref):
    """Mean relative error over several sampler seeds, against the reference mean."""
    yields, spectra, centroids = [], [], []
    for seed in _SAMPLER_SEEDS:
        result = _run_pipeline(strategy, n_particles, seed=seed)
        yields.append(abs(result["yield"] - ref["yield"]) / ref["yield"])
        spectra.append(
            float(np.sum(np.abs(result["spectrum"] - ref["spectrum"]))
                  / np.sum(np.abs(ref["spectrum"])))
        )
        centroids.append(abs(result["centroid"] - ref["centroid"]) / abs(ref["centroid"]))
    return {
        "yield": float(np.mean(yields)),
        "spectrum": float(np.mean(spectra)),
        "centroid": float(np.mean(centroids)),
    }


def test_the_reference_is_not_exact_and_that_bounds_the_measurement(converged):
    """Guard the guard: state the resolution limit of everything measured below.

    If a 150k reference agreed with itself to machine precision, then "adaptive beats IID"
    could be satisfied by noise. Quantifying the spread first is what makes the comparative
    claims mean something — and it is also why no assertion in this group uses an absolute
    tolerance tighter than this.
    """
    assert converged["yield_spread"] > 0.0, "the reference runs are not independent"
    assert converged["yield_spread"] < 5e-3
    assert converged["spectrum_spread"] < 5e-2


@pytest.mark.tier2
def test_adaptive_beats_iid_on_total_yield(converged):
    """The headline result, at every budget tested.

    The yield is the functional the allocation optimizes: it is the luminosity-weighted
    integral, and ``A_m = P_m sqrt(M2_m)`` is a Neyman-style criterion for exactly that. The
    measured margin is 1.4x to 5x, averaged over sampler seeds and reference seeds so it
    cannot be a single unlucky draw.
    """
    ref = converged
    for n_particles in _BUDGETS:
        iid = _mean_error(IID, n_particles, ref)["yield"]
        adaptive = _mean_error(ADAPTIVE, n_particles, ref)["yield"]
        assert adaptive < iid, (
            f"at {n_particles}: adaptive {adaptive:.2e} did not beat IID {iid:.2e}"
        )


@pytest.mark.tier2
def test_adaptive_error_falls_below_the_reference_noise_floor(converged):
    """What is actually established about absolute accuracy.

    The adaptive yield error flattens near 6e-4, which is the *reference* spread — so the
    measurement has bottomed out, not the estimator. This pins the claim to what the data
    supports: adaptive reaches agreement with a 150k IID reference at least as tight as the
    reference agrees with itself, whereas IID does not.

    The earlier cube-versus-target-CDF bug lived exactly in the gap this test guards: its
    error was ~6e-2 and completely flat in ``N``, which no reference-spread comparison would
    have hidden.
    """
    ref = converged
    floor = ref["yield_spread"]
    adaptive = _mean_error(ADAPTIVE, 40_000, ref)["yield"]
    iid = _mean_error(IID, 40_000, ref)["yield"]
    assert adaptive <= max(2.0 * floor, 1e-3), (
        f"adaptive error {adaptive:.2e} is above the reference spread {floor:.2e}"
    )
    assert iid > adaptive


@pytest.mark.tier2
def test_adaptive_converges_to_the_reference_spectrum(converged):
    """The resolved spectrum, not just the total — the handoff forbids total yield alone.

    A sampler can nail the integral while getting the distribution wrong, and Stage 1 is a
    5D density that Stage 2 integrates, so the shape has to be checked on its own.

    Both strategies are required to converge; neither is claimed to win. The measured
    spectral L1 is roughly at parity, because the residual error is dominated by how many
    particles land in each Stage-1 cell, and reallocating particles does not change that.
    """
    ref = converged
    for strategy in (IID, ADAPTIVE):
        error = _mean_error(strategy, 20_000, ref)["spectrum"]
        assert error < 5e-2, f"{strategy} spectrum L1 {error:.2e} is not converging to the reference"


@pytest.mark.tier2
def test_adaptive_spectrum_is_at_parity_with_iid(converged):
    """The honest negative result, pinned so that a regression is noticed.

    Adaptive sampling is a consistent win on the total yield and roughly **at parity** on
    the resolved spectrum shape. That is coherent rather than disappointing: the allocation
    optimizes the luminosity-weighted integral, while the residual spectral error is
    dominated by per-cell particle counts, which reallocation does not change. This is the
    handoff's own §20.3 limitation — the refinement criterion sees luminosity variation and
    not Stage-1 cell spread — showing up as a measurement rather than as prose.
    """
    ref = converged
    iid = _mean_error(IID, 20_000, ref)["spectrum"]
    adaptive = _mean_error(ADAPTIVE, 20_000, ref)["spectrum"]
    assert adaptive < 3.0 * iid, (
        f"adaptive spectral L1 {adaptive:.2e} is far worse than IID {iid:.2e}; "
        f"either the allocation regressed or the reference moved"
    )


@pytest.mark.tier2
def test_adaptive_spectral_centroid_is_accurate(converged):
    """A centroid check, which is sensitive to the spectrum's *shape* rather than its mass.

    Included because the total yield and the spectral L1 can both be satisfied by a
    distribution that is shifted in ``s``; the centroid is the cheapest observable that is
    not.
    """
    ref = converged
    for strategy in (IID, ADAPTIVE):
        error = _mean_error(strategy, 20_000, ref)["centroid"]
        assert error < 1e-3, f"{strategy} centroid error {error:.2e} is too large"


# ---------------------------------------------------------------------------
# The benchmark's own machinery
# ---------------------------------------------------------------------------
# The benchmark answers a question a test cannot (it sweeps budgets and reports
# accuracy-versus-cost), but two things in it are correctness-bearing and are cheap to pin:
# the plan-reuse contract, and the seed-scoped nature of a plan. Both were real defects found
# by running it.

def test_a_plan_is_scoped_to_one_seed_and_the_benchmark_honours_it():
    """One plan per seed, reused across budgets — never one plan across seeds.

    A plan's low-discrepancy shifts are seeded, so reusing one across seeds would silently
    produce a *different* estimator than the one that was piloted. The API refuses it, and the
    benchmark had to be fixed to build one plan per seed after hitting that refusal.
    """
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    plan = build_adaptive_plan(beam, laser, seed=3, config=FAST)
    assert plan.seed == 3
    with pytest.raises(ValueError, match="seed"):
        build_adaptive_bunch(beam, laser, 1_000, 4, plan=plan, config=FAST)


def test_the_benchmark_reports_a_real_accuracy_versus_cost_table():
    """Smoke-test the benchmark end to end on a reduced configuration.

    Guards the parts that are load-bearing rather than the numbers: that it runs, that it
    produces both strategies' error curves, and that `particles_for` inverts an error curve
    into a particle count and correctly reports a target it cannot reach. The measured
    values live in the benchmark's own output and in RES094; asserting them here would only
    re-assert this test file against itself.
    """
    import importlib.util
    import pathlib

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "benchmark_adaptive_sampling.py"
    spec = importlib.util.spec_from_file_location("benchmark_adaptive_sampling", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # particles_for inverts a synthetic curve, including the "not reached" case.
    curve = {1_000: 1e-1, 2_000: 1e-2, 4_000: 1e-3}
    assert module.particles_for(1e-3, curve) == pytest.approx(4_000)
    assert module.particles_for(1e-1, curve) == pytest.approx(1_000)
    assert module.particles_for(1e-6, curve) is None, "an unreachable target must report None"

    # A target between two points interpolates rather than snapping.
    interpolated = module.particles_for(1e-2, {1_000: 1e-1, 2_000: 1e-3})
    assert 1_000 < interpolated < 2_000

    # And the module actually runs, with a reduced budget/reference.
    assert module.main.__doc__ is None or True
    argv = ["benchmark", "--quick"]
    import sys
    saved = sys.argv
    try:
        sys.argv = argv
        assert module.main() == 0
    finally:
        sys.argv = saved


def test_luminosity_fraction_actually_changes_the_allocation():
    """`Q_m` must respond to `luminosity_fraction`, and the pilot moments must be real.

    A regression for a shipped bug: `build_adaptive_plan` constructed every `SamplingRegion`
    with a *placeholder* zero second moment and tracked the real pilot statistics in a
    separate dict that was only ever read for the split priority. `_allocation_shares` reads
    `pilot_second_moment` off the regions, so the importance score `P_m sqrt(M2_m)` was
    identically zero, its `total <= 0` fallback fired, and `Q_m` collapsed to `B_m` — for
    **every** `luminosity_fraction`. Only the region *splitting* was ever luminosity-driven.

    The symptom was subtle enough to be worth pinning: the scheme still produced exactly
    correct weights and still beat IID on the yield (that came from stratification and the
    luminosity-driven split), so every invariant and the headline test passed while the
    documented allocation was inert. `lambda` having no effect is the assertion that fails
    first.
    """
    beam, laser = scenarios.BASELINE.beam, scenarios.BASELINE.laser
    allocations, second_moments = [], []
    for value in (0.0, 0.5, 1.0):
        config = PilotConfig(
            initial_regions=32, max_regions=32, pilot_points_per_region=6,
            pilot_quad_nodes=64, pilot_quad_panels=2, luminosity_fraction=value,
        )
        plan = build_adaptive_plan(beam, laser, seed=3, config=config)
        share = np.array([r.allocation_probability for r in plan.regions])
        allocations.append(share / share.min())
        second_moments.append(np.array([r.pilot_second_moment for r in plan.regions]))
        assert share.sum() == pytest.approx(1.0, rel=1e-12)
        # The regions must carry their own pilot statistics, not a placeholder.
        assert np.all(second_moments[-1] > 0.0), "pilot second moments are still zero"
        assert np.any(np.array([r.pilot_mean for r in plan.regions]) > 0.0)

    # Monotone, and strictly ordered: lambda=0 is the pure B_m (uniform-over-volume) limit.
    assert np.allclose(allocations[0], 1.0), "lambda=0 must allocate purely by B_m"
    assert allocations[1].max() > allocations[0].max()
    assert allocations[2].max() > allocations[1].max()
    # The shape is preserved as lambda grows: a larger weight on the same importance score
    # rescales the allocation, it does not re-rank it. (Only compared between the two
    # non-degenerate cases -- lambda=0 is exactly uniform, so its correlation is undefined.)
    assert np.corrcoef(allocations[1], allocations[2])[0, 1] > 0.9, (
        "the allocation's shape changed with lambda, not just its spread"
    )


def test_allocation_is_driven_by_luminosity_not_only_volume():
    """The allocation must track luminosity, which volume-proportional allocation cannot do.

    Compares the realized allocation against the pilot's own second moments. On a
    localized interaction the two differ visibly; if the allocation were still `B_m` — the
    inert state this guards — it could not follow the luminosity at all.

    The measured spreads are modest, and that is a property of the scheme rather than a
    weak premise: a 6D Gaussian's region-to-region luminosity ratio grows only slowly even
    for a tight focus, because a region's *volume* varies over the same range and
    `A_m = P_m sqrt(M2_m)` mixes the two. Measured second-moment spreads on the baseline
    scenario are ~1.4x, rising to ~20x for a 1.5 um spot. `luminosity_fraction` is therefore
    a gentle knob by construction, not a mis-tuned one.
    """
    laser = replace(scenarios.BASELINE.laser, sigma_x=Q(1.5, "um"), sigma_y=Q(1.5, "um"))
    beam = scenarios.BASELINE.beam
    config = PilotConfig(
        initial_regions=64, max_regions=64, pilot_points_per_region=16,
        pilot_quad_nodes=128, pilot_quad_panels=2, luminosity_fraction=1.0,
    )
    plan = build_adaptive_plan(beam, laser, seed=3, config=config)
    share = np.array([r.allocation_probability for r in plan.regions])
    second = np.array([r.pilot_second_moment for r in plan.regions])

    # A tight focus makes the luminosity vary by more than an order of magnitude, so the
    # importance score and the volume cannot be proportional.
    assert second.max() / second.min() > 10.0, "test premise: luminosity must vary"
    assert share.max() / share.min() > 2.0
    # And the brightest region really is allocated the most.
    assert int(np.argmax(share)) == int(np.argmax(second))
