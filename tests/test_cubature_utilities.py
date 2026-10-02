"""Numerical invariants for the Phase-II cubature utilities (experiment-level).

These modules live under ``experiments/adaptive/`` rather than in ``gammaforge.io``, so this
test reaches them by path. That is deliberate and temporary: the handoff asks for focused
tests on "reusable numerical utilities that survive the experiment", and until it is known
which of these graduates to production, keeping the invariants enforced here is cheaper than
promoting a source rule on the strength of an experiment.

What is asserted is only what must be true for the *conclusions* to mean anything:

- the tensor Gauss-Hermite rule is the **exact** Gaussian rule, not an approximation that
  happens to look close -- checked on moments it can be checked exactly, since Gauss-Hermite
  integrates polynomials to its order and a normal's moments are known in closed form;
- its weights are positive and sum to one, which is what makes it a candidate ``Bunch.weight``
  at all (signed sparse-grid weights would not be);
- the global QMC sequence is **prefix-stable**, without which an error-versus-``N`` curve is
  meaningless because the points move when ``N`` changes;
- every source rule reaches the shared latent-to-physical map, so a difference between arms is
  the sampling rule and not three copies of the Twiss algebra;
- the smooth observables are **not** weighted twice -- ``TrajectorySamples.luminosity`` already
  carries the per-particle weight and ``N_e``. This one earned its place: getting it wrong made
  ``M0`` differ by a factor of ~50 between methods and scaled as ``1/N``.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "adaptive"))

import cubature as C  # noqa: E402
import smooth as S  # noqa: E402


class _Samples:
    """Minimal stand-in with the fields :func:`smooth.evaluate` reads."""

    def __init__(self, n, seed=0):
        rng = np.random.default_rng(seed)
        self.gamma = 1.0 + rng.normal(size=n)
        self.theta_x = rng.normal(scale=1e-3, size=n)
        self.theta_y = rng.normal(scale=1e-3, size=n)
        self.a0_shape = np.abs(rng.normal(size=n))
        self.chirp_mean = rng.normal(scale=1e-2, size=n)
        self.luminosity = np.abs(rng.normal(size=n))


@pytest.mark.parametrize("order", [3, 4, 5, 6])
def test_tensor_gauss_hermite_is_the_exact_gaussian_rule(order):
    """Nodes/weights must reproduce the normal's moments exactly, not approximately."""
    nodes, weights = C.tensor_gauss_hermite(order)
    assert nodes.shape == (order ** 6, 6)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    # Gauss-Hermite of order n is exact for polynomials of degree <= 2n-1, so with n >= 3
    # these three moments must be exact to roundoff.
    for axis in range(6):
        d = nodes[:, axis]
        assert float(np.sum(weights * d)) == pytest.approx(0.0, abs=1e-10)
        assert float(np.sum(weights * d ** 2)) == pytest.approx(1.0, abs=1e-10)
        assert float(np.sum(weights * d ** 4)) == pytest.approx(3.0, abs=1e-8)


@pytest.mark.parametrize("order", [3, 4, 6])
def test_tensor_gauss_hermite_weights_are_positive(order):
    """Positivity is what makes this usable as a macroparticle weight at all."""
    _, weights = C.tensor_gauss_hermite(order)
    assert weights.min() > 0.0


def test_tensor_gauss_hermite_rejects_bad_order():
    with pytest.raises(ValueError):
        C.tensor_gauss_hermite(0)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_global_qmc_weights_are_equal_and_sum_to_one(n):
    deviates, weights = C.global_qmc(n, seed=0)
    assert deviates.shape == (n, 6)
    assert np.all(weights == 1.0 / n)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_global_qmc_is_prefix_stable():
    """Point k must not move when n grows, or the convergence rate measures nothing."""
    small, _ = C.global_qmc(64, seed=3)
    large, _ = C.global_qmc(4096, seed=3)
    assert np.array_equal(small, large[:64])


def test_global_qmc_shifts_are_distinct_per_seed():
    """Different seeds must decorrelate; identical points would make seeds a fake replicate."""
    a, _ = C.global_qmc(256, seed=1)
    b, _ = C.global_qmc(256, seed=2)
    assert not np.array_equal(a, b)


def test_global_qmc_moments_approach_the_normal():
    deviates, weights = C.global_qmc(200_000, seed=0)
    assert float(np.sum(weights * deviates[:, 0] ** 2)) == pytest.approx(1.0, abs=2e-2)
    assert float(np.sum(weights * deviates[:, 0] ** 4)) == pytest.approx(3.0, abs=0.1)


def test_global_qmc_rejects_bad_input():
    with pytest.raises(ValueError):
        C.global_qmc(0)
    with pytest.raises(ValueError):
        C.global_qmc(8, dim=99)


def test_node_count_maps_tensor_order_to_trajectories():
    """The two budget conventions meet here and nowhere else."""
    assert C.node_count("tensor-gh", 4) == 4096
    assert C.node_count("global-qmc", 4096) == 4096
    assert C.node_count("iid", 1000) == 1000


def test_build_source_bunch_matches_the_shared_latent_map():
    """A cubature bunch must be the same object the IID path would build for those deviates."""
    from gammaforge.io.bunch import _bunch_from_standard_deviates
    from gammaforge.validation import scenarios

    beam = scenarios.BASELINE.beam
    deviates, weights = C.global_qmc(32, seed=0)
    built = C.build_source_bunch(beam, deviates, weights)
    direct = _bunch_from_standard_deviates(beam, deviates, weights)
    assert np.allclose(built.x, direct.x)
    assert np.allclose(built.weight, direct.weight)
    assert float(np.sum(built.weight)) == pytest.approx(1.0, abs=1e-12)


def test_build_source_bunch_rejects_mismatched_shapes():
    from gammaforge.validation import scenarios

    with pytest.raises(ValueError):
        C.build_source_bunch(scenarios.BASELINE.beam, np.zeros((4, 6)), np.ones(5))


def test_smooth_observables_do_not_double_count_the_weight():
    """`samples.luminosity` already carries weight and N_e; weighting again scales as 1/N."""
    samples = _Samples(4096, seed=1)
    weights = np.full(4096, 1.0 / 4096)
    standardization = S.Standardization.from_samples(samples, weights)
    value = S.evaluate(samples, weights, standardization)
    assert value["M0"] == pytest.approx(float(np.sum(samples.luminosity)), rel=1e-12)


def test_smooth_observables_reject_weight_length_mismatch():
    """The prefilter drops particles; catching the mismatch beats silently mis-weighting."""
    samples = _Samples(64, seed=1)
    standardization = S.Standardization(samples.gamma.mean() * np.ones(5),
                                        np.ones(5))
    with pytest.raises(ValueError):
        S.evaluate(samples, np.ones(32), standardization)


def test_relative_error_is_zero_against_itself_and_norm_scaled():
    value = {"M0": 2.0, "M1": np.array([1.0, -2.0, 3.0, 0.0, 4.0])}
    errors = S.relative_error(value, value)
    assert errors["M0"] == pytest.approx(0.0)
    assert errors["M1"] == pytest.approx(0.0)
    # A component passing through zero must not blow the vector error up.
    perturbed = {"M0": 2.0, "M1": np.array([1.0, -2.0, 3.0, 1e-18, 4.0])}
    assert S.relative_error(perturbed, value)["M1"] < 1e-12


def test_fit_exponent_recovers_a_known_slope():
    n = np.array([1e3, 1e4, 1e5, 1e6])
    errors = 2.0 * n ** -0.75
    fit = S.fit_exponent(n, errors)
    assert fit["resolved"]
    assert fit["alpha"] == pytest.approx(0.75, abs=1e-9)
    assert S.trajectories_for_error(n, errors, 1e-3) == pytest.approx(
        (2.0 / 1e-3) ** (1 / 0.75), rel=1e-6)


def test_fit_exponent_refuses_to_extrapolate_from_too_few_points():
    """A trajectory count quoted from three unresolved points is how a 100x claim happens."""
    fit = S.fit_exponent([10, 20], [1.0, 0.5])
    assert not fit["resolved"]
    assert math.isnan(S.trajectories_for_error([10, 20], [1.0, 0.5], 1e-3))


def test_fit_exponent_excludes_points_at_the_reference_floor():
    """The defect this exists to prevent: a curve flattening on the *reference*, not converging.

    With a 400k reference the spectral floor is ~3e-3; IID reached 3.6e-3 at 262k, and the
    apparent slope was an artefact of the reference, not the sampler.
    """
    n = np.array([1e3, 1e4, 1e5, 1e6])
    # Every point below 3x the 3e-3 floor, so a correct fit must refuse rather than report a slope.
    errors = 0.2 * n ** -0.75
    gated = S.fit_exponent(n, errors, floor=3e-3, margin=3.0)
    assert gated["n_points"] == 0 and not gated["resolved"]
    assert math.isnan(gated["alpha"])
    assert math.isnan(S.trajectories_for_error(n, errors, 1e-3, floor=3e-3))


def test_fit_exponent_keeps_points_standing_clear_of_the_floor():
    n = np.array([1e3, 1e4, 1e5, 1e6])
    errors = 2.0 * n ** -0.75
    ungated = S.fit_exponent(n, errors, floor=0.0)
    assert ungated["resolved"] and ungated["alpha"] == pytest.approx(0.75, abs=1e-9)
    # A floor far below every measured point must not change the fit.
    low_floor = S.fit_exponent(n, errors, floor=1e-9)
    assert low_floor["n_points"] == ungated["n_points"]


def test_resolved_mask_is_margin_scaled():
    errors = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    # threshold is 3e-3, so 1e-2 and 1e-1 qualify and the two smaller points do not
    assert S.resolved_mask(errors, 1e-3, margin=3.0).tolist() == [False, False, True, True]
    assert S.resolved_mask(errors, 0.0).tolist() == [True] * 4


def test_reference_from_runs_measures_the_floor():
    """A floor is the disagreement between independent references, not an assumption."""
    a = {"M0": 1.0, "M1": np.array([2.0, 3.0])}
    b = {"M0": 1.0, "M1": np.array([2.0, 3.0])}
    mean, floor = S.reference_from_runs([a, b])
    assert floor == pytest.approx(0.0)
    assert mean["M0"] == pytest.approx(1.0)
    c = {"M0": 1.1, "M1": np.array([2.0, 3.0])}
    # Each run sits 0.05 from the mean of 1.05, so the measured *relative* spread is 0.0476 --
    # a half-difference, because the spread is measured from the mean, not between the runs.
    _, floor2 = S.reference_from_runs([a, c])
    assert floor2 == pytest.approx(0.05 / 1.05, abs=1e-12)
    # A single construction has no disagreement to measure, so reports zero rather than a guess.
    _, floor1 = S.reference_from_runs([a])
    assert floor1 == 0.0


def test_effective_sample_fraction_flags_concentrated_weights():
    assert S.effective_sample_fraction(np.full(1000, 1.0 / 1000)) == pytest.approx(1.0)
    """This is the check that would have caught the Gauss-Hermite table result up front."""
    _, w = C.tensor_gauss_hermite(8)
    assert S.effective_sample_fraction(w) < 0.01


def test_smolyak_probe_is_unavailable_and_says_why():
    """The sparse grid is out this phase; the refusal must be loud and specific."""
    assert C.smolyak_probe_available() is False
    with pytest.raises(ValueError, match="partition of unity"):
        C.smolyak_gauss_hermite()