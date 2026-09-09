import numpy as np
import pytest

from gammaforge.validation.delta_comparison import compare_emission_bins, integrate_density_bins

pytestmark = pytest.mark.tier0


def test_constant_and_linear_nonuniform_bins_exact():
    edges = np.array([1., 1.3, 2.7, 5.])
    r = integrate_density_bins(lambda x: 2 + 3*x, edges, quadrature_order=3)
    assert np.allclose(r['bin_mass'], 2*np.diff(edges) + 1.5*(edges[1:]**2-edges[:-1]**2))
    assert np.allclose(r['first_moment'], (edges[1:]**2-edges[:-1]**2) + (edges[1:]**3-edges[:-1]**3))
    assert r['refinement_error'] < 1e-12
    assert r['converged']


def test_line_moments_and_injected_errors():
    edges = [1., 2., 4.]
    good = compare_emission_bins(lambda x: np.where(x < 2, 1., 2./2.), edges, [1.5, 3.], [1., 2.])
    assert good['reference_total'] == 3
    assert good['reference_centroid'] == pytest.approx(2.5)
    wrong = compare_emission_bins(lambda x: np.full_like(x, .5), edges, [1.5, 3.], [1., 2.])
    assert wrong['relative_yield_error'] < 0
    shifted = compare_emission_bins(lambda x: np.where(x < 2, 0., 1.), edges, [1.5, 3.], [1., 2.])
    assert shifted['candidate_mass_in_reference_zero_bins'] == 0


def test_zero_and_support_tails():
    r = compare_emission_bins(lambda x: np.zeros_like(x), [1., 2.], [0.5, 1.5, 3.], [2., 1., 4.])
    assert r['reference_underflow'] == 2 and r['reference_overflow'] == 4
    assert r['candidate_total'] == 0


@pytest.mark.parametrize('bad', [lambda x: np.full_like(x, -1.), lambda x: np.full(x.shape, np.nan), lambda x: np.ones(x.size+1)])
def test_invalid_density_rejected(bad):
    with pytest.raises(ValueError):
        integrate_density_bins(bad, [1., 2.])


def test_invalid_edges_and_lines():
    with pytest.raises(ValueError): integrate_density_bins(lambda x: x, [0., 1.])
    with pytest.raises(ValueError): compare_emission_bins(lambda x: x, [1., 2.], [1.], [-1.])


def test_unresolved_refinement_is_reported():
    r = integrate_density_bins(lambda x: np.exp(-100*x), [1., 2.], quadrature_order=2)
    assert r['refinement_error'] > 1e-6
    assert not r['converged']


def test_energy_shift_is_detected_without_changing_counts():
    # Uniform density on [2, 3] versus a line at 1.5: equal masses, shifted centroid.
    result = compare_emission_bins(lambda x: (x >= 2).astype(float),
                                   [1., 2., 3.], [1.5], [1.])
    assert result['relative_yield_error'] == pytest.approx(0., abs=1e-14)
    assert result['centroid_relative_error'] == pytest.approx(2.5 / 1.5 - 1)
    assert result['l1_mass_error'] == pytest.approx(2.)
    assert result['candidate_mass_in_reference_zero_bins'] == pytest.approx(1.)
    assert not result['zero_reference_with_candidate_signal']


@pytest.mark.parametrize('signal', [0., 1.])
def test_empty_reference_is_explicit_and_json_safe(signal):
    import json
    result = compare_emission_bins(lambda x: np.full_like(x, signal), [1., 2.], [], [])
    assert result['reference_centroid'] is None
    assert result['relative_yield_error'] is None
    assert result['l1_mass_error'] is None
    assert result['zero_reference_with_candidate_signal'] is bool(signal)
    json.dumps(result, default=lambda value: value.tolist(), allow_nan=False)


def test_final_edge_and_exact_window_moment():
    result = compare_emission_bins(lambda x: np.ones_like(x),
                                   [1., 2., 4.], [.5, 1., 2., 4., 5.], [7., 1., 2., 3., 9.])
    assert result['reference_bin_mass'].tolist() == [1., 5.]
    assert result['reference_centroid'] == pytest.approx(17 / 6)
    assert result['reference_underflow'] == 7
    assert result['reference_overflow'] == 9


def test_candidate_metrics_use_refined_integral():
    result = compare_emission_bins(lambda x: x**6, [1., 2.], [1.5], [1.], quadrature_order=2)
    assert result['candidate_total'] == pytest.approx((2**7 - 1) / 7)
    assert result['candidate_total'] != pytest.approx(result['candidate']['bin_mass'].sum(), rel=1e-4)


def test_overflowing_integrals_are_rejected():
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='non-finite'):
            integrate_density_bins(lambda x: np.full_like(x, 1e308), [1., 10.])
