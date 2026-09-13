"""Failure semantics of the independent production emission gate."""
from copy import deepcopy

import numpy as np
import pytest

from gammaforge.validation import delta_validation as delta
from gammaforge.validation import run
from gammaforge.validation.invariance import Check

pytestmark = pytest.mark.tier0


@pytest.fixture
def packet():
    records = []
    for variant in delta.PRODUCTION_VARIANTS:
        for observer in (0, 1):
            for shape, retarget in delta.PRODUCTION_TABLE_CONFIGS:
                records.append({
                    'scenario': 'example', 'variant': variant[0],
                    'observer': {'index': observer}, 'shape': shape,
                    'retarget_bins': retarget, 'edges': [1., 2., 3.],
                    'comparison': {
                        'relative_yield_error': 0., 'l1_mass_error': 0.,
                        'centroid_relative_error': 0.,
                        'candidate': {
                            'refinement_errors': dict.fromkeys(delta.AGREEMENT_BUDGETS, 0.),
                            'refined_bin_mass': np.array([1., 1.]),
                            'refined_first_moment': np.array([1.5, 2.5]),
                        },
                    },
                })
    return {'status': 'completed', 'settings': {'scenarios': ['example']}, 'records': records}


def test_complete_numerically_converged_packet_passes(packet):
    assert all(c.passed for c in delta.packet_checks(packet))


@pytest.mark.parametrize('defect', [
    'missing_case', 'missing_grid', 'duplicate_grid', 'changed_edges', 'changed_source',
    'missing_quadrature_metric', 'unconverged_quadrature', 'failed_agreement',
    'unconverged_table', 'zero_signal', 'nan',
])
def test_incomplete_or_unconverged_packet_fails(packet, defect):
    first = packet['records'][0]
    candidate = first['comparison']['candidate']
    if defect == 'missing_case':
        packet['records'] = packet['records'][3:]
    elif defect == 'missing_grid':
        packet['records'].pop(0)
    elif defect == 'duplicate_grid':
        packet['records'].insert(0, deepcopy(first))
    elif defect == 'changed_edges':
        first['edges'] = [1., 2., 4.]
    elif defect == 'changed_source':
        packet['status'] = 'failed'
    elif defect == 'missing_quadrature_metric':
        del candidate['refinement_errors']['l1']
    elif defect == 'unconverged_quadrature':
        candidate['refinement_errors']['centroid'] = .004
    elif defect == 'failed_agreement':
        packet['records'][2]['comparison']['relative_yield_error'] = .04
    elif defect == 'unconverged_table':
        candidate['refined_bin_mass'] *= 1.02
    elif defect == 'zero_signal':
        candidate['refined_bin_mass'][:] = 0
    elif defect == 'nan':
        candidate['refined_bin_mass'][0] = np.nan
    assert any(not c.passed for c in delta.packet_checks(packet))


def test_only_production_runs_delta_and_propagates_its_failure(monkeypatch):
    for name in ('core_checks', 'identity_checks', 'golden_scalar_checks'):
        monkeypatch.setattr(run, name, lambda scenarios: [])
    monkeypatch.setattr(run, 'production_checks', lambda scenarios: ([], [], []))
    seen = []

    def measure(scenarios):
        seen.append(scenarios)
        return [Check('delta error', False, 'injected')], ['measured delta'], ['sampling open']

    monkeypatch.setattr(delta, 'production_checks', measure)
    run.run_suite(scenarios=[], alpha=True)
    run.run_suite(scenarios=[])
    assert not seen
    assert run.main(['--production'], scenarios=[]) == 1
    assert seen == [[]]
    report = run.run_suite(scenarios=[], production=True)
    assert report.failures == 1 and report.blockers == 1
    assert 'measured delta' in str(report)
