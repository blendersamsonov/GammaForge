"""Acceptance must distinguish convergence from persistent spectral error."""
from types import SimpleNamespace

import numpy as np
import pytest

from gammaforge.validation import delta_validation as validation
from gammaforge.validation.delta_comparison import compare_emission_bins

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize('defect', [None, 'spectral_bias', 'unconverged', 'missing_grid', 'changed_source'])
def test_bounded_refinement_preserves_acceptance_gates(monkeypatch, defect):
    calls = []

    def measure(*, source_scenarios, variants, table_configs, quadrature_order):
        names = [scenario.name for scenario in source_scenarios]
        calls.append(names)
        refined = len(calls) == 2
        records = []
        for name in names:
            for variant in variants:
                for observer in (0, 1):
                    for index, (shape, retarget) in enumerate(table_configs):
                        # Equal count, but coarse redistribution exceeds the L1
                        # refinement budget while reference agreement still passes.
                        unresolved = name == 'unresolved' and (not refined or defect == 'unconverged')
                        shift = .03 if unresolved and index == 0 else 0.
                        scale = 1.1 if refined and defect == 'spectral_bias' else 1.
                        comparison = compare_emission_bins(
                            lambda energy: scale * np.where(energy < 2, 1 + shift, 1 - shift),
                            [1., 2., 3.], [1.5, 2.5], [1., 1.],
                            quadrature_order=quadrature_order,
                        )
                        records.append(dict(scenario=name, variant=variant[0],
                                            observer={'index': observer}, shape=shape,
                                            retarget_bins=retarget, edges=[1., 2., 3.],
                                            comparison=comparison))
        if refined and defect == 'missing_grid':
            records.pop()
        return dict(status='changed' if defect == 'changed_source' and not refined else 'completed',
                    settings={'scenarios': names}, records=records)

    monkeypatch.setattr(validation, 'run_pilot', measure)
    checks, notes, blockers = validation.production_checks(
        [SimpleNamespace(name='converged'), SimpleNamespace(name='unresolved')]
    )

    assert calls == [['converged', 'unresolved'], ['unresolved']]
    assert any('Initial grid diagnostic:' in note and 'refinement l1' in note for note in notes)
    assert blockers  # Numerical convergence never closes unmeasured physics coverage.
    failures = [check.name for check in checks if not check.passed]
    if defect is None:
        assert not failures
    else:
        expected = {'spectral_bias': 'xigma/delta yield',
                    'unconverged': 'angular table refinement l1',
                    'missing_grid': 'delta refinement coverage',
                    'changed_source': 'delta source stability'}[defect]
        assert any(expected in name for name in failures)
