"""Fast orchestration checks; scientific acceptance is deliberately out of scope."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gammaforge.engines.xigma.stages import TrajectorySamples

pytestmark = pytest.mark.tier0


@pytest.fixture
def pilot():
    path = Path(__file__).parents[1] / "scripts" / "validate_delta_emission.py"
    spec = importlib.util.spec_from_file_location("delta_pilot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('source_changed', [False, True])
def test_pilot_converts_energy_once_and_reuses_samples(pilot, monkeypatch, source_changed):
    samples = TrajectorySamples(
        gamma=np.array([2000., 2100.]), theta_x=np.zeros(2), theta_y=np.zeros(2),
        a0_shape=np.array([.1, .2]), luminosity=np.ones(2), intensity_peak=.01, n_steps=1,
    )
    monkeypatch.setattr(pilot.scenarios, 'SCENARIOS', pilot.scenarios.SCENARIOS[:1])
    monkeypatch.setattr(pilot, 'VARIANTS', (("crossed", .3, -.2, .4, .37),))
    monkeypatch.setattr(pilot, 'integrate_trajectories', lambda *a, **kw: samples)
    fingerprints = iter(({'source': 'before'}, {'source': 'after' if source_changed else 'before'}))
    monkeypatch.setattr(pilot, '_hash_sources', lambda: next(fingerprints))
    seen = []

    def deposit(actual, **kwargs):
        assert actual is samples
        return object()

    monkeypatch.setattr(pilot, 'deposit_shape_table', deposit)
    monkeypatch.setattr(pilot, 'retarget_ahat', lambda *a, **kw: SimpleNamespace(H=np.zeros((2, 2, 2, 2))))
    expected_scale = 4 * float(pilot.scenarios.SCENARIOS[0].laser.photon_energy()) * (1 + np.cos(.3) * np.cos(-.2)) / 2

    def lines(actual, *args, **kwargs):
        assert actual is samples
        return np.array([expected_scale, 2 * expected_scale]), np.ones(2)

    monkeypatch.setattr(pilot, 'emission_lines', lines)

    def spectrum(table, tx, ty, s, **geometry):
        assert geometry == dict(theta_xz=.3, theta_yz=-.2, ellipticity=.4, psi_pol=.37)
        np.testing.assert_allclose(s, [1., 2.], rtol=1e-14)
        seen.append((tx, ty))
        return np.array([3., 5.])

    monkeypatch.setattr(pilot, 'spectrum_from_table', spectrum)

    def compare(density, edges, energies, weights, **kwargs):
        np.testing.assert_allclose(density(expected_scale * np.array([1., 2.])),
                                   np.array([3., 5.]) / expected_scale, rtol=1e-14)
        return {'candidate': {'converged': True}}

    monkeypatch.setattr(pilot, 'compare_emission_bins', compare)
    report = pilot.run_pilot(particles=2, n_steps=1)
    assert len(seen) == 6  # Three tables, two observers; all share the same samples.
    assert len(set(seen)) == 2
    assert report['status'] == ('failed' if source_changed else 'completed')
    assert not report['scientific_pass']


def test_failed_cli_writes_report_and_returns_nonzero(pilot, monkeypatch, tmp_path, capsys):
    output = tmp_path / 'failed.json'
    monkeypatch.setattr(pilot.sys, 'argv', ['validate_delta_emission.py', '--output', str(output)])
    monkeypatch.setattr(pilot, 'run_pilot', lambda *a, **kw: {'status': 'failed', 'scientific_pass': False})
    assert pilot.main() == 1
    assert json.loads(output.read_text())['status'] == 'failed'
    capsys.readouterr()


def test_cli_reports_calculation_error(pilot, monkeypatch, capsys):
    monkeypatch.setattr(pilot.sys, 'argv', ['validate_delta_emission.py'])

    def fail(*args, **kwargs):
        raise ValueError('injected failure')

    monkeypatch.setattr(pilot, 'run_pilot', fail)
    assert pilot.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert 'injected failure' in report['error']
    assert not report['scientific_pass']
