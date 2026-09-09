from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from gammaforge.engines.xigma.stages import TrajectorySamples
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import diagnose_delta_doppler as diagnostic


def _samples():
    return TrajectorySamples(
        gamma=np.array([2000.0, 2100.0]), theta_x=np.array([0.0, 0.001]),
        theta_y=np.array([0.0, -0.001]), a0_shape=np.zeros(2),
        luminosity=np.array([1.0, 2.0]), intensity_peak=1.0, n_steps=1,
    )


def _emission(samples, tx, ty, *, photon_energy, doppler, **kwargs):
    nominal = np.array([photon_energy, 2.0 * photon_energy])
    energies = nominal if doppler == "nominal" else nominal * 1.01
    return energies.astype(np.longdouble), np.array([1.0, 2.0], dtype=np.longdouble)


def test_shifted_lines_are_monotonic_and_bins_are_finite():
    result = diagnostic._line_diagnostic(
        _samples(), 1.0, 0.0, 0.0, psi_pol=0.0, ellipticity=0.0,
        theta_xz=0.0, theta_yz=0.0, emission_fn=_emission,
    )
    assert result["weighted_centroid_relative_shift"] == pytest.approx(0.01, rel=1e-12)
    assert result["weighted_rms_relative_line_shift"] == pytest.approx(0.01, rel=1e-12)
    assert all(np.isfinite(result["physical_energy_edges_erg"]))
    assert result["bin_l1_relative"] >= 0.0


def test_monoenergetic_edges_are_padded_and_strictly_increasing():
    edges = diagnostic._energy_edges(np.array([2.0, 2.0]))
    assert edges[0] == pytest.approx(1.6)
    assert edges[-1] == pytest.approx(2.4)
    assert np.all(np.diff(edges) > 0)


def test_changed_weights_are_rejected():
    def altered(samples, tx, ty, *, photon_energy, doppler, **kwargs):
        energies, weights = _emission(samples, tx, ty, photon_energy=photon_energy, doppler="nominal")
        return energies, weights * (2.0 if doppler == "particle" else 1.0)

    with pytest.raises(ValueError, match="identical line weights"):
        diagnostic._line_diagnostic(_samples(), 1.0, 0.0, 0.0, psi_pol=0.0,
                                    ellipticity=0.0, theta_xz=0.0, theta_yz=0.0,
                                    emission_fn=altered)


def test_identical_lines_have_zero_shift():
    def identical(samples, tx, ty, *, photon_energy, doppler, **kwargs):
        return _emission(samples, tx, ty, photon_energy=photon_energy, doppler="nominal")

    result = diagnostic._line_diagnostic(
        _samples(), 1.0, 0.0, 0.0, psi_pol=0.0, ellipticity=0.0,
        theta_xz=0.0, theta_yz=0.0, emission_fn=identical,
    )
    assert result["weighted_centroid_relative_shift"] == 0.0
    assert result["weighted_rms_relative_line_shift"] == 0.0


def test_cli_writes_strict_json(monkeypatch, tmp_path):
    report = {"status": "completed", "scientific_pass": False, "records": [], "stress_lines": []}
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda **kwargs: report)
    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr(sys, "argv", ["diagnose_delta_doppler.py", "--output", str(output)])
    diagnostic.main()
    assert json.loads(output.read_text()) == report


def test_cli_failed_report_is_nonzero_and_written(monkeypatch, tmp_path):
    report = {"status": "failed", "scientific_pass": False}
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda **kwargs: report)
    output = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["diagnose_delta_doppler.py", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 1
    assert json.loads(output.read_text()) == report


def test_source_hash_change_fails_closed(monkeypatch):
    hashes = iter(({"source": "before"}, {"source": "after"}))
    monkeypatch.setattr(diagnostic, "_hash_sources", lambda: next(hashes))
    monkeypatch.setattr(diagnostic, "_stress_diagnostics", lambda photon_energy: [])
    monkeypatch.setattr(diagnostic, "_line_diagnostic", lambda *args, **kwargs: {
        "weighted_centroid_relative_shift": 0.0,
        "weighted_rms_relative_line_shift": 0.0,
    })
    report = diagnostic.run_diagnostic(
        particles=2, n_steps=1, scenario_list=[diagnostic.scenarios.SCENARIOS[0]],
    )
    assert report["status"] == "failed"
    assert report["scientific_pass"] is False
    assert report["source_sha256_before"] != report["source_sha256_after"]
