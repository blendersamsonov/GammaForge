"""Diagnostic for nominal versus per-particle delta-emission Doppler factors.

This is deliberately a diagnostic, not a scientific acceptance test.  It compares
line energies while keeping the Stage-0 samples and line weights identical.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from gammaforge.engines.xigma.stages import TrajectorySamples, integrate_trajectories
import gammaforge.io.laser as laser_module
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios
from gammaforge.validation.references.delta_emission import bin_emission, emission_lines


VARIANTS = (
    ("baseline", 0.0, 0.0, 0.0, 0.0),
    ("crossed_small", 0.02, -0.015, 0.4, 0.37),
    ("crossed_stress", 0.3, 0.2, 0.4, 0.37),
)
def _json_float(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("diagnostic produced a non-finite value")
    return value


def _energy_edges(energies: np.ndarray, bins: int = 32) -> np.ndarray:
    finite = np.asarray(energies, dtype=float)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi < lo:
        raise ValueError("cannot construct finite positive physical energy bins")
    if hi == lo:
        # Keep a useful capture window for a monoenergetic line.  A tiny
        # epsilon-width interval is numerically fragile and does not exercise
        # the same under/overflow bookkeeping as the general case.
        return np.geomspace(lo * 0.8, lo * 1.2, bins + 1)
    # Geometric bins are intentionally nonuniform and common to both line sets.
    return np.geomspace(lo * 0.8, hi * 1.2, bins + 1)


def _line_diagnostic(samples, photon_energy, tx, ty, *, psi_pol, ellipticity, theta_xz, theta_yz,
                     emission_fn=emission_lines, bin_fn=bin_emission):
    common = dict(psi_pol=psi_pol, ellipticity=ellipticity, theta_xz=theta_xz, theta_yz=theta_yz)
    nominal_e, nominal_w = emission_fn(samples, tx, ty, photon_energy=photon_energy, doppler="nominal", **common)
    particle_e, particle_w = emission_fn(samples, tx, ty, photon_energy=photon_energy, doppler="particle", **common)
    nominal_e, nominal_w = np.asarray(nominal_e), np.asarray(nominal_w)
    particle_e, particle_w = np.asarray(particle_e), np.asarray(particle_w)
    if nominal_e.shape != particle_e.shape or nominal_w.shape != particle_w.shape:
        raise ValueError("nominal and particle line arrays differ in shape")
    if not np.array_equal(nominal_w, particle_w):
        raise ValueError("nominal and particle runs did not preserve identical line weights")
    if not (np.all(np.isfinite(nominal_e)) and np.all(np.isfinite(particle_e)) and np.all(np.isfinite(nominal_w)) and np.all(nominal_w >= 0.0)):
        raise ValueError("line arrays are non-finite or have negative weights")
    total = float(np.sum(nominal_w))
    if total <= 0.0:
        raise ValueError("line weights are all zero")
    shift = particle_e / nominal_e - 1.0
    centroid_nominal = float(np.sum(nominal_e * nominal_w) / total)
    centroid_particle = float(np.sum(particle_e * nominal_w) / total)
    rms = float(np.sqrt(np.sum(nominal_w * shift * shift) / total))
    edges = _energy_edges(np.concatenate((nominal_e, particle_e)))
    nominal_bins = bin_fn(nominal_e, nominal_w, edges)
    particle_bins = bin_fn(particle_e, nominal_w, edges)
    masses_n = np.asarray(nominal_bins["bin_mass"], dtype=float)
    masses_p = np.asarray(particle_bins["bin_mass"], dtype=float)
    bin_l1 = float(np.sum(np.abs(masses_p - masses_n)) / total)
    return {
        "observer": {"theta_x": _json_float(tx), "theta_y": _json_float(ty)},
        "line_count": int(nominal_e.size),
        "weighted_centroid_nominal_erg": _json_float(centroid_nominal),
        "weighted_centroid_particle_erg": _json_float(centroid_particle),
        "weighted_centroid_relative_shift": _json_float(centroid_particle / centroid_nominal - 1.0),
        "weighted_rms_relative_line_shift": _json_float(rms),
        "max_absolute_relative_line_shift_positive_weight": _json_float(
            np.max(np.abs(shift[nominal_w > 0.0]))
        ),
        "physical_energy_edges_erg": [_json_float(v) for v in edges],
        "bin_l1_relative": _json_float(bin_l1),
        "underflow_nominal": _json_float(nominal_bins["underflow"]),
        "overflow_nominal": _json_float(nominal_bins["overflow"]),
        "underflow_particle": _json_float(particle_bins["underflow"]),
        "overflow_particle": _json_float(particle_bins["overflow"]),
        "total_weight": _json_float(total),
    }


def _stress_diagnostics(photon_energy: float) -> list[dict]:
    result = []
    cases = [("gamma2000_tx0", 2000.0, 0.0), ("gamma2000_tx+001", 2000.0, 0.001),
             ("gamma2000_tx-001", 2000.0, -0.001), ("gamma2000_tx+01", 2000.0, 0.01),
             ("gamma2000_tx-01", 2000.0, -0.01), ("gamma10000_headon", 10000.0, 0.0)]
    for name, gamma, tx in cases:
        samples = TrajectorySamples(np.array([gamma]), np.array([tx]), np.array([0.0]), np.zeros(1), np.ones(1), 1.0, 1)
        txz, tyz, eps, psi = (0.3, 0.2, 0.4, 0.37) if gamma == 2000.0 else (0.0, 0.0, 0.0, 0.0)
        result.append({"name": name, "particle": {"gamma": gamma, "theta_x": tx, "theta_y": 0.0,
                         "theta_x_slope": tx, "theta_y_slope": 0.0,
                         "theta_xz": txz, "theta_yz": tyz}, "diagnostic": _line_diagnostic(
            samples, photon_energy, 0.0, 0.0, psi_pol=psi, ellipticity=eps, theta_xz=txz, theta_yz=tyz,
        )})
    return result


def _hash_sources() -> dict[str, str]:
    paths = [Path(__file__), Path(emission_lines.__code__.co_filename), Path(integrate_trajectories.__code__.co_filename),
             Path(scenarios.__file__), Path(laser_module.__file__)]
    out = {}
    for path in paths:
        out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def run_diagnostic(*, particles=16_000, seed=20260721, n_steps=64, scenario_list=None,
                   emission_fn=emission_lines, bin_fn=bin_emission) -> dict:
    scenario_list = tuple(scenario_list or scenarios.SCENARIOS)
    source_before = _hash_sources()
    records = []
    for scenario in scenario_list:
        for variant_name, txz, tyz, eps, psi in VARIANTS:
            laser = replace(scenario.laser, theta_xz=Quantity(txz, "rad"), theta_yz=Quantity(tyz, "rad"),
                            ellipticity=eps, psi_pol=Quantity(psi, "rad"))
            interaction = scenarios.build(replace(scenario, laser=laser), SamplingSpec(
                n_particles=particles, seed=seed, prefilter=1e-3,
            ))
            samples = integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e, n_steps=n_steps)
            gamma0 = float(interaction.beam.gamma0())
            photon_energy = float(interaction.laser.photon_energy())
            observations = [(0.0, 0.0), (0.5 / gamma0, -0.25 / gamma0)]
            obs = [_line_diagnostic(samples, photon_energy, tx, ty, psi_pol=psi, ellipticity=eps, theta_xz=txz, theta_yz=tyz,
                                    emission_fn=emission_fn, bin_fn=bin_fn) for tx, ty in observations]
            records.append({"scenario": scenario.name, "variant": variant_name, "inputs": {
                "particles": particles, "seed": seed, "n_steps": n_steps, "theta_xz": txz, "theta_yz": tyz,
                "ellipticity": eps, "psi_pol": psi, "gamma0": gamma0, "photon_energy_erg": photon_energy,
                "scenario_repr": repr(scenario), "target_repr": repr(interaction.target),
                }, "observations": obs,
            })
    # Include the stress matrix in the timed region: a source mutation during
    # any diagnostic work must make the report fail closed.
    stress_lines = _stress_diagnostics(float(scenario_list[0].laser.photon_energy()))
    source_after = _hash_sources()
    status = "completed" if source_before == source_after else "failed"
    return {"status": status, "scientific_pass": False, "caveat": "Shared Stage-0 luminosity/ahat and the reduced line model are not independently validated; this is not a full arbitrary-angle validation.",
            "precision": {"numpy": np.__version__, "longdouble_bits": int(np.finfo(np.longdouble).nmant), "longdouble_decimal_digits": int(np.finfo(np.longdouble).precision),
                          "python": sys.version, "platform": platform.platform()},
            "source_sha256_before": source_before, "source_sha256_after": source_after,
            "settings": {"particles": particles, "seed": seed, "n_steps": n_steps},
            "records": records, "stress_lines": stress_lines}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.particles < 1 or args.n_steps < 1:
        parser.error("--particles and --n-steps must be positive")
    try:
        report = run_diagnostic(particles=args.particles, seed=args.seed, n_steps=args.n_steps)
    except Exception as exc:
        report = {"status": "failed", "scientific_pass": False, "error": f"{type(exc).__name__}: {exc}"}
        payload = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        raise SystemExit(1) from exc
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if report.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
