#!/usr/bin/env python3
"""Fail-closed real-CUDA delta agreement packet; numerical parity, not physics closure."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from gammaforge.engines.xigma import stages
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios
from gammaforge.validation.references import delta_cupy, delta_emission


def fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(module.__file__).resolve() for module in
             (stages, scenarios, delta_cupy, delta_emission)] + [Path(__file__).resolve()]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(particles=4096, seeds=(20260721, 20260914)):
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("actual CUDA execution is required")
    before = fingerprints()
    records = []
    first_call_seconds = None
    variants = (("headon_linear", 0., 0., 0., 0.),
                ("crossed_elliptic", .03, -.02, .4, .37),
                ("crossed_circular", -.03, .01, 1., .71))
    for scenario in scenarios.SCENARIOS:
        for seed in seeds:
            for name, xz, yz, eps, psi in variants:
                laser = replace(scenario.laser, theta_xz=Quantity(xz, "rad"),
                                theta_yz=Quantity(yz, "rad"), ellipticity=eps,
                                psi_pol=Quantity(psi, "rad"))
                interaction = scenarios.build(replace(scenario, laser=laser),
                    replace(scenario.sampling, n_particles=particles, seed=seed))
                samples = stages.integrate_trajectories(interaction.bunch, laser, interaction.N_e,
                                                        n_steps=64, backend="numpy")
                gamma0 = float(interaction.beam.gamma0())
                for tx, ty in ((0., 0.), (.5/gamma0, -.25/gamma0)):
                    kw = dict(photon_energy=float(laser.photon_energy()), theta_xz=xz, theta_yz=yz,
                              ellipticity=eps, psi_pol=psi, doppler="direction")
                    e, w = delta_emission.emission_lines(samples, tx, ty, **kw)
                    if not e.size or not w.sum() > 0:
                        raise RuntimeError("empty case cannot establish numerical agreement")
                    edges = np.geomspace(float(e.min())*.9, float(e.max())*1.1, 65)
                    ref = delta_emission.bin_emission(e, w, edges)
                    for chunk in (127, delta_cupy.DEFAULT_CHUNK):
                        start = time.perf_counter()
                        de, dw = delta_cupy.emission_lines(samples, tx, ty, chunk=chunk, **kw)
                        actual = delta_cupy.bin_emission(de, dw, edges, chunk=chunk)
                        cp.cuda.get_current_stream().synchronize()
                        if first_call_seconds is None:
                            first_call_seconds = time.perf_counter()-start
                        metrics = {
                            "max_relative_energy_error": float(np.max(np.abs(de/e-1))),
                            "max_weight_error_over_peak": float(np.max(np.abs(dw-w))/np.max(w)),
                            "relative_total_weight_error": abs(actual["total_weight"]/ref["total_weight"]-1),
                            "spectral_l1": float(np.sum(np.abs(actual["bin_mass"]-ref["bin_mass"]))/ref["total_weight"]),
                        }
                        scale = 2*kw["photon_energy"]*(1+np.cos(xz)*np.cos(yz))
                        query = dict(theta_xz=xz, theta_yz=yz, ellipticity=eps, psi_pol=psi,
                                     doppler="direction", chunk=chunk)
                        # Warm complete normalized-spectrum calls, including host/device transfers.
                        delta_cupy.resonance_spectrum(samples, edges/scale, tx, ty, **query)
                        times = []
                        for _ in range(3):
                            start = time.perf_counter()
                            spectrum = delta_cupy.resonance_spectrum(samples, edges/scale, tx, ty, **query)
                            cp.cuda.get_current_stream().synchronize()
                            times.append(time.perf_counter()-start)
                        metrics["normalized_spectral_l1"] = float(np.sum(np.abs(
                            spectrum*np.diff(edges/scale)-ref["bin_mass"]))/ref["total_weight"])
                        limits = {key: 3e-8 for key in metrics}
                        limits["max_relative_energy_error"] = 3e-13
                        passed = all(np.isfinite(value) and value <= limits[key]
                                     for key, value in metrics.items())
                        records.append(dict(scenario=scenario.name, seed=seed, variant=name,
                            observer=[tx, ty], actual_particles=samples.n_particles, chunk=chunk,
                            errors=metrics, limits=limits, passed=bool(passed),
                            warm_spectrum_seconds=float(np.median(times))))
    after = fingerprints()
    passed = before == after and bool(records) and all(r["passed"] for r in records)
    return dict(status="passed" if passed else "failed", scientific_acceptance=False,
        source_sha256_before=before, source_sha256_after=after,
        environment=dict(python=sys.version, numpy=np.__version__, cupy=cp.__version__,
            gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(), platform=platform.platform()),
        settings=dict(particles=particles, seeds=list(seeds), n_steps=64, bins=64,
                      doppler="direction", dtype="float64"),
        first_call_seconds=first_call_seconds, records=records,
        caveat="GPU/independent CPU numerical parity on identical Stage-0 samples; "
               "not particle convergence, xigma speedup, or full arbitrary-angle physics closure.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.particles < 1:
        parser.error("particles must be positive")
    try:
        report = run(args.particles)
    except Exception as exc:
        report = dict(status="failed", scientific_acceptance=False,
                      error=f"{type(exc).__name__}: {exc}")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Delta CUDA gate: {report['status']}; {len(report.get('records', []))} cases; {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
