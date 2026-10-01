"""Reproduce DER023 Stage-0 convergence, bounds, DER001, and device measurements.

Run with ``PYTHONPATH=src python scripts/validate_der023_stage0.py --output FILE``.
The CPU checks run without CUDA; ``--gpu`` additionally requires a real device.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from gammaforge.engines.analytical.formulas import overlap_yield
from gammaforge.engines.xigma import chunking
from gammaforge.engines.xigma.gaussian_bound import (
    luminosity_upper_bound,
    trajectory_bound_geometry,
)
from gammaforge.engines.xigma.stages import (BYTES_PER_PARTICLE_STEP,
                                              integrate_trajectories,
                                              photon_density_scale)
from gammaforge.io.bunch import sample_gaussian_bunch
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.laser import PulseTrainParaxialLaser
from gammaforge.io.units import C_CGS, SIGMA_T_CGS, Quantity
from gammaforge.validation.scenarios import SCENARIOS, build


CHANNELS = (
    "luminosity", "a0_shape", "chirp_mean", "var_a_shape", "var_chirp",
    "cov_a_chirp_shape",
)


def errors(actual, reference):
    result = {}
    for name in CHANNELS:
        a, b = getattr(actual, name), getattr(reference, name)
        norm = float(np.sum(np.abs(b)))
        result[name] = float(np.sum(np.abs(a - b)) / norm) if norm else float(np.max(np.abs(a - b)))
    return result


def timing(call, repeats=3):
    call()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        times.append(perf_counter() - start)
    return median(times)


def scenario_scan():
    records = []
    for scenario in SCENARIOS:
        interaction = build(scenario, SamplingSpec(n_particles=128, seed=20260721, prefilter=0))
        args = (interaction.bunch, interaction.laser, interaction.N_e)
        reference = integrate_trajectories(*args, n_steps=4096)
        midpoint = integrate_trajectories(*args, n_steps=200)
        rules = [{"rule": "midpoint", "nodes": 200, "errors": errors(midpoint, reference)}]
        for order in (4, 8, 12, 16, 24, 32, 64, 128, 256):
            samples = integrate_trajectories(*args, quadrature="auto", gaussian_order=order)
            rules.append({"rule": "gauss_hermite", "nodes": order,
                          "errors": errors(samples, reference)})
        records.append({"scenario": scenario.name, "particles": interaction.bunch.n_particles,
                        "reference": "4096-step active-region midpoint", "rules": rules})
    return records


def bound_scan():
    rng = np.random.default_rng(23025)
    baseline = SCENARIOS[0]
    records = []
    for angle in (0.0, 0.05, 0.2, 0.5):
        for duration_fs in (10, 30, 100):
            interaction = build(baseline, SamplingSpec(n_particles=96, seed=20260721, prefilter=0))
            laser = replace(interaction.laser, theta_xz=Quantity(angle, "rad"),
                            theta_yz=Quantity(-angle / 2, "rad"),
                            duration=Quantity(duration_fs, "fs"))
            bunch = replace(
                interaction.bunch,
                x=rng.normal(0, laser.m("sigma_x") * 2, 96),
                y=rng.normal(0, laser.m("sigma_x") * 2, 96),
                z=rng.normal(0, 0.003, 96),
                thx=rng.normal(0, 0.02, 96),
                thy=rng.normal(0, 0.02, 96),
            )
            f, d2, eta_min, curvature = trajectory_bound_geometry(bunch, laser)
            upper = luminosity_upper_bound(bunch, laser)
            common = (interaction.N_e * bunch.weight * photon_density_scale(laser)
                      * C_CGS * SIGMA_T_CGS * laser.intensity_peak()
                      / laser.temporal_envelope.peak_value(np))
            direct = integrate_trajectories(bunch, laser, interaction.N_e, n_steps=4096)
            physical_upper = common * upper
            ratio = np.divide(direct.luminosity, physical_upper,
                              out=np.zeros_like(physical_upper), where=physical_upper > 0)
            records.append({"angle_xz_rad": angle, "angle_yz_rad": -angle / 2,
                            "duration_fs": duration_fs, "particles": len(f),
                            "minimum_f": float(np.min(f)), "minimum_d2": float(np.min(d2)),
                            "maximum_abs_eta_min_s": float(np.max(np.abs(eta_min))),
                            "minimum_curvature_over_b0": float(np.min(curvature)
                                                               / (C_CGS / (2 * laser.rayleigh_x()))),
                            "maximum_direct_over_upper": float(np.max(ratio))})
    return records


def discard_scan():
    interaction = build(SCENARIOS[0], SamplingSpec(n_particles=64, seed=20260721, prefilter=0))
    bunch = replace(interaction.bunch, x=np.r_[interaction.bunch.x[:32], np.full(32, 0.1)])
    args = (bunch, interaction.laser, interaction.N_e)
    full = integrate_trajectories(*args, quadrature="auto", gaussian_order=256)
    direct = integrate_trajectories(*args, n_steps=8192)
    records = []
    for tolerance in (0.001, 0.01, 0.1):
        filtered = integrate_trajectories(*args, quadrature="auto", gaussian_order=256,
                                          discard_tolerance=tolerance)
        discarded = (filtered.luminosity == 0) & (full.luminosity > 0)
        lost = float(np.sum(direct.luminosity[discarded]) / direct.total_yield())
        records.append({"requested": tolerance, "certificate": filtered.discard_certificate,
                        "measured_loss": lost,
                        "direct_reference": "8192-step active-region midpoint",
                        "discarded_particles": int(np.count_nonzero(discarded))})
    return records


def pulse_train_scan():
    interaction = build(SCENARIOS[0], SamplingSpec(n_particles=64, seed=20260721, prefilter=0))
    records = []
    for count in (2, 3):
        for separation_fs in (100, 500):
            for timing_fraction in (-0.5, 0.0, 0.5):
                base = interaction.laser
                laser = PulseTrainParaxialLaser(
                    pulse_energy=base.pulse_energy, wavelength=base.wavelength,
                    sigma_x=base.sigma_x, sigma_y=base.sigma_y,
                    subpulse_duration=Quantity(30, "fs"),
                    repetition_period=Quantity(separation_fs, "fs"), n_subpulses=count,
                    t_off=Quantity(timing_fraction * separation_fs, "fs"),
                )
                args = (interaction.bunch, laser, interaction.N_e)
                reference = integrate_trajectories(*args, n_steps=8192)
                for order in (16, 32, 64, 128):
                    samples = integrate_trajectories(*args, quadrature="auto", gaussian_order=order)
                    records.append({"subpulses": count, "separation_fs": separation_fs,
                                    "timing_over_separation": timing_fraction,
                                    "nodes_per_subpulse": order,
                                    "errors": errors(samples, reference)})
    return records


def der001_scan():
    baseline = SCENARIOS[0]
    beam = replace(baseline.beam, sigma_y=baseline.beam.sigma_x,
                   emit_y=baseline.beam.emit_x)
    laser = replace(baseline.laser, sigma_y=baseline.laser.sigma_x)
    expected = overlap_yield(beam, laser, beam.n_electrons(), n_quad=4001)
    records = []
    for particles in (2048, 8192, 32768, 65536, 131072):
        bunch = sample_gaussian_bunch(beam, particles, 20260721)
        samples = integrate_trajectories(bunch, laser, beam.n_electrons(),
                                         quadrature="auto", gaussian_order=128, chunk=1024)
        records.append({"particles": particles, "sampled_yield": samples.total_yield(),
                        "relative_error": samples.total_yield() / expected - 1})
    return {"der001_yield": expected, "gauss_hermite_order": 128, "samples": records}


def group_delay_scan():
    """Integrate a synthetic full DER025 envelope along ballistic trajectories."""
    rng = np.random.default_rng(25023)
    legendre_x, legendre_w = np.polynomial.legendre.leggauss(512)
    records = []
    spatial_floor = 0.01
    for waist_um in (10, 20):
        for duration_fs in (5, 30, 100):
            for angle in (0.0, 0.2):
                for timing in (-1.0, 0.0, 1.0):
                    baseline = SCENARIOS[0].laser
                    laser = replace(
                        baseline, sigma_x=Quantity(waist_um, "um"),
                        sigma_y=Quantity(waist_um, "um"),
                        duration=Quantity(duration_fs, "fs"),
                        theta_xz=Quantity(angle, "rad"),
                        t_off=Quantity(timing * duration_fs, "fs"),
                    )
                    sigma, sigma_t = laser.m("sigma_x"), laser.m("duration")
                    z_r, omega = laser.rayleigh_x(), laser.omega0()
                    k, f1, f2 = laser.focusing_axes()
                    initial = np.stack((rng.normal(0, sigma, 64),
                                        rng.normal(0, sigma, 64),
                                        rng.normal(0, 0.001, 64)), axis=1)
                    slopes = rng.normal(0, 0.01, (64, 2))
                    direction = np.column_stack((slopes, np.ones(64)))
                    direction /= np.linalg.norm(direction, axis=1)[:, None]
                    f = 1 - direction @ k
                    eta = 10 * sigma_t * legendre_x
                    t = (eta[None, :] + laser.m("t_off")
                         + (initial @ k)[:, None] / C_CGS) / f[:, None]
                    position = initial[:, None, :] + C_CGS * direction[:, None, :] * t[:, :, None]
                    u = position @ k
                    rho2 = (position @ f1)**2 + (position @ f2)**2
                    q = u / z_r
                    p2 = rho2 / (2 * sigma**2)
                    spatial = np.exp(-p2 / (1 + q**2)) / (1 + q**2)
                    baseline_envelope = np.exp(-0.5 * (eta / sigma_t)**2)
                    baseline_integral = np.sum(spatial * baseline_envelope[None, :]
                                               * legendre_w[None, :], axis=1)
                    for gf in (-1.0, 0.0, 1.0, 2.0):
                        delay = q / (omega * (1 + q**2)) * (
                            gf + p2 / 2 * (1 - 2 * gf / (1 + q**2)))
                        full_envelope = np.exp(-0.5 * ((eta[None, :] - delay) / sigma_t)**2)
                        full_integral = np.sum(spatial * full_envelope
                                               * legendre_w[None, :], axis=1)
                        # DER023's spatial-region gate, maximized on a dense q grid.
                        q_bound = np.linspace(0, np.sqrt(1 / spatial_floor - 1), 20001)
                        p_bound = np.maximum(
                            0, -(1 + q_bound**2)
                            * np.log(spatial_floor * (1 + q_bound**2)))
                        delta_star = np.max(
                            q_bound / (omega * (1 + q_bound**2))
                            * (abs(gf) + p_bound / 2
                               * (1 + 2 * abs(gf) / (1 + q_bound**2))))
                        gate = delta_star / (np.sqrt(np.e) * sigma_t)
                        relevant = spatial >= spatial_floor
                        local_change = np.max(np.abs(full_envelope - baseline_envelope[None, :])
                                              [relevant]) if np.any(relevant) else 0.0
                        records.append({
                            "waist_um": waist_um, "duration_fs": duration_fs,
                            "angle_xz_rad": angle, "timing_over_duration": timing,
                            "gf": gf, "group_delay_gate": float(gate),
                            "maximum_relevant_envelope_change": float(local_change),
                            "relative_total_luminosity_change": float(
                                np.sum(full_integral) / np.sum(baseline_integral) - 1),
                        })
    return records


def performance_scan(gpu):
    interaction = build(SCENARIOS[0], SamplingSpec(n_particles=8192, seed=20260721, prefilter=0))
    args = (interaction.bunch, interaction.laser, interaction.N_e)
    records = []
    for name, kwargs, nodes in (
        ("midpoint", {"n_steps": 200}, 200),
        ("gauss_hermite_24", {"quadrature": "auto", "gaussian_order": 24}, 24),
        ("gauss_hermite_128", {"quadrature": "auto", "gaussian_order": 128}, 128),
        ("gauss_hermite_256", {"quadrature": "auto", "gaussian_order": 256}, 256),
    ):
        cpu = integrate_trajectories(*args, backend="numpy", **kwargs)
        record = {"rule": name, "particles": 8192, "nodes_per_particle": nodes,
                  "field_samples": 8192 * nodes,
                  "retained_fraction": 1.0,
                  "estimated_live_bytes_per_particle": BYTES_PER_PARTICLE_STEP * nodes,
                  "estimated_numpy_chunk": chunking.estimate_chunk(
                      8192, BYTES_PER_PARTICLE_STEP * nodes, "numpy"),
                  "numpy_seconds_median_3": timing(
                      lambda: integrate_trajectories(*args, backend="numpy", **kwargs))}
        if gpu:
            record["estimated_cupy_chunk"] = chunking.estimate_chunk(
                8192, BYTES_PER_PARTICLE_STEP * nodes, "cupy")
            device = integrate_trajectories(*args, backend="cupy", **kwargs)
            record["cupy_errors"] = errors(device, cpu)
            record["cupy_seconds_median_3"] = timing(
                lambda: integrate_trajectories(*args, backend="cupy", **kwargs))
        records.append(record)
    far = replace(interaction.bunch,
                  x=np.r_[interaction.bunch.x[:4096], np.full(4096, 0.1)])
    filter_args = (far, interaction.laser, interaction.N_e)
    filter_kwargs = {"quadrature": "auto", "gaussian_order": 256,
                     "discard_tolerance": 0.005}
    full = integrate_trajectories(*filter_args, quadrature="auto", gaussian_order=256)
    filtered = integrate_trajectories(*filter_args, **filter_kwargs)
    retained = int(np.count_nonzero(filtered.luminosity))
    filtered_record = {
        "rule": "gauss_hermite_256_filtered", "particles": 8192,
        "nodes_per_particle": 256, "field_samples": retained * 256,
        "retained_fraction": retained / 8192,
        "discard_certificate": filtered.discard_certificate,
        "measured_loss": (full.total_yield() - filtered.total_yield()) / full.total_yield(),
        "estimated_live_bytes_per_particle": BYTES_PER_PARTICLE_STEP * 256,
        "estimated_numpy_chunk": chunking.estimate_chunk(
            retained, BYTES_PER_PARTICLE_STEP * 256, "numpy"),
        "numpy_seconds_median_3": timing(
            lambda: integrate_trajectories(*filter_args, backend="numpy", **filter_kwargs)),
    }
    if gpu:
        device = integrate_trajectories(*filter_args, backend="cupy", **filter_kwargs)
        filtered_record["cupy_errors"] = errors(device, filtered)
        filtered_record["estimated_cupy_chunk"] = chunking.estimate_chunk(
            retained, BYTES_PER_PARTICLE_STEP * 256, "cupy")
        filtered_record["cupy_seconds_median_3"] = timing(
            lambda: integrate_trajectories(*filter_args, backend="cupy", **filter_kwargs))
    records.append(filtered_record)
    return records


def run(gpu):
    if gpu:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("--gpu requires a real CUDA device")
        device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        device = None
    report = {"settings": {"seed": 20260721, "cpu": "NumPy", "gpu": device,
                           "timing": "median of three warm wall-clock runs, including host transfers"}}
    for name, fn in (
        ("scenarios", scenario_scan), ("bounds", bound_scan),
        ("discard", discard_scan), ("pulse_trains", pulse_train_scan),
        ("der001", der001_scan), ("group_delay", group_delay_scan),
    ):
        report[name] = fn()
        print(name, "done", flush=True)
    report["performance"] = performance_scan(gpu)
    report["status"] = "passed" if (
        all(scenario["rules"][0]["errors"]["luminosity"] < 1e-6
            for scenario in report["scenarios"])
        and all(scenario["rules"][-1]["errors"]["luminosity"] < 2e-4
                and scenario["rules"][-1]["errors"]["var_a_shape"] < 0.02
                for scenario in report["scenarios"])
        and all(x["maximum_direct_over_upper"] <= 1 + 1e-9 for x in report["bounds"])
        and all(x["minimum_curvature_over_b0"] >= 1 - 1e-12 for x in report["bounds"])
        and all(x["errors"]["luminosity"] < 1e-6
                and x["errors"]["var_a_shape"] < 1e-3
                for x in report["pulse_trains"])
        and all(x["measured_loss"] <= x["certificate"] <= x["requested"]
                for x in report["discard"])
        and all(x["maximum_relevant_envelope_change"] <= x["group_delay_gate"] * (1 + 1e-10)
                for x in report["group_delay"])
        and abs(report["der001"]["samples"][-1]["relative_error"]) < 0.002
        and all(max(x.get("cupy_errors", {"none": 0}).values()) < 1e-10
                for x in report["performance"])
    ) else "failed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.gpu)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "passed" else 1)
