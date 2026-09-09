"""Small, reproducible convergence checks for the experimental CuPy sampler."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Callable, Mapping

import numpy as np

from ..engines.xigma.stages import Table, angular_spectrum_from_table, deposit_shape_table, integrate_trajectories, retarget_ahat
from ..io.interaction import SamplingSpec
from ..io.units import Quantity
from ..validation import scenarios

CPU_MASS_TOL = 0.03
CPU_L1_TOL = 0.05
GPU_MASS_TOL = 0.03
GPU_L1_TOL = 0.05
CENTROID_TOL = 0.01


def _integral(cube: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(np.trapezoid(cube, s, axis=2), y, axis=1), x))


def _centroid(cube: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    mass = _integral(cube, x, y, s)
    return _integral(cube * s[None, None, :], x, y, s) / mass if mass > 0.0 else math.nan


def _refine_table(table: Table, factor: int) -> Table:
    density = table.H
    new_edges = []
    for axis, edges in ((1, table.theta_x_edges), (2, table.theta_y_edges)):
        n = len(edges) - 1
        refined = np.linspace(edges[0], edges[-1], n * factor + 1)
        centers = 0.5 * (refined[:-1] + refined[1:])
        coordinate = np.clip((centers - edges[0]) / (edges[1] - edges[0]) - 0.5, 0, n - 1)
        lo = np.clip(np.floor(coordinate).astype(int), 0, n - 2)
        shape = [1] * density.ndim
        shape[axis] = centers.size
        weight = (coordinate - lo).reshape(shape)
        density = np.take(density, lo, axis=axis) * (1 - weight) + np.take(density, lo + 1, axis=axis) * weight
        new_edges.append(refined)
    return replace(table, H=density, theta_x_edges=new_edges[0], theta_y_edges=new_edges[1])


def _grid(table: Table, samples=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(float(table.theta_x_edges[1]), float(table.theta_x_edges[-2]), 5)
    y = np.linspace(float(table.theta_y_edges[1]), float(table.theta_y_edges[-2]), 5)
    gamma_max = float(np.max(samples.gamma)) if samples is not None else float(table.gamma_edges[-1])
    return x, y, np.linspace(0.25 * gamma_max**2, 0.85 * gamma_max**2, 10)


def _compare(actual, reference, x, y, s, mass_tol, l1_tol):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    expected_shape = (x.size, y.size, s.size)
    if actual.shape != expected_shape or reference.shape != expected_shape:
        return {"status": "fail", "pass": False, "error": f"shape {actual.shape} != {expected_shape}"}
    if not np.all(np.isfinite(actual)) or np.any(actual < 0.0) or not np.any(actual > 0.0):
        return {"status": "fail", "pass": False, "error": "actual cube is non-finite, negative, or zero"}
    ref_mass = _integral(reference, x, y, s)
    mass = _integral(actual, x, y, s)
    if not np.all(np.isfinite(reference)) or np.any(reference < 0.0) or not np.isfinite(ref_mass) or ref_mass <= 0.0:
            return {"status": "inconclusive", "pass": False, "error": "non-finite, negative, or zero reference mass"}
    l1 = _integral(np.abs(actual - reference), x, y, s) / ref_mass
    mass_error = abs(mass / ref_mass - 1.0)
    actual_centroid, reference_centroid = _centroid(actual, x, y, s), _centroid(reference, x, y, s)
    if not np.isfinite(actual_centroid) or not np.isfinite(reference_centroid) or reference_centroid == 0.0:
        return {"status": "fail", "pass": False, "error": "non-finite spectral centroid"}
    centroid = abs(actual_centroid / reference_centroid - 1.0)
    passed = mass_error <= mass_tol and l1 <= l1_tol and centroid <= CENTROID_TOL
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "mass_relative_error": float(mass_error),
        "l1_relative": float(l1),
        "spectral_centroid_relative_error": float(centroid),
        "limits": {"mass": mass_tol, "l1": l1_tol, "centroid": CENTROID_TOL},
    }


def _default_cases() -> list[dict]:
    cases = []
    for scenario in scenarios.SCENARIOS:
        interaction = scenarios.build(scenario, SamplingSpec(n_particles=4_000, seed=20260721, prefilter=1e-3))
        samples = integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e, n_steps=32)
        shape = deposit_shape_table(samples, n_bins=(12, 12, 12, 12), scheme="cic")
        cases.append({"name": scenario.name, "table": retarget_ahat(shape, samples.intensity_peak), "samples": samples})

    crossed = scenarios.build(
        replace(scenarios.BASELINE, laser=replace(
            scenarios.BASELINE.laser, theta_xz=Quantity(0.02, "rad"), theta_yz=Quantity(-0.015, "rad"),
            ellipticity=0.4, psi_pol=Quantity(0.37, "rad"),
        )), SamplingSpec(n_particles=4_000, seed=20260721, prefilter=1e-3),
    )
    crossed_samples = integrate_trajectories(crossed.bunch, crossed.laser, crossed.N_e, n_steps=32)
    crossed_shape = deposit_shape_table(crossed_samples, n_bins=(12, 12, 12, 12), scheme="cic")
    cases.append({"name": "crossed", "table": retarget_ahat(crossed_shape, crossed_samples.intensity_peak), "samples": crossed_samples,
                  "kwargs": {"psi_pol": 0.37, "ellipticity": 0.4, "theta_xz": 0.02, "theta_yz": -0.015}})
    cases.extend(_synthetic_cases())
    return cases


def _synthetic_cases() -> list[dict]:
    cases = []
    for name, gamma_edges, x_edges, y_edges in (
        ("wide_offaxis", np.linspace(1800.0, 2200.0, 9), np.linspace(0.014, 0.018, 13), np.linspace(-0.002, 0.002, 13)),
        ("narrow_offaxis", np.linspace(1800.0, 2200.0, 9), np.linspace(0.014, 0.0144, 13), np.linspace(-0.0002, 0.0002, 13)),
        ("highgamma10000", np.linspace(8000.0, 10000.0, 9), np.linspace(-0.001, 0.001, 13), np.linspace(-0.001, 0.001, 13)),
    ):
        ahat_edges = np.array([0.0, 0.001, 0.003, 0.008, 0.02, 0.05, 0.08])
        g = 0.5 * (gamma_edges[:-1] + gamma_edges[1:]); x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:]); a = 0.5 * (ahat_edges[:-1] + ahat_edges[1:])
        G, X, Y, A = np.meshgrid(g, x, y, a, indexing="ij")
        H = 2.0 + 0.001 * (G - gamma_edges[0]) + 2.0 * X + 1.5 * Y + 3.0 * A
        kwargs = {"psi_pol": 0.37, "ellipticity": 0.4, "theta_xz": 0.02, "theta_yz": -0.015}
        if name == "highgamma10000":
            kwargs = {"psi_pol": 0.41, "ellipticity": 1.0, "theta_xz": 0.0, "theta_yz": 0.0}
        case = {"name": name, "table": Table(gamma_edges, x_edges, y_edges, ahat_edges, H, float(H.sum()), "synthetic-convergence"),
                      "kwargs": kwargs}
        if name == "highgamma10000":
            case["cpu_refinements"] = (32, 64)
            crossed = dict(case)
            crossed["name"] = "highgamma10000_crossed"
            crossed["kwargs"] = {"psi_pol": 0.37, "ellipticity": 0.4, "theta_xz": 0.02, "theta_yz": -0.015}
            cases.append(crossed)
            case["name"] = "highgamma10000_circular"
        cases.append(case)
    return cases


def run_convergence_checks(
    cases: list[Mapping] | None = None,
    *,
    cpu_runner: Callable | None = None,
    gpu_runner: Callable | None = None,
    cpu_refinements: tuple[int, int] = (8, 16),
) -> dict:
    """Run CPU-reference and CuPy ring/sampling convergence checks.

    Runners receive ``(table, x, y, s, **kwargs)``. A supplied GPU runner is useful
    for tests and orchestration; without one, GPU checks are explicitly inconclusive
    when CUDA is unavailable rather than being silently skipped.
    """
    if len(cpu_refinements) != 2 or not all(isinstance(f, int) and f > 0 for f in cpu_refinements) or cpu_refinements[0] >= cpu_refinements[1]:
        raise ValueError("cpu_refinements must be two increasing positive integers")
    gpu_rings = (16, 32, 64)
    gpu_subsamplings = (32, 128, 256)
    if cases is None:
        cases = _default_cases()
    if cpu_runner is None:
        cpu_runner = lambda table, x, y, s, **kw: angular_spectrum_from_table(table, x, y, s, backend="numpy", **kw)
    if gpu_runner is None:
        try:
            from ..engines.xigma.spectrum_sampler import calculate_angular_spectrum_gpu, is_gpu_available
            gpu_runner = calculate_angular_spectrum_gpu if is_gpu_available() else None
        except Exception:
            gpu_runner = None

    checks = []
    for case in cases:
        name, table = case["name"], case["table"]
        effective_cpu = tuple(case.get("cpu_refinements", cpu_refinements))
        kwargs = dict(case.get("kwargs", {})); x, y, s = _grid(table, case.get("samples"))
        try:
            coarse = cpu_runner(_refine_table(table, effective_cpu[0]), x, y, s, **kwargs)
            reference = cpu_runner(_refine_table(table, effective_cpu[1]), x, y, s, **kwargs)
        except Exception as exc:
            checks.append({"case": name, "kind": "cpu_reference", "status": "fail", "pass": False, "error": str(exc)})
            continue
        cpu_check = _compare(coarse, reference, x, y, s, CPU_MASS_TOL, CPU_L1_TOL)
        cpu_check.update({"case": name, "kind": "cpu_reference", "settings": {
            "refinements": list(effective_cpu), "table_shape": list(table.H.shape),
            "query_x": x.tolist(), "query_y": y.tolist(), "query_s": s.tolist(),
            "geometry_kwargs": kwargs,
        }})
        if cpu_check["status"] == "fail":
            cpu_check["status"] = "inconclusive"; cpu_check["pass"] = False
        checks.append(cpu_check)
        if gpu_runner is None or cpu_check["status"] != "pass":
            checks.append({"case": name, "kind": "gpu", "status": "inconclusive", "pass": False,
                           "error": "GPU unavailable or CPU reference did not converge"})
            continue
        gpu_results = {}
        for rings in gpu_rings:
            try:
                out = gpu_runner(table, x, y, s, rings=rings, subsampling=256, **kwargs)
            except Exception as exc:
                checks.append({"case": name, "kind": "gpu_rings", "status": "fail", "pass": False, "error": str(exc), "settings": {"rings": rings, "subsampling": 256}})
                continue
            item = _compare(out, reference, x, y, s, GPU_MASS_TOL, GPU_L1_TOL)
            item.update({"case": name, "kind": "gpu_rings", "settings": {"rings": rings, "subsampling": 256}})
            item["required"] = rings == max(gpu_rings)
            checks.append(item)
            gpu_results[(rings, 256)] = np.asarray(out)
        for subsampling in gpu_subsamplings:
            try:
                out = gpu_runner(table, x, y, s, rings=64, subsampling=subsampling, **kwargs)
            except Exception as exc:
                checks.append({"case": name, "kind": "gpu_subsampling", "status": "fail", "pass": False, "error": str(exc), "settings": {"rings": 64, "subsampling": subsampling}})
                continue
            item = _compare(out, reference, x, y, s, GPU_MASS_TOL, GPU_L1_TOL)
            item.update({"case": name, "kind": "gpu_subsampling", "settings": {"rings": 64, "subsampling": subsampling}})
            item["required"] = subsampling == max(gpu_subsamplings)
            checks.append(item)
            gpu_results[(64, subsampling)] = np.asarray(out)
        if gpu_runner is not None and (32, 32) not in gpu_results:
            try:
                out = gpu_runner(table, x, y, s, rings=32, subsampling=32, **kwargs)
                item = _compare(out, reference, x, y, s, GPU_MASS_TOL, GPU_L1_TOL)
                item.update({"case": name, "kind": "gpu_default", "required": True, "settings": {"rings": 32, "subsampling": 32}})
                checks.append(item); gpu_results[(32, 32)] = np.asarray(out)
            except Exception as exc:
                checks.append({"case": name, "kind": "gpu_default", "required": True, "status": "fail", "pass": False, "error": str(exc)})
        if (32, 256) in gpu_results and (64, 256) in gpu_results:
            item = _compare(gpu_results[(32,256)], gpu_results[(64,256)], x,y,s,GPU_MASS_TOL,GPU_L1_TOL)
            item.update({"case": name, "kind": "gpu_ring_refinement", "required": True,
                         "settings": {"from": {"rings": 32, "subsampling": 256}, "to": {"rings": 64, "subsampling": 256}}})
            checks.append(item)
        if (64, 128) in gpu_results and (64, 256) in gpu_results:
            item = _compare(gpu_results[(64,128)], gpu_results[(64,256)], x,y,s,GPU_MASS_TOL,GPU_L1_TOL)
            item.update({"case": name, "kind": "gpu_subsampling_refinement", "required": True,
                         "settings": {"from": {"rings": 64, "subsampling": 128}, "to": {"rings": 64, "subsampling": 256}}})
            checks.append(item)
    return {"settings": {"cpu_refinements": list(cpu_refinements), "gpu_rings": list(gpu_rings),
                          "gpu_subsamplings": list(gpu_subsamplings)}, "checks": checks,
            "pass": bool(checks) and all(check["pass"] for check in checks if check.get("required", True))}


__all__ = ["run_convergence_checks"]
