#!/usr/bin/env python3
"""Run the release-gate CuPy smoke checks and emit a compact JSON report.

This is deliberately a standalone script: it uses no pytest dependency and refuses to
turn a missing or unusable CUDA device into an all-skipped success.  The gate exercises
the public ``XigmaEngine`` collimated-spectrum path on a deposited 4,000-particle crossed
Gaussian table at ``(theta_xz, theta_yz) = (0.3, 0.2)`` and ellipticity ``0.4``.  It also
includes the companion input-refinement convergence report when that validation module
is installed.  The numerical checks validate the sampler implementation only; they are
not independent arbitrary-angle emission-physics validation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import hashlib
import importlib.metadata
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    """Convert NumPy/scalar values without making NumPy a script dependency."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class ReleaseGateError(RuntimeError):
    """A release check was not executable or did not pass."""


def _source_fingerprints() -> dict[str, str]:
    modules = {"sampler": "gammaforge.engines.xigma.spectrum_sampler", "stages": "gammaforge.engines.xigma.stages",
               "collision": "gammaforge.engines.xigma.collision", "schema": "gammaforge.engines.xigma.schema",
               "laser": "gammaforge.io.laser", "scenarios": "gammaforge.validation.scenarios",
               "convergence": "gammaforge.validation.cupy_convergence"}
    paths = {name: Path(importlib.import_module(module).__file__) for name, module in modules.items()}
    paths["release_script"] = Path(__file__).resolve()
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _cuda_info() -> dict[str, Any]:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on installation
        raise ReleaseGateError(f"CuPy import failed: {exc}") from exc
    try:
        available = bool(cp.cuda.is_available())
        count = int(cp.cuda.runtime.getDeviceCount()) if available else 0
        if not available or count < 1:
            raise ReleaseGateError("CuPy is installed but no usable CUDA device is available")
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return {
            "cupy": getattr(cp, "__version__", "unknown"),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
            "device_count": count,
            "device_id": int(device.id),
            "device_name": name,
            "compute_capability": f"{device.compute_capability[0]}.{device.compute_capability[1]}",
        }
    except ReleaseGateError:
        raise
    except Exception as exc:  # pragma: no cover - depends on driver/device
        raise ReleaseGateError(f"CUDA device probe failed: {exc}") from exc


def _convergence_report() -> dict[str, Any]:
    """Call the companion validation API without coupling this script to its internals."""
    try:
        from gammaforge.validation.cupy_convergence import run_convergence_checks
    except Exception as exc:
        return {"status": "unavailable", "error": f"companion API import failed: {exc}"}
    try:
        result = _json_value(run_convergence_checks())
    except Exception as exc:
        return {"status": "failed", "entrypoint": "run_convergence_checks", "error": str(exc)}
    return {"status": "passed" if isinstance(result, dict) and result.get("pass") is True else "failed",
            "entrypoint": "run_convergence_checks", "result": result}


def run_release(*, rings: tuple[int, ...] = (32, 64), subsampling: int = 32) -> dict[str, Any]:
    """Run the public crossed-Gaussian engine checks and return a JSON-safe report."""
    fingerprints_before = _source_fingerprints()
    cuda = _cuda_info()
    import numpy as np
    from gammaforge.engines.xigma.engine import XigmaEngine
    from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
    from gammaforge.io.interaction import SamplingSpec
    from gammaforge.io.target import OutputKind, OutputRequest
    from gammaforge.io.units import Quantity
    from gammaforge.validation import scenarios

    if not is_gpu_available():
        raise ReleaseGateError("xigma reports CUDA unavailable after the device probe")
    if not rings:
        raise ReleaseGateError("at least one sampler ring setting is required")

    laser = replace(
        scenarios.BASELINE.laser,
        theta_xz=Quantity(0.3, "rad"),
        theta_yz=Quantity(0.2, "rad"),
        ellipticity=0.4,
        psi_pol=Quantity(0.37, "rad"),
    )
    target = replace(
        scenarios.BASELINE.target,
        theta_x_col=Quantity(0.0003, "rad"),
        theta_y_col=Quantity(0.0003, "rad"),
        outputs=(OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(9, 9, 16)),),
    )
    scenario = replace(scenarios.BASELINE, laser=laser, target=target)
    interaction = scenarios.build(
        scenario, SamplingSpec(n_particles=4_000, seed=20260721, prefilter=1e-3)
    )

    runs = []
    for index, ring_count in enumerate(rings):
        n_steps = 32 if index == 0 else 64
        params = XigmaEngine.schema.with_values(
            n_steps=n_steps,
            n_bins_gamma=12,
            n_bins_theta_x=12,
            n_bins_theta_y=12,
            n_bins_a0_shape=12,
            n_bins_ahat=12,
            backend="cupy",
            sampler_rings=ring_count,
            sampler_subsampling=subsampling,
        )
        results = XigmaEngine().run(interaction, params)
        slice_ = results.photon_slices[OutputKind.COLLIMATED_SPECTRUM]
        distr = np.asarray(slice_.distr)
        metadata = results.model_specific
        checks = {
            "finite": bool(np.all(np.isfinite(distr))),
            "nonnegative": bool(np.all(distr >= 0.0)),
            "nonzero": bool(np.any(distr > 0.0)),
            "stage2_backend_cupy": metadata.get("stage2_backend") == "cupy",
        }
        if not all(checks.values()):
            raise ReleaseGateError(f"sampler run failed checks: {checks}")
        sampler_meta = metadata.get("stage2_sampler", {})
        if sampler_meta.get("rings") != ring_count or sampler_meta.get("subsampling") != subsampling:
            raise ReleaseGateError("sampler metadata does not match requested controls")
        runs.append({
            "rings": ring_count,
            "subsampling": subsampling,
            "n_steps": n_steps,
            "shape": list(distr.shape),
            "checks": checks,
            "stage2_backend": metadata.get("stage2_backend"),
            "stage2_sampler": metadata.get("stage2_sampler"),
            "warnings": metadata.get("warnings", ()),
        })

    convergence = _convergence_report()
    fingerprints_after = _source_fingerprints()
    unchanged = fingerprints_before == fingerprints_after
    status = "passed" if convergence["status"] == "passed" and unchanged else "failed"
    return {
        "status": status,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cuda": cuda,
        "package_version": importlib.metadata.version("gammaforge"),
        "source_fingerprints": {"before": fingerprints_before, "after": fingerprints_after},
        "source_unchanged": unchanged,
        "settings": {
            "particles": 4000,
            "n_steps": [32, 64],
            "table_bins": [12, 12, 12, 12],
            "crossing": [0.3, 0.2],
            "ellipticity": 0.4,
            "psi_pol": 0.37,
            "rings": list(rings),
            "subsampling": subsampling,
        },
        "runs": runs,
        "convergence": convergence,
        "error": None if unchanged else "source files changed during release run",
        "caveat": "Finite/nonnegative GPU checks and CPU convergence do not constitute independent arbitrary-angle emission-physics validation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--rings", default="32,64", help="comma-separated sampler ring settings")
    parser.add_argument("--subsampling", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        rings = tuple(int(item) for item in args.rings.split(",") if item)
        report = run_release(rings=rings, subsampling=args.subsampling)
        exit_code = 0 if report.get("status") == "passed" else 1
    except Exception as exc:
        report = {
            "status": "failed",
            "error": str(exc),
            "python": sys.version,
            "platform": platform.platform(),
        }
        exit_code = 1
    encoded = json.dumps(_json_value(report), indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
