"""Reproducible CuPy timing comparison for xigma polarization geometries.

The baseline is loaded in-memory from revision 6420781; no alternate checkout or
installed historical package is used.  CUDA compilation is excluded from timings.
This script requires a CUDA device to run (``--help`` is safe on CPU-only hosts).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import linecache
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "6420781"
WARMUP = 20
REPEATS = 15
SUBSAMPLING = 32


def module_from_source(name: str, source: str):
    """Execute historical sampler source in-process, preserving traceback lines."""
    filename = f"<xigma-polarization-benchmark-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(f"gammaforge.engines.xigma._benchmark_{name}")
    module.__package__ = "gammaforge.engines.xigma"
    sys.modules[module.__name__] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


class CaptureLaunch:
    """Capture resident device arguments while retaining the raw-kernel launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.geometry = None
        self.args = None

    def __getitem__(self, geometry):
        def launch(*args):
            self.geometry, self.args = geometry, args
            self.kernel[geometry](*args)

        return launch


def _call(module, table, tx, ty, energy, **case):
    return module.calculate_angular_spectrum_gpu(
        table, tx, ty, energy, subsampling=SUBSAMPLING, **case
    )


def measure(module, table, tx, ty, energy, case):
    """Return resident CUDA-event and separate end-to-end host medians in ms."""
    cp = module.cp
    original = module._kernel
    capture = CaptureLaunch(original)
    module._kernel = capture
    try:
        # This call compiles the kernel and captures stable resident arguments.
        _call(module, table, tx, ty, energy, **case)
        grid, args = capture.geometry, capture.args

        def resident():
            args[0].fill(0)
            capture.kernel[grid](*args)

        for _ in range(WARMUP):
            resident()
        cp.cuda.Stream.null.synchronize()

        device_ms = []
        for _ in range(REPEATS):
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            resident()
            end.record()
            end.synchronize()
            device_ms.append(float(cp.cuda.get_elapsed_time(begin, end)))

        # Host timings are deliberately separate from event timings. Compilation
        # is already complete, but argument setup and transfer remain end-to-end.
        host_ms = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            _call(module, table, tx, ty, energy, **case)
            host_ms.append((time.perf_counter() - start) * 1000.0)

        return {
            "resident_gpu_ms_median": float(np.median(device_ms)),
            "resident_gpu_ms_range": [float(min(device_ms)), float(max(device_ms))],
            "end_to_end_host_ms_median": float(np.median(host_ms)),
        }
    finally:
        module._kernel = original


def main():
    global REPEATS, WARMUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup must be non-negative")
    REPEATS, WARMUP = args.repeats, args.warmup

    from gammaforge.engines.xigma import spectrum_sampler as new_sampler
    from gammaforge.engines.xigma.stages import (
        deposit_shape_table,
        integrate_trajectories,
        retarget_ahat,
    )
    from gammaforge.validation import scenarios

    if not new_sampler.is_gpu_available():
        parser.error("CuPy and a CUDA device are required")

    old_source = subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:src/gammaforge/engines/xigma/spectrum_sampler.py"],
        cwd=ROOT,
        text=True,
    )
    old_sampler = module_from_source("6420781", old_source)

    interaction = scenarios.build(
        scenarios.BASELINE,
        sampling=replace(scenarios.BASELINE.sampling, n_particles=40_000),
    )
    samples = integrate_trajectories(
        interaction.bunch, interaction.laser, interaction.N_e, n_steps=64
    )
    shape = deposit_shape_table(samples, n_bins=(32, 32, 32, 64), scheme="cic")
    tables = {
        "defaultahat1": retarget_ahat(shape, samples.intensity_peak),
        "resolved32": retarget_ahat(
            shape,
            samples.intensity_peak,
            ahat_max=samples.intensity_peak * 1.05,
            n_bins=32,
            decades=0.3,
        ),
    }
    tx = ty = np.linspace(-3e-4, 3e-4, 9)
    energy = np.linspace(0.65, 0.99, 16) * float(np.mean(samples.gamma) ** 2)
    cases = {
        "head_on_linear": {"psi_pol": 0.0, "ellipticity": 0.0, "theta_xz": 0.0, "theta_yz": 0.0},
        "head_on_elliptical": {"psi_pol": 0.0, "ellipticity": 0.5, "theta_xz": 0.0, "theta_yz": 0.0},
        "head_on_circular": {"psi_pol": 0.0, "ellipticity": 1.0, "theta_xz": 0.0, "theta_yz": 0.0},
        "crossing_bothplanes": {"psi_pol": 0.0, "ellipticity": 0.0, "theta_xz": 0.3, "theta_yz": 0.2},
    }
    props = new_sampler.cp.cuda.runtime.getDeviceProperties(new_sampler.cp.cuda.Device().id)
    print(json.dumps({
        "device": props["name"].decode(),
        "cupy": new_sampler.cp.__version__,
        "baseline_revision": BASELINE_REVISION,
        "subsampling": SUBSAMPLING,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "output_shape": [9, 9, 16],
        "tables": {name: list(table.H.shape) for name, table in tables.items()},
    }), flush=True)
    for table_name, table in tables.items():
        for case_name, case in cases.items():
            modules = [("old_6420781", old_sampler), ("new", new_sampler)] if case_name == "head_on_linear" else [("new", new_sampler)]
            for variant, module in modules:
                invocation = {"psi_pol": 0.0} if variant == "old_6420781" else case
                result = measure(module, table, tx, ty, energy, invocation)
                print(json.dumps({"table": table_name, "case": case_name, "variant": variant, **result}), flush=True)


if __name__ == "__main__":
    main()
