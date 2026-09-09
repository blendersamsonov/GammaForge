"""Warm CUDA/wall-time benchmark; the deliberately biased LUT is a timing control only.

Run from a checkout with CUDA: python scripts/benchmark_xigma_sampler.py
Historical alpha source is read from git, never imported from another repository.
No benchmark alternative is available through the production engine API.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import linecache
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import types

import numpy as np
from gammaforge.engines.xigma import spectrum_sampler as sampler
from gammaforge.engines.xigma.stages import deposit_shape_table, integrate_trajectories, retarget_ahat
from gammaforge.validation import scenarios

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "c090909"


def module_from_source(name, source):
    filename = f"<sampler-benchmark-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(f"gammaforge.engines.xigma._benchmark_{name}")
    sys.modules[module.__name__] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def lookup_control(current, old):
    """Change only CDF inversion/storage; retain all other current corrections."""
    # The benchmark source is injected inside the kernel factory; use its local
    # capacity rather than the module's 64-ring benchmark constant.
    constants = "CDF_PHI_RESOLUTION = 32\nCDF_PHI_REPEAT = 1\n"
    current = current.replace("SAMPLES_TOTAL = 256", constants + "SAMPLES_TOTAL = 256", 1)
    start = old.index("            for arc_idx in jit.range(n_arcs):")
    end = old.index("            f_tot = CP_ZERO", start)
    current = current.replace("            f_tot = CP_ZERO", old[start:end] + "            f_tot = CP_ZERO", 1)
    current = current.replace("        TMP_FLOAT_ARRAY = jit.shared_memory", "        inv_cdf = jit.shared_memory(CP_FLOAT, CDF_PHI_RESOLUTION * MAX_ARCS)\n        TMP_FLOAT_ARRAY = jit.shared_memory", 1)
    start = old.index("                        il = CP_UINT(cp.floor(reg")
    end = old.index("                        x = x0 + theta", start)
    new_start = current.index("                        target_cdf = reg * arc_total_weight")
    new_end = current.index("                        x = x0 + theta", new_start)
    return current[:new_start] + old[start:end] + current[new_end:]


class CaptureLaunch:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, geometry):
        def launch(*args):
            self.geometry, self.args = geometry, args
            self.kernel[geometry](*args)
        return launch


def measure(module, table, tx, ty, energy, repeats):
    cp = sampler.cp
    capture = CaptureLaunch(module._kernel)
    module._kernel = capture
    try:
        # First call compiles and captures resident device arguments; excluded from timing.
        module.calculate_angular_spectrum_gpu(table, tx, ty, energy)
        grid, block = capture.geometry
        args = capture.args
        def resident():
            args[0].fill(0)
            capture.kernel[grid, block](*args)
        for _ in range(20):
            resident()
        cp.cuda.Stream.null.synchronize()
        device_ms, wall_ms = [], []
        for _ in range(repeats):
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            resident()
            end.record()
            end.synchronize()
            device_ms.append(float(cp.cuda.get_elapsed_time(begin, end)))
        for _ in range(repeats):
            begin = perf_counter()
            module.calculate_angular_spectrum_gpu(table, tx, ty, energy)
            wall_ms.append((perf_counter() - begin) * 1000)
        return {"resident_gpu_ms_median": float(np.median(device_ms)),
                "end_to_end_ms_median": float(np.median(wall_ms)),
                "resident_gpu_ms_range": [min(device_ms), max(device_ms)]}
    finally:
        module._kernel = capture.kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not sampler.is_gpu_available():
        parser.error("CuPy and a CUDA device are required")
    old = subprocess.check_output(["git", "show", f"{BASELINE_REVISION}:src/gammaforge/engines/xigma/spectrum_sampler.py"], cwd=ROOT, text=True)
    current = Path(sampler.__file__).read_text()
    modules = {
        "original_alpha_biased": module_from_source("original", old),
        "repaired_biased_lut_control": module_from_source("lookup", lookup_control(current, old)),
        "repaired_exact_cdf": sampler,
    }
    interaction = scenarios.build(replace(scenarios.BASELINE,
        sampling=replace(scenarios.BASELINE.sampling, n_particles=40_000)))
    samples = integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e, n_steps=64)
    shape = deposit_shape_table(samples, n_bins=(32,32,32,64), scheme="cic")
    tables = {"alpha_default": retarget_ahat(shape, samples.intensity_peak),
              "resolved_ahat": retarget_ahat(shape, samples.intensity_peak,
                  ahat_max=samples.intensity_peak*1.05, n_bins=32, decades=.3)}
    properties = sampler.cp.cuda.runtime.getDeviceProperties(sampler.cp.cuda.Device().id)
    print(json.dumps({"device": properties["name"].decode(), "cupy": sampler.cp.__version__,
        "repeats": args.repeats, "samples_total": sampler.SAMPLES_TOTAL, "subsampling": 32,
        "note": "LUT controls are biased; timing only. First compilation excluded."}), flush=True)
    tx = ty = np.linspace(-3e-4, 3e-4, 9)
    energy = np.linspace(.65,.99,16) * float(np.mean(samples.gamma)**2)
    for table_name, table in tables.items():
        for name, module in modules.items():
            result = measure(module, table, tx, ty, energy, args.repeats)
            print(json.dumps({"table": table_name, "shape": table.H.shape,
                "output_shape": [len(tx),len(ty),len(energy)], "variant": name, **result}), flush=True)


if __name__ == "__main__":
    main()
