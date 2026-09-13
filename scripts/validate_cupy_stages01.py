"""Compare actual NumPy/CuPy particle stages over the shared scenario bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from gammaforge.engines.xigma import stages, spectrum_sampler
from gammaforge.io.interaction import SamplingSpec
from gammaforge.validation.scenarios import SCENARIOS, build


def timed(call):
    call()  # Warm kernels and exclude compilation from timing.
    start = perf_counter()
    result = call()
    return result, perf_counter() - start


def relative_l1(a, b):
    return float(np.sum(np.abs(a-b)) / max(float(np.sum(np.abs(b))), 1e-300))


def run(particles, steps):
    if not spectrum_sampler.is_gpu_available():
        raise RuntimeError('Actual CUDA is required for the Stage-0/1 gate')
    import gammaforge.io.laser as laser_module
    import gammaforge.engines.xigma.collision as collision_module
    paths = [Path(module.__file__) for module in (stages, laser_module, collision_module)]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    records = []
    for scenario in SCENARIOS:
        inputs = build(scenario, SamplingSpec(n_particles=particles, seed=20260721, prefilter=1e-3))
        samples = {}
        times = {}
        for backend in ('numpy', 'cupy'):
            samples[backend], times[backend] = timed(lambda: stages.integrate_trajectories(
                inputs.bunch, inputs.laser, inputs.N_e, n_steps=steps, backend=backend))
        checks = {
            'luminosity_l1': relative_l1(samples['cupy'].luminosity, samples['numpy'].luminosity),
            'a0_shape_l1': relative_l1(samples['cupy'].a0_shape, samples['numpy'].a0_shape),
        }
        deposits = []
        for scheme in ('nearest', 'cic'):
            tables = {}
            elapsed = {}
            for backend in ('numpy', 'cupy'):
                tables[backend], elapsed[backend] = timed(lambda: stages.deposit_shape_table(
                    samples['numpy'], n_bins=(16,16,16,24), scheme=scheme, backend=backend))
            error = relative_l1(tables['cupy'].H, tables['numpy'].H)
            deposits.append({'scheme': scheme, 'density_l1': error, 'seconds': elapsed,
                             'mass_relative_error': abs(tables['cupy'].total_weight/samples['numpy'].total_yield()-1),
                             'host_output': isinstance(tables['cupy'].H, np.ndarray)})
        passed = (all(np.isfinite(v) and v < 1e-10 for v in checks.values())
                  and all(d['density_l1'] < 1e-10 and d['mass_relative_error'] < 1e-10
                          and d['host_output'] for d in deposits))
        records.append({'scenario': scenario.name, 'stage0_errors': checks, 'stage0_seconds': times,
                        'stage1': deposits, 'pass': passed})
        print(scenario.name, checks, 'pass:', passed, flush=True)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return {'status': 'passed' if before == after and all(r['pass'] for r in records) else 'failed',
            'particles': particles, 'n_steps': steps, 'cupy': spectrum_sampler.cp.__version__,
            'source_sha256_before': before, 'source_sha256_after': after, 'records': records,
            'timing_scope': 'One warm wall-time measurement including host geometry and transfers; not a universal speedup claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=16384)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.particles, args.steps)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if report['status']=='passed' else 1)
