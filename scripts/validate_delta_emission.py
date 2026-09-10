#!/usr/bin/env python
"""Bounded CPU/GPU matched-bin delta pilot (diagnostic only)."""
import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from gammaforge.engines.xigma.stages import integrate_trajectories, deposit_shape_table, retarget_ahat, angular_spectrum_from_table, DEFAULT_AHAT_MIN, DEFAULT_AHAT_MAX, DEFAULT_AHAT_DECADES
import gammaforge.io.laser as laser_module
import gammaforge.io.bunch as bunch_module
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios
from gammaforge.validation.references.delta_emission import emission_lines
from gammaforge.validation.delta_comparison import compare_emission_bins
VARIANTS = (('headon', 0.0, 0.0, 0.0, 0.0), ('crossed_small', 0.02, -0.015, 0.4, 0.37))
DEFAULT_TABLE_CONFIGS = (((16, 8, 8, 16), 1), ((32, 16, 16, 32), 1), ((16, 8, 8, 16), 2))


def parse_table_config(value):
    """Parse G,X,Y,A,RETARGET into the pilot's explicit grid configuration."""
    try:
        fields = tuple(int(part.strip()) for part in value.split(','))
    except (AttributeError, ValueError):
        raise ValueError('table must be five comma-separated positive integers: G,X,Y,A,RETARGET')
    if len(fields) != 5 or any(item < 1 for item in fields):
        raise ValueError('table must be five comma-separated positive integers: G,X,Y,A,RETARGET')
    return (fields[:4], fields[4])


def _hash_sources():
    from gammaforge.engines.xigma import stages
    paths = (Path(__file__), Path(emission_lines.__code__.co_filename), Path(compare_emission_bins.__code__.co_filename), Path(stages.__file__), Path(scenarios.__file__), Path(laser_module.__file__), Path(bunch_module.__file__))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_pilot(particles=16000, n_steps=64, seed=20260721, retarget_bins=64, quadrature_order=8,
              *, table_configs=None, scenario_names=None, backend='numpy', rings=32, subsampling=32):
    before = _hash_sources()
    records = []
    selected_scenarios = scenarios.SCENARIOS if scenario_names is None else tuple(
        scenario for scenario in scenarios.SCENARIOS if scenario.name in scenario_names
    )
    if table_configs is None:
        configs = tuple((shape, multiplier * retarget_bins)
                        for shape, multiplier in DEFAULT_TABLE_CONFIGS)
    else:
        configs = tuple(table_configs)
        if not configs or any(len(shape) != 4 or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in (*shape, rt))
                              for shape, rt in configs):
            raise ValueError('table_configs must contain positive integer (G,X,Y,A,RETARGET) entries')
    if scenario_names is not None:
        known = {scenario.name for scenario in scenarios.SCENARIOS}
        if not scenario_names or any(name not in known for name in scenario_names):
            raise ValueError('scenario_names must contain known scenarios')
    for scenario in selected_scenarios:
        for name, txz, tyz, eps, psi in VARIANTS:
            laser = replace(scenario.laser, theta_xz=Quantity(txz, 'rad'), theta_yz=Quantity(tyz, 'rad'), ellipticity=eps, psi_pol=Quantity(psi, 'rad'))
            interaction = scenarios.build(replace(scenario, laser=laser), SamplingSpec(n_particles=particles, seed=seed, prefilter=0.001))
            samples = integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e, n_steps=n_steps)
            photon = float(interaction.laser.photon_energy())
            C = (1 + np.cos(txz) * np.cos(tyz)) / 2
            tables = []
            for shape, rt in configs:
                tables.append((shape, rt, retarget_ahat(deposit_shape_table(samples, n_bins=shape, scheme='cic'), float(samples.intensity_peak), n_bins=rt)))
            for oi, (tx, ty) in enumerate(((0.0, 0.0), (0.5 / float(interaction.beam.gamma0()), -0.25 / float(interaction.beam.gamma0())))):
                energies, weights = emission_lines(samples, tx, ty, photon_energy=photon, psi_pol=psi, ellipticity=eps, theta_xz=txz, theta_yz=tyz)
                edges = np.asarray(np.geomspace(float(0.9 * energies.min()), float(1.1 * energies.max()), 25), dtype=float)
                for shape, rt, table in tables:
                    scale = 4 * photon * C

                    def density(x, table=table, scale=scale):
                        s = np.asarray(x) / scale
                        return angular_spectrum_from_table(
                            table, np.array([tx]), np.array([ty]), s,
                            psi_pol=psi, ellipticity=eps, theta_xz=txz, theta_yz=tyz,
                            backend=backend, rings=rings, subsampling=subsampling
                        )[0, 0, :] / scale
                    cmp = compare_emission_bins(density, edges, energies, weights, quadrature_order=quadrature_order)
                    records.append({
                        'scenario': scenario.name, 'scenario_repr': repr(scenario), 'variant': name,
                        'geometry': {'theta_xz': txz, 'theta_yz': tyz, 'ellipticity': eps, 'psi_pol': psi},
                        'observer': {'index': oi, 'theta_x': tx, 'theta_y': ty},
                        'photon_energy_erg': photon, 'shape': list(shape),
                        'actual_table_shape': list(table.H.shape), 'intensity_peak': float(samples.intensity_peak),
                        'ahat_defaults': {'min': DEFAULT_AHAT_MIN, 'max': DEFAULT_AHAT_MAX, 'decades': DEFAULT_AHAT_DECADES},
                        'retarget_bins': rt, 'edges': edges.tolist(), 'comparison': cmp,
                    })
    after = _hash_sources()
    return {
        'status': 'completed' if before == after else 'failed',
        'pilot_only': True, 'scientific_pass': False,
        'source_sha256_before': before, 'source_sha256_after': after,
        'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
        'settings': {
            'particles': particles, 'n_steps': n_steps, 'seed': seed,
            'retarget_bins': retarget_bins, 'quadrature_order': quadrature_order,
            'scheme': 'cic', 'prefilter': 1e-3, 'backend': backend,
            'rings': rings, 'subsampling': subsampling,
            'table_configs': [{'shape': list(shape), 'retarget_bins': rt} for shape, rt in configs],
            'scenarios': [scenario.name for scenario in selected_scenarios],
        },
        'records': records, 'unresolved_numerics': True,
        'caveat': (
            'Fixed-direction finite-window pilot, not scientific acceptance or full yield. '
            'Stage 0, sampling, ahat and reduced-model assumptions are shared. '
            'Particle, Stage-0, angular-aperture and CUDA convergence remain untested.'
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--particles', type=int, default=16000)
    p.add_argument('--n-steps', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260721)
    p.add_argument('--retarget-bins', type=int, default=64)
    p.add_argument('--quadrature-order', type=int, default=8)
    p.add_argument('--backend', choices=['numpy', 'cupy'], default='numpy',
                   help='compute backend for Stage 2 (default: numpy)')
    p.add_argument('--rings', type=int, default=32, help='CuPy sampler rings')
    p.add_argument('--subsampling', type=int, default=32, help='CuPy sampler subsampling')
    p.add_argument('--table', action='append', metavar='G,X,Y,A,RETARGET',
                   help='explicit table configuration; repeatable')
    p.add_argument('--scenario', action='append', choices=[s.name for s in scenarios.SCENARIOS],
                   help='scenario to run; repeatable (default: all)')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.particles < 1 or a.n_steps < 1 or a.seed < 0 or (a.retarget_bins < 1) or (a.quadrature_order < 2):
        p.error('counts positive, seed nonnegative, quadrature order >= 2')
    try:
        table_configs = tuple(parse_table_config(value) for value in a.table) if a.table else None
        report = run_pilot(a.particles, a.n_steps, a.seed, a.retarget_bins, a.quadrature_order,
                           table_configs=table_configs, scenario_names=a.scenario,
                           backend=a.backend, rings=a.rings, subsampling=a.subsampling)
    except Exception as exc:
        report = {'status': 'failed', 'pilot_only': True, 'scientific_pass': False, 'error': f'{type(exc).__name__}: {exc}'}

    def encode(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return float(o)
        raise TypeError(type(o).__name__)
    payload = json.dumps(report, indent=2, allow_nan=False, default=encode)
    if a.output:
        a.output.write_text(payload + '\n')
    print(payload)
    return 0 if report['status'] == 'completed' else 1
if __name__ == '__main__':
    raise SystemExit(main())