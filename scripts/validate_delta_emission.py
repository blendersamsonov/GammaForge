#!/usr/bin/env python
"""Bounded CPU matched-bin delta pilot (diagnostic only)."""
import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from gammaforge.engines.xigma.stages import integrate_trajectories, deposit_shape_table, retarget_ahat, spectrum_from_table, DEFAULT_AHAT_MIN, DEFAULT_AHAT_MAX, DEFAULT_AHAT_DECADES
import gammaforge.io.laser as laser_module
import gammaforge.io.bunch as bunch_module
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.units import Quantity
from gammaforge.validation import scenarios
from gammaforge.validation.references.delta_emission import emission_lines
from gammaforge.validation.delta_comparison import compare_emission_bins
VARIANTS = (('headon', 0.0, 0.0, 0.0, 0.0), ('crossed_small', 0.02, -0.015, 0.4, 0.37))

def _hash_sources():
    from gammaforge.engines.xigma import stages
    paths = (Path(__file__), Path(emission_lines.__code__.co_filename), Path(compare_emission_bins.__code__.co_filename), Path(stages.__file__), Path(scenarios.__file__), Path(laser_module.__file__), Path(bunch_module.__file__))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def run_pilot(particles=16000, n_steps=64, seed=20260721, retarget_bins=64, quadrature_order=8):
    before = _hash_sources()
    records = []
    for scenario in scenarios.SCENARIOS:
        for name, txz, tyz, eps, psi in VARIANTS:
            laser = replace(scenario.laser, theta_xz=Quantity(txz, 'rad'), theta_yz=Quantity(tyz, 'rad'), ellipticity=eps, psi_pol=Quantity(psi, 'rad'))
            interaction = scenarios.build(replace(scenario, laser=laser), SamplingSpec(n_particles=particles, seed=seed, prefilter=0.001))
            samples = integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e, n_steps=n_steps)
            photon = float(interaction.laser.photon_energy())
            C = (1 + np.cos(txz) * np.cos(tyz)) / 2
            tables = []
            for shape, rt in [((16, 8, 8, 16), retarget_bins), ((32, 16, 16, 32), retarget_bins), ((16, 8, 8, 16), 2 * retarget_bins)]:
                tables.append((shape, rt, retarget_ahat(deposit_shape_table(samples, n_bins=shape, scheme='cic'), float(samples.intensity_peak), n_bins=rt)))
            for oi, (tx, ty) in enumerate(((0.0, 0.0), (0.5 / float(interaction.beam.gamma0()), -0.25 / float(interaction.beam.gamma0())))):
                energies, weights = emission_lines(samples, tx, ty, photon_energy=photon, psi_pol=psi, ellipticity=eps, theta_xz=txz, theta_yz=tyz)
                edges = np.asarray(np.geomspace(float(0.9 * energies.min()), float(1.1 * energies.max()), 25), dtype=float)
                for shape, rt, table in tables:
                    scale = 4 * photon * C

                    def density(x, table=table, scale=scale):
                        return spectrum_from_table(table, tx, ty, np.asarray(x) / scale, psi_pol=psi, ellipticity=eps, theta_xz=txz, theta_yz=tyz) / scale
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
            'scheme': 'cic', 'prefilter': 1e-3, 'backend': 'numpy',
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
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.particles < 1 or a.n_steps < 1 or a.seed < 0 or (a.retarget_bins < 1) or (a.quadrature_order < 2):
        p.error('counts positive, seed nonnegative, quadrature order >= 2')
    try:
        report = run_pilot(a.particles, a.n_steps, a.seed, a.retarget_bins, a.quadrature_order)
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
