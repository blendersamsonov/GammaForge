"""Independent matched-bin emission diagnostics and provisional production gates."""
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
    from gammaforge.engines.xigma import stages, spectrum_sampler
    paths = (Path(__file__), Path(emission_lines.__code__.co_filename), Path(compare_emission_bins.__code__.co_filename), Path(stages.__file__), Path(spectrum_sampler.__file__), Path(scenarios.__file__), Path(laser_module.__file__), Path(bunch_module.__file__))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_pilot(particles=16000, n_steps=64, seed=20260721, retarget_bins=64, quadrature_order=8,
              *, table_configs=None, scenario_names=None, backend='numpy', rings=32, subsampling=32,
              source_scenarios=None, variants=None):
    before = _hash_sources()
    records = []
    bank = scenarios.SCENARIOS if source_scenarios is None else tuple(source_scenarios)
    selected_scenarios = bank if scenario_names is None else tuple(
        scenario for scenario in bank if scenario.name in scenario_names
    )
    variants = VARIANTS if variants is None else variants
    if table_configs is None:
        configs = tuple((shape, multiplier * retarget_bins)
                        for shape, multiplier in DEFAULT_TABLE_CONFIGS)
    else:
        configs = tuple(table_configs)
        if not configs or any(len(shape) != 4 or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in (*shape, rt))
                              for shape, rt in configs):
            raise ValueError('table_configs must contain positive integer (G,X,Y,A,RETARGET) entries')
    if scenario_names is not None:
        known = {scenario.name for scenario in bank}
        if not scenario_names or any(name not in known for name in scenario_names):
            raise ValueError('scenario_names must contain known scenarios')
    for scenario in selected_scenarios:
        for name, txz, tyz, eps, psi in variants:
            laser = replace(scenario.laser, theta_xz=Quantity(txz, 'rad'), theta_yz=Quantity(tyz, 'rad'), ellipticity=eps, psi_pol=Quantity(psi, 'rad'))
            interaction = scenarios.build(replace(scenario, laser=laser), replace(scenario.sampling, n_particles=particles, seed=seed))
            samples = integrate_trajectories(interaction.bunch, interaction.laser, interaction.N_e, n_steps=n_steps)
            photon = float(interaction.laser.photon_energy())
            C = (1 + np.cos(txz) * np.cos(tyz)) / 2
            observers = []
            for oi, (tx, ty) in enumerate(((0.0, 0.0), (0.5 / float(interaction.beam.gamma0()), -0.25 / float(interaction.beam.gamma0())))):
                energies, weights = emission_lines(samples, tx, ty, photon_energy=photon, psi_pol=psi, ellipticity=eps, theta_xz=txz, theta_yz=tyz, doppler='direction')
                edges = np.asarray(np.geomspace(float(0.9 * energies.min()), float(1.1 * energies.max()), 25), dtype=float)
                observers.append((oi, tx, ty, energies, weights, edges))
            for shape, rt in configs:
                table = retarget_ahat(deposit_shape_table(samples, n_bins=shape, scheme='cic'), float(samples.intensity_peak), n_bins=rt)
                for oi, tx, ty, energies, weights, edges in observers:
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
                        'scenario': scenario.name, 'scenario_repr': repr(scenario), 'prefilter': scenario.sampling.prefilter, 'variant': name,
                        'geometry': {'theta_xz': txz, 'theta_yz': tyz, 'ellipticity': eps, 'psi_pol': psi},
                        'observer': {'index': oi, 'theta_x': tx, 'theta_y': ty},
                        'photon_energy_erg': photon, 'shape': list(shape),
                        'actual_table_shape': list(table.H.shape), 'intensity_peak': float(samples.intensity_peak),
                        'ahat_defaults': {'min': DEFAULT_AHAT_MIN, 'max': DEFAULT_AHAT_MAX, 'decades': DEFAULT_AHAT_DECADES},
                        'retarget_bins': rt, 'edges': edges.tolist(), 'comparison': cmp,
                    })
                del density, table
    after = _hash_sources()
    return {
        'status': 'completed' if before == after else 'failed',
        'pilot_only': True, 'scientific_pass': False,
        'source_sha256_before': before, 'source_sha256_after': after,
        'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
        'settings': {
            'particles': particles, 'n_steps': n_steps, 'seed': seed,
            'retarget_bins': retarget_bins, 'quadrature_order': quadrature_order,
            'scheme': 'cic', 'backend': backend,
            'rings': rings, 'subsampling': subsampling,
            'doppler': 'direction', 'doppler_beta': 1.0,
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


# Fixed-direction regression coverage; full RES074 acceptance remains open (RES084).
PRODUCTION_VARIANTS = (*VARIANTS, ('crossed_circular', -0.03, 0.01, 1.0, 0.71))
PRODUCTION_TABLE_CONFIGS = (
    ((64, 32, 32, 64), 512),
    ((64, 64, 64, 64), 512),
    ((64, 64, 64, 64), 1024),
)
AGREEMENT_BUDGETS = {'yield': 0.03, 'l1': 0.05, 'centroid': 0.01}


def _mass_errors(mass, moment, reference_mass, reference_moment):
    total, ref_total = float(np.sum(mass)), float(np.sum(reference_mass))
    first, ref_first = float(np.sum(moment)), float(np.sum(reference_moment))
    if not np.all(np.isfinite([total, ref_total, first, ref_first])) or min(total, ref_total, first, ref_first) <= 0:
        return dict.fromkeys(AGREEMENT_BUDGETS, float('inf'))
    return {
        'yield': abs(total / ref_total - 1),
        'l1': float(np.sum(np.abs(np.asarray(mass) - reference_mass)) / ref_total),
        'centroid': abs((first / total) / (ref_first / ref_total) - 1),
    }


def packet_checks(packet):
    """Turn a complete production packet into fail-closed numerical checks."""
    from .invariance import Check

    checks = [Check('delta source stability', packet['status'] == 'completed',
                    'source fingerprints before/after must match')]
    expected = {(name, variant[0], observer)
                for name in packet['settings']['scenarios']
                for variant in PRODUCTION_VARIANTS for observer in (0, 1)}
    groups = {}
    for record in packet['records']:
        key = (record['scenario'], record['variant'], record['observer']['index'])
        groups.setdefault(key, []).append(record)
    checks.append(Check('delta scenario coverage', bool(expected) and set(groups) == expected,
                        f'{len(groups)} of {len(expected)} required direction cases'))
    for key in sorted(expected):
        label = '/'.join(map(str, key))
        records = groups.get(key, [])
        configs = [(tuple(r['shape']), r['retarget_bins']) for r in records]
        complete = configs == list(PRODUCTION_TABLE_CONFIGS)
        checks.append(Check(f'{label} delta refinement coverage', complete,
                            'angular-table and retarget grids must all be measured'))
        if not complete:
            continue
        edges_match = all(np.array_equal(records[0]['edges'], r['edges']) for r in records)
        checks.append(Check(f'{label} delta matched energy edges', edges_match,
                            'all table refinements use the identical physical-energy bins'))
        if not edges_match:
            continue
        for index, record in enumerate(records):
            candidate = record['comparison']['candidate']
            for metric in AGREEMENT_BUDGETS:
                error = candidate['refinement_errors'].get(metric)
                budget = AGREEMENT_BUDGETS[metric] / 3
                checks.append(Check(f'{label} grid {index} energy quadrature {metric}',
                                    error is not None and np.isfinite(error) and abs(error) <= budget,
                                    f'error {error}; provisional refinement budget {budget:.4g}'))
        fine = records[-1]['comparison']
        errors = {'yield': fine['relative_yield_error'], 'l1': fine['l1_mass_error'],
                  'centroid': fine['centroid_relative_error']}
        for metric, error in errors.items():
            budget = AGREEMENT_BUDGETS[metric]
            checks.append(Check(f'{label} xigma/delta {metric}',
                                error is not None and np.isfinite(error) and abs(error) <= budget,
                                f'error {error}; provisional agreement budget {budget:.4g}'))
        for name, coarse, refined in (('angular table', records[0], records[1]),
                                      ('retarget', records[1], records[2])):
            c, r = coarse['comparison']['candidate'], refined['comparison']['candidate']
            errors = _mass_errors(c['refined_bin_mass'], c['refined_first_moment'],
                                  r['refined_bin_mass'], r['refined_first_moment'])
            for metric, error in errors.items():
                budget = AGREEMENT_BUDGETS[metric] / 3
                checks.append(Check(f'{label} {name} refinement {metric}',
                                    np.isfinite(error) and error <= budget,
                                    f'error {error:.6g}; provisional refinement budget {budget:.4g}'))
    return checks


def production_checks(source_scenarios):
    """CPU fixed-direction checks; no automatic scientific-acceptance promotion."""
    packet = run_pilot(source_scenarios=source_scenarios,
                       variants=PRODUCTION_VARIANTS, table_configs=PRODUCTION_TABLE_CONFIGS,
                       quadrature_order=16)
    return packet_checks(packet), [
        'Independent delta direction-Doppler emission lines versus NumPy table densities: '
        '16000 particles, 64 steps, seed 20260721, 24 matched energy bins, q16/q32. '
        'Head-on linear, crossed elliptic and crossed circular; on/off axis. '
        'Counts are absolute per-direction finite-window counts, not full photon yield.',
        'Agreement budgets (3% count, 5% spectral L1, 1% centroid) remain provisional; '
        'each measured refinement must stay within one third of its agreement budget. '
        'Stage 0, sampling, ahat and reduced-model assumptions are shared.',
    ], [
        'delta scientific acceptance remains open (RES074): particle/seed, Stage-0, '
        'gamma/shape-grid, angular-aperture and independent CUDA convergence are not '
        'measured by this fixed-direction CPU gate',
    ]


if __name__ == "__main__":
    raise SystemExit(main())
