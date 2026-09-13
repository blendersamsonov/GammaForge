"""Compare the gamma proposal with the validated pre-proposal sampler on identical tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import linecache
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import types

import numpy as np
from gammaforge.engines.xigma import spectrum_sampler as candidate
from gammaforge.engines.xigma.stages import Table, angular_spectrum_from_table
from gammaforge.validation.cupy_convergence import _default_cases, _grid, _refine_table, _compare

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '9ce2d26'


def load_baseline():
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:src/gammaforge/engines/xigma/spectrum_sampler.py'], cwd=ROOT, text=True)
    filename = '<gamma-proposal-baseline>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType('gammaforge.engines.xigma._gamma_proposal_baseline')
    sys.modules[module.__name__] = module
    exec(compile(source, filename, 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def stress_case():
    g = np.linspace(800., 1200., 25)
    x = y = np.linspace(-.002, .002, 17)
    a = np.array([0., .01, .1, .5, 1.])
    G, X, Y, A = np.meshgrid((g[:-1]+g[1:])/2, (x[:-1]+x[1:])/2,
                            (y[:-1]+y[1:])/2, (a[:-1]+a[1:])/2, indexing='ij')
    H = np.exp(-.5*((G-1000-60000*X)/45)**2 - .5*(X/.0007)**2 - .5*(Y/.0007)**2)
    H *= np.where(A < .1, 1., .3)
    return {'name': 'correlated_broad_ahat', 'table': Table(g,x,y,a,H,float(H.sum()),'proposal-stress'),
            'kwargs': {'theta_xz': .3, 'theta_yz': -.2, 'ellipticity': .4}}


def run(output, repeats):
    baseline, baseline_hash = load_baseline()
    path = Path(candidate.__file__)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    records = []
    for case in _default_cases() + [stress_case()]:
        table, kwargs = case['table'], case.get('kwargs', {})
        x,y,s = _grid(table, case.get('samples'))
        x,y,s = x[::2],y[::2],s[::2]
        factors = case.get('cpu_refinements', (8,16))
        references = [angular_spectrum_from_table(_refine_table(table,f),x,y,s,backend='numpy',**kwargs) for f in factors]
        convergence = _compare(*references,x,y,s,.03,.05)
        record = {'case':case['name'], 'cpu_refinements':factors, 'reference_convergence':convergence,
                  'grid': {'x':x.tolist(),'y':y.tolist(),'s':s.tolist()}, 'runs':[]}
        for rings,subs in [(16,16),(32,32),(64,128),(64,256)]:
            for name,module in [('baseline',baseline),('candidate',candidate)]:
                fn=module.calculate_angular_spectrum_gpu
                fn(table,x,y,s,rings=rings,subsampling=subs,**kwargs)
                times=[]
                for _ in range(repeats):
                    start=perf_counter(); result=fn(table,x,y,s,rings=rings,subsampling=subs,**kwargs)
                    times.append(perf_counter()-start)
                metrics=_compare(result,references[-1],x,y,s,.03,.05)
                record['runs'].append({'sampler':name,'rings':rings,'subsampling':subs,
                                       'seconds_median':float(np.median(times)), 'metrics':metrics})
        records.append(record)
        print(case['name'], 'CPU:',convergence['status'],
              [(r['sampler'],r['rings'],round(r['metrics'].get('l1_relative',float('nan'))*100,3)) for r in record['runs']],flush=True)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        output.write_text(json.dumps({'baseline_revision':BASELINE,'baseline_sha256':baseline_hash,
            'candidate_sha256_before':before,'candidate_sha256_after':after,'source_unchanged':before==after,
            'cupy':candidate.cp.__version__,'numpy':np.__version__,'repeats':repeats,'records':records},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    run(args.output,args.repeats)
