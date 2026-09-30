# Production xigma/delta validation — 2026-09-14

RES084 integrates the independent matched-bin measurement into the production
runner. The numerical budgets remain provisional; RES074 scientific acceptance
is open. Kascade implementation and its validation wiring are unchanged.

## Reproduce

From the repository root, select this checkout explicitly when the environment
shares an editable installation with other worktrees:

```sh
PYTHONPATH=src .venv/bin/python -m gammaforge.validation.run --production
```

This CPU command takes substantially longer than the restricted alpha gate. It
returns nonzero for numerical failures and outstanding scientific coverage.

The original raw packet recorded input representations, geometry, observation
directions, physical energy-bin edges, bin masses and moments, environment and
source SHA256 fingerprints. The measured conclusions are below. To save the same matrix
independently of the other production sections:

```python
import json
from pathlib import Path
from gammaforge.validation.delta_validation import (
    PRODUCTION_TABLE_CONFIGS, PRODUCTION_VARIANTS, packet_checks, run_pilot,
)

packet = run_pilot(
    variants=PRODUCTION_VARIANTS,
    table_configs=PRODUCTION_TABLE_CONFIGS,
    quadrature_order=16,
)
Path("delta-production.json").write_text(
    json.dumps(packet, indent=2, allow_nan=False, default=lambda a: a.tolist()) + "\n"
)
for check in packet_checks(packet):
    print(check)
```

## Measurement scope

All three shared-bank scenarios use 16,000 particles, 64 Stage-0 steps and seed
20260721. Each has head-on linear, small two-plane crossed elliptical, and crossed
circular polarization, with an on-axis and an off-axis observer. The independent
reference uses direction Doppler at beta=1 and its own transverse-dipole basis.

Each of the 18 direction cases uses 24 physical energy bins. NumPy densities are
integrated at Gauss orders 16 and 32. CIC tables have shape grids 64×32×32×64 and
64×64×64×64, first retargeted to 512 bins; the finer angular table is also retargeted
to 1024 bins. The 54 records preserve identical energy edges across refinements.

Agreement budgets are 3% count, 5% spectral-mass L1 and 1% centroid. Each measured
refinement uses one third of those budgets. No spectrum is rescaled to agree.
Counts are finite-energy-window counts per unit solid angle at a fixed direction,
not aperture-integrated or full-interaction photon yields.

## Numerical findings

The production run recorded unchanged source fingerprints and 54 measurement
records. Of 362 new checks, 356 passed and six failed. Every finest-grid comparison
met the agreement budgets. Maximum errors across the six direction cases per scenario:

| Scenario | Count error | Spectral L1 | Centroid error |
|---|---:|---:|---:|
| baseline | 0.0985% | 0.9621% | 0.0084% |
| low_a0 | 0.1094% | 1.0878% | 0.0307% |
| near_a0_max | 0.0968% | 0.8655% | 0.0043% |

All energy-quadrature checks passed. The failures are low-a0 L1 refinement errors,
against a 1.6667% budget:

| Geometry/polarization | On-axis retarget refinement | Off-axis angular-table refinement |
|---|---:|---:|
| head-on linear | 3.0210% | 1.7408% |
| crossed elliptical | 2.9355% | 2.0180% |
| crossed circular | 2.2576% | 2.0653% |

The command therefore returned 1 with six numerical failures and two separately
reported coverage blockers. Agreement with delta does not by itself establish
convergence between grids. These failures remain visible; no budget was loosened.

The low-a0 retarget tables retain only one populated ahat bin at both requested
resolutions; changing the requested resolution still changes that bin's edges and
representative value. This is relevant to the existing coarse-floor discussion in
RES032/RES053. No tuned grid default is changed by this validation integration.

## Remaining scope

The methods share trajectories, luminosities, ahat and reduced-model assumptions.
Particle/seed, Stage-0, gamma/shape-grid, angular-aperture and independent CUDA
convergence are not measured here. This measurement leaves scientific acceptance
false, and the production runner keeps this coverage and four-method coverage
as explicit blockers even if every numerical check passes.

## Regression checks

The full suite passed 825 tests with 1 skipped, including actual-CUDA and heavy
checks. The alpha command passed using explicit NumPy execution. All walkthrough
notebooks rebuilt and executed successfully. Failure-semantics tests cover missing
cases/grids/metrics, changed sources/edges, non-finite or zero signal, and excessive
agreement/refinement errors; only the production selector runs the new matrix.
