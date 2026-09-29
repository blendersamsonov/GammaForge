# Production xigma/delta refinement — 2026-09-29

This measurement revisits the six low-a0 refinement failures in the
[initial production report](delta-production-2026-09-14.md). RES092 adds one
bounded finer matrix without changing RES084's budgets or the engine defaults.

## Reproduce

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m gammaforge.validation.run --production
```

The production driver measures all scenarios in the shared bank. A scenario
failing angular-table or retarget refinement is measured once more, including
all geometries, observers, agreement and refinement checks. The initial failed
checks remain report notes; the complete finer matrix determines numerical
acceptance. Missing coverage or changed sources in either pass remain failures.
There are no scenario-name exceptions or repeated retries until success.
The command logs each trajectory/table calculation and completed observer
comparison to stderr so long-running measurements remain observable.

The matrix uses 16,000 particles, 64 integration steps, seed 20260721, three
geometries/polarizations, two observers, 24 physical energy bins and Gauss
quadrature orders 16/32. Gamma and shape each retain 64 bins. The configured
carrier axis has eight bins and collapses to one for these unchirped scenarios.

| Pass | Angular bins per axis | Requested retarget bins |
|---|---|---|
| Initial | 32, 64, 64 | 512, 512, 1024 |
| Bounded retry | 64, 128, 128 | 1024, 1024, 2048 |

Counts remain absolute fixed-direction, finite-energy-window counts, not
aperture-integrated photon yield. Agreement budgets remain 3% count, 5% spectral
mass L1 and 1% centroid; each refinement receives one third of those budgets.

## Low-a0 diagnostic

Rerunning the original matrix reproduced all six original failures. After
pruning unreachable retarget columns before output allocation, the head-on
original-grid spectra and first energy moments agreed with the unoptimized
packet to relative tolerance 1e-13. This optimization leaves the nonuniform grid
law and reachable weights unchanged; it does not tune the default decades.

The finer low-a0 probe passes every agreement and refinement gate. The six
previously failing spectral-L1 measurements are:

| Geometry/polarization | On-axis retarget, initial → finer | Off-axis angular, initial → finer |
|---|---:|---:|
| Head-on linear | 3.0210% → 0.1764% | 1.7408% → 0.7191% |
| Crossed elliptical | 2.9355% → 0.9006% | 2.0180% → 0.7493% |
| Crossed circular | 2.2576% → 0.5675% | 2.0653% → 0.7979% |

All are below the unchanged 1.6667% refinement budget on the finer matrix.
Local raw diagnostic packets, including arrays and source fingerprints, are
saved under the ignored output directory:

- `output/validation/low-a0-original-grid-20260928.json`
- `output/validation/low-a0-refined-grid-probe-20260928.json`

## Full production result

The full three-scenario run completed with 388 passing executed checks, zero
numerical failures and two scientific coverage blockers. The initial matrix
recorded 54 direction/grid measurements. Its 362 checks reproduced exactly
the six low-a0 L1 refinement failures and no others. The runner retried only
low-a0; all 122 checks on its 18 finer-grid measurements passed. Initial
failures remain in the final report as diagnostics.

Across the retried low-a0 directions, the maximum finer angular/retarget L1
refinement was 0.9006%, versus the unchanged 1.6667% budget. The finest-grid
reference comparisons were at most 0.1219% in count, 0.7566% in spectral L1,
and 0.0092% in centroid. Source fingerprints were identical before and after
both measurements and still match the recorded source files.

The local raw packets and complete production report are saved at:

- `output/validation/delta-production-20260929-pass1.json`
- `output/validation/delta-production-20260929-pass2.json`
- `output/validation/production-20260929.txt`

The long run also saved one raw packet after each scenario/geometry calculation,
under `output/validation/delta-production-20260929-pass{1,2}-*.json`. The output
directory is gitignored; these files are local evidence, not committed fixtures.
The command still exits nonzero because the two coverage blockers remain.

## Scientific scope

These measurements share Stage-0 trajectories, luminosities, nonlinear shape and
reduced-model assumptions. Particle/seed, Stage-0, gamma/shape-grid,
angular-aperture and independent CUDA convergence remain unmeasured by this
gate; four-method coverage also remains open. Production must return nonzero
while these scientific blockers remain, even if its numerical checks all pass.
The separate [CPU/CUDA release measurement](cupy-release-2026-09-28.md) does not
replace these independent physics requirements.

## Regression checks

The default suite passed 424 tests with 27 heavy tests deselected. The subsequent
full heavy sweep passed 455 tests, including the first four retry regression
cases. All five retry cases passed separately after adding the persistent
non-convergence case. These tests distinguish resolvable redistribution from
persistent spectral bias, missing grids, changed source fingerprints and
refinement that still fails on the second matrix.

All three walkthrough notebooks rebuilt and executed successfully. The runner
and retry checks also passed after enabling progress logging on the normal
production command. The heavy sweep took about 13 minutes while sharing CPU
resources with the production measurement; this is not a standalone benchmark.
