# RES077 — Matched energy-bin delta comparison

Status: implemented
Type: testing

## Problem

The independent lines in RES076 need comparison with smooth table spectra without
confusing histogram averages with pointwise densities. Total-count agreement alone
can hide spectral redistribution, and a single table resolution cannot distinguish
numerical discretization from an emission-model discrepancy.

## Decision

`src/gammaforge/validation/delta_comparison.py` integrates a candidate physical-energy
density in each supplied bin using Gauss–Legendre quadrature. Compute first energy
moments with the same quadrature, while the reference uses exact weighted line
energies in the window. Compare refined candidate masses without renormalization;
retain reference tails and candidate mass in reference-zero bins. Undefined ratios
and centroids use null values rather than infinities or fabricated zero centroids.

Compare two energy-quadrature orders and report finite-window yield, integrated L1
and centroid refinement metrics. Provisional numerical thresholds are one third of
RES074's outer budgets; they are not a scientific acceptance criterion. Empty
directions cannot establish a nonzero signal or validate a release case.

`scripts/validate_delta_emission.py` runs a CPU-only pilot on the shared scenario
bank with head-on and two-plane crossed geometries and two observer directions.
Use fixed Stage-0 particles for each comparison. Change the shape grid and the
retarget grid separately, preserving the author-owned ahat defaults. Convert the
candidate energy coordinate and density with the once-only nominal crossing factor.
Record settings, inputs, arrays, environment and selected before/after source hashes.
Completion means the measurement ran without source mutation, never scientific
acceptance; the report explicitly retains unresolved numerical work.

## Alternatives considered

- Bin-center density comparisons: compare different integration measures.
- Bin-midpoint centroids: introduce an avoidable bin-width bias into energy-shift checks.
- Rescale spectra to match totals: hides normalization errors.
- Worst relative error in every bin: dark or tiny bins dominate the statistic;
  integrated absolute mass differences retain spurious support without dividing by zero.
- Change low-ahat defaults immediately: bypasses the author checkpoint and obscures
  the separate convergence studies.

## Rationale

Analytic integration and injected-error tests check the comparator independently
of the production engine. The pilot separates numerical error sources while remaining
cheap enough to rerun during development. It extends RES076 and partly implements
RES074; neither decision is superseded.

## Consequences

`docs/validation/delta-numpy-pilot-2026-09-09.json` records the initial measurements.
`docs/handoffs/delta-arbitrary-angle-validation-2026-09-09.md` records interpretation
and remaining work. This is fixed-direction finite-window validation, not integrated
angular yield or a public-engine acceptance gate. Particle, Stage-0, angular-aperture,
broader geometry/high-gamma and actual-CUDA convergence remain open. Production
physics and alpha defaults are unchanged; CuPy stays experimental.

## Amendments

> **2026-09-29 — Documentation relocation.** The completed delta handoff was retired by issue #8. Its pilot interpretation is retained in `docs/validation/delta-convergence-2026-09-10.md` and the cited JSON packet.
