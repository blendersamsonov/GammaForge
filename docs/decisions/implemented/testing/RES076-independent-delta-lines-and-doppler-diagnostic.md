# RES076 — Independent delta lines and Doppler diagnostic

Status: implemented
Class: testing

## Problem

RES074 needs an emission reference independent of production polarization helpers.
Its nominal-axis resonance approximation must also be distinguished from the
manuscript's per-electron incident Doppler factor before interpreting residuals.

## Decision

Implement a separate validation reference in
`src/gammaforge/validation/references/delta_emission.py`. Evaluate the explicit
velocity and double-cross-product polarization projection in extended precision;
fail explicitly when the platform lacks extended long-double precision. Return
individual resonant energies and weights, and preserve finite-bin masses, densities,
underflow and overflow when histogramming. Do not alter the existing delta API.

`scripts/diagnose_delta_doppler.py` compares nominal-axis and per-electron resonance
energies with identical weights. Iterate the shared scenario bank with head-on and
two-plane crossed geometries, on/off-axis observation, and isolated tilted-electron
stress cases. Record unbinned moments, bin differences, settings, environment and
selected source fingerprints in strict JSON. Source mutation or calculation failure
cannot report completion; completion never means scientific acceptance.

## Alternatives considered

- Reuse production polarization: would hide common-mode projection errors.
- Modify production Doppler conversion immediately: would change the reduced model
  before measuring the discrepancy or reviewing the corresponding flux assumption.
- Compare histogram differences alone: arbitrarily small shifts can move a sampled
  line across an edge and produce a much larger finite-sample bin difference.
- Implement the whole RES074 matrix at once: unnecessarily couples reference review
  to table and GPU convergence work.

## Rationale

Holding weights fixed isolates frequency changes. Independent vector algebra and
analytic single-particle tests check the projection without making table agreement
its definition of correctness. Unbinned centroid and weighted RMS line shifts expose
the effect even when histogram bins conceal it or exaggerate edge crossings.

## Consequences

The initial report is `docs/validation/delta-doppler-2026-09-09.json`; measurements
and reproduction instructions are in
`docs/handoffs/delta-arbitrary-angle-validation-2026-09-09.md`.
Stage-0 luminosity, nonlinear shift, sampling and the reduced radiation model remain
shared assumptions. This does not independently validate flux or total yield.
RES074 remains proposed for matched-bin NumPy/CuPy comparisons and convergence;
no existing decision is superseded and CuPy remains experimental.
