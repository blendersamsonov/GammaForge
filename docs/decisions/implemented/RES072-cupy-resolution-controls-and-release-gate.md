# RES072 — CuPy resolution controls and numerical release gate

Status: implemented
Type: testing

## Problem

The repaired CuPy sampler had a fixed ring count and no engine-schema sampling
control. Increasing subsampling alone cannot test the midpoint approximation of
circle/rectangle boundaries. CPU agreement also needs a converged input-angular
quadrature reference, rather than simply increasing output plot resolution.

## Decision

Expose bounded integer sampler ring and subsampling controls through the xigma
schema, angular queries, and result metadata. Keep the existing defaults of 32
rings and 32 subsamples. Ring counts span 8 through 64; sample indices retain the
uint32-safe bound. Use separate 32-capacity and 64-capacity kernels so the default
does not allocate the larger shared-memory arrays. Cumulative-weight construction
strides over every arc when arc count exceeds the fixed thread-block size.

The standalone `scripts/validate_cupy_release.py` command requires actual CUDA and
combines a public-engine crossed-overflow regression with
`src/gammaforge/validation/cupy_convergence.py`. Its report records environment,
settings, and individual numerical checks. It fingerprints the actual imported
sampler/reference and supporting modules before and after validation; a changed
fingerprint invalidates the run. Unavailable CUDA, failed checks, and inconclusive
reference convergence are not successful release results.

Reference quadrature refines the same bilinearly interpolated input H while keeping
gamma/intensity axes and output queries fixed. Independent GPU sweeps vary rings
and subsampling; checks cover default settings, fine-reference agreement, and changes
between refinement levels. Integral error, integrated absolute density error, and
spectral centroid all matter. Supplement the shared scenario bank with explicit
off-axis and high-energy cases, retaining nonuniform intensity bins.

## Alternatives considered

- Increase subsampling alone: cannot establish convergence of ring-boundary geometry.
- Increase output resolution alone: does not refine either integration algorithm.
- Allocate the maximum scratch space for every invocation: unnecessarily changes
  the default kernel's occupancy tradeoff.
- Require every deliberately coarse diagnostic to pass: confuses evidence about
  convergence with the acceptance of default and refined settings.
- Treat a skipped GPU suite or a returned failed report as success: would certify
  an unexecuted or failing backend.
- Remove the experimental warning as soon as the gate exists: implementation of
  a numerical gate is neither measured acceptance nor independent physics closure.

## Rationale

This extends RES068 and RES069 without changing their emission formulas, proposal
measure, normalization, or default backend. Finite-resolution comparisons between
different quadrature algorithms have explicit error budgets; they are not the
roundoff-level backend-invariance check described separately in the grand plan.

## Consequences

Scripts can perform controlled refinement and preserve the actual sampler settings
with results. The numerical gate can fail honestly for an insufficiently resolved
CPU reference. Measurements and remaining promotion limits are recorded in
`docs/ALPHA_GPU_VALIDATION.md`. Independent arbitrary-angle emission validation,
GUI, Kascade optimization, and GPU Stage 0/1 remain outside this work.

## Amendments

> **2026-09-29 — Documentation relocation.** The measurement formerly at `docs/ALPHA_GPU_VALIDATION.md` is retained at `docs/validation/alpha-gpu-sampler-2026-09.md`; newer CUDA evidence is in `docs/validation/cupy-release-2026-09-28.md`.
