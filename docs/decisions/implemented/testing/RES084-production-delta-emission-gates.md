# RES084 — Production delta emission gates

Status: implemented
Class: testing

## Problem

Independent direction-Doppler emission comparisons existed only as a diagnostic
script. The production runner still reported them as unwired, and a completed
pilot did not enforce numerical agreement or refinement budgets.

## Decision

The production selector in `run_suite` runs the shared-bank fixed-direction
measurement in `delta_validation.py`. The standalone
`scripts/validate_delta_emission.py` delegates to the same module. The alpha and
ordinary selectors retain their existing scope. Reduced analytical/xigma checks
explicitly use NumPy so these validation selectors work without CUDA.

Each bank scenario is measured with head-on linear, two-plane crossed elliptical,
and crossed circular polarization, at an on-axis and an off-axis observer. The
reference independently constructs emission lines using DER012 and DER013. Both
methods share Stage-0 samples; the candidate is integrated over the same 24
physical-energy bins, without normalization fitting.

The finest candidate must meet the provisional RES074 budgets: 3% absolute
relative count error, 5% spectral mass L1, and 1% centroid error. Quadrature
refinement and separate angular-table and retarget refinements each have one
third of those budgets. Missing cases, grids or quadrature metrics, non-finite
errors, and changed source fingerprints fail the numerical gate.

These are absolute counts per unit solid angle in a finite energy window, not
an aperture integral or the interaction's full photon yield. The report retains
explicit blockers for unmeasured particle/seed, Stage-0, gamma/shape-grid,
angular-aperture and independent CUDA convergence. Four-method coverage remains
open. This implements a bounded portion of RES074; it does not close that proposal
or change any derivation status. RES077's standalone diagnostic remains available.

## Alternatives considered

- Import the script dynamically from the runner: would make installed validation
  depend on a repository-only script. The reusable measurement lives in the package.
- Run the new matrix in alpha: would expand the restricted release contract and
  increase its routine cost. Only the production selector runs this matrix.
- Treat completed calculations as acceptance: would allow unresolved refinements
  and scientific coverage to disappear behind a successful exit status.
- Require CUDA: would prevent a reproducible CPU measurement. This gate measures
  NumPy; the existing GPU diagnostic remains separately available.

## Rationale

Matched bins distinguish spectral discretization from emission disagreement, and
separate refinement checks expose numerical uncertainty before scientific review.
The existing independent reference is reused without importing production
polarization or energy conversion helpers into it.

## Consequences

Production validation is slower and can fail provisional numerical budgets. Even
when those pass, it returns nonzero while scientific coverage blockers remain.
Shared trajectories and reduced-model assumptions remain common-mode limitations.

The initial production measurement records six low-a0 L1 refinement failures even
though all finest-grid comparisons meet agreement budgets; see
`docs/validation/delta-production-2026-09-14.md`. This is unresolved numerical
convergence, not a reason to widen the budgets or close RES074.

## Amendments

> **2026-09-29 — Bounded refinement schedule.** RES092 extends the measurement
> with one finer fixed matrix for scenarios failing angular-table or retarget
> refinement. The original failures remain diagnostics, and the complete finer
> matrix determines numerical acceptance with the same budgets. Scientific
> coverage blockers remain unchanged.
