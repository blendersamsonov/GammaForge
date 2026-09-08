# RES068 — CuPy sampling measure and stable polarization

Status: implemented
Class: bug-fix

## Problem

The alpha CuPy sampler disagreed with NumPy by 42% in integrated output and produced
non-finite values on a smooth high-gamma table. Increasing subsampling did not close
the discrepancy. DER008's claims of exact inverse-CDF sampling were not true of the
32-quantile approximation actually used by the implementation.

## Decision

The sampler binary-searches its cumulative azimuthal cell weights for each sample,
then interpolates inside that cell and uses that same cell's probability for its
importance weight. The redundant inverse-CDF lookup table and its setup are removed.
Every positive-weight arc receives at least one work item within the fixed sample
budget, with a block barrier between counter initialization and allocation.

The polarization denominator uses the exact identity

    1 - beta*u.n = 1/[gamma^2*(1+beta)] + beta*|u-n|^2/2

for unit particle/observer directions. This preserves RES060's lab-frame convention
without subtracting nearly equal single-precision values. Normalization is unchanged.

Angular interpolation clamps continuous coordinates before calculating fractions;
gamma support matches the NumPy center-domain interpolator. Ring geometry uses the
actual rectangle center and widths, without the previous corner-radius truncation.
The proposal marginal includes the nonuniform ahat widths and a positive floor of
0.001 times its maximum so a zero nearest-cell lookup cannot exclude interpolated
target support. Empty and invalid inputs are handled before launch. Sampler settings
and exact inversion are recorded in result metadata.

This partially supersedes RES062's sampling implementation, not its optional-backend
architecture or NumPy default. DER008 remains derived: these are numerical repairs,
not independent emission-physics closure.

## Alternatives considered

Increasing subsampling cannot fix a wrong proposal PDF or recover arcs assigned zero
samples. Adjusting normalization would mask distribution-dependent numerical errors.

Keeping the inverse-CDF approximation and weighting by its actual Jacobian is valid,
but would retain an unnecessary approximation to the intended proposal. Exact search
requires at most five comparisons for 31 cells and removes shared-memory/setup cost.

Using float64 for the entire kernel avoids the cancellation but exceeds the original
per-block shared-memory layout on the validation GPU and is unnecessary for the
stable algebraic expression. A reduced-capacity float64 diagnostic independently
confirmed the latter.

## Rationale

A physics-free peaked-proposal counterexample integrated unity over the unit interval
as 2.172 with the old lookup/weight pairing. The production CDF search now has direct
GPU regression tests; smooth-density, high-gamma and boundary checks exercise the
assembled kernel. The previously strict expected-failure distribution gate now passes
unchanged and is a normal test.

The CPU sums original angular cell centers while the GPU integrates an interpolated
table. Finite-grid comparisons therefore need independent angular quadrature refinement,
not an assumption that the original coarse sum is exact. Timing methodology and measured
results are in `docs/ALPHA_GPU_VALIDATION.md` and `scripts/benchmark_xigma_sampler.py`.

## Consequences

CuPy remains opt-in and requires table/sampler convergence checks for new calculations.
The current fixed ring discretization is not an exact curved-boundary integration.
Head-on numerical agreement does not validate arbitrary-angle emission or change the
author-owned ahat-grid default. No GUI or kascade optimization is included.
