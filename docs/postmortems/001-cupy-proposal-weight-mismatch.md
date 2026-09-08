# CuPy proposal/weight mismatch

## Executive summary

The ported GPU angular sampler ran successfully but disagreed with CPU integration
by 42% on an alpha diagnostic. Its approximate inverse CDF did not sample the PDF
used in its weights, and single-precision cancellation compounded the error. Loose
median-cell comparisons and an unrefined CPU reference obscured the failure. Sampling
code needs a physics-free measure test, precision stress tests, and a converged
reference before an execution smoke test can become an accuracy claim.

## Summary

The sampler was kept experimental rather than promoted into the default alpha path.
The numerical repairs are recorded in RES068. Neither normalization nor the
author-approved polarization convention was changed.

## Timeline

1. The CuPy port executed on CUDA, but the integrated-density gate failed and was
   retained as a strict expected failure. NumPy remained the default.
2. Increasing subsampling did not reliably reduce the discrepancy.
3. Isolating the proposal exposed a biased integral even without emission physics.
   Precision controls then separated cancellation from the proposal error.
4. Exact CDF inversion, stable algebra and support/allocation repairs restored the
   unchanged distribution gate. Refined-input CPU comparisons supplied stronger
   checks than the original coarse-grid sum.

## Root cause

Interpolating a small inverse-CDF lookup defines a new proposal distribution. The
kernel nevertheless divided by the original histogram probability. More samples
therefore converged toward the wrong integral. A sharply peaked histogram made this
visible: unity on [0, 1] integrated to 2.172 instead of 1.

The float32 subtraction `1 - v.n` was independently ill-conditioned for relativistic
particles. Flooring arc allocations also omitted low-weight arcs entirely, while
zero proposal cells and inconsistent edge handling could exclude target support.

Why the checks were insufficient:

- A median ratio over nonzero output cells can hide lost or misplaced mass.
- Execution, shape and finite-output checks do not test importance weights.
- The derivation described exact inversion, but the port implemented an approximation.
- CPU summation over original angular centers was not a converged integral of the
  interpolated H sampled by the GPU. A finer output grid does not fix that reference.

## Guardrails

- [CDF regressions](../../tests/test_xigma_sampler_cdf.py) exercise the actual production
  device search with peaked and zero-probability cells.
- [Synthetic regressions](../../tests/test_xigma_sampler_regressions.py) check smooth
  densities, high gamma, boundaries and invalid inputs.
- [Scenario regressions](../../tests/test_xigma_sampler_scenarios.py) compare mass and
  absolute density against refined input-angular quadrature for both polarizations.
- [RES068](../decisions/implemented/bug-fix/RES068-cupy-sampling-measure-and-stable-polarization.md)
  records the implementation choices; [the validation record](../ALPHA_GPU_VALIDATION.md)
  preserves measured limitations and a reproducible timing control.
