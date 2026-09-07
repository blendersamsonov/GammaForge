# Alpha GPU sampler validation

Status: blocked on numerical agreement — 2026-09-07.

The CuPy 14.2.0 sampler executes on the GTX 1660 Ti (driver 580.173.02), and its
execution/shape tests pass. This is not validation of numerical equivalence.

For the 40,000-particle baseline with a 32x32x32x64 table and a 5x5x8 observation
cube, the NumPy-reference integral was `5.2868564e7`; the CuPy result at the fixed
256 samples and subsampling 32 was `7.5021186e7`, a ratio of `1.4190`. Increasing
subsampling to 64, 128, and 256 gave ratios `1.2299`, `1.2518`, and `1.3733`.

A constant smooth table also produced non-finite CuPy values at supported query points.
The host wrapper now raises an actionable error for non-finite GPU output. Until the
quadrature discrepancy and that defect are resolved, use `backend="numpy"`; `auto` and
explicit `cupy` are experimental paths only. *DER008* remains derived.

Reproduce the distribution discrepancy on a CUDA host:

```sh
python -m pytest -q tests/test_xigma_gpu_sampler.py --runxfail
```

The stronger integral/absolute-density comparison is a strict expected failure in the
ordinary suite, so it stays visible until resolved. The original 30% median-cell-ratio
check is only a smoke test. Neither the coarse CPU grid nor the GPU sampler has been
shown converged in this diagnostic; these numbers establish disagreement, not which
approximation supplies an exact continuum answer. Do not tune the shared normalization
constant to fit this measurement.
