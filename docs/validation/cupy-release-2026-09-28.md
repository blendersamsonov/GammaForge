# CuPy numerical release gate — 2026-09-28

The complete release gate passed on a GeForce GTX 1660 Ti with Python 3.14.7,
NumPy 2.5.3, CuPy 14.2.0, CUDA runtime 12.9 and driver API version 13.0.
All source fingerprints were unchanged during the run.

Command: `python scripts/validate_cupy_release.py --output <report.json>`.
The local full report is `output/validation/cupy-release-20260928-2214.json`
(generated output, excluded from git). It contains the source fingerprints,
query grids, settings, all metrics, and stage timings.

## Results

Both public-engine runs passed. All 90 convergence checks passed, including all
54 required checks, across the nine cases. No numerical settings or tolerances
were changed to obtain this result.

| Case | Default CUDA versus refined CPU spectral L1 error |
|---|---:|
| baseline | 0.146% |
| low_a0 | 0.146% |
| near_a0_max | 0.286% |
| crossed | 0.144% |
| wide_offaxis | 0.664% |
| narrow_offaxis | 0.445% |
| highgamma10000_crossed | 0.239% |
| highgamma10000_circular | 0.234% |
| finite_line_moment2 | 1.206% |

The finite-line case has a 5.451% L1 correction relative to its delta model.
Its CPU refinement difference is 3.066% L1, below the existing 5% limit.
Default CUDA differs from the refined CPU result by 0.118% in integrated mass
and 0.00356% in spectral centroid, below the existing 3% and 1% limits.

## Runtime

Recorded stages total 1,503 seconds, approximately 25 minutes. CPU calculations
account for 1,489 seconds; CUDA convergence calculations account for 14 seconds.
The two 64-fold high-gamma CPU refinements take 503 and 536 seconds respectively.
The earlier interrupted run therefore did not establish a numerical failure.
Per-stage stderr progress and JSON timings now expose this cost.

## Scientific scope

This closes the outstanding aggregate CPU/CUDA numerical gate for these sources.
It does not close independent arbitrary-angle scientific acceptance. The six
low-a0 refinement failures recorded in `delta-production-2026-09-14.md` belong
to the separate particle-reference validation schedule and remain unresolved.
Independent finite-line, trajectory, particle/seed and angular-aperture acceptance
work remains open. Result metadata states this limitation explicitly.
