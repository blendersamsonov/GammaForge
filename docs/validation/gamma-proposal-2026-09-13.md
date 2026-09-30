# Gamma-resonance proposal integration — 2026-09-13

The original uncommitted RES080 prototype allocates angular samples using the product
of angular and gamma marginals at a representative intensity. Integration preserves the
current DER013 target kernel and adds direction Doppler at each proposal quadrature node.
Invalid proposal roots get a positive floor; the prototype's midpoint guard could assign
zero probability to an interval containing valid emission. The permanent CUDA regression
places all emitting directions beyond the nominal inverse-resonance boundary and checks
nonzero agreement with NumPy within 5%.

## Accuracy and warm runtime

The recorded benchmark compared the validated sampler
at commit 9ce2d26 with the candidate on identical tables. Nine cases cover the
shared bank, crossing, off-axis/high-gamma tables, and
an energy-angle-correlated broad-intensity stress case. All CPU references pass angular
input refinement (8/16, or 32/64 at gamma 10,000). At rings=32, subsampling=32:

| Case | Baseline spectral L1 | Candidate spectral L1 | Candidate/baseline runtime |
|---|---|---|---|
| baseline | 0.377% | 0.089% | 0.95 |
| low_a0 | 0.377% | 0.089% | 1.04 |
| near_a0_max | 0.782% | 0.452% | 1.16 |
| crossed | 0.656% | 0.029% | 1.15 |
| wide_offaxis | 0.595% | 0.584% | 1.03 |
| narrow_offaxis | 0.362% | 0.367% | 1.12 |
| highgamma10000_crossed | 0.184% | 0.170% | 1.10 |
| highgamma10000_circular | 0.181% | 0.168% | 0.94 |
| correlated_broad_ahat | 0.045% | 0.075% | 1.20 |

Runtime is median wall time over three warm calls, including transfer/preparation; it
excludes kernel compilation and Stage 0/1. Millisecond-scale timings are noisy and do not
establish a universal speedup. Measurements also covered rings/subsampling 16/16, 32/32,
64/128 and 64/256. Both samplers satisfy the existing 3% yield, 5% L1 and 1% centroid
budgets at every measured setting. The new proposal improves accuracy per sample for
Gaussian narrow-energy cases; it can lose accuracy when one intensity mean and a
product of marginals poorly represent the target correlations. Positive support
prevents exclusion but does not guarantee lower finite-quadrature error in every case.

Reproduce from this checkout with actual CUDA:

```sh
mkdir -p output/validation
PYTHONPATH=src .venv/bin/python scripts/benchmark_gamma_proposal.py --output output/validation/gamma-proposal-benchmark.json
```

The benchmark is a numerical efficiency comparison, not independent physics closure.
The target integrand, normalizations and beta=1 convention are unchanged from RES082.
The separate prototype worktree's GPU delta implementation is not integrated here.

## Regression checks

The fast tier passed 489 tests. The full `pytest --run-heavy -q` suite passed
802 tests with 1 skipped in 701.50 seconds. The proposal support regression and
existing Doppler, CDF and sampler-control checks pass on actual CUDA. The alpha
validation command and all walkthrough notebooks also passed.

The standalone CUDA release gate
passed all required checks across eight cases on the GTX 1660 Ti, including gamma
10,000 and off-axis support. The run recorded 80 convergence diagnostics and
unchanged source fingerprints. Reproduce with:

```sh
PYTHONPATH=src .venv/bin/python scripts/validate_cupy_release.py --output output/validation/cupy-release-gamma-proposal.json
```
