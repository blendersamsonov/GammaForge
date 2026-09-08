# Alpha GPU sampler validation

Status: numerical repairs verified on CUDA; backend still experimental — 2026-09-07.

CuPy 14.2.0 on the GTX 1660 Ti (driver 580.173.02) passes the original distribution
gate, direct CDF tests, smooth-table/high-gamma regressions, and head-on scenario-bank
comparisons. NumPy remains the alpha default. This does not close independent emission
validation or establish convergence for arbitrary new parameter regimes (RES068).

## What caused the discrepancy

The original 40,000-particle diagnostic deposited a 32x32x32x64 table, then applied
the default ahat retargeting (one bin here). On its 5x5x8 observation cube, NumPy gave
`5.2868564e7` and the old GPU sampler `7.5021186e7`: ratio `1.4190`. Increasing
subsampling did not consistently improve it. These were two finite approximations,
not a comparison against an exact continuum result.

Three numerical mechanisms mattered:

- The 32-point interpolated inverse-CDF lookup generated a different proposal from
  the 31-cell PDF used in its importance weights. A physics-free peaked-proposal
  example integrated unity on [0, 1] as 2.172 instead of 1. Exact CDF bracketing and
  within-cell inversion now use precisely the probability used for weighting.
- Direct float32 evaluation of `1 - v.n` lost accuracy at gamma 2000 and reached zero
  at gamma 10000. The algebraically identical positive-sum expression preserves
  RES060's lab-vector convention. A float64 diagnostic independently checked it.
- Flooring proportional work allocation dropped low-weight arcs completely. Each
  positive arc now receives work; a positive proposal floor preserves interpolated
  support. A missing counter-initialization barrier and boundary/gamma-support
  inconsistencies were also corrected.

No shared normalization constant or physics convention was changed. DER008 remains
derived; its implementation amendment distinguishes numerical repairs from physics
validation. The former strict expected-failure test is now an ordinary passing test,
with its original 10% integrated-mass and absolute-density limits unchanged.

## Convergence-aware regression checks

CPU quadrature sums input angular cell centers; GPU quadrature samples a bilinearly
interpolated H. Refining only the output observation grid cannot reconcile these
integration approximations. The new checks refine the **input angular quadrature of
the same interpolated H**, keeping output queries and gamma/ahat axes fixed.

The scenario check iterates the shared bank with 4,000 particles, seed 20260721,
32 time steps, 12 bins per deposited axis and default ahat retargeting. It compares
GPU subsampling 256 against 8x and 16x angular refinements of H, on the same 5x5x10
output cube, for both x and y linear polarization. Errors below are over that finite
cube, not an all-angle/all-energy total-yield claim.

| Check, maximum across six scenario/polarization cases | Measured error |
|---|---:|
| CPU mass change, 8x to 16x refinement | 0.216% |
| CPU integrated absolute-density change | 0.532% |
| GPU mass difference from 16x CPU reference | 0.077% |
| GPU integrated absolute-density difference / reference mass | 0.182% |

Regression limits are 2%/3% for CPU mass/density convergence and 3%/5% for GPU
mass/density agreement. An independent affine-density table also checks refinement
from 128 to 256 angular bins, with four ahat bins. Other tests cover gamma 10000,
shifted/thin support, zero/empty inputs, invalid inputs and zero-probability CDF cells.
The bank uses head-on Gaussian scenarios and does not exhaust nonlinear/intensity,
tail, boundary or arbitrary-geometry regimes. Fixed 32-ring midpoint geometry remains
an approximation; more subsamples alone cannot eliminate that discretization error.

```sh
python -m pytest -q tests/test_xigma_gpu_sampler.py tests/test_xigma_sampler_cdf.py \
  tests/test_xigma_sampler_regressions.py tests/test_xigma_sampler_scenarios.py
```

## Does binary search slow it down?

Not in the measured complete kernel. Searching 31 cells requires at most five
comparisons per sample. Removing the inverse-CDF setup and buffer also reduces
declared per-block shared storage from 37,892 to 22,916 bytes. This is a plausible
source of the speedup, not an occupancy-profiled causal attribution.

The benchmark excludes first-call compilation, uses 20 warm-up launches, and reports
medians of 15 runs. CUDA events measure the resident kernel plus output reset; wall
times include host validation, allocation and transfers. Stage 0/1 table construction
is excluded. All variants use 256 base samples, subsampling 32, and a 9x9x16 output
cube. Old source is pinned to git revision `c090909`.

| H shape | Old GPU / wall time | Repaired GPU / wall time |
|---|---:|---:|
| 32x32x32x1 (alpha default) | 8.00 / 9.24 ms | 4.94 / 6.78 ms |
| 32x32x32x32 (resolved ahat) | 175.20 / 178.20 ms | 140.12 / 143.97 ms |

An additional timing-only control retains all repairs except restoring the biased
inverse-CDF lookup and its storage. Its GPU times are 9.16 and 215.39 ms respectively,
also slower than exact search. It has slightly more shared storage than the historical
kernel because it retains the repaired scratch layout. This compares complete CDF
implementations, not the isolated cost of a search instruction. The biased control
is never exposed through the engine API. Timings are hardware/workload-specific and
are not performance assertions in the test suite.

```sh
python scripts/benchmark_xigma_sampler.py --repeats 15
```
