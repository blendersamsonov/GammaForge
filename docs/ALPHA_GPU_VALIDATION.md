# Alpha GPU sampler validation

Status: numerical repairs and incident-polarization/crossing extension checked on
CUDA; backend still experimental — 2026-09-08.

CuPy 14.2.0 on the GTX 1660 Ti (driver 580.173.02) passes the original distribution
gate, direct CDF tests, smooth-table/high-gamma regressions, and head-on scenario-bank
comparisons. RES069 extends these checks to incident polarization and two-plane laser
crossing geometry, as recorded in the final section. NumPy remains the alpha default.
This does not close independent emission
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

## Incident polarization and crossing extension (2026-09-08, RES069)

CuPy now accepts the existing ellipticity, polarization-axis orientation and
two-plane laser-crossing inputs. The ring proposal, normalization, Stage-0 flux and
shared energy conversion are unchanged. This supports incident-polarization
dependence of intensity, not outgoing Stokes parameters or spatially varying
polarization. NumPy remains default; auto now selects CUDA for these geometries
when available, and otherwise falls back to NumPy.

Two additional numerical precautions are required:

- Evaluate the original Eq. udef vectors and their weighted squared norms, with
  stable electron/observer direction differences. The expanded norm cancels large
  terms for a longitudinal laser basis: a collinear float32 probe at crossing 0.3 rad
  gives zero at gamma 10000, versus the exact cosine-squared value 0.9126678.
- Apply the existing `1/s^2` inside the prefactor before multiplication by H.
  A crossing benchmark exposed an unscaled intermediate around 1.1e41 (above
  float32's 3.4e38 maximum), although its scaled value is only about 1.2e28.
  Moving the same factor earlier removes this overflow without fitting or clipping.

### Checks

- Twelve CUDA helper checks use an independent extended-precision direct Eq. udef,
  including gamma 2000/10000, both crossing planes, tilted electrons/observers,
  linear/elliptical/circular weights, the collinear cosine-squared limit and circular
  basis-rotation invariance. Limits are 1e-5 relative plus 1e-6 absolute.
- The existing head-on scenario checks remain passing. New integration checks cover
  a genuinely crossed deposited Gaussian table at (0.02, -0.015) rad and a synthetic
  six-bin nonuniform ahat table for head-on elliptical/circular and crossed elliptical
  polarization. CPU input-angular refinement is checked before applying the 3% mass
  and 5% integrated absolute-density agreement limits.
- Routing tests cover missing CUDA and both selection modes. A controlled public
  energy-conversion test checks the crossing correction and density Jacobian once.
  Additional manual public-engine runs with ellipticity 0.4 at (0.02, -0.015) and
  (0.3, 0.2) rad return finite, nonnegative, nonzero 8x3x3 collimated cubes and report
  the CuPy backend. These execution checks do not establish independent physics accuracy.
- The benchmark below also exercises the formerly overflowing (0.3, 0.2) case on
  both ahat grids and raises if any result is non-finite.

```sh
python -m pytest -q tests/test_xigma_gpu_polarization.py \
  tests/test_xigma_sampler_scenarios.py tests/test_xigma_gpu_sampler.py
```

### Final warmed timings

Same GTX 1660 Ti and CuPy 14.2.0; 20 warmups, 15 repetitions, 256 base samples,
subsampling 32 and output 9x9x16. GPU event time includes resident output reset;
host time includes allocations/transfers and excludes Stage 0/1 and first compilation.
The baseline is the repaired head-on source at revision `6420781`, not the older
biased sampler. The final run was exclusive of other CUDA work and includes the
early-energy-scaling fix.

| Case | GPU, ahat 1 | Host, ahat 1 | GPU, ahat 32 | Host, ahat 32 |
|---|---:|---:|---:|---:|
| Previous head-on kernel | 5.04 ms | 7.44 ms | 138.28 ms | 141.86 ms |
| New head-on linear | 3.50 ms | 5.44 ms | 92.57 ms | 96.30 ms |
| New head-on elliptical, epsilon 0.5 | 3.52 ms | 5.84 ms | 92.97 ms | 96.27 ms |
| New head-on circular | 3.53 ms | 5.67 ms | 93.54 ms | 97.51 ms |
| New crossed linear, (0.3, 0.2) rad | 3.55 ms | 5.69 ms | 93.20 ms | 98.16 ms |

The timing cases reuse identical Stage-0/1 tables to compare kernel workloads;
the crossed timing row is not a simulation of a newly sampled crossed collision.
Host-precomputed basis coefficients remove geometry trigonometry from the device
expression, but the speedup is a whole-kernel observation, not a profiled attribution.
Fixed-ring convergence and independent arbitrary-angle emission validation remain
open regardless of these numerical and timing results.

```sh
python scripts/benchmark_xigma_polarization.py --warmup 20 --repeats 15
```

### Reference limitation found during review

The existing NumPy expanded polarization factor also loses precision at sufficiently
high gamma when the electron and observer are collinear and the laser basis has a
longitudinal component. This is why the new direct-helper tests use an independent
extended-precision vector reference rather than treating NumPy's expansion as exact.

```python
from gammaforge.engines.xigma.stages import polarization_factor
polarization_factor(10000., 0., 0., 0., 0., 0., 0., 0.3, 0.)
# Existing NumPy: 1.388697735965252
# Exact collinear limit cos(0.3)**2: 0.9126678074548391
```

At gamma 2000, the same probe gives 0.9124055 (about 0.029% relative deviation).
These are single-ray polarization-factor errors, not measurements of whole-spectrum
or total-yield errors. The new CUDA vector evaluation passes the exact-limit check.
NumPy arithmetic repair is tracked separately in PROGRESS; its emission formulas
were not changed in this CuPy extension. Stage-0 total yields and the analytical
overlap calculation do not use this polarization helper.

Final verification: 604 tests passed, one skipped on CUDA; the minimum Python 3.12
focused run passed 31 tests and skipped 25 GPU tests. The headless alpha selector
also passed its existing analytical/NumPy checks.
