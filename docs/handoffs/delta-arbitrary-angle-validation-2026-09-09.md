# Delta arbitrary-angle emission validation — implementation handoff

Status: independent reference and initial Doppler diagnostic implemented (RES076);
matched-bin comparator and NumPy pilot implemented (RES077). Full convergence and
matched-bin CuPy acceptance remain pending.
Preparation starting commit: b11117f (CuPy RES072).
Decision: RES074 (partly implemented, still proposed). Scope: validation-only direct binning of each electron's
resonant photon energy, comparing with NumPy and CuPy spectral-angular calculations.
No GUI, Kascade optimization or GPU Stokes extension. The author-provided DER012
production polarization correction was merged separately and incorporated below.

## Current implementation and authority

### Current corrected basis (DER012 / RES078)

The author supplied `fix/crossing-transverse-dipole`, merged at c51c5a8. Production
NumPy/CuPy and Stokes now project the radiation basis transverse to each electron's
field-free velocity. Delta constructs that basis independently in extended precision.
The legacy delta automatically follows its production polarization helper; the new
reference does not import that helper. Stokes observer keywords and custom-basis
inputs were preserved during integration.

Corrected reports (distinct from the historical unprojected-basis reports below):

- `docs/validation/delta-numpy-transverse-pilot-2026-09-09.json`: 36 comparisons,
  retarget256/512 and q32/q64, otherwise the same fixed particles and pilot grids.
- `docs/validation/delta-doppler-transverse-2026-09-09.json`: recomputed frequency
  diagnostic with the corrected weights; maximum Gaussian centroid/RMS shifts are
  0.0002545%/0.0007586%. Individual frequency ratios did not change.

Both reports completed with matching before/after source fingerprints. At the
refined shape (32,16,16,32), retarget256, baseline crossed on-axis count error changes
from +3.893% to -0.205%, and L1 from 3.909% to 1.813%. Baseline head-on on-axis L1
is 1.261%. Off-axis baseline head-on/crossed L1 remains 13.00%/14.12%, so no spectral
convergence pass is claimed. Low-field on-axis L1 also remains 8–10%.

Reproduce corrected reports using the existing commands, with distinct output names:

```sh
.venv/bin/python scripts/validate_delta_emission.py --retarget-bins 256 --quadrature-order 32 --output /tmp/delta-numpy-transverse.json
.venv/bin/python scripts/diagnose_delta_doppler.py --output /tmp/delta-doppler-transverse.json
```

Verification includes 39 real-CUDA polarization checks, all 25 tests in the
crossing-yield file on CUDA (including 12 public-engine target-bound cases), and
46 independent delta-reference tests. The production conservation test now evaluates
the actual kernel rather than integrating a prescribed expected curve; independent
delta conservation tests cover cold/divergent particles at gamma 2000 and 10000.
These are finite-regime numerical checks, not exact finite-gamma physics validation.
The combined focused CPU/documentation run passed 139 tests; the minimum Python
3.12 environment passed all 71 delta/comparison/diagnostic tests without CuPy.
The full repository suite was not rerun for this integration.
The complete real-CUDA release gate also passes all 80 numerical checks across
eight cases and both public crossed-angle runs, with unchanged tolerances and
matching source fingerprints. Report:
`docs/validation/cupy-release-transverse-2026-09-09.json`.
This is NumPy/CuPy sampler agreement, not the pending independent delta/CuPy gate.

Next: refine off-axis/angular and low-ahat deposition using fixed reporting bins,
then particle/seed and Stage-0 integration before matched-bin actual-CUDA acceptance.
Do not use the old angular-grid error trend as a finding about the corrected formula.

### Historical initial implementation and unprojected-basis measurements

The files remain the entry points, but all numerical values in this historical
subsection and the following initial pilot subsection used the old radiation basis:

- `src/gammaforge/validation/references/delta_emission.py`: independent extended-
  precision polarization, nominal/particle resonance lines, finite-bin bookkeeping.
- `scripts/diagnose_delta_doppler.py`: fixed-weight Doppler diagnostic, not a gate.
- `tests/test_delta_emission.py` and `tests/test_delta_doppler_diagnostic.py`:
  analytic limits, independence, input/precision guards, histogram bookkeeping and
  diagnostic error injection.
- `docs/validation/delta-doppler-2026-09-09.json`: completed CPU diagnostic with
  matching before/after source fingerprints, 16,000 particles, 64 Stage-0 steps,
  seed 20260721, nine scenario/geometry cases, two observers each, six stress lines.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/diagnose_delta_doppler.py --output /tmp/delta-doppler.json
```

Verification: 148 focused tests passed (both new files, Stage-0/delta, validation,
and decision/derivation documentation checks). The 39 new tests also passed in the
minimal Python 3.12 environment without CuPy. The full repository suite and CUDA
acceptance were not rerun for this validation-only packet.

Maximum absolute centroid shifts across the bank and both observers:

| Laser crossing (xz, yz), rad | Centroid shift, % | Weighted RMS line shift, % |
|---|---:|---:|
| (0, 0) | 0.00000632 | 0.00000632 |
| (0.02, -0.015) | 0.0000152 | 0.0000833 |
| (0.3, 0.2) | 0.0000322 | 0.00125 |

The largest positive-weight individual Gaussian line shift was 0.00306%.
At gamma 2000 and crossing (0.3, 0.2), isolated electron x-slopes of +/-1 mrad
give +0.01493%/-0.01499%; +/-10 mrad give +0.14715%/-0.15200%.
These are shifts in resonance energy with identical weights, not changes in yield.
The Gaussian result suggests this correction is not a percent-level discrepancy
source in these specific inputs; it is not a universal bound or a convergence study.
The bank varies intensity but shares beam parameters and seed. Histogram L1 reaches
0.147% because finite samples cross bin edges; do not interpret that metric alone as
a physical discrepancy. Unbinned moments are the primary diagnostic here.

### Matched-bin CPU pilot checkpoint (RES077)

`src/gammaforge/validation/delta_comparison.py` integrates smooth candidate densities
over the same physical energy bins, including first moments; it never rescales spectra.
`scripts/validate_delta_emission.py` runs all three scenarios with head-on and
crossed (0.02, -0.015) geometries, two observers and three table configurations.
The new checks are in `tests/test_delta_comparison.py` and
`tests/test_delta_emission_pilot.py`. Both are fast CPU-only test files.

```sh
.venv/bin/python scripts/validate_delta_emission.py --output /tmp/delta-numpy-pilot.json
.venv/bin/python scripts/validate_delta_emission.py --retarget-bins 256 --quadrature-order 32 --output /tmp/delta-numpy-refined.json
```

The saved initial report is `docs/validation/delta-numpy-pilot-2026-09-09.json`:
36 fixed-direction comparisons, 16,000 particles, 64 Stage-0 steps, seed 20260721,
24 common physical energy bins and q8/q16 energy integration. Base shape bins are
(16, 8, 8, 16); shape refinement doubles all four axes with fixed retarget64, while
retarget refinement uses 128 bins with the base shape unchanged. All energy-quadrature
pairs satisfy their provisional refinement thresholds, but this does not establish
table or particle convergence. Worst initial L1 is 81.45%; do not call this a pass.

The higher-retarget/q32-q64 report is
`docs/validation/delta-numpy-refined-pilot-2026-09-09.json` (36 comparisons with
matching source fingerprints). At shape (32,16,16,32)/retarget256 the baseline
head-on on-axis L1 is 1.27%, but baseline off-axis head-on/crossed L1 remains
13.75%/15.43%. Low-field on-axis head-on/crossed L1 is 9.53%/10.11%.
All energy-quadrature pairs satisfy the provisional thresholds; other refinements
remain unresolved. No acceptance threshold was loosened to accommodate these results.

Verification of this packet: 75 focused reference/comparator/pilot/doc tests passed;
the 18 comparator/pilot tests also passed in the minimal environment without CuPy.
The broad default pytest run did not finish and was interrupted; it is not counted
as a suite pass. No actual-CUDA acceptance run was performed for this CPU packet.

Baseline on-axis observations expose two distinct sensitivities:

- Head-on: refining only retarget64 to128 at the base shape reduces L1 from
  22.23% to 6.22%, while total-count error stays around -0.35%.
- Crossed: doubling the shape grid at retarget64 reduces count error from +15.41%
  to +3.91%. Retarget refinement alone does not remove that count discrepancy.

Additional controlled baseline probes held 16,000 particles/64 steps/seed20260721
fixed, with the same 24-bin window and unchanged ahat range/decades:

- Head-on, shape (32,16,16,32), q64: retarget64/128/256/512 gives L1
  17.407%/3.442%/1.271%/1.268%. q16/q32/q64 at fixed retarget barely changes
  these values, localizing this sensitivity to retargeting rather than energy integration.
- Crossed on-axis, shape (32,n,n,32), retarget128, q16/q32: n=8/16/32/64 gives
  count errors +15.423%/+3.895%/+0.969%/+0.246%. This is consistent with angular
  deposition/quadrature error decreasing under refinement, not evidence for a new
  production normalization correction. These supplemental probes are not release gates.

Next bounded packet: expose selected grid refinements in the pilot so the controlled
angular/retarget probes can be run directly, then fix reporting windows and measure
particle/seed and Stage-0 convergence before actual-CUDA acceptance. Do not replace
production Doppler/flux formulas or author-owned defaults based on this diagnostic.

Read GRAND_PLAN §§4.5, 7, 9.3, PROGRESS, RES060/061/070/072/073, DER005/006/009,
and the current manuscript before coding. Physics authority is
/home/alexander/Work/Papers/2026/Compton-Numerics/xigma.tex, especially Eq. udef,
Eq. wR, Eq. xsec, and the small-angle reduction. At preparation time Eq. xsec
already typesets the corrected 3/(2 pi) normalization; do not reapply the historical
normalization repair described in older comments.

- `src/gammaforge/validation/references/delta.py`: `resonance_spectrum` already
  bins resonances and returns density per normalized-energy bin at one observer
  direction. It imports production `polarization_factor`; that must not remain the
  oracle for the new independent-projection checks.
- `src/gammaforge/engines/xigma/stages.py`: Stage-0 samples, table deposition,
  retargeting, and NumPy angular quadrature. Stage 0 uses one nominal-axis flux
  factor even though trajectories and polarization use individual directions.
- `src/gammaforge/engines/xigma/collision.py`: physical-energy conversion applies
  the crossing factor once and divides the density by the corresponding Jacobian.
- `src/gammaforge/validation/run.py`: current identities compare head-on scalar
  integrals; production reporting explicitly retains angular/emission blockers.
- `src/gammaforge/validation/metrics.py`: multidimensional `compare_slices` returns
  only an integrated-yield comparison, not a shape test. Its 1D shape path
  interpolates point values; do not use it as a conservative histogram comparator.
- Tests: `test_stage0_delta.py`, `test_stage1_stage2.py`,
  `test_xigma_gpu_polarization.py`, and `test_cupy_release_gate.py` under `tests/`.

Graphify helped locate this dependency chain; verify current source rather than
cached semantic line numbers. No old-repository golden regeneration is needed.

## 1. Pin the reference contract before implementation

For each Stage-0 macroparticle i and small-angle observer (tx, ty), define

    r_i^2 = (theta_e,x - tx)^2 + (theta_e,y - ty)^2
    s_i = gamma_i^2 / (1 + ahat_i + gamma_i^2 r_i^2)
    C = (1 + cos(theta_xz) cos(theta_yz)) / 2
    E_i = 4 E_laser C s_i
    W_i = [3/(2 pi)] L_i P_i gamma_i^2 / (1 + gamma_i^2 r_i^2)^2

This is the existing reduced model's nominal-axis, ultrarelativistic convention
(DER005/009), not an exact arbitrary-velocity resonance formula. L_i already contains
absolute macroparticle weight, flux, Thomson cross section and trajectory integration.
Do not multiply those factors again. Histogram W_i directly at E_i, then divide by
physical energy-bin widths. No gamma-collapse Jacobian belongs in particle binning;
the Gamma^5 Jacobian belongs only to the table method. Never normalize to Xigma's
yield after binning.

Evaluate P_i independently from manuscript Eq. udef using explicit vectors:

    v_i = sqrt(1 - gamma_i^-2) (theta_e,x, theta_e,y, 1) / norm
    n = (tx, ty, 1) / norm
    u_j = n cross [(n - v_i) cross e_j] / (1 - v_i dot n)
    P_i = (|u_0|^2 + ellipticity^2 |u_1|^2) / (1 + ellipticity^2)

Construct raw laser axes with Ry(theta_xz) Rx(theta_yz) and psi_pol. Under DER012,
project the major axis perpendicular to the unit electron direction and normalize;
construct its orthogonal minor partner by a cross product, orienting it toward the
raw minor axis. Use these local transverse e_j in the radiation vector above.
Do not call production axis-rotation, polarization, Stokes or sampler helpers in
the reference. Use extended precision for this deliberately direct evaluation and
prove its precision at gamma 10000; platforms without adequate extended precision
must report that limitation, not silently use a cancellation-prone float64 oracle.
Both ellipticity signs have the same intensity; Q/U/V are outside this task.

### Separate approximation audit / author checkpoint

The manuscript's general Eq. wR instead gives

    F_i = 1 - v_i dot n0
    E_i,paper = 2 E_laser gamma_i^2 F_i / (1 + ahat_i + gamma_i^2 r_i^2)

where n0 is the laser propagation direction. Production uses F0 = 2 C rather than
F_i; DER005's simplified expression assumes a nominal-axis electron and beta -> 1.
The relative resonance shift is therefore F_i/F0 - 1. Record this independently
for finite-beta and divergent/crossed test particles before calling a comparison
general arbitrary-angle validation. Shared Stage-0 L_i also uses F0, so changing
only delta's frequency cannot validate the corresponding flux correction.

Keep nominal-model numerical parity and this general-formula diagnostic distinct.
Do not hide the difference with broad histogram bins or tolerance inflation. If the
requested regime requires the general per-particle factor, stop for author resolution
before implementing a production correction or issuing a scientific pass. This
preparation does not settle that physics choice. Crossing near co-propagation, where
F0 tends to zero, is outside the initial matrix and cannot inherit a pass.

## 2. Compare the same measure

Use common physical energy edges and common observer nodes. Delta returns bin-average
density, whereas NumPy/CuPy return smooth point samples. Integrate each candidate
over every energy bin with refined quadrature; divide by the same bin width only
for plotting. Compare bin masses, not center values. Do not interpolate a histogram
as though it were a smooth density.

At first, compare spectra at fixed observer directions, then integrate both with
the same refined angular quadrature over the same finite rectangular aperture.
For a persisted histogram cube, supply energy widths and angular cell widths via
`PhasespaceSlice`; its values must actually be cell-average densities. A pointwise
angular comparison alone must not pretend to integrate a one-node smooth axis.

Measure absolute finite-window yield, mass-normalized L1 of bin masses, energy
centroid, support/tail mass, and angular projections. Retain per-direction diagnostics
so opposite angular errors cannot cancel. Test spurious support where delta is zero;
do not simply mask all reference-zero bins. For ideal line spectra, use integrated
bin masses/CDFs and a stated bin-width bound for edge location, not pointwise ratios.
Zero/zero directions may be valid but cannot make an all-zero release case pass.

Track energy underflow/overflow before histogramming. Refine/widen the aperture in
a separate capture study; delta's current `captured_fraction` is a head-on disc
formula, not a correction for a crossed, divergent rectangular aperture. Do not
rescale a finite-window count to the all-angle Stage-0 yield using that formula.

## 3. Staged test matrix and convergence

1. Hand-constructed particles: one resonance; unequal weights; nonuniform energy
   bins; left/right edge semantics; empty/dark input; finite/nonnegative values;
   linear and circular head-on limits; signed ellipticity; independent basis tests.
2. Smooth synthetic populations: finite gamma spread, flat/nonuniform ahat,
   divergent and off-axis electrons. Compare particle binning with successively
   finer CIC tables and retarget grids. Keep input particles fixed within each
   table/sampler comparison; never reconstruct the reference from H itself.
3. Real Gaussian Stage-0 samples: iterate `SCENARIOS`, then derive a bounded set of
   crossing variants using both signs and both planes: head-on; (0.02, 0);
   (0, -0.015); (0.02, -0.015); and the prior stress case (0.3, 0.2) rad.
   Cover ellipticity 0, +/-0.4, +/-1 and a nonzero psi_pol using a selected matrix,
   not an expensive full Cartesian product. Include gamma near 2000 and 10000,
   on/off-axis observers, and thin/wide apertures within the small-angle regime.
4. Public-engine physical-energy checks: validate axis order (energy, x, y), density
   conversion, and once-only crossing scaling. Keep these distinct from pure
   normalized-coordinate kernel comparisons.

Refine independent error sources separately: Stage-0 time sampling; particle
population/seed stability; gamma/angular table bins; shape and retarget ahat bins;
candidate energy-bin quadrature; angular aperture quadrature; CuPy rings and samples.
Particle counts such as 16k/64k/256k are pilot budgets, not acceptance evidence.
Do not assume the existing 4000-particle GPU matrix resolves a noisy delta spectrum.
Do not change the author-owned ahat_decades default while refining the reference.

Provisional outer numerical budgets are 3% yield, 5% integrated L1, 1% centroid,
matching the existing CuPy gate for comparability, not a measured delta guarantee.
Require reference-integration changes below one third of each budget on the same
reporting bins before accepting a case. Report table and particle refinements
separately; use paired seed/population studies for finite-sample uncertainty.
If a pilot cannot resolve a reference within the run budget, report inconclusive
rather than loosen limits. Fix final tolerances and reporting windows before the
acceptance run. These limits do not apply to exact histogram bookkeeping tests.

## 4. Bounded implementation packets (cheaper agents)

Root owns physics review, integration, final report and promotion claims. Suggested
packets, each saved to a file checkpoint and independently tested:

- A: initial independent reference and unit tests are implemented (RES076).
  Extend only for missing RES074 matrix coverage; preserve the legacy delta API.
- B: matched-bin comparator and convergence runner. Own a new validation module and
  its tests; coordinate the reference API with A. No engine edits or generic metric
  framework. Include injected-error tests for wrong normalization, shifted energies,
  missing polarization, lost bins, nonconverged reference and unavailable CUDA.
- C, after A/B freeze: standalone report command and opt-in runner wiring, plus
  tests. Do not modify another agent's live implementation files. Report inputs,
  equations/conventions, all shared quantities, seeds, grids, refinements, runtimes,
  environment, source fingerprints and individual failures in strict JSON.

The reference now lives in `src/gammaforge/validation/references/delta_emission.py`.
The comparator and `scripts/validate_delta_emission.py` implement the CPU pilot;
the full acceptance runner and CUDA integration remain to be implemented.
Keep the default CPU suite fast; an explicitly requested GPU gate must fail rather
than pass through CUDA skips. Run expensive GPU acceptance jobs sequentially.

## 5. Definition of done and remaining limits

Require focused CPU tests, real-CUDA tests, full pytest, unchanged headless alpha
checks, and a saved converged delta/NumPy/CuPy report. A separate proposed runner
selector can execute this matrix without adding Kascade to alpha. Remove only the
specific unwired-angular coverage blocker whose checks actually execute; retain
the remaining four-method and model-validity limitations. Do not equate one green
matrix with validation of all crossing angles or automatically remove experimental.

Delta can independently validate resonance binning versus the reduced table
integration and its polarization implementation. It still shares Stage 0, ahat,
input sampling, the delta-line approximation and the manuscript's radiation law.
Validation of those shared assumptions requires separate evidence. Any paper/code
discrepancy is blocking, per GRAND_PLAN P14, not an invitation to fit a correction.
