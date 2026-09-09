# Delta arbitrary-angle emission validation — implementation handoff

Status: prepared, not implemented. Starting commit: b11117f (CuPy RES072).
Decision: RES074 (proposed). Scope: validation-only direct binning of each electron's
resonant photon energy, comparing with NumPy and CuPy spectral-angular calculations.
No GUI, Kascade optimization, GPU Stokes, or production physics changes.

## Current implementation and authority

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

Construct e_j with the documented Ry(theta_xz) Rx(theta_yz) rotation and psi_pol.
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

- A: delta reference and unit tests. Own the existing delta module and a new focused
  test file. Preserve existing public signatures and normalization regressions.
- B: matched-bin comparator and convergence runner. Own a new validation module and
  its tests; coordinate the reference API with A. No engine edits or generic metric
  framework. Include injected-error tests for wrong normalization, shifted energies,
  missing polarization, lost bins, nonconverged reference and unavailable CUDA.
- C, after A/B freeze: standalone report command and opt-in runner wiring, plus
  tests. Do not modify another agent's live implementation files. Report inputs,
  equations/conventions, all shared quantities, seeds, grids, refinements, runtimes,
  environment, source fingerprints and individual failures in strict JSON.

Suggested new files (not yet present): *src/gammaforge/validation/delta_emission.py*,
*scripts/validate_delta_emission.py*, and focused *tests/test_delta_emission.py*.
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
