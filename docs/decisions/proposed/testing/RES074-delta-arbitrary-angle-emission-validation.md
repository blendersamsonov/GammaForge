# RES074 — Delta arbitrary-angle emission validation

Status: proposed
Class: testing

## Problem

The CuPy gate in RES072 compares two table integrators, not table integration with
direct particle resonance binning. The existing `resonance_spectrum` avoids the
table but imports the production `polarization_factor`. Current runner comparisons
are head-on scalar integrals, not crossed spectral-angular distributions.

## Proposal

Extend the validation-only delta path with an independently evaluated manuscript
polarization projection and explicit finite-energy-bin comparisons against NumPy
and CuPy. Share `TrajectorySamples` deliberately, never table interpolation,
production polarization helpers, or production energy-conversion helpers on the
reference side. Independently integrate each candidate density over the reference
energy bins; preserve histogram widths (RES061), absolute normalization, and
per-direction spectral information.

Separate two questions: agreement under DER005's existing nominal-axis,
ultrarelativistic crossing approximation, and the error of that approximation
against the general per-particle resonance in manuscript Eq. wR. The latter is
not permission to change Stage 0 or the production resonance formula. Any unresolved
paper/code discrepancy blocks scientific acceptance and requires the author.

The implementation sequence, equations, matrix, and stopping conditions are in
`docs/handoffs/delta-arbitrary-angle-validation-2026-09-09.md`. This extends
RES060/RES061/RES065/RES072; it does not supersede their current contracts or remove
the experimental warning. RES076 implements the independent line reference and
initial Doppler diagnostic. RES077 adds matched-bin comparison and a CPU pilot;
full convergence acceptance and the actual-CUDA matrix remain proposed here.
RES078 incorporates the author's DER012 basis correction in both production and
the independent reference; earlier unprojected-basis reports are historical only.

## Alternatives considered

- Reuse the production polarization helper: inexpensive, but a common error cancels
  between the candidate and reference; retain such tests only as routing checks.
- Compare density at energy-bin centers: compares a smooth point value with a
  finite-bin average and confuses spectral discretization with a kernel defect.
- Compare only total counts or rescale spectra to agree: hides energy shifts,
  angular redistribution, and absolute-normalization errors.
- Replace the production resonance with the general expression immediately: changes
  the reduced table kernel and shared Stage-0 assumptions before their discrepancy
  and intended validity regime have been reviewed.
- Add Kascade, GPU Stokes, or a new engine: unnecessary for this bounded,
  validation-only direct-binning task.

## Acceptance criteria

- Hand-checkable single-particle resonance, histogram mass, nonuniform-bin and
  zero-signal tests; independent polarization checks including signed ellipticity.
- Matched finite-bin mass, integrated L1, centroid, angular redistribution, and
  spectral-support comparisons, without empirical normalization.
- Separate convergence evidence for particle sampling, Stage-0 integration,
  table deposition/retargeting, reference energy/angular integration, and CuPy
  rings/subsampling. Missing or unconverged required checks cannot pass.
- A reproducible actual-CUDA report covering crossed, polarized, off-axis and
  high-gamma cases, with environment, inputs, grids, seeds and source fingerprints.
- Report shared Stage 0 and shared physical approximations explicitly. A passed
  direct-binning gate validates the reduced emission implementation in its tested
  regime, not the underlying radiation model or full pipeline independently.

## Risks

The general manuscript Doppler factor is particle-dependent; current Stage 0 and
physical-energy conversion use a nominal beam-axis approximation. Shared
luminosities cannot test that flux approximation. Sparse histograms, table smoothing,
retargeting bias and truncated apertures can mimic physics errors. Large laser
crossing angle does not authorize large electron/observer angles outside the
small-angle reduction. Domain and tolerance claims require measured evidence.

## Direction-Doppler amendment (2026-09-12)

RES082 implements the adopted beta=1 per-electron direction extension in Stage 0 and
both Stage-2 backends. DER013 records the extended reduction, with author algebra
review still open. The matched-bin pilot now explicitly compares this convention;
the nominal-axis and finite-speed modes remain diagnostic alternatives. The original
proposal above records the boundary before that decision. Renewed numerical evidence
is in `docs/validation/direction-doppler-2026-09-12.md`; it does not close this proposal's
full scientific acceptance requirements.

## Author verification (2026-09-13)

A. Samsonov explicitly confirmed DER013 as verified. Its status now combines author
verification with the existing symbolic and numerical implementation checks. The
author-review thread recorded above is closed; broader arbitrary-angle scientific
acceptance remains open.

## Production integration (2026-09-14)

RES084 wires the independent fixed-direction matched-bin measurement into the
production runner, with provisional agreement, energy-quadrature, angular-table
and retarget refinement gates. Particle/seed, Stage-0, gamma/shape-grid,
angular-aperture and independent CUDA convergence remain explicit coverage gaps.
This proposal's complete scientific acceptance criteria remain open.
