# RES078 — Integrate local transverse dipole reference

Status: implemented
Class: bug-fix

## Problem

The author's crossing-polarization branch fixes the unprojected laser-basis radiation
law: its longitudinal component could inflate angular photon counts. The independent
delta reference implemented the same old law, so numerical agreement could not expose
that shared physical error. Earlier reports therefore cannot validate the revised law.

## Decision

Integrate the author-provided local transverse-dipole implementation (DER012) in
NumPy, CuPy and Stokes. Retain RES060's field-free electron directions and common
laboratory frame. Replace only the radiation basis: project and normalize the major
laser axis perpendicular to each electron direction, then construct its oriented
orthogonal minor partner. This is the approved ultrarelativistic approximation;
neither the general finite-beta Doppler nor Stage-0 flux approximation is changed.

Construct the same physical basis independently with extended-precision vector
operations in `src/gammaforge/validation/references/delta_emission.py`, without
production projection helpers. Reject degenerate reference projections explicitly.
Preserve the original observer keywords and custom-basis inputs of
`bunch_stokes_parameters`; the branch's incidental API changes are not adopted.

Replace the branch's prescribed-curve conservation test with quadrature of the
actual production projection. Independently integrate delta line weights for cold
and divergent electrons at gamma 2000 and 10000. Preserve old JSON reports as
historical evidence, and save corrected reports under distinct transverse names.

## Alternatives considered

- Keep the previous delta weights: compares two different physical expressions and
  mislabels their discrepancy as table error.
- Import the production transverse-basis helper into delta: hides common coding errors.
- Validate conservation only by integrating the expected analytic curve: does not
  exercise the changed implementation and cannot detect a missing projection.
- Normalize spectra empirically: masks the photon-conservation defect.
- Change Doppler, flux or ahat defaults at the same time: confounds this isolated fix.

## Rationale

The user's branch supplies the physics correction. Independent implementation,
actual-kernel conservation, Stokes compatibility and real-CUDA checks establish a
bounded integration check without claiming exact finite-gamma physics or universal
convergence. RES060/RES069/RES070 retain their lab-frame and stable-evaluation roles;
their unprojected-basis examples are historical, as noted in those records.

## Consequences

The old cosine-squared collinear weight becomes unity. Divergent head-on weights
can also change because projection is local to each particle. Per-line Doppler
ratios are unchanged, but weighted aggregate Doppler diagnostics must be rerun.
RES074 remains partly implemented: off-axis spectral/table convergence and a
matched-bin delta/CuPy acceptance matrix remain open. CuPy stays experimental.
Current reports and remaining work are in
`docs/handoffs/delta-arbitrary-angle-validation-2026-09-09.md`.
