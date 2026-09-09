# RES060 — Polarization uses each particle's field-free lab velocity

Status: implemented
Class: bug-fix

Basis update: RES078/DER012 replaces the unprojected radiation basis below with a
local transverse dipole basis. The field-free velocity and common lab-frame decision
remain in force; the original formula-specific checks are historical.

## Problem

The manuscript's Eq. `udef` defines the polarization vectors from the field-free electron
velocity `v`, the observation direction `n`, and the laser polarization basis `e_i` in one
laboratory frame. `polarization_factor` accepted sampled electron angles but ignored them:
it fixed `v` to the bunch axis and formed `1 - v.n` from the absolute observer angle.
The vectorized Stage-2 path had no electron-direction inputs, and Delta passed relative
angles into the ignored slots. A divergent bunch could therefore produce an apparently
consistent xigma/Delta result while both paths evaluated a different geometry from the
manuscript.

## Decision

The Stage-2 polarization projection evaluates, for every electron or table cell,
`v_e = beta (theta_x,e, theta_y,e, 1) / sqrt(1 + theta_x,e**2 + theta_y,e**2)`, with
`beta = sqrt(1 - gamma**-2)`. It evaluates the observer direction from its requested
lab-frame angles, retains the laser basis rotated once by its configured crossing angle,
and uses the exact `1 - v_e.n` in Eq. `udef`'s dot-product expansion. Delta supplies its
sampled particle angles to the same projection.

The author approved this lab-frame convention on 2026-09-06. It rejects a per-particle
particle-aligned frame for this implementation: such a frame would require transforming
the observer and both laser vectors for every particle, which the engine does not do.

## Alternatives considered

**Keep `v = beta z_hat` and treat sampled angles as relative observer angles.** Rejected:
Eq. `wR` and Eq. `smallangle` distinguish the particle direction from the observer, and
this made the function's electron-angle parameters false inputs.

**Rotate into a particle-aligned frame for every sample.** Rejected by the approved
convention. Rotating only the velocity would mix frames; rotating all vectors would be a
different implementation with unnecessary per-particle transformations.

**Continue using the ultrarelativistic approximation for `1 - v.n`.** Rejected because the
exact normalized lab vectors are inexpensive and are required once electron and observer
angles differ.

## Rationale

The direct Eq. `udef` calculation in `tests/test_stage0_delta.py` is independent of the
production dot-product expansion. It checks a tilted electron, off-axis observer, rotated
basis, and ellipticity, and the Delta regression checks that particle angles reach the
reference path. This is a code/formula check, not independent arbitrary-angle emission
physics validation.

## Consequences

Angle-resolved xigma and Delta now respond to electron divergence through the polarization
projection. Existing head-on checks continue to apply to the collinear limit. Production
validation still reports its angular and arbitrary-angle coverage blockers: the current
histogram integration has no measure contract, and there is no independent emission method
for a scientific non-head-on comparison.
