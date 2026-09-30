# RES010 — `fit_gaussian_paraxial` implements the identity path only

Status: implemented
Type: architecture
Archived: 2026-08-10

## Problem

`fit_gaussian_paraxial` needs to handle at least the case of fitting a
`GaussianParaxialLaser` to itself; whether it should also handle an arbitrary `LaserField`
numerically is an open question with no second implementation to test against yet.

## Decision

`fit_gaussian_paraxial` returns a `GaussianParaxialLaser` input unchanged and raises
`NotImplementedError` for any other `LaserField`.

## Alternatives considered

**A numerical moment-based fit — sample `a0_profile` on a lattice, extract peak, waists
and Rayleigh ranges — written now so the function is total.** Rejected — see Rationale.

## Rationale

No second `LaserField` implementation exists (`Spectral-FEM-Fields` has no Python bindings
yet), so a numerical path written today could only be tested against the analytic laser it
is not for, and would be guessing at a field representation nobody has seen. That is the
speculative abstraction P6 rejects, and RES003's reasoning about empty module files applies:
better for code to exist because something needs it. The identity path is what Phase 1 has
a consumer and an exit criterion for; the numerical path lands with the implementation that
requires it, which will also be able to test it.
