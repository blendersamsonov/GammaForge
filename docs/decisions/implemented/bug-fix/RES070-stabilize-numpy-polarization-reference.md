# RES070 — Stabilize NumPy polarization reference

Status: implemented
Class: bug-fix

## Problem

NumPy's Stage-2 polarization projection evaluated an expanded dot-product expression
derived in DER006. When the electron and observer rays were collinear or nearly
collinear and the laser basis had a longitudinal component (e.g. under laser crossing
angles), the expanded formula subtracted large cancelling terms in float64.
At gamma 10000 and crossing 0.3 rad, the exact collinear factor is
*cos(0.3)^2* (approx 0.9126678), but NumPy returned 1.3886977, losing eight digits of
precision.

While CuPy's implementation in RES069 evaluated the manuscript's Eq. udef vectors
directly and passed this limit, NumPy remained on the expanded expression.

## Decision

Port the stable vector evaluation of manuscript Eq. udef from RES069 to NumPy in
`polarization_factor_vectorized` and `polarization_factor`:

1. Evaluate unit vectors u (electron) and n (observer), and their difference Delta = n - u
   using factored differences of transverse slopes to avoid cancellation in the longitudinal
   component.
2. Evaluate the relativistic denominator stably via *d = 1 - beta*(u.n) = delta + 0.5*beta*|Delta|^2*,
   where *delta = 1 / [gamma^2 * (1 + beta)]*.
3. Compute the vector *q = (Delta + delta*u) / d = (n - v) / (1 - v.n)* and form
   *Ui = q * (n.ei) - ei* directly.
4. Weight the squared norms *|U0|^2* and *|U1|^2* by the elliptical polarization weights
   *xi00 = 1 / (1 + epsilon^2)* and *xi11 = epsilon^2 / (1 + epsilon^2)*.

This uses the same algebraic evaluation as CuPy without altering the lab-frame physics,
normalization, or Stage-0 overlap yields.

## Alternatives considered

- Retaining the expanded formula with higher precision: rejected because `numpy.longdouble`
  is platform-dependent (varying on non-x86 architectures) and does not eliminate the
  catastrophic cancellation.
- Clipping or special-casing the collinear limit: rejected because the precision loss
  occurs across a continuum of nearly collinear rays, not only at exact zero angle.

## Rationale

Direct tests against exact limits and the independent extended-precision Eq. udef
reference confirm that the vector evaluation reproduces *cos(alpha)^2* to machine
precision (zero difference in float64 at gamma 10000 and crossing 0.3 rad) and matches
extended-precision results across off-axis angles, crossing angles, and polarization states
to relative errors of order 1e-13.

## Consequences

NumPy's `polarization_factor` and `polarization_factor_vectorized` are numerically stable at
arbitrarily high gamma. Existing Stage-0 yields and tests remain passing. Independent
arbitrary-angle emission validation remains open as tracked in `PROGRESS.md`.
