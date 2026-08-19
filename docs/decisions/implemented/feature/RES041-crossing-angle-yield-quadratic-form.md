# RES041 — the crossing angle is covered for the yield, by quadratic form; the spectrum stays head-on and says so

Status: implemented
Class: feature

## Problem

RES039 deferred the crossing angle to `GRAND_PLAN.md` §9.3, conflating two different things:
§9.3's open item is the polarization structure of the **emission kernel** — what spectrum
emerges at an angle — while the **overlap geometry** is an independent and entirely
solvable Gaussian problem. DER005's own §2.1/§2.2 already record that the
relative-velocity factor and the resonance frequency are general in the paper, so the
yield was never actually blocked.

## Decision

`engines.analytical.formulas.overlap_yield` accepts a crossing angle (`theta_xz`/
`theta_yz`) instead of raising, implemented by rewriting the overlap integral as a
quadratic form (`engines.analytical.formulas._overlap_quadratic_form`) rather than by
adding a second code path. `AnalyticalEngine` reports on `Results.model_specific
["warnings"]` when it fills `SPECTRUM` under a nonzero crossing angle. This supersedes
RES039's crossing-angle refusal; the `beta_ff` refusal stands.

Writing the exponent as `r^T M r / 2` and eliminating time by `M' = M - g g^T / h` makes
the crossing angle almost free: the transverse integrals become a 2x2 determinant and a
Schur complement, and head-on falls out as an identity (`S = (1+beta_0)^2/D^2`, `sqrt(det
A) sigma_ex sigma_ey s1 s2 = sqrt(det(C_e + C_l))`). One expression now covers every
geometry, which is why no head-on test changed when this landed.

One approximation is unavoidable and is stated rather than buried: with a crossing angle
the two hourglasses vary along *different* directions (`z` and `u = k_hat . r`), so an
exact reduction leaves a 2D quadrature. The spot sizes are sampled at `u = (k_hat . zhat)
z`; the exponent stays exact, so the entire crossing-angle suppression is exact. The bound
is `delta / z_R`, which can exceed 1 in a tight-focus/wide-bunch/large-angle corner —
`test_crossing_angle_width_sampling_approximation_is_negligible` measures the yield error
there directly (< 1.3e-4) instead of asserting the bound is small.

## Alternatives considered

**Keep refusing, per RES039.** Would have been correct only if §9.3 blocked the luminosity,
which it does not; it also left `AnalyticalEngine` narrower than `XigmaEngine`, which runs
with a crossing angle and warns.

**Do the exact 2D quadrature.** Removes the one approximation, but the measured error it
removes is < 1.3e-4 in the corner it was built to expose, at the cost of a second
integration dimension and a coordinate system that degenerates as the angle goes to zero —
worse conditioning at the geometry that matters most.

**Emit a Python `warnings.warn`.** No code in `gammaforge` uses that channel; `validate()`
returning strings is the established convention.

**Narrow `supported_outputs` when a crossing angle is present.** `supported_outputs` is a
static class attribute describing the engine, not the scenario; making it scenario-
dependent would break the `Engine` protocol's contract for a documentation problem.

## Rationale

The `SPECTRUM` decision follows from the engine normalizing the slice to this yield: with a
crossing angle the integral is right and the shape is head-on, which looks *more* correct
than it is. Saying so on `Results` matches this repo's convention that validation reports
rather than raises (`io.laser.validate`/`io.bunch.validate` both return `list[str]`), and
puts the statement where the caller already looks.
