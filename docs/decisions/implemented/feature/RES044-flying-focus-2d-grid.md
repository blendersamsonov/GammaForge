# RES044 — a flying focus is always evaluated on the 2D `(z, ct)` grid; the 1D shortcut exists but is not shipped

Status: implemented
Class: feature

## Problem

A flying focus makes the spot-size coordinate `u_spot = u + beta_ff * ct` time-dependent,
so the widths depend on two independent linear functionals of `(x, y, z, ct)` rather than
one, and the time integration §A.3 relies on is gone — `overlap_yield` needs a path for
`beta_ff != 0` instead of raising.

## Decision

`engines.analytical.formulas.overlap_yield` accepts `beta_ff != 0` instead of raising,
routing it to `engines.analytical.formulas._reduced_integral_flying_focus`, which
quadratures `(z, ct)` and integrates `(x, y)` analytically. There is **no** 1D fast path
for a flying focus, even though one exists and is derived in DER002 §B.4.
`overlap_transverse_profile` raises for `beta_ff != 0`; `overlap_time_profile` handles it.

Counting the functionals a flying focus and a crossing angle each introduce is the useful
result: two of four dimensions stay Gaussian, so **an exact treatment of a crossing angle
and an arbitrary flying-focus velocity together is still only 2D** — the two effects each
contribute one width argument and having both does not add a third.

Grid construction is the load-bearing implementation detail, not an afterthought: the `(z,
ct)` Gaussian is nearly degenerate — the collision lives on a thin diagonal ridge — so a
grid sized from the marginals under-resolves it and came out 3.8% low on a short bunch
before the fix. Nodes go on the principal axes, with counts raised until each step advances
`u_spot` by less than an eighth of a Rayleigh range.

Two physics results fall out that nothing in the implementation encodes, and both are
pinned as tests: `beta_ff = 1` maximizes the yield (2.8x over no flying focus on a 30 um
bunch) because the focal plane then co-moves with the bunch; and for a short bunch the
yield is invariant under `beta_ff -> 1 / beta_ff`, because the spot depends on `(beta_ff -
1) / (beta_ff + 1)` — odd under that map — while the width depends on its square.

**Not cross-checked against the author's own derivation.** The author has previously
derived the head-on synchronized counter-propagating case; those expressions were not
available here, so this is validated against the brute-force Monte Carlo only (1e-3 at
`beta_ff` in -0.5, 0.5, 1, 2, with and without a crossing angle). The `(1 + beta_ff)`
Rayleigh stretch in `io.laser.GaussianParaxialLaser.rayleigh_x`, which the reciprocal
symmetry depends on, was queried and confirmed by the author as a paraxial Maxwell result
rather than a convention — so that symmetry is physical.

## Alternatives considered

**Ship the 1D approximation with a validity guard on `beta_ff * sigma_ez / z_R`.** A guard
that silently switches between a 2 ms path and a 45 ms path based on a derived quantity
makes the cost unpredictable and the accuracy scenario-dependent, for a saving on the one
tier that is already opt-in.

**Keep refusing `beta_ff`.** The refusal was correct when nothing covered it; keeping it
once a validated path exists would be the same mistake RES041 corrected for the crossing
angle.

**Quadrature `(z, u_spot)` for full generality with a crossing angle.** Exact rather than
reusing §A.7's measured 1.9e-4 transverse approximation, but it needs a basis for the
complementary plane and a Jacobian, for an error already an order of magnitude below the
Monte Carlo that validates it. Recorded in §B.2 as the route if that changes.

## Rationale

The 1D shortcut is deliberately withheld. Freezing the widths at the stationary point of
the time integral gives a linear `u_spot ~ kappa z` and reuses §A's machinery unchanged,
and on a short bunch it is accurate to 1e-5. But unlike the crossing-angle approximation —
where the dropped term entered an even, slowly varying prefactor and cancelled to first
order — a flying focus exists precisely to correlate the width with time, so the error is
first order: **34% at beta_ff = 1 on the baseline scenario.** The controlling parameter is
`beta_ff * sigma_ez / z_R`, which is 2.5 * beta_ff for the baseline and 0.025 * beta_ff for
a 30 um bunch. An approximation that is excellent in the regime a flying focus is *for* and
useless just outside it is too sharp a knife to expose as a default.
