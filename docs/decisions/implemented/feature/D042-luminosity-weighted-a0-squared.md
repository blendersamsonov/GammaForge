# D042 — the width's nonlinearity term uses a luminosity-weighted `<a0^2>`, computed, not approximated

Status: implemented
Class: feature

## Problem

D039 closed non-round beams and foci displacement for the total yield but explicitly not
for the width, whose nonlinearity term still used the pulse's own maximum a0 — so moving
the foci changed the yield correctly while leaving that component frozen.

## Decision

`engines.analytical.formulas.overlap_mean_a0_sq` computes the mean square a0 over the
collision, weighted by the luminosity, and `AnalyticalEngine` passes it to
`estimate_spectrum_width` as its new `a0_sq` argument. That argument **defaults to**
`laser.a0_peak() ** 2`, preserving the old behavior for direct callers.

## Alternatives considered

**Compute `<a0^2>` inside `estimate_spectrum_width`.** Forces every caller to pay two more
overlap integrals, and destroys the predecessor pin as below.

**Use the a0 at the bunch centroid's position.** Cheaper and superficially reasonable, but
it is a point sample of a quantity whose whole difficulty is that it varies across the
collision — it would be right only where the correction is negligible anyway.

**Weight by electron density rather than luminosity.** Counts electrons that never
scatter. The luminosity weight is the rate at which each electron actually produces
photons, which is what a yield-normalized spectrum needs.

## Rationale

The gap turned out not to need a new derivation at all: `a0^2` is exactly proportional to
the *normalized* photon density, so the numerator is the same overlap integral with the
laser density squared, which in the quadratic form is one parameter (`laser_power`) that
doubles every laser term. `docs/DERIVATIONS.md` §A.8.

It is a large correction, not a refinement: at the baseline the bunch samples about 0.35 of
the peak `a0^2`, so the previous value overstated the nonlinear broadening by roughly 3x.
There is also a clean exact limit that makes the quantity interpretable and testable — for
a transversally pointlike bunch with no hourglass the ratio is exactly `1/sqrt(2)`,
*independent of bunch length*, because integrating over both `z` and `t` spans every
relative shift and the bunch convolution factors out. A counter-propagating collision can
never reach the peak: it always scans the pulse's full longitudinal profile.

The default is the load-bearing part of the decision. `estimate_spectrum_width` has a
second job — reproducing the predecessor's worked example, pinned by
`_PREDECESSOR_WIDTH_TOTAL` — and changing what it computes by default would break the pin
that exists precisely to catch that drift, converting a port-fidelity test into a test of
the new physics. The engine opts in explicitly instead.
