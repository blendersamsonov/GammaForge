# RES021 — The laser's active region is a cone, not a cylinder

Status: implemented
Class: bug-fix
Archived: 2026-08-10

## Problem

`ActiveRegion`'s contract is to be over-inclusive — the prefilter is a pure optimization
and must never discard a particle that would have contributed (§3.2) — but the Phase-2
harness found a real defect: a pulse diverges, so a bunch longer than the Rayleigh range
meets it far from focus, where the spot is many times larger than the old cylinder's fixed
radius, and particles the expanded pulse still reaches were being thrown away. In the probe
configuration a discarded macroparticle saw `a0` **six times** the threshold meant to bound
it.

## Decision

`ActiveRegion` carries a `radius_slope` alongside its `radius`, and `radius_at` gives
`radius + radius_slope * |u|`. `GaussianParaxialLaser.active_region` derives both by
linearizing the spot hyperbola, `s(u) = sigma sqrt(1 + ((u - z_f)/z_R)^2) <= sigma (1 +
(|u| + |z_f|)/z_R)`. `overlap_time_window` evaluates that cone at the widest point of each
particle's own longitudinal window, so the transverse test stays a single closed-form
quadratic.

## Alternatives considered

**Keep the cylinder and give `active_region` an extra argument for the longitudinal span it
must be valid over.** Works but changes the `LaserField` protocol for every implementer
(P15) and leaves a default that is wrong for anyone who forgets the argument — a footgun in
a safety mechanism.

**Keep the cylinder and size its radius from the largest spot the bunch can ever meet.**
Needs no protocol change but sizes every particle's test by the worst particle in the
bunch, which costs most of the filter's value on exactly the long-bunch case that motivated
the fix.

**Solve the cone inequality in `t` exactly.** The tight answer, but a cone inequality in
`t` is not a single band — the parabola can open downward, making the solution set a union
of two rays — and the chosen linearized cone is already conservative, already closed-form,
and tight in the only regime that matters.

## Rationale

This fixes a real defect, found by the Phase-2 harness rather than by review. The
pre-fix behaviour is now guarded twice: at the laser level
(`test_active_region_is_still_conservative_far_from_focus`) and by the harness check that
found it (`check_prefilter_discards_only_dark_particles`), which samples `a0` along every
discarded trajectory and so depends on none of the geometry it is testing.
