# D022 — The active-region cone is evaluated at the flying-focus coordinate

Status: implemented
Class: bug-fix
Archived: 2026-08-10

## Problem

D021's cone derivation used `s(u)`, treating `beta_ff` as a detail the bounding region need
not model, on the assumption that the `(1 + beta_ff)` factor already in
`rayleigh_x`/`rayleigh_y` accounts for it. It does not — it accounts for it **backwards**.
This is the same defect class as D021, found the same way and one review later: the D021
fix rederived the transverse bound and carried the pre-existing `s(u)` reading forward
without noticing that the spot is not evaluated there.

## Decision

`active_region` derives its cone from `s(v)` where `v = u + beta_ff * ct`, the coordinate
`_local_coordinates` actually evaluates the spot at — not from `s(u)`. Inside the
longitudinal window `|v| <= |1 + beta_ff| |u| + |beta_ff| half_length`, so the slide
multiplies the slope and the drift widens the intercept.

## Alternatives considered

**Keeping the `s(u)` derivation** and treating `beta_ff` as a detail the bounding region
need not model, on the grounds that the `(1 + beta_ff)` factor already in
`rayleigh_x`/`rayleigh_y` accounts for it. Rejected — see Rationale.

## Rationale

The stretched Rayleigh range makes the cone *shallower*, and the omitted slide would have
made it steeper by exactly the same factor. Dropping one of the pair leaves the region
narrower than the pulse it bounds, which is the one direction a conservative bound may not
err in; measured, a flying-focus pulse put 1775 of 4913 above-threshold sample points
outside the region, and the harness check reported a discarded macroparticle at 1.2x its
threshold. With both terms present they cancel and the slope is `beta_ff`-independent —
which is the property
`test_the_flying_focus_cancels_out_of_the_cone_slope_but_not_its_intercept` now pins, since
an assertion that the region merely "gets wider" would pass against the broken version too.
