# RES008 — A `gaussian_charge` pint context beside `light_time`

Status: implemented
Type: architecture
Archived: 2026-08-10

## Problem

`GRAND_PLAN.md` §2.1 already records that pint cannot convert electromagnetic quantities
between SI and CGS-Gaussian, and applies the factor by hand for the one constant that needs
it. The schema needs the same conversion for a user-entered value — a bunch charge in pC.

## Decision

`units.py` registers a second pint context converting between SI charge and statC by the
exact `1 C = c[cm/s] / 10 statC` factor, derived from `C_CGS`. `to_canonical`/
`from_canonical` apply both contexts (`BOUNDARY_CONTEXTS`).

## Alternatives considered

**Special-casing the charge dimensionality inside `to_canonical` with a hand-written
branch.** Rejected — see Rationale.

## Rationale

A named context is the mechanism pint already provides, the one `light_time` uses, and the
one that keeps the exception from firing where nobody asked for it. A branch inside
`to_canonical` would have put unit-system knowledge in a function whose whole job is to not
have any.
