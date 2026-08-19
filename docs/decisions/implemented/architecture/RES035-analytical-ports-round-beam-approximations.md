# RES035 — analytical ports the predecessor's round-beam, no-displacement approximations unchanged; the growth items stay open

Status: implemented
Class: architecture

## Problem

Phase 4's first landing needs a scope decision: whether to invent generalizations (an
elliptical-beam overlap integral, a foci-displaced a0 term) that neither the paper nor the
predecessor derives, or to port the predecessor's approximations as-is and name the growth
items as open.

## Decision

`engines.analytical.formulas.estimate_yield`/`estimate_spectrum_width` carry over the
predecessor's round-beam (geometric-mean waist) and undisplaced-focus approximations
verbatim. §4.3's "growth items" — foci displacement, non-round-beam total yield,
collimated-spectrum construction — are not attempted in this landing and are named as open
in `PROGRESS.md` and `GRAND_PLAN.md` §11's Phase 4 row rather than the row being marked
closed. `laser.a0_peak()` (the pulse's own maximum, not the a0 at the electron bunch's
actual position) stands in for the predecessor's `pulse.a0_interaction` for the same
reason: it is the input the predecessor itself used, not a new approximation.

## Alternatives considered

**Derive an elliptical/displaced-focus overlap integral now, since it is "just" a
multi-dimensional Gaussian integral.** Plausible in principle, but it is new physics
content this repo's own review discipline (P14) requires be checked against the paper or
built with the author, not authored ad hoc inside an engine port.

**Mark Phase 4 closed and file the growth items as a separate future phase.** §11's own
Scope column already lists them under Phase 4; splitting them into an unlisted future phase
would make the plan doc's own table inaccurate rather than honest about what landed.

## Rationale

Neither the paper nor the predecessor derives an elliptical-beam overlap integral or a
foci-displaced a0 term; inventing one now is exactly the P14c failure mode §9.2/§9.3 (RES034)
already names — a formula the source material does not contain, wired in silently.
