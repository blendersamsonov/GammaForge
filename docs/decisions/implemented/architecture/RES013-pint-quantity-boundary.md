# RES013 — Dimensioned fields are pint `Quantity`; bulk arrays declare their units instead

Status: implemented
Class: architecture

## Problem

`GRAND_PLAN.md` §2.1 says "kernels never see pint quantities," which had been over-read as
"the shared layer holds bare floats" — a claim the plan does not make and no entry had
recorded. The shared layer needs a concrete answer for how dimensioned fields are
represented, on `GaussianElectronBeam`/`GaussianParaxialLaser`/`Target` and on `Bunch`.

## Decision

Every dimensioned field of `GaussianElectronBeam`, `GaussianParaxialLaser` and `Target` is
a pint `Quantity`, validated and **converted to canonical CGS** on construction
(`as_canonical_quantity`). Each class declares its canonical units in `UNITS` and unpacks
through `m(name)` once at the top of every numeric routine, so no loop or kernel below
ever sees a `Quantity`. `Bunch` and `PhotonMacroparticles` are **not** wrapped: they carry
raw ndarrays plus a declared `UNITS` mapping and convert via `Bunch.get`.

## Alternatives considered

**(a) Bare CGS floats everywhere, converted by documented convention — what Phase 1
originally built.** Was never actually a decision. P1 removes the *multiplicity* of unit
systems, but not the one conversion each engine must still perform at its own boundary —
kascade is SI internally (§4.4) — and under (a) that conversion is a hand-written literal
whose failure mode is a silent factor of 100, exactly the class of error P1 exists to
prevent. The predecessor had reached the same conclusion independently: its
*PhysicalQuantity* travelled across every model boundary for this reason.

**(b) Quantified numpy arrays for `Bunch` too.** Fails for a reason stronger than
performance. `np.cov`, `np.corrcoef` and `np.linalg.slogdet` have no pint implementation
and `fit_gaussian` needs all three; more fundamentally its 6D covariance is
**dimensionally heterogeneous** — `Sigma[x,x]` is cm², `Sigma[x,gamma]` is cm,
`Sigma[gamma,gamma]` is dimensionless — so it cannot be one quantified array in any units
library, only in a per-element unit matrix. Declared units plus a scale factor is the same
pattern `Axis` already uses for result slices, and it exploits the fact that every
conversion here is a *pure scaling*: there are no offset units anywhere in this project, so
a conversion is one number that a caller can fold into arithmetic it is already doing.
`Bunch.get` returns the array itself, with no copy, when the requested unit is the stored
one — the case for every CGS consumer.

**(c) Floats plus a separate dimensioned "view" object handed to engines.** Leaves the raw
attribute reachable, so the wrong thing stays possible; P12's precedent is to make it
impossible rather than discouraged.
