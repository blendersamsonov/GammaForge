# 07 — Review the real LaserField boundary

Read [coordination rules](README.md). Covers A15. **This handoff is diagnosis and
a design proposal, not permission to implement new field physics.** It can run
read-only alongside other work.

## Entry points and evidence

`src/gammaforge/io/laser.py`: `LaserField`, `fit_gaussian_paraxial`, descriptive
Gaussian methods. Trace callers in `io/target.py`, `io/drawing.py`, xigma Stage 0
and Collision, and analytical/kascade adapters. Read P15, GRAND_PLAN §3.3,
RES010 and RES054.

A wrapper that delegates every current protocol method to a Gaussian laser but
is not itself a GaussianParaxialLaser passes runtime protocol conformance and
then fails at `fit_gaussian_paraxial`'s identity-only implementation. More
importantly, engines use that descriptive fit for physical wavelength,
intensity/polarization and conversion factors, not only rough visual metrics.
The plan's promise of a zero-engine-change arbitrary-field swap is therefore
stronger than the current contract demonstrates. Identity-only fitting itself
was deliberately deferred, not an accidental missing implementation.

## Deliverable

1. Reproduce the wrapper failure and trace the exact method/metadata needs of
   each consumer. Distinguish field sampling, normalization/physical invariants,
   approximation-specific assumptions, approximate autoranging, and drawing.
2. Describe what can actually be computed for an arbitrary pulse with today's
   engine approximations. Identify paper/author decisions rather than assigning
   an invented “effective wavelength” or fitted polarization to unsupported fields.
3. Compare the smallest viable interface changes with an honest explicit
   unsupported-field boundary. Show how existing Gaussian behavior stays the
   same and which future implementation would need which metadata.
4. Produce a proposed decision and specific author questions. If the plan needs
   changing, present/update its proposed contract first; do not mark that proposal
   implemented or claim compatibility from `isinstance(..., LaserField)` alone.

## Acceptance and exclusions

The review includes a consumer/requirement table, an executable small reproduction,
the exact missing contracts, and acceptance tests for a future approved change.
Tests must eventually use a distinct conforming object, not only the concrete
Gaussian type. It must distinguish “supported approximately” from “supported
exactly” and reject unsupported physics transparently.

Do not build the generic fitter, FEM bindings, extra-field implementation, plugin
registry, parameter adaptation framework, or a new LaserFittedParams type. Do not
read/modify the sibling FEM project just because it is named in the plan. Coordinate
with 06 on mutable laser ownership and with 01 on physical conventions. A focused
proposal with unresolved scientific questions is the correct completion here.
