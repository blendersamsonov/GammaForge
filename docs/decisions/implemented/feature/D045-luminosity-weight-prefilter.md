# D045 — the luminosity-weight prefilter is a second, tolerance-based filter, not a replacement for the cone

Status: implemented
Class: feature

## Problem

`prefilter_bunch` drops only particles a geometric bound proves contribute exactly zero,
which is a strong guarantee but, on a mismatched geometry, keeps far more particles than
actually matter — a relevance-ranking filter needs a different contract, not a replacement
for the exact one.

## Decision

`io.bunch.luminosity_weights` computes each macroparticle's expected contribution to the
luminosity in closed form, and `io.bunch.prefilter_by_luminosity` keeps the particles
carrying all but `epsilon` of the total. `io.bunch.prefilter_bunch` is **unchanged** and
remains the default; the two coexist with different contracts.

The weight is closed-form, which is what makes it usable as a filter at all: freezing the
spot sizes makes the trajectory integral Gaussian in `t`, so it is one vectorized pass with
no time stepping — `O(n_particles)`, the same order as sampling the bunch. The frozen
widths are evaluated at each particle's own closest approach by iterating the stationary
point twice, which is ample for a ranking.

## Alternatives considered

**Replace `prefilter_bunch` with this.** Trades a tested exact invariance for a tolerance,
and loses on the common well-matched case where the cone is already optimal.

**Put it in `engines.analytical.formulas` with the rest of the overlap machinery.** That
module states as an invariant that it takes no macroparticle argument anywhere, precisely
so the predecessor's `O(n_particles)` blow-up cannot return; a bunch parameter there would
be the first crack. `io.bunch`, next to `prefilter_bunch`, is where bunch filtering lives.

**Threshold on the weight relative to the maximum rather than on the cumulative sum.**
Simpler, but the quantity a user can reason about is "how much luminosity am I discarding",
which is the cumulative form.

## Rationale

`prefilter_bunch` drops only particles a geometric bound proves contribute exactly zero, so
results are bit-identical with it on or off — an invariance the suite tests. A relevance
ranking necessarily drops particles that contribute *a little*, so it moves the answer.
Those are different guarantees and folding them into one function would quietly destroy
the stronger one; hence a second function, and a contract stated in terms of the
**weights** ("drops particles whose frozen-width weight sums to under `epsilon` of the
total") rather than in terms of the answer, since the induced error is of the same order
but not equal to `epsilon` (measured: 1.2e-3 at `epsilon = 1e-4`).

## Consequences

Whether it is worth using is scenario-dependent, and the measurements say so plainly. On a
well-matched collision there is **no headroom at all** — the cone already keeps 100%, and
nothing should be discarded. It wins exactly where the cone is loose, which is where the
geometry is mismatched or the foci displaced and the cone must widen conservatively: on a
400 um bunch against a 4 um / 1 ps pulse with displaced foci the cone keeps 94% of the
bunch while this keeps **31%** at 1.3e-4 induced error, and 42% at 7.5e-6.
