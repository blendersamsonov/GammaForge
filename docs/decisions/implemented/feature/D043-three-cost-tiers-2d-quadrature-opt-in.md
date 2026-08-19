# D043 — three explicit cost tiers, and the exact 2D quadrature is opt-in

Status: implemented
Class: feature

## Problem

§4.3 calls analytical the only real-time engine, and that claim has to stay true of
*something* once the model grows a semi-analytical mode with a genuine 2D quadrature
option (§A.7) — the tiers and their recompute costs need declaring before Phase 6 exists
to consume them, or a live panel risks getting wired to whatever the default happens to be.

## Decision

The overlap integral has three evaluation tiers — `engines.analytical.formulas.
estimate_yield` (~0.01 ms, closed form), `overlap_yield` with `n_quad_u == 1` (~1-2 ms, the
1D path), and `overlap_yield` with `n_quad_u > 1` (~40-800 ms, the exact 2D quadrature of
`docs/DERIVATIONS.md` §A.7). The schema field `n_quad_u` defaults to 1, and
`AnalyticalEngine.recompute_costs` now declares every quadrature knob `FULL_RERUN`.

## Alternatives considered

**Make the exact 2D path the default.** A 20-400x cost increase to correct an error below
2e-3, on the engine whose defining property is being fast enough to be live.

**Drop the 1D path once the 2D exists.** Loses the real-time tier and the head-on identity
that makes the 1D path exact where most collisions actually sit.

**A single `exact: bool` flag.** Hides that the cost and accuracy both depend continuously
on the node count, and gives no way to check convergence — which is how the truncated-span
bug in the first 2D implementation was found.

## Rationale

Declaring the tiers — and the recompute costs — before Phase 6 exists is the point:
otherwise a live panel gets wired to whatever the default happens to be, which is exactly
the `_MAX_LIVE_N_ENERGY_*` hardcap failure mode the plan already rejects. The first two
tiers are real-time at any interaction rate; the third is a deliberate opt-in.

Making the exact mode available rather than merely arguing the 1D path is adequate is what
turns "the approximation is small" into a measured number. It is measured: the two agree
to 1.9e-4 at 20 mrad and 1.6e-3 at 0.4 rad on the worst corner this model has (a 2 um waist
against a 200 um bunch). Worth recording, because it is counterintuitive: the 2D mode
converges *more slowly* than the approximation it checks at **small** crossing angles,
since the widths barely vary along `q1` there and it is re-integrating a direction the 1D
path does analytically. It earns its cost at large angles.
