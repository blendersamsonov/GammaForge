# RES018 — `engines/base.py` lands in Phase 2, with the `Engine` protocol but no registry

Status: implemented
Type: architecture

## Problem

Phase 2's scope is "runners skeleton", and a skeleton needs a joint: the §4.1 `Engine`
protocol and the §5 `RecomputeCost` enum could land with the first real engine (Phase 3a)
or a phase earlier, and the plan also names an *ENGINES* registry whose timing needs
deciding separately.

## Decision

The §4.1 `Engine` protocol and the §5 `RecomputeCost` enum are written in Phase 2, one
phase before the engines they describe. The *ENGINES* registry named in the same plan
section is *not*: it arrives with the first engine that has something to register.

## Alternatives considered

**(a) Leave `gammaforge.engines` empty until Phase 3a and have `run_engine` accept any
object with a `run` method, checking nothing.** Would have replaced a checked contract
with a duck-typed one at precisely the boundary the plan spends §4.1 pinning down — and
`runtime_checkable` makes the check cost one `isinstance` and produce an error naming what
an engine is missing, rather than an `AttributeError` from inside a run.

**(b) Defer the whole runner layer to Phase 3a and make Phase 2 golden-generation only.**
Would have left the harness untestable: with no engine type there is no stub engine
either, so the chunk/backend/prefilter invariance machinery could only be *written*, never
*exercised*, which is how scaffolding rots.

## Rationale

The registry is a different matter. A lazy optional-dependency import table over zero
engines is exactly the speculative machinery P10 warns about, and it is three lines to add
when xigma exists.

## Amendments

> **2026-09-05 — Deferred engine enumeration implemented (RES058).** The local browser
> UI now obtains concrete calculation engines from `LocalRunner`, using a small
> dictionary rather than a separate global registry. This closes the deferred registry
> question; the Phase 2 protocol and recompute-cost decision remains in force.
