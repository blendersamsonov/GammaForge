# RES030 — `Collision`'s cache is per-instance memoization only; no cross-call staleness detection

Status: implemented
Type: architecture

## Problem

§5's GUI grey-out/cheap-requery model eventually needs a live object that survives
repeated edits and decides, per edit, which cached stage to keep — but that consumer is
Phase 6 (or a scan helper), not Phase 3a, and `XigmaEngine.run()` needs a caching strategy
now for sharing Stage 0/1 work across the several outputs one `run()` call requests.

## Decision

`Collision.build_overlap()`/`_table()` memoize on `self` — one `Collision` is built from
one fixed `InteractionParameters` + xigma `Parameters`, and calling either method twice
returns the cached value. There is no hash-based "did the caller's *new* interaction differ
only in field X" detection across *different* `Collision` instances.

## Alternatives considered

**Hash-key every stage's inputs now, so `Collision` can detect "only pulse energy changed"
across instances.** Requires a policy for constructing that key from an arbitrary
`LaserField` (P15) that the protocol does not provide, has no test or caller to justify it
in this phase, and would very likely need reworking once Phase 6 defines what the GUI
actually keeps alive across a Calculate.

## Rationale

Building the cross-call detector now with nothing to drive it is the speculative machinery
P6 rejects. What *is* needed now — sharing Stage 0/1 work across the several outputs one
`run()` call requests — is exactly what per-instance memoization gives, and
`XigmaEngine.run()` builds one `Collision` per call, matching P3's "opaque by contract."

## Consequences

`XigmaEngine.recompute_costs` (`engine.py`) declares only `n_e` (handled at the `io`
level, `InteractionParameters.with_charge`/`Results.scaled`, §3.5 — no engine run at all)
as cheap. Collimation-window and pulse-energy cheap paths from §5's illustrative table are
not claimed: they are true only if the *caller* keeps reusing one `Collision`, which
nothing in this phase does yet. Everything else defaults `FULL_RERUN` (`base.py`'s
documented default), which is the honest statement of what Phase 3a actually wired.
