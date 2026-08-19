# D052 — *OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION* removed; `COLLIMATED_SPECTRUM` is now the sole `(E, θx, θy)` output kind

Status: implemented
Class: simplification

*(Numbering note: this entry was written as D035 and renumbered to D052 on 2026-08-10 to
clear D035–D051, which the parallel Phase 4 branch `worktree-phase4-analytical-engine`
had already taken.)*

## Problem

*OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION* and `COLLIMATED_SPECTRUM` were produced by
identical code and differed only in which angular window `auto_ranges()` handed them —
raising the question of whether the auto-ranged kind was pulling its weight as a separate
output kind at all.

## Decision

*SPECTRAL_ANGULAR_DISTRIBUTION* is deleted from `OutputKind`, `SLICE_AXES`,
`auto_ranges()`, `_SUPPORTED`/`Collision._fill`, and `XigmaEngine.supported_outputs`
(§3.4). `COLLIMATED_SPECTRUM` — the same `(E, θx, θy)` axes, but windowed by the `Target`'s
collimation half-angles instead of an auto-derived ~1/γ0 radiation cone — is the only 3D
energy-angle output kind left.

## Alternatives considered

**Keep both kinds, since `COLLIMATED_SPECTRUM` alone can't reproduce the exact regression
test that motivated §9.1/D033.** Rejected — the regression is a property of the code path
(`Collision._fill`'s shared branch, `angular_spectrum_from_table`'s kernel), not of which
`OutputKind` requests it; a manually widened `COLLIMATED_SPECTRUM` window exercises the
identical kernel and preserves the guard without carrying a kind nothing else needs.

**Leave D033's text referencing *SPECTRAL_ANGULAR_DISTRIBUTION* unedited.** Decisions are
append-only (this file's own convention) — D033 stays as written, describing what was
true when it was recorded; this entry is the forward pointer.

## Rationale

The two kinds were produced by identical code (`Collision._fill` had one branch for both)
and differed only in which angular window `auto_ranges()` handed them. The auto-ranged
full cone is not a real observable: every experiment measures the collimated emission
through some finite aperture, never the untruncated cone, so
*SPECTRAL_ANGULAR_DISTRIBUTION* was speculative surface area with no consumer it was
uniquely suited for.

## Consequences

One latent bug surfaced while removing it: `make_references._KIND_BY_AXES` excluded
`COLLIMATED_SPECTRUM` from its axis-grouping map specifically to disambiguate it from
*SPECTRAL_ANGULAR_DISTRIBUTION*'s identical axes. With the latter gone, that exclusion
would have left the `(E, θx, θy)` grouping unmapped — fixed by dropping the exclusion now
that axis groupings map 1:1 to `OutputKind`.

`tests/test_xigma_engine.py`'s
`test_the_two_normalization_paths_inside_one_results_object_agree` — the only test
pinning the closed-form-vs-table-kernel normalization identity that D033 closed (the
§9.1 `2 pi` regression guard) — is retargeted onto `COLLIMATED_SPECTRUM` rather than
deleted: it now widens `Target`'s collimation half-angles to `~4.6/gamma0` (the same
order the deleted auto-range used) instead of relying on the baseline scenario's much
narrower default collimation.
