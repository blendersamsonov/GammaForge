# RES006 — `Bunch.weight` is a per-particle array

Status: implemented
Class: architecture
Archived: 2026-08-10

## Problem

`GRAND_PLAN.md` §3.2's wording suggests a scalar `weight`, but two existing requirements
need per-particle weights: `.ele` files may carry unequal weights, and the prefilter must
be able to drop particles without renormalizing.

## Decision

`weight` is an ndarray of relative per-particle weights summing to 1 over an unfiltered
bunch, not the scalar `1/n_particles` the plan's §3.2 wording suggests.

## Alternatives considered

**A scalar `weight`, with non-uniform distributions handled by a separate optional array
or rejected outright.** Rejected — see Rationale.

## Rationale

Two existing requirements need an array. `.ele` files may carry unequal weights (§8), and
`prefilter_bunch` must drop particles *without renormalizing* (§3.2) — with a scalar, the
surviving weight would either silently become wrong or need a second field to record the
count it no longer matches. As an array, the prefilter is `weight[mask]` and the sum
falling below 1 is the honest record of what was dropped. The uniform case costs one array
beside six others already there.
