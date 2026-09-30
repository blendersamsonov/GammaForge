# RES093 — Uniform raw-`ahat` production retarget grid

Status: implemented
Type: bug-fix

## Problem

RES032's production grid concentrated bins near large `ahat`. DER021 shows that the
relative resonance and inverse-root sensitivity to an absolute `ahat` error is greatest
at the low end instead. The first production bin contained the full populated range for
some scenario-bank cases, biasing the spectral centroid.

## Decision

This partly supersedes RES032's target-spacing law. Stage 1 still deposits one
peak-independent `ShapeTable`, and Stage 1.5 still conservatively transfers its mass and
all three moment channels for each requested peak intensity. The ordinary Stage-1.5
target edges are uniformly spaced in raw `ahat` between `ahat_min` and `ahat_max`
(DER021). The optional sub-floor catch bin and one-bin linear mode keep their existing
evaluation at zero. Unreachable trailing bins are still pruned.

The production `ahat_decades` knob is retired. The default `n_bins_ahat` is 256, chosen
from the direct-particle convergence scan in
`docs/validation/uniform-ahat-retarget-2026-09-30.md`. The table and Stage 2 retain
per-bin width arrays so their integration measure also handles the floor bin and
nonuniform synthetic tables.

## Alternatives considered

Retune `ahat_decades`: this would retain a grid whose concentration is contrary to
DER021's sensitivity result.

Shrink the ordinary interval to occupied support or introduce an adaptive coordinate:
either changes the numerical policy beyond this issue and needs separate evidence.

Keep 32 uniform bins: the low-`a0` direct-particle comparison remains visibly coarse.

## Rationale

Uniform raw-`ahat` spacing makes `n_bins_ahat` the one production resolution control and
keeps the reusable table observer independent, as DER015 requires. Increasing the bin
count improves the low-`a0` spectral centroid and integrated L1 error in the measured
scenario bank, while trailing-bin pruning limits allocation to reached bins.

## Consequences

The finest retained grid can use more memory for intense or broad-support scenarios.
Fixed-direction direct-particle spectra still include Stage-1 spatial/angular
discretization and energy-bin errors; this decision does not close the broader
arbitrary-angle scientific acceptance work.
