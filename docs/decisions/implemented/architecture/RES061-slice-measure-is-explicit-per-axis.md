# RES061 — Slice integration measure is explicit per axis

Status: implemented
Class: architecture

## Problem

`PhasespaceSlice` stored only coordinates and a density array. It therefore integrated every
slice with trapezoids over its coordinates. That is correct for smooth point samples but
not for Kascade histograms: their density is counts divided by cell volume, and trapezoids
over bin centres discard half of the first and last cell. A one-bin histogram had a known
width but was rejected because the coordinate array alone could not express it.

## Decision

`PhasespaceSlice.widths` optionally maps an axis to positive cell widths. An axis with
widths is a histogram axis and integrates by its weighted cell sum; an axis without widths
is a smooth point-sampled axis and retains trapezoidal quadrature. Coordinates remain the
plotting centres in both cases. Projections preserve widths for retained axes and apply the
declared rule to eliminated axes. Kascade attaches widths to every histogram slice; HDF5
persists them in an optional `widths` group and treats legacy files without it as smooth
point samples.

## Alternatives considered

**Infer widths from centre spacing.** Rejected: it cannot recover endpoint widths,
nonuniform-bin boundaries, or a one-bin histogram without inventing a convention.

**Change every slice to midpoint quadrature.** Rejected: analytical and xigma produce
smooth samples whose documented trapezoidal convergence would silently change.

**Create separate sampled and binned result class hierarchies.** Rejected: the one slice
contract remains useful; a per-axis measure is the minimal data needed to state its
integration semantics.

## Rationale

The measure follows the producer rather than the ndarray shape. It keeps density units and
plotting coordinates unchanged, permits mixed-axis projections, and lets an integration
identity state exactly which cells it covers. Validation rejects unordered coordinates and
invalid widths at construction so malformed results cannot reach plotting or persistence.

## Consequences

Kascade histogram masses integrate to the captured photon weights, including edge and
one-bin cells. Smooth analytical and xigma slices retain their existing trapezoidal
behavior. Consumers loading a legacy HDF5 result receive explicit point-sample semantics;
they do not receive guessed histogram edges.
