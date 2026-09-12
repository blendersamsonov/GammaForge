# RES081 — Xigma temporal and spatial source diagnostics

Status: implemented
Class: feature

## Problem

Stage 0 already computes the overlap contributions that sum to photon yield, but
`Collision` previously omitted temporal and spatial outputs. The unfinished worktree
implementation used different time grids per chunk and summed their bins as though they
shared coordinates. It also bypassed requested resolutions and spatial overrides.

## Decision

`integrate_trajectories` accepts optional fixed time and transverse-position edge arrays.
Its `TrajectoryDiagnostics` stores bin-average overlap densities in seconds and
centimetres. Every chunk uses the same edges, and completed chunk histograms are reduced
only after successful integration, so an OOM retry cannot double count a partial result.
Time bins accumulate weights directly rather than differencing cumulative sums, preserving
small tail bins in the presence of a large central photon population.

`Collision.run` derives grids from `OutputRequest` and `auto_ranges`, requests both
histograms in one Stage-0 call, and returns explicit-width `PhasespaceSlice` outputs.
The temporal coordinate is laboratory emission time; the spatial coordinates are the
actual transverse positions along the sampled trajectories. Both outputs integrate over
photon energy and emission direction. This is overlap bookkeeping, not a replacement for
the trajectory-averaged intensity or a division into independently radiating segments.

A new requested diagnostic grid requires another Stage-0 integration. Identical grids
reuse the cached samples. Diagnostic arrays stored in the facade are read-only.

Clipped photons are not redistributed. Result metadata records captured/outside fractions,
which remain valid under charge rescaling, and warns when a window loses photons.
For a bunch with no overlap, `auto_ranges` uses the laser active region's longitudinal
passage through the laboratory origin as a finite display interval for a zero temporal
histogram. The xigma result warns that this is a display interval.

## Alternatives considered

- Separate engine bin-count knobs: rejected because `OutputRequest` already owns output
  resolution and permits advanced spatial ranges.
- Rebin cached coarse histograms: rejected because a finer request cannot recover the
  lost coordinates or exact requested-bin masses.
- Rescale a clipped histogram to total yield: rejected because it hides source-window
  loss and changes the physical density.
- Fixed numerical defaults for an empty time axis: rejected in favour of a laser-derived
  display interval, with no claim that emission occurred there.
- Store captured photon counts in metadata: rejected because generic charge rescaling
  would leave duplicated absolute counts stale; fractions are invariant.

## Rationale

The requested outputs follow the same field samples and weights as total yield, while
retaining a well-defined integration measure for single-bin and nonuniform-bin diagnostics.
The engine keeps one existing facade and the shared chunk/OOM utility; no new cache or
capability framework is needed.

## Consequences

Diagnostics allocate additional temporary arrays only when requested. Changing their
grid is a full Stage-0 computation, not a query-only operation. Spatial autoranging keeps
the shared descriptive rule; a displaced source can require the existing manual override.
The source-diagnostic tests cover chunking, nonuniform bins, OOM retries, displaced
emission positions, crossed pulse trains, empty interactions, clipping, requested grids,
and charge rescaling.
