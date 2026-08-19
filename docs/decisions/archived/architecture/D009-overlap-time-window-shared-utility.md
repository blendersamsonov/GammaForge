# D009 — One `overlap_time_window` serves both the prefilter and the temporal autorange

Status: implemented
Class: architecture
Archived: 2026-08-10

## Problem

Both the bunch prefilter and the target's temporal autorange need to know when each
macroparticle's trajectory overlaps the laser's active region — the predecessor solved
this with a head-on-only closed form duplicated across two consumers.

## Decision

`bunch.py` exposes `overlap_time_window(bunch, laser, threshold)`, returning per-particle
`(t0, t1)` from intersecting each straight-line trajectory with the laser's
`active_region`. `prefilter_bunch` filters on `t0 <= t1`; `target.auto_ranges` derives the
`TEMPORAL_ENVELOPE` window from the same call.

## Alternatives considered

**Porting the predecessor's `laser_overlap_time_window` as a closed-form head-on formula
for the autorange, with the prefilter testing points separately.** Rejected — see
Rationale.

## Rationale

The predecessor's version is head-on-only and takes pre-normalized `k0_las`-scaled
arguments, so a crossing angle would have needed a second implementation in each of two
consumers — four code paths for one geometric question. Intersecting a line with the
region is closed-form anyway (linear in `t` longitudinally, quadratic transversely) and
carries the crossing angle for free, because the angle is already in `ActiveRegion.axis`.
This is the §4.2 "one shared utility replacing three inconsistent implementations"
discipline applied before the duplicates appear.
