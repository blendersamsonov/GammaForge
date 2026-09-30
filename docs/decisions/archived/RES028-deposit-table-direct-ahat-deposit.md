# RES028 — `deposit_table` bins directly onto `ahat`; the predecessor's fixed-range `retarget_a0`/`a0_kind` regrid is not ported, but the underlying peak-independence it exploited is — for both `ahat` and `luminosity`

Status: implemented
Type: architecture
Archived: 2026-08-08

**Superseded by RES032** (2026-08-08), within the same session: the physics argument for
*why* a fixed, non-uniform target grid earns its cost (concentrating resolution near
`ahat_max`, where the redshift correction is significant) reverses this entry's "nothing
in this repo needs [the regrid]" conclusion. `deposit_table`/`Table.ahat_edges`-as-direct-
deposit-target, named throughout below, no longer exist — replaced by
`stages.deposit_shape_table` + `stages.retarget_ahat`. Left below as the historical
record of the reasoning that held for the rest of this Phase 3a session, not as a
description of current code.

## Problem

`GRAND_PLAN.md` §4.2 named `retarget_a0` (a0-axis rebin, no re-deposition) as in-scope
for Stage 1: how much of the predecessor's Grid4D/W-matrix conservative regrid mechanism
— which let tables from different peak-a0 runs sit on one fixed target bin range so they
stayed mutually comparable — is worth porting.

## Decision

What is *not* built is the predecessor's Grid4D/W-matrix conservative regrid — a
mechanism for squeezing tables from different peak-a0 runs onto one *fixed* target bin
range so they stayed mutually comparable, which nothing in this repo needs (each
`Collision` builds its own table fresh, RES030). What *is* kept, and is the actual physics
the predecessor's mechanism rested on: for a fixed envelope shape,
`a0_local(t) = a0_peak * envelope(t)` is exactly linear in the peak, so both
`TrajectorySamples.ahat` (`a0_peak**2 * a0_shape`) and `TrajectorySamples.luminosity`
(proportional to `sum(a0_local**2)`) scale as `a0_peak**2` for the *same cached
trajectories* — `retargeted_ahat`/`retargeted_luminosity` are that rescale, and
`stages.deposit_table(samples, a0_peak=...)` deposits both, so a different pulse energy's
redshift *and* total photon count are a fresh Stage 1 deposit from cached Stage 0
samples, no rerun.

(`retargeted_luminosity` was missing from the first cut of this decision — `deposit_table`
deposited the retargeted `ahat` but the *original* `luminosity`, so a retargeted table's
`total_weight` did not move with `a0_peak` at all. Caught by a question about exactly this
reasoning; `tests/test_stage1_stage2.py::
test_retargeted_luminosity_matches_a_fresh_stage_0_run_at_that_a0_peak` cross-checks the
fix against an actual second Stage 0 run at a doubled pulse energy, agreeing to 0.1%.)

## Alternatives considered

**Port `retarget_a0`/Grid4D/`a0_kind` verbatim.** Would add a W-matrix conservative-regrid
module with no measured cost it avoids, contradicting the predecessor's own two
build-then-delete-then-reject pattern this project already tracks for other speculative
machinery (P6/P7/P9/P10/P11).

**Rescale the existing direct-deposit table's edges in place for a new peak a0, without a
full regrid.** Because the ahat axis and the samples deposited into it scale by the same
factor, a table retargeted to a new peak a0 could in principle be produced by rescaling
`Table.ahat_edges` alone (binning is invariant under a uniform positive rescale of both
data and edges by the same factor) — touching no array at all, cheaper than even a fresh
deposit. Not built: nothing calls `deposit_table(a0_peak=...)` from production code today
(`Collision._table()` always asks for the pulse's own a0), the trick does not hold in the
degenerate zero-divergence-beam branch of `_uniform_edges` (its fallback padding does not
scale linearly), and a fresh deposit is already measured cheap next to Stage 0
(`test_deposition_is_cheap_next_to_stage_0`, `DEFAULT_TABLE_BINS`, 50k particles). Revisit
if a real caller (a pulse-energy scan) needs the extra speed.

## Rationale

The peak-independence of a fixed envelope shape (`a0_local(t) = a0_peak * envelope(t)`)
means the deposited data scales predictably with `a0_peak`, so a per-pulse-energy rescale
from cached Stage 0 samples reproduces what a full regrid mechanism would have provided,
without needing the predecessor's cross-run comparability machinery this repo doesn't use.

## Consequences

The illustrative recompute-cost table's "pulse energy → a0 | REUSE_INTERMEDIATES | Stage 1
a0-axis retarget (no re-deposition)" row assumed the ported regrid mechanism.
`GRAND_PLAN.md` v0.14 restates it as "Stage 1 re-deposit from cached Stage 0 samples
(measured cheap)" — same tier, different mechanism. `XigmaEngine.recompute_costs`
(`engine.py`) does not yet claim this tier for pulse energy regardless (see RES030): the
`REUSE_INTERMEDIATES` claim needs a live consumer that knows a laser edit was
pulse-energy-only and maps it to the corresponding `a0_peak`, and none exists until a GUI
or scan helper does that mapping.
