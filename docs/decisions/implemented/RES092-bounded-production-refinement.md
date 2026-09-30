# RES092 — Bounded production refinement

Status: implemented
Type: testing

## Problem

RES084's fixed-direction production measurement failed six low-a0 refinement
checks even though its finest-grid reference comparisons passed. Its production
table configurations also had not acquired the fifth, carrier-rate axis. Running
larger retarget grids allocated large five-dimensional arrays for bins outside
the deposited support, making a convergence measurement unnecessarily costly.

## Decision

The production configurations in `delta_validation.py` use all five table axes.
`production_checks` first measures the original angular 32/64-bin and retarget
512/1024-bin matrix. Scenarios failing angular-table or retarget refinement are
remeasured once with angular 64/128 bins and retarget 1024/2048 bins. Gamma,
shape and carrier resolutions, particles, seed, integration steps, physical
energy bins, quadrature orders and acceptance budgets are unchanged.

Every direction and every numerical gate for a retried scenario is evaluated
on the complete second matrix. Initial failed checks remain in report notes;
the second matrix determines that scenario's numerical acceptance. Both passes'
source-stability and scenario-coverage checks remain mandatory. There is no
unbounded retry, scenario-name exception or automatic scientific acceptance.

`retarget_ahat` removes unreachable trailing overlap-matrix columns before
allocating the output channels. This preserves the requested nonuniform grid's
edges and all reachable weights; it does not stretch the grid onto the occupied
support or change the engine defaults. Final mass-based trimming still applies.

## Alternatives considered

Raise the budgets: this would hide unresolved discretization error rather than
measure convergence. The provisional RES074 budgets remain unchanged.

Use the larger matrix for every scenario: unnecessary work for already-converged
cases, especially at high intensity where more retarget bins are reachable.

Change the shipping retarget grid law or its default decades: this would change
production numerics and the separate author decision tracked under RES053.
Validation resolution can be increased without making that decision.

## Rationale

A bounded second measurement separates insufficient resolution from persistent
disagreement. Keeping failed coarse measurements visible preserves evidence of
why a second pass was needed, while its complete set of gates prevents a single
favorable metric from promoting an unconverged scenario.

## Consequences

Production validation takes longer for failing scenarios and can still fail the
finer matrix. Missing coverage and non-finite measurements remain failures.
The fixed-direction CPU comparison still shares trajectory inputs and does not
close particle/seed, Stage-0, gamma/shape-grid, angular-aperture, independent CUDA
or four-method coverage. RES084's scientific blockers remain in force.
