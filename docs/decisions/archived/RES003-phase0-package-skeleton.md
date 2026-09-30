# RES003 — Phase 0 package skeleton: subpackage `__init__.py` only, no placeholder module files

Status: implemented
Type: process
Archived: 2026-08-10

## Problem

Phase 0 needs to lay out the package tree before later phases populate it — the question
is how much of the plan's eventual module layout to pre-create now.

## Decision

Phase 0 creates six subpackages under `src/gammaforge/` — `io`, `engines/xigma`,
`engines/analytical`, `engines/kascade`, `validation/references`, `gui` — each with a
docstring-only `__init__.py`. It does not pre-create the individual module files the plan
names for later phases (e.g. `schema.py`, `beam.py`, `laser.py`, `stages.py`,
`collision.py`).

## Alternatives considered

**Stub out every module file the plan names now (empty, or raising `NotImplementedError`),
so the full tree matches the plan's layout from day one.** Rejected — see Rationale.

## Rationale

An empty `schema.py` sitting in the tree for the whole of Phase 0 either looks finished
(misleading) or needs a marker comment nobody will remember to remove. Phase 1's own exit
criteria (round-trip tests, schema validation tests, ...) are the actual definition of
"the schema module exists" — better for a file's existence to mean something than to
pre-populate the tree for cosmetic completeness.
