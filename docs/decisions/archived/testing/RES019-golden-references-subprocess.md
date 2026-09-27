# RES019 — Golden references are generated in a subprocess, and committed as ordinary results files

Status: implemented
Class: testing
Archived: 2026-09-21

**Superseded by RES087** (2026-09-21): GammaForge retired the transitional snapshot
bridge after its native validation legs became the maintained scientific authority.

## Problem

Golden references need to be generated from the predecessor's code for cross-validation,
but both repos install a package named `gammaforge`, so they cannot coexist in one Python
process — and the goldens themselves need a storage and versioning strategy.

## Decision

`validation/make_references.py` builds a JSON description of each scenario in the
predecessor's units and runs `validation/_predecessor_driver.py` under a *separate
interpreter* whose path points at the old checkout (the OLD_REPO and OLD_REPO_PYTHON
environment variables — env var names, not package symbols). The driver writes plain
`.npz` files; the translation into this repo's `Axis`/`OutputKind` vocabulary and CGS
units happens on this side of the boundary. Snapshots are stored as the ordinary
`gammaforge.io.formats.hdf5` results file plus a `provenance` group and a `scalars` group,
under `src/gammaforge/validation/references/data/`, and are **committed** — which needs a
`.gitignore` negation, since `*.h5` is otherwise excluded wholesale.

## Alternatives considered

**(a) Import the old package directly and call it in-process.** Impossible, not merely
inadvisable: both repos install a package named `gammaforge`, and no import trick makes two
of them coexist safely in one process. The subprocess is the design. It also buys the old
repo's own environment — it needs scipy, which this repo deliberately does not have.

**(b) Let the driver write this repo's HDF5 format itself.** Would put this repo's
serialization format inside a file that runs against the old repo's dependencies, where it
could not be tested and would silently drift; instead the driver reports what the old code
produced in the old code's own terms, and the one place that knows both vocabularies is
versioned with the format it targets.

**(c) Keep goldens out of git and regenerate them before each run.** Would make the suite
depend on a machine that has the predecessor checked out — the opposite of what a reference
is for. At ~24 kB per snapshot the C3 concern about committed data does not bite; the
negation is scoped to that one directory so it cannot quietly re-admit a stray multi-MB
file elsewhere.
