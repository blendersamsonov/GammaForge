# RES016 — `beam.py` renamed to `bunch.py`; the "canonical bunch" phrasing dropped

Status: implemented
Class: architecture
Archived: 2026-08-10

## Problem

The §3.2 module was first named `beam.py`, but its actual centre of gravity is the
macroparticle bunch, and its docstrings used unexplained "canonical bunch" phrasing.

## Decision

The §3.2 module is `bunch.py`, and the word "canonical" no longer qualifies the bunch or
the sampling path anywhere. It still qualifies *units* — "the canonical CGS unit" is the
one a value is stored in — which is a different and standard usage.

## Alternatives considered

**Keeping `beam.py`, on the grounds that the module holds `GaussianElectronBeam` as well as
`Bunch`.** Rejected — see Rationale.

## Rationale

The module's centre of gravity is the macroparticle bunch — sampling, the prefilter,
propagation, the fit — with the beam description as the analytic summary of it, and the
predecessor named the same content `bunch.py` for the same reason, so this is continuity
rather than churn (the same argument that keeps the package called `io`).

"Canonical bunch" was read by the author as unexplained jargon, which it was: it meant
nothing more than "every engine in a run is given the same bunch", and that is now simply
what the docstring says. A phrase that needs decoding to convey an ordinary fact is a cost
with no benefit, and this project's documentation discipline (goal #8) is about
provenance, not vocabulary.
