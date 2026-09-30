# RES011 — Doc-staleness guard hardened for enum members and ambiguous shapes; its own behaviour is tested

Status: implemented
Type: testing
Archived: 2026-08-10

## Problem

The doc-staleness guard's original first-match token classification was wrong in a way
that grew with usage: shapes like `FieldKind.CHOICE` match multiple interpretations (the
file-like shape and the class-attribute shape), and committing to the first match produced
false positives as soon as Phase 1 gave the docs symbols to reference.

## Decision

`tests/test_doc_staleness.py` now (a) tries **every** applicable interpretation of a token
and reports it stale only if none resolves, rather than committing to the first shape that
matches; (b) indexes enum member names, so `WIDTH` resolves; (c) accepts annotation-only
dataclass fields, so `Bunch.weight` resolves; (d) skips bare file extensions. Three further
tests pin the guard's own behaviour against fixed lists of tokens that must resolve, must
be caught, and must be skipped.

## Alternatives considered

**Keeping the first-match ordering and adding an exceptions list for the tokens it
misclassified.** Rejected — see Rationale.

## Rationale

The first-match rule was not merely imprecise, it was wrong in a way that grew with usage:
`FieldKind.CHOICE` matches the file-like shape ("a name, a dot, a short suffix") just as
`GRAND_PLAN.md` matches the class-attribute shape, so *every* dotted symbol reference in
the docs was reported stale as soon as Phase 1 gave the docs symbols to reference. An
exceptions list would have made RES002's whole premise false — that convention exists
precisely so no manual exception list is needed. Testing the guard itself matters more than
usual here because its failure mode is silent: relaxing a rule to kill a false positive can
quietly stop it detecting anything, and nothing would say so. The stale-token list is drawn
from names this project rejected deliberately (*BeamFittedParams*, *ModelCapabilities*,
*NoConvention*, *Bunch.n_electrons*), so it doubles as a check that they stay gone.
