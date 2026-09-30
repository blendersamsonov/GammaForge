# RES002 — Doc-staleness guard (C2) scope: `DECISIONS.md` only, not `GRAND_PLAN.md` or `PROGRESS.md`

Status: implemented
Type: testing
Archived: 2026-08-19

**Superseded by RES055** (2026-08-19): `DECISIONS.md` and `docs/DECISIONS_ARCHIVE.md` were
migrated to a one-file-per-decision tree under `docs/decisions/{lifecycle}/{class}/`, so
this entry's literal scope claim (a single checked file) is no longer true. RES055 restates
the same reasoning — an after-the-fact decision log is the one place "every backtick
resolves" is true by construction — against the new layout, and expands coverage to
`proposed/`/`rejected/` decisions, which didn't exist as a checkable category before.
Left below as the historical record of the reasoning that held while `DECISIONS.md` was
one flat file, not as a description of the guard's current scope.

## Problem

The Phase-0 doc-staleness smoke test (`tests/test_doc_staleness.py`) needs a defined
scope — which docs' backticked references it holds to a must-resolve standard. Checking
every doc under `docs/` uniformly doesn't work, since some docs are legitimately
forward-looking or narrative rather than after-the-fact records, and a guard that fails
on those would fail universally rather than usefully.

## Decision

The Phase-0 doc-staleness smoke test (`tests/test_doc_staleness.py`) only checks
backticked identifiers in `DECISIONS.md` against the repo/installed package.
`GRAND_PLAN.md` and `PROGRESS.md` are explicitly excluded.

## Alternatives considered

**Checking every doc under `docs/` plus `PROGRESS.md`.** Rejected — see Rationale: both
would fail the guard's actual purpose rather than serve it.

## Rationale

The guard's purpose (per the plan's own risk table, §12) is catching *backward* drift —
docs that used to be true and silently stopped being true, the failure mode observed in
the predecessor repo's *AGENTS.md* (a different repo — not a claim checked against this
one). `GRAND_PLAN.md` is a forward-looking roadmap by design: most of its backticked
identifiers (e.g. *LaserField*, *fit_gaussian_paraxial*) intentionally name things that
don't exist yet and won't until their assigned phase lands — checking it now would fail
universally, not usefully. `PROGRESS.md` is a narrative session log that mixes "what we
did" with "what we just decided," so it has the same problem in miniature. `DECISIONS.md`
entries are only added once a decision is actually implemented, so it's the one doc where
"every backtick resolves" is true by construction *and* worth mechanically enforcing as
things get renamed later.

## Consequences

The guard mechanically enforces `DECISIONS.md`'s backtick convention only; drift in
`GRAND_PLAN.md` or `PROGRESS.md` depends on manual review, not this test. The check can
grow to cover more docs (e.g. a future API reference) once they exist and follow the same
after-the-fact convention.
