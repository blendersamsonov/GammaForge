# RES055 — Doc-staleness guard scope: every `proposed`/`implemented`/`rejected` decision file, not `archived/`; supersedes RES002

Status: implemented
Type: testing

## Problem

RES002 scoped the doc-staleness guard (`tests/test_doc_staleness.py`) to *DECISIONS.md*
alone, on the reasoning that only an after-the-fact decision log makes "every backtick
resolves" true by construction. *DECISIONS.md* and *docs/DECISIONS_ARCHIVE.md* were
migrated to a one-file-per-decision tree under `docs/decisions/{lifecycle}/{class}/`, so
RES002's literal claim (a single checked file) is now false, and the guard needs a scope
statement that matches the new layout.

## Decision

`tests/test_doc_staleness.py` now walks every file under
`docs/decisions/{proposed,implemented,rejected}/**/*.md` and checks their backticked
tokens exactly as before. `docs/decisions/archived/**/*.md` is excluded (same rationale
as RES002's original archive exclusion: an archived entry describes code that's since
moved or gone, so "every backtick resolves" is no longer true by construction there), and
so are `docs/decisions/README.md`/`INDEX.md` (navigation/meta prose, same category as
`GRAND_PLAN.md`/`PROGRESS.md`).

This is a real coverage expansion, not just a reshaping of RES002's scope: `proposed/` and
`rejected/` decisions didn't previously exist as a checkable category (*DECISIONS.md* held
only after-the-fact entries), and an individual decision file is exactly the size where
the backtick/italics convention is easy to hold to rigorously.

## Alternatives considered

**Keep checking one concatenated virtual "document"** (read every file under the three
included lifecycles and treat them as one text blob, same as the old single-file
*CHECKED_DOCS* list conceptually). Rejected: failure messages would lose the per-file
path, which is exactly the information needed to fix a flagged token — the guard already
reports `doc.relative_to(REPO_ROOT)` per failure, and collapsing files back into one
virtual document would only complicate that for no benefit.

**Also check `archived/`.** Rejected for the same reason RES002 excluded historical
scaffolding: an archived decision's code may have moved or been deleted since, so its
backticks are not expected to resolve against the current package — checking it would
produce noise the guard cannot distinguish from a real regression.

## Rationale

Re-running the migration surfaced genuine backtick-convention violations the guard's
expanded scope was designed to catch: several decisions promoted out of
*docs/DECISIONS_ARCHIVE.md* (never checked before, since the old guard only covered the
"live" file) and a handful of freshly-written connective sentences in the migrated files
used backticks for predecessor-only names, not-yet-built symbols, and since-removed
constants (*ENGINES*, *DEFAULT_A0_MAX*, *CYCLE_AVERAGE_FACTOR*,
*OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION*, *reference.py*/*spectrum4d.py*/
*spectrum4d_cpu.py*) where the original prose had correctly used italics. All were fixed
to match the original convention before this decision was recorded, and
`test_guard_resolves_current_references`/`test_guard_still_detects_stale_references`/
`test_guard_skips_shapes_it_cannot_judge` (the guard's own self-tests, unchanged) confirm
the guard's judgment logic itself needed no changes — only its file-discovery scope did.

## Consequences

The guard now runs against 33 files instead of 1 (all currently-`implemented` decisions;
`proposed`/`rejected` are empty today but will be checked automatically the moment either
is used). Runtime is proportionally longer (~55s locally) but still well inside the full
suite's budget. A decision that moves from `implemented`/`rejected` to `archived` drops
out of the checked set at that point, matching RES002's original archive exemption exactly.

## Amendments

> **2026-09-30 — Decision tree flattened.** Decision files now live directly under
> `docs/decisions/{lifecycle}/`, with their kind in a `Type:` header. The class
> subdirectories described above were removed; the historical scope decision is unchanged.
