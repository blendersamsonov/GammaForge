# D056 — Code comments citing a decision are trimmed to a pointer; the rationale lives only in the decision file

Status: implemented
Class: process

## Problem

After *DECISIONS.md*/*docs/DECISIONS_ARCHIVE.md* were migrated to `docs/decisions/`
(D055), many code comments and docstrings still carried the decision's own rationale
inline — a derivation, a list of rejected alternatives, or historical measured numbers —
duplicated from the decision file rather than just cited from it. A session reading a
docstring for an unrelated reason paid to skim paragraphs of decision history it never
asked for, and a duplicated fact can drift once one copy is edited and the other isn't.

## Decision

A comment or docstring that cites a decision id is trimmed to a pointer: one clause of
current behavior or contract, plus `(D0NN)`. The full rationale — derivation, alternatives
considered, historical numbers, discovery narrative — lives only in
`docs/decisions/{lifecycle}/{class}/D0NN-*.md`. The exception is genuinely current,
non-obvious behavioral information a maintainer needs to avoid reintroducing a bug or
breaking an invariant (a derivation needed to verify a formula, a "do not do X here, that
reintroduces Y" warning); that stays in the code, in the code's own voice, even where it
overlaps with what the decision file also says. `docs/decisions/README.md`'s *Citing from
code* section and `AGENTS.md`'s working conventions both state the rule going forward.

## Alternatives considered

**Keep full rationale duplicated in both places, for locality.** Rejected: the
duplication is exactly the cost this decision removes — every unrelated read of the code
pays the attention cost of decision-history it didn't ask for, and the two copies can
silently disagree once one is edited without the other. The migration itself surfaced
several stale mismatches between prose and the backticks/italics convention that this
duplication had let drift unnoticed.

**Drop the code-side citation entirely and rely on `docs/decisions/INDEX.md`'s listing.**
Rejected: a bare `(D0NN)` citation is what lets a maintainer standing at the one line of
code that matters jump straight to the decision, without a repo-wide search. Trimming the
surrounding prose already captures the token saving; removing the citation itself would
lose a real navigation aid for no further gain.

## Rationale

The two kinds of information a comment can carry — *what the code currently does, and
why the current shape is correct or safe to keep* versus *how this was decided and what
lost* — have different audiences and different lifetimes. The first is read by anyone
touching this code, for any reason; the second is read only by someone who deliberately
wants the history. Keeping the second inline taxes the first reader for the second
reader's benefit. `docs/decisions/`'s own existence already solves for the second
reader — a bare id is enough to find the whole story — so the code only needs to carry
what the first reader needs.

## Consequences

Applied in one pass across 16 files under `src/`/`tests/` (−283/+184 lines); the full
`pytest` suite (425 passed, 1 skipped) was unaffected, which is itself evidence no
load-bearing information was lost — a docstring's derivation needed to trust a formula,
or a warning needed to avoid reintroducing a past bug, would have had no test to break if
it had been cut, so that check alone doesn't fully validate the trim, but the review that
accompanied it (checking each site against its decision file before cutting) does. Future
comments citing a decision are held to the same rule as they're written, not just
retroactively cleaned up — stated explicitly in `AGENTS.md` and
`docs/decisions/README.md` so it doesn't silently regress.
