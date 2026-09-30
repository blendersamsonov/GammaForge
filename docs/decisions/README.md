# Decisions

One decision, one file. A decision record captures a real implementation or design
choice — the *why*, the alternatives it beat, and what it cost — the part code and commit
messages can't carry on their own. This file defines where decisions live, when to write
one, and the in-file format.

## Layout and naming

Every decision has a lifecycle folder, a permanent ID in its filename, and a type in its
header: `docs/decisions/{lifecycle}/RESNNN-topic-title.md`.

- **Lifecycle** (top-level folder) is the decision's status, and a decision **moves**
  between folders as that status changes:
  - **`proposed/`** — reviewed, not yet built (or only partly).
  - **`implemented/`** — the decision shipped. Kept current with what actually shipped:
    when the code later moves a file, renames something, or changes a default, update the
    fact in the same change (paths, names, structure — not the decision itself, and not
    its `Alternatives considered`/`Rationale`).
  - **`rejected/`** — considered and declined. Never deleted (see *Archiving* below) —
    only moved to `archived/` once its rationale stops being load-bearing.
- **Type** (`Type:` header tag) is the *kind* of decision, from the closed set below.
- **`RESNNN`** is a permanent, sequential id — `RES001`, `RES002`, … — assigned once, in filename
  order across the whole tree regardless of lifecycle or type, and **never reused**, so
  code comments can cite one bare (`RES013`, not a path) and it stays resolvable even after
  the file it names moves between folders. The slug after the id is a short, lowercase,
  hyphenated summary of the title — free to reword if the title changes, since the id, not
  the slug, is the permanent handle. Slugs are ASCII, letters/digits/hyphens only; a
  section number like "§9.1" in a title becomes `9-1` in the slug, not `9.1` — the
  filename-format gate does not accept a literal period (`RES026-9-1-resolved-missing-
  factor-in-paper.md`, `RES033-9-1-closed-both-transcriptions-1-over-2pi.md`,
  `RES027-phase2-5-review-round-corrections.md` are this repo's three examples).

`docs/decisions/INDEX.md` is the map: `id | title | type | status | path`. Update it in
the same change as any new decision, lifecycle move or type change. The index type must
match the file's `Type:` tag; the index path must point to its lifecycle folder.

## Decision types

Each decision has one `Type:` value from this closed set:

| Type | What it covers |
|---|---|
| `feature` | A new capability that didn't exist before. |
| `bug-fix` | Corrects (or explicitly declines to correct) a case where behavior was, or
  is, wrong relative to what was actually intended — whatever caught it, a test, a
  review, or a derivation someone worked through by hand. |
| `simplification` | Removes machinery, complexity, or surface area without changing
  observable behavior. |
| `architecture` | A structural decision about the shipped system — how modules relate,
  what the data model represents, what the boundaries are. |
| `process` | Tooling, workflow, or doc-discipline decisions *around* the code — not
  runtime behavior. |
| `testing` | Test or validation infrastructure and strategy. |

Choose the type by what *kind* of change the decision produced, not by which domain motivated
it — a missing factor caught by working through the underlying physics by hand
(`RES026`/`RES033`/`RES053`) is still a `bug-fix`, the same as one a unit test would have caught
(`RES007`/`RES012`/`RES014`/`RES021`/`RES022`).

## When to write one

Add a decision when a real choice gets made — a `proposed/` note for something reviewed
but not yet built, an `implemented/` note once something is actually built. **Entries are
added after the fact, not as a promise**: don't file `implemented/` for a plan, and don't
leave a `proposed/` note stale once the thing it describes has shipped or been dropped —
move it. The [index](INDEX.md) records current lifecycle state.

**Every new decision triggers a supersession check**: search the tree for older decisions
covering the same question, and if the new one replaces an old one, say so explicitly in
both and archive the old one in the same change (see *Superseding* below) rather than
leaving two live entries that disagree.

Only a purely mechanical or local edit — no change in behavior, structure, or reasoning —
does not need one.

## Citing from code

A comment or docstring may cite a decision bare, `(RESNNN)`, wherever the code embodies
that decision — no path, no backticks needed, since ids are permanent handles (see
*Layout and naming* above). Keep the citation a **pointer, not a retelling**: state the
current behavior or invariant in one clause, then cite the id. If a citation's
surrounding comment grows past that — a derivation of *why*, a list of alternatives that
lost, historical measured numbers — that's a sign the content belongs in the decision
file, not in the code. Move it there. A reader who wants the full story opens `RESNNN`;
everyone else, reading the code for an unrelated reason, doesn't pay to skim it.

The exception is genuinely current, non-obvious behavioral information a maintainer needs
to avoid reintroducing a bug or breaking an invariant — a derivation needed to verify a
formula, a "do not do X here, that reintroduces Y" warning. That stays in the code, in
the code's own voice, decision id attached, even where it overlaps with what the decision
file also says. RES054 records the full `<a^2> = C a0^2` reasoning while
the relevant code docstrings cite `(RES054)` in one or two clauses (RES056).

## Archiving (and why nothing is ever deleted)

Archive a decision — `implemented/` or `rejected/` alike — once it stops being load-
bearing: an `implemented` decision whose shipped behavior has since been fully replaced,
or a `rejected` decision whose rationale no longer guards against a plausible mistake.
Archiving is a **move**, to `archived/RESNNN-topic-title.md`, plus one appended
header line (`Archived: YYYY-MM-DD`) — the rest of the file, including its `Status:` line,
is untouched. Nothing is rewritten and nothing is deleted, because a bare id might already
be cited from a code comment somewhere, and a citation that resolves today must
keep resolving.

This is a deliberate departure from systems (deepseek-harness's Agent Notes, for one) that
delete stale `rejected/` notes outright — that only works if decisions are cited by
link/path, which breaks loudly (a 404) if the target moves or disappears. Bare-id citation
needs the opposite guarantee: the id must always resolve to *something*, so archiving,
never deleting, is the rule here.

## Superseding

When a new decision replaces an old one: state it in the new decision's opening (`##
Problem` or `## Decision`, "supersedes RESnnn, because..."), then archive the old one with a
short pointer paragraph at the very top of its body, before `## Problem`:

```markdown
**Superseded by RESNNN** (YYYY-MM-DD): <one line on what changed and why>.
```

The rest of the archived file's body is untouched — it's still the historical record of
what was decided and why, at the time it was decided.

**Worked example in this repo:** `archived/RES024-*.md` and
`archived/RES025-*.md` are both fully superseded (by `RES054` and `RES033`
respectively) and carry this pointer. `archived/RES028-*.md` is superseded by
`implemented/RES032-*.md` within the same development session — the pointer
paragraph and the `Archived:` date (`2026-08-08`) were both traced from the commit that
landed the superseding decision, not invented, since the original prose only said "within
the same session."

Not every supersession is a full replacement. `RES041` and `RES054` each supersede only *part*
of an earlier decision (`RES039`'s crossing-angle refusal, and `RES053`'s cycle-average-constant
mechanism respectively) — the earlier decision is still substantially current, so it stays
in `implemented/` rather than moving to `archived/`, with the partial correction noted
inline in its own body rather than via the full archive-and-point mechanism above. Use
judgment: archive-and-point when a decision is *replaced*, note inline when it is merely
*narrowed*.

## Correcting a decision without rewriting it: Amendments

A decision's reasoning sometimes needs a correction — new information, a mistake in the
original argument — without the decision itself changing. Don't edit the original prose;
append a dated, quoted note under a trailing `## Amendments` section instead:

```markdown
## Amendments

> **YYYY-MM-DD — <what's being corrected or reaffirmed>.** <unchanged-style prose
> explaining the correction, in the same voice as the rest of the file.>
```

This keeps the log honest about what was believed when, rather than quietly rewriting
history to look right in hindsight. `archived/RES040-*.md` is this repo's worked
example: two dated corrections, originally interleaved through the middle of the entry,
consolidated into one trailing `## Amendments` section without changing their wording.

## Id collisions across parallel work

Ids are assigned in order, but two branches worked on in parallel can independently claim
the same next id before either merges. When that happens, renumber the *later-merging*
one and say so, once, at the very top of its body (before `## Problem`, after any
superseded-by pointer):

```markdown
*(Numbering note: this entry was written as RESAAA and renumbered to RESBBB on YYYY-MM-DD to
resolve a collision with parallel work on `<branch>`.)*
```

**Worked example in this repo:** `implemented/RES052-*.md` was originally
written as `RES035` and renumbered on `2026-08-10` after a parallel Phase-4 branch had
already claimed `RES035`–`RES051`. The note sits at the top of `RES052`'s body, above `##
Problem`, exactly as above.

## The file format

The first lines of every decision file are fixed:

```markdown
# RESNNN — <title>

Status: <status>
Type: <type>
```

followed by a blank line, then the body. `Status:` is one of:

- `Status: proposed`
- `Status: implemented`
- `Status: rejected — <why, in one line>`

and must agree with the lifecycle folder the file sits in. `Type:` must use a value from
the table above and agree with the index. An archived file adds exactly one more header line,
`Archived: YYYY-MM-DD`, and otherwise keeps whichever `Status:` it had the day it shipped
or was declined — archiving records when a decision stopped being current, not what it
originally said.

### Body skeleton by lifecycle

Every decision opens its body with `## Problem` — the motivation, written to stand on its
own without the resolution. What follows depends on lifecycle; these are the only
recurring section names, and each is required unless marked optional:

**`proposed/`**
```markdown
## Problem
## Proposal
## Alternatives considered      <- mandatory
## Acceptance criteria
## Risks
```
`## Proposal` may speak in future tense — this is the one lifecycle where the thing being
described doesn't exist yet. `## Acceptance criteria` says what observable state means
done. `## Risks` covers both what could go wrong and what's knowingly being given up.

**`implemented/`**
```markdown
## Problem
## Decision
## Alternatives considered      <- mandatory
## Rationale
## Consequences
## Amendments                   <- optional, see above
```
`## Decision` describes shipped reality in present tense. Proposal-era language
(`## Proposal`, `## Acceptance criteria`, `## Risks`) does not belong here — if a decision
started as a `proposed/` note, rewrite those sections into `## Decision`/`## Consequences`
when it ships, don't just leave them. `## Rationale` and `## Consequences` are expected but
not mechanically required — a handful of this repo's entries omit one or the other where
nothing beyond the Decision/Alternatives content would be genuine, rather than padding
with restated material.

**`rejected/`**
```markdown
## Problem
## Proposal
## Alternatives considered      <- mandatory
## Acceptance criteria          <- optional, carried over frozen if the proposal had one
## Risks                        <- optional, carried over frozen if the proposal had one
```
The verdict lives on the `Status:` line, not in a body section — a rejected decision is
frozen at proposal time, so it keeps whatever proposal-era sections it had rather than
having them stripped on the way to `rejected/`.

### Alternatives considered — always mandatory

Every decision, in every lifecycle, has an `## Alternatives considered` section: each
genuine alternative and why it lost, one paragraph per alternative (or a `### Why not
<X>?` subsection per contested one). A decision recorded without what it beat invites
re-litigation — this section is most of the reason any of this is worth writing down.
