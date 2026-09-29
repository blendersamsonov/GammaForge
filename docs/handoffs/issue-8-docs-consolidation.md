# Documentation consolidation for issue #8 — implementation handoff

Status: ready for implementation.

Tracking issue: #8, `Consolidate and retire stale repository documentation`.

Branch base: `main` at `66e876d278f7613fdc50ed2c08218f0100da8551`.

## Goal

Reduce GammaForge's repository documentation to a small set of clearly authoritative artifacts without losing current scientific knowledge, live work, or user-facing instructions.

The desired end state is intentionally simple:

- `README.md` — human-facing project, installation, and usage entry point;
- `AGENTS.md` / `CLAUDE.md` — compact repository-specific agent instructions;
- `docs/derivations/**` — durable physics;
- `docs/decisions/**` — durable implementation/design rationale;
- `docs/validation/**` — curated durable scientific/numerical evidence;
- GitHub issues — unfinished work;
- Git history / pull requests — process and implementation history.

Handoffs are temporary branch/PR execution context. This handoff must be removed before this PR is merged.

## Authority

Issue #8 owns the cleanup scope and acceptance criteria.

For content decisions during cleanup:

- current `DERNNN` records own settled/recorded physics;
- current `RESNNN` records own durable implementation/design decisions;
- current validation records own durable scientific evidence;
- GitHub issues own active work;
- merged code owns current implementation;
- `blendersamsonov/Xigma-Paper` owns manuscript prose and paper-only material.

Do not preserve a stale narrative document merely because another stale document links to it. Replace references with the actual durable authority where one exists.

## Current state

At the branch base, the top-level documentation still includes:

- `PROGRESS.md`;
- `docs/ALPHA.md`;
- `docs/ALPHA_GPU_VALIDATION.md`;
- `docs/GRAND_PLAN.md`;
- `docs/GRAND_PLAN_REVIEW.md`;
- `docs/REPORT_DRAFT.md`;
- `docs/UI_SPEC.md`;
- `docs/xigma-delta-gpu-report.md`;
- `docs/notes/`;
- `docs/postmortems/`;
- legacy completed handoffs in `docs/handoffs/`;
- a mixed collection of concise validation reports and large historical/raw validation packets.

The audit found that several of these documents overlap or contradict newer sources. Examples include old paper-authority language in `GRAND_PLAN.md`, older xigma-table descriptions in narrative report material, and inconsistent backend/default statements across alpha documentation and current code.

The repository already has mature `docs/decisions/` and `docs/derivations/` systems with stable identifiers. Open issues already cover several live work items formerly duplicated in progress/planning prose.

## Required work

### 1. Build a reference/dependency inventory before deleting files

From the branch, search the whole repository for references to every retirement candidate, including:

- Markdown links;
- source comments/docstrings;
- tests;
- notebooks/examples;
- package metadata;
- CI/tooling;
- decision/derivation records;
- validation reports.

Classify each reference as:

1. stale and removable;
2. should point to a stable `RESNNN`/`DERNNN`/validation record instead;
3. user-facing information that belongs in `README.md` or an executable example;
4. live task information that belongs in an issue;
5. genuinely load-bearing information that must be migrated before its source document is removed.

Do not start with mass deletion and repair breakage afterward.

### 2. Retire obsolete planning/status/process narratives

Re-evaluate the candidates in issue #8 against current branch contents and remove those that no longer own unique current information.

Expected strong candidates include:

- `docs/GRAND_PLAN_REVIEW.md`;
- `docs/GRAND_PLAN.md`, after extracting any still-load-bearing repository invariant;
- `PROGRESS.md`, after resolving its live open items into issues or deliberate retirement;
- `docs/UI_SPEC.md`, after preserving only necessary user-facing behavior/current architectural references;
- `docs/ALPHA.md` and `docs/ALPHA_GPU_VALIDATION.md` where README/examples/current validation evidence supersede them;
- manuscript/report material under GammaForge that belongs in Xigma-Paper;
- resolved consultation notes superseded by current derivations/decisions;
- postmortem material whose durable lesson is already captured elsewhere;
- legacy completed handoffs on `main`.

Deletion is preferred over creating an `archive/` dumping ground when Git history already preserves the artifact.

### 3. Migrate live work out of PROGRESS.md

Audit each unfinished item individually.

- If an open issue already owns it, remove the duplicate prose.
- If it is still genuinely desired, sufficiently concrete, and not tracked, create a focused issue.
- If it is obsolete, speculative, or no longer prioritized, drop it.
- Do not create issues mechanically just to preserve every historical TODO.

Issue #8 itself should not become a catch-all for unrelated implementation work discovered during this audit.

### 4. Curate validation evidence conservatively

Treat validation differently from narrative docs: durable scientific evidence is a first-class repository artifact.

For each committed validation file, determine whether it is:

- a current concise evidence record supporting a still-relevant claim;
- a fixture actually required by tests;
- a superseded exploratory/raw packet whose conclusion is already preserved in a later report, `RESNNN`, or `DERNNN`;
- generated output that should instead live under the ignored validation-output path.

Keep evidence required to support current verified derivations, implemented decisions, release gates, or active scientific acceptance claims.

Do not discard raw data that is the only surviving support for a current claim. Conversely, do not keep megabyte-scale historical packets merely because they once existed.

If useful, add one short `docs/validation/README.md` describing what deserves to be committed versus generated locally. Do not create a new complex lifecycle or index system unless the existing repository clearly needs it.

### 5. Simplify surviving entry/convention documents

#### README

Make `README.md` the one concise user-facing entry point for:

- project purpose;
- supported installation/use path;
- current backend/engine support at a stable level;
- links to examples/notebooks;
- links to derivations/decisions/validation only when useful to a user/developer.

Avoid copying current issue status or transient acceptance matrices into README.

#### AGENTS.md

Trim `AGENTS.md` to GammaForge-specific material:

- authority split;
- important architectural/physics invariants that a coding agent can violate;
- derivation/decision/issue/handoff workflow;
- repository-specific testing and validation commands;
- manuscript boundary;
- any genuinely project-specific development conventions.

Remove duplicated project history/current status and generic coding-agent advice that is not specific to GammaForge.

Keep `CLAUDE.md` only as the existing alias/pointer if that mechanism still works after cleanup.

#### Derivation/decision convention docs

Keep their stable format/workflow rules. Remove volatile "current status" summaries that duplicate the indexes and have already gone stale.

Do not rewrite historical individual `DERNNN`/`RESNNN` bodies as part of cleanup.

### 6. Repair references

After the retirement pass:

- update README/AGENTS links;
- replace load-bearing references to deleted planning/status docs with current stable IDs or code locations;
- remove obsolete source/docstring references rather than inventing replacement prose when the code is self-explanatory;
- ensure notebooks/examples do not instruct readers to consult deleted docs;
- ensure issue/PR links remain valid where they are the correct authority.

Prefer stable `DERNNN`/`RESNNN` identifiers over filesystem paths when repository conventions permit.

## Invariants

This is documentation consolidation, not a hidden redesign.

Do not:

- change physics formulas or model assumptions;
- modify numerical algorithms or runtime semantics;
- promote/demote derivation confidence;
- alter decision lifecycle/status except where an independently justified existing decision workflow requires it;
- rewrite historical derivation/decision rationale merely because a linked narrative document is removed;
- modify the Xigma manuscript;
- introduce a replacement "grand plan", "progress log", or broad archive that recreates the same duplication under a new name.

Implementation-code edits are allowed only when needed to repair stale documentation references/comments without changing behavior.

## Validation and acceptance criteria

Before declaring the branch complete:

1. Search the entire repository for every removed path/name and verify no unintended live references remain.
2. Run the repository's current documentation/format/staleness checks that exist on this branch.
3. Run the normal lightweight repository check required for documentation-only changes; if the current repo defines `make check` or an equivalent, use it as appropriate.
4. Verify `docs/decisions/INDEX.md` and `docs/derivations/INDEX.md` remain mechanically consistent with their trees.
5. Verify surviving README/AGENTS instructions agree with current code defaults and repository authority.
6. Verify no desired live task was lost solely because `PROGRESS.md` was removed.
7. Verify validation evidence retained is sufficient for current scientific claims; document any uncertain deletion rather than guessing.
8. Inspect the final diff for accidental implementation changes.
9. Delete this handoff file in a final cleanup commit before the PR is marked ready to merge.
10. Confirm the final PR diff against `main` contains no file under `docs/handoffs/` introduced solely for this implementation.

## Deliverables

Persistent outputs expected from this branch are:

- a substantially smaller, internally consistent documentation surface;
- updated `README.md`;
- compacted `AGENTS.md` if warranted by the audit;
- repaired stable references;
- any newly created focused GitHub issues for genuinely live work formerly owned only by `PROGRESS.md`;
- curated validation evidence and, if useful, a small validation retention policy;
- no completed handoff in the final merged tree.

## Out of scope

- physics work;
- numerical/model changes;
- paper editing;
- implementing unrelated open issues;
- broad test-suite redesign;
- inventing new documentation identifiers or taxonomies.

## Open questions and blockers

No physics blocker exists.

The main judgment call is validation retention. When evidence provenance is unclear, prefer keeping the file during the first cleanup pass and record why, rather than deleting the only scientific record. The cleanup can still remove the clearly obsolete narrative/process material independently.
