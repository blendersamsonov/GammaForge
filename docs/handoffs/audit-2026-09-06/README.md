# Audit implementation handoffs — 2026-09-06

These are task briefs for separate sessions, not new architecture decisions or a
claim that any fix has shipped. Baseline: `f59f6bf` (minimal kascade + NiceGUI).
Reproduce each finding against your actual HEAD before changing anything.

## Start a session

Give another session this prompt, replacing `NN` and the filename:

> Work on GammaForge handoff NN. Read AGENTS.md, the authoritative documents in its
> prescribed order, docs/handoffs/audit-2026-09-06/README.md, and the named handoff.
> Implement only that handoff's ready scope, with regression tests. Respect its
> dependencies, file ownership, and approval gates; an audit recommendation is not
> approval to change a physics formula or a deliberately deferred design. First
> report the current commit, local changes, and your intended scope. Finish with
> changed files, exact checks/results, unresolved items, and integration notes.

The briefs and [audit baseline](audit-baseline.md) are self-contained: they do not
depend on the earlier audit's temporary files, which are no longer present. The
baseline numbers are historical measurements unless explicitly marked rechecked.

## Session map

| Session | Handoff | Ready scope / gate |
|---|---|---|
| 01 | [Physics and validation](01-physics-and-validation.md) | Harness and verification integrity now; emission formula changes require author review. |
| 02 | [Slice integration contract](02-slice-measures.md) | Define explicit integration measure, then fix producers and projections. |
| 03 | [Xigma runtime robustness](03-xigma-runtime.md) | Bounded spectrum allocation, empty inputs, unsupported outputs, honest warnings. |
| 04 | [Kascade event bookkeeping](04-kascade-events.md) | **Done**: correct negative last-emission times, preserve non-emitter focus convention, protect position reconstruction, and add regression tests. |
| 05 | [Persistence and reproducibility](05-persistence.md) | Inspect now; implement after 02's slice contract and 06's request ownership are settled. |
| 06 | [Input and cache ownership](06-input-and-cache-ownership.md) | Boundary validation and fixed-input cache safety; broader cross-run caching is a proposal only. |
| 07 | [Laser protocol review](07-laser-protocol.md) | Design/reproduction only; no new arbitrary-field physics or FEM integration. |
| 08 | [Developer setup and checks](08-developer-tooling.md) | Dependency contracts, optional test tiers, reproducible check commands and local CI configuration. |
| 09 | [Documentation and guardrails](09-docs-and-guards.md) | **Done**: faster, checkout-scoped doc checks; structural format checks; reconciled current-state facts and navigation map. |
| 10 | [Safe cleanup and deferred choices](10-cleanup-and-choices.md) | Mechanical cleanup after overlapping fixes; UI/API removals need a user decision. |

Start 04, 08, and 09 independently. Session 01 can also start its harness work and
author-review packet; session 02 can start the slice contract. This is a menu of
work streams, not a request to launch all ten agents at once.

## Dependencies and collision rules

Use separate worktrees/branches when sessions run concurrently. Never edit someone
else's existing worktree, delete `.claude/`, or clean up untracked files as part of
these tasks. At handoff creation, `.claude/` and `docs/2-2016.pdf` were pre-existing
untracked paths. Check again rather than relying on that snapshot.

These handoffs are initially uncommitted. A new worktree does not inherit
uncommitted files: include this documentation change in its starting branch, or
give the session the absolute handoff paths in the original checkout to read.

| Shared area | Ownership / integration order |
|---|---|
| `io/results.py`, `io/target.py`, slice plotting and histogram construction | 02 first; 06 applies remaining ownership/validation work afterward; 05 serializes the agreed result. |
| `xigma/stages.py`, `xigma/collision.py` | 03 runtime work, then 06 cache safety; 01's approved physics patch must serialize with both. 01 may edit validation/scripts independently. |
| `kascade/solver.py` versus `kascade/engine.py` | 04 owns solver bookkeeping. 02 owns histogram assembly. Coordinate any final-position assertion that needs adapter edits. |
| `io/calculation.py`, `engines/runner.py` | 06 first; 05 consumes the snapshot contract. |
| `io/formats/hdf5.py` | 02 may make only the minimal measure-preserving compatibility update needed for its contract; 05 owns the subsequent full persistence work. |
| `io/laser.py` | 06 owns boundary validation. 07 is read-only until a reviewed design creates a separately assigned implementation task. |
| `pyproject.toml`, top-level `README.md`, CI/check entry points | 08 owns these; 01 requests symbolic dependencies, rather than editing packaging concurrently. |
| Doc test files and current-state documentation | 09 owns the broad reconciliation. Other sessions provide narrowly scoped updates for their actual changes. |
| Cleanup touching any of the above | 10 goes last or leaves those files to their owner. |

02 → 06 → 05 is the main shared-data landing order. 03 → 06 is the xigma ownership
order. These are landing dependencies, not reasons to postpone read-only inspection
or writing tests in an isolated branch. Rebase and rerun affected tests after a
prerequisite lands; do not resolve conflicts by dropping another session's tests.

One integrating session must coordinate shared edits to `docs/GRAND_PLAN.md`,
`PROGRESS.md`, and the decision/derivation indexes. Before a task changes the design,
update the relevant plan section first, as AGENTS.md requires. Allocate new RES/DER
ids against the latest integration branch; these handoffs reserve none. File
implemented decisions only for choices actually built, and preserve superseded
reasoning according to the decision lifecycle rules.

## Common guardrails

- Preserve CGS/pint boundaries, pure float kernels, the `Engine` contract, schema-led
  forms, Calculate gating, relative bunch weights, and explicit `N_e` scaling.
- No `gammaforge.core`, `Results.cfg`, generic adapter framework, capability
  negotiation, mutable engine configuration, speculative backend, or LAN executor.
- A paper/code disagreement is BLOCKING for the affected physics change. Bring
  evidence to the author; do not tune constants, rename the discrepancy an
  approximation, regenerate goldens, or loosen tolerances to make it disappear.
- Prefer deterministic regression tests. MC histogram mass is compared to the
  represented photon weights, not forcibly normalized to a deterministic yield.
- “No in-repo caller” is not proof that a public notebook API is safe to delete.
- No commits, pushes, PRs, remote CI changes, or dependency installation outside
  the assigned environment are implied by these briefs. Follow the user's actual
  authority and the environment's approval mechanism.

## Done and integration checks

Each session supplies: the original failure, a regression that detects it, the
smallest implementation, exact test commands/results, documentation impact, and
any unresolved compatibility or author decisions. A design-only handoff finishes
with a reviewed proposal/question, not pretend implementation or green physics.

After integration, run the full pytest suite; the explicit production-validation
tier from 01; the symbolic tier when installed; core-only and GUI-enabled setup
checks from 08; and the opt-in browser test when GUI behavior or dependencies
changed. Compare failures to the [baseline](audit-baseline.md). Do not treat a green
unit suite as proof of scientific correctness. Refresh graphify after landing code
per repository instructions, keeping generated graph files uncommitted.

Keep completion state in the normal code/decision records and current open threads
in `PROGRESS.md`, not a growing session log here. These handoffs are a dated
starting snapshot; archive or remove them once their work has been absorbed.
