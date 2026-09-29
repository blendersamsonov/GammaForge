# AGENTS.md — GammaForge Repository Instructions

## What this project is

GammaForge computes properties of Compton photons produced by an electron-bunch /
laser-pulse interaction. It originated as a ground-up rebuild of ComptonSuite (also
historically called "GammaForge") and is now completely independent: building, testing,
validation, and development require no predecessor checkout. Historical documents retain
that origin as design provenance.

Note: `CLAUDE.md` is a symlink to this file. Edit the real file, not the symlink.

**Repository:** `blendersamsonov/GammaForge`  
**Primary branch:** `main`

GitHub `main` is the durable project knowledge base. Before planning work, assigning a
derivation id, or constructing an implementation handoff, inspect the current repository
state rather than relying on an older checkout, chat summary, or remembered layout.

**Authority is split by artifact type; there is no single document that overrides every
other domain:**

1. **`docs/GRAND_PLAN.md`** owns the architecture and phase plan. It is versioned
   (`vX.Y`) and carries the design-plan changelog. Within architecture and phase scope,
   reconcile changes against the plan before coding.
2. **`PROGRESS.md`** owns **current implementation state and open threads only**. It is
   not a session log: Git history and GitHub issues preserve the narrative and task
   history.
3. **`docs/decisions/INDEX.md`** is the entry point for durable implementation/design
   decisions (`RESNNN`). One file lives under
   `docs/decisions/{lifecycle}/{class}/`; `docs/decisions/README.md` defines the
   lifecycle, classification, header format, and the backticks-vs-italics convention
   enforced by the documentation tests. IDs are permanent addressing: never renumber or
   reuse them.
4. **`docs/derivations/INDEX.md`** is the entry point for durable physics derivations
   (`DERNNN`). One file lives under `docs/derivations/{status}/DERNNN-*.md`.
   Derivation status is a confidence pipeline, not an implementation lifecycle:
   `derived` (worked out, not yet author-reviewed) → `validated` (domain-expert
   reviewed) → `verified` (checked against code, an independent method, a pinned
   analytical limit, or equivalent evidence). See `docs/derivations/README.md`.
5. **`docs/validation/`** owns durable scientific validation evidence. Tests and
   validation records do not automatically promote a derivation; confidence-state changes
   are separate, explicit actions.
6. **Merged code and repository documentation** own the finished implementation. GitHub
   issues and pull requests own work tracking and implementation history.

For physics, current GammaForge derivations are the repository specification. Do not
silently replace a `DERNNN` result with a formula from an old chat, local note, manuscript,
or historical code. If current code conflicts with the relevant derivation, surface the
conflict and resolve it explicitly rather than re-deriving physics inside an implementation
task. A `derived` result remains unreviewed even if code implementing it already exists.

### Derivation, issue, and handoff workflow

- **New derivations:** when a derivation has been developed and the derivation workflow is
  invoked, inspect current `main`, avoid duplicates, assign the next unused `DERNNN`,
  create it under `docs/derivations/derived/`, update
  `docs/derivations/INDEX.md`, run the relevant derivation-format checks, and commit the
  documentation change to `main`. Derivation-only work does **not** authorize code,
  implementation-test, decision, `PROGRESS.md`, or manuscript changes, and it never
  promotes the new result beyond `Status: derived`.
- **Implementation issues:** GitHub issues are the durable record of work that still needs
  to happen. When a derivation implies code, diagnostic, or validation work, the issue
  should reference the `DERNNN` and treat it as the physics specification instead of
  asking the coding agent to rediscover the physics. Do not create an implementation issue
  for a purely explanatory derivation or work that is already complete.
- **Handoffs:** a handoff is temporary execution context for one implementation branch and
  draft pull request, not permanent project knowledge. Create the implementation branch
  from current `main`; the handoff under `docs/handoffs/` is the first substantive
  commit; implementation follows in later commits. If an issue exists, it remains the
  durable task record and the handoff/PR should reference it.
- **Handoff completion:** before the implementation PR is merged, move any durable results
  into their proper homes (code, `DERNNN`, `RESNNN`, validation evidence, documentation)
  and delete the handoff from the branch so it does not land on `main`. The PR and commit
  history preserve it. Do not merge merely because a handoff exists; merging requires the
  normal explicit authorization/review for the task.
- **Legacy handoffs on `main`:** several older files under `docs/handoffs/` predate this
  workflow. Treat them as historical leftovers, not as precedent for keeping new completed
  handoffs on `main`; do not clean them up as part of unrelated work.

### Manuscript boundary

The authoritative manuscript is the current `main` branch of the separate private
repository `blendersamsonov/Xigma-Paper`. That repository owns LaTeX manuscript text,
bibliography, manuscript figures/material, paper notes, and paper-only issues; it is not a
replacement authority for GammaForge physics. When manuscript text conflicts with a current
GammaForge derivation, use the derivation as the physics specification and update or flag
the manuscript in `Xigma-Paper` rather than changing GammaForge physics to match stale
paper text.

Work that requires new algorithms, diagnostics, simulations, validation infrastructure, or
new numerical evidence belongs in GammaForge (normally as a GammaForge issue). Work that is
only manuscript writing, organization, bibliography, or presentation of already established
results belongs in `Xigma-Paper`. If paper work depends on new GammaForge computation, keep
the implementation/validation task here and make the paper task depend on its result.

## Rules that are easy to violate without reading the whole plan

These are the ones a fresh agent is most likely to get wrong by pattern-matching on
"reasonable Python architecture" instead of this project's specific, hard-won
constraints (`GRAND_PLAN.md` §1 has the full table with provenance — P1–P15):

- **One unit system, CGS-Gaussian, in every shared dataclass** — and **dimensioned types
  at the engine boundary**: each dimensioned field is a pint `Quantity`, stored
  canonically in CGS, so an engine converting to its own internal system (kascade is SI)
  gets a checked conversion rather than a hand-written factor. Engines unpack once at
  `run()`; kernels only ever see floats. Bulk per-particle arrays are *not* wrapped —
  `Bunch` declares its units as data and converts through a scale factor. Never introduce
  a second internal unit system "for convenience", and note there is **no coordinate
  normalization** anywhere: xigma works in CGS directly, `k0_las` scaling is not carried
  over from the predecessor (see RES013–RES015 in `docs/decisions/`).
- **No `gammaforge.core` package.** The shared layer is `gammaforge.io` — yes, that
  name is odd, it's kept for continuity with the predecessor. Don't add an intermediate
  layer between `io` and `engines`.
- **No mutable `Config` object on an engine.** Engine numeric knobs live in the typed
  parameter schema (`Parameters`/`FieldSpec`), validated, not as attributes on an
  adapter.
- **Engines never branch the GUI, and the GUI never touches engine internals.** The
  import boundary is mechanically enforced (or will be, once CI exists) —
  `gammaforge.gui` may only import `Engine.run()`/`Results`/schema and other `io`
  helpers (drawing module, YAML/HDF5 I/O), never `engines/*/stages.py` or any
  engine-specific stateful facade (e.g. xigma's `Collision`).
- **The laser is typed against the `LaserField` protocol, not `GaussianParaxialLaser`
  directly**, anywhere an engine consumes it. `GaussianParaxialLaser` is today's only
  implementation, not a hardcoded assumption — see P15. A sibling project,
  `~/Work/Code/Spectral-FEM-Fields` (C++, Python bindings planned), is expected to
  eventually provide a second implementation for arbitrary non-Gaussian pulses. Nothing
  to build there yet; just don't write engine code that assumes `GaussianParaxialLaser`
  specifically when it only needs `a0_profile`/`field`/`active_region`.
- **Recompute costs are engine-declared data (`QUERY_ONLY`/`REUSE_INTERMEDIATES`/
  `FULL_RERUN`), not a global stage-tied enum.** No engine is real-time except the
  analytical panel; everything else is Calculate-gated. Don't wire up live
  keystroke-triggered requeries — that's the exact failure mode (`_MAX_LIVE_N_ENERGY_*`
  hardcap saga) the plan explicitly rejects.
- **No speculative abstraction.** P6/P7/P9/P10/P11 each name a specific piece of
  machinery the predecessor built, then deleted, then the plan explicitly rejects
  rebuilding (generic spec/adapt-to-model framework, capability registry/protocol,
  `Results.cfg` back-references, derived-property duplication). If you find yourself
  about to build one of these, re-read the relevant P-row first.

## Working conventions

- **Tests:** `pytest` from the repo root (or `source .venv/bin/activate && pytest`).
  Keep `pytest` green — Phase exit criteria in `GRAND_PLAN.md` §11 are the actual
  definition of "done" for a phase, not just "tests pass."
  New tests should primarily protect scientific equations, physical invariants, numerical
  conservation or convergence, independent implementation/reference agreement, or a
  previously observed bug that could silently corrupt scientific output. Do not routinely
  add tests for trivial validation branches, private implementation details, defaults,
  getters, documentation structure, or obvious Python/framework behavior. Prefer a small
  number of strong invariant and end-to-end tests over exhaustive micro-tests.
  The test suite is structured into execution tiers (RES075):
  - `pytest -m fast` (or `pytest --tier=fast`): **~20s fast agentic loop** running Tier 0
    (contracts, schemas, units, formats, persistence) and Tier 1 (fast component physics,
    optics, runner, GPU logic). Default to this during iterative coding.
  - `pytest --tier=tier0`: **~5s ultra-fast check** of contracts, schema, CGS units,
    formats, and persistence.
  - `pytest`: **~1.1m default check** running Tier 0, Tier 1, and Tier 2 (numerical integration,
    trajectory tracking, validation harness). Heavy Tier 3 tests (>30s) are automatically
    deselected on broad sweeps.
  - `pytest --run-heavy` (or `pytest --tier=all`): **~6m full run** including heavy Tier 3
    Monte Carlo validations, 16× quadrature refinement, and external SymPy proofs. Run before
    phase exits or major commits.
  - Specific files or tests run directly without flags (e.g. `pytest tests/test_analytical.py`).
- **Decisions:** file one under `docs/decisions/implemented/<class>/` when you make a
  real implementation choice with a rejected alternative, *after* it's built (not as a
  promise) — or `docs/decisions/proposed/<class>/` for something reviewed but not yet
  built. Append; ids are never renumbered or reused. Add the row to
  `docs/decisions/INDEX.md`. Follow the backticks-vs-italics convention and the
  Problem/Decision/Alternatives-considered/Rationale/Consequences skeleton in
  `docs/decisions/README.md` — `tests/test_doc_staleness.py` and
  `tests/test_decision_format.py` both check it. When an entry is superseded, put a
  pointer at the top of both, then move the old one to `docs/decisions/archived/<class>/`
  **verbatim** plus an `Archived:` line — editing its reasoning would falsify what was
  decided at the time.
- **Citing a decision from code:** a comment/docstring may cite `(RESNNN)` bare — no path
  needed, ids are permanent. Keep it a pointer: one clause of current behavior plus the
  citation. If the surrounding comment grows past that — a derivation of *why*, rejected
  alternatives, historical numbers — that's a sign the content belongs in the decision
  file, not the code; move it there and trim what's left to the pointer (RES056; see
  `docs/decisions/README.md`'s *Citing from code*).
- **`PROGRESS.md`:** **do not append a session log.** Edit the phase table and the open
  threads in place, and delete what stopped being true — git holds the narrative. Anything
  merely *done* belongs in the code and the commit message, not here.
- **Historical provenance:** decisions and older plan text may describe ComptonSuite and
  algorithms first developed there. They are history, not a dependency or an instruction
  to consult another checkout. Current behavior must be justified by this repository's
  code, derivations, decisions, and validation suite.
- **No legacy code:** this is a single-user project and API changes are acceptable. When
  an implementation or interface is superseded, delete the obsolete code, aliases, shims,
  compatibility branches, and dead tests instead of retaining them for history or backward
  compatibility. Git and the archived decision record preserve history.
- **Validation:** `python -m gammaforge.validation.run` is the suite entry point (it runs
  what is runnable and says what it skipped); `gammaforge.validation.scenarios.SCENARIOS`
  is the shared bank — iterate it, don't hardcode a scenario name in a runner.
- **Postmortems:** `docs/postmortems/README.md` has the criteria for when a bug is worth
  a postmortem (subtle + systemic + costly to rediscover) rather than just a fix — distinct
  from a decision, which records a deliberate choice rather than a failure.
- **Graphify:** the repo is now big enough that a codebase-wide question is usually
  better answered from the knowledge graph than by grepping. **The graph is not
  committed** — `graphify-out/` is gitignored, so on a fresh clone it does not exist and
  must be built once (invoke the `graphify` skill on the repo root). Once
  `graphify-out/graph.json` is present: `graphify query "<question>"` for context,
  `graphify path "<A>" "<B>"` for how two things relate, `graphify explain "<concept>"`
  for one node. Each returns a scoped subgraph, far smaller than `GRAPH_REPORT.md` or raw
  grep output; read the full report only for a broad architecture pass. Re-run
  `graphify update .` after landing code — it is AST-only and costs nothing.
- **Human Walkthrough Notebooks (`notebooks/`):** The human architect uses the modular
  notebooks in `notebooks/` to learn, explore, and verify the codebase. They are authored as
  `# %%` scripts (`notebooks/*.py`) paired with Jupyter `.ipynb` notebooks. When modifying
  core data structures (`gammaforge.io`), engine interfaces (`gammaforge.engines`), or
  validation pipelines, **you must update the corresponding script in `notebooks/` and run
  `python tools/build_notebooks.py --run`** so that the human mental model and interactive
  examples stay in sync with the codebase.

## Current status

**`PROGRESS.md` is the authority — read it rather than a summary here.** This section
deliberately carries only what does not change between phases:

- Most module files named in `GRAND_PLAN.md` are **deliberately absent** until their phase
  lands (RES003 in `docs/decisions/`). Check before assuming one exists; the knowledge graph
  (`graphify query`) answers this faster than grep.
- Backend support and defaults are documented in `docs/ALPHA.md` and `PROGRESS.md`.
  Stage 0/1 device execution retains NumPy public stage boundaries (RES083).
  `numba` remains gated; numerical backend agreement does not close independent physics validation.
- **§9.2 and §9.3 emission kernel factors are implemented** (DER004, DER005, DER006, RES060),
  and `ELLIPTICITY_IS_NOOP` and `EMISSION_IS_HEAD_ON` are `False`. The lab-frame per-particle
  velocity projection is author-approved (RES060) and formula-checked against Eq. `udef`.
  **Independent arbitrary-angle emission validation remains open** (tracked in `PROGRESS.md`),
  so production validation reports angular/crossing-angle coverage blockers rather than claiming
  scientific closure. `docs/derivations/` holds the mathematical derivations and verifications.


## Behavioral guidelines to reduce common LLM coding mistakes. 

Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
