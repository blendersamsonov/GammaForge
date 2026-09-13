# AGENTS.md — GammaForge Repository Instructions

## What this project is

GammaForge computes properties of Compton photons produced by an electron-bunch /
laser-pulse interaction. This repo is a **ground-up rebuild** of a predecessor at
`/home/alexander/Work/Code/ComptonSuite` (also historically called "GammaForge") — kept
around solely as a historical reference and a source of golden validation data, never
as code to extend in place.

Note: `CLAUDE.md` is a symlink to this file. Edit the real file, not the symlink.

**The authoritative documents, in the order to read them:**

1. **`docs/GRAND_PLAN.md`** — the architecture and phase plan. Versioned (`vX.Y`), with
   its own changelog of *design* decisions. If something here conflicts with the plan,
   the plan wins; update the plan first, then code.
2. **`PROGRESS.md`** — **current state and open threads only**, deliberately short. It is
   not a session log: `git log` is, and it does not go stale. Read it for what works right
   now and what is unfinished or waiting on the author.
3. **`docs/decisions/INDEX.md`** — start here for implementation-level decisions, in
   the plan's own documentation-discipline spirit (goal #8). One file per decision, under
   `docs/decisions/{lifecycle}/{class}/`; `docs/decisions/README.md` has the full
   lifecycle (`proposed`/`implemented`/`rejected`/`archived`) and classification system,
   the header format, and a load-bearing convention (backticks = resolves right now;
   *italics* = a hypothetical/rejected/future name) that `tests/test_doc_staleness.py`
   enforces on every `proposed`/`implemented`/`rejected` entry, plus a structural format
   check in `tests/test_decision_format.py`.
   **Ids are an addressing scheme** — ~124 code comments cite `RESNNN` bare (not a path), so
   entries are never renumbered and never deleted, only archived.
4. **`docs/derivations/INDEX.md`** — the long-form physics derivations that back specific
   code, one file per result under `docs/derivations/{status}/DERNNN-*.md`. Unlike a
   decision, a derivation's status is a **confidence pipeline**, not a build lifecycle:
   `derived` (worked out, not yet reviewed) → `validated` (a domain expert checked the
   algebra) → `verified` (checked against code — a test, a closed-form limit, an
   independent method). See `docs/derivations/README.md`. Cite `DERNNN` bare from code and
   decisions the same way as a `RESNNN` decision id.

The physics authority is the paper draft at `~/Work/Papers/2026/Compton-Numerics`. It
is in flux. **If code and paper disagree, that is BLOCKING — stop and flag it, don't
guess.** (See `GRAND_PLAN.md` §0 and P14 for the exact three-way handling of
paper-code discrepancies vs. genuinely-absent derivations.)

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
  The test suite is structured into execution tiers (RES075):
  - `pytest -m fast` (or `pytest --tier=fast`): **~20s fast agentic loop** running Tier 0
    (contracts, schemas, units, formats, doc linters) and Tier 1 (fast component physics,
    optics, runner, GPU logic). Default to this during iterative coding.
  - `pytest --tier=tier0`: **~5s ultra-fast check** of contracts, schema, CGS units,
    formats, and doc/decision guards.
  - `pytest`: **~1.2m default check** running Tier 0, Tier 1, and Tier 2 (numerical integration,
    trajectory tracking, validation harness). Heavy Tier 3 tests (>30s) are automatically
    deselected on broad sweeps.
  - `pytest --run-heavy` (or `pytest --tier=all`): **~11m full run** including heavy Tier 3
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
- **Old repo (`ComptonSuite`):** reference only, and only through
  `src/gammaforge/validation/make_references.py` for golden snapshots. It runs the old
  code in a **subprocess** — both repos install a package called `gammaforge`, so they
  cannot share a process — and needs the `OLD_REPO` / `OLD_REPO_PYTHON` environment
  variables. Regenerating goldens is deliberate and manual; the committed snapshots under
  `validation/references/data/` are what the suite compares against. When the plan says
  "port," it usually means port the *algorithm and hard-won constants*, not the code
  verbatim — e.g. the chunking utility (§4.2) explicitly replaces three inconsistent old
  implementations with one, not a copy of any of them.
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
