# AGENTS.md — GammaForge Repository Instructions

## What this project is

GammaForge computes properties of Compton photons produced by an electron-bunch /
laser-pulse interaction. This repo is a **ground-up rebuild** of a predecessor at
`/home/alexander/Work/Code/ComptonSuite` (also historically called "GammaForge") — kept
around solely as a historical reference and a source of golden validation data, never
as code to extend in place.

**The authoritative documents, in the order to read them:**

1. **`docs/GRAND_PLAN.md`** — the architecture and phase plan. Versioned (`vX.Y`), with
   its own changelog of *design* decisions. If something here conflicts with the plan,
   the plan wins; update the plan first, then code.
2. **`PROGRESS.md`** — the execution log. What's actually been built, session by
   session, and current phase status. Check this before assuming what exists.
3. **`DECISIONS.md`** — implementation-level decisions with rejected alternatives, in
   the plan's own documentation-discipline spirit (goal #8). Read its header once: it
   has a load-bearing convention (backticks = resolves right now; *italics* = a
   hypothetical/rejected/future name) that a doc-staleness test enforces.

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
  over from the predecessor (see `DECISIONS.md` D013–D015).
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
- **`DECISIONS.md` entries:** add one when you make a real implementation choice with a
  rejected alternative, *after* it's built (not as a promise). Follow its
  backticks-vs-italics convention — `tests/test_doc_staleness.py` checks it.
- **`PROGRESS.md`:** append a dated section per session/chunk of work; don't rewrite
  history. Link back to `GRAND_PLAN.md`'s changelog for plan changes instead of
  duplicating rationale.
- **Old repo (`ComptonSuite`):** reference only, and only through
  `validation/make_references.py` (once it exists) for golden snapshots. When the plan
  says "port," it usually means port the *algorithm and hard-won constants*, not the
  code verbatim — e.g. the chunking utility (§4.2) explicitly replaces three
  inconsistent old implementations with one, not a copy of any of them.
- **Graphify:** once there's real code to search across, use the `graphify` skill for
  codebase-wide queries instead of ad hoc grepping — not needed yet at Phase 0 scaffold
  size, but worth reaching for once `gammaforge.io`/`engines` have real content.

## Current status

See `PROGRESS.md` for what phase is active and what's actually landed. Don't assume
anything beyond Phase 0 scaffold exists without checking it first — most of the module
files named in `GRAND_PLAN.md` are deliberately not created yet (see `DECISIONS.md`
D003).
