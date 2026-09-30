# AGENTS.md — GammaForge repository instructions

GammaForge computes Compton photons from electron-bunch/laser-pulse interactions.
It builds and validates independently of its predecessor, ComptonSuite. `CLAUDE.md`
is a symlink to this file; edit this file, not the symlink.

**Repository:** `blendersamsonov/GammaForge`; primary branch: `main`.
Inspect current GitHub `main` before planning work, assigning IDs, or preparing a
handoff. Do not infer current behavior from an old checkout or historical prose.

## Where knowledge lives

- `docs/derivations/INDEX.md` maps permanent `DERNNN` physics specifications. A
  `derived` result is unreviewed even when implemented; `validated` means expert
  review; `verified` requires a separate check. Follow the format and confidence
  rules in `docs/derivations/README.md`. Resolve a code/derivation conflict
  explicitly; do not silently substitute a formula from an old note or paper.
- `docs/decisions/INDEX.md` maps permanent `RESNNN` implementation and design
  choices. Records live directly in `archived/`, `implemented/`, `proposed/` or
  `rejected/`; their `Type:` header records the decision kind. Follow
  `docs/decisions/README.md` for lifecycle moves, types, citations and formatting.
  Never reuse an ID or rewrite archived reasoning.
- `docs/validation/` contains curated scientific evidence. Passing tests does
  not by itself promote a derivation or close scientific acceptance.
- Merged code and the root README own current behavior; GitHub issues own unfinished
  work; PRs and Git history own implementation/process history.
- The separate private `blendersamsonov/Xigma-Paper` repository owns manuscript
  prose, figures, bibliography and paper-only issues. Current GammaForge
  derivations take precedence when manuscript text conflicts with physics here.
  New algorithms, diagnostics, validation infrastructure and numerical evidence
  belong in GammaForge; paper presentation of established results belongs there.

## Derivations, issues and handoffs

- When the derivation workflow is requested, inspect current `main`, avoid
  duplicates, assign the next unused `DERNNN`, add it under
  `docs/derivations/derived/`, update the index, run format checks and commit the
  documentation change to `main`. Derivation-only work does not authorize code,
  tests, decisions, manuscript edits or confidence promotion.
- File an implementation issue only for work that remains. Reference its `DERNNN`
  as the physics specification where applicable.
- Create an implementation branch from current `main`. Its temporary handoff in
  `docs/handoffs/` is the first substantive commit; implementation follows in
  later commits. Reference the durable issue in the handoff and PR. Move durable
  results into code, DER/RES records and validation evidence, then delete the
  handoff before merge. Merge requires the normal explicit review/authorization.

## Architecture and physics invariants

- Shared dataclasses in `gammaforge.io` use one canonical CGS-Gaussian unit
  system. Dimensioned fields are pint `Quantity` values; engines convert and
  unpack once at `run()`, and kernels receive floats. `Bunch` bulk arrays stay
  raw with declared units. There is no coordinate normalization (RES013–RES015).
- There is no `gammaforge.core` package. `gammaforge.io` is the shared layer.
  Keep kernels pure and engine caching in the thin facade. Engine numeric knobs
  belong in typed `Parameters`/`FieldSpec`, never a mutable adapter `Config`.
- An engine consumes the `LaserField` protocol, not a concrete Gaussian laser,
  except where a particular analytical method explicitly requires one (RES067).
  Gaussian laser geometry uses `R_y(theta_xz) R_x(theta_yz)`; the transported
  focus and polarization axes follow that rotation.
- The GUI renders schemas, calls the public runner/`Engine.run()`, and renders
  `Results`; it does not inspect engine stages or compute physics. Engines declare
  recompute costs as data
  (`QUERY_ONLY`, `REUSE_INTERMEDIATES`, `FULL_RERUN`). Calculations are gated by
  **Calculate**; analytical estimates alone can preview immediately.
- Do not add speculative capability registries, generic spec adapters, `Results.cfg`
  back-references or duplicated derived properties. Superseded interfaces and
  compatibility shims should be removed rather than maintained as legacy code.
- Xigma and analytical are the main engines; Kascade is a minimal independent
  comparison method and Delta is a validation reference. Scientific acceptance
  requires independent physics checks, not only backend agreement. Use current
  DER records for emission formulas and `docs/validation/` for measured scope.

## Development and evidence

- Run `pytest -m fast` during iterative coding, `pytest` (or `make check`) for
  the default suite, and `pytest --run-heavy` before a phase exit or major
  commit. `pytest --tier=tier0` covers contracts, schemas, units and formats.
  Direct targeted files run without flags. Add tests for scientific invariants,
  independent agreement, convergence or a previously observed corruption bug;
  avoid routine micro-tests of trivial branches and defaults.
- `python -m gammaforge.validation.run` is the suite entry point; its production
  selector reports missing scientific coverage as blockers. Iterate
  `gammaforge.validation.scenarios.SCENARIOS` in runners rather than hardcoding
  one scenario. New validation records must state settings, evidence and limits.
- Write a `RESNNN` for a real choice with a rejected alternative after it is
  built, or a `proposed` record for a reviewed unbuilt choice. Put the file directly
  under its lifecycle folder, set its `Type:` tag and update the index.
  Code comments cite bare IDs in one clause; reasoning lives in the record.
- When changing `gammaforge.io`, engine interfaces or validation pipelines,
  update the corresponding `notebooks/*.py` walkthrough and run
  `python tools/build_notebooks.py --run`.
- For codebase-wide questions, use the existing ignored `graphify-out/graph.json`
  with `graphify query`, `path` or `explain`; build it once on a fresh clone and
  run `graphify update .` after code changes. The graph is a navigation aid,
  never the authority over code and DER/RES records.
