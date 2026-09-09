# RES075 — Test suite tiering and execution markers

Status: implemented
Class: testing

## Problem

The full test suite execution time reached ~11.3 minutes across 35 test files and over
600 tests. Over 85% of this execution time was concentrated in just 6 files executing
compute-intensive macroparticle Monte Carlo sweeps, 16x solid-angle quadratures across the
entire scenario bank, SymPy symbolic verifications, and fine trajectory tracking. Running
the entire test suite during rapid agentic and local development iterations created a severe
bottleneck, even though most architectural, contract, unit, and formatting checks do not
depend on full-scale physics simulations.

## Decision

Introduce structured architectural and speed tiers into `tests/conftest.py` and
`pyproject.toml`, registered as pytest markers:

- *tier0* (*fast*): Unit contracts, schema validation, CGS unit algebra, file formats,
  boundary rules, and doc/format linters (< 1s per test, ~5s total).
- *tier1* (*fast*): Fast component physics, laser field evaluations, runner mechanics,
  and GPU sampler/polarization logic (< 5s per test, ~27s total).
- *tier2*: Trajectory tracking, moderate-resolution multi-stage depositions, and validation
  harness checks (< 30s per test, ~45s total).
- *tier3* (*heavy*): Brute-force Monte Carlo validations, 16x scenario-bank quadrature
  refinements, and external SymPy symbolic verifications (> 30s per test, ~9m total).

In `tests/conftest.py`, heavy *tier3* tests are deselected by default on broad test sweeps,
bringing bare `pytest` runs down from ~11.3 minutes to ~1.2 minutes (~9× speedup).
Rapid agentic loops can invoke `pytest -m fast` (~22s, ~30× speedup) or `pytest --tier=tier0` (~5s).
The slow tests remain fully executable on demand via *--run-heavy*, *--tier=tier3*,
*--tier=all*, or by directly targeting a specific heavy file or test path.

## Alternatives considered

- *addopts = "-m 'not heavy'" in pyproject.toml*: Pytest combines command-line *-m* options
  with logical AND, so running `pytest -m heavy` would resolve to `not heavy and heavy` and
  select zero tests. Hooking collection modification in `tests/conftest.py` avoids this trap.
- *Separate tests/ directory trees (e.g. unit/ vs integration/ vs physics/)*: Reorganizing
  the directory structure would break established file paths, existing references in
  `GRAND_PLAN.md`, and decision citations. Pytest markers provide semantic tiering without
  disturbing filesystem paths.
- *Hardcoded skips with environment variables*: Hardcoding environment skips hides test
  presence from standard test runners and makes selective execution clumsy.

## Rationale

Decoupling rapid contract validation from heavy numerical convergence tests dramatically
speeds up developer and agent feedback loops while preserving complete physics verification.
Because function-level markers take precedence over file-level markers, mixed files like
`tests/test_analytical.py` and `tests/test_stage1_stage2.py` can expose their fast structural
checks to *fast* runs while isolating long-running Monte Carlo quadratures to *tier3*.

## Consequences

Default `pytest` executions run Tier 0, Tier 1, and Tier 2 (~1.2 minutes). Full release or
phase exit validation requires `pytest --run-heavy` or `pytest --tier=all` to run the full
~11.3 minute suite including all Monte Carlo and symbolic proofs.
