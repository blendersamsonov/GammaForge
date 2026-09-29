# DER023 Xigma Stage-0 fast path — implementation handoff

Issue: #17 — Implement DER023 narrowband Gaussian Stage-0 fast path

Branch: `issue-17-der023-stage0-fast-path`

Physics authority: **DER023 — Narrowband Gaussian Stage-0 reduction, conservative trajectory bound, and temporal-weight quadrature**

`docs/derivations/derived/DER023-narrowband-gaussian-stage0-bound-and-temporal-quadrature.md`

DER023 is the physics specification. Do not re-derive, weaken, broaden, or silently reinterpret it during implementation. It remains `Status: derived`; this branch does not promote it.

## Goal

Implement the DER023 Stage-0 optimization in the existing Xigma architecture without creating separate CPU/GPU physics implementations.

The branch should deliver, in the supported Gaussian regime:

- a conservative per-particle luminosity upper bound based on the full Gaussian spatial profile and diffraction;
- a cumulative discard certificate;
- temporal-envelope-weighted quadrature for retained particles;
- unchanged DER016 Stage-0 physical moments;
- unchanged Stage 1 and Stage 2 physics;
- NumPy/CuPy parity through the existing particle-chunked execution model;
- an explicit fallback to the current generic trajectory integrator when DER023 does not apply.

The first implementation should be deliberately narrow. Do not extend DER023 heuristically to geometries or pulse semantics it does not cover.

## Current state on `main`

The branch starts from current `main` at `66e876d278f7613fdc50ed2c08218f0100da8551`.

Relevant current behavior:

- `src/gammaforge/engines/xigma/stages.py::integrate_trajectories` uses `n_steps=200` midpoint samples.
- Particle chunking is already implemented. Within each chunk, Stage 0 evaluates particle-by-step arrays using the selected array module.
- Explicit NumPy/CuPy Stage-0/1 execution and host stage boundaries are established by RES083.
- Stage 0 evaluates `laser.intensity_profile` and the DER016 carrier ratio on the same trajectory samples.
- `src/gammaforge/io/bunch.py` already contains the RES045/047/048 frozen-width luminosity/illumination relevance machinery. Those routines are estimates and are not the DER023 conservative bound.
- `src/gammaforge/io/laser.py::SeparableParaxialLaser` evaluates the temporal envelope at `phase_time = phi / omega0`.
- Gaussian and pulse-train temporal envelopes already exist.
- Stage 1 consumes Stage-0 outputs and does not depend on the midpoint rule itself.
- Production source diagnostics use sampled trajectory points and therefore need separate convergence treatment from smooth trajectory moments.

Likely files to inspect first:

- `src/gammaforge/engines/xigma/stages.py`
- `src/gammaforge/engines/xigma/collision.py`
- `src/gammaforge/engines/xigma/schema.py`
- `src/gammaforge/io/laser.py`
- `src/gammaforge/io/bunch.py`
- `tests/test_stage0_delta.py`
- `tests/test_xigma_cupy_stages01.py`
- `tests/test_xigma_source_diagnostics.py`
- `tests/test_laser_pulse_train.py`

## Hard physics boundary

Do **not** silently change the current laser envelope from `phi/omega0` to DER023's retarded/group-delay-compatible coordinate.

DER023 explicitly records this discrepancy as unresolved for production use on the current `SeparableParaxialLaser`.

Therefore:

- implementation may proceed for the mathematical/numerical machinery whose pulse semantics are explicit;
- validation/reference code may use the DER023 retarded-time model directly;
- reusable helpers may be added where they are clearly model-side rather than Xigma-side duplication;
- production-default selection for the current `SeparableParaxialLaser` must remain disabled until the author resolves the envelope convention.

If this blocker is still unresolved when the rest of the implementation is ready, stop at that boundary and report it in the PR. Do not make a physics choice in code.

## Required work

### 1. Build the supported DER023 geometry

For the initial circular, coincident-focus, no-flying-focus Gaussian specialization:

- use the laser axes and GammaForge CGS conventions already exposed by the laser model;
- implement the DER023 metric construction;
- compute the encounter factor `F_i`;
- compute the closed-form worldline quantities `d_i`, `eta_i`, and the exact trajectory curvature `B_i`;
- enforce the DER023 applicability condition for the global curvature bound;
- use
  `B0 = c / (2 z_R)`
  only in the regime proven by DER023.

The code should expose enough intermediate data for validation but should not create a new persistent Stage-0 data model unless required.

### 2. Implement the conservative luminosity bound

Implement the DER023 bank

`D_k -> C_k(eta)`

as laser/prepared data, not as per-particle quadrature.

Requirements:

- fixed monotone `D_k` bank;
- conservative selection of the largest `D_k <= 1 + d_i^2`;
- interpolation must preserve the upper-bound contract;
- compute `U_i` for each particle;
- include macroparticle weights only when forming the physical/cumulative bound, not in the geometric `U_i` itself;
- implement the cumulative discarded-luminosity certificate from DER023.

Do not use RES045's frozen-width `luminosity_weights` as a substitute for this bound. It may be retained unchanged for its existing contract.

The bank spacing and interpolation resolution are numerical parameters to validate, not physics constants to guess.

### 3. Implement temporal-envelope-weighted quadrature

For the DER023 retarded-time model:

- Gaussian temporal envelope: use the exact rescaled Gauss-Hermite nodes/weights from DER023;
- Gaussian pulse train: use a positive composite subpulse rule unless a better equally robust construction is clearly simpler;
- common nodes/weights must be reused for all particles for a given laser preparation;
- analytically map each common retarded-time node to lab time and ballistic position;
- evaluate the spatial profile and any allowed carrier factor on those points;
- accumulate the existing Stage-0 quantities with the quadrature weights.

Do not introduce a custom CUDA kernel. Use the same NumPy/CuPy source path and current chunking.

For unsupported generic positive temporal envelopes, prefer a clean fallback over implementing an unvalidated general Gauss-Christoffel constructor in the first pass.

### 4. Preserve DER016 exactly

The implementation must reproduce the current Stage-0 physical outputs:

- luminosity;
- `a0_shape` / nonlinear shape mean;
- carrier mean;
- nonlinear-shape variance;
- carrier variance;
- nonlinear/carrier covariance.

Use DER023 Section 6 only as the numerical rewrite of DER016. Do not change DER016 definitions or Stage-1 channel semantics.

### 5. Keep Stage 1 and Stage 2 unchanged

No new table coordinate, no new observer-dependent Stage-0 quantity, and no resonance/query change belongs in this task.

Stage 1 should receive equivalent `TrajectorySamples` data regardless of which Stage-0 integration path produced it.

### 6. Keep diagnostics separate initially

Do not automatically switch temporal/spatial histograms to the low-order weighted quadrature.

Histogram bin indicators are discontinuous. The production quadrature may converge rapidly for integrated moments while giving visibly coarse diagnostics.

Initial acceptable choices:

- retain the current diagnostic sampling path when diagnostics are requested; or
- add a separately validated diagnostic sampling rule.

Do not make diagnostic quality an implicit side effect of the Stage-0 performance optimization.

## Validation sequence

Implement validation before choosing production defaults.

### A. Analytic bound checks

Verify directly:

- the spatial majorant used by DER023;
- the completed-square worldline formulas;
- the head-on minimum `B0=c/(2 z_R)` in the supported paraxial regime;
- `U_i` is an upper bound on high-resolution direct trajectory luminosity for a broad deterministic/random trajectory set.

Include trajectories through focus, large transverse misses, head-on incidence, finite crossing angles, and cases close to the supported-domain boundary.

### B. Quadrature convergence

For a Gaussian temporal envelope compare, at minimum, representative node counts such as

`4, 8, 12, 16, 24, 32`

against an over-resolved direct reference.

Measure all Stage-0 moments, not only luminosity.

Compare with equal-node Gauss-Legendre where useful to demonstrate whether the weighted rule actually buys convergence.

Do not set a new default node count until the scenario scan supports it.

### C. Pulse-train convergence

Vary:

- number of subpulses;
- subpulse separation;
- collision timing.

Confirm that the composite weighted rule resolves contributing subpulses without sampling the long empty intervals between them.

### D. Cumulative discard certificate

Construct discarded/retained sets from `U_i`.

After integrating the retained set, compute the DER023 certificate and independently integrate the discarded set in validation cases.

The measured discarded fraction must not exceed the reported upper bound.

### E. Group-delay applicability

Build an independent validation calculation for the full narrowband group-delay model referenced by DER023 and compare it with the retarded-time approximation.

Scan at least:

- pulse duration;
- waist/Rayleigh range;
- chromatic scaling `g_f`;
- timing offsets;
- transverse offsets / relevant spatial positions;
- representative crossing geometry.

This validation informs the acceptance gate. It must not silently mutate the current `SeparableParaxialLaser` semantics.

### F. Repository cross-checks

- NumPy/CuPy use identical quadrature nodes and weights and agree numerically.
- Sampled Gaussian-bunch luminosity converges to DER001 in common circular/head-on cases.
- Existing Stage-1/Stage-2 results are unchanged when supplied equivalent Stage-0 samples.
- Existing generic `LaserField` behavior remains available.

### G. Performance

Only benchmark after a numerical accuracy target is fixed.

Report at least:

- number of laser/intensity evaluations;
- retained-particle fraction after the conservative filter;
- wall time on NumPy;
- wall time on CuPy when CUDA is available;
- peak temporary memory or effective chunk size;
- comparison to the current 200-step midpoint path.

The goal is evidence for the simpler shared vectorized implementation, not a predetermined speedup number.

## Tests

Follow the repository testing policy: add a small number of strong scientific tests.

Likely permanent tests:

- DER023 upper-bound invariant in a compact scenario set;
- quadrature convergence / parity in one or two pinned Gaussian cases;
- NumPy/CuPy agreement for the new path;
- generic-fallback regression;
- discard-certificate regression.

Avoid tests for trivial helper plumbing, private cache details, or obvious input validation.

During implementation use:

`pytest -m fast`

then targeted Stage-0 tests. Before declaring the branch ready, run the normal relevant suite and the repository's broader test command appropriate to the size of the final change.

## Architecture constraints

- Shared data remain CGS-Gaussian.
- Keep `LaserField` as the engine boundary (RES067).
- Do not hard-code the Gaussian formulas redundantly inside generic Xigma logic if the model can own/precompute them.
- Do not add a speculative capability registry or general plugin abstraction.
- Keep public Stage boundaries NumPy as required by RES083.
- Do not create a second CPU implementation and a second GPU implementation.
- Do not modify `PROGRESS.md` as a session log.
- Record a new RESNNN only if implementation makes a genuine durable design choice with alternatives, following the repository decision workflow.
- If core I/O/engine interfaces change, update the corresponding walkthrough notebook and run the notebook build as required by `AGENTS.md`.

## Deliverables

Before the PR is ready for review, the branch should contain:

- the supported DER023 implementation;
- focused scientific tests;
- any reusable validation/benchmark command needed to reproduce the acceptance evidence;
- concise durable validation evidence under `docs/validation/` if the result warrants it;
- documentation/schema changes only where user-facing behavior actually changes;
- no unresolved production wiring that pretends the envelope-semantics blocker is solved.

The final implementation should make it obvious from metadata or code path which Stage-0 integrator was used and why a fallback occurred, without exposing unnecessary internal complexity to ordinary users.

## Out of scope

- revising DER023 physics;
- promoting DER023;
- implementing DER019;
- changing Stage 1 or Stage 2 physical formulas;
- adding a sixth table dimension;
- custom CUDA Stage-0/1 kernels;
- arbitrary non-Gaussian pulse acceleration;
- flying-focus specialization;
- heuristic use of the circular bound for elliptical/astigmatic displaced-focus pulses;
- manuscript edits;
- unrelated cleanup.

## Completion rule

Issue #17 remains the durable task record. This handoff is temporary branch execution context.

Before this PR is merged:

1. move any durable conclusions into code, DER/RES records, validation evidence, or issue/PR discussion as appropriate;
2. remove this handoff file from the branch;
3. verify the final diff against `main` no longer contains `docs/handoffs/der023-stage0-fast-path-2026-09-29.md`.

Do not merge this PR as part of the handoff task.
