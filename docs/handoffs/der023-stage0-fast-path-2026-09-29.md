# DER023 Xigma Stage-0 fast path — implementation handoff

Issue: #17 — Implement DER023 narrowband Gaussian Stage-0 fast path

Branch: `issue-17-der023-stage0-fast-path`

Physics authorities:

**DER023 — Narrowband Gaussian Stage-0 reduction, conservative trajectory bound, and temporal-weight quadrature**

`docs/derivations/derived/DER023-narrowband-gaussian-stage0-bound-and-temporal-quadrature.md`

**DER025 — Narrowband paraxial pulse group delay and chromatic Gaussian-beam scaling**

`docs/derivations/derived/DER025-narrowband-paraxial-pulse-group-delay.md`

DER023 specifies the Stage-0 reduction and numerical applicability gate. DER025 specifies the pulse-front/group-delay physics and resolves the former `phi/omega0` ambiguity. Do not re-derive, weaken, broaden, or silently reinterpret either derivation during implementation. Both remain `Status: derived`; this branch does not promote them.

## Goal

Implement the DER023 Stage-0 optimization in the existing Xigma architecture without creating separate CPU/GPU physics implementations.

The branch should deliver, in the supported Gaussian regime:

- a conservative per-particle luminosity upper bound based on the full Gaussian spatial profile and diffraction;
- a cumulative discard certificate;
- temporal-envelope-weighted quadrature for retained particles;
- unchanged DER016 Stage-0 physical moments;
- unchanged Stage 1 and Stage 2 physics;
- NumPy/CuPy parity through the existing particle-chunked execution model;
- an explicit fallback to the current generic trajectory integrator when DER023 does not apply;
- corrected baseline separable-pulse envelope semantics: retarded time for the envelope, full paraxial phase for the carrier.

The first implementation should be deliberately narrow. Do not extend DER023 heuristically to geometries or pulse semantics it does not cover, and do not invent chromatic focusing information that the laser model does not provide.

## Current state on `main`

The branch starts from current `main` at `66e876d278f7613fdc50ed2c08218f0100da8551`.

Relevant current behavior:

- `src/gammaforge/engines/xigma/stages.py::integrate_trajectories` uses `n_steps=200` midpoint samples.
- Particle chunking is already implemented. Within each chunk, Stage 0 evaluates particle-by-step arrays using the selected array module.
- Explicit NumPy/CuPy Stage-0/1 execution and host stage boundaries are established by RES083.
- Stage 0 evaluates `laser.intensity_profile` and the DER016 carrier ratio on the same trajectory samples.
- `src/gammaforge/io/bunch.py` already contains the RES045/047/048 frozen-width luminosity/illumination relevance machinery. Those routines are estimates and are not the DER023 conservative bound.
- `src/gammaforge/io/laser.py::SeparableParaxialLaser` currently evaluates the temporal envelope at `phase_time = phi / omega0`; DER025 shows that this is not the physical narrowband group-delay coordinate and issue #17 now requires this legacy behavior to be corrected.
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

## Resolved envelope convention

The former `phi/omega0` versus retarded-time/group-delay blocker is resolved by issue #17 and DER025.

Implement the following convention:

- the built-in separable paraxial temporal envelope uses the baseline retarded-time coordinate
  `eta = (t - t_off) - u/c`, up to a consistent internal sign convention that preserves physical pulse timing and pulse-train ordering;
- the full monochromatic paraxial phase remains the carrier phase used by the period-resolved field, including Gouy and wavefront-curvature phase;
- do **not** translate the envelope by `phi/omega0`;
- diffraction-induced narrowband group delay is a separate correction
  `tau_g = u/c + delta_tau_g` governed by DER025;
- `delta_tau_g` requires chromatic focusing information such as
  `g_f = d ln z_R / d ln omega |_(omega0)`, which is not determined by the monochromatic Gaussian beam;
- do not silently choose a default `g_f`. That quantity belongs to the laser/focusing model.

DER023 deliberately uses the retarded-time approximation and its
`Omega_T(Delta tau_*)` criterion to bound the error from neglecting the DER025 correction. The gate is an applicability certificate for that approximation; it is not a reason to return to `phi/omega0`.

For the present built-in baseline laser, production wiring may proceed once the envelope semantics are corrected. A future laser model that explicitly carries DER025 group-delay physics must not be silently replaced by the retarded-time fast path when the DER023 gate fails.

## Required work

### 1. Correct the baseline paraxial envelope semantics

For `SeparableParaxialLaser` and built-in wrappers sharing the same separable pulse convention:

- separate the temporal-envelope coordinate from the monochromatic paraxial carrier phase;
- evaluate the baseline envelope using `eta = (t - t_off) - u/c`, with a consistent internal sign convention;
- retain the full paraxial phase for period-resolved field evaluation;
- remove the interpretation of `phi/omega0` as envelope or group-delay time;
- rename internal `phase_time` concepts to `retarded_time` or `envelope_time` where practical so the API does not preserve the incorrect physical interpretation;
- preserve pulse-train offsets, ordering, and normalization;
- add focused regression tests proving that Gouy/curvature carrier phase does not translate the baseline envelope.

Do not add a universal or guessed `g_f` in this task.

### 2. Build the supported DER023 geometry

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

### 3. Implement the conservative luminosity bound

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

### 4. Implement temporal-envelope-weighted quadrature

For the DER023 retarded-time model:

- Gaussian temporal envelope: use the exact rescaled Gauss-Hermite nodes/weights from DER023;
- Gaussian pulse train: use a positive composite subpulse rule unless a better equally robust construction is clearly simpler;
- common nodes/weights must be reused for all particles for a given laser preparation;
- analytically map each common retarded-time node to lab time and ballistic position;
- evaluate the spatial profile and any allowed carrier factor on those points;
- accumulate the existing Stage-0 quantities with the quadrature weights.

Do not introduce a custom CUDA kernel. Use the same NumPy/CuPy source path and current chunking.

For unsupported generic positive temporal envelopes, prefer a clean fallback over implementing an unvalidated general Gauss-Christoffel constructor in the first pass.

### 5. Preserve DER016 exactly

The implementation must reproduce the current Stage-0 physical outputs:

- luminosity;
- `a0_shape` / nonlinear shape mean;
- carrier mean;
- nonlinear-shape variance;
- carrier variance;
- nonlinear/carrier covariance.

Use DER023 Section 6 only as the numerical rewrite of DER016. Do not change DER016 definitions or Stage-1 channel semantics.

### 6. Keep Stage 1 and Stage 2 unchanged

No new table coordinate, no new observer-dependent Stage-0 quantity, and no resonance/query change belongs in this task.

Stage 1 should receive equivalent `TrajectorySamples` data regardless of which Stage-0 integration path produced it.

### 7. Keep diagnostics separate initially

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

### E. Envelope semantics and DER025 group-delay applicability

First pin the corrected baseline semantics:

- the envelope follows `(t - t_off) - u/c`;
- frequency-independent Gouy/carrier phase shifts do not move that envelope;
- pulse-train ordering and timing remain physically correct after removing `phase_time = phi/omega0`.

Then build an independent synthetic narrowband Gaussian family carrying explicit chromatic information and compare the DER025 full group-delay model with the retarded-time approximation.

Scan at least:

- pulse duration;
- waist/Rayleigh range;
- chromatic scaling `g_f`;
- timing offsets;
- transverse offsets / relevant spatial positions;
- representative crossing geometry.

Include the DER025 isodiffracting discriminator `g_f=0`: the frequency-independent Gouy phase must contribute zero group delay even though the old `phi/omega0` construction would produce a Gouy-derived time offset.

Use this validation to test the DER023 applicability gate. Do not use it to choose a universal production value of `g_f`.

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

- corrected retarded-time envelope semantics for the built-in separable paraxial laser family;
- the supported DER023 implementation;
- focused scientific tests, including carrier/envelope-separation regression coverage;
- any reusable validation/benchmark command needed to reproduce the acceptance evidence;
- concise durable validation evidence under `docs/validation/` if the result warrants it;
- documentation/schema changes only where user-facing behavior actually changes;
- no remaining `phi/omega0` envelope semantics in the corrected built-in separable pulse path;
- no guessed/default chromatic `g_f` introduced by Xigma.

The final implementation should make it obvious from metadata or code path which Stage-0 integrator was used and why a fallback occurred, without exposing unnecessary internal complexity to ordinary users.

## Out of scope

- revising DER023 or DER025 physics;
- promoting DER023 or DER025;
- selecting a universal physical value of `g_f`;
- implementing a general broadband/polychromatic propagation model;
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
