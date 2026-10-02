# Handoff — Issue #5: deterministic Gaussian analytical engine

## Goal

Implement issue #5 on this branch: extend the current `AnalyticalEngine` into the deterministic/semi-analytical Gaussian hierarchy specified by DER019, while preserving its role as a fast, independent validation/optimization engine alongside Xigma.

The implementation should let Gaussian problems progress from closed or low-dimensional reductions to focused/flying deterministic source quadrature without drawing Gaussian macroparticles. The public interface should hide most of that hierarchy behind automatic per-observable model selection, while still allowing expert pinning for validation.

This handoff is an execution plan, not a physics source. Do not re-derive or alter the formulas in DER019 during implementation.

## Authority

Durable task record:

- GitHub issue #5 — `Implement DER019 deterministic Gaussian analytical spectra`.

Primary physics specification:

- `docs/derivations/derived/DER019-deterministic-gaussian-source-reductions.md`.

Established physics that DER019 composes with:

- DER001 — Gaussian luminosity overlap integral.
- DER002 — flying focus.
- DER009 — reduced delta-resonance kernel.
- DER011 — closed-form angle-integrated linear Thomson spectrum.
- DER012 — physical locally transverse crossing-angle dipole kernel.
- DER013 — direction-dependent Doppler factor `D`.
- DER015 — observer-dependent ponderomotive factor `Q`.
- DER016 — carrier-weighted trajectory moments.
- DER017 — second-order spectral-moment reconstruction.

Relevant durable architecture/behavior decisions on current `main`:

- RES018 — the engine protocol is `name/schema/supported_outputs/recompute_costs/run`; controls belong in the engine `Parameters` schema.
- RES036 — the current sampled energy grid is normalized back to analytical total yield.
- RES039 — general Gaussian overlap quadrature is the analytical total-yield path.
- RES041 — crossing-angle overlap is already supported for yield.
- RES042 — luminosity-weighted mean intensity is computed, not guessed.
- RES043 — analytical overlap already has explicit cost tiers; exact 2D transverse quadrature is opt-in.
- RES049 — the current nonlinear-spread bracket is empirical.
- RES052 — `COLLIMATED_SPECTRUM` is the sole `(E, theta_x, theta_y)` output kind.
- RES053/RES054 — nonlinear physics uses cycle-averaged `<a^2>`, not peak-field `a0^2`.
- RES063 — result/request metadata is persisted and should remain reproducible.
- RES090 — observer-dependent ponderomotive incidence is applied in Stage 2.

If implementation uncovers a conflict between DER019 and a verified derivation, stop at that boundary and report it. Do not resolve physics implicitly in code.

## Branch baseline

This branch was created from `main` commit:

`66e876d278f7613fdc50ed2c08218f0100da8551`

No implementation changes belong in the first commit. This handoff file must be the only substantive change in the initial commit.

## Current state on main

### Analytical engine API

`src/gammaforge/engines/analytical/engine.py` currently:

- accepts no constructor configuration;
- publishes a `Parameters` schema;
- supports only `TOTAL_YIELD` and angle-integrated `SPECTRUM`;
- evaluates total yield through `overlap_yield`;
- obtains a luminosity-weighted mean peak-amplitude-squared from `overlap_mean_a0_sq`;
- converts it to cycle-averaged `ahat` using the laser's `cycle_average_factor()`;
- evaluates the spectrum with one mean `ahat`;
- returns scalar diagnostics in `Results.model_specific`.

Do not force a constructor-level `AnalyticalEngine(mode=...)` API merely because the design discussion used that notation. On the current architecture, `mode` and expert model pinning should normally be expressed as analytical `Parameters` fields so the Engine protocol, GUI, persistence, and request replay keep working.

### Analytical parameters

`src/gammaforge/engines/analytical/schema.py` currently exposes:

- `n_quad` — energy-spread quadrature;
- `n_quad_u` — exact transverse overlap quadrature nodes, with `1` meaning the fast approximation;
- `n_quad_overlap` — overlap-integral quadrature.

New model-selection controls should be compact. Do not expose every internal tier or every numerical detail as a top-level user knob.

### Existing overlap implementation

`src/gammaforge/engines/analytical/formulas.py` already contains mature DER001/DER002 machinery:

- general Gaussian overlap;
- exact/approximate crossing-angle paths;
- flying-focus path;
- offsets and astigmatism;
- resolved time/transverse overlap profiles;
- luminosity-weighted mean intensity.

Reuse this machinery where DER019 says to. Do not duplicate the overlap model in a second analytical implementation.

### Existing spectral implementation

The current `angle_integrated_spectrum`:

- integrates over Gaussian electron energy only;
- uses the independent DER011 polynomial;
- applies nonlinear redshift by replacing `y=s/gamma^2` with `y=s(1+ahat)/gamma^2`;
- uses one mean `ahat`.

The engine then rescales the discretized spectrum so its trapezoidal integral equals `total_yield`.

DER019 replaces the physical mean-`ahat` shape approximation in supported tiers. Preserve a final discrete-grid normalization only as a numerical quadrature correction if useful; first prove/test the continuous model's own photon-count normalization so rescaling cannot hide a wrong shape.

### Current nonlinear bandwidth diagnostic

`estimate_spectrum_width` still uses `NONLINEAR_BROADENING_RANGE = (0.06, 1.12)` for the trajectory-to-trajectory nonlinear spread.

DER019 provides an exact replacement in the constant-transverse-width Gaussian tiers. The empirical bracket remains a fallback for geometries not yet covered by the deterministic replacement; do not delete it globally until every current use has a valid substitute.

### Output vocabulary

`OutputKind` already contains:

- `ANGULAR_DISTRIBUTION`;
- `COLLIMATED_SPECTRUM`.

Use those existing kinds. Do not introduce another spectral-angular output kind; RES052 deliberately removed the duplicate.

## Implementation strategy

Work in small vertical slices. Keep each slice scientifically testable before adding the next dimensional tier.

### 1. Add the analytical model planner without changing physics

Introduce a small internal descriptor/planner layer for analytical models.

Each model should be able to declare, conceptually:

- stable internal name;
- supported `OutputKind` values;
- structural applicability predicate;
- exact vs approximate status;
- validity diagnostics and acceptance policy;
- cost rank or comparable deterministic cost estimate;
- execution function.

The planner operates per requested observable:

```text
(inputs, output kind, requested mode / expert pin)
    -> structurally applicable models
    -> validity diagnostics
    -> accepted candidates
    -> cheapest candidate allowed by mode
    -> execute
```

Public semantics:

- `auto`: cheapest accepted model;
- `fast`: prefer low-cost accepted reductions;
- `reference`: highest-fidelity deterministic Gaussian model supported;
- expert model pin: bypass automatic choice for controlled comparisons, but still reject structurally invalid use rather than silently changing the problem.

Because the current engine protocol puts run-time controls in `Parameters`, prefer schema choices such as `model_mode` and an expert model selector rather than altering engine construction.

Do not invent validity thresholds at this stage. A reduced approximate model can be registered and explicitly pinned before an automatic threshold exists. Automatic acceptance of that model should wait for validation evidence produced later in this issue.

Result metadata must record at least:

- selected model;
- requested mode;
- exact/approximate flag;
- assumptions/symmetries used;
- outer quadrature dimension;
- whether trajectory quadrature was required;
- validity diagnostics;
- rejected cheaper alternatives and reason where useful.

Keep metadata serializable under RES063.

### 2. Implement constant-transverse-width Gaussian source primitives

Add the DER019 closed Gaussian quadratic-form moments `J_n` and the derived luminosity-weighted moments.

Required capabilities:

- centered and displaced Gaussian source where DER019 gives the quadratic form;
- round closed reduction as an explicit limit;
- exact mean `a_shape`;
- exact between-trajectory `Var(a_shape)`;
- luminosity-weighted mean DER016 finite-line variance;
- total-variance decomposition.

Keep the distinction clear:

- between-electron distribution of nominal nonlinear shifts;
- within-trajectory DER016 finite-line variance.

Use current `<a^2>`/cycle-average conventions. Do not reintroduce ambiguous peak-`a0` factors.

Where this tier is structurally valid, use the derived nonlinear spread in analytical diagnostics. Preserve `NONLINEAR_BROADENING_RANGE` as fallback elsewhere.

Prefer a new focused helper/module inside `engines/analytical/` if `formulas.py` would otherwise become harder to audit. Do not build a general symbolic framework.

### 3. Upgrade angle-integrated SPECTRUM

Implement DER019's normalized nonlinear single-electron angle-integrated shape `G(z;h)`.

Then average it over the deterministic nonlinear-coordinate distribution for the supported Gaussian tier, and over the Gaussian energy distribution as required.

Acceptance properties of the primitive itself:

- `h=0` reproduces DER011;
- the physical support endpoint is correct;
- the continuous shape integrates to unity;
- photon-count normalization does not depend on post-hoc engine rescaling.

For round fixed-width geometry, use the 1D nonlinear-coordinate integral from DER019. For more general low-rank fixed-width geometry, use the corresponding deterministic amplitude distribution/integration rather than Gaussian particles.

Keep the analytical expression implementation independent of Xigma's Stage-2 kernel code where independence is scientifically useful. Shared generic utilities are fine, but do not make the analytical validation leg a thin call into the implementation it is supposed to check.

### 4. Add cheap optimizer diagnostics

Implement the low-cost DER019 observables before the higher-dimensional spectra:

- exact circular-aperture capture fraction in its supported head-on tier;
- on-axis centroid and RMS bandwidth;
- nonlinear-shape mean/spread and finite-line decomposition;
- source transverse centroid/covariance/RMS accumulated from existing Gaussian-overlap algebra.

Do not compute a dense spatial image only to obtain moments that can be accumulated from the reduced Gaussian integral.

Expose these through structured, serializable `model_specific` metadata.

If special functions would add an otherwise unnecessary dependency, prefer the small deterministic quadrature explicitly allowed by DER019.

### 5. Add energy-integrated ANGULAR_DISTRIBUTION

Extend `AnalyticalEngine.supported_outputs` only when the corresponding implementation and tests are ready.

Start with zero electron emittance plus finite energy spread:

- use the DER012 physical angular kernel;
- integrate over gamma deterministically;
- normalize to the same analytical total yield.

Then add finite-emittance deterministic angle integration.

For crossing-angle cases use DER012/DER013 geometry. Do not reuse superseded DER005/DER006 polarization expressions.

### 6. Add deterministic COLLIMATED_SPECTRUM

Use the current `OutputKind.COLLIMATED_SPECTRUM` and target-defined angular ranges.

Implement in increasing complexity:

1. zero-emittance constant-width Gaussian source;
2. round finite emittance;
3. anisotropic finite emittance.

Eliminate gamma analytically with the DER015 inverse wherever the derivation permits rather than adding an unnecessary gamma quadrature.

Accumulate the DER017 raw channels `rho0`, `rho1`, `rho2` separately. Validate the raw channels before judging only the reconstructed spectrum.

Keep the analytical implementation independent enough from Xigma to remain a useful oracle.

### 7. Add focused/flying deterministic source maps

Only after the low-dimensional tiers are stable, implement the source-map tiers that retain the real paraxial spot evolution.

Round focused/flying model:

- deterministic outer source coordinates as derived in DER019;
- short 1D trajectory quadrature at each source node;
- compute DER016 trajectory quantities from the real `GaussianParaxialLaser` along that trajectory.

Anisotropic/crossed model:

- extend to the required 3D source coordinates;
- retain full laser spot evolution and crossing geometry.

Do not call these models fixed-width/frozen-spot. Their purpose is precisely to remove that approximation while remaining particle-free.

Reuse DER001/DER002 total overlap as the normalization oracle. A source-map spectrum is not accepted until integrating its luminosity reproduces the established overlap result at quadrature convergence.

### 8. Establish model-selection validity from evidence

After both the reduced and higher-fidelity tiers exist, sweep the dimensionless validity diagnostics proposed in DER019, including diffraction/hourglass/drift measures where applicable.

Compare reduced vs focused deterministic models and Xigma in matching Gaussian scenarios.

Only then choose automatic acceptance thresholds.

If those thresholds become durable project policy, record them in an appropriate RES decision before merge rather than burying them in a magic constant.

Until thresholds are justified:

- `reference` can select the highest-fidelity structurally supported tier;
- exact specializations can be selected automatically;
- approximate reduced tiers may require explicit pinning rather than pretending their domain is settled.

## Expected code areas

Likely production files:

- `src/gammaforge/engines/analytical/engine.py`
- `src/gammaforge/engines/analytical/formulas.py`
- `src/gammaforge/engines/analytical/schema.py`
- one or more new focused analytical modules if separation materially improves readability.

Potential shared interfaces:

- `src/gammaforge/io/results.py` only if current `model_specific` structure cannot express the provenance cleanly;
- `src/gammaforge/io/target.py` should normally need no new output kind.

Tests/validation:

- `tests/test_analytical.py` for compact mathematical/physics regressions;
- existing Xigma/analytical validation infrastructure for cross-engine convergence;
- a new validation record under `docs/validation/` if the PR establishes numerical validity thresholds or new scientific evidence.

Avoid changing Xigma production code unless a genuinely generic helper can be shared without coupling the analytical oracle to Xigma's implementation.

## Invariants

The implementation must preserve:

- DER001/DER002 total-yield physics and existing Gaussian overlap behavior;
- current cycle-average `<a^2>` convention from RES053/RES054;
- DER012 photon-count-normalized physical polarization at crossing angle;
- DER013 direction-Doppler and DER015 observer-dependent nonlinear incidence;
- DER016/DER017 moment definitions;
- `COLLIMATED_SPECTRUM` as the only energy-angle output kind;
- Engine protocol and request/result persistence;
- existing analytical behavior for unsupported/new-model-disabled cases until a tested replacement is selected;
- explicit failure/warning rather than silent fallback outside a model's assumptions.

Do not add a new Xigma table dimension and do not change Xigma's settled physics.

## Validation and acceptance criteria

Keep the test suite physics-focused.

Required analytical checks:

- `J_n` moments recover direct Gaussian integration and the round closed limits.
- Mean nonlinear intensity agrees with `overlap_mean_a0_sq` in the common limit.
- Derived between-trajectory variance is non-negative and matches deterministic/direct numerical integration.
- Nonlinear `G(z;h)` integrates to one and equals DER011 at `h=0`.
- Circular-aperture spectrum integrated over energy equals the closed aperture fraction.
- Angular distribution integrated over its sufficiently wide solid-angle domain converges to analytical total yield.
- Round/anisotropic higher-dimensional models reduce to lower-dimensional symmetry limits.

Cross-engine checks:

- deterministic fixed-width spectra converge to Xigma in matching weak/frozen-spot Gaussian scenarios;
- focused/flying deterministic source-map luminosity converges to DER001/DER002;
- focused/flying spectra converge against Xigma as deterministic quadrature and Xigma sampling/table resolution are refined;
- compare `rho0`, `rho1`, and `rho2` separately where moment2 reconstruction is used.

Planner checks:

- exact specializations are chosen when structurally applicable;
- rejected models record a useful reason;
- `reference` never chooses a lower-fidelity model when a higher-fidelity supported tier exists;
- `auto` does not use an approximate reduced tier before its validity acceptance is established;
- expert pinning cannot force a structurally invalid model silently.

Repository regression checks before final review:

- targeted analytical tests;
- `pytest -m fast`;
- relevant analytical/Xigma validation runs if their reference outputs are touched.

Do not create a combinatorial matrix of trivial software tests.

## Deliverables

Before this PR is ready for merge, it should contain as applicable:

- analytical model-planner implementation;
- deterministic Gaussian source/moment formulas;
- upgraded angle-integrated spectrum;
- new analytical angular/collimated outputs for implemented tiers;
- focused/flying deterministic source maps;
- compact scientific regression tests;
- validation evidence for any automatic validity thresholds introduced;
- durable RES documentation for any new lasting selection policy that is not already determined by DER019/current decisions;
- updates to `PROGRESS.md` only after the corresponding implementation/evidence actually exists.

The handoff itself is temporary and must be deleted from the branch before final merge.

## Out of scope

Do not implement:

- changes to Xigma physics or table dimensionality;
- non-Gaussian bunch fitting/resampling;
- arbitrary aberrated/non-Gaussian laser source reduction;
- explicit higher harmonics;
- quantum recoil;
- manuscript changes;
- automatic promotion of DER019;
- unrelated validation-suite restructuring.

Do not silently expand the issue because another nearby analytical approximation looks convenient.

## Open questions / non-blocking decisions

The following are intentionally left to implementation evidence rather than physics re-derivation:

- numerical validity thresholds for the constant-width approximation;
- best stable numerical representation for the noncentral rank-two amplitude distribution;
- practical quadrature families/node counts for deterministic 1D/2D/3D tiers;
- cost ranking used by the planner;
- the point at which extra correlations/chirp make deterministic Gaussian reduction unattractive.

These do not block the first exact/closed tiers. If any turns into a physics question rather than a numerical/architecture choice, stop and surface it.

## Suggested commit sequence

After this handoff commit:

1. planner/schema/provenance scaffolding with no physics behavior change;
2. deterministic Gaussian nonlinear moments and diagnostics;
3. normalized nonlinear angle-integrated spectrum;
4. angular distribution;
5. collimated spectrum and raw moment channels;
6. focused/flying source maps;
7. validity sweeps, model-selection thresholds, and durable validation/decision docs if warranted;
8. final regressions/documentation;
9. delete this handoff file in a final branch commit before merge.

Issue #5 should remain open while the draft PR is under implementation. The PR may use `Closes #5` so the issue closes only when the completed PR is merged.
