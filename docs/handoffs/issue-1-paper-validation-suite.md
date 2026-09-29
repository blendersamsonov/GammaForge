# Paper validation suite — repository handoff

Status: implementation handoff for GitHub issue #1; branch cut from corrected `main` at commit `0610e36474f386187e9fdd0148c2031de03e3613`.

Date: 2026-09-29

Tracking issue: #1, `Implement paper validation suite: physics, table convergence, pull vs forward`.

## Goal

Produce the scientific validation and benchmarking evidence required by the rebuilt GammaForge/xigma paper.

This is a validation, benchmarking, and paper-evidence task. It is **not** a request to redesign the settled xigma physics. The work must keep three claims logically separate:

1. **Physical-model accuracy:** the reduced first-harmonic resonance and finite-line reconstruction reproduce an independent coherent calculation in the regime claimed by the paper.
2. **Representation accuracy:** the five-dimensional illuminated table and its co-shaped moment channels converge to the same reduced particle model before the CuPy/QMC sampler is involved.
3. **Query-algorithm accuracy and usefulness:** pull querying reproduces the deterministic table integral and has measurable computational advantages for the selective/adaptive workloads claimed in the paper.

A single xigma-versus-delta comparison is not sufficient. The existing delta references share Stage-0 trajectory quantities with xigma, so they cannot detect common-mode Stage-0 physics errors.

## Authority and references

Established physics is owned by the repository derivations, not by this handoff:

- `DER012` — physical transverse-dipole emission under crossing angle — **verified**.
- `DER013` — direction-dependent Doppler reduction — **verified**.
- `DER015` — observer-dependent ponderomotive incidence — **verified**.
- `DER016` — carrier-weighted trajectory moments — **verified**.
- `DER017` — second-order spectral-moment reconstruction — **verified**.
- `DER018` — effective plane-wave surrogate and radiation-phase origin of ponderomotive incidence — **derived** and explicitly one subject of the independent validation below.
- `DER008` — ring/annulus importance-sampling kernel — **derived**; do not turn its historical convergence claims into paper assertions without renewed measurements.

Relevant implemented decisions and current evidence include:

- `RES075` — tiered test-suite policy.
- `RES077` — matched-energy-bin delta comparison.
- `RES079` — table-quadrature discrepancy interpreted as numerical convergence; its 2026-09-28 amendment makes the old four-dimensional numerical values historical rather than current goldens.
- `RES080` — gamma-resonance proposal and its measured limitations.
- `RES082` — consistent per-electron direction Doppler.
- `RES084` — production fixed-direction delta gates.
- `RES085` — independent CuPy delta reference.
- `RES090` — observer-dependent nonlinear incidence in Stage 2 and the current five-dimensional moment-query architecture.

Read before implementation:

- `PROGRESS.md`
- `docs/GRAND_PLAN.md`, especially the validation strategy
- `docs/validation/delta-production-2026-09-14.md`
- `docs/validation/gamma-proposal-2026-09-13.md`
- `docs/validation/cupy-stages01-2026-09-13.md`
- `docs/validation/direction-doppler-2026-09-12.md`
- the derivations and decisions listed above.

The current production implementation of interest is primarily:

- `src/gammaforge/engines/xigma/stages.py`
- `src/gammaforge/engines/xigma/spectrum_sampler.py`
- `src/gammaforge/validation/delta_validation.py`
- `src/gammaforge/validation/references/delta_emission.py`
- `src/gammaforge/validation/references/delta_cupy.py`
- `src/gammaforge/validation/metrics.py`
- `src/gammaforge/validation/run.py`
- `src/gammaforge/validation/scenarios.py`.

## Current state on `main`

Do not rebuild validation infrastructure that already exists.

The repository already has:

- a five-dimensional xigma table over `(gamma, theta_ex, theta_ey, ahat, Cbar)`;
- co-shaped luminosity-weighted channels for the DER016 moments;
- raw Stage-2 spectral channels `rho0`, `rho1`, and `rho2`;
- DER017 second-order reconstruction on nonuniform spectral grids;
- exact observer-dependent `Q` in Stage 2;
- persistent prepared queries that cache old spectral samples;
- independent CPU direct-particle delta emission;
- an independent fused-CUDA direct-particle delta reference;
- matched-bin candidate/reference comparison with count, spectral-L1, and centroid diagnostics;
- a production fixed-direction validation matrix covering the shared bank, head-on and small crossed geometry, linear/elliptical/circular polarization, and on/off-axis views;
- real-CUDA sampler release gates and ring/subsampling refinement infrastructure;
- tiered pytest execution under RES075.

The production fixed-direction comparison is already numerically close at the finest tested grids, but `PROGRESS.md` still records incomplete scientific acceptance. The current production packet also retains low-`a0` refinement failures and common-mode Stage-0 blind spots.

### Important dependency: issue #7

The external handoff previously recommended `ahat_decades=0.3` for paper-quality runs. That recommendation is obsolete.

Issue #7 now owns the author decision to regress Stage 1.5 production retargeting to **uniform raw-`ahat` bins** and remove `ahat_decades`. This change has **not** landed on `main` yet: current `main` still uses RES032's nonuniform grid and exposes `ahat_decades=1.0`.

Therefore:

- do not tune `ahat_decades` as part of this handoff;
- do not choose `0.3` as a new paper or production default here;
- run the publication-quality `ahat` convergence and performance studies against the post-#7 uniform-grid implementation;
- work that does not depend on the retarget spacing, especially the independent single-electron/coherent oracle, may proceed before #7 lands;
- any exploratory table result produced before #2 must be labelled pre-#2 and rerun before being used in the paper.

## Test-suite policy

Keep the full paper study separate from ordinary pytest.

The existing suite is already intentionally tiered and focused on scientific invariants. Do not convert the parameter sweeps below into another large parameterized test matrix.

Use a dedicated paper-validation driver under `gammaforge.validation` or an equivalent repository-native entry point, but reuse the current scenario, reference, metric, and report infrastructure rather than forking it. A command in the spirit of

```text
python -m gammaforge.validation.paper --quick|--full --section ... --backend numpy|cupy --output <dir>
```

is acceptable if it fits the code cleanly.

Every full run must record at least:

- Git commit;
- scenario and numerical settings;
- CPU/GPU identity;
- Python, NumPy, and CuPy versions as applicable;
- deterministic particle seed;
- QMC/low-discrepancy sequence controls;
- source fingerprints where the current validation infrastructure already supports them.

Generated arrays and large benchmark data should not be committed by default. Persistent scientific summaries belong under `docs/validation/`.

## Required work

### 1. Independent coherent plane-wave oracle

Add a **validation-only** single-electron reference that does not call xigma resonance, inverse-root, polarization, or table helpers.

Use the exact plane-wave trajectory identities and radiation phase recorded in DER018, then evaluate the classical spectral radiation directly from the trajectory, e.g. by a numerical Liénard-Wiechert/Fourier radiation integral.

This reference is the missing scientific anchor for the model-level claims.

First establish the oracle itself:

- `kappa` remains constant for an exact plane wave;
- reconstructed `gamma(phi)` and longitudinal momentum agree with the exact plane-wave identities;
- the spectrum converges under phase/time-step and spectral-grid refinement;
- the weak-field angular integral reproduces the Thomson normalization to the accuracy of the numerical radiation integration.

Then extract the fundamental line centre and compare with the production reduced resonance

```text
s_R = D * Cbar * gamma^2 / (1 + Q * ahat + gamma^2 * r^2).
```

Scan, at minimum:

- head-on and representative oblique incidence;
- observation offsets near `0`, `0.5/gamma`, `1/gamma`, and `2/gamma`;
- several relativistic `gamma` values;
- nonlinear strengths spanning the regime the paper intends to claim.

Explicitly test DER018's cone-scaling statement:

- head-on: variation of `Q` across the radiation cone is `O(1/gamma^2)`;
- generic oblique incidence: the variation is `O(1/gamma)`.

Do not invent a universal model-error tolerance before the data exist. Exact plane-wave identities can be hard numerical gates; the difference between the exact coherent line and the reduced ultra-relativistic model is a scientific result to report.

If this oracle disagrees with DER012, DER013, DER015, DER016, or DER017 beyond understood approximation error, stop and report the discrepancy. Do not modify production physics to obtain agreement.

DER018 itself remains `derived` until separately reviewed/promoted through the derivation-confidence workflow.

### 2. Finite fundamental-line / DER017 validation

Use the independent coherent oracle to validate the **fundamental line**, not only its nominal centre.

Cover at least:

1. unchirped finite-envelope ponderomotive broadening;
2. carrier/chirp variation with weak nonlinear variation;
3. a combined case with nonzero `Cov(q,C)`.

Compare:

- direct coherent fundamental spectrum;
- nominal delta line;
- DER017 second-order moment reconstruction.

For each case scan the residual fractional linewidth. Report separately:

- total spectral mass/yield;
- centroid;
- variance;
- normalized integrated L1 shape error.

When the purpose is shape accuracy, normalize the compared shapes to the same mass and report absolute-yield error separately so a normalization difference cannot masquerade as a shape difference.

Measure error versus a dimensionless residual-width measure such as `sqrt(m2)`. The paper needs an **empirical validity region** of the second-order expansion and evidence that moment2 improves on the delta model where claimed.

Do not encode a guessed physical boundary such as `m2 < X` into production or pytest. The author chooses the paper's stated validity range after reviewing the measured curve.

### 3. Independent Stage-0 trajectory checks

Existing xigma/delta comparisons share Stage-0 quantities. Add a small independent high-resolution quadrature reference for selected trajectories without calling `integrate_trajectories`.

For each selected trajectory compare, under `n_steps` refinement:

- luminosity;
- `ahat` / nonlinear shape statistic;
- `Cbar`;
- `Var(q)`;
- `Var(C)`;
- `Cov(q,C)`.

Include at least:

- on-axis trajectory;
- transverse-offset trajectory;
- oblique electron direction;
- one controlled chirped/phase-gradient validation fixture.

Reuse the existing analytical Gaussian-overlap and elementary/Kascade Thomson anchors for absolute yield where useful. Do not make Kascade a blocking oracle for coherent finite-line physics.

### 4. Current five-dimensional table convergence

Freeze one `TrajectorySamples` population so Stage 1/1.5 convergence is isolated from particle sampling and Stage-0 error.

The current repository already has direct-particle delta references and DER017 particle-level moment checks. Extend/reuse them to produce **paper-quality five-dimensional convergence evidence** for `rho0`, `rho1`, and `rho2`.

Refine separately before joint refinement:

- gamma bins;
- `theta_ex` / `theta_ey` bins;
- raw `ahat` bins;
- `Cbar` bins.

Use CIC for the paper convergence study unless deposition schemes themselves are being compared.

Metrics:

- finite-window or requested-aperture yield/mass error;
- spectral centroid error;
- normalized integrated L1;
- for signed `rho1`/`rho2`, an absolute-error normalization based on `integral(abs(reference))`, not naive pointwise relative error.

The old RES079 numerical percentages are historical four-dimensional evidence and must not be quoted as current five-dimensional convergence results.

For the `ahat` axis, follow issue #7: publication convergence should scan `n_bins_ahat` on the uniform production grid once #2 lands, not tune `ahat_decades`.

### 5. Compressed-table forward baseline

Implement the missing **compressed-table forward** validation/reference algorithm.

It must use the same retargeted `Table`, the same DER012/013/015/016/017 physics, and the same line-model choice as the pull calculation. The only intended difference is the direction of evaluation:

- **table-forward:** visit source table elements, evaluate their resonance, and deposit into requested spectral bins;
- **table-pull:** request spectral coordinates, analytically invert the resonance, and interpolate the table at `Gamma`.

This baseline is scientifically important because it separates:

- the gain from compressing `N_p` particles to a table;
- the additional gain from the pull/query formulation.

For correctness, compare **matched physical spectral-bin masses**. Integrate pull densities over the same bins using adequate deterministic quadrature rather than comparing a histogram average with a pointwise density.

Use the deterministic NumPy pull path first. Under bin/spectral-quadrature refinement, forward and pull must approach the same discrete-table result. Do not rescale either spectrum to force agreement.

### 6. Current-model sampler convergence

After the deterministic table-forward/pull comparison is clean, validate the accelerated CuPy ring/QMC sampler against the dense deterministic NumPy pull integral on the **same five-dimensional table and moment channels**.

Reuse RES080/RES084/RES085 infrastructure rather than creating another sampler harness.

Cover:

- shared baseline / low-`a0` / near-`a0`-max bank;
- head-on and crossed geometry;
- linear plus at least one elliptical/circular crossed case;
- on-axis and off-axis views;
- unchirped `Cbar=1` and a validation fixture with nontrivial carrier coordinate.

Sweep ring and subsampling/QMC work separately.

Report:

- integrated three-dimensional angular-spectral mass/yield;
- energy centroid;
- volume-weighted integrated L1 over the spectral-angular cube.

If `validation.metrics` still lacks a proper multidimensional integrated-L1 metric, add one there rather than reducing the comparison to total yield.

Do **not** assert `N^-1` or any other convergence law. DER008 remains derived and RES080 explicitly records finite-sample limitations. Plot the observed error-versus-work relation and state only what is measured.

Keep two ideas separate:

- NumPy/CuPy implementation agreement;
- QMC/ring estimator convergence to the dense deterministic pull integral.

A green backend-invariance gate is not scientific acceptance of the sampler.

### 7. Fair three-way performance benchmark

Benchmark three algorithms after a common trajectory sample set is prepared:

A. direct-particle forward binning;
B. compressed-table forward binning;
C. compressed-table pull querying.

The repository already contains CPU and CUDA direct-particle delta references; reuse them where their contracts match instead of adding a third direct implementation.

Strong paper speed claims require the compared Stage-2 algorithms to run on the **same backend**. Do not publish a GPU-pull versus Python-forward comparison as an algorithmic speedup.

Report both:

- Stage-2-only time;
- amortized end-to-end time including the common precomputation appropriate to each method.

Sweep the variables that distinguish the algorithms:

- `N_p`;
- number of observation directions/views;
- number of requested spectral samples;
- requested spectral-window fraction;
- table resolution.

Ranges must be configurable rather than hard-coded.

For a narrow-window benchmark, a forward method may discard a source contribution after it has evaluated its resonance. It may not use inverse-query support information to skip particles/cells before doing the work that defines it as a forward method.

For GPU timings:

- warm up compilation;
- synchronize around timed regions;
- take multiple repetitions;
- report median and a robust spread;
- record hardware.

No runtime threshold belongs in pytest.

### 8. Persistent selective refinement

`PreparedQuery` is already implemented and tested. Do not reimplement it.

For the paper, demonstrate the computational property it enables:

1. evaluate a sparse spectral grid;
2. insert new spectral points according to a validation-only refinement criterion;
3. show that previously cached raw `rho0/rho1/rho2` samples are reused;
4. compare cumulative work and final error with evaluating the final point set in one shot;
5. compare the same requested-resolution progression with rebuilding a forward histogram.

The adaptive selection policy can remain validation-only. The production API does not need a new optimiser/refiner for this handoff.

### 9. Higher-harmonic validity as a validation-only extension

Issue #1 already contains an author addendum requesting that omitted nonlinear harmonics be treated as a separate model error from the finite width/skewness of the retained `n=1` line.

Use the independent coherent plane-wave oracle, where practical, to measure:

- total classical radiated energy `W_tot`;
- true first-harmonic contribution `W_1`;
- `eta_H = 1 - W_1 / W_tot`;
- an angle/aperture-resolved omitted fraction where relevant.

Weak-field checks should include the appropriate symmetry cases, e.g. odd-harmonic scaling at exact backward linear polarization and an off-axis geometry where the second harmonic is allowed.

This is **validation-only** in issue #1.

Issue #3 owns any public/bunch-level harmonic diagnostic, optional production rescaling, or API work. Do not duplicate that productization here. In particular:

- do not add a harmonic table axis;
- do not implement a full multi-harmonic production spectrum;
- do not rescale the production first-harmonic spectrum under issue #1;
- do not assume an energy-fraction correction is automatically valid for a photon-number spectrum.

There is currently no repository DER for the higher-harmonic working derivation. If the harmonic projection or selection-rule convention exposes unresolved physics, surface it rather than promoting a chat/Drive formula to repository authority.

## Scenario matrix

Use a compact matrix, not a Cartesian explosion.

Minimum paper matrix:

- **anchor:** monoenergetic, zero-divergence, head-on, linear, unchirped;
- **shared bank:** current `baseline`, `low_a0`, `near_a0_max`;
- **geometry:** current small crossed case plus one stronger stress geometry used to map approximation failure, not to define the production regime;
- **polarization:** linear for primary convergence figures, with one elliptical/circular crossed case for the angular kernel;
- **chirp:** validation-only controlled `C(t)` fixtures, including one nonzero `q-C` covariance case, until a production laser UI naturally exposes such inputs.

Do not redesign the public laser API merely to create a validation fixture.

## Metrics

Use observables that are actually sensitive to the effect being tested.

At minimum:

- total mass/yield;
- spectral centroid;
- spectral variance where line shape is under test;
- normalized integrated L1 for one-dimensional spectra;
- volume-weighted integrated L1 for spectral-angular cubes;
- exact/known invariants for oracle self-checks.

Do not use total yield as evidence for a redshift correction; `PROGRESS.md` explicitly records this blind spot.

Do not use pointwise relative error in dark bins or for signed moment channels.

## Paper-facing outputs

A reproducible full run should be able to generate the numerical data for at least:

1. single-electron coherent-vs-reduced line-centre error across geometry and `gamma`;
2. coherent fundamental vs delta vs moment2, including error versus residual linewidth;
3. current five-dimensional table convergence for the base and moment channels;
4. QMC/ring error versus work against dense NumPy pull;
5. direct-particle forward vs table-forward vs pull runtime;
6. runtime/cost versus requested spectral selectivity (`N_s` or window fraction);
7. persistent/adaptive spectral-refinement reuse;
8. first-harmonic omitted-energy fraction versus nonlinear strength, if the harmonic-oracle portion is completed within this issue.

Each full run should emit machine-readable data and a short Markdown summary suitable for `docs/validation/`. Plot generation may be part of the validation driver; large raw arrays should remain generated artifacts unless repository policy changes.

## Scientific and technical invariants

The implementation must preserve:

- observer independence of the reusable Stage-1/retargeted table; `Q` remains query-time geometry;
- DER013 direction Doppler, DER015 `Q`, DER016 carrier moments, and DER017 moment reconstruction unchanged;
- the existing public xigma/Collision API unless a validation-only helper can be added without changing it;
- no normalization fitting between reference and candidate;
- NumPy/CuPy physics parity;
- existing zero-chirp and head-on limits;
- signed covariance information;
- persistent query reuse semantics;
- deterministic/reproducible validation seeds and source fingerprints where already supported.

## Acceptance criteria

Completion means the evidence exists and is reproducible, not merely that a command exits zero.

Required acceptance:

1. The independent plane-wave oracle passes its exact kinematic self-checks and demonstrates numerical convergence.
2. Reduced line-centre deviations from the coherent oracle are mapped over the requested geometry/`gamma`/`a0` domain. Any unexplained conflict with a verified derivation is reported as a blocker rather than patched.
3. DER017's moment2 reconstruction is compared with a direct coherent fundamental line and an empirical validity region is documented.
4. Independent Stage-0 direct quadrature covers all trajectory moments used by the five-dimensional model.
5. Current five-dimensional `rho0/rho1/rho2` table convergence is measured under independent axis refinements; paper configuration is chosen from the measured convergence, not inherited from historical RES079 numbers.
6. The compressed-table forward baseline and deterministic pull integral converge to the same matched-bin discrete-table result.
7. The current CuPy sampler has an error-versus-work curve against dense NumPy pull for the full current moment model.
8. The three performance baselines are compared on a fair same-backend basis, with Stage-2-only and amortized timings.
9. Persistent refinement produces the same final raw channels as a one-shot evaluation and demonstrates incremental-work reuse.
10. All paper-facing runs are reproducible from repository commands and preserve enough metadata to identify the exact source and hardware.
11. Publication-quality `ahat` convergence is rerun after issue #7's uniform-grid change before those results are used in the paper.

The numerical budgets in RES084 remain useful implementation gates, but they are not by themselves the scientific acceptance criterion for the paper.

## Pytest promotion

Do not move the complete paper matrix into pytest.

Only add/promote cheap scientific tripwires that protect genuinely independent invariants, for example:

- exact plane-wave oracle identities;
- one independent Stage-0 moment quadrature check;
- one small current-model direct-particle vs table `rho0/rho1/rho2` check if not already covered equivalently;
- one deterministic table-forward vs pull matched-bin check;
- one CUDA sampler vs dense NumPy smoke check when CUDA is available.

Reuse existing normalization, transverse-dipole, observer-`Q`, moment-reconstruction, prepared-query, delta, and sampler tests instead of duplicating them.

## Deliverables

Expected persistent outputs:

- validation-only independent coherent/plane-wave reference code;
- any minimal validation-only chirp/harmonic fixtures required by the studies;
- compressed-table forward reference/baseline;
- paper-validation driver using the existing validation infrastructure;
- machine-readable convergence/benchmark packets;
- manuscript-ready plots or plot-generation scripts;
- short `docs/validation/` reports for completed full studies;
- minimal new pytest tripwires;
- updates to `PROGRESS.md` and relevant validation documentation **after** evidence has been produced.

Do not add a decision merely to record a measurement. Add/modify a `RESNNN` only if the implementation establishes an actual design/testing decision that belongs in the decision lifecycle.

## Out of scope

Do not:

- change production physics to make validation pass;
- re-derive or rewrite verified DER012/013/015/016/017 inside implementation code;
- promote DER018 automatically;
- implement issue #7's uniform-`ahat` production change as part of issue #1;
- implement issue #3's public harmonic diagnostic or optional spectrum rescaling as part of issue #1;
- implement full higher-harmonic production spectra;
- redesign the public laser API solely for validation fixtures;
- complete all Phase-5/four-method wiring merely because `PROGRESS.md` lists it, unless it becomes necessary for a specific paper validation result;
- modify the manuscript as part of this coding handoff;
- commit large generated benchmark arrays without an existing repository convention requiring them;
- add performance pass/fail thresholds to CI.

## Open questions and blockers

### Dependency on issue #7

The paper's final table-resolution and performance studies should use the uniform raw-`ahat` policy requested by issue #7. Until it lands, those parts are exploratory only.

### DER018 focused-pulse validity

DER018 is still derived. The exact plane-wave oracle can verify its plane-wave phase construction, but the focused-pulse surrogate question remains broader: how to choose an effective `n0` and how large an accumulated radiation-phase error is acceptable in a strongly focused pulse. If the paper intends claims beyond the fixed-effective-direction regime, that requires additional author physics review.

### Physical validity threshold for moment2

The task should measure the error versus residual linewidth. The numerical data should not silently choose the paper's physical validity threshold; that remains an author decision.

### Higher-harmonic boundary

The validation-only harmonic measurement is requested by the issue history, while productization is separately tracked in issue #3 and the underlying working derivation has no repository DER. Keep that boundary explicit.

### Strong-stress geometry

A stronger crossing/focusing case is useful to show where the reduced model deteriorates, but it must be labelled a stress/validity-boundary case rather than folded into the nominal production regime without author approval.

## Suggested implementation order

1. Build and self-validate the independent plane-wave/coherent oracle.
2. Use it for line-centre/DER018 and DER017 finite-line studies.
3. Add the independent Stage-0 moment quadrature checks.
4. Wait for or rebase onto issue #7 before final publication-quality `ahat` convergence.
5. Produce current five-dimensional table convergence.
6. Implement and validate the compressed-table forward baseline.
7. Measure current-model QMC/ring convergence against dense NumPy pull.
8. Run same-backend three-way performance benchmarks.
9. Produce the persistent-refinement benchmark.
10. Add the validation-only harmonic measurements if their projection convention is scientifically settled enough; otherwise record the blocker and leave issue #3 untouched.
11. Promote only minimal tripwires into pytest and write `docs/validation/` summaries after results exist.

## Completion handoff

When implementation finishes, report:

- exact commit(s) used for each evidence packet;
- commands required to reproduce every paper-facing figure/data set;
- which acceptance items passed, failed, or remained scientifically inconclusive;
- any conflict with a verified derivation;
- whether DER018 now has independent evidence sufficient for a separate confidence-promotion review;
- whether issue #7 landed before the final table/performance results;
- whether the harmonic portion remained validation-only or was deferred to issue #3.
