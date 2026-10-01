# Decisions — Index

Lifecycle and type workflow: [`README.md`](README.md). Add or update a row in the same
change as a new decision, lifecycle move or type change. Each path points directly into
one lifecycle folder; the type column matches the file's `Type:` tag.

| id | title | type | status | path |
|----|-------|-------|--------|------|
| RES001 | Build backend: hatchling, src layout | process | implemented | archived/RES001-build-backend-hatchling-src-layout.md |
| RES002 | Doc-staleness guard (C2) scope: `DECISIONS.md` only, not `GRAND_PLAN.md` or `PROGRESS.md` — superseded by RES055 | testing | implemented | archived/RES002-doc-staleness-guard-scope.md |
| RES003 | Phase 0 package skeleton: subpackage `__init__.py` only, no placeholder module files | process | implemented | archived/RES003-phase0-package-skeleton.md |
| RES004 | `FieldSpec` gains `choices` and `integer`; `FieldKind` stays four members | architecture | implemented | archived/RES004-fieldspec-choices-and-integer.md |
| RES005 | Beam correlations are stored as correlation coefficients, not dimensional slopes | architecture | implemented | archived/RES005-beam-correlations-as-coefficients.md |
| RES006 | `Bunch.weight` is a per-particle array | architecture | implemented | archived/RES006-bunch-weight-per-particle-array.md |
| RES007 | `Axis` members carry a `(key, unit)` pair | bug-fix | implemented | archived/RES007-axis-key-unit-pair.md |
| RES008 | A `gaussian_charge` pint context beside `light_time` | architecture | implemented | archived/RES008-gaussian-charge-pint-context.md |
| RES009 | One `overlap_time_window` serves both the prefilter and the temporal autorange | architecture | implemented | archived/RES009-overlap-time-window-shared-utility.md |
| RES010 | `fit_gaussian_paraxial` implements the identity path only | architecture | implemented | archived/RES010-fit-gaussian-paraxial-identity-only.md |
| RES011 | Doc-staleness guard hardened for enum members and ambiguous shapes; its own behaviour is tested | testing | implemented | archived/RES011-doc-staleness-guard-hardened.md |
| RES012 | Angle-energy correlations are stored, so `drift` composes | bug-fix | implemented | archived/RES012-angle-energy-correlations-stored.md |
| RES013 | Dimensioned fields are pint `Quantity`; bulk arrays declare their units instead | architecture | implemented | implemented/RES013-pint-quantity-boundary.md |
| RES014 | The `light_time` context is opt-in per field, not global | bug-fix | implemented | archived/RES014-light-time-context-opt-in.md |
| RES015 | No coordinate normalization: xigma works in CGS directly | architecture | implemented | implemented/RES015-no-coordinate-normalization.md |
| RES016 | `beam.py` renamed to `bunch.py`; the "canonical bunch" phrasing dropped | architecture | implemented | archived/RES016-beam-py-renamed-to-bunch-py.md |
| RES017 | Dimensioned fields are converted to canonical CGS on construction, not stored in the unit given | architecture | implemented | archived/RES017-canonical-cgs-on-construction.md |
| RES018 | `engines/base.py` lands in Phase 2, with the `Engine` protocol but no registry | architecture | implemented | implemented/RES018-engine-protocol-lands-phase2.md |
| RES019 | Golden references are generated in a subprocess, and committed as ordinary results files | testing | implemented | archived/RES019-golden-references-subprocess.md |
| RES020 | The validation runners have no result cache | testing | implemented | archived/RES020-validation-runners-no-cache.md |
| RES021 | The laser's active region is a cone, not a cylinder | bug-fix | implemented | archived/RES021-active-region-cone-not-cylinder.md |
| RES022 | The active-region cone is evaluated at the flying-focus coordinate | bug-fix | implemented | archived/RES022-active-region-flying-focus-coordinate.md |
| RES023 | The window metric counts a window if *either* side has flux in it | testing | implemented | implemented/RES023-window-metric-either-side-flux.md |
| RES024 | Stage 0 reads the whole laser through `a0_profile` | architecture | implemented | archived/RES024-stage0-reads-whole-laser-via-a0-profile.md |
| RES025 | Delta's `2 pi` is derived and reported, not corrected | testing | implemented | archived/RES025-deltas-2pi-derived-and-reported.md |
| RES026 | §9.1 resolved: the missing factor is in the paper, at eq. (xsec) | bug-fix | implemented | implemented/RES026-9-1-resolved-missing-factor-in-paper.md |
| RES027 | Review-round corrections to Phase 2.5, and one committed file that should not have been | bug-fix | implemented | archived/RES027-phase2-5-review-round-corrections.md |
| RES028 | `deposit_table` bins directly onto `ahat`; the predecessor's fixed-range `retarget_a0`/`a0_kind` regrid is not ported, but the underlying peak-independence it exploited is — for both `ahat` and `luminosity` | architecture | implemented | archived/RES028-deposit-table-direct-ahat-deposit.md |
| RES029 | Stage 2's numpy kernel ports the predecessor's brute-force grid quadrature, not the GPU importance sampler; `KERNEL_NORMALIZATION_CONSTANT` isolates the pending §9.1 factor | architecture | implemented | implemented/RES029-stage2-numpy-kernel-brute-force.md |
| RES030 | `Collision`'s cache is per-instance memoization only; no cross-call staleness detection | architecture | implemented | implemented/RES030-collision-cache-per-instance-only.md |
| RES031 | `XigmaEngine` is not passed to `run_suite()` by `validation.run.main()` | testing | implemented | implemented/RES031-xigma-not-wired-into-run-suite.md |
| RES032 | Stage 1 deposits onto `a0_shape`, not `ahat`; a conservative regrid (`retarget_ahat`) onto a fixed, non-uniform `ahat` axis replaces the direct deposit — supersedes RES028 | architecture | implemented | implemented/RES032-shape-table-retarget-ahat.md |
| RES033 | §9.1 closed: both transcriptions of the paper's cross-section carry `1/(2 pi)`; the identity harness now expects one | bug-fix | implemented | implemented/RES033-9-1-closed-both-transcriptions-1-over-2pi.md |
| RES034 | A crossing angle warns that only the geometry is applied (§9.3's half of P14c) | bug-fix | implemented | implemented/RES034-crossing-angle-warns-geometry-only.md |
| RES035 | analytical ports the predecessor's round-beam, no-displacement approximations unchanged; the growth items stay open | architecture | implemented | implemented/RES035-analytical-ports-round-beam-approximations.md |
| RES036 | `SPECTRUM`'s grid integral is rescaled to `estimate_yield`'s total by construction, not as a discrepancy patch | architecture | implemented | implemented/RES036-spectrum-rescaled-to-yield-by-construction.md |
| RES037 | `erfcx` is hand-rolled from `math.erfc`, not a new `scipy` dependency | process | implemented | archived/RES037-erfcx-hand-rolled.md |
| RES038 | the spectrum-width breakdown's `theta_col` is the geometric mean of `Target`'s x/y collimation half-angles | architecture | implemented | implemented/RES038-theta-col-geometric-mean.md |
| RES039 | the analytical yield is a general overlap quadrature, not the round-beam closed form | feature | implemented | implemented/RES039-general-overlap-quadrature-yield.md |
| RES040 | `estimate_yield`'s laser-divergence convention error is flagged and pinned, not fixed | bug-fix | implemented | archived/RES040-estimate-yield-rayleigh-convention-pinned.md |
| RES041 | the crossing angle is covered for the yield, by quadratic form; the spectrum stays head-on and says so | feature | implemented | implemented/RES041-crossing-angle-yield-quadratic-form.md |
| RES042 | the width's nonlinearity term uses a luminosity-weighted `<a0^2>`, computed, not approximated | feature | implemented | implemented/RES042-luminosity-weighted-a0-squared.md |
| RES043 | three explicit cost tiers, and the exact 2D quadrature is opt-in | feature | implemented | implemented/RES043-three-cost-tiers-2d-quadrature-opt-in.md |
| RES044 | a flying focus is always evaluated on the 2D `(z, ct)` grid; the 1D shortcut exists but is not shipped | feature | implemented | implemented/RES044-flying-focus-2d-grid.md |
| RES045 | the luminosity-weight prefilter is a second, tolerance-based filter, not a replacement for the cone | feature | implemented | implemented/RES045-luminosity-weight-prefilter.md |
| RES046 | misalignment lives on the laser as `x_off`/`y_off`/`t_off`, and there is deliberately no `z_off` | feature | implemented | implemented/RES046-misalignment-x-off-y-off-t-off.md |
| RES047 | the illumination filter tests the *bright* region, which is what the cone approximates | feature | implemented | implemented/RES047-illumination-filter-bright-region.md |
| RES048 | the illumination window is the primary result; the filter is a corollary of it | feature | implemented | implemented/RES048-illumination-window-primary-result.md |
| RES049 | the mean red-shift is exact; its spread is reported as a measured bracket | feature | implemented | implemented/RES049-mean-redshift-exact-spread-bracket.md |
| RES050 | xigma can sample trajectories over the illuminated window, opt-in | feature | implemented | implemented/RES050-xigma-illuminated-window-sampling.md |
| RES051 | discarding charge is reported to the user, never decided silently | feature | implemented | implemented/RES051-discarding-charge-reported-not-decided.md |
| RES052 | *OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION* removed; `COLLIMATED_SPECTRUM` is now the sole `(E, θx, θy)` output kind | simplification | implemented | implemented/RES052-remove-spectral-angular-distribution.md |
| RES053 | `ahat` carries the polarization cycle average; the paper was right and the code was short a factor 1/2 | bug-fix | implemented | implemented/RES053-ahat-polarization-cycle-average.md |
| RES054 | physics goes through `<a^2>`, not `a0`: the polarization convention is removed from the yield/red-shift path rather than documented | architecture | implemented | implemented/RES054-physics-through-a-squared-not-a0.md |
| RES055 | Doc-staleness guard scope: every `proposed`/`implemented`/`rejected` decision file, not `archived/`; supersedes RES002 | testing | implemented | implemented/RES055-doc-staleness-guard-scope-decisions-tree.md |
| RES056 | Code comments citing a decision are trimmed to a pointer; the rationale lives only in the decision file | process | implemented | implemented/RES056-code-comments-citing-a-decision-are-pointers.md |
| RES057 | *docs/DERIVATIONS.md* is split into `docs/derivations/`, one file per result on a confidence pipeline instead of a lifecycle | process | implemented | implemented/RES057-derivations-migrated-to-per-file-confidence-pipeline.md |
| RES058 | NiceGUI local browser UI with a separate calculation runner | architecture | implemented | implemented/RES058-nicegui-local-browser-execution-boundary.md |
| RES059 | minimal kascade ports the emission chain, not the predecessor's framework | feature | implemented | archived/RES059-minimal-kascade-emission-chain.md |
| RES060 | Polarization uses each particle's field-free lab velocity | bug-fix | implemented | implemented/RES060-polarization-uses-per-particle-lab-velocity.md |
| RES061 | Slice integration measure is explicit per axis | architecture | implemented | implemented/RES061-slice-measure-is-explicit-per-axis.md |
| RES062 | CuPy ring/annulus sampler integrated experimentally; NumPy remains the alpha default | architecture | implemented | implemented/RES062-cupy-importance-sampler-production-path.md |
| RES063 | Alpha results preserve metadata and the submitted Gaussian request | architecture | implemented | implemented/RES063-alpha-result-provenance.md |
| RES064 | Input boundary snapshots and owned particle arrays | architecture | implemented | implemented/RES064-input-boundary-snapshots-and-array-ownership.md |
| RES065 | Headless alpha validation has an explicit restricted scope | testing | implemented | implemented/RES065-headless-alpha-validation-scope.md |
| RES066 | Alpha dependency floors preserve the numerical reference constants | process | implemented | implemented/RES066-alpha-dependency-floors.md |
| RES067 | LaserField boundary and fitting contract | architecture | implemented | implemented/RES067-laserfield-boundary-and-fitting-contract.md |
| RES068 | CuPy sampling measure and stable polarization | bug-fix | implemented | implemented/RES068-cupy-sampling-measure-and-stable-polarization.md |
| RES069 | CuPy incident polarization and crossing geometry | feature | implemented | implemented/RES069-cupy-incident-polarization-and-crossing-geometry.md |
| RES070 | Stabilize NumPy polarization reference | bug-fix | implemented | implemented/RES070-stabilize-numpy-polarization-reference.md |
| RES071 | Temporal intensity modulation and pulse trains | architecture | implemented | implemented/RES071-temporal-intensity-modulation-and-pulse-trains.md |
| RES072 | CuPy resolution controls and numerical release gate | testing | implemented | implemented/RES072-cupy-resolution-controls-and-release-gate.md |
| RES073 | Smooth laboratory observer basis Stokes parameters | feature | implemented | implemented/RES073-smooth-laboratory-observer-basis-stokes-parameters.md |
| RES074 | Delta arbitrary-angle emission validation | testing | proposed | proposed/RES074-delta-arbitrary-angle-emission-validation.md |
| RES075 | Test suite tiering and execution markers | testing | implemented | implemented/RES075-test-suite-tiering-and-markers.md |
| RES076 | Independent delta lines and Doppler diagnostic | testing | implemented | implemented/RES076-independent-delta-lines-and-doppler-diagnostic.md |
| RES077 | Matched energy-bin delta comparison | testing | implemented | implemented/RES077-matched-energy-bin-delta-comparison.md |
| RES078 | Integrate local transverse dipole reference | bug-fix | implemented | implemented/RES078-integrate-local-transverse-dipole-reference.md |
| RES079 | Xigma off-axis spectral discrepancy is numerical convergence of table quadrature | testing | implemented | implemented/RES079-xigma-spectral-discrepancy-numerical-convergence.md |
| RES080 | Gamma-resonance importance proposal | feature | implemented | implemented/RES080-xigma-gamma-resonance-importance-proposal.md |
| RES081 | Xigma temporal and spatial source diagnostics | feature | implemented | implemented/RES081-xigma-source-diagnostics.md |
| RES082 | Consistent per-electron direction Doppler in xigma | bug-fix | implemented | implemented/RES082-consistent-direction-doppler.md |
| RES083 | Explicit CuPy execution for xigma Stages 0 and 1 | bug-fix | implemented | implemented/RES083-explicit-cupy-stage01-execution.md |
| RES084 | Production delta emission gates | testing | implemented | implemented/RES084-production-delta-emission-gates.md |
| RES085 | Validated CuPy delta emission reference | feature | implemented | implemented/RES085-validated-cupy-delta-reference.md |
| RES087 | Retire the predecessor validation bridge | simplification | implemented | implemented/RES087-retire-predecessor-validation-bridge.md |
| RES088 | Ponderomotive incidence uses the electron beaming direction | bug-fix | implemented | archived/RES088-ponderomotive-incidence-beaming-cone.md |
| RES089 | GUI defaults are sectional and output ranges are per-axis Auto or manual | feature | implemented | implemented/RES089-gui-sectional-defaults-and-output-ranges.md |
| RES090 | Ponderomotive incidence is observer-dependent in Stage 2 | bug-fix | implemented | implemented/RES090-observer-dependent-ponderomotive-incidence.md |
| RES091 | Emission-supported single-bin reductions | bug-fix | implemented | implemented/RES091-emission-supported-single-bin-reductions.md |
| RES092 | Bounded production refinement | testing | implemented | implemented/RES092-bounded-production-refinement.md |
| RES093 | Uniform raw-`ahat` production retarget grid | bug-fix | implemented | implemented/RES093-uniform-raw-ahat-production-retarget.md |
| RES094 | Completion requires active tracker reconciliation | process | implemented | implemented/RES094-completion-requires-active-tracker-reconciliation.md |
| RES095 | One public engine catalog, with engine role as declarative data | architecture | implemented | implemented/RES095-public-engine-catalog-and-roles.md |
| RES096 | Gaussian Stage-0 quadrature remains opt-in | feature | implemented | implemented/RES096-gaussian-stage0-quadrature-remains-opt-in.md |
