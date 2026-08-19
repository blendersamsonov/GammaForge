# Decisions — Index

Full lifecycle/classification system: [`README.md`](README.md). Add a row here in the
same change as any new decision file or any lifecycle move — `tests/test_decision_format.py`
cross-checks this table against the files on disk.

| id | title | class | status | path |
|----|-------|-------|--------|------|
| D001 | Build backend: hatchling, src layout | process | implemented | archived/process/D001-build-backend-hatchling-src-layout.md |
| D002 | Doc-staleness guard (C2) scope: `DECISIONS.md` only, not `GRAND_PLAN.md` or `PROGRESS.md` — superseded by D055 | testing | implemented | archived/testing/D002-doc-staleness-guard-scope.md |
| D003 | Phase 0 package skeleton: subpackage `__init__.py` only, no placeholder module files | process | implemented | archived/process/D003-phase0-package-skeleton.md |
| D004 | `FieldSpec` gains `choices` and `integer`; `FieldKind` stays four members | architecture | implemented | archived/architecture/D004-fieldspec-choices-and-integer.md |
| D005 | Beam correlations are stored as correlation coefficients, not dimensional slopes | architecture | implemented | archived/architecture/D005-beam-correlations-as-coefficients.md |
| D006 | `Bunch.weight` is a per-particle array | architecture | implemented | archived/architecture/D006-bunch-weight-per-particle-array.md |
| D007 | `Axis` members carry a `(key, unit)` pair | bug-fix | implemented | archived/bug-fix/D007-axis-key-unit-pair.md |
| D008 | A `gaussian_charge` pint context beside `light_time` | architecture | implemented | archived/architecture/D008-gaussian-charge-pint-context.md |
| D009 | One `overlap_time_window` serves both the prefilter and the temporal autorange | architecture | implemented | archived/architecture/D009-overlap-time-window-shared-utility.md |
| D010 | `fit_gaussian_paraxial` implements the identity path only | architecture | implemented | archived/architecture/D010-fit-gaussian-paraxial-identity-only.md |
| D011 | Doc-staleness guard hardened for enum members and ambiguous shapes; its own behaviour is tested | testing | implemented | archived/testing/D011-doc-staleness-guard-hardened.md |
| D012 | Angle-energy correlations are stored, so `drift` composes | bug-fix | implemented | archived/bug-fix/D012-angle-energy-correlations-stored.md |
| D013 | Dimensioned fields are pint `Quantity`; bulk arrays declare their units instead | architecture | implemented | implemented/architecture/D013-pint-quantity-boundary.md |
| D014 | The `light_time` context is opt-in per field, not global | bug-fix | implemented | archived/bug-fix/D014-light-time-context-opt-in.md |
| D015 | No coordinate normalization: xigma works in CGS directly | architecture | implemented | implemented/architecture/D015-no-coordinate-normalization.md |
| D016 | `beam.py` renamed to `bunch.py`; the "canonical bunch" phrasing dropped | architecture | implemented | archived/architecture/D016-beam-py-renamed-to-bunch-py.md |
| D017 | Dimensioned fields are converted to canonical CGS on construction, not stored in the unit given | architecture | implemented | archived/architecture/D017-canonical-cgs-on-construction.md |
| D018 | `engines/base.py` lands in Phase 2, with the `Engine` protocol but no registry | architecture | implemented | implemented/architecture/D018-engine-protocol-lands-phase2.md |
| D019 | Golden references are generated in a subprocess, and committed as ordinary results files | testing | implemented | implemented/testing/D019-golden-references-subprocess.md |
| D020 | The validation runners have no result cache | testing | implemented | archived/testing/D020-validation-runners-no-cache.md |
| D021 | The laser's active region is a cone, not a cylinder | bug-fix | implemented | archived/bug-fix/D021-active-region-cone-not-cylinder.md |
| D022 | The active-region cone is evaluated at the flying-focus coordinate | bug-fix | implemented | archived/bug-fix/D022-active-region-flying-focus-coordinate.md |
| D023 | The window metric counts a window if *either* side has flux in it | testing | implemented | implemented/testing/D023-window-metric-either-side-flux.md |
| D024 | Stage 0 reads the whole laser through `a0_profile` | architecture | implemented | archived/architecture/D024-stage0-reads-whole-laser-via-a0-profile.md |
| D025 | Delta's `2 pi` is derived and reported, not corrected | testing | implemented | archived/testing/D025-deltas-2pi-derived-and-reported.md |
| D026 | §9.1 resolved: the missing factor is in the paper, at eq. (xsec) | bug-fix | implemented | implemented/bug-fix/D026-9-1-resolved-missing-factor-in-paper.md |
| D027 | Review-round corrections to Phase 2.5, and one committed file that should not have been | bug-fix | implemented | archived/bug-fix/D027-phase2-5-review-round-corrections.md |
| D028 | `deposit_table` bins directly onto `ahat`; the predecessor's fixed-range `retarget_a0`/`a0_kind` regrid is not ported, but the underlying peak-independence it exploited is — for both `ahat` and `luminosity` | architecture | implemented | archived/architecture/D028-deposit-table-direct-ahat-deposit.md |
| D029 | Stage 2's numpy kernel ports the predecessor's brute-force grid quadrature, not the GPU importance sampler; `KERNEL_NORMALIZATION_CONSTANT` isolates the pending §9.1 factor | architecture | implemented | implemented/architecture/D029-stage2-numpy-kernel-brute-force.md |
| D030 | `Collision`'s cache is per-instance memoization only; no cross-call staleness detection | architecture | implemented | implemented/architecture/D030-collision-cache-per-instance-only.md |
| D031 | `XigmaEngine` is not passed to `run_suite()` by `validation.run.main()` | testing | implemented | implemented/testing/D031-xigma-not-wired-into-run-suite.md |
| D032 | Stage 1 deposits onto `a0_shape`, not `ahat`; a conservative regrid (`retarget_ahat`) onto a fixed, non-uniform `ahat` axis replaces the direct deposit — supersedes D028 | architecture | implemented | implemented/architecture/D032-shape-table-retarget-ahat.md |
| D033 | §9.1 closed: both transcriptions of the paper's cross-section carry `1/(2 pi)`; the identity harness now expects one | bug-fix | implemented | implemented/bug-fix/D033-9-1-closed-both-transcriptions-1-over-2pi.md |
| D034 | A crossing angle warns that only the geometry is applied (§9.3's half of P14c) | bug-fix | implemented | implemented/bug-fix/D034-crossing-angle-warns-geometry-only.md |
| D035 | analytical ports the predecessor's round-beam, no-displacement approximations unchanged; the growth items stay open | architecture | implemented | implemented/architecture/D035-analytical-ports-round-beam-approximations.md |
| D036 | `SPECTRUM`'s grid integral is rescaled to `estimate_yield`'s total by construction, not as a discrepancy patch | architecture | implemented | implemented/architecture/D036-spectrum-rescaled-to-yield-by-construction.md |
| D037 | `erfcx` is hand-rolled from `math.erfc`, not a new `scipy` dependency | process | implemented | implemented/process/D037-erfcx-hand-rolled.md |
| D038 | the spectrum-width breakdown's `theta_col` is the geometric mean of `Target`'s x/y collimation half-angles | architecture | implemented | implemented/architecture/D038-theta-col-geometric-mean.md |
| D039 | the analytical yield is a general overlap quadrature, not the round-beam closed form | feature | implemented | implemented/feature/D039-general-overlap-quadrature-yield.md |
| D040 | `estimate_yield`'s laser-divergence convention error is flagged and pinned, not fixed | bug-fix | implemented | implemented/bug-fix/D040-estimate-yield-rayleigh-convention-pinned.md |
| D041 | the crossing angle is covered for the yield, by quadratic form; the spectrum stays head-on and says so | feature | implemented | implemented/feature/D041-crossing-angle-yield-quadratic-form.md |
| D042 | the width's nonlinearity term uses a luminosity-weighted `<a0^2>`, computed, not approximated | feature | implemented | implemented/feature/D042-luminosity-weighted-a0-squared.md |
| D043 | three explicit cost tiers, and the exact 2D quadrature is opt-in | feature | implemented | implemented/feature/D043-three-cost-tiers-2d-quadrature-opt-in.md |
| D044 | a flying focus is always evaluated on the 2D `(z, ct)` grid; the 1D shortcut exists but is not shipped | feature | implemented | implemented/feature/D044-flying-focus-2d-grid.md |
| D045 | the luminosity-weight prefilter is a second, tolerance-based filter, not a replacement for the cone | feature | implemented | implemented/feature/D045-luminosity-weight-prefilter.md |
| D046 | misalignment lives on the laser as `x_off`/`y_off`/`t_off`, and there is deliberately no `z_off` | feature | implemented | implemented/feature/D046-misalignment-x-off-y-off-t-off.md |
| D047 | the illumination filter tests the *bright* region, which is what the cone approximates | feature | implemented | implemented/feature/D047-illumination-filter-bright-region.md |
| D048 | the illumination window is the primary result; the filter is a corollary of it | feature | implemented | implemented/feature/D048-illumination-window-primary-result.md |
| D049 | the mean red-shift is exact; its spread is reported as a measured bracket | feature | implemented | implemented/feature/D049-mean-redshift-exact-spread-bracket.md |
| D050 | xigma can sample trajectories over the illuminated window, opt-in | feature | implemented | implemented/feature/D050-xigma-illuminated-window-sampling.md |
| D051 | discarding charge is reported to the user, never decided silently | feature | implemented | implemented/feature/D051-discarding-charge-reported-not-decided.md |
| D052 | *OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION* removed; `COLLIMATED_SPECTRUM` is now the sole `(E, θx, θy)` output kind | simplification | implemented | implemented/simplification/D052-remove-spectral-angular-distribution.md |
| D053 | `ahat` carries the polarization cycle average; the paper was right and the code was short a factor 1/2 | bug-fix | implemented | implemented/bug-fix/D053-ahat-polarization-cycle-average.md |
| D054 | physics goes through `<a^2>`, not `a0`: the polarization convention is removed from the yield/red-shift path rather than documented | architecture | implemented | implemented/architecture/D054-physics-through-a-squared-not-a0.md |
| D055 | Doc-staleness guard scope: every `proposed`/`implemented`/`rejected` decision file, not `archived/`; supersedes D002 | testing | implemented | implemented/testing/D055-doc-staleness-guard-scope-decisions-tree.md |
| D056 | Code comments citing a decision are trimmed to a pointer; the rationale lives only in the decision file | process | implemented | implemented/process/D056-code-comments-citing-a-decision-are-pointers.md |
