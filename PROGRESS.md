# GammaForge — Implementation Progress

**Current state and open threads only.** The session-by-session narrative lives in `git log`
— it is more accurate than a hand-written retelling and never goes stale. This file answers
two questions a fresh session actually needs: *what works right now*, and *what is
unfinished or waiting on someone*.

`docs/GRAND_PLAN.md` is the plan and owns the phase definitions (§11); `docs/decisions/`
(via `INDEX.md`) owns the reasoning behind implementation choices.

---

## Phase status

| Phase | Status |
|-------|--------|
| 0. Scaffold | 🟢 done |
| 1. Core (`gammaforge.io`) | 🟢 done; `PulseTrainParaxialLaser` temporal modulation and pulse trains landed (RES071); luminosity-aware adaptive sampling landed as an opt-in `SamplingSpec.strategy` (RES092) |
| 2. Validation harness | 🟢 done |
| 2.5. Stage 0 + minimal delta | 🟢 done |
| 3a. xigma engineering | 🟢 NumPy/CuPy Stage 0/1 execution with host stage boundaries (RES083), five-dimensional shape/physical tables, Stage-2 raw spectral moments, nonuniform second-order reconstruction, persistent adaptive queries, `Collision`, and `XigmaEngine`. Stage 0 accepts carrier phase gradients and accumulates the weighted statistics in DER016; Stage 1 stores raw nonlinear shape and carrier mean with co-shaped moment channels; Stage 2 applies exact observer-dependent $Q$ (DER015/RES090). Gamma-resonance sampling (RES080) retains the ring-shaped proposal while reusing static backend/QMC state. The real-CUDA release gate includes quantitative `moment2` reconstruction agreement; scientific acceptance remains in Phase 3b. |
| 3b. Physics closure | 🟡 §9.1 closed (RES033). Local transverse-dipole crossing correction, direction Doppler, exact observer-dependent nonlinear incidence, and carrier/finite-line moments are implemented (DER012–DER013, DER015–DER017; RES078, RES082, RES090), retaining per-particle lab velocities and Stokes API compatibility; independent arbitrary-angle acceptance/convergence remains open. |
| 4. analytical engine | 🟢 landed and merged to `main`; general overlap-integral yield, width breakdown, quadrature spectrum, flying focus, crossing angle for the yield. Open: collimated-spectrum construction |
| 5. kascade port + delta full role | 🟡 minimal `KascadeEngine`, Thomson sanity anchor, and opt-in GUI integration landed (RES059); independent CuPy delta reference with a real-CUDA agreement gate (RES085); four-method validation wiring remains open |
| 6. GUI | 🟡 NiceGUI local browser UI implemented (RES058, RES089, `docs/UI_SPEC.md`): Inputs/Results, split panes, schema forms, sectional persistent defaults, visible Auto/manual output ranges, preview, worker execution, plots/exports. Cross-run xigma stage reuse remains open; no LAN executor yet. |
| 7. Validation completion | 🟡 restricted headless alpha gate implemented; independent fixed-direction delta gates wired (RES084); full angular/four-method coverage remains open |
| 8. Polish | 🟡 script alpha 0.1.0a1: explicit-input example, figure/data, request/result persistence and installation guide; broader release work remains open |

**Suite:** tiered with execution markers (`pytest -m fast` in ~20s, bare `pytest` in ~1.1m, `pytest --run-heavy` for full ~6m sweep; RES075); `python -m gammaforge.validation.run` passes its runnable
core/identity/golden checks. `python -m gammaforge.validation.run --alpha` adds reduced
analytical/xigma comparisons within the release scope (RES065).
`python -m gammaforge.validation.run --production` runs the
reduced xigma/analytical gates plus independent xigma/delta matched-bin count, spectral-L1
and centroid gates over the shared bank (RES084). It checks energy-quadrature, angular-table
and retarget refinement, and exits nonzero for failed numerical checks or remaining
sampling/integration, angular-aperture, independent CUDA and four-method coverage gaps.

---

## Open threads

Ordered by who is blocked. Each names the file that carries the detail.

### Waiting on the author (physics)

- **Synchronize the manuscript resonance and finite-line derivations.** The repository
  now implements the author-supplied exact observer-dependent $Q$, carrier-weighted
  trajectory moments, and second-order spectral reconstruction (DER015–DER017, RES090).
  Transcribe the settled formulas into the paper before claiming manuscript-level
  closure; any formula conflict is blocking under P14.

- **`ahat_decades`.** RES032's grid defaults were tuned against `ahat` values that RES053 later
  halved, so the bank now sits in the grid's coarse floor. Measured centroid bias ~1%,
  pre-existing rather than introduced; `decades = 1.0 -> 0.3` removes most of it. Pinned by
  test at the current value; changing a tuned default is the author's call. RES053's last
  section has the sweep.

- **Structured light polarization kernels (RES067).** Quasi-monochromatic non-Gaussian
  fields execute through Stage 0, Stage 2, and Kascade without Gaussian fitting (RES067),
  and broadband pulses are confirmed out of scope. Spatially inhomogeneous polarization
  kernels (e.g. radial/azimuthal vortex beams) remain open pending author derivations.

### Available to pick up (no external dependency)

- **Complete independent arbitrary-angle scientific acceptance.** DER012/RES078's
  transverse-dipole model, RES082's direction Doppler, and DER015/RES090's exact
  observer-dependent nonlinear coefficient now have matched-bin evidence
  for the baseline, low-a0 and near-a0-max bank cases, head-on and small crossed
  geometry, on/off axis. The production runner now measures these plus crossed circular
  polarization, with explicit agreement and refinement gates (RES084, RES092). The
  bounded finer grid resolves all six low-a0 refinement failures without changing
  the budgets: the full production run passes 388 executed checks and reports two
  remaining scientific coverage blockers. Evidence:
  `docs/validation/delta-production-2026-09-29.md`.
  The separate nine-case CPU/CUDA release gate passes all 90 checks, including
  reconstructed finite-line spectra; evidence and runtime measurements are in
  `docs/validation/cupy-release-2026-09-28.md`. Finish particle/seed, Stage-0,
  gamma/shape-grid, angular-aperture and independent CUDA convergence and RES074's
  acceptance requirements before closing Phase 3b.

- **Spatial autoranging under displacement.** The current Kascade reproduction captured its
  sampled photon weights inside the symmetric auto range for laser offsets through 120 µm;
  no geometry correction is established. A future change must first trace the event-source
  coordinate through a displaced overlap, rather than infer a shifted Gaussian range.

- **Finish Phase 5 validation wiring.** `KascadeEngine` supplies the independent
  overlap/emission leg and passes its Thomson-limit anchor. Delta's full role and the
  four-method scenario-bank comparison still need wiring into `run_suite()`.
- **Finish MC export.** Current Bunch/photon payloads round-trip through HDF5 (RES063);
  elegant-compatible final-electron export and future typing remain the format items
  in GRAND_PLAN.md §8/§10.8. Kascade is outside alpha support.
- **Extend production coverage.** Reduced analytical/xigma yield and spectral comparisons
  run through the alpha/production selectors; production additionally checks independent
  fixed-direction delta spectra. Angular-aperture distributions and four-method comparisons
  remain unwired; the default command still runs only core,
  identity and scalar-golden checks.
- **Cross-run xigma stage reuse.** The GUI uses the public `LocalRunner`, which enumerates
  available engines and reuses the sampled bunch, but xigma still creates one `Collision`
  per run (RES030). Only charge is currently declared cheap; engine-side stage caching
  remains a separate task.
- **Phase 4's collimated-spectrum construction**, the last growth item in §4.3.

- **Adaptive sampling: measured limits before any default promotion (RES092).** The
  luminosity-aware stratifier is implemented, tested, and **opt-in** (`SamplingSpec.strategy
  = "adaptive"`); `"iid"` remains the default. On the baseline scenario it beats IID on the
  total yield by ~1.3-5x in Stage-0 particles for a given yield target, and is at **parity**
  on the resolved spectrum — the spectral residual is dominated by per-cell particle counts,
  which reallocation does not change, and the refinement criterion `P_m sigma_m` sees
  luminosity variation rather than Stage-1 cell spread. The per-coordinate latent marginals
  improve up to ~10x; the radial (chi-squared-6) distribution does not, because a cell whose
  corner reaches `|d| = inf` spans the whole range of `r^2`. Promoting the default needs the
  wider scenario bank, and a Stage-1-spread-aware refinement criterion (the pilot already
  computes `a0_shape` and `chirp_mean`). `scripts/benchmark_adaptive_sampling.py` reports
  the accuracy-versus-cost table; RES092 records the reasoning.

---

## Known gaps in the checks themselves

Worth keeping visible: these are places where a green suite proves less than it looks.

- **Shared inputs are common-mode.** xigma and delta both read `ahat` from the same
  `TrajectorySamples`, so an error in it cancels in every xigma-vs-delta comparison. This
  is how RES053's factor of two survived the whole suite. `GRAND_PLAN.md` §7 records it.
- **Integrated observables hide redistribution.** The red-shift moves photons along `s`
  while conserving their number, so count-based checks are blind to it: scaling `ahat` over
  an 8x range moves `run.py`'s fourth identity leg by 0.11% total. Centroid-based checks
  exist now for exactly this reason.
- **Four-method distribution coverage is incomplete.** `validation.metrics.compare_slices`
  compares absolute densities, but the full xigma/delta/analytical/kascade scenario-bank
  comparison is not switched on yet (Phase 5/7).

---

## How to update this file

- **Do not append a session log.** Git already has one. Update the phase table and the open
  threads, and delete what stopped being true.
- Keep every entry actionable: what is unfinished, who is blocked, which file has the
  detail. If something is merely *done*, the code and `git log` say so better.
- Reasoning behind an implementation choice goes in `docs/decisions/`; plan changes go in
  `docs/GRAND_PLAN.md`'s changelog. This file links, it does not duplicate.
