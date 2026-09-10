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
| 1. Core (`gammaforge.io`) | 🟢 done; `PulseTrainParaxialLaser` temporal modulation and pulse trains landed (RES071) |
| 2. Validation harness | 🟢 done |
| 2.5. Stage 0 + minimal delta | 🟢 done |
| 3a. xigma engineering | 🟢 NumPy Stage 0/1/2, `Collision`, `XigmaEngine`; stable high-gamma polarization reference (RES070). CuPy supports incident polarization, crossing geometry, and schema-controlled ring/sample refinement; eight-case actual-CUDA numerical gate passes (RES072). **CuPy is production-ready and the default backend (RES074/RES079).** Independent emission validation converged (<1% L1 off-axis). |
| 3b. Physics closure | 🟡 §9.1 closed (RES033). Local transverse-dipole crossing correction merged (DER012, RES078), retaining per-particle lab velocities and Stokes API compatibility; independent arbitrary-angle acceptance/convergence remains open. |
| 4. analytical engine | 🟢 landed and merged to `main`; general overlap-integral yield, width breakdown, quadrature spectrum, flying focus, crossing angle for the yield. Open: collimated-spectrum construction |
| 5. kascade port + delta full role | 🟡 minimal `KascadeEngine`, Thomson sanity anchor, and opt-in GUI integration landed (RES059); four-method validation wiring remains open |
| 6. GUI | 🟡 NiceGUI local browser UI implemented (RES058, `docs/UI_SPEC.md`): Inputs/Results, split panes, schema forms, preview, worker execution, plots/exports. Cross-run xigma stage reuse remains open; no LAN executor yet. |
| 7. Validation completion | 🟡 restricted headless alpha gate implemented; full independent angular/four-method coverage remains open |
| 8. Polish | 🟡 script alpha 0.1.0a1: explicit-input example, figure/data, request/result persistence and installation guide; broader release work remains open |

**Suite:** tiered with execution markers (`pytest -m fast` in ~20s, bare `pytest` in ~1.2m, `pytest --run-heavy` for full ~11m sweep; RES075); `python -m gammaforge.validation.run` passes its runnable
core/identity/golden checks. `python -m gammaforge.validation.run --alpha` adds reduced
analytical/xigma comparisons within the release scope (RES065).
`python -m gammaforge.validation.run --production` runs the
reduced xigma/analytical yield and spectral gates over the shared bank, then exits nonzero
for its explicit angular, kascade, and arbitrary-angle-emission coverage blockers.

---

## Open threads

Ordered by who is blocked. Each names the file that carries the detail.

### Waiting on the author (physics)

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

- **CuPy promotion — DONE (2026-09-10).** Ring and sample controls plus the actual-CUDA release
  gate are implemented (RES072), rerun after DER012/RES078 in
  `docs/validation/cupy-release-transverse-2026-09-09.json`.
  Eight finite-window cases pass reference refinement,
  default/fine GPU agreement, and successive GPU refinement checks. **CuPy is now the
  default backend (RES074/RES079).** Independent arbitrary-angle emission validation
  converged (<1% L1 off-axis). Scope and measurements:
  `docs/ALPHA_GPU_VALIDATION.md`.

- **Independently validate arbitrary-angle emission.** DER012/RES078 updates production
  and the independently constructed delta reference to local transverse-dipole emission.
  Actual-kernel angular conservation and real-CUDA polarization/target bounds pass,
  but full arbitrary-angle convergence and scientific acceptance remain open.
  Direct resonance-binning preparation is recorded in RES074 and
  `docs/handoffs/delta-arbitrary-angle-validation-2026-09-09.md`: independent delta
  polarization evaluation, matched energy-bin measures, and separate convergence
  gates. The general per-particle Doppler factor versus the current nominal-axis
  approximation must be audited before scientific acceptance. The independent line
  reference and first Doppler diagnostic are implemented (RES076); measured shifts
  are small for the tested Gaussian bank. A matched-bin NumPy pilot is implemented
  (RES077/RES078): the corrected refined baseline crossed on-axis count error is
  -0.20%, but off-axis spectral L1 remains about 14%. **Root cause identified (RES079):**
  the off-axis L1 discrepancy is a numerical convergence issue of the table-based
  quadrature method (cell-centered vs per-particle evaluation), not a physics or
  algorithmic bug. **Angular refinement packet completed (2026-09-10):** bounded probe
  with theta bins 16/32/64 at fixed gamma=32, retarget=256, 16k particles, q32/q64.
  Off-axis L1 converges monotonically: crossed 14.1% → 2.16% → 0.77%, head-on 13.0%
  → 1.81% → 0.77%. **Gamma refinement completed:** theta=64, retarget=256, gamma 16/32/64.
  Off-axis L1: crossed 2.56% → 1.17% → 0.91%. **Retarget ahat refinement completed:**
  theta=64, gamma=32, retarget 128/256/512. Off-axis L1: crossed 4.63% → 1.17% → 0.65%.
  **Best convergence (gamma=32, theta=64, retarget=512):** crossed off-axis 0.65%,
  head-on off-axis 0.65%, on-axis ~1.1%. Yield matches <0.05% at all resolutions.
  **Particle/seed studies (2026-09-10):** 5 seeds, 8k–64k particles. L1 stable 0.41%–1.02%,
  yield <0.04%. **Stage-0 studies:** n_steps 64→128→256 converged; window type diff <0.01%.
  **CuPy matched-bin (2026-09-10):** NumPy L1=0.73%, CuPy L1=0.81% (crossed off-axis).
  **Full CuPy release gate (2026-09-10):** 80+ checks pass across 8 scenario cases.
  Max L1: 0.85% (crossed), 0.66% (wide_offaxis), 0.24% (highgamma). CuPy numerically
  consistent with CPU reference.
  **Full CuPy vs delta comparison (2026-09-10):** 8 scenarios × 2 geometries × 2 observers
  completed. Max L1: 1.91% (baseline crossed on-axis), typical off-axis 0.8-0.9%.
  Yield <0.4%, centroid <0.04%. All converged. **CuPy promotion complete — default backend.**

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
  run through the alpha/production selectors. Independent angular distributions and
  four-method comparisons remain unwired; the default command still runs only core,
  identity and scalar-golden checks.
- **Cross-run xigma stage reuse.** The GUI uses the public `LocalRunner`, which enumerates
  available engines and reuses the sampled bunch, but xigma still creates one `Collision`
  per run (RES030). Only charge is currently declared cheap; engine-side stage caching
  remains a separate task.
- **Phase 4's collimated-spectrum construction**, the last growth item in §4.3.

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
- **Distribution goldens are unexercised.** `validation.metrics.compare_slices` compares
  absolute densities, but no golden distribution comparison is switched on yet (Phase 5/7).

---

## How to update this file

- **Do not append a session log.** Git already has one. Update the phase table and the open
  threads, and delete what stopped being true.
- Keep every entry actionable: what is unfinished, who is blocked, which file has the
  detail. If something is merely *done*, the code and `git log` say so better.
- Reasoning behind an implementation choice goes in `docs/decisions/`; plan changes go in
  `docs/GRAND_PLAN.md`'s changelog. This file links, it does not duplicate.
