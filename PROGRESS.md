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
| 1. Core (`gammaforge.io`) | 🟢 done |
| 2. Validation harness | 🟢 done |
| 2.5. Stage 0 + minimal delta | 🟢 done |
| 3a. xigma engineering | 🟢 done — Stage 0/1/2, `Collision`, `XigmaEngine` |
| 3b. Physics closure | 🟢 §9.1 closed (RES033); §9.2 closed (DER004/DER006); §9.3 closed (DER005/DER006) |
| 4. analytical engine | 🟢 landed and merged to `main`; general overlap-integral yield, width breakdown, quadrature spectrum, flying focus, crossing angle for the yield. Open: collimated-spectrum construction |
| 5. kascade port + delta full role | ⚪ not started — `engines/kascade/` is an empty package |
| 6. GUI | 🟡 NiceGUI local browser UI implemented (RES058, `docs/UI_SPEC.md`): Inputs/Results, split panes, schema forms, preview, worker execution, plots/exports. Cross-run xigma stage reuse remains open; no LAN executor yet. |
| 7. Validation completion | ⚪ not started |
| 8. Polish | ⚪ not started |

**Suite:** `pytest` green; `python -m gammaforge.validation.run` all checks pass.

---

## Open threads

Ordered by who is blocked. Each names the file that carries the detail.

### Waiting on the author (physics)

- **`ahat_decades`.** RES032's grid defaults were tuned against `ahat` values that RES053 later
  halved, so the bank now sits in the grid's coarse floor. Measured centroid bias ~1%,
  pre-existing rather than introduced; `decades = 1.0 -> 0.3` removes most of it. Pinned by
  test at the current value; changing a tuned default is the author's call. RES053's last
  section has the sweep.
- **`estimate_yield`'s future.** Kept only for port fidelity to the predecessor's worked
  example (RES040). If those published results no longer need reproducing, it and its three
  tests are a one-commit deletion.
### Available to pick up (no external dependency)

- **Phase 5, kascade.** The largest unblocked item, and the leg §9.3's §2.3 needs. Would
  also give §7 a genuinely independent Stage 0 — which RES053 showed the cross-validation
  currently lacks, since xigma and delta share `TrajectorySamples`.
- **Wire engines into `run_suite()`.** `validation.run.main()` still passes none (RES031), so
  the suite's green never exercises the cross-engine comparison; the xigma-vs-analytical
  agreement (0.31% on `ahat`, 0.32% on yield) has to be measured by hand.
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
