# GammaForge — Implementation Progress

Tracks what has actually been built, session by session. `docs/GRAND_PLAN.md` is the
plan (with its own changelog of *design* decisions); this file is the log of *execution*
against that plan. Update it every session — append, don't rewrite history.

Phase numbers/names match `docs/GRAND_PLAN.md` §11.

---

## Status at a glance

| Phase | Status |
|-------|--------|
| 0. Scaffold | 🟢 done — C4 re-checked 2026-08-08, still clean (see below) |
| 1. Core | 🟢 done — all exit criteria met (see 2026-08-07 session below) |
| 2. Validation harness | 🟢 done — both exit criteria met (see 2026-08-07 Phase 2 session) |
| 2.5. Stage 0 + minimal delta | 🟢 done — all four exit criteria met; §9.1 now a *derived* open question (see below) |
| 3a. xigma engineering | 🟢 done — Stage 1/2, `Collision`, `XigmaEngine` landed 2026-08-08 (see below) |
| 3b. Physics closure | 🟡 §9.1 done (2026-08-08) — §9.2/§9.3 blocked on the author; the paper has no formula for either, and both are non-blocking for 4–6 by design |
| 4. analytical engine | 🟢 landed 2026-08-09 (branch `worktree-phase4-analytical-engine`) — estimates, width breakdown, quadrature spectrum, and the general overlap-integral yield (non-round + displaced foci). Open: collimated-spectrum construction, and the width's nonlinearity term still uses peak a0 |
| 5. kascade port + delta full role | ⚪ not started |
| 6. GUI | ⚪ not started |
| 7. Validation completion | ⚪ not started |
| 8. Polish | ⚪ not started |

---

## 2026-08-07 — Plan v0.8 + Phase 0 kickoff

**Plan changes (before any code):**
- Grand plan reviewed end-to-end across several rounds (v0.2 → v0.7): fact-checked
  against the predecessor repo (`ComptonSuite`) and the physics paper, phase-ordering
  bug fixed (Phase 2.5 no longer depends on Stage 0 before Stage 0 exists — Stage 0 +
  the shared chunking utility moved into 2.5), RNG substream architecture pinned so the
  bunch resample rule doesn't silently break the `REUSE_INTERMEDIATES` cost tier.
- v0.8: laser field source made pluggable — new `LaserField` protocol is what engines
  depend on; `GaussianParaxialLaser` reframed as its first implementation, not *the*
  laser type; `fit_gaussian_paraxial` added (P8-style: descriptive Gaussian metrics —
  waist, Rayleigh range — extracted from *any* `LaserField`, not just an analytic one).
  Anticipates `~/Work/Code/Spectral-FEM-Fields` (sibling C++ project, Python bindings
  planned) as a future second implementation for arbitrary non-paraxial pulses, at the
  interface level only — no work on it yet.

**Phase 0 work (this session):**
- `pyproject.toml`: hatchling build backend, src layout, Python ≥3.12, core deps
  (numpy/pint/h5py/pyyaml/matplotlib), optional extras `gpu` (cupy) / `jit` (numba) /
  `dev` (pytest). Rationale for hatchling over setuptools: `DECISIONS.md` D001.
- Package skeleton: `src/gammaforge/{io,engines/{xigma,analytical,kascade},
  validation/references,gui}/`, each a docstring-only `__init__.py` pointing back at
  its `GRAND_PLAN.md` section. Deliberately no placeholder module files yet —
  `DECISIONS.md` D003.
- `.gitignore` hardened: `*.ele`/`*.bun`/`*.h5`/`*.hdf5` and sync-conflict filename
  patterns, per C3 (the predecessor accumulated multi-MB committed artifacts this way).
- `DECISIONS.md` started (C1) — D001–D003 so far, with an explicit backticks-vs-italics
  convention so the doc-staleness guard can trust every backtick literally.
- `tests/test_smoke.py`: package + all subpackages import.
- `tests/test_doc_staleness.py` (C2): resolves backticked tokens in `DECISIONS.md`
  against real repo files / package symbols / builtins. Scope is `DECISIONS.md` only,
  not all of `docs/*.md` as originally checklisted below — `GRAND_PLAN.md` is a
  forward-looking roadmap and would fail the check by design, not by drift; see
  `DECISIONS.md` D002 for the full reasoning.
- `AGENTS.md` written (repo orientation + the plan's easy-to-violate rules, condensed
  for a fresh agent) with `CLAUDE.md` as a symlink to it.
- Verified clean: `python -m venv .venv && pip install -e .` succeeds, `pytest` → 3
  passed.

**Not done / explicitly deferred:**
- C4's `git branch -a` re-check on `ComptonSuite`'s worktree branches — not needed
  until Phase 3a/6 kickoff, already audited once (see `GRAND_PLAN.md` §11 note); just a
  reminder not to skip re-checking then.
- No CI wiring yet (the doc-staleness test and any future import-boundary/mypy checks
  run locally via `pytest` only). Add a CI workflow when there's a remote to run it on.

### Phase 0 checklist

- [x] `pyproject.toml` (Python 3.12, package skeleton, pytest)
- [x] `src/gammaforge/{io,engines/{xigma,analytical,kascade},validation,gui}/` skeleton
- [x] `.gitignore` hardened for `.ele` files and sync-conflict patterns (C3)
- [x] `DECISIONS.md` provenance doc started (C1)
- [x] `pytest` green on empty-suite smoke test
- [x] `pip install -e .` works
- [x] Doc-staleness guard scaffolding (C2) — scoped to `DECISIONS.md`, see above
- [x] `AGENTS.md` + `CLAUDE.md` symlink
- [ ] `git branch -a` re-check on `ComptonSuite` before Phase 3a/6 kickoff (C4) — not
      needed yet, noted here so it isn't forgotten

---

## 2026-08-07 — Phase 1: the `gammaforge.io` core

Plan bumped to **v0.9** — see its changelog for the design details Phase 1 pinned that
the plan had left as sketches. Implementation-level rationale with rejected alternatives
is in `DECISIONS.md` **D004–D011**.

**Built (all of `src/gammaforge/io/`):**

- `units.py` — CGS-Gaussian constants pulled from pint's CODATA table, plus the one
  hand-applied EM conversion (`1 C = c[cm/s] / 10 statC`). Width/duration convention
  vocabulary and algebra. Two pint boundary contexts: the ported `light_time` and a new
  `gaussian_charge` (D008), which is what lets a user enter a bunch charge in pC.
- `schema.py` — `FieldSpec`/`Parameters`: typed, validated, immutable, with unit and
  convention conversion confined to the boundary. Unit strings are checked at *declaration*
  time, so a bad display unit fails at import rather than on first GUI interaction.
- `beam.py` — `GaussianElectronBeam`, `Bunch`, sampling with per-variable RNG substreams,
  the prefilter, propagation, and the covariance fit (with a scipy-free chi²/KS
  implementation, since scipy is not a declared dependency).
- `laser.py` — the `LaserField` protocol, §2.2 geometry, and `GaussianParaxialLaser`
  (elliptical, astigmatic, flying-focus) as its sole implementation.
- `results.py` / `target.py` / `interaction.py` — the results contract, the `OutputKind`
  vocabulary with auto-ranging, and the compiled interaction bundle.
- `fields.py` — the beam/laser/sampling `FieldSpec` sets, declared once and shared.
- `formats/` — YAML specs (units explicit at the file boundary, driven entirely off
  `fields.py`), elegant `.ele` I/O, and the HDF5 results writer with a YAML sidecar.

**Phase 1 exit criteria (`GRAND_PLAN.md` §11), all met:**

- [x] Round-trip tests — YAML spec, `.ele`, HDF5 results, and the dataclass↔`Parameters`
      bridges
- [x] Schema validation tests
- [x] CGS conversion tests vs known values — constants cross-checked against *independent*
      CGS identities (`r_e = e²/mec²`, `σ_T = 8π/3 r_e²`, `α = e²/ħc`) rather than
      re-asserting pint's own digits, which would only test that pint is pint
- [x] `LaserField` protocol conformance for `GaussianParaxialLaser`, period-averaged and
      period-resolved at arbitrary points
- [x] `fit_gaussian_paraxial` identity test on a `GaussianParaxialLaser` input
- [x] Geometry round-trip test (`R⁻¹` recovers the head-on configuration, §2.2)

`pytest` → **155 passed**.

**Three bugs the tests caught, worth recording because none was visible by inspection:**

1. `Axis` members sharing a unit string became silent `Enum` **aliases** — `Axis.Y is
   Axis.X` was true, so every 2D slice lost an axis (D007).
2. The drift transport of the energy correlation dropped its `rho` factor. Only found by
   checking the analytically drifted description against a refit of the drifted
   macroparticles — which is now a test, parameterized over both drift directions.
3. `.ele` header scalars were written at 6 significant digits (the predecessor's format).
   Since every particle's gamma is reconstructed as `gamma0 * (1 + dP)` from the `Energy`
   header, that capped energy round-trip accuracy at ~1e-7. Now 12 digits.

**Also this session:** the Phase-0 doc-staleness guard was hardened (D011). Its first-match
shape rule broke as soon as the docs had real symbols to reference — `FieldKind.CHOICE`
matches the "file-like" shape exactly as `GRAND_PLAN.md` matches the "class attribute"
shape. It now tries every applicable interpretation, knows about enum members and
annotation-only dataclass fields, and has three tests pinning its *own* behaviour, since a
weakened guard fails silently. It immediately earned its keep by catching a backtick in
D011 that the convention required to be italics.

**Not done / deferred:**

- No `fit_gaussian_paraxial` numerical path — identity only, deliberately (D010).
- `ellipticity` is carried and warned about but not applied (§9.2/P14c); flip
  `laser.ELLIPTICITY_IS_NOOP` when the derivation lands.
- The crossing-angle *geometry* is fully implemented and tested; the crossing-angle
  *physics* (§9.3) belongs to xigma's Stage-2 kernel and is untouched here.
- No engine consumes any of this yet, so the §5 recompute-cost tiers are supported by the
  RNG-substream test but not yet exercised by a real cache.
- Still no CI (§11's import-boundary check is a Phase 6 item); `pytest` runs locally.

---

## 2026-08-07 — Propagation round-trip tests, and the bug they found

Prompted by the question "do we have tests that propagate a bunch, refit it, and confirm
the analytical parameter change is right?". There was one such test, but it was too weak
to be load-bearing, and strengthening it exposed a real defect. Plan bumped to **v0.10**;
rationale in `DECISIONS.md` **D012**.

**Why the original test was weak.** It drifted the bunch, refit it, and compared against
the *parent beam* transported analytically. That folds the bunch's own sampling noise into
the comparison, which forced a ~10% tolerance — loose enough to hide a great deal. Also,
`propagate` was never checked this way at all, and the transport was only ever exercised
for a single step.

**The reformulation.** Anchor the comparison on the bunch's **own** fitted moments rather
than its parent beam. Sampling noise is then common to both sides and cancels, which
splits one loose tolerance into three genuinely different precision classes:

| | agreement | why |
|---|---|---|
| `drift`: sizes, Twiss, emittance | ~1e-15 | exact algebraic identities on second moments, not statements about a Gaussian parent |
| `drift`: energy correlations | sampling noise | needs `cov(x', gamma)`, supplied by the model, exact only for the parent distribution |
| `propagate`: everything | ~1e-9 | per-particle `vz` vs the ensemble reference step `c·dt` |

**The bug.** Asking the transport to *compose* — `drift(25) + drift(25)` versus
`drift(50)` — failed immediately: `rho_x_gamma` came out 0.238 against a true 0.187.
`_drift_plane` was re-deriving `cov(x', gamma)` from `alpha` at every step, on the
assumption that gamma couples to the angle only through the position. That holds for a
freshly sampled bunch and is **destroyed by the first drift**, so a single drift agreed
with a refit while consecutive drifts silently did not. The original test could not have
caught it: it only ever drifted once.

**The fix** (D012): store `rho_thx_gamma`/`rho_thy_gamma` — the dispersion derivative —
instead of re-deriving them. They are genuinely independent parameters (a bunch created at
a waist with dispersion but no dispersion derivative keeps `rho_thx_gamma = 0` however far
it drifts, while `alpha_x` and `rho_x_gamma` both change), so the old formula was silently
imposing a relation that does not hold. The sampler now draws gamma conditionally on all
five variables (`gamma_coefficients`), `fit_gaussian` extracts them, and `validate` reports
the joint admissibility condition in the user's own terms. `drift` now composes and
reverses to round-off (~1e-16).

**Tests added** (`pytest` → **167 passed**): exactness of the size/Twiss transport,
correlation transport within sampling noise, composition, reversibility, `propagate`
against a refit, `stream` snapshot consistency, and a check that `propagate`'s residual
error vanishes as the divergence shrinks — confirming it is the documented
ensemble-reference approximation and not a wrong formula. The exponent of that scaling is
deliberately *not* asserted: it is 2 or 4 depending on whether `sigma_x` or `L·sigma_theta`
dominates the drifted size, so pinning it would test the regime rather than the physics.

Also added: a physical invariant that pins the transport's *direction*, not merely its
self-consistency — with no dispersion derivative, `cov(x, gamma)` cannot change under a
drift at any length.

---

## 2026-08-07 — Units architecture reviewed; k0 normalization dropped

Two author-directed decisions after reviewing what Phase 1 actually built. Plan bumped to
**v0.11**; rationale in `DECISIONS.md` **D013–D015**.

### The units decision was never made — it was over-read

The author asked where "shared dataclasses hold plain floats; kernels never see a
`Quantity`" was decided. Checking: `GRAND_PLAN.md` §2.1 constrains **kernels**, and P1
says shared dataclasses store canonical CGS *values*. Neither says the shared layer holds
bare floats. The plan never mentions `PhysicalQuantity` at all — it never evaluated the
predecessor's approach, which was the opposite (every value crossing a model boundary
travelled as a `PhysicalQuantity`, deliberately). And no `DECISIONS.md` entry recorded the
choice, which is precisely what that file is for. **The docstring stating it was a
paraphrase presented as settled plan.**

On the merits the author was right: P1 removes the *multiplicity* of unit systems but not
the one conversion each engine must still perform at its own boundary, and kascade is SI
internally. Under bare floats that conversion is a hand-written `* 0.01` whose failure mode
is a silent factor of 100 — the exact class of error P1 exists to prevent, surviving at the
one place P1 does not reach.

**Now:** every dimensioned field of `GaussianElectronBeam`, `GaussianParaxialLaser` and
`Target` is a pint `Quantity`, validated and converted to canonical CGS on construction, so
P1 stays literally true while the type carries the unit. Each class unpacks through
`m(name)` once per routine; nothing below sees a `Quantity`.

**Bulk arrays deliberately stay raw** (D013). The author asked whether quantified numpy
arrays would work. Measured: `np.cov`, `np.corrcoef` and `np.linalg.slogdet` have no pint
implementation and `fit_gaussian` needs all three — but the decisive reason is structural,
not a library gap: the 6D beam covariance is **dimensionally heterogeneous** (`Sigma[x,x]`
cm², `Sigma[x,gamma]` cm, `Sigma[gamma,gamma]` dimensionless), so it cannot be one
`Quantity` in any units library. The author's alternative — carry the unit as metadata and
apply the scale factor where it is needed, since every conversion here is a pure scaling —
is what `Bunch.UNITS` + `Bunch.get` now do, and it is the same pattern `Axis` already used
for result slices. `get` returns the array itself, no copy, when the requested unit is the
stored one.

**A hole the new tests found immediately:** the `light_time` context was applied at *every*
boundary, so a **transverse** beam size was accepted in femtoseconds — not a unit choice
but a different physical quantity. It is now opt-in per field (D014); `gaussian_charge`
stays unconditional, since a value either is a charge or is not.

### k0_las normalization dropped (D015)

The author's claim — that normalization buys nothing once the laser owns lab-frame CGS
sampling and the bunch owns CGS trajectories — was checked against the predecessor's
pipeline and holds:

| stage | uses `k0_las`? | what for |
|---|---|---|
| Stage 0 trajectory core | `k0**2` in `contribution` | the **Jacobian** of the normalization: with `u = k0 r`, `tau = k0 c t`, `k0**2` is exactly what replaces `c` |
| Stage 0 `a0_shape` | no | a ratio of envelopes; k0 cancels |
| Stage 1 deposition | no | table axes are all dimensionless |
| Stage 2 kernel | **no** | no `omega`, `hbar` or energy anywhere; works in a dimensionless `s` |
| diagnostics binning | `/k0_las`, `/omega_las` | **un**-normalizing back to cm and s |
| adapter | `4 hbar omega0` | `s` → MeV at the boundary, already outside the kernel |

One refinement to the original claim: units re-enter in *two* places, not one — the
diagnostics binning **and** the spectrum energy axis — but the latter is already an ordinary
adapter-boundary conversion. An argument not raised but stronger than either: normalizing
would actively fight **P15**, since an arbitrary `LaserField` knows nothing about `k0_las`
and would need coordinates scaled on the way in and unscaled on the way out.

No code changed for this — Stage 0 is Phase 2.5, and the only `k0` in the current code is
the carrier wavenumber in `GaussianParaxialLaser.field`, which is genuine physics.
Remaining open item: float32 conditioning of the CGS integrand on the GPU path, to be
measured in Phase 2.5 rather than assumed.

### Also

The doc-staleness guard needed two more fixes, both the same class as before: it could not
resolve class-level names (`UNITS`, `LIGHT_TIME_FIELDS`) and it read third-party dotted
names (`np.cov`) as filenames. It now indexes class attributes and only treats a dotted
token as a path when its suffix is one the repo actually uses — derived from the repo, so it
stays self-maintaining. Its own behaviour tests were extended to cover both.

`pytest` → **176 passed**.

### Follow-up cleanups (same session)

Fixes for issues the discussion surfaced, plus two author-requested renames:

- **`beam.py` → `bunch.py`**, and the phrase "canonical bunch" removed (D016). It was
  unexplained jargon for "every engine in a run is given the same bunch", which the
  docstring now simply says. "Canonical" still qualifies *units*, a different and standard
  usage.
- **Each field's canonical unit is now stated once**, read by `fields.py` from the
  dataclass that stores it. Correction to something claimed earlier in the session: a
  mismatch would *not* have been a correctness bug — the boundary converts either way — so
  this removes redundancy, not a hazard.
- **Display units now lead with what a person would write**, since the first display unit
  is what a saved file uses. A pulse duration was being written as `8.994 um` (correct,
  unreadable); a bunch charge as `0.2998 statC`. Now `30 fs` and `100 pC`. Written values
  are rounded to 12 significant digits so a file says `100.0` rather than
  `99.99999999999999`.
- **`Parameters.merge()` deleted** — no consumer, and P6 rejects speculative machinery. It
  is one line to reinstate when something needs it.

The doc-staleness guard earned its keep twice more: it caught the `beam.py` reference in
D009 the moment the file moved, and then caught a backtick in the *new* D016 entry that the
convention required to be italics.

`pytest` → **183 passed**.

---

## 2026-08-07 — Phase 2: the validation harness

Plan bumped to **v0.12** — see its changelog for the three design items this session
settled (Engine protocol placement, the active-region cone, the golden-reference
subprocess). `DECISIONS.md` D018–D021 carry the rejected alternatives.

**Both §11 exit criteria are met:** golden generation runs (`python -m
gammaforge.validation.make_references`, nine snapshots in ~60 s), and new-vs-golden
comparisons execute — as `pytest`, as the orchestrator's own report, and against a stub
engine end to end.

### What landed

- **`engines/base.py`** — the §4.1 `Engine` protocol and the §5 `RecomputeCost` enum, one
  phase early because a runner skeleton needs a joint (D018). No registry.
- **`validation/scenarios.py`** — `Scenario` (beam, laser, target, sampling — *no* engine
  knobs, unlike the predecessor's, which carried one xigma field and one analytical field
  on a supposedly model-agnostic object) and the three-point bank `BASELINE` / `LOW_A0` /
  `NEAR_A0_MAX`. The operating point is the predecessor's own — gamma0 = 2000, 10 nC, 20 J
  at 1030 nm — written in CGS with explicit units rather than as converted literals. The
  scan varies pulse energy alone, so `a0` moves and nothing else does (0.057 / 0.181 /
  0.405).
- **`validation/metrics.py`** — the window-integrated deviation ported from the
  predecessor, reporting both a reference-flux-weighted L1 and a max-over-windows, plus
  `compare_slices` over `PhasespaceSlice`. Two numbers because one hides a localized
  defect: a bad Compton edge is invisible in a weighted average.
- **`validation/golden.py`** — snapshot format, provenance, and comparison. A golden is
  the ordinary results HDF5 plus a provenance group and the model's own scalars (D019).
- **`validation/make_references.py`** + **`_predecessor_driver.py`** — the one place the
  old repo is referenced. Nine goldens committed: three scenarios x
  {analytical, xigma, delta}.
- **`validation/invariance.py`** — the four §7 properties as engine-generic functions,
  plus the two that need no engine at all.
- **`validation/runners.py`**, **`validation/run.py`** — running one engine on one
  scenario, and the suite entry point with a pass/fail report.
- **`tests/test_validation.py`** — 38 tests. `pytest` → **221 passed**.

### A real Phase-1 bug, found by the harness on its first run

`check_prefilter_discards_only_dark_particles` samples `a0` along every *discarded*
trajectory and compares the peak against the threshold that discarded it — deliberately
independent of the closed-form geometry that made the decision. On a probe configuration
(1 mm beam, 30 fs pulse) it immediately reported a discarded macroparticle seeing
`a0 = 3.6e-2` against a threshold of `5.7e-3`: **six times** the bound.

The cause: `ActiveRegion` was a cylinder whose radius came from the spot within the
*pulse's own length* around focus. A pulse diverges. A bunch longer than the Rayleigh
range (0.3 cm vs 0.12 cm here — the baseline's own numbers) meets it far from focus, where
the spot is many times larger, so the filter was discarding particles the expanded pulse
still reached. That breaks the §3.2 contract outright: the prefilter is a *pure
optimization*, and every result computed with it on would have been quietly wrong.

Fixed by making the region a cone (D021). Post-fix the same probe reports `3.0e-6` against
the same threshold — 1900x under, rather than 6x over — while still discarding 1988 of
2000 particles, so the filter kept its value. Guarded twice now: at the laser level
(`test_active_region_is_still_conservative_far_from_focus`, which fails against the old
cylinder) and by the harness check itself.

Worth stating plainly, since it is the argument for the phase existing: this was not
caught by 183 passing Phase-1 tests. The nearest one sampled points within ±8e-3 cm of
focus — well inside the Rayleigh range — and passed for a configuration the bug does not
reach.

### What the goldens already say

The scalar cross-check is the sharpest comparison available and needs no engine: two
independently written codebases, the same physical input, the same analytic quantity.

| quantity | agreement |
|---|---|
| `n_photons` | exact (0 ULP on two of three scenarios) |
| `gamma0` | exact |
| `n_electrons` | 1.2e-16 |
| `a0_peak` | 6.6e-11 |

`a0_peak`'s 6.6e-11 is the only one not at machine precision, consistent with the
predecessor's hand-entered constants differing from pint's CODATA around the eleventh
digit. Tested at 1e-9, which leaves an order of magnitude of margin without being loose
enough to hide a convention error.

Two observations from the snapshot data itself, recorded rather than acted on:

- xigma and delta agree to **2.6e-8** (weighted L1) on every scenario. Expected — delta
  reuses xigma's Stage 0 and differs only in the Stage-2 kernel (§4.5) — but it means the
  pair is a check on the kernel, not independent evidence about the pipeline.
- The predecessor's **analytical** model gives a total yield **3.28x lower** than xigma
  and delta, uniformly across all three scenarios. This is the neighbourhood of the §9.1
  ~2pi normalization question, but 3.28 is not 2pi and the old analytical model's own
  code carries a flagged self-rescale hack, so it is not evidence for anything yet. Noted
  as the kind of thing the golden set exists to make visible; §9.1's arbiter is delta at
  Phase 2.5, and this does not pre-empt it.

### Deliberately not built

- **No result cache in the runners** (D020). The predecessor's commit-hash-keyed pickle
  store solved a problem this architecture does not have.
- **No `ENGINES` registry** — nothing to register until Phase 3a.
- **Chunk and backend invariance are exercised against stubs only**, which is all that is
  possible: neither the chunking utility (Phase 2.5) nor any real backend exists. The
  checks are written engine-generically and take the engine's own parameter key, so
  wiring them up in 2.5/3a is a call site, not new machinery.
- **No `visualize.py`** — the predecessor had one; plots belong to the shared plotting
  module (§8), which is Phase 6.

### Notes for the next session

- The prefilter is now correct but *conservative in a new place*: for a long bunch the
  cone's radius grows with distance from focus, so it discards less than the old (wrong)
  cylinder did. If a real engine run shows the filter earning too little, the tight fix is
  the exact cone solve rejected in D021, not a return to a fixed radius.
- `make_references` needs OLD_REPO and OLD_REPO_PYTHON. On this machine:
  `/home/alexander/Work/Code/ComptonSuite` and `/home/alexander/miniforge3/envs/core/bin/python`
  (the old repo needs scipy; this one deliberately has none). Goldens were generated from
  predecessor commit `4298070`, clean tree — asserted by a test, so a snapshot taken from a
  dirty tree cannot land unnoticed.

### Review round (same session)

A review of the Phase-2 diff returned eight findings; all eight were verified by running
the code and all eight are fixed. The two that mattered:

- **The active-region cone ignored the flying focus** — the same defect class as the one
  the harness had just found, one layer down. The spot is evaluated at `u + beta_ff*ct`,
  not at `u`, and the `(1 + beta_ff)` stretch already in the Rayleigh range makes the cone
  *shallower*: the omitted slide would have made it steeper by exactly that factor, so
  dropping one of the pair left the region narrower than the pulse. Measured at 1775 of
  4913 above-threshold points outside the region. Fixed (D022), now swept over `beta_ff`
  at both the laser and harness levels. Worth noting how it survived: the D021 fix
  rederived the transverse bound and carried the pre-existing `s(u)` reading forward
  without asking where the spot is actually evaluated.
- **The window metric could not see photons outside the reference's support** — inherited
  from the predecessor's `significant = win_flux_ref > floor`. A candidate carrying 0.12%
  of its yield past a hard reference edge scored zero on *both* reported numbers and
  passed a 2% tolerance, which is precisely the misplaced-Compton-edge case the module
  documents itself as catching (D023).

The rest, briefly: xigma and delta report `a0` where analytical reports `a0_interaction`,
so six of nine goldens carried **no scalars at all** while the suite reported a green
"golden scalars" section — the test asserted a total count rather than per-golden, so it
could not notice; mapping added, goldens regenerated, and the test now requires every
golden to carry something. The leaky stub engine keyed its offset on `hash()`, which is
salted per process, so two negative assertions failed on roughly 1 run in 49 (reproduced
at `PYTHONHASHSEED=45`). `compare_slices` crashed on a single-bin axis that `_total` went
out of its way to tolerate. `make_references` re-resolved scenarios through `by_name` —
crashing on a stale reference directory where `run.py` correctly skipped, and validating
the bank's scenario rather than a caller's modified one; both now share
`run.golden_scalar_checks`. And `_scenario_payload` silently dropped every geometry and
correlation field, so adding a crossing-angle scenario to the bank would have committed a
golden for a different physical configuration under its name — now refused with a message
naming the fields.

`pytest` → **230 passed**, and stable across hash seeds.

---

## 2026-08-07 — Phase 2.5: Stage 0, the chunking utility, and minimal delta

Plan changes: none. `DECISIONS.md` **D024–D025** carry the rejected alternatives.

**All four §11 exit criteria met.** Stage 0 tests green; chunk-invariance holds (exactly,
against a real stage rather than a stub); delta produces independent spectra on every
scenario in the bank; the identity harness runs as a section of
`python -m gammaforge.validation.run`.

### What landed

- **`engines/xigma/chunking.py`** — the one auto-chunk + OOM-retry utility §4.2 asks for,
  replacing the predecessor's three. Ported: proactive sizing with retry as backup, both
  backends chunked (numpy is *not* exempt — a 5M x 1024 broadcast is >100 GB, and on the
  host it meets the OOM-killer rather than a catchable error, so the estimate is the
  defence there and the retry barely a net), and a hard ceiling independent of free
  memory (measured: past chunk≈8–16 every doubling bought nothing while a 128 GB machine
  sized it into the hundreds).
- **`engines/xigma/stages.py`** — Stage 0, `integrate_trajectories`. Ballistic push over
  each particle's own overlap window (Phase 1's `overlap_time_window`, so the cone fix of
  D021/D022 is inherited rather than re-derived), midpoint-rule integration, producing
  per-particle `luminosity` and `a0_shape`.
- **`validation/references/delta.py`** — resonance binning per macroparticle, its angular
  integral, the closed-form single-electron anchor, and `check_normalization`.
- **`tests/test_stage0_delta.py`** — 26 tests. `pytest` → **256 passed**.

### The CGS reformulation is confirmed against real numbers

D015 dropped the predecessor's `k0_las` normalization on an argument. Stage 0 now tests
it: **total yield agrees with the predecessor's xigma golden to 0.12%**, with the *same*
relative offset (+1.248e-3) on all three scenarios — systematic, as it must be, since the
two repos draw different bunches from the same nominal seed and bound the interaction
window differently. The predecessor's `k0**2` really was the Jacobian of its own
coordinate change and nothing more.

Stage 0 also reads the entire laser through `a0_profile` (D024). The pulse energy cancels
out of the density it needs, so P15 costs nothing here — no envelope formula, no spot
sizes, no Gaussian assumption in the engine.

### §9.1: the ~2pi is exactly 2pi, and now derived

The predecessor recorded delta's angle-integrated total as "consistently ~6.3x
[the table-free spectrum] -- suspiciously close to 2*pi, not yet explained". It is not
close to 2pi. With `u = gamma**2 r**2` and the azimuthal average
`<a_fac> = 1 - 2u/(1+u)**2`:

    int dOmega 3 gamma**2 <a_fac> / (1+u)**2 = 3 pi [1 - 2*(1/6)] = 2 pi

exactly. Measured here at 5.85–6.23 depending on how much of the cone the angular grid
covers, tracking `2 pi x captured_fraction` to within ~1%. Reproducing it from an
independent CGS implementation also **rules out the old repo's coordinate normalization
as the cause**.

Which side counts photons is not ambiguous: Stage 0's total is
`flux x cross-section x time` summed, and the closed-form single-electron spectrum
integrates to exactly that same number by an identity (its shape integrates to 1). Two
independent methods agree; the third carries a differential solid-angle measure and an
extra 2pi, which is what an azimuthal integral counted twice looks like.

**No factor was applied** (D025). The derivation says the two are inconsistent and which
one is the count; it does not say which normalization xigma's *Stage-2 kernel* should
carry, and that kernel does not exist yet (Phase 3a). The identity harness reports the
ratio against its derived value, so the suite stays green while the question is open and
turns red the moment the ratio moves — the predecessor's version survived unexplained
precisely because nothing would have noticed it changing.

**For the author:** the open question is now a convention question, not a numerical one —
does the paper's Stage-2 kernel normalization carry the `1/(2 pi)` azimuthal measure that
delta's prefactor is missing? That is the §9.1 decision, and it wants the paper, not more
code.

### A Stage 0 bug the prefilter-invariance property caught immediately

A particle that never enters the pulse gets an empty window, reported as `t0 = +inf`,
`t1 = -inf`. Its span is zero so it contributes nothing — but `inf + 0*0` is still `inf`,
and a trajectory evaluated there hands the laser a NaN that propagates into **every**
particle's sum. Invisible with the prefilter on, since those particles are already gone;
the §7 prefilter-invariance property is what exposed it, on a wide-bunch configuration.
Empty windows are now anchored at a finite time and the zero span does the rest.

### Deliberately not built

- **No cupy or numba Stage 0.** `backend='cupy'` raises rather than silently running on
  the host — which would make the §7 backend-agreement leg pass while comparing numpy
  with numpy. A GPU Stage 0 needs `LaserField.a0_profile` to accept device arrays, a
  change to the *protocol's* contract that every implementation inherits (P15), so it
  belongs with the Phase 3a kernels, not to a wrapper here.
- **No Stage 1 or Stage 2** — Phase 3a, as planned. The identity harness therefore has
  three legs today (Stage 0 total, closed form, delta); `kernel` and `reference` join
  when they exist.
- **No `Collision` facade, no engine registration.** xigma is not an `Engine` yet, so
  the §7 engine sections still report an empty engine list.

### Notes for the next session

- The float32-conditioning question for the GPU path (carried since Phase 1) is still
  open and now measurable: Stage 0's integrand is `a0**2` times CGS constants of order
  1e-24 (`SIGMA_T_CGS`) and 1e26 (`photon_density_scale`), which nearly cancel. Worth
  checking the intermediate ordering before writing the cupy kernel.
- `delta.angle_integrated_spectrum` is a double Python loop over the direction grid —
  fine at the sizes the harness uses (~20 s at 321x321), wrong for anything larger. It
  chunks naturally over directions if it ever needs to.

---

## 2026-08-08 — Review round on Phase 2.5, and §9.1 written into the plan

Plan bumped to **v0.13** (§9.1 rewritten — the ~2pi is traced, not open).
`DECISIONS.md` **D026–D027**.

A review of `42e9609..HEAD` returned nine findings. All nine reproduced, all nine fixed.
`pytest` -> **262 passed**.

**The two that were real defects rather than tidy-ups:**

- **The window metric had gone blind again, on the other side.** D023 raised the flux floor
  from `1e-6` of the *mean window* to `1e-3` of the *total*. Moving the denominator was the
  fix and was argued for; multiplying the threshold by 16000 rode along unexamined. A
  candidate dropping a real feature worth 0.07% of the yield scored **exactly zero** on
  both reported numbers. Now `1e-4`, with the same number serving as significance test and
  denominator guard; the cliff moved from 0.1% to 0.01% and a test pins it.
- **Stage 0 crashed on an empty bunch** — `np.concatenate([])` raises, so the prefilter
  turned a zero yield into an exception for any configuration where it discards
  everything. Directly contradicts the §3.2 pure-optimization claim the suite asserts
  elsewhere.

**Two quadrature errors in delta's arbitration, both in the number that gates §9.1:**
`_captured_fraction` integrated the Lorentz factor without the polarization factor delta's
own integrand carries (0.941 claimed vs 0.917 true at four cone widths), and
`check_normalization` trapezoid-integrated bin-centre densities, dropping half the end
bins. The anchor now reads **1.000002** where it read 0.9937 for a quantity documented as
exactly one.

With both fixed the residue is `+0.41%` and finally has an explanation: the **square grid's
corners** reach `sqrt(2)` beyond the disc the correction assumes. Confirmed by construction
— a monoenergetic zero-divergence beam reproduces it to 0.01 percentage points, so beam
spread is not involved. The identity gate moved to eight cone widths, where that residue is
a fifth of its budget rather than three quarters.

**And one process failure worth naming.** `.claude/settings.json` — written by the graphify
tooling, hardcoding an absolute binary path as a hook on essentially every tool call — was
committed by an unreviewed `git add -A`, together with an auto-appended `AGENTS.md` section
claiming the repo "has a knowledge graph at graphify-out/" one commit after that directory
was gitignored. Any other clone would have fired a nonexistent binary on every tool call.
Moved to `.claude/settings.local.json` (gitignored) and the `AGENTS.md` claim corrected.
`git add -A` after running a tool that writes config is the habit to break.

`AGENTS.md`'s status paragraph was also stale ("don't assume anything beyond Phase 0
exists") and now names what actually exists as of Phase 2.5.

---

## 2026-08-08 — Phase 3a: xigma engineering (Stage 1/2, `Collision`, `XigmaEngine`)

C4 re-checked first (`docs/GRAND_PLAN.md` §11's pre-3a note): all eight ComptonSuite
`worktree-*` branches (4 local, 4 remote) still show 0 commits ahead of `master`. Clean;
proceeded.

**Stage 1** (`engines/xigma/stages.py::deposit_table`, `Table`): 4D histogram over
`(gamma, theta_x, theta_y, ahat)`, nearest and CIC, both weight-conserving by
construction (`tests/test_stage1_stage2.py`). Bins directly onto the physical `ahat`
(`TrajectorySamples.retargeted_ahat`) rather than a separate shape axis rebinned later —
the predecessor's `retarget_a0`/`a0_kind` apparatus is not ported; a benchmark
(`test_deposition_is_cheap_next_to_stage_0`) confirms a fresh deposit from cached Stage 0
samples is cheaper than Stage 0 itself, so there was nothing left for a rebin to save
(`DECISIONS.md` D028).

**Stage 2** (`spectrum_from_table`, `angular_spectrum_from_table`,
`spectrum_in_angular_range`): the numpy production kernel ports the predecessor's
brute-force grid quadrature (its validation-only `reference.py`), not its ~550-line GPU
importance sampler — the predecessor's own audit already rated that sampler trust-level
C. `cupy`/`numba` are gated exactly like Stage 0's `_check_backend` until real kernels
exist. The pending §9.1 constant (`KERNEL_NORMALIZATION_CONSTANT = 1.5`) is isolated to
one module-level location and is pi-free, matching the predecessor's own kernel math
exactly (D029).

**A discretization artefact found while building the Stage2-vs-delta identity check, not
by review:** evaluating `spectrum_from_table` exactly at a beam's own angular centre
against a `scheme="nearest"` table aliases against that table's own cell boundaries —
measured ratios from 0.48 to 1.67 against `delta` across theta-bin counts 10-150 at fixed
particle count. `scheme="cic"` holds within a few percent across the same range. Both the
new identity leg and the regression test use CIC for exactly this reason; `nearest` stays
the schema default.

**`Collision`** (`collision.py`): owns one fixed `(InteractionParameters, Parameters)`
pair and memoizes `TrajectorySamples`/`Table` on itself — not a hash-keyed cross-call
staleness detector. That mechanism belongs to Phase 6's GUI grey-out model, which needs a
live object surviving repeated edits; nothing in this phase has that, so building the
detector now would be exactly the speculative machinery P6 rejects (D030). `run()` fills
`TOTAL_YIELD`, `SPECTRUM`, `ANGULAR_DISTRIBUTION`, `SPECTRAL_ANGULAR_DISTRIBUTION`,
`COLLIMATED_SPECTRUM`; `TEMPORAL_ENVELOPE`/`SPATIAL_DISTRIBUTION`/`MACROPARTICLE_DUMP`
are omitted (no per-step diagnostics in `TrajectorySamples`, no photon-macroparticle
population to dump) — `run()` fills what it can and skips the rest, per the `Engine`
contract, not a silent gap.

**`XigmaEngine`** (`engine.py`): conforms to the `Engine` protocol
(`test_xigma_engine_conforms_to_the_engine_protocol`). `recompute_costs` declares only
`n_e` (bunch charge) — handled at the `io` level without an engine run at all — because
every other cheap path §5 illustrates needs a caller that keeps one `Collision` alive
across edits, which does not exist yet (D030). **Not passed to `validation.run.main()`'s
default `run_suite()` call:** the scenario bank's default output resolution
(`COLLIMATED_SPECTRUM` at 64x16x16) measured 36.6s for one slice at 100k particles against
this phase's numpy kernel — real Calculate cost (§12), not something a routine suite run
should pay repeatedly. `run.py::identity_checks` gained a fourth leg instead — Stage 2's
table kernel against `delta` at one point, both carrying the identical pending §9.1
factor, so the ratio is a genuine ~1 identity today rather than another open question
(D031).

**Verification:** 283 tests pass (up from 262 at the end of Phase 2.5 — 21 new, split
across `tests/test_stage1_stage2.py` and `tests/test_xigma_engine.py`); doc-staleness
guard green against four new `DECISIONS.md` entries (D028-D031); `python -m
gammaforge.validation.run` still runs in ~3s and reports `ALL CHECKS PASS`, now with a
fourth identity leg per scenario.

`docs/GRAND_PLAN.md` bumped to v0.14 (§4.2 rewritten to match what was actually built,
§5's illustrative table annotated with what is and is not wired yet, §11's Phase 3a row
marked landed). `DECISIONS.md` gained D028-D031.

**Carried forward, not started:** Phase 3b (the §9.1 authoring choice, §9.2/§9.3
derivations); cupy/numba Stage 1/2 kernels; the `ENGINES` registry (still deliberately
deferred, D018 — one engine is not enough reason to build it); wiring `XigmaEngine` into
the default validation suite run (needs either real kernels or a suite-appropriate output
resolution, D031).

---

## 2026-08-08 — Stage 1 restructured: shape deposit + non-uniform ahat retarget (D032)

Same-day follow-on to the Phase 3a session above, prompted by the project author (a
physicist) questioning why the direct-onto-`ahat` deposit didn't reuse the predecessor's
`retarget_a0` regrid mechanism. Working through it surfaced that D028's dismissal was
right about the mechanism (the predecessor's fixed-target-range regrid, built for
cross-run comparability) but missed the actual physics reason to want a fixed, *non-uniform*
target grid: the redshift correction `ahat` drives only matters near a pulse's peak, and a
grid that just follows wherever the sampled data spans resolves the peak no better than the
bulk.

**Landed**, planned in full via `/plan` before any code changed (`docs/GRAND_PLAN.md` v0.15,
`DECISIONS.md` D032, superseding D028):

- `stages.ShapeTable` + `deposit_shape_table`: Stage 1 now bins onto `a0_shape` (already
  peak-independent) on a fine, uniform grid — one deposit per `TrajectorySamples`, reusable
  for any peak a0.
- `stages.retarget_ahat`: a conservative, overlap-weighted regrid (adapted from the
  predecessor's `retarget_a0`, `deposition.py:423-511` in the old repo) onto a fixed,
  non-uniform `ahat` axis — `stages._ahat_target_edges`'s "log-spaced in distance from the
  top" formula, bin width shrinking monotonically toward `ahat_max`. Two exact rescales:
  the source edges by `a0_peak**2`, the deposited mass by
  `(a0_peak/source_a0_peak)**2` (the same relation `retargeted_luminosity` established).
  Truncates trailing target bins the source never reaches (exactly zero mass, not
  approximately) — measured a real speedup, most configurations populate only a handful of
  the 32 target bins.
- **Grid defaults tuned against the actual scenario bank, not assumed.** The predecessor-
  matched starting point (`ahat_max=0.5`, `decades=3`) put bin 0 alone wider than
  `near_a0_max`'s entire `ahat` range — every scenario would have collapsed into the floor
  bin, defeating the whole change. Swept `decades` numerically against the bank's measured
  `ahat` distributions before landing on `decades=1.0`, which resolves `near_a0_max` across
  real bins while correctly leaving `low_a0` in the floor (its `ahat` genuinely is small
  enough that the redshift barely matters there).
- `Table.bin_volume` (a single scalar, only valid for a uniform grid) removed; replaced by
  `ahat_widths` (per-bin array) + `gamma_theta_cell_area`, folded into
  `spectrum_from_table`'s cell sum directly.
- `Collision` gained a `_shape_table` singleton cache alongside `_tables`, mirroring
  `build_overlap`'s pattern — `retarget_ahat`'s cost is independent of `n_particles`, so
  multiple peak-a0 queries on one `Collision` now pay the expensive shape deposit once, not
  per query (a genuine improvement D028's "nothing needs this yet" correctly declined to
  build machinery for, since nothing called it repeatedly at the time).
- `schema.py` gained `n_bins_a0_shape`, `ahat_min`, `ahat_max`, `ahat_decades`; `n_bins_ahat`
  kept its key but its default moved 12 -> 32 and its meaning shifted (deposit count ->
  retarget-grid count).

**Verification:** 291 tests pass (up from 283 — one file (`test_stage1_stage2.py`)
substantially rewritten around the two-stage API plus new retarget/truncation/floor-fold
tests, `test_xigma_engine.py` updated for the two-level cache); doc-staleness guard green
against D032 and two now-historical backtick fixes in D028/D031; `python -m
gammaforge.validation.run` still green in ~3s, the fourth identity leg (Stage 2 vs `delta`,
D029) re-verified at the identity harness's 2000-particle scale with the new pipeline
(ratios 0.98-1.00 across the three scenarios).

`docs/GRAND_PLAN.md` bumped to v0.15. `DECISIONS.md` gained D032 and a "superseded by D032"
note prepended to D028 (left as historical record, not rewritten).

---

## 2026-08-08 — Phase 3b: §9.1 closed; §9.3 marked

**§9.1 is done.** The `1/(2π)` D026 derived is now applied, at the two places this repo
transcribes the paper's differential cross-section: `KERNEL_NORMALIZATION_CONSTANT` is
`1.5/(2π)` and delta's new `DIFFERENTIAL_PREFACTOR` is `3/(2π)`. The identity harness's
delta leg is gated at **1** — `NormalizationCheck.expected_ratio` dropped its `2π` and now
reports against the captured fraction alone — which is §11's stated exit criterion for this
phase. It reads 0.9813 against a capture of 0.9773 (+0.41%) on all three scenarios.

**The measurement that licensed the change, and the one that could not.** Every Stage-2
check that existed — including the fourth identity leg — is a *ratio* between two paths
carrying the same normalization constant. All of them were green with the factor present
and are green with it removed; they cannot see normalization at all. So the discriminating
quantity had to be built: angle-integrate the table kernel over a cone and compare with
Stage 0's elementary `flux × cross-section × time` photon count, the same arbitration delta
already gets. Measured **before** touching the constant, on a 49×49 angular grid with 640
`s` bins: `2π × 0.9966` — the factor and nothing but the factor, on a path independent of
delta's. Order matters here and it was the right one (D026 derived → constant set from the
derivation → measurement confirmed), which is the difference between P14-compliant and
fitting a constant until a test passes.

That check now lives in `tests/test_stage1_stage2.py::test_the_table_kernel_angle_integrates_to_stage_0_total`,
at a resolution sized for a ~10s test (reads 1.078, ±15% tolerance; refining to 21 angles
gives 1.015, and 21 angles × 480 `s` bins gives 1.006 — the residue is midpoint-rule
convergence on two peaked integrands, and the distinction the test has to make is 1 vs 6.28).

**A second thing the closure fixed, found on review.** `Results` mixes two spectral paths:
`SPECTRUM` comes from Stage 0's closed form (never touches the table), everything angular
comes from the table kernel. So before today xigma's own outputs were internally
inconsistent by 2π *within one `Results` object* — and no test could see it, because the
only engine test comparing magnitudes has both sides on the non-table path. Integrating
`SPECTRAL_ANGULAR_DISTRIBUTION` back over its angle axes now reproduces `SPECTRUM` to 0.87
(the rest is the auto-range's ~4.6/γ angular span and a 25-point trapezoid, both understood);
`test_the_two_normalization_paths_inside_one_results_object_agree` pins it.

**§9.3 got the marker §9.2 has had since Phase 1.** `io.laser.EMISSION_IS_HEAD_ON` +
a `validate()` warning on nonzero `theta_xz`/`theta_yz`. This was a real P14c gap and a
sneaky one: the crossing angle *is* half-implemented — `rotation_matrix` is applied wherever
the pulse is sampled, so overlap, timing and the sampled a0 all move correctly with a tilt,
while `RELATIVE_VELOCITY` stays 2 and the Stage-2 kernel keeps measuring angles from the
collinear axis. Tilt the beam, watch the yield change, conclude the physics followed. Worse
than a parameter that visibly does nothing, and it was the one with no warning.

**Verification:** 294 tests pass (+3: the absolute-normalization check, the two-path engine
check, and the crossing-angle warning); `python -m gammaforge.validation.run` green in ~3s
with the delta leg re-gated; doc-staleness green. The golden *scalars* the suite compares are
closed-form quantities (`a0_peak`, `n_electrons`, `n_photons` from the luminosity path), none
of which passes through a kernel prefactor, so they are untouched. Golden *slice* comparison
is absolute rather than shape-normalized and is not wired into the default run (D031) — see
D033 for why the predecessor's slices are expected to be self-normalized and why that should
be verified when Phase 5/7 turns the comparison on, not assumed now.

**Blocked, and staying blocked (both non-blocking for Phases 4–6 by design):**
- **§9.2 (ellipticity→a0)** — the paper has no formula; its polarization object is a
  normalized coherence matrix with no scalar ellipticity. Author's derivation.
- **§9.3 (crossing angle)** — genuinely absent from the paper, which warns against extending
  the near-backscattering angular derivation without revisiting the geometry. Author's
  derivation.
- **The manuscript itself.** The repo now computes corrected physics against an
  *uncorrected typeset* eq. `xsec` (annotated, not edited, per D026). Where the `1/(2π)`
  belongs there — prefactor, normalization of `R`, or definition of `Û` — is the author's
  call and changes no observable, which is why the code did not wait. Worth resolving in the
  paper so the two stop diverging.

`docs/GRAND_PLAN.md` bumped to v0.16 (§9.1 marked CLOSED, §9.3 marked, §11's 3b row and
§12's risk row restated). `DECISIONS.md` gained D033 (§9.1 closure, superseding D025's
"report, don't correct") and D034 (the crossing-angle marker).

---

## 2026-08-09 — Phase 4: `engines/analytical` lands (estimates, width breakdown, quadrature spectrum)

Built in an isolated worktree/branch, concurrently with Phase 3b (§11: "Phases 4–6 must
not wait on physics derivations") — verified beforehand that the parallel session's
uncommitted 3b edits don't touch §4.3 or the Phase 4 table row.

`engines/analytical/formulas.py`: `estimate_yield` (closed-form Gaussian-overlap yield,
ported from the predecessor's SI/`scipy.special.erfcx` version onto this repo's CGS
`GaussianElectronBeam`/`GaussianParaxialLaser`, with a hand-rolled `_erfcx` rather than a
new `scipy` dependency — D037), `SpectrumWidthBreakdown` + `estimate_spectrum_width`
(§4.3's four components — collimation, emittance, energy spread, nonlinearity — reported
separately rather than pre-summed, with `.total` as `math.hypot` of the four so the
quadrature-sum relationship is structural rather than something a caller must get right),
and `angle_integrated_spectrum` (quadrature over the beam's Gaussian energy distribution,
cost independent of `n_particles` by construction — the predecessor's own fix for a
real 76.3 GiB OOM is preserved by never touching a macroparticle array at all). This is a
**third, deliberately independent** copy of the linear-Compton kinematic shape already
implemented in `engines.xigma.stages.angle_integrated_spectrum` and
`validation.references.delta.single_electron_spectrum` — importing either would make
§7's analytical-vs-xigma-vs-delta cross-check circular.

`engines/analytical/engine.py`: `AnalyticalEngine`, filling `TOTAL_YIELD` and 1D
`SPECTRUM` only (§4.3 is explicit that analytical does not produce the 3D
`COLLIMATED_SPECTRUM` slice) via `io.target.auto_ranges`/`slice_axis_values`, reused
unmodified from xigma's own facade rather than reimplemented. `SPECTRUM` is defined as
`total_yield * (angle_integrated_spectrum's normalized shape)`, not two independently
estimated quantities — D036 has the full reasoning — which makes
`PhasespaceSlice.integrate() == total_yield` an **exact** identity (§7), verified to
~1e-16 relative on the baseline scenario, not a tolerance. `theta_col` for the width
breakdown is the geometric mean of `Target.theta_x_col`/`theta_y_col` (D038) rather than
a duplicate schema field — the engine's own `Parameters` schema (`schema.py`) carries only
`n_quad`, the one genuine knob nothing else already owns.

**Verification.** Spot-checked `estimate_yield`/`estimate_spectrum_width` against the
predecessor's own worked-example numbers, computed by actually running
`ComptonSuite/src/gammaforge/models/analytical.py` in a throwaway venv (not committed):
width matches to ~1e-11 relative (float precision), yield matches to ~1.9e-6 (explained —
yield is the only quantity using `SIGMA_T_CGS`, and the two repos are separately-installed
`pint` environments with slightly different CODATA constant tables; not a unit-conversion
defect). Both spot-checks are pinned as regression tests
(`tests/test_analytical.py::test_estimate_yield_reproduces_the_predecessors_worked_example`
and the width equivalent) so they don't need the predecessor repo present to run again.
Added a self-contained Thomson-limit closed-form anchor test (§7) that needs no
predecessor comparison at all. Full suite: 313 passed (was 294 before this session).

**What's still open** (§4.3's own "growth items," not attempted this pass, per D035):
foci displacement, non-round-beam total yield, collimated-spectrum construction. `GRAND_PLAN.md`'s
Phase 4 row is marked landed-with-exceptions, not closed, naming these explicitly.
`engines/__init__.py`'s `ENGINES` registry is still not created — analytical follows
xigma's own precedent (D018) of landing without one.

`DECISIONS.md` gained D035–D038 (round-beam approximations kept / growth items open,
the `SPECTRUM`-is-defined-not-reconciled rescale, hand-rolled `erfcx`, geometric-mean
`theta_col`). `docs/GRAND_PLAN.md` bumped to v0.19 (§11's Phase 4 row updated).

---

## 2026-08-09 — Phase 4 follow-up: the general overlap integral closes two growth items

Same worktree/branch as the Phase 4 landing above. Prompted by the observation that the
overlap integral can be done analytically except for one quadrature — which is exactly how
it came out.

**Derived and implemented** (`docs/DERIVATIONS.md` §A, new file): the Gaussian luminosity
overlap integral in general form. Both transverse integrals collapse to a single
determinant `det(C_e + C_l)` (which carries unequal x/y sizes, unequal focusing, astigmatic
laser waists and the `psi_focus` rotation between the two ellipses), and the time integral
collapses to a longitudinal Gaussian of width `D / (1 + beta_0)`. What is left is one
quadrature over `z` with a strictly positive, smooth integrand — milliseconds, no Monte
Carlo. Shipped as `engines.analytical.formulas.overlap_yield`; `AnalyticalEngine` now uses
it instead of the round-beam closed form.

**Two of D035's three growth items are therefore closed *for the total yield*** —
non-round-beam yield and foci displacement. Neither needed new schema state: the electron
waist offset is already `GaussianElectronBeam.alpha_x`/`alpha_y` and the laser side is
already `z_fx`/`z_fy`/`psi_focus` (P9). **Not** closed for `estimate_spectrum_width`: its
nonlinearity term still uses the pulse's peak a0 rather than the a0 the bunch samples,
which needs an overlap-weighted `<a0^2>` — so moving the foci now changes the yield
correctly while leaving that width component unmoved. Collimated-spectrum construction
stays open. Crossing angle stays deferred to §9.3, and `overlap_yield` **raises** on one
rather than approximating — which means `AnalyticalEngine` now raises where it previously
returned a wrong number. No current caller passes a crossing-angle or flying-focus laser
(`validation.scenarios` leaves both zero), but Phase 6 must decide what the GUI estimates
panel does with one instead of propagating the exception.

**How it was verified** (the derivation is only as good as its checks):
- reduces to the round-beam closed form to 5e-10 at `n_quad=32001`, against an
  independently-coded evaluation — the derivation's own strongest test, since the two paths
  share no code. At the schema default (`n_quad_overlap=2001`) the quadrature error is
  ~1e-7 round / ~1e-6 on a displaced astigmatic case, not the machine-precision figure;
- `_electron_sigma2` agrees with `io.bunch._drift_fit` to machine zero across an
  alpha × drift grid, which pins the `alpha` **sign** that no symmetric scenario can catch;
- the yield peaks with the electron waist at +0.0995 cm against a laser focus at +0.1000 cm,
  independently confirming the relative sign of `alpha` and `z_fx`;
- convergence checked on a *displaced, non-round, astigmatic, rotated* scenario, not on the
  aligned baseline where it would pass trivially.

**Found, flagged, not fixed — worth the next session's attention.** The derivation pins the
laser hourglass coefficient as `sigma_l / z_R` exactly. This repo's
`GaussianParaxialLaser.rayleigh_x` *and the predecessor's own pulse class* both define
`z_R = 4 pi sigma^2 / lambda`, but the predecessor's `analytical.py` uses a divergence 4x
larger — so the predecessor is inconsistent with its own laser model, and the port carried
that faithfully. On the baseline scenario, where the hourglass is almost entirely
laser-driven, **it is worth a factor of 3.285 in the yield.** `overlap_yield` is free of it;
`estimate_yield` keeps it, now with a docstring warning and a test pinning the 3.285 so it
cannot drift silently (D040). Nothing in the suite before this was sensitive to that term —
the predecessor pin tests port fidelity, and the Thomson-limit anchor removes the hourglass
term entirely by construction. **This is a judgement call for the author:** if the correct
convention is confirmed, the predecessor's published yields are low by that factor.

Full suite: 324 passed, up from 315 (30 in `tests/test_analytical.py`). `DECISIONS.md` gained D039–D040;
`docs/GRAND_PLAN.md` bumped to v0.20 (§4.3 and §11's Phase 4 row).

**Merge note:** `docs/DERIVATIONS.md` is a new file here *and* an untracked file in the main
checkout (the parallel Phase 3b session's §9.2/§9.3 material). The two hold independent
sections — mine is lettered §A precisely so they concatenate; resolve the add/add conflict
by keeping both.

---

## 2026-08-09 — Phase 4: the crossing angle, and a brute-force overlap check

Same worktree/branch. The previous session deferred the crossing angle to §9.3; that was
wrong, and this session says so. §9.3's open item is the polarization structure of the
**emission kernel**. The **overlap geometry** is a separate, entirely solvable Gaussian
problem — and `docs/DERIVATIONS.md`'s own §9.3 notes (from the parallel 3b session) already
record that the relative-velocity factor and the resonance frequency are general in the
paper. The yield was never blocked.

**Implemented** (`docs/DERIVATIONS.md` §A.6): the overlap integral rewritten as a quadratic
form. Eliminating time replaces `M` by `M - g g^T / h`, and the transverse integrals leave
a 2x2 determinant and a Schur complement. That single expression covers **every** geometry,
with head-on falling out as an identity — the strongest evidence being that no head-on test
changed when this landed, including the exact reduction to the round-beam closed form.

**One approximation, stated and measured.** With a crossing angle the two hourglasses vary
along different directions (`z` for the bunch, `u = k_hat . r` for the pulse), so an exact
reduction leaves a **2D** quadrature — "everything analytic but one integral" is a head-on
statement. The spot sizes are sampled at `u = (k_hat . zhat) z`; the exponent stays exact,
so the whole crossing-angle suppression is exact. The bound is `delta / z_R`, not
`delta / sigma_z`, and it is *not* always small — a 0.4 rad crossing with a 2 um waist and a
200 um bunch puts it at 1.6. Measured rather than argued: the yield still moves by < 1.3e-4
there, because the dropped term enters an even, slowly varying prefactor.

**Validated three ways, in increasing generality:**
- a constant-width closed form (3x3 determinant, no quadrature) — isolates the crossing
  geometry: agreement ~1e-14 out to 0.4 rad, in both crossing planes and combined;
- the Piwinski suppression `1/sqrt(1 + (sigma_s tan(theta)/sigma_perp)^2)` — confirms it is
  the known physics, to 1e-6 at 2 mrad (that formula is itself small-angle);
- **a brute-force Monte Carlo** over `GaussianParaxialLaser.photon_density` with real
  sampled macroparticles, sharing no algebra with `formulas.py` — a few 1e-4, converged
  separately in particle count and in time grid so a passing result cannot be a t-grid
  artifact. This is now a committed test, parametrized over head-on, a crossing angle, and
  a crossing angle combined with non-round/astigmatic/displaced/rotated foci.

**The effect is large.** At the baseline scenario the yield falls by 1.07x at 5 mrad,
**2.18x at 20 mrad**, 5.77x at 50 mrad. Any scenario quoting a crossing angle needs this.

**Scope, explicitly.** The crossing angle is covered for the **total yield**. `SPECTRUM`'s
*shape* is still head-on while its integral is now correct — the engine normalizes the
slice to the yield, so that combination looks more right than it is, and
`AnalyticalEngine` now reports it on `Results.model_specific["warnings"]` rather than
leaving it to `io.laser.validate()`. This also resolves the inconsistency the previous
session introduced, where analytical raised on a crossing angle while xigma warned and
proceeded.

**Grid resolution was checked, not assumed.** Crossing narrows the longitudinal support
while `_overlap_grid`'s outer span still comes from the head-on scale, so `span/core`
reaches ~60 at 50 mrad. The refined `1/sqrt(max S)` and `sigma_i/sin(theta)` windows absorb
it: worst spacing inside the core stays under 0.008 of the core width, and the **schema
default** `n_quad_overlap=2001` matches n=20001 to 1.2e-7 head-on and 2.3e-6 at the 0.4 rad
adversarial fixture. Pinned by `test_schema_default_n_quad_resolves_a_crossed_collision`,
since every other crossing test uses a generous n_quad and would not have noticed.

Full suite: 342 passed, up from 324. `DECISIONS.md` gained D041 (superseding D039's
crossing-angle refusal; the `beta_ff` refusal stands); `docs/GRAND_PLAN.md` bumped to v0.21.

---

## 2026-08-09 — Phase 4: cost tiers, `<a0^2>`, and resolved previews

Same worktree/branch. Four things, all reusing the §A.6 quadratic form rather than adding
machinery.

**The exact 2D quadrature is now available** (`n_quad_u > 1`, `docs/DERIVATIONS.md` §A.7).
With a crossing angle the two hourglasses vary along different directions, so the 1D form
samples the spot sizes along `z` only; rotating the transverse plane so `q1` lies along the
crossing direction makes `u` depend on exactly one transverse coordinate, leaving `q2`
analytic and `(z, q1)` quadratured. Nothing approximated. This makes the 1D path's error a
**measured** number: 1.9e-4 at 20 mrad, 1.6e-3 at 0.4 rad, on the worst corner this model
has (2 um waist against a 200 um bunch). Counterintuitive and worth recording — the 2D mode
converges *more slowly* than the approximation it checks at **small** angles, because the
widths barely vary along `q1` there; it earns its cost at large angles.

**Three cost tiers are now declared** (D043): closed form ~0.01 ms / 1D ~1-2 ms / 2D
~40-800 ms, with `recompute_costs` covering the quadrature knobs. §4.3 calls analytical the
only real-time engine; that claim now stays true of a named tier rather than of whatever
the default happens to be, and Phase 6 has the cost data before it wires a live panel.

**The last width growth item closes** (D042). `<a0^2>` weighted by the luminosity is the
same integral with the laser density squared — `a0^2` is exactly proportional to the
normalized photon density, so it is one parameter, not a new derivation. It is a large
correction: the bunch samples ~0.35 of the peak `a0^2` at the baseline, so the nonlinearity
term was overstated ~3x. There is a clean exact limit that makes it checkable — for a
transversally pointlike bunch with no hourglass the ratio is exactly `1/sqrt(2)`,
*independent of bunch length*, because integrating over both `z` and `t` spans every
relative shift and the bunch convolution factors out. A counter-propagating collision can
never reach peak a0: it always scans the pulse's full longitudinal profile.

**Resolved previews** (§A.9): `overlap_time_profile` (dN/dt) and
`overlap_transverse_profile` (dN/dx dy in the bunch frame), ~8 ms for 400 time points and
~100 ms for a 64x64 image, for drawing the collision before an expensive run starts. Both
satisfy exact integrate-back-to-the-yield identities (~1e-7), asserted not assumed — the
same §7 pattern as `∫SPECTRUM = TOTAL_YIELD`. Angle-resolved is deferred by request; it
needs the emission kernel, not the overlap geometry.

**Two bugs the checks caught, neither visible by inspection:**
1. the 2D path's `q1` span used the bare curvature `m11` instead of the Schur-reduced
   `m11 - m12^2/m22`, silently truncating the integral exactly when the two transverse
   directions are correlated — an 11% error at 20 mrad before the fix;
2. profile evaluation was unchunked, so a 128x128 image allocated `(3,3,128,128,n_quad)`
   and took 3.7 s. Everything that meshes `z` against something else now accumulates in
   blocks (`_Z_BLOCK`), and the profile defaults dropped to preview-grade `n_quad=401`.

Also: `docs/DERIVATIONS.md` §A **rewritten in MathJax** so it pastes into
`~/Work/Papers/2026/Compton-Numerics/xigma.tex` with only environment changes.

Full suite: 355 passed, up from 342. `DECISIONS.md` gained D042-D043; `docs/GRAND_PLAN.md`
bumped to v0.22.

---

## 2026-08-09 — Phase 4: flying focus

Same worktree/branch. Asked to explore whether the flying focus can be derived the same way,
with the expectation that crossing angle plus arbitrary flying-focus velocity would need
more than a 2D quadrature. **It does not** — that is the main result.

**The counting.** A flying focus makes the spot-size coordinate `u_spot = u + beta_ff * ct`
time-dependent, so the widths depend on two independent linear functionals of
`(x, y, z, ct)`: `z` and `u_spot`. Fixing both leaves a 2D affine subspace on which the
integrand is an ordinary Gaussian. So two dimensions stay analytic and two are quadratured,
*with or without* a crossing angle — the two effects each add one width argument and having
both does not add a third. `docs/DERIVATIONS.md` §B.2.

**Implemented** (§B.3) as a `(z, ct)` quadrature with `(x, y)` analytic; with a crossing
angle the residual transverse part of `u_spot` is sampled at the axis, the same
approximation §A.7 already measures at 1.9e-4. `overlap_yield` no longer refuses `beta_ff`;
`overlap_mean_a0_sq` inherits the path; `overlap_time_profile` handles it (time is an
explicit axis there) and `overlap_transverse_profile` raises, because it integrates time
out.

**A 1D shortcut exists and is deliberately not shipped** (§B.4). Freezing the widths at the
stationary point of the time integral reuses §A's machinery unchanged and is accurate to
1e-5 on a short bunch — but its error is *first* order, unlike the crossing-angle
approximation whose dropped term cancelled by evenness. Measured: **34% at `beta_ff = 1`**
on the baseline. Controlling parameter `beta_ff * sigma_ez / z_R`, which is 2.5*beta_ff for
the baseline and 0.025*beta_ff for a 30 um bunch. Excellent in the regime a flying focus is
*for*, useless just outside it — too sharp a knife for a default (D044).

**Two physics results fall out, neither encoded, both pinned as tests:**
- `beta_ff = 1` maximizes the yield — the focal plane co-moves with the bunch at `c`, so the
  electrons sit at the waist throughout. Worth 2.0x on the baseline and **2.8x on a 30 um
  bunch** over no flying focus, and it beats both slower and faster slides.
- For a short bunch the yield is **invariant under `beta_ff -> 1/beta_ff`** (0.5 and 2.0
  agree to seven digits). Along the ridge `z ~ ct` so `u_spot ~ (beta_ff - 1) ct` while the
  Rayleigh range carries `(1 + beta_ff)`; the spot depends on
  `(beta_ff - 1)/(beta_ff + 1)`, odd under that map, and the width on its square.
  `beta_ff = 1` is the fixed point, which is why it is the optimum.

**One bug, found by the numbers not by inspection.** The `(z, ct)` Gaussian is nearly
degenerate — the collision lives on a thin diagonal ridge — so a grid sized from the
marginals under-resolved it and came out **3.8% low** on a short bunch. Nodes now go on the
principal axes, with counts raised until each step advances `u_spot` by less than `z_R/8`.

**Not cross-checked against the author's own derivation.** The head-on synchronized
counter-propagating expressions were not available here, so §B rests on the Monte Carlo
alone. The `(1 + beta_ff)` Rayleigh stretch in `rayleigh_x` is inherited from the code
rather than re-derived, and the reciprocal symmetry depends on it directly — that is the
first thing an independent check should target.

Full suite: 364 passed, up from 355. `DECISIONS.md` gained D044; `docs/GRAND_PLAN.md`
bumped to v0.23.

---

## 2026-08-09 — Phase 4: luminosity-weight particle filter (measured, then landed)

Same worktree/branch. Idea: rank macroparticles by their actual contribution instead of
bounding them with a geometric cone.

**Closed form, so it is cheap.** Each particle's expected luminosity contribution is
`w_i = Int dt n_L(r_i + v_i t, t) (c - v_i . k)`, and freezing the spot sizes makes that
integrand Gaussian in `t` — one vectorized pass, no time stepping, `O(n_particles)`. The
frozen widths are taken at each particle's own closest approach (stationary point, iterated
twice). `io.bunch.luminosity_weights`.

**It is a different contract from `prefilter_bunch`, and that is the load-bearing point.**
The cone drops only particles proven to contribute exactly zero, so results are
bit-identical with it on or off — a tested invariance. A relevance ranking drops particles
that contribute a little. Both now exist side by side; the cone is untouched and remains
the default (D045).

**Measured, and the answer is "it depends", quantitatively:**

| scenario | cone keeps | weight keeps (<=1.3e-4 induced error) |
|---|---|---|
| well-matched baseline | 100% | no headroom — nothing to discard |
| 400 um bunch vs 4 um / 1 ps pulse | 36% | **4.7%** |
| same, with displaced foci | 94% | **31%** |

The third row is the real case: a displaced focus forces the cone to widen conservatively
to 94%, while ranking by relevance keeps 31% at 1.3e-4 error (42% at 7.5e-6). So this is
worth reaching for exactly when the geometry is mismatched, and worth nothing when it is
not — which is the honest recommendation rather than "more precise than the cone".

Full suite: 369 passed, up from 364. `DECISIONS.md` gained D045.

---

## 2026-08-09 — Phase 4: transverse and timing misalignment

Same worktree/branch. Answering "do the derivations allow arbitrary foci displacement in
all 3 dimensions": longitudinal yes (independently per axis — `z_fx`/`z_fy` for the pulse,
`alpha_x`/`alpha_y` for the bunch), **transverse and temporal no** — and the blocker was the
data model, not the derivation. Both distributions were centred on the origin by
construction. Now fixed.

**Added** `x_off`, `y_off`, `t_off` to `GaussianParaxialLaser`, applied once in
`_local_coordinates`, so every consumer of the field inherits them — `photon_density`,
`a0_profile`, `field`, and therefore xigma as well as analytical. `active_region` shifts its
origin to match, in the same change rather than after: it is a bound the cone prefilter
relies on never being too small, and leaving it at the origin while the pulse moved would
discard particles that do interact.

**No `z_off`, deliberately.** For a pulse travelling at `c` a longitudinal spatial offset is
indistinguishable from a timing offset, so `(x_off, y_off, t_off)` is the complete
*independent* set (D046).

**Analytically it is one linear term.** Writing the laser's exponent as a quadratic form in
`(v - D)` with `D = (x_off, y_off, 0, c t_off)` gives `E = v'Mv/2 - v'L + const` with
`L = M_l D`. No new structure — but it has to be carried through *both* completions of the
square, and a dropped piece **shifts** the answer instead of making it diverge, so it would
look plausible. Two guards: `D = 0` must be bit-identical to the previous result, and in the
no-hourglass limit the falloff is exactly `exp(-d^T (C_e + C_l)^-1 d / 2)`, checked to 1e-13
with an off-diagonal displacement (an on-axis test passes with a wrong inverse).

**Verified** against the brute-force Monte Carlo to ~1e-3 for transverse-only, timing-only,
and every combination with a crossing angle and with a flying focus.

**A coupling worth knowing:** a timing offset and a crossing angle are *not* separable. A
timing slip means the beams meet away from the nominal point, and with a crossing angle that
displaces the collision transversely too — so the same 10 ps slip costs 3.7% head-on and
4.9% at 50 mrad. Asserted as a test, since an implementation treating them as two
independent reductions would miss it.

Full suite: 382 passed, up from 369. `DECISIONS.md` gained D046; `docs/GRAND_PLAN.md`
bumped to v0.24.

---

## 2026-08-09 — Phase 4: z_off reconsidered, and the illumination prefilter

Same worktree/branch. Two follow-ups from review.

**The `z_off` justification was wrong even though the conclusion held.** I had written that
a longitudinal offset "is degenerate with `t_off` for a pulse travelling at c". That is not
the reason. The focal plane is fixed in space while the envelope sweeps through it at `c`,
so **focus position and arrival time are independent** — two pulses whose foci coincide
exactly still miss if they arrive at different times, and that configuration is real and
representable (`z_fx = z_fy = 0`, `t_off != 0`). Measured with coincident foci: the yield
falls to 0.99 at 5 ps, 0.86 at 20 ps, 0.41 at 50 ps and 0.01 at 200 ps, because the beams
meet a distance `c t_off / 2` from the focus.

What actually makes `z_off` redundant is that a *rigid* shift moves the focus **and** the
envelope, so it is already `(z_fx += D, z_fy += D, t_off += D/c)`. Three longitudinal
degrees of freedom, not four. Corrected in `laser.py`, `fields.py`, §A.11 and D046, and
pinned by two tests — one that coincident foci can still miss in time, one that focus and
timing shifts are distinguishable.

**Built the illumination prefilter** (D047), which is the idea as actually described: use
the production profile to define the region, then test whether each particle ever enters it.
`peak_illumination` returns the highest photon density a particle ever meets as a fraction
of the pulse peak — closed-form, since frozen widths make the density quadratic in `t` along
a straight trajectory, so its maximum sits at the stationary point. Same machinery as
`luminosity_weights`, read at its peak instead of integrated.

**Why it beats the cone, precisely:** the bright region is not the geometric one. Away from
focus the spot grows but dims as `1/(s1 s2)`, and `active_region` deliberately ignores that
decay. Measured: on a wide bunch meeting a small displaced pulse the cone keeps 94% of the
bunch, and the *median* particle it keeps is illuminated below `1e-6` of the peak.

| filter | keeps | induced error |
|---|---|---|
| cone, 1e-6 | 94.1% | 0 (exact) |
| illumination, 1e-6 | 28.4% | 3.5e-4 |
| weight, 1e-4 | 31.3% | 1.3e-4 |

On a well-matched collision all three keep 100% — correctly, since every electron passes
through the pulse. Both tolerance filters are kept: illumination is a region test (the
cone's own contract shape, drop-in, dwell-time independent), weight thresholds integrated
contribution (directly meaningful when the question is "how much luminosity am I
discarding"). Neither replaces `prefilter_bunch`, whose exact invariance is a stronger
guarantee than either.

One caveat found by measuring: the illumination threshold must be set far lower than
intuition suggests (1e-6, not 1e-3), because many weakly illuminated particles sum to a
non-negligible contribution though none matters alone.

**Default threshold set from measurement, not analogy.** `prefilter_by_illumination`
defaults to `1e-6`, *not* the `1e-3` that looks parallel to `prefilter_bunch`'s — the cone's
threshold is a bound and safe by construction, while `1e-3` here induces **44% error** on
the very scenario the filter exists for. That asymmetry is a genuine hazard for anyone
reasoning by analogy, so it is a `.. warning::` on the function and pinned by a test.

Full suite: 393 passed, up from 382. `DECISIONS.md` gained D047.

---

## 2026-08-09 — Phase 4: the illumination *window*, not just the filter

Same worktree/branch. The filter was only half of what the algebra gives.

**One quadratic, two results.** With frozen widths the photon density along a straight
trajectory is `exp(-(a t^2 + 2 b t + c)/2)` times a brightness factor, so `density >=
threshold` is one inequality in `t`:

    t in t_star +- sqrt(2 ln(peak / threshold) / a)

The filter asks whether that interval exists; the window *is* the interval.
`prefilter_by_illumination` is now defined as "window non-empty", mirroring how
`prefilter_bunch` is defined through `overlap_time_window` (D048).

**The window is the more useful half.** An engine samples each trajectory with a fixed
number of steps between `t0` and `t1`, so the window width sets the step size, and steps
spent where nothing happens are steps not spent resolving the interaction.
`overlap_time_window` brackets time inside the *geometric* active region — a conservative
bound, so far wider than the illuminated stretch. Same particles, same 33 steps, baseline
scenario: **7x more accurate** (2.0e-4 -> 2.8e-5), from a window only 1.5x narrower.

**How much margin to leave, quantified.** Width grows as `sqrt(ln(1/threshold))` while the
truncation floor tracks the threshold roughly decade for decade:

| threshold | median width | accuracy floor |
|---|---|---|
| 1e-3 | 101 ps | 9.2e-5 |
| 1e-6 | 151 ps | 1.9e-8 |
| 1e-9 | 187 ps | 8.9e-12 |
| 1e-12 | 218 ps | 4.1e-15 |

Nine decades of threshold cost about a doubling of the window. Being generous is cheap.

**Two things that are easy to get backwards, both now pinned.** The window errs *wide*: away
from closest approach the true spot is larger than the frozen value, so the real intensity
falls faster than the model and the edges come out ~100x below threshold. For a window that
is the right way to be wrong — a too-narrow one truncates the interaction and no step budget
recovers it. And it is an *estimate*, not a bound, so `prefilter_bunch`'s exact invariance
still rests on the geometric window.

Full suite: 399 passed, up from 393. `DECISIONS.md` gained D048.

---

## 2026-08-09 — Phase 4: illumination window wired into xigma; a0^2 moments

Same worktree/branch.

**xigma can now sample trajectories over the illuminated window** (D050).
`integrate_trajectories` gains `window="active_region" | "illumination"`, defaulting to the
existing behavior. Stage 0 spends a fixed `n_steps` between `t0` and `t1`, so window width
sets step size and any step outside the illuminated stretch is wasted resolution.

Measured, and the honest picture is mixed rather than a clean win:

| steps | baseline: cone | baseline: illum | tight focus: cone | tight focus: illum |
|---|---|---|---|---|
| 20 | 4.7e-3 | 1.3e-3 | 2.1e-2 | 3.6e-2 |
| 50 | 3.3e-5 | 1.9e-6 | 4.7e-3 | 1.7e-3 |
| 200 | 5.3e-14 | 1.9e-8 | 2.1e-5 | 1.7e-3 |

So: **17x faster at 50 steps on the baseline**, but it has a truncation floor the geometric
window does not, and on a tight focus it is already *behind* by 20 steps — the spot varies
so much across the window that the frozen-width estimate mis-sizes it. That is why the
default is unchanged. Worth reaching for at coarse step budgets on well-behaved geometry,
measured per scenario rather than assumed.

**`a0^2` moments** (D049). The spread is as easy as the mean, which is the useful part of
the answer: `a0^2 = K p_L` exactly, so `a0^4 = K^2 p_L^2` and the second moment is the same
overlap integral with the laser density **cubed** — `laser_power = 3`, one more call.
Generally `<a0^(2n)>` needs `laser_power = n + 1`. Validated against the a0-weighted Monte
Carlo to 4e-4.

It is **not a small correction**: `std/mean` is 0.70 baseline, 1.22 tight focus, 0.64 with a
synchronized flying focus. The intensity an electron samples varies by of order its own mean
across the bunch — so the edge is smeared about as much as it is shifted.

**The Compton-edge shift is deliberately NOT wired.** The mean sets where the edge sits
(`1/(1 + ahat)`) and the spread smears it; both are now computable. What is not settled is
the coefficient — `docs/DERIVATIONS.md` §0 records a **BLOCKING** finding that the code's
`ahat` is twice the paper's, and a factor of two there is a factor of two in the shift.
`AGENTS.md` is explicit that a paper-code disagreement stops rather than being guessed, so
the moments are exposed and the edge is left alone. Wiring it is a small, well-defined change
once §0 resolves.

Full suite: 401 passed, up from 399. `DECISIONS.md` gained D049-D050.

---

## 2026-08-09 — Phase 4: nonlinear broadening as a bracket; illumination report

Same worktree/branch. Author correction on the physics, acted on.

**The formation length is the whole trajectory**, so `ahat_i` is one scalar per electron and
the trajectory may not be chopped into locally-constant pieces. The correct order is:
average along each trajectory, *then* take the spread across the beam.

**The mean survives that exactly** — luminosity weighting cancels `ahat_i`'s denominator, so
`<ahat>_L = Int n_e a0^4 / Int n_e a0^2`, which is what the overlap integral already
evaluates, splitting nothing. Now pinned against xigma's per-particle `ahat()`.

**The spread does not, and my earlier `std` was wrong.** Taking moments of the instantaneous
`a0^2` over all (particle, time) pairs folds in the *within-trajectory* variation, which is
already averaged away inside `ahat_i`. Tell: with every electron sharing one `ahat` the true
beam spread is zero and the joint formula returns a positive number. Measured against xigma
it over-stated by ~1.5x (0.70 against a true 0.39). `overlap_a0_sq_moments` is **removed**;
`overlap_mean_a0_sq` stands alone. (Its tests never actually landed — the insertion targeted
a name that did not exist and `str.replace` on a miss is silent — so nothing incorrect was
committed, but I had reported them as added.)

**No closed form for the right quantity.** `ahat_i` is a *ratio* of trajectory integrals, so
`<ahat^2>` needs `(Int a0^4)^2 / (Int a0^2)` per particle — a reciprocal of a Gaussian
integral inside a bunch integral. The shortcut that would have saved it (peak/sqrt(2), exact
for a Gaussian temporal profile) fails: median 0.92 at baseline but 0.33 at tight focus.

**So it is reported as a bracket** (D049), and per the author's constraint the
semi-analytical path takes **no** `O(n_particles)` step — a per-particle trajectory
quadrature was considered and rejected on exactly that ground.

**The proposed 0.4-0.9 bracket does not hold.** Measured across thirteen geometries via
xigma: **0.06 to 1.12**. Five fall below 0.4, one above 0.9. It tracks
`sigma_beam / sigma_laser` almost monotonically — 0.06 loose focus, 0.39 baseline, 0.86
tight focus, 1.12 for a bunch ten times wider than the spot. Shipped as
`NONLINEAR_BROADENING_RANGE = (0.06, 1.12)`, flagged as empirical rather than derived. The
width breakdown gains `nonlinearity_lo`/`_hi` and `total_range`; the legacy scalar sits at a
factor of 1, i.e. near the *top* of the bracket, so the predecessor's formula over-estimates
nonlinear broadening for most geometries. When beam quality dominates the bracket nearly
collapses — asserted, since that is the whole argument for reporting a range.

**Discarding charge is now reported, never decided** (D051). `illumination_report` returns
the fraction of *charge* (not particles — `Bunch.weight` is relative) below a threshold,
paired with the bunch's Gaussian `ks_excess`. The pairing is the point: the estimate rests
on a Gaussian picture, so a poorly-fitting bunch is exactly where the tails it would drop
are least trustworthy. `ks_excess` is `None` for an analytic beam, reported as
`gaussian_by_construction` — no model error, which is stronger than ignorance. `O(n_particles)`
and correctly so: it inspects a real bunch, and that cost belongs to a diagnostic rather than
to any engine.

Still blocked: applying the edge shift, pending §0's factor of two (author rethinking).

Full suite: 407 passed. `DECISIONS.md` D049 rewritten, D051 added.

---

## 2026-08-10 — Phase 4: nonlinear red-shift applied; §0 resolved and handed over

**§0's factor of two is resolved** (author): it is the polarization cycle average, not a
convention. `a0` is the normalized *peak* field magnitude, so `<a^2> = C a0^2` with
`C = 1/2` for linear polarization (the average of `cos^2`) and `C = 1` for circular, whose
magnitude is constant — the same factor by which circular carries twice the cycle-averaged
energy density at fixed `a0`. `io.laser._a0_from_density` implements the linear chain
explicitly, so `C = 1/2` here. Written up as `docs/DERIVATIONS.md` §C.

**Applied in analytical.** `angle_integrated_spectrum` takes `ahat` and puts the resonance
at `gamma^2 / (1 + ahat)`; `AnalyticalEngine` passes `0.5 * <a0^2>` and reports both `ahat`
and the shifted `compton_edge_energy` on `Results`. `ahat = 0` reproduces the linear result
bit for bit, which is what every other spectrum test assumes.

**Two findings for the 3b session, handed over.** The code's `ahat` feeds the resonance
denominator in `validation/references/delta.py` (`s_res = gamma^2/(1 + ahat + ...)`) and in
xigma Stage 2, so both over-state the nonlinear red-shift by two. More importantly: xigma
and delta **share** `TrajectorySamples.ahat()`, so the error is common-mode and §7's
four-way cross-check cannot see it — a gap in the cross-validation design, not only a bug.
Analytical is the leg that exposes it, since it takes `ahat` from the overlap integral
instead; the two should now differ by exactly two until the fix lands.

**§9.2 gains a concrete hook.** `C = 1/2 -> 1` is what `ellipticity` should interpolate, and
it enters twice — the resonance shift *and* the energy-to-`a0` conversion, since
`_a0_from_density`'s `sqrt(8 pi u)` is the linear relation. `ELLIPTICITY_IS_NOOP` is the
placeholder for both.

One process note: a rejected tool call had in fact already written the file, so re-running it
duplicated a test block and produced a syntax error. Caught and removed; worth remembering
that a rejection is not always a no-op.

Full suite: 410 passed. `DECISIONS.md` D049 updated; `docs/DERIVATIONS.md` §C added.

---

## How to update this file

- One dated section per work session (or per meaningful chunk of a session).
- State what actually landed, not what's planned — the plan lives in `GRAND_PLAN.md`.
- If a plan decision changes mid-implementation, record it in `GRAND_PLAN.md`'s own
  changelog (bump the version), and just link back to it here (`plan bumped to vX.Y —
  see its changelog`) rather than duplicating the rationale.
- Flag blockers explicitly (e.g. "waiting on author for §9.1" or "waiting on
  Spectral-FEM-Fields Python bindings") so the next session doesn't have to rediscover
  them.
