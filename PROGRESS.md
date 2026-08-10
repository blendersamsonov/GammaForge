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
| 4. analytical engine | ⚪ not started |
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

## 2026-08-08 — `SPECTRAL_ANGULAR_DISTRIBUTION` removed from the `OutputKind` vocabulary

Dropped `OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION` entirely: it shared its `(E, θx, θy)`
axes and its fill code with `COLLIMATED_SPECTRUM`, differing only in an auto-derived
~1/γ0 radiation-cone angular window versus the target's actual collimation half-angles —
and no experiment measures the untruncated cone, only the collimated emission. Removed
from `OutputKind`, `SLICE_AXES`, `Target.auto_ranges`, `Collision._SUPPORTED`/`_fill`
(the shared branch now keys on `COLLIMATED_SPECTRUM` alone), and
`XigmaEngine.SUPPORTED_OUTPUTS`.

Along the way, fixed a latent bug the removal exposed: `make_references._KIND_BY_AXES`
excluded `COLLIMATED_SPECTRUM` from its axis-grouping map purely to disambiguate it from
`SPECTRAL_ANGULAR_DISTRIBUTION`'s identical axes; with that kind gone the exclusion would
have left the `(E, θx, θy)` grouping unmapped (`KeyError` on any old-repo slice with those
axes). Axis groupings now map 1:1 to `OutputKind`.

`tests/test_xigma_engine.py`'s `test_the_two_normalization_paths_inside_one_results_object_agree`
— the only test pinning the §9.1/D033 closed-form-vs-table-kernel `2 pi` regression guard
— was retargeted onto `COLLIMATED_SPECTRUM` (manually widened to `~4.6/gamma0` collimation
half-angles, matching the deleted auto-range's order) rather than deleted, so the guard it
provides is preserved. `test_spectral_angular_distribution_axis_order_matches_slice_axes`
was deleted outright — redundant once `COLLIMATED_SPECTRUM` is the only kind with those
axes.

`docs/GRAND_PLAN.md` bumped to v0.17 (§3.4's vocabulary list and table). `DECISIONS.md`
gained D052.

**Verification:** full `pytest` suite green.

---

## 2026-08-10 — `ahat` was twice the paper's; the polarization cycle average is now applied

Closed the §0 BLOCKING item `docs/DERIVATIONS.md` raised last session. The author settled
it: `a0` is the normalized **peak** field magnitude, so `<a^2> = C a0^2` with `C = 1/2`
linear and `1` circular, and `_a0_from_density` implements the linear chain — so
`a0_profile` is the linear peak envelope and the `1/2` was simply missing. Every `ahat` in
the repo was twice the paper's, overstating the nonlinear red-shift by 2x at both live
consumers (`delta.resonance_spectrum`'s `s_res`, and xigma's Stage 2 kernel).

`engines.xigma.stages.ahat_from_shape` is now the single route from `a0_shape` to `ahat`
and applies `io.laser.CYCLE_AVERAGE_FACTOR = 0.5`; `TrajectorySamples.ahat()`,
`.retargeted_ahat()` and `retarget_ahat()`'s axis rescale all go through it. `a0_shape`
keeps its literal meaning — the paper's `int|E|^4 / int|E|^2` — so the paper's own grouping
stays readable in the code. The photon count is deliberately untouched:
`photon_density_scale` inverts `_a0_from_density` exactly, so applying `C` there too would
double-count and would break §9.1's yield identity. Total yield is bit-for-bit unchanged.

**The whole suite passed through the fix, 293 tests, and that is the finding worth keeping.**
xigma and `delta` share `TrajectorySamples.ahat()`, so the error was common-mode and
cancelled in every xigma-vs-`delta` comparison — §7's cross-check machinery was blind to it
by construction. Worse, `run.py`'s fourth identity leg compares a **sum over `s`**, and the
red-shift moves photons along `s` while conserving that sum: the leg reads 0.9996 whether
`ahat` is right, doubled or halved. Both blind spots are now written into `GRAND_PLAN.md`
§7, and `tests/test_stage1_stage2.py::test_the_kernel_and_delta_agree_on_where_the_redshift_puts_the_photons`
watches the spectrum's **centroid** instead of its integral (a 1.2x kernel-side `ahat` bias
moves it 0.53%, against a 0.009% clean residual).

Also re-derived the stale annotations the fix invalidated: the synthetic fixtures'
`a0_shape=0.05/4.0 at a0_peak=0.3` now give `ahat = 0.00225/0.18` in target bins 0/6, not
`0.0045/0.36` in bins 0/17.

**Measured, before -> after.** Fourth identity leg: `near_a0_max` `0.9848 -> 1.0245`, the
other two unmoved, all well inside the `0.1` tolerance. Bank `ahat` (luminosity-weighted
mean): `baseline` `0.0114 -> 0.0057`, `low_a0` `0.00114 -> 0.00057`, `near_a0_max`
`0.0571 -> 0.0286`. Golden scalars unchanged (they compare `a0_peak`/`gamma0`/
`n_electrons`/`n_photons`, none of which touch `ahat`).

**Left open deliberately.** D032's target-grid defaults were tuned against `ahat` values
that were 2x too large, and halving them pushes the bank further into the grid's coarse
floor region (`near_a0_max`: four resolved bins -> two). The resulting centroid bias against
`delta` is ~1% and is dominated by the floor bin standing in at its own centre for
everything below 0.035 — **pre-existing, not created here** (`low_a0` moved `-1.59% ->
-1.65%`). `decades = 1.0 -> 0.3` at the same `n_bins`/`ahat_max` removes most of it
(kernel/`delta` at the spectral peak, `near_a0_max`: `0.60 -> 0.99`), but that is a
production default the author tuned with stated reasoning, so it is recorded with numbers
rather than changed. `DECISIONS.md` D053's last section has the sweep, and
`test_the_production_ahat_grid_under_resolves_the_bank_by_a_known_amount` pins the
shipping configuration's bias per scenario (`-1.13%`, `-1.63%`, `-0.04%`) so a future
change to `_ahat_target_edges` has to move it deliberately.

`docs/DERIVATIONS.md` §0 marked resolved, and §1.1 extended: under the author's peak
convention `C` enters both the energy->a0 chain and `ahat`, and the two **cancel**, so at
fixed pulse energy `ahat` is ellipticity-independent either way. §9.2's only physical effect
on the spectrum is therefore §1.2's kernel polarization factor; `a0_peak` itself is the one
reported number that would move. `ELLIPTICITY_IS_NOOP` and `EMISSION_IS_HEAD_ON` both stay
`True`.

`docs/GRAND_PLAN.md` bumped to v0.25 (numbering past the parallel Phase 4 branch's v0.19-v0.24
so the changelogs interleave on merge); `DECISIONS.md` gained D053, and the local
`SPECTRAL_ANGULAR_DISTRIBUTION` entry was renumbered D035 -> D052 to clear D035-D051, which
that branch had already taken.

**Verification:** full `pytest` green (298); `python -m gammaforge.validation.run` all checks
pass.

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
