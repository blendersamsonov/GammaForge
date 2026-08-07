# GammaForge — Implementation Progress

Tracks what has actually been built, session by session. `docs/GRAND_PLAN.md` is the
plan (with its own changelog of *design* decisions); this file is the log of *execution*
against that plan. Update it every session — append, don't rewrite history.

Phase numbers/names match `docs/GRAND_PLAN.md` §11.

---

## Status at a glance

| Phase | Status |
|-------|--------|
| 0. Scaffold | 🟢 done (pending only the pre-3a/6 worktree re-check, C4) |
| 1. Core | 🟢 done — all exit criteria met (see 2026-08-07 session below) |
| 2. Validation harness | 🟢 done — both exit criteria met (see 2026-08-07 Phase 2 session) |
| 2.5. Stage 0 + minimal delta | ⚪ not started |
| 3a. xigma engineering | ⚪ not started |
| 3b. Physics closure | ⚪ not started |
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
