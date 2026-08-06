# GammaForge — Ground-Up Rebuild: Grand Plan

**Status:** draft v0.1 — 2026-08-06
**Author:** OpenAgent, in consultation with A. Samsonov (physics)

---

## 0. Context and goals

GammaForge computes properties of Compton photons produced by the interaction of
an electron bunch with a laser pulse. The predecessor repository
(`/home/alexander/Work/Code/ComptonSuite`, a.k.a. GammaForge) grew iteratively
through many refactors; this project is a **ground-up rebuild** with the old repo
kept solely as a historical reference (its commit log records what was tried and
rejected) and as a source of golden reference data for validation.

Goals, in priority order:

1. **Uniform, clean architecture** with one authoritative physics core and one
   engine interface. No more three different model styles.
2. **First-class engines:** `xigma` (tabulated 4D-overlap pipeline, GPU/CPU) and
   `analytical` (closed-form estimates). `kascade` is a *minimal* port kept only
   as a cross-validation method (a better Monte-Carlo will eventually replace
   it — do not invest in it). `delta` is a validation-only reference, never a
   production model.
3. **Headless-first framework** (library usable from Jupyter/scripts) with a
   **thin, schema-driven Tkinter GUI** as a first-class but non-physics layer.
4. **One internal unit system: CGS-Gaussian**, with engines converting at their
   own boundaries.
5. **Composable, cacheable pipeline stages** with a declarative dependency
   classification ("which parameters force a full rerun vs. a cheap requery").
6. **Validation suite rebuilt from scratch**: cross-engine consistency,
   closed-form limits, and golden snapshots cross-validated against the old
   repo. The old suite's tests existed largely to detect broken assumptions
   after changes — a symptom of poor design we must not reproduce.
7. **Resolve known physics bugs**: the ~2π kernel normalization discrepancy, the
   laser energy→a0 ellipticity gap; implement crossing angle from day one.

The accompanying paper (draft at `~/Work/Papers/2026/Compton-Numerics`) is the
physics authority. It is in flux: **if a discrepancy between the paper and the
code appears, mark it BLOCKING and resolve with the author before proceeding.**
The author performs the math derivations; consult them whenever anything in the
paper seems off.

---

## 1. Design principles (with provenance)

Every principle below either fixes a failure mode observed in the old repo or
honors a decision that was explicitly tried and rejected there.

| # | Principle | Rationale / provenance |
|---|-----------|------------------------|
| P1 | **CGS-Gaussian shared core** — every shared dataclass (beam, laser, target, results) stores canonical CGS-Gaussian values. | The old repo split SI (kascade/analytical) vs CGS+k0_las (xigma_i) and hand-converted at every boundary — the deepest legacy cut. One core system eliminates it. |
| P2 | **Kernels are pure functions; one thin stateful facade per engine owns caching.** | Hidden cache state caused real bugs (stale recompute caches, `engine.params` aliasing). Pure kernels are testable; the facade is the single audited cache-invalidation point. |
| P3 | **Composable, cacheable stages; GUI sees only an opaque `run()`.** | Old adapter orchestrated Stage 0/1/2 internally; the tracked tasks (a0 retarget, gamma rescale, cheap collimation requery) all need reused intermediates. |
| P4 | **Declarative dependency classification** — every schema field is tagged with the pipeline stages it affects; cache keys hash exactly the parameters each stage consumes. | This replaces the old bolted-on `spectrum_in_angular_range` + hardcoded n_energy caps + debouncing, and it is what makes grey-out/release, cheap requery, and scans natural. |
| P5 | **Uniform engine interface: typed parameter schema + `run() -> Results`.** | The old `ModelAdapter` fused a GUI contract (`(label, default, key)` triples, `Job.extra` stringly dict) with physics; kascade/xigma/analytical each did config differently. |
| P6 | **No generic spec/adapt-to-model framework.** Models convert typed parameters to whatever they need at their own boundary. | Rejected in old repo (ModelSpec/`adapt_to_model` built, demonstrated, never wired, then dropped). |
| P7 | **No `gammaforge.core` package.** The shared layer is one package; no intermediate layer between it and engines. | Rejected in old repo (a `core/` package was proposed and never built; consolidation into the shared layer won). |
| P8 | **No `BeamFittedParams` three-way split.** `GaussianElectronBeam` is both the analytic input description and the output of a fit. | Rejected in old repo. |
| P9 | **No `Results.cfg` back-reference**, no derived-value properties duplicating beam/laser fields, no `*_from_shared_fields` factories. | Rejected in old repo; fields are read at the point of use. |
| P10 | **No `ModelCapabilities` mechanism.** Capabilities are declarative *data* (which output slice-groups an engine can produce, which parameter groups are cheap to vary) attached to the engine; no capability registry/protocol machinery. | The old `ModelCapabilities` was dropped; capability detection then degraded to `hasattr` in the GUI. Declarative data restores it without the mechanism. |
| P11 | **No `Config` dataclass for xigma.** Engine numeric knobs live in the typed parameter schema, not in a mutable engine-side config object. | Rejected in old repo (adapters hold knobs as attributes); the schema makes this explicit and validated. |
| P12 | **GUI never computes physics and never branches on engine type.** It renders schema, calls `run()`, renders `Results`. | Old 1685-line monolith did field parsing, orchestration, stats, plotting, and type branching. |
| P13 | **One authoritative formula implementation per observable.** The three old "spectrum from H" implementations must converge on one derivation (paper) and validate against each other. | The ~2π discrepancy came from reimplementing the same math three times. |
| P14 | **Paper-code discrepancies are BLOCKING**, resolved with the author. | Explicit user instruction; the paper is the physics authority. |

---

## 2. Physical conventions

### 2.1 Units — CGS-Gaussian core

- Shared data and results are **CGS-Gaussian**: cm, s, g, erg, statC (electric
  charge), fields in statV/cm / gauss. Time in seconds (identical in SI/CGS).
- Physical constants have one source: pint's CODATA table, extracted to plain
  floats in CGS (`float(Quantity(1, name).to(...))` plus the exact textbook
  EM-unit conversion, `1 C = c[cm/s]/10 statC`, applied by hand — pint cannot
  convert electromagnetic quantities between SI and Gaussian).
- **k0_las normalization is an xigma-internal choice**, applied and undone at
  the xigma engine's boundary. The shared core is dimensional.
- **pint stays** as the display/serialization conversion layer (GUI unit
  dropdowns, YAML I/O) and the `light_time` context (pulse duration ↔ length).
  Kernels never see pint quantities; the schema converts at the boundary.
- **Width-convention semantics are kept for width-type parameters only**:
  transverse sizes (laser/beam) and longitudinal extents (pulse/bunch) carry a
  convention (RMS intensity / FWHM / 1/e² / field-RMS) alongside the unit.
  Everything else (energy, charge, emittance, angle, a0, beta_ff, ...) is a
  plain pint unit — **no `NoConvention` sentinel** for unambiguous fields.

### 2.2 Geometry convention

- Electron bunch propagates along `+z`; the laser counter-propagates along `-z`
  (head-on). `crossing_angle` tilts the laser propagation axis.
- **Crossing angle must be supported from day one.** Per the old task notes it
  should only affect the angular-spectrum derivation (the overlap uses photon
  density, so the electron–polarization angle is irrelevant); the precise form
  is to be derived with the paper — see §9.3.

### 2.3 Validity regime

- Weakly nonlinear: `a0 ≲ a0_max` (default 0.5), where a0 is the laser peak
  normalized vector potential.
- Ultrarelativistic electrons; quantum recoil negligible (q ≪ 1).
- Gamma-axis rescaling (varying mean energy without re-running Stages 0–1) is a
  **future feature** — design the stage boundaries so it *can* be added, do not
  build it now.

---

## 3. Core data model (`gammaforge.io`)

The shared layer. Depends on nothing in the repo. Package name `io` is retained
from the predecessor for continuity (P7: no `core/` package).

### 3.1 Parameter schema (`schema.py`)

The replacement for `(label, default, key)` triples, `Job.extra`, and bare
`StringVar` parsing. A typed, declarative description of every parameter the
GUI shows or an engine consumes.

```python
@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: FieldKind            # SCALAR | WIDTH | DURATION | CHOICE
    unit: str                  # canonical core unit (CGS) or "1" if dimensionless
    default: float | str
    display_units: tuple[str, ...]  # GUI dropdown choices (pint-convertible)
    convention: WidthConvention | TimeConvention | None  # width kinds only (P1)
    value_range: tuple[float, float] | None
    group: ParamGroup          # dependency classification (P4), see §5
    editable_after_run: bool = True   # grey-out/release behavior
```

- A `Parameters` object (frozen, validated) is what engines receive — no
  stringly dict. Validation is centralized (range, unit parse, convention).
- The GUI renders fields from `FieldSpec`: entry + unit dropdown (+ optional
  slider). This is what unlocks the tracked GUI tasks (unit dropdowns, sliders,
  grey-out, scans).
- Electron/laser/target field sets are declared once and shared by GUI, engines,
  and YAML I/O (single source of truth for parameter semantics).

### 3.2 Beam (`beam.py`)

- `GaussianElectronBeam` (frozen dataclass, CGS): bunch charge, kinetic energy
  (or γ0), relative energy spread, transverse sizes (convention-aware),
  emittances, bunch length (convention-aware), chirp/dispersion/Twiss
  correlations (dimensionless), `fit_quality` (None for pure input, populated by
  fit). All dimensional fields are plain CGS floats with the schema as their
  unit authority (P1). *This type is both the input description and the fit
  output (P8).*
- `Bunch` (macroparticle arrays in CGS): `x,y,z,thx,thy,gamma,weight,meta` +
  `gaussian_fit`. Raw simulation data; arrays are plain CGS floats.
- `sample_gaussian_canonical` / `sample_gaussian_bunch`: canonical-variable
  sampling with mass-shell enforcement (`pz` derived, never independently
  sampled).
- `drift` / `propagate` / `stream`: position propagation with the attached
  `gaussian_fit`'s Twiss tilt updated analytically in lockstep (no refit).
- `fit_gaussian`: covariance-based Twiss/chirp/dispersion fit with quality
  metrics (Mahalanobis, KS, log-likelihood).

### 3.3 Laser (`laser.py`)

- `GaussianParaxialLaser` (frozen dataclass, CGS): pulse energy, central
  wavelength, transverse size (convention-aware), pulse duration (convention-
  aware), plus plain scalars `beta_ff` (flying focus), `phi_pol`, `ellipticity`,
  `crossing_angle`.
- **Energy → a0 chain is fixed in the rebuild** (blocking the known old gap):
  pulse energy → peak intensity → a0 must be correct for elliptical
  polarization, using `TrXi/2 = (1+ellipticity²)/2`-style handling, shared by
  GUI and engines. The a0 *shape* along the trajectory stays engine-owned.
- Derived quantities (photon energy, a0, N_photons) are computed by small
  module-level helpers at the point of use, never stored as properties (P9).

### 3.4 Target (`target.py`)

New first-class concept (the old "target" was scattered between GUI fields and
adapter methods):

- Angular collimation window `(theta_x_col, theta_y_col)` (half-angles, rad).
- **Output requirements**: the requested observables with per-output resolution
  and (optional) range — replaces the old flat `OutputSpec`/`SliceRequest` and
  the GUI's "Energy bins/Time bins/..." text fields:
  - `total_yield` (always)
  - `spectrum` (angle-integrated, dN/dE) — resolution + energy range
  - `temporal_envelope` — resolution + time range
  - `spatial_distribution` (x,y) — resolution + range
  - `angular_distribution` (θx,θy) — resolution + range
  - `spectral_angular_distribution` (E,θx,θy) — resolution + ranges
  - `collimated_spectrum` (= "spectrum on target", the spectrum inside the
    collimation window) — resolution + ranges
  - `macroparticle_dump` (MC engines only: final electron + photon
    macroparticles) — statistics + serialization

### 3.5 Interaction (`interaction.py`)

- `InteractionParameters(beam: GaussianElectronBeam, laser: GaussianParaxialLaser,
  bunch: Bunch, target: Target)` — the compiled, pre-sampled, validated bundle
  every engine run consumes. One canonical sampling path (`io` samples,
  engines consume). No fields most engines ignore (P9).

### 3.6 Results (`results.py`)

- Axes are an **enum** (`Axis.ENERGY`, `Axis.TIME`, `Axis.X`, `Axis.Y`,
  `Axis.THETA_X`, `Axis.THETA_Y`) with **canonical CGS units** (erg, s, cm, rad);
  display conversion happens only in the plotting/serialization layers. The
  unit-baked-into-axis-name smell (`"E_eV"`) is gone.
- `PhasespaceSlice(axes: dict[Axis, ndarray], distr: ndarray)`: density over a
  named axis set; empty axes = 0D total yield. The closed set of allowed
  axis-groupings is kept and validated (the nine groupings from the old
  contract).
- `Results(photon_slices, macroparticles: Bunch|None, model_specific: dict)`.
  The MC macroparticle dump is an optional field used only by MC engines
  (P5/P12: GUI renders it as stats + save button, no type branching).
- One slice shape for all engines (density arrays); the old Sampled/Binned
  duck-typing concern disappears entirely (MC engines histogram their samples
  into slices).

---

## 4. Engine architecture (`gammaforge.engines`)

### 4.1 Uniform engine interface (`base.py`)

```python
class Engine(Protocol):
    name: str
    schema: Parameters              # typed parameter schema (P5)
    supported_outputs: tuple[frozenset[Axis], ...]   # declarative data (P10)
    cheap_groups: tuple[ParamGroup, ...]             # variably-cheap groups (P10/P4)

    def run(self, interaction: InteractionParameters,
            params: Parameters) -> Results: ...
```

- No `Job.extra`, no `model_params()` triples, no mutable adapter knobs (P11).
- "Which outputs can this engine produce" = declarative `supported_outputs` data
  (P10) — used by the GUI to enable/disable output checkboxes.
- Engines omit unsupported outputs from their results; the GUI renders what
  came back.
- A small registry (`ENGINES`) with lazy optional-dependency registration:
  xigma needs cupy/numba, everything else is pure.

### 4.2 xigma engine (`engines/xigma/`) — first-class

The tabulated-overlap pipeline, restructured into composable stages:

- **Stage 0 — trajectory integration** (`stages.py::integrate_trajectories`):
  pure function; per-particle ballistic trajectory + time-dependent laser
  envelope overlap, producing per-particle `L` (weight contribution), a0_shape,
  temporal/spatial diagnostics. Backends: numpy / cupy / numba, with the
  chunking machinery (VRAM/RAM-aware, halve-and-retry) preserved from the old
  repo — the *lessons* are requirements, the code is ported.
- **Stage 1 — H-table deposition** (`stages.py::deposit_table`): pure function;
  nearest/CIC deposition of `(gamma, θx, θy, a0)` into the 4D overlap table
  `H`. Includes `retarget_a0` (a0 rescale without re-deposition — an exact
  multiplicative relabel of the a0_shape axis) and the a0_shape vs ahat
  distinction.
- **Stage 2 — spectrum queries** (`stages.py::spectrum_from_table`,
  `angular_spectrum_from_table`, `spectrum_in_angular_range`): pure functions;
  the GPU/CPU kernels. **One authoritative normalization derived from the
  paper** (see §9.1) — the ~2π bug is resolved here, and the constant is
  isolated in one module-level location.
- **`Collision` facade** (`collision.py`): the one stateful object. Owns
  (beam, laser, bunch, target, engine params) and the stage cache. Methods are
  thin wrappers: `build_overlap()`, `spectrum(s)`, `angular_spectrum(...)`,
  `spectrum_in_angular_range(...)`, `run(output_requirements) -> Results`.
  Cache keys are hashes of the parameter groups each stage consumes (§5).
  Notebooks use the facade; validation uses the pure functions directly.
- **`Engine` wrapper** (`engine.py`): the thin `run()` used by the GUI — builds
  the interaction, calls the facade, returns `Results`. Opaque by contract (P3).

### 4.3 analytical engine (`engines/analytical/`) — first-class

Closed-form estimates, no per-particle Monte Carlo:

- `estimate_yield(beam, laser)`: total yield (closed form, SI-independent).
- `estimate_spectrum_width(beam, laser, theta_col)`: collimated width with a
  **per-component breakdown** — collimation `(γθ_col)⁴`, emittance/divergence
  `(γσ_θ)⁴`, energy spread `(σ_γ/γ)²`, nonlinearity `(a0²/2)²` — each reported
  separately (GUI shows a component table; total in quadrature).
- `angle_integrated_spectrum(...)`: quadrature over the beam's Gaussian energy
  distribution (no macroparticles — cost independent of n_particles; the old
  OOM bug class must not return).
- Growth items in scope: foci displacement, non-round beam total yield,
  collimated-spectrum construction (convolution of single-electron spectrum
  with energy distribution + a0).
- Role: GUI quick-estimate panel **and** validation anchor (§7).

### 4.4 kascade engine (`engines/kascade/`) — minimal port

- Ported **as-is** behind the uniform interface, minimal effort: convert the
  dict-based `run_simulation(cfg, n_mc, seed, electrons)` behind the `Engine`
  shape (its own SI internals convert at the boundary, P1).
- Purpose: one of the ≥4 cross-validation methods. Not first-class; not
  polished; may be excluded from the GUI model list by default.
- The future replacement MC is anticipated only at the interface level (P5) —
  no work on it now.

### 4.5 delta — validation reference (`validation/references/delta.py`)

- Brute-force per-macroparticle resonance binning reusing xigma's Stage 0.
- Lives in the **validation suite**, not the engine registry, not the GUI model
  list. Its job: an independent first-principles check on xigma's Stage 2.

---

## 5. Stage caching and dependency classification

Each parameter group declares which pipeline stages consume it. The xigma
facade keys its cache on exactly the consumed groups (P4):

| ParamGroup | Examples | Consumed by | Changing it requires |
|-----------|----------|-------------|----------------------|
| `BEAM_GEOMETRY` | sizes, emittances, Twiss, drift | Stage 0, Stage 1 | full rerun |
| `BEAM_DISTRIBUTION` | γ0, energy spread, chirp, dispersion | Stage 1 | Stage 1+2 rerun |
| `LASER_GEOMETRY` | spot size, pulse duration, beta_ff, crossing angle | Stage 0, Stage 1 | full rerun |
| `LASER_AMPLITUDE` | a0 (via pulse energy), polarization, ellipticity | Stage 1 (a0 axis), Stage 2 | Stage 1 retarget + Stage 2 |
| `CHARGE` | bunch charge (N_e) | Stage 1 weights, Stage 2 | Stage 1 + 2 |
| `TARGET` | collimation window | Stage 2 queries | cheap requery (no stage rerun) |
| `OUTPUT_GRID` | bins, ranges per output | Stage 2 queries | cheap requery |
| `NUMERICS` | n_steps, grid bins, chunk, device | all stages | full rerun |

- **GUI grey-out/release**: fields in `TARGET`/`OUTPUT_GRID` (and, for xigma,
  `LASER_AMPLITUDE` + `BEAM_DISTRIBUTION` when the geometry hash is unchanged)
  stay active after a run; everything else greys out until "release". This
  generalizes the old "XIGMA keeps pulse energy and gamma active" rule without
  hardcoding it in the GUI (P12).
- **Cheap requery**: `spectrum_in_angular_range` becomes a plain Stage-2 query
  against the cached table — no hardcoded n_energy caps in the engine; the GUI
  thread and render cadence handle responsiveness (debounce, and the old
  ceiling lessons move to the GUI layer as render-time budgets).
- **Scans**: a scan over `CHARGE`/`LASER_AMPLITUDE`/`TARGET` reuses cached
  stages; scans over `BEAM_GEOMETRY`/`LASER_GEOMETRY`/`NUMERICS` rerun the
  affected stages per point. Implemented as a small headless helper over the
  facade (§10 open item: GUI presentation of scan results).

Cache invalidation is **exact by construction**: each stage records the hash of
the parameter groups it consumed; a later call with identical hashes reuses the
cached intermediate. No heuristic staleness.

---

## 6. GUI architecture (`gammaforge.gui`)

Thin, schema-driven, two global tabs. No physics, no engine branching (P12).

**Tab 1 — Input & Calculate**
- Electrons panel, Laser panel (fields rendered from `FieldSpec`, unit
  dropdowns per field, sliders where declared).
- Target panel: collimation window fields + **required-output checkboxes** with
  per-output resolution/range controls (from §3.4).
- Model sub-tabs (one per available engine): each engine's typed parameters
  rendered from its schema, plus a Calculate button that runs **all checked
  engines** and shows status.
- Analytical estimates panel (always visible): total yield + per-component
  spectrum-width breakdown (§4.3).
- Geometry sketch panel (2D/3D): axes, electron/laser ellipses, polarization
  arrows, ghost foci at time delays — data derived from beam/laser, drawn by
  the GUI. (Design detail finalized later.)
- Grey-out/release behavior driven by schema groups (§5).

**Tab 2 — Outputs**
- Sub-tabs per requested output.
- Line plots: **all checked engines overlaid** (analytical included), distinct
  colors, per-series show/hide toggles.
- 2D colorplots: a picker list of the available 2D outputs (spatial,
  angular, spectral-angular).
- MC macroparticle output (when produced): statistics + "save particles"
  button.
- Save plots (PNG/PDF) and Save results (structured format, §8) buttons.
- Collimation/collimated-spectrum replot without re-run (§5 cheap requery).

GUI visual design details are deliberately left open for later refinement;
the data/architecture contract above is what the framework guarantees.

---

## 7. Validation strategy (rebuilt from scratch)

Guiding principle: **tests assert physics, not implementation**. No
"detect if an assumption was broken" test zoo.

- **Closed-form limits** (analytical as anchor):
  - Thomson limit: zero-a0 yield ≈ N_e · σ_T · (overlap) closed form.
  - Head-on geometry: xigma reproduces the analytic single-electron spectrum
    shape at the Compton edge; edge location and width vs theory.
  - Total yield: `∫ angular spectrum = ∫ spectrum = total_yield` — exact
    identities, not tolerances, where the contract guarantees them.
  - Convergence: results converge with `n_steps`, `n_bins`; **results are
    invariant under chunk size** (regression guard on the old OOM fixes).
- **Cross-engine consistency** (the ≥4 methods): xigma vs delta vs analytical
  vs kascade on shared `Scenario`s (baseline, scaled pulse energy, ...),
  tolerance-based.
- **Golden references from the old repo**: `validation/make_references.py`
  runs the old repo's models (path from env/config, e.g. `OLD_REPO`) on shared
  scenarios and writes committed snapshots under `validation/references/`.
  Default validation compares new vs golden; goldens are regenerated
  deliberately on your machine only. This is the *only* place the old repo is
  referenced (§11 note: docs should minimize old-repo references).
- **The ~2π resolution is encoded as a test**: the three "spectrum from H"
  paths (reference quadrature, direct binning, kernel) agree to tight
  tolerance; angular spectrum integrates to total yield exactly (§9.1).
- Scenario bank lives in `validation/scenarios.py`; runners per engine; a
  `run.py` orchestrator with pass/fail report.

---

## 8. Serialization and I/O

- **Beam/laser YAML** (evolved v0.2 specs): explicit units at the file
  boundary, converted to CGS on load; the schema is the single source of truth
  for field sets. Round-trip tested.
- **`.ele`/SDDS load** (elegant-format 6D distributions): kept; inherently
  SI/GeV — converted at the boundary (P1). Bunch charge override is explicit
  (the old no-charge-in-.ele caveat stays documented).
- **Results saving**: structured format for slices + parameters — HDF5
  (h5py) for arrays with a YAML sidecar for parameters, or numpy `.npz`+YAML
  (decide in Phase 1; h5py is allowed as a new dependency).
- **Graphs**: matplotlib PNG/PDF via a plotting module shared by GUI and
  headless use.
- **MC macroparticle dump**: per-engine serialization (e.g. `.ele`-compatible
  or HDF5) behind one interface.

---

## 9. Physics work items

### 9.1 ~2π kernel normalization (BLOCKING physics task)

- Status in old repo: three "spectrum from H" implementations
  (`reference.spectrum_from_table`, `spectrum_from_particles.direct_binning_spectrum`,
  `spectrum4d.spectrum_kernel_4d`) agree to ~15% with an unexplained ~2π
  absolute factor at the kernel level; user-facing integrals (angular spectrum
  → total yield) were made consistent.
- Rebuild: derive **one** authoritative normalization from the paper (with the
  author), isolate it in one constant, converge all three paths, encode
  identities as tests. Must be resolved in the rebuild, not deferred.

### 9.2 Laser energy→a0 with ellipticity

- Pulse energy → peak intensity → a0 correct for elliptical polarization
  (`(1+ε²)/2` factor); shared by GUI derivation, analytical, and xigma's
  boundary. Blocks the old documented gap.

### 9.3 Crossing angle (from day one)

- xigma supports `crossing_angle ≠ 0`. Per old notes it should affect only the
  angular-spectrum derivation (overlap uses photon density; the
  electron–polarization angle is irrelevant). Derive the exact form with the
  author; validate against kascade's (already arbitrary-angle) MC at
  intermediate angles. Design the Stage-2 kernels so the angle enters in one
  place.

### 9.4 Deferred (design for, don't build)

- Gamma-axis rescaling (vary mean energy without Stages 0–1 rerun).
- Jitter / shot averaging (ensemble over seeds).
- Strongly nonlinear regime (a0 > a0_max) and quantum regime (q ~ 1).

---

## 10. Open questions (resolve during implementation)

1. Results serialization format: HDF5 vs npz+YAML (decide Phase 1).
2. Scan GUI presentation: table + curve families (headless helper first).
3. Analytical-in-overlay semantics: analytical is a regular checkable engine in
   the GUI model list (checked by default) — confirm at GUI phase.
4. GUI visual design details (panels layout, sketch panel content) — later.
5. MC macroparticle dump format for the future MC.
6. Crossing-angle formula details (§9.3) — with the author.
7. k0_las normalization: confirm it stays fully internal to xigma (no leakage
   into shared results).

---

## 11. Phases and milestones

| Phase | Scope | Exit criteria |
|-------|-------|---------------|
| **0. Scaffold** | Repo, git, pyproject (Python 3.12), pytest, package skeleton, `.gitignore`, README, ADR index | `pytest` green on an empty-suite smoke test; `pip install -e .` works |
| **1. Core** | `io/`: schema, units/conventions (CGS), constants; beam; laser; target; interaction; sampling; results contract; YAML + `.ele` I/O | Round-trip tests; schema validation tests; CGS conversion tests vs known values |
| **2. Validation harness** | scenarios, runners skeleton, `make_references.py` + first golden snapshots from old repo | Golden generation runs; new-vs-golden comparisons execute |
| **3. xigma engine** | Stage 0/1/2 pure functions; Collision facade + stage cache; Engine wrapper; kernels (numpy/cupy/numba) with chunking; **~2π resolution**; **crossing angle**; ellipticity-a0 fix | §7 identities hold; golden cross-checks pass; convergence tests green |
| **4. analytical engine** | estimates + component breakdown; quadrature spectrum; growth items (foci displacement, non-round beam, collimated spectrum) | Closed-form limits match; validation anchor ready |
| **5. kascade port + delta** | minimal kascade behind interface; delta in validation suite | 4-method cross-validation runs |
| **6. GUI** | schema-driven two-tab app; overlays; save; grey-out/release; sketch panel | GUI runs headless-smoke; all planned interactions work |
| **7. Validation completion** | full scenario bank, convergence, chunk-invariance, closed-form identities | Full suite green; results reproducible |
| **8. Polish** | scans, docs, packaging, ADRs, notebook examples | Release-ready |

Order note: Phase 1 and the physics tasks of Phase 3 can overlap with paper work;
the architecture isolates normalization constants so physics fixes land in one
place.

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| ~2π normalization turns out to be a paper-level issue | It is a blocking item by design; architecture isolates it to one constant; cross-check against delta and kascade bounds it empirically |
| Crossing-angle derivation complexity | Validate against kascade MC at intermediate angles; enter geometry in one kernel location |
| Old-repo golden data encodes bugs | Goldens are transitional; closed-form identities + delta reference are the real anchors; goldens regenerated deliberately |
| Chunking regressions (OOM class) | Chunk-invariance property tests from Phase 3 on; VRAM/RAM-aware sizing lessons are requirements |
| pint friction with CGS | pint is confined to schema/serialization; kernels never see it; EM conversions hand-coded with tests vs known values |
| Scope creep (GUI polish, scans, sketches) | Explicitly deferred/late-phase; architecture supports them but they don't block physics milestones |
