# GammaForge — Ground-Up Rebuild: Grand Plan

**Status:** draft v0.2 — 2026-08-06
**Author:** OpenAgent, in consultation with A. Samsonov (physics)

**Changelog**
- **v0.2**: Full 3D collision geometry (§2.2); validity regime scoped to xigma only (§2.3);
  engine-declared recompute costs replace the global stage-tied ParamGroup (§3.1, §5);
  sampling without mass-shell "enforcement" + laser-interaction particle prefilter (§3.2);
  laser owns the field/envelope sampling API (§3.3, §4.2); auto-ranged outputs (§3.4);
  Phase 3 split into 3a (engineering) / 3b (physics closure) with minimal delta moved to
  Phase 2.5 as the ~2π arbiter (A2/A3); §9 reframed per paper audit — only ~2π is a
  "discrepancy to resolve", crossing-angle and ellipticity→a0 are net-new derivations
  (A1); one shared chunking utility instead of "port the code" (B1); enforced GUI
  import boundary (B3); kascade sanity check (B4); provenance-doc + doc-staleness +
  repo-hygiene + worktree-verification items (C1–C4); HDF5 results + elegant macroparticle
  export (§8, open Q1); analytical show/hide semantics (open Q3).

---

## 0. Context and goals

GammaForge computes properties of Compton photons produced by the interaction of an
electron bunch with a laser pulse. The predecessor repository
(`/home/alexander/Work/Code/ComptonSuite`, a.k.a. GammaForge) grew iteratively through
many refactors; this project is a **ground-up rebuild** with the old repo kept solely as
a historical reference (its commit log records what was tried and rejected) and as a
source of golden reference data for validation.

Goals, in priority order:

1. **Uniform, clean architecture** with one authoritative physics core and one engine
   interface. No more three different model styles.
2. **First-class engines:** `xigma` (tabulated 4D-overlap pipeline, GPU/CPU) and
   `analytical` (closed-form estimates). `kascade` is a *minimal* port kept only as a
   cross-validation method (a better Monte-Carlo will eventually replace it — do not
   invest in it). `delta` is a validation-only reference, never a production model.
3. **Headless-first framework** (library usable from Jupyter/scripts) with a **thin,
   schema-driven Tkinter GUI** as a first-class but non-physics layer.
4. **One internal unit system: CGS-Gaussian**, with engines converting at their own
   boundaries.
5. **Composable, cacheable pipeline stages** with engine-declared recompute-cost
   classification ("which parameters force a full rerun vs. a cheap requery").
6. **Validation suite rebuilt from scratch**: cross-engine consistency, closed-form
   limits, and golden snapshots cross-validated against the old repo. The old suite's
   tests existed largely to detect broken assumptions after changes — a symptom of poor
   design we must not reproduce.
7. **Resolve known physics bugs** (~2π kernel normalization) and **carry the
   architecture for the two net-new derivations** (crossing angle, ellipticity→a0)
   without letting them block engineering milestones.
8. **Preserve the predecessor's documentation discipline** as a deliberate deliverable:
   design decisions recorded *with their rejected alternatives* from Phase 0 onward
   (the old repo's docs are where most of the provenance in this plan comes from).

The accompanying paper (draft at `~/Work/Papers/2026/Compton-Numerics`) is the physics
authority. It is in flux: **if a discrepancy between the paper and the code appears,
mark it BLOCKING and resolve with the author before proceeding.** The author performs
the math derivations; consult them whenever anything in the paper seems off.

---

## 1. Design principles (with provenance)

Every principle below either fixes a failure mode observed in the old repo or honors a
decision that was explicitly tried and rejected there.

| # | Principle | Rationale / provenance |
|---|-----------|------------------------|
| P1 | **CGS-Gaussian shared core** — every shared dataclass (beam, laser, target, results) stores canonical CGS-Gaussian values. | The old repo split SI (kascade/analytical) vs CGS+k0_las (xigma_i) and hand-converted at every boundary — the deepest legacy cut. One core system eliminates it. |
| P2 | **Kernels are pure functions; one thin stateful facade per engine owns caching.** | Hidden cache state caused real bugs (stale recompute caches, `engine.params` aliasing). Pure kernels are testable; the facade is the single audited cache-invalidation point. |
| P3 | **Composable, cacheable stages; GUI sees only an opaque `run()`.** | Old adapter orchestrated Stage 0/1/2 internally; the tracked tasks (a0 retarget, cheap collimation requery) all need reused intermediates. |
| P4 | **Engine-declared recompute costs** — each engine declares, per schema field, how expensive it is to vary (QUERY_ONLY / REUSE_INTERMEDIATES / FULL_RERUN); cache keys hash exactly the inputs each engine intermediate consumes. | Replaces the old bolted-on `spectrum_in_angular_range` + hardcoded n_energy caps + debouncing, and generalizes across engines with different pipelines (MC = single stage). |
| P5 | **Uniform engine interface: typed parameter schema + `run() -> Results`.** | The old `ModelAdapter` fused a GUI contract (`(label, default, key)` triples, `Job.extra` stringly dict) with physics; kascade/xigma/analytical each did config differently. |
| P6 | **No generic spec/adapt-to-model framework.** Models convert typed parameters to whatever they need at their own boundary. | Rejected in old repo (ModelSpec/`adapt_to_model` built, demonstrated, never wired, then dropped). |
| P7 | **No `gammaforge.core` package.** The shared layer is one package; no intermediate layer between it and engines. | Rejected in old repo (a `core/` package was proposed and never built; consolidation into the shared layer won). |
| P8 | **No `BeamFittedParams` three-way split.** `GaussianElectronBeam` is both the analytic input description and the output of a fit. | Rejected in old repo. |
| P9 | **No `Results.cfg` back-reference**, no derived-value properties duplicating beam/laser fields, no `*_from_shared_fields` factories. | Rejected in old repo; fields are read at the point of use. |
| P10 | **Capabilities are declarative engine *data*, not a mechanism.** `supported_outputs` / recompute costs are plain tuples/dicts read directly off the `Engine` instance the GUI already holds. **Unlike the abandoned `ModelCapabilities`, there is no registry, no protocol, no `UnavailableAdapter` placeholder.** If this grows a registry, discovery mechanism, or capability-negotiation protocol, that is the old mistake recurring. | Old `ModelCapabilities` was built then deleted as unnecessary machinery; capability detection then degraded to `hasattr` in the GUI. Plain data restores it without the mechanism. |
| P11 | **No `Config` dataclass for xigma.** Engine numeric knobs live in the typed parameter schema, not in a mutable engine-side config object. | Rejected in old repo (adapters hold knobs as attributes); the schema makes this explicit and validated. |
| P12 | **GUI never computes physics and never branches on engine type.** It renders schema, calls `run()`, renders `Results`. **Enforced, not just stated:** a CI import-boundary check asserts `gammaforge.gui` imports only `Engine.run()`/`Results`/schema — never engine stages or kernels. | The old 1685-line monolith did field parsing, orchestration, stats, plotting, and type branching. It was trimmed to 1174 lines once and **regrew** to 1685 through normal feature additions — discipline alone failed; the boundary must be mechanically enforced. |
| P13 | **One authoritative formula implementation per observable.** The three old "spectrum from H" implementations must converge on one derivation and validate against each other (and against delta as the independent arbiter). | The ~2π discrepancy came from reimplementing the same math three times — and the paper's own normalization has never been checked against independent numerics, so convergence-to-paper is necessary but not sufficient (A1). |
| P14 | **Paper-code discrepancies are BLOCKING; absent derivations are parallel work.** Three distinct cases: (a) formula exists in paper, code disagrees → resolve against the paper (blocking); (b) formula exists in paper but is itself unvalidated → resolve in code, arbitrate independently (delta/analytical), treat paper agreement as necessary-not-sufficient; (c) derivation does not exist in paper at all → net-new research task with the author, wired as explicit no-op/identity in code until it lands, run in parallel, never blocking engineering milestones. | The paper audit (A1) found: ~2π has a derived normalization but the paper's validation study is an unwritten placeholder (case b); ellipticity→a0 has no formula in the paper at all — a0 is a given input and the polarization object is a normalized coherence matrix with no scalar ellipticity (case c); crossing angle is entirely absent, the angular-spectrum derivation being near-head-on only, O(θ²) (case c). |

---

## 2. Physical conventions

### 2.1 Units — CGS-Gaussian core

- Shared data and results are **CGS-Gaussian**: cm, s, g, erg, statC (electric charge),
  fields in statV/cm / gauss. Time in seconds (identical in SI/CGS).
- Physical constants have one source: pint's CODATA table, extracted to plain floats in
  CGS (`float(Quantity(1, name).to(...))` plus the exact textbook EM-unit conversion,
  `1 C = c[cm/s]/10 statC`, applied by hand — pint cannot convert electromagnetic
  quantities between SI and Gaussian).
- **k0_las normalization is purely internal to xigma** (confirmed; no leakage into
  shared results).
- **pint stays** as the display/serialization conversion layer (GUI unit dropdowns, YAML
  I/O) and the `light_time` context (pulse duration ↔ length). Kernels never see pint
  quantities; the schema converts at the boundary.
- **Width-convention semantics are kept for width-type parameters only**: transverse
  sizes (laser/beam) and longitudinal extents (pulse/bunch) carry a convention (RMS
  intensity / FWHM / 1/e² / field-RMS) alongside the unit. Everything else (energy,
  charge, emittance, angle, a0, beta_ff, ...) is a plain pint unit — **no
  `NoConvention` sentinel** for unambiguous fields.

### 2.2 Collision geometry (3D)

The electron bunch defines the lab X and Y axes and propagates along +z. The laser's
wave vector **k0** is a property of the laser and is described by **two inclination
angles** measured in the lab frame against the head-on configuration:

- `theta_xz` — tilt of k0 in the xz plane (generalizes the old single `crossing_angle`;
  `crossing_angle` is an alias for `theta_xz` with `theta_yz = 0`).
- `theta_yz` — tilt of k0 in the yz plane. Head-on = (0, 0), i.e. **k0 = −z**.

The plane transverse to k0 carries two more rotations:

- `psi_focus` — rotation of the two orthogonal transverse axes of the laser's focusing
  (for elliptical laser pulses, where the spot/waist differs along the two axes),
  measured from the electron's x-axis.
- `psi_pol` — rotation of the polarization axes (major/minor axis of the polarization
  ellipse) in the same transverse plane, measured from the electron's x-axis.

**Definition strategy (by design):** `psi_focus` and `psi_pol` are defined **as if the
collision were exactly head-on** (k0 = −z), measured from the electron's x-axis; the
whole laser field configuration (k0, focusing axes, polarization axes, field vectors) is
then transformed by the two k0 inclination angles into the actual 3D geometry. This keeps
the four angles independent, intuitive, and physical.

The laser therefore carries **at least four geometry angles** — `theta_xz`, `theta_yz`,
`psi_focus`, `psi_pol` — plus the scalar polarization degree `ellipticity` (see §3.3).
All are laser properties; nothing else in the framework re-defines them.

The **GUI interaction sketch** must visualize all of this: lab axes, electron-bunch
ellipsoid, laser k0 direction, focusing ellipse (two axes), and polarization ellipse —
including "ghost" foci at different time delays. The **same sketching functionality must
be available headless** (a drawing module shared by GUI and notebooks).

### 2.3 Validity regimes (per engine)

- **xigma:** weakly nonlinear (`a0 ≲ a0_max`, default 0.5), ultrarelativistic electrons,
  quantum recoil negligible (q ≪ 1). Crossing angle supported from day one in the
  architecture; its physics derivation is a parallel work item (§9.3). Gamma-axis
  rescaling is a future feature — design stage boundaries so it *can* be added, do not
  build it now.
- **Future Monte-Carlo (kascade's replacement):** full nonlinear (arbitrary a0) and
  quantum recoil support are explicit targets. Not xigma's regime — the two engines have
  different validity domains, and nothing in the core may assume either.
- **analytical:** closed-form estimates; validity as documented per formula.

---

## 3. Core data model (`gammaforge.io`)

The shared layer. Depends on nothing in the repo. Package name `io` is retained from the
predecessor for continuity (P7: no `core/` package).

### 3.1 Parameter schema (`schema.py`)

The replacement for `(label, default, key)` triples, `Job.extra`, and bare `StringVar`
parsing. A typed, declarative description of every parameter the GUI shows or an engine
consumes.

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
    editable_after_run: bool = True   # grey-out/release default (engine may override)
```

- A `Parameters` object (frozen, validated) is what engines receive — no stringly dict.
  Validation is centralized (range, unit parse, convention).
- **No global stage-tied `ParamGroup` on the schema.** Recompute-cost classification is
  **engine-declared** (P4): each engine maps its schema-field keys to a generic cost tier
  (`QUERY_ONLY` / `REUSE_INTERMEDIATES` / `FULL_RERUN`) via a plain dict. This resolves
  the old-plan flaw where the classification was tied to xigma's specific stages — the
  future single-stage MC still fits trivially (all physics fields `FULL_RERUN`, only
  output/target `QUERY_ONLY`). See §5.
- Electron/laser/target field sets are declared once and shared by GUI, engines, and YAML
  I/O (single source of truth for parameter semantics).

### 3.2 Beam (`beam.py`)

- `GaussianElectronBeam` (frozen dataclass, CGS): bunch charge, kinetic energy (or γ0),
  relative energy spread, transverse sizes (convention-aware), emittances, bunch length
  (convention-aware), chirp/dispersion/Twiss correlations (dimensionless), `fit_quality`
  (None for pure input, populated by fit). *This type is both the input description and
  the fit output (P8).*
- `Bunch` (macroparticle arrays in CGS): `x,y,z,thx,thy,gamma,weight,meta` +
  `gaussian_fit`. Raw simulation data; arrays are plain CGS floats.
- **Sampling (`sample_gaussian_*`):** sample the physical slice variables directly —
  `(x, y, z, gamma, thx, thy)` — as independent Gaussians, with the physically-motivated
  correlations applied explicitly (chirp → z–γ correlation, dispersion → x–γ / y–γ,
  Twiss tilt via `alpha`, drift-produced correlations). Momenta are then *derived*,
  `pz = sqrt((γ²−1)/(1+thx²+thy²))`, `px = thx·pz`, `py = thy·pz` — the mass-shell is
  satisfied identically **by construction**, and there is **no separate
  "mass-shell enforcement" step** (no rejection, no adjustment) to get wrong.
- **Laser-interaction prefilter:** not all macroparticles ever meet the laser. The laser
  (§3.3) provides an *active-region* query — the bounding space-time region where its
  amplitude exceeds a threshold. A shared io helper filters the bunch before an engine
  processes it, discarding particles that never enter the active region. **Discarding
  must conserve physics:** weights are renormalized (or the effective N_e reported)
  so total yield/charge stay correct. (This existed in the original xigma code and is
  being promoted to a first-class shared feature.)
- `drift` / `propagate` / `stream`: position propagation with the attached
  `gaussian_fit`'s Twiss tilt updated analytically in lockstep (no refit).
- `fit_gaussian`: covariance-based Twiss/chirp/dispersion fit with quality metrics
  (Mahalanobis, KS, log-likelihood).

### 3.3 Laser (`laser.py`)

- `GaussianParaxialLaser` (frozen dataclass, CGS): pulse energy, central wavelength,
  transverse sizes along the two focusing axes (convention-aware), pulse duration
  (convention-aware), plus plain scalars **`theta_xz`, `theta_yz`, `psi_focus`,
  `psi_pol`, `ellipticity`, `beta_ff`** (flying focus) — the full geometry of §2.2 is
  laser-owned.
- **The laser owns field sampling. It is NOT an engine responsibility.** The laser
  exposes a sampling API, callable **at any point in space and time**, in both modes:
  - **period-averaged**: the intensity envelope / `a0` profile (what the old code called
    `a0_shape`) — `laser.a0_profile(x, y, z, t)` and friends, including the
    trajectory-averaged quantity the engines consume;
  - **period-resolved**: the oscillating field (E/B or vector potential, including
    polarization) — `laser.field(x, y, z, t)`.
  These are **vectorized array-callable functions** (numpy/cupy style) — engines call
  them, never re-implement them. xigma's Stage 0 is not a raw GPU kernel (unlike the
  spectrum kernel), so calling an external Python-provided function vectorized is fine.
  *Design note:* the numba CPU path cannot generally jit an arbitrary external callable;
  the numba fallback must either evaluate the laser API vectorized outside the jitted
  loop or interpolate from a lattice the API builds once. This is an implementation
  requirement for the laser API, not a reason to move sampling back into engines.
- **Energy → a0 chain**: pulse energy → peak intensity → a0, correct for elliptical
  polarization. This is an open modeling task (A1/§9.2) — the paper has no formula; the
  architecture carries `ellipticity`, `psi_pol`, and the focusing axes as first-class
  parameters now, with the derivation wired once it lands.
- Derived quantities (photon energy, a0, N_photons) are computed by small module-level
  helpers at the point of use, never stored as properties (P9).

### 3.4 Target (`target.py`)

New first-class concept (the old "target" was scattered between GUI fields and adapter
methods):

- Angular collimation window `(theta_x_col, theta_y_col)` (half-angles, rad).
- **Output requirements** — the requested observables. **Ranges are auto-derived; the
  user never specifies them.** Each output carries only resolution (bins / sample
  points):
  - `total_yield` (always)
  - `spectrum` (angle-integrated, dN/dE) — resolution; **energy range auto-calculated**
    from beam γ0 and laser photon energy (Compton-edge kinematics)
  - `temporal_envelope` — resolution only; **no range field at all** — the time window
    is always computed automatically (the old codebase already has this function,
    `laser_overlap_time_window`; port it, generalize for crossing angle later)
  - `spatial_distribution` (x,y) — resolution; **autorange** (from beam/laser sizes),
    manual override only as an advanced option
  - `angular_distribution` (θx,θy) — resolution; **autorange** (~1/γ0 window)
  - `spectral_angular_distribution` (E,θx,θy) — resolutions; auto ranges
  - `collimated_spectrum` (= "spectrum on target", the spectrum inside the collimation
    window) — resolution; auto range
  - `macroparticle_dump` (MC engines only: final electron + photon macroparticles) —
    statistics + serialization
- The GUI plots allow **zooming** (best-effort matplotlib affordance) instead of manual
  range entry.

### 3.5 Interaction (`interaction.py`)

- `InteractionParameters(beam: GaussianElectronBeam, laser: GaussianParaxialLaser,
  bunch: Bunch, target: Target)` — the compiled, pre-sampled (and prefiltered, §3.2)
  bundle every engine run consumes. One canonical sampling path (`io` samples and
  prefilters, engines consume). No fields most engines ignore (P9).

### 3.6 Results (`results.py`)

- Axes are an **enum** (`Axis.ENERGY`, `Axis.TIME`, `Axis.X`, `Axis.Y`,
  `Axis.THETA_X`, `Axis.THETA_Y`) with **canonical CGS units** (erg, s, cm, rad);
  display conversion happens only in the plotting/serialization layers. The
  unit-baked-into-axis-name smell (`"E_eV"`) is gone.
- `PhasespaceSlice(axes: dict[Axis, ndarray], distr: ndarray)`: density over a named
  axis set; empty axes = 0D total yield. The closed set of allowed axis-groupings is
  kept and validated (the nine groupings from the old contract).
- `Results(photon_slices, macroparticles: Bunch|None, model_specific: dict)`.
  The MC macroparticle dump is an optional field used only by MC engines.
- One slice shape for all engines (density arrays); the old Sampled/Binned duck-typing
  concern disappears entirely (MC engines histogram their samples into slices).

---

## 4. Engine architecture (`gammaforge.engines`)

### 4.1 Uniform engine interface (`base.py`)

```python
class Engine(Protocol):
    name: str
    schema: Parameters              # typed parameter schema (P5)
    supported_outputs: tuple[frozenset[Axis], ...]   # declarative data (P10)
    recompute_costs: dict[str, RecomputeCost]        # field key -> cost tier (P4/P10)

    def run(self, interaction: InteractionParameters,
            params: Parameters) -> Results: ...
```

- No `Job.extra`, no `model_params()` triples, no mutable adapter knobs (P11).
- "Which outputs can this engine produce" = declarative `supported_outputs` data (P10) —
  used by the GUI to enable/disable output checkboxes. `recompute_costs` drives
  grey-out/release and cheap requery (§5).
- Engines omit unsupported outputs from their results; the GUI renders what came back.
- A small registry (`ENGINES`) with lazy optional-dependency registration: xigma needs
  cupy/numba, everything else is pure.

### 4.2 xigma engine (`engines/xigma/`) — first-class

The tabulated-overlap pipeline, restructured into composable stages:

- **Stage 0 — trajectory integration** (`stages.py::integrate_trajectories`): pure
  function; per-particle ballistic trajectory over the laser interaction window, with
  the laser's own field/envelope API (§3.3) evaluated along each trajectory — **the
  engine calls the laser; it does not define the a0 profile or envelope itself**.
  Produces per-particle `L` (weight contribution), trajectory-averaged a0, temporal/
  spatial diagnostics. Backends: numpy / cupy / numba. **Chunking: one shared
  auto-chunk + OOM-retry utility** (consuming `available_vram_bytes`/
  `available_ram_bytes`) used by *every* chunked stage — the old repo had **three
  inconsistent implementations** (particles.py, spectrum_from_particles.py, and the
  dead `build_table_streaming`); port the *algorithm and the hard-won constants* (the
  `_MAX_S_CHUNK` cap history, halve-and-retry policy), not the triplicated code, and
  retire the unwired manual-chunk design.
- **Stage 1 — H-table deposition** (`stages.py::deposit_table`): pure function;
  nearest/CIC deposition of `(gamma, θx, θy, a0)` into the 4D overlap table `H`.
  Includes `retarget_a0` (a0 rescale without re-deposition) and the a0_shape vs ahat
  distinction.
- **Stage 2 — spectrum queries** (`stages.py::spectrum_from_table`,
  `angular_spectrum_from_table`, `spectrum_in_angular_range`): pure functions; the
  GPU/CPU kernels. **One authoritative normalization, arbitrated against delta** (§9.1)
  — the ~2π discrepancy is resolved here, and the constant is isolated in one
  module-level location.
- **`Collision` facade** (`collision.py`): the one stateful object. Owns (beam, laser,
  bunch, target, engine params) and the stage cache. Methods are thin wrappers:
  `build_overlap()`, `spectrum(s)`, `angular_spectrum(...)`,
  `spectrum_in_angular_range(...)`, `run(output_requirements) -> Results`. Cache keys
  are hashes of the inputs each stage consumed (§5). Notebooks use the facade; validation
  uses the pure functions directly.
- **`Engine` wrapper** (`engine.py`): the thin `run()` used by the GUI — builds the
  interaction, calls the facade, returns `Results`. Opaque by contract (P3).
- **Geometry note:** `theta_xz`/`theta_yz`/`psi_focus`/`psi_pol` are first-class schema
  parameters from day one, but the *physics* of non-head-on geometry is wired as an
  identity/no-op until §9.3's derivation lands (P14c) — never silently approximate.

### 4.3 analytical engine (`engines/analytical/`) — first-class

Closed-form estimates, no per-particle Monte Carlo:

- `estimate_yield(beam, laser)`: total yield (closed form).
- `estimate_spectrum_width(beam, laser, theta_col)`: collimated width with a
  **per-component breakdown** — collimation `(γθ_col)⁴`, emittance/divergence
  `(γσ_θ)⁴`, energy spread `(σ_γ/γ)²`, nonlinearity `(a0²/2)²` — each reported
  separately (GUI shows a component table; total in quadrature).
- `angle_integrated_spectrum(...)`: quadrature over the beam's Gaussian energy
  distribution (no macroparticles — cost independent of n_particles; the old OOM bug
  class must not return).
- Growth items in scope: foci displacement, non-round beam total yield,
  collimated-spectrum construction (convolution of single-electron spectrum with energy
  distribution + a0).
- Role: GUI quick-estimate panel **and** validation anchor (§7).

### 4.4 kascade engine (`engines/kascade/`) — minimal port

- Ported **as-is** behind the uniform interface, minimal effort: convert the dict-based
  `run_simulation(cfg, n_mc, seed, electrons)` behind the `Engine` shape (its own SI
  internals convert at the boundary, P1).
- Purpose: one of the ≥4 cross-validation methods. Not first-class; not polished; off by
  default in the GUI.
- **Minimum sanity bar before it anchors validation:** kascade has **zero dedicated
  tests** in the old repo; it needs at least a closed-form check (e.g. Thomson-limit
  total yield) before the 4-method comparison may treat it as an independent leg.
- The future replacement MC is anticipated only at the interface level (P5) — no work on
  it now. Its regime (full nonlinear, quantum recoil) is explicitly different from
  xigma's (§2.3).

### 4.5 delta — validation reference (`validation/references/delta.py`)

- Brute-force per-macroparticle resonance binning reusing xigma's Stage 0.
- Lives in the **validation suite**, not the engine registry, not the GUI model list.
  Its job: an independent first-principles check on xigma's Stage 2 — and the **arbiter
  for the ~2π normalization** (§9.1), which is why a *minimal* delta is built early
  (Phase 2.5, scoped to Stage-2 normalization) with the full cross-validation role
  coming in Phase 5.

---

## 5. Recompute costs and caching (engine-declared, engine-generic)

The old-plan global `ParamGroup` enum tied to xigma's stages was rejected: the future
single-stage MC would not fit it. The mechanism is now **generic + engine-declared**:

- **`RecomputeCost`** is a small generic enum: `QUERY_ONLY` (varying the field only
  re-queries cached results — no engine stage re-runs), `REUSE_INTERMEDIATES` (varying
  the field reuses some cached engine intermediates — the engine decides which, via its
  internal cache keys), `FULL_RERUN` (varying the field invalidates everything).
- **Each engine declares** `recompute_costs: dict[field-key, RecomputeCost]` (P4/P10).
- **Engine-side cache keys** are derived from each engine's own dataflow: every cached
  intermediate (xigma: Stage 0 output, Stage 1 table; future MC: its single stage)
  records the hash of exactly the inputs it consumed; reuse happens iff hashes match.
  Invalidation is **exact by construction** — no heuristic staleness.

**Illustrative xigma mapping** (engine-declared data, not a global table):

| Field group (examples) | Cost tier | Why |
|------------------------|-----------|-----|
| beam sizes, emittances, Twiss, drift | FULL_RERUN | feeds Stage 0 + Stage 1 |
| γ0, energy spread, chirp, dispersion | REUSE_INTERMEDIATES | Stage 1 re-deposition; Stage 0 hash unchanged |
| laser spot, duration, beta_ff, geometry angles | FULL_RERUN | feeds Stage 0 + Stage 1 |
| pulse energy → a0, ellipticity | REUSE_INTERMEDIATES | Stage 1 a0-axis retarget (no re-deposition) |
| bunch charge (N_e) | REUSE_INTERMEDIATES | Stage 1 weights + Stage 2 rescale |
| collimation window | QUERY_ONLY | Stage 2 requery against cached table |
| output grid (bins per output) | QUERY_ONLY | Stage 2 requery |
| numerics (n_steps, grid bins, chunk, device) | FULL_RERUN | affects all stages |

**GUI behavior derived from the engine's declared costs** (no hardcoding):
- Fields declared `QUERY_ONLY`/`REUSE_INTERMEDIATES` stay active after a run; others grey
  out until "release" — generalizing the old "XIGMA keeps pulse energy and gamma active"
  rule to a mechanism every engine declares.
- **Cheap requery**: `spectrum_in_angular_range` is a plain Stage-2 query against the
  cached table — no hardcoded n_energy caps inside the engine; render-time budgets and
  debouncing live in the GUI layer.
- **Scans**: a scan over `QUERY_ONLY`/`REUSE_INTERMEDIATES` fields reuses cached stages;
  scans over `FULL_RERUN` fields rerun the affected stages per point. Implemented as a
  small headless helper over the facade.

---

## 6. GUI architecture (`gammaforge.gui`)

Thin, schema-driven, two global tabs. No physics, no engine branching (P12, enforced by
the import-boundary check).

**Tab 1 — Input & Calculate**
- Electrons panel, Laser panel (fields rendered from `FieldSpec`, unit dropdowns per
  field, sliders where declared). The Laser panel carries the four geometry angles of
  §2.2.
- Target panel: collimation window fields + **required-output checkboxes** with
  per-output resolution controls (ranges are auto-derived, §3.4).
- Model sub-tabs (one per available engine): each engine's typed parameters rendered
  from its schema, plus a Calculate button that runs **all checked engines** and shows
  status.
- Analytical estimates panel (always visible): total yield + per-component spectrum-width
  breakdown (§4.3). **analytical is not a checkboxable engine** — its estimates panel is
  separate; however, in the outputs tab every engine's curves (analytical included) are
  individually showable/hideable.
- Geometry sketch panel (2D/3D): lab axes, electron ellipsoid, laser k0 direction,
  focusing ellipse, polarization ellipse, ghost foci at time delays (see §2.2). The same
  drawing module is available headless.
- Grey-out/release behavior driven by the engine's declared recompute costs (§5).

**Tab 2 — Outputs**
- Sub-tabs per requested output.
- Line plots: all selected engines' curves drawn; **per-engine show/hide toggles**
  (analytical included); distinct colors.
- 2D colorplots: a picker list of the available 2D outputs (spatial, angular,
  spectral-angular).
- Plot zoom (best-effort) instead of manual range entry (§3.4).
- MC macroparticle output (when produced): statistics + "save particles" button.
- Save plots (PNG/PDF) and Save results (HDF5, §8) buttons.

GUI visual design details are deliberately left open for later refinement; the
data/architecture contract above is what the framework guarantees.

---

## 7. Validation strategy (rebuilt from scratch)

Guiding principle: **tests assert physics, not implementation**. No "detect if an
assumption was broken" test zoo.

- **Closed-form limits** (analytical as anchor):
  - Thomson limit: zero-a0 yield ≈ N_e · σ_T · (overlap) closed form.
  - Head-on geometry: xigma reproduces the analytic single-electron spectrum shape at
    the Compton edge; edge location and width vs theory.
  - Total yield: `∫ angular spectrum = ∫ spectrum = total_yield` — exact identities,
    not tolerances, where the contract guarantees them.
  - Convergence: results converge with `n_steps`, `n_bins`; **results are invariant
    under chunk size** (regression guard on the old OOM fixes; exercises the single
    shared chunking utility of §4.2).
- **Cross-engine consistency** (the ≥4 methods): xigma vs delta vs analytical vs kascade
  on shared `Scenario`s (baseline, scaled pulse energy, ...), tolerance-based. kascade
  only counts once its own Thomson-limit sanity check passes (§4.4).
- **~2π arbitration**: delta (built in Phase 2.5) is the independent arbiter; the paper
  formula is necessary-not-sufficient (its validation section is an unwritten
  placeholder, A1) — encode the identity tests regardless of paper agreement.
- **Golden references from the old repo**: `validation/make_references.py` runs the old
  repo's models (path from env/config, e.g. `OLD_REPO`) on shared scenarios and writes
  committed snapshots under `validation/references/`. Default validation compares new vs
  golden; goldens are regenerated deliberately on your machine only. This is the *only*
  place the old repo is referenced (§11 note: docs should minimize old-repo references).
- Scenario bank lives in `validation/scenarios.py`; runners per engine; a `run.py`
  orchestrator with pass/fail report.

---

## 8. Serialization and I/O

- **Results saving — HDF5** (h5py): slices (density arrays + axis values) + a YAML
  sidecar for parameters. Round-trip tested.
- **Macroparticle dump** (MC engines): HDF5 **and** export to elegant-compatible format
  (`.ele` or `.bun` — exact format to be researched during implementation; the old repo
  already has a hand-rolled SDDS ASCII `.ele` reader/writer to build on).
- **Beam/laser YAML** (evolved v0.2 specs): explicit units at the file boundary,
  converted to CGS on load; the schema is the single source of truth for field sets.
  Round-trip tested.
- **`.ele`/SDDS load** (elegant-format 6D distributions): kept; inherently SI/GeV —
  converted at the boundary (P1). Bunch charge override is explicit (the old
  no-charge-in-.ele caveat stays documented).
- **Graphs**: matplotlib PNG/PDF via a plotting module shared by GUI and headless use.

---

## 9. Physics work items

### 9.1 ~2π kernel normalization (BLOCKING physics task — "discrepancy to resolve")

- Status in old repo: three "spectrum from H" implementations
  (`reference.spectrum_from_table`, `spectrum_from_particles.direct_binning_spectrum`,
  `spectrum4d.spectrum_kernel_4d`) agree to ~15% with an unexplained ~2π absolute factor
  at the kernel level; user-facing integrals (angular spectrum → total yield) were made
  consistent.
- Paper status: the paper *does* give one derived normalization (`eq:main`/`eq:Fmatrix`),
  but its own validation study section is an unwritten placeholder — **the formula has
  never been checked against independent numerics**. Converging code to the paper is
  necessary but not sufficient (P14b).
- Rebuild: **delta (Phase 2.5) is the independent arbiter.** Derive one authoritative
  normalization, isolate it in one constant, converge all three paths, and encode the
  identities (`angular spectrum` → `total yield`; `kernel` ≈ `reference` ≈ `direct
  binning` to tight tolerance) as tests. Must be resolved in the rebuild, not deferred.

### 9.2 Laser energy→a0 with ellipticity (net-new derivation, parallel track)

- Paper status: **no formula exists.** `a0` is a given input in the paper; its
  polarization object is a normalized 2×2 coherence matrix with `Tr Ξ̂ ≡ 1` — there is
  no scalar "ellipticity" and no `(1+ε²)/2` identity in the source. Whether a scalar
  `ellipticity` schema parameter maps cleanly onto `Ξ̂` is an open modeling question for
  the author.
- Rebuild: carry `ellipticity`, `psi_pol`, focusing axes as first-class schema parameters
  now; wire the energy→a0 derivation as an **explicit no-op/identity** (documented)
  until the derivation lands. Never gating engineering milestones (P14c).

### 9.3 Crossing angle (net-new derivation, parallel track)

- Paper status: **genuinely absent.** The angular-spectrum derivation is built for
  near-backscattering geometry, accurate to O(θ²) around the collinear axis; the paper
  warns against extending it without revisiting the geometry.
- Rebuild: `theta_xz`/`theta_yz`/`psi_focus`/`psi_pol` are first-class parameters from
  day one (§2.2, §3.3); the non-head-on physics is wired as an identity/no-op until the
  derivation lands. Derivation proceeds with the author in parallel; validate against
  kascade's arbitrary-angle MC at intermediate angles; the angle enters the Stage-2
  kernels in one place. Never silently approximate (P14c).

### 9.4 Deferred (design for, don't build)

- Gamma-axis rescaling (vary mean energy without Stages 0–1 rerun).
- Jitter / shot averaging (ensemble over seeds).
- Full nonlinear and quantum regimes in the future MC (§2.3 — architecture must not
  preclude them).

---

## 10. Open questions (resolve during implementation)

1. **RESOLVED:** Results serialization = HDF5 + YAML sidecar. Macroparticle export to
   elegant `.ele` vs `.bun` — format research item during Phase 1/5.
2. Scan GUI presentation: table + curve families (headless helper first).
3. **RESOLVED:** analytical is **not** a regular checkboxable engine; its estimates panel
   is always shown, and in the outputs tab every engine's curves (analytical included)
   are individually showable/hideable.
4. GUI visual design details (panel layout, sketch panel content) — later.
5. MC macroparticle dump format (elegant `.ele`/`.bun` research, see 1).
6. Crossing-angle formula details (§9.3) — with the author, parallel track.
7. **RESOLVED:** k0_las normalization stays purely internal to xigma (confirmed).

---

## 11. Phases and milestones

| Phase | Scope | Exit criteria |
|-------|-------|---------------|
| **0. Scaffold** | Repo, git, pyproject (Python 3.12), pytest, package skeleton, README, ADR index + **`DECISIONS.md` provenance doc started** (C1); `.gitignore` **explicitly covers large data formats (`.ele`, notebooks with large outputs) and sync-conflict patterns from day one** (C3) | `pytest` green on an empty-suite smoke test; `pip install -e .` works; doc-staleness guard scaffolding in place (C2) |
| **1. Core** | `io/`: schema, units/conventions (CGS), constants; beam; laser (incl. field sampling API §3.3 and geometry angles); target (auto-ranges §3.4); interaction; sampling + prefilter (§3.2); results contract; YAML + `.ele` I/O; HDF5 results writer | Round-trip tests; schema validation tests; CGS conversion tests vs known values; laser sampling API tests (period-averaged + period-resolved at arbitrary points) |
| **2. Validation harness** | scenarios, runners skeleton, `make_references.py` + first golden snapshots from old repo; chunk-invariance test scaffolding (single shared chunking utility) | Golden generation runs; new-vs-golden comparisons execute |
| **2.5. Minimal delta** | Delta scoped to Stage-2 normalization arbitration (reuses xigma Stage 0 per §4.5) | Delta produces independent spectra on baseline scenarios; identity harness (`kernel` vs `reference` vs `direct binning` vs delta) executable |
| **3a. xigma engineering** | Stage 0/1/2 pure functions; Collision facade + stage cache; Engine wrapper; kernels (numpy/cupy/numba) with the **one shared chunking utility**; geometry/a0/ellipticity parameters wired as explicit identity/no-op placeholders (P14c) | Stage architecture tests green; chunk-invariance holds; placeholders documented |
| **3b. Physics closure** | ~2π resolution (§9.1), crossing-angle derivation (§9.3), ellipticity→a0 (§9.2) — **runs concurrently with Phases 4 and 5, not serially** | §9.1 closed with delta arbitration + identity tests; §9.2/§9.3 derivations landed if author completes them in parallel (never blocking 4–6) |
| **4. analytical engine** | estimates + component breakdown; quadrature spectrum; growth items (foci displacement, non-round beam, collimated spectrum) | Closed-form limits match; validation anchor ready |
| **5. kascade port + delta full role** | minimal kascade behind interface **+ its Thomson-limit sanity check (B4)**; delta full cross-validation role | 4-method cross-validation runs; kascade sanity check passes |
| **6. GUI** | schema-driven two-tab app; overlays + per-engine show/hide; save plots/HDF5; grey-out/release; sketch panel (headless module first); **import-boundary check enforced in CI (B3)** | GUI runs headless-smoke; all planned interactions work; boundary check green |
| **7. Validation completion** | full scenario bank, convergence, chunk-invariance, closed-form identities, golden cross-checks | Full suite green; results reproducible; 3b closures integrated |
| **8. Polish** | scans, docs, packaging, notebook examples, **doc-staleness sweep (C2)** | Release-ready |

Order note: Phase 3b is explicitly parallel; Phases 4–6 must not wait on physics
derivations (A3). **Before Phase 3a/6 kickoff, re-verify the old repo's remote
`worktree-*` branches are merged** (audited 2026-08-06: all five are merged into
`master`; a one-line `git branch -a` check suffices) so no half-finished work is
duplicated (C4).

---

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| ~2π normalization turns out to be a paper-level issue | Blocking item by design; delta (Phase 2.5) arbitrates independently — the paper formula is necessary-not-sufficient (its validation study is an unwritten placeholder); architecture isolates the constant to one location |
| Crossing-angle and ellipticity→a0 have **no existing derivation** in the paper (confirmed by audit, not just undocumented) | Open-ended research tasks, not consult-and-implement: parameters are first-class in schema/architecture now; physics wired as explicit identity/no-op until derivations land; derivation runs in parallel (P14c, §9.2/§9.3); never blocks Phases 3a–6 |
| Old-repo golden data encodes bugs | Goldens are transitional; closed-form identities + delta reference are the real anchors; goldens regenerated deliberately |
| Chunking regressions (OOM class) | Chunk-invariance property tests from Phase 2 on; single shared auto-chunk + OOM-retry utility (porting algorithm + constants, not the old triplicated code) |
| pint friction with CGS | pint confined to schema/serialization; kernels never see it; EM conversions hand-coded with tests vs known values |
| GUI regrows into a monolith (already happened once: 1174 → 1685 lines) | Import-boundary check in CI (P12/B3): `gammaforge.gui` may not import engine stages/kernels — mechanical enforcement, not discipline |
| Capability data regrows into a registry/protocol | P10 explicit guardrail: plain tuples/dicts on the Engine instance; any registry/negotiation layer is the old `ModelCapabilities` mistake recurring |
| Docs drift out of sync with code (old repo's AGENTS.md referenced deleted types) | Doc-staleness guard (grep-based CI smoke test for symbol names mentioned in docs) from Phase 0 (C2) |
| Scope creep (GUI polish, scans, sketches) | Explicitly deferred/late-phase; architecture supports them but they don't block physics milestones |
