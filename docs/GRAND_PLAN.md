# GammaForge — Ground-Up Rebuild: Grand Plan

**Status:** draft v0.30 — 2026-09-07
**Author:** OpenAgent, in consultation with A. Samsonov (physics)


## Navigation and Architecture Map

- **[0. Context and goals](#0-context-and-goals)** — Rebuild rationale and priority goals
- **[1. Design principles (with provenance)](#1-design-principles-with-provenance)** — Hard-won principles P1–P15
- **[2. Physical conventions](#2-physical-conventions)** — Units (CGS-Gaussian), coordinate systems, collision geometry
- **[3. Core data model (`gammaforge.io`)](#3-core-data-model-gammaforgeio)** — Bunch, LaserField, Target, Results
- **[4. Engine architecture (`gammaforge.engines`)](#4-engine-architecture-gammaforgeengines)** — Engine contract, xigma, analytical, kascade
- **[5. Recompute costs and caching](#5-recompute-costs-and-caching-engine-declared-engine-generic)** — Query/intermediate/rerun cost tiers
- **[6. GUI architecture (`gammaforge.gui`)](#6-gui-architecture-gammaforgegui)** — NiceGUI layout and runner boundary
- **[7. Validation strategy](#7-validation-strategy-rebuilt-from-scratch)** — Scenarios, identity legs, golden snapshots
- **[8. Serialization and I/O](#8-serialization-and-io)** — HDF5 and YAML formats
- **[9. Physics work items](#9-physics-work-items)** — §9.1 normalization, §9.2 ellipticity, §9.3 crossing angle & electron direction
- **[10. Open questions](#10-open-questions-resolve-during-implementation)** — Resolved and active design questions
- **[11. Phases and milestones](#11-phases-and-milestones)** — Phase 0 through 8 exit criteria
- **[12. Risks and mitigations](#12-risks-and-mitigations)** — Guards C1–C4, risk register
- **[Changelog](#changelog)** — Plan evolution history (v0.29 down to v0.2)

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
   schema-driven NiceGUI browser GUI** as a first-class but non-physics layer.
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
9. **Pluggable laser field representation**: the physics core depends only on a
   laser-sampling protocol (`LaserField`, §3.3/P15), never a concrete laser type.
   `GaussianParaxialLaser` is today's only implementation; an arbitrary spectrally/FEM-
   represented pulse — from the sibling `Spectral-FEM-Fields` project, once it grows
   Python bindings — is anticipated as a second implementation requiring **zero** engine
   changes when it lands.

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
| P4 | **Engine-declared recompute costs** — each engine declares, per schema field, how expensive it is to vary (QUERY_ONLY / REUSE_INTERMEDIATES / FULL_RERUN); cache keys hash exactly the inputs each engine intermediate consumes. Tiers express how cheap a *re-Calculate* is. | Replaces the old bolted-on `spectrum_in_angular_range` + hardcoded n_energy caps + debounced live updates (engines are now Calculate-gated, §5), and generalizes across engines with different pipelines (MC = single stage). |
| P5 | **Uniform engine interface: typed parameter schema + `run() -> Results`.** | The old `ModelAdapter` fused a GUI contract (`(label, default, key)` triples, `Job.extra` stringly dict) with physics; kascade/xigma/analytical each did config differently. |
| P6 | **No generic spec/adapt-to-model framework.** Models convert typed parameters to whatever they need at their own boundary. | Rejected in old repo (ModelSpec/`adapt_to_model` built, demonstrated, never wired, then dropped). |
| P7 | **No `gammaforge.core` package.** The shared layer is one package; no intermediate layer between it and engines. | Rejected in old repo (a `core/` package was proposed and never built; consolidation into the shared layer won). |
| P8 | **No `BeamFittedParams` three-way split.** `GaussianElectronBeam` is both the analytic input description and the output of a fit. | Rejected in old repo. |
| P9 | **No `Results.cfg` back-reference**, no derived-value properties duplicating beam/laser fields, no `*_from_shared_fields` factories. | Rejected in old repo; fields are read at the point of use. |
| P10 | **Capabilities are declarative engine *data*, not a mechanism.** `supported_outputs` / recompute costs are plain tuples/dicts read directly off the `Engine` instance the GUI already holds. **Unlike the abandoned `ModelCapabilities`, there is no registry, no protocol, no `UnavailableAdapter` placeholder.** If this grows a registry, discovery mechanism, or capability-negotiation protocol, that is the old mistake recurring. | Old `ModelCapabilities` was built then deleted as unnecessary machinery; capability detection then degraded to `hasattr` in the GUI. Plain data restores it without the mechanism. |
| P11 | **No `Config` dataclass for xigma.** Engine numeric knobs live in the typed parameter schema, not in a mutable engine-side config object. | Rejected in old repo (adapters hold knobs as attributes); the schema makes this explicit and validated. |
| P12 | **GUI never computes physics and never branches on engine type.** It renders schema, calls `run()`, renders `Results`. **Enforced, not just stated:** a CI import-boundary check asserts `gammaforge.gui` never imports engine internals — `engines/*/stages.py`, kernel modules, or any engine-specific stateful facade (e.g. xigma's `Collision`) — regardless of what else it legitimately imports from `io` (schema, `Engine.run()`/`Results`, the shared drawing module, YAML/HDF5 I/O). | The old 1685-line monolith did field parsing, orchestration, stats, plotting, and type branching. It was trimmed to 1174 lines once and **regrew** to 1685 through normal feature additions — discipline alone failed; the boundary must be mechanically enforced. |
| P13 | **One authoritative formula implementation per observable.** The three old "spectrum from H" implementations must converge on one derivation and validate against each other (and against delta as the independent arbiter). | The ~2π discrepancy came from reimplementing the same math three times — and the paper's own normalization has never been checked against independent numerics, so convergence-to-paper is necessary but not sufficient (A1). |
| P14 | **Paper-code discrepancies are BLOCKING; absent derivations are parallel work.** Three distinct cases: (a) formula exists in paper, code disagrees → resolve against the paper (blocking); (b) formula exists in paper but is itself unvalidated → resolve in code, arbitrate independently (delta/analytical), treat paper agreement as necessary-not-sufficient; (c) derivation does not exist in paper at all → net-new research task with the author, wired as explicit no-op/identity in code until it lands, run in parallel, never blocking engineering milestones. | The paper audit (A1) found: ~2π has a derived normalization but the paper's validation study is an unwritten placeholder (case b); ellipticity→a0 has no formula in the paper at all — a0 is a given input and the polarization object is a normalized coherence matrix with no scalar ellipticity (case c); crossing angle is entirely absent, the angular-spectrum derivation being near-head-on only, O(θ²) (case c). |
| P15 | **Laser field source is pluggable behind a `LaserField` protocol.** Engines consume its lab-frame, vectorized sampling methods (`intensity_profile`, `a0_profile`, `field`, `active_region`, §3.3) — never a concrete laser type. `GaussianParaxialLaser` is the initial (and currently primary) implementation. Stage 0 samples arbitrary fields; quasi-monochromatic engines (xigma, kascade) consume field sampling alongside physical carrier and polarization invariants; `AnalyticalEngine` explicitly requires `GaussianParaxialLaser` (RES067). | A sibling project (`Spectral-FEM-Fields`) is building a numerically efficient field representation not restricted to the paraxial Gaussian; once it grows Python bindings it can slot in as a second `LaserField` implementation. |

---

## 2. Physical conventions

### 2.1 Units — CGS-Gaussian core

- Shared data and results are **CGS-Gaussian**: cm, s, g, erg, statC (electric charge),
  fields in statV/cm / gauss. Time in seconds (identical in SI/CGS).
- Physical constants have one source: pint's CODATA table, extracted to plain floats in
  CGS (`float(Quantity(1, name).to(...))` plus the exact textbook EM-unit conversion,
  `1 C = c[cm/s]/10 statC`, applied by hand — pint cannot convert electromagnetic
  quantities between SI and Gaussian).
- **No coordinate normalization anywhere.** Engines work in CGS directly. The
  predecessor's `k0_las`-normalized xigma pipeline is not carried over: its `k0` factors
  are the Jacobian of that normalization, not physics (see §4.2), and normalizing would
  conflict with the lab-frame `LaserField` contract (P15).
- **pint is the boundary type as well as the boundary converter.** Every dimensioned field
  of a shared physics dataclass is a `Quantity`, stored canonically in CGS; an engine
  unpacks at its own `run()` boundary, and **kernels never see a `Quantity`**. Bulk
  per-particle arrays are raw ndarrays with declared units instead (`Bunch.UNITS`), since a
  6D beam covariance is dimensionally heterogeneous. The `light_time` context (longitudinal
  extent ↔ duration) is opt-in per field; the `gaussian_charge` context is always on.
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

**Rotation convention (pinned):** the 3D rotation is fixed by composition order —
**R = R_y(θxz) · R_x(θyz)** (extrinsic: first a tilt θyz about the lab x̂, then θxz about
the lab ŷ). The **roll about k̂ is thereby determined by the rotation order**, not a free
parameter; `psi_focus`/`psi_pol` are measured from the transported x̂ (R applied to the
head-on x̂) in the plane perpendicular to k̂. A **round-trip test** (applying R⁻¹ recovers
the head-on lab angles) is required. Compatibility aliases: old `crossing_angle ≡
(θxz, 0)`; old `phi_pol` → `psi_pol`.

The laser therefore carries **at least four geometry angles** — `theta_xz`, `theta_yz`,
`psi_focus`, `psi_pol` — plus the scalar polarization degree `ellipticity` (see §3.3).
All are laser properties; nothing else in the framework re-defines them. **These angles
parameterize `GaussianParaxialLaser` specifically, not the abstract field-sampling
contract** — see the `LaserField` protocol (§3.3/P15) for why a differently-parameterized
future laser implementation is free to place itself in the lab frame however it needs to.

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
```

- A `Parameters` object (frozen, validated) is what engines receive — no stringly dict.
  Validation is centralized (range, unit parse, convention).
- **No global stage-tied `ParamGroup` on the schema.** Recompute-cost classification is
  **engine-declared** (P4): each engine maps its schema-field keys to a generic cost tier
  (`QUERY_ONLY` / `REUSE_INTERMEDIATES` / `FULL_RERUN`) via a plain dict. This resolves
  the old-plan flaw where the classification was tied to xigma's specific stages — the
  future single-stage MC still fits trivially (all physics fields `FULL_RERUN`, only
  output/target `QUERY_ONLY`). See §5. `FieldSpec` carries **no** editable/grey-out flag —
  that state is derived solely from the engine's `recompute_costs` (§5).
- Electron/laser/target field sets are declared once and shared by GUI, engines, and YAML
  I/O (single source of truth for parameter semantics).

### 3.2 Bunch and beam description (`bunch.py`)

- `GaussianElectronBeam` (frozen dataclass, CGS): bunch charge, kinetic energy (or γ0),
  relative energy spread, transverse sizes (convention-aware), emittances, bunch length
  (convention-aware), chirp/dispersion/Twiss correlations (dimensionless), `fit_quality`
  (None for pure input, populated by fit). *This type is both the input description and
  the fit output (P8).*
- `Bunch` (macroparticle arrays in CGS): `x,y,z,thx,thy,gamma,weight,meta` +
  `gaussian_fit`. Raw simulation data; arrays are plain CGS floats. **`weight` is the
  relative weight (`1/n_particles`)**; the old `n_electrons = weight·n` property is gone —
  the total electron count is the interaction's N_e (§3.5).
- **Sampling (`sample_gaussian_*`):** sample the physical slice variables directly —
  `(x, y, z, gamma, thx, thy)` — as independent Gaussians, with the physically-motivated
  correlations applied explicitly (chirp → z–γ correlation, dispersion → x–γ / y–γ,
  Twiss tilt via `alpha`, drift-produced correlations). Momenta are then *derived*,
  `pz = sqrt((γ²−1)/(1+thx²+thy²))`, `px = thx·pz`, `py = thy·pz` — the mass-shell is
  satisfied identically **by construction**, and there is **no separate
  "mass-shell enforcement" step** (no rejection, no adjustment) to get wrong.
- **RNG architecture (pinned):** the sampler draws from **independent per-variable RNG
  substreams keyed off `seed`** (e.g. `numpy.random.Generator.spawn()` or per-variable
  child seeds) — fixed standard-normal deviates per particle per variable, with the
  *current* beam parameters' affine transform (mean, std, correlation) applied on top.
  This is what makes the bunch resample rule (§3.5) compatible with the
  `REUSE_INTERMEDIATES` cost tier (§5): changing only γ0/energy-spread/chirp/dispersion
  perturbs only the gamma draw, leaving the position/angle deviates — and therefore
  Stage 0's actual inputs — bit-identical, so "Stage 0 hash unchanged" (§5 table) holds
  by construction rather than by accident of whichever RNG stream shape happens to get
  implemented.
- **Laser-interaction prefilter:** not all macroparticles ever meet the laser. The laser
  (§3.3) provides an *active-region* query (`active_region(...)`, part of the
  `LaserField` protocol, §3.3/P15) — the bounding space-time region where its
  (period-averaged) amplitude exceeds a threshold, **expressed as a fraction of the peak
  a0** (e.g. `a0_profile ≥ threshold · a0_peak`; default pinned in Phase 1). A shared io helper filters the bunch before an engine
  processes it, discarding particles that never enter the active region (they contribute
  L = 0 to every engine). **Weights are never renormalized on discard** — it is a pure
  computational optimization: with the same seed, results are identical with the
  prefilter on or off (tested invariant, §7). (Existed in the original xigma code;
  promoted to a first-class shared feature.)
- **Weights are relative** (`1/n_particles`): N_e lives as a separate scalar on the
  interaction (§3.5), and every output is exactly linear in N_e — see A3 resolution.
- `drift` / `propagate` / `stream`: position propagation with the attached
  `gaussian_fit`'s Twiss tilt updated analytically in lockstep (no refit).
- `fit_gaussian`: covariance-based Twiss/chirp/dispersion fit with quality metrics
  (Mahalanobis, KS, log-likelihood).

### 3.3 Laser (`laser.py`)

**`LaserField` protocol — the sampling contract engines depend on (P15).** This is the
*only* thing engines are typed against:

```python
class LaserField(Protocol):
    def a0_profile(self, x, y, z, t) -> ndarray: ...   # period-averaged envelope
    def field(self, x, y, z, t) -> ndarray: ...         # period-resolved E/B (or vector potential)
    def active_region(self, ...) -> ...: ...            # bounding space-time region (§3.2 prefilter)
```

- Both sampling methods are **vectorized, array-callable, lab-frame** functions
  (numpy/cupy style) — engines call them, never re-implement field physics themselves.
  xigma's Stage 0 is not a raw GPU kernel (unlike the spectrum kernel), so calling an
  external Python-provided function vectorized is fine. *Design note:* the numba CPU
  path cannot generally jit an arbitrary external callable; the numba fallback must
  either evaluate the API vectorized outside the jitted loop or interpolate from a
  lattice the implementation builds once. This is a requirement on `LaserField`
  implementations, not a reason to move sampling back into engines.
- **How a concrete implementation gets to lab-frame coordinates is its own business.**
  `GaussianParaxialLaser` (below) does it via the four §2.2 geometry angles applied to a
  head-on-frame analytic model. A future implementation is free to use an entirely
  different internal parameterization — e.g. a field already expressed directly in lab
  coordinates — as long as `a0_profile`/`field`/`active_region` behave correctly for
  lab-frame `(x, y, z, t)` input. The protocol says nothing about *how* a field is
  represented, only what engines may assume about querying it.
- **Anticipated second implementation (interface-only, no work now, P15):** the sibling
  `Spectral-FEM-Fields` project (`~/Work/Code/Spectral-FEM-Fields`) is building a
  numerically efficient functional/spectral representation of electromagnetic fields,
  not restricted to the paraxial Gaussian approximation — its own `RepresentationAdapter`
  abstraction (grid vs. functional field evaluation) is a close analogue of the split
  being made here. Once it grows Python bindings, a thin `LaserField`-conforming wrapper
  around it lets GammaForge represent **arbitrary laser pulses**, with Stage 0 sampling
  the field directly and quasi-monochromatic scattering engines consuming carrier
  and polarization metadata (RES067). Not built now; not gated on by anything in this plan.

**`GaussianParaxialLaser` — today's (and Phase 1's only) `LaserField` implementation:**

- Frozen dataclass, CGS: pulse energy, central wavelength, **elliptical, astigmatic
  Gaussian** transverse profile — per-axis waist sizes `sigma_x`, `sigma_y`
  (convention-aware) and per-axis focal offsets `z_fx`, `z_fy` (astigmatism: different
  focal position per axis; the round beam is the degenerate case `sigma_x = sigma_y`,
  `z_fx = z_fy`), with the focusing axes rotated by `psi_focus` — pulse duration
  (convention-aware), plus plain scalars **`theta_xz`, `theta_yz`, `psi_focus`,
  `psi_pol`, `ellipticity`** (polarization degree — *distinct from spot ellipticity*,
  which is expressed by the per-axis waists), **`beta_ff`** (flying focus). The full
  geometry of §2.2 is owned here.
- Implements `LaserField` directly: `a0_profile`/`field` are the analytic Gaussian
  formulas (what the old code called `a0_shape` for the envelope); `active_region` is
  the closed-form Gaussian bounding region.
- **Energy → a0 chain**: pulse energy → peak intensity → a0, correct for elliptical
  polarization. This is an open modeling task (A1/§9.2) — the paper has no formula; the
  architecture carries `ellipticity`, `psi_pol`, and the focusing axes as first-class
  parameters now, with the derivation wired once it lands.
- Derived quantities (photon energy, a0, N_photons) are computed by small module-level
  helpers at the point of use, never stored as properties (P9).

**`fit_gaussian_paraxial(laser: LaserField) -> GaussianParaxialLaser`** — the laser-side
analogue of `Bunch.fit_gaussian` (§3.2/P8): extracts descriptive metrics (waist size,
Rayleigh range, effective wavelength/duration, peak a0, ...) from **any** `LaserField`
implementation, not just an analytic Gaussian one, so callers that only need a rough
physical picture — target autoranging (§3.4), the analytical engine's closed-form
formulas (§4.3), the GUI sketch panel — never have to sample the raw field themselves.
For a `GaussianParaxialLaser` input the fit is an identity (the parameters are already
exact); for a future non-Gaussian `LaserField` (e.g. a `Spectral-FEM-Fields`-backed one)
it does real work — however that's implemented (numerical sampling + least-squares
against the paraxial formula, or reading the source's own internal moments if it exposes
them). Same P8 discipline applies: no `LaserFittedParams` split, no lazily-cached
property (P9) — called explicitly at the point of use, result used immediately.

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
  - `collimated_spectrum` (= "spectrum on target") — a **3D slice in (E, θx, θy)** whose
    **angular ranges are the target's collimation window** (user-defined, not auto;
    energy range auto). It is a distinct output from the fully angle-integrated
    `spectrum`: the 1D "spectrum on target" is a *visualization* of this 3D slice, not a
    separate computation. GUI pipeline: 2D slices at θx = 0 and θy = 0; 2D energy–angle
    distributions summed over the other angle; finally the 1D spectrum on target (summed
    over both angles within the window).
  - `macroparticle_dump` (MC engines only: final electron + photon macroparticles) —
    statistics + serialization; **not a slice** (separate entry, §3.6)
- The GUI plots allow **zooming** (best-effort matplotlib affordance) instead of manual
  range entry.
- **Autoranging reads descriptive laser metrics, never raw field samples.** Anywhere
  "laser size" feeds an autorange (`spatial_distribution`, `angular_distribution`, the
  Compton-edge energy range, ...), it comes from the laser's `GaussianParaxialLaser`
  view — the object itself if it already is one, or `fit_gaussian_paraxial(laser)`'s
  output otherwise (§3.3) — never from ad hoc re-derivation off `LaserField` samples.
  This is what keeps autoranging correct once a non-Gaussian `LaserField` lands (P15).

**OutputKind vocabulary.** The outputs above are the engine-facing vocabulary; each maps
to a slice shape or a special entry:

| OutputKind | Slice axes | Range policy |
|------------|-----------|--------------|
| `0d_yield` | `()` (0D) | implicit — always produced |
| `spectrum` | `(E,)` | energy auto |
| `temporal_envelope` | `(t,)` | auto; no range field |
| `spatial_distribution` | `(x, y)` | auto (manual override advanced) |
| `angular_distribution` | `(θx, θy)` | auto |
| `collimated_spectrum` | `(E, θx, θy)` | angular = target window; energy auto |
| `macroparticle_dump` | — (not a slice; MC-only, §3.6) | — |

Engines declare `supported_outputs` in this vocabulary (§4.1); the GUI enables an output
checkbox iff **at least one selected engine** supports it, and per-engine gaps surface as
absent curves in the outputs tab.

### 3.5 Interaction (`interaction.py`)

- `InteractionParameters(beam: GaussianElectronBeam, laser: LaserField, bunch: Bunch,
  target: Target, N_e: float)` — the compiled, pre-sampled (and prefiltered, §3.2)
  bundle every engine run consumes. `laser` is typed against the **`LaserField`
  protocol** (§3.3/P15), not the concrete `GaussianParaxialLaser` — engines only ever
  call its sampling API, which is what makes a future non-Gaussian laser implementation
  a zero-engine-change swap. One sampling path (`io` samples and prefilters,
  engines consume). No fields most engines ignore (P9).
- **N_e is a separate scalar** (derived from bunch charge): the bunch carries only
  *relative* weights (§3.2), and **every output is exactly linear in N_e** (no space
  charge in the model). Consequences: charge edits are an exact instant rescale
  (`QUERY_ONLY`, §5); the prefilter never touches N_e (§3.2); per-electron averages
  (multiplicity, final-state stats) condition on the kept set where appropriate.
- **`SamplingSpec(n_particles, seed, prefilter)`** drives the io sampler and is part of
  the interaction build — sampling parameters are **interaction-level, not engine
  parameters** (this resolves the old tracked task "move seed / n_mc into model-specific
  parameters": they go into the sampling group, not the engine; per-engine n_mc is
  meaningless once every engine in a run is given the same bunch). **`seed` is a first-class,
  GUI-displayed and -editable field** (§6) — reproducibility is a user-facing guarantee,
  not incidental internal state.
- **Bunch resample rule:** the bunch is re-sampled whenever `SamplingSpec` fields change
  **or** any beam/laser physical parameter changes (i.e. any `FULL_RERUN`-tier field,
  §5) — every resample draws from the current `seed`. So **"same seed" means "same seed +
  same beam/laser parameters + same n_particles ⇒ identical bunch"**, not "same bunch
  regardless of parameters." Target/charge/output edits reuse the existing bunch
  unchanged (they don't affect the sampled distribution).

### 3.6 Results (`results.py`)

- Axes are an **enum** (`Axis.ENERGY`, `Axis.TIME`, `Axis.X`, `Axis.Y`,
  `Axis.THETA_X`, `Axis.THETA_Y`) with **canonical CGS units** (erg, s, cm, rad);
  display conversion happens only in the plotting/serialization layers. The
  unit-baked-into-axis-name smell (`"E_eV"`) is gone.
- `PhasespaceSlice(axes: dict[Axis, ndarray], distr: ndarray, widths: dict[Axis,
  ndarray] | None = None)`: density over a named axis set; empty axes = 0D total yield.
  `axes` always stores plotting coordinates. An optional positive `widths` array declares
  that axis to be a histogram of cell-average densities, so integration is the weighted
  cell sum (and a one-bin histogram is well defined). An axis without widths is a smooth
  point sample and retains trapezoidal quadrature over its ordered coordinates; a one-point
  smooth integral is deliberately undefined. Producers declare this distinction instead of
  making a shared ndarray silently mean both. Projections preserve widths on retained axes
  and use the declared measure on eliminated axes. The closed set of allowed axis-groupings
  is kept and validated (the nine groupings from the old contract).
- `Results(photon_slices, electrons: Bunch | None, photons: PhotonMacroparticles | None,
  model_specific: dict)`. `PhotonMacroparticles` is a small new dataclass with per-photon
  arrays (energy, lab direction, position, time, weight, order/parent); `electrons`
  carries the *final* electron population (energy loss, deflection, final positions,
  emission time). Both are produced only by MC engines. (The old kascade `Results`
  already distinguished these two populations — `ph_*` arrays vs `eps_f`/`thx_f`/... —
  the rebuild keeps that distinction *typed*.) **Open question (§10):** whether
  `electrons` should reuse `Bunch` as-is or get its own symmetric
  `ElectronMacroparticles` type is left open — `Bunch` as specified has no per-particle
  emission-time field, and a second MC engine currently in development (a colleague's
  work, parallel to this plan) should clarify the right shape before this is locked in.
- One slice shape for all engines (density arrays); the old Sampled/Binned duck-typing
  concern disappears entirely (MC engines histogram their samples into slices).

---

## 4. Engine architecture (`gammaforge.engines`)

### 4.1 Uniform engine interface (`base.py`)

```python
class Engine(Protocol):
    name: str
    schema: Parameters              # typed parameter schema (P5)
    supported_outputs: tuple[OutputKind, ...]   # declarative data (P10), §3.4 vocabulary
    recompute_costs: dict[str, RecomputeCost]        # field key -> cost tier (P4/P10)

    def run(self, interaction: InteractionParameters,
            params: Parameters) -> Results: ...
```

- No `Job.extra`, no `model_params()` triples, no mutable adapter knobs (P11).
- "Which outputs can this engine produce" = declarative `supported_outputs` **in the
  `OutputKind` vocabulary of §3.4** (not raw axis-groupings — `collimated_spectrum` is a
  windowed (E,θx,θy) slice and `macroparticle_dump` is not a slice at all). Used by the
  GUI to enable/disable output checkboxes. `recompute_costs` drives grey-out/release and
  cheap requery (§5).
- Engines omit unsupported outputs from their results; the GUI renders what came back.
- A small registry (*ENGINES*) with lazy optional-dependency registration is anticipated —
  xigma needs cupy/numba, everything else is pure — but still not built as of Phase 3a:
  `run_suite`/`XigmaEngine` are both usable without one, and a registry holding exactly one
  entry is the speculative machinery P10 rejects (RES018). Reconsider once a second engine
  or the GUI needs to enumerate them.

### 4.2 xigma engine (`engines/xigma/`) — first-class

**Script alpha update (RES062):** Stage 0/1 remain NumPy. Stage 2 now dispatches
head-on, linearly polarized angular queries to the ported CuPy ring/annulus sampler
when explicitly selected and CUDA is available. NumPy is the alpha default: the
GPU port remains experimental because its density/integral comparison has not closed
(see `docs/ALPHA_GPU_VALIDATION.md`). Other geometries use the NumPy quadrature;
explicit unsupported CuPy requests raise. The alpha exception to the v0.18 target
below keeps deterministic CPU quadrature in production until a validated CPU sampler
exists. Numerical kernel comparison does not close independent emission validation.
`Collision` fixes its public inputs at construction (RES064); cached intermediates
are per-instance, with no new cross-run cache.

The tabulated-overlap pipeline, restructured into composable stages:

- **Stage 0 — trajectory integration** (`stages.py::integrate_trajectories`): pure
  function; per-particle ballistic trajectory over the laser interaction window, with
  the laser's `LaserField` sampling API (§3.3/P15) evaluated along each trajectory —
  **the engine calls the laser; it does not define the a0 profile or envelope itself**.
  Produces per-particle `L` (weight contribution), trajectory-averaged a0, temporal/
  spatial diagnostics. Backends: numpy / cupy / numba. **Chunking: one shared
  auto-chunk + OOM-retry utility** (consuming `available_vram_bytes`/
  `available_ram_bytes`) used by *every* chunked stage — the old repo had **three
  inconsistent implementations** (particles.py, spectrum_from_particles.py, and the
  dead `build_table_streaming`); port the *algorithm and the hard-won constants* (the
  `_MAX_S_CHUNK` cap history, halve-and-retry policy), not the triplicated code, and
  retire the unwired manual-chunk design.
- **Stage 1 — shape deposition** (`stages.py::deposit_shape_table`): pure function;
  nearest/CIC deposition of `(gamma, θx, θy, a0_shape)` into a 4D density table
  (`ShapeTable.H`) — onto Stage 0's a0-independent `a0_shape`, not onto `ahat` directly, so
  one deposit serves any peak a0.
- **Retarget — conservative regrid** (`stages.py::retarget_ahat`): pure function; turns a
  `ShapeTable` plus one peak a0 into the `Table` (`(gamma, θx, θy, ahat)`, Stage 2's actual
  input) via overlap-weighted mass transfer onto a **fixed, non-uniform** `ahat` axis —
  dense near `ahat_max`, coarse toward `ahat_min`, everything below folded into one floor
  bin — because the redshift correction `ahat` drives is only significant near a pulse's
  peak, and a grid that just follows wherever the data spans gives no more resolution there
  than anywhere else. This *is* the predecessor's `retarget_a0`/`a0_kind` regrid, with a
  different target-grid law than its plain `linspace` (`DECISIONS.md` RES032, superseding
  RES028's decision not to port it). Cheap and independent of `n_particles`, so `Collision`
  caches the shape deposit once and retargets many peak-a0 values from it.
- **Stage 2 — spectrum queries** (`stages.py::spectrum_from_table`,
  `angular_spectrum_from_table`, `spectrum_in_angular_range`): pure functions. The numpy
  path ports the predecessor's brute-force grid quadrature (its validation-only
  `reference.py`), not its GPU importance sampler — trust-level C in the predecessor's own
  audit, not something to import as this phase's only implementation (RES029). `cupy`/
  `numba` are gated like Stage 0's until real kernels exist. **Target shape (v0.18,
  unscheduled):** the eventual production kernel is a ring/annulus-based importance
  sampler, built and cross-checked against this brute-force quadrature rather than ported
  from the predecessor's own audited-noisy one — on **every backend it ships for,
  including CPU** (not a GPU-exclusive path), same backend set as Stage 0/1. The
  brute-force quadrature then becomes the validation-only reference, same role
  `reference.py` already had for the predecessor. **One authoritative
  normalization, isolated in one module-level location**
  (`stages.KERNEL_NORMALIZATION_CONSTANT`) and arbitrated against delta (§9.1) — the
  constant itself is pi-free and unchanged from the predecessor's kernel math; the ~2π
  question is which side of the table-free/table-based split the missing factor belongs
  to, still open (RES029, §9.1).
- **`Collision` facade** (`collision.py`): the one stateful object. Owns one fixed
  (`InteractionParameters`, xigma `Parameters`) pair and memoizes what its stages produce
  from them. Methods are thin wrappers: `build_overlap()`, `spectrum(s)`,
  `angular_spectrum(...)`, `spectrum_in_angular_range(...)`,
  `run(output_requirements) -> Results`. Caching is **per-instance memoization**, not
  cross-call hash-keyed staleness detection — the latter needs a live consumer that keeps
  one `Collision` across edits and decides what to keep, which is Phase 6's GUI grey-out
  mechanism, not built here (RES030). Notebooks use the facade; validation uses the pure
  functions directly.
- **`Engine` wrapper** (`engine.py`): the thin `run()` used by the GUI — builds one
  `Collision` per call, returns `Results`. Opaque by contract (P3).
  `recompute_costs` declares only bunch charge (`n_e`) as cheap — handled at the `io`
  level (`InteractionParameters.with_charge`/`Results.scaled`), no engine run at all —
  since the collimation-window/pulse-energy cheap paths §5 illustrates need a caller that
  reuses one `Collision`, which nothing does yet (RES030). `XigmaEngine` is not passed to
  `validation.run.main()`'s default `run_suite()` call: the scenario bank's default output
  resolution costs tens of seconds per slice against this phase's numpy kernel — real
  Calculate cost (§12), not a routine-suite cost (RES031). The validation identity harness
  exercises Stage 1/2 at a suite-appropriate scale instead (§9.1).
- **Geometry note:** `theta_xz`/`theta_yz`/`psi_focus`/`psi_pol` are first-class schema
  parameters from day one, but the *physics* of non-head-on geometry is wired as an
  identity/no-op until §9.3's derivation lands (P14c) — never silently approximate.

### 4.3 analytical engine (`engines/analytical/`) — first-class

Closed-form estimates, no per-particle Monte Carlo:

- `overlap_yield(beam, laser, N_e, n_quad)`: total yield from the **general** Gaussian
  luminosity overlap integral — non-round beams, per-axis focusing, displaced/astigmatic
  foci, rotated laser ellipse. One longitudinal quadrature; everything else is analytic
  (DER001). This is what the engine uses. `N_e` is explicit
  (`InteractionParameters.N_e`) rather than derived from `beam`, so the io-level cheap
  charge-only rescale path (§5) stays correct. Handles a **crossing angle** too (DER001
  §A.6) — for the yield; the spectrum's shape stays head-on pending §9.3, and the engine
  says so. Handles a **flying focus** too (DER002) — on a 2D `(z, ct)` grid, since a
  time-dependent spot size is exactly what forbids doing the time integral first.
- `estimate_yield(beam, laser, N_e)`: the predecessor's round-beam closed form. Retained
  as the analytic limit `overlap_yield` reduces to (a regression anchor) and as the
  port-fidelity pin — **not** for use: it carries a laser-divergence convention error
  worth 3.3x on the baseline scenario (RES040).
- `overlap_mean_a0_sq(beam, laser, n_quad, n_quad_u)`: the luminosity-weighted `<a0²>` —
  the a0 the bunch actually samples, not the pulse's peak. Same overlap integral with the
  laser density squared (§A.8). Feeds the width breakdown's nonlinearity term.
- `overlap_time_profile(...)` / `overlap_transverse_profile(...)`: the same integral
  resolved in time and across the transverse plane — cheap preview plots (~8 ms / ~100 ms)
  drawn *before* an expensive run is launched. Each integrates back to the total yield
  exactly (§7). Angle-resolved output is deliberately deferred: it needs the emission
  kernel, not the overlap geometry.
- `estimate_spectrum_width(beam, laser, theta_col, a0_sq=None)`: collimated width with a
  **per-component breakdown** — collimation `(γθ_col)⁴`, emittance/divergence
  `(γσ_θ)⁴`, energy spread `(σ_γ/γ)²`, nonlinearity `(a0²/2)²` — each reported
  separately (GUI shows a component table; total in quadrature).
- `angle_integrated_spectrum(...)`: quadrature over the beam's Gaussian energy
  distribution (no macroparticles — cost independent of n_particles; the old OOM bug
  class must not return).
- Growth items in scope: foci displacement, non-round beam total yield,
  collimated-spectrum construction (convolution of single-electron spectrum with energy
  distribution + a0). **OutputKind note:** analytical produces the 1D `spectrum` (and its
  1D collimated estimate) — it does **not** produce the 3D `collimated_spectrum` slice of
  §3.4; the GUI overlays analytical's 1D estimate on the collimated sub-tab's "1D
  spectrum on target" view.
- Role: GUI quick-estimate panel **and** validation anchor (§7).

### 4.4 kascade engine (`engines/kascade/`) — minimal port

- Retain the predecessor's sequential optical-depth inversion, polarized Thomson angle
  sampler, optional Klein–Nishina rejection, and per-emission recoil in a pure-array
  solver. Do not carry over its SI configuration model, result container, automatic file
  output, or concrete-laser coupling (RES059).
- `KascadeEngine` owns the uniform `Engine` boundary. It converts shared CGS arrays to
  the solver's SI inputs once and **reconstitutes absolute weights at its boundary**
  (relative weight × N_e, §3.2/§3.5), then returns canonical-CGS `Results`.
- Purpose: one of the ≥4 cross-validation methods. Not first-class; not polished; visible
  but off by default in the GUI. Its typed numerical schema is rendered through the same
  engine-generic path as xigma.
- **Minimum sanity bar before it anchors validation:** kascade has **zero dedicated
  tests** in the old repo; it needs at least a closed-form check (e.g. Thomson-limit
  total yield) before the 4-method comparison may treat it as an independent leg. **Met:**
  the on-axis Thomson photon-column check and scenario-bank analytical comparison pass.
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
- **Scope of independence:** delta reuses xigma's Stage 0 (trajectory integration, laser-
  field sampling), so it arbitrates **Stage-2 kernel normalization only** — it cannot
  catch a bug living in Stage 0 or in the shared `io` laser-sampling code it shares with
  xigma. Treat it as an independent check on the kernel, not on the full pipeline.

---

## 5. Recompute costs and caching (engine-declared, engine-generic)

The old-plan global `ParamGroup` enum tied to xigma's stages was rejected: the future
single-stage MC would not fit it. The mechanism is now **generic + engine-declared**:

- **`RecomputeCost`** is a small generic enum: `QUERY_ONLY` (varying the field re-runs
  only the query stage against cached results — no engine stage re-runs), `REUSE_INTERMEDIATES`
  (varying the field reuses some cached engine intermediates — the engine decides which,
  via its internal cache keys), `FULL_RERUN` (varying the field invalidates everything).
  **Tiers describe how cheap a re-Calculate is — nothing is real-time (see the GUI
  interaction model below).**
- **Each engine declares** `recompute_costs: dict[field-key, RecomputeCost]` (P4/P10).
- **Engine-side cache keys** are derived from each engine's own dataflow: every cached
  intermediate (xigma: Stage 0 output, Stage 1 table; future MC: its single stage)
  records the hash of exactly the inputs it consumed; reuse happens iff hashes match.
  Invalidation is **exact by construction** — no heuristic staleness.

**Illustrative xigma mapping** (engine-declared data, not a global table; **as of Phase
3a, only the bunch-charge row is actually wired** — `XigmaEngine.recompute_costs`
declares `n_e` alone, because it is handled at the `io` level without an engine run at
all regardless of which engine is active. The rest describe what would be cheap *if* a
caller kept one `Collision` alive across edits, which is Phase 6's grey-out mechanism,
not Phase 3a's — `DECISIONS.md` RES030):

| Field group (examples) | Cost tier | Why |
|------------------------|-----------|-----|
| beam sizes, emittances, Twiss, drift | FULL_RERUN | feeds Stage 0 + Stage 1 |
| γ0, energy spread, chirp, dispersion | REUSE_INTERMEDIATES | Stage 1 re-deposition; Stage 0 hash unchanged |
| laser spot, duration, beta_ff, geometry angles | FULL_RERUN | feeds Stage 0 + Stage 1 |
| pulse energy → a0, ellipticity | REUSE_INTERMEDIATES | Stage 1 re-deposit from cached Stage 0 samples (measured cheap, RES028 — not the predecessor's a0-axis retarget) |
| bunch charge (N_e) | QUERY_ONLY | outputs are exactly linear in N_e (§3.5) — instant rescale, no stage rerun |
| collimation window | QUERY_ONLY | Stage 2 requery against cached table |
| output grid (bins per output) | QUERY_ONLY | Stage 2 requery |
| numerics (n_steps, grid bins, chunk, device) | FULL_RERUN | affects all stages |

*The `REUSE_INTERMEDIATES` tier for pulse-energy/a0 assumes a0 enters Stage 2 only through
the a0 axis (the retarget claim) — this holds even while §9.2's energy→a0 derivation is
wired as a no-op.*

**GUI interaction model (global — no engine is real-time):**
- **The analytical estimates panel is the only real-time view.** It re-evaluates
  immediately on any input change (closed-form, microseconds).
- **Every engine is Calculate-gated.** Any input change marks previously computed engine
  results *stale* (visually outdated). Pressing Calculate re-runs the checked engines;
  each engine's facade reuses its cache where hashes allow — xigma skips Stage 0/1 on
  unchanged geometry, the MC just filters particles and rebins. A target change is a
  *cheap re-Calculate*, never a live replot.
- **Grey-out/release**: after a run, fields not declared `QUERY_ONLY`/`REUSE_INTERMEDIATES`
  by the active engine grey out (locked) until "release" — generalizing the old "XIGMA
  keeps pulse energy and gamma active" rule to a mechanism every engine declares
  (`FieldSpec` carries no editable flag; this mapping is the sole source of grey-out
  state, §3.1). Editable fields still require Calculate; their re-runs are just cheap.
- **Charge is the one exception to staleness:** because every output is exactly linear in
  N_e (§3.5), a charge edit rescales the displayed results immediately — a pure display
  operation on existing `Results`, no engine run. **Applies only when charge changes
  alone**; a charge edit combined with any other edit marks results stale (Calculate).
  Every other edit marks results stale.
- **No live engine requery exists.** `spectrum_in_angular_range` is only ever invoked by
  Calculate, against the cached table. The old OOM saga's *mechanism* (auto-firing on
  keystroke) is structurally impossible; the `_MAX_LIVE_N_ENERGY_*` hardcap lesson does
  not apply (a deliberate Calculate may legitimately be slow — see §12).
- **Scans**: a scan over `QUERY_ONLY`/`REUSE_INTERMEDIATES` fields reuses cached stages;
  scans over `FULL_RERUN` fields rerun the affected stages per point. Implemented as a
  small headless helper over the facade.

---

## 6. GUI architecture (`gammaforge.gui`)

Thin, schema-driven NiceGUI application hosted by a local loopback server and viewed
in a browser. Two global tabs: Inputs and Results. A built-in splitter optionally
shows two panes with independent tab selections over the same session state.
`docs/UI_SPEC.md` is the detailed author-approved layout specification. No physics,
no engine branching (P12, enforced by the import-boundary check).

The Inputs tab has three equal-height columns: Electrons (including sampling), Laser
(pulse properties), and Geometry (all four angle inputs plus a 2D/3D sketch).
Below them are full-width rows in this order: Target and requested outputs; analytical
estimates; engine subtabs and one Calculate action for checked engines. This supersedes
the older placement of geometry angles within the Laser panel below.

Calculation execution uses a concrete request in the shared layer and a public local
runner in the engines package. The GUI imports the runner and engine interface only;
engine discovery, analytical evaluation, and calls to engine implementations live
behind that boundary. Requests carry physical inputs and typed engine parameters,
never widgets, callbacks, browser state, or file handles. Future LAN execution can
replace this local execution boundary; network transport, authentication, remote job
storage, and discovery are not implemented in Phase 6. Existing YAML/HDF5 formats are
the future transport starting point, not an invitation to send arbitrary Python objects.

**Tab 1 — Input & Calculate**
- Electrons panel, Laser panel (fields rendered from `FieldSpec`, unit dropdowns per
  field, sliders where declared). The Laser panel carries the four geometry angles of
  §2.2.
- Sampling fields (`n_particles`, `seed`, `prefilter`) — `SamplingSpec` (§3.5), shown
  alongside the Electrons panel (bunch generation is beam-driven). **`seed` is directly
  displayed and editable**, not hidden internal state. Editing any of these, or any
  beam/laser physical parameter, resamples the bunch (§3.5 bunch resample rule).
- Target panel: collimation window fields (which define `collimated_spectrum`'s angular
  ranges) + **required-output checkboxes** with per-output resolution controls (ranges
  auto-derived except the collimated window, §3.4).
- Model sub-tabs (one per available engine): each tab carries a checkbox labelled
  **"Use for calculation"** (unambiguous semantics) and the engine's typed parameters
  rendered from its schema. **One Calculate button** runs exactly the engines whose
  checkbox is on, and shows per-engine status.
- Analytical estimates panel (always visible): total yield + per-component spectrum-width
  breakdown (§4.3). **analytical is not a checkboxable engine** — its estimates panel is
  separate; however, in the outputs tab every engine's curves (analytical included) are
  individually showable/hideable.
- Geometry sketch panel (2D/3D): lab axes, electron ellipsoid, laser k0 direction,
  focusing ellipse, polarization ellipse, ghost foci at time delays (see §2.2). The same
  drawing module is available headless.
- Grey-out/release behavior driven by the engine's declared recompute costs (§5); any
  input edit (except charge, which rescales instantly) marks engine results **stale**
  until Calculate (§5 interaction model).
- Calculate runs the checked engines **sequentially in a single worker thread** — engines
  do heavy compute, so running them concurrently would only contend for the same CPU/GPU
  — with **per-engine progress/status**; the analytical panel and the sketch stay
  responsive throughout.

**Tab 2 — Outputs**
- Sub-tabs per requested output. The collimated-spectrum output is visualized as the
  §3.4 pipeline: 2D slices at θx = 0 / θy = 0, 2D energy–angle sums, and the 1D
  spectrum on target.
- Line plots: all selected engines' curves drawn; **per-engine show/hide toggles**
  (analytical included); distinct colors.
- 2D colorplots: a picker list of the available 2D outputs (spatial, angular,
  spectral-angular).
- Plot zoom (best-effort) instead of manual range entry (§3.4).
- MC macroparticle output (when produced): statistics + "save particles" button.
- Save plots (PNG/PDF) and Save results (HDF5, §8) buttons.

Layout and first-version interaction details are specified in `docs/UI_SPEC.md`.
Long-running engine work stays on a worker thread; sampling runs there too. Per-engine
status is available at the existing opaque run boundary, without invented percentage
progress or cancellation guarantees. Browser sessions own independent input/results
state. Plot projections and the geometry sketch remain available headlessly.

---

## 7. Validation strategy (rebuilt from scratch)

The script-first alpha has a restricted validation selector covering Gaussian
analytical/xigma total yields and their common head-on weak-field spectrum regime.
It runs the existing numerical gates unchanged and explicitly lists out-of-scope
coverage. Full production validation keeps independent angular-emission and
four-method coverage as blockers. GUI and kascade are outside alpha support.

Guiding principle: **tests assert physics, not implementation**. No "detect if an
assumption was broken" test zoo.

- **Closed-form limits** (analytical as anchor):
  - Thomson limit: zero-a0 yield ≈ N_e · σ_T · (overlap) closed form.
  - Head-on geometry: xigma reproduces the analytic single-electron spectrum shape at
    the Compton edge; edge location and width vs theory.
  - Total yield: `∫ angular spectrum = ∫ spectrum = total_yield` — exact identities,
    not tolerances, where the contract guarantees them.
  - Convergence: results converge with `n_steps`, `n_bins`.
  - **Invariance properties:** results identical under (a) chunk size (regression guard
    on the old OOM fixes; exercises the single shared chunking utility of §4.2),
    (b) the **prefilter on/off** (same seed — it is a pure optimization, §3.2),
    (c) **backend** (numpy / cupy / numba) to a tight relative tolerance (~1e-6) — **not**
    bit-identical (GPU float32 vs. CPU float64 preclude exact equality); and **same seed
    → identical results** (seed determinism, with beam/laser parameters held fixed too —
    see the bunch resample rule, §3.5).
- **Cross-engine consistency** (the ≥4 methods): xigma vs delta vs analytical vs kascade
  on shared `Scenario`s (baseline, scaled pulse energy, ...), tolerance-based. MC legs are
  compared with **statistical tolerance** (fixed-seed runs or error bars), not tight
  absolute bounds. kascade only counts once its own Thomson-limit sanity check passes
  (§4.4).
  - **Two structural blind spots, both demonstrated rather than hypothetical** (found by
    RES053's factor-of-2 in `ahat`, which the entire suite passed through unnoticed). A
    cross-check is only as independent as its *inputs* and as sensitive as its *observable*:
    1. **Shared inputs are common-mode.** xigma and delta both read `ahat` from the same
       `TrajectorySamples`, so an error in it cancels in every xigma-vs-delta comparison.
       Legs that share a stage do not cross-validate that stage — they cross-validate what
       comes after it. delta is an independent *kernel*, not an independent *Stage 0*, and
       the leg that finally exposed RES053 was analytical, which computes `ahat` from its
       own overlap integral. When adding a leg, state explicitly which upstream quantities
       it shares.
    2. **Integrated observables hide redistribution.** The nonlinear red-shift moves photons
       along `s` while conserving their number, so any check that compares *counts* is blind
       to it by construction: scaling `ahat` over an 8× range (×0.5 to ×4) moves `run.py`'s
       fourth identity leg by 0.11% in total (0.9987–0.9998, measured on all three
       scenarios) while moving the spectrum's centroid by many percent — the leg is not
       merely insensitive to a factor of two, it is blind to the whole quantity.
       Each physical effect needs a check on an observable that
       effect actually moves — for the red-shift, the spectrum's centroid or edge position,
       not its integral.
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

For the script-first alpha, HDF5 results are versioned and preserve engine warnings
and numerical metadata. An optional embedded YAML calculation request records Gaussian
beam/laser inputs, target, sampling seed and engine settings; loading uses explicitly
supplied engine schemas, never dynamic imports. Metadata supports plain values,
arrays and dataclass field mappings (dataclasses load as mappings). Unknown metadata
objects fail before writing, rather than silently disappearing. Legacy files remain
readable with unknown provenance. Existing Bunch result payloads are preserved without
settling the future final-electron class. GUI run export and kascade validation remain
outside this alpha's supported scope.

- **Results saving — HDF5** (h5py): slices (density arrays + axis values) + a YAML
  sidecar for parameters. Round-trip tested.
- **Macroparticle dump** (MC engines): HDF5 **and** export to elegant-compatible format
  (`.ele` or `.bun` — exact format to be researched during implementation; the old repo
  already has a hand-rolled SDDS ASCII `.ele` reader/writer to build on).
- **Beam/laser YAML** (evolved v0.2 specs): explicit units at the file boundary,
  converted to CGS on load; the schema is the single source of truth for field sets.
  Round-trip tested.
- **`.ele`/SDDS load** (elegant-format 6D distributions): kept; inherently SI/GeV —
  converted at the boundary (P1). On load, weights are normalized to the relative
  convention (`1/n`); N_e comes from the separately-entered charge field (Bunch charge
  override is explicit — the old no-charge-in-.ele caveat stays documented).
- **Graphs**: matplotlib PNG/PDF via a plotting module shared by GUI and headless use.

---

## 9. Physics work items

### 9.1 ~2π kernel normalization — **CLOSED (traced Phase 2.5, applied Phase 3b)**

**Resolved as an investigation.** The factor is exactly `2π`, it is in the **paper**, and
it is not a porting artefact. `eq:xsec` is missing `1/(2π)`, and `eq:main`/`eq:Fmatrix`
inherit it — one error, not two, since the second is the first times the `eq:jacobian`
Jacobian. Full derivation and evidence: `DECISIONS.md` **RES026**; the manuscript is
annotated at both equations.

- The check is two lines. `eq:collision` yields a photon count only if the frequency- and
  angle-integrated cross-section is `σ_T`; from `eq:xsec` with `R → δ` it is `2π σ_T`.
  Both `u`-integrals are elementary (`1` and `1/6`).
- **Which side counts photons is settled.** Stage 0's total is `flux × cross-section ×
  time`, and the closed-form single-electron spectrum reproduces it as an identity. Two
  independent methods agree; the paper's differential form is the outlier, carrying an
  extra azimuthal `2π`.
- Old-repo status, now explained: it absorbed the same factor behind two self-consistent
  rescales (both labelled *QUICK FIX*), and its headline `total_yield` came from the
  luminosity sum — a path that never touches the kernel. That is why only *delta* ever
  showed it.
- **Applied in Phase 3b (`DECISIONS.md` RES033).** Stage 2's constant is `1.5/(2π)` and
  delta's prefactor is `3/(2π)` — one correction at the two places this repo transcribes
  eq. `xsec`. The identity harness's delta leg is gated at **1**, not at a derived `2π`,
  which supersedes RES025's "report, don't correct" stance: the factor is no longer
  unexplained, so reporting it no longer tells anyone anything.
- **Order, because P14 turns on it:** the constant came from the derivation above, and was
  *then* confirmed against an absolute measurement — the table kernel's own angle-integrated
  photon count vs Stage 0's elementary total, which read `2π × 0.997` before the correction.
  Every other Stage-2 check is a ratio between two paths carrying the same factor and is
  blind to this by construction, so that measurement had to be built (`tests/test_stage1_stage2.py`).
- **The one authoring choice left changes no number.** Where the `1/(2π)` belongs in the
  manuscript — the prefactor, the normalization of `R`, or the definition of `Û` — is still
  the author's; all three placements fix the same `d³N/(dω d²Ω)`, which is why the code did
  not wait on it. The manuscript stays annotated rather than edited (RES026), so the repo
  knowingly computes corrected physics against an uncorrected typeset equation.

### 9.2 Laser energy→a0 with ellipticity (net-new derivation, parallel track)

- Paper status: **no formula exists.** `a0` is a given input in the paper; its
  polarization object is a normalized 2×2 coherence matrix with `Tr Ξ̂ ≡ 1` — there is
  no scalar "ellipticity" and no `(1+ε²)/2` identity in the source. Whether a scalar
  `ellipticity` schema parameter maps cleanly onto `Ξ̂` is an open modeling question for
  the author.
- Rebuild: carry `ellipticity`, `psi_pol`, focusing axes as first-class schema parameters
  now; wire the energy→a0 derivation as an **explicit no-op/identity** (documented)
  until the derivation lands. Never gating engineering milestones (P14c).
- **Narrowed as of RES053 and resolved as of DER004/RES054.** DER004 §1.1 showed that
  the energy→a0 relation and `ahat` are invariant with respect to ellipticity at fixed
  pulse energy (the cycle average factors cancel). Ellipticity enters in the
  emission kernel's polarization factor (DER004 §1.2), where
  `(cos^2 psi + eps^2 sin^2 psi) / (1 + eps^2)` is implemented across `stages.py`,
  `delta.py`, and `collision.py`. `ELLIPTICITY_IS_NOOP` is `False`.

### 9.3 Crossing angle and electron direction

The manuscript's polarization projection is defined with the field-free electron
velocity. The author confirmed on 2026-09-06 that Stage 2 uses that velocity directly
in the common lab frame:
`v_e = β (θ_x, θ_y, 1) / √(1 + θ_x² + θ_y²)`.
The observer direction and laser basis remain lab-frame vectors; the engine does not
rotate them into a different frame for each particle. The polarization projection
therefore uses each table cell's electron direction (RES060).
The emission kernel includes the relative velocity factor, resonance frequency factor
`cos^2(alpha/2)`, and polarization projection (DER005, DER006); `EMISSION_IS_HEAD_ON` is `False`.

This settles the coordinate convention for that projection only. Independent
arbitrary-angle emission validation remains open, so validation reporting must not
present crossing-angle observables as a scientific pass merely because the lab-frame
formula is implemented.

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
4. GUI visual design details (panel layout, sketch panel content) — later. **Calculate
   semantics settled:** "Use for calculation" checkbox per model sub-tab; one Calculate
   button runs the checked engines (§6).
5. MC macroparticle dump format (elegant `.ele`/`.bun` research, see 1).
6. Crossing-angle formula details (§9.3) — with the author, parallel track.
7. **RESOLVED (v0.11):** there is no k0_las normalization — xigma works in CGS directly.
   Superseded the earlier "stays purely internal" answer, which satisfied the no-leakage
   requirement but kept bookkeeping that buys nothing (§4.2, `DECISIONS.md` RES015).
8. **Final-electron macroparticle typing** (§3.6): whether `Results.electrons` reuses
   `Bunch` as-is or gets its own `ElectronMacroparticles` type (symmetric with
   `PhotonMacroparticles`; `Bunch` as specified has no emission-time field) is open —
   deferred until a second MC engine, currently in development by a colleague in
   parallel, clarifies the right shape.

---

## 11. Phases and milestones

| Phase | Scope | Exit criteria |
|-------|-------|---------------|
| **0. Scaffold** | Repo, git, pyproject (Python 3.12), pytest, package skeleton, README, ADR index + **`docs/decisions/` provenance system started** (C1); `.gitignore` **explicitly covers large data formats (`.ele`, notebooks with large outputs) and sync-conflict patterns from day one** (C3) | `pytest` green on an empty-suite smoke test; `pip install -e .` works; doc-staleness guard scaffolding in place (C2) — a CI smoke test asserting **every backticked identifier in the docs resolves in the package** |
| **1. Core** | `io/`: schema, units/conventions (CGS), constants; bunch (`Bunch` + `GaussianElectronBeam`); laser (incl. `LaserField` protocol + `GaussianParaxialLaser` as its sole implementation, `fit_gaussian_paraxial`, elliptical+astigmatic model, geometry angles — §3.3/P15); target (auto-ranges + `OutputKind` vocabulary §3.4); interaction (incl. `N_e` scalar + `SamplingSpec` §3.5); sampling + prefilter (§3.2); results contract (incl. `PhotonMacroparticles` §3.6); YAML + `.ele` I/O; HDF5 results writer | Round-trip tests; schema validation tests; CGS conversion tests vs known values; `LaserField` protocol conformance test for `GaussianParaxialLaser` (period-averaged + period-resolved at arbitrary points); `fit_gaussian_paraxial` identity test on a `GaussianParaxialLaser` input; geometry round-trip test (R⁻¹ recovers head-on angles, §2.2) |
| **2. Validation harness** | scenarios, runners skeleton, `make_references.py` + first golden snapshots from old repo; invariance-test scaffolding (chunk, prefilter, backend, seed — §7) | Golden generation runs; new-vs-golden comparisons execute |
| **2.5. Stage 0 + minimal delta** | **Stage 0** (`integrate_trajectories`) and the **shared auto-chunk + OOM-retry utility** (§4.2), pulled forward from 3a because delta needs both; delta itself scoped to Stage-2 normalization arbitration, built on top of Stage 0 (§4.5) | Stage 0 tests green; chunk-invariance holds; delta produces independent spectra on baseline scenarios; identity harness (`kernel` vs `reference` vs `direct binning` vs delta) executable |
| **3a. xigma engineering** — **landed 2026-08-08** | Stage 1/2 pure functions; Collision facade + stage cache; Engine wrapper; numpy kernel for Stages 1/2, cupy/numba gated like Stage 0 until real kernels exist (**Stage 0 and the chunking utility already built in 2.5**; RES029); geometry/a0/ellipticity parameters wired as explicit identity/no-op placeholders (P14c) | Stage architecture tests green; placeholders documented |
| **3b. Physics closure** — **§9.1 landed 2026-08-08; §9.2/§9.3 open, non-blocking** | ~2π resolution (§9.1 — **closed**: traced in 2.5, applied in 3b, RES033), crossing-angle derivation (§9.3), ellipticity→a0 (§9.2) — **runs concurrently with Phases 4 and 5, not serially** | §9.1's constant set in Stage 2 and the identity harness re-gated against 1.0 rather than 2π — **met**; §9.2/§9.3 derivations landed if author completes them in parallel (never blocking 4–6) — **outstanding, and the paper contains no formula for either**, so both stay wired as documented no-ops with `validate()` warnings (RES034) |
| **4. analytical engine** — **landed 2026-08-09; one growth item open** | estimates + component breakdown; quadrature spectrum; general overlap-integral yield (non-round + displaced foci); remaining growth item (collimated spectrum) | Closed-form limits match — **met** (Thomson-limit anchor *and* the analytic reduction of `overlap_yield` to the round-beam closed form, §7); validation anchor ready — **met** for `TOTAL_YIELD`/`SPECTRUM`; foci displacement + non-round beam — **met** via DER001 (RES039); crossing-angle geometry — **met** for the yield via DER001 §A.6 (RES041), validated against a brute-force Monte Carlo; the width's nonlinearity term — **met** via the luminosity-weighted `<a0²>` (DER001 §A.8, RES042); resolved time/transverse previews — **met** (DER001 §A.9). Outstanding: collimated-spectrum construction, angle-resolved previews (deferred), and `SPECTRUM`'s shape under a crossing angle, which is §9.3's emission kernel rather than overlap geometry |
| **5. kascade port + delta full role** — **partially landed 2026-09-06** | minimal kascade behind interface **+ its Thomson-limit sanity check (B4)** — **met** (RES059); delta full cross-validation role — **open** | Kascade sanity check passes — **met**; 4-method cross-validation runs — **open** |
| **6. GUI** | schema-driven two-tab app; overlays + per-engine show/hide; save plots/HDF5; grey-out/release; sketch panel (headless module first); **import-boundary check enforced in CI (B3)** | GUI runs headless-smoke; all planned interactions work; boundary check green |
| **7. Validation completion** | full scenario bank, convergence, chunk-invariance, closed-form identities, golden cross-checks | Full suite green; results reproducible; 3b closures integrated |
| **8. Polish** | scans, docs, packaging, notebook examples, **doc-staleness sweep (C2)** | Release-ready |

Order note: Phase 3b is explicitly parallel; Phases 4–6 must not wait on physics
derivations (A3). **Before Phase 3a/6 kickoff, re-verify the old repo's remote
`worktree-*` branches are merged** (audited 2026-08-06: all eight — 4 local worktree
branches plus 4 remotes — are merged into `master`; a one-line `git branch -a` check
suffices) so no half-finished work is
duplicated (C4).

---

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| ~2π normalization turns out to be a paper-level issue | **It did** (Phase 2.5, RES026): `eq:xsec` is missing `1/(2π)`. The mitigation worked as designed — delta arbitrated independently, the paper formula was treated as necessary-not-sufficient, and the constant was isolated to one location, so applying it in Phase 3b was the one-line change it was meant to be (RES033). Residual: the manuscript is annotated, not corrected, so the repo and the typeset equation knowingly differ |
| Crossing-angle and ellipticity→a0 have **no existing derivation** in the paper (confirmed by audit, not just undocumented) | Open-ended research tasks, not consult-and-implement: parameters are first-class in schema/architecture now; physics wired as explicit identity/no-op until derivations land; derivation runs in parallel (P14c, §9.2/§9.3); never blocks Phases 3a–6 |
| Old-repo golden data encodes bugs | Goldens are transitional; closed-form identities + delta reference are the real anchors; goldens regenerated deliberately |
| Chunking regressions (OOM class) | Chunk-invariance property tests from Phase 2 on; single shared auto-chunk + OOM-retry utility (porting algorithm + constants, not the old triplicated code) |
| Collimated 3D-slice cost: a deliberate Calculate with the collimated (E,θx,θy) output is inherently slow at high resolution (measured 27 s @ 64 energy bins on CPU in the old repo; linear in n_energy) | **Expected, not a defect** — no live auto-requery exists (engines are Calculate-gated, §5), so the old CPU-pegging mechanism is structurally impossible; the analytical panel stays real-time; per-engine progress indication; cache reuse (`QUERY_ONLY`/`REUSE_INTERMEDIATES`) minimizes repeated cost |
| pint friction with CGS | pint confined to schema/serialization; kernels never see it; EM conversions hand-coded with tests vs known values |
| GUI regrows into a monolith (already happened once: 1174 → 1685 lines) | Import-boundary check in CI (P12/B3): `gammaforge.gui` may not import engine stages/kernels — mechanical enforcement, not discipline |
| Capability data regrows into a registry/protocol | P10 explicit guardrail: plain tuples/dicts on the Engine instance; any registry/negotiation layer is the old `ModelCapabilities` mistake recurring |
| Docs drift out of sync with code (old repo's AGENTS.md referenced deleted types) | Doc-staleness guard from Phase 0 (C2): a CI smoke test asserting every backticked identifier in the docs resolves in the package |
| Scope creep (GUI polish, scans, sketches) | Explicitly deferred/late-phase; architecture supports them but they don't block physics milestones |

---

## Changelog

- **v0.30**: Author-scoped script alpha (`0.1.0a1`) covers Gaussian analytical/xigma
  calculations and a reproducible crossing-angle yield figure, excluding GUI/kascade
  support. CuPy angular sampling is integrated experimentally for head-on linear
  polarization; NumPy remains the default after GPU/reference disagreement (RES062).
  Shared input snapshots reject invalid construction and
  isolate caller arrays (RES064). Versioned HDF5 preserves metadata and an optional
  replayable submitted request (RES063). A restricted alpha gate retains numerical
  failures while reporting deferred full-production coverage (RES065).

- **v0.29**: Slice integration now carries an explicit per-axis histogram measure.
  Histogram cell densities use their stored widths, while smooth point samples retain
  trapezoidal quadrature. The distinction survives projection, charge scaling, and HDF5;
  legacy HDF5 slices without widths remain point samples (RES061).
- **v0.28**: The author resolved the pending Stage-2 electron-direction convention:
  polarization uses the field-free, per-particle lab-frame velocity
  `β (θ_x, θ_y, 1) / √(1 + θ_x² + θ_y²)`. The observer direction and laser
  polarization basis remain in the shared lab frame; there are no per-particle
  coordinate rotations. This corrects only the polarization projection in §9.3;
  independent arbitrary-angle emission validation remains open.
- **v0.27**: Phase 5's minimal kascade port retains the independently useful
  sequential emission chain in a pure-array solver rather than copying the predecessor's
  configuration, result, and automatic-file-output framework. `KascadeEngine` owns the
  checked CGS↔SI boundary, consumes `LaserField`, and exposes its typed schema through the
  public runner. It is visible but off by default in the NiceGUI engine tabs. The
  Thomson-limit normalization anchor is met; four-method validation wiring and the
  final-electron export format remain Phase 5 work (RES059).
- **v0.26**: author-directed NiceGUI browser frontend replaces the planned Tkinter
  frontend. A local loopback server hosts Inputs and Results tabs, with an optional
  resizable split view. Inputs use equal-height Electron/Laser/Geometry columns,
  followed by Target/Outputs, analytical estimates, and engine settings rows.
  `docs/UI_SPEC.md` specifies layout and acceptance criteria. A concrete calculation
  request and local runner separate execution from widgets, preserving a future LAN
  execution boundary without implementing networking. The GUI may import this public
  runner, but never engine implementations or kernels. Existing recompute declarations
  remain honest: cross-run xigma stage reuse is a separate unfinished task, not a GUI
  feature implicitly promised by adopting NiceGUI.
- **v0.25**: `ahat` **corrected** — the code was short the polarization cycle average and
  every `ahat` was twice the paper's, overstating the nonlinear red-shift by 2x
  (`DECISIONS.md` RES053). Raised as a §0 BLOCKING code/paper discrepancy and settled by the
  author: `a0` is the normalized **peak** field magnitude, so `<a²> = C a0²` with `C = 1/2`
  linear, `1` circular, and `a0_profile` is the linear peak envelope. §7 gains the two
  structural blind spots this exposed (shared inputs are common-mode; integrated
  observables hide redistribution), and §9.2 gains a concrete hook: `C` is what
  `ellipticity` interpolates. *(v0.19–v0.24 are on the parallel Phase 4 branch
  `worktree-phase4-analytical-engine`; this entry deliberately numbers past them so the
  changelogs interleave rather than collide on merge.)*
- **v0.24**: **arbitrary transverse and timing misalignment** between pulse and bunch
  (`DECISIONS.md` RES046, `docs/DERIVATIONS.md` §A.11). `GaussianParaxialLaser` gains
  `x_off`/`y_off`/`t_off`, applied once in `_local_coordinates` so every field consumer —
  xigma included — inherits them, with `active_region` shifted to match so the cone
  prefilter cannot discard particles that do interact. Deliberately no `z_off`: for a pulse
  at `c` it is degenerate with `t_off`. Analytically this is one linear term in the
  quadratic form, verified by an exact `exp(-dᵀ(C_e+C_l)⁻¹d/2)` falloff to 1e-13 and by the
  Monte Carlo across every combination with crossing angle and flying focus. Closes the last
  gap in "arbitrary foci displacement": longitudinal was already general per axis, the
  transverse and temporal directions had no representation at all. `PROGRESS.md` 2026-08-09.
- **v0.23**: **flying focus covered for the analytical yield** (`docs/DERIVATIONS.md` §B,
  `DECISIONS.md` RES044) — `overlap_yield` no longer refuses `beta_ff`. The useful structural
  result: a flying focus makes the widths depend on *two* linear functionals of
  `(x, y, z, ct)` instead of one, so two of four dimensions stay Gaussian and **an exact
  treatment of a crossing angle plus an arbitrary flying-focus velocity together is still
  only a 2D quadrature** — the two effects do not compound. A 1D shortcut exists and is
  deliberately not shipped: its error is *first* order (34% at `beta_ff = 1` on the
  baseline) because a flying focus exists to correlate the width with time. Validated
  against the brute-force Monte Carlo to 1e-3 across `beta_ff` ∈ {−0.5, 0.5, 1, 2}, with
  and without a crossing angle; **not** cross-checked against the author's own head-on
  synchronized derivation, which was not available. Two physics results fall out and are
  pinned: `beta_ff = 1` maximizes the yield (2.8× on a 30 µm bunch), and for a short bunch
  the yield is invariant under `beta_ff → 1/beta_ff`. `PROGRESS.md` 2026-08-09.
- **v0.22**: analytical becomes explicitly **tiered** (`DECISIONS.md` RES043) — closed form
  (~0.01 ms) / 1D quadrature (~1–2 ms, default) / exact 2D quadrature (~40–800 ms, opt-in
  via `n_quad_u`). The first two keep §4.3's "only real-time engine" claim true of
  something; the third makes the 1D path's error *measured* (1.9e-4 at 20 mrad, 1.6e-3 at
  0.4 rad on the worst corner) instead of argued. `recompute_costs` now declares the
  quadrature knobs, before Phase 6 can wire a live panel to the slow tier.
  **The last growth item for the width closes**: `overlap_mean_a0_sq` computes the
  luminosity-weighted `<a0²>` — the same integral with the laser density squared — and the
  nonlinearity term uses it instead of peak a0, a ~3x correction at the baseline (RES042).
  New cheap previews `overlap_time_profile`/`overlap_transverse_profile` resolve the same
  integral in time and across the transverse plane, both satisfying exact
  integrate-back-to-the-yield identities (§7). `docs/DERIVATIONS.md` §A rewritten in
  MathJax for the paper. `PROGRESS.md` 2026-08-09.
- **v0.21**: **The crossing angle is covered for the analytical total yield** (§4.3,
  `docs/DERIVATIONS.md` §A.6). v0.20 deferred it to §9.3, which conflated two separate
  things: §9.3's open item is the polarization structure of the *emission kernel*, while
  the *overlap geometry* is an independently solvable Gaussian problem — and §9.3's own
  notes already record that the relative-velocity factor and resonance frequency are
  general in the paper. Rewriting the overlap as a quadratic form and eliminating time via
  a Schur complement covers every geometry in one expression, with head-on falling out as
  an identity (no head-on test changed). Validated three ways: a constant-width closed form
  to ~1e-14 out to 0.4 rad in both crossing planes, the Piwinski suppression at small
  angle, and a brute-force Monte Carlo over `GaussianParaxialLaser.photon_density` with
  real macroparticles to a few 1e-4. The effect is large — 20 mrad costs a factor 2.18 in
  baseline yield. **`SPECTRUM`'s shape remains head-on** while its integral is now correct,
  so `AnalyticalEngine` reports that on `Results`. `DECISIONS.md` RES041; `PROGRESS.md`
  2026-08-09.
- **v0.20**: Two of §4.3's three "growth items" **closed** — non-round-beam yield and
  foci displacement — by deriving the Gaussian luminosity overlap integral in general
  (`docs/DERIVATIONS.md` §A) rather than approximating it. All integrations are analytic
  except one longitudinal quadrature over a strictly positive, smooth integrand, so the
  cost stays milliseconds. `overlap_yield` handles per-axis sizes and focusing, Twiss
  `alpha` (the electron waist offset — already in the data model, no new schema), the
  astigmatic `z_fx`/`z_fy` offsets and the `psi_focus` rotation between the two transverse
  ellipses; it reduces analytically to the old closed form in the round/aligned limit,
  which is now a pinned regression anchor. `AnalyticalEngine` ships it. Only
  collimated-spectrum construction remains open of the three. **The derivation also
  exposed a factor-4 laser-divergence convention error in `estimate_yield`, inherited
  from the predecessor and worth 3.3x in the baseline yield** — flagged, pinned by test,
  deliberately not "fixed" (`DECISIONS.md` RES039–RES040). `PROGRESS.md` 2026-08-09.
- **v0.19**: Phase 4 (`engines/analytical`, §4.3) **landed**, built concurrently with
  Phase 3b in an isolated worktree/branch (§11's "Phases 4–6 must not wait on physics
  derivations"). `estimate_yield`, `estimate_spectrum_width` (now returning its four
  components separately, per §4.3's own text, rather than pre-summed), and
  `angle_integrated_spectrum` are ported from the predecessor onto this repo's CGS
  beam/laser types; `AnalyticalEngine` fills `TOTAL_YIELD` and 1D `SPECTRUM` with
  `∫ SPECTRUM = TOTAL_YIELD` as an **exact** identity (§7), not a tolerance. §4.3's own
  "growth items" — foci displacement, non-round-beam total yield, collimated-spectrum
  construction — are **not** attempted in this landing and stay open (§11's Phase 4 row
  below states this explicitly rather than being marked fully closed). `DECISIONS.md`
  RES035–RES038; `PROGRESS.md` 2026-08-09.
- **v0.18**: Stage 2's long-term production path is now stated explicitly (§4.2): once a
  ring/annulus-based importance-sampling kernel is built and validated against the
  existing brute-force grid quadrature, it becomes the **sole production path** for
  `spectrum_from_table`/`angular_spectrum_from_table`, on **every backend it ships for —
  CPU included, not GPU-only** (mirrors Stage 0/1's numpy/cupy/numba backend set, §4.2).
  The current brute-force quadrature then demotes to a validation-only reference kernel,
  the role its own predecessor ancestor (*reference.py*) already had. This does not
  supersede RES029: RES029 rejected porting the predecessor's *specific, already-audited*
  sampler (trust-level C, 3x-30x variance, no GPU to validate against at the time) as
  *this phase's* only implementation — it did not reject an importance-sampling kernel in
  general once it can be built and cross-checked against the working brute-force path.
  Not scheduled against a phase yet; recorded now so the eventual work has a stated target
  shape instead of being invented ad hoc when it starts.
- **v0.17**: `OutputKind.SPECTRAL_ANGULAR_DISTRIBUTION` **removed** from the vocabulary
  (§3.4). It shared its `(E, θx, θy)` axes with `collimated_spectrum` and differed only in
  using an auto-ranged full radiation cone (~1/γ0) instead of the target's collimation
  window — but no experiment measures the untruncated cone, only the collimated emission,
  so the auto-ranged kind was never a real observable. `collimated_spectrum` is now the
  sole 3D energy-angle output kind. `DECISIONS.md` RES052.
- **v0.16**: §9.1 **closed**, and §9.3 given the marker §9.2 already had — Phase 3b's
  code-side work, which was always the only part of 3b that did not depend on a derivation
  the paper lacks. The `1/(2π)` RES026 derived is now applied at both places this repo
  transcribes the paper's differential cross-section (Stage 2's kernel constant and delta's
  prefactor), and the identity harness's delta leg is gated at **1** instead of at a derived
  `2π`, which is §11's stated exit criterion for 3b. The constant was set from the derivation
  and *then* confirmed by an absolute measurement — the table kernel's own angle-integrated
  photon count against Stage 0's elementary total, which no pre-existing test could see
  because every one of them is a ratio between two paths carrying the same factor.
  Separately, a nonzero crossing angle now warns that only the *geometry* is applied: the
  rotation is real, the emission physics is head-on, and §9.3's asymmetry with §9.2 (which
  had a marker and a warning) was itself a P14c violation. `DECISIONS.md` RES033, RES034.
  §9.2 and §9.3's derivations remain the author's, unstarted, and non-blocking.
- **v0.15**: Stage 1 restructured, within the same Phase 3a — supersedes v0.14's item (a).
  Direct deposit onto `ahat` (v0.14) is replaced by a two-step pipeline: `deposit_shape_table`
  bins the a0-independent `a0_shape` onto a fine, uniform grid; `retarget_ahat` conservatively
  regrids it onto a **fixed, non-uniform** `ahat` axis — dense near `ahat_max`, coarse toward
  `ahat_min` — for one specific peak a0. Physics motivation: the redshift correction only
  matters where `ahat` is comparable to 1 (near a pulse's peak); a grid that just follows
  wherever the sampled data spans gives no more resolution there than anywhere else. This is
  the predecessor's `retarget_a0` regrid mechanism after all, with a different target-grid
  law than its plain `linspace`, and RES028's "nothing here needs it" conclusion is superseded
  by RES032 accordingly. `Table.bin_volume` (a single scalar, valid only for a uniform grid) is
  removed; `spectrum_from_table` now folds a per-bin `ahat_widths` array into its cell sum
  instead. Full reasoning, the target-grid formula, and how its defaults were tuned against
  the actual scenario bank: `DECISIONS.md` RES032.
- **v0.14**: Phase 3a landed (Stage 1/2, `Collision`, `XigmaEngine`), four items restated
  against what was actually built rather than what §4.2/§5 anticipated.
  **(a) No `retarget_a0`/`a0_kind` rebin.** `deposit_table` computes
  `TrajectorySamples.retargeted_ahat` and re-deposits from cached Stage 0 samples instead
  — measured cheap relative to Stage 0, so the predecessor's conservative-regrid apparatus
  buys nothing here (`DECISIONS.md` RES028).
  **(b) Stage 2's numpy kernel is the predecessor's brute-force grid quadrature
  (`reference.py`'s), not its GPU importance sampler** — the sampler was the predecessor's
  own trust-level-C path (3x-30x variance in sparse configs), not something to import as
  this phase's only implementation. `cupy`/`numba` stay gated exactly like Stage 0's until
  real kernels exist (RES029).
  **(c) §5's illustrative recompute-cost row for pulse energy is walked back.** "Stage 1
  a0-axis retarget (no re-deposition)" assumed the ported mechanism (a); restated below as
  "Stage 1 re-deposit from cached Stage 0 samples". The tier (`REUSE_INTERMEDIATES`) is
  unchanged, but `XigmaEngine.recompute_costs` does not claim it yet — that needs a caller
  that keeps one `Collision` alive across edits, which is Phase 6, not 3a (RES030).
  **(d) `XigmaEngine` is not passed to `run_suite()` by `validation.run.main()`.** The
  scenario bank's default output resolution is sized for the predecessor's GPU kernel and
  costs tens of seconds per slice against this phase's numpy one — real Calculate cost
  (§12), not something a routine suite run should pay. A fourth identity-harness leg
  (Stage 2 kernel vs delta at one point, both carrying the identical pending §9.1 factor)
  exercises Stage 1/2 in the suite instead (RES031).
- **v0.13**: §9.1 rewritten — the ~2π is **traced, not open**. It is exactly `2π`, it is in
  the paper at `eq:xsec` (inherited by `eq:main`), and it is not a porting artefact; the
  derivation and evidence are in `DECISIONS.md` RES026 and the manuscript is annotated at
  both equations. What remains is an authoring choice about where the factor belongs, not
  a computation, so the §11 exit criterion for Phase 3b and the §12 risk row are restated
  accordingly. The identity harness reports the ratio against its derived value until the
  choice lands (RES025).
- **v0.12**: Phase 2 implementation feedback, three items.
  **(a) `engines/base.py` lands in Phase 2, not 3a.** The §4.1 `Engine` protocol and the
  §5 `RecomputeCost` enum are what a "runners skeleton" is a skeleton *of*: without an
  engine type there is no stub engine either, so the invariance machinery could be written
  but never exercised. The `ENGINES` registry stays in 3a, where there will be something
  to register (`DECISIONS.md` RES018).
  **(b) The §3.2 prefilter's active region is a cone, not a cylinder.** Found by the
  Phase-2 harness, not by review: the region's radius came from the spot near focus, so a
  bunch longer than the Rayleigh range met the *diverged* pulse and had particles
  discarded that it still reached — a discarded macroparticle measured six times the `a0`
  the threshold was meant to bound. The region now carries a radius slope, and
  over-inclusiveness — the property that makes the prefilter a pure optimization — holds
  at any distance from focus (RES021).
  **(c) The golden-reference boundary is a subprocess.** Both repos install a package
  named `gammaforge`, so `make_references` runs the predecessor under its own interpreter
  and translates results at this side of the boundary; §7's "runs the old repo's models"
  is unchanged in intent, and this pins how (RES019).
- **v0.11**: two units decisions, both author-directed after a Phase-1 review.
  **(a) Dimensioned types at the engine boundary.** §2.1's "kernels never see pint
  quantities" constrained *kernels*; it was over-read during Phase 1 as "the shared layer
  holds bare floats", which was never decided and is not what P1 says. Every dimensioned
  field of `GaussianElectronBeam`, `GaussianParaxialLaser` and `Target` is now typed as a
  pint `Quantity`, still **stored** canonically in CGS so P1 stays literally true. P1
  removes the multiplicity of unit systems but not the one conversion each engine must
  still perform at its own boundary — kascade is SI internally — and typing makes that
  conversion checked instead of a hand-written factor. Engines unpack once at `run()`;
  nothing below sees a `Quantity`. Bulk per-particle arrays (`Bunch`,
  `PhotonMacroparticles`) stay raw ndarrays with **declared** units, the pattern `Axis`
  already used, because a 6D beam covariance is dimensionally heterogeneous and cannot be
  a single quantified array in any units library (`DECISIONS.md` RES013). The `light_time`
  context is now opt-in per field rather than global, so a *transverse* size can no longer
  be given in femtoseconds (RES014).
  **(b) k0_las normalization is dropped entirely** (§2.1, §4.2, open question 7). An audit
  of the predecessor found it is not physics: the `k0**2` in Stage 0's `contribution` is
  exactly what replaces `c` when coordinates are normalized, `a0_shape` is k0-free by
  construction, the H-table axes are all dimensionless, Stage 2's kernel contains no energy
  scale at all, and the only other appearances are `/k0_las` and `/omega_las` *un*-normalizing
  the diagnostics. With the laser owning lab-frame CGS sampling (P15) and the bunch owning
  CGS trajectories, Stage 0 in CGS needs no normalization — and normalizing would actively
  fight P15, since an arbitrary `LaserField` knows nothing about `k0_las` (RES015).
- **v0.10**: §3.2's beam correlation set gains the **angle**-energy pair
  (`rho_thx_gamma`/`rho_thy_gamma`, the dispersion derivative), because without it the
  `drift`/`propagate` promise that the attached description is carried "analytically in
  lockstep, no refit" is only true for a *single* drift — two consecutive drifts silently
  disagreed with one of the combined length. They are independent parameters, not
  derivable from `alpha` and the position-energy correlation (`DECISIONS.md` RES012).
- **v0.9**: Phase 1 implementation feedback — details the plan's sketches left open, now
  pinned by working code (`DECISIONS.md` RES004–RES010 carry the rejected alternatives).
  `FieldSpec` gains `choices` (a `CHOICE` field needs its closed set) and `integer` (bin
  and particle counts are not floats); both are orthogonal additions to the §3.1 sketch,
  not changes to it — `FieldKind` stays the four-member convention-semantics vocabulary.
  Beam **chirp and dispersion are stored as correlation coefficients** rather than the
  predecessor's dimensional `chirp_h`/`D_x`, which is what makes §3.2's "(dimensionless)"
  literally true, bounds them visibly by `rho_x²+rho_y²+rho_z² < 1`, and preserves the
  marginal energy spread by construction. **`Bunch.weight` is a per-particle array**, not
  a scalar: `.ele` loads carry unequal weights, and the §3.2 no-renormalization rule needs
  weights to survive filtering per particle. `Axis` members carry a `(key, unit)` pair
  because `X`/`Y` and `THETA_X`/`THETA_Y` share a unit and `Enum` would otherwise alias
  them into one member. A second pint context (`gaussian_charge`) joins `light_time` at
  the boundary — §2.1 already noted pint cannot convert charge between SI and Gaussian,
  and the schema needs exactly that to accept a bunch charge in pC. §3.4's temporal
  autorange and §3.2's prefilter are **one function**, `overlap_time_window`, which
  generalizes the predecessor's head-on-only `laser_overlap_time_window` to arbitrary
  geometry — a crossing angle needs no special case in either consumer.
- **v0.8**: pluggable laser field source (author-directed, ahead of Phase 0 kickoff) —
  new `LaserField` protocol (`a0_profile`/`field`/`active_region`, lab-frame, vectorized)
  is the only thing engines are typed against; `GaussianParaxialLaser` is reframed as its
  first implementation rather than *the* laser type (§3.3, new **P15**); `fit_gaussian_paraxial`
  added as the laser-side analogue of `Bunch.fit_gaussian`/P8, extracting descriptive
  metrics (waist, Rayleigh range, ...) from any `LaserField` so autoranging/analytical/
  the sketch panel never sample raw fields directly (§3.3, §3.4); `InteractionParameters.laser`
  retyped from `GaussianParaxialLaser` to `LaserField` (§3.5); added goal 9 and a Phase 1
  scope/exit-criteria update (§0, §11) anticipating a second implementation backed by the
  sibling `Spectral-FEM-Fields` project (arbitrary, non-paraxial-Gaussian pulses) at the
  interface level only — no work on it now, same treatment as kascade's MC replacement.
- **v0.7**: pre-implementation review — Phase 2.5 was scoped to reuse xigma's Stage 0
  (§4.5) while Stage 0 itself wasn't built until Phase 3a, which follows 2.5 in the
  table; Stage 0 and the shared chunking utility are now pulled forward into Phase 2.5
  (which delta needs anyway), with 3a trimmed to Stage 1/2 + facade + wrapper (§11).
  Also pinned the sampler's RNG architecture — independent per-variable substreams keyed
  off `seed`, deviates transformed by the *current* beam parameters — which is what
  actually makes the v0.6 bunch-resample rule compatible with the `REUSE_INTERMEDIATES`
  cost tier for γ0/energy-spread/chirp/dispersion (§3.2, §5): without it, a resample
  triggered by a γ0-only change could perturb the position/angle draws too and silently
  invalidate Stage 0's cache despite the tier claiming otherwise.
- **v0.6**: fourth review round (author-directed) — `seed` promoted to a first-class,
  GUI-displayed/editable field, and the bunch resample rule extended: any beam/laser
  physical-parameter edit resamples the bunch (not just `SamplingSpec` fields), always
  drawing from the current `seed` — "same seed" now means "same seed + same beam/laser
  parameters + same n_particles ⇒ identical bunch" (§3.5, §6); final-electron
  macroparticle typing (`Bunch` reuse vs a symmetric `ElectronMacroparticles`) left an
  **open question** pending a second MC engine currently in development by a colleague,
  which should clarify the right shape (§3.6, §10); backend-invariance test scoped to a
  tight relative tolerance (~1e-6), explicitly not bit-identical (§7); delta's
  independence caveat added — it reuses xigma's Stage 0, so it arbitrates Stage-2 kernel
  normalization only, not the full pipeline (§4.5); GUI import-boundary rule restated
  around excluded engine internals rather than a three-item allowlist, so it doesn't
  block the GUI's legitimate `io` imports (drawing module, YAML/HDF5 I/O) (P12);
  Calculate explicitly runs checked engines **sequentially** in one worker thread, not
  concurrently, since engines are heavy and would just contend for the same CPU/GPU (§6);
  two factual corrections — the merged worktree-branch count is eight, not five (§11),
  and kascade's `cfg` argument is a dataclass, not a dict (only `electrons` is
  dict-based) (§4.4).
- **v0.5**: third review round — analytical declared as 1D-`spectrum`-only (its 1D
  collimated estimate overlays the collimated sub-tab; it does not produce the 3D slice);
  bunch re-sampling rule (only `SamplingSpec` changes re-sample; target/charge/output
  edits reuse the bunch); kascade adapter reconstitutes absolute weights at its boundary
  so `kascade.py` stays untouched; `.ele` loads normalize to relative weights (N_e from
  the charge field); prefilter threshold defined as a fraction of peak a0; MC
  statistical tolerance in cross-engine validation; doc-staleness guard made concrete
  (backticked identifiers must resolve); charge live-rescale scoped to charge-alone
  edits; `Bunch.n_electrons` property removed.
- **v0.4**: interaction-model correction — **no engine is real-time; only analytical is**.
  Engines are Calculate-gated: input edits mark results stale, Calculate re-runs the
  checked engines with cache reuse (xigma skips Stage 0/1 on unchanged geometry; MC just
  filters particles and rebins). Target changes are a *cheap re-Calculate*, never a live
  replot. Charge is the sole exception (exact N_e linearity → instant display rescale).
  The old "live-requery" risk is structurally impossible (no auto-firing path) — the
  residual risk is only that a deliberate Calculate may be slow (§5, §12); the
  `_MAX_LIVE_N_ENERGY_*` hardcap lesson explicitly does not apply.
- **v0.3**: second review round — `collimated_spectrum` is a 3D (E,θx,θy) slice windowed to
  the target with an explicit GUI visualization pipeline (§3.4); `OutputKind` vocabulary
  replaces axis-grouping capability declarations (§3.4, §4.1); `PhotonMacroparticles`
  typed separately from final electrons (§3.6); N_e decoupled from bunch weights —
  charge is an exact linear rescale, `QUERY_ONLY` (§3.5, §5); prefilter **never
  renormalizes** and is a pure optimization with an invariance test (§3.2, §7);
  `SamplingSpec(n_particles, seed, prefilter)` lives at the interaction level (§3.5);
  `editable_after_run` removed from `FieldSpec` (§3.1); rotation composition pinned —
  roll about k̂ fixed by rotation order + round-trip test (§2.2); elliptical + astigmatic
  Gaussian laser model (§3.3); cross-backend invariance and seed-determinism validation
  categories (§7); "Use for calculation" checkboxes with one Calculate button (§6, §10);
  collimated-slice cost risk added (§12); perf-regression item dropped per author.
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
