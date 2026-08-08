# GammaForge — Decisions

Implementation-level design decisions made *during* the build, with their rejected
alternatives — in the spirit of the predecessor's documentation discipline (grand-plan
goal #8). This is distinct from `docs/GRAND_PLAN.md`'s own changelog, which tracks
*plan* revisions (architecture decisions made before code exists); entries here are
added as things actually get built.

**Convention:** newest entry last, short id (D001, D002, ...) so other docs can
reference it. **Backticks mean "this exists right now and should resolve"** — a real
file path or a real symbol in the installed package. Names being discussed
hypothetically (rejected alternatives, not-yet-built files, forward-looking examples)
use *italics* instead, deliberately, so the doc-staleness guard (D002 below) can trust
every backtick without a manual exceptions list.

---

### D001 — Build backend: hatchling, src layout

**Decision:** `pyproject.toml` uses hatchling as the build backend, with the package
under `src/gammaforge/`.

**Rejected alternative:** *setuptools* — works equally well for a src-layout package,
but needs more explicit configuration (a packages-find table) for the same result;
hatchling's defaults handle src-layout with no extra config.

**Rationale:** one less thing to get wrong in Phase 0; no other project constraint
favors setuptools specifically.

---

### D002 — Doc-staleness guard (C2) scope: `DECISIONS.md` only, not `GRAND_PLAN.md` or `PROGRESS.md`

**Decision:** the Phase-0 doc-staleness smoke test (`tests/test_doc_staleness.py`)
only checks backticked identifiers in `DECISIONS.md` against the repo/installed
package. `GRAND_PLAN.md` and `PROGRESS.md` are explicitly excluded.

**Rejected alternative:** checking every doc under `docs/` plus `PROGRESS.md`.

**Rationale:** the guard's purpose (per the plan's own risk table, §12) is catching
*backward* drift — docs that used to be true and silently stopped being true, the
failure mode observed in the predecessor repo's *AGENTS.md* (a different repo — not
a claim checked against this one). `GRAND_PLAN.md` is a forward-
looking roadmap by design: most of its backticked identifiers (e.g. *LaserField*,
*fit_gaussian_paraxial*) intentionally name things that don't exist yet and won't until
their assigned phase lands — checking it now would fail universally, not usefully.
`PROGRESS.md` is a narrative session log that mixes "what we did" with "what we just
decided," so it has the same problem in miniature. `DECISIONS.md` entries are only
added once a decision is actually implemented, so it's the one doc where "every
backtick resolves" is true by construction *and* worth mechanically enforcing as things
get renamed later. The check can grow to cover more docs (e.g. a future API reference)
once they exist and follow the same after-the-fact convention.

---

### D003 — Phase 0 package skeleton: subpackage `__init__.py` only, no placeholder module files

**Decision:** Phase 0 creates six subpackages under `src/gammaforge/` — `io`,
`engines/xigma`, `engines/analytical`, `engines/kascade`, `validation/references`,
`gui` — each with a docstring-only `__init__.py`. It does not pre-create the
individual module files the plan names for later phases (e.g. *schema.py*, *beam.py*,
*laser.py*, *stages.py*, *collision.py*).

**Rejected alternative:** stub out every module file the plan names now (empty, or
raising *NotImplementedError*), so the full tree matches the plan's layout from day
one.

**Rationale:** an empty *schema.py* sitting in the tree for the whole of Phase 0 either
looks finished (misleading) or needs a marker comment nobody will remember to remove.
Phase 1's own exit criteria (round-trip tests, schema validation tests, ...) are the
actual definition of "the schema module exists" — better for a file's existence to mean
something than to pre-populate the tree for cosmetic completeness.

---

### D004 — `FieldSpec` gains `choices` and `integer`; `FieldKind` stays four members

**Decision:** `FieldSpec` (`schema.py`) carries two fields the `GRAND_PLAN.md` §3.1 sketch
did not name: `choices` (required for `FieldKind.CHOICE`, forbidden otherwise) and
`integer` (a numeric field whose value must be integral — `n_particles`, `seed`, bin
counts). `FieldKind` keeps exactly the four members the plan lists.

**Rejected alternative:** a fifth *INTEGER* member of `FieldKind`.

**Rationale:** `FieldKind` answers one question — *does this value carry a width
convention, and is it a number at all?* `WIDTH`/`DURATION` do carry one, `SCALAR` does
not, `CHOICE` is a string. Integer-ness is orthogonal to that axis, exactly like
`value_range`, and folding it in would make the enum answer two unrelated questions at
once. A `CHOICE` field with no `choices` is unusable, so that slot is not an extension of
the vocabulary either — it is the sketch being completed.

---

### D005 — Beam correlations are stored as correlation coefficients, not dimensional slopes

**Decision:** `GaussianElectronBeam` stores `rho_x_gamma`, `rho_y_gamma`, `rho_z_gamma` —
dimensionless correlation coefficients. `chirp_to_correlation` and
`dispersion_to_correlation` convert from the dimensional forms a user thinks in.

**Rejected alternative:** the predecessor's *chirp_h* (dγ/dz, 1/cm) and *dispersion_x*
(Cov[x,γ]/σ_γ², cm), with the sampler deriving conditional variances from them.

**Rationale:** three things fall out for free. §3.2 calls these correlations
"(dimensionless)" — with slopes that was aspirational, now it is literal. The
admissibility condition becomes a visible `rho_x² + rho_y² + rho_z² < 1` that `validate`
can state in the user's own terms, instead of a conditional variance going negative
somewhere inside the sampler. And because the sampler applies them to *standardized*
deviates, the marginal energy spread stays exactly `sigma_gamma` however strong the
correlations are, rather than by a cancellation that has to be got right.

---

### D006 — `Bunch.weight` is a per-particle array

**Decision:** `weight` is an ndarray of relative per-particle weights summing to 1 over an
unfiltered bunch, not the scalar `1/n_particles` the plan's §3.2 wording suggests.

**Rejected alternative:** a scalar `weight`, with non-uniform distributions handled by a
separate optional array or rejected outright.

**Rationale:** two existing requirements need it. `.ele` files may carry unequal weights
(§8), and `prefilter_bunch` must drop particles *without renormalizing* (§3.2) — with a
scalar, the surviving weight would either silently become wrong or need a second field to
record the count it no longer matches. As an array, the prefilter is `weight[mask]` and
the sum falling below 1 is the honest record of what was dropped. The uniform case costs
one array beside six others already there.

---

### D007 — `Axis` members carry a `(key, unit)` pair

**Decision:** `Axis.X` is `("x", "cm")`, not `"cm"`; the unit is reachable as
`Axis.unit` and a stable serialization key as `Axis.key`.

**Rejected alternative:** the unit alone as the enum value, as first written.

**Rationale:** not a style preference — a bug found by a failing test. `X` and `Y` share
`cm`, and `THETA_X`/`THETA_Y` share `rad`; `Enum` silently collapses members with equal
values into **aliases**, so `Axis.Y is Axis.X` was true and every two-dimensional slice
quietly lost an axis. Distinct keys keep the six members six, and give HDF5 a dataset name
that does not depend on the Python identifier.

---

### D008 — A `gaussian_charge` pint context beside `light_time`

**Decision:** `units.py` registers a second pint context converting between SI charge and
statC by the exact `1 C = c[cm/s] / 10 statC` factor, derived from `C_CGS`.
`to_canonical`/`from_canonical` apply both contexts (`BOUNDARY_CONTEXTS`).

**Rejected alternative:** special-casing the charge dimensionality inside `to_canonical`
with a hand-written branch.

**Rationale:** `GRAND_PLAN.md` §2.1 already records that pint cannot convert
electromagnetic quantities between the two systems, and applies the factor by hand for the
one constant that needs it. The schema needs the same conversion for a *user-entered*
value — a bunch charge in pC — and a named context is the mechanism pint already provides,
the one `light_time` uses, and the one that keeps the exception from firing where nobody
asked for it. A branch inside `to_canonical` would have put unit-system knowledge in a
function whose whole job is to not have any.

---

### D009 — One `overlap_time_window` serves both the prefilter and the temporal autorange

**Decision:** `bunch.py` exposes `overlap_time_window(bunch, laser, threshold)`, returning
per-particle `(t0, t1)` from intersecting each straight-line trajectory with the laser's
`active_region`. `prefilter_bunch` filters on `t0 <= t1`; `target.auto_ranges` derives the
`TEMPORAL_ENVELOPE` window from the same call.

**Rejected alternative:** porting the predecessor's *laser_overlap_time_window* as a
closed-form head-on formula for the autorange, with the prefilter testing points
separately.

**Rationale:** the predecessor's version is head-on-only and takes pre-normalized
`k0_las`-scaled arguments, so a crossing angle would have needed a second implementation
in each of two consumers — four code paths for one geometric question. Intersecting a line
with the region is closed-form anyway (linear in `t` longitudinally, quadratic
transversely) and carries the crossing angle for free, because the angle is already in
`ActiveRegion.axis`. This is the §4.2 "one shared utility replacing three inconsistent
implementations" discipline applied before the duplicates appear.

---

### D010 — `fit_gaussian_paraxial` implements the identity path only

**Decision:** `fit_gaussian_paraxial` returns a `GaussianParaxialLaser` input unchanged
and raises `NotImplementedError` for any other `LaserField`.

**Rejected alternative:** a numerical moment-based fit — sample `a0_profile` on a lattice,
extract peak, waists and Rayleigh ranges — written now so the function is total.

**Rationale:** no second `LaserField` implementation exists (`Spectral-FEM-Fields` has no
Python bindings yet), so a numerical path written today could only be tested against the
analytic laser it is not for, and would be guessing at a field representation nobody has
seen. That is the speculative abstraction P6 rejects, and D003's reasoning about empty
module files applies: better for code to exist because something needs it. The identity
path is what Phase 1 has a consumer and an exit criterion for; the numerical path lands
with the implementation that requires it, which will also be able to test it.

---

### D011 — Doc-staleness guard hardened for enum members and ambiguous shapes; its own behaviour is tested

**Decision:** `tests/test_doc_staleness.py` now (a) tries **every** applicable
interpretation of a token and reports it stale only if none resolves, rather than
committing to the first shape that matches; (b) indexes enum member names, so `WIDTH`
resolves; (c) accepts annotation-only dataclass fields, so `Bunch.weight` resolves;
(d) skips bare file extensions. Three further tests pin the guard's own behaviour against
fixed lists of tokens that must resolve, must be caught, and must be skipped.

**Rejected alternative:** keeping the first-match ordering and adding an exceptions list
for the tokens it misclassified.

**Rationale:** the first-match rule was not merely imprecise, it was wrong in a way that
grew with usage: `FieldKind.CHOICE` matches the file-like shape ("a name, a dot, a short
suffix") just as `GRAND_PLAN.md` matches the class-attribute shape, so *every* dotted
symbol reference in the docs was reported stale as soon as Phase 1 gave the docs symbols
to reference. An exceptions list would have made D002's whole premise false — that
convention exists precisely so no manual exception list is needed. Testing the guard
itself matters more than usual here because its failure mode is silent: relaxing a rule to
kill a false positive can quietly stop it detecting anything, and nothing would say so.
The stale-token list is drawn from names this project rejected deliberately
(*BeamFittedParams*, *ModelCapabilities*, *NoConvention*, *Bunch.n_electrons*), so it
doubles as a check that they stay gone.

---

### D012 — Angle-energy correlations are stored, so `drift` composes

**Decision:** `GaussianElectronBeam` carries `rho_thx_gamma` and `rho_thy_gamma` — the
angle-energy correlations, i.e. the dispersion derivative — alongside the position-energy
ones. `_drift_plane` transports `cov(x, gamma) -> cov(x, gamma) + L cov(x', gamma)` and
leaves the angle-energy correlation alone, since a drift changes neither the angle nor
gamma. `gamma_coefficients` holds the resulting sampler algebra and the joint
admissibility condition.

**Rejected alternative:** the original D005 design, storing only the position-energy
correlations and re-deriving `cov(x', gamma)` from `alpha` at each drift step, on the
model assumption that gamma couples to the angle *only* through the position.

**Rationale:** that assumption is true of a freshly sampled bunch and **destroyed by the
first drift**. The symptom was subtle enough to survive the original test suite: a single
`drift` agreed with a refit of the drifted macroparticles to sampling noise, so the
transport looked right, while two consecutive drifts silently disagreed with one drift of
the combined length (`rho_x_gamma` 0.238 vs 0.186 for 25 + 25 cm against 50 cm). It was
found by asking the transport to *compose* and to *reverse*, neither of which the earlier
test did — see `PROGRESS.md` for how the test came to be written.

The stored form is also the more faithful model, independently of the bug: a bunch created
at a waist with dispersion but no dispersion derivative keeps `rho_thx_gamma = 0` however
far it drifts, while `alpha_x` and `rho_x_gamma` both change — so the two genuinely are
independent parameters, and the old formula silently imposed a relation between them.
`fit_gaussian` now extracts them too, rather than discarding that part of the covariance.

The resulting transport is exact rather than approximate: `drift` composes and reverses to
round-off (~1e-16), which is what `test_drift_composes` and `test_drift_is_reversible` pin.

---

### D013 — Dimensioned fields are pint `Quantity`; bulk arrays declare their units instead

**Decision:** every dimensioned field of `GaussianElectronBeam`, `GaussianParaxialLaser`
and `Target` is a pint `Quantity`, validated and **converted to canonical CGS** on
construction (`as_canonical_quantity`). Each class declares its canonical units in `UNITS`
and unpacks through `m(name)` once at the top of every numeric routine, so no loop or
kernel below ever sees a `Quantity`. `Bunch` and `PhotonMacroparticles` are **not** wrapped:
they carry raw ndarrays plus a declared `UNITS` mapping and convert via `Bunch.get`.

**Rejected alternatives:** (a) bare CGS floats everywhere, converted by documented
convention — what Phase 1 originally built; (b) quantified numpy arrays for `Bunch` too;
(c) floats plus a separate dimensioned "view" object handed to engines.

**Rationale.** (a) was never actually a decision. `GRAND_PLAN.md` §2.1 says "**kernels**
never see pint quantities"; that was over-read as "the shared layer holds bare floats",
which the plan does not say and which no entry here recorded. P1 removes the *multiplicity*
of unit systems, but not the one conversion each engine must still perform at its own
boundary — kascade is SI internally (§4.4) — and under (a) that conversion is a
hand-written literal whose failure mode is a silent factor of 100, exactly the class of
error P1 exists to prevent. The predecessor had reached the same conclusion independently:
its *PhysicalQuantity* travelled across every model boundary for this reason.

(b) fails for a reason stronger than performance. `np.cov`, `np.corrcoef` and
`np.linalg.slogdet` have no pint implementation and `fit_gaussian` needs all three; more
fundamentally its 6D covariance is **dimensionally heterogeneous** — `Sigma[x,x]` is cm²,
`Sigma[x,gamma]` is cm, `Sigma[gamma,gamma]` is dimensionless — so it cannot be one
quantified array in any units library, only in a per-element unit matrix. Declared units
plus a scale factor is the same pattern `Axis` already uses for result slices, and it
exploits the fact that every conversion here is a *pure scaling*: there are no offset units
anywhere in this project, so a conversion is one number that a caller can fold into
arithmetic it is already doing. `Bunch.get` returns the array itself, with no copy, when the
requested unit is the stored one — the case for every CGS consumer.

(c) leaves the raw attribute reachable, so the wrong thing stays possible; P12's precedent
is to make it impossible rather than discouraged.

---

### D014 — The `light_time` context is opt-in per field, not global

**Decision:** `to_canonical`/`from_canonical`/`as_canonical_quantity` take
`light_time=False` by default. Only fields listed in a class's `LIGHT_TIME_FIELDS`
(`GaussianElectronBeam.sigma_z`, `GaussianParaxialLaser.duration`) and `FieldKind.DURATION`
schema fields opt in. The `gaussian_charge` context stays unconditional.

**Rejected alternative:** applying both contexts everywhere, as the first implementation
did.

**Rationale:** found by a test written to check that a wrongly dimensioned value is
refused — and it was not. Globally enabled, `light_time` equates *any* length with *any*
duration, so a **transverse** beam size was happily accepted in femtoseconds. That is not a
unit choice, it is a different physical quantity. §2.1 introduces the context for one
specific pairing — a longitudinal extent that may be quoted either way — and scoping it to
exactly that keeps the rest of the dimensional checking meaningful. `gaussian_charge` needs
no such scoping: a value either is a charge or is not, and the SI/Gaussian split is
notational rather than a physical ambiguity.

---

### D015 — No coordinate normalization: xigma works in CGS directly

**Decision:** the predecessor's `k0_las`-normalized coordinate system is not carried over.
Stage 0 will integrate trajectories, sample `LaserField`, and bin diagnostics in CGS.

**Rejected alternative:** porting the normalization, as `GRAND_PLAN.md` open question 7
previously assumed ("stays purely internal to xigma").

**Rationale:** an audit of the predecessor's pipeline (author-directed) found the
normalization buys nothing. Its `k0**2` in the overlap integrand is precisely the Jacobian
of the normalization — with `u = k0 r` and `tau = k0 c t`, `n_phys * c * dt_phys` becomes
`n_u * k0**2 * dt_u`, so `k0**2` is what *replaces* `c`, not independent physics.
`a0_shape` is built from a ratio of envelopes and is k0-free by construction; the H-table's
axes (gamma, theta_x, theta_y, a0) are all dimensionless; the Stage-2 kernel contains no
`omega`, `hbar` or energy scale at all, working in a dimensionless `s` that the adapter
converts with `4 hbar omega0` at its boundary. Every remaining appearance is `/k0_las` or
`/omega_las` *un*-normalizing the spatial and temporal diagnostics.

So in the rebuilt architecture — where the laser owns lab-frame CGS sampling and the bunch
owns CGS trajectories — normalizing would mean scaling coordinates on the way into
`LaserField` and unscaling on the way out, for no gain. It would also fight **P15**
directly: an arbitrary `Spectral-FEM-Fields`-backed pulse knows nothing about `k0_las`, and
a crossing angle makes the bookkeeping worse. The one remaining argument for it would be
float32 conditioning on the GPU path, which is a Phase-2.5 measurement rather than a reason
to normalize now: in CGS the integrand's factors sit near 1e8 cm⁻³ and 1e-13 s, comfortably
inside float32's range.

---

### D016 — *beam.py* renamed to `bunch.py`; the "canonical bunch" phrasing dropped

**Decision:** the §3.2 module is `bunch.py`, and the word "canonical" no longer qualifies
the bunch or the sampling path anywhere. It still qualifies *units* — "the canonical CGS
unit" is the one a value is stored in — which is a different and standard usage.

**Rejected alternative:** keeping *beam.py*, on the grounds that the module holds
`GaussianElectronBeam` as well as `Bunch`.

**Rationale:** the module's centre of gravity is the macroparticle bunch — sampling, the
prefilter, propagation, the fit — with the beam description as the analytic summary of it,
and the predecessor named the same content `bunch.py` for the same reason, so this is
continuity rather than churn (the same argument that keeps the package called `io`).

"Canonical bunch" was read by the author as unexplained jargon, which it was: it meant
nothing more than "every engine in a run is given the same bunch", and that is now simply
what the docstring says. A phrase that needs decoding to convey an ordinary fact is a cost
with no benefit, and this project's documentation discipline (goal #8) is about
provenance, not vocabulary.

---

### D017 — Dimensioned fields are converted to canonical CGS on construction, not stored in the unit given

**Decision:** `as_canonical_quantity` converts an incoming `Quantity` into the field's
canonical CGS unit and stores *that*. A beam built with `Quantity(20, "um")` holds
`0.002 centimeter`, and `m(name)` is therefore a free attribute read.

**Rejected alternatives:** (a) store the `Quantity` exactly as the user gave it, letting
pint keep the physics right and converting only on access; (b) the same, but with the
canonical magnitudes precomputed into a private dict so `m` stays free.

**Rationale.** The motivating argument for (a) was floating-point precision: 1 µm stored
as `100` looks better conditioned than the same value stored as `1e-4`. That intuition is
from **fixed** point. IEEE 754 doubles carry 52 mantissa bits at *every* exponent, so
relative precision is a constant ~2.2e-16 across the whole normal range — `1e-4` is exactly
as precise as `100`, and the conversions this project performs were measured at **0 ULP**
(`100 um` → `0.01 cm` lands on the same double as the literal). For scale, a measured spot
size is known to about a percent: fourteen orders of magnitude of headroom.

The two places magnitude does matter are absent here. Underflow/overflow: doubles span
1e±308 and these values live between 1e-25 and 1e10; even float32 leaves nineteen orders of
headroom on the Stage-0 integrand. Catastrophic cancellation: real, but unaffected by the
choice of unit.

With no precision case, what remained for (a) was faithfulness of `repr` — seeing
`20 micrometer` rather than `0.002 centimeter`. That is cosmetic, and the unit a user
actually typed is already preserved where it is visible: spec files are written in display
units, and `Parameters.display` converts to any unit on demand. Against it, storing as
given breaks the letter of P1 ("every shared dataclass stores canonical CGS-Gaussian
values") and makes `m` a pint conversion — measured at 65 µs against 94 ns for a plain
attribute read, a 700x difference. That is negligible in the vectorized design, where the
laser is sampled once per chunk of ~1e6 particles, but it is a live trap for any caller
that reaches for `a0_profile` inside a Python loop.

(b) removes the performance objection and keeps the nicer `repr`, at the price of storing
each value twice and adding state that is not quite what P9 forbids but is adjacent to it.
It is the option to revisit if the display of stored values ever becomes a real complaint;
until then it buys presentation, not correctness.

---

### D018 — `engines/base.py` lands in Phase 2, with the `Engine` protocol but no registry

**Decision:** the §4.1 `Engine` protocol and the §5 `RecomputeCost` enum are written in
Phase 2, one phase before the engines they describe. The *ENGINES* registry named in the
same plan section is *not*: it arrives with the first engine that has something to
register.

**Rejected alternatives:** (a) leave `gammaforge.engines` empty until Phase 3a and have
`run_engine` accept any object with a `run` method, checking nothing; (b) defer the whole
runner layer to Phase 3a and make Phase 2 golden-generation only.

**Rationale.** Phase 2's scope is "runners skeleton", and a skeleton needs a joint. (a)
would have replaced a checked contract with a duck-typed one at precisely the boundary the
plan spends §4.1 pinning down — and `runtime_checkable` makes the check cost one
`isinstance` and produce an error naming what an engine is missing, rather than an
`AttributeError` from inside a run. (b) would have left the harness untestable: with no
engine type there is no stub engine either, so the chunk/backend/prefilter invariance
machinery could only be *written*, never *exercised*, which is how scaffolding rots.

The registry is a different matter. A lazy optional-dependency import table over zero
engines is exactly the speculative machinery P10 warns about, and it is three lines to add
when xigma exists.

---

### D019 — Golden references are generated in a subprocess, and committed as ordinary results files

**Decision:** `validation/make_references.py` builds a JSON description of each scenario in
the predecessor's units and runs `validation/_predecessor_driver.py` under a *separate
interpreter* whose path points at the old checkout (the OLD_REPO and OLD_REPO_PYTHON
environment variables — env var names, not package symbols, hence no backticks). The
driver writes plain `.npz` files; the translation into this repo's `Axis`/`OutputKind`
vocabulary and CGS units happens on this side of the boundary. Snapshots are stored as the
ordinary `gammaforge.io.formats.hdf5` results file plus a `provenance` group and a
`scalars` group, under `src/gammaforge/validation/references/data/`, and are **committed**
— which needs a `.gitignore` negation, since `*.h5` is otherwise excluded wholesale.

**Rejected alternatives:** (a) import the old package directly and call it in-process;
(b) let the driver write this repo's HDF5 format itself; (c) keep goldens out of git and
regenerate them before each run.

**Rationale.** (a) is impossible, not merely inadvisable: both repos install a package
named `gammaforge`, and no import trick makes two of them coexist safely in one process.
The subprocess is the design. It also buys the old repo's own environment — it needs
scipy, which this repo deliberately does not have. (b) would put this repo's serialization
format inside a file that runs against the old repo's dependencies, where it could not be
tested and would silently drift; instead the driver reports what the old code produced in
the old code's own terms, and the one place that knows both vocabularies is versioned with
the format it targets. (c) would make the suite depend on a machine that has the
predecessor checked out — the opposite of what a reference is for. At ~24 kB per snapshot
the C3 concern about committed data does not bite; the negation is scoped to that one
directory so it cannot quietly re-admit a stray multi-MB file elsewhere.

---

### D020 — The validation runners have no result cache

**Decision:** `validation/runners.py` runs an engine and returns; nothing is memoized to
disk.

**Rejected alternative:** porting the predecessor's *cache.py* — a commit-hash-keyed
pickle store that skipped recomputation when the tree was clean.

**Rationale.** That cache existed because a GPU run per tier per scenario dominated the
suite's cost, and the tiers each re-ran the models independently. Neither is true here:
the tiers are gone (one run is shared), and engines cache their *own* intermediates keyed
by the exact hash of the inputs each one consumed (§5), which is both finer-grained and
valid by construction rather than by a clean-tree heuristic. A second, coarser cache on
top would be the speculative machinery P6 warns about. It is worth reinstating the moment
a full suite run is *measured* to be too slow — not before.

---

### D021 — The laser's active region is a cone, not a cylinder

**Decision:** `ActiveRegion` carries a `radius_slope` alongside its `radius`, and
`radius_at` gives `radius + radius_slope * |u|`. `GaussianParaxialLaser.active_region`
derives both by linearizing the spot hyperbola,
`s(u) = sigma sqrt(1 + ((u - z_f)/z_R)^2) <= sigma (1 + (|u| + |z_f|)/z_R)`.
`overlap_time_window` evaluates that cone at the widest point of each particle's own
longitudinal window, so the transverse test stays a single closed-form quadratic.

**Rejected alternatives:** (a) keep the cylinder and give `active_region` an extra
argument for the longitudinal span it must be valid over; (b) keep the cylinder and size
its radius from the largest spot the bunch can ever meet; (c) solve the cone inequality in
`t` exactly.

**Rationale.** This fixes a real defect, found by the Phase-2 harness rather than by
review. The region's contract is to be **over-inclusive** — the prefilter is a pure
optimization and must never discard a particle that would have contributed (§3.2) — and
the old radius came from the spot within the pulse's *own* length around focus. A pulse
diverges: a bunch longer than the Rayleigh range meets it far from focus, where the spot
is many times larger, so particles the expanded pulse still reaches were being thrown
away. In the probe configuration a discarded macroparticle saw `a0` **six times** the
threshold meant to bound it.

(a) works but changes the `LaserField` protocol for every implementer (P15) and leaves a
default that is wrong for anyone who forgets the argument — a footgun in a safety
mechanism. (b) needs no protocol change but sizes every particle's test by the worst
particle in the bunch, which costs most of the filter's value on exactly the long-bunch
case that motivated the fix. (c) is the tight answer, but a cone inequality in `t` is not
a single band — the parabola can open downward, making the solution set a union of two
rays — and the per-particle evaluation in (the chosen option) is already conservative,
already closed-form, and tight in the only regime that matters. The pre-fix behaviour is
now guarded twice: at the laser level
(`test_active_region_is_still_conservative_far_from_focus`) and by the harness check that
found it (`check_prefilter_discards_only_dark_particles`), which samples `a0` along every
discarded trajectory and so depends on none of the geometry it is testing.

---

### D022 — The active-region cone is evaluated at the flying-focus coordinate

**Decision:** `active_region` derives its cone from `s(v)` where `v = u + beta_ff * ct`,
the coordinate `_local_coordinates` actually evaluates the spot at — not from `s(u)`.
Inside the longitudinal window `|v| <= |1 + beta_ff| |u| + |beta_ff| half_length`, so the
slide multiplies the slope and the drift widens the intercept.

**Rejected alternative:** *keeping the `s(u)` derivation* and treating `beta_ff` as a
detail the bounding region need not model, on the grounds that the `(1 + beta_ff)` factor
already in `rayleigh_x`/`rayleigh_y` accounts for it.

**Rationale:** it does not — it accounts for it **backwards**. The stretched Rayleigh range
makes the cone *shallower*, and the omitted slide would have made it steeper by exactly the
same factor. Dropping one of the pair leaves the region narrower than the pulse it bounds,
which is the one direction a conservative bound may not err in; measured, a flying-focus
pulse put 1775 of 4913 above-threshold sample points outside the region, and the harness
check reported a discarded macroparticle at 1.2x its threshold. With both terms present
they cancel and the slope is `beta_ff`-independent — which is the property
`test_the_flying_focus_cancels_out_of_the_cone_slope_but_not_its_intercept` now pins, since
an assertion that the region merely "gets wider" would pass against the broken version too.

This is the same defect class as D021, found the same way and one review later: the D021
fix rederived the transverse bound and carried the pre-existing `s(u)` reading forward
without noticing that the spot is not evaluated there.

---

### D023 — The window metric counts a window if *either* side has flux in it

**Decision:** `window_integrated_deviation` includes a window in `max_window` when the
candidate has flux above the floor even if the reference has none, and the floor is a
fraction of the **total** reference flux (1e-3) rather than of the mean window flux.
`weighted_l1` stays reference-weighted. An all-zero reference is scaled by the candidate's
own total instead of returning a perfect match.

**Rejected alternative:** the predecessor's *`significant = win_flux_ref > floor`*, which
restricts both reported numbers to the reference's own support.

**Rationale:** that restriction is invisible in every test one would think to write and it
blinds the metric to the exact failure the module documents itself as catching. Photons
appearing where the reference has none — a Compton edge in the wrong place — contribute
nothing to a reference-*weighted* average, by construction, and were then dropped from the
maximum as well: a candidate carrying 0.12% of its yield in a region the reference sets
hard to zero scored `weighted_l1 = 0.0, max_window = 0.0` and passed a 2% golden tolerance.
The floor moved to a fraction of the total for a related reason: as a fraction of the
*mean window* it shrinks when the binning is refined, so the same spurious flux reports a
different deviation at 64 bins and at 256. Against the total, a spurious peak is measured
in units of a thousandth of the yield — a number a tolerance can be set against, and one
that does not move when the grid does.

---

### D024 — Stage 0 reads the whole laser through `a0_profile`

**Decision:** `integrate_trajectories` needs exactly two things from a `LaserField`:
``a0_profile`` sampled along each trajectory, and ``active_region`` (through
`overlap_time_window`) to bound the integration. The photon density it needs comes from
inverting the laser's own energy→a0 chain — `photon_density_scale` — in which the pulse
energy cancels, leaving ``n_photons(r,t) = a0(r,t)**2 (m_e c)**2 omega0 / (8 pi hbar e**2)``.

**Rejected alternatives:** (a) *extend the `LaserField` protocol with a `photon_density`
method*; (b) have Stage 0 evaluate a Gaussian envelope itself, as the predecessor's
*push_and_sample* did by calling its own *pulse_envelope* with hand-passed
*sigma_lr0*/*sigma_lz*/*beta_ff* scalars.

**Rationale:** (b) is what P15 exists to prevent — an engine that unpacks a laser into
five floats and re-derives its shape has hardcoded a Gaussian, whatever the type
annotation says, and a non-Gaussian `LaserField` cannot be substituted into it. (a) would
work, but it widens the protocol every future implementation must satisfy in order to
supply something already derivable from a method it must supply anyway. The cancellation
of the pulse energy is what makes that true and is worth stating: two quantities that look
independent (how bright the pulse is, how many photons are in it) are one, because ``a0``
was defined from the same energy.

The remaining input, ``omega0``, comes from `fit_gaussian_paraxial` — the descriptive-fit
route §3.4 already established for autoranging, not an attribute read off a concrete type.

---

### D025 — Delta's ``2 pi`` is derived and reported, not corrected

**Decision:** `validation.references.delta` keeps the predecessor's differential prefactor
unchanged. `check_normalization` reports the ratio against ``2 pi``, and the identity
section of `run.py` passes while that ratio holds and fails if it moves.

**Rejected alternatives:** (a) *divide delta's prefactor by 2 pi* so the identity reads
1.0; (b) assert the identity against 1.0 and let the suite run red until §9.1 closes;
(c) report the ratio without an expected value, as the predecessor did.

**Rationale:** the predecessor recorded this as "consistently ~6.3x ... suspiciously close
to 2*pi, not yet explained". It is not close to ``2 pi`` — it is ``2 pi``, and the integral
is elementary: with ``u = gamma**2 r**2``, ``int dOmega 3 gamma**2 <a_fac> / (1+u)**2 =
3 pi [1 - 2/6] = 2 pi``. Reproducing that from an independent CGS implementation also
rules out the old repo's ``k0_las`` normalization as the cause.

(a) is the tempting one and it is what P14 forbids: the derivation says the two methods
are inconsistent by ``2 pi`` and identifies which side counts photons — Stage 0's
``flux x cross-section x time``, corroborated by a closed form that reproduces it exactly
— but *which* normalization xigma's Stage-2 kernel should carry is a statement about the
paper's formalism, and that kernel does not exist yet (Phase 3a). Pasting the factor into
the arbiter now would remove the evidence before the question is asked.

(b) makes the suite permanently red, and a permanently red suite is an ignored suite.
(c) is where the predecessor left it, and "an unexplained 6.3" survived for as long as it
did precisely because nothing would ever notice it changing. Pinning the derived value
keeps the discrepancy visible *and* guarded.

---

### D026 — §9.1 resolved: the missing factor is in the paper, at eq. (xsec)

**Decision:** the ~2pi of `GRAND_PLAN.md` §9.1 is traced, closed as an investigation, and
recorded here. It is **not** a porting artefact and **not** a coding error: `xigma.tex`'s
differential cross-section, eq. *(xsec)*, is missing a factor `1/(2 pi)`, and the paper's
principal result eq. *(main)* inherits it. No code was changed — see D025 for why the
arbiter keeps reporting the factor rather than absorbing it, and why the *value* of
Stage 2's normalization constant is Phase 3a's to set once the author has chosen where the
factor belongs.

**The check.** Eq. *(collision)* writes the emission as
`d3N/(dw d2Omega) = int v_rel (d3sigma/(dw d2Omega)) n_ph f_e dt d3r d3p`, so for a photon
count the frequency- and angle-integrated cross-section must be `sigma_T`. From eq.
*(xsec)* with `R -> delta` and linear polarization (`Xi = diag(1,0)`, whose trace factor is
eq. *(umod)*'s `|u_1|^2`), with `u = gamma^2 theta^2`:

    int dw  (w_R/w) delta(w - w_R) = 1
    int d2Omega  3 sigma_T gamma^2 |u_1|^2 / (1 + gamma^2 theta^2)^2
        = (3 sigma_T / 2) [ 2 pi int_0^inf du/(1+u)^2  -  4 pi int_0^inf u du/(1+u)^4 ]
        = (3 sigma_T / 2) [ 2 pi (1) - 4 pi (1/6) ]
        = 2 pi sigma_T

Both integrals are elementary. So eq. *(xsec)*'s prefactor should read `3 sigma_T/(2 pi)`,
and eq. *(main)*'s `6 sigma_T w_L/w^2` should read `3 sigma_T w_L/(pi w^2)`. These are one
error, not two: eq. *(main)* is eq. *(xsec)* times the Jacobian `dGamma/dw` of eq.
*(jacobian)*, which is why `3 -> 6` and `gamma^2 -> Gamma^5/(1+ahat)` while the `1/(2 pi)`
passes straight through. The paper states the opposite immediately above eq. *(xsec)* —
that for observables with no spectral resolution the yield follows from the cross-section
formalism *exactly* — which is the claim the two lines above falsify as written.

**Why it survived in the predecessor.** It was absorbed twice, knowingly, and both sites
carry a comment saying so: xigma's adapter rescaled the kernel's angular spectrum by
`total_yield / full_integral` ("*1.0 would mean the kernel's own normalisation already
agreed with total_yield; it currently doesn't (~2*pi-ish)*"), and the analytical model
applied the same self-consistent rescale under a *QUICK FIX* label. The headline
`total_yield` came from the luminosity sum, a path that never touches eq. *(main)*, so the
kernel's normalization was never the number anyone read. *delta* was the only consumer of
eq. *(xsec)* raw, and therefore the only place the factor was visible.

**Evidence this repo adds.** `integrate_trajectories` computes the elementary
`flux x cross-section x time` count and agrees with the predecessor's own `total_yield` to
0.12%; `single_electron_spectrum` integrates to that same number as an identity; delta
gives 2 pi times it. Two independent methods against one, and the odd one out is the
paper's differential form. Reproducing the factor in a CGS implementation with no
coordinate normalization also rules out the predecessor's `k0_las` scaling as a cause
(D015).

**Rejected alternatives:** (a) *correct eq. (xsec) in `xigma.tex` directly*; (b) leave the
finding in this repo only.

**Rationale:** (a) is the author's call in more than one sense — the paper is the physics
authority (§0/P14), and *where* the factor belongs is a genuine choice: it can sit in the
prefactor, be absorbed into the normalization of *R*, or into the definition of *U*. The
manuscript is therefore annotated, not edited: LaTeX comments at eq. *(xsec)* and eq.
*(main)* stating the discrepancy and the derivation, leaving the typeset output unchanged.
(b) would leave the paper and the code disagreeing with nothing recording it in the place
the physics is decided — the exact failure mode §0 exists to prevent.

---

### D027 — Review-round corrections to Phase 2.5, and one committed file that should not have been

A review of `42e9609..HEAD` returned nine findings; all nine were reproduced and fixed.
Four are worth recording because the reasoning, not just the patch, matters.

**(a) The window metric's significance threshold was raised 16000x while fixing something
else — reintroducing D023's blindness on the other side.** D023 correctly moved the floor
from a fraction of the *mean window* to a fraction of the *total*, so it stops moving when
the binning is refined. It also, in the same line, changed the fraction from `1e-6` to
`1e-3`, which was never argued for. Measured consequence: a candidate that dropped a real
spectral feature worth 0.07% of the yield scored **exactly zero** on both reported numbers
— a perfect shape match for a lost feature. The two changes are separable and only the
first was justified. The floor is now `SIGNIFICANT_FLUX_FRACTION = 1e-4`, and it serves as
both the significance test and the denominator guard, which keeps the reported number
interpretable at both ends: a window missing entirely reports `1.0`, and spurious flux
against a zero reference reports how many significance-units of it there are.
`tests/test_stage0_delta.py` pins the sensitivity.

**(b) `_captured_fraction` integrated the wrong integrand.** It used the Lorentz factor
alone, giving the tidy `X/(1+X)`, while delta's own integrand carries the polarization
factor too. Since it is the correction the §9.1 arbitration divides by, the error appeared
as a cone-dependent drift in a number that is supposed to be constant. The closed form is
now `1.5 * [X/(1+X) - 1/3 + (1+X)^-2 - (2/3)(1+X)^-3]` (0.917 at four cone widths, against
the 0.941 it claimed). Renamed to `captured_fraction` and made public — it is a statement
about the method, not an implementation detail.

**(c) `check_normalization` integrated bin-centre densities with the trapezoid rule**,
dropping half of the first and last bin. That is why `anchor_ratio` read 0.9937 for a
quantity documented as exactly one, and the same bias sat uncorrected in the §9.1 headline
number. Both integrals are now midpoint sums; the anchor reads 1.000002.

With (b) and (c) fixed, the residual in `deviation` is `+0.41%` at eight cone widths and
is **grid geometry**: the square grid reaches `sqrt(2)` further in its corners than the
disc the correction assumes. A monoenergetic zero-divergence beam reproduces it to within
0.01 percentage points, so it is not beam spread. `identity_checks` now runs at eight cone
widths rather than four, where that residue is a fifth of the gate's budget instead of
three quarters.

**(d) A machine-specific hook file was committed.** `.claude/settings.json` — written by
the graphify tooling, hardcoding `/home/alexander/.local/bin/graphify` as a *PreToolUse*
hook (a Claude Code event name, not a repo symbol) on essentially every tool call — was swept into a commit by an unreviewed `git add
-A`, along with an auto-appended `AGENTS.md` section asserting the repo "has a knowledge
graph at graphify-out/" one commit after that directory was gitignored. Both are fixed:
the settings moved to `.claude/settings.local.json` (gitignored — *settings.json* is the
*shared* file by convention, so machine-specific hooks do not belong in it), and the
`AGENTS.md` claim now says the graph must be built locally because it is deliberately not
committed. **The lesson is the process one:** `git add -A` after running a tool that
writes config is how someone else's checkout ends up invoking a binary that does not exist
on their machine.

**Rejected alternative for (a):** *keeping `1e-3` and widening the golden tolerance
instead*. That treats a reporting defect as a calibration question. The problem was never
that 0.07% passed a 2% gate — it should — but that the metric announced perfect agreement
where there was a real difference, which makes it useless for tracking drift and a false
pass under any tighter tolerance later.

---

### D028 — `deposit_table` bins directly onto `ahat`; the predecessor's fixed-range `retarget_a0`/`a0_kind` regrid is not ported, but the underlying peak-independence it exploited is — for both `ahat` and `luminosity`

**Superseded by D032**, within the same session: the physics argument for *why* a fixed,
non-uniform target grid earns its cost (concentrating resolution near `ahat_max`, where
the redshift correction is significant) reverses this entry's "nothing in this repo needs
[the regrid]" conclusion. `deposit_table`/`Table.ahat_edges`-as-direct-deposit-target,
named throughout below, no longer exist — replaced by `stages.deposit_shape_table` +
`stages.retarget_ahat`. Left below as the historical record of the reasoning that held for
the rest of this Phase 3a session, not as a description of current code.

`GRAND_PLAN.md` §4.2 named `retarget_a0` (a0-axis rebin, no re-deposition) as in-scope for
Stage 1. What is *not* built is the predecessor's Grid4D/W-matrix conservative regrid —
a mechanism for squeezing tables from different peak-a0 runs onto one *fixed* target bin
range so they stayed mutually comparable, which nothing in this repo needs (each
`Collision` builds its own table fresh, D030). What *is* kept, and is the actual physics
the predecessor's mechanism rested on: for a fixed envelope shape,
``a0_local(t) = a0_peak * envelope(t)`` is exactly linear in the peak, so both
`TrajectorySamples.ahat` (`a0_peak**2 * a0_shape`) and `TrajectorySamples.luminosity`
(proportional to ``sum(a0_local**2)``) scale as ``a0_peak**2`` for the *same cached
trajectories* — `retargeted_ahat`/`retargeted_luminosity` are that rescale, and
`stages.deposit_table(samples, a0_peak=...)` deposits both, so a different pulse energy's
redshift *and* total photon count are a fresh Stage 1 deposit from cached Stage 0 samples,
no rerun.

(`retargeted_luminosity` was missing from the first cut of this decision — `deposit_table`
deposited the retargeted `ahat` but the *original* `luminosity`, so a retargeted table's
`total_weight` did not move with `a0_peak` at all. Caught by a question about exactly this
reasoning; `tests/test_stage1_stage2.py::
test_retargeted_luminosity_matches_a_fresh_stage_0_run_at_that_a0_peak` cross-checks the
fix against an actual second Stage 0 run at a doubled pulse energy, agreeing to 0.1%.)

**Why re-deposit rather than rescale the existing table in place.** Because the ahat axis
and the samples deposited into it scale by the *same* factor, a table retargeted to a new
peak a0 could in principle be produced by rescaling `Table.ahat_edges` alone (binning is
invariant under a uniform positive rescale of both data and edges by the same factor) —
touching no array at all, cheaper than even a fresh deposit. Not built: nothing calls
`deposit_table(a0_peak=...)` from production code today (`Collision._table()` always asks
for the pulse's own a0), the trick does not hold in the degenerate zero-divergence-beam
branch of `_uniform_edges` (its fallback padding does not scale linearly), and a fresh
deposit is already measured cheap next to Stage 0
(`test_deposition_is_cheap_next_to_stage_0`, *DEFAULT_TABLE_BINS*, 50k particles). Revisit
if a real caller (a pulse-energy scan) needs the extra speed.

**Consequence for §5.** The illustrative recompute-cost table's "pulse energy → a0 |
REUSE_INTERMEDIATES | Stage 1 a0-axis retarget (no re-deposition)" row assumed the ported
regrid mechanism. `GRAND_PLAN.md` v0.14 restates it as "Stage 1 re-deposit from cached
Stage 0 samples (measured cheap)" — same tier, different mechanism. `XigmaEngine.
recompute_costs` (`engine.py`) does not yet claim this tier for pulse energy regardless
(see D030): the `REUSE_INTERMEDIATES` claim needs a live consumer that knows a laser edit
was pulse-energy-only and maps it to the corresponding `a0_peak`, and none exists until a
GUI or scan helper does that mapping.

**Rejected alternative:** *port `retarget_a0`/Grid4D/`a0_kind` verbatim*. Would add a
W-matrix conservative-regrid module with no measured cost it avoids, contradicting the
predecessor's own two build-then-delete-then-reject pattern this project already tracks
for other speculative machinery (P6/P7/P9/P10/P11).

---

### D029 — Stage 2's numpy kernel ports the predecessor's brute-force grid quadrature, not the GPU importance sampler; `KERNEL_NORMALIZATION_CONSTANT` isolates the pending §9.1 factor

`stages.spectrum_from_table`/`angular_spectrum_from_table`/`spectrum_in_angular_range`
port the predecessor's *reference.py*'s `spectrum_from_table` — a direct sum over the
table's own `(theta_x, theta_y, ahat)` cells with quadrilinear-in-gamma interpolation
(`stages._interp_gamma`) — as the **production** numpy path, not the ~550-line
*cupyx.jit*/CPU-`numba` ring-and-arc importance sampler (*spectrum4d.py*/
*spectrum4d_cpu.py*). The predecessor's own audit (*xigma_passport.md* §8) already rated
that sampler trust-level C, with 3x-30x variance in sparse/narrow-angle configurations —
porting it would import a known-noisy path as this phase's only implementation, with no
GPU to validate it against yet.

`cupy`/`numba` backends are gated exactly like Stage 0's `_check_backend` (`stages.py`,
Phase 2.5): declared in `schema.py`'s `scheme`/eventual-backend vocabulary but rejected at
call time until a real kernel exists, rather than silently falling back to numpy under a
GPU-sounding flag.

**The pending §9.1 constant** (`stages.KERNEL_NORMALIZATION_CONSTANT = 1.5`) is pi-free,
matching the predecessor's own kernel math exactly (its `coef=1.5`, explicitly documented
pi-free in both *spectrum4d.py* and *reference.py*). The predecessor's ~2π gap is *not*
inside this constant — it is the same gap D025/D026 already traced, between this
kernel's differential form and the table-free `angle_integrated_spectrum` shape, both of
which this repo's engine now computes (`stages.angle_integrated_spectrum` for
`Collision.spectrum`, the table kernel for everything else). `run.py::identity_checks`'s
fourth leg (Stage 2 kernel vs `delta.resonance_spectrum` at one point) measures ~1.00
across the scenario bank precisely because both sides carry the identical pending factor —
evidence the constant is isolated correctly, not evidence the §9.1 question is closed.

**A resolution artefact, found while testing this constant, not caused by it: evaluating
`spectrum_from_table` exactly at a beam's own angular centre against a `scheme="nearest"`
table aliases against that table's cell boundaries** — measured ratios from 0.48 to 1.67
against `delta` across theta-bin counts 10-150 at fixed particle count
(`tests/test_stage1_stage2.py`). `scheme="cic"` (already in scope, §4.2) removes it,
holding within a few percent from 40 to 250 theta bins. `run.py`'s fourth identity leg and
the `test_stage1_stage2.py` regression test both use CIC for this reason; `nearest` stays
the schema default (cheaper, and Stage 1's own conservation is scheme-independent).

**Rejected alternative:** *port the GPU sampler now, run it CPU-side via cupy's numpy
fallback or a hand rewrite*. Copies a known-noisy algorithm before there is hardware to
validate it against, and duplicates ~550 lines this phase's exit criteria (`GRAND_PLAN.md`
§11, "Stage architecture tests green; placeholders documented") do not ask for.

---

### D030 — `Collision`'s cache is per-instance memoization only; no cross-call staleness detection

`Collision.build_overlap()`/`_table()` memoize on `self` — one `Collision` is built from
one fixed `InteractionParameters` + xigma `Parameters`, and calling either method twice
returns the cached value. There is no hash-based "did the caller's *new* interaction
differ only in field X" detection across *different* `Collision` instances.

**Why.** §5's GUI grey-out/cheap-requery model needs a live object that survives repeated
edits and decides, per edit, which cached stage to keep — that consumer is Phase 6 (or a
scan helper), and building the cross-call detector now with nothing to drive it is the
speculative machinery P6 rejects. What *is* needed now — sharing Stage 0/1 work across the
several outputs one `run()` call requests — is exactly what per-instance memoization
gives, and `XigmaEngine.run()` builds one `Collision` per call, matching P3's "opaque by
contract."

**Consequence:** `XigmaEngine.recompute_costs` (`engine.py`) declares only `n_e` (handled
at the `io` level, `InteractionParameters.with_charge`/`Results.scaled`, §3.5 — no engine
run at all) as cheap. Collimation-window and pulse-energy cheap paths from §5's
illustrative table are not claimed: they are true only if the *caller* keeps reusing one
`Collision`, which nothing in this phase does yet. Everything else defaults `FULL_RERUN`
(`base.py`'s documented default), which is the honest statement of what Phase 3a actually
wired.

**Rejected alternative:** *hash-key every stage's inputs now, so `Collision` can detect
"only pulse energy changed" across instances*. Requires a policy for constructing that key
from an arbitrary `LaserField` (P15) that the protocol does not provide, has no test or
caller to justify it in this phase, and would very likely need reworking once Phase 6
defines what the GUI actually keeps alive across a Calculate.

---

### D031 — `XigmaEngine` is not passed to `run_suite()` by `validation.run.main()`

The engine exists (`engines/xigma/engine.py`) and is tested
(`tests/test_xigma_engine.py`, `tests/test_stage1_stage2.py`), but
`python -m gammaforge.validation.run` does not exercise it: `main()` still calls
`run_suite()` with no engines, same as before this phase.

**Why.** `run_suite(engines=[...])` already works — `invariance.engine_checks` and
`golden.compare_to_golden` are engine-generic (Phase 2) — but `run_engine` runs an engine
against the scenario's own `Target.outputs`, which is `scenarios._DEFAULT_OUTPUTS`:
`COLLIMATED_SPECTRUM` at `(64, 16, 16)` and `ANGULAR_DISTRIBUTION` at `(64, 64)`, sized for
the predecessor's GPU importance sampler (§12's own risk row: "measured 27s @ 64 energy
bins on CPU... linear in n_energy"). This phase's numpy kernel is also linear in the
*angular* grid size (a plain Python loop over observation points, §4.2/D029), so
`COLLIMATED_SPECTRUM` alone measured 36.6s at that resolution and 100k particles — times
`engine_checks`' four full runs per scenario, times three scenarios, `python -m
gammaforge.validation.run` would go from ~3s to tens of minutes. That is real, expected
Calculate cost for a deliberate query (§12: "expected, not a defect"), not a defect in a
suite meant to be run routinely.

`run.py::identity_checks` gained a fourth leg instead (D029): Stage 2's table kernel
against `delta` at one point, at the identity harness's existing 2000-particle,
small-table scale — the thing that actually exercises `stages.deposit_shape_table`/
`stages.retarget_ahat` (D032)/`angular_spectrum_from_table` in the routine suite.

**Rejected alternatives:** *wire the engine in at a reduced, suite-only resolution* — would
need a second, ad-hoc `Target` variant that no golden or scenario elsewhere uses, and
still would not be testing what the scenario bank's actual outputs cost; *reduce
`_DEFAULT_OUTPUTS`'s resolution to something numpy can afford* — would silently change what
every future engine (including a real GPU xigma) is validated against, for a limitation of
this phase's kernel alone. Revisit once cupy/numba land (D029) or Phase 7's "full scenario
bank" exit criterion is actually being worked.

---

### D032 — Stage 1 deposits onto `a0_shape`, not `ahat`; a conservative regrid (`retarget_ahat`) onto a fixed, non-uniform `ahat` axis replaces the direct deposit — supersedes D028

Discussion with the project's author (a physicist) surfaced that D028's direct-onto-`ahat`
deposit was wrong for where the resonance physics actually needs resolution. `ahat` enters
`s_res = gamma**2/(1+ahat+gamma**2*r**2)`; the redshift only matters where `ahat` is
comparable to 1, near a pulse's peak, while the bulk of a bunch's trajectories sit at much
smaller `ahat` where the correction barely perturbs anything. A grid whose bin density
follows wherever the sampled data happens to span (D028's `_uniform_edges` on the raw
`ahat` array) puts no more resolution near the peak than anywhere else. A **fixed**,
non-uniform target grid — dense near `ahat_max`, coarse toward `ahat_min`, everything below
`ahat_min` folded into one floor bin — is the right shape, and reaching it over the whole
a0-independent population (not tied to one pulse) is exactly what the predecessor's
`retarget_a0` conservative regrid was for (D028 declined to port it, on the grounds that
nothing needed a *fixed target range* — true for cross-run comparability, the predecessor's
own stated reason, but not for concentrating resolution, which D028 did not consider).

**What changed, concretely:**

- **`stages.ShapeTable`** (new type, not a `kind` string flag — matches this codebase's
  existing aversion to stringly-typed discriminators, `Axis`/`OutputKind` enums,
  `io/schema.py`'s "never a stringly dict") replaces direct deposit onto `ahat`.
  `stages.deposit_shape_table` bins Stage 0's `TrajectorySamples.a0_shape` (already
  peak-independent, already bounded — no dynamic-range problem) onto a fine, uniform,
  linear grid (`DEFAULT_SHAPE_BINS = (48, 48, 48, 96)`), with native `luminosity` as the
  deposited weight and the run's own `a0_peak` recorded as `source_a0_peak`. One deposit
  per `TrajectorySamples`, reusable for any peak a0.
- **`stages.retarget_ahat`** (adapted from the predecessor's `retarget_a0`, verbatim
  overlap-weighted-mass-transfer structure — `deposition.py:423-511` in the old repo) turns
  a `ShapeTable` plus one peak a0 into the `Table` (unchanged name, now exclusively the
  `ahat`-axis, Stage-2-ready result) Stage 2 queries. Two rescales, both exact: the source
  edges scale by `a0_peak**2` (`ahat = a0_peak**2 * a0_shape`), and the deposited mass
  rescales by `(a0_peak / source_a0_peak)**2` — the same relation `retargeted_luminosity`
  (D028) already established, needed here because, unlike the predecessor's `retarget_a0`
  (which only ever retargeted onto its own run's laser), this one is meant to serve a
  genuinely different peak a0 than the one Stage 0 ran at.
- **Target grid law** (`stages._ahat_target_edges`), log-spaced in distance from the top —
  bin width shrinking monotonically toward `ahat_max`, the opposite of what a plain
  logarithmic axis gives (constant *relative* width means bins widen in absolute terms
  toward the top; this repo wants the reverse, dense absolute resolution at the top):
  ```
  v_i = (ahat_max - ahat_min) * 10**(-decades * i / n_bins),  i = 0..n_bins
  ahat_i = ahat_max - v_i                                     (edges[-1] snapped to ahat_max)
  ```
  The snap widens the single top bin — negligible at `decades >= 3`, visible at the
  `decades=1` default below (the top bin ends up *wider* than its neighbour, a deliberate,
  bounded exception `tests/test_stage1_stage2.py::
  test_ahat_target_edges_bin_widths_shrink_toward_the_top` checks explicitly, not a defect).
- **Defaults, tuned against the actual scenario bank**, not re-derived from the
  predecessor's *DEFAULT_A0_MAX*/`retarget_a0` defaults, which were sized for a different
  bank: `ahat_max = 0.5` (kept — the predecessor's value, generous headroom over this
  bank's measured `ahat` maxima: `baseline` 0.019, `low_a0` 0.0019, `near_a0_max` 0.095),
  `ahat_min = 0.0`, `n_bins_ahat = 32` (kept). **`ahat_decades = 1.0`, down from an initial
  `decades=3` (matching the predecessor's implicit concentration) that was checked
  numerically and rejected**: at `decades=3`, bin 0 alone spans `[0, 0.097]` — wider than
  `near_a0_max`'s entire `ahat` range — so every scenario in the bank collapsed into the
  floor bin, resolving nothing and defeating the entire point of the change. Swept
  `decades` against the bank's actual mean/max `ahat` (measured this session) before
  picking `1.0`: `near_a0_max` spreads across bins 1-2 (real structure), while `low_a0`
  correctly stays concentrated near the floor — its `ahat` (max 0.0019) genuinely is small
  enough that the redshift correction barely matters there, which is squashing working as
  intended, not a resolution failure.
- **Truncation.** Target `ahat` bins the rescaled source data's true maximum never reaches
  carry **exactly zero** mass (the overlap-weighted transfer cannot place mass where no
  source bin overlaps) — `retarget_ahat` sums the regridded mass over the other three axes,
  finds the last populated `ahat` bin, and slices the mass array/`ahat_edges` down to it before
  returning (keeping >= 1 bin in the degenerate all-zero case). Free: `spectrum_from_table`'s
  cell sum is unaffected in value, only in how many always-zero terms it evaluates —
  `tests/test_stage1_stage2.py::test_retarget_ahat_truncation_does_not_change_the_kernel_output`
  checks the truncated and explicitly-zero-padded tables agree exactly. Measured effect: the
  fourth `run.py::identity_checks` leg (D029) and the `XigmaEngine.run()` smoke test both got
  noticeably faster, since most configurations populate only a handful of the 32 target bins.
- **`Table`'s old *bin_volume* property removed** (no longer generally correct once `ahat_edges` is
  non-uniform; confirmed unused by production code). `Table` gained `ahat_widths`
  (per-bin array, `np.diff(ahat_edges)`) and `gamma_theta_cell_area` (the still-uniform
  2-axis scalar) in its place; `spectrum_from_table` folds `ahat_widths` into the cell sum
  itself rather than multiplying one global scalar in afterward.
- **`Collision`** gained a `_shape_table` singleton cache (mirrors `build_overlap`'s
  pattern) alongside the existing `_tables` dict; `_table(a0_peak)` now calls
  `retarget_ahat(self._shape(), a0_peak, ...)`. A genuine improvement over D028's
  "re-deposit from scratch per `a0_peak`": `retarget_ahat`'s cost is independent of
  `n_particles`, so multiple `_table()` calls on one `Collision` (e.g. a future pulse-energy
  scan) now pay the `n_particles`-scale deposit cost once, not per `a0_peak` — exactly the
  win D028 said nothing currently needed. Cache-key staleness (a grid-shape parameter
  changing without a new `a0_peak` on a long-lived `Collision`) falls under D030's
  already-documented scope ("cheap to reuse, not smart about being replaced") without new
  reasoning — the new `n_bins_a0_shape`/`ahat_min`/`ahat_max`/`ahat_decades` parameters are
  not a new gap, just more instances of the same one.
- **`schema.py`** gained `n_bins_a0_shape` (default 96), `ahat_min` (0.0), `ahat_max` (0.5),
  `ahat_decades` (1.0); `n_bins_ahat`'s existing key is kept but its default moves 12 -> 32
  and its meaning shifts from "direct-deposit bin count" to "retarget target-grid bin
  count." The cross-field constraint `ahat_max > ahat_min` has no home in `FieldSpec` (no
  cross-field hook exists in `io/schema.py`) and is enforced inside `retarget_ahat`/
  `_ahat_target_edges` at call time instead, matching where the predecessor put the
  identical check — not worth building general cross-field schema validation for one case.

**Rejected alternative:** *rescale the existing direct-deposit table's edges in place for a
new peak a0, without a full regrid* (D028's own closing suggestion, since binning is
invariant under a uniform positive rescale of both data and target edges by the same
factor). Would have been cheaper still, but only works when the target grid *is* just a
scaled copy of the source — which stops being true the moment the target law is a fixed,
non-uniform shape rather than "wherever this particular a0_peak's data happens to land."
The whole point of this entry is wanting that fixed, physically-motivated shape, so the
pure-rescale trick is no longer available regardless of its cost.
