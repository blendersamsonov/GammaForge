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
