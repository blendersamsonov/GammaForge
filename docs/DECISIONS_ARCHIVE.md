# GammaForge — Decisions Archive

Decisions whose reasoning is **settled history**: Phase 0–2.5 scaffolding that no longer
constrains anything being written today, plus entries later superseded. Nothing here was
deleted — it was moved, verbatim, so `DECISIONS.md` stays a working document.

**Read this file when** you are about to undo an early structural choice and want to know
whether it was considered, or when tracing why a superseded entry said what it did. For
everything else, `../DECISIONS.md` is the live file.

Superseded entries carry a pointer to what replaced them. The backtick convention here is
the same as the live file's, but the doc-staleness guard does **not** check this file
(`DECISIONS.md` D002): archived entries describe code that has since moved or gone, so
requiring their backticks to resolve would mean rewriting history to keep a test green.

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

---

### D024 — Stage 0 reads the whole laser through `a0_profile`

> **Superseded by D054 (2026-08-10).** Stage 0 now reads *intensity_profile* — the
> cycle-averaged `<a^2>` — and the constant is `4 pi`, not `8 pi`. **The decision below is
> unchanged in substance**: Stage 0 still takes the whole laser through one sampling method
> plus `active_region`, and the pulse energy still cancels. Only *which* method, because
> `<a^2>` is polarization-agnostic where `a0` carries a convention. Archived because its
> text names the old method and constant.

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

---

### D025 — Delta's ``2 pi`` is derived and reported, not corrected

> **Superseded by D033 (Phase 3b).** The factor is now applied rather than reported, and
> the identity harness expects one. What follows is the reasoning that was correct while
> the question was open — kept as the record of why the suite was gated at `2 pi` for two
> phases, not as live policy.

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

---
