# Derivations

Long-form physics derivations that back specific code, kept out of docstrings because
they need more than a paragraph. Each section states its status plainly: whether it is
**implemented** or **pending author review**, and what evidence backs it.

> **Merge note.** A parallel Phase 3b session created a file of the same name holding
> §9.2/§9.3 material (numbered sections 0–3, all *pending review, none implemented*).
> This section is lettered (§A) precisely so the two can be concatenated without
> renumbering — resolve any add/add conflict by keeping both, in either order.

---

## A. The Gaussian luminosity overlap integral

**Status: implemented** as `gammaforge.engines.analytical.formulas.overlap_yield`, and
used by `AnalyticalEngine`. Every step below is either a Gaussian integral or a standard
identity, and the final result is checked numerically against an independent closed form
in `tests/test_analytical.py`. This is not a new physical model — it is the textbook
luminosity overlap for two Gaussian bunches, carried out without the round-beam and
aligned-foci simplifications the predecessor's version made.

### A.1 What is being computed

The photon yield is the Thomson cross-section times the space-time overlap of the two
densities, weighted by the relative-approach factor:

    N = sigma_T (1 + beta_0) c  Int dt Int d^3r  n_e(r, t) n_L(r, t)

`(1 + beta_0) c` is the head-on Møller flux factor for an electron of velocity
`beta_0 c` meeting a counter-propagating photon. The predecessor used `2c`; keeping
`beta_0` exact costs nothing and is the only place the derivation is not a pure identity
at `beta_0 = 1`.

Both densities are those `gammaforge.io` already defines, not idealizations invented
here. For the pulse this is exactly `GaussianParaxialLaser.photon_density`:

    n_L = N_L / ((2 pi)^{3/2} s_1 s_2 s_ct) exp(-xi_1^2/(2 s_1^2) - xi_2^2/(2 s_2^2) - (u - ct)^2/(2 s_ct^2))

with `u = r . k_hat`, `s_i` the spot sizes from `spot_sizes(u_spot)`, and `xi_1`/`xi_2`
the coordinates along the focusing axes. For the bunch:

    n_e = N_e / ((2 pi)^{3/2} sigma_ex(z) sigma_ey(z) sigma_ez) exp(-x^2/(2 sigma_ex^2(z)) - y^2/(2 sigma_ey^2(z)) - (z - beta_0 ct)^2/(2 sigma_ez^2))

**Scope.** §A.2–§A.5 are written head-on (`theta_xz = theta_yz = 0`, so `k_hat = -z` and
`u = -z`), which is the clearest way to see the structure; **§A.6 generalizes to a
crossing angle** and is what `overlap_yield` actually implements. Excluded throughout:
`beta_ff != 0`, which makes `u_spot = u + beta_ff ct` time-dependent and destroys the
time integration in §A.3 — `overlap_yield` raises on it rather than assuming it away.

### A.2 The transverse integrals: two Gaussians, one determinant

Write each transverse profile as a 2x2 covariance in **lab** x/y. The bunch is diagonal:

    C_e(z) = diag(sigma_ex^2(z), sigma_ey^2(z))

The pulse is generally **not**, because `psi_focus` rotates its focusing axes within the
transverse plane. With `R` the rotation by `psi_focus`,

    C_l(z) = R diag(s_1^2(-z), s_2^2(-z)) R^T

— evaluated at `u = -z`, so the focal offsets `z_fx`/`z_fy`, which are measured *along*
`k_hat`, put the pulse's waists at lab `z = -z_fx` and `z = -z_fy`.

For two centered 2D Gaussians the transverse overlap is a standard identity:

    Int d^2r [N(0, C_e)][N(0, C_l)] = 1 / (2 pi sqrt(det(C_e + C_l)))

using `det(C_e^{-1} + C_l^{-1}) = det(C_e + C_l) / (det C_e det C_l)`. This single scalar
is `overlap_det`, and it is where the entire non-round generalization lives: unequal x/y
sizes, unequal x/y focusing, astigmatic laser waists and the rotation between the two
ellipses are all carried exactly. It collapses to the familiar `1/(2 pi sigma_0^2)` only
when *both* ellipses are circular.

### A.3 The time integral: also Gaussian

Only the two longitudinal exponents depend on `t`. With `w = ct`, they are Gaussians in
`w` centered at `z/beta_0` (width `sigma_ez/beta_0`) and at `-z` (width `sigma_lz`), and

    Int dw exp(-(w-a)^2/(2p^2) - (w-b)^2/(2q^2)) = sqrt(2 pi) pq/sqrt(p^2+q^2) exp(-(a-b)^2/(2(p^2+q^2)))

gives, with `D^2 = sigma_ez^2 + beta_0^2 sigma_lz^2`,

    sqrt(2 pi) (sigma_ez sigma_lz / D) exp(-z^2 (1 + beta_0)^2 / (2 D^2))

The factor `(1 + beta_0)` in the exponent is the closing-speed compression: the two
pulses sweep past each other at `(1 + beta_0) c`, so a longitudinal spread `D` in
*arrival time* corresponds to a spread `sigma_z_eff = D / (1 + beta_0)` in *collision
position*.

### A.4 The one integral that does not close

Assembling §A.2 and §A.3, everything cancels except a single longitudinal quadrature:

    N = sigma_T (1 + beta_0) N_e N_L / (2 pi sqrt(2 pi) D)
        * Int dz exp(-z^2 / (2 sigma_z_eff^2)) / sqrt(det(C_e(z) + C_l(z)))

There is no closed form in general: `det(C_e + C_l)` is a quartic in `z` once the two
ellipses are unequal and rotated, and `Gaussian / sqrt(quartic)` is not elementary. But
the integrand is **strictly positive and smooth**, with no cancellation anywhere, so a
fixed grid converges monotonically and fast — this is a millisecond-scale evaluation, not
a Monte Carlo.

The one numerical subtlety is resolution, not accuracy: the four hourglass/Rayleigh
scales can be far shorter than `sigma_z_eff` (in the baseline scenario `z_R` = 1.2 mm
against `sigma_z_eff` ≈ 4.7 mm), and displaced foci move that structure off `z = 0`.
`_overlap_grid` therefore unions a grid over the Gaussian support with one refined window
per waist, so convergence does not depend on where the foci happen to sit.

**Reduction.** Round, aligned, `alpha = 0`: both covariances are circular, so
`sqrt(det) = Sigma_0^2 (1 + z^2/L^2)` with `Sigma_0^2 = sigma_e0^2 + sigma_l0^2` and

    1/L^2 = (sigma_e0^2/beta_0*^2 + sigma_l0^2/z_R^2) / Sigma_0^2

Then `Int exp(-p z^2)/(1 + z^2/c^2) dz = pi c erfcx(sqrt(p) c)` collapses the quadrature
to

    N = sigma_T N_e N_L nu erfcx(nu) / (2 sqrt(pi) Sigma_0^2),   nu = L (1 + beta_0) / (sqrt(2) D)

which is *exactly* `estimate_yield`'s closed form. `test_overlap_yield_reduces_to_the_
round_beam_closed_form` checks this against an independently-coded closed form, and
`test_electron_hourglass_matches_io_bunchs_own_drift` pins the `alpha` sign against
`io.bunch._drift_fit` rather than against this document.

Accuracy is a resolution question, so quote it with the grid: the reduction holds to
5e-10 at `n_quad = 32001` (what the test asserts). At the schema default of
`n_quad_overlap = 2001` the quadrature error is ~1e-7 on the round case and ~1e-6 on a
displaced/astigmatic/rotated one — far below any physical uncertainty in the inputs, but
not the machine-precision figure, and worth raising the knob for a convergence study.

### A.5 A discrepancy this derivation exposes

The reduction in §A.4 identifies the laser term in `nu` as `sigma_l / z_R` exactly. Both
this repository's `GaussianParaxialLaser.rayleigh_x` and the predecessor's own pulse class
define `z_R = 4 pi sigma^2 / lambda` (`w0 = 2 sigma`), giving a laser divergence
`lambda / (4 pi sigma)`.

The predecessor's `analytical.py` — and therefore the faithful port in `estimate_yield` —
instead uses `lambda^2 / (pi^2 sigma^2)`, a divergence of `lambda / (pi sigma)`: **4x too
large**, i.e. 16x in the squared term. The predecessor is inconsistent with its *own*
laser model; this was not introduced by the port to CGS.

It is not a small effect. The baseline scenario's hourglass is almost entirely
laser-driven (laser divergence 3.3e-2 rad against the bunch's 5.0e-6 rad), so the error
passes through nearly in full: **the closed form underestimates the baseline yield by a
factor of 3.285**. `overlap_yield` uses `rayleigh_x()`/`rayleigh_y()` and so is free of
it; `estimate_yield` is left as-is, carrying a warning in its docstring, because its
value is precisely that it reproduces the predecessor (`DECISIONS.md` D040).

Nothing in the pre-existing test suite was sensitive to this: the predecessor pin checks
port fidelity, and the Thomson-limit anchor deliberately drives `nu -> infinity`, which
removes the hourglass term altogether.

### A.6 Crossing angle

**Status: implemented.** A crossing angle changes the *geometry* of the overlap, which is
a solvable Gaussian problem. It is a separate question from `GRAND_PLAN.md` §9.3, whose
open item is the polarization structure of the **emission kernel** — what spectrum comes
out at an angle. The §9.2/§9.3 notes elsewhere in this file record that the
relative-velocity factor and the resonance frequency are already general in the paper, so
nothing there blocks the luminosity.

Write the whole exponent as a quadratic form instead of tracking terms one at a time. At
fixed `t` the two densities contribute

    M_e(z) = xx^T/sigma_ex^2(z) + yy^T/sigma_ey^2(z) + zz^T/sigma_ez^2
    M_l(u) = f1 f1^T/s_1^2(u) + f2 f2^T/s_2^2(u) + k k^T/s_ct^2

with `(k_hat, f1, f2)` from `lab_frame_axes` — head-on `k_hat = -z` recovers §A.2. The
`t`-dependence is entirely in the two longitudinal terms, and collecting it gives

    E = r^T M r / 2 - ct (g . r) + h (ct)^2 / 2,
    g = beta_0 zhat/sigma_ez^2 + khat/s_ct^2,   h = beta_0^2/sigma_ez^2 + 1/s_ct^2

so the time integral is Gaussian and simply replaces `M` by

    M' = M - g g^T / h

Integrating the two transverse directions out of `M'` leaves its upper-left 2x2 block `A`
and the Schur complement `S = M'_zz - b^T A^-1 b` with `b = (M'_xz, M'_yz)`:

    N = sigma_T (1 + beta_0) N_e N_L sqrt(2 pi / h) / (4 pi^2 sigma_ez sigma_lz)
        * Int dz exp(-S(z) z^2 / 2) / (sigma_ex sigma_ey s_1 s_2 sqrt(det A(z)))

**This is not a second code path.** Head-on it collapses to §A.4 identically: `A` becomes
diagonal, `b = 0`, `sigma_ex sigma_ey s_1 s_2 sqrt(det A) = sqrt(det(C_e + C_l))`, and

    S = ab'(1+beta_0)^2/(beta_0^2 a + b') = (1+beta_0)^2/D^2,   a = 1/sigma_ez^2, b' = 1/s_ct^2

with the prefactors matching through `4 pi^2 / sqrt(2 pi) = 2 pi sqrt(2 pi)`.
`overlap_yield` therefore evaluates one expression for every geometry.

**The one approximation.** The bunch's hourglass varies along `z`; the pulse's varies
along `u = k_hat . r`. With a crossing angle these are different directions, so an exact
reduction leaves a **2D** quadrature — "everything analytic but one integral" is a
head-on statement. The spot sizes are therefore sampled at `u = (k_hat . zhat) z`,
dropping `delta = k_x x + k_y y`. Nothing in the *exponent* is approximated, including the
`xi_1 ~ x cos - z sin` term that produces the entire crossing-angle suppression; only the
argument of the slowly varying widths is.

The relevant bound is `delta / z_R`, **not** `delta / sigma_z` — the widths vary on the
Rayleigh scale — and it is not always small: a 0.4 rad crossing with a 2 um waist and a
200 um bunch puts it at 1.6. Measured rather than argued
(`test_crossing_angle_width_sampling_approximation_is_negligible`), the yield still moves
by `< 1.3e-4` even there, because the dropped term enters an *even*, slowly varying
prefactor, so its first-order effect cancels; a coherent `+/- delta` probe is already a
conservative overestimate of a term whose true mean is zero.

**Verification** — three independent checks, in increasing generality:

| check | what it isolates | agreement |
|---|---|---|
| constant-width closed form (3x3 determinant, no quadrature) | the crossing geometry alone | ~1e-14, to 0.4 rad, both planes |
| Piwinski `1/sqrt(1 + (sigma_s tan θ / sigma_perp)^2)` | that the suppression is the known physics | 1e-6 at 2 mrad (the formula is itself small-angle) |
| brute-force Monte Carlo over `GaussianParaxialLaser.photon_density` with real macroparticles | everything at once, sharing no algebra | few 1e-4, converged separately in particle count and time grid |

The suppression is large and worth internalizing. At the baseline scenario (3 mm bunch,
10 um spots) the yield falls by a factor 1.07 at 5 mrad, **2.18 at 20 mrad** and 5.77 at
50 mrad — the geometric cost of crossing dominates anything else in this model.

**What is still head-on.** The emitted spectrum. `AnalyticalEngine` normalizes `SPECTRUM`
to this yield, so with a crossing angle the slice's *integral* is right while its *shape*
is not — a combination that looks more correct than it is, which is why the engine reports
it on `Results.model_specific["warnings"]` rather than leaving it to `io.laser.validate`.
