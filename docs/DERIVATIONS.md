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

**Scope.** Head-on (`theta_xz = theta_yz = 0`, so `k_hat = -z` and `u = -z`) and no
flying focus (`beta_ff = 0`). Both are enforced by `overlap_yield`, not assumed silently:
a crossing angle is `GRAND_PLAN.md` §9.3's open item, and `beta_ff != 0` makes
`u_spot = u + beta_ff ct` time-dependent, which destroys the time integration in §A.3.

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
