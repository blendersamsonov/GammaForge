# Derivations

Long-form physics derivations that back specific code, kept out of docstrings because they
need more than a paragraph. Each section states its status plainly: whether it is
**implemented** or **pending author review**, and what evidence backs it.

Math is MathJax (`$…$` / `$$…$$`), so it renders in Obsidian and pastes into
`~/Work/Papers/2026/Compton-Numerics/xigma.tex` with only environment changes.

**How this file is organised.** Two parallel sessions wrote into it and the halves were
concatenated on merge, which is why the numbering is mixed. It is not arbitrary:

| | sections | status |
|---|---|---|
| **Lettered** | §A Gaussian luminosity overlap, §B flying focus, §C the polarization factor in $\hat a$ | **implemented**, each with its own independent check |
| **Numbered** | §0 the `ahat` discrepancy, §1 §9.2 ellipticity, §2 §9.3 crossing angle, §3 what each would change | §0 **resolved and fixed**; §1–§2 **pending author review, not implemented** |

§C and §0 are the same physics from the two ends — §C derives the cycle-average factor,
§0 reports the discrepancy it exposed in the code and records how it was settled
(`DECISIONS.md` D053). Read §C first if you want the derivation, §0 first if you want the
history.

**Notation.** The bunch travels along $+\hat{\mathbf z}$ with velocity $\beta_0 c$. The
pulse propagates along $\hat{\mathbf k}$, with transverse focusing axes
$\hat{\mathbf f}_1,\hat{\mathbf f}_2$ and its own longitudinal coordinate
$u=\hat{\mathbf k}\cdot\mathbf r$; head-on $\hat{\mathbf k}=-\hat{\mathbf z}$ and $u=-z$.
Bunch sizes are $\sigma_{ex}(z),\sigma_{ey}(z),\sigma_{ez}$; pulse spot sizes are
$s_1(u),s_2(u)$ and $s_{ct}=c\,\tau$. Crossing angle $\theta$ is measured from exact
counter-propagation, matching `theta_xz`/`theta_yz`.

---

## A. The Gaussian luminosity overlap integral

**Status: implemented** as `gammaforge.engines.analytical.formulas.overlap_yield`, used by
`AnalyticalEngine`. Every step is a Gaussian integral or a standard identity, and the
result is checked against an independent closed form, against the Piwinski limit, and
against a brute-force Monte Carlo (§A.10). This is not a new physical model — it is the
textbook luminosity overlap carried out without the round-beam, aligned-foci and head-on
simplifications the predecessor's version made.

### A.1 What is being computed

The photon yield is the Thomson cross-section times the space–time overlap of the two
densities, weighted by the relative-approach factor:

$$
N \;=\; \sigma_T\,(1+\beta_0)\,c \int\!\mathrm{d}t \int\!\mathrm{d}^3r\;
n_e(\mathbf r,t)\,n_L(\mathbf r,t)
$$

$(1+\beta_0)c$ is the head-on Møller flux factor for an electron of velocity $\beta_0c$
meeting a counter-propagating photon; with a crossing angle it generalises to
$c\,(1-\boldsymbol\beta\cdot\hat{\mathbf k})$, which is the same factor
$(1-\mathbf v\cdot\hat{\mathbf n}_0)$ that §2.1 of the §9.3 notes identifies as already
general in the paper. The predecessor used $2c$; keeping $\beta_0$ exact costs nothing.

Both densities are the ones `gammaforge.io` already defines. For the pulse this is exactly
`GaussianParaxialLaser.photon_density`:

$$
n_L \;=\; \frac{N_L}{(2\pi)^{3/2}\,s_1 s_2 s_{ct}}
\exp\!\left[-\frac{\xi_1^2}{2s_1^2}-\frac{\xi_2^2}{2s_2^2}
-\frac{(u-ct)^2}{2s_{ct}^2}\right],
\qquad \xi_i=\hat{\mathbf f}_i\cdot\mathbf r
$$

and for the bunch

$$
n_e \;=\; \frac{N_e}{(2\pi)^{3/2}\,\sigma_{ex}\sigma_{ey}\sigma_{ez}}
\exp\!\left[-\frac{x^2}{2\sigma_{ex}^2(z)}-\frac{y^2}{2\sigma_{ey}^2(z)}
-\frac{(z-\beta_0 ct)^2}{2\sigma_{ez}^2}\right]
$$

The transverse sizes evolve by the ordinary Twiss drift,
$\sigma_{ei}^2(z)=\epsilon_i\big[\beta_{0i}-2\alpha_i z+(1+\alpha_i^2)z^2/\beta_{0i}\big]$,
so the electron-side *foci displacement* needs no new parameter: it is already carried by
$\alpha_x,\alpha_y$, with the waist at $z_w=\alpha\beta_0/(1+\alpha^2)$.

**Scope of §A:** $\beta_{\rm ff}=0$. A flying focus makes the spot-size evaluation point
$u_{\rm spot}=u+\beta_{\rm ff}ct$ depend on time, which destroys §A.3's time integration —
**§B** treats that case and `overlap_yield` routes to it automatically.

### A.2 The transverse integrals: two Gaussians, one determinant

Write each transverse profile as a $2\times2$ covariance in **lab** $x,y$. The bunch is
diagonal, $C_e(z)=\mathrm{diag}\big(\sigma_{ex}^2(z),\sigma_{ey}^2(z)\big)$. The pulse
generally is **not**, because `psi_focus` rotates its focusing axes within the transverse
plane:

$$
C_l(z) \;=\; R(\psi)\,\mathrm{diag}\big(s_1^2(-z),\,s_2^2(-z)\big)\,R(\psi)^{\mathsf T}
$$

evaluated at $u=-z$, so the focal offsets $z_{fx},z_{fy}$ — measured *along*
$\hat{\mathbf k}$ — place the waists at lab $z=-z_{fx},-z_{fy}$. For two centred 2D
Gaussians,

$$
\int \mathrm{d}^2r_\perp\;\mathcal N(0,C_e)\,\mathcal N(0,C_l)
\;=\; \frac{1}{2\pi\sqrt{\det\!\big(C_e+C_l\big)}}
$$

using $\det\!\big(C_e^{-1}+C_l^{-1}\big)=\det(C_e+C_l)/(\det C_e\det C_l)$. This single
scalar is `overlap_det`, and it is where the entire non-round generalisation lives:
unequal $x/y$ sizes, unequal $x/y$ focusing, astigmatic waists and the rotation between
the two ellipses are all exact. It collapses to $1/(2\pi\sigma_0^2)$ only when *both*
ellipses are circular.

### A.3 The time integral: also Gaussian

Only the two longitudinal exponents depend on $t$. With $w=ct$ they are Gaussians centred
at $z/\beta_0$ (width $\sigma_{ez}/\beta_0$) and at $-z$ (width $s_{ct}$), and

$$
\int\!\mathrm{d}w\;
e^{-\frac{(w-a)^2}{2p^2}-\frac{(w-b)^2}{2q^2}}
=\sqrt{2\pi}\,\frac{pq}{\sqrt{p^2+q^2}}\,
\exp\!\left[-\frac{(a-b)^2}{2(p^2+q^2)}\right]
$$

gives, with $D^2=\sigma_{ez}^2+\beta_0^2 s_{ct}^2$,

$$
\sqrt{2\pi}\,\frac{\sigma_{ez}s_{ct}}{D}\,
\exp\!\left[-\frac{z^2(1+\beta_0)^2}{2D^2}\right]
$$

The factor $(1+\beta_0)$ is the closing-speed compression: the two pulses sweep past each
other at $(1+\beta_0)c$, so a spread $D$ in *arrival time* is a spread
$\sigma_{z,\mathrm{eff}}=D/(1+\beta_0)$ in *collision position*.

### A.4 The one integral that does not close

Assembling §A.2 and §A.3, everything cancels except a single longitudinal quadrature:

$$
\boxed{\;
N=\frac{\sigma_T(1+\beta_0)N_eN_L}{2\pi\sqrt{2\pi}\;D}
\int_{-\infty}^{\infty}\!\mathrm{d}z\;
\frac{\exp\!\big[-z^2/2\sigma_{z,\mathrm{eff}}^2\big]}
{\sqrt{\det\!\big(C_e(z)+C_l(z)\big)}}\;}
$$

There is no closed form in general: $\det(C_e+C_l)$ is a quartic in $z$ once the ellipses
are unequal and rotated, and $\text{Gaussian}/\sqrt{\text{quartic}}$ is not elementary.
But the integrand is **strictly positive and smooth**, so a fixed grid converges
monotonically and fast — milliseconds, not Monte Carlo.

**Reduction.** Round, aligned, $\alpha=0$: both covariances are circular, so
$\sqrt{\det}=\Sigma_0^2\big(1+z^2/L^2\big)$ with $\Sigma_0^2=\sigma_{e0}^2+\sigma_{l0}^2$
and

$$
\frac{1}{L^2}=\frac{1}{\Sigma_0^2}\left(\frac{\sigma_{e0}^2}{\beta^{*2}}
+\frac{\sigma_{l0}^2}{z_R^2}\right)
$$

Then $\int e^{-pz^2}/(1+z^2/c^2)\,\mathrm{d}z=\pi c\,\mathrm{erfcx}(\sqrt{p}\,c)$ collapses
the quadrature to

$$
N=\frac{\sigma_T N_e N_L}{2\sqrt{\pi}\,\Sigma_0^2}\;\nu\,\mathrm{erfcx}(\nu),
\qquad
\nu=\frac{L(1+\beta_0)}{\sqrt{2}\,D}
$$

which is *exactly* `estimate_yield`'s closed form. Accuracy, quoted with its grid: the
reduction holds to $5\times10^{-10}$ at $n_{\rm quad}=32001$; at the schema default of
$2001$ the quadrature error is ${\sim}10^{-7}$ round and ${\sim}10^{-6}$ on a
displaced/astigmatic/rotated case.

### A.11 Transverse and timing misalignment

**Status: implemented.** `GaussianParaxialLaser` carries `x_off`, `y_off`, `t_off`; every
consumer of the field inherits them because `_local_coordinates` subtracts them once.

Displacing the pulse means its exponent is a quadratic form in $(\mathbf v-\mathsf D)$
rather than $\mathbf v$, with

$$
\mathbf v=(x,y,z,ct),\qquad
\mathsf D=(x_{\rm off},\,y_{\rm off},\,0,\,c\,t_{\rm off})
$$

so that, writing $\mathbb M_l$ for the laser's own $4\times4$ block,

$$
E=\tfrac12\mathbf v^{\mathsf T}\mathbb M\,\mathbf v-\mathbf v^{\mathsf T}\mathbf L+\tfrac12\mathsf D^{\mathsf T}\mathbb M_l\mathsf D,
\qquad \mathbf L=\mathbb M_l\,\mathsf D
$$

A misalignment therefore adds **exactly one linear term** — no new structure. It does have
to be carried through *both* completions of the square: eliminating $ct$ sends
$\mathbf L_r\to\mathbf L_r+\mathbf g\,L_w/h$ and the constant to
$\text{const}-L_w^2/2h$; the transverse integration then completes the square against it
again. A dropped piece **shifts** the answer rather than making it diverge, so the
regression net is that $\mathsf D=\mathbf 0$ must reproduce the previous result *bit for
bit*, which is asserted.

**Why there is no $z_{\rm off}$ — stated carefully.** It is *not* that a longitudinal
offset simply equals a timing offset. The focal plane is fixed in space (for
$\beta_{\rm ff}=0$) while the envelope sweeps through it at $c$, so **focus position and
arrival time are independent**: two pulses whose foci coincide exactly still miss if they
arrive at different times, and that is a real, representable configuration
($z_{fx}=z_{fy}=0$, $t_{\rm off}\neq0$ — measured below).

What makes $z_{\rm off}$ redundant is that a *rigid* shift by $\Delta$ along
$\hat{\mathbf k}$ moves the focus **and** the envelope together, so it is already

$$
z_{fx}\to z_{fx}+\Delta,\qquad z_{fy}\to z_{fy}+\Delta,\qquad
t_{\rm off}\to t_{\rm off}+\Delta/c
$$

The longitudinal degrees of freedom are therefore three — $z_{fx}$, $z_{fy}$,
$t_{\rm off}$ — and a fourth would be a linear combination of them, not new physics. Both
of the user-facing quantities ("where is the focus" and "when does it arrive") are present
and independent; only the redundant combination is omitted. With coincident foci the yield
falls purely from the timing slip: $0.99$ at 5 ps, $0.86$ at 20 ps, $0.41$ at 50 ps and
$0.01$ at 200 ps, since the beams then meet a distance $c\,t_{\rm off}/2$ away from the
focus.

**Exact check.** With both hourglasses switched off, a transverse misalignment reduces the
yield by exactly

$$
\frac{N(\mathbf d)}{N(\mathbf 0)}=\exp\!\left[-\tfrac12\,
\mathbf d^{\mathsf T}\big(C_e+C_l\big)^{-1}\mathbf d\right]
$$

verified to $10^{-13}$, including an off-diagonal $\mathbf d$ — an on-axis test alone would
pass with a wrong inverse.

**They do not act independently of the crossing angle.** A timing slip makes the beams meet
away from the nominal point, and with a crossing angle that displaces the collision
*transversely* as well, so the same slip costs more when the beams cross (measured: 3.7%
loss at $\theta=0$ against 4.9% at $\theta=50$ mrad, for 10 ps). An implementation treating
the two as separable reductions would miss this; it is asserted as a test.

### A.5 A discrepancy this derivation exposes

§A.4 identifies the laser term in $\nu$ as $\sigma_l/z_R$ **exactly**. Both this
repository's `GaussianParaxialLaser.rayleigh_x` and the predecessor's own pulse class
define $z_R=4\pi\sigma^2/\lambda$ (i.e. $w_0=2\sigma$), giving a divergence
$\lambda/(4\pi\sigma)$.

The predecessor's `analytical.py` — and so the faithful port in `estimate_yield` — instead
uses $\lambda^2/(\pi^2\sigma^2)$, a divergence $\lambda/(\pi\sigma)$: **4× too large**,
16× in the squared term. The predecessor is inconsistent with its *own* laser model; this
was not introduced by the port to CGS.

It is not small. The baseline scenario's hourglass is almost entirely laser-driven
($3.3\times10^{-2}$ rad against the bunch's $5.0\times10^{-6}$), so the error passes
through nearly in full: **the closed form underestimates the baseline yield by 3.285×**.
`overlap_yield` reads `rayleigh_x()`/`rayleigh_y()` and is free of it (`DECISIONS.md`
D040). Nothing in the suite was previously sensitive to that term — the predecessor pin
tests port fidelity, and the Thomson anchor drives $\nu\to\infty$, removing the hourglass
term entirely.

### A.6 Crossing angle

A crossing angle changes the *geometry* of the overlap, which is a solvable Gaussian
problem. It is separate from §9.3, whose open item is the polarisation structure of the
**emission kernel**. Write the whole exponent as a quadratic form rather than tracking
terms individually:

$$
\mathsf M_e(z)=\frac{\hat{\mathbf x}\hat{\mathbf x}^{\mathsf T}}{\sigma_{ex}^2(z)}
+\frac{\hat{\mathbf y}\hat{\mathbf y}^{\mathsf T}}{\sigma_{ey}^2(z)}
+\frac{\hat{\mathbf z}\hat{\mathbf z}^{\mathsf T}}{\sigma_{ez}^2},
\qquad
\mathsf M_l(u)=\frac{\hat{\mathbf f}_1\hat{\mathbf f}_1^{\mathsf T}}{s_1^2(u)}
+\frac{\hat{\mathbf f}_2\hat{\mathbf f}_2^{\mathsf T}}{s_2^2(u)}
+\frac{\hat{\mathbf k}\hat{\mathbf k}^{\mathsf T}}{s_{ct}^2}
$$

Collecting the time dependence, with $\mathsf M=\mathsf M_e+\mathsf M_l$,

$$
E=\tfrac12\,\mathbf r^{\mathsf T}\mathsf M\,\mathbf r-ct\,(\mathbf g\cdot\mathbf r)
+\tfrac12 h\,(ct)^2,
\qquad
\mathbf g=\frac{\beta_0\hat{\mathbf z}}{\sigma_{ez}^2}+\frac{\hat{\mathbf k}}{s_{ct}^2},
\qquad
h=\frac{\beta_0^2}{\sigma_{ez}^2}+\frac{1}{s_{ct}^2}
$$

so the time integral is Gaussian and simply replaces $\mathsf M$ by

$$
\mathsf M' \;=\; \mathsf M-\frac{\mathbf g\,\mathbf g^{\mathsf T}}{h}
$$

Integrating the two transverse directions out of $\mathsf M'$ leaves its upper-left
$2\times2$ block $\mathsf A$ and the Schur complement
$S=\mathsf M'_{zz}-\mathbf b^{\mathsf T}\mathsf A^{-1}\mathbf b$ with
$\mathbf b=(\mathsf M'_{xz},\mathsf M'_{yz})$:

$$
\boxed{\;
N=\frac{\sigma_T(1+\beta_0)N_eN_L}{4\pi^2\,\sigma_{ez}s_{ct}}\sqrt{\frac{2\pi}{h}}
\int_{-\infty}^{\infty}\!\mathrm{d}z\;
\frac{\exp\!\big[-S(z)\,z^2/2\big]}
{\sigma_{ex}\sigma_{ey}\,s_1 s_2\,\sqrt{\det\mathsf A(z)}}\;}
$$

**This is not a second code path.** Head-on it collapses to §A.4 identically:
$\mathsf A$ becomes diagonal, $\mathbf b=0$,
$\sigma_{ex}\sigma_{ey}s_1s_2\sqrt{\det\mathsf A}=\sqrt{\det(C_e+C_l)}$, and with
$a=1/\sigma_{ez}^2$, $b'=1/s_{ct}^2$,

$$
S=\frac{ab'(1+\beta_0)^2}{\beta_0^2a+b'}=\frac{(1+\beta_0)^2}{D^2}
$$

the prefactors matching through $4\pi^2/\sqrt{2\pi}=2\pi\sqrt{2\pi}$. `overlap_yield`
evaluates one expression for every geometry — which is why no head-on test changed when
the crossing angle landed.

**Magnitude.** At the baseline scenario (3 mm bunch, 10 µm spots) the yield falls by
$1.07\times$ at 5 mrad, $\mathbf{2.18\times}$ at 20 mrad and $5.77\times$ at 50 mrad. In
the no-hourglass limit this reproduces the standard Piwinski reduction
$\big[1+(\sigma_s\tan\theta/\sigma_\perp)^2\big]^{-1/2}$ to $10^{-6}$ at 2 mrad.

**What is still head-on.** The emitted *spectrum*. The photon energy scales as
$\cos^2(\theta/2)$ (§2.2 of the §9.3 notes derives this from eq. `wR`, and reports it
already general in the paper), so the shape error is only $1\times10^{-4}$ at 20 mrad and
$4\times10^{-2}$ at 0.4 rad — second order in $\theta$, whereas the *yield* changes by
118% at 20 mrad. `AnalyticalEngine` normalises `SPECTRUM` to the yield, so with a crossing
angle the slice's integral is right and its shape is not; the engine reports this with the
magnitude attached, so it reads as a bound rather than an alarm.

### A.7 The exact two-dimensional form

The bunch's hourglass varies along $z$; the pulse's varies along
$u=\hat{\mathbf k}\cdot\mathbf r$. With a crossing angle these are different directions, so
the reduction in §A.6 is not exact — "everything analytic but one integral" is a head-on
statement. §A.6 samples the *slowly varying* spot sizes at
$u=(\hat{\mathbf k}\cdot\hat{\mathbf z})z$, dropping
$\delta=k_xx+k_yy$. Nothing in the exponent is approximated, including the
$\xi_1\sim x\cos\theta-z\sin\theta$ term that produces the entire Piwinski suppression.

The exact treatment keeps one more dimension. Rotate the transverse plane so $q_1$ lies
along the crossing direction $(k_x,k_y)/\sin\theta$; then

$$
u=(\hat{\mathbf k}\cdot\hat{\mathbf z})\,z+\sin\theta\;q_1
$$

depends on exactly one transverse coordinate, so $q_2$ still integrates analytically and
only $(z,q_1)$ are quadratured. In the rotated frame, integrating $q_2$ gives

$$
\int\!\mathrm{d}q_2\;e^{-E}
=\sqrt{\frac{2\pi}{\mathsf M''_{22}}}\;
\exp\!\left[-\tfrac12\Big(\mathsf M''_{11}q_1^2+2\mathsf M''_{13}q_1z+\mathsf M''_{33}z^2\Big)
+\frac{\big(\mathsf M''_{12}q_1+\mathsf M''_{23}z\big)^2}{2\mathsf M''_{22}}\right]
$$

Note the $q_1$ curvature is the *reduced* $\mathsf M''_{11}-\mathsf M''^2_{12}/\mathsf M''_{22}$,
not $\mathsf M''_{11}$; using the latter to set the integration span silently truncates the
integral exactly when the two transverse directions are correlated.

This degenerates **gracefully** as $\theta\to0$: the widths stop depending on $q_1$ and the
result returns to §A.6 — wasted nodes, not a singularity. That is also its practical
limitation. Measured on the worst corner available (a 2 µm waist against a 200 µm bunch):

| $\theta$ | 1D error vs converged 2D | nodes for the 2D to converge |
|---|---|---|
| 20 mrad | $-1.9\times10^{-4}$ | ${\sim}900$ |
| 100 mrad | $-2.1\times10^{-3}$ | ${\sim}300$ |
| 400 mrad | $-1.6\times10^{-3}$ | ${\sim}300$ |

So the exact mode earns its cost at **large** crossing angles. At small angles it spends
nodes re-integrating a direction the 1D path handles analytically and converges more
slowly than the approximation it is checking.

### A.8 The mean square $a_0$ over the collision

`estimate_spectrum_width`'s nonlinearity term wants the $a_0$ the bunch actually samples,
not the pulse's maximum. Because $a_0^2$ is exactly proportional to the *normalised*
photon density $p_L$ (the energy→$a_0$ chain is a square root of it),

$$
\big\langle a_0^2\big\rangle
=K\,\frac{\displaystyle\int\!\mathrm{d}t\,\mathrm{d}^3r\;n_e\,p_L^{\,2}}
{\displaystyle\int\!\mathrm{d}t\,\mathrm{d}^3r\;n_e\,p_L},
\qquad a_0^2=K\,p_L
$$

and the numerator is the **same** integral with the laser density entering squared. In the
quadratic form that is one parameter: every laser term doubles, $\mathsf M_l\to2\mathsf M_l$
and likewise inside $\mathbf g$ and $h$. Collecting normalisations, with $R_n$ the reduced
integral at laser power $n$,

$$
\big\langle a_0^2\big\rangle
=K\,\sqrt{\frac{h_1}{h_2}}\;\frac{R_2}{R_1\,(2\pi)^{3/2}\,s_{ct}}
$$

**An exact limit worth knowing.** Take a transversally pointlike bunch and switch off both
hourglasses. Integrating over *both* $z$ and $t$ spans every relative shift between the two
distributions, so the bunch convolution factors out of numerator and denominator alike and

$$
\frac{\big\langle a_0^2\big\rangle}{a_{0,\rm peak}^2}
\;\longrightarrow\;\frac{\int g^2}{\int g}\bigg/g(0)=\frac{1}{\sqrt2}
$$

for a normalised Gaussian $g$ — **independent of bunch length**. A counter-propagating
collision can therefore never reach the peak $a_0^2$, however small the bunch: it always
scans the pulse's full longitudinal profile. At the baseline the ratio is $0.35$, so using
$a_{0,\rm peak}^2$ overstated the nonlinear broadening term by about $3\times$.

### A.9 Resolved profiles

The same integral with fewer integrations performed, for preview plots drawn before an
expensive run is launched.

**In time.** Leave $t$ un-integrated; the transverse plane still closes analytically,
leaving one quadrature over $z$ per time point. This is $\mathrm{d}N/\mathrm{d}t$, the
collision's luminosity history.

**Across the transverse plane.** Keep $(x,y)$; time closes via $\mathsf M'$ and $z$ is
quadratured. This is $\mathrm{d}N/\mathrm{d}x\,\mathrm{d}y$ in the bunch's own frame,
which shows a mis-set geometry at a glance.

Both satisfy exact identities that the tests assert rather than assume:

$$
\int\frac{\mathrm{d}N}{\mathrm{d}t}\,\mathrm{d}t
=\iint\frac{\mathrm{d}N}{\mathrm{d}x\,\mathrm{d}y}\,\mathrm{d}x\,\mathrm{d}y
= N
$$

both holding to ${\sim}10^{-7}$. Angle-resolved output is deliberately deferred: it needs
the emission kernel, not the overlap geometry.

### A.10 Validation and cost

Four independent checks, in increasing generality:

| check | isolates | agreement |
|---|---|---|
| round-beam closed form, §A.4 | the whole 1D reduction | $5\times10^{-10}$ |
| constant-width closed form ($3\times3$ determinant, no quadrature) | the crossing geometry alone | ${\sim}10^{-14}$, to 0.4 rad, both planes |
| Piwinski $\big[1+(\sigma_s\tan\theta/\sigma_\perp)^2\big]^{-1/2}$ | that the suppression is known physics | $10^{-6}$ at 2 mrad |
| brute-force Monte Carlo over `photon_density` with sampled macroparticles | everything at once, sharing no algebra | few $\times10^{-4}$ |

The Monte Carlo is converged separately in particle count and in time grid, so a pass
cannot be a grid artefact. `⟨a0²⟩` is validated the same way, weighting each particle by
`a0_profile**2`.

Three deliberate cost tiers (`DECISIONS.md` D043):

| tier | cost | assumes |
|---|---|---|
| `estimate_yield` | ${\sim}0.01$ ms | round, head-on, aligned foci |
| `overlap_yield`, 1D (default) | ${\sim}1$–2 ms | spot sizes sampled along $z$ |
| `overlap_yield`, `n_quad_u > 1` | ${\sim}40$–800 ms | nothing — exact |

The first two are real-time at any interaction rate, which is what keeps §4.3's claim that
analytical is the one real-time engine true of something. Previews cost ${\sim}8$ ms for a
400-point time profile and ${\sim}100$ ms for a $64\times64$ transverse image.

---

## B. Flying focus

**Status: implemented**, `beta_ff != 0` routed through
`gammaforge.engines.analytical.formulas._reduced_integral_flying_focus`, validated against
the brute-force Monte Carlo at $\beta_{\rm ff}\in\{-0.5,0.5,1,2\}$ to ${\sim}10^{-3}$,
with and without a crossing angle. §A's guard against $\beta_{\rm ff}\neq0$ is removed.

### B.1 Why the time integral stops closing

The repository models a flying focus by sliding the spot-size evaluation point with time,
`GaussianParaxialLaser._local_coordinates` returning

$$
u_{\rm spot}=u+\beta_{\rm ff}\,ct,
\qquad u=\hat{\mathbf k}\cdot\mathbf r
$$

while the longitudinal envelope stays a function of $u-ct$. The focal plane therefore sits
at $u_{\rm spot}=z_f$, i.e. at $u=z_f-\beta_{\rm ff}ct$, and **moves along $\hat{\mathbf k}$
at velocity $-\beta_{\rm ff}c$.** Head-on, $\hat{\mathbf k}=-\hat{\mathbf z}$, so the focus
travels at $+\beta_{\rm ff}c$ in $+z$: at $\beta_{\rm ff}=1$ it co-moves with the bunch.
The Rayleigh range carries the $(1+\beta_{\rm ff})$ stretch of `rayleigh_x`. That factor is
**not** a bookkeeping convention: it follows from the paraxial solution of Maxwell's
equations for a sliding focus (author, 2026-08-09). §B.5's reciprocal symmetry depends on
it, and is therefore a physical identity rather than an artefact.

In §A the two spot sizes depended on $u$ alone, hence on $z$ alone head-on, and $t$ entered
only the exponent — which is why §A.3 could integrate it. With $\beta_{\rm ff}\neq0$ the
*coefficients* of the quadratic form depend on $t$, and that step is gone.

### B.2 Dimension counting — the answer is still 2D

The widths depend on exactly **two** linear functionals of $(x,y,z,ct)$:

$$
A=z,\qquad B=u_{\rm spot}=k_xx+k_yy+k_zz+\beta_{\rm ff}\,ct
$$

Everything else in the exponent is quadratic with $(A,B)$-dependent coefficients. Fixing
$A$ and $B$ leaves a **two-dimensional** affine subspace of the four, on which the
integrand is an ordinary Gaussian and integrates in closed form. So

> an exact treatment of a crossing angle **and** an arbitrary flying-focus velocity,
> together, is a two-dimensional quadrature — no worse than the crossing angle alone.

$A$ and $B$ degenerate into one functional exactly when $k_x=k_y=0$ and
$\beta_{\rm ff}=0$, which is §A's 1D case. The two effects each add one width-argument, and
having both does not add a third.

### B.3 The implemented form

Head-on the natural pair is $(z,w)$ with $w=ct$, since $u_{\rm spot}=-z+\beta_{\rm ff}w$ is
then a function of them alone; the analytic pair is $(x,y)$. Integrating those at fixed
$(z,w)$ gives $2\pi/\sqrt{\det\mathsf A}$ exactly as in §A.6 but **without** eliminating
time, so the exponent keeps its explicit $w$ terms:

$$
\boxed{\;
N=\frac{\sigma_T(1+\beta_0)N_eN_L}{4\pi^2\sigma_{ez}s_{ct}}
\iint \mathrm{d}z\,\mathrm{d}w\;
\frac{\exp\!\left[-\dfrac{(z-\beta_0w)^2}{2\sigma_{ez}^2}
-\dfrac{(z+w)^2}{2s_{ct}^2}\right]}
{\sqrt{\det\!\big(C_e(z)+C_l(-z+\beta_{\rm ff}w)\big)}}\;}
$$

With a crossing angle the same $(z,w)$ pair is used and the residual transverse part of
$u_{\rm spot}$, $\delta=k_xx+k_yy$, is sampled at the axis — the *same* approximation §A.7
already measures at $1.9\times10^{-4}$. A fully exact crossing-plus-flying-focus evaluation
would quadrature $(A,B)$ of §B.2 instead; the counting says it costs no more dimensions,
only more coordinate bookkeeping.

**Grid.** This is where a naive implementation fails, and did: the $(z,w)$ Gaussian is
nearly degenerate — the collision lives on a thin diagonal ridge, since $z$ and $w$ are
almost perfectly correlated — so a rectangular grid sized from the *marginals*
under-resolves the ridge and comes out **3.8% low** on a short bunch. Nodes are placed on
the **principal axes** of the quadratic form, and their count is raised until each step
advances $u_{\rm spot}$ by less than $z_R/8$.

### B.4 A 1D approximation exists, and it is not safe in general

One can freeze the widths at the stationary point of the $w$-integral,
$w_\star=(\mathbf g\cdot\mathbf r)/h$, which on the axis gives a linear
$u_{\rm spot}\approx\kappa z$ with

$$
\kappa=k_z+\beta_{\rm ff}\,\frac{g_z}{h},
\qquad g_z=\frac{\beta_0}{\sigma_{ez}^2}+\frac{k_z}{s_{ct}^2},
\qquad h=\frac{\beta_0^2}{\sigma_{ez}^2}+\frac{1}{s_{ct}^2}
$$

so that §A's 1D machinery applies unchanged. **It should not be trusted.** Unlike §A.7's
approximation — where the dropped term entered an even, slowly varying prefactor and
cancelled to first order — a flying focus exists precisely to correlate the width with
time, so replacing a distribution over $w$ by its mean errs at *first* order. Measured
against §B.3 on the baseline scenario:

| $\beta_{\rm ff}$ | 1D error, long bunch ($\sigma_{ez}=3$ mm) | 1D error, short bunch ($\sigma_{ez}=30\,\mu$m) |
|---|---|---|
| 0.25 | $+1.1\times10^{-2}$ | — |
| 0.5 | $+6.0\times10^{-2}$ | $+1\times10^{-5}$ |
| 1.0 | $+3.4\times10^{-1}$ | $+8\times10^{-5}$ |
| 2.0 | $+3.9\times10^{-1}$ | — |

The controlling parameter is the conditional spread of $w$ at fixed $z$ times the slide,
against the Rayleigh range:

$$
\varepsilon\sim\frac{\beta_{\rm ff}\,\sigma_{ez}}{z_R}
$$

which is $2.5\beta_{\rm ff}$ for the baseline and $0.025\beta_{\rm ff}$ for the short
bunch. So the approximation happens to be excellent in the regime a flying focus is
*for* — a short bunch you can synchronise to — and useless outside it. That is too sharp a
knife to ship as a default, so the implemented path is always the 2D one when
$\beta_{\rm ff}\neq0$ (`DECISIONS.md` D044).

### B.5 Physics the derivation reproduces

Two results fall out, neither built in:

**Synchronisation wins.** $\beta_{\rm ff}=1$ maximises the yield — the focal plane
co-moves with the bunch, so the electrons sit at the waist throughout instead of sweeping
through the hourglass. It is worth $2.0\times$ on the baseline and $\mathbf{2.8\times}$ on a
30 µm bunch relative to $\beta_{\rm ff}=0$, and beats both slower and faster slides.

**An exact reciprocal symmetry.** For a short bunch the yield is invariant under

$$
\beta_{\rm ff}\;\longrightarrow\;1/\beta_{\rm ff}
$$

(verified to $2\times10^{-4}$: $\beta_{\rm ff}=0.5$ and $2$ agree to seven digits). Along
the ridge $z\simeq ct$, so $u_{\rm spot}\simeq(\beta_{\rm ff}-1)ct$ while the Rayleigh range
carries $(1+\beta_{\rm ff})$; the spot therefore depends on
$(\beta_{\rm ff}-1)/(\beta_{\rm ff}+1)$, which is **odd** under
$\beta_{\rm ff}\to1/\beta_{\rm ff}$ — and the width depends on its square. $\beta_{\rm ff}=1$
is the fixed point, which is why it is the optimum. Nothing in the implementation knows
this, so it is a sharp check on the whole construction.

### B.6 Not cross-checked against the author's own derivation

The author has previously derived expressions for the head-on counter-propagating flying
focus with the pulse peak synchronised to the bunch. **Those were not available here**, so
§B is validated against the Monte Carlo only. A comparison against an independent analytic
derivation would be worth more than another numerical run — particularly on the
$(1+\beta_{\rm ff})$ Rayleigh convention, which §B inherits from the code rather than
deriving, and which the $\beta_{\rm ff}\to1/\beta_{\rm ff}$ symmetry above depends on
directly.

---

## C. The polarization factor in $\hat a$ (resolves §0)

**Status: resolved by the author (2026-08-10); applied in `engines/analytical`, NOT yet in
xigma or the delta reference.**

The parallel session's §0 records that the code's `ahat` is twice the paper's $\hat a$ and
marks it BLOCKING. It is not a convention mismatch — it is the polarization cycle average,
and the paper is right.

$a_0$ is by definition the **normalized magnitude of the electric field**, a peak amplitude.
The cycle-averaged normalized intensity is therefore

$$
\langle a^2\rangle = C\,a_0^2,\qquad
C=\tfrac12\ \text{(linear)},\qquad C=1\ \text{(circular)}
$$

because a linearly polarized field oscillates as $\cos\varphi$ (so
$\langle\cos^2\rangle=\tfrac12$) while a circularly polarized one has constant magnitude. The
same statement read the other way: **at fixed $a_0$ circular polarization carries twice the
cycle-averaged energy density**; equivalently, at fixed pulse energy circular gives $a_0$
smaller by $\sqrt2$.

`io.laser.GaussianParaxialLaser._a0_from_density` implements the **linear** chain explicitly
($\langle u\rangle=E_0^2/8\pi$, i.e. $E_0=\sqrt{8\pi\langle u\rangle}$), so its `a0_profile`
is the linear-polarization peak amplitude and $C=\tfrac12$.

### C.1 What is affected

`TrajectorySamples.ahat()` returns $a_{0,\rm peak}^2\sum r^2/\sum r$ — the
**amplitude-squared** trajectory average, with no cycle factor. So the code's value is twice
the paper's, and `ahat` feeds the resonance denominator in two places:

| site | expression |
|---|---|
| `validation/references/delta.py` | `s_res = gamma**2 / (1 + ahat + gamma**2 * r**2)` |
| xigma Stage 2 | inverts the same resonance per `ahat` cell |

Both therefore over-state the nonlinear red-shift by a factor of two.

### C.2 Why §7's cross-validation cannot catch it

xigma and the delta reference **share** `TrajectorySamples.ahat()`. A common-mode error
cancels in any comparison between them, so the four-way cross-check — the machinery whose
whole purpose is catching this class of mistake — is blind to it by construction. Worth
recording independently of the fix: it is a gap in the cross-validation design, not only a
bug.

The analytical engine is the leg that exposes it, because it computes $\hat a$ from the
overlap integral rather than from `TrajectorySamples`. As of this branch it applies
$\hat a=\tfrac12\langle a_0^2\rangle$, so an analytical-vs-xigma comparison of the nonlinear
red-shift should differ by exactly two until §C.1's sites are corrected.

### C.3 Connection to §9.2 (ellipticity)

$C=\tfrac12\to1$ from linear to circular is exactly what `ellipticity` should interpolate,
and it enters in **two** places, not one: the resonance shift $\hat a$, and the
energy-to-$a_0$ conversion itself, since `_a0_from_density`'s $\sqrt{8\pi u}$ is the linear
relation. `ELLIPTICITY_IS_NOOP` is the placeholder for both. This gives §9.2 a concrete,
checkable hook rather than an open question.

---

## 0. ~~BLOCKING~~ RESOLVED AND FIXED: the code's `ahat` was twice the paper's $\hat a$

This came up while reading eq. `ahat` for the ellipticity work. It is not part of §9.2 or
§9.3, and it affects every scenario. **§C above is the derivation of the factor**; this
section is the discrepancy report and the record of how it was settled — the two were
written independently, from opposite ends, and agree.

> **Resolution (author, 2026-08-10).** The paper is right and the code was missing the
> $1/2$. $a_0$ is by definition the normalized **peak** magnitude of the electric field, so
> the cycle-averaged normalized intensity is $\langle a^2\rangle = C\,a_0^2$ with $C=1/2$
> for linear polarization ($\langle\cos^2\rangle = 1/2$) and $C=1$ for circular (constant
> magnitude). `_a0_from_density` implements the linear chain explicitly, so
> `a0_profile` is the linear peak amplitude and $C=1/2$ applies.
>
> **Fixed in `DECISIONS.md` D053.** `stages.ahat_from_shape` is now the single route from
> `a0_shape` to $\hat a$ and applies `io.laser.CYCLE_AVERAGE_FACTOR` $= 1/2$. The
> compensating-$2$ worry below was checked and ruled out: `RELATIVE_VELOCITY`, the
> $E = 4\hbar\omega_0 s$ convention and `KERNEL_NORMALIZATION_CONSTANT` all live in the
> photon-**count** path, which §9.1 pins absolutely (D033), while $\hat a$ lives in the
> resonance denominator and moves photons along $s$ without changing how many there are.
> No check that constrains one is sensitive to the other, so they cannot be cancelling.
> Total yield is bit-for-bit unchanged by the fix, as that argument requires.

### The paper

Eq. `field` writes the incident field as

$$
\mathbf{E}(\varphi) = a_0\,\Re\!\left\{E(\varphi)\left(\epsilon_0\mathbf{e}_0+\epsilon_1\mathbf{e}_1\right)\right\},
\qquad \sum_i|\epsilon_i|^2 = 1 ,
$$

so eq. `Xi`'s coherence matrix $\Xi_{ij}=\epsilon_i\epsilon_j^*$ has $\operatorname{Tr}\hat\Xi\equiv 1$
identically. The text below eq. `Xi` then states
$\langle a^2(\varphi)\rangle = a_0^2|E(\varphi)|^2\operatorname{Tr}\hat\Xi/2$, and eq. `ahat` defines

$$
\hat a \;=\; a_0^2\,\frac{\operatorname{Tr}\hat\Xi}{2}\,
\frac{\int |E|^4\,\mathrm{d}\varphi}{\int |E|^2\,\mathrm{d}\varphi}
\;=\; \frac{a_0^2}{2}\,\frac{\int |E|^4}{\int |E|^2} .
$$

### The code

`stages.integrate_trajectories` builds `ratio` $=(a_0^{\text{local}}/a_0^{\text{peak}})^2$ from
`LaserField.a0_profile`, then `a0_shape` $=\sum\text{ratio}^2/\sum\text{ratio}$, and
`TrajectorySamples.ahat()` $= a_0^{\text{peak}\,2}\times$ `a0_shape`. So

$$
\hat a_{\text{code}} \;=\; a_0^2\,\frac{\int |E|^4}{\int |E|^2} \;=\; 2\,\hat a_{\text{paper}} .
$$

### The step that decides it

Whether `a0_profile` is the *amplitude* envelope $a_0|E|$ or something already
cycle-averaged. It is the amplitude envelope — its own docstring says the period-resolved
`field()`'s "envelope *is* `a0_profile`", and measured directly at the focus over one
carrier period $T$:

| quantity | value |
|---|---|
| `a0_profile(0)` | $0.995004$ |
| $\max\lvert\mathbf{a}\rvert$ over $T$ | $0.995004$ &nbsp; (ratio to envelope $1.0000$) |
| $\langle\lvert\mathbf{a}\rvert^2\rangle$ over $T$ | $0.4999\times\text{envelope}^2$ |

So `ratio` $=|E|^2$ exactly, and the missing $\operatorname{Tr}\hat\Xi/2 = 1/2$ is not
hiding in the profile.

### Corroboration from outside both

The textbook linear-polarization ponderomotive denominator is $1 + a_0^2/2$, which is
eq. `ahat` for a flat-top envelope. $\hat a_{\text{code}}$ would make it $1 + a_0^2$ — the
*circular* value. The paper and the standard result agree with each other and not with the
code.

### Provenance

Inherited, not introduced by this rebuild: the predecessor uses the same
$\hat a = a_0^2\times$`a0_shape` (`ComptonSuite/src/gammaforge/models/xigma_i/deposition.py:429`,
`particles.py:263`).

### What it costs

$\hat a$ enters only as the redshift $s_{\text{res}} = \gamma^2/(1+\hat a+\gamma^2 r^2)$ and
the $(1+\hat a)^{-1}$ Jacobian, so there is no cancellation downstream — it is a straight
$2\times$ overestimate of the nonlinear redshift. Order of the effect on the bank: $\sim 1\%$
on `baseline` ($\hat a$ mean $0.0097$), $\sim 5\%$ on `near_a0_max` ($0.0486$).

**A second cost, which is why this needs deciding before §9.2/§9.3 code and not after.**
`retarget_ahat`'s target grid was tuned last session against the *current* values (D032).
Halving them moves the bank down a bin:

| scenario | $\hat a$ mean / max now | bins | halved | bins |
|---|---|---|---|---|
| `baseline` | $0.0097 \,/\, 0.0191$ | $0\to0$ | $0.0049\,/\,0.0096$ | $0\to0$ |
| `low_a0` | $0.00097\,/\,0.0019$ | $0\to0$ | $0.00049\,/\,0.00095$ | $0\to0$ |
| `near_a0_max` | $0.0486\,/\,0.0954$ | $1\to2$ | $0.0243\,/\,0.0477$ | $0\to1$ |

Target edges at the current defaults: $0,\;0.0347,\;0.0670,\;0.0971,\dots$. `near_a0_max` is
the only scenario that currently resolves the redshift at all, and after the correction its
mean falls into the floor bin. `ahat_decades` would need retuning as part of the fix — the
same numerical exercise D032 already documents, not a new kind of problem.

### What was actually done

`ahat_from_shape` applies $C=1/2$; `a0_shape` keeps its literal meaning
$\int|E|^4/\int|E|^2$. The `ahat_decades` retune anticipated above was **measured but not
applied** — the bin-shift table is right, the resulting centroid bias against `delta` is
$\sim1\%$ and pre-existing rather than created by the fix, and changing a production default
the author tuned with stated physics reasoning is their call. Numbers and the recommended
one-parameter change ($\texttt{decades}\;1.0\to0.3$) are in D053's last section.

---

## 1. §9.2 — ellipticity

`GRAND_PLAN.md` §9.2 records "no formula exists" for the energy→$a_0$ relation with
ellipticity. That turns out to be the right observation with the wrong conclusion: the
paper's conventions *determine* the answer, and the answer is that the existing chain is
already correct. What ellipticity actually changes is somewhere else entirely.

### 1.1 The energy→$a_0$ chain needs no ellipticity correction — it is an invariance, not a gap

Because $\sum_i|\epsilon_i|^2 = 1$ forces $\operatorname{Tr}\hat\Xi\equiv 1$, the relation
below eq. `Xi` reads $\langle a^2\rangle = a_0^2|E|^2/2$ *for every polarization state*.
Inverting it at the pulse peak against the cycle-averaged intensity, with $U$ the energy
density:

$$
\langle E^2\rangle = \frac{4\pi I}{c} = 4\pi U ,
$$

$$
a_0^2 \;=\; 2\langle a^2\rangle \;=\; 2\left(\frac{e}{m_e c\,\omega}\right)^{\!2}\langle E^2\rangle
\;=\; 8\pi U\left(\frac{e}{m_e c\,\omega}\right)^{\!2} ,
$$

$$
\boxed{\;a_0 = \frac{e}{m_e c\,\omega}\sqrt{8\pi U}\;}
$$

which is exactly what `GaussianParaxialLaser._a0_from_density` computes. So the current
identity/no-op is not a placeholder standing in for a missing factor — it is the correct
result, and $a_0$ at fixed pulse energy is genuinely independent of ellipticity in this
paper's convention.

The familiar $(1+\varepsilon^2)/2$-type factor from other treatments comes from defining
$a_0$ through the *peak instantaneous* $|a|$ rather than through the cycle average. Under
that definition a circular pulse of the same energy has $|a|$ constant at
$a_0^{\text{lin}}/\sqrt2$ while a linear one peaks at $a_0^{\text{lin}}$; the paper
sidesteps the whole question by normalizing $\hat\Xi$.

#### The author picked the peak convention — and it does not change the conclusion

D053's resolution defines $a_0$ as the **peak instantaneous** magnitude, i.e. the second of
the two conventions above: $\langle a^2\rangle = C a_0^2$ with $C = 1/2$ linear, $1$
circular. That is the opposite bookkeeping from the paper's normalized $\hat\Xi$, so it is
worth writing out that the two agree on every physical number at fixed pulse energy.

Under the peak convention $C$ enters in **two** places, and they cancel. Inverting
$\langle a^2\rangle = C a_0^2$ against $\langle E^2\rangle = 4\pi U$,

$$
a_0^2 = \frac{1}{C}\left(\frac{e}{m_ec\,\omega}\right)^{\!2} 4\pi U
\qquad\Longrightarrow\qquad
\boxed{\;a_0 = \frac{e}{m_ec\,\omega}\sqrt{\frac{4\pi U}{C}}\;}
$$

— which is the code's $\sqrt{8\pi U}$ at $C=1/2$, and smaller by $\sqrt2$ at $C=1$, exactly
as the resolution states. Substituting it into $\hat a = C\,a_0^2 \int|E|^4/\int|E|^2$:

$$
\hat a \;=\; C\cdot\frac{1}{C}\left(\frac{e}{m_ec\,\omega}\right)^{\!2}4\pi U\cdot
\frac{\int|E|^4}{\int|E|^2}
\;=\;\left(\frac{e}{m_ec\,\omega}\right)^{\!2}4\pi U\,\frac{\int|E|^4}{\int|E|^2},
$$

with no $C$ left. **At fixed pulse energy $\hat a$ is ellipticity-independent under either
convention** — as it must be, since it is a ratio of physical intensity moments and the
conventions differ only in what number they call $a_0$. The same cancellation protects the
photon count: `photon_density_scale` inverts `_a0_from_density` exactly, so if one changes
the other changes with it and `luminosity` does not move.

**Consequence for the code:** implementing §9.2 in full would change `a0_peak` — a reported
quantity — and change nothing else through this route. The handover framing "$C$ enters in
two places" is right; worth adding is that those two places *cancel for $\hat a$*, so §9.2's
only physical effect on the spectrum is §1.2's kernel polarization factor. Two pieces of
text also become wrong rather than merely incomplete —

- `_a0_from_density`'s docstring calls itself "the standard **linear-polarization** chain."
  Correct as written, but the reason is a convention choice, not a physical restriction:
  the elliptical case differs from it only by $\sqrt{1/2C}$, and the pulse energy it is
  inverting is polarization-blind.
- `ELLIPTICITY_IS_NOOP`'s framing ("carried but changes nothing", flip to `False` when the
  derivation lands) describes an unimplemented placeholder. Post-D053 it guards two known
  factors that cancel in $\hat a$ and do not cancel in `a0_peak` — with the `validate()`
  warning narrowed, not deleted, because of §1.2.

### 1.2 Ellipticity *does* enter, in the polarization factor of the kernel

The place $\hat\Xi$ is actually consumed is $\hat U^{T}\hat\Xi\hat U$ in eq. `xsec` /
eq. `Fmatrix`. Eq. `trace` gives

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i,j}\Xi_{ij}\,(\mathbf{u}_i\cdot\mathbf{u}_j) ,
$$

and eq. `umod` supplies only the diagonal. Extending eq. `umod` to the off-diagonal, from
eq. `udef` under the same neglect of $\mathbf{v}\cdot\mathbf{e}_i$ (justified head-on — see
§2.3 for where that fails):

$$
\mathbf{u}_i\cdot\mathbf{u}_j
= \delta_{ij} - \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{n}\cdot\mathbf{e}_j)}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
= \delta_{ij} - \frac{4\gamma^2\theta^2\cos\psi_i\cos\psi_j}{(1+\gamma^2\theta^2)^2},
$$

using $\mathbf{n}\cdot\mathbf{e}_i = \theta\cos\psi_i$ and
$1-\mathbf{v}\cdot\mathbf{n} = (1+\gamma^2\theta^2)/2\gamma^2$. The diagonal reproduces
eq. `umod` exactly. With $\mathbf{e}_0\perp\mathbf{e}_1$ we have $\psi_1 = \psi_0-\pi/2$, so
writing $\psi\equiv\psi_0$:

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= 1 - \frac{4\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}
\Bigl[\Xi_{00}\cos^2\psi + \Xi_{11}\sin^2\psi + 2\,\Re(\Xi_{01})\cos\psi\sin\psi\Bigr].
$$

Parametrizing the scalar `ellipticity` $\varepsilon\in[0,1]$ ($0$ = linear, $1$ = circular)
as the ratio of the ellipse's minor to major axis, with the major axis along $\mathbf{e}_0$:

$$
\epsilon_0 = \frac{1}{\sqrt{1+\varepsilon^2}},\qquad
\epsilon_1 = \frac{i\,\varepsilon}{\sqrt{1+\varepsilon^2}},
$$

$$
\Xi_{00} = \frac{1}{1+\varepsilon^2},\qquad
\Xi_{11} = \frac{\varepsilon^2}{1+\varepsilon^2},\qquad
\Xi_{01} = \frac{-i\,\varepsilon}{1+\varepsilon^2},\qquad
\operatorname{Tr}\hat\Xi = 1\ \checkmark
$$

The cross term vanishes because $\Re(\Xi_{01}) = 0$, and that is not luck: $\Xi_{01}$ is
purely imaginary exactly when the two components are in quadrature, which *is* the
statement that the ellipse axes are $\mathbf{e}_0/\mathbf{e}_1$. `psi_pol` already means
"the major axis's azimuth", so the convention the code carries is the one that makes the
cancellation hold. A general relative phase would leave a $\sin\psi\cos\psi$ term and
describe an ellipse tilted with respect to `psi_pol` — a second angle the schema does not
have and, on this reading, does not need.

**Result:**

$$
\boxed{\;
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= 1 - \frac{4\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}\cdot
\frac{\cos^2\psi + \varepsilon^2\sin^2\psi}{1+\varepsilon^2}
\;}
$$

- $\varepsilon = 0 \;\Rightarrow\; 1 - \dfrac{4\gamma^2\theta^2\cos^2\psi}{(1+\gamma^2\theta^2)^2}$
  — eq. `linpol`, and exactly what `spectrum_from_table` / `delta.resonance_spectrum`
  compute today. ✓
- $\varepsilon = 1 \;\Rightarrow\; 1 - \dfrac{2\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}$
  — azimuth-independent, as circular polarization must be. ✓

**Numerically checked** against the direct sum $\sum_{ij}\Xi_{ij}(\mathbf{u}_i\cdot\mathbf{u}_j)$
with $\mathbf{u}_i$ built straight from eq. `udef`, over 4000 random
$(\gamma\in[10,3000],\ \theta\in[0,4/\gamma],\ \psi,\ \varepsilon)$ in head-on geometry.
Keeping $1-\mathbf{v}\cdot\mathbf{n}$ exact: worst error $1.5\times10^{-9}$ (rounding).
Applying the paper's own $1-\mathbf{v}\cdot\mathbf{n}\to(1+\gamma^2\theta^2)/2\gamma^2$:
worst error $1.2\times10^{-2}$ — that residual is eq. `smallangle`'s documented
$O(\theta^2)$, not this derivation, and it is present in the existing linear-only kernel too.

**Consequence for the code, if the author agrees:** a one-line generalization in each of the
two kernels,

$$
\cos^2\psi \;\longrightarrow\; \frac{\cos^2\psi + \varepsilon^2\sin^2\psi}{1+\varepsilon^2},
$$

plus threading `ellipticity` through to them the way `psi_pol` already is. Note this changes
the *angular* distribution only; the yield integrated over azimuth is unaffected, since
$\langle\cos^2\psi\rangle = \langle\sin^2\psi\rangle = 1/2$.

---

## 2. §9.3 — crossing angle

`GRAND_PLAN.md` §9.3 says the paper's angular derivation "is built for near-backscattering
geometry, accurate to $O(\theta^2)$ around the collinear axis; the paper warns against
extending it without revisiting the geometry."

**That warning is about something else.** The passage it refers to (`sec:validity4`, final
paragraph) says: "the small-angle reduction (eq. `smallangle`) presumes near-backscattering,
with accuracy $O(\theta^2)$. It is amply satisfied within the collimation apertures of
interest but should not be extended to **large-angle observation** without revisiting the
geometry." Eq. `smallangle` is

$$
\theta = \sqrt{(n_y-\theta_y)^2 + (n_z-\theta_z)^2}
$$

— the angle between the *electron's* velocity and the *observation* direction. It constrains
how far off-axis you may observe and how divergent the bunch may be. It says nothing about
the laser's direction of incidence, which does not appear in it.

So the crossing angle is a narrower problem than §9.3 assumes. Three places carry it, and
the paper leaves two of them already general.

### 2.1 The relative-velocity factor (already general in the paper)

Eq. `lumfun` defines
$\mathcal{L}(\zeta) = \int v_{\text{rel}}\, n_{\text{ph}}(t,\mathbf{r}(t;\zeta))\,\mathrm{d}t$
and never specializes $v_{\text{rel}}$. For an electron of velocity $\mathbf{v}$ meeting
photons propagating along $\hat{\mathbf{n}}_0$, the flux factor is

$$
v_{\text{rel}} = c\left(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0\right).
$$

Head-on ($\hat{\mathbf{n}}_0 = -\hat{\mathbf{z}}$, $\mathbf{v} = \beta\hat{\mathbf{z}}$) gives
$c(1+\beta)\to 2c$, which is `stages.RELATIVE_VELOCITY = 2.0`. With the geometry `io.laser`
already pins, $R = R_y(\theta_{xz})R_x(\theta_{yz})$ applied to $-\hat{\mathbf{z}}$:

$$
\hat{\mathbf{n}}_0 = \bigl(-\sin\theta_{xz}\cos\theta_{yz},\ \sin\theta_{yz},\ -\cos\theta_{xz}\cos\theta_{yz}\bigr),
$$

$$
\boxed{\;1-\mathbf{v}\cdot\hat{\mathbf{n}}_0 = 1 + \beta\cos\theta_{xz}\cos\theta_{yz}
\;\xrightarrow[\ \beta\to1\ ]{}\; 2\cos^2\!\frac{\alpha}{2},
\qquad \cos\alpha \equiv \cos\theta_{xz}\cos\theta_{yz}\;}
$$

### 2.2 The resonance frequency (already general in the paper)

Eq. `wR` is

$$
\omega_R = \omega_L\,\frac{2\gamma^2\left(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0\right)}{1+\gamma^2\theta^2+\hat a}
$$

— the same factor, written out. It is only eq. `wRgamma` that substitutes
$1-\mathbf{v}\cdot\hat{\mathbf{n}}_0\to 2$ to get the familiar $4\omega_L\gamma^2/(\cdots)$. So

$$
\omega_R = \frac{4\omega_L\gamma^2\cos^2(\alpha/2)}{1+\gamma^2\theta^2+\hat a},
$$

and the kinematic edge of eq. `kinematic` moves with it:
$\omega\theta^2 < 4\omega_L\cos^2(\alpha/2)$.

Note both §2.1 and §2.2 carry the *same* factor $(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0)$,
which is a useful check on any implementation: the crossing angle enters the yield and the
photon energy through one quantity, not two independent ones.

**One convention consequence for the code.** $s$ is defined by $E = 4\hbar\omega_0 s$ with
$s_{\text{res}} = \gamma^2/(1+\hat a+\gamma^2\theta^2)$, i.e. the head-on $4\gamma^2$ is baked
into the *conversion*, not the resonance. The cleanest placement is to keep $s_{\text{res}}$
as it is and carry $\cos^2(\alpha/2)$ in the energy conversion (`Collision`'s
`photon_energy`), so the table axis keeps its meaning and one factor moves in one place.
That is a code-shape opinion, not physics.

### 2.3 The polarization structure — this is the genuine gap

Eq. `umod` derives $|\mathbf{u}_i|^2$ "on neglecting $\mathbf{v}\cdot\mathbf{e}_i$, which is
of order $a_0/\gamma$." That estimate holds when $\mathbf{e}_i\perp\mathbf{v}$: head-on,
$\mathbf{e}_0,\mathbf{e}_1$ span the plane transverse to $\hat{\mathbf{z}}$ and the electron
moves along $\hat{\mathbf{z}}$, so $\mathbf{v}\cdot\mathbf{e}_i$ is only the bunch's own
small divergence times $\beta$. A crossing angle rotates $\mathbf{e}_0,\mathbf{e}_1$ out of
that plane and $\mathbf{v}\cdot\mathbf{e}_i$ becomes $O(\sin\alpha)$ — not small, and not
$a_0/\gamma$-suppressed. Everything else about eq. `udef` survives; it is this one neglect
that breaks.

Redoing $\mathbf{u}_i\cdot\mathbf{u}_j$ from eq. `udef` without it. Write
$a_i \equiv \mathbf{n}\cdot\mathbf{e}_i$, $b_i \equiv \mathbf{v}\cdot\mathbf{e}_i$, and use
$|\mathbf{n}-\mathbf{v}|^2 = 2(1-\mathbf{v}\cdot\mathbf{n}) - 1/\gamma^2$ and
$\mathbf{e}_i\cdot\mathbf{e}_j = \delta_{ij}$:

$$
\begin{aligned}
\mathbf{u}_i\cdot\mathbf{u}_j
&= \frac{a_i a_j\,|\mathbf{n}-\mathbf{v}|^2}{(1-\mathbf{v}\cdot\mathbf{n})^2}
 - \frac{a_i(a_j-b_j) + a_j(a_i-b_i)}{1-\mathbf{v}\cdot\mathbf{n}} + \delta_{ij}\\[4pt]
&= a_i a_j\left[\frac{2}{1-\mathbf{v}\cdot\mathbf{n}} - \frac{1}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}\right]
 - \frac{2a_i a_j - a_i b_j - a_j b_i}{1-\mathbf{v}\cdot\mathbf{n}} + \delta_{ij}
\end{aligned}
$$

giving

$$
\boxed{\;
\mathbf{u}_i\cdot\mathbf{u}_j
= \delta_{ij}
- \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{n}\cdot\mathbf{e}_j)}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{v}\cdot\mathbf{e}_j)
      + (\mathbf{n}\cdot\mathbf{e}_j)(\mathbf{v}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}}
\;}
$$

The first two terms are eq. `umod` (and §1.2's off-diagonal extension); the third is what
the crossing angle switches on.

**Check, $\alpha = 90^\circ$ exactly.** Take $\mathbf{n} = \hat{\mathbf{z}}$,
$\mathbf{v} = \beta\hat{\mathbf{z}}$, and the laser incident from $+\hat{\mathbf{x}}$ so that
$\mathbf{e}_0 = -\hat{\mathbf{z}}$ — the polarization now lies *along* the observation
direction, and a dipole must radiate nothing there. Directly from eq. `udef`:

$$
\mathbf{u}_0 = \frac{(\mathbf{n}-\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_0)}{1-\mathbf{v}\cdot\mathbf{n}} - \mathbf{e}_0
= -\hat{\mathbf{z}}\,\frac{1-\beta}{1-\beta} + \hat{\mathbf{z}} = 0 \quad\checkmark
$$

From the formula, with $a_0 = -1$, $b_0 = -\beta$, and $\gamma^2(1-\beta)^2 = (1-\beta)/(1+\beta)$:

$$
|\mathbf{u}_0|^2 = 1 - \frac{1+\beta}{1-\beta} + \frac{2\beta}{1-\beta}
= \frac{(1-\beta) - (1+\beta) + 2\beta}{1-\beta} = 0 \quad\checkmark
$$

Eq. `umod` alone would give $1 - (1+\beta)/(1-\beta) \approx -4\gamma^2$ — negative, which is
the signature of the dropped term rather than a small error. (I first ran this check with
$1-\beta\approx 1/2\gamma^2$ substituted inconsistently and got $-1$; only the exact algebra
above gives $0$. Worth repeating exactly if you check it.)

**Numerically checked** against eq. `udef` over 2000 random geometries — independent
electron direction, observation direction and laser incidence, $\gamma\in[3,3000]$, all
three of $\mathbf{u}_0\cdot\mathbf{u}_0$, $\mathbf{u}_0\cdot\mathbf{u}_1$,
$\mathbf{u}_1\cdot\mathbf{u}_1$ — worst relative error $1.2\times10^{-12}$. This tests the
algebra at arbitrary crossing angle, not just the $\alpha = 90^\circ$ limit above. It does
**not** test the physics: it confirms the expression equals eq. `udef`, and eq. `udef` is
the paper's.

### 2.4 What is still genuinely open

- **Nothing above is validated against an independent calculation.** The plan's own answer
  for this is kascade's arbitrary-angle MC at intermediate angles (§9.3), which does not
  exist yet. Until it does, §2.3 is a derivation with one exact-limit check, not a verified
  result.
- **How far in $\alpha$ the $O(\theta^2)$ reduction survives** is not addressed by any of
  this. Eq. `smallangle` still assumes electron and observation directions cluster about a
  common axis. That holds for a tilted *laser* with an on-axis bunch, but the paper never
  states a bound in $\alpha$, and I have not derived one.
- **Whether `psi_focus`/`psi_pol` retain their meaning** once $\mathbf{e}_0,\mathbf{e}_1$ are
  tilted. §2.2 of the paper pins them as head-on-frame quantities carried through $R$, which
  is self-consistent, but the $\mathbf{n}\cdot\mathbf{e}_i$ and $\mathbf{v}\cdot\mathbf{e}_i$
  above must then be evaluated with the rotated vectors — mechanical, but it is where a sign
  or an axis convention would hide.
- **$\hat a$ is unaffected**, which is worth stating because it is easy to assume otherwise:
  eq. `ahattraj` is a ratio of two integrals over the same trajectory,

  $$
  \hat a(\zeta) = \frac{\operatorname{Tr}\hat\Xi}{2}\,
  \frac{\int\left[a^2(t;\zeta)\right]^2\mathrm{d}t}{\int a^2(t;\zeta)\,\mathrm{d}t},
  $$

  so the change of variable $\varphi = \omega_L(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0)t$
  (eq. `accel`) cancels between numerator and denominator. Given the sampled $a^2(t)$,
  $\hat a$ does not know the crossing angle.

---

## 3. Summary of what would change in code

Nothing in this table is done. Ordered by how much the author needs to weigh in.

| # | Change | Where | Blocked on |
|---|---|---|---|
| 0 | `a0_shape` gains a factor $1/2$; `ahat_decades` retuned | `stages.py`, `schema.py` | **author's confirmation that no compensating $2$ exists elsewhere** |
| 1.1 | Docstrings only — the ellipticity-independence of energy→$a_0$ is derived, not missing | `laser.py` | author agrees with §1.1 |
| 1.2 | $\cos^2\psi \to (\cos^2\psi + \varepsilon^2\sin^2\psi)/(1+\varepsilon^2)$; thread `ellipticity` to the kernels | `stages.py`, `delta.py`, `collision.py` | author agrees with §1.2 |
| 2.1 | `RELATIVE_VELOCITY` becomes $1 + \beta\cos\theta_{xz}\cos\theta_{yz}$ | `stages.py` | author agrees with §2.1 |
| 2.2 | `photon_energy` gains $\cos^2(\alpha/2)$ | `collision.py` | author agrees with §2.2 |
| 2.3 | $\mathbf{u}_i\cdot\mathbf{u}_j$ gains the $\mathbf{v}\cdot\mathbf{e}_i$ term; kernels take the rotated $\mathbf{e}_0,\mathbf{e}_1$ | `stages.py`, `delta.py` | author agrees **and** an independent check exists (§2.4) |

`ELLIPTICITY_IS_NOOP` can only go `False` after 1.2; `EMISSION_IS_HEAD_ON` only after all of
2.1–2.3, since a partial crossing-angle implementation is exactly the half-applied state
D034 exists to warn about.
