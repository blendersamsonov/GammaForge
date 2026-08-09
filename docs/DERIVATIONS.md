# Derivations

Long-form physics derivations that back specific code, kept out of docstrings because they
need more than a paragraph. Each section states its status plainly: whether it is
**implemented** or **pending author review**, and what evidence backs it.

Math is MathJax (`$…$` / `$$…$$`), so it renders in Obsidian and pastes into
`~/Work/Papers/2026/Compton-Numerics/xigma.tex` with only environment changes.

> **Merge note.** A parallel Phase 3b session created a file of the same name holding
> §9.2/§9.3 material (numbered sections 0–3, all *pending review, none implemented*).
> The sections here are lettered (§A, §B) precisely so the two files can be concatenated
> without renumbering — resolve any add/add conflict by keeping both, in either order.

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
The Rayleigh range carries the repository's own $(1+\beta_{\rm ff})$ stretch
(`rayleigh_x`), which is a modelling convention inherited from the predecessor's xigma,
not something re-derived here.

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
