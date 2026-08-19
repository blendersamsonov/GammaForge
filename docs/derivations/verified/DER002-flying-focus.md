# DER002 — Flying focus

Status: verified

## Setup

DER001's guard against $\beta_{\rm ff}\neq0$ is removed here.

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

In DER001 the two spot sizes depended on $u$ alone, hence on $z$ alone head-on, and $t$
entered only the exponent — which is why DER001 §A.3 could integrate it. With
$\beta_{\rm ff}\neq0$ the *coefficients* of the quadratic form depend on $t$, and that
step is gone.

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
$\beta_{\rm ff}=0$, which is DER001's 1D case. The two effects each add one
width-argument, and having both does not add a third.

## Result

### B.3 The implemented form

Head-on the natural pair is $(z,w)$ with $w=ct$, since $u_{\rm spot}=-z+\beta_{\rm ff}w$ is
then a function of them alone; the analytic pair is $(x,y)$. Integrating those at fixed
$(z,w)$ gives $2\pi/\sqrt{\det\mathsf A}$ exactly as in DER001 §A.6 but **without**
eliminating time, so the exponent keeps its explicit $w$ terms:

$$
\boxed{\;
N=\frac{\sigma_T(1+\beta_0)N_eN_L}{4\pi^2\sigma_{ez}s_{ct}}
\iint \mathrm{d}z\,\mathrm{d}w\;
\frac{\exp\!\left[-\dfrac{(z-\beta_0w)^2}{2\sigma_{ez}^2}
-\dfrac{(z+w)^2}{2s_{ct}^2}\right]}
{\sqrt{\det\!\big(C_e(z)+C_l(-z+\beta_{\rm ff}w)\big)}}\;}
$$

With a crossing angle the same $(z,w)$ pair is used and the residual transverse part of
$u_{\rm spot}$, $\delta=k_xx+k_yy$, is sampled at the axis — the *same* approximation
DER001 §A.7 already measures at $1.9\times10^{-4}$. A fully exact crossing-plus-flying-focus
evaluation would quadrature $(A,B)$ of §B.2 instead; the counting says it costs no more
dimensions, only more coordinate bookkeeping.

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

so that DER001's 1D machinery applies unchanged. **It should not be trusted.** Unlike
DER001 §A.7's approximation — where the dropped term entered an even, slowly varying
prefactor and cancelled to first order — a flying focus exists precisely to correlate the
width with time, so replacing a distribution over $w$ by its mean errs at *first* order.
Measured against §B.3 on the baseline scenario:

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
$\beta_{\rm ff}\neq0$ (RES044).

## Verification

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
this file is validated against the Monte Carlo only. A comparison against an independent
analytic derivation would be worth more than another numerical run — particularly on the
$(1+\beta_{\rm ff})$ Rayleigh convention, which this derivation inherits from the code
rather than deriving, and which the $\beta_{\rm ff}\to1/\beta_{\rm ff}$ symmetry above
depends on directly.

## Used by

`beta_ff != 0` is routed through
`gammaforge.engines.analytical.formulas._reduced_integral_flying_focus`, validated against
the brute-force Monte Carlo at $\beta_{\rm ff}\in\{-0.5,0.5,1,2\}$ to ${\sim}10^{-3}$,
with and without a crossing angle. RES044 records the decision to always take the 2D path
when $\beta_{\rm ff}\neq0$ rather than shipping §B.4's unsafe 1D shortcut.
