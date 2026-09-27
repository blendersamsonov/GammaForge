# DER014 — Beaming-cone ponderomotive incidence factor

Status: verified

## Setup

Let the unit electron direction be $\mathbf e$, the laser propagation direction be
$\mathbf n_0$, and the observation direction be $\mathbf n$. Production uses the
ultrarelativistic ballistic approximation $\beta=1$. Define

$$
F=1-\mathbf e\mathbin{\cdot}\mathbf n_0,qquad
F_0=1-n_{0z},qquad D=F/F_0.
$$

The unspecialized retarded-phase calculation gives the nonlinear numerator coefficient
$(1-\mathbf n\cdot\mathbf n_0)/2$. The author selected the controlled beaming-cone
closure $\mathbf n\simeq\mathbf e$ for this coefficient: radiation from one electron is
concentrated within an $O(1/\gamma)$ cone around its velocity, while the explicit
$\gamma^2r^2$ term retains the observer displacement inside that cone. Therefore

$$
P=\frac{1-\mathbf e\cdot\mathbf n_0}{2}
 =\frac{F}{2}=\frac{F_0D}{2}.
$$

This factor multiplies the trajectory-averaged $\hat a$; it does not multiply the field
amplitude $a_0$, and it does not change the definition of $\hat a$ itself.

## Resonance, inverse, and Jacobian

With $A_P=1+P\hat a$ and normalized photon energy
$s=\omega/(2\omega_LF_0)$, the reduced resonance is

$$
s=\frac{D\gamma^2}{A_P+r^2\gamma^2}.
$$

At fixed electron direction and $\hat a$, both $D$ and $P$ are independent of $s$, so

$$
\Gamma^2=\frac{A_P}{D/s-r^2},\qquad
\left|\frac{d\Gamma}{ds}\right|
=\frac{D\Gamma^3}{2A_Ps^2}.
$$

Thus DER013's inverse and Jacobian remain valid after replacing its
$A=1+\hat a$ by $A_P=1+P\hat a$.

## Table coordinate and CUDA support bounds

Stage 1 evaluates $P$ for every trajectory and deposits the intensity-independent
coordinate

$$q_{\mathrm{shape}}=P\,a_{0,\mathrm{shape}}.$$

Retargeting by the requested peak intensity therefore makes the Stage-2 table coordinate
$q=P\hat a$. The table retains its historical `ahat` field names, but the values already
include $P$. Consequently the inverse used by both NumPy and CUDA is

$$
\Gamma^2=\frac{1+q}{D/s-r^2},
$$

and Stage 2 must not multiply the table coordinate by another incidence factor.

For table bounds $q_{min}\le q\le q_{max}$, conservative radial limits are

$$
r_{lo}^2=\max\left(0,\frac{D_{lo}}s-
 \frac{1+q_{max}}{\gamma_{lo}^2}\right),
$$

$$
r_{hi}^2=\max\left(0,\frac{D_{hi}}s-
 \frac{1+q_{min}}{\gamma_{hi}^2}\right).
$$

The deposited four-dimensional density preserves the correlation among electron angle,
$D$, and $q$ to the resolution of the table. The bounds deliberately combine their
extrema conservatively only to delimit sampling support.

## Result

The nonlinear redshift uses $P\hat a$ with
$P=(1-\mathbf e\cdot\mathbf n_0)/2$. Observation angle continues to enter through
$r^2$, while the ponderomotive incidence coefficient is fixed by that electron's
direction. Head-on collinear geometry recovers $P=1$, and the linear $\hat a=0$ limit is
unchanged.

This is an author-selected ultrarelativistic beaming-cone approximation, not the exact
observer-dependent expression away from the cone. It partially extends DER005 and
DER013; their polarization and linear direction-Doppler results remain current.

## Verification

A direct unit-vector test checks the implementation against
$P=(1-\mathbf e\cdot\mathbf n_0)/2$ for multiple electron directions and verifies the
head-on limit. A deposition test verifies that the fourth ShapeTable coordinate is
$P a_{0,\mathrm{shape}}$ per trajectory, rather than a Stage-2 cell-centre correction.
Independent long-double delta emission pins the resulting nonlinear line
energy, including the fact that changing a diagnostic numerator convention does not
change $P$. Gamma quadrature checks NumPy spectral mass and centroid at head-on and
crossed geometry. The CUDA delta kernel agrees with the long-double reference, and the
production CUDA sampler passes the focused CPU/GPU test set with the corrected table
coordinate and support bounds. A. Samsonov selected and confirmed the beaming-cone
substitution on 2026-09-23, and specified its deposition-stage placement afterward.

## Used by

- `gammaforge.engines.xigma.stages.ponderomotive_incidence_factor`
- xigma Stage-1 shape deposition and retargeting
- NumPy and CuPy xigma resonance roots, Jacobians, proposals, and support bounds consuming
  the already-corrected table coordinate
- CPU long-double, simple histogram, and CuPy delta references
- RES088
