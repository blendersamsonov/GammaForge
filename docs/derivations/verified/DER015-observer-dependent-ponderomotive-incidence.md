# DER015 — Observer-dependent ponderomotive incidence

Status: verified

## Setup

Let the unit electron direction be $\mathbf e$, the laser propagation direction be
$\mathbf n_0$, and the requested observation direction be $\mathbf n$. Production uses
the ultrarelativistic ballistic approximation $\beta=1$. Define

$$
F=1-\mathbf e\mathbin{\cdot}\mathbf n_0,\qquad
F_0=1-n_{0z},\qquad
D=\frac{F}{F_0}.
$$

The nonlinear incidence coefficient is evaluated from the same effective geometry in
Stage 2:

$$
Q=\frac{1-\mathbf n\mathbin{\cdot}\mathbf n_0}
        {1-\mathbf e\mathbin{\cdot}\mathbf n_0}.
$$

Consequently

$$
DQ=\frac{1-\mathbf n\mathbin{\cdot}\mathbf n_0}{F_0}.
$$

Unlike the superseded beaming-cone approximation in DER014, no part of $Q$ is deposited
in Stage 1. That stage stores the raw strength-independent nonlinear shape, and
retargeting produces raw $\hat a$.


## Physical origin and approximation hierarchy

DER018 supplies a trajectory-level derivation of this coefficient. For a plane wave with
fixed propagation direction \(\mathbf n_0\), define dimensionless electron momentum
\(\mathbf u=\mathbf p/(mc)\) and the light-front invariant

$
\kappa=\gamma-\mathbf n_0\mathbin{\cdot}\mathbf u.
$

For an exact plane wave, \(\kappa\) is constant. The radiation phase for a photon of
frequency \(\omega'\) observed along \(\mathbf n\) obeys

$
\frac{d\Psi}{d\phi}
=
\frac{\omega'}{\omega_L}
\frac{\gamma-\mathbf n\mathbin{\cdot}\mathbf u}{\kappa}.
$

The transverse plane-wave motion makes \(\gamma\) and the longitudinal momentum contain
a cycle-averaged term proportional to the nonlinear strength. Its secular contribution
to the phase is proportional to

$
\frac{1-\mathbf n\mathbin{\cdot}\mathbf n_0}{2\kappa^2}.
$

For a weakly deflected ultra-relativistic electron,
\(\kappa\simeq\gamma(1-\mathbf e\cdot\mathbf n_0)=\gamma F\), while for an observation
direction close to the electron direction,

$
\gamma-\mathbf n\mathbin{\cdot}\mathbf u
\simeq
\frac{1+\gamma^2r^2}{2\gamma}.
$

The cycle-averaged first-harmonic resonance therefore has the structure

$
\omega_R
\simeq
\frac{2\omega_L\gamma^2F}
{1+\gamma^2r^2+
 \rho\,\dfrac{1-\mathbf n\cdot\mathbf n_0}{F}},
$

where \(\rho\) denotes the scalar nonlinear line-centre strength of the surrogate
plane-wave problem. GammaForge's trajectory averaging supplies this scalar as the raw
\(\hat a\) defined by DER003/DER016; DER018 derives the geometry multiplying it, not a
new trajectory weighting. Hence

$
Q=\frac{1-\mathbf n\cdot\mathbf n_0}{F}.
$

This separates two approximation layers that are easy to conflate. The dependence of
the plane-wave secular phase on \(1-\mathbf n\cdot\mathbf n_0\) is exact for the
surrogate plane wave. Production \(Q\) additionally uses the ballistic \(\beta=1\)
reduction, a fixed effective laser direction, and the near-electron-direction angular
reduction already assumed by the xigma emission kernel. The focused pulse itself need
not be a global plane wave; the surrogate only requires the field restricted to the
radiation-producing electron trajectory to be sufficiently plane-wave-like.

## Resonance, inverse, and Jacobian

For the trajectory-averaged carrier correction $\bar C$, observation offset

$$
r^2=(\theta_{ex}-\theta_x)^2+(\theta_{ey}-\theta_y)^2,
$$

and

$$
A=1+Q\hat a,
$$

the resonance law is

$$
s_R=\frac{D\bar C\gamma^2}{1+Q\hat a+\gamma^2r^2}.
$$

At fixed table and observation coordinates, $D$, $Q$, $\bar C$, and $A$ are independent
of $s$. The inverse and support condition are therefore

$$
\Gamma^2=\frac{A}{D\bar C/s-r^2},\qquad
D\bar C/s-r^2>0,
$$

and differentiation gives

$$
\left|\frac{d\Gamma}{ds}\right|
=\frac{D\bar C\,\Gamma^3}{2As^2}.
$$

The factor $\bar C$ belongs to the local resonance and Jacobian; it does not alter the
external nominal photon-energy scaling.

## Result

Xigma evaluates the exact observer-dependent $Q$ in Stage 2 from normalized laser,
electron, and observation directions. The table remains observer-independent and stores
raw $\hat a$. Fully collinear head-on geometry ($\mathbf n=\mathbf e=-\mathbf n_0$) gives $Q=1$, and more generally $\mathbf n=\mathbf e$ gives $Q=1$ at any nonsingular incidence.

This result supersedes DER014. DER013's direction-Doppler factor and linear limit remain
current, with $D\bar C$ replacing $D$ in the chirped nonlinear resonance.

## Verification

`tests/test_xigma_doppler.py` compares $Q$ with direct vector dot products, checks the
$DQ$ invariant, head-on and $\mathbf n=\mathbf e$ limits, broadcasting on NumPy/CuPy,
and rejection of the co-propagating singular regime. `tests/test_stage1_stage2.py`
checks the forward resonance against the analytical inverse and the Jacobian against a
finite difference. Independent direct-particle tests in `tests/test_delta_emission.py`
pin the observer-dependent nonlinear line centre, while CPU/CUDA agreement and support
bounds are exercised by the xigma Doppler and sampler suites.


## Clarification on freezing \(Q\) inside one beaming cone

DER018 also quantifies a possible *approximation* that is not used by production. Let
\(\alpha\) be the angle between the electron direction \(\mathbf e\) and laser direction
\(\mathbf n_0\), and let the observation direction lie an angle \(\vartheta\) from
\(\mathbf e\). Then

$
\mathbf n_0\cdot\mathbf n
=
\cos\alpha\cos\vartheta
+\sin\alpha\cos\varphi\sin\vartheta,
$

so

$
Q-1
\simeq
\frac{\tfrac12\cos\alpha\,\vartheta^2
      -\sin\alpha\cos\varphi\,\vartheta}
     {1-\cos\alpha}.
$

Across the relativistic cone \(\vartheta\sim1/\gamma\), the variation is
\(O(1/\gamma^2)\) for exactly head-on incidence but generically \(O(1/\gamma)\) for an
oblique collision. It becomes poorly conditioned near co-propagation, where
\(1-\cos\alpha\) is small. Therefore replacing
\(\mathbf n_0\cdot\mathbf n\) by \(\mathbf n_0\cdot\mathbf e\) can be a controlled
high-\(\gamma\) approximation in selected geometries, but it is not an identity and is
not a reason to remove the exact Stage-2 \(Q\).

## Used by

- `gammaforge.engines.xigma.stages.observer_ponderomotive_factor`
- CPU and CUDA xigma resonance roots, Jacobians, and support bounds
- independent direct-particle delta and moment references
- RES090
