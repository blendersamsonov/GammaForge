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
raw $\hat a$. Head-on collinear geometry and $\mathbf n=\mathbf e$ both give $Q=1$.

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

## Used by

- `gammaforge.engines.xigma.stages.observer_ponderomotive_factor`
- CPU and CUDA xigma resonance roots, Jacobians, and support bounds
- independent direct-particle delta and moment references
- RES090
