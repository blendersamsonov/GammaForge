# DER013 — Direction-dependent Doppler reduction at ultra-relativistic speed

Status: verified

## Setup

The manuscript's Eq. wR gives omega_R = 2 omega_L gamma² F/(1 + ahat + gamma² r²),
where F = 1 - v dot n0. Its later Eq. wRgamma specializes F to 2. For xigma's
existing ballistic speed approximation, take v = e = (theta_ex, theta_ey, 1)/norm.
At fixed electron direction, F is independent of gamma. Define F0 = 1 - n0_z > 0,
D = F/F0, A = 1 + ahat, and s = omega/(2 omega_L F0). All energies remain CGS.
The near-backscattering angular denominator and existing polarization model remain
approximations; this is not a derivation of unrestricted-angle Compton scattering.

## Derivation

The resonance is s = D gamma²/(A + r² gamma²). Consequently

$$\Gamma^2 = \frac{A}{D/s-r^2},\qquad D/s>r^2.$$

For fixed electron direction and ahat, D is constant in this inversion. Differentiation gives

$$\left|\frac{d\Gamma}{ds}\right|=\frac{D\Gamma^3}{2As^2}.$$

Thus the reduced table kernel multiplies its existing factor Gamma^5/(A(1+r² Gamma²)^2)
by D. Correcting only the root, without D in the Jacobian, changes the integrated photon
count by 1/D even when the line's total angular weight was meant to stay fixed.
Stage 0 independently weights each overlap contribution with the encounter flux c F;
this enters luminosity once and is not an additional factor in the Stage-2 cross-section.

For an angle-integrated linear spectrum f0(s), an individual electron's Doppler-shifted
shape is fD(s) = f0(s/D)/D, preserving its luminosity integral. Equivalently its edge
is D gamma² and its density is divided by that edge.

GPU support must use bounds Dlo <= D(x,y) <= Dhi over the complete electron-angle
rectangle, not merely D at the observer. Conservative resonance radii satisfy

$$r_{lo}^2=\max(0,D_{lo}/s-A_{max}/\gamma_{lo}^2),\qquad
r_{hi}^2=\max(0,D_{hi}/s-A_{min}/\gamma_{hi}^2).$$

To bound D, interval-bound q = n0_x x + n0_y y + n0_z and
h = sqrt(1+x²+y²) over the rectangle. The extrema among qlo/hlo, qlo/hhi,
qhi/hlo, qhi/hhi enclose e dot n0. Intersect with [-1,1], convert through
D = (1 - e dot n0)/F0, and pad outward for device floating-point arithmetic.
The sampler evaluates the actual D at each sampled electron direction.

## Result

One direction-dependent factor is used consistently in the encounter flux, resonance,
Jacobian, and the energy scale of the angle-integrated linear approximation. Setting
D = 1 recovers DER009. Retaining exact beta(gamma) would make D gamma-dependent and
require a different inversion and derivative; the finite-beta reference mode is retained
as a diagnostic rather than being labelled identical to this approximation.

## Verification

The symbolic test differentiates the inverse resonance and reduces its difference from
D Gamma³/(2 A s²) to zero. Independent gamma-quadrature line integration checks the
NumPy spectral mass and centroid at head-on and crossed geometry to relative 2e-9.
Separate tests check Stage-0 flux, linear-spectrum mass, and conservative support bounds;
an actual-CUDA stress case emits above the former nominal support cutoff and agrees
with NumPy within 5%. Its amplified angles test the algorithm, not physical validity
of the near-backscattering approximation at those angles.

The scenario-bank matched-bin packet and its scope are recorded in
[the validation record](../../validation/direction-doppler-2026-09-12.md).
Author verification: A. Samsonov explicitly confirmed DER013 as verified on
2026-09-13. Together with the symbolic and numerical checks above, this completes
the derivation confidence pipeline. Broader arbitrary-angle scientific acceptance
remains a separate validation task (RES074).

## Amendments

> **2026-09-23 — Nonlinear incidence coefficient extended by DER014.** The derivation's
> direction-Doppler algebra and linear limit remain current. For nonzero $\hat a$, the
> author-selected beaming-cone model replaces $A=1+\hat a$ by
> $A_P=1+[(1-\mathbf e\cdot\mathbf n_0)/2]\hat a$; DER014 derives and verifies that
> extension.

> **2026-09-28 — DER014 superseded by DER015.** The direction-Doppler algebra and linear
> limit remain current. The nonlinear resonance now uses raw $\hat a$ with the exact
> observer-dependent $Q=(1-\mathbf n\cdot\mathbf n_0)/(1-\mathbf e\cdot\mathbf n_0)$
> evaluated in Stage 2, and $D\bar C$ replaces $D$ for a chirped carrier.


> **2026-09-28 — Plane-wave phase interpretation supplied by DER018.** DER018 derives
> the later observer-dependent nonlinear factor from the radiation phase of an effective
> plane-wave surrogate. It does not alter DER013's direction-Doppler factor \(D\),
> Jacobian, or linear limit. In that derivation the plane-wave light-front invariant is
> \(\kappa=\gamma-\mathbf n_0\cdot\mathbf u\), whose ultra-relativistic ballistic limit is
> \(\kappa\simeq\gamma(1-\mathbf e\cdot\mathbf n_0)\). Combining this with the
> small electron-observer angular reduction yields DER013's \(D\) in the numerator and
> DER015's \(Q\) only in the nonlinear denominator.
