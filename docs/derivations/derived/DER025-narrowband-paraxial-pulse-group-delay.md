# DER025 — Narrowband paraxial pulse group delay and chromatic Gaussian-beam scaling

Status: derived

## Setup

A monochromatic paraxial beam is naturally written as a spatial amplitude times a carrier
phase. For a finite laser pulse it is tempting to reuse that same phase as a retarded-time
coordinate and write the envelope schematically as $T(\Phi/\omega_0)$. This derivation
asks what the correct pulse-front coordinate is in the narrowband paraxial limit, and how
it depends on the chromatic scaling of a Gaussian beam.

The central distinction is between the monochromatic carrier phase and the spectral group
delay. For the positive-frequency convention

$$
E^{(+)}(\mathbf r,t)
=
\int d\omega\,
\widetilde A(\mathbf r,\omega)
\exp\!\left[i\varphi(\mathbf r,\omega)-i\omega t\right],
$$

the narrowband pulse front is controlled by

$$
\tau_g(\mathbf r)
=
\left.
\frac{\partial\varphi(\mathbf r,\omega)}
     {\partial\omega}
\right|_{\omega_0},
$$

not by $\varphi(\mathbf r,\omega_0)/\omega_0$.

The working assumptions are:

- vacuum propagation with no material dispersion;
- a narrow spectrum around a carrier frequency $\omega_0$;
- the paraxial approximation;
- a fundamental circular Gaussian beam with a fixed waist plane;
- a scalar envelope, with polarization effects neglected in the pulse-front derivation;
- the spectral amplitude varies slowly enough with $\omega$ that first-order spectral
  phase is the dominant spatiotemporal effect;
- higher spectral derivatives are small enough that local group-delay dispersion and
  higher-order reshaping can be neglected.

Use GammaForge's laser coordinate $u=\hat{\mathbf k}\cdot(\mathbf r-\mathbf r_f)$
measured from the waist plane, transverse radius $\rho$, focal RMS intensity width
$\sigma$, and

$$
z_R(\omega_0)=\frac{2\omega_0\sigma^2}{c}.
$$

For this circular beam, $w_0=2\sigma$ is the usual $1/e^2$ intensity radius.

This file develops the group-delay result independently and then relates it to current
GammaForge physics. In particular, DER023 already uses the resulting
Gaussian group-delay correction as an applicability bound; the purpose here is to derive
and interpret that correction and to distinguish it from phase divided by frequency.

## 1. Why an arbitrary envelope cannot generally follow a monochromatic phase

Before taking the narrowband limit, consider the stronger scalar ansatz

$$
u(\mathbf r,t)=A(\mathbf r)\,g\!\left(t-\tau(\mathbf r)\right).
$$

Let $\xi=t-\tau(\mathbf r)$. Direct differentiation gives

$$
\nabla^2u
=
(\nabla^2A)g
-
\left[
2\nabla A\cdot\nabla\tau+A\nabla^2\tau
\right]g'
+
A|\nabla\tau|^2g'',
$$

whereas

$$
\frac{1}{c^2}\frac{\partial^2u}{\partial t^2}
=
\frac{A}{c^2}g''.
$$

For this to satisfy the homogeneous wave equation for an arbitrary waveform $g$, the
coefficients of $g$, $g'$, and $g''$ must vanish separately:

$$
\nabla^2A=0,
$$

$$
2\nabla A\cdot\nabla\tau+A\nabla^2\tau=0,
$$

$$
|\nabla\tau|^2=\frac{1}{c^2}.
$$

These conditions are restrictive. A plane wave,
$A=\mathrm{const}$ and $\tau=z/c$, satisfies them. A spherical wave,
$A=1/r$ and $\tau=r/c$, also satisfies them and gives an exact spherical shell

$$
u(\mathbf r,t)=\frac{1}{r}g(t-r/c).
$$

A generic diffracting beam does not.

For comparison, a monochromatic Helmholtz field
$u_0=Ae^{i\phi}$ satisfies

$$
\nabla^2A-A|\nabla\phi|^2+k_0^2A=0,
$$

$$
2\nabla A\cdot\nabla\phi+A\nabla^2\phi=0,
$$

so that

$$
|\nabla\phi|^2
=
k_0^2+\frac{\nabla^2A}{A}.
$$

Diffraction is possible because the amplitude and phase terms compensate each other.
Replacing the harmonic oscillation by an arbitrary waveform
$g[t-\phi/\omega_0]$ would instead require the eikonal and amplitude conditions
separately. Therefore phase divided by frequency is not a general exact pulse coordinate.

In the geometrical-optics/paraxial regime,
$|\nabla^2A/A|\ll k_0^2$, so this construction can become a useful approximation, but
the narrowband spectral derivation below identifies the correct first-order pulse delay.

## 2. Narrowband spectral expansion

Expand the spatial spectral phase around $\omega_0$:

$$
\varphi(\mathbf r,\omega)
=
\varphi_0(\mathbf r)
+
(\omega-\omega_0)\tau_g(\mathbf r)
+
\frac{1}{2}(\omega-\omega_0)^2\varphi_2(\mathbf r)
+\cdots,
$$

with

$$
\tau_g(\mathbf r)
=
\left.
\partial_\omega\varphi(\mathbf r,\omega)
\right|_{\omega_0}.
$$

If the spectral amplitude may be frozen at $\omega_0$ and the terms beginning with
$\varphi_2$ are negligible over the occupied bandwidth, Fourier inversion gives

$$
E^{(+)}(\mathbf r,t)
\simeq
A_0(\mathbf r)
e^{i\varphi_0(\mathbf r)-i\omega_0t}
f\!\left[t-\tau_g(\mathbf r)\right].
$$

Thus the monochromatic carrier phase front is

$$
\varphi_0(\mathbf r)=\mathrm{const},
$$

while the pulse front is

$$
t-\tau_g(\mathbf r)=\mathrm{const}.
$$

An additive phase independent of frequency does not alter $\tau_g$. A linear input
spectral phase adds only a global temporal delay. This immediately shows why
$\varphi/\omega_0$ is not a gauge-invariant definition of propagation time: adding a
frequency-independent phase changes $\varphi/\omega_0$ but leaves
$\partial_\omega\varphi$ unchanged.

## 3. Spectral phase of a circular Gaussian beam

For a fundamental paraxial Gaussian mode, take

$$
\varphi(\rho,u,\omega)
=
k u
+
\frac{k\rho^2}{2R(u,\omega)}
-
\psi(u,\omega)
+
\varphi_{\rm in}(\omega),
$$

where

$$
k=\frac{\omega}{c},
$$

$$
R(u,\omega)
=
u\left[
1+\left(\frac{z_R(\omega)}{u}\right)^2
\right]
=
\frac{u^2+z_R^2(\omega)}{u},
$$

and

$$
\psi(u,\omega)
=
\arctan\!\frac{u}{z_R(\omega)}.
$$

The input spectral phase $\varphi_{\rm in}$ contains any global pulse delay or intrinsic
temporal chirp. The spatial group delay is

$$
\tau_g(\rho,u)
=
\frac{u}{c}
+
\partial_\omega
\left[
\frac{k\rho^2}{2R}
\right]
-
\partial_\omega\psi
+
\varphi_{\rm in}'(\omega_0).
$$

A monochromatic Gaussian beam at $\omega_0$ therefore does not determine the pulse front
by itself: one must specify how its family of spectral constituents changes with
frequency.

## 4. Focused-beam chromatic parameter

Define the logarithmic slope of the focused Rayleigh distance

$$
g_f
\equiv
\left.
\frac{d\ln z_R(\omega)}
     {d\ln\omega}
\right|_{\omega_0}.
$$

Locally,

$$
\frac{dz_R}{d\omega}
=
g_f\frac{z_R}{\omega}.
$$

Since

$$
z_R(\omega)
=
\frac{\omega w_0^2(\omega)}{2c},
$$

one has

$$
g_f
=
1+
2\left.
\frac{d\ln w_0}{d\ln\omega}
\right|_{\omega_0},
$$

and hence, to first logarithmic order,

$$
w_0(\omega)\propto\omega^{(g_f-1)/2}.
$$

The far-field divergence scales as

$$
\theta(\omega)
\simeq
\frac{2c}{\omega w_0(\omega)}
\propto
\omega^{-(g_f+1)/2}.
$$

The parameter $g_f$ is not an additional property of a monochromatic Gaussian beam. It
specifies missing first-order spectral information about how the spatial mode is generated
and focused.

## 5. Group delay of the Gaussian beam

Using

$$
\frac{1}{R}
=
\frac{u}{u^2+z_R^2},
$$

the curvature phase is

$$
\varphi_{\rm curv}
=
\frac{\omega}{c}
\frac{u\rho^2}{2(u^2+z_R^2)}.
$$

Taking the total frequency derivative, including
$dz_R/d\omega=g_fz_R/\omega$, gives

$$
\partial_\omega\varphi_{\rm curv}
=
\frac{u\rho^2}{2c(u^2+z_R^2)}
\left[
1-
\frac{2g_fz_R^2}{u^2+z_R^2}
\right].
$$

For the Gouy phase,

$$
-\partial_\omega\psi
=
\frac{g_fu z_R}
{\omega(u^2+z_R^2)}.
$$

Therefore, at $\omega_0$,

$$
\boxed{
\tau_g(\rho,u)
=
\frac{u}{c}
+
\frac{u\rho^2}{2c(u^2+z_R^2)}
\left[
1-
\frac{2g_fz_R^2}{u^2+z_R^2}
\right]
+
\frac{g_fu z_R}
{\omega_0(u^2+z_R^2)}
+
\varphi_{\rm in}'(\omega_0).
}
$$

The three spatial terms are respectively the plane-wave delay, pulse-front curvature from
the frequency derivative of the wavefront-curvature phase, and the group delay generated
by the chromatic Gouy phase.

In the GammaForge variables used by DER023,

$$
q=\frac{u}{z_R},
\qquad
p^2=\frac{\rho^2}{2\sigma^2},
\qquad
z_R=\frac{2\omega_0\sigma^2}{c},
$$

the correction relative to $u/c$ is

$$
\boxed{
\delta\tau_g(q,p)
=
\frac{q}{\omega_0(1+q^2)}
\left[
g_f
+
\frac{p^2}{2}
\left(
1-\frac{2g_f}{1+q^2}
\right)
\right].
}
$$

This is the formula already used by DER023 to bound the error made when
the group-delay correction is neglected.

## 6. Limiting cases

### 6.1 Fixed focused waist: $g_f=+1$

If $w_0(\omega)$ is frequency independent, then

$$
z_R\propto\omega,
\qquad
\theta\propto\omega^{-1},
$$

and

$$
\tau_g
=
\frac{u}{c}
+
\frac{u\rho^2}{2c}
\frac{u^2-z_R^2}{(u^2+z_R^2)^2}
+
\frac{u z_R}{\omega_0(u^2+z_R^2)}
+
\varphi_{\rm in}'.
$$

The quadratic radial group-delay correction vanishes at $|u|=z_R$.

### 6.2 Isodiffracting beam: $g_f=0$

If the Rayleigh range is frequency independent,

$$
z_R(\omega)=\mathrm{const},
$$

then

$$
w_0\propto\omega^{-1/2},
\qquad
\theta\propto\omega^{-1/2}.
$$

The Gouy phase is frequency independent and gives no first-order group delay:

$$
-\partial_\omega\psi=0.
$$

The result reduces to

$$
\tau_g
=
\frac{u}{c}
+
\frac{u\rho^2}{2c(u^2+z_R^2)}
+
\varphi_{\rm in}'.
$$

### 6.3 Fixed focused divergence: $g_f=-1$

If the focused divergence is frequency independent,

$$
\theta(\omega)=\mathrm{const},
$$

then

$$
w_0\propto\omega^{-1},
\qquad
z_R\propto\omega^{-1}.
$$

This is the scaling produced approximately by an ideal achromatic focusing optic acting on
an incident collimated beam whose transverse radius is frequency independent.

### 6.4 Far field

For $|u|\gg z_R$,

$$
\tau_g
=
\frac{u}{c}
+
\frac{\rho^2}{2cu}
+
O\!\left(\frac{z_R}{\omega_0u}\right)
+
O\!\left(\frac{\rho^2z_R^2}{cu^3}\right).
$$

Since

$$
\sqrt{u^2+\rho^2}
\simeq
u+\frac{\rho^2}{2u}
$$

in the paraxial region, the envelope approaches a spherical shell:

$$
\tau_g
\simeq
\frac{\sqrt{u^2+\rho^2}}{c}.
$$

A pulse of duration $\Delta t$ therefore occupies a shell of radial thickness of order
$c\Delta t$ in this limit.

## 7. Relation to the Porras factor

The ultrafast-beam literature commonly defines the Porras factor $g_0$ for the collimated
input beam through its input Rayleigh distance $Z_R(\omega)$:

$$
g_0
=
\left.
\frac{d\ln Z_R}{d\ln\omega}
\right|_{\omega_0}.
$$

For an ideal achromatic focusing optic of focal length $f$,

$$
Z_R=\frac{\omega W_0^2}{2c},
\qquad
w_0=\frac{2cf}{\omega W_0},
\qquad
z_R=\frac{f^2}{Z_R}.
$$

Hence the focused-beam parameter used here is

$$
\boxed{
g_f=-g_0.
}
$$

The useful limiting-case correspondence is therefore

$$
g_f=+1
\;\Longleftrightarrow\;
g_0=-1
\quad\text{(constant focused waist)},
$$

$$
g_f=0
\;\Longleftrightarrow\;
g_0=0
\quad\text{(constant Rayleigh range / isodiffracting)},
$$

$$
g_f=-1
\;\Longleftrightarrow\;
g_0=+1
\quad\text{(constant focused divergence)}.
$$

The sign must be stated explicitly whenever a model parameter is taken from the
input-beam literature but applied to the focused field.

Relevant literature includes M. A. Porras, Opt. Lett. 34, 1546 (2009),
doi:10.1364/OL.34.001546; M. A. Porras and R. García-Álvarez,
Phys. Rev. A 102, 033522 (2020), doi:10.1103/PhysRevA.102.033522; and
S. W. Jolly, Opt. Lett. 45, 3865 (2020), doi:10.1364/OL.394493.

## 8. Why phase divided by frequency is not group delay

At the carrier frequency, the Gaussian spatial phase correction relative to the plane
wave is

$$
\delta\varphi_0
=
\frac{\omega_0\rho^2}{2cR}
-
\psi.
$$

Using phase divided by carrier frequency as a pulse delay would give

$$
\delta\tau_\phi
=
\frac{\delta\varphi_0}{\omega_0}
=
\frac{\rho^2}{2cR}
-
\frac{\psi}{\omega_0}.
$$

The physical narrowband correction is instead $\delta\tau_g=\partial_\omega
\delta\varphi|_{\omega_0}$. The two are equal only under additional special conditions.

A particularly transparent example is the isodiffracting case $g_f=0$. Then

$$
\delta\tau_g
=
\frac{\rho^2}{2cR},
$$

because the Gouy phase is independent of frequency, whereas

$$
\delta\tau_\phi
=
\frac{\rho^2}{2cR}
-
\frac{\psi}{\omega_0}.
$$

The extra $-\psi/\omega_0$ has no group-delay interpretation: it arises solely from
dividing a frequency-independent phase by $\omega_0$.

This distinction is also invariant under an arbitrary frequency-independent phase shift.
Such a shift changes $\varphi/\omega_0$ but cannot change a physical pulse arrival time,
while $\partial_\omega\varphi$ is unchanged.

## 9. Relation to current GammaForge physics

### DER023

DER023 already states the same $\delta\tau_g(q,p)$ and uses it to construct a
conservative criterion for neglecting diffraction-induced pulse-front delay in a
narrowband Gaussian Stage-0 fast path. The present derivation supplies the underlying
spectral-phase argument, the chromatic limiting cases, the Porras-factor mapping, and the
distinction from phase divided by frequency. It does not supersede DER023.

### DER016

DER016 defines the encountered-carrier factor $C(t)$ from an explicitly supplied
additional carrier phase $\delta\Phi$ and states that the built-in unchirped lasers do not
implicitly include Gouy or wavefront-curvature phase in that correction. The group delay
derived here is a different object: it governs the envelope location through the
frequency derivative of the spatial spectral phase.

Therefore the present result should not be inserted into DER016 by simply reinterpreting
$C$. If both intrinsic carrier chirp and diffraction-induced pulse-front delay are modeled,
their conventions must be made consistent so neither is omitted nor double counted.

### Current SeparableParaxialLaser implementation

Current main evaluates the temporal envelope using a phase-time coordinate obtained from
the full monochromatic paraxial phase divided by $\omega_0$. That phase contains the
wavefront-curvature and Gouy terms.

The derivation above shows that this is not, in general, the narrowband group-delay
coordinate. In particular, a frequency-independent Gouy phase contributes to
$\varphi/\omega_0$ but contributes zero to $\partial_\omega\varphi$.

This is a repository/code discrepancy identified by the derivation, not resolved here.
No implementation change is authorized by this record.

## Result

For a narrowband paraxial pulse, the first-order local pulse delay is

$$
\boxed{
\tau_g(\mathbf r)
=
\left.
\partial_\omega\varphi(\mathbf r,\omega)
\right|_{\omega_0},
}
$$

not $\varphi(\mathbf r,\omega_0)/\omega_0$.

For a circular Gaussian beam with fixed waist plane and focused-beam chromatic slope

$$
g_f=
\left.
\frac{d\ln z_R}{d\ln\omega}
\right|_{\omega_0},
$$

the group delay is

$$
\boxed{
\tau_g(\rho,u)
=
\frac{u}{c}
+
\frac{u\rho^2}{2c(u^2+z_R^2)}
\left[
1-
\frac{2g_fz_R^2}{u^2+z_R^2}
\right]
+
\frac{g_fu z_R}{\omega_0(u^2+z_R^2)}
+
\varphi_{\rm in}'(\omega_0).
}
$$

The focused parameter is related to the standard incident-beam Porras factor by
$g_f=-g_0$ for ideal achromatic focusing.

## Implementation implications

This derivation does not require a change to the xigma table dimensionality, Stage 1, the
Stage-2 resonance inverse, or its Jacobian. The effect belongs to the laser field sampled
in Stage 0.

If the diffraction-induced pulse-front correction is to be represented rather than
bounded away as in DER023, the laser model needs first-order spectral information beyond its
single-frequency spatial profile. For the circular Gaussian case, one additional
chromatic quantity $g_f$ is sufficient under the assumptions above.

The current SeparableParaxialLaser phase-time construction should not be interpreted as a
group-delay construction without an additional approximation argument. A future
implementation could instead expose an analytic group-delay/phase-time mapping at the
laser-model boundary, or derive it from richer spectral beam metadata.

The physical value of $g_f$ belongs to the laser/focusing model, not to xigma. Hard-coding
one of $+1,0,-1$ into the engine would silently assume a particular chromatic focusing
geometry.

Astigmatic beams require separate treatment of the two transverse Rayleigh distances and,
if present, chromatic focal-plane shifts. Those extensions are not derived here.

## Validation ideas

1. **Finite-difference spectral phase.** Construct a synthetic Gaussian family with
   $z_R(\omega)\propto\omega^{g_f}$ and verify that
   $[\varphi(\omega_0+\delta\omega)-\varphi(\omega_0-\delta\omega)]/(2\delta\omega)$
   converges to the analytic $\tau_g$.

2. **Isodiffracting limit.** At $g_f=0$, verify that the Gouy contribution to group delay
   vanishes identically while the curvature contribution remains.

3. **Fixed-waist limit.** At $g_f=+1$, verify that the quadratic radial group-delay
   correction vanishes at $|u|=z_R$.

4. **Far-field geometry.** Verify that
   $\tau_g-u/c\rightarrow\rho^2/(2cu)$ and compare with
   $[\sqrt{u^2+\rho^2}-u]/c$ within paraxial accuracy.

5. **Direct narrowband propagation.** Inverse-Fourier-transform a narrowband spectral
   Gaussian beam, extract the envelope-peak time in space, and compare it with
   $\tau_g(\rho,u)$.

6. **Bandwidth convergence.** Reduce $\Delta\omega/\omega_0$ and verify convergence of
   the full spectral propagation toward the translated-envelope approximation; the
   residual should be controlled by the neglected higher spectral derivatives.

7. **Phase-time discrepancy.** Compare the present $\tau_g$ against the current
   monochromatic phase-time coordinate used by SeparableParaxialLaser. In particular,
   pin the $g_f=0$ Gouy difference so that the two concepts cannot be conflated.

8. **Consistency with DER023.** Reduce the result to $(q,p)$ variables and recover the
   exact group-delay correction already used in DER023.

## Open questions

- What chromatic focusing model should GammaForge's built-in Gaussian laser represent in
  physical applications: constant focused waist, isodiffracting propagation, constant
  focused divergence, or a measured/derived intermediate $g_f$?

- Should the public laser interface expose $g_f$ directly, expose the incident-beam
  Porras factor $g_0$, or expose a more physical spectral beam model from which the
  derivative is calculated?

- The current Gaussian implementation supports astigmatism. What is the appropriate
  extension when $z_{Rx}(\omega)$ and $z_{Ry}(\omega)$ have different chromatic slopes?

- How should chromatic focal-position shifts $z_f(\omega)$ be represented? They produce
  additional group-delay terms not present in the fixed-waist-plane formula.

- Should the existing phase-time envelope in SeparableParaxialLaser be replaced by a
  group-delay coordinate, retained as an explicitly documented approximation, or used
  only for the period-resolved field while the intensity envelope uses group delay?

- How should diffraction-induced group delay be combined with intrinsic temporal chirp
  and DER016's additional carrier-phase gradient without double counting?

- At what bandwidth and focusing strength should the first-order narrowband model be
  rejected in favor of full spectral propagation?

## Used by

- DER023 group-delay applicability bound
- future Gaussian laser pulse-front modeling
- future reconciliation of SeparableParaxialLaser phase-time and physical group delay
