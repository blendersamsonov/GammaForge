# DER022 — Quantum recoil correction in the 5D xigma resonance

Status: derived

## Setup

GammaForge's xigma engine currently uses a classical, narrow-line, first-harmonic
resonance built from DER013, DER015, DER016, DER017, and DER018. In the notation of
DER018, let

- \(\mathbf e\) be the electron direction;
- \(\mathbf n_0\) be the effective laser propagation direction;
- \(\mathbf N\) be the photon observation direction;
- \(F=1-\mathbf e\cdot\mathbf n_0\);
- \(F_0=1-n_{0z}\);
- \(D=F/F_0\);
- \(G=1-\mathbf N\cdot\mathbf n_0\);
- \(Q=G/F\);
- \(r\) be the small electron-observer angle used by the production emission kernel;
- \(\hat a\) be the trajectory-reduced nonlinear strength from DER003/DER016;
- \(\bar C\) be the trajectory-averaged encountered-carrier factor from DER016;
- \(s=\omega/(2\omega_0F_0)\) be xigma's dimensionless photon-frequency coordinate.

The classical production resonance is

$$
s_R=
\frac{D\bar C\,\gamma^2}
     {1+Q\hat a+\gamma^2r^2}.
$$

The question is whether the leading Compton recoil correction can be added while
preserving the current five-dimensional reusable table

$$
(\gamma,\theta_{e,x},\theta_{e,y},\hat a,\bar C),
$$

or whether quantum recoil introduces an additional independent coordinate.

A terminology distinction is important. The extra term in the resonance denominator is
a **Compton recoil kinematic correction** obtained from four-momentum conservation. It is
part of the physics underlying the Klein-Nishina result, but adding this term alone is not
equivalent to replacing the Thomson emission probability, angular distribution, or
polarization transfer by the full Klein-Nishina kernel.

The derivation uses the same approximation hierarchy as the present xigma resonance:

- ultra-relativistic electrons, \(\gamma\gg1\);
- ballistic production geometry, \(\boldsymbol\beta\simeq\mathbf e\);
- weak trajectory deflection;
- the effective-plane-wave surrogate of DER018;
- the unresolved first harmonic;
- the near-electron observation geometry in which
  \(\gamma-\mathbf N\cdot\mathbf u\simeq(1+\gamma^2r^2)/(2\gamma)\);
- for the nominal scalar-carrier result, a local carrier four-wavevector that is a scalar
  rescaling of the nominal one.

The last assumption is written

$$
k_L^\mu=C\,k_0^\mu.
$$

It permits temporal or scalar carrier-frequency variation while keeping the local carrier
direction fixed. A general spatial phase gradient is treated separately below.

Define the dimensionless reference laser photon energy

$$
\epsilon_0=\frac{\hbar\omega_0}{m_ec^2}.
$$

This is metadata fixed by the reference carrier frequency, not a particle-distribution
coordinate.

## 1. Classical effective-plane-wave phase matching

The geometry of the recoil correction is easiest to understand by starting from the
classical radiation-phase construction of DER018.

Set \(c=1\) temporarily and define

$$
\ell^\mu=(1,\mathbf n_0),
\qquad
n_N^\mu=(1,\mathbf N),
$$

together with the incident and emitted wavevectors

$$
k^\mu=\omega_L\ell^\mu,
\qquad
k'^\mu=\omega'n_N^\mu.
$$

Along the trajectory,

$$
\phi=k\cdot x,
\qquad
\Psi=k'\cdot x,
$$

so

$$
\frac{d\Psi}{d\phi}
=
\frac{k'\cdot u}{k\cdot u}.
$$

DER018 packages the cycle-averaged plane-wave motion into the quasi-momentum

$$
q^\mu
=
p_0^\mu+\frac{m\rho}{2\kappa}\ell^\mu,
\qquad
\kappa=\ell\cdot u,
$$

for which

$$
q^2=m^2(1+\rho),
\qquad
\ell\cdot q=m\kappa.
$$

The first-harmonic classical secular phase condition is therefore

$$
\boxed{k'\cdot q=k\cdot q}.
$$

Equivalently,

$$
\omega'_{\rm cl}
=
\omega_L
\frac{\ell\cdot q}
     {n_N\cdot q}.
$$

Since

$$
\frac{n_N\cdot q}{m}
=
\gamma-\mathbf N\cdot\mathbf u
+
\frac{\rho}{2\kappa}G,
$$

the exact effective-plane-wave line centre is

$$
\omega'_{\rm cl}
=
\frac{\omega_L\kappa}
{\gamma-\mathbf N\cdot\mathbf u
 +\dfrac{\rho}{2\kappa}G}.
$$

With

$$
\kappa\simeq\gamma F,
\qquad
\gamma-\mathbf N\cdot\mathbf u
\simeq
\frac{1+\gamma^2r^2}{2\gamma},
$$

this becomes

$$
\boxed{
\omega'_{\rm cl}
\simeq
\frac{2\omega_L\gamma^2F}
{1+\gamma^2r^2+Q\rho}
}.
$$

Thus the complete geometric structure
\(1+\gamma^2r^2+Q\rho\) is already classical and is exactly the structure recovered by
DER018.

## 2. The quantum modification of phase matching

For first-harmonic Compton scattering in the same effective plane wave, impose
four-momentum conservation on the cycle-averaged quasi-momenta,

$$
q+\hbar k=q'+\hbar k',
$$

with

$$
q'^2=q^2,
\qquad
k^2=k'^2=0.
$$

Squaring the conservation equation gives

$$
2\hbar q\cdot k
-2\hbar q\cdot k'
-2\hbar^2k\cdot k'
=0,
$$

hence

$$
\boxed{
q\cdot k
=
q\cdot k'
+
\hbar k\cdot k'
}.
$$

This differs from the classical phase condition in exactly one term. Since

$$
k\cdot k'
=
\omega_L\omega'G
$$

in \(c=1\) units, the recoil-corrected line centre is

$$
\omega'_R
=
\frac{\omega_L\kappa}
{\gamma-\mathbf N\cdot\mathbf u
 +\dfrac{\rho}{2\kappa}G
 +\dfrac{\hbar\omega_L}{m_ec^2}G}.
$$

The geometry therefore remains the same effective-plane-wave geometry. Quantum mechanics
adds the finite energy-momentum transfer associated with one emitted quantum.

## 3. Scalar carrier variation

Assume the local incident wavevector is

$$
k_L^\mu=C\,\omega_0\ell^\mu
$$

in \(c=1\) notation. Then the local incident carrier frequency is
\(\omega_L=C\omega_0\), and the line centre becomes

$$
\omega'_R
=
\frac{\omega_0C\,\kappa}
{\gamma-\mathbf N\cdot\mathbf u
 +\dfrac{\rho}{2\kappa}G
 +\epsilon_0CG}.
$$

Using the same ultra-relativistic reduction as DER018 and multiplying numerator and
denominator by \(2\gamma\),

$$
\boxed{
\omega'_R
\simeq
\frac{2\omega_0C\,\gamma^2F}
{1+\gamma^2r^2
 +Q\rho
 +2\gamma\epsilon_0CG}
}.
$$

At the nominal production line centre,

$$
\rho\longrightarrow\hat a,
\qquad
C\longrightarrow\bar C.
$$

Therefore

$$
\boxed{
s_R
=
\frac{D\bar C\,\gamma^2}
{1+Q\hat a+\gamma^2r^2
 +2\gamma\epsilon_0\bar C\,G}
}.
$$

The recoil term can also be written using the usual collision invariant

$$
x=2\gamma\epsilon_0\bar C\,F,
$$

because

$$
Qx
=
2\gamma\epsilon_0\bar C\,G.
$$

Hence an equivalent form is

$$
s_R
=
\frac{D\bar C\,\gamma^2}
{1+\gamma^2r^2+Q(\hat a+x)}.
$$

The common factor \(Q\) reflects geometry, not a common physical origin:
\(\hat a\) is the nonlinear plane-wave mass/phase correction, whereas \(x\) is the
single-quantum recoil invariant.

## 4. Why this does not add a sixth table dimension

The current table coordinates are

$$
(\gamma,\theta_{e,x},\theta_{e,y},\hat a,\bar C).
$$

For a requested observation direction, the recoil coefficient depends on

$$
\epsilon_0,
\qquad
\bar C,
\qquad
G=1-\mathbf N\cdot\mathbf n_0,
\qquad
\gamma.
$$

Every one of these is already available without a new deposited coordinate:

- \(\gamma\) is the existing first table axis;
- \(\bar C\) is the existing fifth table axis;
- \(G\) is Stage-2 observer geometry;
- \(\epsilon_0\) is fixed by the laser reference frequency.

Thus recoil changes the resonance function but introduces no new independent random
variable in the scalar-carrier model.

The reusable table therefore remains five-dimensional. The reference carrier frequency,
or equivalently \(\epsilon_0\), should be carried as scalar collision/table/query metadata
rather than as an axis.

## 5. Analytical inverse resonance

Define

$$
A=1+Q\hat a,
\qquad
K=D\bar C,
\qquad
L=2\epsilon_0\bar C\,G.
$$

Then

$$
s
=
\frac{K\gamma^2}
{A+L\gamma+r^2\gamma^2}.
$$

Rearranging gives

$$
\left(\frac{K}{s}-r^2\right)\gamma^2
-L\gamma-A=0.
$$

Define

$$
U=\frac{K}{s}-r^2.
$$

For the physical GammaForge regime \(A>0\) and \(L\ge0\), positive support requires

$$
\boxed{U>0}.
$$

The physical root is

$$
\boxed{
\Gamma
=
\frac{L+\sqrt{L^2+4AU}}{2U}
}.
$$

When recoil vanishes, \(L\to0\), this reduces exactly to the current inverse,

$$
\Gamma^2=\frac{A}{U}.
$$

The forward resonance is monotonic in \(\gamma\):

$$
\frac{ds}{d\gamma}
=
\frac{K\gamma(2A+L\gamma)}
     {(A+L\gamma+r^2\gamma^2)^2}
>0,
$$

so there is a unique positive resonant root over the supported production domain.

## 6. Jacobian

Differentiating

$$
U\Gamma^2-L\Gamma-A=0
$$

with

$$
\frac{dU}{ds}=-\frac{K}{s^2}
$$

gives

$$
(2U\Gamma-L)\frac{d\Gamma}{ds}
=
\frac{K\Gamma^2}{s^2}.
$$

Using

$$
U\Gamma^2=L\Gamma+A
$$

yields

$$
2U\Gamma-L
=
L+\frac{2A}{\Gamma}.
$$

Therefore

$$
\boxed{
\left|\frac{d\Gamma}{ds}\right|
=
\frac{K\Gamma^3}
{s^2(2A+L\Gamma)}
}.
$$

The classical limit \(L\to0\) is

$$
\left|\frac{d\Gamma}{ds}\right|
\longrightarrow
\frac{K\Gamma^3}{2As^2},
$$

which is exactly the DER013/DER015 Jacobian used by the present query path.

## 7. Second-order finite-line moments with recoil

DER016 and DER017 retain fluctuations of the local nonlinear strength
\(q(t)=\langle a^2(t)\rangle\) and encountered carrier \(C(t)\). To avoid confusion with
the quasi-momentum \(q^\mu\), call the DER016 scalar \(q_{\rm nl}\) in this section.

At fixed \(\gamma\) and observation geometry, the local recoil-corrected resonance has
the functional form

$$
s(C,q_{\rm nl})
\propto
\frac{C}
{b_0+Qq_{\rm nl}+\eta C},
$$

where

$$
b_0=1+\gamma^2r^2,
\qquad
\eta=2\gamma\epsilon_0G.
$$

At the nominal trajectory means define

$$
T
=
1+\gamma^2r^2+Q\hat a+\eta\bar C,
$$

and the fractional recoil share of the nominal denominator

$$
\zeta
=
\frac{\eta\bar C}{T}.
$$

The first-order fractional line displacement is

$$
\boxed{
\frac{\delta s}{s}
=
(1-\zeta)\frac{\delta C}{\bar C}
-
\frac{Q\,\delta q_{\rm nl}}{T}
}.
$$

Consequently the central fractional variance becomes

$$
\boxed{
m_2
=
(1-\zeta)^2
\frac{\operatorname{Var}(C)}{\bar C^2}
+
\frac{Q^2\operatorname{Var}(q_{\rm nl})}{T^2}
-
\frac{2Q(1-\zeta)\operatorname{Cov}(q_{\rm nl},C)}
     {T\bar C}
}.
$$

Expanding the resonance through second order gives the fractional centroid correction

$$
\boxed{
\Delta_c
=
\frac{Q^2\operatorname{Var}(q_{\rm nl})}{T^2}
+
\frac{Q(2\zeta-1)\operatorname{Cov}(q_{\rm nl},C)}
     {T\bar C}
-
\zeta(1-\zeta)
\frac{\operatorname{Var}(C)}{\bar C^2}
}.
$$

As in DER017,

$$
\mu_1=s_R\Delta_c,
\qquad
\mu_2=s_R^2m_2,
$$

and at an inverted query root \(s_R=s\).

No additional second-order stochastic channel appears: the recoil-aware expansion still
closes on the three quantities already stored by DER016/DER017,

$$
\operatorname{Var}(q_{\rm nl}),
\qquad
\operatorname{Var}(C),
\qquad
\operatorname{Cov}(q_{\rm nl},C).
$$

Thus neither the nominal recoil shift nor the present second-order finite-line model
requires a sixth table coordinate.

## 8. Classical limit of the moment formulas

For

$$
\epsilon_0\to0,
$$

one has

$$
\eta\to0,
\qquad
\zeta\to0,
\qquad
T\to B=1+Q\hat a+\gamma^2r^2.
$$

Then

$$
m_2
\to
\frac{\operatorname{Var}(C)}{\bar C^2}
+
\frac{Q^2\operatorname{Var}(q_{\rm nl})}{B^2}
-
\frac{2Q\operatorname{Cov}(q_{\rm nl},C)}
     {B\bar C},
$$

and

$$
\Delta_c
\to
\frac{Q^2\operatorname{Var}(q_{\rm nl})}{B^2}
-
\frac{Q\operatorname{Cov}(q_{\rm nl},C)}
     {B\bar C},
$$

which are exactly DER017.

## 9. An almost-classical interpretation

The recoil geometry can be reproduced without invoking a photon at the start if one
considers a finite null wave packet with four-momentum

$$
P_{\rm wave}^\mu=Jk^\mu,
$$

where

$$
J=\frac{E}{\omega}
$$

is the packet action. If one incoming packet is converted into one outgoing packet with
the same action \(J\),

$$
q+Jk=q'+Jk'.
$$

The same mass-shell calculation gives

$$
q\cdot k=q\cdot k'+J\,k\cdot k',
$$

and therefore a recoil-like denominator proportional to

$$
2\gamma\frac{J\omega_L}{m_ec^2}G.
$$

The Compton result follows from the one genuinely quantum identification

$$
\boxed{J=\hbar}.
$$

This is useful conceptually but is not a classical derivation of Compton recoil.
Classical electrodynamics allows a wave packet to carry arbitrary \(E/\omega\), so it
does not select a universal recoil shift. The universal dimensionless scale

$$
\frac{\hbar\omega_L}{m_ec^2}
$$

contains the electron Compton wavelength and cannot be generated by pure
Maxwell-Lorentz dynamics.

The practical interpretation for GammaForge is therefore

$$
\text{classical effective-plane-wave geometry}
+
\text{one finite-quantum modification of phase matching}.
$$

The classical condition

$$
k'\cdot q=k\cdot q
$$

is changed to

$$
k'\cdot q+\hbar k\cdot k'=k\cdot q.
$$

All of the existing \(F\), \(D\), \(Q\), \(\hat a\), and quasi-momentum geometry survives.

## 10. General spatial carrier phase gradients

The scalar-carrier result above assumes

$$
k_L^\mu=Ck_0^\mu.
$$

DER016 allows a more general laser phase,

$$
\Phi_L
=
\omega_0
\left(t-\mathbf n_0\cdot\mathbf r/c\right)
+
\delta\Phi.
$$

Along a ballistic electron trajectory, the scalar currently stored by xigma is

$$
C_e(t)
=
1+
\frac{\partial_t\delta\Phi
+c\,\mathbf e\cdot\nabla\delta\Phi}
{\omega_0F}.
$$

This is the additional phase rate contracted with the electron trajectory.

The recoil product \(k_L\cdot k'\), however, needs the contraction with the photon
direction. In the same sign convention define

$$
C_N(t)
=
1+
\frac{\partial_t\delta\Phi
+c\,\mathbf N\cdot\nabla\delta\Phi}
{\omega_0G}.
$$

If the local carrier four-vector is only rescaled, then

$$
C_e=C_N=C.
$$

For a genuinely spatially varying carrier wavevector, \(C_e\) and \(C_N\) are different.
The current scalar \(\bar C\), which is built from the electron contraction, then does
not contain enough information to reconstruct exact recoil for every observation
direction.

A sixth **scalar** table axis is not a lossless solution to this problem because
\(C_N\) is observer-dependent. Depositing it would make the reusable table dependent on
the requested detector direction.

The natural extension is instead to retain the five physical axes and store sufficient
observer-independent weighted moments of the additional carrier-phase four-gradient.
Stage 2 can contract those moments with \(\mathbf e\) and \(\mathbf N\) for each query.
At second order, this may require a covariance tensor of the relevant phase-gradient
components together with their covariance with the nonlinear scalar. The minimal channel
set has not yet been derived.

## 11. Recoil kinematics is not the full Klein-Nishina model

The resonance correction above changes the emitted photon frequency. It does not modify
the scattering probability by itself.

The current Stage-0 trajectory rate is proportional to the Thomson cross section
\(\sigma_T\). A full linear Compton treatment would replace the relevant rate and
differential angular/polarization kernel by their Klein-Nishina forms. Those depend on
the incident photon energy in the electron rest frame, which in the present notation is
controlled by

$$
\epsilon_* \sim \epsilon_0 C\,\kappa
\simeq
\epsilon_0 C\,\gamma F.
$$

For a fixed nominal \(C\), this invariant is determined by existing table coordinates and
geometry and therefore does not by itself imply a sixth axis.

If \(C(t)\) varies along the trajectory, however, the local Klein-Nishina cross section
changes the scattering weight. Then the DER016 trajectory reduction itself must be
re-derived with the quantum rate inside the weighting integrals. It is not generally
correct to multiply the final Thomson spectrum by one mean Klein-Nishina factor.

Similarly, the differential Klein-Nishina kernel modifies angular and polarization
weights. A complete quantum-Compton implementation therefore contains more physics than
the recoil-corrected line centre derived here.

For \(a_0=O(1)\) and beyond, DER020 supplies a separate warning: distinct nonlinear
harmonics become important. In that regime a consistent nonlinear-Compton treatment may
be preferable to combining a first-harmonic classical kernel with only a recoil shift.

## 12. Limiting cases and consistency checks

### Classical limit

For

$$
\epsilon_0\to0,
$$

the recoil coefficient \(L\to0\), and the resonance, inverse, Jacobian, and DER017 moment
formulas reduce exactly to their present classical forms.

### Head-on, on-axis geometry

For

$$
\mathbf e=\mathbf N=-\mathbf n_0,
$$

one has

$$
F=2,\qquad
G=2,\qquad
D=1,\qquad
Q=1,\qquad
r=0.
$$

The physical resonance becomes

$$
\boxed{
\omega'_R
=
\frac{4\omega_0\bar C\,\gamma^2}
{1+\hat a+4\gamma\epsilon_0\bar C}
}.
$$

This is the standard first-harmonic inverse-Compton recoil denominator together with the
nonlinear mass/ponderomotive shift.

### Linear-field limit

For

$$
\hat a\to0,
$$

$$
s_R
=
\frac{D\bar C\,\gamma^2}
{1+\gamma^2r^2+2\gamma\epsilon_0\bar C\,G},
$$

which is the ordinary linear inverse-Compton recoil shift in the present geometry.

### Unchirped limit

For

$$
\bar C=1,
\qquad
\operatorname{Var}(C)=0,
\qquad
\operatorname{Cov}(q_{\rm nl},C)=0,
$$

recoil remains through \(\epsilon_0\), while the finite-line correction contains only
the nonlinear-strength variance.

### Observation along the electron direction

For

$$
\mathbf N=\mathbf e,
$$

at nonsingular incidence,

$$
Q=1,
\qquad
G=F.
$$

The recoil term becomes

$$
2\gamma\epsilon_0\bar C F=x.
$$

### Co-propagating limit

As

$$
F\to0,
$$

the present xigma geometry is singular and outside the production validity domain,
independently of recoil. The recoil extension does not regularize this regime.

### Large recoil

The analytical root remains well defined for finite positive recoil, but retaining the
classical Thomson rate and angular/polarization kernel becomes increasingly inconsistent.
A recoil-only implementation must therefore be described as a kinematic correction, not
as a complete quantum-Compton model.

## 13. Relation to existing derivations

- **DER013** supplies \(D\), the spectral normalization, and the classical analytical
  inverse/Jacobian structure.
- **DER015** supplies the observer-dependent nonlinear projection
  \(Q=G/F\).
- **DER016** supplies \(\hat a\), \(\bar C\), and the variance/covariance channels used by
  the finite-pulse reduction.
- **DER017** supplies the second-order spectral-moment reconstruction. The recoil-aware
  expansion above reduces exactly to DER017 when \(\epsilon_0\to0\).
- **DER018** supplies the effective-plane-wave radiation phase and quasi-momentum from
  which the classical part of the recoil geometry is inherited. This derivation extends
  DER018 by replacing the classical first-harmonic phase condition by finite-quantum
  four-momentum conservation.
- **DER020** concerns higher-harmonic loss of the first-harmonic approximation. Quantum
  recoil shifts harmonic frequencies, whereas a full nonlinear-Compton treatment also
  changes harmonic weights; these are distinct validity questions.
- **DER021** derives raw-\(\hat a\) grid sensitivity for the classical resonance. If a
  recoil-enabled mode is implemented, the corresponding forward sensitivity uses the
  recoil-aware denominator \(T\); because the recoil term is independent of \(\hat a\),
  the monotonic conclusion with respect to \(\hat a\) is not reversed.

No existing derivation is superseded.

## Result

Within the scalar-carrier effective-plane-wave approximation, the first-harmonic
recoil-corrected xigma resonance is

$$
\boxed{
s_R
=
\frac{D\bar C\,\gamma^2}
{1+Q\hat a+\gamma^2r^2
 +2\gamma\epsilon_0\bar C
  (1-\mathbf N\cdot\mathbf n_0)}
}.
$$

It is analytically invertible. With

$$
A=1+Q\hat a,
\qquad
K=D\bar C,
\qquad
L=2\epsilon_0\bar C(1-\mathbf N\cdot\mathbf n_0),
\qquad
U=K/s-r^2,
$$

the physical root and Jacobian are

$$
\boxed{
\Gamma
=
\frac{L+\sqrt{L^2+4AU}}{2U}
},
\qquad
U>0,
$$

and

$$
\boxed{
\left|\frac{d\Gamma}{ds}\right|
=
\frac{K\Gamma^3}
{s^2(2A+L\Gamma)}
}.
$$

The existing five-dimensional table remains sufficient because the new recoil factor is
constructed from the existing \(\gamma\) and \(\bar C\) coordinates, Stage-2 geometry,
and scalar reference-frequency metadata.

Through second order in trajectory fluctuations, the existing
\(\operatorname{Var}(q_{\rm nl})\), \(\operatorname{Var}(C)\), and
\(\operatorname{Cov}(q_{\rm nl},C)\) channels also remain sufficient.

For a general spatially varying carrier four-wavevector, the current scalar \(\bar C\)
is not sufficient for exact recoil at arbitrary observation direction. The appropriate
extension is observer-independent phase-gradient moment channels, not an
observer-dependent sixth table axis.

## Implementation implications

The current main implementation is still classical at this point:

- Table and ShapeTable remain five-dimensional and already carry the three DER016
  moment channels.
- _inverse_resonance_gamma_sq implements
  \(\Gamma^2=A/(K/s-r^2)\).
- _inverse_resonance_jacobian implements
  \(K\Gamma^3/(2As^2)\).
- Stage 0 uses \(\sigma_T\) in the trajectory scattering rate.
- The reference laser frequency is already available in the laser model and is used in
  existing photon-density normalization, but not in a recoil denominator.

A future **recoil-only** implementation should therefore:

1. keep the current five table axes;
2. carry \(\epsilon_0\), or equivalently \(\omega_0\), into Stage 2 as scalar metadata;
3. replace the classical inverse root and Jacobian by the quadratic expressions above;
4. update CPU and CUDA/CuPy query paths consistently;
5. use the recoil-aware DER017 coefficients in the \(\rho_1\) and \(\rho_2\) channels;
6. preserve the exact classical result when \(\epsilon_0=0\);
7. document the scalar-carrier assumption
   \(k_L^\mu=Ck_0^\mu\).

The radial support relation becomes

$$
r^2
=
\frac{K}{s}
-\frac{A}{\gamma^2}
-\frac{L}{\gamma}.
$$

For \(A>0\) and \(L\ge0\),

$$
\frac{d r^2}{d\gamma}
=
\frac{2A}{\gamma^3}
+
\frac{L}{\gamma^2}
>0,
$$

so the current conservative annulus-bound strategy can in principle be extended without
introducing a new sampled dimension.

A **full Klein-Nishina** implementation is a separate piece of work. It must revisit the
trajectory scattering weight and the differential angular/polarization kernel and should
not be presented as already specified by the recoil-only resonance.

No implementation code is changed by this derivation.

## Validation ideas

1. **Four-vector oracle.** For arbitrary electron and observation directions, construct
   \(q^\mu\), \(k^\mu\), and \(k'^\mu\) explicitly and solve
   \(q+\hbar k=q'+\hbar k'\). Compare with the reduced resonance.

2. **Classical regression.** Set \(\epsilon_0=0\) and require the resonance, support,
   inverse root, Jacobian, \(\rho_0\), \(\rho_1\), and \(\rho_2\) to reduce to the
   existing DER013/DER015/DER017 implementation.

3. **Head-on Compton edge.** Verify

   $$
   \omega'_R
   =
   \frac{4\gamma^2\omega_0\bar C}
   {1+\hat a+4\gamma\epsilon_0\bar C}
   $$

   for head-on, on-axis geometry.

4. **Inverse/Jacobian identity.** Substitute the analytical root into
   \(U\Gamma^2-L\Gamma-A=0\) and differentiate independently.

5. **Monotonicity and support.** Scan the production domain and confirm a unique positive
   root exactly when \(U>0\), including points close to the support boundary.

6. **Second-order moment oracle.** Generate a narrow synthetic joint distribution of
   \(q_{\rm nl}\) and \(C\), evaluate the recoil-corrected resonance directly, and compare
   its centroid and variance with \(\Delta_c\) and \(m_2\) as the fluctuation scale is
   reduced.

7. **CPU/CUDA parity.** If implemented, compare recoil-enabled \(\rho_0,\rho_1,\rho_2\)
   over representative head-on, crossed, chirped, and off-axis queries.

8. **General phase-gradient counterexample.** Construct two local carrier phase gradients
   with the same electron contraction \(C_e\) but different observer contractions \(C_N\).
   They must produce the same current scalar carrier input but different recoil shifts.
   This directly tests the information-loss argument for arbitrary spatial chirp.

9. **Full Klein-Nishina extension.** If later implemented, separately test the line
   kinematics and the rate/angular/polarization normalization so one error cannot be hidden
   by another. The quantum rate must reduce continuously to Thomson at low rest-frame
   photon energy.

## Open questions

- What recoil parameter should delimit a public recoil-only mode before the unchanged
  Thomson angular/rate model becomes quantitatively inadequate?
- Should recoil be implemented first as an optional kinematic correction, or only
  together with a full Klein-Nishina differential kernel?
- Is the intended production chirp model sufficiently close to
  \(k_L^\mu=Ck_0^\mu\), or does exact spatial-wavevector recoil need to be supported
  from the outset?
- If general carrier four-gradients are supported, what is the minimal
  observer-independent set of weighted mean/covariance channels required to reconstruct
  both electron and observer contractions through second order?
- If the local scattering probability is changed to Klein-Nishina, what are the correct
  quantum-rate-weighted replacements for DER016's \(\hat a\), \(\bar C\), and correlated
  moments?
- At what combination of \(a_0\), recoil parameter, and strong-field quantum parameter
  should the first-harmonic recoil model be replaced by a nonlinear-Compton harmonic
  treatment?
- Should \(\epsilon_0\) be serialized directly with Table/PreparedQuery, or always
  supplied through the parent collision/laser object?
