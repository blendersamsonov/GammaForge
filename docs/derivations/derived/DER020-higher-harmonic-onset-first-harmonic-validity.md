# DER020 — Higher-harmonic onset and first-harmonic validity diagnostic

Status: derived

## Setup

GammaForge currently models inverse-Compton/Thomson emission with an unresolved
first-harmonic line. DER018 gives the effective-plane-wave construction and the exact
radiation-phase structure that underlies the nonlinear redshift, while DER017 describes
second-order finite-line corrections *within that retained first harmonic*. Neither result
quantifies radiation transferred into distinct nonlinear harmonics.

This derivation addresses three separate questions:

1. where the higher harmonics arise in the plane-wave radiation problem;
2. why they are perturbatively suppressed for $a_0\ll1$ but generically cease to be
   suppressed when $a_0=O(1)$;
3. how to define a compact diagnostic for the radiated energy omitted by a
   first-harmonic model without adding a harmonic axis to the xigma phase-space table.

The treatment is classical unless stated otherwise. It assumes an ultrarelativistic
electron, negligible radiation reaction and laser depletion, and the same
effective-plane-wave surrogate used by DER018. The electron may remain nearly ballistic
through the focal volume even when the emitted radiation is nonlinear.

Two parameters must be kept distinct:

$$
\frac{a_0}{\gamma}
$$

controls the size of the laser-driven velocity/trajectory perturbation relative to the
ultrarelativistic drift, whereas

$$
a_0
$$

controls the nonlinear periodic motion and therefore the harmonic content of the
radiation. In particular, $a_0\sim1$ and $a_0/\gamma\ll1$ can hold simultaneously.

For the explicit harmonic scaling argument below, take a monochromatic linearly
polarized plane wave with phase $\phi$ and vector potential

$$
a(\phi)=a_0\cos\phi.
$$

The special exact-backscatter geometry is used only to expose the simplest Bessel
structure and selection rule. Generic observation directions are discussed separately.

## 1. Harmonics in the exact plane-wave radiation phase

DER018 gives, for radiation of frequency $\omega'$ observed along $\mathbf N$,

$$
\frac{d\Psi}{d\phi}
=
\frac{\omega'}{\omega_L}
\left[
C_0+\mathbf C_1\cdot\mathbf a(\phi)+C_2a^2(\phi)
\right],
$$

with

$$
C_2=\frac{1-\mathbf N\cdot\mathbf n_0}{2\kappa^2},
$$

and $\kappa=\gamma-\mathbf n_0\cdot\mathbf u$ the conserved plane-wave light-front
momentum. Hence

$$
\Psi(\phi)
=
\frac{\omega'}{\omega_L}
\left[
C_0\phi
+
\mathbf C_1\cdot\int^\phi\mathbf a(\varphi)d\varphi
+
C_2\int^\phi a^2(\varphi)d\varphi
\right]
+\mathrm{const}.
$$

The secular cycle average of $a^2$ shifts the line centre. This is the part retained by
the first-harmonic nonlinear redshift of DER015/DER018. The carrier-periodic pieces of
$a$ and $a^2$ instead modulate the radiation phase within every optical cycle. Together
with the periodic acceleration in the radiation prefactor, those terms generate the
nonlinear harmonic ladder.

Thus the first-harmonic approximation is not the same as the ballistic-trajectory
approximation. It retains the secular nonlinear phase but discards the carrier-periodic
phase modulation responsible for distinct harmonics.

## 2. Explicit linearly polarized backscatter example

For

$$
a(\phi)=a_0\cos\phi,
$$

$$
a^2(\phi)
=
\frac{a_0^2}{2}\left(1+\cos2\phi\right).
$$

For a head-on electron observed exactly backward, with vanishing initial transverse
momentum, the linear-in-$a$ phase term vanishes by symmetry. After factoring out the
common Doppler normalization, the phase can be written in the illustrative normalized
form

$$
\Phi(\phi)
=
\Omega
\left[
\left(1+\frac{a_0^2}{2}\right)\phi
+
\frac{a_0^2}{4}\sin2\phi
\right].
$$

The exact numerical definition of $\Omega$ depends on the frequency normalization; the
important point is that the secular redshift scales as $a_0^2$ and the carrier-periodic
phase-modulation index does also.

The transverse acceleration supplies the fundamental carrier factor
$\propto\cos\phi$, so schematically

$$
\mathcal A(\Omega)
\propto
\int d\phi\,
\cos\phi\,
e^{i\Omega(1+a_0^2/2)\phi}
e^{iz\sin2\phi},
$$

with, in this normalization,

$$
z=\frac{\Omega a_0^2}{4}.
$$

Use the Jacobi-Anger expansion

$$
e^{iz\sin2\phi}
=
\sum_{m=-\infty}^{\infty}
J_m(z)e^{i2m\phi}.
$$

Then

$$
\cos\phi\,e^{iz\sin2\phi}
=
\frac12
\sum_m J_m(z)
\left[
e^{i(2m+1)\phi}
+
e^{i(2m-1)\phi}
\right].
$$

The radiation therefore contains the odd harmonic sequence

$$
n=1,3,5,\ldots
$$

in this special symmetric geometry. The absence of even harmonics here is a selection
rule, not a general statement about nonlinear Thomson scattering.

The $n=2m+1$ resonance occurs schematically at

$$
\Omega_n
\simeq
\frac{n}{1+a_0^2/2},
$$

so the corresponding Bessel argument scales as

$$
z_n
\simeq
\frac{n a_0^2}{4(1+a_0^2/2)}.
$$

Only the $a_0$ scaling and the relation between Bessel order and harmonic number are
needed below.

## 3. Weak-field suppression

For fixed low harmonic order and $a_0\ll1$,

$$
z_n\simeq\frac{n a_0^2}{4}\ll1.
$$

The small-argument expansion is

$$
J_m(z)
\simeq
\frac1{m!}
\left(\frac z2\right)^m.
$$

For $n=2m+1$ this gives

$$
\frac{\mathcal A_n}{\mathcal A_1}
\propto
a_0^{n-1},
$$

and therefore, up to geometry-dependent coefficients,

$$
\boxed{
\frac{I_n}{I_1}
\propto
a_0^{2(n-1)}
\qquad
(n=1,3,5,\ldots)
}
$$

in exact backward scattering with linear polarization. In particular,

$$
\frac{I_3}{I_1}=O(a_0^4),
\qquad
\frac{I_5}{I_1}=O(a_0^8).
$$

This is the perturbative reason the fundamental dominates as $a_0\to0$.

In a generic observation geometry the term linear in $a$ in the radiation phase need
not vanish. A first-order phase modulation then allows even harmonics. The second
harmonic field is generically one order higher in $a_0$ than the fundamental, so its
intensity can scale as

$$
\frac{I_2}{I_1}=O(a_0^2).
$$

The detailed powers and zeros are therefore polarization- and geometry-dependent, but
all higher harmonics vanish perturbatively with $a_0$.

## 4. Loss of perturbative suppression near $a_0\sim1$

The same Bessel representation shows why $a_0\sim1$ is the relevant transition scale.
For $a_0\gg1$,

$$
\frac{a_0^2}{1+a_0^2/2}\to2,
$$

so in the symmetric example

$$
z_n\to\frac n2.
$$

The relevant Bessel order is

$$
m=\frac{n-1}{2}\sim\frac n2.
$$

Thus the strong-modulation regime has

$$
z_n\sim m
$$

rather than $z_n\ll m$. Bessel functions with argument of the same order as their
index lie in their transition region and are not suppressed by a high power of $a_0$.
Several harmonic orders can therefore carry appreciable radiation.

There is no sharp threshold at exactly $a_0=1$. Harmonic fractions change continuously
and depend on observation direction and polarization. The statement is only that the
small-$a_0$ perturbative hierarchy ceases to provide a uniformly small parameter when
$a_0=O(1)$.

For $a_0\gg1$, the harmonic comb approaches the synchrotron-like nonlinear-Thomson
regime. The familiar characteristic-harmonic scaling $n_c\sim a_0^3$ is useful as a
parametric guide, but is not derived or used quantitatively here.

## 5. Time-domain interpretation

For $a_0\ll1$, the transverse velocity and observed acceleration are close to
sinusoidal, so the emitted waveform is dominated by one Fourier component.

For $a_0\gtrsim1$, the relativistic velocity direction varies strongly over an optical
cycle. Relativistic beaming then converts the smooth oscillation into a periodic train
of increasingly short radiation bursts for a fixed observer. A periodic train of short
pulses necessarily contains many Fourier harmonics.

This is the same physics as the Bessel expansion expressed in the time domain.

## 6. When may the velocity still be treated as constant?

"Constant velocity" enters the GammaForge approximations in three different places and
must not be treated as one assumption.

### 6.1 Macroscopic trajectory

The transverse quiver momentum is $O(a_0)$, so for an ultrarelativistic electron

$$
\delta\beta_\perp=O\left(\frac{a_0}{\gamma}\right).
$$

The displacement from the ballistic trajectory can therefore remain negligible on the
scale of the focal volume when

$$
\frac{a_0}{\gamma}\ll1
$$

even if $a_0=O(1)$. Harmonic generation does not by itself require abandoning ballistic
transport through the laser focus.

### 6.2 Slowly varying radiation prefactor

For the same reason, using the field-free velocity in slowly varying angular and
polarization prefactors can remain accurate to the intended order when $a_0/\gamma$ is
small.

### 6.3 Retarded phase

The velocity cannot be frozen in the accumulated radiation phase if nonlinear redshift,
ponderomotive chirp, or harmonics are to be retained. The $a^2$ contribution is small
instantaneously but accumulates over the formation length. Its significance is governed
by $a_0$, not by $a_0/\gamma$.

It is therefore internally consistent to use

$$
\text{ballistic macroscopic trajectory}
+
\text{field-free slow prefactor}
+
\text{nonlinear retarded phase}.
$$

The current first-harmonic model keeps the secular/cycle-averaged nonlinear phase but
drops its carrier-periodic part.

## 7. Energy in harmonics above the fundamental

If the goal is only to quantify how much energy the first-harmonic model omits, it is
not necessary to resolve every higher-harmonic line.

For one effective-plane-wave trajectory define

$$
W_{\mathrm{tot}}
=
\text{total classical radiated energy},
$$

$$
W_1
=
\text{energy in the true fundamental harmonic},
$$

and, when the harmonic decomposition is well defined,

$$
\boxed{
W_{>1}=W_{\mathrm{tot}}-W_1.
}
$$

Two useful dimensionless diagnostics are

$$
\boxed{
\eta_H
=
\frac{W_{>1}}{W_{\mathrm{tot}}}
=
1-\frac{W_1}{W_{\mathrm{tot}}},
}
$$

and

$$
\boxed{
\epsilon_H
=
\frac{W_{>1}}{W_1}.
}
$$

The total energy can be obtained independently of any spectral decomposition from the
Liénard power. With physical acceleration
$\mathbf a_{\rm phys}=d\mathbf v/dt$,

$$
P(t)
=
\frac{2e^2}{3c^3}
\gamma^6
\left[
a_{\rm phys}^2
-
(\boldsymbol\beta\times\mathbf a_{\rm phys})^2
\right].
$$

Equivalently, in terms of $\dot{\boldsymbol\beta}$,

$$
P(t)
=
\frac{2e^2}{3c}
\gamma^6
\left[
\dot{\boldsymbol\beta}^{,2}
-
(\boldsymbol\beta\times\dot{\boldsymbol\beta})^2
\right].
$$

The earlier working note used the $c^{-3}$ prefactor together with
$\dot{\boldsymbol\beta}$; the two forms above make the dimensionally consistent
convention explicit.

Then

$$
W_{\mathrm{tot}}=\int P(t)\,dt.
$$

The fundamental must be obtained from the *same exact trajectory*, either by a true
$n=1$ harmonic projection or, in a many-cycle pulse with separated bands, by integrating
the first harmonic band:

$$
W_1
=
\int d\Omega\int_{n=1}d\omega\,
\frac{d^3I}{d\omega\,d\Omega}.
$$

If the exact first-harmonic line can be written

$$
\frac{d^3I_1}{d\omega\,d\Omega}
=
C_1(\mathbf N)R_1(\omega-\omega_{R,1}),
$$

with

$$
\int R_1(\Delta\omega)d\Delta\omega=1,
$$

then

$$
\frac{dI_1}{d\Omega}=C_1(\mathbf N),
\qquad
W_1=\int C_1(\mathbf N)d\Omega.
$$

The subtraction $W_{\rm tot}-W_1$ is a pure higher-harmonic energy only if $W_1$ is
the true first-harmonic component of the same trajectory. If $W_1$ uses an additional
approximation that changes the fundamental normalization, the residual mixes true
higher-harmonic energy with first-harmonic model error.

## 8. Finite pulses and the meaning of a harmonic

For an infinite monochromatic wave,

$$
W_{\rm tot}=\sum_{n\ge1}W_n
$$

is unambiguous.

For a smooth many-cycle pulse, each harmonic has a finite width but neighboring bands
remain sufficiently separated that the decomposition is still operationally clear.
Then

$$
W_{>1}
\simeq
\sum_{n>1}W_n
$$

up to the chosen band/projection convention.

For few-cycle pulses, neighboring bands can overlap strongly. "Fundamental energy" then
requires an explicit definition, for example:

- projection onto the $n=1$ Floquet/carrier harmonic of the underlying plane-wave
  motion;
- integration over a prescribed first-harmonic spectral window;
- an amplitude-level harmonic decomposition before applying the finite envelope.

GammaForge is primarily interested in the multi-cycle regime, but any diagnostic must
state which definition it uses.

## 9. Relation to the first-harmonic moment expansion

DER017 reconstructs the narrow first-harmonic spectrum as a moment expansion of the
retained line. Schematically,

$$
R_1(\omega-\omega_R)
\sim
\delta(\omega-\omega_R)
+
\frac{\mu_2}{2}\delta''(\omega-\omega_R)
+\cdots.
$$

No finite number of derivatives of a delta function about $\omega_R$ can reconstruct
distinct radiation bands centred at the nonlinear harmonic resonances. Therefore the
two approximations address different errors:

- DER017 corrects the finite width/centroid structure of the retained first harmonic;
- $\eta_H$ quantifies energy omitted because separate $n>1$ harmonics are absent.

A second-order-accurate first-harmonic line model can still be physically incomplete if
$\eta_H$ is not small.

## 10. Limiting cases and consistency checks

### 10.1 $a_0\to0$

All higher harmonics vanish perturbatively and

$$
\eta_H\to0.
$$

For exact backward linear polarization the first omitted harmonic is $n=3$ and

$$
W_3/W_1=O(a_0^4).
$$

For generic geometry where $n=2$ is allowed, the leading omitted fraction can begin at

$$
W_2/W_1=O(a_0^2).
$$

### 10.2 $a_0=O(1)$

There is no universal sharp threshold, but the phase-modulation index is no longer small.
The actual harmonic fraction must be evaluated for the relevant polarization, geometry,
and aperture.

### 10.3 $a_0\sim1$ but $a_0/\gamma\ll1$

Higher harmonics may be appreciable while the macroscopic electron trajectory remains
ballistic. This is the important inverse-Compton regime in which the radiation model can
fail before the overlap/transport approximation does.

### 10.4 Flat-top monochromatic wave

The spectrum is a discrete harmonic comb. The secular $a_0^2$ term redshifts the
harmonic positions but does not itself broaden the lines.

### 10.5 Smooth finite envelope

Each harmonic acquires transform-limited width and envelope-induced nonlinear chirp.
For many cycles the bands converge toward the monochromatic harmonic fractions.

### 10.6 Linear polarization and exact backscatter

Only odd harmonics occur in the idealized symmetric plane-wave problem.

### 10.7 Off-axis observation

The term linear in $a$ in the radiation phase generally survives, and even harmonics can
appear. The exact-backscatter weak-field power law must not be applied blindly.

### 10.8 Circular polarization

Different symmetry rules apply, especially on axis. The statement that higher harmonics
are generically unsuppressed for $a_0=O(1)$ does not mean every harmonic is nonzero in
every geometry.

### 10.9 Quantum recoil

This derivation concerns classical harmonic generation. Recoil shifts harmonic
frequencies, while a full nonlinear-Compton treatment also changes harmonic weights and
polarization. A classical $\eta_H$ is not by itself a validity criterion when the
strong-field quantum parameter is appreciable.

## 11. Relation to existing GammaForge derivations

- **DER018** supplies the effective-plane-wave surrogate and the exact phase derivative
  $C_0+\mathbf C_1\cdot\mathbf a+C_2a^2$ used here. DER020 extends that phase argument
  to the carrier-periodic terms that DER018 intentionally leaves outside the unresolved
  first-harmonic model.
- **DER015** is the verified secular observer-dependent ponderomotive incidence factor.
  DER020 does not alter it.
- **DER016** and **DER017** describe trajectory-averaged carrier/nonlinear moments and
  finite-line reconstruction of the fundamental. DER020 is complementary: it concerns
  power in separate harmonic bands.
- The current five-dimensional xigma table remains a first-harmonic representation.
  DER020 does not imply a sixth phase-space coordinate.

## 12. Implementation implications

The immediate implementation consequence is a **validity diagnostic**, not a
multi-harmonic production kernel.

### 12.1 Diagnostic quantity

A reference or diagnostic path should be able to calculate at least

$$
\eta_H=1-W_1/W_{\rm tot}
$$

for representative effective-plane-wave trajectories, together with enough metadata to
state how $W_1$ was defined.

A bunch-level scalar may be formed by summing energies before taking the ratio,

$$
\eta_H^{\rm bunch}
=
1-
\frac{\sum_i W_{1,i}}
     {\sum_i W_{{\rm tot},i}},
$$

with the same macroparticle/interaction weights used by the reference calculation.

### 12.2 No new table dimension for the diagnostic

A scalar validity metric does not require adding a harmonic axis or another coordinate
to the reusable xigma table. The first implementation should preferably live in the
independent validation/reference layer until its cost and required trajectory information
are understood.

Current production Stage 0 is designed around cycle-averaged overlap/intensity
information. An exact carrier-harmonic decomposition may require information not
currently retained there. This should be established rather than inferred by a coding
agent.

### 12.3 Strength retargeting

GammaForge retargets reusable trajectory information to different peak nonlinear
strengths. Since $\eta_H(a_0)$ is nonlinear in $a_0$, storing one fixed harmonic-loss
number per trajectory is not generally compatible with arbitrary retargeting.

Possible future compact representations include:

- strength-independent trajectory moments from which $\eta_H(a_0)$ can be reconstructed;
- a small one-dimensional response in $a_0$ external to the main phase-space table;
- evaluation only at validation time for the requested physical strength.

The minimal representation has not been derived here.

### 12.4 Angular dependence

A global energy fraction can hide a large harmonic fraction inside a restricted detector
cone. For an angle-resolved validity check define, when available,

$$
\eta_H(\mathbf N)
=
\frac{dW_{\rm tot}/d\Omega-dW_1/d\Omega}
     {dW_{\rm tot}/d\Omega},
$$

or integrate the same numerator and denominator over the requested aperture.

Whether GammaForge needs a global, aperture-integrated, or local diagnostic is an
observable-level choice.

### 12.5 Possible zeroth-order rescaling

A tempting zeroth-order correction is

$$
f_1=1-\eta_H=\frac{W_1}{W_{\rm tot}}
$$

and

$$
S_1^{\rm corrected}
\approx
f_1 S_1^{\rm current}.
$$

This is **not** established as a production correction by the derivation. It would only
be appropriate if the current retained first-harmonic spectrum is normalized as though
it carries the total radiated energy. If the production kernel already predicts the
true $W_1$, multiplying by $f_1$ would double-count the harmonic loss.

Moreover, current xigma spectra are photon-number distributions. An energy fraction is
not generally the corresponding photon-number fraction because the omitted harmonics
have larger photon energy. Therefore an energy-based factor must not silently rescale
photon-number output. Any such option is heuristic until a photon-number correction is
derived and validated.

## 13. Validation targets

1. **Monochromatic analytical benchmark.** Compare a numerical plane-wave radiation
   calculation against standard Bessel harmonic weights for a monochromatic linearly
   polarized wave.

2. **Weak-field scaling.** Verify
   $W_3/W_1\propto a_0^4$ in exact backward linear polarization and the expected
   leading $a_0$ power when the second harmonic is symmetry-allowed.

3. **Selection rules.** Recover odd-only emission in the symmetric linear-polarization
   case and the appearance of even harmonics after moving away from that symmetry.

4. **Total-energy closure.** Compare frequency- and angle-integrated radiation against
   the independent Liénard-power integral and verify
   $W_{\rm tot}-W_1\simeq\sum_{n>1}W_n$ for a many-cycle benchmark.

5. **Finite-pulse convergence.** Increase the pulse duration and verify convergence of
   the integrated harmonic fractions toward the monochromatic limit.

6. **Fundamental projection.** Compare the true $n=1$ energy from the reference
   calculation with the integrated GammaForge first-harmonic kernel in the weak-field
   regime.

7. **Ballistic/harmonic separation.** At fixed $a_0$, vary $\gamma$ and verify that the
   harmonic fraction is controlled primarily by $a_0$ while ballistic-trajectory errors
   shrink with $a_0/\gamma$.

8. **Aperture dependence.** Compare a global $\eta_H$ against the omitted fraction in
   representative detector apertures.

9. **Normalization test before rescaling.** Determine whether the current production
   first-harmonic output already represents $W_1$ or instead over-assigns total energy to
   the retained line. Do not enable an $f_1$ rescaling without this check.

## 14. Open questions

- What harmonic-energy fraction should define the practical first-harmonic validity
  boundary: a universal tolerance or an observable-specific one?
- Is a global $\eta_H$ sufficient, or is an aperture-resolved metric required for
  GammaForge's angle-resolved outputs?
- Can $W_{\rm tot}$ and $W_1$ be reduced to a small set of strength-independent
  trajectory moments while preserving peak-strength retargeting?
- What is the minimal carrier-resolved trajectory information needed for an exact
  diagnostic in a general bounded pulse?
- For finite chirped pulses, which operational definition of $W_1$ remains stable when
  harmonic bands begin to overlap?
- How accurately does the existing first-harmonic angular/polarization prefactor reproduce
  the true $n=1$ component as $a_0$ approaches unity?
- At what recoil/quantum parameter should a classical harmonic diagnostic be replaced by
  nonlinear-Compton harmonic weights?
- If explicit harmonics are implemented later, can they be evaluated as a small outer sum
  at query time without adding another phase-space table dimension?
- How should first-harmonic finite-line moments and an explicit multi-harmonic model be
  combined without double counting envelope/ponderomotive broadening?

## Result

The carrier-periodic terms in the exact plane-wave radiation phase generate nonlinear
harmonics. For $a_0\ll1$ they are perturbatively suppressed; in the symmetric
linear-polarization backscatter example,

$$
I_n/I_1\propto a_0^{2(n-1)}
$$

for odd $n$, while generic geometries can admit an $O(a_0^2)$ relative second-harmonic
intensity. When $a_0=O(1)$, the phase-modulation index is no longer small and the
perturbative suppression is not guaranteed.

This failure of the first-harmonic approximation is independent of the ballistic
condition $a_0/\gamma\ll1$.

For a trajectory with a well-defined harmonic decomposition, the total omitted
higher-harmonic energy is

$$
\boxed{
W_{>1}=W_{\rm tot}-W_1,
}
$$

and the natural validity metric is

$$
\boxed{
\eta_H=1-\frac{W_1}{W_{\rm tot}}.
}
$$

This metric can be introduced as a diagnostic without adding a phase-space table
dimension. A global multiplication of the current spectrum by $1-\eta_H$ is only a
candidate zeroth-order normalization correction and is not justified until the current
first-harmonic normalization and the energy-versus-photon-number distinction are checked.
