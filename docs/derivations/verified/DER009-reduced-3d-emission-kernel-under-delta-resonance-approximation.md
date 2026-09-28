# DER009 — Reduced 3D emission kernel under the delta-resonance approximation

Status: verified

## Setup

In the Xigma engine (Stage 2), backscattered Compton photon emission from a phase-space
distribution of relativistic electrons colliding with an intense laser pulse is evaluated
under the delta-resonance approximation.

Rather than executing full multi-particle Liénard-Wiechert numerical integrations for every
electron at every observation coordinate, Stage 2 samples Stage 1's 4D macroparticle
distribution table $H(\gamma, \theta_{e,x}, \theta_{e,y}, \hat a)$. For an observation
direction $\mathbf{n} \approx (\theta_x, \theta_y, 1 - \theta^2/2)$ and an electron of Lorentz factor
$\gamma \gg 1$ moving along direction $\mathbf{v}/c \approx (\theta_{e,x}, \theta_{e,y}, 1 - \theta_e^2/2)$,
the radiation is emitted into a narrow cone around the electron velocity. In the
near-backscattering geometry (counter-propagating laser carrier frequency $\omega_L$), the
emitted photon frequency $\omega$ in the laboratory frame satisfies the relativistic Doppler
resonance condition:

$$
\omega_R(\gamma, \theta, \hat a) = \frac{4\omega_L\gamma^2}{1 + \gamma^2\theta^2 + \hat a}
$$

where $\theta^2 \equiv (\theta_x - \theta_{e,x})^2 + (\theta_y - \theta_{e,y})^2$ is the
transverse angular separation between the electron velocity and the line of sight, and
$\hat a$ is the trajectory-averaged normalized laser intensity parameter (DER003, RES054).

Adopting the normalized spectral energy coordinate:

$$
s \equiv \frac{\omega}{4\omega_L}
$$

the Doppler resonance equation becomes:

$$
s = \frac{\gamma^2}{1 + \gamma^2\theta^2 + \hat a}
$$

In this derivation:
1. We invert the resonance relation $s(\gamma)$ from first principles to determine the unique
   electron energy $\Gamma(s, \theta, \hat a)$ that radiates into energy $s$ at angle $\theta$.
2. We establish the kinematic support condition $1/s > \theta^2$ and the Heaviside cutoff
   $H(1/s - \theta^2)$.
3. We derive the coordinate transformation Jacobians $|d\Gamma/ds|$ and $|d\Gamma/d\omega|$.
4. We derive the emission kernel weight factor $\Gamma^5 / [(1+\Gamma^2\theta^2)^2 (1+\hat a)]$.
5. We prove that `KERNEL_NORMALIZATION_CONSTANT` equals $1.5 / (2\pi)$, combining the $2\pi$
   correction to the paper's differential cross-section (RES026, RES033) with the Jacobian factor.

## Derivation

### 1. Inversion of the Doppler resonance

Starting from the resonance relation for normalized energy $s > 0$:

$$
s = \frac{\gamma^2}{1 + \gamma^2\theta^2 + \hat a}
$$

Multiplying both sides by the denominator $1 + \gamma^2\theta^2 + \hat a$:

$$
s(1 + \hat a) + s\theta^2\gamma^2 = \gamma^2
$$

Collecting terms proportional to $\gamma^2$:

$$
s(1 + \hat a) = \gamma^2(1 - s\theta^2)
$$

Dividing both sides by $s$:

$$
1 + \hat a = \gamma^2\left(\frac{1}{s} - \theta^2\right)
$$

Solving for $\gamma^2$, which we denote as $\Gamma^2(s, \theta, \hat a)$:

$$
\Gamma^2(s, \theta, \hat a) = \frac{1 + \hat a}{\frac{1}{s} - \theta^2}
$$

Taking the positive square root ($\Gamma > 0$):

$$
\Gamma(s, \theta, \hat a) = \sqrt{\frac{1 + \hat a}{\frac{1}{s} - \theta^2}}
$$

### 2. Kinematic support condition and Heaviside cutoff

Physical electron energies require $\Gamma \in \mathbb{R}^+$. Because $1 + \hat a \ge 1 > 0$,
a physical solution exists if and only if the denominator is strictly positive:

$$
\frac{1}{s} - \theta^2 > 0 \iff \theta^2 < \frac{1}{s} \iff s\theta^2 < 1
$$

This is the fundamental kinematic boundary of Thomson/Compton backscattering:
- For a fixed observation angle $\theta$, no electron of any energy—even in the ultrarelativistic
  limit $\gamma \to \infty$—can radiate photons with energy exceeding $s_{\max} = 1/\theta^2$.
- Conversely, for a given photon energy $s$, radiation is strictly confined within a forward
  angular cone of half-angle $\theta < 1/\sqrt{s}$.

Where $1/s - \theta^2 \le 0$, the resonance condition has no physical solution and the
emission kernel vanishes identically. Thus, the physical solution carries the Heaviside step
function $H(1/s - \theta^2)$:

$$
\Gamma(s, \theta, \hat a) = \sqrt{\frac{1 + \hat a}{\frac{1}{s} - \theta^2}} \; H\left(\frac{1}{s} - \theta^2\right)
$$

In `stages.py` (lines 924–931), this condition is implemented directly:
```python
inv_base = 1.0 / s_val - r_sq
valid = inv_base > 0.0
g_sq = (1.0 + a_c) / np.where(valid, inv_base, 1.0)
g = np.where(valid, np.sqrt(g_sq), 0.0)
```

### 3. Transformation Jacobians $|d\Gamma/ds|$ and $|d\Gamma/d\omega|$

In the delta-resonance approximation, the photon spectrum at frequency $\omega$ is obtained by
integrating over the electron energy distribution $\gamma$:

$$
\frac{d^3 N}{d\omega \, d^2\Omega} \propto \int d\gamma \, \delta(\omega - \omega_R(\gamma)) \, (\dots)
$$

Using the composition rule for Dirac delta functions:

$$
\delta(\omega - \omega_R(\gamma)) = \frac{\delta(\gamma - \Gamma)}{\left|\frac{\partial \omega_R}{\partial \gamma}\right|_{\gamma=\Gamma}} = \left|\frac{d\Gamma}{d\omega}\right| \delta(\gamma - \Gamma)
$$

Equivalently, working in the normalized energy variable $s = \omega / (4\omega_L)$:

$$
\delta(s - s_{\text{res}}(\gamma)) = \left|\frac{d\Gamma}{ds}\right| \delta(\gamma - \Gamma)
$$

We differentiate $\Gamma(s, \theta, \hat a) = (1 + \hat a)^{1/2} (s^{-1} - \theta^2)^{-1/2}$
with respect to $s$:

$$
\frac{d\Gamma}{ds} = (1 + \hat a)^{1/2} \left(-\frac{1}{2}\right) \left(\frac{1}{s} - \theta^2\right)^{-3/2} \left(-\frac{1}{s^2}\right) = \frac{1}{2s^2} (1 + \hat a)^{1/2} \left(\frac{1}{s} - \theta^2\right)^{-3/2}
$$

Noting that:

$$
\Gamma^3 = (1 + \hat a)^{3/2} \left(\frac{1}{s} - \theta^2\right)^{-3/2} \implies \left(\frac{1}{s} - \theta^2\right)^{-3/2} = \frac{\Gamma^3}{(1 + \hat a)^{3/2}}
$$

Substituting this into $d\Gamma/ds$:

$$
\frac{d\Gamma}{ds} = \frac{1}{2s^2} (1 + \hat a)^{1/2} \frac{\Gamma^3}{(1 + \hat a)^{3/2}} = \frac{\Gamma^3}{2s^2(1 + \hat a)}
$$

Because $s > 0$, $\Gamma > 0$, and $1 + \hat a > 0$, the derivative is strictly positive on the
support, giving the Jacobian:

$$
\boxed{\;\left|\frac{d\Gamma}{ds}\right| = \frac{\Gamma^3}{2s^2(1 + \hat a)}\;}
$$

To convert to physical frequency $\omega$, using $s = \omega / (4\omega_L)$ and $ds = d\omega / (4\omega_L)$:

$$
\left|\frac{d\Gamma}{d\omega}\right| = \left|\frac{d\Gamma}{ds}\right| \frac{ds}{d\omega} = \frac{\Gamma^3}{2s^2(1 + \hat a)} \frac{1}{4\omega_L}
$$

Substituting $s^2 = \omega^2 / (16\omega_L^2)$:

$$
\left|\frac{d\Gamma}{d\omega}\right| = \frac{\Gamma^3}{2 \left(\frac{\omega^2}{16\omega_L^2}\right) (1 + \hat a)} \frac{1}{4\omega_L} = \frac{16\omega_L^2 \Gamma^3}{8\omega_L \omega^2 (1 + \hat a)} = \frac{2\omega_L\Gamma^3}{\omega^2(1 + \hat a)}
$$

$$
\boxed{\;\left|\frac{d\Gamma}{d\omega}\right| = \frac{2\omega_L\Gamma^3}{\omega^2(1 + \hat a)}\;}
$$

### 4. Emergence of the kernel factor $\Gamma^5 / [(1+\Gamma^2\theta^2)^2 (1+\hat a)]$

The lab-frame Thomson differential cross-section for a relativistic electron of Lorentz factor
$\gamma$ radiating into solid angle $d\Omega$ carries the Lorentz factor and angular profile:

$$
\frac{d^2\sigma}{ds \, d\Omega} \propto \frac{\gamma^2}{(1 + \gamma^2\theta^2)^2} \, \mathcal{P} \, \delta(s - s_{\text{res}}(\gamma))
$$

where $\mathcal{P}$ is the polarization-dependent factor (DER004, DER006).

When integrating this differential cross-section against an electron distribution $f_e(\gamma)$
over $\gamma$ using the resonance delta function:

$$
\int d\gamma \, \frac{\gamma^2}{(1 + \gamma^2\theta^2)^2} \, \delta(s - s_{\text{res}}(\gamma)) = \left[ \frac{\gamma^2}{(1 + \gamma^2\theta^2)^2} \left|\frac{d\Gamma}{ds}\right| \right]_{\gamma = \Gamma}
$$

Substituting the Jacobian $|d\Gamma/ds| = \frac{\Gamma^3}{2s^2(1 + \hat a)}$:

$$
\frac{\Gamma^2}{(1 + \Gamma^2\theta^2)^2} \left[ \frac{\Gamma^3}{2s^2(1 + \hat a)} \right] = \frac{1}{2s^2} \frac{\Gamma^5}{(1 + \Gamma^2\theta^2)^2 (1 + \hat a)}
$$

Factoring out $1/s^2$ across the energy spectrum quadrature, the core Stage 2 table-kernel
integrand evaluates to:

$$
\boxed{\;\mathcal{K}(s, \theta, \hat a) = \frac{\Gamma^5}{(1 + \Gamma^2\theta^2)^2 (1 + \hat a)}\;}
$$

In `stages.py` (lines 932–945), this matches the array operations:
```python
gth_sq_inv = 1.0 / (1.0 + r_sq * g_sq) ** 2
prefac = np.where(valid, pol_factor * g**5 * gth_sq_inv / (1.0 + a_c), 0.0)
out[k] = (
    KERNEL_NORMALIZATION_CONSTANT
    * float(np.sum(H_val * prefac * ahat_widths))
    * theta_cell_area
    / s_val**2
)
```

### 5. Proof of `KERNEL_NORMALIZATION_CONSTANT = 1.5 / (2*pi)`

The bare differential Thomson cross-section before coordinate transformation has the angular
integral (for unpolarized radiation or azimuthally averaged linear polarization, DER011):

$$
\frac{d^2\sigma}{d\Omega \, ds} = \sigma_T \, C_{\text{diff}} \, \frac{\gamma^2}{(1 + \gamma^2\theta^2)^2} \mathcal{P} \, \delta(s - s_{\text{res}})
$$

As proved in RES026 and RES033, the paper's typeset cross-section had $C_{\text{diff}} = 3$, which
omitted a factor of $1/(2\pi)$. The physically correct differential prefactor that integrates over
solid angle to the total Thomson cross section $\sigma_T$ is:

$$
C_{\text{diff}} = \frac{3}{2\pi}
$$

When evaluating the $\gamma$-integrated spectrum from the table, the delta function brings in
the Jacobian factor:

$$
\left|\frac{d\Gamma}{ds}\right| = \frac{1}{2s^2} \frac{\Gamma^3}{1 + \hat a}
$$

The factor of $1/2$ from the Jacobian combines directly with the numerator $3$ of $C_{\text{diff}}$:

$$
C_{\text{kernel}} = C_{\text{diff}} \times \frac{1}{2} = \frac{3}{2\pi} \times \frac{1}{2} = \frac{1.5}{2\pi}
$$

Because Stage 0's macroparticle luminosity $L_i = \int v_{\text{rel}} \sigma_T n_{\text{ph}} dt$
already folds in $\sigma_T$, the overall kernel normalization constant multiplying the cell
summation is:

$$
\boxed{\;\text{KERNEL\_NORMALIZATION\_CONSTANT} = \frac{1.5}{2\pi} \approx 0.238732414637843\; }
$$

This explains the exact relationship between the two key constants in GammaForge:
- `validation.references.delta.DIFFERENTIAL_PREFACTOR = 3.0 / (2.0 * math.pi)` (bare cross-section, no $\gamma$-integral Jacobian $1/2$).
- `engines.xigma.stages.KERNEL_NORMALIZATION_CONSTANT = 1.5 / (2.0 * math.pi)` (table kernel, incorporates the $1/2$ from $d\Gamma/ds$).

## Result

Under the delta-resonance approximation, an electron in laser field $\hat a$ radiating photon
energy $s = \omega / (4\omega_L)$ into relative angle $\theta$ has resonant Lorentz factor:

$$
\boxed{\;\Gamma(s, \theta, \hat a) = \sqrt{\frac{1 + \hat a}{\frac{1}{s} - \theta^2}} \; H\left(\frac{1}{s} - \theta^2\right)\;}
$$

The coordinate transformation Jacobians are:

$$
\boxed{\;\left|\frac{d\Gamma}{ds}\right| = \frac{\Gamma^3}{2s^2(1 + \hat a)}, \qquad \left|\frac{d\Gamma}{d\omega}\right| = \frac{2\omega_L\Gamma^3}{\omega^2(1 + \hat a)}\;}
$$

The Stage 2 differential photon spectrum is given by:

$$
\boxed{\;\frac{d^2 N}{ds \, d\Omega} = \frac{\text{KERNEL\_NORMALIZATION\_CONSTANT}}{s^2} \int d^2\boldsymbol{\theta}_e \, d\hat a \; H(\Gamma, \boldsymbol{\theta}_e, \hat a) \; \mathcal{P} \; \frac{\Gamma^5}{(1 + \Gamma^2\theta^2)^2 (1 + \hat a)}\;}
$$

with:

$$
\boxed{\;\text{KERNEL\_NORMALIZATION\_CONSTANT} = \frac{1.5}{2\pi}\;}
$$

## Verification

### Symbolic verification (Sympy)

The algebraic steps were verified using Sympy:
1. Solving $s = \Gamma^2 / (1 + \hat a + \Gamma^2\theta^2)$ for $\Gamma$ reproduces
   $\sqrt{(1 + \hat a) / (1/s - \theta^2)}$ uniquely.
2. Differentiating $\Gamma$ with respect to $s$ confirms
   $\frac{d\Gamma}{ds} - \frac{\Gamma^3}{2s^2(1+\hat a)} \equiv 0$.
3. Differentiating $\Gamma$ with respect to $\omega$ confirms
   $\frac{d\Gamma}{d\omega} - \frac{2\omega_L\Gamma^3}{\omega^2(1+\hat a)} \equiv 0$.
4. Multiplying the Lorentz profile $\Gamma^2 / (1 + \Gamma^2\theta^2)^2$ by $d\Gamma/ds$
   confirms the exact factor $\frac{1}{2s^2} \frac{\Gamma^5}{(1+\Gamma^2\theta^2)^2 (1+\hat a)}$.

### Code and test suite verification

1. **`stages.query_spectral_moments`**:
   The table kernel implements the DER015 generalization of this reduction. Setting
   $D=Q=\bar C=1$ recovers this derivation's support, inverse, Jacobian, and prefactor,
   while `KERNEL_NORMALIZATION_CONSTANT` applies the same normalization.

2. **Total yield agreement**:
   In `test_stage1_stage2.py::test_the_table_kernel_angle_integrates_to_stage_0_total`, integrating
   `spectrum_from_table` over solid angle $d\Omega$ and energy $ds$ agrees with Stage 0's total
   elementary photon count.

3. **Consistency between delta and table engine**:
   In `test_xigma_engine.py::test_angle_resolved_and_angle_integrated_normalizations_agree`,
   integrating `OutputKind.COLLIMATED_SPECTRUM` (table kernel with `KERNEL_NORMALIZATION_CONSTANT = 1.5 / (2*pi)`)
   over angles agrees with `OutputKind.SPECTRUM` (Stage 0 closed form) within the expected
   angular aperture capture fraction.

## Amendments

> **2026-09-28 — Head-on unchirped reduction of the five-dimensional model.** The
> normalization and delta-manifold reduction remain current. Production now samples
> $H(\gamma,\theta_{e,x},\theta_{e,y},\hat a,\bar C)$ and uses
> $A=1+Q\hat a$ and $D\bar C$ in the inverse/Jacobian (DER015). DER009's displayed
> formulas are the $D=Q=\bar C=1$ limit. DER017 adds the two finite-line spectral moment
> channels without changing the base-kernel normalization.


> **2026-09-28 — Resonance reparameterization does not by itself reduce the physical
> table dimension.** DER018 shows that the nonlinear line-centre shift can be packaged
> into a plane-wave quasi-momentum, so the *resonance position* may be written in terms
> of an effective momentum. That does not remove the information carried by \(\hat a\)
> from the full angle-resolved kernel. The factor
> \(\gamma^2/(1+\gamma^2\theta^2)^2\), together with the polarization geometry, depends
> on the physical electron momentum. Normalizing the quasi-momentum to an effective
> velocity discards its invariant norm \(q^2=m^2(1+\rho)\), so the physical \(p^\mu\)
> cannot in general be reconstructed. Keeping the full quasi-momentum restores that
> information but has the same number of degrees of freedom as \((p^\mu,\rho)\).
> Consequently the current five-dimensional \((\gamma,\theta_{e,x},\theta_{e,y},
> \hat a,\bar C)\) representation is not made redundant by the DER018 change of
> variables.
