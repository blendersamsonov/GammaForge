# DER011 — Closed-form angle-integrated linear Thomson spectrum

Status: verified

## Setup

In Compton/Thomson scattering experiments and simulations, calculating the full angle-integrated
emission spectrum:

$$
\frac{dN}{ds}
$$

is essential for measuring total spectral yield, characterizing radiation bandwidth, and providing
an exact analytical benchmark for numerical angle-resolved quadratures.

In the linear Thomson regime ($a_0 \ll 1$, so the nonlinear red-shift parameter $\hat a \to 0$),
the backscattering geometry is characterized by relativistic electrons with Lorentz factors
$\gamma \gg 1$ colliding with counter-propagating laser photons of frequency $\omega_L$.

In normalized photon energy $s \equiv \omega / (4\omega_L)$, an electron radiating at observation
angle $\boldsymbol{\theta} = (\theta_x, \theta_y)$ relative to its velocity vector emits photons
at the Doppler resonance energy:

$$
s(\theta) = \frac{\gamma^2}{1 + \gamma^2\theta^2}
$$

where $\theta \equiv |\boldsymbol{\theta}|$ is the polar observation angle. The maximum possible
photon energy occurs on-axis ($\theta = 0$) at the linear Compton edge $s_{\max} = \gamma^2$.

Defining the normalized energy variable:

$$
y \equiv \frac{s}{\gamma^2} = \frac{1}{1 + \gamma^2\theta^2}
$$

we have $y \in (0, 1]$ for physical polar angles $\theta \in [0, \infty)$.

The differential Thomson scattering cross-section in the lab frame (with the corrected prefactor
$3 / (2\pi)$ from DER009, RES026, and RES033) is:

$$
\frac{d^2\sigma}{d\Omega \, ds} = \sigma_T \, \frac{3}{2\pi} \, \frac{\gamma^2}{(1 + \gamma^2\theta^2)^2} \, \mathcal{P}(\theta, \phi) \, \delta(s - s(\theta))
$$

where $\sigma_T = \frac{8\pi}{3} r_e^2$ is the total Thomson cross section and $\mathcal{P}(\theta, \phi)$
is the polarization-dependent dipole emission pattern.

In this derivation:
1. We integrate $\frac{d^2\sigma}{d\Omega \, ds}$ over all solid angles $d\Omega \approx \theta \, d\theta \, d\phi$.
2. We evaluate the azimuthal and polar integrals in closed form to derive the universal single-electron
   spectral shape $S(y) = 1.5 [1 - 2y(1-y)]$.
3. We sum over macroparticles of luminosity $L_i$ to obtain the total spectrum:
   $\frac{dN}{ds} = \sum_i L_i \frac{1.5}{\gamma_i^2} [1 - 2y_i(1-y_i)]$.
4. We prove that $\int_0^1 1.5 [1 - 2y(1-y)] \, dy = 1$, demonstrating strict photon count conservation.

## Derivation

### 1. Azimuthal integration of the polarization factor

For linearly polarized incident radiation (e.g., polarized along the $x$-axis) in the head-on
limit, the polarization-dependent emission factor is:

$$
\mathcal{P}(\theta, \phi) = 1 - \frac{4\gamma^2\theta^2 \cos^2\phi}{(1 + \gamma^2\theta^2)^2}
$$

where $\phi$ is the azimuthal angle relative to the laser polarization direction.

Integrating over azimuth $\phi \in [0, 2\pi]$:

$$
\frac{1}{2\pi} \int_0^{2\pi} \cos^2\phi \, d\phi = \frac{1}{2}
$$

The azimuthally averaged polarization factor is therefore:

$$
\langle \mathcal{P}(\theta) \rangle_\phi = \frac{1}{2\pi} \int_0^{2\pi} \mathcal{P}(\theta, \phi) \, d\phi = 1 - \frac{2\gamma^2\theta^2}{(1 + \gamma^2\theta^2)^2}
$$

For circularly polarized laser light, the emission pattern is already rotationally symmetric and
equals this exact value without averaging (DER004). For arbitrary ellipticity $\epsilon$, the
same azimuthal integral yields identical results.

Defining $u \equiv \gamma^2\theta^2$, the factor is:

$$
\langle \mathcal{P}(u) \rangle_\phi = 1 - \frac{2u}{(1 + u)^2}
$$

### 2. Expressing the polarization factor in terms of $y$

From the Doppler resonance relation:

$$
y \equiv \frac{s}{\gamma^2} = \frac{1}{1 + u}
$$

we have:

$$
1 + u = \frac{1}{y} \implies u = \frac{1 - y}{y}
$$

Substituting this into the second term of $\langle \mathcal{P}(u) \rangle_\phi$:

$$
\frac{2u}{(1 + u)^2} = 2 \left(\frac{1 - y}{y}\right) y^2 = 2y(1 - y)
$$

Hence, the azimuthally averaged polarization factor takes the remarkably simple polynomial form:

$$
\boxed{\;\langle \mathcal{P} \rangle_\phi = 1 - 2y(1 - y)\;}
$$

### 3. Integration over solid angle and delta-function reduction

The solid angle element in the paraxial / small-angle approximation is:

$$
d\Omega \approx \theta \, d\theta \, d\phi = \frac{1}{2} d(\theta^2) \, d\phi = \frac{1}{2\gamma^2} du \, d\phi
$$

Integrating the differential cross section over $\phi \in [0, 2\pi]$:

$$
\frac{d\sigma}{ds} = \int d\Omega \, \frac{d^2\sigma}{d\Omega \, ds} = 2\pi \int_0^\infty \frac{1}{2\gamma^2} du \; \sigma_T \, \frac{3}{2\pi} \, \frac{\gamma^2}{(1 + u)^2} \, \langle \mathcal{P}(u) \rangle_\phi \, \delta(s - s(u))
$$

Collecting the constant factors:

$$
2\pi \times \frac{1}{2\gamma^2} \times \frac{3\sigma_T}{2\pi} \times \gamma^2 = \frac{3}{2} \sigma_T = 1.5 \, \sigma_T
$$

The integral over $u$ becomes:

$$
\frac{d\sigma}{ds} = 1.5 \, \sigma_T \int_0^\infty du \, \frac{1}{(1 + u)^2} \, [1 - 2y(u)(1 - y(u))] \, \delta(s - s(u))
$$

We now perform a change of integration variable from $u$ to resonant energy $s'(u) = \frac{\gamma^2}{1 + u}$.
Differentiating $s'$ with respect to $u$:

$$
ds' = -\frac{\gamma^2}{(1 + u)^2} du \implies \frac{1}{(1 + u)^2} du = -\frac{1}{\gamma^2} ds'
$$

The limits of integration transform as:
- At $u = 0$: $s' = \gamma^2$.
- As $u \to \infty$: $s' \to 0$.

Reversing the integration limits to absorb the minus sign:

$$
\int_0^\infty du \, \frac{1}{(1 + u)^2} \delta(s - s'(u)) = \int_0^{\gamma^2} \frac{1}{\gamma^2} ds' \, \delta(s - s') = \frac{1}{\gamma^2} \quad \text{for } 0 < s \le \gamma^2
$$

and $0$ for $s > \gamma^2$ or $s \le 0$.

Combining the prefactor, the Jacobian $1/\gamma^2$, and the shape factor:

$$
\frac{d\sigma}{ds} = \sigma_T \, \frac{1.5}{\gamma^2} \left[ 1 - 2y(1 - y) \right] \quad \left(y = \frac{s}{\gamma^2} \in [0, 1]\right)
$$

### 4. Summation over macroparticles and conservation of photon count

In the simulation framework, an electron bunch is represented by an ensemble of macroparticles,
where each macroparticle $i$ carries Lorentz factor $\gamma_i$ and interaction luminosity:

$$
L_i = \int v_{\text{rel}} \sigma_T n_{\text{ph}} \, dt
$$

Multiplying the single-electron cross-section distribution by the per-particle luminosity yields
the total angle-integrated spectral distribution:

$$
\boxed{\;\frac{dN}{ds} = \sum_i L_i \, \frac{1.5}{\gamma_i^2} \left[ 1 - 2y_i(1 - y_i) \right] \, H(y_i) \, H(1 - y_i), \qquad y_i \equiv \frac{s}{\gamma_i^2}\;}
$$

Now, we evaluate the total photon count by integrating over all photon energies $s \in [0, \infty)$:

$$
N_{\text{total}} = \int_0^\infty \frac{dN}{ds} \, ds = \sum_i L_i \int_0^{\gamma_i^2} \frac{1.5}{\gamma_i^2} \left[ 1 - 2\left(\frac{s}{\gamma_i^2}\right)\left(1 - \frac{s}{\gamma_i^2}\right) \right] ds
$$

Changing variables to $y = s / \gamma_i^2$, with $ds = \gamma_i^2 dy$:

$$
\int_0^{\gamma_i^2} \frac{1.5}{\gamma_i^2} [1 - 2y(1 - y)] \, ds = 1.5 \int_0^1 [1 - 2y(1 - y)] \, dy
$$

Expanding the polynomial integrand:

$$
1 - 2y(1 - y) = 1 - 2y + 2y^2
$$

Evaluating the definite integral term-by-term:

$$
\int_0^1 (1 - 2y + 2y^2) \, dy = \left[ y - y^2 + \frac{2}{3} y^3 \right]_0^1 = 1 - 1 + \frac{2}{3} = \frac{2}{3}
$$

Multiplying by the prefactor $1.5 = 3/2$:

$$
1.5 \times \frac{2}{3} = \frac{3}{2} \times \frac{2}{3} = 1
$$

Therefore:

$$
\boxed{\;\int_0^1 1.5 [1 - 2y(1 - y)] \, dy = 1\;}
$$

Consequently:

$$
N_{\text{total}} = \sum_i L_i \times 1 = \sum_i L_i
$$

The closed-form angle-integrated spectrum strictly conserves total photon number.

## Result

For a collection of relativistic electrons with Lorentz factors $\gamma_i$ and interaction
luminosities $L_i$, the angle-integrated linear Thomson emission spectrum is:

$$
\boxed{\;\frac{dN}{ds} = \sum_i L_i \, \frac{S(s/\gamma_i^2)}{\gamma_i^2}\;}
$$

where the universal single-electron spectral profile $S(y)$ is:

$$
\boxed{\;S(y) = \begin{cases} 1.5 \left[ 1 - 2y(1 - y) \right], & 0 \le y \le 1 \\ 0, & \text{otherwise} \end{cases}\;}
$$

The profile satisfies the exact normalization identity:

$$
\boxed{\;\int_0^1 S(y) \, dy = 1\;}
$$

ensuring that the total integrated photon count matches Stage 0's total yield bit-for-bit:

$$
\boxed{\;\int_0^\infty \frac{dN}{ds} \, ds = \sum_i L_i = N_{\text{total}}\;}
$$

## Verification

### Symbolic verification (Sympy)

The integral of $S(y) = \frac{3}{2}(1 - 2y + 2y^2)$ from $y = 0$ to $y = 1$ was evaluated
symbolically in Sympy and confirmed to equal $1$ identically.
The integration over $u = \gamma^2\theta^2$ of the differential form:

$$
3\pi \int_0^\infty \left[ \frac{1}{(1 + u)^2} - \frac{2u}{(1 + u)^4} \right] du = 3\pi \left[ 1 - \frac{2}{6} \right] = 2\pi
$$

reproduces the uncorrected $2\pi$ total that proved the necessity of the $1/(2\pi)$ factor in RES026.

### Implementation in code

The exact closed-form expression is implemented in two independent locations:
1. **`gammaforge.engines.xigma.stages.angle_integrated_spectrum`** (lines 1068–1072):
   ```python
   gamma_squared = (samples.gamma[start:stop] ** 2)[:, None]
   y = energy[None, :] / gamma_squared
   shape = np.where((y < 0.0) | (y > 1.0), 0.0, 1.5 * (1.0 - 2.0 * y * (1.0 - y)))
   return np.sum(samples.luminosity[start:stop, None] * shape / gamma_squared, axis=0)
   ```
2. **`gammaforge.validation.references.delta.single_electron_spectrum`** (lines 188–192):
   Provides the table-free anchor against which delta's numerical angular integration is validated.

### Test suite verification

1. **`test_xigma_engine.py::test_angle_integrated_spectrum_shape_integrates_to_one`**:
   Numerically integrates `angle_integrated_spectrum` over a fine energy grid and verifies agreement
   with `np.sum(samples.luminosity)` to within quadrature accuracy.
2. **`test_analytical.py::test_angle_integrated_spectrum_shape_and_scalar_input`**:
   Validates boundary cutoff behavior ($s > \gamma^2$ returns zero) and verifies scalar/array
   interface consistency.
