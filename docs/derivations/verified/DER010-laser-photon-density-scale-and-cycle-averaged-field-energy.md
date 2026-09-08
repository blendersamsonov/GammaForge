# DER010 — Laser photon density scale and cycle-averaged field energy

Status: verified

## Setup

In GammaForge, laser-electron bunch collision observables (Stage 0 photon yield, luminosity, and
the mean nonlinear red-shift parameter $\hat a$) are evaluated from the laser field. Historically,
the predecessor code converted laser pulse energy into a peak dimensionless vector potential
$a_0$, which required tracking polarization-dependent cycle-average factors $C$ across multiple
modules (RES053).

RES054 eliminated this source of convention discrepancies by routing all trajectory integration
directly through the local cycle-averaged normalized intensity:

$$
\langle a^2(t, \mathbf{r}) \rangle
$$

provided by `LaserField.intensity_profile`. To compute the physical scattering rate and total
photon yield, this dimensionless intensity must be converted to the local physical photon number
density:

$$
n_{\text{ph}}(t, \mathbf{r}) \quad [\text{photons} \cdot \text{cm}^{-3}]
$$

In this derivation:
1. We compute the total electromagnetic energy density $u$ in CGS-Gaussian units.
2. We establish the exact relationship between the electric field magnitude $\mathbf{E}$ and the
   dimensionless vector potential $\mathbf{a}$.
3. We derive the physical photon number density scale $n_{\text{ph}} = \frac{(m_e c)^2 \omega_0}{4\pi \hbar e^2} \langle a^2 \rangle$.
4. We prove why the denominator contains $4\pi$ (rather than $8\pi$) when multiplying cycle-averaged
   $\langle a^2 \rangle$, and prove why this conversion is strictly polarization-agnostic (RES054).
5. We derive the local collision scattering rate $(1 - \hat{\mathbf{n}}_0 \cdot \boldsymbol{\beta}) c \sigma_T n_{\text{ph}}$
   integrated by Stage 0.

## Derivation

### 1. CGS-Gaussian electromagnetic field energy density

In CGS-Gaussian units, the electric field $\mathbf{E}$ and magnetic field $\mathbf{B}$ have the
same physical dimensions ($\text{statV} \cdot \text{cm}^{-1} = \text{Gauss}$). The instantaneous
electromagnetic energy density in vacuum is:

$$
u(t, \mathbf{r}) = \frac{|\mathbf{E}(t, \mathbf{r})|^2 + |\mathbf{B}(t, \mathbf{r})|^2}{8\pi}
$$

For a transverse electromagnetic wave propagating in vacuum (such as a paraxial laser pulse),
$|\mathbf{B}| = |\mathbf{E}|$ everywhere. Thus, the instantaneous energy density simplifies to:

$$
u(t, \mathbf{r}) = \frac{|\mathbf{E}(t, \mathbf{r})|^2}{4\pi}
$$

Averaging over an optical cycle $T = 2\pi / \omega_0$, where $\langle \dots \rangle$ denotes the
cycle average:

$$
\langle u(t, \mathbf{r}) \rangle = \frac{\langle |\mathbf{E}(t, \mathbf{r})|^2 \rangle}{4\pi}
$$

### 2. Relation between electric field and dimensionless potential

The dimensionless vector potential $\mathbf{a}$ represents the normalized electron quiver momentum:

$$
\mathbf{a} \equiv \frac{e\mathbf{A}}{m_e c^2}
$$

where $e > 0$ is the elementary charge in $\text{statC}$ ($\text{esu}$), $m_e$ is the electron
rest mass, and $c$ is the speed of light.

In the Coulomb/radiation gauge ($\Phi = 0$, $\nabla \cdot \mathbf{A} = 0$), the electric field of a
quasimonochromatic laser carrier with central angular frequency $\omega_0$ is:

$$
\mathbf{E} = -\frac{1}{c} \frac{\partial \mathbf{A}}{\partial t} \approx \frac{\omega_0}{c} \mathbf{A} = \frac{m_e c \omega_0}{e} \mathbf{a}
$$

Squaring this relation and taking the time average over an optical cycle:

$$
\langle |\mathbf{E}|^2 \rangle = \left(\frac{m_e c \omega_0}{e}\right)^2 \langle a^2 \rangle
$$

Here $\langle a^2 \rangle \equiv \langle |\mathbf{a}|^2 \rangle$ is the cycle-averaged square of the
dimensionless vector potential.

Substituting $\langle |\mathbf{E}|^2 \rangle$ into the cycle-averaged energy density:

$$
\langle u \rangle = \frac{1}{4\pi} \left(\frac{m_e c \omega_0}{e}\right)^2 \langle a^2 \rangle
$$

### 3. Physical photon density scale

Each laser photon carries a discrete quantum of energy:

$$
\mathcal{E}_{\text{ph}} = \hbar \omega_0
$$

where $\hbar$ is the reduced Planck constant in $\text{erg} \cdot \text{s}$.

The cycle-averaged physical photon number density $n_{\text{ph}}$ ($\text{photons} \cdot \text{cm}^{-3}$)
is the total field energy density divided by the single-photon energy:

$$
n_{\text{ph}} = \frac{\langle u \rangle}{\hbar \omega_0} = \frac{1}{4\pi \hbar \omega_0} \left(\frac{m_e c \omega_0}{e}\right)^2 \langle a^2 \rangle
$$

Simplifying the factor of $\omega_0$:

$$
\boxed{\;n_{\text{ph}} = \frac{(m_e c)^2 \omega_0}{4\pi \hbar e^2} \langle a^2 \rangle\;}
$$

Checking the physical dimensions in CGS-Gaussian units:
- Numerator $(m_e c)^2 \omega_0$: $(\text{g} \cdot \text{cm} \cdot \text{s}^{-1})^2 \cdot \text{s}^{-1} = \text{g}^2 \cdot \text{cm}^2 \cdot \text{s}^{-3}$.
- Denominator $\hbar e^2$: $(\text{erg} \cdot \text{s}) \cdot (\text{statC}^2) = (\text{g} \cdot \text{cm}^2 \cdot \text{s}^{-1}) \cdot (\text{g} \cdot \text{cm}^3 \cdot \text{s}^{-2}) = \text{g}^2 \cdot \text{cm}^5 \cdot \text{s}^{-3}$.
- Quotient: $\frac{\text{g}^2 \cdot \text{cm}^2 \cdot \text{s}^{-3}}{\text{g}^2 \cdot \text{cm}^5 \cdot \text{s}^{-3}} = \text{cm}^{-3}$.

Since $\langle a^2 \rangle$ is dimensionless, $n_{\text{ph}}$ has exact dimensions of $\text{cm}^{-3}$.

### 4. Proof of the $4\pi$ factor and polarization agnosticism

A frequent point of confusion in strong-field physics literature is whether the prefactor contains
$4\pi$ or $8\pi$. The origin of $8\pi$ lies in referencing the **peak** field amplitude of a
linearly polarized wave rather than the cycle-averaged intensity.

Consider a linearly polarized laser pulse with peak field amplitude $E_0$ and peak dimensionless
vector potential $a_0 = e E_0 / (m_e c \omega_0)$:

$$
\mathbf{E}(t) = E_0 \cos(\omega_0 t) \, \hat{\mathbf{x}} \implies \langle |\mathbf{E}|^2 \rangle = \frac{1}{2} E_0^2
$$

The cycle-averaged energy density expressed in terms of the peak amplitude $E_0$ is:

$$
\langle u \rangle = \frac{\frac{1}{2} E_0^2}{4\pi} = \frac{E_0^2}{8\pi} = \frac{1}{8\pi} \left(\frac{m_e c \omega_0}{e}\right)^2 a_0^2 \quad \text{(linear)}
$$

Now consider an elliptically polarized wave with ellipticity $\epsilon \in [-1, 1]$ and major-axis
peak amplitude $E_0$:

$$
\mathbf{E}(t) = \frac{E_0}{\sqrt{1 + \epsilon^2}} \left[ \cos(\omega_0 t) \hat{\mathbf{x}} + \epsilon \sin(\omega_0 t) \hat{\mathbf{y}} \right]
$$

Its cycle average is:

$$
\langle |\mathbf{E}|^2 \rangle = \frac{E_0^2}{1 + \epsilon^2} \left[ \langle \cos^2(\omega_0 t) \rangle + \epsilon^2 \langle \sin^2(\omega_0 t) \rangle \right] = \frac{E_0^2}{1 + \epsilon^2} \left[ \frac{1}{2} + \frac{\epsilon^2}{2} \right] = \frac{1}{2} E_0^2
$$

In general, defining $\langle a^2 \rangle = C a_0^2$, the cycle-averaging factor is:

$$
C = \frac{1 + \epsilon^2}{2}
$$

For linear polarization ($\epsilon = 0$), $C = 1/2$. For circular polarization ($\epsilon = \pm 1$),
$C = 1$. If one expresses photon density in terms of the peak amplitude $a_0$, the relationship
carries an explicit dependence on $C$:

$$
n_{\text{ph}} = C \, \frac{(m_e c)^2 \omega_0}{4\pi \hbar e^2} a_0^2
$$

However, when expressing photon density directly in terms of the cycle-averaged intensity
$\langle a^2 \rangle$:

$$
n_{\text{ph}} = \frac{(m_e c)^2 \omega_0}{4\pi \hbar e^2} \langle a^2 \rangle
$$

the factor $C$ cancels completely. At fixed pulse energy:

$$
\mathcal{E}_{\text{pulse}} = \int \langle u \rangle \, d^3\mathbf{r} = \hbar \omega_0 \int n_{\text{ph}} \, d^3\mathbf{r}
$$

the total number of photons and the cycle-averaged field energy density are physical invariants
independent of the laser polarization state. A circularly polarized pulse of the same total energy
has peak $a_0$ smaller by $\sqrt{2}$ than a linearly polarized pulse, but carries twice the
cycle-averaged energy per unit $a_0^2$. The two effects offset exactly.

Therefore:
- The factor is $4\pi$, not $8\pi$, because the $1/2$ cycle factor is already subsumed within
  $\langle a^2 \rangle$.
- The scale factor is completely polarization-agnostic.

### 5. Collision scattering rate

An electron with velocity $\mathbf{v} = c\boldsymbol{\beta}$ traversing a laser pulse propagating
along unit vector $\hat{\mathbf{n}}_0$ encounters photons with relative velocity flux factor:

$$
v_{\text{rel}} = c(1 - \hat{\mathbf{n}}_0 \cdot \boldsymbol{\beta})
$$

For head-on collision ($\hat{\mathbf{n}}_0 = -\hat{\mathbf{z}}$ and $\boldsymbol{\beta} \approx \hat{\mathbf{z}}$),
$1 - \hat{\mathbf{n}}_0 \cdot \boldsymbol{\beta} = 1 + \beta \approx 2$.

The collision scattering rate per electron is given by the product of relative velocity, the total
Thomson cross section $\sigma_T = \frac{8\pi}{3} r_e^2$, and the local photon density:

$$
\frac{dN_{\text{scat}}}{dt} = v_{\text{rel}} \, \sigma_T \, n_{\text{ph}}(t, \mathbf{r}) = (1 - \hat{\mathbf{n}}_0 \cdot \boldsymbol{\beta}) c \, \sigma_T \, n_{\text{ph}}(t, \mathbf{r})
$$

Substituting the expression for $n_{\text{ph}}$:

$$
\frac{dN_{\text{scat}}}{dt} = (1 - \hat{\mathbf{n}}_0 \cdot \boldsymbol{\beta}) c \, \sigma_T \left[ \frac{(m_e c)^2 \omega_0}{4\pi \hbar e^2} \right] \langle a^2(t, \mathbf{r}(t)) \rangle
$$

Integrating along the electron trajectory $\mathbf{r}(t)$ yields the single-particle luminosity
(number of scattered photons per electron):

$$
L_i = \int_{-\infty}^{\infty} (1 - \hat{\mathbf{n}}_0 \cdot \boldsymbol{\beta}_i) c \, \sigma_T \, n_{\text{ph}}(t, \mathbf{r}_i(t)) \, dt
$$

Summing over all macroparticles gives the total interaction yield $N_{\text{total}} = \sum_i L_i$.

## Result

The physical photon number density scale per unit cycle-averaged normalized intensity $\langle a^2 \rangle$
in CGS-Gaussian units is:

$$
\boxed{\;\text{photon\_density\_scale} = \frac{(m_e c)^2 \omega_0}{4\pi \hbar e^2}\;}
$$

The local photon number density is:

$$
\boxed{\;n_{\text{ph}}(t, \mathbf{r}) = \text{photon\_density\_scale} \times \langle a^2(t, \mathbf{r}) \rangle\;}
$$

The instantaneous Thomson scattering rate per electron is:

$$
\boxed{\;\frac{dN_{\text{scat}}}{dt} = (1 - \hat{\mathbf{n}}_0 \cdot \boldsymbol{\beta}) c \, \sigma_T \, n_{\text{ph}}(t, \mathbf{r})\;}
$$

## Verification

### Implementation verification

In `gammaforge.engines.xigma.stages.photon_density_scale`, the scale factor is implemented directly:
```python
def photon_density_scale(laser: LaserField) -> float:
    if hasattr(laser, "omega0"):
        omega0 = laser.omega0()
    else:
        omega0 = fit_gaussian_paraxial(laser).omega0()
    return (ME_CGS * C_CGS) ** 2 * omega0 / (4.0 * math.pi * HBAR_CGS * E_ESU**2)
```

### Numerical round-trip and polarization invariance

1. **`test_stage0_delta.py::test_the_photon_density_scale_inverts_the_lasers_own_intensity_chain`**:
   Parametrized across `ellipticity` $\in \{0.0, 0.5, 1.0\}$.
   Tests that `photon_density_scale(laser) * laser.intensity_profile(*point)` matches
   `laser.n_photons() * laser.photon_density(*point)` to machine precision ($10^{-14}$),
   confirming that the scale factor exactly inverts the laser's energy-to-intensity definition
   independent of polarization state.

2. **`test_stage0_delta.py::test_stage_0_is_bit_identical_under_any_polarization`**:
   Confirms that Stage 0 yields, luminosities, and $\hat a$ moments are bit-identical for linear,
   elliptical, and circular pulses at fixed pulse energy.

3. **Gaussian unit identities in `test_units.py`**:
   Confirms elementary charge $e$, classical electron radius $r_e = e^2 / (m_e c^2)$, Thomson cross
   section $\sigma_T = \frac{8\pi}{3} r_e^2$, and fine structure constant $\alpha = e^2 / (\hbar c)$
   satisfy CODATA identities in CGS-Gaussian units without stray $4\pi\epsilon_0$ factors.
