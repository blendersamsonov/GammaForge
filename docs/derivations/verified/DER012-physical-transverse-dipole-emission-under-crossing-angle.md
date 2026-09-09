# DER012 — Physical transverse dipole emission and photon conservation under crossing angle

Status: verified

## Setup

In DER005 and DER006, the polarization factor $\operatorname{Tr}(\hat U^T \hat\Xi \hat U)$ for laser–electron interactions with an oblique laser crossing angle was evaluated from manuscript Eq. `udef`:

$$
\mathbf{u}_i = \frac{(\mathbf{n}-\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}} - \mathbf{e}_i
$$

where $\mathbf{e}_i = R\,\mathbf{e}_i^{(0)}$ is the rotated laser polarization unit vector in the laboratory frame. Because the tilted vector $\mathbf{e}_0$ has a non-zero longitudinal projection $b_0 = \mathbf{v}\cdot\mathbf{e}_0 \approx -\sin\alpha$, DER005 and DER006 retained the cross-term:

$$
\mathbf{u}_i\cdot\mathbf{u}_j = \delta_{ij} - \frac{a_i a_j}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2} + \frac{a_i b_j + a_j b_i}{1-\mathbf{v}\cdot\mathbf{n}}
$$

where $a_i \equiv \mathbf{n}\cdot\mathbf{e}_i$ and $b_i \equiv \mathbf{v}\cdot\mathbf{e}_i$.

When integrated over all solid angles $d\Omega$, this third term introduces an uncompensated $+4(\gamma\alpha)^2$ contribution:

$$
\int d\Omega \, \frac{\gamma^2}{(1+\gamma^2\theta^2)^2} \mathcal{P}(\theta, \phi) = \frac{2\pi}{3}\left[ 1 + \frac{(\gamma\alpha_\parallel)^2}{1+\varepsilon^2} \right]
$$

For relativistic bunches ($\gamma = 2000$) colliding at small angles ($\alpha = 0.5\,\text{mrad} \implies \gamma\alpha = 1.0$), this factor inflates the total cross section to $2.0\,\sigma_T$, causing the collimated target yield to exceed the total bunch yield ($N_{\text{tgt}} > N_{\text{tot}}$, reaching $\sim 110\%$).

This derivation identifies the physical inconsistency in Eq. `udef` and establishes the physical transverse dipole emission model that strictly preserves total Thomson cross-section $\sigma_T$ and photon count conservation under arbitrary crossing angles.

---

## 1. Physical acceleration in the electron rest frame

Consider an ultra-relativistic electron colliding with a laser pulse of frequency $\omega_L$ propagating along unit vector $\hat{\mathbf{n}}_0$ at crossing angle $\alpha$ relative to the electron beam axis $-\hat{\mathbf{z}}$.

In the rest frame of the electron ($K'$), the electron velocity is $\mathbf{v}' = 0$. By the Lorentz transformation of electromagnetic fields, the electric and magnetic fields in $K'$ are:

$$
\mathbf{E}'_\parallel = \mathbf{E}_\parallel,\qquad \mathbf{E}'_\perp = \gamma(\mathbf{E}_\perp + \boldsymbol{\beta}\times\mathbf{B})
$$

For a plane wave with $\mathbf{B} = \hat{\mathbf{n}}_0 \times \mathbf{E}$ and linear polarization in the $xz$-crossing plane ($\mathbf{e}_0 = (\cos\alpha, 0, -\sin\alpha)$):

$$
\mathbf{E}_\perp = E_0 \cos\alpha\,\hat{\mathbf{x}},\qquad \mathbf{E}_\parallel = -E_0 \sin\alpha\,\hat{\mathbf{z}}
$$

$$
\mathbf{B} = -E_0\,\hat{\mathbf{y}} \implies \boldsymbol{\beta}\times\mathbf{B} = \beta E_0\,\hat{\mathbf{x}}
$$

Therefore:

$$
\mathbf{E}'_\perp = \gamma(\cos\alpha + \beta)E_0\,\hat{\mathbf{x}} \approx 2\gamma E_0\,\hat{\mathbf{x}}
$$

$$
\mathbf{E}'_\parallel = -E_0\sin\alpha\,\hat{\mathbf{z}} \approx -E_0\alpha\,\hat{\mathbf{z}}
$$

The ratio of longitudinal to transverse field in the rest frame is:

$$
\frac{E'_z}{E'_x} \approx \frac{-\alpha}{\gamma(1+\beta)} \approx -\frac{\alpha}{2\gamma} = -\frac{\gamma\alpha}{2\gamma^2}
$$

For typical experimental parameters ($\gamma \sim 2000$, $\alpha = 0.5\,\text{mrad}$):

$$
\left|\frac{E'_z}{E'_x}\right| \approx 1.25 \times 10^{-7} \ll 1
$$

In the electron rest frame, the wave is relativistically weak and the electric field is transverse to the electron velocity vector within an accuracy of $O(\alpha / \gamma)$. The radiating dipole is, to order $1/\gamma$, transverse electric dipole radiation:

$$
\frac{d\sigma'}{d\Omega'} = \frac{3\sigma_T}{8\pi}\left[ 1 - (\mathbf{n}'\cdot\hat{\mathbf{e}}')^2 \right]
$$

which integrates over all rest-frame solid angle to identically $\sigma_T$.

---

## 2. Relativistic longitudinal inertia in the laboratory frame

In the laboratory frame, Jackson Eq. (14.67) expresses the radiated field in terms of the electron acceleration $\dot{\mathbf{v}}$:

$$
\mathbf{E}_{\text{rad}} \propto \frac{\mathbf{n}\times[(\mathbf{n}-\mathbf{v})\times\dot{\mathbf{v}}]}{(1-\mathbf{v}\cdot\mathbf{n})^2}
$$

The manuscript Eq. (267) approximated $\dot{\mathbf{v}} \propto \mathbf{E}$, setting $\dot{\mathbf{v}} \propto \mathbf{e}_0$.

However, in relativistic dynamics, the equation of motion is $\mathbf{F} = \frac{d}{dt}(\gamma m \mathbf{v}) = \gamma m \dot{\mathbf{v}} + \gamma^3 m \mathbf{v}(\mathbf{v}\cdot\dot{\mathbf{v}})$, which inverts to:

$$
\dot{\mathbf{v}} = \frac{1}{\gamma m}\left[ \mathbf{F} - \frac{\mathbf{v}}{c^2}(\mathbf{v}\cdot\mathbf{F}) \right]
$$

With $\mathbf{F} = -e(\mathbf{E} + \mathbf{v}\times\mathbf{B})$:
- Transverse force component: $F_x = -e(\cos\alpha + v/c)E_0 \implies \dot{v}_x = -\frac{e}{\gamma m}(\cos\alpha + v/c)E_0$.
- Longitudinal force component: $F_z = e\sin\alpha E_0$.
- Longitudinal acceleration:
  $$
  \dot{v}_z = \frac{1}{\gamma m}\left[ F_z - \frac{v^2}{c^2}F_z \right] = \frac{1}{\gamma^3 m}F_z = \frac{e}{\gamma^3 m}\sin\alpha E_0
  $$

The longitudinal inertia is the **longitudinal relativistic mass** $\gamma^3 m$, not $\gamma m$. The ratio of longitudinal to transverse acceleration in the lab frame is:

$$
\frac{\dot{v}_z}{\dot{v}_x} = -\frac{\sin\alpha}{\gamma^2(\cos\alpha + v/c)} \approx -\frac{\alpha}{2\gamma^2}
$$

For $\gamma = 2000, \alpha = 0.5\,\text{mrad}$:

$$
\frac{|\dot{v}_z|}{|\dot{v}_x|} \approx \frac{5\times 10^{-4}}{2(2000)^2} \approx 6.25\times 10^{-11}
$$

**Conclusion:** Longitudinal acceleration is suppressed by the relativistic longitudinal mass $\gamma^3 m$ by a factor of $2\gamma^2 \approx 8\times 10^6$ relative to the transverse acceleration. The physical acceleration vector $\dot{\mathbf{v}}_e$ is transverse to each individual electron's velocity vector $\mathbf{v}_e$ to order $O(1/\gamma^2)$. Setting $\mathbf{v}_e \cdot \dot{\mathbf{v}}_e \equiv 0$ is an impeccable approximation that discards only these $O(1/\gamma^2) \sim 10^{-11}$ residual terms.

---

## 3. The origin of the $(\gamma\alpha)^2$ artifact

When the tilted laser vector $\mathbf{e}_0 = (\cos\alpha, 0, -\sin\alpha)$ was substituted for $\dot{\mathbf{v}}$ in Eq. `udef`, a spurious longitudinal acceleration $w_z \propto -\sin\alpha \approx -\alpha$ was introduced (larger than physical reality by a factor of $2\gamma^2$).

Because the ultra-relativistic radiation vector carries the Doppler beaming factor:

$$
\frac{\mathbf{n}-\mathbf{v}}{1-\mathbf{v}\cdot\mathbf{n}} \sim 2\gamma^2
$$

multiplying this factor by $-\alpha$ created a term of order $\gamma(\gamma\alpha)$ in $\mathbf{u}_0$. Squaring this term produced the unphysical $+4(\gamma\alpha)^2$ cross term in DER005/DER006, inflating the total cross section by $[1 + (\gamma\alpha)^2]$.

---

## 4. Local per-electron transverse projection

The relevant transverse plane must be defined **locally for each electron's instantaneous velocity** $\hat{\mathbf{v}}_e \approx (\theta_{xe}, \theta_{ye}, 1)/\sqrt{1+\theta_{xe}^2+\theta_{ye}^2}$, not merely relative to the nominal bunch central axis $\hat{\mathbf{z}}$.

If the bunch axis $\hat{\mathbf{z}}$ were used for a divergent bunch with $\sigma_\theta = 10^{-4}$ at $\gamma = 2000$ ($\gamma\sigma_\theta \approx 0.2$), the orientation of the transverse plane would differ by $O(\theta_e)$, which would re-introduce artificial normalization errors of order $(\gamma\sigma_\theta)^2 \approx 4\%$.

Therefore, for each electron with velocity direction $\hat{\mathbf{u}}_e = \mathbf{v}_e / |\mathbf{v}_e|$, the laboratory rotated laser axes $\mathbf{e}_0, \mathbf{e}_1$ are projected locally onto the plane perpendicular to $\hat{\mathbf{u}}_e$:

$$
\hat{\mathbf{e}}_{0,\perp, e} = \frac{\mathbf{e}_0 - (\mathbf{e}_0\cdot\hat{\mathbf{u}}_e)\hat{\mathbf{u}}_e}{\left|\mathbf{e}_0 - (\mathbf{e}_0\cdot\hat{\mathbf{u}}_e)\hat{\mathbf{u}}_e\right|}
$$

$$
\hat{\mathbf{e}}_{1,\perp, e} = \hat{\mathbf{u}}_e \times \hat{\mathbf{e}}_{0,\perp, e}
$$

By construction:

$$
\hat{\mathbf{u}}_e \cdot \hat{\mathbf{e}}_{i,\perp, e} = 0 \implies b_i \equiv 0
$$

The unphysical cross-term $\frac{a_i b_j + a_j b_i}{1-\mathbf{v}\cdot\mathbf{n}}$ vanishes identically for every electron. The single-particle polarization overlap becomes:

$$
\mathbf{u}_i\cdot\mathbf{u}_j = \delta_{ij} - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{i,\perp, e})(\mathbf{n}\cdot\hat{\mathbf{e}}_{j,\perp, e})}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
$$

and for general ellipticity $\varepsilon \in [0, 1]$:

$$
\operatorname{Tr}(\hat U^T \hat\Xi \hat U) = \frac{1}{1+\varepsilon^2}\left[ 1 - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{0,\perp, e})^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2} \right] + \frac{\varepsilon^2}{1+\varepsilon^2}\left[ 1 - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{1,\perp, e})^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2} \right]
$$

---

## 5. Physical angular asymmetry and suppression

In the unprojected formula, the cross-term generated a large laboratory angular tilt scaling as $O(\gamma\alpha) \sim 1$. In reality:

1. **Rest-frame dipole tilt:** The rest-frame electric field is inclined by:
   $$
   \delta \approx \frac{\alpha}{2\gamma} \approx 1.25\times 10^{-7}\text{ rad}
   $$
2. **Thomson pattern asymmetry:** With $\hat{\mathbf{e}}' \approx (1, 0, -\delta)$, the Thomson differential cross-section is:
   $$
   1 - (\mathbf{n}'\cdot\hat{\mathbf{e}}')^2 \approx 1 - n_x'^2 + 2\delta n_x' n_z' + O(\delta^2)
   $$
   The odd term $2\delta n_x' n_z'$ breaks reflection symmetry across the $yz$-plane (inversion symmetry $P(\mathbf{n}') = P(-\mathbf{n}')$ is preserved). The relative distortion of the radiation pattern is:
   $$
   O\left(\frac{\alpha}{\gamma}\right) \sim 2.5\times 10^{-7}
   $$
3. **Laboratory angular deflection:** Boosting along $\hat{\mathbf{z}}$ compresses the laboratory angular centroid displacement by another factor of $1/(2\gamma)$:
   $$
   \Delta\theta_x \sim \frac{\alpha}{2\gamma^2} \sim 6\times 10^{-11}\text{ rad}
   $$
   This physical deflection is ten million times smaller than the $1/\gamma = 0.5\,\text{mrad}$ radiation cone. There is no observable angular tilt in the ultra-relativistic regime.
4. **Quantum recoil:** With $\gamma = 2000$ and $\hbar\omega_0 = 1.2\,\text{eV}$:
   $$
   \frac{4\gamma\hbar\omega_0}{mc^2} \approx 0.019
   $$
   Quantum recoil is at the $\sim 2\%$ level in the emitted photon energy, but does not produce classical angular asymmetry.

---

## Result

1. The polarization factor $\mathcal{P} = \operatorname{Tr}(\hat U^T \hat\Xi \hat U)$ is evaluated locally per electron:
   $$
   \boxed{\;
   \operatorname{Tr}(\hat U^T \hat\Xi \hat U) = 1 - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{0,\perp, e})^2 + \varepsilon^2(\mathbf{n}\cdot\hat{\mathbf{e}}_{1,\perp, e})^2}{(1+\varepsilon^2)\,\gamma^2(1-\mathbf{v}_e\cdot\mathbf{n})^2}
   \;}
   $$
   where $\hat{\mathbf{e}}_{0,\perp, e} = \frac{\mathbf{e}_0 - (\mathbf{e}_0\cdot\hat{\mathbf{u}}_e)\hat{\mathbf{u}}_e}{|\mathbf{e}_0 - (\mathbf{e}_0\cdot\hat{\mathbf{u}}_e)\hat{\mathbf{u}}_e|}$ and $\hat{\mathbf{e}}_{1,\perp, e} = \hat{\mathbf{u}}_e \times \hat{\mathbf{e}}_{0,\perp, e}$.

2. Stage 0 computes the total number of scattering events $N_{\text{tot}}$ from the physical relative velocity and spatial overlap:
   $$
   \frac{dN_{\text{sc}}}{dt} \propto \sigma_T c n_\gamma (1 - \boldsymbol{\beta}_e \cdot \hat{\mathbf{n}}_0)
   $$
   Stage 2 is a normalized conditional angular distribution:
   $$
   \boxed{\;\int P_e(\Omega)\,d\Omega = 1\quad\iff\quad \int d\Omega \, \frac{d\sigma}{d\Omega} = \sigma_T\;}
   $$
   No secondary laser field projection loss is applied, preserving photon number conservation ($N_{\text{tgt}} \le N_{\text{tot}}$).

3. The crossing angle enters the emission through its physical kinematic channels:
   - Relative collision flux: $v_{\text{rel}} = c(1 - \boldsymbol{\beta}_e\cdot\hat{\mathbf{n}}_0) \approx 2c\cos^2(\alpha/2)$.
   - Doppler resonance frequency: $\omega_R \approx \frac{4\gamma^2\omega_L\cos^2(\alpha/2)}{1+\gamma^2\theta^2+\hat a}$.

---

## Verification

- **Longitudinal acceleration suppression:** Relativistic dynamics confirmed $\dot{v}_z / \dot{v}_x \approx -\alpha / (2\gamma^2) \sim 6\times 10^{-11}$.
- **Per-electron vs Exact acceleration:** Agreement verified across emittance $\sigma_\theta = 10^{-4}$ to $< 2 \times 10^{-7}$.
- **Cross-section integration:** $\int d\Omega \, \frac{\gamma^2}{(1+\gamma^2\theta^2)^2}\mathcal{P} = \frac{2\pi}{3}$ verified numerically across crossing angles and ellipticities (`tests/test_xigma_polarization_crossing.py`).
- **Target yield bound:** $N_{\text{tgt}} \le N_{\text{tot}}$ checked across the finite geometry/polarization grid in `tests/test_xigma_polarization_crossing.py`; this is not an all-parameter numerical guarantee.
- **Fast test suite:** All 462 fast tests pass (`tests/test_xigma_stokes.py`, `tests/test_stage0_delta.py`, `tests/test_xigma_gpu_polarization.py`).

---

## Used by

- Supersedes DER005 §2.3 and DER006 §2 for the crossing-angle correction in $\mathbf{u}_i\cdot\mathbf{u}_j$.
- Implemented in `gammaforge.engines.xigma.stages` (`polarization_factor_vectorized`, `compute_stokes_components`) and `spectrum_sampler` (CuPy rawkernel).
