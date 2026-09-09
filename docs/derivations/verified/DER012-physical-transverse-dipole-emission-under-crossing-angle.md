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

This derivation identifies the exact physical inconsistency in Eq. `udef` and derives the physical transverse dipole emission that strictly preserves total Thomson cross-section $\sigma_T$ and photon count conservation under arbitrary crossing angles.

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
\frac{E'_z}{E'_x} \approx \frac{-\alpha}{2\gamma} = -\frac{\gamma\alpha}{2\gamma^2}
$$

For typical experimental parameters ($\gamma \sim 10^2 - 10^4$, $\alpha \lesssim 10^{-2}$):

$$
\left|\frac{E'_z}{E'_x}\right| \lesssim 10^{-6} \ll 1
$$

In the electron rest frame, the wave is relativistically weak and the electric field is **strictly transverse** to the beam axis to an accuracy of $O(\alpha / \gamma)$. The radiating dipole is pure transverse electric dipole radiation:

$$
\frac{d\sigma'}{d\Omega'} = \frac{3\sigma_T}{8\pi}\left[ 1 - (\mathbf{n}'\cdot\hat{\mathbf{x}}')^2 \right]
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

For $\gamma = 2000, \alpha = 0.5\,\text{mrad}$, $\dot{v}_z / \dot{v}_x \approx 6\times 10^{-11} \approx 0$.

**Conclusion:** The electron does not accelerate longitudinally under the laser electric field. The physical acceleration vector $\dot{\mathbf{v}}$ in both frames is purely transverse to the electron velocity vector $\mathbf{v}$.

---

## 3. The origin of the $(\gamma\alpha)^2$ artifact

When the tilted laser vector $\mathbf{e}_0 = (\cos\alpha, 0, -\sin\alpha)$ was substituted for $\dot{\mathbf{v}}$ in Eq. `udef`, a spurious longitudinal acceleration $w_z \propto -\sin\alpha \approx -\alpha$ was introduced (larger than physical reality by a factor of $\gamma^2$).

Because the ultra-relativistic radiation vector carries the beaming factor:

$$
\frac{\mathbf{n}-\mathbf{v}}{1-\mathbf{v}\cdot\mathbf{n}} \sim 2\gamma^2
$$

multiplying this factor by $-\alpha$ created a term of order $\gamma(\gamma\alpha)$ in $\mathbf{u}_0$. Squaring this term produced the unphysical $+4(\gamma\alpha)^2$ cross term in DER005/DER006, inflating the total cross section by $[1 + (\gamma\alpha)^2]$.

---

## 4. Transverse-projected polarization basis

To reflect the true physical acceleration $\dot{\mathbf{v}} \perp \mathbf{v}$, the polarization basis vectors $\mathbf{e}_i$ entering Jackson's radiation formula must be projected onto the plane transverse to the electron velocity $\mathbf{v}$:

$$
\hat{\mathbf{e}}_{i,\perp} = \frac{\mathbf{e}_i - (\mathbf{e}_i\cdot\hat{\mathbf{v}})\hat{\mathbf{v}}}{\left|\mathbf{e}_i - (\mathbf{e}_i\cdot\hat{\mathbf{v}})\hat{\mathbf{v}}\right|}
$$

By construction:

$$
\mathbf{v}\cdot\hat{\mathbf{e}}_{i,\perp} = 0 \implies b_i \equiv 0
$$

The third term $\frac{a_i b_j + a_j b_i}{1-\mathbf{v}\cdot\mathbf{n}}$ vanishes identically.

The single-particle polarization overlap becomes:

$$
\mathbf{u}_i\cdot\mathbf{u}_j = \delta_{ij} - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{i,\perp})(\mathbf{n}\cdot\hat{\mathbf{e}}_{j,\perp})}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
$$

and for general ellipticity $\varepsilon \in [0, 1]$:

$$
\operatorname{Tr}(\hat U^T \hat\Xi \hat U) = \frac{1}{1+\varepsilon^2}\left[ 1 - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{0,\perp})^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2} \right] + \frac{\varepsilon^2}{1+\varepsilon^2}\left[ 1 - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{1,\perp})^2}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2} \right]
$$

---

## Result

1. The polarization factor $\mathcal{P} = \operatorname{Tr}(\hat U^T \hat\Xi \hat U)$ is evaluated using the transverse-projected polarization vectors:
   $$
   \boxed{\;
   \operatorname{Tr}(\hat U^T \hat\Xi \hat U) = 1 - \frac{(\mathbf{n}\cdot\hat{\mathbf{e}}_{0,\perp})^2 + \varepsilon^2(\mathbf{n}\cdot\hat{\mathbf{e}}_{1,\perp})^2}{(1+\varepsilon^2)\,\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
   \;}
   $$
   where $\hat{\mathbf{e}}_{i,\perp} = \frac{\mathbf{e}_i - (\mathbf{e}_i\cdot\hat{\mathbf{v}})\hat{\mathbf{v}}}{|\mathbf{e}_i - (\mathbf{e}_i\cdot\hat{\mathbf{v}})\hat{\mathbf{v}}|}$.

2. In the collinear bunch approximation ($\mathbf{v} \approx \beta\hat{\mathbf{z}}$), for any crossing angle $\alpha$:
   $$
   \hat{\mathbf{e}}_{0,\perp} = \hat{\mathbf{x}},\qquad \hat{\mathbf{e}}_{1,\perp} = \hat{\mathbf{y}}
   $$
   recovering the standard dipole emission pattern with $\mathcal{P} \le 1$ everywhere.

3. The total angle-integrated cross-section is strictly invariant:
   $$
   \boxed{\;\int d\Omega \, \frac{d\sigma}{d\Omega} = \sigma_T = \frac{8\pi}{3}r_e^2\;}
   $$
   guaranteeing photon number conservation ($N_{\text{tgt}} \le N_{\text{tot}}$) across all crossing angles and ellipticities.

4. The crossing angle enters the emission through its two physical kinematic channels (DER005 §2.1 and §2.2):
   - Collision flux rate: $v_{\text{rel}} = c(1 - \mathbf{v}\cdot\hat{\mathbf{n}}_0) \approx 2c\cos^2(\alpha/2)$.
   - Doppler resonance frequency: $\omega_R \approx \frac{4\gamma^2\omega_L\cos^2(\alpha/2)}{1+\gamma^2\theta^2+\hat a}$.

---

## Verification

Verified symbolically and numerically in `verify_der012.py`:
- **Symbolic acceleration scaling:** Relativistic acceleration under tilted laser field confirmed $\dot{v}_z / \dot{v}_x \approx -\alpha / (2\gamma^2)$, suppressed by $\gamma^2$ relative to Eq. `accel`.
- **Rest-frame electric field:** $E'_z / E'_x \approx -\alpha / (2\gamma) \to 0$ verified via Lorentz transformation.
- **Orthogonality:** $\mathbf{v}\cdot\hat{\mathbf{e}}_{i,\perp} = 0$ verified identically.
- **Cross-section integration:** $\int d\Omega \, \frac{\gamma^2}{(1+\gamma^2\theta^2)^2}\mathcal{P} = \frac{2\pi}{3}$ verified numerically to relative error $< 0.07\%$.
- **Photon conservation:** Confirms $N_{\text{tgt}} / N_{\text{tot}} \le 1.0$ unconditionally for all $\alpha$ and $\varepsilon$.

---

## Used by

- Supersedes DER005 §2.3 and DER006 §2 for the crossing-angle correction in $\mathbf{u}_i\cdot\mathbf{u}_j$.
- Implemented in `gammaforge.engines.xigma.stages` (`polarization_factor_vectorized`) and `spectrum_sampler` (CuPy rawkernel).
