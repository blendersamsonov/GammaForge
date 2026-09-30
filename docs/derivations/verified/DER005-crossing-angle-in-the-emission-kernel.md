# DER005 — Crossing angle in the emission kernel

Status: verified

Radiation-basis revision: DER012 replaces the unprojected polarization construction
in §2.3. Flux and nominal Doppler results retain their stated approximations (RES078).
DER013/RES082 extend the nominal-axis flux and resonance to per-electron directions at
beta=1, with the corresponding table Jacobian and GPU support bounds.

## Setup

`GRAND_PLAN.md` §9.3 says the paper's angular derivation "is built for near-backscattering
geometry, accurate to $O(\theta^2)$ around the collinear axis; the paper warns against
extending it without revisiting the geometry."

**That warning is about something else.** The passage it refers to (`sec:validity4`, final
paragraph) says: "the small-angle reduction (eq. `smallangle`) presumes near-backscattering,
with accuracy $O(\theta^2)$. It is amply satisfied within the collimation apertures of
interest but should not be extended to **large-angle observation** without revisiting the
geometry." Eq. `smallangle` is

$$
\theta = \sqrt{(n_y-\theta_y)^2 + (n_z-\theta_z)^2}
$$

— the angle between the *electron's* velocity and the *observation* direction. It constrains
how far off-axis you may observe and how divergent the bunch may be. It says nothing about
the laser's direction of incidence, which does not appear in it.

So the crossing angle is a narrower problem than §9.3 assumes. Three places carry it, and
the paper leaves two of them already general.

### 2.1 The relative-velocity factor (already general in the paper)

Eq. `lumfun` defines
$\mathcal{L}(\zeta) = \int v_{\text{rel}}\, n_{\text{ph}}(t,\mathbf{r}(t;\zeta))\,\mathrm{d}t$
and never specializes $v_{\text{rel}}$. For an electron of velocity $\mathbf{v}$ meeting
photons propagating along $\hat{\mathbf{n}}_0$, the flux factor is

$$
v_{\text{rel}} = c\left(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0\right).
$$

Head-on ($\hat{\mathbf{n}}_0 = -\hat{\mathbf{z}}$, $\mathbf{v} = \beta\hat{\mathbf{z}}$) gives
$c(1+\beta)\to 2c$, which is `stages.relative_velocity(1.0)`. With the geometry `io.laser`
already pins, $R = R_y(\theta_{xz})R_x(\theta_{yz})$ applied to $-\hat{\mathbf{z}}$:

$$
\hat{\mathbf{n}}_0 = \bigl(-\sin\theta_{xz}\cos\theta_{yz},\ \sin\theta_{yz},\ -\cos\theta_{xz}\cos\theta_{yz}\bigr),
$$

$$
\boxed{\;1-\mathbf{v}\cdot\hat{\mathbf{n}}_0 = 1 + \beta\cos\theta_{xz}\cos\theta_{yz}
\;\xrightarrow[\ \beta\to1\ ]{}\; 2\cos^2\!\frac{\alpha}{2},
\qquad \cos\alpha \equiv \cos\theta_{xz}\cos\theta_{yz}\;}
$$

### 2.2 The resonance frequency (already general in the paper)

Eq. `wR` is

$$
\omega_R = \omega_L\,\frac{2\gamma^2\left(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0\right)}{1+\gamma^2\theta^2+\hat a}
$$

— the same factor, written out. It is only eq. `wRgamma` that substitutes
$1-\mathbf{v}\cdot\hat{\mathbf{n}}_0\to 2$ to get the familiar $4\omega_L\gamma^2/(\cdots)$. So

$$
\omega_R = \frac{4\omega_L\gamma^2\cos^2(\alpha/2)}{1+\gamma^2\theta^2+\hat a},
$$

and the kinematic edge of eq. `kinematic` moves with it:
$\omega\theta^2 < 4\omega_L\cos^2(\alpha/2)$.

Note both §2.1 and §2.2 carry the *same* factor $(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0)$,
which is a useful check on any implementation: the crossing angle enters the yield and the
photon energy through one quantity, not two independent ones.

**One convention consequence for the code.** $s$ is defined by $E = 4\hbar\omega_0 s$ with
$s_{\text{res}} = \gamma^2/(1+\hat a+\gamma^2\theta^2)$, i.e. the head-on $4\gamma^2$ is baked
into the *conversion*, not the resonance. The cleanest placement is to keep $s_{\text{res}}$
as it is and carry $\cos^2(\alpha/2)$ in the energy conversion (`Collision`'s
`photon_energy`), so the table axis keeps its meaning and one factor moves in one place.
That is a code-shape opinion, not physics.

### 2.3 The polarization structure — this is the genuine gap

Eq. `umod` derives $|\mathbf{u}_i|^2$ "on neglecting $\mathbf{v}\cdot\mathbf{e}_i$, which is
of order $a_0/\gamma$." That estimate holds when $\mathbf{e}_i\perp\mathbf{v}$: head-on,
$\mathbf{e}_0,\mathbf{e}_1$ span the plane transverse to $\hat{\mathbf{z}}$ and the electron
moves along $\hat{\mathbf{z}}$, so $\mathbf{v}\cdot\mathbf{e}_i$ is only the bunch's own
small divergence times $\beta$. A crossing angle rotates $\mathbf{e}_0,\mathbf{e}_1$ out of
that plane and $\mathbf{v}\cdot\mathbf{e}_i$ becomes $O(\sin\alpha)$ — not small, and not
$a_0/\gamma$-suppressed. Everything else about eq. `udef` survives; it is this one neglect
that breaks.

Redoing $\mathbf{u}_i\cdot\mathbf{u}_j$ from eq. `udef` without it. Write
$a_i \equiv \mathbf{n}\cdot\mathbf{e}_i$, $b_i \equiv \mathbf{v}\cdot\mathbf{e}_i$, and use
$|\mathbf{n}-\mathbf{v}|^2 = 2(1-\mathbf{v}\cdot\mathbf{n}) - 1/\gamma^2$ and
$\mathbf{e}_i\cdot\mathbf{e}_j = \delta_{ij}$:

$$
\begin{aligned}
\mathbf{u}_i\cdot\mathbf{u}_j
&= \frac{a_i a_j\,|\mathbf{n}-\mathbf{v}|^2}{(1-\mathbf{v}\cdot\mathbf{n})^2}
 - \frac{a_i(a_j-b_j) + a_j(a_i-b_i)}{1-\mathbf{v}\cdot\mathbf{n}} + \delta_{ij}\\[4pt]
&= a_i a_j\left[\frac{2}{1-\mathbf{v}\cdot\mathbf{n}} - \frac{1}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}\right]
 - \frac{2a_i a_j - a_i b_j - a_j b_i}{1-\mathbf{v}\cdot\mathbf{n}} + \delta_{ij}
\end{aligned}
$$

giving

$$
\boxed{\;
\mathbf{u}_i\cdot\mathbf{u}_j
= \delta_{ij}
- \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{n}\cdot\mathbf{e}_j)}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
+ \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{v}\cdot\mathbf{e}_j)
      + (\mathbf{n}\cdot\mathbf{e}_j)(\mathbf{v}\cdot\mathbf{e}_i)}{1-\mathbf{v}\cdot\mathbf{n}}
\;}
$$

The first two terms are eq. `umod` (and DER004 §1.2's off-diagonal extension); the third is
what the crossing angle switches on.

## Result

Three boxed formulas, and per RES034 they must land together, not partially: §2.1's
relative-velocity factor $1-\mathbf{v}\cdot\hat{\mathbf{n}}_0$, §2.2's resonance frequency
$\omega_R$ (the same factor, carried into the energy conversion), and §2.3's polarization
structure $\mathbf{u}_i\cdot\mathbf{u}_j$ (eq. `umod` plus the crossing-angle term the
neglected $\mathbf{v}\cdot\mathbf{e}_i$ was hiding).

## Verification

**Symbolically verified with sympy** (`verify_der005.py`):

- **Part 1 (Relative velocity):** $1-\mathbf{v}\cdot\hat{\mathbf{n}}_0 = 1 + \beta\cos\theta_{xz}\cos\theta_{yz}$ verified exactly (difference = 0).
- **Part 2 (Resonance frequency):** $\omega_R = 4\omega_L\gamma^2\cos^2(\alpha/2)/(1+\gamma^2\theta^2+\hat a)$ verified using $1+\cos\alpha = 2\cos^2(\alpha/2)$.
- **Part 3 (Polarization structure):** The boxed formula for $\mathbf{u}_i\cdot\mathbf{u}_j$ matches direct computation from $\mathbf{u}_i = (\mathbf{n}-\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_i)/(1-\mathbf{v}\cdot\mathbf{n}) - \mathbf{e}_i$ exactly (differences = 0 for all three components).
- **Head-on limit** ($\theta_{xz}=0, \theta_{yz}=0$) recovers DER004's eq. `umod` exactly (verified via double-angle identities).
- **$\alpha = 90^\circ$ dipole null:** With $\mathbf{e}_0 = -\hat{\mathbf{z}} \parallel \mathbf{n}$, the boxed formula gives $|\mathbf{u}_0|^2 = 0$ exactly; eq. `umod` alone gives negative (unphysical) result.

**Numerically checked** against eq. `udef` over 2000 random geometries — independent
electron direction, observation direction and laser incidence, $\gamma\in[3,3000]$, all
three of $\mathbf{u}_0\cdot\mathbf{u}_0$, $\mathbf{u}_0\cdot\mathbf{u}_1$,
$\mathbf{u}_1\cdot\mathbf{u}_1$ — worst relative error $1.2\times10^{-12}$. This tests the
algebra at arbitrary crossing angle, not just the $\alpha = 90^\circ$ limit above. It does
**not** test the physics: it confirms the expression equals eq. `udef`, and eq. `udef` is
the paper's.

### What is still genuinely open

- **Nothing above is validated against an independent calculation.** The plan's own answer
  for this is kascade's arbitrary-angle MC at intermediate angles (§9.3), which does not
  exist yet. Until it does, §2.3 is a derivation with one exact-limit check, not a verified
  result.
- **How far in $\alpha$ the $O(\theta^2)$ reduction survives** is not addressed by any of
  this. Eq. `smallangle` still assumes electron and observation directions cluster about a
  common axis. That holds for a tilted *laser* with an on-axis bunch, but the paper never
  states a bound in $\alpha$, and I have not derived one.
- **Whether `psi_focus`/`psi_pol` retain their meaning** once $\mathbf{e}_0,\mathbf{e}_1$ are
  tilted. §2.2 of the paper pins them as head-on-frame quantities carried through $R$, which
  is self-consistent, but the $\mathbf{n}\cdot\mathbf{e}_i$ and $\mathbf{v}\cdot\mathbf{e}_i$
  above must then be evaluated with the rotated vectors — mechanical, but it is where a sign
  or an axis convention would hide.
- **$\hat a$ is unaffected**, which is worth stating because it is easy to assume otherwise:
  eq. `ahattraj` is a ratio of two integrals over the same trajectory,

  $$
  \hat a(\zeta) = \frac{\operatorname{Tr}\hat\Xi}{2}\,
  \frac{\int\left[a^2(t;\zeta)\right]^2\mathrm{d}t}{\int a^2(t;\zeta)\,\mathrm{d}t},
  $$

  so the change of variable $\varphi = \omega_L(1-\mathbf{v}\cdot\hat{\mathbf{n}}_0)t$
  (eq. `accel`) cancels between numerator and denominator. Given the sampled $a^2(t)$,
  $\hat a$ does not know the crossing angle.

## Used by

Implemented together per RES034/RES060:
- `relative_velocity()` carries $1+\beta\cos\theta_{xz}\cos\theta_{yz}$ in `stages.py` (§2.1)
- `photon_energy` carries $\cos^2(\alpha/2)$ in `collision.py` (§2.2)
- $\mathbf{u}_i\cdot\mathbf{u}_j$ carries the $\mathbf{v}\cdot\mathbf{e}_i$ terms in `stages.py` and `delta.py` with rotated $\mathbf{e}_0, \mathbf{e}_1$ and per-particle lab-frame velocity $\mathbf{v}_e$ (RES060, DER006)
- `EMISSION_IS_HEAD_ON` is `False`.

As noted in §Verification, this implements the manuscript's lab-frame formula; independent arbitrary-angle emission physics validation remains an open item in `PROGRESS.md`.

## Amendments

> **2026-09-23 — `ahat` remains unchanged, but its resonance coefficient does not.** The
> trajectory definition of $\hat a$ is still independent of the phase-to-time change of
> variable described above. DER014 adds the distinct incidence multiplier
> $(1-\mathbf e\cdot\mathbf n_0)/2$ when that unchanged $\hat a$ enters the nonlinear
> resonance denominator.

> **2026-09-28 — Nonlinear incidence correction superseded.** DER015 replaces DER014's
> electron-direction multiplier with the exact observer-dependent ratio $Q$ in Stage 2.
> The trajectory definition of raw $\hat a$ remains unchanged.

> **2026-09-30 — The named route to an independent check is retired.** "What is still
> genuinely open" above still holds — nothing here is validated against an independent
> emission calculation — but its first bullet names kascade as the plan's answer, and
> kascade is no longer a maintained engine (RES059 archived, RES095). The gap is therefore
> not a pending port: an independent leg would have to be built anew. Nothing in §2.1–2.3
> or in the $\hat a$ trajectory definition changes with this.
