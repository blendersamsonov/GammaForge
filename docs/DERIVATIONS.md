# Derivations pending author review — §9.2, §9.3, and one blocking discrepancy

**Status: §0 is resolved and fixed (2026-08-10, `DECISIONS.md` D053). Nothing else in this
file is implemented.** `ELLIPTICITY_IS_NOOP` and
`EMISSION_IS_HEAD_ON` are both still `True`; `stages.py` is untouched by §1–§2 below.
That is deliberate — these are physics results derived from the paper by an agent, and
they need the author's check before they become code. This file exists so the checking has
something concrete to work against.

Notation follows `~/Work/Papers/2026/Compton-Numerics/xigma.tex`, referenced by equation
label. `GRAND_PLAN.md` §9.2/§9.3 are the items being closed out; `DECISIONS.md` D026/D033
are the §9.1 precedent for how a paper-level finding gets handled here.

---

## 0. ~~BLOCKING~~ RESOLVED AND FIXED: the code's `ahat` was twice the paper's $\hat a$

This came up while reading eq. `ahat` for the ellipticity work. It is not part of §9.2 or
§9.3, and it affects every scenario.

> **Resolution (author, 2026-08-10).** The paper is right and the code was missing the
> $1/2$. $a_0$ is by definition the normalized **peak** magnitude of the electric field, so
> the cycle-averaged normalized intensity is $\langle a^2\rangle = C\,a_0^2$ with $C=1/2$
> for linear polarization ($\langle\cos^2\rangle = 1/2$) and $C=1$ for circular (constant
> magnitude). `_a0_from_density` implements the linear chain explicitly, so
> `a0_profile` is the linear peak amplitude and $C=1/2$ applies.
>
> **Fixed in `DECISIONS.md` D053.** `stages.ahat_from_shape` is now the single route from
> `a0_shape` to $\hat a$ and applies `io.laser.CYCLE_AVERAGE_FACTOR` $= 1/2$. The
> compensating-$2$ worry below was checked and ruled out: `RELATIVE_VELOCITY`, the
> $E = 4\hbar\omega_0 s$ convention and `KERNEL_NORMALIZATION_CONSTANT` all live in the
> photon-**count** path, which §9.1 pins absolutely (D033), while $\hat a$ lives in the
> resonance denominator and moves photons along $s$ without changing how many there are.
> No check that constrains one is sensitive to the other, so they cannot be cancelling.
> Total yield is bit-for-bit unchanged by the fix, as that argument requires.

### The paper

Eq. `field` writes the incident field as

$$
\mathbf{E}(\varphi) = a_0\,\Re\!\left\{E(\varphi)\left(\epsilon_0\mathbf{e}_0+\epsilon_1\mathbf{e}_1\right)\right\},
\qquad \sum_i|\epsilon_i|^2 = 1 ,
$$

so eq. `Xi`'s coherence matrix $\Xi_{ij}=\epsilon_i\epsilon_j^*$ has $\operatorname{Tr}\hat\Xi\equiv 1$
identically. The text below eq. `Xi` then states
$\langle a^2(\varphi)\rangle = a_0^2|E(\varphi)|^2\operatorname{Tr}\hat\Xi/2$, and eq. `ahat` defines

$$
\hat a \;=\; a_0^2\,\frac{\operatorname{Tr}\hat\Xi}{2}\,
\frac{\int |E|^4\,\mathrm{d}\varphi}{\int |E|^2\,\mathrm{d}\varphi}
\;=\; \frac{a_0^2}{2}\,\frac{\int |E|^4}{\int |E|^2} .
$$

### The code

`stages.integrate_trajectories` builds `ratio` $=(a_0^{\text{local}}/a_0^{\text{peak}})^2$ from
`LaserField.a0_profile`, then `a0_shape` $=\sum\text{ratio}^2/\sum\text{ratio}$, and
`TrajectorySamples.ahat()` $= a_0^{\text{peak}\,2}\times$ `a0_shape`. So

$$
\hat a_{\text{code}} \;=\; a_0^2\,\frac{\int |E|^4}{\int |E|^2} \;=\; 2\,\hat a_{\text{paper}} .
$$

### The step that decides it

Whether `a0_profile` is the *amplitude* envelope $a_0|E|$ or something already
cycle-averaged. It is the amplitude envelope — its own docstring says the period-resolved
`field()`'s "envelope *is* `a0_profile`", and measured directly at the focus over one
carrier period $T$:

| quantity | value |
|---|---|
| `a0_profile(0)` | $0.995004$ |
| $\max\lvert\mathbf{a}\rvert$ over $T$ | $0.995004$ &nbsp; (ratio to envelope $1.0000$) |
| $\langle\lvert\mathbf{a}\rvert^2\rangle$ over $T$ | $0.4999\times\text{envelope}^2$ |

So `ratio` $=|E|^2$ exactly, and the missing $\operatorname{Tr}\hat\Xi/2 = 1/2$ is not
hiding in the profile.

### Corroboration from outside both

The textbook linear-polarization ponderomotive denominator is $1 + a_0^2/2$, which is
eq. `ahat` for a flat-top envelope. $\hat a_{\text{code}}$ would make it $1 + a_0^2$ — the
*circular* value. The paper and the standard result agree with each other and not with the
code.

### Provenance

Inherited, not introduced by this rebuild: the predecessor uses the same
$\hat a = a_0^2\times$`a0_shape` (`ComptonSuite/src/gammaforge/models/xigma_i/deposition.py:429`,
`particles.py:263`).

### What it costs

$\hat a$ enters only as the redshift $s_{\text{res}} = \gamma^2/(1+\hat a+\gamma^2 r^2)$ and
the $(1+\hat a)^{-1}$ Jacobian, so there is no cancellation downstream — it is a straight
$2\times$ overestimate of the nonlinear redshift. Order of the effect on the bank: $\sim 1\%$
on `baseline` ($\hat a$ mean $0.0097$), $\sim 5\%$ on `near_a0_max` ($0.0486$).

**A second cost, which is why this needs deciding before §9.2/§9.3 code and not after.**
`retarget_ahat`'s target grid was tuned last session against the *current* values (D032).
Halving them moves the bank down a bin:

| scenario | $\hat a$ mean / max now | bins | halved | bins |
|---|---|---|---|---|
| `baseline` | $0.0097 \,/\, 0.0191$ | $0\to0$ | $0.0049\,/\,0.0096$ | $0\to0$ |
| `low_a0` | $0.00097\,/\,0.0019$ | $0\to0$ | $0.00049\,/\,0.00095$ | $0\to0$ |
| `near_a0_max` | $0.0486\,/\,0.0954$ | $1\to2$ | $0.0243\,/\,0.0477$ | $0\to1$ |

Target edges at the current defaults: $0,\;0.0347,\;0.0670,\;0.0971,\dots$. `near_a0_max` is
the only scenario that currently resolves the redshift at all, and after the correction its
mean falls into the floor bin. `ahat_decades` would need retuning as part of the fix — the
same numerical exercise D032 already documents, not a new kind of problem.

### What was actually done

`ahat_from_shape` applies $C=1/2$; `a0_shape` keeps its literal meaning
$\int|E|^4/\int|E|^2$. The `ahat_decades` retune anticipated above was **measured but not
applied** — the bin-shift table is right, the resulting centroid bias against `delta` is
$\sim1\%$ and pre-existing rather than created by the fix, and changing a production default
the author tuned with stated physics reasoning is their call. Numbers and the recommended
one-parameter change ($\texttt{decades}\;1.0\to0.3$) are in D053's last section.

---

## 1. §9.2 — ellipticity

`GRAND_PLAN.md` §9.2 records "no formula exists" for the energy→$a_0$ relation with
ellipticity. That turns out to be the right observation with the wrong conclusion: the
paper's conventions *determine* the answer, and the answer is that the existing chain is
already correct. What ellipticity actually changes is somewhere else entirely.

### 1.1 The energy→$a_0$ chain needs no ellipticity correction — it is an invariance, not a gap

Because $\sum_i|\epsilon_i|^2 = 1$ forces $\operatorname{Tr}\hat\Xi\equiv 1$, the relation
below eq. `Xi` reads $\langle a^2\rangle = a_0^2|E|^2/2$ *for every polarization state*.
Inverting it at the pulse peak against the cycle-averaged intensity, with $U$ the energy
density:

$$
\langle E^2\rangle = \frac{4\pi I}{c} = 4\pi U ,
$$

$$
a_0^2 \;=\; 2\langle a^2\rangle \;=\; 2\left(\frac{e}{m_e c\,\omega}\right)^{\!2}\langle E^2\rangle
\;=\; 8\pi U\left(\frac{e}{m_e c\,\omega}\right)^{\!2} ,
$$

$$
\boxed{\;a_0 = \frac{e}{m_e c\,\omega}\sqrt{8\pi U}\;}
$$

which is exactly what `GaussianParaxialLaser._a0_from_density` computes. So the current
identity/no-op is not a placeholder standing in for a missing factor — it is the correct
result, and $a_0$ at fixed pulse energy is genuinely independent of ellipticity in this
paper's convention.

The familiar $(1+\varepsilon^2)/2$-type factor from other treatments comes from defining
$a_0$ through the *peak instantaneous* $|a|$ rather than through the cycle average. Under
that definition a circular pulse of the same energy has $|a|$ constant at
$a_0^{\text{lin}}/\sqrt2$ while a linear one peaks at $a_0^{\text{lin}}$; the paper
sidesteps the whole question by normalizing $\hat\Xi$.

#### The author picked the peak convention — and it does not change the conclusion

D053's resolution defines $a_0$ as the **peak instantaneous** magnitude, i.e. the second of
the two conventions above: $\langle a^2\rangle = C a_0^2$ with $C = 1/2$ linear, $1$
circular. That is the opposite bookkeeping from the paper's normalized $\hat\Xi$, so it is
worth writing out that the two agree on every physical number at fixed pulse energy.

Under the peak convention $C$ enters in **two** places, and they cancel. Inverting
$\langle a^2\rangle = C a_0^2$ against $\langle E^2\rangle = 4\pi U$,

$$
a_0^2 = \frac{1}{C}\left(\frac{e}{m_ec\,\omega}\right)^{\!2} 4\pi U
\qquad\Longrightarrow\qquad
\boxed{\;a_0 = \frac{e}{m_ec\,\omega}\sqrt{\frac{4\pi U}{C}}\;}
$$

— which is the code's $\sqrt{8\pi U}$ at $C=1/2$, and smaller by $\sqrt2$ at $C=1$, exactly
as the resolution states. Substituting it into $\hat a = C\,a_0^2 \int|E|^4/\int|E|^2$:

$$
\hat a \;=\; C\cdot\frac{1}{C}\left(\frac{e}{m_ec\,\omega}\right)^{\!2}4\pi U\cdot
\frac{\int|E|^4}{\int|E|^2}
\;=\;\left(\frac{e}{m_ec\,\omega}\right)^{\!2}4\pi U\,\frac{\int|E|^4}{\int|E|^2},
$$

with no $C$ left. **At fixed pulse energy $\hat a$ is ellipticity-independent under either
convention** — as it must be, since it is a ratio of physical intensity moments and the
conventions differ only in what number they call $a_0$. The same cancellation protects the
photon count: `photon_density_scale` inverts `_a0_from_density` exactly, so if one changes
the other changes with it and `luminosity` does not move.

**Consequence for the code:** implementing §9.2 in full would change `a0_peak` — a reported
quantity — and change nothing else through this route. The handover framing "$C$ enters in
two places" is right; worth adding is that those two places *cancel for $\hat a$*, so §9.2's
only physical effect on the spectrum is §1.2's kernel polarization factor. Two pieces of
text also become wrong rather than merely incomplete —

- `_a0_from_density`'s docstring calls itself "the standard **linear-polarization** chain."
  Correct as written, but the reason is a convention choice, not a physical restriction:
  the elliptical case differs from it only by $\sqrt{1/2C}$, and the pulse energy it is
  inverting is polarization-blind.
- `ELLIPTICITY_IS_NOOP`'s framing ("carried but changes nothing", flip to `False` when the
  derivation lands) describes an unimplemented placeholder. Post-D053 it guards two known
  factors that cancel in $\hat a$ and do not cancel in `a0_peak` — with the `validate()`
  warning narrowed, not deleted, because of §1.2.

### 1.2 Ellipticity *does* enter, in the polarization factor of the kernel

The place $\hat\Xi$ is actually consumed is $\hat U^{T}\hat\Xi\hat U$ in eq. `xsec` /
eq. `Fmatrix`. Eq. `trace` gives

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i,j}\Xi_{ij}\,(\mathbf{u}_i\cdot\mathbf{u}_j) ,
$$

and eq. `umod` supplies only the diagonal. Extending eq. `umod` to the off-diagonal, from
eq. `udef` under the same neglect of $\mathbf{v}\cdot\mathbf{e}_i$ (justified head-on — see
§2.3 for where that fails):

$$
\mathbf{u}_i\cdot\mathbf{u}_j
= \delta_{ij} - \frac{(\mathbf{n}\cdot\mathbf{e}_i)(\mathbf{n}\cdot\mathbf{e}_j)}{\gamma^2(1-\mathbf{v}\cdot\mathbf{n})^2}
= \delta_{ij} - \frac{4\gamma^2\theta^2\cos\psi_i\cos\psi_j}{(1+\gamma^2\theta^2)^2},
$$

using $\mathbf{n}\cdot\mathbf{e}_i = \theta\cos\psi_i$ and
$1-\mathbf{v}\cdot\mathbf{n} = (1+\gamma^2\theta^2)/2\gamma^2$. The diagonal reproduces
eq. `umod` exactly. With $\mathbf{e}_0\perp\mathbf{e}_1$ we have $\psi_1 = \psi_0-\pi/2$, so
writing $\psi\equiv\psi_0$:

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= 1 - \frac{4\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}
\Bigl[\Xi_{00}\cos^2\psi + \Xi_{11}\sin^2\psi + 2\,\Re(\Xi_{01})\cos\psi\sin\psi\Bigr].
$$

Parametrizing the scalar `ellipticity` $\varepsilon\in[0,1]$ ($0$ = linear, $1$ = circular)
as the ratio of the ellipse's minor to major axis, with the major axis along $\mathbf{e}_0$:

$$
\epsilon_0 = \frac{1}{\sqrt{1+\varepsilon^2}},\qquad
\epsilon_1 = \frac{i\,\varepsilon}{\sqrt{1+\varepsilon^2}},
$$

$$
\Xi_{00} = \frac{1}{1+\varepsilon^2},\qquad
\Xi_{11} = \frac{\varepsilon^2}{1+\varepsilon^2},\qquad
\Xi_{01} = \frac{-i\,\varepsilon}{1+\varepsilon^2},\qquad
\operatorname{Tr}\hat\Xi = 1\ \checkmark
$$

The cross term vanishes because $\Re(\Xi_{01}) = 0$, and that is not luck: $\Xi_{01}$ is
purely imaginary exactly when the two components are in quadrature, which *is* the
statement that the ellipse axes are $\mathbf{e}_0/\mathbf{e}_1$. `psi_pol` already means
"the major axis's azimuth", so the convention the code carries is the one that makes the
cancellation hold. A general relative phase would leave a $\sin\psi\cos\psi$ term and
describe an ellipse tilted with respect to `psi_pol` — a second angle the schema does not
have and, on this reading, does not need.

**Result:**

$$
\boxed{\;
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= 1 - \frac{4\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}\cdot
\frac{\cos^2\psi + \varepsilon^2\sin^2\psi}{1+\varepsilon^2}
\;}
$$

- $\varepsilon = 0 \;\Rightarrow\; 1 - \dfrac{4\gamma^2\theta^2\cos^2\psi}{(1+\gamma^2\theta^2)^2}$
  — eq. `linpol`, and exactly what `spectrum_from_table` / `delta.resonance_spectrum`
  compute today. ✓
- $\varepsilon = 1 \;\Rightarrow\; 1 - \dfrac{2\gamma^2\theta^2}{(1+\gamma^2\theta^2)^2}$
  — azimuth-independent, as circular polarization must be. ✓

**Numerically checked** against the direct sum $\sum_{ij}\Xi_{ij}(\mathbf{u}_i\cdot\mathbf{u}_j)$
with $\mathbf{u}_i$ built straight from eq. `udef`, over 4000 random
$(\gamma\in[10,3000],\ \theta\in[0,4/\gamma],\ \psi,\ \varepsilon)$ in head-on geometry.
Keeping $1-\mathbf{v}\cdot\mathbf{n}$ exact: worst error $1.5\times10^{-9}$ (rounding).
Applying the paper's own $1-\mathbf{v}\cdot\mathbf{n}\to(1+\gamma^2\theta^2)/2\gamma^2$:
worst error $1.2\times10^{-2}$ — that residual is eq. `smallangle`'s documented
$O(\theta^2)$, not this derivation, and it is present in the existing linear-only kernel too.

**Consequence for the code, if the author agrees:** a one-line generalization in each of the
two kernels,

$$
\cos^2\psi \;\longrightarrow\; \frac{\cos^2\psi + \varepsilon^2\sin^2\psi}{1+\varepsilon^2},
$$

plus threading `ellipticity` through to them the way `psi_pol` already is. Note this changes
the *angular* distribution only; the yield integrated over azimuth is unaffected, since
$\langle\cos^2\psi\rangle = \langle\sin^2\psi\rangle = 1/2$.

---

## 2. §9.3 — crossing angle

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
$c(1+\beta)\to 2c$, which is `stages.RELATIVE_VELOCITY = 2.0`. With the geometry `io.laser`
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

The first two terms are eq. `umod` (and §1.2's off-diagonal extension); the third is what
the crossing angle switches on.

**Check, $\alpha = 90^\circ$ exactly.** Take $\mathbf{n} = \hat{\mathbf{z}}$,
$\mathbf{v} = \beta\hat{\mathbf{z}}$, and the laser incident from $+\hat{\mathbf{x}}$ so that
$\mathbf{e}_0 = -\hat{\mathbf{z}}$ — the polarization now lies *along* the observation
direction, and a dipole must radiate nothing there. Directly from eq. `udef`:

$$
\mathbf{u}_0 = \frac{(\mathbf{n}-\mathbf{v})(\mathbf{n}\cdot\mathbf{e}_0)}{1-\mathbf{v}\cdot\mathbf{n}} - \mathbf{e}_0
= -\hat{\mathbf{z}}\,\frac{1-\beta}{1-\beta} + \hat{\mathbf{z}} = 0 \quad\checkmark
$$

From the formula, with $a_0 = -1$, $b_0 = -\beta$, and $\gamma^2(1-\beta)^2 = (1-\beta)/(1+\beta)$:

$$
|\mathbf{u}_0|^2 = 1 - \frac{1+\beta}{1-\beta} + \frac{2\beta}{1-\beta}
= \frac{(1-\beta) - (1+\beta) + 2\beta}{1-\beta} = 0 \quad\checkmark
$$

Eq. `umod` alone would give $1 - (1+\beta)/(1-\beta) \approx -4\gamma^2$ — negative, which is
the signature of the dropped term rather than a small error. (I first ran this check with
$1-\beta\approx 1/2\gamma^2$ substituted inconsistently and got $-1$; only the exact algebra
above gives $0$. Worth repeating exactly if you check it.)

**Numerically checked** against eq. `udef` over 2000 random geometries — independent
electron direction, observation direction and laser incidence, $\gamma\in[3,3000]$, all
three of $\mathbf{u}_0\cdot\mathbf{u}_0$, $\mathbf{u}_0\cdot\mathbf{u}_1$,
$\mathbf{u}_1\cdot\mathbf{u}_1$ — worst relative error $1.2\times10^{-12}$. This tests the
algebra at arbitrary crossing angle, not just the $\alpha = 90^\circ$ limit above. It does
**not** test the physics: it confirms the expression equals eq. `udef`, and eq. `udef` is
the paper's.

### 2.4 What is still genuinely open

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

---

## 3. Summary of what would change in code

Nothing in this table is done. Ordered by how much the author needs to weigh in.

| # | Change | Where | Blocked on |
|---|---|---|---|
| 0 | `a0_shape` gains a factor $1/2$; `ahat_decades` retuned | `stages.py`, `schema.py` | **author's confirmation that no compensating $2$ exists elsewhere** |
| 1.1 | Docstrings only — the ellipticity-independence of energy→$a_0$ is derived, not missing | `laser.py` | author agrees with §1.1 |
| 1.2 | $\cos^2\psi \to (\cos^2\psi + \varepsilon^2\sin^2\psi)/(1+\varepsilon^2)$; thread `ellipticity` to the kernels | `stages.py`, `delta.py`, `collision.py` | author agrees with §1.2 |
| 2.1 | `RELATIVE_VELOCITY` becomes $1 + \beta\cos\theta_{xz}\cos\theta_{yz}$ | `stages.py` | author agrees with §2.1 |
| 2.2 | `photon_energy` gains $\cos^2(\alpha/2)$ | `collision.py` | author agrees with §2.2 |
| 2.3 | $\mathbf{u}_i\cdot\mathbf{u}_j$ gains the $\mathbf{v}\cdot\mathbf{e}_i$ term; kernels take the rotated $\mathbf{e}_0,\mathbf{e}_1$ | `stages.py`, `delta.py` | author agrees **and** an independent check exists (§2.4) |

`ELLIPTICITY_IS_NOOP` can only go `False` after 1.2; `EMISSION_IS_HEAD_ON` only after all of
2.1–2.3, since a partial crossing-angle implementation is exactly the half-applied state
D034 exists to warn about.
