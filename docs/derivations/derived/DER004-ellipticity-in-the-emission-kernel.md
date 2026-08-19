# DER004 — Ellipticity in the emission kernel's polarization factor

Status: derived

## Setup

`GRAND_PLAN.md` §9.2 records "no formula exists" for the energy→$a_0$ relation with
ellipticity. That turns out to be the right observation with the wrong conclusion: the
paper's conventions *determine* the answer, and the answer is that the existing chain is
already correct. What ellipticity actually changes is somewhere else entirely.

### 1.1 The energy→$a_0$ chain needs no ellipticity correction — it is an invariance, not a gap

Because $\sum_i|\epsilon_i|^2 = 1$ forces $\operatorname{Tr}\hat\Xi\equiv 1$, the relation
below eq. *Xi* reads $\langle a^2\rangle = a_0^2|E|^2/2$ *for every polarization state*.
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

RES053's resolution defines $a_0$ as the **peak instantaneous** magnitude, i.e. the second of
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

Two pieces of text also become wrong rather than merely incomplete once this is accepted —

- `_a0_from_density`'s docstring calls itself "the standard **linear-polarization** chain."
  Correct as written, but the reason is a convention choice, not a physical restriction:
  the elliptical case differs from it only by $\sqrt{1/2C}$, and the pulse energy it is
  inverting is polarization-blind.
- `ELLIPTICITY_IS_NOOP`'s framing ("carried but changes nothing", flip to `False` when the
  derivation lands) describes an unimplemented placeholder. Post-RES053 it guards two known
  factors that cancel in $\hat a$ and do not cancel in `a0_peak` — with the `validate()`
  warning narrowed, not deleted, because of §1.2.

### 1.2 Ellipticity *does* enter, in the polarization factor of the kernel

The place $\hat\Xi$ is actually consumed is $\hat U^{T}\hat\Xi\hat U$ in eq. `xsec` /
eq. *Fmatrix*. Eq. `trace` gives

$$
\operatorname{Tr}\!\left(\hat U^{T}\hat\Xi\hat U\right)
= \sum_{i,j}\Xi_{ij}\,(\mathbf{u}_i\cdot\mathbf{u}_j) ,
$$

and eq. `umod` supplies only the diagonal. Extending eq. `umod` to the off-diagonal, from
eq. `udef` under the same neglect of $\mathbf{v}\cdot\mathbf{e}_i$ (justified head-on — see
DER005 §2.3 for where that fails):

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

## Result

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

Note this changes the *angular* distribution only; the yield integrated over azimuth is
unaffected, since $\langle\cos^2\psi\rangle = \langle\sin^2\psi\rangle = 1/2$ — consistent
with §1.1's finding that the total photon count never depends on ellipticity.

## Verification

**Numerically checked** against the direct sum $\sum_{ij}\Xi_{ij}(\mathbf{u}_i\cdot\mathbf{u}_j)$
with $\mathbf{u}_i$ built straight from eq. `udef`, over 4000 random
$(\gamma\in[10,3000],\ \theta\in[0,4/\gamma],\ \psi,\ \varepsilon)$ in head-on geometry.
Keeping $1-\mathbf{v}\cdot\mathbf{n}$ exact: worst error $1.5\times10^{-9}$ (rounding).
Applying the paper's own $1-\mathbf{v}\cdot\mathbf{n}\to(1+\gamma^2\theta^2)/2\gamma^2$:
worst error $1.2\times10^{-2}$ — that residual is eq. `smallangle`'s documented
$O(\theta^2)$, not this derivation, and it is present in the existing linear-only kernel too.

This checks the algebra against an independent from-scratch evaluation of the same
formula — it does not check against the production kernels, since `ellipticity` isn't
threaded into `stages.py`/`delta.py` yet (`ELLIPTICITY_IS_NOOP` stays `True`), and it has
not yet been reviewed by the author. Both are what would move this file to `validated/`
and then `verified/`.

## Used by

Implementing §9.2 in full would change `a0_peak` — a reported quantity — and change
nothing else through that route: the two places $C$ enters cancel for $\hat a$ (§1.1), so
§9.2's only physical effect on the spectrum is §1.2's kernel polarization factor.

**If the author agrees with §1.1:** no code changes — the energy→$a_0$ chain
(`laser.py`'s `_a0_from_density`) is already correct, and this closes the open item as a
documentation fix: `_a0_from_density`'s docstring and `ELLIPTICITY_IS_NOOP`'s `validate()`
warning both need rewording (see §1.1's "two pieces of text" above), not a formula change.

**If the author agrees with §1.2:** a one-line generalization in each of the two kernels,

$$
\cos^2\psi \;\longrightarrow\; \frac{\cos^2\psi + \varepsilon^2\sin^2\psi}{1+\varepsilon^2},
$$

in `stages.py`, `delta.py`, and `collision.py`, plus threading `ellipticity` through to
them the way `psi_pol` already is. `ELLIPTICITY_IS_NOOP` can only go `False` after this
lands — see RES034 for why a partial application is specifically disallowed.
