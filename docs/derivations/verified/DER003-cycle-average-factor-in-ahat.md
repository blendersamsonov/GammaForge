# DER003 — The cycle-average factor in `ahat`

Status: verified

## Setup

This came up while reading eq. `ahat` for the ellipticity work (DER004). It is not part of
DER004 or DER005, and it affects every scenario. The question: is the code's `ahat` the same
quantity as the paper's $\hat a$?

### The paper

Eq. `field` writes the incident field as

$$
\mathbf{E}(\varphi) = a_0\,\Re\!\left\{E(\varphi)\left(\epsilon_0\mathbf{e}_0+\epsilon_1\mathbf{e}_1\right)\right\},
\qquad \sum_i|\epsilon_i|^2 = 1 ,
$$

so eq. *Xi*'s coherence matrix $\Xi_{ij}=\epsilon_i\epsilon_j^*$ has $\operatorname{Tr}\hat\Xi\equiv 1$
identically. The text below eq. *Xi* then states
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

## Derivation

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

### The physical origin of the factor

It is not a convention mismatch — it is the polarization cycle average, and the paper is
right. $a_0$ is by definition the **normalized magnitude of the electric field**, a peak
amplitude. The cycle-averaged normalized intensity is therefore

$$
\langle a^2\rangle = C\,a_0^2,\qquad
C=\tfrac12\ \text{(linear)},\qquad C=1\ \text{(circular)}
$$

because a linearly polarized field oscillates as $\cos\varphi$ (so
$\langle\cos^2\rangle=\tfrac12$) while a circularly polarized one has constant magnitude. The
same statement read the other way: **at fixed $a_0$ circular polarization carries twice the
cycle-averaged energy density**; equivalently, at fixed pulse energy circular gives $a_0$
smaller by $\sqrt2$.

`io.laser.GaussianParaxialLaser._a0_from_density` implements the **linear** chain explicitly
($\langle u\rangle=E_0^2/8\pi$, i.e. $E_0=\sqrt{8\pi\langle u\rangle}$), so its `a0_profile`
is the linear-polarization peak amplitude and $C=\tfrac12$.

## Result

The code's `ahat` is twice the paper's $\hat a$ — the code is missing the polarization
factor $C=1/2$ entirely, not using a different but self-consistent convention.
`TrajectorySamples.ahat()` returns $a_{0,\rm peak}^2\sum r^2/\sum r$ — the
**amplitude-squared** trajectory average, with no cycle factor. `ahat` feeds the resonance
denominator in two places:

| site | expression |
|---|---|
| `validation/references/delta.py` | `s_res = gamma**2 / (1 + ahat + gamma**2 * r**2)` |
| xigma Stage 2 | inverts the same resonance per `ahat` cell |

Both therefore over-state the nonlinear red-shift by a factor of two.

## Verification

> **Resolution (author, 2026-08-10).** The paper is right and the code was missing the
> $1/2$. $a_0$ is by definition the normalized **peak** magnitude of the electric field, so
> the cycle-averaged normalized intensity is $\langle a^2\rangle = C\,a_0^2$ with $C=1/2$
> for linear polarization ($\langle\cos^2\rangle = 1/2$) and $C=1$ for circular (constant
> magnitude). `_a0_from_density` implements the linear chain explicitly, so
> `a0_profile` is the linear peak amplitude and $C=1/2$ applies.
>
> **Fixed in RES053.** `stages.ahat_from_shape` is now the single route from
> `a0_shape` to $\hat a$ and applies `io.laser.CYCLE_AVERAGE_FACTOR` $= 1/2$. The
> compensating-$2$ worry below was checked and ruled out: `RELATIVE_VELOCITY`, the
> $E = 4\hbar\omega_0 s$ convention and `KERNEL_NORMALIZATION_CONSTANT` all live in the
> photon-**count** path, which §9.1 pins absolutely (RES033), while $\hat a$ lives in the
> resonance denominator and moves photons along $s$ without changing how many there are.
> No check that constrains one is sensitive to the other, so they cannot be cancelling.
> Total yield is bit-for-bit unchanged by the fix, as that argument requires.

### Corroboration from outside both

The textbook linear-polarization ponderomotive denominator is $1 + a_0^2/2$, which is
eq. `ahat` for a flat-top envelope. $\hat a_{\text{code}}$ would make it $1 + a_0^2$ — the
*circular* value. The paper and the standard result agree with each other and not with the
code.

### Why cross-validation didn't catch it

xigma and the delta reference **share** `TrajectorySamples.ahat()`. A common-mode error
cancels in any comparison between them, so the four-way cross-check — the machinery whose
whole purpose is catching this class of mistake — is blind to it by construction. Worth
recording independently of the fix: it is a gap in the cross-validation design, not only a
bug.

The analytical engine is the leg that exposes it, because it computes $\hat a$ from the
overlap integral rather than from `TrajectorySamples`. Before the fix it applied
$\hat a=\tfrac12\langle a_0^2\rangle$, so an analytical-vs-xigma comparison of the
nonlinear red-shift differed by exactly two until the sites above were corrected.

### Provenance

Inherited, not introduced by this rebuild: the predecessor uses the same
$\hat a = a_0^2\times$`a0_shape` (`ComptonSuite/src/gammaforge/models/xigma_i/deposition.py:429`,
`particles.py:263`).

## Used by

$\hat a$ enters only as the redshift $s_{\text{res}} = \gamma^2/(1+\hat a+\gamma^2 r^2)$ and
the $(1+\hat a)^{-1}$ Jacobian, so there is no cancellation downstream — the bug was a
straight $2\times$ overestimate of the nonlinear redshift. Order of the effect on the bank
before the fix: $\sim 1\%$ on `baseline` ($\hat a$ mean $0.0097$), $\sim 5\%$ on
`near_a0_max` ($0.0486$).

**A second cost, which is why this needed deciding before DER004/DER005 code and not after.**
`retarget_ahat`'s target grid was tuned against the *pre-fix* values (RES032). Halving them
moves the bank down a bin:

| scenario | $\hat a$ mean / max before | bins | halved | bins |
|---|---|---|---|---|
| `baseline` | $0.0097 \,/\, 0.0191$ | $0\to0$ | $0.0049\,/\,0.0096$ | $0\to0$ |
| `low_a0` | $0.00097\,/\,0.0019$ | $0\to0$ | $0.00049\,/\,0.00095$ | $0\to0$ |
| `near_a0_max` | $0.0486\,/\,0.0954$ | $1\to2$ | $0.0243\,/\,0.0477$ | $0\to1$ |

Target edges at the pre-fix defaults: $0,\;0.0347,\;0.0670,\;0.0971,\dots$. `near_a0_max`
was the only scenario that resolved the redshift at all before the fix, and after
correction its mean falls into the floor bin. `ahat_decades` would need retuning as part
of the fix — the same numerical exercise RES032 already documents, not a new kind of
problem.

`ahat_from_shape` applies $C=1/2$; `a0_shape` keeps its literal meaning
$\int|E|^4/\int|E|^2$. The `ahat_decades` retune anticipated above was **measured but not
applied** — the bin-shift table is right, the resulting centroid bias against `delta` is
$\sim1\%$ and pre-existing rather than created by the fix, and changing a production
default the author tuned with stated physics reasoning is their call. Numbers and the
recommended one-parameter change ($\texttt{decades}\;1.0\to0.3$) are in RES053's last
section.

$C=\tfrac12\to1$ from linear to circular is exactly what `ellipticity` should interpolate
(DER004), and it enters in **two** places, not one: the resonance shift $\hat a$, and the
energy-to-$a_0$ conversion itself, since `_a0_from_density`'s $\sqrt{8\pi u}$ is the linear
relation. `ELLIPTICITY_IS_NOOP` is the placeholder for both. This gives DER004 a concrete,
checkable hook rather than an open question. RES054 later removed the polarization
convention from this path entirely rather than documenting it (`intensity_profile`
replaces the `a0`-then-*C* round trip) — see RES054 and `io/laser.py`.
