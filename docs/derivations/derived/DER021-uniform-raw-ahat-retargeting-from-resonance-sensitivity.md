# DER021 — Uniform raw-ahat retargeting from resonance sensitivity

Status: derived

## Setup

GammaForge's xigma engine separates the nonlinear trajectory reduction from the
observer-dependent spectral query. Stage 1 deposits a peak-intensity-independent shape
coordinate. Stage 1.5 retargets that coordinate to the physical raw nonlinear line-centre
strength $\hat a$ for one requested peak intensity. Stage 2 then evaluates the
observer-dependent nonlinear factor $Q$ from DER015.

The current reduced first-harmonic resonance is

$$
s_R =
\frac{D\bar C\,\gamma^2}
     {1+Q\hat a+\gamma^2r^2},
$$

where $D$ is the direction-dependent Doppler factor from DER013, $\bar C$ is the
carrier-frequency trajectory moment from DER016,

$$
Q =
\frac{1-\mathbf n\cdot\mathbf n_0}
     {1-\mathbf e\cdot\mathbf n_0},
$$

and $r$ is the small electron-observer angular separation. Define

$$
K=D\bar C,\qquad
B=1+Q\hat a+\gamma^2r^2.
$$

The question is how the observer-independent Stage-1.5 table should discretize the raw
$\hat a$ coordinate. RES032 introduced a target grid that is logarithmic in the distance
from $\hat a_{\max}$, deliberately concentrating absolute resolution near
$\hat a_{\max}$. The motivation was that the nonlinear redshift itself is largest there.

The quantity relevant to a numerical grid, however, is not the size of the correction but
the sensitivity of the queried resonance and inverse resonance to an error in the table
coordinate. This derivation evaluates that sensitivity.

This is a numerical-coordinate result built on the settled resonance physics. It does not
change the definition of $\hat a$, $D$, $Q$, $\bar C$, the Stage-0 trajectory moments, or
the physical emission model.

## Sensitivity of the forward resonance

At fixed $\gamma$, electron direction, observation direction, and carrier factor,

$$
s_R(\hat a)=\frac{K\gamma^2}{B}.
$$

Differentiating gives

$$
\frac{\partial s_R}{\partial\hat a}
=
-\frac{K\gamma^2Q}{B^2},
$$

or, more directly for relative spectral error,

$$
\boxed{
\frac{\partial\ln s_R}{\partial\hat a}
=
-\frac{Q}{1+Q\hat a+\gamma^2r^2}
}.
$$

For a table bin represented by an evaluation point displaced by $\delta\hat a$ from the
true value,

$$
\frac{\delta s_R}{s_R}
=
-\frac{Q\,\delta\hat a}
       {B+Q\,\delta\hat a},
$$

and therefore, for a sufficiently resolved bin,

$$
\frac{\delta s_R}{s_R}
\simeq
-\frac{Q\,\delta\hat a}{B}.
$$

For the ordinary physical regime $Q>0$ and $B>0$, the magnitude

$$
\left|
\frac{\partial\ln s_R}{\partial\hat a}
\right|
=
\frac{Q}{B}
$$

decreases monotonically with increasing $\hat a$ at fixed geometry. It also decreases as
$\gamma^2r^2$ increases.

Thus a grid whose absolute bin widths shrink toward large $\hat a$ is not justified by
local resonance sensitivity. The opposite end of the coordinate is at least as demanding,
and normally more demanding, in absolute $\hat a$ resolution.

For a uniform bin of width $\Delta\hat a$, midpoint evaluation gives the first-order bound

$$
\left|
\frac{\delta s_R}{s_R}
\right|
\lesssim
\frac{Q\,\Delta\hat a}
     {2\left(1+Q\hat a+\gamma^2r^2\right)}.
$$

This makes n_bins_ahat directly interpretable as a spectral-resolution control once the
physical $\hat a$ interval is specified.

## Sensitivity of the inverse resonance

The pull/query algorithm eliminates the energy coordinate with the inverse resonance

$$
\Gamma^2
=
\frac{1+Q\hat a}{K/s-r^2},
$$

where $K/s-r^2>0$ on the resonant support. At fixed query point and table coordinates other
than $\hat a$,

$$
\boxed{
\frac{\partial\ln\Gamma}{\partial\hat a}
=
\frac{Q}{2(1+Q\hat a)}
}.
$$

This sensitivity also decreases monotonically with increasing $\hat a$ for $Q>0$.
Therefore the analytical inversion used by xigma gives the same conclusion as the forward
resonance: concentrating absolute table resolution toward $\hat a_{\max}$ does not follow
from the local numerical conditioning of the resonance.

The Jacobian contains the same nonlinear factor through $A=1+Q\hat a$,

$$
\left|\frac{d\Gamma}{ds}\right|
=
\frac{K\Gamma^3}{2As^2},
$$

and the DER017 moment corrections contain coefficients proportional to $Q/B$ and
$Q^2/B^2$. None introduces a growing sensitivity with increasing $\hat a$ that would
restore the RES032 argument.

## Why an exact sensitivity-adapted coordinate is not a reusable table coordinate

For a fixed query one could define a transformed coordinate such as

$$
u=\ln\!\left(1+Q\hat a+\gamma^2r^2\right),
$$

for which the resonance obeys $d\ln s_R=-du$ at fixed $K$ and $\gamma$. In the head-on,
on-electron, unchirped limit this reduces to

$$
u=\ln(1+\hat a).
$$

Such a coordinate is not suitable as the general Stage-1.5 table axis because $Q$ depends
on the observation direction and electron direction, while $r$ also depends on the
requested observer. DER015 specifically requires the reusable table to remain
observer-independent and to store raw $\hat a$.

A fixed transformation such as $\ln(1+\hat a)$ would therefore optimize only a special
geometry. Over the present weakly nonlinear range it also provides only modest
redistribution of resolution. A uniform raw-$\hat a$ grid avoids embedding one observer or
one special limit into the reusable table.

## Relation to the previous nonuniform grid

RES032 uses, for $N$ target bins and a parameter $d$ called ahat_decades,

$$
v_i=(\hat a_{\max}-\hat a_{\min})10^{-di/N},
\qquad
\hat a_i=\hat a_{\max}-v_i.
$$

For $\hat a_{\min}=0$, the first bin width is

$$
\Delta\hat a_0
=
\hat a_{\max}\left(1-10^{-d/N}\right).
$$

With the current nominal range $\hat a_{\max}=0.5$ and $N=32$,

$$
\Delta\hat a_0(d=1)\simeq 0.034714,
$$

whereas changing to $d=0.3$ gives

$$
\Delta\hat a_0(d=0.3)\simeq 0.010678.
$$

A uniform grid on the same interval has

$$
\Delta\hat a_{\rm uniform}
=
\frac{0.5}{32}
=
0.015625.
$$

RES032 records historical scenario maxima of approximately $0.019$ for baseline,
$0.0019$ for low_a0, and $0.095$ for near_a0_max. Therefore the improvement observed
when reducing ahat_decades from $1$ toward $0.3$ is consistent with restoring resolution
in the populated low-$\hat a$ region and moving the grid closer to uniform. It is not
evidence that concentration near $\hat a_{\max}$ is required.

This numerical example is scenario-specific and is not the derivation. The durable result
is the monotonic sensitivity above.

## Limiting cases and consistency checks

### Observation along the electron direction

DER015 gives $Q=1$ when $\mathbf n=\mathbf e$ at nonsingular incidence. Then

$$
\frac{\partial\ln s_R}{\partial\hat a}
=
-\frac{1}{1+\hat a+\gamma^2r^2}.
$$

For $r=0$, the maximum magnitude occurs in the linear limit $\hat a\to0$.

### Head-on, unchirped, on-electron limit

For $D=\bar C=Q=1$ and $r=0$,

$$
s_R=\frac{\gamma^2}{1+\hat a},
\qquad
\frac{\partial\ln s_R}{\partial\hat a}
=
-\frac{1}{1+\hat a},
$$

and

$$
\frac{\partial\ln\Gamma}{\partial\hat a}
=
\frac{1}{2(1+\hat a)}.
$$

Both sensitivities decrease with $\hat a$.

### Vanishing nonlinear incidence

If $Q\to0$ while the resonance remains nonsingular, the spectrum becomes locally
insensitive to $\hat a$ and the choice of $\hat a$ grid ceases to matter for the resonance.

### Increasing observation offset

Increasing $r^2$ increases $B$ and decreases the forward-resonance sensitivity to
$\hat a$. It does not create a reason to concentrate resolution near large $\hat a$.

## Result

The current xigma resonance and inverse resonance do not support the RES032 claim that
absolute $\hat a$ resolution should be concentrated near $\hat a_{\max}$. At fixed
geometry, both the relative forward-resonance sensitivity and the relative inverse-root
sensitivity are largest at low $\hat a$ and decrease as $\hat a$ increases.

The production Stage-1.5 representation should therefore use a uniform grid in raw
$\hat a$, with numerical resolution controlled by n_bins_ahat. Stage 1 remains a fine
uniform table in the peak-independent nonlinear shape coordinate, and Stage 1.5 remains a
conservative retarget; only the target-grid spacing policy changes.

This result does not prove that a uniform grid is the globally optimal quadrature grid.
It establishes that the existing top-concentrated nonuniform grid is not justified by the
resonance sensitivity it was intended to resolve, while a uniform raw-$\hat a$ coordinate
is observer-independent, directly interpretable, and free of the extra ahat_decades
tuning parameter.

## Relation to existing derivations and decisions

- DER013 supplies the direction-dependent factor $D$ and the inverse-resonance/Jacobian
  structure.
- DER015 supplies the exact observer-dependent $Q$ and requires Stage 1/Stage 1.5 to store
  raw observer-independent $\hat a$.
- DER016 supplies the trajectory reduction and intensity scaling that Stage 1.5 retargets.
- DER017 supplies the second-order moment channels; their nonlinear coefficients retain the
  same $Q/B$ and $Q^2/B^2$ conditioning.
- RES032 introduced the current shape-table/retarget architecture and the nonuniform target
  law. This derivation challenges only the target-grid-spacing rationale. The shape-table
  cache, conservative retarget, intensity scaling, moment-channel transfer, and
  trailing-empty-bin truncation remain independently useful.

No existing physical derivation is superseded by this result.

## Implementation implications

The production path should:

- keep the Stage-1 shape table and Stage-1.5 conservative retarget architecture;
- generate the ordinary physical target edges uniformly over
  $[\hat a_{\min},\hat a_{\max}]$;
- retain the existing n_bins == 1 ignore-nonlinearity mode;
- if a separate sub-floor catch bin for $\hat a_{\min}>0$ is retained, treat it as a
  sentinel/floor semantic rather than as an alternative nonuniform grid policy;
- remove ahat_decades from the active production schema and call path;
- keep raw $\hat a$ observer-independent and continue applying $Q$ only in Stage 2;
- preserve conservative transfer of $H$ and all DER016 moment channels;
- preserve trailing-zero truncation if useful;
- avoid unrelated changes to Stage-2 physics or the table representation merely because
  production widths become equal.

The old nonuniform law can be recovered from Git history. If temporarily retained for
comparison, it should be a private validation/legacy helper and must not be selectable by
the production schema.

## Validation ideas

The strongest check is a convergence study against the direct-particle reduced-model
reference, which avoids using the Stage-1.5 $\hat a$ discretization as its own oracle.

For baseline, low_a0, near_a0_max, and at least one crossed/off-axis case:

1. compare uniform Stage-1.5 grids at increasing n_bins_ahat;
2. compare, at equal bin count, against the historical RES032 grid if the legacy helper is
   retained;
3. report total yield, spectral centroid, and normalized integrated $L_1$ spectral error;
4. verify conservative transfer of $H$, $H_{\mathrm{var}\,a}$,
   $H_{\mathrm{var}\,C}$, and $H_{\mathrm{cov}\,aC}$;
5. verify that the public production path always constructs uniform ordinary $\hat a$
   edges and that a legacy ahat_decades input cannot alter the result.

The publication validation should scan n_bins_ahat, not tune a nonlinear spacing
parameter.

## Open questions

- A uniform grid is the selected production policy, but the minimum default
  n_bins_ahat required for the desired spectral accuracy remains a numerical convergence
  question.
- It may be useful later to shrink the uniform interval to the populated $\hat a$ support
  or to derive an error-controlled adaptive grid. Either choice should be justified
  separately; neither is required by this derivation.
- An observer-dependent sensitivity-equalized coordinate could reduce work for one query,
  but it would conflict with the observer-independent reusable-table design and is not a
  candidate for the current Stage-1.5 axis.
