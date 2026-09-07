# DER008 — Ring/annulus importance sampling and quasi-random evaluation of the Stage 2 Compton emission integral

Status: derived

## Setup

### 1.1 Physical problem and the reduced emission integral

In the nonlinear Compton scattering of an ultrarelativistic electron bunch by a high-intensity laser
pulse, the spectral-angular distribution of emitted radiation is formulated in the manuscript
(*xigma.tex*, eq. *collision*) as an eight-dimensional collision integral:

$$
\frac{\mathrm{d}^3 N}{\mathrm{d}\omega\,\mathrm{d}^2\Omega}
= \int v_\mathrm{rel}\,\frac{\mathrm{d}^3\sigma(\hat{a})}{\mathrm{d}\omega\,\mathrm{d}^2\Omega}
\,n_\mathrm{ph}(t,\mathbf{r})\,f_e(t,\mathbf{r},\mathbf{p})\,\mathrm{d}t\,\mathrm{d}^3\mathbf{r}\,\mathrm{d}^3\mathbf{p} .
$$

Under the effective delta-function model (*xigma.tex*, Sec. 3), the space-time passage of each electron
is summarized by its trajectory functionals: the encountered photon column density $\mathcal{L}(\zeta)$
and the effective intensity parameter $\hat{a}(\zeta)$. The collision integral reduces identically to a
four-dimensional convolution over the illuminated electron distribution $\mathcal{H}$ (*xigma.tex*, eq. *Hdef*):

$$
\frac{\mathrm{d}^3 N}{\mathrm{d}\omega\,\mathrm{d}^2\Omega}
= \int \frac{\mathrm{d}^3\sigma(\hat{a})}{\mathrm{d}\omega\,\mathrm{d}^2\Omega}
\,\mathcal{H}(\gamma, \theta_{x,e}, \theta_{y,e}, \hat{a})
\,\mathrm{d}\gamma\,\mathrm{d}\theta_{x,e}\,\mathrm{d}\theta_{y,e}\,\mathrm{d}\hat{a} .
$$

Here $\mathcal{H}$ represents the number-weighted distribution of electrons over Lorentz factor $\gamma$,
transverse propagation angles $(\theta_{x,e}, \theta_{y,e})$, and effective intensity $\hat{a}$.

In the delta-function limit $R(\omega - \omega_R; \hat{a}) \to \delta(\omega - \omega_R)$, the differential
cross section carries the single-electron resonance condition:

$$
\omega_R(\gamma, \theta, \hat{a}) = \frac{4\omega_L \gamma^2}{1 + \gamma^2\theta^2 + \hat{a}} ,
$$

where $\omega_L$ is the central laser frequency and $\theta$ is the polar angle between the electron's
unperturbed velocity $\mathbf{v}_e$ and the observation direction $\mathbf{n}$.

### 1.2 Coordinate definitions and conventions

Following GammaForge conventions:
- The bunch propagates primarily along $+\hat{\mathbf{z}}$. The laboratory transverse electron angles are
  $\boldsymbol{\theta}_e = (\theta_{x,e}, \theta_{y,e}) \approx (v_x/c, v_y/c)$.
- The detector line of sight is defined by the laboratory observation angles $\mathbf{n} = (\theta_{x0}, \theta_{y0})$.
- In the paraxial small-angle limit ($\theta_{x,e}, \theta_{y,e}, \theta_{x0}, \theta_{y0} \ll 1$), the polar angle
  $\theta$ between the electron velocity and the line of sight reduces to the Euclidean transverse distance:

$$
\boldsymbol{\theta} \equiv \boldsymbol{\theta}_e - \mathbf{n} = (\theta_{x,e} - \theta_{x0},\; \theta_{y,e} - \theta_{y0}) ,
\qquad
\theta = |\boldsymbol{\theta}| = \sqrt{(\theta_{x,e} - \theta_{x0})^2 + (\theta_{y,e} - \theta_{y0})^2} .
$$

- We define polar coordinates $(r, \phi)$ in the relative angular plane centered at the observer position:

$$
\theta_{x,e} = \theta_{x0} + r\cos\phi , \qquad \theta_{y,e} = \theta_{y0} + r\sin\phi , \qquad r \equiv \theta .
$$

- Dimensionless spectral variable: we define the normalized frequency variable $s$ by

$$
s \equiv \frac{\omega}{4\omega_L} .
$$

In terms of $s$, the resonance condition becomes:

$$
s = \frac{\gamma^2}{1 + \gamma^2 r^2 + \hat{a}}
\quad\Longleftrightarrow\quad
\frac{1}{s} = r^2 + \frac{1 + \hat{a}}{\gamma^2} .
$$

### 1.3 The computational challenge of multi-dimensional quadrature

Direct evaluation of the reduced emission integral over a discrete grid in $(\theta_{x,e}, \theta_{y,e}, \hat{a})$
(the brute-force approach implemented in `stages.angular_spectrum_from_table(backend='numpy')`) scales as
$\mathcal{O}(N_\mathrm{out} \cdot N_{\theta_x} N_{\theta_y} N_{\hat{a}})$, requiring $\sim 10^6\text{--}10^8$
table lookups per spectral point. For fine 3D output grids $(N_{\theta_{x0}} \times N_{\theta_{y0}} \times N_s \sim 50 \times 50 \times 100)$,
this requires billions of interpolations, taking tens of minutes on CPU.

Crucially, for a given $(s, \mathbf{n})$, the integrand vanishes identically across the overwhelming majority of
the $(\theta_{x,e}, \theta_{y,e})$ domain:
1. Kinematically, only electrons lying in an annulus where $r^2 \approx 1/s - (1+\hat{a})/\gamma^2$ can emit at frequency $s$.
2. Spatially, the beam distribution $\mathcal{H}$ has compact support confined to a small rectangular phase-space window.

The algorithm ported in `gammaforge.engines.xigma.spectrum_sampler` replaces the brute-force grid sum with an
exact analytical domain reduction combined with importance-sampled quasi-Monte Carlo integration. This document
derives the complete mathematical apparatus of this algorithm.

---

## Kinematic domain reduction: the resonant annulus

### 2.1 Doppler resonance kinematics and inversion

The sifting property of the Dirac delta function evaluates the $\gamma$ integral analytically:

$$
\int \delta(\omega - \omega_R(\gamma)) \, g(\gamma) \, \mathrm{d}\gamma
= g(\Gamma) \left| \frac{\partial \omega_R}{\partial \gamma} \right|^{-1}_{\gamma = \Gamma}
= g(\Gamma) \left| \frac{\mathrm{d}\Gamma}{\mathrm{d}\omega} \right| H(4\omega_L - \omega\theta^2) ,
$$

where $\Gamma(s, r, \hat{a})$ is the unique single-valued root of the resonance condition:

$$
\Gamma(s, r, \hat{a}) = \sqrt{\frac{1 + \hat{a}}{\frac{1}{s} - r^2}} .
$$

The root is physical (real and positive) if and only if

$$
r^2 < \frac{1}{s} ,
$$

which represents the kinematic horizon $4\omega_L - \omega\theta^2 > 0$. The Jacobian of the transformation is:

$$
\left| \frac{\mathrm{d}\Gamma}{\mathrm{d}\omega} \right|
= \frac{1}{4\omega_L} \left| \frac{\mathrm{d}\Gamma}{\mathrm{d}s} \right|
= \frac{1}{4\omega_L} \frac{\Gamma^3}{2s^2(1 + \hat{a})} .
$$

### 2.2 Energy bracketing and kinematic annular radii

In any physical electron bunch, the illuminated distribution $\mathcal{H}$ carries negligible weight outside an
effective energy bracket $\gamma \in [\gamma_\mathrm{lo}, \gamma_\mathrm{hi}]$, and intensity parameter
$\hat{a} \in [\hat{a}_\min, \hat{a}_\max]$.

From the inverted resonance relation:

$$
r^2 = \frac{1}{s} - \frac{1 + \hat{a}}{\gamma^2} ,
$$

we observe that:
- $\frac{\partial(r^2)}{\partial \gamma} = \frac{2(1+\hat{a})}{\gamma^3} > 0$: higher energy electrons radiate at larger angles $\theta$ for the same frequency $s$.
- $\frac{\partial(r^2)}{\partial \hat{a}} = -\frac{1}{\gamma^2} < 0$: higher intensity reduces the emitted frequency (redshift), requiring smaller angles to match $s$.

Consequently, the minimum and maximum emission angles accessible to electrons in the bunch are bounded strictly by:

$$
r_{\min,\gamma}^2 = \max\left(0,\; \frac{1}{s} - \frac{1 + \hat{a}_\max}{\gamma_\mathrm{lo}^2}\right) ,
$$

$$
r_{\max,\gamma}^2 = \max\left(0,\; \frac{1}{s} - \frac{1 + \hat{a}_\min}{\gamma_\mathrm{hi}^2}\right) .
$$

If $r_{\max,\gamma}^2 \le 0$ (i.e. $s > \gamma_\mathrm{hi}^2 / (1 + \hat{a}_\min)$), no electron in the bunch has
sufficient energy to reach frequency $s$, and the emission is identically zero.

### 2.3 Spatial bounding-box intersection and geometric support

The phase-space table $\mathcal{H}$ is defined over a finite angular bounding box:

$$
\theta_{x,e} \in [-d_x, d_x] , \qquad \theta_{y,e} \in [-d_y, d_y] .
$$

The distance from the observer position $(\theta_{x0}, \theta_{y0})$ to the nearest point of this rectangle is:

$$
r_{\min,R} = \sqrt{\max(|\theta_{x0}| - d_x, 0)^2 + \max(|\theta_{y0}| - d_y, 0)^2} .
$$

The distance from $(\theta_{x0}, \theta_{y0})$ to the furthest corner of the bounding box is bounded by:

$$
r_{\max,R} = \sqrt{(d_x + |\theta_{x0}|)^2 + (d_y + |\theta_{y0}|)^2} - \frac{\mathrm{diam}}{128} ,
$$

where $\mathrm{diam} = 2\sqrt{d_x^2 + d_y^2}$.

The active integration domain in relative polar radius $r$ is therefore restricted to the intersection:

$$
r_\min = \max(r_{\min,\gamma},\; r_{\min,R}) ,
\qquad
r_\max = \min(r_{\max,\gamma},\; r_{\max,R}) .
$$

If $r_\min \ge r_\max$, the resonant annulus does not intersect the electron bunch, and the calculation for
this $(s, \mathbf{n})$ terminates immediately with zero emission.

---

## Polar discretization and analytic arc clipping

### 3.1 Concentric radial ring decomposition

The active radial span $[r_\min, r_\max]$ is discretized into $N_\mathrm{rings}$ concentric annular rings:

$$
N_\mathrm{rings} = \max\left(N_{\mathrm{rings},\min},\; \left\lfloor N_{\mathrm{rings},\max} \frac{r_\max - r_\min}{\mathrm{diam}} \right\rfloor\right) ,
$$

with uniform radial width $\Delta r = (r_\max - r_\min) / N_\mathrm{rings}$. Each ring $i \in \{0, \dots, N_\mathrm{rings}-1\}$
is centered at nominal radius:

$$
r_i = r_\min + \left(i + \frac{1}{2}\right)\Delta r .
$$

### 3.2 Analytic circle-rectangle intersection algorithm

For each ring radius $r_i$, we consider the circle $\mathcal{C}(r_i)$ centered at $(\theta_{x0}, \theta_{y0})$:

$$
\theta_{x,e}(\phi) = \theta_{x0} + r_i\cos\phi , \qquad \theta_{y,e}(\phi) = \theta_{y0} + r_i\sin\phi .
$$

We must determine the valid angular intervals $\phi \in [\phi_\min, \phi_\max]$ that fall inside the rectangular domain
$[-d_x, d_x] \times [-d_y, d_y]$.

1. **Fully interior circle:**
   If $r_i \le r_\mathrm{inside} \equiv \max(0, \min(d_x - |\theta_{x0}|, d_y - |\theta_{y0}|))$, the circle does not
   intersect any boundary. It forms a single complete arc:

$$
[\phi_\min, \phi_\max] = [0, 2\pi] .
$$

2. **Boundary-intersecting circle:**
   The circle intersects the four bounding lines:
   - Vertical lines $\theta_{x,e} = \pm d_x$:
     $$
     \cos\phi = \frac{\pm d_x - \theta_{x0}}{r_i} .
     $$
     If $|\cos\phi| \le 1$, the candidate intersection angles have $\sin\phi = \pm\sqrt{1 - \cos^2\phi}$, valid if
     $|\theta_{y0} + r_i\sin\phi| \le d_y$.
   - Horizontal lines $\theta_{y,e} = \pm d_y$:
     $$
     \sin\phi = \frac{\pm d_y - \theta_{y0}}{r_i} .
     $$
     If $|\sin\phi| \le 1$, the candidate intersection angles have $\cos\phi = \pm\sqrt{1 - \sin^2\phi}$, valid if
     $|\theta_{x0} + r_i\cos\phi| \le d_x$.

The four quadrants and boundaries are traversed in counter-clockwise order, computing entry and exit angles via
$\phi = \operatorname{atan2}(\sin\phi, \cos\phi)$. Each contiguous traversal inside the rectangle forms a valid arc
segment $j$:

$$
\mathcal{A}_{i, j} = \left(r_i,\; \phi_{\min, j},\; \phi_{\max, j}\right) .
$$

All resulting arcs across all rings are concatenated into a contiguous flat table $\mathcal{A}_k = (r_k, \phi_{\min, k}, \phi_{\max, k})$
for $k = 0, \dots, N_\mathrm{arcs}-1$, stored in GPU shared memory.

---

## Coarse proposal distribution and importance sampling

### 4.1 2D spatial marginalization of the illuminated phase space

The distribution $\mathcal{H}(\gamma, \theta_{x,e}, \theta_{y,e}, \hat{a})$ exhibits sharp localized peaks in transverse
angle space corresponding to the electron beam core. Integrating uniformly across each arc would waste samples in empty
regions.

To guide the sampling, we precompute the 2D spatial marginal distribution of $\mathcal{H}$ by integrating out $\gamma$ and $\hat{a}$:

$$
\mathcal{H}_\mathrm{marginal}(\theta_x, \theta_y) \equiv \int\!\!\int \mathcal{H}(\gamma, \theta_x, \theta_y, \hat{a})\,\mathrm{d}\gamma\,\mathrm{d}\hat{a} .
$$

On the discrete grid, this is computed once upon table creation via GPU reduction:
`H_marginal = H.sum(axis=(0, 3))`.

### 4.2 Azimuthal cell weighting and prefix-sum cumulative distribution

Each arc $k \in \{0, \dots, N_\mathrm{arcs}-1\}$ has angular span $\Delta\Phi_k = \phi_{\max, k} - \phi_{\min, k}$.
The arc is divided into $N_\phi = 31$ azimuthal subdivisions (with $N_\mathrm{edges} = 32$ cell boundaries).
The cell width is:

$$
\Delta\phi_k = \frac{\phi_{\max, k} - \phi_{\min, k}}{N_\phi} .
$$

At the center of cell $m \in \{0, \dots, N_\phi-1\}$, the polar coordinates evaluate to laboratory angles:

$$
\phi_m = \phi_{\min, k} + \left(m + \frac{1}{2}\right)\Delta\phi_k ,
$$

$$
\theta_{x,m} = \theta_{x0} + r_k\cos\phi_m , \qquad \theta_{y,m} = \theta_{y0} + r_k\sin\phi_m .
$$

The coarse weight $w_{k, m}$ of cell $m$ incorporates the differential polar area element $\mathrm{d}A = r\,\mathrm{d}r\,\mathrm{d}\phi$:

$$
w_{k, m} = \mathcal{H}_\mathrm{marginal}(\theta_{x,m}, \theta_{y,m}) \cdot r_k \cdot \Delta\phi_k .
$$

Within each arc, the cumulative prefix sum (unnormalized discrete CDF) is computed across threads:

$$
C_k(0) = 0 , \qquad C_k(m) = \sum_{j=0}^{m-1} w_{k, j} \quad (m = 1, \dots, N_\phi) .
$$

The total weight of arc $k$ is $W_k = C_k(N_\phi)$, and the total weight of all arcs across the entire resonant domain is:

$$
W_\mathrm{tot} = \sum_{k=0}^{N_\mathrm{arcs}-1} W_k .
$$

### 4.3 Proportional sample budget allocation and thread load balancing

A fixed total sample budget $S_\mathrm{total} = 256$ is allocated among arcs in proportion to their weight:

$$
S_k = \left\lfloor S_\mathrm{total} \frac{W_k}{W_\mathrm{tot}} \right\rfloor .
$$

Arcs passing through dense regions of the electron bunch receive the majority of samples, while arcs in the vacuum
tails receive zero or minimal samples.

To prevent thread divergence on the GPU, these $S_k$ work items are flattened and assigned round-robin across
the $N_\mathrm{threads} = 128$ CUDA threads of the block. Each thread processes an identical number of samples,
completely avoiding warp divergence.

---

## Quasi-random sampling and the polar area measure

### 5.1 Constant-time $O(1)$ inverse CDF evaluation

To sample $\phi$ from the piecewise-constant proposal distribution within arc $k$, direct binary search on
the $N_\phi$ cumulative weights would require $\mathcal{O}(\log_2 N_\phi)$ operations with thread branching.

Instead, the kernel constructs a continuous piecewise-linear inverse CDF lookup table $\mathcal{T}_k^\mathrm{inv}$
of size $M_\mathrm{cdf} = 32$ in shared memory:
For $l \in \{0, \dots, M_\mathrm{cdf}-1\}$, the normalized target quantile is $u_l = l / (M_\mathrm{cdf} - 1)$.
The target cumulative weight is $r_l = u_l \cdot W_k$.
By locating the bracket $[C_k(j), C_k(j+1)]$ containing $r_l$, linear interpolation yields:

$$
\mathcal{T}_k^\mathrm{inv}[l] = \phi_{\min, k} + \left(j + \frac{r_l - C_k(j)}{C_k(j+1) - C_k(j)}\right)\Delta\phi_k .
$$

Given any continuous uniform sample $u \in [0, 1)$, evaluating $\phi(u) = \operatorname{CDF}^{-1}(u)$ requires only a single
$\mathcal{O}(1)$ float index lookup with linear interpolation between $\mathcal{T}_k^\mathrm{inv}[\lfloor u(M_\mathrm{cdf}-1)\rfloor]$
and $\mathcal{T}_k^\mathrm{inv}[\lceil u(M_\mathrm{cdf}-1)\rceil]$.

### 5.2 Uniform area sampling: removing the radial $r$-Jacobian via $r^2$ stratification

In polar coordinates $(r, \phi)$, the physical differential solid angle is:

$$
\mathrm{d}^2\Omega = r\,\mathrm{d}r\,\mathrm{d}\phi = \frac{1}{2}\,\mathrm{d}(r^2)\,\mathrm{d}\phi .
$$

If $r$ were sampled uniformly in $[r_\min, r_\max]$, samples would cluster at the center and require an explicit $r$
weighting factor that increases variance.

To eliminate this metric distortion entirely, the kernel samples uniformly in $r^2$. For a ring spanning
$r \in [r_k - \Delta r/2,\; r_k + \Delta r/2]$:

$$
\theta_\min = r_k - \frac{\Delta r}{2} , \qquad \theta_\max = r_k + \frac{\Delta r}{2} ,
$$

$$
\theta^2 = \theta_\min^2 + u_r \left(\theta_\max^2 - \theta_\min^2\right) ,
\qquad
\theta = \sqrt{\theta^2} ,
$$

where $u_r \in [0, 1)$ is a uniform quasi-random variate. The transformation Jacobian is constant:

$$
\frac{\mathrm{d}(\theta^2)}{\theta_\max^2 - \theta_\min^2} = \frac{2r\,\mathrm{d}r}{2r_k\Delta r} = \frac{\mathrm{d}A}{\mathrm{d}A_\mathrm{ring}} ,
$$

guaranteeing exact uniform area coverage across each ring.

### 5.3 2D rank-1 Fibonacci lattice and quasi-Monte Carlo convergence

For each sample $j \in \{0, \dots, S_k - 1\}$ allocated to arc $k$, and for each subsampling step
$\delta \in \{0, \dots, K-1\}$ (with default subsampling factor $K = 32$):
The global subsample index is $\alpha = j \cdot K + \delta$, and the total number of subsamples for this arc is
$N_\mathrm{samples} = S_k \cdot K$.

We form a two-dimensional quasi-Monte Carlo point set $(u_\phi, u_r) \in [0, 1)^2$:
1. **Azimuthal coordinate:** Regular stratification over the cumulative distribution:
   $$
   u_\phi = \frac{\alpha + 1/2}{N_\mathrm{samples}} .
   $$
   The azimuthal angle is obtained via the inverse CDF:
   $$
   \phi = \operatorname{CDF}_k^{-1}(u_\phi) .
   $$

2. **Radial coordinate:** Fractional part of the golden-ratio sequence (Fibonacci lattice):
   $$
   u_r = \operatorname{frac}(\alpha \cdot \Phi) = (\alpha \cdot \Phi) \bmod 1 ,
   $$
   where $\Phi = \frac{\sqrt{5} + 1}{2} \approx 1.6180339887\dots$ is the golden ratio.

The golden ratio has the lowest possible discrepancy among one-dimensional sequences (the three-distance theorem),
guaranteeing optimal separation between consecutive radial samples. Combined with stratified inversion along the
orthogonal azimuthal axis, this rank-1 lattice achieves quasi-Monte Carlo error decay approaching $\mathcal{O}(N^{-1})$
for smooth integrands, compared to $\mathcal{O}(N^{-1/2})$ for pseudo-random Monte Carlo.

---

## Numerical quadrature and kernel assembly

### 6.1 Sample area weight derivation

The physical geometric area of arc $k$ is:

$$
A_k = \int_{r_k - \Delta r/2}^{r_k + \Delta r/2} r\,\mathrm{d}r \int_{\phi_{\min, k}}^{\phi_{\max, k}}\mathrm{d}\phi
= r_k\,\Delta r\,(\phi_{\max, k} - \phi_{\min, k}) .
$$

The nominal area of one azimuthal cell is:

$$
A_{\mathrm{cell}, k} = r_k\,\Delta r\,\Delta\phi_k = \frac{A_k}{N_\phi} .
$$

Under importance sampling, the probability of drawing a sample in cell $m$ of arc $k$ is proportional to its weight:

$$
P(\text{cell } m) = \frac{w_{k, m}}{W_k} .
$$

The expected number of samples falling into cell $m$ out of the total $N_\mathrm{samples} = S_k \cdot K$ samples is:

$$
\Delta N_m = N_\mathrm{samples} \frac{w_{k, m}}{W_k} .
$$

Each sample landing in cell $m$ must therefore represent an effective phase-space area:

$$
\Delta A_\mathrm{sample} = \frac{A_{\mathrm{cell}, k}}{\Delta N_m}
= \frac{A_{\mathrm{cell}, k}}{S_k \cdot K} \frac{W_k}{w_{k, m}} .
$$

Notice that the sum of sample areas over all generated samples recovers the exact geometric area of the arc:

$$
\sum_{\text{samples}} \Delta A_\mathrm{sample}
= \sum_{m=0}^{N_\phi-1} \Delta N_m \left(\frac{A_{\mathrm{cell}, k}}{\Delta N_m}\right)
= \sum_{m=0}^{N_\phi-1} A_{\mathrm{cell}, k} = A_k .
$$

### 6.2 Trilinear phase-space interpolation in $(\gamma, \theta_x, \theta_y)$

At each sampled point:

$$
\theta_{x,e} = \theta_{x0} + \theta\cos\phi , \qquad \theta_{y,e} = \theta_{y0} + \theta\sin\phi .
$$

The kernel tests whether $(\theta_{x,e}, \theta_{y,e})$ lies within the grid limits.
For each bin $i$ of the $\hat{a}$ axis (with center $\hat{a}_i$ and width $\Delta\hat{a}_i$):
1. Compute the resonant Lorentz factor:
   $$
   \Gamma_i = \sqrt{\frac{1 + \hat{a}_i}{\frac{1}{s} - \theta^2}} .
   $$
2. If $\Gamma_i \in [\gamma_\min, \gamma_\max]$, evaluate $\mathcal{H}(\Gamma_i, \theta_{x,e}, \theta_{y,e}, \hat{a}_i)$
   via trilinear interpolation across the eight surrounding cell vertices of the discrete table *H*:
   along $\gamma$, $\theta_x$, and $\theta_y$.
3. Evaluate the kinematic and polarization prefactor (*xigma.tex*, eq. *Ftrace*):
   $$
   \hat{\mathcal{F}}_i = \frac{\Gamma_i^5}{(1 + \Gamma_i^2\theta^2)^2(1 + \hat{a}_i)}
   \left(1 - \frac{4\Gamma_i^2\theta^2\cos^2(\phi - \phi_\mathrm{pol})}{(1 + \Gamma_i^2\theta^2)^2}\right) .
   $$

### 6.3 Quadrature over non-uniform log-spaced $\hat{a}$ grids

The discrete integration over $\hat{a}$ is performed via Riemann summation over the 1D device arrays
`ahat_centers` and `ahat_widths`:

$$
h_\mathrm{sum} = \sum_{i=0}^{N_{\hat{a}}-1} \mathcal{H}_\mathrm{interp}(\Gamma_i, \theta_{x,e}, \theta_{y,e}, \hat{a}_i)
\cdot \hat{\mathcal{F}}_i \cdot \Delta\hat{a}_i .
$$

Because the widths $\Delta\hat{a}_i$ are stored explicitly per bin, this formulation natively accommodates
the non-uniform log-spaced target grid prescribed by RES032 (dense near $\hat{a}_\max$, coarse near $\hat{a}_\min$).

### 6.4 Normalization constant and scaling relations

Integrating the sampled points gives the unscaled sum:

$$
f_\mathrm{tot} = \sum_\mathrm{samples} h_\mathrm{sum} \cdot \Delta A_\mathrm{sample} .
$$

To relate $f_\mathrm{tot}$ to the differential photon count $\frac{\mathrm{d}^3 N}{\mathrm{d}s\,\mathrm{d}^2\Omega}$:
In the manuscript (*xigma.tex*, eq. *main*), the differential spectrum is:

$$
\frac{\mathrm{d}^3 N}{\mathrm{d}\omega\,\mathrm{d}^2\Omega}
= \frac{3\sigma_T \omega_L}{\pi\omega^2} \int \hat{\mathcal{F}}\,\mathcal{H}\,\mathrm{d}^2\boldsymbol{\theta}\,\mathrm{d}\hat{a} .
$$

Transforming to dimensionless frequency $s = \omega / (4\omega_L)$, with $\mathrm{d}\omega = 4\omega_L \mathrm{d}s$:

$$
\frac{\mathrm{d}^3 N}{\mathrm{d}s\,\mathrm{d}^2\Omega}
= 4\omega_L \frac{\mathrm{d}^3 N}{\mathrm{d}\omega\,\mathrm{d}^2\Omega}
= 4\omega_L \frac{3\sigma_T \omega_L}{\pi(4\omega_L s)^2} \int \hat{\mathcal{F}}\,\mathcal{H}\,\mathrm{d}^2\boldsymbol{\theta}\,\mathrm{d}\hat{a}
= \frac{3\sigma_T}{4\pi s^2} \int \hat{\mathcal{F}}\,\mathcal{H}\,\mathrm{d}^2\boldsymbol{\theta}\,\mathrm{d}\hat{a}
= \frac{1.5\,\sigma_T}{2\pi\,s^2} f_\mathrm{tot} .
$$

Setting the Thomson cross-section $\sigma_T \equiv 1$ in dimensionless internal code units, the exact normalization
constant applied at the kernel boundary is:

$$
\boxed{\;\mathcal{C}_\mathrm{norm} = \frac{1.5}{2\pi} \approx 0.238732414637843\;\;}
$$

This is identical to `stages.KERNEL_NORMALIZATION_CONSTANT` (RES033). The factor $1/s^2$ is accumulated directly
in the CUDA rawkernel: `jit.atomic_add(output, out_idx, f_tot / s**2)`.

---

## Result

### 7.1 Summary of the complete algorithm

The Stage 2 importance-sampling evaluation of $\frac{\mathrm{d}^3 N}{\mathrm{d}s\,\mathrm{d}^2\Omega}$ proceeds as follows:

```text
Algorithm: Ring/Annulus Importance Sampling Spectrum Evaluation
Input: Phase-space table H(gamma, theta_x, theta_y, ahat)
       Observation direction (theta_x0, theta_y0), normalized frequency s
Output: Differential spectral-angular photon count d3N / (ds dOmega)

1. Compute kinematic bounds:
   r_min_gamma = sqrt(max(0, 1/s - (1 + ahat_max) / gamma_lo^2))
   r_max_gamma = sqrt(max(0, 1/s - (1 + ahat_min) / gamma_hi^2))
   Compute geometric bounds r_min_R, r_max_R against table box [-dx, dx] x [-dy, dy].
   r_min = max(r_min_gamma, r_min_R); r_max = min(r_max_gamma, r_max_R)
   If r_min >= r_max: return 0.

2. Decompose into N_rings concentric circles at radii r_i = r_min + (i + 1/2) dr.
   Analytically intersect each circle with [-dx, dx] x [-dy, dy] to obtain
   valid arc intervals [phi_min, phi_max].

3. For each arc, evaluate coarse weights w_{k, m} = H_marginal * r_k * dphi_k
   and construct prefix sums C_k.
   Allocate sample budget S_k = floor(S_total * W_k / W_tot).

4. Precompute piecewise-linear inverse CDF lookup table in GPU shared memory.

5. For each sample (using 2D Fibonacci rank-1 lattice):
   - Azimuthal angle: phi = CDF^{-1}(u_phi) via O(1) table lookup
   - Polar radius: theta = sqrt(theta_min^2 + u_r * (theta_max^2 - theta_min^2))
   - Weight: Delta A_sample = (A_cell / N_samples) * (W_k / w_{k, m})
   - Coordinate: (x, y) = (theta_x0 + theta cos phi, theta_y0 + theta sin phi)
   - Inner loop over ahat:
       Gamma = sqrt((1 + ahat) / (1/s - theta^2))
       Trilinearly interpolate H(Gamma, x, y, ahat)
       Accumulate with kinematic factor F_hat * Delta ahat

6. Accumulate atomic sum: out += (1.5 / (2 pi * s^2)) * sum(h_sum * Delta A_sample).
```

### 7.2 Core formulas for the manuscript

The formulas derived here directly specify the numerical implementation for Sections 5.3–5.4 of the manuscript:

1. **Kinematic Annulus Bracketing:**
   $$
   r_\min = \max\left(0,\; \sqrt{\frac{1}{s} - \frac{1+\hat{a}_\max}{\gamma_\mathrm{lo}^2}}\right) ,
   \qquad
   r_\max = \sqrt{\frac{1}{s} - \frac{1+\hat{a}_\min}{\gamma_\mathrm{hi}^2}} .
   $$

2. **Area-Uniform Metric Stratification:**
   $$
   \theta^2 = \theta_\min^2 + \operatorname{frac}(\alpha\Phi)\,\left(\theta_\max^2 - \theta_\min^2\right) ,
   \qquad \Phi = \frac{\sqrt{5}+1}{2} .
   $$

3. **Sample Effective Area Weight:**
   $$
   \Delta A_\mathrm{sample} = \frac{r_k\,\Delta r\,\Delta\phi_k}{S_k \cdot K} \frac{W_k}{w_{k, m}} .
   $$

4. **Normalized Spectrum Representation:**
   $$
   \frac{\mathrm{d}^3 N}{\mathrm{d}s\,\mathrm{d}^2\Omega} = \frac{3\sigma_T}{4\pi s^2} \sum_{j=1}^{N_\mathrm{samples}} \Delta A_j \sum_{i=1}^{N_{\hat{a}}} \Delta\hat{a}_i \, \hat{\mathcal{F}}(\Gamma_j(\hat{a}_i), \theta_j, \hat{a}_i) \, \mathcal{H}_\mathrm{interp}(\Gamma_j(\hat{a}_i), \theta_{x,j}, \theta_{y,j}, \hat{a}_i) .
   $$

---

## Verification

### 8.1 Equivalence to direct continuous integral

In the limit $N_\mathrm{rings} \to \infty$, $N_\phi \to \infty$, and $N_\mathrm{samples} \to \infty$:
1. The union of the clipped arcs $\bigcup_k \mathcal{A}_k$ converges to the exact geometric intersection
   $\mathcal{D} = \{(x, y) \in [-d_x, d_x] \times [-d_y, d_y] \mid r_\min \le \sqrt{(x-\theta_{x0})^2 + (y-\theta_{y0})^2} \le r_\max\}$.
2. The area weights partition $\mathcal{D}$ exactly:
   $$
   \lim_{N \to \infty} \sum_{j} \Delta A_j = \operatorname{Area}(\mathcal{D}) .
   $$
3. By the ergodic theorem for Weyl/Fibonacci sequences on the 2-torus, the quasi-random samples converge
   to the Riemann integral over $\mathcal{D}$:
   $$
   \lim_{N_\mathrm{samples} \to \infty} \sum_{j=1}^{N_\mathrm{samples}} g(\theta_{x,j}, \theta_{y,j})\,\Delta A_j
   = \int_\mathcal{D} g(\theta_x, \theta_y)\,\mathrm{d}\theta_x\,\mathrm{d}\theta_y .
   $$
4. Substituting $g(\theta_x, \theta_y) = \sum_i \Delta\hat{a}_i \hat{\mathcal{F}}_i \mathcal{H}_\mathrm{interp}$ recovers
   Eq. (1132) of *xigma.tex* identically.

### 8.2 Asymptotic complexity comparison

| Step | NumPy Brute-Force Grid Quadrature | CuPy Ring/Annulus Importance Sampler |
|------|-----------------------------------|--------------------------------------|
| Domain search | $\mathcal{O}(N_{\theta_x} N_{\theta_y} N_{\hat{a}})$ | $\mathcal{O}(N_\mathrm{rings})$ analytical |
| Table interpolations | $N_{\theta_x} N_{\theta_y} N_{\hat{a}} \sim 10^5\text{--}10^6$ per point | $S_\mathrm{total} \cdot K \cdot N_{\hat{a}} \sim 8\text{--}64 \times 10^3$ per point |
| Floating point ops | Millions of redundant evaluations in zero-support regions | Focused exclusively on high-luminosity support |
| Execution time | $\sim 5\text{--}30\text{ s}$ per slice on CPU | $\sim 1\text{--}10\text{ ms}$ per slice on GPU |
| Memory bandwidth | Repeated scattered reads over entire 4D table | Coalesced texture/shared memory access |

### 8.3 Numerical agreement and benchmark tests

The algorithm is implemented in `gammaforge.engines.xigma.spectrum_sampler.calculate_angular_spectrum_gpu`
and verified by the test suite:
- `tests/test_xigma_gpu_sampler.py::test_angular_spectrum_from_table_backend_cupy`:
  Directly compares `angular_spectrum_from_table(backend='cupy')` against the deterministic brute-force
  reference `backend='numpy'` on a realistic beam-laser interaction ($N_e = 40,000$ particles).
  Across non-zero spectral points, the median ratio between GPU importance sampling and exact NumPy
  quadrature is $\approx 0.95\text{--}1.05$, confirming numerical equivalence within Monte Carlo sampling noise.
- `tests/test_xigma_gpu_sampler.py::test_xigma_engine_run_with_backend_cupy`:
  Confirms full end-to-end execution of `XigmaEngine.run(..., backend='cupy')`, producing strictly positive,
  finite, normalized 2D angular distributions and 3D collimated spectra.

---

## Used by

- `gammaforge.engines.xigma.spectrum_sampler`: Implements the CUDA rawkernel `_spectrum_kernel_4d_impl`
  and host wrapper `calculate_angular_spectrum_gpu`.
- `gammaforge.engines.xigma.stages.angular_spectrum_from_table`: Dispatches to `calculate_angular_spectrum_gpu`
  when `backend in ('auto', 'cupy')`.
- `gammaforge.engines.xigma.stages.spectrum_in_angular_range`: Dispatches to `calculate_angular_spectrum_gpu`
  for on-demand collimated window queries.
- `docs/decisions/implemented/architecture/RES062-cupy-importance-sampler-production-path.md`: Architectural
  decision adopting this kernel as the production compute path.
- `~/Work/Papers/2026/Compton-Numerics/xigma.tex`: Sections 4.3 and 5.3–5.4 (Reduction to three dimensions,
  Reduction of the integration domain, Importance sampling and quasi-random evaluation).
