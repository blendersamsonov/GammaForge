The computational advantage of xigma comes from separating the electron–laser interaction from the evaluation of the emitted spectrum. Both xigma and delta first characterize each macroparticle by its Lorentz factor $\gamma$, propagation angles $\boldsymbol{\theta}_e$, trajectory-integrated illumination weight, and effective intensity parameter $\hat a$, which determines the nonlinear redshift. These quantities are obtained by integrating the laser field along electron trajectories. Since trajectories are independent, this calculation can be parallelized over particles. In the CuPy implementation, particle chunks are transferred to the GPU, where trajectory sampling and integration are performed concurrently. Chunking limits temporary memory consumption; completed results return to host memory at the stage boundaries.

The two methods differ in how they use these trajectory data. Delta directly evaluates the resonance frequency of every macroparticle for each observation direction and deposits its emission weight into an energy histogram. For illustration, in the head-on, paraxial approximation,

$$
s_{\mathrm R,i}
=\frac{\gamma_i^2}
{1+\hat a_i+\gamma_i^2
|\boldsymbol{\theta}_{e,i}-\boldsymbol{\theta}_{\mathrm{obs}}|^2},
\qquad
s=\frac{\omega}{4\omega_L}.
$$

The finite-bin spectral density is therefore

$$
S_j(\boldsymbol{\theta}_{\mathrm{obs}})
=\frac{1}{\Delta s_j}
\sum_{i=1}^{N_p}
W_i(\boldsymbol{\theta}_{\mathrm{obs}})
\,\mathbf 1_{[s_j,s_{j+1})}(s_{\mathrm R,i}),
$$

where $W_i$ includes the illumination, macroparticle weight, and differential emission factor. Crossing geometry requires the corresponding direction-dependent Doppler correction, but the computational structure remains the same.

The CuPy delta implementation evaluates resonance frequencies and emission weights in a fused float64 GPU kernel and deposits them with a weighted CuPy histogram. It accepts NumPy trajectory data, transfers bounded particle chunks, and returns NumPy spectra. In the finite-cone calculation, observation directions are traversed in a Python loop while the transferred particle chunk is reused. Thus parallelism is primarily over particles within each direction. The implementation has been checked against independent extended-precision CPU emission lines, including crossing geometry and polarization, with additional photon-conservation and bin-edge tests. These checks establish numerical correctness within the tested scope; a measured GPU-to-GPU speedup over delta remains a separate comparison.

Resonance evaluation is independent and readily parallelized; histogram accumulation is less favorable. Different threads can address the same energy bin, particularly near a narrow spectral peak. A straightforward GPU histogram uses atomic additions, which serialize competing updates to the same address. Block-local histograms can reduce global contention, but require additional shared memory and a subsequent merging operation. The precise histogram strategy depends on the CuPy implementation. Fusing the emission arithmetic reduces intermediate arrays and kernel launches, but line generation and histogramming still require device memory traffic and repeated execution for each observation direction. Identifying the relative importance of these costs requires profiling.

The more fundamental cost of delta is the repeated traversal of the particle ensemble. With $N_p$ particles and $N_\Omega$ observation directions, its emission calculation requires approximately $O(N_pN_\Omega)$ resonance evaluations. Each traversal produces all energy bins for one direction, so this cost should not be multiplied by the number of energy bins. Nevertheless, restricting the requested spectrum to a narrow energy interval generally does not eliminate the traversal: the resonance must first be evaluated to determine whether a particle contributes. Much of the work can therefore produce no contribution to the observable of interest.

Fine binning also makes the accuracy requirement more expensive. For independently sampled, equally weighted particles, a bin receiving a fraction $p_j$ of the ensemble has relative statistical uncertainty approximately

$$
\frac{\sigma(S_j)}{\mathbb E[S_j]}
\simeq \frac{1}{\sqrt{N_pp_j}},
\qquad p_j\ll1.
$$

Unequal emission weights further reduce the effective sample count. Narrower bins or weak spectral tails contain fewer contributing particles, so maintaining their accuracy requires increasing $N_p$. GPU acceleration reduces the time per particle, but does not remove this occupancy requirement.

Xigma instead deposits the trajectory data once into the four-dimensional illuminated distribution

$$
\mathcal H(\gamma,\theta_{x,e},\theta_{y,e},\hat a).
$$

Its GPU deposition uses parallel coordinate calculations and weighted accumulation, including cloud-in-cell deposition when selected. Xigma therefore also incurs histogramming costs, but pays them during the construction of a reusable table rather than for every observation direction. Subsequent spectral calculations access this table without revisiting the original particles.

The resonance delta function then permits analytical elimination of the $\gamma$ integral. In the head-on approximation, writing $r=|\boldsymbol{\theta}_e-\boldsymbol{\theta}_{\mathrm{obs}}|$, the required electron energy is

$$
\Gamma(s,r,\hat a)
=\sqrt{\frac{1+\hat a}{s^{-1}-r^2}},
\qquad r^2<s^{-1}.
$$

The spectrum becomes a three-dimensional integral,

$$
S(s,\boldsymbol{\theta}_{\mathrm{obs}})
=\int
K(s,\boldsymbol{\theta}_e,\hat a)\,
\mathcal H\!\left(
\Gamma,\boldsymbol{\theta}_e,\hat a
\right)
\,\mathrm d^2\boldsymbol{\theta}_e\,\mathrm d\hat a,
$$

where $K$ includes the emission factor and resonance Jacobian. The implemented xigma crossing correction modifies both the inverse resonance and its support consistently.

This formulation maps naturally onto the GPU: each output frequency–direction point is assigned a thread block, and threads within the block share its integration work. The table is read-only during this stage. The current kernel accumulates contributions locally in each thread before atomic addition to that block's output value. Thus xigma still uses atomic operations, but avoids a separate scattered histogram update for every contributing particle. Its spectral-stage cost is approximately

$$
O(N_{\mathrm{out}}\,M\,N_{\hat a}),
$$

where $M$ is the angular sample budget and $N_{\hat a}$ is the number of intensity quadrature cells. Once the table exists, this cost is independent of $N_p$.

Efficient integration requires exploiting the geometry of the resonance. At fixed photon frequency, only a restricted range of electron angles can place $\Gamma$ inside the populated electron-energy interval. In the head-on limit this region is an annulus centered on the observation direction. Xigma bounds the relevant radial interval, subdivides it into rings, and determines their intersections with the rectangular angular table. Empty output regions can be rejected before expensive interpolation. For narrow electron-energy distributions, the emitting region can occupy only a small fraction of the full angular domain, making unrestricted Cartesian quadrature wasteful.

Within the retained region, importance sampling concentrates evaluations where the integrand is expected to be largest. For an angular integral $I=\int f(\mathbf x)\,\mathrm d\mathbf x$, the underlying identity is

$$
I=\int\frac{f(\mathbf x)}{p(\mathbf x)}
p(\mathbf x)\,\mathrm d\mathbf x
\approx
\frac{1}{M}\sum_{m=1}^{M}
\frac{f(\mathbf x_m)}{p(\mathbf x_m)}.
$$

Sampling more frequently in a region is compensated by the inverse proposal density. For ordinary independent random sampling, the variance is

$$
\operatorname{Var}(\widehat I)
=\frac{1}{M}
\left[
\int\frac{f(\mathbf x)^2}{p(\mathbf x)}
\,\mathrm d\mathbf x-I^2
\right].
$$

A proposal resembling the nonnegative emission integrand reduces the variation of $f/p$, thereby improving accuracy per evaluation. This variance formula motivates the proposal; it is not an error-bar formula for the deterministic quasi-random implementation.

Xigma constructs an inexpensive proposal from the angular marginal of $\mathcal H$, supplemented by its energy marginal evaluated near the inverse resonance at a representative intensity. This energy information is particularly useful for narrow energy spreads: angular overlap alone can assign substantial effort to locations requiring electron energies absent from the bunch. Coarse radial quadrature estimates the resonance weight, cumulative azimuthal weights determine sample positions, and the sample budget is distributed among arcs according to their estimated contribution. A positive proposal floor and minimum allocation preserve access to regions underestimated by the coarse approximation. The proposal controls computational effort; the full table and physical kernel determine the actual contribution.

Xigma combines this importance transformation with quasi-Monte Carlo sampling. The GPU kernel uses a golden-ratio, Fibonacci-type construction, schematically

$$
u_m=\frac{m+\tfrac12}{M},
\qquad
v_m=\operatorname{frac}(m\varphi),
\qquad
\varphi=\frac{1+\sqrt5}{2}.
$$

The first coordinate is mapped through the cumulative azimuthal proposal. The second selects the radius according to

$$
r_m^2=r_{\min}^2+
v_m(r_{\max}^2-r_{\min}^2).
$$

Uniformity in $r^2$, rather than $r$, supplies the correct polar area measure $r\,\mathrm dr\,\mathrm d\phi$. This construction accepts arbitrary integer sample allocations without requiring each arc's budget to factor into a rectangular grid.

Quasi-random points reduce the accidental clusters and gaps produced by independent random draws. For sufficiently regular, low-dimensional transformed integrands, quasi-Monte Carlo can approach an $M^{-1}$-type error decay, up to logarithmic factors, compared with the $M^{-1/2}$ root-mean-square rate of ordinary Monte Carlo. This is a conditional convergence advantage, not a guaranteed rate for every tabulated spectrum. Importance sampling reduces large-scale variation, while the quasi-random sequence distributes evaluations evenly over what remains. [Owen, *Practical quasi-Monte Carlo integration*](https://artowen.su.domains/mc/practicalqmc.pdf).

[Insert Figure 1 here: Spectral error versus elapsed time on logarithmic axes for delta and xigma, using identical physical inputs, observation directions, and energy-bin boundaries. Compare bin-integrated xigma predictions with delta bin counts against a separately refined reference. Vary particle count for delta and table resolution plus integration budget for xigma. Show both total calculation time and emission-stage time; identify CPU/GPU execution explicitly. Use a numerically checked GPU delta port for the GPU-to-GPU comparison and report compilation separately. This plot should establish the range in which xigma reaches a specified accuracy sooner.]

The expected advantage is greatest for large particle ensembles, many observation directions, and narrow spectral regions requiring good precision. It is not universal: delta can be competitive for a small ensemble or a few directions, and it avoids interpolation bias. Xigma still inherits uncertainty from the sampled bunch and introduces finite-table, trajectory, and quadrature errors. Increasing the quasi-random sample count only reduces the last of these; it cannot recover structure absent from the table.

[Insert Figure 2 here: Two panels showing the same narrow spectral feature at equal computational time and its error under refinement. Overlay delta, xigma, and a converged reference in the first panel. In the second, compare xigma's current importance-sampled quasi-random integration with pseudorandom sampling using the same proposal and with a uniform proposal. Hold the input table fixed for this ablation so that the benefit of sample placement is separated from particle noise and table smoothing.]

These comparisons should assess spectral shape as well as total photon count. A smooth curve alone does not establish accuracy, because interpolation can conceal under-resolution. Xigma's superiority is demonstrated when it reproduces the resolved spectral structure and integrated observables at lower computational cost, with particle, table, and quadrature refinement confirming that the improvement survives beyond visual smoothing.
