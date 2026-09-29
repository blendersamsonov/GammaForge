# DER019 — Deterministic Gaussian source reductions for semi-analytical Compton spectra

Status: derived

## Setup

This derivation migrates and consolidates the earlier ChatGPT working derivation “Unified Semi-Analytical Gaussian Collision and Spectrum Models for GammaForge” into the repository confidence pipeline. The repository derivations cited below remain authoritative for the established overlap, emission-kernel, Doppler, ponderomotive-incidence, trajectory-moment, and finite-line results; DER019 derives the deterministic Gaussian source reductions that connect those pieces.

The physical scope is the current GammaForge first-harmonic, ultra-relativistic ballistic-electron model with Gaussian electron distributions and `GaussianParaxialLaser` fields. “Constant-transverse-width” or “frozen-spot” below is only the lowest-cost closed-form tier; the later focused and flying-focus deterministic tiers retain the full paraxial spot evolution.

## 1. Purpose


GammaForge already has two complementary analytical structures.


The first is the Gaussian overlap theory in DER001/DER002. It calculates total luminosity for realistic Gaussian electron and laser beams with elliptical transverse sizes, rotated focusing axes, hourglass evolution, offsets, crossing angle, and flying focus. Depending on geometry it is closed form, one-dimensional quadrature, or two-dimensional quadrature.


The second is the Xigma emission theory in DER009/DER013/DER015/DER016/DER017. It gives the observer-dependent resonance, exact gamma inversion under the delta-resonance approximation, physical angular/polarization kernel, carrier-weighted trajectory moments, and the second-order finite-line reconstruction.


The missing piece is a bridge between them: a semi-analytical source model that predicts not only total yield and one average nonlinear intensity, but the distribution of the Xigma Stage-0 variables across a Gaussian bunch, without drawing macroparticles.


This document derives that bridge. The intended use is broad parameter optimization and independent validation of Xigma. The general sampled Xigma engine remains the final reference for arbitrary non-Gaussian bunches and arbitrary laser fields.


## 2. What the repository already solves


### 2.1 DER001: general Gaussian overlap without flying focus


DER001 treats the electron and laser transverse profiles through 2x2 covariance matrices. The electron transverse covariance is C_e(z); the laser covariance C_L(u) may be elliptical, astigmatic, and rotated by psi_focus. For head-on beta_ff=0, the time integral is Gaussian and the transverse integrals close, leaving only


    N = const × integral dz

        exp[-z^2/(2 sigma_z,eff^2)]

        / sqrt(det[C_e(z)+C_L(-z)]).


Thus unequal x/y beam sizes, unequal focal positions, unequal Rayleigh ranges, rotated ellipses, and electron hourglass evolution are already handled by the semi-analytical engine.


For round aligned alpha=0 beams, DER001 reduces this one-dimensional integral to the erfcx closed form used by estimate_yield.


DER001 also treats transverse/timing offsets by the same Gaussian completion and derives the crossing-angle geometry. With crossing angle, the fast one-dimensional mode samples slowly varying spot sizes along the nominal axis. The exact mode retains one additional transverse coordinate because u = n0·r depends on that coordinate, producing a two-dimensional quadrature.


DER001 further defines overlap_mean_a0_sq by replacing one laser density with its square. This is already a first nonlinear moment of the collision and provides an important normalization anchor for the new distributional model below.


### 2.2 DER002: flying focus


A flying focus makes the spot-size evaluation point


    u_spot = u + beta_ff c t


depend explicitly on time, so the time integral no longer closes before the spot sizes are evaluated.


DER002 shows that the exact Gaussian geometry still depends on only two independent width arguments. Head-on, the implemented integral is a two-dimensional quadrature over (z,ct), with the transverse plane integrated analytically.


More generally, with both crossing angle and flying focus, the width coefficients depend on the two linear functionals


    A = z,

    B = u_spot.


At fixed (A,B), the remaining two directions in spacetime are Gaussian and close analytically. Therefore the exact luminosity problem is still only two-dimensional. This is an important dimension-counting result: flying focus and crossing angle together do not imply a high-dimensional Monte Carlo overlap calculation.


The current implementation uses the exact 2D flying-focus route and a measured axis-sampling approximation for the residual crossing-angle dependence of u_spot. DER002 identifies a fully exact (A,B) 2D construction as possible future work.


### 2.3 DER009/013/015/012: the general delta-resonance emission kernel


For one electron direction e, one observation direction n, and laser direction n0, define


    F = 1 - e·n0,

    F0 = 1 - n0_z,

    D = F/F0,


and the observer-dependent nonlinear projection


    Q = (1 - n·n0)/(1 - e·n0).


DER015 gives the current nominal resonance


    s_R =

      D Cbar gamma^2

      / [1 + Q ahat + gamma^2 r^2],


where


    r^2 = (theta_ex-theta_x)^2 + (theta_ey-theta_y)^2.


At fixed source coordinates the resonance inversion is exact:


    A_R = 1 + Q ahat,


    Gamma^2 =

      A_R / [D Cbar/s - r^2],


with support


    D Cbar/s > r^2,


and


    |dGamma/ds| =

      D Cbar Gamma^3

      / [2 A_R s^2].


The physical angular emission kernel is the locally transverse dipole model from DER012, which preserves the total Thomson cross section at crossing angle. The reduced gamma-integrated spectral kernel is therefore


    K0 =

      [3/(4 pi)]

      [D Cbar/s^2]

      P(Gamma,e,n)

      Gamma^5

      /

      { A_R [1+Gamma^2 r^2]^2 }.


Here P is the DER012 polarization factor. This formula is the general DER009 kernel with the DER013/DER015 direction factors restored.


A central consequence is that gamma does not need numerical quadrature in the new direct spectra. It should be eliminated analytically with this inverse whenever the electron-energy PDF is known.


### 2.4 DER016/017: trajectory moments and finite-line reconstruction


Current Stage 0 defines, with r(t)=<a^2>(t)/I_pk and carrier ratio C(t),


    Z = integral C r dt,


    a_shape = integral C r^2 dt / Z,


    Cbar = integral C^2 r dt / Z,


    V_a,shape = integral C r^3 dt / Z - a_shape^2,


    V_C = integral C^3 r dt / Z - Cbar^2,


    K_aC,shape = integral C^2 r^2 dt / Z - a_shape Cbar.


After retargeting to a requested peak cycle-averaged intensity I_pk',


    ahat = I_pk' a_shape,

    Var(q) = I_pk'^2 V_a,shape,

    Var(C) = V_C,

    Cov(q,C) = I_pk' K_aC,shape.


DER017 then uses, with B=1+Q ahat+gamma^2 r^2,


    Delta_c =

      Q^2 Var(q)/B^2

      - Q Cov(q,C)/(B Cbar),


    m2 =

      Var(C)/Cbar^2

      + Q^2 Var(q)/B^2

      - 2Q Cov(q,C)/(B Cbar),


and reconstructs the spectrum from rho0, rho1, rho2.


The new semi-analytical models below reproduce these same quantities, rather than introducing a separate nonlinear-width convention.


## 3. The central fixed-width Gaussian theorem


The first new result is more general than the round head-on model developed initially.


Assume:


1. the laser has fixed transverse Gaussian widths during the interaction;

2. the laser longitudinal envelope is Gaussian;

3. one conditions on a fixed electron direction e;

4. the electron trajectory is ballistic;

5. the carrier correction is unchirped, or its dependence can be written as a common function of the completed-square trajectory coordinate.


The laser intensity exponent along any electron trajectory is then a quadratic function of time, even at arbitrary laser crossing angle and with elliptical transverse widths.


Write the electron label vector as xi. For a zero-emittance bunch xi can be the three initial spatial coordinates. More generally one may condition on electron angles and let xi be the remaining Gaussian position variables.


Represent the normalized laser intensity along the trajectory as


    r(t;xi) =

      exp[-E(t,xi)],


with


    E(t,xi)

      = 1/2

        [A xi + a t - d]^T

        W

        [A xi + a t - d].


W is the positive-definite precision matrix of the fixed Gaussian laser in its two transverse coordinates plus longitudinal envelope coordinate. A maps the Gaussian electron labels into those laser coordinates; a is the relative worldline direction; d contains offsets.


Define


    h = a^T W a,


and the rank-two Schur projector


    W_perp =

      W - W a a^T W/h.


Completing the square in time gives


    E(t,xi)

      = 1/2 h [t-t_*(xi)]^2

        + E_min(xi),


where


    E_min(xi)

      = 1/2

        [A xi-d]^T

        W_perp

        [A xi-d].


Therefore


    r(t;xi)

      = A_L(xi)

        exp[-h(t-t_*)^2/2],


with


    A_L(xi) = exp[-E_min(xi)].


This is exact under the fixed-width assumptions. Crossing angle, transverse offsets, timing shifts, elliptical laser widths, and rotated laser axes merely change the matrices A, a, W and the offset d.


Because W_perp removes the one trajectory-time direction from a three-dimensional Gaussian laser profile,


    rank(W_perp)=2,


and hence the quadratic form governing A_L has rank at most two. This rank-two fact is what keeps the nonlinear-amplitude distribution analytically tractable even in general fixed-width crossing geometry.


### 3.1 Exact trajectory power integrals


For the unchirped Gaussian envelope,


    integral r^n dt

      = A_L^n sqrt[2 pi/(n h)].


Thus, for every trajectory at the fixed conditioned electron direction,


    a_shape = A_L/sqrt(2),


    Cbar = 1,


    V_a,shape =

      A_L^2(1/sqrt(3)-1/2),


    V_C = 0,


    K_aC,shape = 0.


Equivalently,


    V_a,shape =

      kappa_G a_shape^2,


    kappa_G =

      2/sqrt(3)-1

      approximately 0.154700538.


The crossing angle changes h and A_L through the Gaussian geometry, but it does not change these normalized temporal constants because the completed-square temporal profile is still Gaussian.


Therefore the original statement that the exact a_shape formulas required head-on incidence was too restrictive. They remain exact for arbitrary fixed-width Gaussian crossing geometry at fixed electron direction.


### 3.2 Common separable carrier profile


The same structure survives a common carrier profile.


Write


    r = A_L g(tau),


with tau the centered trajectory coordinate and C=C(tau). Define


    I_pq = integral C(tau)^p g(tau)^q d tau.


Then


    a_shape = A_L c_a,

    c_a = I_12/I_11,


    Cbar = c_C,

    c_C = I_21/I_11,


    V_a,shape = A_L^2 v_a,

    v_a = I_13/I_11 - c_a^2,


    V_C = v_C,

    v_C = I_31/I_11 - c_C^2,


    K_aC,shape = A_L k_aC,

    k_aC = I_22/I_11 - c_a c_C.


After retargeting,


    Var(q) = kappa_a ahat^2,

    kappa_a = v_a/c_a^2,


    Var(C)=v_C,


    Cov(q,C)=kappa_aC ahat,

    kappa_aC=k_aC/c_a.


This provides a semi-analytical DER016 validation model for temporal chirp that is common to all trajectories. General transverse/spatial chirp does not satisfy this reduction and is deferred.


## 4. Exact luminosity-weighted distribution of the nonlinear amplitude


Let the conditioned electron labels be Gaussian,


    xi ~ N(mu_e, Sigma_e).


The trajectory luminosity in the fixed-width model is proportional to


    F A_L sqrt(2 pi/h).


For fixed electron direction F and h are constants, so the luminosity weighting over xi is exactly multiplication by A_L.


Write the quadratic minimum in canonical form


    E_min(xi)

      = E_0

        + 1/2 (xi-xi_0)^T K (xi-xi_0),


where


    K = A^T W_perp A,


with rank(K)<=2.


The luminosity-weighted electron-label distribution is again Gaussian:


    Sigma_L =

      (Sigma_e^{-1}+K)^(-1),


    mu_L =

      Sigma_L

      [Sigma_e^{-1}mu_e + K xi_0].


Hence all source sampling can be eliminated analytically in the fixed-width model.


### 4.1 Centered aligned case: generalized chi-square of rank at most two


If mu_e=xi_0, define the normalized peak amplitude


    A = A_L/A_max

      = exp(-T),


with


    T =

      1/2

      (xi-xi_0)^T K (xi-xi_0).


Under the luminosity-weighted Gaussian distribution, diagonalize


    M =

      Sigma_L^(1/2)

      K

      Sigma_L^(1/2).


Only at most two eigenvalues are nonzero. Call them mu_1 and mu_2. Then


    T =

      1/2

      [mu_1 z_1^2 + mu_2 z_2^2],


with independent z_i~N(0,1).


Define


    nu_i = 1/mu_i.


For two nonzero eigenvalues the exact density is


    f_T(t)

      =

      sqrt(nu_1 nu_2)

      exp[-(nu_1+nu_2)t/2]

      I_0[ |nu_1-nu_2| t/2 ],


    t>=0.


If only one eigenvalue is nonzero, T is a scaled chi-square with one degree of freedom.


Since


    a_shape = a_max exp(-T),


where


    a_max = c_a A_max,


the exact nonlinear-coordinate density is


    f_a(a)

      =

      f_T[ln(a_max/a)]/a,


    0<a<=a_max.


This formula covers centered fixed-width elliptical beams, rotated ellipses, and arbitrary laser crossing angle. The geometry only changes the two eigenvalues.


### 4.2 Round head-on special case


For a round electron beam with transverse rms sigma_e and round laser intensity width sigma_L, the two eigenvalues are equal. The distribution collapses to the elementary form


    f_a(a)

      =

      (nu/a_max)

      (a/a_max)^(nu-1),


with


    nu =

      1 + sigma_L^2/sigma_e^2,


and, for an unchirped Gaussian pulse,


    a_max = 1/sqrt(2).


The CDF is


    F_a(a)=(a/a_max)^nu.


Thus the luminosity mass in [a_1,a_2] is


    P[a_1,a_2]

      =

      (a_2/a_max)^nu

      -

      (a_1/a_max)^nu.


Every a-moment in a bin is closed:


    M_m[a_1,a_2]

      =

      [nu/(nu+m)] a_max^m

      {

        (a_2/a_max)^(nu+m)

        -

        (a_1/a_max)^(nu+m)

      }.


This gives a genuinely closed deterministic Stage-1 nonlinear axis.


### 4.3 Offsets and noncentral quadratic forms


With transverse/timing misalignment, or after conditioning a correlated Gaussian bunch on a nonzero electron angle, mu_L need not coincide with xi_0. Then T is a noncentral quadratic form of rank at most two.


The source distribution is still particle-free. It can be evaluated by a one-dimensional deterministic method for a noncentral weighted chi-square, for example characteristic-function inversion or a stable dedicated quadratic-form CDF routine.


The first implementation need not expose the most general noncentral analytic density. It can use deterministic one-dimensional quadrature over the two principal Gaussian coordinates while retaining all other integrations analytically. This still eliminates Monte Carlo and provides a clear convergence path.


## 5. Relation to DER001 overlap moments


The new amplitude distribution must reproduce DER001's global overlap moments in every common limit.


Let


    J_n = E[A_L^n]


under the unweighted Gaussian electron-label distribution. Because luminosity weighting contributes one factor A_L,


    E_L[A_L^m]

      = J_(m+1)/J_1.


Therefore


    E_L[a_shape]

      = c_a J_2/J_1,


and


    E_L[a_shape^2]

      = c_a^2 J_3/J_1.


DER001's overlap_mean_a0_sq is exactly the same physical structure as the first expression, including the longitudinal envelope factor. In the fixed-width limit, the new distributional model and DER001 must therefore agree identically for the luminosity-weighted mean nonlinear strength.


This is an important conceptual distinction:


DER001's power-n overlap integrals give global event-weighted intensity moments.


DER016's V_a,shape is a within-trajectory variance.


They are not the same variance. The new model keeps both: the distribution f_a describes trajectory-to-trajectory variation, while V_a,shape describes finite-line variation inside each trajectory.


## 6. Exact particle-free Stage-1 table in the factorized fixed-width model


Assume electron energy and angles are statistically independent of the position variables that determine a_shape. At a transverse waist this is true when alpha_x=alpha_y=0 and the configured gamma correlations with position/angle vanish.


Then the continuous Stage-1 luminosity density factorizes as


    H(gamma,theta_x,theta_y,a,Cbar)

      =

      Y

      f_gamma(gamma)

      f_theta(theta_x,theta_y)

      f_a(a)

      delta(Cbar-c_C).


The total yield Y is the corresponding fixed-width Gaussian overlap yield.


For independent Gaussian gamma and angles, gamma/theta bin masses are Gaussian CDF differences. The a-bin masses are either closed form (round) or one-dimensional deterministic integrals of f_a.


The moment-channel densities are


    H_Va = H kappa_a a^2,


    H_VC = H v_C,


    H_KaC = H kappa_aC a.


Therefore, in the round unchirped case, all four Stage-1 channels are exactly integrable into arbitrary bin edges with no macroparticles and no Stage-0 trajectory integration.


This deterministic ShapeTable is particularly valuable as an independent validation oracle for Xigma Stage 0+1.


## 7. General direct spectrum after eliminating gamma


For optimization it is often better not to build Stage 1 at all.


Let eta denote the remaining semi-analytical source coordinates after the electron energy has been separated. At a given eta we know


    electron direction e(eta),

    luminosity measure dY(eta),

    ahat(eta),

    Cbar(eta),

    and DER016 moment coefficients.


For a requested observation direction n and normalized photon energy s, define


    D = [1-e·n0]/F0,


    Q = [1-n·n0]/[1-e·n0],


    r^2 =

      (theta_ex-theta_x)^2

      +

      (theta_ey-theta_y)^2,


    A_R = 1+Q ahat,


    K = D Cbar.


The resonance root is


    Gamma^2 =

      A_R/[K/s-r^2],


with support K/s>r^2.


If f_gamma(gamma|eta) is the normalized conditional energy PDF, the delta-resonance gamma integral is exact. The zeroth spectral moment is


    rho0(s,n)

      =

      integral dY(eta)

      f_gamma(Gamma|eta)

      [3/(4 pi)]

      [K/s^2]

      P(Gamma,e,n)

      Gamma^5

      /

      { A_R [1+Gamma^2 r^2]^2 }.


This is the master direct-spectrum formula for the new semi-analytical models.


No gamma quadrature is required.


If gamma is independent of eta, the same one-dimensional Gaussian PDF is simply evaluated at Gamma for every deterministic source node.


### 7.1 Direct DER017 moments


At the same root define


    B =

      1+Q ahat+Gamma^2 r^2.


Using the semi-analytical trajectory moments,


    Delta_c =

      Q^2 Var(q)/B^2

      - Q Cov(q,C)/(B Cbar),


    m2 =

      Var(C)/Cbar^2

      + Q^2 Var(q)/B^2

      - 2Q Cov(q,C)/(B Cbar).


Then the same source integrand accumulates


    rho1: multiply the rho0 integrand by s Delta_c,


    rho2: multiply the rho0 integrand by s^2 m2.


The final moment-2 spectrum is reconstructed with the existing DER017 operator


    S(s)

      =

      rho0

      -

      (1/s) d[s rho1]/ds

      +

      (1/(2s)) d^2[s rho2]/ds^2.


Thus the proposed semi-analytical path can validate not only Xigma line centres but also all three current spectral-moment channels.


## 8. Zero-emittance spectra: 0D/1D models


For a mono-directional electron bunch, e is fixed. Therefore D, Q, r, the polarization geometry, and the fixed-width amplitude distribution are all independent of gamma.


If the bunch has finite Gaussian energy spread but no energy-position correlation, the direct spectrum is only the one-dimensional integral over a_shape,


    rho0(s,n)

      =

      Y integral da f_a(a)

        f_gamma[Gamma(s,a)]

        K0(s,a).


This remains true at arbitrary laser crossing angle and arbitrary observation direction within the validity of the near-electron Xigma emission kernel.


For a round head-on collision observed on axis,


    D=Q=1,

    r=0,


and, for unchirped Cbar=1,


    Gamma(s,a)

      =

      sqrt[s(1+I_pk a)].


This is the simplest 1D semi-analytical spectrum with finite energy spread.


### 8.1 Fully closed monoenergetic line broadened only by nonlinear intensity


For gamma=gamma_0 exactly, the a integral is also removed by the resonance.


For general fixed electron direction and observation geometry,


    a_s

      =

      [K gamma_0^2/s

       - 1

       - gamma_0^2 r^2]

      /

      [Q I_pk],


provided Q>0.


The Jacobian is


    |da/ds|

      =

      K gamma_0^2

      /

      [Q I_pk s^2].


Therefore


    rho0(s,n)

      =

      Y

      [3/(2 pi)]

      P(gamma_0,e,n)

      gamma_0^2

      /

      [1+gamma_0^2r^2]^2

      ×

      f_a(a_s)

      ×

      K gamma_0^2

      /

      [Q I_pk s^2].


This is a true closed-form spectrum whenever f_a itself is closed, as in the round fixed-width model.


The support is the image of 0<a<=a_max under the resonance map.


If Q=0, observation is along the laser propagation direction and the nonlinear coordinate does not shift the resonance. That case must be handled as a delta-line limit rather than through the formula above.


## 9. Finite emittance without Monte Carlo


Finite emittance does not destroy the fixed-width Gaussian reduction. It changes which Gaussian variables are conditioned before the quadratic-form amplitude distribution is constructed.


Let the transverse phase-space vector be jointly Gaussian. Condition the position variables xi on a chosen electron angle vector theta_e. The conditional distribution


    xi | theta_e


is Gaussian with analytically known mean and covariance. For that fixed direction, the laser intensity history is still quadratic in time and the previous theorem applies. Therefore one obtains an exact conditional amplitude density


    f_a(a | theta_e),


which is generally noncentral rank-two quadratic-form statistics.


The direct spectrum becomes


    rho0

      =

      integral d theta_e

      f_theta(theta_e)

      integral da

      f_a(a|theta_e)

      f_gamma(Gamma|theta_e,a)

      K0.


No macroparticles are required.


### 9.1 Round head-on bunch: two-dimensional quadrature


For a round beam and head-on laser, rotational symmetry makes the conditional source depend on electron-angle magnitude theta but not its azimuth.


The radial angular density is


    f_theta(theta)

      =

      theta/sigma_theta^2

      exp[-theta^2/(2 sigma_theta^2)].


The exact head-on geometry gives


    e_z = 1/sqrt(1+theta^2),


    D(theta) =

      [1+e_z]/2,


    Q(theta)=1/D(theta)


for on-axis observation.


The spectrum is therefore a two-dimensional deterministic integral over


    (theta, a).


Importantly, unlike the first draft, this formulation does not need to neglect transverse electron drift through the fixed-width laser. The angle-dependent trajectory width and the position-angle Twiss correlation are already absorbed into f_a(a|theta).


For nonzero round Twiss alpha the conditional position mean shifts along the electron-angle vector, making f_a noncentral, but the rotational symmetry still leaves only the same two integration variables.


### 9.2 Elliptical or crossed finite-emittance bunch: three-dimensional quadrature


When x/y angular spreads differ, or a laser crossing angle breaks rotational symmetry, retain both angular components explicitly.


The spectrum is then


    integral dtheta_x dtheta_y da


with f_a(a|theta_x,theta_y) supplied by the conditional Gaussian quadratic-form construction.


This 3D model can include:


- unequal x/y emittances;

- nonzero alpha_x and alpha_y;

- rotated elliptical laser widths;

- fixed-width laser crossing angle;

- arbitrary polarization orientation;

- transverse and timing offsets;

- general observation direction within Xigma's small electron-observer-angle regime.


It remains deterministic and still eliminates the electron-energy dimension analytically.


If gamma is linearly correlated only with theta_x/theta_y, use the conditional Gaussian f_gamma(gamma|theta_x,theta_y) and the gamma inversion remains exact. Position-gamma correlations are more complicated because conditioning only on the scalar a loses the sign/orientation information of the position. Those correlations should be deferred or represented by one additional deterministic coordinate.


## 10. Polarization and Stokes outputs


The semi-analytical spectra should not use the superseded unprojected crossing-angle polarization formula from DER005/DER006.


Use the physical local transverse-dipole construction of DER012. At each deterministic source node and resonance root Gamma, project the rotated laser polarization basis onto the plane perpendicular to the electron direction and evaluate the same physical polarization factor P used by Xigma.


For selected observation points, the same deterministic quadrature can also accumulate the Stokes I,Q,U,V expressions of DER007/DER012 rather than only total intensity.


This makes the semi-analytical path useful for validating Xigma polarization and Stokes calculations at finite emittance and crossing angle, not only its scalar spectrum.


## 11. What focusing changes


The exact fixed-width theorem fails once the transverse spot matrix W itself changes appreciably along an electron trajectory.


That is the real boundary between the closed/quadratic-form models and the deterministic trajectory models.


For beta_ff=0, DER001 already evaluates the total luminosity with a 1D head-on or exact 2D crossed-beam quadrature because the spacetime Gaussian integrals can be rearranged globally.


However Stage-0 quantities such as


    a_shape =

      integral r^2 dt / integral r dt


are nonlinear ratios attached to individual electron trajectories. One cannot obtain their distribution merely by taking ratios of global DER001 overlap integrals.


Therefore the focused spectral extension needs to preserve a small set of electron source labels and evaluate a short 1D trajectory quadrature at deterministic nodes.


### 11.1 Round head-on stationary focus: two source dimensions


For a round head-on focused Gaussian pulse and zero transverse emittance, cylindrical symmetry leaves two electron source labels:


    R = transverse impact radius,

    z0 = longitudinal bunch coordinate / collision timing coordinate.


For every deterministic (R,z0) node, compute the current DER016 trajectory integrals in one smooth 1D time quadrature:


    Z,

    a_shape,

    Cbar,

    V_a,shape,

    V_C,

    K_aC,shape.


Then integrate the source nodes with their Gaussian bunch weights and Stage-0 luminosity.


The resulting particle-free spectrum is a two-dimensional deterministic source quadrature per spectral point after the source map has been prepared. Gamma is still eliminated analytically.


The total source luminosity must agree with DER001's exact 1D overlap_yield for the same geometry. That gives an unusually strong independent normalization check.


### 11.2 Elliptical or crossed stationary focus: three source dimensions


If cylindrical symmetry is broken by elliptical focusing, transverse rotation, offset, or crossing angle, retain


    (x0,y0,z0)


as the deterministic source labels for a zero-emittance bunch.


Each node again uses a 1D trajectory quadrature, after which gamma is removed analytically in the spectrum.


This is a three-dimensional deterministic source model, not a six-dimensional particle Monte Carlo.


The source-integrated luminosity should be checked against DER001's exact 2D crossing-angle overlap mode. If the faster DER001 1D crossing approximation is selected, the semi-analytical spectrum should report that approximation explicitly rather than silently mixing exact and approximate geometry.


## 12. Flying focus


Flying focus deserves separate treatment because DER002's low-dimensional total-yield reduction does not automatically imply the same dimensionality for Stage-0 trajectory ratios.


DER002 can integrate the total luminosity in two dimensions because, at fixed values of the two width arguments, the remaining spacetime coordinates are Gaussian and can be eliminated.


But a_shape, Cbar, and the DER016 variances are ratios of integrals along one specific electron trajectory. Gaussian elimination across different electrons would mix trajectories before these ratios are formed and is therefore not valid.


For spectra, a deterministic source map is still feasible.


### 12.1 Round head-on flying focus, zero emittance


For a round head-on flying focus and zero transverse emittance, cylindrical symmetry again leaves


    (R,z0)


as source coordinates.


At each node use the true flying-focus intensity history


    u_spot =

      -z(t)+beta_ff c t


and evaluate the DER016 moments with a 1D trajectory quadrature.


The prepared source map is therefore two-dimensional. Its total luminosity must reproduce DER002's verified 2D overlap_yield.


This model can test, spectrally rather than only in total yield, the beta_ff=1 synchronization optimum and the beta_ff <-> 1/beta_ff reciprocal symmetry identified by DER002 in the short-bunch limit.


### 12.2 Elliptical or crossed flying focus


Without cylindrical symmetry, a zero-emittance spectral source map generally needs


    (x0,y0,z0),


plus the inner 1D trajectory quadrature.


Thus the natural first exact deterministic spectral implementation is 3D in source coordinates.


DER002 shows that the total yield with crossing angle plus flying focus can in principle remain a 2D (A,B) overlap quadrature, but that dimensional reduction integrates across electron labels. It cannot be reused blindly for the per-trajectory nonlinear ratios.


A future derivation may find a more economical conditional representation, but the 3D deterministic source map is already far cheaper and more controlled than millions of random trajectories.


## 13. Existing angle-integrated spectrum and the new collimated spectra


DER011 already supplies a closed-form single-electron angle-integrated linear Thomson spectrum. The current AnalyticalEngine evaluates its finite-energy-spread result with a one-dimensional gamma quadrature and normalizes it to the analytical overlap yield.


The new models are complementary rather than replacements.


DER011 is best for:


- angle-integrated linear spectral shape;

- fast bandwidth estimates;

- a simple cross-check of total spectral normalization.


The new delta-resonance models are best for:


- on-axis/collimated spectra;

- nonlinear redshift distribution across the bunch;

- finite emittance and selected observation directions;

- Xigma Stage-2 validation;

- DER017 moment-channel validation.


Where both apply in the linear limit, integrating the new angle-resolved model over solid angle must reproduce the DER011 total spectrum and the analytical overlap yield.


## 14. Model hierarchy for the semi-analytical engine


The recommended design is an explicit hierarchy. Different models should coexist and advertise their assumptions.


Tier 0A — existing closed-form total yield:

round, aligned, head-on, stationary focus; DER001 erfcx formula.


Tier 0B — new closed-form collimated spectrum:

fixed-width Gaussian, monoenergetic, mono-directional; round centered source gives elementary f_a and no quadrature.


Tier 1A — existing general head-on overlap:

elliptical/astigmatic/focused Gaussian collision; DER001 1D z quadrature.


Tier 1B — new direct spectrum:

fixed-width source, zero emittance, finite Gaussian energy spread; one a quadrature because gamma is inverted analytically.


Tier 2A — existing exact overlap:

crossing angle or flying focus; DER001/DER002 2D quadrature.


Tier 2B — new finite-emittance spectrum:

round fixed-width bunch; two-dimensional (theta,a) deterministic quadrature, including nonzero round Twiss alpha through conditional Gaussian statistics.


Tier 2C — new focused round spectrum:

round head-on focused/flying-focus pulse, zero emittance; two-dimensional deterministic source grid plus a small 1D trajectory preparation integral.


Tier 3A — new anisotropic/crossed fixed-width spectrum:

(theta_x,theta_y,a) deterministic quadrature.


Tier 3B — new focused anisotropic/crossed/flying source:

(x0,y0,z0) deterministic source grid plus a small 1D trajectory integral, for zero emittance.


Beyond these tiers, adding full anisotropic emittance to a focused/crossed/flying source can raise the deterministic dimension to four or five. At that point adaptive Xigma may become the better computational choice.


## 15. Quadrature coordinates


The new deterministic models should use quadratures matched to their probability measures.


For the round power-law nonlinear coordinate,


    u = F_a(a)

      = (a/a_max)^nu


is uniform on [0,1], so


    a = a_max u^(1/nu).


Gauss-Legendre in u avoids endpoint conditioning problems.


For the general central rank-two quadratic form, either integrate over t=-ln(a/a_max) with the Bessel density or use a stable generalized-chi-square CDF/quadrature.


For round Gaussian divergence,


    x = theta^2/(2 sigma_theta^2)


is exponentially distributed, making Gauss-Laguerre natural.


For independent Gaussian theta_x/theta_y, Gauss-Hermite is natural.


For focused source maps, transform Gaussian source coordinates to Hermite variables or Gaussian CDF coordinates and use tensor Gauss-Hermite, sparse grids, or nested deterministic cubature.


The electron-energy dimension should not be quadratured in the delta-resonance models. Evaluate the Gaussian or conditional-Gaussian energy PDF at the DER015 root Gamma.


## 16. Validity diagnostics


The engine should report why a model was selected and how strongly its approximations are expected to hold.


For the fixed-width approximation define, at minimum,


    epsilon_L =

      L_int/z_R,


    epsilon_beta =

      L_int/beta_e*,


    epsilon_drift =

      sigma_theta L_int/sigma_L.


The fixed-width model is best when these are small.


For crossing angle, also report whether the fast DER001 one-dimensional overlap approximation or the exact two-dimensional mode is being used.


For flying focus, do not silently use the unsafe 1D frozen-width approximation from DER002. The verified path is two-dimensional for the total overlap and a deterministic trajectory map for spectral moments.


For the emission kernel, report that the near-electron observation-angle approximation remains the same one used by Xigma. Crossing angle of the laser is supported by D, Q, and DER012 polarization, but this does not imply unrestricted large electron-observer angle validity.


For first-harmonic spectra, the existing higher-harmonic diagnostic remains applicable as a separate validity condition.


## 17. Implementation plan for the AnalyticalEngine


Keep these additions in the semi-analytical engine. Do not modify Xigma physics to implement them.


Recommended new internal primitives:


    fixed_width_trajectory_reduction(...)

        -> h, quadratic K, offsets, temporal constants


    nonlinear_shape_distribution(...)

        -> round closed form or rank-two deterministic density/CDF


    conditional_shape_distribution(theta_x,theta_y,...)

        -> f_a(a|theta)


    direct_spectral_moments(...)

        -> rho0, rho1, rho2


    direct_collimated_spectrum(...)

        -> delta or moment2 spectrum


    deterministic_shape_table(...)

        -> optional particle-free Stage-1 validation table


    focused_source_map(...)

        -> deterministic nodes with luminosity and DER016 moments


Model selection should be explicit, for example through a model enum such as


    "fixed_round_closed",

    "fixed_zero_emittance",

    "fixed_round_emittance",

    "fixed_general_emittance",

    "focused_round",

    "focused_general".


The existing overlap_yield functions remain the absolute normalization and geometry references.


Do not mix a focused DER001 yield with a fixed-width nonlinear distribution and label the combination exact. A hybrid approximation may be useful for optimization, but it must be named and its components reported.


## 18. Validation program


The unified hierarchy should be validated by reduction identities before comparing with Xigma.


1. Fixed-width Gaussian identity:

For arbitrary elliptical widths and crossing angle, compare the completed-square single-trajectory intensity history with direct LaserField evaluation when the widths are frozen.


2. DER001 normalization:

Integrate the new A_L distribution and verify total luminosity against DER001's constant-width Gaussian overlap, including crossing angle and rotated ellipses.


3. DER001 nonlinear mean:

Verify E_L[a_shape] against overlap_mean_a0_sq in the same fixed-width limit.


4. Round reduction:

Verify the rank-two Bessel density collapses exactly to the power law when the two eigenvalues are equal.


5. Stage-0 moment oracle:

Compare a_shape, Cbar, V_a, V_C, K_aC from the closed/separable formulas against over-resolved Xigma Stage 0.


6. Stage-1 oracle:

Construct the deterministic particle-free ShapeTable and compare H plus all three moment channels against Xigma as particle count and bin resolution increase.


7. Emittance reduction:

For a round beam verify the 3D theta_x/theta_y/a model reduces to the 2D theta/a model.


8. Crossing-angle emission:

Compare direct semi-analytical spectra with Xigma at finite crossing angle using the same D, Q, and DER012 polarization geometry.


9. Focused hierarchy:

Compare fixed-width spectra with the focused deterministic source map while scanning epsilon_L and epsilon_drift. This maps the validity boundary of the cheap model.


10. Flying focus:

Verify source-integrated luminosity against DER002 and test the beta_ff=1 optimum and reciprocal symmetry spectrally.


11. DER011 linear limit:

In the linear regime, angle-integrate the new direct spectrum and compare with the existing angle-integrated analytical spectrum.


12. DER017:

Compare rho0, rho1, rho2 and the reconstructed moment2 spectrum against Xigma's direct-particle and table implementations.


## 19. Deferred extensions


General spatial carrier chirp is deferred because C then varies across source coordinates and may destroy the common temporal constants. The deterministic focused-source framework can still handle it later by evaluating C(t) at each source node.


General gamma-position correlations are deferred from the lowest-dimensional direct spectra. Gamma-angle correlations are easy because the conditional gamma PDF remains Gaussian at fixed angle; position-gamma correlations may require one additional source coordinate.


Quantum recoil is deferred from this document, although the existing recoil-corrected resonance remains analytically invertible and should fit naturally into the same direct-root architecture.


Strong higher harmonics remain outside the first-harmonic semi-analytical model.


Arbitrary non-Gaussian laser aberrations and measured non-Gaussian electron bunches remain the domain of Xigma and its adaptive-sampling path.


## 20. Main comparison with the earlier draft


The earlier ChatGPT derivation was directionally correct but unnecessarily narrow.


What remains correct:


- particle-free Stage-1 construction is possible in restricted Gaussian geometries;

- the round head-on nonlinear-coordinate distribution is a power law;

- gamma should be eliminated analytically with DER015 rather than quadratured;

- finite round emittance leads naturally to a 2D deterministic spectrum;

- focused/flying-focus cases are still far cheaper with deterministic source quadrature than with millions of random trajectories.


What changes after comparison with the repository derivations:


- elliptical Gaussian beams are already first-class analytical objects through covariance matrices in DER001;

- fixed-width crossing angle does not break the closed trajectory-moment reduction; after completing the square in time it is still one common Gaussian temporal profile times a trajectory amplitude;

- the nonlinear-amplitude quadratic form has rank at most two, so an analytic rank-two distribution exists even for centered elliptical/crossed fixed-width geometry;

- finite emittance can be handled more cleanly by conditioning the Gaussian position distribution on electron angle, avoiding the earlier ad hoc no-transverse-drift assumption;

- crossing-angle spectra can use the current verified D, Q, and DER012 physical polarization kernel rather than being deferred wholesale;

- DER002's flying-focus 2D luminosity result should be used as a normalization reference, but its dimensional reduction cannot be transferred directly to per-trajectory nonlinear ratios;

- the focused and flying-focus spectral models therefore remain deterministic source maps, typically 2D round or 3D anisotropic at zero emittance.


The unified picture is consequently broader and better connected to the production physics than the first draft.


## 21. Working conclusion


GammaForge can support a genuine semi-analytical hierarchy spanning much more than total yield.


The existing repository already provides a mature Gaussian geometry engine:

closed form in the simplest round case, 1D for general head-on stationary focusing, and 2D for exact crossing-angle and flying-focus overlap.


The new contribution is to attach a deterministic distribution of Xigma's trajectory variables to that geometry.


In the fixed-width Gaussian limit, the laser history along a conditioned electron direction is exactly a completed-square Gaussian in time. The entire DER016 trajectory reduction is therefore controlled by one peak-amplitude variable. Its luminosity-weighted distribution is a rank-at-most-two Gaussian quadratic form. Round beams reduce to an elementary power law; centered elliptical/crossed beams have a closed Bessel-form density.


That gives:

- a true closed spectrum for monoenergetic zero-emittance special cases;

- 1D direct spectra with finite energy spread;

- 2D spectra for round finite emittance, including nonzero round Twiss alpha;

- 3D spectra for anisotropic emittance and general fixed-width crossing geometry;

- exact or deterministic particle-free Stage-1 tables for validation.


When focusing or flying focus makes the spot size evolve along a trajectory, the closed amplitude distribution is lost, but the problem remains low-dimensional. Round head-on cases use a 2D deterministic source map; anisotropic/crossed zero-emittance cases use a 3D map; every node needs only a smooth 1D trajectory quadrature. DER001/DER002 provide independent absolute-yield checks for these source maps.


This is an attractive optimization architecture:

use the cheapest valid semi-analytical tier for broad design searches; use richer deterministic tiers to test sensitivity and validate Xigma; then use adaptive/full Xigma only near the optimum or for genuinely non-Gaussian measured inputs.


## 22. Further analytic reduction: exact nonlinear-shape moments


The unified quadratic-form model gives more than the full density f_a. It gives the nonlinear trajectory-shape moments in closed form, which directly removes one empirical element from the current AnalyticalEngine.


Write the fixed-width trajectory amplitude as


    A_L(xi)

      =

      A_max

      exp[

        -1/2

        (xi-xi_0)^T K (xi-xi_0)

      ],


with Gaussian electron labels


    xi ~ N(mu, Sigma).


Define


    delta = mu-xi_0.


For any positive integer n,


    J_n = E[A_L^n]


is the standard Gaussian quadratic-form integral


    J_n

      =

      A_max^n

      det(I+n Sigma K)^(-1/2)

      exp{

        -n/2

        delta^T

        K

        (I+n Sigma K)^(-1)

        delta

      }.


This formula includes centered/offset elliptical beams and arbitrary fixed-width crossing geometry through K and delta.


Because the luminosity weight is proportional to A_L,


    E_L[A_L^m]

      =

      J_(m+1)/J_1.


For the separable trajectory reduction


    a_shape = c_a A_L,


the exact luminosity-weighted trajectory-to-trajectory moments are therefore


    <a_shape>_L

      =

      c_a J_2/J_1,


    <a_shape^2>_L

      =

      c_a^2 J_3/J_1,


    Var_L(a_shape)

      =

      c_a^2

      [

        J_3/J_1

        -

        (J_2/J_1)^2

      ].


After retargeting,


    sigma_ahat^2

      =

      I_pk^2 Var_L(a_shape).


This is exactly the quantity that the current AnalyticalEngine does not know: estimate_spectrum_width currently brackets std(ahat)/<ahat> empirically with NONLINEAR_BROADENING_RANGE=(0.06,1.12).


For every geometry covered by the fixed-width quadratic model, that empirical bracket can be replaced by the exact expression above.


### 22.1 Round closed form for the relative nonlinear spread


For the round centered power-law distribution


    f_a(a)

      =

      (nu/a_max)

      (a/a_max)^(nu-1),


the raw moments are


    <a^m>

      =

      [nu/(nu+m)]

      a_max^m.


Hence


    <a>

      =

      [nu/(nu+1)] a_max,


    <a^2>

      =

      [nu/(nu+2)] a_max^2,


and the relative trajectory-to-trajectory spread is


    sigma_a/<a>

      =

      1/sqrt[nu(nu+2)].


With


    nu =

      1+sigma_L^2/sigma_e^2,


this is a fully closed expression for the nonlinear broadening caused by transverse sampling of a Gaussian laser.


It has the expected limits:


    sigma_e << sigma_L:

        nu >> 1,

        sigma_a/<a> -> 0

        because the bunch sees nearly uniform intensity;


    sigma_e >> sigma_L:

        nu -> 1,

        sigma_a/<a> -> 1/sqrt(3)

        approximately 0.577.


The larger empirical values seen in focused Xigma scenarios are therefore a diagnostic of longitudinal spot-size evolution/hourglass effects rather than a contradiction of the fixed-width theory.


### 22.2 Combine trajectory-to-trajectory spread with DER016 finite-line variance


There are two distinct nonlinear variances.


The first is the spread of trajectory means,


    Var_L(a_shape).


The second is the finite-line variance inside each trajectory,


    V_a,shape

      =

      v_a A_L^2.


Its luminosity-weighted mean is


    <V_a,shape>_L

      =

      v_a J_3/J_1.


The law of total variance therefore gives the complete luminosity-weighted instantaneous nonlinear variance represented by the separable model:


    Var_total(q)/I_pk^2

      =

      <V_a,shape>_L

      +

      Var_L(a_shape)


      =

      v_a J_3/J_1

      +

      c_a^2

      [

        J_3/J_1

        -

        (J_2/J_1)^2

      ].


This decomposition is useful for interpreting bandwidth diagnostics:


- Var_L(a_shape) broadens the distribution of nominal resonances across electrons;

- <V_a,shape>_L is the unresolved finite-pulse line variance carried by DER016/DER017.


The present empirical nonlinear FWHM term mixes these effects only heuristically. The new semi-analytical engine can report them separately.


## 23. Exact nonlinear angle-integrated single-electron spectrum in the head-on Q≈1 model


The current angle_integrated_spectrum applies a nonlinear redshift by replacing


    y = s/gamma^2


with


    y = s(1+ahat)/gamma^2


inside the linear DER011 polynomial and later renormalizing the full spectrum to the total yield.


That compresses the linear spectrum but it is not the exact angular Jacobian of the nonlinear resonance.


Within the same head-on paraxial model used by DER011, retain a constant nonlinear shift


    h = ahat


and use the nominal resonance


    s

      =

      gamma^2

      /

      (1+h+u),


    u = gamma^2 theta^2.


For the azimuthally averaged physical dipole kernel,


    Pbar(u)

      =

      1

      -

      2u/(1+u)^2.


Define


    z = s/gamma^2.


Then


    u

      =

      1/z

      -

      1

      -

      h,


and


    ds/du

      =

      -gamma^2

      /

      (1+h+u)^2.


Performing the angular delta-function integral gives


    dN/ds

      =

      L

      G(z;h)

      /

      gamma^2,


where


    G(z;h)

      =

      (3/2)

      1/(1-hz)^2

      [

        1

        -

        2z[1-(1+h)z]/(1-hz)^2

      ],


with support


    0 < z <= 1/(1+h).


This shape obeys the exact normalization identity


    integral_0^[1/(1+h)]

      G(z;h) dz

      =

      1.


At h=0 it reduces exactly to DER011:


    G(z;0)

      =

      (3/2)

      [1-2z(1-z)].


Thus a nonlinear redshift does not merely compress the linear DER011 shape; the angular Jacobian changes as well.


If one compares with a separately normalized compressed-linear approximation


    G_comp(z;h)

      =

      (1+h)

      (3/2)

      {

        1

        -

        2(1+h)z[1-(1+h)z]

      },


the leading discrepancy is


    G-G_comp

      =

      (3/2)

      h

      (2z-1)^3

      +

      O(h^2).


This is first order in h and can therefore be visible well before the first-harmonic model itself fails.


### 23.1 Average the nonlinear angle-integrated spectrum over the Gaussian collision


In the round fixed-width model write


    a_shape = a_max x,


    0<x<=1,


with


    f_x(x)

      =

      nu x^(nu-1),


and define


    chi = I_pk a_max,


so the physical nonlinear shift is


    h=chi x.


For a fixed gamma and spectral coordinate z=s/gamma^2, only trajectories satisfying


    z <= 1/(1+chi x)


contribute. Therefore


    0 <= x <= X(z),


where


    X(z)

      =

      min[

        1,

        max(

          0,

          [1/z-1]/chi

        )

      ].


The collision-averaged single-gamma shape is


    Gbar_nu(z;chi)

      =

      integral_0^[X(z)]

      nu x^(nu-1)

      G(z;chi x)

      dx.


This is a one-dimensional deterministic integral. Because the integrand is a power times rational functions of x, it can also be written in terms of Gauss hypergeometric functions, but a small stable quadrature is probably preferable in production.


For finite Gaussian energy spread,


    dN/ds

      =

      Y

      integral dgamma

      f_gamma(gamma)

      Gbar_nu(s/gamma^2;chi)

      /

      gamma^2.


Thus the round fixed-width nonlinear angle-integrated spectrum requires only two small deterministic dimensions, and only one for a monoenergetic beam.


For the general rank-two amplitude distribution, replace the power-law x integral by the deterministic f_a integral derived earlier.


This is a direct upgrade path for AnalyticalEngine.SPECTRUM and removes both the single-mean-ahat approximation and the empirical nonlinear spread bracket in its supported geometry.


## 24. Energy-integrated angular distribution and exact circular-aperture capture


Integrating over photon energy removes the resonance delta function. The angular distribution of a single scattering event is therefore independent of the nonlinear line shift and of the laser temporal envelope; those quantities determine photon energy, not the normalized Thomson angular probability.


For one electron with Lorentz factor gamma and observation offset theta, define


    u = gamma^2 theta^2.


The energy-integrated angular probability is


    dP/dOmega

      =

      [3/(2 pi)]

      gamma^2

      P(theta,phi)

      /

      (1+u)^2,


with P the physical DER012 transverse-dipole factor.


For head-on geometry after azimuthal averaging,


    Pbar(u)

      =

      1

      -

      2u/(1+u)^2.


Using


    dOmega

      =

      pi du/gamma^2,


the normalized radial density is


    dP/du

      =

      (3/2)

      1/(1+u)^2

      [

        1

        -

        2u/(1+u)^2

      ].


It integrates to one.


### 24.1 Closed circular-aperture fraction


For a circular aperture of half-angle theta_c centered on the electron direction,


    u_c = gamma^2 theta_c^2.


The captured photon fraction is exactly


    F_cap(u_c)

      =

      integral_0^[u_c] dP/du du


      =

      u_c

      [2u_c^2+3u_c+3]

      /

      [2(1+u_c)^3].


Useful limits are


    u_c << 1:

        F_cap

          =

          (3/2)u_c

          +

          O(u_c^2),


    u_c >> 1:

        F_cap -> 1.


This result is independent of ahat and of the nonlinear-intensity distribution.


For a zero-emittance bunch with energy spread but energy-independent luminosity, the total circular-collimator efficiency is only a one-dimensional gamma average,


    eta_cap

      =

      E_gamma[

        F_cap(

          gamma^2 theta_c^2

        )

      ].


For a monoenergetic beam it is closed form.


This is directly useful for fast collimation scans and gives an independent check of Xigma's angle-integrated target yield.


### 24.2 AnalyticalEngine angular distribution


The same formula provides a natural new AnalyticalEngine output.


At zero emittance with finite energy spread,


    dN/dOmega

      =

      Y

      integral dgamma

      f_gamma(gamma)

      [3/(2 pi)]

      gamma^2

      P(gamma,n)

      /

      [1+gamma^2 r^2]^2.


This is a one-dimensional deterministic gamma quadrature for every requested angular point.


At finite laser crossing angle, use the current DER012 locally transverse polarization basis. The total yield already contains the encounter-flux/collision-geometry reduction; the energy-integrated angular probability remains normalized to one.


Finite electron emittance adds only deterministic electron-angle quadrature. This makes ANGULAR_DISTRIBUTION a particularly clean semi-analytical validation target because it is independent of the nonlinear redshift model.


### 24.3 Circular-aperture spectrum


The exact nonlinear single-electron angle-integrated spectrum of Section 23 can also be restricted to a circular aperture.


For fixed gamma and h, the resonance maps photon energy to


    u(z,h)

      =

      1/z

      -

      1

      -

      h.


The aperture condition u<=u_c is equivalent to


    1/(1+h+u_c)

      <=

      z

      <=

      1/(1+h).


Therefore the spectrum inside the cone is simply


    dN_cap/ds

      =

      L

      G(z;h)

      /

      gamma^2


on that restricted support and zero outside.


Its integral is exactly


    F_cap(u_c),


independent of h.


This identity is a strong regression test: the nonlinear redshift changes where photons lie in energy but cannot change how many enter a fixed angular cone in this model.


For the round Gaussian collision, averaging this aperture spectrum over f_a and f_gamma is still only a small deterministic quadrature.


## 25. O(1) on-axis centroid and RMS bandwidth for rapid optimization


A broad optimizer often needs only a line centroid and bandwidth, not a full spectrum grid.


For the round fixed-width model define


    x = a_shape/a_max,


    f_x(x)

      =

      nu x^(nu-1),


    chi = I_pk a_max.


For a zero-emittance observation on the electron axis,


    Q=1


exactly for any nonsingular laser incidence, and in the central-direction normalization D=1. Let the common carrier mean be c_C.


The nominal resonance is


    s_R

      =

      c_C gamma^2

      /

      (1+chi x).


The photon line mass per unit solid angle on axis is proportional to gamma^2. Define


    G_p = E[gamma^p]


under the luminosity-weighted electron-energy PDF, assumed independent of x.


Also define


    R_m(chi,nu)

      =

      E_x[

        (1+chi x)^(-m)

      ]


      =

      _2F1(

        m,

        nu;

        nu+1;

        -chi

      ).


Then the normalized nominal on-axis spectral moments are


    <s>

      =

      c_C

      [G_4/G_2]

      R_1,


    <s^2>

      =

      c_C^2

      [G_6/G_2]

      R_2.


Therefore


    sigma_s^2

      =

      <s^2>

      -

      <s>^2


is available without evaluating any spectral grid.


For an ordinary Gaussian gamma distribution with mean gamma_0 and standard deviation sigma_gamma,


    G_2

      =

      gamma_0^2

      +

      sigma_gamma^2,


    G_4

      =

      gamma_0^4

      +

      6 gamma_0^2 sigma_gamma^2

      +

      3 sigma_gamma^4,


    G_6

      =

      gamma_0^6

      +

      15 gamma_0^4 sigma_gamma^2

      +

      45 gamma_0^2 sigma_gamma^4

      +

      15 sigma_gamma^6.


These expressions are exact for the Gaussian model itself.


If introducing a special-function dependency is undesirable, R_1 and R_2 can be evaluated by a tiny one-dimensional Gauss-Legendre quadrature over x. The cost is still effectively O(1) for optimization purposes.


### 25.1 Include the unchirped DER017 second-order line moments


For the unchirped separable Gaussian pulse,


    kappa_G

      =

      2/sqrt(3)-1,


and on axis


    Delta_c

      =

      m_2

      =

      kappa_G

      h^2/(1+h)^2,


with


    h=chi x.


Define


    T_(m,k)

      =

      E[

        x^k

        (1+chi x)^(-m)

      ]


      =

      [nu/(nu+k)]

      _2F1(

        m,

        nu+k;

        nu+k+1;

        -chi

      ).


Through second order, the line-centroid moment becomes


    <s>_moment2

      =

      c_C

      [G_4/G_2]

      [

        R_1

        +

        kappa_G chi^2 T_(3,2)

      ].


The second raw moment is


    <s^2>_moment2

      =

      c_C^2

      [G_6/G_2]

      [

        R_2

        +

        3 kappa_G chi^2 T_(4,2)

      ].


The factor 3 comes from


    (centroid)^2 + central variance

      =

      s_R^2

      [

        1

        +

        2 Delta_c

        +

        m_2

      ]


to second order, and Delta_c=m_2 in this unchirped Gaussian case.


These formulas provide a nearly free independent check of DER017's effect on centroid and RMS width in the simplest nonlinear geometry.


## 26. Cheap photon-source size moments from the existing Gaussian overlap algebra


The existing overlap_transverse_profile already evaluates the full transverse photon-production image by quadrature. If an optimizer only needs the source centroid and RMS size, the complete image is unnecessary.


After DER001's time integration, at each remaining outer quadrature coordinate q the transverse collision density is Gaussian.


Write its transverse exponent as


    -1/2 x^T A(q) x

    +

    l(q)^T x.


The conditional source mean and covariance are


    m(q)

      =

      A(q)^(-1) l(q),


    C(q)

      =

      A(q)^(-1).


Let w(q) be the same reduced luminosity weight used by overlap_yield after the transverse Gaussian normalization is included.


Then the global photon-source centroid is


    mu_src

      =

      [integral w(q) m(q) dq]

      /

      [integral w(q) dq],


and the global covariance is


    Sigma_src

      =

      [integral w(q)

        { C(q)+m(q)m(q)^T }

        dq]

      /

      [integral w(q)dq]

      -

      mu_src mu_src^T.


For the default head-on beta_ff=0 overlap this requires only the same 1D z quadrature already being performed for total yield.


For exact crossed/flying geometries it follows the corresponding 2D outer quadrature.


These source-size moments are useful for brightness/brilliance estimates, for automatic spatial ranges, and as independent checks of Xigma spatial source distributions. They can be accumulated almost for free during overlap_yield rather than constructing a full spatial image.


## 27. Recommended additional AnalyticalEngine features after this investigation


The following additions are sufficiently useful and sufficiently derived to justify implementation.


Priority A — replace empirical nonlinear diagnostics where the model supports it.


Add exact fixed-width Gaussian nonlinear-shape moments J_1,J_2,J_3 and report:


    mean a_shape,

    std a_shape,

    mean DER016 V_a,

    total nonlinear variance decomposition.


Use these values instead of NONLINEAR_BROADENING_RANGE when the selected semi-analytical model is fixed-width. Keep the empirical bracket only as a fallback for focused geometries until their deterministic source map is implemented.


Priority B — correct and extend AnalyticalEngine.SPECTRUM.


Implement the normalized nonlinear angle-integrated single-electron shape G(z;h), then average it over the deterministic nonlinear-coordinate distribution rather than inserting only mean ahat into the linear DER011 polynomial.


This removes a known first-order-in-h shape approximation.


Priority C — add direct collimated/angle-resolved semi-analytical outputs.


Use the general DER015 gamma inverse and DER012 polarization kernel from the unified document to support selected ANGULAR_DISTRIBUTION and COLLIMATED_SPECTRUM tiers without macroparticles.


The simplest zero-emittance fixed-width model is already only a 1D nonlinear-coordinate quadrature per (s,theta_x,theta_y) point.


Priority D — add essentially free optimizer diagnostics.


Report:


    exact/semianalytical on-axis centroid,

    RMS bandwidth,

    circular-aperture capture fraction,

    source rms x/y and covariance.


These are valuable in broad parameter searches because they avoid constructing dense output grids.


Priority E — retain the existing hierarchy rather than forcing one model.


Every result should record the model tier and assumptions used. The AnalyticalEngine should prefer a cheaper exact specialization when available and otherwise move upward through 1D/2D/3D deterministic quadrature before falling back to warnings or Xigma.


## Repository integration notes

On the `main` branch inspected during migration, `AnalyticalEngine` supports `TOTAL_YIELD` and angle-integrated `SPECTRUM`. The overlap/yield side already uses DER001/DER002, including full paraxial Gaussian spot evolution and the exact 2D crossing/flying-focus routes where selected. The spectral side still uses one luminosity-weighted mean `ahat`, the DER011 linear polynomial with a compressed energy coordinate, and the empirical `NONLINEAR_BROADENING_RANGE`; `PROGRESS.md` still lists analytical collimated-spectrum construction as open.

DER019 does not alter DER001–DER018. In particular, DER001/DER002 remain the normalization authorities for focused/flying Gaussian overlap; DER012 is the physical crossing-angle polarization kernel; DER013 supplies `D`; DER015 supplies observer-dependent `Q`; DER016 defines the trajectory moments; and DER017 defines the second-order spectral-moment reconstruction. The deterministic source tiers here provide new ways to evaluate or validate those quantities without Gaussian macroparticle sampling.

## Implementation implications

The derivation supports a staged extension of `AnalyticalEngine`: exact fixed-width nonlinear-shape moments and distributions; a properly normalized nonlinear angle-integrated spectrum; direct deterministic point/collimated spectra using the DER015 inverse; energy-integrated angular distributions and aperture capture; cheap centroid/bandwidth/source-size diagnostics; and, at higher fidelity, deterministic focused/flying source maps with short per-node trajectory quadrature.

The user-facing model hierarchy should normally be selected automatically per requested observable. A compact semantic interface is `auto` (cheapest accepted model), `fast` (controlled lower-cost approximations for broad optimization), and `reference` (highest-fidelity deterministic Gaussian model supported), with expert model pinning for validation. Structural applicability and continuous validity diagnostics must be kept separate, and the chosen model, assumptions, quadrature dimension, validity diagnostics, and rejected cheaper alternatives should be recorded in result metadata.

This derivation does not require a new Xigma table dimension and does not authorize changes to Xigma physics.

## Validation ideas

The strongest checks are the reduction identities developed above: recovery of DER001 overlap moments in common limits; recovery of DER011 at zero nonlinear shift; exact unit normalization of the nonlinear angle-integrated shape; exact recovery of the circular-aperture fraction after energy integration; agreement of deterministic Stage-0/Stage-1 quantities with over-resolved Xigma; separate comparison of `rho0`, `rho1`, and `rho2`; and recovery of DER001/DER002 total luminosity from focused/flying deterministic source maps.

The fixed-width and focused deterministic tiers should be compared while scanning the dimensionless diffraction/hourglass/drift parameters before numerical acceptance thresholds are chosen. Those thresholds are validation outputs, not assumptions to invent in advance.

## Open questions

The numerical acceptance thresholds for the frozen-spot tier, optimal noncentral quadratic-form representation, deterministic quadrature families/node counts, treatment of fully general position–energy correlations, and the cheapest exact representation for simultaneous crossing angle plus flying focus at the per-trajectory-ratio level remain open. Spatial carrier chirp, quantum recoil, higher harmonics, arbitrary aberrated fields, and non-Gaussian measured bunches are outside DER019's first implementation scope.
