# Ponderomotive observation-direction factor — physics handoff

Status: resolved by the author on 2026-09-23 and implemented as DER014/RES088. The
manuscript source still needs synchronization.

Date: 2026-09-23

## Resolution (2026-09-23)

A. Samsonov selected the beaming-cone approximation for production: for each electron,
replace the observation direction in the numerator coefficient by that electron's unit
velocity direction, $\mathbf n\simeq\mathbf e$. The coefficient multiplying `ahat` is
therefore

$$
P=\frac{1-\mathbf e\cdot\mathbf n_0}{2}.
$$

This multiplies `ahat` directly; it is not an amplitude correction to $a_0$. The exact
observer-dependent ratio derived below remains useful provenance but is not the selected
production contract. DER014 records the adopted approximation and its inverse/Jacobian;
RES088 records the implementation choice. Stage 1 deposits
$P a_{0,\mathrm{shape}}$ per trajectory, so the retargeted table coordinate already
equals $P\hat a$; NumPy and CuPy Stage 2 consume it without reapplying $P$. Direct delta
references, which do not use the table, apply the same factor per particle.

## Question and conclusion

The manuscript derives the phase of radiation observed along the unit vector
`n` from the incident plane-wave phase, whose propagation vector is `n0`. The
question was whether the laser-induced longitudinal momentum term should retain
the projection `n . n0`.

Yes. The current manuscript and GammaForge resonance formula omit an
observation-direction factor. The exact plane-wave decomposition gives the factor

$$
C(\mathbf n,\mathbf v,\mathbf n_0)
=
\frac{1-\mathbf n\mathbin{\cdot}\mathbf n_0}
     {1-\mathbf v\mathbin{\cdot}\mathbf n_0}
$$

multiplying the cycle-averaged ponderomotive term `ahat`. The present formula has
implicitly set $C=1$.

The physical intuition behind the report was substantially correct, with one
refinement: the result is not simply a multiplier $\mathbf n\cdot\mathbf n_0$.
The energy and longitudinal-momentum pieces combine into
$1-\mathbf n\cdot\mathbf n_0$, normalized by the incident encounter factor
$1-\mathbf v\cdot\mathbf n_0$. The effect enters through the retarded radiation
phase; describing it solely as a projected force is suggestive but incomplete.

## Exact algebra locating the lost factor

Use units with $c=1$. Decompose the electron momentum relative to the laser
propagation direction:

$$
\mathbf p=\mathbf p_\perp+p_\parallel\mathbf n_0,
\qquad
\rho_0=\gamma-\mathbf n_0\cdot\mathbf p.
$$

The plane-wave invariants used by the manuscript give

$$
p_\parallel=\frac{1+|\mathbf p_\perp|^2-\rho_0^2}{2\rho_0},
\qquad
\gamma=\frac{1+|\mathbf p_\perp|^2+\rho_0^2}{2\rho_0}.
$$

Let $\mu=\mathbf n\cdot\mathbf n_0$. Substitution without specializing the
observation geometry gives

$$
\boxed{
\frac{\gamma-\mathbf n\cdot\mathbf p}{\rho_0}
=
\frac{(1-\mu)(1+|\mathbf p_\perp|^2)}{2\rho_0^2}
+\frac{1+\mu}{2}
-\frac{\mathbf n\cdot\mathbf p_\perp}{\rho_0}
}.
$$

For the usual carrier average, $\langle\mathbf a\rangle=0$. The linear term in
the field is oscillatory, while the field-induced cycle-averaged contribution is

$$
\Delta\left\langle
\frac{\gamma-\mathbf n\cdot\mathbf p}{\rho_0}
\right\rangle
=
\frac{1-\mathbf n\cdot\mathbf n_0}{2\rho_0^2}
\langle a^2\rangle.
$$

Using the field-free identity

$$
\rho_0=\gamma(1-\mathbf v\cdot\mathbf n_0)
$$

and the manuscript's near-axis reduction

$$
1-\mathbf n\cdot\mathbf v
\simeq \frac{1+\gamma^2\theta^2}{2\gamma^2},
$$

the averaged retarded phase is therefore

$$
\left\langle\frac{\mathrm d\varphi_{\mathbf n}}{\mathrm d\varphi}\right\rangle
=
\frac{1+\gamma^2\theta^2+C\hat a}
     {2\gamma^2(1-\mathbf v\cdot\mathbf n_0)}.
$$

The corresponding candidate corrected fundamental resonance is

$$
\boxed{
\omega_R
=
\omega_L
\frac{2\gamma^2(1-\mathbf v\cdot\mathbf n_0)}
     {1+\gamma^2\theta^2+C\hat a}
},
\qquad
C=
\frac{1-\mathbf n\cdot\mathbf n_0}
     {1-\mathbf v\cdot\mathbf n_0}.
$$

This derivation was also checked numerically for 10,000 random unit vectors,
transverse momenta, and positive $\rho_0$: the boxed momentum decomposition agreed
with direct evaluation to a worst absolute difference of approximately
$2.2\times10^{-11}$ in ordinary double precision. That is an algebra check, not
independent physics validation.

## Where the manuscript loses it

Paper authority: `/home/alexander/Work/Papers/2026/Compton-Numerics/xigma.tex`.

The loss occurs between equations `ratio` and `ratio_expanded`/`ratio_final`:

- `ratio` correctly writes the phase derivative as
  $(\gamma-\mathbf n\cdot\mathbf p)/\rho_0$.
- `ratio_expanded` assigns the $|\mathbf p_\perp|^2$ contribution the coefficient
  $1/\rho_0^2$. The unspecialized coefficient is
  $(1-\mathbf n\cdot\mathbf n_0)/(2\rho_0^2)$.
- This is carried into `phin`, whose integrand uses
  $1+\gamma^2\theta^2+\langle a^2\rangle$.
- The prose following `phin` then explicitly claims that the ponderomotive term has
  no observation-geometry dependence.
- Equation `wR` and all later reductions inherit the unit coefficient of `ahat`.

In other words, the derivation silently uses the backscattering replacement
$1-\mathbf n\cdot\mathbf n_0\simeq2$ for the nonlinear term while retaining a
general $1-\mathbf v\cdot\mathbf n_0$ in the incident Doppler numerator.

## Scope: when setting C = 1 is acceptable

This finding does not mean that the familiar resonance formula is useless.

1. **Observation exactly along the electron direction.** At production's
   $\beta=1$ approximation, $\mathbf n=\hat{\mathbf v}$ gives $C=1$ exactly.

2. **Head-on, paraxial observation.** With
   $\mathbf n_0=-\hat{\mathbf v}$,

   $$
   C\xrightarrow{\beta\to1}\frac{1+\cos\theta}{2}
   =\cos^2(\theta/2)=1-\theta^2/4+O(\theta^4).
   $$

   The omitted contribution is then a mixed $O(\hat a\theta^2)$ correction. In
   the usual ultrarelativistic cone, $\theta=O(1/\gamma)$, it is subleading in
   $1/\gamma$. A model that explicitly declares this truncation may consistently
   use $C=1$.

3. **Crossed laser with direction-resolved observation.** If the observation
   direction moves away from the electron direction within the plane containing
   a tilted `n0`, $C-1$ generally contains a term linear in the observation
   displacement. The formula therefore cannot simultaneously be advertised as
   general in laser crossing angle and direction-resolved while silently setting
   $C=1$.

4. **Near co-propagation.** The denominator
   $1-\mathbf v\cdot\mathbf n_0$ becomes small. This is already outside xigma's
   declared regime and should not be used to argue for extrapolating the corrected
   expression there without a fresh approximation audit.

The practical decision is therefore not merely "insert a missing factor." The
author must decide whether GammaForge's contract is the leading near-axis
ultrarelativistic truncation, in which case the discarded order must be stated, or
direction-resolved crossed geometry, in which case the factor must be retained.

## Independent literature cross-check

Krafft, Doyuran, and Rosenzweig, *Pulsed-laser nonlinear Thomson scattering for
general scattering geometries*, Phys. Rev. E **72**, 056502 (2005), derives a
general-geometry fundamental frequency whose intensity-dependent denominator
contains the corresponding observation/laser-direction factor. In the coordinate
convention of their Eq. 21 it appears as an angular multiplier of the nonlinear
term. This supports the structure above, subject to reconciling their $a^2$
normalization with this manuscript's cycle-averaged `ahat`.

- DOI: <https://doi.org/10.1103/PhysRevE.72.056502>
- APS PDF: <https://journals.aps.org/pre/pdf/10.1103/PhysRevE.72.056502>

This literature agreement is supporting evidence, not a substitute for an
independent derivation in the repository's notation.

## Repository impact

### Derivations and decisions

- `docs/derivations/verified/DER005-crossing-angle-in-the-emission-kernel.md`
  calls the resonance "already general in the paper." That conclusion is not
  valid for nonzero `ahat` away from the electron direction.
- `docs/derivations/verified/DER013-direction-dependent-doppler-reduction.md`
  begins from $A=1+\hat a$ and therefore verifies the algebra of an incomplete
  starting resonance. Its $\hat a=0$ direction-Doppler result remains intact.
- `docs/decisions/implemented/bug-fix/RES082-consistent-direction-doppler.md`
  correctly makes the encounter factor consistent through flux, root, Jacobian,
  and GPU support, but does not address the distinct outgoing-phase coefficient
  of `ahat`.
- `docs/GRAND_PLAN.md` section 9.3 and `PROGRESS.md` currently overstate the scope
  of the implemented crossed-angle resonance and will need correction when the
  physics disposition is agreed.

Do not renumber or delete any decision or derivation identifiers. Follow the normal
archive/supersession rules if a new result supersedes part of DER005 or DER013.

### Production and validation code

The current production-scaled resonance is

$$
s_\mathrm{res}=\frac{D\gamma^2}{1+\hat a+\gamma^2r^2},
\qquad
D=\frac{1-\mathbf e\cdot\mathbf n_0}{F_0},
$$

where production uses $\mathbf v=\mathbf e$ at $\beta=1$. The candidate corrected
form is

$$
s_\mathrm{res}=\frac{D\gamma^2}{1+C\hat a+\gamma^2r^2},
\qquad
C=\frac{1-\mathbf n\cdot\mathbf n_0}
        {1-\mathbf e\cdot\mathbf n_0}.
$$

Known affected locations include:

- `src/gammaforge/engines/xigma/stages.py:spectrum_from_table`
  (`g_sq = (1 + a_c) / inv_base` and its Jacobian prefactor);
- `src/gammaforge/engines/xigma/spectrum_sampler.py` (proposal root, production
  root, prefactor, and conservative support bounds);
- `src/gammaforge/validation/references/delta_emission.py:emission_lines`
  (`den = 1 + ahat + gamma**2*r2` for every Doppler mode);
- `src/gammaforge/validation/references/delta.py:resonance_spectrum` (the same
  denominator in the simpler reference);
- tests and validation records that compare implementations sharing this starting
  formula.

At fixed geometry in the $\beta=1$ production approximation, define
$A_C=1+C\hat a$. The inverse root would become

$$
\Gamma^2=\frac{A_C}{D/s-r^2},
$$

and its direct derivative would retain DER013's form with $A$ replaced by $A_C$,

$$
\left|\frac{\mathrm d\Gamma}{\mathrm ds}\right|
=\frac{D\Gamma^3}{2A_Cs^2}.
$$

This does **not** by itself prove that replacing every `1 + ahat` in the reduced
kernel by `1 + C*ahat` is sufficient. The spectral prefactor, residual nonlinear
phase, and CUDA support bounds must be rederived together. In particular, `C`
depends on both observer and electron directions, so support bounds must enclose
the joint variation of `D` and `C`, not treat them as unrelated extrema.

The trajectory definition of `ahat` itself remains unchanged: it is still the
fluence-weighted cycle average sampled by the electron. What changes is its
coefficient in the outgoing radiation phase.

### Existing validation does not close this issue

- The direction-Doppler tests primarily validate $D$, its inverse-root Jacobian,
  and CPU/GPU agreement. They do not independently derive the coefficient of
  `ahat`.
- Delta currently uses the same nonlinear denominator as production. Agreement
  between delta and xigma is therefore circular for this particular factor.
- All `ahat = 0` tests are insensitive to the issue.
- On-axis tests at $\beta=1$ have $C=1$ and are also insensitive.
- The repository already records independent arbitrary-angle emission validation
  as open. This finding is one concrete reason not to claim scientific closure
  from the existing matched-bin packets.

The untracked root file `DOPPLER_PHYSICS_NOTE.md` predates this finding and repeats
the old $1+\hat a+\gamma^2r^2$ denominator. Do not use its nonlinear resonance
section as authority for this issue. Its discussion of the separate incident
per-particle Doppler factor may still be useful after checking it against DER013.

## Recommended next-session sequence

1. Re-derive the full single-electron phase and fundamental line in the manuscript's
   exact `ahat` and polarization conventions. Keep the oscillatory linear-in-`a`
   term separate from the cycle average.
2. Reconcile the result with Krafft et al. Eq. 21 and, preferably, a covariant
   quasi-momentum derivation. Record every normalization conversion explicitly.
3. Decide the model contract with the author:
   - retain $C$ for direction-resolved crossed geometry; or
   - deliberately set $C=1$ as a controlled near-axis truncation and narrow all
     claims/documentation accordingly.
4. Treat the result as a new or superseding derivation. Downgrade/archive/supersede
   affected verified claims according to `docs/derivations/README.md`; do not edit
   historical reasoning in place contrary to the repository lifecycle rules.
5. Before production edits, add independent tests that fail with the old formula:
   - exact plane-wave flat-top resonance at a geometry with $C\ne1$;
   - head-on off-axis limit $C=\cos^2(\theta/2)$;
   - crossed geometry with opposite observation offsets in the collision plane;
   - $\hat a=0$, on-electron-axis, and head-on limits;
   - an independent trajectory-phase or quasi-momentum calculation that does not
     import production resonance helpers.
6. Update NumPy, CuPy, delta, inverse Jacobian, proposal/support bounds, and any
   resonance-importance logic as one coherent change. Partial placement would repeat
   the failure mode that RES082 was created to prevent.
7. Update `GRAND_PLAN.md`, `PROGRESS.md`, affected decisions/derivations, and the
   corresponding human walkthrough notebook if the validation pipeline or exposed
   engine behavior changes. Run `python tools/build_notebooks.py --run` when the
   notebook-update rule applies.
8. Run focused physics tests, `pytest -m fast`, the relevant real-CUDA gate, and
   finally the full/heavy validation appropriate to a phase-level physics change.
9. Run `graphify update .` after the landed repository changes.

## Questions that remain open

- Is the intended paper accuracy leading order in the beaming scaling
  $\theta=O(1/\gamma)$, or genuinely complete through $O(\theta^2)$? The omitted
  head-on term is $O(\hat a\theta^2)$, so the answer changes whether this is an
  error inside the declared approximation or an unstated higher-order truncation.
- Does the paper's residual nonlinear line-shape phase require only
  $\hat a\mapsto C\hat a$, or are further direction factors present before the
  slow-envelope reduction?
- Which independent reference should establish scientific acceptance: a direct
  plane-wave trajectory Fourier calculation, kascade at controlled geometry, or
  both?
- How conservative must GPU support become when `C` and `D` share the same
  electron-direction denominator?

Until these are resolved, the safe statement is:

> GammaForge's implemented direction-Doppler resonance is validated for its shared
> $C=1$ model and remains correct in the linear and on-electron-axis limits. The
> model is not yet independently validated for the observation-direction dependence
> of the nonlinear ponderomotive shift.
