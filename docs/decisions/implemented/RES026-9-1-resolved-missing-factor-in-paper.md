# RES026 — §9.1 resolved: the missing factor is in the paper, at eq. (xsec)

Status: implemented
Type: bug-fix

## Problem

`GRAND_PLAN.md` §9.1's ~2π discrepancy (P13) needed tracing to its actual source — a
porting artefact, a coding error, or the paper itself — before Phase 3a could set the
value of Stage 2's normalization constant.

## Decision

The ~2pi of `GRAND_PLAN.md` §9.1 is traced, closed as an investigation, and recorded here.
It is **not** a porting artefact and **not** a coding error: `xigma.tex`'s differential
cross-section, eq. *(xsec)*, is missing a factor `1/(2 pi)`, and the paper's principal
result eq. *(main)* inherits it. No code was changed — see RES025 for why the arbiter keeps
reporting the factor rather than absorbing it, and why the *value* of Stage 2's
normalization constant is Phase 3a's to set once the author has chosen where the factor
belongs.

**The check.** Eq. *(collision)* writes the emission as `d3N/(dw d2Omega) = int v_rel
(d3sigma/(dw d2Omega)) n_ph f_e dt d3r d3p`, so for a photon count the frequency- and
angle-integrated cross-section must be `sigma_T`. From eq. *(xsec)* with `R -> delta` and
linear polarization (`Xi = diag(1,0)`, whose trace factor is eq. *(umod)*'s `|u_1|^2`),
with `u = gamma^2 theta^2`:

    int dw  (w_R/w) delta(w - w_R) = 1
    int d2Omega  3 sigma_T gamma^2 |u_1|^2 / (1 + gamma^2 theta^2)^2
        = (3 sigma_T / 2) [ 2 pi int_0^inf du/(1+u)^2  -  4 pi int_0^inf u du/(1+u)^4 ]
        = (3 sigma_T / 2) [ 2 pi (1) - 4 pi (1/6) ]
        = 2 pi sigma_T

Both integrals are elementary. So eq. *(xsec)*'s prefactor should read `3 sigma_T/(2 pi)`,
and eq. *(main)*'s `6 sigma_T w_L/w^2` should read `3 sigma_T w_L/(pi w^2)`. These are one
error, not two: eq. *(main)* is eq. *(xsec)* times the Jacobian `dGamma/dw` of eq.
*(jacobian)*, which is why `3 -> 6` and `gamma^2 -> Gamma^5/(1+ahat)` while the `1/(2 pi)`
passes straight through. The paper states the opposite immediately above eq. *(xsec)* —
that for observables with no spectral resolution the yield follows from the cross-section
formalism *exactly* — which is the claim the two lines above falsify as written.

**Why it survived in the predecessor.** It was absorbed twice, knowingly, and both sites
carry a comment saying so: xigma's adapter rescaled the kernel's angular spectrum by
`total_yield / full_integral` ("*1.0 would mean the kernel's own normalisation already
agreed with total_yield; it currently doesn't (~2*pi-ish)*"), and the analytical model
applied the same self-consistent rescale under a *QUICK FIX* label. The headline
`total_yield` came from the luminosity sum, a path that never touches eq. *(main)*, so the
kernel's normalization was never the number anyone read. *delta* was the only consumer of
eq. *(xsec)* raw, and therefore the only place the factor was visible.

**Evidence this repo adds.** `integrate_trajectories` computes the elementary `flux x
cross-section x time` count and agrees with the predecessor's own `total_yield` to 0.12%;
`single_electron_spectrum` integrates to that same number as an identity; delta gives 2 pi
times it. Two independent methods against one, and the odd one out is the paper's
differential form. Reproducing the factor in a CGS implementation with no coordinate
normalization also rules out the predecessor's `k0_las` scaling as a cause (RES015).

## Alternatives considered

**(a) Correct eq. (xsec) in `xigma.tex` directly.** Is the author's call in more than one
sense — the paper is the physics authority (§0/P14), and *where* the factor belongs is a
genuine choice: it can sit in the prefactor, be absorbed into the normalization of *R*, or
into the definition of *U*. The manuscript is therefore annotated, not edited: LaTeX
comments at eq. *(xsec)* and eq. *(main)* stating the discrepancy and the derivation,
leaving the typeset output unchanged.

**(b) Leave the finding in this repo only.** Would leave the paper and the code disagreeing
with nothing recording it in the place the physics is decided — the exact failure mode §0
exists to prevent.
