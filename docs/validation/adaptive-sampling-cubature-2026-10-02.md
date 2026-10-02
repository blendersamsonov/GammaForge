# Gauss-Hermite and the Stage-1 table — preliminary measurement

**Date:** 2026-10-02
**Branch:** `feature/luminosity-aware-adaptive-sampling`
**Implements:** Phase II of `docs/handoffs/luminosity-aware-adaptive-sampling.md`
**Status:** **preliminary.** Small budgets, two scenarios, and a reference that is too coarse
for the fitted convergence rates to mean anything. Recorded because the negative result is
solid and because it changes what the phase should attempt next.

---

## 1. What was tested

Six latent coordinates `d = (dx, dy, dz, dθx, dθy, dγ)`, each independently standard normal.
Stage 0 maps them deterministically to `(γ, θx, θy, a0_shape, chirp_mean, luminosity)`. The
quantity of interest is a six-dimensional Gaussian expectation,

```
E[L] = ∫ p(d) · L(d) d⁶d
```

Three source rules, all feeding the identical Stage-0 pipeline through the shared
latent-to-physical map:

| rule | construction | weights |
|---|---|---|
| **IID** (control) | N random points | `1/N` |
| **Global QMC** | Halton sequence, shifted, mapped through the inverse normal CDF | `1/N` |
| **Tensor Gauss-Hermite** | `hermgauss(n)` in 1D, tensor product in 6D; nodes `√2·x`, weights `Πw / π³` | `Πw_k / π³` |

Two properties verified before any result was read, because without them nothing downstream
means anything:

- **The Gauss-Hermite rule is exact, not approximate.** `E[d²] = 1` and `E[d⁴] = 3` to
  roundoff already at order 3; all weights positive and summing to one.
  (`tests/test_cubature_utilities.py`)
- **The Halton sequence is prefix-stable** — the first *k* points do not move when *N* grows.
  Without that, an error-versus-`N` curve measures nothing, because the points changed.

## 2. Data 1 — weight concentration

Effective sample size `N_eff = (Σw)² / Σw²`, reported as a fraction of the points spent:

| order | nodes | `N_eff/N` | points actually carrying weight |
|---:|---:|---:|---:|
| 4 | 4,096 | 4.7e-2 | ~190 |
| 6 | 46,656 | 1.2e-2 | ~550 |
| 8 | 262,144 | 4.7e-3 | ~1,220 |
| 9 | 531,441 | 3.2e-3 | ~1,710 |

**64× more points buys 6× more effective points.** The weights are a product across six
dimensions, so nearly all weight sits on the few nodes that are near the centre in *every*
dimension simultaneously. Additional nodes land in the tails carrying a vanishing share.

## 3. Data 2 — spectral error through Stage 1 (`baseline`)

| rule | deposition | 16k | 64k | 262k |
|---|---|---:|---:|---:|
| IID | nearest | 0.0197 | 0.0124 | 0.0036 |
| IID | CIC | 0.0155 | 0.0099 | 0.0021 |
| QMC | nearest | 0.0141 | 0.0068 | 0.0052 |
| QMC | CIC | 0.0055 | 0.0036 | 0.0031 |
| GH | nearest | *(4k)* 1.056 | *(16k)* 0.444 | *(46k)* 0.618 → *(262k)* 0.219 |
| GH | CIC | *(4k)* 0.472 | *(16k)* 0.312 | *(46k)* 0.189 → *(262k)* 0.144 |

GH spends a different trajectory budget (its budget parameter is the 1D *order*), so its
columns are labelled with the actual trajectory counts.

Against IID at comparable cost, GH's spectral error is **0.14–0.22 against 0.002–0.02** —
**10–100× worse**. The 5D table marginals are worse still, at **1.2–1.9** (120–190% error)
against IID's 0.018–0.15. `tight_focus` reproduces this: GH marginals 1.2–1.9, IID
0.025–0.15.

CIC roughly halves the error for every rule, so deposition discontinuity is a real effect —
but it does not rescue Gauss-Hermite.

## 4. Conclusion

**Gauss-Hermite does not work for filling the Stage-1 table, and more points will not fix it.**

Filling a table requires particles to land in distinct cells, which is a *sampling*
requirement. A tensor grid with product weights behaves like a Monte Carlo run of its
`N_eff`, not of its `N`. At order 8 that is roughly 1,200 particles, and a 120–190% marginal
error is what ~1,200 particles buys. The numbers are therefore not a defect in the rule; they
are the rule's own behaviour measured honestly.

The rule remains valid for **smooth totals** (`E[L]`, its moments), where it is exact by
construction. The Phase-II handoff says not to reject Gauss-Hermite on its low `N_eff`; that
instruction is correct for smooth sums and wrong for cell-filling, and it was applied too
broadly here without asking what it was scoped to.

**This was checkable before the experiment ran**, in one line, from data already produced by
the first smoke test. It was not checked. A cheap guard now exists in
`tests/test_cubature_utilities.py`.

## 5. What is not established, and how the follow-up addresses it

1. **The reference is too coarse for the convergence rates.** These runs used a 400k-particle
   reference whose own spectral floor is ~3e-3. IID at 262k reaches 0.0036, i.e. it is *hitting
   that floor*, which inflates its apparent rate and flattens the others. **The fitted slopes
   above are unreliable; only the fixed-`N` comparisons should be read.**
2. **QMC's advantage is not yet a result.** QMC is better than IID at every fixed `N`, but by
   less than expected, and its apparent slope is *slower* — which is the reference floor again.
3. **Two scenarios only**, at small budgets. `wide_bunch` — the diffuse case that defeated the
   earlier luminosity allocation — is untested here.
4. **The reference itself was single-construction.** A floor derived from one reference's own
   replicates cannot detect a reference that is wrong the same way every time.

### What changed in the follow-up

Four defects, all found by running it:

- **Fitted slopes are now gated on a measured floor.** The reference is the mean of several
  independent 4M IID runs *per deposition scheme*, and the floor is their disagreement. A point
  within 3x the floor is excluded from the fit, and a fit with fewer than three surviving points
  reports `unresolved` rather than a slope. This is the fix for defect 1.
- **The reference is cross-validated** against two independent constructions — large global QMC
  under several shifts, and large IID under several seeds — and the reference is only taken from
  the tensor rule when *its own successive orders have stabilized*. In the quick run the tensor
  successive step was unstable, the tensor rule disagreed with QMC by 42%, and the reference
  correctly fell back to QMC (which agreed with IID to 0.6%). This is the fix for defect 4.
- **A shared-reference bug was found and fixed.** The reference was deposited with `nearest`
  only, so CIC arms were scored against a nearest-built reference — which inverted the CIC
  result, showing CIC as 2x *worse* when an earlier run had it consistently better. References
  are now built per deposition scheme.
- **Method replicates.** Each arm runs at several seeds (IID) or shifts (QMC), so a
  deterministic method's own spread is separable from reference noise. A single-seed arm cannot
  distinguish "this rule is better" from "this seed was lucky".

The follow-up also adds the observation-direction sweep, 1D and 2D marginals, integrated yield,
and a production-resolution spot check. It does **not** revisit the tensor table result: §4 is
settled, and spending compute to extend it would be spending it on a closed question.

## 6. Recommendation

- Drop Gauss-Hermite from the table half; keep it for the totals half.
- Keep plain-random versus global QMC as the comparison that behaves sensibly on both totals
  and tables.
- Re-run against a 4M reference before drawing any conclusion about QMC.

Open question for review: **is a table-fill rate of ~10³ effective particles intrinsic to
tensor grids in six dimensions, or is there a sparse or factored construction whose weights
stay flat enough to populate cells?** Gauss-Hermite is the wrong tool for cell-filling and the
right one has not been identified here.

## 7. Reproducing

```bash
cd experiments/adaptive
./python.sh exp6_smooth.py --only baseline tight_focus --chunk-mb 2000
./python.sh exp7_deposition.py --only baseline tight_focus --ref-n 4000000 --chunk-mb 2000
```

Note that `run_all.py --stage deposition` defaults to `--ref-n 400000`, which is the coarse
reference described in §5; call the script directly to raise it.