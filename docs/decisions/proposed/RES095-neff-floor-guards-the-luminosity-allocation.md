# RES095 — An effective-sample-size floor guards the luminosity allocation

Status: proposed
Class: feature

*(Numbering note: this entry was written as RES094 and renumbered to RES095 on 2026-09-30 to resolve a collision with parallel work on `main`, which merged PR #15 and took RES093 for the uniform raw-`ahat` production retarget. `main` merges first, so this branch is the later merger.)*

## Problem

The luminosity-weighted allocation is regime-dependent, and in one measured regime it inverts
badly enough to disqualify itself as an unconditional default.

Across the ten-scenario bank (4M reference, 12 paired replicates, spectrum L1 at
`N = 40 000`, `docs/validation/adaptive-sampling-2026-09-29.md` §5–§7), the shipped scheme is
resolved **better** than plain stratified QMC in three scenarios — `tight_focus`,
`focus_3um`, `twiss_corr`, each by a consistent 12–15% — and resolved **worse** in exactly
one, `wide_bunch`, by **+39%**. The asymmetry is the problem: the downside is roughly 40× the
upside in magnitude, so a scheme that wins slightly and often loses catastrophically and
occasionally is not a default.

The mechanism is a collapse of the weight vector, and it is not subtle. With
`N_eff = (Σw)²/Σw²`, the shipped arm's effective sample fraction at 40 000 particles is

| | `N_eff/N` |
|---|---|
| `wide_bunch`, shipped arm | **0.045** |
| `wide_bunch`, `s=1` and `λ=1` | **0.004** |
| every other scenario, every allocation arm | 0.836 … 0.992 |

An order of magnitude separates the failure from everything else, and it is monotone in the
strength of the luminosity weighting. An estimator with `N_eff = 157` carries ~255× the
variance of equal weights, which is the entire explanation for the regression.

The failure mode is a *diffuse density*: `wide_bunch` is a σ=400 µm beam against a σ=4 µm
laser, so luminosity is concentrated in a small part of the sampled volume and the criterion
piles particles into it, starving the rest. Nothing about the allocation notices.

## Decision

*Proposed, not built.* Guard the allocation with an effective-sample-size floor, and make the
guard **self-diagnosing**: the allocation is trusted only while the weight vector it produces
stays non-degenerate.

**Reject the luminosity-weighted allocation and fall back to uniform (`λ = 0`) allocation when
its effective sample fraction falls below a floor**, default `0.5`, as a `PilotConfig` field.

The floor is evaluated **before any particle is drawn**. The allocation assigns `n_m`
particles to region `m` with constant weight `w_m = P_m/n_m` and `Σ_m P_m = 1`, so

```
N_eff = (Σ_m P_m)² / Σ_m P_m²/n_m  =  1 / Σ_m P_m²/n_m
```

which is a function of the plan alone. No pilot overhead, no sampling, no reference, and no
noise — `N_eff` is a deterministic functional of the weights, and it is already computed for
reporting.

Guarded, the shipped arm is **3 resolved better, 0 resolved worse, 6 within noise** across the
bank: the three concentrated-scenario gains are kept and `wide_bunch` reverts to the control it
lost to.

## Alternatives considered

### Why not just ship `s=1` with `λ=0` and drop the allocation?

This is the simplest option and it is genuinely defensible: it is the best or tied-best arm in
7 of 10 scenarios, it never regresses, and it needs no new mechanism. It was the
recommendation from the first measurement, which had only looked at `baseline` and
`wide_bunch` — the two most diffuse scenarios in the bank, where the allocation is at best
neutral. Rejected because it forfeits a real, mechanistically-predicted, reproducible
12–15% gain in the concentrated regime, corroborated independently by the cell-aware
experiment (§8 of the validation record). If the expert review concludes the concentrated gain
is too thin to bank on (3 of 10, sign test p ≈ 0.125), **this alternative should be preferred
and RES095 dropped** — that is a judgement about how much a 12–15% gain in three scenarios is
worth, not a technical question.

### Why not clip the allocation instead of rejecting it?

A bound on `Q_m/P_m` (or on the per-region weight ratio) degrades gracefully where a binary
fallback does not: a mildly over-concentrated allocation is nudged back, and only a genuinely
degenerate one is overridden. This is arguably the better engineering answer and was not
chosen here only because the observed failure is sharply **bimodal** — 0.045 versus 0.836, with
nothing in between — so a clip would almost never engage and the added degree of freedom would
buy nothing measurable. If the bank later shows intermediate cases, the clip should replace the
floor. Worth the expert's view.

### Why not classify the regime from beam and laser parameters?

Detecting "diffuse" from `(σ_x, σ_y, w_0, f)` and switching behaviour is the obvious
alternative. Rejected: it pattern-matches scenario *identity*, which is exactly the kind of
fit that fails on the eleventh scenario. The `N_eff` signal is intrinsic to the allocation's own
output — the degenerate weight vector reports itself — so it generalizes to a geometry nobody
has run.

### Why not make the criterion Stage-1-cell-aware, as the handoff suggests?

Measured (`exp5`, validation record §8). It reproduces the *same* regime split rather than
removing it, and collapses harder on `wide_bunch` (`N_eff = 155`, some regions allocated
zero). It is also unsupportable by the current pilot, which sees 5.9–7.7 cells per region where
the oracle sees 46–63; eight pilot points cannot resolve what the criterion needs. A different
pilot is a research task, not a guard.

### Why not reduce `λ` until the collapse stops?

The λ sweep was run (validation record §6) and the collapse is monotone in λ, so some lower λ
would avoid it — but the sweep also shows the *wins* in the concentrated regime shrink with λ
along with the losses, and the interaction with `s` is unresolved. Tuning λ to dodge one
scenario is a fit to that scenario.

## Rationale

The guard is worth its complexity because it is **free, exact, and early**. It costs no
sampling, no pilot work, and no reference: `N_eff` follows from the masses and counts the plan
already holds, and it is evaluated before the first particle is drawn, so a rejected allocation
costs nothing but the arithmetic. There is no scenario in which the guard makes a run slower,
and no scenario in which it fires spuriously within the observed range.

It is also **self-diagnosing**, which is the property that matters for a safety mechanism. The
alternative — asking the sampler to know that the user has a diffuse bunch — requires a
classifier over beam and laser parameters, i.e. a model of the failure built from the same ten
scenarios that revealed it. `N_eff` needs no model: a weight vector that has collapsed says so
in its own arithmetic, and the sampler believes it. A guard that can be trusted without being
right about the physics is the more robust thing to ship.

The threshold is not delicate. Every allocation arm on every non-degenerate scenario sits at
`N_eff/N ≥ 0.836` and the one failure sits at `0.045`, so any floor in `(0.08, 0.84)` produces
identical behaviour on the entire bank. `0.5` is the geometric midpoint of the empty interval
rather than a fitted value, and the choice is testable against the mechanism (§ Consequences)
rather than only against the data.

## Consequences

- **This bounds a symptom, not the cause.** The original diagnosis stands: the allocation
  optimizes variance in the luminosity-weighted integral, which is not the variance that
  dominates a resolved Stage-1 density (`docs/validation/adaptive-sampling-2026-09-29.md` §4).
  What §5 adds is that the wrong-variance problem is *conditional* — in concentrated regimes
  the luminosity variance does dominate, and the criterion is right. The guard makes the
  failure mode bounded and observable; it does not make the criterion correct. A future
  criterion should still target Stage-1 cell occupancy, and this guard should be understood as
  a safety net pending that work, not as a resolution of it.
- **The floor is fitted to one bank.** The mechanism is identified and the threshold is
  insensitive, but no scenario outside these ten has been measured. A diffuse-but-not-extreme
  case landing at `N_eff/N ≈ 0.3` would be handled by the guard for the *wrong reason* — it
  would be flagged as degenerate while merely being a legitimate diffuse case. Whether the
  right response there is to fall back or to accept a modestly non-uniform allocation is an open
  question, and the guard's binary nature cannot express it. The clip alternative above is the
  natural answer if that case turns out to be common.
- **A decision is added to a production path.** `build_adaptive_plan` would gain a branch, and
  the fallback must reproduce the `λ=0` allocation exactly — verified by asserting that a plan
  whose floor is exceeded is bit-identical to the same plan built with `luminosity_fraction=0`.
  Tests should also assert the floor fires on a diffuse geometry and does *not* fire on
  `tight_focus` and `focus_3um`, or the guard will silently disable the feature it protects.
- **The `s × λ` interaction is untouched by this.** If a broad reference genuinely needs the
  luminosity term to correct its own over-dispersion, then `s` and `λ` are not independently
  tunable and no fixed pair serves the bank. That is a separate question, and the more
  fundamental one; RES095 is worth doing either way, but it does not settle it.
- **No default promotion follows from this.** RES094 keeps `"iid"` as the default, and the
  measured `proposal_scale = sqrt(2)` default is separately wrong on current evidence
  (resolved worse than `s=1` in 9 of 10). Any promotion decision needs the metric question
  answered first — yield and spectrum disagree per scenario, and the acceptance criterion has
  not been chosen.
