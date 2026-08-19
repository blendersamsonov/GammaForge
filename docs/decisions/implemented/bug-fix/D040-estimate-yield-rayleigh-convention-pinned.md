# D040 — `estimate_yield`'s laser-divergence convention error is flagged and pinned, not fixed

Status: implemented
Class: bug-fix

## Problem

`engines.analytical.formulas.estimate_yield`'s hourglass term uses a Rayleigh-range
convention inconsistent with this repo's own laser model
(`io.laser.GaussianParaxialLaser.rayleigh_x`), producing yields that disagree with
`overlap_yield` by a measurable factor. The question is whether to fix the function or
flag and pin the discrepancy.

## Decision

`engines.analytical.formulas.estimate_yield` keeps the predecessor's
`lambda^2 / (pi^2 sigma_lr0^2)` hourglass term unchanged, with a `.. warning::` in its
docstring and a test —
`test_overlap_yield_differs_from_the_legacy_closed_form_by_the_rayleigh_convention` —
pinning the resulting 3.285x disagreement with `overlap_yield` on the baseline scenario.

## Alternatives considered

**Fix `estimate_yield` to use `rayleigh_x()`.** Makes the function correct and useless at
the same time: it then reproduces neither the predecessor (breaking `_PREDECESSOR_YIELD`,
its only remaining purpose) nor anything `overlap_yield` does not already do better.

**Delete `estimate_yield`.** Loses the analytic reduction anchor that gives
`overlap_yield` its strongest test, and the port-fidelity record. (Re-examined later —
see Amendments: the reduction-anchor half of this reason turned out not to hold, but the
port-fidelity half still does.)

**Treat it as a `GRAND_PLAN.md` §0 BLOCKING paper-code discrepancy.** §0 covers code
disagreeing with the paper. The paper does not contain this formula; this is code
disagreeing with *other code* in a way settled by an independent derivation and an exact
numerical reduction, so it is reportable rather than blocking.

## Rationale

The general derivation (`docs/DERIVATIONS.md` §A.5) identifies that term's coefficient as
`sigma_l / z_R` exactly. Both `io.laser.GaussianParaxialLaser.rayleigh_x` and the
predecessor's *own* pulse class define `z_R = 4 pi sigma^2 / lambda` (`w0 = 2 sigma`),
giving `lambda / (4 pi sigma)` — so the predecessor's *analytical.py* is internally
inconsistent with the predecessor's own laser model, by a factor of 4 in the angle. The
port carried that faithfully; it was not introduced here.

Changing it is not this session's call to make. `estimate_yield`'s entire remaining value
is that it reproduces the predecessor's worked example to ~1e-6 (the `_PREDECESSOR_YIELD`
pin), and "correcting" it would destroy that without any code depending on the result —
`AnalyticalEngine` uses `overlap_yield`, which is free of the issue because it reads
`rayleigh_x()`/`rayleigh_y()` directly. Pinning the discrepancy as a test makes it visible
and prevents it drifting silently, which is what the situation actually needs. Worth
recording explicitly: nothing in the suite before this was sensitive to that term at all —
the predecessor pin tests port fidelity, and the Thomson-limit anchor drives `nu` to
infinity, which removes the hourglass term entirely.

## Consequences

Nothing in the suite before this was sensitive to the hourglass term; the new pin test is
now the only thing guarding against the discrepancy drifting further. `estimate_yield`'s
only remaining justification is port fidelity to the predecessor's worked example — see
the amendments below for why deleting it was reconsidered, and declined a second time.

## Amendments

> **2026-08-10 — the convention question is settled, and this entry's reading of it was
> right.** The Rayleigh range is conventionally written `z_R = pi sigma_e2^2 / lambda` in
> the **1/e^2** convention, where `sigma_e2` is the radius at which intensity falls to
> `e^-2`. This repo stores widths as the **RMS of the photon-density profile**, where the
> density at `r = sigma` is down only by `e^-1/2` — a different number. The correct chain
> is therefore *convert first, then apply the standard formula*: `sigma_e2 = 2 sigma_RMS`,
> giving `z_R = 4 pi sigma_RMS^2 / lambda`, which is what
> `io.laser.GaussianParaxialLaser.rayleigh_x` computes (verified numerically: matching
> `exp(-r^2 / 2 sigma^2)` against `exp(-2 r^2 / w0^2)` gives `w0 = 2 sigma` exactly). The
> predecessor's *analytical.py* used `lambda / (pi sigma_RMS)` for the divergence — the
> 1/e^2 formula applied to an RMS width, i.e. the conversion skipped — which is 4x in the
> angle and the discrepancy above. **The consequence for the predecessor's published
> yields stands as stated: they are low by that factor wherever the hourglass term
> mattered.** `rayleigh_x`'s docstring now carries the conversion so the next reader meets
> it before the formula.

> **Re-examined 2026-08-10, on the author's question "why keep it if we don't use it?"**
> One of the two stated reasons does **not** hold. This entry claimed deleting it would
> "lose the analytic reduction anchor that gives `overlap_yield` its strongest test" — but
> `overlap_yield` reproduces the Thomson limit *by itself*, to ratio `1.000000000` at every
> quadrature resolution, with no reference to `estimate_yield` at all. §7's closed-form
> anchor is therefore already independent of it, and
> `test_overlap_yield_reduces_to_the_round_beam_closed_form` compares against
> `_closed_form_round`, a hand-written form in the test file, not against this function.
>
> What genuinely remains is **port fidelity**: `_PREDECESSOR_YIELD` pins agreement with the
> predecessor's worked example to ~1e-6, which is the record that this repo's rebuild
> reproduces what came before — and, given D040's 4x divergence error, the record of
> *exactly which* predecessor numbers were affected. That is a real thing to keep, but it
> is one test's worth of value, not an engine's.
>
> **Kept, with the reason corrected rather than the code changed.** It costs nothing (it is
> never called outside tests), and deleting it would discard the only executable link to
> the predecessor's published results while D040 remains an open question about them. If
> the author decides those results no longer need reproducing, this and its three tests go
> together — that is a one-commit deletion, not a refactor.
