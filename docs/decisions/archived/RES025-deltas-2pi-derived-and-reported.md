# RES025 — Delta's `2 pi` is derived and reported, not corrected

Status: implemented
Type: testing
Archived: 2026-08-08

**Superseded by RES033** (2026-08-08). The factor is now applied rather than reported, and
the identity harness expects one. What follows is the reasoning that was correct while the
question was open — kept as the record of why the suite was gated at `2 pi` for two
phases, not as live policy.

## Problem

The predecessor recorded delta's disagreement with the table kernel as "consistently
~6.3x ... suspiciously close to 2*pi, not yet explained." Whether and how to act on that
discrepancy, before its physical origin was independently confirmed, needed a decision.

## Decision

`validation.references.delta` keeps the predecessor's differential prefactor unchanged.
`check_normalization` reports the ratio against `2 pi`, and the identity section of
`run.py` passes while that ratio holds and fails if it moves.

## Alternatives considered

**Divide delta's prefactor by `2 pi` so the identity reads 1.0.** This is the tempting one
and it is what P14 forbids: the derivation says the two methods are inconsistent by `2 pi`
and identifies which side counts photons — Stage 0's `flux x cross-section x time`,
corroborated by a closed form that reproduces it exactly — but *which* normalization
xigma's Stage-2 kernel should carry is a statement about the paper's formalism, and that
kernel does not exist yet (Phase 3a). Pasting the factor into the arbiter now would remove
the evidence before the question is asked.

**Assert the identity against 1.0 and let the suite run red until §9.1 closes.** Makes the
suite permanently red, and a permanently red suite is an ignored suite.

**Report the ratio without an expected value, as the predecessor did.** Is where the
predecessor left it, and "an unexplained 6.3" survived for as long as it did precisely
because nothing would ever notice it changing.

## Rationale

The predecessor recorded this as "consistently ~6.3x ... suspiciously close to 2*pi, not
yet explained". It is not close to `2 pi` — it is `2 pi`, and the integral is elementary:
with `u = gamma**2 r**2`, `int dOmega 3 gamma**2 <a_fac> / (1+u)**2 = 3 pi [1 - 2/6] =
2 pi`. Reproducing that from an independent CGS implementation also rules out the old
repo's `k0_las` normalization as the cause. Pinning the derived value keeps the discrepancy
visible *and* guarded.
