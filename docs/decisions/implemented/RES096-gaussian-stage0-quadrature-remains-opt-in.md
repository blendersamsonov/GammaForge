# RES096 — Gaussian Stage-0 quadrature remains opt-in

Status: implemented
Type: feature

*(Numbering note: this entry was written as RES095 and renumbered to RES096 on
2026-10-01 to resolve a collision with parallel work on `issue-22-engine-catalog`.)*

## Problem

DER023 supplies a temporal-envelope-weighted trajectory rule and a conservative
spatial discard certificate for circular Gaussian pulses. The rule reduces field
evaluations at low order, but Stage-0 luminosity and higher moments converge at
different rates. A default chosen from total yield alone could alter DER016
moment inputs to Stage 1 and Stage 2.

## Decision

Xigma keeps its 200-step midpoint Stage-0 rule as the production default.
`stage0_quadrature="auto"` explicitly selects the shared NumPy/CuPy DER023
Gaussian rule for the built-in circular, coincident-focus, no-flying-focus
Gaussian pulse and pulse-train models. Unsupported laser geometry and source
histograms retain the generic midpoint path. The Gaussian order remains an
explicit numeric parameter; the discard budget defaults to zero.

When discard is requested, a laser-only bank of conservative Gaussian-Lorentzian
convolutions supplies per-particle upper bounds. A cumulative upper/lower
certificate selects the largest admissible discarded set. The certificate
covers discarded luminosity in the unchirped model, not retained quadrature
error. The built-in supported pulses have `C=1`; custom carrier-gradient fields
remain on the generic path.

## Alternatives considered

Make the 24-node Gaussian rule the default: it is fast, but the scenario-bank
scan found 6.06% luminosity L1 error and 46.0% shape-variance L1 error against
the 4096-step reference. Aggregate yield alone concealed these errors.

Make a 256-node Gaussian rule the default: its luminosity L1 error fell to
0.0107%, but shape variance remained 0.923% different. On the measured
8192-particle host case, it was slower than midpoint on both NumPy and CuPy.

Use a per-particle maximum-relative discard cutoff: many individually small
positive bounds can accumulate beyond the requested total loss. The retained
lower bound and discarded upper sum provide a cumulative certificate instead.

## Rationale

`docs/validation/der023-stage0-review-2026-10-01.md` measures convergence,
DER001 agreement, an explicit DER025 chromatic family, conservative bounds,
real-CUDA parity, and runtime. It supports an optional specialist rule and
an explicit discard budget, but no generally better production order. Keeping
the established default avoids changing unrequested Stage-0 physics inputs.

## Consequences

Opt-in Gaussian runs can be much faster at low order but require the user to
check convergence of the moments relevant to their output. A nonzero discard
budget certifies only prefilter loss and may spend much of that budget. The
laser-only table is prepared on first use and cached thereafter; its host-side
cost can outweigh GPU savings in filtered runs. Neither this choice nor the
measurements promotes DER023 or DER025 beyond `derived`.
