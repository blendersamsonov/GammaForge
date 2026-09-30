# RES017 — Dimensioned fields are converted to canonical CGS on construction, not stored in the unit given

Status: implemented
Type: architecture
Archived: 2026-08-10

## Problem

A dimensioned field could either be stored in the unit a user constructed it with (with
pint converting on each access) or converted once, on construction, to the field's
canonical CGS unit. The motivating argument for storing-as-given was floating-point
precision, which needed checking rather than assuming.

## Decision

`as_canonical_quantity` converts an incoming `Quantity` into the field's canonical CGS unit
and stores *that*. A beam built with `Quantity(20, "um")` holds `0.002 centimeter`, and
`m(name)` is therefore a free attribute read.

## Alternatives considered

**Store the `Quantity` exactly as the user gave it, letting pint keep the physics right and
converting only on access.** The motivating argument was floating-point precision: 1 µm
stored as `100` looks better conditioned than the same value stored as `1e-4`. That
intuition is from **fixed** point. IEEE 754 doubles carry 52 mantissa bits at *every*
exponent, so relative precision is a constant ~2.2e-16 across the whole normal range —
`1e-4` is exactly as precise as `100`, and the conversions this project performs were
measured at **0 ULP** (`100 um` → `0.01 cm` lands on the same double as the literal). For
scale, a measured spot size is known to about a percent: fourteen orders of magnitude of
headroom. The two places magnitude does matter are absent here: underflow/overflow (doubles
span 1e±308 and these values live between 1e-25 and 1e10; even float32 leaves nineteen
orders of headroom on the Stage-0 integrand) and catastrophic cancellation (real, but
unaffected by the choice of unit). With no precision case, what remained was faithfulness
of `repr` — seeing `20 micrometer` rather than `0.002 centimeter`. That is cosmetic, and
the unit a user actually typed is already preserved where it is visible: spec files are
written in display units, and `Parameters.display` converts to any unit on demand. Against
it, storing as given breaks the letter of P1 ("every shared dataclass stores canonical
CGS-Gaussian values") and makes `m` a pint conversion — measured at 65 µs against 94 ns for
a plain attribute read, a 700x difference. That is negligible in the vectorized design,
where the laser is sampled once per chunk of ~1e6 particles, but it is a live trap for any
caller that reaches for `a0_profile` inside a Python loop.

**The same, but with the canonical magnitudes precomputed into a private dict so `m` stays
free.** Removes the performance objection and keeps the nicer `repr`, at the price of
storing each value twice and adding state that is not quite what P9 forbids but is adjacent
to it. It is the option to revisit if the display of stored values ever becomes a real
complaint; until then it buys presentation, not correctness.
