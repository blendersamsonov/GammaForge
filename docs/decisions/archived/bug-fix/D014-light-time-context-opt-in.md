# D014 — The `light_time` context is opt-in per field, not global

Status: implemented
Class: bug-fix
Archived: 2026-08-10

## Problem

A test written to check that a wrongly dimensioned value is refused found that it was not:
with `light_time` enabled globally, a transverse beam size was happily accepted in
femtoseconds.

## Decision

`to_canonical`/`from_canonical`/`as_canonical_quantity` take `light_time=False` by default.
Only fields listed in a class's `LIGHT_TIME_FIELDS` (`GaussianElectronBeam.sigma_z`,
`GaussianParaxialLaser.duration`) and `FieldKind.DURATION` schema fields opt in. The
`gaussian_charge` context stays unconditional.

## Alternatives considered

**Applying both contexts everywhere, as the first implementation did.** Rejected — see
Rationale.

## Rationale

Globally enabled, `light_time` equates *any* length with *any* duration, so a **transverse**
beam size was happily accepted in femtoseconds. That is not a unit choice, it is a
different physical quantity. §2.1 introduces the context for one specific pairing — a
longitudinal extent that may be quoted either way — and scoping it to exactly that keeps
the rest of the dimensional checking meaningful. `gaussian_charge` needs no such scoping: a
value either is a charge or is not, and the SI/Gaussian split is notational rather than a
physical ambiguity.
