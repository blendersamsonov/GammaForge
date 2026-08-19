# RES007 — `Axis` members carry a `(key, unit)` pair

Status: implemented
Class: bug-fix
Archived: 2026-08-10

## Problem

`Axis.X` was first written with the unit string alone as the enum value (e.g. `"cm"`), and
a failing test found that this silently merges distinct axes.

## Decision

`Axis.X` is `("x", "cm")`, not `"cm"`; the unit is reachable as `Axis.unit` and a stable
serialization key as `Axis.key`.

## Alternatives considered

**The unit alone as the enum value, as first written.** Rejected — see Rationale.

## Rationale

Not a style preference — a bug found by a failing test. `X` and `Y` share `cm`, and
`THETA_X`/`THETA_Y` share `rad`; `Enum` silently collapses members with equal values into
**aliases**, so `Axis.Y is Axis.X` was true and every two-dimensional slice quietly lost an
axis. Distinct keys keep the six members six, and give HDF5 a dataset name that does not
depend on the Python identifier.
