# RES004 — `FieldSpec` gains `choices` and `integer`; `FieldKind` stays four members

Status: implemented
Type: architecture
Archived: 2026-08-10

## Problem

`GRAND_PLAN.md` §3.1 sketches `FieldSpec`/`FieldKind` without naming how a choice field's
options are represented or how an integer-valued field is distinguished from a general
numeric one.

## Decision

`FieldSpec` (`schema.py`) carries two fields the `GRAND_PLAN.md` §3.1 sketch did not name:
`choices` (required for `FieldKind.CHOICE`, forbidden otherwise) and `integer` (a numeric
field whose value must be integral — `n_particles`, `seed`, bin counts). `FieldKind` keeps
exactly the four members the plan lists.

## Alternatives considered

**A fifth `INTEGER` member of `FieldKind`.** Rejected — see Rationale.

## Rationale

`FieldKind` answers one question — does this value carry a width convention, and is it a
number at all? `WIDTH`/`DURATION` do carry one, `SCALAR` does not, `CHOICE` is a string.
Integer-ness is orthogonal to that axis, exactly like `value_range`, and folding it in
would make the enum answer two unrelated questions at once. A `CHOICE` field with no
`choices` is unusable, so that slot is not an extension of the vocabulary either — it is
the sketch being completed.
