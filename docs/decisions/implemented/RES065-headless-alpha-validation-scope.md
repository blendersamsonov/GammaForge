# RES065 — Headless alpha validation has an explicit restricted scope

Status: implemented
Type: testing

## Problem

The author wants to use analytical and xigma from scripts while GUI and kascade remain
outside the alpha release. The full production validation tier deliberately fails for
unfinished four-method and independent arbitrary-angle emission coverage, including
work outside that release scope.

## Decision

The alpha CLI selector runs the existing core, identity and golden checks plus
`production_checks` over the shared scenario bank. Every numerical failure remains
fatal. The report identifies the restricted Gaussian analytical/xigma yield and
head-on weak-field spectrum scope, and lists the omitted validation as outside that scope.
The full production selector retains its coverage blockers and nonzero exit status.

This partially extends RES031: reduced engine comparisons now have a headless alpha
gate; the full-resolution default engine loop remains opt-in.

## Alternatives considered

Treating the full production gate as the alpha gate would make the requested release
depend on kascade and unrelated GUI work. Removing its blockers globally would falsely
claim complete scientific coverage. Keeping two explicitly named scopes avoids both.

## Rationale

The same numerical gates and tolerances run in both tiers; only the declared release
scope differs. Unit tests demonstrate that a wrong numerical result cannot pass alpha.
The crossing-angle yield example supplies an additional overlap/convergence measurement;
GPU kernel checks remain separate tests and do not independently certify emission physics.

## Consequences

A passing alpha gate is not a complete physics certification. Arbitrary-angle angular
spectra remain a validation limitation, and kascade/four-method work remains open.
