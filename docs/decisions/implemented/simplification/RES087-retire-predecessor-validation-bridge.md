# RES087 — Retire the predecessor validation bridge

Status: implemented
Class: simplification

## Problem

GammaForge originated as a ground-up rebuild and initially used committed snapshots from
the earlier implementation as a transitional cross-check. That mechanism required a
second checkout and interpreter to regenerate nine HDF5 files, retained translation code
for an obsolete schema, and could not represent newer geometry, chirp, pulse, or
collimation features. The default runner used the snapshots only for repeated scalar
checks; its distribution comparisons were not active. Meanwhile the maintained native
suite now has closed-form identities, analytical/xigma comparisons, CPU and CUDA delta
references, convergence gates, invariance properties, and kascade's Thomson anchor.

## Decision

GammaForge is operationally independent. The snapshot generator, subprocess driver,
generic golden-snapshot layer, committed external-result files, and their environment
variables are removed. `validation.run` reports only checks reproducible from this
checkout. The scenario bank, comparison metrics, native delta references, engine
invariance checks, and current cross-engine gates remain.

This decision supersedes RES019. RES019 is archived as the accurate record of why the
subprocess boundary and committed snapshots were appropriate during the rebuild.
Current-facing documentation states that GammaForge originated from ComptonSuite and is
now completely independent; historical plan, derivation, review, and decision text retains
its provenance.

## Alternatives considered

**Keep the bridge dormant.** This would preserve a regeneration path, but also preserve
an external checkout contract and obsolete translation surface that no maintained
validation gate needs.

**Regenerate snapshots from GammaForge itself.** Self-generated fixtures would freeze
numerical output without providing an independent physics implementation. Convergence,
identity, and cross-method comparisons are stronger and state what failure mode they
cover.

**Erase every historical reference.** That would contradict the permanent decision-id
and archive rules and would make the design provenance less honest. Independence is an
operational property, not a claim that the project has no history.

## Rationale

The external snapshots covered only the original three Gaussian scenarios and rejected
newer inputs that their schema could not carry. Their default-run contribution was 18
near-duplicate checks of `a0_peak`, `gamma0`, electron count, and photon count. The one
direct Stage-0 yield comparison is now covered more transparently by the analytical
closed-form anchor and delta normalization identity. Removing the bridge makes every
shipped validation command reproducible from the repository while preserving more
independent numerical and physics paths than the snapshots exercised.

## Consequences

No code, test, environment variable, or committed data file refers to or requires the
earlier checkout. HDF5 result serialization and `validation.metrics.compare_slices`
remain general project facilities. Historical documents may still name ComptonSuite,
but code comments and docstrings describe current behavior intrinsically.

The retirement does not close the independently tracked arbitrary-angle and four-method
coverage gaps. Those remain visible in `PROGRESS.md` and production validation reports.

## Amendments

> **2026-09-29 — Documentation relocation.** The retired `PROGRESS.md` backlog moved to GitHub issues. Issues #1 and #10 track the independent arbitrary-angle and four-method coverage gaps.
