# 01 — Physics review and trustworthy validation

Read [coordination rules](README.md) first. Covers A01, A02, A12 and the
physics-dependent part of A18. Two separate deliverables: make validation reporting
trustworthy now; resolve emission physics only after the author's review.

## Entry points and evidence

- `src/gammaforge/engines/xigma/stages.py`: `polarization_factor`, its vectorized
  counterpart, and `spectrum_from_table`.
- `src/gammaforge/validation/references/delta.py`: `resonance_spectrum` imports the
  production polarization function, and passes relative electron/observer angles
  into arguments documented as electron trajectory angles.
- `src/gammaforge/validation/{run,scenarios,runners,invariance}.py` and
  `tests/test_validation.py`, `tests/test_stage0_delta.py`.
- `scripts/verifications/verify_all.py`, the DER004–DER007 verification scripts,
  and the corresponding derivations. Read RES031, RES033, RES053, RES054.
- Physics authority: `/home/alexander/Work/Papers/2026/Compton-Numerics`, especially
  the vector/polarization definitions in the current manuscript, not remembered
  equations or the audit's interpretation.

Rechecked baseline diagnostic, runnable with the project Python:

```python
from gammaforge.engines.xigma.stages import polarization_factor
for tilt in (0.0, 0.0005):
    print(polarization_factor(2000, tilt, 0, tilt, 0, 0, 0))
# Baseline: 1.0, 2.499999374183659e-07
```

The scalar implementation does not use its electron-angle arguments. Delta's
resonance uses relative angles but its factor uses absolute observer angles, so
both paths can agree for the wrong reason. The probe is evidence of a coordinate
problem, **not an approved expected formula or a requirement of exact invariance
under this particular partial rotation**.

## Ready scope

1. Reproduce the discrepancy at scalar, vectorized and Delta call boundaries.
   Prepare an author-review packet: parameter conventions, relevant manuscript
   equations, minimal numbers, and which observations existing tests miss. Mark
   the affected physics BLOCKING and stop that portion before changing formulas.
2. Wire a small, explicit production-validation tier into the suite, using the
   shared scenario bank. Report exactly which engines, observables, approximations
   and shared upstream computations each comparison exercises. Keep expensive
   convergence runs separately selectable and missing coverage visible.
3. Add distribution-sensitive checks, not just integrated counts: e.g. centroid,
   edge and angular structure within each method's documented common validity
   regime. Do not require agreement where methods deliberately approximate
   different physics. Coordinate histogram integration with 02 and timing with 04.
4. Make symbolic claims executable: failed comparisons must produce failure;
   fix selector matching against Path names; test unknown selectors and failing
   subprocesses. Resolve absent numerical-script claims by locating actual evidence
   or correcting the claim, never by adding an unconditional “verified” stub.
5. After author approval only, implement the approved convention consistently.
   Preserve an independent reference calculation; sharing the production factor
   does not independently validate it. Coordinate `stages.py` with 03/06.

## Acceptance and limits

- A deliberately wrong supported engine produces a failed suite result/nonzero
  CLI exit, while omitted or blocked checks cannot appear as scientific passes.
- Tests prove a false symbolic identity and a failing child process fail the
  wrapper. Both all-script and single-script commands work when dependencies exist.
- The report names common-mode inputs and includes a genuinely independent
  emission check for any physics fix declared verified.
- Relevant pytest tests stay green by testing honest failure reporting; do not
  silently exclude known failing physics, bless today's bad value, inflate MC
  tolerances, or overwrite goldens. An explicitly reported unresolved check is
  unfinished science, even if harness unit tests pass.

Ask 08 to provide symbolic/test dependencies. Supply 09 with verified facts about
derivation status; don't promote or demote confidence based on prose alone. A
postmortem may be warranted after the cause is settled; follow its actual criteria.

Starting checks: `.venv/bin/pytest -q tests/test_validation.py tests/test_stage0_delta.py`,
`.venv/bin/python -m gammaforge.validation.run`,
`.venv/bin/python scripts/verifications/verify_all.py`, and
`.venv/bin/python scripts/verifications/verify_all.py der004`.
The last two require sympy; the final command is expected to reproduce A12 before
the fix. The production tier's new command must be documented when implemented.
