# Evidence baseline and coverage

Audited source: `f59f6bf`. This ledger condenses the earlier audit for durable use by
other sessions; A01–A22 are local handoff identifiers, not RES/DER decisions. An
entry may contain a reproduced defect, a source-inspected gap, or a design choice;
its handoff distinguishes the implementation scope from recommendations.

| ID | Finding / evidence | Owner |
|---|---|---|
| A01 | Polarization implementation ignores supplied electron direction; a co-tilted electron/observer probe changes the factor from 1 to about 2.5e-7. Delta imports the same production factor. Physics expectations need author review. | 01 |
| A02 | Validation CLI supplies no production engines; shared reference code and three closely related head-on scenarios leave common-mode and geometry gaps. | 01 |
| A03 | Kascade histogram density loses edge-bin mass when integrated with trapezoids over centers: two unit bins integrate to 1 instead of 2; a 2×2×2 cube to 1 instead of 8. | 02 |
| A04 | Table-free xigma spectrum broadcasts particles × energy samples without a bound: 1,000,000 × 2,048 doubles = 16.384 GB per full temporary. | 03 |
| A05 | HDF5 omits final electrons and model metadata; default load returns string output keys. GUI result downloads lack the complete request; custom snapshot YAML has no matching complete-request loader. | 05 |
| A06 | Last-emission time uses a zero-initialized maximum, incorrectly replacing every all-negative history's last time with zero; final positions depend on it. | 04 |
| A07 | Empty xigma bunches work for yield/spectrum but can raise for angular/collimated outputs; unsupported requests can fail during autoranging before omission. | 03 |
| A08 | Declared NumPy minimum predates `np.trapezoid`; optional GUI tests fail in a core-only environment; dev extras do not declare browser/symbolic tooling. | 08 |
| A09 | No tracked CI/check workflow or reproducible dependency specification beyond broad package lower bounds; tested local environment is not the supported-version matrix. | 08 |
| A10 | Doc guard repeatedly traverses the checkout and resolves paths by basename. Nested worktrees can make nonexistent current-checkout references pass. | 09 |
| A11 | AGENTS/plan still describe polarization no-ops while markers are false and PROGRESS says closed; derivation usage/status prose is also inconsistent. | 09, informed by 01 |
| A12 | Symbolic scripts can print failed comparisons without failing; selector `verify_all.py der004` raises TypeError; claimed numerical verifier files are absent. | 01 |
| A13 | Frozen dataclasses contain mutable mappings/arrays; CalculationRequest's dict can change; Collision inputs can be replaced after intermediates are cached. | 06 |
| A14 | Direct constructors admit fractional counts, invalid seeds, non-finite scalar fields, unequal particle-array lengths, and decreasing axes. One-sample output requests conflict with current integration. | 02 for axes/output counts; 06 otherwise |
| A15 | A protocol-conforming non-Gaussian laser still fails through Gaussian-only fitting; descriptive fit supplies physics-critical wavelength/intensity/polarization metadata. | 07 |
| A16 | Runner only reuses sampled interactions; xigma creates a new Collision each run. Private multi-intensity table cache has no public retarget workflow. | 06; removal review in 10 |
| A17 | Xigma's table-free spectrum omits nonlinear redshift without Results-level warnings; offset-insensitive spatial autoranging deserves a separate reproducible check. | 03 for warnings; 02 for autorange investigation |
| A18 | Duplicate production polarization paths, supported-output declarations and defaults; target schema lives in GUI. Some duplicate reference formulas are intentionally independent and must stay so. | 03 for owned declarations; 01 for physics; 10 for schema cleanup |
| A19 | GUI lock/release state duplicates some stale-result/Calculate workflow responsibilities. Removing it changes deliberate UX, not a consequence-free cleanup. | 10, approval required |
| A20 | Long mandatory documents and historical rationale in code increase onboarding cost; navigation and current truth need separation without erasing provenance. | 09; 10 for owned comments |
| A21 | Structural doc tests can skip malformed filenames and obscure duplicate ids/status inconsistencies; two empty template test trees always skip; a correctness test uses wall-clock timing. | 09 for format guards; 08 for timing; 10 for templates |
| A22 | GPU/JIT extras install dependencies for implementations that remain gated stubs. Removing public extras has a compatibility cost; labeling them honestly is safe. | 08 |

## Previous audit checks (not rerun in this handoff-writing session)

- `.venv/bin/pytest -q`: **470 passed, 3 skipped, 20 warnings**, about 314 s.
  Skips included empty template examples and the opt-in browser test. Warnings
  included zero-gamma divisions/invalid products in the polarization path.
- `GAMMAFORGE_BROWSER_TEST=1 .venv/bin/pytest -q tests/test_gui_browser.py`:
  **1 passed**, about 26 s.
- `.venv/bin/python -m gammaforge.validation.run`: printed all checks passing,
  explicitly with **no production engines supplied**. This is a coverage gap,
  not evidence that the production methods agree.
- `.venv/bin/python scripts/verifications/verify_all.py`: exit 0, subject to A12.
  Selecting `der004` failed with the Path membership TypeError.
- `pip check`: no broken installed dependencies. A simulated missing-GUI-dependency
  test failed in plotting. No actual clean-install/minimum-version matrix was run.
- Local environment then: Python 3.14.6, NumPy 2.5.1, pytest 9.1.1, NiceGUI 3.16.0,
  Plotly 6.9.0. These versions are observations, not new support requirements.
- Additional diagnostics reproduced A01, A03, A05–A07, A10, A13–A15.

During handoff preparation, the two slice-mass examples and direct polarization
factor values were rechecked at the same commit. Their runnable snippets are in
02 and 01. The other numeric observations remain prior-audit evidence to reproduce,
not freshly executed checks. Source entry points in each brief replace unavailable
temporary probe files.

For the NumPy compatibility fact, the official [`numpy.trapezoid` documentation](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html)
marks the API as introduced in NumPy 2.0. Session 08 should test the dependency
contract it chooses, not assume raising one bound validates the whole environment.
