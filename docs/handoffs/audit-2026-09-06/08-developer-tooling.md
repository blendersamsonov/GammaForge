# 08 — Reproducible developer setup and honest test tiers

Read [coordination rules](README.md). Covers A08, A09, A22 and A21's wall-clock
correctness test. This can start independently; own packaging and check commands.

## Evidence and entry points

`pyproject.toml`, top-level `README.md`, `tests/test_gui_plotting.py`,
`tests/test_gui_browser.py`, `tests/test_stage1_stage2.py`, and symbolic scripts.
At baseline NumPy is declared `>=1.26` while code uses `np.trapezoid` (introduced
in 2.0). The dev extra contains only pytest; GUI, browser and sympy requirements
are not a complete runnable developer recipe. Some plotting tests require GUI
extras unconditionally. No tracked CI/check workflow or lock/constraints setup
was present. GPU/JIT extras install libraries for still-gated backends.

## Work

1. Make minimum dependencies match used APIs: either raise justified bounds or
   use a compatible supported API. Test the choice in a genuinely clean
   environment; do not infer support from the audit's Python 3.14/NumPy 2.5 venv.
2. Define documented core, GUI, browser and symbolic test environments. Optional
   tests skip explicitly when their extra is absent; tests for a declared installed
   tier must fail on broken imports instead of hiding failures behind broad skips.
   Browser setup includes its separate browser-binary installation requirement.
3. Add one simple local check entry point and a checked-in dependency reproducibility
   mechanism appropriate to this small Python project. Explain the split between
   reusable library dependency ranges and a pinned developer environment. Don't
   add several competing package managers or redundant task runners.
4. Add repository CI configuration for the chosen tiers, core import/smoke,
   supported minimum Python/dependency combination and a current combination,
   GUI import boundary, and built-wheel installation smoke. Preparing local
   workflow files does not authorize enabling services or changing remote settings.
5. Start with correctness-focused lint if useful; no repository-wide formatting
   or strict-type rewrite bundled into this fix. Change the Stage 1 timing
   assertion into structural/call-count coverage or an explicitly optional
   benchmark, preserving the invariant it meant to protect.
6. Label GPU/JIT extras as development-only/unimplemented unless the user approves
   their removal. Installing an extra must not promise a working backend.

## Acceptance

- In a new environment, the documented core install and tests pass without GUI
  libraries; package import does not require optional dependencies.
- The documented GUI-enabled environment runs GUI tests; browser and symbolic
  commands run when their explicit prerequisites are installed. Ask 01 for its
  required validation commands and 09 for doc checks.
- A built wheel installs/imports outside the checkout without relying on editable
  source paths or untracked files. Dependency resolution and `pip check` pass for
  the tested combinations; any unavailable platform/version is reported honestly.
- The single local check command matches the CI intent. Record exact versions and
  results, including tests deliberately excluded from the fast tier.

Never commit a virtualenv or generated browser cache. Ask for environment/network
approval through the normal mechanism where needed; don't modify another session's
shared venv while its tests are running. Leave documentation-policy rewrites to 09.
