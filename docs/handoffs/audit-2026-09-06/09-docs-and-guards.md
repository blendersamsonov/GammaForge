# 09 — Fast checkout-scoped guards and consistent instructions

**Status:** Done (2026-09-06)

Read [coordination rules](README.md). Covers A10, A11, A20 and structural A21.
This can start independently, with a coordinator for shared plan/index edits.

## Entry points and evidence

`tests/test_doc_staleness.py`, `test_decision_format.py`,
`test_derivation_format.py`; `AGENTS.md` (not its CLAUDE symlink),
`docs/GRAND_PLAN.md`, `PROGRESS.md`, decision/derivation READMEs/indexes and active
entries, especially RES034, RES054, RES055 and DER004–DER007.

The guard uses repeated `rglob` traversals and basename resolution. Baseline probes
accepted nonexistent `src/does/not/exist/laser.py` and root `DECISIONS.md`; the latter
was found only inside a nested `.claude` worktree. A scan took roughly a second;
an in-memory one-index prototype reduced the guard run to about 2.84 s. A tracked-file
index exposed a stale RES055 reference that nested files had hidden. These timings
are observations, not portable pass/fail thresholds.

Current documents disagree: plan/AGENTS say polarization paths are no-ops;
PROGRESS says closed and source markers are false. Some derivations still say
“not implemented” or cite absent verifiers. DNNN/DVNNN examples disagree with real
RES/DER identifiers. The plan is about 14,000 words with architecture starting
after roughly 327 lines of change history.

## Work

1. Build one intended-checkout file/symbol index per check run. Resolve qualified
   paths as paths, not merely basenames. Handle bare filename shorthand under an
   explicit policy. Exclude nested worktrees, virtualenvs and generated graphs
   before traversal, not only after expensive recursion.
2. Choose tracked-file indexing versus an explicitly pruned source-tree index and
   document untracked/new-file and source-archive behavior. Do not make tests
   accidentally depend on a local `.git`, graph or sibling checkout.
3. Make malformed doc filenames and duplicate ids visible. Cross-check path,
   declared status and index semantics carefully: archived decisions intentionally
   retain their original `Status:` header. Do not rewrite history to force a naive
   “folder equals status” rule.
4. Reconcile current-state facts against code and actual evidence from 01. Update
   the authoritative plan before dependent instructions. Do not claim author
   review/verification happened merely because a file is under `verified/`.
5. Reduce onboarding duplication: current architecture and a navigable map before
   lengthy history, with stable links if history moves. Preserve principle rationale,
   permanent ids, derivations and archived reasoning. Keep PROGRESS short and
   current; do not append a remediation diary or turn these handoffs into authority.

## Acceptance

- Temporary-fixture tests prove a nested worktree/generated file cannot satisfy a
  missing current-checkout path, while legitimate same-checkout references work.
- A test/spy proves indexing occurs once, without fragile wall-clock assertions.
  Report before/after timings on the same machine as supporting evidence.
- Malformed names, duplicate ids and genuinely inconsistent indexes fail. Archived
  records with preserved historical statuses still pass according to the lifecycle.
- The actual guard is green because active references were corrected, not silently
  excluded. Physics uncertainty is stated consistently, without inventing closure.

09 owns broad documentation surgery; other sessions supply narrow changes and
evidence for their landed work. Do not renumber/delete RES/DER entries, erase
existing worktrees, or introduce a general Markdown compiler to solve this guard.

Starting check: `.venv/bin/pytest -q tests/test_doc_staleness.py tests/test_decision_format.py tests/test_derivation_format.py`.
