# RES094 — Completion requires active tracker reconciliation

Status: implemented
Type: process

## Problem

GammaForge issues own unfinished work while pull requests and temporary handoffs own
the execution state of particular implementation attempts. Completing one issue can
therefore invalidate assumptions recorded in other still-open work even when the code,
derivations, decisions, and validation records on `main` are already correct.

A concrete example is a numerical-policy issue landing while another validation issue
still says that policy change is pending. Closing the implemented issue alone leaves the
active tracker internally inconsistent and causes later agents to plan from obsolete
dependencies, parameters, engine roles, or acceptance criteria.

## Decision

Issue/PR completion includes a mandatory reconciliation pass over the **active tracker**.

After the final shipped state is known, the completing agent searches all open issues and
open/draft PRs for material references to the completed work. The search is semantic as
well as literal: issue and PR numbers, DER/RES identifiers, feature/component names,
removed configuration parameters or interfaces, and dependency wording are all relevant.

Each match is classified as either:

- **historical provenance**, which remains unchanged; or
- **an active statement about current/future work**, which is updated when the completed
  work makes it false, obsolete, newly unblocked, redundant, or narrower.

The reconciliation applies to open issue titles/bodies, open PR descriptions, and
branch-local handoffs that still govern unfinished work. A different open issue may be
closed or superseded when the completed change fully satisfies it; partial overlap should
instead narrow or revise that issue.

Where an explicit cross-repository dependency exists, the same reconciliation is applied
to the linked active work in that repository.

Completion reporting records either the tracker artifacts that were updated or that the
scan found no required changes.

## Alternatives considered

**Update only the issue being closed and its own PR.** Rejected because dependency drift
usually appears in *other* active issues and PRs. This preserves a clean local history
while leaving the actual backlog inconsistent.

**Search only for literal `#N` references.** Rejected because stale assumptions often
survive as feature names, old parameters, component roles, or prose such as "once the new
grid lands" without mentioning the issue number.

**Rewrite every historical reference after a merge.** Rejected because closed issues,
merged PRs, DER/RES records, validation reports, and commit history are historical
evidence. Retrofitting them erases chronology and creates needless churn. Only active
task/execution state is reconciled unless a durable current record is factually wrong
under its own maintenance rules.

## Rationale

The repository deliberately separates current truth (code, DER/RES records, validation
evidence) from unfinished work (issues) and execution state (open PRs and handoffs).
A completion-time dependency scan is the natural point to keep those active layers
consistent with what actually shipped, while preserving historical artifacts unchanged.

## Consequences

- Agents must budget a tracker scan as part of issue/PR completion rather than treating
  merge as the final operation.
- Open issues and PRs should stop accumulating references to already-completed blockers,
  retired parameters, or components that no longer exist.
- Branch-local handoffs may require a follow-up edit when another merge changes their
  assumptions before they themselves merge.
- The scan can legitimately produce no edits; the requirement is to perform and report
  the reconciliation, not to manufacture changes.
