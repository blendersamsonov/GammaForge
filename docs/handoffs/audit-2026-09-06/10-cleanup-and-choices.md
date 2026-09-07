# 10 — Mechanical cleanup, with explicit gates on API/UX changes

Read [coordination rules](README.md). Covers residual A16/A18/A19/A20/A21 and
the audit's removal candidates. Run after overlapping fixes or leave owned files
to their active session. This is not permission for broad “tidying.”

## Ready cleanup: verify each candidate before deleting

At baseline these imports appeared unused:

- `Results` in `src/gammaforge/io/plotting.py`.
- `OutputKind` in `src/gammaforge/validation/make_references.py`.
- `math`, `C_CGS`, `EV_CGS` in `tests/test_target_results_interaction.py`.
- `ShapeTable` in `tests/test_stage1_stage2.py`.

The repeated-axis check in `PhasespaceSlice.__post_init__` compares a mapping's
keys to their frozenset; a mapping cannot contain duplicate keys. Let 02 remove
that branch while fixing the real contract. Two empty example/template test trees
always skip; remove vacuous template scaffolding only after checking no real
examples have landed. Seventeen `.gitkeep` files were tracked: remove redundant
ones in nonempty folders; retain intentionally represented empty lifecycle folders
unless their documented/discovery behavior genuinely does not require them.

`recoil_parameter` in `engines/kascade/solver.py` had no caller and was absent from
its export list; coordinate any removal with 04 and verify its calculation is not
used indirectly. `TYPE_CHECKING` imports are not automatically dead: the `Engine`
annotation import in GUI state was genuinely used.

## Consolidation within existing boundaries

Review moving the common target field declarations from GUI state to the shared
schema/field layer, without a generic registration system or changed defaults.
Coordinate with 02/06/08. Trim duplicated historical rationale from owned code to
decision pointers under RES056, but keep information needed to use/verify formulas.
Let 03 own supported-output/default consolidation and 01 own any approved
production polarization consolidation. Keep independent reference math independent.

## Review only: do not silently remove

| Candidate | Decision needed |
|---|---|
| Public `emit_norm_x/y`, `plot_collimated`, `matplotlib_geometry`, `retargeted_luminosity`, `a0_shape_centers` | No in-repo callers were found; external notebooks may use them. Confirm usage or propose deprecation, not deletion on grep alone. |
| Analytical `estimate_yield` and related helper/tests | RES040 deliberately preserves predecessor port fidelity. The author's open choice in PROGRESS must be resolved first. |
| Collision's multi-intensity `_tables` cache | No public retarget workflow today, but tests exercise it and RES032 describes it. Coordinate 06's reuse proposal; simplify only with an explicit contract/decision update. |
| GUI lock/release controls | `gui/app.py` locks most fields after a calculation despite stale-result/Calculate gating. Removing this changes the chosen workflow; present the UX tradeoff and obtain approval first. |
| Polarization no-op markers | Replace weak marker-only tests with behavior coverage before deletion; defer physics truth to 01 and documentation reconciliation to 09. |

For the GUI proposal, preserve Calculate gating, charge-only exact rescaling,
submitted-result snapshots and worker exclusion. Recompute cost is not inherently
an editing-permission policy, but the current plan deliberately couples them;
update it only after the user chooses a new behavior.

## Acceptance

Each deletion has a search/usage justification and passing affected tests. Public
API changes and changed workflow are listed separately from no-behavior-change
cleanup. No scientific formula/default, archived decision, user file, active
worktree or deliberately independent Delta formula is deleted. Avoid massive
formatting diffs. Deliver the small cleanup plus a concise decision queue for the
items that remain gated; do not call those unimplemented choices completed fixes.
