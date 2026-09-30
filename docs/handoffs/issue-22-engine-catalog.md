# Issue #22 — Retire Kascade and establish a generic engine catalog

Status: implementation handoff  
Issue: #22 — Retire Kascade and establish a generic engine catalog  
Branch: `issue-22-engine-catalog`  
Base: current `main` at handoff creation (`ee37092443ec87e687fbb9f40280f7493c3629ec`)

## Goal

Remove the unmaintained Kascade engine and finish the architectural separation already begun by the shared `Engine` protocol: GammaForge should expose a small, public engine catalog through which scripts and the GUI can enumerate, configure, select, and run engines without importing or inspecting their implementation internals.

This task is architectural cleanup. It must not invent replacement Compton physics.

## Authority

The durable task record is GitHub issue #22.

Relevant repository decisions and current code are:

- `RES018`: the `Engine` protocol and `RecomputeCost` contract are the established engine boundary.
- `RES058`: `CalculationRequest` + `LocalRunner` form the frontend-independent local execution seam; analytical preview remains a special always-visible overlay.
- `RES059`: historical authority for why the current minimal Kascade port was added. Once Kascade is removed, this decision is no longer load-bearing and should be archived according to the decision workflow rather than rewritten.
- `AGENTS.md`: shared I/O, typed schemas, no GUI engine internals, no speculative capability machinery, and documentation/test expectations remain binding.
- Current `main` code is authoritative for the concrete layout.

There is no `DERNNN` that makes Kascade a maintained physics specification. Do not re-derive Kascade or create replacement physics in this task.

## Current state on main

### Engine contract

`src/gammaforge/engines/base.py` defines a runtime-checkable `Engine` protocol with:

- stable `name`;
- typed `schema: Parameters`;
- declarative `supported_outputs`;
- declarative `recompute_costs`;
- `run(interaction, params) -> Results`.

This contract should remain the core execution abstraction unless a minimal extension is genuinely required by the catalog.

### Enumeration and execution

`src/gammaforge/engines/runner.py` is still coupled to concrete engines. It imports `XigmaEngine`, `KascadeEngine`, and `DeltaEngine` directly and constructs:

```python
{"xigma": XigmaEngine(), "kascade": KascadeEngine(), "delta": DeltaEngine()}
```

when no engine mapping is injected.

That hard-coded dictionary is the main architectural gap: consumers have a generic engine contract, but there is no single public catalog/discovery boundary.

`LocalRunner(engines=...)` already supports dependency injection. Preserve that useful test/specialized-caller seam.

### GUI and request boundary

`CalculationRequest.engine_params` is a mapping from engine name to typed `Parameters`.

`InputState` receives an engine mapping, constructs per-engine parameter groups from each engine's schema, checks `supported_outputs`, and builds requests without importing engine implementation modules. This is already close to the desired frontend behavior.

The GUI gets its engines through `Workspace.runner.engines`, so making the runner use the public catalog should make the GUI generic automatically. Remove remaining hard-coded presentation text such as the about/help line that currently names Kascade explicitly.

GUI defaults already ignore saved engine sections whose engine no longer exists. Preserve this behavior so an old `~/.config/gammaforge/gui-defaults.yaml` containing `kascade` does not break startup.

### Kascade

Active implementation:

- `src/gammaforge/engines/kascade/__init__.py`
- `src/gammaforge/engines/kascade/engine.py`
- `src/gammaforge/engines/kascade/schema.py`
- `src/gammaforge/engines/kascade/solver.py`
- `tests/test_kascade_engine.py`

Current documentation also mentions Kascade in `README.md`, `AGENTS.md`, `src/gammaforge/engines/__init__.py`, and the engine/dataflow walkthrough.

`RES059` records why the minimal port was created. Preserve that historical reasoning; archive the decision once the implementation has actually been removed.

### Validation implications

`src/gammaforge/validation/run.py` currently states that Kascade is not wired into the production validation tier. Its production comparisons are Xigma/analytical plus Delta reference checks. Therefore removal must not be represented as preserving a Kascade validation gate that never existed.

Open issue #10 explicitly requests future four-method Xigma/Delta/analytical/Kascade validation. That task conflicts with issue #22 and must be reconciled once Kascade is retired.

Historical evidence that happened to use Kascade can remain historical evidence when still scientifically meaningful; do not delete records merely to erase the name.

### Delta ambiguity

The repository currently has `src/gammaforge/engines/delta/` and `LocalRunner` includes `delta`, while some prose still describes Delta as validation-only. The new catalog must make this distinction declarative rather than forcing the GUI or scripts to know special engine names.

## Required work

### 1. Remove Kascade from active implementation

Delete the Kascade engine package and its dedicated implementation tests.

Remove Kascade from:

- default executable engine enumeration;
- GUI engine choices and hard-coded help/about text;
- current README/layout/status descriptions;
- active walkthrough material;
- active validation/planning text that assumes Kascade will be expanded.

Do not retain compatibility shims that import a removed `KascadeEngine`.

Old GUI defaults containing a `kascade` section should continue to be ignored harmlessly.

### 2. Add one public engine catalog

Implement one small authoritative catalog/discovery boundary under `gammaforge.engines`.

The exact internal representation is an implementation choice. Prefer the smallest design that satisfies the contract; examples could be an immutable mapping of descriptors/factories plus small query helpers. Do not add a mutable global plugin manager or a broad capability-negotiation framework.

The public boundary must allow ordinary callers to:

- enumerate available user-facing calculation engines;
- obtain/select an engine by stable name;
- access the engine schema and declared output/recompute metadata through the same abstraction;
- construct a runner/request flow without importing `gammaforge.engines.xigma.engine` or any other concrete implementation module.

There should be one source of truth for engine enumeration. Do not create separate GUI and script registries.

### 3. Represent role/visibility declaratively

GammaForge needs to be able to contain engines/references with different roles without name checks in frontends.

At minimum, the catalog must be able to express whether an implementation is an ordinary user-selectable calculation engine versus an internal/validation-only engine/reference. Resolve the current Delta ambiguity through this mechanism.

Do not over-generalize this into arbitrary feature negotiation. Existing `supported_outputs`, schema, and recompute-cost declarations remain the capability contract.

### 4. Make LocalRunner consume the catalog

When `LocalRunner` is constructed without an injected engine mapping, obtain its default selectable engines from the public catalog rather than importing concrete engine classes in `runner.py`.

Preserve explicit injection of an engine mapping for tests and specialized callers.

Keep the sampling cache, error isolation, status callbacks, and result semantics unchanged.

Analytical preview/overlay behavior from RES058 remains unchanged in this task. It may continue to be handled specially by the runner if that is the smallest design consistent with current behavior.

### 5. Keep GUI and serialization engine-agnostic

The GUI must derive engine choices, engine parameter forms, and supported-output gating from the same catalog/runner boundary.

Do not add engine-name branches to GUI modules.

`CalculationRequest` remains the shared immutable snapshot and keeps engine parameters keyed by stable engine name unless a concrete compatibility problem requires a narrowly justified change.

Input/calculation serialization should resolve engine schemas through the public catalog rather than requiring a caller to know implementation modules. Preserve current error behavior for genuinely unknown engine names.

### 6. Update documentation and durable decisions

Because this changes the shipped architecture:

- create the appropriate new architecture `RESNNN` after the implementation is settled, following `docs/decisions/README.md`;
- describe the catalog, its role/visibility semantics, and why it replaces hard-coded runner enumeration;
- reference the relevant existing decisions rather than rewriting their history;
- archive `RES059` when Kascade is actually removed, adding only the archival/supersession metadata required by the decision workflow;
- update `docs/decisions/INDEX.md`;
- update `README.md`, `AGENTS.md`, and `notebooks/02_engine_stages_and_dataflow.py` (and regenerate its notebook form with the repository tool);
- repair stale engine-package prose, including the current inconsistency about where Delta lives.

Do not create a physics derivation for this architectural task.

### 7. Reconcile issue tracking

Issue #10 must not remain as an apparently actionable request to expand Kascade validation after this work is complete. Update or close it as obsolete/superseded by the Kascade retirement, preserving useful Delta/angular-aperture work separately if it still deserves tracking.

Review other open issues for statements that assume Kascade is a current engine. Correct only materially stale task descriptions; do not rewrite historical evidence.

## Invariants

- Xigma physics and numerical behavior do not change.
- Shared `gammaforge.io` CGS-Gaussian inputs/results do not change.
- `Engine.run(interaction, params) -> Results` remains the execution model.
- Engine numeric controls remain typed `Parameters`; no mutable engine config object is introduced.
- GUI code does not import engine stages, `Collision`, solvers, or concrete model internals.
- `CalculationRequest` remains frontend-independent.
- Analytical preview/overlay behavior remains available as specified by RES058.
- Existing `LocalRunner` injection remains possible.
- Removed Kascade validation coverage is not silently relabeled as passing coverage.
- No compatibility layer is kept solely to preserve imports of Kascade.

## Validation and acceptance criteria

Keep tests focused on the architectural contract rather than replacing every deleted Kascade test with a new micro-test.

At minimum verify:

1. The public catalog enumerates the intended user-facing calculation engine(s) and can resolve them by stable name.
2. Validation-only/internal entries, if kept in the catalog, are not exposed as ordinary GUI choices unless explicitly requested through an appropriate API.
3. `LocalRunner()` uses catalog-provided selectable engines and still accepts an injected fake/custom engine mapping.
4. A representative script-facing path can choose an engine by name, obtain its schema, build a `CalculationRequest`, and execute it without importing the concrete engine module.
5. GUI state renders/accepts catalog-provided engine schemas and supported outputs without name-specific code.
6. Unknown engine names still fail clearly.
7. Old GUI defaults with a removed `kascade` section remain harmless.
8. Existing Xigma runner behavior and analytical estimates remain unchanged.
9. Kascade source/tests/current docs are absent after the removal.
10. The normal repository checks pass.

Run, as appropriate:

```bash
pytest -m fast
pytest
python tools/build_notebooks.py --run
```

If the full suite has environment-specific blockers, record exactly what ran and what did not; do not claim success for unexecuted checks.

## Deliverables

Persistent branch outputs should include:

- Kascade code/test removal;
- public engine catalog/discovery API;
- runner integration with the catalog;
- any minimal role/visibility metadata needed for Delta/user-facing separation;
- focused catalog/runner/GUI tests;
- updated current documentation and rebuilt walkthrough notebook;
- new implemented architecture `RESNNN`;
- archived `RES059` with correct index update;
- issue-tracking reconciliation for #10 and materially stale Kascade assumptions.

The temporary handoff file is not a deliverable. Remove it before the PR is ready to merge.

## Out of scope

- a replacement Monte-Carlo engine;
- re-deriving Kascade or preserving its equations;
- new Xigma physics;
- changes to DER confidence states;
- dynamic loading of arbitrary separately installed third-party Python packages via entry points;
- remote workers, LAN transport, scheduling, cancellation, progress protocols;
- broad redesign of `Results`, units, beam/laser schemas, or validation architecture.

The catalog should leave a clean future seam for external/plugin discovery if that is later desired, but this branch should solve the concrete in-repository multi-engine problem first.

## Open questions and blockers

There is no known physics blocker.

One architectural choice is intentionally left to implementation: the smallest concrete catalog representation and public function names. Judge it against the acceptance criteria above, not against a desire for a general plugin framework.

If implementation reveals that Delta cannot be cleanly classified without changing scientific validation semantics, preserve its validation behavior and surface the conflict in the PR instead of hiding it behind a GUI-specific exception.
