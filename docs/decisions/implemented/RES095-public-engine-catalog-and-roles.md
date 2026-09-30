# RES095 — one public engine catalog, with engine role as declarative data

Status: implemented
Type: architecture

*(Numbering note: this entry was written as RES094 and renumbered to RES095 on 2026-09-30
to resolve a collision with the tracker-reconciliation workflow merged to `main` as
RES094 while this branch was in review.)*

## Problem

RES018 defined the `Engine` protocol and deliberately deferred enumeration. RES058 closed
that deferral with a small dictionary inside `LocalRunner`, which was adequate for one
GUI and one local script but left two problems as soon as a second consumer appeared.

The dictionary was the single remaining place that knew every engine by name. Anything
else that needed an engine had to either import the implementation module
(`gammaforge.engines.xigma.engine`) or import the runner and read `runner.engines`. The
first breaks the boundary the protocol exists to hold — `LocalRunner`, the GUI and
calculation serialization all gained reach into engine internals. The second makes the
runner a registry it was never meant to be: a script cannot ask which engines exist, and
`request_from_dict` still demanded a caller-supplied schema mapping because no shared
answer existed.

The second problem was the absence of a way to say what an engine *is*. `analytical` is
an always-visible preview and result overlay that must never appear as a selectable
calculation engine; `delta` is a validation reference that shares xigma's Stage 0 and so
is not a user-facing choice. Both were reachable as ordinary engines, and the distinction
survived only as prose — `engines/__init__.py` claimed delta lived in
`gammaforge.validation.references` while `engines/delta/` and the runner's dictionary both
said otherwise. Only a name check in a frontend could have expressed the intent, and none
existed, which is why the prose could drift.

Retiring kascade (RES059, now archived) forced the question. Removing an engine from a
hard-coded dictionary is a three-line change that silently requires editing a frontend
anytime; removing one from a catalog entry is a one-line change that does not.

## Decision

`gammaforge.engines.catalog` is the single public enumeration of the engines GammaForge
ships. It holds a `MappingProxyType` of name to a frozen `EngineSpec` of
`(name, role, factory)`, and offers five queries: `engine_names`, `get_engine`,
`selectable_engines`, `engine_schemas` and `estimate_engine`. Each returns fresh engine
instances, so no caller shares mutable state with another.

`EngineRole` is a three-member enum carrying that distinction as data: `CALCULATION` for
an ordinary user-selectable engine, `ESTIMATE` for the always-visible analytical
preview/overlay, and `REFERENCE` for an internal or validation-only implementation. The
current catalog is `xigma` (calculation), `analytical` (estimate), and `delta`
(reference), in that insertion order so the first entry remains the GUI's initial choice.

`LocalRunner` takes its default engines from `selectable_engines()` and no longer imports
any engine implementation module. Its analytical overlay is resolved by the estimate role
rather than a written-in name, so the name is data too. An injected `engines` mapping
still overrides the catalog unchanged, keeping the test and specialized-caller seam.

Calculation serialization resolves schemas from `engine_schemas()`: `request_from_dict`
and `load_request` take `engine_schemas=None` and default to the catalog through a
deferred import, because `gammaforge.io` is the shared layer and must not import
`engines` at module scope. Reference engines are included in `engine_schemas()` so a
request recorded against one still round-trips; a genuinely unknown name — including a
retired one — still raises `missing engine schemas`.

The GUI needed no engine-specific change to become generic. It reaches engines through
`Workspace.runner.engines`, which is now the catalog's selectable set, so delta's engine
tab is gone as a consequence of its role rather than as an exception. The only
name-bearing text left was the Settings caption, now read from
`engine_names((CALCULATION, ESTIMATE))`. The pre-existing behavior that saved GUI defaults
ignore sections for engines the workspace does not have is what keeps an existing
`~/.config/gammaforge/gui-defaults.yaml` naming a retired engine harmless.

The production validation selector's blocker was **reworded, not deleted**. With kascade
gone there is no independent emission-physics leg, and xigma/analytical share the Gaussian
overlap regime while delta shares xigma's Stage 0, so the tier still cannot separate
emission-model error. The tier keeps reporting that as missing coverage and keeps exiting
nonzero; retirement must not be recorded as passing coverage.

## Alternatives considered

**Leave the dictionary in `LocalRunner` and only drop kascade from it.** The smallest
possible diff, and it does satisfy the literal removal requirement. It leaves every
problem in the Problem section standing: the runner stays the enumeration, so a script
still cannot ask what engines exist, serialization still needs a caller-supplied schema
mapping, and adding an engine still means editing a module that has nothing to do with
running engines. It would also have left role as prose, which is the drift this change
exists to close.

**A mutable global registry with `register()` and entry-point discovery.** This is the
shape most plugin systems converge on, and it is the one RES018 explicitly rejected: a
lazy optional-dependency import table over engines with nothing to contribute is
speculative machinery. A mutable global also makes enumeration order and contents depend
on import side effects, so the GUI's default engine choice would stop being a property of
the source tree. Separately installed third-party packages are out of scope for this
change; if discovery across packages is ever wanted, this catalog is the seam to extend,
and extending it does not require breaking the query surface used today.

**Have `Engine` carry its own role as a protocol attribute.** This would put the role next
to the schema and `supported_outputs` and avoid a second declaration. It also widens
RES018's protocol for the benefit of a single in-repository catalog, and it makes role
optional in practice — a conforming engine built before this decision, or a test double,
would have no role to declare and every consumer would need a fallback. Role describes
catalog membership, which is a fact about this repository, not about the calling
convention an engine implements.

**Expose every catalogued engine to the GUI and let the user choose delta.** Delta shares
xigma's Stage 0, so a delta tab cannot independently check trajectory integration and
presenting it beside xigma invites reading agreement there as independent validation.
Hiding it is a scientific-claim decision, which is why it is expressed as a role rather
than as GUI code. Delta remains reachable by name for validation and scripts, so
retiring the tab costs the existing validation path nothing.

**Delete the production blocker now that kascade is gone.** The tier would then report no
missing coverage and exit zero, presenting a retired leg as a passing one. The blocker
described a comparison that would now never happen; the honest replacement describes the
permanent absence of an independent leg, which is still unmeasured coverage.

## Rationale

The catalog is the smallest structure that makes enumeration a public answer rather than a
side effect of importing the runner. It is one module, one frozen mapping and five query
functions, with no state to mutate and nothing to configure — a frontend asks a question
and gets either an engine or a clear `ValueError`.

Making role declarative rather than inferred is what keeps the decision honest. A frontend
that filters on `role` cannot accidentally expose a validation reference, cannot go stale
when a name changes, and needs no edit when a new engine ships. The test that pins the
intended classification deliberately asserts a literal table instead of deriving
expectations from the catalog, because a test generated from the thing it validates will
redefine correctness along with a regression.

The serialization change is a deferred import rather than a module-level one so the
dependency direction stays `engines` → `io`. `gammaforge.io` remains the shared layer with
no knowledge of which engines exist, which is the invariant that lets a future engine be
added without touching the physics core.

## Consequences

Adding an in-repository engine that satisfies the public contract is now one catalog entry
plus its own package; no GUI, runner, or serialization edit follows, and the new engine
appears as an engine choice automatically if it is given the calculation role.

Delta is no longer a GUI engine choice, and its engine parameter group no longer appears
in the workspace. It is still resolvable through the catalog by name and still returns
schemas for request restoration, so `validation/run.py` and any script that used it are
unaffected.

The production tier continues to report a missing-coverage blocker and to exit nonzero.
That blocker is now permanent rather than pending: independent emission-physics validation
requires a new independent leg, which is separate work and not part of this change.

`runner.py` names no engine. A reader looking for which engines exist now looks in
`engines/catalog.py`, and code that reached into `runner.engines` to discover them should
call the catalog instead — the attribute remains for the GUI and for injection, not as an
enumeration API.

Saved GUI defaults may name engines that no longer exist. That was already tolerated
before this change and is now the general mechanism, covered by a test rather than left
implicit.
