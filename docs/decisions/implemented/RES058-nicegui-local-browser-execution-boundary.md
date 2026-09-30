# RES058 — NiceGUI local browser UI with a separate calculation runner

Status: implemented
Type: architecture

## Problem

The author selected a local browser application instead of the unbuilt desktop GUI,
specified the panel order, and requested simple split panes plus a future path to
calculations on another LAN machine. The frontend must preserve the existing schema,
CGS, engine boundary, and explicit Calculate semantics.

## Decision

The optional NiceGUI frontend starts explicitly through `gammaforge.gui.__main__` on
loopback. Two page-local views share one `Workspace`: Inputs and Results, optionally
shown in two independently selectable panes using the framework's standard splitter.
The current layout lives in `src/gammaforge/gui/` and is summarized in the root
`README.md`. The retired UI specification and plan remain in Git history.

`CalculationRequest` captures physical inputs, target, sampling, and typed engine
parameters. `LocalRunner` consumes the request without importing NiceGUI. It samples
once and calls selected engines sequentially; the GUI invokes it on a worker thread.
Analytical previews use an empty bunch and run separately from the calculation action.
Revision checks discard obsolete previews and mark results outdated after intervening
edits. Charge-only rescaling applies only to a clean completed snapshot.

The public runner enumerates the concrete available calculation engines with a small
dictionary, closing RES018's deferred enumeration question without replacing its
engine protocol. Analytical remains an always-visible estimate and optional result overlay.
The GUI imports only that runner and the engine interface from the engines package.

`LocalRunner` reuses the sampled bunch where its physical inputs are unchanged.
It does not retain xigma stages across calls: RES030's per-instance cache behavior
is unchanged, and no new cheap recompute tiers are claimed.

Plot projections and geometry live in the shared drawing/plotting modules. Browser
plots use Plotly; PNG/PDF exports use Matplotlib with the same projection and density
conversion helpers. Existing HDF5 handles result downloads. Input snapshots contain a
standard beam/laser/sampling YAML file and a separate YAML calculation-settings file.

## Alternatives considered

The planned Tkinter interface was not implemented. Building it would contradict the
author's browser requirement and create avoidable native GUI dependencies.

A custom JavaScript frontend and docking manager would provide more layout freedom,
but the required two panes are covered by NiceGUI's built-in splitter and tabs.

Running an engine in a UI event callback blocks interaction. A process pool would
isolate CPU execution further but requires explicit transport for large arrays and
mapping-proxy-backed objects; that work is premature for the local version. A worker
thread preserves the existing single-worker plan and the in-memory bunch cache.

A remote-service framework or speculative executor hierarchy would add unexercised
machinery. The concrete request/runner boundary provides the seam for a future LAN
implementation without implementing transport, discovery, or authentication now.

## Rationale

The existing typed physics interface already separates inputs and results from the
frontend. Keeping execution behind that interface makes browser state and future
transport choices independent of engine formulas. Small schema editors share unit and
convention behavior across panels, while placement remains an explicit visual choice.

## Consequences

The browser server defaults to local use. Refresh creates a fresh workspace initialized
from any per-panel defaults saved under RES089; switching tabs or split mode preserves
state. Run history is not persisted. No LAN execution or durable job service ships.
The current engine boundary reports per-engine state, not percentage progress or
interruptible cancellation. Missing engine outputs and calculation failures remain
visible. The interface is tested with controlled engine failures, unit conversions,
projection integrals, and an opt-in real Chromium calculation workflow.

## Amendments

> **2026-09-29 — Documentation relocation.** The historical UI specification and plan were retired by issue #8. The root `README.md` now describes current browser behavior; this decision records the architecture.

> **2026-09-30 — Engine enumeration moved to a public catalog (RES095).** The small
> dictionary described above, constructed in `LocalRunner` from imports of the concrete
> engine classes, is replaced by `gammaforge.engines.catalog`. `LocalRunner` still supplies
> the GUI's engines and analytical remains the always-visible estimate and optional result
> overlay, so this decision's runner/frontend boundary stands; only the enumeration
> mechanism it recorded has been superseded. The GUI and calculation serialization now
> resolve engines through the catalog rather than through the runner.
