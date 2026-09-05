# GammaForge browser UI specification

Status: implementation specification, author-directed 2026-09-05.
Authority: GRAND_PLAN.md v0.26, especially §§3–6. Physics stays in the existing shared
types and engines. This document specifies presentation and local execution only.

## Launch and execution boundary

- Launch a local NiceGUI server with `python -m gammaforge.gui`; bind to 127.0.0.1
  and open the normal browser. Allow a port override and a no-browser flag for tests.
- NiceGUI is an optional GUI dependency; importing the physics library must not start
  a server or require NiceGUI.
- Each browser page owns its input state, results, and local runner. Split panes share
  that page state. A refresh starts a fresh workspace in the initial version; say so
  in user documentation. Changing tabs or split mode preserves the workspace.
- Calculation input is an explicit snapshot of beam, LaserField, sampling, target,
  and selected engine parameters. The runner receives no widgets or browser objects.
- The local runner samples once and calls checked engines sequentially. Sampling and
  engine calls run off the browser server's event loop. Guard duplicate Calculate clicks.
- Future LAN execution is a documented replacement of this boundary. Do not implement
  remote transport, address controls, authentication, discovery, or a job broker now.
  Requests are not assumed pickleable; existing Parameters/Results use mapping proxies.
  A future transport must encode input specifications and transfer results explicitly.

## Navigation and layout

Default: two tabs, **Inputs**, then **Results**. An optional **Split view** toggle uses
NiceGUI's standard resizable splitter. Each pane has its own Inputs/Results selection;
the initial split shows Inputs on the left and Results on the right. No docking tree,
draggable tabs, arbitrary extra panes, or external window manager.

Inputs, in exactly this vertical order:

```text
┌──────────────────┬──────────────────┬────────────────────────┐
│ A. Electrons     │ B. Laser         │ C. Geometry            │
│ Beam fields      │ Pulse properties │ Four angle inputs      │
│ Sampling + seed  │                  │ [2D / 3D]              │
│                  │                  │ Interaction sketch     │
├──────────────────┴──────────────────┴────────────────────────┤
│ E. Target and requested outputs                              │
├─────────────────────────────────────────────────────────────┤
│ D. Analytical estimates                                     │
├─────────────────────────────────────────────────────────────┤
│ Engine subtabs: engine settings + Use for calculation        │
│ [Calculate] [Release inputs]     Per-engine status            │
└─────────────────────────────────────────────────────────────┘
```

A/B/C use equal column widths and stretch to the same height. Long forms may scroll
inside their column. At narrow viewport or pane widths, stack them in A/B/C order
instead of clipping fields; equal height applies to the three-column arrangement.
Every row below spans the available pane width.

## Inputs and field semantics

- A contains BEAM_FIELDS plus SAMPLING_FIELDS, including editable random seed and
  prefilter threshold. B contains LASER_FIELDS except the four angles assigned to C.
- C owns theta_xz, theta_yz, psi_focus, and psi_pol controls. These are views of the
  same laser parameter set, not duplicate geometry parameters. Ellipticity stays in B.
- Every field uses FieldSpec validation, canonical CGS storage, and available display
  units. Width/duration fields expose their convention selector. Unit/convention
  selection converts the shown value while preserving its physical meaning.
- Scientific notation is accepted. Incomplete or invalid edits show an inline error
  and disable Calculate; never silently calculate using the old valid value.
- The sketch shows lab axes, bunch envelope, laser direction, focusing axes/ellipse,
  polarization orientation, and astigmatic foci. Offer static 2D and 3D views.
  Drawings are schematic, not sampled radiation or new emission physics. Use the
  existing laser rotation convention; time-evolution animation is deferred.
- Validation warnings from the beam/laser/results are visible, reflecting the
  limits reported by the current physics implementation.

## Target and outputs row

- Collimation half-angles in x and y, with units.
- Total yield is always requested. Checkboxes select other output kinds, with integer
  resolution controls per axis. No manual energy/time/angular range fields.
- Enable a requested output only when a selected calculation engine supports it;
  unavailable outputs are disabled and explained. Missing engine curves are shown as
  unavailable, not fabricated. The analytical preview is a separate always-on panel.
- Default request: total yield and spectrum; other outputs are opt-in. Do not change
  engine numerical defaults merely to make the GUI appear faster.

## Analytical estimates row

- Always visible below Target/Outputs. Show total yield, total spectral width, and the
  existing collimation, emittance, energy-spread, and nonlinearity width components.
- Re-evaluate after valid input edits using the existing analytical engine; no large
  bunch sampling for a preview. Coalesce edits and discard obsolete preview responses.
- Preview computation also stays off the event loop: exact quadrature and flying-focus
  cases are not guaranteed instantaneous. A pending preview must not look current.
- Analytical is not a checkboxable calculation engine. At Calculate, preserve its
  result for that input snapshot as an optional overlay in Results.

## Engines and calculation lifecycle

- Engine subtabs come from concrete available engines and their public schemas. Today
  xigma is available; kascade's empty package does not become a fake selectable engine.
- Each engine tab has its settings and **Use for calculation** checkbox. One Calculate
  button below the subtabs runs exactly the checked engines; analytical stays separate.
- Report queued/running/completed/failed per engine. No invented percent progress.
  Retain useful results and the actual error if an engine fails; permit a later retry.
- Inputs are a draft; running jobs use their captured snapshot. A result arriving after
  another edit must remain marked outdated. Show status on both Inputs and Results.
- After completion, lock fields according to selected engines' recompute-cost data;
  absent declarations mean FULL_RERUN. Release inputs unlocks them explicitly.
  Apply the strictest selected-engine cost for shared fields.
- Input changes mark existing results outdated until Calculate. Charge alone may
  rescale clean completed results exactly; combined edits remain outdated. Display-only
  changes (units, plot visibility, tabs, split mode) do not invalidate physical results.
- Current xigma only declares charge cheap. Do not promise stage reuse across Calculate
  clicks or import Collision into the GUI to manufacture it. Reuse of the sampled bunch
  for target/output/charge-only changes belongs in the local runner.

## Results tab

- Subtabs follow the requested outputs. Preserve access to the last completed results
  with a clear outdated indicator when current draft requests differ.
- Overlay compatible 1D engine curves, including analytical where produced, with
  independent show/hide controls. Use consistent engine colors.
- For 2D outputs, select among available engines. For collimated spectra, expose
  zero-angle slices, energy-angle projections integrated over the other angle, and
  the spectrum integrated over both collimation angles. Use axis measures, not plain
  array sums, and preserve density units after display conversion.
- Browser plots support zoom/pan. Provide PNG/PDF plot export and HDF5 results download;
  attach a YAML input specification to saved calculations. No new save format.
- If particle output becomes available, show counts/statistics and download using
  existing serialization; do not invent MC results for currently absent engines.
- Empty state explains that Calculate populates results. Failed calculations show a
  readable error; no empty successful graph standing in for a failed computation.

## Implementation ownership and shared interfaces

Workers implement bounded features, with the coordinating agent integrating the page:

1. Execution: io/calculation.py with CalculationRequest; engines/runner.py with
   LocalRunner, available calculation engines, preview and sequential execution.
2. Inputs: gui/inputs.py and gui/state.py with schema field editors, A/B/C/E panels,
   engine settings, and input-state snapshot construction.
3. Visualization: io/plotting.py and io/drawing.py for headless figures/projections;
   gui/outputs.py for NiceGUI results and geometry components.
4. Coordinator: gui/app.py, launcher, estimates/status integration, split panes,
   dependency setup, architectural checks, browser verification, and documentation.

The concrete interfaces are agreed in worker task messages. No worker changes physics
formulas, tuned numerical constants, unrelated derivations, or the historical repo.

## Acceptance checks

- Local launch and a real browser smoke test; no server on library import.
- Inputs/Results navigation and independent split-pane selections preserve edits and
  results; no duplicate calculations from two views of the same state.
- Equal-height A/B/C columns at desktop width and correctly ordered full-width E/D/
  engines rows; usable narrow layout; C's four angles appear exactly once per form.
- Unit/convention conversion preserves the physical input. Invalid values block runs.
- Small real xigma calculation returns results; the UI responds during work. Engine
  failure and edit-during-run state have meaningful tests using controlled fake engines.
- Analytical preview uses existing formulas and avoids macroparticle sampling.
- Projection/integration and unit conversion preserve the appropriate photon integral;
  HDF5 exports can be read by existing I/O.
- Import-boundary test rejects GUI imports of engine implementations, stages, kernels,
  or Collision. Existing physics and documentation tests stay green.

## Deferred

LAN execution; cross-run xigma stage reuse; kascade implementation; physics derivations;
scans; arbitrary docking; persistence across browser refresh; multi-user deployment.
