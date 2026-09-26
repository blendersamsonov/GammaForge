# RES089 — GUI defaults are sectional and output ranges are per-axis Auto or manual

Status: implemented
Class: feature

## Problem

The browser always started from package defaults, so a user repeatedly re-entered the
same beam, laser, geometry, target, and engine settings. Output grids also exposed only
resolution: their derived bounds were hidden, and the conservative energy Auto range
from zero to above the Compton edge was often much wider than the useful part of a
collimated spectrum.

## Decision

Add independent Save as default actions for electrons, sampling, laser, geometry,
target, requested outputs, and each named engine. `GuiDefaultsStore` writes one
versioned YAML document under the user configuration directory, updates one section at
a time, and atomically replaces the file. Values use the existing parameter YAML
representation and are validated against the current schemas before any section is
applied to a new browser workspace. Run history and arbitrary workspace state are not
persisted.

Generalize `OutputRequest.manual_ranges` to any axis of a slice output. The GUI presents
an Auto checkbox per axis, always displays the current derived bound in display units,
and enables manual Min/Max fields only when Auto is off. A manual bound replaces only
its named axis in `auto_ranges`; every omitted axis retains the existing conservative
rule. Exact temporal Auto bounds remain explicitly identified as sampled at Calculate.

## Alternatives considered

- Store one whole-form default snapshot: rejected because changing laser defaults would
  unexpectedly overwrite beam, target, output, and engine choices.
- Use *browser local storage*: rejected because the loopback application already owns a
  readable YAML boundary, and browser-profile storage would make defaults harder to
  inspect, back up, and share between browsers on the same workstation.
- Replace the conservative energy Auto range with an analytical width estimate: rejected
  because that would change a physics-grid policy and could clip engine-specific tails.
  The explicit manual energy bound solves the display/computation need without making a
  narrower bound look universally safe.
- Use one Auto toggle for an entire multidimensional output: rejected because narrowing
  the energy grid of a collimated spectrum should not detach its angular axes from the
  target window.

## Rationale

Sectional persistence matches the panels the user edits and keeps every saved value on
the same `Parameters` validation path as ordinary YAML input. Per-axis policy preserves
the established autorange as a visible safe reference while allowing a focused energy
grid without forcing unrelated spatial or angular choices. Keeping manual bounds in the
public request also makes calculations, downloads, and reloads describe the same grid.

## Consequences

The default file is user-local rather than repository state. A malformed file is not
partially applied and its error is shown in the GUI. Display units and conventions are
not saved; physical values are. Fully manual temporal bounds no longer require a sampled
bunch, while Auto temporal bounds still use the actual sampled overlap. Manual grids can
exclude physical weight, so Auto remains the initial setting and its bounds remain
visible even while a manual range is active.
