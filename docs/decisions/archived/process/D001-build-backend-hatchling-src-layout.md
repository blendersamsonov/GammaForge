# D001 — Build backend: hatchling, src layout

Status: implemented
Class: process
Archived: 2026-08-10

## Problem

Phase 0 needs a build backend and package layout choice for `pyproject.toml`.

## Decision

`pyproject.toml` uses hatchling as the build backend, with the package under
`src/gammaforge/`.

## Alternatives considered

**`setuptools`.** Works equally well for a src-layout package, but needs more explicit
configuration (a packages-find table) for the same result; hatchling's defaults handle
src-layout with no extra config.

## Rationale

One less thing to get wrong in Phase 0; no other project constraint favors setuptools
specifically.
