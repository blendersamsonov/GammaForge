# RES066 — Alpha dependency floors preserve the numerical reference constants

Status: implemented
Class: process

## Problem

A successful installation in the existing development environment did not demonstrate
that the declared lower-bound dependencies could import or reproduce the reference
calculations. Clean Python 3.12 checks exposed incompatible older package combinations.

## Decision

The alpha requires NumPy 2.0, Pint 0.25, h5py 3.11, Matplotlib 3.8.4 and PyYAML 6.0.1
or newer. Exact lower bounds are exercised with `requirements/minimum.txt`; the recorded
developer set remains separate in `requirements/developer.lock`. Core CI and installed-wheel
CI also run the restricted alpha validation selector from RES065.

## Alternatives considered

Keeping the previous permissive floors would advertise combinations that fail import,
build or numerical validation. NumPy 2 APIs are already used by the implementation.

Loosening scalar-golden tolerances to accommodate older Pint constants would conceal an
environment-dependent physics-input change. No golden data, constants or tolerances are
changed to accommodate the dependency check.

Pinning the entire runtime environment in the library metadata would unnecessarily
constrain scripts that install GammaForge alongside other scientific packages. Exact
constraints belong to the tested minimum and developer environments instead.

## Rationale

Pint 0.24 fails import with the resolved parser dependencies; 0.24.4 imports but its
constants move the peak-field scalar beyond the existing tight golden tolerance.
Pint 0.25 supplies CODATA 2022 and passes those checks. The h5py and Matplotlib floors
support NumPy 2, and PyYAML 6.0.1 provides a working Python 3.12 installation.
See [Pint's release notes](https://pint.readthedocs.io/en/0.25/changes.html).

## Consequences

The minimum environment is a tested combination, not a guarantee for every future
dependency release. Reference comparisons and recorded calculation settings remain
necessary when changing environments. GUI and symbolic tests skip only when their
optional dependency is absent, not when an installed dependency is broken.

The GitHub CI workflow was removed on 2026-09-29. The dependency-floor check is now
run locally using `requirements/minimum.txt` and `make check`.
