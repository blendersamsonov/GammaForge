"""Thin, schema-driven Tkinter GUI (GRAND_PLAN.md §6).

No physics, no engine branching (P12). Import boundary is mechanically enforced in CI:
this package may not import engine internals — `engines/*/stages.py`, kernel modules,
or any engine-specific stateful facade — regardless of what it legitimately imports
from `gammaforge.io` (schema, `Engine.run()`/`Results`, the shared drawing module,
YAML/HDF5 I/O).
"""
