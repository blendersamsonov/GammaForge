"""Optional NiceGUI browser interface.

No physics, no engine branching. This package may not import engine internals —
`engines/*/stages.py`, kernel modules,
or any engine-specific stateful facade — regardless of what it legitimately imports
from `gammaforge.io` (schema, `Engine.run()`/`Results`, the shared drawing module,
YAML/HDF5 I/O). Launch explicitly with ``python -m gammaforge.gui``.
"""
