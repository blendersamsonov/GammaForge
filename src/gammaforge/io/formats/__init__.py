"""Serialization formats (GRAND_PLAN.md §8).

Three boundaries, all of which convert to CGS-Gaussian on the way in and out (P1):

* `yaml_spec` — beam/laser/sampling parameter files, with **explicit units at the file
  boundary**. Driven entirely by `gammaforge.io.fields`, so a renamed or added parameter
  needs no change here.
* `sdds` — elegant ``.ele`` 6D distributions, inherently SI/GeV.
* `hdf5` — results: density slices plus their axis values, with a YAML sidecar for the
  parameters that produced them.
"""
