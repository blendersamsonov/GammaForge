# 05 — Lossless supported results and reproducible run export

Read [coordination rules](README.md). Covers A05. Inspect now; land after 02's
slice measure and 06's calculation snapshot/ownership interfaces settle.

## Entry points and evidence

`src/gammaforge/io/formats/{hdf5,yaml_spec,sdds}.py`, `io/calculation.py`,
`io/results.py`, `gui/outputs.py::_download_hdf5`, `gui/app.py` snapshot download,
and `tests/test_formats.py`. Read GRAND_PLAN §8/§10.8 and RES058/RES059.

`save_results` writes slices and photons but not electrons or `model_specific`.
`load_results` returns string keys unless a caller supplies `kind_from_name`.
An earlier round trip of results containing final electrons, warning/seed
metadata and a yield slice came back with `electrons=None`, empty metadata and
`"0d_yield"` as the key. The GUI HDF5 download supplies no parameter groups; its
separate snapshot ZIP has custom calculation YAML without a complete matching
loader. Existing slice round-trip tests miss these omissions.

## Work

1. Specify a small versioned on-disk contract for supported Results and a separate
   run record containing the submitted request, engine settings, seed, warnings
   and actual results. Update the plan first if this changes its HDF5/YAML layout.
   Keep `Results.cfg` absent: provenance belongs to the run record, not a mutable
   engine back-reference.
2. Preserve slices including 02's integration metadata, axis order and canonical
   units; photon arrays; and the currently supported final-electron Bunch fields
   and relevant metadata. Do not decide the future final-electron class merely
   to serialize today's data.
3. Define safe supported metadata types and unknown/unsupported-value behavior.
   No pickle, dynamic class imports, or silent metadata truncation. Decide/document
   typed output-key defaults while retaining an explicit custom-key escape hatch
   if current public usage needs it.
4. Make the GUI download serialize the completed calculation snapshot, not newer
   editable form values. Use the same loadable request representation headlessly;
   avoid two almost-equivalent YAML schemas with separate meanings.
5. Define legacy-file handling and version errors. Legacy missing metadata cannot
   be reconstructed truthfully: return/document unknown provenance, not defaults
   disguised as the settings actually used.

## Acceptance

- Round-trip a synthetic complete result and a small actual kascade result,
  including nonzero/negative event-time metadata, warnings, output keys and units.
- Round-trip nondefault target, requested resolutions, sampling seed and engine
  parameters through the run-record API. Where reproducible, rerun the loaded
  request and compare outputs with appropriate deterministic/MC expectations.
- Preserve 02's one-bin/nonuniform measure and integrate/project identities after
  loading. Test old unversioned files and unsupported future versions explicitly.
- A form edit during/after calculation cannot change the exported request that
  produced the displayed result. Failure paths leave no misleading complete file.

Elegant-compatible final-electron export remains a separate format-research item
unless explicitly assigned with an agreed convention; don't expand this fix into
that work or a plugin serialization framework. Run format, runner/controller and
relevant GUI export tests; browser smoke if the download flow changes.

Starting check: `.venv/bin/pytest -q tests/test_formats.py tests/test_calculation_runner.py tests/test_gui_controller.py`.
