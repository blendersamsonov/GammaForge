# Crossing-angle yield example handoff

**Status:** complete and measured on 2026-09-06 UTC.  This worktree was created from
`main` at `54e9132`; it is intentionally isolated from the runtime fixes currently
uncommitted in the primary checkout.

## Intended deliverables

- `examples/crossing_angle_yield.py`: standalone, public-API-only comparison of
  `AnalyticalEngine` and `XigmaEngine`.
- `examples/output/crossing_angle_yield/`: CSV/JSON provenance plus paper-style PNG/PDF.

## Run command

```sh
cd /tmp/gammaforge-alpha-example
PYTHONPATH=$PWD/src python examples/crossing_angle_yield.py
```

The script will use a bounded CPU configuration and overwrite only its own output
directory.  Its figure compares unnormalised Stage-0 total yields; this does not validate
the angle-resolved Stage-2 emission kernel.

## Measured result

The public-engine sweep covers `theta_xz = 0..40 mrad` in 5 mrad increments, with zero
defined as head-on.  It uses three independently sampled 15,000-particle bunches at
600 trajectory steps (`20260906`, `20260907`, `20260908`) and does **not** normalise
xigma to the analytical curve.

- Mean absolute xigma--analytical difference: **0.0342%**; maximum: **0.0919%**
  (at head-on).  At 40 mrad it is **+0.00785%**.
- The yield falls from `1.19565e11` photons at head-on to `2.60732e10` at 40 mrad;
  that is the intended geometric crossing-angle suppression, not an engine discrepancy.
- Analytical `n_quad_overlap/n_quad_u = (4001, 51), (8001, 101), (12001, 151)` differs
  from the finest result by at most `2.78e-8` at the 0/20/40 mrad convergence anchors.
- Xigma's 200, 400, and 800 time steps agree at the same anchors to at worst `9.12e-8`
  relative (40 mrad); 600 steps is therefore conservative here.  The seed SEM is retained
  as the plotted uncertainty rather than being hidden by a single lucky sample.
- The explicit 15,000-to-30,000 particle check at 0/20/40 mrad changes one fixed-seed
  sample by `-0.256%`, `-0.140%`, and `-0.0718%`, respectively.  This is a one-seed
  sampling-sensitivity check, not an alternative normalisation or a physics correction;
  `xigma_seed_samples.csv` and the SEM in the figure remain the comparison uncertainty.

The CSV files are the evidence for these statements.  There is no physics discrepancy to
escalate: the unnormalised three-seed xigma mean agrees with the analytical result within
its seeded uncertainty at the low-`a0` (`a0_peak = 0.181`) representative point.

## Original XIGMA workflow observations

Read-only inspection of `/home/alexander/Work/Code/XIGMA/git-repo/example-config.toml`
and its `calculate-spec-ang.py` consumer established the source basis: 10 nC,
`gamma0=2000`, relative energy spread 0.005, 10 um RMS transverse beam sizes, 10 ps
bunch duration, plus a 20 J, 1030 nm, 10 um RMS laser with 30 ps duration.  The old
validation figure style is a single-column Agg/Matplotlib PDF with a 3.4-inch width; the
new figure follows that compact, vector-friendly convention while writing both PDF and
300 dpi PNG.  No ComptonSuite source or results were used.
