"""Experiment: the direct-table comparison (IID vs global QMC vs regional `s=1`).

Answers the question the Phase-II handoff narrowed to after the Gauss-Hermite result: **is
global QMC a better source rule for Stage-1/Stage-2 output, and does the regional `s=1`
stratifier add anything on top of it?**

Scope, following the handoff's revision:

- Tensor Gauss-Hermite stays available as context (``--include gh``) but is **not** part of the
  comparison. Its table-filling failure is already settled, and its weights leave ~10³
  effective points at any useful budget, so extending that study spends compute on a closed
  question.
- No new smoother kernel, no Stage-1 semantic change, no production default. This is evidence
  for a decision, not the decision.

Three things this module is careful about, each of which corrupted an earlier pass:

**Nothing retains a table.** Every metric is reduced to a scalar while the deposited array is
live, and the array is dropped immediately. That is what makes the production-resolution spot
check possible at all: at production bins one table is 680 MB, so the Phase-I pattern of
holding `arms x replicates` tables is ~57 GB per worker, and holding *one* is 680 MB.

**The reference floor is measured, not assumed.** The reference is the mean of several
independent 4M IID runs and its floor is their disagreement. Fitted convergence rates are then
gated on that floor: a point whose error is within a few times the reference's own error
measures the *reference*, whatever its trend appears to do.

**Methods get their own replicates.** Each arm runs at several seeds (IID) or shifts (QMC), so
deterministic-method performance is separable from reference noise. A single-seed arm cannot
distinguish "this rule is better" from "this seed was lucky".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np

import exp6_smooth as E6
import harness as H
import smooth as S
from gammaforge.engines.xigma.stages import retarget_ahat, spectrum_from_table
from gammaforge.io.units import Quantity as Q
from gammaforge.validation import scenarios as scen

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SCHEMES = ("nearest", "cic")
DEFAULT_N = (4_000, 16_000, 64_000, 262_144)
TARGETS = (1e-2, 1e-3, 1e-4)

#: Observation directions as multiples of the scenario's own collimation half-angles. The
#: spectrum depends on the observation direction through Stage 2, so one direction cannot
#: stand in for the rest; these span on-axis, tighter, and two off-axis tilts.
DIRECTIONS = {
    "nominal": (1.0, 1.0),
    "narrow": (0.5, 0.5),
    "off_x": (2.0, 0.5),
    "off_y": (0.5, 2.0),
}


def make_target(target, sx: float, sy: float):
    """A copy of ``target`` with scaled collimation half-angles."""
    return dataclasses.replace(
        target,
        theta_x_col=target.theta_x_col * sx,
        theta_y_col=target.theta_y_col * sy,
    )


def deposition_scalars(samples, edges, grid, scheme: str) -> dict:
    """Deposit once and reduce to scalars. **The 5D table is dropped here, deliberately.**

    Every downstream number is a scalar or a small fixed vector, so nothing that could grow
    with replicate count or bin count escapes this function.
    """
    shape = H.deposit_fixed(samples, edges, scheme=scheme)
    table = retarget_ahat(shape, float(samples.intensity_peak))
    spectrum = spectrum_from_table(table, 0.0, 0.0, grid)
    total = float(np.sum(spectrum))
    centroid = float(np.sum(grid * spectrum) / max(total, 1e-300))
    # Free the retargeted copy before touching anything else: at production bins one table is
    # 680 MB and `deposit_fixed` already peaks near 8x that in temporaries, so holding two live
    # tables is the difference between a comfortable run and an OOM.
    del table
    marginals = H.marginals(shape.H)           # 1D and 2D, each unit-normalized
    del shape
    return {"spectrum": spectrum, "marginals": marginals, "centroid": centroid,
            "yield": float(np.sum(samples.luminosity))}


def compare(value: dict, reference: dict) -> dict:
    """Scalar errors of one arm against the reference, for one direction."""
    out = {
        "spectrum": float(np.sum(np.abs(value["spectrum"] - reference["spectrum"]))
                          / max(float(np.sum(np.abs(reference["spectrum"]))), 1e-300)),
        "centroid": abs(value["centroid"] - reference["centroid"])
        / max(abs(reference["centroid"]), 1e-300),
        "yield": abs(value["yield"] - reference["yield"])
        / max(abs(reference["yield"]), 1e-300),
    }
    per = {k: float(np.sum(np.abs(value["marginals"][k] - reference["marginals"][k])))
           for k in reference["marginals"]}
    one_d = [v for k, v in per.items() if k.startswith("m1_")]
    two_d = [v for k, v in per.items() if k.startswith("m2_")]
    out["marg_1d_worst"] = max(one_d) if one_d else float("nan")
    out["marg_2d_worst"] = max(two_d) if two_d else float("nan")
    out["marg_worst"] = max(per.values()) if per else float("nan")
    return out


def _reference_for_scheme(beam, laser, target, edges, grid, scheme, *, ref_n, ref_seeds,
                          cross_n, cross_shifts, chunk):
    """Reference mean for ONE deposition scheme, with its measured floor.

    Per scheme, not shared: the reference is itself a deposited table, so scoring CIC arms
    against a nearest-deposited reference compares two different operators and makes CIC look
    worse than it is. That mistake showed up as CIC being 2x *worse* here when an earlier run
    had it consistently better.
    """
    runs = []
    for seed in ref_seeds:
        samples, _ = E6.run_stage0(beam, laser, target, "iid", ref_n, seed=seed, chunk=chunk)
        runs.append(deposition_scalars(samples, edges, grid, scheme))
        del samples
    mean = {
        "spectrum": np.mean([r["spectrum"] for r in runs], axis=0),
        "marginals": {k: np.mean([r["marginals"][k] for r in runs], axis=0)
                      for k in runs[0]["marginals"]},
        "centroid": float(np.mean([r["centroid"] for r in runs])),
        "yield": float(np.mean([r["yield"] for r in runs])),
    }
    floor = 0.0
    if len(runs) > 1:
        gaps = []
        stack = np.stack([r["spectrum"] for r in runs])
        denom = float(np.max(np.abs(np.mean(stack, axis=0))))
        gaps.append(float(np.max(np.abs(stack - np.mean(stack, axis=0))))
                    / max(denom, 1e-300))
        for key in mean["marginals"]:
            stack = np.stack([r["marginals"][key] for r in runs])
            gaps.append(float(np.max(np.abs(stack - np.mean(stack, axis=0)))))
        gaps.append(float(np.ptp([r["centroid"] for r in runs])
                          / max(abs(mean["centroid"]), 1e-300)))
        floor = max(gaps)

    cross_gap = None
    if cross_n:
        cross = []
        for shift in range(cross_shifts):
            samples, _ = E6.run_stage0(beam, laser, target, "global-qmc", cross_n,
                                       seed=1000 + shift, chunk=chunk)
            cross.append(compare(deposition_scalars(samples, edges, grid, scheme), mean))
            del samples
        cross_gap = float(max(c["spectrum"] for c in cross))
    return mean, floor, cross_gap


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*",
                        default=["iid", "global-qmc", "strat-s1"])
    parser.add_argument("--include-gh", action="store_true",
                        help="add tensor Gauss-Hermite as context (not part of the comparison)")
    parser.add_argument("--n", type=int, nargs="*", default=list(DEFAULT_N))
    parser.add_argument("--bins", default="reduced", choices=["reduced", "production"])
    parser.add_argument("--ref-n", type=int, default=4_000_000)
    parser.add_argument("--ref-seeds", type=int, default=3)
    parser.add_argument("--cross-n", type=int, default=1_000_000,
                        help="independent global-QMC cross-check trajectories; 0 disables")
    parser.add_argument("--cross-shifts", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=4,
                        help="seeds (IID) / shifts (QMC) per arm, to separate method "
                             "performance from reference noise")
    parser.add_argument("--directions", nargs="*", default=list(DIRECTIONS))
    parser.add_argument("--chunk-mb", type=float, default=2000.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    import preflight
    preflight.check()
    H.CHUNK = H.chunk_for_mb(args.chunk_mb)
    if args.quick:
        args.n, args.ref_n = [2_000, 8_000], 20_000
        args.ref_seeds, args.cross_n, args.cross_shifts, args.replicates = 2, 4_000, 2, 2
        args.directions = ["nominal"]

    methods = list(args.methods) + (["tensor-gh"] if args.include_gh else [])
    bins = H.PRODUCTION_BINS if args.bins == "production" else H.REDUCED_BINS
    wanted = {s.name for s in H.scenarios()} if not args.only else set(args.only)
    ref_seeds = tuple(3 + 8 * i for i in range(args.ref_seeds))
    out: dict = {"config": {k: v for k, v in vars(args).items()}, "scenarios": {}}

    for scenario in H.scenarios():
        if scenario.name not in wanted:
            continue
        beam0, laser0, target0 = (scen.BASELINE.beam, scen.BASELINE.laser,
                                  scen.BASELINE.target)
        beam, laser = H.build(scenario, laser0, beam0)
        print(f"\n=== {scenario.name} / {args.bins} bins ===", flush=True)

        # Fixed edges from one reference-scale run, shared by every arm and direction so no arm
        # is charged for choosing its own bins.
        probe, _ = E6.run_stage0(beam, laser, target0, "iid", args.ref_n, seed=0,
                                 chunk=H.CHUNK)
        edges = H.reference_edges(probe, bins)
        del probe

        scenario_out: dict = {}
        for direction in args.directions:
            sx, sy = DIRECTIONS[direction]
            target = make_target(target0, sx, sy)
            t0 = time.perf_counter()
            grid = H.spectral_grid(beam, laser, target)
            references, floors, cross_gaps = {}, {}, {}
            for scheme in SCHEMES:
                references[scheme], floors[scheme], cross_gaps[scheme] = _reference_for_scheme(
                    beam, laser, target, edges, grid, scheme, ref_n=args.ref_n,
                    ref_seeds=ref_seeds, cross_n=args.cross_n,
                    cross_shifts=args.cross_shifts, chunk=H.CHUNK)
                print(f"  [{direction}/{scheme}] reference {args.ref_n} x "
                      f"{args.ref_seeds} IID: floor {floors[scheme]:.2e}, QMC "
                      f"cross-check "
                      f"{'n/a' if cross_gaps[scheme] is None else f'{cross_gaps[scheme]:.2e}'}",
                      flush=True)
            print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

            curves: dict[tuple[str, str], list] = {}
            for method in methods:
                for budget in args.n:
                    per_replicate: dict[tuple[str, str], list] = {}
                    for rep in range(args.replicates):
                        samples, weights = E6.run_stage0(
                            beam, laser, target, method, budget, seed=rep, chunk=H.CHUNK)
                        neff = S.effective_sample_fraction(weights)
                        for scheme in SCHEMES:
                            value = deposition_scalars(samples, edges, grid, scheme)
                            errors = compare(value, references[scheme])
                            errors["n_eff_fraction"] = neff
                            per_replicate.setdefault((method, scheme), []).append(errors)
                        del samples, weights
                    for key, reps in per_replicate.items():
                        mean_err = {k: float(np.mean([r[k] for r in reps]))
                                    for k in reps[0]}
                        mean_err["spectrum_spread"] = float(
                            np.std([r["spectrum"] for r in reps], ddof=1)
                        if len(reps) > 1 else 0.0)
                        curves.setdefault(key, []).append(
                            (E6.C.node_count(method, budget), mean_err))
                    best = curves[(method, SCHEMES[0])][-1][1]
                    print(f"    {method:>11} N={budget:<8} "
                          f"N_eff/N={best['n_eff_fraction']:.2e} "
                          f"spectrum={best['spectrum']:.3e}", flush=True)

            summary = {}
            for (method, scheme), points in curves.items():
                n_values = [p[0] for p in points]
                entry = {"trajectories": n_values, "n_eff_fraction":
                         [p[1]["n_eff_fraction"] for p in points]}
                floor = floors[scheme]
                for metric in ("spectrum", "marg_1d_worst", "marg_2d_worst", "marg_worst",
                               "centroid", "yield"):
                    errors = [p[1][metric] for p in points]
                    entry.setdefault("errors", {})[metric] = errors
                    fit = S.fit_exponent(n_values, errors, floor=floor)
                    entry.setdefault("fits", {})[metric] = fit
                    entry.setdefault("trajectories_for", {})[metric] = {
                        f"{t:.0e}": S.trajectories_for_error(n_values, errors, t, floor=floor)
                        for t in TARGETS}
                summary[f"{method}|{scheme}"] = entry
            scenario_out[direction] = {"reference_floor": floors,
                                       "cross_check_gap": cross_gaps,
                                       "results": summary}
        out["scenarios"][scenario.name] = scenario_out

    RESULTS.mkdir(exist_ok=True)
    tag = "quick" if args.quick else args.bins
    target_path = RESULTS / f"exp7_direct_{tag}.json"
    with open(target_path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {target_path}")


if __name__ == "__main__":
    main()