"""Experiment 2 (Phase II): does the cubature advantage survive Stage 1?

The smooth-observable result (:mod:`exp6_smooth`) is only half the answer. Stage 1 deposits
those samples into a 5D table, and a histogram cell indicator is a *discontinuous* test
function. A rule can integrate a smooth push-forward beautifully and still land in the right
cells with the wrong counts, because the error that matters downstream is per-cell Poisson
noise -- which is exactly the residual Phase I measured and could not move.

So this runs the same source rules through the **existing** Stage-1/Stage-2 pipeline unchanged,
under two deposition schemes:

- ``nearest`` -- discontinuous; one cell per particle.
- ``cic`` -- continuous piecewise-linear in the deposited coordinates.

Same Stage-0 evaluations for both schemes wherever possible, so the comparison isolates
deposition. If cubature/QMC wins on smooth observables, loses under ``nearest``, and partly
returns under ``cic``, that is direct evidence that **Stage 1, not the Stage-0 source
integration, is the remaining bottleneck** (handoff §7) -- which would close this phase as
Case A and hand Stage-1 representation to a separate task.

Stage-1 semantics are untouched: no new kernel, no changed binning, no production default.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import cubature as C
import exp6_smooth as E6
import harness as H
import smooth as S
from gammaforge.engines.xigma.stages import retarget_ahat, spectrum_from_table
from gammaforge.validation import scenarios as scen

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SCHEMES = ("nearest", "cic")
DEFAULT_ORDERS = (4, 5, 6, 7, 8)
DEFAULT_N = (16_000, 64_000, 262_144)
TARGETS = (1e-2, 1e-3)


def table_metrics(samples, weights, edges, grid, scheme) -> dict:
    """Deposit once and return the histogrammed outputs, in the harness's own convention.

    Fixed edges from the scenario's reference, so no arm is charged for choosing its own bins.
    """
    shape = H.deposit_fixed(samples, edges, scheme=scheme)
    table = retarget_ahat(shape, float(samples.intensity_peak))
    spectrum = spectrum_from_table(table, 0.0, 0.0, grid)
    return {
        "yield": float(np.sum(samples.luminosity)),
        "H": shape.H.copy(),
        "spectrum": spectrum,
        "centroid": float(np.sum(grid * spectrum) / max(np.sum(spectrum), 1e-300)),
    }


def histogram_errors(value: dict, reference: dict) -> dict[str, float]:
    """Relative errors for the histogrammed outputs, on the same scale convention as exp6."""
    out = {
        "spectrum": float(np.sum(np.abs(value["spectrum"] - reference["spectrum"]))
                         / max(np.sum(np.abs(reference["spectrum"])), 1e-300)),
        "centroid": abs(value["centroid"] - reference["centroid"])
        / max(abs(reference["centroid"]), 1e-300),
    }
    marginals = H.marginal_l1(value["H"], reference["H"])
    out["marg_worst"] = float(marginals["marg_worst"])
    out["marg_mean"] = float(marginals["marg_mean"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*",
                        default=["iid", "global-qmc", "tensor-gh"])
    parser.add_argument("--orders", type=int, nargs="*", default=list(DEFAULT_ORDERS))
    parser.add_argument("--n", type=int, nargs="*", default=list(DEFAULT_N))
    parser.add_argument("--bins", default="reduced", choices=["reduced", "production"])
    parser.add_argument("--ref-n", type=int, default=4_000_000)
    parser.add_argument("--chunk-mb", type=float, default=2000.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    import preflight
    preflight.check()
    H.CHUNK = H.chunk_for_mb(args.chunk_mb)
    if args.quick:
        args.orders, args.n, args.ref_n = [4], [4_000, 16_000], 200_000

    bins = H.PRODUCTION_BINS if args.bins == "production" else H.REDUCED_BINS
    wanted = {s.name for s in H.scenarios()} if not args.only else set(args.only)
    out: dict = {"config": {**vars(args), "methods": args.methods}, "scenarios": {}}

    for scenario in H.scenarios():
        if scenario.name not in wanted:
            continue
        beam0, laser0, target = (scen.BASELINE.beam, scen.BASELINE.laser,
                                 scen.BASELINE.target)
        beam, laser = H.build(scenario, laser0, beam0)
        grid = H.spectral_grid(beam, laser, target)
        print(f"\n=== {scenario.name} / {args.bins} bins ===", flush=True)
        t0 = time.perf_counter()

        # Fixed edges + a large IID reference, from the same construction for every arm.
        probe_samples, _ = E6.run_stage0(beam, laser, target, "iid", args.ref_n,
                                         seed=0, chunk=H.CHUNK)
        edges = H.reference_edges(probe_samples, bins)
        del probe_samples
        reference = {}
        for scheme in SCHEMES:
            samples, _ = E6.run_stage0(beam, laser, target, "iid", args.ref_n,
                                       seed=0, chunk=H.CHUNK)
            reference[scheme] = table_metrics(samples, None, edges, grid, scheme)
            del samples
        print(f"  reference: {args.ref_n} IID, {time.perf_counter() - t0:.0f}s", flush=True)

        curves: dict[tuple[str, str], list[tuple[int, dict[str, float]]]] = {}
        for method in args.methods:
            for budget in (args.orders if method == "tensor-gh" else args.n):
                t1 = time.perf_counter()
                samples, weights = E6.run_stage0(beam, laser, target, method, budget,
                                                 seed=0, chunk=H.CHUNK)
                trajectories = C.node_count(method, budget)
                for scheme in SCHEMES:
                    metrics = table_metrics(samples, weights, edges, grid, scheme)
                    curves.setdefault((method, scheme), []).append(
                        (trajectories, histogram_errors(metrics, reference[scheme])))
                print(f"  {method:>11} budget={budget:<8} -> {trajectories:>8} trajectories"
                      f"  ({time.perf_counter() - t1:.0f}s)", flush=True)
                del samples, weights

        summary = {}
        for (method, scheme), points in curves.items():
            n_values = [p[0] for p in points]
            # Spectrum error drives the verdict; the marginals are reported alongside because
            # "worse on the spectrum" and "worse on the 5D table" can disagree.
            spectrum_err = [p[1]["spectrum"] for p in points]
            fit = S.fit_exponent(n_values, spectrum_err)
            summary[f"{method}|{scheme}"] = {
                "trajectories": n_values,
                "errors": {k: [p[1][k] for p in points]
                           for k in ("spectrum", "marg_worst", "marg_mean", "centroid")},
                "fit": fit,
                "trajectories_for": {f"{t:.0e}": S.trajectories_for_error(n_values,
                                                                         spectrum_err, t)
                                     for t in TARGETS},
            }
            print(f"  {method:>11} / {scheme:>7}: alpha={fit['alpha']:+.3f} "
                  f"({fit['n_points']} pts)  best spectrum={min(spectrum_err):.3e}",
                  flush=True)
        out["scenarios"][scenario.name] = {"reference_n": args.ref_n, "results": summary}

    RESULTS.mkdir(exist_ok=True)
    target_path = RESULTS / f"exp7_deposition_{'quick' if args.quick else args.bins}.json"
    with open(target_path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {target_path}")


if __name__ == "__main__":
    main()