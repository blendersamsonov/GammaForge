"""Experiment 1 (Phase II): smooth Stage-0 observables, before any histogramming.

The question this answers, from the revised handoff §2 and §15:

> Does global QMC or Gaussian cubature converge substantially faster than IID's ``N**(-1/2)``
> on smooth functionals of the Stage-0 map -- and if so, does that advantage survive Stage 1?

Phase I could not ask it. Every number there was a histogrammed output, whose residual is
per-cell Poisson noise that reallocating particles does not touch. Here nothing is deposited:
each method runs Stage 0 once and the smooth observables are read off the samples directly
(:mod:`smooth`). A histogram cell indicator is a *discontinuous* test function, so comparing
this module's convergence against :mod:`exp7_deposition`'s is what localizes the loss.

Reference construction (handoff §5). A noisy 4M IID run is not the only authority available
here, and using one would put an ``N**(-1/2)`` floor under the very thing being compared against
it. The reference is instead the **best converged deterministic** value, and its uncertainty is
the disagreement between *independent* constructions -- Gauss-Hermite and global QMC. A
quantity is marked resolved only when the two agree; the gap between them is recorded so the
reader can see the floor rather than having to trust it.

Deliberately not done: any change to Stage-1 semantics, any production default, and the
sparse-grid probe (see :func:`cubature.smolyak_gauss_hermite` for why it is unavailable).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

import cubature as C
import harness as H
import smooth as S
from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.adaptive_sampling import PilotConfig, build_adaptive_plan
from gammaforge.io.bunch import prefilter_bunch
from gammaforge.io.interaction import IID, SamplingSpec, build_interaction
from gammaforge.validation import scenarios as scen

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

#: 1D orders for the tensor rule. Order 9 and 10 are 5e5 and 1e6 trajectories; opt-in.
DEFAULT_ORDERS = (3, 4, 5, 6, 7, 8)
#: Trajectory counts for IID, global QMC and the regional stratifier.
DEFAULT_N = (1_000, 4_000, 16_000, 64_000, 262_144, 1_000_000)
#: Error targets for the "trajectories needed" column (handoff §6).
TARGETS = (1e-2, 1e-3, 1e-4)


def run_stage0(beam, laser, target, method, budget, *, seed=0, chunk=None):
    """Stage 0 once for one source rule. Returns ``(samples, weights)``.

    ``N_e`` is ``beam.n_electrons()`` for every method, not a function of the sample: charge is
    a beam property (§3.2) and every output is exactly linear in it, so deriving it from the
    rule instead would rescale one arm's luminosity relative to the others and quietly change
    the comparison.

    The luminosity prefilter is applied to *every* method, including the cubature rules that
    bypass ``build_interaction``. It is a pure optimization with a tested invariance property
    (§7: discarded particles contribute ``L = 0`` and weights are never renormalized), so
    applying it or not does not move an answer -- but not applying it would make the cubature
    arms *pay* Stage-0 trajectories that IID never spends, which would bias the
    trajectories-needed column against exactly the rules under test.
    """
    n_electrons = float(beam.n_electrons())
    if method == "iid":
        interaction = build_interaction(
            beam, laser, target,
            SamplingSpec(n_particles=budget, seed=seed, prefilter=1e-3, strategy=IID),
        )
        bunch, n_electrons = interaction.bunch, interaction.N_e
    elif method in ("global-qmc", "tensor-gh"):
        if method == "global-qmc":
            deviates, weights = C.global_qmc(budget, seed=seed)
        else:
            deviates, weights = C.tensor_gauss_hermite(int(budget))
        bunch = C.build_source_bunch(beam, deviates, weights)
        prefiltered = prefilter_bunch(bunch, laser, 1e-3)
    elif method == "strat-s1":
        config = PilotConfig(proposal_scale=1.0, luminosity_fraction=0.0,
                             initial_regions=256, max_regions=256,
                             pilot_points_per_region=8, pilot_quad_nodes=128,
                             pilot_quad_panels=2)
        plan = build_adaptive_plan(beam, laser, seed=seed, config=config)
        interaction = build_interaction(
            beam, laser, target,
            SamplingSpec(n_particles=budget, seed=seed, prefilter=1e-3, strategy="adaptive"),
            plan=plan,
        )
        bunch, n_electrons = interaction.bunch, interaction.N_e
    else:
        raise ValueError(f"unknown method {method!r}")

    if method in ("global-qmc", "tensor-gh"):
        bunch = prefiltered
    samples = integrate_trajectories(
        bunch, laser, n_electrons, n_steps=H.STAGE0_STEPS, threshold=1e-3, chunk=chunk,
    )
    return samples, np.asarray(bunch.weight, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*",
                        default=["iid", "global-qmc", "tensor-gh", "strat-s1"])
    parser.add_argument("--orders", type=int, nargs="*", default=list(DEFAULT_ORDERS))
    parser.add_argument("--n", type=int, nargs="*", default=list(DEFAULT_N))
    parser.add_argument("--chunk-mb", type=float, default=2000.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    import preflight
    preflight.check()
    H.CHUNK = H.chunk_for_mb(args.chunk_mb)

    if args.quick:
        args.orders, args.n = [3, 4], [2_000, 8_000]
    if "smolyak" in args.methods:
        print("note: dropping the sparse-grid probe -- unavailable this phase; see "
              "cubature.smolyak_gauss_hermite for the measured reason", file=sys.stderr)
        args.methods = [m for m in args.methods if m != "smolyak"]

    wanted = {s.name for s in H.scenarios()} if not args.only else set(args.only)
    out: dict = {"config": {"methods": args.methods, "orders": args.orders, "n": args.n},
                 "scenarios": {}}

    for scenario in H.scenarios():
        if scenario.name not in wanted:
            continue
        beam, laser, target = scen.BASELINE.beam, scen.BASELINE.laser, scen.BASELINE.target
        beam, laser = H.build(scenario, laser, beam)
        print(f"\n=== {scenario.name} ===", flush=True)
        t_start = time.perf_counter()

        # Standardization from a large IID run: probes become dimensionless and one probe set
        # stays comparable across geometries whose angular widths differ by orders of magnitude.
        big = max(args.n)
        ref_samples, ref_weights = run_stage0(beam, laser, target, "iid", big,
                                              seed=0, chunk=H.CHUNK)
        standardization = S.Standardization.from_samples(ref_samples, ref_weights)
        print(f"  standardization from {big} IID particles "
              f"({time.perf_counter() - t_start:.0f}s)", flush=True)

        curves: dict[str, list[tuple[int, dict]]] = {}
        for method in args.methods:
            points = []
            for budget in (args.orders if method == "tensor-gh" else args.n):
                t0 = time.perf_counter()
                samples, weights = run_stage0(beam, laser, target, method, budget,
                                               seed=0, chunk=H.CHUNK)
                value = S.evaluate(samples, weights, standardization)
                points.append((C.node_count(method, budget), value, weights))
                neff = S.effective_sample_fraction(weights)
                print(f"  {method:>11} budget={budget:<8} -> {points[-1][0]:>8} trajectories"
                      f"  M0={value['M0']:.6e}  N_eff/N={neff:.2e}"
                      f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
            curves[method] = points

        # Reference: best converged deterministic value, with the independent-construction gap
        # as its uncertainty.
        #
        # The arm that *supplies* the reference cannot also be scored against it: doing so
        # reports its largest budget as error 0.0 and quietly anchors every other arm's error
        # to whatever that arm happens to be. So the reference point is recorded as a row with
        # ``is_reference`` and its error left undefined, and the tensor rule's own convergence
        # is reported separately as its successive-order differences (handoff §5).
        gh, qmc = curves.get("tensor-gh") or [], curves.get("global-qmc") or []
        if gh:
            best, ref_key = gh[-1][1], ("tensor-gh", gh[-1][0])
        elif qmc:
            best, ref_key = qmc[-1][1], ("global-qmc", qmc[-1][0])
        else:
            best, ref_key = S.evaluate(ref_samples, ref_weights, standardization), None
        gap = max(S.scalar_errors(qmc[-1][1], gh[-1][1]).values()) if (gh and qmc) else None
        if gap is None:
            print("  reference: only one deterministic construction available, "
                  "so no independent-construction gap", flush=True)
        else:
            print(f"  reference: tensor-gh order {args.orders[-1]}, "
                  f"independent-construction gap {gap:.3e}", flush=True)

        # Successive-budget differences: a rule's own convergence, with no reference involved,
        # so it stays meaningful even for the arm that supplies the reference.
        self_convergence = {}
        for method, points in curves.items():
            if len(points) > 1:
                self_convergence[method] = [
                    S.aggregate(S.scalar_errors(points[i + 1][1], points[i][1]))
                    for i in range(len(points) - 1)
                ]

        summary = {}
        for method, points in curves.items():
            n_values = [p[0] for p in points]
            errors = [float('nan') if ref_key == (method, n)
                      else S.aggregate(S.scalar_errors(v, best))
                      for n, v, _ in points]
            scored = [(n, e) for n, e in zip(n_values, errors) if np.isfinite(e)]
            fit = S.fit_exponent([n for n, _ in scored], [e for _, e in scored])
            needed = {f"{t:.0e}": S.trajectories_for_error([n for n, _ in scored],
                                                           [e for _, e in scored], t)
                      for t in TARGETS}
            print(f"\n  {method}: alpha={fit['alpha']:.3f} from {fit['n_points']} points "
                  f"(resolved={fit['resolved']})")
            for label, value in needed.items():
                shown = f"{value:.4g}" if np.isfinite(value) else "unresolved"
                print(f"    trajectories for {label}: {shown}")
            summary[method] = {
                "trajectories": n_values,
                "error": errors,
                "fit": fit,
                "trajectories_for": needed,
                "n_eff_fraction": [S.effective_sample_fraction(w) for _, _, w in points],
                "self_convergence": self_convergence.get(method),
                "groups": [{} if ref_key == (method, n)
                           else S.group_errors(S.scalar_errors(v, best))
                           for n, v, _ in points],
            }
        out["scenarios"][scenario.name] = {
            "reference": {"method": ref_key[0] if ref_key else "iid",
                          "trajectories": ref_key[1] if ref_key else None,
                          "construction_gap": gap,
                          "note": "this arm's largest budget is the reference; its own error "
                                  "is undefined and its convergence is in self_convergence"},
            "methods": summary,
        }

    RESULTS.mkdir(exist_ok=True)
    target_path = RESULTS / f"exp6_smooth_{'quick' if args.quick else 'full'}.json"
    with open(target_path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {target_path}")



if __name__ == "__main__":
    main()