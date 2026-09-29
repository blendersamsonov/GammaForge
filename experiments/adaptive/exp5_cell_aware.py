"""Experiment 5: a cell-aware Neyman allocation, tested offline.

The current score is ``A_m = P_m sqrt(M2_m)`` with ``M2_m = E_m[l^2]``. It charges a region
for how much its luminosity *varies*, which is the right criterion for the total yield and
the wrong one for a binned density: a region whose luminosity is large but whose particles
all land in one Stage-1 cell contributes almost nothing to the table's error.

For a stratified estimator the summed L2 variance of the Stage-1 histogram contains
``P_m^2 / n_m * sum_j Var_m[Y_j]``, so the allocation minimizing it is

    n_m  proportional to  P_m * sqrt( sum_j Var_m[Y_j] )

and for nearest deposition, where one particle contributes ``Y_j = l * 1_{c(x)=j}``,

    sum_j Var_m[Y_j]  =  E_m[l^2]  -  sum_j ( E_m[l * 1_{c=j}] )^2 .

The subtracted term is exactly the missing piece: it is large when one cell holds nearly all
the region's luminosity, and small when the same luminosity is spread over many cells.

Two versions are computed and compared, which separates two different questions:

* **pilot** — the criterion evaluated from the cheap pilot's *predicted* Stage-1 coordinates
  and luminosity, i.e. what a production implementation could actually afford.
* **oracle** — the same criterion evaluated from a real Stage-0 run on a large sample drawn
  from each region. This is an upper bound on what any pilot-based criterion could achieve,
  and the gap between the two is the cost of the pilot's error.

Neither touches production code; both drive the same plan-shaped allocation through the
existing `AdaptiveSamplingPlan` machinery.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

import harness as H
from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.adaptive_sampling import (
    AdaptiveSamplingPlan,
    _conditional_deviates,
    _legendre_rule,
    build_adaptive_plan,
    _PPF_CEIL,
    _PPF_FLOOR,
    _standard_normal_cdf,
    norm_ppf,
)
from gammaforge.io.bunch import _bunch_from_standard_deviates
from gammaforge.io.interaction import IID, SamplingSpec, build_interaction
from gammaforge.io.laser import laser_propagation_direction
from gammaforge.io.units import C_CGS
from gammaforge.validation import scenarios as scen

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def pilot_luminosity_and_shape(bunch, laser, *, n_quad=128, panels=2, threshold=1e-8):
    """Per-particle ``(luminosity, a0_shape, chirp_mean)`` from a pilot quadrature.

    Reproduces Stage 0's definitions at pilot cost: the same `F`, the same
    ``C(t) = 1 + (d_t + v.grad dPhi) / (omega0 F)``, and the same ratio ``r = I / I_peak`` in
    the ``a0_shape`` and ``chirp_mean`` moments (handoff §14). Written out here rather than
    imported so the production module stays untouched by a research experiment.
    """
    from gammaforge.io.bunch import overlap_time_window

    t0, t1 = overlap_time_window(bunch, laser, threshold)
    span = np.maximum(0.0, t1 - t0)
    start = np.where(span > 0.0, t0, 0.0)
    nodes, weights = _legendre_rule(n_quad, panels)
    times = start[:, None] + nodes * span[:, None]
    dts = weights * span[:, None]

    n0 = laser_propagation_direction(laser)
    omega0 = float(laser.omega0())
    norm = np.sqrt(1.0 + bunch.thx**2 + bunch.thy**2)
    vx, vy, vz = C_CGS * bunch.thx / norm, C_CGS * bunch.thy / norm, C_CGS / norm
    encounter = 1.0 - (n0[0] * bunch.thx + n0[1] * bunch.thy + n0[2]) / norm
    positions = (
        bunch.x[:, None] + vx[:, None] * times,
        bunch.y[:, None] + vy[:, None] * times,
        bunch.z[:, None] + vz[:, None] * times,
    )
    intensity = np.asarray(laser.intensity_profile(*positions, times), dtype=float)
    gradient = laser.carrier_phase_four_gradient(*positions, times)
    d_t, d_x, d_y, d_z = np.broadcast_arrays(
        *(np.asarray(c, dtype=float) for c in gradient)
    )
    carrier = 1.0 + (d_t + vx[:, None] * d_x + vy[:, None] * d_y + vz[:, None] * d_z) / (
        omega0 * encounter[:, None]
    )
    contributing = (intensity > 0.0) & (dts > 0.0)
    carrier = np.where(contributing, carrier, 1.0)

    peak = float(laser.intensity_peak()) if hasattr(laser, "intensity_peak") else float(
        np.max(intensity)
    )
    ratio = intensity / peak
    weighted = carrier * ratio
    moment_1 = np.sum(weighted, axis=1)
    safe = np.maximum(moment_1, 1e-300)
    usable = moment_1 > 0.0
    a0_shape = np.where(usable, np.sum(carrier * ratio**2, axis=1) / safe, 0.0)
    chirp_mean = np.where(usable, np.sum(carrier**2 * ratio, axis=1) / safe, 0.0)
    luminosity = encounter * np.sum(dts * carrier * intensity, axis=1)
    return luminosity, a0_shape, chirp_mean


def cell_of(values, edges, names=("gamma", "theta_x", "theta_y", "a0_shape", "chirp_mean")):
    """Flat 5D cell index, or -1 when a particle falls outside the fixed edges."""
    indices = []
    inside = np.ones(values["gamma"].shape, dtype=bool)
    for axis, name in enumerate(names):
        edge = edges[axis]
        v = np.asarray(values[name], dtype=float)
        idx = np.searchsorted(edge, v, side="right") - 1
        inside &= (idx >= 0) & (idx < len(edge) - 1)
        indices.append(np.clip(idx, 0, len(edge) - 2))
    return np.ravel_multi_index(tuple(indices), tuple(len(e) - 1 for e in edges)), inside


def cell_aware_scores(plan, beam, laser, edges, *, oracle_samples=64, seed=0,
                      n_quad=128, panels=2):
    """``sum_j Var_m[Y_j]`` per region, from the pilot and from an oracle sample.

    ``Y_j = l * 1_{c=j}`` per particle, so with a region's pilot sample
    ``sum_j Var_m[Y_j] = E[l^2] - sum_j (E[l 1_{c=j}])^2``.
    """
    n_cells = int(np.prod([len(e) - 1 for e in edges]))
    rng = np.random.default_rng(seed)
    pilot_score, oracle_score = {}, {}
    pilot_cells_seen, oracle_cells_seen = {}, {}

    for region in plan.regions:
        # --- pilot: the cheap predicted (l, coords) ---
        deviates = _conditional_deviates(
            region.lo, region.hi, region.proposal_scale,
            plan.pilot_config.pilot_points_per_region, plan.seed, region.id, "pilot",
        )
        probe = _bunch_from_standard_deviates(
            beam, deviates, np.full(deviates.shape[0], 1.0 / deviates.shape[0])
        )
        lum, shape, chirp = pilot_luminosity_and_shape(
            probe, laser, n_quad=n_quad, panels=panels,
            threshold=plan.pilot_config.pilot_window_threshold,
        )
        values = {
            "gamma": probe.gamma, "theta_x": probe.thx, "theta_y": probe.thy,
            "a0_shape": shape, "chirp_mean": chirp,
        }
        flat, inside = cell_of(values, edges)
        pilot_score[region.id] = _var_sum(lum, flat, inside, n_cells)
        pilot_cells_seen[region.id] = int(np.unique(flat[inside]).size) if inside.any() else 0

        # --- oracle: a real Stage-0 run on a larger draw from the same region ---
        n_oracle = oracle_samples
        big = _conditional_deviates(
            region.lo, region.hi, region.proposal_scale, n_oracle, plan.seed + 977, region.id,
            "production",
        )
        probe_big = _bunch_from_standard_deviates(
            beam, big, np.full(n_oracle, 1.0 / n_oracle)
        )
        samples = integrate_trajectories(
            probe_big, laser, 1.0, n_steps=200, threshold=1e-3
        )
        flat_o, inside_o = cell_of(
            {
                "gamma": np.asarray(samples.gamma), "theta_x": np.asarray(samples.theta_x),
                "theta_y": np.asarray(samples.theta_y),
                "a0_shape": np.asarray(samples.a0_shape),
                "chirp_mean": np.asarray(samples.chirp_mean),
            },
            edges,
        )
        oracle_score[region.id] = _var_sum(
            np.asarray(samples.luminosity), flat_o, inside_o, n_cells
        )
        oracle_cells_seen[region.id] = int(np.unique(flat_o[inside_o]).size) if inside_o.any() else 0
    return pilot_score, oracle_score, pilot_cells_seen, oracle_cells_seen


def _var_sum(luminosity, flat, inside, n_cells):
    """``E[l^2] - sum_j (E[l 1_{c=j}])^2`` over the inside particles."""
    l = np.asarray(luminosity, dtype=float)[inside]
    if l.size == 0:
        return 0.0
    total = float(np.mean(l**2))
    per_cell = np.bincount(flat[inside], weights=l, minlength=n_cells) / l.size
    return max(total - float(np.sum(per_cell**2)), 0.0)


def allocation_from_scores(plan, scores, *, blend=0.0):
    """``n_m proportional to P_m sqrt(score_m)``, with an optional B_m floor."""
    importance = np.array(
        [plan.regions[i].target_mass * math.sqrt(max(scores[plan.regions[i].id], 0.0))
         for i in range(len(plan.regions))]
    )
    reference = np.array([r.reference_mass for r in plan.regions])
    if blend > 0.0:
        reference = reference / reference.sum()
        importance = importance / max(importance.sum(), 1e-300)
        share = blend * reference + (1.0 - blend) * importance
    else:
        share = importance
    share = share / share.sum()
    # Feed through the plan's own apportionment so the counts are identical in kind.
    return AdaptiveSamplingPlan(
        seed=plan.seed,
        regions=tuple(
            type(r)(**{**r.__dict__, "allocation_probability": float(share[i])})
            for i, r in enumerate(plan.regions)
        ),
        pilot_config=plan.pilot_config,
    ), share


RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ref-n", type=int, default=400_000)
    parser.add_argument("--ref-seeds", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    args = parser.parse_args()

    beam0, laser0, target = scen.BASELINE.beam, scen.BASELINE.laser, scen.BASELINE.target
    bins = H.REDUCED_BINS
    ref_n = args.ref_n
    seeds = tuple(3 + 8 * i for i in range(args.ref_seeds))[:2]
    budgets = tuple(args.budgets) if args.budgets else (20_000, 40_000)
    if args.quick:
        budgets = (5_000, 10_000)
        seeds = seeds[:1]

    # Built through the harness so the unit conversions match the scenario bank exactly.
    wanted = {"baseline", "focus_3um", "tight_focus", "wide_bunch"}
    if args.only:
        wanted &= set(args.only)
    cases = [
        (s, *H.build(s, laser0, beam0)[:2])
        for s in (H.Scenario("baseline", {}, {}),
                  H.Scenario("focus_3um", {}, dict(sigma_x=3.0, sigma_y=3.0)),
                  H.Scenario("tight_focus", {}, dict(sigma_x=1.5, sigma_y=1.5)),
                  H.Scenario("wide_bunch", dict(sigma_x=400.0, sigma_y=400.0),
                             dict(sigma_x=4.0, sigma_y=4.0, duration=1.0)))
        if s.name in wanted
    ]
    out = []
    for scenario, beam, laser in cases:
        grid = H.spectral_grid(beam, laser, target)
        probe = build_interaction(
            beam, laser, target, SamplingSpec(n_particles=ref_n, seed=3)
        )
        ps = integrate_trajectories(probe.bunch, laser, probe.N_e, n_steps=200, threshold=1e-3)
        edges = H.reference_edges(ps, bins)
        ref = H.make_reference(None, beam, laser, target, ref_n, seeds, edges, grid=grid)
        print(f"\n=== {scenario.name} ===")
        print("  floor: yield {:.1e} | marg worst {:.1e} mean {:.1e} | spectrum {:.1e}".format(
            ref["yield_spread"], ref["marg_worst_spread"], ref["marg_mean_spread"],
            ref["spectrum_spread"]), flush=True)

        from gammaforge.io.adaptive_sampling import PilotConfig
        plan_configs = (
            ("s1.41-lam0.75", dict(proposal_scale=math.sqrt(2.0), luminosity_fraction=0.75,
                                   initial_regions=256, max_regions=256)),
            ("s1", dict(proposal_scale=1.0, luminosity_fraction=0.0,
                        initial_regions=256, max_regions=256)),
        )
        # A plan is seed-scoped, so one per (config, seed).
        plans = {}
        for name, config in plan_configs:
            cfg = PilotConfig(pilot_points_per_region=8, pilot_quad_nodes=128,
                              pilot_quad_panels=2, **config)
            plans[name] = {s: build_adaptive_plan(beam, laser, seed=s, config=cfg)
                           for s in seeds}

        for plan_name, per_seed in plans.items():
            plan = per_seed[seeds[0]]
            start = time.perf_counter()
            pilot_s, oracle_s, pilot_cells, oracle_cells = cell_aware_scores(
                plan, beam, laser, edges
            )
            pilot_plan, pilot_share = allocation_from_scores(plan, pilot_s)
            oracle_plan, oracle_share = allocation_from_scores(plan, oracle_s)
            # Recomputed per seed: the low-discrepancy shifts and therefore the pilot
            # deviates are seed-scoped, so reusing the seed-3 scores would compare a seed-3
            # allocation against a seed-11 estimator.
            def _pilot_for(pl):
                s, _, _, _ = cell_aware_scores(pl, beam, laser, edges)
                return allocation_from_scores(pl, s)[0]

            def _oracle_for(pl):
                _, s, _, _ = cell_aware_scores(pl, beam, laser, edges)
                return allocation_from_scores(pl, s)[0]

            pilot_plan = _pilot_for
            oracle_plan = _oracle_for
            current_share = np.array([r.allocation_probability for r in plan.regions])
            target_mass = np.array([r.target_mass for r in plan.regions])
            importance = np.array([r.pilot_second_moment for r in plan.regions])
            print(f"  [{plan_name}] allocation comparison ({time.perf_counter()-start:.0f}s)")
            print("    share spread: current {:.2f} | cell-aware(pilot) {:.2f} | "
                  "cell-aware(oracle) {:.2f}".format(
                      current_share.max() / current_share.min(),
                      pilot_share.max() / pilot_share.min(),
                      oracle_share.max() / oracle_share.min()))
            print("    cells/region: pilot mean {:.1f} | oracle mean {:.1f} (of {} cells)".format(
                np.mean(list(pilot_cells.values())), np.mean(list(oracle_cells.values())),
                int(np.prod([len(e) - 1 for e in edges]))))
            # How much of the score's dynamic range is the concentration term?
            naive = np.array([target_mass[i] * math.sqrt(importance[i])
                              for i in range(len(plan.regions))])
            naive = naive / naive.sum()
            print("    corr(cell-aware_pilot, P_m sqrt(M2)) = {:.3f}".format(
                float(np.corrcoef(pilot_share, naive)[0, 1])), flush=True)

            header = (f"    {'variant':>22} {'N':>7} {'yield':>10} {'marg_worst':>11} "
                      f"{'marg_mean':>10} {'spectrum':>10} {'N_eff':>7}")
            print(header)
            for label, builder in (
                ("current (P sqrt(M2))", lambda pl: pl),
                ("cell-aware pilot", pilot_plan),
                ("cell-aware oracle", oracle_plan),
            ):
                for n in budgets:
                    runs = []
                    for seed in seeds:
                        sp = SamplingSpec(n_particles=n, seed=seed, prefilter=1e-3,
                                          strategy="adaptive")
                        variant_plan = builder(per_seed[seed])
                        it = build_interaction(beam, laser, target, sp, plan=variant_plan)
                        s = integrate_trajectories(it.bunch, laser, it.N_e,
                                                   n_steps=200, threshold=1e-3)
                        st = H.deposit_fixed(s, edges)
                        tb = __import__(
                            "gammaforge.engines.xigma.stages", fromlist=["retarget_ahat"]
                        ).retarget_ahat(st, float(s.intensity_peak))
                        spec = __import__(
                            "gammaforge.engines.xigma.stages", fromlist=["spectrum_from_table"]
                        ).spectrum_from_table(tb, 0.0, 0.0, grid)
                        runs.append({
                            "yield": float(np.sum(s.luminosity)), "H": st.H.copy(),
                            "spectrum": spec,
                            "centroid": float(np.sum(grid * spec) / np.sum(spec)),
                            "n_eff": float((it.bunch.weight.sum() ** 2)
                                           / np.sum(it.bunch.weight ** 2)),
                        })
                    merged = {
                        "yield": float(np.mean([r["yield"] for r in runs])),
                        "H": np.mean([r["H"] for r in runs], axis=0),
                        "spectrum": np.mean([r["spectrum"] for r in runs], axis=0),
                        "centroid": float(np.mean([r["centroid"] for r in runs])),
                    }
                    err = H.errors(merged, ref)
                    row = {"scenario": scenario.name, "plan": plan_name, "variant": label,
                           "n": n, "yield": err["yield"], "marg_worst": err["marg_worst"],
                           "marg_mean": err["marg_mean"], "spectrum": err["spectrum"],
                           "centroid": err["centroid"],
                           "n_eff": float(np.mean([r["n_eff"] for r in runs])),
                           "floor_spectrum": ref["spectrum_spread"],
                           "floor_marg_worst": ref["marg_worst_spread"]}
                    out.append(row)
                    print(f"    {label:>22} {n:>7} {err['yield']:>10.2e} "
                          f"{err['marg_worst']:>11.2e} {err['marg_mean']:>10.2e} "
                          f"{err['spectrum']:>10.2e} {row['n_eff']:>7.0f}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "exp5_cell_aware.json", "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote results/exp5_cell_aware.json ({len(out)} rows)")


if __name__ == "__main__":
    main()
