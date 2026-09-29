"""Shared machinery for the adaptive-sampling experiments (RES092 follow-up).

Not production code and not committed: this is the harness for the ablation, the lambda and
proposal-scale sweeps, the wider scenario bank, the production-resolution check, and the
cell-aware allocation study. It lives outside `src/` so nothing here can be mistaken for
shipped behaviour.

Two design points that everything else depends on:

**Fixed reference-table edges.** The engine derives its Stage-1 edges from the data it is
given, so two runs never share a grid and a cell-by-cell comparison is ill-posed. Every run
here is therefore deposited onto **one** edge set, derived once from a high-statistics
reference, by constructing a `ShapeTable` directly. Stage 1 and Stage 2 then operate on a
common grid, and the numbers mean what they look like they mean.

**Equal-cost comparison, not equal-N-of-a-cheaper-run.** All variants cost the same Stage-0
work per particle, so the comparison is genuinely at equal budget, and the only thing that
differs is *which* particles are drawn and how they are weighted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from gammaforge.engines.xigma.stages import (
    ShapeTable,
    Table,
    integrate_trajectories,
    retarget_ahat,
    spectrum_from_table,
)
from gammaforge.io.adaptive_sampling import PilotConfig, build_adaptive_plan
from gammaforge.io.interaction import ADAPTIVE, IID, SamplingSpec, build_interaction
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, auto_ranges
from gammaforge.engines.xigma.stages import BYTES_PER_PARTICLE_STEP
from gammaforge.io.units import C_CGS

# Stage-0 quadrature for a run. Held fixed across every variant so the cost per particle is
# identical and the comparison is not confounded.
STAGE0_STEPS = 200

#: Production Stage-1 bins, and the reduced set the earlier benchmark used.
PRODUCTION_BINS = (48, 48, 48, 96, 8)
REDUCED_BINS = (24, 16, 16, 32, 8)

#: Arms must not reuse the reference's seeds.
#:
#: They used to: `SEEDS = REF_SEEDS[:2]`, so an IID arm at seed 3 drew the *same* particle
#: stream as the reference at seed 3 -- the arm's 40k particles are literally the first 40k of
#: the reference's 4M. That correlates each arm with the thing it is being scored against and
#: makes every arm look better than it is, with the effect largest for the arm that shares the
#: most structure with the reference (IID). Arm seeds are now offset clear of reference seeds.
ARM_SEED_OFFSET = 900

#: Particles per Stage-0 chunk, or None to let the engine auto-size. Set via
#: :func:`chunk_for_mb` from the experiment scripts' ``--chunk-mb``.
#:
#: Left unset, the engine's own auto-chunker budgets ``free_ram * 0.5`` for a *single*
#: process -- it has no notion of how many workers are running. Stage 0's inner loop holds
#: ~25 live float64 temporaries per (particle, step), so that is 40 KB per particle at
#: n_steps=200, and a 4M-particle reference is a ~1.5M-particle chunk and ~60 GB. With a
#: fan-out of workers each sizing off the same machine-wide figure, the "safety" fraction
#: bounds nothing in aggregate. Chunking cannot change the answer (particles are
#: independent), so a fixed per-worker cap is free of correctness risk and costs only speed.
CHUNK = None


def chunk_for_mb(mb: float, n_steps: int = STAGE0_STEPS) -> int:
    """Particles per Stage-0 chunk for a target peak of ``mb`` MiB *per worker*.

    Expressed in megabytes because that is the thing a user can budget: the particle count
    that fills a given memory depends on ``n_steps`` and on the engine's calibrated
    bytes-per-particle-step, both of which are implementation detail.
    """
    return max(1, int(mb * 1024 ** 2 / (n_steps * BYTES_PER_PARTICLE_STEP)))


def arm_seeds(n: int) -> tuple[int, ...]:
    """``n`` arm seeds, disjoint from the reference's."""
    return tuple(ARM_SEED_OFFSET + 8 * i for i in range(n))


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """One sampling scheme, as a name plus either IID or a `PilotConfig`."""

    name: str
    strategy: str
    config: PilotConfig | None = None
    note: str = ""

    def interaction(self, beam, laser, target, n_particles, seed, prefilter=1e-3):
        spec = SamplingSpec(
            n_particles=n_particles, seed=seed, prefilter=prefilter, strategy=self.strategy
        )
        if self.strategy == IID:
            return build_interaction(beam, laser, target, spec)
        plan = build_adaptive_plan(beam, laser, seed=seed, config=self.config)
        return build_interaction(beam, laser, target, spec, plan=plan)


def _base_config(**overrides) -> PilotConfig:
    """Reduced pilot cost (the plan is a *proposal*, and its precision is not the subject)."""
    values = dict(
        initial_regions=64,
        max_regions=256,
        pilot_points_per_region=8,
        pilot_quad_nodes=128,
        pilot_quad_panels=2,
    )
    values.update(overrides)
    return PilotConfig(**values)


def ablation_variants() -> list[Variant]:
    """The six arms of the ablation, isolating one effect at a time.

    Every arm is expressible with the shipped `PilotConfig` knobs, so this needs no
    production changes: `initial_regions == max_regions` disables refinement, and
    `luminosity_fraction = 0` disables luminosity allocation.
    """
    return [
        Variant("iid", IID, note="baseline"),
        Variant(
            "strat-s1-lam0", ADAPTIVE,
            _base_config(proposal_scale=1.0, luminosity_fraction=0.0,
                         initial_regions=256, max_regions=256),
            note="fixed balanced, s=1, pure QMC stratification",
        ),
        Variant(
            "strat-s1.41-lam0", ADAPTIVE,
            _base_config(proposal_scale=math.sqrt(2.0), luminosity_fraction=0.0,
                         initial_regions=256, max_regions=256),
            note="fixed balanced, s=sqrt2, broad reference",
        ),
        Variant(
            "split-lam0", ADAPTIVE,
            _base_config(luminosity_fraction=0.0, initial_regions=64, max_regions=256),
            note="adaptive splitting, uniform allocation (Q=B)",
        ),
        Variant(
            "fixed-lam0.75", ADAPTIVE,
            _base_config(luminosity_fraction=0.75, initial_regions=256, max_regions=256),
            note="fixed partition, luminosity allocation",
        ),
        Variant("full", ADAPTIVE, _base_config(), note="current shipped scheme"),
    ]


def lambda_variants(values=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[Variant]:
    return [
        Variant(f"lam{value:g}", ADAPTIVE,
                _base_config(luminosity_fraction=value, initial_regions=256, max_regions=256),
                note="fixed partition, lambda sweep")
        for value in values
    ]


def scale_variants(scales=(1.0, 1.2, math.sqrt(2.0)), lam=0.75) -> list[Variant]:
    return [
        Variant(f"s{scale:.3g}-lam{lam:g}", ADAPTIVE,
                _base_config(proposal_scale=scale, luminosity_fraction=lam,
                             initial_regions=256, max_regions=256),
                note="proposal-scale sweep at fixed lambda")
        for scale in scales
    ]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    beam_overrides: dict
    laser_overrides: dict
    note: str = ""


def scenarios() -> list[Scenario]:
    """The wider bank, chosen so the useful luminosity occupies very different fractions.

    The hypothesis under test is that localized interactions favour adaptive sampling most,
    because there the luminosity is concentrated in a smaller piece of the bunch. `tight_focus`
    and `wide_bunch_mismatch` are the extremes of that; `baseline` is the least favourable
    case because its overlap is comparatively smooth.
    """
    wide = dict(sigma_x=400.0, sigma_y=400.0)
    return [
        Scenario("baseline", {}, {}, "smooth, well-matched: least favourable case"),
        Scenario("tight_focus", {}, dict(sigma_x=1.5, sigma_y=1.5), "very localized pulse"),
        Scenario("focus_3um", {}, dict(sigma_x=3.0, sigma_y=3.0), "localized pulse"),
        Scenario("wide_bunch", wide, dict(sigma_x=4.0, sigma_y=4.0, duration=1.0),
                 "bunch 100x wider than the spot"),
        Scenario("x_offset", {}, dict(x_off=30.0), "transverse offset"),
        Scenario("t_offset", {}, dict(t_off=0.2), "temporal offset (ps)"),
        Scenario("crossing", {}, dict(theta_xz=15.0), "crossing angle 15 mrad"),
        Scenario("astigmatic", {}, dict(sigma_x=5.0, sigma_y=20.0), "astigmatic focus"),
        Scenario("twiss_corr", dict(alpha_x=0.8, alpha_y=-0.5, rho_z_gamma=0.5,
                                    rho_x_gamma=0.3, rho_thx_gamma=0.2), {},
                 "Twiss tilt + gamma correlations"),
        Scenario("displaced_foci", {}, dict(z_fx=3.0, z_fy=3.0), "displaced foci (mm)"),
    ]


def build(scenario: Scenario, laser, beam):
    """Apply a scenario's overrides, in the unit the caller wrote them in."""
    # One unit map for both dataclasses: the overrides are written in the same friendly
    # units the scenario list uses (um, ps, mrad, mm).
    unit_of = {
        "sigma_x": "um", "sigma_y": "um", "duration": "ps", "theta_xz": "mrad",
        "z_fx": "mm", "z_fy": "mm", "x_off": "um", "t_off": "ps",
    }
    from gammaforge.io.units import Quantity

    beam_ = beam
    for key, value in scenario.beam_overrides.items():
        if key in unit_of:
            beam_ = replace(beam_, **{key: Quantity(value, unit_of[key])})
        else:
            beam_ = replace(beam_, **{key: value})
    laser_ = laser
    for key, value in scenario.laser_overrides.items():
        laser_ = replace(laser_, **{key: Quantity(value, unit_of[key])})
    return beam_, laser_


# ---------------------------------------------------------------------------
# Fixed-edge deposition
# ---------------------------------------------------------------------------
def reference_edges(samples, bins) -> tuple[np.ndarray, ...]:
    """Edge set for the 5D table, from a high-statistics run's own support.

    Percentile-clipped rather than min/max: a 400k-particle reference has a few far-flung
    particles, and letting one of them stretch an edge would push every real cell into a
    single corner. The clip is a fixed fraction, applied identically to every scenario.
    """
    out = []
    for axis, name in enumerate(("gamma", "theta_x", "theta_y", "a0_shape", "chirp_mean")):
        values = np.asarray(getattr(samples, name), dtype=float)
        lo, hi = np.quantile(values, [1e-5, 1 - 1e-5])
        if not hi > lo:
            lo, hi = float(np.min(values)), float(np.max(values)) + 1.0
        margin = 0.02 * (hi - lo)
        out.append(np.linspace(lo - margin, hi + margin, bins[axis] + 1))
    return tuple(out)


def cell_index(samples, edges) -> np.ndarray:
    """Flat 5D cell index per particle; out-of-range particles are dropped by the caller."""
    indices = []
    for axis, name in enumerate(("gamma", "theta_x", "theta_y", "a0_shape", "chirp_mean")):
        values = np.asarray(getattr(samples, name), dtype=float)
        idx = np.searchsorted(edges[axis], values, side="right") - 1
        indices.append(np.clip(idx, 0, len(edges[axis]) - 2))
    flat = np.ravel_multi_index(tuple(indices), tuple(len(e) - 1 for e in edges))
    return flat


def _moment_channels(samples, flat, n_cells):
    """Replicate DER016's co-shaped channels on a caller-supplied cell assignment."""
    luminosity = np.asarray(samples.luminosity, dtype=float)
    shape = np.asarray(samples.a0_shape, dtype=float)
    chirp = np.asarray(samples.chirp_mean, dtype=float)
    volume = 1.0
    for edges in CURRENT_EDGES:
        volume *= float(np.diff(edges).mean())
    H = np.bincount(flat, weights=luminosity, minlength=n_cells) / volume
    var_a = np.bincount(flat, weights=luminosity * shape**2, minlength=n_cells)
    var_c = np.bincount(flat, weights=luminosity * chirp**2, minlength=n_cells)
    cov = np.bincount(flat, weights=luminosity * shape * chirp, minlength=n_cells)
    first = np.bincount(flat, weights=luminosity, minlength=n_cells)
    safe = np.maximum(first, 1e-300)
    shape_mean = np.bincount(flat, weights=luminosity * shape, minlength=n_cells) / safe
    chirp_mean = np.bincount(flat, weights=luminosity * chirp, minlength=n_cells) / safe
    H_var_a = (var_a - first * shape_mean**2) / safe / volume
    H_var_c = (var_c - first * chirp_mean**2) / safe / volume
    H_cov = (cov - first * shape_mean * chirp_mean) / safe / volume
    return H, H_var_a, H_var_c, H_cov, float(luminosity.sum())


#: Set by `deposit_fixed`; the channel helper needs the volume. Module-level rather than
#: threaded because it is only ever read immediately after being set.
CURRENT_EDGES = ()


def deposit_fixed(samples, edges, scheme="nearest") -> ShapeTable:
    """Deposit Stage-0 samples onto *fixed* edges, producing a real `ShapeTable`.

    Building the dataclass directly is what lets Stage 1.5 and Stage 2 run on a common grid
    across every variant — the engine's own `deposit_shape_table` re-derives edges per call,
    which is right for production and makes cross-run comparison meaningless here.

    CIC spreads each particle over its 32 neighbours, which is the scheme the "per-cell
    population noise" diagnosis most directly implicates.
    """
    global CURRENT_EDGES
    CURRENT_EDGES = edges
    shape_grid = tuple(len(e) - 1 for e in edges)
    n_cells = int(np.prod(shape_grid))

    if scheme == "nearest":
        flat = cell_index(samples, edges)
        H, H_var_a, H_var_c, H_cov, total = _moment_channels(samples, flat, n_cells)
    elif scheme == "cic":
        H, H_var_a, H_var_c, H_cov, total = _cic_channels(samples, edges, shape_grid, n_cells)
    else:
        raise ValueError(f"deposit_fixed: scheme must be nearest|cic, got {scheme!r}")

    return ShapeTable(
        gamma_edges=edges[0], theta_x_edges=edges[1], theta_y_edges=edges[2],
        a0_shape_edges=edges[3], chirp_edges=edges[4],
        H=H.reshape(shape_grid),
        H_var_a_shape=H_var_a.reshape(shape_grid),
        H_var_chirp=H_var_c.reshape(shape_grid),
        H_cov_a_chirp_shape=H_cov.reshape(shape_grid),
        total_weight=total, scheme=scheme,
        source_intensity_peak=float(samples.intensity_peak),
    )


def _cic_channels(samples, edges, shape_grid, n_cells):
    """CIC deposition of H and the three moment channels onto fixed edges.

    Accumulates over all 2^5 nearest-corner combinations. Every channel is a *first* moment
    of a per-particle quantity, so the same loop serves all four plus the two sums the
    variances need — CIC conserves total mass exactly, which is the property that makes it
    worth including at all.
    """
    volume = 1.0
    for edge in edges:
        volume *= float(np.diff(edge).mean())
    lum = np.asarray(samples.luminosity, dtype=float)
    shape = np.asarray(samples.a0_shape, dtype=float)
    chirp = np.asarray(samples.chirp_mean, dtype=float)

    keys = ("H", "m_la2", "m_lc2", "m_lac", "first", "sum_la", "sum_lc")
    acc = {key: np.zeros(n_cells) for key in keys}

    # Per-axis (lower_index, lower_weight) and (upper_index, upper_weight).
    per_axis = []
    for axis, name in enumerate(("gamma", "theta_x", "theta_y", "a0_shape", "chirp_mean")):
        values = np.asarray(getattr(samples, name), dtype=float)
        edge = edges[axis]
        upper = np.searchsorted(edge, values, side="right") - 1
        upper = np.clip(upper, 0, len(edge) - 2)
        lower = np.clip(upper - 1, 0, len(edge) - 2)
        width = edge[upper + 1] - edge[upper]
        weight_upper = (values - edge[upper]) / width
        per_axis.append(((lower, 1.0 - weight_upper), (upper, weight_upper)))

    for combination in range(2**5):
        index, weight = [], np.ones(lum.shape, dtype=float)
        for axis in range(5):
            pick = (combination >> axis) & 1
            idx_axis, w_axis = per_axis[axis][pick]
            index.append(idx_axis)
            weight = weight * w_axis
        flat = np.ravel_multi_index(tuple(index), shape_grid)
        acc["H"] += np.bincount(flat, weights=weight * lum, minlength=n_cells)
        acc["m_la2"] += np.bincount(flat, weights=weight * lum * shape**2, minlength=n_cells)
        acc["m_lc2"] += np.bincount(flat, weights=weight * lum * chirp**2, minlength=n_cells)
        acc["m_lac"] += np.bincount(flat, weights=weight * lum * shape * chirp, minlength=n_cells)
        acc["first"] += np.bincount(flat, weights=weight * lum, minlength=n_cells)
        acc["sum_la"] += np.bincount(flat, weights=weight * lum * shape, minlength=n_cells)
        acc["sum_lc"] += np.bincount(flat, weights=weight * lum * chirp, minlength=n_cells)

    safe = np.maximum(acc["first"], 1e-300)
    shape_mean = acc["sum_la"] / safe
    chirp_mean = acc["sum_lc"] / safe
    H = acc["H"] / volume
    H_var_a = (acc["m_la2"] - acc["first"] * shape_mean**2) / safe / volume
    H_var_c = (acc["m_lc2"] - acc["first"] * chirp_mean**2) / safe / volume
    H_cov = (acc["m_lac"] - acc["first"] * shape_mean * chirp_mean) / safe / volume
    return H, H_var_a, H_var_c, H_cov, float(lum.sum())


# ---------------------------------------------------------------------------
# Run + metrics
# ---------------------------------------------------------------------------
def spectral_grid(beam, laser, target) -> np.ndarray:
    ranges = auto_ranges(target, beam, laser)
    energy_hi = float(ranges[OutputKind.SPECTRUM][Axis.ENERGY][1])
    return np.linspace(0.0, energy_hi / (4.0 * float(laser.photon_energy())), 600)


def run(variant: Variant, beam, laser, target, n_particles, seed, edges, *,
        scheme="nearest", grid=None, prefilter=1e-3) -> dict:
    """One variant at one budget, deposited on fixed edges and pushed through Stage 2."""
    start = time.perf_counter()
    interaction = variant.interaction(beam, laser, target, n_particles, seed, prefilter)
    samples = integrate_trajectories(
        interaction.bunch, laser, interaction.N_e, n_steps=STAGE0_STEPS,
        threshold=prefilter, chunk=CHUNK,
    )
    shape_table = deposit_fixed(samples, edges, scheme=scheme)
    table = retarget_ahat(shape_table, float(samples.intensity_peak))
    spectrum = spectrum_from_table(table, 0.0, 0.0, grid)
    weights = interaction.bunch.weight
    return {
        "yield": float(np.sum(samples.luminosity)),
        "H": shape_table.H.copy(),
        "H_sum": float(shape_table.H.sum()),
        "spectrum": spectrum,
        "centroid": float(np.sum(grid * spectrum) / np.sum(spectrum)),
        "n_particles": interaction.bunch.n_particles,
        "weight_spread": float(weights.max() / weights.min()) if interaction.bunch.n_particles else 1.0,
        "n_eff": float(
            (weights.sum() ** 2 / np.sum(weights**2)) if interaction.bunch.n_particles else 0.0
        ),
        "seconds": time.perf_counter() - start,
    }


def table_l1(a, b) -> float:
    """L1 between two normalized 5D densities on the same grid.

    **Not usable as a quality metric at these resolutions, and reported only as a
    diagnostic.** A 5D table at (24,16,16,32,8) has 1.57e6 cells; a 400k-particle reference
    disagrees with *itself* at L1 ~ 0.2 across independent seeds, and a 40k run sits at
    ~0.45. The number is dominated by which cells happen to be empty, not by the estimator.
    `marginal_l1` below is the metric with resolving power.
    """
    pa = a / max(a.sum(), 1e-300)
    pb = b / max(b.sum(), 1e-300)
    return float(np.sum(np.abs(pa - pb)))


#: Axis pairs for the 2D marginals: the pairs Stage 2 and the physics actually couple.
AXES_5D = ("gamma", "theta_x", "theta_y", "a0_shape", "chirp_mean")
MARGINAL_PAIRS = ((0, 1), (0, 3), (0, 4), (1, 2), (3, 4))


def marginals(H: np.ndarray) -> dict:
    """1D and 2D marginals of the 5D table, each normalized to unit sum."""
    out = {}
    for axis in range(H.ndim):
        out[f"m1_{AXES_5D[axis]}"] = H.sum(axis=tuple(i for i in range(H.ndim) if i != axis))
    for a, b in MARGINAL_PAIRS:
        rest = tuple(i for i in range(H.ndim) if i not in (a, b))
        out[f"m2_{AXES_5D[a]}_{AXES_5D[b]}"] = H.sum(axis=rest)
    return {k: v / max(v.sum(), 1e-300) for k, v in out.items()}


def marginal_l1(a: np.ndarray, b: np.ndarray) -> dict:
    """Worst and mean L1 over the marginals of two 5D tables.

    Marginals are what the physics and Stage 2 consume, and — unlike the raw 5D density —
    they are densely occupied even at 10k particles, so they can actually distinguish
    estimators. The worst case is reported because a sampler that is good on average and bad
    on one axis (say the nonlinear one) would hide in the mean.
    """
    ma, mb = marginals(a), marginals(b)
    values = {k: float(np.sum(np.abs(ma[k] - mb[k]))) for k in ma}
    return {
        "marg_worst": max(values.values()),
        "marg_mean": float(np.mean(list(values.values()))),
        **{f"L1_{k}": v for k, v in values.items()},
    }


def errors(result, reference) -> dict:
    out = {
        "yield": abs(result["yield"] - reference["yield"]) / reference["yield"],
        "table_l1": table_l1(result["H"], reference["H"]),
        "spectrum": float(
            np.sum(np.abs(result["spectrum"] - reference["spectrum"]))
            / np.sum(np.abs(reference["spectrum"]))
        ),
        "centroid": abs(result["centroid"] - reference["centroid"])
        / abs(reference["centroid"]),
    }
    out.update(marginal_l1(result["H"], reference["H"]))
    return out


def merge(runs: list[dict]) -> dict:
    """Average replicates into the single result the errors are scored against.

    The *merged* result is what an R-seed estimator actually produces, so it is the honest
    point estimate. It is not, however, a measure of uncertainty -- for that see `stats`.
    """
    return {
        "yield": float(np.mean([r["yield"] for r in runs])),
        "H": np.mean([r["H"] for r in runs], axis=0),
        "spectrum": np.mean([r["spectrum"] for r in runs], axis=0),
        "centroid": float(np.mean([r["centroid"] for r in runs])),
    }


def make_reference(variant_name, beam, laser, target, n_particles, seeds, edges, **kw):
    """Mean reference over several independent IID runs, with the spread that bounds it."""
    runs = [
        run(Variant("iid", IID), beam, laser, target, n_particles, seed, edges, **kw)
        for seed in seeds
    ]
    return {
        "yield": float(np.mean([r["yield"] for r in runs])),
        "H": np.mean([r["H"] for r in runs], axis=0),
        "spectrum": np.mean([r["spectrum"] for r in runs], axis=0),
        "centroid": float(np.mean([r["centroid"] for r in runs])),
        # Provenance. Without these the tables printed "reference: ? particles", because a
        # floor is meaningless without the run that produced it -- a floor measured at 400k and
        # one measured at 4M differ by more than most of the effects being measured.
        "n_particles": int(n_particles),
        "seeds": [int(s) for s in seeds],
        "yield_spread": float(
            (max(r["yield"] for r in runs) - min(r["yield"] for r in runs))
            / np.mean([r["yield"] for r in runs])
        ),
        "table_l1_spread": float(
            np.mean([table_l1(r["H"], runs[0]["H"]) for r in runs])
        ),
        "marg_worst_spread": float(
            np.mean([marginal_l1(r["H"], runs[0]["H"])["marg_worst"] for r in runs])
        ),
        "marg_mean_spread": float(
            np.mean([marginal_l1(r["H"], runs[0]["H"])["marg_mean"] for r in runs])
        ),
        "spectrum_spread": float(
            np.mean([
                np.sum(np.abs(r["spectrum"] - runs[0]["spectrum"]))
                / np.sum(np.abs(runs[0]["spectrum"]))
                for r in runs
            ])
        ),
    }
