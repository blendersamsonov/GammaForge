#!/usr/bin/env python3
"""Compare unnormalised analytical and xigma total yields versus crossing angle.

The original XIGMA representative operating point is used: a 10 nC, gamma=2000
electron bunch colliding with a 20 J, 1030 nm laser.  Both engines receive the
same typed GammaForge interaction; only public engine APIs are used.  Zero angle
means a head-on collision (laser wave vector along -z), and positive angles are
``theta_xz`` tilts in the xz plane.

Run from the repository root::

    PYTHONPATH=$PWD/src python examples/crossing_angle_yield.py

The output directory contains the figure, input/provenance JSON, the raw seeded
xigma samples, and both numerical-convergence tables.  The comparison concerns
the Stage-0 overlap total only.  It is not a validation of xigma's angle-resolved
Stage-2 emission kernel.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gammaforge.engines.analytical.engine import AnalyticalEngine
from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.io import (
    GaussianElectronBeam,
    GaussianParaxialLaser,
    OutputKind,
    OutputRequest,
    SamplingSpec,
    Target,
    build_interaction,
)
from gammaforge.io.units import MEC2_CGS, Quantity


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output" / "crossing_angle_yield"

# This is deliberately the original XIGMA `git-repo/example-config.toml` point (not a
# new model configuration): gamma=2000, 10 nC, 20 J, 1030 nm, 10 um RMS beam/laser widths.
# Its peak linear-equivalent a0 is approximately 0.181, safely in xigma's low-a0
# regime.  Angles through 40 mrad expose the geometric suppression while keeping the
# quantity a total overlap, whose crossing-angle treatment is implemented in both engines.
ANGLES_MRAD = np.arange(0.0, 45.0, 5.0)
SEEDS = (20260906, 20260907, 20260908)
N_PARTICLES = 15_000
XIGMA_STEPS = 600
PREFILTER = 1e-4

# The highest setting is the plotted analytical reference.  The other settings make the
# longitudinal and exact-transverse quadrature convergence auditable in the data output.
ANALYTICAL_CONVERGENCE = ((4001, 51), (8001, 101), (12001, 151))
XIGMA_STEP_CONVERGENCE = (200, 400, 800)
CONVERGENCE_ANGLES_MRAD = (0.0, 20.0, 40.0)

GAMMA0 = 2000.0
KINETIC_ENERGY_ERG = (GAMMA0 - 1.0) * MEC2_CGS
# The source's 0.005 is relative to gamma.  GammaForge's field is relative to kinetic
# energy, so convert it explicitly instead of treating the two as identical.
REL_ENERGY_SPREAD = 0.005 * GAMMA0 * MEC2_CGS / KINETIC_ENERGY_ERG
BEAM = GaussianElectronBeam(
    bunch_charge=Quantity(10.0, "nC"),
    kinetic_energy=Quantity(KINETIC_ENERGY_ERG, "erg"),
    rel_energy_spread=REL_ENERGY_SPREAD,
    sigma_x=Quantity(10.0, "um"),
    sigma_y=Quantity(10.0, "um"),
    emit_x=Quantity(1e-4, "cm * rad") / GAMMA0,
    emit_y=Quantity(1e-6, "cm * rad") / GAMMA0,
    sigma_z=Quantity(10.0, "ps"),
)
LASER = GaussianParaxialLaser(
    pulse_energy=Quantity(20.0, "J"),
    wavelength=Quantity(1030.0, "nm"),
    sigma_x=Quantity(10.0, "um"),
    sigma_y=Quantity(10.0, "um"),
    duration=Quantity(30.0, "ps"),
)
TARGET = Target(theta_x_col=Quantity(1.0, "mrad"), theta_y_col=Quantity(1.0, "mrad"))


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _total_yield(result) -> float:
    """Read the public 0D result without touching a xigma stage or Collision."""
    return result.photon_slices[OutputKind.TOTAL_YIELD].integrate()


def interaction_for(angle_mrad: float, seed: int, n_particles: int):
    """Build one shared, explicitly seeded interaction at an xz crossing angle."""
    target = replace(
        TARGET,
        outputs=(OutputRequest(OutputKind.TOTAL_YIELD),),
    )
    return build_interaction(
        beam=BEAM,
        laser=replace(LASER, theta_xz=Quantity(angle_mrad, "mrad")),
        target=target,
        sampling=SamplingSpec(n_particles=n_particles, seed=seed, prefilter=PREFILTER),
    )


def analytical_yield(angle_mrad: float, n_quad_overlap: int, n_quad_u: int) -> float:
    interaction = interaction_for(angle_mrad, SEEDS[0], 1)
    params = AnalyticalEngine.schema.with_values(
        n_quad=401,
        n_quad_overlap=n_quad_overlap,
        n_quad_u=n_quad_u,
    )
    return _total_yield(AnalyticalEngine().run(interaction, params))


def xigma_yield(
    angle_mrad: float, seed: int, n_steps: int, n_particles: int = N_PARTICLES
) -> tuple[float, int]:
    interaction = interaction_for(angle_mrad, seed, n_particles)
    params = XigmaEngine.schema.with_values(n_steps=n_steps, threshold=PREFILTER)
    return _total_yield(XigmaEngine().run(interaction, params)), interaction.bunch.n_particles


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary: list[dict], agreement: dict) -> None:
    x = np.asarray([row["angle_mrad"] for row in summary])
    analytical = np.asarray([row["analytical_yield"] for row in summary])
    xigma = np.asarray([row["xigma_mean"] for row in summary])
    xigma_sem = np.asarray([row["xigma_sem"] for row in summary])
    rel_percent = 100.0 * (xigma / analytical - 1.0)
    rel_sem_percent = 100.0 * xigma_sem / analytical

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    figure, (yield_ax, residual_ax) = plt.subplots(
        2, 1, figsize=(3.45, 4.25), sharex=True, height_ratios=(2.2, 1.0), layout="constrained"
    )
    yield_ax.plot(x, analytical / 1e10, color="#1b6ca8", lw=1.6, label="Analytical (converged quadrature)")
    yield_ax.errorbar(
        x,
        xigma / 1e10,
        yerr=xigma_sem / 1e10,
        color="#c9622f",
        marker="o",
        ms=3.8,
        capsize=2.2,
        lw=1.0,
        label="xigma (mean $\\pm$ SEM, 3 seeds)",
    )
    yield_ax.set_ylabel(r"Total yield [$10^{10}$ photons]")
    yield_ax.grid(alpha=0.28, lw=0.5)
    yield_ax.legend(frameon=False, loc="upper right")
    yield_ax.text(
        0.03,
        0.08,
        f"max |difference| = {agreement['max_abs_percent']:.2f}%",
        transform=yield_ax.transAxes,
        fontsize=8,
    )

    residual_ax.axhline(0.0, color="0.25", lw=0.8)
    residual_ax.errorbar(x, rel_percent, yerr=rel_sem_percent, color="#c9622f", marker="o", ms=3.8, capsize=2.2, lw=1.0)
    residual_ax.set_xlabel(r"Crossing angle $\theta_{xz}$ [mrad] (0 = head-on)")
    residual_ax.set_ylabel("xigma - analytical [%]")
    residual_ax.grid(alpha=0.28, lw=0.5)
    figure.savefig(OUTPUT / "crossing_angle_yield.png")
    figure.savefig(OUTPUT / "crossing_angle_yield.pdf")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    analytical_rows: list[dict] = []
    for angle in CONVERGENCE_ANGLES_MRAD:
        for n_quad_overlap, n_quad_u in ANALYTICAL_CONVERGENCE:
            analytical_rows.append(
                {
                    "angle_mrad": angle,
                    "n_quad_overlap": n_quad_overlap,
                    "n_quad_u": n_quad_u,
                    "yield": analytical_yield(angle, n_quad_overlap, n_quad_u),
                }
            )
    for angle in CONVERGENCE_ANGLES_MRAD:
        reference = next(
            row["yield"]
            for row in analytical_rows
            if row["angle_mrad"] == angle and row["n_quad_overlap"] == 12001 and row["n_quad_u"] == 151
        )
        for row in analytical_rows:
            if row["angle_mrad"] == angle:
                row["relative_to_finest"] = row["yield"] / reference - 1.0
    write_csv(OUTPUT / "analytical_convergence.csv", analytical_rows)

    finest_n_quad, finest_n_quad_u = ANALYTICAL_CONVERGENCE[-1]
    analytical_scan = {
        float(angle): analytical_yield(float(angle), finest_n_quad, finest_n_quad_u)
        for angle in ANGLES_MRAD
    }

    seed_rows: list[dict] = []
    for angle in ANGLES_MRAD:
        for seed in SEEDS:
            yield_value, kept_particles = xigma_yield(float(angle), seed, XIGMA_STEPS)
            seed_rows.append(
                {
                    "angle_mrad": float(angle),
                    "seed": seed,
                    "n_particles_requested": N_PARTICLES,
                    "n_particles_after_prefilter": kept_particles,
                    "n_steps": XIGMA_STEPS,
                    "yield": yield_value,
                }
            )
    write_csv(OUTPUT / "xigma_seed_samples.csv", seed_rows)

    step_rows: list[dict] = []
    for angle in CONVERGENCE_ANGLES_MRAD:
        for n_steps in XIGMA_STEP_CONVERGENCE:
            yield_value, kept_particles = xigma_yield(angle, SEEDS[0], n_steps)
            step_rows.append(
                {
                    "angle_mrad": angle,
                    "seed": SEEDS[0],
                    "n_steps": n_steps,
                    "n_particles_after_prefilter": kept_particles,
                    "yield": yield_value,
                }
            )
    for angle in CONVERGENCE_ANGLES_MRAD:
        reference = next(row["yield"] for row in step_rows if row["angle_mrad"] == angle and row["n_steps"] == 800)
        for row in step_rows:
            if row["angle_mrad"] == angle:
                row["relative_to_800_steps"] = row["yield"] / reference - 1.0
    write_csv(OUTPUT / "xigma_time_quadrature_convergence.csv", step_rows)

    particle_rows: list[dict] = []
    for angle in CONVERGENCE_ANGLES_MRAD:
        coarse, coarse_kept = xigma_yield(angle, SEEDS[0], XIGMA_STEPS, N_PARTICLES)
        fine, fine_kept = xigma_yield(angle, SEEDS[0], XIGMA_STEPS, 2 * N_PARTICLES)
        particle_rows.extend(
            (
                {
                    "angle_mrad": angle,
                    "seed": SEEDS[0],
                    "n_particles_requested": N_PARTICLES,
                    "n_particles_after_prefilter": coarse_kept,
                    "yield": coarse,
                    "relative_to_doubled_particles": coarse / fine - 1.0,
                },
                {
                    "angle_mrad": angle,
                    "seed": SEEDS[0],
                    "n_particles_requested": 2 * N_PARTICLES,
                    "n_particles_after_prefilter": fine_kept,
                    "yield": fine,
                    "relative_to_doubled_particles": 0.0,
                },
            )
        )
    write_csv(OUTPUT / "xigma_particle_count_convergence.csv", particle_rows)

    summary: list[dict] = []
    for angle in ANGLES_MRAD:
        samples = np.asarray([row["yield"] for row in seed_rows if row["angle_mrad"] == float(angle)])
        mean = float(np.mean(samples))
        sem = float(np.std(samples, ddof=1) / np.sqrt(samples.size))
        reference = analytical_scan[float(angle)]
        summary.append(
            {
                "angle_mrad": float(angle),
                "analytical_yield": reference,
                "xigma_mean": mean,
                "xigma_sample_std": float(np.std(samples, ddof=1)),
                "xigma_sem": sem,
                "relative_difference": mean / reference - 1.0,
            }
        )
    write_csv(OUTPUT / "crossing_angle_yield.csv", summary)

    differences = np.asarray([row["relative_difference"] for row in summary])
    agreement = {
        "mean_abs_percent": float(100.0 * np.mean(np.abs(differences))),
        "max_abs_percent": float(100.0 * np.max(np.abs(differences))),
        "head_on_relative_percent": float(100.0 * summary[0]["relative_difference"]),
        "largest_angle_relative_percent": float(100.0 * summary[-1]["relative_difference"]),
    }
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "configuration": {
            "source": "/home/alexander/Work/Code/XIGMA/git-repo/example-config.toml",
            "source_inputs": {
                "electron": {
                    "charge": "10 nC",
                    "gamma": GAMMA0,
                    "relative_energy_spread_to_gamma": 0.005,
                    "sigma_x": "10 um RMS",
                    "sigma_y": "10 um RMS",
                    "duration": "10 ps RMS",
                    "normalized_emit_x": "1e-4 cm rad",
                    "normalized_emit_y": "1e-6 cm rad",
                },
                "laser": {
                    "energy": "20 J",
                    "wavelength": "1030 nm",
                    "sigma_x": "10 um RMS",
                    "sigma_y": "10 um RMS",
                    "duration": "30 ps RMS",
                    "polarization_angle": "0 rad",
                    "beta_ff": 0.0,
                },
            },
            "gammaforge_mapping": {
                "kinetic_energy_erg": KINETIC_ENERGY_ERG,
                "relative_energy_spread_to_kinetic_energy": REL_ENERGY_SPREAD,
                "geometric_emit_x_cm_rad": BEAM.m("emit_x"),
                "geometric_emit_y_cm_rad": BEAM.m("emit_y"),
                "sigma_z_cm": BEAM.m("sigma_z"),
                "laser_duration_s": LASER.m("duration"),
            },
            "peak_linear_equivalent_a0": LASER.a0_peak(),
            "angles_mrad": ANGLES_MRAD.tolist(),
            "angle_definition": "theta_xz; 0 mrad is head-on (laser k along -z)",
            "sampling": {
                "n_particles": N_PARTICLES,
                "seeds": list(SEEDS),
                "prefilter": PREFILTER,
                "particle_count_convergence": [N_PARTICLES, 2 * N_PARTICLES],
            },
            "xigma": {"n_steps": XIGMA_STEPS, "step_convergence": list(XIGMA_STEP_CONVERGENCE)},
            "analytical": {
                "n_quad": 401,
                "plotted_n_quad_overlap": finest_n_quad,
                "plotted_n_quad_u": finest_n_quad_u,
                "convergence": [list(item) for item in ANALYTICAL_CONVERGENCE],
            },
        },
        "agreement": agreement,
        "scope_and_limits": (
            "Unnormalised total yield only: AnalyticalEngine's Gaussian-overlap integral versus "
            "XigmaEngine Stage-0 trajectory overlap. This comparison does not validate the "
            "angle-resolved Stage-2 emission kernel or a crossed spectrum shape."
        ),
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    make_figure(summary, agreement)

    print(f"Wrote {OUTPUT}")
    print(
        "Agreement: mean |xigma - analytical| = "
        f"{agreement['mean_abs_percent']:.3f}%, max = {agreement['max_abs_percent']:.3f}%"
    )


if __name__ == "__main__":
    main()
