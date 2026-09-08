#!/usr/bin/env python3
"""Calculate total Thomson/Compton photon yield as a function of duty cycle.

Compares two approaches to simulate a laser pulse train colliding with an electron bunch:
1. Direct Pulse Train Field (Method 1):
   Uses PulseTrainParaxialLaser with an expanded ActiveRegion spanning the burst.
2. Sum of Sub-Pulses (Method 2):
   Programmatically creates N_p individual sub-pulses with delays t_k, simulates each
   interaction independently in its own compact active region, and sums the resulting yields.

Usage:
    python scripts/calculate_duty_cycle_yield.py
    python scripts/calculate_duty_cycle_yield.py --n-subpulses 10 --plot duty_cycle_yield.png
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

import numpy as np

from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.bunch import GaussianElectronBeam, sample_gaussian_bunch
from gammaforge.io.laser import PulseTrainParaxialLaser
from gammaforge.io.units import C_CGS, Quantity, ureg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and validate total photon yield vs duty cycle for laser pulse trains."
    )
    parser.add_argument(
        "--n-subpulses",
        type=int,
        default=5,
        help="Number of sub-pulses in the train (default: 5).",
    )
    parser.add_argument(
        "--pulse-energy",
        type=float,
        default=0.5,
        help="Total laser energy in Joules summed across all sub-pulses (default: 0.5 J).",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=800.0,
        help="Laser carrier wavelength in nm (default: 800.0 nm).",
    )
    parser.add_argument(
        "--waist",
        type=float,
        default=20.0,
        help="Laser RMS intensity waist sigma_x = sigma_y in um (default: 20.0 um).",
    )
    parser.add_argument(
        "--subpulse-duration",
        type=float,
        default=30.0,
        help="RMS intensity duration of a single sub-pulse in fs (default: 30.0 fs).",
    )
    parser.add_argument(
        "--bunch-charge",
        type=float,
        default=100.0,
        help="Electron bunch charge in pC (default: 100.0 pC).",
    )
    parser.add_argument(
        "--bunch-energy",
        type=float,
        default=100.0,
        help="Electron beam kinetic energy in MeV (default: 100.0 MeV).",
    )
    parser.add_argument(
        "--bunch-sigma-z",
        type=float,
        default=30.0,
        help="Electron bunch RMS length sigma_z in um (default: 30.0 um).",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=200,
        help="Number of macroparticles in sampled bunch (default: 200).",
    )
    parser.add_argument(
        "--n-steps-train",
        type=int,
        default=512,
        help="Trajectory integration steps for the full pulse train (default: 512).",
    )
    parser.add_argument(
        "--n-steps-subpulse",
        type=int,
        default=64,
        help="Trajectory integration steps for each individual sub-pulse (default: 64).",
    )
    parser.add_argument(
        "--duty-cycles",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
        help="List of duty cycles D = tau_p / T_rep to evaluate (default: 0.01 to 1.0).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional path to save plot (e.g. 'duty_cycle_yield.png').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bunch sampling (default: 42).",
    )
    return parser.parse_args(argv)


def run_sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    print("=" * 80)
    print("GammaForge: Total Yield vs Duty Cycle Sweep & Validation")
    print("=" * 80)
    print(f"Total Laser Energy:    {args.pulse_energy:.3f} J")
    print(f"Sub-pulse Duration:    {args.subpulse_duration:.1f} fs (RMS)")
    print(f"Laser Waist (sigma):   {args.waist:.1f} µm")
    print(f"Wavelength:            {args.wavelength:.1f} nm")
    print(f"Number of Sub-pulses:  {args.n_subpulses}")
    print(f"Electron Bunch Energy: {args.bunch_energy:.1f} MeV (Charge: {args.bunch_charge:.1f} pC)")
    print(f"Macroparticles:        {args.n_particles}")
    print(f"Integration Steps:     Train: {args.n_steps_train}, Sub-pulse: {args.n_steps_subpulse}")
    print("-" * 80)

    # 1. Build electron bunch
    beam = GaussianElectronBeam(
        bunch_charge=Quantity(args.bunch_charge, "pC"),
        kinetic_energy=Quantity(args.bunch_energy, "MeV"),
        rel_energy_spread=0.005,
        sigma_x=Quantity(10.0, "um"),
        sigma_y=Quantity(10.0, "um"),
        emit_x=Quantity(1.0, "mm*mrad"),
        emit_y=Quantity(1.0, "mm*mrad"),
        sigma_z=Quantity(args.bunch_sigma_z, "um"),
    )
    bunch = sample_gaussian_bunch(beam, n_particles=args.n_particles, seed=args.seed)
    n_electrons = beam.n_electrons()

    # Calculate Rayleigh range z_R
    sigma_cm = args.waist * 1e-4
    lambda_cm = args.wavelength * 1e-7
    z_r_cm = 4.0 * math.pi * sigma_cm**2 / lambda_cm
    z_r_um = z_r_cm * 1e4
    print(f"Laser Rayleigh Range z_R: {z_r_um:.1f} µm ({z_r_cm*10:.2f} mm)")
    print("-" * 80)

    results: list[dict[str, float]] = []

    header = (
        f"{'Duty Cycle D':>12} | {'T_rep [fs]':>10} | {'ΔZ_coll [µm]':>12} | "
        f"{'ΔZ/z_R':>8} | {'Yield (Train)':>14} | {'Yield (Sub-Sum)':>15} | {'Diff (%)':>9}"
    )
    print(header)
    print("-" * len(header))

    duty_cycles = sorted(args.duty_cycles)

    for d in duty_cycles:
        t_rep_fs = args.subpulse_duration / d
        # Head-on collision spacing: z_coll = c * t_k / 2
        burst_duration_s = (args.n_subpulses - 1) * (t_rep_fs * 1e-15)
        delta_z_coll_cm = 0.5 * C_CGS * burst_duration_s
        delta_z_coll_um = delta_z_coll_cm * 1e4
        ratio_zr = delta_z_coll_cm / z_r_cm

        # Create PulseTrainParaxialLaser
        train = PulseTrainParaxialLaser(
            pulse_energy=Quantity(args.pulse_energy, "J"),
            wavelength=Quantity(args.wavelength, "nm"),
            sigma_x=Quantity(args.waist, "um"),
            sigma_y=Quantity(args.waist, "um"),
            subpulse_duration=Quantity(args.subpulse_duration, "fs"),
            repetition_period=Quantity(t_rep_fs, "fs"),
            n_subpulses=args.n_subpulses,
        )

        # Method 1: Train integration
        samples_train = integrate_trajectories(
            bunch, train, n_electrons=n_electrons, n_steps=args.n_steps_train
        )
        yield_train = samples_train.total_yield()

        # Method 2: Sum of individual sub-pulses
        subpulses = train.subpulses()
        sub_yields = [
            integrate_trajectories(
                bunch, sub, n_electrons=n_electrons, n_steps=args.n_steps_subpulse
            ).total_yield()
            for sub in subpulses
        ]
        yield_sub_sum = sum(sub_yields)

        diff_pct = 100.0 * (yield_train - yield_sub_sum) / yield_sub_sum if yield_sub_sum > 0 else 0.0

        row = (
            f"{d:12.4f} | {t_rep_fs:10.1f} | {delta_z_coll_um:12.1f} | "
            f"{ratio_zr:8.3f} | {yield_train:14.4e} | {yield_sub_sum:15.4e} | {diff_pct:+8.2f}%"
        )
        print(row)

        results.append({
            "duty_cycle": d,
            "t_rep_fs": t_rep_fs,
            "delta_z_coll_um": delta_z_coll_um,
            "ratio_zr": ratio_zr,
            "yield_train": yield_train,
            "yield_sub_sum": yield_sub_sum,
            "diff_pct": diff_pct,
        })

    print("-" * len(header))
    print("\nSummary & Physical Insights:")
    print("1. Compact Burst Regime (High Duty Cycle, D ~ 1, ΔZ_coll << z_R):")
    print("   All sub-pulses collide within the optical depth of focus where beam waist is minimal (w ≈ w0).")
    print(f"   Max Yield achieved at D = {duty_cycles[-1]}: {results[-1]['yield_sub_sum']:.4e} photons.")
    print("2. Sparse Burst Regime (Low Duty Cycle, D << 1, ΔZ_coll >> z_R):")
    print("   Peripheral sub-pulses collide in the diffraction wings (|z| >> z_R) where spot size")
    print("   expands geometrically as w(z) = w0*sqrt(1 + (z/z_R)^2), lowering the local photon density.")
    drop_pct = 100.0 * (1.0 - results[0]['yield_sub_sum'] / results[-1]['yield_sub_sum'])
    print(f"   Yield at lowest duty cycle (D = {duty_cycles[0]}): {results[0]['yield_sub_sum']:.4e} "
          f"({drop_pct:.1f}% reduction due to hourglass divergence).")
    print("3. Trajectory Sampling Fidelity:")
    print("   Method 2 (summing independent sub-pulses) allocates dedicated time steps to each")
    print("   sub-pulse's individual active window, avoiding spike-undersampling across wide gaps.")

    if args.plot:
        plot_results(results, args.plot, z_r_um, args.n_subpulses)

    return results


def plot_results(
    results: list[dict[str, float]],
    save_path: str,
    z_r_um: float,
    n_subpulses: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n[Warning] matplotlib not installed; skipping plot generation to '{save_path}'.")
        return

    d_vals = [r["duty_cycle"] for r in results]
    y_train = [r["yield_train"] for r in results]
    y_sub_sum = [r["yield_sub_sum"] for r in results]
    y_norm = y_sub_sum[-1] if y_sub_sum[-1] > 0 else 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top panel: Normalized Yield vs Duty Cycle
    ax1.plot(
        d_vals,
        [y / y_norm for y in y_sub_sum],
        "o-",
        color="#1f77b4",
        label="Method 2: Sum of Sub-Pulses (Benchmark)",
        lw=2.2,
        markersize=6,
    )
    ax1.plot(
        d_vals,
        [y / y_norm for y in y_train],
        "s--",
        color="#d62728",
        label="Method 1: Direct Pulse Train Field",
        lw=1.8,
        markersize=5,
        alpha=0.85,
    )
    ax1.set_xscale("log")
    ax1.set_ylabel(r"Normalized Yield $N_\gamma / N_{\gamma, \rm max}$", fontsize=11)
    ax1.set_title(
        f"Photon Yield vs Duty Cycle ($N_p = {n_subpulses}$ Sub-pulses, Rayleigh range $z_R = {z_r_um:.0f}\\,\\mu\\mathrm{{m}}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", frameon=True)

    # Bottom panel: Collision extent relative to Rayleigh range & sampling diff
    ax2.plot(
        d_vals,
        [r["ratio_zr"] for r in results],
        "^-",
        color="#2ca02c",
        label=r"Burst Collision Span $\Delta Z_{\rm coll} / z_R$",
        lw=2.0,
    )
    ax2.axhline(1.0, color="gray", linestyle=":", label=r"Rayleigh Limit ($\Delta Z_{\rm coll} = z_R$)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Duty Cycle $D = \tau_p / T_{\rm rep}$", fontsize=11)
    ax2.set_ylabel(r"$\Delta Z_{\rm coll} / z_R$", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot successfully saved to: {save_path}")


def main() -> None:
    args = parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
