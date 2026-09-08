#!/usr/bin/env python3
"""Generate publication-quality plots of Compton photon yield vs laser pulse duty cycle
for different ratios of electron bunch duration to laser pulse total duration.

Physical Context:
    In laser-electron interactions, delivering a fixed total laser energy E_tot across
    a train of N_p sub-pulses (duty cycle D = tau_p / T_rep) mitigates optical damage
    and suppresses nonlinear spectral broadening. However, decreasing D expands the
    temporal train duration T_burst = (N_p - 1) * tau_p / D and spreads collision
    locations over Delta Z_coll = c * T_burst / 2.

    When Delta Z_coll exceeds the Rayleigh range z_R, peripheral sub-pulses collide
    in the diverging wings w(z) > w0, reducing local photon density and degrading yield.
    The extent of this degradation depends strongly on the electron bunch duration
    tau_e = sigma_ez / c relative to the laser pulse duration T_burst.

Modes:
    1. 'fixed-bunch' (default):
       Each curve represents a specific electron bunch duration tau_e, specified by the
       ratio R0 = tau_e / T_burst_ref at compact packing (D = 1). As D is varied, the
       bunch duration remains fixed, illustrating how different physical bunches respond
       as the laser pulse train is stretched out.
    2. 'fixed-ratio':
       Each curve maintains a strictly constant ratio R = tau_e / T_burst(D) at every
       duty cycle point, scaling bunch duration with the stretched burst.

Usage:
    python scripts/plot_duty_cycle_vs_bunch_duration.py
    python scripts/plot_duty_cycle_vs_bunch_duration.py --ratios 0.1 0.5 1.0 2.0 5.0 --output duty_cycle_plot.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from gammaforge.engines.xigma.stages import integrate_trajectories
from gammaforge.io.bunch import GaussianElectronBeam, sample_gaussian_bunch
from gammaforge.io.laser import PulseTrainParaxialLaser
from gammaforge.io.units import C_CGS, Quantity


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot total photon yield vs duty cycle for different bunch-to-pulse duration ratios."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0, 3.0, 10.0],
        help="List of ratios R = tau_e / T_burst to evaluate (default: 0.1, 0.3, 1.0, 3.0, 10.0).",
    )
    parser.add_argument(
        "--duty-cycles",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
        help="List of duty cycles D = tau_p / T_rep (default: 0.01 to 1.0).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed-bunch", "fixed-ratio"],
        default="fixed-bunch",
        help="Sweep mode: 'fixed-bunch' (fixed tau_e based on ratio at D=1) or 'fixed-ratio' (constant tau_e/T_burst(D)).",
    )
    parser.add_argument(
        "--n-subpulses",
        type=int,
        default=10,
        help="Number of sub-pulses in the train N_p (default: 10).",
    )
    parser.add_argument(
        "--subpulse-duration",
        type=float,
        default=30.0,
        help="RMS intensity duration of a single sub-pulse in fs (default: 30.0 fs).",
    )
    parser.add_argument(
        "--pulse-energy",
        type=float,
        default=1.0,
        help="Total laser energy in Joules summed across all sub-pulses (default: 1.0 J).",
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
        default=15.0,
        help="Laser RMS intensity waist sigma_x = sigma_y in um (default: 15.0 um).",
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
        "--bunch-transverse-sigma",
        type=float,
        default=10.0,
        help="Electron beam RMS transverse size sigma_x = sigma_y in um (default: 10.0 um).",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=100,
        help="Number of macroparticles in sampled bunch (default: 100).",
    )
    parser.add_argument(
        "--n-steps-subpulse",
        type=int,
        default=48,
        help="Trajectory integration steps per sub-pulse (default: 48).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bunch sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/output/duty_cycle_yield/duty_cycle_vs_bunch_ratio.png",
        help="Output image path for the plot (default: examples/output/duty_cycle_yield/duty_cycle_vs_bunch_ratio.png).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="examples/output/duty_cycle_yield/duty_cycle_vs_bunch_ratio.csv",
        help="Output CSV path for the numerical results (default: examples/output/duty_cycle_yield/duty_cycle_vs_bunch_ratio.csv).",
    )
    return parser.parse_args(argv)


def run_multi_ratio_sweep(args: argparse.Namespace) -> dict[str, object]:
    print("=" * 85)
    print("GammaForge: Yield vs Duty Cycle Sweep for Multiple Bunch/Pulse Duration Ratios")
    print("=" * 85)
    print(f"Total Laser Energy:      {args.pulse_energy:.2f} J (Summed over {args.n_subpulses} sub-pulses)")
    print(f"Sub-pulse Duration:      {args.subpulse_duration:.1f} fs (RMS)")
    print(f"Laser Spot Waist:        {args.waist:.1f} µm (RMS)")
    print(f"Laser Wavelength:        {args.wavelength:.1f} nm")
    print(f"Electron Bunch Energy:   {args.bunch_energy:.1f} MeV (Charge: {args.bunch_charge:.1f} pC)")
    print(f"Sweep Mode:              {args.mode}")
    print(f"Tested Duration Ratios:  {args.ratios}")
    print(f"Tested Duty Cycles:      {sorted(args.duty_cycles)}")
    print("-" * 85)

    # Compute Rayleigh range z_R = 4*pi*sigma^2 / lambda (RES040 convention)
    sigma_cm = args.waist * 1e-4
    lambda_cm = args.wavelength * 1e-7
    z_r_cm = 4.0 * math.pi * sigma_cm**2 / lambda_cm
    z_r_um = z_r_cm * 1e4
    print(f"Laser Rayleigh Range z_R: {z_r_um:.1f} µm ({z_r_cm * 10:.2f} mm)")

    # Reference compact burst duration (at D = 1.0)
    t_burst_ref_fs = (args.n_subpulses - 1) * args.subpulse_duration
    t_burst_ref_s = t_burst_ref_fs * 1e-15
    print(f"Reference Burst Duration T_burst(D=1): {t_burst_ref_fs:.1f} fs ({t_burst_ref_s * 1e12:.3f} ps)")
    print("-" * 85)

    duty_cycles = sorted(args.duty_cycles)
    ratios = sorted(args.ratios)

    # Dictionary to hold results per ratio: ratio -> list of point dicts
    all_results: dict[float, list[dict[str, float]]] = {}

    for r in ratios:
        ratio_results: list[dict[str, float]] = []
        print(f"\nEvaluating Ratio R = {r:.2f} ...")

        # Determine reference bunch length
        if args.mode == "fixed-bunch":
            tau_bunch_s = r * t_burst_ref_s
            sigma_z_cm = C_CGS * tau_bunch_s
            sigma_z_um = sigma_z_cm * 1e4
            print(f"  Fixed Bunch Duration tau_e: {tau_bunch_s * 1e12:.3f} ps (sigma_z = {sigma_z_um:.1f} µm)")

            # Sample the electron bunch once for this ratio
            beam = GaussianElectronBeam(
                bunch_charge=Quantity(args.bunch_charge, "pC"),
                kinetic_energy=Quantity(args.bunch_energy, "MeV"),
                rel_energy_spread=0.005,
                sigma_x=Quantity(args.bunch_transverse_sigma, "um"),
                sigma_y=Quantity(args.bunch_transverse_sigma, "um"),
                emit_x=Quantity(1.0, "mm*mrad"),
                emit_y=Quantity(1.0, "mm*mrad"),
                sigma_z=Quantity(sigma_z_um, "um"),
            )
            bunch = sample_gaussian_bunch(beam, n_particles=args.n_particles, seed=args.seed)
            n_electrons = beam.n_electrons()

        for d in duty_cycles:
            t_rep_fs = args.subpulse_duration / d
            burst_duration_s = (args.n_subpulses - 1) * (t_rep_fs * 1e-15)
            delta_z_coll_cm = 0.5 * C_CGS * burst_duration_s
            delta_z_coll_um = delta_z_coll_cm * 1e4
            ratio_zr = delta_z_coll_cm / z_r_cm

            if args.mode == "fixed-ratio":
                # Scale tau_bunch to maintain exact ratio with stretched T_burst(D)
                tau_bunch_s = r * burst_duration_s
                sigma_z_cm = C_CGS * tau_bunch_s
                sigma_z_um = sigma_z_cm * 1e4
                beam = GaussianElectronBeam(
                    bunch_charge=Quantity(args.bunch_charge, "pC"),
                    kinetic_energy=Quantity(args.bunch_energy, "MeV"),
                    rel_energy_spread=0.005,
                    sigma_x=Quantity(args.bunch_transverse_sigma, "um"),
                    sigma_y=Quantity(args.bunch_transverse_sigma, "um"),
                    emit_x=Quantity(1.0, "mm*mrad"),
                    emit_y=Quantity(1.0, "mm*mrad"),
                    sigma_z=Quantity(sigma_z_um, "um"),
                )
                bunch = sample_gaussian_bunch(beam, n_particles=args.n_particles, seed=args.seed)
                n_electrons = beam.n_electrons()

            # Construct the pulse train
            train = PulseTrainParaxialLaser(
                pulse_energy=Quantity(args.pulse_energy, "J"),
                wavelength=Quantity(args.wavelength, "nm"),
                sigma_x=Quantity(args.waist, "um"),
                sigma_y=Quantity(args.waist, "um"),
                subpulse_duration=Quantity(args.subpulse_duration, "fs"),
                repetition_period=Quantity(t_rep_fs, "fs"),
                n_subpulses=args.n_subpulses,
            )

            # Evaluate yield using the stable sub-pulse sum (Method 2)
            subpulses = train.subpulses()
            sub_yields = [
                integrate_trajectories(
                    bunch, sub, n_electrons=n_electrons, n_steps=args.n_steps_subpulse
                ).total_yield()
                for sub in subpulses
            ]
            total_yield = sum(sub_yields)

            ratio_results.append({
                "ratio": r,
                "duty_cycle": d,
                "t_rep_fs": t_rep_fs,
                "t_burst_ps": burst_duration_s * 1e12,
                "sigma_z_um": sigma_z_um,
                "delta_z_coll_um": delta_z_coll_um,
                "delta_z_over_zr": ratio_zr,
                "total_yield": total_yield,
            })
            sys.stdout.write(f"  [D = {d:6.3f}] -> Yield = {total_yield:.4e} (ΔZ/z_R = {ratio_zr:.3f})\n")
            sys.stdout.flush()

        all_results[r] = ratio_results

    return {
        "all_results": all_results,
        "z_r_um": z_r_um,
        "z_r_cm": z_r_cm,
        "args": args,
    }


def save_csv(data: dict[str, object], csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    all_results: dict[float, list[dict[str, float]]] = data["all_results"]  # type: ignore

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ratio_tau_e_over_T_burst",
            "duty_cycle",
            "t_rep_fs",
            "t_burst_ps",
            "sigma_z_um",
            "delta_z_coll_um",
            "delta_z_over_zr",
            "total_yield",
            "normalized_yield",
        ])
        for r, points in all_results.items():
            max_yield = points[-1]["total_yield"] if points[-1]["total_yield"] > 0 else 1.0
            for p in points:
                norm_y = p["total_yield"] / max_yield
                writer.writerow([
                    f"{r:.4f}",
                    f"{p['duty_cycle']:.4f}",
                    f"{p['t_rep_fs']:.2f}",
                    f"{p['t_burst_ps']:.4f}",
                    f"{p['sigma_z_um']:.2f}",
                    f"{p['delta_z_coll_um']:.2f}",
                    f"{p['delta_z_over_zr']:.4f}",
                    f"{p['total_yield']:.6e}",
                    f"{norm_y:.6e}",
                ])
    print(f"\nNumerical data saved to CSV: {csv_path}")


def plot_results(data: dict[str, object], save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not found; skipping plot generation.")
        return

    all_results: dict[float, list[dict[str, float]]] = data["all_results"]  # type: ignore
    z_r_um: float = data["z_r_um"]  # type: ignore
    args: argparse.Namespace = data["args"]  # type: ignore

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Color palette
    colors = plt.cm.plasma(np.linspace(0.08, 0.88, len(all_results)))
    markers = ["o", "s", "^", "D", "v", "p", "h"]

    for idx, (r, points) in enumerate(all_results.items()):
        d_vals = [p["duty_cycle"] for p in points]
        y_vals = [p["total_yield"] for p in points]
        y_max = y_vals[-1] if y_vals[-1] > 0 else 1.0
        y_norm = [y / y_max for y in y_vals]
        sigma_z_um = points[-1]["sigma_z_um"]

        col = colors[idx]
        mark = markers[idx % len(markers)]

        if args.mode == "fixed-bunch":
            label_text = rf"$\tau_e / T_{{\rm burst, 0}} = {r:.1f}$ ($\sigma_z = {sigma_z_um:.0f}\,\mu\mathrm{{m}}$)"
        else:
            label_text = rf"$\tau_e / T_{{\rm burst}} = {r:.1f}$ (const)"

        # Left Panel: Normalized Yield vs Duty Cycle
        ax1.plot(
            d_vals,
            y_norm,
            f"-{mark}",
            color=col,
            label=label_text,
            linewidth=2.0,
            markersize=6,
            alpha=0.92,
        )

        # Right Panel: Absolute Yield vs Duty Cycle
        ax2.plot(
            d_vals,
            y_vals,
            f"-{mark}",
            color=col,
            label=label_text,
            linewidth=2.0,
            markersize=6,
            alpha=0.92,
        )

    # Format Left Panel (Normalized)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Duty Cycle $D = \tau_p / T_{\rm rep}$", fontsize=12)
    ax1.set_ylabel(r"Normalized Photon Yield $N_\gamma(D) / N_\gamma(D=1)$", fontsize=12)
    ax1.set_title(
        r"Relative Yield Retention vs. Duty Cycle",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.9)
    ax1.set_ylim(-0.02, 1.05)

    # Format Right Panel (Absolute)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Duty Cycle $D = \tau_p / T_{\rm rep}$", fontsize=12)
    ax2.set_ylabel(r"Total Scattered Photons $N_\gamma$", fontsize=12)
    ax2.set_title(
        rf"Absolute Yield ($N_p = {args.n_subpulses}$, $E_{{\rm tot}} = {args.pulse_energy:.1f}\,\mathrm{{J}}$, $z_R = {z_r_um:.0f}\,\mu\mathrm{{m}}$)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.9)

    plt.tight_layout()

    # Save both PNG and PDF
    pdf_path = str(Path(save_path).with_suffix(".pdf"))
    plt.savefig(save_path, dpi=200)
    plt.savefig(pdf_path)
    print(f"\nPlots successfully saved to:\n  - PNG: {save_path}\n  - PDF: {pdf_path}")


def main() -> None:
    args = parse_args()
    data = run_multi_ratio_sweep(args)
    save_csv(data, args.output_csv)
    plot_results(data, args.output)


if __name__ == "__main__":
    main()
