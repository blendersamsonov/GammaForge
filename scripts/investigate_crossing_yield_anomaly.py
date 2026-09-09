"""Investigate and explain the crossing angle yield anomaly (N_tgt > N_tot).

Uses the user's exact parameters from `calculate_spectral_angular_table.ipynb`:
- gamma0 = 2000 (1/gamma = 0.5 mrad)
- Target: 1 cm at 10 m -> half-angle 0.5 mrad = 1/gamma
- Crossing angle: 0.5 mrad -> gamma * alpha = 1.0
- Laser: 10 mJ, 1030 nm, sigma = 25 um, duration = 30 ps, psi_pol = 0.0 rad
- Beam: 100 pC, gamma0 * 0.511 MeV, rel_spread = 0.01, sigma_xy = 10 um, emit_xy = 1 mm*mrad/gamma0

This script:
1. Computes the 3x3 table with the exact parameters and prints the yield table.
2. Dissects the mathematical structure of the polarization factor P(theta_x, theta_y).
3. Demonstrates why P exceeds 1.0 (peaking at ~1.96) and integrates to (1 + (gamma*alpha)^2).
4. Generates explanatory plots of the polarization factor and angular distributions.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from gammaforge.engines.xigma.engine import XigmaEngine
from gammaforge.engines.xigma.stages import (
    KERNEL_NORMALIZATION_CONSTANT,
    polarization_factor_vectorized,
)
from gammaforge.io.bunch import GaussianElectronBeam
from gammaforge.io.interaction import SamplingSpec, build_interaction
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.results import Axis
from gammaforge.io.target import OutputKind, OutputRequest, Target
from gammaforge.io.units import EV_CGS, ureg


def run_user_simulation():
    """Run the 3x3 grid simulation with the user's exact parameters."""
    gamma0 = 2000
    beam = GaussianElectronBeam(
        bunch_charge=100.0 * ureg.picocoulomb,
        kinetic_energy=gamma0 * 0.511 * ureg.megaelectronvolt,
        rel_energy_spread=0.01,
        sigma_x=10.0 * ureg.micrometer,
        sigma_y=10.0 * ureg.micrometer,
        emit_x=1.0 * ureg.millimeter * ureg.milliradian / gamma0,
        emit_y=1.0 * ureg.millimeter * ureg.milliradian / gamma0,
        sigma_z=100.0 * ureg.micrometer,
    )

    target_size = 1.0 * ureg.centimeter
    target_distance = 10.0 * ureg.meter

    cone_half_angle_rad = 0.5 * (target_size / target_distance).to("rad").magnitude
    cone_half_angle_mrad = cone_half_angle_rad * 1e3

    energy_res = 128
    angle_res = 32

    target = Target(
        theta_x_col=cone_half_angle_rad * ureg.radian,
        theta_y_col=cone_half_angle_rad * ureg.radian,
        outputs=(
            OutputRequest(
                OutputKind.COLLIMATED_SPECTRUM,
                resolution=(energy_res, angle_res, angle_res),
            ),
            OutputRequest(OutputKind.TOTAL_YIELD),
        ),
    )

    from gammaforge.engines.xigma.spectrum_sampler import is_gpu_available
    backend = "cupy" if is_gpu_available() else "numpy"

    engine = XigmaEngine()
    params = engine.schema.with_values(
        n_bins_gamma=36,
        n_bins_theta_x=64,
        n_bins_theta_y=64,
        n_bins_a0_shape=2,
        n_bins_ahat=2,
        scheme="cic",
        backend=backend,
    )

    sampling = SamplingSpec(n_particles=5_000, seed=42)

    crossing_angle = 0.5 * ureg.mrad
    crossing_cases = [
        ("Head-on", 0.0 * ureg.rad, 0.0 * ureg.rad, r"Head-on ($\alpha = 0$)"),
        ("Cross xz", crossing_angle, 0.0 * ureg.rad, r"Cross $xz$ ($\alpha = 0.5\,$mrad)"),
        ("Cross yz", 0.0 * ureg.rad, crossing_angle, r"Cross $yz$ ($\alpha = 0.5\,$mrad)"),
    ]
    ellipticities = [0.0, 0.5, 1.0]

    print("=" * 80)
    print("RUNNING SIMULATION WITH USER PARAMETERS")
    print(f"gamma0 = {gamma0}, aperture = {cone_half_angle_mrad:.2f} mrad (= 1/gamma)")
    print(f"crossing angle alpha = 0.5 mrad -> gamma * alpha = {gamma0 * 0.0005:.2f}")
    print("=" * 80)

    sim_results = {}

    for row_name, txz, tyz, label in crossing_cases:
        for eps in ellipticities:
            t0 = time.perf_counter()
            laser = GaussianParaxialLaser(
                pulse_energy=10.0 * ureg.millijoule,
                wavelength=1030.0 * ureg.nanometer,
                sigma_x=25.0 * ureg.micrometer,
                sigma_y=25.0 * ureg.micrometer,
                duration=30.0 * ureg.picosecond,
                ellipticity=eps,
                psi_pol=0.0 * ureg.radian,
                theta_xz=txz,
                theta_yz=tyz,
            )

            interaction = build_interaction(beam, laser, target, sampling)
            res = engine.run(interaction, params)
            elapsed = time.perf_counter() - t0

            tot_yield = float(res.photon_slices[OutputKind.TOTAL_YIELD].distr)
            col_slice = res.photon_slices[OutputKind.COLLIMATED_SPECTRUM]

            e_cgs = col_slice.axes[Axis.ENERGY]
            tx_rad = col_slice.axes[Axis.THETA_X]
            ty_rad = col_slice.axes[Axis.THETA_Y]

            cube = col_slice.distr  # (n_E, n_tx, n_ty)
            d2N_dOmega_cgs = np.trapezoid(cube, e_cgs, axis=0)
            dN_dE_cgs = np.trapezoid(np.trapezoid(cube, ty_rad, axis=2), tx_rad, axis=1)
            col_yield = float(np.trapezoid(dN_dE_cgs, e_cgs))
            captured_pct = (col_yield / tot_yield * 100.0) if tot_yield > 0 else 0.0

            key = (row_name, eps)
            sim_results[key] = {
                "tot_yield": tot_yield,
                "col_yield": col_yield,
                "captured_pct": captured_pct,
                "d2N_dOmega_mrad": d2N_dOmega_cgs / 1e6,
                "tx_mrad": tx_rad * 1e3,
                "ty_mrad": ty_rad * 1e3,
                "elapsed": elapsed,
            }

            status = "ANOMALY (>100%)" if captured_pct > 100.0 else "NORMAL"
            print(
                f"{row_name:<10} | eps={eps:.1f} | N_tot={tot_yield:.3e} | "
                f"N_tgt={col_yield:.3e} ({captured_pct:6.2f}%) | {status} ({elapsed:.2f}s)"
            )

    return sim_results


def analyze_mathematical_kernel():
    """Evaluate the theoretical polarization factor and angular integrals."""
    gamma = 2000.0
    alpha = 0.0005  # 0.5 mrad
    gamma_alpha = gamma * alpha  # = 1.0

    print("\n" + "=" * 80)
    print("MATHEMATICAL ANALYSIS OF THE POLARIZATION FACTOR")
    print(f"gamma = {gamma:.1f}, alpha = {alpha*1e3:.2f} mrad, gamma * alpha = {gamma_alpha:.2f}")
    print("=" * 80)

    # 1D slice along theta_x (with theta_y = 0)
    th_x_rad = np.linspace(-1.0e-3, 1.0e-3, 500)
    th_x_mrad = th_x_rad * 1e3
    tilde_th_x = gamma * th_x_rad
    tilde_alpha = gamma * alpha

    # Analytical formula for linear polarization (DER006 / Eq. udef):
    # P = 1 - 4*(tilde_th_x - tilde_alpha)*(tilde_th_x + tilde_alpha*tilde_th_x^2) / (1 + tilde_th_x^2)^2
    pol_headon = 1.0 - 4.0 * (tilde_th_x**2) / (1.0 + tilde_th_x**2) ** 2
    pol_cross_xz = 1.0 - 4.0 * (tilde_th_x - tilde_alpha) * (
        tilde_th_x + tilde_alpha * tilde_th_x**2
    ) / (1.0 + tilde_th_x**2) ** 2

    # Verification using stages.py polarization_factor_vectorized
    pol_code_headon = np.array([
        float(polarization_factor_vectorized(
            gamma, 0.0, 0.0, float(x), 0.0, ellipticity=0.0, psi_pol=0.0, theta_xz=0.0, theta_yz=0.0
        ))
        for x in th_x_rad
    ])
    pol_code_cross = np.array([
        float(polarization_factor_vectorized(
            gamma, 0.0, 0.0, float(x), 0.0, ellipticity=0.0, psi_pol=0.0, theta_xz=alpha, theta_yz=0.0
        ))
        for x in th_x_rad
    ])

    diff_headon = np.max(np.abs(pol_headon - pol_code_headon))
    diff_cross = np.max(np.abs(pol_cross_xz - pol_code_cross))
    print(f"Discrepancy between formula and stages.py code: head-on={diff_headon:.2e}, cross={diff_cross:.2e}")

    # Maximum of P
    max_idx = np.argmax(pol_cross_xz)
    th_max_mrad = th_x_mrad[max_idx]
    pol_max = pol_cross_xz[max_idx]
    print(f"Peak polarization factor: P_max = {pol_max:.4f} at theta_x = {th_max_mrad:.3f} mrad")
    print(f"Polarization factor at on-axis theta_x = 0: P(0) = {pol_cross_xz[len(th_x_rad)//2]:.4f}")

    # Numerical integral over full solid angle for various gamma * alpha values
    print("\nAngular Integral of [gamma^2 / (1 + gamma^2 * theta^2)^2 * P(theta)] over all solid angle:")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'gamma * alpha':<15} | {'Numerical Integral':<20} | {'Theoretical 2pi/3 * (1 + (ga)^2)':<30} | {'Ratio':<10}")
    print("-" * 80)

    u_grid = np.linspace(0, 1000.0, 100000)  # u = gamma^2 * theta^2
    test_gas = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    integral_results = []

    for ga in test_gas:
        # P_avg(u) = 1 - 2*u / (1+u)^2 + 4*ga^2 * u / (1+u)^2
        p_avg = 1.0 - 2.0 * u_grid / (1.0 + u_grid) ** 2 + 4.0 * (ga**2) * u_grid / (1.0 + u_grid) ** 2
        integrand = np.pi * p_avg / (1.0 + u_grid) ** 2
        num_int = float(np.trapezoid(integrand, u_grid))
        theo_int = (2.0 * np.pi / 3.0) * (1.0 + ga**2)
        ratio = num_int / theo_int
        integral_results.append((ga, num_int, theo_int, ratio))
        print(f"{ga:<15.2f} | {num_int:<20.4f} | {theo_int:<30.4f} | {ratio:<10.4f}")

    return {
        "th_x_mrad": th_x_mrad,
        "pol_headon": pol_headon,
        "pol_cross_xz": pol_cross_xz,
        "pol_max": pol_max,
        "th_max_mrad": th_max_mrad,
        "integral_results": integral_results,
    }


def generate_figures(sim_results, math_results, output_dir: Path):
    """Generate diagnostic figures illustrating the findings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = []

    # Figure 1: Polarization factor 1D profile and mechanism
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    th_x = math_results["th_x_mrad"]
    pol_headon = math_results["pol_headon"]
    pol_cross = math_results["pol_cross_xz"]

    ax1.plot(th_x, pol_headon, "k--", label=r"Head-on ($\gamma\alpha = 0$)")
    ax1.plot(th_x, pol_cross, "r-", lw=2, label=r"Cross $xz$ ($\gamma\alpha = 1.0$)")
    ax1.axhline(1.0, color="gray", linestyle=":", label="Physical upper bound for dipole (1.0)")
    ax1.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(0.5, color="blue", linestyle=":", label=r"Collimator edge ($\theta_x = +0.5\,$mrad)")
    ax1.axvline(-0.5, color="blue", linestyle=":", label=r"Collimator edge ($\theta_x = -0.5\,$mrad)")

    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-0.1, 2.2)
    ax1.set_xlabel(r"Observation angle $\theta_x$ [mrad]", fontsize=12)
    ax1.set_ylabel(r"Polarization factor $\mathcal{P}(\theta_x, 0)$", fontsize=12)
    ax1.set_title(r"Polarization Factor $\mathcal{P}$ across Collimator ($\gamma=2000, \alpha=0.5\,$mrad)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotate peak
    peak_x = math_results["th_max_mrad"]
    peak_y = math_results["pol_max"]
    ax1.annotate(
        f"Peak $\\mathcal{{P}} = {peak_y:.2f}$\n(Inverted interference)",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 0.15, peak_y - 0.2),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=10,
        fontweight="bold",
        color="red",
    )

    # Plot 2: Total angular integral vs (gamma * alpha)
    gas = [r[0] for r in math_results["integral_results"]]
    num_ints = [r[1] for r in math_results["integral_results"]]
    theo_ints = [r[2] for r in math_results["integral_results"]]

    ax2.plot(gas, num_ints, "ro", label=r"Numerical $\int d\Omega\, \frac{d\sigma}{d\Omega}$")
    ax2.plot(gas, theo_ints, "b-", lw=2, label=r"Analytical $\sigma_T \cdot (1 + (\gamma\alpha)^2)$")
    ax2.axhline(2.0944, color="k", linestyle="--", label=r"Head-on Total $\sigma_T$ ($2\pi/3$)")
    ax2.axvline(1.0, color="green", linestyle=":", label=r"User case: $\gamma\alpha = 1.0$ ($2\times \sigma_T$)")

    ax2.set_xlabel(r"Scaled crossing angle $\gamma\alpha$", fontsize=12)
    ax2.set_ylabel(r"Angle-integrated cross-section [relative to $\sigma_T$]", fontsize=12)
    ax2.set_title(r"Inflation of Total Integrated Cross-Section by $(\gamma\alpha)^2$", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p1 = output_dir / "crossing_anomaly_polarization_mechanism.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    fig_paths.append(p1)

    # Figure 2: 2D Angular Distributions on Target Detector
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    keys = [
        (("Head-on", 0.0), r"Head-on ($\varepsilon = 0.0$)", axes[0]),
        (("Cross xz", 0.0), r"Cross $xz$ ($\varepsilon = 0.0$, $\alpha = 0.5\,$mrad)", axes[1]),
        (("Cross yz", 0.0), r"Cross $yz$ ($\varepsilon = 0.0$, $\alpha = 0.5\,$mrad)", axes[2]),
    ]

    vmax = max(np.max(sim_results[k]["d2N_dOmega_mrad"]) for k, _, _ in keys)

    for (k, title, ax) in keys:
        data = sim_results[k]
        tx = data["tx_mrad"]
        ty = data["ty_mrad"]
        distr = data["d2N_dOmega_mrad"]
        im = ax.imshow(
            distr.T,
            extent=[tx[0], tx[-1], ty[0], ty[-1]],
            origin="lower",
            cmap="inferno",
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(
            f"{title}\n$N_{{tgt}} = {data['col_yield']:.2e}$ ({data['captured_pct']:.1f}%)",
            fontsize=11,
        )
        ax.set_xlabel(r"$\theta_x$ [mrad]", fontsize=11)
        ax.set_ylabel(r"$\theta_y$ [mrad]", fontsize=11)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.03)
    cbar.set_label(r"$d^2N / d\Omega$ [photons / mrad$^2$]", fontsize=11)

    p2 = output_dir / "crossing_anomaly_angular_distributions.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    fig_paths.append(p2)

    print(f"\nSaved diagnostic figures:")
    for p in fig_paths:
        print(f" - {p}")

    return fig_paths


def main():
    sim_results = run_user_simulation()
    math_results = analyze_mathematical_kernel()
    output_dir = Path("output/crossing_yield_investigation")
    fig_paths = generate_figures(sim_results, math_results, output_dir)
    print("\nInvestigation complete.")


if __name__ == "__main__":
    main()
