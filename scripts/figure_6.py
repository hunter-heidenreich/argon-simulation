#!/usr/bin/env python3
"""
Generate Figure 6: Van Hove G_d(r,t) Time-Dependent Pair Correlation Function

This script reproduces the Van Hove distinct correlation function analysis from Rahman's 1964 paper,
showing the time evolution of pair correlations and comparison with Vineyard approximation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim import (
    CONSTANTS,
    compute_gdt, 
    compute_gs, 
    compute_radial_distribution,
    setup_logging_and_artifacts,
    setup_plotting_style,
    apply_rahman_style,
    save_figure
)
from argon_sim.structure import fourier_transform_3d, inverse_fourier_transform_3d

# Set up logging and output directory
logger = setup_logging_and_artifacts("figure_6")


def main():
    """Generate Van Hove G_d(r,t) analysis with Vineyard approximation comparison."""

    # --- Data Calculation ---
    time_points_ps = [1.0, 2.5]
    TRAJ_FILE = "traj.lammpstrj"

    logger.info("Calculating G_d(r, t)...")
    gdt_data = compute_gdt(TRAJ_FILE, t_lags_ps=time_points_ps)

    logger.info("Calculating G_s(r, t)...")
    gs_data = compute_gs(TRAJ_FILE, t_lags_ps=time_points_ps)

    logger.info("Calculating g(r)...")
    r_vals, g_r, density = compute_radial_distribution(TRAJ_FILE)
    logger.info(
        f"g(r) data types: r_vals={type(r_vals)}, g_r={type(g_r)}, density={type(density)}"
    )
    logger.info(
        f"g(r) shapes: r_vals={getattr(r_vals, 'shape', 'N/A')}, g_r={getattr(g_r, 'shape', 'N/A')}"
    )

    # Convert to numpy arrays if needed
    if isinstance(r_vals, str) or isinstance(g_r, str):
        logger.error(
            "Received string instead of array from compute_radial_distribution"
        )
        return

    r_vals = np.asarray(r_vals)
    g_r = np.asarray(g_r)
    density = float(density)

    # --- Vineyard Approximation using Fourier Transforms ---
    vineyard_data = {}
    logger.info("Calculating Vineyard Approximation via Fourier Transforms...")

    # Create a k-space grid
    k_vals = np.linspace(0.1, 20.0, 1000)

    # 1. Transform g(r) to get the structure factor S(k)
    s_k_minus_1 = fourier_transform_3d(r_vals, density * (g_r - 1), k_vals)

    for t_ps in time_points_ps:
        if t_ps in gs_data:
            r_gs, gs_t = gs_data[t_ps]

            # 2. Transform G_s(r,t) to get F_s(k,t)
            f_s_k_t = fourier_transform_3d(r_gs, gs_t, k_vals)

            # 3. Multiply in k-space
            i_d_k_t = s_k_minus_1 * f_s_k_t

            # 4. Inverse transform the product to get the distinct part correlation
            rho_gd_minus_1 = inverse_fourier_transform_3d(k_vals, i_d_k_t, r_vals)

            # 5. Reconstruct the final function
            g_vineyard = 1 + rho_gd_minus_1 / density
            vineyard_data[t_ps] = (r_vals, g_vineyard)

    # --- Plotting ---
    setup_plotting_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True)

    # Subplot (a): t = 1.0 ps
    if 1.0 in gdt_data:
        r_a, gdt_a = gdt_data[1.0]
        ax1.plot(
            r_a / CONSTANTS["SIGMA"],
            gdt_a,
            ".",
            markersize=5,
            color="black",
            label=f"$G_d(r, t={1.0}ps)$",
        )
        if 1.0 in vineyard_data:
            r_v, g_v = vineyard_data[1.0]
            ax1.plot(
                r_v / CONSTANTS["SIGMA"],
                g_v,
                "x",
                markersize=5,
                color="black",
                label="Vineyard Approx.",
            )
        ax1.text(
            0.05, 0.9, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold"
        )
        ax1.legend(loc="upper right", fontsize=12)

    # Subplot (b): t = 2.5 ps
    if 2.5 in gdt_data:
        r_b, gdt_b = gdt_data[2.5]
        ax2.plot(
            r_b / CONSTANTS["SIGMA"],
            gdt_b,
            ".",
            markersize=5,
            color="black",
            label=f"$G_d(r, t={2.5}ps)$",
        )
        if 2.5 in vineyard_data:
            r_v, g_v = vineyard_data[2.5]
            ax2.plot(
                r_v / CONSTANTS["SIGMA"],
                g_v,
                "x",
                markersize=5,
                color="black",
                label="Vineyard Approx.",
            )
        ax2.text(
            0.05, 0.9, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold"
        )
        ax2.legend(loc="upper right", fontsize=12)

    # --- Formatting ---
    apply_rahman_style(
        ax1,
        ylabel="$G_d(r, t)$",
        title="Van Hove Distinct Correlation Function - Rahman 1964"
    )
    apply_rahman_style(
        ax2,
        xlabel="r / $\\sigma$",
        ylabel="$G_d(r, t)$"
    )
    
    for ax in [ax1, ax2]:
        ax.axhline(1.0, color="gray", linewidth=1.5, linestyle="-")

    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(0, 1.6)

    # Save and display
    output_file = "artifacts/figure_6.png"
    save_figure(output_file)
    logger.info(f"Figure saved to {output_file}")

    # Display final statistics
    logger.info("\nFinal Statistics:")
    logger.info(f"  Density: {density:.6f} atoms/Å³")
    logger.info(f"  Time points analyzed: {time_points_ps}")

    for t_ps in time_points_ps:
        if t_ps in gdt_data:
            r_gdt, gdt_vals = gdt_data[t_ps]
            max_gdt = np.max(gdt_vals)
            logger.info(f"  G_d(r,t) max at t={t_ps}ps: {max_gdt:.3f}")

        if t_ps in vineyard_data:
            r_vine, vine_vals = vineyard_data[t_ps]
            max_vine = np.max(vine_vals)
            logger.info(f"  Vineyard approx max at t={t_ps}ps: {max_vine:.3f}")


if __name__ == "__main__":
    main()
