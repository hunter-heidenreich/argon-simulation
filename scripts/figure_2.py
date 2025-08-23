#!/usr/bin/env python3
"""
Generate Figure 2: Radial Distribution Function g(r) and Static Structure Factor S(k)

This script reproduces the pair correlation function and static structure factor
analysis from Rahman's 1964 paper, showing the structural properties of liquid argon.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim import (
    CONSTANTS,
    compute_radial_distribution,
    compute_structure_factor,
    setup_logging_and_artifacts,
    setup_plotting_style,
    apply_rahman_style,
    save_figure
)

# Set up logging and output directory
logger = setup_logging_and_artifacts("figure_2")


def main():
    """Generate radial distribution function and static structure factor plots."""

    # Calculate g(r) using the built-in function
    logger.info("Calculating pair correlation function g(r)...")
    r, g_r, density = compute_radial_distribution("traj.lammpstrj", dr=0.05)

    # Calculate S(k) from g(r) using the new module function
    logger.info("Calculating static structure factor S(k)...")
    k, s_k = compute_structure_factor(r, g_r, density)

    # --- Plotting ---
    setup_plotting_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: g(r)
    ax1.plot(r / CONSTANTS["SIGMA"], g_r, "o", markersize=3, color="black", markerfacecolor="none")
    ax1.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    apply_rahman_style(
        ax1,
        xlabel="r / $\\sigma$",
        ylabel="g(r)",
        title="Radial Distribution Function g(r) - Rahman 1964"
    )
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 3.1)

    # Find and annotate peaks in g(r)
    gr_peaks_indices, properties = find_peaks(
        g_r,
        height=1,
        prominence=0.2,
        distance=16,  # Approximately 0.8 Angstroms / 0.05 dr
    )

    logger.info("\n--- g(r) Analysis ---")
    logger.info("Expected peak locations from paper: r = 3.7, 7.0, 10.4 Å")
    logger.info(f"Calculated peak locations: r = {np.round(r[gr_peaks_indices], 2)} Å")
    logger.info(f"Peak heights: {np.round(g_r[gr_peaks_indices], 2)}")
    logger.info(f"Peak prominences: {np.round(properties['prominences'], 2)}")

    # Annotate the first 3 peaks found
    for i in range(min(3, len(gr_peaks_indices))):
        peak_idx = gr_peaks_indices[i]
        r_val = r[peak_idx]
        gr_val = g_r[peak_idx]
        ax1.annotate(
            f"{r_val:.1f} Å",
            xy=(r_val / CONSTANTS["SIGMA"], gr_val),
            xytext=(r_val / CONSTANTS["SIGMA"], gr_val + 0.3),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
            ha="center",
            fontsize=12,
        )

    # Plot 2: S(k)
    ax2.plot(k * CONSTANTS["SIGMA"], s_k, color="#2E86AB", linewidth=2)
    apply_rahman_style(
        ax2,
        xlabel="k$\\sigma$",
        ylabel="S(k)",
        title="Static Structure Factor S(k) - Rahman 1964"
    )
    ax2.set_xlim(0, 30)
    ax2.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)

    # Find and annotate peaks in S(k)
    sk_peaks_indices, sk_properties = find_peaks(
        s_k,
        height=1,  # Higher threshold for S(k)
        distance=5,  # Minimum separation in data points
        prominence=0.05,  # Peak prominence
    )
    peak_k_sigma = k[sk_peaks_indices] * CONSTANTS["SIGMA"]
    logger.info("\n--- S(k) Analysis ---")
    logger.info("Expected peak locations from paper: kσ = 6.8, 12.5, 18.5, 24.8")
    logger.info(f"Calculated peak locations: kσ = {np.round(peak_k_sigma, 2)}")
    logger.info(f"Peak heights: {np.round(s_k[sk_peaks_indices], 2)}")
    logger.info(f"Peak prominences: {np.round(sk_properties['prominences'], 2)}")

    # Annotate peaks on the plot
    for i in range(min(4, len(sk_peaks_indices))):
        peak_idx = sk_peaks_indices[i]
        k_val = k[peak_idx] * CONSTANTS["SIGMA"]
        sk_val = s_k[peak_idx]
        ax2.annotate(
            f"{k_val:.1f}" + " Å$^{-1}$",
            xy=(k_val, sk_val),
            xytext=(k_val, sk_val + 0.1),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
            ha="center",
            fontsize=12,
        )

    # Save and display
    output_file = "artifacts/figure_2.png"
    save_figure(output_file)
    logger.info(f"Figure saved to {output_file}")

    # Display final statistics
    logger.info("\nFinal Statistics:")
    logger.info(f"  Density: {density:.6f} atoms/Å³")
    logger.info(f"  Maximum g(r) value: {np.max(g_r):.3f}")
    logger.info(f"  g(r) peaks found: {len(gr_peaks_indices)}")
    logger.info(f"  S(k) peaks found: {len(sk_peaks_indices)}")
    logger.info(f"  Maximum S(k) value: {np.max(s_k):.3f}")


if __name__ == "__main__":
    main()
