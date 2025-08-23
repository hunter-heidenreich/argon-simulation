#!/usr/bin/env python3
"""
Generate Figure 1b: Velocity Distribution Analysis

This script reproduces the velocity distribution analysis from Rahman's 1964 paper,
showing the Maxwell-Boltzmann velocity distribution in reduced units.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim import (
    get_velocities,
    setup_logging_and_artifacts,
    setup_plotting_style,
    apply_rahman_style,
    save_figure
)

# Set up logging and output directory
logger = setup_logging_and_artifacts("figure_1b")


def main():
    """Generate velocity distribution plot with detailed analysis."""

    # --- 1. Load Data & Flatten Components ---
    # Load velocities in native LAMMPS units (Ångstroms/femtosecond)
    logger.info("Loading velocity data...")
    vs_A_fs = get_velocities("traj.lammpstrj")
    logger.info(f"Loaded velocity data for {len(vs_A_fs)} frames")

    # For the distribution, we need every vx, vy, and vz component from every
    # atom across all frames in a single 1D array.
    all_velocity_components_A_fs = vs_A_fs.flatten()

    # --- 2. Correctly Convert Velocities to Native Units of (ε/M)^1/2 ---

    # Define physical constants for the conversion
    EPSILON_KCAL_MOL = 0.23846  # epsilon/kB = 120 K, in kcal/mol
    MASS_G_MOL = 39.95  # Molar mass of Argon in g/mol
    NA = 6.02214e23  # Avogadro's number
    KCAL_TO_J = 4184  # Joules per kcal

    # Calculate epsilon (ε) for a single atom in Joules
    epsilon_J = (EPSILON_KCAL_MOL * KCAL_TO_J) / NA
    # Calculate mass (M) for a single atom in kg
    mass_kg = (MASS_G_MOL / 1000) / NA

    # A) Calculate the value of one native velocity unit in m/s
    unit_velocity_m_s = np.sqrt(epsilon_J / mass_kg)
    # B) Define the value of our data's unit (Å/fs) in m/s
    A_fs_to_m_s = 1.0e5

    # C) The conversion factor is the ratio of these two values
    # This tells us how many native units are in one Å/fs
    conversion_factor = A_fs_to_m_s / unit_velocity_m_s

    # D) Apply the conversion by MULTIPLYING
    velocities_native_units = all_velocity_components_A_fs * conversion_factor

    # --- 3. Get Histogram Data ---
    # Use NumPy to get the raw histogram data (counts and bin edges)
    counts, bin_edges = np.histogram(velocities_native_units, bins=200, density=False)
    # Normalize the counts manually to match the paper's approximate scale
    counts = counts / np.max(counts) * 15.5
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 4. Plot the Distribution Curve ---
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(bin_centers, counts, color="black", linewidth=1.0)

    # --- 5. Validate and Annotate with Distribution Widths ---
    max_height = np.max(counts)
    peak_index = np.argmax(counts)

    heights = {
        "e^-1/2": max_height * np.exp(-0.5),
        "e^-1": max_height * np.exp(-1.0),
        "e^-2": max_height * np.exp(-2.0),
    }
    paper_widths = {"e^-1/2": 1.77, "e^-1": 2.52, "e^-2": 3.52}

    logger.info("\n--- Distribution Width Validation ---")
    logger.info(f"{'Level':<10} | {'Paper Width':<15} | {'Calculated Width':<20}")
    logger.info("-" * 50)

    for name, h in heights.items():
        # Find the left and right crossings using interpolation
        x_left = np.interp(h, counts[:peak_index], bin_centers[:peak_index])
        x_right = np.interp(
            h, counts[peak_index:][::-1], bin_centers[peak_index:][::-1]
        )
        width = x_right - x_left

        ax.hlines(
            y=h, xmin=x_left, xmax=x_right, color="#333333", linestyle="-", linewidth=1
        )
        ax.text(
            x_right + 0.05,
            h,
            f"{width:.2f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        logger.info(f"{name:<10} | {paper_widths[name]:<15.2f} | {width:<20.2f}")

    # --- 6. Final Formatting ---
    apply_rahman_style(
        ax,
        xlabel="Velocity in ($\\varepsilon$/M)$^{1/2}$",
        ylabel="Velocity Distribution",
        title="Velocity Distribution - Rahman 1964"
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 17)

    # Save and display
    output_file = "artifacts/figure_1b.png"
    save_figure(output_file)
    logger.info(f"Figure saved to {output_file}")

    # Display final statistics
    logger.info("\nFinal Statistics:")
    logger.info(f"  Total velocity components: {len(all_velocity_components_A_fs)}")
    logger.info(f"  Peak distribution value: {max_height:.2f}")
    logger.info(f"  Distribution center: {bin_centers[peak_index]:.3f} (ε/M)^1/2")


if __name__ == "__main__":
    main()
