#!/usr/bin/env python3
"""
Generate Figure 4: Velocity Autocorrelation Function (VACF)

This script reproduces the velocity autocorrelation function analysis from Rahman's 1964 paper,
comparing the calculated VACF with Langevin theory predictions for liquid argon.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim import (
    CONSTANTS, 
    load_trajectory_data, 
    compute_vacf,
    setup_logging_and_artifacts,
    setup_plotting_style,
    apply_rahman_style,
    save_figure
)
from argon_sim.dynamics import langevin_decay_vacf

# Set up logging and output directory
logger = setup_logging_and_artifacts("figure_4")


def main():
    """Generate velocity autocorrelation function analysis and comparison with theory."""

    # --- Data Loading and Primary Calculation ---
    logger.info("Loading trajectory data...")
    _, vs, _ = load_trajectory_data("traj.lammpstrj")
    if vs.size == 0:
        logger.error("Could not load velocity data. Aborting.")
        return

    logger.info(f"Loaded velocity data for {len(vs)} frames")

    logger.info("Computing velocity autocorrelation function...")
    vacf_normalized = compute_vacf("traj.lammpstrj")

    n_frames = vs.shape[0]
    time_ps = np.arange(n_frames) * CONSTANTS["TIMESTEP_FS"] / 1000.0

    # Calculate Langevin theory comparison
    logger.info("Computing Langevin theory prediction...")
    langevin_vacf_vals = langevin_decay_vacf(
        time_ps, CONSTANTS["TEMPERATURE"], CONSTANTS["MASS_AMU"], CONSTANTS["D_PAPER"]
    )

    # --- Plotting ---
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot calculated VACF
    ax.plot(
        time_ps,
        vacf_normalized,
        "-",
        color="#2E86AB",
        linewidth=2,
        label="Calculated VACF",
    )

    # Plot Langevin theory points
    point_indices = np.linspace(0, len(time_ps) // 3, 10, dtype=int)
    ax.plot(
        time_ps[point_indices],
        langevin_vacf_vals[point_indices],
        "o",
        color="black",
        markersize=6,
        label="Langevin Theory",
    )

    # Formatting
    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    apply_rahman_style(
        ax,
        xlabel="Time (ps)",
        ylabel="Velocity Autocorrelation",
        title="Velocity Autocorrelation Function - Rahman 1964"
    )
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=12, framealpha=0.9, edgecolor="gray")

    # Save and display
    output_file = "artifacts/figure_4.png"
    save_figure(output_file)
    logger.info(f"Figure saved to {output_file}")

    # Display final statistics
    logger.info("\nFinal Statistics:")
    logger.info(f"  VACF initial value: {vacf_normalized[0]:.3f}")
    logger.info(
        f"  VACF at 1 ps: {vacf_normalized[int(1000 / CONSTANTS['TIMESTEP_FS'])]:.3f}"
    )
    logger.info(f"  VACF minimum value: {np.min(vacf_normalized):.3f}")
    logger.info(
        f"  Langevin theory at 1 ps: {langevin_vacf_vals[int(1000 / CONSTANTS['TIMESTEP_FS'])]:.3f}"
    )

    # Find when VACF crosses zero
    zero_crossings = np.where(np.diff(np.signbit(vacf_normalized)))[0]
    if len(zero_crossings) > 0:
        first_zero_time = time_ps[zero_crossings[0]]
        logger.info(f"  First zero crossing at: {first_zero_time:.3f} ps")


if __name__ == "__main__":
    main()
