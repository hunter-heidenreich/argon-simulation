#!/usr/bin/env python3
"""
Generate Figure 1a: Temperature vs Time Analysis

This script reproduces the temperature analysis from Rahman's 1964 paper,
showing the instantaneous temperature evolution over the simulation trajectory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim import (
    CONSTANTS, 
    compute_temperatures, 
    setup_logging_and_artifacts,
    setup_plotting_style,
    apply_rahman_style,
    save_figure
)

# Set up logging and output directory
logger = setup_logging_and_artifacts("figure_1a")


def main():
    """Generate temperature vs time plot with detailed insets."""

    # Compute temperatures from trajectory
    logger.info("Computing temperatures from trajectory...")
    temperatures = compute_temperatures("traj.lammpstrj")
    n_frames = len(temperatures)
    logger.info(f"Processed {n_frames} frames")

    # Create time axis
    time_ps = np.arange(n_frames) * CONSTANTS["TIMESTEP_FS"] / 1000.0

    # Create the main plot
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    # Main temperature plot
    ax.plot(
        time_ps,
        temperatures,
        color="#2E86AB",
        linewidth=1.5,
        alpha=0.8,
        label="Temperature",
    )

    # Add target temperature line
    target_temp = CONSTANTS["TEMPERATURE"]
    ax.axhline(
        target_temp,
        color="gray",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label="Target Temperature",
    )

    # Create first inset (1.0-2.0 ps region)
    inset1 = inset_axes(
        ax,
        width="25%",
        height="30%",
        loc="upper left",
        bbox_to_anchor=(0.14, 0.35, 0.65, 0.65),
        bbox_transform=ax.transAxes,
    )
    mask1 = (time_ps >= 1.0) & (time_ps <= 2.0)
    inset1.plot(time_ps[mask1], temperatures[mask1], color="#A23B72", linewidth=2)
    inset1.axhline(target_temp, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    inset1.set_xlim(1.0, 2.0)
    inset1.set_ylim(temperatures[mask1].min() - 0.5, temperatures[mask1].max() + 0.5)
    inset1.set_title("1.0-2.0 ps Detail", fontsize=9, pad=3, fontweight="bold")
    inset1.tick_params(labelsize=7)
    inset1.grid(True, alpha=0.3)
    inset1.set_xlabel("Time (ps)", fontsize=7)
    inset1.set_ylabel("Temperature (K)", fontsize=7)
    for spine in inset1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#A23B72")

    # Create second inset (5.0-6.0 ps region)
    inset2 = inset_axes(
        ax,
        width="25%",
        height="30%",
        loc="upper right",
        bbox_to_anchor=(-0.1, -0.38, 0.65, 0.65),
        bbox_transform=ax.transAxes,
    )
    mask2 = (time_ps >= 5.0) & (time_ps <= 6.0)
    inset2.plot(time_ps[mask2], temperatures[mask2], color="#F18F01", linewidth=2)
    inset2.axhline(target_temp, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    inset2.set_xlim(5.0, 6.0)
    inset2.set_ylim(temperatures[mask2].min() - 0.5, temperatures[mask2].max() + 0.5)
    inset2.set_title("5.0-6.0 ps Detail", fontsize=9, pad=3, fontweight="bold")
    inset2.tick_params(labelsize=7)
    inset2.grid(True, alpha=0.3)
    inset2.set_xlabel("Time (ps)", fontsize=7)
    inset2.set_ylabel("Temperature (K)", fontsize=7)
    for spine in inset2.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#F18F01")

    # Highlight the inset regions on the main plot
    ax.axvspan(1.0, 2.0, alpha=0.1, color="#A23B72")
    ax.axvspan(5.0, 6.0, alpha=0.1, color="#F18F01")

    # Format the main plot
    apply_rahman_style(
        ax,
        xlabel="Time (ps)",
        ylabel="Temperature (K)",
        title="Temperature vs Time - Rahman 1964"
    )

    # Add statistics text box
    stats_text = (
        f"Mean: {temperatures.mean():.1f} K\n"
        f"Min: {temperatures.min():.1f} K\n"
        f"Max: {temperatures.max():.1f} K\n"
        f"Std: {temperatures.std():.1f} K"
    )
    ax.text(
        0.02,
        0.92,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "gray",
        },
    )

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.3)

    # Save and display
    output_file = "artifacts/figure_1a.png"
    save_figure(output_file)
    logger.info(f"Figure saved to {output_file}")

    # Display final statistics
    logger.info("\nFinal Statistics:")
    logger.info(f"  Target temperature: {target_temp:.1f} K")
    logger.info(f"  Mean temperature: {temperatures.mean():.2f} K")
    logger.info(
        f"  Temperature deviation: {abs(temperatures.mean() - target_temp):.2f} K"
    )
    logger.info(f"  Standard deviation: {temperatures.std():.2f} K")


if __name__ == "__main__":
    main()
