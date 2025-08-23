#!/usr/bin/env python3
"""
Generate Table 1: Temperature Statistics Summary

This script computes cumulative temperature statistics from the trajectory,
showing convergence of the ensemble average temperature and RMS fluctuations.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim import (
    CONSTANTS,
    compute_temperatures,
    setup_logging_and_artifacts
)

# Set up logging and output directory
logger = setup_logging_and_artifacts("table_1")


def main():
    """Generate temperature statistics table and analysis."""

    # --- Load and Compute Temperature Statistics ---
    logger.info("Loading and computing temperature data...")
    temps = compute_temperatures("traj.lammpstrj")
    logger.info(f"Loaded temperature data for {len(temps)} frames")

    cumulative_temps = np.cumsum(temps) / np.arange(1, len(temps) + 1)
    cumulative_temp_squared = np.cumsum(temps**2) / np.arange(1, len(temps) + 1)
    rms_fluctuations = (
        np.sqrt(cumulative_temp_squared - cumulative_temps**2) / cumulative_temps
    )

    # --- Print Results Table ---
    logger.info(f"{'Frame':>6} | {'Cumulative Temp (K)':>20} | {'RMS Fluctuation':>16}")
    logger.info("-" * 50)

    for i, temp in enumerate(cumulative_temps):
        if (i + 1) % 100 == 0:
            logger.info(f"{i + 1:>6} | {temp:>20.2f} | {rms_fluctuations[i]:>16.2f}")

    # --- Summary Statistics ---
    target_temp = CONSTANTS["TEMPERATURE"]
    logger.info("\n--- Summary Statistics ---")
    logger.info(f"Final average temperature: {cumulative_temps[-1]:.2f} K")
    logger.info(f"Final RMS fluctuation: {rms_fluctuations[-1]:.4f}")
    logger.info(f"Target temperature: {target_temp} K")
    logger.info(f"Temperature deviation: {abs(cumulative_temps[-1] - target_temp):.2f} K")

    # Additional detailed statistics
    logger.info("\n--- Detailed Analysis ---")
    logger.info(f"Initial temperature: {temps[0]:.2f} K")
    logger.info(f"Minimum temperature: {np.min(temps):.2f} K")
    logger.info(f"Maximum temperature: {np.max(temps):.2f} K")
    logger.info(f"Standard deviation: {np.std(temps):.2f} K")
    logger.info(f"Total frames analyzed: {len(temps)}")

    # Convergence analysis
    convergence_threshold = 0.01  # 1% change threshold
    for i in range(100, len(cumulative_temps)):
        if i > 100:
            relative_change = (
                abs(cumulative_temps[i] - cumulative_temps[i - 100])
                / cumulative_temps[i - 100]
            )
            if relative_change < convergence_threshold:
                logger.info(
                    f"Temperature converged (< {convergence_threshold * 100}% change) after frame {i}"
                )
                break

    logger.info("Table 1 analysis complete.")


if __name__ == "__main__":
    main()
