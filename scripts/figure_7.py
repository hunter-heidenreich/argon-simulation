#!/usr/bin/env python3
"""
Generate Figure 7: Non-Gaussian Behavior of G_s(r, t)

This script computes and plots the non-Gaussian parameters α_n(t) for the
self-part of the Van Hove correlation function, analyzing deviations from
Gaussian displacement distributions.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim import (
    CONSTANTS, 
    compute_displacement_moments,
    setup_logging_and_artifacts,
    setup_plotting_style,
    apply_rahman_style,
    save_figure
)

# Set up logging and output directory
logger = setup_logging_and_artifacts("figure_7")


def main():
    """Generate non-Gaussian behavior analysis plot."""

    # Data Calculation
    logger.info("Computing displacement moments...")
    moments = compute_displacement_moments("traj.lammpstrj")
    n_frames = moments.shape[0]

    # Time axis in picoseconds
    time_ps = np.arange(n_frames) * CONSTANTS["TIMESTEP_FS"] / 1000.0

    # Unpack moments
    r2 = moments[:, 0]
    r4 = moments[:, 1]
    r6 = moments[:, 2]
    r8 = moments[:, 3]

    # Avoid division by zero for the first frame
    r2[0] = 1.0

    # Calculate alpha_n(t) parameters
    # C_n = (2n+1)!! / 3^n, so C_2 = 5/3, C_3=35/9, C_4=315/27 = 35/3
    alpha_2 = (r4 / ((5 / 3) * r2**2)) - 1
    alpha_3 = (r6 / ((35 / 9) * r2**3)) - 1
    alpha_4 = (r8 / ((35 / 3) * r2**4)) - 1

    # Plotting
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(time_ps, alpha_2, "-", label=r"$\alpha_2(t)$")
    ax.plot(time_ps, alpha_3, "-", label=r"$\alpha_3(t)$")
    ax.plot(time_ps, alpha_4, "-", label=r"$\alpha_4(t)$")

    # Formatting
    apply_rahman_style(
        ax,
        xlabel="Time (ps)",
        ylabel="Non-Gaussian Parameters",
        title="Non-Gaussian Displacement Parameters - Rahman 1964"
    )
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)

    # Save and display
    output_file = "artifacts/figure_7.png"
    save_figure(output_file)
    logger.info(f"Figure saved to {output_file}")

    # Display statistics
    logger.info("\nNon-Gaussian parameter statistics:")
    logger.info(f"  α₂ max: {np.max(alpha_2):.3f}")
    logger.info(f"  α₃ max: {np.max(alpha_3):.3f}")
    logger.info(f"  α₄ max: {np.max(alpha_4):.3f}")


if __name__ == "__main__":
    main()
