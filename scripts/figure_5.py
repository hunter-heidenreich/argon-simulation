#!/usr/bin/env python3
"""
Generate Figure 5: VACF Frequency Spectrum Analysis

This script reproduces the frequency spectrum analysis of the velocity autocorrelation
function from Rahman's 1964 paper, comparing calculated spectrum with Langevin theory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq

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
from argon_sim.dynamics import langevin_spectrum

# Set up logging and output directory
logger = setup_logging_and_artifacts("figure_5")


def main():
    """Generate VACF frequency spectrum analysis and comparison with Langevin theory."""

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

    # --- Spectrum Calculation ---
    logger.info("Calculating spectrum of VACF...")
    # The spectrum is the Fourier Cosine Transform of the VACF.
    # We can get this from the real part of a one-sided Fast Fourier Transform (FFT).
    spectrum_raw = np.real(rfft(vacf_normalized))

    # Normalize the spectrum so its value at frequency=0 is 1.0
    spectrum_normalized = spectrum_raw / spectrum_raw[0]

    # Calculate the dimensionless frequency axis, β = ħω / k_B*T
    timestep_s = CONSTANTS["TIMESTEP_FS"] * 1e-15
    freq_hz = rfftfreq(n_frames, d=timestep_s)  # Frequencies in Hz
    omega = 2 * np.pi * freq_hz  # Angular frequencies in rad/s
    beta = (CONSTANTS["HBAR_SI"] * omega) / (
        CONSTANTS["KB_SI"] * CONSTANTS["TEMPERATURE"]
    )
    logger.info("Spectrum calculation complete.")

    # Calculate Langevin theory comparison
    logger.info("Computing Langevin theory spectrum...")
    langevin_spec_vals = langevin_spectrum(
        beta, CONSTANTS["MASS_AMU"], CONSTANTS["D_PAPER"]
    )

    # --- Plotting ---
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot calculated spectrum
    ax.plot(
        beta,
        spectrum_normalized,
        ".-",
        color="#2E86AB",
        markersize=4,
        linewidth=2,
        label="Calculated Spectrum",
    )
    ax.plot(
        beta,
        langevin_spec_vals,
        "-",
        color="black",
        linewidth=1.5,
        label="Langevin Theory",
    )

    # Add annotations
    ax.annotate(
        "This Calculation",
        xy=(0.3, 1.3),
        xytext=(0.4, 1.4),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        fontsize=12,
    )
    ax.annotate(
        "Langevin Diffusion",
        xy=(0.5, 0.4),
        xytext=(0.1, 0.3),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        fontsize=12,
    )

    # Formatting
    apply_rahman_style(
        ax,
        xlabel=r"$\beta = \hbar \omega / k_B T$",
        ylabel=r"$f(\beta)$",
        title="VACF Frequency Spectrum - Rahman 1964"
    )
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=12, framealpha=0.9, edgecolor="gray")

    # Save and display
    output_file = "artifacts/figure_5.png"
    save_figure(output_file)
    logger.info(f"Figure saved to {output_file}")

    # Display final statistics
    logger.info("\nFinal Statistics:")
    logger.info(f"  Spectrum peak value: {np.max(spectrum_normalized):.3f}")
    logger.info(f"  Spectrum at β=0: {spectrum_normalized[0]:.3f}")
    logger.info(f"  Langevin spectrum at β=0: {langevin_spec_vals[0]:.3f}")
    logger.info(f"  Maximum β value: {np.max(beta):.3f}")
    logger.info(f"  Frequency resolution: {freq_hz[1]:.2e} Hz")

    # Find where spectrum drops to half maximum
    half_max_idx = np.where(spectrum_normalized <= 0.5)[0]
    if len(half_max_idx) > 0:
        beta_half_max = beta[half_max_idx[0]]
        logger.info(f"  β at half maximum: {beta_half_max:.3f}")


if __name__ == "__main__":
    main()
