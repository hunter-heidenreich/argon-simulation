# figure_5.py
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

from utils import load_trajectory_data, compute_vacf, langevin_spectrum, CONSTANTS

Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_5.log")],
)
logger = logging.getLogger(__name__)

# --- Data Loading and Primary Calculation ---
logger.info("Loading trajectory data...")
_, vs, _ = load_trajectory_data("traj.lammpstrj")
if vs.size == 0:
    logger.error("Could not load velocity data. Aborting.")
    exit()

logger.info(f"Loaded velocity data for {len(vs)} frames")
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
beta = (CONSTANTS["HBAR_SI"] * omega) / (CONSTANTS["KB_SI"] * CONSTANTS["TEMPERATURE"])
logger.info("Spectrum calculation complete.")

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot calculated spectrum
langevin_spec_vals = langevin_spectrum(beta, CONSTANTS["MASS_AMU"], CONSTANTS["D_PAPER"])
ax.plot(beta, spectrum_normalized, ".-", color="#2E86AB", markersize=4, linewidth=2, label="Calculated Spectrum")
ax.plot(beta, langevin_spec_vals, "-", color="black", linewidth=1.5, label="Langevin Theory")

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
ax.set_title("Spectrum of the VACF", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel(r"$\beta = \hbar \omega / k_B T$", fontsize=16, fontweight="bold")
ax.set_ylabel(r"$f(\beta)$", fontsize=16, fontweight="bold")
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.5)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, framealpha=0.9, edgecolor="gray")

plt.tight_layout()
plt.savefig("artifacts/figure_5.png", dpi=300, bbox_inches="tight")
plt.show()
