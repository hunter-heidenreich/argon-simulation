# figure_3.py
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_trajectory_data, compute_msd, CONSTANTS

Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_3.log")],
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# All constants now centralized in utils.py

# --- Load Data ---
logger.info("Loading trajectory data...")
xs, _, _ = load_trajectory_data("traj.lammpstrj")
logger.info(f"Loaded {len(xs)} frames")

# --- Calculate MSD using cached function ---
msd = compute_msd("traj.lammpstrj")

# --- Calculate diffusion coefficient ---
n_frames = xs.shape[0]
time_ps = np.arange(n_frames) * CONSTANTS["TIMESTEP_FS"] / 1000.0

# Perform a linear fit on the diffusive region
fit_start_index = n_frames // 2
slope, intercept = np.polyfit(time_ps[fit_start_index:], msd[fit_start_index:], 1)

# Calculate the Diffusion Coefficient (D)
conversion_factor = 1e-4  # from Å²/ps to cm²/s
D = slope / 6.0 * conversion_factor

logger.info(f"\nCalculated Diffusion Coefficient (D): {D:.2e} cm²/s")
logger.info("Paper's value: 2.43e-05 cm²/s")

# --- Calculate diffusion coefficient for early time (1-3 ps) ---
# Find indices for 1-3 ps range
time_1ps_index = np.where(time_ps >= 1.0)[0][0] if np.any(time_ps >= 1.0) else 0
time_3ps_index = np.where(time_ps <= 3.0)[0][-1] if np.any(time_ps <= 3.0) else len(time_ps) - 1

if time_3ps_index > time_1ps_index:
    early_time = time_ps[time_1ps_index:time_3ps_index+1]
    early_msd = msd[time_1ps_index:time_3ps_index+1]
    early_slope, early_intercept = np.polyfit(early_time, early_msd, 1)
    D_early = early_slope / 6.0 * conversion_factor
    logger.info(f"Early time (1-3 ps) Diffusion Coefficient (D): {D_early:.2e} cm²/s")
else:
    D_early = None
    logger.warning("Not enough data points in 1-3 ps range for early time fit")

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Main panel - full time range
ax1.plot(time_ps, msd, "o", markersize=4, color="#2E86AB", alpha=0.7, label="Calculated MSD")
fit_time = time_ps[fit_start_index:]
fit_msd = slope * fit_time + intercept
ax1.plot(fit_time, fit_msd, "-", linewidth=2, color="#A23B72", label="Linear Fit")

ax1.set_xlabel("Time (ps)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Mean Square Displacement ($\\AA^2$)", fontsize=14, fontweight="bold")
ax1.set_title("Mean Square Displacement of Liquid Argon", fontsize=16, fontweight="bold", pad=20)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.tick_params(axis="both", which="major", labelsize=12)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.grid(True, alpha=0.3)

ax1.annotate(
    f"D = {D:.2e}" + " cm$^2$ s$^{-1}$",
    xy=(time_ps[-1] * 0.85, msd[-1] * 0.9),
    xytext=(time_ps[-1] * 0.1, msd[-1] * 0.95),
    arrowprops=dict(facecolor="black", arrowstyle="->", connectionstyle="arc3"),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1),
)

ax1.legend(fontsize=12, framealpha=0.9, edgecolor="gray")

# Second panel - early time (0-3 ps)
time_3ps_cutoff = np.where(time_ps <= 3.0)[0][-1] if np.any(time_ps <= 3.0) else len(time_ps) - 1
early_time_range = time_ps[:time_3ps_cutoff+1]
early_msd_range = msd[:time_3ps_cutoff+1]

ax2.plot(early_time_range, early_msd_range, "o", markersize=6, color="#2E86AB", alpha=0.7, label="Calculated MSD")

# Add early time fit if available
if D_early is not None:
    early_fit_msd = early_slope * early_time + early_intercept
    ax2.plot(early_time, early_fit_msd, "-", linewidth=3, color="#F18F01", label="Linear Fit (1-3 ps)")

ax2.set_xlabel("Time (ps)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Mean Square Displacement ($\\AA^2$)", fontsize=14, fontweight="bold")
ax2.set_title("Early Time MSD (0-3 ps)", fontsize=16, fontweight="bold", pad=20)
ax2.set_xlim(left=0, right=3.0)
ax2.set_ylim(bottom=0)
ax2.tick_params(axis="both", which="major", labelsize=12)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.grid(True, alpha=0.3)

if D_early is not None:
    ax2.annotate(
        f"D = {D_early:.2e} " + "cm$^2$ s$^{-1}$",
        xy=(2.5, early_msd_range[-1] * 0.85),
        xytext=(0.5, early_msd_range[-1] * 0.9),
        arrowprops=dict(facecolor="black", arrowstyle="->", connectionstyle="arc3"),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1),
    )

ax2.legend(fontsize=12, framealpha=0.9, edgecolor="gray")

plt.tight_layout()
plt.savefig("artifacts/figure_3.png", dpi=300, bbox_inches="tight")
plt.show()
