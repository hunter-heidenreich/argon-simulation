# figure_4.py
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import load_trajectory_data, compute_vacf, langevin_decay_vacf, CONSTANTS

Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_4.log")],
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
time_ps = np.arange(n_frames) * CONSTANTS["TIMESTEP_FS"] / 1000.0

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot calculated VACF
langevin_vacf_vals = langevin_decay_vacf(time_ps, CONSTANTS["TEMPERATURE"], CONSTANTS["MASS_AMU"], CONSTANTS["D_PAPER"])
ax.plot(time_ps, vacf_normalized, "-", color="#2E86AB", linewidth=2, label="Calculated VACF")

# Plot Langevin theory points
point_indices = np.linspace(0, len(time_ps) // 3, 10, dtype=int)
ax.plot(
    time_ps[point_indices],
    langevin_vacf_vals[point_indices],
    "o",
    color="black",
    markersize=6,
    label="Langevin Theory"
)

# Formatting
ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
ax.set_title("Velocity Autocorrelation Function", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Time (ps)", fontsize=14, fontweight="bold")
ax.set_ylabel("Velocity Autocorrelation", fontsize=14, fontweight="bold")
ax.set_xlim(0, 3)
ax.set_ylim(-0.1, 1.05)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, framealpha=0.9, edgecolor="gray")

plt.tight_layout()
plt.savefig("artifacts/figure_4.png", dpi=300, bbox_inches="tight")
plt.show()
