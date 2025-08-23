# figure_7.py
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import compute_displacement_moments, CONSTANTS

# --- Setup ---
Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_7.log")],
)
logger = logging.getLogger(__name__)

# --- Data Calculation ---
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
alpha_2 = (r4 / ( (5/3) * r2**2)) - 1
alpha_3 = (r6 / ( (35/9) * r2**3)) - 1
alpha_4 = (r8 / ( (35/3) * r2**4)) - 1


# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(time_ps, alpha_2, "-", label=r"$\alpha_2(t)$")
ax.plot(time_ps, alpha_3, "-", label=r"$\alpha_3(t)$")
ax.plot(time_ps, alpha_4, "-", label=r"$\alpha_4(t)$")

# --- Formatting ---
ax.set_title("Non-Gaussian Behavior of $G_s(r, t)$", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Time in $10^{-12}$ sec", fontsize=14, fontweight="bold")
ax.set_ylabel("Non-Gaussian Behavior", fontsize=14, fontweight="bold")
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.0)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig("artifacts/figure_7.png", dpi=300, bbox_inches="tight")
plt.show()