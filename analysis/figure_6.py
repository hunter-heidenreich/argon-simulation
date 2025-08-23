# figure_6.py
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from utils import compute_gdt

# --- Setup ---
Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_6.log")],
)
logger = logging.getLogger(__name__)

# Constants
SIGMA = 3.4  # Lennard-Jones sigma for Argon in Angstroms

# --- Data Calculation ---
# Define the time points from Rahman's paper (Fig. 6) in picoseconds
time_points_ps = [1.0, 2.5]

logger.info("Calculating time-dependent pair correlation function G_d(r, t)...")
# This will be slow the first time, but cached for subsequent runs.
gdt_data = compute_gdt("traj.lammpstrj", t_lags_ps=time_points_ps)
logger.info("G_d(r, t) calculation complete.")

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 12), sharex=True, sharey=True
)

# Subplot (a): t = 1.0 ps
if 1.0 in gdt_data:
    r_values_a, gdt_a = gdt_data[1.0]
    ax1.plot(
        r_values_a / SIGMA,
        gdt_a,
        ".",
        markersize=5,
        color="black",
        label=f"t = {1.0} ps",
    )
    ax1.text(0.9, 0.9, "(a)", transform=ax1.transAxes, fontsize=14, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=12)

# Subplot (b): t = 2.5 ps
if 2.5 in gdt_data:
    r_values_b, gdt_b = gdt_data[2.5]
    ax2.plot(
        r_values_b / SIGMA,
        gdt_b,
        ".",
        markersize=5,
        color="black",
        label=f"t = {2.5} ps",
    )
    ax2.text(0.9, 0.9, "(b)", transform=ax2.transAxes, fontsize=14, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=12)

# --- Formatting ---
fig.suptitle(
    "Time-Dependent Pair-Correlation Function $G_d(r, t)$",
    fontsize=16,
    fontweight="bold",
)

for ax in [ax1, ax2]:
    ax.axhline(1.0, color="gray", linewidth=1.5, linestyle="-")
    ax.set_ylabel("$G_d(r, t)$", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

ax2.set_xlabel("r / $\\sigma$", fontsize=14, fontweight="bold")
ax1.set_xlim(0, 3.5)
ax1.set_ylim(0, 1.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
plt.savefig("artifacts/figure_6.png", dpi=300, bbox_inches="tight")
plt.show()