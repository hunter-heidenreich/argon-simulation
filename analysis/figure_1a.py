import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils import compute_temperatures

Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_1a.log")],
)
logger = logging.getLogger(__name__)

# --- 1. Compute Temperatures ---
temperatures = compute_temperatures("traj.lammpstrj")
n_frames = len(temperatures)

# --- 2. Create the Time Axis ---
timestep_fs = 2.0
time_ps = np.arange(n_frames) * timestep_fs / 1000.0

# --- 3. Plot the Figure ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))

# Main plot
ax.plot(
    time_ps,
    temperatures,
    color="#2E86AB",
    linewidth=1.5,
    alpha=0.8,
    label="Temperature",
)

# --- 4. Add Insets for Detailed Regions ---

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
inset1.axhline(94.4, color="gray", linestyle="--", linewidth=1, alpha=0.7)
inset1.set_xlim(1.0, 2.0)
inset1.set_ylim(temperatures[mask1].min() - 0.5, temperatures[mask1].max() + 0.5)
inset1.set_title("1.0-2.0 ps Detail", fontsize=9, pad=3, fontweight="bold")
inset1.tick_params(labelsize=7)
inset1.grid(True, alpha=0.3)
inset1.set_xlabel("Time (ps)", fontsize=7)
inset1.set_ylabel("T (K)", fontsize=7)
# Add border to make inset more visible
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
inset2.axhline(94.4, color="gray", linestyle="--", linewidth=1, alpha=0.7)
inset2.set_xlim(5.0, 6.0)
inset2.set_ylim(temperatures[mask2].min() - 0.5, temperatures[mask2].max() + 0.5)
inset2.set_title("5.0-6.0 ps Detail", fontsize=9, pad=3, fontweight="bold")
inset2.tick_params(labelsize=7)
inset2.grid(True, alpha=0.3)
inset2.set_xlabel("Time (ps)", fontsize=7)
inset2.set_ylabel("T (K)", fontsize=7)
# Add border to make inset more visible
for spine in inset2.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color("#F18F01")

# Highlight the inset regions on the main plot (more subtle)
ax.axvspan(1.0, 2.0, alpha=0.1, color="#A23B72")
ax.axvspan(5.0, 6.0, alpha=0.1, color="#F18F01")

# --- 5. Format the Main Plot ---
ax.axhline(
    94.4,
    color="gray",
    linestyle="-",
    linewidth=1.5,
    alpha=0.8,
    label="Target Temperature",
)
ax.set_ylabel("Temperature (K)", fontsize=14, fontweight="bold")
ax.set_xlabel("Time (ps)", fontsize=14, fontweight="bold")
ax.set_title(
    "Temperature vs Time",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Add statistics text box
stats_text = f"Mean: {temperatures.mean():.1f} K\nMin: {temperatures.min():.1f} K\nMax: {temperatures.max():.1f} K"
ax.text(
    0.02,
    0.92,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"),
)

ax.tick_params(axis="both", which="major", labelsize=12)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Position legend more strategically to avoid overlap
ax.legend(loc="lower right", fontsize=12, framealpha=0.9, edgecolor="gray")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/figure_1a.png", dpi=300, bbox_inches="tight")
plt.show()
