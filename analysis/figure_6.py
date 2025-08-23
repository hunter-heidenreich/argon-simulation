# figure_6.py
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    compute_gdt,
    compute_gs,
    calculate_gr,
    fourier_transform_3d,
    inverse_fourier_transform_3d,
)

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
time_points_ps = [1.0, 2.5]
TRAJ_FILE = "traj.lammpstrj"

logger.info("Calculating G_d(r, t)...")
gdt_data = compute_gdt(TRAJ_FILE, t_lags_ps=time_points_ps)

logger.info("Calculating G_s(r, t)...")
gs_data = compute_gs(TRAJ_FILE, t_lags_ps=time_points_ps)

logger.info("Calculating g(r)...")
r_vals, g_r, density = calculate_gr(TRAJ_FILE)

# --- Vineyard Approximation using Fourier Transforms ---
vineyard_data = {}
logger.info("Calculating Vineyard Approximation via Fourier Transforms...")

# Create a k-space grid
k_vals = np.linspace(0.1, 20.0, 1000)

# 1. Transform g(r) to get the structure factor S(k)
s_k_minus_1 = fourier_transform_3d(r_vals, density * (g_r - 1), k_vals)

for t_ps in time_points_ps:
    if t_ps in gs_data:
        r_gs, gs_t = gs_data[t_ps]

        # 2. Transform G_s(r,t) to get F_s(k,t)
        f_s_k_t = fourier_transform_3d(r_gs, gs_t, k_vals)

        # 3. Multiply in k-space
        i_d_k_t = s_k_minus_1 * f_s_k_t
        
        # 4. Inverse transform the product to get the distinct part correlation
        rho_gd_minus_1 = inverse_fourier_transform_3d(k_vals, i_d_k_t, r_vals)
        
        # 5. Reconstruct the final function
        g_vineyard = 1 + rho_gd_minus_1 / density
        vineyard_data[t_ps] = (r_vals, g_vineyard)


# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 12), sharex=True, sharey=True
)

# Subplot (a): t = 1.0 ps
if 1.0 in gdt_data:
    r_a, gdt_a = gdt_data[1.0]
    ax1.plot(
        r_a / SIGMA, gdt_a, ".", markersize=5, color="black", label=f"$G_d(r, t={1.0}ps)$"
    )
    if 1.0 in vineyard_data:
        r_v, g_v = vineyard_data[1.0]
        ax1.plot(
            r_v / SIGMA, g_v, "x", markersize=5, color="black", label="Vineyard Approx."
        )
    ax1.text(0.05, 0.9, "(a)", transform=ax1.transAxes, fontsize=14, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=12)

# Subplot (b): t = 2.5 ps
if 2.5 in gdt_data:
    r_b, gdt_b = gdt_data[2.5]
    ax2.plot(
        r_b / SIGMA, gdt_b, ".", markersize=5, color="black", label=f"$G_d(r, t={2.5}ps)$"
    )
    if 2.5 in vineyard_data:
        r_v, g_v = vineyard_data[2.5]
        ax2.plot(
            r_v / SIGMA, g_v, "x", markersize=5, color="black", label="Vineyard Approx."
        )
    ax2.text(0.05, 0.9, "(b)", transform=ax2.transAxes, fontsize=14, fontweight='bold')
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

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("artifacts/figure_6.png", dpi=300, bbox_inches="tight")
plt.show()