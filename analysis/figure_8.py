# figure_8.py
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
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_8.log")],
)
logger = logging.getLogger(__name__)

# Constants
SIGMA = 3.4  # Lennard-Jones sigma for Argon in Angstroms
TRAJ_FILE = "traj.lammpstrj"

# --- Data Calculation ---
# Define the (t, t') pairs from the paper [cite: 287]
time_pairs = {
    0.8: 0.5,  # Gd(t=0.8) is compared with Gs(t'=0.5)
    2.3: 1.5,  # Gd(t=2.3) is compared with Gs(t'=1.5)
}
gdt_times_ps = list(time_pairs.keys())
gs_times_ps = list(time_pairs.values())

logger.info(f"Calculating G_d(r, t) for t = {gdt_times_ps} ps...")
gdt_data = compute_gdt(TRAJ_FILE, t_lags_ps=gdt_times_ps)

logger.info(f"Calculating G_s(r, t') for t' = {gs_times_ps} ps...")
gs_data = compute_gs(TRAJ_FILE, t_lags_ps=gs_times_ps)

logger.info("Calculating g(r)...")
r_vals, g_r, density = calculate_gr(TRAJ_FILE)

# --- Delayed Convolution Calculation ---
delayed_conv_data = {}
logger.info("Calculating Delayed Convolution via Fourier Transforms...")

# Create a k-space grid
k_vals = np.linspace(0.1, 20.0, 1000)

# 1. Transform g(r) to get the structure factor S(k)
s_k_minus_1 = fourier_transform_3d(r_vals, density * (g_r - 1), k_vals)

for t_actual, t_delayed in time_pairs.items():
    if t_delayed in gs_data:
        r_gs, gs_t_delayed = gs_data[t_delayed]

        # 2. Transform G_s(r, t') to get F_s(k, t')
        f_s_k_t_delayed = fourier_transform_3d(r_gs, gs_t_delayed, k_vals)

        # 3. Multiply in k-space
        i_d_k_t = s_k_minus_1 * f_s_k_t_delayed
        
        # 4. Inverse transform
        rho_gd_minus_1 = inverse_fourier_transform_3d(k_vals, i_d_k_t, r_vals)
        
        # 5. Reconstruct the final function
        g_delayed = 1 + rho_gd_minus_1 / density
        delayed_conv_data[t_actual] = (r_vals, g_delayed)

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 12), sharex=True, sharey=True
)

# Panel (a): t=0.8ps, t'=0.5ps
t_actual_a, t_delayed_a = 0.8, 0.5
if t_actual_a in gdt_data and t_actual_a in delayed_conv_data:
    r_a, gdt_a = gdt_data[t_actual_a]
    r_c, conv_a = delayed_conv_data[t_actual_a]
    ax1.plot(r_a / SIGMA, gdt_a, ".", ms=5, color="black", label=f"$G_d(r, t={t_actual_a}ps)$")
    ax1.plot(r_c / SIGMA, conv_a, "x", ms=5, color="black", label=f"Convolution at $t'={t_delayed_a}ps$")
    ax1.text(0.05, 0.9, "(a)", transform=ax1.transAxes, fontsize=14, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=11)

# Panel (b): t=2.3ps, t'=1.5ps
t_actual_b, t_delayed_b = 2.3, 1.5
if t_actual_b in gdt_data and t_actual_b in delayed_conv_data:
    r_b, gdt_b = gdt_data[t_actual_b]
    r_c, conv_b = delayed_conv_data[t_actual_b]
    ax2.plot(r_b / SIGMA, gdt_b, ".", ms=5, color="black", label=f"$G_d(r, t={t_actual_b}ps)$")
    ax2.plot(r_c / SIGMA, conv_b, "x", ms=5, color="black", label=f"Convolution at $t'={t_delayed_b}ps$")
    ax2.text(0.05, 0.9, "(b)", transform=ax2.transAxes, fontsize=14, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=11)

# --- Formatting ---
fig.suptitle(
    "Delayed Convolution Approximation of $G_d(r, t)$",
    fontsize=16,
    fontweight="bold",
)
for ax in [ax1, ax2]:
    ax.axhline(1.0, color="gray", linewidth=1.5, linestyle="-")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

ax2.set_xlabel("r / $\\sigma$", fontsize=14, fontweight="bold")
ax1.set_xlim(0, 3.5)
ax1.set_ylim(0, 1.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("artifacts/figure_8.png", dpi=300, bbox_inches="tight")
plt.show()