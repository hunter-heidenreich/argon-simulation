# figure_2.py
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

from utils import load_trajectory_data, wrap_positions

Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/figure_2.log")],
)
logger = logging.getLogger(__name__)


def calculate_gr(positions: np.ndarray, box_dims: np.ndarray, dr: float, max_r: float):
    """Calculates the pair correlation function g(r)."""
    num_atoms = positions.shape[0]
    volume = np.prod(box_dims)
    density = num_atoms / volume

    bins = np.arange(0, max_r + dr, dr)
    num_bins = len(bins) - 1
    g_r = np.zeros(num_bins)

    for i in tqdm(range(num_atoms), desc="Calculating g(r)", unit="atom"):
        for j in range(i + 1, num_atoms):
            rij = positions[i] - positions[j]
            rij = rij - box_dims * np.round(rij / box_dims)
            r = np.linalg.norm(rij)

            if r < max_r:
                bin_index = int(r / dr)
                g_r[bin_index] += 2

    r_values = bins[:-1] + dr / 2.0
    shell_volumes = 4.0 * np.pi * r_values**2 * dr
    n_ideal = density * shell_volumes

    g_r /= num_atoms * n_ideal

    return r_values, g_r, density


def calculate_sk(r: np.ndarray, g_r: np.ndarray, density: float):
    """Calculates the static structure factor S(k) from g(r)."""
    k = np.linspace(0.1, 10.0, 1000)
    integrand = g_r - 1.0

    s_k = []
    for k_val in k:
        bessel = np.sinc(k_val * r / np.pi)
        fourier_integrand = 4 * np.pi * r**2 * integrand * bessel
        integral = np.trapezoid(fourier_integrand, r)
        s_k.append(1 + density * integral)

    return k, np.array(s_k)


# --- Main Script ---

# Constants from the paper for Argon
SIGMA = 3.4  # Angstroms

# Load the data
logger.info("Loading trajectory data...")
try:
    xs, _, box_dims_all = load_trajectory_data("traj.lammpstrj")
    xs = wrap_positions(xs, box_dims_all)
    logger.info(f"Loaded {len(xs)} frames")
except ValueError:
    logger.error("Could not unpack data. Check the load_trajectory_data function in utils.py")
    exit()

if xs.size == 0:
    logger.error("No data loaded. Exiting.")
    exit()

# Use the last frame for calculation
positions_last_frame = xs[-1]
box_dims_last_frame = box_dims_all[-1]

# Set parameters
dr = 0.05
max_r = box_dims_last_frame[0] / 2.0

# Calculate g(r) and S(k)
r, g_r, density = calculate_gr(positions_last_frame, box_dims_last_frame, dr, max_r)
k, s_k = calculate_sk(r, g_r, density)

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: g(r)
ax1.plot(r / SIGMA, g_r, "o", markersize=3, color="black", markerfacecolor="none")
ax1.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
ax1.set_xlabel("r / $\\sigma$", fontsize=14, fontweight="bold")
ax1.set_ylabel("g(r)", fontsize=14, fontweight="bold")
ax1.set_title("Pair Correlation Function g(r)", fontsize=16, fontweight="bold", pad=20)
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 3.1)
ax1.tick_params(axis="both", which="major", labelsize=12)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.grid(True, alpha=0.3)

# *** NEW: Find and annotate peaks in g(r) ***
# Use more stringent criteria for peak detection
gr_peaks_indices, properties = find_peaks(
    g_r, 
    height=1,
    prominence=0.25,
    distance=0.8/dr,
)

logger.info("\n--- g(r) Analysis ---")
logger.info("Expected peak locations from paper: r = 3.7, 7.0, 10.4 Å")
logger.info(f"Calculated peak locations: r = {np.round(r[gr_peaks_indices], 2)} Å")
logger.info(f"Peak heights: {np.round(g_r[gr_peaks_indices], 2)}")

# Annotate the first 3 peaks found
for i in range(min(3, len(gr_peaks_indices))):
    peak_idx = gr_peaks_indices[i]
    r_val = r[peak_idx]
    gr_val = g_r[peak_idx]
    ax1.annotate(f'{r_val:.1f} Å',
                 xy=(r_val / SIGMA, gr_val),
                 xytext=(r_val / SIGMA, gr_val + 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                 ha='center', fontsize=12)

# Plot 2: S(k)
ax2.plot(k * SIGMA, s_k, color="#2E86AB", linewidth=2)
ax2.set_xlabel("k$\\sigma$", fontsize=14, fontweight="bold")
ax2.set_ylabel("S(k)", fontsize=14, fontweight="bold")
ax2.set_title("Static Structure Factor S(k)", fontsize=16, fontweight="bold", pad=20)
ax2.set_xlim(0, 30)
ax2.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
ax2.tick_params(axis="both", which="major", labelsize=12)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.grid(True, alpha=0.3)

# Find and print peaks to compare with paper
sk_peaks_indices, sk_properties = find_peaks(
    s_k, 
    height=1,           # Higher threshold for S(k)
    distance=5,           # Minimum separation in data points
    prominence=0.05,       # Peak prominence
)
peak_k_sigma = k[sk_peaks_indices] * SIGMA
logger.info("\n--- S(k) Analysis ---")
logger.info("Expected peak locations from paper: kσ = 6.8, 12.5, 18.5, 24.8")
logger.info(f"Calculated peak locations: kσ = {np.round(peak_k_sigma, 2)}")
logger.info(f"Peak heights: {np.round(s_k[sk_peaks_indices], 2)}")

# Annotate peaks on the plot
for i in range(min(4, len(sk_peaks_indices))):
    peak_idx = sk_peaks_indices[i]
    k_val = k[peak_idx] * SIGMA
    sk_val = s_k[peak_idx]
    ax2.annotate(f'{k_val:.1f}' + ' Å$^{-1}$',
                 xy=(k_val, sk_val),
                 xytext=(k_val, sk_val + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                 ha='center', fontsize=12)

plt.tight_layout()
plt.savefig("artifacts/figure_2.png", dpi=300, bbox_inches="tight")
plt.show()
