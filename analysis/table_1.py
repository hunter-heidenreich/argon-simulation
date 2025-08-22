import logging
from pathlib import Path

import numpy as np

from utils import compute_temperatures

Path("artifacts").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("artifacts/table_1.log")],
)
logger = logging.getLogger(__name__)

# --- Load and Compute Temperature Statistics ---
logger.info("Loading and computing temperature data...")
temps = compute_temperatures("traj.lammpstrj")
logger.info(f"Loaded temperature data for {len(temps)} frames")

cumulative_temps = np.cumsum(temps) / np.arange(1, len(temps) + 1)
cumulative_temp_squared = np.cumsum(temps**2) / np.arange(1, len(temps) + 1)
rms_fluctuations = (
    np.sqrt(cumulative_temp_squared - cumulative_temps**2) / cumulative_temps
)

# --- Print Results Table ---
logger.info(f"{'Frame':>6} | {'Cumulative Temp (K)':>20} | {'RMS Fluctuation':>16}")
logger.info("-" * 50)

for i, temp in enumerate(cumulative_temps):
    if (i + 1) % 100 == 0:
        logger.info(f"{i + 1:>6} | {temp:>20.2f} | {rms_fluctuations[i]:>16.2f}")

# --- Summary Statistics ---
logger.info("\n--- Summary Statistics ---")
logger.info(f"Final average temperature: {cumulative_temps[-1]:.2f} K")
logger.info(f"Final RMS fluctuation: {rms_fluctuations[-1]:.4f}")
logger.info("Target temperature: 94.4 K")
logger.info(f"Temperature deviation: {abs(cumulative_temps[-1] - 94.4):.2f} K")
