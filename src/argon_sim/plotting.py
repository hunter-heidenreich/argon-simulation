"""
Plotting utilities for consistent figure generation across all analysis scripts.

This module provides common plotting functions and styling utilities to ensure
consistency across all figure generation scripts.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def setup_plotting_style():
    """Set up consistent plotting style for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")


def setup_figure_directory():
    """Create artifacts directory if it doesn't exist."""
    Path("artifacts").mkdir(exist_ok=True)


def clean_axes(ax):
    """Apply consistent axis cleaning to remove top and right spines."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.3)


def save_figure(filename: str, dpi: int = 300):
    """Save figure with consistent settings."""
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")


def apply_rahman_style(ax, xlabel: str = None, ylabel: str = None, title: str = None):
    """Apply Rahman 1964 paper consistent styling to an axis."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    ax.tick_params(axis="both", which="major", labelsize=12)
    clean_axes(ax)


def setup_logging_and_artifacts(script_name: str):
    """Set up logging and artifacts directory for a script."""
    import logging
    
    setup_figure_directory()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(), 
            logging.FileHandler(f"artifacts/{script_name}.log")
        ],
    )
    return logging.getLogger(__name__)
