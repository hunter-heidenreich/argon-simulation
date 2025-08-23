"""
Argon Simulation Analysis Package

A comprehensive toolkit for analyzing molecular dynamics simulations of liquid argon,
implementing the analysis methods from Rahman's foundational 1964 paper.
"""

__version__ = "1.0.0"
__author__ = "Hunter Heidenreich"

from argon_sim.constants import CONSTANTS
from argon_sim.trajectory import load_trajectory_data, get_velocities
from argon_sim.thermodynamics import (
    compute_temperatures,
    calculate_temperature_from_velocities,
)
from argon_sim.dynamics import compute_msd, compute_vacf, compute_displacement_moments
from argon_sim.structure import (
    compute_radial_distribution,
    compute_structure_factor,
    compute_gdt,
    compute_gs,
    fourier_transform_3d,
    inverse_fourier_transform_3d,
)
from argon_sim.caching import CacheManager, cached_computation
from argon_sim.plotting import (
    setup_plotting_style,
    setup_figure_directory,
    clean_axes,
    save_figure,
    apply_rahman_style,
    setup_logging_and_artifacts,
)

__all__ = [
    "CONSTANTS",
    "load_trajectory_data",
    "get_velocities",
    "compute_temperatures",
    "calculate_temperature_from_velocities",
    "compute_msd",
    "compute_vacf",
    "compute_displacement_moments",
    "compute_radial_distribution",
    "compute_structure_factor",
    "compute_gdt",
    "compute_gs",
    "fourier_transform_3d",
    "inverse_fourier_transform_3d",
    "CacheManager",
    "cached_computation",
    "setup_plotting_style",
    "setup_figure_directory",
    "clean_axes",
    "save_figure",
    "apply_rahman_style",
    "setup_logging_and_artifacts",
]
