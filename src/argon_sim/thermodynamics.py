"""
Thermodynamic calculations for molecular dynamics simulations.

This module contains functions for calculating thermodynamic properties
like temperature from velocity data.
"""

import logging

import numpy as np

from argon_sim.caching import cached_computation
from argon_sim.constants import CONSTANTS
from argon_sim.trajectory import get_velocities

logger = logging.getLogger(__name__)


@cached_computation("temperatures")
def compute_temperatures(filename: str) -> np.ndarray:
    """
    Compute instantaneous temperatures from trajectory file.

    Temperature is calculated from the kinetic energy of atoms using
    the equipartition theorem: <KE> = (3/2) * N * kB * T

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        Array of temperatures for each frame in Kelvin.
    """
    velocities = get_velocities(filename)
    return calculate_temperature_from_velocities(velocities)


def calculate_temperature_from_velocities(velocities: np.ndarray) -> np.ndarray:
    """
    Calculate temperature from velocity data using the equipartition theorem.

    Args:
        velocities: Array of shape (n_frames, n_atoms, 3) with velocity data
                   in units of Angstrom/femtosecond.

    Returns:
        Array of temperatures for each frame in Kelvin.
    """
    mass_kg = CONSTANTS["MASS_KG"]
    n_atoms = velocities.shape[1]

    # Temperature factor: m / (3 * N * kB)
    temp_factor = mass_kg / (3 * n_atoms * CONSTANTS["KB_SI"])

    # Calculate kinetic energy per frame
    v_squared = np.sum(velocities**2, axis=(1, 2))  # Sum over atoms and dimensions
    v_squared_si = v_squared * (CONSTANTS["A_fs_to_m_s"] ** 2)

    return temp_factor * v_squared_si


def wrap_positions(positions: np.ndarray, box_dims: np.ndarray) -> np.ndarray:
    """
    Apply periodic boundary conditions to unwrapped positions.

    Args:
        positions: Array of positions (n_frames, n_atoms, 3) in Angstroms.
        box_dims: Array of box dimensions (n_frames, 3) in Angstroms.

    Returns:
        Wrapped positions array with same shape as input.
    """
    return positions % box_dims[:, np.newaxis, :]
