"""
Dynamics calculations for molecular dynamics simulations.

This module contains functions for calculating dynamic properties
like mean square displacement (MSD) and velocity autocorrelation function (VACF).
"""

import logging

import numpy as np
from tqdm import tqdm

from argon_sim.caching import cached_computation
from argon_sim.constants import CONSTANTS
from argon_sim.trajectory import load_trajectory_data, get_velocities

logger = logging.getLogger(__name__)


def _calculate_msd_averaged(positions: np.ndarray) -> np.ndarray:
    """
    Calculates the Mean Square Displacement by averaging over multiple time origins.
    This highly optimized version pre-computes all squared displacements to avoid redundant calculations.

    Args:
        positions: The array of atom positions over time,
                  with shape (n_frames, n_atoms, 3).

    Returns:
        A 1D array of the averaged MSD values for each time lag.
    """
    n_frames, n_atoms, _ = positions.shape
    msd_per_lag = np.zeros(n_frames)

    logger.info(
        f"Pre-computing all squared displacements for {n_frames} frames and {n_atoms} atoms..."
    )

    # Memory-efficient approach: compute chunk by chunk if needed
    max_memory_gb = 128  # Adjust based on available RAM
    bytes_per_element = 8  # float64
    elements_per_gb = 1024**3 / bytes_per_element
    max_elements = max_memory_gb * elements_per_gb

    if n_frames * n_frames * n_atoms > max_elements:
        logger.info(
            "Large dataset detected, using memory-efficient chunked approach..."
        )
        return _calculate_msd_chunked(positions)

    # For smaller datasets, use the fastest approach
    logger.info("Computing all squared displacements at once...")

    # Expand positions for broadcasting: (n_frames, 1, n_atoms, 3) and (1, n_frames, n_atoms, 3)
    pos_i = positions[:, None, :, :]  # Shape: (n_frames, 1, n_atoms, 3)
    pos_j = positions[None, :, :, :]  # Shape: (1, n_frames, n_atoms, 3)

    # Compute all pairwise squared displacements at once
    # Shape: (n_frames, n_frames, n_atoms)
    sq_displacements = np.sum((pos_j - pos_i) ** 2, axis=3)

    logger.info("Computing MSD for each lag...")

    # Now just slice and average for each lag - this is super fast!
    for dt in range(1, n_frames):
        # Extract diagonal slice for this lag
        # We want pairs (i, i+dt) where i+dt < n_frames
        diagonal_indices = np.arange(n_frames - dt)
        lag_displacements = sq_displacements[diagonal_indices, diagonal_indices + dt, :]

        # Average over all time origins and all atoms
        msd_per_lag[dt] = np.mean(lag_displacements)

        if dt % 100 == 0:
            logger.info(f"Processed lag {dt}/{n_frames}...")

    # The MSD at lag 0 is always 0
    msd_per_lag[0] = 0.0

    return msd_per_lag


def _calculate_msd_chunked(positions: np.ndarray) -> np.ndarray:
    """
    Memory-efficient version for very large datasets.
    Computes MSD in chunks to avoid memory overflow.

    Args:
        positions: Array of positions with shape (n_frames, n_atoms, 3)

    Returns:
        Array of MSD values for each time lag
    """
    n_frames, _, _ = positions.shape
    msd_per_lag = np.zeros(n_frames)

    logger.info("Using chunked computation to manage memory...")

    # Process each lag individually but still use vectorized operations
    for dt in range(1, n_frames):
        # For this lag, compute all valid displacements
        origins = positions[:-dt]  # Shape: (n_frames-dt, n_atoms, 3)
        targets = positions[dt:]  # Shape: (n_frames-dt, n_atoms, 3)

        # Compute squared displacements for this lag
        sq_displacements = np.sum((targets - origins) ** 2, axis=2)
        msd_per_lag[dt] = np.mean(sq_displacements)

        if dt % 50 == 0:
            logger.info(f"Processed lag {dt}/{n_frames}...")

    msd_per_lag[0] = 0.0
    return msd_per_lag


@cached_computation("msd")
def compute_msd(filename: str) -> np.ndarray:
    """
    Compute Mean Square Displacement from trajectory file with caching.

    The MSD quantifies how far particles move from their initial positions
    as a function of time, providing insight into diffusive behavior.

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        Array of MSD values for each time lag in Angstrom^2.
    """
    positions, _, _ = load_trajectory_data(filename)
    logger.info("Calculating time-averaged MSD... (This may take a moment)")
    msd = _calculate_msd_averaged(positions)
    logger.info("MSD calculation complete.")
    return msd


def _calculate_vacf_averaged(velocities: np.ndarray) -> np.ndarray:
    """
    Calculates the time-averaged Velocity Autocorrelation Function (VACF).

    Args:
        velocities: Array of shape (n_frames, n_atoms, 3) with velocity data.

    Returns:
        Normalized VACF array.
    """
    n_frames, n_atoms, _ = velocities.shape
    vacf = np.zeros(n_frames)
    logger.info("Calculating VACF...")

    for dt in range(n_frames):
        v_origins = velocities[: -dt if dt > 0 else None]
        v_targets = velocities[dt:]
        dot_products = np.sum(v_origins * v_targets, axis=2)
        vacf[dt] = np.mean(dot_products)
        if dt % 250 == 0:
            logger.info(f"  Processed VACF lag {dt}/{n_frames}...")

    normalized_vacf = vacf / vacf[0] if vacf[0] != 0 else vacf
    logger.info("VACF calculation complete.")
    return normalized_vacf


@cached_computation("vacf")
def compute_vacf(filename: str) -> np.ndarray:
    """
    Compute Velocity Autocorrelation Function from trajectory file with caching.

    The VACF measures how correlated an atom's velocity is with its velocity
    at a later time, revealing information about atomic motion patterns.

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        Array of normalized VACF values for each time lag.
    """
    velocities = get_velocities(filename)
    logger.info("Calculating time-averaged VACF... (This may take a moment)")
    vacf = _calculate_vacf_averaged(velocities)
    logger.info("VACF calculation complete.")
    return vacf


def langevin_decay_vacf(
    time_ps: np.ndarray, temp_K: float, mass_amu: float, D_cm2_s: float
) -> np.ndarray:
    """
    Calculates the Langevin-type exponential decay for the VACF.

    This represents the theoretical VACF for Brownian motion, providing
    a comparison baseline for the simulated VACF.

    Args:
        time_ps: Time array in picoseconds
        temp_K: Temperature in Kelvin
        mass_amu: Mass in atomic mass units
        D_cm2_s: Diffusion coefficient in cm^2/s

    Returns:
        Langevin VACF decay array
    """
    mass_kg = mass_amu * CONSTANTS["AMU_to_KG"]
    D_m2_s = D_cm2_s * 1e-4
    time_s = time_ps * 1e-12
    zeta = (CONSTANTS["KB_SI"] * temp_K) / (mass_kg * D_m2_s)
    return np.exp(-zeta * time_s)


def langevin_spectrum(beta: np.ndarray, mass_amu: float, D_cm2_s: float) -> np.ndarray:
    """
    Calculates the Lorentzian spectrum for Langevin diffusion.

    The frequency spectrum f(β) = λ² / (λ² + β²) where λ = ħ / (M*D)

    Args:
        beta: Frequency array
        mass_amu: Mass in atomic mass units
        D_cm2_s: Diffusion coefficient in cm^2/s

    Returns:
        Lorentzian spectrum array
    """
    mass_kg = mass_amu * CONSTANTS["AMU_to_KG"]
    D_m2_s = D_cm2_s * 1e-4

    # Calculate the dimensionless constant λ = ħ / (M*D)
    lam = CONSTANTS["HBAR_SI"] / (mass_kg * D_m2_s)

    return lam**2 / (lam**2 + beta**2)


@cached_computation("displacement_moments")
def compute_displacement_moments(filename: str) -> np.ndarray:
    """
    Computes the moments <r^2n> of the displacement for n=1, 2, 3, 4.

    This function calculates higher-order moments of particle displacements,
    which are used to analyze non-Gaussian behavior in the Van Hove correlation function.

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        A numpy array of shape (n_frames, 4) where columns correspond to
        n=1, 2, 3, 4 (i.e., <r^2>, <r^4>, <r^6>, <r^8>).
    """
    positions, _, _ = load_trajectory_data(filename)
    n_frames = positions.shape[0]
    moments = np.zeros((n_frames, 4))

    for dt in tqdm(range(1, n_frames), desc="Computing displacement moments"):
        displacements = positions[dt:] - positions[:-dt]
        r_squared = np.sum(displacements**2, axis=2)

        moments[dt, 0] = np.mean(r_squared)
        moments[dt, 1] = np.mean(r_squared**2)
        moments[dt, 2] = np.mean(r_squared**3)
        moments[dt, 3] = np.mean(r_squared**4)

    return moments
