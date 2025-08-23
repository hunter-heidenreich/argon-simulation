"""
Structural analysis functions for molecular dynamics simulations.

This module contains functions for calculating structural properties
like radial distribution functions and Van Hove correlation functions.
"""

import logging

import numpy as np
from tqdm import tqdm

from argon_sim.caching import cached_computation
from argon_sim.constants import CONSTANTS
from argon_sim.trajectory import load_trajectory_data

logger = logging.getLogger(__name__)


@cached_computation("gr")
def compute_radial_distribution(filename: str, dr: float = 0.05):
    """
    Calculates the pair correlation function g(r) from the last frame.

    The radial distribution function g(r) describes how the atomic density
    varies as a function of distance from a reference atom, normalized by
    the bulk density.

    Args:
        filename: Path to the LAMMPS trajectory file
        dr: Bin width for the radial distance in Angstroms

    Returns:
        Tuple of (r_values, g_r, density) where:
        - r_values: Array of radial distances in Angstroms
        - g_r: Array of g(r) values
        - density: Number density in atoms/Angstrom^3
    """
    positions, _, box_dims_all = load_trajectory_data(filename)

    positions_last_frame = positions[-1]
    box_dims = box_dims_all.mean(axis=0)

    num_atoms = positions_last_frame.shape[0]
    volume = np.prod(box_dims)
    density = num_atoms / volume
    max_r = np.min(box_dims) / 2.0

    bins = np.arange(0, max_r + dr, dr)
    num_bins = len(bins) - 1
    g_r = np.zeros(num_bins)

    for i in tqdm(range(num_atoms), desc="Calculating g(r)", unit="atom"):
        rij = positions_last_frame[i] - positions_last_frame
        rij -= box_dims * np.round(rij / box_dims)
        r = np.linalg.norm(rij, axis=1)
        r[i] = np.inf  # Exclude self

        hist, _ = np.histogram(r, bins=bins)
        g_r += hist

    r_values = bins[:-1] + dr / 2.0
    shell_volumes = 4.0 * np.pi * r_values**2 * dr
    n_ideal = density * shell_volumes

    # Normalize by the number of atoms and ideal shell density
    g_r /= num_atoms * n_ideal

    return r_values, g_r, density


def compute_structure_factor(r: np.ndarray, g_r: np.ndarray, density: float):
    """
    Calculates the static structure factor S(k) from g(r).
    
    This function computes S(k) = 1 + ρ * ∫[g(r) - 1] * 4πr² * sinc(kr/π) dr
    
    Args:
        r: Radial distance array in Angstroms
        g_r: Radial distribution function values
        density: Number density in atoms/Angstrom^3
        
    Returns:
        Tuple of (k_values, s_k) where:
        - k_values: Array of k values in inverse Angstroms
        - s_k: Array of structure factor S(k) values
    """
    k = np.linspace(0.1, 10.0, 1000)
    integrand = g_r - 1.0

    s_k = []
    for k_val in k:
        bessel = np.sinc(k_val * r / np.pi)
        fourier_integrand = 4 * np.pi * r**2 * integrand * bessel
        integral = np.trapezoid(fourier_integrand, r)
        s_k.append(1 + density * integral)

    return k, np.array(s_k)


@cached_computation("gdt")
def compute_gdt(filename: str, t_lags_ps: list, dr: float = 0.05) -> dict:
    """
    Computes the time-dependent pair correlation function G_d(r, t) for specified time lags.

    This function averages over all possible time origins in the trajectory.
    G_d(r,t) describes the conditional probability of finding a particle at
    distance r from any particle at time t, given positions at time 0.

    Args:
        filename: Path to the LAMMPS trajectory file.
        t_lags_ps: A list of time lags in picoseconds for which to calculate G_d(r, t).
        dr: The bin width for the radial distance (in Angstroms).

    Returns:
        A dictionary where keys are the time lags (in ps) and values are tuples
        containing (r_values, gdt_values).
    """
    positions, _, box_dims_all = load_trajectory_data(filename)
    n_frames, n_atoms, _ = positions.shape

    # Use an average box dimension for normalization
    box_dims = box_dims_all.mean(axis=0)
    volume = np.prod(box_dims)

    # Setup bins for histogram
    max_r = np.min(box_dims) / 2.0
    bins = np.arange(0, max_r + dr, dr)
    r_values = bins[:-1] + dr / 2.0
    shell_volumes = 4.0 * np.pi * r_values**2 * dr

    # Normalization factor for an ideal gas
    n_ideal = (n_atoms - 1) / volume * shell_volumes

    # Convert time lags from picoseconds to integer timesteps
    dt_lags = [int(t_ps * 1000 / CONSTANTS["TIMESTEP_FS"]) for t_ps in t_lags_ps]

    results = {}

    for i, dt in enumerate(dt_lags):
        t_ps = t_lags_ps[i]
        logger.info(f"Calculating G_d(r, t) for t = {t_ps} ps (dt = {dt} steps)...")

        if dt >= n_frames:
            logger.warning(
                f"Time lag {t_ps} ps is too large for the trajectory length. Skipping."
            )
            continue

        total_counts = np.zeros(len(r_values))
        num_origins = n_frames - dt

        # Average over all possible time origins
        for t0 in tqdm(range(num_origins), desc=f"Lag {t_ps} ps", unit="origin"):
            pos_initial = positions[t0]
            pos_final = positions[t0 + dt]

            # Vectorized calculation of all-pairs distances
            displacements = pos_final[np.newaxis, :, :] - pos_initial[:, np.newaxis, :]

            # Apply periodic boundary conditions
            displacements -= box_dims * np.round(displacements / box_dims)

            distances = np.linalg.norm(displacements, axis=2)

            # Exclude self-correlation (i -> i)
            np.fill_diagonal(distances, np.inf)

            # Histogram the distances and add to total
            counts, _ = np.histogram(distances.flatten(), bins=bins)
            total_counts += counts

        # Normalize the histogram to get G_d(r, t)
        # The normalization is by the total number of pairs considered (N_origins * N_atoms)
        # and the ideal gas distribution for N-1 particles.
        gdt_values = total_counts / (num_origins * n_atoms * n_ideal)
        results[t_ps] = (r_values, gdt_values)

    return results


@cached_computation("gs")
def compute_gs(filename: str, t_lags_ps: list, dr: float = 0.05) -> dict:
    """
    Computes the self part of the Van Hove correlation function G_s(r, t).

    G_s(r,t) describes the probability that a particle moves a distance r
    in time t, providing insight into single-particle dynamics.

    Args:
        filename: Path to the LAMMPS trajectory file.
        t_lags_ps: A list of time lags in picoseconds for which to calculate G_s(r, t).
        dr: The bin width for the radial distance (in Angstroms).

    Returns:
        A dictionary where keys are the time lags (in ps) and values are tuples
        containing (r_values, gs_values).
    """
    positions, _, box_dims_all = load_trajectory_data(filename)
    n_frames, n_atoms, _ = positions.shape

    # Use an average box dimension
    box_dims = box_dims_all.mean(axis=0)

    # Convert time lags from ps to frame indices
    timestep_fs = 2.0  # From constants
    timestep_ps = timestep_fs / 1000.0
    t_lag_frames = [int(t_ps / timestep_ps) for t_ps in t_lags_ps]

    # Set up radial bins
    max_r = np.min(box_dims) / 2.0
    bins = np.arange(0, max_r + dr, dr)
    r_values = bins[:-1] + dr / 2.0
    num_bins = len(bins) - 1

    results = {}

    for t_ps, dt_frames in zip(t_lags_ps, t_lag_frames):
        if dt_frames >= n_frames:
            logger.warning(f"Time lag {t_ps} ps exceeds trajectory length")
            continue

        logger.info(f"Computing G_s(r, t) for t = {t_ps} ps (frame lag {dt_frames})")

        gs = np.zeros(num_bins)
        num_origins = n_frames - dt_frames

        for t0 in range(num_origins):
            pos_t0 = positions[t0]
            pos_t = positions[t0 + dt_frames]

            # Calculate displacement for each particle
            displacements = pos_t - pos_t0
            # Note: Not applying PBC here as we want actual displacements
            r = np.linalg.norm(displacements, axis=1)

            hist, _ = np.histogram(r, bins=bins)
            gs += hist

        # Normalize by number of origins and atoms
        shell_volumes = 4.0 * np.pi * r_values**2 * dr
        normalization = num_origins * n_atoms * shell_volumes

        # Avoid division by zero
        mask = shell_volumes > 0
        gs[mask] = gs[mask] / normalization[mask]

        results[t_ps] = (r_values, gs)

    return results


def fourier_transform_3d(
    r: np.ndarray, func_r: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """
    Performs a 3D Fourier transform on an isotropic function f(r).

    For a spherically symmetric function, the 3D FT reduces to:
    F(k) = 4π ∫₀^∞ r² f(r) sinc(kr/π) dr

    Args:
        r: Radial coordinate array
        func_r: Function values at r
        k: Wavevector magnitude array

    Returns:
        Fourier transform F(k)
    """
    integrand = 4 * np.pi * r**2 * func_r * np.sinc(k[:, np.newaxis] * r / np.pi)
    f_k = np.trapezoid(integrand, r, axis=1)
    return f_k


def inverse_fourier_transform_3d(
    k: np.ndarray, func_k: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """
    Performs an inverse 3D Fourier transform on an isotropic function f(k).

    For a spherically symmetric function:
    f(r) = (1/2π²) ∫₀^∞ k² F(k) sinc(kr/π) dk

    Args:
        k: Wavevector magnitude array
        func_k: Function values at k
        r: Radial coordinate array

    Returns:
        Inverse Fourier transform f(r)
    """
    integrand = (
        (1 / (2 * np.pi**2)) * k**2 * func_k * np.sinc(r[:, np.newaxis] * k / np.pi)
    )
    f_r = np.trapezoid(integrand, k, axis=1)
    return f_r
