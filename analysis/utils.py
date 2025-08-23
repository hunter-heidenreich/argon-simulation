"""
Utilities for loading and analyzing LAMMPS trajectory data.

This module provides functions for:
- Loading LAMMPS trajectory files with intelligent caching
- Computing physical properties from simulation data
- Managing periodic boundary conditions
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
from functools import wraps

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Physical constants for Argon simulations
CONSTANTS = {
    "MASS_AMU": 39.95,  # Argon atomic mass in AMU
    "KB_SI": 1.38065e-23,  # Boltzmann constant in J/K
    "AMU_to_KG": 1.66054e-27,  # AMU to kg conversion
    "A_fs_to_m_s": 1.0e5,  # Angstrom/femtosecond to m/s conversion
    "HBAR_SI": 1.05457e-34,  # Reduced Planck constant in J·s
    "TIMESTEP_FS": 2.0,  # Simulation timestep in femtoseconds
    "TEMPERATURE": 94.4,  # Simulation temperature in K
    "D_PAPER": 2.43e-5,  # Diffusion coefficient in cm^2/s
}


class CacheManager:
    """General-purpose caching system for computed data."""

    def __init__(self, cache_root: str = "cache"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(exist_ok=True)

    def _get_cache_key(self, source_path: Path, operation: str, **kwargs) -> str:
        """Generate a unique cache key based on source file and operation parameters."""
        # Include file name, operation, and any additional parameters in hash
        key_data = f"{source_path.name}_{operation}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _get_cache_dir(self, source_path: Path, operation: str, **kwargs) -> Path:
        """Get cache directory for a specific operation."""
        cache_key = self._get_cache_key(source_path, operation, **kwargs)
        cache_dir = self.cache_root / source_path.name / f"{operation}_{cache_key}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _is_cache_valid(self, cache_dir: Path, source_path: Path) -> bool:
        """Check if cached data is valid based on source file modification time."""
        meta_file = cache_dir / "meta.json"
        if not meta_file.exists():
            return False

        try:
            src_mtime = source_path.stat().st_mtime
            with open(meta_file, "r") as f:
                meta = json.load(f)
            cached_mtime = meta.get("source_mtime")
            return cached_mtime is not None and cached_mtime >= src_mtime
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    def get_cached_data(
        self, source_path: Path, operation: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if valid."""
        cache_dir = self._get_cache_dir(source_path, operation, **kwargs)

        if not self._is_cache_valid(cache_dir, source_path):
            return None

        try:
            data_file = cache_dir / "data.npz"
            if data_file.exists():
                logger.info(f"Loading cached {operation} data from {data_file}")
                with np.load(data_file) as npz:
                    return dict(npz)
        except Exception as e:
            logger.warning(f"Failed to load cache for {operation}: {e}")

        return None

    def save_cached_data(
        self, source_path: Path, operation: str, data: Dict[str, Any], **kwargs
    ) -> None:
        """Save computed data to cache."""
        cache_dir = self._get_cache_dir(source_path, operation, **kwargs)

        try:
            # Save data
            data_file = cache_dir / "data.npz"
            np.savez_compressed(data_file, **data)

            # Save metadata
            meta_file = cache_dir / "meta.json"
            meta = {
                "source_mtime": source_path.stat().st_mtime,
                "operation": operation,
                "parameters": kwargs,
            }
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"Cached {operation} data to {data_file}")
        except Exception as e:
            logger.warning(f"Failed to cache {operation} data: {e}")


# Global cache manager instance
_cache_manager = CacheManager()


def cached_computation(operation: str, **cache_kwargs):
    """Decorator for caching expensive computations based on source files."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(filename: Union[str, Path], *args, **kwargs):
            source_path = Path(filename)

            if not source_path.exists():
                logger.error(f"Source file '{filename}' not found.")
                return None

            # Try to load from cache
            cache_params = {**cache_kwargs, **kwargs}
            cached_data = _cache_manager.get_cached_data(
                source_path, operation, **cache_params
            )

            if cached_data is not None:
                # Return cached result in the same format as the original function
                if operation == "trajectory_data":
                    return (
                        cached_data["positions"],
                        cached_data["velocities"],
                        cached_data["box_dims"],
                    )
                elif operation == "temperatures":
                    return cached_data["temperatures"]
                elif operation == "msd":
                    return cached_data["msd"]
                elif operation == "vacf":
                    return cached_data["vacf"]
                elif operation == "gdt":
                    # Reconstruct the nested structure from flattened cache
                    gdt_result = {}
                    if "time_lags" in cached_data:
                        for t_lag in cached_data["time_lags"]:
                            r_key = f"r_values_{t_lag}"
                            gdt_key = f"gdt_values_{t_lag}"
                            if r_key in cached_data and gdt_key in cached_data:
                                gdt_result[float(t_lag)] = (cached_data[r_key], cached_data[gdt_key])
                    return gdt_result
                elif operation == "gs":
                    gs_result = {}
                    if "time_lags" in cached_data:
                        for t_lag in cached_data["time_lags"]:
                            r_key = f"r_values_{t_lag}"
                            gs_key = f"gs_values_{t_lag}"
                            if r_key in cached_data and gs_key in cached_data:
                                gs_result[float(t_lag)] = (cached_data[r_key], cached_data[gs_key])
                    return gs_result
                elif operation == "displacement_moments":
                    return cached_data["moments"]
                elif operation == "gr":
                    return cached_data["r"], cached_data["g_r"], cached_data["density"]
                else:
                    return cached_data

            # Compute and cache result
            result = func(filename, *args, **kwargs)

            # Prepare data for caching
            if operation == "trajectory_data" and result is not None:
                positions, velocities, box_dims = result
                if positions.size > 0:  # Only cache if we have valid data
                    cache_data = {
                        "positions": positions,
                        "velocities": velocities,
                        "box_dims": box_dims,
                    }
                    _cache_manager.save_cached_data(
                        source_path, operation, cache_data, **cache_params
                    )
            elif operation == "temperatures" and result is not None:
                cache_data = {"temperatures": result}
                _cache_manager.save_cached_data(
                    source_path, operation, cache_data, **cache_params
                )
            elif operation == "msd" and result is not None:
                cache_data = {"msd": result}
                _cache_manager.save_cached_data(
                    source_path, operation, cache_data, **cache_params
                )
            elif operation == "vacf" and result is not None:
                cache_data = {"vacf": result}
                _cache_manager.save_cached_data(
                    source_path, operation, cache_data, **cache_params
                )
            elif operation == "gdt" and result is not None:
                # For gdt, we need to flatten the nested structure for caching
                # result is a dict: {t_lag: (r_values, gdt_values), ...}
                cache_data = {}
                for t_lag, (r_vals, gdt_vals) in result.items():
                    cache_data[f"r_values_{t_lag}"] = r_vals
                    cache_data[f"gdt_values_{t_lag}"] = gdt_vals
                cache_data["time_lags"] = np.array(list(result.keys()))
                _cache_manager.save_cached_data(
                    source_path, operation, cache_data, **cache_params
                )
            elif operation == "gs" and result is not None:
                cache_data = {}
                for t_lag, (r_vals, gs_vals) in result.items():
                    cache_data[f"r_values_{t_lag}"] = r_vals
                    cache_data[f"gs_values_{t_lag}"] = gs_vals
                cache_data["time_lags"] = np.array(list(result.keys()))
                _cache_manager.save_cached_data(
                    source_path, operation, cache_data, **cache_params
                )
            elif operation == "displacement_moments" and result is not None:
                 cache_data = {"moments": result}
                 _cache_manager.save_cached_data(
                    source_path, operation, cache_data, **cache_params
                )
            elif operation == "gr" and result is not None:
                r, g_r, density = result
                cache_data = {"r": r, "g_r": g_r, "density": density}
                _cache_manager.save_cached_data(
                    source_path, operation, cache_data, **cache_params
                )


            return result

        return wrapper

    return decorator


def _parse_lammps_frame(
    lines: list, line_idx: int
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single LAMMPS trajectory frame."""
    # We start at "ITEM: TIMESTEP", advance to get number of atoms
    line_idx += 1  # Skip timestep value
    line_idx += 1  # Skip "ITEM: NUMBER OF ATOMS"
    line_idx += 1  # Now we're at the number of atoms line
    num_atoms = int(lines[line_idx])
    line_idx += 1  # Move to "ITEM: BOX BOUNDS ..."
    line_idx += 1  # Skip to first box bounds line

    # Parse box dimensions
    box_lo, box_hi = [], []
    for _ in range(3):
        parts = lines[line_idx].strip().split()
        box_lo.append(float(parts[0]))
        box_hi.append(float(parts[1]))
        line_idx += 1

    box_dims = np.array(box_hi) - np.array(box_lo)
    line_idx += 1  # Skip "ITEM: ATOMS ..." header

    # Parse atom data
    frame_data = []
    for _ in range(num_atoms):
        parts = lines[line_idx].strip().split()
        frame_data.append([float(p) for p in parts])
        line_idx += 1

    # Sort by atom ID and extract positions/velocities
    frame_array = np.array(frame_data)
    frame_array = frame_array[frame_array[:, 0].argsort()]

    positions = frame_array[:, 2:5]
    velocities = frame_array[:, 5:8]

    return line_idx, positions, velocities, box_dims


@cached_computation("trajectory_data")
def load_trajectory_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectory data from a LAMMPS dump file.

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        Tuple of (positions, velocities, box_dimensions) arrays.
        Each array has shape (n_frames, n_atoms, 3) or (n_frames, 3) for box_dims.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    positions_list, velocities_list, box_dims_list = [], [], []
    line_idx = 0

    logger.info(f"Parsing trajectory file: {filename}")

    while line_idx < len(lines):
        line = lines[line_idx].strip()

        if line == "ITEM: TIMESTEP":
            line_idx, positions, velocities, box_dims = _parse_lammps_frame(
                lines, line_idx
            )
            positions_list.append(positions)
            velocities_list.append(velocities)
            box_dims_list.append(box_dims)
        else:
            line_idx += 1

    logger.info(f"Loaded {len(positions_list)} frames from {filename}")

    return (
        np.array(positions_list),
        np.array(velocities_list),
        np.array(box_dims_list),
    )


def get_velocities(filename: str) -> np.ndarray:
    """Extract atomic velocities from trajectory file."""
    _, velocities, _ = load_trajectory_data(filename)
    return velocities


@cached_computation("temperatures")
def compute_temperatures(filename: str) -> np.ndarray:
    """
    Compute instantaneous temperatures from trajectory file.

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        Array of temperatures for each frame.
    """
    velocities = get_velocities(filename)
    return calculate_temperature_from_velocities(velocities)


def calculate_temperature_from_velocities(velocities: np.ndarray) -> np.ndarray:
    """
    Calculate temperature from velocity data.

    Args:
        velocities: Array of shape (n_frames, n_atoms, 3) with velocity data.

    Returns:
        Array of temperatures for each frame.
    """
    mass_kg = CONSTANTS["MASS_AMU"] * CONSTANTS["AMU_to_KG"]
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
        positions: Array of positions (n_frames, n_atoms, 3).
        box_dims: Array of box dimensions (n_frames, 3).

    Returns:
        Wrapped positions array.
    """
    return positions % box_dims[:, np.newaxis, :]


def _calculate_msd_averaged(positions: np.ndarray) -> np.ndarray:
    """
    Calculates the Mean Square Displacement by averaging over multiple time origins.
    This highly optimized version pre-computes all squared displacements to avoid redundant calculations.

    Args:
        positions (np.ndarray): The array of atom positions over time,
                                with shape (n_frames, n_atoms, 3).

    Returns:
        np.ndarray: A 1D array of the averaged MSD values for each time lag.
    """
    n_frames, n_atoms, _ = positions.shape
    msd_per_lag = np.zeros(n_frames)

    logger.info(
        f"Pre-computing all squared displacements for {n_frames} frames and {n_atoms} atoms..."
    )

    # More memory-efficient approach: compute chunk by chunk if needed
    max_memory_gb = 128  # Adjust based on available RAM
    bytes_per_element = 8  # float64
    elements_per_gb = 1024**3 / bytes_per_element
    max_elements = max_memory_gb * elements_per_gb

    if n_frames * n_frames * n_atoms > max_elements:
        logger.info("Large dataset detected, using memory-efficient chunked approach...")
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

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        Array of MSD values for each time lag.
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
    """Calculates the Langevin-type exponential decay for the VACF."""
    mass_kg = mass_amu * CONSTANTS["AMU_to_KG"]
    D_m2_s = D_cm2_s * 1e-4
    time_s = time_ps * 1e-12
    zeta = (CONSTANTS["KB_SI"] * temp_K) / (mass_kg * D_m2_s)
    return np.exp(-zeta * time_s)


def langevin_spectrum(beta: np.ndarray, mass_amu: float, D_cm2_s: float) -> np.ndarray:
    """
    Calculates the Lorentzian spectrum for Langevin diffusion.
    f(β) = λ² / (λ² + β²)
    """
    mass_kg = mass_amu * CONSTANTS["AMU_to_KG"]
    D_m2_s = D_cm2_s * 1e-4

    # Calculate the dimensionless constant λ = ħ / (M*D)
    lam = CONSTANTS["HBAR_SI"] / (mass_kg * D_m2_s)

    return lam**2 / (lam**2 + beta**2)

@cached_computation("gdt")
def compute_gdt(filename: str, t_lags_ps: list, dr: float = 0.05) -> dict:
    """
    Computes the time-dependent pair correlation function G_d(r, t) for specified time lags.

    This function averages over all possible time origins in the trajectory.

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
            logger.warning(f"Time lag {t_ps} ps is too large for the trajectory length. Skipping.")
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
    Computes the self-part of the Van Hove correlation function, G_s(r, t).

    Args:
        filename: Path to the LAMMPS trajectory file.
        t_lags_ps: A list of time lags in picoseconds.
        dr: The bin width for the radial distance (in Angstroms).

    Returns:
        A dictionary where keys are time lags (ps) and values are (r_values, gs_values).
    """
    positions, _, box_dims_all = load_trajectory_data(filename)
    n_frames, n_atoms, _ = positions.shape
    box_dims = box_dims_all.mean(axis=0)
    max_r = np.min(box_dims) / 2.0
    bins = np.arange(0, max_r + dr, dr)
    r_values = bins[:-1] + dr / 2.0
    shell_volumes = 4.0 * np.pi * r_values**2 * dr

    dt_lags = [int(t_ps * 1000 / CONSTANTS["TIMESTEP_FS"]) for t_ps in t_lags_ps]
    results = {}

    for i, dt in enumerate(dt_lags):
        t_ps = t_lags_ps[i]
        logger.info(f"Calculating G_s(r, t) for t = {t_ps} ps (dt = {dt} steps)...")

        if dt >= n_frames:
            logger.warning(f"Time lag {t_ps} ps is too large. Skipping.")
            continue

        displacements = positions[dt:] - positions[:-dt]
        distances = np.linalg.norm(displacements, axis=2).flatten()
        
        counts, _ = np.histogram(distances, bins=bins)
        
        # Normalize by the number of samples and shell volume
        n_samples = (n_frames - dt) * n_atoms
        gs_values = counts / (n_samples * shell_volumes)
        results[t_ps] = (r_values, gs_values)
        
    return results

@cached_computation("displacement_moments")
def compute_displacement_moments(filename: str) -> np.ndarray:
    """
    Computes the moments <r^2n> of the displacement for n=1, 2, 3, 4.

    Returns:
        A numpy array of shape (n_frames, 4) where columns correspond to
        n=1, 2, 3, 4.
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


@cached_computation("gr")
def calculate_gr(filename: str, dr: float = 0.05):
    """Calculates the pair correlation function g(r) from the last frame."""
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
        r[i] = np.inf # Exclude self
        
        hist, _ = np.histogram(r, bins=bins)
        g_r += hist
        
    r_values = bins[:-1] + dr / 2.0
    shell_volumes = 4.0 * np.pi * r_values**2 * dr
    n_ideal = density * shell_volumes
    
    # CORRECTED LINE: Removed the erroneous extra factor of (num_atoms - 1)
    g_r /= (num_atoms * n_ideal)

    return r_values, g_r, density


def fourier_transform_3d(r, func_r, k):
    """Performs a 3D Fourier transform on an isotropic function f(r)."""
    integrand = 4 * np.pi * r**2 * func_r * np.sinc(k[:, np.newaxis] * r / np.pi)
    f_k = np.trapezoid(integrand, r, axis=1)
    return f_k

def inverse_fourier_transform_3d(k, func_k, r):
    """Performs an inverse 3D Fourier transform on an isotropic function f(k)."""
    integrand = (1 / (2 * np.pi**2)) * k**2 * func_k * np.sinc(r[:, np.newaxis] * k / np.pi)
    f_r = np.trapezoid(integrand, k, axis=1)
    return f_r


def get_cache_info() -> Dict[str, Any]:
    """Get information about cached data."""
    cache_root = _cache_manager.cache_root
    if not cache_root.exists():
        return {"total_size": 0, "cached_files": []}

    cached_operations = []
    total_size = 0

    # Find all cache directories with meta.json files
    for meta_file in cache_root.rglob("meta.json"):
        try:
            cache_dir = meta_file.parent
            data_file = cache_dir / "data.npz"
            
            # Get the actual data file size
            data_size = 0
            if data_file.exists():
                data_size = data_file.stat().st_size
                total_size += data_size
            
            # Get metadata
            with open(meta_file, "r") as f:
                meta = json.load(f)
            
            cached_operations.append(
                {
                    "operation": meta.get("operation", "unknown"),
                    "path": str(cache_dir),
                    "size_mb": data_size / (1024**2),
                    "data_file": str(data_file) if data_file.exists() else "missing",
                }
            )
        except Exception as e:
            logger.warning(f"Failed to process cache metadata {meta_file}: {e}")

    return {"total_size_mb": total_size / (1024**2), "cached_operations": cached_operations}
