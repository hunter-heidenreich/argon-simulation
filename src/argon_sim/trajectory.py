"""
LAMMPS trajectory file reading and parsing.

This module handles loading and parsing LAMMPS trajectory files,
extracting positions, velocities, and box dimensions.
"""

import logging
from typing import Tuple

import numpy as np

from argon_sim.caching import cached_computation

logger = logging.getLogger(__name__)


def _parse_lammps_frame(
    lines: list, line_idx: int
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single LAMMPS trajectory frame.

    Args:
        lines: List of lines from the trajectory file
        line_idx: Current line index

    Returns:
        Tuple of (next_line_idx, positions, velocities, box_dims)
    """
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

    This function parses a LAMMPS trajectory file and extracts positions,
    velocities, and box dimensions for all frames. Data is automatically
    cached for faster subsequent access.

    Args:
        filename: Path to the LAMMPS trajectory file.

    Returns:
        Tuple of (positions, velocities, box_dimensions) arrays.
        - positions: shape (n_frames, n_atoms, 3)
        - velocities: shape (n_frames, n_atoms, 3)
        - box_dims: shape (n_frames, 3)
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
    """Extract atomic velocities from trajectory file.

    Args:
        filename: Path to the LAMMPS trajectory file

    Returns:
        Velocity array with shape (n_frames, n_atoms, 3)
    """
    _, velocities, _ = load_trajectory_data(filename)
    return velocities
