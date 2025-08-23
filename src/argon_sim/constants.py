"""
Physical constants for Argon molecular dynamics simulations.

This module contains all the physical constants used in the simulation analysis,
based on the parameters from Rahman's 1964 paper and standard physical constants.
"""

# Physical constants for Argon simulations
CONSTANTS = {
    # Argon properties
    "MASS_AMU": 39.95,  # Argon atomic mass in AMU
    "SIGMA": 3.4,  # Lennard-Jones sigma parameter in Angstroms
    # Fundamental constants
    "KB_SI": 1.38065e-23,  # Boltzmann constant in J/K
    "AMU_to_KG": 1.66054e-27,  # AMU to kg conversion
    "HBAR_SI": 1.05457e-34,  # Reduced Planck constant in J·s
    # Unit conversions
    "A_fs_to_m_s": 1.0e5,  # Angstrom/femtosecond to m/s conversion
    # Simulation parameters
    "TIMESTEP_FS": 2.0,  # Simulation timestep in femtoseconds
    "TEMPERATURE": 94.4,  # Simulation temperature in K (Rahman's value)
    # Reference values from Rahman's paper
    "D_PAPER": 2.43e-5,  # Diffusion coefficient in cm^2/s
}

# Derived constants
CONSTANTS["MASS_KG"] = CONSTANTS["MASS_AMU"] * CONSTANTS["AMU_to_KG"]  # kg
CONSTANTS["TIMESTEP_S"] = CONSTANTS["TIMESTEP_FS"] * 1e-15  # seconds
