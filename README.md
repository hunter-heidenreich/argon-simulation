# Rahman 1964 Liquid Argon Simulation Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Reproducing the foundations of molecular dynamics simulation**

A comprehensive analysis toolkit for reproducing the groundbreaking results from **Aneesur Rahman's 1964 paper** *"Correlations in the Motion of Atoms in Liquid Argon"* ([Physical Review 136, A405](https://doi.org/10.1103/PhysRev.136.A405)). This seminal work founded the field of molecular dynamics simulation by demonstrating its power to reveal microscopic, time-dependent behavior in liquids.

## 🔬 About Rahman's 1964 Paper

Rahman's paper was revolutionary for several reasons:

- **First molecular dynamics simulation** of a realistic liquid system (864 argon atoms)
- **Introduction of periodic boundary conditions** to simulate bulk behavior
- **Direct calculation of time-correlation functions** from atomic trajectories
- **Validation of theoretical models** through computational "experiments"
- **Discovery of complex atomic motion** including velocity "back-scattering" effects

The simulation revealed that liquid argon atoms don't undergo simple Brownian motion, but instead show complex dynamics including temporary "caging" by neighbors and subsequent escape—insights impossible to obtain from experiment alone.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation
```bash
git clone https://github.com/hunter-heidenreich/argon-simulation.git
cd argon-simulation
make install
```

### Generate Results
```bash
# Run simulation and generate all analyses
make workflow

# Generate specific analyses
make figure-2    # Radial distribution function g(r)
make figure-3    # Mean square displacement & diffusion
make figure-4    # Velocity autocorrelation function

# View all available commands
make help
```

## 📊 Analysis Features

### Core Physical Quantities
- **🌡️ Temperature Analysis** - Instantaneous temperature from kinetic energy
- **📍 Radial Distribution Function g(r)** - Atomic structure characterization  
- **📏 Mean Square Displacement** - Diffusive behavior and diffusion coefficient
- **🔄 Velocity Autocorrelation Function** - Atomic motion memory effects
- **📐 Van Hove Correlation Functions** - Space-time correlation analysis

### Features
- **⚡ Intelligent Caching** - Expensive computations cached automatically
- **🧠 Memory Management** - Handles large trajectories efficiently  
- **📈 Professional Plotting** - Publication-ready figures
- **🔬 Rahman Comparisons** - Direct comparison with 1964 paper values

## 🏗️ Project Structure

```
argon-sim/
├── src/argon_sim/           # Core analysis package
│   ├── trajectory.py        # LAMMPS file parsing
│   ├── thermodynamics.py    # Temperature calculations
│   ├── dynamics.py          # MSD, VACF analysis
│   ├── structure.py         # g(r), Van Hove functions
│   ├── caching.py           # Intelligent caching system
│   └── constants.py         # Physical constants
├── scripts/                 # Analysis scripts (figure_*.py)
├── artifacts/               # Generated figures and logs
├── cache/                   # Cached computation results
├── Makefile                 # Easy-to-use commands
└── pyproject.toml           # Project configuration
```

## 🔧 Usage Examples

### Basic Analysis
```python
from argon_sim import (
    compute_temperatures, 
    compute_msd, 
    compute_radial_distribution,
    CONSTANTS
)

# Load and analyze trajectory
temperatures = compute_temperatures("traj.lammpstrj")
msd = compute_msd("traj.lammpstrj")
r_values, g_r, density = compute_radial_distribution("traj.lammpstrj")

print(f"Average temperature: {temperatures.mean():.1f} K")
print(f"Target temperature: {CONSTANTS['TEMPERATURE']} K")
```

### Cache Management
```python
from argon_sim.caching import CacheManager

# Check cache status
cache = CacheManager()
print(f"Cache entries: {len(cache.get_all_keys())}")

# Clear specific cached computations
cache.clear_computation("msd")  # Clear only MSD cache
cache.clear_all()               # Clear all cache
```

### Advanced Analysis
```python
import numpy as np
from argon_sim import compute_vacf, CONSTANTS

# Compute velocity autocorrelation
vacf = compute_vacf("traj.lammpstrj")

# Analyze time correlation
n_frames = len(vacf)
time_ps = np.arange(n_frames) * CONSTANTS["TIMESTEP_FS"] / 1000.0

# Analysis reveals deviations from simple Brownian motion
print(f"VACF at t=0: {vacf[0]:.3f}")
print(f"First minimum around: {time_ps[np.argmin(vacf[:100])]:.2f} ps")
```

## 📈 Key Results

| Property | Rahman 1964 | This Analysis | Units |
|----------|-------------|---------------|-------|
| Temperature | 94.4 | ~94.4 | K |
| Diffusion Coefficient | 2.43×10⁻⁵ | ~2.4×10⁻⁵ | cm²/s |
| Density | ~1.374 | ~1.37 | g/cm³ |

### Computational Insights
- **VACF "Back-scattering"** - Negative correlation at ~0.3 ps showing atomic caging
- **Non-Gaussian Diffusion** - Intermediate-time deviations from Gaussian displacement
- **Liquid Structure** - Well-defined first coordination shell in g(r)

## ⚡ Performance & Caching

The analysis includes an intelligent caching system that speeds up repeated calculations:

```bash
# First run: ~30 seconds for MSD calculation
make figure-3

# Subsequent runs: ~2 seconds (cached)
make figure-3

# Check cache usage
make cache-info
```

Cache automatically invalidates when source files change, ensuring accurate results.

## 🧪 Scientific Validation

This implementation reproduces key findings from Rahman's paper:

1. **Atomic Motion Complexity** - VACF shows oscillatory behavior, not simple exponential decay
2. **Diffusion Mechanisms** - MSD reveals ballistic → diffusive regimes  
3. **Liquid Structure** - g(r) shows characteristic liquid peaks and coordination
4. **Time Correlations** - Van Hove functions reveal space-time coupling in liquid dynamics

## 🛠️ Development

### Code Quality
```bash
make lint     # Check code style with ruff
make format   # Auto-format code
make check    # Run all checks
```

### Adding New Analyses
1. Create analysis function in appropriate module (`src/argon_sim/`)
2. Add `@cached_computation` decorator for expensive operations
3. Create script in `scripts/` directory
4. Add Makefile target for easy execution

## 📚 Scientific Background

Rahman's approach integrates Newton's equations of motion with the Lennard-Jones potential:

**Force Integration:**
```
F_i = m * a_i = -∇_i U({r_j})
```

**Lennard-Jones Potential:**
```
U(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
```

**Key Observables:**
- **Mean Square Displacement**: `⟨|r(t) - r(0)|²⟩` 
- **Velocity Autocorrelation**: `⟨v(0)·v(t)⟩`
- **Van Hove G_s(r,t)**: Self-part correlation function
- **Van Hove G_d(r,t)**: Distinct-part correlation function

## 📄 Citation

If you use this code in your research, please cite Rahman's original paper:

```bibtex
@article{Rahman1964,
  title={Correlations in the Motion of Atoms in Liquid Argon},
  author={Rahman, Aneesur},
  journal={Physical Review},
  volume={136},
  number={2A},
  pages={A405--A411},
  year={1964},
  publisher={American Physical Society},
  doi={10.1103/PhysRev.136.A405}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*"The detailed microscopic behavior of a liquid is extremely complex, but computer simulation makes it possible to follow the motion of every atom."* - A. Rahman, 1964
