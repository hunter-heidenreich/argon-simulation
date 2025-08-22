# Argon Simulation Analysis

This project contains analysis scripts for molecular dynamics simulations of liquid argon.

## Features

- **Intelligent Caching**: All expensive computations (trajectory loading, MSD calculations, temperature calculations) are automatically cached based on source file modification times
- **Mean Square Displacement Analysis**: Highly optimized MSD calculation with memory management for large datasets
- **Diffusion Coefficient Calculation**: Automated calculation of diffusion coefficients from MSD data
- **Temperature Analysis**: Instantaneous temperature calculations from velocity data

## Usage

### Running Analysis Scripts

```bash
# Generate temperature analysis plot
uv run python analysis/figure_1a.py

# Generate velocity distribution plot  
uv run python analysis/figure_1b.py

# Generate radial distribution function
uv run python analysis/figure_2.py

# Generate mean square displacement plot (with caching)
uv run python analysis/figure_3.py
```

### Cache Management

The analysis uses an intelligent caching system to avoid recomputing expensive operations:

```bash
# View cache information
uv run python analysis/cache_info.py --info

# Clear all cached data
uv run python analysis/cache_info.py --clear
```

**Benefits of Caching:**
- MSD calculations are cached, allowing you to re-run styling/plotting changes instantly
- Trajectory data is cached, speeding up all subsequent analyses
- Temperature calculations are cached for quick access
- Cache is automatically invalidated when source files change

### Cache Structure

Cached data is stored in the `cache/` directory with the following structure:
```
cache/
└── traj.lammpstrj/
    ├── trajectory_data_<hash>/
    │   ├── data.npz
    │   └── meta.json
    ├── msd_<hash>/
    │   ├── data.npz
    │   └── meta.json
    └── temperatures_<hash>/
        ├── data.npz
        └── meta.json
```

## Analysis Scripts

- `figure_1a.py` - Temperature vs time analysis
- `figure_1b.py` - Velocity distribution analysis  
- `figure_2.py` - Radial distribution function
- `figure_3.py` - Mean square displacement and diffusion coefficient
- `figure_4.py` - Additional analysis
- `table_1.py` - Summary statistics
- `utils.py` - Shared utilities and caching framework
- `cache_info.py` - Cache management utility

## Dependencies

This project uses `uv` for dependency management. All required packages are specified in `pyproject.toml`.
