# Makefile for Argon Simulation Analysis
# Reproducing Rahman's 1964 Liquid Argon Molecular Dynamics Results

.PHONY: help install clean cache-info cache-clear figures tables all test lint format check simulate

# Default target
help:
	@echo "Rahman 1964 Argon Simulation Analysis"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install dependencies using uv"
	@echo "  simulate     - Run LAMMPS simulation (generates traj.lammpstrj)"
	@echo "  figures      - Generate all figures (1a through 8)"
	@echo "  tables       - Generate all tables"
	@echo "  all          - Generate all figures and tables"
	@echo "  clean        - Remove generated artifacts"
	@echo "  cache-info   - Show cache statistics"
	@echo "  cache-clear  - Clear all cached data"
	@echo "  test         - Run unit tests"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with ruff"
	@echo "  check        - Run all checks (lint + test)"
	@echo ""
	@echo "Individual figures:"
	@echo "  figure-1a    - Temperature vs time analysis"
	@echo "  figure-1b    - Velocity distribution analysis"
	@echo "  figure-2     - Radial distribution function g(r)"
	@echo "  figure-3     - Mean square displacement & diffusion"
	@echo "  figure-4     - Velocity autocorrelation function"
	@echo "  figure-5     - Van Hove G_s(r,t) self-correlation"
	@echo "  figure-6     - Van Hove G_d(r,t) distinct correlation"
	@echo "  figure-7     - Non-Gaussian behavior analysis"
	@echo "  figure-8     - Delayed convolution approximation"

# Installation
install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "Installation complete!"

# Simulation
simulate:
	@echo "Running LAMMPS simulation..."
	@echo "This will generate traj.lammpstrj for analysis"
	lmp_serial -in in.argon
	@echo "Simulation complete! Output trajectory: traj.lammpstrj"

# Cache management
cache-info:
	@echo "Cache information:"
	uv run python scripts/cache_info.py --info

cache-clear:
	@echo "Clearing cache..."
	uv run python scripts/cache_info.py --clear
	@echo "Cache cleared!"

# Clean artifacts
clean:
	@echo "Cleaning artifacts..."
	rm -rf artifacts/*.png artifacts/*.log
	@echo "Artifacts cleaned!"

# Individual figures
figure-1a:
	@echo "Generating Figure 1a: Temperature vs Time..."
	uv run python scripts/figure_1a.py

figure-1b:
	@echo "Generating Figure 1b: Velocity Distribution..."
	uv run python scripts/figure_1b.py

figure-2:
	@echo "Generating Figure 2: Radial Distribution Function..."
	uv run python scripts/figure_2.py

figure-3:
	@echo "Generating Figure 3: Mean Square Displacement..."
	uv run python scripts/figure_3.py

figure-4:
	@echo "Generating Figure 4: Velocity Autocorrelation Function..."
	uv run python scripts/figure_4.py

figure-5:
	@echo "Generating Figure 5: Van Hove G_s(r,t)..."
	uv run python scripts/figure_5.py

figure-6:
	@echo "Generating Figure 6: Van Hove G_d(r,t)..."
	uv run python scripts/figure_6.py

figure-7:
	@echo "Generating Figure 7: Non-Gaussian Behavior Analysis..."
	uv run python scripts/figure_7.py

figure-8:
	@echo "Generating Figure 8: Delayed Convolution Approximation..."
	uv run python scripts/figure_8.py

# Generate all figures
figures: figure-1a figure-1b figure-2 figure-3 figure-4 figure-5 figure-6 figure-7 figure-8
	@echo "All figures generated!"

# Tables
table-1:
	@echo "Generating Table 1: Summary Statistics..."
	uv run python scripts/table_1.py

tables: table-1
	@echo "All tables generated!"

# Generate everything
all: figures tables
	@echo "All analyses complete!"
	@echo "Results available in artifacts/ directory"

# Complete workflow from simulation to analysis
workflow: simulate all
	@echo "Complete workflow finished!"
	@echo "Generated trajectory and all analysis results"

# Development tools
lint:
	@echo "Running linting..."
	uv run ruff check src/ scripts/

format:
	@echo "Formatting code..."
	uv run ruff format src/ scripts/

check: lint
	@echo "All checks passed!"
