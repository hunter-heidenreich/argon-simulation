#!/usr/bin/env python3
"""
Cache Management Utility

This script provides tools for managing the intelligent cache system
used to speed up expensive computations in the argon simulation analysis.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from argon_sim.caching import get_cache_info, clear_cache

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def display_cache_info():
    """Display detailed cache information."""
    logger.info("Analyzing cache data...")
    info = get_cache_info()

    print("\n" + "=" * 50)
    print("           CACHE INFORMATION")
    print("=" * 50)

    print(f"\nTotal Cache Size: {info['total_size_mb']:.2f} MB")
    print(f"Number of Cached Operations: {len(info['cached_operations'])}")

    if info["cached_operations"]:
        print("\nCached Operations:")
        print("-" * 70)
        print(f"{'Operation':<20} {'Size (MB)':<10} {'Status':<10} {'Path'}")
        print("-" * 70)

        for op in info["cached_operations"]:
            status = "✓ Valid" if op["data_file"] != "missing" else "✗ Missing"
            print(
                f"{op['operation']:<20} {op['size_mb']:<10.2f} {status:<10} {op['path']}"
            )
    else:
        print("\nNo cached data found.")

    print("\n" + "=" * 50)


def clear_cache_interactive():
    """Clear cache with user confirmation."""
    info = get_cache_info()

    if not info["cached_operations"]:
        logger.info("No cached data to clear.")
        return

    print(f"\nThis will clear {info['total_size_mb']:.2f} MB of cached data.")
    print("Cached operations will need to be recomputed on next run.")

    confirm = input("\nAre you sure you want to clear all cache? [y/N]: ").lower()

    if confirm in ["y", "yes"]:
        logger.info("Clearing cache...")
        clear_cache()
        logger.info("✓ Cache cleared successfully!")
    else:
        logger.info("Cache clearing cancelled.")


def clear_specific_operation(operation: str):
    """Clear cache for a specific operation."""
    logger.info(f"Clearing cache for operation: {operation}")
    clear_cache(operation)
    logger.info(f"✓ Cache cleared for '{operation}'")


def main():
    """Main function to handle command line arguments."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Manage cache for argon simulation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cache_info.py --info              # Show cache information
  python cache_info.py --clear             # Clear all cache (interactive)
  python cache_info.py --clear-operation msd  # Clear only MSD cache
  
Available operations to clear:
  - trajectory_data  (trajectory loading)
  - temperatures     (temperature calculations)
  - msd             (mean square displacement)
  - vacf            (velocity autocorrelation)
  - gr              (radial distribution)
  - gdt             (Van Hove G_d(r,t))
  - gs              (Van Hove G_s(r,t))
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--info", action="store_true", help="Display cache information")
    group.add_argument(
        "--clear", action="store_true", help="Clear all cached data (interactive)"
    )
    group.add_argument(
        "--clear-operation",
        metavar="OPERATION",
        help="Clear cache for specific operation",
    )

    args = parser.parse_args()

    try:
        if args.info:
            display_cache_info()
        elif args.clear:
            clear_cache_interactive()
        elif args.clear_operation:
            clear_specific_operation(args.clear_operation)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
