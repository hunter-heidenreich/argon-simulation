#!/usr/bin/env python3
"""
Cache information utility for the argon simulation project.
Shows cache usage and allows cache management.
"""

import argparse
from pathlib import Path
import shutil

from utils import get_cache_info


def main():
    parser = argparse.ArgumentParser(description="Manage analysis cache")
    parser.add_argument(
        "--clear", action="store_true", help="Clear all cached data"
    )
    parser.add_argument(
        "--info", action="store_true", help="Show cache information (default)"
    )
    
    args = parser.parse_args()
    
    if args.clear:
        cache_dir = Path("cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("✅ Cache cleared successfully")
        else:
            print("ℹ️  No cache directory found")
    else:
        # Show cache info by default
        cache_info = get_cache_info()
        
        print("📊 Cache Information")
        print("=" * 40)
        print(f"Total cache size: {cache_info['total_size_mb']:.2f} MB")
        print(f"Number of cached operations: {len(cache_info['cached_operations'])}")
        print()
        
        if cache_info['cached_operations']:
            print("Cached Operations:")
            print("-" * 40)
            for op in cache_info['cached_operations']:
                print(f"  • {op['operation']}: {op['size_mb']:.2f} MB")
                print(f"    Path: {op['path']}")
                print()
        else:
            print("No cached data found")


if __name__ == "__main__":
    main()
