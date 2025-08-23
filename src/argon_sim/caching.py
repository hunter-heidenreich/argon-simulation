"""
Intelligent caching system for molecular dynamics analysis.

This module provides a robust caching framework that automatically manages
expensive computations based on source file modification times.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class CacheConstants:
    """Constants used throughout the caching system."""

    CACHE_KEY_LENGTH = 16
    DATA_FILE_NAME = "data.npz"
    META_FILE_NAME = "meta.json"
    SIZE_CONVERSION_FACTOR = 1024**2  # bytes to MB


class DataSerializer(ABC):
    """Abstract base class for data serialization strategies."""

    @abstractmethod
    def serialize(self, data: Any) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Serialize data for caching."""
        pass

    @abstractmethod
    def deserialize(self, cache_data: dict[str, npt.NDArray[np.floating[Any]]]) -> Any:
        """Deserialize cached data."""
        pass


class TrajectoryDataSerializer(DataSerializer):
    """Serializer for trajectory data (positions, velocities, box_dims)."""

    def serialize(self, data: tuple) -> dict[str, npt.NDArray[np.floating[Any]]]:
        positions, velocities, box_dims = data
        return {
            "positions": positions,
            "velocities": velocities,
            "box_dims": box_dims,
        }

    def deserialize(
        self, cache_data: dict[str, npt.NDArray[np.floating[Any]]]
    ) -> tuple:
        return (
            cache_data["positions"],
            cache_data["velocities"],
            cache_data["box_dims"],
        )


class SimpleDataSerializer(DataSerializer):
    """Serializer for simple single-array data."""

    def __init__(self, key_name: str):
        self.key_name = key_name

    def serialize(
        self, data: npt.NDArray[np.floating[Any]]
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        return {self.key_name: data}

    def deserialize(
        self, cache_data: dict[str, npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        return cache_data[self.key_name]


class RadialDistributionSerializer(DataSerializer):
    """Serializer for radial distribution function data (r_values, g_r, density)."""

    def serialize(self, data: tuple) -> dict[str, npt.NDArray[np.floating[Any]]]:
        r_values, g_r, density = data
        return {
            "r_values": r_values,
            "g_r": g_r,
            "density": np.array([density]),  # Store scalar as array
        }

    def deserialize(
        self, cache_data: dict[str, npt.NDArray[np.floating[Any]]]
    ) -> tuple:
        return (
            cache_data["r_values"],
            cache_data["g_r"],
            float(cache_data["density"][0]),  # Extract scalar from array
        )


class CorrelationFunctionSerializer(DataSerializer):
    """Serializer for correlation function dictionaries (gdt, gs)."""

    def serialize(self, data: dict) -> dict[str, npt.NDArray[np.floating[Any]]]:
        serialized = {}
        for time_lag, (r_values, corr_values) in data.items():
            # Use string keys for numpy savez compatibility
            r_key = f"r_values_{time_lag}"
            corr_key = f"corr_values_{time_lag}"
            serialized[r_key] = r_values
            serialized[corr_key] = corr_values

        # Store the time lags for reconstruction
        serialized["time_lags"] = np.array(list(data.keys()))
        return serialized

    def deserialize(self, cache_data: dict[str, npt.NDArray[np.floating[Any]]]) -> dict:
        result = {}
        time_lags = cache_data["time_lags"]

        for time_lag in time_lags:
            r_key = f"r_values_{time_lag}"
            corr_key = f"corr_values_{time_lag}"
            result[float(time_lag)] = (cache_data[r_key], cache_data[corr_key])

        return result


class GenericDataSerializer(DataSerializer):
    """Serializer for generic data types."""

    def serialize(self, data: Any) -> dict[str, npt.NDArray[np.floating[Any]]]:
        if isinstance(data, dict):
            return data
        else:
            return {"result": data}

    def deserialize(self, cache_data: dict[str, npt.NDArray[np.floating[Any]]]) -> Any:
        return cache_data


# Registry of operation serializers
OPERATION_SERIALIZERS: dict[str, DataSerializer] = {
    "trajectory_data": TrajectoryDataSerializer(),
    "temperatures": SimpleDataSerializer("temperatures"),
    "msd": SimpleDataSerializer("msd"),
    "vacf": SimpleDataSerializer("vacf"),
    "displacement_moments": SimpleDataSerializer("displacement_moments"),
    "gr": RadialDistributionSerializer(),
    "gdt": CorrelationFunctionSerializer(),
    "gs": CorrelationFunctionSerializer(),
}


# Type definitions for better type safety
class CacheMetadata(TypedDict):
    """Metadata stored with cached data."""

    source_mtime: float
    operation: str
    parameters: dict[str, Any]


class CacheOperationInfo(TypedDict):
    """Information about a cached operation."""

    operation: str
    path: str
    size_mb: float
    data_file: str


class CacheInfo(TypedDict):
    """Overall cache information."""

    total_size_mb: float
    cached_operations: list[CacheOperationInfo]


# Type alias for numpy arrays stored in cache
CacheData = dict[str, npt.NDArray[np.floating[Any]]]


class CacheManager:
    """General-purpose caching system for computed data."""

    def __init__(self, cache_root: str = "cache"):
        """Initialize cache manager.

        Args:
            cache_root: Root directory for cache storage
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(exist_ok=True)

    def _get_cache_key(self, source_path: Path, operation: str, **kwargs) -> str:
        """Generate a unique cache key based on source file and operation parameters.

        Args:
            source_path: Path to source file
            operation: Name of the operation
            **kwargs: Additional parameters affecting the computation

        Returns:
            Unique cache key string
        """
        key_data = f"{source_path.name}_{operation}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()[
            : CacheConstants.CACHE_KEY_LENGTH
        ]

    def _get_cache_dir(self, source_path: Path, operation: str, **kwargs) -> Path:
        """Get cache directory for a specific operation.

        Args:
            source_path: Path to source file
            operation: Name of the operation
            **kwargs: Additional parameters affecting the computation

        Returns:
            Path to cache directory
        """
        cache_key = self._get_cache_key(source_path, operation, **kwargs)
        cache_dir = self.cache_root / source_path.name / f"{operation}_{cache_key}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _is_cache_valid(self, cache_dir: Path, source_path: Path) -> bool:
        """Check if cached data is valid based on source file modification time.

        Args:
            cache_dir: Path to cache directory
            source_path: Path to source file

        Returns:
            True if cache is valid, False otherwise
        """
        meta_file = cache_dir / CacheConstants.META_FILE_NAME
        if not meta_file.exists():
            return False

        try:
            src_mtime = source_path.stat().st_mtime
            with open(meta_file) as f:
                meta = json.load(f)
            cached_mtime = meta.get("source_mtime")
            return cached_mtime is not None and cached_mtime >= src_mtime
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    def get_cached_data(
        self, source_path: Path, operation: str, **kwargs
    ) -> CacheData | None:
        """Retrieve cached data if valid.

        Args:
            source_path: Path to source file
            operation: Name of the operation
            **kwargs: Additional parameters affecting the computation

        Returns:
            Cached data dictionary if valid, None otherwise
        """
        cache_dir = self._get_cache_dir(source_path, operation, **kwargs)

        if not self._is_cache_valid(cache_dir, source_path):
            return None

        try:
            data_file = cache_dir / CacheConstants.DATA_FILE_NAME
            if data_file.exists():
                logger.info(f"Loading cached {operation} data from {data_file}")
                with np.load(data_file) as npz:
                    return dict(npz)
        except Exception as e:
            logger.warning(f"Failed to load cache for {operation}: {e}")

        return None

    def save_cached_data(
        self, source_path: Path, operation: str, data: CacheData, **kwargs
    ) -> None:
        """Save computed data to cache.

        Args:
            source_path: Path to source file
            operation: Name of the operation
            data: Data dictionary to cache
            **kwargs: Additional parameters affecting the computation
        """
        cache_dir = self._get_cache_dir(source_path, operation, **kwargs)

        try:
            # Save data
            data_file = cache_dir / CacheConstants.DATA_FILE_NAME
            np.savez_compressed(data_file, **data)

            # Save metadata
            meta_file = cache_dir / CacheConstants.META_FILE_NAME
            meta: CacheMetadata = {
                "source_mtime": source_path.stat().st_mtime,
                "operation": operation,
                "parameters": kwargs,
            }
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"Cached {operation} data to {data_file}")
        except Exception as e:
            logger.warning(f"Failed to cache {operation} data: {e}")

    def clear_cache(self, operation: str | None = None) -> None:
        """Clear cached data.

        Args:
            operation: Specific operation to clear, or None to clear all
        """
        import shutil

        if operation is None:
            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
                self.cache_root.mkdir(exist_ok=True)
                logger.info("Cleared all cached data")
        else:
            # Clear specific operation caches
            for cache_dir in self.cache_root.rglob(f"{operation}_*"):
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
            logger.info(f"Cleared cached data for operation: {operation}")

    def get_cache_info(self) -> CacheInfo:
        """Get information about cached data.

        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        cached_operations = []

        if not self.cache_root.exists():
            return {"total_size_mb": 0, "cached_operations": []}

        for cache_dir in self.cache_root.rglob("*/"):
            if not cache_dir.is_dir():
                continue

            meta_file = cache_dir / CacheConstants.META_FILE_NAME
            data_file = cache_dir / CacheConstants.DATA_FILE_NAME

            if not meta_file.exists():
                continue

            try:
                data_size = data_file.stat().st_size if data_file.exists() else 0
                total_size += data_size

                with open(meta_file) as f:
                    meta = json.load(f)

                cached_operations.append(
                    {
                        "operation": meta.get("operation", "unknown"),
                        "path": str(cache_dir),
                        "size_mb": data_size / CacheConstants.SIZE_CONVERSION_FACTOR,
                        "data_file": str(data_file)
                        if data_file.exists()
                        else "missing",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to process cache metadata {meta_file}: {e}")

        return {
            "total_size_mb": total_size / CacheConstants.SIZE_CONVERSION_FACTOR,
            "cached_operations": cached_operations,
        }


# Global cache manager instance
_cache_manager = CacheManager()


def _get_serializer(operation: str) -> DataSerializer:
    """Get the appropriate serializer for an operation.

    Args:
        operation: Name of the operation

    Returns:
        DataSerializer instance for the operation
    """
    return OPERATION_SERIALIZERS.get(operation, GenericDataSerializer())


def _load_from_cache(source_path: Path, operation: str, cache_params: dict) -> Any:
    """Load data from cache if available and valid.

    Args:
        source_path: Path to source file
        operation: Name of the operation
        cache_params: Cache parameters

    Returns:
        Cached data or None if not available
    """
    cached_data = _cache_manager.get_cached_data(source_path, operation, **cache_params)

    if cached_data is not None:
        serializer = _get_serializer(operation)
        return serializer.deserialize(cached_data)

    return None


def _save_to_cache(
    source_path: Path, operation: str, result: Any, cache_params: dict
) -> None:
    """Save computation result to cache.

    Args:
        source_path: Path to source file
        operation: Name of the operation
        result: Result to cache
        cache_params: Cache parameters
    """
    if result is not None:
        serializer = _get_serializer(operation)
        cache_data = serializer.serialize(result)
        _cache_manager.save_cached_data(
            source_path, operation, cache_data, **cache_params
        )


def cached_computation(operation: str, **cache_kwargs):
    """Decorator for caching expensive computations based on source files.

    Args:
        operation: Name of the operation for cache identification
        **cache_kwargs: Additional cache parameters

    Returns:
        Decorated function with caching capabilities
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(filename: str | Path, *args, **kwargs):
            source_path = Path(filename)

            if not source_path.exists():
                logger.error(f"Source file '{filename}' not found.")
                return None

            # Try to load from cache
            cache_params = {**cache_kwargs, **kwargs}
            cached_result = _load_from_cache(source_path, operation, cache_params)

            if cached_result is not None:
                return cached_result

            # Compute fresh data
            logger.info(f"Computing {operation} for {filename}...")
            result = func(filename, *args, **kwargs)

            # Cache the result
            _save_to_cache(source_path, operation, result, cache_params)

            return result

        return wrapper

    return decorator


# Convenience functions for cache management
def clear_cache(operation: str | None = None) -> None:
    """Clear cached data."""
    _cache_manager.clear_cache(operation)


def get_cache_info() -> CacheInfo:
    """Get cache information."""
    return _cache_manager.get_cache_info()
