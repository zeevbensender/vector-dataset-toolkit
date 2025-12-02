"""HDF5 file reader for vector datasets.

This module provides functionality to read HDF5 files containing vector
datasets, list groups and datasets, extract metadata, and sample vectors.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class HDF5Reader:
    """Reader for HDF5 (.h5, .hdf5) files containing vector datasets.
    
    Supports:
    - Listing groups and datasets in the file
    - Metadata extraction for each dataset
    - Sampling API for previewing vectors
    """

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the HDF5 reader.
        
        Args:
            file_path: Path to the HDF5 file.
        """
        self.file_path = Path(file_path)
        self._file: h5py.File | None = None
        self._metadata: dict[str, Any] | None = None

    def _ensure_open(self) -> h5py.File:
        """Ensure the file is open and return the file handle."""
        if self._file is None:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
            self._file = h5py.File(str(self.file_path), "r")
        return self._file

    def list_contents(self) -> dict[str, list[str]]:
        """List all groups and datasets in the HDF5 file.
        
        Returns:
            Dictionary with 'groups' and 'datasets' keys containing lists of paths.
        """
        f = self._ensure_open()
        groups: list[str] = []
        datasets: list[str] = []

        def visitor(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Group):
                groups.append(name)
            elif isinstance(obj, h5py.Dataset):
                datasets.append(name)

        f.visititems(visitor)
        return {"groups": groups, "datasets": datasets}

    def get_metadata(self, dataset_path: str | None = None) -> dict[str, Any]:
        """Extract metadata from the HDF5 file.
        
        Args:
            dataset_path: Optional path to a specific dataset. If None, returns
                         file-level metadata with info about all datasets.
        
        Returns:
            Dictionary containing metadata about the file or specific dataset.
        """
        f = self._ensure_open()
        file_size = self.file_path.stat().st_size

        if dataset_path is not None:
            # Get metadata for a specific dataset
            if dataset_path not in f:
                raise KeyError(f"Dataset not found: {dataset_path}")
            
            dataset = f[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"Path is not a dataset: {dataset_path}")

            return {
                "dataset_path": dataset_path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "ndim": dataset.ndim,
                "vector_count": dataset.shape[0] if dataset.ndim >= 1 else 1,
                "dimension": dataset.shape[1] if dataset.ndim >= 2 else (
                    dataset.shape[0] if dataset.ndim == 1 else 1
                ),
                "chunks": dataset.chunks,
                "compression": dataset.compression,
            }

        # Get file-level metadata
        contents = self.list_contents()
        
        dataset_info = []
        for ds_path in contents["datasets"]:
            ds = f[ds_path]
            if isinstance(ds, h5py.Dataset):
                dataset_info.append({
                    "path": ds_path,
                    "shape": ds.shape,
                    "dtype": str(ds.dtype),
                })

        return {
            "file_path": str(self.file_path),
            "format": "hdf5",
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "groups": contents["groups"],
            "datasets": dataset_info,
            "dataset_count": len(contents["datasets"]),
        }

    def sample(
        self, dataset_path: str, start: int = 0, count: int = 10
    ) -> np.ndarray:
        """Sample vectors from a dataset.
        
        Args:
            dataset_path: Path to the dataset within the HDF5 file.
            start: Starting index for sampling.
            count: Number of vectors to sample.
            
        Returns:
            NumPy array containing the sampled vectors.
        """
        f = self._ensure_open()
        
        if dataset_path not in f:
            raise KeyError(f"Dataset not found: {dataset_path}")
        
        dataset = f[dataset_path]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Path is not a dataset: {dataset_path}")

        return dataset[start:start + count]

    def get_vector(self, dataset_path: str, index: int) -> np.ndarray:
        """Get a single vector by index from a dataset.
        
        Args:
            dataset_path: Path to the dataset within the HDF5 file.
            index: Index of the vector to retrieve.
            
        Returns:
            The vector at the specified index.
        """
        f = self._ensure_open()
        
        if dataset_path not in f:
            raise KeyError(f"Dataset not found: {dataset_path}")
        
        dataset = f[dataset_path]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Path is not a dataset: {dataset_path}")

        return dataset[index]

    def get_dataset(self, dataset_path: str) -> h5py.Dataset:
        """Get a dataset object for direct access.
        
        Args:
            dataset_path: Path to the dataset within the HDF5 file.
            
        Returns:
            The h5py Dataset object.
        """
        f = self._ensure_open()
        
        if dataset_path not in f:
            raise KeyError(f"Dataset not found: {dataset_path}")
        
        dataset = f[dataset_path]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Path is not a dataset: {dataset_path}")

        return dataset

    def close(self) -> None:
        """Close the HDF5 file and release resources."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "HDF5Reader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
