"""NPY file reader for vector datasets.

This module provides functionality to read NumPy .npy files containing
vector datasets, extract metadata, and sample vectors efficiently using
memory-mapping.
"""

from pathlib import Path
from typing import Any

import numpy as np


class NPYReader:
    """Reader for NumPy .npy files containing vector datasets.
    
    Supports:
    - Metadata extraction (shape, dtype, file size)
    - Memory-mapped reading for large files
    - Sampling API for previewing vectors
    """

    def __init__(self, file_path: str | Path, mmap_mode: str | None = "r") -> None:
        """Initialize the NPY reader.
        
        Args:
            file_path: Path to the .npy file.
            mmap_mode: Memory-map mode. Use 'r' for read-only, None to load into memory.
        """
        self.file_path = Path(file_path)
        self.mmap_mode = mmap_mode
        self._array: np.ndarray | None = None
        self._metadata: dict[str, Any] | None = None

    def get_metadata(self) -> dict[str, Any]:
        """Extract metadata from the NPY file without loading all data.
        
        Returns:
            Dictionary containing:
            - file_path: Path to the file
            - shape: Shape of the array
            - dtype: Data type of the array
            - ndim: Number of dimensions
            - vector_count: Number of vectors (first dimension)
            - dimension: Vector dimension (second dimension, if 2D)
            - file_size_bytes: Size of the file in bytes
            - file_size_mb: Size of the file in megabytes
        """
        if self._metadata is not None:
            return self._metadata

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        file_size = self.file_path.stat().st_size

        # Load with mmap to read metadata without loading all data
        array = np.load(str(self.file_path), mmap_mode="r")

        self._metadata = {
            "file_path": str(self.file_path),
            "format": "npy",
            "shape": array.shape,
            "dtype": str(array.dtype),
            "ndim": array.ndim,
            "vector_count": array.shape[0] if array.ndim >= 1 else 1,
            "dimension": array.shape[1] if array.ndim >= 2 else (array.shape[0] if array.ndim == 1 else 1),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
        }

        return self._metadata

    def load(self) -> np.ndarray:
        """Load the array from the file.
        
        Returns:
            The loaded NumPy array (memory-mapped if mmap_mode is set).
        """
        if self._array is None:
            self._array = np.load(str(self.file_path), mmap_mode=self.mmap_mode)
        return self._array

    def sample(self, start: int = 0, count: int = 10) -> np.ndarray:
        """Sample vectors from the dataset.
        
        Args:
            start: Starting index for sampling.
            count: Number of vectors to sample.
            
        Returns:
            NumPy array containing the sampled vectors.
        """
        array = self.load()
        if array.ndim == 1:
            return array[start:start + count]
        return array[start:start + count]

    def get_vector(self, index: int) -> np.ndarray:
        """Get a single vector by index.
        
        Args:
            index: Index of the vector to retrieve.
            
        Returns:
            The vector at the specified index.
        """
        array = self.load()
        return array[index]

    def __len__(self) -> int:
        """Return the number of vectors in the dataset."""
        metadata = self.get_metadata()
        return metadata["vector_count"]

    def close(self) -> None:
        """Close the reader and release resources."""
        if self._array is not None:
            # For mmap arrays, delete reference to unmap
            del self._array
            self._array = None
