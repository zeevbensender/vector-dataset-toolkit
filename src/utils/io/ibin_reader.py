"""IBIN file reader for vector datasets.

This module provides functionality to read IBIN (binary int32 vector) files.
IBIN format is commonly used for storing ground truth indices in ANN benchmarks.

IBIN Format Specification:
- Header: 8 bytes
  - Bytes 0-3: uint32 little-endian - number of vectors/queries
  - Bytes 4-7: uint32 little-endian - number of neighbors per query (k)
- Data: (num_vectors * k * 4) bytes
  - int32 little-endian values, row-major order

TODO: This implementation is based on the common IBIN format used in ANN benchmarks.
The format typically stores ground truth neighbor indices for each query.
If different variants exist, additional format detection may be needed.
"""

import struct
from pathlib import Path
from typing import Any

import numpy as np


# IBIN header size in bytes
IBIN_HEADER_SIZE = 8


class IBINReader:
    """Reader for IBIN (binary int32 vector) files.
    
    Supports:
    - Metadata extraction from header
    - Memory-mapped reading for large files
    - Sequential and random access to index vectors
    
    Assumptions/TODOs:
    - Assumes standard IBIN format with 8-byte header (uint32 count, uint32 k)
    - Assumes int32 data in little-endian format
    - Typically used for ground truth neighbor indices
    - TODO: Add support for detecting alternative IBIN variants if they exist
    - TODO: Validate relationship between IBIN and corresponding FBIN files
    """

    def __init__(self, file_path: str | Path, mmap_mode: bool = True) -> None:
        """Initialize the IBIN reader.
        
        Args:
            file_path: Path to the .ibin file.
            mmap_mode: If True, use memory-mapping for efficient large file access.
        """
        self.file_path = Path(file_path)
        self.mmap_mode = mmap_mode
        self._array: np.ndarray | None = None
        self._metadata: dict[str, Any] | None = None
        self._num_vectors: int = 0
        self._k: int = 0

    def _read_header(self) -> tuple[int, int]:
        """Read and parse the IBIN header.
        
        Returns:
            Tuple of (num_vectors, k).
        """
        with open(self.file_path, "rb") as f:
            header = f.read(IBIN_HEADER_SIZE)
            if len(header) < IBIN_HEADER_SIZE:
                raise ValueError(f"Invalid IBIN file: header too short ({len(header)} bytes)")
            
            num_vectors, k = struct.unpack("<II", header)
            return num_vectors, k

    def get_metadata(self) -> dict[str, Any]:
        """Extract metadata from the IBIN file header.
        
        Returns:
            Dictionary containing:
            - file_path: Path to the file
            - format: 'ibin'
            - vector_count: Number of query/result vectors
            - k: Number of neighbors per vector
            - dtype: Data type (int32)
            - file_size_bytes: Size of the file in bytes
            - file_size_mb: Size of the file in megabytes
            - expected_size_bytes: Expected file size based on header
        """
        if self._metadata is not None:
            return self._metadata

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        file_size = self.file_path.stat().st_size
        self._num_vectors, self._k = self._read_header()
        
        expected_data_size = self._num_vectors * self._k * 4  # int32 = 4 bytes
        expected_file_size = IBIN_HEADER_SIZE + expected_data_size

        self._metadata = {
            "file_path": str(self.file_path),
            "format": "ibin",
            "vector_count": self._num_vectors,
            "k": self._k,
            "dimension": self._k,  # For consistency with other readers
            "dtype": "int32",
            "shape": (self._num_vectors, self._k),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "expected_size_bytes": expected_file_size,
            "size_match": file_size == expected_file_size,
        }

        return self._metadata

    def load(self) -> np.ndarray:
        """Load the index data from the file.
        
        Returns:
            NumPy array of shape (num_vectors, k) with int32 dtype.
        """
        if self._array is not None:
            return self._array

        metadata = self.get_metadata()
        
        if self.mmap_mode:
            # Memory-map the data portion
            self._array = np.memmap(
                str(self.file_path),
                dtype=np.int32,
                mode="r",
                offset=IBIN_HEADER_SIZE,
                shape=(metadata["vector_count"], metadata["k"]),
            )
        else:
            # Load entirely into memory
            with open(self.file_path, "rb") as f:
                f.seek(IBIN_HEADER_SIZE)
                data = f.read()
                self._array = np.frombuffer(data, dtype=np.int32).reshape(
                    metadata["vector_count"], metadata["k"]
                )

        return self._array

    def sample(self, start: int = 0, count: int = 10) -> np.ndarray:
        """Sample index vectors from the dataset.
        
        Args:
            start: Starting index for sampling.
            count: Number of vectors to sample.
            
        Returns:
            NumPy array containing the sampled index vectors.
        """
        array = self.load()
        return np.array(array[start:start + count])  # Copy from mmap

    def get_vector(self, index: int) -> np.ndarray:
        """Get a single index vector by index.
        
        Args:
            index: Index of the vector to retrieve.
            
        Returns:
            The index vector at the specified index.
        """
        array = self.load()
        return np.array(array[index])  # Copy from mmap

    def get_neighbors(self, query_index: int) -> np.ndarray:
        """Get the neighbor indices for a specific query.
        
        This is an alias for get_vector, with semantically meaningful naming
        for ground truth data.
        
        Args:
            query_index: Index of the query.
            
        Returns:
            Array of neighbor indices for the query.
        """
        return self.get_vector(query_index)

    def read_sequential(
        self, start: int = 0, count: int | None = None, chunk_size: int = 1000
    ):
        """Generator for sequential reading of index vectors in chunks.
        
        This is memory-efficient for processing large files.
        
        Args:
            start: Starting index.
            count: Number of vectors to read (None for all remaining).
            chunk_size: Number of vectors per chunk.
            
        Yields:
            NumPy arrays of index vectors, each of size chunk_size (or smaller for last chunk).
        """
        metadata = self.get_metadata()
        total = metadata["vector_count"]
        
        if count is None:
            count = total - start
        
        end = min(start + count, total)
        
        with open(self.file_path, "rb") as f:
            current = start
            while current < end:
                chunk_count = min(chunk_size, end - current)
                offset = IBIN_HEADER_SIZE + current * metadata["k"] * 4
                f.seek(offset)
                data = f.read(chunk_count * metadata["k"] * 4)
                chunk = np.frombuffer(data, dtype=np.int32).reshape(
                    chunk_count, metadata["k"]
                )
                yield chunk
                current += chunk_count

    def __len__(self) -> int:
        """Return the number of vectors in the dataset."""
        metadata = self.get_metadata()
        return metadata["vector_count"]

    def close(self) -> None:
        """Close the reader and release resources."""
        if self._array is not None:
            del self._array
            self._array = None
