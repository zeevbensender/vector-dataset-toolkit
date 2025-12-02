"""FBIN file reader for vector datasets.

This module provides functionality to read FBIN (binary float32 vector) files.
FBIN format is commonly used for storing dense vector embeddings.

FBIN Format Specification (based on ANN Benchmarks convention):
- Header: 8 bytes
  - Bytes 0-3: uint32 little-endian - number of vectors
  - Bytes 4-7: uint32 little-endian - dimension of each vector
- Data: (num_vectors * dimension * 4) bytes
  - float32 little-endian values, row-major order

Format Notes:
- This implementation follows the FBIN format used by ANN Benchmarks
  (https://github.com/erikbern/ann-benchmarks) and Big ANN Benchmarks.
- Some datasets may use big-endian byte order (rare but possible).
- Some variants may include additional metadata in the header.

TODOs for M2:
- Add endianness detection by checking if dimension value seems reasonable.
- Add support for detecting float16/float64 variants if header values seem invalid.
- Consider adding file magic number validation for stricter format checking.
"""

import struct
from pathlib import Path
from typing import Any

import numpy as np


# FBIN header size in bytes
FBIN_HEADER_SIZE = 8


class FBINReader:
    """Reader for FBIN (binary float32 vector) files.
    
    Supports:
    - Metadata extraction from header
    - Memory-mapped reading for large files
    - Sequential and random access to vectors
    
    Assumptions:
    - Standard FBIN format with 8-byte header (uint32 count, uint32 dim)
    - float32 data in little-endian byte order
    - Row-major vector storage
    
    Known Limitations (TODOs for M2):
    - No endianness detection (assumes little-endian)
    - No support for float16/float64 variants
    - No checksum or magic number validation
    - File size validation only warns, does not reject mismatched files
    """

    def __init__(self, file_path: str | Path, mmap_mode: bool = True) -> None:
        """Initialize the FBIN reader.
        
        Args:
            file_path: Path to the .fbin file.
            mmap_mode: If True, use memory-mapping for efficient large file access.
        """
        self.file_path = Path(file_path)
        self.mmap_mode = mmap_mode
        self._array: np.ndarray | None = None
        self._metadata: dict[str, Any] | None = None
        self._num_vectors: int = 0
        self._dimension: int = 0

    def _read_header(self) -> tuple[int, int]:
        """Read and parse the FBIN header.
        
        Returns:
            Tuple of (num_vectors, dimension).
        """
        with open(self.file_path, "rb") as f:
            header = f.read(FBIN_HEADER_SIZE)
            if len(header) < FBIN_HEADER_SIZE:
                raise ValueError(f"Invalid FBIN file: header too short ({len(header)} bytes)")
            
            num_vectors, dimension = struct.unpack("<II", header)
            return num_vectors, dimension

    def get_metadata(self) -> dict[str, Any]:
        """Extract metadata from the FBIN file header.
        
        Returns:
            Dictionary containing:
            - file_path: Path to the file
            - format: 'fbin'
            - vector_count: Number of vectors
            - dimension: Vector dimension
            - dtype: Data type (float32)
            - file_size_bytes: Size of the file in bytes
            - file_size_mb: Size of the file in megabytes
            - expected_size_bytes: Expected file size based on header
        """
        if self._metadata is not None:
            return self._metadata

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        file_size = self.file_path.stat().st_size
        self._num_vectors, self._dimension = self._read_header()
        
        expected_data_size = self._num_vectors * self._dimension * 4  # float32 = 4 bytes
        expected_file_size = FBIN_HEADER_SIZE + expected_data_size

        self._metadata = {
            "file_path": str(self.file_path),
            "format": "fbin",
            "vector_count": self._num_vectors,
            "dimension": self._dimension,
            "dtype": "float32",
            "shape": (self._num_vectors, self._dimension),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "expected_size_bytes": expected_file_size,
            "size_match": file_size == expected_file_size,
        }

        return self._metadata

    def load(self) -> np.ndarray:
        """Load the vector data from the file.
        
        Returns:
            NumPy array of shape (num_vectors, dimension) with float32 dtype.
        """
        if self._array is not None:
            return self._array

        metadata = self.get_metadata()
        
        if self.mmap_mode:
            # Memory-map the data portion
            self._array = np.memmap(
                str(self.file_path),
                dtype=np.float32,
                mode="r",
                offset=FBIN_HEADER_SIZE,
                shape=(metadata["vector_count"], metadata["dimension"]),
            )
        else:
            # Load entirely into memory
            with open(self.file_path, "rb") as f:
                f.seek(FBIN_HEADER_SIZE)
                data = f.read()
                self._array = np.frombuffer(data, dtype=np.float32).reshape(
                    metadata["vector_count"], metadata["dimension"]
                )

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
        return np.array(array[start:start + count])  # Copy from mmap

    def get_vector(self, index: int) -> np.ndarray:
        """Get a single vector by index.
        
        Args:
            index: Index of the vector to retrieve.
            
        Returns:
            The vector at the specified index.
        """
        array = self.load()
        return np.array(array[index])  # Copy from mmap

    def read_sequential(
        self, start: int = 0, count: int | None = None, chunk_size: int = 1000
    ):
        """Generator for sequential reading of vectors in chunks.
        
        This is memory-efficient for processing large files.
        
        Args:
            start: Starting index.
            count: Number of vectors to read (None for all remaining).
            chunk_size: Number of vectors per chunk.
            
        Yields:
            NumPy arrays of vectors, each of size chunk_size (or smaller for last chunk).
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
                offset = FBIN_HEADER_SIZE + current * metadata["dimension"] * 4
                f.seek(offset)
                data = f.read(chunk_count * metadata["dimension"] * 4)
                chunk = np.frombuffer(data, dtype=np.float32).reshape(
                    chunk_count, metadata["dimension"]
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
