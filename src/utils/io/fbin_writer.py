"""FBIN file writer for vector datasets.

This module provides functionality to write FBIN (binary float32 vector) files.
The writer supports atomic writes via temporary files to ensure data integrity.

FBIN Format Specification:
- Header: 8 bytes
  - Bytes 0-3: uint32 little-endian - number of vectors
  - Bytes 4-7: uint32 little-endian - dimension of each vector
- Data: (num_vectors * dimension * 4) bytes
  - float32 little-endian values, row-major order

Safety Features:
- Writes to a temporary file first, then atomically replaces the destination
- Supports checksum generation for data verification
- Validates input data types and shapes

TODOs for future milestones:
- Add support for writing float16/float64 variants
- Add streaming write API for very large datasets
"""

import hashlib
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .fbin_reader import FBIN_HEADER_SIZE


class FBINWriter:
    """Writer for FBIN (binary float32 vector) files.
    
    Supports:
    - Writing vectors to FBIN format with proper header
    - Atomic writes via temporary files
    - Checksum generation for verification
    - Progress callbacks for UI integration
    
    Safety Guarantees:
    - Data is written to a temp file first, then atomically renamed
    - Original file is not modified until write is fully complete
    - On cancellation, the temp file is cleaned up
    """

    def __init__(
        self,
        output_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize the FBIN writer.
        
        Args:
            output_path: Path where the FBIN file will be written.
            progress_callback: Optional callback function(current, total) for progress updates.
        """
        self.output_path = Path(output_path)
        self.progress_callback = progress_callback
        self._cancelled = False
        self._temp_path: Path | None = None

    def cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True

    def _report_progress(self, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total)

    def _cleanup_temp(self) -> None:
        """Clean up any temporary file."""
        if self._temp_path and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except OSError:
                pass  # Best effort cleanup

    def write(
        self,
        data: np.ndarray,
        chunk_size: int = 10000,
        compute_checksum: bool = False,
    ) -> dict[str, Any]:
        """Write vector data to an FBIN file.
        
        The write is performed atomically: data is first written to a temporary
        file, then renamed to the final destination.
        
        Args:
            data: NumPy array of shape (num_vectors, dimension) with float32 dtype.
            chunk_size: Number of vectors to write per chunk for progress updates.
            compute_checksum: If True, compute and return SHA256 checksum of the data.
            
        Returns:
            Dictionary containing write statistics:
            - output_path: Path to the written file
            - vector_count: Number of vectors written
            - dimension: Vector dimension
            - file_size_bytes: Size of the written file
            - checksum: SHA256 hex digest (if compute_checksum=True)
            
        Raises:
            ValueError: If data is not a 2D array or has wrong dtype.
            RuntimeError: If write is cancelled.
        """
        self._cancelled = False
        
        # Validate input
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array, got shape {data.shape}")
        
        # Ensure float32 dtype
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        num_vectors, dimension = data.shape
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        fd, temp_path_str = tempfile.mkstemp(
            suffix=".fbin.tmp",
            dir=self.output_path.parent
        )
        self._temp_path = Path(temp_path_str)
        
        hasher = hashlib.sha256() if compute_checksum else None
        
        try:
            with os.fdopen(fd, "wb") as f:
                # Write header
                header = struct.pack("<II", num_vectors, dimension)
                f.write(header)
                if hasher:
                    hasher.update(header)
                
                # Write data in chunks
                written = 0
                while written < num_vectors:
                    if self._cancelled:
                        self._cleanup_temp()
                        raise RuntimeError("Write cancelled")
                    
                    chunk_end = min(written + chunk_size, num_vectors)
                    chunk_data = data[written:chunk_end].tobytes()
                    f.write(chunk_data)
                    if hasher:
                        hasher.update(chunk_data)
                    
                    written = chunk_end
                    self._report_progress(written, num_vectors)
            
            # Atomic rename
            self._temp_path.replace(self.output_path)
            self._temp_path = None  # Successfully moved
            
        except Exception:
            self._cleanup_temp()
            raise
        
        result = {
            "output_path": str(self.output_path),
            "vector_count": num_vectors,
            "dimension": dimension,
            "file_size_bytes": self.output_path.stat().st_size,
        }
        
        if hasher:
            result["checksum"] = hasher.hexdigest()
        
        return result

    def write_from_chunks(
        self,
        chunks,
        num_vectors: int,
        dimension: int,
        compute_checksum: bool = False,
    ) -> dict[str, Any]:
        """Write vector data from an iterable of chunks.
        
        This is useful for streaming writes where data is not fully loaded in memory.
        
        Args:
            chunks: Iterable of NumPy arrays, each with shape (chunk_size, dimension).
            num_vectors: Total number of vectors to expect.
            dimension: Expected dimension of each vector.
            compute_checksum: If True, compute and return SHA256 checksum.
            
        Returns:
            Dictionary containing write statistics (same as write()).
            
        Raises:
            ValueError: If chunk dimensions don't match.
            RuntimeError: If write is cancelled.
        """
        self._cancelled = False
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        fd, temp_path_str = tempfile.mkstemp(
            suffix=".fbin.tmp",
            dir=self.output_path.parent
        )
        self._temp_path = Path(temp_path_str)
        
        hasher = hashlib.sha256() if compute_checksum else None
        
        try:
            with os.fdopen(fd, "wb") as f:
                # Write header
                header = struct.pack("<II", num_vectors, dimension)
                f.write(header)
                if hasher:
                    hasher.update(header)
                
                # Write chunks
                written = 0
                for chunk in chunks:
                    if self._cancelled:
                        self._cleanup_temp()
                        raise RuntimeError("Write cancelled")
                    
                    # Validate chunk
                    if chunk.ndim != 2 or chunk.shape[1] != dimension:
                        raise ValueError(
                            f"Chunk has invalid shape {chunk.shape}, expected (*, {dimension})"
                        )
                    
                    # Ensure float32
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    
                    chunk_data = chunk.tobytes()
                    f.write(chunk_data)
                    if hasher:
                        hasher.update(chunk_data)
                    
                    written += len(chunk)
                    self._report_progress(written, num_vectors)
                
                # Validate total count
                if written != num_vectors:
                    raise ValueError(
                        f"Expected {num_vectors} vectors but wrote {written}"
                    )
            
            # Atomic rename
            self._temp_path.replace(self.output_path)
            self._temp_path = None
            
        except Exception:
            self._cleanup_temp()
            raise
        
        result = {
            "output_path": str(self.output_path),
            "vector_count": num_vectors,
            "dimension": dimension,
            "file_size_bytes": self.output_path.stat().st_size,
        }
        
        if hasher:
            result["checksum"] = hasher.hexdigest()
        
        return result


def write_fbin(
    output_path: str | Path,
    data: np.ndarray,
    compute_checksum: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience function to write an FBIN file.
    
    Args:
        output_path: Path for the output file.
        data: NumPy array of shape (num_vectors, dimension).
        compute_checksum: If True, compute and return SHA256 checksum.
        progress_callback: Optional callback function(current, total).
        
    Returns:
        Dictionary with write statistics.
    """
    writer = FBINWriter(output_path, progress_callback=progress_callback)
    return writer.write(data, compute_checksum=compute_checksum)
