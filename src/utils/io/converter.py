"""Conversion core for vector dataset files.

This module provides chunked, memory-efficient conversion between NPY and HDF5
formats with progress reporting support.
"""

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np


# Default chunk size for conversion (number of vectors per chunk)
DEFAULT_CHUNK_SIZE = 10000


class Converter:
    """Converter for vector dataset files.
    
    Supports:
    - NPY to HDF5 conversion
    - HDF5 to NPY conversion
    - Chunked processing for memory efficiency
    - Progress callbacks for UI integration
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize the converter.
        
        Args:
            chunk_size: Number of vectors to process per chunk.
            progress_callback: Optional callback function(current, total) for progress updates.
        """
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True

    def _report_progress(self, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total)

    def npy_to_hdf5(
        self,
        input_path: str | Path,
        output_path: str | Path,
        dataset_name: str = "vectors",
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
    ) -> dict[str, Any]:
        """Convert NPY file to HDF5 format.
        
        Args:
            input_path: Path to the input .npy file.
            output_path: Path for the output .h5 file.
            dataset_name: Name for the dataset in the HDF5 file.
            compression: Compression algorithm (None, 'gzip', 'lzf').
            compression_opts: Compression level (for gzip: 0-9).
            
        Returns:
            Dictionary with conversion statistics.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            RuntimeError: If conversion is cancelled.
        """
        self._cancelled = False
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load source with memory mapping
        source = np.load(str(input_path), mmap_mode="r")
        total_vectors = source.shape[0]
        
        # Create output file
        with h5py.File(str(output_path), "w") as f:
            # Create dataset with chunking for efficient access
            chunks = (min(self.chunk_size, total_vectors),) + source.shape[1:]
            dataset = f.create_dataset(
                dataset_name,
                shape=source.shape,
                dtype=source.dtype,
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts if compression else None,
            )

            # Store metadata as attributes
            dataset.attrs["source_file"] = str(input_path)
            dataset.attrs["source_format"] = "npy"

            # Process in chunks
            processed = 0
            while processed < total_vectors:
                if self._cancelled:
                    raise RuntimeError("Conversion cancelled")

                chunk_end = min(processed + self.chunk_size, total_vectors)
                dataset[processed:chunk_end] = source[processed:chunk_end]
                processed = chunk_end
                self._report_progress(processed, total_vectors)

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "vectors_converted": total_vectors,
            "shape": source.shape,
            "dtype": str(source.dtype),
            "compression": compression,
        }

    def hdf5_to_npy(
        self,
        input_path: str | Path,
        output_path: str | Path,
        dataset_path: str = "vectors",
    ) -> dict[str, Any]:
        """Convert HDF5 dataset to NPY format.
        
        Args:
            input_path: Path to the input .h5 file.
            output_path: Path for the output .npy file.
            dataset_path: Path to the dataset within the HDF5 file.
            
        Returns:
            Dictionary with conversion statistics.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            KeyError: If dataset_path is not found.
            RuntimeError: If conversion is cancelled.
        """
        self._cancelled = False
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with h5py.File(str(input_path), "r") as f:
            if dataset_path not in f:
                raise KeyError(f"Dataset not found: {dataset_path}")
            
            source = f[dataset_path]
            if not isinstance(source, h5py.Dataset):
                raise TypeError(f"Path is not a dataset: {dataset_path}")

            total_vectors = source.shape[0]
            
            # Create memory-mapped output file for streaming write
            output_array = np.lib.format.open_memmap(
                str(output_path),
                mode="w+",
                dtype=source.dtype,
                shape=source.shape,
            )

            # Process in chunks
            processed = 0
            while processed < total_vectors:
                if self._cancelled:
                    del output_array
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError("Conversion cancelled")

                chunk_end = min(processed + self.chunk_size, total_vectors)
                output_array[processed:chunk_end] = source[processed:chunk_end]
                processed = chunk_end
                self._report_progress(processed, total_vectors)

            # Flush to disk
            del output_array

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "dataset_path": dataset_path,
            "vectors_converted": total_vectors,
            "shape": source.shape,
            "dtype": str(source.dtype),
        }


def convert_npy_to_hdf5(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "vectors",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compression: str | None = "gzip",
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience function for NPY to HDF5 conversion.
    
    Args:
        input_path: Path to the input .npy file.
        output_path: Path for the output .h5 file.
        dataset_name: Name for the dataset in the HDF5 file.
        chunk_size: Number of vectors per processing chunk.
        compression: Compression algorithm (None, 'gzip', 'lzf').
        progress_callback: Optional callback function(current, total).
        
    Returns:
        Dictionary with conversion statistics.
    """
    converter = Converter(chunk_size=chunk_size, progress_callback=progress_callback)
    return converter.npy_to_hdf5(input_path, output_path, dataset_name, compression)


def convert_hdf5_to_npy(
    input_path: str | Path,
    output_path: str | Path,
    dataset_path: str = "vectors",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience function for HDF5 to NPY conversion.
    
    Args:
        input_path: Path to the input .h5 file.
        output_path: Path for the output .npy file.
        dataset_path: Path to the dataset within the HDF5 file.
        chunk_size: Number of vectors per processing chunk.
        progress_callback: Optional callback function(current, total).
        
    Returns:
        Dictionary with conversion statistics.
    """
    converter = Converter(chunk_size=chunk_size, progress_callback=progress_callback)
    return converter.hdf5_to_npy(input_path, output_path, dataset_path)
