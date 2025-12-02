"""FBIN conversion utilities.

This module provides chunked, memory-efficient conversion between FBIN and other formats:
- FBIN ↔ NPY conversion
- FBIN → HDF5 conversion

Features:
- Chunked processing to avoid memory issues with large files
- Memory-mapping support for source files
- Progress callbacks for UI integration
- Dry-run mode to preview conversion without writing
- Verbose logging option
- Atomic writes via temporary files

Safety Guarantees:
- Output is written to a temp file first, then atomically renamed
- On cancellation, partial outputs are cleaned up
- Original files are never modified
"""

import hashlib
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from .fbin_reader import FBIN_HEADER_SIZE, FBINReader
from .fbin_writer import FBINWriter


# Default chunk size (number of vectors per chunk)
DEFAULT_CHUNK_SIZE = 10000


class FBINConverter:
    """Converter for FBIN format files.
    
    Supports:
    - FBIN to NPY conversion
    - NPY to FBIN conversion
    - FBIN to HDF5 conversion
    - Chunked processing for memory efficiency
    - Progress callbacks for UI integration
    - Dry-run mode for preview
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_callback: Callable[[int, int], None] | None = None,
        verbose: bool = False,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the converter.
        
        Args:
            chunk_size: Number of vectors to process per chunk.
            progress_callback: Optional callback function(current, total) for progress.
            verbose: If True, emit detailed log messages.
            log_callback: Optional callback for log messages (verbose mode).
        """
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self.verbose = verbose
        self.log_callback = log_callback
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True

    def _log(self, message: str) -> None:
        """Emit a log message if verbose mode is enabled."""
        if self.verbose and self.log_callback:
            self.log_callback(message)

    def _report_progress(self, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total)

    def fbin_to_npy(
        self,
        input_path: str | Path,
        output_path: str | Path,
        use_mmap: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Convert FBIN file to NPY format.
        
        Args:
            input_path: Path to the input .fbin file.
            output_path: Path for the output .npy file.
            use_mmap: If True, use memory-mapping for reading source.
            dry_run: If True, only validate and return metadata without writing.
            
        Returns:
            Dictionary containing:
            - input_path: Path to input file
            - output_path: Path to output file
            - vector_count: Number of vectors
            - dimension: Vector dimension
            - dtype: Data type
            - dry_run: Whether this was a dry run
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            RuntimeError: If conversion is cancelled.
        """
        self._cancelled = False
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self._log(f"Reading FBIN header from {input_path}")
        reader = FBINReader(input_path, mmap_mode=use_mmap)
        metadata = reader.get_metadata()

        num_vectors = metadata["vector_count"]
        dimension = metadata["dimension"]

        result = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "vector_count": num_vectors,
            "dimension": dimension,
            "shape": (num_vectors, dimension),
            "dtype": "float32",
            "dry_run": dry_run,
        }

        if dry_run:
            self._log("Dry run mode: no file written")
            expected_size = 128 + (num_vectors * dimension * 4)  # NPY header + data
            result["expected_size_bytes"] = expected_size
            return result

        self._log(f"Converting {num_vectors} vectors to NPY format")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use temp file for atomic write
        fd, temp_path_str = tempfile.mkstemp(
            suffix=".npy.tmp",
            dir=output_path.parent
        )
        temp_path = Path(temp_path_str)
        os.close(fd)

        try:
            # Create memory-mapped output
            output_array = np.lib.format.open_memmap(
                str(temp_path),
                mode="w+",
                dtype=np.float32,
                shape=(num_vectors, dimension),
            )

            # Process in chunks
            processed = 0
            for chunk in reader.read_sequential(chunk_size=self.chunk_size):
                if self._cancelled:
                    del output_array
                    temp_path.unlink(missing_ok=True)
                    raise RuntimeError("Conversion cancelled")

                chunk_end = processed + len(chunk)
                output_array[processed:chunk_end] = chunk
                processed = chunk_end
                self._report_progress(processed, num_vectors)
                self._log(f"Processed {processed:,}/{num_vectors:,} vectors")

            # Flush to disk
            del output_array

            # Atomic rename
            temp_path.replace(output_path)

            result["file_size_bytes"] = output_path.stat().st_size
            self._log(f"Conversion complete: {output_path}")

        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        finally:
            reader.close()

        return result

    def npy_to_fbin(
        self,
        input_path: str | Path,
        output_path: str | Path,
        use_mmap: bool = True,
        dry_run: bool = False,
        compute_checksum: bool = False,
    ) -> dict[str, Any]:
        """Convert NPY file to FBIN format.
        
        Args:
            input_path: Path to the input .npy file.
            output_path: Path for the output .fbin file.
            use_mmap: If True, use memory-mapping for reading source.
            dry_run: If True, only validate and return metadata without writing.
            compute_checksum: If True, compute SHA256 checksum of output.
            
        Returns:
            Dictionary with conversion statistics.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If input array is not 2D.
            RuntimeError: If conversion is cancelled.
        """
        self._cancelled = False
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self._log(f"Reading NPY file: {input_path}")
        mmap_mode = "r" if use_mmap else None
        source = np.load(str(input_path), mmap_mode=mmap_mode)

        if source.ndim != 2:
            raise ValueError(f"NPY array must be 2D, got shape {source.shape}")

        num_vectors, dimension = source.shape

        result = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "vector_count": num_vectors,
            "dimension": dimension,
            "shape": (num_vectors, dimension),
            "dtype": str(source.dtype),
            "dry_run": dry_run,
        }

        if dry_run:
            self._log("Dry run mode: no file written")
            expected_size = FBIN_HEADER_SIZE + (num_vectors * dimension * 4)
            result["expected_size_bytes"] = expected_size
            return result

        self._log(f"Converting {num_vectors} vectors to FBIN format")

        def progress_wrapper(current: int, total: int) -> None:
            self._report_progress(current, total)
            self._log(f"Processed {current:,}/{total:,} vectors")
            if self._cancelled:
                raise RuntimeError("Conversion cancelled")

        writer = FBINWriter(output_path, progress_callback=progress_wrapper)
        write_result = writer.write(
            source,
            chunk_size=self.chunk_size,
            compute_checksum=compute_checksum,
        )

        result["file_size_bytes"] = write_result["file_size_bytes"]
        if compute_checksum:
            result["checksum"] = write_result["checksum"]

        self._log(f"Conversion complete: {output_path}")
        return result

    def fbin_to_hdf5(
        self,
        input_path: str | Path,
        output_path: str | Path,
        dataset_name: str = "vectors",
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
        use_mmap: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Convert FBIN file to HDF5 format.
        
        Args:
            input_path: Path to the input .fbin file.
            output_path: Path for the output .h5 file.
            dataset_name: Name for the dataset in HDF5 file.
            compression: Compression algorithm (None, 'gzip', 'lzf').
            compression_opts: Compression level (for gzip: 0-9).
            use_mmap: If True, use memory-mapping for reading source.
            dry_run: If True, only validate and return metadata without writing.
            
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

        self._log(f"Reading FBIN header from {input_path}")
        reader = FBINReader(input_path, mmap_mode=use_mmap)
        metadata = reader.get_metadata()

        num_vectors = metadata["vector_count"]
        dimension = metadata["dimension"]

        result = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "vector_count": num_vectors,
            "dimension": dimension,
            "shape": (num_vectors, dimension),
            "dtype": "float32",
            "dataset_name": dataset_name,
            "compression": compression,
            "dry_run": dry_run,
        }

        if dry_run:
            self._log("Dry run mode: no file written")
            return result

        self._log(f"Converting {num_vectors} vectors to HDF5 format")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first
        fd, temp_path_str = tempfile.mkstemp(
            suffix=".h5.tmp",
            dir=output_path.parent
        )
        temp_path = Path(temp_path_str)
        os.close(fd)

        try:
            with h5py.File(str(temp_path), "w") as f:
                # Create dataset with chunking
                chunks = (min(self.chunk_size, num_vectors), dimension)
                dataset = f.create_dataset(
                    dataset_name,
                    shape=(num_vectors, dimension),
                    dtype=np.float32,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts if compression else None,
                )

                # Store source metadata
                dataset.attrs["source_file"] = str(input_path)
                dataset.attrs["source_format"] = "fbin"

                # Process in chunks
                processed = 0
                for chunk in reader.read_sequential(chunk_size=self.chunk_size):
                    if self._cancelled:
                        raise RuntimeError("Conversion cancelled")

                    chunk_end = processed + len(chunk)
                    dataset[processed:chunk_end] = chunk
                    processed = chunk_end
                    self._report_progress(processed, num_vectors)
                    self._log(f"Processed {processed:,}/{num_vectors:,} vectors")

            # Atomic rename
            temp_path.replace(output_path)
            result["file_size_bytes"] = output_path.stat().st_size
            self._log(f"Conversion complete: {output_path}")

        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        finally:
            reader.close()

        return result


# Convenience functions

def convert_fbin_to_npy(
    input_path: str | Path,
    output_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    use_mmap: bool = True,
    dry_run: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience function for FBIN to NPY conversion.
    
    Args:
        input_path: Path to the input .fbin file.
        output_path: Path for the output .npy file.
        chunk_size: Number of vectors per processing chunk.
        use_mmap: If True, use memory-mapping for source file.
        dry_run: If True, only validate without writing.
        progress_callback: Optional callback function(current, total).
        
    Returns:
        Dictionary with conversion statistics.
    """
    converter = FBINConverter(
        chunk_size=chunk_size,
        progress_callback=progress_callback,
    )
    return converter.fbin_to_npy(input_path, output_path, use_mmap=use_mmap, dry_run=dry_run)


def convert_npy_to_fbin(
    input_path: str | Path,
    output_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    use_mmap: bool = True,
    dry_run: bool = False,
    compute_checksum: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience function for NPY to FBIN conversion.
    
    Args:
        input_path: Path to the input .npy file.
        output_path: Path for the output .fbin file.
        chunk_size: Number of vectors per processing chunk.
        use_mmap: If True, use memory-mapping for source file.
        dry_run: If True, only validate without writing.
        compute_checksum: If True, compute SHA256 checksum of output.
        progress_callback: Optional callback function(current, total).
        
    Returns:
        Dictionary with conversion statistics.
    """
    converter = FBINConverter(
        chunk_size=chunk_size,
        progress_callback=progress_callback,
    )
    return converter.npy_to_fbin(
        input_path, output_path,
        use_mmap=use_mmap,
        dry_run=dry_run,
        compute_checksum=compute_checksum,
    )


def convert_fbin_to_hdf5(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "vectors",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compression: str | None = "gzip",
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience function for FBIN to HDF5 conversion.
    
    Args:
        input_path: Path to the input .fbin file.
        output_path: Path for the output .h5 file.
        dataset_name: Name for the dataset in HDF5 file.
        chunk_size: Number of vectors per processing chunk.
        compression: Compression algorithm (None, 'gzip', 'lzf').
        progress_callback: Optional callback function(current, total).
        
    Returns:
        Dictionary with conversion statistics.
    """
    converter = FBINConverter(
        chunk_size=chunk_size,
        progress_callback=progress_callback,
    )
    return converter.fbin_to_hdf5(
        input_path, output_path,
        dataset_name=dataset_name,
        compression=compression,
    )
