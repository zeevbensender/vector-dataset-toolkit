"""FBIN shard merging utilities.

This module provides functionality to merge multiple FBIN shard files into
a single contiguous file. This is commonly needed when datasets are distributed
across multiple shard files.

Features:
- Validates compatibility of shards before merging
- Supports output to FBIN or NPY format
- Chunked processing for memory efficiency
- Progress callbacks for UI integration
- Checksum generation for verification
- Dry-run mode for preview

Safety Guarantees:
- Original shard files are never modified
- Output is written to a temp file first, then atomically renamed
- On cancellation, partial outputs are cleaned up
- Incompatible shards are rejected with clear error messages
"""

import hashlib
import os
import struct
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .fbin_reader import FBIN_HEADER_SIZE, FBINReader
from .fbin_writer import FBINWriter


class ShardValidationResult(Enum):
    """Result of shard validation."""
    COMPATIBLE = "compatible"
    INCOMPATIBLE_DIMENSION = "incompatible_dimension"
    INCOMPATIBLE_DTYPE = "incompatible_dtype"
    FILE_CORRUPTED = "file_corrupted"
    FILE_NOT_FOUND = "file_not_found"
    FILE_SIZE_MISMATCH = "file_size_mismatch"


@dataclass
class ShardInfo:
    """Information about a single shard file."""
    path: Path
    vector_count: int
    dimension: int
    file_size_bytes: int
    file_size_mb: float
    size_match: bool
    validation_result: ShardValidationResult
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": str(self.path),
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": self.file_size_mb,
            "size_match": self.size_match,
            "validation_result": self.validation_result.value,
            "error_message": self.error_message,
        }


@dataclass
class MergePreview:
    """Preview of merge operation results."""
    shards: list[ShardInfo]
    total_vectors: int
    total_file_size_bytes: int
    dimension: int
    output_format: str
    expected_output_size_bytes: int
    all_compatible: bool
    incompatible_shards: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "shards": [s.to_dict() for s in self.shards],
            "total_vectors": self.total_vectors,
            "total_file_size_bytes": self.total_file_size_bytes,
            "dimension": self.dimension,
            "output_format": self.output_format,
            "expected_output_size_bytes": self.expected_output_size_bytes,
            "all_compatible": self.all_compatible,
            "incompatible_shards": self.incompatible_shards,
            "warnings": self.warnings,
        }


class ShardMerger:
    """Merger for FBIN shard files.
    
    Supports:
    - Validation of shard compatibility
    - Merging to FBIN or NPY output format
    - Dry-run mode for previewing merge results
    - Progress callbacks for UI integration
    - Checksum generation for verification
    
    Safety Guarantees:
    - Original shards are never modified
    - Output is written to temp file first, then atomically renamed
    - On cancellation, partial outputs are cleaned up
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the merger.
        
        Args:
            chunk_size: Number of vectors to process per chunk.
            progress_callback: Optional callback function(current, total).
            log_callback: Optional callback for log messages.
        """
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True

    def _log(self, message: str) -> None:
        """Emit a log message if callback is set."""
        if self.log_callback:
            self.log_callback(message)

    def _report_progress(self, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total)

    def validate_shard(
        self,
        shard_path: str | Path,
        reference_dimension: int | None = None,
    ) -> ShardInfo:
        """Validate a single shard file.
        
        Args:
            shard_path: Path to the shard file.
            reference_dimension: Expected dimension (None for first shard).
            
        Returns:
            ShardInfo with validation results.
        """
        shard_path = Path(shard_path)
        
        # Check file exists
        if not shard_path.exists():
            return ShardInfo(
                path=shard_path,
                vector_count=0,
                dimension=0,
                file_size_bytes=0,
                file_size_mb=0.0,
                size_match=False,
                validation_result=ShardValidationResult.FILE_NOT_FOUND,
                error_message=f"File not found: {shard_path}",
            )
        
        try:
            reader = FBINReader(shard_path)
            metadata = reader.get_metadata()
            reader.close()
            
            vector_count = metadata["vector_count"]
            dimension = metadata["dimension"]
            file_size_bytes = metadata["file_size_bytes"]
            size_match = metadata["size_match"]
            
            # Check dimension compatibility
            if reference_dimension is not None and dimension != reference_dimension:
                return ShardInfo(
                    path=shard_path,
                    vector_count=vector_count,
                    dimension=dimension,
                    file_size_bytes=file_size_bytes,
                    file_size_mb=metadata["file_size_mb"],
                    size_match=size_match,
                    validation_result=ShardValidationResult.INCOMPATIBLE_DIMENSION,
                    error_message=f"Dimension mismatch: expected {reference_dimension}, got {dimension}",
                )
            
            # Check file size
            if not size_match:
                return ShardInfo(
                    path=shard_path,
                    vector_count=vector_count,
                    dimension=dimension,
                    file_size_bytes=file_size_bytes,
                    file_size_mb=metadata["file_size_mb"],
                    size_match=size_match,
                    validation_result=ShardValidationResult.FILE_SIZE_MISMATCH,
                    error_message="File size does not match header metadata",
                )
            
            return ShardInfo(
                path=shard_path,
                vector_count=vector_count,
                dimension=dimension,
                file_size_bytes=file_size_bytes,
                file_size_mb=metadata["file_size_mb"],
                size_match=size_match,
                validation_result=ShardValidationResult.COMPATIBLE,
            )
            
        except Exception as e:
            return ShardInfo(
                path=shard_path,
                vector_count=0,
                dimension=0,
                file_size_bytes=shard_path.stat().st_size if shard_path.exists() else 0,
                file_size_mb=0.0,
                size_match=False,
                validation_result=ShardValidationResult.FILE_CORRUPTED,
                error_message=f"Failed to read shard: {e}",
            )

    def validate_shards(
        self,
        shard_paths: list[str | Path],
    ) -> list[ShardInfo]:
        """Validate a list of shard files for compatibility.
        
        Args:
            shard_paths: List of paths to shard files.
            
        Returns:
            List of ShardInfo with validation results for each shard.
        """
        results: list[ShardInfo] = []
        reference_dimension: int | None = None
        
        for path in shard_paths:
            info = self.validate_shard(path, reference_dimension)
            results.append(info)
            
            # Set reference dimension from first valid shard
            if (reference_dimension is None 
                and info.validation_result == ShardValidationResult.COMPATIBLE):
                reference_dimension = info.dimension
        
        return results

    def preview_merge(
        self,
        shard_paths: list[str | Path],
        output_format: str = "fbin",
    ) -> MergePreview:
        """Preview the merge operation without writing any files.
        
        Args:
            shard_paths: List of paths to shard files.
            output_format: Output format ("fbin" or "npy").
            
        Returns:
            MergePreview with expected results and any issues.
        """
        shard_infos = self.validate_shards(shard_paths)
        
        # Calculate totals from compatible shards
        compatible_shards = [
            s for s in shard_infos 
            if s.validation_result == ShardValidationResult.COMPATIBLE
        ]
        
        total_vectors = sum(s.vector_count for s in compatible_shards)
        total_file_size = sum(s.file_size_bytes for s in compatible_shards)
        dimension = compatible_shards[0].dimension if compatible_shards else 0
        
        # Calculate expected output size
        if output_format == "fbin":
            expected_size = FBIN_HEADER_SIZE + (total_vectors * dimension * 4)
        else:  # npy
            # NPY has a header (typically ~128 bytes for simple arrays)
            expected_size = 128 + (total_vectors * dimension * 4)
        
        # Find incompatible shards
        incompatible = [
            str(s.path) for s in shard_infos
            if s.validation_result != ShardValidationResult.COMPATIBLE
        ]
        
        # Generate warnings
        warnings: list[str] = []
        if len(compatible_shards) < 2:
            warnings.append("At least 2 compatible shards are needed for merge")
        
        size_mismatches = [
            s for s in shard_infos
            if s.validation_result == ShardValidationResult.FILE_SIZE_MISMATCH
        ]
        for s in size_mismatches:
            warnings.append(f"Shard {s.path.name} has file size mismatch")
        
        return MergePreview(
            shards=shard_infos,
            total_vectors=total_vectors,
            total_file_size_bytes=total_file_size,
            dimension=dimension,
            output_format=output_format,
            expected_output_size_bytes=expected_size,
            all_compatible=len(incompatible) == 0 and len(compatible_shards) >= 2,
            incompatible_shards=incompatible,
            warnings=warnings,
        )

    def merge(
        self,
        shard_paths: list[str | Path],
        output_path: str | Path,
        output_format: str = "fbin",
        compute_checksum: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Merge multiple FBIN shards into a single file.
        
        Args:
            shard_paths: List of paths to shard files.
            output_path: Path for the output file.
            output_format: Output format ("fbin" or "npy").
            compute_checksum: If True, compute SHA256 checksum of output.
            dry_run: If True, only validate and return preview.
            
        Returns:
            Dictionary containing:
            - output_path: Path to the output file
            - total_vectors: Number of vectors merged
            - dimension: Vector dimension
            - shards_merged: Number of shards merged
            - file_size_bytes: Size of output file (not in dry run)
            - checksum: SHA256 hex digest (if compute_checksum=True)
            - dry_run: Whether this was a dry run
            
        Raises:
            ValueError: If shards are incompatible or insufficient.
            RuntimeError: If merge is cancelled.
        """
        self._cancelled = False
        output_path = Path(output_path)
        
        # Validate shards
        preview = self.preview_merge(shard_paths, output_format)
        
        if not preview.all_compatible:
            incompatible_msg = "\n".join(preview.incompatible_shards[:5])
            if len(preview.incompatible_shards) > 5:
                incompatible_msg += f"\n... and {len(preview.incompatible_shards) - 5} more"
            raise ValueError(f"Cannot merge incompatible shards:\n{incompatible_msg}")
        
        result = {
            "output_path": str(output_path),
            "total_vectors": preview.total_vectors,
            "dimension": preview.dimension,
            "shards_merged": len(preview.shards),
            "output_format": output_format,
            "dry_run": dry_run,
        }
        
        if dry_run:
            result["expected_size_bytes"] = preview.expected_output_size_bytes
            self._log("Dry run mode: no file written")
            return result
        
        self._log(f"Merging {len(preview.shards)} shards -> {output_path}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == "fbin":
            return self._merge_to_fbin(
                preview, output_path, compute_checksum, result
            )
        else:
            return self._merge_to_npy(
                preview, output_path, compute_checksum, result
            )

    def _merge_to_fbin(
        self,
        preview: MergePreview,
        output_path: Path,
        compute_checksum: bool,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge shards to FBIN format."""
        total_vectors = preview.total_vectors
        dimension = preview.dimension
        
        # Use temp file for atomic write
        fd, temp_path_str = tempfile.mkstemp(
            suffix=".fbin.tmp",
            dir=output_path.parent
        )
        temp_path = Path(temp_path_str)
        
        hasher = hashlib.sha256() if compute_checksum else None
        
        try:
            with os.fdopen(fd, "wb") as f:
                # Write header
                header = struct.pack("<II", total_vectors, dimension)
                f.write(header)
                if hasher:
                    hasher.update(header)
                
                # Process each shard
                processed = 0
                for shard_info in preview.shards:
                    if self._cancelled:
                        raise RuntimeError("Merge cancelled")
                    
                    self._log(f"Processing shard: {shard_info.path.name}")
                    reader = FBINReader(shard_info.path, mmap_mode=True)
                    
                    try:
                        for chunk in reader.read_sequential(chunk_size=self.chunk_size):
                            if self._cancelled:
                                raise RuntimeError("Merge cancelled")
                            
                            chunk_data = chunk.astype(np.float32).tobytes()
                            f.write(chunk_data)
                            if hasher:
                                hasher.update(chunk_data)
                            
                            processed += len(chunk)
                            self._report_progress(processed, total_vectors)
                    finally:
                        reader.close()
            
            # Atomic rename
            temp_path.replace(output_path)
            
            result["file_size_bytes"] = output_path.stat().st_size
            if hasher:
                result["checksum"] = hasher.hexdigest()
            
            self._log(f"Merge complete: {output_path}")
            return result
            
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _merge_to_npy(
        self,
        preview: MergePreview,
        output_path: Path,
        compute_checksum: bool,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge shards to NPY format."""
        total_vectors = preview.total_vectors
        dimension = preview.dimension
        
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
                shape=(total_vectors, dimension),
            )
            
            # Process each shard
            processed = 0
            for shard_info in preview.shards:
                if self._cancelled:
                    del output_array
                    raise RuntimeError("Merge cancelled")
                
                self._log(f"Processing shard: {shard_info.path.name}")
                reader = FBINReader(shard_info.path, mmap_mode=True)
                
                try:
                    for chunk in reader.read_sequential(chunk_size=self.chunk_size):
                        if self._cancelled:
                            del output_array
                            raise RuntimeError("Merge cancelled")
                        
                        chunk_end = processed + len(chunk)
                        output_array[processed:chunk_end] = chunk
                        processed = chunk_end
                        self._report_progress(processed, total_vectors)
                finally:
                    reader.close()
            
            # Flush to disk
            del output_array
            
            # Atomic rename
            temp_path.replace(output_path)
            
            result["file_size_bytes"] = output_path.stat().st_size
            
            # Compute checksum if requested (read the final file)
            if compute_checksum:
                hasher = hashlib.sha256()
                with open(output_path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        hasher.update(chunk)
                result["checksum"] = hasher.hexdigest()
            
            self._log(f"Merge complete: {output_path}")
            return result
            
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise


def merge_fbin_shards(
    shard_paths: list[str | Path],
    output_path: str | Path,
    output_format: str = "fbin",
    chunk_size: int = 10000,
    compute_checksum: bool = False,
    dry_run: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience function for merging FBIN shards.
    
    Args:
        shard_paths: List of paths to shard files.
        output_path: Path for the output file.
        output_format: Output format ("fbin" or "npy").
        chunk_size: Number of vectors per processing chunk.
        compute_checksum: If True, compute SHA256 checksum of output.
        dry_run: If True, only validate without writing.
        progress_callback: Optional callback function(current, total).
        
    Returns:
        Dictionary with merge statistics.
    """
    merger = ShardMerger(
        chunk_size=chunk_size,
        progress_callback=progress_callback,
    )
    return merger.merge(
        shard_paths,
        output_path,
        output_format=output_format,
        compute_checksum=compute_checksum,
        dry_run=dry_run,
    )
