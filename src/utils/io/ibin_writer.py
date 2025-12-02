"""Writer for IBIN (binary int32 neighbor) files.

Mirrors FBINWriter behavior for ground-truth neighbor indices while
supporting atomic writes and progress callbacks.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

class IBINWriter:
    """Writer for IBIN (binary int32 neighbor) files."""

    def __init__(
        self,
        output_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.progress_callback = progress_callback
        self._cancelled = False
        self._temp_path: Path | None = None

    def cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True

    def _cleanup_temp(self) -> None:
        if self._temp_path and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except OSError:
                pass

    def _report_progress(self, current: int, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(current, total)

    def write(
        self,
        data: np.ndarray,
        chunk_size: int = 10000,
        compute_checksum: bool | None = False,
    ) -> dict[str, Any]:
        """Write neighbor indices to IBIN format."""
        self._cancelled = False

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array, got shape {data.shape}")

        if data.dtype != np.int32:
            data = data.astype(np.int32)

        num_vectors, k = data.shape
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_path_str = tempfile.mkstemp(
            suffix=".ibin.tmp", dir=self.output_path.parent
        )
        self._temp_path = Path(temp_path_str)

        hasher = None
        if compute_checksum:
            import hashlib

            hasher = hashlib.sha256()

        try:
            with os.fdopen(fd, "wb") as f:
                header = struct.pack("<II", num_vectors, k)
                f.write(header)
                if hasher:
                    hasher.update(header)

                written = 0
                while written < num_vectors:
                    if self._cancelled:
                        self._cleanup_temp()
                        raise RuntimeError("Write cancelled")

                    chunk_end = min(written + chunk_size, num_vectors)
                    chunk = data[written:chunk_end].tobytes()
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)

                    written = chunk_end
                    self._report_progress(written, num_vectors)

            self._temp_path.replace(self.output_path)
            self._temp_path = None

        except Exception:
            self._cleanup_temp()
            raise

        result = {
            "output_path": str(self.output_path),
            "vector_count": num_vectors,
            "k": k,
            "file_size_bytes": self.output_path.stat().st_size,
        }
        if hasher:
            result["checksum"] = hasher.hexdigest()
        return result

    def write_from_chunks(
        self,
        chunks: Iterable[np.ndarray],
        num_vectors: int,
        k: int,
        compute_checksum: bool | None = False,
    ) -> dict[str, Any]:
        """Write neighbor indices from iterable chunks."""
        self._cancelled = False
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_path_str = tempfile.mkstemp(
            suffix=".ibin.tmp", dir=self.output_path.parent
        )
        self._temp_path = Path(temp_path_str)

        hasher = None
        if compute_checksum:
            import hashlib

            hasher = hashlib.sha256()

        try:
            with os.fdopen(fd, "wb") as f:
                header = struct.pack("<II", num_vectors, k)
                f.write(header)
                if hasher:
                    hasher.update(header)

                written = 0
                for chunk_array in chunks:
                    if self._cancelled:
                        self._cleanup_temp()
                        raise RuntimeError("Write cancelled")

                    if chunk_array.dtype != np.int32:
                        chunk_array = chunk_array.astype(np.int32)

                    flat = chunk_array.tobytes()
                    f.write(flat)
                    if hasher:
                        hasher.update(flat)

                    written += chunk_array.shape[0]
                    self._report_progress(written, num_vectors)

            self._temp_path.replace(self.output_path)
            self._temp_path = None
        except Exception:
            self._cleanup_temp()
            raise

        result = {
            "output_path": str(self.output_path),
            "vector_count": num_vectors,
            "k": k,
            "file_size_bytes": self.output_path.stat().st_size,
        }
        if hasher:
            result["checksum"] = hasher.hexdigest()
        return result


def write_ibin(
    output_path: str | Path,
    data: np.ndarray,
    chunk_size: int = 10000,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for writing IBIN data."""
    writer = IBINWriter(output_path, progress_callback=progress_callback)
    return writer.write(data, chunk_size=chunk_size)
