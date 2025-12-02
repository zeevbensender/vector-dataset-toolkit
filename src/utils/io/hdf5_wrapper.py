"""Utilities for wrapping FBIN/IBIN files into an HDF5 package.

This module provides validation and wrapping helpers to combine one or more
FBIN vector shards (and an optional IBIN ground-truth file) into a single
HDF5 container with configurable dataset names and compression.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from .fbin_reader import FBINReader
from .ibin_reader import IBINReader


class HDF5Wrapper:
    """Wrapper utility for bundling FBIN/IBIN inputs into HDF5 files."""

    def __init__(
        self,
        chunk_size: int = 10000,
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the current operation."""

        self._cancelled = True

    def _report_progress(self, current: int, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(current, total)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    def validate_inputs(
        self, fbin_paths: list[str | Path], ibin_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Validate FBIN shards and optional IBIN neighbor file."""

        files: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        if not fbin_paths:
            errors.append("At least one FBIN file is required")
            return {
                "valid": False,
                "files": files,
                "errors": errors,
                "warnings": warnings,
            }

        total_vectors = 0
        dimension: int | None = None

        for path_str in fbin_paths:
            path = Path(path_str)
            try:
                reader = FBINReader(path)
                meta = reader.get_metadata()
                reader.close()
            except Exception as exc:  # pragma: no cover - defensive
                files.append(
                    {
                        "path": str(path),
                        "status": "error",
                        "message": str(exc),
                        "vector_count": 0,
                        "dimension": 0,
                    }
                )
                errors.append(f"Failed to read {path}: {exc}")
                continue

            if dimension is None:
                dimension = meta["dimension"]
            elif meta["dimension"] != dimension:
                errors.append(
                    f"Dimension mismatch for {path} (expected {dimension}, got {meta['dimension']})"
                )

            if not meta.get("size_match", True):
                warnings.append(f"File size mismatch for {path}")

            total_vectors += meta["vector_count"]
            files.append(
                {
                    "path": str(path),
                    "status": "ok",
                    "message": "",
                    "vector_count": meta["vector_count"],
                    "dimension": meta["dimension"],
                }
            )

        k: int | None = None
        ibin_meta: dict[str, Any] | None = None
        if ibin_path:
            ibin_path_obj = Path(ibin_path)
            try:
                ibin_reader = IBINReader(ibin_path_obj)
                ibin_meta = ibin_reader.get_metadata()
                ibin_reader.close()
                k = ibin_meta.get("k")
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Failed to read IBIN file: {exc}")
                ibin_meta = {"path": str(ibin_path_obj), "status": "error", "message": str(exc)}

            if ibin_meta and "vector_count" in ibin_meta and total_vectors:
                if ibin_meta["vector_count"] != total_vectors:
                    errors.append(
                        f"IBIN query count ({ibin_meta['vector_count']}) does not match total vectors ({total_vectors})"
                    )

        estimated_vectors_bytes = total_vectors * (dimension or 0) * 4
        estimated_neighbors_bytes = (
            total_vectors * (k or 0) * 4 if ibin_path and k else 0
        )

        return {
            "valid": not errors,
            "files": files,
            "ibin": ibin_meta,
            "errors": errors,
            "warnings": warnings,
            "total_vectors": total_vectors,
            "dimension": dimension,
            "k": k,
            "estimated_bytes": estimated_vectors_bytes + estimated_neighbors_bytes,
        }

    def wrap_into_hdf5(
        self,
        fbin_paths: list[str | Path],
        output_path: str | Path,
        *,
        vector_dataset: str = "vectors",
        neighbor_dataset: str = "neighbors",
        compression: str | None = None,
        ibin_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Wrap FBIN shards and optional IBIN neighbors into an HDF5 file."""

        self._cancelled = False
        validation = self.validate_inputs(fbin_paths, ibin_path)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))

        total_vectors = validation["total_vectors"]
        dimension = validation["dimension"] or 0
        k = validation.get("k") or 0

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path_str = tempfile.mkstemp(suffix=".h5.tmp", dir=output_path.parent)
        os.close(fd)
        temp_path = Path(temp_path_str)

        total_steps = total_vectors + (total_vectors if ibin_path and k else 0)
        vector_written = 0
        neighbor_written = 0

        try:
            with h5py.File(temp_path, "w") as h5:
                vector_chunks = (min(self.chunk_size, total_vectors), dimension)
                vectors_ds = h5.create_dataset(
                    vector_dataset,
                    shape=(total_vectors, dimension),
                    dtype=np.float32,
                    chunks=vector_chunks,
                    compression=compression,
                )
                vectors_ds.attrs["vector_count"] = total_vectors
                vectors_ds.attrs["dimension"] = dimension
                vectors_ds.attrs["source_files"] = [str(Path(p)) for p in fbin_paths]

                self._log("Copying vector shards into HDF5 dataset")
                for fbin_path in fbin_paths:
                    reader = FBINReader(fbin_path)
                    for chunk in reader.read_sequential(chunk_size=self.chunk_size):
                        if self._cancelled:
                            raise RuntimeError("Wrap cancelled")
                        end = vector_written + len(chunk)
                        vectors_ds[vector_written:end] = chunk
                        vector_written = end
                        self._report_progress(vector_written + neighbor_written, total_steps)
                    reader.close()

                if ibin_path and k:
                    neighbor_chunks = (min(self.chunk_size, total_vectors), k)
                    neighbors_ds = h5.create_dataset(
                        neighbor_dataset,
                        shape=(total_vectors, k),
                        dtype=np.int32,
                        chunks=neighbor_chunks,
                        compression=compression,
                    )
                    neighbors_ds.attrs["vector_count"] = total_vectors
                    neighbors_ds.attrs["k"] = k
                    neighbors_ds.attrs["source_files"] = [str(Path(ibin_path))]

                    self._log("Copying neighbor indices into HDF5 dataset")
                    ibin_reader = IBINReader(ibin_path)
                    for chunk in ibin_reader.read_sequential(chunk_size=self.chunk_size):
                        if self._cancelled:
                            raise RuntimeError("Wrap cancelled")
                        end = neighbor_written + len(chunk)
                        neighbors_ds[neighbor_written:end] = chunk
                        neighbor_written = end
                        self._report_progress(vector_written + neighbor_written, total_steps)
                    ibin_reader.close()

                h5.attrs["vector_count"] = total_vectors
                h5.attrs["dimension"] = dimension
                if k:
                    h5.attrs["k"] = k
                h5.attrs["source_files"] = [str(Path(p)) for p in fbin_paths]
                if ibin_path:
                    h5.attrs["ibin_source"] = str(Path(ibin_path))

            temp_path.replace(output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        result = {
            "output_path": str(output_path),
            "vector_dataset": vector_dataset,
            "neighbor_dataset": neighbor_dataset if ibin_path else None,
            "vectors": {
                "count": total_vectors,
                "dimension": dimension,
            },
            "neighbors": {
                "count": total_vectors if ibin_path else 0,
                "k": k,
            }
            if ibin_path
            else None,
        }

        self._log(f"Wrap complete: wrote {output_path}")
        return result
