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
        self,
        base_fbin_paths: list[str | Path],
        query_fbin_path: str | Path,
        ibin_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Validate FBIN shards (base/train), queries, and optional IBIN neighbors."""

        base_files: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        if not base_fbin_paths:
            errors.append("At least one base FBIN file is required")

        dimension: int | None = None
        total_base_vectors = 0
        for path_str in base_fbin_paths:
            path = Path(path_str)
            try:
                reader = FBINReader(path)
                meta = reader.get_metadata()
                reader.close()
            except Exception as exc:  # pragma: no cover - defensive
                base_files.append(
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

            total_base_vectors += meta["vector_count"]
            base_files.append(
                {
                    "path": str(path),
                    "status": "ok",
                    "message": "",
                    "vector_count": meta["vector_count"],
                    "dimension": meta["dimension"],
                }
            )

        # Queries
        query_meta: dict[str, Any] | None = None
        if not query_fbin_path:
            errors.append("A queries FBIN file is required")
        else:
            query_path = Path(query_fbin_path)
            try:
                query_reader = FBINReader(query_path)
                query_meta = query_reader.get_metadata()
                query_reader.close()
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Failed to read queries FBIN: {exc}")
            else:
                if dimension is not None and query_meta["dimension"] != dimension:
                    errors.append(
                        "Query dimension does not match base vectors "
                        f"(expected {dimension}, got {query_meta['dimension']})"
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

            if ibin_meta and "vector_count" in ibin_meta and query_meta:
                if ibin_meta["vector_count"] != query_meta.get("vector_count", 0):
                    errors.append(
                        "IBIN query count "
                        f"({ibin_meta['vector_count']}) does not match queries ({query_meta.get('vector_count', 0)})"
                    )

        estimated_base_bytes = total_base_vectors * (dimension or 0) * 4
        estimated_query_bytes = (query_meta.get("vector_count", 0) if query_meta else 0) * (dimension or 0) * 4
        estimated_neighbors_bytes = (
            (query_meta.get("vector_count", 0) if query_meta else 0) * (k or 0) * 4
            if ibin_path and k
            else 0
        )

        return {
            "valid": not errors,
            "base_files": base_files,
            "query_file": query_meta | {"path": str(query_fbin_path)} if query_meta else None,
            "ibin": ibin_meta,
            "errors": errors,
            "warnings": warnings,
            "total_base_vectors": total_base_vectors,
            "total_query_vectors": query_meta.get("vector_count", 0) if query_meta else 0,
            "dimension": dimension if dimension is not None else query_meta.get("dimension") if query_meta else None,
            "k": k,
            "estimated_bytes": estimated_base_bytes + estimated_query_bytes + estimated_neighbors_bytes,
        }

    def wrap_into_hdf5(
        self,
        base_fbin_paths: list[str | Path],
        query_fbin_path: str | Path,
        output_path: str | Path,
        *,
        base_dataset: str = "base",
        train_dataset: str = "train",
        query_dataset: str = "test",
        neighbor_dataset: str = "neighbors",
        include_train_alias: bool = True,
        compression: str | None = None,
        ibin_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Wrap base + query FBINs (and optional IBIN) into an HDF5 file."""

        self._cancelled = False
        validation = self.validate_inputs(base_fbin_paths, query_fbin_path, ibin_path)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))

        base_vectors = validation["total_base_vectors"]
        query_vectors = validation["total_query_vectors"]
        dimension = validation["dimension"] or 0
        k = validation.get("k") or 0

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path_str = tempfile.mkstemp(suffix=".h5.tmp", dir=output_path.parent)
        os.close(fd)
        temp_path = Path(temp_path_str)

        total_steps = base_vectors + query_vectors + (query_vectors if ibin_path and k else 0)
        base_written = 0
        query_written = 0
        neighbor_written = 0

        try:
            with h5py.File(temp_path, "w") as h5:
                base_chunks = (min(self.chunk_size, base_vectors), dimension)
                base_ds = h5.create_dataset(
                    base_dataset,
                    shape=(base_vectors, dimension),
                    dtype=np.float32,
                    chunks=base_chunks,
                    compression=compression,
                )
                base_ds.attrs["vector_count"] = base_vectors
                base_ds.attrs["dimension"] = dimension
                base_ds.attrs["source_files"] = [str(Path(p)) for p in base_fbin_paths]

                self._log("Copying base/train vectors into HDF5 datasets")
                for fbin_path in base_fbin_paths:
                    reader = FBINReader(fbin_path)
                    self._log(f"Reading base shard {Path(fbin_path).name}")
                    for chunk in reader.read_sequential(chunk_size=self.chunk_size):
                        if self._cancelled:
                            raise RuntimeError("Wrap cancelled")
                        self._log(
                            f"Writing base vectors {base_written}–{base_written + len(chunk)}"
                        )
                        end = base_written + len(chunk)
                        base_ds[base_written:end] = chunk
                        base_written = end
                        self._report_progress(base_written + query_written + neighbor_written, total_steps)
                    reader.close()

                if include_train_alias and train_dataset and train_dataset != base_dataset:
                    h5[train_dataset] = base_ds

                query_chunks = (min(self.chunk_size, query_vectors), dimension)
                query_ds = h5.create_dataset(
                    query_dataset,
                    shape=(query_vectors, dimension),
                    dtype=np.float32,
                    chunks=query_chunks,
                    compression=compression,
                )
                query_ds.attrs["vector_count"] = query_vectors
                query_ds.attrs["dimension"] = dimension
                query_ds.attrs["source_files"] = [str(Path(query_fbin_path))]

                self._log("Copying query vectors into HDF5 dataset")
                query_reader = FBINReader(query_fbin_path)
                for chunk in query_reader.read_sequential(chunk_size=self.chunk_size):
                    if self._cancelled:
                        raise RuntimeError("Wrap cancelled")
                    self._log(
                        f"Writing query vectors {query_written}–{query_written + len(chunk)}"
                    )
                    end = query_written + len(chunk)
                    query_ds[query_written:end] = chunk
                    query_written = end
                    self._report_progress(base_written + query_written + neighbor_written, total_steps)
                query_reader.close()

                neighbors_ds = None
                distances_ds = None
                if ibin_path and k:
                    neighbor_chunks = (min(self.chunk_size, query_vectors), k)
                    neighbors_ds = h5.create_dataset(
                        neighbor_dataset,
                        shape=(query_vectors, k),
                        dtype=np.int32,
                        chunks=neighbor_chunks,
                        compression=compression,
                    )
                    neighbors_ds.attrs["vector_count"] = query_vectors
                    neighbors_ds.attrs["k"] = k
                    neighbors_ds.attrs["source_files"] = [str(Path(ibin_path))]

                    distances_ds = h5.create_dataset(
                        "distances",
                        shape=(query_vectors, k),
                        dtype=np.float32,
                        chunks=neighbor_chunks,
                        compression=compression,
                    )
                    distances_ds.attrs["vector_count"] = query_vectors
                    distances_ds.attrs["k"] = k
                    distances_ds.attrs["source_files"] = [str(Path(ibin_path))]

                    self._log("Copying neighbor indices into HDF5 dataset")
                    ibin_reader = IBINReader(ibin_path)
                    neighbor_chunk_idx = 0
                    for chunk in ibin_reader.read_sequential(chunk_size=self.chunk_size):
                        if self._cancelled:
                            raise RuntimeError("Wrap cancelled")
                        self._log(
                            "Processing neighbor chunk "
                            f"{neighbor_chunk_idx} (rows {neighbor_written}–{neighbor_written + len(chunk)})"
                        )
                        end = neighbor_written + len(chunk)
                        neighbors_ds[neighbor_written:end] = chunk

                        if distances_ds is not None:
                            query_chunk = query_ds[neighbor_written:end]

                            self._log(
                                f"Computing distances for chunk {neighbor_chunk_idx} with "
                                f"{len(chunk)} queries and {chunk.shape[1]} neighbors"
                            )

                            # h5py fancy indexing requires monotonically increasing
                            # indices. Fetch unique neighbors in sorted order and
                            # broadcast back to each query row to compute distances
                            # without per-row loops (which are prohibitively slow at
                            # large scale).
                            unique_neighbors, inverse = np.unique(
                                chunk, return_inverse=True
                            )
                            self._log(
                                f"Chunk {neighbor_chunk_idx}: {len(unique_neighbors)} unique neighbors"
                            )
                            unique_vectors = base_ds[unique_neighbors]

                            # inverse maps each flattened neighbor position back to
                            # the corresponding unique vector. Reshape to align with
                            # the original chunk layout.
                            expanded_vectors = unique_vectors[inverse].reshape(
                                chunk.shape + (dimension,)
                            )

                            distances = np.linalg.norm(
                                expanded_vectors - query_chunk[:, None, :], axis=2
                            ).astype(np.float32)

                            distances_ds[neighbor_written:end] = distances

                            self._log(
                                f"Finished distances for chunk {neighbor_chunk_idx} "
                                f"({neighbor_written}–{end})"
                            )

                        neighbor_written = end
                        neighbor_chunk_idx += 1
                        self._report_progress(base_written + query_written + neighbor_written, total_steps)
                    ibin_reader.close()

                h5.attrs["base_count"] = base_vectors
                h5.attrs["query_count"] = query_vectors
                h5.attrs["dimension"] = dimension
                if k:
                    h5.attrs["k"] = k
                h5.attrs["base_sources"] = [str(Path(p)) for p in base_fbin_paths]
                h5.attrs["query_source"] = str(Path(query_fbin_path))
                if ibin_path:
                    h5.attrs["ibin_source"] = str(Path(ibin_path))

            temp_path.replace(output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        result = {
            "output_path": str(output_path),
            "base_dataset": base_dataset,
            "train_dataset": train_dataset if include_train_alias else None,
            "query_dataset": query_dataset,
            "neighbor_dataset": neighbor_dataset if ibin_path else None,
            "base": {
                "count": base_vectors,
                "dimension": dimension,
            },
            "queries": {
                "count": query_vectors,
                "dimension": dimension,
            },
            "neighbors": {
                "count": query_vectors if ibin_path else 0,
                "k": k,
            }
            if ibin_path
            else None,
        }

        self._log(f"Wrap complete: wrote {output_path}")
        return result
