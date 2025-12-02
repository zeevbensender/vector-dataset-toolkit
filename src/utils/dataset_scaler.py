"""Utilities for scaling HDF5 vector datasets.

This module provides a helper class that reads an input HDF5 package,
tiles the base vectors by an integer scale factor, regenerates the
ground-truth neighbors to match the expanded base, and writes the
resulting datasets into a new HDF5 file using the original dataset
names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np


class DatasetScaler:
    """Scale HDF5 vector datasets by tiling the base vectors.

    The scaler expands the base dataset by repeating it ``scale_factor``
    times, keeps the queries unchanged, and regenerates the ground truth
    by offsetting the original neighbor indices for each tiled copy of
    the base vectors.
    """

    def __init__(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> None:
        self.progress_callback = progress_callback

    def _emit_progress(self, current: int, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(current, total)

    def scale_hdf5(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        base_dataset: str,
        query_dataset: str,
        neighbor_dataset: str,
        scale_factor: int,
        compression: str | None = None,
    ) -> dict:
        """Scale an HDF5 dataset by tiling the base vectors.

        Args:
            input_path: Path to the source HDF5 file.
            output_path: Destination path for the scaled HDF5 file.
            base_dataset: Dataset path for the base vectors in the input file.
            query_dataset: Dataset path for the queries in the input file.
            neighbor_dataset: Dataset path for the ground-truth neighbors.
            scale_factor: Integer factor by which to tile the base vectors.
            compression: Optional compression algorithm name for outputs.

        Returns:
            Summary information about the scaled output.
        """

        if scale_factor < 1:
            raise ValueError("Scale factor must be at least 1")

        input_path = Path(input_path)
        output_path = Path(output_path)

        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output paths must differ")

        with h5py.File(str(input_path), "r") as src, h5py.File(
            str(output_path), "w"
        ) as dst:
            if base_dataset not in src:
                raise KeyError(f"Base dataset not found: {base_dataset}")
            if query_dataset not in src:
                raise KeyError(f"Query dataset not found: {query_dataset}")
            if neighbor_dataset not in src:
                raise KeyError(f"Neighbor dataset not found: {neighbor_dataset}")

            base_ds = src[base_dataset]
            query_ds = src[query_dataset]
            neighbor_ds = src[neighbor_dataset]

            if neighbor_ds.shape[0] != query_ds.shape[0]:
                raise ValueError(
                    "Ground truth first dimension must match number of queries"
                )

            base_count = base_ds.shape[0]
            feature_shape = base_ds.shape[1:]
            scaled_base_count = base_count * scale_factor

            total_steps = scale_factor + 2
            self._emit_progress(0, total_steps)

            scaled_base = dst.create_dataset(
                base_dataset,
                shape=(scaled_base_count, *feature_shape),
                dtype=base_ds.dtype,
                compression=compression,
                chunks=base_ds.chunks,
            )

            base_chunk = base_ds.chunks[0] if base_ds.chunks else 1024
            for factor_index in range(scale_factor):
                offset = factor_index * base_count
                for start in range(0, base_count, base_chunk):
                    end = min(start + base_chunk, base_count)
                    scaled_base[offset + start : offset + end] = base_ds[start:end]
                self._emit_progress(factor_index + 1, total_steps)

            self._copy_dataset(query_ds, dst, query_dataset, compression)
            self._emit_progress(scale_factor + 1, total_steps)

            neighbors_per_query = neighbor_ds.shape[1]
            scaled_neighbors = dst.create_dataset(
                neighbor_dataset,
                shape=(neighbor_ds.shape[0], neighbors_per_query * scale_factor),
                dtype=neighbor_ds.dtype,
                compression=compression,
                chunks=(neighbor_ds.chunks[0] if neighbor_ds.chunks else None),
            )

            neighbor_chunk = neighbor_ds.chunks[0] if neighbor_ds.chunks else 1024
            for start in range(0, neighbor_ds.shape[0], neighbor_chunk):
                end = min(start + neighbor_chunk, neighbor_ds.shape[0])
                src_neighbors = neighbor_ds[start:end]
                expanded = np.empty(
                    (src_neighbors.shape[0], neighbors_per_query * scale_factor),
                    dtype=neighbor_ds.dtype,
                )
                for factor_index in range(scale_factor):
                    col_start = factor_index * neighbors_per_query
                    col_end = col_start + neighbors_per_query
                    expanded[:, col_start:col_end] = src_neighbors + (
                        factor_index * base_count
                    )
                scaled_neighbors[start:end] = expanded

            self._emit_progress(total_steps, total_steps)

            return {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "scale_factor": scale_factor,
                "base_dataset": base_dataset,
                "query_dataset": query_dataset,
                "neighbor_dataset": neighbor_dataset,
                "base_vectors": scaled_base_count,
                "query_count": int(query_ds.shape[0]),
                "neighbor_shape": (
                    int(neighbor_ds.shape[0]),
                    int(neighbors_per_query * scale_factor),
                ),
                "compression": compression or "none",
            }

    @staticmethod
    def _create_dataset(
        file: h5py.File, path: str, data: np.ndarray, compression: str | None
    ) -> None:
        """Create a dataset, ensuring parent groups exist."""

        parent_path = "/".join(path.split("/")[:-1])
        if parent_path:
            file.require_group(parent_path)

        file.create_dataset(path, data=data, compression=compression)

    @staticmethod
    def _copy_dataset(
        source: h5py.Dataset,
        destination_file: h5py.File,
        path: str,
        compression: str | None,
    ) -> None:
        """Copy an HDF5 dataset in chunks to avoid large allocations."""

        chunks = source.chunks
        chunk_size = chunks[0] if chunks else 1024

        dest = destination_file.create_dataset(
            path,
            shape=source.shape,
            dtype=source.dtype,
            compression=compression,
            chunks=chunks,
        )

        for start in range(0, source.shape[0], chunk_size):
            end = min(start + chunk_size, source.shape[0])
            dest[start:end] = source[start:end]

