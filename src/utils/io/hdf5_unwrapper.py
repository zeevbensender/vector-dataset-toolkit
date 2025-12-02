"""Utilities for extracting FBIN/IBIN datasets from HDF5 sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from .fbin_writer import FBINWriter
from .fbin_reader import FBINReader
from .ibin_reader import IBINReader
from .ibin_writer import IBINWriter


DATASET_SYNONYMS = {
    "train": ["train", "base", "vectors", "x"],
    "test": ["test", "query", "queries", "q"],
    "neighbors": ["neighbors", "neigh", "gt", "knn"],
    "distances": ["distances", "dists", "metric"],
}


@dataclass
class DetectedDatasets:
    train: str
    test: str
    neighbors: str
    distances: str | None


class HDF5Unwrapper:
    """Extract vector datasets from HDF5 files into FBIN/IBIN outputs."""

    def __init__(
        self,
        max_vectors: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.max_vectors = max_vectors
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.log_callback:
            self.log_callback(message, level)

    def _list_datasets(self, file: h5py.File) -> list[str]:
        paths: list[str] = []

        def visitor(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                paths.append(name)

        file.visititems(visitor)
        return paths

    def _resolve_dataset(self, paths: list[str], key: str) -> str | None:
        synonyms = DATASET_SYNONYMS[key]
        for candidate in synonyms:
            for path in paths:
                if path == candidate or path.split("/")[-1] == candidate:
                    return path
        return None

    def _detect_datasets(self, file: h5py.File) -> DetectedDatasets:
        paths = self._list_datasets(file)
        train_path = self._resolve_dataset(paths, "train")
        test_path = self._resolve_dataset(paths, "test")
        neighbors_path = self._resolve_dataset(paths, "neighbors")
        distances_path = self._resolve_dataset(paths, "distances")

        missing = []
        if not train_path:
            missing.append("base/train dataset (train/base/vectors/x)")
        if not test_path:
            missing.append("query dataset (test/query/queries/q)")
        if not neighbors_path:
            missing.append("ground-truth neighbors (neighbors/neigh/gt/knn)")

        if missing:
            raise ValueError("Missing required dataset(s): " + ", ".join(missing))

        return DetectedDatasets(
            train=train_path,
            test=test_path,
            neighbors=neighbors_path,
            distances=distances_path,
        )

    def scan(self, file_path: str | Path) -> dict:
        """Scan an HDF5 file and return metadata and dataset matches."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with h5py.File(path, "r") as h5:
            datasets = self._detect_datasets(h5)
            file_size = path.stat().st_size

            def meta_for(ds_path: str) -> dict:
                ds = h5[ds_path]
                return {
                    "path": ds_path,
                    "shape": ds.shape,
                    "dtype": str(ds.dtype),
                    "ndim": ds.ndim,
                }

            return {
                "file_path": str(path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "datasets": {
                    "train": meta_for(datasets.train),
                    "test": meta_for(datasets.test),
                    "neighbors": meta_for(datasets.neighbors),
                    "distances": meta_for(datasets.distances) if datasets.distances else None,
                },
                "max_vectors": self.max_vectors,
            }

    def _apply_limit(self, dataset: h5py.Dataset, name: str) -> np.ndarray:
        count = dataset.shape[0]
        limit = self.max_vectors if self.max_vectors and self.max_vectors > 0 else count
        actual = min(count, limit)
        self._log(f"Reading {actual:,} vectors from {name} (limit={self.max_vectors or 'all'})")
        return dataset[:actual]

    def _validate_shapes(self, train: np.ndarray, test: np.ndarray, neighbors: np.ndarray) -> None:
        if train.ndim != 2 or test.ndim != 2:
            raise ValueError("Train and test datasets must be 2D arrays")
        if neighbors.ndim != 2:
            raise ValueError("Neighbors dataset must be 2D array")
        if neighbors.shape[0] != test.shape[0]:
            raise ValueError(
                f"Neighbors row count {neighbors.shape[0]} does not match queries {test.shape[0]}"
            )

    def _write_outputs(
        self,
        train: np.ndarray,
        test: np.ndarray,
        neighbors: np.ndarray,
        output_dir: Path,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_path = output_dir / "base.fbin"
        query_path = output_dir / "queries.fbin"
        gt_path = output_dir / "gt.ibin"

        total = train.shape[0] + test.shape[0] + neighbors.shape[0]

        def stage_progress(offset: int, current: int, total_stage: int) -> None:
            if self.progress_callback and total > 0:
                overall = offset + current
                self.progress_callback(min(overall, total), total)

        base_writer = FBINWriter(base_path, progress_callback=lambda c, t: stage_progress(0, c, t))
        query_writer = FBINWriter(
            query_path, progress_callback=lambda c, t: stage_progress(train.shape[0], c, t)
        )
        ibin_writer = IBINWriter(
            gt_path,
            progress_callback=lambda c, t: stage_progress(train.shape[0] + test.shape[0], c, t),
        )

        base_result = base_writer.write(train)
        query_result = query_writer.write(test)
        gt_result = ibin_writer.write(neighbors)

        return {
            "base": base_result,
            "queries": query_result,
            "gt": gt_result,
        }

    def _validate_outputs(
        self, output_dir: Path, expected_dim: int, query_rows: int, neighbor_k: int
    ) -> dict:
        base_path = output_dir / "base.fbin"
        query_path = output_dir / "queries.fbin"
        gt_path = output_dir / "gt.ibin"

        base_meta = FBINReader(base_path).get_metadata()
        query_meta = FBINReader(query_path).get_metadata()
        gt_meta = IBINReader(gt_path).get_metadata()

        errors: list[str] = []
        if base_meta["dimension"] != expected_dim:
            errors.append(
                f"Base dimension {base_meta['dimension']} does not match expected {expected_dim}"
            )
        if query_meta["dimension"] != expected_dim:
            errors.append(
                f"Query dimension {query_meta['dimension']} does not match expected {expected_dim}"
            )
        if query_meta["vector_count"] != query_rows:
            errors.append(
                f"Query count {query_meta['vector_count']} does not match expected {query_rows}"
            )
        if gt_meta["vector_count"] != query_rows:
            errors.append(
                f"Ground truth rows {gt_meta['vector_count']} does not match queries {query_rows}"
            )
        if gt_meta["k"] != neighbor_k:
            errors.append(f"Ground truth k {gt_meta['k']} does not match expected {neighbor_k}")

        return {
            "base": base_meta,
            "queries": query_meta,
            "gt": gt_meta,
            "errors": errors,
            "valid": not errors,
        }

    def extract(self, file_path: str | Path, output_dir: str | Path | None = None) -> dict:
        """Extract datasets into FBIN/IBIN outputs."""
        path = Path(file_path)
        if output_dir is None:
            output_dir = path.parent
        output_dir = Path(output_dir)

        with h5py.File(path, "r") as h5:
            datasets = self._detect_datasets(h5)
            train_ds = h5[datasets.train]
            test_ds = h5[datasets.test]
            neighbors_ds = h5[datasets.neighbors]
            distances_ds = h5[datasets.distances] if datasets.distances else None

            train = self._apply_limit(train_ds, datasets.train)
            test = self._apply_limit(test_ds, datasets.test)
            neighbors = self._apply_limit(neighbors_ds, datasets.neighbors)

            self._validate_shapes(train, test, neighbors)

            if distances_ds is not None:
                distances = self._apply_limit(distances_ds, datasets.distances or "distances")
                if distances.shape[:2] != neighbors.shape:
                    raise ValueError(
                        f"Distances shape {distances.shape} does not match neighbors {neighbors.shape}"
                    )
                self._log("Distances dataset found; validation enabled")
            else:
                distances = None
                self._log("No distances dataset found; skipping distance validation", "WARNING")

            if self._cancelled:
                raise RuntimeError("Extraction cancelled")

            outputs = self._write_outputs(train.astype(np.float32), test.astype(np.float32), neighbors.astype(np.int32), output_dir)

            try:
                validation = self._validate_outputs(
                    output_dir,
                    expected_dim=train.shape[1],
                    query_rows=test.shape[0],
                    neighbor_k=neighbors.shape[1],
                )
            except Exception:
                for target in (output_dir / "base.fbin", output_dir / "queries.fbin", output_dir / "gt.ibin"):
                    if target.exists():
                        target.unlink()
                raise

            if not validation["valid"]:
                for target in (output_dir / "base.fbin", output_dir / "queries.fbin", output_dir / "gt.ibin"):
                    if target.exists():
                        target.unlink()
                raise ValueError("Validation failed: " + "; ".join(validation["errors"]))

            summary = {
                "output_dir": str(output_dir),
                "base_path": str(output_dir / "base.fbin"),
                "queries_path": str(output_dir / "queries.fbin"),
                "gt_path": str(output_dir / "gt.ibin"),
                "vectors": {
                    "base": train.shape,
                    "queries": test.shape,
                },
                "neighbors": neighbors.shape,
                "validation": validation,
                "distances_found": distances is not None,
            }

            if distances is not None:
                summary["distances_shape"] = distances.shape

            return summary
