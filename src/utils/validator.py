"""Validation helpers for vector dataset files.

This module provides lightweight validation logic for FBIN, IBIN, and HDF5
files. It is intentionally conservative so it can run in a background worker
without blocking the UI, while still surfacing meaningful issues to the
Inspector view and the Logs panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np

from .io import FBINReader, HDF5Reader, IBINReader


Severity = str


@dataclass
class ValidationEntry:
    """A single validation check result."""

    check: str
    result: str
    details: str
    severity: Severity

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "result": self.result, "details": self.details}


@dataclass
class ValidationReport:
    """Structured validation results and associated logs."""

    format: str
    entries: list[ValidationEntry] = field(default_factory=list)
    logs: list[tuple[str, str]] = field(default_factory=list)

    def add_entry(self, check: str, result: str, details: str, severity: Severity) -> None:
        self.entries.append(ValidationEntry(check, result, details, severity))

    def add_log(self, message: str, level: str = "INFO") -> None:
        self.logs.append((message, level))

    def to_dict(self) -> dict:
        buckets: dict[str, list[dict[str, str]] | str] = {
            "format": self.format,
            "errors": [],
            "warnings": [],
            "passed": [],
            "fatal": [],
            "info": [],
        }

        for entry in self.entries:
            record = entry.to_dict()
            if entry.severity == "warning":
                buckets["warnings"].append(record)
            elif entry.severity == "error":
                buckets["errors"].append(record)
            elif entry.severity == "fatal":
                buckets["fatal"].append(record)
            elif entry.severity == "info":
                buckets["info"].append(record)
            else:
                buckets["passed"].append(record)

        return buckets


class FileValidator:
    """Validate dataset files with progress reporting."""

    def __init__(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self._progress_callback = progress_callback

    def validate(
        self,
        file_path: str,
        metadata: dict | None = None,
    ) -> ValidationReport:
        path = Path(file_path)
        suffix = path.suffix.lower()
        fmt = metadata.get("format") if metadata else None
        detected_format = fmt or self._detect_format(suffix)
        report = ValidationReport(detected_format or "unknown")

        steps = 8
        current_step = 0

        def step() -> None:
            nonlocal current_step
            current_step += 1
            if self._progress_callback:
                self._progress_callback(current_step, steps)

        if not detected_format:
            report.add_log(f"Detected format from extension: {suffix}")
            if detected_format is None:
                detected_format = self._detect_format(suffix)
                report.format = detected_format or "unknown"

        try:
            if detected_format == "fbin":
                step()
                self._validate_fbin(path, metadata, report, step)
            elif detected_format == "ibin":
                step()
                self._validate_ibin(path, metadata, report, step)
            elif detected_format in {"h5", "hdf5", "hdf"}:
                report.format = "hdf5"
                step()
                self._validate_hdf5(path, report, step)
            else:
                report.add_entry(
                    "Format detection", "Unsupported", f"Cannot validate {suffix}", "fatal"
                )
        except Exception as exc:  # pragma: no cover - defensive catch for UI safety
            report.add_entry(
                "Unexpected error", "Fatal", str(exc), "fatal"
            )
        finally:
            # Ensure progress reports completion
            if self._progress_callback:
                self._progress_callback(steps, steps)

        return report

    def _detect_format(self, suffix: str) -> str | None:
        if suffix in {".fbin"}:
            return "fbin"
        if suffix in {".ibin"}:
            return "ibin"
        if suffix in {".h5", ".hdf5", ".hdf"}:
            return "hdf5"
        return None

    def _validate_fbin(
        self,
        path: Path,
        metadata: dict | None,
        report: ValidationReport,
        step: Callable[[], None],
    ) -> None:
        report.format = "fbin"
        if not path.exists():
            report.add_entry("File exists", "Missing", "File not found", "fatal")
            return

        file_size = path.stat().st_size
        report.add_entry("File readable", "OK", "", "ok")
        if file_size <= 0:
            report.add_entry("File size", "Invalid", "File is empty", "fatal")
            return
        if file_size % 4 != 0:
            report.add_entry("Alignment", "Invalid", "Size not divisible by 4", "error")
        else:
            report.add_entry("Alignment", "OK", "Size divisible by 4", "ok")

        reader = FBINReader(str(path))
        meta = metadata or reader.get_metadata()
        expected = meta.get("expected_size_bytes")
        if expected:
            severity = "ok" if expected == file_size else "warning"
            message = "Matches expected layout" if severity == "ok" else "Size differs from header"
            report.add_entry("Header vs size", message, f"expected {expected}, got {file_size}", severity)

        dimension = meta.get("dimension")
        vector_count = meta.get("vector_count")
        if dimension and vector_count:
            if 2 <= dimension <= 32768:
                report.add_entry("Dimension range", "OK", str(dimension), "ok")
            else:
                report.add_entry("Dimension range", "Warning", str(dimension), "warning")

            estimated = vector_count * dimension * 4 + 8
            severity = "ok" if estimated == file_size else "warning"
            report.add_entry(
                "Shape matches size",
                "OK" if severity == "ok" else "Mismatch",
                f"calculated {estimated} bytes",
                severity,
            )

        step()
        try:
            sample = reader.sample(0, 1)
            if sample.size == 0:
                report.add_entry("Sample", "Invalid", "No vectors available", "error")
            else:
                vector = np.asarray(sample[0])
                if np.isnan(vector).any() or np.isinf(vector).any():
                    report.add_entry("Sample integrity", "Invalid", "NaN/Inf detected", "error")
                elif np.all(vector == 0):
                    report.add_entry("Sample integrity", "Warning", "All zeros vector", "warning")
                else:
                    report.add_entry("Sample integrity", "OK", "Values look valid", "ok")
        except Exception as exc:  # pragma: no cover - defensive for corrupted files
            report.add_entry("Sample read", "Failed", str(exc), "error")

        report.add_log(
            f"Validated FBIN: dim={meta.get('dimension')}, vectors={meta.get('vector_count')}"
        )

    def _validate_ibin(
        self,
        path: Path,
        metadata: dict | None,
        report: ValidationReport,
        step: Callable[[], None],
    ) -> None:
        report.format = "ibin"
        if not path.exists():
            report.add_entry("File exists", "Missing", "File not found", "fatal")
            return

        file_size = path.stat().st_size
        if file_size <= 0:
            report.add_entry("File size", "Invalid", "File is empty", "fatal")
            return
        if file_size % 4 != 0:
            report.add_entry("Alignment", "Invalid", "Size not divisible by 4", "error")
        else:
            report.add_entry("Alignment", "OK", "Size divisible by 4", "ok")

        reader = IBINReader(str(path))
        meta = metadata or reader.get_metadata()
        expected = meta.get("expected_size_bytes")
        if expected:
            severity = "ok" if expected == file_size else "warning"
            report.add_entry(
                "Header vs size",
                "OK" if severity == "ok" else "Mismatch",
                f"expected {expected}, got {file_size}",
                severity,
            )

        step()
        try:
            sample = reader.sample(0, 1)
            if sample.size == 0:
                report.add_entry("Sample", "Invalid", "No rows available", "error")
            else:
                row = np.asarray(sample[0])
                if (row < 0).any():
                    report.add_entry("Index bounds", "Invalid", "Negative indices detected", "error")
                elif len(set(row.tolist())) != len(row):
                    report.add_entry("Duplicates", "Warning", "Duplicate indices in row", "warning")
                else:
                    report.add_entry("Row integrity", "OK", "No duplicates or negatives", "ok")
        except Exception as exc:  # pragma: no cover - defensive for corrupted files
            report.add_entry("Sample read", "Failed", str(exc), "error")

        report.add_log(
            f"Validated IBIN: k={meta.get('k')}, rows={meta.get('vector_count')}"
        )

    def _validate_hdf5(
        self,
        path: Path,
        report: ValidationReport,
        step: Callable[[], None],
    ) -> None:
        if not path.exists():
            report.add_entry("File exists", "Missing", "File not found", "fatal")
            return

        try:
            reader = HDF5Reader(str(path))
            metadata = reader.get_metadata()
            report.format = "hdf5"
        except Exception as exc:
            report.add_entry("Open file", "Failed", str(exc), "fatal")
            return

        report.add_entry("File readable", "OK", "Opened with h5py", "ok")
        dataset_aliases = {
            "base": ["train", "base", "vectors", "x"],
            "query": ["test", "query", "queries", "q"],
            "neighbors": ["neighbors", "neigh", "gt", "knn"],
            "distances": ["distances", "dists", "metric"],
        }

        with h5py.File(str(path), "r") as f:
            found: dict[str, str] = {}
            for key, aliases in dataset_aliases.items():
                for name in aliases:
                    if name in f and isinstance(f[name], h5py.Dataset):
                        found[key] = name
                        break

            step()
            if not found.get("base"):
                report.add_entry("Base dataset", "Missing", "No base/train dataset found", "fatal")
            else:
                report.add_entry("Base dataset", "Found", found["base"], "ok")

            if found.get("query"):
                report.add_entry("Query dataset", "Found", found["query"], "ok")
            else:
                report.add_entry("Query dataset", "Missing", "No queries dataset found", "warning")

            if found.get("neighbors"):
                report.add_entry("Neighbors", "Found", found["neighbors"], "ok")
            else:
                report.add_entry("Neighbors", "Missing", "Ground truth neighbors not present", "warning")

            if found.get("distances"):
                report.add_entry("Distances", "Found", found["distances"], "ok")
            else:
                report.add_entry("Distances", "Info", "Distances not provided", "info")

            def _vector_dataset(name: str | None) -> h5py.Dataset | None:
                if name and name in f and isinstance(f[name], h5py.Dataset):
                    return f[name]
                return None

            base_ds = _vector_dataset(found.get("base"))
            query_ds = _vector_dataset(found.get("query"))
            neighbors_ds = _vector_dataset(found.get("neighbors"))
            distances_ds = _vector_dataset(found.get("distances"))

            def _check_vector_dataset(label: str, ds: h5py.Dataset | None) -> None:
                if ds is None:
                    return
                if ds.ndim != 2:
                    report.add_entry(label, "Invalid", "Dataset must be 2D", "error")
                if ds.dtype not in (np.float32, np.float64):
                    report.add_entry(
                        f"{label} dtype",
                        "Warning",
                        f"Unexpected dtype {ds.dtype}",
                        "warning",
                    )
                else:
                    report.add_entry(f"{label} dtype", "OK", str(ds.dtype), "ok")

            _check_vector_dataset("Base vectors", base_ds)
            _check_vector_dataset("Query vectors", query_ds)

            if base_ds is not None and query_ds is not None:
                if base_ds.shape[1] != query_ds.shape[1]:
                    report.add_entry(
                        "Dimension match",
                        "Mismatch",
                        f"base dim {base_ds.shape[1]} vs query dim {query_ds.shape[1]}",
                        "error",
                    )
                else:
                    report.add_entry("Dimension match", "OK", str(base_ds.shape[1]), "ok")

            if neighbors_ds is not None:
                if neighbors_ds.dtype not in (np.int32, np.int64):
                    report.add_entry(
                        "Neighbors dtype",
                        "Warning",
                        f"Unexpected dtype {neighbors_ds.dtype}",
                        "warning",
                    )
                if query_ds is not None and neighbors_ds.shape[0] != query_ds.shape[0]:
                    report.add_entry(
                        "Neighbors rows",
                        "Mismatch",
                        f"neighbors rows {neighbors_ds.shape[0]} vs queries {query_ds.shape[0]}",
                        "error",
                    )
                else:
                    report.add_entry("Neighbors rows", "OK", str(neighbors_ds.shape[0]), "ok")

            if distances_ds is not None:
                if distances_ds.dtype not in (np.float32, np.float64):
                    report.add_entry(
                        "Distances dtype",
                        "Warning",
                        f"Unexpected dtype {distances_ds.dtype}",
                        "warning",
                    )
                if neighbors_ds is not None and distances_ds.shape != neighbors_ds.shape:
                    report.add_entry(
                        "Distances shape",
                        "Mismatch",
                        f"{distances_ds.shape} vs neighbors {neighbors_ds.shape}",
                        "error",
                    )
                else:
                    report.add_entry("Distances shape", "OK", str(distances_ds.shape), "ok")

            step()
            # Sample small slices for integrity checks
            samples: Iterable[h5py.Dataset | None] = (
                base_ds,
                query_ds,
            )
            for ds in samples:
                if ds is None:
                    continue
                try:
                    vector = np.asarray(ds[0])
                    if np.isnan(vector).any() or np.isinf(vector).any():
                        report.add_entry(
                            f"{ds.name} sample", "Invalid", "NaN/Inf detected", "error"
                        )
                    elif np.all(vector == 0):
                        report.add_entry(
                            f"{ds.name} sample", "Warning", "All zeros row", "warning"
                        )
                    else:
                        report.add_entry(
                            f"{ds.name} sample", "OK", "Values look valid", "ok"
                        )
                except Exception as exc:  # pragma: no cover - corrupted datasets
                    report.add_entry(f"{ds.name} sample", "Failed", str(exc), "error")

        report.add_log(
            f"Validated HDF5: datasets={metadata.get('dataset_count', 0)}, size={metadata.get('file_size_mb', 0)} MB"
        )
