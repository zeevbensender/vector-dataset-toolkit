"""Advanced Inspector view for detailed file inspection.

This module provides an advanced inspector panel that shows:
- Complete parsed header fields with descriptions
- Configurable sampling controls (first N, random, strided)
- Metadata export as JSON or copy to clipboard
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# Field descriptions for common metadata fields
FIELD_DESCRIPTIONS = {
    "file_path": "Full path to the file on disk",
    "format": "File format (fbin, ibin, npy, hdf5)",
    "vector_count": "Total number of vectors in the dataset",
    "dimension": "Dimensionality of each vector",
    "dtype": "Data type of vector elements (e.g., float32, int32)",
    "shape": "Shape of the data array (vectors, dimension)",
    "file_size_bytes": "Actual file size in bytes",
    "file_size_mb": "File size in megabytes",
    "expected_size_bytes": "Expected file size based on header metadata",
    "size_match": "Whether actual file size matches expected size",
    "k": "Number of nearest neighbors per query (IBIN format)",
    "compression": "Compression algorithm used (HDF5)",
    "chunks": "Chunking configuration (HDF5)",
    "dataset_count": "Number of datasets in file (HDF5)",
    "groups": "Hierarchical groups in file (HDF5)",
    "datasets": "Dataset paths and information (HDF5)",
    "ndim": "Number of array dimensions",
}


class AdvancedInspectorDialog(QDialog):
    """Dialog for advanced file inspection with detailed metadata and sampling."""

    def __init__(
        self,
        parent: QWidget | None,
        metadata: dict[str, Any],
        file_path: str,
    ) -> None:
        super().__init__(parent)
        self.metadata = metadata
        self.file_path = file_path
        self._sample_data: Any = None
        
        self.setWindowTitle(f"Advanced Inspector - {Path(file_path).name}")
        self.setMinimumSize(900, 700)
        self.resize(1000, 750)
        
        self._setup_ui()
        self._populate_metadata()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QVBoxLayout(self)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - metadata table
        left_panel = self._create_metadata_panel()
        splitter.addWidget(left_panel)

        # Right panel - sampling and preview
        right_panel = self._create_sampling_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([450, 450])
        layout.addWidget(splitter, 1)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_metadata_panel(self) -> QWidget:
        """Create the metadata panel with detailed field information."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Metadata table with descriptions
        metadata_group = QGroupBox("File Metadata")
        metadata_layout = QVBoxLayout(metadata_group)

        self.metadata_table = QTableWidget()
        self.metadata_table.setColumnCount(3)
        self.metadata_table.setHorizontalHeaderLabels([
            "Field", "Value", "Description"
        ])
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.metadata_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metadata_table.setAlternatingRowColors(True)
        metadata_layout.addWidget(self.metadata_table)

        # Export buttons
        export_row = QHBoxLayout()
        self.export_json_btn = QPushButton("Export as JSON...")
        self.export_json_btn.clicked.connect(self._on_export_json)
        export_row.addWidget(self.export_json_btn)

        self.copy_metadata_btn = QPushButton("Copy to Clipboard")
        self.copy_metadata_btn.clicked.connect(self._on_copy_metadata)
        export_row.addWidget(self.copy_metadata_btn)
        export_row.addStretch()
        metadata_layout.addLayout(export_row)

        layout.addWidget(metadata_group)

        # Format-specific info
        self.format_info_group = QGroupBox("Format Details")
        format_layout = QVBoxLayout(self.format_info_group)
        self.format_info_text = QTextEdit()
        self.format_info_text.setReadOnly(True)
        self.format_info_text.setMaximumHeight(150)
        format_layout.addWidget(self.format_info_text)
        layout.addWidget(self.format_info_group)

        return panel

    def _create_sampling_panel(self) -> QWidget:
        """Create the sampling panel with controls and preview."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Sampling controls
        sampling_group = QGroupBox("Sampling Options")
        sampling_layout = QVBoxLayout(sampling_group)

        # Sampling mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Sample Mode:"))
        self.sample_mode = QComboBox()
        self.sample_mode.addItems(["First N", "Random", "Strided"])
        self.sample_mode.currentIndexChanged.connect(self._on_sample_mode_changed)
        mode_row.addWidget(self.sample_mode)
        mode_row.addStretch()
        sampling_layout.addLayout(mode_row)

        # First N / Count options
        self.count_row = QHBoxLayout()
        self.count_row_label = QLabel("Count:")
        self.count_row.addWidget(self.count_row_label)
        self.sample_count = QSpinBox()
        self.sample_count.setRange(1, 1000)
        self.sample_count.setValue(10)
        self.count_row.addWidget(self.sample_count)
        self.count_row.addStretch()
        sampling_layout.addLayout(self.count_row)

        # Start index (for First N)
        self.start_row = QHBoxLayout()
        self.start_row.addWidget(QLabel("Start Index:"))
        self.start_index = QSpinBox()
        self.start_index.setRange(0, 1000000)
        self.start_index.setValue(0)
        self.start_row.addWidget(self.start_index)
        self.start_row.addStretch()
        sampling_layout.addLayout(self.start_row)

        # Random seed (for Random mode)
        self.seed_row = QHBoxLayout()
        self.seed_row.addWidget(QLabel("Random Seed:"))
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 999999)
        self.random_seed.setValue(42)
        self.seed_row.addWidget(self.random_seed)
        self.use_random_seed = QCheckBox("Use fixed seed")
        self.use_random_seed.setChecked(True)
        self.seed_row.addWidget(self.use_random_seed)
        self.seed_row.addStretch()
        sampling_layout.addLayout(self.seed_row)

        # Stride (for Strided mode)
        self.stride_row = QHBoxLayout()
        self.stride_row.addWidget(QLabel("Stride:"))
        self.stride_value = QSpinBox()
        self.stride_value.setRange(1, 100000)
        self.stride_value.setValue(100)
        self.stride_row.addWidget(self.stride_value)
        self.stride_row.addStretch()
        sampling_layout.addLayout(self.stride_row)

        # Sample button
        self.sample_btn = QPushButton("Load Sample")
        self.sample_btn.clicked.connect(self._on_sample)
        sampling_layout.addWidget(self.sample_btn)

        layout.addWidget(sampling_group)

        # Update visibility based on initial mode
        self._on_sample_mode_changed(0)

        # Vector preview
        preview_group = QGroupBox("Vector Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFontFamily("monospace")
        self.preview_text.setPlaceholderText(
            "Select sampling options and click 'Load Sample' to preview vectors"
        )
        preview_layout.addWidget(self.preview_text)

        # Preview info
        self.preview_info = QLabel("")
        preview_layout.addWidget(self.preview_info)

        layout.addWidget(preview_group, 1)

        return panel

    def _populate_metadata(self) -> None:
        """Populate the metadata table with field information."""
        # Flatten nested structures for display
        flat_items: list[tuple[str, str, str]] = []
        
        for key, value in self.metadata.items():
            # Skip complex nested structures for the table
            if key == "datasets" and isinstance(value, list):
                flat_items.append((
                    key,
                    f"{len(value)} dataset(s)",
                    FIELD_DESCRIPTIONS.get(key, "")
                ))
                continue
            
            if isinstance(value, (str, int, float, bool)):
                value_str = str(value)
            elif isinstance(value, (list, tuple)) and len(value) <= 5:
                value_str = str(value)
            else:
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            
            description = FIELD_DESCRIPTIONS.get(key, "")
            flat_items.append((key, value_str, description))

        self.metadata_table.setRowCount(len(flat_items))
        for row, (key, value, desc) in enumerate(flat_items):
            self.metadata_table.setItem(row, 0, QTableWidgetItem(key))
            self.metadata_table.setItem(row, 1, QTableWidgetItem(value))
            self.metadata_table.setItem(row, 2, QTableWidgetItem(desc))

        # Populate format-specific info
        format_type = self.metadata.get("format", "unknown")
        self._show_format_info(format_type)

    def _show_format_info(self, format_type: str) -> None:
        """Show format-specific information."""
        info_lines = []
        
        if format_type == "fbin":
            info_lines = [
                "FBIN Format (Binary Float32 Vectors)",
                "",
                "Header: 8 bytes",
                "  - Bytes 0-3: uint32 little-endian (vector count)",
                "  - Bytes 4-7: uint32 little-endian (dimension)",
                "",
                "Data: float32 values, row-major order",
                "",
                "Used by: ANN Benchmarks, Big ANN Benchmarks",
            ]
        elif format_type == "ibin":
            info_lines = [
                "IBIN Format (Binary Int32 Indices)",
                "",
                "Header: 8 bytes",
                "  - Bytes 0-3: uint32 little-endian (query count)",
                "  - Bytes 4-7: uint32 little-endian (k neighbors)",
                "",
                "Data: int32 values, row-major order",
                "",
                "Used by: Ground truth files for ANN benchmarks",
            ]
        elif format_type == "npy":
            info_lines = [
                "NPY Format (NumPy Array)",
                "",
                "Standard NumPy binary array format",
                "Header contains shape, dtype, and order info",
                "",
                "Widely supported by scientific Python ecosystem",
            ]
        elif format_type == "hdf5":
            info_lines = [
                "HDF5 Format (Hierarchical Data Format)",
                "",
                "Self-describing format with groups and datasets",
                "Supports compression and chunking",
                "",
                f"Datasets in file: {self.metadata.get('dataset_count', 'unknown')}",
            ]
        
        self.format_info_text.setText("\n".join(info_lines))

    def _on_sample_mode_changed(self, index: int) -> None:
        """Handle sample mode change."""
        # Show/hide relevant controls
        is_first_n = index == 0
        is_random = index == 1
        is_strided = index == 2

        # Start index only for First N
        for i in range(self.start_row.count()):
            widget = self.start_row.itemAt(i).widget()
            if widget:
                widget.setVisible(is_first_n)

        # Random seed only for Random
        for i in range(self.seed_row.count()):
            widget = self.seed_row.itemAt(i).widget()
            if widget:
                widget.setVisible(is_random)

        # Stride only for Strided
        for i in range(self.stride_row.count()):
            widget = self.stride_row.itemAt(i).widget()
            if widget:
                widget.setVisible(is_strided)

        # Update count label
        if is_strided:
            self.count_row_label.setText("Max Count:")
        else:
            self.count_row_label.setText("Count:")

    def _on_sample(self) -> None:
        """Handle sample button click."""
        main_window = self.parent()
        if main_window is None:
            main_window = self.window()

        mode = self.sample_mode.currentIndex()
        count = self.sample_count.value()

        if mode == 0:  # First N
            start = self.start_index.value()
            self._request_sample("sequential", start=start, count=count)
        elif mode == 1:  # Random
            seed = self.random_seed.value() if self.use_random_seed.isChecked() else None
            self._request_sample("random", count=count, seed=seed)
        elif mode == 2:  # Strided
            stride = self.stride_value.value()
            self._request_sample("strided", stride=stride, max_count=count)

    def _request_sample(self, method: str, **kwargs) -> None:
        """Request a sample from the main window."""
        main_window = self.parent()
        while main_window and not hasattr(main_window, "sample_file_advanced"):
            main_window = main_window.parent()

        if main_window and hasattr(main_window, "sample_file_advanced"):
            main_window.sample_file_advanced(
                self.file_path,
                method,
                callback=self._display_sample,
                **kwargs
            )
        else:
            # Fallback - try direct sampling
            self._do_direct_sample(method, **kwargs)

    def _do_direct_sample(self, method: str, **kwargs) -> None:
        """Perform direct sampling without going through main window."""
        try:
            from ..utils.io import FBINReader, IBINReader, NPYReader, HDF5Reader
            
            path = Path(self.file_path)
            suffix = path.suffix.lower()
            
            if suffix == ".fbin":
                reader = FBINReader(self.file_path)
            elif suffix == ".ibin":
                reader = IBINReader(self.file_path)
            elif suffix == ".npy":
                reader = NPYReader(self.file_path)
            elif suffix in (".h5", ".hdf5"):
                reader = HDF5Reader(self.file_path)
                # For HDF5, we need to find the dataset
                contents = reader.list_contents()
                if contents["datasets"]:
                    # Sample from first dataset
                    ds_path = contents["datasets"][0]
                    if method == "sequential":
                        data = reader.sample(ds_path, kwargs.get("start", 0), kwargs.get("count", 10))
                    else:
                        data = reader.sample(ds_path, 0, kwargs.get("count", 10))
                    self._display_sample(data, None)
                return
            else:
                self.preview_text.setText(f"Unsupported format: {suffix}")
                return
            
            # Sample based on method
            if method == "sequential":
                data = reader.sample(kwargs.get("start", 0), kwargs.get("count", 10))
                indices = None
            elif method == "random":
                if hasattr(reader, "sample_random"):
                    indices, data = reader.sample_random(kwargs.get("count", 10), kwargs.get("seed"))
                else:
                    data = reader.sample(0, kwargs.get("count", 10))
                    indices = None
            elif method == "strided":
                if hasattr(reader, "sample_strided"):
                    data = reader.sample_strided(kwargs.get("stride", 10), kwargs.get("max_count"))
                    indices = None
                else:
                    data = reader.sample(0, kwargs.get("max_count", 10))
                    indices = None
            else:
                data = reader.sample(0, 10)
                indices = None
            
            self._display_sample(data, indices)
            reader.close()
            
        except Exception as e:
            self.preview_text.setText(f"Error sampling: {e}")

    def _display_sample(self, data, indices=None) -> None:
        """Display sampled data in the preview."""
        self._sample_data = data
        
        lines = []
        for i, vector in enumerate(data):
            idx_str = f"[{indices[i]}]" if indices is not None else f"[{i}]"
            
            if len(vector) > 10:
                # Truncate long vectors
                vec_str = (
                    f"[{', '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in vector[:5])} "
                    f"... {', '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in vector[-3:])}]"
                )
            else:
                vec_str = f"[{', '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in vector)}]"
            
            lines.append(f"{idx_str}: {vec_str}")
        
        self.preview_text.setText("\n".join(lines))
        self.preview_info.setText(f"Showing {len(data)} vectors")

    def _on_export_json(self) -> None:
        """Export metadata as JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metadata as JSON",
            f"{Path(self.file_path).stem}_metadata.json",
            "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(self.metadata, f, indent=2, default=str)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Metadata exported to {file_path}"
                )
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

    def _on_copy_metadata(self) -> None:
        """Copy metadata to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(json.dumps(self.metadata, indent=2, default=str))
