"""UI for scaling HDF5 vector datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QProgressBar,
)

from ..utils.io import HDF5Reader


class ScaleView(QWidget):
    """Interactive panel for scaling HDF5 datasets."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: list[dict[str, Any]] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([480, 420])
        layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        input_group = QGroupBox("Input HDF5")
        input_layout = QVBoxLayout(input_group)

        input_row = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Choose input .h5 file…")
        input_row.addWidget(self.input_path)
        browse_in = QPushButton("Browse…")
        browse_in.clicked.connect(self._on_browse_input)
        input_row.addWidget(browse_in)
        input_layout.addLayout(input_row)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Base dataset:"))
        self.base_dataset = QComboBox()
        dataset_row.addWidget(self.base_dataset)

        dataset_row.addWidget(QLabel("Queries dataset:"))
        self.query_dataset = QComboBox()
        dataset_row.addWidget(self.query_dataset)
        dataset_row.addStretch()
        input_layout.addLayout(dataset_row)

        neighbor_row = QHBoxLayout()
        neighbor_row.addWidget(QLabel("Neighbors dataset:"))
        self.neighbor_dataset = QComboBox()
        neighbor_row.addWidget(self.neighbor_dataset)
        neighbor_row.addStretch()
        input_layout.addLayout(neighbor_row)

        layout.addWidget(input_group)

        scale_group = QGroupBox("Scaling Options")
        scale_layout = QHBoxLayout(scale_group)

        scale_layout.addWidget(QLabel("Scale factor:"))
        self.scale_factor = QSpinBox()
        self.scale_factor.setRange(1, 64)
        self.scale_factor.setValue(2)
        scale_layout.addWidget(self.scale_factor)

        scale_layout.addWidget(QLabel("Compression:"))
        self.compression = QComboBox()
        self.compression.addItems(["None", "gzip", "lzf"])
        scale_layout.addWidget(self.compression)
        scale_layout.addStretch()
        layout.addWidget(scale_group)

        output_group = QGroupBox("Output HDF5")
        output_layout = QHBoxLayout(output_group)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Choose output .h5 path…")
        output_layout.addWidget(self.output_path)
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._on_browse_output)
        output_layout.addWidget(browse_out)
        layout.addWidget(output_group)

        action_row = QHBoxLayout()
        self.scale_btn = QPushButton("Scale Dataset")
        self.scale_btn.clicked.connect(self._on_scale)
        self.scale_btn.setEnabled(False)
        action_row.addWidget(self.scale_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        action_row.addWidget(self.cancel_btn)
        layout.addLayout(action_row)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(3)
        self.preview_table.setHorizontalHeaderLabels(["Dataset", "Shape", "DType"])
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        layout.addWidget(progress_group)

        self.status_label = QLabel("Load an HDF5 file to begin.")
        layout.addWidget(self.status_label)
        return panel

    def _on_browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 file",
            "",
            "HDF5 Files (*.h5 *.hdf5)",
        )
        if not path:
            return

        self.input_path.setText(path)
        self._load_datasets(path)
        if self.output_path.text() == "":
            path_obj = Path(path)
            suggested = path_obj.with_name(f"{path_obj.stem}_scaled{path_obj.suffix}")
            self.output_path.setText(str(suggested))
        self._update_actions()

    def _on_browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output HDF5 path",
            "",
            "HDF5 Files (*.h5 *.hdf5)",
        )
        if path:
            self.output_path.setText(path)
            self._update_actions()

    def _load_datasets(self, file_path: str) -> None:
        """Read dataset names and update selectors."""
        try:
            reader = HDF5Reader(file_path)
            metadata = reader.get_metadata()
            reader.close()
        except Exception as exc:  # pragma: no cover - UI only
            QMessageBox.critical(self, "Failed to read file", str(exc))
            return

        self._datasets = metadata.get("datasets", [])
        dataset_names = [d.get("path", "") for d in self._datasets]

        for combo in (self.base_dataset, self.query_dataset, self.neighbor_dataset):
            combo.clear()
            combo.addItems(dataset_names)

        base, queries, neighbors = self._infer_datasets(self._datasets)
        if base:
            self.base_dataset.setCurrentText(base)
        if queries:
            self.query_dataset.setCurrentText(queries)
        if neighbors:
            self.neighbor_dataset.setCurrentText(neighbors)

        self._refresh_preview_table()
        self.status_label.setText("Datasets loaded. Configure scaling options.")

    def _infer_datasets(
        self, datasets: list[dict[str, Any]]
    ) -> tuple[str | None, str | None, str | None]:
        """Heuristically choose base, query, and neighbor datasets."""

        floats = [d for d in datasets if str(d.get("dtype", "")).startswith("float")]
        ints = [d for d in datasets if str(d.get("dtype", "")).startswith("int")]

        base_candidate = None
        query_candidate = None
        neighbor_candidate = None

        if floats:
            floats_sorted = sorted(
                floats, key=lambda d: d.get("shape", [0])[0], reverse=True
            )
            base_candidate = floats_sorted[0].get("path")
            if len(floats_sorted) > 1:
                query_candidate = floats_sorted[1].get("path")

        if not query_candidate and floats:
            query_candidate = floats[0].get("path")

        query_rows = None
        if query_candidate:
            for item in datasets:
                if item.get("path") == query_candidate:
                    shape = item.get("shape", [])
                    query_rows = shape[0] if shape else None
                    break

        for item in ints:
            shape = item.get("shape", [])
            if shape and len(shape) >= 2 and (query_rows is None or shape[0] == query_rows):
                neighbor_candidate = item.get("path")
                break

        if ints and neighbor_candidate is None:
            neighbor_candidate = ints[0].get("path")

        return base_candidate, query_candidate, neighbor_candidate

    def _refresh_preview_table(self) -> None:
        self.preview_table.setRowCount(len(self._datasets))
        for row, entry in enumerate(self._datasets):
            self.preview_table.setItem(row, 0, QTableWidgetItem(str(entry.get("path", ""))))
            self.preview_table.setItem(row, 1, QTableWidgetItem(str(entry.get("shape", ""))))
            self.preview_table.setItem(row, 2, QTableWidgetItem(str(entry.get("dtype", ""))))
        self.preview_table.resizeColumnsToContents()

    def _update_actions(self) -> None:
        has_input = bool(self.input_path.text())
        has_output = bool(self.output_path.text())
        self.scale_btn.setEnabled(has_input and has_output)

    def _on_scale(self) -> None:
        input_path = self.input_path.text().strip()
        output_path = self.output_path.text().strip()
        base_ds = self.base_dataset.currentText().strip()
        query_ds = self.query_dataset.currentText().strip()
        neighbor_ds = self.neighbor_dataset.currentText().strip()

        if not input_path or not output_path:
            QMessageBox.warning(self, "Missing paths", "Please select input and output paths.")
            return

        if not (base_ds and query_ds and neighbor_ds):
            QMessageBox.warning(
                self, "Missing datasets", "Please select base, queries, and neighbors datasets."
            )
            return

        if Path(input_path).resolve() == Path(output_path).resolve():
            QMessageBox.warning(
                self, "Invalid output", "Output path must differ from the input path."
            )
            return

        options = {
            "base_dataset": base_ds,
            "query_dataset": query_ds,
            "neighbor_dataset": neighbor_ds,
            "scale_factor": self.scale_factor.value(),
            "compression": self.compression.currentText()
            if self.compression.currentText() != "None"
            else None,
        }

        main_window = self.window()
        if hasattr(main_window, "scale_dataset"):
            main_window.scale_dataset(input_path, output_path, options)
            self.scale_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_label.setText("Scaling…")

    def _on_cancel(self) -> None:
        main_window = self.window()
        if hasattr(main_window, "cancel_operation"):
            main_window.cancel_operation()

    # Public slots used by MainWindow
    def update_progress(self, current: int, total: int) -> None:
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(
                f"Scaling… {current} / {total} steps ({percent}%)"
            )

    def scaling_complete(self, result: dict) -> None:
        self.progress_bar.setValue(100)
        self.progress_label.setText("Complete")
        self.cancel_btn.setEnabled(False)
        self.scale_btn.setEnabled(True)
        self.status_label.setText(
            "Scaled base dataset written to output with regenerated neighbors."
        )
        details = (
            f"Output: {result.get('output_path')}\n"
            f"Base vectors: {result.get('base_vectors', 0):,}\n"
            f"Queries: {result.get('query_count', 0):,}\n"
            f"Neighbors shape: {result.get('neighbor_shape', '')}\n"
            f"Scale factor: {result.get('scale_factor', 1)}"
        )
        QMessageBox.information(self, "Scale complete", details)

    def scaling_error(self, error: str) -> None:
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self.cancel_btn.setEnabled(False)
        self.scale_btn.setEnabled(True)
        QMessageBox.critical(self, "Scale error", error)

