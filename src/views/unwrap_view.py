"""UI panel for extracting datasets from HDF5 files (Unwrap)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
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


from ..utils.settings import SettingsManager


class UnwrapView(QWidget):
    """Panel that extracts datasets from HDF5 files into FBIN/IBIN outputs."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings_manager: SettingsManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._current_scan: dict[str, Any] | None = None
        self._settings = settings_manager
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self._create_left_panel())
        self.splitter.addWidget(self._create_right_panel())
        self.splitter.setSizes([460, 480])
        layout.addWidget(self.splitter)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File selection
        file_group = QGroupBox("HDF5 Input")
        file_layout = QHBoxLayout(file_group)
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select .h5 / .hdf5 file…")
        self.input_path.setReadOnly(True)
        file_layout.addWidget(self.input_path)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_input)
        file_layout.addWidget(browse_btn)
        layout.addWidget(file_group)

        # Output directory
        output_group = QGroupBox("Output Directory (optional)")
        output_layout = QHBoxLayout(output_group)
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Defaults to the input file directory")
        output_layout.addWidget(self.output_dir)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._on_browse_output)
        output_layout.addWidget(out_browse)
        layout.addWidget(output_group)

        # Settings
        settings_group = QGroupBox("Settings")
        form = QFormLayout(settings_group)
        self.max_vectors = QSpinBox()
        self.max_vectors.setRange(0, 1_000_000_000)
        self.max_vectors.setValue(0)
        self.max_vectors.setSpecialValueText("All")
        form.addRow("Max vectors to extract (0 = all)", self.max_vectors)
        layout.addWidget(settings_group)

        # Actions
        action_row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan HDF5")
        self.scan_btn.clicked.connect(self._on_scan)
        action_row.addWidget(self.scan_btn)

        self.extract_btn = QPushButton("Extract Datasets")
        self.extract_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self._on_extract)
        action_row.addWidget(self.extract_btn)
        layout.addLayout(action_row)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        meta_group = QGroupBox("Detected Metadata")
        meta_layout = QVBoxLayout(meta_group)
        self.meta_table = QTableWidget()
        self.meta_table.setColumnCount(2)
        self.meta_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.meta_table.horizontalHeader().setStretchLastSection(True)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        meta_layout.addWidget(self.meta_table)
        layout.addWidget(meta_group)

        summary_group = QGroupBox("Extraction Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Run an extraction to see results…")
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(summary_group)

        log_group = QGroupBox("Unwrap Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)
        self.log_text.setPlaceholderText("Progress and validation results will appear here…")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        layout.addStretch()
        return panel

    def _on_browse_input(self) -> None:
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File",
            start_dir,
            "HDF5 Files (*.h5 *.hdf5)",
        )
        if file_path:
            self.input_path.setText(file_path)
            self.extract_btn.setEnabled(False)
            self.meta_table.setRowCount(0)
            self.summary_text.clear()
            if self._settings:
                self._settings.update_last_directory(file_path)

    def _on_browse_output(self) -> None:
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start_dir
        )
        if directory:
            self.output_dir.setText(directory)
            if self._settings:
                self._settings.update_last_directory(directory)

    def _on_scan(self) -> None:
        if not self.input_path.text():
            QMessageBox.warning(self, "Select file", "Please select an HDF5 file to scan.")
            return
        self.extract_btn.setEnabled(False)
        main_window = self.window()
        if hasattr(main_window, "scan_hdf5_for_unwrap"):
            main_window.scan_hdf5_for_unwrap(self.input_path.text(), self.max_vectors.value())
            self.log_text.append("Started scanning HDF5 file…")

    def _on_extract(self) -> None:
        if not self.input_path.text():
            QMessageBox.warning(self, "Select file", "Please select an HDF5 file to extract.")
            return

        output_dir = Path(self.output_dir.text()) if self.output_dir.text() else Path(self.input_path.text()).parent
        base_path = output_dir / "base.fbin"
        query_path = output_dir / "queries.fbin"
        gt_path = output_dir / "gt.ibin"

        if any(p.exists() for p in (base_path, query_path, gt_path)):
            reply = QMessageBox.warning(
                self,
                "Overwrite?",
                "Output files already exist. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        main_window = self.window()
        if hasattr(main_window, "extract_hdf5_datasets"):
            self.extract_btn.setEnabled(False)
            self.scan_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            main_window.extract_hdf5_datasets(
                self.input_path.text(),
                self.output_dir.text() or None,
                self.max_vectors.value(),
            )
            self.log_text.append("Extraction started…")

    def _on_cancel(self) -> None:
        main_window = self.window()
        if hasattr(main_window, "cancel_operation"):
            main_window.cancel_operation()
        self.cancel_btn.setEnabled(False)

    def display_metadata(self, metadata: dict[str, Any]) -> None:
        self._current_scan = metadata
        rows = []
        file_size_mb = metadata.get("file_size_mb")
        rows.append(("File", metadata.get("file_path", "")))
        if file_size_mb is not None:
            rows.append(("File size", f"{file_size_mb} MB"))

        datasets = metadata.get("datasets", {})
        for key in ("train", "test", "neighbors", "distances"):
            info = datasets.get(key)
            if not info:
                continue
            rows.append((f"{key.title()} path", info.get("path", "-")))
            rows.append((f"{key.title()} shape", str(info.get("shape"))))
            rows.append((f"{key.title()} dtype", str(info.get("dtype"))))

        self.meta_table.setRowCount(len(rows))
        for i, (field, value) in enumerate(rows):
            self.meta_table.setItem(i, 0, QTableWidgetItem(field))
            self.meta_table.setItem(i, 1, QTableWidgetItem(str(value)))
        self.meta_table.resizeRowsToContents()
        self.extract_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)

    def display_summary(self, summary: dict[str, Any]) -> None:
        text = [
            "Extraction completed successfully.",
            f"Base: {summary.get('base_path')}",
            f"Queries: {summary.get('queries_path')}",
            f"Ground truth: {summary.get('gt_path')}",
            f"Base vectors: {summary.get('vectors', {}).get('base')}",
            f"Query vectors: {summary.get('vectors', {}).get('queries')}",
            f"Neighbors shape: {summary.get('neighbors')}",
        ]
        if summary.get("distances_found"):
            text.append(f"Distances shape: {summary.get('distances_shape')}")
        validation = summary.get("validation", {})
        if validation.get("valid"):
            text.append("Validation: OK")
        else:
            text.append("Validation: FAILED")
            for err in validation.get("errors", []):
                text.append(f"- {err}")

        self.summary_text.setText("\n".join(text))
        self.cancel_btn.setEnabled(False)
        self.extract_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)

    def append_log(self, message: str, level: str = "INFO") -> None:
        self.log_text.append(f"[{level}] {message}")

    def extraction_error(self, error: str) -> None:
        self.cancel_btn.setEnabled(False)
        self.extract_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        QMessageBox.critical(self, "Extraction Error", error)
        self.append_log(error, "ERROR")

    def toggle_busy(self, busy: bool) -> None:
        self.scan_btn.setEnabled(not busy)
        self.extract_btn.setEnabled(not busy and self.input_path.text() != "")
        self.cancel_btn.setEnabled(busy)
