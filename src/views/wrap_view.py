"""View for wrapping FBIN/IBIN inputs into HDF5 packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)


from ..utils.settings import SettingsManager


class WrapView(QWidget):
    """UI for the guided FBIN/IBIN → HDF5 wrapping flow."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings_manager: SettingsManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._fbin_paths: list[str] = []
        self._last_validation: dict[str, Any] | None = None
        self._settings = settings_manager
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self._create_left_panel())
        self.splitter.addWidget(self._create_right_panel())
        self.splitter.setSizes([480, 420])
        layout.addWidget(self.splitter)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Base/train FBIN picker (supports shards)
        fbin_group = QGroupBox("Base FBINs (train/base, shards allowed)")
        fbin_layout = QVBoxLayout(fbin_group)

        self.fbin_table = QTableWidget()
        self.fbin_table.setColumnCount(4)
        self.fbin_table.setHorizontalHeaderLabels([
            "File",
            "Vectors",
            "Dimension",
            "Status",
        ])
        self.fbin_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.fbin_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        fbin_layout.addWidget(self.fbin_table)

        fbin_buttons = QHBoxLayout()
        add_btn = QPushButton("Add FBINs…")
        add_btn.clicked.connect(self._on_add_fbins)
        fbin_buttons.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove_selected)
        fbin_buttons.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_all)
        fbin_buttons.addWidget(clear_btn)
        fbin_layout.addLayout(fbin_buttons)
        layout.addWidget(fbin_group)

        # Queries FBIN (single)
        query_group = QGroupBox("Queries FBIN (test)")
        query_layout = QHBoxLayout(query_group)
        self.query_path = QLineEdit()
        self.query_path.setPlaceholderText("Select queries.fbin…")
        self.query_path.textChanged.connect(self._update_actions)
        query_layout.addWidget(self.query_path)
        query_browse = QPushButton("Browse…")
        query_browse.clicked.connect(self._on_browse_queries)
        query_layout.addWidget(query_browse)
        clear_query = QPushButton("Clear")
        clear_query.clicked.connect(lambda: self.query_path.setText(""))
        query_layout.addWidget(clear_query)
        layout.addWidget(query_group)

        # IBIN optional input
        ibin_group = QGroupBox("Optional IBIN (ground truth)")
        ibin_layout = QHBoxLayout(ibin_group)
        self.ibin_path = QLineEdit()
        self.ibin_path.setPlaceholderText("Select IBIN file (optional)…")
        ibin_layout.addWidget(self.ibin_path)
        ibin_browse = QPushButton("Browse…")
        ibin_browse.clicked.connect(self._on_browse_ibin)
        ibin_layout.addWidget(ibin_browse)
        clear_ibin = QPushButton("Clear")
        clear_ibin.clicked.connect(lambda: self.ibin_path.setText(""))
        ibin_layout.addWidget(clear_ibin)
        layout.addWidget(ibin_group)

        # Output options
        output_group = QGroupBox("Output HDF5")
        output_layout = QVBoxLayout(output_group)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output File:"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Choose .h5 destination…")
        output_row.addWidget(self.output_path)
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._on_browse_output)
        output_row.addWidget(browse_out)
        output_layout.addLayout(output_row)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Base dataset:"))
        self.base_dataset = QLineEdit("base")
        dataset_row.addWidget(self.base_dataset)
        dataset_row.addWidget(QLabel("Train alias:"))
        self.train_dataset = QLineEdit("train")
        dataset_row.addWidget(self.train_dataset)
        output_layout.addLayout(dataset_row)

        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("Queries dataset:"))
        self.query_dataset = QLineEdit("test")
        query_row.addWidget(self.query_dataset)
        query_row.addWidget(QLabel("Neighbors dataset:"))
        self.neighbor_dataset = QLineEdit("neighbors")
        query_row.addWidget(self.neighbor_dataset)
        output_layout.addLayout(query_row)

        compression_row = QHBoxLayout()
        compression_row.addWidget(QLabel("Compression:"))
        self.compression = QComboBox()
        self.compression.addItems(["None", "gzip", "lzf"])
        compression_row.addWidget(self.compression)
        compression_row.addStretch()
        output_layout.addLayout(compression_row)

        layout.addWidget(output_group)

        action_row = QHBoxLayout()
        self.validate_btn = QPushButton("Validate Inputs")
        self.validate_btn.clicked.connect(self._on_validate)
        action_row.addWidget(self.validate_btn)

        self.wrap_btn = QPushButton("Wrap Into HDF5")
        self.wrap_btn.clicked.connect(self._on_wrap)
        self.wrap_btn.setEnabled(False)
        action_row.addWidget(self.wrap_btn)

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

        preview_group = QGroupBox("Preview & Validation")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(2)
        self.preview_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.preview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.preview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        preview_layout.addWidget(self.preview_table)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlaceholderText("Validation messages will appear here…")
        preview_layout.addWidget(self.status_text)
        layout.addWidget(preview_group)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        layout.addWidget(progress_group)

        logs_group = QGroupBox("Wrap Log")
        logs_layout = QVBoxLayout(logs_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("monospace")
        self.log_text.setPlaceholderText("Progress and log messages will stream here…")
        logs_layout.addWidget(self.log_text)
        layout.addWidget(logs_group)

        return panel

    # UI event handlers
    def _on_add_fbins(self) -> None:
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select FBIN Files", start_dir, "FBIN Files (*.fbin)"
        )
        if files:
            for f in files:
                if f not in self._fbin_paths:
                    self._fbin_paths.append(f)
            if self._settings:
                self._settings.update_last_directory(files[0])
            self._refresh_fbin_table()
            self._update_actions()

    def _on_remove_selected(self) -> None:
        selected_rows = {idx.row() for idx in self.fbin_table.selectionModel().selectedRows()}
        if not selected_rows:
            return
        self._fbin_paths = [p for i, p in enumerate(self._fbin_paths) if i not in selected_rows]
        self._refresh_fbin_table()
        self._update_actions()

    def _on_clear_all(self) -> None:
        self._fbin_paths.clear()
        self._refresh_fbin_table()
        self._update_actions()

    def _on_browse_ibin(self) -> None:
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select IBIN File", start_dir, "IBIN Files (*.ibin)"
        )
        if path:
            self.ibin_path.setText(path)
            if self._settings:
                self._settings.update_last_directory(path)

    def _on_browse_queries(self) -> None:
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Queries FBIN", start_dir, "FBIN Files (*.fbin)"
        )
        if path:
            self.query_path.setText(path)
            self._update_actions()
            if self._settings:
                self._settings.update_last_directory(path)

    def _on_browse_output(self) -> None:
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Select Output HDF5", start_dir, "HDF5 Files (*.h5 *.hdf5)"
        )
        if path:
            self.output_path.setText(path)
            self._update_actions()
            if self._settings:
                self._settings.update_last_directory(path)

    def _on_validate(self) -> None:
        if not self._fbin_paths:
            QMessageBox.warning(self, "Validation", "Please add at least one FBIN file.")
            return
        if not self.query_path.text():
            QMessageBox.warning(self, "Validation", "Please select a queries FBIN file.")
            return
        main_window = self.window()
        if hasattr(main_window, "validate_wrap_inputs"):
            self.status_text.clear()
            main_window.validate_wrap_inputs(
                self._fbin_paths, self.query_path.text(), self.ibin_path.text() or None
            )

    def _on_wrap(self) -> None:
        if not self._last_validation or not self._last_validation.get("valid"):
            QMessageBox.information(self, "Wrap", "Validate inputs before wrapping.")
            return

        main_window = self.window()
        if hasattr(main_window, "wrap_into_hdf5"):
            self.wrap_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            options = {
                "base_dataset": self.base_dataset.text() or "base",
                "train_dataset": self.train_dataset.text() or "train",
                "query_dataset": self.query_dataset.text() or "test",
                "neighbor_dataset": self.neighbor_dataset.text() or "neighbors",
                "compression": None if self.compression.currentText() == "None" else self.compression.currentText(),
                "ibin_path": self.ibin_path.text() or None,
                "include_train_alias": bool(self.train_dataset.text()),
            }
            main_window.wrap_into_hdf5(
                self._fbin_paths, self.query_path.text(), self.output_path.text(), options
            )

    def _on_cancel(self) -> None:
        main_window = self.window()
        if hasattr(main_window, "cancel_operation"):
            main_window.cancel_operation()

    # UI helpers
    def _refresh_fbin_table(self) -> None:
        self.fbin_table.setRowCount(len(self._fbin_paths))
        for row, path in enumerate(self._fbin_paths):
            item = QTableWidgetItem(Path(path).name)
            item.setToolTip(path)
            self.fbin_table.setItem(row, 0, item)
            self.fbin_table.setItem(row, 1, QTableWidgetItem("-"))
            self.fbin_table.setItem(row, 2, QTableWidgetItem("-"))
            self.fbin_table.setItem(row, 3, QTableWidgetItem("Pending"))

    def _update_actions(self) -> None:
        has_inputs = bool(self._fbin_paths)
        has_queries = bool(self.query_path.text())
        self.validate_btn.setEnabled(has_inputs and has_queries)
        self.wrap_btn.setEnabled(
            has_inputs
            and has_queries
            and bool(self.output_path.text())
            and bool(self._last_validation and self._last_validation.get("valid"))
        )

    # Public slots used by MainWindow
    def display_validation(self, result: dict[str, Any]) -> None:
        self._last_validation = result
        self.fbin_table.setRowCount(len(result.get("base_files", [])))
        for row, meta in enumerate(result.get("base_files", [])):
            name_item = QTableWidgetItem(Path(meta.get("path", "")).name)
            name_item.setToolTip(str(meta.get("path", "")))
            self.fbin_table.setItem(row, 0, name_item)
            self.fbin_table.setItem(row, 1, QTableWidgetItem(f"{meta.get('vector_count', 0):,}"))
            self.fbin_table.setItem(row, 2, QTableWidgetItem(str(meta.get("dimension", "-"))))
            status_text = meta.get("status", "-")
            if meta.get("message"):
                status_text += f" ({meta['message']})"
            self.fbin_table.setItem(row, 3, QTableWidgetItem(status_text))

        query_meta = result.get("query_file")
        if query_meta:
            self.query_path.setText(str(query_meta.get("path", "")))

        # Preview summary
        summary = [
            ("Base vectors", f"{result.get('total_base_vectors', 0):,}"),
            ("Query vectors", f"{result.get('total_query_vectors', 0):,}"),
            ("Dimension", result.get("dimension", "-")),
        ]
        if result.get("k"):
            summary.append(("k (neighbors)", result.get("k")))
        if result.get("estimated_bytes"):
            est_mb = result["estimated_bytes"] / (1024 * 1024)
            summary.append(("Estimated size", f"{est_mb:.2f} MB"))

        self.preview_table.setRowCount(len(summary))
        for row, (label, value) in enumerate(summary):
            self.preview_table.setItem(row, 0, QTableWidgetItem(str(label)))
            self.preview_table.setItem(row, 1, QTableWidgetItem(str(value)))

        messages: list[str] = []
        messages.extend(result.get("warnings", []))
        messages.extend(result.get("errors", []))
        if not messages:
            messages.append("Validation succeeded.")

        self.status_text.setPlainText("\n".join(messages))
        self.wrap_btn.setEnabled(result.get("valid", False) and bool(self.output_path.text()))

    def update_progress(self, current: int, total: int) -> None:
        if total:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Wrapping… {percent}%")

    def wrap_complete(self, result: dict[str, Any]) -> None:
        self.progress_bar.setValue(100)
        self.progress_label.setText("Wrap complete")
        self.cancel_btn.setEnabled(False)
        self.wrap_btn.setEnabled(True)
        msg_lines = ["HDF5 package created:", result.get("output_path", "-")]
        if result.get("base_dataset"):
            msg_lines.append(f"Base dataset: {result['base_dataset']}")
        if result.get("train_dataset"):
            msg_lines.append(f"Train alias: {result['train_dataset']}")
        if result.get("query_dataset"):
            msg_lines.append(f"Queries dataset: {result['query_dataset']}")
        if result.get("neighbor_dataset"):
            msg_lines.append(f"Neighbors dataset: {result['neighbor_dataset']}")
        self.log_text.append("\n".join(msg_lines))

    def wrap_error(self, message: str) -> None:
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self.cancel_btn.setEnabled(False)
        self.wrap_btn.setEnabled(True)
        QMessageBox.critical(self, "Wrap Error", message)

    def append_log(self, message: str) -> None:
        self.log_text.append(message)

