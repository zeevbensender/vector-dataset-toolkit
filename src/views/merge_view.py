"""Merge view for merging FBIN shard files."""

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
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


class MergeView(QWidget):
    """View for merging FBIN shard files.
    
    Features:
    - Add/remove multiple FBIN shard files
    - Validation status per shard (compatible/incompatible)
    - Output format selection (FBIN or NPY)
    - Chunk size and checksum options
    - Dry-run preview mode
    - Progress and log display
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings_manager: SettingsManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._shard_paths: list[str] = []
        self._shard_infos: list[dict[str, Any]] = []
        self._preview_result: dict[str, Any] | None = None
        self._settings = settings_manager
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - shard list and controls
        left_panel = self._create_left_panel()
        self.splitter.addWidget(left_panel)

        # Right panel - preview, progress, logs
        right_panel = self._create_right_panel()
        self.splitter.addWidget(right_panel)

        self.splitter.setSizes([450, 450])
        layout.addWidget(self.splitter)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with shard list and merge options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Shard files group
        shard_group = QGroupBox("Shard Files")
        shard_layout = QVBoxLayout(shard_group)

        # Shard table
        self.shard_table = QTableWidget()
        self.shard_table.setColumnCount(4)
        self.shard_table.setHorizontalHeaderLabels([
            "File", "Vectors", "Dimension", "Status"
        ])
        self.shard_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.shard_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.shard_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        shard_layout.addWidget(self.shard_table)

        # Shard buttons
        btn_row = QHBoxLayout()
        self.add_shards_btn = QPushButton("Add Shards...")
        self.add_shards_btn.clicked.connect(self._on_add_shards)
        btn_row.addWidget(self.add_shards_btn)

        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self._on_remove_selected)
        btn_row.addWidget(self.remove_selected_btn)

        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self._on_clear_all)
        btn_row.addWidget(self.clear_all_btn)
        shard_layout.addLayout(btn_row)

        layout.addWidget(shard_group)

        # Output options group
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)

        # Output path
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Output File:"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output file...")
        path_row.addWidget(self.output_path)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse_output)
        path_row.addWidget(self.browse_btn)
        output_layout.addLayout(path_row)

        # Output format
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Output Format:"))
        self.output_format = QComboBox()
        self.output_format.addItems(["FBIN (.fbin)", "NPY (.npy)"])
        self.output_format.currentIndexChanged.connect(self._on_format_changed)
        format_row.addWidget(self.output_format)
        format_row.addStretch()
        output_layout.addLayout(format_row)

        # Chunk size
        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Chunk Size:"))
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(100, 100000)
        self.chunk_size.setValue(10000)
        self.chunk_size.setSingleStep(1000)
        chunk_row.addWidget(self.chunk_size)
        chunk_row.addStretch()
        output_layout.addLayout(chunk_row)

        # Checksum option
        self.compute_checksum = QCheckBox("Compute SHA256 checksum")
        output_layout.addWidget(self.compute_checksum)

        layout.addWidget(output_group)

        # Action buttons
        action_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Dry Run / Preview")
        self.preview_btn.clicked.connect(self._on_preview)
        self.preview_btn.setEnabled(False)
        action_layout.addWidget(self.preview_btn)

        self.merge_btn = QPushButton("Merge Shards")
        self.merge_btn.clicked.connect(self._on_merge)
        self.merge_btn.setEnabled(False)
        action_layout.addWidget(self.merge_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        action_layout.addWidget(self.cancel_btn)

        layout.addLayout(action_layout)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with preview and logs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Preview/Summary group
        preview_group = QGroupBox("Merge Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFontFamily("monospace")
        self.preview_text.setPlaceholderText(
            "Add shards and click 'Dry Run / Preview' to see merge details"
        )
        preview_layout.addWidget(self.preview_text)

        # Export metadata button
        export_row = QHBoxLayout()
        self.export_json_btn = QPushButton("Export as JSON")
        self.export_json_btn.clicked.connect(self._on_export_json)
        self.export_json_btn.setEnabled(False)
        export_row.addWidget(self.export_json_btn)

        self.copy_metadata_btn = QPushButton("Copy to Clipboard")
        self.copy_metadata_btn.clicked.connect(self._on_copy_metadata)
        self.copy_metadata_btn.setEnabled(False)
        export_row.addWidget(self.copy_metadata_btn)
        export_row.addStretch()
        preview_layout.addLayout(export_row)

        layout.addWidget(preview_group)

        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_group)

        # Logs group
        logs_group = QGroupBox("Merge Log")
        logs_layout = QVBoxLayout(logs_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("monospace")
        self.log_text.setMaximumHeight(120)
        logs_layout.addWidget(self.log_text)

        layout.addWidget(logs_group)

        return panel

    def _on_add_shards(self) -> None:
        """Handle adding shard files."""
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select FBIN Shard Files",
            start_dir,
            "FBIN Files (*.fbin)"
        )
        if files:
            for file_path in files:
                if file_path not in self._shard_paths:
                    self._shard_paths.append(file_path)
                    if self._settings:
                        self._settings.update_last_directory(file_path)
            self._validate_shards()

    def _on_remove_selected(self) -> None:
        """Remove selected shards from the list."""
        selected_rows = set(item.row() for item in self.shard_table.selectedItems())
        for row in sorted(selected_rows, reverse=True):
            if row < len(self._shard_paths):
                del self._shard_paths[row]
        self._validate_shards()

    def _on_clear_all(self) -> None:
        """Clear all shards from the list."""
        self._shard_paths.clear()
        self._shard_infos.clear()
        self.shard_table.setRowCount(0)
        self._update_merge_state()
        self.preview_text.clear()

    def _validate_shards(self) -> None:
        """Validate all shards and update the table."""
        main_window = self.window()
        if hasattr(main_window, "validate_shards"):
            main_window.validate_shards(self._shard_paths)

    def display_shard_validation(self, shard_infos: list[dict[str, Any]]) -> None:
        """Display shard validation results in the table.
        
        Args:
            shard_infos: List of shard info dictionaries from ShardMerger.
        """
        self._shard_infos = shard_infos
        self.shard_table.setRowCount(len(shard_infos))

        for row, info in enumerate(shard_infos):
            # File name
            path = Path(info.get("path", ""))
            name_item = QTableWidgetItem(path.name)
            name_item.setToolTip(str(path))
            self.shard_table.setItem(row, 0, name_item)

            # Vector count
            count = info.get("vector_count", 0)
            count_item = QTableWidgetItem(f"{count:,}" if count else "-")
            self.shard_table.setItem(row, 1, count_item)

            # Dimension
            dim = info.get("dimension", 0)
            dim_item = QTableWidgetItem(str(dim) if dim else "-")
            self.shard_table.setItem(row, 2, dim_item)

            # Status
            result = info.get("validation_result", "unknown")
            status_item = QTableWidgetItem(result.replace("_", " ").title())
            if result == "compatible":
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            else:
                status_item.setForeground(Qt.GlobalColor.red)
                if info.get("error_message"):
                    status_item.setToolTip(info["error_message"])
            self.shard_table.setItem(row, 3, status_item)

        self._update_merge_state()

    def _update_merge_state(self) -> None:
        """Update the enabled state of merge controls."""
        has_shards = len(self._shard_paths) >= 2
        has_output = bool(self.output_path.text())
        
        # Check if all shards are compatible
        all_compatible = all(
            info.get("validation_result") == "compatible"
            for info in self._shard_infos
        ) if self._shard_infos else False

        self.preview_btn.setEnabled(has_shards)
        self.merge_btn.setEnabled(has_shards and has_output and all_compatible)

    def _on_browse_output(self) -> None:
        """Handle output file browsing."""
        ext = ".fbin" if self.output_format.currentIndex() == 0 else ".npy"
        filter_text = "FBIN Files (*.fbin)" if ext == ".fbin" else "NPY Files (*.npy)"

        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            start_dir,
            filter_text
        )
        if file_path:
            if not file_path.endswith(ext):
                file_path += ext
            self.output_path.setText(file_path)
            self._update_merge_state()
            if self._settings:
                self._settings.update_last_directory(file_path)

    def _on_format_changed(self) -> None:
        """Handle output format change."""
        current_path = self.output_path.text()
        if current_path:
            path = Path(current_path)
            new_ext = ".fbin" if self.output_format.currentIndex() == 0 else ".npy"
            new_path = path.with_suffix(new_ext)
            self.output_path.setText(str(new_path))

    def _on_preview(self) -> None:
        """Handle dry run / preview."""
        main_window = self.window()
        if hasattr(main_window, "preview_merge"):
            output_format = "fbin" if self.output_format.currentIndex() == 0 else "npy"
            main_window.preview_merge(self._shard_paths, output_format)

    def display_preview(self, preview: dict[str, Any]) -> None:
        """Display merge preview results.
        
        Args:
            preview: Preview dictionary from ShardMerger.preview_merge().
        """
        self._preview_result = preview
        
        lines = [
            "=== Merge Preview ===",
            "",
            f"Shards: {len(preview.get('shards', []))}",
            f"Total Vectors: {preview.get('total_vectors', 0):,}",
            f"Dimension: {preview.get('dimension', 0)}",
            f"Output Format: {preview.get('output_format', '-').upper()}",
            f"Expected Output Size: {preview.get('expected_output_size_bytes', 0) / (1024*1024):.2f} MB",
            "",
        ]

        if preview.get("all_compatible"):
            lines.append("✓ All shards compatible - ready to merge")
        else:
            lines.append("✗ Some shards are incompatible:")
            for path in preview.get("incompatible_shards", [])[:5]:
                lines.append(f"  - {Path(path).name}")

        if preview.get("warnings"):
            lines.append("")
            lines.append("Warnings:")
            for warning in preview["warnings"]:
                lines.append(f"  ⚠ {warning}")

        self.preview_text.setText("\n".join(lines))
        self.export_json_btn.setEnabled(True)
        self.copy_metadata_btn.setEnabled(True)
        self._update_merge_state()

    def _on_merge(self) -> None:
        """Handle merge button click."""
        output_path = self.output_path.text()
        if not output_path:
            QMessageBox.warning(self, "Error", "Please select an output file.")
            return

        main_window = self.window()
        if hasattr(main_window, "merge_shards"):
            options = {
                "output_format": "fbin" if self.output_format.currentIndex() == 0 else "npy",
                "chunk_size": self.chunk_size.value(),
                "compute_checksum": self.compute_checksum.isChecked(),
            }
            main_window.merge_shards(self._shard_paths, output_path, options)
            self.merge_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_label.setText("Merging...")

    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        main_window = self.window()
        if hasattr(main_window, "cancel_operation"):
            main_window.cancel_operation()

    def update_progress(self, current: int, total: int) -> None:
        """Update progress display."""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(
                f"Merging... {current:,} / {total:,} vectors ({percent}%)"
            )

    def merge_complete(self, result: dict[str, Any]) -> None:
        """Handle merge completion."""
        self.progress_bar.setValue(100)
        self.progress_label.setText("Complete!")
        self.merge_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        result_text = f"""Merge successful!

Output: {result.get('output_path', 'N/A')}
Vectors: {result.get('total_vectors', 'N/A'):,}
Shards Merged: {result.get('shards_merged', 'N/A')}
File Size: {result.get('file_size_bytes', 0) / (1024*1024):.2f} MB
"""
        if result.get("checksum"):
            result_text += f"Checksum (SHA256): {result['checksum']}\n"

        self.preview_text.setText(result_text)
        self.log_text.append("Merge completed successfully")

    def merge_error(self, error: str) -> None:
        """Handle merge error."""
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self.merge_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.log_text.append(f"Error: {error}")

    def append_log(self, message: str) -> None:
        """Append a message to the log."""
        self.log_text.append(message)

    def _on_export_json(self) -> None:
        """Export preview as JSON file."""
        if not self._preview_result:
            return

        start_dir = Path(self._settings.get_last_directory()) if self._settings else Path.home()
        default_path = start_dir / "merge_preview.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Preview as JSON",
            str(default_path),
            "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(self._preview_result, f, indent=2, default=str)
                self.log_text.append(f"Exported preview to {file_path}")
                if self._settings:
                    self._settings.update_last_directory(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

    def _on_copy_metadata(self) -> None:
        """Copy preview metadata to clipboard."""
        if not self._preview_result:
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(json.dumps(self._preview_result, indent=2, default=str))
        self.log_text.append("Preview copied to clipboard")
