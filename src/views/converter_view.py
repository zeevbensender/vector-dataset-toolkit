"""Converter view for format conversion operations."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


from ..utils.settings import SettingsManager


class ConverterView(QWidget):
    """View for converting between vector dataset formats."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings_manager: SettingsManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings_manager
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - input/settings
        left_panel = self._create_left_panel()
        self.splitter.addWidget(left_panel)

        # Right panel - output/progress
        right_panel = self._create_right_panel()
        self.splitter.addWidget(right_panel)

        self.splitter.setSizes([400, 500])
        layout.addWidget(self.splitter)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with conversion settings."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Input file group
        input_group = QGroupBox("Input File")
        input_layout = QVBoxLayout(input_group)

        input_row = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select input file...")
        self.input_path.setReadOnly(True)
        input_row.addWidget(self.input_path)

        self.input_browse = QPushButton("Browse...")
        self.input_browse.clicked.connect(self._on_browse_input)
        input_row.addWidget(self.input_browse)
        input_layout.addLayout(input_row)

        self.input_format_label = QLabel("Format: -")
        input_layout.addWidget(self.input_format_label)

        layout.addWidget(input_group)

        # Output file group
        output_group = QGroupBox("Output File")
        output_layout = QVBoxLayout(output_group)

        output_row = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output file...")
        output_row.addWidget(self.output_path)

        self.output_browse = QPushButton("Browse...")
        self.output_browse.clicked.connect(self._on_browse_output)
        output_row.addWidget(self.output_browse)
        output_layout.addLayout(output_row)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Output Format:"))
        self.output_format = QComboBox()
        self.output_format.addItems(["HDF5 (.h5)", "NPY (.npy)", "FBIN (.fbin)"])
        self.output_format.currentIndexChanged.connect(self._on_format_changed)
        format_row.addWidget(self.output_format)
        format_row.addStretch()
        output_layout.addLayout(format_row)

        layout.addWidget(output_group)

        # Conversion options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        # Chunk size
        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Chunk Size:"))
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(100, 100000)
        self.chunk_size.setValue(10000)
        self.chunk_size.setSingleStep(1000)
        chunk_row.addWidget(self.chunk_size)
        chunk_row.addStretch()
        options_layout.addLayout(chunk_row)

        # Compression (for HDF5)
        compression_row = QHBoxLayout()
        compression_row.addWidget(QLabel("Compression:"))
        self.compression = QComboBox()
        self.compression.addItems(["gzip", "lzf", "None"])
        compression_row.addWidget(self.compression)
        compression_row.addStretch()
        options_layout.addLayout(compression_row)

        # Dataset name (for HDF5)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Dataset Name:"))
        self.dataset_name = QLineEdit("vectors")
        dataset_row.addWidget(self.dataset_name)
        options_layout.addLayout(dataset_row)

        layout.addWidget(options_group)

        # Convert button
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self._on_convert)
        layout.addWidget(self.convert_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with progress and results."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

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

        # Results group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_label = QLabel("No conversion performed yet")
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)

        layout.addWidget(results_group)

        layout.addStretch()
        return panel

    def _on_browse_input(self) -> None:
        """Handle input file browsing."""
        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            start_dir,
            "All Supported Files (*.npy *.h5 *.hdf5 *.fbin);;NPY Files (*.npy);;HDF5 Files (*.h5 *.hdf5);;FBIN Files (*.fbin)"
        )
        if file_path:
            self.input_path.setText(file_path)
            suffix = Path(file_path).suffix.lower()
            format_name = {".npy": "NPY", ".h5": "HDF5", ".hdf5": "HDF5", ".fbin": "FBIN"}.get(suffix, "Unknown")
            self.input_format_label.setText(f"Format: {format_name}")
            self._update_convert_state()
            self._update_output_format_options(format_name)
            if self._settings:
                self._settings.update_last_directory(file_path)

    def _on_browse_output(self) -> None:
        """Handle output file browsing."""
        index = self.output_format.currentIndex()
        ext_map = {0: ".h5", 1: ".npy", 2: ".fbin"}
        filter_map = {
            0: "HDF5 Files (*.h5)",
            1: "NPY Files (*.npy)",
            2: "FBIN Files (*.fbin)"
        }
        ext = ext_map.get(index, ".h5")
        filter_text = filter_map.get(index, "HDF5 Files (*.h5)")

        start_dir = self._settings.get_last_directory() if self._settings else str(Path.home())
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            start_dir,
            filter_text
        )
        if file_path:
            # Ensure correct extension
            if not file_path.endswith(ext):
                file_path += ext
            self.output_path.setText(file_path)
            self._update_convert_state()
            if self._settings:
                self._settings.update_last_directory(file_path)

    def _on_format_changed(self) -> None:
        """Handle output format change."""
        index = self.output_format.currentIndex()
        is_hdf5 = index == 0
        is_fbin = index == 2
        
        # HDF5-only options
        self.compression.setEnabled(is_hdf5)
        self.dataset_name.setEnabled(is_hdf5)
        
        # Update output path extension if set
        current_path = self.output_path.text()
        if current_path:
            path = Path(current_path)
            ext_map = {0: ".h5", 1: ".npy", 2: ".fbin"}
            new_ext = ext_map.get(index, ".h5")
            new_path = path.with_suffix(new_ext)
            self.output_path.setText(str(new_path))

    def _update_output_format_options(self, input_format: str) -> None:
        """Update output format options based on input format."""
        if input_format == "NPY":
            self.output_format.setCurrentIndex(0)  # Default to HDF5
        elif input_format == "FBIN":
            self.output_format.setCurrentIndex(1)  # Default to NPY
        else:
            self.output_format.setCurrentIndex(1)  # Default to NPY

    def _update_convert_state(self) -> None:
        """Update the convert button state."""
        has_input = bool(self.input_path.text())
        has_output = bool(self.output_path.text())
        self.convert_btn.setEnabled(has_input and has_output)

    def _on_convert(self) -> None:
        """Handle convert button click."""
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        
        if not input_path or not output_path:
            return

        main_window = self.window()
        if hasattr(main_window, "convert_file"):
            options = {
                "chunk_size": self.chunk_size.value(),
                "compression": self.compression.currentText() if self.compression.currentText() != "None" else None,
                "dataset_name": self.dataset_name.text(),
            }
            main_window.convert_file(input_path, output_path, options)
            self.convert_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_label.setText("Converting...")

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
            self.progress_label.setText(f"Converting... {current:,} / {total:,} vectors ({percent}%)")

    def conversion_complete(self, result: dict) -> None:
        """Handle conversion completion."""
        self.progress_bar.setValue(100)
        self.progress_label.setText("Complete!")
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        result_text = f"""Conversion successful!

Input: {result.get('input_path', 'N/A')}
Output: {result.get('output_path', 'N/A')}
Vectors: {result.get('vectors_converted', 'N/A'):,}
Shape: {result.get('shape', 'N/A')}
"""
        self.results_label.setText(result_text)

    def conversion_error(self, error: str) -> None:
        """Handle conversion error."""
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.results_label.setText(f"Error: {error}")
