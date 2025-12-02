"""Inspector view for displaying file metadata and samples."""

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class InspectorView(QWidget):
    """View for inspecting file metadata and sampling vectors."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_file: str | None = None
        self._current_reader: Any = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - file selection
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - metadata and preview
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 600])
        layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with file selection."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        self.open_btn = QPushButton("Open File...")
        self.open_btn.clicked.connect(self._on_open_file)
        file_layout.addWidget(self.open_btn)

        self.scan_btn = QPushButton("Scan File")
        self.scan_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self._on_scan_file)
        file_layout.addWidget(self.scan_btn)

        layout.addWidget(file_group)

        # Dataset selection for HDF5 files
        self.dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout(self.dataset_group)
        
        self.dataset_list = QTableWidget()
        self.dataset_list.setColumnCount(2)
        self.dataset_list.setHorizontalHeaderLabels(["Dataset", "Shape"])
        self.dataset_list.horizontalHeader().setStretchLastSection(True)
        self.dataset_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dataset_list.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.dataset_list.itemSelectionChanged.connect(self._on_dataset_selected)
        dataset_layout.addWidget(self.dataset_list)
        
        self.dataset_group.setVisible(False)
        layout.addWidget(self.dataset_group)

        # Sample controls
        sample_group = QGroupBox("Sample Preview")
        sample_layout = QVBoxLayout(sample_group)

        self.sample_btn = QPushButton("Show Sample (first 10 vectors)")
        self.sample_btn.setEnabled(False)
        self.sample_btn.clicked.connect(self._on_show_sample)
        sample_layout.addWidget(self.sample_btn)

        layout.addWidget(sample_group)

        # Advanced placeholder
        self.advanced_btn = QPushButton("Advanced Inspector...")
        self.advanced_btn.setEnabled(False)
        self.advanced_btn.setToolTip("Coming in a future milestone")
        layout.addWidget(self.advanced_btn)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with metadata table and preview."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Metadata table
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout(metadata_group)

        self.metadata_table = QTableWidget()
        self.metadata_table.setColumnCount(2)
        self.metadata_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.metadata_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        metadata_layout.addWidget(self.metadata_table)

        layout.addWidget(metadata_group)

        # Vector preview
        preview_group = QGroupBox("Vector Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFontFamily("monospace")
        self.preview_text.setPlaceholderText("Select a file and click 'Show Sample' to preview vectors")
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group)

        return panel

    def _on_open_file(self) -> None:
        """Handle file open dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Vector Dataset File",
            "",
            "All Supported Files (*.npy *.h5 *.hdf5 *.fbin *.ibin);;NPY Files (*.npy);;HDF5 Files (*.h5 *.hdf5);;FBIN Files (*.fbin);;IBIN Files (*.ibin)"
        )
        if file_path:
            self._current_file = file_path
            self.file_label.setText(file_path)
            self.scan_btn.setEnabled(True)
            self._clear_metadata()

    def _on_scan_file(self) -> None:
        """Handle file scanning."""
        if not self._current_file:
            return

        # Get the main window for worker access
        main_window = self.window()
        if hasattr(main_window, "scan_file"):
            main_window.scan_file(self._current_file)

    def display_metadata(self, metadata: dict[str, Any]) -> None:
        """Display metadata in the table.
        
        Args:
            metadata: Dictionary of metadata key-value pairs.
        """
        # Filter out complex nested structures for the simple table view
        display_items = []
        datasets = []
        
        for key, value in metadata.items():
            if key == "datasets" and isinstance(value, list):
                datasets = value
                continue
            if isinstance(value, (str, int, float, bool, tuple)):
                display_items.append((key, str(value)))
            elif isinstance(value, list) and len(value) <= 5:
                display_items.append((key, str(value)))

        self.metadata_table.setRowCount(len(display_items))
        for row, (key, value) in enumerate(display_items):
            self.metadata_table.setItem(row, 0, QTableWidgetItem(key))
            self.metadata_table.setItem(row, 1, QTableWidgetItem(str(value)))

        # Show dataset list for HDF5 files
        if datasets:
            self.dataset_list.setRowCount(len(datasets))
            for row, ds_info in enumerate(datasets):
                path_item = QTableWidgetItem(ds_info.get("path", ""))
                shape_item = QTableWidgetItem(str(ds_info.get("shape", "")))
                self.dataset_list.setItem(row, 0, path_item)
                self.dataset_list.setItem(row, 1, shape_item)
            self.dataset_group.setVisible(True)
            if datasets:
                self.dataset_list.selectRow(0)
        else:
            self.dataset_group.setVisible(False)

        self.sample_btn.setEnabled(True)
        self.advanced_btn.setEnabled(True)

    def _on_dataset_selected(self) -> None:
        """Handle dataset selection in HDF5 files."""
        selected = self.dataset_list.selectedItems()
        if selected and self._current_reader:
            dataset_path = selected[0].text()
            # Store selected dataset path for sampling
            self._selected_dataset = dataset_path

    def _on_show_sample(self) -> None:
        """Handle sample preview request."""
        if not self._current_file:
            return

        main_window = self.window()
        
        # Get selected dataset for HDF5 files
        dataset_path = None
        if hasattr(self, "_selected_dataset"):
            dataset_path = self._selected_dataset
        
        if hasattr(main_window, "sample_file"):
            main_window.sample_file(self._current_file, dataset_path)

    def display_sample(self, sample_data) -> None:
        """Display sampled vectors in the preview pane.
        
        Args:
            sample_data: NumPy array of sampled vectors.
        """
        import numpy as np
        
        # Format the sample for display
        lines = []
        for i, vector in enumerate(sample_data):
            if len(vector) > 10:
                # Truncate long vectors
                vec_str = f"[{', '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in vector[:5])} ... {', '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in vector[-3:])}]"
            else:
                vec_str = f"[{', '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in vector)}]"
            lines.append(f"[{i}]: {vec_str}")
        
        self.preview_text.setText("\n".join(lines))

    def _clear_metadata(self) -> None:
        """Clear the metadata display."""
        self.metadata_table.setRowCount(0)
        self.preview_text.clear()
        self.dataset_list.setRowCount(0)
        self.dataset_group.setVisible(False)
        self.sample_btn.setEnabled(False)
        self.advanced_btn.setEnabled(False)

    def set_reader(self, reader: Any) -> None:
        """Set the current reader for the loaded file."""
        self._current_reader = reader
