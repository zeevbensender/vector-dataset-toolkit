"""Main window for the Vector Dataset Toolkit application."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..utils.io import (
    Converter,
    FBINReader,
    HDF5Reader,
    IBINReader,
    NPYReader,
)
from ..views.converter_view import ConverterView
from ..views.inspector_view import InspectorView
from ..views.logs_view import LogsView
from ..views.merge_view import MergeView
from ..views.settings_view import SettingsView
from ..workers.worker import Worker, WorkerManager


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vector Dataset Tool")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self._worker_manager = WorkerManager()
        self._current_worker: Worker | None = None
        self._current_reader: Any = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create toolbar
        self._create_toolbar()

        # Main content area with sidebar
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left sidebar
        self.sidebar = self._create_sidebar()
        content_splitter.addWidget(self.sidebar)

        # Stacked widget for views
        self.view_stack = QStackedWidget()
        self._create_views()
        content_splitter.addWidget(self.view_stack)

        # Set sidebar to fixed width initially
        content_splitter.setSizes([180, 1020])
        main_layout.addWidget(content_splitter, 1)

        # Bottom dock with logs and progress
        bottom_dock = self._create_bottom_dock()
        main_layout.addWidget(bottom_dock)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_toolbar(self) -> None:
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.action_open = toolbar.addAction("Open File")
        self.action_open.triggered.connect(self._on_open_file)

        toolbar.addSeparator()

        self.action_refresh = toolbar.addAction("Refresh")
        self.action_refresh.triggered.connect(self._on_refresh)

    def _create_sidebar(self) -> QWidget:
        """Create the left sidebar navigation."""
        sidebar = QWidget()
        sidebar.setMaximumWidth(200)
        sidebar.setMinimumWidth(150)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)

        self.nav_list = QListWidget()
        self.nav_list.setSpacing(2)

        # Add navigation items
        sections = [
            ("Inspector", "Inspect file metadata"),
            ("Converter", "Convert between formats"),
            ("Merge", "Merge shard files"),
            ("Logs", "View application logs"),
            ("Settings", "Application settings"),
        ]

        for name, tooltip in sections:
            item = QListWidgetItem(name)
            item.setToolTip(tooltip)
            self.nav_list.addItem(item)

        self.nav_list.setCurrentRow(0)
        layout.addWidget(self.nav_list)

        return sidebar

    def _create_views(self) -> None:
        """Create the view widgets."""
        self.inspector_view = InspectorView()
        self.view_stack.addWidget(self.inspector_view)

        self.converter_view = ConverterView()
        self.view_stack.addWidget(self.converter_view)

        self.merge_view = MergeView()
        self.view_stack.addWidget(self.merge_view)

        self.logs_view = LogsView()
        self.view_stack.addWidget(self.logs_view)

        self.settings_view = SettingsView()
        self.view_stack.addWidget(self.settings_view)

    def _create_bottom_dock(self) -> QWidget:
        """Create the bottom dock with logs panel and progress bar."""
        dock = QWidget()
        dock.setMaximumHeight(150)
        layout = QVBoxLayout(dock)
        layout.setContentsMargins(8, 4, 8, 4)

        # Progress bar
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(300)
        progress_layout.addWidget(self.progress_bar)

        progress_layout.addStretch()
        layout.addLayout(progress_layout)

        # Quick log output
        self.quick_log = QTextEdit()
        self.quick_log.setReadOnly(True)
        self.quick_log.setMaximumHeight(80)
        self.quick_log.setFontFamily("monospace")
        self.quick_log.setPlaceholderText("Application logs will appear here...")
        layout.addWidget(self.quick_log)

        return dock

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

    @Slot(int)
    def _on_nav_changed(self, index: int) -> None:
        """Handle navigation selection change."""
        self.view_stack.setCurrentIndex(index)

    def _on_open_file(self) -> None:
        """Handle open file action from toolbar."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Vector Dataset File",
            "",
            "All Supported Files (*.npy *.h5 *.hdf5 *.fbin *.ibin);;NPY Files (*.npy);;HDF5 Files (*.h5 *.hdf5);;FBIN Files (*.fbin);;IBIN Files (*.ibin)"
        )
        if file_path:
            # Switch to inspector view and scan file
            self.nav_list.setCurrentRow(0)
            self.inspector_view._current_file = file_path
            self.inspector_view.file_label.setText(file_path)
            self.inspector_view.scan_btn.setEnabled(True)
            self.scan_file(file_path)

    def _on_refresh(self) -> None:
        """Handle refresh action."""
        if hasattr(self.inspector_view, "_current_file") and self.inspector_view._current_file:
            self.scan_file(self.inspector_view._current_file)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message to both the quick log and the logs view.
        
        Args:
            message: The log message.
            level: Log level (INFO, WARNING, ERROR).
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        # Append to quick log
        self.quick_log.append(formatted)
        
        # Also append to full logs view
        self.logs_view.append_log(message, level)

    def scan_file(self, file_path: str) -> None:
        """Scan a file and display its metadata.
        
        Args:
            file_path: Path to the file to scan.
        """
        self.log(f"Scanning file: {file_path}")
        self.progress_label.setText("Scanning...")
        self.progress_bar.setRange(0, 0)  # Indeterminate

        def do_scan(progress_callback=None) -> dict:
            path = Path(file_path)
            suffix = path.suffix.lower()

            if suffix == ".npy":
                reader = NPYReader(file_path)
                metadata = reader.get_metadata()
                return {"reader": reader, "metadata": metadata}
            elif suffix in (".h5", ".hdf5"):
                reader = HDF5Reader(file_path)
                metadata = reader.get_metadata()
                return {"reader": reader, "metadata": metadata}
            elif suffix == ".fbin":
                reader = FBINReader(file_path)
                metadata = reader.get_metadata()
                return {"reader": reader, "metadata": metadata}
            elif suffix == ".ibin":
                reader = IBINReader(file_path)
                metadata = reader.get_metadata()
                return {"reader": reader, "metadata": metadata}
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        self._current_worker = self._worker_manager.run_task(
            do_scan,
            on_result=self._on_scan_complete,
            on_error=self._on_scan_error,
        )

    def _on_scan_complete(self, result: dict) -> None:
        """Handle scan completion."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Ready")
        
        self._current_reader = result["reader"]
        metadata = result["metadata"]
        
        self.inspector_view.display_metadata(metadata)
        self.inspector_view.set_reader(self._current_reader)
        self.log(f"File scanned successfully: {metadata.get('vector_count', 'N/A')} vectors")
        self.status_bar.showMessage("File scanned successfully")

    def _on_scan_error(self, error: Exception, traceback_str: str) -> None:
        """Handle scan error."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        
        self.log(f"Error scanning file: {error}", "ERROR")
        self.status_bar.showMessage(f"Error: {error}")
        QMessageBox.critical(self, "Scan Error", f"Failed to scan file:\n{error}")

    def sample_file(self, file_path: str, dataset_path: str | None = None) -> None:
        """Sample vectors from a file.
        
        Args:
            file_path: Path to the file.
            dataset_path: For HDF5 files, the path to the dataset.
        """
        self.log("Sampling vectors...")
        
        def do_sample(progress_callback=None):
            path = Path(file_path)
            suffix = path.suffix.lower()

            if suffix == ".npy":
                reader = NPYReader(file_path)
                return reader.sample(0, 10)
            elif suffix in (".h5", ".hdf5"):
                reader = HDF5Reader(file_path)
                if dataset_path is None:
                    # Use first dataset
                    contents = reader.list_contents()
                    if contents["datasets"]:
                        ds_path = contents["datasets"][0]
                    else:
                        raise ValueError("No datasets found in HDF5 file")
                else:
                    ds_path = dataset_path
                return reader.sample(ds_path, 0, 10)
            elif suffix == ".fbin":
                reader = FBINReader(file_path)
                return reader.sample(0, 10)
            elif suffix == ".ibin":
                reader = IBINReader(file_path)
                return reader.sample(0, 10)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        self._current_worker = self._worker_manager.run_task(
            do_sample,
            on_result=self._on_sample_complete,
            on_error=self._on_sample_error,
        )

    def _on_sample_complete(self, result) -> None:
        """Handle sample completion."""
        self.inspector_view.display_sample(result)
        self.log(f"Sampled {len(result)} vectors")

    def _on_sample_error(self, error: Exception, traceback_str: str) -> None:
        """Handle sample error."""
        self.log(f"Error sampling: {error}", "ERROR")
        QMessageBox.warning(self, "Sample Error", f"Failed to sample vectors:\n{error}")

    def convert_file(
        self, input_path: str, output_path: str, options: dict
    ) -> None:
        """Convert a file to another format.
        
        Args:
            input_path: Path to the input file.
            output_path: Path for the output file.
            options: Conversion options.
        """
        self.log(f"Starting conversion: {input_path} -> {output_path}")
        self.progress_label.setText("Converting...")
        self.progress_bar.setValue(0)

        input_suffix = Path(input_path).suffix.lower()
        output_suffix = Path(output_path).suffix.lower()

        def do_convert(progress_callback=None):
            converter = Converter(
                chunk_size=options.get("chunk_size", 10000),
                progress_callback=progress_callback,
            )

            if input_suffix == ".npy" and output_suffix in (".h5", ".hdf5"):
                return converter.npy_to_hdf5(
                    input_path,
                    output_path,
                    dataset_name=options.get("dataset_name", "vectors"),
                    compression=options.get("compression"),
                )
            elif input_suffix in (".h5", ".hdf5") and output_suffix == ".npy":
                return converter.hdf5_to_npy(
                    input_path,
                    output_path,
                    dataset_path=options.get("dataset_path", "vectors"),
                )
            else:
                raise ValueError(
                    f"Unsupported conversion: {input_suffix} -> {output_suffix}"
                )

        self._current_worker = self._worker_manager.run_task(
            do_convert,
            on_progress=self._on_convert_progress,
            on_result=self._on_convert_complete,
            on_error=self._on_convert_error,
        )

    def _on_convert_progress(self, current: int, total: int) -> None:
        """Handle conversion progress update."""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Converting... {percent}%")
            self.converter_view.update_progress(current, total)

    def _on_convert_complete(self, result: dict) -> None:
        """Handle conversion completion."""
        self.progress_bar.setValue(100)
        self.progress_label.setText("Ready")
        
        self.converter_view.conversion_complete(result)
        self.log(f"Conversion complete: {result.get('vectors_converted', 0)} vectors")
        self.status_bar.showMessage("Conversion complete")

    def _on_convert_error(self, error: Exception, traceback_str: str) -> None:
        """Handle conversion error."""
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        
        self.converter_view.conversion_error(str(error))
        self.log(f"Conversion error: {error}", "ERROR")
        self.status_bar.showMessage(f"Error: {error}")

    def cancel_operation(self) -> None:
        """Cancel the current operation."""
        if self._current_worker:
            self._current_worker.cancel()
            self.log("Operation cancelled", "WARNING")

    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Cancel any running workers
        self._worker_manager.cancel_all()
        
        # Close any open readers
        if self._current_reader and hasattr(self._current_reader, "close"):
            self._current_reader.close()
        
        event.accept()
