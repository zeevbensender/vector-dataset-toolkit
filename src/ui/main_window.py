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
    FBINConverter,
    FBINReader,
    HDF5Reader,
    HDF5Unwrapper,
    HDF5Wrapper,
    IBINReader,
    NPYReader,
    ShardMerger,
)
from ..utils.validator import FileValidator
from ..views.converter_view import ConverterView
from ..views.inspector_view import InspectorView
from ..views.logs_view import LogsView
from ..views.merge_view import MergeView
from ..views.unwrap_view import UnwrapView
from ..views.wrap_view import WrapView
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
        self._current_metadata: dict | None = None
        self._current_file_path: str | None = None
        self._cancel_callback: callable | None = None
        self._post_scan_action: str | None = None

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
            ("Wrap", "Wrap FBIN/IBIN into HDF5"),
            ("Unwrap", "Extract datasets from HDF5"),
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

        self.wrap_view = WrapView()
        self.view_stack.addWidget(self.wrap_view)

        self.unwrap_view = UnwrapView()
        self.view_stack.addWidget(self.unwrap_view)

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
            self.inspector_view.validate_btn.setEnabled(True)
            self._current_metadata = None
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
        self._current_file_path = file_path

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
        self._current_metadata = metadata

        self.inspector_view.display_metadata(metadata)
        self.inspector_view.set_reader(self._current_reader)
        self.log(f"File scanned successfully: {metadata.get('vector_count', 'N/A')} vectors")
        self.status_bar.showMessage("File scanned successfully")

        if self._post_scan_action == "validate" and self._current_file_path == self.inspector_view._current_file:
            self._post_scan_action = None
            self._start_validation(self._current_file_path or "")

    def _on_scan_error(self, error: Exception, traceback_str: str) -> None:
        """Handle scan error."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self._current_metadata = None
        self._current_reader = None
        self._post_scan_action = None

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

    def validate_file(self, file_path: str) -> None:
        """Run validation for the provided file, auto-scanning metadata if needed."""
        if not file_path:
            return

        if (
            self._current_metadata is None
            or self._current_file_path != file_path
            or self._current_reader is None
        ):
            self._post_scan_action = "validate"
            self.scan_file(file_path)
            return

        self._start_validation(file_path)

    def _start_validation(self, file_path: str) -> None:
        """Internal helper to launch validation worker."""
        self.log(f"Validating file: {file_path}")
        self.progress_label.setText("Validating...")
        self.progress_bar.setValue(0)
        self._post_scan_action = None
        self.inspector_view.set_validation_running(True)

        def do_validate(progress_callback=None):
            validator = FileValidator(progress_callback=progress_callback)
            report = validator.validate(file_path, metadata=self._current_metadata)
            entries = [
                {**entry.to_dict(), "severity": entry.severity}
                for entry in report.entries
            ]
            log_level = {
                "fatal": "ERROR",
                "error": "ERROR",
                "warning": "WARNING",
                "info": "INFO",
                "ok": "INFO",
            }
            entry_logs = [
                (
                    f"[{item['severity'].upper()}] {item['check']}: {item['result']} - {item['details']}",
                    log_level.get(item["severity"], "INFO"),
                )
                for item in entries
            ]
            return {
                "report": report.to_dict(),
                "entries": entries,
                "logs": report.logs + entry_logs,
            }

        self._current_worker = self._worker_manager.run_task(
            do_validate,
            on_progress=self._on_validation_progress,
            on_result=self._on_validation_complete,
            on_error=self._on_validation_error,
        )

    def _on_validation_progress(self, current: int, total: int) -> None:
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Validating... {percent}%")

    def _on_validation_complete(self, result: dict) -> None:
        self.progress_bar.setValue(100)
        self.progress_label.setText("Ready")
        self.inspector_view.set_validation_running(False)

        entries = result.get("entries", [])
        self.inspector_view.display_validation_results(entries)

        report = result.get("report", {})
        fatal_entries = report.get("fatal", [])
        error_entries = report.get("errors", [])
        warning_entries = report.get("warnings", [])

        summary = (
            "Validation complete: "
            f"{len(report.get('passed', []))} ok, "
            f"{len(warning_entries)} warnings, "
            f"{len(error_entries)} errors, "
            f"{len(fatal_entries)} fatal"
        )
        level = "ERROR" if fatal_entries or error_entries else (
            "WARNING" if warning_entries else "INFO"
        )
        self.log(summary, level)
        for message, msg_level in result.get("logs", []):
            self.logs_view.append_log(message, msg_level)

        if fatal_entries:
            details = fatal_entries[0].get("details", "Fatal validation issues detected")
            QMessageBox.critical(self, "Validation failed", details)
            self.status_bar.showMessage("Validation failed")
        else:
            self.status_bar.showMessage("Validation complete")

    def _on_validation_error(self, error: Exception, traceback_str: str) -> None:
        self.inspector_view.set_validation_running(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self._post_scan_action = None

        self.log(f"Validation error: {error}", "ERROR")
        self.logs_view.append_log(traceback_str, "ERROR")
        self.status_bar.showMessage(f"Validation error: {error}")
        QMessageBox.critical(self, "Validation Error", str(error))

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
            elif input_suffix == ".fbin" and output_suffix == ".npy":
                fbin_converter = FBINConverter(
                    chunk_size=options.get("chunk_size", 10000),
                    progress_callback=progress_callback,
                )
                return fbin_converter.fbin_to_npy(input_path, output_path)
            elif input_suffix == ".npy" and output_suffix == ".fbin":
                fbin_converter = FBINConverter(
                    chunk_size=options.get("chunk_size", 10000),
                    progress_callback=progress_callback,
                )
                return fbin_converter.npy_to_fbin(input_path, output_path)
            elif input_suffix == ".fbin" and output_suffix in (".h5", ".hdf5"):
                fbin_converter = FBINConverter(
                    chunk_size=options.get("chunk_size", 10000),
                    progress_callback=progress_callback,
                )
                return fbin_converter.fbin_to_hdf5(
                    input_path, output_path,
                    dataset_name=options.get("dataset_name", "vectors"),
                    compression=options.get("compression"),
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
        if self._cancel_callback:
            try:
                self._cancel_callback()
            finally:
                self._cancel_callback = None

    # Shard validation and merge methods
    def validate_shards(self, shard_paths: list[str]) -> None:
        """Validate shard files for compatibility.
        
        Args:
            shard_paths: List of paths to shard files.
        """
        self.log(f"Validating {len(shard_paths)} shards...")

        def do_validate(progress_callback=None):
            merger = ShardMerger()
            infos = merger.validate_shards(shard_paths)
            return [info.to_dict() for info in infos]

        self._current_worker = self._worker_manager.run_task(
            do_validate,
            on_result=self._on_validate_complete,
            on_error=self._on_validate_error,
        )

    def _on_validate_complete(self, result: list[dict]) -> None:
        """Handle shard validation completion."""
        self.merge_view.display_shard_validation(result)
        compatible = sum(1 for r in result if r.get("validation_result") == "compatible")
        self.log(f"Validated {len(result)} shards: {compatible} compatible")

    def _on_validate_error(self, error: Exception, traceback_str: str) -> None:
        """Handle validation error."""
        self.log(f"Validation error: {error}", "ERROR")

    def preview_merge(self, shard_paths: list[str], output_format: str) -> None:
        """Preview merge operation (dry run).
        
        Args:
            shard_paths: List of paths to shard files.
            output_format: Output format ("fbin" or "npy").
        """
        self.log("Generating merge preview...")

        def do_preview(progress_callback=None):
            merger = ShardMerger()
            preview = merger.preview_merge(shard_paths, output_format)
            return preview.to_dict()

        self._current_worker = self._worker_manager.run_task(
            do_preview,
            on_result=self._on_preview_complete,
            on_error=self._on_preview_error,
        )

    def _on_preview_complete(self, result: dict) -> None:
        """Handle preview completion."""
        self.merge_view.display_preview(result)
        self.log(f"Preview: {result.get('total_vectors', 0):,} vectors from {len(result.get('shards', []))} shards")

    def _on_preview_error(self, error: Exception, traceback_str: str) -> None:
        """Handle preview error."""
        self.log(f"Preview error: {error}", "ERROR")

    def merge_shards(
        self, shard_paths: list[str], output_path: str, options: dict
    ) -> None:
        """Merge shard files into a single file.
        
        Args:
            shard_paths: List of paths to shard files.
            output_path: Path for the output file.
            options: Merge options (output_format, chunk_size, compute_checksum).
        """
        self.log(f"Merging {len(shard_paths)} shards -> {output_path}")
        self.progress_label.setText("Merging...")
        self.progress_bar.setValue(0)

        def do_merge(progress_callback=None):
            merger = ShardMerger(
                chunk_size=options.get("chunk_size", 10000),
                progress_callback=progress_callback,
                log_callback=lambda msg: None,  # Silent logging
            )
            return merger.merge(
                shard_paths,
                output_path,
                output_format=options.get("output_format", "fbin"),
                compute_checksum=options.get("compute_checksum", False),
            )

        self._current_worker = self._worker_manager.run_task(
            do_merge,
            on_progress=self._on_merge_progress,
            on_result=self._on_merge_complete,
            on_error=self._on_merge_error,
        )

    def _on_merge_progress(self, current: int, total: int) -> None:
        """Handle merge progress update."""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Merging... {percent}%")
            self.merge_view.update_progress(current, total)

    def _on_merge_complete(self, result: dict) -> None:
        """Handle merge completion."""
        self.progress_bar.setValue(100)
        self.progress_label.setText("Ready")
        
        self.merge_view.merge_complete(result)
        self.log(f"Merge complete: {result.get('total_vectors', 0):,} vectors")
        self.status_bar.showMessage("Merge complete")

    def _on_merge_error(self, error: Exception, traceback_str: str) -> None:
        """Handle merge error."""
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")

        self.merge_view.merge_error(str(error))
        self.log(f"Merge error: {error}", "ERROR")
        self.status_bar.showMessage(f"Error: {error}")

    # Unwrap HDF5 into FBIN/IBIN outputs
    def scan_hdf5_for_unwrap(self, file_path: str, max_vectors: int | None) -> None:
        """Scan an HDF5 file for unwrap metadata."""

        self.log(f"Scanning HDF5 for unwrap: {file_path}")
        self.progress_label.setText("Scanning HDF5…")
        self.progress_bar.setRange(0, 0)
        self.unwrap_view.toggle_busy(True)

        log_cb = lambda msg, level="INFO": (
            self.log(msg, level),
            self.unwrap_view.append_log(msg, level),
        )
        unwrapper = HDF5Unwrapper(
            max_vectors=max_vectors or None,
            log_callback=log_cb,
        )

        def do_scan(progress_callback=None):
            return unwrapper.scan(file_path)

        self._current_worker = self._worker_manager.run_task(
            do_scan,
            on_result=self._on_unwrap_scan_complete,
            on_error=lambda e, tb: self._on_unwrap_error(e, tb, during_scan=True),
        )

    def extract_hdf5_datasets(
        self, file_path: str, output_dir: str | None, max_vectors: int | None
    ) -> None:
        """Extract HDF5 datasets into base/queries/gt outputs."""

        self.log(
            f"Extracting datasets from {file_path} -> {output_dir or Path(file_path).parent}"
        )
        self.progress_label.setText("Extracting…")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.unwrap_view.toggle_busy(True)

        log_cb = lambda msg, level="INFO": (
            self.log(msg, level),
            self.unwrap_view.append_log(msg, level),
        )
        unwrapper = HDF5Unwrapper(
            max_vectors=max_vectors or None,
            progress_callback=None,
            log_callback=log_cb,
        )
        self._cancel_callback = unwrapper.cancel

        def do_extract(progress_callback=None):
            unwrapper.progress_callback = progress_callback
            return unwrapper.extract(file_path, output_dir)

        self._current_worker = self._worker_manager.run_task(
            do_extract,
            on_progress=self._on_unwrap_progress,
            on_result=self._on_unwrap_complete,
            on_error=self._on_unwrap_error,
        )

    def _on_unwrap_scan_complete(self, metadata: dict) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Ready")
        self.unwrap_view.toggle_busy(False)

        self.unwrap_view.display_metadata(metadata)
        self.log("HDF5 scan complete for unwrap")
        self.status_bar.showMessage("HDF5 scan complete")

    def _on_unwrap_progress(self, current: int, total: int) -> None:
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Extracting… {percent}%")

    def _on_unwrap_complete(self, result: dict) -> None:
        self.progress_bar.setValue(100)
        self.progress_label.setText("Ready")
        self.unwrap_view.toggle_busy(False)
        self._cancel_callback = None

        self.unwrap_view.display_summary(result)
        base_shape = result.get("vectors", {}).get("base")
        query_shape = result.get("vectors", {}).get("queries")
        self.unwrap_view.append_log("Extraction completed successfully.")
        self.log("Extraction completed successfully.")
        if base_shape or query_shape:
            self.log(
                f"Extracted base={base_shape} queries={query_shape}",
            )
        self.status_bar.showMessage("Extraction completed successfully")

    def _on_unwrap_error(
        self, error: Exception, traceback_str: str, during_scan: bool = False
    ) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self.unwrap_view.toggle_busy(False)
        self._cancel_callback = None

        self.unwrap_view.append_log(traceback_str, "ERROR")
        self.log(f"Unwrap error: {error}", "ERROR")
        self.status_bar.showMessage(f"Error: {error}")
        if during_scan:
            QMessageBox.critical(self, "Scan Error", f"Failed to scan HDF5 file:\n{error}")
        else:
            self.unwrap_view.extraction_error(str(error))

    # Wrap FBIN/IBIN into HDF5
    def validate_wrap_inputs(
        self, base_paths: list[str], query_path: str, ibin_path: str | None
    ) -> None:
        """Validate wrap inputs before running the wrap job."""

        self.log(
            f"Validating base={len(base_paths)} FBIN file(s) and queries file for wrapping…"
        )
        self._cancel_callback = None

        def do_validate(progress_callback=None):
            wrapper = HDF5Wrapper()
            return wrapper.validate_inputs(base_paths, query_path, ibin_path)

        self._current_worker = self._worker_manager.run_task(
            do_validate,
            on_result=self._on_wrap_validation_complete,
            on_error=self._on_wrap_error,
        )

    def wrap_into_hdf5(
        self, base_paths: list[str], query_path: str, output_path: str, options: dict
    ) -> None:
        """Execute the wrapping job in a background worker."""

        self.log(
            "Wrapping base + query FBIN files to "
            f"{output_path} (base={len(base_paths)}, queries={Path(query_path).name})"
        )
        self.progress_label.setText("Wrapping…")
        self.progress_bar.setValue(0)

        wrapper = HDF5Wrapper(
            chunk_size=options.get("chunk_size", 10000),
            progress_callback=None,
        )
        self._cancel_callback = wrapper.cancel

        def do_wrap(progress_callback=None):
            wrapper.progress_callback = progress_callback
            return wrapper.wrap_into_hdf5(
                base_paths,
                query_path,
                output_path,
                base_dataset=options.get("base_dataset", "base"),
                train_dataset=options.get("train_dataset", "train"),
                query_dataset=options.get("query_dataset", "test"),
                neighbor_dataset=options.get("neighbor_dataset", "neighbors"),
                include_train_alias=options.get("include_train_alias", True),
                compression=options.get("compression"),
                ibin_path=options.get("ibin_path"),
            )

        self._current_worker = self._worker_manager.run_task(
            do_wrap,
            on_progress=self._on_wrap_progress,
            on_result=self._on_wrap_complete,
            on_error=self._on_wrap_error,
        )

    def _on_wrap_validation_complete(self, result: dict) -> None:
        self.wrap_view.display_validation(result)
        if result.get("valid"):
            self.log("Wrap inputs validated")
        else:
            self.log("Wrap validation failed", "ERROR")

    def _on_wrap_progress(self, current: int, total: int) -> None:
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Wrapping… {percent}%")
        self.wrap_view.update_progress(current, total)

    def _on_wrap_complete(self, result: dict) -> None:
        self.progress_bar.setValue(100)
        self.progress_label.setText("Ready")
        self._cancel_callback = None
        self.wrap_view.wrap_complete(result)
        self.log(
            "Wrapped vectors to "
            f"{result.get('output_path')} (base={result.get('base_dataset')}, "
            f"queries={result.get('query_dataset')})"
        )
        self.status_bar.showMessage("Wrap complete")

    def _on_wrap_error(self, error: Exception, traceback_str: str) -> None:
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error")
        self._cancel_callback = None
        self.wrap_view.wrap_error(str(error))
        self.log(f"Wrap error: {error}", "ERROR")
        self.status_bar.showMessage(f"Error: {error}")

    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Cancel any running workers
        self._worker_manager.cancel_all()
        
        # Close any open readers
        if self._current_reader and hasattr(self._current_reader, "close"):
            self._current_reader.close()
        
        event.accept()
