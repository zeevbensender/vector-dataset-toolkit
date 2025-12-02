"""Settings view placeholder for application settings."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsView(QWidget):
    """View for application settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QVBoxLayout(self)

        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout(perf_group)

        thread_row = QHBoxLayout()
        thread_row.addWidget(QLabel("Worker Threads:"))
        self.thread_count = QSpinBox()
        self.thread_count.setRange(1, 16)
        self.thread_count.setValue(4)
        thread_row.addWidget(self.thread_count)
        thread_row.addStretch()
        perf_layout.addLayout(thread_row)

        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Default Chunk Size:"))
        self.default_chunk = QSpinBox()
        self.default_chunk.setRange(1000, 100000)
        self.default_chunk.setValue(10000)
        self.default_chunk.setSingleStep(1000)
        chunk_row.addWidget(self.default_chunk)
        chunk_row.addStretch()
        perf_layout.addLayout(chunk_row)

        layout.addWidget(perf_group)

        # Appearance settings
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance_group)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:"))
        self.theme = QComboBox()
        self.theme.addItems(["Dark", "Light", "System"])
        theme_row.addWidget(self.theme)
        theme_row.addStretch()
        appearance_layout.addLayout(theme_row)

        layout.addWidget(appearance_group)

        # Logging settings
        log_group = QGroupBox("Logging")
        log_layout = QVBoxLayout(log_group)

        self.verbose_logging = QCheckBox("Verbose logging")
        log_layout.addWidget(self.verbose_logging)

        self.log_to_file = QCheckBox("Write logs to file")
        log_layout.addWidget(self.log_to_file)

        layout.addWidget(log_group)

        layout.addStretch()

        # Note
        note = QLabel(
            "Note: Settings persistence will be implemented in a future milestone."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
