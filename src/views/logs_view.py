"""Logs view for viewing application logs."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class LogsView(QWidget):
    """View for displaying application logs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear Logs")
        self.clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(self.clear_btn)
        
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self._on_copy)
        toolbar.addWidget(self.copy_btn)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("monospace")
        layout.addWidget(self.log_text)

    def append_log(self, message: str, level: str = "INFO") -> None:
        """Append a log message.
        
        Args:
            message: The log message.
            level: Log level (INFO, WARNING, ERROR).
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        # Color based on level
        if level == "ERROR":
            formatted = f'<span style="color: #ff6b6b;">{formatted}</span>'
        elif level == "WARNING":
            formatted = f'<span style="color: #ffd93d;">{formatted}</span>'
        else:
            formatted = f'<span style="color: #6bcb77;">{formatted}</span>'
        
        self.log_text.append(formatted)

    def _on_clear(self) -> None:
        """Clear the log display."""
        self.log_text.clear()

    def _on_copy(self) -> None:
        """Copy logs to clipboard."""
        from PySide6.QtWidgets import QApplication
        
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_text.toPlainText())
