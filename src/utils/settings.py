"""Application settings management using Qt QSettings.

This module centralizes persistent UI state such as last-used directories,
window geometry, and splitter sizes to provide a consistent user experience
across sessions.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QMainWindow, QSplitter


class SettingsManager:
    """Helper for persisting and restoring application UI state."""

    def __init__(self) -> None:
        self._settings = QSettings()

    def get_last_directory(self) -> str:
        """Return the last directory used by file dialogs or the user's home.

        Returns:
            String path to the directory to use as a starting location.
        """

        value = self._settings.value("file_dialogs/last_directory")
        if isinstance(value, str) and value:
            directory = Path(value)
            if directory.is_dir():
                return str(directory)

        return str(Path.home())

    def update_last_directory(self, path: str | Path) -> None:
        """Persist the directory component of the provided path."""

        directory = Path(path)
        if directory.is_file():
            directory = directory.parent
        if directory.exists():
            self._settings.setValue("file_dialogs/last_directory", str(directory))
            self._settings.sync()

    def restore_window(self, window: QMainWindow) -> None:
        """Restore geometry/state for the provided main window."""

        geometry = self._settings.value("ui/window_geometry")
        if isinstance(geometry, (QByteArray, bytes, bytearray)):
            window.restoreGeometry(QByteArray(geometry))

        state = self._settings.value("ui/window_state")
        if isinstance(state, (QByteArray, bytes, bytearray)):
            window.restoreState(QByteArray(state))

    def save_window(self, window: QMainWindow) -> None:
        """Save geometry/state for the provided main window."""

        self._settings.setValue("ui/window_geometry", window.saveGeometry())
        self._settings.setValue("ui/window_state", window.saveState())

    def restore_splitter(self, splitter: QSplitter, key: str) -> None:
        """Restore splitter sizes from settings if available."""

        sizes = self._settings.value(key)
        parsed_sizes: list[int] = []

        if isinstance(sizes, list) and sizes:
            parsed_sizes = [int(size) for size in sizes if int(size) > 0]
        elif isinstance(sizes, str) and sizes:
            try:
                parsed_sizes = [int(size.strip()) for size in sizes.split(",")]
            except ValueError:
                parsed_sizes = []

        if parsed_sizes:
            splitter.setSizes(parsed_sizes)

    def save_splitter(self, splitter: QSplitter, key: str) -> None:
        """Persist splitter sizes to settings."""

        self._settings.setValue(key, splitter.sizes())

    def reset_all(self) -> None:
        """Clear all stored settings."""

        self._settings.clear()
        self._settings.sync()

    def sync(self) -> None:
        """Flush pending settings changes."""

        self._settings.sync()

