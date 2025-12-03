"""Vector Dataset Toolkit - Main application entry point.

This module provides the main entry point for the PySide6 desktop application.

Usage:
    python -m src.app
"""

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def _load_window_icon() -> QIcon:
    """Load the application icon with graceful fallbacks.

    The Qt resource module (``resources_rc``) is optional to avoid tracking
    generated binary assets in the repository. When it is not available, the
    icon is resolved from the on-disk SVG instead.
    """

    try:  # Prefer resources compiled by ``pyside6-rcc`` when present.
        from resources import resources_rc  # type: ignore  # noqa: F401
    except Exception:
        resources_rc = None

    if resources_rc:
        for candidate in (":/icons/vector_dataset_tool.png", ":/icons/vector_dataset_tool.svg"):
            icon = QIcon(candidate)
            if not icon.isNull():
                return icon

    icons_dir = Path(__file__).resolve().parent.parent / "resources" / "icons"
    for filename in ("vector_dataset_tool.png", "vector_dataset_tool.svg"):
        file_path = icons_dir / filename
        if file_path.exists():
            icon = QIcon(str(file_path))
            if not icon.isNull():
                return icon

    return QIcon()


def main() -> int:
    """Main entry point for the application.
    
    Returns:
        Exit code (0 for success).
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Vector Dataset Tool")
    app.setOrganizationName("VectorDatasetToolkit")
    app.setWindowIcon(_load_window_icon())
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
