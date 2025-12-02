"""Vector Dataset Toolkit - Main application entry point.

This module provides the main entry point for the PySide6 desktop application.

Usage:
    python -m src.app
"""

import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> int:
    """Main entry point for the application.
    
    Returns:
        Exit code (0 for success).
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Vector Dataset Tool")
    app.setOrganizationName("VectorDatasetToolkit")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
