#!/usr/bin/env python3
"""Main entry point for the ArUco Grasp Annotator application."""

import sys
from pathlib import Path

try:
    from PyQt6.QtWidgets import QApplication
    from .gui.main_window import MainWindow
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please ensure all dependencies are installed with: uv sync")
    sys.exit(1)


def main() -> None:
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("ArUco Grasp Annotator")
    app.setApplicationVersion("0.1.0")
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
