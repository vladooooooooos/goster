"""Application entry point for Indexator."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.utils.config import load_config


def main() -> int:
    """Start the Indexator desktop application."""
    app_root = Path(__file__).resolve().parents[1]
    config = load_config(app_root / "config.json")

    app = QApplication(sys.argv)
    app.setApplicationName(config.app.name)

    window = MainWindow(config=config)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
