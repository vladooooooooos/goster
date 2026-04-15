"""Application entry point for Indexator."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.utils.config import load_config


def resolve_app_icon(app_root: Path) -> QIcon | None:
    """Return the preferred application icon when it is available."""
    icon_path = app_root / "logos" / "logo 256x256.ico"
    if not icon_path.is_file():
        return None
    return QIcon(str(icon_path))


def main() -> int:
    """Start the Indexator desktop application."""
    app_root = Path(__file__).resolve().parents[1]
    config = load_config(app_root / "config.json")

    app = QApplication(sys.argv)
    app.setApplicationName(config.app.name)
    app_icon = resolve_app_icon(app_root)
    if app_icon is not None:
        app.setWindowIcon(app_icon)

    window = MainWindow(config=config)
    if app_icon is not None:
        window.setWindowIcon(app_icon)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
