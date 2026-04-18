from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TextIO


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.qdrant_server import QdrantServerConfig, ensure_qdrant_server


USER_LOG_FILE = PROJECT_DIR / "indexator_launcher.log"


def run_launcher(log_file: Path | None = USER_LOG_FILE) -> int:
    log_handle: TextIO | None = None
    try:
        if log_file is not None:
            log_handle = log_file.open("a", encoding="utf-8")

        _write_log(log_handle, "Checking local Qdrant server.")
        result = ensure_qdrant_server(
            QdrantServerConfig(url=_resolve_qdrant_url(), compose_project_dir=PROJECT_ROOT)
        )
        if result.started_by_launcher:
            _write_log(log_handle, f"Started local Qdrant server at {result.url}.")
        else:
            _write_log(log_handle, f"Local Qdrant server is already reachable at {result.url}.")

        from app.main import main

        return main()
    except Exception as exc:
        _write_log(log_handle, f"Indexator launcher failed: {exc}")
        return 1
    finally:
        if log_handle is not None:
            log_handle.close()


def _write_log(log_handle: TextIO | None, message: str) -> None:
    if log_handle is None:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_handle.write(f"{timestamp} {message}\n")
    log_handle.flush()


def _resolve_qdrant_url() -> str:
    from app.utils.config import load_config

    return load_config(PROJECT_DIR / "config.json").storage.url
