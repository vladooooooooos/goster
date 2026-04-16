from __future__ import annotations

import atexit
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


HOST = "127.0.0.1"
PORT = 8000
STARTUP_TIMEOUT_SECONDS = 60
URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{URL}/health"
SERVER_COMMAND = [
    sys.executable,
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    HOST,
    "--port",
    str(PORT),
]

PROJECT_DIR = Path(__file__).resolve().parent
USER_LOG_FILE = PROJECT_DIR / "goster_chat_user.log"


@dataclass(frozen=True)
class LauncherConfig:
    debug: bool
    reload: bool = False
    open_browser: bool = True
    log_file: Path | None = None


def run_launcher(config: LauncherConfig) -> int:
    log_handle: TextIO | None = None
    process: subprocess.Popen | None = None

    try:
        if _is_http_ready():
            _log_or_file(
                config,
                log_handle,
                f"Backend is already reachable at {HEALTH_URL}.",
            )
            if config.open_browser:
                _open_browser(config, log_handle)
            return 0

        command = [*SERVER_COMMAND]
        if config.reload:
            command.append("--reload")

        if config.debug:
            _log("Starting GOSTer Chat backend.")
            _log(f"Working directory: {PROJECT_DIR}")
            _log(f"Command: {_format_command(command)}")
            stdout = None
            stderr = None
        else:
            log_path = config.log_file or USER_LOG_FILE
            log_handle = log_path.open("a", encoding="utf-8")
            _write_log(log_handle, "Starting GOSTer Chat backend.")
            _write_log(log_handle, f"Working directory: {PROJECT_DIR}")
            _write_log(log_handle, f"Command: {_format_command(command)}")
            stdout = log_handle
            stderr = subprocess.STDOUT

        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        atexit.register(_terminate_process, process, log_handle)

        if not _wait_until_ready(process, config, log_handle):
            _terminate_process(process, log_handle)
            return 1

        if config.open_browser:
            _open_browser(config, log_handle)

        _log_or_file(config, log_handle, "Backend is running. Press Ctrl+C to stop.")
        return process.wait()
    except KeyboardInterrupt:
        _log_or_file(config, log_handle, "Stopping GOSTer Chat backend.")
        if process is not None:
            _terminate_process(process, log_handle)
        return 130
    except Exception as exc:
        _log_or_file(config, log_handle, f"Launcher failed: {exc}")
        if process is not None:
            _terminate_process(process, log_handle)
        return 1
    finally:
        if log_handle is not None:
            log_handle.close()


def _wait_until_ready(
    process: subprocess.Popen,
    config: LauncherConfig,
    log_handle: TextIO | None,
) -> bool:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            _log_or_file(
                config,
                log_handle,
                f"Backend exited before becoming ready. Exit code: {exit_code}",
            )
            return False

        if _is_http_ready():
            _log_or_file(config, log_handle, f"Backend is reachable at {HEALTH_URL}.")
            return True

        time.sleep(0.5)

    _log_or_file(
        config,
        log_handle,
        f"Backend did not become reachable within {STARTUP_TIMEOUT_SECONDS} seconds.",
    )
    return False


def _is_http_ready() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def _open_browser(config: LauncherConfig, log_handle: TextIO | None) -> None:
    _log_or_file(config, log_handle, f"Opening {URL} in the default browser.")
    webbrowser.open(URL, new=2)


def _terminate_process(process: subprocess.Popen, log_handle: TextIO | None) -> None:
    if process.poll() is not None:
        return

    _write_log(log_handle, "Terminating backend process.")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _write_log(log_handle, "Backend did not stop cleanly; killing it.")
        process.kill()
        process.wait(timeout=5)


def _log_or_file(
    config: LauncherConfig,
    log_handle: TextIO | None,
    message: str,
) -> None:
    if config.debug:
        _log(message)
    else:
        _write_log(log_handle, message)


def _log(message: str) -> None:
    print(f"[goster-chat] {message}", flush=True)


def _write_log(log_handle: TextIO | None, message: str) -> None:
    if log_handle is None:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_handle.write(f"{timestamp} {message}\n")
    log_handle.flush()


def _format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)
