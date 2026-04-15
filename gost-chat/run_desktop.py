from __future__ import annotations

import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8000
STARTUP_TIMEOUT_SECONDS = 60
WINDOW_TITLE = "Goster Chat"
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

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 900
READINESS_PATH = "/health"
APP_PATH = "/"

PROJECT_DIR = Path(__file__).resolve().parent
STARTUP_LOG_LINES = 80


def format_url(path: str) -> str:
    return f"http://{HOST}:{PORT}{path}"


def start_backend() -> tuple[subprocess.Popen[str], deque[str]]:
    process = subprocess.Popen(
        SERVER_COMMAND,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines: deque[str] = deque(maxlen=STARTUP_LOG_LINES)

    def collect_output() -> None:
        if process.stdout is None:
            return

        for line in process.stdout:
            print(line, end="")
            output_lines.append(line.rstrip())

    threading.Thread(target=collect_output, daemon=True).start()
    return process, output_lines


def wait_for_backend(process: subprocess.Popen[str], output_lines: deque[str]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    readiness_url = format_url(READINESS_PATH)
    last_error = ""

    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                "Backend process exited before it became reachable "
                f"(exit code {exit_code}).\n{format_backend_output(output_lines)}"
            )

        try:
            with urllib.request.urlopen(readiness_url, timeout=1.5) as response:
                if 200 <= response.status < 400:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)

        time.sleep(0.5)

    stop_backend(process)
    raise TimeoutError(
        f"Timed out after {STARTUP_TIMEOUT_SECONDS} seconds waiting for {readiness_url}.\n"
        f"Last readiness error: {last_error or 'no response'}\n"
        f"{format_backend_output(output_lines)}"
    )


def format_backend_output(output_lines: deque[str]) -> str:
    if not output_lines:
        return "Backend produced no output."

    return "Recent backend output:\n" + "\n".join(output_lines)


def stop_backend(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_desktop() -> int:
    try:
        import webview
    except ImportError:
        print(
            "PyWebView is not installed. Install it with:\n"
            "python -m pip install pywebview",
            file=sys.stderr,
        )
        return 1

    process, output_lines = start_backend()

    try:
        wait_for_backend(process, output_lines)
        webview.create_window(
            WINDOW_TITLE,
            format_url(APP_PATH),
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            resizable=True,
        )
        webview.start()
        return 0
    except Exception as exc:
        print(f"Failed to launch desktop window: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_backend(process)


if __name__ == "__main__":
    raise SystemExit(run_desktop())
