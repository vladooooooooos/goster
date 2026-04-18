"""Helpers for managing the local Qdrant server during app startup."""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_QDRANT_SERVICE = "qdrant"


@dataclass(frozen=True)
class QdrantServerConfig:
    """Local Qdrant server startup settings."""

    url: str = DEFAULT_QDRANT_URL
    service_name: str = DEFAULT_QDRANT_SERVICE
    compose_project_dir: Path = Path(__file__).resolve().parents[1]
    startup_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.5


@dataclass(frozen=True)
class QdrantServerEnsureResult:
    """Result of ensuring the local Qdrant server is reachable."""

    url: str
    started: bool
    started_by_launcher: bool


RunCommand = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
ReadyCheck = Callable[[str], bool]
Sleep = Callable[[float], None]
Monotonic = Callable[[], float]


def ensure_qdrant_server(
    config: QdrantServerConfig | None = None,
    *,
    is_ready: ReadyCheck = None,
    run_command: RunCommand = None,
    sleep: Sleep = time.sleep,
    monotonic: Monotonic = time.monotonic,
) -> QdrantServerEnsureResult:
    """Start Qdrant with Docker Compose when it is not already reachable."""
    resolved_config = config or QdrantServerConfig()
    ready = is_ready or is_qdrant_ready
    runner = run_command or run_docker_compose

    if ready(resolved_config.url):
        return QdrantServerEnsureResult(
            url=resolved_config.url,
            started=True,
            started_by_launcher=False,
        )

    command = ["docker", "compose", "up", "-d", resolved_config.service_name]
    result = runner(command, resolved_config.compose_project_dir)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Docker Compose exited with a non-zero status.").strip()
        raise RuntimeError(
            "Could not start Qdrant server. Start Docker Desktop, then run "
            f"`docker compose up -d {resolved_config.service_name}`. Details: {details}"
        )

    deadline = monotonic() + resolved_config.startup_timeout_seconds
    while monotonic() < deadline:
        if ready(resolved_config.url):
            return QdrantServerEnsureResult(
                url=resolved_config.url,
                started=True,
                started_by_launcher=True,
            )
        sleep(resolved_config.poll_interval_seconds)

    raise RuntimeError(
        f"Qdrant server did not become reachable at {resolved_config.url} "
        f"within {resolved_config.startup_timeout_seconds:.0f} seconds."
    )


def stop_qdrant_server(
    config: QdrantServerConfig | None = None,
    *,
    run_command: RunCommand = None,
) -> subprocess.CompletedProcess[str]:
    """Stop the local Qdrant Docker Compose service without deleting its volume."""
    resolved_config = config or QdrantServerConfig()
    runner = run_command or run_docker_compose
    command = ["docker", "compose", "stop", resolved_config.service_name]
    result = runner(command, resolved_config.compose_project_dir)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Docker Compose exited with a non-zero status.").strip()
        raise RuntimeError(f"Could not stop Qdrant server. Details: {details}")
    return result


def is_qdrant_ready(url: str) -> bool:
    """Return True when the Qdrant HTTP endpoint responds."""
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def run_docker_compose(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a Docker Compose command in the repository root."""
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
