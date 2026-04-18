from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from shared.qdrant_server import QdrantServerConfig, ensure_qdrant_server, stop_qdrant_server


class QdrantServerManagerTest(unittest.TestCase):
    def test_ensure_does_not_start_compose_when_server_is_ready(self) -> None:
        commands: list[list[str]] = []

        result = ensure_qdrant_server(
            QdrantServerConfig(compose_project_dir=Path("C:/goster")),
            is_ready=lambda url: True,
            run_command=lambda command, cwd: commands.append(command) or completed(command),
            sleep=lambda seconds: None,
            monotonic=counter([0.0, 0.1]),
        )

        self.assertTrue(result.started)
        self.assertFalse(result.started_by_launcher)
        self.assertEqual(commands, [])

    def test_ensure_starts_compose_and_waits_until_ready(self) -> None:
        commands: list[list[str]] = []
        readiness = iter([False, False, True])

        result = ensure_qdrant_server(
            QdrantServerConfig(compose_project_dir=Path("C:/goster"), startup_timeout_seconds=5.0),
            is_ready=lambda url: next(readiness),
            run_command=lambda command, cwd: commands.append(command) or completed(command),
            sleep=lambda seconds: None,
            monotonic=counter([0.0, 0.1, 0.2, 0.3]),
        )

        self.assertTrue(result.started)
        self.assertTrue(result.started_by_launcher)
        self.assertEqual(commands, [["docker", "compose", "up", "-d", "qdrant"]])

    def test_ensure_reports_compose_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Could not start Qdrant server"):
            ensure_qdrant_server(
                QdrantServerConfig(compose_project_dir=Path("C:/goster")),
                is_ready=lambda url: False,
                run_command=lambda command, cwd: completed(command, returncode=1, stderr="docker failed"),
                sleep=lambda seconds: None,
                monotonic=counter([0.0, 0.1]),
            )

    def test_stop_runs_compose_stop(self) -> None:
        commands: list[list[str]] = []

        stop_qdrant_server(
            QdrantServerConfig(compose_project_dir=Path("C:/goster")),
            run_command=lambda command, cwd: commands.append(command) or completed(command),
        )

        self.assertEqual(commands, [["docker", "compose", "stop", "qdrant"]])


def completed(command: list[str], returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode=returncode, stdout="", stderr=stderr)


def counter(values: list[float]):
    iterator = iter(values)

    def monotonic() -> float:
        return next(iterator)

    return monotonic


if __name__ == "__main__":
    unittest.main()
