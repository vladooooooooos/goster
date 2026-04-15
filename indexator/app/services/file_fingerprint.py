"""Source PDF fingerprint helpers for index state detection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SourceFileFingerprint:
    """Stable metadata and optional content hash for one source file."""

    file_size: int
    modified_at: str
    source_fingerprint: str | None = None


class FileFingerprintService:
    """Build cheap metadata fingerprints and stronger content hashes."""

    def get_metadata(self, file_path: Path) -> SourceFileFingerprint:
        """Return file size and mtime without reading the whole file."""
        stat = file_path.stat()
        return SourceFileFingerprint(
            file_size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        )

    def get_content_fingerprint(self, file_path: Path) -> SourceFileFingerprint:
        """Return file metadata plus SHA-256 of the source PDF bytes."""
        metadata = self.get_metadata(file_path)
        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return SourceFileFingerprint(
            file_size=metadata.file_size,
            modified_at=metadata.modified_at,
            source_fingerprint=digest.hexdigest(),
        )
